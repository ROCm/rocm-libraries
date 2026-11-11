// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2
.p2align 8
.type RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2,@function
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 9467) 
v_and_b32 v3, 63, v1 // (op 9467) 
v_sub_u32 v4, v3, v0 // (op 9467) 

v_lshrrev_b32 v3, 1, v4 // (op 9467) 
v_add_lshl_u32 v5, v0, v3, 6 // (op 9467) 


v_and_b32 v0, 1, v4 // (op 9467) 

v_lshlrev_b32 v3, 3, v0 // (op 9467) 

v_lshrrev_b32 v0, 7, v1 // (op 9467) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 9467) 


v_add_u32 v0, v5, v4 // (op 9467) 


v_lshrrev_b32 v3, 6, v1 // (op 9542) 
v_and_b32 v4, 1, v3 // (op 9542) 

v_and_b32 v3, 15, v1 // (op 9542) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 9542) 


v_and_b32 v3, 48, v1 // (op 9542) 
v_lshl_add_u32 v4, v5, 7, v3 // (op 9542) 


v_lshrrev_b32 v3, 7, v1 // (op 9618) 
v_and_b32 v5, 15, v1 // (op 9618) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 9618) 


v_and_b32 v3, 63, v1 // (op 9618) 
v_lshrrev_b32 v5, 4, v3 // (op 9618) 

v_lshl_add_u32 v3, v5, 6, v6 // (op 9618) 


v_lshrrev_b32 v5, 6, v1 // (op 9694) 
v_and_b32 v6, 1, v5 // (op 9694) 

v_and_b32 v5, 15, v1 // (op 9694) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 9694) 


v_and_b32 v5, 63, v1 // (op 9694) 
v_lshrrev_b32 v6, 4, v5 // (op 9694) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 9694) 


v_and_b32 v6, 48, v1 // (op 9821) 
v_and_b32 v7, 63, v1 // (op 9821) 
v_sub_u32 v8, v7, v6 // (op 9821) 

v_lshrrev_b32 v7, 1, v8 // (op 9821) 
v_add_lshl_u32 v9, v6, v7, 6 // (op 9821) 


v_and_b32 v6, 1, v8 // (op 9821) 

v_lshlrev_b32 v7, 3, v6 // (op 9821) 

v_lshrrev_b32 v6, 7, v1 // (op 9821) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 9821) 


v_add_u32 v6, v9, v8 // (op 9821) 


v_lshrrev_b32 v7, 6, v1 // (op 9896) 
v_and_b32 v8, 1, v7 // (op 9896) 

v_and_b32 v7, 15, v1 // (op 9896) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 9896) 


v_and_b32 v7, 48, v1 // (op 9896) 
v_lshl_add_u32 v8, v9, 7, v7 // (op 9896) 


v_lshrrev_b32 v7, 7, v1 // (op 9972) 
v_and_b32 v9, 15, v1 // (op 9972) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 9972) 


v_and_b32 v7, 63, v1 // (op 9972) 
v_lshrrev_b32 v9, 4, v7 // (op 9972) 

v_lshl_add_u32 v7, v9, 6, v10 // (op 9972) 


v_lshrrev_b32 v9, 6, v1 // (op 10048) 
v_and_b32 v10, 1, v9 // (op 10048) 

v_and_b32 v9, 15, v1 // (op 10048) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 10048) 


v_and_b32 v9, 63, v1 // (op 10048) 
v_lshrrev_b32 v10, 4, v9 // (op 10048) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 10048) 


v_and_b32 v10, 48, v1 // (op 10162) 
v_and_b32 v11, 63, v1 // (op 10162) 
v_sub_u32 v12, v11, v10 // (op 10162) 

v_lshrrev_b32 v11, 1, v12 // (op 10162) 
v_add_lshl_u32 v13, v10, v11, 6 // (op 10162) 


v_and_b32 v10, 1, v12 // (op 10162) 

v_lshlrev_b32 v11, 3, v10 // (op 10162) 

v_lshrrev_b32 v10, 7, v1 // (op 10162) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 10162) 


v_add_u32 v10, v13, v12 // (op 10162) 


v_lshrrev_b32 v11, 6, v1 // (op 10206) 
v_and_b32 v12, 1, v11 // (op 10206) 

v_and_b32 v11, 15, v1 // (op 10206) 
v_lshl_add_u32 v13, v12, 4, v11 // (op 10206) 


v_and_b32 v11, 48, v1 // (op 10206) 
v_lshl_add_u32 v12, v13, 7, v11 // (op 10206) 


v_lshrrev_b32 v11, 7, v1 // (op 10235) 
v_and_b32 v13, 15, v1 // (op 10235) 
v_lshl_add_u32 v14, v11, 4, v13 // (op 10235) 


v_and_b32 v11, 63, v1 // (op 10235) 
v_lshrrev_b32 v13, 4, v11 // (op 10235) 

v_lshl_add_u32 v11, v13, 6, v14 // (op 10235) 


v_lshrrev_b32 v13, 6, v1 // (op 10264) 
v_and_b32 v14, 1, v13 // (op 10264) 

v_and_b32 v13, 15, v1 // (op 10264) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 10264) 


v_and_b32 v13, 63, v1 // (op 10264) 
v_lshrrev_b32 v14, 4, v13 // (op 10264) 

v_lshl_add_u32 v13, v15, 2, v14 // (op 10264) 


v_and_b32 v14, 48, v1 // (op 10308) 
v_and_b32 v15, 63, v1 // (op 10308) 
v_sub_u32 v16, v15, v14 // (op 10308) 

v_lshrrev_b32 v15, 1, v16 // (op 10308) 
v_add_lshl_u32 v17, v14, v15, 6 // (op 10308) 


v_and_b32 v14, 1, v16 // (op 10308) 

v_lshlrev_b32 v15, 3, v14 // (op 10308) 

v_lshrrev_b32 v14, 7, v1 // (op 10308) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 10308) 


v_add_u32 v14, v17, v16 // (op 10308) 


v_lshrrev_b32 v15, 6, v1 // (op 10352) 
v_and_b32 v16, 1, v15 // (op 10352) 

v_and_b32 v15, 15, v1 // (op 10352) 
v_lshl_add_u32 v17, v16, 4, v15 // (op 10352) 


v_and_b32 v15, 48, v1 // (op 10352) 
v_lshl_add_u32 v16, v17, 7, v15 // (op 10352) 


v_lshrrev_b32 v15, 7, v1 // (op 10381) 
v_and_b32 v17, 15, v1 // (op 10381) 
v_lshl_add_u32 v18, v15, 4, v17 // (op 10381) 


v_and_b32 v15, 63, v1 // (op 10381) 
v_lshrrev_b32 v17, 4, v15 // (op 10381) 

v_lshl_add_u32 v15, v17, 6, v18 // (op 10381) 


v_lshrrev_b32 v17, 6, v1 // (op 10410) 
v_and_b32 v18, 1, v17 // (op 10410) 

v_and_b32 v17, 15, v1 // (op 10410) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 10410) 


v_and_b32 v17, 63, v1 // (op 10410) 
v_lshrrev_b32 v18, 4, v17 // (op 10410) 

v_lshl_add_u32 v17, v19, 2, v18 // (op 10410) 


v_mov_b32 v18, s6 // (op 2) Move value
v_mov_b32 v19, s7 // (op 43) Move value


s_mov_b32 s0, 0 // (op 387) call()
s_mov_b32 s1, 0 // (op 388) call()
s_mov_b32 s6, 0 // (op 389) call()
s_mov_b32 s7, 0 // (op 390) call()
s_mov_b32 s18, 0 // (op 391) call()
s_mov_b32 s19, 0 // (op 392) call()
s_mov_b32 s0, 0 // (op 287) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 286) 
s_cmp_gt_u32 s9, s52 // (op 286) 
s_mov_b32 s53, scc // (op 286) 
s_sub_u32 s54, s9, s52 // (op 286) 

s_cmp_eq_i32 s53, 0 // (op 286) 
s_cselect_b32 s52, 0, s54 // (op 286) 


s_cmp_lt_u32 s8, s52 // (op 286) 
s_mov_b32 s53, scc // (op 286) 
s_cmp_eq_i32 s53, 0 // (op 286) 
s_cselect_b32 s54, s52, s8 // (op 286) 


s_cmp_lt_u32 s0, s54 // (op 286) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_286 // (op 286) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_286 if false)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_SKStreamTileLoop_286:
v_lshrrev_b32 v20, 7, v1 // (op 9378) 
s_mul_i32 s52, s16, s8 // (op 9378) 
s_add_u32 s52, s52, s0 // (op 9378) 
v_and_b32 v21, 63, v1 // (op 9378) 
s_mul_hi_u32 s53, s52, s13 // (op 9378) 
v_lshrrev_b32 v22, 4, v21 // (op 9378) 
s_sub_u32 s54, s52, s53 // (op 9378) 
v_lshlrev_b32 v23, 2, v22 // (op 9378) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 9378) 


s_lshr_b32 s55, s54, 1 // (op 9378) 

s_add_u32 s54, s55, s53 // (op 9378) 


v_lshrrev_b32 v20, 2, v21 // (op 9378) 

s_lshr_b32 s53, s54, s11 // (op 9378) 

v_and_b32 v21, 3, v20 // (op 9378) 

s_and_b32 s54, s11, 2147483648 // (op 9378) 
v_and_b32 v20, 3, v1 // (op 9378) 
s_cmp_eq_i32 s54, 0 // (op 9378) 
s_mov_b32 s55, scc // (op 9378) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 9378) 


s_cmp_eq_i32 s55, 0 // (op 9378) 
s_cselect_b32 s54, s52, s53 // (op 9378) 



v_lshrrev_b32 v20, 6, v1 // (op 9378) 
s_mul_hi_u32 s52, s54, s12 // (op 9378) 
v_and_b32 v21, 1, v20 // (op 9378) 

s_sub_u32 s53, s54, s52 // (op 9378) 
v_lshl_add_u32 v20, v21, 4, v23 // (op 9378) 


s_lshr_b32 s55, s53, 1 // (op 9378) 

s_add_u32 s53, s55, s52 // (op 9378) 


s_lshr_b32 s52, s53, s10 // (op 9378) 

s_and_b32 s53, s10, 2147483648 // (op 9378) 
s_cmp_eq_i32 s53, 0 // (op 9378) 
s_mov_b32 s55, scc // (op 9378) 

s_cmp_eq_i32 s55, 0 // (op 9378) 
s_cselect_b32 s53, s54, s52 // (op 9378) 


s_mul_i32 s52, s53, s14 // (op 9378) 
v_lshl_add_u32 v21, s53, 6, v22 // (op 9378) 


s_sub_u32 s53, s54, s52 // (op 9378) 


v_lshl_add_u32 v22, s53, 7, v20 // (op 9378) 


v_mul_lo_u32 v20, v22, s4 // (op 9378) 

v_add_lshl_u32 v22, v21, v20, 2 // (op 9378) 


s_mul_i32 s53, s4, s23 // (op 9380) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 9380) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 9380) least significant
s_mul_hi_u32 s55, s4, s22 // (op 9380) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9380) most significant: sum
s_add_u32 s53, s53, s55 // (op 9380) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 9380) 

s_mov_b32 s56, s20 // (op 9380) reg expression
s_mov_b32 s57, s21 // (op 9380) reg expression
s_mov_b32 s58, s54 // (op 9380) reg expression
s_mov_b32 s59, 131072 // (op 9380) call()

v_and_b32 v20, 48, v1 // (op 9484) 
v_and_b32 v21, 63, v1 // (op 9484) 
v_sub_u32 v23, v21, v20 // (op 9484) 

v_lshrrev_b32 v21, 1, v23 // (op 9484) 
v_add_lshl_u32 v24, v20, v21, 6 // (op 9484) 


v_and_b32 v20, 1, v23 // (op 9484) 

v_lshlrev_b32 v21, 3, v20 // (op 9484) 

v_lshrrev_b32 v20, 7, v1 // (op 9484) 
v_lshl_add_u32 v23, v20, 4, v21 // (op 9484) 


v_add_u32 v20, v24, v23 // (op 9484) 


v_lshrrev_b32 v21, 6, v1 // (op 9559) 
v_and_b32 v23, 1, v21 // (op 9559) 

v_and_b32 v21, 15, v1 // (op 9559) 
v_lshl_add_u32 v24, v23, 4, v21 // (op 9559) 


v_and_b32 v21, 48, v1 // (op 9559) 
v_lshl_add_u32 v23, v24, 7, v21 // (op 9559) 


v_lshrrev_b32 v21, 7, v1 // (op 9635) 
v_and_b32 v24, 15, v1 // (op 9635) 
v_lshl_add_u32 v25, v21, 4, v24 // (op 9635) 


v_and_b32 v21, 63, v1 // (op 9635) 
v_lshrrev_b32 v24, 4, v21 // (op 9635) 

v_lshl_add_u32 v21, v24, 6, v25 // (op 9635) 


v_lshrrev_b32 v24, 6, v1 // (op 9711) 
v_and_b32 v25, 1, v24 // (op 9711) 

v_and_b32 v24, 15, v1 // (op 9711) 
v_lshl_add_u32 v26, v25, 4, v24 // (op 9711) 


v_and_b32 v24, 63, v1 // (op 9711) 
v_lshrrev_b32 v25, 4, v24 // (op 9711) 

v_lshl_add_u32 v24, v26, 2, v25 // (op 9711) 


v_lshrrev_b32 v25, 7, v1 // (op 9734) 
s_mul_i32 s52, s16, s8 // (op 9734) 
s_add_u32 s52, s52, s0 // (op 9734) 
v_and_b32 v26, 63, v1 // (op 9734) 
s_mul_hi_u32 s53, s52, s13 // (op 9734) 
v_lshrrev_b32 v27, 4, v26 // (op 9734) 
s_sub_u32 s54, s52, s53 // (op 9734) 
v_lshlrev_b32 v28, 2, v27 // (op 9734) 

s_lshr_b32 s55, s54, 1 // (op 9734) 

v_lshl_add_u32 v27, v25, 4, v28 // (op 9734) 


s_add_u32 s54, s55, s53 // (op 9734) 


v_lshrrev_b32 v25, 2, v26 // (op 9734) 

s_lshr_b32 s53, s54, s11 // (op 9734) 

v_and_b32 v26, 3, v25 // (op 9734) 

s_and_b32 s54, s11, 2147483648 // (op 9734) 
v_and_b32 v25, 3, v1 // (op 9734) 
s_cmp_eq_i32 s54, 0 // (op 9734) 
s_mov_b32 s55, scc // (op 9734) 

v_lshl_add_u32 v28, v26, 2, v25 // (op 9734) 


s_cmp_eq_i32 s55, 0 // (op 9734) 
s_cselect_b32 s54, s52, s53 // (op 9734) 



v_lshrrev_b32 v25, 6, v1 // (op 9734) 
s_mul_hi_u32 s55, s54, s12 // (op 9734) 
v_and_b32 v26, 1, v25 // (op 9734) 

s_sub_u32 s52, s54, s55 // (op 9734) 
v_lshl_add_u32 v25, v26, 4, v28 // (op 9734) 


s_lshr_b32 s53, s52, 1 // (op 9734) 

s_add_u32 s52, s53, s55 // (op 9734) 


s_lshr_b32 s53, s52, s10 // (op 9734) 

s_and_b32 s52, s10, 2147483648 // (op 9734) 
s_cmp_eq_i32 s52, 0 // (op 9734) 
s_mov_b32 s55, scc // (op 9734) 

s_cmp_eq_i32 s55, 0 // (op 9734) 
s_cselect_b32 s52, s54, s53 // (op 9734) 


s_mul_i32 s53, s52, s14 // (op 9734) 
v_lshl_add_u32 v26, s52, 6, v27 // (op 9734) 


s_sub_u32 s52, s54, s53 // (op 9734) 


v_lshl_add_u32 v27, s52, 7, v25 // (op 9734) 


v_mul_lo_u32 v25, v27, s24 // (op 9734) 

v_add_lshl_u32 v27, v26, v25, 2 // (op 9734) 


s_mul_i32 s53, s24, s23 // (op 9736) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 9736) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 9736) least significant
s_mul_hi_u32 s55, s24, s22 // (op 9736) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9736) most significant: sum
s_add_u32 s53, s53, s55 // (op 9736) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 9736) 

s_mov_b32 s60, s26 // (op 9736) reg expression
s_mov_b32 s61, s27 // (op 9736) reg expression
s_mov_b32 s62, s54 // (op 9736) reg expression
s_mov_b32 s63, 131072 // (op 9736) call()

v_lshrrev_b32 v25, 5, v1 // (op 10088) 
v_lshlrev_b32 v26, 2, v25 // (op 10088) 

v_lshl_add_u32 v25, s16, 6, v26 // (op 10088) 

v_and_b32 v26, -32, v1 // (op 10088) 
v_sub_u32 v28, v1, v26 // (op 10088) 

v_lshl_add_u32 v26, v25, 7, v28 // (op 10088) 


v_lshlrev_b32 v25, 2, v26 // (op 10088) 

s_lshl_b32 s52, s15, 15 // (op 10090) 
s_mov_b32 s64, s28 // (op 10090) reg expression
s_mov_b32 s65, s29 // (op 10090) reg expression
s_mov_b32 s66, s52 // (op 10090) reg expression
s_mov_b32 s67, 131072 // (op 10090) call()

s_mul_i32 s6, s8, s16 // (op 291) 
s_add_u32 s6, s6, s0 // (op 291) 
s_mul_hi_u32 s52, s6, s13 // (op 292) 
s_sub_u32 s53, s6, s52 // (op 292) 
s_lshr_b32 s54, s53, 1 // (op 292) 

s_add_u32 s53, s54, s52 // (op 292) 


s_lshr_b32 s52, s53, s11 // (op 292) 

s_and_b32 s53, s11, 2147483648 // (op 292) 
s_cmp_eq_i32 s53, 0 // (op 292) 
s_mov_b32 s54, scc // (op 292) 

s_cmp_eq_i32 s54, 0 // (op 292) 
s_cselect_b32 s53, s6, s52 // (op 292) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 292) 
s_mul_i32 s52, s53, s54 // (op 292) 


s_sub_u32 s18, s6, s52 // (op 292) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 293) 
s_cmp_gt_u32 s52, s18 // (op 293) 
s_mov_b32 s54, scc // (op 293) 
s_sub_u32 s55, s52, s18 // (op 293) 

s_cmp_eq_i32 s54, 0 // (op 293) 
s_cselect_b32 s52, 0, s55 // (op 293) 


s_cmp_gt_u32 s8, s0 // (op 293) 
s_mov_b32 s53, scc // (op 293) 
s_sub_u32 s54, s8, s0 // (op 293) 
s_cmp_eq_i32 s53, 0 // (op 293) 
s_cselect_b32 s55, 0, s54 // (op 293) 


s_cmp_lt_u32 s52, s55 // (op 293) 
s_mov_b32 s53, scc // (op 293) 
s_cmp_eq_i32 s53, 0 // (op 293) 
s_cselect_b32 s54, s55, s52 // (op 293) 



s_cmp_gt_u32 s9, s6 // (op 293) 
s_mov_b32 s55, scc // (op 293) 
s_sub_u32 s52, s9, s6 // (op 293) 
s_cmp_eq_i32 s55, 0 // (op 293) 
s_cselect_b32 s53, 0, s52 // (op 293) 


s_cmp_lt_u32 s54, s53 // (op 293) 
s_mov_b32 s52, scc // (op 293) 
s_cmp_eq_i32 s52, 0 // (op 293) 
s_cselect_b32 s7, s53, s54 // (op 293) 



s_add_u32 s52, s18, s7 // (op 294) 
s_sub_u32 s19, s52, 1 // (op 294) 

v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 2666) call()
v_accvgpr_write a5, 0.00000 // (op 2666) call()
v_accvgpr_write a6, 0.00000 // (op 2666) call()
v_accvgpr_write a7, 0.00000 // (op 2666) call()
v_accvgpr_write a8, 0.00000 // (op 2842) call()
v_accvgpr_write a9, 0.00000 // (op 2842) call()
v_accvgpr_write a10, 0.00000 // (op 2842) call()
v_accvgpr_write a11, 0.00000 // (op 2842) call()
v_accvgpr_write a12, 0.00000 // (op 3018) call()
v_accvgpr_write a13, 0.00000 // (op 3018) call()
v_accvgpr_write a14, 0.00000 // (op 3018) call()
v_accvgpr_write a15, 0.00000 // (op 3018) call()
v_accvgpr_write a16, 0.00000 // (op 3282) call()
v_accvgpr_write a17, 0.00000 // (op 3282) call()
v_accvgpr_write a18, 0.00000 // (op 3282) call()
v_accvgpr_write a19, 0.00000 // (op 3282) call()
v_accvgpr_write a20, 0.00000 // (op 3591) call()
v_accvgpr_write a21, 0.00000 // (op 3591) call()
v_accvgpr_write a22, 0.00000 // (op 3591) call()
v_accvgpr_write a23, 0.00000 // (op 3591) call()
v_accvgpr_write a24, 0.00000 // (op 3592) call()
v_accvgpr_write a25, 0.00000 // (op 3592) call()
v_accvgpr_write a26, 0.00000 // (op 3592) call()
v_accvgpr_write a27, 0.00000 // (op 3592) call()
v_accvgpr_write a28, 0.00000 // (op 3593) call()
v_accvgpr_write a29, 0.00000 // (op 3593) call()
v_accvgpr_write a30, 0.00000 // (op 3593) call()
v_accvgpr_write a31, 0.00000 // (op 3593) call()
v_and_b32 v26, -4, v1 // (op 8945) 
s_mul_i32 s52, s16, s8 // (op 8945) 
s_add_u32 s52, s52, s0 // (op 8945) 
v_sub_u32 v28, v1, v26 // (op 8945) 

s_mul_hi_u32 s53, s52, s13 // (op 8945) 
v_lshlrev_b32 v26, 4, v28 // (op 8945) 

s_sub_u32 s54, s52, s53 // (op 8945) 
v_lshrrev_b32 v28, 2, v1 // (op 8945) 
s_lshr_b32 s55, s54, 1 // (op 8945) 

s_add_u32 s54, s55, s53 // (op 8945) 


s_lshr_b32 s53, s54, s11 // (op 8945) 

s_and_b32 s54, s11, 2147483648 // (op 8945) 
s_cmp_eq_i32 s54, 0 // (op 8945) 
s_mov_b32 s55, scc // (op 8945) 

s_cmp_eq_i32 s55, 0 // (op 8945) 
s_cselect_b32 s54, s52, s53 // (op 8945) 


s_mul_hi_u32 s53, s54, s12 // (op 8945) 
s_sub_u32 s55, s54, s53 // (op 8945) 
s_lshr_b32 s68, s55, 1 // (op 8945) 

s_add_u32 s55, s68, s53 // (op 8945) 


s_lshr_b32 s53, s55, s10 // (op 8945) 

s_and_b32 s55, s10, 2147483648 // (op 8945) 
s_cmp_eq_i32 s55, 0 // (op 8945) 
s_mov_b32 s68, scc // (op 8945) 

s_cmp_eq_i32 s68, 0 // (op 8945) 
s_cselect_b32 s55, s54, s53 // (op 8945) 


v_lshl_add_u32 v29, s55, 6, v26 // (op 8945) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 8945) 
v_mov_b32 v31, 0 // (op 8945) convert
v_mov_b32 v30, v29 // (op 8945) convert

s_mul_i32 s53, s54, s68 // (op 8945) 


s_sub_u32 s54, s52, s53 // (op 8945) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 8945) 


v_mul_lo_u32 v29, v26, s31 // (op 8945) most significant: low of low * high
v_mul_hi_u32 v32, v26, s30 // (op 8945) most significant: high of low * low
v_mul_lo_u32 v28, v26, s30 // (op 8945) least significant: low of low * low
v_add_u32 v29, v29, v32 // (op 8945) most significant: sum


v_add_co_u32 v32, s[52:53], v30, v28 // (op 8945) least significant half
v_addc_co_u32 v33, s[52:53], v31, v29, s[52:53] // (op 8945) most significant half



s_mul_i32 s53, s30, s3 // (op 8947) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 8947) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 8947) least significant
s_mul_hi_u32 s55, s30, s2 // (op 8947) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 8947) most significant: sum
s_add_u32 s53, s53, s55 // (op 8947) most significant: sum


s_mov_b32 s68, s32 // (op 8947) reg expression
s_mov_b32 s69, s33 // (op 8947) reg expression
s_mov_b32 s70, s52 // (op 8947) reg expression
s_mov_b32 s71, 131072 // (op 8947) call()

v_and_b32 v26, -4, v1 // (op 8964) 
s_mul_i32 s52, s16, s8 // (op 8964) 
s_add_u32 s52, s52, s0 // (op 8964) 
v_sub_u32 v28, v1, v26 // (op 8964) 

s_mul_hi_u32 s53, s52, s13 // (op 8964) 
v_lshlrev_b32 v26, 4, v28 // (op 8964) 

s_sub_u32 s54, s52, s53 // (op 8964) 
v_lshrrev_b32 v28, 2, v1 // (op 8964) 
s_lshr_b32 s55, s54, 1 // (op 8964) 

s_add_u32 s54, s55, s53 // (op 8964) 


s_lshr_b32 s53, s54, s11 // (op 8964) 

s_and_b32 s54, s11, 2147483648 // (op 8964) 
s_cmp_eq_i32 s54, 0 // (op 8964) 
s_mov_b32 s55, scc // (op 8964) 

s_cmp_eq_i32 s55, 0 // (op 8964) 
s_cselect_b32 s54, s52, s53 // (op 8964) 


s_mul_hi_u32 s53, s54, s12 // (op 8964) 
s_sub_u32 s55, s54, s53 // (op 8964) 
s_lshr_b32 s72, s55, 1 // (op 8964) 

s_add_u32 s55, s72, s53 // (op 8964) 


s_lshr_b32 s53, s55, s10 // (op 8964) 

s_and_b32 s55, s10, 2147483648 // (op 8964) 
s_cmp_eq_i32 s55, 0 // (op 8964) 
s_mov_b32 s72, scc // (op 8964) 

s_cmp_eq_i32 s72, 0 // (op 8964) 
s_cselect_b32 s55, s54, s53 // (op 8964) 


v_lshl_add_u32 v29, s55, 6, v26 // (op 8964) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 8964) 
v_mov_b32 v31, 0 // (op 8964) convert
v_mov_b32 v30, v29 // (op 8964) convert

s_mul_i32 s53, s54, s72 // (op 8964) 


s_sub_u32 s54, s52, s53 // (op 8964) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 8964) 


v_mul_lo_u32 v29, v26, s31 // (op 8964) most significant: low of low * high
v_mul_hi_u32 v34, v26, s30 // (op 8964) most significant: high of low * low
v_mul_lo_u32 v28, v26, s30 // (op 8964) least significant: low of low * low
v_add_u32 v29, v29, v34 // (op 8964) most significant: sum


v_add_co_u32 v34, s[52:53], v30, v28 // (op 8964) least significant half
v_addc_co_u32 v35, s[52:53], v31, v29, s[52:53] // (op 8964) most significant half



s_mul_i32 s53, s30, s3 // (op 8966) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 8966) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 8966) least significant
s_mul_hi_u32 s55, s30, s2 // (op 8966) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 8966) most significant: sum
s_add_u32 s53, s53, s55 // (op 8966) most significant: sum


s_mov_b32 s68, s32 // (op 8966) reg expression
s_mov_b32 s69, s33 // (op 8966) reg expression
s_mov_b32 s70, s52 // (op 8966) reg expression
s_mov_b32 s71, 131072 // (op 8966) call()

v_lshrrev_b32 v26, 3, v1 // (op 9073) 
s_mul_i32 s52, s16, s8 // (op 9073) 
s_add_u32 s52, s52, s0 // (op 9073) 
v_and_b32 v28, -8, v1 // (op 9073) 
s_mul_hi_u32 s53, s52, s13 // (op 9073) 
v_sub_u32 v29, v1, v28 // (op 9073) 

s_sub_u32 s54, s52, s53 // (op 9073) 
v_lshlrev_b32 v28, 4, v29 // (op 9073) 

s_lshr_b32 s55, s54, 1 // (op 9073) 

s_add_u32 s54, s55, s53 // (op 9073) 


s_lshr_b32 s53, s54, s11 // (op 9073) 

s_and_b32 s54, s11, 2147483648 // (op 9073) 
s_cmp_eq_i32 s54, 0 // (op 9073) 
s_mov_b32 s55, scc // (op 9073) 

s_cmp_eq_i32 s55, 0 // (op 9073) 
s_cselect_b32 s54, s52, s53 // (op 9073) 


s_mul_hi_u32 s53, s54, s12 // (op 9073) 
s_sub_u32 s55, s54, s53 // (op 9073) 
s_lshr_b32 s72, s55, 1 // (op 9073) 

s_add_u32 s55, s72, s53 // (op 9073) 


s_lshr_b32 s53, s55, s10 // (op 9073) 

s_and_b32 s55, s10, 2147483648 // (op 9073) 
s_cmp_eq_i32 s55, 0 // (op 9073) 
s_mov_b32 s72, scc // (op 9073) 

s_cmp_eq_i32 s72, 0 // (op 9073) 
s_cselect_b32 s55, s54, s53 // (op 9073) 


s_mul_i32 s53, s55, s14 // (op 9073) 

s_sub_u32 s55, s54, s53 // (op 9073) 

v_lshl_add_u32 v29, s55, 7, v26 // (op 9073) 


v_mul_lo_u32 v31, v29, s35 // (op 9073) most significant: low of low * high
v_mul_hi_u32 v26, v29, s34 // (op 9073) most significant: high of low * low
v_mul_lo_u32 v30, v29, s34 // (op 9073) least significant: low of low * low
v_add_u32 v31, v31, v26 // (op 9073) most significant: sum


s_ashr_i64 s[72:73], s[2:3], 7 // (op 9073) 
s_mul_i32 s53, s54, s72 // (op 9073) 


s_sub_u32 s54, s52, s53 // (op 9073) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 9073) 


v_mov_b32 v29, 0 // (op 9073) convert
v_mov_b32 v28, v26 // (op 9073) convert

v_add_co_u32 v36, s[52:53], v28, v30 // (op 9073) least significant half
v_addc_co_u32 v37, s[52:53], v29, v31, s[52:53] // (op 9073) most significant half



s_mul_i32 s53, s34, s23 // (op 9075) most significant: low of low * high
s_mul_i32 s54, s35, s22 // (op 9075) most significant: low of high * low
s_mul_i32 s52, s34, s22 // (op 9075) least significant
s_mul_hi_u32 s55, s34, s22 // (op 9075) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9075) most significant: sum
s_add_u32 s53, s53, s55 // (op 9075) most significant: sum


s_mov_b32 s72, s36 // (op 9075) reg expression
s_mov_b32 s73, s37 // (op 9075) reg expression
s_mov_b32 s74, s52 // (op 9075) reg expression
s_mov_b32 s75, 131072 // (op 9075) call()

v_lshrrev_b32 v26, 3, v1 // (op 9092) 
s_mul_i32 s52, s16, s8 // (op 9092) 
s_add_u32 s52, s52, s0 // (op 9092) 
v_and_b32 v28, -8, v1 // (op 9092) 
s_mul_hi_u32 s53, s52, s13 // (op 9092) 
v_sub_u32 v29, v1, v28 // (op 9092) 

s_sub_u32 s54, s52, s53 // (op 9092) 
v_lshlrev_b32 v28, 4, v29 // (op 9092) 

s_lshr_b32 s55, s54, 1 // (op 9092) 

s_add_u32 s54, s55, s53 // (op 9092) 


s_lshr_b32 s53, s54, s11 // (op 9092) 

s_and_b32 s54, s11, 2147483648 // (op 9092) 
s_cmp_eq_i32 s54, 0 // (op 9092) 
s_mov_b32 s55, scc // (op 9092) 

s_cmp_eq_i32 s55, 0 // (op 9092) 
s_cselect_b32 s54, s52, s53 // (op 9092) 


s_mul_hi_u32 s53, s54, s12 // (op 9092) 
s_sub_u32 s55, s54, s53 // (op 9092) 
s_lshr_b32 s76, s55, 1 // (op 9092) 

s_add_u32 s55, s76, s53 // (op 9092) 


s_lshr_b32 s53, s55, s10 // (op 9092) 

s_and_b32 s55, s10, 2147483648 // (op 9092) 
s_cmp_eq_i32 s55, 0 // (op 9092) 
s_mov_b32 s76, scc // (op 9092) 

s_cmp_eq_i32 s76, 0 // (op 9092) 
s_cselect_b32 s55, s54, s53 // (op 9092) 


s_mul_i32 s53, s55, s14 // (op 9092) 

s_sub_u32 s55, s54, s53 // (op 9092) 

v_lshl_add_u32 v29, s55, 7, v26 // (op 9092) 


v_mul_lo_u32 v31, v29, s35 // (op 9092) most significant: low of low * high
v_mul_hi_u32 v26, v29, s34 // (op 9092) most significant: high of low * low
v_mul_lo_u32 v30, v29, s34 // (op 9092) least significant: low of low * low
v_add_u32 v31, v31, v26 // (op 9092) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 9092) 
s_mul_i32 s53, s54, s76 // (op 9092) 


s_sub_u32 s54, s52, s53 // (op 9092) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 9092) 


v_mov_b32 v29, 0 // (op 9092) convert
v_mov_b32 v28, v26 // (op 9092) convert

v_add_co_u32 v38, s[52:53], v28, v30 // (op 9092) least significant half
v_addc_co_u32 v39, s[52:53], v29, v31, s[52:53] // (op 9092) most significant half



s_mul_i32 s53, s34, s23 // (op 9094) most significant: low of low * high
s_mul_i32 s54, s35, s22 // (op 9094) most significant: low of high * low
s_mul_i32 s52, s34, s22 // (op 9094) least significant
s_mul_hi_u32 s55, s34, s22 // (op 9094) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9094) most significant: sum
s_add_u32 s53, s53, s55 // (op 9094) most significant: sum


s_mov_b32 s72, s36 // (op 9094) reg expression
s_mov_b32 s73, s37 // (op 9094) reg expression
s_mov_b32 s74, s52 // (op 9094) reg expression
s_mov_b32 s75, 131072 // (op 9094) call()

v_and_b32 v26, -64, v1 // (op 9201) 
s_mul_i32 s52, s16, s8 // (op 9201) 
s_add_u32 s52, s52, s0 // (op 9201) 
v_sub_u32 v28, v1, v26 // (op 9201) 

s_mul_hi_u32 s53, s52, s13 // (op 9201) 
v_lshrrev_b32 v26, 6, v1 // (op 9201) 
s_sub_u32 s54, s52, s53 // (op 9201) 
s_lshr_b32 s55, s54, 1 // (op 9201) 

s_add_u32 s54, s55, s53 // (op 9201) 


s_lshr_b32 s53, s54, s11 // (op 9201) 

s_and_b32 s54, s11, 2147483648 // (op 9201) 
s_cmp_eq_i32 s54, 0 // (op 9201) 
s_mov_b32 s55, scc // (op 9201) 

s_cmp_eq_i32 s55, 0 // (op 9201) 
s_cselect_b32 s54, s52, s53 // (op 9201) 


s_mul_hi_u32 s53, s54, s12 // (op 9201) 
s_sub_u32 s55, s54, s53 // (op 9201) 
s_lshr_b32 s76, s55, 1 // (op 9201) 

s_add_u32 s55, s76, s53 // (op 9201) 


s_lshr_b32 s53, s55, s10 // (op 9201) 

s_and_b32 s55, s10, 2147483648 // (op 9201) 
s_cmp_eq_i32 s55, 0 // (op 9201) 
s_mov_b32 s76, scc // (op 9201) 

s_cmp_eq_i32 s76, 0 // (op 9201) 
s_cselect_b32 s55, s54, s53 // (op 9201) 


v_lshl_add_u32 v29, s55, 6, v28 // (op 9201) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 9201) 
s_mul_i32 s53, s54, s76 // (op 9201) 


s_sub_u32 s54, s52, s53 // (op 9201) 


v_lshl_add_u32 v28, s54, 2, v26 // (op 9201) 


v_mul_lo_u32 v26, v28, s38 // (op 9201) 

v_add_u32 v28, v29, v26 // (op 9201) 


s_mul_i32 s53, s38, s43 // (op 9203) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 9203) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 9203) least significant
s_mul_hi_u32 s55, s38, s42 // (op 9203) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9203) most significant: sum
s_add_u32 s53, s53, s55 // (op 9203) most significant: sum


s_mov_b32 s76, s40 // (op 9203) reg expression
s_mov_b32 s77, s41 // (op 9203) reg expression
s_mov_b32 s78, s52 // (op 9203) reg expression
s_mov_b32 s79, 131072 // (op 9203) call()

v_and_b32 v26, -64, v1 // (op 9220) 
s_mul_i32 s52, s16, s8 // (op 9220) 
s_add_u32 s52, s52, s0 // (op 9220) 
v_sub_u32 v29, v1, v26 // (op 9220) 

s_mul_hi_u32 s53, s52, s13 // (op 9220) 
v_lshrrev_b32 v26, 6, v1 // (op 9220) 
s_sub_u32 s54, s52, s53 // (op 9220) 
s_lshr_b32 s55, s54, 1 // (op 9220) 

s_add_u32 s54, s55, s53 // (op 9220) 


s_lshr_b32 s53, s54, s11 // (op 9220) 

s_and_b32 s54, s11, 2147483648 // (op 9220) 
s_cmp_eq_i32 s54, 0 // (op 9220) 
s_mov_b32 s55, scc // (op 9220) 

s_cmp_eq_i32 s55, 0 // (op 9220) 
s_cselect_b32 s54, s52, s53 // (op 9220) 


s_mul_hi_u32 s53, s54, s12 // (op 9220) 
s_sub_u32 s55, s54, s53 // (op 9220) 
s_lshr_b32 s80, s55, 1 // (op 9220) 

s_add_u32 s55, s80, s53 // (op 9220) 


s_lshr_b32 s53, s55, s10 // (op 9220) 

s_and_b32 s55, s10, 2147483648 // (op 9220) 
s_cmp_eq_i32 s55, 0 // (op 9220) 
s_mov_b32 s80, scc // (op 9220) 

s_cmp_eq_i32 s80, 0 // (op 9220) 
s_cselect_b32 s55, s54, s53 // (op 9220) 


v_lshl_add_u32 v30, s55, 6, v29 // (op 9220) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 9220) 
s_mul_i32 s53, s54, s80 // (op 9220) 


s_sub_u32 s54, s52, s53 // (op 9220) 


v_lshl_add_u32 v29, s54, 2, v26 // (op 9220) 


v_mul_lo_u32 v26, v29, s38 // (op 9220) 

v_add_u32 v29, v30, v26 // (op 9220) 


s_mul_i32 s53, s38, s43 // (op 9222) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 9222) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 9222) least significant
s_mul_hi_u32 s55, s38, s42 // (op 9222) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9222) most significant: sum
s_add_u32 s53, s53, s55 // (op 9222) most significant: sum


s_mov_b32 s76, s40 // (op 9222) reg expression
s_mov_b32 s77, s41 // (op 9222) reg expression
s_mov_b32 s78, s52 // (op 9222) reg expression
s_mov_b32 s79, 131072 // (op 9222) call()

v_lshrrev_b32 v26, 2, v1 // (op 9329) 
s_mul_i32 s52, s16, s8 // (op 9329) 
s_add_u32 s52, s52, s0 // (op 9329) 
v_lshlrev_b32 v30, 1, v26 // (op 9329) 

s_mul_hi_u32 s53, s52, s13 // (op 9329) 
v_and_b32 v26, -4, v1 // (op 9329) 
s_sub_u32 s54, s52, s53 // (op 9329) 
v_sub_u32 v31, v1, v26 // (op 9329) 

s_lshr_b32 s55, s54, 1 // (op 9329) 

s_add_u32 s54, s55, s53 // (op 9329) 


s_lshr_b32 s53, s54, s11 // (op 9329) 

s_and_b32 s54, s11, 2147483648 // (op 9329) 
s_cmp_eq_i32 s54, 0 // (op 9329) 
s_mov_b32 s55, scc // (op 9329) 

s_cmp_eq_i32 s55, 0 // (op 9329) 
s_cselect_b32 s54, s52, s53 // (op 9329) 


s_mul_hi_u32 s53, s54, s12 // (op 9329) 
s_sub_u32 s55, s54, s53 // (op 9329) 
s_lshr_b32 s80, s55, 1 // (op 9329) 

s_add_u32 s55, s80, s53 // (op 9329) 


s_lshr_b32 s53, s55, s10 // (op 9329) 

s_and_b32 s55, s10, 2147483648 // (op 9329) 
s_cmp_eq_i32 s55, 0 // (op 9329) 
s_mov_b32 s80, scc // (op 9329) 

s_cmp_eq_i32 s80, 0 // (op 9329) 
s_cselect_b32 s55, s54, s53 // (op 9329) 


s_mul_i32 s53, s55, s14 // (op 9329) 

s_sub_u32 s55, s54, s53 // (op 9329) 

v_lshl_add_u32 v26, s55, 7, v30 // (op 9329) 


v_mul_lo_u32 v30, v26, s44 // (op 9329) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 9329) 
s_mul_i32 s53, s54, s80 // (op 9329) 


s_sub_u32 s54, s52, s53 // (op 9329) 


v_lshl_add_u32 v26, s54, 2, v31 // (op 9329) 


v_add_u32 v31, v26, v30 // (op 9329) 


s_mul_i32 s53, s44, s23 // (op 9331) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 9331) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 9331) least significant
s_mul_hi_u32 s55, s44, s22 // (op 9331) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9331) most significant: sum
s_add_u32 s53, s53, s55 // (op 9331) most significant: sum


s_mov_b32 s80, s46 // (op 9331) reg expression
s_mov_b32 s81, s47 // (op 9331) reg expression
s_mov_b32 s82, s52 // (op 9331) reg expression
s_mov_b32 s83, 131072 // (op 9331) call()

v_lshrrev_b32 v26, 2, v1 // (op 9348) 
s_mul_i32 s52, s16, s8 // (op 9348) 
s_add_u32 s52, s52, s0 // (op 9348) 
v_lshlrev_b32 v30, 1, v26 // (op 9348) 

s_mul_hi_u32 s53, s52, s13 // (op 9348) 
v_and_b32 v26, -4, v1 // (op 9348) 
s_sub_u32 s54, s52, s53 // (op 9348) 
v_sub_u32 v40, v1, v26 // (op 9348) 

s_lshr_b32 s55, s54, 1 // (op 9348) 

s_add_u32 s54, s55, s53 // (op 9348) 


s_lshr_b32 s53, s54, s11 // (op 9348) 

s_and_b32 s54, s11, 2147483648 // (op 9348) 
s_cmp_eq_i32 s54, 0 // (op 9348) 
s_mov_b32 s55, scc // (op 9348) 

s_cmp_eq_i32 s55, 0 // (op 9348) 
s_cselect_b32 s54, s52, s53 // (op 9348) 


s_mul_hi_u32 s53, s54, s12 // (op 9348) 
s_sub_u32 s55, s54, s53 // (op 9348) 
s_lshr_b32 s84, s55, 1 // (op 9348) 

s_add_u32 s55, s84, s53 // (op 9348) 


s_lshr_b32 s53, s55, s10 // (op 9348) 

s_and_b32 s55, s10, 2147483648 // (op 9348) 
s_cmp_eq_i32 s55, 0 // (op 9348) 
s_mov_b32 s84, scc // (op 9348) 

s_cmp_eq_i32 s84, 0 // (op 9348) 
s_cselect_b32 s55, s54, s53 // (op 9348) 


s_mul_i32 s53, s55, s14 // (op 9348) 

s_sub_u32 s55, s54, s53 // (op 9348) 

v_lshl_add_u32 v26, s55, 7, v30 // (op 9348) 


v_mul_lo_u32 v30, v26, s44 // (op 9348) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 9348) 
s_mul_i32 s53, s54, s84 // (op 9348) 


s_sub_u32 s54, s52, s53 // (op 9348) 


v_lshl_add_u32 v26, s54, 2, v40 // (op 9348) 


v_add_u32 v40, v26, v30 // (op 9348) 


s_mul_i32 s53, s44, s23 // (op 9350) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 9350) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 9350) least significant
s_mul_hi_u32 s55, s44, s22 // (op 9350) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9350) most significant: sum
s_add_u32 s53, s53, s55 // (op 9350) most significant: sum


s_mov_b32 s80, s46 // (op 9350) reg expression
s_mov_b32 s81, s47 // (op 9350) reg expression
s_mov_b32 s82, s52 // (op 9350) reg expression
s_mov_b32 s83, 131072 // (op 9350) call()

v_and_b32 v26, -8, v1 // (op 9525) 
v_sub_u32 v30, v1, v26 // (op 9525) 

v_lshlrev_b32 v26, 4, v30 // (op 9525) 

v_lshrrev_b32 v30, 3, v1 // (op 9525) 
v_lshl_add_u32 v41, v30, 7, v26 // (op 9525) 


s_nop 0
v_readfirstlane_b32 s52, v41 // (op 9525) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10600) 

s_lshl_b64 s[54:55], s[34:35], 5 // (op 8811) 
v_mov_b64 v[42:43], s[54:55] // (op 8811) convert


v_mov_b64 v[44:45], v[36:37] // (op 8811) 
s_mov_b32 m0, s52 // (op 8811) reg expression
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 8811) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 8811) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 8811) most significant half

s_add_u32 m0, m0, 4096 // (op 8811) 
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 8811) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 8811) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 8811) most significant half

s_add_u32 m0, m0, 4096 // (op 8811) 
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 8811) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 8811) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 8811) most significant half

s_add_u32 m0, m0, 4096 // (op 8811) 
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 8811) Load value direct to lds




v_mov_b64 v[42:43], s[44:45] // (op 8546) convert

v_mov_b32 v26, v31 // (op 8546) 
buffer_load_ubyte v44, v26, s[80:83], 0 offen offset: 0 // (op 8546) Load value
v_add_u32 v26, v26, v42 // (op 8546) 
buffer_load_ubyte v45, v26, s[80:83], 0 offen offset: 0 // (op 8546) Load value


v_and_b32 v26, -4, v1 // (op 9450) 
v_sub_u32 v30, v1, v26 // (op 9450) 

v_lshlrev_b32 v26, 4, v30 // (op 9450) 

v_lshrrev_b32 v30, 2, v1 // (op 9450) 
v_lshl_add_u32 v41, v30, 6, v26 // (op 9450) 


s_nop 0
v_readfirstlane_b32 s52, v41 // (op 9450) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10581) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 8816) 
v_mov_b64 v[42:43], s[54:55] // (op 8816) convert


v_mov_b64 v[46:47], v[32:33] // (op 8816) reg expression
s_add_u32 s53, 16384, s52 // (op 8816) 
s_mov_b32 m0, s53 // (op 8816) reg expression
buffer_load_dwordx4 v46, s[68:71], 0 offen offset: 0 lds // (op 8816) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v42 // (op 8816) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 8816) most significant half

s_add_u32 m0, m0, 4096 // (op 8816) 
buffer_load_dwordx4 v46, s[68:71], 0 offen offset: 0 lds // (op 8816) Load value direct to lds






v_mov_b32 v26, v28 // (op 8552) reg expression

buffer_load_ubyte v30, v26, s[76:79], 0 offen offset: 0 // (op 8552) Load value

v_and_b32 v26, -8, v1 // (op 10335) 
v_sub_u32 v41, v1, v26 // (op 10335) 

v_lshlrev_b32 v26, 4, v41 // (op 10335) 

v_lshrrev_b32 v41, 3, v1 // (op 10335) 
v_lshl_add_u32 v42, v41, 7, v26 // (op 10335) 


s_nop 0
v_readfirstlane_b32 s52, v42 // (op 10335) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10783) 

s_lshl_b64 s[54:55], s[34:35], 5 // (op 8821) 
v_mov_b64 v[42:43], s[54:55] // (op 8821) convert


v_mov_b64 v[46:47], v[36:37] // (op 8821) 
s_add_u32 s53, 24576, s52 // (op 8821) 
s_mov_b32 m0, s53 // (op 8821) reg expression
s_mov_b32 s54, 128 // (op 8821) call()
buffer_load_dwordx4 v46, s[72:75], s54 offen offset: 0 lds // (op 8821) Load value direct to lds
v_add_co_u32 v46, s[84:85], v46, v42 // (op 8821) least significant half
v_addc_co_u32 v47, s[84:85], 0, v47, s[84:85] // (op 8821) most significant half

s_add_u32 m0, m0, 4096 // (op 8821) 
buffer_load_dwordx4 v46, s[72:75], s54 offen offset: 0 lds // (op 8821) Load value direct to lds
v_add_co_u32 v46, s[84:85], v46, v42 // (op 8821) least significant half
v_addc_co_u32 v47, s[84:85], 0, v47, s[84:85] // (op 8821) most significant half

s_add_u32 m0, m0, 4096 // (op 8821) 
buffer_load_dwordx4 v46, s[72:75], s54 offen offset: 0 lds // (op 8821) Load value direct to lds
v_add_co_u32 v46, s[84:85], v46, v42 // (op 8821) least significant half
v_addc_co_u32 v47, s[84:85], 0, v47, s[84:85] // (op 8821) most significant half

s_add_u32 m0, m0, 4096 // (op 8821) 
buffer_load_dwordx4 v46, s[72:75], s54 offen offset: 0 lds // (op 8821) Load value direct to lds







v_mov_b64 v[36:37], s[44:45] // (op 8558) convert

v_mov_b32 v26, v31 // (op 8558) 
buffer_load_ubyte v42, v26, s[80:83], 0 offen offset:4 // (op 8558) Load value
v_add_u32 v26, v26, v36 // (op 8558) 
buffer_load_ubyte v43, v26, s[80:83], 0 offen offset:4 // (op 8558) Load value



v_and_b32 v26, -4, v1 // (op 10291) 
v_sub_u32 v31, v1, v26 // (op 10291) 

v_lshlrev_b32 v26, 4, v31 // (op 10291) 

v_lshrrev_b32 v31, 2, v1 // (op 10291) 
v_lshl_add_u32 v41, v31, 6, v26 // (op 10291) 


s_nop 0
v_readfirstlane_b32 s52, v41 // (op 10291) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10773) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 8823) 
v_mov_b64 v[36:37], s[54:55] // (op 8823) convert


s_lshl_b64 s[54:55], s[30:31], 7 // (op 8823) 
v_mov_b32 v26, s54 // (op 8823) 
v_mov_b32 v31, s55 // (op 8823) 
v_add_co_u32 v46, s[84:85], v32, v26 // (op 8823) least significant half
v_addc_co_u32 v47, s[84:85], v33, v31, s[84:85] // (op 8823) most significant half




s_add_u32 s53, 40960, s52 // (op 8823) 
s_mov_b32 m0, s53 // (op 8823) reg expression
buffer_load_dwordx4 v46, s[68:71], 0 offen offset: 0 lds // (op 8823) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v36 // (op 8823) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 8823) most significant half

s_add_u32 m0, m0, 4096 // (op 8823) 
buffer_load_dwordx4 v46, s[68:71], 0 offen offset: 0 lds // (op 8823) Load value direct to lds







s_lshl_b32 s52, s38, 2 // (op 8564) 
v_add_u32 v26, v28, s52 // (op 8564) 


buffer_load_ubyte v41, v26, s[76:79], 0 offen offset: 0 // (op 8564) Load value


v_and_b32 v26, -4, v1 // (op 9677) 
v_sub_u32 v28, v1, v26 // (op 9677) 

v_lshrrev_b32 v26, 2, v1 // (op 9677) 
v_lshl_add_u32 v31, v26, 3, v28 // (op 9677) 


s_barrier  // (op 10641) 



s_waitcnt vmcnt(13)
ds_write_b8 v31, v44 offset:49152 // (op 8567) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v31, v45 offset:49156 // (op 8567) Store local data 

v_and_b32 v26, -64, v1 // (op 9601) 
v_sub_u32 v28, v1, v26 // (op 9601) 

v_and_b32 v26, 4294967232, v1 // (op 9601) 
v_add_u32 v31, v26, v28 // (op 9601) 


s_barrier  // (op 10620) 



s_waitcnt vmcnt(9)
ds_write_b8 v31, v30 offset:49664 // (op 8569) Store local data 

s_barrier  // (op 10597) 



ds_read_b128 v[46:49], v23 // (op 8594) Load local data 
ds_read_b128 v[50:53], v23 offset:64 // (op 8594) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10638) 



ds_read_u8 v26, v24 offset:49152 // (op 8624) Load local data 



ds_read_b128 v[54:57], v23 offset:4096 // (op 8599) Load local data 
ds_read_b128 v[58:61], v23 offset:4160 // (op 8599) Load local data 



ds_read_u8 v28, v24 offset:49280 // (op 8629) Load local data 



ds_read_b128 v[62:65], v23 offset:8192 // (op 8604) Load local data 
ds_read_b128 v[66:69], v23 offset:8256 // (op 8604) Load local data 



ds_read_u8 v31, v24 offset:49408 // (op 8634) Load local data 



ds_read_b128 v[70:73], v23 offset:12288 // (op 8609) Load local data 
ds_read_b128 v[74:77], v23 offset:12352 // (op 8609) Load local data 



ds_read_u8 v32, v24 offset:49536 // (op 8639) Load local data 
s_barrier  // (op 10578) 



ds_read_b64_tr_b8 v[78:79], v20 offset:16384 // (op 8584) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v20 offset:16896 // (op 8584) Transpose load local data 
ds_read_b64_tr_b8 v[82:83], v20 offset:20480 // (op 8584) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v20 offset:20992 // (op 8584) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 10617) 


ds_read_u8 v33, v21 offset:49664 // (op 8614) Load local data 



ds_read_b64_tr_b8 v[86:87], v20 offset:16416 // (op 8589) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v20 offset:16928 // (op 8589) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v20 offset:20512 // (op 8589) Transpose load local data 
ds_read_b64_tr_b8 v[92:93], v20 offset:21024 // (op 8589) Transpose load local data 


ds_read_u8 v36, v21 offset:49696 // (op 8619) Load local data 
v_and_b32 v37, -4, v1 // (op 9422) 
v_sub_u32 v94, v1, v37 // (op 9422) 

v_lshlrev_b32 v37, 4, v94 // (op 9422) 

v_lshrrev_b32 v94, 2, v1 // (op 9422) 
v_lshl_add_u32 v95, v94, 6, v37 // (op 9422) 


s_nop 0
v_readfirstlane_b32 s52, v95 // (op 9422) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -8, v1 // (op 9497) 
v_sub_u32 v94, v1, v37 // (op 9497) 

v_lshlrev_b32 v37, 4, v94 // (op 9497) 

v_lshrrev_b32 v94, 3, v1 // (op 9497) 
v_lshl_add_u32 v95, v94, 7, v37 // (op 9497) 


s_nop 0
v_readfirstlane_b32 s53, v95 // (op 9497) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -64, v1 // (op 9572) 
v_sub_u32 v94, v1, v37 // (op 9572) 

v_and_b32 v37, 4294967232, v1 // (op 9572) 
v_add_u32 v95, v37, v94 // (op 9572) 


v_and_b32 v37, -4, v1 // (op 9648) 
v_sub_u32 v94, v1, v37 // (op 9648) 

v_lshrrev_b32 v37, 2, v1 // (op 9648) 
v_lshl_add_u32 v96, v37, 3, v94 // (op 9648) 


v_and_b32 v37, -4, v1 // (op 10276) 
v_sub_u32 v94, v1, v37 // (op 10276) 

v_lshlrev_b32 v37, 4, v94 // (op 10276) 

v_lshrrev_b32 v94, 2, v1 // (op 10276) 
v_lshl_add_u32 v97, v94, 6, v37 // (op 10276) 


s_nop 0
v_readfirstlane_b32 s54, v97 // (op 10276) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -8, v1 // (op 10320) 
v_sub_u32 v94, v1, v37 // (op 10320) 

v_lshlrev_b32 v37, 4, v94 // (op 10320) 

v_lshrrev_b32 v94, 3, v1 // (op 10320) 
v_lshl_add_u32 v97, v94, 7, v37 // (op 10320) 


s_nop 0
v_readfirstlane_b32 s55, v97 // (op 10320) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -64, v1 // (op 10364) 
v_sub_u32 v94, v1, v37 // (op 10364) 

v_and_b32 v37, 4294967232, v1 // (op 10364) 
v_add_u32 v97, v37, v94 // (op 10364) 


v_and_b32 v37, -4, v1 // (op 10393) 
v_sub_u32 v94, v1, v37 // (op 10393) 

v_lshrrev_b32 v37, 2, v1 // (op 10393) 
v_lshl_add_u32 v98, v37, 3, v94 // (op 10393) 


s_mov_b32 s1, 0 // (op 3346) call()
s_and_b32 s84, s7, 4294967294 // (op 3345) 
s_cmp_lt_u32 s1, s84 // (op 3345) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoop_3345 // (op 3345) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoop_3345 if false)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoop_3345:
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:85], v[46:53], a[0:3], v33, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 88) 
s_barrier  // (op 10603) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:85], v[54:61], a[4:7], v33, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2706) 
s_add_u32 s1, s1, 2 // (op 3347) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:85], v[62:69], a[8:11], v33, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2882) 
s_lshl_b64 s[84:85], s[34:35], 5 // (op 8736) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[78:85], v[70:77], a[12:15], v33, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3058) 
v_mov_b64 v[100:101], s[84:85] // (op 8736) convert


v_mov_b64 v[102:103], v[38:39] // (op 8736) 
v_mov_b64 v[104:105], v[102:103] // (op 8736) 
s_mov_b32 m0, s53 // (op 8736) reg expression
s_mov_b32 s84, 256 // (op 8736) call()
buffer_load_dwordx4 v104, s[72:75], s84 offen offset: 0 lds // (op 8736) Load value direct to lds
v_add_co_u32 v104, s[86:87], v104, v100 // (op 8736) least significant half
v_addc_co_u32 v105, s[86:87], 0, v105, s[86:87] // (op 8736) most significant half

s_add_u32 m0, m0, 4096 // (op 8736) 
buffer_load_dwordx4 v104, s[72:75], s84 offen offset: 0 lds // (op 8736) Load value direct to lds
v_add_co_u32 v104, s[86:87], v104, v100 // (op 8736) least significant half
v_addc_co_u32 v105, s[86:87], 0, v105, s[86:87] // (op 8736) most significant half

s_add_u32 m0, m0, 4096 // (op 8736) 
buffer_load_dwordx4 v104, s[72:75], s84 offen offset: 0 lds // (op 8736) Load value direct to lds
v_add_co_u32 v104, s[86:87], v104, v100 // (op 8736) least significant half
v_addc_co_u32 v105, s[86:87], 0, v105, s[86:87] // (op 8736) most significant half

s_add_u32 m0, m0, 4096 // (op 8736) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[86:93], v[46:53], a[16:19], v36, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3553) 
buffer_load_dwordx4 v104, s[72:75], s84 offen offset: 0 lds // (op 8736) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[86:93], v[54:61], a[20:23], v36, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3507) 
v_mov_b64 v[100:101], s[44:45] // (op 25) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[86:93], v[62:69], a[24:27], v36, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3461) 
v_mov_b32 v37, v40 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[86:93], v[70:77], a[28:31], v36, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3415) 
v_mov_b32 v94, v37 // (op 25) 
buffer_load_ubyte v44, v94, s[80:83], 0 offen offset:8 // (op 25) Load value
v_add_u32 v94, v94, v100 // (op 25) 
buffer_load_ubyte v45, v94, s[80:83], 0 offen offset:8 // (op 25) Load value


s_barrier  // (op 10584) 

s_lshl_b64 s[84:85], s[30:31], 6 // (op 8731) 
v_mov_b64 v[100:101], s[84:85] // (op 8731) convert


v_mov_b64 v[104:105], v[34:35] // (op 8731) 
s_lshl_b64 s[84:85], s[30:31], 7 // (op 8731) 
s_lshl_b64 s[86:87], s[84:85], 1 // (op 8731) 

v_mov_b32 v94, s86 // (op 8731) 
v_mov_b32 v99, s87 // (op 8731) 
v_add_co_u32 v106, s[84:85], v104, v94 // (op 8731) least significant half
v_addc_co_u32 v107, s[84:85], v105, v99, s[84:85] // (op 8731) most significant half




s_add_u32 s84, 16384, s52 // (op 8731) 
s_nop 0
s_mov_b32 m0, s84 // (op 8731) reg expression
buffer_load_dwordx4 v106, s[68:71], 0 offen offset: 0 lds // (op 8731) Load value direct to lds
v_add_co_u32 v106, s[86:87], v106, v100 // (op 8731) least significant half
v_addc_co_u32 v107, s[86:87], 0, v107, s[86:87] // (op 8731) most significant half

s_add_u32 m0, m0, 4096 // (op 8731) 
buffer_load_dwordx4 v106, s[68:71], 0 offen offset: 0 lds // (op 8731) Load value direct to lds





v_mov_b32 v94, v29 // (op 17) 
s_lshl_b32 s84, s38, 3 // (op 17) 
v_add_u32 v99, v94, s84 // (op 17) 


buffer_load_ubyte v30, v99, s[76:79], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 10799) 



s_waitcnt vmcnt(13)
ds_write_b8 v98, v42 offset:49920 // (op 2585) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v98, v43 offset:49924 // (op 2585) Store local data 
s_barrier  // (op 10793) 



s_waitcnt vmcnt(9)
ds_write_b8 v97, v41 offset:50432 // (op 2584) Store local data 
s_barrier  // (op 10779) 



ds_read_b128 v[106:109], v16 offset:24576 // (op 2577) Load local data 
ds_read_b128 v[110:113], v16 offset:24640 // (op 2577) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10796) 



ds_read_u8 v99, v17 offset:49920 // (op 2581) Load local data 



ds_read_b128 v[114:117], v16 offset:28672 // (op 2711) Load local data 
ds_read_b128 v[118:121], v16 offset:28736 // (op 2711) Load local data 



ds_read_u8 v100, v17 offset:50048 // (op 2715) Load local data 



ds_read_b128 v[122:125], v16 offset:32768 // (op 2887) Load local data 
ds_read_b128 v[126:129], v16 offset:32832 // (op 2887) Load local data 



ds_read_u8 v101, v17 offset:50176 // (op 2891) Load local data 



ds_read_b128 v[130:133], v16 offset:36864 // (op 3063) Load local data 
ds_read_b128 v[134:137], v16 offset:36928 // (op 3063) Load local data 



ds_read_u8 v138, v17 offset:50304 // (op 3067) Load local data 
s_barrier  // (op 10770) 
s_barrier  // (op 11002) 



ds_read_b64_tr_b8 v[78:79], v14 offset:40960 // (op 2575) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v14 offset:41472 // (op 2575) Transpose load local data 
ds_read_b64_tr_b8 v[82:83], v14 offset:45056 // (op 2575) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v14 offset:45568 // (op 2575) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 10789) 


ds_read_u8 v33, v15 offset:50432 // (op 2579) Load local data 



ds_read_b64_tr_b8 v[86:87], v14 offset:40992 // (op 3418) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v14 offset:41504 // (op 3418) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v14 offset:45088 // (op 3418) Transpose load local data 
ds_read_b64_tr_b8 v[92:93], v14 offset:45600 // (op 3418) Transpose load local data 


ds_read_u8 v36, v15 offset:50464 // (op 3421) Load local data 
s_barrier  // (op 11028) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:85], v[106:113], a[0:3], v33, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2582) 
s_barrier  // (op 10786) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:85], v[114:121], a[4:7], v33, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2707) 
s_lshl_b64 s[84:85], s[34:35], 5 // (op 8792) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:85], v[122:129], a[8:11], v33, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2883) 
v_mov_b64 v[140:141], s[84:85] // (op 8792) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[78:85], v[130:137], a[12:15], v33, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3059) 
v_mov_b64 v[142:143], v[102:103] // (op 8792) 
s_add_u32 s84, 24576, s55 // (op 8792) 
s_mov_b32 m0, s84 // (op 8792) reg expression
s_mov_b32 s85, 384 // (op 8792) call()
buffer_load_dwordx4 v142, s[72:75], s85 offen offset: 0 lds // (op 8792) Load value direct to lds
v_add_co_u32 v142, s[86:87], v142, v140 // (op 8792) least significant half
v_addc_co_u32 v143, s[86:87], 0, v143, s[86:87] // (op 8792) most significant half

s_add_u32 m0, m0, 4096 // (op 8792) 
buffer_load_dwordx4 v142, s[72:75], s85 offen offset: 0 lds // (op 8792) Load value direct to lds
v_add_co_u32 v142, s[86:87], v142, v140 // (op 8792) least significant half
v_addc_co_u32 v143, s[86:87], 0, v143, s[86:87] // (op 8792) most significant half

s_add_u32 m0, m0, 4096 // (op 8792) 
buffer_load_dwordx4 v142, s[72:75], s85 offen offset: 0 lds // (op 8792) Load value direct to lds
v_add_co_u32 v142, s[86:87], v142, v140 // (op 8792) least significant half
v_addc_co_u32 v143, s[86:87], 0, v143, s[86:87] // (op 8792) most significant half

s_add_u32 m0, m0, 4096 // (op 8792) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[86:93], v[106:113], a[16:19], v36, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3554) 
buffer_load_dwordx4 v142, s[72:75], s85 offen offset: 0 lds // (op 8792) Load value direct to lds








v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[86:93], v[114:121], a[20:23], v36, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3508) 
v_add_co_u32 v38, vcc, 256, v38 // (op 9084) least significant half


v_addc_co_u32 v39, vcc, 0, v39, vcc // (op 9084) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[86:93], v[122:129], a[24:27], v36, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3462) 
v_mov_b64 v[102:103], s[44:45] // (op 2588) convert



v_mov_b32 v99, v37 // (op 2588) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[86:93], v[130:137], a[28:31], v36, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3416) 
buffer_load_ubyte v42, v99, s[80:83], 0 offen offset:12 // (op 2588) Load value


v_add_u32 v99, v99, v102 // (op 2588) 
buffer_load_ubyte v43, v99, s[80:83], 0 offen offset:12 // (op 2588) Load value



s_barrier  // (op 10776) 
v_add_u32 v40, 8, v40 // (op 9340) 

s_lshl_b64 s[84:85], s[30:31], 6 // (op 8787) 
v_mov_b64 v[102:103], s[84:85] // (op 8787) convert


s_lshl_b64 s[84:85], s[30:31], 7 // (op 8787) 
s_mul_i32 s87, 3, s85 // (op 8787) most significant: low of low * high
s_mul_i32 s86, 3, s84 // (op 8787) least significant
s_mul_hi_u32 s88, 3, s84 // (op 8787) most significant: high of low * low
s_add_u32 s87, s87, s88 // (op 8787) most significant: sum


v_mov_b32 v37, s86 // (op 8787) 
v_mov_b32 v99, s87 // (op 8787) 
v_add_co_u32 v100, s[84:85], v104, v37 // (op 8787) least significant half
v_addc_co_u32 v101, s[84:85], v105, v99, s[84:85] // (op 8787) most significant half




s_add_u32 s84, 40960, s54 // (op 8787) 
s_nop 0
s_mov_b32 m0, s84 // (op 8787) reg expression
buffer_load_dwordx4 v100, s[68:71], 0 offen offset: 0 lds // (op 8787) Load value direct to lds
v_add_co_u32 v100, s[86:87], v100, v102 // (op 8787) least significant half
v_addc_co_u32 v101, s[86:87], 0, v101, s[86:87] // (op 8787) most significant half

s_add_u32 m0, m0, 4096 // (op 8787) 
buffer_load_dwordx4 v100, s[68:71], 0 offen offset: 0 lds // (op 8787) Load value direct to lds






s_lshl_b32 s84, s38, 2 // (op 2587) 
s_mul_i32 s85, 3, s84 // (op 2587) 

v_add_u32 v37, v94, s85 // (op 2587) 


buffer_load_ubyte v41, v37, s[76:79], 0 offen offset: 0 // (op 2587) Load value

s_lshl_b64 s[84:85], s[30:31], 7 // (op 8956) 

s_lshl_b64 s[86:87], s[84:85], 1 // (op 8956) 

s_barrier  // (op 10644) 
s_lshl_b32 s84, s38, 3 // (op 9212) 



v_add_u32 v29, v29, s84 // (op 9212) 

v_mov_b32 v37, s86 // (op 8956) 
v_mov_b32 v94, s87 // (op 8956) 
v_add_co_u32 v34, s[84:85], v34, v37 // (op 8956) least significant half
v_addc_co_u32 v35, s[84:85], v35, v94, s[84:85] // (op 8956) most significant half




s_waitcnt vmcnt(13)
ds_write_b8 v96, v44 offset:49152 // (op 30) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v96, v45 offset:49156 // (op 30) Store local data 
s_barrier  // (op 10623) 



s_waitcnt vmcnt(9)
ds_write_b8 v95, v30 offset:49664 // (op 21) Store local data 
s_barrier  // (op 10609) 



ds_read_b128 v[46:49], v4 // (op 75) Load local data 
ds_read_b128 v[50:53], v4 offset:64 // (op 75) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10651) 



ds_read_u8 v26, v5 offset:49152 // (op 85) Load local data 



ds_read_b128 v[54:57], v4 offset:4096 // (op 2701) Load local data 
ds_read_b128 v[58:61], v4 offset:4160 // (op 2701) Load local data 



ds_read_u8 v28, v5 offset:49280 // (op 2705) Load local data 



ds_read_b128 v[62:65], v4 offset:8192 // (op 2877) Load local data 
ds_read_b128 v[66:69], v4 offset:8256 // (op 2877) Load local data 



ds_read_u8 v31, v5 offset:49408 // (op 2881) Load local data 



ds_read_b128 v[70:73], v4 offset:12288 // (op 3053) Load local data 
ds_read_b128 v[74:77], v4 offset:12352 // (op 3053) Load local data 



ds_read_u8 v32, v5 offset:49536 // (op 3057) Load local data 
s_barrier  // (op 10590) 



ds_read_b64_tr_b8 v[78:79], v0 offset:16384 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v0 offset:16896 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[82:83], v0 offset:20480 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v0 offset:20992 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 10630) 


ds_read_u8 v33, v3 offset:49664 // (op 80) Load local data 



ds_read_b64_tr_b8 v[86:87], v0 offset:16416 // (op 3410) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v0 offset:16928 // (op 3410) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v0 offset:20512 // (op 3410) Transpose load local data 
ds_read_b64_tr_b8 v[92:93], v0 offset:21024 // (op 3410) Transpose load local data 


ds_read_u8 v36, v3 offset:49696 // (op 3413) Load local data 
s_and_b32 s84, s7, 4294967294 // (op 3345) 
s_cmp_lt_u32 s1, s84 // (op 3345) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoop_3345 // (op 3345) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoop_3345 if true)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoop_3345:


















s_barrier  // (op 11010) 
s_and_b32 s52, s7, 4294967294 // (op 8860) 
v_and_b32 v26, -4, v1 // (op 8860) 
s_add_u32 s53, s0, s52 // (op 8860) 

v_sub_u32 v28, v1, v26 // (op 8860) 

v_lshlrev_b32 v26, 4, v28 // (op 8860) 

s_mul_i32 s52, s16, s8 // (op 8860) 
s_add_u32 s52, s52, s53 // (op 8860) 

v_lshrrev_b32 v28, 2, v1 // (op 8860) 
s_mul_hi_u32 s53, s52, s13 // (op 8860) 
s_sub_u32 s54, s52, s53 // (op 8860) 
s_lshr_b32 s55, s54, 1 // (op 8860) 

s_add_u32 s54, s55, s53 // (op 8860) 


s_lshr_b32 s53, s54, s11 // (op 8860) 

s_and_b32 s54, s11, 2147483648 // (op 8860) 
s_cmp_eq_i32 s54, 0 // (op 8860) 
s_mov_b32 s55, scc // (op 8860) 

s_cmp_eq_i32 s55, 0 // (op 8860) 
s_cselect_b32 s54, s52, s53 // (op 8860) 


s_mul_hi_u32 s53, s54, s12 // (op 8860) 
s_sub_u32 s55, s54, s53 // (op 8860) 
s_lshr_b32 s84, s55, 1 // (op 8860) 

s_add_u32 s55, s84, s53 // (op 8860) 


s_lshr_b32 s53, s55, s10 // (op 8860) 

s_and_b32 s55, s10, 2147483648 // (op 8860) 
s_cmp_eq_i32 s55, 0 // (op 8860) 
s_mov_b32 s84, scc // (op 8860) 

s_cmp_eq_i32 s84, 0 // (op 8860) 
s_cselect_b32 s55, s54, s53 // (op 8860) 


v_lshl_add_u32 v29, s55, 6, v26 // (op 8860) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 8860) 
v_mov_b32 v35, 0 // (op 8860) convert
v_mov_b32 v34, v29 // (op 8860) convert

s_mul_i32 s53, s54, s84 // (op 8860) 


s_sub_u32 s54, s52, s53 // (op 8860) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 8860) 


v_mul_lo_u32 v29, v26, s31 // (op 8860) most significant: low of low * high
v_mul_hi_u32 v30, v26, s30 // (op 8860) most significant: high of low * low
v_mul_lo_u32 v28, v26, s30 // (op 8860) least significant: low of low * low
v_add_u32 v29, v29, v30 // (op 8860) most significant: sum


v_add_co_u32 v44, s[52:53], v34, v28 // (op 8860) least significant half
v_addc_co_u32 v45, s[52:53], v35, v29, s[52:53] // (op 8860) most significant half



s_mul_i32 s53, s30, s3 // (op 8862) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 8862) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 8862) least significant
s_mul_hi_u32 s55, s30, s2 // (op 8862) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 8862) most significant: sum
s_add_u32 s53, s53, s55 // (op 8862) most significant: sum


s_mov_b32 s68, s32 // (op 8862) reg expression
s_mov_b32 s69, s33 // (op 8862) reg expression
s_mov_b32 s70, s52 // (op 8862) reg expression
s_mov_b32 s71, 131072 // (op 8862) call()

s_and_b32 s52, s7, 4294967294 // (op 9008) 
v_lshrrev_b32 v26, 3, v1 // (op 9008) 
s_add_u32 s53, s0, s52 // (op 9008) 

v_and_b32 v28, -8, v1 // (op 9008) 
v_sub_u32 v29, v1, v28 // (op 9008) 

s_mul_i32 s52, s16, s8 // (op 9008) 
s_add_u32 s52, s52, s53 // (op 9008) 

v_lshlrev_b32 v28, 4, v29 // (op 9008) 

s_mul_hi_u32 s53, s52, s13 // (op 9008) 
s_sub_u32 s54, s52, s53 // (op 9008) 
s_lshr_b32 s55, s54, 1 // (op 9008) 

s_add_u32 s54, s55, s53 // (op 9008) 


s_lshr_b32 s53, s54, s11 // (op 9008) 

s_and_b32 s54, s11, 2147483648 // (op 9008) 
s_cmp_eq_i32 s54, 0 // (op 9008) 
s_mov_b32 s55, scc // (op 9008) 

s_cmp_eq_i32 s55, 0 // (op 9008) 
s_cselect_b32 s54, s52, s53 // (op 9008) 


s_mul_hi_u32 s53, s54, s12 // (op 9008) 
s_sub_u32 s55, s54, s53 // (op 9008) 
s_lshr_b32 s84, s55, 1 // (op 9008) 

s_add_u32 s55, s84, s53 // (op 9008) 


s_lshr_b32 s53, s55, s10 // (op 9008) 

s_and_b32 s55, s10, 2147483648 // (op 9008) 
s_cmp_eq_i32 s55, 0 // (op 9008) 
s_mov_b32 s84, scc // (op 9008) 

s_cmp_eq_i32 s84, 0 // (op 9008) 
s_cselect_b32 s55, s54, s53 // (op 9008) 


s_mul_i32 s53, s55, s14 // (op 9008) 

s_sub_u32 s55, s54, s53 // (op 9008) 

v_lshl_add_u32 v29, s55, 7, v26 // (op 9008) 


v_mul_lo_u32 v35, v29, s35 // (op 9008) most significant: low of low * high
v_mul_hi_u32 v26, v29, s34 // (op 9008) most significant: high of low * low
v_mul_lo_u32 v34, v29, s34 // (op 9008) least significant: low of low * low
v_add_u32 v35, v35, v26 // (op 9008) most significant: sum


s_ashr_i64 s[84:85], s[2:3], 7 // (op 9008) 
s_mul_i32 s53, s54, s84 // (op 9008) 


s_sub_u32 s54, s52, s53 // (op 9008) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 9008) 


v_mov_b32 v29, 0 // (op 9008) convert
v_mov_b32 v28, v26 // (op 9008) convert

v_add_co_u32 v46, s[52:53], v28, v34 // (op 9008) least significant half
v_addc_co_u32 v47, s[52:53], v29, v35, s[52:53] // (op 9008) most significant half



s_mul_i32 s53, s34, s23 // (op 9010) most significant: low of low * high
s_mul_i32 s54, s35, s22 // (op 9010) most significant: low of high * low
s_mul_i32 s52, s34, s22 // (op 9010) least significant
s_mul_hi_u32 s55, s34, s22 // (op 9010) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9010) most significant: sum
s_add_u32 s53, s53, s55 // (op 9010) most significant: sum


s_mov_b32 s72, s36 // (op 9010) reg expression
s_mov_b32 s73, s37 // (op 9010) reg expression
s_mov_b32 s74, s52 // (op 9010) reg expression
s_mov_b32 s75, 131072 // (op 9010) call()

s_and_b32 s52, s7, 4294967294 // (op 9136) 
v_and_b32 v26, -64, v1 // (op 9136) 
s_add_u32 s53, s0, s52 // (op 9136) 

v_sub_u32 v28, v1, v26 // (op 9136) 

v_lshrrev_b32 v26, 6, v1 // (op 9136) 
s_mul_i32 s52, s16, s8 // (op 9136) 
s_add_u32 s52, s52, s53 // (op 9136) 

s_mul_hi_u32 s53, s52, s13 // (op 9136) 
s_sub_u32 s54, s52, s53 // (op 9136) 
s_lshr_b32 s55, s54, 1 // (op 9136) 

s_add_u32 s54, s55, s53 // (op 9136) 


s_lshr_b32 s53, s54, s11 // (op 9136) 

s_and_b32 s54, s11, 2147483648 // (op 9136) 
s_cmp_eq_i32 s54, 0 // (op 9136) 
s_mov_b32 s55, scc // (op 9136) 

s_cmp_eq_i32 s55, 0 // (op 9136) 
s_cselect_b32 s54, s52, s53 // (op 9136) 


s_mul_hi_u32 s53, s54, s12 // (op 9136) 
s_sub_u32 s55, s54, s53 // (op 9136) 
s_lshr_b32 s84, s55, 1 // (op 9136) 

s_add_u32 s55, s84, s53 // (op 9136) 


s_lshr_b32 s53, s55, s10 // (op 9136) 

s_and_b32 s55, s10, 2147483648 // (op 9136) 
s_cmp_eq_i32 s55, 0 // (op 9136) 
s_mov_b32 s84, scc // (op 9136) 

s_cmp_eq_i32 s84, 0 // (op 9136) 
s_cselect_b32 s55, s54, s53 // (op 9136) 


v_lshl_add_u32 v29, s55, 6, v28 // (op 9136) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 9136) 
s_mul_i32 s53, s54, s84 // (op 9136) 


s_sub_u32 s54, s52, s53 // (op 9136) 


v_lshl_add_u32 v28, s54, 2, v26 // (op 9136) 


v_mul_lo_u32 v26, v28, s38 // (op 9136) 

v_add_u32 v28, v29, v26 // (op 9136) 


s_mul_i32 s53, s38, s43 // (op 9138) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 9138) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 9138) least significant
s_mul_hi_u32 s55, s38, s42 // (op 9138) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9138) most significant: sum
s_add_u32 s53, s53, s55 // (op 9138) most significant: sum


s_mov_b32 s76, s40 // (op 9138) reg expression
s_mov_b32 s77, s41 // (op 9138) reg expression
s_mov_b32 s78, s52 // (op 9138) reg expression
s_mov_b32 s79, 131072 // (op 9138) call()

s_and_b32 s52, s7, 4294967294 // (op 9264) 
v_lshrrev_b32 v26, 2, v1 // (op 9264) 
s_add_u32 s53, s0, s52 // (op 9264) 

v_lshlrev_b32 v29, 1, v26 // (op 9264) 

v_and_b32 v26, -4, v1 // (op 9264) 
s_mul_i32 s52, s16, s8 // (op 9264) 
s_add_u32 s52, s52, s53 // (op 9264) 

v_sub_u32 v30, v1, v26 // (op 9264) 

s_mul_hi_u32 s53, s52, s13 // (op 9264) 
s_sub_u32 s54, s52, s53 // (op 9264) 
s_lshr_b32 s55, s54, 1 // (op 9264) 

s_add_u32 s54, s55, s53 // (op 9264) 


s_lshr_b32 s53, s54, s11 // (op 9264) 

s_and_b32 s54, s11, 2147483648 // (op 9264) 
s_cmp_eq_i32 s54, 0 // (op 9264) 
s_mov_b32 s55, scc // (op 9264) 

s_cmp_eq_i32 s55, 0 // (op 9264) 
s_cselect_b32 s54, s52, s53 // (op 9264) 


s_mul_hi_u32 s53, s54, s12 // (op 9264) 
s_sub_u32 s55, s54, s53 // (op 9264) 
s_lshr_b32 s84, s55, 1 // (op 9264) 

s_add_u32 s55, s84, s53 // (op 9264) 


s_lshr_b32 s53, s55, s10 // (op 9264) 

s_and_b32 s55, s10, 2147483648 // (op 9264) 
s_cmp_eq_i32 s55, 0 // (op 9264) 
s_mov_b32 s84, scc // (op 9264) 

s_cmp_eq_i32 s84, 0 // (op 9264) 
s_cselect_b32 s55, s54, s53 // (op 9264) 


s_mul_i32 s53, s55, s14 // (op 9264) 

s_sub_u32 s55, s54, s53 // (op 9264) 

v_lshl_add_u32 v26, s55, 7, v29 // (op 9264) 


v_mul_lo_u32 v29, v26, s44 // (op 9264) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 9264) 
s_mul_i32 s53, s54, s84 // (op 9264) 


s_sub_u32 s54, s52, s53 // (op 9264) 


v_lshl_add_u32 v26, s54, 2, v30 // (op 9264) 


v_add_u32 v30, v26, v29 // (op 9264) 


s_mul_i32 s53, s44, s23 // (op 9266) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 9266) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 9266) least significant
s_mul_hi_u32 s55, s44, s22 // (op 9266) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9266) most significant: sum
s_add_u32 s53, s53, s55 // (op 9266) most significant: sum


s_mov_b32 s80, s46 // (op 9266) reg expression
s_mov_b32 s81, s47 // (op 9266) reg expression
s_mov_b32 s82, s52 // (op 9266) reg expression
s_mov_b32 s83, 131072 // (op 9266) call()

v_and_b32 v26, -4, v1 // (op 9435) 
v_sub_u32 v29, v1, v26 // (op 9435) 

v_lshlrev_b32 v26, 4, v29 // (op 9435) 

v_lshrrev_b32 v29, 2, v1 // (op 9435) 
v_lshl_add_u32 v34, v29, 6, v26 // (op 9435) 


s_nop 0
v_readfirstlane_b32 s52, v34 // (op 9435) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -8, v1 // (op 9510) 
v_sub_u32 v29, v1, v26 // (op 9510) 

v_lshlrev_b32 v26, 4, v29 // (op 9510) 

v_lshrrev_b32 v29, 3, v1 // (op 9510) 
v_lshl_add_u32 v34, v29, 7, v26 // (op 9510) 


s_nop 0
v_readfirstlane_b32 s53, v34 // (op 9510) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -64, v1 // (op 9585) 
v_sub_u32 v29, v1, v26 // (op 9585) 

v_and_b32 v26, 4294967232, v1 // (op 9585) 
v_add_u32 v34, v26, v29 // (op 9585) 


v_and_b32 v26, -4, v1 // (op 9661) 
v_sub_u32 v29, v1, v26 // (op 9661) 

v_lshrrev_b32 v26, 2, v1 // (op 9661) 
v_lshl_add_u32 v35, v26, 3, v29 // (op 9661) 


s_and_b32 s54, s7, 4294967294 // (op 3351) 
s_cmp_lt_u32 s54, s7 // (op 3350) 










s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoopTail_3350 // (op 3350) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoopTail_3350 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoopTail_3350:
s_barrier  // (op 10587) 
s_add_u32 s54, s54, 1 // (op 3352) 

s_lshl_b64 s[84:85], s[30:31], 6 // (op 8774) 
v_mov_b64 v[32:33], s[84:85] // (op 8774) convert


v_mov_b64 v[36:37], v[44:45] // (op 8774) 
v_mov_b64 v[38:39], v[36:37] // (op 8774) reg expression
s_add_u32 s55, 16384, s52 // (op 8774) 
s_mov_b32 m0, s55 // (op 8774) reg expression
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 8774) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v32 // (op 8774) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 8774) most significant half

s_add_u32 m0, m0, 4096 // (op 8774) 
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 8774) Load value direct to lds




s_lshl_b64 s[84:85], s[30:31], 7 // (op 8850) 
v_mov_b32 v26, s84 // (op 8850) 
v_mov_b32 v29, s85 // (op 8850) 
v_add_co_u32 v44, s[86:87], v44, v26 // (op 8850) least significant half
v_addc_co_u32 v45, s[86:87], v45, v29, s[86:87] // (op 8850) most significant half




s_barrier  // (op 10606) 

s_lshl_b64 s[84:85], s[34:35], 5 // (op 8780) 
v_mov_b64 v[32:33], s[84:85] // (op 8780) convert


v_mov_b64 v[36:37], v[46:47] // (op 8780) 
v_mov_b64 v[38:39], v[36:37] // (op 8780) 
s_mov_b32 m0, s53 // (op 8780) reg expression
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 8780) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v32 // (op 8780) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 8780) most significant half

s_add_u32 m0, m0, 4096 // (op 8780) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 8780) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v32 // (op 8780) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 8780) most significant half

s_add_u32 m0, m0, 4096 // (op 8780) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 8780) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v32 // (op 8780) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 8780) most significant half

s_add_u32 m0, m0, 4096 // (op 8780) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 8780) Load value direct to lds



v_add_co_u32 v46, vcc, 128, v46 // (op 9000) least significant half
v_addc_co_u32 v47, vcc, 0, v47, vcc // (op 9000) most significant half


v_mov_b32 v26, v28 // (op 2521) 
v_mov_b32 v29, v26 // (op 2521) reg expression

buffer_load_ubyte v31, v29, s[76:79], 0 offen offset: 0 // (op 2521) Load value


s_lshl_b32 s55, s38, 2 // (op 9128) 
v_add_u32 v28, v28, s55 // (op 9128) 


s_barrier  // (op 10626) 
v_mov_b64 v[32:33], s[44:45] // (op 2522) convert




v_mov_b32 v26, v30 // (op 2522) 
v_mov_b32 v29, v26 // (op 2522) 
buffer_load_ubyte v36, v29, s[80:83], 0 offen offset: 0 // (op 2522) Load value
v_add_u32 v29, v29, v32 // (op 2522) 
buffer_load_ubyte v37, v29, s[80:83], 0 offen offset: 0 // (op 2522) Load value



v_add_u32 v30, 4, v30 // (op 9256) 
s_waitcnt vmcnt(2)
ds_write_b8 v34, v31 offset:49664 // (op 2518) Store local data 

s_barrier  // (op 10593) 



s_barrier  // (op 10647) 
ds_read_b64_tr_b8 v[48:49], v0 offset:16384 // (op 2509) Transpose load local data 



ds_read_b64_tr_b8 v[50:51], v0 offset:16896 // (op 2509) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:20480 // (op 2509) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:20992 // (op 2509) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v35, v36 offset:49152 // (op 2519) Store local data 
s_barrier  // (op 10613) 
s_waitcnt vmcnt(0)
ds_write_b8 v35, v37 offset:49156 // (op 2519) Store local data 




ds_read_b128 v[36:39], v4 // (op 2511) Load local data 
ds_read_b128 v[40:43], v4 offset:64 // (op 2511) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 10634) 


ds_read_u8 v26, v3 offset:49664 // (op 2513) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10654) 



ds_read_u8 v29, v5 offset:49152 // (op 2515) Load local data 



ds_read_b128 v[56:59], v4 offset:4096 // (op 2680) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[36:43], a[0:3], v26, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2516) 
ds_read_b128 v[60:63], v4 offset:4160 // (op 2680) Load local data 



ds_read_u8 v31, v5 offset:49280 // (op 2684) Load local data 



ds_read_b128 v[64:67], v4 offset:8192 // (op 2856) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:55], v[56:63], a[4:7], v26, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2685) 
ds_read_b128 v[68:71], v4 offset:8256 // (op 2856) Load local data 



ds_read_u8 v32, v5 offset:49408 // (op 2860) Load local data 



ds_read_b128 v[72:75], v4 offset:12288 // (op 3032) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:55], v[64:71], a[8:11], v26, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2861) 
ds_read_b128 v[76:79], v4 offset:12352 // (op 3032) Load local data 



ds_read_u8 v33, v5 offset:49536 // (op 3036) Load local data 



ds_read_b64_tr_b8 v[80:81], v0 offset:16416 // (op 3294) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:55], v[72:79], a[12:15], v26, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3037) 
ds_read_b64_tr_b8 v[82:83], v0 offset:16928 // (op 3294) Transpose load local data 


ds_read_b64_tr_b8 v[84:85], v0 offset:20512 // (op 3294) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v0 offset:21024 // (op 3294) Transpose load local data 


ds_read_u8 v26, v3 offset:49696 // (op 3297) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[80:87], v[36:43], a[16:19], v26, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3299) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[80:87], v[56:63], a[20:23], v26, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3330) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[80:87], v[64:71], a[24:27], v26, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3361) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:87], v[72:79], a[28:31], v26, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3392) 




s_cmp_lt_u32 s54, s7 // (op 3350) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoopTail_3350 // (op 3350) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoopTail_3350 if true)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoopTail_3350:









s_barrier  // (op 11014) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 288) 
s_cmp_lg_u32 s18, 0 // (op 314) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Send_Tile_314 // (op 314) Condition: False, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Send_Tile_314



v_add_u32 v26, 16384, v25 // (op 4224) 
buffer_store_dword a16, v26, s[64:67], 0 offen sc1 // (op 4224) Store value

v_add_u32 v26, 16896, v25 // (op 4224) 
buffer_store_dword a17, v26, s[64:67], 0 offen sc1 // (op 4224) Store value

v_add_u32 v26, 17408, v25 // (op 4224) 
buffer_store_dword a18, v26, s[64:67], 0 offen sc1 // (op 4224) Store value

v_add_u32 v26, 17920, v25 // (op 4224) 
buffer_store_dword a19, v26, s[64:67], 0 offen sc1 // (op 4224) Store value




v_add_u32 v26, 16512, v25 // (op 4226) 
buffer_store_dword a20, v26, s[64:67], 0 offen sc1 // (op 4226) Store value

v_add_u32 v26, 17024, v25 // (op 4226) 
buffer_store_dword a21, v26, s[64:67], 0 offen sc1 // (op 4226) Store value

v_add_u32 v26, 17536, v25 // (op 4226) 
buffer_store_dword a22, v26, s[64:67], 0 offen sc1 // (op 4226) Store value

v_add_u32 v26, 18048, v25 // (op 4226) 
buffer_store_dword a23, v26, s[64:67], 0 offen sc1 // (op 4226) Store value




v_add_u32 v26, 16640, v25 // (op 4228) 
buffer_store_dword a24, v26, s[64:67], 0 offen sc1 // (op 4228) Store value

v_add_u32 v26, 17152, v25 // (op 4228) 
buffer_store_dword a25, v26, s[64:67], 0 offen sc1 // (op 4228) Store value

v_add_u32 v26, 17664, v25 // (op 4228) 
buffer_store_dword a26, v26, s[64:67], 0 offen sc1 // (op 4228) Store value

v_add_u32 v26, 18176, v25 // (op 4228) 
buffer_store_dword a27, v26, s[64:67], 0 offen sc1 // (op 4228) Store value




v_add_u32 v26, 16768, v25 // (op 4230) 
buffer_store_dword a28, v26, s[64:67], 0 offen sc1 // (op 4230) Store value

v_add_u32 v26, 17280, v25 // (op 4230) 
buffer_store_dword a29, v26, s[64:67], 0 offen sc1 // (op 4230) Store value

v_add_u32 v26, 17792, v25 // (op 4230) 
buffer_store_dword a30, v26, s[64:67], 0 offen sc1 // (op 4230) Store value

v_add_u32 v26, 18304, v25 // (op 4230) 
buffer_store_dword a31, v26, s[64:67], 0 offen sc1 // (op 4230) Store value




buffer_store_dword a0, v25, s[64:67], 0 offen sc1 // (op 311) Store value
buffer_store_dword a1, v25, s[64:67], 0 offen offset:512 sc1 // (op 311) Store value
buffer_store_dword a2, v25, s[64:67], 0 offen offset:1024 sc1 // (op 311) Store value
buffer_store_dword a3, v25, s[64:67], 0 offen offset:1536 sc1 // (op 311) Store value



buffer_store_dword a4, v25, s[64:67], 0 offen offset:128 sc1 // (op 4198) Store value
buffer_store_dword a5, v25, s[64:67], 0 offen offset:640 sc1 // (op 4198) Store value
buffer_store_dword a6, v25, s[64:67], 0 offen offset:1152 sc1 // (op 4198) Store value
buffer_store_dword a7, v25, s[64:67], 0 offen offset:1664 sc1 // (op 4198) Store value



buffer_store_dword a8, v25, s[64:67], 0 offen offset:256 sc1 // (op 4199) Store value
buffer_store_dword a9, v25, s[64:67], 0 offen offset:768 sc1 // (op 4199) Store value
buffer_store_dword a10, v25, s[64:67], 0 offen offset:1280 sc1 // (op 4199) Store value
buffer_store_dword a11, v25, s[64:67], 0 offen offset:1792 sc1 // (op 4199) Store value



buffer_store_dword a12, v25, s[64:67], 0 offen offset:384 sc1 // (op 4200) Store value
buffer_store_dword a13, v25, s[64:67], 0 offen offset:896 sc1 // (op 4200) Store value
buffer_store_dword a14, v25, s[64:67], 0 offen offset:1408 sc1 // (op 4200) Store value
buffer_store_dword a15, v25, s[64:67], 0 offen offset:1920 sc1 // (op 4200) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 315) 
v_cmp_eq_i32 vcc, v1, 0 // (op 319) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_319 // (op 319) Condition: False, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_319
s_mov_b32 s52, 1 // (op 317) call()
s_mov_b32 s55, 0 // (op 318) convert
s_mov_b32 s54, s16 // (op 318) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 318) 
s_add_u32 s84, s48, s54 // (op 318) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 318) End of Int64 add, unlocking SCC
s_store_dword s52, s[84:85], 0 glc // (op 318) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_319 // (op 319) Condition: Done, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_319
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_319:
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_319:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Send_Tile_314 // (op 314) Condition: Done, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Send_Tile_314
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Send_Tile_314:
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Send_Tile_314:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 330) 
s_mov_b32 s52, scc // (op 330) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 330) 
s_mov_b32 s53, s54 // (op 330) convert

s_sub_u32 s54, s53, 1 // (op 330) 

s_cmp_lt_u32 s19, s54 // (op 330) 
s_mov_b32 s53, scc // (op 330) 

s_and_b32 s54, s52, s53 // (op 330) 
s_cmp_lg_u32 s54, 0 // (op 330) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Receive_Tile_330 // (op 330) Condition: False, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Receive_Tile_330
s_mov_b32 s52, 0 // (op 296) call()
s_mul_i32 s53, s52, s8 // (op 295) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 295) 
s_mov_b32 s84, s54 // (op 295) convert

s_sub_u32 s54, s84, s19 // (op 295) 

s_sub_u32 s55, s54, 1 // (op 295) 

s_cmp_lt_u32 s53, s55 // (op 295) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_295 // (op 295) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_295 if false)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_ReceiveTileLoop_295:
s_add_u32 s53, s16, 1 // (op 10115) 
v_lshrrev_b32 v20, 5, v1 // (op 10115) 
s_add_u32 s54, s53, s52 // (op 10115) 
s_lshl_b32 s54, s54, 6 // (op 10115) 

v_lshlrev_b32 v21, 2, v20 // (op 10115) 

v_add_lshl_u32 v20, s54, v21, 7 // (op 10115) 


v_and_b32 v21, -32, v1 // (op 10115) 
v_sub_u32 v23, v1, v21 // (op 10115) 

v_add_lshl_u32 v21, v20, v23, 2 // (op 10115) 


s_lshl_b32 s53, s15, 15 // (op 10117) 
s_mov_b32 s64, s28 // (op 10117) reg expression
s_mov_b32 s65, s29 // (op 10117) reg expression
s_mov_b32 s66, s53 // (op 10117) reg expression
s_mov_b32 s67, 131072 // (op 10117) call()

RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_334:
s_add_u32 s53, s16, 1 // (op 332) 
s_add_u32 s54, s53, s52 // (op 332) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 332) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 332) 

s_add_u32 s84, s48, s54 // (op 332) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 332) End of Int64 add, unlocking SCC
s_load_dword s53, s[84:85], 0 glc // (op 332) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 334) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_334 // (op 334) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_334 if true)
s_barrier  // (op 338) 
v_cmp_eq_i32 vcc, v1, 0 // (op 337) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_337 // (op 337) Condition: False, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_337
s_mov_b32 s53, 0 // (op 335) call()
s_add_u32 s54, s16, 1 // (op 336) 
s_add_u32 s84, s54, s52 // (op 336) Start of Int64 add, locking SCC
s_addc_u32 s85, 0, 0 // (op 336) End of Int64 add, unlocking SCC
s_lshl_b64 s[84:85], s[84:85], 2 // (op 336) 

s_add_u32 s54, s48, s84 // (op 336) Start of Int64 add, locking SCC
s_addc_u32 s55, s49, s85 // (op 336) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 336) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_337 // (op 337) Condition: Done, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_337
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_337:
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_337:




buffer_load_dword v28, v21, s[64:67], 0 offen // (op 349) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:512 // (op 349) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:1024 // (op 349) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:1536 // (op 349) Load value
v_accvgpr_read v32, a0 // (op 352) reg expression
v_accvgpr_read v33, a1 // (op 352) reg expression
v_accvgpr_read v34, a2 // (op 352) reg expression
v_accvgpr_read v35, a3 // (op 352) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 354) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 354) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 354) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 354) 

v_accvgpr_write a0, v32 // (op 356) reg expression
v_accvgpr_write a1, v33 // (op 356) reg expression
v_accvgpr_write a2, v34 // (op 356) reg expression
v_accvgpr_write a3, v35 // (op 356) reg expression




buffer_load_dword v28, v21, s[64:67], 0 offen offset:128 // (op 4271) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:640 // (op 4271) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:1152 // (op 4271) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:1664 // (op 4271) Load value
v_accvgpr_read v32, a4 // (op 4272) reg expression
v_accvgpr_read v33, a5 // (op 4272) reg expression
v_accvgpr_read v34, a6 // (op 4272) reg expression
v_accvgpr_read v35, a7 // (op 4272) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4273) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4273) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4273) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4273) 

v_accvgpr_write a4, v32 // (op 4274) reg expression
v_accvgpr_write a5, v33 // (op 4274) reg expression
v_accvgpr_write a6, v34 // (op 4274) reg expression
v_accvgpr_write a7, v35 // (op 4274) reg expression




buffer_load_dword v28, v21, s[64:67], 0 offen offset:256 // (op 4278) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:768 // (op 4278) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:1280 // (op 4278) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:1792 // (op 4278) Load value
v_accvgpr_read v32, a8 // (op 4279) reg expression
v_accvgpr_read v33, a9 // (op 4279) reg expression
v_accvgpr_read v34, a10 // (op 4279) reg expression
v_accvgpr_read v35, a11 // (op 4279) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4280) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4280) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4280) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4280) 

v_accvgpr_write a8, v32 // (op 4281) reg expression
v_accvgpr_write a9, v33 // (op 4281) reg expression
v_accvgpr_write a10, v34 // (op 4281) reg expression
v_accvgpr_write a11, v35 // (op 4281) reg expression




buffer_load_dword v28, v21, s[64:67], 0 offen offset:384 // (op 4285) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:896 // (op 4285) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:1408 // (op 4285) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:1920 // (op 4285) Load value
v_accvgpr_read v32, a12 // (op 4286) reg expression
v_accvgpr_read v33, a13 // (op 4286) reg expression
v_accvgpr_read v34, a14 // (op 4286) reg expression
v_accvgpr_read v35, a15 // (op 4286) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4287) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4287) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4287) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4287) 

v_accvgpr_write a12, v32 // (op 4288) reg expression
v_accvgpr_write a13, v33 // (op 4288) reg expression
v_accvgpr_write a14, v34 // (op 4288) reg expression
v_accvgpr_write a15, v35 // (op 4288) reg expression




v_mov_b32 v20, 16384 // (op 4322) 
v_add_i32 v23, v21, v20 // (op 4322) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4322) Load value

v_mov_b32 v20, 16896 // (op 4322) 
v_add_i32 v23, v21, v20 // (op 4322) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4322) Load value

v_mov_b32 v20, 17408 // (op 4322) 
v_add_i32 v23, v21, v20 // (op 4322) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4322) Load value

v_mov_b32 v20, 17920 // (op 4322) 
v_add_i32 v23, v21, v20 // (op 4322) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4322) Load value

v_accvgpr_read v32, a16 // (op 4319) reg expression
v_accvgpr_read v33, a17 // (op 4319) reg expression
v_accvgpr_read v34, a18 // (op 4319) reg expression
v_accvgpr_read v35, a19 // (op 4319) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4320) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4320) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4320) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4320) 

v_accvgpr_write a16, v32 // (op 4321) reg expression
v_accvgpr_write a17, v33 // (op 4321) reg expression
v_accvgpr_write a18, v34 // (op 4321) reg expression
v_accvgpr_write a19, v35 // (op 4321) reg expression




v_mov_b32 v20, 16512 // (op 4327) 
v_add_i32 v23, v21, v20 // (op 4327) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4327) Load value

v_mov_b32 v20, 17024 // (op 4327) 
v_add_i32 v23, v21, v20 // (op 4327) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4327) Load value

v_mov_b32 v20, 17536 // (op 4327) 
v_add_i32 v23, v21, v20 // (op 4327) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4327) Load value

v_mov_b32 v20, 18048 // (op 4327) 
v_add_i32 v23, v21, v20 // (op 4327) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4327) Load value

v_accvgpr_read v32, a20 // (op 4324) reg expression
v_accvgpr_read v33, a21 // (op 4324) reg expression
v_accvgpr_read v34, a22 // (op 4324) reg expression
v_accvgpr_read v35, a23 // (op 4324) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4325) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4325) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4325) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4325) 

v_accvgpr_write a20, v32 // (op 4326) reg expression
v_accvgpr_write a21, v33 // (op 4326) reg expression
v_accvgpr_write a22, v34 // (op 4326) reg expression
v_accvgpr_write a23, v35 // (op 4326) reg expression




v_mov_b32 v20, 16640 // (op 4332) 
v_add_i32 v23, v21, v20 // (op 4332) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4332) Load value

v_mov_b32 v20, 17152 // (op 4332) 
v_add_i32 v23, v21, v20 // (op 4332) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4332) Load value

v_mov_b32 v20, 17664 // (op 4332) 
v_add_i32 v23, v21, v20 // (op 4332) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4332) Load value

v_mov_b32 v20, 18176 // (op 4332) 
v_add_i32 v23, v21, v20 // (op 4332) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4332) Load value

v_accvgpr_read v32, a24 // (op 4329) reg expression
v_accvgpr_read v33, a25 // (op 4329) reg expression
v_accvgpr_read v34, a26 // (op 4329) reg expression
v_accvgpr_read v35, a27 // (op 4329) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4330) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4330) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4330) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4330) 

v_accvgpr_write a24, v32 // (op 4331) reg expression
v_accvgpr_write a25, v33 // (op 4331) reg expression
v_accvgpr_write a26, v34 // (op 4331) reg expression
v_accvgpr_write a27, v35 // (op 4331) reg expression




v_mov_b32 v20, 16768 // (op 4337) 
v_add_i32 v23, v21, v20 // (op 4337) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4337) Load value

v_mov_b32 v20, 17280 // (op 4337) 
v_add_i32 v23, v21, v20 // (op 4337) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4337) Load value

v_mov_b32 v20, 17792 // (op 4337) 
v_add_i32 v23, v21, v20 // (op 4337) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4337) Load value

v_mov_b32 v20, 18304 // (op 4337) 
v_add_i32 v23, v21, v20 // (op 4337) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4337) Load value

v_accvgpr_read v32, a28 // (op 4334) reg expression
v_accvgpr_read v33, a29 // (op 4334) reg expression
v_accvgpr_read v34, a30 // (op 4334) reg expression
v_accvgpr_read v35, a31 // (op 4334) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4335) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4335) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4335) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4335) 

v_accvgpr_write a28, v32 // (op 4336) reg expression
v_accvgpr_write a29, v33 // (op 4336) reg expression
v_accvgpr_write a30, v34 // (op 4336) reg expression
v_accvgpr_write a31, v35 // (op 4336) reg expression

s_add_u32 s52, s52, 1 // (op 297) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 295) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 295) 
s_mov_b32 s84, s54 // (op 295) convert

s_sub_u32 s54, s84, s19 // (op 295) 

s_sub_u32 s55, s54, 1 // (op 295) 

s_cmp_lt_u32 s53, s55 // (op 295) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_ReceiveTileLoop_295 // (op 295) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_ReceiveTileLoop_295 if true)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_295:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Receive_Tile_330 // (op 330) Condition: Done, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Receive_Tile_330
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Receive_Tile_330:
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Receive_Tile_330:

s_cmp_eq_i32 s18, 0 // (op 373) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_373 // (op 373) Condition: False, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_373
v_cmp_eq_f32 vcc, 0.00000, v18 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_0_beta_133
v_accvgpr_read v28, a0 // (op 50) 
v_accvgpr_read v29, a1 // (op 50) 
v_accvgpr_read v30, a2 // (op 50) 
v_accvgpr_read v31, a3 // (op 50) 
v_mul_f32 v32, v19, v28 // (op 50) 
v_mul_f32 v33, v19, v29 // (op 50) 
v_mul_f32 v34, v19, v30 // (op 50) 
v_mul_f32 v35, v19, v31 // (op 50) 



v_mov_b32 v20, v27 // (op 56) reg expression

buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 56) Store value


v_accvgpr_read v28, a4 // (op 4391) 
v_accvgpr_read v29, a5 // (op 4391) 
v_accvgpr_read v30, a6 // (op 4391) 
v_accvgpr_read v31, a7 // (op 4391) 
v_mul_f32 v32, v19, v28 // (op 4391) 
v_mul_f32 v33, v19, v29 // (op 4391) 
v_mul_f32 v34, v19, v30 // (op 4391) 
v_mul_f32 v35, v19, v31 // (op 4391) 



s_lshl_b32 s52, s24, 7 // (op 4393) 
v_add_u32 v20, v27, s52 // (op 4393) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 4393) Store value


v_accvgpr_read v28, a8 // (op 4403) 
v_accvgpr_read v29, a9 // (op 4403) 
v_accvgpr_read v30, a10 // (op 4403) 
v_accvgpr_read v31, a11 // (op 4403) 
v_mul_f32 v32, v19, v28 // (op 4403) 
v_mul_f32 v33, v19, v29 // (op 4403) 
v_mul_f32 v34, v19, v30 // (op 4403) 
v_mul_f32 v35, v19, v31 // (op 4403) 



s_lshl_b32 s52, s24, 8 // (op 4405) 
v_add_u32 v20, v27, s52 // (op 4405) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 4405) Store value


v_accvgpr_read v28, a12 // (op 4415) 
v_accvgpr_read v29, a13 // (op 4415) 
v_accvgpr_read v30, a14 // (op 4415) 
v_accvgpr_read v31, a15 // (op 4415) 
v_mul_f32 v32, v19, v28 // (op 4415) 
v_mul_f32 v33, v19, v29 // (op 4415) 
v_mul_f32 v34, v19, v30 // (op 4415) 
v_mul_f32 v35, v19, v31 // (op 4415) 



s_lshl_b32 s52, s24, 7 // (op 4417) 
s_mul_i32 s53, 3, s52 // (op 4417) 

v_add_u32 v20, v27, s53 // (op 4417) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 4417) Store value


v_accvgpr_read v28, a16 // (op 4449) 
v_accvgpr_read v29, a17 // (op 4449) 
v_accvgpr_read v30, a18 // (op 4449) 
v_accvgpr_read v31, a19 // (op 4449) 
v_mul_f32 v32, v19, v28 // (op 4449) 
v_mul_f32 v33, v19, v29 // (op 4449) 
v_mul_f32 v34, v19, v30 // (op 4449) 
v_mul_f32 v35, v19, v31 // (op 4449) 



v_add_u32 v20, 128, v27 // (op 4452) 

buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 4452) Store value


v_accvgpr_read v28, a20 // (op 4462) 
v_accvgpr_read v29, a21 // (op 4462) 
v_accvgpr_read v30, a22 // (op 4462) 
v_accvgpr_read v31, a23 // (op 4462) 
v_mul_f32 v32, v19, v28 // (op 4462) 
v_mul_f32 v33, v19, v29 // (op 4462) 
v_mul_f32 v34, v19, v30 // (op 4462) 
v_mul_f32 v35, v19, v31 // (op 4462) 



s_lshl_b32 s52, s24, 7 // (op 4454) 
s_add_u32 s52, s52, 128 // (op 4454) 
v_add_u32 v20, v27, s52 // (op 4454) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 4454) Store value


v_accvgpr_read v28, a24 // (op 4467) 
v_accvgpr_read v29, a25 // (op 4467) 
v_accvgpr_read v30, a26 // (op 4467) 
v_accvgpr_read v31, a27 // (op 4467) 
v_mul_f32 v32, v19, v28 // (op 4467) 
v_mul_f32 v33, v19, v29 // (op 4467) 
v_mul_f32 v34, v19, v30 // (op 4467) 
v_mul_f32 v35, v19, v31 // (op 4467) 



s_lshl_b32 s52, s24, 8 // (op 4456) 
s_add_u32 s52, s52, 128 // (op 4456) 
v_add_u32 v20, v27, s52 // (op 4456) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 4456) Store value


v_accvgpr_read v28, a28 // (op 4472) 
v_accvgpr_read v29, a29 // (op 4472) 
v_accvgpr_read v30, a30 // (op 4472) 
v_accvgpr_read v31, a31 // (op 4472) 
v_mul_f32 v32, v19, v28 // (op 4472) 
v_mul_f32 v33, v19, v29 // (op 4472) 
v_mul_f32 v34, v19, v30 // (op 4472) 
v_mul_f32 v35, v19, v31 // (op 4472) 



s_lshl_b32 s52, s24, 7 // (op 4458) 
s_mul_i32 s53, 3, s52 // (op 4458) 
s_add_u32 s53, s53, 128 // (op 4458) 

v_add_u32 v20, v27, s53 // (op 4458) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 4458) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_0_beta_133
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_0_beta_133:


v_add_u32 v20, 128, v22 // (op 4630) 

buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 4630) Load value

v_accvgpr_read v32, a16 // (op 4618) 
v_accvgpr_read v33, a17 // (op 4618) 
v_accvgpr_read v34, a18 // (op 4618) 
v_accvgpr_read v35, a19 // (op 4618) 
v_mul_f32 v36, v19, v32 // (op 4618) 
v_mul_f32 v37, v19, v33 // (op 4618) 
v_mul_f32 v38, v19, v34 // (op 4618) 
v_mul_f32 v39, v19, v35 // (op 4618) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 4629) 
v_mul_f32 v33, v18, v29 // (op 4629) 
v_mul_f32 v34, v18, v30 // (op 4629) 
v_mul_f32 v35, v18, v31 // (op 4629) 

v_add_f32 v28, v32, v36 // (op 4619) 
v_add_f32 v29, v33, v37 // (op 4619) 
v_add_f32 v30, v34, v38 // (op 4619) 
v_add_f32 v31, v35, v39 // (op 4619) 




v_add_u32 v20, 128, v27 // (op 4621) 

buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 4621) Store value




s_lshl_b32 s52, s4, 7 // (op 4634) 
s_add_u32 s52, s52, 128 // (op 4634) 
v_add_u32 v20, v22, s52 // (op 4634) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 4634) Load value

v_accvgpr_read v32, a20 // (op 4644) 
v_accvgpr_read v33, a21 // (op 4644) 
v_accvgpr_read v34, a22 // (op 4644) 
v_accvgpr_read v35, a23 // (op 4644) 
v_mul_f32 v36, v19, v32 // (op 4644) 
v_mul_f32 v37, v19, v33 // (op 4644) 
v_mul_f32 v38, v19, v34 // (op 4644) 
v_mul_f32 v39, v19, v35 // (op 4644) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 4632) 
v_mul_f32 v33, v18, v29 // (op 4632) 
v_mul_f32 v34, v18, v30 // (op 4632) 
v_mul_f32 v35, v18, v31 // (op 4632) 

v_add_f32 v28, v32, v36 // (op 4633) 
v_add_f32 v29, v33, v37 // (op 4633) 
v_add_f32 v30, v34, v38 // (op 4633) 
v_add_f32 v31, v35, v39 // (op 4633) 




s_lshl_b32 s52, s24, 7 // (op 4623) 
s_add_u32 s52, s52, 128 // (op 4623) 
v_add_u32 v20, v27, s52 // (op 4623) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 4623) Store value




s_lshl_b32 s52, s4, 8 // (op 4638) 
s_add_u32 s52, s52, 128 // (op 4638) 
v_add_u32 v20, v22, s52 // (op 4638) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 4638) Load value

v_accvgpr_read v32, a24 // (op 4646) 
v_accvgpr_read v33, a25 // (op 4646) 
v_accvgpr_read v34, a26 // (op 4646) 
v_accvgpr_read v35, a27 // (op 4646) 
v_mul_f32 v36, v19, v32 // (op 4646) 
v_mul_f32 v37, v19, v33 // (op 4646) 
v_mul_f32 v38, v19, v34 // (op 4646) 
v_mul_f32 v39, v19, v35 // (op 4646) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 4636) 
v_mul_f32 v33, v18, v29 // (op 4636) 
v_mul_f32 v34, v18, v30 // (op 4636) 
v_mul_f32 v35, v18, v31 // (op 4636) 

v_add_f32 v28, v32, v36 // (op 4637) 
v_add_f32 v29, v33, v37 // (op 4637) 
v_add_f32 v30, v34, v38 // (op 4637) 
v_add_f32 v31, v35, v39 // (op 4637) 




s_lshl_b32 s52, s24, 8 // (op 4625) 
s_add_u32 s52, s52, 128 // (op 4625) 
v_add_u32 v20, v27, s52 // (op 4625) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 4625) Store value




s_lshl_b32 s52, s4, 7 // (op 4642) 
s_mul_i32 s53, 3, s52 // (op 4642) 
s_add_u32 s53, s53, 128 // (op 4642) 

v_add_u32 v20, v22, s53 // (op 4642) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 4642) Load value

v_accvgpr_read v32, a28 // (op 4648) 
v_accvgpr_read v33, a29 // (op 4648) 
v_accvgpr_read v34, a30 // (op 4648) 
v_accvgpr_read v35, a31 // (op 4648) 
v_mul_f32 v36, v19, v32 // (op 4648) 
v_mul_f32 v37, v19, v33 // (op 4648) 
v_mul_f32 v38, v19, v34 // (op 4648) 
v_mul_f32 v39, v19, v35 // (op 4648) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 4640) 
v_mul_f32 v33, v18, v29 // (op 4640) 
v_mul_f32 v34, v18, v30 // (op 4640) 
v_mul_f32 v35, v18, v31 // (op 4640) 

v_add_f32 v28, v32, v36 // (op 4641) 
v_add_f32 v29, v33, v37 // (op 4641) 
v_add_f32 v30, v34, v38 // (op 4641) 
v_add_f32 v31, v35, v39 // (op 4641) 




s_lshl_b32 s52, s24, 7 // (op 4627) 
s_mul_i32 s53, 3, s52 // (op 4627) 
s_add_u32 s53, s53, 128 // (op 4627) 

v_add_u32 v20, v27, s53 // (op 4627) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 4627) Store value




v_mov_b32 v20, v22 // (op 145) reg expression

s_nop 0
buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 145) Load value

v_accvgpr_read v32, a0 // (op 142) 
v_accvgpr_read v33, a1 // (op 142) 
v_accvgpr_read v34, a2 // (op 142) 
v_accvgpr_read v35, a3 // (op 142) 
v_mul_f32 v36, v19, v32 // (op 142) 
v_mul_f32 v37, v19, v33 // (op 142) 
v_mul_f32 v38, v19, v34 // (op 142) 
v_mul_f32 v39, v19, v35 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 146) 
v_mul_f32 v33, v18, v29 // (op 146) 
v_mul_f32 v34, v18, v30 // (op 146) 
v_mul_f32 v35, v18, v31 // (op 146) 

v_add_f32 v28, v32, v36 // (op 143) 
v_add_f32 v29, v33, v37 // (op 143) 
v_add_f32 v30, v34, v38 // (op 143) 
v_add_f32 v31, v35, v39 // (op 143) 




v_mov_b32 v20, v27 // (op 144) reg expression

buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 144) Store value




s_lshl_b32 s52, s4, 7 // (op 4544) 
v_add_u32 v20, v22, s52 // (op 4544) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 4544) Load value

v_accvgpr_read v32, a4 // (op 4541) 
v_accvgpr_read v33, a5 // (op 4541) 
v_accvgpr_read v34, a6 // (op 4541) 
v_accvgpr_read v35, a7 // (op 4541) 
v_mul_f32 v36, v19, v32 // (op 4541) 
v_mul_f32 v37, v19, v33 // (op 4541) 
v_mul_f32 v38, v19, v34 // (op 4541) 
v_mul_f32 v39, v19, v35 // (op 4541) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 4545) 
v_mul_f32 v33, v18, v29 // (op 4545) 
v_mul_f32 v34, v18, v30 // (op 4545) 
v_mul_f32 v35, v18, v31 // (op 4545) 

v_add_f32 v28, v32, v36 // (op 4542) 
v_add_f32 v29, v33, v37 // (op 4542) 
v_add_f32 v30, v34, v38 // (op 4542) 
v_add_f32 v31, v35, v39 // (op 4542) 




s_lshl_b32 s52, s24, 7 // (op 4543) 
v_add_u32 v20, v27, s52 // (op 4543) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 4543) Store value




s_lshl_b32 s52, s4, 8 // (op 4556) 
v_add_u32 v20, v22, s52 // (op 4556) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 4556) Load value

v_accvgpr_read v32, a8 // (op 4553) 
v_accvgpr_read v33, a9 // (op 4553) 
v_accvgpr_read v34, a10 // (op 4553) 
v_accvgpr_read v35, a11 // (op 4553) 
v_mul_f32 v36, v19, v32 // (op 4553) 
v_mul_f32 v37, v19, v33 // (op 4553) 
v_mul_f32 v38, v19, v34 // (op 4553) 
v_mul_f32 v39, v19, v35 // (op 4553) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 4557) 
v_mul_f32 v33, v18, v29 // (op 4557) 
v_mul_f32 v34, v18, v30 // (op 4557) 
v_mul_f32 v35, v18, v31 // (op 4557) 

v_add_f32 v28, v32, v36 // (op 4554) 
v_add_f32 v29, v33, v37 // (op 4554) 
v_add_f32 v30, v34, v38 // (op 4554) 
v_add_f32 v31, v35, v39 // (op 4554) 




s_lshl_b32 s52, s24, 8 // (op 4555) 
v_add_u32 v20, v27, s52 // (op 4555) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 4555) Store value




s_lshl_b32 s52, s4, 7 // (op 4568) 
s_mul_i32 s53, 3, s52 // (op 4568) 

v_add_u32 v20, v22, s53 // (op 4568) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 4568) Load value

v_accvgpr_read v32, a12 // (op 4565) 
v_accvgpr_read v33, a13 // (op 4565) 
v_accvgpr_read v34, a14 // (op 4565) 
v_accvgpr_read v35, a15 // (op 4565) 
v_mul_f32 v36, v19, v32 // (op 4565) 
v_mul_f32 v37, v19, v33 // (op 4565) 
v_mul_f32 v38, v19, v34 // (op 4565) 
v_mul_f32 v39, v19, v35 // (op 4565) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 4569) 
v_mul_f32 v33, v18, v29 // (op 4569) 
v_mul_f32 v34, v18, v30 // (op 4569) 
v_mul_f32 v35, v18, v31 // (op 4569) 

v_add_f32 v28, v32, v36 // (op 4566) 
v_add_f32 v29, v33, v37 // (op 4566) 
v_add_f32 v30, v34, v38 // (op 4566) 
v_add_f32 v31, v35, v39 // (op 4566) 




s_lshl_b32 s52, s24, 7 // (op 4567) 
s_mul_i32 s53, 3, s52 // (op 4567) 

v_add_u32 v20, v27, s53 // (op 4567) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 4567) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_0_beta_133:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_373 // (op 373) Condition: Done, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_373
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_373:
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_373:


s_mul_i32 s52, s8, s16 // (op 286) 
s_cmp_gt_u32 s9, s52 // (op 286) 
s_mov_b32 s53, scc // (op 286) 
s_sub_u32 s54, s9, s52 // (op 286) 

s_cmp_eq_i32 s53, 0 // (op 286) 
s_cselect_b32 s52, 0, s54 // (op 286) 


s_cmp_lt_u32 s8, s52 // (op 286) 
s_mov_b32 s53, scc // (op 286) 
s_cmp_eq_i32 s53, 0 // (op 286) 
s_cselect_b32 s54, s52, s8 // (op 286) 


s_cmp_lt_u32 s0, s54 // (op 286) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_SKStreamTileLoop_286 // (op 286) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_SKStreamTileLoop_286 if true)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_286:

















s_mov_b32 s0, 0 // (op 427) call()
s_mov_b32 s1, 0 // (op 428) call()
s_mov_b32 s0, 0 // (op 442) call()
s_mul_i32 s15, s50, s16 // (op 441) 
s_cmp_gt_u32 s51, s15 // (op 441) 
s_mov_b32 s6, scc // (op 441) 
s_sub_u32 s7, s51, s15 // (op 441) 

s_cmp_eq_i32 s6, 0 // (op 441) 
s_cselect_b32 s8, 0, s7 // (op 441) 


s_cmp_lt_u32 s50, s8 // (op 441) 
s_mov_b32 s15, scc // (op 441) 
s_cmp_eq_i32 s15, 0 // (op 441) 
s_cselect_b32 s6, s8, s50 // (op 441) 


s_cmp_lt_u32 s0, s6 // (op 441) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_441 // (op 441) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_441 if false)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_DPStreamTileLoop_441:
v_lshrrev_b32 v0, 7, v1 // (op 9405) 
s_mul_i32 s15, s16, s50 // (op 9405) 
s_add_u32 s15, s15, s0 // (op 9405) 
v_and_b32 v3, 63, v1 // (op 9405) 
s_add_u32 s6, s15, s9 // (op 9405) 

v_lshrrev_b32 v4, 4, v3 // (op 9405) 
s_mul_hi_u32 s15, s6, s13 // (op 9405) 
v_lshlrev_b32 v5, 2, v4 // (op 9405) 

s_sub_u32 s7, s6, s15 // (op 9405) 
v_lshl_add_u32 v4, v0, 4, v5 // (op 9405) 


s_lshr_b32 s8, s7, 1 // (op 9405) 

v_lshrrev_b32 v0, 2, v3 // (op 9405) 

s_add_u32 s7, s8, s15 // (op 9405) 


v_and_b32 v3, 3, v0 // (op 9405) 

s_lshr_b32 s8, s7, s11 // (op 9405) 

v_and_b32 v0, 3, v1 // (op 9405) 
s_and_b32 s7, s11, 2147483648 // (op 9405) 
v_lshl_add_u32 v5, v3, 2, v0 // (op 9405) 


s_cmp_eq_i32 s7, 0 // (op 9405) 
s_mov_b32 s15, scc // (op 9405) 

v_lshrrev_b32 v0, 6, v1 // (op 9405) 
s_cmp_eq_i32 s15, 0 // (op 9405) 
s_cselect_b32 s7, s6, s8 // (op 9405) 



v_and_b32 v3, 1, v0 // (op 9405) 

s_mul_hi_u32 s6, s7, s12 // (op 9405) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 9405) 


s_sub_u32 s8, s7, s6 // (op 9405) 
s_lshr_b32 s15, s8, 1 // (op 9405) 

s_add_u32 s8, s15, s6 // (op 9405) 


s_lshr_b32 s6, s8, s10 // (op 9405) 

s_and_b32 s8, s10, 2147483648 // (op 9405) 
s_cmp_eq_i32 s8, 0 // (op 9405) 
s_mov_b32 s15, scc // (op 9405) 

s_cmp_eq_i32 s15, 0 // (op 9405) 
s_cselect_b32 s8, s7, s6 // (op 9405) 


s_mul_i32 s6, s8, s14 // (op 9405) 
v_lshl_add_u32 v3, s8, 6, v4 // (op 9405) 


s_sub_u32 s8, s7, s6 // (op 9405) 


v_lshl_add_u32 v4, s8, 7, v0 // (op 9405) 


v_mul_lo_u32 v0, v4, s4 // (op 9405) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 9405) 


s_mul_i32 s19, s4, s23 // (op 9407) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 9407) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 9407) least significant
s_mul_hi_u32 s6, s4, s22 // (op 9407) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9407) most significant: sum
s_add_u32 s19, s19, s6 // (op 9407) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 9407) 

s_mov_b32 s56, s20 // (op 9407) reg expression
s_mov_b32 s57, s21 // (op 9407) reg expression
s_mov_b32 s58, s28 // (op 9407) reg expression
s_mov_b32 s59, 131072 // (op 9407) call()

v_lshrrev_b32 v0, 7, v1 // (op 9759) 
s_mul_i32 s15, s16, s50 // (op 9759) 
s_add_u32 s15, s15, s0 // (op 9759) 
v_and_b32 v3, 63, v1 // (op 9759) 
s_add_u32 s6, s15, s9 // (op 9759) 

v_lshrrev_b32 v5, 4, v3 // (op 9759) 
s_mul_hi_u32 s15, s6, s13 // (op 9759) 
v_lshlrev_b32 v14, 2, v5 // (op 9759) 

s_sub_u32 s7, s6, s15 // (op 9759) 
v_lshl_add_u32 v5, v0, 4, v14 // (op 9759) 


s_lshr_b32 s8, s7, 1 // (op 9759) 

v_lshrrev_b32 v0, 2, v3 // (op 9759) 

s_add_u32 s7, s8, s15 // (op 9759) 


v_and_b32 v3, 3, v0 // (op 9759) 

s_lshr_b32 s8, s7, s11 // (op 9759) 

v_and_b32 v0, 3, v1 // (op 9759) 
s_and_b32 s7, s11, 2147483648 // (op 9759) 
v_lshl_add_u32 v14, v3, 2, v0 // (op 9759) 


s_cmp_eq_i32 s7, 0 // (op 9759) 
s_mov_b32 s15, scc // (op 9759) 

v_lshrrev_b32 v0, 6, v1 // (op 9759) 
s_cmp_eq_i32 s15, 0 // (op 9759) 
s_cselect_b32 s7, s6, s8 // (op 9759) 



v_and_b32 v3, 1, v0 // (op 9759) 

s_mul_hi_u32 s6, s7, s12 // (op 9759) 
v_lshl_add_u32 v0, v3, 4, v14 // (op 9759) 


s_sub_u32 s8, s7, s6 // (op 9759) 
s_lshr_b32 s15, s8, 1 // (op 9759) 

s_add_u32 s8, s15, s6 // (op 9759) 


s_lshr_b32 s6, s8, s10 // (op 9759) 

s_and_b32 s8, s10, 2147483648 // (op 9759) 
s_cmp_eq_i32 s8, 0 // (op 9759) 
s_mov_b32 s15, scc // (op 9759) 

s_cmp_eq_i32 s15, 0 // (op 9759) 
s_cselect_b32 s8, s7, s6 // (op 9759) 


s_mul_i32 s6, s8, s14 // (op 9759) 
v_lshl_add_u32 v3, s8, 6, v5 // (op 9759) 


s_sub_u32 s8, s7, s6 // (op 9759) 


v_lshl_add_u32 v5, s8, 7, v0 // (op 9759) 


v_mul_lo_u32 v0, v5, s24 // (op 9759) 

v_add_lshl_u32 v5, v3, v0, 2 // (op 9759) 


s_mul_i32 s19, s24, s23 // (op 9761) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 9761) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 9761) least significant
s_mul_hi_u32 s6, s24, s22 // (op 9761) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9761) most significant: sum
s_add_u32 s19, s19, s6 // (op 9761) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 9761) 

s_mov_b32 s60, s26 // (op 9761) reg expression
s_mov_b32 s61, s27 // (op 9761) reg expression
s_mov_b32 s62, s28 // (op 9761) reg expression
s_mov_b32 s63, 131072 // (op 9761) call()

v_and_b32 v0, 48, v1 // (op 9838) 
v_and_b32 v3, 63, v1 // (op 9838) 
v_sub_u32 v14, v3, v0 // (op 9838) 

v_lshrrev_b32 v3, 1, v14 // (op 9838) 
v_add_lshl_u32 v15, v0, v3, 6 // (op 9838) 


v_and_b32 v0, 1, v14 // (op 9838) 

v_lshlrev_b32 v3, 3, v0 // (op 9838) 

v_lshrrev_b32 v0, 7, v1 // (op 9838) 
v_lshl_add_u32 v14, v0, 4, v3 // (op 9838) 


v_add_u32 v0, v15, v14 // (op 9838) 


v_lshrrev_b32 v3, 6, v1 // (op 9913) 
v_and_b32 v14, 1, v3 // (op 9913) 

v_and_b32 v3, 15, v1 // (op 9913) 
v_lshl_add_u32 v15, v14, 4, v3 // (op 9913) 


v_and_b32 v3, 48, v1 // (op 9913) 
v_lshl_add_u32 v14, v15, 7, v3 // (op 9913) 


v_lshrrev_b32 v3, 7, v1 // (op 9989) 
v_and_b32 v15, 15, v1 // (op 9989) 
v_lshl_add_u32 v16, v3, 4, v15 // (op 9989) 


v_and_b32 v3, 63, v1 // (op 9989) 
v_lshrrev_b32 v15, 4, v3 // (op 9989) 

v_lshl_add_u32 v3, v15, 6, v16 // (op 9989) 


v_lshrrev_b32 v15, 6, v1 // (op 10065) 
v_and_b32 v16, 1, v15 // (op 10065) 

v_and_b32 v15, 15, v1 // (op 10065) 
v_lshl_add_u32 v17, v16, 4, v15 // (op 10065) 


v_and_b32 v15, 63, v1 // (op 10065) 
v_lshrrev_b32 v16, 4, v15 // (op 10065) 

v_lshl_add_u32 v15, v17, 2, v16 // (op 10065) 


v_accvgpr_write a0, 0.00000 // (op 180) call()
v_accvgpr_write a1, 0.00000 // (op 180) call()
v_accvgpr_write a2, 0.00000 // (op 180) call()
v_accvgpr_write a3, 0.00000 // (op 180) call()
v_accvgpr_write a4, 0.00000 // (op 614) call()
v_accvgpr_write a5, 0.00000 // (op 614) call()
v_accvgpr_write a6, 0.00000 // (op 614) call()
v_accvgpr_write a7, 0.00000 // (op 614) call()
v_accvgpr_write a8, 0.00000 // (op 790) call()
v_accvgpr_write a9, 0.00000 // (op 790) call()
v_accvgpr_write a10, 0.00000 // (op 790) call()
v_accvgpr_write a11, 0.00000 // (op 790) call()
v_accvgpr_write a12, 0.00000 // (op 966) call()
v_accvgpr_write a13, 0.00000 // (op 966) call()
v_accvgpr_write a14, 0.00000 // (op 966) call()
v_accvgpr_write a15, 0.00000 // (op 966) call()
v_accvgpr_write a16, 0.00000 // (op 1230) call()
v_accvgpr_write a17, 0.00000 // (op 1230) call()
v_accvgpr_write a18, 0.00000 // (op 1230) call()
v_accvgpr_write a19, 0.00000 // (op 1230) call()
v_accvgpr_write a20, 0.00000 // (op 1539) call()
v_accvgpr_write a21, 0.00000 // (op 1539) call()
v_accvgpr_write a22, 0.00000 // (op 1539) call()
v_accvgpr_write a23, 0.00000 // (op 1539) call()
v_accvgpr_write a24, 0.00000 // (op 1540) call()
v_accvgpr_write a25, 0.00000 // (op 1540) call()
v_accvgpr_write a26, 0.00000 // (op 1540) call()
v_accvgpr_write a27, 0.00000 // (op 1540) call()
v_accvgpr_write a28, 0.00000 // (op 1541) call()
v_accvgpr_write a29, 0.00000 // (op 1541) call()
v_accvgpr_write a30, 0.00000 // (op 1541) call()
v_accvgpr_write a31, 0.00000 // (op 1541) call()
v_and_b32 v16, -4, v1 // (op 8893) 
s_mul_i32 s15, s16, s50 // (op 8893) 
s_add_u32 s15, s15, s0 // (op 8893) 
v_sub_u32 v17, v1, v16 // (op 8893) 

s_add_u32 s6, s15, s9 // (op 8893) 

v_lshlrev_b32 v16, 4, v17 // (op 8893) 

s_mul_hi_u32 s15, s6, s13 // (op 8893) 
v_lshrrev_b32 v17, 2, v1 // (op 8893) 
s_sub_u32 s7, s6, s15 // (op 8893) 
s_lshr_b32 s8, s7, 1 // (op 8893) 

s_add_u32 s7, s8, s15 // (op 8893) 


s_lshr_b32 s8, s7, s11 // (op 8893) 

s_and_b32 s7, s11, 2147483648 // (op 8893) 
s_cmp_eq_i32 s7, 0 // (op 8893) 
s_mov_b32 s15, scc // (op 8893) 

s_cmp_eq_i32 s15, 0 // (op 8893) 
s_cselect_b32 s7, s6, s8 // (op 8893) 


s_mul_hi_u32 s8, s7, s12 // (op 8893) 
s_sub_u32 s15, s7, s8 // (op 8893) 
s_lshr_b32 s18, s15, 1 // (op 8893) 

s_add_u32 s15, s18, s8 // (op 8893) 


s_lshr_b32 s8, s15, s10 // (op 8893) 

s_and_b32 s15, s10, 2147483648 // (op 8893) 
s_cmp_eq_i32 s15, 0 // (op 8893) 
s_mov_b32 s18, scc // (op 8893) 

s_cmp_eq_i32 s18, 0 // (op 8893) 
s_cselect_b32 s15, s7, s8 // (op 8893) 


v_lshl_add_u32 v20, s15, 6, v16 // (op 8893) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 8893) 
v_mov_b32 v23, 0 // (op 8893) convert
v_mov_b32 v22, v20 // (op 8893) convert

s_mul_i32 s8, s7, s18 // (op 8893) 


s_sub_u32 s7, s6, s8 // (op 8893) 


v_lshl_add_u32 v16, s7, 7, v17 // (op 8893) 


v_mul_lo_u32 v21, v16, s31 // (op 8893) most significant: low of low * high
v_mul_hi_u32 v17, v16, s30 // (op 8893) most significant: high of low * low
v_mul_lo_u32 v20, v16, s30 // (op 8893) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 8893) most significant: sum


v_add_co_u32 v16, s[18:19], v22, v20 // (op 8893) least significant half
v_addc_co_u32 v17, s[18:19], v23, v21, s[18:19] // (op 8893) most significant half



s_mul_i32 s19, s30, s3 // (op 8895) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 8895) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 8895) least significant
s_mul_hi_u32 s6, s30, s2 // (op 8895) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 8895) most significant: sum
s_add_u32 s19, s19, s6 // (op 8895) most significant: sum


s_mov_b32 s68, s32 // (op 8895) reg expression
s_mov_b32 s69, s33 // (op 8895) reg expression
s_mov_b32 s70, s18 // (op 8895) reg expression
s_mov_b32 s71, 131072 // (op 8895) call()

v_and_b32 v20, -4, v1 // (op 8912) 
s_mul_i32 s15, s16, s50 // (op 8912) 
s_add_u32 s15, s15, s0 // (op 8912) 
v_sub_u32 v21, v1, v20 // (op 8912) 

s_add_u32 s6, s15, s9 // (op 8912) 

v_lshlrev_b32 v20, 4, v21 // (op 8912) 

s_mul_hi_u32 s15, s6, s13 // (op 8912) 
v_lshrrev_b32 v21, 2, v1 // (op 8912) 
s_sub_u32 s7, s6, s15 // (op 8912) 
s_lshr_b32 s8, s7, 1 // (op 8912) 

s_add_u32 s7, s8, s15 // (op 8912) 


s_lshr_b32 s8, s7, s11 // (op 8912) 

s_and_b32 s7, s11, 2147483648 // (op 8912) 
s_cmp_eq_i32 s7, 0 // (op 8912) 
s_mov_b32 s15, scc // (op 8912) 

s_cmp_eq_i32 s15, 0 // (op 8912) 
s_cselect_b32 s7, s6, s8 // (op 8912) 


s_mul_hi_u32 s8, s7, s12 // (op 8912) 
s_sub_u32 s15, s7, s8 // (op 8912) 
s_lshr_b32 s18, s15, 1 // (op 8912) 

s_add_u32 s15, s18, s8 // (op 8912) 


s_lshr_b32 s8, s15, s10 // (op 8912) 

s_and_b32 s15, s10, 2147483648 // (op 8912) 
s_cmp_eq_i32 s15, 0 // (op 8912) 
s_mov_b32 s18, scc // (op 8912) 

s_cmp_eq_i32 s18, 0 // (op 8912) 
s_cselect_b32 s15, s7, s8 // (op 8912) 


v_lshl_add_u32 v22, s15, 6, v20 // (op 8912) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 8912) 
v_mov_b32 v25, 0 // (op 8912) convert
v_mov_b32 v24, v22 // (op 8912) convert

s_mul_i32 s8, s7, s18 // (op 8912) 


s_sub_u32 s7, s6, s8 // (op 8912) 


v_lshl_add_u32 v20, s7, 7, v21 // (op 8912) 


v_mul_lo_u32 v23, v20, s31 // (op 8912) most significant: low of low * high
v_mul_hi_u32 v21, v20, s30 // (op 8912) most significant: high of low * low
v_mul_lo_u32 v22, v20, s30 // (op 8912) least significant: low of low * low
v_add_u32 v23, v23, v21 // (op 8912) most significant: sum


v_add_co_u32 v20, s[18:19], v24, v22 // (op 8912) least significant half
v_addc_co_u32 v21, s[18:19], v25, v23, s[18:19] // (op 8912) most significant half



s_mul_i32 s19, s30, s3 // (op 8914) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 8914) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 8914) least significant
s_mul_hi_u32 s6, s30, s2 // (op 8914) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 8914) most significant: sum
s_add_u32 s19, s19, s6 // (op 8914) most significant: sum


s_mov_b32 s68, s32 // (op 8914) reg expression
s_mov_b32 s69, s33 // (op 8914) reg expression
s_mov_b32 s70, s18 // (op 8914) reg expression
s_mov_b32 s71, 131072 // (op 8914) call()

v_lshrrev_b32 v22, 3, v1 // (op 9031) 
s_mul_i32 s15, s16, s50 // (op 9031) 
s_add_u32 s15, s15, s0 // (op 9031) 
v_and_b32 v23, -8, v1 // (op 9031) 
s_add_u32 s6, s15, s9 // (op 9031) 

v_sub_u32 v24, v1, v23 // (op 9031) 

s_mul_hi_u32 s15, s6, s13 // (op 9031) 
v_lshlrev_b32 v23, 4, v24 // (op 9031) 

s_sub_u32 s7, s6, s15 // (op 9031) 
s_lshr_b32 s8, s7, 1 // (op 9031) 

s_add_u32 s7, s8, s15 // (op 9031) 


s_lshr_b32 s8, s7, s11 // (op 9031) 

s_and_b32 s7, s11, 2147483648 // (op 9031) 
s_cmp_eq_i32 s7, 0 // (op 9031) 
s_mov_b32 s15, scc // (op 9031) 

s_cmp_eq_i32 s15, 0 // (op 9031) 
s_cselect_b32 s7, s6, s8 // (op 9031) 


s_mul_hi_u32 s8, s7, s12 // (op 9031) 
s_sub_u32 s15, s7, s8 // (op 9031) 
s_lshr_b32 s18, s15, 1 // (op 9031) 

s_add_u32 s15, s18, s8 // (op 9031) 


s_lshr_b32 s8, s15, s10 // (op 9031) 

s_and_b32 s15, s10, 2147483648 // (op 9031) 
s_cmp_eq_i32 s15, 0 // (op 9031) 
s_mov_b32 s18, scc // (op 9031) 

s_cmp_eq_i32 s18, 0 // (op 9031) 
s_cselect_b32 s15, s7, s8 // (op 9031) 


s_mul_i32 s8, s15, s14 // (op 9031) 

s_sub_u32 s15, s7, s8 // (op 9031) 

v_lshl_add_u32 v24, s15, 7, v22 // (op 9031) 


v_mul_lo_u32 v27, v24, s35 // (op 9031) most significant: low of low * high
v_mul_hi_u32 v22, v24, s34 // (op 9031) most significant: high of low * low
v_mul_lo_u32 v26, v24, s34 // (op 9031) least significant: low of low * low
v_add_u32 v27, v27, v22 // (op 9031) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 9031) 
s_mul_i32 s8, s7, s18 // (op 9031) 


s_sub_u32 s7, s6, s8 // (op 9031) 


v_lshl_add_u32 v22, s7, 7, v23 // (op 9031) 


v_mov_b32 v25, 0 // (op 9031) convert
v_mov_b32 v24, v22 // (op 9031) convert

v_add_co_u32 v22, s[18:19], v24, v26 // (op 9031) least significant half
v_addc_co_u32 v23, s[18:19], v25, v27, s[18:19] // (op 9031) most significant half



s_mul_i32 s19, s34, s23 // (op 9033) most significant: low of low * high
s_mul_i32 s15, s35, s22 // (op 9033) most significant: low of high * low
s_mul_i32 s18, s34, s22 // (op 9033) least significant
s_mul_hi_u32 s6, s34, s22 // (op 9033) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9033) most significant: sum
s_add_u32 s19, s19, s6 // (op 9033) most significant: sum


s_mov_b32 s72, s36 // (op 9033) reg expression
s_mov_b32 s73, s37 // (op 9033) reg expression
s_mov_b32 s74, s18 // (op 9033) reg expression
s_mov_b32 s75, 131072 // (op 9033) call()

v_lshrrev_b32 v24, 3, v1 // (op 9050) 
s_mul_i32 s15, s16, s50 // (op 9050) 
s_add_u32 s15, s15, s0 // (op 9050) 
v_and_b32 v25, -8, v1 // (op 9050) 
s_add_u32 s6, s15, s9 // (op 9050) 

v_sub_u32 v26, v1, v25 // (op 9050) 

s_mul_hi_u32 s15, s6, s13 // (op 9050) 
v_lshlrev_b32 v25, 4, v26 // (op 9050) 

s_sub_u32 s7, s6, s15 // (op 9050) 
s_lshr_b32 s8, s7, 1 // (op 9050) 

s_add_u32 s7, s8, s15 // (op 9050) 


s_lshr_b32 s8, s7, s11 // (op 9050) 

s_and_b32 s7, s11, 2147483648 // (op 9050) 
s_cmp_eq_i32 s7, 0 // (op 9050) 
s_mov_b32 s15, scc // (op 9050) 

s_cmp_eq_i32 s15, 0 // (op 9050) 
s_cselect_b32 s7, s6, s8 // (op 9050) 


s_mul_hi_u32 s8, s7, s12 // (op 9050) 
s_sub_u32 s15, s7, s8 // (op 9050) 
s_lshr_b32 s18, s15, 1 // (op 9050) 

s_add_u32 s15, s18, s8 // (op 9050) 


s_lshr_b32 s8, s15, s10 // (op 9050) 

s_and_b32 s15, s10, 2147483648 // (op 9050) 
s_cmp_eq_i32 s15, 0 // (op 9050) 
s_mov_b32 s18, scc // (op 9050) 

s_cmp_eq_i32 s18, 0 // (op 9050) 
s_cselect_b32 s15, s7, s8 // (op 9050) 


s_mul_i32 s8, s15, s14 // (op 9050) 

s_sub_u32 s15, s7, s8 // (op 9050) 

v_lshl_add_u32 v26, s15, 7, v24 // (op 9050) 


v_mul_lo_u32 v29, v26, s35 // (op 9050) most significant: low of low * high
v_mul_hi_u32 v24, v26, s34 // (op 9050) most significant: high of low * low
v_mul_lo_u32 v28, v26, s34 // (op 9050) least significant: low of low * low
v_add_u32 v29, v29, v24 // (op 9050) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 9050) 
s_mul_i32 s8, s7, s18 // (op 9050) 


s_sub_u32 s7, s6, s8 // (op 9050) 


v_lshl_add_u32 v24, s7, 7, v25 // (op 9050) 


v_mov_b32 v27, 0 // (op 9050) convert
v_mov_b32 v26, v24 // (op 9050) convert

v_add_co_u32 v24, s[18:19], v26, v28 // (op 9050) least significant half
v_addc_co_u32 v25, s[18:19], v27, v29, s[18:19] // (op 9050) most significant half



s_mul_i32 s19, s34, s23 // (op 9052) most significant: low of low * high
s_mul_i32 s15, s35, s22 // (op 9052) most significant: low of high * low
s_mul_i32 s18, s34, s22 // (op 9052) least significant
s_mul_hi_u32 s6, s34, s22 // (op 9052) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9052) most significant: sum
s_add_u32 s19, s19, s6 // (op 9052) most significant: sum


s_mov_b32 s72, s36 // (op 9052) reg expression
s_mov_b32 s73, s37 // (op 9052) reg expression
s_mov_b32 s74, s18 // (op 9052) reg expression
s_mov_b32 s75, 131072 // (op 9052) call()

v_and_b32 v26, -64, v1 // (op 9159) 
s_mul_i32 s15, s16, s50 // (op 9159) 
s_add_u32 s15, s15, s0 // (op 9159) 
v_sub_u32 v27, v1, v26 // (op 9159) 

s_add_u32 s6, s15, s9 // (op 9159) 

v_lshrrev_b32 v26, 6, v1 // (op 9159) 
s_mul_hi_u32 s15, s6, s13 // (op 9159) 
s_sub_u32 s7, s6, s15 // (op 9159) 
s_lshr_b32 s8, s7, 1 // (op 9159) 

s_add_u32 s7, s8, s15 // (op 9159) 


s_lshr_b32 s8, s7, s11 // (op 9159) 

s_and_b32 s7, s11, 2147483648 // (op 9159) 
s_cmp_eq_i32 s7, 0 // (op 9159) 
s_mov_b32 s15, scc // (op 9159) 

s_cmp_eq_i32 s15, 0 // (op 9159) 
s_cselect_b32 s7, s6, s8 // (op 9159) 


s_mul_hi_u32 s8, s7, s12 // (op 9159) 
s_sub_u32 s15, s7, s8 // (op 9159) 
s_lshr_b32 s18, s15, 1 // (op 9159) 

s_add_u32 s15, s18, s8 // (op 9159) 


s_lshr_b32 s8, s15, s10 // (op 9159) 

s_and_b32 s15, s10, 2147483648 // (op 9159) 
s_cmp_eq_i32 s15, 0 // (op 9159) 
s_mov_b32 s18, scc // (op 9159) 

s_cmp_eq_i32 s18, 0 // (op 9159) 
s_cselect_b32 s15, s7, s8 // (op 9159) 


v_lshl_add_u32 v28, s15, 6, v27 // (op 9159) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 9159) 
s_mul_i32 s8, s7, s18 // (op 9159) 


s_sub_u32 s7, s6, s8 // (op 9159) 


v_lshl_add_u32 v27, s7, 2, v26 // (op 9159) 


v_mul_lo_u32 v26, v27, s38 // (op 9159) 

v_add_u32 v27, v28, v26 // (op 9159) 


s_mul_i32 s19, s38, s43 // (op 9161) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 9161) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 9161) least significant
s_mul_hi_u32 s6, s38, s42 // (op 9161) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9161) most significant: sum
s_add_u32 s19, s19, s6 // (op 9161) most significant: sum


s_mov_b32 s76, s40 // (op 9161) reg expression
s_mov_b32 s77, s41 // (op 9161) reg expression
s_mov_b32 s78, s18 // (op 9161) reg expression
s_mov_b32 s79, 131072 // (op 9161) call()

v_and_b32 v26, -64, v1 // (op 9178) 
s_mul_i32 s15, s16, s50 // (op 9178) 
s_add_u32 s15, s15, s0 // (op 9178) 
v_sub_u32 v28, v1, v26 // (op 9178) 

s_add_u32 s6, s15, s9 // (op 9178) 

v_lshrrev_b32 v26, 6, v1 // (op 9178) 
s_mul_hi_u32 s15, s6, s13 // (op 9178) 
s_sub_u32 s7, s6, s15 // (op 9178) 
s_lshr_b32 s8, s7, 1 // (op 9178) 

s_add_u32 s7, s8, s15 // (op 9178) 


s_lshr_b32 s8, s7, s11 // (op 9178) 

s_and_b32 s7, s11, 2147483648 // (op 9178) 
s_cmp_eq_i32 s7, 0 // (op 9178) 
s_mov_b32 s15, scc // (op 9178) 

s_cmp_eq_i32 s15, 0 // (op 9178) 
s_cselect_b32 s7, s6, s8 // (op 9178) 


s_mul_hi_u32 s8, s7, s12 // (op 9178) 
s_sub_u32 s15, s7, s8 // (op 9178) 
s_lshr_b32 s18, s15, 1 // (op 9178) 

s_add_u32 s15, s18, s8 // (op 9178) 


s_lshr_b32 s8, s15, s10 // (op 9178) 

s_and_b32 s15, s10, 2147483648 // (op 9178) 
s_cmp_eq_i32 s15, 0 // (op 9178) 
s_mov_b32 s18, scc // (op 9178) 

s_cmp_eq_i32 s18, 0 // (op 9178) 
s_cselect_b32 s15, s7, s8 // (op 9178) 


v_lshl_add_u32 v29, s15, 6, v28 // (op 9178) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 9178) 
s_mul_i32 s8, s7, s18 // (op 9178) 


s_sub_u32 s7, s6, s8 // (op 9178) 


v_lshl_add_u32 v28, s7, 2, v26 // (op 9178) 


v_mul_lo_u32 v26, v28, s38 // (op 9178) 

v_add_u32 v28, v29, v26 // (op 9178) 


s_mul_i32 s19, s38, s43 // (op 9180) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 9180) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 9180) least significant
s_mul_hi_u32 s6, s38, s42 // (op 9180) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9180) most significant: sum
s_add_u32 s19, s19, s6 // (op 9180) most significant: sum


s_mov_b32 s76, s40 // (op 9180) reg expression
s_mov_b32 s77, s41 // (op 9180) reg expression
s_mov_b32 s78, s18 // (op 9180) reg expression
s_mov_b32 s79, 131072 // (op 9180) call()

v_lshrrev_b32 v26, 2, v1 // (op 9287) 
s_mul_i32 s15, s16, s50 // (op 9287) 
s_add_u32 s15, s15, s0 // (op 9287) 
v_lshlrev_b32 v29, 1, v26 // (op 9287) 

s_add_u32 s6, s15, s9 // (op 9287) 

v_and_b32 v26, -4, v1 // (op 9287) 
s_mul_hi_u32 s15, s6, s13 // (op 9287) 
v_sub_u32 v30, v1, v26 // (op 9287) 

s_sub_u32 s7, s6, s15 // (op 9287) 
s_lshr_b32 s8, s7, 1 // (op 9287) 

s_add_u32 s7, s8, s15 // (op 9287) 


s_lshr_b32 s8, s7, s11 // (op 9287) 

s_and_b32 s7, s11, 2147483648 // (op 9287) 
s_cmp_eq_i32 s7, 0 // (op 9287) 
s_mov_b32 s15, scc // (op 9287) 

s_cmp_eq_i32 s15, 0 // (op 9287) 
s_cselect_b32 s7, s6, s8 // (op 9287) 


s_mul_hi_u32 s8, s7, s12 // (op 9287) 
s_sub_u32 s15, s7, s8 // (op 9287) 
s_lshr_b32 s18, s15, 1 // (op 9287) 

s_add_u32 s15, s18, s8 // (op 9287) 


s_lshr_b32 s8, s15, s10 // (op 9287) 

s_and_b32 s15, s10, 2147483648 // (op 9287) 
s_cmp_eq_i32 s15, 0 // (op 9287) 
s_mov_b32 s18, scc // (op 9287) 

s_cmp_eq_i32 s18, 0 // (op 9287) 
s_cselect_b32 s15, s7, s8 // (op 9287) 


s_mul_i32 s8, s15, s14 // (op 9287) 

s_sub_u32 s15, s7, s8 // (op 9287) 

v_lshl_add_u32 v26, s15, 7, v29 // (op 9287) 


v_mul_lo_u32 v29, v26, s44 // (op 9287) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 9287) 
s_mul_i32 s8, s7, s18 // (op 9287) 


s_sub_u32 s7, s6, s8 // (op 9287) 


v_lshl_add_u32 v26, s7, 2, v30 // (op 9287) 


v_add_u32 v30, v26, v29 // (op 9287) 


s_mul_i32 s19, s44, s23 // (op 9289) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 9289) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 9289) least significant
s_mul_hi_u32 s6, s44, s22 // (op 9289) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9289) most significant: sum
s_add_u32 s19, s19, s6 // (op 9289) most significant: sum


s_mov_b32 s80, s46 // (op 9289) reg expression
s_mov_b32 s81, s47 // (op 9289) reg expression
s_mov_b32 s82, s18 // (op 9289) reg expression
s_mov_b32 s83, 131072 // (op 9289) call()

v_lshrrev_b32 v26, 2, v1 // (op 9306) 
s_mul_i32 s15, s16, s50 // (op 9306) 
s_add_u32 s15, s15, s0 // (op 9306) 
v_lshlrev_b32 v29, 1, v26 // (op 9306) 

s_add_u32 s6, s15, s9 // (op 9306) 

v_and_b32 v26, -4, v1 // (op 9306) 
s_mul_hi_u32 s15, s6, s13 // (op 9306) 
v_sub_u32 v31, v1, v26 // (op 9306) 

s_sub_u32 s7, s6, s15 // (op 9306) 
s_lshr_b32 s8, s7, 1 // (op 9306) 

s_add_u32 s7, s8, s15 // (op 9306) 


s_lshr_b32 s8, s7, s11 // (op 9306) 

s_and_b32 s7, s11, 2147483648 // (op 9306) 
s_cmp_eq_i32 s7, 0 // (op 9306) 
s_mov_b32 s15, scc // (op 9306) 

s_cmp_eq_i32 s15, 0 // (op 9306) 
s_cselect_b32 s7, s6, s8 // (op 9306) 


s_mul_hi_u32 s8, s7, s12 // (op 9306) 
s_sub_u32 s15, s7, s8 // (op 9306) 
s_lshr_b32 s18, s15, 1 // (op 9306) 

s_add_u32 s15, s18, s8 // (op 9306) 


s_lshr_b32 s8, s15, s10 // (op 9306) 

s_and_b32 s15, s10, 2147483648 // (op 9306) 
s_cmp_eq_i32 s15, 0 // (op 9306) 
s_mov_b32 s18, scc // (op 9306) 

s_cmp_eq_i32 s18, 0 // (op 9306) 
s_cselect_b32 s15, s7, s8 // (op 9306) 


s_mul_i32 s8, s15, s14 // (op 9306) 

s_sub_u32 s15, s7, s8 // (op 9306) 

v_lshl_add_u32 v26, s15, 7, v29 // (op 9306) 


v_mul_lo_u32 v29, v26, s44 // (op 9306) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 9306) 
s_mul_i32 s8, s7, s18 // (op 9306) 


s_sub_u32 s7, s6, s8 // (op 9306) 


v_lshl_add_u32 v26, s7, 2, v31 // (op 9306) 


v_add_u32 v31, v26, v29 // (op 9306) 


s_mul_i32 s19, s44, s23 // (op 9308) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 9308) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 9308) least significant
s_mul_hi_u32 s6, s44, s22 // (op 9308) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9308) most significant: sum
s_add_u32 s19, s19, s6 // (op 9308) most significant: sum


s_mov_b32 s80, s46 // (op 9308) reg expression
s_mov_b32 s81, s47 // (op 9308) reg expression
s_mov_b32 s82, s18 // (op 9308) reg expression
s_mov_b32 s83, 131072 // (op 9308) call()

v_and_b32 v26, -8, v1 // (op 9879) 
v_sub_u32 v29, v1, v26 // (op 9879) 

v_lshlrev_b32 v26, 4, v29 // (op 9879) 

v_lshrrev_b32 v29, 3, v1 // (op 9879) 
v_lshl_add_u32 v32, v29, 7, v26 // (op 9879) 


s_nop 0
v_readfirstlane_b32 s15, v32 // (op 9879) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10680) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 8797) 
v_mov_b64 v[32:33], s[18:19] // (op 8797) convert


v_mov_b64 v[34:35], v[22:23] // (op 8797) 
s_mov_b32 m0, s15 // (op 8797) reg expression
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 8797) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 8797) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 8797) most significant half

s_add_u32 m0, m0, 4096 // (op 8797) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 8797) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 8797) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 8797) most significant half

s_add_u32 m0, m0, 4096 // (op 8797) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 8797) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 8797) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 8797) most significant half

s_add_u32 m0, m0, 4096 // (op 8797) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 8797) Load value direct to lds




v_mov_b64 v[32:33], s[44:45] // (op 8345) convert

v_mov_b32 v26, v30 // (op 8345) 
buffer_load_ubyte v34, v26, s[80:83], 0 offen offset: 0 // (op 8345) Load value
v_add_u32 v26, v26, v32 // (op 8345) 
buffer_load_ubyte v35, v26, s[80:83], 0 offen offset: 0 // (op 8345) Load value


v_and_b32 v26, -4, v1 // (op 9804) 
v_sub_u32 v29, v1, v26 // (op 9804) 

v_lshlrev_b32 v26, 4, v29 // (op 9804) 

v_lshrrev_b32 v29, 2, v1 // (op 9804) 
v_lshl_add_u32 v32, v29, 6, v26 // (op 9804) 


s_nop 0
v_readfirstlane_b32 s15, v32 // (op 9804) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10661) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 8802) 
v_mov_b64 v[32:33], s[18:19] // (op 8802) convert


v_mov_b64 v[36:37], v[16:17] // (op 8802) reg expression
s_add_u32 s6, 16384, s15 // (op 8802) 
s_mov_b32 m0, s6 // (op 8802) reg expression
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 8802) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 8802) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 8802) most significant half

s_add_u32 m0, m0, 4096 // (op 8802) 
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 8802) Load value direct to lds






v_mov_b32 v26, v27 // (op 8351) reg expression

buffer_load_ubyte v29, v26, s[76:79], 0 offen offset: 0 // (op 8351) Load value

v_and_b32 v26, -8, v1 // (op 10189) 
v_sub_u32 v32, v1, v26 // (op 10189) 

v_lshlrev_b32 v26, 4, v32 // (op 10189) 

v_lshrrev_b32 v32, 3, v1 // (op 10189) 
v_lshl_add_u32 v33, v32, 7, v26 // (op 10189) 


s_nop 0
v_readfirstlane_b32 s15, v33 // (op 10189) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10751) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 8807) 
v_mov_b64 v[32:33], s[18:19] // (op 8807) convert


v_mov_b64 v[36:37], v[22:23] // (op 8807) 
s_add_u32 s6, 24576, s15 // (op 8807) 
s_mov_b32 m0, s6 // (op 8807) reg expression
s_mov_b32 s7, 128 // (op 8807) call()
buffer_load_dwordx4 v36, s[72:75], s7 offen offset: 0 lds // (op 8807) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 8807) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 8807) most significant half

s_add_u32 m0, m0, 4096 // (op 8807) 
buffer_load_dwordx4 v36, s[72:75], s7 offen offset: 0 lds // (op 8807) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 8807) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 8807) most significant half

s_add_u32 m0, m0, 4096 // (op 8807) 
buffer_load_dwordx4 v36, s[72:75], s7 offen offset: 0 lds // (op 8807) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 8807) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 8807) most significant half

s_add_u32 m0, m0, 4096 // (op 8807) 
buffer_load_dwordx4 v36, s[72:75], s7 offen offset: 0 lds // (op 8807) Load value direct to lds







v_mov_b64 v[22:23], s[44:45] // (op 8357) convert

v_mov_b32 v26, v30 // (op 8357) 
buffer_load_ubyte v32, v26, s[80:83], 0 offen offset:4 // (op 8357) Load value
v_add_u32 v26, v26, v22 // (op 8357) 
buffer_load_ubyte v33, v26, s[80:83], 0 offen offset:4 // (op 8357) Load value



v_and_b32 v26, -4, v1 // (op 10145) 
v_sub_u32 v30, v1, v26 // (op 10145) 

v_lshlrev_b32 v26, 4, v30 // (op 10145) 

v_lshrrev_b32 v30, 2, v1 // (op 10145) 
v_lshl_add_u32 v22, v30, 6, v26 // (op 10145) 


s_nop 0
v_readfirstlane_b32 s15, v22 // (op 10145) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10741) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 8809) 
v_mov_b64 v[22:23], s[18:19] // (op 8809) convert


s_lshl_b64 s[18:19], s[30:31], 7 // (op 8809) 
v_mov_b32 v26, s18 // (op 8809) 
v_mov_b32 v30, s19 // (op 8809) 
v_add_co_u32 v36, s[28:29], v16, v26 // (op 8809) least significant half
v_addc_co_u32 v37, s[28:29], v17, v30, s[28:29] // (op 8809) most significant half




s_add_u32 s6, 40960, s15 // (op 8809) 
s_mov_b32 m0, s6 // (op 8809) reg expression
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 8809) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v22 // (op 8809) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 8809) most significant half

s_add_u32 m0, m0, 4096 // (op 8809) 
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 8809) Load value direct to lds







s_lshl_b32 s15, s38, 2 // (op 8363) 
v_add_u32 v26, v27, s15 // (op 8363) 


buffer_load_ubyte v30, v26, s[76:79], 0 offen offset: 0 // (op 8363) Load value


v_and_b32 v16, -4, v1 // (op 10031) 
v_sub_u32 v17, v1, v16 // (op 10031) 

v_lshrrev_b32 v16, 2, v1 // (op 10031) 
v_lshl_add_u32 v22, v16, 3, v17 // (op 10031) 


s_barrier  // (op 10721) 



s_waitcnt vmcnt(13)
ds_write_b8 v22, v34 offset:49152 // (op 8366) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v22, v35 offset:49156 // (op 8366) Store local data 

v_and_b32 v16, -64, v1 // (op 9955) 
v_sub_u32 v17, v1, v16 // (op 9955) 

v_and_b32 v16, 4294967232, v1 // (op 9955) 
v_add_u32 v22, v16, v17 // (op 9955) 


s_barrier  // (op 10700) 



s_waitcnt vmcnt(9)
ds_write_b8 v22, v29 offset:49664 // (op 8368) Store local data 

s_barrier  // (op 10677) 



ds_read_b128 v[36:39], v14 // (op 8393) Load local data 
ds_read_b128 v[40:43], v14 offset:64 // (op 8393) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10718) 



ds_read_u8 v16, v15 offset:49152 // (op 8423) Load local data 



ds_read_b128 v[44:47], v14 offset:4096 // (op 8398) Load local data 
ds_read_b128 v[48:51], v14 offset:4160 // (op 8398) Load local data 



ds_read_u8 v17, v15 offset:49280 // (op 8428) Load local data 



ds_read_b128 v[52:55], v14 offset:8192 // (op 8403) Load local data 
ds_read_b128 v[56:59], v14 offset:8256 // (op 8403) Load local data 



ds_read_u8 v22, v15 offset:49408 // (op 8433) Load local data 



ds_read_b128 v[60:63], v14 offset:12288 // (op 8408) Load local data 
ds_read_b128 v[64:67], v14 offset:12352 // (op 8408) Load local data 



ds_read_u8 v23, v15 offset:49536 // (op 8438) Load local data 
s_barrier  // (op 10658) 



ds_read_b64_tr_b8 v[68:69], v0 offset:16384 // (op 8383) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v0 offset:16896 // (op 8383) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v0 offset:20480 // (op 8383) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v0 offset:20992 // (op 8383) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 10697) 


ds_read_u8 v26, v3 offset:49664 // (op 8413) Load local data 



ds_read_b64_tr_b8 v[76:77], v0 offset:16416 // (op 8388) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v0 offset:16928 // (op 8388) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v0 offset:20512 // (op 8388) Transpose load local data 
ds_read_b64_tr_b8 v[82:83], v0 offset:21024 // (op 8388) Transpose load local data 


ds_read_u8 v27, v3 offset:49696 // (op 8418) Load local data 
v_and_b32 v84, -4, v1 // (op 9789) 
v_sub_u32 v85, v1, v84 // (op 9789) 

v_lshlrev_b32 v84, 4, v85 // (op 9789) 

v_lshrrev_b32 v85, 2, v1 // (op 9789) 
v_lshl_add_u32 v86, v85, 6, v84 // (op 9789) 


s_nop 0
v_readfirstlane_b32 s15, v86 // (op 9789) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -8, v1 // (op 9864) 
v_sub_u32 v85, v1, v84 // (op 9864) 

v_lshlrev_b32 v84, 4, v85 // (op 9864) 

v_lshrrev_b32 v85, 3, v1 // (op 9864) 
v_lshl_add_u32 v86, v85, 7, v84 // (op 9864) 


s_nop 0
v_readfirstlane_b32 s6, v86 // (op 9864) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -64, v1 // (op 9939) 
v_sub_u32 v85, v1, v84 // (op 9939) 

v_and_b32 v84, 4294967232, v1 // (op 9939) 
v_add_u32 v86, v84, v85 // (op 9939) 


v_and_b32 v84, -4, v1 // (op 10015) 
v_sub_u32 v85, v1, v84 // (op 10015) 

v_lshrrev_b32 v84, 2, v1 // (op 10015) 
v_lshl_add_u32 v87, v84, 3, v85 // (op 10015) 


v_and_b32 v84, -4, v1 // (op 10130) 
v_sub_u32 v85, v1, v84 // (op 10130) 

v_lshlrev_b32 v84, 4, v85 // (op 10130) 

v_lshrrev_b32 v85, 2, v1 // (op 10130) 
v_lshl_add_u32 v88, v85, 6, v84 // (op 10130) 


s_nop 0
v_readfirstlane_b32 s7, v88 // (op 10130) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -8, v1 // (op 10174) 
v_sub_u32 v85, v1, v84 // (op 10174) 

v_lshlrev_b32 v84, 4, v85 // (op 10174) 

v_lshrrev_b32 v85, 3, v1 // (op 10174) 
v_lshl_add_u32 v88, v85, 7, v84 // (op 10174) 


s_nop 0
v_readfirstlane_b32 s8, v88 // (op 10174) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -64, v1 // (op 10218) 
v_sub_u32 v85, v1, v84 // (op 10218) 

v_and_b32 v84, 4294967232, v1 // (op 10218) 
v_add_u32 v88, v84, v85 // (op 10218) 


v_and_b32 v84, -4, v1 // (op 10247) 
v_sub_u32 v85, v1, v84 // (op 10247) 

v_lshrrev_b32 v84, 2, v1 // (op 10247) 
v_lshl_add_u32 v89, v84, 3, v85 // (op 10247) 


s_mov_b32 s1, 0 // (op 1325) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1324) 
s_mov_b32 s28, s18 // (op 1324) convert

s_and_b32 s29, s28, 4294967294 // (op 1324) 

s_cmp_lt_u32 s1, s29 // (op 1324) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoop_1324 // (op 1324) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoop_1324 if false)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoop_1324:
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:75], v[36:43], a[0:3], v26, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 194) 
s_barrier  // (op 10683) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:75], v[44:51], a[4:7], v26, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 654) 
s_add_u32 s1, s1, 2 // (op 1326) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[68:75], v[52:59], a[8:11], v26, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 830) 
s_lshl_b64 s[18:19], s[34:35], 5 // (op 8746) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:75], v[60:67], a[12:15], v26, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1006) 
v_mov_b64 v[84:85], s[18:19] // (op 8746) convert


v_mov_b64 v[90:91], v[24:25] // (op 8746) 
v_mov_b64 v[92:93], v[90:91] // (op 8746) 
s_mov_b32 m0, s6 // (op 8746) reg expression
s_mov_b32 s18, 256 // (op 8746) call()
buffer_load_dwordx4 v92, s[72:75], s18 offen offset: 0 lds // (op 8746) Load value direct to lds
v_add_co_u32 v92, s[28:29], v92, v84 // (op 8746) least significant half
v_addc_co_u32 v93, s[28:29], 0, v93, s[28:29] // (op 8746) most significant half

s_add_u32 m0, m0, 4096 // (op 8746) 
buffer_load_dwordx4 v92, s[72:75], s18 offen offset: 0 lds // (op 8746) Load value direct to lds
v_add_co_u32 v92, s[28:29], v92, v84 // (op 8746) least significant half
v_addc_co_u32 v93, s[28:29], 0, v93, s[28:29] // (op 8746) most significant half

s_add_u32 m0, m0, 4096 // (op 8746) 
buffer_load_dwordx4 v92, s[72:75], s18 offen offset: 0 lds // (op 8746) Load value direct to lds
v_add_co_u32 v92, s[28:29], v92, v84 // (op 8746) least significant half
v_addc_co_u32 v93, s[28:29], 0, v93, s[28:29] // (op 8746) most significant half

s_add_u32 m0, m0, 4096 // (op 8746) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:83], v[36:43], a[16:19], v27, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1501) 
buffer_load_dwordx4 v92, s[72:75], s18 offen offset: 0 lds // (op 8746) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:83], v[44:51], a[20:23], v27, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1455) 
v_mov_b64 v[84:85], s[44:45] // (op 200) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:83], v[52:59], a[24:27], v27, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1409) 
v_mov_b32 v92, v31 // (op 200) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:83], v[60:67], a[28:31], v27, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1363) 
v_mov_b32 v93, v92 // (op 200) 
buffer_load_ubyte v34, v93, s[80:83], 0 offen offset:8 // (op 200) Load value
v_add_u32 v93, v93, v84 // (op 200) 
buffer_load_ubyte v35, v93, s[80:83], 0 offen offset:8 // (op 200) Load value


s_barrier  // (op 10664) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 8741) 
v_mov_b64 v[84:85], s[18:19] // (op 8741) convert


v_mov_b64 v[94:95], v[20:21] // (op 8741) 
s_lshl_b64 s[18:19], s[30:31], 7 // (op 8741) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 8741) 

v_mov_b32 v93, s28 // (op 8741) 
v_mov_b32 v96, s29 // (op 8741) 
v_add_co_u32 v98, s[18:19], v94, v93 // (op 8741) least significant half
v_addc_co_u32 v99, s[18:19], v95, v96, s[18:19] // (op 8741) most significant half




s_add_u32 s18, 16384, s15 // (op 8741) 
s_nop 0
s_mov_b32 m0, s18 // (op 8741) reg expression
buffer_load_dwordx4 v98, s[68:71], 0 offen offset: 0 lds // (op 8741) Load value direct to lds
v_add_co_u32 v98, s[28:29], v98, v84 // (op 8741) least significant half
v_addc_co_u32 v99, s[28:29], 0, v99, s[28:29] // (op 8741) most significant half

s_add_u32 m0, m0, 4096 // (op 8741) 
buffer_load_dwordx4 v98, s[68:71], 0 offen offset: 0 lds // (op 8741) Load value direct to lds





v_mov_b32 v93, v28 // (op 199) 
s_lshl_b32 s18, s38, 3 // (op 199) 
v_add_u32 v84, v93, s18 // (op 199) 


buffer_load_ubyte v29, v84, s[76:79], 0 offen offset: 0 // (op 199) Load value

s_barrier  // (op 10767) 



s_waitcnt vmcnt(13)
ds_write_b8 v89, v32 offset:49920 // (op 533) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v89, v33 offset:49924 // (op 533) Store local data 
s_barrier  // (op 10761) 



s_waitcnt vmcnt(9)
ds_write_b8 v88, v30 offset:50432 // (op 532) Store local data 
s_barrier  // (op 10747) 



ds_read_b128 v[96:99], v12 offset:24576 // (op 525) Load local data 
ds_read_b128 v[100:103], v12 offset:24640 // (op 525) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10764) 



ds_read_u8 v84, v13 offset:49920 // (op 529) Load local data 



ds_read_b128 v[104:107], v12 offset:28672 // (op 659) Load local data 
ds_read_b128 v[108:111], v12 offset:28736 // (op 659) Load local data 



ds_read_u8 v85, v13 offset:50048 // (op 663) Load local data 



ds_read_b128 v[112:115], v12 offset:32768 // (op 835) Load local data 
ds_read_b128 v[116:119], v12 offset:32832 // (op 835) Load local data 



ds_read_u8 v120, v13 offset:50176 // (op 839) Load local data 



ds_read_b128 v[122:125], v12 offset:36864 // (op 1011) Load local data 
ds_read_b128 v[126:129], v12 offset:36928 // (op 1011) Load local data 



ds_read_u8 v121, v13 offset:50304 // (op 1015) Load local data 
s_barrier  // (op 10738) 
s_barrier  // (op 10878) 



ds_read_b64_tr_b8 v[68:69], v10 offset:40960 // (op 523) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v10 offset:41472 // (op 523) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v10 offset:45056 // (op 523) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v10 offset:45568 // (op 523) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 10757) 


ds_read_u8 v26, v11 offset:50432 // (op 527) Load local data 



ds_read_b64_tr_b8 v[76:77], v10 offset:40992 // (op 1366) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v10 offset:41504 // (op 1366) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v10 offset:45088 // (op 1366) Transpose load local data 
ds_read_b64_tr_b8 v[82:83], v10 offset:45600 // (op 1366) Transpose load local data 


ds_read_u8 v27, v11 offset:50464 // (op 1369) Load local data 
s_barrier  // (op 10904) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:75], v[96:103], a[0:3], v26, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 530) 
s_barrier  // (op 10754) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:75], v[104:111], a[4:7], v26, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 655) 
s_lshl_b64 s[18:19], s[34:35], 5 // (op 8769) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[68:75], v[112:119], a[8:11], v26, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 831) 
v_mov_b64 v[130:131], s[18:19] // (op 8769) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:75], v[122:129], a[12:15], v26, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1007) 
v_mov_b64 v[132:133], v[90:91] // (op 8769) 
s_add_u32 s18, 24576, s8 // (op 8769) 
s_mov_b32 m0, s18 // (op 8769) reg expression
s_mov_b32 s19, 384 // (op 8769) call()
buffer_load_dwordx4 v132, s[72:75], s19 offen offset: 0 lds // (op 8769) Load value direct to lds
v_add_co_u32 v132, s[28:29], v132, v130 // (op 8769) least significant half
v_addc_co_u32 v133, s[28:29], 0, v133, s[28:29] // (op 8769) most significant half

s_add_u32 m0, m0, 4096 // (op 8769) 
buffer_load_dwordx4 v132, s[72:75], s19 offen offset: 0 lds // (op 8769) Load value direct to lds
v_add_co_u32 v132, s[28:29], v132, v130 // (op 8769) least significant half
v_addc_co_u32 v133, s[28:29], 0, v133, s[28:29] // (op 8769) most significant half

s_add_u32 m0, m0, 4096 // (op 8769) 
buffer_load_dwordx4 v132, s[72:75], s19 offen offset: 0 lds // (op 8769) Load value direct to lds
v_add_co_u32 v132, s[28:29], v132, v130 // (op 8769) least significant half
v_addc_co_u32 v133, s[28:29], 0, v133, s[28:29] // (op 8769) most significant half

s_add_u32 m0, m0, 4096 // (op 8769) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:83], v[96:103], a[16:19], v27, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1502) 
buffer_load_dwordx4 v132, s[72:75], s19 offen offset: 0 lds // (op 8769) Load value direct to lds








v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:83], v[104:111], a[20:23], v27, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1456) 
v_add_co_u32 v24, vcc, 256, v24 // (op 9042) least significant half


v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 9042) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:83], v[112:119], a[24:27], v27, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1410) 
v_mov_b64 v[84:85], s[44:45] // (op 536) convert



v_mov_b32 v90, v92 // (op 536) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:83], v[122:129], a[28:31], v27, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1364) 
buffer_load_ubyte v32, v90, s[80:83], 0 offen offset:12 // (op 536) Load value


v_add_u32 v90, v90, v84 // (op 536) 
buffer_load_ubyte v33, v90, s[80:83], 0 offen offset:12 // (op 536) Load value



s_barrier  // (op 10744) 
v_add_u32 v31, 8, v31 // (op 9298) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 8764) 
v_mov_b64 v[84:85], s[18:19] // (op 8764) convert


s_lshl_b64 s[18:19], s[30:31], 7 // (op 8764) 
s_mul_i32 s29, 3, s19 // (op 8764) most significant: low of low * high
s_mul_i32 s28, 3, s18 // (op 8764) least significant
s_mul_hi_u32 s48, 3, s18 // (op 8764) most significant: high of low * low
s_add_u32 s29, s29, s48 // (op 8764) most significant: sum


v_mov_b32 v90, s28 // (op 8764) 
v_mov_b32 v91, s29 // (op 8764) 
v_add_co_u32 v96, s[18:19], v94, v90 // (op 8764) least significant half
v_addc_co_u32 v97, s[18:19], v95, v91, s[18:19] // (op 8764) most significant half




s_add_u32 s18, 40960, s7 // (op 8764) 
s_nop 0
s_mov_b32 m0, s18 // (op 8764) reg expression
buffer_load_dwordx4 v96, s[68:71], 0 offen offset: 0 lds // (op 8764) Load value direct to lds
v_add_co_u32 v96, s[28:29], v96, v84 // (op 8764) least significant half
v_addc_co_u32 v97, s[28:29], 0, v97, s[28:29] // (op 8764) most significant half

s_add_u32 m0, m0, 4096 // (op 8764) 
buffer_load_dwordx4 v96, s[68:71], 0 offen offset: 0 lds // (op 8764) Load value direct to lds






s_lshl_b32 s18, s38, 2 // (op 535) 
s_mul_i32 s19, 3, s18 // (op 535) 

v_add_u32 v84, v93, s19 // (op 535) 


buffer_load_ubyte v30, v84, s[76:79], 0 offen offset: 0 // (op 535) Load value

s_lshl_b64 s[18:19], s[30:31], 7 // (op 8904) 

s_lshl_b64 s[28:29], s[18:19], 1 // (op 8904) 

s_barrier  // (op 10724) 
s_lshl_b32 s18, s38, 3 // (op 9170) 



v_add_u32 v28, v28, s18 // (op 9170) 

v_mov_b32 v84, s28 // (op 8904) 
v_mov_b32 v85, s29 // (op 8904) 
v_add_co_u32 v20, s[18:19], v20, v84 // (op 8904) least significant half
v_addc_co_u32 v21, s[18:19], v21, v85, s[18:19] // (op 8904) most significant half




s_waitcnt vmcnt(13)
ds_write_b8 v87, v34 offset:49152 // (op 197) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v87, v35 offset:49156 // (op 197) Store local data 
s_barrier  // (op 10703) 



s_waitcnt vmcnt(9)
ds_write_b8 v86, v29 offset:49664 // (op 196) Store local data 
s_barrier  // (op 10689) 



ds_read_b128 v[36:39], v8 // (op 189) Load local data 
ds_read_b128 v[40:43], v8 offset:64 // (op 189) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10731) 



ds_read_u8 v16, v9 offset:49152 // (op 193) Load local data 



ds_read_b128 v[44:47], v8 offset:4096 // (op 649) Load local data 
ds_read_b128 v[48:51], v8 offset:4160 // (op 649) Load local data 



ds_read_u8 v17, v9 offset:49280 // (op 653) Load local data 



ds_read_b128 v[52:55], v8 offset:8192 // (op 825) Load local data 
ds_read_b128 v[56:59], v8 offset:8256 // (op 825) Load local data 



ds_read_u8 v22, v9 offset:49408 // (op 829) Load local data 



ds_read_b128 v[60:63], v8 offset:12288 // (op 1001) Load local data 
ds_read_b128 v[64:67], v8 offset:12352 // (op 1001) Load local data 



ds_read_u8 v23, v9 offset:49536 // (op 1005) Load local data 
s_barrier  // (op 10670) 



ds_read_b64_tr_b8 v[68:69], v6 offset:16384 // (op 187) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v6 offset:16896 // (op 187) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v6 offset:20480 // (op 187) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v6 offset:20992 // (op 187) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 10710) 


ds_read_u8 v26, v7 offset:49664 // (op 191) Load local data 



ds_read_b64_tr_b8 v[76:77], v6 offset:16416 // (op 1358) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v6 offset:16928 // (op 1358) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v6 offset:20512 // (op 1358) Transpose load local data 
ds_read_b64_tr_b8 v[82:83], v6 offset:21024 // (op 1358) Transpose load local data 


ds_read_u8 v27, v7 offset:49696 // (op 1361) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1324) 
s_mov_b32 s28, s18 // (op 1324) convert

s_and_b32 s29, s28, 4294967294 // (op 1324) 

s_cmp_lt_u32 s1, s29 // (op 1324) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoop_1324 // (op 1324) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoop_1324 if true)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoop_1324:


















s_barrier  // (op 10894) 
v_and_b32 v31, -4, v1 // (op 8836) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 8836) 
v_sub_u32 v16, v1, v31 // (op 8836) 

s_and_b32 s15, s18, 4294967294 // (op 8836) 
v_lshlrev_b32 v17, 4, v16 // (op 8836) 

s_add_u32 s6, s0, s15 // (op 8836) 

v_lshrrev_b32 v16, 2, v1 // (op 8836) 
s_mul_i32 s15, s16, s50 // (op 8836) 
s_add_u32 s15, s15, s6 // (op 8836) 

s_add_u32 s6, s15, s9 // (op 8836) 

s_mul_hi_u32 s15, s6, s13 // (op 8836) 
s_sub_u32 s7, s6, s15 // (op 8836) 
s_lshr_b32 s8, s7, 1 // (op 8836) 

s_add_u32 s7, s8, s15 // (op 8836) 


s_lshr_b32 s8, s7, s11 // (op 8836) 

s_and_b32 s7, s11, 2147483648 // (op 8836) 
s_cmp_eq_i32 s7, 0 // (op 8836) 
s_mov_b32 s15, scc // (op 8836) 

s_cmp_eq_i32 s15, 0 // (op 8836) 
s_cselect_b32 s7, s6, s8 // (op 8836) 


s_mul_hi_u32 s8, s7, s12 // (op 8836) 
s_sub_u32 s15, s7, s8 // (op 8836) 
s_lshr_b32 s28, s15, 1 // (op 8836) 

s_add_u32 s15, s28, s8 // (op 8836) 


s_lshr_b32 s8, s15, s10 // (op 8836) 

s_and_b32 s15, s10, 2147483648 // (op 8836) 
s_cmp_eq_i32 s15, 0 // (op 8836) 
s_mov_b32 s28, scc // (op 8836) 

s_cmp_eq_i32 s28, 0 // (op 8836) 
s_cselect_b32 s15, s7, s8 // (op 8836) 


s_mul_i32 s8, s7, s18 // (op 8836) 


v_lshl_add_u32 v31, s15, 6, v17 // (op 8836) 


s_sub_u32 s7, s6, s8 // (op 8836) 


v_mov_b32 v21, 0 // (op 8836) convert
v_mov_b32 v20, v31 // (op 8836) convert

v_lshl_add_u32 v17, s7, 7, v16 // (op 8836) 


v_mul_lo_u32 v25, v17, s31 // (op 8836) most significant: low of low * high
v_mul_hi_u32 v16, v17, s30 // (op 8836) most significant: high of low * low
v_mul_lo_u32 v24, v17, s30 // (op 8836) least significant: low of low * low
v_add_u32 v25, v25, v16 // (op 8836) most significant: sum


v_add_co_u32 v16, s[18:19], v20, v24 // (op 8836) least significant half
v_addc_co_u32 v17, s[18:19], v21, v25, s[18:19] // (op 8836) most significant half



s_mul_i32 s19, s30, s3 // (op 8838) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 8838) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 8838) least significant
s_mul_hi_u32 s6, s30, s2 // (op 8838) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 8838) most significant: sum
s_add_u32 s19, s19, s6 // (op 8838) most significant: sum


s_mov_b32 s68, s32 // (op 8838) reg expression
s_mov_b32 s69, s33 // (op 8838) reg expression
s_mov_b32 s70, s18 // (op 8838) reg expression
s_mov_b32 s71, 131072 // (op 8838) call()

v_lshrrev_b32 v31, 3, v1 // (op 8986) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 8986) 
v_and_b32 v20, -8, v1 // (op 8986) 
s_and_b32 s15, s18, 4294967294 // (op 8986) 
v_sub_u32 v21, v1, v20 // (op 8986) 

s_add_u32 s6, s0, s15 // (op 8986) 

v_lshlrev_b32 v20, 4, v21 // (op 8986) 

s_mul_i32 s15, s16, s50 // (op 8986) 
s_add_u32 s15, s15, s6 // (op 8986) 

s_add_u32 s6, s15, s9 // (op 8986) 

s_mul_hi_u32 s15, s6, s13 // (op 8986) 
s_sub_u32 s7, s6, s15 // (op 8986) 
s_lshr_b32 s8, s7, 1 // (op 8986) 

s_add_u32 s7, s8, s15 // (op 8986) 


s_lshr_b32 s8, s7, s11 // (op 8986) 

s_and_b32 s7, s11, 2147483648 // (op 8986) 
s_cmp_eq_i32 s7, 0 // (op 8986) 
s_mov_b32 s15, scc // (op 8986) 

s_cmp_eq_i32 s15, 0 // (op 8986) 
s_cselect_b32 s7, s6, s8 // (op 8986) 


s_mul_hi_u32 s8, s7, s12 // (op 8986) 
s_sub_u32 s15, s7, s8 // (op 8986) 
s_lshr_b32 s28, s15, 1 // (op 8986) 

s_add_u32 s15, s28, s8 // (op 8986) 


s_lshr_b32 s8, s15, s10 // (op 8986) 

s_and_b32 s15, s10, 2147483648 // (op 8986) 
s_cmp_eq_i32 s15, 0 // (op 8986) 
s_mov_b32 s28, scc // (op 8986) 

s_cmp_eq_i32 s28, 0 // (op 8986) 
s_cselect_b32 s15, s7, s8 // (op 8986) 


s_mul_i32 s8, s15, s14 // (op 8986) 

s_sub_u32 s15, s7, s8 // (op 8986) 

s_mul_i32 s8, s7, s18 // (op 8986) 


v_lshl_add_u32 v21, s15, 7, v31 // (op 8986) 


s_sub_u32 s7, s6, s8 // (op 8986) 


v_mul_lo_u32 v25, v21, s35 // (op 8986) most significant: low of low * high
v_mul_hi_u32 v31, v21, s34 // (op 8986) most significant: high of low * low
v_mul_lo_u32 v24, v21, s34 // (op 8986) least significant: low of low * low
v_add_u32 v25, v25, v31 // (op 8986) most significant: sum


v_lshl_add_u32 v21, s7, 7, v20 // (op 8986) 


v_mov_b32 v29, 0 // (op 8986) convert
v_mov_b32 v28, v21 // (op 8986) convert

v_add_co_u32 v20, s[18:19], v28, v24 // (op 8986) least significant half
v_addc_co_u32 v21, s[18:19], v29, v25, s[18:19] // (op 8986) most significant half



s_mul_i32 s19, s34, s23 // (op 8988) most significant: low of low * high
s_mul_i32 s15, s35, s22 // (op 8988) most significant: low of high * low
s_mul_i32 s18, s34, s22 // (op 8988) least significant
s_mul_hi_u32 s6, s34, s22 // (op 8988) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 8988) most significant: sum
s_add_u32 s19, s19, s6 // (op 8988) most significant: sum


s_mov_b32 s72, s36 // (op 8988) reg expression
s_mov_b32 s73, s37 // (op 8988) reg expression
s_mov_b32 s74, s18 // (op 8988) reg expression
s_mov_b32 s75, 131072 // (op 8988) call()

v_and_b32 v31, -64, v1 // (op 9114) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 9114) 
v_sub_u32 v24, v1, v31 // (op 9114) 

s_and_b32 s15, s18, 4294967294 // (op 9114) 
v_lshrrev_b32 v25, 6, v1 // (op 9114) 
s_add_u32 s6, s0, s15 // (op 9114) 

s_mul_i32 s15, s16, s50 // (op 9114) 
s_add_u32 s15, s15, s6 // (op 9114) 

s_add_u32 s6, s15, s9 // (op 9114) 

s_mul_hi_u32 s15, s6, s13 // (op 9114) 
s_sub_u32 s7, s6, s15 // (op 9114) 
s_lshr_b32 s8, s7, 1 // (op 9114) 

s_add_u32 s7, s8, s15 // (op 9114) 


s_lshr_b32 s8, s7, s11 // (op 9114) 

s_and_b32 s7, s11, 2147483648 // (op 9114) 
s_cmp_eq_i32 s7, 0 // (op 9114) 
s_mov_b32 s15, scc // (op 9114) 

s_cmp_eq_i32 s15, 0 // (op 9114) 
s_cselect_b32 s7, s6, s8 // (op 9114) 


s_mul_hi_u32 s8, s7, s12 // (op 9114) 
s_sub_u32 s15, s7, s8 // (op 9114) 
s_lshr_b32 s28, s15, 1 // (op 9114) 

s_add_u32 s15, s28, s8 // (op 9114) 


s_lshr_b32 s8, s15, s10 // (op 9114) 

s_and_b32 s15, s10, 2147483648 // (op 9114) 
s_cmp_eq_i32 s15, 0 // (op 9114) 
s_mov_b32 s28, scc // (op 9114) 

s_cmp_eq_i32 s28, 0 // (op 9114) 
s_cselect_b32 s15, s7, s8 // (op 9114) 


s_mul_i32 s8, s7, s18 // (op 9114) 


v_lshl_add_u32 v31, s15, 6, v24 // (op 9114) 


s_sub_u32 s7, s6, s8 // (op 9114) 


v_lshl_add_u32 v24, s7, 2, v25 // (op 9114) 


v_mul_lo_u32 v25, v24, s38 // (op 9114) 

v_add_u32 v24, v31, v25 // (op 9114) 


s_mul_i32 s19, s38, s43 // (op 9116) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 9116) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 9116) least significant
s_mul_hi_u32 s6, s38, s42 // (op 9116) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9116) most significant: sum
s_add_u32 s19, s19, s6 // (op 9116) most significant: sum


s_mov_b32 s76, s40 // (op 9116) reg expression
s_mov_b32 s77, s41 // (op 9116) reg expression
s_mov_b32 s78, s18 // (op 9116) reg expression
s_mov_b32 s79, 131072 // (op 9116) call()

v_lshrrev_b32 v25, 2, v1 // (op 9242) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 9242) 
v_lshlrev_b32 v31, 1, v25 // (op 9242) 

s_and_b32 s15, s18, 4294967294 // (op 9242) 
v_and_b32 v25, -4, v1 // (op 9242) 
s_add_u32 s6, s0, s15 // (op 9242) 

v_sub_u32 v28, v1, v25 // (op 9242) 

s_mul_i32 s15, s16, s50 // (op 9242) 
s_add_u32 s15, s15, s6 // (op 9242) 

s_add_u32 s6, s15, s9 // (op 9242) 

s_mul_hi_u32 s15, s6, s13 // (op 9242) 
s_sub_u32 s7, s6, s15 // (op 9242) 
s_lshr_b32 s8, s7, 1 // (op 9242) 

s_add_u32 s7, s8, s15 // (op 9242) 


s_lshr_b32 s8, s7, s11 // (op 9242) 

s_and_b32 s7, s11, 2147483648 // (op 9242) 
s_cmp_eq_i32 s7, 0 // (op 9242) 
s_mov_b32 s15, scc // (op 9242) 

s_cmp_eq_i32 s15, 0 // (op 9242) 
s_cselect_b32 s7, s6, s8 // (op 9242) 


s_mul_hi_u32 s8, s7, s12 // (op 9242) 
s_sub_u32 s15, s7, s8 // (op 9242) 
s_lshr_b32 s28, s15, 1 // (op 9242) 

s_add_u32 s15, s28, s8 // (op 9242) 


s_lshr_b32 s8, s15, s10 // (op 9242) 

s_and_b32 s15, s10, 2147483648 // (op 9242) 
s_cmp_eq_i32 s15, 0 // (op 9242) 
s_mov_b32 s28, scc // (op 9242) 

s_cmp_eq_i32 s28, 0 // (op 9242) 
s_cselect_b32 s15, s7, s8 // (op 9242) 


s_mul_i32 s8, s15, s14 // (op 9242) 

s_sub_u32 s15, s7, s8 // (op 9242) 

v_lshl_add_u32 v25, s15, 7, v31 // (op 9242) 


s_mul_i32 s8, s7, s18 // (op 9242) 


v_mul_lo_u32 v29, v25, s44 // (op 9242) 

s_sub_u32 s7, s6, s8 // (op 9242) 


v_lshl_add_u32 v25, s7, 2, v28 // (op 9242) 


v_add_u32 v28, v25, v29 // (op 9242) 


s_mul_i32 s19, s44, s23 // (op 9244) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 9244) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 9244) least significant
s_mul_hi_u32 s6, s44, s22 // (op 9244) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9244) most significant: sum
s_add_u32 s19, s19, s6 // (op 9244) most significant: sum


s_mov_b32 s80, s46 // (op 9244) reg expression
s_mov_b32 s81, s47 // (op 9244) reg expression
s_mov_b32 s82, s18 // (op 9244) reg expression
s_mov_b32 s83, 131072 // (op 9244) call()

v_and_b32 v25, -4, v1 // (op 9774) 
v_sub_u32 v29, v1, v25 // (op 9774) 

v_lshlrev_b32 v25, 4, v29 // (op 9774) 

v_lshrrev_b32 v29, 2, v1 // (op 9774) 
v_lshl_add_u32 v31, v29, 6, v25 // (op 9774) 


s_nop 0
v_readfirstlane_b32 s15, v31 // (op 9774) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -8, v1 // (op 9851) 
v_sub_u32 v29, v1, v25 // (op 9851) 

v_lshlrev_b32 v25, 4, v29 // (op 9851) 

v_lshrrev_b32 v29, 3, v1 // (op 9851) 
v_lshl_add_u32 v31, v29, 7, v25 // (op 9851) 


s_nop 0
v_readfirstlane_b32 s6, v31 // (op 9851) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -64, v1 // (op 9926) 
v_sub_u32 v29, v1, v25 // (op 9926) 

v_and_b32 v25, 4294967232, v1 // (op 9926) 
v_add_u32 v31, v25, v29 // (op 9926) 


v_and_b32 v25, -4, v1 // (op 10002) 
v_sub_u32 v29, v1, v25 // (op 10002) 

v_lshrrev_b32 v25, 2, v1 // (op 10002) 
v_lshl_add_u32 v34, v25, 3, v29 // (op 10002) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1299) 
s_and_b32 s7, s18, 4294967294 // (op 1299) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1298) 
s_mov_b32 s8, s18 // (op 1298) convert

s_cmp_lt_u32 s7, s8 // (op 1298) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoopTail_1298 // (op 1298) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoopTail_1298 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoopTail_1298:
s_barrier  // (op 10667) 
s_add_u32 s7, s7, 1 // (op 1300) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 8751) 
v_mov_b64 v[22:23], s[18:19] // (op 8751) convert


v_mov_b64 v[26:27], v[16:17] // (op 8751) 
v_mov_b64 v[32:33], v[26:27] // (op 8751) reg expression
s_add_u32 s8, 16384, s15 // (op 8751) 
s_mov_b32 m0, s8 // (op 8751) reg expression
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 8751) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v22 // (op 8751) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 8751) most significant half

s_add_u32 m0, m0, 4096 // (op 8751) 
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 8751) Load value direct to lds




s_lshl_b64 s[18:19], s[30:31], 7 // (op 8826) 
v_mov_b32 v22, s18 // (op 8826) 
v_mov_b32 v23, s19 // (op 8826) 
v_add_co_u32 v16, s[28:29], v16, v22 // (op 8826) least significant half
v_addc_co_u32 v17, s[28:29], v17, v23, s[28:29] // (op 8826) most significant half




s_barrier  // (op 10686) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 8757) 
v_mov_b64 v[22:23], s[18:19] // (op 8757) convert


v_mov_b64 v[26:27], v[20:21] // (op 8757) 
v_mov_b64 v[32:33], v[26:27] // (op 8757) 
s_mov_b32 m0, s6 // (op 8757) reg expression
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 8757) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v22 // (op 8757) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 8757) most significant half

s_add_u32 m0, m0, 4096 // (op 8757) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 8757) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v22 // (op 8757) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 8757) most significant half

s_add_u32 m0, m0, 4096 // (op 8757) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 8757) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v22 // (op 8757) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 8757) most significant half

s_add_u32 m0, m0, 4096 // (op 8757) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 8757) Load value direct to lds



v_add_co_u32 v20, vcc, 128, v20 // (op 8978) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 8978) most significant half


v_mov_b32 v22, v24 // (op 469) 
v_mov_b32 v23, v22 // (op 469) reg expression

buffer_load_ubyte v25, v23, s[76:79], 0 offen offset: 0 // (op 469) Load value


s_lshl_b32 s8, s38, 2 // (op 9106) 
v_add_u32 v24, v24, s8 // (op 9106) 


s_barrier  // (op 10706) 
v_mov_b64 v[22:23], s[44:45] // (op 470) convert




v_mov_b32 v26, v28 // (op 470) 
v_mov_b32 v27, v26 // (op 470) 
buffer_load_ubyte v32, v27, s[80:83], 0 offen offset: 0 // (op 470) Load value
v_add_u32 v27, v27, v22 // (op 470) 
buffer_load_ubyte v33, v27, s[80:83], 0 offen offset: 0 // (op 470) Load value



v_add_u32 v28, 4, v28 // (op 9234) 
s_waitcnt vmcnt(2)
ds_write_b8 v31, v25 offset:49664 // (op 466) Store local data 

s_barrier  // (op 10673) 



s_barrier  // (op 10727) 
ds_read_b64_tr_b8 v[36:37], v6 offset:16384 // (op 457) Transpose load local data 



ds_read_b64_tr_b8 v[38:39], v6 offset:16896 // (op 457) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v6 offset:20480 // (op 457) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v6 offset:20992 // (op 457) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v34, v32 offset:49152 // (op 467) Store local data 
s_barrier  // (op 10693) 
s_waitcnt vmcnt(0)
ds_write_b8 v34, v33 offset:49156 // (op 467) Store local data 




ds_read_b128 v[44:47], v8 // (op 459) Load local data 
ds_read_b128 v[48:51], v8 offset:64 // (op 459) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 10714) 


ds_read_u8 v35, v7 offset:49664 // (op 461) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10734) 



ds_read_u8 v22, v9 offset:49152 // (op 463) Load local data 



ds_read_b128 v[52:55], v8 offset:4096 // (op 628) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:51], a[0:3], v35, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 464) 
ds_read_b128 v[56:59], v8 offset:4160 // (op 628) Load local data 



ds_read_u8 v23, v9 offset:49280 // (op 632) Load local data 



ds_read_b128 v[60:63], v8 offset:8192 // (op 804) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:43], v[52:59], a[4:7], v35, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 633) 
ds_read_b128 v[64:67], v8 offset:8256 // (op 804) Load local data 



ds_read_u8 v25, v9 offset:49408 // (op 808) Load local data 



ds_read_b128 v[68:71], v8 offset:12288 // (op 980) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[36:43], v[60:67], a[8:11], v35, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 809) 
ds_read_b128 v[72:75], v8 offset:12352 // (op 980) Load local data 



ds_read_u8 v26, v9 offset:49536 // (op 984) Load local data 



ds_read_b64_tr_b8 v[76:77], v6 offset:16416 // (op 1242) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[36:43], v[68:75], a[12:15], v35, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 985) 
ds_read_b64_tr_b8 v[78:79], v6 offset:16928 // (op 1242) Transpose load local data 


ds_read_b64_tr_b8 v[80:81], v6 offset:20512 // (op 1242) Transpose load local data 
ds_read_b64_tr_b8 v[82:83], v6 offset:21024 // (op 1242) Transpose load local data 


ds_read_u8 v27, v7 offset:49696 // (op 1245) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:83], v[44:51], a[16:19], v27, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1247) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:83], v[52:59], a[20:23], v27, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1278) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:83], v[60:67], a[24:27], v27, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1309) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:83], v[68:75], a[28:31], v27, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1340) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 1298) 
s_mov_b32 s8, s18 // (op 1298) convert

s_cmp_lt_u32 s7, s8 // (op 1298) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoopTail_1298 // (op 1298) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_KLoopTail_1298 if true)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_KLoopTail_1298:









s_barrier  // (op 10886) 




v_cmp_eq_f32 vcc, 0.00000, v18 // (op 234) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_0_beta_234 // (op 234) Condition: False, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_0_beta_234
v_accvgpr_read v14, a0 // (op 242) 
v_accvgpr_read v15, a1 // (op 242) 
v_accvgpr_read v16, a2 // (op 242) 
v_accvgpr_read v17, a3 // (op 242) 
v_mul_f32 v20, v19, v14 // (op 242) 
v_mul_f32 v21, v19, v15 // (op 242) 
v_mul_f32 v22, v19, v16 // (op 242) 
v_mul_f32 v23, v19, v17 // (op 242) 



v_mov_b32 v0, v5 // (op 244) reg expression

buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 244) Store value


v_accvgpr_read v14, a4 // (op 2147) 
v_accvgpr_read v15, a5 // (op 2147) 
v_accvgpr_read v16, a6 // (op 2147) 
v_accvgpr_read v17, a7 // (op 2147) 
v_mul_f32 v20, v19, v14 // (op 2147) 
v_mul_f32 v21, v19, v15 // (op 2147) 
v_mul_f32 v22, v19, v16 // (op 2147) 
v_mul_f32 v23, v19, v17 // (op 2147) 



s_lshl_b32 s15, s24, 7 // (op 2149) 
v_add_u32 v0, v5, s15 // (op 2149) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 2149) Store value


v_accvgpr_read v14, a8 // (op 2159) 
v_accvgpr_read v15, a9 // (op 2159) 
v_accvgpr_read v16, a10 // (op 2159) 
v_accvgpr_read v17, a11 // (op 2159) 
v_mul_f32 v20, v19, v14 // (op 2159) 
v_mul_f32 v21, v19, v15 // (op 2159) 
v_mul_f32 v22, v19, v16 // (op 2159) 
v_mul_f32 v23, v19, v17 // (op 2159) 



s_lshl_b32 s15, s24, 8 // (op 2161) 
v_add_u32 v0, v5, s15 // (op 2161) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 2161) Store value


v_accvgpr_read v14, a12 // (op 2171) 
v_accvgpr_read v15, a13 // (op 2171) 
v_accvgpr_read v16, a14 // (op 2171) 
v_accvgpr_read v17, a15 // (op 2171) 
v_mul_f32 v20, v19, v14 // (op 2171) 
v_mul_f32 v21, v19, v15 // (op 2171) 
v_mul_f32 v22, v19, v16 // (op 2171) 
v_mul_f32 v23, v19, v17 // (op 2171) 



s_lshl_b32 s15, s24, 7 // (op 2173) 
s_mul_i32 s6, 3, s15 // (op 2173) 

v_add_u32 v0, v5, s6 // (op 2173) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 2173) Store value


v_accvgpr_read v14, a16 // (op 2205) 
v_accvgpr_read v15, a17 // (op 2205) 
v_accvgpr_read v16, a18 // (op 2205) 
v_accvgpr_read v17, a19 // (op 2205) 
v_mul_f32 v20, v19, v14 // (op 2205) 
v_mul_f32 v21, v19, v15 // (op 2205) 
v_mul_f32 v22, v19, v16 // (op 2205) 
v_mul_f32 v23, v19, v17 // (op 2205) 



v_add_u32 v0, 128, v5 // (op 2208) 

buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 2208) Store value


v_accvgpr_read v14, a20 // (op 2218) 
v_accvgpr_read v15, a21 // (op 2218) 
v_accvgpr_read v16, a22 // (op 2218) 
v_accvgpr_read v17, a23 // (op 2218) 
v_mul_f32 v20, v19, v14 // (op 2218) 
v_mul_f32 v21, v19, v15 // (op 2218) 
v_mul_f32 v22, v19, v16 // (op 2218) 
v_mul_f32 v23, v19, v17 // (op 2218) 



s_lshl_b32 s15, s24, 7 // (op 2210) 
s_add_u32 s15, s15, 128 // (op 2210) 
v_add_u32 v0, v5, s15 // (op 2210) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 2210) Store value


v_accvgpr_read v14, a24 // (op 2223) 
v_accvgpr_read v15, a25 // (op 2223) 
v_accvgpr_read v16, a26 // (op 2223) 
v_accvgpr_read v17, a27 // (op 2223) 
v_mul_f32 v20, v19, v14 // (op 2223) 
v_mul_f32 v21, v19, v15 // (op 2223) 
v_mul_f32 v22, v19, v16 // (op 2223) 
v_mul_f32 v23, v19, v17 // (op 2223) 



s_lshl_b32 s15, s24, 8 // (op 2212) 
s_add_u32 s15, s15, 128 // (op 2212) 
v_add_u32 v0, v5, s15 // (op 2212) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 2212) Store value


v_accvgpr_read v14, a28 // (op 2228) 
v_accvgpr_read v15, a29 // (op 2228) 
v_accvgpr_read v16, a30 // (op 2228) 
v_accvgpr_read v17, a31 // (op 2228) 
v_mul_f32 v20, v19, v14 // (op 2228) 
v_mul_f32 v21, v19, v15 // (op 2228) 
v_mul_f32 v22, v19, v16 // (op 2228) 
v_mul_f32 v23, v19, v17 // (op 2228) 



s_lshl_b32 s15, s24, 7 // (op 2214) 
s_mul_i32 s6, 3, s15 // (op 2214) 
s_add_u32 s6, s6, 128 // (op 2214) 

v_add_u32 v0, v5, s6 // (op 2214) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 2214) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_0_beta_234 // (op 234) Condition: Done, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_0_beta_234
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalFalse_0_beta_234:


v_add_u32 v0, 128, v4 // (op 2386) 

buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 2386) Load value

v_accvgpr_read v20, a16 // (op 2374) 
v_accvgpr_read v21, a17 // (op 2374) 
v_accvgpr_read v22, a18 // (op 2374) 
v_accvgpr_read v23, a19 // (op 2374) 
v_mul_f32 v24, v19, v20 // (op 2374) 
v_mul_f32 v25, v19, v21 // (op 2374) 
v_mul_f32 v26, v19, v22 // (op 2374) 
v_mul_f32 v27, v19, v23 // (op 2374) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 2385) 
v_mul_f32 v21, v18, v15 // (op 2385) 
v_mul_f32 v22, v18, v16 // (op 2385) 
v_mul_f32 v23, v18, v17 // (op 2385) 

v_add_f32 v14, v20, v24 // (op 2375) 
v_add_f32 v15, v21, v25 // (op 2375) 
v_add_f32 v16, v22, v26 // (op 2375) 
v_add_f32 v17, v23, v27 // (op 2375) 




v_add_u32 v0, 128, v5 // (op 2377) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 2377) Store value




s_lshl_b32 s15, s4, 7 // (op 2390) 
s_add_u32 s15, s15, 128 // (op 2390) 
v_add_u32 v0, v4, s15 // (op 2390) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 2390) Load value

v_accvgpr_read v20, a20 // (op 2400) 
v_accvgpr_read v21, a21 // (op 2400) 
v_accvgpr_read v22, a22 // (op 2400) 
v_accvgpr_read v23, a23 // (op 2400) 
v_mul_f32 v24, v19, v20 // (op 2400) 
v_mul_f32 v25, v19, v21 // (op 2400) 
v_mul_f32 v26, v19, v22 // (op 2400) 
v_mul_f32 v27, v19, v23 // (op 2400) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 2388) 
v_mul_f32 v21, v18, v15 // (op 2388) 
v_mul_f32 v22, v18, v16 // (op 2388) 
v_mul_f32 v23, v18, v17 // (op 2388) 

v_add_f32 v14, v20, v24 // (op 2389) 
v_add_f32 v15, v21, v25 // (op 2389) 
v_add_f32 v16, v22, v26 // (op 2389) 
v_add_f32 v17, v23, v27 // (op 2389) 




s_lshl_b32 s15, s24, 7 // (op 2379) 
s_add_u32 s15, s15, 128 // (op 2379) 
v_add_u32 v0, v5, s15 // (op 2379) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 2379) Store value




s_lshl_b32 s15, s4, 8 // (op 2394) 
s_add_u32 s15, s15, 128 // (op 2394) 
v_add_u32 v0, v4, s15 // (op 2394) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 2394) Load value

v_accvgpr_read v20, a24 // (op 2402) 
v_accvgpr_read v21, a25 // (op 2402) 
v_accvgpr_read v22, a26 // (op 2402) 
v_accvgpr_read v23, a27 // (op 2402) 
v_mul_f32 v24, v19, v20 // (op 2402) 
v_mul_f32 v25, v19, v21 // (op 2402) 
v_mul_f32 v26, v19, v22 // (op 2402) 
v_mul_f32 v27, v19, v23 // (op 2402) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 2392) 
v_mul_f32 v21, v18, v15 // (op 2392) 
v_mul_f32 v22, v18, v16 // (op 2392) 
v_mul_f32 v23, v18, v17 // (op 2392) 

v_add_f32 v14, v20, v24 // (op 2393) 
v_add_f32 v15, v21, v25 // (op 2393) 
v_add_f32 v16, v22, v26 // (op 2393) 
v_add_f32 v17, v23, v27 // (op 2393) 




s_lshl_b32 s15, s24, 8 // (op 2381) 
s_add_u32 s15, s15, 128 // (op 2381) 
v_add_u32 v0, v5, s15 // (op 2381) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 2381) Store value




s_lshl_b32 s15, s4, 7 // (op 2398) 
s_mul_i32 s6, 3, s15 // (op 2398) 
s_add_u32 s6, s6, 128 // (op 2398) 

v_add_u32 v0, v4, s6 // (op 2398) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 2398) Load value

v_accvgpr_read v20, a28 // (op 2404) 
v_accvgpr_read v21, a29 // (op 2404) 
v_accvgpr_read v22, a30 // (op 2404) 
v_accvgpr_read v23, a31 // (op 2404) 
v_mul_f32 v24, v19, v20 // (op 2404) 
v_mul_f32 v25, v19, v21 // (op 2404) 
v_mul_f32 v26, v19, v22 // (op 2404) 
v_mul_f32 v27, v19, v23 // (op 2404) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 2396) 
v_mul_f32 v21, v18, v15 // (op 2396) 
v_mul_f32 v22, v18, v16 // (op 2396) 
v_mul_f32 v23, v18, v17 // (op 2396) 

v_add_f32 v14, v20, v24 // (op 2397) 
v_add_f32 v15, v21, v25 // (op 2397) 
v_add_f32 v16, v22, v26 // (op 2397) 
v_add_f32 v17, v23, v27 // (op 2397) 




s_lshl_b32 s15, s24, 7 // (op 2383) 
s_mul_i32 s6, 3, s15 // (op 2383) 
s_add_u32 s6, s6, 128 // (op 2383) 

v_add_u32 v0, v5, s6 // (op 2383) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 2383) Store value




v_mov_b32 v0, v4 // (op 257) reg expression

s_nop 0
buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 257) Load value

v_accvgpr_read v20, a0 // (op 254) 
v_accvgpr_read v21, a1 // (op 254) 
v_accvgpr_read v22, a2 // (op 254) 
v_accvgpr_read v23, a3 // (op 254) 
v_mul_f32 v24, v19, v20 // (op 254) 
v_mul_f32 v25, v19, v21 // (op 254) 
v_mul_f32 v26, v19, v22 // (op 254) 
v_mul_f32 v27, v19, v23 // (op 254) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 258) 
v_mul_f32 v21, v18, v15 // (op 258) 
v_mul_f32 v22, v18, v16 // (op 258) 
v_mul_f32 v23, v18, v17 // (op 258) 

v_add_f32 v14, v20, v24 // (op 255) 
v_add_f32 v15, v21, v25 // (op 255) 
v_add_f32 v16, v22, v26 // (op 255) 
v_add_f32 v17, v23, v27 // (op 255) 




v_mov_b32 v0, v5 // (op 256) reg expression

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 256) Store value




s_lshl_b32 s15, s4, 7 // (op 2300) 
v_add_u32 v0, v4, s15 // (op 2300) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 2300) Load value

v_accvgpr_read v20, a4 // (op 2297) 
v_accvgpr_read v21, a5 // (op 2297) 
v_accvgpr_read v22, a6 // (op 2297) 
v_accvgpr_read v23, a7 // (op 2297) 
v_mul_f32 v24, v19, v20 // (op 2297) 
v_mul_f32 v25, v19, v21 // (op 2297) 
v_mul_f32 v26, v19, v22 // (op 2297) 
v_mul_f32 v27, v19, v23 // (op 2297) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 2301) 
v_mul_f32 v21, v18, v15 // (op 2301) 
v_mul_f32 v22, v18, v16 // (op 2301) 
v_mul_f32 v23, v18, v17 // (op 2301) 

v_add_f32 v14, v20, v24 // (op 2298) 
v_add_f32 v15, v21, v25 // (op 2298) 
v_add_f32 v16, v22, v26 // (op 2298) 
v_add_f32 v17, v23, v27 // (op 2298) 




s_lshl_b32 s15, s24, 7 // (op 2299) 
v_add_u32 v0, v5, s15 // (op 2299) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 2299) Store value




s_lshl_b32 s15, s4, 8 // (op 2312) 
v_add_u32 v0, v4, s15 // (op 2312) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 2312) Load value

v_accvgpr_read v20, a8 // (op 2309) 
v_accvgpr_read v21, a9 // (op 2309) 
v_accvgpr_read v22, a10 // (op 2309) 
v_accvgpr_read v23, a11 // (op 2309) 
v_mul_f32 v24, v19, v20 // (op 2309) 
v_mul_f32 v25, v19, v21 // (op 2309) 
v_mul_f32 v26, v19, v22 // (op 2309) 
v_mul_f32 v27, v19, v23 // (op 2309) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 2313) 
v_mul_f32 v21, v18, v15 // (op 2313) 
v_mul_f32 v22, v18, v16 // (op 2313) 
v_mul_f32 v23, v18, v17 // (op 2313) 

v_add_f32 v14, v20, v24 // (op 2310) 
v_add_f32 v15, v21, v25 // (op 2310) 
v_add_f32 v16, v22, v26 // (op 2310) 
v_add_f32 v17, v23, v27 // (op 2310) 




s_lshl_b32 s15, s24, 8 // (op 2311) 
v_add_u32 v0, v5, s15 // (op 2311) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 2311) Store value




s_lshl_b32 s15, s4, 7 // (op 2324) 
s_mul_i32 s6, 3, s15 // (op 2324) 

v_add_u32 v0, v4, s6 // (op 2324) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 2324) Load value

v_accvgpr_read v20, a12 // (op 2321) 
v_accvgpr_read v21, a13 // (op 2321) 
v_accvgpr_read v22, a14 // (op 2321) 
v_accvgpr_read v23, a15 // (op 2321) 
v_mul_f32 v24, v19, v20 // (op 2321) 
v_mul_f32 v25, v19, v21 // (op 2321) 
v_mul_f32 v26, v19, v22 // (op 2321) 
v_mul_f32 v27, v19, v23 // (op 2321) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 2325) 
v_mul_f32 v21, v18, v15 // (op 2325) 
v_mul_f32 v22, v18, v16 // (op 2325) 
v_mul_f32 v23, v18, v17 // (op 2325) 

v_add_f32 v14, v20, v24 // (op 2322) 
v_add_f32 v15, v21, v25 // (op 2322) 
v_add_f32 v16, v22, v26 // (op 2322) 
v_add_f32 v17, v23, v27 // (op 2322) 




s_lshl_b32 s15, s24, 7 // (op 2323) 
s_mul_i32 s6, 3, s15 // (op 2323) 

v_add_u32 v0, v5, s6 // (op 2323) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 2323) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ConditionalBottom_0_beta_234:
s_add_u32 s0, s0, s1 // (op 443) 


s_mul_i32 s15, s50, s16 // (op 441) 
s_cmp_gt_u32 s51, s15 // (op 441) 
s_mov_b32 s6, scc // (op 441) 
s_sub_u32 s7, s51, s15 // (op 441) 

s_cmp_eq_i32 s6, 0 // (op 441) 
s_cselect_b32 s8, 0, s7 // (op 441) 


s_cmp_lt_u32 s50, s8 // (op 441) 
s_mov_b32 s15, scc // (op 441) 
s_cmp_eq_i32 s15, 0 // (op 441) 
s_cselect_b32 s6, s8, s50 // (op 441) 


s_cmp_lt_u32 s0, s6 // (op 441) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_DPStreamTileLoop_441 // (op 441) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopTop_DPStreamTileLoop_441 if true)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_441:

















































s_endpgm  // End of RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2
.LRR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_end:
.size RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2, .LRR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2_end-RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: B8
    DestDataType: s
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
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
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
  - .name: RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2
    .symbol: RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x128x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 50688
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 89
    .vgpr_count: 144
    .agpr_count: 32
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_size_1_9
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
      - .name: Tensor_0_stride_1_8
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_1_11
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_14
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_15
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: Tensor_7_stride_1_17
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

