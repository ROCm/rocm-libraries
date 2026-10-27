// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2,@function
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 9642) 
v_and_b32 v3, 15, v1 // (op 9642) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 9642) 


v_and_b32 v0, 48, v1 // (op 9642) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 9642) 


v_lshrrev_b32 v0, 6, v1 // (op 9717) 
v_and_b32 v4, 1, v0 // (op 9717) 

v_and_b32 v0, 15, v1 // (op 9717) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 9717) 


v_and_b32 v0, 48, v1 // (op 9717) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 9717) 


v_lshrrev_b32 v0, 7, v1 // (op 9793) 
v_and_b32 v5, 15, v1 // (op 9793) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 9793) 


v_and_b32 v0, 63, v1 // (op 9793) 
v_lshrrev_b32 v5, 4, v0 // (op 9793) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 9793) 


v_lshrrev_b32 v5, 6, v1 // (op 9869) 
v_and_b32 v6, 1, v5 // (op 9869) 

v_and_b32 v5, 15, v1 // (op 9869) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 9869) 


v_and_b32 v5, 63, v1 // (op 9869) 
v_lshrrev_b32 v6, 4, v5 // (op 9869) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 9869) 


v_lshrrev_b32 v6, 7, v1 // (op 9996) 
v_and_b32 v7, 15, v1 // (op 9996) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 9996) 


v_and_b32 v6, 48, v1 // (op 9996) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 9996) 


v_lshrrev_b32 v6, 6, v1 // (op 10071) 
v_and_b32 v8, 1, v6 // (op 10071) 

v_and_b32 v6, 15, v1 // (op 10071) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 10071) 


v_and_b32 v6, 48, v1 // (op 10071) 
v_lshl_add_u32 v8, v9, 7, v6 // (op 10071) 


v_lshrrev_b32 v6, 7, v1 // (op 10147) 
v_and_b32 v9, 15, v1 // (op 10147) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 10147) 


v_and_b32 v6, 63, v1 // (op 10147) 
v_lshrrev_b32 v9, 4, v6 // (op 10147) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 10147) 


v_lshrrev_b32 v9, 6, v1 // (op 10223) 
v_and_b32 v10, 1, v9 // (op 10223) 

v_and_b32 v9, 15, v1 // (op 10223) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 10223) 


v_and_b32 v9, 63, v1 // (op 10223) 
v_lshrrev_b32 v10, 4, v9 // (op 10223) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 10223) 


v_lshrrev_b32 v10, 7, v1 // (op 10337) 
v_and_b32 v11, 15, v1 // (op 10337) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 10337) 


v_and_b32 v10, 48, v1 // (op 10337) 
v_lshl_add_u32 v11, v12, 7, v10 // (op 10337) 


v_lshrrev_b32 v10, 6, v1 // (op 10381) 
v_and_b32 v12, 1, v10 // (op 10381) 

v_and_b32 v10, 15, v1 // (op 10381) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 10381) 


v_and_b32 v10, 48, v1 // (op 10381) 
v_lshl_add_u32 v12, v13, 7, v10 // (op 10381) 


v_lshrrev_b32 v10, 7, v1 // (op 10410) 
v_and_b32 v13, 15, v1 // (op 10410) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 10410) 


v_and_b32 v10, 63, v1 // (op 10410) 
v_lshrrev_b32 v13, 4, v10 // (op 10410) 

v_lshl_add_u32 v10, v14, 2, v13 // (op 10410) 


v_lshrrev_b32 v13, 6, v1 // (op 10439) 
v_and_b32 v14, 1, v13 // (op 10439) 

v_and_b32 v13, 15, v1 // (op 10439) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 10439) 


v_and_b32 v13, 63, v1 // (op 10439) 
v_lshrrev_b32 v14, 4, v13 // (op 10439) 

v_lshl_add_u32 v13, v15, 2, v14 // (op 10439) 


v_lshrrev_b32 v14, 7, v1 // (op 10483) 
v_and_b32 v15, 15, v1 // (op 10483) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 10483) 


v_and_b32 v14, 48, v1 // (op 10483) 
v_lshl_add_u32 v15, v16, 7, v14 // (op 10483) 


v_lshrrev_b32 v14, 6, v1 // (op 10527) 
v_and_b32 v16, 1, v14 // (op 10527) 

v_and_b32 v14, 15, v1 // (op 10527) 
v_lshl_add_u32 v17, v16, 4, v14 // (op 10527) 


v_and_b32 v14, 48, v1 // (op 10527) 
v_lshl_add_u32 v16, v17, 7, v14 // (op 10527) 


v_lshrrev_b32 v14, 7, v1 // (op 10556) 
v_and_b32 v17, 15, v1 // (op 10556) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 10556) 


v_and_b32 v14, 63, v1 // (op 10556) 
v_lshrrev_b32 v17, 4, v14 // (op 10556) 

v_lshl_add_u32 v14, v18, 2, v17 // (op 10556) 


v_lshrrev_b32 v17, 6, v1 // (op 10585) 
v_and_b32 v18, 1, v17 // (op 10585) 

v_and_b32 v17, 15, v1 // (op 10585) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 10585) 


v_and_b32 v17, 63, v1 // (op 10585) 
v_lshrrev_b32 v18, 4, v17 // (op 10585) 

v_lshl_add_u32 v17, v19, 2, v18 // (op 10585) 


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
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_294 // (op 294) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_294 if false)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_SKStreamTileLoop_294:
v_lshrrev_b32 v20, 7, v1 // (op 9553) 
s_mul_i32 s52, s16, s8 // (op 9553) 
s_add_u32 s52, s52, s0 // (op 9553) 
v_and_b32 v21, 63, v1 // (op 9553) 
s_mul_hi_u32 s53, s52, s13 // (op 9553) 
v_lshrrev_b32 v22, 4, v21 // (op 9553) 
s_sub_u32 s54, s52, s53 // (op 9553) 
v_lshlrev_b32 v23, 2, v22 // (op 9553) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 9553) 


s_lshr_b32 s55, s54, 1 // (op 9553) 

s_add_u32 s54, s55, s53 // (op 9553) 


v_lshrrev_b32 v20, 2, v21 // (op 9553) 

s_lshr_b32 s53, s54, s11 // (op 9553) 

v_and_b32 v21, 3, v20 // (op 9553) 

s_and_b32 s54, s11, 2147483648 // (op 9553) 
v_and_b32 v20, 3, v1 // (op 9553) 
s_cmp_eq_i32 s54, 0 // (op 9553) 
s_mov_b32 s55, scc // (op 9553) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 9553) 


s_cmp_eq_i32 s55, 0 // (op 9553) 
s_cselect_b32 s54, s52, s53 // (op 9553) 



v_lshrrev_b32 v20, 6, v1 // (op 9553) 
s_mul_hi_u32 s52, s54, s12 // (op 9553) 
v_and_b32 v21, 1, v20 // (op 9553) 

s_sub_u32 s53, s54, s52 // (op 9553) 
v_lshl_add_u32 v20, v21, 4, v23 // (op 9553) 


s_lshr_b32 s55, s53, 1 // (op 9553) 

s_add_u32 s53, s55, s52 // (op 9553) 


s_lshr_b32 s52, s53, s10 // (op 9553) 

s_and_b32 s53, s10, 2147483648 // (op 9553) 
s_cmp_eq_i32 s53, 0 // (op 9553) 
s_mov_b32 s55, scc // (op 9553) 

s_cmp_eq_i32 s55, 0 // (op 9553) 
s_cselect_b32 s53, s54, s52 // (op 9553) 


s_mul_i32 s52, s53, s14 // (op 9553) 
v_lshl_add_u32 v21, s53, 7, v22 // (op 9553) 


s_sub_u32 s53, s54, s52 // (op 9553) 


v_lshl_add_u32 v22, s53, 6, v20 // (op 9553) 


v_mul_lo_u32 v20, v22, s4 // (op 9553) 

v_add_lshl_u32 v22, v21, v20, 1 // (op 9553) 


s_mul_i32 s53, s4, s23 // (op 9555) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 9555) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 9555) least significant
s_mul_hi_u32 s55, s4, s22 // (op 9555) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9555) most significant: sum
s_add_u32 s53, s53, s55 // (op 9555) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 9555) 

s_mov_b32 s56, s20 // (op 9555) reg expression
s_mov_b32 s57, s21 // (op 9555) reg expression
s_mov_b32 s58, s54 // (op 9555) reg expression
s_mov_b32 s59, 131072 // (op 9555) call()

v_lshrrev_b32 v20, 7, v1 // (op 9659) 
v_and_b32 v21, 15, v1 // (op 9659) 
v_lshl_add_u32 v23, v20, 4, v21 // (op 9659) 


v_and_b32 v20, 48, v1 // (op 9659) 
v_lshl_add_u32 v21, v23, 7, v20 // (op 9659) 


v_lshrrev_b32 v20, 6, v1 // (op 9734) 
v_and_b32 v23, 1, v20 // (op 9734) 

v_and_b32 v20, 15, v1 // (op 9734) 
v_lshl_add_u32 v24, v23, 4, v20 // (op 9734) 


v_and_b32 v20, 48, v1 // (op 9734) 
v_lshl_add_u32 v23, v24, 7, v20 // (op 9734) 


v_lshrrev_b32 v20, 7, v1 // (op 9810) 
v_and_b32 v24, 15, v1 // (op 9810) 
v_lshl_add_u32 v25, v20, 4, v24 // (op 9810) 


v_and_b32 v20, 63, v1 // (op 9810) 
v_lshrrev_b32 v24, 4, v20 // (op 9810) 

v_lshl_add_u32 v20, v25, 2, v24 // (op 9810) 


v_lshrrev_b32 v24, 6, v1 // (op 9886) 
v_and_b32 v25, 1, v24 // (op 9886) 

v_and_b32 v24, 15, v1 // (op 9886) 
v_lshl_add_u32 v26, v25, 4, v24 // (op 9886) 


v_and_b32 v24, 63, v1 // (op 9886) 
v_lshrrev_b32 v25, 4, v24 // (op 9886) 

v_lshl_add_u32 v24, v26, 2, v25 // (op 9886) 


v_lshrrev_b32 v25, 7, v1 // (op 9909) 
s_mul_i32 s52, s16, s8 // (op 9909) 
s_add_u32 s52, s52, s0 // (op 9909) 
v_and_b32 v26, 63, v1 // (op 9909) 
s_mul_hi_u32 s53, s52, s13 // (op 9909) 
v_lshrrev_b32 v27, 4, v26 // (op 9909) 
s_sub_u32 s54, s52, s53 // (op 9909) 
v_lshlrev_b32 v28, 2, v27 // (op 9909) 

s_lshr_b32 s55, s54, 1 // (op 9909) 

v_lshl_add_u32 v27, v25, 4, v28 // (op 9909) 


s_add_u32 s54, s55, s53 // (op 9909) 


v_lshrrev_b32 v25, 2, v26 // (op 9909) 

s_lshr_b32 s53, s54, s11 // (op 9909) 

v_and_b32 v26, 3, v25 // (op 9909) 

s_and_b32 s54, s11, 2147483648 // (op 9909) 
v_and_b32 v25, 3, v1 // (op 9909) 
s_cmp_eq_i32 s54, 0 // (op 9909) 
s_mov_b32 s55, scc // (op 9909) 

v_lshl_add_u32 v28, v26, 2, v25 // (op 9909) 


s_cmp_eq_i32 s55, 0 // (op 9909) 
s_cselect_b32 s54, s52, s53 // (op 9909) 



v_lshrrev_b32 v25, 6, v1 // (op 9909) 
s_mul_hi_u32 s55, s54, s12 // (op 9909) 
v_and_b32 v26, 1, v25 // (op 9909) 

s_sub_u32 s52, s54, s55 // (op 9909) 
v_lshl_add_u32 v25, v26, 4, v28 // (op 9909) 


s_lshr_b32 s53, s52, 1 // (op 9909) 

s_add_u32 s52, s53, s55 // (op 9909) 


s_lshr_b32 s53, s52, s10 // (op 9909) 

s_and_b32 s52, s10, 2147483648 // (op 9909) 
s_cmp_eq_i32 s52, 0 // (op 9909) 
s_mov_b32 s55, scc // (op 9909) 

s_cmp_eq_i32 s55, 0 // (op 9909) 
s_cselect_b32 s52, s54, s53 // (op 9909) 


s_mul_i32 s53, s52, s14 // (op 9909) 
v_lshl_add_u32 v26, s52, 7, v27 // (op 9909) 


s_sub_u32 s52, s54, s53 // (op 9909) 


v_lshl_add_u32 v27, s52, 6, v25 // (op 9909) 


v_mul_lo_u32 v25, v27, s24 // (op 9909) 

v_add_lshl_u32 v27, v26, v25, 1 // (op 9909) 


s_mul_i32 s53, s24, s23 // (op 9911) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 9911) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 9911) least significant
s_mul_hi_u32 s55, s24, s22 // (op 9911) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9911) most significant: sum
s_add_u32 s53, s53, s55 // (op 9911) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 9911) 

s_mov_b32 s60, s26 // (op 9911) reg expression
s_mov_b32 s61, s27 // (op 9911) reg expression
s_mov_b32 s62, s54 // (op 9911) reg expression
s_mov_b32 s63, 131072 // (op 9911) call()

v_lshrrev_b32 v25, 5, v1 // (op 10263) 
v_lshlrev_b32 v26, 2, v25 // (op 10263) 

v_lshl_add_u32 v25, s16, 7, v26 // (op 10263) 

v_and_b32 v26, -32, v1 // (op 10263) 
v_sub_u32 v28, v1, v26 // (op 10263) 

v_lshl_add_u32 v26, v25, 6, v28 // (op 10263) 


v_lshlrev_b32 v25, 2, v26 // (op 10263) 

s_lshl_b32 s52, s15, 15 // (op 10265) 
s_mov_b32 s64, s28 // (op 10265) reg expression
s_mov_b32 s65, s29 // (op 10265) reg expression
s_mov_b32 s66, s52 // (op 10265) reg expression
s_mov_b32 s67, 131072 // (op 10265) call()

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
v_accvgpr_write a4, 0.00000 // (op 2718) call()
v_accvgpr_write a5, 0.00000 // (op 2718) call()
v_accvgpr_write a6, 0.00000 // (op 2718) call()
v_accvgpr_write a7, 0.00000 // (op 2718) call()
v_accvgpr_write a8, 0.00000 // (op 2938) call()
v_accvgpr_write a9, 0.00000 // (op 2938) call()
v_accvgpr_write a10, 0.00000 // (op 2938) call()
v_accvgpr_write a11, 0.00000 // (op 2938) call()
v_accvgpr_write a12, 0.00000 // (op 3093) call()
v_accvgpr_write a13, 0.00000 // (op 3093) call()
v_accvgpr_write a14, 0.00000 // (op 3093) call()
v_accvgpr_write a15, 0.00000 // (op 3093) call()
v_accvgpr_write a16, 0.00000 // (op 3324) call()
v_accvgpr_write a17, 0.00000 // (op 3324) call()
v_accvgpr_write a18, 0.00000 // (op 3324) call()
v_accvgpr_write a19, 0.00000 // (op 3324) call()
v_accvgpr_write a20, 0.00000 // (op 3479) call()
v_accvgpr_write a21, 0.00000 // (op 3479) call()
v_accvgpr_write a22, 0.00000 // (op 3479) call()
v_accvgpr_write a23, 0.00000 // (op 3479) call()
v_accvgpr_write a24, 0.00000 // (op 3710) call()
v_accvgpr_write a25, 0.00000 // (op 3710) call()
v_accvgpr_write a26, 0.00000 // (op 3710) call()
v_accvgpr_write a27, 0.00000 // (op 3710) call()
v_accvgpr_write a28, 0.00000 // (op 3865) call()
v_accvgpr_write a29, 0.00000 // (op 3865) call()
v_accvgpr_write a30, 0.00000 // (op 3865) call()
v_accvgpr_write a31, 0.00000 // (op 3865) call()
v_lshrrev_b32 v26, 3, v1 // (op 9120) 
s_mul_i32 s52, s16, s8 // (op 9120) 
s_add_u32 s52, s52, s0 // (op 9120) 
v_and_b32 v28, -8, v1 // (op 9120) 
s_mul_hi_u32 s53, s52, s13 // (op 9120) 
v_sub_u32 v29, v1, v28 // (op 9120) 

s_sub_u32 s54, s52, s53 // (op 9120) 
v_lshlrev_b32 v28, 4, v29 // (op 9120) 

s_lshr_b32 s55, s54, 1 // (op 9120) 

s_add_u32 s54, s55, s53 // (op 9120) 


s_lshr_b32 s53, s54, s11 // (op 9120) 

s_and_b32 s54, s11, 2147483648 // (op 9120) 
s_cmp_eq_i32 s54, 0 // (op 9120) 
s_mov_b32 s55, scc // (op 9120) 

s_cmp_eq_i32 s55, 0 // (op 9120) 
s_cselect_b32 s54, s52, s53 // (op 9120) 


s_mul_hi_u32 s53, s54, s12 // (op 9120) 
s_sub_u32 s55, s54, s53 // (op 9120) 
s_lshr_b32 s68, s55, 1 // (op 9120) 

s_add_u32 s55, s68, s53 // (op 9120) 


s_lshr_b32 s53, s55, s10 // (op 9120) 

s_and_b32 s55, s10, 2147483648 // (op 9120) 
s_cmp_eq_i32 s55, 0 // (op 9120) 
s_mov_b32 s68, scc // (op 9120) 

s_cmp_eq_i32 s68, 0 // (op 9120) 
s_cselect_b32 s55, s54, s53 // (op 9120) 


v_lshl_add_u32 v29, s55, 7, v26 // (op 9120) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 9120) 
s_mul_i32 s53, s54, s68 // (op 9120) 


s_sub_u32 s54, s52, s53 // (op 9120) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 9120) 


v_mov_b32 v31, 0 // (op 9120) convert
v_mov_b32 v30, v26 // (op 9120) convert

v_mul_lo_u32 v33, v29, s31 // (op 9120) most significant: low of low * high
v_mul_hi_u32 v26, v29, s30 // (op 9120) most significant: high of low * low
v_mul_lo_u32 v32, v29, s30 // (op 9120) least significant: low of low * low
v_add_u32 v33, v33, v26 // (op 9120) most significant: sum

v_add_co_u32 v32, s[52:53], v32, v30 // (op 9120) least significant half
v_addc_co_u32 v33, s[52:53], v33, v31, s[52:53] // (op 9120) most significant half



s_mul_i32 s53, s30, s35 // (op 9122) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 9122) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 9122) least significant
s_mul_hi_u32 s55, s30, s34 // (op 9122) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9122) most significant: sum
s_add_u32 s53, s53, s55 // (op 9122) most significant: sum


s_mov_b32 s68, s32 // (op 9122) reg expression
s_mov_b32 s69, s33 // (op 9122) reg expression
s_mov_b32 s70, s52 // (op 9122) reg expression
s_mov_b32 s71, 131072 // (op 9122) call()

v_lshrrev_b32 v26, 3, v1 // (op 9139) 
s_mul_i32 s52, s16, s8 // (op 9139) 
s_add_u32 s52, s52, s0 // (op 9139) 
v_and_b32 v28, -8, v1 // (op 9139) 
s_mul_hi_u32 s53, s52, s13 // (op 9139) 
v_sub_u32 v29, v1, v28 // (op 9139) 

s_sub_u32 s54, s52, s53 // (op 9139) 
v_lshlrev_b32 v28, 4, v29 // (op 9139) 

s_lshr_b32 s55, s54, 1 // (op 9139) 

s_add_u32 s54, s55, s53 // (op 9139) 


s_lshr_b32 s53, s54, s11 // (op 9139) 

s_and_b32 s54, s11, 2147483648 // (op 9139) 
s_cmp_eq_i32 s54, 0 // (op 9139) 
s_mov_b32 s55, scc // (op 9139) 

s_cmp_eq_i32 s55, 0 // (op 9139) 
s_cselect_b32 s54, s52, s53 // (op 9139) 


s_mul_hi_u32 s53, s54, s12 // (op 9139) 
s_sub_u32 s55, s54, s53 // (op 9139) 
s_lshr_b32 s72, s55, 1 // (op 9139) 

s_add_u32 s55, s72, s53 // (op 9139) 


s_lshr_b32 s53, s55, s10 // (op 9139) 

s_and_b32 s55, s10, 2147483648 // (op 9139) 
s_cmp_eq_i32 s55, 0 // (op 9139) 
s_mov_b32 s72, scc // (op 9139) 

s_cmp_eq_i32 s72, 0 // (op 9139) 
s_cselect_b32 s55, s54, s53 // (op 9139) 


v_lshl_add_u32 v29, s55, 7, v26 // (op 9139) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 9139) 
s_mul_i32 s53, s54, s72 // (op 9139) 


s_sub_u32 s54, s52, s53 // (op 9139) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 9139) 


v_mov_b32 v31, 0 // (op 9139) convert
v_mov_b32 v30, v26 // (op 9139) convert

v_mul_lo_u32 v35, v29, s31 // (op 9139) most significant: low of low * high
v_mul_hi_u32 v26, v29, s30 // (op 9139) most significant: high of low * low
v_mul_lo_u32 v34, v29, s30 // (op 9139) least significant: low of low * low
v_add_u32 v35, v35, v26 // (op 9139) most significant: sum

v_add_co_u32 v34, s[52:53], v34, v30 // (op 9139) least significant half
v_addc_co_u32 v35, s[52:53], v35, v31, s[52:53] // (op 9139) most significant half



s_mul_i32 s53, s30, s35 // (op 9141) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 9141) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 9141) least significant
s_mul_hi_u32 s55, s30, s34 // (op 9141) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9141) most significant: sum
s_add_u32 s53, s53, s55 // (op 9141) most significant: sum


s_mov_b32 s68, s32 // (op 9141) reg expression
s_mov_b32 s69, s33 // (op 9141) reg expression
s_mov_b32 s70, s52 // (op 9141) reg expression
s_mov_b32 s71, 131072 // (op 9141) call()

v_lshrrev_b32 v26, 3, v1 // (op 9248) 
s_mul_i32 s52, s16, s8 // (op 9248) 
s_add_u32 s52, s52, s0 // (op 9248) 
v_and_b32 v28, -8, v1 // (op 9248) 
s_mul_hi_u32 s53, s52, s13 // (op 9248) 
v_sub_u32 v29, v1, v28 // (op 9248) 

s_sub_u32 s54, s52, s53 // (op 9248) 
v_lshlrev_b32 v28, 4, v29 // (op 9248) 

s_lshr_b32 s55, s54, 1 // (op 9248) 

s_add_u32 s54, s55, s53 // (op 9248) 


s_lshr_b32 s53, s54, s11 // (op 9248) 

s_and_b32 s54, s11, 2147483648 // (op 9248) 
s_cmp_eq_i32 s54, 0 // (op 9248) 
s_mov_b32 s55, scc // (op 9248) 

s_cmp_eq_i32 s55, 0 // (op 9248) 
s_cselect_b32 s54, s52, s53 // (op 9248) 


s_mul_hi_u32 s53, s54, s12 // (op 9248) 
s_sub_u32 s55, s54, s53 // (op 9248) 
s_lshr_b32 s72, s55, 1 // (op 9248) 

s_add_u32 s55, s72, s53 // (op 9248) 


s_lshr_b32 s53, s55, s10 // (op 9248) 

s_and_b32 s55, s10, 2147483648 // (op 9248) 
s_cmp_eq_i32 s55, 0 // (op 9248) 
s_mov_b32 s72, scc // (op 9248) 

s_cmp_eq_i32 s72, 0 // (op 9248) 
s_cselect_b32 s55, s54, s53 // (op 9248) 


s_mul_i32 s53, s55, s14 // (op 9248) 

s_sub_u32 s55, s54, s53 // (op 9248) 

v_lshl_add_u32 v29, s55, 6, v26 // (op 9248) 


v_mul_lo_u32 v31, v29, s37 // (op 9248) most significant: low of low * high
v_mul_hi_u32 v26, v29, s36 // (op 9248) most significant: high of low * low
v_mul_lo_u32 v30, v29, s36 // (op 9248) least significant: low of low * low
v_add_u32 v31, v31, v26 // (op 9248) most significant: sum


s_ashr_i64 s[72:73], s[2:3], 7 // (op 9248) 
s_mul_i32 s53, s54, s72 // (op 9248) 


s_sub_u32 s54, s52, s53 // (op 9248) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 9248) 


v_mov_b32 v29, 0 // (op 9248) convert
v_mov_b32 v28, v26 // (op 9248) convert

v_add_co_u32 v36, s[52:53], v28, v30 // (op 9248) least significant half
v_addc_co_u32 v37, s[52:53], v29, v31, s[52:53] // (op 9248) most significant half



s_mul_i32 s53, s36, s23 // (op 9250) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 9250) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 9250) least significant
s_mul_hi_u32 s55, s36, s22 // (op 9250) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9250) most significant: sum
s_add_u32 s53, s53, s55 // (op 9250) most significant: sum


s_mov_b32 s72, s38 // (op 9250) reg expression
s_mov_b32 s73, s39 // (op 9250) reg expression
s_mov_b32 s74, s52 // (op 9250) reg expression
s_mov_b32 s75, 131072 // (op 9250) call()

v_lshrrev_b32 v26, 3, v1 // (op 9267) 
s_mul_i32 s52, s16, s8 // (op 9267) 
s_add_u32 s52, s52, s0 // (op 9267) 
v_and_b32 v28, -8, v1 // (op 9267) 
s_mul_hi_u32 s53, s52, s13 // (op 9267) 
v_sub_u32 v29, v1, v28 // (op 9267) 

s_sub_u32 s54, s52, s53 // (op 9267) 
v_lshlrev_b32 v28, 4, v29 // (op 9267) 

s_lshr_b32 s55, s54, 1 // (op 9267) 

s_add_u32 s54, s55, s53 // (op 9267) 


s_lshr_b32 s53, s54, s11 // (op 9267) 

s_and_b32 s54, s11, 2147483648 // (op 9267) 
s_cmp_eq_i32 s54, 0 // (op 9267) 
s_mov_b32 s55, scc // (op 9267) 

s_cmp_eq_i32 s55, 0 // (op 9267) 
s_cselect_b32 s54, s52, s53 // (op 9267) 


s_mul_hi_u32 s53, s54, s12 // (op 9267) 
s_sub_u32 s55, s54, s53 // (op 9267) 
s_lshr_b32 s76, s55, 1 // (op 9267) 

s_add_u32 s55, s76, s53 // (op 9267) 


s_lshr_b32 s53, s55, s10 // (op 9267) 

s_and_b32 s55, s10, 2147483648 // (op 9267) 
s_cmp_eq_i32 s55, 0 // (op 9267) 
s_mov_b32 s76, scc // (op 9267) 

s_cmp_eq_i32 s76, 0 // (op 9267) 
s_cselect_b32 s55, s54, s53 // (op 9267) 


s_mul_i32 s53, s55, s14 // (op 9267) 

s_sub_u32 s55, s54, s53 // (op 9267) 

v_lshl_add_u32 v29, s55, 6, v26 // (op 9267) 


v_mul_lo_u32 v31, v29, s37 // (op 9267) most significant: low of low * high
v_mul_hi_u32 v26, v29, s36 // (op 9267) most significant: high of low * low
v_mul_lo_u32 v30, v29, s36 // (op 9267) least significant: low of low * low
v_add_u32 v31, v31, v26 // (op 9267) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 9267) 
s_mul_i32 s53, s54, s76 // (op 9267) 


s_sub_u32 s54, s52, s53 // (op 9267) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 9267) 


v_mov_b32 v29, 0 // (op 9267) convert
v_mov_b32 v28, v26 // (op 9267) convert

v_add_co_u32 v38, s[52:53], v28, v30 // (op 9267) least significant half
v_addc_co_u32 v39, s[52:53], v29, v31, s[52:53] // (op 9267) most significant half



s_mul_i32 s53, s36, s23 // (op 9269) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 9269) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 9269) least significant
s_mul_hi_u32 s55, s36, s22 // (op 9269) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9269) most significant: sum
s_add_u32 s53, s53, s55 // (op 9269) most significant: sum


s_mov_b32 s72, s38 // (op 9269) reg expression
s_mov_b32 s73, s39 // (op 9269) reg expression
s_mov_b32 s74, s52 // (op 9269) reg expression
s_mov_b32 s75, 131072 // (op 9269) call()

v_lshrrev_b32 v26, 2, v1 // (op 9376) 
s_mul_i32 s52, s16, s8 // (op 9376) 
s_add_u32 s52, s52, s0 // (op 9376) 
v_lshlrev_b32 v28, 1, v26 // (op 9376) 

s_mul_hi_u32 s53, s52, s13 // (op 9376) 
v_and_b32 v26, -4, v1 // (op 9376) 
s_sub_u32 s54, s52, s53 // (op 9376) 
v_sub_u32 v29, v1, v26 // (op 9376) 

s_lshr_b32 s55, s54, 1 // (op 9376) 

s_add_u32 s54, s55, s53 // (op 9376) 


s_lshr_b32 s53, s54, s11 // (op 9376) 

s_and_b32 s54, s11, 2147483648 // (op 9376) 
s_cmp_eq_i32 s54, 0 // (op 9376) 
s_mov_b32 s55, scc // (op 9376) 

s_cmp_eq_i32 s55, 0 // (op 9376) 
s_cselect_b32 s54, s52, s53 // (op 9376) 


s_mul_hi_u32 s53, s54, s12 // (op 9376) 
s_sub_u32 s55, s54, s53 // (op 9376) 
s_lshr_b32 s76, s55, 1 // (op 9376) 

s_add_u32 s55, s76, s53 // (op 9376) 


s_lshr_b32 s53, s55, s10 // (op 9376) 

s_and_b32 s55, s10, 2147483648 // (op 9376) 
s_cmp_eq_i32 s55, 0 // (op 9376) 
s_mov_b32 s76, scc // (op 9376) 

s_cmp_eq_i32 s76, 0 // (op 9376) 
s_cselect_b32 s55, s54, s53 // (op 9376) 


v_lshl_add_u32 v26, s55, 7, v28 // (op 9376) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 9376) 
s_mul_i32 s53, s54, s76 // (op 9376) 


s_sub_u32 s54, s52, s53 // (op 9376) 


v_lshl_add_u32 v28, s54, 2, v29 // (op 9376) 


v_mul_lo_u32 v29, v26, s40 // (op 9376) 
v_add_u32 v29, v29, v28 // (op 9376) 


s_mul_i32 s53, s40, s35 // (op 9378) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 9378) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 9378) least significant
s_mul_hi_u32 s55, s40, s34 // (op 9378) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9378) most significant: sum
s_add_u32 s53, s53, s55 // (op 9378) most significant: sum


s_mov_b32 s76, s42 // (op 9378) reg expression
s_mov_b32 s77, s43 // (op 9378) reg expression
s_mov_b32 s78, s52 // (op 9378) reg expression
s_mov_b32 s79, 131072 // (op 9378) call()

v_lshrrev_b32 v26, 2, v1 // (op 9395) 
s_mul_i32 s52, s16, s8 // (op 9395) 
s_add_u32 s52, s52, s0 // (op 9395) 
v_lshlrev_b32 v28, 1, v26 // (op 9395) 

s_mul_hi_u32 s53, s52, s13 // (op 9395) 
v_and_b32 v26, -4, v1 // (op 9395) 
s_sub_u32 s54, s52, s53 // (op 9395) 
v_sub_u32 v30, v1, v26 // (op 9395) 

s_lshr_b32 s55, s54, 1 // (op 9395) 

s_add_u32 s54, s55, s53 // (op 9395) 


s_lshr_b32 s53, s54, s11 // (op 9395) 

s_and_b32 s54, s11, 2147483648 // (op 9395) 
s_cmp_eq_i32 s54, 0 // (op 9395) 
s_mov_b32 s55, scc // (op 9395) 

s_cmp_eq_i32 s55, 0 // (op 9395) 
s_cselect_b32 s54, s52, s53 // (op 9395) 


s_mul_hi_u32 s53, s54, s12 // (op 9395) 
s_sub_u32 s55, s54, s53 // (op 9395) 
s_lshr_b32 s80, s55, 1 // (op 9395) 

s_add_u32 s55, s80, s53 // (op 9395) 


s_lshr_b32 s53, s55, s10 // (op 9395) 

s_and_b32 s55, s10, 2147483648 // (op 9395) 
s_cmp_eq_i32 s55, 0 // (op 9395) 
s_mov_b32 s80, scc // (op 9395) 

s_cmp_eq_i32 s80, 0 // (op 9395) 
s_cselect_b32 s55, s54, s53 // (op 9395) 


v_lshl_add_u32 v26, s55, 7, v28 // (op 9395) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 9395) 
s_mul_i32 s53, s54, s80 // (op 9395) 


s_sub_u32 s54, s52, s53 // (op 9395) 


v_lshl_add_u32 v28, s54, 2, v30 // (op 9395) 


v_mul_lo_u32 v30, v26, s40 // (op 9395) 
v_add_u32 v30, v30, v28 // (op 9395) 


s_mul_i32 s53, s40, s35 // (op 9397) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 9397) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 9397) least significant
s_mul_hi_u32 s55, s40, s34 // (op 9397) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9397) most significant: sum
s_add_u32 s53, s53, s55 // (op 9397) most significant: sum


s_mov_b32 s76, s42 // (op 9397) reg expression
s_mov_b32 s77, s43 // (op 9397) reg expression
s_mov_b32 s78, s52 // (op 9397) reg expression
s_mov_b32 s79, 131072 // (op 9397) call()

v_lshrrev_b32 v26, 2, v1 // (op 9504) 
s_mul_i32 s52, s16, s8 // (op 9504) 
s_add_u32 s52, s52, s0 // (op 9504) 
v_and_b32 v28, -4, v1 // (op 9504) 
s_mul_hi_u32 s53, s52, s13 // (op 9504) 
v_sub_u32 v31, v1, v28 // (op 9504) 

s_sub_u32 s54, s52, s53 // (op 9504) 
s_lshr_b32 s55, s54, 1 // (op 9504) 

s_add_u32 s54, s55, s53 // (op 9504) 


s_lshr_b32 s53, s54, s11 // (op 9504) 

s_and_b32 s54, s11, 2147483648 // (op 9504) 
s_cmp_eq_i32 s54, 0 // (op 9504) 
s_mov_b32 s55, scc // (op 9504) 

s_cmp_eq_i32 s55, 0 // (op 9504) 
s_cselect_b32 s54, s52, s53 // (op 9504) 


s_mul_hi_u32 s53, s54, s12 // (op 9504) 
s_sub_u32 s55, s54, s53 // (op 9504) 
s_lshr_b32 s80, s55, 1 // (op 9504) 

s_add_u32 s55, s80, s53 // (op 9504) 


s_lshr_b32 s53, s55, s10 // (op 9504) 

s_and_b32 s55, s10, 2147483648 // (op 9504) 
s_cmp_eq_i32 s55, 0 // (op 9504) 
s_mov_b32 s80, scc // (op 9504) 

s_cmp_eq_i32 s80, 0 // (op 9504) 
s_cselect_b32 s55, s54, s53 // (op 9504) 


s_mul_i32 s53, s55, s14 // (op 9504) 

s_sub_u32 s55, s54, s53 // (op 9504) 

v_lshl_add_u32 v28, s55, 6, v26 // (op 9504) 


v_mul_lo_u32 v26, v28, s44 // (op 9504) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 9504) 
s_mul_i32 s53, s54, s80 // (op 9504) 


s_sub_u32 s54, s52, s53 // (op 9504) 


v_lshl_add_u32 v28, s54, 2, v31 // (op 9504) 


v_add_u32 v31, v28, v26 // (op 9504) 


s_mul_i32 s53, s44, s23 // (op 9506) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 9506) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 9506) least significant
s_mul_hi_u32 s55, s44, s22 // (op 9506) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9506) most significant: sum
s_add_u32 s53, s53, s55 // (op 9506) most significant: sum


s_mov_b32 s80, s46 // (op 9506) reg expression
s_mov_b32 s81, s47 // (op 9506) reg expression
s_mov_b32 s82, s52 // (op 9506) reg expression
s_mov_b32 s83, 131072 // (op 9506) call()

v_lshrrev_b32 v26, 2, v1 // (op 9523) 
s_mul_i32 s52, s16, s8 // (op 9523) 
s_add_u32 s52, s52, s0 // (op 9523) 
v_and_b32 v28, -4, v1 // (op 9523) 
s_mul_hi_u32 s53, s52, s13 // (op 9523) 
v_sub_u32 v40, v1, v28 // (op 9523) 

s_sub_u32 s54, s52, s53 // (op 9523) 
s_lshr_b32 s55, s54, 1 // (op 9523) 

s_add_u32 s54, s55, s53 // (op 9523) 


s_lshr_b32 s53, s54, s11 // (op 9523) 

s_and_b32 s54, s11, 2147483648 // (op 9523) 
s_cmp_eq_i32 s54, 0 // (op 9523) 
s_mov_b32 s55, scc // (op 9523) 

s_cmp_eq_i32 s55, 0 // (op 9523) 
s_cselect_b32 s54, s52, s53 // (op 9523) 


s_mul_hi_u32 s53, s54, s12 // (op 9523) 
s_sub_u32 s55, s54, s53 // (op 9523) 
s_lshr_b32 s84, s55, 1 // (op 9523) 

s_add_u32 s55, s84, s53 // (op 9523) 


s_lshr_b32 s53, s55, s10 // (op 9523) 

s_and_b32 s55, s10, 2147483648 // (op 9523) 
s_cmp_eq_i32 s55, 0 // (op 9523) 
s_mov_b32 s84, scc // (op 9523) 

s_cmp_eq_i32 s84, 0 // (op 9523) 
s_cselect_b32 s55, s54, s53 // (op 9523) 


s_mul_i32 s53, s55, s14 // (op 9523) 

s_sub_u32 s55, s54, s53 // (op 9523) 

v_lshl_add_u32 v28, s55, 6, v26 // (op 9523) 


v_mul_lo_u32 v26, v28, s44 // (op 9523) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 9523) 
s_mul_i32 s53, s54, s84 // (op 9523) 


s_sub_u32 s54, s52, s53 // (op 9523) 


v_lshl_add_u32 v28, s54, 2, v40 // (op 9523) 


v_add_u32 v40, v28, v26 // (op 9523) 


s_mul_i32 s53, s44, s23 // (op 9525) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 9525) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 9525) least significant
s_mul_hi_u32 s55, s44, s22 // (op 9525) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9525) most significant: sum
s_add_u32 s53, s53, s55 // (op 9525) most significant: sum


s_mov_b32 s80, s46 // (op 9525) reg expression
s_mov_b32 s81, s47 // (op 9525) reg expression
s_mov_b32 s82, s52 // (op 9525) reg expression
s_mov_b32 s83, 131072 // (op 9525) call()

v_and_b32 v26, -8, v1 // (op 9700) 
v_sub_u32 v28, v1, v26 // (op 9700) 

v_lshlrev_b32 v26, 4, v28 // (op 9700) 

v_lshrrev_b32 v28, 3, v1 // (op 9700) 
v_lshl_add_u32 v41, v28, 7, v26 // (op 9700) 


s_nop 0
v_readfirstlane_b32 s52, v41 // (op 9700) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10775) 

s_lshl_b64 s[54:55], s[36:37], 5 // (op 8986) 
v_mov_b64 v[42:43], s[54:55] // (op 8986) convert


v_mov_b64 v[44:45], v[36:37] // (op 8986) 
s_mov_b32 m0, s52 // (op 8986) reg expression
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 8986) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 8986) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 8986) most significant half

s_add_u32 m0, m0, 4096 // (op 8986) 
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 8986) Load value direct to lds






buffer_load_ubyte v26, v31, s[80:83], 0 offen offset: 0 // (op 8721) Load value
v_and_b32 v28, -8, v1 // (op 9625) 
v_sub_u32 v41, v1, v28 // (op 9625) 

v_lshlrev_b32 v28, 4, v41 // (op 9625) 

v_lshrrev_b32 v41, 3, v1 // (op 9625) 
v_lshl_add_u32 v42, v41, 7, v28 // (op 9625) 


s_nop 0
v_readfirstlane_b32 s52, v42 // (op 9625) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10756) 

s_lshl_b64 s[54:55], s[30:31], 5 // (op 8991) 
v_mov_b64 v[42:43], s[54:55] // (op 8991) convert


v_mov_b64 v[44:45], v[32:33] // (op 8991) 
s_add_u32 s53, 8192, s52 // (op 8991) 
s_mov_b32 m0, s53 // (op 8991) reg expression
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 8991) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 8991) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 8991) most significant half

s_add_u32 m0, m0, 4096 // (op 8991) 
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 8991) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 8991) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 8991) most significant half

s_add_u32 m0, m0, 4096 // (op 8991) 
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 8991) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 8991) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 8991) most significant half

s_add_u32 m0, m0, 4096 // (op 8991) 
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 8991) Load value direct to lds





v_mov_b64 v[42:43], s[40:41] // (op 8727) convert

v_mov_b32 v28, v29 // (op 8727) 
buffer_load_ubyte v44, v28, s[76:79], 0 offen offset: 0 // (op 8727) Load value
v_add_u32 v28, v28, v42 // (op 8727) 
buffer_load_ubyte v45, v28, s[76:79], 0 offen offset: 0 // (op 8727) Load value


v_and_b32 v28, -8, v1 // (op 10510) 
v_sub_u32 v41, v1, v28 // (op 10510) 

v_lshlrev_b32 v28, 4, v41 // (op 10510) 

v_lshrrev_b32 v41, 3, v1 // (op 10510) 
v_lshl_add_u32 v42, v41, 7, v28 // (op 10510) 


s_nop 0
v_readfirstlane_b32 s52, v42 // (op 10510) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10958) 

s_lshl_b64 s[54:55], s[36:37], 5 // (op 8996) 
v_mov_b64 v[42:43], s[54:55] // (op 8996) convert


v_mov_b64 v[46:47], v[36:37] // (op 8996) 
s_add_u32 s53, 24576, s52 // (op 8996) 
s_mov_b32 m0, s53 // (op 8996) reg expression
s_mov_b32 s54, 128 // (op 8996) call()
buffer_load_dwordx4 v46, s[72:75], s54 offen offset: 0 lds // (op 8996) Load value direct to lds
v_add_co_u32 v46, s[84:85], v46, v42 // (op 8996) least significant half
v_addc_co_u32 v47, s[84:85], 0, v47, s[84:85] // (op 8996) most significant half

s_add_u32 m0, m0, 4096 // (op 8996) 
buffer_load_dwordx4 v46, s[72:75], s54 offen offset: 0 lds // (op 8996) Load value direct to lds









buffer_load_ubyte v28, v31, s[80:83], 0 offen offset:4 // (op 8733) Load value

v_and_b32 v31, -8, v1 // (op 10466) 
v_sub_u32 v36, v1, v31 // (op 10466) 

v_lshlrev_b32 v31, 4, v36 // (op 10466) 

v_lshrrev_b32 v36, 3, v1 // (op 10466) 
v_lshl_add_u32 v37, v36, 7, v31 // (op 10466) 


s_nop 0
v_readfirstlane_b32 s52, v37 // (op 10466) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10948) 

s_lshl_b64 s[54:55], s[30:31], 5 // (op 8998) 
v_mov_b64 v[36:37], s[54:55] // (op 8998) convert


v_mov_b64 v[42:43], v[32:33] // (op 8998) 
s_add_u32 s53, 32768, s52 // (op 8998) 
s_mov_b32 m0, s53 // (op 8998) reg expression
s_mov_b32 s54, 128 // (op 8998) call()
buffer_load_dwordx4 v42, s[68:71], s54 offen offset: 0 lds // (op 8998) Load value direct to lds
v_add_co_u32 v42, s[84:85], v42, v36 // (op 8998) least significant half
v_addc_co_u32 v43, s[84:85], 0, v43, s[84:85] // (op 8998) most significant half

s_add_u32 m0, m0, 4096 // (op 8998) 
buffer_load_dwordx4 v42, s[68:71], s54 offen offset: 0 lds // (op 8998) Load value direct to lds
v_add_co_u32 v42, s[84:85], v42, v36 // (op 8998) least significant half
v_addc_co_u32 v43, s[84:85], 0, v43, s[84:85] // (op 8998) most significant half

s_add_u32 m0, m0, 4096 // (op 8998) 
buffer_load_dwordx4 v42, s[68:71], s54 offen offset: 0 lds // (op 8998) Load value direct to lds
v_add_co_u32 v42, s[84:85], v42, v36 // (op 8998) least significant half
v_addc_co_u32 v43, s[84:85], 0, v43, s[84:85] // (op 8998) most significant half

s_add_u32 m0, m0, 4096 // (op 8998) 
buffer_load_dwordx4 v42, s[68:71], s54 offen offset: 0 lds // (op 8998) Load value direct to lds







v_mov_b64 v[32:33], s[40:41] // (op 8739) convert

v_mov_b32 v31, v29 // (op 8739) 
buffer_load_ubyte v36, v31, s[76:79], 0 offen offset:4 // (op 8739) Load value
v_add_u32 v31, v31, v32 // (op 8739) 
buffer_load_ubyte v37, v31, s[76:79], 0 offen offset:4 // (op 8739) Load value



v_and_b32 v29, -4, v1 // (op 9852) 
v_sub_u32 v31, v1, v29 // (op 9852) 

v_and_b32 v29, 4294967292, v1 // (op 9852) 
v_add_u32 v32, v29, v31 // (op 9852) 


s_barrier  // (op 10816) 



s_waitcnt vmcnt(15)
ds_write_b8 v32, v26 offset:49152 // (op 8742) Store local data 

v_and_b32 v29, -4, v1 // (op 9776) 
v_sub_u32 v31, v1, v29 // (op 9776) 

v_lshrrev_b32 v29, 2, v1 // (op 9776) 
v_lshl_add_u32 v32, v29, 3, v31 // (op 9776) 


s_barrier  // (op 10795) 



s_waitcnt vmcnt(10)
ds_write_b8 v32, v44 offset:49408 // (op 8744) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v32, v45 offset:49412 // (op 8744) Store local data 

s_barrier  // (op 10772) 



ds_read_b128 v[46:49], v23 // (op 8779) Load local data 
ds_read_b128 v[50:53], v23 offset:64 // (op 8779) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 10813) 



ds_read_u8 v29, v24 offset:49152 // (op 8809) Load local data 



ds_read_b128 v[54:57], v23 offset:4096 // (op 8784) Load local data 
ds_read_b128 v[58:61], v23 offset:4160 // (op 8784) Load local data 



ds_read_u8 v31, v24 offset:49280 // (op 8814) Load local data 
s_barrier  // (op 10753) 



ds_read_b128 v[62:65], v21 offset:8192 // (op 8759) Load local data 
ds_read_b128 v[66:69], v21 offset:8256 // (op 8759) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 10792) 



ds_read_u8 v32, v20 offset:49408 // (op 8789) Load local data 



ds_read_b128 v[70:73], v21 offset:12288 // (op 8764) Load local data 
ds_read_b128 v[74:77], v21 offset:12352 // (op 8764) Load local data 



ds_read_u8 v33, v20 offset:49536 // (op 8794) Load local data 



ds_read_b128 v[78:81], v21 offset:16384 // (op 8769) Load local data 
ds_read_b128 v[82:85], v21 offset:16448 // (op 8769) Load local data 



ds_read_u8 v41, v20 offset:49664 // (op 8799) Load local data 



ds_read_b128 v[86:89], v21 offset:20480 // (op 8774) Load local data 
ds_read_b128 v[90:93], v21 offset:20544 // (op 8774) Load local data 



ds_read_u8 v42, v20 offset:49792 // (op 8804) Load local data 
v_and_b32 v43, -8, v1 // (op 9597) 
v_sub_u32 v94, v1, v43 // (op 9597) 

v_lshlrev_b32 v43, 4, v94 // (op 9597) 

v_lshrrev_b32 v94, 3, v1 // (op 9597) 
v_lshl_add_u32 v95, v94, 7, v43 // (op 9597) 


s_nop 0
v_readfirstlane_b32 s52, v95 // (op 9597) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v43, -8, v1 // (op 9672) 
v_sub_u32 v94, v1, v43 // (op 9672) 

v_lshlrev_b32 v43, 4, v94 // (op 9672) 

v_lshrrev_b32 v94, 3, v1 // (op 9672) 
v_lshl_add_u32 v95, v94, 7, v43 // (op 9672) 


s_nop 0
v_readfirstlane_b32 s53, v95 // (op 9672) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v43, -4, v1 // (op 9747) 
v_sub_u32 v94, v1, v43 // (op 9747) 

v_lshrrev_b32 v43, 2, v1 // (op 9747) 
v_lshl_add_u32 v95, v43, 3, v94 // (op 9747) 


v_and_b32 v43, -4, v1 // (op 9823) 
v_sub_u32 v94, v1, v43 // (op 9823) 

v_and_b32 v43, 4294967292, v1 // (op 9823) 
v_add_u32 v96, v43, v94 // (op 9823) 


v_and_b32 v43, -8, v1 // (op 10451) 
v_sub_u32 v94, v1, v43 // (op 10451) 

v_lshlrev_b32 v43, 4, v94 // (op 10451) 

v_lshrrev_b32 v94, 3, v1 // (op 10451) 
v_lshl_add_u32 v97, v94, 7, v43 // (op 10451) 


s_nop 0
v_readfirstlane_b32 s54, v97 // (op 10451) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v43, -8, v1 // (op 10495) 
v_sub_u32 v94, v1, v43 // (op 10495) 

v_lshlrev_b32 v43, 4, v94 // (op 10495) 

v_lshrrev_b32 v94, 3, v1 // (op 10495) 
v_lshl_add_u32 v97, v94, 7, v43 // (op 10495) 


s_nop 0
v_readfirstlane_b32 s55, v97 // (op 10495) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v43, -4, v1 // (op 10539) 
v_sub_u32 v94, v1, v43 // (op 10539) 

v_lshrrev_b32 v43, 2, v1 // (op 10539) 
v_lshl_add_u32 v97, v43, 3, v94 // (op 10539) 


v_and_b32 v43, -4, v1 // (op 10568) 
v_sub_u32 v94, v1, v43 // (op 10568) 

v_and_b32 v43, 4294967292, v1 // (op 10568) 
v_add_u32 v98, v43, v94 // (op 10568) 


s_mov_b32 s1, 0 // (op 3743) call()
s_and_b32 s84, s7, 4294967294 // (op 3742) 
s_cmp_lt_u32 s1, s84 // (op 3742) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoop_3742 // (op 3742) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoop_3742 if false)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoop_3742:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[46:53], a[0:3], v32, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 90) 
s_barrier  // (op 10778) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:69], v[54:61], a[4:7], v32, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2758) 
s_add_u32 s1, s1, 2 // (op 3744) 
s_lshl_b64 s[84:85], s[36:37], 5 // (op 8911) 
v_mov_b64 v[100:101], s[84:85] // (op 8911) convert


v_mov_b64 v[102:103], v[38:39] // (op 8911) 
v_mov_b64 v[104:105], v[102:103] // (op 8911) 
s_mov_b32 m0, s53 // (op 8911) reg expression
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[70:77], v[46:53], a[8:11], v33, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3055) 
s_mov_b32 s84, 256 // (op 8911) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:77], v[54:61], a[12:15], v33, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3009) 
buffer_load_dwordx4 v104, s[72:75], s84 offen offset: 0 lds // (op 8911) Load value direct to lds
v_add_co_u32 v104, s[86:87], v104, v100 // (op 8911) least significant half
v_addc_co_u32 v105, s[86:87], 0, v105, s[86:87] // (op 8911) most significant half

s_add_u32 m0, m0, 4096 // (op 8911) 
buffer_load_dwordx4 v104, s[72:75], s84 offen offset: 0 lds // (op 8911) Load value direct to lds



s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[78:85], v[46:53], a[16:19], v41, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3441) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:85], v[54:61], a[20:23], v41, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3395) 
v_mov_b32 v43, v40 // (op 25) 

buffer_load_ubyte v26, v43, s[80:83], 0 offen offset:8 // (op 25) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[86:93], v[46:53], a[24:27], v42, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3827) 
s_barrier  // (op 10759) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[86:93], v[54:61], a[28:31], v42, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3781) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 8906) 
v_mov_b64 v[100:101], s[84:85] // (op 8906) convert


v_mov_b64 v[104:105], v[34:35] // (op 8906) 
v_mov_b64 v[106:107], v[104:105] // (op 8906) 
s_add_u32 s84, 8192, s52 // (op 8906) 
s_mov_b32 m0, s84 // (op 8906) reg expression
s_mov_b32 s85, 256 // (op 8906) call()
buffer_load_dwordx4 v106, s[68:71], s85 offen offset: 0 lds // (op 8906) Load value direct to lds
v_add_co_u32 v106, s[86:87], v106, v100 // (op 8906) least significant half
v_addc_co_u32 v107, s[86:87], 0, v107, s[86:87] // (op 8906) most significant half

s_add_u32 m0, m0, 4096 // (op 8906) 
buffer_load_dwordx4 v106, s[68:71], s85 offen offset: 0 lds // (op 8906) Load value direct to lds
v_add_co_u32 v106, s[86:87], v106, v100 // (op 8906) least significant half
v_addc_co_u32 v107, s[86:87], 0, v107, s[86:87] // (op 8906) most significant half

s_add_u32 m0, m0, 4096 // (op 8906) 
buffer_load_dwordx4 v106, s[68:71], s85 offen offset: 0 lds // (op 8906) Load value direct to lds
v_add_co_u32 v106, s[86:87], v106, v100 // (op 8906) least significant half
v_addc_co_u32 v107, s[86:87], 0, v107, s[86:87] // (op 8906) most significant half

s_add_u32 m0, m0, 4096 // (op 8906) 
buffer_load_dwordx4 v106, s[68:71], s85 offen offset: 0 lds // (op 8906) Load value direct to lds





v_mov_b64 v[100:101], s[40:41] // (op 17) convert

v_mov_b32 v94, v30 // (op 17) 
v_mov_b32 v99, v94 // (op 17) 
buffer_load_ubyte v44, v99, s[76:79], 0 offen offset:8 // (op 17) Load value
v_add_u32 v99, v99, v100 // (op 17) 
buffer_load_ubyte v45, v99, s[76:79], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 10974) 



s_waitcnt vmcnt(15)
ds_write_b8 v98, v28 offset:49920 // (op 2637) Store local data 
s_barrier  // (op 10968) 



s_waitcnt vmcnt(10)
ds_write_b8 v97, v36 offset:50176 // (op 2636) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v97, v37 offset:50180 // (op 2636) Store local data 
s_barrier  // (op 10954) 



ds_read_b128 v[106:109], v16 offset:24576 // (op 2629) Load local data 
ds_read_b128 v[110:113], v16 offset:24640 // (op 2629) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 10971) 



ds_read_u8 v99, v17 offset:49920 // (op 2633) Load local data 



ds_read_b128 v[114:117], v16 offset:28672 // (op 2763) Load local data 
ds_read_b128 v[118:121], v16 offset:28736 // (op 2763) Load local data 



ds_read_u8 v100, v17 offset:50048 // (op 2767) Load local data 
s_barrier  // (op 10945) 
s_barrier  // (op 11213) 



ds_read_b128 v[62:65], v15 offset:32768 // (op 2627) Load local data 
ds_read_b128 v[66:69], v15 offset:32832 // (op 2627) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 10964) 



ds_read_u8 v32, v14 offset:50176 // (op 2631) Load local data 



ds_read_b128 v[70:73], v15 offset:36864 // (op 3012) Load local data 
ds_read_b128 v[74:77], v15 offset:36928 // (op 3012) Load local data 



ds_read_u8 v33, v14 offset:50304 // (op 3015) Load local data 



ds_read_b128 v[78:81], v15 offset:40960 // (op 3398) Load local data 
ds_read_b128 v[82:85], v15 offset:41024 // (op 3398) Load local data 



ds_read_u8 v41, v14 offset:50432 // (op 3401) Load local data 



ds_read_b128 v[86:89], v15 offset:45056 // (op 3784) Load local data 
ds_read_b128 v[90:93], v15 offset:45120 // (op 3784) Load local data 



ds_read_u8 v42, v14 offset:50560 // (op 3787) Load local data 
s_barrier  // (op 11243) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[106:113], a[0:3], v32, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2634) 
s_barrier  // (op 10961) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:69], v[114:121], a[4:7], v32, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2759) 
s_lshl_b64 s[84:85], s[36:37], 5 // (op 8967) 
v_mov_b64 v[122:123], s[84:85] // (op 8967) convert


v_mov_b64 v[124:125], v[102:103] // (op 8967) 
s_add_u32 s84, 24576, s55 // (op 8967) 
s_mov_b32 m0, s84 // (op 8967) reg expression
s_mov_b32 s85, 384 // (op 8967) call()
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[70:77], v[106:113], a[8:11], v33, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3056) 
buffer_load_dwordx4 v124, s[72:75], s85 offen offset: 0 lds // (op 8967) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:77], v[114:121], a[12:15], v33, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3010) 
v_add_co_u32 v124, s[86:87], v124, v122 // (op 8967) least significant half
v_addc_co_u32 v125, s[86:87], 0, v125, s[86:87] // (op 8967) most significant half

s_add_u32 m0, m0, 4096 // (op 8967) 
buffer_load_dwordx4 v124, s[72:75], s85 offen offset: 0 lds // (op 8967) Load value direct to lds




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[78:85], v[106:113], a[16:19], v41, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3442) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:85], v[114:121], a[20:23], v41, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3396) 
buffer_load_ubyte v28, v43, s[80:83], 0 offen offset:12 // (op 2640) Load value

v_add_co_u32 v38, vcc, 256, v38 // (op 9259) least significant half
v_addc_co_u32 v39, vcc, 0, v39, vcc // (op 9259) most significant half
v_add_u32 v40, 8, v40 // (op 9515) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[86:93], v[106:113], a[24:27], v42, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3828) 
s_barrier  // (op 10951) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[86:93], v[114:121], a[28:31], v42, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3782) 
s_lshl_b64 s[84:85], s[30:31], 5 // (op 8962) 


v_mov_b64 v[102:103], s[84:85] // (op 8962) convert


v_mov_b64 v[100:101], v[104:105] // (op 8962) 
s_add_u32 s84, 32768, s54 // (op 8962) 
s_mov_b32 m0, s84 // (op 8962) reg expression
s_mov_b32 s85, 384 // (op 8962) call()
buffer_load_dwordx4 v100, s[68:71], s85 offen offset: 0 lds // (op 8962) Load value direct to lds
v_add_co_u32 v100, s[86:87], v100, v102 // (op 8962) least significant half
v_addc_co_u32 v101, s[86:87], 0, v101, s[86:87] // (op 8962) most significant half

s_add_u32 m0, m0, 4096 // (op 8962) 
buffer_load_dwordx4 v100, s[68:71], s85 offen offset: 0 lds // (op 8962) Load value direct to lds
v_add_co_u32 v100, s[86:87], v100, v102 // (op 8962) least significant half
v_addc_co_u32 v101, s[86:87], 0, v101, s[86:87] // (op 8962) most significant half

s_add_u32 m0, m0, 4096 // (op 8962) 
buffer_load_dwordx4 v100, s[68:71], s85 offen offset: 0 lds // (op 8962) Load value direct to lds
v_add_co_u32 v100, s[86:87], v100, v102 // (op 8962) least significant half
v_addc_co_u32 v101, s[86:87], 0, v101, s[86:87] // (op 8962) most significant half

s_add_u32 m0, m0, 4096 // (op 8962) 
buffer_load_dwordx4 v100, s[68:71], s85 offen offset: 0 lds // (op 8962) Load value direct to lds






v_add_co_u32 v34, vcc, 256, v34 // (op 9131) least significant half
v_addc_co_u32 v35, vcc, 0, v35, vcc // (op 9131) most significant half
v_mov_b64 v[100:101], s[40:41] // (op 2639) convert

v_mov_b32 v43, v94 // (op 2639) 
buffer_load_ubyte v36, v43, s[76:79], 0 offen offset:12 // (op 2639) Load value
v_add_u32 v43, v43, v100 // (op 2639) 
buffer_load_ubyte v37, v43, s[76:79], 0 offen offset:12 // (op 2639) Load value



s_barrier  // (op 10819) 
v_add_u32 v30, 8, v30 // (op 9387) 



s_waitcnt vmcnt(15)
ds_write_b8 v96, v26 offset:49152 // (op 30) Store local data 
s_barrier  // (op 10798) 



s_waitcnt vmcnt(10)
ds_write_b8 v95, v44 offset:49408 // (op 21) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v95, v45 offset:49412 // (op 21) Store local data 
s_barrier  // (op 10784) 



ds_read_b128 v[46:49], v4 // (op 77) Load local data 
ds_read_b128 v[50:53], v4 offset:64 // (op 77) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 10826) 



ds_read_u8 v29, v5 offset:49152 // (op 87) Load local data 



ds_read_b128 v[54:57], v4 offset:4096 // (op 2753) Load local data 
ds_read_b128 v[58:61], v4 offset:4160 // (op 2753) Load local data 



ds_read_u8 v31, v5 offset:49280 // (op 2757) Load local data 
s_barrier  // (op 10765) 



ds_read_b128 v[62:65], v3 offset:8192 // (op 72) Load local data 
ds_read_b128 v[66:69], v3 offset:8256 // (op 72) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 10805) 



ds_read_u8 v32, v0 offset:49408 // (op 82) Load local data 



ds_read_b128 v[70:73], v3 offset:12288 // (op 3004) Load local data 
ds_read_b128 v[74:77], v3 offset:12352 // (op 3004) Load local data 



ds_read_u8 v33, v0 offset:49536 // (op 3007) Load local data 



ds_read_b128 v[78:81], v3 offset:16384 // (op 3390) Load local data 
ds_read_b128 v[82:85], v3 offset:16448 // (op 3390) Load local data 



ds_read_u8 v41, v0 offset:49664 // (op 3393) Load local data 



ds_read_b128 v[86:89], v3 offset:20480 // (op 3776) Load local data 
ds_read_b128 v[90:93], v3 offset:20544 // (op 3776) Load local data 



ds_read_u8 v42, v0 offset:49792 // (op 3779) Load local data 
s_and_b32 s84, s7, 4294967294 // (op 3742) 
s_cmp_lt_u32 s1, s84 // (op 3742) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoop_3742 // (op 3742) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoop_3742 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoop_3742:
















s_barrier  // (op 11227) 
s_and_b32 s52, s7, 4294967294 // (op 9035) 
v_lshrrev_b32 v26, 3, v1 // (op 9035) 
s_add_u32 s53, s0, s52 // (op 9035) 

v_and_b32 v29, -8, v1 // (op 9035) 
v_sub_u32 v30, v1, v29 // (op 9035) 

s_mul_i32 s52, s16, s8 // (op 9035) 
s_add_u32 s52, s52, s53 // (op 9035) 

v_lshlrev_b32 v29, 4, v30 // (op 9035) 

s_mul_hi_u32 s53, s52, s13 // (op 9035) 
s_sub_u32 s54, s52, s53 // (op 9035) 
s_lshr_b32 s55, s54, 1 // (op 9035) 

s_add_u32 s54, s55, s53 // (op 9035) 


s_lshr_b32 s53, s54, s11 // (op 9035) 

s_and_b32 s54, s11, 2147483648 // (op 9035) 
s_cmp_eq_i32 s54, 0 // (op 9035) 
s_mov_b32 s55, scc // (op 9035) 

s_cmp_eq_i32 s55, 0 // (op 9035) 
s_cselect_b32 s54, s52, s53 // (op 9035) 


s_mul_hi_u32 s53, s54, s12 // (op 9035) 
s_sub_u32 s55, s54, s53 // (op 9035) 
s_lshr_b32 s84, s55, 1 // (op 9035) 

s_add_u32 s55, s84, s53 // (op 9035) 


s_lshr_b32 s53, s55, s10 // (op 9035) 

s_and_b32 s55, s10, 2147483648 // (op 9035) 
s_cmp_eq_i32 s55, 0 // (op 9035) 
s_mov_b32 s84, scc // (op 9035) 

s_cmp_eq_i32 s84, 0 // (op 9035) 
s_cselect_b32 s55, s54, s53 // (op 9035) 


v_lshl_add_u32 v30, s55, 7, v26 // (op 9035) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 9035) 
s_mul_i32 s53, s54, s84 // (op 9035) 


s_sub_u32 s54, s52, s53 // (op 9035) 


v_lshl_add_u32 v26, s54, 7, v29 // (op 9035) 


v_mov_b32 v35, 0 // (op 9035) convert
v_mov_b32 v34, v26 // (op 9035) convert

v_mul_lo_u32 v39, v30, s31 // (op 9035) most significant: low of low * high
v_mul_hi_u32 v26, v30, s30 // (op 9035) most significant: high of low * low
v_mul_lo_u32 v38, v30, s30 // (op 9035) least significant: low of low * low
v_add_u32 v39, v39, v26 // (op 9035) most significant: sum

v_add_co_u32 v38, s[52:53], v38, v34 // (op 9035) least significant half
v_addc_co_u32 v39, s[52:53], v39, v35, s[52:53] // (op 9035) most significant half



s_mul_i32 s53, s30, s35 // (op 9037) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 9037) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 9037) least significant
s_mul_hi_u32 s55, s30, s34 // (op 9037) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9037) most significant: sum
s_add_u32 s53, s53, s55 // (op 9037) most significant: sum


s_mov_b32 s68, s32 // (op 9037) reg expression
s_mov_b32 s69, s33 // (op 9037) reg expression
s_mov_b32 s70, s52 // (op 9037) reg expression
s_mov_b32 s71, 131072 // (op 9037) call()

s_and_b32 s52, s7, 4294967294 // (op 9183) 
v_lshrrev_b32 v26, 3, v1 // (op 9183) 
s_add_u32 s53, s0, s52 // (op 9183) 

v_and_b32 v40, -8, v1 // (op 9183) 
v_sub_u32 v29, v1, v40 // (op 9183) 

s_mul_i32 s52, s16, s8 // (op 9183) 
s_add_u32 s52, s52, s53 // (op 9183) 

v_lshlrev_b32 v30, 4, v29 // (op 9183) 

s_mul_hi_u32 s53, s52, s13 // (op 9183) 
s_sub_u32 s54, s52, s53 // (op 9183) 
s_lshr_b32 s55, s54, 1 // (op 9183) 

s_add_u32 s54, s55, s53 // (op 9183) 


s_lshr_b32 s53, s54, s11 // (op 9183) 

s_and_b32 s54, s11, 2147483648 // (op 9183) 
s_cmp_eq_i32 s54, 0 // (op 9183) 
s_mov_b32 s55, scc // (op 9183) 

s_cmp_eq_i32 s55, 0 // (op 9183) 
s_cselect_b32 s54, s52, s53 // (op 9183) 


s_mul_hi_u32 s53, s54, s12 // (op 9183) 
s_sub_u32 s55, s54, s53 // (op 9183) 
s_lshr_b32 s84, s55, 1 // (op 9183) 

s_add_u32 s55, s84, s53 // (op 9183) 


s_lshr_b32 s53, s55, s10 // (op 9183) 

s_and_b32 s55, s10, 2147483648 // (op 9183) 
s_cmp_eq_i32 s55, 0 // (op 9183) 
s_mov_b32 s84, scc // (op 9183) 

s_cmp_eq_i32 s84, 0 // (op 9183) 
s_cselect_b32 s55, s54, s53 // (op 9183) 


s_mul_i32 s53, s55, s14 // (op 9183) 

s_sub_u32 s55, s54, s53 // (op 9183) 

v_lshl_add_u32 v29, s55, 6, v26 // (op 9183) 


v_mul_lo_u32 v35, v29, s37 // (op 9183) most significant: low of low * high
v_mul_hi_u32 v26, v29, s36 // (op 9183) most significant: high of low * low
v_mul_lo_u32 v34, v29, s36 // (op 9183) least significant: low of low * low
v_add_u32 v35, v35, v26 // (op 9183) most significant: sum


s_ashr_i64 s[84:85], s[2:3], 7 // (op 9183) 
s_mul_i32 s53, s54, s84 // (op 9183) 


s_sub_u32 s54, s52, s53 // (op 9183) 


v_lshl_add_u32 v26, s54, 7, v30 // (op 9183) 


v_mov_b32 v53, 0 // (op 9183) convert
v_mov_b32 v52, v26 // (op 9183) convert

v_add_co_u32 v50, s[52:53], v52, v34 // (op 9183) least significant half
v_addc_co_u32 v51, s[52:53], v53, v35, s[52:53] // (op 9183) most significant half



s_mul_i32 s53, s36, s23 // (op 9185) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 9185) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 9185) least significant
s_mul_hi_u32 s55, s36, s22 // (op 9185) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9185) most significant: sum
s_add_u32 s53, s53, s55 // (op 9185) most significant: sum


s_mov_b32 s72, s38 // (op 9185) reg expression
s_mov_b32 s73, s39 // (op 9185) reg expression
s_mov_b32 s74, s52 // (op 9185) reg expression
s_mov_b32 s75, 131072 // (op 9185) call()

s_and_b32 s52, s7, 4294967294 // (op 9311) 
v_lshrrev_b32 v26, 2, v1 // (op 9311) 
s_add_u32 s53, s0, s52 // (op 9311) 

v_lshlrev_b32 v40, 1, v26 // (op 9311) 

v_and_b32 v26, -4, v1 // (op 9311) 
s_mul_i32 s52, s16, s8 // (op 9311) 
s_add_u32 s52, s52, s53 // (op 9311) 

v_sub_u32 v29, v1, v26 // (op 9311) 

s_mul_hi_u32 s53, s52, s13 // (op 9311) 
s_sub_u32 s54, s52, s53 // (op 9311) 
s_lshr_b32 s55, s54, 1 // (op 9311) 

s_add_u32 s54, s55, s53 // (op 9311) 


s_lshr_b32 s53, s54, s11 // (op 9311) 

s_and_b32 s54, s11, 2147483648 // (op 9311) 
s_cmp_eq_i32 s54, 0 // (op 9311) 
s_mov_b32 s55, scc // (op 9311) 

s_cmp_eq_i32 s55, 0 // (op 9311) 
s_cselect_b32 s54, s52, s53 // (op 9311) 


s_mul_hi_u32 s53, s54, s12 // (op 9311) 
s_sub_u32 s55, s54, s53 // (op 9311) 
s_lshr_b32 s84, s55, 1 // (op 9311) 

s_add_u32 s55, s84, s53 // (op 9311) 


s_lshr_b32 s53, s55, s10 // (op 9311) 

s_and_b32 s55, s10, 2147483648 // (op 9311) 
s_cmp_eq_i32 s55, 0 // (op 9311) 
s_mov_b32 s84, scc // (op 9311) 

s_cmp_eq_i32 s84, 0 // (op 9311) 
s_cselect_b32 s55, s54, s53 // (op 9311) 


v_lshl_add_u32 v26, s55, 7, v40 // (op 9311) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 9311) 
s_mul_i32 s53, s54, s84 // (op 9311) 


s_sub_u32 s54, s52, s53 // (op 9311) 


v_lshl_add_u32 v30, s54, 2, v29 // (op 9311) 


v_mul_lo_u32 v29, v26, s40 // (op 9311) 
v_add_u32 v29, v29, v30 // (op 9311) 


s_mul_i32 s53, s40, s35 // (op 9313) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 9313) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 9313) least significant
s_mul_hi_u32 s55, s40, s34 // (op 9313) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9313) most significant: sum
s_add_u32 s53, s53, s55 // (op 9313) most significant: sum


s_mov_b32 s76, s42 // (op 9313) reg expression
s_mov_b32 s77, s43 // (op 9313) reg expression
s_mov_b32 s78, s52 // (op 9313) reg expression
s_mov_b32 s79, 131072 // (op 9313) call()

s_and_b32 s52, s7, 4294967294 // (op 9439) 
v_lshrrev_b32 v26, 2, v1 // (op 9439) 
s_add_u32 s53, s0, s52 // (op 9439) 

v_and_b32 v30, -4, v1 // (op 9439) 
v_sub_u32 v40, v1, v30 // (op 9439) 

s_mul_i32 s52, s16, s8 // (op 9439) 
s_add_u32 s52, s52, s53 // (op 9439) 

s_mul_hi_u32 s53, s52, s13 // (op 9439) 
s_sub_u32 s54, s52, s53 // (op 9439) 
s_lshr_b32 s55, s54, 1 // (op 9439) 

s_add_u32 s54, s55, s53 // (op 9439) 


s_lshr_b32 s53, s54, s11 // (op 9439) 

s_and_b32 s54, s11, 2147483648 // (op 9439) 
s_cmp_eq_i32 s54, 0 // (op 9439) 
s_mov_b32 s55, scc // (op 9439) 

s_cmp_eq_i32 s55, 0 // (op 9439) 
s_cselect_b32 s54, s52, s53 // (op 9439) 


s_mul_hi_u32 s53, s54, s12 // (op 9439) 
s_sub_u32 s55, s54, s53 // (op 9439) 
s_lshr_b32 s84, s55, 1 // (op 9439) 

s_add_u32 s55, s84, s53 // (op 9439) 


s_lshr_b32 s53, s55, s10 // (op 9439) 

s_and_b32 s55, s10, 2147483648 // (op 9439) 
s_cmp_eq_i32 s55, 0 // (op 9439) 
s_mov_b32 s84, scc // (op 9439) 

s_cmp_eq_i32 s84, 0 // (op 9439) 
s_cselect_b32 s55, s54, s53 // (op 9439) 


s_mul_i32 s53, s55, s14 // (op 9439) 

s_sub_u32 s55, s54, s53 // (op 9439) 

v_lshl_add_u32 v30, s55, 6, v26 // (op 9439) 


v_mul_lo_u32 v26, v30, s44 // (op 9439) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 9439) 
s_mul_i32 s53, s54, s84 // (op 9439) 


s_sub_u32 s54, s52, s53 // (op 9439) 


v_lshl_add_u32 v30, s54, 2, v40 // (op 9439) 


v_add_u32 v40, v30, v26 // (op 9439) 


s_mul_i32 s53, s44, s23 // (op 9441) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 9441) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 9441) least significant
s_mul_hi_u32 s55, s44, s22 // (op 9441) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9441) most significant: sum
s_add_u32 s53, s53, s55 // (op 9441) most significant: sum


s_mov_b32 s80, s46 // (op 9441) reg expression
s_mov_b32 s81, s47 // (op 9441) reg expression
s_mov_b32 s82, s52 // (op 9441) reg expression
s_mov_b32 s83, 131072 // (op 9441) call()

v_and_b32 v26, -8, v1 // (op 9610) 
v_sub_u32 v30, v1, v26 // (op 9610) 

v_lshlrev_b32 v26, 4, v30 // (op 9610) 

v_lshrrev_b32 v30, 3, v1 // (op 9610) 
v_lshl_add_u32 v34, v30, 7, v26 // (op 9610) 


s_nop 0
v_readfirstlane_b32 s52, v34 // (op 9610) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -8, v1 // (op 9685) 
v_sub_u32 v30, v1, v26 // (op 9685) 

v_lshlrev_b32 v26, 4, v30 // (op 9685) 

v_lshrrev_b32 v30, 3, v1 // (op 9685) 
v_lshl_add_u32 v34, v30, 7, v26 // (op 9685) 


s_nop 0
v_readfirstlane_b32 s53, v34 // (op 9685) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 9760) 
v_sub_u32 v30, v1, v26 // (op 9760) 

v_lshrrev_b32 v26, 2, v1 // (op 9760) 
v_lshl_add_u32 v34, v26, 3, v30 // (op 9760) 


v_and_b32 v26, -4, v1 // (op 9836) 
v_sub_u32 v30, v1, v26 // (op 9836) 

v_and_b32 v26, 4294967292, v1 // (op 9836) 
v_add_u32 v35, v26, v30 // (op 9836) 


s_and_b32 s54, s7, 4294967294 // (op 3748) 
s_cmp_lt_u32 s54, s7 // (op 3747) 












s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoopTail_3747 // (op 3747) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoopTail_3747 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoopTail_3747:
s_barrier  // (op 10762) 
s_add_u32 s54, s54, 1 // (op 3749) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 8949) 
v_mov_b64 v[36:37], s[84:85] // (op 8949) convert


v_mov_b64 v[30:31], v[38:39] // (op 8949) 
v_mov_b64 v[32:33], v[30:31] // (op 8949) 
s_add_u32 s55, 8192, s52 // (op 8949) 
s_mov_b32 m0, s55 // (op 8949) reg expression
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 8949) Load value direct to lds
v_add_co_u32 v32, s[84:85], v32, v36 // (op 8949) least significant half
v_addc_co_u32 v33, s[84:85], 0, v33, s[84:85] // (op 8949) most significant half

s_add_u32 m0, m0, 4096 // (op 8949) 
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 8949) Load value direct to lds
v_add_co_u32 v32, s[84:85], v32, v36 // (op 8949) least significant half
v_addc_co_u32 v33, s[84:85], 0, v33, s[84:85] // (op 8949) most significant half

s_add_u32 m0, m0, 4096 // (op 8949) 
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 8949) Load value direct to lds
v_add_co_u32 v32, s[84:85], v32, v36 // (op 8949) least significant half
v_addc_co_u32 v33, s[84:85], 0, v33, s[84:85] // (op 8949) most significant half

s_add_u32 m0, m0, 4096 // (op 8949) 
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 8949) Load value direct to lds




v_add_co_u32 v38, vcc, 128, v38 // (op 9025) least significant half
v_addc_co_u32 v39, vcc, 0, v39, vcc // (op 9025) most significant half
s_barrier  // (op 10781) 

s_lshl_b64 s[84:85], s[36:37], 5 // (op 8955) 
v_mov_b64 v[36:37], s[84:85] // (op 8955) convert


v_mov_b64 v[30:31], v[50:51] // (op 8955) 
v_mov_b64 v[32:33], v[30:31] // (op 8955) 
s_mov_b32 m0, s53 // (op 8955) reg expression
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 8955) Load value direct to lds
v_add_co_u32 v32, s[84:85], v32, v36 // (op 8955) least significant half
v_addc_co_u32 v33, s[84:85], 0, v33, s[84:85] // (op 8955) most significant half

s_add_u32 m0, m0, 4096 // (op 8955) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 8955) Load value direct to lds



v_add_co_u32 v50, vcc, 128, v50 // (op 9175) least significant half
v_addc_co_u32 v51, vcc, 0, v51, vcc // (op 9175) most significant half

v_mov_b64 v[36:37], s[40:41] // (op 2573) convert

v_mov_b32 v26, v29 // (op 2573) 
v_mov_b32 v28, v26 // (op 2573) 
buffer_load_ubyte v30, v28, s[76:79], 0 offen offset: 0 // (op 2573) Load value
v_add_u32 v28, v28, v36 // (op 2573) 
buffer_load_ubyte v31, v28, s[76:79], 0 offen offset: 0 // (op 2573) Load value



v_add_u32 v29, 4, v29 // (op 9303) 


s_barrier  // (op 10801) 
v_mov_b32 v26, v40 // (op 2574) 




buffer_load_ubyte v28, v26, s[80:83], 0 offen offset: 0 // (op 2574) Load value

v_add_u32 v40, 4, v40 // (op 9431) 
s_waitcnt vmcnt(2)
ds_write_b8 v34, v30 offset:49408 // (op 2570) Store local data 
s_barrier  // (op 10768) 
s_waitcnt vmcnt(1)
ds_write_b8 v34, v31 offset:49412 // (op 2570) Store local data 




ds_read_b128 v[42:45], v3 offset:8192 // (op 2561) Load local data 
s_barrier  // (op 10822) 
ds_read_b128 v[46:49], v3 offset:8256 // (op 2561) Load local data 



s_barrier  // (op 10788) 
s_waitcnt vmcnt(0)
ds_write_b8 v35, v28 offset:49152 // (op 2571) Store local data 




ds_read_b128 v[52:55], v4 // (op 2563) Load local data 
ds_read_b128 v[56:59], v4 offset:64 // (op 2563) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 10809) 



ds_read_u8 v26, v0 offset:49408 // (op 2565) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10829) 



ds_read_u8 v28, v5 offset:49152 // (op 2567) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[52:59], a[0:3], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2568) 
ds_read_b128 v[60:63], v4 offset:4096 // (op 2732) Load local data 
ds_read_b128 v[64:67], v4 offset:4160 // (op 2732) Load local data 



ds_read_u8 v41, v5 offset:49280 // (op 2736) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:49], v[60:67], a[4:7], v26, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2737) 
ds_read_b128 v[68:71], v3 offset:12288 // (op 2950) Load local data 


ds_read_b128 v[72:75], v3 offset:12352 // (op 2950) Load local data 



ds_read_u8 v26, v0 offset:49536 // (op 2953) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[68:75], v[52:59], a[8:11], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2955) 
ds_read_b128 v[42:45], v3 offset:16384 // (op 3336) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:75], v[60:67], a[12:15], v26, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2986) 
ds_read_b128 v[46:49], v3 offset:16448 // (op 3336) Load local data 





ds_read_u8 v26, v0 offset:49664 // (op 3339) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[42:49], v[52:59], a[16:19], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3341) 
ds_read_b128 v[68:71], v3 offset:20480 // (op 3722) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[42:49], v[60:67], a[20:23], v26, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3372) 
ds_read_b128 v[72:75], v3 offset:20544 // (op 3722) Load local data 





ds_read_u8 v26, v0 offset:49792 // (op 3725) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[68:75], v[52:59], a[24:27], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3727) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[68:75], v[60:67], a[28:31], v26, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 3758) 




s_cmp_lt_u32 s54, s7 // (op 3747) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoopTail_3747 // (op 3747) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoopTail_3747 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoopTail_3747:









s_barrier  // (op 11231) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 296) 
s_cmp_lg_u32 s18, 0 // (op 322) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Send_Tile_322 // (op 322) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Send_Tile_322



v_add_u32 v26, 24576, v25 // (op 4292) 
buffer_store_dword a24, v26, s[64:67], 0 offen sc1 // (op 4292) Store value

v_add_u32 v26, 24832, v25 // (op 4292) 
buffer_store_dword a25, v26, s[64:67], 0 offen sc1 // (op 4292) Store value

v_add_u32 v26, 25088, v25 // (op 4292) 
buffer_store_dword a26, v26, s[64:67], 0 offen sc1 // (op 4292) Store value

v_add_u32 v26, 25344, v25 // (op 4292) 
buffer_store_dword a27, v26, s[64:67], 0 offen sc1 // (op 4292) Store value




v_add_u32 v26, 24704, v25 // (op 4294) 
buffer_store_dword a28, v26, s[64:67], 0 offen sc1 // (op 4294) Store value

v_add_u32 v26, 24960, v25 // (op 4294) 
buffer_store_dword a29, v26, s[64:67], 0 offen sc1 // (op 4294) Store value

v_add_u32 v26, 25216, v25 // (op 4294) 
buffer_store_dword a30, v26, s[64:67], 0 offen sc1 // (op 4294) Store value

v_add_u32 v26, 25472, v25 // (op 4294) 
buffer_store_dword a31, v26, s[64:67], 0 offen sc1 // (op 4294) Store value




v_add_u32 v26, 16384, v25 // (op 4278) 
buffer_store_dword a16, v26, s[64:67], 0 offen sc1 // (op 4278) Store value

v_add_u32 v26, 16640, v25 // (op 4278) 
buffer_store_dword a17, v26, s[64:67], 0 offen sc1 // (op 4278) Store value

v_add_u32 v26, 16896, v25 // (op 4278) 
buffer_store_dword a18, v26, s[64:67], 0 offen sc1 // (op 4278) Store value

v_add_u32 v26, 17152, v25 // (op 4278) 
buffer_store_dword a19, v26, s[64:67], 0 offen sc1 // (op 4278) Store value




v_add_u32 v26, 16512, v25 // (op 4280) 
buffer_store_dword a20, v26, s[64:67], 0 offen sc1 // (op 4280) Store value

v_add_u32 v26, 16768, v25 // (op 4280) 
buffer_store_dword a21, v26, s[64:67], 0 offen sc1 // (op 4280) Store value

v_add_u32 v26, 17024, v25 // (op 4280) 
buffer_store_dword a22, v26, s[64:67], 0 offen sc1 // (op 4280) Store value

v_add_u32 v26, 17280, v25 // (op 4280) 
buffer_store_dword a23, v26, s[64:67], 0 offen sc1 // (op 4280) Store value




v_add_u32 v26, 8192, v25 // (op 4264) 
buffer_store_dword a8, v26, s[64:67], 0 offen sc1 // (op 4264) Store value

v_add_u32 v26, 8448, v25 // (op 4264) 
buffer_store_dword a9, v26, s[64:67], 0 offen sc1 // (op 4264) Store value

v_add_u32 v26, 8704, v25 // (op 4264) 
buffer_store_dword a10, v26, s[64:67], 0 offen sc1 // (op 4264) Store value

v_add_u32 v26, 8960, v25 // (op 4264) 
buffer_store_dword a11, v26, s[64:67], 0 offen sc1 // (op 4264) Store value




v_add_u32 v26, 8320, v25 // (op 4266) 
buffer_store_dword a12, v26, s[64:67], 0 offen sc1 // (op 4266) Store value

v_add_u32 v26, 8576, v25 // (op 4266) 
buffer_store_dword a13, v26, s[64:67], 0 offen sc1 // (op 4266) Store value

v_add_u32 v26, 8832, v25 // (op 4266) 
buffer_store_dword a14, v26, s[64:67], 0 offen sc1 // (op 4266) Store value

v_add_u32 v26, 9088, v25 // (op 4266) 
buffer_store_dword a15, v26, s[64:67], 0 offen sc1 // (op 4266) Store value




buffer_store_dword a0, v25, s[64:67], 0 offen sc1 // (op 319) Store value
buffer_store_dword a1, v25, s[64:67], 0 offen offset:256 sc1 // (op 319) Store value
buffer_store_dword a2, v25, s[64:67], 0 offen offset:512 sc1 // (op 319) Store value
buffer_store_dword a3, v25, s[64:67], 0 offen offset:768 sc1 // (op 319) Store value



buffer_store_dword a4, v25, s[64:67], 0 offen offset:128 sc1 // (op 4250) Store value
buffer_store_dword a5, v25, s[64:67], 0 offen offset:384 sc1 // (op 4250) Store value
buffer_store_dword a6, v25, s[64:67], 0 offen offset:640 sc1 // (op 4250) Store value
buffer_store_dword a7, v25, s[64:67], 0 offen offset:896 sc1 // (op 4250) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 323) 
v_cmp_eq_i32 vcc, v1, 0 // (op 327) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_327 // (op 327) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_327
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
s_branch RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_327 // (op 327) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_327
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_327:
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_327:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Send_Tile_322 // (op 322) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Send_Tile_322
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Send_Tile_322:
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Send_Tile_322:

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
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Receive_Tile_338 // (op 338) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Receive_Tile_338
s_mov_b32 s52, 0 // (op 304) call()
s_mul_i32 s53, s52, s8 // (op 303) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 303) 
s_mov_b32 s84, s54 // (op 303) convert

s_sub_u32 s54, s84, s19 // (op 303) 

s_sub_u32 s55, s54, 1 // (op 303) 

s_cmp_lt_u32 s53, s55 // (op 303) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_303 // (op 303) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_303 if false)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_ReceiveTileLoop_303:
s_add_u32 s53, s16, 1 // (op 10290) 
v_lshrrev_b32 v20, 5, v1 // (op 10290) 
s_add_u32 s54, s53, s52 // (op 10290) 
s_lshl_b32 s54, s54, 7 // (op 10290) 

v_lshlrev_b32 v21, 2, v20 // (op 10290) 

v_add_lshl_u32 v20, s54, v21, 6 // (op 10290) 


v_and_b32 v21, -32, v1 // (op 10290) 
v_sub_u32 v23, v1, v21 // (op 10290) 

v_add_lshl_u32 v21, v20, v23, 2 // (op 10290) 


s_lshl_b32 s53, s15, 15 // (op 10292) 
s_mov_b32 s64, s28 // (op 10292) reg expression
s_mov_b32 s65, s29 // (op 10292) reg expression
s_mov_b32 s66, s53 // (op 10292) reg expression
s_mov_b32 s67, 131072 // (op 10292) call()

RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_342:
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
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_342 // (op 342) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_342 if true)
s_barrier  // (op 346) 
v_cmp_eq_i32 vcc, v1, 0 // (op 345) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_345 // (op 345) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_345
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
s_branch RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_345 // (op 345) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_345
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_345:
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_345:




buffer_load_dword v28, v21, s[64:67], 0 offen // (op 357) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:256 // (op 357) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:512 // (op 357) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:768 // (op 357) Load value
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




buffer_load_dword v28, v21, s[64:67], 0 offen offset:128 // (op 4333) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:384 // (op 4333) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:640 // (op 4333) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:896 // (op 4333) Load value
v_accvgpr_read v32, a4 // (op 4334) reg expression
v_accvgpr_read v33, a5 // (op 4334) reg expression
v_accvgpr_read v34, a6 // (op 4334) reg expression
v_accvgpr_read v35, a7 // (op 4334) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4335) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4335) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4335) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4335) 

v_accvgpr_write a4, v32 // (op 4336) reg expression
v_accvgpr_write a5, v33 // (op 4336) reg expression
v_accvgpr_write a6, v34 // (op 4336) reg expression
v_accvgpr_write a7, v35 // (op 4336) reg expression




v_mov_b32 v20, 8192 // (op 4358) 
v_add_i32 v23, v21, v20 // (op 4358) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4358) Load value

v_mov_b32 v20, 8448 // (op 4358) 
v_add_i32 v23, v21, v20 // (op 4358) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4358) Load value

v_mov_b32 v20, 8704 // (op 4358) 
v_add_i32 v23, v21, v20 // (op 4358) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4358) Load value

v_mov_b32 v20, 8960 // (op 4358) 
v_add_i32 v23, v21, v20 // (op 4358) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4358) Load value

v_accvgpr_read v32, a8 // (op 4355) reg expression
v_accvgpr_read v33, a9 // (op 4355) reg expression
v_accvgpr_read v34, a10 // (op 4355) reg expression
v_accvgpr_read v35, a11 // (op 4355) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4356) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4356) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4356) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4356) 

v_accvgpr_write a8, v32 // (op 4357) reg expression
v_accvgpr_write a9, v33 // (op 4357) reg expression
v_accvgpr_write a10, v34 // (op 4357) reg expression
v_accvgpr_write a11, v35 // (op 4357) reg expression




v_mov_b32 v20, 8320 // (op 4363) 
v_add_i32 v23, v21, v20 // (op 4363) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4363) Load value

v_mov_b32 v20, 8576 // (op 4363) 
v_add_i32 v23, v21, v20 // (op 4363) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4363) Load value

v_mov_b32 v20, 8832 // (op 4363) 
v_add_i32 v23, v21, v20 // (op 4363) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4363) Load value

v_mov_b32 v20, 9088 // (op 4363) 
v_add_i32 v23, v21, v20 // (op 4363) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4363) Load value

v_accvgpr_read v32, a12 // (op 4360) reg expression
v_accvgpr_read v33, a13 // (op 4360) reg expression
v_accvgpr_read v34, a14 // (op 4360) reg expression
v_accvgpr_read v35, a15 // (op 4360) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4361) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4361) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4361) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4361) 

v_accvgpr_write a12, v32 // (op 4362) reg expression
v_accvgpr_write a13, v33 // (op 4362) reg expression
v_accvgpr_write a14, v34 // (op 4362) reg expression
v_accvgpr_write a15, v35 // (op 4362) reg expression




v_mov_b32 v20, 16384 // (op 4384) 
v_add_i32 v23, v21, v20 // (op 4384) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4384) Load value

v_mov_b32 v20, 16640 // (op 4384) 
v_add_i32 v23, v21, v20 // (op 4384) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4384) Load value

v_mov_b32 v20, 16896 // (op 4384) 
v_add_i32 v23, v21, v20 // (op 4384) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4384) Load value

v_mov_b32 v20, 17152 // (op 4384) 
v_add_i32 v23, v21, v20 // (op 4384) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4384) Load value

v_accvgpr_read v32, a16 // (op 4381) reg expression
v_accvgpr_read v33, a17 // (op 4381) reg expression
v_accvgpr_read v34, a18 // (op 4381) reg expression
v_accvgpr_read v35, a19 // (op 4381) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4382) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4382) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4382) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4382) 

v_accvgpr_write a16, v32 // (op 4383) reg expression
v_accvgpr_write a17, v33 // (op 4383) reg expression
v_accvgpr_write a18, v34 // (op 4383) reg expression
v_accvgpr_write a19, v35 // (op 4383) reg expression




v_mov_b32 v20, 16512 // (op 4389) 
v_add_i32 v23, v21, v20 // (op 4389) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4389) Load value

v_mov_b32 v20, 16768 // (op 4389) 
v_add_i32 v23, v21, v20 // (op 4389) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4389) Load value

v_mov_b32 v20, 17024 // (op 4389) 
v_add_i32 v23, v21, v20 // (op 4389) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4389) Load value

v_mov_b32 v20, 17280 // (op 4389) 
v_add_i32 v23, v21, v20 // (op 4389) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4389) Load value

v_accvgpr_read v32, a20 // (op 4386) reg expression
v_accvgpr_read v33, a21 // (op 4386) reg expression
v_accvgpr_read v34, a22 // (op 4386) reg expression
v_accvgpr_read v35, a23 // (op 4386) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4387) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4387) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4387) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4387) 

v_accvgpr_write a20, v32 // (op 4388) reg expression
v_accvgpr_write a21, v33 // (op 4388) reg expression
v_accvgpr_write a22, v34 // (op 4388) reg expression
v_accvgpr_write a23, v35 // (op 4388) reg expression




v_mov_b32 v20, 24576 // (op 4410) 
v_add_i32 v23, v21, v20 // (op 4410) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4410) Load value

v_mov_b32 v20, 24832 // (op 4410) 
v_add_i32 v23, v21, v20 // (op 4410) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4410) Load value

v_mov_b32 v20, 25088 // (op 4410) 
v_add_i32 v23, v21, v20 // (op 4410) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4410) Load value

v_mov_b32 v20, 25344 // (op 4410) 
v_add_i32 v23, v21, v20 // (op 4410) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4410) Load value

v_accvgpr_read v32, a24 // (op 4407) reg expression
v_accvgpr_read v33, a25 // (op 4407) reg expression
v_accvgpr_read v34, a26 // (op 4407) reg expression
v_accvgpr_read v35, a27 // (op 4407) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4408) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4408) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4408) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4408) 

v_accvgpr_write a24, v32 // (op 4409) reg expression
v_accvgpr_write a25, v33 // (op 4409) reg expression
v_accvgpr_write a26, v34 // (op 4409) reg expression
v_accvgpr_write a27, v35 // (op 4409) reg expression




v_mov_b32 v20, 24704 // (op 4415) 
v_add_i32 v23, v21, v20 // (op 4415) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4415) Load value

v_mov_b32 v20, 24960 // (op 4415) 
v_add_i32 v23, v21, v20 // (op 4415) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4415) Load value

v_mov_b32 v20, 25216 // (op 4415) 
v_add_i32 v23, v21, v20 // (op 4415) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4415) Load value

v_mov_b32 v20, 25472 // (op 4415) 
v_add_i32 v23, v21, v20 // (op 4415) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4415) Load value

v_accvgpr_read v32, a28 // (op 4412) reg expression
v_accvgpr_read v33, a29 // (op 4412) reg expression
v_accvgpr_read v34, a30 // (op 4412) reg expression
v_accvgpr_read v35, a31 // (op 4412) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4413) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4413) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4413) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4413) 

v_accvgpr_write a28, v32 // (op 4414) reg expression
v_accvgpr_write a29, v33 // (op 4414) reg expression
v_accvgpr_write a30, v34 // (op 4414) reg expression
v_accvgpr_write a31, v35 // (op 4414) reg expression

s_add_u32 s52, s52, 1 // (op 305) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 303) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 303) 
s_mov_b32 s84, s54 // (op 303) convert

s_sub_u32 s54, s84, s19 // (op 303) 

s_sub_u32 s55, s54, 1 // (op 303) 

s_cmp_lt_u32 s53, s55 // (op 303) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_ReceiveTileLoop_303 // (op 303) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_ReceiveTileLoop_303 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_303:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Receive_Tile_338 // (op 338) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Receive_Tile_338
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Receive_Tile_338:
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Receive_Tile_338:

s_cmp_eq_i32 s18, 0 // (op 381) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_381 // (op 381) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_381
v_cmp_eq_f32 vcc, 0.00000, v18 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_0_beta_135
v_accvgpr_read v28, a0 // (op 50) 
v_accvgpr_read v29, a1 // (op 50) 
v_accvgpr_read v30, a2 // (op 50) 
v_accvgpr_read v31, a3 // (op 50) 
v_mul_f32 v32, v19, v28 // (op 50) 
v_mul_f32 v33, v19, v29 // (op 50) 
v_mul_f32 v34, v19, v30 // (op 50) 
v_mul_f32 v35, v19, v31 // (op 50) 

v_cvt_f16_f32 v28, v32 // (op 56) 
v_cvt_f16_f32 v29, v33 // (op 56) 
v_cvt_f16_f32 v30, v34 // (op 56) 
v_cvt_f16_f32 v31, v35 // (op 56) 


v_pack_b32_f16 v20, v28, v29 // (op 58) 
v_pack_b32_f16 v21, v30, v31 // (op 58) 

v_mov_b32 v23, v27 // (op 58) reg expression

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 58) Store value



v_accvgpr_read v28, a4 // (op 4461) 
v_accvgpr_read v29, a5 // (op 4461) 
v_accvgpr_read v30, a6 // (op 4461) 
v_accvgpr_read v31, a7 // (op 4461) 
v_mul_f32 v32, v19, v28 // (op 4461) 
v_mul_f32 v33, v19, v29 // (op 4461) 
v_mul_f32 v34, v19, v30 // (op 4461) 
v_mul_f32 v35, v19, v31 // (op 4461) 

v_cvt_f16_f32 v28, v32 // (op 4463) 
v_cvt_f16_f32 v29, v33 // (op 4463) 
v_cvt_f16_f32 v30, v34 // (op 4463) 
v_cvt_f16_f32 v31, v35 // (op 4463) 


v_pack_b32_f16 v20, v28, v29 // (op 4464) 
v_pack_b32_f16 v21, v30, v31 // (op 4464) 

s_lshl_b32 s52, s24, 6 // (op 4464) 
v_add_u32 v23, v27, s52 // (op 4464) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4464) Store value



v_accvgpr_read v28, a8 // (op 4487) 
v_accvgpr_read v29, a9 // (op 4487) 
v_accvgpr_read v30, a10 // (op 4487) 
v_accvgpr_read v31, a11 // (op 4487) 
v_mul_f32 v32, v19, v28 // (op 4487) 
v_mul_f32 v33, v19, v29 // (op 4487) 
v_mul_f32 v34, v19, v30 // (op 4487) 
v_mul_f32 v35, v19, v31 // (op 4487) 

v_cvt_f16_f32 v28, v32 // (op 4489) 
v_cvt_f16_f32 v29, v33 // (op 4489) 
v_cvt_f16_f32 v30, v34 // (op 4489) 
v_cvt_f16_f32 v31, v35 // (op 4489) 


v_pack_b32_f16 v20, v28, v29 // (op 4491) 
v_pack_b32_f16 v21, v30, v31 // (op 4491) 

v_add_u32 v23, 64, v27 // (op 4491) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4491) Store value



v_accvgpr_read v28, a12 // (op 4497) 
v_accvgpr_read v29, a13 // (op 4497) 
v_accvgpr_read v30, a14 // (op 4497) 
v_accvgpr_read v31, a15 // (op 4497) 
v_mul_f32 v32, v19, v28 // (op 4497) 
v_mul_f32 v33, v19, v29 // (op 4497) 
v_mul_f32 v34, v19, v30 // (op 4497) 
v_mul_f32 v35, v19, v31 // (op 4497) 

v_cvt_f16_f32 v28, v32 // (op 4499) 
v_cvt_f16_f32 v29, v33 // (op 4499) 
v_cvt_f16_f32 v30, v34 // (op 4499) 
v_cvt_f16_f32 v31, v35 // (op 4499) 


v_pack_b32_f16 v20, v28, v29 // (op 4493) 
v_pack_b32_f16 v21, v30, v31 // (op 4493) 

s_lshl_b32 s52, s24, 6 // (op 4493) 
s_add_u32 s52, s52, 64 // (op 4493) 
v_add_u32 v23, v27, s52 // (op 4493) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4493) Store value



v_accvgpr_read v28, a16 // (op 4527) 
v_accvgpr_read v29, a17 // (op 4527) 
v_accvgpr_read v30, a18 // (op 4527) 
v_accvgpr_read v31, a19 // (op 4527) 
v_mul_f32 v32, v19, v28 // (op 4527) 
v_mul_f32 v33, v19, v29 // (op 4527) 
v_mul_f32 v34, v19, v30 // (op 4527) 
v_mul_f32 v35, v19, v31 // (op 4527) 

v_cvt_f16_f32 v28, v32 // (op 4529) 
v_cvt_f16_f32 v29, v33 // (op 4529) 
v_cvt_f16_f32 v30, v34 // (op 4529) 
v_cvt_f16_f32 v31, v35 // (op 4529) 


v_pack_b32_f16 v20, v28, v29 // (op 4531) 
v_pack_b32_f16 v21, v30, v31 // (op 4531) 

v_add_u32 v23, 128, v27 // (op 4531) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4531) Store value



v_accvgpr_read v28, a20 // (op 4537) 
v_accvgpr_read v29, a21 // (op 4537) 
v_accvgpr_read v30, a22 // (op 4537) 
v_accvgpr_read v31, a23 // (op 4537) 
v_mul_f32 v32, v19, v28 // (op 4537) 
v_mul_f32 v33, v19, v29 // (op 4537) 
v_mul_f32 v34, v19, v30 // (op 4537) 
v_mul_f32 v35, v19, v31 // (op 4537) 

v_cvt_f16_f32 v28, v32 // (op 4539) 
v_cvt_f16_f32 v29, v33 // (op 4539) 
v_cvt_f16_f32 v30, v34 // (op 4539) 
v_cvt_f16_f32 v31, v35 // (op 4539) 


v_pack_b32_f16 v20, v28, v29 // (op 4533) 
v_pack_b32_f16 v21, v30, v31 // (op 4533) 

s_lshl_b32 s52, s24, 6 // (op 4533) 
s_add_u32 s52, s52, 128 // (op 4533) 
v_add_u32 v23, v27, s52 // (op 4533) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4533) Store value



v_accvgpr_read v28, a24 // (op 4567) 
v_accvgpr_read v29, a25 // (op 4567) 
v_accvgpr_read v30, a26 // (op 4567) 
v_accvgpr_read v31, a27 // (op 4567) 
v_mul_f32 v32, v19, v28 // (op 4567) 
v_mul_f32 v33, v19, v29 // (op 4567) 
v_mul_f32 v34, v19, v30 // (op 4567) 
v_mul_f32 v35, v19, v31 // (op 4567) 

v_cvt_f16_f32 v28, v32 // (op 4569) 
v_cvt_f16_f32 v29, v33 // (op 4569) 
v_cvt_f16_f32 v30, v34 // (op 4569) 
v_cvt_f16_f32 v31, v35 // (op 4569) 


v_pack_b32_f16 v20, v28, v29 // (op 4571) 
v_pack_b32_f16 v21, v30, v31 // (op 4571) 

v_add_u32 v23, 192, v27 // (op 4571) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4571) Store value



v_accvgpr_read v28, a28 // (op 4577) 
v_accvgpr_read v29, a29 // (op 4577) 
v_accvgpr_read v30, a30 // (op 4577) 
v_accvgpr_read v31, a31 // (op 4577) 
v_mul_f32 v32, v19, v28 // (op 4577) 
v_mul_f32 v33, v19, v29 // (op 4577) 
v_mul_f32 v34, v19, v30 // (op 4577) 
v_mul_f32 v35, v19, v31 // (op 4577) 

v_cvt_f16_f32 v28, v32 // (op 4579) 
v_cvt_f16_f32 v29, v33 // (op 4579) 
v_cvt_f16_f32 v30, v34 // (op 4579) 
v_cvt_f16_f32 v31, v35 // (op 4579) 


v_pack_b32_f16 v20, v28, v29 // (op 4573) 
v_pack_b32_f16 v21, v30, v31 // (op 4573) 

s_lshl_b32 s52, s24, 6 // (op 4573) 
s_add_u32 s52, s52, 192 // (op 4573) 
v_add_u32 v23, v27, s52 // (op 4573) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4573) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_0_beta_135
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_0_beta_135:


v_add_u32 v20, 192, v22 // (op 4769) 

buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 4769) Load value

v_accvgpr_read v28, a24 // (op 4760) 
v_accvgpr_read v29, a25 // (op 4760) 
v_accvgpr_read v30, a26 // (op 4760) 
v_accvgpr_read v31, a27 // (op 4760) 
v_mul_f32 v32, v19, v28 // (op 4760) 
v_mul_f32 v33, v19, v29 // (op 4760) 
v_mul_f32 v34, v19, v30 // (op 4760) 
v_mul_f32 v35, v19, v31 // (op 4760) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 4768) 
v_lshrrev_b32 v21, 16, v24 // (op 4768) 
v_cvt_f32_f16 v21, v21 // (op 4768) 
v_mul_f32 v28, v18, v20 // (op 4768) 
v_mul_f32 v29, v18, v21 // (op 4768) 
v_cvt_f32_f16 v20, v25 // (op 4768) 
v_lshrrev_b32 v21, 16, v25 // (op 4768) 
v_cvt_f32_f16 v21, v21 // (op 4768) 
v_mul_f32 v30, v18, v20 // (op 4768) 
v_mul_f32 v31, v18, v21 // (op 4768) 


v_add_f32 v36, v28, v32 // (op 4761) 
v_add_f32 v37, v29, v33 // (op 4761) 
v_add_f32 v38, v30, v34 // (op 4761) 
v_add_f32 v39, v31, v35 // (op 4761) 


v_cvt_f16_f32 v28, v36 // (op 4762) 
v_cvt_f16_f32 v29, v37 // (op 4762) 
v_cvt_f16_f32 v30, v38 // (op 4762) 
v_cvt_f16_f32 v31, v39 // (op 4762) 


v_pack_b32_f16 v20, v28, v29 // (op 4764) 
v_pack_b32_f16 v21, v30, v31 // (op 4764) 

v_add_u32 v23, 192, v27 // (op 4764) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4764) Store value





s_lshl_b32 s52, s4, 6 // (op 4774) 
s_add_u32 s52, s52, 192 // (op 4774) 
v_add_u32 v20, v22, s52 // (op 4774) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 4774) Load value

v_accvgpr_read v28, a28 // (op 4776) 
v_accvgpr_read v29, a29 // (op 4776) 
v_accvgpr_read v30, a30 // (op 4776) 
v_accvgpr_read v31, a31 // (op 4776) 
v_mul_f32 v32, v19, v28 // (op 4776) 
v_mul_f32 v33, v19, v29 // (op 4776) 
v_mul_f32 v34, v19, v30 // (op 4776) 
v_mul_f32 v35, v19, v31 // (op 4776) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 4771) 
v_lshrrev_b32 v21, 16, v24 // (op 4771) 
v_cvt_f32_f16 v21, v21 // (op 4771) 
v_mul_f32 v28, v18, v20 // (op 4771) 
v_mul_f32 v29, v18, v21 // (op 4771) 
v_cvt_f32_f16 v20, v25 // (op 4771) 
v_lshrrev_b32 v21, 16, v25 // (op 4771) 
v_cvt_f32_f16 v21, v21 // (op 4771) 
v_mul_f32 v30, v18, v20 // (op 4771) 
v_mul_f32 v31, v18, v21 // (op 4771) 


v_add_f32 v36, v28, v32 // (op 4772) 
v_add_f32 v37, v29, v33 // (op 4772) 
v_add_f32 v38, v30, v34 // (op 4772) 
v_add_f32 v39, v31, v35 // (op 4772) 


v_cvt_f16_f32 v28, v36 // (op 4773) 
v_cvt_f16_f32 v29, v37 // (op 4773) 
v_cvt_f16_f32 v30, v38 // (op 4773) 
v_cvt_f16_f32 v31, v39 // (op 4773) 


v_pack_b32_f16 v20, v28, v29 // (op 4766) 
v_pack_b32_f16 v21, v30, v31 // (op 4766) 

s_lshl_b32 s52, s24, 6 // (op 4766) 
s_add_u32 s52, s52, 192 // (op 4766) 
v_add_u32 v23, v27, s52 // (op 4766) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4766) Store value





v_add_u32 v20, 128, v22 // (op 4724) 

buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 4724) Load value

v_accvgpr_read v28, a16 // (op 4715) 
v_accvgpr_read v29, a17 // (op 4715) 
v_accvgpr_read v30, a18 // (op 4715) 
v_accvgpr_read v31, a19 // (op 4715) 
v_mul_f32 v32, v19, v28 // (op 4715) 
v_mul_f32 v33, v19, v29 // (op 4715) 
v_mul_f32 v34, v19, v30 // (op 4715) 
v_mul_f32 v35, v19, v31 // (op 4715) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 4723) 
v_lshrrev_b32 v21, 16, v24 // (op 4723) 
v_cvt_f32_f16 v21, v21 // (op 4723) 
v_mul_f32 v28, v18, v20 // (op 4723) 
v_mul_f32 v29, v18, v21 // (op 4723) 
v_cvt_f32_f16 v20, v25 // (op 4723) 
v_lshrrev_b32 v21, 16, v25 // (op 4723) 
v_cvt_f32_f16 v21, v21 // (op 4723) 
v_mul_f32 v30, v18, v20 // (op 4723) 
v_mul_f32 v31, v18, v21 // (op 4723) 


v_add_f32 v36, v28, v32 // (op 4716) 
v_add_f32 v37, v29, v33 // (op 4716) 
v_add_f32 v38, v30, v34 // (op 4716) 
v_add_f32 v39, v31, v35 // (op 4716) 


v_cvt_f16_f32 v28, v36 // (op 4717) 
v_cvt_f16_f32 v29, v37 // (op 4717) 
v_cvt_f16_f32 v30, v38 // (op 4717) 
v_cvt_f16_f32 v31, v39 // (op 4717) 


v_pack_b32_f16 v20, v28, v29 // (op 4719) 
v_pack_b32_f16 v21, v30, v31 // (op 4719) 

v_add_u32 v23, 128, v27 // (op 4719) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4719) Store value





s_lshl_b32 s52, s4, 6 // (op 4729) 
s_add_u32 s52, s52, 128 // (op 4729) 
v_add_u32 v20, v22, s52 // (op 4729) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 4729) Load value

v_accvgpr_read v28, a20 // (op 4731) 
v_accvgpr_read v29, a21 // (op 4731) 
v_accvgpr_read v30, a22 // (op 4731) 
v_accvgpr_read v31, a23 // (op 4731) 
v_mul_f32 v32, v19, v28 // (op 4731) 
v_mul_f32 v33, v19, v29 // (op 4731) 
v_mul_f32 v34, v19, v30 // (op 4731) 
v_mul_f32 v35, v19, v31 // (op 4731) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 4726) 
v_lshrrev_b32 v21, 16, v24 // (op 4726) 
v_cvt_f32_f16 v21, v21 // (op 4726) 
v_mul_f32 v28, v18, v20 // (op 4726) 
v_mul_f32 v29, v18, v21 // (op 4726) 
v_cvt_f32_f16 v20, v25 // (op 4726) 
v_lshrrev_b32 v21, 16, v25 // (op 4726) 
v_cvt_f32_f16 v21, v21 // (op 4726) 
v_mul_f32 v30, v18, v20 // (op 4726) 
v_mul_f32 v31, v18, v21 // (op 4726) 


v_add_f32 v36, v28, v32 // (op 4727) 
v_add_f32 v37, v29, v33 // (op 4727) 
v_add_f32 v38, v30, v34 // (op 4727) 
v_add_f32 v39, v31, v35 // (op 4727) 


v_cvt_f16_f32 v28, v36 // (op 4728) 
v_cvt_f16_f32 v29, v37 // (op 4728) 
v_cvt_f16_f32 v30, v38 // (op 4728) 
v_cvt_f16_f32 v31, v39 // (op 4728) 


v_pack_b32_f16 v20, v28, v29 // (op 4721) 
v_pack_b32_f16 v21, v30, v31 // (op 4721) 

s_lshl_b32 s52, s24, 6 // (op 4721) 
s_add_u32 s52, s52, 128 // (op 4721) 
v_add_u32 v23, v27, s52 // (op 4721) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4721) Store value





v_add_u32 v20, 64, v22 // (op 4679) 

buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 4679) Load value

v_accvgpr_read v28, a8 // (op 4670) 
v_accvgpr_read v29, a9 // (op 4670) 
v_accvgpr_read v30, a10 // (op 4670) 
v_accvgpr_read v31, a11 // (op 4670) 
v_mul_f32 v32, v19, v28 // (op 4670) 
v_mul_f32 v33, v19, v29 // (op 4670) 
v_mul_f32 v34, v19, v30 // (op 4670) 
v_mul_f32 v35, v19, v31 // (op 4670) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 4678) 
v_lshrrev_b32 v21, 16, v24 // (op 4678) 
v_cvt_f32_f16 v21, v21 // (op 4678) 
v_mul_f32 v28, v18, v20 // (op 4678) 
v_mul_f32 v29, v18, v21 // (op 4678) 
v_cvt_f32_f16 v20, v25 // (op 4678) 
v_lshrrev_b32 v21, 16, v25 // (op 4678) 
v_cvt_f32_f16 v21, v21 // (op 4678) 
v_mul_f32 v30, v18, v20 // (op 4678) 
v_mul_f32 v31, v18, v21 // (op 4678) 


v_add_f32 v36, v28, v32 // (op 4671) 
v_add_f32 v37, v29, v33 // (op 4671) 
v_add_f32 v38, v30, v34 // (op 4671) 
v_add_f32 v39, v31, v35 // (op 4671) 


v_cvt_f16_f32 v28, v36 // (op 4672) 
v_cvt_f16_f32 v29, v37 // (op 4672) 
v_cvt_f16_f32 v30, v38 // (op 4672) 
v_cvt_f16_f32 v31, v39 // (op 4672) 


v_pack_b32_f16 v20, v28, v29 // (op 4674) 
v_pack_b32_f16 v21, v30, v31 // (op 4674) 

v_add_u32 v23, 64, v27 // (op 4674) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4674) Store value





s_lshl_b32 s52, s4, 6 // (op 4684) 
s_add_u32 s52, s52, 64 // (op 4684) 
v_add_u32 v20, v22, s52 // (op 4684) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 4684) Load value

v_accvgpr_read v28, a12 // (op 4686) 
v_accvgpr_read v29, a13 // (op 4686) 
v_accvgpr_read v30, a14 // (op 4686) 
v_accvgpr_read v31, a15 // (op 4686) 
v_mul_f32 v32, v19, v28 // (op 4686) 
v_mul_f32 v33, v19, v29 // (op 4686) 
v_mul_f32 v34, v19, v30 // (op 4686) 
v_mul_f32 v35, v19, v31 // (op 4686) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 4681) 
v_lshrrev_b32 v21, 16, v24 // (op 4681) 
v_cvt_f32_f16 v21, v21 // (op 4681) 
v_mul_f32 v28, v18, v20 // (op 4681) 
v_mul_f32 v29, v18, v21 // (op 4681) 
v_cvt_f32_f16 v20, v25 // (op 4681) 
v_lshrrev_b32 v21, 16, v25 // (op 4681) 
v_cvt_f32_f16 v21, v21 // (op 4681) 
v_mul_f32 v30, v18, v20 // (op 4681) 
v_mul_f32 v31, v18, v21 // (op 4681) 


v_add_f32 v36, v28, v32 // (op 4682) 
v_add_f32 v37, v29, v33 // (op 4682) 
v_add_f32 v38, v30, v34 // (op 4682) 
v_add_f32 v39, v31, v35 // (op 4682) 


v_cvt_f16_f32 v28, v36 // (op 4683) 
v_cvt_f16_f32 v29, v37 // (op 4683) 
v_cvt_f16_f32 v30, v38 // (op 4683) 
v_cvt_f16_f32 v31, v39 // (op 4683) 


v_pack_b32_f16 v20, v28, v29 // (op 4676) 
v_pack_b32_f16 v21, v30, v31 // (op 4676) 

s_lshl_b32 s52, s24, 6 // (op 4676) 
s_add_u32 s52, s52, 64 // (op 4676) 
v_add_u32 v23, v27, s52 // (op 4676) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4676) Store value





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
v_cvt_f32_f16 v20, v24 // (op 149) 
v_lshrrev_b32 v21, 16, v24 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
v_mul_f32 v28, v18, v20 // (op 149) 
v_mul_f32 v29, v18, v21 // (op 149) 
v_cvt_f32_f16 v20, v25 // (op 149) 
v_lshrrev_b32 v21, 16, v25 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
v_mul_f32 v30, v18, v20 // (op 149) 
v_mul_f32 v31, v18, v21 // (op 149) 


v_add_f32 v36, v28, v32 // (op 145) 
v_add_f32 v37, v29, v33 // (op 145) 
v_add_f32 v38, v30, v34 // (op 145) 
v_add_f32 v39, v31, v35 // (op 145) 


v_cvt_f16_f32 v28, v36 // (op 146) 
v_cvt_f16_f32 v29, v37 // (op 146) 
v_cvt_f16_f32 v30, v38 // (op 146) 
v_cvt_f16_f32 v31, v39 // (op 146) 


v_pack_b32_f16 v20, v28, v29 // (op 147) 
v_pack_b32_f16 v21, v30, v31 // (op 147) 

v_mov_b32 v23, v27 // (op 147) reg expression

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 147) Store value





s_lshl_b32 s52, s4, 6 // (op 4639) 
v_add_u32 v20, v22, s52 // (op 4639) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 4639) Load value

v_accvgpr_read v28, a4 // (op 4635) 
v_accvgpr_read v29, a5 // (op 4635) 
v_accvgpr_read v30, a6 // (op 4635) 
v_accvgpr_read v31, a7 // (op 4635) 
v_mul_f32 v32, v19, v28 // (op 4635) 
v_mul_f32 v33, v19, v29 // (op 4635) 
v_mul_f32 v34, v19, v30 // (op 4635) 
v_mul_f32 v35, v19, v31 // (op 4635) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 4640) 
v_lshrrev_b32 v21, 16, v24 // (op 4640) 
v_cvt_f32_f16 v21, v21 // (op 4640) 
v_mul_f32 v28, v18, v20 // (op 4640) 
v_mul_f32 v29, v18, v21 // (op 4640) 
v_cvt_f32_f16 v20, v25 // (op 4640) 
v_lshrrev_b32 v21, 16, v25 // (op 4640) 
v_cvt_f32_f16 v21, v21 // (op 4640) 
v_mul_f32 v30, v18, v20 // (op 4640) 
v_mul_f32 v31, v18, v21 // (op 4640) 


v_add_f32 v36, v28, v32 // (op 4636) 
v_add_f32 v37, v29, v33 // (op 4636) 
v_add_f32 v38, v30, v34 // (op 4636) 
v_add_f32 v39, v31, v35 // (op 4636) 


v_cvt_f16_f32 v28, v36 // (op 4637) 
v_cvt_f16_f32 v29, v37 // (op 4637) 
v_cvt_f16_f32 v30, v38 // (op 4637) 
v_cvt_f16_f32 v31, v39 // (op 4637) 


v_pack_b32_f16 v20, v28, v29 // (op 4638) 
v_pack_b32_f16 v21, v30, v31 // (op 4638) 

s_lshl_b32 s52, s24, 6 // (op 4638) 
v_add_u32 v23, v27, s52 // (op 4638) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 4638) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_0_beta_135:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_381 // (op 381) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_381
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_381:
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_381:


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
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_SKStreamTileLoop_294 // (op 294) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_SKStreamTileLoop_294 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_294:

















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
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_449 // (op 449) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_449 if false)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_DPStreamTileLoop_449:
v_lshrrev_b32 v0, 7, v1 // (op 9580) 
s_mul_i32 s15, s16, s50 // (op 9580) 
s_add_u32 s15, s15, s0 // (op 9580) 
v_and_b32 v3, 63, v1 // (op 9580) 
s_add_u32 s6, s15, s9 // (op 9580) 

v_lshrrev_b32 v4, 4, v3 // (op 9580) 
s_mul_hi_u32 s15, s6, s13 // (op 9580) 
v_lshlrev_b32 v5, 2, v4 // (op 9580) 

s_sub_u32 s7, s6, s15 // (op 9580) 
v_lshl_add_u32 v4, v0, 4, v5 // (op 9580) 


s_lshr_b32 s8, s7, 1 // (op 9580) 

v_lshrrev_b32 v0, 2, v3 // (op 9580) 

s_add_u32 s7, s8, s15 // (op 9580) 


v_and_b32 v3, 3, v0 // (op 9580) 

s_lshr_b32 s8, s7, s11 // (op 9580) 

v_and_b32 v0, 3, v1 // (op 9580) 
s_and_b32 s7, s11, 2147483648 // (op 9580) 
v_lshl_add_u32 v5, v3, 2, v0 // (op 9580) 


s_cmp_eq_i32 s7, 0 // (op 9580) 
s_mov_b32 s15, scc // (op 9580) 

v_lshrrev_b32 v0, 6, v1 // (op 9580) 
s_cmp_eq_i32 s15, 0 // (op 9580) 
s_cselect_b32 s7, s6, s8 // (op 9580) 



v_and_b32 v3, 1, v0 // (op 9580) 

s_mul_hi_u32 s6, s7, s12 // (op 9580) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 9580) 


s_sub_u32 s8, s7, s6 // (op 9580) 
s_lshr_b32 s15, s8, 1 // (op 9580) 

s_add_u32 s8, s15, s6 // (op 9580) 


s_lshr_b32 s6, s8, s10 // (op 9580) 

s_and_b32 s8, s10, 2147483648 // (op 9580) 
s_cmp_eq_i32 s8, 0 // (op 9580) 
s_mov_b32 s15, scc // (op 9580) 

s_cmp_eq_i32 s15, 0 // (op 9580) 
s_cselect_b32 s8, s7, s6 // (op 9580) 


s_mul_i32 s6, s8, s14 // (op 9580) 
v_lshl_add_u32 v3, s8, 7, v4 // (op 9580) 


s_sub_u32 s8, s7, s6 // (op 9580) 


v_lshl_add_u32 v4, s8, 6, v0 // (op 9580) 


v_mul_lo_u32 v0, v4, s4 // (op 9580) 

v_add_lshl_u32 v4, v3, v0, 1 // (op 9580) 


s_mul_i32 s19, s4, s23 // (op 9582) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 9582) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 9582) least significant
s_mul_hi_u32 s6, s4, s22 // (op 9582) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9582) most significant: sum
s_add_u32 s19, s19, s6 // (op 9582) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 9582) 

s_mov_b32 s56, s20 // (op 9582) reg expression
s_mov_b32 s57, s21 // (op 9582) reg expression
s_mov_b32 s58, s28 // (op 9582) reg expression
s_mov_b32 s59, 131072 // (op 9582) call()

v_lshrrev_b32 v0, 7, v1 // (op 9934) 
s_mul_i32 s15, s16, s50 // (op 9934) 
s_add_u32 s15, s15, s0 // (op 9934) 
v_and_b32 v3, 63, v1 // (op 9934) 
s_add_u32 s6, s15, s9 // (op 9934) 

v_lshrrev_b32 v5, 4, v3 // (op 9934) 
s_mul_hi_u32 s15, s6, s13 // (op 9934) 
v_lshlrev_b32 v14, 2, v5 // (op 9934) 

s_sub_u32 s7, s6, s15 // (op 9934) 
v_lshl_add_u32 v5, v0, 4, v14 // (op 9934) 


s_lshr_b32 s8, s7, 1 // (op 9934) 

v_lshrrev_b32 v0, 2, v3 // (op 9934) 

s_add_u32 s7, s8, s15 // (op 9934) 


v_and_b32 v3, 3, v0 // (op 9934) 

s_lshr_b32 s8, s7, s11 // (op 9934) 

v_and_b32 v0, 3, v1 // (op 9934) 
s_and_b32 s7, s11, 2147483648 // (op 9934) 
v_lshl_add_u32 v14, v3, 2, v0 // (op 9934) 


s_cmp_eq_i32 s7, 0 // (op 9934) 
s_mov_b32 s15, scc // (op 9934) 

v_lshrrev_b32 v0, 6, v1 // (op 9934) 
s_cmp_eq_i32 s15, 0 // (op 9934) 
s_cselect_b32 s7, s6, s8 // (op 9934) 



v_and_b32 v3, 1, v0 // (op 9934) 

s_mul_hi_u32 s6, s7, s12 // (op 9934) 
v_lshl_add_u32 v0, v3, 4, v14 // (op 9934) 


s_sub_u32 s8, s7, s6 // (op 9934) 
s_lshr_b32 s15, s8, 1 // (op 9934) 

s_add_u32 s8, s15, s6 // (op 9934) 


s_lshr_b32 s6, s8, s10 // (op 9934) 

s_and_b32 s8, s10, 2147483648 // (op 9934) 
s_cmp_eq_i32 s8, 0 // (op 9934) 
s_mov_b32 s15, scc // (op 9934) 

s_cmp_eq_i32 s15, 0 // (op 9934) 
s_cselect_b32 s8, s7, s6 // (op 9934) 


s_mul_i32 s6, s8, s14 // (op 9934) 
v_lshl_add_u32 v3, s8, 7, v5 // (op 9934) 


s_sub_u32 s8, s7, s6 // (op 9934) 


v_lshl_add_u32 v5, s8, 6, v0 // (op 9934) 


v_mul_lo_u32 v0, v5, s24 // (op 9934) 

v_add_lshl_u32 v5, v3, v0, 1 // (op 9934) 


s_mul_i32 s19, s24, s23 // (op 9936) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 9936) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 9936) least significant
s_mul_hi_u32 s6, s24, s22 // (op 9936) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9936) most significant: sum
s_add_u32 s19, s19, s6 // (op 9936) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 9936) 

s_mov_b32 s60, s26 // (op 9936) reg expression
s_mov_b32 s61, s27 // (op 9936) reg expression
s_mov_b32 s62, s28 // (op 9936) reg expression
s_mov_b32 s63, 131072 // (op 9936) call()

v_lshrrev_b32 v0, 7, v1 // (op 10013) 
v_and_b32 v3, 15, v1 // (op 10013) 
v_lshl_add_u32 v14, v0, 4, v3 // (op 10013) 


v_and_b32 v0, 48, v1 // (op 10013) 
v_lshl_add_u32 v3, v14, 7, v0 // (op 10013) 


v_lshrrev_b32 v0, 6, v1 // (op 10088) 
v_and_b32 v14, 1, v0 // (op 10088) 

v_and_b32 v0, 15, v1 // (op 10088) 
v_lshl_add_u32 v15, v14, 4, v0 // (op 10088) 


v_and_b32 v0, 48, v1 // (op 10088) 
v_lshl_add_u32 v14, v15, 7, v0 // (op 10088) 


v_lshrrev_b32 v0, 7, v1 // (op 10164) 
v_and_b32 v15, 15, v1 // (op 10164) 
v_lshl_add_u32 v16, v0, 4, v15 // (op 10164) 


v_and_b32 v0, 63, v1 // (op 10164) 
v_lshrrev_b32 v15, 4, v0 // (op 10164) 

v_lshl_add_u32 v0, v16, 2, v15 // (op 10164) 


v_lshrrev_b32 v15, 6, v1 // (op 10240) 
v_and_b32 v16, 1, v15 // (op 10240) 

v_and_b32 v15, 15, v1 // (op 10240) 
v_lshl_add_u32 v17, v16, 4, v15 // (op 10240) 


v_and_b32 v15, 63, v1 // (op 10240) 
v_lshrrev_b32 v16, 4, v15 // (op 10240) 

v_lshl_add_u32 v15, v17, 2, v16 // (op 10240) 


v_accvgpr_write a0, 0.00000 // (op 184) call()
v_accvgpr_write a1, 0.00000 // (op 184) call()
v_accvgpr_write a2, 0.00000 // (op 184) call()
v_accvgpr_write a3, 0.00000 // (op 184) call()
v_accvgpr_write a4, 0.00000 // (op 622) call()
v_accvgpr_write a5, 0.00000 // (op 622) call()
v_accvgpr_write a6, 0.00000 // (op 622) call()
v_accvgpr_write a7, 0.00000 // (op 622) call()
v_accvgpr_write a8, 0.00000 // (op 842) call()
v_accvgpr_write a9, 0.00000 // (op 842) call()
v_accvgpr_write a10, 0.00000 // (op 842) call()
v_accvgpr_write a11, 0.00000 // (op 842) call()
v_accvgpr_write a12, 0.00000 // (op 997) call()
v_accvgpr_write a13, 0.00000 // (op 997) call()
v_accvgpr_write a14, 0.00000 // (op 997) call()
v_accvgpr_write a15, 0.00000 // (op 997) call()
v_accvgpr_write a16, 0.00000 // (op 1228) call()
v_accvgpr_write a17, 0.00000 // (op 1228) call()
v_accvgpr_write a18, 0.00000 // (op 1228) call()
v_accvgpr_write a19, 0.00000 // (op 1228) call()
v_accvgpr_write a20, 0.00000 // (op 1383) call()
v_accvgpr_write a21, 0.00000 // (op 1383) call()
v_accvgpr_write a22, 0.00000 // (op 1383) call()
v_accvgpr_write a23, 0.00000 // (op 1383) call()
v_accvgpr_write a24, 0.00000 // (op 1614) call()
v_accvgpr_write a25, 0.00000 // (op 1614) call()
v_accvgpr_write a26, 0.00000 // (op 1614) call()
v_accvgpr_write a27, 0.00000 // (op 1614) call()
v_accvgpr_write a28, 0.00000 // (op 1769) call()
v_accvgpr_write a29, 0.00000 // (op 1769) call()
v_accvgpr_write a30, 0.00000 // (op 1769) call()
v_accvgpr_write a31, 0.00000 // (op 1769) call()
v_lshrrev_b32 v16, 3, v1 // (op 9068) 
s_mul_i32 s15, s16, s50 // (op 9068) 
s_add_u32 s15, s15, s0 // (op 9068) 
v_and_b32 v17, -8, v1 // (op 9068) 
s_add_u32 s6, s15, s9 // (op 9068) 

v_sub_u32 v20, v1, v17 // (op 9068) 

s_mul_hi_u32 s15, s6, s13 // (op 9068) 
v_lshlrev_b32 v17, 4, v20 // (op 9068) 

s_sub_u32 s7, s6, s15 // (op 9068) 
s_lshr_b32 s8, s7, 1 // (op 9068) 

s_add_u32 s7, s8, s15 // (op 9068) 


s_lshr_b32 s8, s7, s11 // (op 9068) 

s_and_b32 s7, s11, 2147483648 // (op 9068) 
s_cmp_eq_i32 s7, 0 // (op 9068) 
s_mov_b32 s15, scc // (op 9068) 

s_cmp_eq_i32 s15, 0 // (op 9068) 
s_cselect_b32 s7, s6, s8 // (op 9068) 


s_mul_hi_u32 s8, s7, s12 // (op 9068) 
s_sub_u32 s15, s7, s8 // (op 9068) 
s_lshr_b32 s18, s15, 1 // (op 9068) 

s_add_u32 s15, s18, s8 // (op 9068) 


s_lshr_b32 s8, s15, s10 // (op 9068) 

s_and_b32 s15, s10, 2147483648 // (op 9068) 
s_cmp_eq_i32 s15, 0 // (op 9068) 
s_mov_b32 s18, scc // (op 9068) 

s_cmp_eq_i32 s18, 0 // (op 9068) 
s_cselect_b32 s15, s7, s8 // (op 9068) 


v_lshl_add_u32 v20, s15, 7, v16 // (op 9068) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 9068) 
s_mul_i32 s8, s7, s18 // (op 9068) 


s_sub_u32 s7, s6, s8 // (op 9068) 


v_lshl_add_u32 v16, s7, 7, v17 // (op 9068) 


v_mov_b32 v23, 0 // (op 9068) convert
v_mov_b32 v22, v16 // (op 9068) convert

v_mul_lo_u32 v17, v20, s31 // (op 9068) most significant: low of low * high
v_mul_hi_u32 v21, v20, s30 // (op 9068) most significant: high of low * low
v_mul_lo_u32 v16, v20, s30 // (op 9068) least significant: low of low * low
v_add_u32 v17, v17, v21 // (op 9068) most significant: sum

v_add_co_u32 v16, s[18:19], v16, v22 // (op 9068) least significant half
v_addc_co_u32 v17, s[18:19], v17, v23, s[18:19] // (op 9068) most significant half



s_mul_i32 s19, s30, s35 // (op 9070) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 9070) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 9070) least significant
s_mul_hi_u32 s6, s30, s34 // (op 9070) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9070) most significant: sum
s_add_u32 s19, s19, s6 // (op 9070) most significant: sum


s_mov_b32 s68, s32 // (op 9070) reg expression
s_mov_b32 s69, s33 // (op 9070) reg expression
s_mov_b32 s70, s18 // (op 9070) reg expression
s_mov_b32 s71, 131072 // (op 9070) call()

v_lshrrev_b32 v20, 3, v1 // (op 9087) 
s_mul_i32 s15, s16, s50 // (op 9087) 
s_add_u32 s15, s15, s0 // (op 9087) 
v_and_b32 v21, -8, v1 // (op 9087) 
s_add_u32 s6, s15, s9 // (op 9087) 

v_sub_u32 v22, v1, v21 // (op 9087) 

s_mul_hi_u32 s15, s6, s13 // (op 9087) 
v_lshlrev_b32 v21, 4, v22 // (op 9087) 

s_sub_u32 s7, s6, s15 // (op 9087) 
s_lshr_b32 s8, s7, 1 // (op 9087) 

s_add_u32 s7, s8, s15 // (op 9087) 


s_lshr_b32 s8, s7, s11 // (op 9087) 

s_and_b32 s7, s11, 2147483648 // (op 9087) 
s_cmp_eq_i32 s7, 0 // (op 9087) 
s_mov_b32 s15, scc // (op 9087) 

s_cmp_eq_i32 s15, 0 // (op 9087) 
s_cselect_b32 s7, s6, s8 // (op 9087) 


s_mul_hi_u32 s8, s7, s12 // (op 9087) 
s_sub_u32 s15, s7, s8 // (op 9087) 
s_lshr_b32 s18, s15, 1 // (op 9087) 

s_add_u32 s15, s18, s8 // (op 9087) 


s_lshr_b32 s8, s15, s10 // (op 9087) 

s_and_b32 s15, s10, 2147483648 // (op 9087) 
s_cmp_eq_i32 s15, 0 // (op 9087) 
s_mov_b32 s18, scc // (op 9087) 

s_cmp_eq_i32 s18, 0 // (op 9087) 
s_cselect_b32 s15, s7, s8 // (op 9087) 


v_lshl_add_u32 v22, s15, 7, v20 // (op 9087) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 9087) 
s_mul_i32 s8, s7, s18 // (op 9087) 


s_sub_u32 s7, s6, s8 // (op 9087) 


v_lshl_add_u32 v20, s7, 7, v21 // (op 9087) 


v_mov_b32 v25, 0 // (op 9087) convert
v_mov_b32 v24, v20 // (op 9087) convert

v_mul_lo_u32 v21, v22, s31 // (op 9087) most significant: low of low * high
v_mul_hi_u32 v23, v22, s30 // (op 9087) most significant: high of low * low
v_mul_lo_u32 v20, v22, s30 // (op 9087) least significant: low of low * low
v_add_u32 v21, v21, v23 // (op 9087) most significant: sum

v_add_co_u32 v20, s[18:19], v20, v24 // (op 9087) least significant half
v_addc_co_u32 v21, s[18:19], v21, v25, s[18:19] // (op 9087) most significant half



s_mul_i32 s19, s30, s35 // (op 9089) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 9089) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 9089) least significant
s_mul_hi_u32 s6, s30, s34 // (op 9089) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9089) most significant: sum
s_add_u32 s19, s19, s6 // (op 9089) most significant: sum


s_mov_b32 s68, s32 // (op 9089) reg expression
s_mov_b32 s69, s33 // (op 9089) reg expression
s_mov_b32 s70, s18 // (op 9089) reg expression
s_mov_b32 s71, 131072 // (op 9089) call()

v_lshrrev_b32 v22, 3, v1 // (op 9206) 
s_mul_i32 s15, s16, s50 // (op 9206) 
s_add_u32 s15, s15, s0 // (op 9206) 
v_and_b32 v23, -8, v1 // (op 9206) 
s_add_u32 s6, s15, s9 // (op 9206) 

v_sub_u32 v24, v1, v23 // (op 9206) 

s_mul_hi_u32 s15, s6, s13 // (op 9206) 
v_lshlrev_b32 v23, 4, v24 // (op 9206) 

s_sub_u32 s7, s6, s15 // (op 9206) 
s_lshr_b32 s8, s7, 1 // (op 9206) 

s_add_u32 s7, s8, s15 // (op 9206) 


s_lshr_b32 s8, s7, s11 // (op 9206) 

s_and_b32 s7, s11, 2147483648 // (op 9206) 
s_cmp_eq_i32 s7, 0 // (op 9206) 
s_mov_b32 s15, scc // (op 9206) 

s_cmp_eq_i32 s15, 0 // (op 9206) 
s_cselect_b32 s7, s6, s8 // (op 9206) 


s_mul_hi_u32 s8, s7, s12 // (op 9206) 
s_sub_u32 s15, s7, s8 // (op 9206) 
s_lshr_b32 s18, s15, 1 // (op 9206) 

s_add_u32 s15, s18, s8 // (op 9206) 


s_lshr_b32 s8, s15, s10 // (op 9206) 

s_and_b32 s15, s10, 2147483648 // (op 9206) 
s_cmp_eq_i32 s15, 0 // (op 9206) 
s_mov_b32 s18, scc // (op 9206) 

s_cmp_eq_i32 s18, 0 // (op 9206) 
s_cselect_b32 s15, s7, s8 // (op 9206) 


s_mul_i32 s8, s15, s14 // (op 9206) 

s_sub_u32 s15, s7, s8 // (op 9206) 

v_lshl_add_u32 v24, s15, 6, v22 // (op 9206) 


v_mul_lo_u32 v27, v24, s37 // (op 9206) most significant: low of low * high
v_mul_hi_u32 v22, v24, s36 // (op 9206) most significant: high of low * low
v_mul_lo_u32 v26, v24, s36 // (op 9206) least significant: low of low * low
v_add_u32 v27, v27, v22 // (op 9206) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 9206) 
s_mul_i32 s8, s7, s18 // (op 9206) 


s_sub_u32 s7, s6, s8 // (op 9206) 


v_lshl_add_u32 v22, s7, 7, v23 // (op 9206) 


v_mov_b32 v25, 0 // (op 9206) convert
v_mov_b32 v24, v22 // (op 9206) convert

v_add_co_u32 v22, s[18:19], v24, v26 // (op 9206) least significant half
v_addc_co_u32 v23, s[18:19], v25, v27, s[18:19] // (op 9206) most significant half



s_mul_i32 s19, s36, s23 // (op 9208) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 9208) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 9208) least significant
s_mul_hi_u32 s6, s36, s22 // (op 9208) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9208) most significant: sum
s_add_u32 s19, s19, s6 // (op 9208) most significant: sum


s_mov_b32 s72, s38 // (op 9208) reg expression
s_mov_b32 s73, s39 // (op 9208) reg expression
s_mov_b32 s74, s18 // (op 9208) reg expression
s_mov_b32 s75, 131072 // (op 9208) call()

v_lshrrev_b32 v24, 3, v1 // (op 9225) 
s_mul_i32 s15, s16, s50 // (op 9225) 
s_add_u32 s15, s15, s0 // (op 9225) 
v_and_b32 v25, -8, v1 // (op 9225) 
s_add_u32 s6, s15, s9 // (op 9225) 

v_sub_u32 v26, v1, v25 // (op 9225) 

s_mul_hi_u32 s15, s6, s13 // (op 9225) 
v_lshlrev_b32 v25, 4, v26 // (op 9225) 

s_sub_u32 s7, s6, s15 // (op 9225) 
s_lshr_b32 s8, s7, 1 // (op 9225) 

s_add_u32 s7, s8, s15 // (op 9225) 


s_lshr_b32 s8, s7, s11 // (op 9225) 

s_and_b32 s7, s11, 2147483648 // (op 9225) 
s_cmp_eq_i32 s7, 0 // (op 9225) 
s_mov_b32 s15, scc // (op 9225) 

s_cmp_eq_i32 s15, 0 // (op 9225) 
s_cselect_b32 s7, s6, s8 // (op 9225) 


s_mul_hi_u32 s8, s7, s12 // (op 9225) 
s_sub_u32 s15, s7, s8 // (op 9225) 
s_lshr_b32 s18, s15, 1 // (op 9225) 

s_add_u32 s15, s18, s8 // (op 9225) 


s_lshr_b32 s8, s15, s10 // (op 9225) 

s_and_b32 s15, s10, 2147483648 // (op 9225) 
s_cmp_eq_i32 s15, 0 // (op 9225) 
s_mov_b32 s18, scc // (op 9225) 

s_cmp_eq_i32 s18, 0 // (op 9225) 
s_cselect_b32 s15, s7, s8 // (op 9225) 


s_mul_i32 s8, s15, s14 // (op 9225) 

s_sub_u32 s15, s7, s8 // (op 9225) 

v_lshl_add_u32 v26, s15, 6, v24 // (op 9225) 


v_mul_lo_u32 v29, v26, s37 // (op 9225) most significant: low of low * high
v_mul_hi_u32 v24, v26, s36 // (op 9225) most significant: high of low * low
v_mul_lo_u32 v28, v26, s36 // (op 9225) least significant: low of low * low
v_add_u32 v29, v29, v24 // (op 9225) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 9225) 
s_mul_i32 s8, s7, s18 // (op 9225) 


s_sub_u32 s7, s6, s8 // (op 9225) 


v_lshl_add_u32 v24, s7, 7, v25 // (op 9225) 


v_mov_b32 v27, 0 // (op 9225) convert
v_mov_b32 v26, v24 // (op 9225) convert

v_add_co_u32 v24, s[18:19], v26, v28 // (op 9225) least significant half
v_addc_co_u32 v25, s[18:19], v27, v29, s[18:19] // (op 9225) most significant half



s_mul_i32 s19, s36, s23 // (op 9227) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 9227) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 9227) least significant
s_mul_hi_u32 s6, s36, s22 // (op 9227) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9227) most significant: sum
s_add_u32 s19, s19, s6 // (op 9227) most significant: sum


s_mov_b32 s72, s38 // (op 9227) reg expression
s_mov_b32 s73, s39 // (op 9227) reg expression
s_mov_b32 s74, s18 // (op 9227) reg expression
s_mov_b32 s75, 131072 // (op 9227) call()

v_lshrrev_b32 v26, 2, v1 // (op 9334) 
s_mul_i32 s15, s16, s50 // (op 9334) 
s_add_u32 s15, s15, s0 // (op 9334) 
v_lshlrev_b32 v27, 1, v26 // (op 9334) 

s_add_u32 s6, s15, s9 // (op 9334) 

v_and_b32 v26, -4, v1 // (op 9334) 
s_mul_hi_u32 s15, s6, s13 // (op 9334) 
v_sub_u32 v28, v1, v26 // (op 9334) 

s_sub_u32 s7, s6, s15 // (op 9334) 
s_lshr_b32 s8, s7, 1 // (op 9334) 

s_add_u32 s7, s8, s15 // (op 9334) 


s_lshr_b32 s8, s7, s11 // (op 9334) 

s_and_b32 s7, s11, 2147483648 // (op 9334) 
s_cmp_eq_i32 s7, 0 // (op 9334) 
s_mov_b32 s15, scc // (op 9334) 

s_cmp_eq_i32 s15, 0 // (op 9334) 
s_cselect_b32 s7, s6, s8 // (op 9334) 


s_mul_hi_u32 s8, s7, s12 // (op 9334) 
s_sub_u32 s15, s7, s8 // (op 9334) 
s_lshr_b32 s18, s15, 1 // (op 9334) 

s_add_u32 s15, s18, s8 // (op 9334) 


s_lshr_b32 s8, s15, s10 // (op 9334) 

s_and_b32 s15, s10, 2147483648 // (op 9334) 
s_cmp_eq_i32 s15, 0 // (op 9334) 
s_mov_b32 s18, scc // (op 9334) 

s_cmp_eq_i32 s18, 0 // (op 9334) 
s_cselect_b32 s15, s7, s8 // (op 9334) 


v_lshl_add_u32 v26, s15, 7, v27 // (op 9334) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 9334) 
s_mul_i32 s8, s7, s18 // (op 9334) 


s_sub_u32 s7, s6, s8 // (op 9334) 


v_lshl_add_u32 v27, s7, 2, v28 // (op 9334) 


v_mul_lo_u32 v28, v26, s40 // (op 9334) 
v_add_u32 v28, v28, v27 // (op 9334) 


s_mul_i32 s19, s40, s35 // (op 9336) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 9336) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 9336) least significant
s_mul_hi_u32 s6, s40, s34 // (op 9336) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9336) most significant: sum
s_add_u32 s19, s19, s6 // (op 9336) most significant: sum


s_mov_b32 s76, s42 // (op 9336) reg expression
s_mov_b32 s77, s43 // (op 9336) reg expression
s_mov_b32 s78, s18 // (op 9336) reg expression
s_mov_b32 s79, 131072 // (op 9336) call()

v_lshrrev_b32 v26, 2, v1 // (op 9353) 
s_mul_i32 s15, s16, s50 // (op 9353) 
s_add_u32 s15, s15, s0 // (op 9353) 
v_lshlrev_b32 v27, 1, v26 // (op 9353) 

s_add_u32 s6, s15, s9 // (op 9353) 

v_and_b32 v26, -4, v1 // (op 9353) 
s_mul_hi_u32 s15, s6, s13 // (op 9353) 
v_sub_u32 v29, v1, v26 // (op 9353) 

s_sub_u32 s7, s6, s15 // (op 9353) 
s_lshr_b32 s8, s7, 1 // (op 9353) 

s_add_u32 s7, s8, s15 // (op 9353) 


s_lshr_b32 s8, s7, s11 // (op 9353) 

s_and_b32 s7, s11, 2147483648 // (op 9353) 
s_cmp_eq_i32 s7, 0 // (op 9353) 
s_mov_b32 s15, scc // (op 9353) 

s_cmp_eq_i32 s15, 0 // (op 9353) 
s_cselect_b32 s7, s6, s8 // (op 9353) 


s_mul_hi_u32 s8, s7, s12 // (op 9353) 
s_sub_u32 s15, s7, s8 // (op 9353) 
s_lshr_b32 s18, s15, 1 // (op 9353) 

s_add_u32 s15, s18, s8 // (op 9353) 


s_lshr_b32 s8, s15, s10 // (op 9353) 

s_and_b32 s15, s10, 2147483648 // (op 9353) 
s_cmp_eq_i32 s15, 0 // (op 9353) 
s_mov_b32 s18, scc // (op 9353) 

s_cmp_eq_i32 s18, 0 // (op 9353) 
s_cselect_b32 s15, s7, s8 // (op 9353) 


v_lshl_add_u32 v26, s15, 7, v27 // (op 9353) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 9353) 
s_mul_i32 s8, s7, s18 // (op 9353) 


s_sub_u32 s7, s6, s8 // (op 9353) 


v_lshl_add_u32 v27, s7, 2, v29 // (op 9353) 


v_mul_lo_u32 v29, v26, s40 // (op 9353) 
v_add_u32 v29, v29, v27 // (op 9353) 


s_mul_i32 s19, s40, s35 // (op 9355) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 9355) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 9355) least significant
s_mul_hi_u32 s6, s40, s34 // (op 9355) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9355) most significant: sum
s_add_u32 s19, s19, s6 // (op 9355) most significant: sum


s_mov_b32 s76, s42 // (op 9355) reg expression
s_mov_b32 s77, s43 // (op 9355) reg expression
s_mov_b32 s78, s18 // (op 9355) reg expression
s_mov_b32 s79, 131072 // (op 9355) call()

v_lshrrev_b32 v26, 2, v1 // (op 9462) 
s_mul_i32 s15, s16, s50 // (op 9462) 
s_add_u32 s15, s15, s0 // (op 9462) 
v_and_b32 v27, -4, v1 // (op 9462) 
s_add_u32 s6, s15, s9 // (op 9462) 

v_sub_u32 v30, v1, v27 // (op 9462) 

s_mul_hi_u32 s15, s6, s13 // (op 9462) 
s_sub_u32 s7, s6, s15 // (op 9462) 
s_lshr_b32 s8, s7, 1 // (op 9462) 

s_add_u32 s7, s8, s15 // (op 9462) 


s_lshr_b32 s8, s7, s11 // (op 9462) 

s_and_b32 s7, s11, 2147483648 // (op 9462) 
s_cmp_eq_i32 s7, 0 // (op 9462) 
s_mov_b32 s15, scc // (op 9462) 

s_cmp_eq_i32 s15, 0 // (op 9462) 
s_cselect_b32 s7, s6, s8 // (op 9462) 


s_mul_hi_u32 s8, s7, s12 // (op 9462) 
s_sub_u32 s15, s7, s8 // (op 9462) 
s_lshr_b32 s18, s15, 1 // (op 9462) 

s_add_u32 s15, s18, s8 // (op 9462) 


s_lshr_b32 s8, s15, s10 // (op 9462) 

s_and_b32 s15, s10, 2147483648 // (op 9462) 
s_cmp_eq_i32 s15, 0 // (op 9462) 
s_mov_b32 s18, scc // (op 9462) 

s_cmp_eq_i32 s18, 0 // (op 9462) 
s_cselect_b32 s15, s7, s8 // (op 9462) 


s_mul_i32 s8, s15, s14 // (op 9462) 

s_sub_u32 s15, s7, s8 // (op 9462) 

v_lshl_add_u32 v27, s15, 6, v26 // (op 9462) 


v_mul_lo_u32 v26, v27, s44 // (op 9462) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 9462) 
s_mul_i32 s8, s7, s18 // (op 9462) 


s_sub_u32 s7, s6, s8 // (op 9462) 


v_lshl_add_u32 v27, s7, 2, v30 // (op 9462) 


v_add_u32 v30, v27, v26 // (op 9462) 


s_mul_i32 s19, s44, s23 // (op 9464) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 9464) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 9464) least significant
s_mul_hi_u32 s6, s44, s22 // (op 9464) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9464) most significant: sum
s_add_u32 s19, s19, s6 // (op 9464) most significant: sum


s_mov_b32 s80, s46 // (op 9464) reg expression
s_mov_b32 s81, s47 // (op 9464) reg expression
s_mov_b32 s82, s18 // (op 9464) reg expression
s_mov_b32 s83, 131072 // (op 9464) call()

v_lshrrev_b32 v26, 2, v1 // (op 9481) 
s_mul_i32 s15, s16, s50 // (op 9481) 
s_add_u32 s15, s15, s0 // (op 9481) 
v_and_b32 v27, -4, v1 // (op 9481) 
s_add_u32 s6, s15, s9 // (op 9481) 

v_sub_u32 v31, v1, v27 // (op 9481) 

s_mul_hi_u32 s15, s6, s13 // (op 9481) 
s_sub_u32 s7, s6, s15 // (op 9481) 
s_lshr_b32 s8, s7, 1 // (op 9481) 

s_add_u32 s7, s8, s15 // (op 9481) 


s_lshr_b32 s8, s7, s11 // (op 9481) 

s_and_b32 s7, s11, 2147483648 // (op 9481) 
s_cmp_eq_i32 s7, 0 // (op 9481) 
s_mov_b32 s15, scc // (op 9481) 

s_cmp_eq_i32 s15, 0 // (op 9481) 
s_cselect_b32 s7, s6, s8 // (op 9481) 


s_mul_hi_u32 s8, s7, s12 // (op 9481) 
s_sub_u32 s15, s7, s8 // (op 9481) 
s_lshr_b32 s18, s15, 1 // (op 9481) 

s_add_u32 s15, s18, s8 // (op 9481) 


s_lshr_b32 s8, s15, s10 // (op 9481) 

s_and_b32 s15, s10, 2147483648 // (op 9481) 
s_cmp_eq_i32 s15, 0 // (op 9481) 
s_mov_b32 s18, scc // (op 9481) 

s_cmp_eq_i32 s18, 0 // (op 9481) 
s_cselect_b32 s15, s7, s8 // (op 9481) 


s_mul_i32 s8, s15, s14 // (op 9481) 

s_sub_u32 s15, s7, s8 // (op 9481) 

v_lshl_add_u32 v27, s15, 6, v26 // (op 9481) 


v_mul_lo_u32 v26, v27, s44 // (op 9481) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 9481) 
s_mul_i32 s8, s7, s18 // (op 9481) 


s_sub_u32 s7, s6, s8 // (op 9481) 


v_lshl_add_u32 v27, s7, 2, v31 // (op 9481) 


v_add_u32 v31, v27, v26 // (op 9481) 


s_mul_i32 s19, s44, s23 // (op 9483) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 9483) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 9483) least significant
s_mul_hi_u32 s6, s44, s22 // (op 9483) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9483) most significant: sum
s_add_u32 s19, s19, s6 // (op 9483) most significant: sum


s_mov_b32 s80, s46 // (op 9483) reg expression
s_mov_b32 s81, s47 // (op 9483) reg expression
s_mov_b32 s82, s18 // (op 9483) reg expression
s_mov_b32 s83, 131072 // (op 9483) call()

v_and_b32 v26, -8, v1 // (op 10054) 
v_sub_u32 v27, v1, v26 // (op 10054) 

v_lshlrev_b32 v26, 4, v27 // (op 10054) 

v_lshrrev_b32 v27, 3, v1 // (op 10054) 
v_lshl_add_u32 v32, v27, 7, v26 // (op 10054) 


s_nop 0
v_readfirstlane_b32 s15, v32 // (op 10054) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10855) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 8972) 
v_mov_b64 v[26:27], s[18:19] // (op 8972) convert


v_mov_b64 v[32:33], v[22:23] // (op 8972) 
s_mov_b32 m0, s15 // (op 8972) reg expression
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 8972) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v26 // (op 8972) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 8972) most significant half

s_add_u32 m0, m0, 4096 // (op 8972) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 8972) Load value direct to lds






buffer_load_ubyte v26, v30, s[80:83], 0 offen offset: 0 // (op 8520) Load value
v_and_b32 v27, -8, v1 // (op 9979) 
v_sub_u32 v32, v1, v27 // (op 9979) 

v_lshlrev_b32 v27, 4, v32 // (op 9979) 

v_lshrrev_b32 v32, 3, v1 // (op 9979) 
v_lshl_add_u32 v33, v32, 7, v27 // (op 9979) 


s_nop 0
v_readfirstlane_b32 s15, v33 // (op 9979) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10836) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 8977) 
v_mov_b64 v[32:33], s[18:19] // (op 8977) convert


v_mov_b64 v[34:35], v[16:17] // (op 8977) 
s_add_u32 s6, 8192, s15 // (op 8977) 
s_mov_b32 m0, s6 // (op 8977) reg expression
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 8977) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 8977) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 8977) most significant half

s_add_u32 m0, m0, 4096 // (op 8977) 
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 8977) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 8977) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 8977) most significant half

s_add_u32 m0, m0, 4096 // (op 8977) 
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 8977) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 8977) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 8977) most significant half

s_add_u32 m0, m0, 4096 // (op 8977) 
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 8977) Load value direct to lds





v_mov_b64 v[32:33], s[40:41] // (op 8526) convert

v_mov_b32 v27, v28 // (op 8526) 
buffer_load_ubyte v34, v27, s[76:79], 0 offen offset: 0 // (op 8526) Load value
v_add_u32 v27, v27, v32 // (op 8526) 
buffer_load_ubyte v35, v27, s[76:79], 0 offen offset: 0 // (op 8526) Load value


v_and_b32 v27, -8, v1 // (op 10364) 
v_sub_u32 v32, v1, v27 // (op 10364) 

v_lshlrev_b32 v27, 4, v32 // (op 10364) 

v_lshrrev_b32 v32, 3, v1 // (op 10364) 
v_lshl_add_u32 v33, v32, 7, v27 // (op 10364) 


s_nop 0
v_readfirstlane_b32 s15, v33 // (op 10364) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10926) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 8982) 
v_mov_b64 v[32:33], s[18:19] // (op 8982) convert


v_mov_b64 v[36:37], v[22:23] // (op 8982) 
s_add_u32 s6, 24576, s15 // (op 8982) 
s_mov_b32 m0, s6 // (op 8982) reg expression
s_mov_b32 s7, 128 // (op 8982) call()
buffer_load_dwordx4 v36, s[72:75], s7 offen offset: 0 lds // (op 8982) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 8982) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 8982) most significant half

s_add_u32 m0, m0, 4096 // (op 8982) 
buffer_load_dwordx4 v36, s[72:75], s7 offen offset: 0 lds // (op 8982) Load value direct to lds









buffer_load_ubyte v27, v30, s[80:83], 0 offen offset:4 // (op 8532) Load value

v_and_b32 v30, -8, v1 // (op 10320) 
v_sub_u32 v22, v1, v30 // (op 10320) 

v_lshlrev_b32 v23, 4, v22 // (op 10320) 

v_lshrrev_b32 v22, 3, v1 // (op 10320) 
v_lshl_add_u32 v30, v22, 7, v23 // (op 10320) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 10320) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10916) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 8984) 
v_mov_b64 v[22:23], s[18:19] // (op 8984) convert


v_mov_b64 v[32:33], v[16:17] // (op 8984) 
s_add_u32 s6, 32768, s15 // (op 8984) 
s_mov_b32 m0, s6 // (op 8984) reg expression
s_mov_b32 s7, 128 // (op 8984) call()
buffer_load_dwordx4 v32, s[68:71], s7 offen offset: 0 lds // (op 8984) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v22 // (op 8984) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 8984) most significant half

s_add_u32 m0, m0, 4096 // (op 8984) 
buffer_load_dwordx4 v32, s[68:71], s7 offen offset: 0 lds // (op 8984) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v22 // (op 8984) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 8984) most significant half

s_add_u32 m0, m0, 4096 // (op 8984) 
buffer_load_dwordx4 v32, s[68:71], s7 offen offset: 0 lds // (op 8984) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v22 // (op 8984) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 8984) most significant half

s_add_u32 m0, m0, 4096 // (op 8984) 
buffer_load_dwordx4 v32, s[68:71], s7 offen offset: 0 lds // (op 8984) Load value direct to lds







v_mov_b64 v[16:17], s[40:41] // (op 8538) convert

v_mov_b32 v30, v28 // (op 8538) 
buffer_load_ubyte v22, v30, s[76:79], 0 offen offset:4 // (op 8538) Load value
v_add_u32 v30, v30, v16 // (op 8538) 
buffer_load_ubyte v23, v30, s[76:79], 0 offen offset:4 // (op 8538) Load value



v_and_b32 v28, -4, v1 // (op 10206) 
v_sub_u32 v30, v1, v28 // (op 10206) 

v_and_b32 v28, 4294967292, v1 // (op 10206) 
v_add_u32 v16, v28, v30 // (op 10206) 


s_barrier  // (op 10896) 



s_waitcnt vmcnt(15)
ds_write_b8 v16, v26 offset:49152 // (op 8541) Store local data 

v_and_b32 v28, -4, v1 // (op 10130) 
v_sub_u32 v30, v1, v28 // (op 10130) 

v_lshrrev_b32 v28, 2, v1 // (op 10130) 
v_lshl_add_u32 v16, v28, 3, v30 // (op 10130) 


s_barrier  // (op 10875) 



s_waitcnt vmcnt(10)
ds_write_b8 v16, v34 offset:49408 // (op 8543) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v16, v35 offset:49412 // (op 8543) Store local data 

s_barrier  // (op 10852) 



ds_read_b128 v[36:39], v14 // (op 8578) Load local data 
ds_read_b128 v[40:43], v14 offset:64 // (op 8578) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 10893) 



ds_read_u8 v28, v15 offset:49152 // (op 8608) Load local data 



ds_read_b128 v[44:47], v14 offset:4096 // (op 8583) Load local data 
ds_read_b128 v[48:51], v14 offset:4160 // (op 8583) Load local data 



ds_read_u8 v30, v15 offset:49280 // (op 8613) Load local data 
s_barrier  // (op 10833) 



ds_read_b128 v[52:55], v3 offset:8192 // (op 8558) Load local data 
ds_read_b128 v[56:59], v3 offset:8256 // (op 8558) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 10872) 



ds_read_u8 v16, v0 offset:49408 // (op 8588) Load local data 



ds_read_b128 v[60:63], v3 offset:12288 // (op 8563) Load local data 
ds_read_b128 v[64:67], v3 offset:12352 // (op 8563) Load local data 



ds_read_u8 v17, v0 offset:49536 // (op 8593) Load local data 



ds_read_b128 v[68:71], v3 offset:16384 // (op 8568) Load local data 
ds_read_b128 v[72:75], v3 offset:16448 // (op 8568) Load local data 



ds_read_u8 v32, v0 offset:49664 // (op 8598) Load local data 



ds_read_b128 v[76:79], v3 offset:20480 // (op 8573) Load local data 
ds_read_b128 v[80:83], v3 offset:20544 // (op 8573) Load local data 



ds_read_u8 v33, v0 offset:49792 // (op 8603) Load local data 
v_and_b32 v84, -8, v1 // (op 9951) 
v_sub_u32 v85, v1, v84 // (op 9951) 

v_lshlrev_b32 v84, 4, v85 // (op 9951) 

v_lshrrev_b32 v85, 3, v1 // (op 9951) 
v_lshl_add_u32 v86, v85, 7, v84 // (op 9951) 


s_nop 0
v_readfirstlane_b32 s15, v86 // (op 9951) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -8, v1 // (op 10026) 
v_sub_u32 v85, v1, v84 // (op 10026) 

v_lshlrev_b32 v84, 4, v85 // (op 10026) 

v_lshrrev_b32 v85, 3, v1 // (op 10026) 
v_lshl_add_u32 v86, v85, 7, v84 // (op 10026) 


s_nop 0
v_readfirstlane_b32 s6, v86 // (op 10026) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -4, v1 // (op 10101) 
v_sub_u32 v85, v1, v84 // (op 10101) 

v_lshrrev_b32 v84, 2, v1 // (op 10101) 
v_lshl_add_u32 v86, v84, 3, v85 // (op 10101) 


v_and_b32 v84, -4, v1 // (op 10177) 
v_sub_u32 v85, v1, v84 // (op 10177) 

v_and_b32 v84, 4294967292, v1 // (op 10177) 
v_add_u32 v87, v84, v85 // (op 10177) 


v_and_b32 v84, -8, v1 // (op 10305) 
v_sub_u32 v85, v1, v84 // (op 10305) 

v_lshlrev_b32 v84, 4, v85 // (op 10305) 

v_lshrrev_b32 v85, 3, v1 // (op 10305) 
v_lshl_add_u32 v88, v85, 7, v84 // (op 10305) 


s_nop 0
v_readfirstlane_b32 s7, v88 // (op 10305) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -8, v1 // (op 10349) 
v_sub_u32 v85, v1, v84 // (op 10349) 

v_lshlrev_b32 v84, 4, v85 // (op 10349) 

v_lshrrev_b32 v85, 3, v1 // (op 10349) 
v_lshl_add_u32 v88, v85, 7, v84 // (op 10349) 


s_nop 0
v_readfirstlane_b32 s8, v88 // (op 10349) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -4, v1 // (op 10393) 
v_sub_u32 v85, v1, v84 // (op 10393) 

v_lshrrev_b32 v84, 2, v1 // (op 10393) 
v_lshl_add_u32 v88, v84, 3, v85 // (op 10393) 


v_and_b32 v84, -4, v1 // (op 10422) 
v_sub_u32 v85, v1, v84 // (op 10422) 

v_and_b32 v84, 4294967292, v1 // (op 10422) 
v_add_u32 v89, v84, v85 // (op 10422) 


s_mov_b32 s1, 0 // (op 1647) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1646) 
s_mov_b32 s28, s18 // (op 1646) convert

s_and_b32 s29, s28, 4294967294 // (op 1646) 

s_cmp_lt_u32 s1, s29 // (op 1646) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoop_1646 // (op 1646) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoop_1646 if false)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoop_1646:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:59], v[36:43], a[0:3], v16, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 198) 
s_barrier  // (op 10858) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:59], v[44:51], a[4:7], v16, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 662) 
s_add_u32 s1, s1, 2 // (op 1648) 
s_lshl_b64 s[18:19], s[36:37], 5 // (op 8921) 
v_mov_b64 v[84:85], s[18:19] // (op 8921) convert


v_mov_b64 v[90:91], v[24:25] // (op 8921) 
v_mov_b64 v[92:93], v[90:91] // (op 8921) 
s_mov_b32 m0, s6 // (op 8921) reg expression
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:67], v[36:43], a[8:11], v17, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 959) 
s_mov_b32 s18, 256 // (op 8921) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[60:67], v[44:51], a[12:15], v17, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 913) 
buffer_load_dwordx4 v92, s[72:75], s18 offen offset: 0 lds // (op 8921) Load value direct to lds
v_add_co_u32 v92, s[28:29], v92, v84 // (op 8921) least significant half
v_addc_co_u32 v93, s[28:29], 0, v93, s[28:29] // (op 8921) most significant half

s_add_u32 m0, m0, 4096 // (op 8921) 
buffer_load_dwordx4 v92, s[72:75], s18 offen offset: 0 lds // (op 8921) Load value direct to lds



s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[68:75], v[36:43], a[16:19], v32, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1345) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[68:75], v[44:51], a[20:23], v32, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1299) 
v_mov_b32 v84, v31 // (op 204) 

buffer_load_ubyte v26, v84, s[80:83], 0 offen offset:8 // (op 204) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:83], v[36:43], a[24:27], v33, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1731) 
s_barrier  // (op 10839) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:83], v[44:51], a[28:31], v33, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1685) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 8916) 
v_mov_b64 v[92:93], s[18:19] // (op 8916) convert


v_mov_b64 v[94:95], v[20:21] // (op 8916) 
v_mov_b64 v[96:97], v[94:95] // (op 8916) 
s_add_u32 s18, 8192, s15 // (op 8916) 
s_mov_b32 m0, s18 // (op 8916) reg expression
s_mov_b32 s19, 256 // (op 8916) call()
buffer_load_dwordx4 v96, s[68:71], s19 offen offset: 0 lds // (op 8916) Load value direct to lds
v_add_co_u32 v96, s[28:29], v96, v92 // (op 8916) least significant half
v_addc_co_u32 v97, s[28:29], 0, v97, s[28:29] // (op 8916) most significant half

s_add_u32 m0, m0, 4096 // (op 8916) 
buffer_load_dwordx4 v96, s[68:71], s19 offen offset: 0 lds // (op 8916) Load value direct to lds
v_add_co_u32 v96, s[28:29], v96, v92 // (op 8916) least significant half
v_addc_co_u32 v97, s[28:29], 0, v97, s[28:29] // (op 8916) most significant half

s_add_u32 m0, m0, 4096 // (op 8916) 
buffer_load_dwordx4 v96, s[68:71], s19 offen offset: 0 lds // (op 8916) Load value direct to lds
v_add_co_u32 v96, s[28:29], v96, v92 // (op 8916) least significant half
v_addc_co_u32 v97, s[28:29], 0, v97, s[28:29] // (op 8916) most significant half

s_add_u32 m0, m0, 4096 // (op 8916) 
buffer_load_dwordx4 v96, s[68:71], s19 offen offset: 0 lds // (op 8916) Load value direct to lds





v_mov_b64 v[92:93], s[40:41] // (op 203) convert

v_mov_b32 v85, v29 // (op 203) 
v_mov_b32 v96, v85 // (op 203) 
buffer_load_ubyte v34, v96, s[76:79], 0 offen offset:8 // (op 203) Load value
v_add_u32 v96, v96, v92 // (op 203) 
buffer_load_ubyte v35, v96, s[76:79], 0 offen offset:8 // (op 203) Load value


s_barrier  // (op 10942) 



s_waitcnt vmcnt(15)
ds_write_b8 v89, v27 offset:49920 // (op 541) Store local data 
s_barrier  // (op 10936) 



s_waitcnt vmcnt(10)
ds_write_b8 v88, v22 offset:50176 // (op 540) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v88, v23 offset:50180 // (op 540) Store local data 
s_barrier  // (op 10922) 



ds_read_b128 v[96:99], v12 offset:24576 // (op 533) Load local data 
ds_read_b128 v[100:103], v12 offset:24640 // (op 533) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 10939) 



ds_read_u8 v92, v13 offset:49920 // (op 537) Load local data 



ds_read_b128 v[104:107], v12 offset:28672 // (op 667) Load local data 
ds_read_b128 v[108:111], v12 offset:28736 // (op 667) Load local data 



ds_read_u8 v93, v13 offset:50048 // (op 671) Load local data 
s_barrier  // (op 10913) 
s_barrier  // (op 11061) 



ds_read_b128 v[52:55], v11 offset:32768 // (op 531) Load local data 
ds_read_b128 v[56:59], v11 offset:32832 // (op 531) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 10932) 



ds_read_u8 v16, v10 offset:50176 // (op 535) Load local data 



ds_read_b128 v[60:63], v11 offset:36864 // (op 916) Load local data 
ds_read_b128 v[64:67], v11 offset:36928 // (op 916) Load local data 



ds_read_u8 v17, v10 offset:50304 // (op 919) Load local data 



ds_read_b128 v[68:71], v11 offset:40960 // (op 1302) Load local data 
ds_read_b128 v[72:75], v11 offset:41024 // (op 1302) Load local data 



ds_read_u8 v32, v10 offset:50432 // (op 1305) Load local data 



ds_read_b128 v[76:79], v11 offset:45056 // (op 1688) Load local data 
ds_read_b128 v[80:83], v11 offset:45120 // (op 1688) Load local data 



ds_read_u8 v33, v10 offset:50560 // (op 1691) Load local data 
s_barrier  // (op 11091) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:59], v[96:103], a[0:3], v16, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 538) 
s_barrier  // (op 10929) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:59], v[104:111], a[4:7], v16, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 663) 
s_lshl_b64 s[18:19], s[36:37], 5 // (op 8944) 
v_mov_b64 v[112:113], s[18:19] // (op 8944) convert


v_mov_b64 v[114:115], v[90:91] // (op 8944) 
s_add_u32 s18, 24576, s8 // (op 8944) 
s_mov_b32 m0, s18 // (op 8944) reg expression
s_mov_b32 s19, 384 // (op 8944) call()
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:67], v[96:103], a[8:11], v17, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 960) 
buffer_load_dwordx4 v114, s[72:75], s19 offen offset: 0 lds // (op 8944) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[60:67], v[104:111], a[12:15], v17, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 914) 
v_add_co_u32 v114, s[28:29], v114, v112 // (op 8944) least significant half
v_addc_co_u32 v115, s[28:29], 0, v115, s[28:29] // (op 8944) most significant half

s_add_u32 m0, m0, 4096 // (op 8944) 
buffer_load_dwordx4 v114, s[72:75], s19 offen offset: 0 lds // (op 8944) Load value direct to lds




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[68:75], v[96:103], a[16:19], v32, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1346) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[68:75], v[104:111], a[20:23], v32, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1300) 
buffer_load_ubyte v27, v84, s[80:83], 0 offen offset:12 // (op 544) Load value

v_add_co_u32 v24, vcc, 256, v24 // (op 9217) least significant half
v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 9217) most significant half
v_add_u32 v31, 8, v31 // (op 9473) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:83], v[96:103], a[24:27], v33, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1732) 
s_barrier  // (op 10919) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:83], v[104:111], a[28:31], v33, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1686) 
s_lshl_b64 s[18:19], s[30:31], 5 // (op 8939) 


v_mov_b64 v[90:91], s[18:19] // (op 8939) convert


v_mov_b64 v[92:93], v[94:95] // (op 8939) 
s_add_u32 s18, 32768, s7 // (op 8939) 
s_mov_b32 m0, s18 // (op 8939) reg expression
s_mov_b32 s19, 384 // (op 8939) call()
buffer_load_dwordx4 v92, s[68:71], s19 offen offset: 0 lds // (op 8939) Load value direct to lds
v_add_co_u32 v92, s[28:29], v92, v90 // (op 8939) least significant half
v_addc_co_u32 v93, s[28:29], 0, v93, s[28:29] // (op 8939) most significant half

s_add_u32 m0, m0, 4096 // (op 8939) 
buffer_load_dwordx4 v92, s[68:71], s19 offen offset: 0 lds // (op 8939) Load value direct to lds
v_add_co_u32 v92, s[28:29], v92, v90 // (op 8939) least significant half
v_addc_co_u32 v93, s[28:29], 0, v93, s[28:29] // (op 8939) most significant half

s_add_u32 m0, m0, 4096 // (op 8939) 
buffer_load_dwordx4 v92, s[68:71], s19 offen offset: 0 lds // (op 8939) Load value direct to lds
v_add_co_u32 v92, s[28:29], v92, v90 // (op 8939) least significant half
v_addc_co_u32 v93, s[28:29], 0, v93, s[28:29] // (op 8939) most significant half

s_add_u32 m0, m0, 4096 // (op 8939) 
buffer_load_dwordx4 v92, s[68:71], s19 offen offset: 0 lds // (op 8939) Load value direct to lds






v_add_co_u32 v20, vcc, 256, v20 // (op 9079) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 9079) most significant half
v_mov_b64 v[90:91], s[40:41] // (op 543) convert

v_mov_b32 v84, v85 // (op 543) 
buffer_load_ubyte v22, v84, s[76:79], 0 offen offset:12 // (op 543) Load value
v_add_u32 v84, v84, v90 // (op 543) 
buffer_load_ubyte v23, v84, s[76:79], 0 offen offset:12 // (op 543) Load value



s_barrier  // (op 10899) 
v_add_u32 v29, 8, v29 // (op 9345) 



s_waitcnt vmcnt(15)
ds_write_b8 v87, v26 offset:49152 // (op 201) Store local data 
s_barrier  // (op 10878) 



s_waitcnt vmcnt(10)
ds_write_b8 v86, v34 offset:49408 // (op 200) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v86, v35 offset:49412 // (op 200) Store local data 
s_barrier  // (op 10864) 



ds_read_b128 v[36:39], v8 // (op 193) Load local data 
ds_read_b128 v[40:43], v8 offset:64 // (op 193) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 10906) 



ds_read_u8 v28, v9 offset:49152 // (op 197) Load local data 



ds_read_b128 v[44:47], v8 offset:4096 // (op 657) Load local data 
ds_read_b128 v[48:51], v8 offset:4160 // (op 657) Load local data 



ds_read_u8 v30, v9 offset:49280 // (op 661) Load local data 
s_barrier  // (op 10845) 



ds_read_b128 v[52:55], v7 offset:8192 // (op 191) Load local data 
ds_read_b128 v[56:59], v7 offset:8256 // (op 191) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 10885) 



ds_read_u8 v16, v6 offset:49408 // (op 195) Load local data 



ds_read_b128 v[60:63], v7 offset:12288 // (op 908) Load local data 
ds_read_b128 v[64:67], v7 offset:12352 // (op 908) Load local data 



ds_read_u8 v17, v6 offset:49536 // (op 911) Load local data 



ds_read_b128 v[68:71], v7 offset:16384 // (op 1294) Load local data 
ds_read_b128 v[72:75], v7 offset:16448 // (op 1294) Load local data 



ds_read_u8 v32, v6 offset:49664 // (op 1297) Load local data 



ds_read_b128 v[76:79], v7 offset:20480 // (op 1680) Load local data 
ds_read_b128 v[80:83], v7 offset:20544 // (op 1680) Load local data 



ds_read_u8 v33, v6 offset:49792 // (op 1683) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1646) 
s_mov_b32 s28, s18 // (op 1646) convert

s_and_b32 s29, s28, 4294967294 // (op 1646) 

s_cmp_lt_u32 s1, s29 // (op 1646) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoop_1646 // (op 1646) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoop_1646 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoop_1646:
















s_barrier  // (op 11075) 
v_lshrrev_b32 v31, 3, v1 // (op 9011) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 9011) 
v_and_b32 v20, -8, v1 // (op 9011) 
s_and_b32 s15, s18, 4294967294 // (op 9011) 
v_sub_u32 v21, v1, v20 // (op 9011) 

s_add_u32 s6, s0, s15 // (op 9011) 

v_lshlrev_b32 v20, 4, v21 // (op 9011) 

s_mul_i32 s15, s16, s50 // (op 9011) 
s_add_u32 s15, s15, s6 // (op 9011) 

s_add_u32 s6, s15, s9 // (op 9011) 

s_mul_hi_u32 s15, s6, s13 // (op 9011) 
s_sub_u32 s7, s6, s15 // (op 9011) 
s_lshr_b32 s8, s7, 1 // (op 9011) 

s_add_u32 s7, s8, s15 // (op 9011) 


s_lshr_b32 s8, s7, s11 // (op 9011) 

s_and_b32 s7, s11, 2147483648 // (op 9011) 
s_cmp_eq_i32 s7, 0 // (op 9011) 
s_mov_b32 s15, scc // (op 9011) 

s_cmp_eq_i32 s15, 0 // (op 9011) 
s_cselect_b32 s7, s6, s8 // (op 9011) 


s_mul_hi_u32 s8, s7, s12 // (op 9011) 
s_sub_u32 s15, s7, s8 // (op 9011) 
s_lshr_b32 s28, s15, 1 // (op 9011) 

s_add_u32 s15, s28, s8 // (op 9011) 


s_lshr_b32 s8, s15, s10 // (op 9011) 

s_and_b32 s15, s10, 2147483648 // (op 9011) 
s_cmp_eq_i32 s15, 0 // (op 9011) 
s_mov_b32 s28, scc // (op 9011) 

s_cmp_eq_i32 s28, 0 // (op 9011) 
s_cselect_b32 s15, s7, s8 // (op 9011) 


s_mul_i32 s8, s7, s18 // (op 9011) 


v_lshl_add_u32 v21, s15, 7, v31 // (op 9011) 


s_sub_u32 s7, s6, s8 // (op 9011) 


v_lshl_add_u32 v31, s7, 7, v20 // (op 9011) 


v_mov_b32 v29, 0 // (op 9011) convert
v_mov_b32 v28, v31 // (op 9011) convert

v_mul_lo_u32 v25, v21, s31 // (op 9011) most significant: low of low * high
v_mul_hi_u32 v20, v21, s30 // (op 9011) most significant: high of low * low
v_mul_lo_u32 v24, v21, s30 // (op 9011) least significant: low of low * low
v_add_u32 v25, v25, v20 // (op 9011) most significant: sum

v_add_co_u32 v24, s[18:19], v24, v28 // (op 9011) least significant half
v_addc_co_u32 v25, s[18:19], v25, v29, s[18:19] // (op 9011) most significant half



s_mul_i32 s19, s30, s35 // (op 9013) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 9013) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 9013) least significant
s_mul_hi_u32 s6, s30, s34 // (op 9013) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9013) most significant: sum
s_add_u32 s19, s19, s6 // (op 9013) most significant: sum


s_mov_b32 s68, s32 // (op 9013) reg expression
s_mov_b32 s69, s33 // (op 9013) reg expression
s_mov_b32 s70, s18 // (op 9013) reg expression
s_mov_b32 s71, 131072 // (op 9013) call()

v_lshrrev_b32 v26, 3, v1 // (op 9161) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 9161) 
v_and_b32 v31, -8, v1 // (op 9161) 
s_and_b32 s15, s18, 4294967294 // (op 9161) 
v_sub_u32 v20, v1, v31 // (op 9161) 

s_add_u32 s6, s0, s15 // (op 9161) 

v_lshlrev_b32 v21, 4, v20 // (op 9161) 

s_mul_i32 s15, s16, s50 // (op 9161) 
s_add_u32 s15, s15, s6 // (op 9161) 

s_add_u32 s6, s15, s9 // (op 9161) 

s_mul_hi_u32 s15, s6, s13 // (op 9161) 
s_sub_u32 s7, s6, s15 // (op 9161) 
s_lshr_b32 s8, s7, 1 // (op 9161) 

s_add_u32 s7, s8, s15 // (op 9161) 


s_lshr_b32 s8, s7, s11 // (op 9161) 

s_and_b32 s7, s11, 2147483648 // (op 9161) 
s_cmp_eq_i32 s7, 0 // (op 9161) 
s_mov_b32 s15, scc // (op 9161) 

s_cmp_eq_i32 s15, 0 // (op 9161) 
s_cselect_b32 s7, s6, s8 // (op 9161) 


s_mul_hi_u32 s8, s7, s12 // (op 9161) 
s_sub_u32 s15, s7, s8 // (op 9161) 
s_lshr_b32 s28, s15, 1 // (op 9161) 

s_add_u32 s15, s28, s8 // (op 9161) 


s_lshr_b32 s8, s15, s10 // (op 9161) 

s_and_b32 s15, s10, 2147483648 // (op 9161) 
s_cmp_eq_i32 s15, 0 // (op 9161) 
s_mov_b32 s28, scc // (op 9161) 

s_cmp_eq_i32 s28, 0 // (op 9161) 
s_cselect_b32 s15, s7, s8 // (op 9161) 


s_mul_i32 s8, s15, s14 // (op 9161) 

s_sub_u32 s15, s7, s8 // (op 9161) 

s_mul_i32 s8, s7, s18 // (op 9161) 


v_lshl_add_u32 v20, s15, 6, v26 // (op 9161) 


s_sub_u32 s7, s6, s8 // (op 9161) 


v_mul_lo_u32 v29, v20, s37 // (op 9161) most significant: low of low * high
v_mul_hi_u32 v26, v20, s36 // (op 9161) most significant: high of low * low
v_mul_lo_u32 v28, v20, s36 // (op 9161) least significant: low of low * low
v_add_u32 v29, v29, v26 // (op 9161) most significant: sum


v_lshl_add_u32 v20, s7, 7, v21 // (op 9161) 


v_mov_b32 v35, 0 // (op 9161) convert
v_mov_b32 v34, v20 // (op 9161) convert

v_add_co_u32 v20, s[18:19], v34, v28 // (op 9161) least significant half
v_addc_co_u32 v21, s[18:19], v35, v29, s[18:19] // (op 9161) most significant half



s_mul_i32 s19, s36, s23 // (op 9163) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 9163) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 9163) least significant
s_mul_hi_u32 s6, s36, s22 // (op 9163) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9163) most significant: sum
s_add_u32 s19, s19, s6 // (op 9163) most significant: sum


s_mov_b32 s72, s38 // (op 9163) reg expression
s_mov_b32 s73, s39 // (op 9163) reg expression
s_mov_b32 s74, s18 // (op 9163) reg expression
s_mov_b32 s75, 131072 // (op 9163) call()

v_lshrrev_b32 v26, 2, v1 // (op 9289) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 9289) 
v_lshlrev_b32 v31, 1, v26 // (op 9289) 

s_and_b32 s15, s18, 4294967294 // (op 9289) 
v_and_b32 v26, -4, v1 // (op 9289) 
s_add_u32 s6, s0, s15 // (op 9289) 

v_sub_u32 v28, v1, v26 // (op 9289) 

s_mul_i32 s15, s16, s50 // (op 9289) 
s_add_u32 s15, s15, s6 // (op 9289) 

s_add_u32 s6, s15, s9 // (op 9289) 

s_mul_hi_u32 s15, s6, s13 // (op 9289) 
s_sub_u32 s7, s6, s15 // (op 9289) 
s_lshr_b32 s8, s7, 1 // (op 9289) 

s_add_u32 s7, s8, s15 // (op 9289) 


s_lshr_b32 s8, s7, s11 // (op 9289) 

s_and_b32 s7, s11, 2147483648 // (op 9289) 
s_cmp_eq_i32 s7, 0 // (op 9289) 
s_mov_b32 s15, scc // (op 9289) 

s_cmp_eq_i32 s15, 0 // (op 9289) 
s_cselect_b32 s7, s6, s8 // (op 9289) 


s_mul_hi_u32 s8, s7, s12 // (op 9289) 
s_sub_u32 s15, s7, s8 // (op 9289) 
s_lshr_b32 s28, s15, 1 // (op 9289) 

s_add_u32 s15, s28, s8 // (op 9289) 


s_lshr_b32 s8, s15, s10 // (op 9289) 

s_and_b32 s15, s10, 2147483648 // (op 9289) 
s_cmp_eq_i32 s15, 0 // (op 9289) 
s_mov_b32 s28, scc // (op 9289) 

s_cmp_eq_i32 s28, 0 // (op 9289) 
s_cselect_b32 s15, s7, s8 // (op 9289) 


s_mul_i32 s8, s7, s18 // (op 9289) 


v_lshl_add_u32 v26, s15, 7, v31 // (op 9289) 


s_sub_u32 s7, s6, s8 // (op 9289) 


v_lshl_add_u32 v29, s7, 2, v28 // (op 9289) 


v_mul_lo_u32 v28, v26, s40 // (op 9289) 
v_add_u32 v28, v28, v29 // (op 9289) 


s_mul_i32 s19, s40, s35 // (op 9291) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 9291) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 9291) least significant
s_mul_hi_u32 s6, s40, s34 // (op 9291) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9291) most significant: sum
s_add_u32 s19, s19, s6 // (op 9291) most significant: sum


s_mov_b32 s76, s42 // (op 9291) reg expression
s_mov_b32 s77, s43 // (op 9291) reg expression
s_mov_b32 s78, s18 // (op 9291) reg expression
s_mov_b32 s79, 131072 // (op 9291) call()

v_lshrrev_b32 v26, 2, v1 // (op 9417) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 9417) 
v_and_b32 v29, -4, v1 // (op 9417) 
s_and_b32 s15, s18, 4294967294 // (op 9417) 
v_sub_u32 v31, v1, v29 // (op 9417) 

s_add_u32 s6, s0, s15 // (op 9417) 

s_mul_i32 s15, s16, s50 // (op 9417) 
s_add_u32 s15, s15, s6 // (op 9417) 

s_add_u32 s6, s15, s9 // (op 9417) 

s_mul_hi_u32 s15, s6, s13 // (op 9417) 
s_sub_u32 s7, s6, s15 // (op 9417) 
s_lshr_b32 s8, s7, 1 // (op 9417) 

s_add_u32 s7, s8, s15 // (op 9417) 


s_lshr_b32 s8, s7, s11 // (op 9417) 

s_and_b32 s7, s11, 2147483648 // (op 9417) 
s_cmp_eq_i32 s7, 0 // (op 9417) 
s_mov_b32 s15, scc // (op 9417) 

s_cmp_eq_i32 s15, 0 // (op 9417) 
s_cselect_b32 s7, s6, s8 // (op 9417) 


s_mul_hi_u32 s8, s7, s12 // (op 9417) 
s_sub_u32 s15, s7, s8 // (op 9417) 
s_lshr_b32 s28, s15, 1 // (op 9417) 

s_add_u32 s15, s28, s8 // (op 9417) 


s_lshr_b32 s8, s15, s10 // (op 9417) 

s_and_b32 s15, s10, 2147483648 // (op 9417) 
s_cmp_eq_i32 s15, 0 // (op 9417) 
s_mov_b32 s28, scc // (op 9417) 

s_cmp_eq_i32 s28, 0 // (op 9417) 
s_cselect_b32 s15, s7, s8 // (op 9417) 


s_mul_i32 s8, s15, s14 // (op 9417) 

s_sub_u32 s15, s7, s8 // (op 9417) 

v_lshl_add_u32 v29, s15, 6, v26 // (op 9417) 


s_mul_i32 s8, s7, s18 // (op 9417) 


v_mul_lo_u32 v26, v29, s44 // (op 9417) 

s_sub_u32 s7, s6, s8 // (op 9417) 


v_lshl_add_u32 v29, s7, 2, v31 // (op 9417) 


v_add_u32 v31, v29, v26 // (op 9417) 


s_mul_i32 s19, s44, s23 // (op 9419) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 9419) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 9419) least significant
s_mul_hi_u32 s6, s44, s22 // (op 9419) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9419) most significant: sum
s_add_u32 s19, s19, s6 // (op 9419) most significant: sum


s_mov_b32 s80, s46 // (op 9419) reg expression
s_mov_b32 s81, s47 // (op 9419) reg expression
s_mov_b32 s82, s18 // (op 9419) reg expression
s_mov_b32 s83, 131072 // (op 9419) call()

v_and_b32 v26, -8, v1 // (op 9964) 
v_sub_u32 v29, v1, v26 // (op 9964) 

v_lshlrev_b32 v26, 4, v29 // (op 9964) 

v_lshrrev_b32 v29, 3, v1 // (op 9964) 
v_lshl_add_u32 v34, v29, 7, v26 // (op 9964) 


s_nop 0
v_readfirstlane_b32 s15, v34 // (op 9964) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -8, v1 // (op 10039) 
v_sub_u32 v29, v1, v26 // (op 10039) 

v_lshlrev_b32 v26, 4, v29 // (op 10039) 

v_lshrrev_b32 v29, 3, v1 // (op 10039) 
v_lshl_add_u32 v34, v29, 7, v26 // (op 10039) 


s_nop 0
v_readfirstlane_b32 s6, v34 // (op 10039) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 10114) 
v_sub_u32 v29, v1, v26 // (op 10114) 

v_lshrrev_b32 v26, 2, v1 // (op 10114) 
v_lshl_add_u32 v34, v26, 3, v29 // (op 10114) 


v_and_b32 v26, -4, v1 // (op 10190) 
v_sub_u32 v29, v1, v26 // (op 10190) 

v_and_b32 v26, 4294967292, v1 // (op 10190) 
v_add_u32 v35, v26, v29 // (op 10190) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1652) 
s_and_b32 s7, s18, 4294967294 // (op 1652) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1651) 
s_mov_b32 s8, s18 // (op 1651) convert

s_cmp_lt_u32 s7, s8 // (op 1651) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoopTail_1651 // (op 1651) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoopTail_1651 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoopTail_1651:
s_barrier  // (op 10842) 
s_add_u32 s7, s7, 1 // (op 1653) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 8926) 
v_mov_b64 v[16:17], s[18:19] // (op 8926) convert


v_mov_b64 v[22:23], v[24:25] // (op 8926) 
v_mov_b64 v[26:27], v[22:23] // (op 8926) 
s_add_u32 s8, 8192, s15 // (op 8926) 
s_mov_b32 m0, s8 // (op 8926) reg expression
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 8926) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 8926) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 8926) most significant half

s_add_u32 m0, m0, 4096 // (op 8926) 
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 8926) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 8926) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 8926) most significant half

s_add_u32 m0, m0, 4096 // (op 8926) 
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 8926) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 8926) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 8926) most significant half

s_add_u32 m0, m0, 4096 // (op 8926) 
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 8926) Load value direct to lds




v_add_co_u32 v24, vcc, 128, v24 // (op 9001) least significant half
v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 9001) most significant half
s_barrier  // (op 10861) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 8932) 
v_mov_b64 v[16:17], s[18:19] // (op 8932) convert


v_mov_b64 v[22:23], v[20:21] // (op 8932) 
v_mov_b64 v[26:27], v[22:23] // (op 8932) 
s_mov_b32 m0, s6 // (op 8932) reg expression
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 8932) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 8932) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 8932) most significant half

s_add_u32 m0, m0, 4096 // (op 8932) 
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 8932) Load value direct to lds



v_add_co_u32 v20, vcc, 128, v20 // (op 9153) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 9153) most significant half

v_mov_b64 v[16:17], s[40:41] // (op 477) convert

v_mov_b32 v22, v28 // (op 477) 
v_mov_b32 v23, v22 // (op 477) 
buffer_load_ubyte v26, v23, s[76:79], 0 offen offset: 0 // (op 477) Load value
v_add_u32 v23, v23, v16 // (op 477) 
buffer_load_ubyte v27, v23, s[76:79], 0 offen offset: 0 // (op 477) Load value



v_add_u32 v28, 4, v28 // (op 9281) 


s_barrier  // (op 10881) 
v_mov_b32 v16, v31 // (op 478) 




buffer_load_ubyte v17, v16, s[80:83], 0 offen offset: 0 // (op 478) Load value

v_add_u32 v31, 4, v31 // (op 9409) 
s_waitcnt vmcnt(2)
ds_write_b8 v34, v26 offset:49408 // (op 474) Store local data 
s_barrier  // (op 10848) 
s_waitcnt vmcnt(1)
ds_write_b8 v34, v27 offset:49412 // (op 474) Store local data 




ds_read_b128 v[36:39], v7 offset:8192 // (op 465) Load local data 
s_barrier  // (op 10902) 
ds_read_b128 v[40:43], v7 offset:8256 // (op 465) Load local data 



s_barrier  // (op 10868) 
s_waitcnt vmcnt(0)
ds_write_b8 v35, v17 offset:49152 // (op 475) Store local data 




ds_read_b128 v[44:47], v8 // (op 467) Load local data 
ds_read_b128 v[48:51], v8 offset:64 // (op 467) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 10889) 



ds_read_u8 v16, v6 offset:49408 // (op 469) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10909) 



ds_read_u8 v17, v9 offset:49152 // (op 471) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:51], a[0:3], v16, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 472) 
ds_read_b128 v[52:55], v8 offset:4096 // (op 636) Load local data 
ds_read_b128 v[56:59], v8 offset:4160 // (op 636) Load local data 



ds_read_u8 v22, v9 offset:49280 // (op 640) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:43], v[52:59], a[4:7], v16, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 641) 
ds_read_b128 v[60:63], v7 offset:12288 // (op 854) Load local data 


ds_read_b128 v[64:67], v7 offset:12352 // (op 854) Load local data 



ds_read_u8 v16, v6 offset:49536 // (op 857) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:67], v[44:51], a[8:11], v16, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 859) 
ds_read_b128 v[36:39], v7 offset:16384 // (op 1240) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[60:67], v[52:59], a[12:15], v16, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 890) 
ds_read_b128 v[40:43], v7 offset:16448 // (op 1240) Load local data 





ds_read_u8 v16, v6 offset:49664 // (op 1243) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[36:43], v[44:51], a[16:19], v16, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1245) 
ds_read_b128 v[60:63], v7 offset:20480 // (op 1626) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:43], v[52:59], a[20:23], v16, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1276) 
ds_read_b128 v[64:67], v7 offset:20544 // (op 1626) Load local data 





ds_read_u8 v16, v6 offset:49792 // (op 1629) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:67], v[44:51], a[24:27], v16, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1631) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[60:67], v[52:59], a[28:31], v16, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1662) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 1651) 
s_mov_b32 s8, s18 // (op 1651) convert

s_cmp_lt_u32 s7, s8 // (op 1651) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoopTail_1651 // (op 1651) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoopTail_1651 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoopTail_1651:









s_barrier  // (op 11079) 




v_cmp_eq_f32 vcc, 0.00000, v18 // (op 238) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_0_beta_238 // (op 238) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_0_beta_238
v_accvgpr_read v14, a0 // (op 246) 
v_accvgpr_read v15, a1 // (op 246) 
v_accvgpr_read v16, a2 // (op 246) 
v_accvgpr_read v17, a3 // (op 246) 
v_mul_f32 v20, v19, v14 // (op 246) 
v_mul_f32 v21, v19, v15 // (op 246) 
v_mul_f32 v22, v19, v16 // (op 246) 
v_mul_f32 v23, v19, v17 // (op 246) 

v_cvt_f16_f32 v14, v20 // (op 248) 
v_cvt_f16_f32 v15, v21 // (op 248) 
v_cvt_f16_f32 v16, v22 // (op 248) 
v_cvt_f16_f32 v17, v23 // (op 248) 


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

v_cvt_f16_f32 v14, v20 // (op 2157) 
v_cvt_f16_f32 v15, v21 // (op 2157) 
v_cvt_f16_f32 v16, v22 // (op 2157) 
v_cvt_f16_f32 v17, v23 // (op 2157) 


v_pack_b32_f16 v20, v14, v15 // (op 2158) 
v_pack_b32_f16 v21, v16, v17 // (op 2158) 

s_lshl_b32 s15, s24, 6 // (op 2158) 
v_add_u32 v0, v5, s15 // (op 2158) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 2158) Store value



v_accvgpr_read v14, a8 // (op 2181) 
v_accvgpr_read v15, a9 // (op 2181) 
v_accvgpr_read v16, a10 // (op 2181) 
v_accvgpr_read v17, a11 // (op 2181) 
v_mul_f32 v20, v19, v14 // (op 2181) 
v_mul_f32 v21, v19, v15 // (op 2181) 
v_mul_f32 v22, v19, v16 // (op 2181) 
v_mul_f32 v23, v19, v17 // (op 2181) 

v_cvt_f16_f32 v14, v20 // (op 2183) 
v_cvt_f16_f32 v15, v21 // (op 2183) 
v_cvt_f16_f32 v16, v22 // (op 2183) 
v_cvt_f16_f32 v17, v23 // (op 2183) 


v_pack_b32_f16 v20, v14, v15 // (op 2185) 
v_pack_b32_f16 v21, v16, v17 // (op 2185) 

v_add_u32 v0, 64, v5 // (op 2185) 

buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 2185) Store value



v_accvgpr_read v14, a12 // (op 2191) 
v_accvgpr_read v15, a13 // (op 2191) 
v_accvgpr_read v16, a14 // (op 2191) 
v_accvgpr_read v17, a15 // (op 2191) 
v_mul_f32 v20, v19, v14 // (op 2191) 
v_mul_f32 v21, v19, v15 // (op 2191) 
v_mul_f32 v22, v19, v16 // (op 2191) 
v_mul_f32 v23, v19, v17 // (op 2191) 

v_cvt_f16_f32 v14, v20 // (op 2193) 
v_cvt_f16_f32 v15, v21 // (op 2193) 
v_cvt_f16_f32 v16, v22 // (op 2193) 
v_cvt_f16_f32 v17, v23 // (op 2193) 


v_pack_b32_f16 v20, v14, v15 // (op 2187) 
v_pack_b32_f16 v21, v16, v17 // (op 2187) 

s_lshl_b32 s15, s24, 6 // (op 2187) 
s_add_u32 s15, s15, 64 // (op 2187) 
v_add_u32 v0, v5, s15 // (op 2187) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 2187) Store value



v_accvgpr_read v14, a16 // (op 2221) 
v_accvgpr_read v15, a17 // (op 2221) 
v_accvgpr_read v16, a18 // (op 2221) 
v_accvgpr_read v17, a19 // (op 2221) 
v_mul_f32 v20, v19, v14 // (op 2221) 
v_mul_f32 v21, v19, v15 // (op 2221) 
v_mul_f32 v22, v19, v16 // (op 2221) 
v_mul_f32 v23, v19, v17 // (op 2221) 

v_cvt_f16_f32 v14, v20 // (op 2223) 
v_cvt_f16_f32 v15, v21 // (op 2223) 
v_cvt_f16_f32 v16, v22 // (op 2223) 
v_cvt_f16_f32 v17, v23 // (op 2223) 


v_pack_b32_f16 v20, v14, v15 // (op 2225) 
v_pack_b32_f16 v21, v16, v17 // (op 2225) 

v_add_u32 v0, 128, v5 // (op 2225) 

buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 2225) Store value



v_accvgpr_read v14, a20 // (op 2231) 
v_accvgpr_read v15, a21 // (op 2231) 
v_accvgpr_read v16, a22 // (op 2231) 
v_accvgpr_read v17, a23 // (op 2231) 
v_mul_f32 v20, v19, v14 // (op 2231) 
v_mul_f32 v21, v19, v15 // (op 2231) 
v_mul_f32 v22, v19, v16 // (op 2231) 
v_mul_f32 v23, v19, v17 // (op 2231) 

v_cvt_f16_f32 v14, v20 // (op 2233) 
v_cvt_f16_f32 v15, v21 // (op 2233) 
v_cvt_f16_f32 v16, v22 // (op 2233) 
v_cvt_f16_f32 v17, v23 // (op 2233) 


v_pack_b32_f16 v20, v14, v15 // (op 2227) 
v_pack_b32_f16 v21, v16, v17 // (op 2227) 

s_lshl_b32 s15, s24, 6 // (op 2227) 
s_add_u32 s15, s15, 128 // (op 2227) 
v_add_u32 v0, v5, s15 // (op 2227) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 2227) Store value



v_accvgpr_read v14, a24 // (op 2261) 
v_accvgpr_read v15, a25 // (op 2261) 
v_accvgpr_read v16, a26 // (op 2261) 
v_accvgpr_read v17, a27 // (op 2261) 
v_mul_f32 v20, v19, v14 // (op 2261) 
v_mul_f32 v21, v19, v15 // (op 2261) 
v_mul_f32 v22, v19, v16 // (op 2261) 
v_mul_f32 v23, v19, v17 // (op 2261) 

v_cvt_f16_f32 v14, v20 // (op 2263) 
v_cvt_f16_f32 v15, v21 // (op 2263) 
v_cvt_f16_f32 v16, v22 // (op 2263) 
v_cvt_f16_f32 v17, v23 // (op 2263) 


v_pack_b32_f16 v20, v14, v15 // (op 2265) 
v_pack_b32_f16 v21, v16, v17 // (op 2265) 

v_add_u32 v0, 192, v5 // (op 2265) 

buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 2265) Store value



v_accvgpr_read v14, a28 // (op 2271) 
v_accvgpr_read v15, a29 // (op 2271) 
v_accvgpr_read v16, a30 // (op 2271) 
v_accvgpr_read v17, a31 // (op 2271) 
v_mul_f32 v20, v19, v14 // (op 2271) 
v_mul_f32 v21, v19, v15 // (op 2271) 
v_mul_f32 v22, v19, v16 // (op 2271) 
v_mul_f32 v23, v19, v17 // (op 2271) 

v_cvt_f16_f32 v14, v20 // (op 2273) 
v_cvt_f16_f32 v15, v21 // (op 2273) 
v_cvt_f16_f32 v16, v22 // (op 2273) 
v_cvt_f16_f32 v17, v23 // (op 2273) 


v_pack_b32_f16 v20, v14, v15 // (op 2267) 
v_pack_b32_f16 v21, v16, v17 // (op 2267) 

s_lshl_b32 s15, s24, 6 // (op 2267) 
s_add_u32 s15, s15, 192 // (op 2267) 
v_add_u32 v0, v5, s15 // (op 2267) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 2267) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_0_beta_238 // (op 238) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_0_beta_238
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_0_beta_238:


v_add_u32 v0, 192, v4 // (op 2463) 

buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 2463) Load value

v_accvgpr_read v20, a24 // (op 2454) 
v_accvgpr_read v21, a25 // (op 2454) 
v_accvgpr_read v22, a26 // (op 2454) 
v_accvgpr_read v23, a27 // (op 2454) 
v_mul_f32 v24, v19, v20 // (op 2454) 
v_mul_f32 v25, v19, v21 // (op 2454) 
v_mul_f32 v26, v19, v22 // (op 2454) 
v_mul_f32 v27, v19, v23 // (op 2454) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 2462) 
v_lshrrev_b32 v17, 16, v14 // (op 2462) 
v_cvt_f32_f16 v17, v17 // (op 2462) 
v_mul_f32 v20, v18, v16 // (op 2462) 
v_mul_f32 v21, v18, v17 // (op 2462) 
v_cvt_f32_f16 v16, v15 // (op 2462) 
v_lshrrev_b32 v17, 16, v15 // (op 2462) 
v_cvt_f32_f16 v17, v17 // (op 2462) 
v_mul_f32 v22, v18, v16 // (op 2462) 
v_mul_f32 v23, v18, v17 // (op 2462) 


v_add_f32 v14, v20, v24 // (op 2455) 
v_add_f32 v15, v21, v25 // (op 2455) 
v_add_f32 v16, v22, v26 // (op 2455) 
v_add_f32 v17, v23, v27 // (op 2455) 


v_cvt_f16_f32 v20, v14 // (op 2456) 
v_cvt_f16_f32 v21, v15 // (op 2456) 
v_cvt_f16_f32 v22, v16 // (op 2456) 
v_cvt_f16_f32 v23, v17 // (op 2456) 


v_pack_b32_f16 v14, v20, v21 // (op 2458) 
v_pack_b32_f16 v15, v22, v23 // (op 2458) 

v_add_u32 v0, 192, v5 // (op 2458) 

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 2458) Store value





s_lshl_b32 s15, s4, 6 // (op 2468) 
s_add_u32 s15, s15, 192 // (op 2468) 
v_add_u32 v0, v4, s15 // (op 2468) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 2468) Load value

v_accvgpr_read v20, a28 // (op 2470) 
v_accvgpr_read v21, a29 // (op 2470) 
v_accvgpr_read v22, a30 // (op 2470) 
v_accvgpr_read v23, a31 // (op 2470) 
v_mul_f32 v24, v19, v20 // (op 2470) 
v_mul_f32 v25, v19, v21 // (op 2470) 
v_mul_f32 v26, v19, v22 // (op 2470) 
v_mul_f32 v27, v19, v23 // (op 2470) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 2465) 
v_lshrrev_b32 v17, 16, v14 // (op 2465) 
v_cvt_f32_f16 v17, v17 // (op 2465) 
v_mul_f32 v20, v18, v16 // (op 2465) 
v_mul_f32 v21, v18, v17 // (op 2465) 
v_cvt_f32_f16 v16, v15 // (op 2465) 
v_lshrrev_b32 v17, 16, v15 // (op 2465) 
v_cvt_f32_f16 v17, v17 // (op 2465) 
v_mul_f32 v22, v18, v16 // (op 2465) 
v_mul_f32 v23, v18, v17 // (op 2465) 


v_add_f32 v14, v20, v24 // (op 2466) 
v_add_f32 v15, v21, v25 // (op 2466) 
v_add_f32 v16, v22, v26 // (op 2466) 
v_add_f32 v17, v23, v27 // (op 2466) 


v_cvt_f16_f32 v20, v14 // (op 2467) 
v_cvt_f16_f32 v21, v15 // (op 2467) 
v_cvt_f16_f32 v22, v16 // (op 2467) 
v_cvt_f16_f32 v23, v17 // (op 2467) 


v_pack_b32_f16 v14, v20, v21 // (op 2460) 
v_pack_b32_f16 v15, v22, v23 // (op 2460) 

s_lshl_b32 s15, s24, 6 // (op 2460) 
s_add_u32 s15, s15, 192 // (op 2460) 
v_add_u32 v0, v5, s15 // (op 2460) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 2460) Store value





v_add_u32 v0, 128, v4 // (op 2418) 

buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 2418) Load value

v_accvgpr_read v20, a16 // (op 2409) 
v_accvgpr_read v21, a17 // (op 2409) 
v_accvgpr_read v22, a18 // (op 2409) 
v_accvgpr_read v23, a19 // (op 2409) 
v_mul_f32 v24, v19, v20 // (op 2409) 
v_mul_f32 v25, v19, v21 // (op 2409) 
v_mul_f32 v26, v19, v22 // (op 2409) 
v_mul_f32 v27, v19, v23 // (op 2409) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 2417) 
v_lshrrev_b32 v17, 16, v14 // (op 2417) 
v_cvt_f32_f16 v17, v17 // (op 2417) 
v_mul_f32 v20, v18, v16 // (op 2417) 
v_mul_f32 v21, v18, v17 // (op 2417) 
v_cvt_f32_f16 v16, v15 // (op 2417) 
v_lshrrev_b32 v17, 16, v15 // (op 2417) 
v_cvt_f32_f16 v17, v17 // (op 2417) 
v_mul_f32 v22, v18, v16 // (op 2417) 
v_mul_f32 v23, v18, v17 // (op 2417) 


v_add_f32 v14, v20, v24 // (op 2410) 
v_add_f32 v15, v21, v25 // (op 2410) 
v_add_f32 v16, v22, v26 // (op 2410) 
v_add_f32 v17, v23, v27 // (op 2410) 


v_cvt_f16_f32 v20, v14 // (op 2411) 
v_cvt_f16_f32 v21, v15 // (op 2411) 
v_cvt_f16_f32 v22, v16 // (op 2411) 
v_cvt_f16_f32 v23, v17 // (op 2411) 


v_pack_b32_f16 v14, v20, v21 // (op 2413) 
v_pack_b32_f16 v15, v22, v23 // (op 2413) 

v_add_u32 v0, 128, v5 // (op 2413) 

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 2413) Store value





s_lshl_b32 s15, s4, 6 // (op 2423) 
s_add_u32 s15, s15, 128 // (op 2423) 
v_add_u32 v0, v4, s15 // (op 2423) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 2423) Load value

v_accvgpr_read v20, a20 // (op 2425) 
v_accvgpr_read v21, a21 // (op 2425) 
v_accvgpr_read v22, a22 // (op 2425) 
v_accvgpr_read v23, a23 // (op 2425) 
v_mul_f32 v24, v19, v20 // (op 2425) 
v_mul_f32 v25, v19, v21 // (op 2425) 
v_mul_f32 v26, v19, v22 // (op 2425) 
v_mul_f32 v27, v19, v23 // (op 2425) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 2420) 
v_lshrrev_b32 v17, 16, v14 // (op 2420) 
v_cvt_f32_f16 v17, v17 // (op 2420) 
v_mul_f32 v20, v18, v16 // (op 2420) 
v_mul_f32 v21, v18, v17 // (op 2420) 
v_cvt_f32_f16 v16, v15 // (op 2420) 
v_lshrrev_b32 v17, 16, v15 // (op 2420) 
v_cvt_f32_f16 v17, v17 // (op 2420) 
v_mul_f32 v22, v18, v16 // (op 2420) 
v_mul_f32 v23, v18, v17 // (op 2420) 


v_add_f32 v14, v20, v24 // (op 2421) 
v_add_f32 v15, v21, v25 // (op 2421) 
v_add_f32 v16, v22, v26 // (op 2421) 
v_add_f32 v17, v23, v27 // (op 2421) 


v_cvt_f16_f32 v20, v14 // (op 2422) 
v_cvt_f16_f32 v21, v15 // (op 2422) 
v_cvt_f16_f32 v22, v16 // (op 2422) 
v_cvt_f16_f32 v23, v17 // (op 2422) 


v_pack_b32_f16 v14, v20, v21 // (op 2415) 
v_pack_b32_f16 v15, v22, v23 // (op 2415) 

s_lshl_b32 s15, s24, 6 // (op 2415) 
s_add_u32 s15, s15, 128 // (op 2415) 
v_add_u32 v0, v5, s15 // (op 2415) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 2415) Store value





v_add_u32 v0, 64, v4 // (op 2373) 

buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 2373) Load value

v_accvgpr_read v20, a8 // (op 2364) 
v_accvgpr_read v21, a9 // (op 2364) 
v_accvgpr_read v22, a10 // (op 2364) 
v_accvgpr_read v23, a11 // (op 2364) 
v_mul_f32 v24, v19, v20 // (op 2364) 
v_mul_f32 v25, v19, v21 // (op 2364) 
v_mul_f32 v26, v19, v22 // (op 2364) 
v_mul_f32 v27, v19, v23 // (op 2364) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 2372) 
v_lshrrev_b32 v17, 16, v14 // (op 2372) 
v_cvt_f32_f16 v17, v17 // (op 2372) 
v_mul_f32 v20, v18, v16 // (op 2372) 
v_mul_f32 v21, v18, v17 // (op 2372) 
v_cvt_f32_f16 v16, v15 // (op 2372) 
v_lshrrev_b32 v17, 16, v15 // (op 2372) 
v_cvt_f32_f16 v17, v17 // (op 2372) 
v_mul_f32 v22, v18, v16 // (op 2372) 
v_mul_f32 v23, v18, v17 // (op 2372) 


v_add_f32 v14, v20, v24 // (op 2365) 
v_add_f32 v15, v21, v25 // (op 2365) 
v_add_f32 v16, v22, v26 // (op 2365) 
v_add_f32 v17, v23, v27 // (op 2365) 


v_cvt_f16_f32 v20, v14 // (op 2366) 
v_cvt_f16_f32 v21, v15 // (op 2366) 
v_cvt_f16_f32 v22, v16 // (op 2366) 
v_cvt_f16_f32 v23, v17 // (op 2366) 


v_pack_b32_f16 v14, v20, v21 // (op 2368) 
v_pack_b32_f16 v15, v22, v23 // (op 2368) 

v_add_u32 v0, 64, v5 // (op 2368) 

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 2368) Store value





s_lshl_b32 s15, s4, 6 // (op 2378) 
s_add_u32 s15, s15, 64 // (op 2378) 
v_add_u32 v0, v4, s15 // (op 2378) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 2378) Load value

v_accvgpr_read v20, a12 // (op 2380) 
v_accvgpr_read v21, a13 // (op 2380) 
v_accvgpr_read v22, a14 // (op 2380) 
v_accvgpr_read v23, a15 // (op 2380) 
v_mul_f32 v24, v19, v20 // (op 2380) 
v_mul_f32 v25, v19, v21 // (op 2380) 
v_mul_f32 v26, v19, v22 // (op 2380) 
v_mul_f32 v27, v19, v23 // (op 2380) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 2375) 
v_lshrrev_b32 v17, 16, v14 // (op 2375) 
v_cvt_f32_f16 v17, v17 // (op 2375) 
v_mul_f32 v20, v18, v16 // (op 2375) 
v_mul_f32 v21, v18, v17 // (op 2375) 
v_cvt_f32_f16 v16, v15 // (op 2375) 
v_lshrrev_b32 v17, 16, v15 // (op 2375) 
v_cvt_f32_f16 v17, v17 // (op 2375) 
v_mul_f32 v22, v18, v16 // (op 2375) 
v_mul_f32 v23, v18, v17 // (op 2375) 


v_add_f32 v14, v20, v24 // (op 2376) 
v_add_f32 v15, v21, v25 // (op 2376) 
v_add_f32 v16, v22, v26 // (op 2376) 
v_add_f32 v17, v23, v27 // (op 2376) 


v_cvt_f16_f32 v20, v14 // (op 2377) 
v_cvt_f16_f32 v21, v15 // (op 2377) 
v_cvt_f16_f32 v22, v16 // (op 2377) 
v_cvt_f16_f32 v23, v17 // (op 2377) 


v_pack_b32_f16 v14, v20, v21 // (op 2370) 
v_pack_b32_f16 v15, v22, v23 // (op 2370) 

s_lshl_b32 s15, s24, 6 // (op 2370) 
s_add_u32 s15, s15, 64 // (op 2370) 
v_add_u32 v0, v5, s15 // (op 2370) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 2370) Store value





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
v_cvt_f32_f16 v16, v14 // (op 264) 
v_lshrrev_b32 v17, 16, v14 // (op 264) 
v_cvt_f32_f16 v17, v17 // (op 264) 
v_mul_f32 v20, v18, v16 // (op 264) 
v_mul_f32 v21, v18, v17 // (op 264) 
v_cvt_f32_f16 v16, v15 // (op 264) 
v_lshrrev_b32 v17, 16, v15 // (op 264) 
v_cvt_f32_f16 v17, v17 // (op 264) 
v_mul_f32 v22, v18, v16 // (op 264) 
v_mul_f32 v23, v18, v17 // (op 264) 


v_add_f32 v14, v20, v24 // (op 260) 
v_add_f32 v15, v21, v25 // (op 260) 
v_add_f32 v16, v22, v26 // (op 260) 
v_add_f32 v17, v23, v27 // (op 260) 


v_cvt_f16_f32 v20, v14 // (op 261) 
v_cvt_f16_f32 v21, v15 // (op 261) 
v_cvt_f16_f32 v22, v16 // (op 261) 
v_cvt_f16_f32 v23, v17 // (op 261) 


v_pack_b32_f16 v14, v20, v21 // (op 262) 
v_pack_b32_f16 v15, v22, v23 // (op 262) 

v_mov_b32 v0, v5 // (op 262) reg expression

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 262) Store value





s_lshl_b32 s15, s4, 6 // (op 2333) 
v_add_u32 v0, v4, s15 // (op 2333) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 2333) Load value

v_accvgpr_read v20, a4 // (op 2329) 
v_accvgpr_read v21, a5 // (op 2329) 
v_accvgpr_read v22, a6 // (op 2329) 
v_accvgpr_read v23, a7 // (op 2329) 
v_mul_f32 v24, v19, v20 // (op 2329) 
v_mul_f32 v25, v19, v21 // (op 2329) 
v_mul_f32 v26, v19, v22 // (op 2329) 
v_mul_f32 v27, v19, v23 // (op 2329) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 2334) 
v_lshrrev_b32 v17, 16, v14 // (op 2334) 
v_cvt_f32_f16 v17, v17 // (op 2334) 
v_mul_f32 v20, v18, v16 // (op 2334) 
v_mul_f32 v21, v18, v17 // (op 2334) 
v_cvt_f32_f16 v16, v15 // (op 2334) 
v_lshrrev_b32 v17, 16, v15 // (op 2334) 
v_cvt_f32_f16 v17, v17 // (op 2334) 
v_mul_f32 v22, v18, v16 // (op 2334) 
v_mul_f32 v23, v18, v17 // (op 2334) 


v_add_f32 v14, v20, v24 // (op 2330) 
v_add_f32 v15, v21, v25 // (op 2330) 
v_add_f32 v16, v22, v26 // (op 2330) 
v_add_f32 v17, v23, v27 // (op 2330) 


v_cvt_f16_f32 v20, v14 // (op 2331) 
v_cvt_f16_f32 v21, v15 // (op 2331) 
v_cvt_f16_f32 v22, v16 // (op 2331) 
v_cvt_f16_f32 v23, v17 // (op 2331) 


v_pack_b32_f16 v14, v20, v21 // (op 2332) 
v_pack_b32_f16 v15, v22, v23 // (op 2332) 

s_lshl_b32 s15, s24, 6 // (op 2332) 
v_add_u32 v0, v5, s15 // (op 2332) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 2332) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_0_beta_238:
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
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_DPStreamTileLoop_449 // (op 449) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_DPStreamTileLoop_449 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_449:

















































s_endpgm  // End of RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2
.LRR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_end:
.size RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2, .LRR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_end-RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2
  .amdhsa_next_free_vgpr 160
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 50688 // lds bytes
  .amdhsa_accum_offset 128
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
  ProblemType:
    OperationType: GEMM
    DataType: B8
    DataTypeA: B8
    DataTypeB: B8
    DestDataType: h
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
            { type: int32, semantic: RR_SKMagicShiftsNumTiles1 },
            { type: int32, semantic: RR_SKMagicShiftsIters },
            { type: uint32, semantic: RR_SKMagicMultipleNumTiles1 },
            { type: uint32, semantic: RR_SKMagicMultipleIters },
            { type: uint32, semantic: RR_SKNumTiles1 },
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
    macrotile: [128, 64, 128]
    threads: [256, 1, 1]
    grid: [RocRollerStreamK, One, One]
    workspaceType: RocRollerStreamK
    occupancy: 3
    workspaceSizePerElemC: 4
    workspaceSizePerElemBias: 0
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2
    .symbol: RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 50688
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 88
    .vgpr_count: 126
    .agpr_count: 32
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: LogicalShiftR, lhs: {type: Subtract, lhs: {type: ShiftLAdd, lhs: {type: Kernel Argument, name: numWGs, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: Convert, arg: {type: CommandArgument, size: 4, offset: 248, name: numWGs, variableType: {dataType: UInt32, pointerType: Value}, direction: read_only}, dataType: UInt32}, offset: 0, size: 4}, r1hs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}, r2hs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_0_size_1_22
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_stride_1_18
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 16
        .expression: {type: CommandArgument, size: 4, offset: 204, name: user_Float_Value_14, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 20
        .expression: {type: CommandArgument, size: 4, offset: 200, name: user_Float_Value_12, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: numSKTilesPerWG
        .size: 4
        .offset: 24
        .expression: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Conditional, lhs: {type: GreaterThan, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, dataType: UInt32}, rhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, dataType: UInt32}}, rhs: {type: Subtract, lhs: {type: CommandArgument, size: 4, offset: 248, name: numWGs, variableType: {dataType: UInt32, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, r1hs: {type: Multiply, lhs: {type: Add, lhs: {type: Modulo, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, dataType: UInt32}, rhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, dataType: UInt32}}, rhs: {type: CommandArgument, size: 4, offset: 248, name: numWGs, variableType: {dataType: UInt32, pointerType: Value}, direction: read_only}}, rhs: {type: CommandArgument, size: 4, offset: 248, name: numWGs, variableType: {dataType: UInt32, pointerType: Value}, direction: read_only}}, rhs: {type: Convert, arg: {type: Convert, arg: {type: Divide, lhs: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, dataType: UInt32}, dataType: UInt32}}, r2hs: {type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, dataType: UInt32}, rhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, dataType: UInt32}}, rhs: {type: Convert, arg: {type: Convert, arg: {type: Divide, lhs: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, dataType: UInt32}, dataType: UInt32}}}, rhs: {type: CommandArgument, size: 4, offset: 248, name: numWGs, variableType: {dataType: UInt32, pointerType: Value}, direction: read_only}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: CommandArgument, size: 4, offset: 248, name: numWGs, variableType: {dataType: UInt32, pointerType: Value}, direction: read_only}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: numSKTiles
        .size: 4
        .offset: 28
        .expression: {type: Conditional, lhs: {type: GreaterThan, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, dataType: UInt32}, rhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, dataType: UInt32}}, rhs: {type: Subtract, lhs: {type: CommandArgument, size: 4, offset: 248, name: numWGs, variableType: {dataType: UInt32, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, r1hs: {type: Multiply, lhs: {type: Add, lhs: {type: Modulo, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, dataType: UInt32}, rhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, dataType: UInt32}}, rhs: {type: CommandArgument, size: 4, offset: 248, name: numWGs, variableType: {dataType: UInt32, pointerType: Value}, direction: read_only}}, rhs: {type: CommandArgument, size: 4, offset: 248, name: numWGs, variableType: {dataType: UInt32, pointerType: Value}, direction: read_only}}, rhs: {type: Convert, arg: {type: Convert, arg: {type: Divide, lhs: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, dataType: UInt32}, dataType: UInt32}}, r2hs: {type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, dataType: UInt32}, rhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, dataType: UInt32}}, rhs: {type: Convert, arg: {type: Convert, arg: {type: Divide, lhs: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, dataType: UInt32}, dataType: UInt32}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShifts_31
        .size: 4
        .offset: 32
        .expression: {type: MagicShifts, arg: {type: Kernel Argument, name: numTiles1, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, dataType: UInt32}, offset: 8, size: 4}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShifts_29
        .size: 4
        .offset: 36
        .expression: {type: MagicShifts, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Kernel Argument, name: Tensor_0_size_1_22, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 136, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: UInt32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 40
        .expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: numTiles1, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, dataType: UInt32}, offset: 8, size: 4}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 44
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Kernel Argument, name: Tensor_0_size_1_22, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 136, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: UInt32}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: numTiles1
        .size: 4
        .offset: 48
        .expression: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, dataType: UInt32}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: numWGs
        .size: 4
        .offset: 52
        .expression: {type: Convert, arg: {type: CommandArgument, size: 4, offset: 248, name: numWGs, variableType: {dataType: UInt32, pointerType: Value}, direction: read_only}, dataType: UInt32}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_12
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_24
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: SCRATCH_None
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 256, name: SCRATCH_None, variableType: {dataType: UInt32, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: UInt32, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_8
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_9
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_stride_1_11
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 128
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_14
        .size: 8
        .offset: 136
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 144
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_4_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_16
        .size: 8
        .offset: 152
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_7_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 160
        .expression: {type: CommandArgument, size: 8, offset: 120, name: Tensor_7_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: SCRATCH_ZeroedBeforeAndAfter
        .size: 8
        .offset: 168
        .expression: {type: CommandArgument, size: 8, offset: 264, name: SCRATCH_ZeroedBeforeAndAfter, variableType: {dataType: UInt32, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: UInt32, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: numDPTilesPerWG
        .size: 4
        .offset: 176
        .expression: {type: Conditional, lhs: {type: GreaterThan, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, dataType: UInt32}, rhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, dataType: UInt32}}, rhs: {type: Subtract, lhs: {type: CommandArgument, size: 4, offset: 248, name: numWGs, variableType: {dataType: UInt32, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, r1hs: {type: Multiply, lhs: {type: Subtract, lhs: {type: Divide, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, dataType: UInt32}, rhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, dataType: UInt32}}, rhs: {type: CommandArgument, size: 4, offset: 248, name: numWGs, variableType: {dataType: UInt32, pointerType: Value}, direction: read_only}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: Convert, arg: {type: Convert, arg: {type: Divide, lhs: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, dataType: UInt32}, dataType: UInt32}}, r2hs: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: numDPTiles
        .size: 4
        .offset: 180
        .expression: {type: Conditional, lhs: {type: GreaterThan, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, dataType: UInt32}, rhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, dataType: UInt32}}, rhs: {type: Subtract, lhs: {type: CommandArgument, size: 4, offset: 248, name: numWGs, variableType: {dataType: UInt32, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, r1hs: {type: Multiply, lhs: {type: Multiply, lhs: {type: Subtract, lhs: {type: Divide, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, dataType: UInt32}, rhs: {type: Convert, arg: {type: Divide, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, dataType: UInt32}}, rhs: {type: CommandArgument, size: 4, offset: 248, name: numWGs, variableType: {dataType: UInt32, pointerType: Value}, direction: read_only}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: CommandArgument, size: 4, offset: 248, name: numWGs, variableType: {dataType: UInt32, pointerType: Value}, direction: read_only}}, rhs: {type: Convert, arg: {type: Convert, arg: {type: Divide, lhs: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, dataType: UInt32}, dataType: UInt32}}, r2hs: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
...

.end_amdgpu_metadata

