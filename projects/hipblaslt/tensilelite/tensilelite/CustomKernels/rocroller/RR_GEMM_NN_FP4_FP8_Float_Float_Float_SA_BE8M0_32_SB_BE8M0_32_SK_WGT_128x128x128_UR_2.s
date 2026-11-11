// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
.p2align 8
.type RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2,@function
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 18427) 
v_and_b32 v3, 48, v1 // (op 18427) 
v_sub_u32 v4, v0, v3 // (op 18427) 

v_lshrrev_b32 v3, 4, v0 // (op 18427) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 18427) 


v_lshrrev_b32 v3, 7, v1 // (op 18427) 
v_lshlrev_b32 v4, 4, v3 // (op 18427) 

v_lshl_add_u32 v3, v0, 7, v4 // (op 18427) 


v_lshlrev_b32 v0, 2, v3 // (op 18427) 

v_lshrrev_b32 v3, 3, v0 // (op 18427) 

v_lshrrev_b32 v0, 6, v1 // (op 18502) 
v_and_b32 v4, 1, v0 // (op 18502) 

v_and_b32 v0, 15, v1 // (op 18502) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 18502) 


v_and_b32 v0, 48, v1 // (op 18502) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 18502) 


v_lshrrev_b32 v0, 7, v1 // (op 18578) 
v_and_b32 v5, 15, v1 // (op 18578) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 18578) 


v_and_b32 v0, 63, v1 // (op 18578) 
v_lshrrev_b32 v5, 4, v0 // (op 18578) 

v_lshl_add_u32 v0, v5, 7, v6 // (op 18578) 


v_lshrrev_b32 v5, 6, v1 // (op 18654) 
v_and_b32 v6, 1, v5 // (op 18654) 

v_and_b32 v5, 15, v1 // (op 18654) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 18654) 


v_and_b32 v5, 63, v1 // (op 18654) 
v_lshrrev_b32 v6, 4, v5 // (op 18654) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 18654) 


v_and_b32 v6, 63, v1 // (op 18781) 
v_and_b32 v7, 48, v1 // (op 18781) 
v_sub_u32 v8, v6, v7 // (op 18781) 

v_lshrrev_b32 v7, 4, v6 // (op 18781) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 18781) 


v_lshrrev_b32 v7, 7, v1 // (op 18781) 
v_lshlrev_b32 v8, 4, v7 // (op 18781) 

v_lshl_add_u32 v7, v6, 7, v8 // (op 18781) 


v_lshlrev_b32 v6, 2, v7 // (op 18781) 

v_lshrrev_b32 v7, 3, v6 // (op 18781) 

v_lshrrev_b32 v6, 6, v1 // (op 18856) 
v_and_b32 v8, 1, v6 // (op 18856) 

v_and_b32 v6, 15, v1 // (op 18856) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 18856) 


v_and_b32 v6, 48, v1 // (op 18856) 
v_lshl_add_u32 v8, v9, 7, v6 // (op 18856) 


v_lshrrev_b32 v6, 7, v1 // (op 18932) 
v_and_b32 v9, 15, v1 // (op 18932) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 18932) 


v_and_b32 v6, 63, v1 // (op 18932) 
v_lshrrev_b32 v9, 4, v6 // (op 18932) 

v_lshl_add_u32 v6, v9, 7, v10 // (op 18932) 


v_lshrrev_b32 v9, 6, v1 // (op 19008) 
v_and_b32 v10, 1, v9 // (op 19008) 

v_and_b32 v9, 15, v1 // (op 19008) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 19008) 


v_and_b32 v9, 63, v1 // (op 19008) 
v_lshrrev_b32 v10, 4, v9 // (op 19008) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 19008) 


v_and_b32 v10, 63, v1 // (op 19122) 
v_and_b32 v11, 48, v1 // (op 19122) 
v_sub_u32 v12, v10, v11 // (op 19122) 

v_lshrrev_b32 v11, 4, v10 // (op 19122) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 19122) 


v_lshrrev_b32 v11, 7, v1 // (op 19122) 
v_lshlrev_b32 v12, 4, v11 // (op 19122) 

v_lshl_add_u32 v11, v10, 7, v12 // (op 19122) 


v_lshlrev_b32 v10, 2, v11 // (op 19122) 

v_lshrrev_b32 v11, 3, v10 // (op 19122) 

v_lshrrev_b32 v10, 6, v1 // (op 19166) 
v_and_b32 v12, 1, v10 // (op 19166) 

v_and_b32 v10, 15, v1 // (op 19166) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 19166) 


v_and_b32 v10, 48, v1 // (op 19166) 
v_lshl_add_u32 v12, v13, 7, v10 // (op 19166) 


v_lshrrev_b32 v10, 7, v1 // (op 19195) 
v_and_b32 v13, 15, v1 // (op 19195) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 19195) 


v_and_b32 v10, 63, v1 // (op 19195) 
v_lshrrev_b32 v13, 4, v10 // (op 19195) 

v_lshl_add_u32 v10, v13, 7, v14 // (op 19195) 


v_lshrrev_b32 v13, 6, v1 // (op 19224) 
v_and_b32 v14, 1, v13 // (op 19224) 

v_and_b32 v13, 15, v1 // (op 19224) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 19224) 


v_and_b32 v13, 63, v1 // (op 19224) 
v_lshrrev_b32 v14, 4, v13 // (op 19224) 

v_lshl_add_u32 v13, v15, 2, v14 // (op 19224) 


v_and_b32 v14, 63, v1 // (op 19268) 
v_and_b32 v15, 48, v1 // (op 19268) 
v_sub_u32 v16, v14, v15 // (op 19268) 

v_lshrrev_b32 v15, 4, v14 // (op 19268) 

v_lshl_add_u32 v14, v15, 5, v16 // (op 19268) 


v_lshrrev_b32 v15, 7, v1 // (op 19268) 
v_lshlrev_b32 v16, 4, v15 // (op 19268) 

v_lshl_add_u32 v15, v14, 7, v16 // (op 19268) 


v_lshlrev_b32 v14, 2, v15 // (op 19268) 

v_lshrrev_b32 v15, 3, v14 // (op 19268) 

v_lshrrev_b32 v14, 6, v1 // (op 19312) 
v_and_b32 v16, 1, v14 // (op 19312) 

v_and_b32 v14, 15, v1 // (op 19312) 
v_lshl_add_u32 v17, v16, 4, v14 // (op 19312) 


v_and_b32 v14, 48, v1 // (op 19312) 
v_lshl_add_u32 v16, v17, 7, v14 // (op 19312) 


v_lshrrev_b32 v14, 7, v1 // (op 19341) 
v_and_b32 v17, 15, v1 // (op 19341) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 19341) 


v_and_b32 v14, 63, v1 // (op 19341) 
v_lshrrev_b32 v17, 4, v14 // (op 19341) 

v_lshl_add_u32 v14, v17, 7, v18 // (op 19341) 


v_lshrrev_b32 v17, 6, v1 // (op 19370) 
v_and_b32 v18, 1, v17 // (op 19370) 

v_and_b32 v17, 15, v1 // (op 19370) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 19370) 


v_and_b32 v17, 63, v1 // (op 19370) 
v_lshrrev_b32 v18, 4, v17 // (op 19370) 

v_lshl_add_u32 v17, v19, 2, v18 // (op 19370) 


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
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_286 // (op 286) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_286 if false)
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_SKStreamTileLoop_286:
v_lshrrev_b32 v20, 7, v1 // (op 18338) 
s_mul_i32 s52, s16, s8 // (op 18338) 
s_add_u32 s52, s52, s0 // (op 18338) 
v_and_b32 v21, 63, v1 // (op 18338) 
s_mul_hi_u32 s53, s52, s13 // (op 18338) 
v_lshrrev_b32 v22, 4, v21 // (op 18338) 
s_sub_u32 s54, s52, s53 // (op 18338) 
v_lshlrev_b32 v23, 2, v22 // (op 18338) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 18338) 


s_lshr_b32 s55, s54, 1 // (op 18338) 

s_add_u32 s54, s55, s53 // (op 18338) 


v_lshrrev_b32 v20, 2, v21 // (op 18338) 

s_lshr_b32 s53, s54, s11 // (op 18338) 

v_and_b32 v21, 3, v20 // (op 18338) 

s_and_b32 s54, s11, 2147483648 // (op 18338) 
v_and_b32 v20, 3, v1 // (op 18338) 
s_cmp_eq_i32 s54, 0 // (op 18338) 
s_mov_b32 s55, scc // (op 18338) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 18338) 


s_cmp_eq_i32 s55, 0 // (op 18338) 
s_cselect_b32 s54, s52, s53 // (op 18338) 



v_lshrrev_b32 v20, 6, v1 // (op 18338) 
s_mul_hi_u32 s52, s54, s12 // (op 18338) 
v_and_b32 v21, 1, v20 // (op 18338) 

s_sub_u32 s53, s54, s52 // (op 18338) 
v_lshl_add_u32 v20, v21, 4, v23 // (op 18338) 


s_lshr_b32 s55, s53, 1 // (op 18338) 

s_add_u32 s53, s55, s52 // (op 18338) 


s_lshr_b32 s52, s53, s10 // (op 18338) 

s_and_b32 s53, s10, 2147483648 // (op 18338) 
s_cmp_eq_i32 s53, 0 // (op 18338) 
s_mov_b32 s55, scc // (op 18338) 

s_cmp_eq_i32 s55, 0 // (op 18338) 
s_cselect_b32 s53, s54, s52 // (op 18338) 


s_mul_i32 s52, s53, s14 // (op 18338) 
v_lshl_add_u32 v21, s53, 7, v22 // (op 18338) 


s_sub_u32 s53, s54, s52 // (op 18338) 


v_lshl_add_u32 v22, s53, 7, v20 // (op 18338) 


v_mul_lo_u32 v20, v22, s4 // (op 18338) 

v_add_lshl_u32 v22, v21, v20, 2 // (op 18338) 


s_mul_i32 s53, s4, s23 // (op 18340) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 18340) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 18340) least significant
s_mul_hi_u32 s55, s4, s22 // (op 18340) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18340) most significant: sum
s_add_u32 s53, s53, s55 // (op 18340) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 18340) 

s_mov_b32 s56, s20 // (op 18340) reg expression
s_mov_b32 s57, s21 // (op 18340) reg expression
s_mov_b32 s58, s54 // (op 18340) reg expression
s_mov_b32 s59, 131072 // (op 18340) call()

v_and_b32 v20, 63, v1 // (op 18444) 
v_and_b32 v21, 48, v1 // (op 18444) 
v_sub_u32 v23, v20, v21 // (op 18444) 

v_lshrrev_b32 v21, 4, v20 // (op 18444) 

v_lshl_add_u32 v20, v21, 5, v23 // (op 18444) 


v_lshrrev_b32 v21, 7, v1 // (op 18444) 
v_lshlrev_b32 v23, 4, v21 // (op 18444) 

v_lshl_add_u32 v21, v20, 7, v23 // (op 18444) 


v_lshlrev_b32 v20, 2, v21 // (op 18444) 

v_lshrrev_b32 v21, 3, v20 // (op 18444) 

v_lshrrev_b32 v20, 6, v1 // (op 18519) 
v_and_b32 v23, 1, v20 // (op 18519) 

v_and_b32 v20, 15, v1 // (op 18519) 
v_lshl_add_u32 v24, v23, 4, v20 // (op 18519) 


v_and_b32 v20, 48, v1 // (op 18519) 
v_lshl_add_u32 v23, v24, 7, v20 // (op 18519) 


v_lshrrev_b32 v20, 7, v1 // (op 18595) 
v_and_b32 v24, 15, v1 // (op 18595) 
v_lshl_add_u32 v25, v20, 4, v24 // (op 18595) 


v_and_b32 v20, 63, v1 // (op 18595) 
v_lshrrev_b32 v24, 4, v20 // (op 18595) 

v_lshl_add_u32 v20, v24, 7, v25 // (op 18595) 


v_lshrrev_b32 v24, 6, v1 // (op 18671) 
v_and_b32 v25, 1, v24 // (op 18671) 

v_and_b32 v24, 15, v1 // (op 18671) 
v_lshl_add_u32 v26, v25, 4, v24 // (op 18671) 


v_and_b32 v24, 63, v1 // (op 18671) 
v_lshrrev_b32 v25, 4, v24 // (op 18671) 

v_lshl_add_u32 v24, v26, 2, v25 // (op 18671) 


v_lshrrev_b32 v25, 7, v1 // (op 18694) 
s_mul_i32 s52, s16, s8 // (op 18694) 
s_add_u32 s52, s52, s0 // (op 18694) 
v_and_b32 v26, 63, v1 // (op 18694) 
s_mul_hi_u32 s53, s52, s13 // (op 18694) 
v_lshrrev_b32 v27, 4, v26 // (op 18694) 
s_sub_u32 s54, s52, s53 // (op 18694) 
v_lshlrev_b32 v28, 2, v27 // (op 18694) 

s_lshr_b32 s55, s54, 1 // (op 18694) 

v_lshl_add_u32 v27, v25, 4, v28 // (op 18694) 


s_add_u32 s54, s55, s53 // (op 18694) 


v_lshrrev_b32 v25, 2, v26 // (op 18694) 

s_lshr_b32 s53, s54, s11 // (op 18694) 

v_and_b32 v26, 3, v25 // (op 18694) 

s_and_b32 s54, s11, 2147483648 // (op 18694) 
v_and_b32 v25, 3, v1 // (op 18694) 
s_cmp_eq_i32 s54, 0 // (op 18694) 
s_mov_b32 s55, scc // (op 18694) 

v_lshl_add_u32 v28, v26, 2, v25 // (op 18694) 


s_cmp_eq_i32 s55, 0 // (op 18694) 
s_cselect_b32 s54, s52, s53 // (op 18694) 



v_lshrrev_b32 v25, 6, v1 // (op 18694) 
s_mul_hi_u32 s55, s54, s12 // (op 18694) 
v_and_b32 v26, 1, v25 // (op 18694) 

s_sub_u32 s52, s54, s55 // (op 18694) 
v_lshl_add_u32 v25, v26, 4, v28 // (op 18694) 


s_lshr_b32 s53, s52, 1 // (op 18694) 

s_add_u32 s52, s53, s55 // (op 18694) 


s_lshr_b32 s53, s52, s10 // (op 18694) 

s_and_b32 s52, s10, 2147483648 // (op 18694) 
s_cmp_eq_i32 s52, 0 // (op 18694) 
s_mov_b32 s55, scc // (op 18694) 

s_cmp_eq_i32 s55, 0 // (op 18694) 
s_cselect_b32 s52, s54, s53 // (op 18694) 


s_mul_i32 s53, s52, s14 // (op 18694) 
v_lshl_add_u32 v26, s52, 7, v27 // (op 18694) 


s_sub_u32 s52, s54, s53 // (op 18694) 


v_lshl_add_u32 v27, s52, 7, v25 // (op 18694) 


v_mul_lo_u32 v25, v27, s24 // (op 18694) 

v_add_lshl_u32 v27, v26, v25, 2 // (op 18694) 


s_mul_i32 s53, s24, s23 // (op 18696) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 18696) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 18696) least significant
s_mul_hi_u32 s55, s24, s22 // (op 18696) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18696) most significant: sum
s_add_u32 s53, s53, s55 // (op 18696) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 18696) 

s_mov_b32 s60, s26 // (op 18696) reg expression
s_mov_b32 s61, s27 // (op 18696) reg expression
s_mov_b32 s62, s54 // (op 18696) reg expression
s_mov_b32 s63, 131072 // (op 18696) call()

v_lshrrev_b32 v25, 5, v1 // (op 19048) 
v_lshlrev_b32 v26, 2, v25 // (op 19048) 

v_lshl_add_u32 v25, s16, 7, v26 // (op 19048) 

v_and_b32 v26, -32, v1 // (op 19048) 
v_sub_u32 v28, v1, v26 // (op 19048) 

v_lshl_add_u32 v26, v25, 7, v28 // (op 19048) 


v_lshlrev_b32 v25, 2, v26 // (op 19048) 

s_lshl_b32 s52, s15, 16 // (op 19050) 
s_mov_b32 s64, s28 // (op 19050) reg expression
s_mov_b32 s65, s29 // (op 19050) reg expression
s_mov_b32 s66, s52 // (op 19050) reg expression
s_mov_b32 s67, 131072 // (op 19050) call()

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
v_accvgpr_write a4, 0.00000 // (op 4734) call()
v_accvgpr_write a5, 0.00000 // (op 4734) call()
v_accvgpr_write a6, 0.00000 // (op 4734) call()
v_accvgpr_write a7, 0.00000 // (op 4734) call()
v_accvgpr_write a8, 0.00000 // (op 4910) call()
v_accvgpr_write a9, 0.00000 // (op 4910) call()
v_accvgpr_write a10, 0.00000 // (op 4910) call()
v_accvgpr_write a11, 0.00000 // (op 4910) call()
v_accvgpr_write a12, 0.00000 // (op 5086) call()
v_accvgpr_write a13, 0.00000 // (op 5086) call()
v_accvgpr_write a14, 0.00000 // (op 5086) call()
v_accvgpr_write a15, 0.00000 // (op 5086) call()
v_accvgpr_write a16, 0.00000 // (op 5350) call()
v_accvgpr_write a17, 0.00000 // (op 5350) call()
v_accvgpr_write a18, 0.00000 // (op 5350) call()
v_accvgpr_write a19, 0.00000 // (op 5350) call()
v_accvgpr_write a20, 0.00000 // (op 5659) call()
v_accvgpr_write a21, 0.00000 // (op 5659) call()
v_accvgpr_write a22, 0.00000 // (op 5659) call()
v_accvgpr_write a23, 0.00000 // (op 5659) call()
v_accvgpr_write a24, 0.00000 // (op 5660) call()
v_accvgpr_write a25, 0.00000 // (op 5660) call()
v_accvgpr_write a26, 0.00000 // (op 5660) call()
v_accvgpr_write a27, 0.00000 // (op 5660) call()
v_accvgpr_write a28, 0.00000 // (op 5661) call()
v_accvgpr_write a29, 0.00000 // (op 5661) call()
v_accvgpr_write a30, 0.00000 // (op 5661) call()
v_accvgpr_write a31, 0.00000 // (op 5661) call()
v_accvgpr_write a32, 0.00000 // (op 6120) call()
v_accvgpr_write a33, 0.00000 // (op 6120) call()
v_accvgpr_write a34, 0.00000 // (op 6120) call()
v_accvgpr_write a35, 0.00000 // (op 6120) call()
v_accvgpr_write a36, 0.00000 // (op 6429) call()
v_accvgpr_write a37, 0.00000 // (op 6429) call()
v_accvgpr_write a38, 0.00000 // (op 6429) call()
v_accvgpr_write a39, 0.00000 // (op 6429) call()
v_accvgpr_write a40, 0.00000 // (op 6430) call()
v_accvgpr_write a41, 0.00000 // (op 6430) call()
v_accvgpr_write a42, 0.00000 // (op 6430) call()
v_accvgpr_write a43, 0.00000 // (op 6430) call()
v_accvgpr_write a44, 0.00000 // (op 6431) call()
v_accvgpr_write a45, 0.00000 // (op 6431) call()
v_accvgpr_write a46, 0.00000 // (op 6431) call()
v_accvgpr_write a47, 0.00000 // (op 6431) call()
v_accvgpr_write a48, 0.00000 // (op 6890) call()
v_accvgpr_write a49, 0.00000 // (op 6890) call()
v_accvgpr_write a50, 0.00000 // (op 6890) call()
v_accvgpr_write a51, 0.00000 // (op 6890) call()
v_accvgpr_write a52, 0.00000 // (op 7199) call()
v_accvgpr_write a53, 0.00000 // (op 7199) call()
v_accvgpr_write a54, 0.00000 // (op 7199) call()
v_accvgpr_write a55, 0.00000 // (op 7199) call()
v_accvgpr_write a56, 0.00000 // (op 7200) call()
v_accvgpr_write a57, 0.00000 // (op 7200) call()
v_accvgpr_write a58, 0.00000 // (op 7200) call()
v_accvgpr_write a59, 0.00000 // (op 7200) call()
v_accvgpr_write a60, 0.00000 // (op 7201) call()
v_accvgpr_write a61, 0.00000 // (op 7201) call()
v_accvgpr_write a62, 0.00000 // (op 7201) call()
v_accvgpr_write a63, 0.00000 // (op 7201) call()
v_and_b32 v26, -4, v1 // (op 17905) 
s_mul_i32 s52, s16, s8 // (op 17905) 
s_add_u32 s52, s52, s0 // (op 17905) 
v_sub_u32 v28, v1, v26 // (op 17905) 

s_mul_hi_u32 s53, s52, s13 // (op 17905) 
v_lshlrev_b32 v26, 5, v28 // (op 17905) 

s_sub_u32 s54, s52, s53 // (op 17905) 
v_lshrrev_b32 v28, 2, v1 // (op 17905) 
s_lshr_b32 s55, s54, 1 // (op 17905) 

s_add_u32 s54, s55, s53 // (op 17905) 


s_lshr_b32 s53, s54, s11 // (op 17905) 

s_and_b32 s54, s11, 2147483648 // (op 17905) 
s_cmp_eq_i32 s54, 0 // (op 17905) 
s_mov_b32 s55, scc // (op 17905) 

s_cmp_eq_i32 s55, 0 // (op 17905) 
s_cselect_b32 s54, s52, s53 // (op 17905) 


s_mul_hi_u32 s53, s54, s12 // (op 17905) 
s_sub_u32 s55, s54, s53 // (op 17905) 
s_lshr_b32 s68, s55, 1 // (op 17905) 

s_add_u32 s55, s68, s53 // (op 17905) 


s_lshr_b32 s53, s55, s10 // (op 17905) 

s_and_b32 s55, s10, 2147483648 // (op 17905) 
s_cmp_eq_i32 s55, 0 // (op 17905) 
s_mov_b32 s68, scc // (op 17905) 

s_cmp_eq_i32 s68, 0 // (op 17905) 
s_cselect_b32 s55, s54, s53 // (op 17905) 


v_lshl_add_u32 v29, s55, 7, v26 // (op 17905) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 17905) 
v_mov_b32 v31, 0 // (op 17905) convert
v_mov_b32 v30, v29 // (op 17905) convert

s_mul_i32 s53, s54, s68 // (op 17905) 


s_sub_u32 s54, s52, s53 // (op 17905) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 17905) 


v_mul_lo_u32 v29, v26, s31 // (op 17905) most significant: low of low * high
v_mul_hi_u32 v32, v26, s30 // (op 17905) most significant: high of low * low
v_mul_lo_u32 v28, v26, s30 // (op 17905) least significant: low of low * low
v_add_u32 v29, v29, v32 // (op 17905) most significant: sum


v_add_co_u32 v32, s[52:53], v30, v28 // (op 17905) least significant half
v_addc_co_u32 v33, s[52:53], v31, v29, s[52:53] // (op 17905) most significant half

v_lshlrev_b64 v[32:33], 2, v[32:33] // (op 17905) 


v_lshrrev_b64 v[28:29], 3, v[32:33] // (op 17905) 

s_mul_i32 s53, s30, s3 // (op 17907) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 17907) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 17907) least significant
s_mul_hi_u32 s55, s30, s2 // (op 17907) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 17907) most significant: sum
s_add_u32 s53, s53, s55 // (op 17907) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 17907) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 17907) 

s_mov_b32 s68, s32 // (op 17907) reg expression
s_mov_b32 s69, s33 // (op 17907) reg expression
s_mov_b32 s70, s52 // (op 17907) reg expression
s_mov_b32 s71, 131072 // (op 17907) call()

v_and_b32 v26, -4, v1 // (op 17924) 
s_mul_i32 s52, s16, s8 // (op 17924) 
s_add_u32 s52, s52, s0 // (op 17924) 
v_sub_u32 v30, v1, v26 // (op 17924) 

s_mul_hi_u32 s53, s52, s13 // (op 17924) 
v_lshlrev_b32 v26, 5, v30 // (op 17924) 

s_sub_u32 s54, s52, s53 // (op 17924) 
v_lshrrev_b32 v30, 2, v1 // (op 17924) 
s_lshr_b32 s55, s54, 1 // (op 17924) 

s_add_u32 s54, s55, s53 // (op 17924) 


s_lshr_b32 s53, s54, s11 // (op 17924) 

s_and_b32 s54, s11, 2147483648 // (op 17924) 
s_cmp_eq_i32 s54, 0 // (op 17924) 
s_mov_b32 s55, scc // (op 17924) 

s_cmp_eq_i32 s55, 0 // (op 17924) 
s_cselect_b32 s54, s52, s53 // (op 17924) 


s_mul_hi_u32 s53, s54, s12 // (op 17924) 
s_sub_u32 s55, s54, s53 // (op 17924) 
s_lshr_b32 s72, s55, 1 // (op 17924) 

s_add_u32 s55, s72, s53 // (op 17924) 


s_lshr_b32 s53, s55, s10 // (op 17924) 

s_and_b32 s55, s10, 2147483648 // (op 17924) 
s_cmp_eq_i32 s55, 0 // (op 17924) 
s_mov_b32 s72, scc // (op 17924) 

s_cmp_eq_i32 s72, 0 // (op 17924) 
s_cselect_b32 s55, s54, s53 // (op 17924) 


v_lshl_add_u32 v31, s55, 7, v26 // (op 17924) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 17924) 
v_mov_b32 v33, 0 // (op 17924) convert
v_mov_b32 v32, v31 // (op 17924) convert

s_mul_i32 s53, s54, s72 // (op 17924) 


s_sub_u32 s54, s52, s53 // (op 17924) 


v_lshl_add_u32 v26, s54, 7, v30 // (op 17924) 


v_mul_lo_u32 v31, v26, s31 // (op 17924) most significant: low of low * high
v_mul_hi_u32 v34, v26, s30 // (op 17924) most significant: high of low * low
v_mul_lo_u32 v30, v26, s30 // (op 17924) least significant: low of low * low
v_add_u32 v31, v31, v34 // (op 17924) most significant: sum


v_add_co_u32 v34, s[52:53], v32, v30 // (op 17924) least significant half
v_addc_co_u32 v35, s[52:53], v33, v31, s[52:53] // (op 17924) most significant half

v_lshlrev_b64 v[34:35], 2, v[34:35] // (op 17924) 


v_lshrrev_b64 v[30:31], 3, v[34:35] // (op 17924) 

s_mul_i32 s53, s30, s3 // (op 17926) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 17926) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 17926) least significant
s_mul_hi_u32 s55, s30, s2 // (op 17926) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 17926) most significant: sum
s_add_u32 s53, s53, s55 // (op 17926) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 17926) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 17926) 

s_mov_b32 s68, s32 // (op 17926) reg expression
s_mov_b32 s69, s33 // (op 17926) reg expression
s_mov_b32 s70, s52 // (op 17926) reg expression
s_mov_b32 s71, 131072 // (op 17926) call()

v_lshrrev_b32 v26, 3, v1 // (op 18033) 
s_mul_i32 s52, s16, s8 // (op 18033) 
s_add_u32 s52, s52, s0 // (op 18033) 
v_and_b32 v32, -8, v1 // (op 18033) 
s_mul_hi_u32 s53, s52, s13 // (op 18033) 
v_sub_u32 v33, v1, v32 // (op 18033) 

s_sub_u32 s54, s52, s53 // (op 18033) 
v_lshlrev_b32 v32, 4, v33 // (op 18033) 

s_lshr_b32 s55, s54, 1 // (op 18033) 

s_add_u32 s54, s55, s53 // (op 18033) 


s_lshr_b32 s53, s54, s11 // (op 18033) 

s_and_b32 s54, s11, 2147483648 // (op 18033) 
s_cmp_eq_i32 s54, 0 // (op 18033) 
s_mov_b32 s55, scc // (op 18033) 

s_cmp_eq_i32 s55, 0 // (op 18033) 
s_cselect_b32 s54, s52, s53 // (op 18033) 


s_mul_hi_u32 s53, s54, s12 // (op 18033) 
s_sub_u32 s55, s54, s53 // (op 18033) 
s_lshr_b32 s72, s55, 1 // (op 18033) 

s_add_u32 s55, s72, s53 // (op 18033) 


s_lshr_b32 s53, s55, s10 // (op 18033) 

s_and_b32 s55, s10, 2147483648 // (op 18033) 
s_cmp_eq_i32 s55, 0 // (op 18033) 
s_mov_b32 s72, scc // (op 18033) 

s_cmp_eq_i32 s72, 0 // (op 18033) 
s_cselect_b32 s55, s54, s53 // (op 18033) 


s_mul_i32 s53, s55, s14 // (op 18033) 

s_sub_u32 s55, s54, s53 // (op 18033) 

v_lshl_add_u32 v33, s55, 7, v26 // (op 18033) 


v_mul_lo_u32 v35, v33, s35 // (op 18033) most significant: low of low * high
v_mul_hi_u32 v26, v33, s34 // (op 18033) most significant: high of low * low
v_mul_lo_u32 v34, v33, s34 // (op 18033) least significant: low of low * low
v_add_u32 v35, v35, v26 // (op 18033) most significant: sum


s_ashr_i64 s[72:73], s[2:3], 7 // (op 18033) 
s_mul_i32 s53, s54, s72 // (op 18033) 


s_sub_u32 s54, s52, s53 // (op 18033) 


v_lshl_add_u32 v26, s54, 7, v32 // (op 18033) 


v_mov_b32 v33, 0 // (op 18033) convert
v_mov_b32 v32, v26 // (op 18033) convert

v_add_co_u32 v36, s[52:53], v32, v34 // (op 18033) least significant half
v_addc_co_u32 v37, s[52:53], v33, v35, s[52:53] // (op 18033) most significant half



s_mul_i32 s53, s34, s23 // (op 18035) most significant: low of low * high
s_mul_i32 s54, s35, s22 // (op 18035) most significant: low of high * low
s_mul_i32 s52, s34, s22 // (op 18035) least significant
s_mul_hi_u32 s55, s34, s22 // (op 18035) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18035) most significant: sum
s_add_u32 s53, s53, s55 // (op 18035) most significant: sum


s_mov_b32 s72, s36 // (op 18035) reg expression
s_mov_b32 s73, s37 // (op 18035) reg expression
s_mov_b32 s74, s52 // (op 18035) reg expression
s_mov_b32 s75, 131072 // (op 18035) call()

v_lshrrev_b32 v26, 3, v1 // (op 18052) 
s_mul_i32 s52, s16, s8 // (op 18052) 
s_add_u32 s52, s52, s0 // (op 18052) 
v_and_b32 v32, -8, v1 // (op 18052) 
s_mul_hi_u32 s53, s52, s13 // (op 18052) 
v_sub_u32 v33, v1, v32 // (op 18052) 

s_sub_u32 s54, s52, s53 // (op 18052) 
v_lshlrev_b32 v32, 4, v33 // (op 18052) 

s_lshr_b32 s55, s54, 1 // (op 18052) 

s_add_u32 s54, s55, s53 // (op 18052) 


s_lshr_b32 s53, s54, s11 // (op 18052) 

s_and_b32 s54, s11, 2147483648 // (op 18052) 
s_cmp_eq_i32 s54, 0 // (op 18052) 
s_mov_b32 s55, scc // (op 18052) 

s_cmp_eq_i32 s55, 0 // (op 18052) 
s_cselect_b32 s54, s52, s53 // (op 18052) 


s_mul_hi_u32 s53, s54, s12 // (op 18052) 
s_sub_u32 s55, s54, s53 // (op 18052) 
s_lshr_b32 s76, s55, 1 // (op 18052) 

s_add_u32 s55, s76, s53 // (op 18052) 


s_lshr_b32 s53, s55, s10 // (op 18052) 

s_and_b32 s55, s10, 2147483648 // (op 18052) 
s_cmp_eq_i32 s55, 0 // (op 18052) 
s_mov_b32 s76, scc // (op 18052) 

s_cmp_eq_i32 s76, 0 // (op 18052) 
s_cselect_b32 s55, s54, s53 // (op 18052) 


s_mul_i32 s53, s55, s14 // (op 18052) 

s_sub_u32 s55, s54, s53 // (op 18052) 

v_lshl_add_u32 v33, s55, 7, v26 // (op 18052) 


v_mul_lo_u32 v35, v33, s35 // (op 18052) most significant: low of low * high
v_mul_hi_u32 v26, v33, s34 // (op 18052) most significant: high of low * low
v_mul_lo_u32 v34, v33, s34 // (op 18052) least significant: low of low * low
v_add_u32 v35, v35, v26 // (op 18052) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 18052) 
s_mul_i32 s53, s54, s76 // (op 18052) 


s_sub_u32 s54, s52, s53 // (op 18052) 


v_lshl_add_u32 v26, s54, 7, v32 // (op 18052) 


v_mov_b32 v33, 0 // (op 18052) convert
v_mov_b32 v32, v26 // (op 18052) convert

v_add_co_u32 v38, s[52:53], v32, v34 // (op 18052) least significant half
v_addc_co_u32 v39, s[52:53], v33, v35, s[52:53] // (op 18052) most significant half



s_mul_i32 s53, s34, s23 // (op 18054) most significant: low of low * high
s_mul_i32 s54, s35, s22 // (op 18054) most significant: low of high * low
s_mul_i32 s52, s34, s22 // (op 18054) least significant
s_mul_hi_u32 s55, s34, s22 // (op 18054) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18054) most significant: sum
s_add_u32 s53, s53, s55 // (op 18054) most significant: sum


s_mov_b32 s72, s36 // (op 18054) reg expression
s_mov_b32 s73, s37 // (op 18054) reg expression
s_mov_b32 s74, s52 // (op 18054) reg expression
s_mov_b32 s75, 131072 // (op 18054) call()

v_and_b32 v26, -128, v1 // (op 18161) 
s_mul_i32 s52, s16, s8 // (op 18161) 
s_add_u32 s52, s52, s0 // (op 18161) 
v_sub_u32 v32, v1, v26 // (op 18161) 

s_mul_hi_u32 s53, s52, s13 // (op 18161) 
v_lshrrev_b32 v26, 7, v1 // (op 18161) 
s_sub_u32 s54, s52, s53 // (op 18161) 
v_lshlrev_b32 v33, 1, v26 // (op 18161) 

s_lshr_b32 s55, s54, 1 // (op 18161) 

s_add_u32 s54, s55, s53 // (op 18161) 


s_lshr_b32 s53, s54, s11 // (op 18161) 

s_and_b32 s54, s11, 2147483648 // (op 18161) 
s_cmp_eq_i32 s54, 0 // (op 18161) 
s_mov_b32 s55, scc // (op 18161) 

s_cmp_eq_i32 s55, 0 // (op 18161) 
s_cselect_b32 s54, s52, s53 // (op 18161) 


s_mul_hi_u32 s53, s54, s12 // (op 18161) 
s_sub_u32 s55, s54, s53 // (op 18161) 
s_lshr_b32 s76, s55, 1 // (op 18161) 

s_add_u32 s55, s76, s53 // (op 18161) 


s_lshr_b32 s53, s55, s10 // (op 18161) 

s_and_b32 s55, s10, 2147483648 // (op 18161) 
s_cmp_eq_i32 s55, 0 // (op 18161) 
s_mov_b32 s76, scc // (op 18161) 

s_cmp_eq_i32 s76, 0 // (op 18161) 
s_cselect_b32 s55, s54, s53 // (op 18161) 


v_lshl_add_u32 v26, s55, 7, v32 // (op 18161) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 18161) 
s_mul_i32 s53, s54, s76 // (op 18161) 


s_sub_u32 s54, s52, s53 // (op 18161) 


v_lshl_add_u32 v32, s54, 2, v33 // (op 18161) 


v_mul_lo_u32 v33, v32, s38 // (op 18161) 

v_add_u32 v32, v26, v33 // (op 18161) 


s_mul_i32 s53, s38, s43 // (op 18163) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 18163) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 18163) least significant
s_mul_hi_u32 s55, s38, s42 // (op 18163) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18163) most significant: sum
s_add_u32 s53, s53, s55 // (op 18163) most significant: sum


s_mov_b32 s76, s40 // (op 18163) reg expression
s_mov_b32 s77, s41 // (op 18163) reg expression
s_mov_b32 s78, s52 // (op 18163) reg expression
s_mov_b32 s79, 131072 // (op 18163) call()

v_and_b32 v26, -128, v1 // (op 18180) 
s_mul_i32 s52, s16, s8 // (op 18180) 
s_add_u32 s52, s52, s0 // (op 18180) 
v_sub_u32 v33, v1, v26 // (op 18180) 

s_mul_hi_u32 s53, s52, s13 // (op 18180) 
v_lshrrev_b32 v26, 7, v1 // (op 18180) 
s_sub_u32 s54, s52, s53 // (op 18180) 
v_lshlrev_b32 v34, 1, v26 // (op 18180) 

s_lshr_b32 s55, s54, 1 // (op 18180) 

s_add_u32 s54, s55, s53 // (op 18180) 


s_lshr_b32 s53, s54, s11 // (op 18180) 

s_and_b32 s54, s11, 2147483648 // (op 18180) 
s_cmp_eq_i32 s54, 0 // (op 18180) 
s_mov_b32 s55, scc // (op 18180) 

s_cmp_eq_i32 s55, 0 // (op 18180) 
s_cselect_b32 s54, s52, s53 // (op 18180) 


s_mul_hi_u32 s53, s54, s12 // (op 18180) 
s_sub_u32 s55, s54, s53 // (op 18180) 
s_lshr_b32 s80, s55, 1 // (op 18180) 

s_add_u32 s55, s80, s53 // (op 18180) 


s_lshr_b32 s53, s55, s10 // (op 18180) 

s_and_b32 s55, s10, 2147483648 // (op 18180) 
s_cmp_eq_i32 s55, 0 // (op 18180) 
s_mov_b32 s80, scc // (op 18180) 

s_cmp_eq_i32 s80, 0 // (op 18180) 
s_cselect_b32 s55, s54, s53 // (op 18180) 


v_lshl_add_u32 v26, s55, 7, v33 // (op 18180) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 18180) 
s_mul_i32 s53, s54, s80 // (op 18180) 


s_sub_u32 s54, s52, s53 // (op 18180) 


v_lshl_add_u32 v33, s54, 2, v34 // (op 18180) 


v_mul_lo_u32 v34, v33, s38 // (op 18180) 

v_add_u32 v33, v26, v34 // (op 18180) 


s_mul_i32 s53, s38, s43 // (op 18182) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 18182) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 18182) least significant
s_mul_hi_u32 s55, s38, s42 // (op 18182) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18182) most significant: sum
s_add_u32 s53, s53, s55 // (op 18182) most significant: sum


s_mov_b32 s76, s40 // (op 18182) reg expression
s_mov_b32 s77, s41 // (op 18182) reg expression
s_mov_b32 s78, s52 // (op 18182) reg expression
s_mov_b32 s79, 131072 // (op 18182) call()

v_lshrrev_b32 v26, 2, v1 // (op 18289) 
s_mul_i32 s52, s16, s8 // (op 18289) 
s_add_u32 s52, s52, s0 // (op 18289) 
v_lshlrev_b32 v34, 1, v26 // (op 18289) 

s_mul_hi_u32 s53, s52, s13 // (op 18289) 
v_and_b32 v26, -4, v1 // (op 18289) 
s_sub_u32 s54, s52, s53 // (op 18289) 
v_sub_u32 v35, v1, v26 // (op 18289) 

s_lshr_b32 s55, s54, 1 // (op 18289) 

s_add_u32 s54, s55, s53 // (op 18289) 


s_lshr_b32 s53, s54, s11 // (op 18289) 

s_and_b32 s54, s11, 2147483648 // (op 18289) 
s_cmp_eq_i32 s54, 0 // (op 18289) 
s_mov_b32 s55, scc // (op 18289) 

s_cmp_eq_i32 s55, 0 // (op 18289) 
s_cselect_b32 s54, s52, s53 // (op 18289) 


s_mul_hi_u32 s53, s54, s12 // (op 18289) 
s_sub_u32 s55, s54, s53 // (op 18289) 
s_lshr_b32 s80, s55, 1 // (op 18289) 

s_add_u32 s55, s80, s53 // (op 18289) 


s_lshr_b32 s53, s55, s10 // (op 18289) 

s_and_b32 s55, s10, 2147483648 // (op 18289) 
s_cmp_eq_i32 s55, 0 // (op 18289) 
s_mov_b32 s80, scc // (op 18289) 

s_cmp_eq_i32 s80, 0 // (op 18289) 
s_cselect_b32 s55, s54, s53 // (op 18289) 


s_mul_i32 s53, s55, s14 // (op 18289) 

s_sub_u32 s55, s54, s53 // (op 18289) 

v_lshl_add_u32 v26, s55, 7, v34 // (op 18289) 


v_mul_lo_u32 v34, v26, s44 // (op 18289) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 18289) 
s_mul_i32 s53, s54, s80 // (op 18289) 


s_sub_u32 s54, s52, s53 // (op 18289) 


v_lshl_add_u32 v26, s54, 2, v35 // (op 18289) 


v_add_u32 v35, v26, v34 // (op 18289) 


s_mul_i32 s53, s44, s23 // (op 18291) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 18291) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 18291) least significant
s_mul_hi_u32 s55, s44, s22 // (op 18291) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18291) most significant: sum
s_add_u32 s53, s53, s55 // (op 18291) most significant: sum


s_mov_b32 s80, s46 // (op 18291) reg expression
s_mov_b32 s81, s47 // (op 18291) reg expression
s_mov_b32 s82, s52 // (op 18291) reg expression
s_mov_b32 s83, 131072 // (op 18291) call()

v_lshrrev_b32 v26, 2, v1 // (op 18308) 
s_mul_i32 s52, s16, s8 // (op 18308) 
s_add_u32 s52, s52, s0 // (op 18308) 
v_lshlrev_b32 v34, 1, v26 // (op 18308) 

s_mul_hi_u32 s53, s52, s13 // (op 18308) 
v_and_b32 v26, -4, v1 // (op 18308) 
s_sub_u32 s54, s52, s53 // (op 18308) 
v_sub_u32 v40, v1, v26 // (op 18308) 

s_lshr_b32 s55, s54, 1 // (op 18308) 

s_add_u32 s54, s55, s53 // (op 18308) 


s_lshr_b32 s53, s54, s11 // (op 18308) 

s_and_b32 s54, s11, 2147483648 // (op 18308) 
s_cmp_eq_i32 s54, 0 // (op 18308) 
s_mov_b32 s55, scc // (op 18308) 

s_cmp_eq_i32 s55, 0 // (op 18308) 
s_cselect_b32 s54, s52, s53 // (op 18308) 


s_mul_hi_u32 s53, s54, s12 // (op 18308) 
s_sub_u32 s55, s54, s53 // (op 18308) 
s_lshr_b32 s84, s55, 1 // (op 18308) 

s_add_u32 s55, s84, s53 // (op 18308) 


s_lshr_b32 s53, s55, s10 // (op 18308) 

s_and_b32 s55, s10, 2147483648 // (op 18308) 
s_cmp_eq_i32 s55, 0 // (op 18308) 
s_mov_b32 s84, scc // (op 18308) 

s_cmp_eq_i32 s84, 0 // (op 18308) 
s_cselect_b32 s55, s54, s53 // (op 18308) 


s_mul_i32 s53, s55, s14 // (op 18308) 

s_sub_u32 s55, s54, s53 // (op 18308) 

v_lshl_add_u32 v26, s55, 7, v34 // (op 18308) 


v_mul_lo_u32 v34, v26, s44 // (op 18308) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 18308) 
s_mul_i32 s53, s54, s84 // (op 18308) 


s_sub_u32 s54, s52, s53 // (op 18308) 


v_lshl_add_u32 v26, s54, 2, v40 // (op 18308) 


v_add_u32 v40, v26, v34 // (op 18308) 


s_mul_i32 s53, s44, s23 // (op 18310) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 18310) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 18310) least significant
s_mul_hi_u32 s55, s44, s22 // (op 18310) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18310) most significant: sum
s_add_u32 s53, s53, s55 // (op 18310) most significant: sum


s_mov_b32 s80, s46 // (op 18310) reg expression
s_mov_b32 s81, s47 // (op 18310) reg expression
s_mov_b32 s82, s52 // (op 18310) reg expression
s_mov_b32 s83, 131072 // (op 18310) call()

v_and_b32 v26, -8, v1 // (op 18485) 
v_sub_u32 v34, v1, v26 // (op 18485) 

v_lshlrev_b32 v26, 4, v34 // (op 18485) 

v_lshrrev_b32 v34, 3, v1 // (op 18485) 
v_lshl_add_u32 v41, v34, 7, v26 // (op 18485) 


s_nop 0
v_readfirstlane_b32 s52, v41 // (op 18485) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 19624) 

s_lshl_b64 s[54:55], s[34:35], 5 // (op 17755) 
v_mov_b64 v[42:43], s[54:55] // (op 17755) convert


v_mov_b64 v[44:45], v[36:37] // (op 17755) 
s_mov_b32 m0, s52 // (op 17755) reg expression
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 17755) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 17755) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 17755) most significant half

s_add_u32 m0, m0, 4096 // (op 17755) 
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 17755) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 17755) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 17755) most significant half

s_add_u32 m0, m0, 4096 // (op 17755) 
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 17755) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 17755) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 17755) most significant half

s_add_u32 m0, m0, 4096 // (op 17755) 
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 17755) Load value direct to lds




v_mov_b64 v[42:43], s[44:45] // (op 17426) convert

v_mov_b32 v26, v35 // (op 17426) 
buffer_load_ubyte v44, v26, s[80:83], 0 offen offset: 0 // (op 17426) Load value
v_add_u32 v26, v26, v42 // (op 17426) 
buffer_load_ubyte v45, v26, s[80:83], 0 offen offset: 0 // (op 17426) Load value


v_and_b32 v26, -4, v1 // (op 18410) 
v_sub_u32 v34, v1, v26 // (op 18410) 

v_lshlrev_b32 v26, 5, v34 // (op 18410) 

v_lshrrev_b32 v34, 2, v1 // (op 18410) 
v_lshl_add_u32 v41, v34, 7, v26 // (op 18410) 


v_lshlrev_b32 v26, 2, v41 // (op 18410) 

v_lshrrev_b32 v34, 3, v26 // (op 18410) 

s_nop 0
v_readfirstlane_b32 s52, v34 // (op 18410) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 19605) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 17760) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 17760) 

v_lshrrev_b64 v[42:43], 3, s[84:85] // (op 17760) 


v_mov_b64 v[46:47], v[28:29] // (op 17760) reg expression
s_add_u32 s53, 16384, s52 // (op 17760) 
s_mov_b32 m0, s53 // (op 17760) reg expression
buffer_load_dwordx4 v46, s[68:71], 0 offen offset: 0 lds // (op 17760) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v42 // (op 17760) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 17760) most significant half

s_add_u32 m0, m0, 4096 // (op 17760) 
buffer_load_dwordx4 v46, s[68:71], 0 offen offset: 0 lds // (op 17760) Load value direct to lds





v_mov_b64 v[42:43], s[38:39] // (op 17432) convert

v_mov_b32 v26, v32 // (op 17432) reg expression
buffer_load_ubyte v46, v26, s[76:79], 0 offen offset: 0 // (op 17432) Load value
v_add_u32 v26, v26, v42 // (op 17432) 
buffer_load_ubyte v47, v26, s[76:79], 0 offen offset: 0 // (op 17432) Load value


v_and_b32 v26, -8, v1 // (op 19295) 
v_sub_u32 v34, v1, v26 // (op 19295) 

v_lshlrev_b32 v26, 4, v34 // (op 19295) 

v_lshrrev_b32 v34, 3, v1 // (op 19295) 
v_lshl_add_u32 v41, v34, 7, v26 // (op 19295) 


s_nop 0
v_readfirstlane_b32 s52, v41 // (op 19295) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 19804) 

s_lshl_b64 s[54:55], s[34:35], 5 // (op 17765) 
v_mov_b64 v[42:43], s[54:55] // (op 17765) convert


v_mov_b64 v[48:49], v[36:37] // (op 17765) 
s_add_u32 s53, 24576, s52 // (op 17765) 
s_mov_b32 m0, s53 // (op 17765) reg expression
s_mov_b32 s54, 128 // (op 17765) call()
buffer_load_dwordx4 v48, s[72:75], s54 offen offset: 0 lds // (op 17765) Load value direct to lds
v_add_co_u32 v48, s[84:85], v48, v42 // (op 17765) least significant half
v_addc_co_u32 v49, s[84:85], 0, v49, s[84:85] // (op 17765) most significant half

s_add_u32 m0, m0, 4096 // (op 17765) 
buffer_load_dwordx4 v48, s[72:75], s54 offen offset: 0 lds // (op 17765) Load value direct to lds
v_add_co_u32 v48, s[84:85], v48, v42 // (op 17765) least significant half
v_addc_co_u32 v49, s[84:85], 0, v49, s[84:85] // (op 17765) most significant half

s_add_u32 m0, m0, 4096 // (op 17765) 
buffer_load_dwordx4 v48, s[72:75], s54 offen offset: 0 lds // (op 17765) Load value direct to lds
v_add_co_u32 v48, s[84:85], v48, v42 // (op 17765) least significant half
v_addc_co_u32 v49, s[84:85], 0, v49, s[84:85] // (op 17765) most significant half

s_add_u32 m0, m0, 4096 // (op 17765) 
buffer_load_dwordx4 v48, s[72:75], s54 offen offset: 0 lds // (op 17765) Load value direct to lds







v_mov_b64 v[36:37], s[44:45] // (op 17438) convert

v_mov_b32 v26, v35 // (op 17438) 
buffer_load_ubyte v42, v26, s[80:83], 0 offen offset:4 // (op 17438) Load value
v_add_u32 v26, v26, v36 // (op 17438) 
buffer_load_ubyte v43, v26, s[80:83], 0 offen offset:4 // (op 17438) Load value



v_and_b32 v26, -4, v1 // (op 19251) 
v_sub_u32 v41, v1, v26 // (op 19251) 

v_lshlrev_b32 v26, 5, v41 // (op 19251) 

v_lshrrev_b32 v41, 2, v1 // (op 19251) 
v_lshl_add_u32 v34, v41, 7, v26 // (op 19251) 


v_lshlrev_b32 v26, 2, v34 // (op 19251) 

v_lshrrev_b32 v41, 3, v26 // (op 19251) 

s_nop 0
v_readfirstlane_b32 s52, v41 // (op 19251) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 19794) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 17767) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 17767) 

v_lshrrev_b64 v[34:35], 3, s[84:85] // (op 17767) 


s_lshl_b64 s[54:55], s[30:31], 7 // (op 17767) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 17767) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 17767) 

v_mov_b32 v26, s54 // (op 17767) 
v_mov_b32 v41, s55 // (op 17767) 
v_add_co_u32 v36, s[84:85], v28, v26 // (op 17767) least significant half
v_addc_co_u32 v37, s[84:85], v29, v41, s[84:85] // (op 17767) most significant half




s_add_u32 s53, 40960, s52 // (op 17767) 
s_mov_b32 m0, s53 // (op 17767) reg expression
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 17767) Load value direct to lds
v_add_co_u32 v36, s[54:55], v36, v34 // (op 17767) least significant half
v_addc_co_u32 v37, s[54:55], 0, v37, s[54:55] // (op 17767) most significant half

s_add_u32 m0, m0, 4096 // (op 17767) 
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 17767) Load value direct to lds






v_mov_b64 v[28:29], s[38:39] // (op 17444) convert

s_lshl_b32 s52, s38, 2 // (op 17444) 
v_add_u32 v26, v32, s52 // (op 17444) 

buffer_load_ubyte v34, v26, s[76:79], 0 offen offset: 0 // (op 17444) Load value
v_add_u32 v26, v26, v28 // (op 17444) 
buffer_load_ubyte v35, v26, s[76:79], 0 offen offset: 0 // (op 17444) Load value



v_and_b32 v26, -4, v1 // (op 18637) 
v_sub_u32 v32, v1, v26 // (op 18637) 

v_lshrrev_b32 v26, 2, v1 // (op 18637) 
v_lshl_add_u32 v41, v26, 3, v32 // (op 18637) 


s_barrier  // (op 19664) 



s_waitcnt vmcnt(15)
ds_write_b8 v41, v44 offset:49152 // (op 17447) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v41, v45 offset:49156 // (op 17447) Store local data 

v_and_b32 v26, -128, v1 // (op 18561) 
v_sub_u32 v32, v1, v26 // (op 18561) 

v_lshrrev_b32 v26, 7, v1 // (op 18561) 
v_lshl_add_u32 v41, v26, 8, v32 // (op 18561) 


s_barrier  // (op 19644) 



s_waitcnt vmcnt(11)
ds_write_b8 v41, v46 offset:49664 // (op 17449) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v41, v47 offset:49792 // (op 17449) Store local data 

s_barrier  // (op 19621) 



ds_read_b128 v[48:51], v23 // (op 17484) Load local data 
ds_read_b128 v[52:55], v23 offset:64 // (op 17484) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 19661) 



ds_read_u8 v26, v24 offset:49152 // (op 17524) Load local data 



ds_read_b128 v[56:59], v23 offset:4096 // (op 17489) Load local data 
ds_read_b128 v[60:63], v23 offset:4160 // (op 17489) Load local data 



ds_read_u8 v32, v24 offset:49280 // (op 17529) Load local data 



ds_read_b128 v[64:67], v23 offset:8192 // (op 17494) Load local data 
ds_read_b128 v[68:71], v23 offset:8256 // (op 17494) Load local data 



ds_read_u8 v41, v24 offset:49408 // (op 17534) Load local data 



ds_read_b128 v[72:75], v23 offset:12288 // (op 17499) Load local data 
ds_read_b128 v[76:79], v23 offset:12352 // (op 17499) Load local data 



ds_read_u8 v28, v24 offset:49536 // (op 17539) Load local data 
s_barrier  // (op 19602) 



ds_read_b64_tr_b4 v[80:81], v21 offset:16384 // (op 17464) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v21 offset:17408 // (op 17464) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 19641) 


ds_read_u8 v29, v20 offset:49664 // (op 17504) Load local data 



ds_read_b64_tr_b4 v[84:85], v21 offset:16400 // (op 17469) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v21 offset:17424 // (op 17469) Transpose load local data 


ds_read_u8 v36, v20 offset:49696 // (op 17509) Load local data 



ds_read_b64_tr_b4 v[88:89], v21 offset:16416 // (op 17474) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v21 offset:17440 // (op 17474) Transpose load local data 


ds_read_u8 v37, v20 offset:49728 // (op 17514) Load local data 



ds_read_b64_tr_b4 v[92:93], v21 offset:16432 // (op 17479) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v21 offset:17456 // (op 17479) Transpose load local data 


ds_read_u8 v96, v20 offset:49760 // (op 17519) Load local data 
v_and_b32 v97, -4, v1 // (op 18382) 
v_sub_u32 v98, v1, v97 // (op 18382) 

v_lshlrev_b32 v97, 5, v98 // (op 18382) 

v_lshrrev_b32 v98, 2, v1 // (op 18382) 
v_lshl_add_u32 v99, v98, 7, v97 // (op 18382) 


v_lshlrev_b32 v97, 2, v99 // (op 18382) 

v_lshrrev_b32 v98, 3, v97 // (op 18382) 

s_nop 0
v_readfirstlane_b32 s52, v98 // (op 18382) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v97, -8, v1 // (op 18457) 
v_sub_u32 v98, v1, v97 // (op 18457) 

v_lshlrev_b32 v97, 4, v98 // (op 18457) 

v_lshrrev_b32 v98, 3, v1 // (op 18457) 
v_lshl_add_u32 v99, v98, 7, v97 // (op 18457) 


s_nop 0
v_readfirstlane_b32 s53, v99 // (op 18457) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v97, -128, v1 // (op 18532) 
v_sub_u32 v98, v1, v97 // (op 18532) 

v_lshrrev_b32 v97, 7, v1 // (op 18532) 
v_lshl_add_u32 v99, v97, 8, v98 // (op 18532) 


v_and_b32 v97, -4, v1 // (op 18608) 
v_sub_u32 v98, v1, v97 // (op 18608) 

v_lshrrev_b32 v97, 2, v1 // (op 18608) 
v_lshl_add_u32 v100, v97, 3, v98 // (op 18608) 


v_and_b32 v97, -4, v1 // (op 19236) 
v_sub_u32 v98, v1, v97 // (op 19236) 

v_lshlrev_b32 v97, 5, v98 // (op 19236) 

v_lshrrev_b32 v98, 2, v1 // (op 19236) 
v_lshl_add_u32 v101, v98, 7, v97 // (op 19236) 


v_lshlrev_b32 v97, 2, v101 // (op 19236) 

v_lshrrev_b32 v98, 3, v97 // (op 19236) 

s_nop 0
v_readfirstlane_b32 s54, v98 // (op 19236) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v97, -8, v1 // (op 19280) 
v_sub_u32 v98, v1, v97 // (op 19280) 

v_lshlrev_b32 v97, 4, v98 // (op 19280) 

v_lshrrev_b32 v98, 3, v1 // (op 19280) 
v_lshl_add_u32 v101, v98, 7, v97 // (op 19280) 


s_nop 0
v_readfirstlane_b32 s55, v101 // (op 19280) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v97, -128, v1 // (op 19324) 
v_sub_u32 v98, v1, v97 // (op 19324) 

v_lshrrev_b32 v97, 7, v1 // (op 19324) 
v_lshl_add_u32 v101, v97, 8, v98 // (op 19324) 


v_and_b32 v97, -4, v1 // (op 19353) 
v_sub_u32 v98, v1, v97 // (op 19353) 

v_lshrrev_b32 v97, 2, v1 // (op 19353) 
v_lshl_add_u32 v102, v97, 3, v98 // (op 19353) 


s_mov_b32 s1, 0 // (op 6985) call()
s_and_b32 s84, s7, 4294967294 // (op 6984) 
s_cmp_lt_u32 s1, s84 // (op 6984) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_6984 // (op 6984) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_6984 if false)
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_6984:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:83], v[48:55], a[0:3], v29, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 88) 
s_barrier  // (op 19627) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[80:83], v[56:63], a[4:7], v29, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4774) 
s_add_u32 s1, s1, 2 // (op 6986) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[80:83], v[64:71], a[8:11], v29, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4950) 
s_lshl_b64 s[84:85], s[34:35], 5 // (op 17680) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[80:83], v[72:79], a[12:15], v29, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5126) 
v_mov_b64 v[104:105], s[84:85] // (op 17680) convert


v_mov_b64 v[106:107], v[38:39] // (op 17680) 
v_mov_b64 v[108:109], v[106:107] // (op 17680) 
s_mov_b32 m0, s53 // (op 17680) reg expression
s_mov_b32 s84, 256 // (op 17680) call()
buffer_load_dwordx4 v108, s[72:75], s84 offen offset: 0 lds // (op 17680) Load value direct to lds
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[84:87], v[48:55], a[16:19], v36, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5621) 
v_add_co_u32 v108, s[86:87], v108, v104 // (op 17680) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[84:87], v[56:63], a[20:23], v36, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5575) 
v_addc_co_u32 v109, s[86:87], 0, v109, s[86:87] // (op 17680) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:87], v[64:71], a[24:27], v36, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5529) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[84:87], v[72:79], a[28:31], v36, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5483) 
s_add_u32 m0, m0, 4096 // (op 17680) 
buffer_load_dwordx4 v108, s[72:75], s84 offen offset: 0 lds // (op 17680) Load value direct to lds
v_add_co_u32 v108, s[86:87], v108, v104 // (op 17680) least significant half
v_addc_co_u32 v109, s[86:87], 0, v109, s[86:87] // (op 17680) most significant half

s_add_u32 m0, m0, 4096 // (op 17680) 
buffer_load_dwordx4 v108, s[72:75], s84 offen offset: 0 lds // (op 17680) Load value direct to lds
v_add_co_u32 v108, s[86:87], v108, v104 // (op 17680) least significant half
v_addc_co_u32 v109, s[86:87], 0, v109, s[86:87] // (op 17680) most significant half

s_add_u32 m0, m0, 4096 // (op 17680) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[48:55], a[32:35], v37, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6391) 
buffer_load_dwordx4 v108, s[72:75], s84 offen offset: 0 lds // (op 17680) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[56:63], a[36:39], v37, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6345) 
v_mov_b64 v[104:105], s[44:45] // (op 25) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:91], v[64:71], a[40:43], v37, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6299) 
v_mov_b32 v103, v40 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:91], v[72:79], a[44:47], v37, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6253) 
v_mov_b32 v97, v103 // (op 25) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:95], v[48:55], a[48:51], v96, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7161) 
buffer_load_ubyte v44, v97, s[80:83], 0 offen offset:8 // (op 25) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:95], v[56:63], a[52:55], v96, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7115) 
v_add_u32 v97, v97, v104 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[64:71], a[56:59], v96, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7069) 
buffer_load_ubyte v45, v97, s[80:83], 0 offen offset:8 // (op 25) Load value


s_barrier  // (op 19608) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[72:79], a[60:63], v96, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7023) 

s_lshl_b64 s[84:85], s[30:31], 6 // (op 17675) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 17675) 

v_lshrrev_b64 v[104:105], 3, s[86:87] // (op 17675) 


v_mov_b64 v[108:109], v[30:31] // (op 17675) 
s_lshl_b64 s[84:85], s[30:31], 7 // (op 17675) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 17675) 

s_lshr_b64 s[84:85], s[86:87], 3 // (op 17675) 

s_lshl_b64 s[86:87], s[84:85], 1 // (op 17675) 

v_mov_b32 v97, s86 // (op 17675) 
v_mov_b32 v98, s87 // (op 17675) 
v_add_co_u32 v110, s[84:85], v108, v97 // (op 17675) least significant half
v_addc_co_u32 v111, s[84:85], v109, v98, s[84:85] // (op 17675) most significant half




s_add_u32 s84, 16384, s52 // (op 17675) 
s_nop 0
s_mov_b32 m0, s84 // (op 17675) reg expression
buffer_load_dwordx4 v110, s[68:71], 0 offen offset: 0 lds // (op 17675) Load value direct to lds
v_add_co_u32 v110, s[86:87], v110, v104 // (op 17675) least significant half
v_addc_co_u32 v111, s[86:87], 0, v111, s[86:87] // (op 17675) most significant half

s_add_u32 m0, m0, 4096 // (op 17675) 
buffer_load_dwordx4 v110, s[68:71], 0 offen offset: 0 lds // (op 17675) Load value direct to lds




v_mov_b64 v[104:105], s[38:39] // (op 17) convert

v_mov_b32 v97, v33 // (op 17) 
s_lshl_b32 s84, s38, 3 // (op 17) 
v_add_u32 v98, v97, s84 // (op 17) 

buffer_load_ubyte v46, v98, s[76:79], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v98, v98, v104 // (op 17) 
buffer_load_ubyte v47, v98, s[76:79], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 19819) 



s_waitcnt vmcnt(15)
ds_write_b8 v102, v42 offset:50176 // (op 4653) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v102, v43 offset:50180 // (op 4653) Store local data 
s_barrier  // (op 19813) 



s_waitcnt vmcnt(11)
ds_write_b8 v101, v34 offset:50688 // (op 4652) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v101, v35 offset:50816 // (op 4652) Store local data 
s_barrier  // (op 19800) 



ds_read_b128 v[110:113], v16 offset:24576 // (op 4645) Load local data 
ds_read_b128 v[114:117], v16 offset:24640 // (op 4645) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 19816) 



ds_read_u8 v98, v17 offset:50176 // (op 4649) Load local data 



ds_read_b128 v[118:121], v16 offset:28672 // (op 4779) Load local data 
ds_read_b128 v[122:125], v16 offset:28736 // (op 4779) Load local data 



ds_read_u8 v36, v17 offset:50304 // (op 4783) Load local data 



ds_read_b128 v[126:129], v16 offset:32768 // (op 4955) Load local data 
ds_read_b128 v[130:133], v16 offset:32832 // (op 4955) Load local data 



ds_read_u8 v37, v17 offset:50432 // (op 4959) Load local data 



ds_read_b128 v[134:137], v16 offset:36864 // (op 5131) Load local data 
ds_read_b128 v[138:141], v16 offset:36928 // (op 5131) Load local data 



ds_read_u8 v104, v17 offset:50560 // (op 5135) Load local data 
s_barrier  // (op 19791) 
s_barrier  // (op 20094) 



ds_read_b64_tr_b4 v[80:81], v15 offset:40960 // (op 4643) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v15 offset:41984 // (op 4643) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 19810) 


ds_read_u8 v29, v14 offset:50688 // (op 4647) Load local data 



ds_read_b64_tr_b4 v[84:85], v15 offset:40976 // (op 5486) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v15 offset:42000 // (op 5486) Transpose load local data 


ds_read_u8 v32, v14 offset:50720 // (op 5489) Load local data 



ds_read_b64_tr_b4 v[88:89], v15 offset:40992 // (op 6256) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v15 offset:42016 // (op 6256) Transpose load local data 


ds_read_u8 v41, v14 offset:50752 // (op 6259) Load local data 



ds_read_b64_tr_b4 v[92:93], v15 offset:41008 // (op 7026) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v15 offset:42032 // (op 7026) Transpose load local data 


ds_read_u8 v96, v14 offset:50784 // (op 7029) Load local data 
s_barrier  // (op 20128) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:83], v[110:117], a[0:3], v29, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4650) 
s_barrier  // (op 19807) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[80:83], v[118:125], a[4:7], v29, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4775) 
s_lshl_b64 s[84:85], s[34:35], 5 // (op 17736) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[80:83], v[126:133], a[8:11], v29, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4951) 
v_mov_b64 v[142:143], s[84:85] // (op 17736) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[80:83], v[134:141], a[12:15], v29, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5127) 
v_mov_b64 v[144:145], v[106:107] // (op 17736) 
s_add_u32 s84, 24576, s55 // (op 17736) 
s_mov_b32 m0, s84 // (op 17736) reg expression
s_mov_b32 s85, 384 // (op 17736) call()
buffer_load_dwordx4 v144, s[72:75], s85 offen offset: 0 lds // (op 17736) Load value direct to lds
v_add_co_u32 v144, s[86:87], v144, v142 // (op 17736) least significant half
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[84:87], v[110:117], a[16:19], v32, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5622) 
v_addc_co_u32 v145, s[86:87], 0, v145, s[86:87] // (op 17736) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[84:87], v[118:125], a[20:23], v32, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5576) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:87], v[126:133], a[24:27], v32, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5530) 
s_add_u32 m0, m0, 4096 // (op 17736) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[84:87], v[134:141], a[28:31], v32, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5484) 
buffer_load_dwordx4 v144, s[72:75], s85 offen offset: 0 lds // (op 17736) Load value direct to lds
v_add_co_u32 v144, s[86:87], v144, v142 // (op 17736) least significant half
v_addc_co_u32 v145, s[86:87], 0, v145, s[86:87] // (op 17736) most significant half

s_add_u32 m0, m0, 4096 // (op 17736) 
buffer_load_dwordx4 v144, s[72:75], s85 offen offset: 0 lds // (op 17736) Load value direct to lds
v_add_co_u32 v144, s[86:87], v144, v142 // (op 17736) least significant half
v_addc_co_u32 v145, s[86:87], 0, v145, s[86:87] // (op 17736) most significant half

s_add_u32 m0, m0, 4096 // (op 17736) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[110:117], a[32:35], v41, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6392) 
buffer_load_dwordx4 v144, s[72:75], s85 offen offset: 0 lds // (op 17736) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[118:125], a[36:39], v41, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6346) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:91], v[126:133], a[40:43], v41, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6300) 
v_add_co_u32 v38, vcc, 256, v38 // (op 18044) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:91], v[134:141], a[44:47], v41, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6254) 
v_addc_co_u32 v39, vcc, 0, v39, vcc // (op 18044) most significant half
v_mov_b64 v[106:107], s[44:45] // (op 4656) convert

v_mov_b32 v105, v103 // (op 4656) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:95], v[110:117], a[48:51], v96, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7162) 
buffer_load_ubyte v42, v105, s[80:83], 0 offen offset:12 // (op 4656) Load value


v_add_u32 v105, v105, v106 // (op 4656) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:95], v[118:125], a[52:55], v96, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7116) 
buffer_load_ubyte v43, v105, s[80:83], 0 offen offset:12 // (op 4656) Load value




s_barrier  // (op 19797) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[126:133], a[56:59], v96, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7070) 


v_add_u32 v40, 8, v40 // (op 18300) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[134:141], a[60:63], v96, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7024) 
s_lshl_b64 s[84:85], s[30:31], 6 // (op 17731) 


s_lshl_b64 s[86:87], s[84:85], 2 // (op 17731) 

v_lshrrev_b64 v[106:107], 3, s[86:87] // (op 17731) 


s_lshl_b64 s[84:85], s[30:31], 7 // (op 17731) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 17731) 

s_lshr_b64 s[84:85], s[86:87], 3 // (op 17731) 

s_mul_i32 s87, 3, s85 // (op 17731) most significant: low of low * high
s_mul_i32 s86, 3, s84 // (op 17731) least significant
s_mul_hi_u32 s88, 3, s84 // (op 17731) most significant: high of low * low
s_add_u32 s87, s87, s88 // (op 17731) most significant: sum


v_mov_b32 v98, s86 // (op 17731) 
v_mov_b32 v103, s87 // (op 17731) 
v_add_co_u32 v104, s[84:85], v108, v98 // (op 17731) least significant half
v_addc_co_u32 v105, s[84:85], v109, v103, s[84:85] // (op 17731) most significant half




s_add_u32 s84, 40960, s54 // (op 17731) 
s_nop 0
s_mov_b32 m0, s84 // (op 17731) reg expression
buffer_load_dwordx4 v104, s[68:71], 0 offen offset: 0 lds // (op 17731) Load value direct to lds
v_add_co_u32 v104, s[86:87], v104, v106 // (op 17731) least significant half
v_addc_co_u32 v105, s[86:87], 0, v105, s[86:87] // (op 17731) most significant half

s_add_u32 m0, m0, 4096 // (op 17731) 
buffer_load_dwordx4 v104, s[68:71], 0 offen offset: 0 lds // (op 17731) Load value direct to lds





v_mov_b64 v[104:105], s[38:39] // (op 4655) convert

s_lshl_b32 s84, s38, 2 // (op 4655) 
s_mul_i32 s85, 3, s84 // (op 4655) 

v_add_u32 v98, v97, s85 // (op 4655) 

s_lshl_b64 s[84:85], s[30:31], 7 // (op 17916) 
buffer_load_ubyte v34, v98, s[76:79], 0 offen offset: 0 // (op 4655) Load value
v_add_u32 v98, v98, v104 // (op 4655) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 17916) 

s_lshr_b64 s[84:85], s[86:87], 3 // (op 17916) 

s_lshl_b64 s[86:87], s[84:85], 1 // (op 17916) 

v_mov_b32 v103, s86 // (op 17916) 
v_mov_b32 v106, s87 // (op 17916) 
v_add_co_u32 v30, s[84:85], v30, v103 // (op 17916) least significant half
v_addc_co_u32 v31, s[84:85], v31, v106, s[84:85] // (op 17916) most significant half




buffer_load_ubyte v35, v98, s[76:79], 0 offen offset: 0 // (op 4655) Load value



s_barrier  // (op 19667) 
s_lshl_b32 s84, s38, 3 // (op 18172) 



v_add_u32 v33, v33, s84 // (op 18172) 

s_waitcnt vmcnt(15)
ds_write_b8 v100, v44 offset:49152 // (op 30) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v100, v45 offset:49156 // (op 30) Store local data 
s_barrier  // (op 19647) 



s_waitcnt vmcnt(11)
ds_write_b8 v99, v46 offset:49664 // (op 21) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v99, v47 offset:49792 // (op 21) Store local data 
s_barrier  // (op 19633) 



ds_read_b128 v[48:51], v4 // (op 75) Load local data 
ds_read_b128 v[52:55], v4 offset:64 // (op 75) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 19674) 



ds_read_u8 v26, v5 offset:49152 // (op 85) Load local data 



ds_read_b128 v[56:59], v4 offset:4096 // (op 4769) Load local data 
ds_read_b128 v[60:63], v4 offset:4160 // (op 4769) Load local data 



ds_read_u8 v32, v5 offset:49280 // (op 4773) Load local data 



ds_read_b128 v[64:67], v4 offset:8192 // (op 4945) Load local data 
ds_read_b128 v[68:71], v4 offset:8256 // (op 4945) Load local data 



ds_read_u8 v41, v5 offset:49408 // (op 4949) Load local data 



ds_read_b128 v[72:75], v4 offset:12288 // (op 5121) Load local data 
ds_read_b128 v[76:79], v4 offset:12352 // (op 5121) Load local data 



ds_read_u8 v28, v5 offset:49536 // (op 5125) Load local data 
s_barrier  // (op 19614) 



ds_read_b64_tr_b4 v[80:81], v3 offset:16384 // (op 70) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v3 offset:17408 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 19654) 


ds_read_u8 v29, v0 offset:49664 // (op 80) Load local data 



ds_read_b64_tr_b4 v[84:85], v3 offset:16400 // (op 5478) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v3 offset:17424 // (op 5478) Transpose load local data 


ds_read_u8 v36, v0 offset:49696 // (op 5481) Load local data 



ds_read_b64_tr_b4 v[88:89], v3 offset:16416 // (op 6248) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v3 offset:17440 // (op 6248) Transpose load local data 


ds_read_u8 v37, v0 offset:49728 // (op 6251) Load local data 



ds_read_b64_tr_b4 v[92:93], v3 offset:16432 // (op 7018) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v3 offset:17456 // (op 7018) Transpose load local data 


ds_read_u8 v96, v0 offset:49760 // (op 7021) Load local data 
s_and_b32 s84, s7, 4294967294 // (op 6984) 
s_cmp_lt_u32 s1, s84 // (op 6984) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_6984 // (op 6984) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_6984 if true)
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_6984:




















s_barrier  // (op 20112) 
s_and_b32 s52, s7, 4294967294 // (op 17804) 
v_and_b32 v26, -4, v1 // (op 17804) 
s_add_u32 s53, s0, s52 // (op 17804) 

v_sub_u32 v30, v1, v26 // (op 17804) 

v_lshlrev_b32 v26, 5, v30 // (op 17804) 

s_mul_i32 s52, s16, s8 // (op 17804) 
s_add_u32 s52, s52, s53 // (op 17804) 

v_lshrrev_b32 v30, 2, v1 // (op 17804) 
s_mul_hi_u32 s53, s52, s13 // (op 17804) 
s_sub_u32 s54, s52, s53 // (op 17804) 
s_lshr_b32 s55, s54, 1 // (op 17804) 

s_add_u32 s54, s55, s53 // (op 17804) 


s_lshr_b32 s53, s54, s11 // (op 17804) 

s_and_b32 s54, s11, 2147483648 // (op 17804) 
s_cmp_eq_i32 s54, 0 // (op 17804) 
s_mov_b32 s55, scc // (op 17804) 

s_cmp_eq_i32 s55, 0 // (op 17804) 
s_cselect_b32 s54, s52, s53 // (op 17804) 


s_mul_hi_u32 s53, s54, s12 // (op 17804) 
s_sub_u32 s55, s54, s53 // (op 17804) 
s_lshr_b32 s84, s55, 1 // (op 17804) 

s_add_u32 s55, s84, s53 // (op 17804) 


s_lshr_b32 s53, s55, s10 // (op 17804) 

s_and_b32 s55, s10, 2147483648 // (op 17804) 
s_cmp_eq_i32 s55, 0 // (op 17804) 
s_mov_b32 s84, scc // (op 17804) 

s_cmp_eq_i32 s84, 0 // (op 17804) 
s_cselect_b32 s55, s54, s53 // (op 17804) 


v_lshl_add_u32 v31, s55, 7, v26 // (op 17804) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 17804) 
v_mov_b32 v33, 0 // (op 17804) convert
v_mov_b32 v32, v31 // (op 17804) convert

s_mul_i32 s53, s54, s84 // (op 17804) 


s_sub_u32 s54, s52, s53 // (op 17804) 


v_lshl_add_u32 v26, s54, 7, v30 // (op 17804) 


v_mul_lo_u32 v31, v26, s31 // (op 17804) most significant: low of low * high
v_mul_hi_u32 v38, v26, s30 // (op 17804) most significant: high of low * low
v_mul_lo_u32 v30, v26, s30 // (op 17804) least significant: low of low * low
v_add_u32 v31, v31, v38 // (op 17804) most significant: sum


v_add_co_u32 v38, s[52:53], v32, v30 // (op 17804) least significant half
v_addc_co_u32 v39, s[52:53], v33, v31, s[52:53] // (op 17804) most significant half

v_lshlrev_b64 v[38:39], 2, v[38:39] // (op 17804) 


v_lshrrev_b64 v[40:41], 3, v[38:39] // (op 17804) 

s_mul_i32 s53, s30, s3 // (op 17806) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 17806) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 17806) least significant
s_mul_hi_u32 s55, s30, s2 // (op 17806) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 17806) most significant: sum
s_add_u32 s53, s53, s55 // (op 17806) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 17806) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 17806) 

s_mov_b32 s68, s32 // (op 17806) reg expression
s_mov_b32 s69, s33 // (op 17806) reg expression
s_mov_b32 s70, s52 // (op 17806) reg expression
s_mov_b32 s71, 131072 // (op 17806) call()

s_and_b32 s52, s7, 4294967294 // (op 17968) 
v_lshrrev_b32 v26, 3, v1 // (op 17968) 
s_add_u32 s53, s0, s52 // (op 17968) 

v_and_b32 v30, -8, v1 // (op 17968) 
v_sub_u32 v31, v1, v30 // (op 17968) 

s_mul_i32 s52, s16, s8 // (op 17968) 
s_add_u32 s52, s52, s53 // (op 17968) 

v_lshlrev_b32 v30, 4, v31 // (op 17968) 

s_mul_hi_u32 s53, s52, s13 // (op 17968) 
s_sub_u32 s54, s52, s53 // (op 17968) 
s_lshr_b32 s55, s54, 1 // (op 17968) 

s_add_u32 s54, s55, s53 // (op 17968) 


s_lshr_b32 s53, s54, s11 // (op 17968) 

s_and_b32 s54, s11, 2147483648 // (op 17968) 
s_cmp_eq_i32 s54, 0 // (op 17968) 
s_mov_b32 s55, scc // (op 17968) 

s_cmp_eq_i32 s55, 0 // (op 17968) 
s_cselect_b32 s54, s52, s53 // (op 17968) 


s_mul_hi_u32 s53, s54, s12 // (op 17968) 
s_sub_u32 s55, s54, s53 // (op 17968) 
s_lshr_b32 s84, s55, 1 // (op 17968) 

s_add_u32 s55, s84, s53 // (op 17968) 


s_lshr_b32 s53, s55, s10 // (op 17968) 

s_and_b32 s55, s10, 2147483648 // (op 17968) 
s_cmp_eq_i32 s55, 0 // (op 17968) 
s_mov_b32 s84, scc // (op 17968) 

s_cmp_eq_i32 s84, 0 // (op 17968) 
s_cselect_b32 s55, s54, s53 // (op 17968) 


s_mul_i32 s53, s55, s14 // (op 17968) 

s_sub_u32 s55, s54, s53 // (op 17968) 

v_lshl_add_u32 v31, s55, 7, v26 // (op 17968) 


v_mul_lo_u32 v33, v31, s35 // (op 17968) most significant: low of low * high
v_mul_hi_u32 v26, v31, s34 // (op 17968) most significant: high of low * low
v_mul_lo_u32 v32, v31, s34 // (op 17968) least significant: low of low * low
v_add_u32 v33, v33, v26 // (op 17968) most significant: sum


s_ashr_i64 s[84:85], s[2:3], 7 // (op 17968) 
s_mul_i32 s53, s54, s84 // (op 17968) 


s_sub_u32 s54, s52, s53 // (op 17968) 


v_lshl_add_u32 v26, s54, 7, v30 // (op 17968) 


v_mov_b32 v31, 0 // (op 17968) convert
v_mov_b32 v30, v26 // (op 17968) convert

v_add_co_u32 v38, s[52:53], v30, v32 // (op 17968) least significant half
v_addc_co_u32 v39, s[52:53], v31, v33, s[52:53] // (op 17968) most significant half



s_mul_i32 s53, s34, s23 // (op 17970) most significant: low of low * high
s_mul_i32 s54, s35, s22 // (op 17970) most significant: low of high * low
s_mul_i32 s52, s34, s22 // (op 17970) least significant
s_mul_hi_u32 s55, s34, s22 // (op 17970) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 17970) most significant: sum
s_add_u32 s53, s53, s55 // (op 17970) most significant: sum


s_mov_b32 s72, s36 // (op 17970) reg expression
s_mov_b32 s73, s37 // (op 17970) reg expression
s_mov_b32 s74, s52 // (op 17970) reg expression
s_mov_b32 s75, 131072 // (op 17970) call()

s_and_b32 s52, s7, 4294967294 // (op 18096) 
v_and_b32 v26, -128, v1 // (op 18096) 
s_add_u32 s53, s0, s52 // (op 18096) 

v_sub_u32 v30, v1, v26 // (op 18096) 

v_lshrrev_b32 v26, 7, v1 // (op 18096) 
s_mul_i32 s52, s16, s8 // (op 18096) 
s_add_u32 s52, s52, s53 // (op 18096) 

v_lshlrev_b32 v31, 1, v26 // (op 18096) 

s_mul_hi_u32 s53, s52, s13 // (op 18096) 
s_sub_u32 s54, s52, s53 // (op 18096) 
s_lshr_b32 s55, s54, 1 // (op 18096) 

s_add_u32 s54, s55, s53 // (op 18096) 


s_lshr_b32 s53, s54, s11 // (op 18096) 

s_and_b32 s54, s11, 2147483648 // (op 18096) 
s_cmp_eq_i32 s54, 0 // (op 18096) 
s_mov_b32 s55, scc // (op 18096) 

s_cmp_eq_i32 s55, 0 // (op 18096) 
s_cselect_b32 s54, s52, s53 // (op 18096) 


s_mul_hi_u32 s53, s54, s12 // (op 18096) 
s_sub_u32 s55, s54, s53 // (op 18096) 
s_lshr_b32 s84, s55, 1 // (op 18096) 

s_add_u32 s55, s84, s53 // (op 18096) 


s_lshr_b32 s53, s55, s10 // (op 18096) 

s_and_b32 s55, s10, 2147483648 // (op 18096) 
s_cmp_eq_i32 s55, 0 // (op 18096) 
s_mov_b32 s84, scc // (op 18096) 

s_cmp_eq_i32 s84, 0 // (op 18096) 
s_cselect_b32 s55, s54, s53 // (op 18096) 


v_lshl_add_u32 v26, s55, 7, v30 // (op 18096) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 18096) 
s_mul_i32 s53, s54, s84 // (op 18096) 


s_sub_u32 s54, s52, s53 // (op 18096) 


v_lshl_add_u32 v30, s54, 2, v31 // (op 18096) 


v_mul_lo_u32 v31, v30, s38 // (op 18096) 

v_add_u32 v30, v26, v31 // (op 18096) 


s_mul_i32 s53, s38, s43 // (op 18098) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 18098) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 18098) least significant
s_mul_hi_u32 s55, s38, s42 // (op 18098) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18098) most significant: sum
s_add_u32 s53, s53, s55 // (op 18098) most significant: sum


s_mov_b32 s76, s40 // (op 18098) reg expression
s_mov_b32 s77, s41 // (op 18098) reg expression
s_mov_b32 s78, s52 // (op 18098) reg expression
s_mov_b32 s79, 131072 // (op 18098) call()

s_and_b32 s52, s7, 4294967294 // (op 18224) 
v_lshrrev_b32 v26, 2, v1 // (op 18224) 
s_add_u32 s53, s0, s52 // (op 18224) 

v_lshlrev_b32 v31, 1, v26 // (op 18224) 

v_and_b32 v26, -4, v1 // (op 18224) 
s_mul_i32 s52, s16, s8 // (op 18224) 
s_add_u32 s52, s52, s53 // (op 18224) 

v_sub_u32 v32, v1, v26 // (op 18224) 

s_mul_hi_u32 s53, s52, s13 // (op 18224) 
s_sub_u32 s54, s52, s53 // (op 18224) 
s_lshr_b32 s55, s54, 1 // (op 18224) 

s_add_u32 s54, s55, s53 // (op 18224) 


s_lshr_b32 s53, s54, s11 // (op 18224) 

s_and_b32 s54, s11, 2147483648 // (op 18224) 
s_cmp_eq_i32 s54, 0 // (op 18224) 
s_mov_b32 s55, scc // (op 18224) 

s_cmp_eq_i32 s55, 0 // (op 18224) 
s_cselect_b32 s54, s52, s53 // (op 18224) 


s_mul_hi_u32 s53, s54, s12 // (op 18224) 
s_sub_u32 s55, s54, s53 // (op 18224) 
s_lshr_b32 s84, s55, 1 // (op 18224) 

s_add_u32 s55, s84, s53 // (op 18224) 


s_lshr_b32 s53, s55, s10 // (op 18224) 

s_and_b32 s55, s10, 2147483648 // (op 18224) 
s_cmp_eq_i32 s55, 0 // (op 18224) 
s_mov_b32 s84, scc // (op 18224) 

s_cmp_eq_i32 s84, 0 // (op 18224) 
s_cselect_b32 s55, s54, s53 // (op 18224) 


s_mul_i32 s53, s55, s14 // (op 18224) 

s_sub_u32 s55, s54, s53 // (op 18224) 

v_lshl_add_u32 v26, s55, 7, v31 // (op 18224) 


v_mul_lo_u32 v31, v26, s44 // (op 18224) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 18224) 
s_mul_i32 s53, s54, s84 // (op 18224) 


s_sub_u32 s54, s52, s53 // (op 18224) 


v_lshl_add_u32 v26, s54, 2, v32 // (op 18224) 


v_add_u32 v32, v26, v31 // (op 18224) 


s_mul_i32 s53, s44, s23 // (op 18226) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 18226) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 18226) least significant
s_mul_hi_u32 s55, s44, s22 // (op 18226) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 18226) most significant: sum
s_add_u32 s53, s53, s55 // (op 18226) most significant: sum


s_mov_b32 s80, s46 // (op 18226) reg expression
s_mov_b32 s81, s47 // (op 18226) reg expression
s_mov_b32 s82, s52 // (op 18226) reg expression
s_mov_b32 s83, 131072 // (op 18226) call()

v_and_b32 v26, -4, v1 // (op 18395) 
v_sub_u32 v31, v1, v26 // (op 18395) 

v_lshlrev_b32 v26, 5, v31 // (op 18395) 

v_lshrrev_b32 v31, 2, v1 // (op 18395) 
v_lshl_add_u32 v33, v31, 7, v26 // (op 18395) 


v_lshlrev_b32 v26, 2, v33 // (op 18395) 

v_lshrrev_b32 v31, 3, v26 // (op 18395) 

s_nop 0
v_readfirstlane_b32 s52, v31 // (op 18395) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -8, v1 // (op 18470) 
v_sub_u32 v31, v1, v26 // (op 18470) 

v_lshlrev_b32 v26, 4, v31 // (op 18470) 

v_lshrrev_b32 v31, 3, v1 // (op 18470) 
v_lshl_add_u32 v33, v31, 7, v26 // (op 18470) 


s_nop 0
v_readfirstlane_b32 s53, v33 // (op 18470) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -128, v1 // (op 18545) 
v_sub_u32 v31, v1, v26 // (op 18545) 

v_lshrrev_b32 v26, 7, v1 // (op 18545) 
v_lshl_add_u32 v33, v26, 8, v31 // (op 18545) 


v_and_b32 v26, -4, v1 // (op 18621) 
v_sub_u32 v31, v1, v26 // (op 18621) 

v_lshrrev_b32 v26, 2, v1 // (op 18621) 
v_lshl_add_u32 v44, v26, 3, v31 // (op 18621) 


s_and_b32 s54, s7, 4294967294 // (op 6990) 
s_cmp_lt_u32 s54, s7 // (op 6989) 












s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_6989 // (op 6989) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_6989 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_6989:
s_barrier  // (op 19611) 
s_add_u32 s54, s54, 1 // (op 6991) 

s_lshl_b64 s[84:85], s[30:31], 6 // (op 17718) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 17718) 

v_lshrrev_b64 v[28:29], 3, s[86:87] // (op 17718) 


v_mov_b64 v[42:43], v[40:41] // (op 17718) 
v_mov_b64 v[34:35], v[42:43] // (op 17718) reg expression
s_add_u32 s55, 16384, s52 // (op 17718) 
s_mov_b32 m0, s55 // (op 17718) reg expression
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 17718) Load value direct to lds
v_add_co_u32 v34, s[84:85], v34, v28 // (op 17718) least significant half
v_addc_co_u32 v35, s[84:85], 0, v35, s[84:85] // (op 17718) most significant half

s_add_u32 m0, m0, 4096 // (op 17718) 
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 17718) Load value direct to lds




s_lshl_b64 s[84:85], s[30:31], 7 // (op 17794) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 17794) 

s_lshr_b64 s[84:85], s[86:87], 3 // (op 17794) 

v_mov_b32 v26, s84 // (op 17794) 
v_mov_b32 v31, s85 // (op 17794) 
v_add_co_u32 v40, s[86:87], v40, v26 // (op 17794) least significant half
v_addc_co_u32 v41, s[86:87], v41, v31, s[86:87] // (op 17794) most significant half




s_barrier  // (op 19630) 

s_lshl_b64 s[84:85], s[34:35], 5 // (op 17724) 
v_mov_b64 v[28:29], s[84:85] // (op 17724) convert


v_mov_b64 v[42:43], v[38:39] // (op 17724) 
v_mov_b64 v[34:35], v[42:43] // (op 17724) 
s_mov_b32 m0, s53 // (op 17724) reg expression
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 17724) Load value direct to lds
v_add_co_u32 v34, s[84:85], v34, v28 // (op 17724) least significant half
v_addc_co_u32 v35, s[84:85], 0, v35, s[84:85] // (op 17724) most significant half

s_add_u32 m0, m0, 4096 // (op 17724) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 17724) Load value direct to lds
v_add_co_u32 v34, s[84:85], v34, v28 // (op 17724) least significant half
v_addc_co_u32 v35, s[84:85], 0, v35, s[84:85] // (op 17724) most significant half

s_add_u32 m0, m0, 4096 // (op 17724) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 17724) Load value direct to lds
v_add_co_u32 v34, s[84:85], v34, v28 // (op 17724) least significant half
v_addc_co_u32 v35, s[84:85], 0, v35, s[84:85] // (op 17724) most significant half

s_add_u32 m0, m0, 4096 // (op 17724) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 17724) Load value direct to lds



v_add_co_u32 v38, vcc, 128, v38 // (op 17960) least significant half
v_addc_co_u32 v39, vcc, 0, v39, vcc // (op 17960) most significant half

v_mov_b64 v[28:29], s[38:39] // (op 4589) convert

v_mov_b32 v26, v30 // (op 4589) 
v_mov_b32 v31, v26 // (op 4589) reg expression
buffer_load_ubyte v42, v31, s[76:79], 0 offen offset: 0 // (op 4589) Load value
v_add_u32 v31, v31, v28 // (op 4589) 
buffer_load_ubyte v43, v31, s[76:79], 0 offen offset: 0 // (op 4589) Load value



s_lshl_b32 s55, s38, 2 // (op 18088) 
v_add_u32 v30, v30, s55 // (op 18088) 


s_barrier  // (op 19650) 
v_mov_b64 v[28:29], s[44:45] // (op 4590) convert




v_mov_b32 v26, v32 // (op 4590) 
v_mov_b32 v31, v26 // (op 4590) 
buffer_load_ubyte v34, v31, s[80:83], 0 offen offset: 0 // (op 4590) Load value
v_add_u32 v31, v31, v28 // (op 4590) 
s_waitcnt vmcnt(2)
ds_write_b8 v33, v42 offset:49664 // (op 4586) Store local data 
buffer_load_ubyte v35, v31, s[80:83], 0 offen offset: 0 // (op 4590) Load value



v_add_u32 v32, 4, v32 // (op 18216) 
s_waitcnt vmcnt(2)
ds_write_b8 v33, v43 offset:49792 // (op 4586) Store local data 

s_barrier  // (op 19617) 



s_barrier  // (op 19670) 
ds_read_b64_tr_b4 v[46:47], v3 offset:16384 // (op 4577) Transpose load local data 



ds_read_b64_tr_b4 v[48:49], v3 offset:17408 // (op 4577) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v44, v34 offset:49152 // (op 4587) Store local data 
s_barrier  // (op 19637) 
s_waitcnt vmcnt(0)
ds_write_b8 v44, v35 offset:49156 // (op 4587) Store local data 




ds_read_b128 v[50:53], v4 // (op 4579) Load local data 
ds_read_b128 v[54:57], v4 offset:64 // (op 4579) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 19657) 


ds_read_u8 v26, v0 offset:49664 // (op 4581) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 19677) 



ds_read_u8 v31, v5 offset:49152 // (op 4583) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:49], v[50:57], a[0:3], v26, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4584) 
ds_read_b128 v[58:61], v4 offset:4096 // (op 4748) Load local data 
ds_read_b128 v[62:65], v4 offset:4160 // (op 4748) Load local data 



ds_read_u8 v45, v5 offset:49280 // (op 4752) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:49], v[58:65], a[4:7], v26, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4753) 
ds_read_b128 v[66:69], v4 offset:8192 // (op 4924) Load local data 
ds_read_b128 v[70:73], v4 offset:8256 // (op 4924) Load local data 



ds_read_u8 v28, v5 offset:49408 // (op 4928) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[46:49], v[66:73], a[8:11], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4929) 
ds_read_b128 v[74:77], v4 offset:12288 // (op 5100) Load local data 
ds_read_b128 v[78:81], v4 offset:12352 // (op 5100) Load local data 



ds_read_u8 v29, v5 offset:49536 // (op 5104) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[46:49], v[74:81], a[12:15], v26, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5105) 
ds_read_b64_tr_b4 v[34:35], v3 offset:16400 // (op 5362) Transpose load local data 


ds_read_b64_tr_b4 v[36:37], v3 offset:17424 // (op 5362) Transpose load local data 


ds_read_u8 v26, v0 offset:49696 // (op 5365) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[34:37], v[50:57], a[16:19], v26, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5367) 
ds_read_b64_tr_b4 v[46:47], v3 offset:16416 // (op 6132) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[34:37], v[58:65], a[20:23], v26, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5398) 
ds_read_b64_tr_b4 v[48:49], v3 offset:17440 // (op 6132) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[34:37], v[66:73], a[24:27], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5429) 
ds_read_u8 v42, v0 offset:49728 // (op 6135) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[34:37], v[74:81], a[28:31], v26, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5460) 
ds_read_b64_tr_b4 v[82:83], v3 offset:16432 // (op 6902) Transpose load local data 


ds_read_b64_tr_b4 v[84:85], v3 offset:17456 // (op 6902) Transpose load local data 


s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[46:49], v[50:57], a[32:35], v42, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6137) 
ds_read_u8 v26, v0 offset:49760 // (op 6905) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[46:49], v[58:65], a[36:39], v42, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6168) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[46:49], v[66:73], a[40:43], v42, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6199) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[46:49], v[74:81], a[44:47], v42, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6230) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[82:85], v[50:57], a[48:51], v26, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6907) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[82:85], v[58:65], a[52:55], v26, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6938) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[82:85], v[66:73], a[56:59], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6969) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[82:85], v[74:81], a[60:63], v26, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7000) 




s_cmp_lt_u32 s54, s7 // (op 6989) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_6989 // (op 6989) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_6989 if true)
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_6989:









s_barrier  // (op 20116) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 288) 
s_cmp_lg_u32 s18, 0 // (op 314) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Send_Tile_314 // (op 314) Condition: False, jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Send_Tile_314



v_add_u32 v26, 32768, v25 // (op 8008) 
buffer_store_dword a32, v26, s[64:67], 0 offen sc1 // (op 8008) Store value

v_add_u32 v26, 33280, v25 // (op 8008) 
buffer_store_dword a33, v26, s[64:67], 0 offen sc1 // (op 8008) Store value

v_add_u32 v26, 33792, v25 // (op 8008) 
buffer_store_dword a34, v26, s[64:67], 0 offen sc1 // (op 8008) Store value

v_add_u32 v26, 34304, v25 // (op 8008) 
buffer_store_dword a35, v26, s[64:67], 0 offen sc1 // (op 8008) Store value




v_add_u32 v26, 32896, v25 // (op 8010) 
buffer_store_dword a36, v26, s[64:67], 0 offen sc1 // (op 8010) Store value

v_add_u32 v26, 33408, v25 // (op 8010) 
buffer_store_dword a37, v26, s[64:67], 0 offen sc1 // (op 8010) Store value

v_add_u32 v26, 33920, v25 // (op 8010) 
buffer_store_dword a38, v26, s[64:67], 0 offen sc1 // (op 8010) Store value

v_add_u32 v26, 34432, v25 // (op 8010) 
buffer_store_dword a39, v26, s[64:67], 0 offen sc1 // (op 8010) Store value




v_add_u32 v26, 33024, v25 // (op 8012) 
buffer_store_dword a40, v26, s[64:67], 0 offen sc1 // (op 8012) Store value

v_add_u32 v26, 33536, v25 // (op 8012) 
buffer_store_dword a41, v26, s[64:67], 0 offen sc1 // (op 8012) Store value

v_add_u32 v26, 34048, v25 // (op 8012) 
buffer_store_dword a42, v26, s[64:67], 0 offen sc1 // (op 8012) Store value

v_add_u32 v26, 34560, v25 // (op 8012) 
buffer_store_dword a43, v26, s[64:67], 0 offen sc1 // (op 8012) Store value




v_add_u32 v26, 33152, v25 // (op 8014) 
buffer_store_dword a44, v26, s[64:67], 0 offen sc1 // (op 8014) Store value

v_add_u32 v26, 33664, v25 // (op 8014) 
buffer_store_dword a45, v26, s[64:67], 0 offen sc1 // (op 8014) Store value

v_add_u32 v26, 34176, v25 // (op 8014) 
buffer_store_dword a46, v26, s[64:67], 0 offen sc1 // (op 8014) Store value

v_add_u32 v26, 34688, v25 // (op 8014) 
buffer_store_dword a47, v26, s[64:67], 0 offen sc1 // (op 8014) Store value




v_add_u32 v26, 16384, v25 // (op 7984) 
buffer_store_dword a16, v26, s[64:67], 0 offen sc1 // (op 7984) Store value

v_add_u32 v26, 16896, v25 // (op 7984) 
buffer_store_dword a17, v26, s[64:67], 0 offen sc1 // (op 7984) Store value

v_add_u32 v26, 17408, v25 // (op 7984) 
buffer_store_dword a18, v26, s[64:67], 0 offen sc1 // (op 7984) Store value

v_add_u32 v26, 17920, v25 // (op 7984) 
buffer_store_dword a19, v26, s[64:67], 0 offen sc1 // (op 7984) Store value




v_add_u32 v26, 16512, v25 // (op 7986) 
buffer_store_dword a20, v26, s[64:67], 0 offen sc1 // (op 7986) Store value

v_add_u32 v26, 17024, v25 // (op 7986) 
buffer_store_dword a21, v26, s[64:67], 0 offen sc1 // (op 7986) Store value

v_add_u32 v26, 17536, v25 // (op 7986) 
buffer_store_dword a22, v26, s[64:67], 0 offen sc1 // (op 7986) Store value

v_add_u32 v26, 18048, v25 // (op 7986) 
buffer_store_dword a23, v26, s[64:67], 0 offen sc1 // (op 7986) Store value




v_add_u32 v26, 16640, v25 // (op 7988) 
buffer_store_dword a24, v26, s[64:67], 0 offen sc1 // (op 7988) Store value

v_add_u32 v26, 17152, v25 // (op 7988) 
buffer_store_dword a25, v26, s[64:67], 0 offen sc1 // (op 7988) Store value

v_add_u32 v26, 17664, v25 // (op 7988) 
buffer_store_dword a26, v26, s[64:67], 0 offen sc1 // (op 7988) Store value

v_add_u32 v26, 18176, v25 // (op 7988) 
buffer_store_dword a27, v26, s[64:67], 0 offen sc1 // (op 7988) Store value




v_add_u32 v26, 16768, v25 // (op 7990) 
buffer_store_dword a28, v26, s[64:67], 0 offen sc1 // (op 7990) Store value

v_add_u32 v26, 17280, v25 // (op 7990) 
buffer_store_dword a29, v26, s[64:67], 0 offen sc1 // (op 7990) Store value

v_add_u32 v26, 17792, v25 // (op 7990) 
buffer_store_dword a30, v26, s[64:67], 0 offen sc1 // (op 7990) Store value

v_add_u32 v26, 18304, v25 // (op 7990) 
buffer_store_dword a31, v26, s[64:67], 0 offen sc1 // (op 7990) Store value




v_add_u32 v26, 49152, v25 // (op 8032) 
buffer_store_dword a48, v26, s[64:67], 0 offen sc1 // (op 8032) Store value

v_add_u32 v26, 49664, v25 // (op 8032) 
buffer_store_dword a49, v26, s[64:67], 0 offen sc1 // (op 8032) Store value

v_add_u32 v26, 50176, v25 // (op 8032) 
buffer_store_dword a50, v26, s[64:67], 0 offen sc1 // (op 8032) Store value

v_add_u32 v26, 50688, v25 // (op 8032) 
buffer_store_dword a51, v26, s[64:67], 0 offen sc1 // (op 8032) Store value




v_add_u32 v26, 49280, v25 // (op 8034) 
buffer_store_dword a52, v26, s[64:67], 0 offen sc1 // (op 8034) Store value

v_add_u32 v26, 49792, v25 // (op 8034) 
buffer_store_dword a53, v26, s[64:67], 0 offen sc1 // (op 8034) Store value

v_add_u32 v26, 50304, v25 // (op 8034) 
buffer_store_dword a54, v26, s[64:67], 0 offen sc1 // (op 8034) Store value

v_add_u32 v26, 50816, v25 // (op 8034) 
buffer_store_dword a55, v26, s[64:67], 0 offen sc1 // (op 8034) Store value




v_add_u32 v26, 49408, v25 // (op 8036) 
buffer_store_dword a56, v26, s[64:67], 0 offen sc1 // (op 8036) Store value

v_add_u32 v26, 49920, v25 // (op 8036) 
buffer_store_dword a57, v26, s[64:67], 0 offen sc1 // (op 8036) Store value

v_add_u32 v26, 50432, v25 // (op 8036) 
buffer_store_dword a58, v26, s[64:67], 0 offen sc1 // (op 8036) Store value

v_add_u32 v26, 50944, v25 // (op 8036) 
buffer_store_dword a59, v26, s[64:67], 0 offen sc1 // (op 8036) Store value




v_add_u32 v26, 49536, v25 // (op 8038) 
buffer_store_dword a60, v26, s[64:67], 0 offen sc1 // (op 8038) Store value

v_add_u32 v26, 50048, v25 // (op 8038) 
buffer_store_dword a61, v26, s[64:67], 0 offen sc1 // (op 8038) Store value

v_add_u32 v26, 50560, v25 // (op 8038) 
buffer_store_dword a62, v26, s[64:67], 0 offen sc1 // (op 8038) Store value

v_add_u32 v26, 51072, v25 // (op 8038) 
buffer_store_dword a63, v26, s[64:67], 0 offen sc1 // (op 8038) Store value




buffer_store_dword a0, v25, s[64:67], 0 offen sc1 // (op 311) Store value
buffer_store_dword a1, v25, s[64:67], 0 offen offset:512 sc1 // (op 311) Store value
buffer_store_dword a2, v25, s[64:67], 0 offen offset:1024 sc1 // (op 311) Store value
buffer_store_dword a3, v25, s[64:67], 0 offen offset:1536 sc1 // (op 311) Store value



buffer_store_dword a4, v25, s[64:67], 0 offen offset:128 sc1 // (op 7958) Store value
buffer_store_dword a5, v25, s[64:67], 0 offen offset:640 sc1 // (op 7958) Store value
buffer_store_dword a6, v25, s[64:67], 0 offen offset:1152 sc1 // (op 7958) Store value
buffer_store_dword a7, v25, s[64:67], 0 offen offset:1664 sc1 // (op 7958) Store value



buffer_store_dword a8, v25, s[64:67], 0 offen offset:256 sc1 // (op 7959) Store value
buffer_store_dword a9, v25, s[64:67], 0 offen offset:768 sc1 // (op 7959) Store value
buffer_store_dword a10, v25, s[64:67], 0 offen offset:1280 sc1 // (op 7959) Store value
buffer_store_dword a11, v25, s[64:67], 0 offen offset:1792 sc1 // (op 7959) Store value



buffer_store_dword a12, v25, s[64:67], 0 offen offset:384 sc1 // (op 7960) Store value
buffer_store_dword a13, v25, s[64:67], 0 offen offset:896 sc1 // (op 7960) Store value
buffer_store_dword a14, v25, s[64:67], 0 offen offset:1408 sc1 // (op 7960) Store value
buffer_store_dword a15, v25, s[64:67], 0 offen offset:1920 sc1 // (op 7960) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 315) 
v_cmp_eq_i32 vcc, v1, 0 // (op 319) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_319 // (op 319) Condition: False, jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_319
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
s_branch RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_319 // (op 319) Condition: Done, jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_319
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_319:
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_319:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Send_Tile_314 // (op 314) Condition: Done, jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Send_Tile_314
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Send_Tile_314:
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Send_Tile_314:

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
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Receive_Tile_330 // (op 330) Condition: False, jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Receive_Tile_330
s_mov_b32 s52, 0 // (op 296) call()
s_mul_i32 s53, s52, s8 // (op 295) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 295) 
s_mov_b32 s84, s54 // (op 295) convert

s_sub_u32 s54, s84, s19 // (op 295) 

s_sub_u32 s55, s54, 1 // (op 295) 

s_cmp_lt_u32 s53, s55 // (op 295) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_295 // (op 295) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_295 if false)
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_ReceiveTileLoop_295:
s_add_u32 s53, s16, 1 // (op 19075) 
v_lshrrev_b32 v20, 5, v1 // (op 19075) 
s_add_u32 s54, s53, s52 // (op 19075) 
s_lshl_b32 s54, s54, 7 // (op 19075) 

v_lshlrev_b32 v21, 2, v20 // (op 19075) 

v_add_lshl_u32 v20, s54, v21, 7 // (op 19075) 


v_and_b32 v21, -32, v1 // (op 19075) 
v_sub_u32 v23, v1, v21 // (op 19075) 

v_add_lshl_u32 v21, v20, v23, 2 // (op 19075) 


s_lshl_b32 s53, s15, 16 // (op 19077) 
s_mov_b32 s64, s28 // (op 19077) reg expression
s_mov_b32 s65, s29 // (op 19077) reg expression
s_mov_b32 s66, s53 // (op 19077) reg expression
s_mov_b32 s67, 131072 // (op 19077) call()

RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_334:
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
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_334 // (op 334) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_334 if true)
s_barrier  // (op 338) 
v_cmp_eq_i32 vcc, v1, 0 // (op 337) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_337 // (op 337) Condition: False, jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_337
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
s_branch RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_337 // (op 337) Condition: Done, jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_337
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_337:
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_337:




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




buffer_load_dword v28, v21, s[64:67], 0 offen offset:128 // (op 8107) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:640 // (op 8107) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:1152 // (op 8107) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:1664 // (op 8107) Load value
v_accvgpr_read v32, a4 // (op 8108) reg expression
v_accvgpr_read v33, a5 // (op 8108) reg expression
v_accvgpr_read v34, a6 // (op 8108) reg expression
v_accvgpr_read v35, a7 // (op 8108) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8109) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8109) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8109) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8109) 

v_accvgpr_write a4, v32 // (op 8110) reg expression
v_accvgpr_write a5, v33 // (op 8110) reg expression
v_accvgpr_write a6, v34 // (op 8110) reg expression
v_accvgpr_write a7, v35 // (op 8110) reg expression




buffer_load_dword v28, v21, s[64:67], 0 offen offset:256 // (op 8114) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:768 // (op 8114) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:1280 // (op 8114) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:1792 // (op 8114) Load value
v_accvgpr_read v32, a8 // (op 8115) reg expression
v_accvgpr_read v33, a9 // (op 8115) reg expression
v_accvgpr_read v34, a10 // (op 8115) reg expression
v_accvgpr_read v35, a11 // (op 8115) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8116) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8116) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8116) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8116) 

v_accvgpr_write a8, v32 // (op 8117) reg expression
v_accvgpr_write a9, v33 // (op 8117) reg expression
v_accvgpr_write a10, v34 // (op 8117) reg expression
v_accvgpr_write a11, v35 // (op 8117) reg expression




buffer_load_dword v28, v21, s[64:67], 0 offen offset:384 // (op 8121) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:896 // (op 8121) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:1408 // (op 8121) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:1920 // (op 8121) Load value
v_accvgpr_read v32, a12 // (op 8122) reg expression
v_accvgpr_read v33, a13 // (op 8122) reg expression
v_accvgpr_read v34, a14 // (op 8122) reg expression
v_accvgpr_read v35, a15 // (op 8122) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8123) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8123) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8123) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8123) 

v_accvgpr_write a12, v32 // (op 8124) reg expression
v_accvgpr_write a13, v33 // (op 8124) reg expression
v_accvgpr_write a14, v34 // (op 8124) reg expression
v_accvgpr_write a15, v35 // (op 8124) reg expression




v_mov_b32 v20, 16384 // (op 8158) 
v_add_i32 v23, v21, v20 // (op 8158) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8158) Load value

v_mov_b32 v20, 16896 // (op 8158) 
v_add_i32 v23, v21, v20 // (op 8158) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8158) Load value

v_mov_b32 v20, 17408 // (op 8158) 
v_add_i32 v23, v21, v20 // (op 8158) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8158) Load value

v_mov_b32 v20, 17920 // (op 8158) 
v_add_i32 v23, v21, v20 // (op 8158) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8158) Load value

v_accvgpr_read v32, a16 // (op 8155) reg expression
v_accvgpr_read v33, a17 // (op 8155) reg expression
v_accvgpr_read v34, a18 // (op 8155) reg expression
v_accvgpr_read v35, a19 // (op 8155) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8156) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8156) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8156) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8156) 

v_accvgpr_write a16, v32 // (op 8157) reg expression
v_accvgpr_write a17, v33 // (op 8157) reg expression
v_accvgpr_write a18, v34 // (op 8157) reg expression
v_accvgpr_write a19, v35 // (op 8157) reg expression




v_mov_b32 v20, 16512 // (op 8163) 
v_add_i32 v23, v21, v20 // (op 8163) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8163) Load value

v_mov_b32 v20, 17024 // (op 8163) 
v_add_i32 v23, v21, v20 // (op 8163) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8163) Load value

v_mov_b32 v20, 17536 // (op 8163) 
v_add_i32 v23, v21, v20 // (op 8163) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8163) Load value

v_mov_b32 v20, 18048 // (op 8163) 
v_add_i32 v23, v21, v20 // (op 8163) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8163) Load value

v_accvgpr_read v32, a20 // (op 8160) reg expression
v_accvgpr_read v33, a21 // (op 8160) reg expression
v_accvgpr_read v34, a22 // (op 8160) reg expression
v_accvgpr_read v35, a23 // (op 8160) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8161) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8161) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8161) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8161) 

v_accvgpr_write a20, v32 // (op 8162) reg expression
v_accvgpr_write a21, v33 // (op 8162) reg expression
v_accvgpr_write a22, v34 // (op 8162) reg expression
v_accvgpr_write a23, v35 // (op 8162) reg expression




v_mov_b32 v20, 16640 // (op 8168) 
v_add_i32 v23, v21, v20 // (op 8168) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8168) Load value

v_mov_b32 v20, 17152 // (op 8168) 
v_add_i32 v23, v21, v20 // (op 8168) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8168) Load value

v_mov_b32 v20, 17664 // (op 8168) 
v_add_i32 v23, v21, v20 // (op 8168) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8168) Load value

v_mov_b32 v20, 18176 // (op 8168) 
v_add_i32 v23, v21, v20 // (op 8168) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8168) Load value

v_accvgpr_read v32, a24 // (op 8165) reg expression
v_accvgpr_read v33, a25 // (op 8165) reg expression
v_accvgpr_read v34, a26 // (op 8165) reg expression
v_accvgpr_read v35, a27 // (op 8165) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8166) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8166) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8166) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8166) 

v_accvgpr_write a24, v32 // (op 8167) reg expression
v_accvgpr_write a25, v33 // (op 8167) reg expression
v_accvgpr_write a26, v34 // (op 8167) reg expression
v_accvgpr_write a27, v35 // (op 8167) reg expression




v_mov_b32 v20, 16768 // (op 8173) 
v_add_i32 v23, v21, v20 // (op 8173) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8173) Load value

v_mov_b32 v20, 17280 // (op 8173) 
v_add_i32 v23, v21, v20 // (op 8173) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8173) Load value

v_mov_b32 v20, 17792 // (op 8173) 
v_add_i32 v23, v21, v20 // (op 8173) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8173) Load value

v_mov_b32 v20, 18304 // (op 8173) 
v_add_i32 v23, v21, v20 // (op 8173) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8173) Load value

v_accvgpr_read v32, a28 // (op 8170) reg expression
v_accvgpr_read v33, a29 // (op 8170) reg expression
v_accvgpr_read v34, a30 // (op 8170) reg expression
v_accvgpr_read v35, a31 // (op 8170) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8171) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8171) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8171) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8171) 

v_accvgpr_write a28, v32 // (op 8172) reg expression
v_accvgpr_write a29, v33 // (op 8172) reg expression
v_accvgpr_write a30, v34 // (op 8172) reg expression
v_accvgpr_write a31, v35 // (op 8172) reg expression




v_mov_b32 v20, 32768 // (op 8206) 
v_add_i32 v23, v21, v20 // (op 8206) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8206) Load value

v_mov_b32 v20, 33280 // (op 8206) 
v_add_i32 v23, v21, v20 // (op 8206) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8206) Load value

v_mov_b32 v20, 33792 // (op 8206) 
v_add_i32 v23, v21, v20 // (op 8206) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8206) Load value

v_mov_b32 v20, 34304 // (op 8206) 
v_add_i32 v23, v21, v20 // (op 8206) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8206) Load value

v_accvgpr_read v32, a32 // (op 8203) reg expression
v_accvgpr_read v33, a33 // (op 8203) reg expression
v_accvgpr_read v34, a34 // (op 8203) reg expression
v_accvgpr_read v35, a35 // (op 8203) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8204) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8204) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8204) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8204) 

v_accvgpr_write a32, v32 // (op 8205) reg expression
v_accvgpr_write a33, v33 // (op 8205) reg expression
v_accvgpr_write a34, v34 // (op 8205) reg expression
v_accvgpr_write a35, v35 // (op 8205) reg expression




v_mov_b32 v20, 32896 // (op 8211) 
v_add_i32 v23, v21, v20 // (op 8211) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8211) Load value

v_mov_b32 v20, 33408 // (op 8211) 
v_add_i32 v23, v21, v20 // (op 8211) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8211) Load value

v_mov_b32 v20, 33920 // (op 8211) 
v_add_i32 v23, v21, v20 // (op 8211) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8211) Load value

v_mov_b32 v20, 34432 // (op 8211) 
v_add_i32 v23, v21, v20 // (op 8211) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8211) Load value

v_accvgpr_read v32, a36 // (op 8208) reg expression
v_accvgpr_read v33, a37 // (op 8208) reg expression
v_accvgpr_read v34, a38 // (op 8208) reg expression
v_accvgpr_read v35, a39 // (op 8208) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8209) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8209) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8209) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8209) 

v_accvgpr_write a36, v32 // (op 8210) reg expression
v_accvgpr_write a37, v33 // (op 8210) reg expression
v_accvgpr_write a38, v34 // (op 8210) reg expression
v_accvgpr_write a39, v35 // (op 8210) reg expression




v_mov_b32 v20, 33024 // (op 8216) 
v_add_i32 v23, v21, v20 // (op 8216) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8216) Load value

v_mov_b32 v20, 33536 // (op 8216) 
v_add_i32 v23, v21, v20 // (op 8216) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8216) Load value

v_mov_b32 v20, 34048 // (op 8216) 
v_add_i32 v23, v21, v20 // (op 8216) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8216) Load value

v_mov_b32 v20, 34560 // (op 8216) 
v_add_i32 v23, v21, v20 // (op 8216) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8216) Load value

v_accvgpr_read v32, a40 // (op 8213) reg expression
v_accvgpr_read v33, a41 // (op 8213) reg expression
v_accvgpr_read v34, a42 // (op 8213) reg expression
v_accvgpr_read v35, a43 // (op 8213) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8214) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8214) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8214) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8214) 

v_accvgpr_write a40, v32 // (op 8215) reg expression
v_accvgpr_write a41, v33 // (op 8215) reg expression
v_accvgpr_write a42, v34 // (op 8215) reg expression
v_accvgpr_write a43, v35 // (op 8215) reg expression




v_mov_b32 v20, 33152 // (op 8221) 
v_add_i32 v23, v21, v20 // (op 8221) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8221) Load value

v_mov_b32 v20, 33664 // (op 8221) 
v_add_i32 v23, v21, v20 // (op 8221) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8221) Load value

v_mov_b32 v20, 34176 // (op 8221) 
v_add_i32 v23, v21, v20 // (op 8221) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8221) Load value

v_mov_b32 v20, 34688 // (op 8221) 
v_add_i32 v23, v21, v20 // (op 8221) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8221) Load value

v_accvgpr_read v32, a44 // (op 8218) reg expression
v_accvgpr_read v33, a45 // (op 8218) reg expression
v_accvgpr_read v34, a46 // (op 8218) reg expression
v_accvgpr_read v35, a47 // (op 8218) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8219) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8219) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8219) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8219) 

v_accvgpr_write a44, v32 // (op 8220) reg expression
v_accvgpr_write a45, v33 // (op 8220) reg expression
v_accvgpr_write a46, v34 // (op 8220) reg expression
v_accvgpr_write a47, v35 // (op 8220) reg expression




v_mov_b32 v20, 49152 // (op 8254) 
v_add_i32 v23, v21, v20 // (op 8254) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8254) Load value

v_mov_b32 v20, 49664 // (op 8254) 
v_add_i32 v23, v21, v20 // (op 8254) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8254) Load value

v_mov_b32 v20, 50176 // (op 8254) 
v_add_i32 v23, v21, v20 // (op 8254) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8254) Load value

v_mov_b32 v20, 50688 // (op 8254) 
v_add_i32 v23, v21, v20 // (op 8254) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8254) Load value

v_accvgpr_read v32, a48 // (op 8251) reg expression
v_accvgpr_read v33, a49 // (op 8251) reg expression
v_accvgpr_read v34, a50 // (op 8251) reg expression
v_accvgpr_read v35, a51 // (op 8251) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8252) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8252) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8252) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8252) 

v_accvgpr_write a48, v32 // (op 8253) reg expression
v_accvgpr_write a49, v33 // (op 8253) reg expression
v_accvgpr_write a50, v34 // (op 8253) reg expression
v_accvgpr_write a51, v35 // (op 8253) reg expression




v_mov_b32 v20, 49280 // (op 8259) 
v_add_i32 v23, v21, v20 // (op 8259) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8259) Load value

v_mov_b32 v20, 49792 // (op 8259) 
v_add_i32 v23, v21, v20 // (op 8259) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8259) Load value

v_mov_b32 v20, 50304 // (op 8259) 
v_add_i32 v23, v21, v20 // (op 8259) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8259) Load value

v_mov_b32 v20, 50816 // (op 8259) 
v_add_i32 v23, v21, v20 // (op 8259) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8259) Load value

v_accvgpr_read v32, a52 // (op 8256) reg expression
v_accvgpr_read v33, a53 // (op 8256) reg expression
v_accvgpr_read v34, a54 // (op 8256) reg expression
v_accvgpr_read v35, a55 // (op 8256) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8257) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8257) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8257) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8257) 

v_accvgpr_write a52, v32 // (op 8258) reg expression
v_accvgpr_write a53, v33 // (op 8258) reg expression
v_accvgpr_write a54, v34 // (op 8258) reg expression
v_accvgpr_write a55, v35 // (op 8258) reg expression




v_mov_b32 v20, 49408 // (op 8264) 
v_add_i32 v23, v21, v20 // (op 8264) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8264) Load value

v_mov_b32 v20, 49920 // (op 8264) 
v_add_i32 v23, v21, v20 // (op 8264) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8264) Load value

v_mov_b32 v20, 50432 // (op 8264) 
v_add_i32 v23, v21, v20 // (op 8264) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8264) Load value

v_mov_b32 v20, 50944 // (op 8264) 
v_add_i32 v23, v21, v20 // (op 8264) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8264) Load value

v_accvgpr_read v32, a56 // (op 8261) reg expression
v_accvgpr_read v33, a57 // (op 8261) reg expression
v_accvgpr_read v34, a58 // (op 8261) reg expression
v_accvgpr_read v35, a59 // (op 8261) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8262) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8262) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8262) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8262) 

v_accvgpr_write a56, v32 // (op 8263) reg expression
v_accvgpr_write a57, v33 // (op 8263) reg expression
v_accvgpr_write a58, v34 // (op 8263) reg expression
v_accvgpr_write a59, v35 // (op 8263) reg expression




v_mov_b32 v20, 49536 // (op 8269) 
v_add_i32 v23, v21, v20 // (op 8269) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 8269) Load value

v_mov_b32 v20, 50048 // (op 8269) 
v_add_i32 v23, v21, v20 // (op 8269) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 8269) Load value

v_mov_b32 v20, 50560 // (op 8269) 
v_add_i32 v23, v21, v20 // (op 8269) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 8269) Load value

v_mov_b32 v20, 51072 // (op 8269) 
v_add_i32 v23, v21, v20 // (op 8269) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 8269) Load value

v_accvgpr_read v32, a60 // (op 8266) reg expression
v_accvgpr_read v33, a61 // (op 8266) reg expression
v_accvgpr_read v34, a62 // (op 8266) reg expression
v_accvgpr_read v35, a63 // (op 8266) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 8267) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 8267) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 8267) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 8267) 

v_accvgpr_write a60, v32 // (op 8268) reg expression
v_accvgpr_write a61, v33 // (op 8268) reg expression
v_accvgpr_write a62, v34 // (op 8268) reg expression
v_accvgpr_write a63, v35 // (op 8268) reg expression

s_add_u32 s52, s52, 1 // (op 297) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 295) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 295) 
s_mov_b32 s84, s54 // (op 295) convert

s_sub_u32 s54, s84, s19 // (op 295) 

s_sub_u32 s55, s54, 1 // (op 295) 

s_cmp_lt_u32 s53, s55 // (op 295) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_ReceiveTileLoop_295 // (op 295) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_ReceiveTileLoop_295 if true)
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_295:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Receive_Tile_330 // (op 330) Condition: Done, jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Receive_Tile_330
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Receive_Tile_330:
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Receive_Tile_330:

s_cmp_eq_i32 s18, 0 // (op 373) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_373 // (op 373) Condition: False, jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_373
v_cmp_eq_f32 vcc, 0.00000, v18 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_133
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


v_accvgpr_read v28, a4 // (op 8351) 
v_accvgpr_read v29, a5 // (op 8351) 
v_accvgpr_read v30, a6 // (op 8351) 
v_accvgpr_read v31, a7 // (op 8351) 
v_mul_f32 v32, v19, v28 // (op 8351) 
v_mul_f32 v33, v19, v29 // (op 8351) 
v_mul_f32 v34, v19, v30 // (op 8351) 
v_mul_f32 v35, v19, v31 // (op 8351) 



s_lshl_b32 s52, s24, 7 // (op 8353) 
v_add_u32 v20, v27, s52 // (op 8353) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 8353) Store value


v_accvgpr_read v28, a8 // (op 8363) 
v_accvgpr_read v29, a9 // (op 8363) 
v_accvgpr_read v30, a10 // (op 8363) 
v_accvgpr_read v31, a11 // (op 8363) 
v_mul_f32 v32, v19, v28 // (op 8363) 
v_mul_f32 v33, v19, v29 // (op 8363) 
v_mul_f32 v34, v19, v30 // (op 8363) 
v_mul_f32 v35, v19, v31 // (op 8363) 



s_lshl_b32 s52, s24, 8 // (op 8365) 
v_add_u32 v20, v27, s52 // (op 8365) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 8365) Store value


v_accvgpr_read v28, a12 // (op 8375) 
v_accvgpr_read v29, a13 // (op 8375) 
v_accvgpr_read v30, a14 // (op 8375) 
v_accvgpr_read v31, a15 // (op 8375) 
v_mul_f32 v32, v19, v28 // (op 8375) 
v_mul_f32 v33, v19, v29 // (op 8375) 
v_mul_f32 v34, v19, v30 // (op 8375) 
v_mul_f32 v35, v19, v31 // (op 8375) 



s_lshl_b32 s52, s24, 7 // (op 8377) 
s_mul_i32 s53, 3, s52 // (op 8377) 

v_add_u32 v20, v27, s53 // (op 8377) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 8377) Store value


v_accvgpr_read v28, a16 // (op 8409) 
v_accvgpr_read v29, a17 // (op 8409) 
v_accvgpr_read v30, a18 // (op 8409) 
v_accvgpr_read v31, a19 // (op 8409) 
v_mul_f32 v32, v19, v28 // (op 8409) 
v_mul_f32 v33, v19, v29 // (op 8409) 
v_mul_f32 v34, v19, v30 // (op 8409) 
v_mul_f32 v35, v19, v31 // (op 8409) 



v_add_u32 v20, 128, v27 // (op 8412) 

buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 8412) Store value


v_accvgpr_read v28, a20 // (op 8422) 
v_accvgpr_read v29, a21 // (op 8422) 
v_accvgpr_read v30, a22 // (op 8422) 
v_accvgpr_read v31, a23 // (op 8422) 
v_mul_f32 v32, v19, v28 // (op 8422) 
v_mul_f32 v33, v19, v29 // (op 8422) 
v_mul_f32 v34, v19, v30 // (op 8422) 
v_mul_f32 v35, v19, v31 // (op 8422) 



s_lshl_b32 s52, s24, 7 // (op 8414) 
s_add_u32 s52, s52, 128 // (op 8414) 
v_add_u32 v20, v27, s52 // (op 8414) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 8414) Store value


v_accvgpr_read v28, a24 // (op 8427) 
v_accvgpr_read v29, a25 // (op 8427) 
v_accvgpr_read v30, a26 // (op 8427) 
v_accvgpr_read v31, a27 // (op 8427) 
v_mul_f32 v32, v19, v28 // (op 8427) 
v_mul_f32 v33, v19, v29 // (op 8427) 
v_mul_f32 v34, v19, v30 // (op 8427) 
v_mul_f32 v35, v19, v31 // (op 8427) 



s_lshl_b32 s52, s24, 8 // (op 8416) 
s_add_u32 s52, s52, 128 // (op 8416) 
v_add_u32 v20, v27, s52 // (op 8416) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 8416) Store value


v_accvgpr_read v28, a28 // (op 8432) 
v_accvgpr_read v29, a29 // (op 8432) 
v_accvgpr_read v30, a30 // (op 8432) 
v_accvgpr_read v31, a31 // (op 8432) 
v_mul_f32 v32, v19, v28 // (op 8432) 
v_mul_f32 v33, v19, v29 // (op 8432) 
v_mul_f32 v34, v19, v30 // (op 8432) 
v_mul_f32 v35, v19, v31 // (op 8432) 



s_lshl_b32 s52, s24, 7 // (op 8418) 
s_mul_i32 s53, 3, s52 // (op 8418) 
s_add_u32 s53, s53, 128 // (op 8418) 

v_add_u32 v20, v27, s53 // (op 8418) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 8418) Store value


v_accvgpr_read v28, a32 // (op 8477) 
v_accvgpr_read v29, a33 // (op 8477) 
v_accvgpr_read v30, a34 // (op 8477) 
v_accvgpr_read v31, a35 // (op 8477) 
v_mul_f32 v32, v19, v28 // (op 8477) 
v_mul_f32 v33, v19, v29 // (op 8477) 
v_mul_f32 v34, v19, v30 // (op 8477) 
v_mul_f32 v35, v19, v31 // (op 8477) 



v_add_u32 v20, 256, v27 // (op 8480) 

buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 8480) Store value


v_accvgpr_read v28, a36 // (op 8490) 
v_accvgpr_read v29, a37 // (op 8490) 
v_accvgpr_read v30, a38 // (op 8490) 
v_accvgpr_read v31, a39 // (op 8490) 
v_mul_f32 v32, v19, v28 // (op 8490) 
v_mul_f32 v33, v19, v29 // (op 8490) 
v_mul_f32 v34, v19, v30 // (op 8490) 
v_mul_f32 v35, v19, v31 // (op 8490) 



s_lshl_b32 s52, s24, 7 // (op 8482) 
s_add_u32 s52, s52, 256 // (op 8482) 
v_add_u32 v20, v27, s52 // (op 8482) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 8482) Store value


v_accvgpr_read v28, a40 // (op 8495) 
v_accvgpr_read v29, a41 // (op 8495) 
v_accvgpr_read v30, a42 // (op 8495) 
v_accvgpr_read v31, a43 // (op 8495) 
v_mul_f32 v32, v19, v28 // (op 8495) 
v_mul_f32 v33, v19, v29 // (op 8495) 
v_mul_f32 v34, v19, v30 // (op 8495) 
v_mul_f32 v35, v19, v31 // (op 8495) 



s_lshl_b32 s52, s24, 8 // (op 8484) 
s_add_u32 s52, s52, 256 // (op 8484) 
v_add_u32 v20, v27, s52 // (op 8484) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 8484) Store value


v_accvgpr_read v28, a44 // (op 8500) 
v_accvgpr_read v29, a45 // (op 8500) 
v_accvgpr_read v30, a46 // (op 8500) 
v_accvgpr_read v31, a47 // (op 8500) 
v_mul_f32 v32, v19, v28 // (op 8500) 
v_mul_f32 v33, v19, v29 // (op 8500) 
v_mul_f32 v34, v19, v30 // (op 8500) 
v_mul_f32 v35, v19, v31 // (op 8500) 



s_lshl_b32 s52, s24, 7 // (op 8486) 
s_mul_i32 s53, 3, s52 // (op 8486) 
s_add_u32 s53, s53, 256 // (op 8486) 

v_add_u32 v20, v27, s53 // (op 8486) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 8486) Store value


v_accvgpr_read v28, a48 // (op 8545) 
v_accvgpr_read v29, a49 // (op 8545) 
v_accvgpr_read v30, a50 // (op 8545) 
v_accvgpr_read v31, a51 // (op 8545) 
v_mul_f32 v32, v19, v28 // (op 8545) 
v_mul_f32 v33, v19, v29 // (op 8545) 
v_mul_f32 v34, v19, v30 // (op 8545) 
v_mul_f32 v35, v19, v31 // (op 8545) 



v_add_u32 v20, 384, v27 // (op 8548) 

buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 8548) Store value


v_accvgpr_read v28, a52 // (op 8558) 
v_accvgpr_read v29, a53 // (op 8558) 
v_accvgpr_read v30, a54 // (op 8558) 
v_accvgpr_read v31, a55 // (op 8558) 
v_mul_f32 v32, v19, v28 // (op 8558) 
v_mul_f32 v33, v19, v29 // (op 8558) 
v_mul_f32 v34, v19, v30 // (op 8558) 
v_mul_f32 v35, v19, v31 // (op 8558) 



s_lshl_b32 s52, s24, 7 // (op 8550) 
s_add_u32 s52, s52, 384 // (op 8550) 
v_add_u32 v20, v27, s52 // (op 8550) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 8550) Store value


v_accvgpr_read v28, a56 // (op 8563) 
v_accvgpr_read v29, a57 // (op 8563) 
v_accvgpr_read v30, a58 // (op 8563) 
v_accvgpr_read v31, a59 // (op 8563) 
v_mul_f32 v32, v19, v28 // (op 8563) 
v_mul_f32 v33, v19, v29 // (op 8563) 
v_mul_f32 v34, v19, v30 // (op 8563) 
v_mul_f32 v35, v19, v31 // (op 8563) 



s_lshl_b32 s52, s24, 8 // (op 8552) 
s_add_u32 s52, s52, 384 // (op 8552) 
v_add_u32 v20, v27, s52 // (op 8552) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 8552) Store value


v_accvgpr_read v28, a60 // (op 8568) 
v_accvgpr_read v29, a61 // (op 8568) 
v_accvgpr_read v30, a62 // (op 8568) 
v_accvgpr_read v31, a63 // (op 8568) 
v_mul_f32 v32, v19, v28 // (op 8568) 
v_mul_f32 v33, v19, v29 // (op 8568) 
v_mul_f32 v34, v19, v30 // (op 8568) 
v_mul_f32 v35, v19, v31 // (op 8568) 



s_lshl_b32 s52, s24, 7 // (op 8554) 
s_mul_i32 s53, 3, s52 // (op 8554) 
s_add_u32 s53, s53, 384 // (op 8554) 

v_add_u32 v20, v27, s53 // (op 8554) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 8554) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_133
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_133:


v_add_u32 v20, 384, v22 // (op 8912) 

buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 8912) Load value

v_accvgpr_read v32, a48 // (op 8900) 
v_accvgpr_read v33, a49 // (op 8900) 
v_accvgpr_read v34, a50 // (op 8900) 
v_accvgpr_read v35, a51 // (op 8900) 
v_mul_f32 v36, v19, v32 // (op 8900) 
v_mul_f32 v37, v19, v33 // (op 8900) 
v_mul_f32 v38, v19, v34 // (op 8900) 
v_mul_f32 v39, v19, v35 // (op 8900) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 8911) 
v_mul_f32 v33, v18, v29 // (op 8911) 
v_mul_f32 v34, v18, v30 // (op 8911) 
v_mul_f32 v35, v18, v31 // (op 8911) 

v_add_f32 v28, v32, v36 // (op 8901) 
v_add_f32 v29, v33, v37 // (op 8901) 
v_add_f32 v30, v34, v38 // (op 8901) 
v_add_f32 v31, v35, v39 // (op 8901) 




v_add_u32 v20, 384, v27 // (op 8903) 

buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 8903) Store value




s_lshl_b32 s52, s4, 7 // (op 8916) 
s_add_u32 s52, s52, 384 // (op 8916) 
v_add_u32 v20, v22, s52 // (op 8916) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 8916) Load value

v_accvgpr_read v32, a52 // (op 8926) 
v_accvgpr_read v33, a53 // (op 8926) 
v_accvgpr_read v34, a54 // (op 8926) 
v_accvgpr_read v35, a55 // (op 8926) 
v_mul_f32 v36, v19, v32 // (op 8926) 
v_mul_f32 v37, v19, v33 // (op 8926) 
v_mul_f32 v38, v19, v34 // (op 8926) 
v_mul_f32 v39, v19, v35 // (op 8926) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 8914) 
v_mul_f32 v33, v18, v29 // (op 8914) 
v_mul_f32 v34, v18, v30 // (op 8914) 
v_mul_f32 v35, v18, v31 // (op 8914) 

v_add_f32 v28, v32, v36 // (op 8915) 
v_add_f32 v29, v33, v37 // (op 8915) 
v_add_f32 v30, v34, v38 // (op 8915) 
v_add_f32 v31, v35, v39 // (op 8915) 




s_lshl_b32 s52, s24, 7 // (op 8905) 
s_add_u32 s52, s52, 384 // (op 8905) 
v_add_u32 v20, v27, s52 // (op 8905) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 8905) Store value




s_lshl_b32 s52, s4, 8 // (op 8920) 
s_add_u32 s52, s52, 384 // (op 8920) 
v_add_u32 v20, v22, s52 // (op 8920) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 8920) Load value

v_accvgpr_read v32, a56 // (op 8928) 
v_accvgpr_read v33, a57 // (op 8928) 
v_accvgpr_read v34, a58 // (op 8928) 
v_accvgpr_read v35, a59 // (op 8928) 
v_mul_f32 v36, v19, v32 // (op 8928) 
v_mul_f32 v37, v19, v33 // (op 8928) 
v_mul_f32 v38, v19, v34 // (op 8928) 
v_mul_f32 v39, v19, v35 // (op 8928) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 8918) 
v_mul_f32 v33, v18, v29 // (op 8918) 
v_mul_f32 v34, v18, v30 // (op 8918) 
v_mul_f32 v35, v18, v31 // (op 8918) 

v_add_f32 v28, v32, v36 // (op 8919) 
v_add_f32 v29, v33, v37 // (op 8919) 
v_add_f32 v30, v34, v38 // (op 8919) 
v_add_f32 v31, v35, v39 // (op 8919) 




s_lshl_b32 s52, s24, 8 // (op 8907) 
s_add_u32 s52, s52, 384 // (op 8907) 
v_add_u32 v20, v27, s52 // (op 8907) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 8907) Store value




s_lshl_b32 s52, s4, 7 // (op 8924) 
s_mul_i32 s53, 3, s52 // (op 8924) 
s_add_u32 s53, s53, 384 // (op 8924) 

v_add_u32 v20, v22, s53 // (op 8924) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 8924) Load value

v_accvgpr_read v32, a60 // (op 8930) 
v_accvgpr_read v33, a61 // (op 8930) 
v_accvgpr_read v34, a62 // (op 8930) 
v_accvgpr_read v35, a63 // (op 8930) 
v_mul_f32 v36, v19, v32 // (op 8930) 
v_mul_f32 v37, v19, v33 // (op 8930) 
v_mul_f32 v38, v19, v34 // (op 8930) 
v_mul_f32 v39, v19, v35 // (op 8930) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 8922) 
v_mul_f32 v33, v18, v29 // (op 8922) 
v_mul_f32 v34, v18, v30 // (op 8922) 
v_mul_f32 v35, v18, v31 // (op 8922) 

v_add_f32 v28, v32, v36 // (op 8923) 
v_add_f32 v29, v33, v37 // (op 8923) 
v_add_f32 v30, v34, v38 // (op 8923) 
v_add_f32 v31, v35, v39 // (op 8923) 




s_lshl_b32 s52, s24, 7 // (op 8909) 
s_mul_i32 s53, 3, s52 // (op 8909) 
s_add_u32 s53, s53, 384 // (op 8909) 

v_add_u32 v20, v27, s53 // (op 8909) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 8909) Store value




v_add_u32 v20, 256, v22 // (op 8833) 

s_nop 0
buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 8833) Load value

v_accvgpr_read v32, a32 // (op 8821) 
v_accvgpr_read v33, a33 // (op 8821) 
v_accvgpr_read v34, a34 // (op 8821) 
v_accvgpr_read v35, a35 // (op 8821) 
v_mul_f32 v36, v19, v32 // (op 8821) 
v_mul_f32 v37, v19, v33 // (op 8821) 
v_mul_f32 v38, v19, v34 // (op 8821) 
v_mul_f32 v39, v19, v35 // (op 8821) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 8832) 
v_mul_f32 v33, v18, v29 // (op 8832) 
v_mul_f32 v34, v18, v30 // (op 8832) 
v_mul_f32 v35, v18, v31 // (op 8832) 

v_add_f32 v28, v32, v36 // (op 8822) 
v_add_f32 v29, v33, v37 // (op 8822) 
v_add_f32 v30, v34, v38 // (op 8822) 
v_add_f32 v31, v35, v39 // (op 8822) 




v_add_u32 v20, 256, v27 // (op 8824) 

buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 8824) Store value




s_lshl_b32 s52, s4, 7 // (op 8837) 
s_add_u32 s52, s52, 256 // (op 8837) 
v_add_u32 v20, v22, s52 // (op 8837) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 8837) Load value

v_accvgpr_read v32, a36 // (op 8847) 
v_accvgpr_read v33, a37 // (op 8847) 
v_accvgpr_read v34, a38 // (op 8847) 
v_accvgpr_read v35, a39 // (op 8847) 
v_mul_f32 v36, v19, v32 // (op 8847) 
v_mul_f32 v37, v19, v33 // (op 8847) 
v_mul_f32 v38, v19, v34 // (op 8847) 
v_mul_f32 v39, v19, v35 // (op 8847) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 8835) 
v_mul_f32 v33, v18, v29 // (op 8835) 
v_mul_f32 v34, v18, v30 // (op 8835) 
v_mul_f32 v35, v18, v31 // (op 8835) 

v_add_f32 v28, v32, v36 // (op 8836) 
v_add_f32 v29, v33, v37 // (op 8836) 
v_add_f32 v30, v34, v38 // (op 8836) 
v_add_f32 v31, v35, v39 // (op 8836) 




s_lshl_b32 s52, s24, 7 // (op 8826) 
s_add_u32 s52, s52, 256 // (op 8826) 
v_add_u32 v20, v27, s52 // (op 8826) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 8826) Store value




s_lshl_b32 s52, s4, 8 // (op 8841) 
s_add_u32 s52, s52, 256 // (op 8841) 
v_add_u32 v20, v22, s52 // (op 8841) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 8841) Load value

v_accvgpr_read v32, a40 // (op 8849) 
v_accvgpr_read v33, a41 // (op 8849) 
v_accvgpr_read v34, a42 // (op 8849) 
v_accvgpr_read v35, a43 // (op 8849) 
v_mul_f32 v36, v19, v32 // (op 8849) 
v_mul_f32 v37, v19, v33 // (op 8849) 
v_mul_f32 v38, v19, v34 // (op 8849) 
v_mul_f32 v39, v19, v35 // (op 8849) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 8839) 
v_mul_f32 v33, v18, v29 // (op 8839) 
v_mul_f32 v34, v18, v30 // (op 8839) 
v_mul_f32 v35, v18, v31 // (op 8839) 

v_add_f32 v28, v32, v36 // (op 8840) 
v_add_f32 v29, v33, v37 // (op 8840) 
v_add_f32 v30, v34, v38 // (op 8840) 
v_add_f32 v31, v35, v39 // (op 8840) 




s_lshl_b32 s52, s24, 8 // (op 8828) 
s_add_u32 s52, s52, 256 // (op 8828) 
v_add_u32 v20, v27, s52 // (op 8828) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 8828) Store value




s_lshl_b32 s52, s4, 7 // (op 8845) 
s_mul_i32 s53, 3, s52 // (op 8845) 
s_add_u32 s53, s53, 256 // (op 8845) 

v_add_u32 v20, v22, s53 // (op 8845) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 8845) Load value

v_accvgpr_read v32, a44 // (op 8851) 
v_accvgpr_read v33, a45 // (op 8851) 
v_accvgpr_read v34, a46 // (op 8851) 
v_accvgpr_read v35, a47 // (op 8851) 
v_mul_f32 v36, v19, v32 // (op 8851) 
v_mul_f32 v37, v19, v33 // (op 8851) 
v_mul_f32 v38, v19, v34 // (op 8851) 
v_mul_f32 v39, v19, v35 // (op 8851) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 8843) 
v_mul_f32 v33, v18, v29 // (op 8843) 
v_mul_f32 v34, v18, v30 // (op 8843) 
v_mul_f32 v35, v18, v31 // (op 8843) 

v_add_f32 v28, v32, v36 // (op 8844) 
v_add_f32 v29, v33, v37 // (op 8844) 
v_add_f32 v30, v34, v38 // (op 8844) 
v_add_f32 v31, v35, v39 // (op 8844) 




s_lshl_b32 s52, s24, 7 // (op 8830) 
s_mul_i32 s53, 3, s52 // (op 8830) 
s_add_u32 s53, s53, 256 // (op 8830) 

v_add_u32 v20, v27, s53 // (op 8830) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 8830) Store value




v_add_u32 v20, 128, v22 // (op 8754) 

s_nop 0
buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 8754) Load value

v_accvgpr_read v32, a16 // (op 8742) 
v_accvgpr_read v33, a17 // (op 8742) 
v_accvgpr_read v34, a18 // (op 8742) 
v_accvgpr_read v35, a19 // (op 8742) 
v_mul_f32 v36, v19, v32 // (op 8742) 
v_mul_f32 v37, v19, v33 // (op 8742) 
v_mul_f32 v38, v19, v34 // (op 8742) 
v_mul_f32 v39, v19, v35 // (op 8742) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 8753) 
v_mul_f32 v33, v18, v29 // (op 8753) 
v_mul_f32 v34, v18, v30 // (op 8753) 
v_mul_f32 v35, v18, v31 // (op 8753) 

v_add_f32 v28, v32, v36 // (op 8743) 
v_add_f32 v29, v33, v37 // (op 8743) 
v_add_f32 v30, v34, v38 // (op 8743) 
v_add_f32 v31, v35, v39 // (op 8743) 




v_add_u32 v20, 128, v27 // (op 8745) 

buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 8745) Store value




s_lshl_b32 s52, s4, 7 // (op 8758) 
s_add_u32 s52, s52, 128 // (op 8758) 
v_add_u32 v20, v22, s52 // (op 8758) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 8758) Load value

v_accvgpr_read v32, a20 // (op 8768) 
v_accvgpr_read v33, a21 // (op 8768) 
v_accvgpr_read v34, a22 // (op 8768) 
v_accvgpr_read v35, a23 // (op 8768) 
v_mul_f32 v36, v19, v32 // (op 8768) 
v_mul_f32 v37, v19, v33 // (op 8768) 
v_mul_f32 v38, v19, v34 // (op 8768) 
v_mul_f32 v39, v19, v35 // (op 8768) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 8756) 
v_mul_f32 v33, v18, v29 // (op 8756) 
v_mul_f32 v34, v18, v30 // (op 8756) 
v_mul_f32 v35, v18, v31 // (op 8756) 

v_add_f32 v28, v32, v36 // (op 8757) 
v_add_f32 v29, v33, v37 // (op 8757) 
v_add_f32 v30, v34, v38 // (op 8757) 
v_add_f32 v31, v35, v39 // (op 8757) 




s_lshl_b32 s52, s24, 7 // (op 8747) 
s_add_u32 s52, s52, 128 // (op 8747) 
v_add_u32 v20, v27, s52 // (op 8747) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 8747) Store value




s_lshl_b32 s52, s4, 8 // (op 8762) 
s_add_u32 s52, s52, 128 // (op 8762) 
v_add_u32 v20, v22, s52 // (op 8762) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 8762) Load value

v_accvgpr_read v32, a24 // (op 8770) 
v_accvgpr_read v33, a25 // (op 8770) 
v_accvgpr_read v34, a26 // (op 8770) 
v_accvgpr_read v35, a27 // (op 8770) 
v_mul_f32 v36, v19, v32 // (op 8770) 
v_mul_f32 v37, v19, v33 // (op 8770) 
v_mul_f32 v38, v19, v34 // (op 8770) 
v_mul_f32 v39, v19, v35 // (op 8770) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 8760) 
v_mul_f32 v33, v18, v29 // (op 8760) 
v_mul_f32 v34, v18, v30 // (op 8760) 
v_mul_f32 v35, v18, v31 // (op 8760) 

v_add_f32 v28, v32, v36 // (op 8761) 
v_add_f32 v29, v33, v37 // (op 8761) 
v_add_f32 v30, v34, v38 // (op 8761) 
v_add_f32 v31, v35, v39 // (op 8761) 




s_lshl_b32 s52, s24, 8 // (op 8749) 
s_add_u32 s52, s52, 128 // (op 8749) 
v_add_u32 v20, v27, s52 // (op 8749) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 8749) Store value




s_lshl_b32 s52, s4, 7 // (op 8766) 
s_mul_i32 s53, 3, s52 // (op 8766) 
s_add_u32 s53, s53, 128 // (op 8766) 

v_add_u32 v20, v22, s53 // (op 8766) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 8766) Load value

v_accvgpr_read v32, a28 // (op 8772) 
v_accvgpr_read v33, a29 // (op 8772) 
v_accvgpr_read v34, a30 // (op 8772) 
v_accvgpr_read v35, a31 // (op 8772) 
v_mul_f32 v36, v19, v32 // (op 8772) 
v_mul_f32 v37, v19, v33 // (op 8772) 
v_mul_f32 v38, v19, v34 // (op 8772) 
v_mul_f32 v39, v19, v35 // (op 8772) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 8764) 
v_mul_f32 v33, v18, v29 // (op 8764) 
v_mul_f32 v34, v18, v30 // (op 8764) 
v_mul_f32 v35, v18, v31 // (op 8764) 

v_add_f32 v28, v32, v36 // (op 8765) 
v_add_f32 v29, v33, v37 // (op 8765) 
v_add_f32 v30, v34, v38 // (op 8765) 
v_add_f32 v31, v35, v39 // (op 8765) 




s_lshl_b32 s52, s24, 7 // (op 8751) 
s_mul_i32 s53, 3, s52 // (op 8751) 
s_add_u32 s53, s53, 128 // (op 8751) 

v_add_u32 v20, v27, s53 // (op 8751) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 8751) Store value




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




s_lshl_b32 s52, s4, 7 // (op 8668) 
v_add_u32 v20, v22, s52 // (op 8668) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 8668) Load value

v_accvgpr_read v32, a4 // (op 8665) 
v_accvgpr_read v33, a5 // (op 8665) 
v_accvgpr_read v34, a6 // (op 8665) 
v_accvgpr_read v35, a7 // (op 8665) 
v_mul_f32 v36, v19, v32 // (op 8665) 
v_mul_f32 v37, v19, v33 // (op 8665) 
v_mul_f32 v38, v19, v34 // (op 8665) 
v_mul_f32 v39, v19, v35 // (op 8665) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 8669) 
v_mul_f32 v33, v18, v29 // (op 8669) 
v_mul_f32 v34, v18, v30 // (op 8669) 
v_mul_f32 v35, v18, v31 // (op 8669) 

v_add_f32 v28, v32, v36 // (op 8666) 
v_add_f32 v29, v33, v37 // (op 8666) 
v_add_f32 v30, v34, v38 // (op 8666) 
v_add_f32 v31, v35, v39 // (op 8666) 




s_lshl_b32 s52, s24, 7 // (op 8667) 
v_add_u32 v20, v27, s52 // (op 8667) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 8667) Store value




s_lshl_b32 s52, s4, 8 // (op 8680) 
v_add_u32 v20, v22, s52 // (op 8680) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 8680) Load value

v_accvgpr_read v32, a8 // (op 8677) 
v_accvgpr_read v33, a9 // (op 8677) 
v_accvgpr_read v34, a10 // (op 8677) 
v_accvgpr_read v35, a11 // (op 8677) 
v_mul_f32 v36, v19, v32 // (op 8677) 
v_mul_f32 v37, v19, v33 // (op 8677) 
v_mul_f32 v38, v19, v34 // (op 8677) 
v_mul_f32 v39, v19, v35 // (op 8677) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 8681) 
v_mul_f32 v33, v18, v29 // (op 8681) 
v_mul_f32 v34, v18, v30 // (op 8681) 
v_mul_f32 v35, v18, v31 // (op 8681) 

v_add_f32 v28, v32, v36 // (op 8678) 
v_add_f32 v29, v33, v37 // (op 8678) 
v_add_f32 v30, v34, v38 // (op 8678) 
v_add_f32 v31, v35, v39 // (op 8678) 




s_lshl_b32 s52, s24, 8 // (op 8679) 
v_add_u32 v20, v27, s52 // (op 8679) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 8679) Store value




s_lshl_b32 s52, s4, 7 // (op 8692) 
s_mul_i32 s53, 3, s52 // (op 8692) 

v_add_u32 v20, v22, s53 // (op 8692) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 8692) Load value

v_accvgpr_read v32, a12 // (op 8689) 
v_accvgpr_read v33, a13 // (op 8689) 
v_accvgpr_read v34, a14 // (op 8689) 
v_accvgpr_read v35, a15 // (op 8689) 
v_mul_f32 v36, v19, v32 // (op 8689) 
v_mul_f32 v37, v19, v33 // (op 8689) 
v_mul_f32 v38, v19, v34 // (op 8689) 
v_mul_f32 v39, v19, v35 // (op 8689) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 8693) 
v_mul_f32 v33, v18, v29 // (op 8693) 
v_mul_f32 v34, v18, v30 // (op 8693) 
v_mul_f32 v35, v18, v31 // (op 8693) 

v_add_f32 v28, v32, v36 // (op 8690) 
v_add_f32 v29, v33, v37 // (op 8690) 
v_add_f32 v30, v34, v38 // (op 8690) 
v_add_f32 v31, v35, v39 // (op 8690) 




s_lshl_b32 s52, s24, 7 // (op 8691) 
s_mul_i32 s53, 3, s52 // (op 8691) 

v_add_u32 v20, v27, s53 // (op 8691) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 8691) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_133:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_373 // (op 373) Condition: Done, jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_373
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_373:
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_373:


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
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_SKStreamTileLoop_286 // (op 286) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_SKStreamTileLoop_286 if true)
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_286:

















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
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_441 // (op 441) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_441 if false)
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_DPStreamTileLoop_441:
v_lshrrev_b32 v0, 7, v1 // (op 18365) 
s_mul_i32 s15, s16, s50 // (op 18365) 
s_add_u32 s15, s15, s0 // (op 18365) 
v_and_b32 v3, 63, v1 // (op 18365) 
s_add_u32 s6, s15, s9 // (op 18365) 

v_lshrrev_b32 v4, 4, v3 // (op 18365) 
s_mul_hi_u32 s15, s6, s13 // (op 18365) 
v_lshlrev_b32 v5, 2, v4 // (op 18365) 

s_sub_u32 s7, s6, s15 // (op 18365) 
v_lshl_add_u32 v4, v0, 4, v5 // (op 18365) 


s_lshr_b32 s8, s7, 1 // (op 18365) 

v_lshrrev_b32 v0, 2, v3 // (op 18365) 

s_add_u32 s7, s8, s15 // (op 18365) 


v_and_b32 v3, 3, v0 // (op 18365) 

s_lshr_b32 s8, s7, s11 // (op 18365) 

v_and_b32 v0, 3, v1 // (op 18365) 
s_and_b32 s7, s11, 2147483648 // (op 18365) 
v_lshl_add_u32 v5, v3, 2, v0 // (op 18365) 


s_cmp_eq_i32 s7, 0 // (op 18365) 
s_mov_b32 s15, scc // (op 18365) 

v_lshrrev_b32 v0, 6, v1 // (op 18365) 
s_cmp_eq_i32 s15, 0 // (op 18365) 
s_cselect_b32 s7, s6, s8 // (op 18365) 



v_and_b32 v3, 1, v0 // (op 18365) 

s_mul_hi_u32 s6, s7, s12 // (op 18365) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 18365) 


s_sub_u32 s8, s7, s6 // (op 18365) 
s_lshr_b32 s15, s8, 1 // (op 18365) 

s_add_u32 s8, s15, s6 // (op 18365) 


s_lshr_b32 s6, s8, s10 // (op 18365) 

s_and_b32 s8, s10, 2147483648 // (op 18365) 
s_cmp_eq_i32 s8, 0 // (op 18365) 
s_mov_b32 s15, scc // (op 18365) 

s_cmp_eq_i32 s15, 0 // (op 18365) 
s_cselect_b32 s8, s7, s6 // (op 18365) 


s_mul_i32 s6, s8, s14 // (op 18365) 
v_lshl_add_u32 v3, s8, 7, v4 // (op 18365) 


s_sub_u32 s8, s7, s6 // (op 18365) 


v_lshl_add_u32 v4, s8, 7, v0 // (op 18365) 


v_mul_lo_u32 v0, v4, s4 // (op 18365) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 18365) 


s_mul_i32 s19, s4, s23 // (op 18367) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 18367) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 18367) least significant
s_mul_hi_u32 s6, s4, s22 // (op 18367) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18367) most significant: sum
s_add_u32 s19, s19, s6 // (op 18367) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 18367) 

s_mov_b32 s56, s20 // (op 18367) reg expression
s_mov_b32 s57, s21 // (op 18367) reg expression
s_mov_b32 s58, s28 // (op 18367) reg expression
s_mov_b32 s59, 131072 // (op 18367) call()

v_lshrrev_b32 v0, 7, v1 // (op 18719) 
s_mul_i32 s15, s16, s50 // (op 18719) 
s_add_u32 s15, s15, s0 // (op 18719) 
v_and_b32 v3, 63, v1 // (op 18719) 
s_add_u32 s6, s15, s9 // (op 18719) 

v_lshrrev_b32 v5, 4, v3 // (op 18719) 
s_mul_hi_u32 s15, s6, s13 // (op 18719) 
v_lshlrev_b32 v14, 2, v5 // (op 18719) 

s_sub_u32 s7, s6, s15 // (op 18719) 
v_lshl_add_u32 v5, v0, 4, v14 // (op 18719) 


s_lshr_b32 s8, s7, 1 // (op 18719) 

v_lshrrev_b32 v0, 2, v3 // (op 18719) 

s_add_u32 s7, s8, s15 // (op 18719) 


v_and_b32 v3, 3, v0 // (op 18719) 

s_lshr_b32 s8, s7, s11 // (op 18719) 

v_and_b32 v0, 3, v1 // (op 18719) 
s_and_b32 s7, s11, 2147483648 // (op 18719) 
v_lshl_add_u32 v14, v3, 2, v0 // (op 18719) 


s_cmp_eq_i32 s7, 0 // (op 18719) 
s_mov_b32 s15, scc // (op 18719) 

v_lshrrev_b32 v0, 6, v1 // (op 18719) 
s_cmp_eq_i32 s15, 0 // (op 18719) 
s_cselect_b32 s7, s6, s8 // (op 18719) 



v_and_b32 v3, 1, v0 // (op 18719) 

s_mul_hi_u32 s6, s7, s12 // (op 18719) 
v_lshl_add_u32 v0, v3, 4, v14 // (op 18719) 


s_sub_u32 s8, s7, s6 // (op 18719) 
s_lshr_b32 s15, s8, 1 // (op 18719) 

s_add_u32 s8, s15, s6 // (op 18719) 


s_lshr_b32 s6, s8, s10 // (op 18719) 

s_and_b32 s8, s10, 2147483648 // (op 18719) 
s_cmp_eq_i32 s8, 0 // (op 18719) 
s_mov_b32 s15, scc // (op 18719) 

s_cmp_eq_i32 s15, 0 // (op 18719) 
s_cselect_b32 s8, s7, s6 // (op 18719) 


s_mul_i32 s6, s8, s14 // (op 18719) 
v_lshl_add_u32 v3, s8, 7, v5 // (op 18719) 


s_sub_u32 s8, s7, s6 // (op 18719) 


v_lshl_add_u32 v5, s8, 7, v0 // (op 18719) 


v_mul_lo_u32 v0, v5, s24 // (op 18719) 

v_add_lshl_u32 v5, v3, v0, 2 // (op 18719) 


s_mul_i32 s19, s24, s23 // (op 18721) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 18721) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 18721) least significant
s_mul_hi_u32 s6, s24, s22 // (op 18721) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18721) most significant: sum
s_add_u32 s19, s19, s6 // (op 18721) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 18721) 

s_mov_b32 s60, s26 // (op 18721) reg expression
s_mov_b32 s61, s27 // (op 18721) reg expression
s_mov_b32 s62, s28 // (op 18721) reg expression
s_mov_b32 s63, 131072 // (op 18721) call()

v_and_b32 v0, 63, v1 // (op 18798) 
v_and_b32 v3, 48, v1 // (op 18798) 
v_sub_u32 v14, v0, v3 // (op 18798) 

v_lshrrev_b32 v3, 4, v0 // (op 18798) 

v_lshl_add_u32 v0, v3, 5, v14 // (op 18798) 


v_lshrrev_b32 v3, 7, v1 // (op 18798) 
v_lshlrev_b32 v14, 4, v3 // (op 18798) 

v_lshl_add_u32 v3, v0, 7, v14 // (op 18798) 


v_lshlrev_b32 v0, 2, v3 // (op 18798) 

v_lshrrev_b32 v3, 3, v0 // (op 18798) 

v_lshrrev_b32 v0, 6, v1 // (op 18873) 
v_and_b32 v14, 1, v0 // (op 18873) 

v_and_b32 v0, 15, v1 // (op 18873) 
v_lshl_add_u32 v15, v14, 4, v0 // (op 18873) 


v_and_b32 v0, 48, v1 // (op 18873) 
v_lshl_add_u32 v14, v15, 7, v0 // (op 18873) 


v_lshrrev_b32 v0, 7, v1 // (op 18949) 
v_and_b32 v15, 15, v1 // (op 18949) 
v_lshl_add_u32 v16, v0, 4, v15 // (op 18949) 


v_and_b32 v0, 63, v1 // (op 18949) 
v_lshrrev_b32 v15, 4, v0 // (op 18949) 

v_lshl_add_u32 v0, v15, 7, v16 // (op 18949) 


v_lshrrev_b32 v15, 6, v1 // (op 19025) 
v_and_b32 v16, 1, v15 // (op 19025) 

v_and_b32 v15, 15, v1 // (op 19025) 
v_lshl_add_u32 v17, v16, 4, v15 // (op 19025) 


v_and_b32 v15, 63, v1 // (op 19025) 
v_lshrrev_b32 v16, 4, v15 // (op 19025) 

v_lshl_add_u32 v15, v17, 2, v16 // (op 19025) 


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
v_accvgpr_write a32, 0.00000 // (op 2000) call()
v_accvgpr_write a33, 0.00000 // (op 2000) call()
v_accvgpr_write a34, 0.00000 // (op 2000) call()
v_accvgpr_write a35, 0.00000 // (op 2000) call()
v_accvgpr_write a36, 0.00000 // (op 2309) call()
v_accvgpr_write a37, 0.00000 // (op 2309) call()
v_accvgpr_write a38, 0.00000 // (op 2309) call()
v_accvgpr_write a39, 0.00000 // (op 2309) call()
v_accvgpr_write a40, 0.00000 // (op 2310) call()
v_accvgpr_write a41, 0.00000 // (op 2310) call()
v_accvgpr_write a42, 0.00000 // (op 2310) call()
v_accvgpr_write a43, 0.00000 // (op 2310) call()
v_accvgpr_write a44, 0.00000 // (op 2311) call()
v_accvgpr_write a45, 0.00000 // (op 2311) call()
v_accvgpr_write a46, 0.00000 // (op 2311) call()
v_accvgpr_write a47, 0.00000 // (op 2311) call()
v_accvgpr_write a48, 0.00000 // (op 2770) call()
v_accvgpr_write a49, 0.00000 // (op 2770) call()
v_accvgpr_write a50, 0.00000 // (op 2770) call()
v_accvgpr_write a51, 0.00000 // (op 2770) call()
v_accvgpr_write a52, 0.00000 // (op 3079) call()
v_accvgpr_write a53, 0.00000 // (op 3079) call()
v_accvgpr_write a54, 0.00000 // (op 3079) call()
v_accvgpr_write a55, 0.00000 // (op 3079) call()
v_accvgpr_write a56, 0.00000 // (op 3080) call()
v_accvgpr_write a57, 0.00000 // (op 3080) call()
v_accvgpr_write a58, 0.00000 // (op 3080) call()
v_accvgpr_write a59, 0.00000 // (op 3080) call()
v_accvgpr_write a60, 0.00000 // (op 3081) call()
v_accvgpr_write a61, 0.00000 // (op 3081) call()
v_accvgpr_write a62, 0.00000 // (op 3081) call()
v_accvgpr_write a63, 0.00000 // (op 3081) call()
v_and_b32 v16, -4, v1 // (op 17845) 
s_mul_i32 s15, s16, s50 // (op 17845) 
s_add_u32 s15, s15, s0 // (op 17845) 
v_sub_u32 v17, v1, v16 // (op 17845) 

s_add_u32 s6, s15, s9 // (op 17845) 

v_lshlrev_b32 v16, 5, v17 // (op 17845) 

s_mul_hi_u32 s15, s6, s13 // (op 17845) 
v_lshrrev_b32 v17, 2, v1 // (op 17845) 
s_sub_u32 s7, s6, s15 // (op 17845) 
s_lshr_b32 s8, s7, 1 // (op 17845) 

s_add_u32 s7, s8, s15 // (op 17845) 


s_lshr_b32 s8, s7, s11 // (op 17845) 

s_and_b32 s7, s11, 2147483648 // (op 17845) 
s_cmp_eq_i32 s7, 0 // (op 17845) 
s_mov_b32 s15, scc // (op 17845) 

s_cmp_eq_i32 s15, 0 // (op 17845) 
s_cselect_b32 s7, s6, s8 // (op 17845) 


s_mul_hi_u32 s8, s7, s12 // (op 17845) 
s_sub_u32 s15, s7, s8 // (op 17845) 
s_lshr_b32 s18, s15, 1 // (op 17845) 

s_add_u32 s15, s18, s8 // (op 17845) 


s_lshr_b32 s8, s15, s10 // (op 17845) 

s_and_b32 s15, s10, 2147483648 // (op 17845) 
s_cmp_eq_i32 s15, 0 // (op 17845) 
s_mov_b32 s18, scc // (op 17845) 

s_cmp_eq_i32 s18, 0 // (op 17845) 
s_cselect_b32 s15, s7, s8 // (op 17845) 


v_lshl_add_u32 v20, s15, 7, v16 // (op 17845) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 17845) 
v_mov_b32 v23, 0 // (op 17845) convert
v_mov_b32 v22, v20 // (op 17845) convert

s_mul_i32 s8, s7, s18 // (op 17845) 


s_sub_u32 s7, s6, s8 // (op 17845) 


v_lshl_add_u32 v16, s7, 7, v17 // (op 17845) 


v_mul_lo_u32 v21, v16, s31 // (op 17845) most significant: low of low * high
v_mul_hi_u32 v17, v16, s30 // (op 17845) most significant: high of low * low
v_mul_lo_u32 v20, v16, s30 // (op 17845) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 17845) most significant: sum


v_add_co_u32 v16, s[18:19], v22, v20 // (op 17845) least significant half
v_addc_co_u32 v17, s[18:19], v23, v21, s[18:19] // (op 17845) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 17845) 


v_lshrrev_b64 v[20:21], 3, v[16:17] // (op 17845) 

s_mul_i32 s19, s30, s3 // (op 17847) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 17847) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 17847) least significant
s_mul_hi_u32 s6, s30, s2 // (op 17847) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 17847) most significant: sum
s_add_u32 s19, s19, s6 // (op 17847) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 17847) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 17847) 

s_mov_b32 s68, s32 // (op 17847) reg expression
s_mov_b32 s69, s33 // (op 17847) reg expression
s_mov_b32 s70, s18 // (op 17847) reg expression
s_mov_b32 s71, 131072 // (op 17847) call()

v_and_b32 v16, -4, v1 // (op 17864) 
s_mul_i32 s15, s16, s50 // (op 17864) 
s_add_u32 s15, s15, s0 // (op 17864) 
v_sub_u32 v17, v1, v16 // (op 17864) 

s_add_u32 s6, s15, s9 // (op 17864) 

v_lshlrev_b32 v16, 5, v17 // (op 17864) 

s_mul_hi_u32 s15, s6, s13 // (op 17864) 
v_lshrrev_b32 v17, 2, v1 // (op 17864) 
s_sub_u32 s7, s6, s15 // (op 17864) 
s_lshr_b32 s8, s7, 1 // (op 17864) 

s_add_u32 s7, s8, s15 // (op 17864) 


s_lshr_b32 s8, s7, s11 // (op 17864) 

s_and_b32 s7, s11, 2147483648 // (op 17864) 
s_cmp_eq_i32 s7, 0 // (op 17864) 
s_mov_b32 s15, scc // (op 17864) 

s_cmp_eq_i32 s15, 0 // (op 17864) 
s_cselect_b32 s7, s6, s8 // (op 17864) 


s_mul_hi_u32 s8, s7, s12 // (op 17864) 
s_sub_u32 s15, s7, s8 // (op 17864) 
s_lshr_b32 s18, s15, 1 // (op 17864) 

s_add_u32 s15, s18, s8 // (op 17864) 


s_lshr_b32 s8, s15, s10 // (op 17864) 

s_and_b32 s15, s10, 2147483648 // (op 17864) 
s_cmp_eq_i32 s15, 0 // (op 17864) 
s_mov_b32 s18, scc // (op 17864) 

s_cmp_eq_i32 s18, 0 // (op 17864) 
s_cselect_b32 s15, s7, s8 // (op 17864) 


v_lshl_add_u32 v22, s15, 7, v16 // (op 17864) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 17864) 
v_mov_b32 v25, 0 // (op 17864) convert
v_mov_b32 v24, v22 // (op 17864) convert

s_mul_i32 s8, s7, s18 // (op 17864) 


s_sub_u32 s7, s6, s8 // (op 17864) 


v_lshl_add_u32 v16, s7, 7, v17 // (op 17864) 


v_mul_lo_u32 v23, v16, s31 // (op 17864) most significant: low of low * high
v_mul_hi_u32 v17, v16, s30 // (op 17864) most significant: high of low * low
v_mul_lo_u32 v22, v16, s30 // (op 17864) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 17864) most significant: sum


v_add_co_u32 v16, s[18:19], v24, v22 // (op 17864) least significant half
v_addc_co_u32 v17, s[18:19], v25, v23, s[18:19] // (op 17864) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 17864) 


v_lshrrev_b64 v[22:23], 3, v[16:17] // (op 17864) 

s_mul_i32 s19, s30, s3 // (op 17866) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 17866) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 17866) least significant
s_mul_hi_u32 s6, s30, s2 // (op 17866) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 17866) most significant: sum
s_add_u32 s19, s19, s6 // (op 17866) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 17866) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 17866) 

s_mov_b32 s68, s32 // (op 17866) reg expression
s_mov_b32 s69, s33 // (op 17866) reg expression
s_mov_b32 s70, s18 // (op 17866) reg expression
s_mov_b32 s71, 131072 // (op 17866) call()

v_lshrrev_b32 v16, 3, v1 // (op 17991) 
s_mul_i32 s15, s16, s50 // (op 17991) 
s_add_u32 s15, s15, s0 // (op 17991) 
v_and_b32 v17, -8, v1 // (op 17991) 
s_add_u32 s6, s15, s9 // (op 17991) 

v_sub_u32 v24, v1, v17 // (op 17991) 

s_mul_hi_u32 s15, s6, s13 // (op 17991) 
v_lshlrev_b32 v17, 4, v24 // (op 17991) 

s_sub_u32 s7, s6, s15 // (op 17991) 
s_lshr_b32 s8, s7, 1 // (op 17991) 

s_add_u32 s7, s8, s15 // (op 17991) 


s_lshr_b32 s8, s7, s11 // (op 17991) 

s_and_b32 s7, s11, 2147483648 // (op 17991) 
s_cmp_eq_i32 s7, 0 // (op 17991) 
s_mov_b32 s15, scc // (op 17991) 

s_cmp_eq_i32 s15, 0 // (op 17991) 
s_cselect_b32 s7, s6, s8 // (op 17991) 


s_mul_hi_u32 s8, s7, s12 // (op 17991) 
s_sub_u32 s15, s7, s8 // (op 17991) 
s_lshr_b32 s18, s15, 1 // (op 17991) 

s_add_u32 s15, s18, s8 // (op 17991) 


s_lshr_b32 s8, s15, s10 // (op 17991) 

s_and_b32 s15, s10, 2147483648 // (op 17991) 
s_cmp_eq_i32 s15, 0 // (op 17991) 
s_mov_b32 s18, scc // (op 17991) 

s_cmp_eq_i32 s18, 0 // (op 17991) 
s_cselect_b32 s15, s7, s8 // (op 17991) 


s_mul_i32 s8, s15, s14 // (op 17991) 

s_sub_u32 s15, s7, s8 // (op 17991) 

v_lshl_add_u32 v24, s15, 7, v16 // (op 17991) 


v_mul_lo_u32 v27, v24, s35 // (op 17991) most significant: low of low * high
v_mul_hi_u32 v16, v24, s34 // (op 17991) most significant: high of low * low
v_mul_lo_u32 v26, v24, s34 // (op 17991) least significant: low of low * low
v_add_u32 v27, v27, v16 // (op 17991) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 17991) 
s_mul_i32 s8, s7, s18 // (op 17991) 


s_sub_u32 s7, s6, s8 // (op 17991) 


v_lshl_add_u32 v16, s7, 7, v17 // (op 17991) 


v_mov_b32 v25, 0 // (op 17991) convert
v_mov_b32 v24, v16 // (op 17991) convert

v_add_co_u32 v16, s[18:19], v24, v26 // (op 17991) least significant half
v_addc_co_u32 v17, s[18:19], v25, v27, s[18:19] // (op 17991) most significant half



s_mul_i32 s19, s34, s23 // (op 17993) most significant: low of low * high
s_mul_i32 s15, s35, s22 // (op 17993) most significant: low of high * low
s_mul_i32 s18, s34, s22 // (op 17993) least significant
s_mul_hi_u32 s6, s34, s22 // (op 17993) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 17993) most significant: sum
s_add_u32 s19, s19, s6 // (op 17993) most significant: sum


s_mov_b32 s72, s36 // (op 17993) reg expression
s_mov_b32 s73, s37 // (op 17993) reg expression
s_mov_b32 s74, s18 // (op 17993) reg expression
s_mov_b32 s75, 131072 // (op 17993) call()

v_lshrrev_b32 v24, 3, v1 // (op 18010) 
s_mul_i32 s15, s16, s50 // (op 18010) 
s_add_u32 s15, s15, s0 // (op 18010) 
v_and_b32 v25, -8, v1 // (op 18010) 
s_add_u32 s6, s15, s9 // (op 18010) 

v_sub_u32 v26, v1, v25 // (op 18010) 

s_mul_hi_u32 s15, s6, s13 // (op 18010) 
v_lshlrev_b32 v25, 4, v26 // (op 18010) 

s_sub_u32 s7, s6, s15 // (op 18010) 
s_lshr_b32 s8, s7, 1 // (op 18010) 

s_add_u32 s7, s8, s15 // (op 18010) 


s_lshr_b32 s8, s7, s11 // (op 18010) 

s_and_b32 s7, s11, 2147483648 // (op 18010) 
s_cmp_eq_i32 s7, 0 // (op 18010) 
s_mov_b32 s15, scc // (op 18010) 

s_cmp_eq_i32 s15, 0 // (op 18010) 
s_cselect_b32 s7, s6, s8 // (op 18010) 


s_mul_hi_u32 s8, s7, s12 // (op 18010) 
s_sub_u32 s15, s7, s8 // (op 18010) 
s_lshr_b32 s18, s15, 1 // (op 18010) 

s_add_u32 s15, s18, s8 // (op 18010) 


s_lshr_b32 s8, s15, s10 // (op 18010) 

s_and_b32 s15, s10, 2147483648 // (op 18010) 
s_cmp_eq_i32 s15, 0 // (op 18010) 
s_mov_b32 s18, scc // (op 18010) 

s_cmp_eq_i32 s18, 0 // (op 18010) 
s_cselect_b32 s15, s7, s8 // (op 18010) 


s_mul_i32 s8, s15, s14 // (op 18010) 

s_sub_u32 s15, s7, s8 // (op 18010) 

v_lshl_add_u32 v26, s15, 7, v24 // (op 18010) 


v_mul_lo_u32 v29, v26, s35 // (op 18010) most significant: low of low * high
v_mul_hi_u32 v24, v26, s34 // (op 18010) most significant: high of low * low
v_mul_lo_u32 v28, v26, s34 // (op 18010) least significant: low of low * low
v_add_u32 v29, v29, v24 // (op 18010) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 18010) 
s_mul_i32 s8, s7, s18 // (op 18010) 


s_sub_u32 s7, s6, s8 // (op 18010) 


v_lshl_add_u32 v24, s7, 7, v25 // (op 18010) 


v_mov_b32 v27, 0 // (op 18010) convert
v_mov_b32 v26, v24 // (op 18010) convert

v_add_co_u32 v24, s[18:19], v26, v28 // (op 18010) least significant half
v_addc_co_u32 v25, s[18:19], v27, v29, s[18:19] // (op 18010) most significant half



s_mul_i32 s19, s34, s23 // (op 18012) most significant: low of low * high
s_mul_i32 s15, s35, s22 // (op 18012) most significant: low of high * low
s_mul_i32 s18, s34, s22 // (op 18012) least significant
s_mul_hi_u32 s6, s34, s22 // (op 18012) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18012) most significant: sum
s_add_u32 s19, s19, s6 // (op 18012) most significant: sum


s_mov_b32 s72, s36 // (op 18012) reg expression
s_mov_b32 s73, s37 // (op 18012) reg expression
s_mov_b32 s74, s18 // (op 18012) reg expression
s_mov_b32 s75, 131072 // (op 18012) call()

v_and_b32 v26, -128, v1 // (op 18119) 
s_mul_i32 s15, s16, s50 // (op 18119) 
s_add_u32 s15, s15, s0 // (op 18119) 
v_sub_u32 v27, v1, v26 // (op 18119) 

s_add_u32 s6, s15, s9 // (op 18119) 

v_lshrrev_b32 v26, 7, v1 // (op 18119) 
s_mul_hi_u32 s15, s6, s13 // (op 18119) 
v_lshlrev_b32 v28, 1, v26 // (op 18119) 

s_sub_u32 s7, s6, s15 // (op 18119) 
s_lshr_b32 s8, s7, 1 // (op 18119) 

s_add_u32 s7, s8, s15 // (op 18119) 


s_lshr_b32 s8, s7, s11 // (op 18119) 

s_and_b32 s7, s11, 2147483648 // (op 18119) 
s_cmp_eq_i32 s7, 0 // (op 18119) 
s_mov_b32 s15, scc // (op 18119) 

s_cmp_eq_i32 s15, 0 // (op 18119) 
s_cselect_b32 s7, s6, s8 // (op 18119) 


s_mul_hi_u32 s8, s7, s12 // (op 18119) 
s_sub_u32 s15, s7, s8 // (op 18119) 
s_lshr_b32 s18, s15, 1 // (op 18119) 

s_add_u32 s15, s18, s8 // (op 18119) 


s_lshr_b32 s8, s15, s10 // (op 18119) 

s_and_b32 s15, s10, 2147483648 // (op 18119) 
s_cmp_eq_i32 s15, 0 // (op 18119) 
s_mov_b32 s18, scc // (op 18119) 

s_cmp_eq_i32 s18, 0 // (op 18119) 
s_cselect_b32 s15, s7, s8 // (op 18119) 


v_lshl_add_u32 v26, s15, 7, v27 // (op 18119) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 18119) 
s_mul_i32 s8, s7, s18 // (op 18119) 


s_sub_u32 s7, s6, s8 // (op 18119) 


v_lshl_add_u32 v27, s7, 2, v28 // (op 18119) 


v_mul_lo_u32 v28, v27, s38 // (op 18119) 

v_add_u32 v27, v26, v28 // (op 18119) 


s_mul_i32 s19, s38, s43 // (op 18121) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 18121) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 18121) least significant
s_mul_hi_u32 s6, s38, s42 // (op 18121) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18121) most significant: sum
s_add_u32 s19, s19, s6 // (op 18121) most significant: sum


s_mov_b32 s76, s40 // (op 18121) reg expression
s_mov_b32 s77, s41 // (op 18121) reg expression
s_mov_b32 s78, s18 // (op 18121) reg expression
s_mov_b32 s79, 131072 // (op 18121) call()

v_and_b32 v26, -128, v1 // (op 18138) 
s_mul_i32 s15, s16, s50 // (op 18138) 
s_add_u32 s15, s15, s0 // (op 18138) 
v_sub_u32 v28, v1, v26 // (op 18138) 

s_add_u32 s6, s15, s9 // (op 18138) 

v_lshrrev_b32 v26, 7, v1 // (op 18138) 
s_mul_hi_u32 s15, s6, s13 // (op 18138) 
v_lshlrev_b32 v29, 1, v26 // (op 18138) 

s_sub_u32 s7, s6, s15 // (op 18138) 
s_lshr_b32 s8, s7, 1 // (op 18138) 

s_add_u32 s7, s8, s15 // (op 18138) 


s_lshr_b32 s8, s7, s11 // (op 18138) 

s_and_b32 s7, s11, 2147483648 // (op 18138) 
s_cmp_eq_i32 s7, 0 // (op 18138) 
s_mov_b32 s15, scc // (op 18138) 

s_cmp_eq_i32 s15, 0 // (op 18138) 
s_cselect_b32 s7, s6, s8 // (op 18138) 


s_mul_hi_u32 s8, s7, s12 // (op 18138) 
s_sub_u32 s15, s7, s8 // (op 18138) 
s_lshr_b32 s18, s15, 1 // (op 18138) 

s_add_u32 s15, s18, s8 // (op 18138) 


s_lshr_b32 s8, s15, s10 // (op 18138) 

s_and_b32 s15, s10, 2147483648 // (op 18138) 
s_cmp_eq_i32 s15, 0 // (op 18138) 
s_mov_b32 s18, scc // (op 18138) 

s_cmp_eq_i32 s18, 0 // (op 18138) 
s_cselect_b32 s15, s7, s8 // (op 18138) 


v_lshl_add_u32 v26, s15, 7, v28 // (op 18138) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 18138) 
s_mul_i32 s8, s7, s18 // (op 18138) 


s_sub_u32 s7, s6, s8 // (op 18138) 


v_lshl_add_u32 v28, s7, 2, v29 // (op 18138) 


v_mul_lo_u32 v29, v28, s38 // (op 18138) 

v_add_u32 v28, v26, v29 // (op 18138) 


s_mul_i32 s19, s38, s43 // (op 18140) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 18140) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 18140) least significant
s_mul_hi_u32 s6, s38, s42 // (op 18140) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18140) most significant: sum
s_add_u32 s19, s19, s6 // (op 18140) most significant: sum


s_mov_b32 s76, s40 // (op 18140) reg expression
s_mov_b32 s77, s41 // (op 18140) reg expression
s_mov_b32 s78, s18 // (op 18140) reg expression
s_mov_b32 s79, 131072 // (op 18140) call()

v_lshrrev_b32 v26, 2, v1 // (op 18247) 
s_mul_i32 s15, s16, s50 // (op 18247) 
s_add_u32 s15, s15, s0 // (op 18247) 
v_lshlrev_b32 v29, 1, v26 // (op 18247) 

s_add_u32 s6, s15, s9 // (op 18247) 

v_and_b32 v26, -4, v1 // (op 18247) 
s_mul_hi_u32 s15, s6, s13 // (op 18247) 
v_sub_u32 v30, v1, v26 // (op 18247) 

s_sub_u32 s7, s6, s15 // (op 18247) 
s_lshr_b32 s8, s7, 1 // (op 18247) 

s_add_u32 s7, s8, s15 // (op 18247) 


s_lshr_b32 s8, s7, s11 // (op 18247) 

s_and_b32 s7, s11, 2147483648 // (op 18247) 
s_cmp_eq_i32 s7, 0 // (op 18247) 
s_mov_b32 s15, scc // (op 18247) 

s_cmp_eq_i32 s15, 0 // (op 18247) 
s_cselect_b32 s7, s6, s8 // (op 18247) 


s_mul_hi_u32 s8, s7, s12 // (op 18247) 
s_sub_u32 s15, s7, s8 // (op 18247) 
s_lshr_b32 s18, s15, 1 // (op 18247) 

s_add_u32 s15, s18, s8 // (op 18247) 


s_lshr_b32 s8, s15, s10 // (op 18247) 

s_and_b32 s15, s10, 2147483648 // (op 18247) 
s_cmp_eq_i32 s15, 0 // (op 18247) 
s_mov_b32 s18, scc // (op 18247) 

s_cmp_eq_i32 s18, 0 // (op 18247) 
s_cselect_b32 s15, s7, s8 // (op 18247) 


s_mul_i32 s8, s15, s14 // (op 18247) 

s_sub_u32 s15, s7, s8 // (op 18247) 

v_lshl_add_u32 v26, s15, 7, v29 // (op 18247) 


v_mul_lo_u32 v29, v26, s44 // (op 18247) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 18247) 
s_mul_i32 s8, s7, s18 // (op 18247) 


s_sub_u32 s7, s6, s8 // (op 18247) 


v_lshl_add_u32 v26, s7, 2, v30 // (op 18247) 


v_add_u32 v30, v26, v29 // (op 18247) 


s_mul_i32 s19, s44, s23 // (op 18249) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 18249) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 18249) least significant
s_mul_hi_u32 s6, s44, s22 // (op 18249) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18249) most significant: sum
s_add_u32 s19, s19, s6 // (op 18249) most significant: sum


s_mov_b32 s80, s46 // (op 18249) reg expression
s_mov_b32 s81, s47 // (op 18249) reg expression
s_mov_b32 s82, s18 // (op 18249) reg expression
s_mov_b32 s83, 131072 // (op 18249) call()

v_lshrrev_b32 v26, 2, v1 // (op 18266) 
s_mul_i32 s15, s16, s50 // (op 18266) 
s_add_u32 s15, s15, s0 // (op 18266) 
v_lshlrev_b32 v29, 1, v26 // (op 18266) 

s_add_u32 s6, s15, s9 // (op 18266) 

v_and_b32 v26, -4, v1 // (op 18266) 
s_mul_hi_u32 s15, s6, s13 // (op 18266) 
v_sub_u32 v31, v1, v26 // (op 18266) 

s_sub_u32 s7, s6, s15 // (op 18266) 
s_lshr_b32 s8, s7, 1 // (op 18266) 

s_add_u32 s7, s8, s15 // (op 18266) 


s_lshr_b32 s8, s7, s11 // (op 18266) 

s_and_b32 s7, s11, 2147483648 // (op 18266) 
s_cmp_eq_i32 s7, 0 // (op 18266) 
s_mov_b32 s15, scc // (op 18266) 

s_cmp_eq_i32 s15, 0 // (op 18266) 
s_cselect_b32 s7, s6, s8 // (op 18266) 


s_mul_hi_u32 s8, s7, s12 // (op 18266) 
s_sub_u32 s15, s7, s8 // (op 18266) 
s_lshr_b32 s18, s15, 1 // (op 18266) 

s_add_u32 s15, s18, s8 // (op 18266) 


s_lshr_b32 s8, s15, s10 // (op 18266) 

s_and_b32 s15, s10, 2147483648 // (op 18266) 
s_cmp_eq_i32 s15, 0 // (op 18266) 
s_mov_b32 s18, scc // (op 18266) 

s_cmp_eq_i32 s18, 0 // (op 18266) 
s_cselect_b32 s15, s7, s8 // (op 18266) 


s_mul_i32 s8, s15, s14 // (op 18266) 

s_sub_u32 s15, s7, s8 // (op 18266) 

v_lshl_add_u32 v26, s15, 7, v29 // (op 18266) 


v_mul_lo_u32 v29, v26, s44 // (op 18266) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 18266) 
s_mul_i32 s8, s7, s18 // (op 18266) 


s_sub_u32 s7, s6, s8 // (op 18266) 


v_lshl_add_u32 v26, s7, 2, v31 // (op 18266) 


v_add_u32 v31, v26, v29 // (op 18266) 


s_mul_i32 s19, s44, s23 // (op 18268) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 18268) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 18268) least significant
s_mul_hi_u32 s6, s44, s22 // (op 18268) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18268) most significant: sum
s_add_u32 s19, s19, s6 // (op 18268) most significant: sum


s_mov_b32 s80, s46 // (op 18268) reg expression
s_mov_b32 s81, s47 // (op 18268) reg expression
s_mov_b32 s82, s18 // (op 18268) reg expression
s_mov_b32 s83, 131072 // (op 18268) call()

v_and_b32 v26, -8, v1 // (op 18839) 
v_sub_u32 v29, v1, v26 // (op 18839) 

v_lshlrev_b32 v26, 4, v29 // (op 18839) 

v_lshrrev_b32 v29, 3, v1 // (op 18839) 
v_lshl_add_u32 v32, v29, 7, v26 // (op 18839) 


s_nop 0
v_readfirstlane_b32 s15, v32 // (op 18839) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 19703) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 17741) 
v_mov_b64 v[32:33], s[18:19] // (op 17741) convert


v_mov_b64 v[34:35], v[16:17] // (op 17741) 
s_mov_b32 m0, s15 // (op 17741) reg expression
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 17741) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 17741) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 17741) most significant half

s_add_u32 m0, m0, 4096 // (op 17741) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 17741) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 17741) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 17741) most significant half

s_add_u32 m0, m0, 4096 // (op 17741) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 17741) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 17741) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 17741) most significant half

s_add_u32 m0, m0, 4096 // (op 17741) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 17741) Load value direct to lds




v_mov_b64 v[32:33], s[44:45] // (op 17153) convert

v_mov_b32 v26, v30 // (op 17153) 
buffer_load_ubyte v34, v26, s[80:83], 0 offen offset: 0 // (op 17153) Load value
v_add_u32 v26, v26, v32 // (op 17153) 
buffer_load_ubyte v35, v26, s[80:83], 0 offen offset: 0 // (op 17153) Load value


v_and_b32 v26, -4, v1 // (op 18764) 
v_sub_u32 v29, v1, v26 // (op 18764) 

v_lshlrev_b32 v26, 5, v29 // (op 18764) 

v_lshrrev_b32 v29, 2, v1 // (op 18764) 
v_lshl_add_u32 v32, v29, 7, v26 // (op 18764) 


v_lshlrev_b32 v26, 2, v32 // (op 18764) 

v_lshrrev_b32 v29, 3, v26 // (op 18764) 

s_nop 0
v_readfirstlane_b32 s15, v29 // (op 18764) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 19684) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 17746) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 17746) 

v_lshrrev_b64 v[32:33], 3, s[28:29] // (op 17746) 


v_mov_b64 v[36:37], v[20:21] // (op 17746) reg expression
s_add_u32 s6, 16384, s15 // (op 17746) 
s_mov_b32 m0, s6 // (op 17746) reg expression
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 17746) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 17746) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 17746) most significant half

s_add_u32 m0, m0, 4096 // (op 17746) 
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 17746) Load value direct to lds





v_mov_b64 v[32:33], s[38:39] // (op 17159) convert

v_mov_b32 v26, v27 // (op 17159) reg expression
buffer_load_ubyte v36, v26, s[76:79], 0 offen offset: 0 // (op 17159) Load value
v_add_u32 v26, v26, v32 // (op 17159) 
buffer_load_ubyte v37, v26, s[76:79], 0 offen offset: 0 // (op 17159) Load value


v_and_b32 v26, -8, v1 // (op 19149) 
v_sub_u32 v29, v1, v26 // (op 19149) 

v_lshlrev_b32 v26, 4, v29 // (op 19149) 

v_lshrrev_b32 v29, 3, v1 // (op 19149) 
v_lshl_add_u32 v32, v29, 7, v26 // (op 19149) 


s_nop 0
v_readfirstlane_b32 s15, v32 // (op 19149) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 19773) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 17751) 
v_mov_b64 v[32:33], s[18:19] // (op 17751) convert


v_mov_b64 v[38:39], v[16:17] // (op 17751) 
s_add_u32 s6, 24576, s15 // (op 17751) 
s_mov_b32 m0, s6 // (op 17751) reg expression
s_mov_b32 s7, 128 // (op 17751) call()
buffer_load_dwordx4 v38, s[72:75], s7 offen offset: 0 lds // (op 17751) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v32 // (op 17751) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 17751) most significant half

s_add_u32 m0, m0, 4096 // (op 17751) 
buffer_load_dwordx4 v38, s[72:75], s7 offen offset: 0 lds // (op 17751) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v32 // (op 17751) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 17751) most significant half

s_add_u32 m0, m0, 4096 // (op 17751) 
buffer_load_dwordx4 v38, s[72:75], s7 offen offset: 0 lds // (op 17751) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v32 // (op 17751) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 17751) most significant half

s_add_u32 m0, m0, 4096 // (op 17751) 
buffer_load_dwordx4 v38, s[72:75], s7 offen offset: 0 lds // (op 17751) Load value direct to lds







v_mov_b64 v[16:17], s[44:45] // (op 17165) convert

v_mov_b32 v26, v30 // (op 17165) 
buffer_load_ubyte v32, v26, s[80:83], 0 offen offset:4 // (op 17165) Load value
v_add_u32 v26, v26, v16 // (op 17165) 
buffer_load_ubyte v33, v26, s[80:83], 0 offen offset:4 // (op 17165) Load value



v_and_b32 v26, -4, v1 // (op 19105) 
v_sub_u32 v16, v1, v26 // (op 19105) 

v_lshlrev_b32 v17, 5, v16 // (op 19105) 

v_lshrrev_b32 v16, 2, v1 // (op 19105) 
v_lshl_add_u32 v26, v16, 7, v17 // (op 19105) 


v_lshlrev_b32 v16, 2, v26 // (op 19105) 

v_lshrrev_b32 v17, 3, v16 // (op 19105) 

s_nop 0
v_readfirstlane_b32 s15, v17 // (op 19105) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 19763) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 17753) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 17753) 

v_lshrrev_b64 v[16:17], 3, s[28:29] // (op 17753) 


s_lshl_b64 s[18:19], s[30:31], 7 // (op 17753) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 17753) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 17753) 

v_mov_b32 v26, s18 // (op 17753) 
v_mov_b32 v29, s19 // (op 17753) 
v_add_co_u32 v38, s[28:29], v20, v26 // (op 17753) least significant half
v_addc_co_u32 v39, s[28:29], v21, v29, s[28:29] // (op 17753) most significant half




s_add_u32 s6, 40960, s15 // (op 17753) 
s_mov_b32 m0, s6 // (op 17753) reg expression
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 17753) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v16 // (op 17753) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 17753) most significant half

s_add_u32 m0, m0, 4096 // (op 17753) 
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 17753) Load value direct to lds






v_mov_b64 v[16:17], s[38:39] // (op 17171) convert

s_lshl_b32 s15, s38, 2 // (op 17171) 
v_add_u32 v26, v27, s15 // (op 17171) 

buffer_load_ubyte v20, v26, s[76:79], 0 offen offset: 0 // (op 17171) Load value
v_add_u32 v26, v26, v16 // (op 17171) 
buffer_load_ubyte v21, v26, s[76:79], 0 offen offset: 0 // (op 17171) Load value



v_and_b32 v16, -4, v1 // (op 18991) 
v_sub_u32 v17, v1, v16 // (op 18991) 

v_lshrrev_b32 v16, 2, v1 // (op 18991) 
v_lshl_add_u32 v26, v16, 3, v17 // (op 18991) 


s_barrier  // (op 19743) 



s_waitcnt vmcnt(15)
ds_write_b8 v26, v34 offset:49152 // (op 17174) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v26, v35 offset:49156 // (op 17174) Store local data 

v_and_b32 v16, -128, v1 // (op 18915) 
v_sub_u32 v17, v1, v16 // (op 18915) 

v_lshrrev_b32 v16, 7, v1 // (op 18915) 
v_lshl_add_u32 v26, v16, 8, v17 // (op 18915) 


s_barrier  // (op 19723) 



s_waitcnt vmcnt(11)
ds_write_b8 v26, v36 offset:49664 // (op 17176) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v26, v37 offset:49792 // (op 17176) Store local data 

s_barrier  // (op 19700) 



ds_read_b128 v[38:41], v14 // (op 17211) Load local data 
ds_read_b128 v[42:45], v14 offset:64 // (op 17211) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 19740) 



ds_read_u8 v16, v15 offset:49152 // (op 17251) Load local data 



ds_read_b128 v[46:49], v14 offset:4096 // (op 17216) Load local data 
ds_read_b128 v[50:53], v14 offset:4160 // (op 17216) Load local data 



ds_read_u8 v17, v15 offset:49280 // (op 17256) Load local data 



ds_read_b128 v[54:57], v14 offset:8192 // (op 17221) Load local data 
ds_read_b128 v[58:61], v14 offset:8256 // (op 17221) Load local data 



ds_read_u8 v26, v15 offset:49408 // (op 17261) Load local data 



ds_read_b128 v[62:65], v14 offset:12288 // (op 17226) Load local data 
ds_read_b128 v[66:69], v14 offset:12352 // (op 17226) Load local data 



ds_read_u8 v27, v15 offset:49536 // (op 17266) Load local data 
s_barrier  // (op 19681) 



ds_read_b64_tr_b4 v[70:71], v3 offset:16384 // (op 17191) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v3 offset:17408 // (op 17191) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 19720) 


ds_read_u8 v29, v0 offset:49664 // (op 17231) Load local data 



ds_read_b64_tr_b4 v[74:75], v3 offset:16400 // (op 17196) Transpose load local data 
ds_read_b64_tr_b4 v[76:77], v3 offset:17424 // (op 17196) Transpose load local data 


ds_read_u8 v30, v0 offset:49696 // (op 17236) Load local data 



ds_read_b64_tr_b4 v[78:79], v3 offset:16416 // (op 17201) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v3 offset:17440 // (op 17201) Transpose load local data 


ds_read_u8 v82, v0 offset:49728 // (op 17241) Load local data 



ds_read_b64_tr_b4 v[84:85], v3 offset:16432 // (op 17206) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v3 offset:17456 // (op 17206) Transpose load local data 


ds_read_u8 v83, v0 offset:49760 // (op 17246) Load local data 
v_and_b32 v88, -4, v1 // (op 18736) 
v_sub_u32 v89, v1, v88 // (op 18736) 

v_lshlrev_b32 v88, 5, v89 // (op 18736) 

v_lshrrev_b32 v89, 2, v1 // (op 18736) 
v_lshl_add_u32 v90, v89, 7, v88 // (op 18736) 


v_lshlrev_b32 v88, 2, v90 // (op 18736) 

v_lshrrev_b32 v89, 3, v88 // (op 18736) 

s_nop 0
v_readfirstlane_b32 s15, v89 // (op 18736) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v88, -8, v1 // (op 18811) 
v_sub_u32 v89, v1, v88 // (op 18811) 

v_lshlrev_b32 v88, 4, v89 // (op 18811) 

v_lshrrev_b32 v89, 3, v1 // (op 18811) 
v_lshl_add_u32 v90, v89, 7, v88 // (op 18811) 


s_nop 0
v_readfirstlane_b32 s6, v90 // (op 18811) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v88, -128, v1 // (op 18886) 
v_sub_u32 v89, v1, v88 // (op 18886) 

v_lshrrev_b32 v88, 7, v1 // (op 18886) 
v_lshl_add_u32 v90, v88, 8, v89 // (op 18886) 


v_and_b32 v88, -4, v1 // (op 18962) 
v_sub_u32 v89, v1, v88 // (op 18962) 

v_lshrrev_b32 v88, 2, v1 // (op 18962) 
v_lshl_add_u32 v91, v88, 3, v89 // (op 18962) 


v_and_b32 v88, -4, v1 // (op 19090) 
v_sub_u32 v89, v1, v88 // (op 19090) 

v_lshlrev_b32 v88, 5, v89 // (op 19090) 

v_lshrrev_b32 v89, 2, v1 // (op 19090) 
v_lshl_add_u32 v92, v89, 7, v88 // (op 19090) 


v_lshlrev_b32 v88, 2, v92 // (op 19090) 

v_lshrrev_b32 v89, 3, v88 // (op 19090) 

s_nop 0
v_readfirstlane_b32 s7, v89 // (op 19090) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v88, -8, v1 // (op 19134) 
v_sub_u32 v89, v1, v88 // (op 19134) 

v_lshlrev_b32 v88, 4, v89 // (op 19134) 

v_lshrrev_b32 v89, 3, v1 // (op 19134) 
v_lshl_add_u32 v92, v89, 7, v88 // (op 19134) 


s_nop 0
v_readfirstlane_b32 s8, v92 // (op 19134) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v88, -128, v1 // (op 19178) 
v_sub_u32 v89, v1, v88 // (op 19178) 

v_lshrrev_b32 v88, 7, v1 // (op 19178) 
v_lshl_add_u32 v92, v88, 8, v89 // (op 19178) 


v_and_b32 v88, -4, v1 // (op 19207) 
v_sub_u32 v89, v1, v88 // (op 19207) 

v_lshrrev_b32 v88, 2, v1 // (op 19207) 
v_lshl_add_u32 v93, v88, 3, v89 // (op 19207) 


s_mov_b32 s1, 0 // (op 2865) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2864) 
s_mov_b32 s28, s18 // (op 2864) convert

s_and_b32 s29, s28, 4294967294 // (op 2864) 

s_cmp_lt_u32 s1, s29 // (op 2864) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_2864 // (op 2864) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_2864 if false)
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_2864:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:73], v[38:45], a[0:3], v29, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 194) 
s_barrier  // (op 19706) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[46:53], a[4:7], v29, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 654) 
s_add_u32 s1, s1, 2 // (op 2866) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[70:73], v[54:61], a[8:11], v29, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 830) 
s_lshl_b64 s[18:19], s[34:35], 5 // (op 17690) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:73], v[62:69], a[12:15], v29, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1006) 
v_mov_b64 v[88:89], s[18:19] // (op 17690) convert


v_mov_b64 v[94:95], v[24:25] // (op 17690) 
v_mov_b64 v[96:97], v[94:95] // (op 17690) 
s_mov_b32 m0, s6 // (op 17690) reg expression
s_mov_b32 s18, 256 // (op 17690) call()
buffer_load_dwordx4 v96, s[72:75], s18 offen offset: 0 lds // (op 17690) Load value direct to lds
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[38:45], a[16:19], v30, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1501) 
v_add_co_u32 v96, s[28:29], v96, v88 // (op 17690) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[46:53], a[20:23], v30, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1455) 
v_addc_co_u32 v97, s[28:29], 0, v97, s[28:29] // (op 17690) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[54:61], a[24:27], v30, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1409) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[62:69], a[28:31], v30, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1363) 
s_add_u32 m0, m0, 4096 // (op 17690) 
buffer_load_dwordx4 v96, s[72:75], s18 offen offset: 0 lds // (op 17690) Load value direct to lds
v_add_co_u32 v96, s[28:29], v96, v88 // (op 17690) least significant half
v_addc_co_u32 v97, s[28:29], 0, v97, s[28:29] // (op 17690) most significant half

s_add_u32 m0, m0, 4096 // (op 17690) 
buffer_load_dwordx4 v96, s[72:75], s18 offen offset: 0 lds // (op 17690) Load value direct to lds
v_add_co_u32 v96, s[28:29], v96, v88 // (op 17690) least significant half
v_addc_co_u32 v97, s[28:29], 0, v97, s[28:29] // (op 17690) most significant half

s_add_u32 m0, m0, 4096 // (op 17690) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[38:45], a[32:35], v82, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2271) 
buffer_load_dwordx4 v96, s[72:75], s18 offen offset: 0 lds // (op 17690) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[46:53], a[36:39], v82, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2225) 
v_mov_b64 v[88:89], s[44:45] // (op 200) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[54:61], a[40:43], v82, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2179) 
v_mov_b32 v96, v31 // (op 200) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[62:69], a[44:47], v82, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2133) 
v_mov_b32 v97, v96 // (op 200) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[84:87], v[38:45], a[48:51], v83, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3041) 
buffer_load_ubyte v34, v97, s[80:83], 0 offen offset:8 // (op 200) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[84:87], v[46:53], a[52:55], v83, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2995) 
v_add_u32 v97, v97, v88 // (op 200) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[84:87], v[54:61], a[56:59], v83, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2949) 
buffer_load_ubyte v35, v97, s[80:83], 0 offen offset:8 // (op 200) Load value


s_barrier  // (op 19687) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[84:87], v[62:69], a[60:63], v83, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2903) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 17685) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 17685) 

v_lshrrev_b64 v[88:89], 3, s[28:29] // (op 17685) 


v_mov_b64 v[98:99], v[22:23] // (op 17685) 
s_lshl_b64 s[18:19], s[30:31], 7 // (op 17685) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 17685) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 17685) 

s_lshl_b64 s[28:29], s[18:19], 1 // (op 17685) 

v_mov_b32 v97, s28 // (op 17685) 
v_mov_b32 v100, s29 // (op 17685) 
v_add_co_u32 v102, s[18:19], v98, v97 // (op 17685) least significant half
v_addc_co_u32 v103, s[18:19], v99, v100, s[18:19] // (op 17685) most significant half




s_add_u32 s18, 16384, s15 // (op 17685) 
s_nop 0
s_mov_b32 m0, s18 // (op 17685) reg expression
buffer_load_dwordx4 v102, s[68:71], 0 offen offset: 0 lds // (op 17685) Load value direct to lds
v_add_co_u32 v102, s[28:29], v102, v88 // (op 17685) least significant half
v_addc_co_u32 v103, s[28:29], 0, v103, s[28:29] // (op 17685) most significant half

s_add_u32 m0, m0, 4096 // (op 17685) 
buffer_load_dwordx4 v102, s[68:71], 0 offen offset: 0 lds // (op 17685) Load value direct to lds




v_mov_b64 v[88:89], s[38:39] // (op 199) convert

v_mov_b32 v97, v28 // (op 199) 
s_lshl_b32 s18, s38, 3 // (op 199) 
v_add_u32 v100, v97, s18 // (op 199) 

buffer_load_ubyte v36, v100, s[76:79], 0 offen offset: 0 // (op 199) Load value
v_add_u32 v100, v100, v88 // (op 199) 
buffer_load_ubyte v37, v100, s[76:79], 0 offen offset: 0 // (op 199) Load value


s_barrier  // (op 19788) 



s_waitcnt vmcnt(15)
ds_write_b8 v93, v32 offset:50176 // (op 533) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v93, v33 offset:50180 // (op 533) Store local data 
s_barrier  // (op 19782) 



s_waitcnt vmcnt(11)
ds_write_b8 v92, v20 offset:50688 // (op 532) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v92, v21 offset:50816 // (op 532) Store local data 
s_barrier  // (op 19769) 



ds_read_b128 v[100:103], v12 offset:24576 // (op 525) Load local data 
ds_read_b128 v[104:107], v12 offset:24640 // (op 525) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 19785) 



ds_read_u8 v88, v13 offset:50176 // (op 529) Load local data 



ds_read_b128 v[108:111], v12 offset:28672 // (op 659) Load local data 
ds_read_b128 v[112:115], v12 offset:28736 // (op 659) Load local data 



ds_read_u8 v30, v13 offset:50304 // (op 663) Load local data 



ds_read_b128 v[116:119], v12 offset:32768 // (op 835) Load local data 
ds_read_b128 v[120:123], v12 offset:32832 // (op 835) Load local data 



ds_read_u8 v82, v13 offset:50432 // (op 839) Load local data 



ds_read_b128 v[124:127], v12 offset:36864 // (op 1011) Load local data 
ds_read_b128 v[128:131], v12 offset:36928 // (op 1011) Load local data 



ds_read_u8 v89, v13 offset:50560 // (op 1015) Load local data 
s_barrier  // (op 19760) 
s_barrier  // (op 19898) 



ds_read_b64_tr_b4 v[70:71], v11 offset:40960 // (op 523) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v11 offset:41984 // (op 523) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 19779) 


ds_read_u8 v29, v10 offset:50688 // (op 527) Load local data 



ds_read_b64_tr_b4 v[74:75], v11 offset:40976 // (op 1366) Transpose load local data 
ds_read_b64_tr_b4 v[76:77], v11 offset:42000 // (op 1366) Transpose load local data 


ds_read_u8 v17, v10 offset:50720 // (op 1369) Load local data 



ds_read_b64_tr_b4 v[78:79], v11 offset:40992 // (op 2136) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v11 offset:42016 // (op 2136) Transpose load local data 


ds_read_u8 v26, v10 offset:50752 // (op 2139) Load local data 



ds_read_b64_tr_b4 v[84:85], v11 offset:41008 // (op 2906) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v11 offset:42032 // (op 2906) Transpose load local data 


ds_read_u8 v83, v10 offset:50784 // (op 2909) Load local data 
s_barrier  // (op 19932) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:73], v[100:107], a[0:3], v29, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 530) 
s_barrier  // (op 19776) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[108:115], a[4:7], v29, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 655) 
s_lshl_b64 s[18:19], s[34:35], 5 // (op 17713) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[70:73], v[116:123], a[8:11], v29, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 831) 
v_mov_b64 v[132:133], s[18:19] // (op 17713) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:73], v[124:131], a[12:15], v29, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1007) 
v_mov_b64 v[134:135], v[94:95] // (op 17713) 
s_add_u32 s18, 24576, s8 // (op 17713) 
s_mov_b32 m0, s18 // (op 17713) reg expression
s_mov_b32 s19, 384 // (op 17713) call()
buffer_load_dwordx4 v134, s[72:75], s19 offen offset: 0 lds // (op 17713) Load value direct to lds
v_add_co_u32 v134, s[28:29], v134, v132 // (op 17713) least significant half
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[100:107], a[16:19], v17, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1502) 
v_addc_co_u32 v135, s[28:29], 0, v135, s[28:29] // (op 17713) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[108:115], a[20:23], v17, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1456) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[116:123], a[24:27], v17, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1410) 
s_add_u32 m0, m0, 4096 // (op 17713) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[124:131], a[28:31], v17, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1364) 
buffer_load_dwordx4 v134, s[72:75], s19 offen offset: 0 lds // (op 17713) Load value direct to lds
v_add_co_u32 v134, s[28:29], v134, v132 // (op 17713) least significant half
v_addc_co_u32 v135, s[28:29], 0, v135, s[28:29] // (op 17713) most significant half

s_add_u32 m0, m0, 4096 // (op 17713) 
buffer_load_dwordx4 v134, s[72:75], s19 offen offset: 0 lds // (op 17713) Load value direct to lds
v_add_co_u32 v134, s[28:29], v134, v132 // (op 17713) least significant half
v_addc_co_u32 v135, s[28:29], 0, v135, s[28:29] // (op 17713) most significant half

s_add_u32 m0, m0, 4096 // (op 17713) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[100:107], a[32:35], v26, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2272) 
buffer_load_dwordx4 v134, s[72:75], s19 offen offset: 0 lds // (op 17713) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[108:115], a[36:39], v26, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2226) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[116:123], a[40:43], v26, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2180) 
v_add_co_u32 v24, vcc, 256, v24 // (op 18002) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[124:131], a[44:47], v26, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2134) 
v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 18002) most significant half
v_mov_b64 v[94:95], s[44:45] // (op 536) convert

v_mov_b32 v132, v96 // (op 536) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[84:87], v[100:107], a[48:51], v83, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3042) 
buffer_load_ubyte v32, v132, s[80:83], 0 offen offset:12 // (op 536) Load value


v_add_u32 v132, v132, v94 // (op 536) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[84:87], v[108:115], a[52:55], v83, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2996) 
buffer_load_ubyte v33, v132, s[80:83], 0 offen offset:12 // (op 536) Load value




s_barrier  // (op 19766) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[84:87], v[116:123], a[56:59], v83, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2950) 


v_add_u32 v31, 8, v31 // (op 18258) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[84:87], v[124:131], a[60:63], v83, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2904) 
s_lshl_b64 s[18:19], s[30:31], 6 // (op 17708) 


s_lshl_b64 s[28:29], s[18:19], 2 // (op 17708) 

v_lshrrev_b64 v[88:89], 3, s[28:29] // (op 17708) 


s_lshl_b64 s[18:19], s[30:31], 7 // (op 17708) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 17708) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 17708) 

s_mul_i32 s29, 3, s19 // (op 17708) most significant: low of low * high
s_mul_i32 s28, 3, s18 // (op 17708) least significant
s_mul_hi_u32 s48, 3, s18 // (op 17708) most significant: high of low * low
s_add_u32 s29, s29, s48 // (op 17708) most significant: sum


v_mov_b32 v94, s28 // (op 17708) 
v_mov_b32 v95, s29 // (op 17708) 
v_add_co_u32 v100, s[18:19], v98, v94 // (op 17708) least significant half
v_addc_co_u32 v101, s[18:19], v99, v95, s[18:19] // (op 17708) most significant half




s_add_u32 s18, 40960, s7 // (op 17708) 
s_nop 0
s_mov_b32 m0, s18 // (op 17708) reg expression
buffer_load_dwordx4 v100, s[68:71], 0 offen offset: 0 lds // (op 17708) Load value direct to lds
v_add_co_u32 v100, s[28:29], v100, v88 // (op 17708) least significant half
v_addc_co_u32 v101, s[28:29], 0, v101, s[28:29] // (op 17708) most significant half

s_add_u32 m0, m0, 4096 // (op 17708) 
buffer_load_dwordx4 v100, s[68:71], 0 offen offset: 0 lds // (op 17708) Load value direct to lds





v_mov_b64 v[88:89], s[38:39] // (op 535) convert

s_lshl_b32 s18, s38, 2 // (op 535) 
s_mul_i32 s19, 3, s18 // (op 535) 

v_add_u32 v94, v97, s19 // (op 535) 

s_lshl_b64 s[18:19], s[30:31], 7 // (op 17856) 
buffer_load_ubyte v20, v94, s[76:79], 0 offen offset: 0 // (op 535) Load value
v_add_u32 v94, v94, v88 // (op 535) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 17856) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 17856) 

s_lshl_b64 s[28:29], s[18:19], 1 // (op 17856) 

v_mov_b32 v95, s28 // (op 17856) 
v_mov_b32 v96, s29 // (op 17856) 
v_add_co_u32 v22, s[18:19], v22, v95 // (op 17856) least significant half
v_addc_co_u32 v23, s[18:19], v23, v96, s[18:19] // (op 17856) most significant half




buffer_load_ubyte v21, v94, s[76:79], 0 offen offset: 0 // (op 535) Load value



s_barrier  // (op 19746) 
s_lshl_b32 s18, s38, 3 // (op 18130) 



v_add_u32 v28, v28, s18 // (op 18130) 

s_waitcnt vmcnt(15)
ds_write_b8 v91, v34 offset:49152 // (op 197) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v91, v35 offset:49156 // (op 197) Store local data 
s_barrier  // (op 19726) 



s_waitcnt vmcnt(11)
ds_write_b8 v90, v36 offset:49664 // (op 196) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v90, v37 offset:49792 // (op 196) Store local data 
s_barrier  // (op 19712) 



ds_read_b128 v[38:41], v8 // (op 189) Load local data 
ds_read_b128 v[42:45], v8 offset:64 // (op 189) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 19753) 



ds_read_u8 v16, v9 offset:49152 // (op 193) Load local data 



ds_read_b128 v[46:49], v8 offset:4096 // (op 649) Load local data 
ds_read_b128 v[50:53], v8 offset:4160 // (op 649) Load local data 



ds_read_u8 v17, v9 offset:49280 // (op 653) Load local data 



ds_read_b128 v[54:57], v8 offset:8192 // (op 825) Load local data 
ds_read_b128 v[58:61], v8 offset:8256 // (op 825) Load local data 



ds_read_u8 v26, v9 offset:49408 // (op 829) Load local data 



ds_read_b128 v[62:65], v8 offset:12288 // (op 1001) Load local data 
ds_read_b128 v[66:69], v8 offset:12352 // (op 1001) Load local data 



ds_read_u8 v27, v9 offset:49536 // (op 1005) Load local data 
s_barrier  // (op 19693) 



ds_read_b64_tr_b4 v[70:71], v7 offset:16384 // (op 187) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v7 offset:17408 // (op 187) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 19733) 


ds_read_u8 v29, v6 offset:49664 // (op 191) Load local data 



ds_read_b64_tr_b4 v[74:75], v7 offset:16400 // (op 1358) Transpose load local data 
ds_read_b64_tr_b4 v[76:77], v7 offset:17424 // (op 1358) Transpose load local data 


ds_read_u8 v30, v6 offset:49696 // (op 1361) Load local data 



ds_read_b64_tr_b4 v[78:79], v7 offset:16416 // (op 2128) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v7 offset:17440 // (op 2128) Transpose load local data 


ds_read_u8 v82, v6 offset:49728 // (op 2131) Load local data 



ds_read_b64_tr_b4 v[84:85], v7 offset:16432 // (op 2898) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v7 offset:17456 // (op 2898) Transpose load local data 


ds_read_u8 v83, v6 offset:49760 // (op 2901) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2864) 
s_mov_b32 s28, s18 // (op 2864) convert

s_and_b32 s29, s28, 4294967294 // (op 2864) 

s_cmp_lt_u32 s1, s29 // (op 2864) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_2864 // (op 2864) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_2864 if true)
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_2864:




















s_barrier  // (op 19916) 
v_and_b32 v28, -4, v1 // (op 17780) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 17780) 
v_sub_u32 v31, v1, v28 // (op 17780) 

s_and_b32 s15, s18, 4294967294 // (op 17780) 
v_lshlrev_b32 v28, 5, v31 // (op 17780) 

s_add_u32 s6, s0, s15 // (op 17780) 

v_lshrrev_b32 v31, 2, v1 // (op 17780) 
s_mul_i32 s15, s16, s50 // (op 17780) 
s_add_u32 s15, s15, s6 // (op 17780) 

s_add_u32 s6, s15, s9 // (op 17780) 

s_mul_hi_u32 s15, s6, s13 // (op 17780) 
s_sub_u32 s7, s6, s15 // (op 17780) 
s_lshr_b32 s8, s7, 1 // (op 17780) 

s_add_u32 s7, s8, s15 // (op 17780) 


s_lshr_b32 s8, s7, s11 // (op 17780) 

s_and_b32 s7, s11, 2147483648 // (op 17780) 
s_cmp_eq_i32 s7, 0 // (op 17780) 
s_mov_b32 s15, scc // (op 17780) 

s_cmp_eq_i32 s15, 0 // (op 17780) 
s_cselect_b32 s7, s6, s8 // (op 17780) 


s_mul_hi_u32 s8, s7, s12 // (op 17780) 
s_sub_u32 s15, s7, s8 // (op 17780) 
s_lshr_b32 s28, s15, 1 // (op 17780) 

s_add_u32 s15, s28, s8 // (op 17780) 


s_lshr_b32 s8, s15, s10 // (op 17780) 

s_and_b32 s15, s10, 2147483648 // (op 17780) 
s_cmp_eq_i32 s15, 0 // (op 17780) 
s_mov_b32 s28, scc // (op 17780) 

s_cmp_eq_i32 s28, 0 // (op 17780) 
s_cselect_b32 s15, s7, s8 // (op 17780) 


s_mul_i32 s8, s7, s18 // (op 17780) 


v_lshl_add_u32 v16, s15, 7, v28 // (op 17780) 


s_sub_u32 s7, s6, s8 // (op 17780) 


v_mov_b32 v23, 0 // (op 17780) convert
v_mov_b32 v22, v16 // (op 17780) convert

v_lshl_add_u32 v28, s7, 7, v31 // (op 17780) 


v_mul_lo_u32 v17, v28, s31 // (op 17780) most significant: low of low * high
v_mul_hi_u32 v31, v28, s30 // (op 17780) most significant: high of low * low
v_mul_lo_u32 v16, v28, s30 // (op 17780) least significant: low of low * low
v_add_u32 v17, v17, v31 // (op 17780) most significant: sum


v_add_co_u32 v24, s[18:19], v22, v16 // (op 17780) least significant half
v_addc_co_u32 v25, s[18:19], v23, v17, s[18:19] // (op 17780) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 17780) 


v_lshrrev_b64 v[16:17], 3, v[24:25] // (op 17780) 

s_mul_i32 s19, s30, s3 // (op 17782) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 17782) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 17782) least significant
s_mul_hi_u32 s6, s30, s2 // (op 17782) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 17782) most significant: sum
s_add_u32 s19, s19, s6 // (op 17782) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 17782) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 17782) 

s_mov_b32 s68, s32 // (op 17782) reg expression
s_mov_b32 s69, s33 // (op 17782) reg expression
s_mov_b32 s70, s18 // (op 17782) reg expression
s_mov_b32 s71, 131072 // (op 17782) call()

v_lshrrev_b32 v28, 3, v1 // (op 17946) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 17946) 
v_and_b32 v31, -8, v1 // (op 17946) 
s_and_b32 s15, s18, 4294967294 // (op 17946) 
v_sub_u32 v22, v1, v31 // (op 17946) 

s_add_u32 s6, s0, s15 // (op 17946) 

v_lshlrev_b32 v31, 4, v22 // (op 17946) 

s_mul_i32 s15, s16, s50 // (op 17946) 
s_add_u32 s15, s15, s6 // (op 17946) 

s_add_u32 s6, s15, s9 // (op 17946) 

s_mul_hi_u32 s15, s6, s13 // (op 17946) 
s_sub_u32 s7, s6, s15 // (op 17946) 
s_lshr_b32 s8, s7, 1 // (op 17946) 

s_add_u32 s7, s8, s15 // (op 17946) 


s_lshr_b32 s8, s7, s11 // (op 17946) 

s_and_b32 s7, s11, 2147483648 // (op 17946) 
s_cmp_eq_i32 s7, 0 // (op 17946) 
s_mov_b32 s15, scc // (op 17946) 

s_cmp_eq_i32 s15, 0 // (op 17946) 
s_cselect_b32 s7, s6, s8 // (op 17946) 


s_mul_hi_u32 s8, s7, s12 // (op 17946) 
s_sub_u32 s15, s7, s8 // (op 17946) 
s_lshr_b32 s28, s15, 1 // (op 17946) 

s_add_u32 s15, s28, s8 // (op 17946) 


s_lshr_b32 s8, s15, s10 // (op 17946) 

s_and_b32 s15, s10, 2147483648 // (op 17946) 
s_cmp_eq_i32 s15, 0 // (op 17946) 
s_mov_b32 s28, scc // (op 17946) 

s_cmp_eq_i32 s28, 0 // (op 17946) 
s_cselect_b32 s15, s7, s8 // (op 17946) 


s_mul_i32 s8, s15, s14 // (op 17946) 

s_sub_u32 s15, s7, s8 // (op 17946) 

s_mul_i32 s8, s7, s18 // (op 17946) 


v_lshl_add_u32 v22, s15, 7, v28 // (op 17946) 


s_sub_u32 s7, s6, s8 // (op 17946) 


v_mul_lo_u32 v35, v22, s35 // (op 17946) most significant: low of low * high
v_mul_hi_u32 v28, v22, s34 // (op 17946) most significant: high of low * low
v_mul_lo_u32 v34, v22, s34 // (op 17946) least significant: low of low * low
v_add_u32 v35, v35, v28 // (op 17946) most significant: sum


v_lshl_add_u32 v28, s7, 7, v31 // (op 17946) 


v_mov_b32 v23, 0 // (op 17946) convert
v_mov_b32 v22, v28 // (op 17946) convert

v_add_co_u32 v24, s[18:19], v22, v34 // (op 17946) least significant half
v_addc_co_u32 v25, s[18:19], v23, v35, s[18:19] // (op 17946) most significant half



s_mul_i32 s19, s34, s23 // (op 17948) most significant: low of low * high
s_mul_i32 s15, s35, s22 // (op 17948) most significant: low of high * low
s_mul_i32 s18, s34, s22 // (op 17948) least significant
s_mul_hi_u32 s6, s34, s22 // (op 17948) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 17948) most significant: sum
s_add_u32 s19, s19, s6 // (op 17948) most significant: sum


s_mov_b32 s72, s36 // (op 17948) reg expression
s_mov_b32 s73, s37 // (op 17948) reg expression
s_mov_b32 s74, s18 // (op 17948) reg expression
s_mov_b32 s75, 131072 // (op 17948) call()

v_and_b32 v26, -128, v1 // (op 18074) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 18074) 
v_sub_u32 v28, v1, v26 // (op 18074) 

s_and_b32 s15, s18, 4294967294 // (op 18074) 
v_lshrrev_b32 v26, 7, v1 // (op 18074) 
s_add_u32 s6, s0, s15 // (op 18074) 

v_lshlrev_b32 v31, 1, v26 // (op 18074) 

s_mul_i32 s15, s16, s50 // (op 18074) 
s_add_u32 s15, s15, s6 // (op 18074) 

s_add_u32 s6, s15, s9 // (op 18074) 

s_mul_hi_u32 s15, s6, s13 // (op 18074) 
s_sub_u32 s7, s6, s15 // (op 18074) 
s_lshr_b32 s8, s7, 1 // (op 18074) 

s_add_u32 s7, s8, s15 // (op 18074) 


s_lshr_b32 s8, s7, s11 // (op 18074) 

s_and_b32 s7, s11, 2147483648 // (op 18074) 
s_cmp_eq_i32 s7, 0 // (op 18074) 
s_mov_b32 s15, scc // (op 18074) 

s_cmp_eq_i32 s15, 0 // (op 18074) 
s_cselect_b32 s7, s6, s8 // (op 18074) 


s_mul_hi_u32 s8, s7, s12 // (op 18074) 
s_sub_u32 s15, s7, s8 // (op 18074) 
s_lshr_b32 s28, s15, 1 // (op 18074) 

s_add_u32 s15, s28, s8 // (op 18074) 


s_lshr_b32 s8, s15, s10 // (op 18074) 

s_and_b32 s15, s10, 2147483648 // (op 18074) 
s_cmp_eq_i32 s15, 0 // (op 18074) 
s_mov_b32 s28, scc // (op 18074) 

s_cmp_eq_i32 s28, 0 // (op 18074) 
s_cselect_b32 s15, s7, s8 // (op 18074) 


s_mul_i32 s8, s7, s18 // (op 18074) 


v_lshl_add_u32 v26, s15, 7, v28 // (op 18074) 


s_sub_u32 s7, s6, s8 // (op 18074) 


v_lshl_add_u32 v28, s7, 2, v31 // (op 18074) 


v_mul_lo_u32 v31, v28, s38 // (op 18074) 

v_add_u32 v28, v26, v31 // (op 18074) 


s_mul_i32 s19, s38, s43 // (op 18076) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 18076) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 18076) least significant
s_mul_hi_u32 s6, s38, s42 // (op 18076) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18076) most significant: sum
s_add_u32 s19, s19, s6 // (op 18076) most significant: sum


s_mov_b32 s76, s40 // (op 18076) reg expression
s_mov_b32 s77, s41 // (op 18076) reg expression
s_mov_b32 s78, s18 // (op 18076) reg expression
s_mov_b32 s79, 131072 // (op 18076) call()

v_lshrrev_b32 v26, 2, v1 // (op 18202) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 18202) 
v_lshlrev_b32 v31, 1, v26 // (op 18202) 

s_and_b32 s15, s18, 4294967294 // (op 18202) 
v_and_b32 v26, -4, v1 // (op 18202) 
s_add_u32 s6, s0, s15 // (op 18202) 

v_sub_u32 v22, v1, v26 // (op 18202) 

s_mul_i32 s15, s16, s50 // (op 18202) 
s_add_u32 s15, s15, s6 // (op 18202) 

s_add_u32 s6, s15, s9 // (op 18202) 

s_mul_hi_u32 s15, s6, s13 // (op 18202) 
s_sub_u32 s7, s6, s15 // (op 18202) 
s_lshr_b32 s8, s7, 1 // (op 18202) 

s_add_u32 s7, s8, s15 // (op 18202) 


s_lshr_b32 s8, s7, s11 // (op 18202) 

s_and_b32 s7, s11, 2147483648 // (op 18202) 
s_cmp_eq_i32 s7, 0 // (op 18202) 
s_mov_b32 s15, scc // (op 18202) 

s_cmp_eq_i32 s15, 0 // (op 18202) 
s_cselect_b32 s7, s6, s8 // (op 18202) 


s_mul_hi_u32 s8, s7, s12 // (op 18202) 
s_sub_u32 s15, s7, s8 // (op 18202) 
s_lshr_b32 s28, s15, 1 // (op 18202) 

s_add_u32 s15, s28, s8 // (op 18202) 


s_lshr_b32 s8, s15, s10 // (op 18202) 

s_and_b32 s15, s10, 2147483648 // (op 18202) 
s_cmp_eq_i32 s15, 0 // (op 18202) 
s_mov_b32 s28, scc // (op 18202) 

s_cmp_eq_i32 s28, 0 // (op 18202) 
s_cselect_b32 s15, s7, s8 // (op 18202) 


s_mul_i32 s8, s15, s14 // (op 18202) 

s_sub_u32 s15, s7, s8 // (op 18202) 

v_lshl_add_u32 v23, s15, 7, v31 // (op 18202) 


s_mul_i32 s8, s7, s18 // (op 18202) 


v_mul_lo_u32 v26, v23, s44 // (op 18202) 

s_sub_u32 s7, s6, s8 // (op 18202) 


v_lshl_add_u32 v23, s7, 2, v22 // (op 18202) 


v_add_u32 v22, v23, v26 // (op 18202) 


s_mul_i32 s19, s44, s23 // (op 18204) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 18204) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 18204) least significant
s_mul_hi_u32 s6, s44, s22 // (op 18204) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 18204) most significant: sum
s_add_u32 s19, s19, s6 // (op 18204) most significant: sum


s_mov_b32 s80, s46 // (op 18204) reg expression
s_mov_b32 s81, s47 // (op 18204) reg expression
s_mov_b32 s82, s18 // (op 18204) reg expression
s_mov_b32 s83, 131072 // (op 18204) call()

v_and_b32 v23, -4, v1 // (op 18749) 
v_sub_u32 v26, v1, v23 // (op 18749) 

v_lshlrev_b32 v23, 5, v26 // (op 18749) 

v_lshrrev_b32 v26, 2, v1 // (op 18749) 
v_lshl_add_u32 v31, v26, 7, v23 // (op 18749) 


v_lshlrev_b32 v23, 2, v31 // (op 18749) 

v_lshrrev_b32 v26, 3, v23 // (op 18749) 

s_nop 0
v_readfirstlane_b32 s15, v26 // (op 18749) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -8, v1 // (op 18824) 
v_sub_u32 v26, v1, v23 // (op 18824) 

v_lshlrev_b32 v23, 4, v26 // (op 18824) 

v_lshrrev_b32 v26, 3, v1 // (op 18824) 
v_lshl_add_u32 v31, v26, 7, v23 // (op 18824) 


s_nop 0
v_readfirstlane_b32 s6, v31 // (op 18824) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -128, v1 // (op 18899) 
v_sub_u32 v26, v1, v23 // (op 18899) 

v_lshrrev_b32 v23, 7, v1 // (op 18899) 
v_lshl_add_u32 v31, v23, 8, v26 // (op 18899) 


v_and_b32 v23, -4, v1 // (op 18975) 
v_sub_u32 v26, v1, v23 // (op 18975) 

v_lshrrev_b32 v23, 2, v1 // (op 18975) 
v_lshl_add_u32 v34, v23, 3, v26 // (op 18975) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2870) 
s_and_b32 s7, s18, 4294967294 // (op 2870) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 2869) 
s_mov_b32 s8, s18 // (op 2869) convert

s_cmp_lt_u32 s7, s8 // (op 2869) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_2869 // (op 2869) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_2869 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_2869:
s_barrier  // (op 19690) 
s_add_u32 s7, s7, 1 // (op 2871) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 17695) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 17695) 

v_lshrrev_b64 v[20:21], 3, s[28:29] // (op 17695) 


v_mov_b64 v[26:27], v[16:17] // (op 17695) 
v_mov_b64 v[32:33], v[26:27] // (op 17695) reg expression
s_add_u32 s8, 16384, s15 // (op 17695) 
s_mov_b32 m0, s8 // (op 17695) reg expression
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 17695) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v20 // (op 17695) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 17695) most significant half

s_add_u32 m0, m0, 4096 // (op 17695) 
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 17695) Load value direct to lds




s_lshl_b64 s[18:19], s[30:31], 7 // (op 17770) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 17770) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 17770) 

v_mov_b32 v23, s18 // (op 17770) 
v_mov_b32 v20, s19 // (op 17770) 
v_add_co_u32 v16, s[28:29], v16, v23 // (op 17770) least significant half
v_addc_co_u32 v17, s[28:29], v17, v20, s[28:29] // (op 17770) most significant half




s_barrier  // (op 19709) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 17701) 
v_mov_b64 v[20:21], s[18:19] // (op 17701) convert


v_mov_b64 v[26:27], v[24:25] // (op 17701) 
v_mov_b64 v[32:33], v[26:27] // (op 17701) 
s_mov_b32 m0, s6 // (op 17701) reg expression
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 17701) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v20 // (op 17701) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 17701) most significant half

s_add_u32 m0, m0, 4096 // (op 17701) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 17701) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v20 // (op 17701) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 17701) most significant half

s_add_u32 m0, m0, 4096 // (op 17701) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 17701) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v20 // (op 17701) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 17701) most significant half

s_add_u32 m0, m0, 4096 // (op 17701) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 17701) Load value direct to lds



v_add_co_u32 v24, vcc, 128, v24 // (op 17938) least significant half
v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 17938) most significant half

v_mov_b64 v[20:21], s[38:39] // (op 469) convert

v_mov_b32 v23, v28 // (op 469) 
v_mov_b32 v26, v23 // (op 469) reg expression
buffer_load_ubyte v32, v26, s[76:79], 0 offen offset: 0 // (op 469) Load value
v_add_u32 v26, v26, v20 // (op 469) 
buffer_load_ubyte v33, v26, s[76:79], 0 offen offset: 0 // (op 469) Load value



s_lshl_b32 s8, s38, 2 // (op 18066) 
v_add_u32 v28, v28, s8 // (op 18066) 


s_barrier  // (op 19729) 
v_mov_b64 v[20:21], s[44:45] // (op 470) convert




v_mov_b32 v23, v22 // (op 470) 
v_mov_b32 v26, v23 // (op 470) 
buffer_load_ubyte v36, v26, s[80:83], 0 offen offset: 0 // (op 470) Load value
v_add_u32 v26, v26, v20 // (op 470) 
s_waitcnt vmcnt(2)
ds_write_b8 v31, v32 offset:49664 // (op 466) Store local data 
buffer_load_ubyte v37, v26, s[80:83], 0 offen offset: 0 // (op 470) Load value



v_add_u32 v22, 4, v22 // (op 18194) 
s_waitcnt vmcnt(2)
ds_write_b8 v31, v33 offset:49792 // (op 466) Store local data 

s_barrier  // (op 19696) 



s_barrier  // (op 19749) 
ds_read_b64_tr_b4 v[38:39], v7 offset:16384 // (op 457) Transpose load local data 



ds_read_b64_tr_b4 v[40:41], v7 offset:17408 // (op 457) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v34, v36 offset:49152 // (op 467) Store local data 
s_barrier  // (op 19716) 
s_waitcnt vmcnt(0)
ds_write_b8 v34, v37 offset:49156 // (op 467) Store local data 




ds_read_b128 v[42:45], v8 // (op 459) Load local data 
ds_read_b128 v[46:49], v8 offset:64 // (op 459) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 19736) 


ds_read_u8 v23, v6 offset:49664 // (op 461) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 19756) 



ds_read_u8 v20, v9 offset:49152 // (op 463) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:41], v[42:49], a[0:3], v23, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 464) 
ds_read_b128 v[50:53], v8 offset:4096 // (op 628) Load local data 
ds_read_b128 v[54:57], v8 offset:4160 // (op 628) Load local data 



ds_read_u8 v21, v9 offset:49280 // (op 632) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:41], v[50:57], a[4:7], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 633) 
ds_read_b128 v[58:61], v8 offset:8192 // (op 804) Load local data 
ds_read_b128 v[62:65], v8 offset:8256 // (op 804) Load local data 



ds_read_u8 v26, v9 offset:49408 // (op 808) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[38:41], v[58:65], a[8:11], v23, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 809) 
ds_read_b128 v[66:69], v8 offset:12288 // (op 980) Load local data 
ds_read_b128 v[70:73], v8 offset:12352 // (op 980) Load local data 



ds_read_u8 v27, v9 offset:49536 // (op 984) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[38:41], v[66:73], a[12:15], v23, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 985) 
ds_read_b64_tr_b4 v[74:75], v7 offset:16400 // (op 1242) Transpose load local data 


ds_read_b64_tr_b4 v[76:77], v7 offset:17424 // (op 1242) Transpose load local data 


ds_read_u8 v23, v6 offset:49696 // (op 1245) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[42:49], a[16:19], v23, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1247) 
ds_read_b64_tr_b4 v[38:39], v7 offset:16416 // (op 2012) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[50:57], a[20:23], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1278) 
ds_read_b64_tr_b4 v[40:41], v7 offset:17440 // (op 2012) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[58:65], a[24:27], v23, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1309) 
ds_read_u8 v29, v6 offset:49728 // (op 2015) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[66:73], a[28:31], v23, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1340) 
ds_read_b64_tr_b4 v[78:79], v7 offset:16432 // (op 2782) Transpose load local data 


ds_read_b64_tr_b4 v[80:81], v7 offset:17456 // (op 2782) Transpose load local data 


s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[38:41], v[42:49], a[32:35], v29, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2017) 
ds_read_u8 v23, v6 offset:49760 // (op 2785) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[38:41], v[50:57], a[36:39], v29, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2048) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[38:41], v[58:65], a[40:43], v29, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2079) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[38:41], v[66:73], a[44:47], v29, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2110) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[78:81], v[42:49], a[48:51], v23, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2787) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[78:81], v[50:57], a[52:55], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2818) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[78:81], v[58:65], a[56:59], v23, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2849) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[78:81], v[66:73], a[60:63], v23, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2880) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 2869) 
s_mov_b32 s8, s18 // (op 2869) convert

s_cmp_lt_u32 s7, s8 // (op 2869) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_2869 // (op 2869) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_2869 if true)
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_2869:









s_barrier  // (op 19920) 




v_cmp_eq_f32 vcc, 0.00000, v18 // (op 234) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_234 // (op 234) Condition: False, jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_234
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


v_accvgpr_read v14, a4 // (op 3839) 
v_accvgpr_read v15, a5 // (op 3839) 
v_accvgpr_read v16, a6 // (op 3839) 
v_accvgpr_read v17, a7 // (op 3839) 
v_mul_f32 v20, v19, v14 // (op 3839) 
v_mul_f32 v21, v19, v15 // (op 3839) 
v_mul_f32 v22, v19, v16 // (op 3839) 
v_mul_f32 v23, v19, v17 // (op 3839) 



s_lshl_b32 s15, s24, 7 // (op 3841) 
v_add_u32 v0, v5, s15 // (op 3841) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 3841) Store value


v_accvgpr_read v14, a8 // (op 3851) 
v_accvgpr_read v15, a9 // (op 3851) 
v_accvgpr_read v16, a10 // (op 3851) 
v_accvgpr_read v17, a11 // (op 3851) 
v_mul_f32 v20, v19, v14 // (op 3851) 
v_mul_f32 v21, v19, v15 // (op 3851) 
v_mul_f32 v22, v19, v16 // (op 3851) 
v_mul_f32 v23, v19, v17 // (op 3851) 



s_lshl_b32 s15, s24, 8 // (op 3853) 
v_add_u32 v0, v5, s15 // (op 3853) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 3853) Store value


v_accvgpr_read v14, a12 // (op 3863) 
v_accvgpr_read v15, a13 // (op 3863) 
v_accvgpr_read v16, a14 // (op 3863) 
v_accvgpr_read v17, a15 // (op 3863) 
v_mul_f32 v20, v19, v14 // (op 3863) 
v_mul_f32 v21, v19, v15 // (op 3863) 
v_mul_f32 v22, v19, v16 // (op 3863) 
v_mul_f32 v23, v19, v17 // (op 3863) 



s_lshl_b32 s15, s24, 7 // (op 3865) 
s_mul_i32 s6, 3, s15 // (op 3865) 

v_add_u32 v0, v5, s6 // (op 3865) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 3865) Store value


v_accvgpr_read v14, a16 // (op 3897) 
v_accvgpr_read v15, a17 // (op 3897) 
v_accvgpr_read v16, a18 // (op 3897) 
v_accvgpr_read v17, a19 // (op 3897) 
v_mul_f32 v20, v19, v14 // (op 3897) 
v_mul_f32 v21, v19, v15 // (op 3897) 
v_mul_f32 v22, v19, v16 // (op 3897) 
v_mul_f32 v23, v19, v17 // (op 3897) 



v_add_u32 v0, 128, v5 // (op 3900) 

buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 3900) Store value


v_accvgpr_read v14, a20 // (op 3910) 
v_accvgpr_read v15, a21 // (op 3910) 
v_accvgpr_read v16, a22 // (op 3910) 
v_accvgpr_read v17, a23 // (op 3910) 
v_mul_f32 v20, v19, v14 // (op 3910) 
v_mul_f32 v21, v19, v15 // (op 3910) 
v_mul_f32 v22, v19, v16 // (op 3910) 
v_mul_f32 v23, v19, v17 // (op 3910) 



s_lshl_b32 s15, s24, 7 // (op 3902) 
s_add_u32 s15, s15, 128 // (op 3902) 
v_add_u32 v0, v5, s15 // (op 3902) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 3902) Store value


v_accvgpr_read v14, a24 // (op 3915) 
v_accvgpr_read v15, a25 // (op 3915) 
v_accvgpr_read v16, a26 // (op 3915) 
v_accvgpr_read v17, a27 // (op 3915) 
v_mul_f32 v20, v19, v14 // (op 3915) 
v_mul_f32 v21, v19, v15 // (op 3915) 
v_mul_f32 v22, v19, v16 // (op 3915) 
v_mul_f32 v23, v19, v17 // (op 3915) 



s_lshl_b32 s15, s24, 8 // (op 3904) 
s_add_u32 s15, s15, 128 // (op 3904) 
v_add_u32 v0, v5, s15 // (op 3904) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 3904) Store value


v_accvgpr_read v14, a28 // (op 3920) 
v_accvgpr_read v15, a29 // (op 3920) 
v_accvgpr_read v16, a30 // (op 3920) 
v_accvgpr_read v17, a31 // (op 3920) 
v_mul_f32 v20, v19, v14 // (op 3920) 
v_mul_f32 v21, v19, v15 // (op 3920) 
v_mul_f32 v22, v19, v16 // (op 3920) 
v_mul_f32 v23, v19, v17 // (op 3920) 



s_lshl_b32 s15, s24, 7 // (op 3906) 
s_mul_i32 s6, 3, s15 // (op 3906) 
s_add_u32 s6, s6, 128 // (op 3906) 

v_add_u32 v0, v5, s6 // (op 3906) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 3906) Store value


v_accvgpr_read v14, a32 // (op 3965) 
v_accvgpr_read v15, a33 // (op 3965) 
v_accvgpr_read v16, a34 // (op 3965) 
v_accvgpr_read v17, a35 // (op 3965) 
v_mul_f32 v20, v19, v14 // (op 3965) 
v_mul_f32 v21, v19, v15 // (op 3965) 
v_mul_f32 v22, v19, v16 // (op 3965) 
v_mul_f32 v23, v19, v17 // (op 3965) 



v_add_u32 v0, 256, v5 // (op 3968) 

buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 3968) Store value


v_accvgpr_read v14, a36 // (op 3978) 
v_accvgpr_read v15, a37 // (op 3978) 
v_accvgpr_read v16, a38 // (op 3978) 
v_accvgpr_read v17, a39 // (op 3978) 
v_mul_f32 v20, v19, v14 // (op 3978) 
v_mul_f32 v21, v19, v15 // (op 3978) 
v_mul_f32 v22, v19, v16 // (op 3978) 
v_mul_f32 v23, v19, v17 // (op 3978) 



s_lshl_b32 s15, s24, 7 // (op 3970) 
s_add_u32 s15, s15, 256 // (op 3970) 
v_add_u32 v0, v5, s15 // (op 3970) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 3970) Store value


v_accvgpr_read v14, a40 // (op 3983) 
v_accvgpr_read v15, a41 // (op 3983) 
v_accvgpr_read v16, a42 // (op 3983) 
v_accvgpr_read v17, a43 // (op 3983) 
v_mul_f32 v20, v19, v14 // (op 3983) 
v_mul_f32 v21, v19, v15 // (op 3983) 
v_mul_f32 v22, v19, v16 // (op 3983) 
v_mul_f32 v23, v19, v17 // (op 3983) 



s_lshl_b32 s15, s24, 8 // (op 3972) 
s_add_u32 s15, s15, 256 // (op 3972) 
v_add_u32 v0, v5, s15 // (op 3972) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 3972) Store value


v_accvgpr_read v14, a44 // (op 3988) 
v_accvgpr_read v15, a45 // (op 3988) 
v_accvgpr_read v16, a46 // (op 3988) 
v_accvgpr_read v17, a47 // (op 3988) 
v_mul_f32 v20, v19, v14 // (op 3988) 
v_mul_f32 v21, v19, v15 // (op 3988) 
v_mul_f32 v22, v19, v16 // (op 3988) 
v_mul_f32 v23, v19, v17 // (op 3988) 



s_lshl_b32 s15, s24, 7 // (op 3974) 
s_mul_i32 s6, 3, s15 // (op 3974) 
s_add_u32 s6, s6, 256 // (op 3974) 

v_add_u32 v0, v5, s6 // (op 3974) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 3974) Store value


v_accvgpr_read v14, a48 // (op 4033) 
v_accvgpr_read v15, a49 // (op 4033) 
v_accvgpr_read v16, a50 // (op 4033) 
v_accvgpr_read v17, a51 // (op 4033) 
v_mul_f32 v20, v19, v14 // (op 4033) 
v_mul_f32 v21, v19, v15 // (op 4033) 
v_mul_f32 v22, v19, v16 // (op 4033) 
v_mul_f32 v23, v19, v17 // (op 4033) 



v_add_u32 v0, 384, v5 // (op 4036) 

buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 4036) Store value


v_accvgpr_read v14, a52 // (op 4046) 
v_accvgpr_read v15, a53 // (op 4046) 
v_accvgpr_read v16, a54 // (op 4046) 
v_accvgpr_read v17, a55 // (op 4046) 
v_mul_f32 v20, v19, v14 // (op 4046) 
v_mul_f32 v21, v19, v15 // (op 4046) 
v_mul_f32 v22, v19, v16 // (op 4046) 
v_mul_f32 v23, v19, v17 // (op 4046) 



s_lshl_b32 s15, s24, 7 // (op 4038) 
s_add_u32 s15, s15, 384 // (op 4038) 
v_add_u32 v0, v5, s15 // (op 4038) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 4038) Store value


v_accvgpr_read v14, a56 // (op 4051) 
v_accvgpr_read v15, a57 // (op 4051) 
v_accvgpr_read v16, a58 // (op 4051) 
v_accvgpr_read v17, a59 // (op 4051) 
v_mul_f32 v20, v19, v14 // (op 4051) 
v_mul_f32 v21, v19, v15 // (op 4051) 
v_mul_f32 v22, v19, v16 // (op 4051) 
v_mul_f32 v23, v19, v17 // (op 4051) 



s_lshl_b32 s15, s24, 8 // (op 4040) 
s_add_u32 s15, s15, 384 // (op 4040) 
v_add_u32 v0, v5, s15 // (op 4040) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 4040) Store value


v_accvgpr_read v14, a60 // (op 4056) 
v_accvgpr_read v15, a61 // (op 4056) 
v_accvgpr_read v16, a62 // (op 4056) 
v_accvgpr_read v17, a63 // (op 4056) 
v_mul_f32 v20, v19, v14 // (op 4056) 
v_mul_f32 v21, v19, v15 // (op 4056) 
v_mul_f32 v22, v19, v16 // (op 4056) 
v_mul_f32 v23, v19, v17 // (op 4056) 



s_lshl_b32 s15, s24, 7 // (op 4042) 
s_mul_i32 s6, 3, s15 // (op 4042) 
s_add_u32 s6, s6, 384 // (op 4042) 

v_add_u32 v0, v5, s6 // (op 4042) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 4042) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_234 // (op 234) Condition: Done, jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_234
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_234:


v_add_u32 v0, 256, v4 // (op 4321) 

buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 4321) Load value

v_accvgpr_read v20, a32 // (op 4309) 
v_accvgpr_read v21, a33 // (op 4309) 
v_accvgpr_read v22, a34 // (op 4309) 
v_accvgpr_read v23, a35 // (op 4309) 
v_mul_f32 v24, v19, v20 // (op 4309) 
v_mul_f32 v25, v19, v21 // (op 4309) 
v_mul_f32 v26, v19, v22 // (op 4309) 
v_mul_f32 v27, v19, v23 // (op 4309) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 4320) 
v_mul_f32 v21, v18, v15 // (op 4320) 
v_mul_f32 v22, v18, v16 // (op 4320) 
v_mul_f32 v23, v18, v17 // (op 4320) 

v_add_f32 v14, v20, v24 // (op 4310) 
v_add_f32 v15, v21, v25 // (op 4310) 
v_add_f32 v16, v22, v26 // (op 4310) 
v_add_f32 v17, v23, v27 // (op 4310) 




v_add_u32 v0, 256, v5 // (op 4312) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 4312) Store value




s_lshl_b32 s15, s4, 7 // (op 4325) 
s_add_u32 s15, s15, 256 // (op 4325) 
v_add_u32 v0, v4, s15 // (op 4325) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 4325) Load value

v_accvgpr_read v20, a36 // (op 4335) 
v_accvgpr_read v21, a37 // (op 4335) 
v_accvgpr_read v22, a38 // (op 4335) 
v_accvgpr_read v23, a39 // (op 4335) 
v_mul_f32 v24, v19, v20 // (op 4335) 
v_mul_f32 v25, v19, v21 // (op 4335) 
v_mul_f32 v26, v19, v22 // (op 4335) 
v_mul_f32 v27, v19, v23 // (op 4335) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 4323) 
v_mul_f32 v21, v18, v15 // (op 4323) 
v_mul_f32 v22, v18, v16 // (op 4323) 
v_mul_f32 v23, v18, v17 // (op 4323) 

v_add_f32 v14, v20, v24 // (op 4324) 
v_add_f32 v15, v21, v25 // (op 4324) 
v_add_f32 v16, v22, v26 // (op 4324) 
v_add_f32 v17, v23, v27 // (op 4324) 




s_lshl_b32 s15, s24, 7 // (op 4314) 
s_add_u32 s15, s15, 256 // (op 4314) 
v_add_u32 v0, v5, s15 // (op 4314) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 4314) Store value




s_lshl_b32 s15, s4, 8 // (op 4329) 
s_add_u32 s15, s15, 256 // (op 4329) 
v_add_u32 v0, v4, s15 // (op 4329) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 4329) Load value

v_accvgpr_read v20, a40 // (op 4337) 
v_accvgpr_read v21, a41 // (op 4337) 
v_accvgpr_read v22, a42 // (op 4337) 
v_accvgpr_read v23, a43 // (op 4337) 
v_mul_f32 v24, v19, v20 // (op 4337) 
v_mul_f32 v25, v19, v21 // (op 4337) 
v_mul_f32 v26, v19, v22 // (op 4337) 
v_mul_f32 v27, v19, v23 // (op 4337) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 4327) 
v_mul_f32 v21, v18, v15 // (op 4327) 
v_mul_f32 v22, v18, v16 // (op 4327) 
v_mul_f32 v23, v18, v17 // (op 4327) 

v_add_f32 v14, v20, v24 // (op 4328) 
v_add_f32 v15, v21, v25 // (op 4328) 
v_add_f32 v16, v22, v26 // (op 4328) 
v_add_f32 v17, v23, v27 // (op 4328) 




s_lshl_b32 s15, s24, 8 // (op 4316) 
s_add_u32 s15, s15, 256 // (op 4316) 
v_add_u32 v0, v5, s15 // (op 4316) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 4316) Store value




s_lshl_b32 s15, s4, 7 // (op 4333) 
s_mul_i32 s6, 3, s15 // (op 4333) 
s_add_u32 s6, s6, 256 // (op 4333) 

v_add_u32 v0, v4, s6 // (op 4333) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 4333) Load value

v_accvgpr_read v20, a44 // (op 4339) 
v_accvgpr_read v21, a45 // (op 4339) 
v_accvgpr_read v22, a46 // (op 4339) 
v_accvgpr_read v23, a47 // (op 4339) 
v_mul_f32 v24, v19, v20 // (op 4339) 
v_mul_f32 v25, v19, v21 // (op 4339) 
v_mul_f32 v26, v19, v22 // (op 4339) 
v_mul_f32 v27, v19, v23 // (op 4339) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 4331) 
v_mul_f32 v21, v18, v15 // (op 4331) 
v_mul_f32 v22, v18, v16 // (op 4331) 
v_mul_f32 v23, v18, v17 // (op 4331) 

v_add_f32 v14, v20, v24 // (op 4332) 
v_add_f32 v15, v21, v25 // (op 4332) 
v_add_f32 v16, v22, v26 // (op 4332) 
v_add_f32 v17, v23, v27 // (op 4332) 




s_lshl_b32 s15, s24, 7 // (op 4318) 
s_mul_i32 s6, 3, s15 // (op 4318) 
s_add_u32 s6, s6, 256 // (op 4318) 

v_add_u32 v0, v5, s6 // (op 4318) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 4318) Store value




v_add_u32 v0, 128, v4 // (op 4242) 

s_nop 0
buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 4242) Load value

v_accvgpr_read v20, a16 // (op 4230) 
v_accvgpr_read v21, a17 // (op 4230) 
v_accvgpr_read v22, a18 // (op 4230) 
v_accvgpr_read v23, a19 // (op 4230) 
v_mul_f32 v24, v19, v20 // (op 4230) 
v_mul_f32 v25, v19, v21 // (op 4230) 
v_mul_f32 v26, v19, v22 // (op 4230) 
v_mul_f32 v27, v19, v23 // (op 4230) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 4241) 
v_mul_f32 v21, v18, v15 // (op 4241) 
v_mul_f32 v22, v18, v16 // (op 4241) 
v_mul_f32 v23, v18, v17 // (op 4241) 

v_add_f32 v14, v20, v24 // (op 4231) 
v_add_f32 v15, v21, v25 // (op 4231) 
v_add_f32 v16, v22, v26 // (op 4231) 
v_add_f32 v17, v23, v27 // (op 4231) 




v_add_u32 v0, 128, v5 // (op 4233) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 4233) Store value




s_lshl_b32 s15, s4, 7 // (op 4246) 
s_add_u32 s15, s15, 128 // (op 4246) 
v_add_u32 v0, v4, s15 // (op 4246) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 4246) Load value

v_accvgpr_read v20, a20 // (op 4256) 
v_accvgpr_read v21, a21 // (op 4256) 
v_accvgpr_read v22, a22 // (op 4256) 
v_accvgpr_read v23, a23 // (op 4256) 
v_mul_f32 v24, v19, v20 // (op 4256) 
v_mul_f32 v25, v19, v21 // (op 4256) 
v_mul_f32 v26, v19, v22 // (op 4256) 
v_mul_f32 v27, v19, v23 // (op 4256) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 4244) 
v_mul_f32 v21, v18, v15 // (op 4244) 
v_mul_f32 v22, v18, v16 // (op 4244) 
v_mul_f32 v23, v18, v17 // (op 4244) 

v_add_f32 v14, v20, v24 // (op 4245) 
v_add_f32 v15, v21, v25 // (op 4245) 
v_add_f32 v16, v22, v26 // (op 4245) 
v_add_f32 v17, v23, v27 // (op 4245) 




s_lshl_b32 s15, s24, 7 // (op 4235) 
s_add_u32 s15, s15, 128 // (op 4235) 
v_add_u32 v0, v5, s15 // (op 4235) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 4235) Store value




s_lshl_b32 s15, s4, 8 // (op 4250) 
s_add_u32 s15, s15, 128 // (op 4250) 
v_add_u32 v0, v4, s15 // (op 4250) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 4250) Load value

v_accvgpr_read v20, a24 // (op 4258) 
v_accvgpr_read v21, a25 // (op 4258) 
v_accvgpr_read v22, a26 // (op 4258) 
v_accvgpr_read v23, a27 // (op 4258) 
v_mul_f32 v24, v19, v20 // (op 4258) 
v_mul_f32 v25, v19, v21 // (op 4258) 
v_mul_f32 v26, v19, v22 // (op 4258) 
v_mul_f32 v27, v19, v23 // (op 4258) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 4248) 
v_mul_f32 v21, v18, v15 // (op 4248) 
v_mul_f32 v22, v18, v16 // (op 4248) 
v_mul_f32 v23, v18, v17 // (op 4248) 

v_add_f32 v14, v20, v24 // (op 4249) 
v_add_f32 v15, v21, v25 // (op 4249) 
v_add_f32 v16, v22, v26 // (op 4249) 
v_add_f32 v17, v23, v27 // (op 4249) 




s_lshl_b32 s15, s24, 8 // (op 4237) 
s_add_u32 s15, s15, 128 // (op 4237) 
v_add_u32 v0, v5, s15 // (op 4237) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 4237) Store value




s_lshl_b32 s15, s4, 7 // (op 4254) 
s_mul_i32 s6, 3, s15 // (op 4254) 
s_add_u32 s6, s6, 128 // (op 4254) 

v_add_u32 v0, v4, s6 // (op 4254) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 4254) Load value

v_accvgpr_read v20, a28 // (op 4260) 
v_accvgpr_read v21, a29 // (op 4260) 
v_accvgpr_read v22, a30 // (op 4260) 
v_accvgpr_read v23, a31 // (op 4260) 
v_mul_f32 v24, v19, v20 // (op 4260) 
v_mul_f32 v25, v19, v21 // (op 4260) 
v_mul_f32 v26, v19, v22 // (op 4260) 
v_mul_f32 v27, v19, v23 // (op 4260) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 4252) 
v_mul_f32 v21, v18, v15 // (op 4252) 
v_mul_f32 v22, v18, v16 // (op 4252) 
v_mul_f32 v23, v18, v17 // (op 4252) 

v_add_f32 v14, v20, v24 // (op 4253) 
v_add_f32 v15, v21, v25 // (op 4253) 
v_add_f32 v16, v22, v26 // (op 4253) 
v_add_f32 v17, v23, v27 // (op 4253) 




s_lshl_b32 s15, s24, 7 // (op 4239) 
s_mul_i32 s6, 3, s15 // (op 4239) 
s_add_u32 s6, s6, 128 // (op 4239) 

v_add_u32 v0, v5, s6 // (op 4239) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 4239) Store value




v_add_u32 v0, 384, v4 // (op 4400) 

s_nop 0
buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 4400) Load value

v_accvgpr_read v20, a48 // (op 4388) 
v_accvgpr_read v21, a49 // (op 4388) 
v_accvgpr_read v22, a50 // (op 4388) 
v_accvgpr_read v23, a51 // (op 4388) 
v_mul_f32 v24, v19, v20 // (op 4388) 
v_mul_f32 v25, v19, v21 // (op 4388) 
v_mul_f32 v26, v19, v22 // (op 4388) 
v_mul_f32 v27, v19, v23 // (op 4388) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 4399) 
v_mul_f32 v21, v18, v15 // (op 4399) 
v_mul_f32 v22, v18, v16 // (op 4399) 
v_mul_f32 v23, v18, v17 // (op 4399) 

v_add_f32 v14, v20, v24 // (op 4389) 
v_add_f32 v15, v21, v25 // (op 4389) 
v_add_f32 v16, v22, v26 // (op 4389) 
v_add_f32 v17, v23, v27 // (op 4389) 




v_add_u32 v0, 384, v5 // (op 4391) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 4391) Store value




s_lshl_b32 s15, s4, 7 // (op 4404) 
s_add_u32 s15, s15, 384 // (op 4404) 
v_add_u32 v0, v4, s15 // (op 4404) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 4404) Load value

v_accvgpr_read v20, a52 // (op 4414) 
v_accvgpr_read v21, a53 // (op 4414) 
v_accvgpr_read v22, a54 // (op 4414) 
v_accvgpr_read v23, a55 // (op 4414) 
v_mul_f32 v24, v19, v20 // (op 4414) 
v_mul_f32 v25, v19, v21 // (op 4414) 
v_mul_f32 v26, v19, v22 // (op 4414) 
v_mul_f32 v27, v19, v23 // (op 4414) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 4402) 
v_mul_f32 v21, v18, v15 // (op 4402) 
v_mul_f32 v22, v18, v16 // (op 4402) 
v_mul_f32 v23, v18, v17 // (op 4402) 

v_add_f32 v14, v20, v24 // (op 4403) 
v_add_f32 v15, v21, v25 // (op 4403) 
v_add_f32 v16, v22, v26 // (op 4403) 
v_add_f32 v17, v23, v27 // (op 4403) 




s_lshl_b32 s15, s24, 7 // (op 4393) 
s_add_u32 s15, s15, 384 // (op 4393) 
v_add_u32 v0, v5, s15 // (op 4393) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 4393) Store value




s_lshl_b32 s15, s4, 8 // (op 4408) 
s_add_u32 s15, s15, 384 // (op 4408) 
v_add_u32 v0, v4, s15 // (op 4408) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 4408) Load value

v_accvgpr_read v20, a56 // (op 4416) 
v_accvgpr_read v21, a57 // (op 4416) 
v_accvgpr_read v22, a58 // (op 4416) 
v_accvgpr_read v23, a59 // (op 4416) 
v_mul_f32 v24, v19, v20 // (op 4416) 
v_mul_f32 v25, v19, v21 // (op 4416) 
v_mul_f32 v26, v19, v22 // (op 4416) 
v_mul_f32 v27, v19, v23 // (op 4416) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 4406) 
v_mul_f32 v21, v18, v15 // (op 4406) 
v_mul_f32 v22, v18, v16 // (op 4406) 
v_mul_f32 v23, v18, v17 // (op 4406) 

v_add_f32 v14, v20, v24 // (op 4407) 
v_add_f32 v15, v21, v25 // (op 4407) 
v_add_f32 v16, v22, v26 // (op 4407) 
v_add_f32 v17, v23, v27 // (op 4407) 




s_lshl_b32 s15, s24, 8 // (op 4395) 
s_add_u32 s15, s15, 384 // (op 4395) 
v_add_u32 v0, v5, s15 // (op 4395) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 4395) Store value




s_lshl_b32 s15, s4, 7 // (op 4412) 
s_mul_i32 s6, 3, s15 // (op 4412) 
s_add_u32 s6, s6, 384 // (op 4412) 

v_add_u32 v0, v4, s6 // (op 4412) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 4412) Load value

v_accvgpr_read v20, a60 // (op 4418) 
v_accvgpr_read v21, a61 // (op 4418) 
v_accvgpr_read v22, a62 // (op 4418) 
v_accvgpr_read v23, a63 // (op 4418) 
v_mul_f32 v24, v19, v20 // (op 4418) 
v_mul_f32 v25, v19, v21 // (op 4418) 
v_mul_f32 v26, v19, v22 // (op 4418) 
v_mul_f32 v27, v19, v23 // (op 4418) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 4410) 
v_mul_f32 v21, v18, v15 // (op 4410) 
v_mul_f32 v22, v18, v16 // (op 4410) 
v_mul_f32 v23, v18, v17 // (op 4410) 

v_add_f32 v14, v20, v24 // (op 4411) 
v_add_f32 v15, v21, v25 // (op 4411) 
v_add_f32 v16, v22, v26 // (op 4411) 
v_add_f32 v17, v23, v27 // (op 4411) 




s_lshl_b32 s15, s24, 7 // (op 4397) 
s_mul_i32 s6, 3, s15 // (op 4397) 
s_add_u32 s6, s6, 384 // (op 4397) 

v_add_u32 v0, v5, s6 // (op 4397) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 4397) Store value




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




s_lshl_b32 s15, s4, 7 // (op 4156) 
v_add_u32 v0, v4, s15 // (op 4156) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 4156) Load value

v_accvgpr_read v20, a4 // (op 4153) 
v_accvgpr_read v21, a5 // (op 4153) 
v_accvgpr_read v22, a6 // (op 4153) 
v_accvgpr_read v23, a7 // (op 4153) 
v_mul_f32 v24, v19, v20 // (op 4153) 
v_mul_f32 v25, v19, v21 // (op 4153) 
v_mul_f32 v26, v19, v22 // (op 4153) 
v_mul_f32 v27, v19, v23 // (op 4153) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 4157) 
v_mul_f32 v21, v18, v15 // (op 4157) 
v_mul_f32 v22, v18, v16 // (op 4157) 
v_mul_f32 v23, v18, v17 // (op 4157) 

v_add_f32 v14, v20, v24 // (op 4154) 
v_add_f32 v15, v21, v25 // (op 4154) 
v_add_f32 v16, v22, v26 // (op 4154) 
v_add_f32 v17, v23, v27 // (op 4154) 




s_lshl_b32 s15, s24, 7 // (op 4155) 
v_add_u32 v0, v5, s15 // (op 4155) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 4155) Store value




s_lshl_b32 s15, s4, 8 // (op 4168) 
v_add_u32 v0, v4, s15 // (op 4168) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 4168) Load value

v_accvgpr_read v20, a8 // (op 4165) 
v_accvgpr_read v21, a9 // (op 4165) 
v_accvgpr_read v22, a10 // (op 4165) 
v_accvgpr_read v23, a11 // (op 4165) 
v_mul_f32 v24, v19, v20 // (op 4165) 
v_mul_f32 v25, v19, v21 // (op 4165) 
v_mul_f32 v26, v19, v22 // (op 4165) 
v_mul_f32 v27, v19, v23 // (op 4165) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 4169) 
v_mul_f32 v21, v18, v15 // (op 4169) 
v_mul_f32 v22, v18, v16 // (op 4169) 
v_mul_f32 v23, v18, v17 // (op 4169) 

v_add_f32 v14, v20, v24 // (op 4166) 
v_add_f32 v15, v21, v25 // (op 4166) 
v_add_f32 v16, v22, v26 // (op 4166) 
v_add_f32 v17, v23, v27 // (op 4166) 




s_lshl_b32 s15, s24, 8 // (op 4167) 
v_add_u32 v0, v5, s15 // (op 4167) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 4167) Store value




s_lshl_b32 s15, s4, 7 // (op 4180) 
s_mul_i32 s6, 3, s15 // (op 4180) 

v_add_u32 v0, v4, s6 // (op 4180) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 4180) Load value

v_accvgpr_read v20, a12 // (op 4177) 
v_accvgpr_read v21, a13 // (op 4177) 
v_accvgpr_read v22, a14 // (op 4177) 
v_accvgpr_read v23, a15 // (op 4177) 
v_mul_f32 v24, v19, v20 // (op 4177) 
v_mul_f32 v25, v19, v21 // (op 4177) 
v_mul_f32 v26, v19, v22 // (op 4177) 
v_mul_f32 v27, v19, v23 // (op 4177) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 4181) 
v_mul_f32 v21, v18, v15 // (op 4181) 
v_mul_f32 v22, v18, v16 // (op 4181) 
v_mul_f32 v23, v18, v17 // (op 4181) 

v_add_f32 v14, v20, v24 // (op 4178) 
v_add_f32 v15, v21, v25 // (op 4178) 
v_add_f32 v16, v22, v26 // (op 4178) 
v_add_f32 v17, v23, v27 // (op 4178) 




s_lshl_b32 s15, s24, 7 // (op 4179) 
s_mul_i32 s6, 3, s15 // (op 4179) 

v_add_u32 v0, v5, s6 // (op 4179) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 4179) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_234:
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
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_DPStreamTileLoop_441 // (op 441) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_DPStreamTileLoop_441 if true)
RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_441:

























































s_endpgm  // End of RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
.LRR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_end:
.size RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2, .LRR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_end-RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
  .amdhsa_next_free_vgpr 212
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 51200 // lds bytes
  .amdhsa_accum_offset 148
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
    DataTypeB: F8
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
    macrotile: [128, 128, 128]
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
  MIWaveTile: [4, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
    .symbol: RR_GEMM_NN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 51200
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 89
    .vgpr_count: 146
    .agpr_count: 64
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

