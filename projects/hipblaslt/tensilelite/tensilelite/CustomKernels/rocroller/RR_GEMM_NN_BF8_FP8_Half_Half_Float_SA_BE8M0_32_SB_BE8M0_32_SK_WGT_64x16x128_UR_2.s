// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2
.p2align 8
.type RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2,@function
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_exec_begin:


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


v_and_b32 v0, 15, v1 // (op 2758) 
v_and_b32 v3, 48, v1 // (op 2758) 
v_lshl_add_u32 v4, v0, 7, v3 // (op 2758) 


v_and_b32 v0, 15, v1 // (op 3016) 
v_and_b32 v3, 48, v1 // (op 3016) 
v_lshl_add_u32 v5, v0, 7, v3 // (op 3016) 


v_and_b32 v0, 15, v1 // (op 3230) 
v_and_b32 v3, 48, v1 // (op 3230) 
v_lshl_add_u32 v6, v0, 7, v3 // (op 3230) 


v_and_b32 v0, 15, v1 // (op 3342) 
v_and_b32 v3, 48, v1 // (op 3342) 
v_lshl_add_u32 v7, v0, 7, v3 // (op 3342) 


v_and_b32 v0, 48, v1 // (op 2688) 
v_and_b32 v3, 63, v1 // (op 2688) 
v_sub_u32 v8, v3, v0 // (op 2688) 

v_lshrrev_b32 v3, 1, v8 // (op 2688) 
v_add_lshl_u32 v9, v0, v3, 6 // (op 2688) 


v_and_b32 v0, 1, v8 // (op 2688) 

v_lshlrev_b32 v3, 3, v0 // (op 2688) 

v_lshrrev_b32 v0, 6, v1 // (op 2688) 
v_lshl_add_u32 v8, v0, 4, v3 // (op 2688) 


v_add_u32 v0, v9, v8 // (op 2688) 


v_lshrrev_b32 v3, 6, v1 // (op 2829) 
v_and_b32 v8, 15, v1 // (op 2829) 
v_lshl_add_u32 v9, v3, 4, v8 // (op 2829) 


v_and_b32 v3, 63, v1 // (op 2829) 
v_lshrrev_b32 v8, 4, v3 // (op 2829) 

v_lshl_add_u32 v3, v8, 6, v9 // (op 2829) 


v_and_b32 v8, 48, v1 // (op 2946) 
v_and_b32 v9, 63, v1 // (op 2946) 
v_sub_u32 v10, v9, v8 // (op 2946) 

v_lshrrev_b32 v9, 1, v10 // (op 2946) 
v_add_lshl_u32 v11, v8, v9, 6 // (op 2946) 


v_and_b32 v8, 1, v10 // (op 2946) 

v_lshlrev_b32 v9, 3, v8 // (op 2946) 

v_lshrrev_b32 v8, 6, v1 // (op 2946) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 2946) 


v_add_u32 v8, v11, v10 // (op 2946) 


v_lshrrev_b32 v9, 6, v1 // (op 3087) 
v_and_b32 v10, 15, v1 // (op 3087) 
v_lshl_add_u32 v11, v9, 4, v10 // (op 3087) 


v_and_b32 v9, 63, v1 // (op 3087) 
v_lshrrev_b32 v10, 4, v9 // (op 3087) 

v_lshl_add_u32 v9, v10, 6, v11 // (op 3087) 


v_and_b32 v10, 48, v1 // (op 3191) 
v_and_b32 v11, 63, v1 // (op 3191) 
v_sub_u32 v12, v11, v10 // (op 3191) 

v_lshrrev_b32 v11, 1, v12 // (op 3191) 
v_add_lshl_u32 v13, v10, v11, 6 // (op 3191) 


v_and_b32 v10, 1, v12 // (op 3191) 

v_lshlrev_b32 v11, 3, v10 // (op 3191) 

v_lshrrev_b32 v10, 6, v1 // (op 3191) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 3191) 


v_add_u32 v10, v13, v12 // (op 3191) 


v_lshrrev_b32 v11, 6, v1 // (op 3259) 
v_and_b32 v12, 15, v1 // (op 3259) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 3259) 


v_and_b32 v11, 63, v1 // (op 3259) 
v_lshrrev_b32 v12, 4, v11 // (op 3259) 

v_lshl_add_u32 v11, v12, 6, v13 // (op 3259) 


v_and_b32 v12, 48, v1 // (op 3303) 
v_and_b32 v13, 63, v1 // (op 3303) 
v_sub_u32 v14, v13, v12 // (op 3303) 

v_lshrrev_b32 v13, 1, v14 // (op 3303) 
v_add_lshl_u32 v15, v12, v13, 6 // (op 3303) 


v_and_b32 v12, 1, v14 // (op 3303) 

v_lshlrev_b32 v13, 3, v12 // (op 3303) 

v_lshrrev_b32 v12, 6, v1 // (op 3303) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 3303) 


v_add_u32 v12, v15, v14 // (op 3303) 


v_lshrrev_b32 v13, 6, v1 // (op 3371) 
v_and_b32 v14, 15, v1 // (op 3371) 
v_lshl_add_u32 v15, v13, 4, v14 // (op 3371) 


v_and_b32 v13, 63, v1 // (op 3371) 
v_lshrrev_b32 v14, 4, v13 // (op 3371) 

v_lshl_add_u32 v13, v14, 6, v15 // (op 3371) 


v_mov_b32 v14, s6 // (op 2) Move value
v_mov_b32 v15, s7 // (op 39) Move value


s_mov_b32 s0, 0 // (op 379) call()
s_mov_b32 s1, 0 // (op 380) call()
s_mov_b32 s6, 0 // (op 381) call()
s_mov_b32 s7, 0 // (op 382) call()
s_mov_b32 s18, 0 // (op 383) call()
s_mov_b32 s19, 0 // (op 384) call()
s_mov_b32 s0, 0 // (op 279) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 278) 
s_cmp_gt_u32 s9, s52 // (op 278) 
s_mov_b32 s53, scc // (op 278) 
s_sub_u32 s54, s9, s52 // (op 278) 

s_cmp_eq_i32 s53, 0 // (op 278) 
s_cselect_b32 s52, 0, s54 // (op 278) 


s_cmp_lt_u32 s8, s52 // (op 278) 
s_mov_b32 s53, scc // (op 278) 
s_cmp_eq_i32 s53, 0 // (op 278) 
s_cselect_b32 s54, s52, s8 // (op 278) 


s_cmp_lt_u32 s0, s54 // (op 278) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_278 // (op 278) Condition: Top (jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_278 if false)
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_SKStreamTileLoop_278:
v_lshrrev_b32 v16, 6, v1 // (op 2604) 
s_mul_i32 s52, s16, s8 // (op 2604) 
s_add_u32 s52, s52, s0 // (op 2604) 
v_and_b32 v17, 63, v1 // (op 2604) 
s_mul_hi_u32 s53, s52, s13 // (op 2604) 
v_lshrrev_b32 v18, 4, v17 // (op 2604) 
s_sub_u32 s54, s52, s53 // (op 2604) 
v_lshlrev_b32 v19, 2, v18 // (op 2604) 

v_lshl_add_u32 v18, v16, 4, v19 // (op 2604) 


s_lshr_b32 s55, s54, 1 // (op 2604) 

s_add_u32 s54, s55, s53 // (op 2604) 


v_lshrrev_b32 v16, 2, v17 // (op 2604) 

s_lshr_b32 s53, s54, s11 // (op 2604) 

v_and_b32 v17, 3, v16 // (op 2604) 

s_and_b32 s54, s11, 2147483648 // (op 2604) 
v_and_b32 v16, 3, v1 // (op 2604) 
s_cmp_eq_i32 s54, 0 // (op 2604) 
s_mov_b32 s55, scc // (op 2604) 

v_lshl_add_u32 v19, v17, 2, v16 // (op 2604) 


s_cmp_eq_i32 s55, 0 // (op 2604) 
s_cselect_b32 s54, s52, s53 // (op 2604) 



s_mul_hi_u32 s52, s54, s12 // (op 2604) 
s_sub_u32 s53, s54, s52 // (op 2604) 
s_lshr_b32 s55, s53, 1 // (op 2604) 

s_add_u32 s53, s55, s52 // (op 2604) 


s_lshr_b32 s52, s53, s10 // (op 2604) 

s_and_b32 s53, s10, 2147483648 // (op 2604) 
s_cmp_eq_i32 s53, 0 // (op 2604) 
s_mov_b32 s55, scc // (op 2604) 

s_cmp_eq_i32 s55, 0 // (op 2604) 
s_cselect_b32 s53, s54, s52 // (op 2604) 


s_mul_i32 s52, s53, s14 // (op 2604) 
v_lshl_add_u32 v16, s53, 6, v18 // (op 2604) 


s_sub_u32 s53, s54, s52 // (op 2604) 


v_lshl_add_u32 v17, s53, 4, v19 // (op 2604) 


v_mul_lo_u32 v18, v17, s4 // (op 2604) 

v_add_lshl_u32 v17, v16, v18, 1 // (op 2604) 


s_mul_i32 s53, s4, s23 // (op 2606) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 2606) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 2606) least significant
s_mul_hi_u32 s55, s4, s22 // (op 2606) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2606) most significant: sum
s_add_u32 s53, s53, s55 // (op 2606) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 2606) 

s_mov_b32 s56, s20 // (op 2606) reg expression
s_mov_b32 s57, s21 // (op 2606) reg expression
s_mov_b32 s58, s54 // (op 2606) reg expression
s_mov_b32 s59, 131072 // (op 2606) call()

v_and_b32 v16, 48, v1 // (op 2705) 
v_and_b32 v18, 63, v1 // (op 2705) 
v_sub_u32 v19, v18, v16 // (op 2705) 

v_lshrrev_b32 v18, 1, v19 // (op 2705) 
v_add_lshl_u32 v20, v16, v18, 6 // (op 2705) 


v_and_b32 v16, 1, v19 // (op 2705) 

v_lshlrev_b32 v18, 3, v16 // (op 2705) 

v_lshrrev_b32 v16, 6, v1 // (op 2705) 
v_lshl_add_u32 v19, v16, 4, v18 // (op 2705) 


v_add_u32 v16, v20, v19 // (op 2705) 


v_and_b32 v18, 15, v1 // (op 2770) 
v_and_b32 v19, 48, v1 // (op 2770) 
v_lshl_add_u32 v20, v18, 7, v19 // (op 2770) 


v_lshrrev_b32 v18, 6, v1 // (op 2846) 
v_and_b32 v19, 15, v1 // (op 2846) 
v_lshl_add_u32 v21, v18, 4, v19 // (op 2846) 


v_and_b32 v18, 63, v1 // (op 2846) 
v_lshrrev_b32 v19, 4, v18 // (op 2846) 

v_lshl_add_u32 v18, v19, 6, v21 // (op 2846) 


v_lshrrev_b32 v19, 6, v1 // (op 2864) 
s_mul_i32 s52, s16, s8 // (op 2864) 
s_add_u32 s52, s52, s0 // (op 2864) 
v_and_b32 v21, 63, v1 // (op 2864) 
s_mul_hi_u32 s53, s52, s13 // (op 2864) 
v_lshrrev_b32 v22, 4, v21 // (op 2864) 
s_sub_u32 s54, s52, s53 // (op 2864) 
v_lshlrev_b32 v23, 2, v22 // (op 2864) 

s_lshr_b32 s55, s54, 1 // (op 2864) 

v_lshl_add_u32 v22, v19, 4, v23 // (op 2864) 


s_add_u32 s54, s55, s53 // (op 2864) 


v_lshrrev_b32 v19, 2, v21 // (op 2864) 

s_lshr_b32 s53, s54, s11 // (op 2864) 

v_and_b32 v21, 3, v19 // (op 2864) 

s_and_b32 s54, s11, 2147483648 // (op 2864) 
v_and_b32 v19, 3, v1 // (op 2864) 
s_cmp_eq_i32 s54, 0 // (op 2864) 
s_mov_b32 s55, scc // (op 2864) 

v_lshl_add_u32 v23, v21, 2, v19 // (op 2864) 


s_cmp_eq_i32 s55, 0 // (op 2864) 
s_cselect_b32 s54, s52, s53 // (op 2864) 



s_mul_hi_u32 s55, s54, s12 // (op 2864) 
s_sub_u32 s52, s54, s55 // (op 2864) 
s_lshr_b32 s53, s52, 1 // (op 2864) 

s_add_u32 s52, s53, s55 // (op 2864) 


s_lshr_b32 s53, s52, s10 // (op 2864) 

s_and_b32 s52, s10, 2147483648 // (op 2864) 
s_cmp_eq_i32 s52, 0 // (op 2864) 
s_mov_b32 s55, scc // (op 2864) 

s_cmp_eq_i32 s55, 0 // (op 2864) 
s_cselect_b32 s52, s54, s53 // (op 2864) 


s_mul_i32 s53, s52, s14 // (op 2864) 
v_lshl_add_u32 v19, s52, 6, v22 // (op 2864) 


s_sub_u32 s52, s54, s53 // (op 2864) 


v_lshl_add_u32 v21, s52, 4, v23 // (op 2864) 


v_mul_lo_u32 v22, v21, s24 // (op 2864) 

v_add_lshl_u32 v21, v19, v22, 1 // (op 2864) 


s_mul_i32 s53, s24, s23 // (op 2866) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 2866) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 2866) least significant
s_mul_hi_u32 s55, s24, s22 // (op 2866) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2866) most significant: sum
s_add_u32 s53, s53, s55 // (op 2866) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 2866) 

s_mov_b32 s60, s26 // (op 2866) reg expression
s_mov_b32 s61, s27 // (op 2866) reg expression
s_mov_b32 s62, s54 // (op 2866) reg expression
s_mov_b32 s63, 131072 // (op 2866) call()

v_lshrrev_b32 v19, 4, v1 // (op 3122) 
v_lshlrev_b32 v22, 2, v19 // (op 3122) 

v_lshl_add_u32 v19, s16, 6, v22 // (op 3122) 

v_and_b32 v22, -16, v1 // (op 3122) 
v_sub_u32 v23, v1, v22 // (op 3122) 

v_lshl_add_u32 v22, v19, 4, v23 // (op 3122) 


v_lshlrev_b32 v19, 2, v22 // (op 3122) 

s_lshl_b32 s52, s15, 12 // (op 3124) 
s_mov_b32 s64, s28 // (op 3124) reg expression
s_mov_b32 s65, s29 // (op 3124) reg expression
s_mov_b32 s66, s52 // (op 3124) reg expression
s_mov_b32 s67, 131072 // (op 3124) call()

s_mul_i32 s6, s8, s16 // (op 283) 
s_add_u32 s6, s6, s0 // (op 283) 
s_mul_hi_u32 s52, s6, s13 // (op 284) 
s_sub_u32 s53, s6, s52 // (op 284) 
s_lshr_b32 s54, s53, 1 // (op 284) 

s_add_u32 s53, s54, s52 // (op 284) 


s_lshr_b32 s52, s53, s11 // (op 284) 

s_and_b32 s53, s11, 2147483648 // (op 284) 
s_cmp_eq_i32 s53, 0 // (op 284) 
s_mov_b32 s54, scc // (op 284) 

s_cmp_eq_i32 s54, 0 // (op 284) 
s_cselect_b32 s53, s6, s52 // (op 284) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 284) 
s_mul_i32 s52, s53, s54 // (op 284) 


s_sub_u32 s18, s6, s52 // (op 284) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 285) 
s_cmp_gt_u32 s52, s18 // (op 285) 
s_mov_b32 s54, scc // (op 285) 
s_sub_u32 s55, s52, s18 // (op 285) 

s_cmp_eq_i32 s54, 0 // (op 285) 
s_cselect_b32 s52, 0, s55 // (op 285) 


s_cmp_gt_u32 s8, s0 // (op 285) 
s_mov_b32 s53, scc // (op 285) 
s_sub_u32 s54, s8, s0 // (op 285) 
s_cmp_eq_i32 s53, 0 // (op 285) 
s_cselect_b32 s55, 0, s54 // (op 285) 


s_cmp_lt_u32 s52, s55 // (op 285) 
s_mov_b32 s53, scc // (op 285) 
s_cmp_eq_i32 s53, 0 // (op 285) 
s_cselect_b32 s54, s55, s52 // (op 285) 



s_cmp_gt_u32 s9, s6 // (op 285) 
s_mov_b32 s55, scc // (op 285) 
s_sub_u32 s52, s9, s6 // (op 285) 
s_cmp_eq_i32 s55, 0 // (op 285) 
s_cselect_b32 s53, 0, s52 // (op 285) 


s_cmp_lt_u32 s54, s53 // (op 285) 
s_mov_b32 s52, scc // (op 285) 
s_cmp_eq_i32 s52, 0 // (op 285) 
s_cselect_b32 s7, s53, s54 // (op 285) 



s_add_u32 s52, s18, s7 // (op 286) 
s_sub_u32 s19, s52, 1 // (op 286) 

v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 1004) call()
v_accvgpr_write a5, 0.00000 // (op 1004) call()
v_accvgpr_write a6, 0.00000 // (op 1004) call()
v_accvgpr_write a7, 0.00000 // (op 1004) call()
v_and_b32 v22, -4, v1 // (op 2176) 
s_mul_i32 s52, s16, s8 // (op 2176) 
s_add_u32 s52, s52, s0 // (op 2176) 
v_sub_u32 v23, v1, v22 // (op 2176) 

s_mul_hi_u32 s53, s52, s13 // (op 2176) 
v_lshlrev_b32 v22, 4, v23 // (op 2176) 

s_sub_u32 s54, s52, s53 // (op 2176) 
v_lshrrev_b32 v23, 2, v1 // (op 2176) 
s_lshr_b32 s55, s54, 1 // (op 2176) 

s_add_u32 s54, s55, s53 // (op 2176) 


s_lshr_b32 s53, s54, s11 // (op 2176) 

s_and_b32 s54, s11, 2147483648 // (op 2176) 
s_cmp_eq_i32 s54, 0 // (op 2176) 
s_mov_b32 s55, scc // (op 2176) 

s_cmp_eq_i32 s55, 0 // (op 2176) 
s_cselect_b32 s54, s52, s53 // (op 2176) 


s_mul_hi_u32 s53, s54, s12 // (op 2176) 
s_sub_u32 s55, s54, s53 // (op 2176) 
s_lshr_b32 s68, s55, 1 // (op 2176) 

s_add_u32 s55, s68, s53 // (op 2176) 


s_lshr_b32 s53, s55, s10 // (op 2176) 

s_and_b32 s55, s10, 2147483648 // (op 2176) 
s_cmp_eq_i32 s55, 0 // (op 2176) 
s_mov_b32 s68, scc // (op 2176) 

s_cmp_eq_i32 s68, 0 // (op 2176) 
s_cselect_b32 s55, s54, s53 // (op 2176) 


v_lshl_add_u32 v24, s55, 6, v22 // (op 2176) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 2176) 
v_mov_b32 v27, 0 // (op 2176) convert
v_mov_b32 v26, v24 // (op 2176) convert

s_mul_i32 s53, s54, s68 // (op 2176) 


s_sub_u32 s54, s52, s53 // (op 2176) 


v_lshl_add_u32 v22, s54, 7, v23 // (op 2176) 


v_mul_lo_u32 v25, v22, s31 // (op 2176) most significant: low of low * high
v_mul_hi_u32 v23, v22, s30 // (op 2176) most significant: high of low * low
v_mul_lo_u32 v24, v22, s30 // (op 2176) least significant: low of low * low
v_add_u32 v25, v25, v23 // (op 2176) most significant: sum


v_add_co_u32 v22, s[52:53], v26, v24 // (op 2176) least significant half
v_addc_co_u32 v23, s[52:53], v27, v25, s[52:53] // (op 2176) most significant half



s_mul_i32 s53, s30, s3 // (op 2178) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 2178) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 2178) least significant
s_mul_hi_u32 s55, s30, s2 // (op 2178) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2178) most significant: sum
s_add_u32 s53, s53, s55 // (op 2178) most significant: sum


s_mov_b32 s68, s32 // (op 2178) reg expression
s_mov_b32 s69, s33 // (op 2178) reg expression
s_mov_b32 s70, s52 // (op 2178) reg expression
s_mov_b32 s71, 131072 // (op 2178) call()

v_and_b32 v24, -4, v1 // (op 2195) 
s_mul_i32 s52, s16, s8 // (op 2195) 
s_add_u32 s52, s52, s0 // (op 2195) 
v_sub_u32 v25, v1, v24 // (op 2195) 

s_mul_hi_u32 s53, s52, s13 // (op 2195) 
v_lshlrev_b32 v24, 4, v25 // (op 2195) 

s_sub_u32 s54, s52, s53 // (op 2195) 
v_lshrrev_b32 v25, 2, v1 // (op 2195) 
s_lshr_b32 s55, s54, 1 // (op 2195) 

s_add_u32 s54, s55, s53 // (op 2195) 


s_lshr_b32 s53, s54, s11 // (op 2195) 

s_and_b32 s54, s11, 2147483648 // (op 2195) 
s_cmp_eq_i32 s54, 0 // (op 2195) 
s_mov_b32 s55, scc // (op 2195) 

s_cmp_eq_i32 s55, 0 // (op 2195) 
s_cselect_b32 s54, s52, s53 // (op 2195) 


s_mul_hi_u32 s53, s54, s12 // (op 2195) 
s_sub_u32 s55, s54, s53 // (op 2195) 
s_lshr_b32 s72, s55, 1 // (op 2195) 

s_add_u32 s55, s72, s53 // (op 2195) 


s_lshr_b32 s53, s55, s10 // (op 2195) 

s_and_b32 s55, s10, 2147483648 // (op 2195) 
s_cmp_eq_i32 s55, 0 // (op 2195) 
s_mov_b32 s72, scc // (op 2195) 

s_cmp_eq_i32 s72, 0 // (op 2195) 
s_cselect_b32 s55, s54, s53 // (op 2195) 


v_lshl_add_u32 v26, s55, 6, v24 // (op 2195) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 2195) 
v_mov_b32 v29, 0 // (op 2195) convert
v_mov_b32 v28, v26 // (op 2195) convert

s_mul_i32 s53, s54, s72 // (op 2195) 


s_sub_u32 s54, s52, s53 // (op 2195) 


v_lshl_add_u32 v24, s54, 7, v25 // (op 2195) 


v_mul_lo_u32 v27, v24, s31 // (op 2195) most significant: low of low * high
v_mul_hi_u32 v25, v24, s30 // (op 2195) most significant: high of low * low
v_mul_lo_u32 v26, v24, s30 // (op 2195) least significant: low of low * low
v_add_u32 v27, v27, v25 // (op 2195) most significant: sum


v_add_co_u32 v24, s[52:53], v28, v26 // (op 2195) least significant half
v_addc_co_u32 v25, s[52:53], v29, v27, s[52:53] // (op 2195) most significant half



s_mul_i32 s53, s30, s3 // (op 2197) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 2197) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 2197) least significant
s_mul_hi_u32 s55, s30, s2 // (op 2197) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2197) most significant: sum
s_add_u32 s53, s53, s55 // (op 2197) most significant: sum


s_mov_b32 s68, s32 // (op 2197) reg expression
s_mov_b32 s69, s33 // (op 2197) reg expression
s_mov_b32 s70, s52 // (op 2197) reg expression
s_mov_b32 s71, 131072 // (op 2197) call()

v_lshrrev_b32 v26, 3, v1 // (op 2304) 
s_mul_i32 s52, s16, s8 // (op 2304) 
s_add_u32 s52, s52, s0 // (op 2304) 
v_and_b32 v27, -8, v1 // (op 2304) 
s_mul_hi_u32 s53, s52, s13 // (op 2304) 
v_sub_u32 v28, v1, v27 // (op 2304) 

s_sub_u32 s54, s52, s53 // (op 2304) 
v_lshlrev_b32 v27, 4, v28 // (op 2304) 

s_lshr_b32 s55, s54, 1 // (op 2304) 

s_add_u32 s54, s55, s53 // (op 2304) 


s_lshr_b32 s53, s54, s11 // (op 2304) 

s_and_b32 s54, s11, 2147483648 // (op 2304) 
s_cmp_eq_i32 s54, 0 // (op 2304) 
s_mov_b32 s55, scc // (op 2304) 

s_cmp_eq_i32 s55, 0 // (op 2304) 
s_cselect_b32 s54, s52, s53 // (op 2304) 


s_mul_hi_u32 s53, s54, s12 // (op 2304) 
s_sub_u32 s55, s54, s53 // (op 2304) 
s_lshr_b32 s72, s55, 1 // (op 2304) 

s_add_u32 s55, s72, s53 // (op 2304) 


s_lshr_b32 s53, s55, s10 // (op 2304) 

s_and_b32 s55, s10, 2147483648 // (op 2304) 
s_cmp_eq_i32 s55, 0 // (op 2304) 
s_mov_b32 s72, scc // (op 2304) 

s_cmp_eq_i32 s72, 0 // (op 2304) 
s_cselect_b32 s55, s54, s53 // (op 2304) 


s_mul_i32 s53, s55, s14 // (op 2304) 

s_sub_u32 s55, s54, s53 // (op 2304) 

v_lshl_add_u32 v28, s55, 4, v26 // (op 2304) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 2304) 
v_mul_lo_u32 v31, v28, s35 // (op 2304) most significant: low of low * high
v_mul_hi_u32 v26, v28, s34 // (op 2304) most significant: high of low * low
v_mul_lo_u32 v30, v28, s34 // (op 2304) least significant: low of low * low
v_add_u32 v31, v31, v26 // (op 2304) most significant: sum


s_mul_i32 s53, s54, s72 // (op 2304) 


s_sub_u32 s54, s52, s53 // (op 2304) 


v_lshl_add_u32 v26, s54, 7, v27 // (op 2304) 


v_mov_b32 v29, 0 // (op 2304) convert
v_mov_b32 v28, v26 // (op 2304) convert

v_add_co_u32 v26, s[52:53], v28, v30 // (op 2304) least significant half
v_addc_co_u32 v27, s[52:53], v29, v31, s[52:53] // (op 2304) most significant half



s_mul_i32 s53, s34, s23 // (op 2306) most significant: low of low * high
s_mul_i32 s54, s35, s22 // (op 2306) most significant: low of high * low
s_mul_i32 s52, s34, s22 // (op 2306) least significant
s_mul_hi_u32 s55, s34, s22 // (op 2306) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2306) most significant: sum
s_add_u32 s53, s53, s55 // (op 2306) most significant: sum


s_mov_b32 s72, s36 // (op 2306) reg expression
s_mov_b32 s73, s37 // (op 2306) reg expression
s_mov_b32 s74, s52 // (op 2306) reg expression
s_mov_b32 s75, 131072 // (op 2306) call()

v_lshrrev_b32 v28, 3, v1 // (op 2323) 
s_mul_i32 s52, s16, s8 // (op 2323) 
s_add_u32 s52, s52, s0 // (op 2323) 
v_and_b32 v29, -8, v1 // (op 2323) 
s_mul_hi_u32 s53, s52, s13 // (op 2323) 
v_sub_u32 v30, v1, v29 // (op 2323) 

s_sub_u32 s54, s52, s53 // (op 2323) 
v_lshlrev_b32 v29, 4, v30 // (op 2323) 

s_lshr_b32 s55, s54, 1 // (op 2323) 

s_add_u32 s54, s55, s53 // (op 2323) 


s_lshr_b32 s53, s54, s11 // (op 2323) 

s_and_b32 s54, s11, 2147483648 // (op 2323) 
s_cmp_eq_i32 s54, 0 // (op 2323) 
s_mov_b32 s55, scc // (op 2323) 

s_cmp_eq_i32 s55, 0 // (op 2323) 
s_cselect_b32 s54, s52, s53 // (op 2323) 


s_mul_hi_u32 s53, s54, s12 // (op 2323) 
s_sub_u32 s55, s54, s53 // (op 2323) 
s_lshr_b32 s76, s55, 1 // (op 2323) 

s_add_u32 s55, s76, s53 // (op 2323) 


s_lshr_b32 s53, s55, s10 // (op 2323) 

s_and_b32 s55, s10, 2147483648 // (op 2323) 
s_cmp_eq_i32 s55, 0 // (op 2323) 
s_mov_b32 s76, scc // (op 2323) 

s_cmp_eq_i32 s76, 0 // (op 2323) 
s_cselect_b32 s55, s54, s53 // (op 2323) 


s_mul_i32 s53, s55, s14 // (op 2323) 

s_sub_u32 s55, s54, s53 // (op 2323) 

v_lshl_add_u32 v30, s55, 4, v28 // (op 2323) 


v_mul_lo_u32 v33, v30, s35 // (op 2323) most significant: low of low * high
v_mul_hi_u32 v28, v30, s34 // (op 2323) most significant: high of low * low
v_mul_lo_u32 v32, v30, s34 // (op 2323) least significant: low of low * low
v_add_u32 v33, v33, v28 // (op 2323) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 2323) 
s_mul_i32 s53, s54, s76 // (op 2323) 


s_sub_u32 s54, s52, s53 // (op 2323) 


v_lshl_add_u32 v28, s54, 7, v29 // (op 2323) 


v_mov_b32 v31, 0 // (op 2323) convert
v_mov_b32 v30, v28 // (op 2323) convert

v_add_co_u32 v28, s[52:53], v30, v32 // (op 2323) least significant half
v_addc_co_u32 v29, s[52:53], v31, v33, s[52:53] // (op 2323) most significant half



s_mul_i32 s53, s34, s23 // (op 2325) most significant: low of low * high
s_mul_i32 s54, s35, s22 // (op 2325) most significant: low of high * low
s_mul_i32 s52, s34, s22 // (op 2325) least significant
s_mul_hi_u32 s55, s34, s22 // (op 2325) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2325) most significant: sum
s_add_u32 s53, s53, s55 // (op 2325) most significant: sum


s_mov_b32 s72, s36 // (op 2325) reg expression
s_mov_b32 s73, s37 // (op 2325) reg expression
s_mov_b32 s74, s52 // (op 2325) reg expression
s_mov_b32 s75, 131072 // (op 2325) call()

v_and_b32 v30, -64, v1 // (op 2432) 
s_mul_i32 s52, s16, s8 // (op 2432) 
s_add_u32 s52, s52, s0 // (op 2432) 
v_sub_u32 v31, v1, v30 // (op 2432) 

s_mul_hi_u32 s53, s52, s13 // (op 2432) 
v_lshrrev_b32 v30, 6, v1 // (op 2432) 
s_sub_u32 s54, s52, s53 // (op 2432) 
v_lshlrev_b32 v32, 1, v30 // (op 2432) 

s_lshr_b32 s55, s54, 1 // (op 2432) 

s_add_u32 s54, s55, s53 // (op 2432) 


s_lshr_b32 s53, s54, s11 // (op 2432) 

s_and_b32 s54, s11, 2147483648 // (op 2432) 
s_cmp_eq_i32 s54, 0 // (op 2432) 
s_mov_b32 s55, scc // (op 2432) 

s_cmp_eq_i32 s55, 0 // (op 2432) 
s_cselect_b32 s54, s52, s53 // (op 2432) 


s_mul_hi_u32 s53, s54, s12 // (op 2432) 
s_sub_u32 s55, s54, s53 // (op 2432) 
s_lshr_b32 s76, s55, 1 // (op 2432) 

s_add_u32 s55, s76, s53 // (op 2432) 


s_lshr_b32 s53, s55, s10 // (op 2432) 

s_and_b32 s55, s10, 2147483648 // (op 2432) 
s_cmp_eq_i32 s55, 0 // (op 2432) 
s_mov_b32 s76, scc // (op 2432) 

s_cmp_eq_i32 s76, 0 // (op 2432) 
s_cselect_b32 s55, s54, s53 // (op 2432) 


v_lshl_add_u32 v30, s55, 6, v31 // (op 2432) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 2432) 
s_mul_i32 s53, s54, s76 // (op 2432) 


s_sub_u32 s54, s52, s53 // (op 2432) 


v_lshl_add_u32 v31, s54, 2, v32 // (op 2432) 


v_mul_lo_u32 v32, v31, s38 // (op 2432) 

v_add_u32 v31, v30, v32 // (op 2432) 


s_mul_i32 s53, s38, s43 // (op 2434) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 2434) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 2434) least significant
s_mul_hi_u32 s55, s38, s42 // (op 2434) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2434) most significant: sum
s_add_u32 s53, s53, s55 // (op 2434) most significant: sum


s_mov_b32 s76, s40 // (op 2434) reg expression
s_mov_b32 s77, s41 // (op 2434) reg expression
s_mov_b32 s78, s52 // (op 2434) reg expression
s_mov_b32 s79, 131072 // (op 2434) call()

v_and_b32 v30, -64, v1 // (op 2451) 
s_mul_i32 s52, s16, s8 // (op 2451) 
s_add_u32 s52, s52, s0 // (op 2451) 
v_sub_u32 v32, v1, v30 // (op 2451) 

s_mul_hi_u32 s53, s52, s13 // (op 2451) 
v_lshrrev_b32 v30, 6, v1 // (op 2451) 
s_sub_u32 s54, s52, s53 // (op 2451) 
v_lshlrev_b32 v33, 1, v30 // (op 2451) 

s_lshr_b32 s55, s54, 1 // (op 2451) 

s_add_u32 s54, s55, s53 // (op 2451) 


s_lshr_b32 s53, s54, s11 // (op 2451) 

s_and_b32 s54, s11, 2147483648 // (op 2451) 
s_cmp_eq_i32 s54, 0 // (op 2451) 
s_mov_b32 s55, scc // (op 2451) 

s_cmp_eq_i32 s55, 0 // (op 2451) 
s_cselect_b32 s54, s52, s53 // (op 2451) 


s_mul_hi_u32 s53, s54, s12 // (op 2451) 
s_sub_u32 s55, s54, s53 // (op 2451) 
s_lshr_b32 s80, s55, 1 // (op 2451) 

s_add_u32 s55, s80, s53 // (op 2451) 


s_lshr_b32 s53, s55, s10 // (op 2451) 

s_and_b32 s55, s10, 2147483648 // (op 2451) 
s_cmp_eq_i32 s55, 0 // (op 2451) 
s_mov_b32 s80, scc // (op 2451) 

s_cmp_eq_i32 s80, 0 // (op 2451) 
s_cselect_b32 s55, s54, s53 // (op 2451) 


v_lshl_add_u32 v30, s55, 6, v32 // (op 2451) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 2451) 
s_mul_i32 s53, s54, s80 // (op 2451) 


s_sub_u32 s54, s52, s53 // (op 2451) 


v_lshl_add_u32 v32, s54, 2, v33 // (op 2451) 


v_mul_lo_u32 v33, v32, s38 // (op 2451) 

v_add_u32 v32, v30, v33 // (op 2451) 


s_mul_i32 s53, s38, s43 // (op 2453) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 2453) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 2453) least significant
s_mul_hi_u32 s55, s38, s42 // (op 2453) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2453) most significant: sum
s_add_u32 s53, s53, s55 // (op 2453) most significant: sum


s_mov_b32 s76, s40 // (op 2453) reg expression
s_mov_b32 s77, s41 // (op 2453) reg expression
s_mov_b32 s78, s52 // (op 2453) reg expression
s_mov_b32 s79, 131072 // (op 2453) call()

v_and_b32 v30, 15, v1 // (op 2560) 
s_mul_i32 s52, s16, s8 // (op 2560) 
s_add_u32 s52, s52, s0 // (op 2560) 
v_and_b32 v33, 63, v1 // (op 2560) 
s_mul_hi_u32 s53, s52, s13 // (op 2560) 
v_lshrrev_b32 v34, 4, v33 // (op 2560) 

s_sub_u32 s54, s52, s53 // (op 2560) 
s_lshr_b32 s55, s54, 1 // (op 2560) 

s_add_u32 s54, s55, s53 // (op 2560) 


s_lshr_b32 s53, s54, s11 // (op 2560) 

s_and_b32 s54, s11, 2147483648 // (op 2560) 
s_cmp_eq_i32 s54, 0 // (op 2560) 
s_mov_b32 s55, scc // (op 2560) 

s_cmp_eq_i32 s55, 0 // (op 2560) 
s_cselect_b32 s54, s52, s53 // (op 2560) 


s_mul_hi_u32 s53, s54, s12 // (op 2560) 
s_sub_u32 s55, s54, s53 // (op 2560) 
s_lshr_b32 s80, s55, 1 // (op 2560) 

s_add_u32 s55, s80, s53 // (op 2560) 


s_lshr_b32 s53, s55, s10 // (op 2560) 

s_and_b32 s55, s10, 2147483648 // (op 2560) 
s_cmp_eq_i32 s55, 0 // (op 2560) 
s_mov_b32 s80, scc // (op 2560) 

s_cmp_eq_i32 s80, 0 // (op 2560) 
s_cselect_b32 s55, s54, s53 // (op 2560) 


s_mul_i32 s53, s55, s14 // (op 2560) 

s_sub_u32 s55, s54, s53 // (op 2560) 

v_lshl_add_u32 v33, s55, 4, v30 // (op 2560) 


v_mul_lo_u32 v30, v33, s44 // (op 2560) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 2560) 
s_mul_i32 s53, s54, s80 // (op 2560) 


s_sub_u32 s54, s52, s53 // (op 2560) 


v_lshl_add_u32 v33, s54, 2, v34 // (op 2560) 


v_add_u32 v34, v33, v30 // (op 2560) 


s_mul_i32 s53, s44, s23 // (op 2562) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 2562) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 2562) least significant
s_mul_hi_u32 s55, s44, s22 // (op 2562) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2562) most significant: sum
s_add_u32 s53, s53, s55 // (op 2562) most significant: sum


s_mov_b32 s80, s46 // (op 2562) reg expression
s_mov_b32 s81, s47 // (op 2562) reg expression
s_mov_b32 s82, s52 // (op 2562) reg expression
s_mov_b32 s83, 131072 // (op 2562) call()

v_and_b32 v30, 15, v1 // (op 2579) 
s_mul_i32 s52, s16, s8 // (op 2579) 
s_add_u32 s52, s52, s0 // (op 2579) 
v_and_b32 v33, 63, v1 // (op 2579) 
s_mul_hi_u32 s53, s52, s13 // (op 2579) 
v_lshrrev_b32 v35, 4, v33 // (op 2579) 

s_sub_u32 s54, s52, s53 // (op 2579) 
s_lshr_b32 s55, s54, 1 // (op 2579) 

s_add_u32 s54, s55, s53 // (op 2579) 


s_lshr_b32 s53, s54, s11 // (op 2579) 

s_and_b32 s54, s11, 2147483648 // (op 2579) 
s_cmp_eq_i32 s54, 0 // (op 2579) 
s_mov_b32 s55, scc // (op 2579) 

s_cmp_eq_i32 s55, 0 // (op 2579) 
s_cselect_b32 s54, s52, s53 // (op 2579) 


s_mul_hi_u32 s53, s54, s12 // (op 2579) 
s_sub_u32 s55, s54, s53 // (op 2579) 
s_lshr_b32 s84, s55, 1 // (op 2579) 

s_add_u32 s55, s84, s53 // (op 2579) 


s_lshr_b32 s53, s55, s10 // (op 2579) 

s_and_b32 s55, s10, 2147483648 // (op 2579) 
s_cmp_eq_i32 s55, 0 // (op 2579) 
s_mov_b32 s84, scc // (op 2579) 

s_cmp_eq_i32 s84, 0 // (op 2579) 
s_cselect_b32 s55, s54, s53 // (op 2579) 


s_mul_i32 s53, s55, s14 // (op 2579) 

s_sub_u32 s55, s54, s53 // (op 2579) 

v_lshl_add_u32 v33, s55, 4, v30 // (op 2579) 


v_mul_lo_u32 v30, v33, s44 // (op 2579) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 2579) 
s_mul_i32 s53, s54, s84 // (op 2579) 


s_sub_u32 s54, s52, s53 // (op 2579) 


v_lshl_add_u32 v33, s54, 2, v35 // (op 2579) 


v_add_u32 v35, v33, v30 // (op 2579) 


s_mul_i32 s53, s44, s23 // (op 2581) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 2581) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 2581) least significant
s_mul_hi_u32 s55, s44, s22 // (op 2581) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2581) most significant: sum
s_add_u32 s53, s53, s55 // (op 2581) most significant: sum


s_mov_b32 s80, s46 // (op 2581) reg expression
s_mov_b32 s81, s47 // (op 2581) reg expression
s_mov_b32 s82, s52 // (op 2581) reg expression
s_mov_b32 s83, 131072 // (op 2581) call()




buffer_load_ubyte v30, v34, s[80:83], 0 offen offset: 0 // (op 1865) Load value
v_and_b32 v33, -8, v1 // (op 2746) 
v_sub_u32 v36, v1, v33 // (op 2746) 

v_lshlrev_b32 v33, 4, v36 // (op 2746) 

v_lshrrev_b32 v36, 3, v1 // (op 2746) 
v_lshl_add_u32 v37, v36, 7, v33 // (op 2746) 


s_nop 0
v_readfirstlane_b32 s52, v37 // (op 2746) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3471) 



s_nop 0
s_mov_b32 m0, s52 // (op 2054) reg expression
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 2054) Load value direct to lds

v_and_b32 v33, -4, v1 // (op 2671) 
v_sub_u32 v36, v1, v33 // (op 2671) 

v_lshlrev_b32 v33, 4, v36 // (op 2671) 

v_lshrrev_b32 v36, 2, v1 // (op 2671) 
v_lshl_add_u32 v37, v36, 6, v33 // (op 2671) 


s_nop 0
v_readfirstlane_b32 s52, v37 // (op 2671) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3452) 

s_lshl_b64 s[54:55], s[30:31], 5 // (op 2059) 
v_mov_b64 v[36:37], s[54:55] // (op 2059) convert


v_mov_b64 v[38:39], v[22:23] // (op 2059) reg expression
s_add_u32 s53, 2048, s52 // (op 2059) 
s_mov_b32 m0, s53 // (op 2059) reg expression
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 2059) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 2059) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 2059) most significant half

s_add_u32 m0, m0, 2048 // (op 2059) 
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 2059) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 2059) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 2059) most significant half

s_add_u32 m0, m0, 2048 // (op 2059) 
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 2059) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 2059) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 2059) most significant half

s_add_u32 m0, m0, 2048 // (op 2059) 
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 2059) Load value direct to lds





v_mov_b64 v[36:37], s[38:39] // (op 1875) convert

v_mov_b32 v33, v31 // (op 1875) reg expression
buffer_load_ubyte v38, v33, s[76:79], 0 offen offset: 0 // (op 1875) Load value
v_add_u32 v33, v33, v36 // (op 1875) 
buffer_load_ubyte v39, v33, s[76:79], 0 offen offset: 0 // (op 1875) Load value





buffer_load_ubyte v33, v34, s[80:83], 0 offen offset:4 // (op 1879) Load value

v_and_b32 v34, -8, v1 // (op 3330) 
v_sub_u32 v36, v1, v34 // (op 3330) 

v_lshlrev_b32 v34, 4, v36 // (op 3330) 

v_lshrrev_b32 v36, 3, v1 // (op 3330) 
v_lshl_add_u32 v37, v36, 7, v34 // (op 3330) 


s_nop 0
v_readfirstlane_b32 s52, v37 // (op 3330) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3601) 


s_nop 0
s_add_u32 s53, 10240, s52 // (op 2064) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s53 // (op 2064) reg expression
s_mov_b32 s54, 128 // (op 2064) call()
buffer_load_dwordx4 v26, s[72:75], s54 offen offset: 0 lds // (op 2064) Load value direct to lds




v_and_b32 v34, -4, v1 // (op 3286) 
v_sub_u32 v26, v1, v34 // (op 3286) 

v_lshlrev_b32 v27, 4, v26 // (op 3286) 

v_lshrrev_b32 v26, 2, v1 // (op 3286) 
v_lshl_add_u32 v34, v26, 6, v27 // (op 3286) 


s_nop 0
v_readfirstlane_b32 s52, v34 // (op 3286) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3592) 

s_lshl_b64 s[54:55], s[30:31], 5 // (op 2066) 
v_mov_b64 v[26:27], s[54:55] // (op 2066) convert


s_lshl_b64 s[54:55], s[30:31], 7 // (op 2066) 
v_mov_b32 v34, s54 // (op 2066) 
v_mov_b32 v36, s55 // (op 2066) 
v_add_co_u32 v40, s[84:85], v22, v34 // (op 2066) least significant half
v_addc_co_u32 v41, s[84:85], v23, v36, s[84:85] // (op 2066) most significant half




s_add_u32 s53, 12288, s52 // (op 2066) 
s_mov_b32 m0, s53 // (op 2066) reg expression
buffer_load_dwordx4 v40, s[68:71], 0 offen offset: 0 lds // (op 2066) Load value direct to lds
v_add_co_u32 v40, s[54:55], v40, v26 // (op 2066) least significant half
v_addc_co_u32 v41, s[54:55], 0, v41, s[54:55] // (op 2066) most significant half

s_add_u32 m0, m0, 2048 // (op 2066) 
buffer_load_dwordx4 v40, s[68:71], 0 offen offset: 0 lds // (op 2066) Load value direct to lds
v_add_co_u32 v40, s[54:55], v40, v26 // (op 2066) least significant half
v_addc_co_u32 v41, s[54:55], 0, v41, s[54:55] // (op 2066) most significant half

s_add_u32 m0, m0, 2048 // (op 2066) 
buffer_load_dwordx4 v40, s[68:71], 0 offen offset: 0 lds // (op 2066) Load value direct to lds
v_add_co_u32 v40, s[54:55], v40, v26 // (op 2066) least significant half
v_addc_co_u32 v41, s[54:55], 0, v41, s[54:55] // (op 2066) most significant half

s_add_u32 m0, m0, 2048 // (op 2066) 
buffer_load_dwordx4 v40, s[68:71], 0 offen offset: 0 lds // (op 2066) Load value direct to lds






v_mov_b64 v[22:23], s[38:39] // (op 1889) convert

s_lshl_b32 s52, s38, 2 // (op 1889) 
v_add_u32 v34, v31, s52 // (op 1889) 

buffer_load_ubyte v26, v34, s[76:79], 0 offen offset: 0 // (op 1889) Load value
v_add_u32 v34, v34, v22 // (op 1889) 
buffer_load_ubyte v27, v34, s[76:79], 0 offen offset: 0 // (op 1889) Load value



v_and_b32 v31, -64, v1 // (op 2812) 
v_sub_u32 v34, v1, v31 // (op 2812) 

v_lshrrev_b32 v31, 6, v1 // (op 2812) 
v_lshl_add_u32 v22, v31, 7, v34 // (op 2812) 


s_barrier  // (op 3490) 



s_waitcnt vmcnt(9)
ds_write_b8 v22, v38 offset:20480 // (op 1893) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v22, v39 offset:20544 // (op 1893) Store local data 

s_barrier  // (op 3468) 



ds_read_b128 v[40:43], v20 // (op 1916) Load local data 
ds_read_b128 v[44:47], v20 offset:64 // (op 1916) Load local data 
s_barrier  // (op 3449) 



ds_read_b64_tr_b8 v[48:49], v16 offset:2048 // (op 1907) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v16 offset:2560 // (op 1907) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v16 offset:6144 // (op 1907) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v16 offset:6656 // (op 1907) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 3487) 


ds_read_u8 v31, v18 offset:20480 // (op 1920) Load local data 



ds_read_b64_tr_b8 v[56:57], v16 offset:2080 // (op 1912) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v16 offset:2592 // (op 1912) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v16 offset:6176 // (op 1912) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v16 offset:6688 // (op 1912) Transpose load local data 


ds_read_u8 v34, v18 offset:20512 // (op 1925) Load local data 
v_and_b32 v22, -4, v1 // (op 2643) 
v_sub_u32 v23, v1, v22 // (op 2643) 

v_lshlrev_b32 v22, 4, v23 // (op 2643) 

v_lshrrev_b32 v23, 2, v1 // (op 2643) 
v_lshl_add_u32 v36, v23, 6, v22 // (op 2643) 


s_nop 0
v_readfirstlane_b32 s52, v36 // (op 2643) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -8, v1 // (op 2718) 
v_sub_u32 v23, v1, v22 // (op 2718) 

v_lshlrev_b32 v22, 4, v23 // (op 2718) 

v_lshrrev_b32 v23, 3, v1 // (op 2718) 
v_lshl_add_u32 v36, v23, 7, v22 // (op 2718) 


s_nop 0
v_readfirstlane_b32 s53, v36 // (op 2718) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -64, v1 // (op 2783) 
v_sub_u32 v23, v1, v22 // (op 2783) 

v_lshrrev_b32 v22, 6, v1 // (op 2783) 
v_lshl_add_u32 v36, v22, 7, v23 // (op 2783) 


v_and_b32 v37, -4, v1 // (op 3271) 
v_sub_u32 v22, v1, v37 // (op 3271) 

v_lshlrev_b32 v23, 4, v22 // (op 3271) 

v_lshrrev_b32 v22, 2, v1 // (op 3271) 
v_lshl_add_u32 v37, v22, 6, v23 // (op 3271) 


s_nop 0
v_readfirstlane_b32 s54, v37 // (op 3271) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -8, v1 // (op 3315) 
v_sub_u32 v22, v1, v37 // (op 3315) 

v_lshlrev_b32 v23, 4, v22 // (op 3315) 

v_lshrrev_b32 v22, 3, v1 // (op 3315) 
v_lshl_add_u32 v37, v22, 7, v23 // (op 3315) 


s_nop 0
v_readfirstlane_b32 s55, v37 // (op 3315) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -64, v1 // (op 3354) 
v_sub_u32 v22, v1, v37 // (op 3354) 

v_lshrrev_b32 v23, 6, v1 // (op 3354) 
v_lshl_add_u32 v37, v23, 7, v22 // (op 3354) 


s_mov_b32 s1, 0 // (op 1006) call()
s_and_b32 s84, s7, 4294967294 // (op 1005) 
s_cmp_lt_u32 s1, s84 // (op 1005) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_1005 // (op 1005) Condition: Top (jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_1005 if false)
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoop_1005:
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[40:47], a[0:3], v31, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 84) 
s_add_u32 s1, s1, 2 // (op 1007) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:63], v[40:47], a[4:7], v34, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 1041) 


v_mov_b32 v22, v35 // (op 82) 

buffer_load_ubyte v30, v22, s[80:83], 0 offen offset:8 // (op 82) Load value
s_barrier  // (op 3474) 


v_mov_b64 v[64:65], v[28:29] // (op 1979) 

s_mov_b32 m0, s53 // (op 1979) reg expression
s_mov_b32 s84, 256 // (op 1979) call()
buffer_load_dwordx4 v64, s[72:75], s84 offen offset: 0 lds // (op 1979) Load value direct to lds

s_barrier  // (op 3455) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 1974) 
v_mov_b64 v[66:67], s[84:85] // (op 1974) convert


v_mov_b64 v[68:69], v[24:25] // (op 1974) 
s_lshl_b64 s[84:85], s[30:31], 7 // (op 1974) 
s_lshl_b64 s[86:87], s[84:85], 1 // (op 1974) 

v_mov_b32 v23, s86 // (op 1974) 
v_mov_b32 v70, s87 // (op 1974) 
v_add_co_u32 v72, s[84:85], v68, v23 // (op 1974) least significant half
v_addc_co_u32 v73, s[84:85], v69, v70, s[84:85] // (op 1974) most significant half




s_add_u32 s84, 2048, s52 // (op 1974) 
s_nop 0
s_mov_b32 m0, s84 // (op 1974) reg expression
buffer_load_dwordx4 v72, s[68:71], 0 offen offset: 0 lds // (op 1974) Load value direct to lds
v_add_co_u32 v72, s[86:87], v72, v66 // (op 1974) least significant half
v_addc_co_u32 v73, s[86:87], 0, v73, s[86:87] // (op 1974) most significant half

s_add_u32 m0, m0, 2048 // (op 1974) 
buffer_load_dwordx4 v72, s[68:71], 0 offen offset: 0 lds // (op 1974) Load value direct to lds
v_add_co_u32 v72, s[86:87], v72, v66 // (op 1974) least significant half
v_addc_co_u32 v73, s[86:87], 0, v73, s[86:87] // (op 1974) most significant half

s_add_u32 m0, m0, 2048 // (op 1974) 
buffer_load_dwordx4 v72, s[68:71], 0 offen offset: 0 lds // (op 1974) Load value direct to lds
v_add_co_u32 v72, s[86:87], v72, v66 // (op 1974) least significant half
v_addc_co_u32 v73, s[86:87], 0, v73, s[86:87] // (op 1974) most significant half

s_add_u32 m0, m0, 2048 // (op 1974) 
buffer_load_dwordx4 v72, s[68:71], 0 offen offset: 0 lds // (op 1974) Load value direct to lds




v_mov_b64 v[66:67], s[38:39] // (op 17) convert

v_mov_b32 v23, v32 // (op 17) 
s_lshl_b32 s84, s38, 3 // (op 17) 
v_add_u32 v70, v23, s84 // (op 17) 

buffer_load_ubyte v38, v70, s[76:79], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v70, v70, v66 // (op 17) 
buffer_load_ubyte v39, v70, s[76:79], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 3610) 



s_waitcnt vmcnt(9)
ds_write_b8 v37, v26 offset:20736 // (op 931) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v37, v27 offset:20800 // (op 931) Store local data 
s_barrier  // (op 3598) 



ds_read_b128 v[40:43], v7 offset:10240 // (op 924) Load local data 
ds_read_b128 v[44:47], v7 offset:10304 // (op 924) Load local data 
s_barrier  // (op 3589) 



ds_read_b64_tr_b8 v[48:49], v12 offset:12288 // (op 922) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v12 offset:12800 // (op 922) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v12 offset:16384 // (op 922) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v12 offset:16896 // (op 922) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 3607) 


ds_read_u8 v31, v13 offset:20736 // (op 926) Load local data 



ds_read_b64_tr_b8 v[56:57], v12 offset:12320 // (op 1048) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v12 offset:12832 // (op 1048) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v12 offset:16416 // (op 1048) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v12 offset:16928 // (op 1048) Transpose load local data 


ds_read_u8 v34, v13 offset:20768 // (op 1052) Load local data 
s_barrier  // (op 3765) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[40:47], a[0:3], v31, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 929) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:63], v[40:47], a[4:7], v34, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 1042) 



buffer_load_ubyte v33, v22, s[80:83], 0 offen offset:12 // (op 928) Load value

s_barrier  // (op 3604) 
v_add_u32 v35, 8, v35 // (op 2571) 


s_add_u32 s84, 10240, s55 // (op 2035) 

s_mov_b32 m0, s84 // (op 2035) reg expression
s_mov_b32 s85, 384 // (op 2035) call()
buffer_load_dwordx4 v64, s[72:75], s85 offen offset: 0 lds // (op 2035) Load value direct to lds



s_barrier  // (op 3595) 
v_add_co_u32 v28, vcc, 256, v28 // (op 2315) least significant half

v_addc_co_u32 v29, vcc, 0, v29, vcc // (op 2315) most significant half
s_lshl_b64 s[84:85], s[30:31], 5 // (op 2030) 
v_mov_b64 v[64:65], s[84:85] // (op 2030) convert


s_lshl_b64 s[84:85], s[30:31], 7 // (op 2030) 
s_mul_i32 s87, 3, s85 // (op 2030) most significant: low of low * high
s_mul_i32 s86, 3, s84 // (op 2030) least significant
s_mul_hi_u32 s88, 3, s84 // (op 2030) most significant: high of low * low
s_add_u32 s87, s87, s88 // (op 2030) most significant: sum


v_mov_b32 v22, s86 // (op 2030) 
v_mov_b32 v66, s87 // (op 2030) 
v_add_co_u32 v70, s[84:85], v68, v22 // (op 2030) least significant half
v_addc_co_u32 v71, s[84:85], v69, v66, s[84:85] // (op 2030) most significant half




s_add_u32 s84, 12288, s54 // (op 2030) 
s_nop 0
s_mov_b32 m0, s84 // (op 2030) reg expression
buffer_load_dwordx4 v70, s[68:71], 0 offen offset: 0 lds // (op 2030) Load value direct to lds
v_add_co_u32 v70, s[86:87], v70, v64 // (op 2030) least significant half
v_addc_co_u32 v71, s[86:87], 0, v71, s[86:87] // (op 2030) most significant half

s_add_u32 m0, m0, 2048 // (op 2030) 
buffer_load_dwordx4 v70, s[68:71], 0 offen offset: 0 lds // (op 2030) Load value direct to lds
v_add_co_u32 v70, s[86:87], v70, v64 // (op 2030) least significant half
v_addc_co_u32 v71, s[86:87], 0, v71, s[86:87] // (op 2030) most significant half

s_add_u32 m0, m0, 2048 // (op 2030) 
buffer_load_dwordx4 v70, s[68:71], 0 offen offset: 0 lds // (op 2030) Load value direct to lds
v_add_co_u32 v70, s[86:87], v70, v64 // (op 2030) least significant half
v_addc_co_u32 v71, s[86:87], 0, v71, s[86:87] // (op 2030) most significant half

s_add_u32 m0, m0, 2048 // (op 2030) 
buffer_load_dwordx4 v70, s[68:71], 0 offen offset: 0 lds // (op 2030) Load value direct to lds





v_mov_b64 v[64:65], s[38:39] // (op 933) convert

s_lshl_b32 s84, s38, 2 // (op 933) 
s_mul_i32 s85, 3, s84 // (op 933) 

v_add_u32 v22, v23, s85 // (op 933) 

s_lshl_b64 s[84:85], s[30:31], 7 // (op 2187) 
s_lshl_b64 s[86:87], s[84:85], 1 // (op 2187) 

v_mov_b32 v66, s86 // (op 2187) 
v_mov_b32 v67, s87 // (op 2187) 
v_add_co_u32 v24, s[84:85], v24, v66 // (op 2187) least significant half
v_addc_co_u32 v25, s[84:85], v25, v67, s[84:85] // (op 2187) most significant half




buffer_load_ubyte v26, v22, s[76:79], 0 offen offset: 0 // (op 933) Load value
v_add_u32 v22, v22, v64 // (op 933) 
buffer_load_ubyte v27, v22, s[76:79], 0 offen offset: 0 // (op 933) Load value



s_lshl_b32 s84, s38, 3 // (op 2443) 
s_barrier  // (op 3493) 
v_add_u32 v32, v32, s84 // (op 2443) 




s_waitcnt vmcnt(9)
ds_write_b8 v36, v38 offset:20480 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v36, v39 offset:20544 // (op 21) Store local data 
s_barrier  // (op 3480) 



ds_read_b128 v[40:43], v4 // (op 72) Load local data 
ds_read_b128 v[44:47], v4 offset:64 // (op 72) Load local data 
s_barrier  // (op 3461) 



ds_read_b64_tr_b8 v[48:49], v0 offset:2048 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:2560 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:6144 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:6656 // (op 67) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 3500) 


ds_read_u8 v31, v3 offset:20480 // (op 77) Load local data 



ds_read_b64_tr_b8 v[56:57], v0 offset:2080 // (op 1035) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:2592 // (op 1035) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:6176 // (op 1035) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:6688 // (op 1035) Transpose load local data 


ds_read_u8 v34, v3 offset:20512 // (op 1039) Load local data 
s_and_b32 s84, s7, 4294967294 // (op 1005) 
s_cmp_lt_u32 s1, s84 // (op 1005) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoop_1005 // (op 1005) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoop_1005 if true)
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_1005:












s_barrier  // (op 3749) 
s_and_b32 s52, s7, 4294967294 // (op 2103) 
v_and_b32 v32, -4, v1 // (op 2103) 
s_add_u32 s53, s0, s52 // (op 2103) 

v_sub_u32 v22, v1, v32 // (op 2103) 

v_lshlrev_b32 v32, 4, v22 // (op 2103) 

s_mul_i32 s52, s16, s8 // (op 2103) 
s_add_u32 s52, s52, s53 // (op 2103) 

v_lshrrev_b32 v22, 2, v1 // (op 2103) 
s_mul_hi_u32 s53, s52, s13 // (op 2103) 
s_sub_u32 s54, s52, s53 // (op 2103) 
s_lshr_b32 s55, s54, 1 // (op 2103) 

s_add_u32 s54, s55, s53 // (op 2103) 


s_lshr_b32 s53, s54, s11 // (op 2103) 

s_and_b32 s54, s11, 2147483648 // (op 2103) 
s_cmp_eq_i32 s54, 0 // (op 2103) 
s_mov_b32 s55, scc // (op 2103) 

s_cmp_eq_i32 s55, 0 // (op 2103) 
s_cselect_b32 s54, s52, s53 // (op 2103) 


s_mul_hi_u32 s53, s54, s12 // (op 2103) 
s_sub_u32 s55, s54, s53 // (op 2103) 
s_lshr_b32 s84, s55, 1 // (op 2103) 

s_add_u32 s55, s84, s53 // (op 2103) 


s_lshr_b32 s53, s55, s10 // (op 2103) 

s_and_b32 s55, s10, 2147483648 // (op 2103) 
s_cmp_eq_i32 s55, 0 // (op 2103) 
s_mov_b32 s84, scc // (op 2103) 

s_cmp_eq_i32 s84, 0 // (op 2103) 
s_cselect_b32 s55, s54, s53 // (op 2103) 


v_lshl_add_u32 v23, s55, 6, v32 // (op 2103) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 2103) 
v_mov_b32 v25, 0 // (op 2103) convert
v_mov_b32 v24, v23 // (op 2103) convert

s_mul_i32 s53, s54, s84 // (op 2103) 


s_sub_u32 s54, s52, s53 // (op 2103) 


v_lshl_add_u32 v23, s54, 7, v22 // (op 2103) 


v_mul_lo_u32 v29, v23, s31 // (op 2103) most significant: low of low * high
v_mul_hi_u32 v22, v23, s30 // (op 2103) most significant: high of low * low
v_mul_lo_u32 v28, v23, s30 // (op 2103) least significant: low of low * low
v_add_u32 v29, v29, v22 // (op 2103) most significant: sum


v_add_co_u32 v22, s[52:53], v24, v28 // (op 2103) least significant half
v_addc_co_u32 v23, s[52:53], v25, v29, s[52:53] // (op 2103) most significant half



s_mul_i32 s53, s30, s3 // (op 2105) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 2105) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 2105) least significant
s_mul_hi_u32 s55, s30, s2 // (op 2105) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2105) most significant: sum
s_add_u32 s53, s53, s55 // (op 2105) most significant: sum


s_mov_b32 s68, s32 // (op 2105) reg expression
s_mov_b32 s69, s33 // (op 2105) reg expression
s_mov_b32 s70, s52 // (op 2105) reg expression
s_mov_b32 s71, 131072 // (op 2105) call()

s_and_b32 s52, s7, 4294967294 // (op 2239) 
v_lshrrev_b32 v32, 3, v1 // (op 2239) 
s_add_u32 s53, s0, s52 // (op 2239) 

v_and_b32 v24, -8, v1 // (op 2239) 
v_sub_u32 v25, v1, v24 // (op 2239) 

s_mul_i32 s52, s16, s8 // (op 2239) 
s_add_u32 s52, s52, s53 // (op 2239) 

v_lshlrev_b32 v24, 4, v25 // (op 2239) 

s_mul_hi_u32 s53, s52, s13 // (op 2239) 
s_sub_u32 s54, s52, s53 // (op 2239) 
s_lshr_b32 s55, s54, 1 // (op 2239) 

s_add_u32 s54, s55, s53 // (op 2239) 


s_lshr_b32 s53, s54, s11 // (op 2239) 

s_and_b32 s54, s11, 2147483648 // (op 2239) 
s_cmp_eq_i32 s54, 0 // (op 2239) 
s_mov_b32 s55, scc // (op 2239) 

s_cmp_eq_i32 s55, 0 // (op 2239) 
s_cselect_b32 s54, s52, s53 // (op 2239) 


s_mul_hi_u32 s53, s54, s12 // (op 2239) 
s_sub_u32 s55, s54, s53 // (op 2239) 
s_lshr_b32 s84, s55, 1 // (op 2239) 

s_add_u32 s55, s84, s53 // (op 2239) 


s_lshr_b32 s53, s55, s10 // (op 2239) 

s_and_b32 s55, s10, 2147483648 // (op 2239) 
s_cmp_eq_i32 s55, 0 // (op 2239) 
s_mov_b32 s84, scc // (op 2239) 

s_cmp_eq_i32 s84, 0 // (op 2239) 
s_cselect_b32 s55, s54, s53 // (op 2239) 


s_mul_i32 s53, s55, s14 // (op 2239) 

s_sub_u32 s55, s54, s53 // (op 2239) 

v_lshl_add_u32 v25, s55, 4, v32 // (op 2239) 


v_mul_lo_u32 v29, v25, s35 // (op 2239) most significant: low of low * high
v_mul_hi_u32 v30, v25, s34 // (op 2239) most significant: high of low * low
v_mul_lo_u32 v28, v25, s34 // (op 2239) least significant: low of low * low
v_add_u32 v29, v29, v30 // (op 2239) most significant: sum


s_ashr_i64 s[84:85], s[2:3], 7 // (op 2239) 
s_mul_i32 s53, s54, s84 // (op 2239) 


s_sub_u32 s54, s52, s53 // (op 2239) 


v_lshl_add_u32 v25, s54, 7, v24 // (op 2239) 


v_mov_b32 v39, 0 // (op 2239) convert
v_mov_b32 v38, v25 // (op 2239) convert

v_add_co_u32 v24, s[52:53], v38, v28 // (op 2239) least significant half
v_addc_co_u32 v25, s[52:53], v39, v29, s[52:53] // (op 2239) most significant half



s_mul_i32 s53, s34, s23 // (op 2241) most significant: low of low * high
s_mul_i32 s54, s35, s22 // (op 2241) most significant: low of high * low
s_mul_i32 s52, s34, s22 // (op 2241) least significant
s_mul_hi_u32 s55, s34, s22 // (op 2241) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2241) most significant: sum
s_add_u32 s53, s53, s55 // (op 2241) most significant: sum


s_mov_b32 s72, s36 // (op 2241) reg expression
s_mov_b32 s73, s37 // (op 2241) reg expression
s_mov_b32 s74, s52 // (op 2241) reg expression
s_mov_b32 s75, 131072 // (op 2241) call()

s_and_b32 s52, s7, 4294967294 // (op 2367) 
v_and_b32 v32, -64, v1 // (op 2367) 
s_add_u32 s53, s0, s52 // (op 2367) 

v_sub_u32 v28, v1, v32 // (op 2367) 

v_lshrrev_b32 v32, 6, v1 // (op 2367) 
s_mul_i32 s52, s16, s8 // (op 2367) 
s_add_u32 s52, s52, s53 // (op 2367) 

v_lshlrev_b32 v29, 1, v32 // (op 2367) 

s_mul_hi_u32 s53, s52, s13 // (op 2367) 
s_sub_u32 s54, s52, s53 // (op 2367) 
s_lshr_b32 s55, s54, 1 // (op 2367) 

s_add_u32 s54, s55, s53 // (op 2367) 


s_lshr_b32 s53, s54, s11 // (op 2367) 

s_and_b32 s54, s11, 2147483648 // (op 2367) 
s_cmp_eq_i32 s54, 0 // (op 2367) 
s_mov_b32 s55, scc // (op 2367) 

s_cmp_eq_i32 s55, 0 // (op 2367) 
s_cselect_b32 s54, s52, s53 // (op 2367) 


s_mul_hi_u32 s53, s54, s12 // (op 2367) 
s_sub_u32 s55, s54, s53 // (op 2367) 
s_lshr_b32 s84, s55, 1 // (op 2367) 

s_add_u32 s55, s84, s53 // (op 2367) 


s_lshr_b32 s53, s55, s10 // (op 2367) 

s_and_b32 s55, s10, 2147483648 // (op 2367) 
s_cmp_eq_i32 s55, 0 // (op 2367) 
s_mov_b32 s84, scc // (op 2367) 

s_cmp_eq_i32 s84, 0 // (op 2367) 
s_cselect_b32 s55, s54, s53 // (op 2367) 


v_lshl_add_u32 v30, s55, 6, v28 // (op 2367) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 2367) 
s_mul_i32 s53, s54, s84 // (op 2367) 


s_sub_u32 s54, s52, s53 // (op 2367) 


v_lshl_add_u32 v28, s54, 2, v29 // (op 2367) 


v_mul_lo_u32 v29, v28, s38 // (op 2367) 

v_add_u32 v28, v30, v29 // (op 2367) 


s_mul_i32 s53, s38, s43 // (op 2369) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 2369) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 2369) least significant
s_mul_hi_u32 s55, s38, s42 // (op 2369) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2369) most significant: sum
s_add_u32 s53, s53, s55 // (op 2369) most significant: sum


s_mov_b32 s76, s40 // (op 2369) reg expression
s_mov_b32 s77, s41 // (op 2369) reg expression
s_mov_b32 s78, s52 // (op 2369) reg expression
s_mov_b32 s79, 131072 // (op 2369) call()

s_and_b32 s52, s7, 4294967294 // (op 2495) 
v_and_b32 v32, 15, v1 // (op 2495) 
s_add_u32 s53, s0, s52 // (op 2495) 

v_and_b32 v29, 63, v1 // (op 2495) 
v_lshrrev_b32 v30, 4, v29 // (op 2495) 

s_mul_i32 s52, s16, s8 // (op 2495) 
s_add_u32 s52, s52, s53 // (op 2495) 

s_mul_hi_u32 s53, s52, s13 // (op 2495) 
s_sub_u32 s54, s52, s53 // (op 2495) 
s_lshr_b32 s55, s54, 1 // (op 2495) 

s_add_u32 s54, s55, s53 // (op 2495) 


s_lshr_b32 s53, s54, s11 // (op 2495) 

s_and_b32 s54, s11, 2147483648 // (op 2495) 
s_cmp_eq_i32 s54, 0 // (op 2495) 
s_mov_b32 s55, scc // (op 2495) 

s_cmp_eq_i32 s55, 0 // (op 2495) 
s_cselect_b32 s54, s52, s53 // (op 2495) 


s_mul_hi_u32 s53, s54, s12 // (op 2495) 
s_sub_u32 s55, s54, s53 // (op 2495) 
s_lshr_b32 s84, s55, 1 // (op 2495) 

s_add_u32 s55, s84, s53 // (op 2495) 


s_lshr_b32 s53, s55, s10 // (op 2495) 

s_and_b32 s55, s10, 2147483648 // (op 2495) 
s_cmp_eq_i32 s55, 0 // (op 2495) 
s_mov_b32 s84, scc // (op 2495) 

s_cmp_eq_i32 s84, 0 // (op 2495) 
s_cselect_b32 s55, s54, s53 // (op 2495) 


s_mul_i32 s53, s55, s14 // (op 2495) 

s_sub_u32 s55, s54, s53 // (op 2495) 

v_lshl_add_u32 v29, s55, 4, v32 // (op 2495) 


v_mul_lo_u32 v32, v29, s44 // (op 2495) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 2495) 
s_mul_i32 s53, s54, s84 // (op 2495) 


s_sub_u32 s54, s52, s53 // (op 2495) 


v_lshl_add_u32 v29, s54, 2, v30 // (op 2495) 


v_add_u32 v30, v29, v32 // (op 2495) 


s_mul_i32 s53, s44, s23 // (op 2497) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 2497) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 2497) least significant
s_mul_hi_u32 s55, s44, s22 // (op 2497) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2497) most significant: sum
s_add_u32 s53, s53, s55 // (op 2497) most significant: sum


s_mov_b32 s80, s46 // (op 2497) reg expression
s_mov_b32 s81, s47 // (op 2497) reg expression
s_mov_b32 s82, s52 // (op 2497) reg expression
s_mov_b32 s83, 131072 // (op 2497) call()

v_and_b32 v29, -4, v1 // (op 2656) 
v_sub_u32 v32, v1, v29 // (op 2656) 

v_lshlrev_b32 v29, 4, v32 // (op 2656) 

v_lshrrev_b32 v32, 2, v1 // (op 2656) 
v_lshl_add_u32 v35, v32, 6, v29 // (op 2656) 


s_nop 0
v_readfirstlane_b32 s52, v35 // (op 2656) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -8, v1 // (op 2731) 
v_sub_u32 v32, v1, v29 // (op 2731) 

v_lshlrev_b32 v29, 4, v32 // (op 2731) 

v_lshrrev_b32 v32, 3, v1 // (op 2731) 
v_lshl_add_u32 v35, v32, 7, v29 // (op 2731) 


s_nop 0
v_readfirstlane_b32 s53, v35 // (op 2731) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -64, v1 // (op 2796) 
v_sub_u32 v32, v1, v29 // (op 2796) 

v_lshrrev_b32 v29, 6, v1 // (op 2796) 
v_lshl_add_u32 v35, v29, 7, v32 // (op 2796) 


s_and_b32 s54, s7, 4294967294 // (op 1011) 
s_cmp_lt_u32 s54, s7 // (op 1010) 







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_1010 // (op 1010) Condition: Top (jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_1010 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_1010:
s_barrier  // (op 3458) 
s_add_u32 s54, s54, 1 // (op 1012) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 2017) 
v_mov_b64 v[26:27], s[84:85] // (op 2017) convert


v_mov_b64 v[32:33], v[22:23] // (op 2017) 
v_mov_b64 v[36:37], v[32:33] // (op 2017) reg expression
s_add_u32 s55, 2048, s52 // (op 2017) 
s_mov_b32 m0, s55 // (op 2017) reg expression
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 2017) Load value direct to lds
v_add_co_u32 v36, s[84:85], v36, v26 // (op 2017) least significant half
v_addc_co_u32 v37, s[84:85], 0, v37, s[84:85] // (op 2017) most significant half

s_add_u32 m0, m0, 2048 // (op 2017) 
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 2017) Load value direct to lds
v_add_co_u32 v36, s[84:85], v36, v26 // (op 2017) least significant half
v_addc_co_u32 v37, s[84:85], 0, v37, s[84:85] // (op 2017) most significant half

s_add_u32 m0, m0, 2048 // (op 2017) 
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 2017) Load value direct to lds
v_add_co_u32 v36, s[84:85], v36, v26 // (op 2017) least significant half
v_addc_co_u32 v37, s[84:85], 0, v37, s[84:85] // (op 2017) most significant half

s_add_u32 m0, m0, 2048 // (op 2017) 
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 2017) Load value direct to lds




s_lshl_b64 s[84:85], s[30:31], 7 // (op 2093) 
v_mov_b32 v29, s84 // (op 2093) 
v_mov_b32 v26, s85 // (op 2093) 
v_add_co_u32 v22, s[86:87], v22, v29 // (op 2093) least significant half
v_addc_co_u32 v23, s[86:87], v23, v26, s[86:87] // (op 2093) most significant half




s_barrier  // (op 3477) 


v_mov_b64 v[26:27], v[24:25] // (op 2023) 

s_mov_b32 m0, s53 // (op 2023) reg expression
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 2023) Load value direct to lds

v_add_co_u32 v24, vcc, 128, v24 // (op 2231) least significant half
v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 2231) most significant half

v_mov_b64 v[26:27], s[38:39] // (op 875) convert

v_mov_b32 v29, v28 // (op 875) 
v_mov_b32 v31, v29 // (op 875) reg expression
buffer_load_ubyte v32, v31, s[76:79], 0 offen offset: 0 // (op 875) Load value
v_add_u32 v31, v31, v26 // (op 875) 
buffer_load_ubyte v33, v31, s[76:79], 0 offen offset: 0 // (op 875) Load value



s_lshl_b32 s55, s38, 2 // (op 2359) 
v_add_u32 v28, v28, s55 // (op 2359) 

s_barrier  // (op 3496) 
s_waitcnt vmcnt(3)
s_barrier  // (op 3464) 






ds_read_b64_tr_b8 v[36:37], v0 offset:2048 // (op 864) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:2560 // (op 864) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:6144 // (op 864) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:6656 // (op 864) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v35, v32 offset:20480 // (op 873) Store local data 
s_barrier  // (op 3483) 
s_waitcnt vmcnt(0)
ds_write_b8 v35, v33 offset:20544 // (op 873) Store local data 




ds_read_b128 v[44:47], v4 // (op 866) Load local data 
ds_read_b128 v[48:51], v4 offset:64 // (op 866) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 3503) 


ds_read_u8 v29, v3 offset:20480 // (op 868) Load local data 


v_mov_b32 v26, v30 // (op 870) 

buffer_load_ubyte v27, v26, s[80:83], 0 offen offset: 0 // (op 870) Load value

v_add_u32 v30, 4, v30 // (op 2487) 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:51], a[0:3], v29, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 871) 
ds_read_b64_tr_b8 v[52:53], v0 offset:2080 // (op 1016) Transpose load local data 


ds_read_b64_tr_b8 v[54:55], v0 offset:2592 // (op 1016) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:6176 // (op 1016) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:6688 // (op 1016) Transpose load local data 


ds_read_u8 v26, v3 offset:20512 // (op 1020) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:59], v[44:51], a[4:7], v26, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 1022) 




s_cmp_lt_u32 s54, s7 // (op 1010) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_1010 // (op 1010) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_1010 if true)
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_1010:








s_barrier  // (op 3753) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 280) 
s_cmp_lg_u32 s18, 0 // (op 306) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Send_Tile_306 // (op 306) Condition: False, jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Send_Tile_306



s_nop 1
buffer_store_dword a4, v19, s[64:67], 0 offen offset:2048 sc1 // (op 1207) Store value
buffer_store_dword a5, v19, s[64:67], 0 offen offset:2112 sc1 // (op 1207) Store value
buffer_store_dword a6, v19, s[64:67], 0 offen offset:2176 sc1 // (op 1207) Store value
buffer_store_dword a7, v19, s[64:67], 0 offen offset:2240 sc1 // (op 1207) Store value



buffer_store_dword a0, v19, s[64:67], 0 offen sc1 // (op 303) Store value
buffer_store_dword a1, v19, s[64:67], 0 offen offset:64 sc1 // (op 303) Store value
buffer_store_dword a2, v19, s[64:67], 0 offen offset:128 sc1 // (op 303) Store value
buffer_store_dword a3, v19, s[64:67], 0 offen offset:192 sc1 // (op 303) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 307) 
v_cmp_eq_i32 vcc, v1, 0 // (op 311) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_311 // (op 311) Condition: False, jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_311
s_mov_b32 s52, 1 // (op 309) call()
s_mov_b32 s55, 0 // (op 310) convert
s_mov_b32 s54, s16 // (op 310) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 310) 
s_add_u32 s84, s48, s54 // (op 310) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 310) End of Int64 add, unlocking SCC
s_store_dword s52, s[84:85], 0 glc // (op 310) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_311 // (op 311) Condition: Done, jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_311
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_311:
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_311:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Send_Tile_306 // (op 306) Condition: Done, jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Send_Tile_306
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Send_Tile_306:
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Send_Tile_306:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 322) 
s_mov_b32 s52, scc // (op 322) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 322) 
s_mov_b32 s53, s54 // (op 322) convert

s_sub_u32 s54, s53, 1 // (op 322) 

s_cmp_lt_u32 s19, s54 // (op 322) 
s_mov_b32 s53, scc // (op 322) 

s_and_b32 s54, s52, s53 // (op 322) 
s_cmp_lg_u32 s54, 0 // (op 322) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Receive_Tile_322 // (op 322) Condition: False, jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Receive_Tile_322
s_mov_b32 s52, 0 // (op 288) call()
s_mul_i32 s53, s52, s8 // (op 287) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 287) 
s_mov_b32 s84, s54 // (op 287) convert

s_sub_u32 s54, s84, s19 // (op 287) 

s_sub_u32 s55, s54, 1 // (op 287) 

s_cmp_lt_u32 s53, s55 // (op 287) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_287 // (op 287) Condition: Top (jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_287 if false)
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_ReceiveTileLoop_287:
s_add_u32 s53, s16, 1 // (op 3144) 
v_lshrrev_b32 v16, 4, v1 // (op 3144) 
s_add_u32 s54, s53, s52 // (op 3144) 
s_lshl_b32 s54, s54, 6 // (op 3144) 

v_lshlrev_b32 v18, 2, v16 // (op 3144) 

v_add_lshl_u32 v16, s54, v18, 4 // (op 3144) 


v_and_b32 v18, -16, v1 // (op 3144) 
v_sub_u32 v19, v1, v18 // (op 3144) 

v_add_lshl_u32 v18, v16, v19, 2 // (op 3144) 


s_lshl_b32 s53, s15, 12 // (op 3146) 
s_mov_b32 s64, s28 // (op 3146) reg expression
s_mov_b32 s65, s29 // (op 3146) reg expression
s_mov_b32 s66, s53 // (op 3146) reg expression
s_mov_b32 s67, 131072 // (op 3146) call()

RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_326:
s_add_u32 s53, s16, 1 // (op 324) 
s_add_u32 s54, s53, s52 // (op 324) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 324) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 324) 

s_add_u32 s84, s48, s54 // (op 324) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 324) End of Int64 add, unlocking SCC
s_load_dword s53, s[84:85], 0 glc // (op 324) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 326) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_326 // (op 326) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_326 if true)
s_barrier  // (op 330) 
v_cmp_eq_i32 vcc, v1, 0 // (op 329) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_329 // (op 329) Condition: False, jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_329
s_mov_b32 s53, 0 // (op 327) call()
s_add_u32 s54, s16, 1 // (op 328) 
s_add_u32 s84, s54, s52 // (op 328) Start of Int64 add, locking SCC
s_addc_u32 s85, 0, 0 // (op 328) End of Int64 add, unlocking SCC
s_lshl_b64 s[84:85], s[84:85], 2 // (op 328) 

s_add_u32 s54, s48, s84 // (op 328) Start of Int64 add, locking SCC
s_addc_u32 s55, s49, s85 // (op 328) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 328) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_329 // (op 329) Condition: Done, jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_329
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_329:
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_329:




buffer_load_dword v22, v18, s[64:67], 0 offen // (op 341) Load value
buffer_load_dword v23, v18, s[64:67], 0 offen offset:64 // (op 341) Load value
buffer_load_dword v24, v18, s[64:67], 0 offen offset:128 // (op 341) Load value
buffer_load_dword v25, v18, s[64:67], 0 offen offset:192 // (op 341) Load value
v_accvgpr_read v26, a0 // (op 344) reg expression
v_accvgpr_read v27, a1 // (op 344) reg expression
v_accvgpr_read v28, a2 // (op 344) reg expression
v_accvgpr_read v29, a3 // (op 344) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v26, v26, v22 // (op 346) 
s_waitcnt vmcnt(2)
v_add_f32 v27, v27, v23 // (op 346) 
s_waitcnt vmcnt(1)
v_add_f32 v28, v28, v24 // (op 346) 
s_waitcnt vmcnt(0)
v_add_f32 v29, v29, v25 // (op 346) 

v_accvgpr_write a0, v26 // (op 348) reg expression
v_accvgpr_write a1, v27 // (op 348) reg expression
v_accvgpr_write a2, v28 // (op 348) reg expression
v_accvgpr_write a3, v29 // (op 348) reg expression




buffer_load_dword v22, v18, s[64:67], 0 offen offset:2048 // (op 1223) Load value
buffer_load_dword v23, v18, s[64:67], 0 offen offset:2112 // (op 1223) Load value
buffer_load_dword v24, v18, s[64:67], 0 offen offset:2176 // (op 1223) Load value
buffer_load_dword v25, v18, s[64:67], 0 offen offset:2240 // (op 1223) Load value
v_accvgpr_read v26, a4 // (op 1224) reg expression
v_accvgpr_read v27, a5 // (op 1224) reg expression
v_accvgpr_read v28, a6 // (op 1224) reg expression
v_accvgpr_read v29, a7 // (op 1224) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v26, v26, v22 // (op 1225) 
s_waitcnt vmcnt(2)
v_add_f32 v27, v27, v23 // (op 1225) 
s_waitcnt vmcnt(1)
v_add_f32 v28, v28, v24 // (op 1225) 
s_waitcnt vmcnt(0)
v_add_f32 v29, v29, v25 // (op 1225) 

v_accvgpr_write a4, v26 // (op 1226) reg expression
v_accvgpr_write a5, v27 // (op 1226) reg expression
v_accvgpr_write a6, v28 // (op 1226) reg expression
v_accvgpr_write a7, v29 // (op 1226) reg expression

s_add_u32 s52, s52, 1 // (op 289) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 287) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 287) 
s_mov_b32 s84, s54 // (op 287) convert

s_sub_u32 s54, s84, s19 // (op 287) 

s_sub_u32 s55, s54, 1 // (op 287) 

s_cmp_lt_u32 s53, s55 // (op 287) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_ReceiveTileLoop_287 // (op 287) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_ReceiveTileLoop_287 if true)
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_287:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Receive_Tile_322 // (op 322) Condition: Done, jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Receive_Tile_322
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Receive_Tile_322:
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Receive_Tile_322:

s_cmp_eq_i32 s18, 0 // (op 365) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_365 // (op 365) Condition: False, jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_365
v_cmp_eq_f32 vcc, 0.00000, v14 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_125
v_accvgpr_read v22, a0 // (op 46) 
v_accvgpr_read v23, a1 // (op 46) 
v_accvgpr_read v24, a2 // (op 46) 
v_accvgpr_read v25, a3 // (op 46) 
v_mul_f32 v26, v15, v22 // (op 46) 
v_mul_f32 v27, v15, v23 // (op 46) 
v_mul_f32 v28, v15, v24 // (op 46) 
v_mul_f32 v29, v15, v25 // (op 46) 

v_cvt_f16_f32 v22, v26 // (op 52) 
v_cvt_f16_f32 v23, v27 // (op 52) 
v_cvt_f16_f32 v24, v28 // (op 52) 
v_cvt_f16_f32 v25, v29 // (op 52) 


v_pack_b32_f16 v18, v22, v23 // (op 54) 
v_pack_b32_f16 v19, v24, v25 // (op 54) 


buffer_store_dwordx2 v[18:19], v21, s[60:63], 0 offen // (op 54) Store value


v_accvgpr_read v22, a4 // (op 1248) 
v_accvgpr_read v23, a5 // (op 1248) 
v_accvgpr_read v24, a6 // (op 1248) 
v_accvgpr_read v25, a7 // (op 1248) 
v_mul_f32 v26, v15, v22 // (op 1248) 
v_mul_f32 v27, v15, v23 // (op 1248) 
v_mul_f32 v28, v15, v24 // (op 1248) 
v_mul_f32 v29, v15, v25 // (op 1248) 

v_cvt_f16_f32 v22, v26 // (op 1250) 
v_cvt_f16_f32 v23, v27 // (op 1250) 
v_cvt_f16_f32 v24, v28 // (op 1250) 
v_cvt_f16_f32 v25, v29 // (op 1250) 


v_pack_b32_f16 v18, v22, v23 // (op 1251) 
v_pack_b32_f16 v19, v24, v25 // (op 1251) 


buffer_store_dwordx2 v[18:19], v21, s[60:63], 0 offen offset:64 // (op 1251) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[18:19], v17, s[56:59], 0 offen offset:64 // (op 1281) Load value
v_accvgpr_read v22, a4 // (op 1277) 
v_accvgpr_read v23, a5 // (op 1277) 
v_accvgpr_read v24, a6 // (op 1277) 
v_accvgpr_read v25, a7 // (op 1277) 
v_mul_f32 v26, v15, v22 // (op 1277) 
v_mul_f32 v27, v15, v23 // (op 1277) 
v_mul_f32 v28, v15, v24 // (op 1277) 
v_mul_f32 v29, v15, v25 // (op 1277) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v22, v18 // (op 1282) 
v_lshrrev_b32 v23, 16, v18 // (op 1282) 
v_cvt_f32_f16 v23, v23 // (op 1282) 
v_mul_f32 v30, v14, v22 // (op 1282) 
v_mul_f32 v31, v14, v23 // (op 1282) 
v_cvt_f32_f16 v22, v19 // (op 1282) 
v_lshrrev_b32 v23, 16, v19 // (op 1282) 
v_cvt_f32_f16 v23, v23 // (op 1282) 
v_mul_f32 v32, v14, v22 // (op 1282) 
v_mul_f32 v33, v14, v23 // (op 1282) 


v_add_f32 v22, v30, v26 // (op 1278) 
v_add_f32 v23, v31, v27 // (op 1278) 
v_add_f32 v24, v32, v28 // (op 1278) 
v_add_f32 v25, v33, v29 // (op 1278) 


v_cvt_f16_f32 v26, v22 // (op 1279) 
v_cvt_f16_f32 v27, v23 // (op 1279) 
v_cvt_f16_f32 v28, v24 // (op 1279) 
v_cvt_f16_f32 v29, v25 // (op 1279) 


v_pack_b32_f16 v18, v26, v27 // (op 1280) 
v_pack_b32_f16 v19, v28, v29 // (op 1280) 


buffer_store_dwordx2 v[18:19], v21, s[60:63], 0 offen offset:64 // (op 1280) Store value





buffer_load_dwordx2 v[18:19], v17, s[56:59], 0 offen // (op 138) Load value
v_accvgpr_read v22, a0 // (op 134) 
v_accvgpr_read v23, a1 // (op 134) 
v_accvgpr_read v24, a2 // (op 134) 
v_accvgpr_read v25, a3 // (op 134) 
v_mul_f32 v26, v15, v22 // (op 134) 
v_mul_f32 v27, v15, v23 // (op 134) 
v_mul_f32 v28, v15, v24 // (op 134) 
v_mul_f32 v29, v15, v25 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v22, v18 // (op 139) 
v_lshrrev_b32 v23, 16, v18 // (op 139) 
v_cvt_f32_f16 v23, v23 // (op 139) 
v_mul_f32 v30, v14, v22 // (op 139) 
v_mul_f32 v31, v14, v23 // (op 139) 
v_cvt_f32_f16 v22, v19 // (op 139) 
v_lshrrev_b32 v23, 16, v19 // (op 139) 
v_cvt_f32_f16 v23, v23 // (op 139) 
v_mul_f32 v32, v14, v22 // (op 139) 
v_mul_f32 v33, v14, v23 // (op 139) 


v_add_f32 v22, v30, v26 // (op 135) 
v_add_f32 v23, v31, v27 // (op 135) 
v_add_f32 v24, v32, v28 // (op 135) 
v_add_f32 v25, v33, v29 // (op 135) 


v_cvt_f16_f32 v26, v22 // (op 136) 
v_cvt_f16_f32 v27, v23 // (op 136) 
v_cvt_f16_f32 v28, v24 // (op 136) 
v_cvt_f16_f32 v29, v25 // (op 136) 


v_pack_b32_f16 v18, v26, v27 // (op 137) 
v_pack_b32_f16 v19, v28, v29 // (op 137) 


buffer_store_dwordx2 v[18:19], v21, s[60:63], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_125:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_365 // (op 365) Condition: Done, jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_365
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_365:
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_365:


s_mul_i32 s52, s8, s16 // (op 278) 
s_cmp_gt_u32 s9, s52 // (op 278) 
s_mov_b32 s53, scc // (op 278) 
s_sub_u32 s54, s9, s52 // (op 278) 

s_cmp_eq_i32 s53, 0 // (op 278) 
s_cselect_b32 s52, 0, s54 // (op 278) 


s_cmp_lt_u32 s8, s52 // (op 278) 
s_mov_b32 s53, scc // (op 278) 
s_cmp_eq_i32 s53, 0 // (op 278) 
s_cselect_b32 s54, s52, s8 // (op 278) 


s_cmp_lt_u32 s0, s54 // (op 278) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_SKStreamTileLoop_278 // (op 278) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_SKStreamTileLoop_278 if true)
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_278:















s_mov_b32 s0, 0 // (op 419) call()
s_mov_b32 s1, 0 // (op 420) call()
s_mov_b32 s0, 0 // (op 434) call()
s_mul_i32 s15, s50, s16 // (op 433) 
s_cmp_gt_u32 s51, s15 // (op 433) 
s_mov_b32 s6, scc // (op 433) 
s_sub_u32 s7, s51, s15 // (op 433) 

s_cmp_eq_i32 s6, 0 // (op 433) 
s_cselect_b32 s8, 0, s7 // (op 433) 


s_cmp_lt_u32 s50, s8 // (op 433) 
s_mov_b32 s15, scc // (op 433) 
s_cmp_eq_i32 s15, 0 // (op 433) 
s_cselect_b32 s6, s8, s50 // (op 433) 


s_cmp_lt_u32 s0, s6 // (op 433) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_433 // (op 433) Condition: Top (jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_433 if false)
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_DPStreamTileLoop_433:
v_lshrrev_b32 v0, 6, v1 // (op 2626) 
s_mul_i32 s15, s16, s50 // (op 2626) 
s_add_u32 s15, s15, s0 // (op 2626) 
v_and_b32 v7, 63, v1 // (op 2626) 
s_add_u32 s6, s15, s9 // (op 2626) 

v_lshrrev_b32 v3, 4, v7 // (op 2626) 
s_mul_hi_u32 s15, s6, s13 // (op 2626) 
v_lshlrev_b32 v4, 2, v3 // (op 2626) 

s_sub_u32 s7, s6, s15 // (op 2626) 
v_lshl_add_u32 v3, v0, 4, v4 // (op 2626) 


s_lshr_b32 s8, s7, 1 // (op 2626) 

v_lshrrev_b32 v0, 2, v7 // (op 2626) 

s_add_u32 s7, s8, s15 // (op 2626) 


v_and_b32 v4, 3, v0 // (op 2626) 

s_lshr_b32 s8, s7, s11 // (op 2626) 

v_and_b32 v0, 3, v1 // (op 2626) 
s_and_b32 s7, s11, 2147483648 // (op 2626) 
v_lshl_add_u32 v7, v4, 2, v0 // (op 2626) 


s_cmp_eq_i32 s7, 0 // (op 2626) 
s_mov_b32 s15, scc // (op 2626) 

s_cmp_eq_i32 s15, 0 // (op 2626) 
s_cselect_b32 s7, s6, s8 // (op 2626) 



s_mul_hi_u32 s6, s7, s12 // (op 2626) 
s_sub_u32 s8, s7, s6 // (op 2626) 
s_lshr_b32 s15, s8, 1 // (op 2626) 

s_add_u32 s8, s15, s6 // (op 2626) 


s_lshr_b32 s6, s8, s10 // (op 2626) 

s_and_b32 s8, s10, 2147483648 // (op 2626) 
s_cmp_eq_i32 s8, 0 // (op 2626) 
s_mov_b32 s15, scc // (op 2626) 

s_cmp_eq_i32 s15, 0 // (op 2626) 
s_cselect_b32 s8, s7, s6 // (op 2626) 


s_mul_i32 s6, s8, s14 // (op 2626) 
v_lshl_add_u32 v0, s8, 6, v3 // (op 2626) 


s_sub_u32 s8, s7, s6 // (op 2626) 


v_lshl_add_u32 v3, s8, 4, v7 // (op 2626) 


v_mul_lo_u32 v4, v3, s4 // (op 2626) 

v_add_lshl_u32 v3, v0, v4, 1 // (op 2626) 


s_mul_i32 s19, s4, s23 // (op 2628) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 2628) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 2628) least significant
s_mul_hi_u32 s6, s4, s22 // (op 2628) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2628) most significant: sum
s_add_u32 s19, s19, s6 // (op 2628) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 2628) 

s_mov_b32 s56, s20 // (op 2628) reg expression
s_mov_b32 s57, s21 // (op 2628) reg expression
s_mov_b32 s58, s28 // (op 2628) reg expression
s_mov_b32 s59, 131072 // (op 2628) call()

v_lshrrev_b32 v0, 6, v1 // (op 2884) 
s_mul_i32 s15, s16, s50 // (op 2884) 
s_add_u32 s15, s15, s0 // (op 2884) 
v_and_b32 v4, 63, v1 // (op 2884) 
s_add_u32 s6, s15, s9 // (op 2884) 

v_lshrrev_b32 v7, 4, v4 // (op 2884) 
s_mul_hi_u32 s15, s6, s13 // (op 2884) 
v_lshlrev_b32 v12, 2, v7 // (op 2884) 

s_sub_u32 s7, s6, s15 // (op 2884) 
v_lshl_add_u32 v7, v0, 4, v12 // (op 2884) 


s_lshr_b32 s8, s7, 1 // (op 2884) 

v_lshrrev_b32 v0, 2, v4 // (op 2884) 

s_add_u32 s7, s8, s15 // (op 2884) 


v_and_b32 v4, 3, v0 // (op 2884) 

s_lshr_b32 s8, s7, s11 // (op 2884) 

v_and_b32 v0, 3, v1 // (op 2884) 
s_and_b32 s7, s11, 2147483648 // (op 2884) 
v_lshl_add_u32 v12, v4, 2, v0 // (op 2884) 


s_cmp_eq_i32 s7, 0 // (op 2884) 
s_mov_b32 s15, scc // (op 2884) 

s_cmp_eq_i32 s15, 0 // (op 2884) 
s_cselect_b32 s7, s6, s8 // (op 2884) 



s_mul_hi_u32 s6, s7, s12 // (op 2884) 
s_sub_u32 s8, s7, s6 // (op 2884) 
s_lshr_b32 s15, s8, 1 // (op 2884) 

s_add_u32 s8, s15, s6 // (op 2884) 


s_lshr_b32 s6, s8, s10 // (op 2884) 

s_and_b32 s8, s10, 2147483648 // (op 2884) 
s_cmp_eq_i32 s8, 0 // (op 2884) 
s_mov_b32 s15, scc // (op 2884) 

s_cmp_eq_i32 s15, 0 // (op 2884) 
s_cselect_b32 s8, s7, s6 // (op 2884) 


s_mul_i32 s6, s8, s14 // (op 2884) 
v_lshl_add_u32 v0, s8, 6, v7 // (op 2884) 


s_sub_u32 s8, s7, s6 // (op 2884) 


v_lshl_add_u32 v4, s8, 4, v12 // (op 2884) 


v_mul_lo_u32 v7, v4, s24 // (op 2884) 

v_add_lshl_u32 v4, v0, v7, 1 // (op 2884) 


s_mul_i32 s19, s24, s23 // (op 2886) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 2886) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 2886) least significant
s_mul_hi_u32 s6, s24, s22 // (op 2886) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2886) most significant: sum
s_add_u32 s19, s19, s6 // (op 2886) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 2886) 

s_mov_b32 s60, s26 // (op 2886) reg expression
s_mov_b32 s61, s27 // (op 2886) reg expression
s_mov_b32 s62, s28 // (op 2886) reg expression
s_mov_b32 s63, 131072 // (op 2886) call()

v_and_b32 v0, 48, v1 // (op 2963) 
v_and_b32 v7, 63, v1 // (op 2963) 
v_sub_u32 v12, v7, v0 // (op 2963) 

v_lshrrev_b32 v7, 1, v12 // (op 2963) 
v_add_lshl_u32 v13, v0, v7, 6 // (op 2963) 


v_and_b32 v0, 1, v12 // (op 2963) 

v_lshlrev_b32 v7, 3, v0 // (op 2963) 

v_lshrrev_b32 v0, 6, v1 // (op 2963) 
v_lshl_add_u32 v12, v0, 4, v7 // (op 2963) 


v_add_u32 v0, v13, v12 // (op 2963) 


v_and_b32 v7, 15, v1 // (op 3028) 
v_and_b32 v12, 48, v1 // (op 3028) 
v_lshl_add_u32 v13, v7, 7, v12 // (op 3028) 


v_lshrrev_b32 v7, 6, v1 // (op 3104) 
v_and_b32 v12, 15, v1 // (op 3104) 
v_lshl_add_u32 v16, v7, 4, v12 // (op 3104) 


v_and_b32 v7, 63, v1 // (op 3104) 
v_lshrrev_b32 v12, 4, v7 // (op 3104) 

v_lshl_add_u32 v7, v12, 6, v16 // (op 3104) 


v_accvgpr_write a0, 0.00000 // (op 174) call()
v_accvgpr_write a1, 0.00000 // (op 174) call()
v_accvgpr_write a2, 0.00000 // (op 174) call()
v_accvgpr_write a3, 0.00000 // (op 174) call()
v_accvgpr_write a4, 0.00000 // (op 589) call()
v_accvgpr_write a5, 0.00000 // (op 589) call()
v_accvgpr_write a6, 0.00000 // (op 589) call()
v_accvgpr_write a7, 0.00000 // (op 589) call()
v_and_b32 v12, -4, v1 // (op 2130) 
s_mul_i32 s15, s16, s50 // (op 2130) 
s_add_u32 s15, s15, s0 // (op 2130) 
v_sub_u32 v16, v1, v12 // (op 2130) 

s_add_u32 s6, s15, s9 // (op 2130) 

v_lshlrev_b32 v12, 4, v16 // (op 2130) 

s_mul_hi_u32 s15, s6, s13 // (op 2130) 
v_lshrrev_b32 v16, 2, v1 // (op 2130) 
s_sub_u32 s7, s6, s15 // (op 2130) 
s_lshr_b32 s8, s7, 1 // (op 2130) 

s_add_u32 s7, s8, s15 // (op 2130) 


s_lshr_b32 s8, s7, s11 // (op 2130) 

s_and_b32 s7, s11, 2147483648 // (op 2130) 
s_cmp_eq_i32 s7, 0 // (op 2130) 
s_mov_b32 s15, scc // (op 2130) 

s_cmp_eq_i32 s15, 0 // (op 2130) 
s_cselect_b32 s7, s6, s8 // (op 2130) 


s_mul_hi_u32 s8, s7, s12 // (op 2130) 
s_sub_u32 s15, s7, s8 // (op 2130) 
s_lshr_b32 s18, s15, 1 // (op 2130) 

s_add_u32 s15, s18, s8 // (op 2130) 


s_lshr_b32 s8, s15, s10 // (op 2130) 

s_and_b32 s15, s10, 2147483648 // (op 2130) 
s_cmp_eq_i32 s15, 0 // (op 2130) 
s_mov_b32 s18, scc // (op 2130) 

s_cmp_eq_i32 s18, 0 // (op 2130) 
s_cselect_b32 s15, s7, s8 // (op 2130) 


v_lshl_add_u32 v17, s15, 6, v12 // (op 2130) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2130) 
v_mov_b32 v19, 0 // (op 2130) convert
v_mov_b32 v18, v17 // (op 2130) convert

s_mul_i32 s8, s7, s18 // (op 2130) 


s_sub_u32 s7, s6, s8 // (op 2130) 


v_lshl_add_u32 v12, s7, 7, v16 // (op 2130) 


v_mul_lo_u32 v17, v12, s31 // (op 2130) most significant: low of low * high
v_mul_hi_u32 v20, v12, s30 // (op 2130) most significant: high of low * low
v_mul_lo_u32 v16, v12, s30 // (op 2130) least significant: low of low * low
v_add_u32 v17, v17, v20 // (op 2130) most significant: sum


v_add_co_u32 v20, s[18:19], v18, v16 // (op 2130) least significant half
v_addc_co_u32 v21, s[18:19], v19, v17, s[18:19] // (op 2130) most significant half



s_mul_i32 s19, s30, s3 // (op 2132) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 2132) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 2132) least significant
s_mul_hi_u32 s6, s30, s2 // (op 2132) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2132) most significant: sum
s_add_u32 s19, s19, s6 // (op 2132) most significant: sum


s_mov_b32 s68, s32 // (op 2132) reg expression
s_mov_b32 s69, s33 // (op 2132) reg expression
s_mov_b32 s70, s18 // (op 2132) reg expression
s_mov_b32 s71, 131072 // (op 2132) call()

v_and_b32 v12, -4, v1 // (op 2149) 
s_mul_i32 s15, s16, s50 // (op 2149) 
s_add_u32 s15, s15, s0 // (op 2149) 
v_sub_u32 v16, v1, v12 // (op 2149) 

s_add_u32 s6, s15, s9 // (op 2149) 

v_lshlrev_b32 v12, 4, v16 // (op 2149) 

s_mul_hi_u32 s15, s6, s13 // (op 2149) 
v_lshrrev_b32 v16, 2, v1 // (op 2149) 
s_sub_u32 s7, s6, s15 // (op 2149) 
s_lshr_b32 s8, s7, 1 // (op 2149) 

s_add_u32 s7, s8, s15 // (op 2149) 


s_lshr_b32 s8, s7, s11 // (op 2149) 

s_and_b32 s7, s11, 2147483648 // (op 2149) 
s_cmp_eq_i32 s7, 0 // (op 2149) 
s_mov_b32 s15, scc // (op 2149) 

s_cmp_eq_i32 s15, 0 // (op 2149) 
s_cselect_b32 s7, s6, s8 // (op 2149) 


s_mul_hi_u32 s8, s7, s12 // (op 2149) 
s_sub_u32 s15, s7, s8 // (op 2149) 
s_lshr_b32 s18, s15, 1 // (op 2149) 

s_add_u32 s15, s18, s8 // (op 2149) 


s_lshr_b32 s8, s15, s10 // (op 2149) 

s_and_b32 s15, s10, 2147483648 // (op 2149) 
s_cmp_eq_i32 s15, 0 // (op 2149) 
s_mov_b32 s18, scc // (op 2149) 

s_cmp_eq_i32 s18, 0 // (op 2149) 
s_cselect_b32 s15, s7, s8 // (op 2149) 


v_lshl_add_u32 v17, s15, 6, v12 // (op 2149) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2149) 
v_mov_b32 v19, 0 // (op 2149) convert
v_mov_b32 v18, v17 // (op 2149) convert

s_mul_i32 s8, s7, s18 // (op 2149) 


s_sub_u32 s7, s6, s8 // (op 2149) 


v_lshl_add_u32 v12, s7, 7, v16 // (op 2149) 


v_mul_lo_u32 v17, v12, s31 // (op 2149) most significant: low of low * high
v_mul_hi_u32 v22, v12, s30 // (op 2149) most significant: high of low * low
v_mul_lo_u32 v16, v12, s30 // (op 2149) least significant: low of low * low
v_add_u32 v17, v17, v22 // (op 2149) most significant: sum


v_add_co_u32 v22, s[18:19], v18, v16 // (op 2149) least significant half
v_addc_co_u32 v23, s[18:19], v19, v17, s[18:19] // (op 2149) most significant half



s_mul_i32 s19, s30, s3 // (op 2151) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 2151) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 2151) least significant
s_mul_hi_u32 s6, s30, s2 // (op 2151) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2151) most significant: sum
s_add_u32 s19, s19, s6 // (op 2151) most significant: sum


s_mov_b32 s68, s32 // (op 2151) reg expression
s_mov_b32 s69, s33 // (op 2151) reg expression
s_mov_b32 s70, s18 // (op 2151) reg expression
s_mov_b32 s71, 131072 // (op 2151) call()

v_lshrrev_b32 v12, 3, v1 // (op 2262) 
s_mul_i32 s15, s16, s50 // (op 2262) 
s_add_u32 s15, s15, s0 // (op 2262) 
v_and_b32 v16, -8, v1 // (op 2262) 
s_add_u32 s6, s15, s9 // (op 2262) 

v_sub_u32 v17, v1, v16 // (op 2262) 

s_mul_hi_u32 s15, s6, s13 // (op 2262) 
v_lshlrev_b32 v16, 4, v17 // (op 2262) 

s_sub_u32 s7, s6, s15 // (op 2262) 
s_lshr_b32 s8, s7, 1 // (op 2262) 

s_add_u32 s7, s8, s15 // (op 2262) 


s_lshr_b32 s8, s7, s11 // (op 2262) 

s_and_b32 s7, s11, 2147483648 // (op 2262) 
s_cmp_eq_i32 s7, 0 // (op 2262) 
s_mov_b32 s15, scc // (op 2262) 

s_cmp_eq_i32 s15, 0 // (op 2262) 
s_cselect_b32 s7, s6, s8 // (op 2262) 


s_mul_hi_u32 s8, s7, s12 // (op 2262) 
s_sub_u32 s15, s7, s8 // (op 2262) 
s_lshr_b32 s18, s15, 1 // (op 2262) 

s_add_u32 s15, s18, s8 // (op 2262) 


s_lshr_b32 s8, s15, s10 // (op 2262) 

s_and_b32 s15, s10, 2147483648 // (op 2262) 
s_cmp_eq_i32 s15, 0 // (op 2262) 
s_mov_b32 s18, scc // (op 2262) 

s_cmp_eq_i32 s18, 0 // (op 2262) 
s_cselect_b32 s15, s7, s8 // (op 2262) 


s_mul_i32 s8, s15, s14 // (op 2262) 

s_sub_u32 s15, s7, s8 // (op 2262) 

v_lshl_add_u32 v17, s15, 4, v12 // (op 2262) 


v_mul_lo_u32 v19, v17, s35 // (op 2262) most significant: low of low * high
v_mul_hi_u32 v12, v17, s34 // (op 2262) most significant: high of low * low
v_mul_lo_u32 v18, v17, s34 // (op 2262) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 2262) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2262) 
s_mul_i32 s8, s7, s18 // (op 2262) 


s_sub_u32 s7, s6, s8 // (op 2262) 


v_lshl_add_u32 v12, s7, 7, v16 // (op 2262) 


v_mov_b32 v17, 0 // (op 2262) convert
v_mov_b32 v16, v12 // (op 2262) convert

v_add_co_u32 v24, s[18:19], v16, v18 // (op 2262) least significant half
v_addc_co_u32 v25, s[18:19], v17, v19, s[18:19] // (op 2262) most significant half



s_mul_i32 s19, s34, s23 // (op 2264) most significant: low of low * high
s_mul_i32 s15, s35, s22 // (op 2264) most significant: low of high * low
s_mul_i32 s18, s34, s22 // (op 2264) least significant
s_mul_hi_u32 s6, s34, s22 // (op 2264) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2264) most significant: sum
s_add_u32 s19, s19, s6 // (op 2264) most significant: sum


s_mov_b32 s72, s36 // (op 2264) reg expression
s_mov_b32 s73, s37 // (op 2264) reg expression
s_mov_b32 s74, s18 // (op 2264) reg expression
s_mov_b32 s75, 131072 // (op 2264) call()

v_lshrrev_b32 v12, 3, v1 // (op 2281) 
s_mul_i32 s15, s16, s50 // (op 2281) 
s_add_u32 s15, s15, s0 // (op 2281) 
v_and_b32 v16, -8, v1 // (op 2281) 
s_add_u32 s6, s15, s9 // (op 2281) 

v_sub_u32 v17, v1, v16 // (op 2281) 

s_mul_hi_u32 s15, s6, s13 // (op 2281) 
v_lshlrev_b32 v16, 4, v17 // (op 2281) 

s_sub_u32 s7, s6, s15 // (op 2281) 
s_lshr_b32 s8, s7, 1 // (op 2281) 

s_add_u32 s7, s8, s15 // (op 2281) 


s_lshr_b32 s8, s7, s11 // (op 2281) 

s_and_b32 s7, s11, 2147483648 // (op 2281) 
s_cmp_eq_i32 s7, 0 // (op 2281) 
s_mov_b32 s15, scc // (op 2281) 

s_cmp_eq_i32 s15, 0 // (op 2281) 
s_cselect_b32 s7, s6, s8 // (op 2281) 


s_mul_hi_u32 s8, s7, s12 // (op 2281) 
s_sub_u32 s15, s7, s8 // (op 2281) 
s_lshr_b32 s18, s15, 1 // (op 2281) 

s_add_u32 s15, s18, s8 // (op 2281) 


s_lshr_b32 s8, s15, s10 // (op 2281) 

s_and_b32 s15, s10, 2147483648 // (op 2281) 
s_cmp_eq_i32 s15, 0 // (op 2281) 
s_mov_b32 s18, scc // (op 2281) 

s_cmp_eq_i32 s18, 0 // (op 2281) 
s_cselect_b32 s15, s7, s8 // (op 2281) 


s_mul_i32 s8, s15, s14 // (op 2281) 

s_sub_u32 s15, s7, s8 // (op 2281) 

v_lshl_add_u32 v17, s15, 4, v12 // (op 2281) 


v_mul_lo_u32 v19, v17, s35 // (op 2281) most significant: low of low * high
v_mul_hi_u32 v12, v17, s34 // (op 2281) most significant: high of low * low
v_mul_lo_u32 v18, v17, s34 // (op 2281) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 2281) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2281) 
s_mul_i32 s8, s7, s18 // (op 2281) 


s_sub_u32 s7, s6, s8 // (op 2281) 


v_lshl_add_u32 v12, s7, 7, v16 // (op 2281) 


v_mov_b32 v17, 0 // (op 2281) convert
v_mov_b32 v16, v12 // (op 2281) convert

v_add_co_u32 v26, s[18:19], v16, v18 // (op 2281) least significant half
v_addc_co_u32 v27, s[18:19], v17, v19, s[18:19] // (op 2281) most significant half



s_mul_i32 s19, s34, s23 // (op 2283) most significant: low of low * high
s_mul_i32 s15, s35, s22 // (op 2283) most significant: low of high * low
s_mul_i32 s18, s34, s22 // (op 2283) least significant
s_mul_hi_u32 s6, s34, s22 // (op 2283) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2283) most significant: sum
s_add_u32 s19, s19, s6 // (op 2283) most significant: sum


s_mov_b32 s72, s36 // (op 2283) reg expression
s_mov_b32 s73, s37 // (op 2283) reg expression
s_mov_b32 s74, s18 // (op 2283) reg expression
s_mov_b32 s75, 131072 // (op 2283) call()

v_and_b32 v12, -64, v1 // (op 2390) 
s_mul_i32 s15, s16, s50 // (op 2390) 
s_add_u32 s15, s15, s0 // (op 2390) 
v_sub_u32 v16, v1, v12 // (op 2390) 

s_add_u32 s6, s15, s9 // (op 2390) 

v_lshrrev_b32 v12, 6, v1 // (op 2390) 
s_mul_hi_u32 s15, s6, s13 // (op 2390) 
v_lshlrev_b32 v17, 1, v12 // (op 2390) 

s_sub_u32 s7, s6, s15 // (op 2390) 
s_lshr_b32 s8, s7, 1 // (op 2390) 

s_add_u32 s7, s8, s15 // (op 2390) 


s_lshr_b32 s8, s7, s11 // (op 2390) 

s_and_b32 s7, s11, 2147483648 // (op 2390) 
s_cmp_eq_i32 s7, 0 // (op 2390) 
s_mov_b32 s15, scc // (op 2390) 

s_cmp_eq_i32 s15, 0 // (op 2390) 
s_cselect_b32 s7, s6, s8 // (op 2390) 


s_mul_hi_u32 s8, s7, s12 // (op 2390) 
s_sub_u32 s15, s7, s8 // (op 2390) 
s_lshr_b32 s18, s15, 1 // (op 2390) 

s_add_u32 s15, s18, s8 // (op 2390) 


s_lshr_b32 s8, s15, s10 // (op 2390) 

s_and_b32 s15, s10, 2147483648 // (op 2390) 
s_cmp_eq_i32 s15, 0 // (op 2390) 
s_mov_b32 s18, scc // (op 2390) 

s_cmp_eq_i32 s18, 0 // (op 2390) 
s_cselect_b32 s15, s7, s8 // (op 2390) 


v_lshl_add_u32 v12, s15, 6, v16 // (op 2390) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2390) 
s_mul_i32 s8, s7, s18 // (op 2390) 


s_sub_u32 s7, s6, s8 // (op 2390) 


v_lshl_add_u32 v16, s7, 2, v17 // (op 2390) 


v_mul_lo_u32 v17, v16, s38 // (op 2390) 

v_add_u32 v16, v12, v17 // (op 2390) 


s_mul_i32 s19, s38, s43 // (op 2392) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 2392) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 2392) least significant
s_mul_hi_u32 s6, s38, s42 // (op 2392) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2392) most significant: sum
s_add_u32 s19, s19, s6 // (op 2392) most significant: sum


s_mov_b32 s76, s40 // (op 2392) reg expression
s_mov_b32 s77, s41 // (op 2392) reg expression
s_mov_b32 s78, s18 // (op 2392) reg expression
s_mov_b32 s79, 131072 // (op 2392) call()

v_and_b32 v12, -64, v1 // (op 2409) 
s_mul_i32 s15, s16, s50 // (op 2409) 
s_add_u32 s15, s15, s0 // (op 2409) 
v_sub_u32 v17, v1, v12 // (op 2409) 

s_add_u32 s6, s15, s9 // (op 2409) 

v_lshrrev_b32 v12, 6, v1 // (op 2409) 
s_mul_hi_u32 s15, s6, s13 // (op 2409) 
v_lshlrev_b32 v18, 1, v12 // (op 2409) 

s_sub_u32 s7, s6, s15 // (op 2409) 
s_lshr_b32 s8, s7, 1 // (op 2409) 

s_add_u32 s7, s8, s15 // (op 2409) 


s_lshr_b32 s8, s7, s11 // (op 2409) 

s_and_b32 s7, s11, 2147483648 // (op 2409) 
s_cmp_eq_i32 s7, 0 // (op 2409) 
s_mov_b32 s15, scc // (op 2409) 

s_cmp_eq_i32 s15, 0 // (op 2409) 
s_cselect_b32 s7, s6, s8 // (op 2409) 


s_mul_hi_u32 s8, s7, s12 // (op 2409) 
s_sub_u32 s15, s7, s8 // (op 2409) 
s_lshr_b32 s18, s15, 1 // (op 2409) 

s_add_u32 s15, s18, s8 // (op 2409) 


s_lshr_b32 s8, s15, s10 // (op 2409) 

s_and_b32 s15, s10, 2147483648 // (op 2409) 
s_cmp_eq_i32 s15, 0 // (op 2409) 
s_mov_b32 s18, scc // (op 2409) 

s_cmp_eq_i32 s18, 0 // (op 2409) 
s_cselect_b32 s15, s7, s8 // (op 2409) 


v_lshl_add_u32 v12, s15, 6, v17 // (op 2409) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2409) 
s_mul_i32 s8, s7, s18 // (op 2409) 


s_sub_u32 s7, s6, s8 // (op 2409) 


v_lshl_add_u32 v17, s7, 2, v18 // (op 2409) 


v_mul_lo_u32 v18, v17, s38 // (op 2409) 

v_add_u32 v17, v12, v18 // (op 2409) 


s_mul_i32 s19, s38, s43 // (op 2411) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 2411) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 2411) least significant
s_mul_hi_u32 s6, s38, s42 // (op 2411) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2411) most significant: sum
s_add_u32 s19, s19, s6 // (op 2411) most significant: sum


s_mov_b32 s76, s40 // (op 2411) reg expression
s_mov_b32 s77, s41 // (op 2411) reg expression
s_mov_b32 s78, s18 // (op 2411) reg expression
s_mov_b32 s79, 131072 // (op 2411) call()

v_and_b32 v12, 15, v1 // (op 2518) 
s_mul_i32 s15, s16, s50 // (op 2518) 
s_add_u32 s15, s15, s0 // (op 2518) 
v_and_b32 v18, 63, v1 // (op 2518) 
s_add_u32 s6, s15, s9 // (op 2518) 

v_lshrrev_b32 v19, 4, v18 // (op 2518) 

s_mul_hi_u32 s15, s6, s13 // (op 2518) 
s_sub_u32 s7, s6, s15 // (op 2518) 
s_lshr_b32 s8, s7, 1 // (op 2518) 

s_add_u32 s7, s8, s15 // (op 2518) 


s_lshr_b32 s8, s7, s11 // (op 2518) 

s_and_b32 s7, s11, 2147483648 // (op 2518) 
s_cmp_eq_i32 s7, 0 // (op 2518) 
s_mov_b32 s15, scc // (op 2518) 

s_cmp_eq_i32 s15, 0 // (op 2518) 
s_cselect_b32 s7, s6, s8 // (op 2518) 


s_mul_hi_u32 s8, s7, s12 // (op 2518) 
s_sub_u32 s15, s7, s8 // (op 2518) 
s_lshr_b32 s18, s15, 1 // (op 2518) 

s_add_u32 s15, s18, s8 // (op 2518) 


s_lshr_b32 s8, s15, s10 // (op 2518) 

s_and_b32 s15, s10, 2147483648 // (op 2518) 
s_cmp_eq_i32 s15, 0 // (op 2518) 
s_mov_b32 s18, scc // (op 2518) 

s_cmp_eq_i32 s18, 0 // (op 2518) 
s_cselect_b32 s15, s7, s8 // (op 2518) 


s_mul_i32 s8, s15, s14 // (op 2518) 

s_sub_u32 s15, s7, s8 // (op 2518) 

v_lshl_add_u32 v18, s15, 4, v12 // (op 2518) 


v_mul_lo_u32 v12, v18, s44 // (op 2518) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 2518) 
s_mul_i32 s8, s7, s18 // (op 2518) 


s_sub_u32 s7, s6, s8 // (op 2518) 


v_lshl_add_u32 v18, s7, 2, v19 // (op 2518) 


v_add_u32 v19, v18, v12 // (op 2518) 


s_mul_i32 s19, s44, s23 // (op 2520) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 2520) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 2520) least significant
s_mul_hi_u32 s6, s44, s22 // (op 2520) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2520) most significant: sum
s_add_u32 s19, s19, s6 // (op 2520) most significant: sum


s_mov_b32 s80, s46 // (op 2520) reg expression
s_mov_b32 s81, s47 // (op 2520) reg expression
s_mov_b32 s82, s18 // (op 2520) reg expression
s_mov_b32 s83, 131072 // (op 2520) call()

v_and_b32 v12, 15, v1 // (op 2537) 
s_mul_i32 s15, s16, s50 // (op 2537) 
s_add_u32 s15, s15, s0 // (op 2537) 
v_and_b32 v18, 63, v1 // (op 2537) 
s_add_u32 s6, s15, s9 // (op 2537) 

v_lshrrev_b32 v28, 4, v18 // (op 2537) 

s_mul_hi_u32 s15, s6, s13 // (op 2537) 
s_sub_u32 s7, s6, s15 // (op 2537) 
s_lshr_b32 s8, s7, 1 // (op 2537) 

s_add_u32 s7, s8, s15 // (op 2537) 


s_lshr_b32 s8, s7, s11 // (op 2537) 

s_and_b32 s7, s11, 2147483648 // (op 2537) 
s_cmp_eq_i32 s7, 0 // (op 2537) 
s_mov_b32 s15, scc // (op 2537) 

s_cmp_eq_i32 s15, 0 // (op 2537) 
s_cselect_b32 s7, s6, s8 // (op 2537) 


s_mul_hi_u32 s8, s7, s12 // (op 2537) 
s_sub_u32 s15, s7, s8 // (op 2537) 
s_lshr_b32 s18, s15, 1 // (op 2537) 

s_add_u32 s15, s18, s8 // (op 2537) 


s_lshr_b32 s8, s15, s10 // (op 2537) 

s_and_b32 s15, s10, 2147483648 // (op 2537) 
s_cmp_eq_i32 s15, 0 // (op 2537) 
s_mov_b32 s18, scc // (op 2537) 

s_cmp_eq_i32 s18, 0 // (op 2537) 
s_cselect_b32 s15, s7, s8 // (op 2537) 


s_mul_i32 s8, s15, s14 // (op 2537) 

s_sub_u32 s15, s7, s8 // (op 2537) 

v_lshl_add_u32 v18, s15, 4, v12 // (op 2537) 


v_mul_lo_u32 v12, v18, s44 // (op 2537) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 2537) 
s_mul_i32 s8, s7, s18 // (op 2537) 


s_sub_u32 s7, s6, s8 // (op 2537) 


v_lshl_add_u32 v18, s7, 2, v28 // (op 2537) 


v_add_u32 v28, v18, v12 // (op 2537) 


s_mul_i32 s19, s44, s23 // (op 2539) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 2539) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 2539) least significant
s_mul_hi_u32 s6, s44, s22 // (op 2539) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2539) most significant: sum
s_add_u32 s19, s19, s6 // (op 2539) most significant: sum


s_mov_b32 s80, s46 // (op 2539) reg expression
s_mov_b32 s81, s47 // (op 2539) reg expression
s_mov_b32 s82, s18 // (op 2539) reg expression
s_mov_b32 s83, 131072 // (op 2539) call()




buffer_load_ubyte v12, v19, s[80:83], 0 offen offset: 0 // (op 1748) Load value
v_and_b32 v18, -8, v1 // (op 3004) 
v_sub_u32 v29, v1, v18 // (op 3004) 

v_lshlrev_b32 v18, 4, v29 // (op 3004) 

v_lshrrev_b32 v29, 3, v1 // (op 3004) 
v_lshl_add_u32 v30, v29, 7, v18 // (op 3004) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 3004) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3529) 



s_nop 0
s_mov_b32 m0, s15 // (op 2040) reg expression
buffer_load_dwordx4 v24, s[72:75], 0 offen offset: 0 lds // (op 2040) Load value direct to lds

v_and_b32 v18, -4, v1 // (op 2929) 
v_sub_u32 v29, v1, v18 // (op 2929) 

v_lshlrev_b32 v18, 4, v29 // (op 2929) 

v_lshrrev_b32 v29, 2, v1 // (op 2929) 
v_lshl_add_u32 v30, v29, 6, v18 // (op 2929) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 2929) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3510) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 2045) 
v_mov_b64 v[30:31], s[18:19] // (op 2045) convert


v_mov_b64 v[32:33], v[20:21] // (op 2045) reg expression
s_add_u32 s6, 2048, s15 // (op 2045) 
s_mov_b32 m0, s6 // (op 2045) reg expression
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 2045) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 2045) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 2045) most significant half

s_add_u32 m0, m0, 2048 // (op 2045) 
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 2045) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 2045) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 2045) most significant half

s_add_u32 m0, m0, 2048 // (op 2045) 
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 2045) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 2045) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 2045) most significant half

s_add_u32 m0, m0, 2048 // (op 2045) 
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 2045) Load value direct to lds





v_mov_b64 v[30:31], s[38:39] // (op 1758) convert

v_mov_b32 v18, v16 // (op 1758) reg expression
buffer_load_ubyte v32, v18, s[76:79], 0 offen offset: 0 // (op 1758) Load value
v_add_u32 v18, v18, v30 // (op 1758) 
buffer_load_ubyte v33, v18, s[76:79], 0 offen offset: 0 // (op 1758) Load value





buffer_load_ubyte v18, v19, s[80:83], 0 offen offset:4 // (op 1762) Load value

v_and_b32 v19, -8, v1 // (op 3218) 
v_sub_u32 v29, v1, v19 // (op 3218) 

v_lshlrev_b32 v19, 4, v29 // (op 3218) 

v_lshrrev_b32 v29, 3, v1 // (op 3218) 
v_lshl_add_u32 v30, v29, 7, v19 // (op 3218) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 3218) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3577) 


s_nop 0
s_add_u32 s6, 10240, s15 // (op 2050) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s6 // (op 2050) reg expression
s_mov_b32 s7, 128 // (op 2050) call()
buffer_load_dwordx4 v24, s[72:75], s7 offen offset: 0 lds // (op 2050) Load value direct to lds




v_and_b32 v19, -4, v1 // (op 3174) 
v_sub_u32 v24, v1, v19 // (op 3174) 

v_lshlrev_b32 v19, 4, v24 // (op 3174) 

v_lshrrev_b32 v24, 2, v1 // (op 3174) 
v_lshl_add_u32 v25, v24, 6, v19 // (op 3174) 


s_nop 0
v_readfirstlane_b32 s15, v25 // (op 3174) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3568) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 2052) 
v_mov_b64 v[24:25], s[18:19] // (op 2052) convert


s_lshl_b64 s[18:19], s[30:31], 7 // (op 2052) 
v_mov_b32 v19, s18 // (op 2052) 
v_mov_b32 v29, s19 // (op 2052) 
v_add_co_u32 v30, s[28:29], v20, v19 // (op 2052) least significant half
v_addc_co_u32 v31, s[28:29], v21, v29, s[28:29] // (op 2052) most significant half




s_add_u32 s6, 12288, s15 // (op 2052) 
s_mov_b32 m0, s6 // (op 2052) reg expression
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 2052) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v24 // (op 2052) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 2052) most significant half

s_add_u32 m0, m0, 2048 // (op 2052) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 2052) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v24 // (op 2052) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 2052) most significant half

s_add_u32 m0, m0, 2048 // (op 2052) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 2052) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v24 // (op 2052) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 2052) most significant half

s_add_u32 m0, m0, 2048 // (op 2052) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 2052) Load value direct to lds






v_mov_b64 v[20:21], s[38:39] // (op 1772) convert

s_lshl_b32 s15, s38, 2 // (op 1772) 
v_add_u32 v19, v16, s15 // (op 1772) 

buffer_load_ubyte v24, v19, s[76:79], 0 offen offset: 0 // (op 1772) Load value
v_add_u32 v19, v19, v20 // (op 1772) 
buffer_load_ubyte v25, v19, s[76:79], 0 offen offset: 0 // (op 1772) Load value



v_and_b32 v16, -64, v1 // (op 3070) 
v_sub_u32 v19, v1, v16 // (op 3070) 

v_lshrrev_b32 v16, 6, v1 // (op 3070) 
v_lshl_add_u32 v20, v16, 7, v19 // (op 3070) 


s_barrier  // (op 3548) 



s_waitcnt vmcnt(9)
ds_write_b8 v20, v32 offset:20480 // (op 1776) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v20, v33 offset:20544 // (op 1776) Store local data 

s_barrier  // (op 3526) 



ds_read_b128 v[34:37], v13 // (op 1799) Load local data 
ds_read_b128 v[38:41], v13 offset:64 // (op 1799) Load local data 
s_barrier  // (op 3507) 



ds_read_b64_tr_b8 v[42:43], v0 offset:2048 // (op 1790) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:2560 // (op 1790) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:6144 // (op 1790) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:6656 // (op 1790) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 3545) 


ds_read_u8 v16, v7 offset:20480 // (op 1803) Load local data 



ds_read_b64_tr_b8 v[50:51], v0 offset:2080 // (op 1795) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:2592 // (op 1795) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:6176 // (op 1795) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:6688 // (op 1795) Transpose load local data 


ds_read_u8 v19, v7 offset:20512 // (op 1808) Load local data 
v_and_b32 v20, -4, v1 // (op 2901) 
v_sub_u32 v21, v1, v20 // (op 2901) 

v_lshlrev_b32 v20, 4, v21 // (op 2901) 

v_lshrrev_b32 v21, 2, v1 // (op 2901) 
v_lshl_add_u32 v29, v21, 6, v20 // (op 2901) 


s_nop 0
v_readfirstlane_b32 s15, v29 // (op 2901) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -8, v1 // (op 2976) 
v_sub_u32 v21, v1, v20 // (op 2976) 

v_lshlrev_b32 v20, 4, v21 // (op 2976) 

v_lshrrev_b32 v21, 3, v1 // (op 2976) 
v_lshl_add_u32 v29, v21, 7, v20 // (op 2976) 


s_nop 0
v_readfirstlane_b32 s6, v29 // (op 2976) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -64, v1 // (op 3041) 
v_sub_u32 v21, v1, v20 // (op 3041) 

v_lshrrev_b32 v20, 6, v1 // (op 3041) 
v_lshl_add_u32 v29, v20, 7, v21 // (op 3041) 


v_and_b32 v20, -4, v1 // (op 3159) 
v_sub_u32 v21, v1, v20 // (op 3159) 

v_lshlrev_b32 v20, 4, v21 // (op 3159) 

v_lshrrev_b32 v21, 2, v1 // (op 3159) 
v_lshl_add_u32 v30, v21, 6, v20 // (op 3159) 


s_nop 0
v_readfirstlane_b32 s7, v30 // (op 3159) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -8, v1 // (op 3203) 
v_sub_u32 v21, v1, v20 // (op 3203) 

v_lshlrev_b32 v20, 4, v21 // (op 3203) 

v_lshrrev_b32 v21, 3, v1 // (op 3203) 
v_lshl_add_u32 v30, v21, 7, v20 // (op 3203) 


s_nop 0
v_readfirstlane_b32 s8, v30 // (op 3203) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -64, v1 // (op 3242) 
v_sub_u32 v21, v1, v20 // (op 3242) 

v_lshrrev_b32 v20, 6, v1 // (op 3242) 
v_lshl_add_u32 v30, v20, 7, v21 // (op 3242) 


s_mov_b32 s1, 0 // (op 591) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 590) 
s_mov_b32 s28, s18 // (op 590) convert

s_and_b32 s29, s28, 4294967294 // (op 590) 

s_cmp_lt_u32 s1, s29 // (op 590) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_590 // (op 590) Condition: Top (jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_590 if false)
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoop_590:
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[34:41], a[0:3], v16, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 188) 
s_add_u32 s1, s1, 2 // (op 592) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[34:41], a[4:7], v19, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 626) 


v_mov_b32 v31, v28 // (op 187) 

buffer_load_ubyte v12, v31, s[80:83], 0 offen offset:8 // (op 187) Load value
s_barrier  // (op 3532) 


v_mov_b64 v[20:21], v[26:27] // (op 1989) 

s_mov_b32 m0, s6 // (op 1989) reg expression
s_mov_b32 s18, 256 // (op 1989) call()
buffer_load_dwordx4 v20, s[72:75], s18 offen offset: 0 lds // (op 1989) Load value direct to lds

s_barrier  // (op 3513) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 1984) 
v_mov_b64 v[58:59], s[18:19] // (op 1984) convert


v_mov_b64 v[60:61], v[22:23] // (op 1984) 
s_lshl_b64 s[18:19], s[30:31], 7 // (op 1984) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 1984) 

v_mov_b32 v62, s28 // (op 1984) 
v_mov_b32 v63, s29 // (op 1984) 
v_add_co_u32 v64, s[18:19], v60, v62 // (op 1984) least significant half
v_addc_co_u32 v65, s[18:19], v61, v63, s[18:19] // (op 1984) most significant half




s_add_u32 s18, 2048, s15 // (op 1984) 
s_nop 0
s_mov_b32 m0, s18 // (op 1984) reg expression
buffer_load_dwordx4 v64, s[68:71], 0 offen offset: 0 lds // (op 1984) Load value direct to lds
v_add_co_u32 v64, s[28:29], v64, v58 // (op 1984) least significant half
v_addc_co_u32 v65, s[28:29], 0, v65, s[28:29] // (op 1984) most significant half

s_add_u32 m0, m0, 2048 // (op 1984) 
buffer_load_dwordx4 v64, s[68:71], 0 offen offset: 0 lds // (op 1984) Load value direct to lds
v_add_co_u32 v64, s[28:29], v64, v58 // (op 1984) least significant half
v_addc_co_u32 v65, s[28:29], 0, v65, s[28:29] // (op 1984) most significant half

s_add_u32 m0, m0, 2048 // (op 1984) 
buffer_load_dwordx4 v64, s[68:71], 0 offen offset: 0 lds // (op 1984) Load value direct to lds
v_add_co_u32 v64, s[28:29], v64, v58 // (op 1984) least significant half
v_addc_co_u32 v65, s[28:29], 0, v65, s[28:29] // (op 1984) most significant half

s_add_u32 m0, m0, 2048 // (op 1984) 
buffer_load_dwordx4 v64, s[68:71], 0 offen offset: 0 lds // (op 1984) Load value direct to lds




v_mov_b64 v[58:59], s[38:39] // (op 192) convert

v_mov_b32 v62, v17 // (op 192) 
s_lshl_b32 s18, s38, 3 // (op 192) 
v_add_u32 v63, v62, s18 // (op 192) 

buffer_load_ubyte v32, v63, s[76:79], 0 offen offset: 0 // (op 192) Load value
v_add_u32 v63, v63, v58 // (op 192) 
buffer_load_ubyte v33, v63, s[76:79], 0 offen offset: 0 // (op 192) Load value


s_barrier  // (op 3586) 



s_waitcnt vmcnt(9)
ds_write_b8 v30, v24 offset:20736 // (op 516) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v30, v25 offset:20800 // (op 516) Store local data 
s_barrier  // (op 3574) 



ds_read_b128 v[34:37], v6 offset:10240 // (op 509) Load local data 
ds_read_b128 v[38:41], v6 offset:10304 // (op 509) Load local data 
s_barrier  // (op 3565) 



ds_read_b64_tr_b8 v[42:43], v10 offset:12288 // (op 507) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v10 offset:12800 // (op 507) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v10 offset:16384 // (op 507) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v10 offset:16896 // (op 507) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 3583) 


ds_read_u8 v16, v11 offset:20736 // (op 511) Load local data 



ds_read_b64_tr_b8 v[50:51], v10 offset:12320 // (op 633) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v10 offset:12832 // (op 633) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v10 offset:16416 // (op 633) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v10 offset:16928 // (op 633) Transpose load local data 


ds_read_u8 v19, v11 offset:20768 // (op 637) Load local data 
s_barrier  // (op 3705) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[34:41], a[0:3], v16, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 514) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[34:41], a[4:7], v19, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 627) 



buffer_load_ubyte v18, v31, s[80:83], 0 offen offset:12 // (op 513) Load value

s_barrier  // (op 3580) 
v_add_u32 v28, 8, v28 // (op 2529) 


s_add_u32 s18, 10240, s8 // (op 2012) 

s_mov_b32 m0, s18 // (op 2012) reg expression
s_mov_b32 s19, 384 // (op 2012) call()
buffer_load_dwordx4 v20, s[72:75], s19 offen offset: 0 lds // (op 2012) Load value direct to lds



s_barrier  // (op 3571) 
v_add_co_u32 v26, vcc, 256, v26 // (op 2273) least significant half

v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 2273) most significant half
s_lshl_b64 s[18:19], s[30:31], 5 // (op 2007) 
v_mov_b64 v[20:21], s[18:19] // (op 2007) convert


s_lshl_b64 s[18:19], s[30:31], 7 // (op 2007) 
s_mul_i32 s29, 3, s19 // (op 2007) most significant: low of low * high
s_mul_i32 s28, 3, s18 // (op 2007) least significant
s_mul_hi_u32 s48, 3, s18 // (op 2007) most significant: high of low * low
s_add_u32 s29, s29, s48 // (op 2007) most significant: sum


v_mov_b32 v31, s28 // (op 2007) 
v_mov_b32 v58, s29 // (op 2007) 
v_add_co_u32 v64, s[18:19], v60, v31 // (op 2007) least significant half
v_addc_co_u32 v65, s[18:19], v61, v58, s[18:19] // (op 2007) most significant half




s_add_u32 s18, 12288, s7 // (op 2007) 
s_nop 0
s_mov_b32 m0, s18 // (op 2007) reg expression
buffer_load_dwordx4 v64, s[68:71], 0 offen offset: 0 lds // (op 2007) Load value direct to lds
v_add_co_u32 v64, s[28:29], v64, v20 // (op 2007) least significant half
v_addc_co_u32 v65, s[28:29], 0, v65, s[28:29] // (op 2007) most significant half

s_add_u32 m0, m0, 2048 // (op 2007) 
buffer_load_dwordx4 v64, s[68:71], 0 offen offset: 0 lds // (op 2007) Load value direct to lds
v_add_co_u32 v64, s[28:29], v64, v20 // (op 2007) least significant half
v_addc_co_u32 v65, s[28:29], 0, v65, s[28:29] // (op 2007) most significant half

s_add_u32 m0, m0, 2048 // (op 2007) 
buffer_load_dwordx4 v64, s[68:71], 0 offen offset: 0 lds // (op 2007) Load value direct to lds
v_add_co_u32 v64, s[28:29], v64, v20 // (op 2007) least significant half
v_addc_co_u32 v65, s[28:29], 0, v65, s[28:29] // (op 2007) most significant half

s_add_u32 m0, m0, 2048 // (op 2007) 
buffer_load_dwordx4 v64, s[68:71], 0 offen offset: 0 lds // (op 2007) Load value direct to lds





v_mov_b64 v[20:21], s[38:39] // (op 518) convert

s_lshl_b32 s18, s38, 2 // (op 518) 
s_mul_i32 s19, 3, s18 // (op 518) 

v_add_u32 v31, v62, s19 // (op 518) 

s_lshl_b64 s[18:19], s[30:31], 7 // (op 2141) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 2141) 

v_mov_b32 v58, s28 // (op 2141) 
v_mov_b32 v59, s29 // (op 2141) 
v_add_co_u32 v22, s[18:19], v22, v58 // (op 2141) least significant half
v_addc_co_u32 v23, s[18:19], v23, v59, s[18:19] // (op 2141) most significant half




buffer_load_ubyte v24, v31, s[76:79], 0 offen offset: 0 // (op 518) Load value
v_add_u32 v31, v31, v20 // (op 518) 
buffer_load_ubyte v25, v31, s[76:79], 0 offen offset: 0 // (op 518) Load value



s_lshl_b32 s18, s38, 3 // (op 2401) 
s_barrier  // (op 3551) 
v_add_u32 v17, v17, s18 // (op 2401) 




s_waitcnt vmcnt(9)
ds_write_b8 v29, v32 offset:20480 // (op 190) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v29, v33 offset:20544 // (op 190) Store local data 
s_barrier  // (op 3538) 



ds_read_b128 v[34:37], v5 // (op 183) Load local data 
ds_read_b128 v[38:41], v5 offset:64 // (op 183) Load local data 
s_barrier  // (op 3519) 



ds_read_b64_tr_b8 v[42:43], v8 offset:2048 // (op 181) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v8 offset:2560 // (op 181) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v8 offset:6144 // (op 181) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v8 offset:6656 // (op 181) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 3558) 


ds_read_u8 v16, v9 offset:20480 // (op 185) Load local data 



ds_read_b64_tr_b8 v[50:51], v8 offset:2080 // (op 620) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v8 offset:2592 // (op 620) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v8 offset:6176 // (op 620) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v8 offset:6688 // (op 620) Transpose load local data 


ds_read_u8 v19, v9 offset:20512 // (op 624) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 590) 
s_mov_b32 s28, s18 // (op 590) convert

s_and_b32 s29, s28, 4294967294 // (op 590) 

s_cmp_lt_u32 s1, s29 // (op 590) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoop_590 // (op 590) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoop_590 if true)
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_590:












s_barrier  // (op 3689) 
v_and_b32 v12, -4, v1 // (op 2079) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2079) 
v_sub_u32 v17, v1, v12 // (op 2079) 

s_and_b32 s15, s18, 4294967294 // (op 2079) 
v_lshlrev_b32 v12, 4, v17 // (op 2079) 

s_add_u32 s6, s0, s15 // (op 2079) 

v_lshrrev_b32 v17, 2, v1 // (op 2079) 
s_mul_i32 s15, s16, s50 // (op 2079) 
s_add_u32 s15, s15, s6 // (op 2079) 

s_add_u32 s6, s15, s9 // (op 2079) 

s_mul_hi_u32 s15, s6, s13 // (op 2079) 
s_sub_u32 s7, s6, s15 // (op 2079) 
s_lshr_b32 s8, s7, 1 // (op 2079) 

s_add_u32 s7, s8, s15 // (op 2079) 


s_lshr_b32 s8, s7, s11 // (op 2079) 

s_and_b32 s7, s11, 2147483648 // (op 2079) 
s_cmp_eq_i32 s7, 0 // (op 2079) 
s_mov_b32 s15, scc // (op 2079) 

s_cmp_eq_i32 s15, 0 // (op 2079) 
s_cselect_b32 s7, s6, s8 // (op 2079) 


s_mul_hi_u32 s8, s7, s12 // (op 2079) 
s_sub_u32 s15, s7, s8 // (op 2079) 
s_lshr_b32 s28, s15, 1 // (op 2079) 

s_add_u32 s15, s28, s8 // (op 2079) 


s_lshr_b32 s8, s15, s10 // (op 2079) 

s_and_b32 s15, s10, 2147483648 // (op 2079) 
s_cmp_eq_i32 s15, 0 // (op 2079) 
s_mov_b32 s28, scc // (op 2079) 

s_cmp_eq_i32 s28, 0 // (op 2079) 
s_cselect_b32 s15, s7, s8 // (op 2079) 


s_mul_i32 s8, s7, s18 // (op 2079) 


v_lshl_add_u32 v20, s15, 6, v12 // (op 2079) 


s_sub_u32 s7, s6, s8 // (op 2079) 


v_mov_b32 v23, 0 // (op 2079) convert
v_mov_b32 v22, v20 // (op 2079) convert

v_lshl_add_u32 v12, s7, 7, v17 // (op 2079) 


v_mul_lo_u32 v21, v12, s31 // (op 2079) most significant: low of low * high
v_mul_hi_u32 v17, v12, s30 // (op 2079) most significant: high of low * low
v_mul_lo_u32 v20, v12, s30 // (op 2079) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 2079) most significant: sum


v_add_co_u32 v26, s[18:19], v22, v20 // (op 2079) least significant half
v_addc_co_u32 v27, s[18:19], v23, v21, s[18:19] // (op 2079) most significant half



s_mul_i32 s19, s30, s3 // (op 2081) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 2081) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 2081) least significant
s_mul_hi_u32 s6, s30, s2 // (op 2081) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2081) most significant: sum
s_add_u32 s19, s19, s6 // (op 2081) most significant: sum


s_mov_b32 s68, s32 // (op 2081) reg expression
s_mov_b32 s69, s33 // (op 2081) reg expression
s_mov_b32 s70, s18 // (op 2081) reg expression
s_mov_b32 s71, 131072 // (op 2081) call()

v_lshrrev_b32 v12, 3, v1 // (op 2217) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2217) 
v_and_b32 v17, -8, v1 // (op 2217) 
s_and_b32 s15, s18, 4294967294 // (op 2217) 
v_sub_u32 v20, v1, v17 // (op 2217) 

s_add_u32 s6, s0, s15 // (op 2217) 

v_lshlrev_b32 v17, 4, v20 // (op 2217) 

s_mul_i32 s15, s16, s50 // (op 2217) 
s_add_u32 s15, s15, s6 // (op 2217) 

s_add_u32 s6, s15, s9 // (op 2217) 

s_mul_hi_u32 s15, s6, s13 // (op 2217) 
s_sub_u32 s7, s6, s15 // (op 2217) 
s_lshr_b32 s8, s7, 1 // (op 2217) 

s_add_u32 s7, s8, s15 // (op 2217) 


s_lshr_b32 s8, s7, s11 // (op 2217) 

s_and_b32 s7, s11, 2147483648 // (op 2217) 
s_cmp_eq_i32 s7, 0 // (op 2217) 
s_mov_b32 s15, scc // (op 2217) 

s_cmp_eq_i32 s15, 0 // (op 2217) 
s_cselect_b32 s7, s6, s8 // (op 2217) 


s_mul_hi_u32 s8, s7, s12 // (op 2217) 
s_sub_u32 s15, s7, s8 // (op 2217) 
s_lshr_b32 s28, s15, 1 // (op 2217) 

s_add_u32 s15, s28, s8 // (op 2217) 


s_lshr_b32 s8, s15, s10 // (op 2217) 

s_and_b32 s15, s10, 2147483648 // (op 2217) 
s_cmp_eq_i32 s15, 0 // (op 2217) 
s_mov_b32 s28, scc // (op 2217) 

s_cmp_eq_i32 s28, 0 // (op 2217) 
s_cselect_b32 s15, s7, s8 // (op 2217) 


s_mul_i32 s8, s15, s14 // (op 2217) 

s_sub_u32 s15, s7, s8 // (op 2217) 

s_mul_i32 s8, s7, s18 // (op 2217) 


v_lshl_add_u32 v20, s15, 4, v12 // (op 2217) 


s_sub_u32 s7, s6, s8 // (op 2217) 


v_mul_lo_u32 v23, v20, s35 // (op 2217) most significant: low of low * high
v_mul_hi_u32 v12, v20, s34 // (op 2217) most significant: high of low * low
v_mul_lo_u32 v22, v20, s34 // (op 2217) least significant: low of low * low
v_add_u32 v23, v23, v12 // (op 2217) most significant: sum


v_lshl_add_u32 v12, s7, 7, v17 // (op 2217) 


v_mov_b32 v21, 0 // (op 2217) convert
v_mov_b32 v20, v12 // (op 2217) convert

v_add_co_u32 v28, s[18:19], v20, v22 // (op 2217) least significant half
v_addc_co_u32 v29, s[18:19], v21, v23, s[18:19] // (op 2217) most significant half



s_mul_i32 s19, s34, s23 // (op 2219) most significant: low of low * high
s_mul_i32 s15, s35, s22 // (op 2219) most significant: low of high * low
s_mul_i32 s18, s34, s22 // (op 2219) least significant
s_mul_hi_u32 s6, s34, s22 // (op 2219) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2219) most significant: sum
s_add_u32 s19, s19, s6 // (op 2219) most significant: sum


s_mov_b32 s72, s36 // (op 2219) reg expression
s_mov_b32 s73, s37 // (op 2219) reg expression
s_mov_b32 s74, s18 // (op 2219) reg expression
s_mov_b32 s75, 131072 // (op 2219) call()

v_and_b32 v12, -64, v1 // (op 2345) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2345) 
v_sub_u32 v17, v1, v12 // (op 2345) 

s_and_b32 s15, s18, 4294967294 // (op 2345) 
v_lshrrev_b32 v12, 6, v1 // (op 2345) 
s_add_u32 s6, s0, s15 // (op 2345) 

v_lshlrev_b32 v20, 1, v12 // (op 2345) 

s_mul_i32 s15, s16, s50 // (op 2345) 
s_add_u32 s15, s15, s6 // (op 2345) 

s_add_u32 s6, s15, s9 // (op 2345) 

s_mul_hi_u32 s15, s6, s13 // (op 2345) 
s_sub_u32 s7, s6, s15 // (op 2345) 
s_lshr_b32 s8, s7, 1 // (op 2345) 

s_add_u32 s7, s8, s15 // (op 2345) 


s_lshr_b32 s8, s7, s11 // (op 2345) 

s_and_b32 s7, s11, 2147483648 // (op 2345) 
s_cmp_eq_i32 s7, 0 // (op 2345) 
s_mov_b32 s15, scc // (op 2345) 

s_cmp_eq_i32 s15, 0 // (op 2345) 
s_cselect_b32 s7, s6, s8 // (op 2345) 


s_mul_hi_u32 s8, s7, s12 // (op 2345) 
s_sub_u32 s15, s7, s8 // (op 2345) 
s_lshr_b32 s28, s15, 1 // (op 2345) 

s_add_u32 s15, s28, s8 // (op 2345) 


s_lshr_b32 s8, s15, s10 // (op 2345) 

s_and_b32 s15, s10, 2147483648 // (op 2345) 
s_cmp_eq_i32 s15, 0 // (op 2345) 
s_mov_b32 s28, scc // (op 2345) 

s_cmp_eq_i32 s28, 0 // (op 2345) 
s_cselect_b32 s15, s7, s8 // (op 2345) 


s_mul_i32 s8, s7, s18 // (op 2345) 


v_lshl_add_u32 v12, s15, 6, v17 // (op 2345) 


s_sub_u32 s7, s6, s8 // (op 2345) 


v_lshl_add_u32 v17, s7, 2, v20 // (op 2345) 


v_mul_lo_u32 v20, v17, s38 // (op 2345) 

v_add_u32 v17, v12, v20 // (op 2345) 


s_mul_i32 s19, s38, s43 // (op 2347) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 2347) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 2347) least significant
s_mul_hi_u32 s6, s38, s42 // (op 2347) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2347) most significant: sum
s_add_u32 s19, s19, s6 // (op 2347) most significant: sum


s_mov_b32 s76, s40 // (op 2347) reg expression
s_mov_b32 s77, s41 // (op 2347) reg expression
s_mov_b32 s78, s18 // (op 2347) reg expression
s_mov_b32 s79, 131072 // (op 2347) call()

v_and_b32 v12, 15, v1 // (op 2473) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2473) 
v_and_b32 v20, 63, v1 // (op 2473) 
s_and_b32 s15, s18, 4294967294 // (op 2473) 
v_lshrrev_b32 v21, 4, v20 // (op 2473) 

s_add_u32 s6, s0, s15 // (op 2473) 

s_mul_i32 s15, s16, s50 // (op 2473) 
s_add_u32 s15, s15, s6 // (op 2473) 

s_add_u32 s6, s15, s9 // (op 2473) 

s_mul_hi_u32 s15, s6, s13 // (op 2473) 
s_sub_u32 s7, s6, s15 // (op 2473) 
s_lshr_b32 s8, s7, 1 // (op 2473) 

s_add_u32 s7, s8, s15 // (op 2473) 


s_lshr_b32 s8, s7, s11 // (op 2473) 

s_and_b32 s7, s11, 2147483648 // (op 2473) 
s_cmp_eq_i32 s7, 0 // (op 2473) 
s_mov_b32 s15, scc // (op 2473) 

s_cmp_eq_i32 s15, 0 // (op 2473) 
s_cselect_b32 s7, s6, s8 // (op 2473) 


s_mul_hi_u32 s8, s7, s12 // (op 2473) 
s_sub_u32 s15, s7, s8 // (op 2473) 
s_lshr_b32 s28, s15, 1 // (op 2473) 

s_add_u32 s15, s28, s8 // (op 2473) 


s_lshr_b32 s8, s15, s10 // (op 2473) 

s_and_b32 s15, s10, 2147483648 // (op 2473) 
s_cmp_eq_i32 s15, 0 // (op 2473) 
s_mov_b32 s28, scc // (op 2473) 

s_cmp_eq_i32 s28, 0 // (op 2473) 
s_cselect_b32 s15, s7, s8 // (op 2473) 


s_mul_i32 s8, s15, s14 // (op 2473) 

s_sub_u32 s15, s7, s8 // (op 2473) 

v_lshl_add_u32 v20, s15, 4, v12 // (op 2473) 


s_mul_i32 s8, s7, s18 // (op 2473) 


v_mul_lo_u32 v12, v20, s44 // (op 2473) 

s_sub_u32 s7, s6, s8 // (op 2473) 


v_lshl_add_u32 v20, s7, 2, v21 // (op 2473) 


v_add_u32 v21, v20, v12 // (op 2473) 


s_mul_i32 s19, s44, s23 // (op 2475) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 2475) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 2475) least significant
s_mul_hi_u32 s6, s44, s22 // (op 2475) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2475) most significant: sum
s_add_u32 s19, s19, s6 // (op 2475) most significant: sum


s_mov_b32 s80, s46 // (op 2475) reg expression
s_mov_b32 s81, s47 // (op 2475) reg expression
s_mov_b32 s82, s18 // (op 2475) reg expression
s_mov_b32 s83, 131072 // (op 2475) call()

v_and_b32 v12, -4, v1 // (op 2914) 
v_sub_u32 v20, v1, v12 // (op 2914) 

v_lshlrev_b32 v12, 4, v20 // (op 2914) 

v_lshrrev_b32 v20, 2, v1 // (op 2914) 
v_lshl_add_u32 v22, v20, 6, v12 // (op 2914) 


s_nop 0
v_readfirstlane_b32 s15, v22 // (op 2914) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -8, v1 // (op 2989) 
v_sub_u32 v20, v1, v12 // (op 2989) 

v_lshlrev_b32 v12, 4, v20 // (op 2989) 

v_lshrrev_b32 v20, 3, v1 // (op 2989) 
v_lshl_add_u32 v22, v20, 7, v12 // (op 2989) 


s_nop 0
v_readfirstlane_b32 s6, v22 // (op 2989) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -64, v1 // (op 3054) 
v_sub_u32 v20, v1, v12 // (op 3054) 

v_lshrrev_b32 v12, 6, v1 // (op 3054) 
v_lshl_add_u32 v22, v12, 7, v20 // (op 3054) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 596) 
s_and_b32 s7, s18, 4294967294 // (op 596) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 595) 
s_mov_b32 s8, s18 // (op 595) convert

s_cmp_lt_u32 s7, s8 // (op 595) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_595 // (op 595) Condition: Top (jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_595 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_595:
s_barrier  // (op 3516) 
s_add_u32 s7, s7, 1 // (op 597) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 1994) 
v_mov_b64 v[24:25], s[18:19] // (op 1994) convert


v_mov_b64 v[18:19], v[26:27] // (op 1994) 
v_mov_b64 v[30:31], v[18:19] // (op 1994) reg expression
s_add_u32 s8, 2048, s15 // (op 1994) 
s_mov_b32 m0, s8 // (op 1994) reg expression
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 1994) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v24 // (op 1994) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 1994) most significant half

s_add_u32 m0, m0, 2048 // (op 1994) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 1994) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v24 // (op 1994) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 1994) most significant half

s_add_u32 m0, m0, 2048 // (op 1994) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 1994) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v24 // (op 1994) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 1994) most significant half

s_add_u32 m0, m0, 2048 // (op 1994) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 1994) Load value direct to lds




s_lshl_b64 s[18:19], s[30:31], 7 // (op 2069) 
v_mov_b32 v12, s18 // (op 2069) 
v_mov_b32 v16, s19 // (op 2069) 
v_add_co_u32 v26, s[28:29], v26, v12 // (op 2069) least significant half
v_addc_co_u32 v27, s[28:29], v27, v16, s[28:29] // (op 2069) most significant half




s_barrier  // (op 3535) 


v_mov_b64 v[24:25], v[28:29] // (op 2000) 

s_mov_b32 m0, s6 // (op 2000) reg expression
buffer_load_dwordx4 v24, s[72:75], 0 offen offset: 0 lds // (op 2000) Load value direct to lds

v_add_co_u32 v28, vcc, 128, v28 // (op 2209) least significant half
v_addc_co_u32 v29, vcc, 0, v29, vcc // (op 2209) most significant half

v_mov_b64 v[24:25], s[38:39] // (op 460) convert

v_mov_b32 v12, v17 // (op 460) 
v_mov_b32 v16, v12 // (op 460) reg expression
buffer_load_ubyte v18, v16, s[76:79], 0 offen offset: 0 // (op 460) Load value
v_add_u32 v16, v16, v24 // (op 460) 
buffer_load_ubyte v19, v16, s[76:79], 0 offen offset: 0 // (op 460) Load value



s_lshl_b32 s8, s38, 2 // (op 2337) 
v_add_u32 v17, v17, s8 // (op 2337) 

s_barrier  // (op 3554) 
s_waitcnt vmcnt(3)
s_barrier  // (op 3522) 






ds_read_b64_tr_b8 v[30:31], v8 offset:2048 // (op 449) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v8 offset:2560 // (op 449) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v8 offset:6144 // (op 449) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v8 offset:6656 // (op 449) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v22, v18 offset:20480 // (op 458) Store local data 
s_barrier  // (op 3541) 
s_waitcnt vmcnt(0)
ds_write_b8 v22, v19 offset:20544 // (op 458) Store local data 




ds_read_b128 v[38:41], v5 // (op 451) Load local data 
ds_read_b128 v[42:45], v5 offset:64 // (op 451) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 3561) 


ds_read_u8 v12, v9 offset:20480 // (op 453) Load local data 


v_mov_b32 v16, v21 // (op 455) 

buffer_load_ubyte v18, v16, s[80:83], 0 offen offset: 0 // (op 455) Load value

v_add_u32 v21, 4, v21 // (op 2465) 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:45], a[0:3], v12, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 456) 
ds_read_b64_tr_b8 v[46:47], v8 offset:2080 // (op 601) Transpose load local data 


ds_read_b64_tr_b8 v[48:49], v8 offset:2592 // (op 601) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v8 offset:6176 // (op 601) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v8 offset:6688 // (op 601) Transpose load local data 


ds_read_u8 v12, v9 offset:20512 // (op 605) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[38:45], a[4:7], v12, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 607) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 595) 
s_mov_b32 s8, s18 // (op 595) convert

s_cmp_lt_u32 s7, s8 // (op 595) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_595 // (op 595) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_595 if true)
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_595:








s_barrier  // (op 3693) 



v_cmp_eq_f32 vcc, 0.00000, v14 // (op 222) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_222 // (op 222) Condition: False, jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_222
v_accvgpr_read v16, a0 // (op 230) 
v_accvgpr_read v17, a1 // (op 230) 
v_accvgpr_read v18, a2 // (op 230) 
v_accvgpr_read v19, a3 // (op 230) 
v_mul_f32 v20, v15, v16 // (op 230) 
v_mul_f32 v21, v15, v17 // (op 230) 
v_mul_f32 v22, v15, v18 // (op 230) 
v_mul_f32 v23, v15, v19 // (op 230) 

v_cvt_f16_f32 v16, v20 // (op 232) 
v_cvt_f16_f32 v17, v21 // (op 232) 
v_cvt_f16_f32 v18, v22 // (op 232) 
v_cvt_f16_f32 v19, v23 // (op 232) 


v_pack_b32_f16 v12, v16, v17 // (op 233) 
v_pack_b32_f16 v13, v18, v19 // (op 233) 


buffer_store_dwordx2 v[12:13], v4, s[60:63], 0 offen // (op 233) Store value


v_accvgpr_read v16, a4 // (op 793) 
v_accvgpr_read v17, a5 // (op 793) 
v_accvgpr_read v18, a6 // (op 793) 
v_accvgpr_read v19, a7 // (op 793) 
v_mul_f32 v20, v15, v16 // (op 793) 
v_mul_f32 v21, v15, v17 // (op 793) 
v_mul_f32 v22, v15, v18 // (op 793) 
v_mul_f32 v23, v15, v19 // (op 793) 

v_cvt_f16_f32 v16, v20 // (op 795) 
v_cvt_f16_f32 v17, v21 // (op 795) 
v_cvt_f16_f32 v18, v22 // (op 795) 
v_cvt_f16_f32 v19, v23 // (op 795) 


v_pack_b32_f16 v12, v16, v17 // (op 796) 
v_pack_b32_f16 v13, v18, v19 // (op 796) 


buffer_store_dwordx2 v[12:13], v4, s[60:63], 0 offen offset:64 // (op 796) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_222 // (op 222) Condition: Done, jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_222
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_222:



buffer_load_dwordx2 v[12:13], v3, s[56:59], 0 offen offset:64 // (op 826) Load value
v_accvgpr_read v16, a4 // (op 822) 
v_accvgpr_read v17, a5 // (op 822) 
v_accvgpr_read v18, a6 // (op 822) 
v_accvgpr_read v19, a7 // (op 822) 
v_mul_f32 v20, v15, v16 // (op 822) 
v_mul_f32 v21, v15, v17 // (op 822) 
v_mul_f32 v22, v15, v18 // (op 822) 
v_mul_f32 v23, v15, v19 // (op 822) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v12 // (op 827) 
v_lshrrev_b32 v17, 16, v12 // (op 827) 
v_cvt_f32_f16 v17, v17 // (op 827) 
v_mul_f32 v24, v14, v16 // (op 827) 
v_mul_f32 v25, v14, v17 // (op 827) 
v_cvt_f32_f16 v16, v13 // (op 827) 
v_lshrrev_b32 v17, 16, v13 // (op 827) 
v_cvt_f32_f16 v17, v17 // (op 827) 
v_mul_f32 v26, v14, v16 // (op 827) 
v_mul_f32 v27, v14, v17 // (op 827) 


v_add_f32 v16, v24, v20 // (op 823) 
v_add_f32 v17, v25, v21 // (op 823) 
v_add_f32 v18, v26, v22 // (op 823) 
v_add_f32 v19, v27, v23 // (op 823) 


v_cvt_f16_f32 v20, v16 // (op 824) 
v_cvt_f16_f32 v21, v17 // (op 824) 
v_cvt_f16_f32 v22, v18 // (op 824) 
v_cvt_f16_f32 v23, v19 // (op 824) 


v_pack_b32_f16 v12, v20, v21 // (op 825) 
v_pack_b32_f16 v13, v22, v23 // (op 825) 


buffer_store_dwordx2 v[12:13], v4, s[60:63], 0 offen offset:64 // (op 825) Store value





buffer_load_dwordx2 v[12:13], v3, s[56:59], 0 offen // (op 247) Load value
v_accvgpr_read v16, a0 // (op 243) 
v_accvgpr_read v17, a1 // (op 243) 
v_accvgpr_read v18, a2 // (op 243) 
v_accvgpr_read v19, a3 // (op 243) 
v_mul_f32 v20, v15, v16 // (op 243) 
v_mul_f32 v21, v15, v17 // (op 243) 
v_mul_f32 v22, v15, v18 // (op 243) 
v_mul_f32 v23, v15, v19 // (op 243) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v12 // (op 248) 
v_lshrrev_b32 v17, 16, v12 // (op 248) 
v_cvt_f32_f16 v17, v17 // (op 248) 
v_mul_f32 v24, v14, v16 // (op 248) 
v_mul_f32 v25, v14, v17 // (op 248) 
v_cvt_f32_f16 v16, v13 // (op 248) 
v_lshrrev_b32 v17, 16, v13 // (op 248) 
v_cvt_f32_f16 v17, v17 // (op 248) 
v_mul_f32 v26, v14, v16 // (op 248) 
v_mul_f32 v27, v14, v17 // (op 248) 


v_add_f32 v16, v24, v20 // (op 244) 
v_add_f32 v17, v25, v21 // (op 244) 
v_add_f32 v18, v26, v22 // (op 244) 
v_add_f32 v19, v27, v23 // (op 244) 


v_cvt_f16_f32 v20, v16 // (op 245) 
v_cvt_f16_f32 v21, v17 // (op 245) 
v_cvt_f16_f32 v22, v18 // (op 245) 
v_cvt_f16_f32 v23, v19 // (op 245) 


v_pack_b32_f16 v12, v20, v21 // (op 246) 
v_pack_b32_f16 v13, v22, v23 // (op 246) 


buffer_store_dwordx2 v[12:13], v4, s[60:63], 0 offen // (op 246) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_222:
s_add_u32 s0, s0, s1 // (op 435) 


s_mul_i32 s15, s50, s16 // (op 433) 
s_cmp_gt_u32 s51, s15 // (op 433) 
s_mov_b32 s6, scc // (op 433) 
s_sub_u32 s7, s51, s15 // (op 433) 

s_cmp_eq_i32 s6, 0 // (op 433) 
s_cselect_b32 s8, 0, s7 // (op 433) 


s_cmp_lt_u32 s50, s8 // (op 433) 
s_mov_b32 s15, scc // (op 433) 
s_cmp_eq_i32 s15, 0 // (op 433) 
s_cselect_b32 s6, s8, s50 // (op 433) 


s_cmp_lt_u32 s0, s6 // (op 433) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_DPStreamTileLoop_433 // (op 433) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_DPStreamTileLoop_433 if true)
RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_433:









































s_endpgm  // End of RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2
.LRR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_end:
.size RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2, .LRR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_end-RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2
  .amdhsa_next_free_vgpr 84
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 20992 // lds bytes
  .amdhsa_accum_offset 76
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
    macrotile: [64, 16, 128]
    threads: [128, 1, 1]
    grid: [RocRollerStreamK, One, One]
    occupancy: 7
    workspaceType: RocRollerStreamK
    workspaceSizePerElemC: 4
    workspaceSizePerElemBias: 0
    unroll: 2
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2
    .symbol: RR_GEMM_NN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 20992
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 89
    .vgpr_count: 74
    .agpr_count: 8
    .max_flat_workgroup_size: 128
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

