// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2,@function
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 9573) 
v_and_b32 v3, 15, v1 // (op 9573) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 9573) 


v_and_b32 v0, 48, v1 // (op 9573) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 9573) 


v_lshrrev_b32 v0, 6, v1 // (op 9648) 
v_and_b32 v4, 1, v0 // (op 9648) 

v_and_b32 v0, 15, v1 // (op 9648) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 9648) 


v_and_b32 v0, 48, v1 // (op 9648) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 9648) 


v_lshrrev_b32 v0, 7, v1 // (op 9724) 
v_and_b32 v5, 15, v1 // (op 9724) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 9724) 


v_and_b32 v0, 63, v1 // (op 9724) 
v_lshrrev_b32 v5, 4, v0 // (op 9724) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 9724) 


v_lshrrev_b32 v5, 6, v1 // (op 9800) 
v_and_b32 v6, 1, v5 // (op 9800) 

v_and_b32 v5, 15, v1 // (op 9800) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 9800) 


v_and_b32 v5, 63, v1 // (op 9800) 
v_lshrrev_b32 v6, 4, v5 // (op 9800) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 9800) 


v_lshrrev_b32 v6, 7, v1 // (op 9927) 
v_and_b32 v7, 15, v1 // (op 9927) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 9927) 


v_and_b32 v6, 48, v1 // (op 9927) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 9927) 


v_lshrrev_b32 v6, 6, v1 // (op 10002) 
v_and_b32 v8, 1, v6 // (op 10002) 

v_and_b32 v6, 15, v1 // (op 10002) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 10002) 


v_and_b32 v6, 48, v1 // (op 10002) 
v_lshl_add_u32 v8, v9, 7, v6 // (op 10002) 


v_lshrrev_b32 v6, 7, v1 // (op 10078) 
v_and_b32 v9, 15, v1 // (op 10078) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 10078) 


v_and_b32 v6, 63, v1 // (op 10078) 
v_lshrrev_b32 v9, 4, v6 // (op 10078) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 10078) 


v_lshrrev_b32 v9, 6, v1 // (op 10154) 
v_and_b32 v10, 1, v9 // (op 10154) 

v_and_b32 v9, 15, v1 // (op 10154) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 10154) 


v_and_b32 v9, 63, v1 // (op 10154) 
v_lshrrev_b32 v10, 4, v9 // (op 10154) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 10154) 


v_lshrrev_b32 v10, 7, v1 // (op 10268) 
v_and_b32 v11, 15, v1 // (op 10268) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 10268) 


v_and_b32 v10, 48, v1 // (op 10268) 
v_lshl_add_u32 v11, v12, 7, v10 // (op 10268) 


v_lshrrev_b32 v10, 6, v1 // (op 10312) 
v_and_b32 v12, 1, v10 // (op 10312) 

v_and_b32 v10, 15, v1 // (op 10312) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 10312) 


v_and_b32 v10, 48, v1 // (op 10312) 
v_lshl_add_u32 v12, v13, 7, v10 // (op 10312) 


v_lshrrev_b32 v10, 7, v1 // (op 10341) 
v_and_b32 v13, 15, v1 // (op 10341) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 10341) 


v_and_b32 v10, 63, v1 // (op 10341) 
v_lshrrev_b32 v13, 4, v10 // (op 10341) 

v_lshl_add_u32 v10, v14, 2, v13 // (op 10341) 


v_lshrrev_b32 v13, 6, v1 // (op 10370) 
v_and_b32 v14, 1, v13 // (op 10370) 

v_and_b32 v13, 15, v1 // (op 10370) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 10370) 


v_and_b32 v13, 63, v1 // (op 10370) 
v_lshrrev_b32 v14, 4, v13 // (op 10370) 

v_lshl_add_u32 v13, v15, 2, v14 // (op 10370) 


v_lshrrev_b32 v14, 7, v1 // (op 10414) 
v_and_b32 v15, 15, v1 // (op 10414) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 10414) 


v_and_b32 v14, 48, v1 // (op 10414) 
v_lshl_add_u32 v15, v16, 7, v14 // (op 10414) 


v_lshrrev_b32 v14, 6, v1 // (op 10458) 
v_and_b32 v16, 1, v14 // (op 10458) 

v_and_b32 v14, 15, v1 // (op 10458) 
v_lshl_add_u32 v17, v16, 4, v14 // (op 10458) 


v_and_b32 v14, 48, v1 // (op 10458) 
v_lshl_add_u32 v16, v17, 7, v14 // (op 10458) 


v_lshrrev_b32 v14, 7, v1 // (op 10487) 
v_and_b32 v17, 15, v1 // (op 10487) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 10487) 


v_and_b32 v14, 63, v1 // (op 10487) 
v_lshrrev_b32 v17, 4, v14 // (op 10487) 

v_lshl_add_u32 v14, v18, 2, v17 // (op 10487) 


v_lshrrev_b32 v17, 6, v1 // (op 10516) 
v_and_b32 v18, 1, v17 // (op 10516) 

v_and_b32 v17, 15, v1 // (op 10516) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 10516) 


v_and_b32 v17, 63, v1 // (op 10516) 
v_lshrrev_b32 v18, 4, v17 // (op 10516) 

v_lshl_add_u32 v17, v19, 2, v18 // (op 10516) 


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
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_286 // (op 286) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_286 if false)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_SKStreamTileLoop_286:
v_lshrrev_b32 v20, 7, v1 // (op 9484) 
s_mul_i32 s52, s16, s8 // (op 9484) 
s_add_u32 s52, s52, s0 // (op 9484) 
v_and_b32 v21, 63, v1 // (op 9484) 
s_mul_hi_u32 s53, s52, s13 // (op 9484) 
v_lshrrev_b32 v22, 4, v21 // (op 9484) 
s_sub_u32 s54, s52, s53 // (op 9484) 
v_lshlrev_b32 v23, 2, v22 // (op 9484) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 9484) 


s_lshr_b32 s55, s54, 1 // (op 9484) 

s_add_u32 s54, s55, s53 // (op 9484) 


v_lshrrev_b32 v20, 2, v21 // (op 9484) 

s_lshr_b32 s53, s54, s11 // (op 9484) 

v_and_b32 v21, 3, v20 // (op 9484) 

s_and_b32 s54, s11, 2147483648 // (op 9484) 
v_and_b32 v20, 3, v1 // (op 9484) 
s_cmp_eq_i32 s54, 0 // (op 9484) 
s_mov_b32 s55, scc // (op 9484) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 9484) 


s_cmp_eq_i32 s55, 0 // (op 9484) 
s_cselect_b32 s54, s52, s53 // (op 9484) 



v_lshrrev_b32 v20, 6, v1 // (op 9484) 
s_mul_hi_u32 s52, s54, s12 // (op 9484) 
v_and_b32 v21, 1, v20 // (op 9484) 

s_sub_u32 s53, s54, s52 // (op 9484) 
v_lshl_add_u32 v20, v21, 4, v23 // (op 9484) 


s_lshr_b32 s55, s53, 1 // (op 9484) 

s_add_u32 s53, s55, s52 // (op 9484) 


s_lshr_b32 s52, s53, s10 // (op 9484) 

s_and_b32 s53, s10, 2147483648 // (op 9484) 
s_cmp_eq_i32 s53, 0 // (op 9484) 
s_mov_b32 s55, scc // (op 9484) 

s_cmp_eq_i32 s55, 0 // (op 9484) 
s_cselect_b32 s53, s54, s52 // (op 9484) 


s_mul_i32 s52, s53, s14 // (op 9484) 
v_lshl_add_u32 v21, s53, 7, v22 // (op 9484) 


s_sub_u32 s53, s54, s52 // (op 9484) 


v_lshl_add_u32 v22, s53, 6, v20 // (op 9484) 


v_mul_lo_u32 v20, v22, s4 // (op 9484) 

v_add_lshl_u32 v22, v21, v20, 2 // (op 9484) 


s_mul_i32 s53, s4, s23 // (op 9486) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 9486) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 9486) least significant
s_mul_hi_u32 s55, s4, s22 // (op 9486) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9486) most significant: sum
s_add_u32 s53, s53, s55 // (op 9486) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 9486) 

s_mov_b32 s56, s20 // (op 9486) reg expression
s_mov_b32 s57, s21 // (op 9486) reg expression
s_mov_b32 s58, s54 // (op 9486) reg expression
s_mov_b32 s59, 131072 // (op 9486) call()

v_lshrrev_b32 v20, 7, v1 // (op 9590) 
v_and_b32 v21, 15, v1 // (op 9590) 
v_lshl_add_u32 v23, v20, 4, v21 // (op 9590) 


v_and_b32 v20, 48, v1 // (op 9590) 
v_lshl_add_u32 v21, v23, 7, v20 // (op 9590) 


v_lshrrev_b32 v20, 6, v1 // (op 9665) 
v_and_b32 v23, 1, v20 // (op 9665) 

v_and_b32 v20, 15, v1 // (op 9665) 
v_lshl_add_u32 v24, v23, 4, v20 // (op 9665) 


v_and_b32 v20, 48, v1 // (op 9665) 
v_lshl_add_u32 v23, v24, 7, v20 // (op 9665) 


v_lshrrev_b32 v20, 7, v1 // (op 9741) 
v_and_b32 v24, 15, v1 // (op 9741) 
v_lshl_add_u32 v25, v20, 4, v24 // (op 9741) 


v_and_b32 v20, 63, v1 // (op 9741) 
v_lshrrev_b32 v24, 4, v20 // (op 9741) 

v_lshl_add_u32 v20, v25, 2, v24 // (op 9741) 


v_lshrrev_b32 v24, 6, v1 // (op 9817) 
v_and_b32 v25, 1, v24 // (op 9817) 

v_and_b32 v24, 15, v1 // (op 9817) 
v_lshl_add_u32 v26, v25, 4, v24 // (op 9817) 


v_and_b32 v24, 63, v1 // (op 9817) 
v_lshrrev_b32 v25, 4, v24 // (op 9817) 

v_lshl_add_u32 v24, v26, 2, v25 // (op 9817) 


v_lshrrev_b32 v25, 7, v1 // (op 9840) 
s_mul_i32 s52, s16, s8 // (op 9840) 
s_add_u32 s52, s52, s0 // (op 9840) 
v_and_b32 v26, 63, v1 // (op 9840) 
s_mul_hi_u32 s53, s52, s13 // (op 9840) 
v_lshrrev_b32 v27, 4, v26 // (op 9840) 
s_sub_u32 s54, s52, s53 // (op 9840) 
v_lshlrev_b32 v28, 2, v27 // (op 9840) 

s_lshr_b32 s55, s54, 1 // (op 9840) 

v_lshl_add_u32 v27, v25, 4, v28 // (op 9840) 


s_add_u32 s54, s55, s53 // (op 9840) 


v_lshrrev_b32 v25, 2, v26 // (op 9840) 

s_lshr_b32 s53, s54, s11 // (op 9840) 

v_and_b32 v26, 3, v25 // (op 9840) 

s_and_b32 s54, s11, 2147483648 // (op 9840) 
v_and_b32 v25, 3, v1 // (op 9840) 
s_cmp_eq_i32 s54, 0 // (op 9840) 
s_mov_b32 s55, scc // (op 9840) 

v_lshl_add_u32 v28, v26, 2, v25 // (op 9840) 


s_cmp_eq_i32 s55, 0 // (op 9840) 
s_cselect_b32 s54, s52, s53 // (op 9840) 



v_lshrrev_b32 v25, 6, v1 // (op 9840) 
s_mul_hi_u32 s55, s54, s12 // (op 9840) 
v_and_b32 v26, 1, v25 // (op 9840) 

s_sub_u32 s52, s54, s55 // (op 9840) 
v_lshl_add_u32 v25, v26, 4, v28 // (op 9840) 


s_lshr_b32 s53, s52, 1 // (op 9840) 

s_add_u32 s52, s53, s55 // (op 9840) 


s_lshr_b32 s53, s52, s10 // (op 9840) 

s_and_b32 s52, s10, 2147483648 // (op 9840) 
s_cmp_eq_i32 s52, 0 // (op 9840) 
s_mov_b32 s55, scc // (op 9840) 

s_cmp_eq_i32 s55, 0 // (op 9840) 
s_cselect_b32 s52, s54, s53 // (op 9840) 


s_mul_i32 s53, s52, s14 // (op 9840) 
v_lshl_add_u32 v26, s52, 7, v27 // (op 9840) 


s_sub_u32 s52, s54, s53 // (op 9840) 


v_lshl_add_u32 v27, s52, 6, v25 // (op 9840) 


v_mul_lo_u32 v25, v27, s24 // (op 9840) 

v_add_lshl_u32 v27, v26, v25, 2 // (op 9840) 


s_mul_i32 s53, s24, s23 // (op 9842) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 9842) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 9842) least significant
s_mul_hi_u32 s55, s24, s22 // (op 9842) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9842) most significant: sum
s_add_u32 s53, s53, s55 // (op 9842) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 9842) 

s_mov_b32 s60, s26 // (op 9842) reg expression
s_mov_b32 s61, s27 // (op 9842) reg expression
s_mov_b32 s62, s54 // (op 9842) reg expression
s_mov_b32 s63, 131072 // (op 9842) call()

v_lshrrev_b32 v25, 5, v1 // (op 10194) 
v_lshlrev_b32 v26, 2, v25 // (op 10194) 

v_lshl_add_u32 v25, s16, 7, v26 // (op 10194) 

v_and_b32 v26, -32, v1 // (op 10194) 
v_sub_u32 v28, v1, v26 // (op 10194) 

v_lshl_add_u32 v26, v25, 6, v28 // (op 10194) 


v_lshlrev_b32 v25, 2, v26 // (op 10194) 

s_lshl_b32 s52, s15, 15 // (op 10196) 
s_mov_b32 s64, s28 // (op 10196) reg expression
s_mov_b32 s65, s29 // (op 10196) reg expression
s_mov_b32 s66, s52 // (op 10196) reg expression
s_mov_b32 s67, 131072 // (op 10196) call()

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
v_accvgpr_write a4, 0.00000 // (op 2682) call()
v_accvgpr_write a5, 0.00000 // (op 2682) call()
v_accvgpr_write a6, 0.00000 // (op 2682) call()
v_accvgpr_write a7, 0.00000 // (op 2682) call()
v_accvgpr_write a8, 0.00000 // (op 2902) call()
v_accvgpr_write a9, 0.00000 // (op 2902) call()
v_accvgpr_write a10, 0.00000 // (op 2902) call()
v_accvgpr_write a11, 0.00000 // (op 2902) call()
v_accvgpr_write a12, 0.00000 // (op 3057) call()
v_accvgpr_write a13, 0.00000 // (op 3057) call()
v_accvgpr_write a14, 0.00000 // (op 3057) call()
v_accvgpr_write a15, 0.00000 // (op 3057) call()
v_accvgpr_write a16, 0.00000 // (op 3288) call()
v_accvgpr_write a17, 0.00000 // (op 3288) call()
v_accvgpr_write a18, 0.00000 // (op 3288) call()
v_accvgpr_write a19, 0.00000 // (op 3288) call()
v_accvgpr_write a20, 0.00000 // (op 3443) call()
v_accvgpr_write a21, 0.00000 // (op 3443) call()
v_accvgpr_write a22, 0.00000 // (op 3443) call()
v_accvgpr_write a23, 0.00000 // (op 3443) call()
v_accvgpr_write a24, 0.00000 // (op 3674) call()
v_accvgpr_write a25, 0.00000 // (op 3674) call()
v_accvgpr_write a26, 0.00000 // (op 3674) call()
v_accvgpr_write a27, 0.00000 // (op 3674) call()
v_accvgpr_write a28, 0.00000 // (op 3829) call()
v_accvgpr_write a29, 0.00000 // (op 3829) call()
v_accvgpr_write a30, 0.00000 // (op 3829) call()
v_accvgpr_write a31, 0.00000 // (op 3829) call()
v_lshrrev_b32 v26, 3, v1 // (op 9051) 
s_mul_i32 s52, s16, s8 // (op 9051) 
s_add_u32 s52, s52, s0 // (op 9051) 
v_and_b32 v28, -8, v1 // (op 9051) 
s_mul_hi_u32 s53, s52, s13 // (op 9051) 
v_sub_u32 v29, v1, v28 // (op 9051) 

s_sub_u32 s54, s52, s53 // (op 9051) 
v_lshlrev_b32 v28, 4, v29 // (op 9051) 

s_lshr_b32 s55, s54, 1 // (op 9051) 

s_add_u32 s54, s55, s53 // (op 9051) 


s_lshr_b32 s53, s54, s11 // (op 9051) 

s_and_b32 s54, s11, 2147483648 // (op 9051) 
s_cmp_eq_i32 s54, 0 // (op 9051) 
s_mov_b32 s55, scc // (op 9051) 

s_cmp_eq_i32 s55, 0 // (op 9051) 
s_cselect_b32 s54, s52, s53 // (op 9051) 


s_mul_hi_u32 s53, s54, s12 // (op 9051) 
s_sub_u32 s55, s54, s53 // (op 9051) 
s_lshr_b32 s68, s55, 1 // (op 9051) 

s_add_u32 s55, s68, s53 // (op 9051) 


s_lshr_b32 s53, s55, s10 // (op 9051) 

s_and_b32 s55, s10, 2147483648 // (op 9051) 
s_cmp_eq_i32 s55, 0 // (op 9051) 
s_mov_b32 s68, scc // (op 9051) 

s_cmp_eq_i32 s68, 0 // (op 9051) 
s_cselect_b32 s55, s54, s53 // (op 9051) 


v_lshl_add_u32 v29, s55, 7, v26 // (op 9051) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 9051) 
s_mul_i32 s53, s54, s68 // (op 9051) 


s_sub_u32 s54, s52, s53 // (op 9051) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 9051) 


v_mov_b32 v31, 0 // (op 9051) convert
v_mov_b32 v30, v26 // (op 9051) convert

v_mul_lo_u32 v33, v29, s31 // (op 9051) most significant: low of low * high
v_mul_hi_u32 v26, v29, s30 // (op 9051) most significant: high of low * low
v_mul_lo_u32 v32, v29, s30 // (op 9051) least significant: low of low * low
v_add_u32 v33, v33, v26 // (op 9051) most significant: sum

v_add_co_u32 v32, s[52:53], v32, v30 // (op 9051) least significant half
v_addc_co_u32 v33, s[52:53], v33, v31, s[52:53] // (op 9051) most significant half



s_mul_i32 s53, s30, s35 // (op 9053) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 9053) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 9053) least significant
s_mul_hi_u32 s55, s30, s34 // (op 9053) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9053) most significant: sum
s_add_u32 s53, s53, s55 // (op 9053) most significant: sum


s_mov_b32 s68, s32 // (op 9053) reg expression
s_mov_b32 s69, s33 // (op 9053) reg expression
s_mov_b32 s70, s52 // (op 9053) reg expression
s_mov_b32 s71, 131072 // (op 9053) call()

v_lshrrev_b32 v26, 3, v1 // (op 9070) 
s_mul_i32 s52, s16, s8 // (op 9070) 
s_add_u32 s52, s52, s0 // (op 9070) 
v_and_b32 v28, -8, v1 // (op 9070) 
s_mul_hi_u32 s53, s52, s13 // (op 9070) 
v_sub_u32 v29, v1, v28 // (op 9070) 

s_sub_u32 s54, s52, s53 // (op 9070) 
v_lshlrev_b32 v28, 4, v29 // (op 9070) 

s_lshr_b32 s55, s54, 1 // (op 9070) 

s_add_u32 s54, s55, s53 // (op 9070) 


s_lshr_b32 s53, s54, s11 // (op 9070) 

s_and_b32 s54, s11, 2147483648 // (op 9070) 
s_cmp_eq_i32 s54, 0 // (op 9070) 
s_mov_b32 s55, scc // (op 9070) 

s_cmp_eq_i32 s55, 0 // (op 9070) 
s_cselect_b32 s54, s52, s53 // (op 9070) 


s_mul_hi_u32 s53, s54, s12 // (op 9070) 
s_sub_u32 s55, s54, s53 // (op 9070) 
s_lshr_b32 s72, s55, 1 // (op 9070) 

s_add_u32 s55, s72, s53 // (op 9070) 


s_lshr_b32 s53, s55, s10 // (op 9070) 

s_and_b32 s55, s10, 2147483648 // (op 9070) 
s_cmp_eq_i32 s55, 0 // (op 9070) 
s_mov_b32 s72, scc // (op 9070) 

s_cmp_eq_i32 s72, 0 // (op 9070) 
s_cselect_b32 s55, s54, s53 // (op 9070) 


v_lshl_add_u32 v29, s55, 7, v26 // (op 9070) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 9070) 
s_mul_i32 s53, s54, s72 // (op 9070) 


s_sub_u32 s54, s52, s53 // (op 9070) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 9070) 


v_mov_b32 v31, 0 // (op 9070) convert
v_mov_b32 v30, v26 // (op 9070) convert

v_mul_lo_u32 v35, v29, s31 // (op 9070) most significant: low of low * high
v_mul_hi_u32 v26, v29, s30 // (op 9070) most significant: high of low * low
v_mul_lo_u32 v34, v29, s30 // (op 9070) least significant: low of low * low
v_add_u32 v35, v35, v26 // (op 9070) most significant: sum

v_add_co_u32 v34, s[52:53], v34, v30 // (op 9070) least significant half
v_addc_co_u32 v35, s[52:53], v35, v31, s[52:53] // (op 9070) most significant half



s_mul_i32 s53, s30, s35 // (op 9072) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 9072) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 9072) least significant
s_mul_hi_u32 s55, s30, s34 // (op 9072) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9072) most significant: sum
s_add_u32 s53, s53, s55 // (op 9072) most significant: sum


s_mov_b32 s68, s32 // (op 9072) reg expression
s_mov_b32 s69, s33 // (op 9072) reg expression
s_mov_b32 s70, s52 // (op 9072) reg expression
s_mov_b32 s71, 131072 // (op 9072) call()

v_lshrrev_b32 v26, 3, v1 // (op 9179) 
s_mul_i32 s52, s16, s8 // (op 9179) 
s_add_u32 s52, s52, s0 // (op 9179) 
v_and_b32 v28, -8, v1 // (op 9179) 
s_mul_hi_u32 s53, s52, s13 // (op 9179) 
v_sub_u32 v29, v1, v28 // (op 9179) 

s_sub_u32 s54, s52, s53 // (op 9179) 
v_lshlrev_b32 v28, 4, v29 // (op 9179) 

s_lshr_b32 s55, s54, 1 // (op 9179) 

s_add_u32 s54, s55, s53 // (op 9179) 


s_lshr_b32 s53, s54, s11 // (op 9179) 

s_and_b32 s54, s11, 2147483648 // (op 9179) 
s_cmp_eq_i32 s54, 0 // (op 9179) 
s_mov_b32 s55, scc // (op 9179) 

s_cmp_eq_i32 s55, 0 // (op 9179) 
s_cselect_b32 s54, s52, s53 // (op 9179) 


s_mul_hi_u32 s53, s54, s12 // (op 9179) 
s_sub_u32 s55, s54, s53 // (op 9179) 
s_lshr_b32 s72, s55, 1 // (op 9179) 

s_add_u32 s55, s72, s53 // (op 9179) 


s_lshr_b32 s53, s55, s10 // (op 9179) 

s_and_b32 s55, s10, 2147483648 // (op 9179) 
s_cmp_eq_i32 s55, 0 // (op 9179) 
s_mov_b32 s72, scc // (op 9179) 

s_cmp_eq_i32 s72, 0 // (op 9179) 
s_cselect_b32 s55, s54, s53 // (op 9179) 


s_mul_i32 s53, s55, s14 // (op 9179) 

s_sub_u32 s55, s54, s53 // (op 9179) 

v_lshl_add_u32 v29, s55, 6, v26 // (op 9179) 


v_mul_lo_u32 v31, v29, s37 // (op 9179) most significant: low of low * high
v_mul_hi_u32 v26, v29, s36 // (op 9179) most significant: high of low * low
v_mul_lo_u32 v30, v29, s36 // (op 9179) least significant: low of low * low
v_add_u32 v31, v31, v26 // (op 9179) most significant: sum


s_ashr_i64 s[72:73], s[2:3], 7 // (op 9179) 
s_mul_i32 s53, s54, s72 // (op 9179) 


s_sub_u32 s54, s52, s53 // (op 9179) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 9179) 


v_mov_b32 v29, 0 // (op 9179) convert
v_mov_b32 v28, v26 // (op 9179) convert

v_add_co_u32 v36, s[52:53], v28, v30 // (op 9179) least significant half
v_addc_co_u32 v37, s[52:53], v29, v31, s[52:53] // (op 9179) most significant half



s_mul_i32 s53, s36, s23 // (op 9181) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 9181) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 9181) least significant
s_mul_hi_u32 s55, s36, s22 // (op 9181) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9181) most significant: sum
s_add_u32 s53, s53, s55 // (op 9181) most significant: sum


s_mov_b32 s72, s38 // (op 9181) reg expression
s_mov_b32 s73, s39 // (op 9181) reg expression
s_mov_b32 s74, s52 // (op 9181) reg expression
s_mov_b32 s75, 131072 // (op 9181) call()

v_lshrrev_b32 v26, 3, v1 // (op 9198) 
s_mul_i32 s52, s16, s8 // (op 9198) 
s_add_u32 s52, s52, s0 // (op 9198) 
v_and_b32 v28, -8, v1 // (op 9198) 
s_mul_hi_u32 s53, s52, s13 // (op 9198) 
v_sub_u32 v29, v1, v28 // (op 9198) 

s_sub_u32 s54, s52, s53 // (op 9198) 
v_lshlrev_b32 v28, 4, v29 // (op 9198) 

s_lshr_b32 s55, s54, 1 // (op 9198) 

s_add_u32 s54, s55, s53 // (op 9198) 


s_lshr_b32 s53, s54, s11 // (op 9198) 

s_and_b32 s54, s11, 2147483648 // (op 9198) 
s_cmp_eq_i32 s54, 0 // (op 9198) 
s_mov_b32 s55, scc // (op 9198) 

s_cmp_eq_i32 s55, 0 // (op 9198) 
s_cselect_b32 s54, s52, s53 // (op 9198) 


s_mul_hi_u32 s53, s54, s12 // (op 9198) 
s_sub_u32 s55, s54, s53 // (op 9198) 
s_lshr_b32 s76, s55, 1 // (op 9198) 

s_add_u32 s55, s76, s53 // (op 9198) 


s_lshr_b32 s53, s55, s10 // (op 9198) 

s_and_b32 s55, s10, 2147483648 // (op 9198) 
s_cmp_eq_i32 s55, 0 // (op 9198) 
s_mov_b32 s76, scc // (op 9198) 

s_cmp_eq_i32 s76, 0 // (op 9198) 
s_cselect_b32 s55, s54, s53 // (op 9198) 


s_mul_i32 s53, s55, s14 // (op 9198) 

s_sub_u32 s55, s54, s53 // (op 9198) 

v_lshl_add_u32 v29, s55, 6, v26 // (op 9198) 


v_mul_lo_u32 v31, v29, s37 // (op 9198) most significant: low of low * high
v_mul_hi_u32 v26, v29, s36 // (op 9198) most significant: high of low * low
v_mul_lo_u32 v30, v29, s36 // (op 9198) least significant: low of low * low
v_add_u32 v31, v31, v26 // (op 9198) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 9198) 
s_mul_i32 s53, s54, s76 // (op 9198) 


s_sub_u32 s54, s52, s53 // (op 9198) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 9198) 


v_mov_b32 v29, 0 // (op 9198) convert
v_mov_b32 v28, v26 // (op 9198) convert

v_add_co_u32 v38, s[52:53], v28, v30 // (op 9198) least significant half
v_addc_co_u32 v39, s[52:53], v29, v31, s[52:53] // (op 9198) most significant half



s_mul_i32 s53, s36, s23 // (op 9200) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 9200) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 9200) least significant
s_mul_hi_u32 s55, s36, s22 // (op 9200) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9200) most significant: sum
s_add_u32 s53, s53, s55 // (op 9200) most significant: sum


s_mov_b32 s72, s38 // (op 9200) reg expression
s_mov_b32 s73, s39 // (op 9200) reg expression
s_mov_b32 s74, s52 // (op 9200) reg expression
s_mov_b32 s75, 131072 // (op 9200) call()

v_lshrrev_b32 v26, 2, v1 // (op 9307) 
s_mul_i32 s52, s16, s8 // (op 9307) 
s_add_u32 s52, s52, s0 // (op 9307) 
v_lshlrev_b32 v28, 1, v26 // (op 9307) 

s_mul_hi_u32 s53, s52, s13 // (op 9307) 
v_and_b32 v26, -4, v1 // (op 9307) 
s_sub_u32 s54, s52, s53 // (op 9307) 
v_sub_u32 v29, v1, v26 // (op 9307) 

s_lshr_b32 s55, s54, 1 // (op 9307) 

s_add_u32 s54, s55, s53 // (op 9307) 


s_lshr_b32 s53, s54, s11 // (op 9307) 

s_and_b32 s54, s11, 2147483648 // (op 9307) 
s_cmp_eq_i32 s54, 0 // (op 9307) 
s_mov_b32 s55, scc // (op 9307) 

s_cmp_eq_i32 s55, 0 // (op 9307) 
s_cselect_b32 s54, s52, s53 // (op 9307) 


s_mul_hi_u32 s53, s54, s12 // (op 9307) 
s_sub_u32 s55, s54, s53 // (op 9307) 
s_lshr_b32 s76, s55, 1 // (op 9307) 

s_add_u32 s55, s76, s53 // (op 9307) 


s_lshr_b32 s53, s55, s10 // (op 9307) 

s_and_b32 s55, s10, 2147483648 // (op 9307) 
s_cmp_eq_i32 s55, 0 // (op 9307) 
s_mov_b32 s76, scc // (op 9307) 

s_cmp_eq_i32 s76, 0 // (op 9307) 
s_cselect_b32 s55, s54, s53 // (op 9307) 


v_lshl_add_u32 v26, s55, 7, v28 // (op 9307) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 9307) 
s_mul_i32 s53, s54, s76 // (op 9307) 


s_sub_u32 s54, s52, s53 // (op 9307) 


v_lshl_add_u32 v28, s54, 2, v29 // (op 9307) 


v_mul_lo_u32 v29, v26, s40 // (op 9307) 
v_add_u32 v29, v29, v28 // (op 9307) 


s_mul_i32 s53, s40, s35 // (op 9309) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 9309) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 9309) least significant
s_mul_hi_u32 s55, s40, s34 // (op 9309) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9309) most significant: sum
s_add_u32 s53, s53, s55 // (op 9309) most significant: sum


s_mov_b32 s76, s42 // (op 9309) reg expression
s_mov_b32 s77, s43 // (op 9309) reg expression
s_mov_b32 s78, s52 // (op 9309) reg expression
s_mov_b32 s79, 131072 // (op 9309) call()

v_lshrrev_b32 v26, 2, v1 // (op 9326) 
s_mul_i32 s52, s16, s8 // (op 9326) 
s_add_u32 s52, s52, s0 // (op 9326) 
v_lshlrev_b32 v28, 1, v26 // (op 9326) 

s_mul_hi_u32 s53, s52, s13 // (op 9326) 
v_and_b32 v26, -4, v1 // (op 9326) 
s_sub_u32 s54, s52, s53 // (op 9326) 
v_sub_u32 v30, v1, v26 // (op 9326) 

s_lshr_b32 s55, s54, 1 // (op 9326) 

s_add_u32 s54, s55, s53 // (op 9326) 


s_lshr_b32 s53, s54, s11 // (op 9326) 

s_and_b32 s54, s11, 2147483648 // (op 9326) 
s_cmp_eq_i32 s54, 0 // (op 9326) 
s_mov_b32 s55, scc // (op 9326) 

s_cmp_eq_i32 s55, 0 // (op 9326) 
s_cselect_b32 s54, s52, s53 // (op 9326) 


s_mul_hi_u32 s53, s54, s12 // (op 9326) 
s_sub_u32 s55, s54, s53 // (op 9326) 
s_lshr_b32 s80, s55, 1 // (op 9326) 

s_add_u32 s55, s80, s53 // (op 9326) 


s_lshr_b32 s53, s55, s10 // (op 9326) 

s_and_b32 s55, s10, 2147483648 // (op 9326) 
s_cmp_eq_i32 s55, 0 // (op 9326) 
s_mov_b32 s80, scc // (op 9326) 

s_cmp_eq_i32 s80, 0 // (op 9326) 
s_cselect_b32 s55, s54, s53 // (op 9326) 


v_lshl_add_u32 v26, s55, 7, v28 // (op 9326) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 9326) 
s_mul_i32 s53, s54, s80 // (op 9326) 


s_sub_u32 s54, s52, s53 // (op 9326) 


v_lshl_add_u32 v28, s54, 2, v30 // (op 9326) 


v_mul_lo_u32 v30, v26, s40 // (op 9326) 
v_add_u32 v30, v30, v28 // (op 9326) 


s_mul_i32 s53, s40, s35 // (op 9328) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 9328) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 9328) least significant
s_mul_hi_u32 s55, s40, s34 // (op 9328) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9328) most significant: sum
s_add_u32 s53, s53, s55 // (op 9328) most significant: sum


s_mov_b32 s76, s42 // (op 9328) reg expression
s_mov_b32 s77, s43 // (op 9328) reg expression
s_mov_b32 s78, s52 // (op 9328) reg expression
s_mov_b32 s79, 131072 // (op 9328) call()

v_lshrrev_b32 v26, 2, v1 // (op 9435) 
s_mul_i32 s52, s16, s8 // (op 9435) 
s_add_u32 s52, s52, s0 // (op 9435) 
v_and_b32 v28, -4, v1 // (op 9435) 
s_mul_hi_u32 s53, s52, s13 // (op 9435) 
v_sub_u32 v31, v1, v28 // (op 9435) 

s_sub_u32 s54, s52, s53 // (op 9435) 
s_lshr_b32 s55, s54, 1 // (op 9435) 

s_add_u32 s54, s55, s53 // (op 9435) 


s_lshr_b32 s53, s54, s11 // (op 9435) 

s_and_b32 s54, s11, 2147483648 // (op 9435) 
s_cmp_eq_i32 s54, 0 // (op 9435) 
s_mov_b32 s55, scc // (op 9435) 

s_cmp_eq_i32 s55, 0 // (op 9435) 
s_cselect_b32 s54, s52, s53 // (op 9435) 


s_mul_hi_u32 s53, s54, s12 // (op 9435) 
s_sub_u32 s55, s54, s53 // (op 9435) 
s_lshr_b32 s80, s55, 1 // (op 9435) 

s_add_u32 s55, s80, s53 // (op 9435) 


s_lshr_b32 s53, s55, s10 // (op 9435) 

s_and_b32 s55, s10, 2147483648 // (op 9435) 
s_cmp_eq_i32 s55, 0 // (op 9435) 
s_mov_b32 s80, scc // (op 9435) 

s_cmp_eq_i32 s80, 0 // (op 9435) 
s_cselect_b32 s55, s54, s53 // (op 9435) 


s_mul_i32 s53, s55, s14 // (op 9435) 

s_sub_u32 s55, s54, s53 // (op 9435) 

v_lshl_add_u32 v28, s55, 6, v26 // (op 9435) 


v_mul_lo_u32 v26, v28, s44 // (op 9435) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 9435) 
s_mul_i32 s53, s54, s80 // (op 9435) 


s_sub_u32 s54, s52, s53 // (op 9435) 


v_lshl_add_u32 v28, s54, 2, v31 // (op 9435) 


v_add_u32 v31, v28, v26 // (op 9435) 


s_mul_i32 s53, s44, s23 // (op 9437) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 9437) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 9437) least significant
s_mul_hi_u32 s55, s44, s22 // (op 9437) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9437) most significant: sum
s_add_u32 s53, s53, s55 // (op 9437) most significant: sum


s_mov_b32 s80, s46 // (op 9437) reg expression
s_mov_b32 s81, s47 // (op 9437) reg expression
s_mov_b32 s82, s52 // (op 9437) reg expression
s_mov_b32 s83, 131072 // (op 9437) call()

v_lshrrev_b32 v26, 2, v1 // (op 9454) 
s_mul_i32 s52, s16, s8 // (op 9454) 
s_add_u32 s52, s52, s0 // (op 9454) 
v_and_b32 v28, -4, v1 // (op 9454) 
s_mul_hi_u32 s53, s52, s13 // (op 9454) 
v_sub_u32 v40, v1, v28 // (op 9454) 

s_sub_u32 s54, s52, s53 // (op 9454) 
s_lshr_b32 s55, s54, 1 // (op 9454) 

s_add_u32 s54, s55, s53 // (op 9454) 


s_lshr_b32 s53, s54, s11 // (op 9454) 

s_and_b32 s54, s11, 2147483648 // (op 9454) 
s_cmp_eq_i32 s54, 0 // (op 9454) 
s_mov_b32 s55, scc // (op 9454) 

s_cmp_eq_i32 s55, 0 // (op 9454) 
s_cselect_b32 s54, s52, s53 // (op 9454) 


s_mul_hi_u32 s53, s54, s12 // (op 9454) 
s_sub_u32 s55, s54, s53 // (op 9454) 
s_lshr_b32 s84, s55, 1 // (op 9454) 

s_add_u32 s55, s84, s53 // (op 9454) 


s_lshr_b32 s53, s55, s10 // (op 9454) 

s_and_b32 s55, s10, 2147483648 // (op 9454) 
s_cmp_eq_i32 s55, 0 // (op 9454) 
s_mov_b32 s84, scc // (op 9454) 

s_cmp_eq_i32 s84, 0 // (op 9454) 
s_cselect_b32 s55, s54, s53 // (op 9454) 


s_mul_i32 s53, s55, s14 // (op 9454) 

s_sub_u32 s55, s54, s53 // (op 9454) 

v_lshl_add_u32 v28, s55, 6, v26 // (op 9454) 


v_mul_lo_u32 v26, v28, s44 // (op 9454) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 9454) 
s_mul_i32 s53, s54, s84 // (op 9454) 


s_sub_u32 s54, s52, s53 // (op 9454) 


v_lshl_add_u32 v28, s54, 2, v40 // (op 9454) 


v_add_u32 v40, v28, v26 // (op 9454) 


s_mul_i32 s53, s44, s23 // (op 9456) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 9456) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 9456) least significant
s_mul_hi_u32 s55, s44, s22 // (op 9456) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9456) most significant: sum
s_add_u32 s53, s53, s55 // (op 9456) most significant: sum


s_mov_b32 s80, s46 // (op 9456) reg expression
s_mov_b32 s81, s47 // (op 9456) reg expression
s_mov_b32 s82, s52 // (op 9456) reg expression
s_mov_b32 s83, 131072 // (op 9456) call()

v_and_b32 v26, -8, v1 // (op 9631) 
v_sub_u32 v28, v1, v26 // (op 9631) 

v_lshlrev_b32 v26, 4, v28 // (op 9631) 

v_lshrrev_b32 v28, 3, v1 // (op 9631) 
v_lshl_add_u32 v41, v28, 7, v26 // (op 9631) 


s_nop 0
v_readfirstlane_b32 s52, v41 // (op 9631) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10706) 

s_lshl_b64 s[54:55], s[36:37], 5 // (op 8917) 
v_mov_b64 v[42:43], s[54:55] // (op 8917) convert


v_mov_b64 v[44:45], v[36:37] // (op 8917) 
s_mov_b32 m0, s52 // (op 8917) reg expression
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 8917) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 8917) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 8917) most significant half

s_add_u32 m0, m0, 4096 // (op 8917) 
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 8917) Load value direct to lds






buffer_load_ubyte v26, v31, s[80:83], 0 offen offset: 0 // (op 8652) Load value
v_and_b32 v28, -8, v1 // (op 9556) 
v_sub_u32 v41, v1, v28 // (op 9556) 

v_lshlrev_b32 v28, 4, v41 // (op 9556) 

v_lshrrev_b32 v41, 3, v1 // (op 9556) 
v_lshl_add_u32 v42, v41, 7, v28 // (op 9556) 


s_nop 0
v_readfirstlane_b32 s52, v42 // (op 9556) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10687) 

s_lshl_b64 s[54:55], s[30:31], 5 // (op 8922) 
v_mov_b64 v[42:43], s[54:55] // (op 8922) convert


v_mov_b64 v[44:45], v[32:33] // (op 8922) 
s_add_u32 s53, 8192, s52 // (op 8922) 
s_mov_b32 m0, s53 // (op 8922) reg expression
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 8922) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 8922) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 8922) most significant half

s_add_u32 m0, m0, 4096 // (op 8922) 
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 8922) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 8922) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 8922) most significant half

s_add_u32 m0, m0, 4096 // (op 8922) 
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 8922) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 8922) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 8922) most significant half

s_add_u32 m0, m0, 4096 // (op 8922) 
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 8922) Load value direct to lds





v_mov_b64 v[42:43], s[40:41] // (op 8658) convert

v_mov_b32 v28, v29 // (op 8658) 
buffer_load_ubyte v44, v28, s[76:79], 0 offen offset: 0 // (op 8658) Load value
v_add_u32 v28, v28, v42 // (op 8658) 
buffer_load_ubyte v45, v28, s[76:79], 0 offen offset: 0 // (op 8658) Load value


v_and_b32 v28, -8, v1 // (op 10441) 
v_sub_u32 v41, v1, v28 // (op 10441) 

v_lshlrev_b32 v28, 4, v41 // (op 10441) 

v_lshrrev_b32 v41, 3, v1 // (op 10441) 
v_lshl_add_u32 v42, v41, 7, v28 // (op 10441) 


s_nop 0
v_readfirstlane_b32 s52, v42 // (op 10441) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10889) 

s_lshl_b64 s[54:55], s[36:37], 5 // (op 8927) 
v_mov_b64 v[42:43], s[54:55] // (op 8927) convert


v_mov_b64 v[46:47], v[36:37] // (op 8927) 
s_add_u32 s53, 24576, s52 // (op 8927) 
s_mov_b32 m0, s53 // (op 8927) reg expression
s_mov_b32 s54, 128 // (op 8927) call()
buffer_load_dwordx4 v46, s[72:75], s54 offen offset: 0 lds // (op 8927) Load value direct to lds
v_add_co_u32 v46, s[84:85], v46, v42 // (op 8927) least significant half
v_addc_co_u32 v47, s[84:85], 0, v47, s[84:85] // (op 8927) most significant half

s_add_u32 m0, m0, 4096 // (op 8927) 
buffer_load_dwordx4 v46, s[72:75], s54 offen offset: 0 lds // (op 8927) Load value direct to lds









buffer_load_ubyte v28, v31, s[80:83], 0 offen offset:4 // (op 8664) Load value

v_and_b32 v31, -8, v1 // (op 10397) 
v_sub_u32 v36, v1, v31 // (op 10397) 

v_lshlrev_b32 v31, 4, v36 // (op 10397) 

v_lshrrev_b32 v36, 3, v1 // (op 10397) 
v_lshl_add_u32 v37, v36, 7, v31 // (op 10397) 


s_nop 0
v_readfirstlane_b32 s52, v37 // (op 10397) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10879) 

s_lshl_b64 s[54:55], s[30:31], 5 // (op 8929) 
v_mov_b64 v[36:37], s[54:55] // (op 8929) convert


v_mov_b64 v[42:43], v[32:33] // (op 8929) 
s_add_u32 s53, 32768, s52 // (op 8929) 
s_mov_b32 m0, s53 // (op 8929) reg expression
s_mov_b32 s54, 128 // (op 8929) call()
buffer_load_dwordx4 v42, s[68:71], s54 offen offset: 0 lds // (op 8929) Load value direct to lds
v_add_co_u32 v42, s[84:85], v42, v36 // (op 8929) least significant half
v_addc_co_u32 v43, s[84:85], 0, v43, s[84:85] // (op 8929) most significant half

s_add_u32 m0, m0, 4096 // (op 8929) 
buffer_load_dwordx4 v42, s[68:71], s54 offen offset: 0 lds // (op 8929) Load value direct to lds
v_add_co_u32 v42, s[84:85], v42, v36 // (op 8929) least significant half
v_addc_co_u32 v43, s[84:85], 0, v43, s[84:85] // (op 8929) most significant half

s_add_u32 m0, m0, 4096 // (op 8929) 
buffer_load_dwordx4 v42, s[68:71], s54 offen offset: 0 lds // (op 8929) Load value direct to lds
v_add_co_u32 v42, s[84:85], v42, v36 // (op 8929) least significant half
v_addc_co_u32 v43, s[84:85], 0, v43, s[84:85] // (op 8929) most significant half

s_add_u32 m0, m0, 4096 // (op 8929) 
buffer_load_dwordx4 v42, s[68:71], s54 offen offset: 0 lds // (op 8929) Load value direct to lds







v_mov_b64 v[32:33], s[40:41] // (op 8670) convert

v_mov_b32 v31, v29 // (op 8670) 
buffer_load_ubyte v36, v31, s[76:79], 0 offen offset:4 // (op 8670) Load value
v_add_u32 v31, v31, v32 // (op 8670) 
buffer_load_ubyte v37, v31, s[76:79], 0 offen offset:4 // (op 8670) Load value



v_and_b32 v29, -4, v1 // (op 9783) 
v_sub_u32 v31, v1, v29 // (op 9783) 

v_and_b32 v29, 4294967292, v1 // (op 9783) 
v_add_u32 v32, v29, v31 // (op 9783) 


s_barrier  // (op 10747) 



s_waitcnt vmcnt(15)
ds_write_b8 v32, v26 offset:49152 // (op 8673) Store local data 

v_and_b32 v29, -4, v1 // (op 9707) 
v_sub_u32 v31, v1, v29 // (op 9707) 

v_lshrrev_b32 v29, 2, v1 // (op 9707) 
v_lshl_add_u32 v32, v29, 3, v31 // (op 9707) 


s_barrier  // (op 10726) 



s_waitcnt vmcnt(10)
ds_write_b8 v32, v44 offset:49408 // (op 8675) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v32, v45 offset:49412 // (op 8675) Store local data 

s_barrier  // (op 10703) 



ds_read_b128 v[46:49], v23 // (op 8710) Load local data 
ds_read_b128 v[50:53], v23 offset:64 // (op 8710) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 10744) 



ds_read_u8 v29, v24 offset:49152 // (op 8740) Load local data 



ds_read_b128 v[54:57], v23 offset:4096 // (op 8715) Load local data 
ds_read_b128 v[58:61], v23 offset:4160 // (op 8715) Load local data 



ds_read_u8 v31, v24 offset:49280 // (op 8745) Load local data 
s_barrier  // (op 10684) 



ds_read_b128 v[62:65], v21 offset:8192 // (op 8690) Load local data 
ds_read_b128 v[66:69], v21 offset:8256 // (op 8690) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 10723) 



ds_read_u8 v32, v20 offset:49408 // (op 8720) Load local data 



ds_read_b128 v[70:73], v21 offset:12288 // (op 8695) Load local data 
ds_read_b128 v[74:77], v21 offset:12352 // (op 8695) Load local data 



ds_read_u8 v33, v20 offset:49536 // (op 8725) Load local data 



ds_read_b128 v[78:81], v21 offset:16384 // (op 8700) Load local data 
ds_read_b128 v[82:85], v21 offset:16448 // (op 8700) Load local data 



ds_read_u8 v41, v20 offset:49664 // (op 8730) Load local data 



ds_read_b128 v[86:89], v21 offset:20480 // (op 8705) Load local data 
ds_read_b128 v[90:93], v21 offset:20544 // (op 8705) Load local data 



ds_read_u8 v42, v20 offset:49792 // (op 8735) Load local data 
v_and_b32 v43, -8, v1 // (op 9528) 
v_sub_u32 v94, v1, v43 // (op 9528) 

v_lshlrev_b32 v43, 4, v94 // (op 9528) 

v_lshrrev_b32 v94, 3, v1 // (op 9528) 
v_lshl_add_u32 v95, v94, 7, v43 // (op 9528) 


s_nop 0
v_readfirstlane_b32 s52, v95 // (op 9528) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v43, -8, v1 // (op 9603) 
v_sub_u32 v94, v1, v43 // (op 9603) 

v_lshlrev_b32 v43, 4, v94 // (op 9603) 

v_lshrrev_b32 v94, 3, v1 // (op 9603) 
v_lshl_add_u32 v95, v94, 7, v43 // (op 9603) 


s_nop 0
v_readfirstlane_b32 s53, v95 // (op 9603) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v43, -4, v1 // (op 9678) 
v_sub_u32 v94, v1, v43 // (op 9678) 

v_lshrrev_b32 v43, 2, v1 // (op 9678) 
v_lshl_add_u32 v95, v43, 3, v94 // (op 9678) 


v_and_b32 v43, -4, v1 // (op 9754) 
v_sub_u32 v94, v1, v43 // (op 9754) 

v_and_b32 v43, 4294967292, v1 // (op 9754) 
v_add_u32 v96, v43, v94 // (op 9754) 


v_and_b32 v43, -8, v1 // (op 10382) 
v_sub_u32 v94, v1, v43 // (op 10382) 

v_lshlrev_b32 v43, 4, v94 // (op 10382) 

v_lshrrev_b32 v94, 3, v1 // (op 10382) 
v_lshl_add_u32 v97, v94, 7, v43 // (op 10382) 


s_nop 0
v_readfirstlane_b32 s54, v97 // (op 10382) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v43, -8, v1 // (op 10426) 
v_sub_u32 v94, v1, v43 // (op 10426) 

v_lshlrev_b32 v43, 4, v94 // (op 10426) 

v_lshrrev_b32 v94, 3, v1 // (op 10426) 
v_lshl_add_u32 v97, v94, 7, v43 // (op 10426) 


s_nop 0
v_readfirstlane_b32 s55, v97 // (op 10426) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v43, -4, v1 // (op 10470) 
v_sub_u32 v94, v1, v43 // (op 10470) 

v_lshrrev_b32 v43, 2, v1 // (op 10470) 
v_lshl_add_u32 v97, v43, 3, v94 // (op 10470) 


v_and_b32 v43, -4, v1 // (op 10499) 
v_sub_u32 v94, v1, v43 // (op 10499) 

v_and_b32 v43, 4294967292, v1 // (op 10499) 
v_add_u32 v98, v43, v94 // (op 10499) 


s_mov_b32 s1, 0 // (op 3707) call()
s_and_b32 s84, s7, 4294967294 // (op 3706) 
s_cmp_lt_u32 s1, s84 // (op 3706) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoop_3706 // (op 3706) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoop_3706 if false)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoop_3706:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[46:53], a[0:3], v32, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 88) 
s_barrier  // (op 10709) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:69], v[54:61], a[4:7], v32, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2722) 
s_add_u32 s1, s1, 2 // (op 3708) 
s_lshl_b64 s[84:85], s[36:37], 5 // (op 8842) 
v_mov_b64 v[100:101], s[84:85] // (op 8842) convert


v_mov_b64 v[102:103], v[38:39] // (op 8842) 
v_mov_b64 v[104:105], v[102:103] // (op 8842) 
s_mov_b32 m0, s53 // (op 8842) reg expression
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[70:77], v[46:53], a[8:11], v33, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3019) 
s_mov_b32 s84, 256 // (op 8842) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:77], v[54:61], a[12:15], v33, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2973) 
buffer_load_dwordx4 v104, s[72:75], s84 offen offset: 0 lds // (op 8842) Load value direct to lds
v_add_co_u32 v104, s[86:87], v104, v100 // (op 8842) least significant half
v_addc_co_u32 v105, s[86:87], 0, v105, s[86:87] // (op 8842) most significant half

s_add_u32 m0, m0, 4096 // (op 8842) 
buffer_load_dwordx4 v104, s[72:75], s84 offen offset: 0 lds // (op 8842) Load value direct to lds



s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[78:85], v[46:53], a[16:19], v41, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3405) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:85], v[54:61], a[20:23], v41, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3359) 
v_mov_b32 v43, v40 // (op 25) 

buffer_load_ubyte v26, v43, s[80:83], 0 offen offset:8 // (op 25) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[86:93], v[46:53], a[24:27], v42, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3791) 
s_barrier  // (op 10690) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[86:93], v[54:61], a[28:31], v42, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3745) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 8837) 
v_mov_b64 v[100:101], s[84:85] // (op 8837) convert


v_mov_b64 v[104:105], v[34:35] // (op 8837) 
v_mov_b64 v[106:107], v[104:105] // (op 8837) 
s_add_u32 s84, 8192, s52 // (op 8837) 
s_mov_b32 m0, s84 // (op 8837) reg expression
s_mov_b32 s85, 256 // (op 8837) call()
buffer_load_dwordx4 v106, s[68:71], s85 offen offset: 0 lds // (op 8837) Load value direct to lds
v_add_co_u32 v106, s[86:87], v106, v100 // (op 8837) least significant half
v_addc_co_u32 v107, s[86:87], 0, v107, s[86:87] // (op 8837) most significant half

s_add_u32 m0, m0, 4096 // (op 8837) 
buffer_load_dwordx4 v106, s[68:71], s85 offen offset: 0 lds // (op 8837) Load value direct to lds
v_add_co_u32 v106, s[86:87], v106, v100 // (op 8837) least significant half
v_addc_co_u32 v107, s[86:87], 0, v107, s[86:87] // (op 8837) most significant half

s_add_u32 m0, m0, 4096 // (op 8837) 
buffer_load_dwordx4 v106, s[68:71], s85 offen offset: 0 lds // (op 8837) Load value direct to lds
v_add_co_u32 v106, s[86:87], v106, v100 // (op 8837) least significant half
v_addc_co_u32 v107, s[86:87], 0, v107, s[86:87] // (op 8837) most significant half

s_add_u32 m0, m0, 4096 // (op 8837) 
buffer_load_dwordx4 v106, s[68:71], s85 offen offset: 0 lds // (op 8837) Load value direct to lds





v_mov_b64 v[100:101], s[40:41] // (op 17) convert

v_mov_b32 v94, v30 // (op 17) 
v_mov_b32 v99, v94 // (op 17) 
buffer_load_ubyte v44, v99, s[76:79], 0 offen offset:8 // (op 17) Load value
v_add_u32 v99, v99, v100 // (op 17) 
buffer_load_ubyte v45, v99, s[76:79], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 10905) 



s_waitcnt vmcnt(15)
ds_write_b8 v98, v28 offset:49920 // (op 2601) Store local data 
s_barrier  // (op 10899) 



s_waitcnt vmcnt(10)
ds_write_b8 v97, v36 offset:50176 // (op 2600) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v97, v37 offset:50180 // (op 2600) Store local data 
s_barrier  // (op 10885) 



ds_read_b128 v[106:109], v16 offset:24576 // (op 2593) Load local data 
ds_read_b128 v[110:113], v16 offset:24640 // (op 2593) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 10902) 



ds_read_u8 v99, v17 offset:49920 // (op 2597) Load local data 



ds_read_b128 v[114:117], v16 offset:28672 // (op 2727) Load local data 
ds_read_b128 v[118:121], v16 offset:28736 // (op 2727) Load local data 



ds_read_u8 v100, v17 offset:50048 // (op 2731) Load local data 
s_barrier  // (op 10876) 
s_barrier  // (op 11108) 



ds_read_b128 v[62:65], v15 offset:32768 // (op 2591) Load local data 
ds_read_b128 v[66:69], v15 offset:32832 // (op 2591) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 10895) 



ds_read_u8 v32, v14 offset:50176 // (op 2595) Load local data 



ds_read_b128 v[70:73], v15 offset:36864 // (op 2976) Load local data 
ds_read_b128 v[74:77], v15 offset:36928 // (op 2976) Load local data 



ds_read_u8 v33, v14 offset:50304 // (op 2979) Load local data 



ds_read_b128 v[78:81], v15 offset:40960 // (op 3362) Load local data 
ds_read_b128 v[82:85], v15 offset:41024 // (op 3362) Load local data 



ds_read_u8 v41, v14 offset:50432 // (op 3365) Load local data 



ds_read_b128 v[86:89], v15 offset:45056 // (op 3748) Load local data 
ds_read_b128 v[90:93], v15 offset:45120 // (op 3748) Load local data 



ds_read_u8 v42, v14 offset:50560 // (op 3751) Load local data 
s_barrier  // (op 11138) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[106:113], a[0:3], v32, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2598) 
s_barrier  // (op 10892) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:69], v[114:121], a[4:7], v32, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2723) 
s_lshl_b64 s[84:85], s[36:37], 5 // (op 8898) 
v_mov_b64 v[122:123], s[84:85] // (op 8898) convert


v_mov_b64 v[124:125], v[102:103] // (op 8898) 
s_add_u32 s84, 24576, s55 // (op 8898) 
s_mov_b32 m0, s84 // (op 8898) reg expression
s_mov_b32 s85, 384 // (op 8898) call()
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[70:77], v[106:113], a[8:11], v33, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3020) 
buffer_load_dwordx4 v124, s[72:75], s85 offen offset: 0 lds // (op 8898) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:77], v[114:121], a[12:15], v33, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2974) 
v_add_co_u32 v124, s[86:87], v124, v122 // (op 8898) least significant half
v_addc_co_u32 v125, s[86:87], 0, v125, s[86:87] // (op 8898) most significant half

s_add_u32 m0, m0, 4096 // (op 8898) 
buffer_load_dwordx4 v124, s[72:75], s85 offen offset: 0 lds // (op 8898) Load value direct to lds




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[78:85], v[106:113], a[16:19], v41, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3406) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:85], v[114:121], a[20:23], v41, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3360) 
buffer_load_ubyte v28, v43, s[80:83], 0 offen offset:12 // (op 2604) Load value

v_add_co_u32 v38, vcc, 256, v38 // (op 9190) least significant half
v_addc_co_u32 v39, vcc, 0, v39, vcc // (op 9190) most significant half
v_add_u32 v40, 8, v40 // (op 9446) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[86:93], v[106:113], a[24:27], v42, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3792) 
s_barrier  // (op 10882) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[86:93], v[114:121], a[28:31], v42, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3746) 
s_lshl_b64 s[84:85], s[30:31], 5 // (op 8893) 


v_mov_b64 v[102:103], s[84:85] // (op 8893) convert


v_mov_b64 v[100:101], v[104:105] // (op 8893) 
s_add_u32 s84, 32768, s54 // (op 8893) 
s_mov_b32 m0, s84 // (op 8893) reg expression
s_mov_b32 s85, 384 // (op 8893) call()
buffer_load_dwordx4 v100, s[68:71], s85 offen offset: 0 lds // (op 8893) Load value direct to lds
v_add_co_u32 v100, s[86:87], v100, v102 // (op 8893) least significant half
v_addc_co_u32 v101, s[86:87], 0, v101, s[86:87] // (op 8893) most significant half

s_add_u32 m0, m0, 4096 // (op 8893) 
buffer_load_dwordx4 v100, s[68:71], s85 offen offset: 0 lds // (op 8893) Load value direct to lds
v_add_co_u32 v100, s[86:87], v100, v102 // (op 8893) least significant half
v_addc_co_u32 v101, s[86:87], 0, v101, s[86:87] // (op 8893) most significant half

s_add_u32 m0, m0, 4096 // (op 8893) 
buffer_load_dwordx4 v100, s[68:71], s85 offen offset: 0 lds // (op 8893) Load value direct to lds
v_add_co_u32 v100, s[86:87], v100, v102 // (op 8893) least significant half
v_addc_co_u32 v101, s[86:87], 0, v101, s[86:87] // (op 8893) most significant half

s_add_u32 m0, m0, 4096 // (op 8893) 
buffer_load_dwordx4 v100, s[68:71], s85 offen offset: 0 lds // (op 8893) Load value direct to lds






v_add_co_u32 v34, vcc, 256, v34 // (op 9062) least significant half
v_addc_co_u32 v35, vcc, 0, v35, vcc // (op 9062) most significant half
v_mov_b64 v[100:101], s[40:41] // (op 2603) convert

v_mov_b32 v43, v94 // (op 2603) 
buffer_load_ubyte v36, v43, s[76:79], 0 offen offset:12 // (op 2603) Load value
v_add_u32 v43, v43, v100 // (op 2603) 
buffer_load_ubyte v37, v43, s[76:79], 0 offen offset:12 // (op 2603) Load value



s_barrier  // (op 10750) 
v_add_u32 v30, 8, v30 // (op 9318) 



s_waitcnt vmcnt(15)
ds_write_b8 v96, v26 offset:49152 // (op 30) Store local data 
s_barrier  // (op 10729) 



s_waitcnt vmcnt(10)
ds_write_b8 v95, v44 offset:49408 // (op 21) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v95, v45 offset:49412 // (op 21) Store local data 
s_barrier  // (op 10715) 



ds_read_b128 v[46:49], v4 // (op 75) Load local data 
ds_read_b128 v[50:53], v4 offset:64 // (op 75) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 10757) 



ds_read_u8 v29, v5 offset:49152 // (op 85) Load local data 



ds_read_b128 v[54:57], v4 offset:4096 // (op 2717) Load local data 
ds_read_b128 v[58:61], v4 offset:4160 // (op 2717) Load local data 



ds_read_u8 v31, v5 offset:49280 // (op 2721) Load local data 
s_barrier  // (op 10696) 



ds_read_b128 v[62:65], v3 offset:8192 // (op 70) Load local data 
ds_read_b128 v[66:69], v3 offset:8256 // (op 70) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 10736) 



ds_read_u8 v32, v0 offset:49408 // (op 80) Load local data 



ds_read_b128 v[70:73], v3 offset:12288 // (op 2968) Load local data 
ds_read_b128 v[74:77], v3 offset:12352 // (op 2968) Load local data 



ds_read_u8 v33, v0 offset:49536 // (op 2971) Load local data 



ds_read_b128 v[78:81], v3 offset:16384 // (op 3354) Load local data 
ds_read_b128 v[82:85], v3 offset:16448 // (op 3354) Load local data 



ds_read_u8 v41, v0 offset:49664 // (op 3357) Load local data 



ds_read_b128 v[86:89], v3 offset:20480 // (op 3740) Load local data 
ds_read_b128 v[90:93], v3 offset:20544 // (op 3740) Load local data 



ds_read_u8 v42, v0 offset:49792 // (op 3743) Load local data 
s_and_b32 s84, s7, 4294967294 // (op 3706) 
s_cmp_lt_u32 s1, s84 // (op 3706) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoop_3706 // (op 3706) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoop_3706 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoop_3706:
















s_barrier  // (op 11122) 
s_and_b32 s52, s7, 4294967294 // (op 8966) 
v_lshrrev_b32 v26, 3, v1 // (op 8966) 
s_add_u32 s53, s0, s52 // (op 8966) 

v_and_b32 v29, -8, v1 // (op 8966) 
v_sub_u32 v30, v1, v29 // (op 8966) 

s_mul_i32 s52, s16, s8 // (op 8966) 
s_add_u32 s52, s52, s53 // (op 8966) 

v_lshlrev_b32 v29, 4, v30 // (op 8966) 

s_mul_hi_u32 s53, s52, s13 // (op 8966) 
s_sub_u32 s54, s52, s53 // (op 8966) 
s_lshr_b32 s55, s54, 1 // (op 8966) 

s_add_u32 s54, s55, s53 // (op 8966) 


s_lshr_b32 s53, s54, s11 // (op 8966) 

s_and_b32 s54, s11, 2147483648 // (op 8966) 
s_cmp_eq_i32 s54, 0 // (op 8966) 
s_mov_b32 s55, scc // (op 8966) 

s_cmp_eq_i32 s55, 0 // (op 8966) 
s_cselect_b32 s54, s52, s53 // (op 8966) 


s_mul_hi_u32 s53, s54, s12 // (op 8966) 
s_sub_u32 s55, s54, s53 // (op 8966) 
s_lshr_b32 s84, s55, 1 // (op 8966) 

s_add_u32 s55, s84, s53 // (op 8966) 


s_lshr_b32 s53, s55, s10 // (op 8966) 

s_and_b32 s55, s10, 2147483648 // (op 8966) 
s_cmp_eq_i32 s55, 0 // (op 8966) 
s_mov_b32 s84, scc // (op 8966) 

s_cmp_eq_i32 s84, 0 // (op 8966) 
s_cselect_b32 s55, s54, s53 // (op 8966) 


v_lshl_add_u32 v30, s55, 7, v26 // (op 8966) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 8966) 
s_mul_i32 s53, s54, s84 // (op 8966) 


s_sub_u32 s54, s52, s53 // (op 8966) 


v_lshl_add_u32 v26, s54, 7, v29 // (op 8966) 


v_mov_b32 v35, 0 // (op 8966) convert
v_mov_b32 v34, v26 // (op 8966) convert

v_mul_lo_u32 v39, v30, s31 // (op 8966) most significant: low of low * high
v_mul_hi_u32 v26, v30, s30 // (op 8966) most significant: high of low * low
v_mul_lo_u32 v38, v30, s30 // (op 8966) least significant: low of low * low
v_add_u32 v39, v39, v26 // (op 8966) most significant: sum

v_add_co_u32 v38, s[52:53], v38, v34 // (op 8966) least significant half
v_addc_co_u32 v39, s[52:53], v39, v35, s[52:53] // (op 8966) most significant half



s_mul_i32 s53, s30, s35 // (op 8968) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 8968) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 8968) least significant
s_mul_hi_u32 s55, s30, s34 // (op 8968) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 8968) most significant: sum
s_add_u32 s53, s53, s55 // (op 8968) most significant: sum


s_mov_b32 s68, s32 // (op 8968) reg expression
s_mov_b32 s69, s33 // (op 8968) reg expression
s_mov_b32 s70, s52 // (op 8968) reg expression
s_mov_b32 s71, 131072 // (op 8968) call()

s_and_b32 s52, s7, 4294967294 // (op 9114) 
v_lshrrev_b32 v26, 3, v1 // (op 9114) 
s_add_u32 s53, s0, s52 // (op 9114) 

v_and_b32 v40, -8, v1 // (op 9114) 
v_sub_u32 v29, v1, v40 // (op 9114) 

s_mul_i32 s52, s16, s8 // (op 9114) 
s_add_u32 s52, s52, s53 // (op 9114) 

v_lshlrev_b32 v30, 4, v29 // (op 9114) 

s_mul_hi_u32 s53, s52, s13 // (op 9114) 
s_sub_u32 s54, s52, s53 // (op 9114) 
s_lshr_b32 s55, s54, 1 // (op 9114) 

s_add_u32 s54, s55, s53 // (op 9114) 


s_lshr_b32 s53, s54, s11 // (op 9114) 

s_and_b32 s54, s11, 2147483648 // (op 9114) 
s_cmp_eq_i32 s54, 0 // (op 9114) 
s_mov_b32 s55, scc // (op 9114) 

s_cmp_eq_i32 s55, 0 // (op 9114) 
s_cselect_b32 s54, s52, s53 // (op 9114) 


s_mul_hi_u32 s53, s54, s12 // (op 9114) 
s_sub_u32 s55, s54, s53 // (op 9114) 
s_lshr_b32 s84, s55, 1 // (op 9114) 

s_add_u32 s55, s84, s53 // (op 9114) 


s_lshr_b32 s53, s55, s10 // (op 9114) 

s_and_b32 s55, s10, 2147483648 // (op 9114) 
s_cmp_eq_i32 s55, 0 // (op 9114) 
s_mov_b32 s84, scc // (op 9114) 

s_cmp_eq_i32 s84, 0 // (op 9114) 
s_cselect_b32 s55, s54, s53 // (op 9114) 


s_mul_i32 s53, s55, s14 // (op 9114) 

s_sub_u32 s55, s54, s53 // (op 9114) 

v_lshl_add_u32 v29, s55, 6, v26 // (op 9114) 


v_mul_lo_u32 v35, v29, s37 // (op 9114) most significant: low of low * high
v_mul_hi_u32 v26, v29, s36 // (op 9114) most significant: high of low * low
v_mul_lo_u32 v34, v29, s36 // (op 9114) least significant: low of low * low
v_add_u32 v35, v35, v26 // (op 9114) most significant: sum


s_ashr_i64 s[84:85], s[2:3], 7 // (op 9114) 
s_mul_i32 s53, s54, s84 // (op 9114) 


s_sub_u32 s54, s52, s53 // (op 9114) 


v_lshl_add_u32 v26, s54, 7, v30 // (op 9114) 


v_mov_b32 v53, 0 // (op 9114) convert
v_mov_b32 v52, v26 // (op 9114) convert

v_add_co_u32 v50, s[52:53], v52, v34 // (op 9114) least significant half
v_addc_co_u32 v51, s[52:53], v53, v35, s[52:53] // (op 9114) most significant half



s_mul_i32 s53, s36, s23 // (op 9116) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 9116) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 9116) least significant
s_mul_hi_u32 s55, s36, s22 // (op 9116) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9116) most significant: sum
s_add_u32 s53, s53, s55 // (op 9116) most significant: sum


s_mov_b32 s72, s38 // (op 9116) reg expression
s_mov_b32 s73, s39 // (op 9116) reg expression
s_mov_b32 s74, s52 // (op 9116) reg expression
s_mov_b32 s75, 131072 // (op 9116) call()

s_and_b32 s52, s7, 4294967294 // (op 9242) 
v_lshrrev_b32 v26, 2, v1 // (op 9242) 
s_add_u32 s53, s0, s52 // (op 9242) 

v_lshlrev_b32 v40, 1, v26 // (op 9242) 

v_and_b32 v26, -4, v1 // (op 9242) 
s_mul_i32 s52, s16, s8 // (op 9242) 
s_add_u32 s52, s52, s53 // (op 9242) 

v_sub_u32 v29, v1, v26 // (op 9242) 

s_mul_hi_u32 s53, s52, s13 // (op 9242) 
s_sub_u32 s54, s52, s53 // (op 9242) 
s_lshr_b32 s55, s54, 1 // (op 9242) 

s_add_u32 s54, s55, s53 // (op 9242) 


s_lshr_b32 s53, s54, s11 // (op 9242) 

s_and_b32 s54, s11, 2147483648 // (op 9242) 
s_cmp_eq_i32 s54, 0 // (op 9242) 
s_mov_b32 s55, scc // (op 9242) 

s_cmp_eq_i32 s55, 0 // (op 9242) 
s_cselect_b32 s54, s52, s53 // (op 9242) 


s_mul_hi_u32 s53, s54, s12 // (op 9242) 
s_sub_u32 s55, s54, s53 // (op 9242) 
s_lshr_b32 s84, s55, 1 // (op 9242) 

s_add_u32 s55, s84, s53 // (op 9242) 


s_lshr_b32 s53, s55, s10 // (op 9242) 

s_and_b32 s55, s10, 2147483648 // (op 9242) 
s_cmp_eq_i32 s55, 0 // (op 9242) 
s_mov_b32 s84, scc // (op 9242) 

s_cmp_eq_i32 s84, 0 // (op 9242) 
s_cselect_b32 s55, s54, s53 // (op 9242) 


v_lshl_add_u32 v26, s55, 7, v40 // (op 9242) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 9242) 
s_mul_i32 s53, s54, s84 // (op 9242) 


s_sub_u32 s54, s52, s53 // (op 9242) 


v_lshl_add_u32 v30, s54, 2, v29 // (op 9242) 


v_mul_lo_u32 v29, v26, s40 // (op 9242) 
v_add_u32 v29, v29, v30 // (op 9242) 


s_mul_i32 s53, s40, s35 // (op 9244) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 9244) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 9244) least significant
s_mul_hi_u32 s55, s40, s34 // (op 9244) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9244) most significant: sum
s_add_u32 s53, s53, s55 // (op 9244) most significant: sum


s_mov_b32 s76, s42 // (op 9244) reg expression
s_mov_b32 s77, s43 // (op 9244) reg expression
s_mov_b32 s78, s52 // (op 9244) reg expression
s_mov_b32 s79, 131072 // (op 9244) call()

s_and_b32 s52, s7, 4294967294 // (op 9370) 
v_lshrrev_b32 v26, 2, v1 // (op 9370) 
s_add_u32 s53, s0, s52 // (op 9370) 

v_and_b32 v30, -4, v1 // (op 9370) 
v_sub_u32 v40, v1, v30 // (op 9370) 

s_mul_i32 s52, s16, s8 // (op 9370) 
s_add_u32 s52, s52, s53 // (op 9370) 

s_mul_hi_u32 s53, s52, s13 // (op 9370) 
s_sub_u32 s54, s52, s53 // (op 9370) 
s_lshr_b32 s55, s54, 1 // (op 9370) 

s_add_u32 s54, s55, s53 // (op 9370) 


s_lshr_b32 s53, s54, s11 // (op 9370) 

s_and_b32 s54, s11, 2147483648 // (op 9370) 
s_cmp_eq_i32 s54, 0 // (op 9370) 
s_mov_b32 s55, scc // (op 9370) 

s_cmp_eq_i32 s55, 0 // (op 9370) 
s_cselect_b32 s54, s52, s53 // (op 9370) 


s_mul_hi_u32 s53, s54, s12 // (op 9370) 
s_sub_u32 s55, s54, s53 // (op 9370) 
s_lshr_b32 s84, s55, 1 // (op 9370) 

s_add_u32 s55, s84, s53 // (op 9370) 


s_lshr_b32 s53, s55, s10 // (op 9370) 

s_and_b32 s55, s10, 2147483648 // (op 9370) 
s_cmp_eq_i32 s55, 0 // (op 9370) 
s_mov_b32 s84, scc // (op 9370) 

s_cmp_eq_i32 s84, 0 // (op 9370) 
s_cselect_b32 s55, s54, s53 // (op 9370) 


s_mul_i32 s53, s55, s14 // (op 9370) 

s_sub_u32 s55, s54, s53 // (op 9370) 

v_lshl_add_u32 v30, s55, 6, v26 // (op 9370) 


v_mul_lo_u32 v26, v30, s44 // (op 9370) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 9370) 
s_mul_i32 s53, s54, s84 // (op 9370) 


s_sub_u32 s54, s52, s53 // (op 9370) 


v_lshl_add_u32 v30, s54, 2, v40 // (op 9370) 


v_add_u32 v40, v30, v26 // (op 9370) 


s_mul_i32 s53, s44, s23 // (op 9372) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 9372) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 9372) least significant
s_mul_hi_u32 s55, s44, s22 // (op 9372) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 9372) most significant: sum
s_add_u32 s53, s53, s55 // (op 9372) most significant: sum


s_mov_b32 s80, s46 // (op 9372) reg expression
s_mov_b32 s81, s47 // (op 9372) reg expression
s_mov_b32 s82, s52 // (op 9372) reg expression
s_mov_b32 s83, 131072 // (op 9372) call()

v_and_b32 v26, -8, v1 // (op 9541) 
v_sub_u32 v30, v1, v26 // (op 9541) 

v_lshlrev_b32 v26, 4, v30 // (op 9541) 

v_lshrrev_b32 v30, 3, v1 // (op 9541) 
v_lshl_add_u32 v34, v30, 7, v26 // (op 9541) 


s_nop 0
v_readfirstlane_b32 s52, v34 // (op 9541) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -8, v1 // (op 9616) 
v_sub_u32 v30, v1, v26 // (op 9616) 

v_lshlrev_b32 v26, 4, v30 // (op 9616) 

v_lshrrev_b32 v30, 3, v1 // (op 9616) 
v_lshl_add_u32 v34, v30, 7, v26 // (op 9616) 


s_nop 0
v_readfirstlane_b32 s53, v34 // (op 9616) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 9691) 
v_sub_u32 v30, v1, v26 // (op 9691) 

v_lshrrev_b32 v26, 2, v1 // (op 9691) 
v_lshl_add_u32 v34, v26, 3, v30 // (op 9691) 


v_and_b32 v26, -4, v1 // (op 9767) 
v_sub_u32 v30, v1, v26 // (op 9767) 

v_and_b32 v26, 4294967292, v1 // (op 9767) 
v_add_u32 v35, v26, v30 // (op 9767) 


s_and_b32 s54, s7, 4294967294 // (op 3712) 
s_cmp_lt_u32 s54, s7 // (op 3711) 












s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoopTail_3711 // (op 3711) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoopTail_3711 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoopTail_3711:
s_barrier  // (op 10693) 
s_add_u32 s54, s54, 1 // (op 3713) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 8880) 
v_mov_b64 v[36:37], s[84:85] // (op 8880) convert


v_mov_b64 v[30:31], v[38:39] // (op 8880) 
v_mov_b64 v[32:33], v[30:31] // (op 8880) 
s_add_u32 s55, 8192, s52 // (op 8880) 
s_mov_b32 m0, s55 // (op 8880) reg expression
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 8880) Load value direct to lds
v_add_co_u32 v32, s[84:85], v32, v36 // (op 8880) least significant half
v_addc_co_u32 v33, s[84:85], 0, v33, s[84:85] // (op 8880) most significant half

s_add_u32 m0, m0, 4096 // (op 8880) 
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 8880) Load value direct to lds
v_add_co_u32 v32, s[84:85], v32, v36 // (op 8880) least significant half
v_addc_co_u32 v33, s[84:85], 0, v33, s[84:85] // (op 8880) most significant half

s_add_u32 m0, m0, 4096 // (op 8880) 
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 8880) Load value direct to lds
v_add_co_u32 v32, s[84:85], v32, v36 // (op 8880) least significant half
v_addc_co_u32 v33, s[84:85], 0, v33, s[84:85] // (op 8880) most significant half

s_add_u32 m0, m0, 4096 // (op 8880) 
buffer_load_dwordx4 v32, s[68:71], 0 offen offset: 0 lds // (op 8880) Load value direct to lds




v_add_co_u32 v38, vcc, 128, v38 // (op 8956) least significant half
v_addc_co_u32 v39, vcc, 0, v39, vcc // (op 8956) most significant half
s_barrier  // (op 10712) 

s_lshl_b64 s[84:85], s[36:37], 5 // (op 8886) 
v_mov_b64 v[36:37], s[84:85] // (op 8886) convert


v_mov_b64 v[30:31], v[50:51] // (op 8886) 
v_mov_b64 v[32:33], v[30:31] // (op 8886) 
s_mov_b32 m0, s53 // (op 8886) reg expression
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 8886) Load value direct to lds
v_add_co_u32 v32, s[84:85], v32, v36 // (op 8886) least significant half
v_addc_co_u32 v33, s[84:85], 0, v33, s[84:85] // (op 8886) most significant half

s_add_u32 m0, m0, 4096 // (op 8886) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 8886) Load value direct to lds



v_add_co_u32 v50, vcc, 128, v50 // (op 9106) least significant half
v_addc_co_u32 v51, vcc, 0, v51, vcc // (op 9106) most significant half

v_mov_b64 v[36:37], s[40:41] // (op 2537) convert

v_mov_b32 v26, v29 // (op 2537) 
v_mov_b32 v28, v26 // (op 2537) 
buffer_load_ubyte v30, v28, s[76:79], 0 offen offset: 0 // (op 2537) Load value
v_add_u32 v28, v28, v36 // (op 2537) 
buffer_load_ubyte v31, v28, s[76:79], 0 offen offset: 0 // (op 2537) Load value



v_add_u32 v29, 4, v29 // (op 9234) 


s_barrier  // (op 10732) 
v_mov_b32 v26, v40 // (op 2538) 




buffer_load_ubyte v28, v26, s[80:83], 0 offen offset: 0 // (op 2538) Load value

v_add_u32 v40, 4, v40 // (op 9362) 
s_waitcnt vmcnt(2)
ds_write_b8 v34, v30 offset:49408 // (op 2534) Store local data 
s_barrier  // (op 10699) 
s_waitcnt vmcnt(1)
ds_write_b8 v34, v31 offset:49412 // (op 2534) Store local data 




ds_read_b128 v[42:45], v3 offset:8192 // (op 2525) Load local data 
s_barrier  // (op 10753) 
ds_read_b128 v[46:49], v3 offset:8256 // (op 2525) Load local data 



s_barrier  // (op 10719) 
s_waitcnt vmcnt(0)
ds_write_b8 v35, v28 offset:49152 // (op 2535) Store local data 




ds_read_b128 v[52:55], v4 // (op 2527) Load local data 
ds_read_b128 v[56:59], v4 offset:64 // (op 2527) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 10740) 



ds_read_u8 v26, v0 offset:49408 // (op 2529) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10760) 



ds_read_u8 v28, v5 offset:49152 // (op 2531) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[52:59], a[0:3], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2532) 
ds_read_b128 v[60:63], v4 offset:4096 // (op 2696) Load local data 
ds_read_b128 v[64:67], v4 offset:4160 // (op 2696) Load local data 



ds_read_u8 v41, v5 offset:49280 // (op 2700) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:49], v[60:67], a[4:7], v26, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2701) 
ds_read_b128 v[68:71], v3 offset:12288 // (op 2914) Load local data 


ds_read_b128 v[72:75], v3 offset:12352 // (op 2914) Load local data 



ds_read_u8 v26, v0 offset:49536 // (op 2917) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[68:75], v[52:59], a[8:11], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2919) 
ds_read_b128 v[42:45], v3 offset:16384 // (op 3300) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:75], v[60:67], a[12:15], v26, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2950) 
ds_read_b128 v[46:49], v3 offset:16448 // (op 3300) Load local data 





ds_read_u8 v26, v0 offset:49664 // (op 3303) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[42:49], v[52:59], a[16:19], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3305) 
ds_read_b128 v[68:71], v3 offset:20480 // (op 3686) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[42:49], v[60:67], a[20:23], v26, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3336) 
ds_read_b128 v[72:75], v3 offset:20544 // (op 3686) Load local data 





ds_read_u8 v26, v0 offset:49792 // (op 3689) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[68:75], v[52:59], a[24:27], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3691) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[68:75], v[60:67], a[28:31], v26, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 3722) 




s_cmp_lt_u32 s54, s7 // (op 3711) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoopTail_3711 // (op 3711) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoopTail_3711 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoopTail_3711:









s_barrier  // (op 11126) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 288) 
s_cmp_lg_u32 s18, 0 // (op 314) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Send_Tile_314 // (op 314) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Send_Tile_314



v_add_u32 v26, 24576, v25 // (op 4256) 
buffer_store_dword a24, v26, s[64:67], 0 offen sc1 // (op 4256) Store value

v_add_u32 v26, 24832, v25 // (op 4256) 
buffer_store_dword a25, v26, s[64:67], 0 offen sc1 // (op 4256) Store value

v_add_u32 v26, 25088, v25 // (op 4256) 
buffer_store_dword a26, v26, s[64:67], 0 offen sc1 // (op 4256) Store value

v_add_u32 v26, 25344, v25 // (op 4256) 
buffer_store_dword a27, v26, s[64:67], 0 offen sc1 // (op 4256) Store value




v_add_u32 v26, 24704, v25 // (op 4258) 
buffer_store_dword a28, v26, s[64:67], 0 offen sc1 // (op 4258) Store value

v_add_u32 v26, 24960, v25 // (op 4258) 
buffer_store_dword a29, v26, s[64:67], 0 offen sc1 // (op 4258) Store value

v_add_u32 v26, 25216, v25 // (op 4258) 
buffer_store_dword a30, v26, s[64:67], 0 offen sc1 // (op 4258) Store value

v_add_u32 v26, 25472, v25 // (op 4258) 
buffer_store_dword a31, v26, s[64:67], 0 offen sc1 // (op 4258) Store value




v_add_u32 v26, 16384, v25 // (op 4242) 
buffer_store_dword a16, v26, s[64:67], 0 offen sc1 // (op 4242) Store value

v_add_u32 v26, 16640, v25 // (op 4242) 
buffer_store_dword a17, v26, s[64:67], 0 offen sc1 // (op 4242) Store value

v_add_u32 v26, 16896, v25 // (op 4242) 
buffer_store_dword a18, v26, s[64:67], 0 offen sc1 // (op 4242) Store value

v_add_u32 v26, 17152, v25 // (op 4242) 
buffer_store_dword a19, v26, s[64:67], 0 offen sc1 // (op 4242) Store value




v_add_u32 v26, 16512, v25 // (op 4244) 
buffer_store_dword a20, v26, s[64:67], 0 offen sc1 // (op 4244) Store value

v_add_u32 v26, 16768, v25 // (op 4244) 
buffer_store_dword a21, v26, s[64:67], 0 offen sc1 // (op 4244) Store value

v_add_u32 v26, 17024, v25 // (op 4244) 
buffer_store_dword a22, v26, s[64:67], 0 offen sc1 // (op 4244) Store value

v_add_u32 v26, 17280, v25 // (op 4244) 
buffer_store_dword a23, v26, s[64:67], 0 offen sc1 // (op 4244) Store value




v_add_u32 v26, 8192, v25 // (op 4228) 
buffer_store_dword a8, v26, s[64:67], 0 offen sc1 // (op 4228) Store value

v_add_u32 v26, 8448, v25 // (op 4228) 
buffer_store_dword a9, v26, s[64:67], 0 offen sc1 // (op 4228) Store value

v_add_u32 v26, 8704, v25 // (op 4228) 
buffer_store_dword a10, v26, s[64:67], 0 offen sc1 // (op 4228) Store value

v_add_u32 v26, 8960, v25 // (op 4228) 
buffer_store_dword a11, v26, s[64:67], 0 offen sc1 // (op 4228) Store value




v_add_u32 v26, 8320, v25 // (op 4230) 
buffer_store_dword a12, v26, s[64:67], 0 offen sc1 // (op 4230) Store value

v_add_u32 v26, 8576, v25 // (op 4230) 
buffer_store_dword a13, v26, s[64:67], 0 offen sc1 // (op 4230) Store value

v_add_u32 v26, 8832, v25 // (op 4230) 
buffer_store_dword a14, v26, s[64:67], 0 offen sc1 // (op 4230) Store value

v_add_u32 v26, 9088, v25 // (op 4230) 
buffer_store_dword a15, v26, s[64:67], 0 offen sc1 // (op 4230) Store value




buffer_store_dword a0, v25, s[64:67], 0 offen sc1 // (op 311) Store value
buffer_store_dword a1, v25, s[64:67], 0 offen offset:256 sc1 // (op 311) Store value
buffer_store_dword a2, v25, s[64:67], 0 offen offset:512 sc1 // (op 311) Store value
buffer_store_dword a3, v25, s[64:67], 0 offen offset:768 sc1 // (op 311) Store value



buffer_store_dword a4, v25, s[64:67], 0 offen offset:128 sc1 // (op 4214) Store value
buffer_store_dword a5, v25, s[64:67], 0 offen offset:384 sc1 // (op 4214) Store value
buffer_store_dword a6, v25, s[64:67], 0 offen offset:640 sc1 // (op 4214) Store value
buffer_store_dword a7, v25, s[64:67], 0 offen offset:896 sc1 // (op 4214) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 315) 
v_cmp_eq_i32 vcc, v1, 0 // (op 319) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_319 // (op 319) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_319
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
s_branch RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_319 // (op 319) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_319
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_319:
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_319:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Send_Tile_314 // (op 314) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Send_Tile_314
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Send_Tile_314:
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Send_Tile_314:

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
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Receive_Tile_330 // (op 330) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Receive_Tile_330
s_mov_b32 s52, 0 // (op 296) call()
s_mul_i32 s53, s52, s8 // (op 295) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 295) 
s_mov_b32 s84, s54 // (op 295) convert

s_sub_u32 s54, s84, s19 // (op 295) 

s_sub_u32 s55, s54, 1 // (op 295) 

s_cmp_lt_u32 s53, s55 // (op 295) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_295 // (op 295) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_295 if false)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_ReceiveTileLoop_295:
s_add_u32 s53, s16, 1 // (op 10221) 
v_lshrrev_b32 v20, 5, v1 // (op 10221) 
s_add_u32 s54, s53, s52 // (op 10221) 
s_lshl_b32 s54, s54, 7 // (op 10221) 

v_lshlrev_b32 v21, 2, v20 // (op 10221) 

v_add_lshl_u32 v20, s54, v21, 6 // (op 10221) 


v_and_b32 v21, -32, v1 // (op 10221) 
v_sub_u32 v23, v1, v21 // (op 10221) 

v_add_lshl_u32 v21, v20, v23, 2 // (op 10221) 


s_lshl_b32 s53, s15, 15 // (op 10223) 
s_mov_b32 s64, s28 // (op 10223) reg expression
s_mov_b32 s65, s29 // (op 10223) reg expression
s_mov_b32 s66, s53 // (op 10223) reg expression
s_mov_b32 s67, 131072 // (op 10223) call()

RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_334:
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
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_334 // (op 334) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_334 if true)
s_barrier  // (op 338) 
v_cmp_eq_i32 vcc, v1, 0 // (op 337) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_337 // (op 337) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_337
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
s_branch RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_337 // (op 337) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_337
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_337:
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_337:




buffer_load_dword v28, v21, s[64:67], 0 offen // (op 349) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:256 // (op 349) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:512 // (op 349) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:768 // (op 349) Load value
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




buffer_load_dword v28, v21, s[64:67], 0 offen offset:128 // (op 4297) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:384 // (op 4297) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:640 // (op 4297) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:896 // (op 4297) Load value
v_accvgpr_read v32, a4 // (op 4298) reg expression
v_accvgpr_read v33, a5 // (op 4298) reg expression
v_accvgpr_read v34, a6 // (op 4298) reg expression
v_accvgpr_read v35, a7 // (op 4298) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4299) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4299) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4299) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4299) 

v_accvgpr_write a4, v32 // (op 4300) reg expression
v_accvgpr_write a5, v33 // (op 4300) reg expression
v_accvgpr_write a6, v34 // (op 4300) reg expression
v_accvgpr_write a7, v35 // (op 4300) reg expression




v_mov_b32 v20, 8192 // (op 4322) 
v_add_i32 v23, v21, v20 // (op 4322) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4322) Load value

v_mov_b32 v20, 8448 // (op 4322) 
v_add_i32 v23, v21, v20 // (op 4322) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4322) Load value

v_mov_b32 v20, 8704 // (op 4322) 
v_add_i32 v23, v21, v20 // (op 4322) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4322) Load value

v_mov_b32 v20, 8960 // (op 4322) 
v_add_i32 v23, v21, v20 // (op 4322) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4322) Load value

v_accvgpr_read v32, a8 // (op 4319) reg expression
v_accvgpr_read v33, a9 // (op 4319) reg expression
v_accvgpr_read v34, a10 // (op 4319) reg expression
v_accvgpr_read v35, a11 // (op 4319) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4320) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4320) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4320) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4320) 

v_accvgpr_write a8, v32 // (op 4321) reg expression
v_accvgpr_write a9, v33 // (op 4321) reg expression
v_accvgpr_write a10, v34 // (op 4321) reg expression
v_accvgpr_write a11, v35 // (op 4321) reg expression




v_mov_b32 v20, 8320 // (op 4327) 
v_add_i32 v23, v21, v20 // (op 4327) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4327) Load value

v_mov_b32 v20, 8576 // (op 4327) 
v_add_i32 v23, v21, v20 // (op 4327) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4327) Load value

v_mov_b32 v20, 8832 // (op 4327) 
v_add_i32 v23, v21, v20 // (op 4327) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4327) Load value

v_mov_b32 v20, 9088 // (op 4327) 
v_add_i32 v23, v21, v20 // (op 4327) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4327) Load value

v_accvgpr_read v32, a12 // (op 4324) reg expression
v_accvgpr_read v33, a13 // (op 4324) reg expression
v_accvgpr_read v34, a14 // (op 4324) reg expression
v_accvgpr_read v35, a15 // (op 4324) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4325) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4325) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4325) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4325) 

v_accvgpr_write a12, v32 // (op 4326) reg expression
v_accvgpr_write a13, v33 // (op 4326) reg expression
v_accvgpr_write a14, v34 // (op 4326) reg expression
v_accvgpr_write a15, v35 // (op 4326) reg expression




v_mov_b32 v20, 16384 // (op 4348) 
v_add_i32 v23, v21, v20 // (op 4348) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4348) Load value

v_mov_b32 v20, 16640 // (op 4348) 
v_add_i32 v23, v21, v20 // (op 4348) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4348) Load value

v_mov_b32 v20, 16896 // (op 4348) 
v_add_i32 v23, v21, v20 // (op 4348) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4348) Load value

v_mov_b32 v20, 17152 // (op 4348) 
v_add_i32 v23, v21, v20 // (op 4348) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4348) Load value

v_accvgpr_read v32, a16 // (op 4345) reg expression
v_accvgpr_read v33, a17 // (op 4345) reg expression
v_accvgpr_read v34, a18 // (op 4345) reg expression
v_accvgpr_read v35, a19 // (op 4345) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4346) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4346) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4346) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4346) 

v_accvgpr_write a16, v32 // (op 4347) reg expression
v_accvgpr_write a17, v33 // (op 4347) reg expression
v_accvgpr_write a18, v34 // (op 4347) reg expression
v_accvgpr_write a19, v35 // (op 4347) reg expression




v_mov_b32 v20, 16512 // (op 4353) 
v_add_i32 v23, v21, v20 // (op 4353) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4353) Load value

v_mov_b32 v20, 16768 // (op 4353) 
v_add_i32 v23, v21, v20 // (op 4353) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4353) Load value

v_mov_b32 v20, 17024 // (op 4353) 
v_add_i32 v23, v21, v20 // (op 4353) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4353) Load value

v_mov_b32 v20, 17280 // (op 4353) 
v_add_i32 v23, v21, v20 // (op 4353) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4353) Load value

v_accvgpr_read v32, a20 // (op 4350) reg expression
v_accvgpr_read v33, a21 // (op 4350) reg expression
v_accvgpr_read v34, a22 // (op 4350) reg expression
v_accvgpr_read v35, a23 // (op 4350) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4351) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4351) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4351) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4351) 

v_accvgpr_write a20, v32 // (op 4352) reg expression
v_accvgpr_write a21, v33 // (op 4352) reg expression
v_accvgpr_write a22, v34 // (op 4352) reg expression
v_accvgpr_write a23, v35 // (op 4352) reg expression




v_mov_b32 v20, 24576 // (op 4374) 
v_add_i32 v23, v21, v20 // (op 4374) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4374) Load value

v_mov_b32 v20, 24832 // (op 4374) 
v_add_i32 v23, v21, v20 // (op 4374) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4374) Load value

v_mov_b32 v20, 25088 // (op 4374) 
v_add_i32 v23, v21, v20 // (op 4374) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4374) Load value

v_mov_b32 v20, 25344 // (op 4374) 
v_add_i32 v23, v21, v20 // (op 4374) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4374) Load value

v_accvgpr_read v32, a24 // (op 4371) reg expression
v_accvgpr_read v33, a25 // (op 4371) reg expression
v_accvgpr_read v34, a26 // (op 4371) reg expression
v_accvgpr_read v35, a27 // (op 4371) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4372) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4372) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4372) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4372) 

v_accvgpr_write a24, v32 // (op 4373) reg expression
v_accvgpr_write a25, v33 // (op 4373) reg expression
v_accvgpr_write a26, v34 // (op 4373) reg expression
v_accvgpr_write a27, v35 // (op 4373) reg expression




v_mov_b32 v20, 24704 // (op 4379) 
v_add_i32 v23, v21, v20 // (op 4379) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 4379) Load value

v_mov_b32 v20, 24960 // (op 4379) 
v_add_i32 v23, v21, v20 // (op 4379) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 4379) Load value

v_mov_b32 v20, 25216 // (op 4379) 
v_add_i32 v23, v21, v20 // (op 4379) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 4379) Load value

v_mov_b32 v20, 25472 // (op 4379) 
v_add_i32 v23, v21, v20 // (op 4379) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 4379) Load value

v_accvgpr_read v32, a28 // (op 4376) reg expression
v_accvgpr_read v33, a29 // (op 4376) reg expression
v_accvgpr_read v34, a30 // (op 4376) reg expression
v_accvgpr_read v35, a31 // (op 4376) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 4377) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 4377) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 4377) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 4377) 

v_accvgpr_write a28, v32 // (op 4378) reg expression
v_accvgpr_write a29, v33 // (op 4378) reg expression
v_accvgpr_write a30, v34 // (op 4378) reg expression
v_accvgpr_write a31, v35 // (op 4378) reg expression

s_add_u32 s52, s52, 1 // (op 297) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 295) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 295) 
s_mov_b32 s84, s54 // (op 295) convert

s_sub_u32 s54, s84, s19 // (op 295) 

s_sub_u32 s55, s54, 1 // (op 295) 

s_cmp_lt_u32 s53, s55 // (op 295) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_ReceiveTileLoop_295 // (op 295) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_ReceiveTileLoop_295 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_295:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Receive_Tile_330 // (op 330) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Receive_Tile_330
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Receive_Tile_330:
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Receive_Tile_330:

s_cmp_eq_i32 s18, 0 // (op 373) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_373 // (op 373) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_373
v_cmp_eq_f32 vcc, 0.00000, v18 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_0_beta_133
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


v_accvgpr_read v28, a4 // (op 4425) 
v_accvgpr_read v29, a5 // (op 4425) 
v_accvgpr_read v30, a6 // (op 4425) 
v_accvgpr_read v31, a7 // (op 4425) 
v_mul_f32 v32, v19, v28 // (op 4425) 
v_mul_f32 v33, v19, v29 // (op 4425) 
v_mul_f32 v34, v19, v30 // (op 4425) 
v_mul_f32 v35, v19, v31 // (op 4425) 



s_lshl_b32 s52, s24, 7 // (op 4427) 
v_add_u32 v20, v27, s52 // (op 4427) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 4427) Store value


v_accvgpr_read v28, a8 // (op 4449) 
v_accvgpr_read v29, a9 // (op 4449) 
v_accvgpr_read v30, a10 // (op 4449) 
v_accvgpr_read v31, a11 // (op 4449) 
v_mul_f32 v32, v19, v28 // (op 4449) 
v_mul_f32 v33, v19, v29 // (op 4449) 
v_mul_f32 v34, v19, v30 // (op 4449) 
v_mul_f32 v35, v19, v31 // (op 4449) 



v_add_u32 v20, 128, v27 // (op 4452) 

buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 4452) Store value


v_accvgpr_read v28, a12 // (op 4458) 
v_accvgpr_read v29, a13 // (op 4458) 
v_accvgpr_read v30, a14 // (op 4458) 
v_accvgpr_read v31, a15 // (op 4458) 
v_mul_f32 v32, v19, v28 // (op 4458) 
v_mul_f32 v33, v19, v29 // (op 4458) 
v_mul_f32 v34, v19, v30 // (op 4458) 
v_mul_f32 v35, v19, v31 // (op 4458) 



s_lshl_b32 s52, s24, 7 // (op 4454) 
s_add_u32 s52, s52, 128 // (op 4454) 
v_add_u32 v20, v27, s52 // (op 4454) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 4454) Store value


v_accvgpr_read v28, a16 // (op 4485) 
v_accvgpr_read v29, a17 // (op 4485) 
v_accvgpr_read v30, a18 // (op 4485) 
v_accvgpr_read v31, a19 // (op 4485) 
v_mul_f32 v32, v19, v28 // (op 4485) 
v_mul_f32 v33, v19, v29 // (op 4485) 
v_mul_f32 v34, v19, v30 // (op 4485) 
v_mul_f32 v35, v19, v31 // (op 4485) 



v_add_u32 v20, 256, v27 // (op 4488) 

buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 4488) Store value


v_accvgpr_read v28, a20 // (op 4494) 
v_accvgpr_read v29, a21 // (op 4494) 
v_accvgpr_read v30, a22 // (op 4494) 
v_accvgpr_read v31, a23 // (op 4494) 
v_mul_f32 v32, v19, v28 // (op 4494) 
v_mul_f32 v33, v19, v29 // (op 4494) 
v_mul_f32 v34, v19, v30 // (op 4494) 
v_mul_f32 v35, v19, v31 // (op 4494) 



s_lshl_b32 s52, s24, 7 // (op 4490) 
s_add_u32 s52, s52, 256 // (op 4490) 
v_add_u32 v20, v27, s52 // (op 4490) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 4490) Store value


v_accvgpr_read v28, a24 // (op 4521) 
v_accvgpr_read v29, a25 // (op 4521) 
v_accvgpr_read v30, a26 // (op 4521) 
v_accvgpr_read v31, a27 // (op 4521) 
v_mul_f32 v32, v19, v28 // (op 4521) 
v_mul_f32 v33, v19, v29 // (op 4521) 
v_mul_f32 v34, v19, v30 // (op 4521) 
v_mul_f32 v35, v19, v31 // (op 4521) 



v_add_u32 v20, 384, v27 // (op 4524) 

buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 4524) Store value


v_accvgpr_read v28, a28 // (op 4530) 
v_accvgpr_read v29, a29 // (op 4530) 
v_accvgpr_read v30, a30 // (op 4530) 
v_accvgpr_read v31, a31 // (op 4530) 
v_mul_f32 v32, v19, v28 // (op 4530) 
v_mul_f32 v33, v19, v29 // (op 4530) 
v_mul_f32 v34, v19, v30 // (op 4530) 
v_mul_f32 v35, v19, v31 // (op 4530) 



s_lshl_b32 s52, s24, 7 // (op 4526) 
s_add_u32 s52, s52, 384 // (op 4526) 
v_add_u32 v20, v27, s52 // (op 4526) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 4526) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_0_beta_133
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_0_beta_133:


v_add_u32 v20, 384, v22 // (op 4708) 

buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 4708) Load value

v_accvgpr_read v32, a24 // (op 4700) 
v_accvgpr_read v33, a25 // (op 4700) 
v_accvgpr_read v34, a26 // (op 4700) 
v_accvgpr_read v35, a27 // (op 4700) 
v_mul_f32 v36, v19, v32 // (op 4700) 
v_mul_f32 v37, v19, v33 // (op 4700) 
v_mul_f32 v38, v19, v34 // (op 4700) 
v_mul_f32 v39, v19, v35 // (op 4700) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 4707) 
v_mul_f32 v33, v18, v29 // (op 4707) 
v_mul_f32 v34, v18, v30 // (op 4707) 
v_mul_f32 v35, v18, v31 // (op 4707) 

v_add_f32 v28, v32, v36 // (op 4701) 
v_add_f32 v29, v33, v37 // (op 4701) 
v_add_f32 v30, v34, v38 // (op 4701) 
v_add_f32 v31, v35, v39 // (op 4701) 




v_add_u32 v20, 384, v27 // (op 4703) 

buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 4703) Store value




s_lshl_b32 s52, s4, 7 // (op 4712) 
s_add_u32 s52, s52, 384 // (op 4712) 
v_add_u32 v20, v22, s52 // (op 4712) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 4712) Load value

v_accvgpr_read v32, a28 // (op 4714) 
v_accvgpr_read v33, a29 // (op 4714) 
v_accvgpr_read v34, a30 // (op 4714) 
v_accvgpr_read v35, a31 // (op 4714) 
v_mul_f32 v36, v19, v32 // (op 4714) 
v_mul_f32 v37, v19, v33 // (op 4714) 
v_mul_f32 v38, v19, v34 // (op 4714) 
v_mul_f32 v39, v19, v35 // (op 4714) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 4710) 
v_mul_f32 v33, v18, v29 // (op 4710) 
v_mul_f32 v34, v18, v30 // (op 4710) 
v_mul_f32 v35, v18, v31 // (op 4710) 

v_add_f32 v28, v32, v36 // (op 4711) 
v_add_f32 v29, v33, v37 // (op 4711) 
v_add_f32 v30, v34, v38 // (op 4711) 
v_add_f32 v31, v35, v39 // (op 4711) 




s_lshl_b32 s52, s24, 7 // (op 4705) 
s_add_u32 s52, s52, 384 // (op 4705) 
v_add_u32 v20, v27, s52 // (op 4705) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 4705) Store value




v_add_u32 v20, 256, v22 // (op 4667) 

s_nop 0
buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 4667) Load value

v_accvgpr_read v32, a16 // (op 4659) 
v_accvgpr_read v33, a17 // (op 4659) 
v_accvgpr_read v34, a18 // (op 4659) 
v_accvgpr_read v35, a19 // (op 4659) 
v_mul_f32 v36, v19, v32 // (op 4659) 
v_mul_f32 v37, v19, v33 // (op 4659) 
v_mul_f32 v38, v19, v34 // (op 4659) 
v_mul_f32 v39, v19, v35 // (op 4659) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 4666) 
v_mul_f32 v33, v18, v29 // (op 4666) 
v_mul_f32 v34, v18, v30 // (op 4666) 
v_mul_f32 v35, v18, v31 // (op 4666) 

v_add_f32 v28, v32, v36 // (op 4660) 
v_add_f32 v29, v33, v37 // (op 4660) 
v_add_f32 v30, v34, v38 // (op 4660) 
v_add_f32 v31, v35, v39 // (op 4660) 




v_add_u32 v20, 256, v27 // (op 4662) 

buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 4662) Store value




s_lshl_b32 s52, s4, 7 // (op 4671) 
s_add_u32 s52, s52, 256 // (op 4671) 
v_add_u32 v20, v22, s52 // (op 4671) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 4671) Load value

v_accvgpr_read v32, a20 // (op 4673) 
v_accvgpr_read v33, a21 // (op 4673) 
v_accvgpr_read v34, a22 // (op 4673) 
v_accvgpr_read v35, a23 // (op 4673) 
v_mul_f32 v36, v19, v32 // (op 4673) 
v_mul_f32 v37, v19, v33 // (op 4673) 
v_mul_f32 v38, v19, v34 // (op 4673) 
v_mul_f32 v39, v19, v35 // (op 4673) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 4669) 
v_mul_f32 v33, v18, v29 // (op 4669) 
v_mul_f32 v34, v18, v30 // (op 4669) 
v_mul_f32 v35, v18, v31 // (op 4669) 

v_add_f32 v28, v32, v36 // (op 4670) 
v_add_f32 v29, v33, v37 // (op 4670) 
v_add_f32 v30, v34, v38 // (op 4670) 
v_add_f32 v31, v35, v39 // (op 4670) 




s_lshl_b32 s52, s24, 7 // (op 4664) 
s_add_u32 s52, s52, 256 // (op 4664) 
v_add_u32 v20, v27, s52 // (op 4664) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 4664) Store value




v_add_u32 v20, 128, v22 // (op 4626) 

s_nop 0
buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 4626) Load value

v_accvgpr_read v32, a8 // (op 4618) 
v_accvgpr_read v33, a9 // (op 4618) 
v_accvgpr_read v34, a10 // (op 4618) 
v_accvgpr_read v35, a11 // (op 4618) 
v_mul_f32 v36, v19, v32 // (op 4618) 
v_mul_f32 v37, v19, v33 // (op 4618) 
v_mul_f32 v38, v19, v34 // (op 4618) 
v_mul_f32 v39, v19, v35 // (op 4618) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 4625) 
v_mul_f32 v33, v18, v29 // (op 4625) 
v_mul_f32 v34, v18, v30 // (op 4625) 
v_mul_f32 v35, v18, v31 // (op 4625) 

v_add_f32 v28, v32, v36 // (op 4619) 
v_add_f32 v29, v33, v37 // (op 4619) 
v_add_f32 v30, v34, v38 // (op 4619) 
v_add_f32 v31, v35, v39 // (op 4619) 




v_add_u32 v20, 128, v27 // (op 4621) 

buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 4621) Store value




s_lshl_b32 s52, s4, 7 // (op 4630) 
s_add_u32 s52, s52, 128 // (op 4630) 
v_add_u32 v20, v22, s52 // (op 4630) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 4630) Load value

v_accvgpr_read v32, a12 // (op 4632) 
v_accvgpr_read v33, a13 // (op 4632) 
v_accvgpr_read v34, a14 // (op 4632) 
v_accvgpr_read v35, a15 // (op 4632) 
v_mul_f32 v36, v19, v32 // (op 4632) 
v_mul_f32 v37, v19, v33 // (op 4632) 
v_mul_f32 v38, v19, v34 // (op 4632) 
v_mul_f32 v39, v19, v35 // (op 4632) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 4628) 
v_mul_f32 v33, v18, v29 // (op 4628) 
v_mul_f32 v34, v18, v30 // (op 4628) 
v_mul_f32 v35, v18, v31 // (op 4628) 

v_add_f32 v28, v32, v36 // (op 4629) 
v_add_f32 v29, v33, v37 // (op 4629) 
v_add_f32 v30, v34, v38 // (op 4629) 
v_add_f32 v31, v35, v39 // (op 4629) 




s_lshl_b32 s52, s24, 7 // (op 4623) 
s_add_u32 s52, s52, 128 // (op 4623) 
v_add_u32 v20, v27, s52 // (op 4623) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 4623) Store value




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




s_lshl_b32 s52, s4, 7 // (op 4588) 
v_add_u32 v20, v22, s52 // (op 4588) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 4588) Load value

v_accvgpr_read v32, a4 // (op 4585) 
v_accvgpr_read v33, a5 // (op 4585) 
v_accvgpr_read v34, a6 // (op 4585) 
v_accvgpr_read v35, a7 // (op 4585) 
v_mul_f32 v36, v19, v32 // (op 4585) 
v_mul_f32 v37, v19, v33 // (op 4585) 
v_mul_f32 v38, v19, v34 // (op 4585) 
v_mul_f32 v39, v19, v35 // (op 4585) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 4589) 
v_mul_f32 v33, v18, v29 // (op 4589) 
v_mul_f32 v34, v18, v30 // (op 4589) 
v_mul_f32 v35, v18, v31 // (op 4589) 

v_add_f32 v28, v32, v36 // (op 4586) 
v_add_f32 v29, v33, v37 // (op 4586) 
v_add_f32 v30, v34, v38 // (op 4586) 
v_add_f32 v31, v35, v39 // (op 4586) 




s_lshl_b32 s52, s24, 7 // (op 4587) 
v_add_u32 v20, v27, s52 // (op 4587) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 4587) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_0_beta_133:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_373 // (op 373) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_373
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_373:
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_373:


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
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_SKStreamTileLoop_286 // (op 286) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_SKStreamTileLoop_286 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_286:

















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
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_441 // (op 441) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_441 if false)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_DPStreamTileLoop_441:
v_lshrrev_b32 v0, 7, v1 // (op 9511) 
s_mul_i32 s15, s16, s50 // (op 9511) 
s_add_u32 s15, s15, s0 // (op 9511) 
v_and_b32 v3, 63, v1 // (op 9511) 
s_add_u32 s6, s15, s9 // (op 9511) 

v_lshrrev_b32 v4, 4, v3 // (op 9511) 
s_mul_hi_u32 s15, s6, s13 // (op 9511) 
v_lshlrev_b32 v5, 2, v4 // (op 9511) 

s_sub_u32 s7, s6, s15 // (op 9511) 
v_lshl_add_u32 v4, v0, 4, v5 // (op 9511) 


s_lshr_b32 s8, s7, 1 // (op 9511) 

v_lshrrev_b32 v0, 2, v3 // (op 9511) 

s_add_u32 s7, s8, s15 // (op 9511) 


v_and_b32 v3, 3, v0 // (op 9511) 

s_lshr_b32 s8, s7, s11 // (op 9511) 

v_and_b32 v0, 3, v1 // (op 9511) 
s_and_b32 s7, s11, 2147483648 // (op 9511) 
v_lshl_add_u32 v5, v3, 2, v0 // (op 9511) 


s_cmp_eq_i32 s7, 0 // (op 9511) 
s_mov_b32 s15, scc // (op 9511) 

v_lshrrev_b32 v0, 6, v1 // (op 9511) 
s_cmp_eq_i32 s15, 0 // (op 9511) 
s_cselect_b32 s7, s6, s8 // (op 9511) 



v_and_b32 v3, 1, v0 // (op 9511) 

s_mul_hi_u32 s6, s7, s12 // (op 9511) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 9511) 


s_sub_u32 s8, s7, s6 // (op 9511) 
s_lshr_b32 s15, s8, 1 // (op 9511) 

s_add_u32 s8, s15, s6 // (op 9511) 


s_lshr_b32 s6, s8, s10 // (op 9511) 

s_and_b32 s8, s10, 2147483648 // (op 9511) 
s_cmp_eq_i32 s8, 0 // (op 9511) 
s_mov_b32 s15, scc // (op 9511) 

s_cmp_eq_i32 s15, 0 // (op 9511) 
s_cselect_b32 s8, s7, s6 // (op 9511) 


s_mul_i32 s6, s8, s14 // (op 9511) 
v_lshl_add_u32 v3, s8, 7, v4 // (op 9511) 


s_sub_u32 s8, s7, s6 // (op 9511) 


v_lshl_add_u32 v4, s8, 6, v0 // (op 9511) 


v_mul_lo_u32 v0, v4, s4 // (op 9511) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 9511) 


s_mul_i32 s19, s4, s23 // (op 9513) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 9513) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 9513) least significant
s_mul_hi_u32 s6, s4, s22 // (op 9513) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9513) most significant: sum
s_add_u32 s19, s19, s6 // (op 9513) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 9513) 

s_mov_b32 s56, s20 // (op 9513) reg expression
s_mov_b32 s57, s21 // (op 9513) reg expression
s_mov_b32 s58, s28 // (op 9513) reg expression
s_mov_b32 s59, 131072 // (op 9513) call()

v_lshrrev_b32 v0, 7, v1 // (op 9865) 
s_mul_i32 s15, s16, s50 // (op 9865) 
s_add_u32 s15, s15, s0 // (op 9865) 
v_and_b32 v3, 63, v1 // (op 9865) 
s_add_u32 s6, s15, s9 // (op 9865) 

v_lshrrev_b32 v5, 4, v3 // (op 9865) 
s_mul_hi_u32 s15, s6, s13 // (op 9865) 
v_lshlrev_b32 v14, 2, v5 // (op 9865) 

s_sub_u32 s7, s6, s15 // (op 9865) 
v_lshl_add_u32 v5, v0, 4, v14 // (op 9865) 


s_lshr_b32 s8, s7, 1 // (op 9865) 

v_lshrrev_b32 v0, 2, v3 // (op 9865) 

s_add_u32 s7, s8, s15 // (op 9865) 


v_and_b32 v3, 3, v0 // (op 9865) 

s_lshr_b32 s8, s7, s11 // (op 9865) 

v_and_b32 v0, 3, v1 // (op 9865) 
s_and_b32 s7, s11, 2147483648 // (op 9865) 
v_lshl_add_u32 v14, v3, 2, v0 // (op 9865) 


s_cmp_eq_i32 s7, 0 // (op 9865) 
s_mov_b32 s15, scc // (op 9865) 

v_lshrrev_b32 v0, 6, v1 // (op 9865) 
s_cmp_eq_i32 s15, 0 // (op 9865) 
s_cselect_b32 s7, s6, s8 // (op 9865) 



v_and_b32 v3, 1, v0 // (op 9865) 

s_mul_hi_u32 s6, s7, s12 // (op 9865) 
v_lshl_add_u32 v0, v3, 4, v14 // (op 9865) 


s_sub_u32 s8, s7, s6 // (op 9865) 
s_lshr_b32 s15, s8, 1 // (op 9865) 

s_add_u32 s8, s15, s6 // (op 9865) 


s_lshr_b32 s6, s8, s10 // (op 9865) 

s_and_b32 s8, s10, 2147483648 // (op 9865) 
s_cmp_eq_i32 s8, 0 // (op 9865) 
s_mov_b32 s15, scc // (op 9865) 

s_cmp_eq_i32 s15, 0 // (op 9865) 
s_cselect_b32 s8, s7, s6 // (op 9865) 


s_mul_i32 s6, s8, s14 // (op 9865) 
v_lshl_add_u32 v3, s8, 7, v5 // (op 9865) 


s_sub_u32 s8, s7, s6 // (op 9865) 


v_lshl_add_u32 v5, s8, 6, v0 // (op 9865) 


v_mul_lo_u32 v0, v5, s24 // (op 9865) 

v_add_lshl_u32 v5, v3, v0, 2 // (op 9865) 


s_mul_i32 s19, s24, s23 // (op 9867) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 9867) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 9867) least significant
s_mul_hi_u32 s6, s24, s22 // (op 9867) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9867) most significant: sum
s_add_u32 s19, s19, s6 // (op 9867) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 9867) 

s_mov_b32 s60, s26 // (op 9867) reg expression
s_mov_b32 s61, s27 // (op 9867) reg expression
s_mov_b32 s62, s28 // (op 9867) reg expression
s_mov_b32 s63, 131072 // (op 9867) call()

v_lshrrev_b32 v0, 7, v1 // (op 9944) 
v_and_b32 v3, 15, v1 // (op 9944) 
v_lshl_add_u32 v14, v0, 4, v3 // (op 9944) 


v_and_b32 v0, 48, v1 // (op 9944) 
v_lshl_add_u32 v3, v14, 7, v0 // (op 9944) 


v_lshrrev_b32 v0, 6, v1 // (op 10019) 
v_and_b32 v14, 1, v0 // (op 10019) 

v_and_b32 v0, 15, v1 // (op 10019) 
v_lshl_add_u32 v15, v14, 4, v0 // (op 10019) 


v_and_b32 v0, 48, v1 // (op 10019) 
v_lshl_add_u32 v14, v15, 7, v0 // (op 10019) 


v_lshrrev_b32 v0, 7, v1 // (op 10095) 
v_and_b32 v15, 15, v1 // (op 10095) 
v_lshl_add_u32 v16, v0, 4, v15 // (op 10095) 


v_and_b32 v0, 63, v1 // (op 10095) 
v_lshrrev_b32 v15, 4, v0 // (op 10095) 

v_lshl_add_u32 v0, v16, 2, v15 // (op 10095) 


v_lshrrev_b32 v15, 6, v1 // (op 10171) 
v_and_b32 v16, 1, v15 // (op 10171) 

v_and_b32 v15, 15, v1 // (op 10171) 
v_lshl_add_u32 v17, v16, 4, v15 // (op 10171) 


v_and_b32 v15, 63, v1 // (op 10171) 
v_lshrrev_b32 v16, 4, v15 // (op 10171) 

v_lshl_add_u32 v15, v17, 2, v16 // (op 10171) 


v_accvgpr_write a0, 0.00000 // (op 180) call()
v_accvgpr_write a1, 0.00000 // (op 180) call()
v_accvgpr_write a2, 0.00000 // (op 180) call()
v_accvgpr_write a3, 0.00000 // (op 180) call()
v_accvgpr_write a4, 0.00000 // (op 614) call()
v_accvgpr_write a5, 0.00000 // (op 614) call()
v_accvgpr_write a6, 0.00000 // (op 614) call()
v_accvgpr_write a7, 0.00000 // (op 614) call()
v_accvgpr_write a8, 0.00000 // (op 834) call()
v_accvgpr_write a9, 0.00000 // (op 834) call()
v_accvgpr_write a10, 0.00000 // (op 834) call()
v_accvgpr_write a11, 0.00000 // (op 834) call()
v_accvgpr_write a12, 0.00000 // (op 989) call()
v_accvgpr_write a13, 0.00000 // (op 989) call()
v_accvgpr_write a14, 0.00000 // (op 989) call()
v_accvgpr_write a15, 0.00000 // (op 989) call()
v_accvgpr_write a16, 0.00000 // (op 1220) call()
v_accvgpr_write a17, 0.00000 // (op 1220) call()
v_accvgpr_write a18, 0.00000 // (op 1220) call()
v_accvgpr_write a19, 0.00000 // (op 1220) call()
v_accvgpr_write a20, 0.00000 // (op 1375) call()
v_accvgpr_write a21, 0.00000 // (op 1375) call()
v_accvgpr_write a22, 0.00000 // (op 1375) call()
v_accvgpr_write a23, 0.00000 // (op 1375) call()
v_accvgpr_write a24, 0.00000 // (op 1606) call()
v_accvgpr_write a25, 0.00000 // (op 1606) call()
v_accvgpr_write a26, 0.00000 // (op 1606) call()
v_accvgpr_write a27, 0.00000 // (op 1606) call()
v_accvgpr_write a28, 0.00000 // (op 1761) call()
v_accvgpr_write a29, 0.00000 // (op 1761) call()
v_accvgpr_write a30, 0.00000 // (op 1761) call()
v_accvgpr_write a31, 0.00000 // (op 1761) call()
v_lshrrev_b32 v16, 3, v1 // (op 8999) 
s_mul_i32 s15, s16, s50 // (op 8999) 
s_add_u32 s15, s15, s0 // (op 8999) 
v_and_b32 v17, -8, v1 // (op 8999) 
s_add_u32 s6, s15, s9 // (op 8999) 

v_sub_u32 v20, v1, v17 // (op 8999) 

s_mul_hi_u32 s15, s6, s13 // (op 8999) 
v_lshlrev_b32 v17, 4, v20 // (op 8999) 

s_sub_u32 s7, s6, s15 // (op 8999) 
s_lshr_b32 s8, s7, 1 // (op 8999) 

s_add_u32 s7, s8, s15 // (op 8999) 


s_lshr_b32 s8, s7, s11 // (op 8999) 

s_and_b32 s7, s11, 2147483648 // (op 8999) 
s_cmp_eq_i32 s7, 0 // (op 8999) 
s_mov_b32 s15, scc // (op 8999) 

s_cmp_eq_i32 s15, 0 // (op 8999) 
s_cselect_b32 s7, s6, s8 // (op 8999) 


s_mul_hi_u32 s8, s7, s12 // (op 8999) 
s_sub_u32 s15, s7, s8 // (op 8999) 
s_lshr_b32 s18, s15, 1 // (op 8999) 

s_add_u32 s15, s18, s8 // (op 8999) 


s_lshr_b32 s8, s15, s10 // (op 8999) 

s_and_b32 s15, s10, 2147483648 // (op 8999) 
s_cmp_eq_i32 s15, 0 // (op 8999) 
s_mov_b32 s18, scc // (op 8999) 

s_cmp_eq_i32 s18, 0 // (op 8999) 
s_cselect_b32 s15, s7, s8 // (op 8999) 


v_lshl_add_u32 v20, s15, 7, v16 // (op 8999) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 8999) 
s_mul_i32 s8, s7, s18 // (op 8999) 


s_sub_u32 s7, s6, s8 // (op 8999) 


v_lshl_add_u32 v16, s7, 7, v17 // (op 8999) 


v_mov_b32 v23, 0 // (op 8999) convert
v_mov_b32 v22, v16 // (op 8999) convert

v_mul_lo_u32 v17, v20, s31 // (op 8999) most significant: low of low * high
v_mul_hi_u32 v21, v20, s30 // (op 8999) most significant: high of low * low
v_mul_lo_u32 v16, v20, s30 // (op 8999) least significant: low of low * low
v_add_u32 v17, v17, v21 // (op 8999) most significant: sum

v_add_co_u32 v16, s[18:19], v16, v22 // (op 8999) least significant half
v_addc_co_u32 v17, s[18:19], v17, v23, s[18:19] // (op 8999) most significant half



s_mul_i32 s19, s30, s35 // (op 9001) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 9001) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 9001) least significant
s_mul_hi_u32 s6, s30, s34 // (op 9001) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9001) most significant: sum
s_add_u32 s19, s19, s6 // (op 9001) most significant: sum


s_mov_b32 s68, s32 // (op 9001) reg expression
s_mov_b32 s69, s33 // (op 9001) reg expression
s_mov_b32 s70, s18 // (op 9001) reg expression
s_mov_b32 s71, 131072 // (op 9001) call()

v_lshrrev_b32 v20, 3, v1 // (op 9018) 
s_mul_i32 s15, s16, s50 // (op 9018) 
s_add_u32 s15, s15, s0 // (op 9018) 
v_and_b32 v21, -8, v1 // (op 9018) 
s_add_u32 s6, s15, s9 // (op 9018) 

v_sub_u32 v22, v1, v21 // (op 9018) 

s_mul_hi_u32 s15, s6, s13 // (op 9018) 
v_lshlrev_b32 v21, 4, v22 // (op 9018) 

s_sub_u32 s7, s6, s15 // (op 9018) 
s_lshr_b32 s8, s7, 1 // (op 9018) 

s_add_u32 s7, s8, s15 // (op 9018) 


s_lshr_b32 s8, s7, s11 // (op 9018) 

s_and_b32 s7, s11, 2147483648 // (op 9018) 
s_cmp_eq_i32 s7, 0 // (op 9018) 
s_mov_b32 s15, scc // (op 9018) 

s_cmp_eq_i32 s15, 0 // (op 9018) 
s_cselect_b32 s7, s6, s8 // (op 9018) 


s_mul_hi_u32 s8, s7, s12 // (op 9018) 
s_sub_u32 s15, s7, s8 // (op 9018) 
s_lshr_b32 s18, s15, 1 // (op 9018) 

s_add_u32 s15, s18, s8 // (op 9018) 


s_lshr_b32 s8, s15, s10 // (op 9018) 

s_and_b32 s15, s10, 2147483648 // (op 9018) 
s_cmp_eq_i32 s15, 0 // (op 9018) 
s_mov_b32 s18, scc // (op 9018) 

s_cmp_eq_i32 s18, 0 // (op 9018) 
s_cselect_b32 s15, s7, s8 // (op 9018) 


v_lshl_add_u32 v22, s15, 7, v20 // (op 9018) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 9018) 
s_mul_i32 s8, s7, s18 // (op 9018) 


s_sub_u32 s7, s6, s8 // (op 9018) 


v_lshl_add_u32 v20, s7, 7, v21 // (op 9018) 


v_mov_b32 v25, 0 // (op 9018) convert
v_mov_b32 v24, v20 // (op 9018) convert

v_mul_lo_u32 v21, v22, s31 // (op 9018) most significant: low of low * high
v_mul_hi_u32 v23, v22, s30 // (op 9018) most significant: high of low * low
v_mul_lo_u32 v20, v22, s30 // (op 9018) least significant: low of low * low
v_add_u32 v21, v21, v23 // (op 9018) most significant: sum

v_add_co_u32 v20, s[18:19], v20, v24 // (op 9018) least significant half
v_addc_co_u32 v21, s[18:19], v21, v25, s[18:19] // (op 9018) most significant half



s_mul_i32 s19, s30, s35 // (op 9020) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 9020) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 9020) least significant
s_mul_hi_u32 s6, s30, s34 // (op 9020) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9020) most significant: sum
s_add_u32 s19, s19, s6 // (op 9020) most significant: sum


s_mov_b32 s68, s32 // (op 9020) reg expression
s_mov_b32 s69, s33 // (op 9020) reg expression
s_mov_b32 s70, s18 // (op 9020) reg expression
s_mov_b32 s71, 131072 // (op 9020) call()

v_lshrrev_b32 v22, 3, v1 // (op 9137) 
s_mul_i32 s15, s16, s50 // (op 9137) 
s_add_u32 s15, s15, s0 // (op 9137) 
v_and_b32 v23, -8, v1 // (op 9137) 
s_add_u32 s6, s15, s9 // (op 9137) 

v_sub_u32 v24, v1, v23 // (op 9137) 

s_mul_hi_u32 s15, s6, s13 // (op 9137) 
v_lshlrev_b32 v23, 4, v24 // (op 9137) 

s_sub_u32 s7, s6, s15 // (op 9137) 
s_lshr_b32 s8, s7, 1 // (op 9137) 

s_add_u32 s7, s8, s15 // (op 9137) 


s_lshr_b32 s8, s7, s11 // (op 9137) 

s_and_b32 s7, s11, 2147483648 // (op 9137) 
s_cmp_eq_i32 s7, 0 // (op 9137) 
s_mov_b32 s15, scc // (op 9137) 

s_cmp_eq_i32 s15, 0 // (op 9137) 
s_cselect_b32 s7, s6, s8 // (op 9137) 


s_mul_hi_u32 s8, s7, s12 // (op 9137) 
s_sub_u32 s15, s7, s8 // (op 9137) 
s_lshr_b32 s18, s15, 1 // (op 9137) 

s_add_u32 s15, s18, s8 // (op 9137) 


s_lshr_b32 s8, s15, s10 // (op 9137) 

s_and_b32 s15, s10, 2147483648 // (op 9137) 
s_cmp_eq_i32 s15, 0 // (op 9137) 
s_mov_b32 s18, scc // (op 9137) 

s_cmp_eq_i32 s18, 0 // (op 9137) 
s_cselect_b32 s15, s7, s8 // (op 9137) 


s_mul_i32 s8, s15, s14 // (op 9137) 

s_sub_u32 s15, s7, s8 // (op 9137) 

v_lshl_add_u32 v24, s15, 6, v22 // (op 9137) 


v_mul_lo_u32 v27, v24, s37 // (op 9137) most significant: low of low * high
v_mul_hi_u32 v22, v24, s36 // (op 9137) most significant: high of low * low
v_mul_lo_u32 v26, v24, s36 // (op 9137) least significant: low of low * low
v_add_u32 v27, v27, v22 // (op 9137) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 9137) 
s_mul_i32 s8, s7, s18 // (op 9137) 


s_sub_u32 s7, s6, s8 // (op 9137) 


v_lshl_add_u32 v22, s7, 7, v23 // (op 9137) 


v_mov_b32 v25, 0 // (op 9137) convert
v_mov_b32 v24, v22 // (op 9137) convert

v_add_co_u32 v22, s[18:19], v24, v26 // (op 9137) least significant half
v_addc_co_u32 v23, s[18:19], v25, v27, s[18:19] // (op 9137) most significant half



s_mul_i32 s19, s36, s23 // (op 9139) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 9139) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 9139) least significant
s_mul_hi_u32 s6, s36, s22 // (op 9139) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9139) most significant: sum
s_add_u32 s19, s19, s6 // (op 9139) most significant: sum


s_mov_b32 s72, s38 // (op 9139) reg expression
s_mov_b32 s73, s39 // (op 9139) reg expression
s_mov_b32 s74, s18 // (op 9139) reg expression
s_mov_b32 s75, 131072 // (op 9139) call()

v_lshrrev_b32 v24, 3, v1 // (op 9156) 
s_mul_i32 s15, s16, s50 // (op 9156) 
s_add_u32 s15, s15, s0 // (op 9156) 
v_and_b32 v25, -8, v1 // (op 9156) 
s_add_u32 s6, s15, s9 // (op 9156) 

v_sub_u32 v26, v1, v25 // (op 9156) 

s_mul_hi_u32 s15, s6, s13 // (op 9156) 
v_lshlrev_b32 v25, 4, v26 // (op 9156) 

s_sub_u32 s7, s6, s15 // (op 9156) 
s_lshr_b32 s8, s7, 1 // (op 9156) 

s_add_u32 s7, s8, s15 // (op 9156) 


s_lshr_b32 s8, s7, s11 // (op 9156) 

s_and_b32 s7, s11, 2147483648 // (op 9156) 
s_cmp_eq_i32 s7, 0 // (op 9156) 
s_mov_b32 s15, scc // (op 9156) 

s_cmp_eq_i32 s15, 0 // (op 9156) 
s_cselect_b32 s7, s6, s8 // (op 9156) 


s_mul_hi_u32 s8, s7, s12 // (op 9156) 
s_sub_u32 s15, s7, s8 // (op 9156) 
s_lshr_b32 s18, s15, 1 // (op 9156) 

s_add_u32 s15, s18, s8 // (op 9156) 


s_lshr_b32 s8, s15, s10 // (op 9156) 

s_and_b32 s15, s10, 2147483648 // (op 9156) 
s_cmp_eq_i32 s15, 0 // (op 9156) 
s_mov_b32 s18, scc // (op 9156) 

s_cmp_eq_i32 s18, 0 // (op 9156) 
s_cselect_b32 s15, s7, s8 // (op 9156) 


s_mul_i32 s8, s15, s14 // (op 9156) 

s_sub_u32 s15, s7, s8 // (op 9156) 

v_lshl_add_u32 v26, s15, 6, v24 // (op 9156) 


v_mul_lo_u32 v29, v26, s37 // (op 9156) most significant: low of low * high
v_mul_hi_u32 v24, v26, s36 // (op 9156) most significant: high of low * low
v_mul_lo_u32 v28, v26, s36 // (op 9156) least significant: low of low * low
v_add_u32 v29, v29, v24 // (op 9156) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 9156) 
s_mul_i32 s8, s7, s18 // (op 9156) 


s_sub_u32 s7, s6, s8 // (op 9156) 


v_lshl_add_u32 v24, s7, 7, v25 // (op 9156) 


v_mov_b32 v27, 0 // (op 9156) convert
v_mov_b32 v26, v24 // (op 9156) convert

v_add_co_u32 v24, s[18:19], v26, v28 // (op 9156) least significant half
v_addc_co_u32 v25, s[18:19], v27, v29, s[18:19] // (op 9156) most significant half



s_mul_i32 s19, s36, s23 // (op 9158) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 9158) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 9158) least significant
s_mul_hi_u32 s6, s36, s22 // (op 9158) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9158) most significant: sum
s_add_u32 s19, s19, s6 // (op 9158) most significant: sum


s_mov_b32 s72, s38 // (op 9158) reg expression
s_mov_b32 s73, s39 // (op 9158) reg expression
s_mov_b32 s74, s18 // (op 9158) reg expression
s_mov_b32 s75, 131072 // (op 9158) call()

v_lshrrev_b32 v26, 2, v1 // (op 9265) 
s_mul_i32 s15, s16, s50 // (op 9265) 
s_add_u32 s15, s15, s0 // (op 9265) 
v_lshlrev_b32 v27, 1, v26 // (op 9265) 

s_add_u32 s6, s15, s9 // (op 9265) 

v_and_b32 v26, -4, v1 // (op 9265) 
s_mul_hi_u32 s15, s6, s13 // (op 9265) 
v_sub_u32 v28, v1, v26 // (op 9265) 

s_sub_u32 s7, s6, s15 // (op 9265) 
s_lshr_b32 s8, s7, 1 // (op 9265) 

s_add_u32 s7, s8, s15 // (op 9265) 


s_lshr_b32 s8, s7, s11 // (op 9265) 

s_and_b32 s7, s11, 2147483648 // (op 9265) 
s_cmp_eq_i32 s7, 0 // (op 9265) 
s_mov_b32 s15, scc // (op 9265) 

s_cmp_eq_i32 s15, 0 // (op 9265) 
s_cselect_b32 s7, s6, s8 // (op 9265) 


s_mul_hi_u32 s8, s7, s12 // (op 9265) 
s_sub_u32 s15, s7, s8 // (op 9265) 
s_lshr_b32 s18, s15, 1 // (op 9265) 

s_add_u32 s15, s18, s8 // (op 9265) 


s_lshr_b32 s8, s15, s10 // (op 9265) 

s_and_b32 s15, s10, 2147483648 // (op 9265) 
s_cmp_eq_i32 s15, 0 // (op 9265) 
s_mov_b32 s18, scc // (op 9265) 

s_cmp_eq_i32 s18, 0 // (op 9265) 
s_cselect_b32 s15, s7, s8 // (op 9265) 


v_lshl_add_u32 v26, s15, 7, v27 // (op 9265) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 9265) 
s_mul_i32 s8, s7, s18 // (op 9265) 


s_sub_u32 s7, s6, s8 // (op 9265) 


v_lshl_add_u32 v27, s7, 2, v28 // (op 9265) 


v_mul_lo_u32 v28, v26, s40 // (op 9265) 
v_add_u32 v28, v28, v27 // (op 9265) 


s_mul_i32 s19, s40, s35 // (op 9267) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 9267) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 9267) least significant
s_mul_hi_u32 s6, s40, s34 // (op 9267) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9267) most significant: sum
s_add_u32 s19, s19, s6 // (op 9267) most significant: sum


s_mov_b32 s76, s42 // (op 9267) reg expression
s_mov_b32 s77, s43 // (op 9267) reg expression
s_mov_b32 s78, s18 // (op 9267) reg expression
s_mov_b32 s79, 131072 // (op 9267) call()

v_lshrrev_b32 v26, 2, v1 // (op 9284) 
s_mul_i32 s15, s16, s50 // (op 9284) 
s_add_u32 s15, s15, s0 // (op 9284) 
v_lshlrev_b32 v27, 1, v26 // (op 9284) 

s_add_u32 s6, s15, s9 // (op 9284) 

v_and_b32 v26, -4, v1 // (op 9284) 
s_mul_hi_u32 s15, s6, s13 // (op 9284) 
v_sub_u32 v29, v1, v26 // (op 9284) 

s_sub_u32 s7, s6, s15 // (op 9284) 
s_lshr_b32 s8, s7, 1 // (op 9284) 

s_add_u32 s7, s8, s15 // (op 9284) 


s_lshr_b32 s8, s7, s11 // (op 9284) 

s_and_b32 s7, s11, 2147483648 // (op 9284) 
s_cmp_eq_i32 s7, 0 // (op 9284) 
s_mov_b32 s15, scc // (op 9284) 

s_cmp_eq_i32 s15, 0 // (op 9284) 
s_cselect_b32 s7, s6, s8 // (op 9284) 


s_mul_hi_u32 s8, s7, s12 // (op 9284) 
s_sub_u32 s15, s7, s8 // (op 9284) 
s_lshr_b32 s18, s15, 1 // (op 9284) 

s_add_u32 s15, s18, s8 // (op 9284) 


s_lshr_b32 s8, s15, s10 // (op 9284) 

s_and_b32 s15, s10, 2147483648 // (op 9284) 
s_cmp_eq_i32 s15, 0 // (op 9284) 
s_mov_b32 s18, scc // (op 9284) 

s_cmp_eq_i32 s18, 0 // (op 9284) 
s_cselect_b32 s15, s7, s8 // (op 9284) 


v_lshl_add_u32 v26, s15, 7, v27 // (op 9284) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 9284) 
s_mul_i32 s8, s7, s18 // (op 9284) 


s_sub_u32 s7, s6, s8 // (op 9284) 


v_lshl_add_u32 v27, s7, 2, v29 // (op 9284) 


v_mul_lo_u32 v29, v26, s40 // (op 9284) 
v_add_u32 v29, v29, v27 // (op 9284) 


s_mul_i32 s19, s40, s35 // (op 9286) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 9286) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 9286) least significant
s_mul_hi_u32 s6, s40, s34 // (op 9286) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9286) most significant: sum
s_add_u32 s19, s19, s6 // (op 9286) most significant: sum


s_mov_b32 s76, s42 // (op 9286) reg expression
s_mov_b32 s77, s43 // (op 9286) reg expression
s_mov_b32 s78, s18 // (op 9286) reg expression
s_mov_b32 s79, 131072 // (op 9286) call()

v_lshrrev_b32 v26, 2, v1 // (op 9393) 
s_mul_i32 s15, s16, s50 // (op 9393) 
s_add_u32 s15, s15, s0 // (op 9393) 
v_and_b32 v27, -4, v1 // (op 9393) 
s_add_u32 s6, s15, s9 // (op 9393) 

v_sub_u32 v30, v1, v27 // (op 9393) 

s_mul_hi_u32 s15, s6, s13 // (op 9393) 
s_sub_u32 s7, s6, s15 // (op 9393) 
s_lshr_b32 s8, s7, 1 // (op 9393) 

s_add_u32 s7, s8, s15 // (op 9393) 


s_lshr_b32 s8, s7, s11 // (op 9393) 

s_and_b32 s7, s11, 2147483648 // (op 9393) 
s_cmp_eq_i32 s7, 0 // (op 9393) 
s_mov_b32 s15, scc // (op 9393) 

s_cmp_eq_i32 s15, 0 // (op 9393) 
s_cselect_b32 s7, s6, s8 // (op 9393) 


s_mul_hi_u32 s8, s7, s12 // (op 9393) 
s_sub_u32 s15, s7, s8 // (op 9393) 
s_lshr_b32 s18, s15, 1 // (op 9393) 

s_add_u32 s15, s18, s8 // (op 9393) 


s_lshr_b32 s8, s15, s10 // (op 9393) 

s_and_b32 s15, s10, 2147483648 // (op 9393) 
s_cmp_eq_i32 s15, 0 // (op 9393) 
s_mov_b32 s18, scc // (op 9393) 

s_cmp_eq_i32 s18, 0 // (op 9393) 
s_cselect_b32 s15, s7, s8 // (op 9393) 


s_mul_i32 s8, s15, s14 // (op 9393) 

s_sub_u32 s15, s7, s8 // (op 9393) 

v_lshl_add_u32 v27, s15, 6, v26 // (op 9393) 


v_mul_lo_u32 v26, v27, s44 // (op 9393) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 9393) 
s_mul_i32 s8, s7, s18 // (op 9393) 


s_sub_u32 s7, s6, s8 // (op 9393) 


v_lshl_add_u32 v27, s7, 2, v30 // (op 9393) 


v_add_u32 v30, v27, v26 // (op 9393) 


s_mul_i32 s19, s44, s23 // (op 9395) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 9395) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 9395) least significant
s_mul_hi_u32 s6, s44, s22 // (op 9395) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9395) most significant: sum
s_add_u32 s19, s19, s6 // (op 9395) most significant: sum


s_mov_b32 s80, s46 // (op 9395) reg expression
s_mov_b32 s81, s47 // (op 9395) reg expression
s_mov_b32 s82, s18 // (op 9395) reg expression
s_mov_b32 s83, 131072 // (op 9395) call()

v_lshrrev_b32 v26, 2, v1 // (op 9412) 
s_mul_i32 s15, s16, s50 // (op 9412) 
s_add_u32 s15, s15, s0 // (op 9412) 
v_and_b32 v27, -4, v1 // (op 9412) 
s_add_u32 s6, s15, s9 // (op 9412) 

v_sub_u32 v31, v1, v27 // (op 9412) 

s_mul_hi_u32 s15, s6, s13 // (op 9412) 
s_sub_u32 s7, s6, s15 // (op 9412) 
s_lshr_b32 s8, s7, 1 // (op 9412) 

s_add_u32 s7, s8, s15 // (op 9412) 


s_lshr_b32 s8, s7, s11 // (op 9412) 

s_and_b32 s7, s11, 2147483648 // (op 9412) 
s_cmp_eq_i32 s7, 0 // (op 9412) 
s_mov_b32 s15, scc // (op 9412) 

s_cmp_eq_i32 s15, 0 // (op 9412) 
s_cselect_b32 s7, s6, s8 // (op 9412) 


s_mul_hi_u32 s8, s7, s12 // (op 9412) 
s_sub_u32 s15, s7, s8 // (op 9412) 
s_lshr_b32 s18, s15, 1 // (op 9412) 

s_add_u32 s15, s18, s8 // (op 9412) 


s_lshr_b32 s8, s15, s10 // (op 9412) 

s_and_b32 s15, s10, 2147483648 // (op 9412) 
s_cmp_eq_i32 s15, 0 // (op 9412) 
s_mov_b32 s18, scc // (op 9412) 

s_cmp_eq_i32 s18, 0 // (op 9412) 
s_cselect_b32 s15, s7, s8 // (op 9412) 


s_mul_i32 s8, s15, s14 // (op 9412) 

s_sub_u32 s15, s7, s8 // (op 9412) 

v_lshl_add_u32 v27, s15, 6, v26 // (op 9412) 


v_mul_lo_u32 v26, v27, s44 // (op 9412) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 9412) 
s_mul_i32 s8, s7, s18 // (op 9412) 


s_sub_u32 s7, s6, s8 // (op 9412) 


v_lshl_add_u32 v27, s7, 2, v31 // (op 9412) 


v_add_u32 v31, v27, v26 // (op 9412) 


s_mul_i32 s19, s44, s23 // (op 9414) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 9414) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 9414) least significant
s_mul_hi_u32 s6, s44, s22 // (op 9414) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9414) most significant: sum
s_add_u32 s19, s19, s6 // (op 9414) most significant: sum


s_mov_b32 s80, s46 // (op 9414) reg expression
s_mov_b32 s81, s47 // (op 9414) reg expression
s_mov_b32 s82, s18 // (op 9414) reg expression
s_mov_b32 s83, 131072 // (op 9414) call()

v_and_b32 v26, -8, v1 // (op 9985) 
v_sub_u32 v27, v1, v26 // (op 9985) 

v_lshlrev_b32 v26, 4, v27 // (op 9985) 

v_lshrrev_b32 v27, 3, v1 // (op 9985) 
v_lshl_add_u32 v32, v27, 7, v26 // (op 9985) 


s_nop 0
v_readfirstlane_b32 s15, v32 // (op 9985) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10786) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 8903) 
v_mov_b64 v[26:27], s[18:19] // (op 8903) convert


v_mov_b64 v[32:33], v[22:23] // (op 8903) 
s_mov_b32 m0, s15 // (op 8903) reg expression
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 8903) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v26 // (op 8903) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 8903) most significant half

s_add_u32 m0, m0, 4096 // (op 8903) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 8903) Load value direct to lds






buffer_load_ubyte v26, v30, s[80:83], 0 offen offset: 0 // (op 8451) Load value
v_and_b32 v27, -8, v1 // (op 9910) 
v_sub_u32 v32, v1, v27 // (op 9910) 

v_lshlrev_b32 v27, 4, v32 // (op 9910) 

v_lshrrev_b32 v32, 3, v1 // (op 9910) 
v_lshl_add_u32 v33, v32, 7, v27 // (op 9910) 


s_nop 0
v_readfirstlane_b32 s15, v33 // (op 9910) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10767) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 8908) 
v_mov_b64 v[32:33], s[18:19] // (op 8908) convert


v_mov_b64 v[34:35], v[16:17] // (op 8908) 
s_add_u32 s6, 8192, s15 // (op 8908) 
s_mov_b32 m0, s6 // (op 8908) reg expression
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 8908) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 8908) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 8908) most significant half

s_add_u32 m0, m0, 4096 // (op 8908) 
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 8908) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 8908) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 8908) most significant half

s_add_u32 m0, m0, 4096 // (op 8908) 
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 8908) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 8908) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 8908) most significant half

s_add_u32 m0, m0, 4096 // (op 8908) 
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 8908) Load value direct to lds





v_mov_b64 v[32:33], s[40:41] // (op 8457) convert

v_mov_b32 v27, v28 // (op 8457) 
buffer_load_ubyte v34, v27, s[76:79], 0 offen offset: 0 // (op 8457) Load value
v_add_u32 v27, v27, v32 // (op 8457) 
buffer_load_ubyte v35, v27, s[76:79], 0 offen offset: 0 // (op 8457) Load value


v_and_b32 v27, -8, v1 // (op 10295) 
v_sub_u32 v32, v1, v27 // (op 10295) 

v_lshlrev_b32 v27, 4, v32 // (op 10295) 

v_lshrrev_b32 v32, 3, v1 // (op 10295) 
v_lshl_add_u32 v33, v32, 7, v27 // (op 10295) 


s_nop 0
v_readfirstlane_b32 s15, v33 // (op 10295) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10857) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 8913) 
v_mov_b64 v[32:33], s[18:19] // (op 8913) convert


v_mov_b64 v[36:37], v[22:23] // (op 8913) 
s_add_u32 s6, 24576, s15 // (op 8913) 
s_mov_b32 m0, s6 // (op 8913) reg expression
s_mov_b32 s7, 128 // (op 8913) call()
buffer_load_dwordx4 v36, s[72:75], s7 offen offset: 0 lds // (op 8913) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 8913) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 8913) most significant half

s_add_u32 m0, m0, 4096 // (op 8913) 
buffer_load_dwordx4 v36, s[72:75], s7 offen offset: 0 lds // (op 8913) Load value direct to lds









buffer_load_ubyte v27, v30, s[80:83], 0 offen offset:4 // (op 8463) Load value

v_and_b32 v30, -8, v1 // (op 10251) 
v_sub_u32 v22, v1, v30 // (op 10251) 

v_lshlrev_b32 v23, 4, v22 // (op 10251) 

v_lshrrev_b32 v22, 3, v1 // (op 10251) 
v_lshl_add_u32 v30, v22, 7, v23 // (op 10251) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 10251) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 10847) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 8915) 
v_mov_b64 v[22:23], s[18:19] // (op 8915) convert


v_mov_b64 v[32:33], v[16:17] // (op 8915) 
s_add_u32 s6, 32768, s15 // (op 8915) 
s_mov_b32 m0, s6 // (op 8915) reg expression
s_mov_b32 s7, 128 // (op 8915) call()
buffer_load_dwordx4 v32, s[68:71], s7 offen offset: 0 lds // (op 8915) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v22 // (op 8915) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 8915) most significant half

s_add_u32 m0, m0, 4096 // (op 8915) 
buffer_load_dwordx4 v32, s[68:71], s7 offen offset: 0 lds // (op 8915) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v22 // (op 8915) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 8915) most significant half

s_add_u32 m0, m0, 4096 // (op 8915) 
buffer_load_dwordx4 v32, s[68:71], s7 offen offset: 0 lds // (op 8915) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v22 // (op 8915) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 8915) most significant half

s_add_u32 m0, m0, 4096 // (op 8915) 
buffer_load_dwordx4 v32, s[68:71], s7 offen offset: 0 lds // (op 8915) Load value direct to lds







v_mov_b64 v[16:17], s[40:41] // (op 8469) convert

v_mov_b32 v30, v28 // (op 8469) 
buffer_load_ubyte v22, v30, s[76:79], 0 offen offset:4 // (op 8469) Load value
v_add_u32 v30, v30, v16 // (op 8469) 
buffer_load_ubyte v23, v30, s[76:79], 0 offen offset:4 // (op 8469) Load value



v_and_b32 v28, -4, v1 // (op 10137) 
v_sub_u32 v30, v1, v28 // (op 10137) 

v_and_b32 v28, 4294967292, v1 // (op 10137) 
v_add_u32 v16, v28, v30 // (op 10137) 


s_barrier  // (op 10827) 



s_waitcnt vmcnt(15)
ds_write_b8 v16, v26 offset:49152 // (op 8472) Store local data 

v_and_b32 v28, -4, v1 // (op 10061) 
v_sub_u32 v30, v1, v28 // (op 10061) 

v_lshrrev_b32 v28, 2, v1 // (op 10061) 
v_lshl_add_u32 v16, v28, 3, v30 // (op 10061) 


s_barrier  // (op 10806) 



s_waitcnt vmcnt(10)
ds_write_b8 v16, v34 offset:49408 // (op 8474) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v16, v35 offset:49412 // (op 8474) Store local data 

s_barrier  // (op 10783) 



ds_read_b128 v[36:39], v14 // (op 8509) Load local data 
ds_read_b128 v[40:43], v14 offset:64 // (op 8509) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 10824) 



ds_read_u8 v28, v15 offset:49152 // (op 8539) Load local data 



ds_read_b128 v[44:47], v14 offset:4096 // (op 8514) Load local data 
ds_read_b128 v[48:51], v14 offset:4160 // (op 8514) Load local data 



ds_read_u8 v30, v15 offset:49280 // (op 8544) Load local data 
s_barrier  // (op 10764) 



ds_read_b128 v[52:55], v3 offset:8192 // (op 8489) Load local data 
ds_read_b128 v[56:59], v3 offset:8256 // (op 8489) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 10803) 



ds_read_u8 v16, v0 offset:49408 // (op 8519) Load local data 



ds_read_b128 v[60:63], v3 offset:12288 // (op 8494) Load local data 
ds_read_b128 v[64:67], v3 offset:12352 // (op 8494) Load local data 



ds_read_u8 v17, v0 offset:49536 // (op 8524) Load local data 



ds_read_b128 v[68:71], v3 offset:16384 // (op 8499) Load local data 
ds_read_b128 v[72:75], v3 offset:16448 // (op 8499) Load local data 



ds_read_u8 v32, v0 offset:49664 // (op 8529) Load local data 



ds_read_b128 v[76:79], v3 offset:20480 // (op 8504) Load local data 
ds_read_b128 v[80:83], v3 offset:20544 // (op 8504) Load local data 



ds_read_u8 v33, v0 offset:49792 // (op 8534) Load local data 
v_and_b32 v84, -8, v1 // (op 9882) 
v_sub_u32 v85, v1, v84 // (op 9882) 

v_lshlrev_b32 v84, 4, v85 // (op 9882) 

v_lshrrev_b32 v85, 3, v1 // (op 9882) 
v_lshl_add_u32 v86, v85, 7, v84 // (op 9882) 


s_nop 0
v_readfirstlane_b32 s15, v86 // (op 9882) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -8, v1 // (op 9957) 
v_sub_u32 v85, v1, v84 // (op 9957) 

v_lshlrev_b32 v84, 4, v85 // (op 9957) 

v_lshrrev_b32 v85, 3, v1 // (op 9957) 
v_lshl_add_u32 v86, v85, 7, v84 // (op 9957) 


s_nop 0
v_readfirstlane_b32 s6, v86 // (op 9957) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -4, v1 // (op 10032) 
v_sub_u32 v85, v1, v84 // (op 10032) 

v_lshrrev_b32 v84, 2, v1 // (op 10032) 
v_lshl_add_u32 v86, v84, 3, v85 // (op 10032) 


v_and_b32 v84, -4, v1 // (op 10108) 
v_sub_u32 v85, v1, v84 // (op 10108) 

v_and_b32 v84, 4294967292, v1 // (op 10108) 
v_add_u32 v87, v84, v85 // (op 10108) 


v_and_b32 v84, -8, v1 // (op 10236) 
v_sub_u32 v85, v1, v84 // (op 10236) 

v_lshlrev_b32 v84, 4, v85 // (op 10236) 

v_lshrrev_b32 v85, 3, v1 // (op 10236) 
v_lshl_add_u32 v88, v85, 7, v84 // (op 10236) 


s_nop 0
v_readfirstlane_b32 s7, v88 // (op 10236) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -8, v1 // (op 10280) 
v_sub_u32 v85, v1, v84 // (op 10280) 

v_lshlrev_b32 v84, 4, v85 // (op 10280) 

v_lshrrev_b32 v85, 3, v1 // (op 10280) 
v_lshl_add_u32 v88, v85, 7, v84 // (op 10280) 


s_nop 0
v_readfirstlane_b32 s8, v88 // (op 10280) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -4, v1 // (op 10324) 
v_sub_u32 v85, v1, v84 // (op 10324) 

v_lshrrev_b32 v84, 2, v1 // (op 10324) 
v_lshl_add_u32 v88, v84, 3, v85 // (op 10324) 


v_and_b32 v84, -4, v1 // (op 10353) 
v_sub_u32 v85, v1, v84 // (op 10353) 

v_and_b32 v84, 4294967292, v1 // (op 10353) 
v_add_u32 v89, v84, v85 // (op 10353) 


s_mov_b32 s1, 0 // (op 1639) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1638) 
s_mov_b32 s28, s18 // (op 1638) convert

s_and_b32 s29, s28, 4294967294 // (op 1638) 

s_cmp_lt_u32 s1, s29 // (op 1638) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoop_1638 // (op 1638) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoop_1638 if false)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoop_1638:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:59], v[36:43], a[0:3], v16, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 194) 
s_barrier  // (op 10789) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:59], v[44:51], a[4:7], v16, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 654) 
s_add_u32 s1, s1, 2 // (op 1640) 
s_lshl_b64 s[18:19], s[36:37], 5 // (op 8852) 
v_mov_b64 v[84:85], s[18:19] // (op 8852) convert


v_mov_b64 v[90:91], v[24:25] // (op 8852) 
v_mov_b64 v[92:93], v[90:91] // (op 8852) 
s_mov_b32 m0, s6 // (op 8852) reg expression
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:67], v[36:43], a[8:11], v17, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 951) 
s_mov_b32 s18, 256 // (op 8852) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[60:67], v[44:51], a[12:15], v17, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 905) 
buffer_load_dwordx4 v92, s[72:75], s18 offen offset: 0 lds // (op 8852) Load value direct to lds
v_add_co_u32 v92, s[28:29], v92, v84 // (op 8852) least significant half
v_addc_co_u32 v93, s[28:29], 0, v93, s[28:29] // (op 8852) most significant half

s_add_u32 m0, m0, 4096 // (op 8852) 
buffer_load_dwordx4 v92, s[72:75], s18 offen offset: 0 lds // (op 8852) Load value direct to lds



s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[68:75], v[36:43], a[16:19], v32, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1337) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[68:75], v[44:51], a[20:23], v32, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1291) 
v_mov_b32 v84, v31 // (op 200) 

buffer_load_ubyte v26, v84, s[80:83], 0 offen offset:8 // (op 200) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:83], v[36:43], a[24:27], v33, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1723) 
s_barrier  // (op 10770) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:83], v[44:51], a[28:31], v33, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1677) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 8847) 
v_mov_b64 v[92:93], s[18:19] // (op 8847) convert


v_mov_b64 v[94:95], v[20:21] // (op 8847) 
v_mov_b64 v[96:97], v[94:95] // (op 8847) 
s_add_u32 s18, 8192, s15 // (op 8847) 
s_mov_b32 m0, s18 // (op 8847) reg expression
s_mov_b32 s19, 256 // (op 8847) call()
buffer_load_dwordx4 v96, s[68:71], s19 offen offset: 0 lds // (op 8847) Load value direct to lds
v_add_co_u32 v96, s[28:29], v96, v92 // (op 8847) least significant half
v_addc_co_u32 v97, s[28:29], 0, v97, s[28:29] // (op 8847) most significant half

s_add_u32 m0, m0, 4096 // (op 8847) 
buffer_load_dwordx4 v96, s[68:71], s19 offen offset: 0 lds // (op 8847) Load value direct to lds
v_add_co_u32 v96, s[28:29], v96, v92 // (op 8847) least significant half
v_addc_co_u32 v97, s[28:29], 0, v97, s[28:29] // (op 8847) most significant half

s_add_u32 m0, m0, 4096 // (op 8847) 
buffer_load_dwordx4 v96, s[68:71], s19 offen offset: 0 lds // (op 8847) Load value direct to lds
v_add_co_u32 v96, s[28:29], v96, v92 // (op 8847) least significant half
v_addc_co_u32 v97, s[28:29], 0, v97, s[28:29] // (op 8847) most significant half

s_add_u32 m0, m0, 4096 // (op 8847) 
buffer_load_dwordx4 v96, s[68:71], s19 offen offset: 0 lds // (op 8847) Load value direct to lds





v_mov_b64 v[92:93], s[40:41] // (op 199) convert

v_mov_b32 v85, v29 // (op 199) 
v_mov_b32 v96, v85 // (op 199) 
buffer_load_ubyte v34, v96, s[76:79], 0 offen offset:8 // (op 199) Load value
v_add_u32 v96, v96, v92 // (op 199) 
buffer_load_ubyte v35, v96, s[76:79], 0 offen offset:8 // (op 199) Load value


s_barrier  // (op 10873) 



s_waitcnt vmcnt(15)
ds_write_b8 v89, v27 offset:49920 // (op 533) Store local data 
s_barrier  // (op 10867) 



s_waitcnt vmcnt(10)
ds_write_b8 v88, v22 offset:50176 // (op 532) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v88, v23 offset:50180 // (op 532) Store local data 
s_barrier  // (op 10853) 



ds_read_b128 v[96:99], v12 offset:24576 // (op 525) Load local data 
ds_read_b128 v[100:103], v12 offset:24640 // (op 525) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 10870) 



ds_read_u8 v92, v13 offset:49920 // (op 529) Load local data 



ds_read_b128 v[104:107], v12 offset:28672 // (op 659) Load local data 
ds_read_b128 v[108:111], v12 offset:28736 // (op 659) Load local data 



ds_read_u8 v93, v13 offset:50048 // (op 663) Load local data 
s_barrier  // (op 10844) 
s_barrier  // (op 10984) 



ds_read_b128 v[52:55], v11 offset:32768 // (op 523) Load local data 
ds_read_b128 v[56:59], v11 offset:32832 // (op 523) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 10863) 



ds_read_u8 v16, v10 offset:50176 // (op 527) Load local data 



ds_read_b128 v[60:63], v11 offset:36864 // (op 908) Load local data 
ds_read_b128 v[64:67], v11 offset:36928 // (op 908) Load local data 



ds_read_u8 v17, v10 offset:50304 // (op 911) Load local data 



ds_read_b128 v[68:71], v11 offset:40960 // (op 1294) Load local data 
ds_read_b128 v[72:75], v11 offset:41024 // (op 1294) Load local data 



ds_read_u8 v32, v10 offset:50432 // (op 1297) Load local data 



ds_read_b128 v[76:79], v11 offset:45056 // (op 1680) Load local data 
ds_read_b128 v[80:83], v11 offset:45120 // (op 1680) Load local data 



ds_read_u8 v33, v10 offset:50560 // (op 1683) Load local data 
s_barrier  // (op 11014) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:59], v[96:103], a[0:3], v16, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 530) 
s_barrier  // (op 10860) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:59], v[104:111], a[4:7], v16, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 655) 
s_lshl_b64 s[18:19], s[36:37], 5 // (op 8875) 
v_mov_b64 v[112:113], s[18:19] // (op 8875) convert


v_mov_b64 v[114:115], v[90:91] // (op 8875) 
s_add_u32 s18, 24576, s8 // (op 8875) 
s_mov_b32 m0, s18 // (op 8875) reg expression
s_mov_b32 s19, 384 // (op 8875) call()
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:67], v[96:103], a[8:11], v17, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 952) 
buffer_load_dwordx4 v114, s[72:75], s19 offen offset: 0 lds // (op 8875) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[60:67], v[104:111], a[12:15], v17, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 906) 
v_add_co_u32 v114, s[28:29], v114, v112 // (op 8875) least significant half
v_addc_co_u32 v115, s[28:29], 0, v115, s[28:29] // (op 8875) most significant half

s_add_u32 m0, m0, 4096 // (op 8875) 
buffer_load_dwordx4 v114, s[72:75], s19 offen offset: 0 lds // (op 8875) Load value direct to lds




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[68:75], v[96:103], a[16:19], v32, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1338) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[68:75], v[104:111], a[20:23], v32, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1292) 
buffer_load_ubyte v27, v84, s[80:83], 0 offen offset:12 // (op 536) Load value

v_add_co_u32 v24, vcc, 256, v24 // (op 9148) least significant half
v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 9148) most significant half
v_add_u32 v31, 8, v31 // (op 9404) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:83], v[96:103], a[24:27], v33, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1724) 
s_barrier  // (op 10850) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:83], v[104:111], a[28:31], v33, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1678) 
s_lshl_b64 s[18:19], s[30:31], 5 // (op 8870) 


v_mov_b64 v[90:91], s[18:19] // (op 8870) convert


v_mov_b64 v[92:93], v[94:95] // (op 8870) 
s_add_u32 s18, 32768, s7 // (op 8870) 
s_mov_b32 m0, s18 // (op 8870) reg expression
s_mov_b32 s19, 384 // (op 8870) call()
buffer_load_dwordx4 v92, s[68:71], s19 offen offset: 0 lds // (op 8870) Load value direct to lds
v_add_co_u32 v92, s[28:29], v92, v90 // (op 8870) least significant half
v_addc_co_u32 v93, s[28:29], 0, v93, s[28:29] // (op 8870) most significant half

s_add_u32 m0, m0, 4096 // (op 8870) 
buffer_load_dwordx4 v92, s[68:71], s19 offen offset: 0 lds // (op 8870) Load value direct to lds
v_add_co_u32 v92, s[28:29], v92, v90 // (op 8870) least significant half
v_addc_co_u32 v93, s[28:29], 0, v93, s[28:29] // (op 8870) most significant half

s_add_u32 m0, m0, 4096 // (op 8870) 
buffer_load_dwordx4 v92, s[68:71], s19 offen offset: 0 lds // (op 8870) Load value direct to lds
v_add_co_u32 v92, s[28:29], v92, v90 // (op 8870) least significant half
v_addc_co_u32 v93, s[28:29], 0, v93, s[28:29] // (op 8870) most significant half

s_add_u32 m0, m0, 4096 // (op 8870) 
buffer_load_dwordx4 v92, s[68:71], s19 offen offset: 0 lds // (op 8870) Load value direct to lds






v_add_co_u32 v20, vcc, 256, v20 // (op 9010) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 9010) most significant half
v_mov_b64 v[90:91], s[40:41] // (op 535) convert

v_mov_b32 v84, v85 // (op 535) 
buffer_load_ubyte v22, v84, s[76:79], 0 offen offset:12 // (op 535) Load value
v_add_u32 v84, v84, v90 // (op 535) 
buffer_load_ubyte v23, v84, s[76:79], 0 offen offset:12 // (op 535) Load value



s_barrier  // (op 10830) 
v_add_u32 v29, 8, v29 // (op 9276) 



s_waitcnt vmcnt(15)
ds_write_b8 v87, v26 offset:49152 // (op 197) Store local data 
s_barrier  // (op 10809) 



s_waitcnt vmcnt(10)
ds_write_b8 v86, v34 offset:49408 // (op 196) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v86, v35 offset:49412 // (op 196) Store local data 
s_barrier  // (op 10795) 



ds_read_b128 v[36:39], v8 // (op 189) Load local data 
ds_read_b128 v[40:43], v8 offset:64 // (op 189) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 10837) 



ds_read_u8 v28, v9 offset:49152 // (op 193) Load local data 



ds_read_b128 v[44:47], v8 offset:4096 // (op 649) Load local data 
ds_read_b128 v[48:51], v8 offset:4160 // (op 649) Load local data 



ds_read_u8 v30, v9 offset:49280 // (op 653) Load local data 
s_barrier  // (op 10776) 



ds_read_b128 v[52:55], v7 offset:8192 // (op 187) Load local data 
ds_read_b128 v[56:59], v7 offset:8256 // (op 187) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 10816) 



ds_read_u8 v16, v6 offset:49408 // (op 191) Load local data 



ds_read_b128 v[60:63], v7 offset:12288 // (op 900) Load local data 
ds_read_b128 v[64:67], v7 offset:12352 // (op 900) Load local data 



ds_read_u8 v17, v6 offset:49536 // (op 903) Load local data 



ds_read_b128 v[68:71], v7 offset:16384 // (op 1286) Load local data 
ds_read_b128 v[72:75], v7 offset:16448 // (op 1286) Load local data 



ds_read_u8 v32, v6 offset:49664 // (op 1289) Load local data 



ds_read_b128 v[76:79], v7 offset:20480 // (op 1672) Load local data 
ds_read_b128 v[80:83], v7 offset:20544 // (op 1672) Load local data 



ds_read_u8 v33, v6 offset:49792 // (op 1675) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1638) 
s_mov_b32 s28, s18 // (op 1638) convert

s_and_b32 s29, s28, 4294967294 // (op 1638) 

s_cmp_lt_u32 s1, s29 // (op 1638) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoop_1638 // (op 1638) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoop_1638 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoop_1638:
















s_barrier  // (op 10998) 
v_lshrrev_b32 v31, 3, v1 // (op 8942) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 8942) 
v_and_b32 v20, -8, v1 // (op 8942) 
s_and_b32 s15, s18, 4294967294 // (op 8942) 
v_sub_u32 v21, v1, v20 // (op 8942) 

s_add_u32 s6, s0, s15 // (op 8942) 

v_lshlrev_b32 v20, 4, v21 // (op 8942) 

s_mul_i32 s15, s16, s50 // (op 8942) 
s_add_u32 s15, s15, s6 // (op 8942) 

s_add_u32 s6, s15, s9 // (op 8942) 

s_mul_hi_u32 s15, s6, s13 // (op 8942) 
s_sub_u32 s7, s6, s15 // (op 8942) 
s_lshr_b32 s8, s7, 1 // (op 8942) 

s_add_u32 s7, s8, s15 // (op 8942) 


s_lshr_b32 s8, s7, s11 // (op 8942) 

s_and_b32 s7, s11, 2147483648 // (op 8942) 
s_cmp_eq_i32 s7, 0 // (op 8942) 
s_mov_b32 s15, scc // (op 8942) 

s_cmp_eq_i32 s15, 0 // (op 8942) 
s_cselect_b32 s7, s6, s8 // (op 8942) 


s_mul_hi_u32 s8, s7, s12 // (op 8942) 
s_sub_u32 s15, s7, s8 // (op 8942) 
s_lshr_b32 s28, s15, 1 // (op 8942) 

s_add_u32 s15, s28, s8 // (op 8942) 


s_lshr_b32 s8, s15, s10 // (op 8942) 

s_and_b32 s15, s10, 2147483648 // (op 8942) 
s_cmp_eq_i32 s15, 0 // (op 8942) 
s_mov_b32 s28, scc // (op 8942) 

s_cmp_eq_i32 s28, 0 // (op 8942) 
s_cselect_b32 s15, s7, s8 // (op 8942) 


s_mul_i32 s8, s7, s18 // (op 8942) 


v_lshl_add_u32 v21, s15, 7, v31 // (op 8942) 


s_sub_u32 s7, s6, s8 // (op 8942) 


v_lshl_add_u32 v31, s7, 7, v20 // (op 8942) 


v_mov_b32 v29, 0 // (op 8942) convert
v_mov_b32 v28, v31 // (op 8942) convert

v_mul_lo_u32 v25, v21, s31 // (op 8942) most significant: low of low * high
v_mul_hi_u32 v20, v21, s30 // (op 8942) most significant: high of low * low
v_mul_lo_u32 v24, v21, s30 // (op 8942) least significant: low of low * low
v_add_u32 v25, v25, v20 // (op 8942) most significant: sum

v_add_co_u32 v24, s[18:19], v24, v28 // (op 8942) least significant half
v_addc_co_u32 v25, s[18:19], v25, v29, s[18:19] // (op 8942) most significant half



s_mul_i32 s19, s30, s35 // (op 8944) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 8944) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 8944) least significant
s_mul_hi_u32 s6, s30, s34 // (op 8944) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 8944) most significant: sum
s_add_u32 s19, s19, s6 // (op 8944) most significant: sum


s_mov_b32 s68, s32 // (op 8944) reg expression
s_mov_b32 s69, s33 // (op 8944) reg expression
s_mov_b32 s70, s18 // (op 8944) reg expression
s_mov_b32 s71, 131072 // (op 8944) call()

v_lshrrev_b32 v26, 3, v1 // (op 9092) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 9092) 
v_and_b32 v31, -8, v1 // (op 9092) 
s_and_b32 s15, s18, 4294967294 // (op 9092) 
v_sub_u32 v20, v1, v31 // (op 9092) 

s_add_u32 s6, s0, s15 // (op 9092) 

v_lshlrev_b32 v21, 4, v20 // (op 9092) 

s_mul_i32 s15, s16, s50 // (op 9092) 
s_add_u32 s15, s15, s6 // (op 9092) 

s_add_u32 s6, s15, s9 // (op 9092) 

s_mul_hi_u32 s15, s6, s13 // (op 9092) 
s_sub_u32 s7, s6, s15 // (op 9092) 
s_lshr_b32 s8, s7, 1 // (op 9092) 

s_add_u32 s7, s8, s15 // (op 9092) 


s_lshr_b32 s8, s7, s11 // (op 9092) 

s_and_b32 s7, s11, 2147483648 // (op 9092) 
s_cmp_eq_i32 s7, 0 // (op 9092) 
s_mov_b32 s15, scc // (op 9092) 

s_cmp_eq_i32 s15, 0 // (op 9092) 
s_cselect_b32 s7, s6, s8 // (op 9092) 


s_mul_hi_u32 s8, s7, s12 // (op 9092) 
s_sub_u32 s15, s7, s8 // (op 9092) 
s_lshr_b32 s28, s15, 1 // (op 9092) 

s_add_u32 s15, s28, s8 // (op 9092) 


s_lshr_b32 s8, s15, s10 // (op 9092) 

s_and_b32 s15, s10, 2147483648 // (op 9092) 
s_cmp_eq_i32 s15, 0 // (op 9092) 
s_mov_b32 s28, scc // (op 9092) 

s_cmp_eq_i32 s28, 0 // (op 9092) 
s_cselect_b32 s15, s7, s8 // (op 9092) 


s_mul_i32 s8, s15, s14 // (op 9092) 

s_sub_u32 s15, s7, s8 // (op 9092) 

s_mul_i32 s8, s7, s18 // (op 9092) 


v_lshl_add_u32 v20, s15, 6, v26 // (op 9092) 


s_sub_u32 s7, s6, s8 // (op 9092) 


v_mul_lo_u32 v29, v20, s37 // (op 9092) most significant: low of low * high
v_mul_hi_u32 v26, v20, s36 // (op 9092) most significant: high of low * low
v_mul_lo_u32 v28, v20, s36 // (op 9092) least significant: low of low * low
v_add_u32 v29, v29, v26 // (op 9092) most significant: sum


v_lshl_add_u32 v20, s7, 7, v21 // (op 9092) 


v_mov_b32 v35, 0 // (op 9092) convert
v_mov_b32 v34, v20 // (op 9092) convert

v_add_co_u32 v20, s[18:19], v34, v28 // (op 9092) least significant half
v_addc_co_u32 v21, s[18:19], v35, v29, s[18:19] // (op 9092) most significant half



s_mul_i32 s19, s36, s23 // (op 9094) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 9094) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 9094) least significant
s_mul_hi_u32 s6, s36, s22 // (op 9094) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9094) most significant: sum
s_add_u32 s19, s19, s6 // (op 9094) most significant: sum


s_mov_b32 s72, s38 // (op 9094) reg expression
s_mov_b32 s73, s39 // (op 9094) reg expression
s_mov_b32 s74, s18 // (op 9094) reg expression
s_mov_b32 s75, 131072 // (op 9094) call()

v_lshrrev_b32 v26, 2, v1 // (op 9220) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 9220) 
v_lshlrev_b32 v31, 1, v26 // (op 9220) 

s_and_b32 s15, s18, 4294967294 // (op 9220) 
v_and_b32 v26, -4, v1 // (op 9220) 
s_add_u32 s6, s0, s15 // (op 9220) 

v_sub_u32 v28, v1, v26 // (op 9220) 

s_mul_i32 s15, s16, s50 // (op 9220) 
s_add_u32 s15, s15, s6 // (op 9220) 

s_add_u32 s6, s15, s9 // (op 9220) 

s_mul_hi_u32 s15, s6, s13 // (op 9220) 
s_sub_u32 s7, s6, s15 // (op 9220) 
s_lshr_b32 s8, s7, 1 // (op 9220) 

s_add_u32 s7, s8, s15 // (op 9220) 


s_lshr_b32 s8, s7, s11 // (op 9220) 

s_and_b32 s7, s11, 2147483648 // (op 9220) 
s_cmp_eq_i32 s7, 0 // (op 9220) 
s_mov_b32 s15, scc // (op 9220) 

s_cmp_eq_i32 s15, 0 // (op 9220) 
s_cselect_b32 s7, s6, s8 // (op 9220) 


s_mul_hi_u32 s8, s7, s12 // (op 9220) 
s_sub_u32 s15, s7, s8 // (op 9220) 
s_lshr_b32 s28, s15, 1 // (op 9220) 

s_add_u32 s15, s28, s8 // (op 9220) 


s_lshr_b32 s8, s15, s10 // (op 9220) 

s_and_b32 s15, s10, 2147483648 // (op 9220) 
s_cmp_eq_i32 s15, 0 // (op 9220) 
s_mov_b32 s28, scc // (op 9220) 

s_cmp_eq_i32 s28, 0 // (op 9220) 
s_cselect_b32 s15, s7, s8 // (op 9220) 


s_mul_i32 s8, s7, s18 // (op 9220) 


v_lshl_add_u32 v26, s15, 7, v31 // (op 9220) 


s_sub_u32 s7, s6, s8 // (op 9220) 


v_lshl_add_u32 v29, s7, 2, v28 // (op 9220) 


v_mul_lo_u32 v28, v26, s40 // (op 9220) 
v_add_u32 v28, v28, v29 // (op 9220) 


s_mul_i32 s19, s40, s35 // (op 9222) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 9222) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 9222) least significant
s_mul_hi_u32 s6, s40, s34 // (op 9222) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9222) most significant: sum
s_add_u32 s19, s19, s6 // (op 9222) most significant: sum


s_mov_b32 s76, s42 // (op 9222) reg expression
s_mov_b32 s77, s43 // (op 9222) reg expression
s_mov_b32 s78, s18 // (op 9222) reg expression
s_mov_b32 s79, 131072 // (op 9222) call()

v_lshrrev_b32 v26, 2, v1 // (op 9348) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 9348) 
v_and_b32 v29, -4, v1 // (op 9348) 
s_and_b32 s15, s18, 4294967294 // (op 9348) 
v_sub_u32 v31, v1, v29 // (op 9348) 

s_add_u32 s6, s0, s15 // (op 9348) 

s_mul_i32 s15, s16, s50 // (op 9348) 
s_add_u32 s15, s15, s6 // (op 9348) 

s_add_u32 s6, s15, s9 // (op 9348) 

s_mul_hi_u32 s15, s6, s13 // (op 9348) 
s_sub_u32 s7, s6, s15 // (op 9348) 
s_lshr_b32 s8, s7, 1 // (op 9348) 

s_add_u32 s7, s8, s15 // (op 9348) 


s_lshr_b32 s8, s7, s11 // (op 9348) 

s_and_b32 s7, s11, 2147483648 // (op 9348) 
s_cmp_eq_i32 s7, 0 // (op 9348) 
s_mov_b32 s15, scc // (op 9348) 

s_cmp_eq_i32 s15, 0 // (op 9348) 
s_cselect_b32 s7, s6, s8 // (op 9348) 


s_mul_hi_u32 s8, s7, s12 // (op 9348) 
s_sub_u32 s15, s7, s8 // (op 9348) 
s_lshr_b32 s28, s15, 1 // (op 9348) 

s_add_u32 s15, s28, s8 // (op 9348) 


s_lshr_b32 s8, s15, s10 // (op 9348) 

s_and_b32 s15, s10, 2147483648 // (op 9348) 
s_cmp_eq_i32 s15, 0 // (op 9348) 
s_mov_b32 s28, scc // (op 9348) 

s_cmp_eq_i32 s28, 0 // (op 9348) 
s_cselect_b32 s15, s7, s8 // (op 9348) 


s_mul_i32 s8, s15, s14 // (op 9348) 

s_sub_u32 s15, s7, s8 // (op 9348) 

v_lshl_add_u32 v29, s15, 6, v26 // (op 9348) 


s_mul_i32 s8, s7, s18 // (op 9348) 


v_mul_lo_u32 v26, v29, s44 // (op 9348) 

s_sub_u32 s7, s6, s8 // (op 9348) 


v_lshl_add_u32 v29, s7, 2, v31 // (op 9348) 


v_add_u32 v31, v29, v26 // (op 9348) 


s_mul_i32 s19, s44, s23 // (op 9350) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 9350) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 9350) least significant
s_mul_hi_u32 s6, s44, s22 // (op 9350) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 9350) most significant: sum
s_add_u32 s19, s19, s6 // (op 9350) most significant: sum


s_mov_b32 s80, s46 // (op 9350) reg expression
s_mov_b32 s81, s47 // (op 9350) reg expression
s_mov_b32 s82, s18 // (op 9350) reg expression
s_mov_b32 s83, 131072 // (op 9350) call()

v_and_b32 v26, -8, v1 // (op 9895) 
v_sub_u32 v29, v1, v26 // (op 9895) 

v_lshlrev_b32 v26, 4, v29 // (op 9895) 

v_lshrrev_b32 v29, 3, v1 // (op 9895) 
v_lshl_add_u32 v34, v29, 7, v26 // (op 9895) 


s_nop 0
v_readfirstlane_b32 s15, v34 // (op 9895) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -8, v1 // (op 9970) 
v_sub_u32 v29, v1, v26 // (op 9970) 

v_lshlrev_b32 v26, 4, v29 // (op 9970) 

v_lshrrev_b32 v29, 3, v1 // (op 9970) 
v_lshl_add_u32 v34, v29, 7, v26 // (op 9970) 


s_nop 0
v_readfirstlane_b32 s6, v34 // (op 9970) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 10045) 
v_sub_u32 v29, v1, v26 // (op 10045) 

v_lshrrev_b32 v26, 2, v1 // (op 10045) 
v_lshl_add_u32 v34, v26, 3, v29 // (op 10045) 


v_and_b32 v26, -4, v1 // (op 10121) 
v_sub_u32 v29, v1, v26 // (op 10121) 

v_and_b32 v26, 4294967292, v1 // (op 10121) 
v_add_u32 v35, v26, v29 // (op 10121) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1644) 
s_and_b32 s7, s18, 4294967294 // (op 1644) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1643) 
s_mov_b32 s8, s18 // (op 1643) convert

s_cmp_lt_u32 s7, s8 // (op 1643) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoopTail_1643 // (op 1643) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoopTail_1643 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoopTail_1643:
s_barrier  // (op 10773) 
s_add_u32 s7, s7, 1 // (op 1645) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 8857) 
v_mov_b64 v[16:17], s[18:19] // (op 8857) convert


v_mov_b64 v[22:23], v[24:25] // (op 8857) 
v_mov_b64 v[26:27], v[22:23] // (op 8857) 
s_add_u32 s8, 8192, s15 // (op 8857) 
s_mov_b32 m0, s8 // (op 8857) reg expression
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 8857) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 8857) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 8857) most significant half

s_add_u32 m0, m0, 4096 // (op 8857) 
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 8857) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 8857) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 8857) most significant half

s_add_u32 m0, m0, 4096 // (op 8857) 
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 8857) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 8857) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 8857) most significant half

s_add_u32 m0, m0, 4096 // (op 8857) 
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 8857) Load value direct to lds




v_add_co_u32 v24, vcc, 128, v24 // (op 8932) least significant half
v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 8932) most significant half
s_barrier  // (op 10792) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 8863) 
v_mov_b64 v[16:17], s[18:19] // (op 8863) convert


v_mov_b64 v[22:23], v[20:21] // (op 8863) 
v_mov_b64 v[26:27], v[22:23] // (op 8863) 
s_mov_b32 m0, s6 // (op 8863) reg expression
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 8863) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 8863) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 8863) most significant half

s_add_u32 m0, m0, 4096 // (op 8863) 
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 8863) Load value direct to lds



v_add_co_u32 v20, vcc, 128, v20 // (op 9084) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 9084) most significant half

v_mov_b64 v[16:17], s[40:41] // (op 469) convert

v_mov_b32 v22, v28 // (op 469) 
v_mov_b32 v23, v22 // (op 469) 
buffer_load_ubyte v26, v23, s[76:79], 0 offen offset: 0 // (op 469) Load value
v_add_u32 v23, v23, v16 // (op 469) 
buffer_load_ubyte v27, v23, s[76:79], 0 offen offset: 0 // (op 469) Load value



v_add_u32 v28, 4, v28 // (op 9212) 


s_barrier  // (op 10812) 
v_mov_b32 v16, v31 // (op 470) 




buffer_load_ubyte v17, v16, s[80:83], 0 offen offset: 0 // (op 470) Load value

v_add_u32 v31, 4, v31 // (op 9340) 
s_waitcnt vmcnt(2)
ds_write_b8 v34, v26 offset:49408 // (op 466) Store local data 
s_barrier  // (op 10779) 
s_waitcnt vmcnt(1)
ds_write_b8 v34, v27 offset:49412 // (op 466) Store local data 




ds_read_b128 v[36:39], v7 offset:8192 // (op 457) Load local data 
s_barrier  // (op 10833) 
ds_read_b128 v[40:43], v7 offset:8256 // (op 457) Load local data 



s_barrier  // (op 10799) 
s_waitcnt vmcnt(0)
ds_write_b8 v35, v17 offset:49152 // (op 467) Store local data 




ds_read_b128 v[44:47], v8 // (op 459) Load local data 
ds_read_b128 v[48:51], v8 offset:64 // (op 459) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 10820) 



ds_read_u8 v16, v6 offset:49408 // (op 461) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 10840) 



ds_read_u8 v17, v9 offset:49152 // (op 463) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:51], a[0:3], v16, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 464) 
ds_read_b128 v[52:55], v8 offset:4096 // (op 628) Load local data 
ds_read_b128 v[56:59], v8 offset:4160 // (op 628) Load local data 



ds_read_u8 v22, v9 offset:49280 // (op 632) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:43], v[52:59], a[4:7], v16, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 633) 
ds_read_b128 v[60:63], v7 offset:12288 // (op 846) Load local data 


ds_read_b128 v[64:67], v7 offset:12352 // (op 846) Load local data 



ds_read_u8 v16, v6 offset:49536 // (op 849) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:67], v[44:51], a[8:11], v16, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 851) 
ds_read_b128 v[36:39], v7 offset:16384 // (op 1232) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[60:67], v[52:59], a[12:15], v16, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 882) 
ds_read_b128 v[40:43], v7 offset:16448 // (op 1232) Load local data 





ds_read_u8 v16, v6 offset:49664 // (op 1235) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[36:43], v[44:51], a[16:19], v16, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1237) 
ds_read_b128 v[60:63], v7 offset:20480 // (op 1618) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:43], v[52:59], a[20:23], v16, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1268) 
ds_read_b128 v[64:67], v7 offset:20544 // (op 1618) Load local data 





ds_read_u8 v16, v6 offset:49792 // (op 1621) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:67], v[44:51], a[24:27], v16, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1623) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[60:67], v[52:59], a[28:31], v16, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1654) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 1643) 
s_mov_b32 s8, s18 // (op 1643) convert

s_cmp_lt_u32 s7, s8 // (op 1643) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoopTail_1643 // (op 1643) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_KLoopTail_1643 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_KLoopTail_1643:









s_barrier  // (op 11002) 




v_cmp_eq_f32 vcc, 0.00000, v18 // (op 234) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_0_beta_234 // (op 234) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_0_beta_234
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


v_accvgpr_read v14, a8 // (op 2171) 
v_accvgpr_read v15, a9 // (op 2171) 
v_accvgpr_read v16, a10 // (op 2171) 
v_accvgpr_read v17, a11 // (op 2171) 
v_mul_f32 v20, v19, v14 // (op 2171) 
v_mul_f32 v21, v19, v15 // (op 2171) 
v_mul_f32 v22, v19, v16 // (op 2171) 
v_mul_f32 v23, v19, v17 // (op 2171) 



v_add_u32 v0, 128, v5 // (op 2174) 

buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 2174) Store value


v_accvgpr_read v14, a12 // (op 2180) 
v_accvgpr_read v15, a13 // (op 2180) 
v_accvgpr_read v16, a14 // (op 2180) 
v_accvgpr_read v17, a15 // (op 2180) 
v_mul_f32 v20, v19, v14 // (op 2180) 
v_mul_f32 v21, v19, v15 // (op 2180) 
v_mul_f32 v22, v19, v16 // (op 2180) 
v_mul_f32 v23, v19, v17 // (op 2180) 



s_lshl_b32 s15, s24, 7 // (op 2176) 
s_add_u32 s15, s15, 128 // (op 2176) 
v_add_u32 v0, v5, s15 // (op 2176) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 2176) Store value


v_accvgpr_read v14, a16 // (op 2207) 
v_accvgpr_read v15, a17 // (op 2207) 
v_accvgpr_read v16, a18 // (op 2207) 
v_accvgpr_read v17, a19 // (op 2207) 
v_mul_f32 v20, v19, v14 // (op 2207) 
v_mul_f32 v21, v19, v15 // (op 2207) 
v_mul_f32 v22, v19, v16 // (op 2207) 
v_mul_f32 v23, v19, v17 // (op 2207) 



v_add_u32 v0, 256, v5 // (op 2210) 

buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 2210) Store value


v_accvgpr_read v14, a20 // (op 2216) 
v_accvgpr_read v15, a21 // (op 2216) 
v_accvgpr_read v16, a22 // (op 2216) 
v_accvgpr_read v17, a23 // (op 2216) 
v_mul_f32 v20, v19, v14 // (op 2216) 
v_mul_f32 v21, v19, v15 // (op 2216) 
v_mul_f32 v22, v19, v16 // (op 2216) 
v_mul_f32 v23, v19, v17 // (op 2216) 



s_lshl_b32 s15, s24, 7 // (op 2212) 
s_add_u32 s15, s15, 256 // (op 2212) 
v_add_u32 v0, v5, s15 // (op 2212) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 2212) Store value


v_accvgpr_read v14, a24 // (op 2243) 
v_accvgpr_read v15, a25 // (op 2243) 
v_accvgpr_read v16, a26 // (op 2243) 
v_accvgpr_read v17, a27 // (op 2243) 
v_mul_f32 v20, v19, v14 // (op 2243) 
v_mul_f32 v21, v19, v15 // (op 2243) 
v_mul_f32 v22, v19, v16 // (op 2243) 
v_mul_f32 v23, v19, v17 // (op 2243) 



v_add_u32 v0, 384, v5 // (op 2246) 

buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 2246) Store value


v_accvgpr_read v14, a28 // (op 2252) 
v_accvgpr_read v15, a29 // (op 2252) 
v_accvgpr_read v16, a30 // (op 2252) 
v_accvgpr_read v17, a31 // (op 2252) 
v_mul_f32 v20, v19, v14 // (op 2252) 
v_mul_f32 v21, v19, v15 // (op 2252) 
v_mul_f32 v22, v19, v16 // (op 2252) 
v_mul_f32 v23, v19, v17 // (op 2252) 



s_lshl_b32 s15, s24, 7 // (op 2248) 
s_add_u32 s15, s15, 384 // (op 2248) 
v_add_u32 v0, v5, s15 // (op 2248) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 2248) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_0_beta_234 // (op 234) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_0_beta_234
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalFalse_0_beta_234:


v_add_u32 v0, 384, v4 // (op 2430) 

buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 2430) Load value

v_accvgpr_read v20, a24 // (op 2422) 
v_accvgpr_read v21, a25 // (op 2422) 
v_accvgpr_read v22, a26 // (op 2422) 
v_accvgpr_read v23, a27 // (op 2422) 
v_mul_f32 v24, v19, v20 // (op 2422) 
v_mul_f32 v25, v19, v21 // (op 2422) 
v_mul_f32 v26, v19, v22 // (op 2422) 
v_mul_f32 v27, v19, v23 // (op 2422) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 2429) 
v_mul_f32 v21, v18, v15 // (op 2429) 
v_mul_f32 v22, v18, v16 // (op 2429) 
v_mul_f32 v23, v18, v17 // (op 2429) 

v_add_f32 v14, v20, v24 // (op 2423) 
v_add_f32 v15, v21, v25 // (op 2423) 
v_add_f32 v16, v22, v26 // (op 2423) 
v_add_f32 v17, v23, v27 // (op 2423) 




v_add_u32 v0, 384, v5 // (op 2425) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 2425) Store value




s_lshl_b32 s15, s4, 7 // (op 2434) 
s_add_u32 s15, s15, 384 // (op 2434) 
v_add_u32 v0, v4, s15 // (op 2434) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 2434) Load value

v_accvgpr_read v20, a28 // (op 2436) 
v_accvgpr_read v21, a29 // (op 2436) 
v_accvgpr_read v22, a30 // (op 2436) 
v_accvgpr_read v23, a31 // (op 2436) 
v_mul_f32 v24, v19, v20 // (op 2436) 
v_mul_f32 v25, v19, v21 // (op 2436) 
v_mul_f32 v26, v19, v22 // (op 2436) 
v_mul_f32 v27, v19, v23 // (op 2436) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 2432) 
v_mul_f32 v21, v18, v15 // (op 2432) 
v_mul_f32 v22, v18, v16 // (op 2432) 
v_mul_f32 v23, v18, v17 // (op 2432) 

v_add_f32 v14, v20, v24 // (op 2433) 
v_add_f32 v15, v21, v25 // (op 2433) 
v_add_f32 v16, v22, v26 // (op 2433) 
v_add_f32 v17, v23, v27 // (op 2433) 




s_lshl_b32 s15, s24, 7 // (op 2427) 
s_add_u32 s15, s15, 384 // (op 2427) 
v_add_u32 v0, v5, s15 // (op 2427) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 2427) Store value




v_add_u32 v0, 256, v4 // (op 2389) 

s_nop 0
buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 2389) Load value

v_accvgpr_read v20, a16 // (op 2381) 
v_accvgpr_read v21, a17 // (op 2381) 
v_accvgpr_read v22, a18 // (op 2381) 
v_accvgpr_read v23, a19 // (op 2381) 
v_mul_f32 v24, v19, v20 // (op 2381) 
v_mul_f32 v25, v19, v21 // (op 2381) 
v_mul_f32 v26, v19, v22 // (op 2381) 
v_mul_f32 v27, v19, v23 // (op 2381) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 2388) 
v_mul_f32 v21, v18, v15 // (op 2388) 
v_mul_f32 v22, v18, v16 // (op 2388) 
v_mul_f32 v23, v18, v17 // (op 2388) 

v_add_f32 v14, v20, v24 // (op 2382) 
v_add_f32 v15, v21, v25 // (op 2382) 
v_add_f32 v16, v22, v26 // (op 2382) 
v_add_f32 v17, v23, v27 // (op 2382) 




v_add_u32 v0, 256, v5 // (op 2384) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 2384) Store value




s_lshl_b32 s15, s4, 7 // (op 2393) 
s_add_u32 s15, s15, 256 // (op 2393) 
v_add_u32 v0, v4, s15 // (op 2393) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 2393) Load value

v_accvgpr_read v20, a20 // (op 2395) 
v_accvgpr_read v21, a21 // (op 2395) 
v_accvgpr_read v22, a22 // (op 2395) 
v_accvgpr_read v23, a23 // (op 2395) 
v_mul_f32 v24, v19, v20 // (op 2395) 
v_mul_f32 v25, v19, v21 // (op 2395) 
v_mul_f32 v26, v19, v22 // (op 2395) 
v_mul_f32 v27, v19, v23 // (op 2395) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 2391) 
v_mul_f32 v21, v18, v15 // (op 2391) 
v_mul_f32 v22, v18, v16 // (op 2391) 
v_mul_f32 v23, v18, v17 // (op 2391) 

v_add_f32 v14, v20, v24 // (op 2392) 
v_add_f32 v15, v21, v25 // (op 2392) 
v_add_f32 v16, v22, v26 // (op 2392) 
v_add_f32 v17, v23, v27 // (op 2392) 




s_lshl_b32 s15, s24, 7 // (op 2386) 
s_add_u32 s15, s15, 256 // (op 2386) 
v_add_u32 v0, v5, s15 // (op 2386) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 2386) Store value




v_add_u32 v0, 128, v4 // (op 2348) 

s_nop 0
buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 2348) Load value

v_accvgpr_read v20, a8 // (op 2340) 
v_accvgpr_read v21, a9 // (op 2340) 
v_accvgpr_read v22, a10 // (op 2340) 
v_accvgpr_read v23, a11 // (op 2340) 
v_mul_f32 v24, v19, v20 // (op 2340) 
v_mul_f32 v25, v19, v21 // (op 2340) 
v_mul_f32 v26, v19, v22 // (op 2340) 
v_mul_f32 v27, v19, v23 // (op 2340) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 2347) 
v_mul_f32 v21, v18, v15 // (op 2347) 
v_mul_f32 v22, v18, v16 // (op 2347) 
v_mul_f32 v23, v18, v17 // (op 2347) 

v_add_f32 v14, v20, v24 // (op 2341) 
v_add_f32 v15, v21, v25 // (op 2341) 
v_add_f32 v16, v22, v26 // (op 2341) 
v_add_f32 v17, v23, v27 // (op 2341) 




v_add_u32 v0, 128, v5 // (op 2343) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 2343) Store value




s_lshl_b32 s15, s4, 7 // (op 2352) 
s_add_u32 s15, s15, 128 // (op 2352) 
v_add_u32 v0, v4, s15 // (op 2352) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 2352) Load value

v_accvgpr_read v20, a12 // (op 2354) 
v_accvgpr_read v21, a13 // (op 2354) 
v_accvgpr_read v22, a14 // (op 2354) 
v_accvgpr_read v23, a15 // (op 2354) 
v_mul_f32 v24, v19, v20 // (op 2354) 
v_mul_f32 v25, v19, v21 // (op 2354) 
v_mul_f32 v26, v19, v22 // (op 2354) 
v_mul_f32 v27, v19, v23 // (op 2354) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 2350) 
v_mul_f32 v21, v18, v15 // (op 2350) 
v_mul_f32 v22, v18, v16 // (op 2350) 
v_mul_f32 v23, v18, v17 // (op 2350) 

v_add_f32 v14, v20, v24 // (op 2351) 
v_add_f32 v15, v21, v25 // (op 2351) 
v_add_f32 v16, v22, v26 // (op 2351) 
v_add_f32 v17, v23, v27 // (op 2351) 




s_lshl_b32 s15, s24, 7 // (op 2345) 
s_add_u32 s15, s15, 128 // (op 2345) 
v_add_u32 v0, v5, s15 // (op 2345) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 2345) Store value




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




s_lshl_b32 s15, s4, 7 // (op 2310) 
v_add_u32 v0, v4, s15 // (op 2310) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 2310) Load value

v_accvgpr_read v20, a4 // (op 2307) 
v_accvgpr_read v21, a5 // (op 2307) 
v_accvgpr_read v22, a6 // (op 2307) 
v_accvgpr_read v23, a7 // (op 2307) 
v_mul_f32 v24, v19, v20 // (op 2307) 
v_mul_f32 v25, v19, v21 // (op 2307) 
v_mul_f32 v26, v19, v22 // (op 2307) 
v_mul_f32 v27, v19, v23 // (op 2307) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 2311) 
v_mul_f32 v21, v18, v15 // (op 2311) 
v_mul_f32 v22, v18, v16 // (op 2311) 
v_mul_f32 v23, v18, v17 // (op 2311) 

v_add_f32 v14, v20, v24 // (op 2308) 
v_add_f32 v15, v21, v25 // (op 2308) 
v_add_f32 v16, v22, v26 // (op 2308) 
v_add_f32 v17, v23, v27 // (op 2308) 




s_lshl_b32 s15, s24, 7 // (op 2309) 
v_add_u32 v0, v5, s15 // (op 2309) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 2309) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ConditionalBottom_0_beta_234:
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
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_DPStreamTileLoop_441 // (op 441) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopTop_DPStreamTileLoop_441 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_441:

















































s_endpgm  // End of RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2
.LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_end:
.size RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2, .LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2_end-RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: F8
    DestDataType: s
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
  - .name: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2
    .symbol: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x64x128_UR_2.kd
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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

