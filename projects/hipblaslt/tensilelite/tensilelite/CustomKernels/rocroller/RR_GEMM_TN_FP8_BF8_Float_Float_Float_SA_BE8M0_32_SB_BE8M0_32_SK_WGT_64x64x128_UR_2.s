// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2,@function
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 5147) 
v_and_b32 v3, 15, v1 // (op 5147) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 5147) 


v_and_b32 v0, 48, v1 // (op 5147) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 5147) 


v_lshrrev_b32 v0, 6, v1 // (op 5222) 
v_and_b32 v4, 1, v0 // (op 5222) 

v_and_b32 v0, 15, v1 // (op 5222) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 5222) 


v_and_b32 v0, 48, v1 // (op 5222) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 5222) 


v_lshrrev_b32 v0, 7, v1 // (op 5298) 
v_and_b32 v5, 15, v1 // (op 5298) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 5298) 


v_and_b32 v0, 63, v1 // (op 5298) 
v_lshrrev_b32 v5, 4, v0 // (op 5298) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 5298) 


v_lshrrev_b32 v5, 6, v1 // (op 5374) 
v_and_b32 v6, 1, v5 // (op 5374) 

v_and_b32 v5, 15, v1 // (op 5374) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 5374) 


v_and_b32 v5, 63, v1 // (op 5374) 
v_lshrrev_b32 v6, 4, v5 // (op 5374) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 5374) 


v_lshrrev_b32 v6, 7, v1 // (op 5501) 
v_and_b32 v7, 15, v1 // (op 5501) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 5501) 


v_and_b32 v6, 48, v1 // (op 5501) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 5501) 


v_lshrrev_b32 v6, 6, v1 // (op 5576) 
v_and_b32 v8, 1, v6 // (op 5576) 

v_and_b32 v6, 15, v1 // (op 5576) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 5576) 


v_and_b32 v6, 48, v1 // (op 5576) 
v_lshl_add_u32 v8, v9, 7, v6 // (op 5576) 


v_lshrrev_b32 v6, 7, v1 // (op 5652) 
v_and_b32 v9, 15, v1 // (op 5652) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 5652) 


v_and_b32 v6, 63, v1 // (op 5652) 
v_lshrrev_b32 v9, 4, v6 // (op 5652) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 5652) 


v_lshrrev_b32 v9, 6, v1 // (op 5728) 
v_and_b32 v10, 1, v9 // (op 5728) 

v_and_b32 v9, 15, v1 // (op 5728) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 5728) 


v_and_b32 v9, 63, v1 // (op 5728) 
v_lshrrev_b32 v10, 4, v9 // (op 5728) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 5728) 


v_lshrrev_b32 v10, 7, v1 // (op 5842) 
v_and_b32 v11, 15, v1 // (op 5842) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 5842) 


v_and_b32 v10, 48, v1 // (op 5842) 
v_lshl_add_u32 v11, v12, 7, v10 // (op 5842) 


v_lshrrev_b32 v10, 6, v1 // (op 5886) 
v_and_b32 v12, 1, v10 // (op 5886) 

v_and_b32 v10, 15, v1 // (op 5886) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 5886) 


v_and_b32 v10, 48, v1 // (op 5886) 
v_lshl_add_u32 v12, v13, 7, v10 // (op 5886) 


v_lshrrev_b32 v10, 7, v1 // (op 5915) 
v_and_b32 v13, 15, v1 // (op 5915) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 5915) 


v_and_b32 v10, 63, v1 // (op 5915) 
v_lshrrev_b32 v13, 4, v10 // (op 5915) 

v_lshl_add_u32 v10, v14, 2, v13 // (op 5915) 


v_lshrrev_b32 v13, 6, v1 // (op 5944) 
v_and_b32 v14, 1, v13 // (op 5944) 

v_and_b32 v13, 15, v1 // (op 5944) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 5944) 


v_and_b32 v13, 63, v1 // (op 5944) 
v_lshrrev_b32 v14, 4, v13 // (op 5944) 

v_lshl_add_u32 v13, v15, 2, v14 // (op 5944) 


v_lshrrev_b32 v14, 7, v1 // (op 5988) 
v_and_b32 v15, 15, v1 // (op 5988) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 5988) 


v_and_b32 v14, 48, v1 // (op 5988) 
v_lshl_add_u32 v15, v16, 7, v14 // (op 5988) 


v_lshrrev_b32 v14, 6, v1 // (op 6032) 
v_and_b32 v16, 1, v14 // (op 6032) 

v_and_b32 v14, 15, v1 // (op 6032) 
v_lshl_add_u32 v17, v16, 4, v14 // (op 6032) 


v_and_b32 v14, 48, v1 // (op 6032) 
v_lshl_add_u32 v16, v17, 7, v14 // (op 6032) 


v_lshrrev_b32 v14, 7, v1 // (op 6061) 
v_and_b32 v17, 15, v1 // (op 6061) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 6061) 


v_and_b32 v14, 63, v1 // (op 6061) 
v_lshrrev_b32 v17, 4, v14 // (op 6061) 

v_lshl_add_u32 v14, v18, 2, v17 // (op 6061) 


v_lshrrev_b32 v17, 6, v1 // (op 6090) 
v_and_b32 v18, 1, v17 // (op 6090) 

v_and_b32 v17, 15, v1 // (op 6090) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 6090) 


v_and_b32 v17, 63, v1 // (op 6090) 
v_lshrrev_b32 v18, 4, v17 // (op 6090) 

v_lshl_add_u32 v17, v19, 2, v18 // (op 6090) 


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
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_286 // (op 286) Condition: Top (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_286 if false)
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_SKStreamTileLoop_286:
v_lshrrev_b32 v20, 7, v1 // (op 5058) 
s_mul_i32 s52, s16, s8 // (op 5058) 
s_add_u32 s52, s52, s0 // (op 5058) 
v_and_b32 v21, 63, v1 // (op 5058) 
s_mul_hi_u32 s53, s52, s13 // (op 5058) 
v_lshrrev_b32 v22, 4, v21 // (op 5058) 
s_sub_u32 s54, s52, s53 // (op 5058) 
v_lshlrev_b32 v23, 2, v22 // (op 5058) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 5058) 


s_lshr_b32 s55, s54, 1 // (op 5058) 

s_add_u32 s54, s55, s53 // (op 5058) 


v_lshrrev_b32 v20, 2, v21 // (op 5058) 

s_lshr_b32 s53, s54, s11 // (op 5058) 

v_and_b32 v21, 3, v20 // (op 5058) 

s_and_b32 s54, s11, 2147483648 // (op 5058) 
v_and_b32 v20, 3, v1 // (op 5058) 
s_cmp_eq_i32 s54, 0 // (op 5058) 
s_mov_b32 s55, scc // (op 5058) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 5058) 


s_cmp_eq_i32 s55, 0 // (op 5058) 
s_cselect_b32 s54, s52, s53 // (op 5058) 



v_lshrrev_b32 v20, 6, v1 // (op 5058) 
s_mul_hi_u32 s52, s54, s12 // (op 5058) 
v_and_b32 v21, 1, v20 // (op 5058) 

s_sub_u32 s53, s54, s52 // (op 5058) 
v_lshl_add_u32 v20, v21, 4, v23 // (op 5058) 


s_lshr_b32 s55, s53, 1 // (op 5058) 

s_add_u32 s53, s55, s52 // (op 5058) 


s_lshr_b32 s52, s53, s10 // (op 5058) 

s_and_b32 s53, s10, 2147483648 // (op 5058) 
s_cmp_eq_i32 s53, 0 // (op 5058) 
s_mov_b32 s55, scc // (op 5058) 

s_cmp_eq_i32 s55, 0 // (op 5058) 
s_cselect_b32 s53, s54, s52 // (op 5058) 


s_mul_i32 s52, s53, s14 // (op 5058) 
v_lshl_add_u32 v21, s53, 6, v22 // (op 5058) 


s_sub_u32 s53, s54, s52 // (op 5058) 


v_lshl_add_u32 v22, s53, 6, v20 // (op 5058) 


v_mul_lo_u32 v20, v22, s4 // (op 5058) 

v_add_lshl_u32 v22, v21, v20, 2 // (op 5058) 


s_mul_i32 s53, s4, s23 // (op 5060) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 5060) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 5060) least significant
s_mul_hi_u32 s55, s4, s22 // (op 5060) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 5060) most significant: sum
s_add_u32 s53, s53, s55 // (op 5060) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 5060) 

s_mov_b32 s56, s20 // (op 5060) reg expression
s_mov_b32 s57, s21 // (op 5060) reg expression
s_mov_b32 s58, s54 // (op 5060) reg expression
s_mov_b32 s59, 131072 // (op 5060) call()

v_lshrrev_b32 v20, 7, v1 // (op 5164) 
v_and_b32 v21, 15, v1 // (op 5164) 
v_lshl_add_u32 v23, v20, 4, v21 // (op 5164) 


v_and_b32 v20, 48, v1 // (op 5164) 
v_lshl_add_u32 v21, v23, 7, v20 // (op 5164) 


v_lshrrev_b32 v20, 6, v1 // (op 5239) 
v_and_b32 v23, 1, v20 // (op 5239) 

v_and_b32 v20, 15, v1 // (op 5239) 
v_lshl_add_u32 v24, v23, 4, v20 // (op 5239) 


v_and_b32 v20, 48, v1 // (op 5239) 
v_lshl_add_u32 v23, v24, 7, v20 // (op 5239) 


v_lshrrev_b32 v20, 7, v1 // (op 5315) 
v_and_b32 v24, 15, v1 // (op 5315) 
v_lshl_add_u32 v25, v20, 4, v24 // (op 5315) 


v_and_b32 v20, 63, v1 // (op 5315) 
v_lshrrev_b32 v24, 4, v20 // (op 5315) 

v_lshl_add_u32 v20, v25, 2, v24 // (op 5315) 


v_lshrrev_b32 v24, 6, v1 // (op 5391) 
v_and_b32 v25, 1, v24 // (op 5391) 

v_and_b32 v24, 15, v1 // (op 5391) 
v_lshl_add_u32 v26, v25, 4, v24 // (op 5391) 


v_and_b32 v24, 63, v1 // (op 5391) 
v_lshrrev_b32 v25, 4, v24 // (op 5391) 

v_lshl_add_u32 v24, v26, 2, v25 // (op 5391) 


v_lshrrev_b32 v25, 7, v1 // (op 5414) 
s_mul_i32 s52, s16, s8 // (op 5414) 
s_add_u32 s52, s52, s0 // (op 5414) 
v_and_b32 v26, 63, v1 // (op 5414) 
s_mul_hi_u32 s53, s52, s13 // (op 5414) 
v_lshrrev_b32 v27, 4, v26 // (op 5414) 
s_sub_u32 s54, s52, s53 // (op 5414) 
v_lshlrev_b32 v28, 2, v27 // (op 5414) 

s_lshr_b32 s55, s54, 1 // (op 5414) 

v_lshl_add_u32 v27, v25, 4, v28 // (op 5414) 


s_add_u32 s54, s55, s53 // (op 5414) 


v_lshrrev_b32 v25, 2, v26 // (op 5414) 

s_lshr_b32 s53, s54, s11 // (op 5414) 

v_and_b32 v26, 3, v25 // (op 5414) 

s_and_b32 s54, s11, 2147483648 // (op 5414) 
v_and_b32 v25, 3, v1 // (op 5414) 
s_cmp_eq_i32 s54, 0 // (op 5414) 
s_mov_b32 s55, scc // (op 5414) 

v_lshl_add_u32 v28, v26, 2, v25 // (op 5414) 


s_cmp_eq_i32 s55, 0 // (op 5414) 
s_cselect_b32 s54, s52, s53 // (op 5414) 



v_lshrrev_b32 v25, 6, v1 // (op 5414) 
s_mul_hi_u32 s55, s54, s12 // (op 5414) 
v_and_b32 v26, 1, v25 // (op 5414) 

s_sub_u32 s52, s54, s55 // (op 5414) 
v_lshl_add_u32 v25, v26, 4, v28 // (op 5414) 


s_lshr_b32 s53, s52, 1 // (op 5414) 

s_add_u32 s52, s53, s55 // (op 5414) 


s_lshr_b32 s53, s52, s10 // (op 5414) 

s_and_b32 s52, s10, 2147483648 // (op 5414) 
s_cmp_eq_i32 s52, 0 // (op 5414) 
s_mov_b32 s55, scc // (op 5414) 

s_cmp_eq_i32 s55, 0 // (op 5414) 
s_cselect_b32 s52, s54, s53 // (op 5414) 


s_mul_i32 s53, s52, s14 // (op 5414) 
v_lshl_add_u32 v26, s52, 6, v27 // (op 5414) 


s_sub_u32 s52, s54, s53 // (op 5414) 


v_lshl_add_u32 v27, s52, 6, v25 // (op 5414) 


v_mul_lo_u32 v25, v27, s24 // (op 5414) 

v_add_lshl_u32 v27, v26, v25, 2 // (op 5414) 


s_mul_i32 s53, s24, s23 // (op 5416) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 5416) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 5416) least significant
s_mul_hi_u32 s55, s24, s22 // (op 5416) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 5416) most significant: sum
s_add_u32 s53, s53, s55 // (op 5416) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 5416) 

s_mov_b32 s60, s26 // (op 5416) reg expression
s_mov_b32 s61, s27 // (op 5416) reg expression
s_mov_b32 s62, s54 // (op 5416) reg expression
s_mov_b32 s63, 131072 // (op 5416) call()

v_lshrrev_b32 v25, 5, v1 // (op 5768) 
v_lshlrev_b32 v26, 2, v25 // (op 5768) 

v_lshl_add_u32 v25, s16, 6, v26 // (op 5768) 

v_and_b32 v26, -32, v1 // (op 5768) 
v_sub_u32 v28, v1, v26 // (op 5768) 

v_lshl_add_u32 v26, v25, 6, v28 // (op 5768) 


v_lshlrev_b32 v25, 2, v26 // (op 5768) 

s_lshl_b32 s52, s15, 14 // (op 5770) 
s_mov_b32 s64, s28 // (op 5770) reg expression
s_mov_b32 s65, s29 // (op 5770) reg expression
s_mov_b32 s66, s52 // (op 5770) reg expression
s_mov_b32 s67, 131072 // (op 5770) call()

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
v_accvgpr_write a4, 0.00000 // (op 1630) call()
v_accvgpr_write a5, 0.00000 // (op 1630) call()
v_accvgpr_write a6, 0.00000 // (op 1630) call()
v_accvgpr_write a7, 0.00000 // (op 1630) call()
v_accvgpr_write a8, 0.00000 // (op 1850) call()
v_accvgpr_write a9, 0.00000 // (op 1850) call()
v_accvgpr_write a10, 0.00000 // (op 1850) call()
v_accvgpr_write a11, 0.00000 // (op 1850) call()
v_accvgpr_write a12, 0.00000 // (op 2005) call()
v_accvgpr_write a13, 0.00000 // (op 2005) call()
v_accvgpr_write a14, 0.00000 // (op 2005) call()
v_accvgpr_write a15, 0.00000 // (op 2005) call()
v_lshrrev_b32 v26, 3, v1 // (op 4625) 
s_mul_i32 s52, s16, s8 // (op 4625) 
s_add_u32 s52, s52, s0 // (op 4625) 
v_and_b32 v28, -8, v1 // (op 4625) 
s_mul_hi_u32 s53, s52, s13 // (op 4625) 
v_sub_u32 v29, v1, v28 // (op 4625) 

s_sub_u32 s54, s52, s53 // (op 4625) 
v_lshlrev_b32 v28, 4, v29 // (op 4625) 

s_lshr_b32 s55, s54, 1 // (op 4625) 

s_add_u32 s54, s55, s53 // (op 4625) 


s_lshr_b32 s53, s54, s11 // (op 4625) 

s_and_b32 s54, s11, 2147483648 // (op 4625) 
s_cmp_eq_i32 s54, 0 // (op 4625) 
s_mov_b32 s55, scc // (op 4625) 

s_cmp_eq_i32 s55, 0 // (op 4625) 
s_cselect_b32 s54, s52, s53 // (op 4625) 


s_mul_hi_u32 s53, s54, s12 // (op 4625) 
s_sub_u32 s55, s54, s53 // (op 4625) 
s_lshr_b32 s68, s55, 1 // (op 4625) 

s_add_u32 s55, s68, s53 // (op 4625) 


s_lshr_b32 s53, s55, s10 // (op 4625) 

s_and_b32 s55, s10, 2147483648 // (op 4625) 
s_cmp_eq_i32 s55, 0 // (op 4625) 
s_mov_b32 s68, scc // (op 4625) 

s_cmp_eq_i32 s68, 0 // (op 4625) 
s_cselect_b32 s55, s54, s53 // (op 4625) 


v_lshl_add_u32 v29, s55, 6, v26 // (op 4625) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 4625) 
s_mul_i32 s53, s54, s68 // (op 4625) 


s_sub_u32 s54, s52, s53 // (op 4625) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 4625) 


v_mov_b32 v31, 0 // (op 4625) convert
v_mov_b32 v30, v26 // (op 4625) convert

v_mul_lo_u32 v33, v29, s31 // (op 4625) most significant: low of low * high
v_mul_hi_u32 v26, v29, s30 // (op 4625) most significant: high of low * low
v_mul_lo_u32 v32, v29, s30 // (op 4625) least significant: low of low * low
v_add_u32 v33, v33, v26 // (op 4625) most significant: sum

v_add_co_u32 v32, s[52:53], v32, v30 // (op 4625) least significant half
v_addc_co_u32 v33, s[52:53], v33, v31, s[52:53] // (op 4625) most significant half



s_mul_i32 s53, s30, s35 // (op 4627) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 4627) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 4627) least significant
s_mul_hi_u32 s55, s30, s34 // (op 4627) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4627) most significant: sum
s_add_u32 s53, s53, s55 // (op 4627) most significant: sum


s_mov_b32 s68, s32 // (op 4627) reg expression
s_mov_b32 s69, s33 // (op 4627) reg expression
s_mov_b32 s70, s52 // (op 4627) reg expression
s_mov_b32 s71, 131072 // (op 4627) call()

v_lshrrev_b32 v26, 3, v1 // (op 4644) 
s_mul_i32 s52, s16, s8 // (op 4644) 
s_add_u32 s52, s52, s0 // (op 4644) 
v_and_b32 v28, -8, v1 // (op 4644) 
s_mul_hi_u32 s53, s52, s13 // (op 4644) 
v_sub_u32 v29, v1, v28 // (op 4644) 

s_sub_u32 s54, s52, s53 // (op 4644) 
v_lshlrev_b32 v28, 4, v29 // (op 4644) 

s_lshr_b32 s55, s54, 1 // (op 4644) 

s_add_u32 s54, s55, s53 // (op 4644) 


s_lshr_b32 s53, s54, s11 // (op 4644) 

s_and_b32 s54, s11, 2147483648 // (op 4644) 
s_cmp_eq_i32 s54, 0 // (op 4644) 
s_mov_b32 s55, scc // (op 4644) 

s_cmp_eq_i32 s55, 0 // (op 4644) 
s_cselect_b32 s54, s52, s53 // (op 4644) 


s_mul_hi_u32 s53, s54, s12 // (op 4644) 
s_sub_u32 s55, s54, s53 // (op 4644) 
s_lshr_b32 s72, s55, 1 // (op 4644) 

s_add_u32 s55, s72, s53 // (op 4644) 


s_lshr_b32 s53, s55, s10 // (op 4644) 

s_and_b32 s55, s10, 2147483648 // (op 4644) 
s_cmp_eq_i32 s55, 0 // (op 4644) 
s_mov_b32 s72, scc // (op 4644) 

s_cmp_eq_i32 s72, 0 // (op 4644) 
s_cselect_b32 s55, s54, s53 // (op 4644) 


v_lshl_add_u32 v29, s55, 6, v26 // (op 4644) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 4644) 
s_mul_i32 s53, s54, s72 // (op 4644) 


s_sub_u32 s54, s52, s53 // (op 4644) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 4644) 


v_mov_b32 v31, 0 // (op 4644) convert
v_mov_b32 v30, v26 // (op 4644) convert

v_mul_lo_u32 v35, v29, s31 // (op 4644) most significant: low of low * high
v_mul_hi_u32 v26, v29, s30 // (op 4644) most significant: high of low * low
v_mul_lo_u32 v34, v29, s30 // (op 4644) least significant: low of low * low
v_add_u32 v35, v35, v26 // (op 4644) most significant: sum

v_add_co_u32 v34, s[52:53], v34, v30 // (op 4644) least significant half
v_addc_co_u32 v35, s[52:53], v35, v31, s[52:53] // (op 4644) most significant half



s_mul_i32 s53, s30, s35 // (op 4646) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 4646) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 4646) least significant
s_mul_hi_u32 s55, s30, s34 // (op 4646) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4646) most significant: sum
s_add_u32 s53, s53, s55 // (op 4646) most significant: sum


s_mov_b32 s68, s32 // (op 4646) reg expression
s_mov_b32 s69, s33 // (op 4646) reg expression
s_mov_b32 s70, s52 // (op 4646) reg expression
s_mov_b32 s71, 131072 // (op 4646) call()

v_lshrrev_b32 v26, 3, v1 // (op 4753) 
s_mul_i32 s52, s16, s8 // (op 4753) 
s_add_u32 s52, s52, s0 // (op 4753) 
v_and_b32 v28, -8, v1 // (op 4753) 
s_mul_hi_u32 s53, s52, s13 // (op 4753) 
v_sub_u32 v29, v1, v28 // (op 4753) 

s_sub_u32 s54, s52, s53 // (op 4753) 
v_lshlrev_b32 v28, 4, v29 // (op 4753) 

s_lshr_b32 s55, s54, 1 // (op 4753) 

s_add_u32 s54, s55, s53 // (op 4753) 


s_lshr_b32 s53, s54, s11 // (op 4753) 

s_and_b32 s54, s11, 2147483648 // (op 4753) 
s_cmp_eq_i32 s54, 0 // (op 4753) 
s_mov_b32 s55, scc // (op 4753) 

s_cmp_eq_i32 s55, 0 // (op 4753) 
s_cselect_b32 s54, s52, s53 // (op 4753) 


s_mul_hi_u32 s53, s54, s12 // (op 4753) 
s_sub_u32 s55, s54, s53 // (op 4753) 
s_lshr_b32 s72, s55, 1 // (op 4753) 

s_add_u32 s55, s72, s53 // (op 4753) 


s_lshr_b32 s53, s55, s10 // (op 4753) 

s_and_b32 s55, s10, 2147483648 // (op 4753) 
s_cmp_eq_i32 s55, 0 // (op 4753) 
s_mov_b32 s72, scc // (op 4753) 

s_cmp_eq_i32 s72, 0 // (op 4753) 
s_cselect_b32 s55, s54, s53 // (op 4753) 


s_mul_i32 s53, s55, s14 // (op 4753) 

s_sub_u32 s55, s54, s53 // (op 4753) 

v_lshl_add_u32 v29, s55, 6, v26 // (op 4753) 


v_mul_lo_u32 v31, v29, s37 // (op 4753) most significant: low of low * high
v_mul_hi_u32 v26, v29, s36 // (op 4753) most significant: high of low * low
v_mul_lo_u32 v30, v29, s36 // (op 4753) least significant: low of low * low
v_add_u32 v31, v31, v26 // (op 4753) most significant: sum


s_ashr_i64 s[72:73], s[2:3], 7 // (op 4753) 
s_mul_i32 s53, s54, s72 // (op 4753) 


s_sub_u32 s54, s52, s53 // (op 4753) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 4753) 


v_mov_b32 v29, 0 // (op 4753) convert
v_mov_b32 v28, v26 // (op 4753) convert

v_add_co_u32 v36, s[52:53], v28, v30 // (op 4753) least significant half
v_addc_co_u32 v37, s[52:53], v29, v31, s[52:53] // (op 4753) most significant half



s_mul_i32 s53, s36, s23 // (op 4755) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 4755) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 4755) least significant
s_mul_hi_u32 s55, s36, s22 // (op 4755) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4755) most significant: sum
s_add_u32 s53, s53, s55 // (op 4755) most significant: sum


s_mov_b32 s72, s38 // (op 4755) reg expression
s_mov_b32 s73, s39 // (op 4755) reg expression
s_mov_b32 s74, s52 // (op 4755) reg expression
s_mov_b32 s75, 131072 // (op 4755) call()

v_lshrrev_b32 v26, 3, v1 // (op 4772) 
s_mul_i32 s52, s16, s8 // (op 4772) 
s_add_u32 s52, s52, s0 // (op 4772) 
v_and_b32 v28, -8, v1 // (op 4772) 
s_mul_hi_u32 s53, s52, s13 // (op 4772) 
v_sub_u32 v29, v1, v28 // (op 4772) 

s_sub_u32 s54, s52, s53 // (op 4772) 
v_lshlrev_b32 v28, 4, v29 // (op 4772) 

s_lshr_b32 s55, s54, 1 // (op 4772) 

s_add_u32 s54, s55, s53 // (op 4772) 


s_lshr_b32 s53, s54, s11 // (op 4772) 

s_and_b32 s54, s11, 2147483648 // (op 4772) 
s_cmp_eq_i32 s54, 0 // (op 4772) 
s_mov_b32 s55, scc // (op 4772) 

s_cmp_eq_i32 s55, 0 // (op 4772) 
s_cselect_b32 s54, s52, s53 // (op 4772) 


s_mul_hi_u32 s53, s54, s12 // (op 4772) 
s_sub_u32 s55, s54, s53 // (op 4772) 
s_lshr_b32 s76, s55, 1 // (op 4772) 

s_add_u32 s55, s76, s53 // (op 4772) 


s_lshr_b32 s53, s55, s10 // (op 4772) 

s_and_b32 s55, s10, 2147483648 // (op 4772) 
s_cmp_eq_i32 s55, 0 // (op 4772) 
s_mov_b32 s76, scc // (op 4772) 

s_cmp_eq_i32 s76, 0 // (op 4772) 
s_cselect_b32 s55, s54, s53 // (op 4772) 


s_mul_i32 s53, s55, s14 // (op 4772) 

s_sub_u32 s55, s54, s53 // (op 4772) 

v_lshl_add_u32 v29, s55, 6, v26 // (op 4772) 


v_mul_lo_u32 v31, v29, s37 // (op 4772) most significant: low of low * high
v_mul_hi_u32 v26, v29, s36 // (op 4772) most significant: high of low * low
v_mul_lo_u32 v30, v29, s36 // (op 4772) least significant: low of low * low
v_add_u32 v31, v31, v26 // (op 4772) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 4772) 
s_mul_i32 s53, s54, s76 // (op 4772) 


s_sub_u32 s54, s52, s53 // (op 4772) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 4772) 


v_mov_b32 v29, 0 // (op 4772) convert
v_mov_b32 v28, v26 // (op 4772) convert

v_add_co_u32 v38, s[52:53], v28, v30 // (op 4772) least significant half
v_addc_co_u32 v39, s[52:53], v29, v31, s[52:53] // (op 4772) most significant half



s_mul_i32 s53, s36, s23 // (op 4774) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 4774) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 4774) least significant
s_mul_hi_u32 s55, s36, s22 // (op 4774) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4774) most significant: sum
s_add_u32 s53, s53, s55 // (op 4774) most significant: sum


s_mov_b32 s72, s38 // (op 4774) reg expression
s_mov_b32 s73, s39 // (op 4774) reg expression
s_mov_b32 s74, s52 // (op 4774) reg expression
s_mov_b32 s75, 131072 // (op 4774) call()

v_lshrrev_b32 v26, 2, v1 // (op 4881) 
s_mul_i32 s52, s16, s8 // (op 4881) 
s_add_u32 s52, s52, s0 // (op 4881) 
v_and_b32 v28, -4, v1 // (op 4881) 
s_mul_hi_u32 s53, s52, s13 // (op 4881) 
v_sub_u32 v29, v1, v28 // (op 4881) 

s_sub_u32 s54, s52, s53 // (op 4881) 
s_lshr_b32 s55, s54, 1 // (op 4881) 

s_add_u32 s54, s55, s53 // (op 4881) 


s_lshr_b32 s53, s54, s11 // (op 4881) 

s_and_b32 s54, s11, 2147483648 // (op 4881) 
s_cmp_eq_i32 s54, 0 // (op 4881) 
s_mov_b32 s55, scc // (op 4881) 

s_cmp_eq_i32 s55, 0 // (op 4881) 
s_cselect_b32 s54, s52, s53 // (op 4881) 


s_mul_hi_u32 s53, s54, s12 // (op 4881) 
s_sub_u32 s55, s54, s53 // (op 4881) 
s_lshr_b32 s76, s55, 1 // (op 4881) 

s_add_u32 s55, s76, s53 // (op 4881) 


s_lshr_b32 s53, s55, s10 // (op 4881) 

s_and_b32 s55, s10, 2147483648 // (op 4881) 
s_cmp_eq_i32 s55, 0 // (op 4881) 
s_mov_b32 s76, scc // (op 4881) 

s_cmp_eq_i32 s76, 0 // (op 4881) 
s_cselect_b32 s55, s54, s53 // (op 4881) 


v_lshl_add_u32 v28, s55, 6, v26 // (op 4881) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 4881) 
s_mul_i32 s53, s54, s76 // (op 4881) 


s_sub_u32 s54, s52, s53 // (op 4881) 


v_lshl_add_u32 v26, s54, 2, v29 // (op 4881) 


v_mul_lo_u32 v29, v28, s40 // (op 4881) 
v_add_u32 v29, v29, v26 // (op 4881) 


s_mul_i32 s53, s40, s35 // (op 4883) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 4883) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 4883) least significant
s_mul_hi_u32 s55, s40, s34 // (op 4883) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4883) most significant: sum
s_add_u32 s53, s53, s55 // (op 4883) most significant: sum


s_mov_b32 s76, s42 // (op 4883) reg expression
s_mov_b32 s77, s43 // (op 4883) reg expression
s_mov_b32 s78, s52 // (op 4883) reg expression
s_mov_b32 s79, 131072 // (op 4883) call()

v_lshrrev_b32 v26, 2, v1 // (op 4900) 
s_mul_i32 s52, s16, s8 // (op 4900) 
s_add_u32 s52, s52, s0 // (op 4900) 
v_and_b32 v28, -4, v1 // (op 4900) 
s_mul_hi_u32 s53, s52, s13 // (op 4900) 
v_sub_u32 v30, v1, v28 // (op 4900) 

s_sub_u32 s54, s52, s53 // (op 4900) 
s_lshr_b32 s55, s54, 1 // (op 4900) 

s_add_u32 s54, s55, s53 // (op 4900) 


s_lshr_b32 s53, s54, s11 // (op 4900) 

s_and_b32 s54, s11, 2147483648 // (op 4900) 
s_cmp_eq_i32 s54, 0 // (op 4900) 
s_mov_b32 s55, scc // (op 4900) 

s_cmp_eq_i32 s55, 0 // (op 4900) 
s_cselect_b32 s54, s52, s53 // (op 4900) 


s_mul_hi_u32 s53, s54, s12 // (op 4900) 
s_sub_u32 s55, s54, s53 // (op 4900) 
s_lshr_b32 s80, s55, 1 // (op 4900) 

s_add_u32 s55, s80, s53 // (op 4900) 


s_lshr_b32 s53, s55, s10 // (op 4900) 

s_and_b32 s55, s10, 2147483648 // (op 4900) 
s_cmp_eq_i32 s55, 0 // (op 4900) 
s_mov_b32 s80, scc // (op 4900) 

s_cmp_eq_i32 s80, 0 // (op 4900) 
s_cselect_b32 s55, s54, s53 // (op 4900) 


v_lshl_add_u32 v28, s55, 6, v26 // (op 4900) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 4900) 
s_mul_i32 s53, s54, s80 // (op 4900) 


s_sub_u32 s54, s52, s53 // (op 4900) 


v_lshl_add_u32 v26, s54, 2, v30 // (op 4900) 


v_mul_lo_u32 v30, v28, s40 // (op 4900) 
v_add_u32 v30, v30, v26 // (op 4900) 


s_mul_i32 s53, s40, s35 // (op 4902) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 4902) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 4902) least significant
s_mul_hi_u32 s55, s40, s34 // (op 4902) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4902) most significant: sum
s_add_u32 s53, s53, s55 // (op 4902) most significant: sum


s_mov_b32 s76, s42 // (op 4902) reg expression
s_mov_b32 s77, s43 // (op 4902) reg expression
s_mov_b32 s78, s52 // (op 4902) reg expression
s_mov_b32 s79, 131072 // (op 4902) call()

v_lshrrev_b32 v26, 2, v1 // (op 5009) 
s_mul_i32 s52, s16, s8 // (op 5009) 
s_add_u32 s52, s52, s0 // (op 5009) 
v_and_b32 v28, -4, v1 // (op 5009) 
s_mul_hi_u32 s53, s52, s13 // (op 5009) 
v_sub_u32 v31, v1, v28 // (op 5009) 

s_sub_u32 s54, s52, s53 // (op 5009) 
s_lshr_b32 s55, s54, 1 // (op 5009) 

s_add_u32 s54, s55, s53 // (op 5009) 


s_lshr_b32 s53, s54, s11 // (op 5009) 

s_and_b32 s54, s11, 2147483648 // (op 5009) 
s_cmp_eq_i32 s54, 0 // (op 5009) 
s_mov_b32 s55, scc // (op 5009) 

s_cmp_eq_i32 s55, 0 // (op 5009) 
s_cselect_b32 s54, s52, s53 // (op 5009) 


s_mul_hi_u32 s53, s54, s12 // (op 5009) 
s_sub_u32 s55, s54, s53 // (op 5009) 
s_lshr_b32 s80, s55, 1 // (op 5009) 

s_add_u32 s55, s80, s53 // (op 5009) 


s_lshr_b32 s53, s55, s10 // (op 5009) 

s_and_b32 s55, s10, 2147483648 // (op 5009) 
s_cmp_eq_i32 s55, 0 // (op 5009) 
s_mov_b32 s80, scc // (op 5009) 

s_cmp_eq_i32 s80, 0 // (op 5009) 
s_cselect_b32 s55, s54, s53 // (op 5009) 


s_mul_i32 s53, s55, s14 // (op 5009) 

s_sub_u32 s55, s54, s53 // (op 5009) 

v_lshl_add_u32 v28, s55, 6, v26 // (op 5009) 


v_mul_lo_u32 v26, v28, s44 // (op 5009) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 5009) 
s_mul_i32 s53, s54, s80 // (op 5009) 


s_sub_u32 s54, s52, s53 // (op 5009) 


v_lshl_add_u32 v28, s54, 2, v31 // (op 5009) 


v_add_u32 v31, v28, v26 // (op 5009) 


s_mul_i32 s53, s44, s23 // (op 5011) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 5011) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 5011) least significant
s_mul_hi_u32 s55, s44, s22 // (op 5011) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 5011) most significant: sum
s_add_u32 s53, s53, s55 // (op 5011) most significant: sum


s_mov_b32 s80, s46 // (op 5011) reg expression
s_mov_b32 s81, s47 // (op 5011) reg expression
s_mov_b32 s82, s52 // (op 5011) reg expression
s_mov_b32 s83, 131072 // (op 5011) call()

v_lshrrev_b32 v26, 2, v1 // (op 5028) 
s_mul_i32 s52, s16, s8 // (op 5028) 
s_add_u32 s52, s52, s0 // (op 5028) 
v_and_b32 v28, -4, v1 // (op 5028) 
s_mul_hi_u32 s53, s52, s13 // (op 5028) 
v_sub_u32 v40, v1, v28 // (op 5028) 

s_sub_u32 s54, s52, s53 // (op 5028) 
s_lshr_b32 s55, s54, 1 // (op 5028) 

s_add_u32 s54, s55, s53 // (op 5028) 


s_lshr_b32 s53, s54, s11 // (op 5028) 

s_and_b32 s54, s11, 2147483648 // (op 5028) 
s_cmp_eq_i32 s54, 0 // (op 5028) 
s_mov_b32 s55, scc // (op 5028) 

s_cmp_eq_i32 s55, 0 // (op 5028) 
s_cselect_b32 s54, s52, s53 // (op 5028) 


s_mul_hi_u32 s53, s54, s12 // (op 5028) 
s_sub_u32 s55, s54, s53 // (op 5028) 
s_lshr_b32 s84, s55, 1 // (op 5028) 

s_add_u32 s55, s84, s53 // (op 5028) 


s_lshr_b32 s53, s55, s10 // (op 5028) 

s_and_b32 s55, s10, 2147483648 // (op 5028) 
s_cmp_eq_i32 s55, 0 // (op 5028) 
s_mov_b32 s84, scc // (op 5028) 

s_cmp_eq_i32 s84, 0 // (op 5028) 
s_cselect_b32 s55, s54, s53 // (op 5028) 


s_mul_i32 s53, s55, s14 // (op 5028) 

s_sub_u32 s55, s54, s53 // (op 5028) 

v_lshl_add_u32 v28, s55, 6, v26 // (op 5028) 


v_mul_lo_u32 v26, v28, s44 // (op 5028) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 5028) 
s_mul_i32 s53, s54, s84 // (op 5028) 


s_sub_u32 s54, s52, s53 // (op 5028) 


v_lshl_add_u32 v28, s54, 2, v40 // (op 5028) 


v_add_u32 v40, v28, v26 // (op 5028) 


s_mul_i32 s53, s44, s23 // (op 5030) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 5030) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 5030) least significant
s_mul_hi_u32 s55, s44, s22 // (op 5030) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 5030) most significant: sum
s_add_u32 s53, s53, s55 // (op 5030) most significant: sum


s_mov_b32 s80, s46 // (op 5030) reg expression
s_mov_b32 s81, s47 // (op 5030) reg expression
s_mov_b32 s82, s52 // (op 5030) reg expression
s_mov_b32 s83, 131072 // (op 5030) call()

v_and_b32 v26, -8, v1 // (op 5205) 
v_sub_u32 v28, v1, v26 // (op 5205) 

v_lshlrev_b32 v26, 4, v28 // (op 5205) 

v_lshrrev_b32 v28, 3, v1 // (op 5205) 
v_lshl_add_u32 v41, v28, 7, v26 // (op 5205) 


s_nop 0
v_readfirstlane_b32 s52, v41 // (op 5205) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6237) 

s_lshl_b64 s[54:55], s[36:37], 5 // (op 4499) 
v_mov_b64 v[42:43], s[54:55] // (op 4499) convert


v_mov_b64 v[44:45], v[36:37] // (op 4499) 
s_mov_b32 m0, s52 // (op 4499) reg expression
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 4499) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 4499) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 4499) most significant half

s_add_u32 m0, m0, 4096 // (op 4499) 
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 4499) Load value direct to lds






buffer_load_ubyte v26, v31, s[80:83], 0 offen offset: 0 // (op 4282) Load value
v_and_b32 v28, -8, v1 // (op 5130) 
v_sub_u32 v41, v1, v28 // (op 5130) 

v_lshlrev_b32 v28, 4, v41 // (op 5130) 

v_lshrrev_b32 v41, 3, v1 // (op 5130) 
v_lshl_add_u32 v42, v41, 7, v28 // (op 5130) 


s_nop 0
v_readfirstlane_b32 s52, v42 // (op 5130) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6218) 

s_lshl_b64 s[54:55], s[30:31], 5 // (op 4504) 
v_mov_b64 v[42:43], s[54:55] // (op 4504) convert


v_mov_b64 v[44:45], v[32:33] // (op 4504) 
s_add_u32 s53, 8192, s52 // (op 4504) 
s_mov_b32 m0, s53 // (op 4504) reg expression
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 4504) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 4504) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 4504) most significant half

s_add_u32 m0, m0, 4096 // (op 4504) 
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 4504) Load value direct to lds







buffer_load_ubyte v28, v29, s[76:79], 0 offen offset: 0 // (op 4288) Load value
v_and_b32 v41, -8, v1 // (op 6015) 
v_sub_u32 v42, v1, v41 // (op 6015) 

v_lshlrev_b32 v41, 4, v42 // (op 6015) 

v_lshrrev_b32 v42, 3, v1 // (op 6015) 
v_lshl_add_u32 v43, v42, 7, v41 // (op 6015) 


s_nop 0
v_readfirstlane_b32 s52, v43 // (op 6015) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6417) 

s_lshl_b64 s[54:55], s[36:37], 5 // (op 4509) 
v_mov_b64 v[42:43], s[54:55] // (op 4509) convert


v_mov_b64 v[44:45], v[36:37] // (op 4509) 
s_add_u32 s53, 16384, s52 // (op 4509) 
s_mov_b32 m0, s53 // (op 4509) reg expression
s_mov_b32 s54, 128 // (op 4509) call()
buffer_load_dwordx4 v44, s[72:75], s54 offen offset: 0 lds // (op 4509) Load value direct to lds
v_add_co_u32 v44, s[84:85], v44, v42 // (op 4509) least significant half
v_addc_co_u32 v45, s[84:85], 0, v45, s[84:85] // (op 4509) most significant half

s_add_u32 m0, m0, 4096 // (op 4509) 
buffer_load_dwordx4 v44, s[72:75], s54 offen offset: 0 lds // (op 4509) Load value direct to lds









buffer_load_ubyte v36, v31, s[80:83], 0 offen offset:4 // (op 4294) Load value

v_and_b32 v31, -8, v1 // (op 5971) 
v_sub_u32 v37, v1, v31 // (op 5971) 

v_lshlrev_b32 v31, 4, v37 // (op 5971) 

v_lshrrev_b32 v37, 3, v1 // (op 5971) 
v_lshl_add_u32 v41, v37, 7, v31 // (op 5971) 


s_nop 0
v_readfirstlane_b32 s52, v41 // (op 5971) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6407) 

s_lshl_b64 s[54:55], s[30:31], 5 // (op 4511) 
v_mov_b64 v[42:43], s[54:55] // (op 4511) convert


v_mov_b64 v[44:45], v[32:33] // (op 4511) 
s_add_u32 s53, 24576, s52 // (op 4511) 
s_mov_b32 m0, s53 // (op 4511) reg expression
s_mov_b32 s54, 128 // (op 4511) call()
buffer_load_dwordx4 v44, s[68:71], s54 offen offset: 0 lds // (op 4511) Load value direct to lds
v_add_co_u32 v44, s[84:85], v44, v42 // (op 4511) least significant half
v_addc_co_u32 v45, s[84:85], 0, v45, s[84:85] // (op 4511) most significant half

s_add_u32 m0, m0, 4096 // (op 4511) 
buffer_load_dwordx4 v44, s[68:71], s54 offen offset: 0 lds // (op 4511) Load value direct to lds









buffer_load_ubyte v37, v29, s[76:79], 0 offen offset:4 // (op 4300) Load value

v_and_b32 v29, -4, v1 // (op 5357) 
v_sub_u32 v31, v1, v29 // (op 5357) 

v_and_b32 v29, 4294967292, v1 // (op 5357) 
v_add_u32 v32, v29, v31 // (op 5357) 


s_barrier  // (op 6277) 



s_waitcnt vmcnt(9)
ds_write_b8 v32, v26 offset:32768 // (op 4303) Store local data 

v_and_b32 v29, -4, v1 // (op 5281) 
v_sub_u32 v31, v1, v29 // (op 5281) 

v_and_b32 v29, 4294967292, v1 // (op 5281) 
v_add_u32 v32, v29, v31 // (op 5281) 


s_barrier  // (op 6257) 



s_waitcnt vmcnt(6)
ds_write_b8 v32, v28 offset:33024 // (op 4305) Store local data 

s_barrier  // (op 6234) 



ds_read_b128 v[42:45], v23 // (op 4330) Load local data 
ds_read_b128 v[46:49], v23 offset:64 // (op 4330) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 6274) 



ds_read_u8 v29, v24 offset:32768 // (op 4350) Load local data 



ds_read_b128 v[50:53], v23 offset:4096 // (op 4335) Load local data 
ds_read_b128 v[54:57], v23 offset:4160 // (op 4335) Load local data 



ds_read_u8 v41, v24 offset:32896 // (op 4355) Load local data 
s_barrier  // (op 6215) 



ds_read_b128 v[58:61], v21 offset:8192 // (op 4320) Load local data 
ds_read_b128 v[62:65], v21 offset:8256 // (op 4320) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 6254) 



ds_read_u8 v31, v20 offset:33024 // (op 4340) Load local data 



ds_read_b128 v[66:69], v21 offset:12288 // (op 4325) Load local data 
ds_read_b128 v[70:73], v21 offset:12352 // (op 4325) Load local data 



ds_read_u8 v32, v20 offset:33152 // (op 4345) Load local data 
v_and_b32 v33, -8, v1 // (op 5102) 
v_sub_u32 v74, v1, v33 // (op 5102) 

v_lshlrev_b32 v33, 4, v74 // (op 5102) 

v_lshrrev_b32 v74, 3, v1 // (op 5102) 
v_lshl_add_u32 v75, v74, 7, v33 // (op 5102) 


s_nop 0
v_readfirstlane_b32 s52, v75 // (op 5102) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -8, v1 // (op 5177) 
v_sub_u32 v74, v1, v33 // (op 5177) 

v_lshlrev_b32 v33, 4, v74 // (op 5177) 

v_lshrrev_b32 v74, 3, v1 // (op 5177) 
v_lshl_add_u32 v75, v74, 7, v33 // (op 5177) 


s_nop 0
v_readfirstlane_b32 s53, v75 // (op 5177) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 5252) 
v_sub_u32 v74, v1, v33 // (op 5252) 

v_and_b32 v33, 4294967292, v1 // (op 5252) 
v_add_u32 v75, v33, v74 // (op 5252) 


v_and_b32 v33, -4, v1 // (op 5328) 
v_sub_u32 v74, v1, v33 // (op 5328) 

v_and_b32 v33, 4294967292, v1 // (op 5328) 
v_add_u32 v76, v33, v74 // (op 5328) 


v_and_b32 v33, -8, v1 // (op 5956) 
v_sub_u32 v74, v1, v33 // (op 5956) 

v_lshlrev_b32 v33, 4, v74 // (op 5956) 

v_lshrrev_b32 v74, 3, v1 // (op 5956) 
v_lshl_add_u32 v77, v74, 7, v33 // (op 5956) 


s_nop 0
v_readfirstlane_b32 s54, v77 // (op 5956) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -8, v1 // (op 6000) 
v_sub_u32 v74, v1, v33 // (op 6000) 

v_lshlrev_b32 v33, 4, v74 // (op 6000) 

v_lshrrev_b32 v74, 3, v1 // (op 6000) 
v_lshl_add_u32 v77, v74, 7, v33 // (op 6000) 


s_nop 0
v_readfirstlane_b32 s55, v77 // (op 6000) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 6044) 
v_sub_u32 v74, v1, v33 // (op 6044) 

v_and_b32 v33, 4294967292, v1 // (op 6044) 
v_add_u32 v77, v33, v74 // (op 6044) 


v_and_b32 v33, -4, v1 // (op 6073) 
v_sub_u32 v74, v1, v33 // (op 6073) 

v_and_b32 v33, 4294967292, v1 // (op 6073) 
v_add_u32 v78, v33, v74 // (op 6073) 


s_mov_b32 s1, 0 // (op 1883) call()
s_and_b32 s84, s7, 4294967294 // (op 1882) 
s_cmp_lt_u32 s1, s84 // (op 1882) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_1882 // (op 1882) Condition: Top (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_1882 if false)
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoop_1882:
s_add_u32 s1, s1, 2 // (op 1884) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:65], v[42:49], a[0:3], v31, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 88) 
s_barrier  // (op 6240) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:65], v[50:57], a[4:7], v31, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1670) 
s_lshl_b64 s[84:85], s[36:37], 5 // (op 4424) 
v_mov_b64 v[80:81], s[84:85] // (op 4424) convert


v_mov_b64 v[82:83], v[38:39] // (op 4424) 
v_mov_b64 v[84:85], v[82:83] // (op 4424) 
s_mov_b32 m0, s53 // (op 4424) reg expression
s_mov_b32 s84, 256 // (op 4424) call()
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:73], v[42:49], a[8:11], v32, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1967) 
buffer_load_dwordx4 v84, s[72:75], s84 offen offset: 0 lds // (op 4424) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:73], v[50:57], a[12:15], v32, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1921) 
v_add_co_u32 v84, s[86:87], v84, v80 // (op 4424) least significant half
v_addc_co_u32 v85, s[86:87], 0, v85, s[86:87] // (op 4424) most significant half

s_add_u32 m0, m0, 4096 // (op 4424) 
buffer_load_dwordx4 v84, s[72:75], s84 offen offset: 0 lds // (op 4424) Load value direct to lds





v_mov_b32 v33, v40 // (op 25) 

buffer_load_ubyte v26, v33, s[80:83], 0 offen offset:8 // (op 25) Load value
s_barrier  // (op 6221) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 4419) 
v_mov_b64 v[80:81], s[84:85] // (op 4419) convert


v_mov_b64 v[84:85], v[34:35] // (op 4419) 
v_mov_b64 v[86:87], v[84:85] // (op 4419) 
s_add_u32 s84, 8192, s52 // (op 4419) 
s_mov_b32 m0, s84 // (op 4419) reg expression
s_mov_b32 s85, 256 // (op 4419) call()
buffer_load_dwordx4 v86, s[68:71], s85 offen offset: 0 lds // (op 4419) Load value direct to lds
v_add_co_u32 v86, s[86:87], v86, v80 // (op 4419) least significant half
v_addc_co_u32 v87, s[86:87], 0, v87, s[86:87] // (op 4419) most significant half

s_add_u32 m0, m0, 4096 // (op 4419) 
buffer_load_dwordx4 v86, s[68:71], s85 offen offset: 0 lds // (op 4419) Load value direct to lds






v_mov_b32 v74, v30 // (op 17) 

buffer_load_ubyte v28, v74, s[76:79], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 6432) 



s_waitcnt vmcnt(9)
ds_write_b8 v78, v36 offset:33280 // (op 1549) Store local data 
s_barrier  // (op 6426) 



s_waitcnt vmcnt(6)
ds_write_b8 v77, v37 offset:33536 // (op 1548) Store local data 
s_barrier  // (op 6413) 



ds_read_b128 v[42:45], v16 offset:16384 // (op 1541) Load local data 
ds_read_b128 v[46:49], v16 offset:16448 // (op 1541) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 6429) 



ds_read_u8 v29, v17 offset:33280 // (op 1545) Load local data 



ds_read_b128 v[50:53], v16 offset:20480 // (op 1675) Load local data 
ds_read_b128 v[54:57], v16 offset:20544 // (op 1675) Load local data 



ds_read_u8 v41, v17 offset:33408 // (op 1679) Load local data 
s_barrier  // (op 6404) 
s_barrier  // (op 6595) 



ds_read_b128 v[58:61], v15 offset:24576 // (op 1539) Load local data 
ds_read_b128 v[62:65], v15 offset:24640 // (op 1539) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 6423) 



ds_read_u8 v31, v14 offset:33536 // (op 1543) Load local data 



ds_read_b128 v[66:69], v15 offset:28672 // (op 1924) Load local data 
ds_read_b128 v[70:73], v15 offset:28736 // (op 1924) Load local data 



ds_read_u8 v32, v14 offset:33664 // (op 1927) Load local data 
s_barrier  // (op 6617) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:65], v[42:49], a[0:3], v31, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1546) 
s_barrier  // (op 6420) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:65], v[50:57], a[4:7], v31, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1671) 
s_lshl_b64 s[84:85], s[36:37], 5 // (op 4480) 
v_mov_b64 v[80:81], s[84:85] // (op 4480) convert


v_mov_b64 v[86:87], v[82:83] // (op 4480) 
s_add_u32 s84, 16384, s55 // (op 4480) 
s_mov_b32 m0, s84 // (op 4480) reg expression
s_mov_b32 s85, 384 // (op 4480) call()
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:73], v[42:49], a[8:11], v32, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1968) 
buffer_load_dwordx4 v86, s[72:75], s85 offen offset: 0 lds // (op 4480) Load value direct to lds
v_add_co_u32 v86, s[86:87], v86, v80 // (op 4480) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:73], v[50:57], a[12:15], v32, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1922) 
v_addc_co_u32 v87, s[86:87], 0, v87, s[86:87] // (op 4480) most significant half

s_add_u32 m0, m0, 4096 // (op 4480) 
buffer_load_dwordx4 v86, s[72:75], s85 offen offset: 0 lds // (op 4480) Load value direct to lds








v_add_co_u32 v38, vcc, 256, v38 // (op 4764) least significant half
buffer_load_ubyte v36, v33, s[80:83], 0 offen offset:12 // (op 1552) Load value
v_addc_co_u32 v39, vcc, 0, v39, vcc // (op 4764) most significant half

s_barrier  // (op 6410) 
v_add_u32 v40, 8, v40 // (op 5020) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 4475) 
v_mov_b64 v[82:83], s[84:85] // (op 4475) convert


v_mov_b64 v[80:81], v[84:85] // (op 4475) 
s_add_u32 s84, 24576, s54 // (op 4475) 
s_mov_b32 m0, s84 // (op 4475) reg expression
s_mov_b32 s85, 384 // (op 4475) call()
buffer_load_dwordx4 v80, s[68:71], s85 offen offset: 0 lds // (op 4475) Load value direct to lds
v_add_co_u32 v80, s[86:87], v80, v82 // (op 4475) least significant half
v_addc_co_u32 v81, s[86:87], 0, v81, s[86:87] // (op 4475) most significant half

s_add_u32 m0, m0, 4096 // (op 4475) 
buffer_load_dwordx4 v80, s[68:71], s85 offen offset: 0 lds // (op 4475) Load value direct to lds








v_add_co_u32 v34, vcc, 256, v34 // (op 4636) least significant half
buffer_load_ubyte v37, v74, s[76:79], 0 offen offset:12 // (op 1551) Load value
v_addc_co_u32 v35, vcc, 0, v35, vcc // (op 4636) most significant half

s_barrier  // (op 6280) 
v_add_u32 v30, 8, v30 // (op 4892) 



s_waitcnt vmcnt(9)
ds_write_b8 v76, v26 offset:32768 // (op 30) Store local data 
s_barrier  // (op 6260) 



s_waitcnt vmcnt(6)
ds_write_b8 v75, v28 offset:33024 // (op 21) Store local data 
s_barrier  // (op 6246) 



ds_read_b128 v[42:45], v4 // (op 75) Load local data 
ds_read_b128 v[46:49], v4 offset:64 // (op 75) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 6287) 



ds_read_u8 v29, v5 offset:32768 // (op 85) Load local data 



ds_read_b128 v[50:53], v4 offset:4096 // (op 1665) Load local data 
ds_read_b128 v[54:57], v4 offset:4160 // (op 1665) Load local data 



ds_read_u8 v41, v5 offset:32896 // (op 1669) Load local data 
s_barrier  // (op 6227) 



ds_read_b128 v[58:61], v3 offset:8192 // (op 70) Load local data 
ds_read_b128 v[62:65], v3 offset:8256 // (op 70) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 6267) 



ds_read_u8 v31, v0 offset:33024 // (op 80) Load local data 



ds_read_b128 v[66:69], v3 offset:12288 // (op 1916) Load local data 
ds_read_b128 v[70:73], v3 offset:12352 // (op 1916) Load local data 



ds_read_u8 v32, v0 offset:33152 // (op 1919) Load local data 
s_and_b32 s84, s7, 4294967294 // (op 1882) 
s_cmp_lt_u32 s1, s84 // (op 1882) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoop_1882 // (op 1882) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoop_1882 if true)
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_1882:














s_barrier  // (op 6601) 
s_and_b32 s52, s7, 4294967294 // (op 4548) 
v_lshrrev_b32 v26, 3, v1 // (op 4548) 
s_add_u32 s53, s0, s52 // (op 4548) 

v_and_b32 v28, -8, v1 // (op 4548) 
v_sub_u32 v30, v1, v28 // (op 4548) 

s_mul_i32 s52, s16, s8 // (op 4548) 
s_add_u32 s52, s52, s53 // (op 4548) 

v_lshlrev_b32 v28, 4, v30 // (op 4548) 

s_mul_hi_u32 s53, s52, s13 // (op 4548) 
s_sub_u32 s54, s52, s53 // (op 4548) 
s_lshr_b32 s55, s54, 1 // (op 4548) 

s_add_u32 s54, s55, s53 // (op 4548) 


s_lshr_b32 s53, s54, s11 // (op 4548) 

s_and_b32 s54, s11, 2147483648 // (op 4548) 
s_cmp_eq_i32 s54, 0 // (op 4548) 
s_mov_b32 s55, scc // (op 4548) 

s_cmp_eq_i32 s55, 0 // (op 4548) 
s_cselect_b32 s54, s52, s53 // (op 4548) 


s_mul_hi_u32 s53, s54, s12 // (op 4548) 
s_sub_u32 s55, s54, s53 // (op 4548) 
s_lshr_b32 s84, s55, 1 // (op 4548) 

s_add_u32 s55, s84, s53 // (op 4548) 


s_lshr_b32 s53, s55, s10 // (op 4548) 

s_and_b32 s55, s10, 2147483648 // (op 4548) 
s_cmp_eq_i32 s55, 0 // (op 4548) 
s_mov_b32 s84, scc // (op 4548) 

s_cmp_eq_i32 s84, 0 // (op 4548) 
s_cselect_b32 s55, s54, s53 // (op 4548) 


v_lshl_add_u32 v30, s55, 6, v26 // (op 4548) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 4548) 
s_mul_i32 s53, s54, s84 // (op 4548) 


s_sub_u32 s54, s52, s53 // (op 4548) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 4548) 


v_mov_b32 v35, 0 // (op 4548) convert
v_mov_b32 v34, v26 // (op 4548) convert

v_mul_lo_u32 v39, v30, s31 // (op 4548) most significant: low of low * high
v_mul_hi_u32 v26, v30, s30 // (op 4548) most significant: high of low * low
v_mul_lo_u32 v38, v30, s30 // (op 4548) least significant: low of low * low
v_add_u32 v39, v39, v26 // (op 4548) most significant: sum

v_add_co_u32 v38, s[52:53], v38, v34 // (op 4548) least significant half
v_addc_co_u32 v39, s[52:53], v39, v35, s[52:53] // (op 4548) most significant half



s_mul_i32 s53, s30, s35 // (op 4550) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 4550) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 4550) least significant
s_mul_hi_u32 s55, s30, s34 // (op 4550) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4550) most significant: sum
s_add_u32 s53, s53, s55 // (op 4550) most significant: sum


s_mov_b32 s68, s32 // (op 4550) reg expression
s_mov_b32 s69, s33 // (op 4550) reg expression
s_mov_b32 s70, s52 // (op 4550) reg expression
s_mov_b32 s71, 131072 // (op 4550) call()

s_and_b32 s52, s7, 4294967294 // (op 4688) 
v_lshrrev_b32 v26, 3, v1 // (op 4688) 
s_add_u32 s53, s0, s52 // (op 4688) 

v_and_b32 v28, -8, v1 // (op 4688) 
v_sub_u32 v30, v1, v28 // (op 4688) 

s_mul_i32 s52, s16, s8 // (op 4688) 
s_add_u32 s52, s52, s53 // (op 4688) 

v_lshlrev_b32 v28, 4, v30 // (op 4688) 

s_mul_hi_u32 s53, s52, s13 // (op 4688) 
s_sub_u32 s54, s52, s53 // (op 4688) 
s_lshr_b32 s55, s54, 1 // (op 4688) 

s_add_u32 s54, s55, s53 // (op 4688) 


s_lshr_b32 s53, s54, s11 // (op 4688) 

s_and_b32 s54, s11, 2147483648 // (op 4688) 
s_cmp_eq_i32 s54, 0 // (op 4688) 
s_mov_b32 s55, scc // (op 4688) 

s_cmp_eq_i32 s55, 0 // (op 4688) 
s_cselect_b32 s54, s52, s53 // (op 4688) 


s_mul_hi_u32 s53, s54, s12 // (op 4688) 
s_sub_u32 s55, s54, s53 // (op 4688) 
s_lshr_b32 s84, s55, 1 // (op 4688) 

s_add_u32 s55, s84, s53 // (op 4688) 


s_lshr_b32 s53, s55, s10 // (op 4688) 

s_and_b32 s55, s10, 2147483648 // (op 4688) 
s_cmp_eq_i32 s55, 0 // (op 4688) 
s_mov_b32 s84, scc // (op 4688) 

s_cmp_eq_i32 s84, 0 // (op 4688) 
s_cselect_b32 s55, s54, s53 // (op 4688) 


s_mul_i32 s53, s55, s14 // (op 4688) 

s_sub_u32 s55, s54, s53 // (op 4688) 

v_lshl_add_u32 v30, s55, 6, v26 // (op 4688) 


v_mul_lo_u32 v35, v30, s37 // (op 4688) most significant: low of low * high
v_mul_hi_u32 v26, v30, s36 // (op 4688) most significant: high of low * low
v_mul_lo_u32 v34, v30, s36 // (op 4688) least significant: low of low * low
v_add_u32 v35, v35, v26 // (op 4688) most significant: sum


s_ashr_i64 s[84:85], s[2:3], 7 // (op 4688) 
s_mul_i32 s53, s54, s84 // (op 4688) 


s_sub_u32 s54, s52, s53 // (op 4688) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 4688) 


v_mov_b32 v75, 0 // (op 4688) convert
v_mov_b32 v74, v26 // (op 4688) convert

v_add_co_u32 v76, s[52:53], v74, v34 // (op 4688) least significant half
v_addc_co_u32 v77, s[52:53], v75, v35, s[52:53] // (op 4688) most significant half



s_mul_i32 s53, s36, s23 // (op 4690) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 4690) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 4690) least significant
s_mul_hi_u32 s55, s36, s22 // (op 4690) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4690) most significant: sum
s_add_u32 s53, s53, s55 // (op 4690) most significant: sum


s_mov_b32 s72, s38 // (op 4690) reg expression
s_mov_b32 s73, s39 // (op 4690) reg expression
s_mov_b32 s74, s52 // (op 4690) reg expression
s_mov_b32 s75, 131072 // (op 4690) call()

s_and_b32 s52, s7, 4294967294 // (op 4816) 
v_lshrrev_b32 v26, 2, v1 // (op 4816) 
s_add_u32 s53, s0, s52 // (op 4816) 

v_and_b32 v28, -4, v1 // (op 4816) 
v_sub_u32 v30, v1, v28 // (op 4816) 

s_mul_i32 s52, s16, s8 // (op 4816) 
s_add_u32 s52, s52, s53 // (op 4816) 

s_mul_hi_u32 s53, s52, s13 // (op 4816) 
s_sub_u32 s54, s52, s53 // (op 4816) 
s_lshr_b32 s55, s54, 1 // (op 4816) 

s_add_u32 s54, s55, s53 // (op 4816) 


s_lshr_b32 s53, s54, s11 // (op 4816) 

s_and_b32 s54, s11, 2147483648 // (op 4816) 
s_cmp_eq_i32 s54, 0 // (op 4816) 
s_mov_b32 s55, scc // (op 4816) 

s_cmp_eq_i32 s55, 0 // (op 4816) 
s_cselect_b32 s54, s52, s53 // (op 4816) 


s_mul_hi_u32 s53, s54, s12 // (op 4816) 
s_sub_u32 s55, s54, s53 // (op 4816) 
s_lshr_b32 s84, s55, 1 // (op 4816) 

s_add_u32 s55, s84, s53 // (op 4816) 


s_lshr_b32 s53, s55, s10 // (op 4816) 

s_and_b32 s55, s10, 2147483648 // (op 4816) 
s_cmp_eq_i32 s55, 0 // (op 4816) 
s_mov_b32 s84, scc // (op 4816) 

s_cmp_eq_i32 s84, 0 // (op 4816) 
s_cselect_b32 s55, s54, s53 // (op 4816) 


v_lshl_add_u32 v28, s55, 6, v26 // (op 4816) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 4816) 
s_mul_i32 s53, s54, s84 // (op 4816) 


s_sub_u32 s54, s52, s53 // (op 4816) 


v_lshl_add_u32 v26, s54, 2, v30 // (op 4816) 


v_mul_lo_u32 v30, v28, s40 // (op 4816) 
v_add_u32 v30, v30, v26 // (op 4816) 


s_mul_i32 s53, s40, s35 // (op 4818) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 4818) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 4818) least significant
s_mul_hi_u32 s55, s40, s34 // (op 4818) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4818) most significant: sum
s_add_u32 s53, s53, s55 // (op 4818) most significant: sum


s_mov_b32 s76, s42 // (op 4818) reg expression
s_mov_b32 s77, s43 // (op 4818) reg expression
s_mov_b32 s78, s52 // (op 4818) reg expression
s_mov_b32 s79, 131072 // (op 4818) call()

s_and_b32 s52, s7, 4294967294 // (op 4944) 
v_lshrrev_b32 v26, 2, v1 // (op 4944) 
s_add_u32 s53, s0, s52 // (op 4944) 

v_and_b32 v28, -4, v1 // (op 4944) 
v_sub_u32 v40, v1, v28 // (op 4944) 

s_mul_i32 s52, s16, s8 // (op 4944) 
s_add_u32 s52, s52, s53 // (op 4944) 

s_mul_hi_u32 s53, s52, s13 // (op 4944) 
s_sub_u32 s54, s52, s53 // (op 4944) 
s_lshr_b32 s55, s54, 1 // (op 4944) 

s_add_u32 s54, s55, s53 // (op 4944) 


s_lshr_b32 s53, s54, s11 // (op 4944) 

s_and_b32 s54, s11, 2147483648 // (op 4944) 
s_cmp_eq_i32 s54, 0 // (op 4944) 
s_mov_b32 s55, scc // (op 4944) 

s_cmp_eq_i32 s55, 0 // (op 4944) 
s_cselect_b32 s54, s52, s53 // (op 4944) 


s_mul_hi_u32 s53, s54, s12 // (op 4944) 
s_sub_u32 s55, s54, s53 // (op 4944) 
s_lshr_b32 s84, s55, 1 // (op 4944) 

s_add_u32 s55, s84, s53 // (op 4944) 


s_lshr_b32 s53, s55, s10 // (op 4944) 

s_and_b32 s55, s10, 2147483648 // (op 4944) 
s_cmp_eq_i32 s55, 0 // (op 4944) 
s_mov_b32 s84, scc // (op 4944) 

s_cmp_eq_i32 s84, 0 // (op 4944) 
s_cselect_b32 s55, s54, s53 // (op 4944) 


s_mul_i32 s53, s55, s14 // (op 4944) 

s_sub_u32 s55, s54, s53 // (op 4944) 

v_lshl_add_u32 v28, s55, 6, v26 // (op 4944) 


v_mul_lo_u32 v26, v28, s44 // (op 4944) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 4944) 
s_mul_i32 s53, s54, s84 // (op 4944) 


s_sub_u32 s54, s52, s53 // (op 4944) 


v_lshl_add_u32 v28, s54, 2, v40 // (op 4944) 


v_add_u32 v40, v28, v26 // (op 4944) 


s_mul_i32 s53, s44, s23 // (op 4946) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 4946) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 4946) least significant
s_mul_hi_u32 s55, s44, s22 // (op 4946) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4946) most significant: sum
s_add_u32 s53, s53, s55 // (op 4946) most significant: sum


s_mov_b32 s80, s46 // (op 4946) reg expression
s_mov_b32 s81, s47 // (op 4946) reg expression
s_mov_b32 s82, s52 // (op 4946) reg expression
s_mov_b32 s83, 131072 // (op 4946) call()

v_and_b32 v26, -8, v1 // (op 5115) 
v_sub_u32 v28, v1, v26 // (op 5115) 

v_lshlrev_b32 v26, 4, v28 // (op 5115) 

v_lshrrev_b32 v28, 3, v1 // (op 5115) 
v_lshl_add_u32 v33, v28, 7, v26 // (op 5115) 


s_nop 0
v_readfirstlane_b32 s52, v33 // (op 5115) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -8, v1 // (op 5190) 
v_sub_u32 v28, v1, v26 // (op 5190) 

v_lshlrev_b32 v26, 4, v28 // (op 5190) 

v_lshrrev_b32 v28, 3, v1 // (op 5190) 
v_lshl_add_u32 v33, v28, 7, v26 // (op 5190) 


s_nop 0
v_readfirstlane_b32 s53, v33 // (op 5190) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 5265) 
v_sub_u32 v28, v1, v26 // (op 5265) 

v_and_b32 v26, 4294967292, v1 // (op 5265) 
v_add_u32 v33, v26, v28 // (op 5265) 


v_and_b32 v26, -4, v1 // (op 5341) 
v_sub_u32 v28, v1, v26 // (op 5341) 

v_and_b32 v26, 4294967292, v1 // (op 5341) 
v_add_u32 v34, v26, v28 // (op 5341) 


s_and_b32 s54, s7, 4294967294 // (op 1888) 
s_cmp_lt_u32 s54, s7 // (op 1887) 










s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_1887 // (op 1887) Condition: Top (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_1887 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_1887:
s_barrier  // (op 6224) 
s_add_u32 s54, s54, 1 // (op 1889) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 4462) 
v_mov_b64 v[28:29], s[84:85] // (op 4462) convert


v_mov_b64 v[36:37], v[38:39] // (op 4462) 
v_mov_b64 v[74:75], v[36:37] // (op 4462) 
s_add_u32 s55, 8192, s52 // (op 4462) 
s_mov_b32 m0, s55 // (op 4462) reg expression
buffer_load_dwordx4 v74, s[68:71], 0 offen offset: 0 lds // (op 4462) Load value direct to lds
v_add_co_u32 v74, s[84:85], v74, v28 // (op 4462) least significant half
v_addc_co_u32 v75, s[84:85], 0, v75, s[84:85] // (op 4462) most significant half

s_add_u32 m0, m0, 4096 // (op 4462) 
buffer_load_dwordx4 v74, s[68:71], 0 offen offset: 0 lds // (op 4462) Load value direct to lds




v_add_co_u32 v38, vcc, 128, v38 // (op 4538) least significant half
v_addc_co_u32 v39, vcc, 0, v39, vcc // (op 4538) most significant half
s_barrier  // (op 6243) 

s_lshl_b64 s[84:85], s[36:37], 5 // (op 4468) 
v_mov_b64 v[28:29], s[84:85] // (op 4468) convert


v_mov_b64 v[36:37], v[76:77] // (op 4468) 
v_mov_b64 v[74:75], v[36:37] // (op 4468) 
s_mov_b32 m0, s53 // (op 4468) reg expression
buffer_load_dwordx4 v74, s[72:75], 0 offen offset: 0 lds // (op 4468) Load value direct to lds
v_add_co_u32 v74, s[84:85], v74, v28 // (op 4468) least significant half
v_addc_co_u32 v75, s[84:85], 0, v75, s[84:85] // (op 4468) most significant half

s_add_u32 m0, m0, 4096 // (op 4468) 
buffer_load_dwordx4 v74, s[72:75], 0 offen offset: 0 lds // (op 4468) Load value direct to lds



v_add_co_u32 v76, vcc, 128, v76 // (op 4680) least significant half
v_addc_co_u32 v77, vcc, 0, v77, vcc // (op 4680) most significant half


v_mov_b32 v26, v30 // (op 1485) 

buffer_load_ubyte v28, v26, s[76:79], 0 offen offset: 0 // (op 1485) Load value

v_add_u32 v30, 4, v30 // (op 4808) 


s_barrier  // (op 6263) 
v_mov_b32 v26, v40 // (op 1486) 




buffer_load_ubyte v29, v26, s[80:83], 0 offen offset: 0 // (op 1486) Load value

v_add_u32 v40, 4, v40 // (op 4936) 
s_waitcnt vmcnt(1)
ds_write_b8 v33, v28 offset:33024 // (op 1482) Store local data 

s_barrier  // (op 6230) 



s_barrier  // (op 6283) 
ds_read_b128 v[68:71], v3 offset:8192 // (op 1473) Load local data 



ds_read_b128 v[72:75], v3 offset:8256 // (op 1473) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v34, v29 offset:32768 // (op 1483) Store local data 

s_barrier  // (op 6250) 



ds_read_b128 v[60:63], v4 // (op 1475) Load local data 
ds_read_b128 v[64:67], v4 offset:64 // (op 1475) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 6270) 



ds_read_u8 v26, v0 offset:33024 // (op 1477) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 6290) 



ds_read_u8 v28, v5 offset:32768 // (op 1479) Load local data 



ds_read_b128 v[52:55], v4 offset:4096 // (op 1644) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:75], v[60:67], a[0:3], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1480) 
ds_read_b128 v[56:59], v4 offset:4160 // (op 1644) Load local data 



ds_read_u8 v29, v5 offset:32896 // (op 1648) Load local data 



ds_read_b128 v[44:47], v3 offset:12288 // (op 1862) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:75], v[52:59], a[4:7], v26, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1649) 
ds_read_b128 v[48:51], v3 offset:12352 // (op 1862) Load local data 





ds_read_u8 v26, v0 offset:33152 // (op 1865) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[44:51], v[60:67], a[8:11], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1867) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[44:51], v[52:59], a[12:15], v26, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1898) 




s_cmp_lt_u32 s54, s7 // (op 1887) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_1887 // (op 1887) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_1887 if true)
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_1887:









s_barrier  // (op 6605) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 288) 
s_cmp_lg_u32 s18, 0 // (op 314) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Send_Tile_314 // (op 314) Condition: False, jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Send_Tile_314



v_add_u32 v26, 8192, v25 // (op 2324) 
buffer_store_dword a8, v26, s[64:67], 0 offen sc1 // (op 2324) Store value

v_add_u32 v26, 8448, v25 // (op 2324) 
buffer_store_dword a9, v26, s[64:67], 0 offen sc1 // (op 2324) Store value

v_add_u32 v26, 8704, v25 // (op 2324) 
buffer_store_dword a10, v26, s[64:67], 0 offen sc1 // (op 2324) Store value

v_add_u32 v26, 8960, v25 // (op 2324) 
buffer_store_dword a11, v26, s[64:67], 0 offen sc1 // (op 2324) Store value




v_add_u32 v26, 8320, v25 // (op 2326) 
buffer_store_dword a12, v26, s[64:67], 0 offen sc1 // (op 2326) Store value

v_add_u32 v26, 8576, v25 // (op 2326) 
buffer_store_dword a13, v26, s[64:67], 0 offen sc1 // (op 2326) Store value

v_add_u32 v26, 8832, v25 // (op 2326) 
buffer_store_dword a14, v26, s[64:67], 0 offen sc1 // (op 2326) Store value

v_add_u32 v26, 9088, v25 // (op 2326) 
buffer_store_dword a15, v26, s[64:67], 0 offen sc1 // (op 2326) Store value




buffer_store_dword a0, v25, s[64:67], 0 offen sc1 // (op 311) Store value
buffer_store_dword a1, v25, s[64:67], 0 offen offset:256 sc1 // (op 311) Store value
buffer_store_dword a2, v25, s[64:67], 0 offen offset:512 sc1 // (op 311) Store value
buffer_store_dword a3, v25, s[64:67], 0 offen offset:768 sc1 // (op 311) Store value



buffer_store_dword a4, v25, s[64:67], 0 offen offset:128 sc1 // (op 2310) Store value
buffer_store_dword a5, v25, s[64:67], 0 offen offset:384 sc1 // (op 2310) Store value
buffer_store_dword a6, v25, s[64:67], 0 offen offset:640 sc1 // (op 2310) Store value
buffer_store_dword a7, v25, s[64:67], 0 offen offset:896 sc1 // (op 2310) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 315) 
v_cmp_eq_i32 vcc, v1, 0 // (op 319) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_319 // (op 319) Condition: False, jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_319
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
s_branch RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_319 // (op 319) Condition: Done, jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_319
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_319:
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_319:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Send_Tile_314 // (op 314) Condition: Done, jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Send_Tile_314
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Send_Tile_314:
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Send_Tile_314:

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
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Receive_Tile_330 // (op 330) Condition: False, jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Receive_Tile_330
s_mov_b32 s52, 0 // (op 296) call()
s_mul_i32 s53, s52, s8 // (op 295) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 295) 
s_mov_b32 s84, s54 // (op 295) convert

s_sub_u32 s54, s84, s19 // (op 295) 

s_sub_u32 s55, s54, 1 // (op 295) 

s_cmp_lt_u32 s53, s55 // (op 295) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_295 // (op 295) Condition: Top (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_295 if false)
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_ReceiveTileLoop_295:
s_add_u32 s53, s16, 1 // (op 5795) 
v_lshrrev_b32 v20, 5, v1 // (op 5795) 
s_add_u32 s54, s53, s52 // (op 5795) 
s_lshl_b32 s54, s54, 6 // (op 5795) 

v_lshlrev_b32 v21, 2, v20 // (op 5795) 

v_add_lshl_u32 v20, s54, v21, 6 // (op 5795) 


v_and_b32 v21, -32, v1 // (op 5795) 
v_sub_u32 v23, v1, v21 // (op 5795) 

v_add_lshl_u32 v21, v20, v23, 2 // (op 5795) 


s_lshl_b32 s53, s15, 14 // (op 5797) 
s_mov_b32 s64, s28 // (op 5797) reg expression
s_mov_b32 s65, s29 // (op 5797) reg expression
s_mov_b32 s66, s53 // (op 5797) reg expression
s_mov_b32 s67, 131072 // (op 5797) call()

RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_334:
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
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_334 // (op 334) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_334 if true)
s_barrier  // (op 338) 
v_cmp_eq_i32 vcc, v1, 0 // (op 337) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_337 // (op 337) Condition: False, jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_337
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
s_branch RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_337 // (op 337) Condition: Done, jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_337
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_337:
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_337:




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




buffer_load_dword v28, v21, s[64:67], 0 offen offset:128 // (op 2349) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:384 // (op 2349) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:640 // (op 2349) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:896 // (op 2349) Load value
v_accvgpr_read v32, a4 // (op 2350) reg expression
v_accvgpr_read v33, a5 // (op 2350) reg expression
v_accvgpr_read v34, a6 // (op 2350) reg expression
v_accvgpr_read v35, a7 // (op 2350) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 2351) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 2351) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 2351) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 2351) 

v_accvgpr_write a4, v32 // (op 2352) reg expression
v_accvgpr_write a5, v33 // (op 2352) reg expression
v_accvgpr_write a6, v34 // (op 2352) reg expression
v_accvgpr_write a7, v35 // (op 2352) reg expression




v_mov_b32 v20, 8192 // (op 2374) 
v_add_i32 v23, v21, v20 // (op 2374) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 2374) Load value

v_mov_b32 v20, 8448 // (op 2374) 
v_add_i32 v23, v21, v20 // (op 2374) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 2374) Load value

v_mov_b32 v20, 8704 // (op 2374) 
v_add_i32 v23, v21, v20 // (op 2374) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 2374) Load value

v_mov_b32 v20, 8960 // (op 2374) 
v_add_i32 v23, v21, v20 // (op 2374) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 2374) Load value

v_accvgpr_read v32, a8 // (op 2371) reg expression
v_accvgpr_read v33, a9 // (op 2371) reg expression
v_accvgpr_read v34, a10 // (op 2371) reg expression
v_accvgpr_read v35, a11 // (op 2371) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 2372) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 2372) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 2372) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 2372) 

v_accvgpr_write a8, v32 // (op 2373) reg expression
v_accvgpr_write a9, v33 // (op 2373) reg expression
v_accvgpr_write a10, v34 // (op 2373) reg expression
v_accvgpr_write a11, v35 // (op 2373) reg expression




v_mov_b32 v20, 8320 // (op 2379) 
v_add_i32 v23, v21, v20 // (op 2379) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 2379) Load value

v_mov_b32 v20, 8576 // (op 2379) 
v_add_i32 v23, v21, v20 // (op 2379) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 2379) Load value

v_mov_b32 v20, 8832 // (op 2379) 
v_add_i32 v23, v21, v20 // (op 2379) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 2379) Load value

v_mov_b32 v20, 9088 // (op 2379) 
v_add_i32 v23, v21, v20 // (op 2379) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 2379) Load value

v_accvgpr_read v32, a12 // (op 2376) reg expression
v_accvgpr_read v33, a13 // (op 2376) reg expression
v_accvgpr_read v34, a14 // (op 2376) reg expression
v_accvgpr_read v35, a15 // (op 2376) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 2377) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 2377) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 2377) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 2377) 

v_accvgpr_write a12, v32 // (op 2378) reg expression
v_accvgpr_write a13, v33 // (op 2378) reg expression
v_accvgpr_write a14, v34 // (op 2378) reg expression
v_accvgpr_write a15, v35 // (op 2378) reg expression

s_add_u32 s52, s52, 1 // (op 297) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 295) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 295) 
s_mov_b32 s84, s54 // (op 295) convert

s_sub_u32 s54, s84, s19 // (op 295) 

s_sub_u32 s55, s54, 1 // (op 295) 

s_cmp_lt_u32 s53, s55 // (op 295) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_ReceiveTileLoop_295 // (op 295) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_ReceiveTileLoop_295 if true)
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_295:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Receive_Tile_330 // (op 330) Condition: Done, jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Receive_Tile_330
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Receive_Tile_330:
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Receive_Tile_330:

s_cmp_eq_i32 s18, 0 // (op 373) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_373 // (op 373) Condition: False, jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_373
v_cmp_eq_f32 vcc, 0.00000, v18 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_133
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


v_accvgpr_read v28, a4 // (op 2409) 
v_accvgpr_read v29, a5 // (op 2409) 
v_accvgpr_read v30, a6 // (op 2409) 
v_accvgpr_read v31, a7 // (op 2409) 
v_mul_f32 v32, v19, v28 // (op 2409) 
v_mul_f32 v33, v19, v29 // (op 2409) 
v_mul_f32 v34, v19, v30 // (op 2409) 
v_mul_f32 v35, v19, v31 // (op 2409) 



s_lshl_b32 s52, s24, 7 // (op 2411) 
v_add_u32 v20, v27, s52 // (op 2411) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 2411) Store value


v_accvgpr_read v28, a8 // (op 2433) 
v_accvgpr_read v29, a9 // (op 2433) 
v_accvgpr_read v30, a10 // (op 2433) 
v_accvgpr_read v31, a11 // (op 2433) 
v_mul_f32 v32, v19, v28 // (op 2433) 
v_mul_f32 v33, v19, v29 // (op 2433) 
v_mul_f32 v34, v19, v30 // (op 2433) 
v_mul_f32 v35, v19, v31 // (op 2433) 



v_add_u32 v20, 128, v27 // (op 2436) 

buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 2436) Store value


v_accvgpr_read v28, a12 // (op 2442) 
v_accvgpr_read v29, a13 // (op 2442) 
v_accvgpr_read v30, a14 // (op 2442) 
v_accvgpr_read v31, a15 // (op 2442) 
v_mul_f32 v32, v19, v28 // (op 2442) 
v_mul_f32 v33, v19, v29 // (op 2442) 
v_mul_f32 v34, v19, v30 // (op 2442) 
v_mul_f32 v35, v19, v31 // (op 2442) 



s_lshl_b32 s52, s24, 7 // (op 2438) 
s_add_u32 s52, s52, 128 // (op 2438) 
v_add_u32 v20, v27, s52 // (op 2438) 


buffer_store_dwordx4 v[32:35], v20, s[60:63], 0 offen // (op 2438) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_133
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_133:


v_add_u32 v20, 128, v22 // (op 2522) 

buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 2522) Load value

v_accvgpr_read v32, a8 // (op 2514) 
v_accvgpr_read v33, a9 // (op 2514) 
v_accvgpr_read v34, a10 // (op 2514) 
v_accvgpr_read v35, a11 // (op 2514) 
v_mul_f32 v36, v19, v32 // (op 2514) 
v_mul_f32 v37, v19, v33 // (op 2514) 
v_mul_f32 v38, v19, v34 // (op 2514) 
v_mul_f32 v39, v19, v35 // (op 2514) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 2521) 
v_mul_f32 v33, v18, v29 // (op 2521) 
v_mul_f32 v34, v18, v30 // (op 2521) 
v_mul_f32 v35, v18, v31 // (op 2521) 

v_add_f32 v28, v32, v36 // (op 2515) 
v_add_f32 v29, v33, v37 // (op 2515) 
v_add_f32 v30, v34, v38 // (op 2515) 
v_add_f32 v31, v35, v39 // (op 2515) 




v_add_u32 v20, 128, v27 // (op 2517) 

buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 2517) Store value




s_lshl_b32 s52, s4, 7 // (op 2526) 
s_add_u32 s52, s52, 128 // (op 2526) 
v_add_u32 v20, v22, s52 // (op 2526) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 2526) Load value

v_accvgpr_read v32, a12 // (op 2528) 
v_accvgpr_read v33, a13 // (op 2528) 
v_accvgpr_read v34, a14 // (op 2528) 
v_accvgpr_read v35, a15 // (op 2528) 
v_mul_f32 v36, v19, v32 // (op 2528) 
v_mul_f32 v37, v19, v33 // (op 2528) 
v_mul_f32 v38, v19, v34 // (op 2528) 
v_mul_f32 v39, v19, v35 // (op 2528) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 2524) 
v_mul_f32 v33, v18, v29 // (op 2524) 
v_mul_f32 v34, v18, v30 // (op 2524) 
v_mul_f32 v35, v18, v31 // (op 2524) 

v_add_f32 v28, v32, v36 // (op 2525) 
v_add_f32 v29, v33, v37 // (op 2525) 
v_add_f32 v30, v34, v38 // (op 2525) 
v_add_f32 v31, v35, v39 // (op 2525) 




s_lshl_b32 s52, s24, 7 // (op 2519) 
s_add_u32 s52, s52, 128 // (op 2519) 
v_add_u32 v20, v27, s52 // (op 2519) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 2519) Store value




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




s_lshl_b32 s52, s4, 7 // (op 2484) 
v_add_u32 v20, v22, s52 // (op 2484) 


buffer_load_dwordx4 v[28:31], v20, s[56:59], 0 offen // (op 2484) Load value

v_accvgpr_read v32, a4 // (op 2481) 
v_accvgpr_read v33, a5 // (op 2481) 
v_accvgpr_read v34, a6 // (op 2481) 
v_accvgpr_read v35, a7 // (op 2481) 
v_mul_f32 v36, v19, v32 // (op 2481) 
v_mul_f32 v37, v19, v33 // (op 2481) 
v_mul_f32 v38, v19, v34 // (op 2481) 
v_mul_f32 v39, v19, v35 // (op 2481) 

s_waitcnt vmcnt(0)
v_mul_f32 v32, v18, v28 // (op 2485) 
v_mul_f32 v33, v18, v29 // (op 2485) 
v_mul_f32 v34, v18, v30 // (op 2485) 
v_mul_f32 v35, v18, v31 // (op 2485) 

v_add_f32 v28, v32, v36 // (op 2482) 
v_add_f32 v29, v33, v37 // (op 2482) 
v_add_f32 v30, v34, v38 // (op 2482) 
v_add_f32 v31, v35, v39 // (op 2482) 




s_lshl_b32 s52, s24, 7 // (op 2483) 
v_add_u32 v20, v27, s52 // (op 2483) 


buffer_store_dwordx4 v[28:31], v20, s[60:63], 0 offen // (op 2483) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_133:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_373 // (op 373) Condition: Done, jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_373
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_373:
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_373:


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
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_SKStreamTileLoop_286 // (op 286) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_SKStreamTileLoop_286 if true)
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_286:

















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
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_441 // (op 441) Condition: Top (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_441 if false)
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_DPStreamTileLoop_441:
v_lshrrev_b32 v0, 7, v1 // (op 5085) 
s_mul_i32 s15, s16, s50 // (op 5085) 
s_add_u32 s15, s15, s0 // (op 5085) 
v_and_b32 v3, 63, v1 // (op 5085) 
s_add_u32 s6, s15, s9 // (op 5085) 

v_lshrrev_b32 v4, 4, v3 // (op 5085) 
s_mul_hi_u32 s15, s6, s13 // (op 5085) 
v_lshlrev_b32 v5, 2, v4 // (op 5085) 

s_sub_u32 s7, s6, s15 // (op 5085) 
v_lshl_add_u32 v4, v0, 4, v5 // (op 5085) 


s_lshr_b32 s8, s7, 1 // (op 5085) 

v_lshrrev_b32 v0, 2, v3 // (op 5085) 

s_add_u32 s7, s8, s15 // (op 5085) 


v_and_b32 v3, 3, v0 // (op 5085) 

s_lshr_b32 s8, s7, s11 // (op 5085) 

v_and_b32 v0, 3, v1 // (op 5085) 
s_and_b32 s7, s11, 2147483648 // (op 5085) 
v_lshl_add_u32 v5, v3, 2, v0 // (op 5085) 


s_cmp_eq_i32 s7, 0 // (op 5085) 
s_mov_b32 s15, scc // (op 5085) 

v_lshrrev_b32 v0, 6, v1 // (op 5085) 
s_cmp_eq_i32 s15, 0 // (op 5085) 
s_cselect_b32 s7, s6, s8 // (op 5085) 



v_and_b32 v3, 1, v0 // (op 5085) 

s_mul_hi_u32 s6, s7, s12 // (op 5085) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 5085) 


s_sub_u32 s8, s7, s6 // (op 5085) 
s_lshr_b32 s15, s8, 1 // (op 5085) 

s_add_u32 s8, s15, s6 // (op 5085) 


s_lshr_b32 s6, s8, s10 // (op 5085) 

s_and_b32 s8, s10, 2147483648 // (op 5085) 
s_cmp_eq_i32 s8, 0 // (op 5085) 
s_mov_b32 s15, scc // (op 5085) 

s_cmp_eq_i32 s15, 0 // (op 5085) 
s_cselect_b32 s8, s7, s6 // (op 5085) 


s_mul_i32 s6, s8, s14 // (op 5085) 
v_lshl_add_u32 v3, s8, 6, v4 // (op 5085) 


s_sub_u32 s8, s7, s6 // (op 5085) 


v_lshl_add_u32 v4, s8, 6, v0 // (op 5085) 


v_mul_lo_u32 v0, v4, s4 // (op 5085) 

v_add_lshl_u32 v4, v3, v0, 2 // (op 5085) 


s_mul_i32 s19, s4, s23 // (op 5087) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 5087) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 5087) least significant
s_mul_hi_u32 s6, s4, s22 // (op 5087) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 5087) most significant: sum
s_add_u32 s19, s19, s6 // (op 5087) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 5087) 

s_mov_b32 s56, s20 // (op 5087) reg expression
s_mov_b32 s57, s21 // (op 5087) reg expression
s_mov_b32 s58, s28 // (op 5087) reg expression
s_mov_b32 s59, 131072 // (op 5087) call()

v_lshrrev_b32 v0, 7, v1 // (op 5439) 
s_mul_i32 s15, s16, s50 // (op 5439) 
s_add_u32 s15, s15, s0 // (op 5439) 
v_and_b32 v3, 63, v1 // (op 5439) 
s_add_u32 s6, s15, s9 // (op 5439) 

v_lshrrev_b32 v5, 4, v3 // (op 5439) 
s_mul_hi_u32 s15, s6, s13 // (op 5439) 
v_lshlrev_b32 v14, 2, v5 // (op 5439) 

s_sub_u32 s7, s6, s15 // (op 5439) 
v_lshl_add_u32 v5, v0, 4, v14 // (op 5439) 


s_lshr_b32 s8, s7, 1 // (op 5439) 

v_lshrrev_b32 v0, 2, v3 // (op 5439) 

s_add_u32 s7, s8, s15 // (op 5439) 


v_and_b32 v3, 3, v0 // (op 5439) 

s_lshr_b32 s8, s7, s11 // (op 5439) 

v_and_b32 v0, 3, v1 // (op 5439) 
s_and_b32 s7, s11, 2147483648 // (op 5439) 
v_lshl_add_u32 v14, v3, 2, v0 // (op 5439) 


s_cmp_eq_i32 s7, 0 // (op 5439) 
s_mov_b32 s15, scc // (op 5439) 

v_lshrrev_b32 v0, 6, v1 // (op 5439) 
s_cmp_eq_i32 s15, 0 // (op 5439) 
s_cselect_b32 s7, s6, s8 // (op 5439) 



v_and_b32 v3, 1, v0 // (op 5439) 

s_mul_hi_u32 s6, s7, s12 // (op 5439) 
v_lshl_add_u32 v0, v3, 4, v14 // (op 5439) 


s_sub_u32 s8, s7, s6 // (op 5439) 
s_lshr_b32 s15, s8, 1 // (op 5439) 

s_add_u32 s8, s15, s6 // (op 5439) 


s_lshr_b32 s6, s8, s10 // (op 5439) 

s_and_b32 s8, s10, 2147483648 // (op 5439) 
s_cmp_eq_i32 s8, 0 // (op 5439) 
s_mov_b32 s15, scc // (op 5439) 

s_cmp_eq_i32 s15, 0 // (op 5439) 
s_cselect_b32 s8, s7, s6 // (op 5439) 


s_mul_i32 s6, s8, s14 // (op 5439) 
v_lshl_add_u32 v3, s8, 6, v5 // (op 5439) 


s_sub_u32 s8, s7, s6 // (op 5439) 


v_lshl_add_u32 v5, s8, 6, v0 // (op 5439) 


v_mul_lo_u32 v0, v5, s24 // (op 5439) 

v_add_lshl_u32 v5, v3, v0, 2 // (op 5439) 


s_mul_i32 s19, s24, s23 // (op 5441) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 5441) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 5441) least significant
s_mul_hi_u32 s6, s24, s22 // (op 5441) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 5441) most significant: sum
s_add_u32 s19, s19, s6 // (op 5441) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 5441) 

s_mov_b32 s60, s26 // (op 5441) reg expression
s_mov_b32 s61, s27 // (op 5441) reg expression
s_mov_b32 s62, s28 // (op 5441) reg expression
s_mov_b32 s63, 131072 // (op 5441) call()

v_lshrrev_b32 v0, 7, v1 // (op 5518) 
v_and_b32 v3, 15, v1 // (op 5518) 
v_lshl_add_u32 v14, v0, 4, v3 // (op 5518) 


v_and_b32 v0, 48, v1 // (op 5518) 
v_lshl_add_u32 v3, v14, 7, v0 // (op 5518) 


v_lshrrev_b32 v0, 6, v1 // (op 5593) 
v_and_b32 v14, 1, v0 // (op 5593) 

v_and_b32 v0, 15, v1 // (op 5593) 
v_lshl_add_u32 v15, v14, 4, v0 // (op 5593) 


v_and_b32 v0, 48, v1 // (op 5593) 
v_lshl_add_u32 v14, v15, 7, v0 // (op 5593) 


v_lshrrev_b32 v0, 7, v1 // (op 5669) 
v_and_b32 v15, 15, v1 // (op 5669) 
v_lshl_add_u32 v16, v0, 4, v15 // (op 5669) 


v_and_b32 v0, 63, v1 // (op 5669) 
v_lshrrev_b32 v15, 4, v0 // (op 5669) 

v_lshl_add_u32 v0, v16, 2, v15 // (op 5669) 


v_lshrrev_b32 v15, 6, v1 // (op 5745) 
v_and_b32 v16, 1, v15 // (op 5745) 

v_and_b32 v15, 15, v1 // (op 5745) 
v_lshl_add_u32 v17, v16, 4, v15 // (op 5745) 


v_and_b32 v15, 63, v1 // (op 5745) 
v_lshrrev_b32 v16, 4, v15 // (op 5745) 

v_lshl_add_u32 v15, v17, 2, v16 // (op 5745) 


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
v_lshrrev_b32 v16, 3, v1 // (op 4577) 
s_mul_i32 s15, s16, s50 // (op 4577) 
s_add_u32 s15, s15, s0 // (op 4577) 
v_and_b32 v17, -8, v1 // (op 4577) 
s_add_u32 s6, s15, s9 // (op 4577) 

v_sub_u32 v20, v1, v17 // (op 4577) 

s_mul_hi_u32 s15, s6, s13 // (op 4577) 
v_lshlrev_b32 v17, 4, v20 // (op 4577) 

s_sub_u32 s7, s6, s15 // (op 4577) 
s_lshr_b32 s8, s7, 1 // (op 4577) 

s_add_u32 s7, s8, s15 // (op 4577) 


s_lshr_b32 s8, s7, s11 // (op 4577) 

s_and_b32 s7, s11, 2147483648 // (op 4577) 
s_cmp_eq_i32 s7, 0 // (op 4577) 
s_mov_b32 s15, scc // (op 4577) 

s_cmp_eq_i32 s15, 0 // (op 4577) 
s_cselect_b32 s7, s6, s8 // (op 4577) 


s_mul_hi_u32 s8, s7, s12 // (op 4577) 
s_sub_u32 s15, s7, s8 // (op 4577) 
s_lshr_b32 s18, s15, 1 // (op 4577) 

s_add_u32 s15, s18, s8 // (op 4577) 


s_lshr_b32 s8, s15, s10 // (op 4577) 

s_and_b32 s15, s10, 2147483648 // (op 4577) 
s_cmp_eq_i32 s15, 0 // (op 4577) 
s_mov_b32 s18, scc // (op 4577) 

s_cmp_eq_i32 s18, 0 // (op 4577) 
s_cselect_b32 s15, s7, s8 // (op 4577) 


v_lshl_add_u32 v20, s15, 6, v16 // (op 4577) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4577) 
s_mul_i32 s8, s7, s18 // (op 4577) 


s_sub_u32 s7, s6, s8 // (op 4577) 


v_lshl_add_u32 v16, s7, 7, v17 // (op 4577) 


v_mov_b32 v23, 0 // (op 4577) convert
v_mov_b32 v22, v16 // (op 4577) convert

v_mul_lo_u32 v17, v20, s31 // (op 4577) most significant: low of low * high
v_mul_hi_u32 v21, v20, s30 // (op 4577) most significant: high of low * low
v_mul_lo_u32 v16, v20, s30 // (op 4577) least significant: low of low * low
v_add_u32 v17, v17, v21 // (op 4577) most significant: sum

v_add_co_u32 v16, s[18:19], v16, v22 // (op 4577) least significant half
v_addc_co_u32 v17, s[18:19], v17, v23, s[18:19] // (op 4577) most significant half



s_mul_i32 s19, s30, s35 // (op 4579) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 4579) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 4579) least significant
s_mul_hi_u32 s6, s30, s34 // (op 4579) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4579) most significant: sum
s_add_u32 s19, s19, s6 // (op 4579) most significant: sum


s_mov_b32 s68, s32 // (op 4579) reg expression
s_mov_b32 s69, s33 // (op 4579) reg expression
s_mov_b32 s70, s18 // (op 4579) reg expression
s_mov_b32 s71, 131072 // (op 4579) call()

v_lshrrev_b32 v20, 3, v1 // (op 4596) 
s_mul_i32 s15, s16, s50 // (op 4596) 
s_add_u32 s15, s15, s0 // (op 4596) 
v_and_b32 v21, -8, v1 // (op 4596) 
s_add_u32 s6, s15, s9 // (op 4596) 

v_sub_u32 v22, v1, v21 // (op 4596) 

s_mul_hi_u32 s15, s6, s13 // (op 4596) 
v_lshlrev_b32 v21, 4, v22 // (op 4596) 

s_sub_u32 s7, s6, s15 // (op 4596) 
s_lshr_b32 s8, s7, 1 // (op 4596) 

s_add_u32 s7, s8, s15 // (op 4596) 


s_lshr_b32 s8, s7, s11 // (op 4596) 

s_and_b32 s7, s11, 2147483648 // (op 4596) 
s_cmp_eq_i32 s7, 0 // (op 4596) 
s_mov_b32 s15, scc // (op 4596) 

s_cmp_eq_i32 s15, 0 // (op 4596) 
s_cselect_b32 s7, s6, s8 // (op 4596) 


s_mul_hi_u32 s8, s7, s12 // (op 4596) 
s_sub_u32 s15, s7, s8 // (op 4596) 
s_lshr_b32 s18, s15, 1 // (op 4596) 

s_add_u32 s15, s18, s8 // (op 4596) 


s_lshr_b32 s8, s15, s10 // (op 4596) 

s_and_b32 s15, s10, 2147483648 // (op 4596) 
s_cmp_eq_i32 s15, 0 // (op 4596) 
s_mov_b32 s18, scc // (op 4596) 

s_cmp_eq_i32 s18, 0 // (op 4596) 
s_cselect_b32 s15, s7, s8 // (op 4596) 


v_lshl_add_u32 v22, s15, 6, v20 // (op 4596) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4596) 
s_mul_i32 s8, s7, s18 // (op 4596) 


s_sub_u32 s7, s6, s8 // (op 4596) 


v_lshl_add_u32 v20, s7, 7, v21 // (op 4596) 


v_mov_b32 v25, 0 // (op 4596) convert
v_mov_b32 v24, v20 // (op 4596) convert

v_mul_lo_u32 v21, v22, s31 // (op 4596) most significant: low of low * high
v_mul_hi_u32 v23, v22, s30 // (op 4596) most significant: high of low * low
v_mul_lo_u32 v20, v22, s30 // (op 4596) least significant: low of low * low
v_add_u32 v21, v21, v23 // (op 4596) most significant: sum

v_add_co_u32 v20, s[18:19], v20, v24 // (op 4596) least significant half
v_addc_co_u32 v21, s[18:19], v21, v25, s[18:19] // (op 4596) most significant half



s_mul_i32 s19, s30, s35 // (op 4598) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 4598) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 4598) least significant
s_mul_hi_u32 s6, s30, s34 // (op 4598) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4598) most significant: sum
s_add_u32 s19, s19, s6 // (op 4598) most significant: sum


s_mov_b32 s68, s32 // (op 4598) reg expression
s_mov_b32 s69, s33 // (op 4598) reg expression
s_mov_b32 s70, s18 // (op 4598) reg expression
s_mov_b32 s71, 131072 // (op 4598) call()

v_lshrrev_b32 v22, 3, v1 // (op 4711) 
s_mul_i32 s15, s16, s50 // (op 4711) 
s_add_u32 s15, s15, s0 // (op 4711) 
v_and_b32 v23, -8, v1 // (op 4711) 
s_add_u32 s6, s15, s9 // (op 4711) 

v_sub_u32 v24, v1, v23 // (op 4711) 

s_mul_hi_u32 s15, s6, s13 // (op 4711) 
v_lshlrev_b32 v23, 4, v24 // (op 4711) 

s_sub_u32 s7, s6, s15 // (op 4711) 
s_lshr_b32 s8, s7, 1 // (op 4711) 

s_add_u32 s7, s8, s15 // (op 4711) 


s_lshr_b32 s8, s7, s11 // (op 4711) 

s_and_b32 s7, s11, 2147483648 // (op 4711) 
s_cmp_eq_i32 s7, 0 // (op 4711) 
s_mov_b32 s15, scc // (op 4711) 

s_cmp_eq_i32 s15, 0 // (op 4711) 
s_cselect_b32 s7, s6, s8 // (op 4711) 


s_mul_hi_u32 s8, s7, s12 // (op 4711) 
s_sub_u32 s15, s7, s8 // (op 4711) 
s_lshr_b32 s18, s15, 1 // (op 4711) 

s_add_u32 s15, s18, s8 // (op 4711) 


s_lshr_b32 s8, s15, s10 // (op 4711) 

s_and_b32 s15, s10, 2147483648 // (op 4711) 
s_cmp_eq_i32 s15, 0 // (op 4711) 
s_mov_b32 s18, scc // (op 4711) 

s_cmp_eq_i32 s18, 0 // (op 4711) 
s_cselect_b32 s15, s7, s8 // (op 4711) 


s_mul_i32 s8, s15, s14 // (op 4711) 

s_sub_u32 s15, s7, s8 // (op 4711) 

v_lshl_add_u32 v24, s15, 6, v22 // (op 4711) 


v_mul_lo_u32 v27, v24, s37 // (op 4711) most significant: low of low * high
v_mul_hi_u32 v22, v24, s36 // (op 4711) most significant: high of low * low
v_mul_lo_u32 v26, v24, s36 // (op 4711) least significant: low of low * low
v_add_u32 v27, v27, v22 // (op 4711) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4711) 
s_mul_i32 s8, s7, s18 // (op 4711) 


s_sub_u32 s7, s6, s8 // (op 4711) 


v_lshl_add_u32 v22, s7, 7, v23 // (op 4711) 


v_mov_b32 v25, 0 // (op 4711) convert
v_mov_b32 v24, v22 // (op 4711) convert

v_add_co_u32 v22, s[18:19], v24, v26 // (op 4711) least significant half
v_addc_co_u32 v23, s[18:19], v25, v27, s[18:19] // (op 4711) most significant half



s_mul_i32 s19, s36, s23 // (op 4713) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 4713) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 4713) least significant
s_mul_hi_u32 s6, s36, s22 // (op 4713) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4713) most significant: sum
s_add_u32 s19, s19, s6 // (op 4713) most significant: sum


s_mov_b32 s72, s38 // (op 4713) reg expression
s_mov_b32 s73, s39 // (op 4713) reg expression
s_mov_b32 s74, s18 // (op 4713) reg expression
s_mov_b32 s75, 131072 // (op 4713) call()

v_lshrrev_b32 v24, 3, v1 // (op 4730) 
s_mul_i32 s15, s16, s50 // (op 4730) 
s_add_u32 s15, s15, s0 // (op 4730) 
v_and_b32 v25, -8, v1 // (op 4730) 
s_add_u32 s6, s15, s9 // (op 4730) 

v_sub_u32 v26, v1, v25 // (op 4730) 

s_mul_hi_u32 s15, s6, s13 // (op 4730) 
v_lshlrev_b32 v25, 4, v26 // (op 4730) 

s_sub_u32 s7, s6, s15 // (op 4730) 
s_lshr_b32 s8, s7, 1 // (op 4730) 

s_add_u32 s7, s8, s15 // (op 4730) 


s_lshr_b32 s8, s7, s11 // (op 4730) 

s_and_b32 s7, s11, 2147483648 // (op 4730) 
s_cmp_eq_i32 s7, 0 // (op 4730) 
s_mov_b32 s15, scc // (op 4730) 

s_cmp_eq_i32 s15, 0 // (op 4730) 
s_cselect_b32 s7, s6, s8 // (op 4730) 


s_mul_hi_u32 s8, s7, s12 // (op 4730) 
s_sub_u32 s15, s7, s8 // (op 4730) 
s_lshr_b32 s18, s15, 1 // (op 4730) 

s_add_u32 s15, s18, s8 // (op 4730) 


s_lshr_b32 s8, s15, s10 // (op 4730) 

s_and_b32 s15, s10, 2147483648 // (op 4730) 
s_cmp_eq_i32 s15, 0 // (op 4730) 
s_mov_b32 s18, scc // (op 4730) 

s_cmp_eq_i32 s18, 0 // (op 4730) 
s_cselect_b32 s15, s7, s8 // (op 4730) 


s_mul_i32 s8, s15, s14 // (op 4730) 

s_sub_u32 s15, s7, s8 // (op 4730) 

v_lshl_add_u32 v26, s15, 6, v24 // (op 4730) 


v_mul_lo_u32 v29, v26, s37 // (op 4730) most significant: low of low * high
v_mul_hi_u32 v24, v26, s36 // (op 4730) most significant: high of low * low
v_mul_lo_u32 v28, v26, s36 // (op 4730) least significant: low of low * low
v_add_u32 v29, v29, v24 // (op 4730) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4730) 
s_mul_i32 s8, s7, s18 // (op 4730) 


s_sub_u32 s7, s6, s8 // (op 4730) 


v_lshl_add_u32 v24, s7, 7, v25 // (op 4730) 


v_mov_b32 v27, 0 // (op 4730) convert
v_mov_b32 v26, v24 // (op 4730) convert

v_add_co_u32 v24, s[18:19], v26, v28 // (op 4730) least significant half
v_addc_co_u32 v25, s[18:19], v27, v29, s[18:19] // (op 4730) most significant half



s_mul_i32 s19, s36, s23 // (op 4732) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 4732) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 4732) least significant
s_mul_hi_u32 s6, s36, s22 // (op 4732) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4732) most significant: sum
s_add_u32 s19, s19, s6 // (op 4732) most significant: sum


s_mov_b32 s72, s38 // (op 4732) reg expression
s_mov_b32 s73, s39 // (op 4732) reg expression
s_mov_b32 s74, s18 // (op 4732) reg expression
s_mov_b32 s75, 131072 // (op 4732) call()

v_lshrrev_b32 v26, 2, v1 // (op 4839) 
s_mul_i32 s15, s16, s50 // (op 4839) 
s_add_u32 s15, s15, s0 // (op 4839) 
v_and_b32 v27, -4, v1 // (op 4839) 
s_add_u32 s6, s15, s9 // (op 4839) 

v_sub_u32 v28, v1, v27 // (op 4839) 

s_mul_hi_u32 s15, s6, s13 // (op 4839) 
s_sub_u32 s7, s6, s15 // (op 4839) 
s_lshr_b32 s8, s7, 1 // (op 4839) 

s_add_u32 s7, s8, s15 // (op 4839) 


s_lshr_b32 s8, s7, s11 // (op 4839) 

s_and_b32 s7, s11, 2147483648 // (op 4839) 
s_cmp_eq_i32 s7, 0 // (op 4839) 
s_mov_b32 s15, scc // (op 4839) 

s_cmp_eq_i32 s15, 0 // (op 4839) 
s_cselect_b32 s7, s6, s8 // (op 4839) 


s_mul_hi_u32 s8, s7, s12 // (op 4839) 
s_sub_u32 s15, s7, s8 // (op 4839) 
s_lshr_b32 s18, s15, 1 // (op 4839) 

s_add_u32 s15, s18, s8 // (op 4839) 


s_lshr_b32 s8, s15, s10 // (op 4839) 

s_and_b32 s15, s10, 2147483648 // (op 4839) 
s_cmp_eq_i32 s15, 0 // (op 4839) 
s_mov_b32 s18, scc // (op 4839) 

s_cmp_eq_i32 s18, 0 // (op 4839) 
s_cselect_b32 s15, s7, s8 // (op 4839) 


v_lshl_add_u32 v27, s15, 6, v26 // (op 4839) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4839) 
s_mul_i32 s8, s7, s18 // (op 4839) 


s_sub_u32 s7, s6, s8 // (op 4839) 


v_lshl_add_u32 v26, s7, 2, v28 // (op 4839) 


v_mul_lo_u32 v28, v27, s40 // (op 4839) 
v_add_u32 v28, v28, v26 // (op 4839) 


s_mul_i32 s19, s40, s35 // (op 4841) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 4841) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 4841) least significant
s_mul_hi_u32 s6, s40, s34 // (op 4841) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4841) most significant: sum
s_add_u32 s19, s19, s6 // (op 4841) most significant: sum


s_mov_b32 s76, s42 // (op 4841) reg expression
s_mov_b32 s77, s43 // (op 4841) reg expression
s_mov_b32 s78, s18 // (op 4841) reg expression
s_mov_b32 s79, 131072 // (op 4841) call()

v_lshrrev_b32 v26, 2, v1 // (op 4858) 
s_mul_i32 s15, s16, s50 // (op 4858) 
s_add_u32 s15, s15, s0 // (op 4858) 
v_and_b32 v27, -4, v1 // (op 4858) 
s_add_u32 s6, s15, s9 // (op 4858) 

v_sub_u32 v29, v1, v27 // (op 4858) 

s_mul_hi_u32 s15, s6, s13 // (op 4858) 
s_sub_u32 s7, s6, s15 // (op 4858) 
s_lshr_b32 s8, s7, 1 // (op 4858) 

s_add_u32 s7, s8, s15 // (op 4858) 


s_lshr_b32 s8, s7, s11 // (op 4858) 

s_and_b32 s7, s11, 2147483648 // (op 4858) 
s_cmp_eq_i32 s7, 0 // (op 4858) 
s_mov_b32 s15, scc // (op 4858) 

s_cmp_eq_i32 s15, 0 // (op 4858) 
s_cselect_b32 s7, s6, s8 // (op 4858) 


s_mul_hi_u32 s8, s7, s12 // (op 4858) 
s_sub_u32 s15, s7, s8 // (op 4858) 
s_lshr_b32 s18, s15, 1 // (op 4858) 

s_add_u32 s15, s18, s8 // (op 4858) 


s_lshr_b32 s8, s15, s10 // (op 4858) 

s_and_b32 s15, s10, 2147483648 // (op 4858) 
s_cmp_eq_i32 s15, 0 // (op 4858) 
s_mov_b32 s18, scc // (op 4858) 

s_cmp_eq_i32 s18, 0 // (op 4858) 
s_cselect_b32 s15, s7, s8 // (op 4858) 


v_lshl_add_u32 v27, s15, 6, v26 // (op 4858) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4858) 
s_mul_i32 s8, s7, s18 // (op 4858) 


s_sub_u32 s7, s6, s8 // (op 4858) 


v_lshl_add_u32 v26, s7, 2, v29 // (op 4858) 


v_mul_lo_u32 v29, v27, s40 // (op 4858) 
v_add_u32 v29, v29, v26 // (op 4858) 


s_mul_i32 s19, s40, s35 // (op 4860) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 4860) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 4860) least significant
s_mul_hi_u32 s6, s40, s34 // (op 4860) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4860) most significant: sum
s_add_u32 s19, s19, s6 // (op 4860) most significant: sum


s_mov_b32 s76, s42 // (op 4860) reg expression
s_mov_b32 s77, s43 // (op 4860) reg expression
s_mov_b32 s78, s18 // (op 4860) reg expression
s_mov_b32 s79, 131072 // (op 4860) call()

v_lshrrev_b32 v26, 2, v1 // (op 4967) 
s_mul_i32 s15, s16, s50 // (op 4967) 
s_add_u32 s15, s15, s0 // (op 4967) 
v_and_b32 v27, -4, v1 // (op 4967) 
s_add_u32 s6, s15, s9 // (op 4967) 

v_sub_u32 v30, v1, v27 // (op 4967) 

s_mul_hi_u32 s15, s6, s13 // (op 4967) 
s_sub_u32 s7, s6, s15 // (op 4967) 
s_lshr_b32 s8, s7, 1 // (op 4967) 

s_add_u32 s7, s8, s15 // (op 4967) 


s_lshr_b32 s8, s7, s11 // (op 4967) 

s_and_b32 s7, s11, 2147483648 // (op 4967) 
s_cmp_eq_i32 s7, 0 // (op 4967) 
s_mov_b32 s15, scc // (op 4967) 

s_cmp_eq_i32 s15, 0 // (op 4967) 
s_cselect_b32 s7, s6, s8 // (op 4967) 


s_mul_hi_u32 s8, s7, s12 // (op 4967) 
s_sub_u32 s15, s7, s8 // (op 4967) 
s_lshr_b32 s18, s15, 1 // (op 4967) 

s_add_u32 s15, s18, s8 // (op 4967) 


s_lshr_b32 s8, s15, s10 // (op 4967) 

s_and_b32 s15, s10, 2147483648 // (op 4967) 
s_cmp_eq_i32 s15, 0 // (op 4967) 
s_mov_b32 s18, scc // (op 4967) 

s_cmp_eq_i32 s18, 0 // (op 4967) 
s_cselect_b32 s15, s7, s8 // (op 4967) 


s_mul_i32 s8, s15, s14 // (op 4967) 

s_sub_u32 s15, s7, s8 // (op 4967) 

v_lshl_add_u32 v27, s15, 6, v26 // (op 4967) 


v_mul_lo_u32 v26, v27, s44 // (op 4967) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 4967) 
s_mul_i32 s8, s7, s18 // (op 4967) 


s_sub_u32 s7, s6, s8 // (op 4967) 


v_lshl_add_u32 v27, s7, 2, v30 // (op 4967) 


v_add_u32 v30, v27, v26 // (op 4967) 


s_mul_i32 s19, s44, s23 // (op 4969) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 4969) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 4969) least significant
s_mul_hi_u32 s6, s44, s22 // (op 4969) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4969) most significant: sum
s_add_u32 s19, s19, s6 // (op 4969) most significant: sum


s_mov_b32 s80, s46 // (op 4969) reg expression
s_mov_b32 s81, s47 // (op 4969) reg expression
s_mov_b32 s82, s18 // (op 4969) reg expression
s_mov_b32 s83, 131072 // (op 4969) call()

v_lshrrev_b32 v26, 2, v1 // (op 4986) 
s_mul_i32 s15, s16, s50 // (op 4986) 
s_add_u32 s15, s15, s0 // (op 4986) 
v_and_b32 v27, -4, v1 // (op 4986) 
s_add_u32 s6, s15, s9 // (op 4986) 

v_sub_u32 v31, v1, v27 // (op 4986) 

s_mul_hi_u32 s15, s6, s13 // (op 4986) 
s_sub_u32 s7, s6, s15 // (op 4986) 
s_lshr_b32 s8, s7, 1 // (op 4986) 

s_add_u32 s7, s8, s15 // (op 4986) 


s_lshr_b32 s8, s7, s11 // (op 4986) 

s_and_b32 s7, s11, 2147483648 // (op 4986) 
s_cmp_eq_i32 s7, 0 // (op 4986) 
s_mov_b32 s15, scc // (op 4986) 

s_cmp_eq_i32 s15, 0 // (op 4986) 
s_cselect_b32 s7, s6, s8 // (op 4986) 


s_mul_hi_u32 s8, s7, s12 // (op 4986) 
s_sub_u32 s15, s7, s8 // (op 4986) 
s_lshr_b32 s18, s15, 1 // (op 4986) 

s_add_u32 s15, s18, s8 // (op 4986) 


s_lshr_b32 s8, s15, s10 // (op 4986) 

s_and_b32 s15, s10, 2147483648 // (op 4986) 
s_cmp_eq_i32 s15, 0 // (op 4986) 
s_mov_b32 s18, scc // (op 4986) 

s_cmp_eq_i32 s18, 0 // (op 4986) 
s_cselect_b32 s15, s7, s8 // (op 4986) 


s_mul_i32 s8, s15, s14 // (op 4986) 

s_sub_u32 s15, s7, s8 // (op 4986) 

v_lshl_add_u32 v27, s15, 6, v26 // (op 4986) 


v_mul_lo_u32 v26, v27, s44 // (op 4986) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 4986) 
s_mul_i32 s8, s7, s18 // (op 4986) 


s_sub_u32 s7, s6, s8 // (op 4986) 


v_lshl_add_u32 v27, s7, 2, v31 // (op 4986) 


v_add_u32 v31, v27, v26 // (op 4986) 


s_mul_i32 s19, s44, s23 // (op 4988) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 4988) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 4988) least significant
s_mul_hi_u32 s6, s44, s22 // (op 4988) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4988) most significant: sum
s_add_u32 s19, s19, s6 // (op 4988) most significant: sum


s_mov_b32 s80, s46 // (op 4988) reg expression
s_mov_b32 s81, s47 // (op 4988) reg expression
s_mov_b32 s82, s18 // (op 4988) reg expression
s_mov_b32 s83, 131072 // (op 4988) call()

v_and_b32 v26, -8, v1 // (op 5559) 
v_sub_u32 v27, v1, v26 // (op 5559) 

v_lshlrev_b32 v26, 4, v27 // (op 5559) 

v_lshrrev_b32 v27, 3, v1 // (op 5559) 
v_lshl_add_u32 v32, v27, 7, v26 // (op 5559) 


s_nop 0
v_readfirstlane_b32 s15, v32 // (op 5559) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6316) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 4485) 
v_mov_b64 v[26:27], s[18:19] // (op 4485) convert


v_mov_b64 v[32:33], v[22:23] // (op 4485) 
s_mov_b32 m0, s15 // (op 4485) reg expression
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 4485) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v26 // (op 4485) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 4485) most significant half

s_add_u32 m0, m0, 4096 // (op 4485) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 4485) Load value direct to lds






buffer_load_ubyte v26, v30, s[80:83], 0 offen offset: 0 // (op 4133) Load value
v_and_b32 v27, -8, v1 // (op 5484) 
v_sub_u32 v32, v1, v27 // (op 5484) 

v_lshlrev_b32 v27, 4, v32 // (op 5484) 

v_lshrrev_b32 v32, 3, v1 // (op 5484) 
v_lshl_add_u32 v33, v32, 7, v27 // (op 5484) 


s_nop 0
v_readfirstlane_b32 s15, v33 // (op 5484) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6297) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 4490) 
v_mov_b64 v[32:33], s[18:19] // (op 4490) convert


v_mov_b64 v[34:35], v[16:17] // (op 4490) 
s_add_u32 s6, 8192, s15 // (op 4490) 
s_mov_b32 m0, s6 // (op 4490) reg expression
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 4490) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 4490) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 4490) most significant half

s_add_u32 m0, m0, 4096 // (op 4490) 
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 4490) Load value direct to lds







buffer_load_ubyte v27, v28, s[76:79], 0 offen offset: 0 // (op 4139) Load value
v_and_b32 v32, -8, v1 // (op 5869) 
v_sub_u32 v33, v1, v32 // (op 5869) 

v_lshlrev_b32 v32, 4, v33 // (op 5869) 

v_lshrrev_b32 v33, 3, v1 // (op 5869) 
v_lshl_add_u32 v34, v33, 7, v32 // (op 5869) 


s_nop 0
v_readfirstlane_b32 s15, v34 // (op 5869) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6386) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 4495) 
v_mov_b64 v[32:33], s[18:19] // (op 4495) convert


v_mov_b64 v[34:35], v[22:23] // (op 4495) 
s_add_u32 s6, 16384, s15 // (op 4495) 
s_mov_b32 m0, s6 // (op 4495) reg expression
s_mov_b32 s7, 128 // (op 4495) call()
buffer_load_dwordx4 v34, s[72:75], s7 offen offset: 0 lds // (op 4495) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 4495) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 4495) most significant half

s_add_u32 m0, m0, 4096 // (op 4495) 
buffer_load_dwordx4 v34, s[72:75], s7 offen offset: 0 lds // (op 4495) Load value direct to lds









buffer_load_ubyte v22, v30, s[80:83], 0 offen offset:4 // (op 4145) Load value

v_and_b32 v23, -8, v1 // (op 5825) 
v_sub_u32 v30, v1, v23 // (op 5825) 

v_lshlrev_b32 v23, 4, v30 // (op 5825) 

v_lshrrev_b32 v30, 3, v1 // (op 5825) 
v_lshl_add_u32 v32, v30, 7, v23 // (op 5825) 


s_nop 0
v_readfirstlane_b32 s15, v32 // (op 5825) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6376) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 4497) 
v_mov_b64 v[32:33], s[18:19] // (op 4497) convert


v_mov_b64 v[34:35], v[16:17] // (op 4497) 
s_add_u32 s6, 24576, s15 // (op 4497) 
s_mov_b32 m0, s6 // (op 4497) reg expression
s_mov_b32 s7, 128 // (op 4497) call()
buffer_load_dwordx4 v34, s[68:71], s7 offen offset: 0 lds // (op 4497) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 4497) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 4497) most significant half

s_add_u32 m0, m0, 4096 // (op 4497) 
buffer_load_dwordx4 v34, s[68:71], s7 offen offset: 0 lds // (op 4497) Load value direct to lds









buffer_load_ubyte v23, v28, s[76:79], 0 offen offset:4 // (op 4151) Load value

v_and_b32 v28, -4, v1 // (op 5711) 
v_sub_u32 v30, v1, v28 // (op 5711) 

v_and_b32 v28, 4294967292, v1 // (op 5711) 
v_add_u32 v16, v28, v30 // (op 5711) 


s_barrier  // (op 6356) 



s_waitcnt vmcnt(9)
ds_write_b8 v16, v26 offset:32768 // (op 4154) Store local data 

v_and_b32 v28, -4, v1 // (op 5635) 
v_sub_u32 v30, v1, v28 // (op 5635) 

v_and_b32 v28, 4294967292, v1 // (op 5635) 
v_add_u32 v16, v28, v30 // (op 5635) 


s_barrier  // (op 6336) 



s_waitcnt vmcnt(6)
ds_write_b8 v16, v27 offset:33024 // (op 4156) Store local data 

s_barrier  // (op 6313) 



ds_read_b128 v[32:35], v14 // (op 4181) Load local data 
ds_read_b128 v[36:39], v14 offset:64 // (op 4181) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 6353) 



ds_read_u8 v28, v15 offset:32768 // (op 4201) Load local data 



ds_read_b128 v[40:43], v14 offset:4096 // (op 4186) Load local data 
ds_read_b128 v[44:47], v14 offset:4160 // (op 4186) Load local data 



ds_read_u8 v30, v15 offset:32896 // (op 4206) Load local data 
s_barrier  // (op 6294) 



ds_read_b128 v[48:51], v3 offset:8192 // (op 4171) Load local data 
ds_read_b128 v[52:55], v3 offset:8256 // (op 4171) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 6333) 



ds_read_u8 v16, v0 offset:33024 // (op 4191) Load local data 



ds_read_b128 v[56:59], v3 offset:12288 // (op 4176) Load local data 
ds_read_b128 v[60:63], v3 offset:12352 // (op 4176) Load local data 



ds_read_u8 v17, v0 offset:33152 // (op 4196) Load local data 
v_and_b32 v64, -8, v1 // (op 5456) 
v_sub_u32 v65, v1, v64 // (op 5456) 

v_lshlrev_b32 v64, 4, v65 // (op 5456) 

v_lshrrev_b32 v65, 3, v1 // (op 5456) 
v_lshl_add_u32 v66, v65, 7, v64 // (op 5456) 


s_nop 0
v_readfirstlane_b32 s15, v66 // (op 5456) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v64, -8, v1 // (op 5531) 
v_sub_u32 v65, v1, v64 // (op 5531) 

v_lshlrev_b32 v64, 4, v65 // (op 5531) 

v_lshrrev_b32 v65, 3, v1 // (op 5531) 
v_lshl_add_u32 v66, v65, 7, v64 // (op 5531) 


s_nop 0
v_readfirstlane_b32 s6, v66 // (op 5531) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v64, -4, v1 // (op 5606) 
v_sub_u32 v65, v1, v64 // (op 5606) 

v_and_b32 v64, 4294967292, v1 // (op 5606) 
v_add_u32 v66, v64, v65 // (op 5606) 


v_and_b32 v64, -4, v1 // (op 5682) 
v_sub_u32 v65, v1, v64 // (op 5682) 

v_and_b32 v64, 4294967292, v1 // (op 5682) 
v_add_u32 v67, v64, v65 // (op 5682) 


v_and_b32 v64, -8, v1 // (op 5810) 
v_sub_u32 v65, v1, v64 // (op 5810) 

v_lshlrev_b32 v64, 4, v65 // (op 5810) 

v_lshrrev_b32 v65, 3, v1 // (op 5810) 
v_lshl_add_u32 v68, v65, 7, v64 // (op 5810) 


s_nop 0
v_readfirstlane_b32 s7, v68 // (op 5810) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v64, -8, v1 // (op 5854) 
v_sub_u32 v65, v1, v64 // (op 5854) 

v_lshlrev_b32 v64, 4, v65 // (op 5854) 

v_lshrrev_b32 v65, 3, v1 // (op 5854) 
v_lshl_add_u32 v68, v65, 7, v64 // (op 5854) 


s_nop 0
v_readfirstlane_b32 s8, v68 // (op 5854) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v64, -4, v1 // (op 5898) 
v_sub_u32 v65, v1, v64 // (op 5898) 

v_and_b32 v64, 4294967292, v1 // (op 5898) 
v_add_u32 v68, v64, v65 // (op 5898) 


v_and_b32 v64, -4, v1 // (op 5927) 
v_sub_u32 v65, v1, v64 // (op 5927) 

v_and_b32 v64, 4294967292, v1 // (op 5927) 
v_add_u32 v69, v64, v65 // (op 5927) 


s_mov_b32 s1, 0 // (op 867) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 866) 
s_mov_b32 s28, s18 // (op 866) convert

s_and_b32 s29, s28, 4294967294 // (op 866) 

s_cmp_lt_u32 s1, s29 // (op 866) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_866 // (op 866) Condition: Top (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_866 if false)
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoop_866:
s_add_u32 s1, s1, 2 // (op 868) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[32:39], a[0:3], v16, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 194) 
s_barrier  // (op 6319) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:55], v[40:47], a[4:7], v16, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 654) 
s_lshl_b64 s[18:19], s[36:37], 5 // (op 4434) 
v_mov_b64 v[64:65], s[18:19] // (op 4434) convert


v_mov_b64 v[70:71], v[24:25] // (op 4434) 
v_mov_b64 v[72:73], v[70:71] // (op 4434) 
s_mov_b32 m0, s6 // (op 4434) reg expression
s_mov_b32 s18, 256 // (op 4434) call()
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:63], v[32:39], a[8:11], v17, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 951) 
buffer_load_dwordx4 v72, s[72:75], s18 offen offset: 0 lds // (op 4434) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:63], v[40:47], a[12:15], v17, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 905) 
v_add_co_u32 v72, s[28:29], v72, v64 // (op 4434) least significant half
v_addc_co_u32 v73, s[28:29], 0, v73, s[28:29] // (op 4434) most significant half

s_add_u32 m0, m0, 4096 // (op 4434) 
buffer_load_dwordx4 v72, s[72:75], s18 offen offset: 0 lds // (op 4434) Load value direct to lds





v_mov_b32 v64, v31 // (op 200) 

buffer_load_ubyte v26, v64, s[80:83], 0 offen offset:8 // (op 200) Load value
s_barrier  // (op 6300) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 4429) 
v_mov_b64 v[72:73], s[18:19] // (op 4429) convert


v_mov_b64 v[74:75], v[20:21] // (op 4429) 
v_mov_b64 v[76:77], v[74:75] // (op 4429) 
s_add_u32 s18, 8192, s15 // (op 4429) 
s_mov_b32 m0, s18 // (op 4429) reg expression
s_mov_b32 s19, 256 // (op 4429) call()
buffer_load_dwordx4 v76, s[68:71], s19 offen offset: 0 lds // (op 4429) Load value direct to lds
v_add_co_u32 v76, s[28:29], v76, v72 // (op 4429) least significant half
v_addc_co_u32 v77, s[28:29], 0, v77, s[28:29] // (op 4429) most significant half

s_add_u32 m0, m0, 4096 // (op 4429) 
buffer_load_dwordx4 v76, s[68:71], s19 offen offset: 0 lds // (op 4429) Load value direct to lds






v_mov_b32 v65, v29 // (op 199) 

buffer_load_ubyte v27, v65, s[76:79], 0 offen offset:8 // (op 199) Load value
s_barrier  // (op 6401) 



s_waitcnt vmcnt(9)
ds_write_b8 v69, v22 offset:33280 // (op 533) Store local data 
s_barrier  // (op 6395) 



s_waitcnt vmcnt(6)
ds_write_b8 v68, v23 offset:33536 // (op 532) Store local data 
s_barrier  // (op 6382) 



ds_read_b128 v[32:35], v12 offset:16384 // (op 525) Load local data 
ds_read_b128 v[36:39], v12 offset:16448 // (op 525) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 6398) 



ds_read_u8 v28, v13 offset:33280 // (op 529) Load local data 



ds_read_b128 v[40:43], v12 offset:20480 // (op 659) Load local data 
ds_read_b128 v[44:47], v12 offset:20544 // (op 659) Load local data 



ds_read_u8 v30, v13 offset:33408 // (op 663) Load local data 
s_barrier  // (op 6373) 
s_barrier  // (op 6511) 



ds_read_b128 v[48:51], v11 offset:24576 // (op 523) Load local data 
ds_read_b128 v[52:55], v11 offset:24640 // (op 523) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 6392) 



ds_read_u8 v16, v10 offset:33536 // (op 527) Load local data 



ds_read_b128 v[56:59], v11 offset:28672 // (op 908) Load local data 
ds_read_b128 v[60:63], v11 offset:28736 // (op 908) Load local data 



ds_read_u8 v17, v10 offset:33664 // (op 911) Load local data 
s_barrier  // (op 6533) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[32:39], a[0:3], v16, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 530) 
s_barrier  // (op 6389) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:55], v[40:47], a[4:7], v16, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 655) 
s_lshl_b64 s[18:19], s[36:37], 5 // (op 4457) 
v_mov_b64 v[72:73], s[18:19] // (op 4457) convert


v_mov_b64 v[76:77], v[70:71] // (op 4457) 
s_add_u32 s18, 16384, s8 // (op 4457) 
s_mov_b32 m0, s18 // (op 4457) reg expression
s_mov_b32 s19, 384 // (op 4457) call()
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:63], v[32:39], a[8:11], v17, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 952) 
buffer_load_dwordx4 v76, s[72:75], s19 offen offset: 0 lds // (op 4457) Load value direct to lds
v_add_co_u32 v76, s[28:29], v76, v72 // (op 4457) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:63], v[40:47], a[12:15], v17, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 906) 
v_addc_co_u32 v77, s[28:29], 0, v77, s[28:29] // (op 4457) most significant half

s_add_u32 m0, m0, 4096 // (op 4457) 
buffer_load_dwordx4 v76, s[72:75], s19 offen offset: 0 lds // (op 4457) Load value direct to lds








v_add_co_u32 v24, vcc, 256, v24 // (op 4722) least significant half
buffer_load_ubyte v22, v64, s[80:83], 0 offen offset:12 // (op 536) Load value
v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 4722) most significant half

s_barrier  // (op 6379) 
v_add_u32 v31, 8, v31 // (op 4978) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 4452) 
v_mov_b64 v[70:71], s[18:19] // (op 4452) convert


v_mov_b64 v[72:73], v[74:75] // (op 4452) 
s_add_u32 s18, 24576, s7 // (op 4452) 
s_mov_b32 m0, s18 // (op 4452) reg expression
s_mov_b32 s19, 384 // (op 4452) call()
buffer_load_dwordx4 v72, s[68:71], s19 offen offset: 0 lds // (op 4452) Load value direct to lds
v_add_co_u32 v72, s[28:29], v72, v70 // (op 4452) least significant half
v_addc_co_u32 v73, s[28:29], 0, v73, s[28:29] // (op 4452) most significant half

s_add_u32 m0, m0, 4096 // (op 4452) 
buffer_load_dwordx4 v72, s[68:71], s19 offen offset: 0 lds // (op 4452) Load value direct to lds








v_add_co_u32 v20, vcc, 256, v20 // (op 4588) least significant half
buffer_load_ubyte v23, v65, s[76:79], 0 offen offset:12 // (op 535) Load value
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 4588) most significant half

s_barrier  // (op 6359) 
v_add_u32 v29, 8, v29 // (op 4850) 



s_waitcnt vmcnt(9)
ds_write_b8 v67, v26 offset:32768 // (op 197) Store local data 
s_barrier  // (op 6339) 



s_waitcnt vmcnt(6)
ds_write_b8 v66, v27 offset:33024 // (op 196) Store local data 
s_barrier  // (op 6325) 



ds_read_b128 v[32:35], v8 // (op 189) Load local data 
ds_read_b128 v[36:39], v8 offset:64 // (op 189) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 6366) 



ds_read_u8 v28, v9 offset:32768 // (op 193) Load local data 



ds_read_b128 v[40:43], v8 offset:4096 // (op 649) Load local data 
ds_read_b128 v[44:47], v8 offset:4160 // (op 649) Load local data 



ds_read_u8 v30, v9 offset:32896 // (op 653) Load local data 
s_barrier  // (op 6306) 



ds_read_b128 v[48:51], v7 offset:8192 // (op 187) Load local data 
ds_read_b128 v[52:55], v7 offset:8256 // (op 187) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 6346) 



ds_read_u8 v16, v6 offset:33024 // (op 191) Load local data 



ds_read_b128 v[56:59], v7 offset:12288 // (op 900) Load local data 
ds_read_b128 v[60:63], v7 offset:12352 // (op 900) Load local data 



ds_read_u8 v17, v6 offset:33152 // (op 903) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 866) 
s_mov_b32 s28, s18 // (op 866) convert

s_and_b32 s29, s28, 4294967294 // (op 866) 

s_cmp_lt_u32 s1, s29 // (op 866) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoop_866 // (op 866) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoop_866 if true)
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_866:














s_barrier  // (op 6517) 
v_lshrrev_b32 v29, 3, v1 // (op 4524) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4524) 
v_and_b32 v31, -8, v1 // (op 4524) 
s_and_b32 s15, s18, 4294967294 // (op 4524) 
v_sub_u32 v20, v1, v31 // (op 4524) 

s_add_u32 s6, s0, s15 // (op 4524) 

v_lshlrev_b32 v21, 4, v20 // (op 4524) 

s_mul_i32 s15, s16, s50 // (op 4524) 
s_add_u32 s15, s15, s6 // (op 4524) 

s_add_u32 s6, s15, s9 // (op 4524) 

s_mul_hi_u32 s15, s6, s13 // (op 4524) 
s_sub_u32 s7, s6, s15 // (op 4524) 
s_lshr_b32 s8, s7, 1 // (op 4524) 

s_add_u32 s7, s8, s15 // (op 4524) 


s_lshr_b32 s8, s7, s11 // (op 4524) 

s_and_b32 s7, s11, 2147483648 // (op 4524) 
s_cmp_eq_i32 s7, 0 // (op 4524) 
s_mov_b32 s15, scc // (op 4524) 

s_cmp_eq_i32 s15, 0 // (op 4524) 
s_cselect_b32 s7, s6, s8 // (op 4524) 


s_mul_hi_u32 s8, s7, s12 // (op 4524) 
s_sub_u32 s15, s7, s8 // (op 4524) 
s_lshr_b32 s28, s15, 1 // (op 4524) 

s_add_u32 s15, s28, s8 // (op 4524) 


s_lshr_b32 s8, s15, s10 // (op 4524) 

s_and_b32 s15, s10, 2147483648 // (op 4524) 
s_cmp_eq_i32 s15, 0 // (op 4524) 
s_mov_b32 s28, scc // (op 4524) 

s_cmp_eq_i32 s28, 0 // (op 4524) 
s_cselect_b32 s15, s7, s8 // (op 4524) 


s_mul_i32 s8, s7, s18 // (op 4524) 


v_lshl_add_u32 v20, s15, 6, v29 // (op 4524) 


s_sub_u32 s7, s6, s8 // (op 4524) 


v_lshl_add_u32 v29, s7, 7, v21 // (op 4524) 


v_mov_b32 v25, 0 // (op 4524) convert
v_mov_b32 v24, v29 // (op 4524) convert

v_mul_lo_u32 v27, v20, s31 // (op 4524) most significant: low of low * high
v_mul_hi_u32 v21, v20, s30 // (op 4524) most significant: high of low * low
v_mul_lo_u32 v26, v20, s30 // (op 4524) least significant: low of low * low
v_add_u32 v27, v27, v21 // (op 4524) most significant: sum

v_add_co_u32 v26, s[18:19], v26, v24 // (op 4524) least significant half
v_addc_co_u32 v27, s[18:19], v27, v25, s[18:19] // (op 4524) most significant half



s_mul_i32 s19, s30, s35 // (op 4526) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 4526) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 4526) least significant
s_mul_hi_u32 s6, s30, s34 // (op 4526) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4526) most significant: sum
s_add_u32 s19, s19, s6 // (op 4526) most significant: sum


s_mov_b32 s68, s32 // (op 4526) reg expression
s_mov_b32 s69, s33 // (op 4526) reg expression
s_mov_b32 s70, s18 // (op 4526) reg expression
s_mov_b32 s71, 131072 // (op 4526) call()

v_lshrrev_b32 v29, 3, v1 // (op 4666) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4666) 
v_and_b32 v31, -8, v1 // (op 4666) 
s_and_b32 s15, s18, 4294967294 // (op 4666) 
v_sub_u32 v20, v1, v31 // (op 4666) 

s_add_u32 s6, s0, s15 // (op 4666) 

v_lshlrev_b32 v21, 4, v20 // (op 4666) 

s_mul_i32 s15, s16, s50 // (op 4666) 
s_add_u32 s15, s15, s6 // (op 4666) 

s_add_u32 s6, s15, s9 // (op 4666) 

s_mul_hi_u32 s15, s6, s13 // (op 4666) 
s_sub_u32 s7, s6, s15 // (op 4666) 
s_lshr_b32 s8, s7, 1 // (op 4666) 

s_add_u32 s7, s8, s15 // (op 4666) 


s_lshr_b32 s8, s7, s11 // (op 4666) 

s_and_b32 s7, s11, 2147483648 // (op 4666) 
s_cmp_eq_i32 s7, 0 // (op 4666) 
s_mov_b32 s15, scc // (op 4666) 

s_cmp_eq_i32 s15, 0 // (op 4666) 
s_cselect_b32 s7, s6, s8 // (op 4666) 


s_mul_hi_u32 s8, s7, s12 // (op 4666) 
s_sub_u32 s15, s7, s8 // (op 4666) 
s_lshr_b32 s28, s15, 1 // (op 4666) 

s_add_u32 s15, s28, s8 // (op 4666) 


s_lshr_b32 s8, s15, s10 // (op 4666) 

s_and_b32 s15, s10, 2147483648 // (op 4666) 
s_cmp_eq_i32 s15, 0 // (op 4666) 
s_mov_b32 s28, scc // (op 4666) 

s_cmp_eq_i32 s28, 0 // (op 4666) 
s_cselect_b32 s15, s7, s8 // (op 4666) 


s_mul_i32 s8, s15, s14 // (op 4666) 

s_sub_u32 s15, s7, s8 // (op 4666) 

s_mul_i32 s8, s7, s18 // (op 4666) 


v_lshl_add_u32 v20, s15, 6, v29 // (op 4666) 


s_sub_u32 s7, s6, s8 // (op 4666) 


v_mul_lo_u32 v25, v20, s37 // (op 4666) most significant: low of low * high
v_mul_hi_u32 v29, v20, s36 // (op 4666) most significant: high of low * low
v_mul_lo_u32 v24, v20, s36 // (op 4666) least significant: low of low * low
v_add_u32 v25, v25, v29 // (op 4666) most significant: sum


v_lshl_add_u32 v20, s7, 7, v21 // (op 4666) 


v_mov_b32 v65, 0 // (op 4666) convert
v_mov_b32 v64, v20 // (op 4666) convert

v_add_co_u32 v20, s[18:19], v64, v24 // (op 4666) least significant half
v_addc_co_u32 v21, s[18:19], v65, v25, s[18:19] // (op 4666) most significant half



s_mul_i32 s19, s36, s23 // (op 4668) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 4668) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 4668) least significant
s_mul_hi_u32 s6, s36, s22 // (op 4668) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4668) most significant: sum
s_add_u32 s19, s19, s6 // (op 4668) most significant: sum


s_mov_b32 s72, s38 // (op 4668) reg expression
s_mov_b32 s73, s39 // (op 4668) reg expression
s_mov_b32 s74, s18 // (op 4668) reg expression
s_mov_b32 s75, 131072 // (op 4668) call()

v_lshrrev_b32 v29, 2, v1 // (op 4794) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4794) 
v_and_b32 v31, -4, v1 // (op 4794) 
s_and_b32 s15, s18, 4294967294 // (op 4794) 
v_sub_u32 v24, v1, v31 // (op 4794) 

s_add_u32 s6, s0, s15 // (op 4794) 

s_mul_i32 s15, s16, s50 // (op 4794) 
s_add_u32 s15, s15, s6 // (op 4794) 

s_add_u32 s6, s15, s9 // (op 4794) 

s_mul_hi_u32 s15, s6, s13 // (op 4794) 
s_sub_u32 s7, s6, s15 // (op 4794) 
s_lshr_b32 s8, s7, 1 // (op 4794) 

s_add_u32 s7, s8, s15 // (op 4794) 


s_lshr_b32 s8, s7, s11 // (op 4794) 

s_and_b32 s7, s11, 2147483648 // (op 4794) 
s_cmp_eq_i32 s7, 0 // (op 4794) 
s_mov_b32 s15, scc // (op 4794) 

s_cmp_eq_i32 s15, 0 // (op 4794) 
s_cselect_b32 s7, s6, s8 // (op 4794) 


s_mul_hi_u32 s8, s7, s12 // (op 4794) 
s_sub_u32 s15, s7, s8 // (op 4794) 
s_lshr_b32 s28, s15, 1 // (op 4794) 

s_add_u32 s15, s28, s8 // (op 4794) 


s_lshr_b32 s8, s15, s10 // (op 4794) 

s_and_b32 s15, s10, 2147483648 // (op 4794) 
s_cmp_eq_i32 s15, 0 // (op 4794) 
s_mov_b32 s28, scc // (op 4794) 

s_cmp_eq_i32 s28, 0 // (op 4794) 
s_cselect_b32 s15, s7, s8 // (op 4794) 


s_mul_i32 s8, s7, s18 // (op 4794) 


v_lshl_add_u32 v25, s15, 6, v29 // (op 4794) 


s_sub_u32 s7, s6, s8 // (op 4794) 


v_lshl_add_u32 v29, s7, 2, v24 // (op 4794) 


v_mul_lo_u32 v24, v25, s40 // (op 4794) 
v_add_u32 v24, v24, v29 // (op 4794) 


s_mul_i32 s19, s40, s35 // (op 4796) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 4796) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 4796) least significant
s_mul_hi_u32 s6, s40, s34 // (op 4796) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4796) most significant: sum
s_add_u32 s19, s19, s6 // (op 4796) most significant: sum


s_mov_b32 s76, s42 // (op 4796) reg expression
s_mov_b32 s77, s43 // (op 4796) reg expression
s_mov_b32 s78, s18 // (op 4796) reg expression
s_mov_b32 s79, 131072 // (op 4796) call()

v_lshrrev_b32 v25, 2, v1 // (op 4922) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4922) 
v_and_b32 v29, -4, v1 // (op 4922) 
s_and_b32 s15, s18, 4294967294 // (op 4922) 
v_sub_u32 v31, v1, v29 // (op 4922) 

s_add_u32 s6, s0, s15 // (op 4922) 

s_mul_i32 s15, s16, s50 // (op 4922) 
s_add_u32 s15, s15, s6 // (op 4922) 

s_add_u32 s6, s15, s9 // (op 4922) 

s_mul_hi_u32 s15, s6, s13 // (op 4922) 
s_sub_u32 s7, s6, s15 // (op 4922) 
s_lshr_b32 s8, s7, 1 // (op 4922) 

s_add_u32 s7, s8, s15 // (op 4922) 


s_lshr_b32 s8, s7, s11 // (op 4922) 

s_and_b32 s7, s11, 2147483648 // (op 4922) 
s_cmp_eq_i32 s7, 0 // (op 4922) 
s_mov_b32 s15, scc // (op 4922) 

s_cmp_eq_i32 s15, 0 // (op 4922) 
s_cselect_b32 s7, s6, s8 // (op 4922) 


s_mul_hi_u32 s8, s7, s12 // (op 4922) 
s_sub_u32 s15, s7, s8 // (op 4922) 
s_lshr_b32 s28, s15, 1 // (op 4922) 

s_add_u32 s15, s28, s8 // (op 4922) 


s_lshr_b32 s8, s15, s10 // (op 4922) 

s_and_b32 s15, s10, 2147483648 // (op 4922) 
s_cmp_eq_i32 s15, 0 // (op 4922) 
s_mov_b32 s28, scc // (op 4922) 

s_cmp_eq_i32 s28, 0 // (op 4922) 
s_cselect_b32 s15, s7, s8 // (op 4922) 


s_mul_i32 s8, s15, s14 // (op 4922) 

s_sub_u32 s15, s7, s8 // (op 4922) 

v_lshl_add_u32 v29, s15, 6, v25 // (op 4922) 


s_mul_i32 s8, s7, s18 // (op 4922) 


v_mul_lo_u32 v25, v29, s44 // (op 4922) 

s_sub_u32 s7, s6, s8 // (op 4922) 


v_lshl_add_u32 v29, s7, 2, v31 // (op 4922) 


v_add_u32 v31, v29, v25 // (op 4922) 


s_mul_i32 s19, s44, s23 // (op 4924) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 4924) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 4924) least significant
s_mul_hi_u32 s6, s44, s22 // (op 4924) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4924) most significant: sum
s_add_u32 s19, s19, s6 // (op 4924) most significant: sum


s_mov_b32 s80, s46 // (op 4924) reg expression
s_mov_b32 s81, s47 // (op 4924) reg expression
s_mov_b32 s82, s18 // (op 4924) reg expression
s_mov_b32 s83, 131072 // (op 4924) call()

v_and_b32 v25, -8, v1 // (op 5469) 
v_sub_u32 v29, v1, v25 // (op 5469) 

v_lshlrev_b32 v25, 4, v29 // (op 5469) 

v_lshrrev_b32 v29, 3, v1 // (op 5469) 
v_lshl_add_u32 v64, v29, 7, v25 // (op 5469) 


s_nop 0
v_readfirstlane_b32 s15, v64 // (op 5469) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -8, v1 // (op 5544) 
v_sub_u32 v29, v1, v25 // (op 5544) 

v_lshlrev_b32 v25, 4, v29 // (op 5544) 

v_lshrrev_b32 v29, 3, v1 // (op 5544) 
v_lshl_add_u32 v64, v29, 7, v25 // (op 5544) 


s_nop 0
v_readfirstlane_b32 s6, v64 // (op 5544) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -4, v1 // (op 5619) 
v_sub_u32 v29, v1, v25 // (op 5619) 

v_and_b32 v25, 4294967292, v1 // (op 5619) 
v_add_u32 v64, v25, v29 // (op 5619) 


v_and_b32 v25, -4, v1 // (op 5695) 
v_sub_u32 v29, v1, v25 // (op 5695) 

v_and_b32 v25, 4294967292, v1 // (op 5695) 
v_add_u32 v65, v25, v29 // (op 5695) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 872) 
s_and_b32 s7, s18, 4294967294 // (op 872) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 871) 
s_mov_b32 s8, s18 // (op 871) convert

s_cmp_lt_u32 s7, s8 // (op 871) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_871 // (op 871) Condition: Top (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_871 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_871:
s_barrier  // (op 6303) 
s_add_u32 s7, s7, 1 // (op 873) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 4439) 
v_mov_b64 v[16:17], s[18:19] // (op 4439) convert


v_mov_b64 v[22:23], v[26:27] // (op 4439) 
v_mov_b64 v[28:29], v[22:23] // (op 4439) 
s_add_u32 s8, 8192, s15 // (op 4439) 
s_mov_b32 m0, s8 // (op 4439) reg expression
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 4439) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v16 // (op 4439) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 4439) most significant half

s_add_u32 m0, m0, 4096 // (op 4439) 
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 4439) Load value direct to lds




v_add_co_u32 v26, vcc, 128, v26 // (op 4514) least significant half
v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 4514) most significant half
s_barrier  // (op 6322) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 4445) 
v_mov_b64 v[16:17], s[18:19] // (op 4445) convert


v_mov_b64 v[22:23], v[20:21] // (op 4445) 
v_mov_b64 v[28:29], v[22:23] // (op 4445) 
s_mov_b32 m0, s6 // (op 4445) reg expression
buffer_load_dwordx4 v28, s[72:75], 0 offen offset: 0 lds // (op 4445) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v16 // (op 4445) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 4445) most significant half

s_add_u32 m0, m0, 4096 // (op 4445) 
buffer_load_dwordx4 v28, s[72:75], 0 offen offset: 0 lds // (op 4445) Load value direct to lds



v_add_co_u32 v20, vcc, 128, v20 // (op 4658) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 4658) most significant half


v_mov_b32 v25, v24 // (op 469) 

buffer_load_ubyte v16, v25, s[76:79], 0 offen offset: 0 // (op 469) Load value

v_add_u32 v24, 4, v24 // (op 4786) 


s_barrier  // (op 6342) 
v_mov_b32 v17, v31 // (op 470) 




buffer_load_ubyte v25, v17, s[80:83], 0 offen offset: 0 // (op 470) Load value

v_add_u32 v31, 4, v31 // (op 4914) 
s_waitcnt vmcnt(1)
ds_write_b8 v64, v16 offset:33024 // (op 466) Store local data 

s_barrier  // (op 6309) 



s_barrier  // (op 6362) 
ds_read_b128 v[32:35], v7 offset:8192 // (op 457) Load local data 



ds_read_b128 v[36:39], v7 offset:8256 // (op 457) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v65, v25 offset:32768 // (op 467) Store local data 

s_barrier  // (op 6329) 



ds_read_b128 v[40:43], v8 // (op 459) Load local data 
ds_read_b128 v[44:47], v8 offset:64 // (op 459) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 6349) 



ds_read_u8 v25, v6 offset:33024 // (op 461) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 6369) 



ds_read_u8 v16, v9 offset:32768 // (op 463) Load local data 



ds_read_b128 v[48:51], v8 offset:4096 // (op 628) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[40:47], a[0:3], v25, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 464) 
ds_read_b128 v[52:55], v8 offset:4160 // (op 628) Load local data 



ds_read_u8 v17, v9 offset:32896 // (op 632) Load local data 



ds_read_b128 v[56:59], v7 offset:12288 // (op 846) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:39], v[48:55], a[4:7], v25, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 633) 
ds_read_b128 v[60:63], v7 offset:12352 // (op 846) Load local data 





ds_read_u8 v25, v6 offset:33152 // (op 849) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:63], v[40:47], a[8:11], v25, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 851) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:63], v[48:55], a[12:15], v25, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 882) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 871) 
s_mov_b32 s8, s18 // (op 871) convert

s_cmp_lt_u32 s7, s8 // (op 871) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_871 // (op 871) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_871 if true)
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_871:









s_barrier  // (op 6521) 




v_cmp_eq_f32 vcc, 0.00000, v18 // (op 234) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_234 // (op 234) Condition: False, jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_234
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


v_accvgpr_read v14, a4 // (op 1295) 
v_accvgpr_read v15, a5 // (op 1295) 
v_accvgpr_read v16, a6 // (op 1295) 
v_accvgpr_read v17, a7 // (op 1295) 
v_mul_f32 v20, v19, v14 // (op 1295) 
v_mul_f32 v21, v19, v15 // (op 1295) 
v_mul_f32 v22, v19, v16 // (op 1295) 
v_mul_f32 v23, v19, v17 // (op 1295) 



s_lshl_b32 s15, s24, 7 // (op 1297) 
v_add_u32 v0, v5, s15 // (op 1297) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 1297) Store value


v_accvgpr_read v14, a8 // (op 1319) 
v_accvgpr_read v15, a9 // (op 1319) 
v_accvgpr_read v16, a10 // (op 1319) 
v_accvgpr_read v17, a11 // (op 1319) 
v_mul_f32 v20, v19, v14 // (op 1319) 
v_mul_f32 v21, v19, v15 // (op 1319) 
v_mul_f32 v22, v19, v16 // (op 1319) 
v_mul_f32 v23, v19, v17 // (op 1319) 



v_add_u32 v0, 128, v5 // (op 1322) 

buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 1322) Store value


v_accvgpr_read v14, a12 // (op 1328) 
v_accvgpr_read v15, a13 // (op 1328) 
v_accvgpr_read v16, a14 // (op 1328) 
v_accvgpr_read v17, a15 // (op 1328) 
v_mul_f32 v20, v19, v14 // (op 1328) 
v_mul_f32 v21, v19, v15 // (op 1328) 
v_mul_f32 v22, v19, v16 // (op 1328) 
v_mul_f32 v23, v19, v17 // (op 1328) 



s_lshl_b32 s15, s24, 7 // (op 1324) 
s_add_u32 s15, s15, 128 // (op 1324) 
v_add_u32 v0, v5, s15 // (op 1324) 


buffer_store_dwordx4 v[20:23], v0, s[60:63], 0 offen // (op 1324) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_234 // (op 234) Condition: Done, jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_234
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_234:


v_add_u32 v0, 128, v4 // (op 1408) 

buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 1408) Load value

v_accvgpr_read v20, a8 // (op 1400) 
v_accvgpr_read v21, a9 // (op 1400) 
v_accvgpr_read v22, a10 // (op 1400) 
v_accvgpr_read v23, a11 // (op 1400) 
v_mul_f32 v24, v19, v20 // (op 1400) 
v_mul_f32 v25, v19, v21 // (op 1400) 
v_mul_f32 v26, v19, v22 // (op 1400) 
v_mul_f32 v27, v19, v23 // (op 1400) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 1407) 
v_mul_f32 v21, v18, v15 // (op 1407) 
v_mul_f32 v22, v18, v16 // (op 1407) 
v_mul_f32 v23, v18, v17 // (op 1407) 

v_add_f32 v14, v20, v24 // (op 1401) 
v_add_f32 v15, v21, v25 // (op 1401) 
v_add_f32 v16, v22, v26 // (op 1401) 
v_add_f32 v17, v23, v27 // (op 1401) 




v_add_u32 v0, 128, v5 // (op 1403) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1403) Store value




s_lshl_b32 s15, s4, 7 // (op 1412) 
s_add_u32 s15, s15, 128 // (op 1412) 
v_add_u32 v0, v4, s15 // (op 1412) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 1412) Load value

v_accvgpr_read v20, a12 // (op 1414) 
v_accvgpr_read v21, a13 // (op 1414) 
v_accvgpr_read v22, a14 // (op 1414) 
v_accvgpr_read v23, a15 // (op 1414) 
v_mul_f32 v24, v19, v20 // (op 1414) 
v_mul_f32 v25, v19, v21 // (op 1414) 
v_mul_f32 v26, v19, v22 // (op 1414) 
v_mul_f32 v27, v19, v23 // (op 1414) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 1410) 
v_mul_f32 v21, v18, v15 // (op 1410) 
v_mul_f32 v22, v18, v16 // (op 1410) 
v_mul_f32 v23, v18, v17 // (op 1410) 

v_add_f32 v14, v20, v24 // (op 1411) 
v_add_f32 v15, v21, v25 // (op 1411) 
v_add_f32 v16, v22, v26 // (op 1411) 
v_add_f32 v17, v23, v27 // (op 1411) 




s_lshl_b32 s15, s24, 7 // (op 1405) 
s_add_u32 s15, s15, 128 // (op 1405) 
v_add_u32 v0, v5, s15 // (op 1405) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1405) Store value




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




s_lshl_b32 s15, s4, 7 // (op 1370) 
v_add_u32 v0, v4, s15 // (op 1370) 


buffer_load_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 1370) Load value

v_accvgpr_read v20, a4 // (op 1367) 
v_accvgpr_read v21, a5 // (op 1367) 
v_accvgpr_read v22, a6 // (op 1367) 
v_accvgpr_read v23, a7 // (op 1367) 
v_mul_f32 v24, v19, v20 // (op 1367) 
v_mul_f32 v25, v19, v21 // (op 1367) 
v_mul_f32 v26, v19, v22 // (op 1367) 
v_mul_f32 v27, v19, v23 // (op 1367) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v18, v14 // (op 1371) 
v_mul_f32 v21, v18, v15 // (op 1371) 
v_mul_f32 v22, v18, v16 // (op 1371) 
v_mul_f32 v23, v18, v17 // (op 1371) 

v_add_f32 v14, v20, v24 // (op 1368) 
v_add_f32 v15, v21, v25 // (op 1368) 
v_add_f32 v16, v22, v26 // (op 1368) 
v_add_f32 v17, v23, v27 // (op 1368) 




s_lshl_b32 s15, s24, 7 // (op 1369) 
v_add_u32 v0, v5, s15 // (op 1369) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1369) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_234:
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
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_DPStreamTileLoop_441 // (op 441) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_DPStreamTileLoop_441 if true)
RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_441:













































s_endpgm  // End of RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2
.LRR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_end:
.size RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2, .LRR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_end-RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2
  .amdhsa_next_free_vgpr 104
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 33792 // lds bytes
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: B8
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
    macrotile: [64, 64, 128]
    threads: [256, 1, 1]
    grid: [RocRollerStreamK, One, One]
    occupancy: 4
    workspaceType: RocRollerStreamK
    workspaceSizePerElemC: 4
    workspaceSizePerElemBias: 0
    unroll: 2
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2
    .symbol: RR_GEMM_TN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 33792
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 88
    .vgpr_count: 88
    .agpr_count: 16
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

