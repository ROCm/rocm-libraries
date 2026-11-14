// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2
.p2align 8
.type RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2,@function
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 5180) 
v_and_b32 v3, 63, v1 // (op 5180) 
v_sub_u32 v4, v3, v0 // (op 5180) 

v_lshrrev_b32 v3, 1, v4 // (op 5180) 
v_add_lshl_u32 v5, v0, v3, 6 // (op 5180) 


v_and_b32 v0, 1, v4 // (op 5180) 

v_lshlrev_b32 v3, 3, v0 // (op 5180) 

v_lshrrev_b32 v0, 7, v1 // (op 5180) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 5180) 


v_add_u32 v0, v5, v4 // (op 5180) 


v_lshrrev_b32 v3, 6, v1 // (op 5255) 
v_and_b32 v4, 1, v3 // (op 5255) 

v_and_b32 v3, 15, v1 // (op 5255) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 5255) 


v_and_b32 v3, 48, v1 // (op 5255) 
v_lshl_add_u32 v4, v5, 7, v3 // (op 5255) 


v_lshrrev_b32 v3, 7, v1 // (op 5331) 
v_and_b32 v5, 15, v1 // (op 5331) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 5331) 


v_and_b32 v3, 63, v1 // (op 5331) 
v_lshrrev_b32 v5, 4, v3 // (op 5331) 

v_lshl_add_u32 v3, v5, 6, v6 // (op 5331) 


v_lshrrev_b32 v5, 6, v1 // (op 5407) 
v_and_b32 v6, 1, v5 // (op 5407) 

v_and_b32 v5, 15, v1 // (op 5407) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 5407) 


v_and_b32 v5, 63, v1 // (op 5407) 
v_lshrrev_b32 v6, 4, v5 // (op 5407) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 5407) 


v_and_b32 v6, 48, v1 // (op 5534) 
v_and_b32 v7, 63, v1 // (op 5534) 
v_sub_u32 v8, v7, v6 // (op 5534) 

v_lshrrev_b32 v7, 1, v8 // (op 5534) 
v_add_lshl_u32 v9, v6, v7, 6 // (op 5534) 


v_and_b32 v6, 1, v8 // (op 5534) 

v_lshlrev_b32 v7, 3, v6 // (op 5534) 

v_lshrrev_b32 v6, 7, v1 // (op 5534) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 5534) 


v_add_u32 v6, v9, v8 // (op 5534) 


v_lshrrev_b32 v7, 6, v1 // (op 5609) 
v_and_b32 v8, 1, v7 // (op 5609) 

v_and_b32 v7, 15, v1 // (op 5609) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 5609) 


v_and_b32 v7, 48, v1 // (op 5609) 
v_lshl_add_u32 v8, v9, 7, v7 // (op 5609) 


v_lshrrev_b32 v7, 7, v1 // (op 5685) 
v_and_b32 v9, 15, v1 // (op 5685) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 5685) 


v_and_b32 v7, 63, v1 // (op 5685) 
v_lshrrev_b32 v9, 4, v7 // (op 5685) 

v_lshl_add_u32 v7, v9, 6, v10 // (op 5685) 


v_lshrrev_b32 v9, 6, v1 // (op 5761) 
v_and_b32 v10, 1, v9 // (op 5761) 

v_and_b32 v9, 15, v1 // (op 5761) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 5761) 


v_and_b32 v9, 63, v1 // (op 5761) 
v_lshrrev_b32 v10, 4, v9 // (op 5761) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 5761) 


v_and_b32 v10, 48, v1 // (op 5875) 
v_and_b32 v11, 63, v1 // (op 5875) 
v_sub_u32 v12, v11, v10 // (op 5875) 

v_lshrrev_b32 v11, 1, v12 // (op 5875) 
v_add_lshl_u32 v13, v10, v11, 6 // (op 5875) 


v_and_b32 v10, 1, v12 // (op 5875) 

v_lshlrev_b32 v11, 3, v10 // (op 5875) 

v_lshrrev_b32 v10, 7, v1 // (op 5875) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 5875) 


v_add_u32 v10, v13, v12 // (op 5875) 


v_lshrrev_b32 v11, 6, v1 // (op 5919) 
v_and_b32 v12, 1, v11 // (op 5919) 

v_and_b32 v11, 15, v1 // (op 5919) 
v_lshl_add_u32 v13, v12, 4, v11 // (op 5919) 


v_and_b32 v11, 48, v1 // (op 5919) 
v_lshl_add_u32 v12, v13, 7, v11 // (op 5919) 


v_lshrrev_b32 v11, 7, v1 // (op 5948) 
v_and_b32 v13, 15, v1 // (op 5948) 
v_lshl_add_u32 v14, v11, 4, v13 // (op 5948) 


v_and_b32 v11, 63, v1 // (op 5948) 
v_lshrrev_b32 v13, 4, v11 // (op 5948) 

v_lshl_add_u32 v11, v13, 6, v14 // (op 5948) 


v_lshrrev_b32 v13, 6, v1 // (op 5977) 
v_and_b32 v14, 1, v13 // (op 5977) 

v_and_b32 v13, 15, v1 // (op 5977) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 5977) 


v_and_b32 v13, 63, v1 // (op 5977) 
v_lshrrev_b32 v14, 4, v13 // (op 5977) 

v_lshl_add_u32 v13, v15, 2, v14 // (op 5977) 


v_and_b32 v14, 48, v1 // (op 6021) 
v_and_b32 v15, 63, v1 // (op 6021) 
v_sub_u32 v16, v15, v14 // (op 6021) 

v_lshrrev_b32 v15, 1, v16 // (op 6021) 
v_add_lshl_u32 v17, v14, v15, 6 // (op 6021) 


v_and_b32 v14, 1, v16 // (op 6021) 

v_lshlrev_b32 v15, 3, v14 // (op 6021) 

v_lshrrev_b32 v14, 7, v1 // (op 6021) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 6021) 


v_add_u32 v14, v17, v16 // (op 6021) 


v_lshrrev_b32 v15, 6, v1 // (op 6065) 
v_and_b32 v16, 1, v15 // (op 6065) 

v_and_b32 v15, 15, v1 // (op 6065) 
v_lshl_add_u32 v17, v16, 4, v15 // (op 6065) 


v_and_b32 v15, 48, v1 // (op 6065) 
v_lshl_add_u32 v16, v17, 7, v15 // (op 6065) 


v_lshrrev_b32 v15, 7, v1 // (op 6094) 
v_and_b32 v17, 15, v1 // (op 6094) 
v_lshl_add_u32 v18, v15, 4, v17 // (op 6094) 


v_and_b32 v15, 63, v1 // (op 6094) 
v_lshrrev_b32 v17, 4, v15 // (op 6094) 

v_lshl_add_u32 v15, v17, 6, v18 // (op 6094) 


v_lshrrev_b32 v17, 6, v1 // (op 6123) 
v_and_b32 v18, 1, v17 // (op 6123) 

v_and_b32 v17, 15, v1 // (op 6123) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 6123) 


v_and_b32 v17, 63, v1 // (op 6123) 
v_lshrrev_b32 v18, 4, v17 // (op 6123) 

v_lshl_add_u32 v17, v19, 2, v18 // (op 6123) 


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
s_cbranch_scc0 RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_294 // (op 294) Condition: Top (jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_294 if false)
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_SKStreamTileLoop_294:
v_lshrrev_b32 v20, 7, v1 // (op 5091) 
s_mul_i32 s52, s16, s8 // (op 5091) 
s_add_u32 s52, s52, s0 // (op 5091) 
v_and_b32 v21, 63, v1 // (op 5091) 
s_mul_hi_u32 s53, s52, s13 // (op 5091) 
v_lshrrev_b32 v22, 4, v21 // (op 5091) 
s_sub_u32 s54, s52, s53 // (op 5091) 
v_lshlrev_b32 v23, 2, v22 // (op 5091) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 5091) 


s_lshr_b32 s55, s54, 1 // (op 5091) 

s_add_u32 s54, s55, s53 // (op 5091) 


v_lshrrev_b32 v20, 2, v21 // (op 5091) 

s_lshr_b32 s53, s54, s11 // (op 5091) 

v_and_b32 v21, 3, v20 // (op 5091) 

s_and_b32 s54, s11, 2147483648 // (op 5091) 
v_and_b32 v20, 3, v1 // (op 5091) 
s_cmp_eq_i32 s54, 0 // (op 5091) 
s_mov_b32 s55, scc // (op 5091) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 5091) 


s_cmp_eq_i32 s55, 0 // (op 5091) 
s_cselect_b32 s54, s52, s53 // (op 5091) 



v_lshrrev_b32 v20, 6, v1 // (op 5091) 
s_mul_hi_u32 s52, s54, s12 // (op 5091) 
v_and_b32 v21, 1, v20 // (op 5091) 

s_sub_u32 s53, s54, s52 // (op 5091) 
v_lshl_add_u32 v20, v21, 4, v23 // (op 5091) 


s_lshr_b32 s55, s53, 1 // (op 5091) 

s_add_u32 s53, s55, s52 // (op 5091) 


s_lshr_b32 s52, s53, s10 // (op 5091) 

s_and_b32 s53, s10, 2147483648 // (op 5091) 
s_cmp_eq_i32 s53, 0 // (op 5091) 
s_mov_b32 s55, scc // (op 5091) 

s_cmp_eq_i32 s55, 0 // (op 5091) 
s_cselect_b32 s53, s54, s52 // (op 5091) 


s_mul_i32 s52, s53, s14 // (op 5091) 
v_lshl_add_u32 v21, s53, 6, v22 // (op 5091) 


s_sub_u32 s53, s54, s52 // (op 5091) 


v_lshl_add_u32 v22, s53, 6, v20 // (op 5091) 


v_mul_lo_u32 v20, v22, s4 // (op 5091) 

v_add_lshl_u32 v22, v21, v20, 1 // (op 5091) 


s_mul_i32 s53, s4, s23 // (op 5093) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 5093) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 5093) least significant
s_mul_hi_u32 s55, s4, s22 // (op 5093) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 5093) most significant: sum
s_add_u32 s53, s53, s55 // (op 5093) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 5093) 

s_mov_b32 s56, s20 // (op 5093) reg expression
s_mov_b32 s57, s21 // (op 5093) reg expression
s_mov_b32 s58, s54 // (op 5093) reg expression
s_mov_b32 s59, 131072 // (op 5093) call()

v_and_b32 v20, 48, v1 // (op 5197) 
v_and_b32 v21, 63, v1 // (op 5197) 
v_sub_u32 v23, v21, v20 // (op 5197) 

v_lshrrev_b32 v21, 1, v23 // (op 5197) 
v_add_lshl_u32 v24, v20, v21, 6 // (op 5197) 


v_and_b32 v20, 1, v23 // (op 5197) 

v_lshlrev_b32 v21, 3, v20 // (op 5197) 

v_lshrrev_b32 v20, 7, v1 // (op 5197) 
v_lshl_add_u32 v23, v20, 4, v21 // (op 5197) 


v_add_u32 v20, v24, v23 // (op 5197) 


v_lshrrev_b32 v21, 6, v1 // (op 5272) 
v_and_b32 v23, 1, v21 // (op 5272) 

v_and_b32 v21, 15, v1 // (op 5272) 
v_lshl_add_u32 v24, v23, 4, v21 // (op 5272) 


v_and_b32 v21, 48, v1 // (op 5272) 
v_lshl_add_u32 v23, v24, 7, v21 // (op 5272) 


v_lshrrev_b32 v21, 7, v1 // (op 5348) 
v_and_b32 v24, 15, v1 // (op 5348) 
v_lshl_add_u32 v25, v21, 4, v24 // (op 5348) 


v_and_b32 v21, 63, v1 // (op 5348) 
v_lshrrev_b32 v24, 4, v21 // (op 5348) 

v_lshl_add_u32 v21, v24, 6, v25 // (op 5348) 


v_lshrrev_b32 v24, 6, v1 // (op 5424) 
v_and_b32 v25, 1, v24 // (op 5424) 

v_and_b32 v24, 15, v1 // (op 5424) 
v_lshl_add_u32 v26, v25, 4, v24 // (op 5424) 


v_and_b32 v24, 63, v1 // (op 5424) 
v_lshrrev_b32 v25, 4, v24 // (op 5424) 

v_lshl_add_u32 v24, v26, 2, v25 // (op 5424) 


v_lshrrev_b32 v25, 7, v1 // (op 5447) 
s_mul_i32 s52, s16, s8 // (op 5447) 
s_add_u32 s52, s52, s0 // (op 5447) 
v_and_b32 v26, 63, v1 // (op 5447) 
s_mul_hi_u32 s53, s52, s13 // (op 5447) 
v_lshrrev_b32 v27, 4, v26 // (op 5447) 
s_sub_u32 s54, s52, s53 // (op 5447) 
v_lshlrev_b32 v28, 2, v27 // (op 5447) 

s_lshr_b32 s55, s54, 1 // (op 5447) 

v_lshl_add_u32 v27, v25, 4, v28 // (op 5447) 


s_add_u32 s54, s55, s53 // (op 5447) 


v_lshrrev_b32 v25, 2, v26 // (op 5447) 

s_lshr_b32 s53, s54, s11 // (op 5447) 

v_and_b32 v26, 3, v25 // (op 5447) 

s_and_b32 s54, s11, 2147483648 // (op 5447) 
v_and_b32 v25, 3, v1 // (op 5447) 
s_cmp_eq_i32 s54, 0 // (op 5447) 
s_mov_b32 s55, scc // (op 5447) 

v_lshl_add_u32 v28, v26, 2, v25 // (op 5447) 


s_cmp_eq_i32 s55, 0 // (op 5447) 
s_cselect_b32 s54, s52, s53 // (op 5447) 



v_lshrrev_b32 v25, 6, v1 // (op 5447) 
s_mul_hi_u32 s55, s54, s12 // (op 5447) 
v_and_b32 v26, 1, v25 // (op 5447) 

s_sub_u32 s52, s54, s55 // (op 5447) 
v_lshl_add_u32 v25, v26, 4, v28 // (op 5447) 


s_lshr_b32 s53, s52, 1 // (op 5447) 

s_add_u32 s52, s53, s55 // (op 5447) 


s_lshr_b32 s53, s52, s10 // (op 5447) 

s_and_b32 s52, s10, 2147483648 // (op 5447) 
s_cmp_eq_i32 s52, 0 // (op 5447) 
s_mov_b32 s55, scc // (op 5447) 

s_cmp_eq_i32 s55, 0 // (op 5447) 
s_cselect_b32 s52, s54, s53 // (op 5447) 


s_mul_i32 s53, s52, s14 // (op 5447) 
v_lshl_add_u32 v26, s52, 6, v27 // (op 5447) 


s_sub_u32 s52, s54, s53 // (op 5447) 


v_lshl_add_u32 v27, s52, 6, v25 // (op 5447) 


v_mul_lo_u32 v25, v27, s24 // (op 5447) 

v_add_lshl_u32 v27, v26, v25, 1 // (op 5447) 


s_mul_i32 s53, s24, s23 // (op 5449) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 5449) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 5449) least significant
s_mul_hi_u32 s55, s24, s22 // (op 5449) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 5449) most significant: sum
s_add_u32 s53, s53, s55 // (op 5449) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 5449) 

s_mov_b32 s60, s26 // (op 5449) reg expression
s_mov_b32 s61, s27 // (op 5449) reg expression
s_mov_b32 s62, s54 // (op 5449) reg expression
s_mov_b32 s63, 131072 // (op 5449) call()

v_lshrrev_b32 v25, 5, v1 // (op 5801) 
v_lshlrev_b32 v26, 2, v25 // (op 5801) 

v_lshl_add_u32 v25, s16, 6, v26 // (op 5801) 

v_and_b32 v26, -32, v1 // (op 5801) 
v_sub_u32 v28, v1, v26 // (op 5801) 

v_lshl_add_u32 v26, v25, 6, v28 // (op 5801) 


v_lshlrev_b32 v25, 2, v26 // (op 5801) 

s_lshl_b32 s52, s15, 14 // (op 5803) 
s_mov_b32 s64, s28 // (op 5803) reg expression
s_mov_b32 s65, s29 // (op 5803) reg expression
s_mov_b32 s66, s52 // (op 5803) reg expression
s_mov_b32 s67, 131072 // (op 5803) call()

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
v_accvgpr_write a4, 0.00000 // (op 1650) call()
v_accvgpr_write a5, 0.00000 // (op 1650) call()
v_accvgpr_write a6, 0.00000 // (op 1650) call()
v_accvgpr_write a7, 0.00000 // (op 1650) call()
v_accvgpr_write a8, 0.00000 // (op 1870) call()
v_accvgpr_write a9, 0.00000 // (op 1870) call()
v_accvgpr_write a10, 0.00000 // (op 1870) call()
v_accvgpr_write a11, 0.00000 // (op 1870) call()
v_accvgpr_write a12, 0.00000 // (op 2025) call()
v_accvgpr_write a13, 0.00000 // (op 2025) call()
v_accvgpr_write a14, 0.00000 // (op 2025) call()
v_accvgpr_write a15, 0.00000 // (op 2025) call()
v_and_b32 v26, -4, v1 // (op 4658) 
s_mul_i32 s52, s16, s8 // (op 4658) 
s_add_u32 s52, s52, s0 // (op 4658) 
v_sub_u32 v28, v1, v26 // (op 4658) 

s_mul_hi_u32 s53, s52, s13 // (op 4658) 
v_lshlrev_b32 v26, 4, v28 // (op 4658) 

s_sub_u32 s54, s52, s53 // (op 4658) 
v_lshrrev_b32 v28, 2, v1 // (op 4658) 
s_lshr_b32 s55, s54, 1 // (op 4658) 

s_add_u32 s54, s55, s53 // (op 4658) 


s_lshr_b32 s53, s54, s11 // (op 4658) 

s_and_b32 s54, s11, 2147483648 // (op 4658) 
s_cmp_eq_i32 s54, 0 // (op 4658) 
s_mov_b32 s55, scc // (op 4658) 

s_cmp_eq_i32 s55, 0 // (op 4658) 
s_cselect_b32 s54, s52, s53 // (op 4658) 


s_mul_hi_u32 s53, s54, s12 // (op 4658) 
s_sub_u32 s55, s54, s53 // (op 4658) 
s_lshr_b32 s68, s55, 1 // (op 4658) 

s_add_u32 s55, s68, s53 // (op 4658) 


s_lshr_b32 s53, s55, s10 // (op 4658) 

s_and_b32 s55, s10, 2147483648 // (op 4658) 
s_cmp_eq_i32 s55, 0 // (op 4658) 
s_mov_b32 s68, scc // (op 4658) 

s_cmp_eq_i32 s68, 0 // (op 4658) 
s_cselect_b32 s55, s54, s53 // (op 4658) 


v_lshl_add_u32 v29, s55, 6, v26 // (op 4658) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 4658) 
v_mov_b32 v31, 0 // (op 4658) convert
v_mov_b32 v30, v29 // (op 4658) convert

s_mul_i32 s53, s54, s68 // (op 4658) 


s_sub_u32 s54, s52, s53 // (op 4658) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 4658) 


v_mul_lo_u32 v29, v26, s31 // (op 4658) most significant: low of low * high
v_mul_hi_u32 v32, v26, s30 // (op 4658) most significant: high of low * low
v_mul_lo_u32 v28, v26, s30 // (op 4658) least significant: low of low * low
v_add_u32 v29, v29, v32 // (op 4658) most significant: sum


v_add_co_u32 v32, s[52:53], v30, v28 // (op 4658) least significant half
v_addc_co_u32 v33, s[52:53], v31, v29, s[52:53] // (op 4658) most significant half



s_mul_i32 s53, s30, s3 // (op 4660) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 4660) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 4660) least significant
s_mul_hi_u32 s55, s30, s2 // (op 4660) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4660) most significant: sum
s_add_u32 s53, s53, s55 // (op 4660) most significant: sum


s_mov_b32 s68, s32 // (op 4660) reg expression
s_mov_b32 s69, s33 // (op 4660) reg expression
s_mov_b32 s70, s52 // (op 4660) reg expression
s_mov_b32 s71, 131072 // (op 4660) call()

v_and_b32 v26, -4, v1 // (op 4677) 
s_mul_i32 s52, s16, s8 // (op 4677) 
s_add_u32 s52, s52, s0 // (op 4677) 
v_sub_u32 v28, v1, v26 // (op 4677) 

s_mul_hi_u32 s53, s52, s13 // (op 4677) 
v_lshlrev_b32 v26, 4, v28 // (op 4677) 

s_sub_u32 s54, s52, s53 // (op 4677) 
v_lshrrev_b32 v28, 2, v1 // (op 4677) 
s_lshr_b32 s55, s54, 1 // (op 4677) 

s_add_u32 s54, s55, s53 // (op 4677) 


s_lshr_b32 s53, s54, s11 // (op 4677) 

s_and_b32 s54, s11, 2147483648 // (op 4677) 
s_cmp_eq_i32 s54, 0 // (op 4677) 
s_mov_b32 s55, scc // (op 4677) 

s_cmp_eq_i32 s55, 0 // (op 4677) 
s_cselect_b32 s54, s52, s53 // (op 4677) 


s_mul_hi_u32 s53, s54, s12 // (op 4677) 
s_sub_u32 s55, s54, s53 // (op 4677) 
s_lshr_b32 s72, s55, 1 // (op 4677) 

s_add_u32 s55, s72, s53 // (op 4677) 


s_lshr_b32 s53, s55, s10 // (op 4677) 

s_and_b32 s55, s10, 2147483648 // (op 4677) 
s_cmp_eq_i32 s55, 0 // (op 4677) 
s_mov_b32 s72, scc // (op 4677) 

s_cmp_eq_i32 s72, 0 // (op 4677) 
s_cselect_b32 s55, s54, s53 // (op 4677) 


v_lshl_add_u32 v29, s55, 6, v26 // (op 4677) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 4677) 
v_mov_b32 v31, 0 // (op 4677) convert
v_mov_b32 v30, v29 // (op 4677) convert

s_mul_i32 s53, s54, s72 // (op 4677) 


s_sub_u32 s54, s52, s53 // (op 4677) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 4677) 


v_mul_lo_u32 v29, v26, s31 // (op 4677) most significant: low of low * high
v_mul_hi_u32 v34, v26, s30 // (op 4677) most significant: high of low * low
v_mul_lo_u32 v28, v26, s30 // (op 4677) least significant: low of low * low
v_add_u32 v29, v29, v34 // (op 4677) most significant: sum


v_add_co_u32 v34, s[52:53], v30, v28 // (op 4677) least significant half
v_addc_co_u32 v35, s[52:53], v31, v29, s[52:53] // (op 4677) most significant half



s_mul_i32 s53, s30, s3 // (op 4679) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 4679) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 4679) least significant
s_mul_hi_u32 s55, s30, s2 // (op 4679) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4679) most significant: sum
s_add_u32 s53, s53, s55 // (op 4679) most significant: sum


s_mov_b32 s68, s32 // (op 4679) reg expression
s_mov_b32 s69, s33 // (op 4679) reg expression
s_mov_b32 s70, s52 // (op 4679) reg expression
s_mov_b32 s71, 131072 // (op 4679) call()

v_lshrrev_b32 v26, 3, v1 // (op 4786) 
s_mul_i32 s52, s16, s8 // (op 4786) 
s_add_u32 s52, s52, s0 // (op 4786) 
v_and_b32 v28, -8, v1 // (op 4786) 
s_mul_hi_u32 s53, s52, s13 // (op 4786) 
v_sub_u32 v29, v1, v28 // (op 4786) 

s_sub_u32 s54, s52, s53 // (op 4786) 
v_lshlrev_b32 v28, 4, v29 // (op 4786) 

s_lshr_b32 s55, s54, 1 // (op 4786) 

s_add_u32 s54, s55, s53 // (op 4786) 


s_lshr_b32 s53, s54, s11 // (op 4786) 

s_and_b32 s54, s11, 2147483648 // (op 4786) 
s_cmp_eq_i32 s54, 0 // (op 4786) 
s_mov_b32 s55, scc // (op 4786) 

s_cmp_eq_i32 s55, 0 // (op 4786) 
s_cselect_b32 s54, s52, s53 // (op 4786) 


s_mul_hi_u32 s53, s54, s12 // (op 4786) 
s_sub_u32 s55, s54, s53 // (op 4786) 
s_lshr_b32 s72, s55, 1 // (op 4786) 

s_add_u32 s55, s72, s53 // (op 4786) 


s_lshr_b32 s53, s55, s10 // (op 4786) 

s_and_b32 s55, s10, 2147483648 // (op 4786) 
s_cmp_eq_i32 s55, 0 // (op 4786) 
s_mov_b32 s72, scc // (op 4786) 

s_cmp_eq_i32 s72, 0 // (op 4786) 
s_cselect_b32 s55, s54, s53 // (op 4786) 


s_mul_i32 s53, s55, s14 // (op 4786) 

s_sub_u32 s55, s54, s53 // (op 4786) 

v_lshl_add_u32 v29, s55, 6, v26 // (op 4786) 


v_mul_lo_u32 v31, v29, s35 // (op 4786) most significant: low of low * high
v_mul_hi_u32 v26, v29, s34 // (op 4786) most significant: high of low * low
v_mul_lo_u32 v30, v29, s34 // (op 4786) least significant: low of low * low
v_add_u32 v31, v31, v26 // (op 4786) most significant: sum


s_ashr_i64 s[72:73], s[2:3], 7 // (op 4786) 
s_mul_i32 s53, s54, s72 // (op 4786) 


s_sub_u32 s54, s52, s53 // (op 4786) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 4786) 


v_mov_b32 v29, 0 // (op 4786) convert
v_mov_b32 v28, v26 // (op 4786) convert

v_add_co_u32 v36, s[52:53], v28, v30 // (op 4786) least significant half
v_addc_co_u32 v37, s[52:53], v29, v31, s[52:53] // (op 4786) most significant half



s_mul_i32 s53, s34, s23 // (op 4788) most significant: low of low * high
s_mul_i32 s54, s35, s22 // (op 4788) most significant: low of high * low
s_mul_i32 s52, s34, s22 // (op 4788) least significant
s_mul_hi_u32 s55, s34, s22 // (op 4788) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4788) most significant: sum
s_add_u32 s53, s53, s55 // (op 4788) most significant: sum


s_mov_b32 s72, s36 // (op 4788) reg expression
s_mov_b32 s73, s37 // (op 4788) reg expression
s_mov_b32 s74, s52 // (op 4788) reg expression
s_mov_b32 s75, 131072 // (op 4788) call()

v_lshrrev_b32 v26, 3, v1 // (op 4805) 
s_mul_i32 s52, s16, s8 // (op 4805) 
s_add_u32 s52, s52, s0 // (op 4805) 
v_and_b32 v28, -8, v1 // (op 4805) 
s_mul_hi_u32 s53, s52, s13 // (op 4805) 
v_sub_u32 v29, v1, v28 // (op 4805) 

s_sub_u32 s54, s52, s53 // (op 4805) 
v_lshlrev_b32 v28, 4, v29 // (op 4805) 

s_lshr_b32 s55, s54, 1 // (op 4805) 

s_add_u32 s54, s55, s53 // (op 4805) 


s_lshr_b32 s53, s54, s11 // (op 4805) 

s_and_b32 s54, s11, 2147483648 // (op 4805) 
s_cmp_eq_i32 s54, 0 // (op 4805) 
s_mov_b32 s55, scc // (op 4805) 

s_cmp_eq_i32 s55, 0 // (op 4805) 
s_cselect_b32 s54, s52, s53 // (op 4805) 


s_mul_hi_u32 s53, s54, s12 // (op 4805) 
s_sub_u32 s55, s54, s53 // (op 4805) 
s_lshr_b32 s76, s55, 1 // (op 4805) 

s_add_u32 s55, s76, s53 // (op 4805) 


s_lshr_b32 s53, s55, s10 // (op 4805) 

s_and_b32 s55, s10, 2147483648 // (op 4805) 
s_cmp_eq_i32 s55, 0 // (op 4805) 
s_mov_b32 s76, scc // (op 4805) 

s_cmp_eq_i32 s76, 0 // (op 4805) 
s_cselect_b32 s55, s54, s53 // (op 4805) 


s_mul_i32 s53, s55, s14 // (op 4805) 

s_sub_u32 s55, s54, s53 // (op 4805) 

v_lshl_add_u32 v29, s55, 6, v26 // (op 4805) 


v_mul_lo_u32 v31, v29, s35 // (op 4805) most significant: low of low * high
v_mul_hi_u32 v26, v29, s34 // (op 4805) most significant: high of low * low
v_mul_lo_u32 v30, v29, s34 // (op 4805) least significant: low of low * low
v_add_u32 v31, v31, v26 // (op 4805) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 4805) 
s_mul_i32 s53, s54, s76 // (op 4805) 


s_sub_u32 s54, s52, s53 // (op 4805) 


v_lshl_add_u32 v26, s54, 7, v28 // (op 4805) 


v_mov_b32 v29, 0 // (op 4805) convert
v_mov_b32 v28, v26 // (op 4805) convert

v_add_co_u32 v38, s[52:53], v28, v30 // (op 4805) least significant half
v_addc_co_u32 v39, s[52:53], v29, v31, s[52:53] // (op 4805) most significant half



s_mul_i32 s53, s34, s23 // (op 4807) most significant: low of low * high
s_mul_i32 s54, s35, s22 // (op 4807) most significant: low of high * low
s_mul_i32 s52, s34, s22 // (op 4807) least significant
s_mul_hi_u32 s55, s34, s22 // (op 4807) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4807) most significant: sum
s_add_u32 s53, s53, s55 // (op 4807) most significant: sum


s_mov_b32 s72, s36 // (op 4807) reg expression
s_mov_b32 s73, s37 // (op 4807) reg expression
s_mov_b32 s74, s52 // (op 4807) reg expression
s_mov_b32 s75, 131072 // (op 4807) call()

v_and_b32 v26, -64, v1 // (op 4914) 
s_mul_i32 s52, s16, s8 // (op 4914) 
s_add_u32 s52, s52, s0 // (op 4914) 
v_sub_u32 v28, v1, v26 // (op 4914) 

s_mul_hi_u32 s53, s52, s13 // (op 4914) 
v_lshrrev_b32 v26, 6, v1 // (op 4914) 
s_sub_u32 s54, s52, s53 // (op 4914) 
s_lshr_b32 s55, s54, 1 // (op 4914) 

s_add_u32 s54, s55, s53 // (op 4914) 


s_lshr_b32 s53, s54, s11 // (op 4914) 

s_and_b32 s54, s11, 2147483648 // (op 4914) 
s_cmp_eq_i32 s54, 0 // (op 4914) 
s_mov_b32 s55, scc // (op 4914) 

s_cmp_eq_i32 s55, 0 // (op 4914) 
s_cselect_b32 s54, s52, s53 // (op 4914) 


s_mul_hi_u32 s53, s54, s12 // (op 4914) 
s_sub_u32 s55, s54, s53 // (op 4914) 
s_lshr_b32 s76, s55, 1 // (op 4914) 

s_add_u32 s55, s76, s53 // (op 4914) 


s_lshr_b32 s53, s55, s10 // (op 4914) 

s_and_b32 s55, s10, 2147483648 // (op 4914) 
s_cmp_eq_i32 s55, 0 // (op 4914) 
s_mov_b32 s76, scc // (op 4914) 

s_cmp_eq_i32 s76, 0 // (op 4914) 
s_cselect_b32 s55, s54, s53 // (op 4914) 


v_lshl_add_u32 v29, s55, 6, v28 // (op 4914) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 4914) 
s_mul_i32 s53, s54, s76 // (op 4914) 


s_sub_u32 s54, s52, s53 // (op 4914) 


v_lshl_add_u32 v28, s54, 2, v26 // (op 4914) 


v_mul_lo_u32 v26, v28, s38 // (op 4914) 

v_add_u32 v28, v29, v26 // (op 4914) 


s_mul_i32 s53, s38, s43 // (op 4916) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 4916) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 4916) least significant
s_mul_hi_u32 s55, s38, s42 // (op 4916) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4916) most significant: sum
s_add_u32 s53, s53, s55 // (op 4916) most significant: sum


s_mov_b32 s76, s40 // (op 4916) reg expression
s_mov_b32 s77, s41 // (op 4916) reg expression
s_mov_b32 s78, s52 // (op 4916) reg expression
s_mov_b32 s79, 131072 // (op 4916) call()

v_and_b32 v26, -64, v1 // (op 4933) 
s_mul_i32 s52, s16, s8 // (op 4933) 
s_add_u32 s52, s52, s0 // (op 4933) 
v_sub_u32 v29, v1, v26 // (op 4933) 

s_mul_hi_u32 s53, s52, s13 // (op 4933) 
v_lshrrev_b32 v26, 6, v1 // (op 4933) 
s_sub_u32 s54, s52, s53 // (op 4933) 
s_lshr_b32 s55, s54, 1 // (op 4933) 

s_add_u32 s54, s55, s53 // (op 4933) 


s_lshr_b32 s53, s54, s11 // (op 4933) 

s_and_b32 s54, s11, 2147483648 // (op 4933) 
s_cmp_eq_i32 s54, 0 // (op 4933) 
s_mov_b32 s55, scc // (op 4933) 

s_cmp_eq_i32 s55, 0 // (op 4933) 
s_cselect_b32 s54, s52, s53 // (op 4933) 


s_mul_hi_u32 s53, s54, s12 // (op 4933) 
s_sub_u32 s55, s54, s53 // (op 4933) 
s_lshr_b32 s80, s55, 1 // (op 4933) 

s_add_u32 s55, s80, s53 // (op 4933) 


s_lshr_b32 s53, s55, s10 // (op 4933) 

s_and_b32 s55, s10, 2147483648 // (op 4933) 
s_cmp_eq_i32 s55, 0 // (op 4933) 
s_mov_b32 s80, scc // (op 4933) 

s_cmp_eq_i32 s80, 0 // (op 4933) 
s_cselect_b32 s55, s54, s53 // (op 4933) 


v_lshl_add_u32 v30, s55, 6, v29 // (op 4933) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 4933) 
s_mul_i32 s53, s54, s80 // (op 4933) 


s_sub_u32 s54, s52, s53 // (op 4933) 


v_lshl_add_u32 v29, s54, 2, v26 // (op 4933) 


v_mul_lo_u32 v26, v29, s38 // (op 4933) 

v_add_u32 v29, v30, v26 // (op 4933) 


s_mul_i32 s53, s38, s43 // (op 4935) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 4935) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 4935) least significant
s_mul_hi_u32 s55, s38, s42 // (op 4935) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4935) most significant: sum
s_add_u32 s53, s53, s55 // (op 4935) most significant: sum


s_mov_b32 s76, s40 // (op 4935) reg expression
s_mov_b32 s77, s41 // (op 4935) reg expression
s_mov_b32 s78, s52 // (op 4935) reg expression
s_mov_b32 s79, 131072 // (op 4935) call()

v_lshrrev_b32 v26, 2, v1 // (op 5042) 
s_mul_i32 s52, s16, s8 // (op 5042) 
s_add_u32 s52, s52, s0 // (op 5042) 
v_and_b32 v30, -4, v1 // (op 5042) 
s_mul_hi_u32 s53, s52, s13 // (op 5042) 
v_sub_u32 v31, v1, v30 // (op 5042) 

s_sub_u32 s54, s52, s53 // (op 5042) 
s_lshr_b32 s55, s54, 1 // (op 5042) 

s_add_u32 s54, s55, s53 // (op 5042) 


s_lshr_b32 s53, s54, s11 // (op 5042) 

s_and_b32 s54, s11, 2147483648 // (op 5042) 
s_cmp_eq_i32 s54, 0 // (op 5042) 
s_mov_b32 s55, scc // (op 5042) 

s_cmp_eq_i32 s55, 0 // (op 5042) 
s_cselect_b32 s54, s52, s53 // (op 5042) 


s_mul_hi_u32 s53, s54, s12 // (op 5042) 
s_sub_u32 s55, s54, s53 // (op 5042) 
s_lshr_b32 s80, s55, 1 // (op 5042) 

s_add_u32 s55, s80, s53 // (op 5042) 


s_lshr_b32 s53, s55, s10 // (op 5042) 

s_and_b32 s55, s10, 2147483648 // (op 5042) 
s_cmp_eq_i32 s55, 0 // (op 5042) 
s_mov_b32 s80, scc // (op 5042) 

s_cmp_eq_i32 s80, 0 // (op 5042) 
s_cselect_b32 s55, s54, s53 // (op 5042) 


s_mul_i32 s53, s55, s14 // (op 5042) 

s_sub_u32 s55, s54, s53 // (op 5042) 

v_lshl_add_u32 v30, s55, 6, v26 // (op 5042) 


v_mul_lo_u32 v26, v30, s44 // (op 5042) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 5042) 
s_mul_i32 s53, s54, s80 // (op 5042) 


s_sub_u32 s54, s52, s53 // (op 5042) 


v_lshl_add_u32 v30, s54, 2, v31 // (op 5042) 


v_add_u32 v31, v30, v26 // (op 5042) 


s_mul_i32 s53, s44, s23 // (op 5044) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 5044) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 5044) least significant
s_mul_hi_u32 s55, s44, s22 // (op 5044) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 5044) most significant: sum
s_add_u32 s53, s53, s55 // (op 5044) most significant: sum


s_mov_b32 s80, s46 // (op 5044) reg expression
s_mov_b32 s81, s47 // (op 5044) reg expression
s_mov_b32 s82, s52 // (op 5044) reg expression
s_mov_b32 s83, 131072 // (op 5044) call()

v_lshrrev_b32 v26, 2, v1 // (op 5061) 
s_mul_i32 s52, s16, s8 // (op 5061) 
s_add_u32 s52, s52, s0 // (op 5061) 
v_and_b32 v30, -4, v1 // (op 5061) 
s_mul_hi_u32 s53, s52, s13 // (op 5061) 
v_sub_u32 v40, v1, v30 // (op 5061) 

s_sub_u32 s54, s52, s53 // (op 5061) 
s_lshr_b32 s55, s54, 1 // (op 5061) 

s_add_u32 s54, s55, s53 // (op 5061) 


s_lshr_b32 s53, s54, s11 // (op 5061) 

s_and_b32 s54, s11, 2147483648 // (op 5061) 
s_cmp_eq_i32 s54, 0 // (op 5061) 
s_mov_b32 s55, scc // (op 5061) 

s_cmp_eq_i32 s55, 0 // (op 5061) 
s_cselect_b32 s54, s52, s53 // (op 5061) 


s_mul_hi_u32 s53, s54, s12 // (op 5061) 
s_sub_u32 s55, s54, s53 // (op 5061) 
s_lshr_b32 s84, s55, 1 // (op 5061) 

s_add_u32 s55, s84, s53 // (op 5061) 


s_lshr_b32 s53, s55, s10 // (op 5061) 

s_and_b32 s55, s10, 2147483648 // (op 5061) 
s_cmp_eq_i32 s55, 0 // (op 5061) 
s_mov_b32 s84, scc // (op 5061) 

s_cmp_eq_i32 s84, 0 // (op 5061) 
s_cselect_b32 s55, s54, s53 // (op 5061) 


s_mul_i32 s53, s55, s14 // (op 5061) 

s_sub_u32 s55, s54, s53 // (op 5061) 

v_lshl_add_u32 v30, s55, 6, v26 // (op 5061) 


v_mul_lo_u32 v26, v30, s44 // (op 5061) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 5061) 
s_mul_i32 s53, s54, s84 // (op 5061) 


s_sub_u32 s54, s52, s53 // (op 5061) 


v_lshl_add_u32 v30, s54, 2, v40 // (op 5061) 


v_add_u32 v40, v30, v26 // (op 5061) 


s_mul_i32 s53, s44, s23 // (op 5063) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 5063) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 5063) least significant
s_mul_hi_u32 s55, s44, s22 // (op 5063) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 5063) most significant: sum
s_add_u32 s53, s53, s55 // (op 5063) most significant: sum


s_mov_b32 s80, s46 // (op 5063) reg expression
s_mov_b32 s81, s47 // (op 5063) reg expression
s_mov_b32 s82, s52 // (op 5063) reg expression
s_mov_b32 s83, 131072 // (op 5063) call()

v_and_b32 v26, -8, v1 // (op 5238) 
v_sub_u32 v30, v1, v26 // (op 5238) 

v_lshlrev_b32 v26, 4, v30 // (op 5238) 

v_lshrrev_b32 v30, 3, v1 // (op 5238) 
v_lshl_add_u32 v41, v30, 7, v26 // (op 5238) 


s_nop 0
v_readfirstlane_b32 s52, v41 // (op 5238) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6273) 

s_lshl_b64 s[54:55], s[34:35], 5 // (op 4532) 
v_mov_b64 v[42:43], s[54:55] // (op 4532) convert


v_mov_b64 v[44:45], v[36:37] // (op 4532) 
s_mov_b32 m0, s52 // (op 4532) reg expression
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 4532) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 4532) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 4532) most significant half

s_add_u32 m0, m0, 4096 // (op 4532) 
buffer_load_dwordx4 v44, s[72:75], 0 offen offset: 0 lds // (op 4532) Load value direct to lds






buffer_load_ubyte v26, v31, s[80:83], 0 offen offset: 0 // (op 4315) Load value
v_and_b32 v30, -4, v1 // (op 5163) 
v_sub_u32 v41, v1, v30 // (op 5163) 

v_lshlrev_b32 v30, 4, v41 // (op 5163) 

v_lshrrev_b32 v41, 2, v1 // (op 5163) 
v_lshl_add_u32 v42, v41, 6, v30 // (op 5163) 


s_nop 0
v_readfirstlane_b32 s52, v42 // (op 5163) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6254) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 4537) 
v_mov_b64 v[42:43], s[54:55] // (op 4537) convert


v_mov_b64 v[44:45], v[32:33] // (op 4537) reg expression
s_add_u32 s53, 8192, s52 // (op 4537) 
s_mov_b32 m0, s53 // (op 4537) reg expression
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 4537) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 4537) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 4537) most significant half

s_add_u32 m0, m0, 4096 // (op 4537) 
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 4537) Load value direct to lds






v_mov_b32 v30, v28 // (op 4321) reg expression

buffer_load_ubyte v41, v30, s[76:79], 0 offen offset: 0 // (op 4321) Load value

v_and_b32 v30, -8, v1 // (op 6048) 
v_sub_u32 v42, v1, v30 // (op 6048) 

v_lshlrev_b32 v30, 4, v42 // (op 6048) 

v_lshrrev_b32 v42, 3, v1 // (op 6048) 
v_lshl_add_u32 v43, v42, 7, v30 // (op 6048) 


s_nop 0
v_readfirstlane_b32 s52, v43 // (op 6048) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6456) 

s_lshl_b64 s[54:55], s[34:35], 5 // (op 4542) 
v_mov_b64 v[42:43], s[54:55] // (op 4542) convert


v_mov_b64 v[44:45], v[36:37] // (op 4542) 
s_add_u32 s53, 16384, s52 // (op 4542) 
s_mov_b32 m0, s53 // (op 4542) reg expression
s_mov_b32 s54, 128 // (op 4542) call()
buffer_load_dwordx4 v44, s[72:75], s54 offen offset: 0 lds // (op 4542) Load value direct to lds
v_add_co_u32 v44, s[84:85], v44, v42 // (op 4542) least significant half
v_addc_co_u32 v45, s[84:85], 0, v45, s[84:85] // (op 4542) most significant half

s_add_u32 m0, m0, 4096 // (op 4542) 
buffer_load_dwordx4 v44, s[72:75], s54 offen offset: 0 lds // (op 4542) Load value direct to lds









buffer_load_ubyte v30, v31, s[80:83], 0 offen offset:4 // (op 4327) Load value

v_and_b32 v31, -4, v1 // (op 6004) 
v_sub_u32 v36, v1, v31 // (op 6004) 

v_lshlrev_b32 v31, 4, v36 // (op 6004) 

v_lshrrev_b32 v36, 2, v1 // (op 6004) 
v_lshl_add_u32 v37, v36, 6, v31 // (op 6004) 


s_nop 0
v_readfirstlane_b32 s52, v37 // (op 6004) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6446) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 4544) 
v_mov_b64 v[36:37], s[54:55] // (op 4544) convert


s_lshl_b64 s[54:55], s[30:31], 7 // (op 4544) 
v_mov_b32 v31, s54 // (op 4544) 
v_mov_b32 v42, s55 // (op 4544) 
v_add_co_u32 v44, s[84:85], v32, v31 // (op 4544) least significant half
v_addc_co_u32 v45, s[84:85], v33, v42, s[84:85] // (op 4544) most significant half




s_add_u32 s53, 24576, s52 // (op 4544) 
s_mov_b32 m0, s53 // (op 4544) reg expression
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 4544) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v36 // (op 4544) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 4544) most significant half

s_add_u32 m0, m0, 4096 // (op 4544) 
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 4544) Load value direct to lds







s_lshl_b32 s52, s38, 2 // (op 4333) 
v_add_u32 v31, v28, s52 // (op 4333) 


buffer_load_ubyte v32, v31, s[76:79], 0 offen offset: 0 // (op 4333) Load value


v_and_b32 v28, -4, v1 // (op 5390) 
v_sub_u32 v31, v1, v28 // (op 5390) 

v_and_b32 v28, 4294967292, v1 // (op 5390) 
v_add_u32 v33, v28, v31 // (op 5390) 


s_barrier  // (op 6314) 



s_waitcnt vmcnt(9)
ds_write_b8 v33, v26 offset:32768 // (op 4336) Store local data 

v_and_b32 v28, -64, v1 // (op 5314) 
v_sub_u32 v31, v1, v28 // (op 5314) 

v_and_b32 v28, 4294967232, v1 // (op 5314) 
v_add_u32 v33, v28, v31 // (op 5314) 


s_barrier  // (op 6293) 



s_waitcnt vmcnt(6)
ds_write_b8 v33, v41 offset:33024 // (op 4338) Store local data 

s_barrier  // (op 6270) 



ds_read_b128 v[42:45], v23 // (op 4363) Load local data 
ds_read_b128 v[46:49], v23 offset:64 // (op 4363) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 6311) 



ds_read_u8 v28, v24 offset:32768 // (op 4383) Load local data 



ds_read_b128 v[50:53], v23 offset:4096 // (op 4368) Load local data 
ds_read_b128 v[54:57], v23 offset:4160 // (op 4368) Load local data 



ds_read_u8 v31, v24 offset:32896 // (op 4388) Load local data 
s_barrier  // (op 6251) 



ds_read_b64_tr_b8 v[58:59], v20 offset:8192 // (op 4353) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v20 offset:8704 // (op 4353) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v20 offset:12288 // (op 4353) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v20 offset:12800 // (op 4353) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 6290) 


ds_read_u8 v33, v21 offset:33024 // (op 4373) Load local data 



ds_read_b64_tr_b8 v[66:67], v20 offset:8224 // (op 4358) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v20 offset:8736 // (op 4358) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v20 offset:12320 // (op 4358) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v20 offset:12832 // (op 4358) Transpose load local data 


ds_read_u8 v36, v21 offset:33056 // (op 4378) Load local data 
v_and_b32 v37, -4, v1 // (op 5135) 
v_sub_u32 v74, v1, v37 // (op 5135) 

v_lshlrev_b32 v37, 4, v74 // (op 5135) 

v_lshrrev_b32 v74, 2, v1 // (op 5135) 
v_lshl_add_u32 v75, v74, 6, v37 // (op 5135) 


s_nop 0
v_readfirstlane_b32 s52, v75 // (op 5135) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -8, v1 // (op 5210) 
v_sub_u32 v74, v1, v37 // (op 5210) 

v_lshlrev_b32 v37, 4, v74 // (op 5210) 

v_lshrrev_b32 v74, 3, v1 // (op 5210) 
v_lshl_add_u32 v75, v74, 7, v37 // (op 5210) 


s_nop 0
v_readfirstlane_b32 s53, v75 // (op 5210) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -64, v1 // (op 5285) 
v_sub_u32 v74, v1, v37 // (op 5285) 

v_and_b32 v37, 4294967232, v1 // (op 5285) 
v_add_u32 v75, v37, v74 // (op 5285) 


v_and_b32 v37, -4, v1 // (op 5361) 
v_sub_u32 v74, v1, v37 // (op 5361) 

v_and_b32 v37, 4294967292, v1 // (op 5361) 
v_add_u32 v76, v37, v74 // (op 5361) 


v_and_b32 v37, -4, v1 // (op 5989) 
v_sub_u32 v74, v1, v37 // (op 5989) 

v_lshlrev_b32 v37, 4, v74 // (op 5989) 

v_lshrrev_b32 v74, 2, v1 // (op 5989) 
v_lshl_add_u32 v77, v74, 6, v37 // (op 5989) 


s_nop 0
v_readfirstlane_b32 s54, v77 // (op 5989) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -8, v1 // (op 6033) 
v_sub_u32 v74, v1, v37 // (op 6033) 

v_lshlrev_b32 v37, 4, v74 // (op 6033) 

v_lshrrev_b32 v74, 3, v1 // (op 6033) 
v_lshl_add_u32 v77, v74, 7, v37 // (op 6033) 


s_nop 0
v_readfirstlane_b32 s55, v77 // (op 6033) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -64, v1 // (op 6077) 
v_sub_u32 v74, v1, v37 // (op 6077) 

v_and_b32 v37, 4294967232, v1 // (op 6077) 
v_add_u32 v77, v37, v74 // (op 6077) 


v_and_b32 v37, -4, v1 // (op 6106) 
v_sub_u32 v74, v1, v37 // (op 6106) 

v_and_b32 v37, 4294967292, v1 // (op 6106) 
v_add_u32 v78, v37, v74 // (op 6106) 


s_mov_b32 s1, 0 // (op 1903) call()
s_and_b32 s84, s7, 4294967294 // (op 1902) 
s_cmp_lt_u32 s1, s84 // (op 1902) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_1902 // (op 1902) Condition: Top (jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_1902 if false)
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoop_1902:
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:65], v[42:49], a[0:3], v33, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 90) 
s_barrier  // (op 6276) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:65], v[50:57], a[4:7], v33, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1690) 
s_add_u32 s1, s1, 2 // (op 1904) 
s_lshl_b64 s[84:85], s[34:35], 5 // (op 4457) 
v_mov_b64 v[80:81], s[84:85] // (op 4457) convert


v_mov_b64 v[82:83], v[38:39] // (op 4457) 
v_mov_b64 v[84:85], v[82:83] // (op 4457) 
s_mov_b32 m0, s53 // (op 4457) reg expression
s_mov_b32 s84, 256 // (op 4457) call()
buffer_load_dwordx4 v84, s[72:75], s84 offen offset: 0 lds // (op 4457) Load value direct to lds
v_add_co_u32 v84, s[86:87], v84, v80 // (op 4457) least significant half
v_addc_co_u32 v85, s[86:87], 0, v85, s[86:87] // (op 4457) most significant half

s_add_u32 m0, m0, 4096 // (op 4457) 
buffer_load_dwordx4 v84, s[72:75], s84 offen offset: 0 lds // (op 4457) Load value direct to lds



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:73], v[42:49], a[8:11], v36, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1987) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:73], v[50:57], a[12:15], v36, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1941) 
v_mov_b32 v37, v40 // (op 25) 

buffer_load_ubyte v26, v37, s[80:83], 0 offen offset:8 // (op 25) Load value
s_barrier  // (op 6257) 

s_lshl_b64 s[84:85], s[30:31], 6 // (op 4452) 
v_mov_b64 v[80:81], s[84:85] // (op 4452) convert


v_mov_b64 v[84:85], v[34:35] // (op 4452) 
s_lshl_b64 s[84:85], s[30:31], 7 // (op 4452) 
s_lshl_b64 s[86:87], s[84:85], 1 // (op 4452) 

v_mov_b32 v74, s86 // (op 4452) 
v_mov_b32 v79, s87 // (op 4452) 
v_add_co_u32 v86, s[84:85], v84, v74 // (op 4452) least significant half
v_addc_co_u32 v87, s[84:85], v85, v79, s[84:85] // (op 4452) most significant half




s_add_u32 s84, 8192, s52 // (op 4452) 
s_nop 0
s_mov_b32 m0, s84 // (op 4452) reg expression
buffer_load_dwordx4 v86, s[68:71], 0 offen offset: 0 lds // (op 4452) Load value direct to lds
v_add_co_u32 v86, s[86:87], v86, v80 // (op 4452) least significant half
v_addc_co_u32 v87, s[86:87], 0, v87, s[86:87] // (op 4452) most significant half

s_add_u32 m0, m0, 4096 // (op 4452) 
buffer_load_dwordx4 v86, s[68:71], 0 offen offset: 0 lds // (op 4452) Load value direct to lds





v_mov_b32 v74, v29 // (op 17) 
s_lshl_b32 s84, s38, 3 // (op 17) 
v_add_u32 v79, v74, s84 // (op 17) 


buffer_load_ubyte v41, v79, s[76:79], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 6472) 



s_waitcnt vmcnt(9)
ds_write_b8 v78, v30 offset:33280 // (op 1569) Store local data 
s_barrier  // (op 6466) 



s_waitcnt vmcnt(6)
ds_write_b8 v77, v32 offset:33536 // (op 1568) Store local data 
s_barrier  // (op 6452) 



ds_read_b128 v[86:89], v16 offset:16384 // (op 1561) Load local data 
ds_read_b128 v[90:93], v16 offset:16448 // (op 1561) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 6469) 



ds_read_u8 v79, v17 offset:33280 // (op 1565) Load local data 



ds_read_b128 v[94:97], v16 offset:20480 // (op 1695) Load local data 
ds_read_b128 v[98:101], v16 offset:20544 // (op 1695) Load local data 



ds_read_u8 v80, v17 offset:33408 // (op 1699) Load local data 
s_barrier  // (op 6443) 
s_barrier  // (op 6655) 



ds_read_b64_tr_b8 v[58:59], v14 offset:24576 // (op 1559) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v14 offset:25088 // (op 1559) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v14 offset:28672 // (op 1559) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v14 offset:29184 // (op 1559) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 6462) 


ds_read_u8 v33, v15 offset:33536 // (op 1563) Load local data 



ds_read_b64_tr_b8 v[66:67], v14 offset:24608 // (op 1944) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v14 offset:25120 // (op 1944) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v14 offset:28704 // (op 1944) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v14 offset:29216 // (op 1944) Transpose load local data 


ds_read_u8 v36, v15 offset:33568 // (op 1947) Load local data 
s_barrier  // (op 6677) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:65], v[86:93], a[0:3], v33, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1566) 
s_barrier  // (op 6459) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:65], v[94:101], a[4:7], v33, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1691) 
s_lshl_b64 s[84:85], s[34:35], 5 // (op 4513) 
v_mov_b64 v[102:103], s[84:85] // (op 4513) convert


v_mov_b64 v[104:105], v[82:83] // (op 4513) 
s_add_u32 s84, 16384, s55 // (op 4513) 
s_mov_b32 m0, s84 // (op 4513) reg expression
s_mov_b32 s85, 384 // (op 4513) call()
buffer_load_dwordx4 v104, s[72:75], s85 offen offset: 0 lds // (op 4513) Load value direct to lds
v_add_co_u32 v104, s[86:87], v104, v102 // (op 4513) least significant half
v_addc_co_u32 v105, s[86:87], 0, v105, s[86:87] // (op 4513) most significant half

s_add_u32 m0, m0, 4096 // (op 4513) 
buffer_load_dwordx4 v104, s[72:75], s85 offen offset: 0 lds // (op 4513) Load value direct to lds




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:73], v[86:93], a[8:11], v36, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1988) 






v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:73], v[94:101], a[12:15], v36, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1942) 
buffer_load_ubyte v30, v37, s[80:83], 0 offen offset:12 // (op 1572) Load value



v_add_co_u32 v38, vcc, 256, v38 // (op 4797) least significant half
s_barrier  // (op 6449) 
v_addc_co_u32 v39, vcc, 0, v39, vcc // (op 4797) most significant half

v_add_u32 v40, 8, v40 // (op 5053) 
s_lshl_b64 s[84:85], s[30:31], 6 // (op 4508) 
v_mov_b64 v[82:83], s[84:85] // (op 4508) convert


s_lshl_b64 s[84:85], s[30:31], 7 // (op 4508) 
s_mul_i32 s87, 3, s85 // (op 4508) most significant: low of low * high
s_mul_i32 s86, 3, s84 // (op 4508) least significant
s_mul_hi_u32 s88, 3, s84 // (op 4508) most significant: high of low * low
s_add_u32 s87, s87, s88 // (op 4508) most significant: sum


v_mov_b32 v37, s86 // (op 4508) 
v_mov_b32 v79, s87 // (op 4508) 
v_add_co_u32 v80, s[84:85], v84, v37 // (op 4508) least significant half
v_addc_co_u32 v81, s[84:85], v85, v79, s[84:85] // (op 4508) most significant half




s_add_u32 s84, 24576, s54 // (op 4508) 
s_nop 0
s_mov_b32 m0, s84 // (op 4508) reg expression
buffer_load_dwordx4 v80, s[68:71], 0 offen offset: 0 lds // (op 4508) Load value direct to lds
v_add_co_u32 v80, s[86:87], v80, v82 // (op 4508) least significant half
v_addc_co_u32 v81, s[86:87], 0, v81, s[86:87] // (op 4508) most significant half

s_add_u32 m0, m0, 4096 // (op 4508) 
buffer_load_dwordx4 v80, s[68:71], 0 offen offset: 0 lds // (op 4508) Load value direct to lds






s_lshl_b32 s84, s38, 2 // (op 1571) 
s_mul_i32 s85, 3, s84 // (op 1571) 

v_add_u32 v37, v74, s85 // (op 1571) 


s_lshl_b64 s[84:85], s[30:31], 7 // (op 4669) 
s_lshl_b64 s[86:87], s[84:85], 1 // (op 4669) 

v_mov_b32 v79, s86 // (op 4669) 
v_mov_b32 v80, s87 // (op 4669) 
v_add_co_u32 v34, s[84:85], v34, v79 // (op 4669) least significant half
v_addc_co_u32 v35, s[84:85], v35, v80, s[84:85] // (op 4669) most significant half




buffer_load_ubyte v32, v37, s[76:79], 0 offen offset: 0 // (op 1571) Load value


s_barrier  // (op 6317) 
s_lshl_b32 s84, s38, 3 // (op 4925) 



v_add_u32 v29, v29, s84 // (op 4925) 

s_waitcnt vmcnt(9)
ds_write_b8 v76, v26 offset:32768 // (op 30) Store local data 
s_barrier  // (op 6296) 



s_waitcnt vmcnt(6)
ds_write_b8 v75, v41 offset:33024 // (op 21) Store local data 
s_barrier  // (op 6282) 



ds_read_b128 v[42:45], v4 // (op 77) Load local data 
ds_read_b128 v[46:49], v4 offset:64 // (op 77) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 6324) 



ds_read_u8 v28, v5 offset:32768 // (op 87) Load local data 



ds_read_b128 v[50:53], v4 offset:4096 // (op 1685) Load local data 
ds_read_b128 v[54:57], v4 offset:4160 // (op 1685) Load local data 



ds_read_u8 v31, v5 offset:32896 // (op 1689) Load local data 
s_barrier  // (op 6263) 



ds_read_b64_tr_b8 v[58:59], v0 offset:8192 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:8704 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:12288 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v0 offset:12800 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 6303) 


ds_read_u8 v33, v3 offset:33024 // (op 82) Load local data 



ds_read_b64_tr_b8 v[66:67], v0 offset:8224 // (op 1936) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v0 offset:8736 // (op 1936) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v0 offset:12320 // (op 1936) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v0 offset:12832 // (op 1936) Transpose load local data 


ds_read_u8 v36, v3 offset:33056 // (op 1939) Load local data 
s_and_b32 s84, s7, 4294967294 // (op 1902) 
s_cmp_lt_u32 s1, s84 // (op 1902) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoop_1902 // (op 1902) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoop_1902 if true)
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_1902:














s_barrier  // (op 6661) 
s_and_b32 s52, s7, 4294967294 // (op 4581) 
v_and_b32 v26, -4, v1 // (op 4581) 
s_add_u32 s53, s0, s52 // (op 4581) 

v_sub_u32 v29, v1, v26 // (op 4581) 

v_lshlrev_b32 v26, 4, v29 // (op 4581) 

s_mul_i32 s52, s16, s8 // (op 4581) 
s_add_u32 s52, s52, s53 // (op 4581) 

v_lshrrev_b32 v29, 2, v1 // (op 4581) 
s_mul_hi_u32 s53, s52, s13 // (op 4581) 
s_sub_u32 s54, s52, s53 // (op 4581) 
s_lshr_b32 s55, s54, 1 // (op 4581) 

s_add_u32 s54, s55, s53 // (op 4581) 


s_lshr_b32 s53, s54, s11 // (op 4581) 

s_and_b32 s54, s11, 2147483648 // (op 4581) 
s_cmp_eq_i32 s54, 0 // (op 4581) 
s_mov_b32 s55, scc // (op 4581) 

s_cmp_eq_i32 s55, 0 // (op 4581) 
s_cselect_b32 s54, s52, s53 // (op 4581) 


s_mul_hi_u32 s53, s54, s12 // (op 4581) 
s_sub_u32 s55, s54, s53 // (op 4581) 
s_lshr_b32 s84, s55, 1 // (op 4581) 

s_add_u32 s55, s84, s53 // (op 4581) 


s_lshr_b32 s53, s55, s10 // (op 4581) 

s_and_b32 s55, s10, 2147483648 // (op 4581) 
s_cmp_eq_i32 s55, 0 // (op 4581) 
s_mov_b32 s84, scc // (op 4581) 

s_cmp_eq_i32 s84, 0 // (op 4581) 
s_cselect_b32 s55, s54, s53 // (op 4581) 


v_lshl_add_u32 v34, s55, 6, v26 // (op 4581) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 4581) 
v_mov_b32 v41, 0 // (op 4581) convert
v_mov_b32 v40, v34 // (op 4581) convert

s_mul_i32 s53, s54, s84 // (op 4581) 


s_sub_u32 s54, s52, s53 // (op 4581) 


v_lshl_add_u32 v26, s54, 7, v29 // (op 4581) 


v_mul_lo_u32 v35, v26, s31 // (op 4581) most significant: low of low * high
v_mul_hi_u32 v29, v26, s30 // (op 4581) most significant: high of low * low
v_mul_lo_u32 v34, v26, s30 // (op 4581) least significant: low of low * low
v_add_u32 v35, v35, v29 // (op 4581) most significant: sum


v_add_co_u32 v38, s[52:53], v40, v34 // (op 4581) least significant half
v_addc_co_u32 v39, s[52:53], v41, v35, s[52:53] // (op 4581) most significant half



s_mul_i32 s53, s30, s3 // (op 4583) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 4583) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 4583) least significant
s_mul_hi_u32 s55, s30, s2 // (op 4583) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4583) most significant: sum
s_add_u32 s53, s53, s55 // (op 4583) most significant: sum


s_mov_b32 s68, s32 // (op 4583) reg expression
s_mov_b32 s69, s33 // (op 4583) reg expression
s_mov_b32 s70, s52 // (op 4583) reg expression
s_mov_b32 s71, 131072 // (op 4583) call()

s_and_b32 s52, s7, 4294967294 // (op 4721) 
v_lshrrev_b32 v26, 3, v1 // (op 4721) 
s_add_u32 s53, s0, s52 // (op 4721) 

v_and_b32 v29, -8, v1 // (op 4721) 
v_sub_u32 v37, v1, v29 // (op 4721) 

s_mul_i32 s52, s16, s8 // (op 4721) 
s_add_u32 s52, s52, s53 // (op 4721) 

v_lshlrev_b32 v29, 4, v37 // (op 4721) 

s_mul_hi_u32 s53, s52, s13 // (op 4721) 
s_sub_u32 s54, s52, s53 // (op 4721) 
s_lshr_b32 s55, s54, 1 // (op 4721) 

s_add_u32 s54, s55, s53 // (op 4721) 


s_lshr_b32 s53, s54, s11 // (op 4721) 

s_and_b32 s54, s11, 2147483648 // (op 4721) 
s_cmp_eq_i32 s54, 0 // (op 4721) 
s_mov_b32 s55, scc // (op 4721) 

s_cmp_eq_i32 s55, 0 // (op 4721) 
s_cselect_b32 s54, s52, s53 // (op 4721) 


s_mul_hi_u32 s53, s54, s12 // (op 4721) 
s_sub_u32 s55, s54, s53 // (op 4721) 
s_lshr_b32 s84, s55, 1 // (op 4721) 

s_add_u32 s55, s84, s53 // (op 4721) 


s_lshr_b32 s53, s55, s10 // (op 4721) 

s_and_b32 s55, s10, 2147483648 // (op 4721) 
s_cmp_eq_i32 s55, 0 // (op 4721) 
s_mov_b32 s84, scc // (op 4721) 

s_cmp_eq_i32 s84, 0 // (op 4721) 
s_cselect_b32 s55, s54, s53 // (op 4721) 


s_mul_i32 s53, s55, s14 // (op 4721) 

s_sub_u32 s55, s54, s53 // (op 4721) 

v_lshl_add_u32 v37, s55, 6, v26 // (op 4721) 


v_mul_lo_u32 v35, v37, s35 // (op 4721) most significant: low of low * high
v_mul_hi_u32 v26, v37, s34 // (op 4721) most significant: high of low * low
v_mul_lo_u32 v34, v37, s34 // (op 4721) least significant: low of low * low
v_add_u32 v35, v35, v26 // (op 4721) most significant: sum


s_ashr_i64 s[84:85], s[2:3], 7 // (op 4721) 
s_mul_i32 s53, s54, s84 // (op 4721) 


s_sub_u32 s54, s52, s53 // (op 4721) 


v_lshl_add_u32 v26, s54, 7, v29 // (op 4721) 


v_mov_b32 v41, 0 // (op 4721) convert
v_mov_b32 v40, v26 // (op 4721) convert

v_add_co_u32 v74, s[52:53], v40, v34 // (op 4721) least significant half
v_addc_co_u32 v75, s[52:53], v41, v35, s[52:53] // (op 4721) most significant half



s_mul_i32 s53, s34, s23 // (op 4723) most significant: low of low * high
s_mul_i32 s54, s35, s22 // (op 4723) most significant: low of high * low
s_mul_i32 s52, s34, s22 // (op 4723) least significant
s_mul_hi_u32 s55, s34, s22 // (op 4723) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4723) most significant: sum
s_add_u32 s53, s53, s55 // (op 4723) most significant: sum


s_mov_b32 s72, s36 // (op 4723) reg expression
s_mov_b32 s73, s37 // (op 4723) reg expression
s_mov_b32 s74, s52 // (op 4723) reg expression
s_mov_b32 s75, 131072 // (op 4723) call()

s_and_b32 s52, s7, 4294967294 // (op 4849) 
v_and_b32 v26, -64, v1 // (op 4849) 
s_add_u32 s53, s0, s52 // (op 4849) 

v_sub_u32 v29, v1, v26 // (op 4849) 

v_lshrrev_b32 v26, 6, v1 // (op 4849) 
s_mul_i32 s52, s16, s8 // (op 4849) 
s_add_u32 s52, s52, s53 // (op 4849) 

s_mul_hi_u32 s53, s52, s13 // (op 4849) 
s_sub_u32 s54, s52, s53 // (op 4849) 
s_lshr_b32 s55, s54, 1 // (op 4849) 

s_add_u32 s54, s55, s53 // (op 4849) 


s_lshr_b32 s53, s54, s11 // (op 4849) 

s_and_b32 s54, s11, 2147483648 // (op 4849) 
s_cmp_eq_i32 s54, 0 // (op 4849) 
s_mov_b32 s55, scc // (op 4849) 

s_cmp_eq_i32 s55, 0 // (op 4849) 
s_cselect_b32 s54, s52, s53 // (op 4849) 


s_mul_hi_u32 s53, s54, s12 // (op 4849) 
s_sub_u32 s55, s54, s53 // (op 4849) 
s_lshr_b32 s84, s55, 1 // (op 4849) 

s_add_u32 s55, s84, s53 // (op 4849) 


s_lshr_b32 s53, s55, s10 // (op 4849) 

s_and_b32 s55, s10, 2147483648 // (op 4849) 
s_cmp_eq_i32 s55, 0 // (op 4849) 
s_mov_b32 s84, scc // (op 4849) 

s_cmp_eq_i32 s84, 0 // (op 4849) 
s_cselect_b32 s55, s54, s53 // (op 4849) 


v_lshl_add_u32 v37, s55, 6, v29 // (op 4849) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 4849) 
s_mul_i32 s53, s54, s84 // (op 4849) 


s_sub_u32 s54, s52, s53 // (op 4849) 


v_lshl_add_u32 v29, s54, 2, v26 // (op 4849) 


v_mul_lo_u32 v26, v29, s38 // (op 4849) 

v_add_u32 v29, v37, v26 // (op 4849) 


s_mul_i32 s53, s38, s43 // (op 4851) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 4851) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 4851) least significant
s_mul_hi_u32 s55, s38, s42 // (op 4851) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4851) most significant: sum
s_add_u32 s53, s53, s55 // (op 4851) most significant: sum


s_mov_b32 s76, s40 // (op 4851) reg expression
s_mov_b32 s77, s41 // (op 4851) reg expression
s_mov_b32 s78, s52 // (op 4851) reg expression
s_mov_b32 s79, 131072 // (op 4851) call()

s_and_b32 s52, s7, 4294967294 // (op 4977) 
v_lshrrev_b32 v26, 2, v1 // (op 4977) 
s_add_u32 s53, s0, s52 // (op 4977) 

v_and_b32 v37, -4, v1 // (op 4977) 
v_sub_u32 v34, v1, v37 // (op 4977) 

s_mul_i32 s52, s16, s8 // (op 4977) 
s_add_u32 s52, s52, s53 // (op 4977) 

s_mul_hi_u32 s53, s52, s13 // (op 4977) 
s_sub_u32 s54, s52, s53 // (op 4977) 
s_lshr_b32 s55, s54, 1 // (op 4977) 

s_add_u32 s54, s55, s53 // (op 4977) 


s_lshr_b32 s53, s54, s11 // (op 4977) 

s_and_b32 s54, s11, 2147483648 // (op 4977) 
s_cmp_eq_i32 s54, 0 // (op 4977) 
s_mov_b32 s55, scc // (op 4977) 

s_cmp_eq_i32 s55, 0 // (op 4977) 
s_cselect_b32 s54, s52, s53 // (op 4977) 


s_mul_hi_u32 s53, s54, s12 // (op 4977) 
s_sub_u32 s55, s54, s53 // (op 4977) 
s_lshr_b32 s84, s55, 1 // (op 4977) 

s_add_u32 s55, s84, s53 // (op 4977) 


s_lshr_b32 s53, s55, s10 // (op 4977) 

s_and_b32 s55, s10, 2147483648 // (op 4977) 
s_cmp_eq_i32 s55, 0 // (op 4977) 
s_mov_b32 s84, scc // (op 4977) 

s_cmp_eq_i32 s84, 0 // (op 4977) 
s_cselect_b32 s55, s54, s53 // (op 4977) 


s_mul_i32 s53, s55, s14 // (op 4977) 

s_sub_u32 s55, s54, s53 // (op 4977) 

v_lshl_add_u32 v35, s55, 6, v26 // (op 4977) 


v_mul_lo_u32 v26, v35, s44 // (op 4977) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 4977) 
s_mul_i32 s53, s54, s84 // (op 4977) 


s_sub_u32 s54, s52, s53 // (op 4977) 


v_lshl_add_u32 v35, s54, 2, v34 // (op 4977) 


v_add_u32 v34, v35, v26 // (op 4977) 


s_mul_i32 s53, s44, s23 // (op 4979) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 4979) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 4979) least significant
s_mul_hi_u32 s55, s44, s22 // (op 4979) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4979) most significant: sum
s_add_u32 s53, s53, s55 // (op 4979) most significant: sum


s_mov_b32 s80, s46 // (op 4979) reg expression
s_mov_b32 s81, s47 // (op 4979) reg expression
s_mov_b32 s82, s52 // (op 4979) reg expression
s_mov_b32 s83, 131072 // (op 4979) call()

v_and_b32 v26, -4, v1 // (op 5148) 
v_sub_u32 v35, v1, v26 // (op 5148) 

v_lshlrev_b32 v26, 4, v35 // (op 5148) 

v_lshrrev_b32 v35, 2, v1 // (op 5148) 
v_lshl_add_u32 v37, v35, 6, v26 // (op 5148) 


s_nop 0
v_readfirstlane_b32 s52, v37 // (op 5148) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -8, v1 // (op 5223) 
v_sub_u32 v35, v1, v26 // (op 5223) 

v_lshlrev_b32 v26, 4, v35 // (op 5223) 

v_lshrrev_b32 v35, 3, v1 // (op 5223) 
v_lshl_add_u32 v37, v35, 7, v26 // (op 5223) 


s_nop 0
v_readfirstlane_b32 s53, v37 // (op 5223) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -64, v1 // (op 5298) 
v_sub_u32 v35, v1, v26 // (op 5298) 

v_and_b32 v26, 4294967232, v1 // (op 5298) 
v_add_u32 v37, v26, v35 // (op 5298) 


v_and_b32 v26, -4, v1 // (op 5374) 
v_sub_u32 v35, v1, v26 // (op 5374) 

v_and_b32 v26, 4294967292, v1 // (op 5374) 
v_add_u32 v40, v26, v35 // (op 5374) 


s_and_b32 s54, s7, 4294967294 // (op 1908) 
s_cmp_lt_u32 s54, s7 // (op 1907) 










s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_1907 // (op 1907) Condition: Top (jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_1907 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_1907:
s_barrier  // (op 6260) 
s_add_u32 s54, s54, 1 // (op 1909) 

s_lshl_b64 s[84:85], s[30:31], 6 // (op 4495) 
v_mov_b64 v[30:31], s[84:85] // (op 4495) convert


v_mov_b64 v[32:33], v[38:39] // (op 4495) 
v_mov_b64 v[72:73], v[32:33] // (op 4495) reg expression
s_add_u32 s55, 8192, s52 // (op 4495) 
s_mov_b32 m0, s55 // (op 4495) reg expression
buffer_load_dwordx4 v72, s[68:71], 0 offen offset: 0 lds // (op 4495) Load value direct to lds
v_add_co_u32 v72, s[84:85], v72, v30 // (op 4495) least significant half
v_addc_co_u32 v73, s[84:85], 0, v73, s[84:85] // (op 4495) most significant half

s_add_u32 m0, m0, 4096 // (op 4495) 
buffer_load_dwordx4 v72, s[68:71], 0 offen offset: 0 lds // (op 4495) Load value direct to lds




s_lshl_b64 s[84:85], s[30:31], 7 // (op 4571) 
v_mov_b32 v26, s84 // (op 4571) 
v_mov_b32 v28, s85 // (op 4571) 
v_add_co_u32 v38, s[86:87], v38, v26 // (op 4571) least significant half
v_addc_co_u32 v39, s[86:87], v39, v28, s[86:87] // (op 4571) most significant half




s_barrier  // (op 6279) 

s_lshl_b64 s[84:85], s[34:35], 5 // (op 4501) 
v_mov_b64 v[30:31], s[84:85] // (op 4501) convert


v_mov_b64 v[32:33], v[74:75] // (op 4501) 
v_mov_b64 v[72:73], v[32:33] // (op 4501) 
s_mov_b32 m0, s53 // (op 4501) reg expression
buffer_load_dwordx4 v72, s[72:75], 0 offen offset: 0 lds // (op 4501) Load value direct to lds
v_add_co_u32 v72, s[84:85], v72, v30 // (op 4501) least significant half
v_addc_co_u32 v73, s[84:85], 0, v73, s[84:85] // (op 4501) most significant half

s_add_u32 m0, m0, 4096 // (op 4501) 
buffer_load_dwordx4 v72, s[72:75], 0 offen offset: 0 lds // (op 4501) Load value direct to lds



v_add_co_u32 v74, vcc, 128, v74 // (op 4713) least significant half
v_addc_co_u32 v75, vcc, 0, v75, vcc // (op 4713) most significant half


v_mov_b32 v26, v29 // (op 1505) 
v_mov_b32 v28, v26 // (op 1505) reg expression

buffer_load_ubyte v30, v28, s[76:79], 0 offen offset: 0 // (op 1505) Load value


s_lshl_b32 s55, s38, 2 // (op 4841) 
v_add_u32 v29, v29, s55 // (op 4841) 



s_barrier  // (op 6299) 
v_mov_b32 v26, v34 // (op 1506) 




buffer_load_ubyte v28, v26, s[80:83], 0 offen offset: 0 // (op 1506) Load value

v_add_u32 v34, 4, v34 // (op 4969) 
s_waitcnt vmcnt(1)
ds_write_b8 v37, v30 offset:33024 // (op 1502) Store local data 

s_barrier  // (op 6266) 



s_barrier  // (op 6320) 
ds_read_b64_tr_b8 v[66:67], v0 offset:8192 // (op 1493) Transpose load local data 



ds_read_b64_tr_b8 v[68:69], v0 offset:8704 // (op 1493) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v0 offset:12288 // (op 1493) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v0 offset:12800 // (op 1493) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v40, v28 offset:32768 // (op 1503) Store local data 

s_barrier  // (op 6286) 



ds_read_b128 v[58:61], v4 // (op 1495) Load local data 
ds_read_b128 v[62:65], v4 offset:64 // (op 1495) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 6307) 


ds_read_u8 v26, v3 offset:33024 // (op 1497) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 6327) 



ds_read_u8 v28, v5 offset:32768 // (op 1499) Load local data 



ds_read_b128 v[50:53], v4 offset:4096 // (op 1664) Load local data 
ds_read_b128 v[54:57], v4 offset:4160 // (op 1664) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:73], v[58:65], a[0:3], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1500) 



ds_read_u8 v30, v5 offset:32896 // (op 1668) Load local data 



ds_read_b64_tr_b8 v[42:43], v0 offset:8224 // (op 1882) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:8736 // (op 1882) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:73], v[50:57], a[4:7], v26, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1669) 
ds_read_b64_tr_b8 v[46:47], v0 offset:12320 // (op 1882) Transpose load local data 


ds_read_b64_tr_b8 v[48:49], v0 offset:12832 // (op 1882) Transpose load local data 


ds_read_u8 v26, v3 offset:33056 // (op 1885) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[42:49], v[58:65], a[8:11], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1887) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:49], v[50:57], a[12:15], v26, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1918) 




s_cmp_lt_u32 s54, s7 // (op 1907) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_1907 // (op 1907) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_1907 if true)
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_1907:









s_barrier  // (op 6665) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 296) 
s_cmp_lg_u32 s18, 0 // (op 322) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Send_Tile_322 // (op 322) Condition: False, jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Send_Tile_322



v_add_u32 v26, 8192, v25 // (op 2344) 
buffer_store_dword a8, v26, s[64:67], 0 offen sc1 // (op 2344) Store value

v_add_u32 v26, 8448, v25 // (op 2344) 
buffer_store_dword a9, v26, s[64:67], 0 offen sc1 // (op 2344) Store value

v_add_u32 v26, 8704, v25 // (op 2344) 
buffer_store_dword a10, v26, s[64:67], 0 offen sc1 // (op 2344) Store value

v_add_u32 v26, 8960, v25 // (op 2344) 
buffer_store_dword a11, v26, s[64:67], 0 offen sc1 // (op 2344) Store value




v_add_u32 v26, 8320, v25 // (op 2346) 
buffer_store_dword a12, v26, s[64:67], 0 offen sc1 // (op 2346) Store value

v_add_u32 v26, 8576, v25 // (op 2346) 
buffer_store_dword a13, v26, s[64:67], 0 offen sc1 // (op 2346) Store value

v_add_u32 v26, 8832, v25 // (op 2346) 
buffer_store_dword a14, v26, s[64:67], 0 offen sc1 // (op 2346) Store value

v_add_u32 v26, 9088, v25 // (op 2346) 
buffer_store_dword a15, v26, s[64:67], 0 offen sc1 // (op 2346) Store value




buffer_store_dword a0, v25, s[64:67], 0 offen sc1 // (op 319) Store value
buffer_store_dword a1, v25, s[64:67], 0 offen offset:256 sc1 // (op 319) Store value
buffer_store_dword a2, v25, s[64:67], 0 offen offset:512 sc1 // (op 319) Store value
buffer_store_dword a3, v25, s[64:67], 0 offen offset:768 sc1 // (op 319) Store value



buffer_store_dword a4, v25, s[64:67], 0 offen offset:128 sc1 // (op 2330) Store value
buffer_store_dword a5, v25, s[64:67], 0 offen offset:384 sc1 // (op 2330) Store value
buffer_store_dword a6, v25, s[64:67], 0 offen offset:640 sc1 // (op 2330) Store value
buffer_store_dword a7, v25, s[64:67], 0 offen offset:896 sc1 // (op 2330) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 323) 
v_cmp_eq_i32 vcc, v1, 0 // (op 327) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_327 // (op 327) Condition: False, jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_327
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
s_branch RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_327 // (op 327) Condition: Done, jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_327
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_327:
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_327:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Send_Tile_322 // (op 322) Condition: Done, jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Send_Tile_322
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Send_Tile_322:
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Send_Tile_322:

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
s_cbranch_scc0 RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Receive_Tile_338 // (op 338) Condition: False, jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Receive_Tile_338
s_mov_b32 s52, 0 // (op 304) call()
s_mul_i32 s53, s52, s8 // (op 303) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 303) 
s_mov_b32 s84, s54 // (op 303) convert

s_sub_u32 s54, s84, s19 // (op 303) 

s_sub_u32 s55, s54, 1 // (op 303) 

s_cmp_lt_u32 s53, s55 // (op 303) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_303 // (op 303) Condition: Top (jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_303 if false)
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_ReceiveTileLoop_303:
s_add_u32 s53, s16, 1 // (op 5828) 
v_lshrrev_b32 v20, 5, v1 // (op 5828) 
s_add_u32 s54, s53, s52 // (op 5828) 
s_lshl_b32 s54, s54, 6 // (op 5828) 

v_lshlrev_b32 v21, 2, v20 // (op 5828) 

v_add_lshl_u32 v20, s54, v21, 6 // (op 5828) 


v_and_b32 v21, -32, v1 // (op 5828) 
v_sub_u32 v23, v1, v21 // (op 5828) 

v_add_lshl_u32 v21, v20, v23, 2 // (op 5828) 


s_lshl_b32 s53, s15, 14 // (op 5830) 
s_mov_b32 s64, s28 // (op 5830) reg expression
s_mov_b32 s65, s29 // (op 5830) reg expression
s_mov_b32 s66, s53 // (op 5830) reg expression
s_mov_b32 s67, 131072 // (op 5830) call()

RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_342:
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
s_cbranch_scc1 RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_342 // (op 342) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_342 if true)
s_barrier  // (op 346) 
v_cmp_eq_i32 vcc, v1, 0 // (op 345) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_345 // (op 345) Condition: False, jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_345
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
s_branch RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_345 // (op 345) Condition: Done, jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_345
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_345:
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_345:




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




buffer_load_dword v28, v21, s[64:67], 0 offen offset:128 // (op 2369) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:384 // (op 2369) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:640 // (op 2369) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:896 // (op 2369) Load value
v_accvgpr_read v32, a4 // (op 2370) reg expression
v_accvgpr_read v33, a5 // (op 2370) reg expression
v_accvgpr_read v34, a6 // (op 2370) reg expression
v_accvgpr_read v35, a7 // (op 2370) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 2371) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 2371) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 2371) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 2371) 

v_accvgpr_write a4, v32 // (op 2372) reg expression
v_accvgpr_write a5, v33 // (op 2372) reg expression
v_accvgpr_write a6, v34 // (op 2372) reg expression
v_accvgpr_write a7, v35 // (op 2372) reg expression




v_mov_b32 v20, 8192 // (op 2394) 
v_add_i32 v23, v21, v20 // (op 2394) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 2394) Load value

v_mov_b32 v20, 8448 // (op 2394) 
v_add_i32 v23, v21, v20 // (op 2394) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 2394) Load value

v_mov_b32 v20, 8704 // (op 2394) 
v_add_i32 v23, v21, v20 // (op 2394) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 2394) Load value

v_mov_b32 v20, 8960 // (op 2394) 
v_add_i32 v23, v21, v20 // (op 2394) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 2394) Load value

v_accvgpr_read v32, a8 // (op 2391) reg expression
v_accvgpr_read v33, a9 // (op 2391) reg expression
v_accvgpr_read v34, a10 // (op 2391) reg expression
v_accvgpr_read v35, a11 // (op 2391) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 2392) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 2392) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 2392) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 2392) 

v_accvgpr_write a8, v32 // (op 2393) reg expression
v_accvgpr_write a9, v33 // (op 2393) reg expression
v_accvgpr_write a10, v34 // (op 2393) reg expression
v_accvgpr_write a11, v35 // (op 2393) reg expression




v_mov_b32 v20, 8320 // (op 2399) 
v_add_i32 v23, v21, v20 // (op 2399) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 2399) Load value

v_mov_b32 v20, 8576 // (op 2399) 
v_add_i32 v23, v21, v20 // (op 2399) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 2399) Load value

v_mov_b32 v20, 8832 // (op 2399) 
v_add_i32 v23, v21, v20 // (op 2399) 

buffer_load_dword v30, v23, s[64:67], 0 offen // (op 2399) Load value

v_mov_b32 v20, 9088 // (op 2399) 
v_add_i32 v23, v21, v20 // (op 2399) 

buffer_load_dword v31, v23, s[64:67], 0 offen // (op 2399) Load value

v_accvgpr_read v32, a12 // (op 2396) reg expression
v_accvgpr_read v33, a13 // (op 2396) reg expression
v_accvgpr_read v34, a14 // (op 2396) reg expression
v_accvgpr_read v35, a15 // (op 2396) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 2397) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 2397) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 2397) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 2397) 

v_accvgpr_write a12, v32 // (op 2398) reg expression
v_accvgpr_write a13, v33 // (op 2398) reg expression
v_accvgpr_write a14, v34 // (op 2398) reg expression
v_accvgpr_write a15, v35 // (op 2398) reg expression

s_add_u32 s52, s52, 1 // (op 305) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 303) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 303) 
s_mov_b32 s84, s54 // (op 303) convert

s_sub_u32 s54, s84, s19 // (op 303) 

s_sub_u32 s55, s54, 1 // (op 303) 

s_cmp_lt_u32 s53, s55 // (op 303) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_ReceiveTileLoop_303 // (op 303) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_ReceiveTileLoop_303 if true)
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_303:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Receive_Tile_338 // (op 338) Condition: Done, jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Receive_Tile_338
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Receive_Tile_338:
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Receive_Tile_338:

s_cmp_eq_i32 s18, 0 // (op 381) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_381 // (op 381) Condition: False, jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_381
v_cmp_eq_f32 vcc, 0.00000, v18 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_135
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



v_accvgpr_read v28, a4 // (op 2429) 
v_accvgpr_read v29, a5 // (op 2429) 
v_accvgpr_read v30, a6 // (op 2429) 
v_accvgpr_read v31, a7 // (op 2429) 
v_mul_f32 v32, v19, v28 // (op 2429) 
v_mul_f32 v33, v19, v29 // (op 2429) 
v_mul_f32 v34, v19, v30 // (op 2429) 
v_mul_f32 v35, v19, v31 // (op 2429) 

v_lshrrev_b32 v28, 16, v32 // (op 2431) 
v_lshrrev_b32 v29, 16, v33 // (op 2431) 
v_lshrrev_b32 v30, 16, v34 // (op 2431) 
v_lshrrev_b32 v31, 16, v35 // (op 2431) 


v_pack_b32_f16 v20, v28, v29 // (op 2432) 
v_pack_b32_f16 v21, v30, v31 // (op 2432) 

s_lshl_b32 s52, s24, 6 // (op 2432) 
v_add_u32 v23, v27, s52 // (op 2432) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 2432) Store value



v_accvgpr_read v28, a8 // (op 2455) 
v_accvgpr_read v29, a9 // (op 2455) 
v_accvgpr_read v30, a10 // (op 2455) 
v_accvgpr_read v31, a11 // (op 2455) 
v_mul_f32 v32, v19, v28 // (op 2455) 
v_mul_f32 v33, v19, v29 // (op 2455) 
v_mul_f32 v34, v19, v30 // (op 2455) 
v_mul_f32 v35, v19, v31 // (op 2455) 

v_lshrrev_b32 v28, 16, v32 // (op 2457) 
v_lshrrev_b32 v29, 16, v33 // (op 2457) 
v_lshrrev_b32 v30, 16, v34 // (op 2457) 
v_lshrrev_b32 v31, 16, v35 // (op 2457) 


v_pack_b32_f16 v20, v28, v29 // (op 2459) 
v_pack_b32_f16 v21, v30, v31 // (op 2459) 

v_add_u32 v23, 64, v27 // (op 2459) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 2459) Store value



v_accvgpr_read v28, a12 // (op 2465) 
v_accvgpr_read v29, a13 // (op 2465) 
v_accvgpr_read v30, a14 // (op 2465) 
v_accvgpr_read v31, a15 // (op 2465) 
v_mul_f32 v32, v19, v28 // (op 2465) 
v_mul_f32 v33, v19, v29 // (op 2465) 
v_mul_f32 v34, v19, v30 // (op 2465) 
v_mul_f32 v35, v19, v31 // (op 2465) 

v_lshrrev_b32 v28, 16, v32 // (op 2467) 
v_lshrrev_b32 v29, 16, v33 // (op 2467) 
v_lshrrev_b32 v30, 16, v34 // (op 2467) 
v_lshrrev_b32 v31, 16, v35 // (op 2467) 


v_pack_b32_f16 v20, v28, v29 // (op 2461) 
v_pack_b32_f16 v21, v30, v31 // (op 2461) 

s_lshl_b32 s52, s24, 6 // (op 2461) 
s_add_u32 s52, s52, 64 // (op 2461) 
v_add_u32 v23, v27, s52 // (op 2461) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 2461) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_135
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_135:


v_add_u32 v20, 64, v22 // (op 2551) 

buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 2551) Load value

v_accvgpr_read v28, a8 // (op 2542) 
v_accvgpr_read v29, a9 // (op 2542) 
v_accvgpr_read v30, a10 // (op 2542) 
v_accvgpr_read v31, a11 // (op 2542) 
v_mul_f32 v32, v19, v28 // (op 2542) 
v_mul_f32 v33, v19, v29 // (op 2542) 
v_mul_f32 v34, v19, v30 // (op 2542) 
v_mul_f32 v35, v19, v31 // (op 2542) 

s_waitcnt vmcnt(0)
v_and_b32 v20, 65535, v24 // (op 2550) 
v_lshrrev_b32 v21, 16, v24 // (op 2550) 
v_lshlrev_b32 v20, 16, v20 // (op 2550) 
v_lshlrev_b32 v21, 16, v21 // (op 2550) 
v_mul_f32 v28, v18, v20 // (op 2550) 
v_mul_f32 v29, v18, v21 // (op 2550) 
v_and_b32 v20, 65535, v25 // (op 2550) 
v_lshrrev_b32 v21, 16, v25 // (op 2550) 
v_lshlrev_b32 v20, 16, v20 // (op 2550) 
v_lshlrev_b32 v21, 16, v21 // (op 2550) 
v_mul_f32 v30, v18, v20 // (op 2550) 
v_mul_f32 v31, v18, v21 // (op 2550) 


v_add_f32 v36, v28, v32 // (op 2543) 
v_add_f32 v37, v29, v33 // (op 2543) 
v_add_f32 v38, v30, v34 // (op 2543) 
v_add_f32 v39, v31, v35 // (op 2543) 


v_lshrrev_b32 v28, 16, v36 // (op 2544) 
v_lshrrev_b32 v29, 16, v37 // (op 2544) 
v_lshrrev_b32 v30, 16, v38 // (op 2544) 
v_lshrrev_b32 v31, 16, v39 // (op 2544) 


v_pack_b32_f16 v20, v28, v29 // (op 2546) 
v_pack_b32_f16 v21, v30, v31 // (op 2546) 

v_add_u32 v23, 64, v27 // (op 2546) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 2546) Store value





s_lshl_b32 s52, s4, 6 // (op 2556) 
s_add_u32 s52, s52, 64 // (op 2556) 
v_add_u32 v20, v22, s52 // (op 2556) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 2556) Load value

v_accvgpr_read v28, a12 // (op 2558) 
v_accvgpr_read v29, a13 // (op 2558) 
v_accvgpr_read v30, a14 // (op 2558) 
v_accvgpr_read v31, a15 // (op 2558) 
v_mul_f32 v32, v19, v28 // (op 2558) 
v_mul_f32 v33, v19, v29 // (op 2558) 
v_mul_f32 v34, v19, v30 // (op 2558) 
v_mul_f32 v35, v19, v31 // (op 2558) 

s_waitcnt vmcnt(0)
v_and_b32 v20, 65535, v24 // (op 2553) 
v_lshrrev_b32 v21, 16, v24 // (op 2553) 
v_lshlrev_b32 v20, 16, v20 // (op 2553) 
v_lshlrev_b32 v21, 16, v21 // (op 2553) 
v_mul_f32 v28, v18, v20 // (op 2553) 
v_mul_f32 v29, v18, v21 // (op 2553) 
v_and_b32 v20, 65535, v25 // (op 2553) 
v_lshrrev_b32 v21, 16, v25 // (op 2553) 
v_lshlrev_b32 v20, 16, v20 // (op 2553) 
v_lshlrev_b32 v21, 16, v21 // (op 2553) 
v_mul_f32 v30, v18, v20 // (op 2553) 
v_mul_f32 v31, v18, v21 // (op 2553) 


v_add_f32 v36, v28, v32 // (op 2554) 
v_add_f32 v37, v29, v33 // (op 2554) 
v_add_f32 v38, v30, v34 // (op 2554) 
v_add_f32 v39, v31, v35 // (op 2554) 


v_lshrrev_b32 v28, 16, v36 // (op 2555) 
v_lshrrev_b32 v29, 16, v37 // (op 2555) 
v_lshrrev_b32 v30, 16, v38 // (op 2555) 
v_lshrrev_b32 v31, 16, v39 // (op 2555) 


v_pack_b32_f16 v20, v28, v29 // (op 2548) 
v_pack_b32_f16 v21, v30, v31 // (op 2548) 

s_lshl_b32 s52, s24, 6 // (op 2548) 
s_add_u32 s52, s52, 64 // (op 2548) 
v_add_u32 v23, v27, s52 // (op 2548) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 2548) Store value





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





s_lshl_b32 s52, s4, 6 // (op 2511) 
v_add_u32 v20, v22, s52 // (op 2511) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 2511) Load value

v_accvgpr_read v28, a4 // (op 2507) 
v_accvgpr_read v29, a5 // (op 2507) 
v_accvgpr_read v30, a6 // (op 2507) 
v_accvgpr_read v31, a7 // (op 2507) 
v_mul_f32 v32, v19, v28 // (op 2507) 
v_mul_f32 v33, v19, v29 // (op 2507) 
v_mul_f32 v34, v19, v30 // (op 2507) 
v_mul_f32 v35, v19, v31 // (op 2507) 

s_waitcnt vmcnt(0)
v_and_b32 v20, 65535, v24 // (op 2512) 
v_lshrrev_b32 v21, 16, v24 // (op 2512) 
v_lshlrev_b32 v20, 16, v20 // (op 2512) 
v_lshlrev_b32 v21, 16, v21 // (op 2512) 
v_mul_f32 v28, v18, v20 // (op 2512) 
v_mul_f32 v29, v18, v21 // (op 2512) 
v_and_b32 v20, 65535, v25 // (op 2512) 
v_lshrrev_b32 v21, 16, v25 // (op 2512) 
v_lshlrev_b32 v20, 16, v20 // (op 2512) 
v_lshlrev_b32 v21, 16, v21 // (op 2512) 
v_mul_f32 v30, v18, v20 // (op 2512) 
v_mul_f32 v31, v18, v21 // (op 2512) 


v_add_f32 v36, v28, v32 // (op 2508) 
v_add_f32 v37, v29, v33 // (op 2508) 
v_add_f32 v38, v30, v34 // (op 2508) 
v_add_f32 v39, v31, v35 // (op 2508) 


v_lshrrev_b32 v28, 16, v36 // (op 2509) 
v_lshrrev_b32 v29, 16, v37 // (op 2509) 
v_lshrrev_b32 v30, 16, v38 // (op 2509) 
v_lshrrev_b32 v31, 16, v39 // (op 2509) 


v_pack_b32_f16 v20, v28, v29 // (op 2510) 
v_pack_b32_f16 v21, v30, v31 // (op 2510) 

s_lshl_b32 s52, s24, 6 // (op 2510) 
v_add_u32 v23, v27, s52 // (op 2510) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 2510) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_135:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_381 // (op 381) Condition: Done, jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_381
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_381:
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_381:


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
s_cbranch_scc1 RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_SKStreamTileLoop_294 // (op 294) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_SKStreamTileLoop_294 if true)
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_294:

















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
s_cbranch_scc0 RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_449 // (op 449) Condition: Top (jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_449 if false)
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_DPStreamTileLoop_449:
v_lshrrev_b32 v0, 7, v1 // (op 5118) 
s_mul_i32 s15, s16, s50 // (op 5118) 
s_add_u32 s15, s15, s0 // (op 5118) 
v_and_b32 v3, 63, v1 // (op 5118) 
s_add_u32 s6, s15, s9 // (op 5118) 

v_lshrrev_b32 v4, 4, v3 // (op 5118) 
s_mul_hi_u32 s15, s6, s13 // (op 5118) 
v_lshlrev_b32 v5, 2, v4 // (op 5118) 

s_sub_u32 s7, s6, s15 // (op 5118) 
v_lshl_add_u32 v4, v0, 4, v5 // (op 5118) 


s_lshr_b32 s8, s7, 1 // (op 5118) 

v_lshrrev_b32 v0, 2, v3 // (op 5118) 

s_add_u32 s7, s8, s15 // (op 5118) 


v_and_b32 v3, 3, v0 // (op 5118) 

s_lshr_b32 s8, s7, s11 // (op 5118) 

v_and_b32 v0, 3, v1 // (op 5118) 
s_and_b32 s7, s11, 2147483648 // (op 5118) 
v_lshl_add_u32 v5, v3, 2, v0 // (op 5118) 


s_cmp_eq_i32 s7, 0 // (op 5118) 
s_mov_b32 s15, scc // (op 5118) 

v_lshrrev_b32 v0, 6, v1 // (op 5118) 
s_cmp_eq_i32 s15, 0 // (op 5118) 
s_cselect_b32 s7, s6, s8 // (op 5118) 



v_and_b32 v3, 1, v0 // (op 5118) 

s_mul_hi_u32 s6, s7, s12 // (op 5118) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 5118) 


s_sub_u32 s8, s7, s6 // (op 5118) 
s_lshr_b32 s15, s8, 1 // (op 5118) 

s_add_u32 s8, s15, s6 // (op 5118) 


s_lshr_b32 s6, s8, s10 // (op 5118) 

s_and_b32 s8, s10, 2147483648 // (op 5118) 
s_cmp_eq_i32 s8, 0 // (op 5118) 
s_mov_b32 s15, scc // (op 5118) 

s_cmp_eq_i32 s15, 0 // (op 5118) 
s_cselect_b32 s8, s7, s6 // (op 5118) 


s_mul_i32 s6, s8, s14 // (op 5118) 
v_lshl_add_u32 v3, s8, 6, v4 // (op 5118) 


s_sub_u32 s8, s7, s6 // (op 5118) 


v_lshl_add_u32 v4, s8, 6, v0 // (op 5118) 


v_mul_lo_u32 v0, v4, s4 // (op 5118) 

v_add_lshl_u32 v4, v3, v0, 1 // (op 5118) 


s_mul_i32 s19, s4, s23 // (op 5120) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 5120) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 5120) least significant
s_mul_hi_u32 s6, s4, s22 // (op 5120) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 5120) most significant: sum
s_add_u32 s19, s19, s6 // (op 5120) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 5120) 

s_mov_b32 s56, s20 // (op 5120) reg expression
s_mov_b32 s57, s21 // (op 5120) reg expression
s_mov_b32 s58, s28 // (op 5120) reg expression
s_mov_b32 s59, 131072 // (op 5120) call()

v_lshrrev_b32 v0, 7, v1 // (op 5472) 
s_mul_i32 s15, s16, s50 // (op 5472) 
s_add_u32 s15, s15, s0 // (op 5472) 
v_and_b32 v3, 63, v1 // (op 5472) 
s_add_u32 s6, s15, s9 // (op 5472) 

v_lshrrev_b32 v5, 4, v3 // (op 5472) 
s_mul_hi_u32 s15, s6, s13 // (op 5472) 
v_lshlrev_b32 v14, 2, v5 // (op 5472) 

s_sub_u32 s7, s6, s15 // (op 5472) 
v_lshl_add_u32 v5, v0, 4, v14 // (op 5472) 


s_lshr_b32 s8, s7, 1 // (op 5472) 

v_lshrrev_b32 v0, 2, v3 // (op 5472) 

s_add_u32 s7, s8, s15 // (op 5472) 


v_and_b32 v3, 3, v0 // (op 5472) 

s_lshr_b32 s8, s7, s11 // (op 5472) 

v_and_b32 v0, 3, v1 // (op 5472) 
s_and_b32 s7, s11, 2147483648 // (op 5472) 
v_lshl_add_u32 v14, v3, 2, v0 // (op 5472) 


s_cmp_eq_i32 s7, 0 // (op 5472) 
s_mov_b32 s15, scc // (op 5472) 

v_lshrrev_b32 v0, 6, v1 // (op 5472) 
s_cmp_eq_i32 s15, 0 // (op 5472) 
s_cselect_b32 s7, s6, s8 // (op 5472) 



v_and_b32 v3, 1, v0 // (op 5472) 

s_mul_hi_u32 s6, s7, s12 // (op 5472) 
v_lshl_add_u32 v0, v3, 4, v14 // (op 5472) 


s_sub_u32 s8, s7, s6 // (op 5472) 
s_lshr_b32 s15, s8, 1 // (op 5472) 

s_add_u32 s8, s15, s6 // (op 5472) 


s_lshr_b32 s6, s8, s10 // (op 5472) 

s_and_b32 s8, s10, 2147483648 // (op 5472) 
s_cmp_eq_i32 s8, 0 // (op 5472) 
s_mov_b32 s15, scc // (op 5472) 

s_cmp_eq_i32 s15, 0 // (op 5472) 
s_cselect_b32 s8, s7, s6 // (op 5472) 


s_mul_i32 s6, s8, s14 // (op 5472) 
v_lshl_add_u32 v3, s8, 6, v5 // (op 5472) 


s_sub_u32 s8, s7, s6 // (op 5472) 


v_lshl_add_u32 v5, s8, 6, v0 // (op 5472) 


v_mul_lo_u32 v0, v5, s24 // (op 5472) 

v_add_lshl_u32 v5, v3, v0, 1 // (op 5472) 


s_mul_i32 s19, s24, s23 // (op 5474) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 5474) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 5474) least significant
s_mul_hi_u32 s6, s24, s22 // (op 5474) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 5474) most significant: sum
s_add_u32 s19, s19, s6 // (op 5474) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 5474) 

s_mov_b32 s60, s26 // (op 5474) reg expression
s_mov_b32 s61, s27 // (op 5474) reg expression
s_mov_b32 s62, s28 // (op 5474) reg expression
s_mov_b32 s63, 131072 // (op 5474) call()

v_and_b32 v0, 48, v1 // (op 5551) 
v_and_b32 v3, 63, v1 // (op 5551) 
v_sub_u32 v14, v3, v0 // (op 5551) 

v_lshrrev_b32 v3, 1, v14 // (op 5551) 
v_add_lshl_u32 v15, v0, v3, 6 // (op 5551) 


v_and_b32 v0, 1, v14 // (op 5551) 

v_lshlrev_b32 v3, 3, v0 // (op 5551) 

v_lshrrev_b32 v0, 7, v1 // (op 5551) 
v_lshl_add_u32 v14, v0, 4, v3 // (op 5551) 


v_add_u32 v0, v15, v14 // (op 5551) 


v_lshrrev_b32 v3, 6, v1 // (op 5626) 
v_and_b32 v14, 1, v3 // (op 5626) 

v_and_b32 v3, 15, v1 // (op 5626) 
v_lshl_add_u32 v15, v14, 4, v3 // (op 5626) 


v_and_b32 v3, 48, v1 // (op 5626) 
v_lshl_add_u32 v14, v15, 7, v3 // (op 5626) 


v_lshrrev_b32 v3, 7, v1 // (op 5702) 
v_and_b32 v15, 15, v1 // (op 5702) 
v_lshl_add_u32 v16, v3, 4, v15 // (op 5702) 


v_and_b32 v3, 63, v1 // (op 5702) 
v_lshrrev_b32 v15, 4, v3 // (op 5702) 

v_lshl_add_u32 v3, v15, 6, v16 // (op 5702) 


v_lshrrev_b32 v15, 6, v1 // (op 5778) 
v_and_b32 v16, 1, v15 // (op 5778) 

v_and_b32 v15, 15, v1 // (op 5778) 
v_lshl_add_u32 v17, v16, 4, v15 // (op 5778) 


v_and_b32 v15, 63, v1 // (op 5778) 
v_lshrrev_b32 v16, 4, v15 // (op 5778) 

v_lshl_add_u32 v15, v17, 2, v16 // (op 5778) 


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
v_and_b32 v16, -4, v1 // (op 4610) 
s_mul_i32 s15, s16, s50 // (op 4610) 
s_add_u32 s15, s15, s0 // (op 4610) 
v_sub_u32 v17, v1, v16 // (op 4610) 

s_add_u32 s6, s15, s9 // (op 4610) 

v_lshlrev_b32 v16, 4, v17 // (op 4610) 

s_mul_hi_u32 s15, s6, s13 // (op 4610) 
v_lshrrev_b32 v17, 2, v1 // (op 4610) 
s_sub_u32 s7, s6, s15 // (op 4610) 
s_lshr_b32 s8, s7, 1 // (op 4610) 

s_add_u32 s7, s8, s15 // (op 4610) 


s_lshr_b32 s8, s7, s11 // (op 4610) 

s_and_b32 s7, s11, 2147483648 // (op 4610) 
s_cmp_eq_i32 s7, 0 // (op 4610) 
s_mov_b32 s15, scc // (op 4610) 

s_cmp_eq_i32 s15, 0 // (op 4610) 
s_cselect_b32 s7, s6, s8 // (op 4610) 


s_mul_hi_u32 s8, s7, s12 // (op 4610) 
s_sub_u32 s15, s7, s8 // (op 4610) 
s_lshr_b32 s18, s15, 1 // (op 4610) 

s_add_u32 s15, s18, s8 // (op 4610) 


s_lshr_b32 s8, s15, s10 // (op 4610) 

s_and_b32 s15, s10, 2147483648 // (op 4610) 
s_cmp_eq_i32 s15, 0 // (op 4610) 
s_mov_b32 s18, scc // (op 4610) 

s_cmp_eq_i32 s18, 0 // (op 4610) 
s_cselect_b32 s15, s7, s8 // (op 4610) 


v_lshl_add_u32 v20, s15, 6, v16 // (op 4610) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4610) 
v_mov_b32 v23, 0 // (op 4610) convert
v_mov_b32 v22, v20 // (op 4610) convert

s_mul_i32 s8, s7, s18 // (op 4610) 


s_sub_u32 s7, s6, s8 // (op 4610) 


v_lshl_add_u32 v16, s7, 7, v17 // (op 4610) 


v_mul_lo_u32 v21, v16, s31 // (op 4610) most significant: low of low * high
v_mul_hi_u32 v17, v16, s30 // (op 4610) most significant: high of low * low
v_mul_lo_u32 v20, v16, s30 // (op 4610) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 4610) most significant: sum


v_add_co_u32 v16, s[18:19], v22, v20 // (op 4610) least significant half
v_addc_co_u32 v17, s[18:19], v23, v21, s[18:19] // (op 4610) most significant half



s_mul_i32 s19, s30, s3 // (op 4612) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 4612) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 4612) least significant
s_mul_hi_u32 s6, s30, s2 // (op 4612) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4612) most significant: sum
s_add_u32 s19, s19, s6 // (op 4612) most significant: sum


s_mov_b32 s68, s32 // (op 4612) reg expression
s_mov_b32 s69, s33 // (op 4612) reg expression
s_mov_b32 s70, s18 // (op 4612) reg expression
s_mov_b32 s71, 131072 // (op 4612) call()

v_and_b32 v20, -4, v1 // (op 4629) 
s_mul_i32 s15, s16, s50 // (op 4629) 
s_add_u32 s15, s15, s0 // (op 4629) 
v_sub_u32 v21, v1, v20 // (op 4629) 

s_add_u32 s6, s15, s9 // (op 4629) 

v_lshlrev_b32 v20, 4, v21 // (op 4629) 

s_mul_hi_u32 s15, s6, s13 // (op 4629) 
v_lshrrev_b32 v21, 2, v1 // (op 4629) 
s_sub_u32 s7, s6, s15 // (op 4629) 
s_lshr_b32 s8, s7, 1 // (op 4629) 

s_add_u32 s7, s8, s15 // (op 4629) 


s_lshr_b32 s8, s7, s11 // (op 4629) 

s_and_b32 s7, s11, 2147483648 // (op 4629) 
s_cmp_eq_i32 s7, 0 // (op 4629) 
s_mov_b32 s15, scc // (op 4629) 

s_cmp_eq_i32 s15, 0 // (op 4629) 
s_cselect_b32 s7, s6, s8 // (op 4629) 


s_mul_hi_u32 s8, s7, s12 // (op 4629) 
s_sub_u32 s15, s7, s8 // (op 4629) 
s_lshr_b32 s18, s15, 1 // (op 4629) 

s_add_u32 s15, s18, s8 // (op 4629) 


s_lshr_b32 s8, s15, s10 // (op 4629) 

s_and_b32 s15, s10, 2147483648 // (op 4629) 
s_cmp_eq_i32 s15, 0 // (op 4629) 
s_mov_b32 s18, scc // (op 4629) 

s_cmp_eq_i32 s18, 0 // (op 4629) 
s_cselect_b32 s15, s7, s8 // (op 4629) 


v_lshl_add_u32 v22, s15, 6, v20 // (op 4629) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4629) 
v_mov_b32 v25, 0 // (op 4629) convert
v_mov_b32 v24, v22 // (op 4629) convert

s_mul_i32 s8, s7, s18 // (op 4629) 


s_sub_u32 s7, s6, s8 // (op 4629) 


v_lshl_add_u32 v20, s7, 7, v21 // (op 4629) 


v_mul_lo_u32 v23, v20, s31 // (op 4629) most significant: low of low * high
v_mul_hi_u32 v21, v20, s30 // (op 4629) most significant: high of low * low
v_mul_lo_u32 v22, v20, s30 // (op 4629) least significant: low of low * low
v_add_u32 v23, v23, v21 // (op 4629) most significant: sum


v_add_co_u32 v20, s[18:19], v24, v22 // (op 4629) least significant half
v_addc_co_u32 v21, s[18:19], v25, v23, s[18:19] // (op 4629) most significant half



s_mul_i32 s19, s30, s3 // (op 4631) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 4631) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 4631) least significant
s_mul_hi_u32 s6, s30, s2 // (op 4631) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4631) most significant: sum
s_add_u32 s19, s19, s6 // (op 4631) most significant: sum


s_mov_b32 s68, s32 // (op 4631) reg expression
s_mov_b32 s69, s33 // (op 4631) reg expression
s_mov_b32 s70, s18 // (op 4631) reg expression
s_mov_b32 s71, 131072 // (op 4631) call()

v_lshrrev_b32 v22, 3, v1 // (op 4744) 
s_mul_i32 s15, s16, s50 // (op 4744) 
s_add_u32 s15, s15, s0 // (op 4744) 
v_and_b32 v23, -8, v1 // (op 4744) 
s_add_u32 s6, s15, s9 // (op 4744) 

v_sub_u32 v24, v1, v23 // (op 4744) 

s_mul_hi_u32 s15, s6, s13 // (op 4744) 
v_lshlrev_b32 v23, 4, v24 // (op 4744) 

s_sub_u32 s7, s6, s15 // (op 4744) 
s_lshr_b32 s8, s7, 1 // (op 4744) 

s_add_u32 s7, s8, s15 // (op 4744) 


s_lshr_b32 s8, s7, s11 // (op 4744) 

s_and_b32 s7, s11, 2147483648 // (op 4744) 
s_cmp_eq_i32 s7, 0 // (op 4744) 
s_mov_b32 s15, scc // (op 4744) 

s_cmp_eq_i32 s15, 0 // (op 4744) 
s_cselect_b32 s7, s6, s8 // (op 4744) 


s_mul_hi_u32 s8, s7, s12 // (op 4744) 
s_sub_u32 s15, s7, s8 // (op 4744) 
s_lshr_b32 s18, s15, 1 // (op 4744) 

s_add_u32 s15, s18, s8 // (op 4744) 


s_lshr_b32 s8, s15, s10 // (op 4744) 

s_and_b32 s15, s10, 2147483648 // (op 4744) 
s_cmp_eq_i32 s15, 0 // (op 4744) 
s_mov_b32 s18, scc // (op 4744) 

s_cmp_eq_i32 s18, 0 // (op 4744) 
s_cselect_b32 s15, s7, s8 // (op 4744) 


s_mul_i32 s8, s15, s14 // (op 4744) 

s_sub_u32 s15, s7, s8 // (op 4744) 

v_lshl_add_u32 v24, s15, 6, v22 // (op 4744) 


v_mul_lo_u32 v27, v24, s35 // (op 4744) most significant: low of low * high
v_mul_hi_u32 v22, v24, s34 // (op 4744) most significant: high of low * low
v_mul_lo_u32 v26, v24, s34 // (op 4744) least significant: low of low * low
v_add_u32 v27, v27, v22 // (op 4744) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4744) 
s_mul_i32 s8, s7, s18 // (op 4744) 


s_sub_u32 s7, s6, s8 // (op 4744) 


v_lshl_add_u32 v22, s7, 7, v23 // (op 4744) 


v_mov_b32 v25, 0 // (op 4744) convert
v_mov_b32 v24, v22 // (op 4744) convert

v_add_co_u32 v22, s[18:19], v24, v26 // (op 4744) least significant half
v_addc_co_u32 v23, s[18:19], v25, v27, s[18:19] // (op 4744) most significant half



s_mul_i32 s19, s34, s23 // (op 4746) most significant: low of low * high
s_mul_i32 s15, s35, s22 // (op 4746) most significant: low of high * low
s_mul_i32 s18, s34, s22 // (op 4746) least significant
s_mul_hi_u32 s6, s34, s22 // (op 4746) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4746) most significant: sum
s_add_u32 s19, s19, s6 // (op 4746) most significant: sum


s_mov_b32 s72, s36 // (op 4746) reg expression
s_mov_b32 s73, s37 // (op 4746) reg expression
s_mov_b32 s74, s18 // (op 4746) reg expression
s_mov_b32 s75, 131072 // (op 4746) call()

v_lshrrev_b32 v24, 3, v1 // (op 4763) 
s_mul_i32 s15, s16, s50 // (op 4763) 
s_add_u32 s15, s15, s0 // (op 4763) 
v_and_b32 v25, -8, v1 // (op 4763) 
s_add_u32 s6, s15, s9 // (op 4763) 

v_sub_u32 v26, v1, v25 // (op 4763) 

s_mul_hi_u32 s15, s6, s13 // (op 4763) 
v_lshlrev_b32 v25, 4, v26 // (op 4763) 

s_sub_u32 s7, s6, s15 // (op 4763) 
s_lshr_b32 s8, s7, 1 // (op 4763) 

s_add_u32 s7, s8, s15 // (op 4763) 


s_lshr_b32 s8, s7, s11 // (op 4763) 

s_and_b32 s7, s11, 2147483648 // (op 4763) 
s_cmp_eq_i32 s7, 0 // (op 4763) 
s_mov_b32 s15, scc // (op 4763) 

s_cmp_eq_i32 s15, 0 // (op 4763) 
s_cselect_b32 s7, s6, s8 // (op 4763) 


s_mul_hi_u32 s8, s7, s12 // (op 4763) 
s_sub_u32 s15, s7, s8 // (op 4763) 
s_lshr_b32 s18, s15, 1 // (op 4763) 

s_add_u32 s15, s18, s8 // (op 4763) 


s_lshr_b32 s8, s15, s10 // (op 4763) 

s_and_b32 s15, s10, 2147483648 // (op 4763) 
s_cmp_eq_i32 s15, 0 // (op 4763) 
s_mov_b32 s18, scc // (op 4763) 

s_cmp_eq_i32 s18, 0 // (op 4763) 
s_cselect_b32 s15, s7, s8 // (op 4763) 


s_mul_i32 s8, s15, s14 // (op 4763) 

s_sub_u32 s15, s7, s8 // (op 4763) 

v_lshl_add_u32 v26, s15, 6, v24 // (op 4763) 


v_mul_lo_u32 v29, v26, s35 // (op 4763) most significant: low of low * high
v_mul_hi_u32 v24, v26, s34 // (op 4763) most significant: high of low * low
v_mul_lo_u32 v28, v26, s34 // (op 4763) least significant: low of low * low
v_add_u32 v29, v29, v24 // (op 4763) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4763) 
s_mul_i32 s8, s7, s18 // (op 4763) 


s_sub_u32 s7, s6, s8 // (op 4763) 


v_lshl_add_u32 v24, s7, 7, v25 // (op 4763) 


v_mov_b32 v27, 0 // (op 4763) convert
v_mov_b32 v26, v24 // (op 4763) convert

v_add_co_u32 v24, s[18:19], v26, v28 // (op 4763) least significant half
v_addc_co_u32 v25, s[18:19], v27, v29, s[18:19] // (op 4763) most significant half



s_mul_i32 s19, s34, s23 // (op 4765) most significant: low of low * high
s_mul_i32 s15, s35, s22 // (op 4765) most significant: low of high * low
s_mul_i32 s18, s34, s22 // (op 4765) least significant
s_mul_hi_u32 s6, s34, s22 // (op 4765) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4765) most significant: sum
s_add_u32 s19, s19, s6 // (op 4765) most significant: sum


s_mov_b32 s72, s36 // (op 4765) reg expression
s_mov_b32 s73, s37 // (op 4765) reg expression
s_mov_b32 s74, s18 // (op 4765) reg expression
s_mov_b32 s75, 131072 // (op 4765) call()

v_and_b32 v26, -64, v1 // (op 4872) 
s_mul_i32 s15, s16, s50 // (op 4872) 
s_add_u32 s15, s15, s0 // (op 4872) 
v_sub_u32 v27, v1, v26 // (op 4872) 

s_add_u32 s6, s15, s9 // (op 4872) 

v_lshrrev_b32 v26, 6, v1 // (op 4872) 
s_mul_hi_u32 s15, s6, s13 // (op 4872) 
s_sub_u32 s7, s6, s15 // (op 4872) 
s_lshr_b32 s8, s7, 1 // (op 4872) 

s_add_u32 s7, s8, s15 // (op 4872) 


s_lshr_b32 s8, s7, s11 // (op 4872) 

s_and_b32 s7, s11, 2147483648 // (op 4872) 
s_cmp_eq_i32 s7, 0 // (op 4872) 
s_mov_b32 s15, scc // (op 4872) 

s_cmp_eq_i32 s15, 0 // (op 4872) 
s_cselect_b32 s7, s6, s8 // (op 4872) 


s_mul_hi_u32 s8, s7, s12 // (op 4872) 
s_sub_u32 s15, s7, s8 // (op 4872) 
s_lshr_b32 s18, s15, 1 // (op 4872) 

s_add_u32 s15, s18, s8 // (op 4872) 


s_lshr_b32 s8, s15, s10 // (op 4872) 

s_and_b32 s15, s10, 2147483648 // (op 4872) 
s_cmp_eq_i32 s15, 0 // (op 4872) 
s_mov_b32 s18, scc // (op 4872) 

s_cmp_eq_i32 s18, 0 // (op 4872) 
s_cselect_b32 s15, s7, s8 // (op 4872) 


v_lshl_add_u32 v28, s15, 6, v27 // (op 4872) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4872) 
s_mul_i32 s8, s7, s18 // (op 4872) 


s_sub_u32 s7, s6, s8 // (op 4872) 


v_lshl_add_u32 v27, s7, 2, v26 // (op 4872) 


v_mul_lo_u32 v26, v27, s38 // (op 4872) 

v_add_u32 v27, v28, v26 // (op 4872) 


s_mul_i32 s19, s38, s43 // (op 4874) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 4874) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 4874) least significant
s_mul_hi_u32 s6, s38, s42 // (op 4874) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4874) most significant: sum
s_add_u32 s19, s19, s6 // (op 4874) most significant: sum


s_mov_b32 s76, s40 // (op 4874) reg expression
s_mov_b32 s77, s41 // (op 4874) reg expression
s_mov_b32 s78, s18 // (op 4874) reg expression
s_mov_b32 s79, 131072 // (op 4874) call()

v_and_b32 v26, -64, v1 // (op 4891) 
s_mul_i32 s15, s16, s50 // (op 4891) 
s_add_u32 s15, s15, s0 // (op 4891) 
v_sub_u32 v28, v1, v26 // (op 4891) 

s_add_u32 s6, s15, s9 // (op 4891) 

v_lshrrev_b32 v26, 6, v1 // (op 4891) 
s_mul_hi_u32 s15, s6, s13 // (op 4891) 
s_sub_u32 s7, s6, s15 // (op 4891) 
s_lshr_b32 s8, s7, 1 // (op 4891) 

s_add_u32 s7, s8, s15 // (op 4891) 


s_lshr_b32 s8, s7, s11 // (op 4891) 

s_and_b32 s7, s11, 2147483648 // (op 4891) 
s_cmp_eq_i32 s7, 0 // (op 4891) 
s_mov_b32 s15, scc // (op 4891) 

s_cmp_eq_i32 s15, 0 // (op 4891) 
s_cselect_b32 s7, s6, s8 // (op 4891) 


s_mul_hi_u32 s8, s7, s12 // (op 4891) 
s_sub_u32 s15, s7, s8 // (op 4891) 
s_lshr_b32 s18, s15, 1 // (op 4891) 

s_add_u32 s15, s18, s8 // (op 4891) 


s_lshr_b32 s8, s15, s10 // (op 4891) 

s_and_b32 s15, s10, 2147483648 // (op 4891) 
s_cmp_eq_i32 s15, 0 // (op 4891) 
s_mov_b32 s18, scc // (op 4891) 

s_cmp_eq_i32 s18, 0 // (op 4891) 
s_cselect_b32 s15, s7, s8 // (op 4891) 


v_lshl_add_u32 v29, s15, 6, v28 // (op 4891) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4891) 
s_mul_i32 s8, s7, s18 // (op 4891) 


s_sub_u32 s7, s6, s8 // (op 4891) 


v_lshl_add_u32 v28, s7, 2, v26 // (op 4891) 


v_mul_lo_u32 v26, v28, s38 // (op 4891) 

v_add_u32 v28, v29, v26 // (op 4891) 


s_mul_i32 s19, s38, s43 // (op 4893) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 4893) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 4893) least significant
s_mul_hi_u32 s6, s38, s42 // (op 4893) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4893) most significant: sum
s_add_u32 s19, s19, s6 // (op 4893) most significant: sum


s_mov_b32 s76, s40 // (op 4893) reg expression
s_mov_b32 s77, s41 // (op 4893) reg expression
s_mov_b32 s78, s18 // (op 4893) reg expression
s_mov_b32 s79, 131072 // (op 4893) call()

v_lshrrev_b32 v26, 2, v1 // (op 5000) 
s_mul_i32 s15, s16, s50 // (op 5000) 
s_add_u32 s15, s15, s0 // (op 5000) 
v_and_b32 v29, -4, v1 // (op 5000) 
s_add_u32 s6, s15, s9 // (op 5000) 

v_sub_u32 v30, v1, v29 // (op 5000) 

s_mul_hi_u32 s15, s6, s13 // (op 5000) 
s_sub_u32 s7, s6, s15 // (op 5000) 
s_lshr_b32 s8, s7, 1 // (op 5000) 

s_add_u32 s7, s8, s15 // (op 5000) 


s_lshr_b32 s8, s7, s11 // (op 5000) 

s_and_b32 s7, s11, 2147483648 // (op 5000) 
s_cmp_eq_i32 s7, 0 // (op 5000) 
s_mov_b32 s15, scc // (op 5000) 

s_cmp_eq_i32 s15, 0 // (op 5000) 
s_cselect_b32 s7, s6, s8 // (op 5000) 


s_mul_hi_u32 s8, s7, s12 // (op 5000) 
s_sub_u32 s15, s7, s8 // (op 5000) 
s_lshr_b32 s18, s15, 1 // (op 5000) 

s_add_u32 s15, s18, s8 // (op 5000) 


s_lshr_b32 s8, s15, s10 // (op 5000) 

s_and_b32 s15, s10, 2147483648 // (op 5000) 
s_cmp_eq_i32 s15, 0 // (op 5000) 
s_mov_b32 s18, scc // (op 5000) 

s_cmp_eq_i32 s18, 0 // (op 5000) 
s_cselect_b32 s15, s7, s8 // (op 5000) 


s_mul_i32 s8, s15, s14 // (op 5000) 

s_sub_u32 s15, s7, s8 // (op 5000) 

v_lshl_add_u32 v29, s15, 6, v26 // (op 5000) 


v_mul_lo_u32 v26, v29, s44 // (op 5000) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 5000) 
s_mul_i32 s8, s7, s18 // (op 5000) 


s_sub_u32 s7, s6, s8 // (op 5000) 


v_lshl_add_u32 v29, s7, 2, v30 // (op 5000) 


v_add_u32 v30, v29, v26 // (op 5000) 


s_mul_i32 s19, s44, s23 // (op 5002) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 5002) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 5002) least significant
s_mul_hi_u32 s6, s44, s22 // (op 5002) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 5002) most significant: sum
s_add_u32 s19, s19, s6 // (op 5002) most significant: sum


s_mov_b32 s80, s46 // (op 5002) reg expression
s_mov_b32 s81, s47 // (op 5002) reg expression
s_mov_b32 s82, s18 // (op 5002) reg expression
s_mov_b32 s83, 131072 // (op 5002) call()

v_lshrrev_b32 v26, 2, v1 // (op 5019) 
s_mul_i32 s15, s16, s50 // (op 5019) 
s_add_u32 s15, s15, s0 // (op 5019) 
v_and_b32 v29, -4, v1 // (op 5019) 
s_add_u32 s6, s15, s9 // (op 5019) 

v_sub_u32 v31, v1, v29 // (op 5019) 

s_mul_hi_u32 s15, s6, s13 // (op 5019) 
s_sub_u32 s7, s6, s15 // (op 5019) 
s_lshr_b32 s8, s7, 1 // (op 5019) 

s_add_u32 s7, s8, s15 // (op 5019) 


s_lshr_b32 s8, s7, s11 // (op 5019) 

s_and_b32 s7, s11, 2147483648 // (op 5019) 
s_cmp_eq_i32 s7, 0 // (op 5019) 
s_mov_b32 s15, scc // (op 5019) 

s_cmp_eq_i32 s15, 0 // (op 5019) 
s_cselect_b32 s7, s6, s8 // (op 5019) 


s_mul_hi_u32 s8, s7, s12 // (op 5019) 
s_sub_u32 s15, s7, s8 // (op 5019) 
s_lshr_b32 s18, s15, 1 // (op 5019) 

s_add_u32 s15, s18, s8 // (op 5019) 


s_lshr_b32 s8, s15, s10 // (op 5019) 

s_and_b32 s15, s10, 2147483648 // (op 5019) 
s_cmp_eq_i32 s15, 0 // (op 5019) 
s_mov_b32 s18, scc // (op 5019) 

s_cmp_eq_i32 s18, 0 // (op 5019) 
s_cselect_b32 s15, s7, s8 // (op 5019) 


s_mul_i32 s8, s15, s14 // (op 5019) 

s_sub_u32 s15, s7, s8 // (op 5019) 

v_lshl_add_u32 v29, s15, 6, v26 // (op 5019) 


v_mul_lo_u32 v26, v29, s44 // (op 5019) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 5019) 
s_mul_i32 s8, s7, s18 // (op 5019) 


s_sub_u32 s7, s6, s8 // (op 5019) 


v_lshl_add_u32 v29, s7, 2, v31 // (op 5019) 


v_add_u32 v31, v29, v26 // (op 5019) 


s_mul_i32 s19, s44, s23 // (op 5021) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 5021) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 5021) least significant
s_mul_hi_u32 s6, s44, s22 // (op 5021) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 5021) most significant: sum
s_add_u32 s19, s19, s6 // (op 5021) most significant: sum


s_mov_b32 s80, s46 // (op 5021) reg expression
s_mov_b32 s81, s47 // (op 5021) reg expression
s_mov_b32 s82, s18 // (op 5021) reg expression
s_mov_b32 s83, 131072 // (op 5021) call()

v_and_b32 v26, -8, v1 // (op 5592) 
v_sub_u32 v29, v1, v26 // (op 5592) 

v_lshlrev_b32 v26, 4, v29 // (op 5592) 

v_lshrrev_b32 v29, 3, v1 // (op 5592) 
v_lshl_add_u32 v32, v29, 7, v26 // (op 5592) 


s_nop 0
v_readfirstlane_b32 s15, v32 // (op 5592) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6353) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 4518) 
v_mov_b64 v[32:33], s[18:19] // (op 4518) convert


v_mov_b64 v[34:35], v[22:23] // (op 4518) 
s_mov_b32 m0, s15 // (op 4518) reg expression
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 4518) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 4518) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 4518) most significant half

s_add_u32 m0, m0, 4096 // (op 4518) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 4518) Load value direct to lds






buffer_load_ubyte v26, v30, s[80:83], 0 offen offset: 0 // (op 4166) Load value
v_and_b32 v29, -4, v1 // (op 5517) 
v_sub_u32 v32, v1, v29 // (op 5517) 

v_lshlrev_b32 v29, 4, v32 // (op 5517) 

v_lshrrev_b32 v32, 2, v1 // (op 5517) 
v_lshl_add_u32 v33, v32, 6, v29 // (op 5517) 


s_nop 0
v_readfirstlane_b32 s15, v33 // (op 5517) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6334) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 4523) 
v_mov_b64 v[32:33], s[18:19] // (op 4523) convert


v_mov_b64 v[34:35], v[16:17] // (op 4523) reg expression
s_add_u32 s6, 8192, s15 // (op 4523) 
s_mov_b32 m0, s6 // (op 4523) reg expression
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 4523) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 4523) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 4523) most significant half

s_add_u32 m0, m0, 4096 // (op 4523) 
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 4523) Load value direct to lds






v_mov_b32 v29, v27 // (op 4172) reg expression

buffer_load_ubyte v32, v29, s[76:79], 0 offen offset: 0 // (op 4172) Load value

v_and_b32 v29, -8, v1 // (op 5902) 
v_sub_u32 v33, v1, v29 // (op 5902) 

v_lshlrev_b32 v29, 4, v33 // (op 5902) 

v_lshrrev_b32 v33, 3, v1 // (op 5902) 
v_lshl_add_u32 v34, v33, 7, v29 // (op 5902) 


s_nop 0
v_readfirstlane_b32 s15, v34 // (op 5902) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6424) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 4528) 
v_mov_b64 v[34:35], s[18:19] // (op 4528) convert


v_mov_b64 v[36:37], v[22:23] // (op 4528) 
s_add_u32 s6, 16384, s15 // (op 4528) 
s_mov_b32 m0, s6 // (op 4528) reg expression
s_mov_b32 s7, 128 // (op 4528) call()
buffer_load_dwordx4 v36, s[72:75], s7 offen offset: 0 lds // (op 4528) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v34 // (op 4528) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 4528) most significant half

s_add_u32 m0, m0, 4096 // (op 4528) 
buffer_load_dwordx4 v36, s[72:75], s7 offen offset: 0 lds // (op 4528) Load value direct to lds









buffer_load_ubyte v29, v30, s[80:83], 0 offen offset:4 // (op 4178) Load value

v_and_b32 v30, -4, v1 // (op 5858) 
v_sub_u32 v22, v1, v30 // (op 5858) 

v_lshlrev_b32 v23, 4, v22 // (op 5858) 

v_lshrrev_b32 v22, 2, v1 // (op 5858) 
v_lshl_add_u32 v30, v22, 6, v23 // (op 5858) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 5858) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6414) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 4530) 
v_mov_b64 v[22:23], s[18:19] // (op 4530) convert


s_lshl_b64 s[18:19], s[30:31], 7 // (op 4530) 
v_mov_b32 v30, s18 // (op 4530) 
v_mov_b32 v33, s19 // (op 4530) 
v_add_co_u32 v34, s[28:29], v16, v30 // (op 4530) least significant half
v_addc_co_u32 v35, s[28:29], v17, v33, s[28:29] // (op 4530) most significant half




s_add_u32 s6, 24576, s15 // (op 4530) 
s_mov_b32 m0, s6 // (op 4530) reg expression
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 4530) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v22 // (op 4530) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 4530) most significant half

s_add_u32 m0, m0, 4096 // (op 4530) 
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 4530) Load value direct to lds







s_lshl_b32 s15, s38, 2 // (op 4184) 
v_add_u32 v30, v27, s15 // (op 4184) 


buffer_load_ubyte v16, v30, s[76:79], 0 offen offset: 0 // (op 4184) Load value


v_and_b32 v17, -4, v1 // (op 5744) 
v_sub_u32 v27, v1, v17 // (op 5744) 

v_and_b32 v17, 4294967292, v1 // (op 5744) 
v_add_u32 v30, v17, v27 // (op 5744) 


s_barrier  // (op 6394) 



s_waitcnt vmcnt(9)
ds_write_b8 v30, v26 offset:32768 // (op 4187) Store local data 

v_and_b32 v17, -64, v1 // (op 5668) 
v_sub_u32 v27, v1, v17 // (op 5668) 

v_and_b32 v17, 4294967232, v1 // (op 5668) 
v_add_u32 v30, v17, v27 // (op 5668) 


s_barrier  // (op 6373) 



s_waitcnt vmcnt(6)
ds_write_b8 v30, v32 offset:33024 // (op 4189) Store local data 

s_barrier  // (op 6350) 



ds_read_b128 v[34:37], v14 // (op 4214) Load local data 
ds_read_b128 v[38:41], v14 offset:64 // (op 4214) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 6391) 



ds_read_u8 v17, v15 offset:32768 // (op 4234) Load local data 



ds_read_b128 v[42:45], v14 offset:4096 // (op 4219) Load local data 
ds_read_b128 v[46:49], v14 offset:4160 // (op 4219) Load local data 



ds_read_u8 v27, v15 offset:32896 // (op 4239) Load local data 
s_barrier  // (op 6331) 



ds_read_b64_tr_b8 v[50:51], v0 offset:8192 // (op 4204) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:8704 // (op 4204) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:12288 // (op 4204) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:12800 // (op 4204) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 6370) 


ds_read_u8 v30, v3 offset:33024 // (op 4224) Load local data 



ds_read_b64_tr_b8 v[58:59], v0 offset:8224 // (op 4209) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:8736 // (op 4209) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:12320 // (op 4209) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v0 offset:12832 // (op 4209) Transpose load local data 


ds_read_u8 v33, v3 offset:33056 // (op 4229) Load local data 
v_and_b32 v22, -4, v1 // (op 5489) 
v_sub_u32 v23, v1, v22 // (op 5489) 

v_lshlrev_b32 v22, 4, v23 // (op 5489) 

v_lshrrev_b32 v23, 2, v1 // (op 5489) 
v_lshl_add_u32 v66, v23, 6, v22 // (op 5489) 


s_nop 0
v_readfirstlane_b32 s15, v66 // (op 5489) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -8, v1 // (op 5564) 
v_sub_u32 v23, v1, v22 // (op 5564) 

v_lshlrev_b32 v22, 4, v23 // (op 5564) 

v_lshrrev_b32 v23, 3, v1 // (op 5564) 
v_lshl_add_u32 v66, v23, 7, v22 // (op 5564) 


s_nop 0
v_readfirstlane_b32 s6, v66 // (op 5564) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -64, v1 // (op 5639) 
v_sub_u32 v23, v1, v22 // (op 5639) 

v_and_b32 v22, 4294967232, v1 // (op 5639) 
v_add_u32 v66, v22, v23 // (op 5639) 


v_and_b32 v22, -4, v1 // (op 5715) 
v_sub_u32 v23, v1, v22 // (op 5715) 

v_and_b32 v22, 4294967292, v1 // (op 5715) 
v_add_u32 v67, v22, v23 // (op 5715) 


v_and_b32 v22, -4, v1 // (op 5843) 
v_sub_u32 v23, v1, v22 // (op 5843) 

v_lshlrev_b32 v22, 4, v23 // (op 5843) 

v_lshrrev_b32 v23, 2, v1 // (op 5843) 
v_lshl_add_u32 v68, v23, 6, v22 // (op 5843) 


s_nop 0
v_readfirstlane_b32 s7, v68 // (op 5843) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -8, v1 // (op 5887) 
v_sub_u32 v23, v1, v22 // (op 5887) 

v_lshlrev_b32 v22, 4, v23 // (op 5887) 

v_lshrrev_b32 v23, 3, v1 // (op 5887) 
v_lshl_add_u32 v68, v23, 7, v22 // (op 5887) 


s_nop 0
v_readfirstlane_b32 s8, v68 // (op 5887) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -64, v1 // (op 5931) 
v_sub_u32 v23, v1, v22 // (op 5931) 

v_and_b32 v22, 4294967232, v1 // (op 5931) 
v_add_u32 v68, v22, v23 // (op 5931) 


v_and_b32 v22, -4, v1 // (op 5960) 
v_sub_u32 v23, v1, v22 // (op 5960) 

v_and_b32 v22, 4294967292, v1 // (op 5960) 
v_add_u32 v69, v22, v23 // (op 5960) 


s_mov_b32 s1, 0 // (op 875) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 874) 
s_mov_b32 s28, s18 // (op 874) convert

s_and_b32 s29, s28, 4294967294 // (op 874) 

s_cmp_lt_u32 s1, s29 // (op 874) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_874 // (op 874) Condition: Top (jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_874 if false)
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoop_874:
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:57], v[34:41], a[0:3], v30, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 198) 
s_barrier  // (op 6356) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[42:49], a[4:7], v30, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 662) 
s_add_u32 s1, s1, 2 // (op 876) 
s_lshl_b64 s[18:19], s[34:35], 5 // (op 4467) 
v_mov_b64 v[22:23], s[18:19] // (op 4467) convert


v_mov_b64 v[70:71], v[24:25] // (op 4467) 
v_mov_b64 v[72:73], v[70:71] // (op 4467) 
s_mov_b32 m0, s6 // (op 4467) reg expression
s_mov_b32 s18, 256 // (op 4467) call()
buffer_load_dwordx4 v72, s[72:75], s18 offen offset: 0 lds // (op 4467) Load value direct to lds
v_add_co_u32 v72, s[28:29], v72, v22 // (op 4467) least significant half
v_addc_co_u32 v73, s[28:29], 0, v73, s[28:29] // (op 4467) most significant half

s_add_u32 m0, m0, 4096 // (op 4467) 
buffer_load_dwordx4 v72, s[72:75], s18 offen offset: 0 lds // (op 4467) Load value direct to lds



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[58:65], v[34:41], a[8:11], v33, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 959) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:65], v[42:49], a[12:15], v33, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 913) 
v_mov_b32 v22, v31 // (op 204) 

buffer_load_ubyte v26, v22, s[80:83], 0 offen offset:8 // (op 204) Load value
s_barrier  // (op 6337) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 4462) 
v_mov_b64 v[72:73], s[18:19] // (op 4462) convert


v_mov_b64 v[74:75], v[20:21] // (op 4462) 
s_lshl_b64 s[18:19], s[30:31], 7 // (op 4462) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 4462) 

v_mov_b32 v23, s28 // (op 4462) 
v_mov_b32 v76, s29 // (op 4462) 
v_add_co_u32 v78, s[18:19], v74, v23 // (op 4462) least significant half
v_addc_co_u32 v79, s[18:19], v75, v76, s[18:19] // (op 4462) most significant half




s_add_u32 s18, 8192, s15 // (op 4462) 
s_nop 0
s_mov_b32 m0, s18 // (op 4462) reg expression
buffer_load_dwordx4 v78, s[68:71], 0 offen offset: 0 lds // (op 4462) Load value direct to lds
v_add_co_u32 v78, s[28:29], v78, v72 // (op 4462) least significant half
v_addc_co_u32 v79, s[28:29], 0, v79, s[28:29] // (op 4462) most significant half

s_add_u32 m0, m0, 4096 // (op 4462) 
buffer_load_dwordx4 v78, s[68:71], 0 offen offset: 0 lds // (op 4462) Load value direct to lds





v_mov_b32 v23, v28 // (op 203) 
s_lshl_b32 s18, s38, 3 // (op 203) 
v_add_u32 v72, v23, s18 // (op 203) 


buffer_load_ubyte v32, v72, s[76:79], 0 offen offset: 0 // (op 203) Load value

s_barrier  // (op 6440) 



s_waitcnt vmcnt(9)
ds_write_b8 v69, v29 offset:33280 // (op 541) Store local data 
s_barrier  // (op 6434) 



s_waitcnt vmcnt(6)
ds_write_b8 v68, v16 offset:33536 // (op 540) Store local data 
s_barrier  // (op 6420) 



ds_read_b128 v[76:79], v12 offset:16384 // (op 533) Load local data 
ds_read_b128 v[80:83], v12 offset:16448 // (op 533) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 6437) 



ds_read_u8 v72, v13 offset:33280 // (op 537) Load local data 



ds_read_b128 v[84:87], v12 offset:20480 // (op 667) Load local data 
ds_read_b128 v[88:91], v12 offset:20544 // (op 667) Load local data 



ds_read_u8 v73, v13 offset:33408 // (op 671) Load local data 
s_barrier  // (op 6411) 
s_barrier  // (op 6559) 



ds_read_b64_tr_b8 v[50:51], v10 offset:24576 // (op 531) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v10 offset:25088 // (op 531) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v10 offset:28672 // (op 531) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v10 offset:29184 // (op 531) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 6430) 


ds_read_u8 v30, v11 offset:33536 // (op 535) Load local data 



ds_read_b64_tr_b8 v[58:59], v10 offset:24608 // (op 916) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v10 offset:25120 // (op 916) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v10 offset:28704 // (op 916) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v10 offset:29216 // (op 916) Transpose load local data 


ds_read_u8 v33, v11 offset:33568 // (op 919) Load local data 
s_barrier  // (op 6581) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:57], v[76:83], a[0:3], v30, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 538) 
s_barrier  // (op 6427) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[84:91], a[4:7], v30, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 663) 
s_lshl_b64 s[18:19], s[34:35], 5 // (op 4490) 
v_mov_b64 v[92:93], s[18:19] // (op 4490) convert


v_mov_b64 v[94:95], v[70:71] // (op 4490) 
s_add_u32 s18, 16384, s8 // (op 4490) 
s_mov_b32 m0, s18 // (op 4490) reg expression
s_mov_b32 s19, 384 // (op 4490) call()
buffer_load_dwordx4 v94, s[72:75], s19 offen offset: 0 lds // (op 4490) Load value direct to lds
v_add_co_u32 v94, s[28:29], v94, v92 // (op 4490) least significant half
v_addc_co_u32 v95, s[28:29], 0, v95, s[28:29] // (op 4490) most significant half

s_add_u32 m0, m0, 4096 // (op 4490) 
buffer_load_dwordx4 v94, s[72:75], s19 offen offset: 0 lds // (op 4490) Load value direct to lds




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[58:65], v[76:83], a[8:11], v33, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 960) 






v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:65], v[84:91], a[12:15], v33, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 914) 
buffer_load_ubyte v29, v22, s[80:83], 0 offen offset:12 // (op 544) Load value



v_add_co_u32 v24, vcc, 256, v24 // (op 4755) least significant half
s_barrier  // (op 6417) 
v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 4755) most significant half

v_add_u32 v31, 8, v31 // (op 5011) 
s_lshl_b64 s[18:19], s[30:31], 6 // (op 4485) 
v_mov_b64 v[70:71], s[18:19] // (op 4485) convert


s_lshl_b64 s[18:19], s[30:31], 7 // (op 4485) 
s_mul_i32 s29, 3, s19 // (op 4485) most significant: low of low * high
s_mul_i32 s28, 3, s18 // (op 4485) least significant
s_mul_hi_u32 s48, 3, s18 // (op 4485) most significant: high of low * low
s_add_u32 s29, s29, s48 // (op 4485) most significant: sum


v_mov_b32 v22, s28 // (op 4485) 
v_mov_b32 v72, s29 // (op 4485) 
v_add_co_u32 v76, s[18:19], v74, v22 // (op 4485) least significant half
v_addc_co_u32 v77, s[18:19], v75, v72, s[18:19] // (op 4485) most significant half




s_add_u32 s18, 24576, s7 // (op 4485) 
s_nop 0
s_mov_b32 m0, s18 // (op 4485) reg expression
buffer_load_dwordx4 v76, s[68:71], 0 offen offset: 0 lds // (op 4485) Load value direct to lds
v_add_co_u32 v76, s[28:29], v76, v70 // (op 4485) least significant half
v_addc_co_u32 v77, s[28:29], 0, v77, s[28:29] // (op 4485) most significant half

s_add_u32 m0, m0, 4096 // (op 4485) 
buffer_load_dwordx4 v76, s[68:71], 0 offen offset: 0 lds // (op 4485) Load value direct to lds






s_lshl_b32 s18, s38, 2 // (op 543) 
s_mul_i32 s19, 3, s18 // (op 543) 

v_add_u32 v22, v23, s19 // (op 543) 


s_lshl_b64 s[18:19], s[30:31], 7 // (op 4621) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 4621) 

v_mov_b32 v70, s28 // (op 4621) 
v_mov_b32 v71, s29 // (op 4621) 
v_add_co_u32 v20, s[18:19], v20, v70 // (op 4621) least significant half
v_addc_co_u32 v21, s[18:19], v21, v71, s[18:19] // (op 4621) most significant half




buffer_load_ubyte v16, v22, s[76:79], 0 offen offset: 0 // (op 543) Load value


s_barrier  // (op 6397) 
s_lshl_b32 s18, s38, 3 // (op 4883) 



v_add_u32 v28, v28, s18 // (op 4883) 

s_waitcnt vmcnt(9)
ds_write_b8 v67, v26 offset:32768 // (op 201) Store local data 
s_barrier  // (op 6376) 



s_waitcnt vmcnt(6)
ds_write_b8 v66, v32 offset:33024 // (op 200) Store local data 
s_barrier  // (op 6362) 



ds_read_b128 v[34:37], v8 // (op 193) Load local data 
ds_read_b128 v[38:41], v8 offset:64 // (op 193) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 6404) 



ds_read_u8 v17, v9 offset:32768 // (op 197) Load local data 



ds_read_b128 v[42:45], v8 offset:4096 // (op 657) Load local data 
ds_read_b128 v[46:49], v8 offset:4160 // (op 657) Load local data 



ds_read_u8 v27, v9 offset:32896 // (op 661) Load local data 
s_barrier  // (op 6343) 



ds_read_b64_tr_b8 v[50:51], v6 offset:8192 // (op 191) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v6 offset:8704 // (op 191) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v6 offset:12288 // (op 191) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v6 offset:12800 // (op 191) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 6383) 


ds_read_u8 v30, v7 offset:33024 // (op 195) Load local data 



ds_read_b64_tr_b8 v[58:59], v6 offset:8224 // (op 908) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v6 offset:8736 // (op 908) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v6 offset:12320 // (op 908) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v6 offset:12832 // (op 908) Transpose load local data 


ds_read_u8 v33, v7 offset:33056 // (op 911) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 874) 
s_mov_b32 s28, s18 // (op 874) convert

s_and_b32 s29, s28, 4294967294 // (op 874) 

s_cmp_lt_u32 s1, s29 // (op 874) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoop_874 // (op 874) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoop_874 if true)
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_874:














s_barrier  // (op 6565) 
v_and_b32 v28, -4, v1 // (op 4557) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4557) 
v_sub_u32 v20, v1, v28 // (op 4557) 

s_and_b32 s15, s18, 4294967294 // (op 4557) 
v_lshlrev_b32 v28, 4, v20 // (op 4557) 

s_add_u32 s6, s0, s15 // (op 4557) 

v_lshrrev_b32 v20, 2, v1 // (op 4557) 
s_mul_i32 s15, s16, s50 // (op 4557) 
s_add_u32 s15, s15, s6 // (op 4557) 

s_add_u32 s6, s15, s9 // (op 4557) 

s_mul_hi_u32 s15, s6, s13 // (op 4557) 
s_sub_u32 s7, s6, s15 // (op 4557) 
s_lshr_b32 s8, s7, 1 // (op 4557) 

s_add_u32 s7, s8, s15 // (op 4557) 


s_lshr_b32 s8, s7, s11 // (op 4557) 

s_and_b32 s7, s11, 2147483648 // (op 4557) 
s_cmp_eq_i32 s7, 0 // (op 4557) 
s_mov_b32 s15, scc // (op 4557) 

s_cmp_eq_i32 s15, 0 // (op 4557) 
s_cselect_b32 s7, s6, s8 // (op 4557) 


s_mul_hi_u32 s8, s7, s12 // (op 4557) 
s_sub_u32 s15, s7, s8 // (op 4557) 
s_lshr_b32 s28, s15, 1 // (op 4557) 

s_add_u32 s15, s28, s8 // (op 4557) 


s_lshr_b32 s8, s15, s10 // (op 4557) 

s_and_b32 s15, s10, 2147483648 // (op 4557) 
s_cmp_eq_i32 s15, 0 // (op 4557) 
s_mov_b32 s28, scc // (op 4557) 

s_cmp_eq_i32 s28, 0 // (op 4557) 
s_cselect_b32 s15, s7, s8 // (op 4557) 


s_mul_i32 s8, s7, s18 // (op 4557) 


v_lshl_add_u32 v21, s15, 6, v28 // (op 4557) 


s_sub_u32 s7, s6, s8 // (op 4557) 


v_mov_b32 v23, 0 // (op 4557) convert
v_mov_b32 v22, v21 // (op 4557) convert

v_lshl_add_u32 v21, s7, 7, v20 // (op 4557) 


v_mul_lo_u32 v25, v21, s31 // (op 4557) most significant: low of low * high
v_mul_hi_u32 v20, v21, s30 // (op 4557) most significant: high of low * low
v_mul_lo_u32 v24, v21, s30 // (op 4557) least significant: low of low * low
v_add_u32 v25, v25, v20 // (op 4557) most significant: sum


v_add_co_u32 v20, s[18:19], v22, v24 // (op 4557) least significant half
v_addc_co_u32 v21, s[18:19], v23, v25, s[18:19] // (op 4557) most significant half



s_mul_i32 s19, s30, s3 // (op 4559) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 4559) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 4559) least significant
s_mul_hi_u32 s6, s30, s2 // (op 4559) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4559) most significant: sum
s_add_u32 s19, s19, s6 // (op 4559) most significant: sum


s_mov_b32 s68, s32 // (op 4559) reg expression
s_mov_b32 s69, s33 // (op 4559) reg expression
s_mov_b32 s70, s18 // (op 4559) reg expression
s_mov_b32 s71, 131072 // (op 4559) call()

v_lshrrev_b32 v28, 3, v1 // (op 4699) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4699) 
v_and_b32 v22, -8, v1 // (op 4699) 
s_and_b32 s15, s18, 4294967294 // (op 4699) 
v_sub_u32 v23, v1, v22 // (op 4699) 

s_add_u32 s6, s0, s15 // (op 4699) 

v_lshlrev_b32 v22, 4, v23 // (op 4699) 

s_mul_i32 s15, s16, s50 // (op 4699) 
s_add_u32 s15, s15, s6 // (op 4699) 

s_add_u32 s6, s15, s9 // (op 4699) 

s_mul_hi_u32 s15, s6, s13 // (op 4699) 
s_sub_u32 s7, s6, s15 // (op 4699) 
s_lshr_b32 s8, s7, 1 // (op 4699) 

s_add_u32 s7, s8, s15 // (op 4699) 


s_lshr_b32 s8, s7, s11 // (op 4699) 

s_and_b32 s7, s11, 2147483648 // (op 4699) 
s_cmp_eq_i32 s7, 0 // (op 4699) 
s_mov_b32 s15, scc // (op 4699) 

s_cmp_eq_i32 s15, 0 // (op 4699) 
s_cselect_b32 s7, s6, s8 // (op 4699) 


s_mul_hi_u32 s8, s7, s12 // (op 4699) 
s_sub_u32 s15, s7, s8 // (op 4699) 
s_lshr_b32 s28, s15, 1 // (op 4699) 

s_add_u32 s15, s28, s8 // (op 4699) 


s_lshr_b32 s8, s15, s10 // (op 4699) 

s_and_b32 s15, s10, 2147483648 // (op 4699) 
s_cmp_eq_i32 s15, 0 // (op 4699) 
s_mov_b32 s28, scc // (op 4699) 

s_cmp_eq_i32 s28, 0 // (op 4699) 
s_cselect_b32 s15, s7, s8 // (op 4699) 


s_mul_i32 s8, s15, s14 // (op 4699) 

s_sub_u32 s15, s7, s8 // (op 4699) 

s_mul_i32 s8, s7, s18 // (op 4699) 


v_lshl_add_u32 v23, s15, 6, v28 // (op 4699) 


s_sub_u32 s7, s6, s8 // (op 4699) 


v_mul_lo_u32 v25, v23, s35 // (op 4699) most significant: low of low * high
v_mul_hi_u32 v26, v23, s34 // (op 4699) most significant: high of low * low
v_mul_lo_u32 v24, v23, s34 // (op 4699) least significant: low of low * low
v_add_u32 v25, v25, v26 // (op 4699) most significant: sum


v_lshl_add_u32 v23, s7, 7, v22 // (op 4699) 


v_mov_b32 v67, 0 // (op 4699) convert
v_mov_b32 v66, v23 // (op 4699) convert

v_add_co_u32 v22, s[18:19], v66, v24 // (op 4699) least significant half
v_addc_co_u32 v23, s[18:19], v67, v25, s[18:19] // (op 4699) most significant half



s_mul_i32 s19, s34, s23 // (op 4701) most significant: low of low * high
s_mul_i32 s15, s35, s22 // (op 4701) most significant: low of high * low
s_mul_i32 s18, s34, s22 // (op 4701) least significant
s_mul_hi_u32 s6, s34, s22 // (op 4701) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4701) most significant: sum
s_add_u32 s19, s19, s6 // (op 4701) most significant: sum


s_mov_b32 s72, s36 // (op 4701) reg expression
s_mov_b32 s73, s37 // (op 4701) reg expression
s_mov_b32 s74, s18 // (op 4701) reg expression
s_mov_b32 s75, 131072 // (op 4701) call()

v_and_b32 v28, -64, v1 // (op 4827) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4827) 
v_sub_u32 v24, v1, v28 // (op 4827) 

s_and_b32 s15, s18, 4294967294 // (op 4827) 
v_lshrrev_b32 v28, 6, v1 // (op 4827) 
s_add_u32 s6, s0, s15 // (op 4827) 

s_mul_i32 s15, s16, s50 // (op 4827) 
s_add_u32 s15, s15, s6 // (op 4827) 

s_add_u32 s6, s15, s9 // (op 4827) 

s_mul_hi_u32 s15, s6, s13 // (op 4827) 
s_sub_u32 s7, s6, s15 // (op 4827) 
s_lshr_b32 s8, s7, 1 // (op 4827) 

s_add_u32 s7, s8, s15 // (op 4827) 


s_lshr_b32 s8, s7, s11 // (op 4827) 

s_and_b32 s7, s11, 2147483648 // (op 4827) 
s_cmp_eq_i32 s7, 0 // (op 4827) 
s_mov_b32 s15, scc // (op 4827) 

s_cmp_eq_i32 s15, 0 // (op 4827) 
s_cselect_b32 s7, s6, s8 // (op 4827) 


s_mul_hi_u32 s8, s7, s12 // (op 4827) 
s_sub_u32 s15, s7, s8 // (op 4827) 
s_lshr_b32 s28, s15, 1 // (op 4827) 

s_add_u32 s15, s28, s8 // (op 4827) 


s_lshr_b32 s8, s15, s10 // (op 4827) 

s_and_b32 s15, s10, 2147483648 // (op 4827) 
s_cmp_eq_i32 s15, 0 // (op 4827) 
s_mov_b32 s28, scc // (op 4827) 

s_cmp_eq_i32 s28, 0 // (op 4827) 
s_cselect_b32 s15, s7, s8 // (op 4827) 


s_mul_i32 s8, s7, s18 // (op 4827) 


v_lshl_add_u32 v25, s15, 6, v24 // (op 4827) 


s_sub_u32 s7, s6, s8 // (op 4827) 


v_lshl_add_u32 v24, s7, 2, v28 // (op 4827) 


v_mul_lo_u32 v26, v24, s38 // (op 4827) 

v_add_u32 v24, v25, v26 // (op 4827) 


s_mul_i32 s19, s38, s43 // (op 4829) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 4829) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 4829) least significant
s_mul_hi_u32 s6, s38, s42 // (op 4829) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4829) most significant: sum
s_add_u32 s19, s19, s6 // (op 4829) most significant: sum


s_mov_b32 s76, s40 // (op 4829) reg expression
s_mov_b32 s77, s41 // (op 4829) reg expression
s_mov_b32 s78, s18 // (op 4829) reg expression
s_mov_b32 s79, 131072 // (op 4829) call()

v_lshrrev_b32 v28, 2, v1 // (op 4955) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4955) 
v_and_b32 v25, -4, v1 // (op 4955) 
s_and_b32 s15, s18, 4294967294 // (op 4955) 
v_sub_u32 v26, v1, v25 // (op 4955) 

s_add_u32 s6, s0, s15 // (op 4955) 

s_mul_i32 s15, s16, s50 // (op 4955) 
s_add_u32 s15, s15, s6 // (op 4955) 

s_add_u32 s6, s15, s9 // (op 4955) 

s_mul_hi_u32 s15, s6, s13 // (op 4955) 
s_sub_u32 s7, s6, s15 // (op 4955) 
s_lshr_b32 s8, s7, 1 // (op 4955) 

s_add_u32 s7, s8, s15 // (op 4955) 


s_lshr_b32 s8, s7, s11 // (op 4955) 

s_and_b32 s7, s11, 2147483648 // (op 4955) 
s_cmp_eq_i32 s7, 0 // (op 4955) 
s_mov_b32 s15, scc // (op 4955) 

s_cmp_eq_i32 s15, 0 // (op 4955) 
s_cselect_b32 s7, s6, s8 // (op 4955) 


s_mul_hi_u32 s8, s7, s12 // (op 4955) 
s_sub_u32 s15, s7, s8 // (op 4955) 
s_lshr_b32 s28, s15, 1 // (op 4955) 

s_add_u32 s15, s28, s8 // (op 4955) 


s_lshr_b32 s8, s15, s10 // (op 4955) 

s_and_b32 s15, s10, 2147483648 // (op 4955) 
s_cmp_eq_i32 s15, 0 // (op 4955) 
s_mov_b32 s28, scc // (op 4955) 

s_cmp_eq_i32 s28, 0 // (op 4955) 
s_cselect_b32 s15, s7, s8 // (op 4955) 


s_mul_i32 s8, s15, s14 // (op 4955) 

s_sub_u32 s15, s7, s8 // (op 4955) 

v_lshl_add_u32 v25, s15, 6, v28 // (op 4955) 


s_mul_i32 s8, s7, s18 // (op 4955) 


v_mul_lo_u32 v28, v25, s44 // (op 4955) 

s_sub_u32 s7, s6, s8 // (op 4955) 


v_lshl_add_u32 v25, s7, 2, v26 // (op 4955) 


v_add_u32 v26, v25, v28 // (op 4955) 


s_mul_i32 s19, s44, s23 // (op 4957) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 4957) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 4957) least significant
s_mul_hi_u32 s6, s44, s22 // (op 4957) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4957) most significant: sum
s_add_u32 s19, s19, s6 // (op 4957) most significant: sum


s_mov_b32 s80, s46 // (op 4957) reg expression
s_mov_b32 s81, s47 // (op 4957) reg expression
s_mov_b32 s82, s18 // (op 4957) reg expression
s_mov_b32 s83, 131072 // (op 4957) call()

v_and_b32 v25, -4, v1 // (op 5502) 
v_sub_u32 v28, v1, v25 // (op 5502) 

v_lshlrev_b32 v25, 4, v28 // (op 5502) 

v_lshrrev_b32 v28, 2, v1 // (op 5502) 
v_lshl_add_u32 v31, v28, 6, v25 // (op 5502) 


s_nop 0
v_readfirstlane_b32 s15, v31 // (op 5502) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -8, v1 // (op 5577) 
v_sub_u32 v28, v1, v25 // (op 5577) 

v_lshlrev_b32 v25, 4, v28 // (op 5577) 

v_lshrrev_b32 v28, 3, v1 // (op 5577) 
v_lshl_add_u32 v31, v28, 7, v25 // (op 5577) 


s_nop 0
v_readfirstlane_b32 s6, v31 // (op 5577) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -64, v1 // (op 5652) 
v_sub_u32 v28, v1, v25 // (op 5652) 

v_and_b32 v25, 4294967232, v1 // (op 5652) 
v_add_u32 v31, v25, v28 // (op 5652) 


v_and_b32 v25, -4, v1 // (op 5728) 
v_sub_u32 v28, v1, v25 // (op 5728) 

v_and_b32 v25, 4294967292, v1 // (op 5728) 
v_add_u32 v32, v25, v28 // (op 5728) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 880) 
s_and_b32 s7, s18, 4294967294 // (op 880) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 879) 
s_mov_b32 s8, s18 // (op 879) convert

s_cmp_lt_u32 s7, s8 // (op 879) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_879 // (op 879) Condition: Top (jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_879 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_879:
s_barrier  // (op 6340) 
s_add_u32 s7, s7, 1 // (op 881) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 4472) 
v_mov_b64 v[16:17], s[18:19] // (op 4472) convert


v_mov_b64 v[28:29], v[20:21] // (op 4472) 
v_mov_b64 v[34:35], v[28:29] // (op 4472) reg expression
s_add_u32 s8, 8192, s15 // (op 4472) 
s_mov_b32 m0, s8 // (op 4472) reg expression
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 4472) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v16 // (op 4472) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 4472) most significant half

s_add_u32 m0, m0, 4096 // (op 4472) 
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 4472) Load value direct to lds




s_lshl_b64 s[18:19], s[30:31], 7 // (op 4547) 
v_mov_b32 v25, s18 // (op 4547) 
v_mov_b32 v16, s19 // (op 4547) 
v_add_co_u32 v20, s[28:29], v20, v25 // (op 4547) least significant half
v_addc_co_u32 v21, s[28:29], v21, v16, s[28:29] // (op 4547) most significant half




s_barrier  // (op 6359) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 4478) 
v_mov_b64 v[16:17], s[18:19] // (op 4478) convert


v_mov_b64 v[28:29], v[22:23] // (op 4478) 
v_mov_b64 v[34:35], v[28:29] // (op 4478) 
s_mov_b32 m0, s6 // (op 4478) reg expression
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 4478) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v16 // (op 4478) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 4478) most significant half

s_add_u32 m0, m0, 4096 // (op 4478) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 4478) Load value direct to lds



v_add_co_u32 v22, vcc, 128, v22 // (op 4691) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 4691) most significant half


v_mov_b32 v25, v24 // (op 477) 
v_mov_b32 v16, v25 // (op 477) reg expression

buffer_load_ubyte v17, v16, s[76:79], 0 offen offset: 0 // (op 477) Load value


s_lshl_b32 s8, s38, 2 // (op 4819) 
v_add_u32 v24, v24, s8 // (op 4819) 



s_barrier  // (op 6379) 
v_mov_b32 v16, v26 // (op 478) 




buffer_load_ubyte v25, v16, s[80:83], 0 offen offset: 0 // (op 478) Load value

v_add_u32 v26, 4, v26 // (op 4947) 
s_waitcnt vmcnt(1)
ds_write_b8 v31, v17 offset:33024 // (op 474) Store local data 

s_barrier  // (op 6346) 



s_barrier  // (op 6400) 
ds_read_b64_tr_b8 v[34:35], v6 offset:8192 // (op 465) Transpose load local data 



ds_read_b64_tr_b8 v[36:37], v6 offset:8704 // (op 465) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v6 offset:12288 // (op 465) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v6 offset:12800 // (op 465) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v32, v25 offset:32768 // (op 475) Store local data 

s_barrier  // (op 6366) 



ds_read_b128 v[42:45], v8 // (op 467) Load local data 
ds_read_b128 v[46:49], v8 offset:64 // (op 467) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 6387) 


ds_read_u8 v25, v7 offset:33024 // (op 469) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 6407) 



ds_read_u8 v33, v9 offset:32768 // (op 471) Load local data 



ds_read_b128 v[50:53], v8 offset:4096 // (op 636) Load local data 
ds_read_b128 v[54:57], v8 offset:4160 // (op 636) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[42:49], a[0:3], v25, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 472) 



ds_read_u8 v16, v9 offset:32896 // (op 640) Load local data 



ds_read_b64_tr_b8 v[58:59], v6 offset:8224 // (op 854) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v6 offset:8736 // (op 854) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:41], v[50:57], a[4:7], v25, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 641) 
ds_read_b64_tr_b8 v[62:63], v6 offset:12320 // (op 854) Transpose load local data 


ds_read_b64_tr_b8 v[64:65], v6 offset:12832 // (op 854) Transpose load local data 


ds_read_u8 v17, v7 offset:33056 // (op 857) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[58:65], v[42:49], a[8:11], v17, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 859) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:65], v[50:57], a[12:15], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 890) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 879) 
s_mov_b32 s8, s18 // (op 879) convert

s_cmp_lt_u32 s7, s8 // (op 879) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_879 // (op 879) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_879 if true)
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_879:









s_barrier  // (op 6569) 




v_cmp_eq_f32 vcc, 0.00000, v18 // (op 238) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_238 // (op 238) Condition: False, jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_238
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



v_accvgpr_read v14, a4 // (op 1303) 
v_accvgpr_read v15, a5 // (op 1303) 
v_accvgpr_read v16, a6 // (op 1303) 
v_accvgpr_read v17, a7 // (op 1303) 
v_mul_f32 v20, v19, v14 // (op 1303) 
v_mul_f32 v21, v19, v15 // (op 1303) 
v_mul_f32 v22, v19, v16 // (op 1303) 
v_mul_f32 v23, v19, v17 // (op 1303) 

v_lshrrev_b32 v14, 16, v20 // (op 1305) 
v_lshrrev_b32 v15, 16, v21 // (op 1305) 
v_lshrrev_b32 v16, 16, v22 // (op 1305) 
v_lshrrev_b32 v17, 16, v23 // (op 1305) 


v_pack_b32_f16 v20, v14, v15 // (op 1306) 
v_pack_b32_f16 v21, v16, v17 // (op 1306) 

s_lshl_b32 s15, s24, 6 // (op 1306) 
v_add_u32 v0, v5, s15 // (op 1306) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 1306) Store value



v_accvgpr_read v14, a8 // (op 1329) 
v_accvgpr_read v15, a9 // (op 1329) 
v_accvgpr_read v16, a10 // (op 1329) 
v_accvgpr_read v17, a11 // (op 1329) 
v_mul_f32 v20, v19, v14 // (op 1329) 
v_mul_f32 v21, v19, v15 // (op 1329) 
v_mul_f32 v22, v19, v16 // (op 1329) 
v_mul_f32 v23, v19, v17 // (op 1329) 

v_lshrrev_b32 v14, 16, v20 // (op 1331) 
v_lshrrev_b32 v15, 16, v21 // (op 1331) 
v_lshrrev_b32 v16, 16, v22 // (op 1331) 
v_lshrrev_b32 v17, 16, v23 // (op 1331) 


v_pack_b32_f16 v20, v14, v15 // (op 1333) 
v_pack_b32_f16 v21, v16, v17 // (op 1333) 

v_add_u32 v0, 64, v5 // (op 1333) 

buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 1333) Store value



v_accvgpr_read v14, a12 // (op 1339) 
v_accvgpr_read v15, a13 // (op 1339) 
v_accvgpr_read v16, a14 // (op 1339) 
v_accvgpr_read v17, a15 // (op 1339) 
v_mul_f32 v20, v19, v14 // (op 1339) 
v_mul_f32 v21, v19, v15 // (op 1339) 
v_mul_f32 v22, v19, v16 // (op 1339) 
v_mul_f32 v23, v19, v17 // (op 1339) 

v_lshrrev_b32 v14, 16, v20 // (op 1341) 
v_lshrrev_b32 v15, 16, v21 // (op 1341) 
v_lshrrev_b32 v16, 16, v22 // (op 1341) 
v_lshrrev_b32 v17, 16, v23 // (op 1341) 


v_pack_b32_f16 v20, v14, v15 // (op 1335) 
v_pack_b32_f16 v21, v16, v17 // (op 1335) 

s_lshl_b32 s15, s24, 6 // (op 1335) 
s_add_u32 s15, s15, 64 // (op 1335) 
v_add_u32 v0, v5, s15 // (op 1335) 


buffer_store_dwordx2 v[20:21], v0, s[60:63], 0 offen // (op 1335) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_238 // (op 238) Condition: Done, jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_238
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_238:


v_add_u32 v0, 64, v4 // (op 1425) 

buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 1425) Load value

v_accvgpr_read v20, a8 // (op 1416) 
v_accvgpr_read v21, a9 // (op 1416) 
v_accvgpr_read v22, a10 // (op 1416) 
v_accvgpr_read v23, a11 // (op 1416) 
v_mul_f32 v24, v19, v20 // (op 1416) 
v_mul_f32 v25, v19, v21 // (op 1416) 
v_mul_f32 v26, v19, v22 // (op 1416) 
v_mul_f32 v27, v19, v23 // (op 1416) 

s_waitcnt vmcnt(0)
v_and_b32 v16, 65535, v14 // (op 1424) 
v_lshrrev_b32 v17, 16, v14 // (op 1424) 
v_lshlrev_b32 v16, 16, v16 // (op 1424) 
v_lshlrev_b32 v17, 16, v17 // (op 1424) 
v_mul_f32 v20, v18, v16 // (op 1424) 
v_mul_f32 v21, v18, v17 // (op 1424) 
v_and_b32 v16, 65535, v15 // (op 1424) 
v_lshrrev_b32 v17, 16, v15 // (op 1424) 
v_lshlrev_b32 v16, 16, v16 // (op 1424) 
v_lshlrev_b32 v17, 16, v17 // (op 1424) 
v_mul_f32 v22, v18, v16 // (op 1424) 
v_mul_f32 v23, v18, v17 // (op 1424) 


v_add_f32 v14, v20, v24 // (op 1417) 
v_add_f32 v15, v21, v25 // (op 1417) 
v_add_f32 v16, v22, v26 // (op 1417) 
v_add_f32 v17, v23, v27 // (op 1417) 


v_lshrrev_b32 v20, 16, v14 // (op 1418) 
v_lshrrev_b32 v21, 16, v15 // (op 1418) 
v_lshrrev_b32 v22, 16, v16 // (op 1418) 
v_lshrrev_b32 v23, 16, v17 // (op 1418) 


v_pack_b32_f16 v14, v20, v21 // (op 1420) 
v_pack_b32_f16 v15, v22, v23 // (op 1420) 

v_add_u32 v0, 64, v5 // (op 1420) 

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 1420) Store value





s_lshl_b32 s15, s4, 6 // (op 1430) 
s_add_u32 s15, s15, 64 // (op 1430) 
v_add_u32 v0, v4, s15 // (op 1430) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 1430) Load value

v_accvgpr_read v20, a12 // (op 1432) 
v_accvgpr_read v21, a13 // (op 1432) 
v_accvgpr_read v22, a14 // (op 1432) 
v_accvgpr_read v23, a15 // (op 1432) 
v_mul_f32 v24, v19, v20 // (op 1432) 
v_mul_f32 v25, v19, v21 // (op 1432) 
v_mul_f32 v26, v19, v22 // (op 1432) 
v_mul_f32 v27, v19, v23 // (op 1432) 

s_waitcnt vmcnt(0)
v_and_b32 v16, 65535, v14 // (op 1427) 
v_lshrrev_b32 v17, 16, v14 // (op 1427) 
v_lshlrev_b32 v16, 16, v16 // (op 1427) 
v_lshlrev_b32 v17, 16, v17 // (op 1427) 
v_mul_f32 v20, v18, v16 // (op 1427) 
v_mul_f32 v21, v18, v17 // (op 1427) 
v_and_b32 v16, 65535, v15 // (op 1427) 
v_lshrrev_b32 v17, 16, v15 // (op 1427) 
v_lshlrev_b32 v16, 16, v16 // (op 1427) 
v_lshlrev_b32 v17, 16, v17 // (op 1427) 
v_mul_f32 v22, v18, v16 // (op 1427) 
v_mul_f32 v23, v18, v17 // (op 1427) 


v_add_f32 v14, v20, v24 // (op 1428) 
v_add_f32 v15, v21, v25 // (op 1428) 
v_add_f32 v16, v22, v26 // (op 1428) 
v_add_f32 v17, v23, v27 // (op 1428) 


v_lshrrev_b32 v20, 16, v14 // (op 1429) 
v_lshrrev_b32 v21, 16, v15 // (op 1429) 
v_lshrrev_b32 v22, 16, v16 // (op 1429) 
v_lshrrev_b32 v23, 16, v17 // (op 1429) 


v_pack_b32_f16 v14, v20, v21 // (op 1422) 
v_pack_b32_f16 v15, v22, v23 // (op 1422) 

s_lshl_b32 s15, s24, 6 // (op 1422) 
s_add_u32 s15, s15, 64 // (op 1422) 
v_add_u32 v0, v5, s15 // (op 1422) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 1422) Store value





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





s_lshl_b32 s15, s4, 6 // (op 1385) 
v_add_u32 v0, v4, s15 // (op 1385) 


buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 1385) Load value

v_accvgpr_read v20, a4 // (op 1381) 
v_accvgpr_read v21, a5 // (op 1381) 
v_accvgpr_read v22, a6 // (op 1381) 
v_accvgpr_read v23, a7 // (op 1381) 
v_mul_f32 v24, v19, v20 // (op 1381) 
v_mul_f32 v25, v19, v21 // (op 1381) 
v_mul_f32 v26, v19, v22 // (op 1381) 
v_mul_f32 v27, v19, v23 // (op 1381) 

s_waitcnt vmcnt(0)
v_and_b32 v16, 65535, v14 // (op 1386) 
v_lshrrev_b32 v17, 16, v14 // (op 1386) 
v_lshlrev_b32 v16, 16, v16 // (op 1386) 
v_lshlrev_b32 v17, 16, v17 // (op 1386) 
v_mul_f32 v20, v18, v16 // (op 1386) 
v_mul_f32 v21, v18, v17 // (op 1386) 
v_and_b32 v16, 65535, v15 // (op 1386) 
v_lshrrev_b32 v17, 16, v15 // (op 1386) 
v_lshlrev_b32 v16, 16, v16 // (op 1386) 
v_lshlrev_b32 v17, 16, v17 // (op 1386) 
v_mul_f32 v22, v18, v16 // (op 1386) 
v_mul_f32 v23, v18, v17 // (op 1386) 


v_add_f32 v14, v20, v24 // (op 1382) 
v_add_f32 v15, v21, v25 // (op 1382) 
v_add_f32 v16, v22, v26 // (op 1382) 
v_add_f32 v17, v23, v27 // (op 1382) 


v_lshrrev_b32 v20, 16, v14 // (op 1383) 
v_lshrrev_b32 v21, 16, v15 // (op 1383) 
v_lshrrev_b32 v22, 16, v16 // (op 1383) 
v_lshrrev_b32 v23, 16, v17 // (op 1383) 


v_pack_b32_f16 v14, v20, v21 // (op 1384) 
v_pack_b32_f16 v15, v22, v23 // (op 1384) 

s_lshl_b32 s15, s24, 6 // (op 1384) 
v_add_u32 v0, v5, s15 // (op 1384) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 1384) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_238:
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
s_cbranch_scc1 RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_DPStreamTileLoop_449 // (op 449) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_DPStreamTileLoop_449 if true)
RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_449:













































s_endpgm  // End of RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2
.LRR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_end:
.size RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2, .LRR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_end-RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2
  .amdhsa_next_free_vgpr 124
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 33792 // lds bytes
  .amdhsa_accum_offset 108
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
    DestDataType: b
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
  - .name: RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2
    .symbol: RR_GEMM_NN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 33792
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 89
    .vgpr_count: 106
    .agpr_count: 16
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

