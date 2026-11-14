// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2
.p2align 8
.type RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2,@function
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 5180) 
v_and_b32 v3, 15, v1 // (op 5180) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 5180) 


v_and_b32 v0, 48, v1 // (op 5180) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 5180) 


v_and_b32 v0, 48, v1 // (op 5255) 
v_and_b32 v4, 63, v1 // (op 5255) 
v_sub_u32 v5, v4, v0 // (op 5255) 

v_lshrrev_b32 v4, 1, v5 // (op 5255) 
v_add_lshl_u32 v6, v0, v4, 6 // (op 5255) 


v_and_b32 v0, 1, v5 // (op 5255) 

v_lshlrev_b32 v4, 3, v0 // (op 5255) 

v_lshrrev_b32 v0, 6, v1 // (op 5255) 
v_and_b32 v5, 1, v0 // (op 5255) 

v_lshl_add_u32 v0, v5, 4, v4 // (op 5255) 


v_add_u32 v4, v6, v0 // (op 5255) 


v_lshrrev_b32 v0, 7, v1 // (op 5331) 
v_and_b32 v5, 15, v1 // (op 5331) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 5331) 


v_and_b32 v0, 63, v1 // (op 5331) 
v_lshrrev_b32 v5, 4, v0 // (op 5331) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 5331) 


v_lshrrev_b32 v5, 6, v1 // (op 5407) 
v_and_b32 v6, 1, v5 // (op 5407) 

v_and_b32 v5, 15, v1 // (op 5407) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 5407) 


v_and_b32 v5, 63, v1 // (op 5407) 
v_lshrrev_b32 v6, 4, v5 // (op 5407) 

v_lshl_add_u32 v5, v6, 6, v7 // (op 5407) 


v_lshrrev_b32 v6, 7, v1 // (op 5534) 
v_and_b32 v7, 15, v1 // (op 5534) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 5534) 


v_and_b32 v6, 48, v1 // (op 5534) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 5534) 


v_and_b32 v6, 48, v1 // (op 5609) 
v_and_b32 v8, 63, v1 // (op 5609) 
v_sub_u32 v9, v8, v6 // (op 5609) 

v_lshrrev_b32 v8, 1, v9 // (op 5609) 
v_add_lshl_u32 v10, v6, v8, 6 // (op 5609) 


v_and_b32 v6, 1, v9 // (op 5609) 

v_lshlrev_b32 v8, 3, v6 // (op 5609) 

v_lshrrev_b32 v6, 6, v1 // (op 5609) 
v_and_b32 v9, 1, v6 // (op 5609) 

v_lshl_add_u32 v6, v9, 4, v8 // (op 5609) 


v_add_u32 v8, v10, v6 // (op 5609) 


v_lshrrev_b32 v6, 7, v1 // (op 5685) 
v_and_b32 v9, 15, v1 // (op 5685) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 5685) 


v_and_b32 v6, 63, v1 // (op 5685) 
v_lshrrev_b32 v9, 4, v6 // (op 5685) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 5685) 


v_lshrrev_b32 v9, 6, v1 // (op 5761) 
v_and_b32 v10, 1, v9 // (op 5761) 

v_and_b32 v9, 15, v1 // (op 5761) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 5761) 


v_and_b32 v9, 63, v1 // (op 5761) 
v_lshrrev_b32 v10, 4, v9 // (op 5761) 

v_lshl_add_u32 v9, v10, 6, v11 // (op 5761) 


v_lshrrev_b32 v10, 7, v1 // (op 5875) 
v_and_b32 v11, 15, v1 // (op 5875) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 5875) 


v_and_b32 v10, 48, v1 // (op 5875) 
v_lshl_add_u32 v11, v12, 7, v10 // (op 5875) 


v_and_b32 v10, 48, v1 // (op 5919) 
v_and_b32 v12, 63, v1 // (op 5919) 
v_sub_u32 v13, v12, v10 // (op 5919) 

v_lshrrev_b32 v12, 1, v13 // (op 5919) 
v_add_lshl_u32 v14, v10, v12, 6 // (op 5919) 


v_and_b32 v10, 1, v13 // (op 5919) 

v_lshlrev_b32 v12, 3, v10 // (op 5919) 

v_lshrrev_b32 v10, 6, v1 // (op 5919) 
v_and_b32 v13, 1, v10 // (op 5919) 

v_lshl_add_u32 v10, v13, 4, v12 // (op 5919) 


v_add_u32 v12, v14, v10 // (op 5919) 


v_lshrrev_b32 v10, 7, v1 // (op 5948) 
v_and_b32 v13, 15, v1 // (op 5948) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 5948) 


v_and_b32 v10, 63, v1 // (op 5948) 
v_lshrrev_b32 v13, 4, v10 // (op 5948) 

v_lshl_add_u32 v10, v14, 2, v13 // (op 5948) 


v_lshrrev_b32 v13, 6, v1 // (op 5977) 
v_and_b32 v14, 1, v13 // (op 5977) 

v_and_b32 v13, 15, v1 // (op 5977) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 5977) 


v_and_b32 v13, 63, v1 // (op 5977) 
v_lshrrev_b32 v14, 4, v13 // (op 5977) 

v_lshl_add_u32 v13, v14, 6, v15 // (op 5977) 


v_lshrrev_b32 v14, 7, v1 // (op 6021) 
v_and_b32 v15, 15, v1 // (op 6021) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 6021) 


v_and_b32 v14, 48, v1 // (op 6021) 
v_lshl_add_u32 v15, v16, 7, v14 // (op 6021) 


v_and_b32 v14, 48, v1 // (op 6065) 
v_and_b32 v16, 63, v1 // (op 6065) 
v_sub_u32 v17, v16, v14 // (op 6065) 

v_lshrrev_b32 v16, 1, v17 // (op 6065) 
v_add_lshl_u32 v18, v14, v16, 6 // (op 6065) 


v_and_b32 v14, 1, v17 // (op 6065) 

v_lshlrev_b32 v16, 3, v14 // (op 6065) 

v_lshrrev_b32 v14, 6, v1 // (op 6065) 
v_and_b32 v17, 1, v14 // (op 6065) 

v_lshl_add_u32 v14, v17, 4, v16 // (op 6065) 


v_add_u32 v16, v18, v14 // (op 6065) 


v_lshrrev_b32 v14, 7, v1 // (op 6094) 
v_and_b32 v17, 15, v1 // (op 6094) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 6094) 


v_and_b32 v14, 63, v1 // (op 6094) 
v_lshrrev_b32 v17, 4, v14 // (op 6094) 

v_lshl_add_u32 v14, v18, 2, v17 // (op 6094) 


v_lshrrev_b32 v17, 6, v1 // (op 6123) 
v_and_b32 v18, 1, v17 // (op 6123) 

v_and_b32 v17, 15, v1 // (op 6123) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 6123) 


v_and_b32 v17, 63, v1 // (op 6123) 
v_lshrrev_b32 v18, 4, v17 // (op 6123) 

v_lshl_add_u32 v17, v18, 6, v19 // (op 6123) 


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
s_mul_i32 s54, s8, s16 // (op 294) 
s_cmp_gt_u32 s9, s54 // (op 294) 
s_mov_b32 s55, scc // (op 294) 
s_sub_u32 s56, s9, s54 // (op 294) 

s_cmp_eq_i32 s55, 0 // (op 294) 
s_cselect_b32 s54, 0, s56 // (op 294) 


s_cmp_lt_u32 s8, s54 // (op 294) 
s_mov_b32 s55, scc // (op 294) 
s_cmp_eq_i32 s55, 0 // (op 294) 
s_cselect_b32 s56, s54, s8 // (op 294) 


s_cmp_lt_u32 s0, s56 // (op 294) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_294 // (op 294) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_294 if false)
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_SKStreamTileLoop_294:
v_lshrrev_b32 v20, 7, v1 // (op 5091) 
s_mul_i32 s54, s16, s8 // (op 5091) 
s_add_u32 s54, s54, s0 // (op 5091) 
v_and_b32 v21, 63, v1 // (op 5091) 
s_mul_hi_u32 s55, s54, s13 // (op 5091) 
v_lshrrev_b32 v22, 4, v21 // (op 5091) 
s_sub_u32 s56, s54, s55 // (op 5091) 
v_lshlrev_b32 v23, 2, v22 // (op 5091) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 5091) 


s_lshr_b32 s57, s56, 1 // (op 5091) 

s_add_u32 s56, s57, s55 // (op 5091) 


v_lshrrev_b32 v20, 2, v21 // (op 5091) 

s_lshr_b32 s55, s56, s11 // (op 5091) 

v_and_b32 v21, 3, v20 // (op 5091) 

s_and_b32 s56, s11, 2147483648 // (op 5091) 
v_and_b32 v20, 3, v1 // (op 5091) 
s_cmp_eq_i32 s56, 0 // (op 5091) 
s_mov_b32 s57, scc // (op 5091) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 5091) 


s_cmp_eq_i32 s57, 0 // (op 5091) 
s_cselect_b32 s56, s54, s55 // (op 5091) 



v_lshrrev_b32 v20, 6, v1 // (op 5091) 
s_mul_hi_u32 s54, s56, s12 // (op 5091) 
v_and_b32 v21, 1, v20 // (op 5091) 

s_sub_u32 s55, s56, s54 // (op 5091) 
v_lshl_add_u32 v20, v21, 4, v23 // (op 5091) 


s_lshr_b32 s57, s55, 1 // (op 5091) 

s_add_u32 s55, s57, s54 // (op 5091) 


s_lshr_b32 s54, s55, s10 // (op 5091) 

s_and_b32 s55, s10, 2147483648 // (op 5091) 
s_cmp_eq_i32 s55, 0 // (op 5091) 
s_mov_b32 s57, scc // (op 5091) 

s_cmp_eq_i32 s57, 0 // (op 5091) 
s_cselect_b32 s55, s56, s54 // (op 5091) 


s_mul_i32 s54, s55, s14 // (op 5091) 
v_lshl_add_u32 v21, s55, 6, v22 // (op 5091) 


s_sub_u32 s55, s56, s54 // (op 5091) 


v_lshl_add_u32 v22, s55, 6, v20 // (op 5091) 


v_mul_lo_u32 v20, v22, s4 // (op 5091) 

v_add_lshl_u32 v22, v21, v20, 1 // (op 5091) 


s_mul_i32 s55, s4, s23 // (op 5093) most significant: low of low * high
s_mul_i32 s56, s5, s22 // (op 5093) most significant: low of high * low
s_mul_i32 s54, s4, s22 // (op 5093) least significant
s_mul_hi_u32 s57, s4, s22 // (op 5093) most significant: high of low * low
s_add_u32 s55, s55, s56 // (op 5093) most significant: sum
s_add_u32 s55, s55, s57 // (op 5093) most significant: sum


s_lshl_b64 s[56:57], s[54:55], 1 // (op 5093) 

s_mov_b32 s60, s20 // (op 5093) reg expression
s_mov_b32 s61, s21 // (op 5093) reg expression
s_mov_b32 s62, s56 // (op 5093) reg expression
s_mov_b32 s63, 131072 // (op 5093) call()

v_lshrrev_b32 v20, 7, v1 // (op 5197) 
v_and_b32 v21, 15, v1 // (op 5197) 
v_lshl_add_u32 v23, v20, 4, v21 // (op 5197) 


v_and_b32 v20, 48, v1 // (op 5197) 
v_lshl_add_u32 v21, v23, 7, v20 // (op 5197) 


v_and_b32 v20, 48, v1 // (op 5272) 
v_and_b32 v23, 63, v1 // (op 5272) 
v_sub_u32 v24, v23, v20 // (op 5272) 

v_lshrrev_b32 v23, 1, v24 // (op 5272) 
v_add_lshl_u32 v25, v20, v23, 6 // (op 5272) 


v_and_b32 v20, 1, v24 // (op 5272) 

v_lshlrev_b32 v23, 3, v20 // (op 5272) 

v_lshrrev_b32 v20, 6, v1 // (op 5272) 
v_and_b32 v24, 1, v20 // (op 5272) 

v_lshl_add_u32 v20, v24, 4, v23 // (op 5272) 


v_add_u32 v23, v25, v20 // (op 5272) 


v_lshrrev_b32 v20, 7, v1 // (op 5348) 
v_and_b32 v24, 15, v1 // (op 5348) 
v_lshl_add_u32 v25, v20, 4, v24 // (op 5348) 


v_and_b32 v20, 63, v1 // (op 5348) 
v_lshrrev_b32 v24, 4, v20 // (op 5348) 

v_lshl_add_u32 v20, v25, 2, v24 // (op 5348) 


v_lshrrev_b32 v24, 6, v1 // (op 5424) 
v_and_b32 v25, 1, v24 // (op 5424) 

v_and_b32 v24, 15, v1 // (op 5424) 
v_lshl_add_u32 v26, v25, 4, v24 // (op 5424) 


v_and_b32 v24, 63, v1 // (op 5424) 
v_lshrrev_b32 v25, 4, v24 // (op 5424) 

v_lshl_add_u32 v24, v25, 6, v26 // (op 5424) 


v_lshrrev_b32 v25, 7, v1 // (op 5447) 
s_mul_i32 s54, s16, s8 // (op 5447) 
s_add_u32 s54, s54, s0 // (op 5447) 
v_and_b32 v26, 63, v1 // (op 5447) 
s_mul_hi_u32 s55, s54, s13 // (op 5447) 
v_lshrrev_b32 v27, 4, v26 // (op 5447) 
s_sub_u32 s56, s54, s55 // (op 5447) 
v_lshlrev_b32 v28, 2, v27 // (op 5447) 

s_lshr_b32 s57, s56, 1 // (op 5447) 

v_lshl_add_u32 v27, v25, 4, v28 // (op 5447) 


s_add_u32 s56, s57, s55 // (op 5447) 


v_lshrrev_b32 v25, 2, v26 // (op 5447) 

s_lshr_b32 s55, s56, s11 // (op 5447) 

v_and_b32 v26, 3, v25 // (op 5447) 

s_and_b32 s56, s11, 2147483648 // (op 5447) 
v_and_b32 v25, 3, v1 // (op 5447) 
s_cmp_eq_i32 s56, 0 // (op 5447) 
s_mov_b32 s57, scc // (op 5447) 

v_lshl_add_u32 v28, v26, 2, v25 // (op 5447) 


s_cmp_eq_i32 s57, 0 // (op 5447) 
s_cselect_b32 s56, s54, s55 // (op 5447) 



v_lshrrev_b32 v25, 6, v1 // (op 5447) 
s_mul_hi_u32 s54, s56, s12 // (op 5447) 
v_and_b32 v26, 1, v25 // (op 5447) 

s_sub_u32 s55, s56, s54 // (op 5447) 
v_lshl_add_u32 v25, v26, 4, v28 // (op 5447) 


s_lshr_b32 s57, s55, 1 // (op 5447) 

s_add_u32 s55, s57, s54 // (op 5447) 


s_lshr_b32 s54, s55, s10 // (op 5447) 

s_and_b32 s55, s10, 2147483648 // (op 5447) 
s_cmp_eq_i32 s55, 0 // (op 5447) 
s_mov_b32 s57, scc // (op 5447) 

s_cmp_eq_i32 s57, 0 // (op 5447) 
s_cselect_b32 s55, s56, s54 // (op 5447) 


s_mul_i32 s54, s55, s14 // (op 5447) 
v_lshl_add_u32 v26, s55, 6, v27 // (op 5447) 


s_sub_u32 s55, s56, s54 // (op 5447) 


v_lshl_add_u32 v27, s55, 6, v25 // (op 5447) 


v_mul_lo_u32 v25, v27, s24 // (op 5447) 

v_add_lshl_u32 v27, v26, v25, 1 // (op 5447) 


s_mul_i32 s55, s24, s23 // (op 5449) most significant: low of low * high
s_mul_i32 s56, s25, s22 // (op 5449) most significant: low of high * low
s_mul_i32 s54, s24, s22 // (op 5449) least significant
s_mul_hi_u32 s57, s24, s22 // (op 5449) most significant: high of low * low
s_add_u32 s55, s55, s56 // (op 5449) most significant: sum
s_add_u32 s55, s55, s57 // (op 5449) most significant: sum


s_lshl_b64 s[56:57], s[54:55], 1 // (op 5449) 

s_mov_b32 s64, s26 // (op 5449) reg expression
s_mov_b32 s65, s27 // (op 5449) reg expression
s_mov_b32 s66, s56 // (op 5449) reg expression
s_mov_b32 s67, 131072 // (op 5449) call()

v_lshrrev_b32 v25, 5, v1 // (op 5801) 
v_lshlrev_b32 v26, 2, v25 // (op 5801) 

v_lshl_add_u32 v25, s16, 6, v26 // (op 5801) 

v_and_b32 v26, -32, v1 // (op 5801) 
v_sub_u32 v28, v1, v26 // (op 5801) 

v_lshl_add_u32 v26, v25, 6, v28 // (op 5801) 


v_lshlrev_b32 v25, 2, v26 // (op 5801) 

s_lshl_b32 s54, s15, 14 // (op 5803) 
s_mov_b32 s56, s28 // (op 5803) reg expression
s_mov_b32 s57, s29 // (op 5803) reg expression
s_mov_b32 s58, s54 // (op 5803) reg expression
s_mov_b32 s59, 131072 // (op 5803) call()

s_mul_i32 s6, s8, s16 // (op 299) 
s_add_u32 s6, s6, s0 // (op 299) 
s_mul_hi_u32 s54, s6, s13 // (op 300) 
s_sub_u32 s55, s6, s54 // (op 300) 
s_lshr_b32 s68, s55, 1 // (op 300) 

s_add_u32 s55, s68, s54 // (op 300) 


s_lshr_b32 s54, s55, s11 // (op 300) 

s_and_b32 s55, s11, 2147483648 // (op 300) 
s_cmp_eq_i32 s55, 0 // (op 300) 
s_mov_b32 s68, scc // (op 300) 

s_cmp_eq_i32 s68, 0 // (op 300) 
s_cselect_b32 s55, s6, s54 // (op 300) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 300) 
s_mul_i32 s54, s55, s68 // (op 300) 


s_sub_u32 s18, s6, s54 // (op 300) 

s_ashr_i64 s[54:55], s[2:3], 7 // (op 301) 
s_cmp_gt_u32 s54, s18 // (op 301) 
s_mov_b32 s68, scc // (op 301) 
s_sub_u32 s69, s54, s18 // (op 301) 

s_cmp_eq_i32 s68, 0 // (op 301) 
s_cselect_b32 s54, 0, s69 // (op 301) 


s_cmp_gt_u32 s8, s0 // (op 301) 
s_mov_b32 s55, scc // (op 301) 
s_sub_u32 s68, s8, s0 // (op 301) 
s_cmp_eq_i32 s55, 0 // (op 301) 
s_cselect_b32 s69, 0, s68 // (op 301) 


s_cmp_lt_u32 s54, s69 // (op 301) 
s_mov_b32 s55, scc // (op 301) 
s_cmp_eq_i32 s55, 0 // (op 301) 
s_cselect_b32 s68, s69, s54 // (op 301) 



s_cmp_gt_u32 s9, s6 // (op 301) 
s_mov_b32 s54, scc // (op 301) 
s_sub_u32 s55, s9, s6 // (op 301) 
s_cmp_eq_i32 s54, 0 // (op 301) 
s_cselect_b32 s69, 0, s55 // (op 301) 


s_cmp_lt_u32 s68, s69 // (op 301) 
s_mov_b32 s54, scc // (op 301) 
s_cmp_eq_i32 s54, 0 // (op 301) 
s_cselect_b32 s7, s69, s68 // (op 301) 



s_add_u32 s54, s18, s7 // (op 302) 
s_sub_u32 s19, s54, 1 // (op 302) 

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
v_lshrrev_b32 v26, 3, v1 // (op 4658) 
s_mul_i32 s54, s16, s8 // (op 4658) 
s_add_u32 s54, s54, s0 // (op 4658) 
v_and_b32 v28, -8, v1 // (op 4658) 
s_mul_hi_u32 s55, s54, s13 // (op 4658) 
v_sub_u32 v29, v1, v28 // (op 4658) 

s_sub_u32 s68, s54, s55 // (op 4658) 
v_lshlrev_b32 v28, 4, v29 // (op 4658) 

s_lshr_b32 s69, s68, 1 // (op 4658) 

s_add_u32 s68, s69, s55 // (op 4658) 


s_lshr_b32 s55, s68, s11 // (op 4658) 

s_and_b32 s68, s11, 2147483648 // (op 4658) 
s_cmp_eq_i32 s68, 0 // (op 4658) 
s_mov_b32 s69, scc // (op 4658) 

s_cmp_eq_i32 s69, 0 // (op 4658) 
s_cselect_b32 s68, s54, s55 // (op 4658) 


s_mul_hi_u32 s55, s68, s12 // (op 4658) 
s_sub_u32 s69, s68, s55 // (op 4658) 
s_lshr_b32 s70, s69, 1 // (op 4658) 

s_add_u32 s69, s70, s55 // (op 4658) 


s_lshr_b32 s55, s69, s10 // (op 4658) 

s_and_b32 s69, s10, 2147483648 // (op 4658) 
s_cmp_eq_i32 s69, 0 // (op 4658) 
s_mov_b32 s70, scc // (op 4658) 

s_cmp_eq_i32 s70, 0 // (op 4658) 
s_cselect_b32 s69, s68, s55 // (op 4658) 


v_lshl_add_u32 v29, s69, 6, v26 // (op 4658) 


s_ashr_i64 s[70:71], s[2:3], 7 // (op 4658) 
s_mul_i32 s55, s68, s70 // (op 4658) 


s_sub_u32 s68, s54, s55 // (op 4658) 


v_lshl_add_u32 v26, s68, 7, v28 // (op 4658) 


v_mov_b32 v31, 0 // (op 4658) convert
v_mov_b32 v30, v26 // (op 4658) convert

v_mul_lo_u32 v33, v29, s31 // (op 4658) most significant: low of low * high
v_mul_hi_u32 v26, v29, s30 // (op 4658) most significant: high of low * low
v_mul_lo_u32 v32, v29, s30 // (op 4658) least significant: low of low * low
v_add_u32 v33, v33, v26 // (op 4658) most significant: sum

v_add_co_u32 v32, s[54:55], v32, v30 // (op 4658) least significant half
v_addc_co_u32 v33, s[54:55], v33, v31, s[54:55] // (op 4658) most significant half



s_mul_i32 s55, s30, s35 // (op 4660) most significant: low of low * high
s_mul_i32 s68, s31, s34 // (op 4660) most significant: low of high * low
s_mul_i32 s54, s30, s34 // (op 4660) least significant
s_mul_hi_u32 s69, s30, s34 // (op 4660) most significant: high of low * low
s_add_u32 s55, s55, s68 // (op 4660) most significant: sum
s_add_u32 s55, s55, s69 // (op 4660) most significant: sum


s_mov_b32 s68, s32 // (op 4660) reg expression
s_mov_b32 s69, s33 // (op 4660) reg expression
s_mov_b32 s70, s54 // (op 4660) reg expression
s_mov_b32 s71, 131072 // (op 4660) call()

v_lshrrev_b32 v26, 3, v1 // (op 4677) 
s_mul_i32 s54, s16, s8 // (op 4677) 
s_add_u32 s54, s54, s0 // (op 4677) 
v_and_b32 v28, -8, v1 // (op 4677) 
s_mul_hi_u32 s55, s54, s13 // (op 4677) 
v_sub_u32 v29, v1, v28 // (op 4677) 

v_lshlrev_b32 v28, 4, v29 // (op 4677) 

s_sub_u32 s72, s54, s55 // (op 4677) 
s_lshr_b32 s73, s72, 1 // (op 4677) 

s_add_u32 s72, s73, s55 // (op 4677) 


s_lshr_b32 s55, s72, s11 // (op 4677) 

s_and_b32 s72, s11, 2147483648 // (op 4677) 
s_cmp_eq_i32 s72, 0 // (op 4677) 
s_mov_b32 s73, scc // (op 4677) 

s_cmp_eq_i32 s73, 0 // (op 4677) 
s_cselect_b32 s72, s54, s55 // (op 4677) 


s_mul_hi_u32 s55, s72, s12 // (op 4677) 
s_sub_u32 s73, s72, s55 // (op 4677) 
s_lshr_b32 s74, s73, 1 // (op 4677) 

s_add_u32 s73, s74, s55 // (op 4677) 


s_lshr_b32 s55, s73, s10 // (op 4677) 

s_and_b32 s73, s10, 2147483648 // (op 4677) 
s_cmp_eq_i32 s73, 0 // (op 4677) 
s_mov_b32 s74, scc // (op 4677) 

s_cmp_eq_i32 s74, 0 // (op 4677) 
s_cselect_b32 s73, s72, s55 // (op 4677) 


v_lshl_add_u32 v29, s73, 6, v26 // (op 4677) 


s_ashr_i64 s[74:75], s[2:3], 7 // (op 4677) 
s_mul_i32 s55, s72, s74 // (op 4677) 


s_sub_u32 s72, s54, s55 // (op 4677) 


v_lshl_add_u32 v26, s72, 7, v28 // (op 4677) 


v_mov_b32 v31, 0 // (op 4677) convert
v_mov_b32 v30, v26 // (op 4677) convert

v_mul_lo_u32 v35, v29, s31 // (op 4677) most significant: low of low * high
v_mul_hi_u32 v26, v29, s30 // (op 4677) most significant: high of low * low
v_mul_lo_u32 v34, v29, s30 // (op 4677) least significant: low of low * low
v_add_u32 v35, v35, v26 // (op 4677) most significant: sum

v_add_co_u32 v34, s[54:55], v34, v30 // (op 4677) least significant half
v_addc_co_u32 v35, s[54:55], v35, v31, s[54:55] // (op 4677) most significant half



s_mul_i32 s55, s30, s35 // (op 4679) most significant: low of low * high
s_mul_i32 s72, s31, s34 // (op 4679) most significant: low of high * low
s_mul_i32 s54, s30, s34 // (op 4679) least significant
s_mul_hi_u32 s73, s30, s34 // (op 4679) most significant: high of low * low
s_add_u32 s55, s55, s72 // (op 4679) most significant: sum
s_add_u32 s55, s55, s73 // (op 4679) most significant: sum


s_mov_b32 s68, s32 // (op 4679) reg expression
s_mov_b32 s69, s33 // (op 4679) reg expression
s_mov_b32 s70, s54 // (op 4679) reg expression
s_mov_b32 s71, 131072 // (op 4679) call()

v_and_b32 v26, -4, v1 // (op 4786) 
s_mul_i32 s54, s16, s8 // (op 4786) 
s_add_u32 s54, s54, s0 // (op 4786) 
v_sub_u32 v28, v1, v26 // (op 4786) 

s_mul_hi_u32 s55, s54, s13 // (op 4786) 
v_lshlrev_b32 v26, 4, v28 // (op 4786) 

s_sub_u32 s72, s54, s55 // (op 4786) 
v_lshrrev_b32 v28, 2, v1 // (op 4786) 
s_lshr_b32 s73, s72, 1 // (op 4786) 

s_add_u32 s72, s73, s55 // (op 4786) 


s_lshr_b32 s55, s72, s11 // (op 4786) 

s_and_b32 s72, s11, 2147483648 // (op 4786) 
s_cmp_eq_i32 s72, 0 // (op 4786) 
s_mov_b32 s73, scc // (op 4786) 

s_cmp_eq_i32 s73, 0 // (op 4786) 
s_cselect_b32 s72, s54, s55 // (op 4786) 


s_mul_hi_u32 s55, s72, s12 // (op 4786) 
s_sub_u32 s73, s72, s55 // (op 4786) 
s_lshr_b32 s74, s73, 1 // (op 4786) 

s_add_u32 s73, s74, s55 // (op 4786) 


s_lshr_b32 s55, s73, s10 // (op 4786) 

s_and_b32 s73, s10, 2147483648 // (op 4786) 
s_cmp_eq_i32 s73, 0 // (op 4786) 
s_mov_b32 s74, scc // (op 4786) 

s_cmp_eq_i32 s74, 0 // (op 4786) 
s_cselect_b32 s73, s72, s55 // (op 4786) 


s_mul_i32 s55, s73, s14 // (op 4786) 

s_sub_u32 s73, s72, s55 // (op 4786) 

v_lshl_add_u32 v29, s73, 6, v26 // (op 4786) 


s_ashr_i64 s[74:75], s[2:3], 7 // (op 4786) 
v_mov_b32 v31, 0 // (op 4786) convert
v_mov_b32 v30, v29 // (op 4786) convert

s_mul_i32 s55, s72, s74 // (op 4786) 


s_sub_u32 s72, s54, s55 // (op 4786) 


v_lshl_add_u32 v26, s72, 7, v28 // (op 4786) 


v_mul_lo_u32 v29, v26, s37 // (op 4786) most significant: low of low * high
v_mul_hi_u32 v36, v26, s36 // (op 4786) most significant: high of low * low
v_mul_lo_u32 v28, v26, s36 // (op 4786) least significant: low of low * low
v_add_u32 v29, v29, v36 // (op 4786) most significant: sum

v_add_co_u32 v28, s[54:55], v28, v30 // (op 4786) least significant half
v_addc_co_u32 v29, s[54:55], v29, v31, s[54:55] // (op 4786) most significant half



s_mul_i32 s55, s36, s3 // (op 4788) most significant: low of low * high
s_mul_i32 s72, s37, s2 // (op 4788) most significant: low of high * low
s_mul_i32 s54, s36, s2 // (op 4788) least significant
s_mul_hi_u32 s73, s36, s2 // (op 4788) most significant: high of low * low
s_add_u32 s55, s55, s72 // (op 4788) most significant: sum
s_add_u32 s55, s55, s73 // (op 4788) most significant: sum


s_mov_b32 s72, s38 // (op 4788) reg expression
s_mov_b32 s73, s39 // (op 4788) reg expression
s_mov_b32 s74, s54 // (op 4788) reg expression
s_mov_b32 s75, 131072 // (op 4788) call()

v_and_b32 v26, -4, v1 // (op 4805) 
s_mul_i32 s54, s16, s8 // (op 4805) 
s_add_u32 s54, s54, s0 // (op 4805) 
v_sub_u32 v30, v1, v26 // (op 4805) 

s_mul_hi_u32 s55, s54, s13 // (op 4805) 
v_lshlrev_b32 v26, 4, v30 // (op 4805) 

v_lshrrev_b32 v30, 2, v1 // (op 4805) 
s_sub_u32 s76, s54, s55 // (op 4805) 
s_lshr_b32 s77, s76, 1 // (op 4805) 

s_add_u32 s76, s77, s55 // (op 4805) 


s_lshr_b32 s55, s76, s11 // (op 4805) 

s_and_b32 s76, s11, 2147483648 // (op 4805) 
s_cmp_eq_i32 s76, 0 // (op 4805) 
s_mov_b32 s77, scc // (op 4805) 

s_cmp_eq_i32 s77, 0 // (op 4805) 
s_cselect_b32 s76, s54, s55 // (op 4805) 


s_mul_hi_u32 s55, s76, s12 // (op 4805) 
s_sub_u32 s77, s76, s55 // (op 4805) 
s_lshr_b32 s78, s77, 1 // (op 4805) 

s_add_u32 s77, s78, s55 // (op 4805) 


s_lshr_b32 s55, s77, s10 // (op 4805) 

s_and_b32 s77, s10, 2147483648 // (op 4805) 
s_cmp_eq_i32 s77, 0 // (op 4805) 
s_mov_b32 s78, scc // (op 4805) 

s_cmp_eq_i32 s78, 0 // (op 4805) 
s_cselect_b32 s77, s76, s55 // (op 4805) 


s_mul_i32 s55, s77, s14 // (op 4805) 

s_sub_u32 s77, s76, s55 // (op 4805) 

v_lshl_add_u32 v31, s77, 6, v26 // (op 4805) 


s_ashr_i64 s[78:79], s[2:3], 7 // (op 4805) 
v_mov_b32 v37, 0 // (op 4805) convert
v_mov_b32 v36, v31 // (op 4805) convert

s_mul_i32 s55, s76, s78 // (op 4805) 


s_sub_u32 s76, s54, s55 // (op 4805) 


v_lshl_add_u32 v26, s76, 7, v30 // (op 4805) 


v_mul_lo_u32 v31, v26, s37 // (op 4805) most significant: low of low * high
v_mul_hi_u32 v38, v26, s36 // (op 4805) most significant: high of low * low
v_mul_lo_u32 v30, v26, s36 // (op 4805) least significant: low of low * low
v_add_u32 v31, v31, v38 // (op 4805) most significant: sum

v_add_co_u32 v30, s[54:55], v30, v36 // (op 4805) least significant half
v_addc_co_u32 v31, s[54:55], v31, v37, s[54:55] // (op 4805) most significant half



s_mul_i32 s55, s36, s3 // (op 4807) most significant: low of low * high
s_mul_i32 s76, s37, s2 // (op 4807) most significant: low of high * low
s_mul_i32 s54, s36, s2 // (op 4807) least significant
s_mul_hi_u32 s77, s36, s2 // (op 4807) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 4807) most significant: sum
s_add_u32 s55, s55, s77 // (op 4807) most significant: sum


s_mov_b32 s72, s38 // (op 4807) reg expression
s_mov_b32 s73, s39 // (op 4807) reg expression
s_mov_b32 s74, s54 // (op 4807) reg expression
s_mov_b32 s75, 131072 // (op 4807) call()

v_lshrrev_b32 v26, 2, v1 // (op 4914) 
s_mul_i32 s54, s16, s8 // (op 4914) 
s_add_u32 s54, s54, s0 // (op 4914) 
v_and_b32 v36, -4, v1 // (op 4914) 
s_mul_hi_u32 s55, s54, s13 // (op 4914) 
v_sub_u32 v37, v1, v36 // (op 4914) 

s_sub_u32 s76, s54, s55 // (op 4914) 
s_lshr_b32 s77, s76, 1 // (op 4914) 

s_add_u32 s76, s77, s55 // (op 4914) 


s_lshr_b32 s55, s76, s11 // (op 4914) 

s_and_b32 s76, s11, 2147483648 // (op 4914) 
s_cmp_eq_i32 s76, 0 // (op 4914) 
s_mov_b32 s77, scc // (op 4914) 

s_cmp_eq_i32 s77, 0 // (op 4914) 
s_cselect_b32 s76, s54, s55 // (op 4914) 


s_mul_hi_u32 s55, s76, s12 // (op 4914) 
s_sub_u32 s77, s76, s55 // (op 4914) 
s_lshr_b32 s78, s77, 1 // (op 4914) 

s_add_u32 s77, s78, s55 // (op 4914) 


s_lshr_b32 s55, s77, s10 // (op 4914) 

s_and_b32 s77, s10, 2147483648 // (op 4914) 
s_cmp_eq_i32 s77, 0 // (op 4914) 
s_mov_b32 s78, scc // (op 4914) 

s_cmp_eq_i32 s78, 0 // (op 4914) 
s_cselect_b32 s77, s76, s55 // (op 4914) 


v_lshl_add_u32 v36, s77, 6, v26 // (op 4914) 


s_ashr_i64 s[78:79], s[2:3], 7 // (op 4914) 
s_mul_i32 s55, s76, s78 // (op 4914) 


s_sub_u32 s76, s54, s55 // (op 4914) 


v_lshl_add_u32 v26, s76, 2, v37 // (op 4914) 


v_mul_lo_u32 v37, v36, s40 // (op 4914) 
v_add_u32 v37, v37, v26 // (op 4914) 


s_mul_i32 s55, s40, s35 // (op 4916) most significant: low of low * high
s_mul_i32 s76, s41, s34 // (op 4916) most significant: low of high * low
s_mul_i32 s54, s40, s34 // (op 4916) least significant
s_mul_hi_u32 s77, s40, s34 // (op 4916) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 4916) most significant: sum
s_add_u32 s55, s55, s77 // (op 4916) most significant: sum


s_mov_b32 s76, s42 // (op 4916) reg expression
s_mov_b32 s77, s43 // (op 4916) reg expression
s_mov_b32 s78, s54 // (op 4916) reg expression
s_mov_b32 s79, 131072 // (op 4916) call()

v_lshrrev_b32 v26, 2, v1 // (op 4933) 
s_mul_i32 s54, s16, s8 // (op 4933) 
s_add_u32 s54, s54, s0 // (op 4933) 
v_and_b32 v36, -4, v1 // (op 4933) 
s_mul_hi_u32 s55, s54, s13 // (op 4933) 
v_sub_u32 v38, v1, v36 // (op 4933) 

s_sub_u32 s80, s54, s55 // (op 4933) 
s_lshr_b32 s81, s80, 1 // (op 4933) 

s_add_u32 s80, s81, s55 // (op 4933) 


s_lshr_b32 s55, s80, s11 // (op 4933) 

s_and_b32 s80, s11, 2147483648 // (op 4933) 
s_cmp_eq_i32 s80, 0 // (op 4933) 
s_mov_b32 s81, scc // (op 4933) 

s_cmp_eq_i32 s81, 0 // (op 4933) 
s_cselect_b32 s80, s54, s55 // (op 4933) 


s_mul_hi_u32 s55, s80, s12 // (op 4933) 
s_sub_u32 s81, s80, s55 // (op 4933) 
s_lshr_b32 s82, s81, 1 // (op 4933) 

s_add_u32 s81, s82, s55 // (op 4933) 


s_lshr_b32 s55, s81, s10 // (op 4933) 

s_and_b32 s81, s10, 2147483648 // (op 4933) 
s_cmp_eq_i32 s81, 0 // (op 4933) 
s_mov_b32 s82, scc // (op 4933) 

s_cmp_eq_i32 s82, 0 // (op 4933) 
s_cselect_b32 s81, s80, s55 // (op 4933) 


v_lshl_add_u32 v36, s81, 6, v26 // (op 4933) 


s_ashr_i64 s[82:83], s[2:3], 7 // (op 4933) 
s_mul_i32 s55, s80, s82 // (op 4933) 


s_sub_u32 s80, s54, s55 // (op 4933) 


v_lshl_add_u32 v26, s80, 2, v38 // (op 4933) 


v_mul_lo_u32 v38, v36, s40 // (op 4933) 
v_add_u32 v38, v38, v26 // (op 4933) 


s_mul_i32 s55, s40, s35 // (op 4935) most significant: low of low * high
s_mul_i32 s80, s41, s34 // (op 4935) most significant: low of high * low
s_mul_i32 s54, s40, s34 // (op 4935) least significant
s_mul_hi_u32 s81, s40, s34 // (op 4935) most significant: high of low * low
s_add_u32 s55, s55, s80 // (op 4935) most significant: sum
s_add_u32 s55, s55, s81 // (op 4935) most significant: sum


s_mov_b32 s76, s42 // (op 4935) reg expression
s_mov_b32 s77, s43 // (op 4935) reg expression
s_mov_b32 s78, s54 // (op 4935) reg expression
s_mov_b32 s79, 131072 // (op 4935) call()

v_and_b32 v26, -64, v1 // (op 5042) 
s_mul_i32 s54, s16, s8 // (op 5042) 
s_add_u32 s54, s54, s0 // (op 5042) 
v_sub_u32 v36, v1, v26 // (op 5042) 

s_mul_hi_u32 s55, s54, s13 // (op 5042) 
v_lshrrev_b32 v26, 6, v1 // (op 5042) 
s_sub_u32 s80, s54, s55 // (op 5042) 
s_lshr_b32 s81, s80, 1 // (op 5042) 

s_add_u32 s80, s81, s55 // (op 5042) 


s_lshr_b32 s55, s80, s11 // (op 5042) 

s_and_b32 s80, s11, 2147483648 // (op 5042) 
s_cmp_eq_i32 s80, 0 // (op 5042) 
s_mov_b32 s81, scc // (op 5042) 

s_cmp_eq_i32 s81, 0 // (op 5042) 
s_cselect_b32 s80, s54, s55 // (op 5042) 


s_mul_hi_u32 s55, s80, s12 // (op 5042) 
s_sub_u32 s81, s80, s55 // (op 5042) 
s_lshr_b32 s82, s81, 1 // (op 5042) 

s_add_u32 s81, s82, s55 // (op 5042) 


s_lshr_b32 s55, s81, s10 // (op 5042) 

s_and_b32 s81, s10, 2147483648 // (op 5042) 
s_cmp_eq_i32 s81, 0 // (op 5042) 
s_mov_b32 s82, scc // (op 5042) 

s_cmp_eq_i32 s82, 0 // (op 5042) 
s_cselect_b32 s81, s80, s55 // (op 5042) 


s_mul_i32 s55, s81, s14 // (op 5042) 

s_sub_u32 s81, s80, s55 // (op 5042) 

s_ashr_i64 s[82:83], s[2:3], 7 // (op 5042) 
v_lshl_add_u32 v39, s81, 6, v36 // (op 5042) 


s_mul_i32 s55, s80, s82 // (op 5042) 


s_sub_u32 s80, s54, s55 // (op 5042) 


v_lshl_add_u32 v36, s80, 2, v26 // (op 5042) 


v_mul_lo_u32 v26, v36, s44 // (op 5042) 
v_add_u32 v26, v26, v39 // (op 5042) 


s_mul_i32 s55, s44, s49 // (op 5044) most significant: low of low * high
s_mul_i32 s80, s45, s48 // (op 5044) most significant: low of high * low
s_mul_i32 s54, s44, s48 // (op 5044) least significant
s_mul_hi_u32 s81, s44, s48 // (op 5044) most significant: high of low * low
s_add_u32 s55, s55, s80 // (op 5044) most significant: sum
s_add_u32 s55, s55, s81 // (op 5044) most significant: sum


s_mov_b32 s80, s46 // (op 5044) reg expression
s_mov_b32 s81, s47 // (op 5044) reg expression
s_mov_b32 s82, s54 // (op 5044) reg expression
s_mov_b32 s83, 131072 // (op 5044) call()

v_and_b32 v36, -64, v1 // (op 5061) 
s_mul_i32 s54, s16, s8 // (op 5061) 
s_add_u32 s54, s54, s0 // (op 5061) 
v_sub_u32 v39, v1, v36 // (op 5061) 

s_mul_hi_u32 s55, s54, s13 // (op 5061) 
v_lshrrev_b32 v36, 6, v1 // (op 5061) 
s_sub_u32 s84, s54, s55 // (op 5061) 
s_lshr_b32 s85, s84, 1 // (op 5061) 

s_add_u32 s84, s85, s55 // (op 5061) 


s_lshr_b32 s55, s84, s11 // (op 5061) 

s_and_b32 s84, s11, 2147483648 // (op 5061) 
s_cmp_eq_i32 s84, 0 // (op 5061) 
s_mov_b32 s85, scc // (op 5061) 

s_cmp_eq_i32 s85, 0 // (op 5061) 
s_cselect_b32 s84, s54, s55 // (op 5061) 


s_mul_hi_u32 s55, s84, s12 // (op 5061) 
s_sub_u32 s85, s84, s55 // (op 5061) 
s_lshr_b32 s86, s85, 1 // (op 5061) 

s_add_u32 s85, s86, s55 // (op 5061) 


s_lshr_b32 s55, s85, s10 // (op 5061) 

s_and_b32 s85, s10, 2147483648 // (op 5061) 
s_cmp_eq_i32 s85, 0 // (op 5061) 
s_mov_b32 s86, scc // (op 5061) 

s_cmp_eq_i32 s86, 0 // (op 5061) 
s_cselect_b32 s85, s84, s55 // (op 5061) 


s_mul_i32 s55, s85, s14 // (op 5061) 

s_sub_u32 s85, s84, s55 // (op 5061) 

v_lshl_add_u32 v40, s85, 6, v39 // (op 5061) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 5061) 
s_mul_i32 s55, s84, s86 // (op 5061) 


s_sub_u32 s84, s54, s55 // (op 5061) 


v_lshl_add_u32 v39, s84, 2, v36 // (op 5061) 


v_mul_lo_u32 v36, v39, s44 // (op 5061) 
v_add_u32 v36, v36, v40 // (op 5061) 


s_mul_i32 s55, s44, s49 // (op 5063) most significant: low of low * high
s_mul_i32 s84, s45, s48 // (op 5063) most significant: low of high * low
s_mul_i32 s54, s44, s48 // (op 5063) least significant
s_mul_hi_u32 s85, s44, s48 // (op 5063) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 5063) most significant: sum
s_add_u32 s55, s55, s85 // (op 5063) most significant: sum


s_mov_b32 s80, s46 // (op 5063) reg expression
s_mov_b32 s81, s47 // (op 5063) reg expression
s_mov_b32 s82, s54 // (op 5063) reg expression
s_mov_b32 s83, 131072 // (op 5063) call()

v_and_b32 v39, -4, v1 // (op 5238) 
v_sub_u32 v40, v1, v39 // (op 5238) 

v_lshlrev_b32 v39, 4, v40 // (op 5238) 

v_lshrrev_b32 v40, 2, v1 // (op 5238) 
v_lshl_add_u32 v41, v40, 6, v39 // (op 5238) 


s_nop 0
v_readfirstlane_b32 s54, v41 // (op 5238) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6273) 

s_lshl_b64 s[84:85], s[36:37], 6 // (op 4532) 
v_mov_b64 v[40:41], s[84:85] // (op 4532) convert


v_mov_b64 v[42:43], v[28:29] // (op 4532) reg expression
s_mov_b32 m0, s54 // (op 4532) reg expression
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 4532) Load value direct to lds
v_add_co_u32 v42, s[84:85], v42, v40 // (op 4532) least significant half
v_addc_co_u32 v43, s[84:85], 0, v43, s[84:85] // (op 4532) most significant half

s_add_u32 m0, m0, 4096 // (op 4532) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 4532) Load value direct to lds





v_mov_b32 v39, v26 // (op 4315) reg expression

buffer_load_ubyte v40, v39, s[80:83], 0 offen offset: 0 // (op 4315) Load value

v_and_b32 v39, -8, v1 // (op 5163) 
v_sub_u32 v41, v1, v39 // (op 5163) 

v_lshlrev_b32 v39, 4, v41 // (op 5163) 

v_lshrrev_b32 v41, 3, v1 // (op 5163) 
v_lshl_add_u32 v42, v41, 7, v39 // (op 5163) 


s_nop 0
v_readfirstlane_b32 s54, v42 // (op 5163) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6254) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 4537) 
v_mov_b64 v[42:43], s[84:85] // (op 4537) convert


v_mov_b64 v[44:45], v[32:33] // (op 4537) 
s_add_u32 s55, 8192, s54 // (op 4537) 
s_mov_b32 m0, s55 // (op 4537) reg expression
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 4537) Load value direct to lds
v_add_co_u32 v44, s[84:85], v44, v42 // (op 4537) least significant half
v_addc_co_u32 v45, s[84:85], 0, v45, s[84:85] // (op 4537) most significant half

s_add_u32 m0, m0, 4096 // (op 4537) 
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 4537) Load value direct to lds







buffer_load_ubyte v39, v37, s[76:79], 0 offen offset: 0 // (op 4321) Load value
v_and_b32 v41, -4, v1 // (op 6048) 
v_sub_u32 v42, v1, v41 // (op 6048) 

v_lshlrev_b32 v41, 4, v42 // (op 6048) 

v_lshrrev_b32 v42, 2, v1 // (op 6048) 
v_lshl_add_u32 v43, v42, 6, v41 // (op 6048) 


s_nop 0
v_readfirstlane_b32 s54, v43 // (op 6048) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6456) 

s_lshl_b64 s[84:85], s[36:37], 6 // (op 4542) 
v_mov_b64 v[42:43], s[84:85] // (op 4542) convert


s_lshl_b64 s[84:85], s[36:37], 7 // (op 4542) 
v_mov_b32 v41, s84 // (op 4542) 
v_mov_b32 v44, s85 // (op 4542) 
v_add_co_u32 v46, s[86:87], v28, v41 // (op 4542) least significant half
v_addc_co_u32 v47, s[86:87], v29, v44, s[86:87] // (op 4542) most significant half




s_add_u32 s55, 16384, s54 // (op 4542) 
s_mov_b32 m0, s55 // (op 4542) reg expression
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 4542) Load value direct to lds
v_add_co_u32 v46, s[84:85], v46, v42 // (op 4542) least significant half
v_addc_co_u32 v47, s[84:85], 0, v47, s[84:85] // (op 4542) most significant half

s_add_u32 m0, m0, 4096 // (op 4542) 
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 4542) Load value direct to lds







s_lshl_b32 s54, s44, 2 // (op 4327) 
v_add_u32 v28, v26, s54 // (op 4327) 


buffer_load_ubyte v29, v28, s[80:83], 0 offen offset: 0 // (op 4327) Load value


v_and_b32 v26, -8, v1 // (op 6004) 
v_sub_u32 v28, v1, v26 // (op 6004) 

v_lshlrev_b32 v26, 4, v28 // (op 6004) 

v_lshrrev_b32 v28, 3, v1 // (op 6004) 
v_lshl_add_u32 v41, v28, 7, v26 // (op 6004) 


s_nop 0
v_readfirstlane_b32 s54, v41 // (op 6004) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6446) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 4544) 
v_mov_b64 v[42:43], s[84:85] // (op 4544) convert


v_mov_b64 v[44:45], v[32:33] // (op 4544) 
s_add_u32 s55, 24576, s54 // (op 4544) 
s_mov_b32 m0, s55 // (op 4544) reg expression
s_mov_b32 s84, 128 // (op 4544) call()
buffer_load_dwordx4 v44, s[68:71], s84 offen offset: 0 lds // (op 4544) Load value direct to lds
v_add_co_u32 v44, s[86:87], v44, v42 // (op 4544) least significant half
v_addc_co_u32 v45, s[86:87], 0, v45, s[86:87] // (op 4544) most significant half

s_add_u32 m0, m0, 4096 // (op 4544) 
buffer_load_dwordx4 v44, s[68:71], s84 offen offset: 0 lds // (op 4544) Load value direct to lds









buffer_load_ubyte v26, v37, s[76:79], 0 offen offset:4 // (op 4333) Load value

v_and_b32 v28, -64, v1 // (op 5390) 
v_sub_u32 v37, v1, v28 // (op 5390) 

v_and_b32 v28, 4294967232, v1 // (op 5390) 
v_add_u32 v32, v28, v37 // (op 5390) 


s_barrier  // (op 6314) 



s_waitcnt vmcnt(9)
ds_write_b8 v32, v40 offset:32768 // (op 4336) Store local data 

v_and_b32 v28, -4, v1 // (op 5314) 
v_sub_u32 v37, v1, v28 // (op 5314) 

v_and_b32 v28, 4294967292, v1 // (op 5314) 
v_add_u32 v32, v28, v37 // (op 5314) 


s_barrier  // (op 6293) 



s_waitcnt vmcnt(6)
ds_write_b8 v32, v39 offset:33024 // (op 4338) Store local data 

s_barrier  // (op 6270) 



ds_read_b64_tr_b8 v[42:43], v23 // (op 4363) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v23 offset:512 // (op 4363) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v23 offset:4096 // (op 4363) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v23 offset:4608 // (op 4363) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 6311) 


ds_read_u8 v28, v24 offset:32768 // (op 4383) Load local data 



ds_read_b64_tr_b8 v[50:51], v23 offset:32 // (op 4368) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v23 offset:544 // (op 4368) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v23 offset:4128 // (op 4368) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v23 offset:4640 // (op 4368) Transpose load local data 


ds_read_u8 v37, v24 offset:32800 // (op 4388) Load local data 
s_barrier  // (op 6251) 



ds_read_b128 v[58:61], v21 offset:8192 // (op 4353) Load local data 
ds_read_b128 v[62:65], v21 offset:8256 // (op 4353) Load local data 
s_waitcnt lgkmcnt(12)
s_barrier  // (op 6290) 



ds_read_u8 v41, v20 offset:33024 // (op 4373) Load local data 



ds_read_b128 v[66:69], v21 offset:12288 // (op 4358) Load local data 
ds_read_b128 v[70:73], v21 offset:12352 // (op 4358) Load local data 



ds_read_u8 v32, v20 offset:33152 // (op 4378) Load local data 
v_and_b32 v33, -8, v1 // (op 5135) 
v_sub_u32 v74, v1, v33 // (op 5135) 

v_lshlrev_b32 v33, 4, v74 // (op 5135) 

v_lshrrev_b32 v74, 3, v1 // (op 5135) 
v_lshl_add_u32 v75, v74, 7, v33 // (op 5135) 


s_nop 0
v_readfirstlane_b32 s54, v75 // (op 5135) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 5210) 
v_sub_u32 v74, v1, v33 // (op 5210) 

v_lshlrev_b32 v33, 4, v74 // (op 5210) 

v_lshrrev_b32 v74, 2, v1 // (op 5210) 
v_lshl_add_u32 v75, v74, 6, v33 // (op 5210) 


s_nop 0
v_readfirstlane_b32 s55, v75 // (op 5210) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 5285) 
v_sub_u32 v74, v1, v33 // (op 5285) 

v_and_b32 v33, 4294967292, v1 // (op 5285) 
v_add_u32 v75, v33, v74 // (op 5285) 


v_and_b32 v33, -64, v1 // (op 5361) 
v_sub_u32 v74, v1, v33 // (op 5361) 

v_and_b32 v33, 4294967232, v1 // (op 5361) 
v_add_u32 v76, v33, v74 // (op 5361) 


v_and_b32 v33, -8, v1 // (op 5989) 
v_sub_u32 v74, v1, v33 // (op 5989) 

v_lshlrev_b32 v33, 4, v74 // (op 5989) 

v_lshrrev_b32 v74, 3, v1 // (op 5989) 
v_lshl_add_u32 v77, v74, 7, v33 // (op 5989) 


s_nop 0
v_readfirstlane_b32 s84, v77 // (op 5989) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 6033) 
v_sub_u32 v74, v1, v33 // (op 6033) 

v_lshlrev_b32 v33, 4, v74 // (op 6033) 

v_lshrrev_b32 v74, 2, v1 // (op 6033) 
v_lshl_add_u32 v77, v74, 6, v33 // (op 6033) 


s_nop 0
v_readfirstlane_b32 s85, v77 // (op 6033) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v33, -4, v1 // (op 6077) 
v_sub_u32 v74, v1, v33 // (op 6077) 

v_and_b32 v33, 4294967292, v1 // (op 6077) 
v_add_u32 v77, v33, v74 // (op 6077) 


v_and_b32 v33, -64, v1 // (op 6106) 
v_sub_u32 v74, v1, v33 // (op 6106) 

v_and_b32 v33, 4294967232, v1 // (op 6106) 
v_add_u32 v78, v33, v74 // (op 6106) 


s_mov_b32 s1, 0 // (op 1903) call()
s_and_b32 s86, s7, 4294967294 // (op 1902) 
s_cmp_lt_u32 s1, s86 // (op 1902) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_1902 // (op 1902) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_1902 if false)
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoop_1902:
s_add_u32 s1, s1, 2 // (op 1904) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:65], v[42:49], a[0:3], v41, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 90) 
s_barrier  // (op 6276) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:65], v[50:57], a[4:7], v41, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1690) 
s_lshl_b64 s[86:87], s[36:37], 6 // (op 4457) 
v_mov_b64 v[80:81], s[86:87] // (op 4457) convert


v_mov_b64 v[82:83], v[30:31] // (op 4457) 
s_lshl_b64 s[86:87], s[36:37], 7 // (op 4457) 
s_lshl_b64 s[88:89], s[86:87], 1 // (op 4457) 

v_mov_b32 v33, s88 // (op 4457) 
v_mov_b32 v74, s89 // (op 4457) 
v_add_co_u32 v84, s[86:87], v82, v33 // (op 4457) least significant half
v_addc_co_u32 v85, s[86:87], v83, v74, s[86:87] // (op 4457) most significant half




s_mov_b32 m0, s55 // (op 4457) reg expression
buffer_load_dwordx4 v84, s[72:75], 0 offen offset: 0 lds // (op 4457) Load value direct to lds
v_add_co_u32 v84, s[86:87], v84, v80 // (op 4457) least significant half
v_addc_co_u32 v85, s[86:87], 0, v85, s[86:87] // (op 4457) most significant half

s_add_u32 m0, m0, 4096 // (op 4457) 
buffer_load_dwordx4 v84, s[72:75], 0 offen offset: 0 lds // (op 4457) Load value direct to lds


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:73], v[42:49], a[8:11], v32, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1987) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:73], v[50:57], a[12:15], v32, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1941) 
v_mov_b32 v33, v36 // (op 25) 
s_lshl_b32 s86, s44, 3 // (op 25) 
v_add_u32 v74, v33, s86 // (op 25) 


buffer_load_ubyte v40, v74, s[80:83], 0 offen offset: 0 // (op 25) Load value

s_barrier  // (op 6257) 

s_lshl_b64 s[86:87], s[30:31], 5 // (op 4452) 
v_mov_b64 v[80:81], s[86:87] // (op 4452) convert


v_mov_b64 v[84:85], v[34:35] // (op 4452) 
v_mov_b64 v[86:87], v[84:85] // (op 4452) 
s_add_u32 s86, 8192, s54 // (op 4452) 
s_mov_b32 m0, s86 // (op 4452) reg expression
s_mov_b32 s87, 256 // (op 4452) call()
buffer_load_dwordx4 v86, s[68:71], s87 offen offset: 0 lds // (op 4452) Load value direct to lds
v_add_co_u32 v86, s[88:89], v86, v80 // (op 4452) least significant half
v_addc_co_u32 v87, s[88:89], 0, v87, s[88:89] // (op 4452) most significant half

s_add_u32 m0, m0, 4096 // (op 4452) 
buffer_load_dwordx4 v86, s[68:71], s87 offen offset: 0 lds // (op 4452) Load value direct to lds






v_mov_b32 v74, v38 // (op 17) 

buffer_load_ubyte v39, v74, s[76:79], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 6472) 



s_waitcnt vmcnt(9)
ds_write_b8 v78, v29 offset:33280 // (op 1569) Store local data 
s_barrier  // (op 6466) 



s_waitcnt vmcnt(6)
ds_write_b8 v77, v26 offset:33536 // (op 1568) Store local data 
s_barrier  // (op 6452) 



ds_read_b64_tr_b8 v[86:87], v16 offset:16384 // (op 1561) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v16 offset:16896 // (op 1561) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v16 offset:20480 // (op 1561) Transpose load local data 
ds_read_b64_tr_b8 v[92:93], v16 offset:20992 // (op 1561) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 6469) 


ds_read_u8 v79, v17 offset:33280 // (op 1565) Load local data 



ds_read_b64_tr_b8 v[94:95], v16 offset:16416 // (op 1695) Transpose load local data 
ds_read_b64_tr_b8 v[96:97], v16 offset:16928 // (op 1695) Transpose load local data 
ds_read_b64_tr_b8 v[98:99], v16 offset:20512 // (op 1695) Transpose load local data 
ds_read_b64_tr_b8 v[100:101], v16 offset:21024 // (op 1695) Transpose load local data 


ds_read_u8 v80, v17 offset:33312 // (op 1699) Load local data 
s_barrier  // (op 6443) 
s_barrier  // (op 6655) 



ds_read_b128 v[58:61], v15 offset:24576 // (op 1559) Load local data 
ds_read_b128 v[62:65], v15 offset:24640 // (op 1559) Load local data 
s_waitcnt lgkmcnt(12)
s_barrier  // (op 6462) 



ds_read_u8 v41, v14 offset:33536 // (op 1563) Load local data 



ds_read_b128 v[66:69], v15 offset:28672 // (op 1944) Load local data 
ds_read_b128 v[70:73], v15 offset:28736 // (op 1944) Load local data 



ds_read_u8 v32, v14 offset:33664 // (op 1947) Load local data 
s_barrier  // (op 6677) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:65], v[86:93], a[0:3], v41, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1566) 
s_barrier  // (op 6459) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:65], v[94:101], a[4:7], v41, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1691) 
s_lshl_b64 s[86:87], s[36:37], 6 // (op 4513) 
v_mov_b64 v[102:103], s[86:87] // (op 4513) convert


s_lshl_b64 s[86:87], s[36:37], 7 // (op 4513) 
s_mul_i32 s89, 3, s87 // (op 4513) most significant: low of low * high
s_mul_i32 s88, 3, s86 // (op 4513) least significant
s_mul_hi_u32 s90, 3, s86 // (op 4513) most significant: high of low * low
s_add_u32 s89, s89, s90 // (op 4513) most significant: sum


v_mov_b32 v81, s88 // (op 4513) 
v_mov_b32 v104, s89 // (op 4513) 
v_add_co_u32 v106, s[86:87], v82, v81 // (op 4513) least significant half
v_addc_co_u32 v107, s[86:87], v83, v104, s[86:87] // (op 4513) most significant half




s_add_u32 s86, 16384, s85 // (op 4513) 
s_nop 0
s_mov_b32 m0, s86 // (op 4513) reg expression
buffer_load_dwordx4 v106, s[72:75], 0 offen offset: 0 lds // (op 4513) Load value direct to lds
v_add_co_u32 v106, s[88:89], v106, v102 // (op 4513) least significant half
v_addc_co_u32 v107, s[88:89], 0, v107, s[88:89] // (op 4513) most significant half

s_add_u32 m0, m0, 4096 // (op 4513) 
buffer_load_dwordx4 v106, s[72:75], 0 offen offset: 0 lds // (op 4513) Load value direct to lds



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:73], v[86:93], a[8:11], v32, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1988) 





v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:73], v[94:101], a[12:15], v32, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1942) 
s_lshl_b32 s86, s44, 2 // (op 1572) 


s_mul_i32 s87, 3, s86 // (op 1572) 

v_add_u32 v79, v33, s87 // (op 1572) 


buffer_load_ubyte v29, v79, s[80:83], 0 offen offset: 0 // (op 1572) Load value

s_lshl_b64 s[86:87], s[36:37], 7 // (op 4797) 

s_lshl_b64 s[88:89], s[86:87], 1 // (op 4797) 

s_barrier  // (op 6449) 
s_lshl_b32 s86, s44, 3 // (op 5053) 

v_add_u32 v36, v36, s86 // (op 5053) 

v_mov_b32 v33, s88 // (op 4797) 
v_mov_b32 v79, s89 // (op 4797) 
s_lshl_b64 s[86:87], s[30:31], 5 // (op 4508) 
v_mov_b64 v[80:81], s[86:87] // (op 4508) convert


v_mov_b64 v[82:83], v[84:85] // (op 4508) 
s_add_u32 s86, 24576, s84 // (op 4508) 
s_mov_b32 m0, s86 // (op 4508) reg expression
s_mov_b32 s87, 384 // (op 4508) call()
v_add_co_u32 v30, s[90:91], v30, v33 // (op 4797) least significant half
v_addc_co_u32 v31, s[90:91], v31, v79, s[90:91] // (op 4797) most significant half




buffer_load_dwordx4 v82, s[68:71], s87 offen offset: 0 lds // (op 4508) Load value direct to lds
v_add_co_u32 v82, s[88:89], v82, v80 // (op 4508) least significant half
v_addc_co_u32 v83, s[88:89], 0, v83, s[88:89] // (op 4508) most significant half

s_add_u32 m0, m0, 4096 // (op 4508) 
buffer_load_dwordx4 v82, s[68:71], s87 offen offset: 0 lds // (op 4508) Load value direct to lds








v_add_co_u32 v34, vcc, 256, v34 // (op 4669) least significant half
v_addc_co_u32 v35, vcc, 0, v35, vcc // (op 4669) most significant half
buffer_load_ubyte v26, v74, s[76:79], 0 offen offset:12 // (op 1571) Load value

s_barrier  // (op 6317) 
v_add_u32 v38, 8, v38 // (op 4925) 



s_waitcnt vmcnt(9)
ds_write_b8 v76, v40 offset:32768 // (op 30) Store local data 
s_barrier  // (op 6296) 



s_waitcnt vmcnt(6)
ds_write_b8 v75, v39 offset:33024 // (op 21) Store local data 
s_barrier  // (op 6282) 



ds_read_b64_tr_b8 v[42:43], v4 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:512 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v4 offset:4096 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v4 offset:4608 // (op 77) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 6324) 


ds_read_u8 v28, v5 offset:32768 // (op 87) Load local data 



ds_read_b64_tr_b8 v[50:51], v4 offset:32 // (op 1685) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v4 offset:544 // (op 1685) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v4 offset:4128 // (op 1685) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v4 offset:4640 // (op 1685) Transpose load local data 


ds_read_u8 v37, v5 offset:32800 // (op 1689) Load local data 
s_barrier  // (op 6263) 



ds_read_b128 v[58:61], v3 offset:8192 // (op 72) Load local data 
ds_read_b128 v[62:65], v3 offset:8256 // (op 72) Load local data 
s_waitcnt lgkmcnt(12)
s_barrier  // (op 6303) 



ds_read_u8 v41, v0 offset:33024 // (op 82) Load local data 



ds_read_b128 v[66:69], v3 offset:12288 // (op 1936) Load local data 
ds_read_b128 v[70:73], v3 offset:12352 // (op 1936) Load local data 



ds_read_u8 v32, v0 offset:33152 // (op 1939) Load local data 
s_and_b32 s86, s7, 4294967294 // (op 1902) 
s_cmp_lt_u32 s1, s86 // (op 1902) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoop_1902 // (op 1902) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoop_1902 if true)
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_1902:














s_barrier  // (op 6661) 
s_and_b32 s54, s7, 4294967294 // (op 4581) 
v_lshrrev_b32 v30, 3, v1 // (op 4581) 
s_add_u32 s55, s0, s54 // (op 4581) 

v_and_b32 v31, -8, v1 // (op 4581) 
v_sub_u32 v33, v1, v31 // (op 4581) 

s_mul_i32 s54, s16, s8 // (op 4581) 
s_add_u32 s54, s54, s55 // (op 4581) 

v_lshlrev_b32 v31, 4, v33 // (op 4581) 

s_mul_hi_u32 s55, s54, s13 // (op 4581) 
s_sub_u32 s84, s54, s55 // (op 4581) 
s_lshr_b32 s85, s84, 1 // (op 4581) 

s_add_u32 s84, s85, s55 // (op 4581) 


s_lshr_b32 s55, s84, s11 // (op 4581) 

s_and_b32 s84, s11, 2147483648 // (op 4581) 
s_cmp_eq_i32 s84, 0 // (op 4581) 
s_mov_b32 s85, scc // (op 4581) 

s_cmp_eq_i32 s85, 0 // (op 4581) 
s_cselect_b32 s84, s54, s55 // (op 4581) 


s_mul_hi_u32 s55, s84, s12 // (op 4581) 
s_sub_u32 s85, s84, s55 // (op 4581) 
s_lshr_b32 s86, s85, 1 // (op 4581) 

s_add_u32 s85, s86, s55 // (op 4581) 


s_lshr_b32 s55, s85, s10 // (op 4581) 

s_and_b32 s85, s10, 2147483648 // (op 4581) 
s_cmp_eq_i32 s85, 0 // (op 4581) 
s_mov_b32 s86, scc // (op 4581) 

s_cmp_eq_i32 s86, 0 // (op 4581) 
s_cselect_b32 s85, s84, s55 // (op 4581) 


v_lshl_add_u32 v33, s85, 6, v30 // (op 4581) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 4581) 
s_mul_i32 s55, s84, s86 // (op 4581) 


s_sub_u32 s84, s54, s55 // (op 4581) 


v_lshl_add_u32 v30, s84, 7, v31 // (op 4581) 


v_mov_b32 v35, 0 // (op 4581) convert
v_mov_b32 v34, v30 // (op 4581) convert

v_mul_lo_u32 v31, v33, s31 // (op 4581) most significant: low of low * high
v_mul_hi_u32 v36, v33, s30 // (op 4581) most significant: high of low * low
v_mul_lo_u32 v30, v33, s30 // (op 4581) least significant: low of low * low
v_add_u32 v31, v31, v36 // (op 4581) most significant: sum

v_add_co_u32 v30, s[54:55], v30, v34 // (op 4581) least significant half
v_addc_co_u32 v31, s[54:55], v31, v35, s[54:55] // (op 4581) most significant half



s_mul_i32 s55, s30, s35 // (op 4583) most significant: low of low * high
s_mul_i32 s84, s31, s34 // (op 4583) most significant: low of high * low
s_mul_i32 s54, s30, s34 // (op 4583) least significant
s_mul_hi_u32 s85, s30, s34 // (op 4583) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 4583) most significant: sum
s_add_u32 s55, s55, s85 // (op 4583) most significant: sum


s_mov_b32 s68, s32 // (op 4583) reg expression
s_mov_b32 s69, s33 // (op 4583) reg expression
s_mov_b32 s70, s54 // (op 4583) reg expression
s_mov_b32 s71, 131072 // (op 4583) call()

s_and_b32 s54, s7, 4294967294 // (op 4721) 
v_and_b32 v33, -4, v1 // (op 4721) 
s_add_u32 s55, s0, s54 // (op 4721) 

v_sub_u32 v34, v1, v33 // (op 4721) 

v_lshlrev_b32 v33, 4, v34 // (op 4721) 

s_mul_i32 s54, s16, s8 // (op 4721) 
s_add_u32 s54, s54, s55 // (op 4721) 

v_lshrrev_b32 v34, 2, v1 // (op 4721) 
s_mul_hi_u32 s55, s54, s13 // (op 4721) 
s_sub_u32 s84, s54, s55 // (op 4721) 
s_lshr_b32 s85, s84, 1 // (op 4721) 

s_add_u32 s84, s85, s55 // (op 4721) 


s_lshr_b32 s55, s84, s11 // (op 4721) 

s_and_b32 s84, s11, 2147483648 // (op 4721) 
s_cmp_eq_i32 s84, 0 // (op 4721) 
s_mov_b32 s85, scc // (op 4721) 

s_cmp_eq_i32 s85, 0 // (op 4721) 
s_cselect_b32 s84, s54, s55 // (op 4721) 


s_mul_hi_u32 s55, s84, s12 // (op 4721) 
s_sub_u32 s85, s84, s55 // (op 4721) 
s_lshr_b32 s86, s85, 1 // (op 4721) 

s_add_u32 s85, s86, s55 // (op 4721) 


s_lshr_b32 s55, s85, s10 // (op 4721) 

s_and_b32 s85, s10, 2147483648 // (op 4721) 
s_cmp_eq_i32 s85, 0 // (op 4721) 
s_mov_b32 s86, scc // (op 4721) 

s_cmp_eq_i32 s86, 0 // (op 4721) 
s_cselect_b32 s85, s84, s55 // (op 4721) 


s_mul_i32 s55, s85, s14 // (op 4721) 

s_sub_u32 s85, s84, s55 // (op 4721) 

v_lshl_add_u32 v35, s85, 6, v33 // (op 4721) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 4721) 
v_mov_b32 v39, 0 // (op 4721) convert
v_mov_b32 v38, v35 // (op 4721) convert

s_mul_i32 s55, s84, s86 // (op 4721) 


s_sub_u32 s84, s54, s55 // (op 4721) 


v_lshl_add_u32 v33, s84, 7, v34 // (op 4721) 


v_mul_lo_u32 v35, v33, s37 // (op 4721) most significant: low of low * high
v_mul_hi_u32 v36, v33, s36 // (op 4721) most significant: high of low * low
v_mul_lo_u32 v34, v33, s36 // (op 4721) least significant: low of low * low
v_add_u32 v35, v35, v36 // (op 4721) most significant: sum

v_add_co_u32 v34, s[54:55], v34, v38 // (op 4721) least significant half
v_addc_co_u32 v35, s[54:55], v35, v39, s[54:55] // (op 4721) most significant half



s_mul_i32 s55, s36, s3 // (op 4723) most significant: low of low * high
s_mul_i32 s84, s37, s2 // (op 4723) most significant: low of high * low
s_mul_i32 s54, s36, s2 // (op 4723) least significant
s_mul_hi_u32 s85, s36, s2 // (op 4723) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 4723) most significant: sum
s_add_u32 s55, s55, s85 // (op 4723) most significant: sum


s_mov_b32 s72, s38 // (op 4723) reg expression
s_mov_b32 s73, s39 // (op 4723) reg expression
s_mov_b32 s74, s54 // (op 4723) reg expression
s_mov_b32 s75, 131072 // (op 4723) call()

s_and_b32 s54, s7, 4294967294 // (op 4849) 
v_lshrrev_b32 v33, 2, v1 // (op 4849) 
s_add_u32 s55, s0, s54 // (op 4849) 

v_and_b32 v36, -4, v1 // (op 4849) 
v_sub_u32 v38, v1, v36 // (op 4849) 

s_mul_i32 s54, s16, s8 // (op 4849) 
s_add_u32 s54, s54, s55 // (op 4849) 

s_mul_hi_u32 s55, s54, s13 // (op 4849) 
s_sub_u32 s84, s54, s55 // (op 4849) 
s_lshr_b32 s85, s84, 1 // (op 4849) 

s_add_u32 s84, s85, s55 // (op 4849) 


s_lshr_b32 s55, s84, s11 // (op 4849) 

s_and_b32 s84, s11, 2147483648 // (op 4849) 
s_cmp_eq_i32 s84, 0 // (op 4849) 
s_mov_b32 s85, scc // (op 4849) 

s_cmp_eq_i32 s85, 0 // (op 4849) 
s_cselect_b32 s84, s54, s55 // (op 4849) 


s_mul_hi_u32 s55, s84, s12 // (op 4849) 
s_sub_u32 s85, s84, s55 // (op 4849) 
s_lshr_b32 s86, s85, 1 // (op 4849) 

s_add_u32 s85, s86, s55 // (op 4849) 


s_lshr_b32 s55, s85, s10 // (op 4849) 

s_and_b32 s85, s10, 2147483648 // (op 4849) 
s_cmp_eq_i32 s85, 0 // (op 4849) 
s_mov_b32 s86, scc // (op 4849) 

s_cmp_eq_i32 s86, 0 // (op 4849) 
s_cselect_b32 s85, s84, s55 // (op 4849) 


v_lshl_add_u32 v36, s85, 6, v33 // (op 4849) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 4849) 
s_mul_i32 s55, s84, s86 // (op 4849) 


s_sub_u32 s84, s54, s55 // (op 4849) 


v_lshl_add_u32 v33, s84, 2, v38 // (op 4849) 


v_mul_lo_u32 v38, v36, s40 // (op 4849) 
v_add_u32 v38, v38, v33 // (op 4849) 


s_mul_i32 s55, s40, s35 // (op 4851) most significant: low of low * high
s_mul_i32 s84, s41, s34 // (op 4851) most significant: low of high * low
s_mul_i32 s54, s40, s34 // (op 4851) least significant
s_mul_hi_u32 s85, s40, s34 // (op 4851) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 4851) most significant: sum
s_add_u32 s55, s55, s85 // (op 4851) most significant: sum


s_mov_b32 s76, s42 // (op 4851) reg expression
s_mov_b32 s77, s43 // (op 4851) reg expression
s_mov_b32 s78, s54 // (op 4851) reg expression
s_mov_b32 s79, 131072 // (op 4851) call()

s_and_b32 s54, s7, 4294967294 // (op 4977) 
v_and_b32 v33, -64, v1 // (op 4977) 
s_add_u32 s55, s0, s54 // (op 4977) 

v_sub_u32 v36, v1, v33 // (op 4977) 

v_lshrrev_b32 v33, 6, v1 // (op 4977) 
s_mul_i32 s54, s16, s8 // (op 4977) 
s_add_u32 s54, s54, s55 // (op 4977) 

s_mul_hi_u32 s55, s54, s13 // (op 4977) 
s_sub_u32 s84, s54, s55 // (op 4977) 
s_lshr_b32 s85, s84, 1 // (op 4977) 

s_add_u32 s84, s85, s55 // (op 4977) 


s_lshr_b32 s55, s84, s11 // (op 4977) 

s_and_b32 s84, s11, 2147483648 // (op 4977) 
s_cmp_eq_i32 s84, 0 // (op 4977) 
s_mov_b32 s85, scc // (op 4977) 

s_cmp_eq_i32 s85, 0 // (op 4977) 
s_cselect_b32 s84, s54, s55 // (op 4977) 


s_mul_hi_u32 s55, s84, s12 // (op 4977) 
s_sub_u32 s85, s84, s55 // (op 4977) 
s_lshr_b32 s86, s85, 1 // (op 4977) 

s_add_u32 s85, s86, s55 // (op 4977) 


s_lshr_b32 s55, s85, s10 // (op 4977) 

s_and_b32 s85, s10, 2147483648 // (op 4977) 
s_cmp_eq_i32 s85, 0 // (op 4977) 
s_mov_b32 s86, scc // (op 4977) 

s_cmp_eq_i32 s86, 0 // (op 4977) 
s_cselect_b32 s85, s84, s55 // (op 4977) 


s_mul_i32 s55, s85, s14 // (op 4977) 

s_sub_u32 s85, s84, s55 // (op 4977) 

v_lshl_add_u32 v39, s85, 6, v36 // (op 4977) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 4977) 
s_mul_i32 s55, s84, s86 // (op 4977) 


s_sub_u32 s84, s54, s55 // (op 4977) 


v_lshl_add_u32 v36, s84, 2, v33 // (op 4977) 


v_mul_lo_u32 v33, v36, s44 // (op 4977) 
v_add_u32 v33, v33, v39 // (op 4977) 


s_mul_i32 s55, s44, s49 // (op 4979) most significant: low of low * high
s_mul_i32 s84, s45, s48 // (op 4979) most significant: low of high * low
s_mul_i32 s54, s44, s48 // (op 4979) least significant
s_mul_hi_u32 s85, s44, s48 // (op 4979) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 4979) most significant: sum
s_add_u32 s55, s55, s85 // (op 4979) most significant: sum


s_mov_b32 s80, s46 // (op 4979) reg expression
s_mov_b32 s81, s47 // (op 4979) reg expression
s_mov_b32 s82, s54 // (op 4979) reg expression
s_mov_b32 s83, 131072 // (op 4979) call()

v_and_b32 v36, -8, v1 // (op 5148) 
v_sub_u32 v39, v1, v36 // (op 5148) 

v_lshlrev_b32 v36, 4, v39 // (op 5148) 

v_lshrrev_b32 v39, 3, v1 // (op 5148) 
v_lshl_add_u32 v40, v39, 7, v36 // (op 5148) 


s_nop 0
v_readfirstlane_b32 s54, v40 // (op 5148) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v36, -4, v1 // (op 5223) 
v_sub_u32 v39, v1, v36 // (op 5223) 

v_lshlrev_b32 v36, 4, v39 // (op 5223) 

v_lshrrev_b32 v39, 2, v1 // (op 5223) 
v_lshl_add_u32 v40, v39, 6, v36 // (op 5223) 


s_nop 0
v_readfirstlane_b32 s55, v40 // (op 5223) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v36, -4, v1 // (op 5298) 
v_sub_u32 v39, v1, v36 // (op 5298) 

v_and_b32 v36, 4294967292, v1 // (op 5298) 
v_add_u32 v40, v36, v39 // (op 5298) 


v_and_b32 v36, -64, v1 // (op 5374) 
v_sub_u32 v39, v1, v36 // (op 5374) 

v_and_b32 v36, 4294967232, v1 // (op 5374) 
v_add_u32 v74, v36, v39 // (op 5374) 


s_and_b32 s84, s7, 4294967294 // (op 1908) 
s_cmp_lt_u32 s84, s7 // (op 1907) 










s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_1907 // (op 1907) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_1907 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_1907:
s_barrier  // (op 6260) 
s_add_u32 s84, s84, 1 // (op 1909) 

s_lshl_b64 s[86:87], s[30:31], 5 // (op 4495) 
v_mov_b64 v[28:29], s[86:87] // (op 4495) convert


v_mov_b64 v[36:37], v[30:31] // (op 4495) 
v_mov_b64 v[72:73], v[36:37] // (op 4495) 
s_add_u32 s85, 8192, s54 // (op 4495) 
s_mov_b32 m0, s85 // (op 4495) reg expression
buffer_load_dwordx4 v72, s[68:71], 0 offen offset: 0 lds // (op 4495) Load value direct to lds
v_add_co_u32 v72, s[86:87], v72, v28 // (op 4495) least significant half
v_addc_co_u32 v73, s[86:87], 0, v73, s[86:87] // (op 4495) most significant half

s_add_u32 m0, m0, 4096 // (op 4495) 
buffer_load_dwordx4 v72, s[68:71], 0 offen offset: 0 lds // (op 4495) Load value direct to lds




v_add_co_u32 v30, vcc, 128, v30 // (op 4571) least significant half
v_addc_co_u32 v31, vcc, 0, v31, vcc // (op 4571) most significant half
s_barrier  // (op 6279) 

s_lshl_b64 s[86:87], s[36:37], 6 // (op 4501) 
v_mov_b64 v[28:29], s[86:87] // (op 4501) convert


v_mov_b64 v[36:37], v[34:35] // (op 4501) 
v_mov_b64 v[72:73], v[36:37] // (op 4501) reg expression
s_mov_b32 m0, s55 // (op 4501) reg expression
buffer_load_dwordx4 v72, s[72:75], 0 offen offset: 0 lds // (op 4501) Load value direct to lds
v_add_co_u32 v72, s[86:87], v72, v28 // (op 4501) least significant half
v_addc_co_u32 v73, s[86:87], 0, v73, s[86:87] // (op 4501) most significant half

s_add_u32 m0, m0, 4096 // (op 4501) 
buffer_load_dwordx4 v72, s[72:75], 0 offen offset: 0 lds // (op 4501) Load value direct to lds



s_lshl_b64 s[86:87], s[36:37], 7 // (op 4713) 
v_mov_b32 v26, s86 // (op 4713) 
v_mov_b32 v32, s87 // (op 4713) 
v_add_co_u32 v34, s[88:89], v34, v26 // (op 4713) least significant half
v_addc_co_u32 v35, s[88:89], v35, v32, s[88:89] // (op 4713) most significant half






v_mov_b32 v26, v38 // (op 1505) 

buffer_load_ubyte v32, v26, s[76:79], 0 offen offset: 0 // (op 1505) Load value

v_add_u32 v38, 4, v38 // (op 4841) 


s_barrier  // (op 6299) 
v_mov_b32 v26, v33 // (op 1506) 



v_mov_b32 v39, v26 // (op 1506) reg expression

buffer_load_ubyte v28, v39, s[80:83], 0 offen offset: 0 // (op 1506) Load value


s_lshl_b32 s85, s44, 2 // (op 4969) 
v_add_u32 v33, v33, s85 // (op 4969) 

s_waitcnt vmcnt(1)
ds_write_b8 v40, v32 offset:33024 // (op 1502) Store local data 

s_barrier  // (op 6266) 



s_barrier  // (op 6320) 
ds_read_b128 v[66:69], v3 offset:8192 // (op 1493) Load local data 



ds_read_b128 v[70:73], v3 offset:8256 // (op 1493) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v74, v28 offset:32768 // (op 1503) Store local data 

s_barrier  // (op 6286) 



ds_read_b64_tr_b8 v[58:59], v4 // (op 1495) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v4 offset:512 // (op 1495) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v4 offset:4096 // (op 1495) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v4 offset:4608 // (op 1495) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 6307) 



ds_read_u8 v26, v0 offset:33024 // (op 1497) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 6327) 


ds_read_u8 v32, v5 offset:32768 // (op 1499) Load local data 



ds_read_b64_tr_b8 v[50:51], v4 offset:32 // (op 1664) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v4 offset:544 // (op 1664) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:73], v[58:65], a[0:3], v26, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1500) 
ds_read_b64_tr_b8 v[54:55], v4 offset:4128 // (op 1664) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v4 offset:4640 // (op 1664) Transpose load local data 


ds_read_u8 v39, v5 offset:32800 // (op 1668) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:73], v[50:57], a[4:7], v26, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1669) 
ds_read_b128 v[42:45], v3 offset:12288 // (op 1882) Load local data 


ds_read_b128 v[46:49], v3 offset:12352 // (op 1882) Load local data 



ds_read_u8 v26, v0 offset:33152 // (op 1885) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[42:49], v[58:65], a[8:11], v26, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1887) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:49], v[50:57], a[12:15], v26, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1918) 




s_cmp_lt_u32 s84, s7 // (op 1907) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_1907 // (op 1907) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_1907 if true)
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_1907:









s_barrier  // (op 6665) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 296) 
s_cmp_lg_u32 s18, 0 // (op 322) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Send_Tile_322 // (op 322) Condition: False, jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Send_Tile_322



v_add_u32 v26, 8192, v25 // (op 2344) 
buffer_store_dword a8, v26, s[56:59], 0 offen sc1 // (op 2344) Store value

v_add_u32 v26, 8448, v25 // (op 2344) 
buffer_store_dword a9, v26, s[56:59], 0 offen sc1 // (op 2344) Store value

v_add_u32 v26, 8704, v25 // (op 2344) 
buffer_store_dword a10, v26, s[56:59], 0 offen sc1 // (op 2344) Store value

v_add_u32 v26, 8960, v25 // (op 2344) 
buffer_store_dword a11, v26, s[56:59], 0 offen sc1 // (op 2344) Store value




v_add_u32 v26, 8320, v25 // (op 2346) 
buffer_store_dword a12, v26, s[56:59], 0 offen sc1 // (op 2346) Store value

v_add_u32 v26, 8576, v25 // (op 2346) 
buffer_store_dword a13, v26, s[56:59], 0 offen sc1 // (op 2346) Store value

v_add_u32 v26, 8832, v25 // (op 2346) 
buffer_store_dword a14, v26, s[56:59], 0 offen sc1 // (op 2346) Store value

v_add_u32 v26, 9088, v25 // (op 2346) 
buffer_store_dword a15, v26, s[56:59], 0 offen sc1 // (op 2346) Store value




buffer_store_dword a0, v25, s[56:59], 0 offen sc1 // (op 319) Store value
buffer_store_dword a1, v25, s[56:59], 0 offen offset:256 sc1 // (op 319) Store value
buffer_store_dword a2, v25, s[56:59], 0 offen offset:512 sc1 // (op 319) Store value
buffer_store_dword a3, v25, s[56:59], 0 offen offset:768 sc1 // (op 319) Store value



buffer_store_dword a4, v25, s[56:59], 0 offen offset:128 sc1 // (op 2330) Store value
buffer_store_dword a5, v25, s[56:59], 0 offen offset:384 sc1 // (op 2330) Store value
buffer_store_dword a6, v25, s[56:59], 0 offen offset:640 sc1 // (op 2330) Store value
buffer_store_dword a7, v25, s[56:59], 0 offen offset:896 sc1 // (op 2330) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 323) 
v_cmp_eq_i32 vcc, v1, 0 // (op 327) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_327 // (op 327) Condition: False, jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_327
s_mov_b32 s54, 1 // (op 325) call()
s_mov_b32 s85, 0 // (op 326) convert
s_mov_b32 s84, s16 // (op 326) convert
s_lshl_b64 s[84:85], s[84:85], 2 // (op 326) 
s_add_u32 s86, s50, s84 // (op 326) Start of Int64 add, locking SCC
s_addc_u32 s87, s51, s85 // (op 326) End of Int64 add, unlocking SCC
s_store_dword s54, s[86:87], 0 glc // (op 326) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_327 // (op 327) Condition: Done, jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_327
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_327:
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_327:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Send_Tile_322 // (op 322) Condition: Done, jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Send_Tile_322
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Send_Tile_322:
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Send_Tile_322:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 338) 
s_mov_b32 s54, scc // (op 338) 
s_ashr_i64 s[84:85], s[2:3], 7 // (op 338) 
s_mov_b32 s55, s84 // (op 338) convert

s_sub_u32 s84, s55, 1 // (op 338) 

s_cmp_lt_u32 s19, s84 // (op 338) 
s_mov_b32 s55, scc // (op 338) 

s_and_b32 s84, s54, s55 // (op 338) 
s_cmp_lg_u32 s84, 0 // (op 338) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Receive_Tile_338 // (op 338) Condition: False, jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Receive_Tile_338
s_mov_b32 s54, 0 // (op 304) call()
s_mul_i32 s55, s54, s8 // (op 303) 
s_ashr_i64 s[84:85], s[2:3], 7 // (op 303) 
s_mov_b32 s86, s84 // (op 303) convert

s_sub_u32 s84, s86, s19 // (op 303) 

s_sub_u32 s85, s84, 1 // (op 303) 

s_cmp_lt_u32 s55, s85 // (op 303) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_303 // (op 303) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_303 if false)
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_ReceiveTileLoop_303:
s_add_u32 s55, s16, 1 // (op 5828) 
v_lshrrev_b32 v20, 5, v1 // (op 5828) 
s_add_u32 s84, s55, s54 // (op 5828) 
s_lshl_b32 s84, s84, 6 // (op 5828) 

v_lshlrev_b32 v21, 2, v20 // (op 5828) 

v_add_lshl_u32 v20, s84, v21, 6 // (op 5828) 


v_and_b32 v21, -32, v1 // (op 5828) 
v_sub_u32 v23, v1, v21 // (op 5828) 

v_add_lshl_u32 v21, v20, v23, 2 // (op 5828) 


s_lshl_b32 s55, s15, 14 // (op 5830) 
s_mov_b32 s56, s28 // (op 5830) reg expression
s_mov_b32 s57, s29 // (op 5830) reg expression
s_mov_b32 s58, s55 // (op 5830) reg expression
s_mov_b32 s59, 131072 // (op 5830) call()

RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_342:
s_add_u32 s55, s16, 1 // (op 340) 
s_add_u32 s84, s55, s54 // (op 340) Start of Int64 add, locking SCC
s_addc_u32 s85, 0, 0 // (op 340) End of Int64 add, unlocking SCC
s_lshl_b64 s[84:85], s[84:85], 2 // (op 340) 

s_add_u32 s86, s50, s84 // (op 340) Start of Int64 add, locking SCC
s_addc_u32 s87, s51, s85 // (op 340) End of Int64 add, unlocking SCC
s_load_dword s55, s[86:87], 0 glc // (op 340) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s55, 0 // (op 342) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_342 // (op 342) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_342 if true)
s_barrier  // (op 346) 
v_cmp_eq_i32 vcc, v1, 0 // (op 345) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_345 // (op 345) Condition: False, jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_345
s_mov_b32 s55, 0 // (op 343) call()
s_add_u32 s84, s16, 1 // (op 344) 
s_add_u32 s86, s84, s54 // (op 344) Start of Int64 add, locking SCC
s_addc_u32 s87, 0, 0 // (op 344) End of Int64 add, unlocking SCC
s_lshl_b64 s[86:87], s[86:87], 2 // (op 344) 

s_add_u32 s84, s50, s86 // (op 344) Start of Int64 add, locking SCC
s_addc_u32 s85, s51, s87 // (op 344) End of Int64 add, unlocking SCC
s_store_dword s55, s[84:85], 0 glc // (op 344) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_345 // (op 345) Condition: Done, jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_345
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_345:
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_345:




buffer_load_dword v28, v21, s[56:59], 0 offen // (op 357) Load value
buffer_load_dword v29, v21, s[56:59], 0 offen offset:256 // (op 357) Load value
buffer_load_dword v30, v21, s[56:59], 0 offen offset:512 // (op 357) Load value
buffer_load_dword v31, v21, s[56:59], 0 offen offset:768 // (op 357) Load value
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




buffer_load_dword v28, v21, s[56:59], 0 offen offset:128 // (op 2369) Load value
buffer_load_dword v29, v21, s[56:59], 0 offen offset:384 // (op 2369) Load value
buffer_load_dword v30, v21, s[56:59], 0 offen offset:640 // (op 2369) Load value
buffer_load_dword v31, v21, s[56:59], 0 offen offset:896 // (op 2369) Load value
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

buffer_load_dword v28, v23, s[56:59], 0 offen // (op 2394) Load value

v_mov_b32 v20, 8448 // (op 2394) 
v_add_i32 v23, v21, v20 // (op 2394) 

buffer_load_dword v29, v23, s[56:59], 0 offen // (op 2394) Load value

v_mov_b32 v20, 8704 // (op 2394) 
v_add_i32 v23, v21, v20 // (op 2394) 

buffer_load_dword v30, v23, s[56:59], 0 offen // (op 2394) Load value

v_mov_b32 v20, 8960 // (op 2394) 
v_add_i32 v23, v21, v20 // (op 2394) 

buffer_load_dword v31, v23, s[56:59], 0 offen // (op 2394) Load value

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

buffer_load_dword v28, v23, s[56:59], 0 offen // (op 2399) Load value

v_mov_b32 v20, 8576 // (op 2399) 
v_add_i32 v23, v21, v20 // (op 2399) 

buffer_load_dword v29, v23, s[56:59], 0 offen // (op 2399) Load value

v_mov_b32 v20, 8832 // (op 2399) 
v_add_i32 v23, v21, v20 // (op 2399) 

buffer_load_dword v30, v23, s[56:59], 0 offen // (op 2399) Load value

v_mov_b32 v20, 9088 // (op 2399) 
v_add_i32 v23, v21, v20 // (op 2399) 

buffer_load_dword v31, v23, s[56:59], 0 offen // (op 2399) Load value

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

s_add_u32 s54, s54, 1 // (op 305) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s55, s54, s8 // (op 303) 
s_ashr_i64 s[84:85], s[2:3], 7 // (op 303) 
s_mov_b32 s86, s84 // (op 303) convert

s_sub_u32 s84, s86, s19 // (op 303) 

s_sub_u32 s85, s84, 1 // (op 303) 

s_cmp_lt_u32 s55, s85 // (op 303) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_ReceiveTileLoop_303 // (op 303) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_ReceiveTileLoop_303 if true)
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_303:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Receive_Tile_338 // (op 338) Condition: Done, jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Receive_Tile_338
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Receive_Tile_338:
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Receive_Tile_338:

s_cmp_eq_i32 s18, 0 // (op 381) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_381 // (op 381) Condition: False, jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_381
v_cmp_eq_f32 vcc, 0.00000, v18 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_135
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

buffer_store_dwordx2 v[20:21], v23, s[64:67], 0 offen // (op 58) Store value



v_accvgpr_read v28, a4 // (op 2429) 
v_accvgpr_read v29, a5 // (op 2429) 
v_accvgpr_read v30, a6 // (op 2429) 
v_accvgpr_read v31, a7 // (op 2429) 
v_mul_f32 v32, v19, v28 // (op 2429) 
v_mul_f32 v33, v19, v29 // (op 2429) 
v_mul_f32 v34, v19, v30 // (op 2429) 
v_mul_f32 v35, v19, v31 // (op 2429) 

v_cvt_f16_f32 v28, v32 // (op 2431) 
v_cvt_f16_f32 v29, v33 // (op 2431) 
v_cvt_f16_f32 v30, v34 // (op 2431) 
v_cvt_f16_f32 v31, v35 // (op 2431) 


v_pack_b32_f16 v20, v28, v29 // (op 2432) 
v_pack_b32_f16 v21, v30, v31 // (op 2432) 

s_lshl_b32 s54, s24, 6 // (op 2432) 
v_add_u32 v23, v27, s54 // (op 2432) 


buffer_store_dwordx2 v[20:21], v23, s[64:67], 0 offen // (op 2432) Store value



v_accvgpr_read v28, a8 // (op 2455) 
v_accvgpr_read v29, a9 // (op 2455) 
v_accvgpr_read v30, a10 // (op 2455) 
v_accvgpr_read v31, a11 // (op 2455) 
v_mul_f32 v32, v19, v28 // (op 2455) 
v_mul_f32 v33, v19, v29 // (op 2455) 
v_mul_f32 v34, v19, v30 // (op 2455) 
v_mul_f32 v35, v19, v31 // (op 2455) 

v_cvt_f16_f32 v28, v32 // (op 2457) 
v_cvt_f16_f32 v29, v33 // (op 2457) 
v_cvt_f16_f32 v30, v34 // (op 2457) 
v_cvt_f16_f32 v31, v35 // (op 2457) 


v_pack_b32_f16 v20, v28, v29 // (op 2459) 
v_pack_b32_f16 v21, v30, v31 // (op 2459) 

v_add_u32 v23, 64, v27 // (op 2459) 

buffer_store_dwordx2 v[20:21], v23, s[64:67], 0 offen // (op 2459) Store value



v_accvgpr_read v28, a12 // (op 2465) 
v_accvgpr_read v29, a13 // (op 2465) 
v_accvgpr_read v30, a14 // (op 2465) 
v_accvgpr_read v31, a15 // (op 2465) 
v_mul_f32 v32, v19, v28 // (op 2465) 
v_mul_f32 v33, v19, v29 // (op 2465) 
v_mul_f32 v34, v19, v30 // (op 2465) 
v_mul_f32 v35, v19, v31 // (op 2465) 

v_cvt_f16_f32 v28, v32 // (op 2467) 
v_cvt_f16_f32 v29, v33 // (op 2467) 
v_cvt_f16_f32 v30, v34 // (op 2467) 
v_cvt_f16_f32 v31, v35 // (op 2467) 


v_pack_b32_f16 v20, v28, v29 // (op 2461) 
v_pack_b32_f16 v21, v30, v31 // (op 2461) 

s_lshl_b32 s54, s24, 6 // (op 2461) 
s_add_u32 s54, s54, 64 // (op 2461) 
v_add_u32 v23, v27, s54 // (op 2461) 


buffer_store_dwordx2 v[20:21], v23, s[64:67], 0 offen // (op 2461) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_135
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_135:


v_add_u32 v20, 64, v22 // (op 2551) 

buffer_load_dwordx2 v[24:25], v20, s[60:63], 0 offen // (op 2551) Load value

v_accvgpr_read v28, a8 // (op 2542) 
v_accvgpr_read v29, a9 // (op 2542) 
v_accvgpr_read v30, a10 // (op 2542) 
v_accvgpr_read v31, a11 // (op 2542) 
v_mul_f32 v32, v19, v28 // (op 2542) 
v_mul_f32 v33, v19, v29 // (op 2542) 
v_mul_f32 v34, v19, v30 // (op 2542) 
v_mul_f32 v35, v19, v31 // (op 2542) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 2550) 
v_lshrrev_b32 v21, 16, v24 // (op 2550) 
v_cvt_f32_f16 v21, v21 // (op 2550) 
v_mul_f32 v28, v18, v20 // (op 2550) 
v_mul_f32 v29, v18, v21 // (op 2550) 
v_cvt_f32_f16 v20, v25 // (op 2550) 
v_lshrrev_b32 v21, 16, v25 // (op 2550) 
v_cvt_f32_f16 v21, v21 // (op 2550) 
v_mul_f32 v30, v18, v20 // (op 2550) 
v_mul_f32 v31, v18, v21 // (op 2550) 


v_add_f32 v36, v28, v32 // (op 2543) 
v_add_f32 v37, v29, v33 // (op 2543) 
v_add_f32 v38, v30, v34 // (op 2543) 
v_add_f32 v39, v31, v35 // (op 2543) 


v_cvt_f16_f32 v28, v36 // (op 2544) 
v_cvt_f16_f32 v29, v37 // (op 2544) 
v_cvt_f16_f32 v30, v38 // (op 2544) 
v_cvt_f16_f32 v31, v39 // (op 2544) 


v_pack_b32_f16 v20, v28, v29 // (op 2546) 
v_pack_b32_f16 v21, v30, v31 // (op 2546) 

v_add_u32 v23, 64, v27 // (op 2546) 

buffer_store_dwordx2 v[20:21], v23, s[64:67], 0 offen // (op 2546) Store value





s_lshl_b32 s54, s4, 6 // (op 2556) 
s_add_u32 s54, s54, 64 // (op 2556) 
v_add_u32 v20, v22, s54 // (op 2556) 


buffer_load_dwordx2 v[24:25], v20, s[60:63], 0 offen // (op 2556) Load value

v_accvgpr_read v28, a12 // (op 2558) 
v_accvgpr_read v29, a13 // (op 2558) 
v_accvgpr_read v30, a14 // (op 2558) 
v_accvgpr_read v31, a15 // (op 2558) 
v_mul_f32 v32, v19, v28 // (op 2558) 
v_mul_f32 v33, v19, v29 // (op 2558) 
v_mul_f32 v34, v19, v30 // (op 2558) 
v_mul_f32 v35, v19, v31 // (op 2558) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 2553) 
v_lshrrev_b32 v21, 16, v24 // (op 2553) 
v_cvt_f32_f16 v21, v21 // (op 2553) 
v_mul_f32 v28, v18, v20 // (op 2553) 
v_mul_f32 v29, v18, v21 // (op 2553) 
v_cvt_f32_f16 v20, v25 // (op 2553) 
v_lshrrev_b32 v21, 16, v25 // (op 2553) 
v_cvt_f32_f16 v21, v21 // (op 2553) 
v_mul_f32 v30, v18, v20 // (op 2553) 
v_mul_f32 v31, v18, v21 // (op 2553) 


v_add_f32 v36, v28, v32 // (op 2554) 
v_add_f32 v37, v29, v33 // (op 2554) 
v_add_f32 v38, v30, v34 // (op 2554) 
v_add_f32 v39, v31, v35 // (op 2554) 


v_cvt_f16_f32 v28, v36 // (op 2555) 
v_cvt_f16_f32 v29, v37 // (op 2555) 
v_cvt_f16_f32 v30, v38 // (op 2555) 
v_cvt_f16_f32 v31, v39 // (op 2555) 


v_pack_b32_f16 v20, v28, v29 // (op 2548) 
v_pack_b32_f16 v21, v30, v31 // (op 2548) 

s_lshl_b32 s54, s24, 6 // (op 2548) 
s_add_u32 s54, s54, 64 // (op 2548) 
v_add_u32 v23, v27, s54 // (op 2548) 


buffer_store_dwordx2 v[20:21], v23, s[64:67], 0 offen // (op 2548) Store value





v_mov_b32 v20, v22 // (op 148) reg expression

buffer_load_dwordx2 v[24:25], v20, s[60:63], 0 offen // (op 148) Load value

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

buffer_store_dwordx2 v[20:21], v23, s[64:67], 0 offen // (op 147) Store value





s_lshl_b32 s54, s4, 6 // (op 2511) 
v_add_u32 v20, v22, s54 // (op 2511) 


buffer_load_dwordx2 v[24:25], v20, s[60:63], 0 offen // (op 2511) Load value

v_accvgpr_read v28, a4 // (op 2507) 
v_accvgpr_read v29, a5 // (op 2507) 
v_accvgpr_read v30, a6 // (op 2507) 
v_accvgpr_read v31, a7 // (op 2507) 
v_mul_f32 v32, v19, v28 // (op 2507) 
v_mul_f32 v33, v19, v29 // (op 2507) 
v_mul_f32 v34, v19, v30 // (op 2507) 
v_mul_f32 v35, v19, v31 // (op 2507) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 2512) 
v_lshrrev_b32 v21, 16, v24 // (op 2512) 
v_cvt_f32_f16 v21, v21 // (op 2512) 
v_mul_f32 v28, v18, v20 // (op 2512) 
v_mul_f32 v29, v18, v21 // (op 2512) 
v_cvt_f32_f16 v20, v25 // (op 2512) 
v_lshrrev_b32 v21, 16, v25 // (op 2512) 
v_cvt_f32_f16 v21, v21 // (op 2512) 
v_mul_f32 v30, v18, v20 // (op 2512) 
v_mul_f32 v31, v18, v21 // (op 2512) 


v_add_f32 v36, v28, v32 // (op 2508) 
v_add_f32 v37, v29, v33 // (op 2508) 
v_add_f32 v38, v30, v34 // (op 2508) 
v_add_f32 v39, v31, v35 // (op 2508) 


v_cvt_f16_f32 v28, v36 // (op 2509) 
v_cvt_f16_f32 v29, v37 // (op 2509) 
v_cvt_f16_f32 v30, v38 // (op 2509) 
v_cvt_f16_f32 v31, v39 // (op 2509) 


v_pack_b32_f16 v20, v28, v29 // (op 2510) 
v_pack_b32_f16 v21, v30, v31 // (op 2510) 

s_lshl_b32 s54, s24, 6 // (op 2510) 
v_add_u32 v23, v27, s54 // (op 2510) 


buffer_store_dwordx2 v[20:21], v23, s[64:67], 0 offen // (op 2510) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_135:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_381 // (op 381) Condition: Done, jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_381
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_381:
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_381:


s_mul_i32 s54, s8, s16 // (op 294) 
s_cmp_gt_u32 s9, s54 // (op 294) 
s_mov_b32 s55, scc // (op 294) 
s_sub_u32 s56, s9, s54 // (op 294) 

s_cmp_eq_i32 s55, 0 // (op 294) 
s_cselect_b32 s54, 0, s56 // (op 294) 


s_cmp_lt_u32 s8, s54 // (op 294) 
s_mov_b32 s55, scc // (op 294) 
s_cmp_eq_i32 s55, 0 // (op 294) 
s_cselect_b32 s56, s54, s8 // (op 294) 


s_cmp_lt_u32 s0, s56 // (op 294) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_SKStreamTileLoop_294 // (op 294) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_SKStreamTileLoop_294 if true)
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_294:

















s_mov_b32 s0, 0 // (op 435) call()
s_mov_b32 s1, 0 // (op 436) call()
s_mov_b32 s0, 0 // (op 450) call()
s_mul_i32 s15, s52, s16 // (op 449) 
s_cmp_gt_u32 s53, s15 // (op 449) 
s_mov_b32 s6, scc // (op 449) 
s_sub_u32 s7, s53, s15 // (op 449) 

s_cmp_eq_i32 s6, 0 // (op 449) 
s_cselect_b32 s8, 0, s7 // (op 449) 


s_cmp_lt_u32 s52, s8 // (op 449) 
s_mov_b32 s15, scc // (op 449) 
s_cmp_eq_i32 s15, 0 // (op 449) 
s_cselect_b32 s6, s8, s52 // (op 449) 


s_cmp_lt_u32 s0, s6 // (op 449) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_449 // (op 449) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_449 if false)
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_DPStreamTileLoop_449:
v_lshrrev_b32 v0, 7, v1 // (op 5118) 
s_mul_i32 s15, s16, s52 // (op 5118) 
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

s_mov_b32 s60, s20 // (op 5120) reg expression
s_mov_b32 s61, s21 // (op 5120) reg expression
s_mov_b32 s62, s28 // (op 5120) reg expression
s_mov_b32 s63, 131072 // (op 5120) call()

v_lshrrev_b32 v0, 7, v1 // (op 5472) 
s_mul_i32 s15, s16, s52 // (op 5472) 
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

s_mov_b32 s64, s26 // (op 5474) reg expression
s_mov_b32 s65, s27 // (op 5474) reg expression
s_mov_b32 s66, s28 // (op 5474) reg expression
s_mov_b32 s67, 131072 // (op 5474) call()

v_lshrrev_b32 v0, 7, v1 // (op 5551) 
v_and_b32 v3, 15, v1 // (op 5551) 
v_lshl_add_u32 v14, v0, 4, v3 // (op 5551) 


v_and_b32 v0, 48, v1 // (op 5551) 
v_lshl_add_u32 v3, v14, 7, v0 // (op 5551) 


v_and_b32 v0, 48, v1 // (op 5626) 
v_and_b32 v14, 63, v1 // (op 5626) 
v_sub_u32 v15, v14, v0 // (op 5626) 

v_lshrrev_b32 v14, 1, v15 // (op 5626) 
v_add_lshl_u32 v16, v0, v14, 6 // (op 5626) 


v_and_b32 v0, 1, v15 // (op 5626) 

v_lshlrev_b32 v17, 3, v0 // (op 5626) 

v_lshrrev_b32 v0, 6, v1 // (op 5626) 
v_and_b32 v14, 1, v0 // (op 5626) 

v_lshl_add_u32 v0, v14, 4, v17 // (op 5626) 


v_add_u32 v17, v16, v0 // (op 5626) 


v_lshrrev_b32 v0, 7, v1 // (op 5702) 
v_and_b32 v14, 15, v1 // (op 5702) 
v_lshl_add_u32 v15, v0, 4, v14 // (op 5702) 


v_and_b32 v0, 63, v1 // (op 5702) 
v_lshrrev_b32 v14, 4, v0 // (op 5702) 

v_lshl_add_u32 v0, v15, 2, v14 // (op 5702) 


v_lshrrev_b32 v14, 6, v1 // (op 5778) 
v_and_b32 v15, 1, v14 // (op 5778) 

v_and_b32 v14, 15, v1 // (op 5778) 
v_lshl_add_u32 v16, v15, 4, v14 // (op 5778) 


v_and_b32 v14, 63, v1 // (op 5778) 
v_lshrrev_b32 v15, 4, v14 // (op 5778) 

v_lshl_add_u32 v14, v15, 6, v16 // (op 5778) 


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
v_lshrrev_b32 v15, 3, v1 // (op 4610) 
s_mul_i32 s15, s16, s52 // (op 4610) 
s_add_u32 s15, s15, s0 // (op 4610) 
v_and_b32 v16, -8, v1 // (op 4610) 
s_add_u32 s6, s15, s9 // (op 4610) 

v_sub_u32 v20, v1, v16 // (op 4610) 

s_mul_hi_u32 s15, s6, s13 // (op 4610) 
v_lshlrev_b32 v16, 4, v20 // (op 4610) 

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


v_lshl_add_u32 v20, s15, 6, v15 // (op 4610) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4610) 
s_mul_i32 s8, s7, s18 // (op 4610) 


s_sub_u32 s7, s6, s8 // (op 4610) 


v_lshl_add_u32 v15, s7, 7, v16 // (op 4610) 


v_mov_b32 v23, 0 // (op 4610) convert
v_mov_b32 v22, v15 // (op 4610) convert

v_mul_lo_u32 v25, v20, s31 // (op 4610) most significant: low of low * high
v_mul_hi_u32 v21, v20, s30 // (op 4610) most significant: high of low * low
v_mul_lo_u32 v24, v20, s30 // (op 4610) least significant: low of low * low
v_add_u32 v25, v25, v21 // (op 4610) most significant: sum

v_add_co_u32 v24, s[18:19], v24, v22 // (op 4610) least significant half
v_addc_co_u32 v25, s[18:19], v25, v23, s[18:19] // (op 4610) most significant half



s_mul_i32 s19, s30, s35 // (op 4612) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 4612) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 4612) least significant
s_mul_hi_u32 s6, s30, s34 // (op 4612) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4612) most significant: sum
s_add_u32 s19, s19, s6 // (op 4612) most significant: sum


s_mov_b32 s68, s32 // (op 4612) reg expression
s_mov_b32 s69, s33 // (op 4612) reg expression
s_mov_b32 s70, s18 // (op 4612) reg expression
s_mov_b32 s71, 131072 // (op 4612) call()

v_lshrrev_b32 v15, 3, v1 // (op 4629) 
s_mul_i32 s15, s16, s52 // (op 4629) 
s_add_u32 s15, s15, s0 // (op 4629) 
v_and_b32 v16, -8, v1 // (op 4629) 
s_add_u32 s6, s15, s9 // (op 4629) 

v_sub_u32 v20, v1, v16 // (op 4629) 

s_mul_hi_u32 s15, s6, s13 // (op 4629) 
v_lshlrev_b32 v16, 4, v20 // (op 4629) 

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


v_lshl_add_u32 v20, s15, 6, v15 // (op 4629) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4629) 
s_mul_i32 s8, s7, s18 // (op 4629) 


s_sub_u32 s7, s6, s8 // (op 4629) 


v_lshl_add_u32 v15, s7, 7, v16 // (op 4629) 


v_mov_b32 v23, 0 // (op 4629) convert
v_mov_b32 v22, v15 // (op 4629) convert

v_mul_lo_u32 v27, v20, s31 // (op 4629) most significant: low of low * high
v_mul_hi_u32 v21, v20, s30 // (op 4629) most significant: high of low * low
v_mul_lo_u32 v26, v20, s30 // (op 4629) least significant: low of low * low
v_add_u32 v27, v27, v21 // (op 4629) most significant: sum

v_add_co_u32 v26, s[18:19], v26, v22 // (op 4629) least significant half
v_addc_co_u32 v27, s[18:19], v27, v23, s[18:19] // (op 4629) most significant half



s_mul_i32 s19, s30, s35 // (op 4631) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 4631) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 4631) least significant
s_mul_hi_u32 s6, s30, s34 // (op 4631) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4631) most significant: sum
s_add_u32 s19, s19, s6 // (op 4631) most significant: sum


s_mov_b32 s68, s32 // (op 4631) reg expression
s_mov_b32 s69, s33 // (op 4631) reg expression
s_mov_b32 s70, s18 // (op 4631) reg expression
s_mov_b32 s71, 131072 // (op 4631) call()

v_and_b32 v15, -4, v1 // (op 4744) 
s_mul_i32 s15, s16, s52 // (op 4744) 
s_add_u32 s15, s15, s0 // (op 4744) 
v_sub_u32 v16, v1, v15 // (op 4744) 

s_add_u32 s6, s15, s9 // (op 4744) 

v_lshlrev_b32 v15, 4, v16 // (op 4744) 

s_mul_hi_u32 s15, s6, s13 // (op 4744) 
v_lshrrev_b32 v16, 2, v1 // (op 4744) 
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

v_lshl_add_u32 v20, s15, 6, v15 // (op 4744) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4744) 
v_mov_b32 v23, 0 // (op 4744) convert
v_mov_b32 v22, v20 // (op 4744) convert

s_mul_i32 s8, s7, s18 // (op 4744) 


s_sub_u32 s7, s6, s8 // (op 4744) 


v_lshl_add_u32 v15, s7, 7, v16 // (op 4744) 


v_mul_lo_u32 v21, v15, s37 // (op 4744) most significant: low of low * high
v_mul_hi_u32 v16, v15, s36 // (op 4744) most significant: high of low * low
v_mul_lo_u32 v20, v15, s36 // (op 4744) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 4744) most significant: sum

v_add_co_u32 v20, s[18:19], v20, v22 // (op 4744) least significant half
v_addc_co_u32 v21, s[18:19], v21, v23, s[18:19] // (op 4744) most significant half



s_mul_i32 s19, s36, s3 // (op 4746) most significant: low of low * high
s_mul_i32 s15, s37, s2 // (op 4746) most significant: low of high * low
s_mul_i32 s18, s36, s2 // (op 4746) least significant
s_mul_hi_u32 s6, s36, s2 // (op 4746) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4746) most significant: sum
s_add_u32 s19, s19, s6 // (op 4746) most significant: sum


s_mov_b32 s72, s38 // (op 4746) reg expression
s_mov_b32 s73, s39 // (op 4746) reg expression
s_mov_b32 s74, s18 // (op 4746) reg expression
s_mov_b32 s75, 131072 // (op 4746) call()

v_and_b32 v15, -4, v1 // (op 4763) 
s_mul_i32 s15, s16, s52 // (op 4763) 
s_add_u32 s15, s15, s0 // (op 4763) 
v_sub_u32 v16, v1, v15 // (op 4763) 

s_add_u32 s6, s15, s9 // (op 4763) 

v_lshlrev_b32 v15, 4, v16 // (op 4763) 

s_mul_hi_u32 s15, s6, s13 // (op 4763) 
v_lshrrev_b32 v16, 2, v1 // (op 4763) 
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

v_lshl_add_u32 v22, s15, 6, v15 // (op 4763) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4763) 
v_mov_b32 v29, 0 // (op 4763) convert
v_mov_b32 v28, v22 // (op 4763) convert

s_mul_i32 s8, s7, s18 // (op 4763) 


s_sub_u32 s7, s6, s8 // (op 4763) 


v_lshl_add_u32 v15, s7, 7, v16 // (op 4763) 


v_mul_lo_u32 v23, v15, s37 // (op 4763) most significant: low of low * high
v_mul_hi_u32 v16, v15, s36 // (op 4763) most significant: high of low * low
v_mul_lo_u32 v22, v15, s36 // (op 4763) least significant: low of low * low
v_add_u32 v23, v23, v16 // (op 4763) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v28 // (op 4763) least significant half
v_addc_co_u32 v23, s[18:19], v23, v29, s[18:19] // (op 4763) most significant half



s_mul_i32 s19, s36, s3 // (op 4765) most significant: low of low * high
s_mul_i32 s15, s37, s2 // (op 4765) most significant: low of high * low
s_mul_i32 s18, s36, s2 // (op 4765) least significant
s_mul_hi_u32 s6, s36, s2 // (op 4765) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4765) most significant: sum
s_add_u32 s19, s19, s6 // (op 4765) most significant: sum


s_mov_b32 s72, s38 // (op 4765) reg expression
s_mov_b32 s73, s39 // (op 4765) reg expression
s_mov_b32 s74, s18 // (op 4765) reg expression
s_mov_b32 s75, 131072 // (op 4765) call()

v_lshrrev_b32 v15, 2, v1 // (op 4872) 
s_mul_i32 s15, s16, s52 // (op 4872) 
s_add_u32 s15, s15, s0 // (op 4872) 
v_and_b32 v16, -4, v1 // (op 4872) 
s_add_u32 s6, s15, s9 // (op 4872) 

v_sub_u32 v28, v1, v16 // (op 4872) 

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


v_lshl_add_u32 v16, s15, 6, v15 // (op 4872) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4872) 
s_mul_i32 s8, s7, s18 // (op 4872) 


s_sub_u32 s7, s6, s8 // (op 4872) 


v_lshl_add_u32 v15, s7, 2, v28 // (op 4872) 


v_mul_lo_u32 v28, v16, s40 // (op 4872) 
v_add_u32 v28, v28, v15 // (op 4872) 


s_mul_i32 s19, s40, s35 // (op 4874) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 4874) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 4874) least significant
s_mul_hi_u32 s6, s40, s34 // (op 4874) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4874) most significant: sum
s_add_u32 s19, s19, s6 // (op 4874) most significant: sum


s_mov_b32 s76, s42 // (op 4874) reg expression
s_mov_b32 s77, s43 // (op 4874) reg expression
s_mov_b32 s78, s18 // (op 4874) reg expression
s_mov_b32 s79, 131072 // (op 4874) call()

v_lshrrev_b32 v15, 2, v1 // (op 4891) 
s_mul_i32 s15, s16, s52 // (op 4891) 
s_add_u32 s15, s15, s0 // (op 4891) 
v_and_b32 v16, -4, v1 // (op 4891) 
s_add_u32 s6, s15, s9 // (op 4891) 

v_sub_u32 v29, v1, v16 // (op 4891) 

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


v_lshl_add_u32 v16, s15, 6, v15 // (op 4891) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4891) 
s_mul_i32 s8, s7, s18 // (op 4891) 


s_sub_u32 s7, s6, s8 // (op 4891) 


v_lshl_add_u32 v15, s7, 2, v29 // (op 4891) 


v_mul_lo_u32 v29, v16, s40 // (op 4891) 
v_add_u32 v29, v29, v15 // (op 4891) 


s_mul_i32 s19, s40, s35 // (op 4893) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 4893) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 4893) least significant
s_mul_hi_u32 s6, s40, s34 // (op 4893) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4893) most significant: sum
s_add_u32 s19, s19, s6 // (op 4893) most significant: sum


s_mov_b32 s76, s42 // (op 4893) reg expression
s_mov_b32 s77, s43 // (op 4893) reg expression
s_mov_b32 s78, s18 // (op 4893) reg expression
s_mov_b32 s79, 131072 // (op 4893) call()

v_and_b32 v15, -64, v1 // (op 5000) 
s_mul_i32 s15, s16, s52 // (op 5000) 
s_add_u32 s15, s15, s0 // (op 5000) 
v_sub_u32 v16, v1, v15 // (op 5000) 

s_add_u32 s6, s15, s9 // (op 5000) 

v_lshrrev_b32 v15, 6, v1 // (op 5000) 
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

v_lshl_add_u32 v30, s15, 6, v16 // (op 5000) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 5000) 
s_mul_i32 s8, s7, s18 // (op 5000) 


s_sub_u32 s7, s6, s8 // (op 5000) 


v_lshl_add_u32 v16, s7, 2, v15 // (op 5000) 


v_mul_lo_u32 v15, v16, s44 // (op 5000) 
v_add_u32 v15, v15, v30 // (op 5000) 


s_mul_i32 s19, s44, s49 // (op 5002) most significant: low of low * high
s_mul_i32 s15, s45, s48 // (op 5002) most significant: low of high * low
s_mul_i32 s18, s44, s48 // (op 5002) least significant
s_mul_hi_u32 s6, s44, s48 // (op 5002) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 5002) most significant: sum
s_add_u32 s19, s19, s6 // (op 5002) most significant: sum


s_mov_b32 s80, s46 // (op 5002) reg expression
s_mov_b32 s81, s47 // (op 5002) reg expression
s_mov_b32 s82, s18 // (op 5002) reg expression
s_mov_b32 s83, 131072 // (op 5002) call()

v_and_b32 v16, -64, v1 // (op 5019) 
s_mul_i32 s15, s16, s52 // (op 5019) 
s_add_u32 s15, s15, s0 // (op 5019) 
v_sub_u32 v30, v1, v16 // (op 5019) 

s_add_u32 s6, s15, s9 // (op 5019) 

v_lshrrev_b32 v16, 6, v1 // (op 5019) 
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

v_lshl_add_u32 v31, s15, 6, v30 // (op 5019) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 5019) 
s_mul_i32 s8, s7, s18 // (op 5019) 


s_sub_u32 s7, s6, s8 // (op 5019) 


v_lshl_add_u32 v30, s7, 2, v16 // (op 5019) 


v_mul_lo_u32 v16, v30, s44 // (op 5019) 
v_add_u32 v16, v16, v31 // (op 5019) 


s_mul_i32 s19, s44, s49 // (op 5021) most significant: low of low * high
s_mul_i32 s15, s45, s48 // (op 5021) most significant: low of high * low
s_mul_i32 s18, s44, s48 // (op 5021) least significant
s_mul_hi_u32 s6, s44, s48 // (op 5021) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 5021) most significant: sum
s_add_u32 s19, s19, s6 // (op 5021) most significant: sum


s_mov_b32 s80, s46 // (op 5021) reg expression
s_mov_b32 s81, s47 // (op 5021) reg expression
s_mov_b32 s82, s18 // (op 5021) reg expression
s_mov_b32 s83, 131072 // (op 5021) call()

v_and_b32 v30, -4, v1 // (op 5592) 
v_sub_u32 v31, v1, v30 // (op 5592) 

v_lshlrev_b32 v30, 4, v31 // (op 5592) 

v_lshrrev_b32 v31, 2, v1 // (op 5592) 
v_lshl_add_u32 v32, v31, 6, v30 // (op 5592) 


s_nop 0
v_readfirstlane_b32 s15, v32 // (op 5592) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6353) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 4518) 
v_mov_b64 v[30:31], s[18:19] // (op 4518) convert


v_mov_b64 v[32:33], v[20:21] // (op 4518) reg expression
s_mov_b32 m0, s15 // (op 4518) reg expression
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 4518) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 4518) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 4518) most significant half

s_add_u32 m0, m0, 4096 // (op 4518) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 4518) Load value direct to lds





v_mov_b32 v30, v15 // (op 4166) reg expression

buffer_load_ubyte v31, v30, s[80:83], 0 offen offset: 0 // (op 4166) Load value

v_and_b32 v30, -8, v1 // (op 5517) 
v_sub_u32 v32, v1, v30 // (op 5517) 

v_lshlrev_b32 v30, 4, v32 // (op 5517) 

v_lshrrev_b32 v32, 3, v1 // (op 5517) 
v_lshl_add_u32 v33, v32, 7, v30 // (op 5517) 


s_nop 0
v_readfirstlane_b32 s15, v33 // (op 5517) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6334) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 4523) 
v_mov_b64 v[32:33], s[18:19] // (op 4523) convert


v_mov_b64 v[34:35], v[24:25] // (op 4523) 
s_add_u32 s6, 8192, s15 // (op 4523) 
s_mov_b32 m0, s6 // (op 4523) reg expression
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 4523) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 4523) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 4523) most significant half

s_add_u32 m0, m0, 4096 // (op 4523) 
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 4523) Load value direct to lds







buffer_load_ubyte v30, v28, s[76:79], 0 offen offset: 0 // (op 4172) Load value
v_and_b32 v32, -4, v1 // (op 5902) 
v_sub_u32 v33, v1, v32 // (op 5902) 

v_lshlrev_b32 v32, 4, v33 // (op 5902) 

v_lshrrev_b32 v33, 2, v1 // (op 5902) 
v_lshl_add_u32 v34, v33, 6, v32 // (op 5902) 


s_nop 0
v_readfirstlane_b32 s15, v34 // (op 5902) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6424) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 4528) 
v_mov_b64 v[32:33], s[18:19] // (op 4528) convert


s_lshl_b64 s[18:19], s[36:37], 7 // (op 4528) 
v_mov_b32 v34, s18 // (op 4528) 
v_mov_b32 v35, s19 // (op 4528) 
v_add_co_u32 v36, s[28:29], v20, v34 // (op 4528) least significant half
v_addc_co_u32 v37, s[28:29], v21, v35, s[28:29] // (op 4528) most significant half




s_add_u32 s6, 16384, s15 // (op 4528) 
s_mov_b32 m0, s6 // (op 4528) reg expression
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 4528) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 4528) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 4528) most significant half

s_add_u32 m0, m0, 4096 // (op 4528) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 4528) Load value direct to lds







s_lshl_b32 s15, s44, 2 // (op 4178) 
v_add_u32 v20, v15, s15 // (op 4178) 


buffer_load_ubyte v21, v20, s[80:83], 0 offen offset: 0 // (op 4178) Load value


v_and_b32 v15, -8, v1 // (op 5858) 
v_sub_u32 v20, v1, v15 // (op 5858) 

v_lshlrev_b32 v15, 4, v20 // (op 5858) 

v_lshrrev_b32 v20, 3, v1 // (op 5858) 
v_lshl_add_u32 v32, v20, 7, v15 // (op 5858) 


s_nop 0
v_readfirstlane_b32 s15, v32 // (op 5858) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 6414) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 4530) 
v_mov_b64 v[32:33], s[18:19] // (op 4530) convert


v_mov_b64 v[34:35], v[24:25] // (op 4530) 
s_add_u32 s6, 24576, s15 // (op 4530) 
s_mov_b32 m0, s6 // (op 4530) reg expression
s_mov_b32 s7, 128 // (op 4530) call()
buffer_load_dwordx4 v34, s[68:71], s7 offen offset: 0 lds // (op 4530) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 4530) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 4530) most significant half

s_add_u32 m0, m0, 4096 // (op 4530) 
buffer_load_dwordx4 v34, s[68:71], s7 offen offset: 0 lds // (op 4530) Load value direct to lds









buffer_load_ubyte v15, v28, s[76:79], 0 offen offset:4 // (op 4184) Load value

v_and_b32 v20, -64, v1 // (op 5744) 
v_sub_u32 v28, v1, v20 // (op 5744) 

v_and_b32 v20, 4294967232, v1 // (op 5744) 
v_add_u32 v24, v20, v28 // (op 5744) 


s_barrier  // (op 6394) 



s_waitcnt vmcnt(9)
ds_write_b8 v24, v31 offset:32768 // (op 4187) Store local data 

v_and_b32 v20, -4, v1 // (op 5668) 
v_sub_u32 v28, v1, v20 // (op 5668) 

v_and_b32 v20, 4294967292, v1 // (op 5668) 
v_add_u32 v24, v20, v28 // (op 5668) 


s_barrier  // (op 6373) 



s_waitcnt vmcnt(6)
ds_write_b8 v24, v30 offset:33024 // (op 4189) Store local data 

s_barrier  // (op 6350) 



ds_read_b64_tr_b8 v[32:33], v17 // (op 4214) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v17 offset:512 // (op 4214) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v17 offset:4096 // (op 4214) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v17 offset:4608 // (op 4214) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 6391) 


ds_read_u8 v20, v14 offset:32768 // (op 4234) Load local data 



ds_read_b64_tr_b8 v[40:41], v17 offset:32 // (op 4219) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v17 offset:544 // (op 4219) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v17 offset:4128 // (op 4219) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v17 offset:4640 // (op 4219) Transpose load local data 


ds_read_u8 v28, v14 offset:32800 // (op 4239) Load local data 
s_barrier  // (op 6331) 



ds_read_b128 v[48:51], v3 offset:8192 // (op 4204) Load local data 
ds_read_b128 v[52:55], v3 offset:8256 // (op 4204) Load local data 
s_waitcnt lgkmcnt(12)
s_barrier  // (op 6370) 



ds_read_u8 v24, v0 offset:33024 // (op 4224) Load local data 



ds_read_b128 v[56:59], v3 offset:12288 // (op 4209) Load local data 
ds_read_b128 v[60:63], v3 offset:12352 // (op 4209) Load local data 



ds_read_u8 v25, v0 offset:33152 // (op 4229) Load local data 
v_and_b32 v64, -8, v1 // (op 5489) 
v_sub_u32 v65, v1, v64 // (op 5489) 

v_lshlrev_b32 v64, 4, v65 // (op 5489) 

v_lshrrev_b32 v65, 3, v1 // (op 5489) 
v_lshl_add_u32 v66, v65, 7, v64 // (op 5489) 


s_nop 0
v_readfirstlane_b32 s15, v66 // (op 5489) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v64, -4, v1 // (op 5564) 
v_sub_u32 v65, v1, v64 // (op 5564) 

v_lshlrev_b32 v64, 4, v65 // (op 5564) 

v_lshrrev_b32 v65, 2, v1 // (op 5564) 
v_lshl_add_u32 v66, v65, 6, v64 // (op 5564) 


s_nop 0
v_readfirstlane_b32 s6, v66 // (op 5564) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v64, -4, v1 // (op 5639) 
v_sub_u32 v65, v1, v64 // (op 5639) 

v_and_b32 v64, 4294967292, v1 // (op 5639) 
v_add_u32 v66, v64, v65 // (op 5639) 


v_and_b32 v64, -64, v1 // (op 5715) 
v_sub_u32 v65, v1, v64 // (op 5715) 

v_and_b32 v64, 4294967232, v1 // (op 5715) 
v_add_u32 v67, v64, v65 // (op 5715) 


v_and_b32 v64, -8, v1 // (op 5843) 
v_sub_u32 v65, v1, v64 // (op 5843) 

v_lshlrev_b32 v64, 4, v65 // (op 5843) 

v_lshrrev_b32 v65, 3, v1 // (op 5843) 
v_lshl_add_u32 v68, v65, 7, v64 // (op 5843) 


s_nop 0
v_readfirstlane_b32 s7, v68 // (op 5843) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v64, -4, v1 // (op 5887) 
v_sub_u32 v65, v1, v64 // (op 5887) 

v_lshlrev_b32 v64, 4, v65 // (op 5887) 

v_lshrrev_b32 v65, 2, v1 // (op 5887) 
v_lshl_add_u32 v68, v65, 6, v64 // (op 5887) 


s_nop 0
v_readfirstlane_b32 s8, v68 // (op 5887) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v64, -4, v1 // (op 5931) 
v_sub_u32 v65, v1, v64 // (op 5931) 

v_and_b32 v64, 4294967292, v1 // (op 5931) 
v_add_u32 v68, v64, v65 // (op 5931) 


v_and_b32 v64, -64, v1 // (op 5960) 
v_sub_u32 v65, v1, v64 // (op 5960) 

v_and_b32 v64, 4294967232, v1 // (op 5960) 
v_add_u32 v69, v64, v65 // (op 5960) 


s_mov_b32 s1, 0 // (op 875) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 874) 
s_mov_b32 s28, s18 // (op 874) convert

s_and_b32 s29, s28, 4294967294 // (op 874) 

s_cmp_lt_u32 s1, s29 // (op 874) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_874 // (op 874) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_874 if false)
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoop_874:
s_add_u32 s1, s1, 2 // (op 876) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[32:39], a[0:3], v24, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 198) 
s_barrier  // (op 6356) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:55], v[40:47], a[4:7], v24, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 662) 
s_lshl_b64 s[18:19], s[36:37], 6 // (op 4467) 
v_mov_b64 v[64:65], s[18:19] // (op 4467) convert


v_mov_b64 v[70:71], v[22:23] // (op 4467) 
s_lshl_b64 s[18:19], s[36:37], 7 // (op 4467) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 4467) 

v_mov_b32 v72, s28 // (op 4467) 
v_mov_b32 v73, s29 // (op 4467) 
v_add_co_u32 v74, s[18:19], v70, v72 // (op 4467) least significant half
v_addc_co_u32 v75, s[18:19], v71, v73, s[18:19] // (op 4467) most significant half




s_mov_b32 m0, s6 // (op 4467) reg expression
buffer_load_dwordx4 v74, s[72:75], 0 offen offset: 0 lds // (op 4467) Load value direct to lds
v_add_co_u32 v74, s[18:19], v74, v64 // (op 4467) least significant half
v_addc_co_u32 v75, s[18:19], 0, v75, s[18:19] // (op 4467) most significant half

s_add_u32 m0, m0, 4096 // (op 4467) 
buffer_load_dwordx4 v74, s[72:75], 0 offen offset: 0 lds // (op 4467) Load value direct to lds


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:63], v[32:39], a[8:11], v25, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 959) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:63], v[40:47], a[12:15], v25, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 913) 
v_mov_b32 v64, v16 // (op 204) 
s_lshl_b32 s18, s44, 3 // (op 204) 
v_add_u32 v65, v64, s18 // (op 204) 


buffer_load_ubyte v31, v65, s[80:83], 0 offen offset: 0 // (op 204) Load value

s_barrier  // (op 6337) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 4462) 
v_mov_b64 v[72:73], s[18:19] // (op 4462) convert


v_mov_b64 v[74:75], v[26:27] // (op 4462) 
v_mov_b64 v[76:77], v[74:75] // (op 4462) 
s_add_u32 s18, 8192, s15 // (op 4462) 
s_mov_b32 m0, s18 // (op 4462) reg expression
s_mov_b32 s19, 256 // (op 4462) call()
buffer_load_dwordx4 v76, s[68:71], s19 offen offset: 0 lds // (op 4462) Load value direct to lds
v_add_co_u32 v76, s[28:29], v76, v72 // (op 4462) least significant half
v_addc_co_u32 v77, s[28:29], 0, v77, s[28:29] // (op 4462) most significant half

s_add_u32 m0, m0, 4096 // (op 4462) 
buffer_load_dwordx4 v76, s[68:71], s19 offen offset: 0 lds // (op 4462) Load value direct to lds






v_mov_b32 v65, v29 // (op 203) 

buffer_load_ubyte v30, v65, s[76:79], 0 offen offset:8 // (op 203) Load value
s_barrier  // (op 6440) 



s_waitcnt vmcnt(9)
ds_write_b8 v69, v21 offset:33280 // (op 541) Store local data 
s_barrier  // (op 6434) 



s_waitcnt vmcnt(6)
ds_write_b8 v68, v15 offset:33536 // (op 540) Store local data 
s_barrier  // (op 6420) 



ds_read_b64_tr_b8 v[76:77], v12 offset:16384 // (op 533) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v12 offset:16896 // (op 533) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v12 offset:20480 // (op 533) Transpose load local data 
ds_read_b64_tr_b8 v[82:83], v12 offset:20992 // (op 533) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 6437) 


ds_read_u8 v72, v13 offset:33280 // (op 537) Load local data 



ds_read_b64_tr_b8 v[84:85], v12 offset:16416 // (op 667) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v12 offset:16928 // (op 667) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v12 offset:20512 // (op 667) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v12 offset:21024 // (op 667) Transpose load local data 


ds_read_u8 v73, v13 offset:33312 // (op 671) Load local data 
s_barrier  // (op 6411) 
s_barrier  // (op 6559) 



ds_read_b128 v[48:51], v11 offset:24576 // (op 531) Load local data 
ds_read_b128 v[52:55], v11 offset:24640 // (op 531) Load local data 
s_waitcnt lgkmcnt(12)
s_barrier  // (op 6430) 



ds_read_u8 v24, v10 offset:33536 // (op 535) Load local data 



ds_read_b128 v[56:59], v11 offset:28672 // (op 916) Load local data 
ds_read_b128 v[60:63], v11 offset:28736 // (op 916) Load local data 



ds_read_u8 v25, v10 offset:33664 // (op 919) Load local data 
s_barrier  // (op 6581) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[76:83], a[0:3], v24, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 538) 
s_barrier  // (op 6427) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:55], v[84:91], a[4:7], v24, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 663) 
s_lshl_b64 s[18:19], s[36:37], 6 // (op 4490) 
v_mov_b64 v[92:93], s[18:19] // (op 4490) convert


s_lshl_b64 s[18:19], s[36:37], 7 // (op 4490) 
s_mul_i32 s29, 3, s19 // (op 4490) most significant: low of low * high
s_mul_i32 s28, 3, s18 // (op 4490) least significant
s_mul_hi_u32 s50, 3, s18 // (op 4490) most significant: high of low * low
s_add_u32 s29, s29, s50 // (op 4490) most significant: sum


v_mov_b32 v94, s28 // (op 4490) 
v_mov_b32 v95, s29 // (op 4490) 
v_add_co_u32 v96, s[18:19], v70, v94 // (op 4490) least significant half
v_addc_co_u32 v97, s[18:19], v71, v95, s[18:19] // (op 4490) most significant half




s_add_u32 s18, 16384, s8 // (op 4490) 
s_nop 0
s_mov_b32 m0, s18 // (op 4490) reg expression
buffer_load_dwordx4 v96, s[72:75], 0 offen offset: 0 lds // (op 4490) Load value direct to lds
v_add_co_u32 v96, s[28:29], v96, v92 // (op 4490) least significant half
v_addc_co_u32 v97, s[28:29], 0, v97, s[28:29] // (op 4490) most significant half

s_add_u32 m0, m0, 4096 // (op 4490) 
buffer_load_dwordx4 v96, s[72:75], 0 offen offset: 0 lds // (op 4490) Load value direct to lds



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:63], v[76:83], a[8:11], v25, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 960) 





v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:63], v[84:91], a[12:15], v25, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 914) 
s_lshl_b32 s18, s44, 2 // (op 544) 


s_mul_i32 s19, 3, s18 // (op 544) 

v_add_u32 v70, v64, s19 // (op 544) 


buffer_load_ubyte v21, v70, s[80:83], 0 offen offset: 0 // (op 544) Load value

s_lshl_b64 s[18:19], s[36:37], 7 // (op 4755) 

s_lshl_b64 s[28:29], s[18:19], 1 // (op 4755) 

s_barrier  // (op 6417) 
s_lshl_b32 s18, s44, 3 // (op 5011) 

v_add_u32 v16, v16, s18 // (op 5011) 

v_mov_b32 v64, s28 // (op 4755) 
v_mov_b32 v70, s29 // (op 4755) 
s_lshl_b64 s[18:19], s[30:31], 5 // (op 4485) 
v_mov_b64 v[72:73], s[18:19] // (op 4485) convert


v_mov_b64 v[76:77], v[74:75] // (op 4485) 
s_add_u32 s18, 24576, s7 // (op 4485) 
s_mov_b32 m0, s18 // (op 4485) reg expression
s_mov_b32 s19, 384 // (op 4485) call()
v_add_co_u32 v22, s[50:51], v22, v64 // (op 4755) least significant half
v_addc_co_u32 v23, s[50:51], v23, v70, s[50:51] // (op 4755) most significant half




buffer_load_dwordx4 v76, s[68:71], s19 offen offset: 0 lds // (op 4485) Load value direct to lds
v_add_co_u32 v76, s[28:29], v76, v72 // (op 4485) least significant half
v_addc_co_u32 v77, s[28:29], 0, v77, s[28:29] // (op 4485) most significant half

s_add_u32 m0, m0, 4096 // (op 4485) 
buffer_load_dwordx4 v76, s[68:71], s19 offen offset: 0 lds // (op 4485) Load value direct to lds








v_add_co_u32 v26, vcc, 256, v26 // (op 4621) least significant half
v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 4621) most significant half
buffer_load_ubyte v15, v65, s[76:79], 0 offen offset:12 // (op 543) Load value

s_barrier  // (op 6397) 
v_add_u32 v29, 8, v29 // (op 4883) 



s_waitcnt vmcnt(9)
ds_write_b8 v67, v31 offset:32768 // (op 201) Store local data 
s_barrier  // (op 6376) 



s_waitcnt vmcnt(6)
ds_write_b8 v66, v30 offset:33024 // (op 200) Store local data 
s_barrier  // (op 6362) 



ds_read_b64_tr_b8 v[32:33], v8 // (op 193) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v8 offset:512 // (op 193) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v8 offset:4096 // (op 193) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v8 offset:4608 // (op 193) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 6404) 


ds_read_u8 v20, v9 offset:32768 // (op 197) Load local data 



ds_read_b64_tr_b8 v[40:41], v8 offset:32 // (op 657) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v8 offset:544 // (op 657) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v8 offset:4128 // (op 657) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v8 offset:4640 // (op 657) Transpose load local data 


ds_read_u8 v28, v9 offset:32800 // (op 661) Load local data 
s_barrier  // (op 6343) 



ds_read_b128 v[48:51], v7 offset:8192 // (op 191) Load local data 
ds_read_b128 v[52:55], v7 offset:8256 // (op 191) Load local data 
s_waitcnt lgkmcnt(12)
s_barrier  // (op 6383) 



ds_read_u8 v24, v6 offset:33024 // (op 195) Load local data 



ds_read_b128 v[56:59], v7 offset:12288 // (op 908) Load local data 
ds_read_b128 v[60:63], v7 offset:12352 // (op 908) Load local data 



ds_read_u8 v25, v6 offset:33152 // (op 911) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 874) 
s_mov_b32 s28, s18 // (op 874) convert

s_and_b32 s29, s28, 4294967294 // (op 874) 

s_cmp_lt_u32 s1, s29 // (op 874) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoop_874 // (op 874) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoop_874 if true)
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoop_874:














s_barrier  // (op 6565) 
v_lshrrev_b32 v16, 3, v1 // (op 4557) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4557) 
v_and_b32 v22, -8, v1 // (op 4557) 
s_and_b32 s15, s18, 4294967294 // (op 4557) 
v_sub_u32 v23, v1, v22 // (op 4557) 

s_add_u32 s6, s0, s15 // (op 4557) 

v_lshlrev_b32 v22, 4, v23 // (op 4557) 

s_mul_i32 s15, s16, s52 // (op 4557) 
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


v_lshl_add_u32 v23, s15, 6, v16 // (op 4557) 


s_sub_u32 s7, s6, s8 // (op 4557) 


v_lshl_add_u32 v16, s7, 7, v22 // (op 4557) 


v_mov_b32 v27, 0 // (op 4557) convert
v_mov_b32 v26, v16 // (op 4557) convert

v_mul_lo_u32 v31, v23, s31 // (op 4557) most significant: low of low * high
v_mul_hi_u32 v16, v23, s30 // (op 4557) most significant: high of low * low
v_mul_lo_u32 v30, v23, s30 // (op 4557) least significant: low of low * low
v_add_u32 v31, v31, v16 // (op 4557) most significant: sum

v_add_co_u32 v30, s[18:19], v30, v26 // (op 4557) least significant half
v_addc_co_u32 v31, s[18:19], v31, v27, s[18:19] // (op 4557) most significant half



s_mul_i32 s19, s30, s35 // (op 4559) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 4559) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 4559) least significant
s_mul_hi_u32 s6, s30, s34 // (op 4559) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4559) most significant: sum
s_add_u32 s19, s19, s6 // (op 4559) most significant: sum


s_mov_b32 s68, s32 // (op 4559) reg expression
s_mov_b32 s69, s33 // (op 4559) reg expression
s_mov_b32 s70, s18 // (op 4559) reg expression
s_mov_b32 s71, 131072 // (op 4559) call()

v_and_b32 v16, -4, v1 // (op 4699) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4699) 
v_sub_u32 v29, v1, v16 // (op 4699) 

s_and_b32 s15, s18, 4294967294 // (op 4699) 
v_lshlrev_b32 v16, 4, v29 // (op 4699) 

s_add_u32 s6, s0, s15 // (op 4699) 

v_lshrrev_b32 v29, 2, v1 // (op 4699) 
s_mul_i32 s15, s16, s52 // (op 4699) 
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


v_lshl_add_u32 v22, s15, 6, v16 // (op 4699) 


s_sub_u32 s7, s6, s8 // (op 4699) 


v_mov_b32 v27, 0 // (op 4699) convert
v_mov_b32 v26, v22 // (op 4699) convert

v_lshl_add_u32 v16, s7, 7, v29 // (op 4699) 


v_mul_lo_u32 v23, v16, s37 // (op 4699) most significant: low of low * high
v_mul_hi_u32 v29, v16, s36 // (op 4699) most significant: high of low * low
v_mul_lo_u32 v22, v16, s36 // (op 4699) least significant: low of low * low
v_add_u32 v23, v23, v29 // (op 4699) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v26 // (op 4699) least significant half
v_addc_co_u32 v23, s[18:19], v23, v27, s[18:19] // (op 4699) most significant half



s_mul_i32 s19, s36, s3 // (op 4701) most significant: low of low * high
s_mul_i32 s15, s37, s2 // (op 4701) most significant: low of high * low
s_mul_i32 s18, s36, s2 // (op 4701) least significant
s_mul_hi_u32 s6, s36, s2 // (op 4701) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4701) most significant: sum
s_add_u32 s19, s19, s6 // (op 4701) most significant: sum


s_mov_b32 s72, s38 // (op 4701) reg expression
s_mov_b32 s73, s39 // (op 4701) reg expression
s_mov_b32 s74, s18 // (op 4701) reg expression
s_mov_b32 s75, 131072 // (op 4701) call()

v_lshrrev_b32 v16, 2, v1 // (op 4827) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4827) 
v_and_b32 v29, -4, v1 // (op 4827) 
s_and_b32 s15, s18, 4294967294 // (op 4827) 
v_sub_u32 v26, v1, v29 // (op 4827) 

s_add_u32 s6, s0, s15 // (op 4827) 

s_mul_i32 s15, s16, s52 // (op 4827) 
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


v_lshl_add_u32 v27, s15, 6, v16 // (op 4827) 


s_sub_u32 s7, s6, s8 // (op 4827) 


v_lshl_add_u32 v16, s7, 2, v26 // (op 4827) 


v_mul_lo_u32 v26, v27, s40 // (op 4827) 
v_add_u32 v26, v26, v16 // (op 4827) 


s_mul_i32 s19, s40, s35 // (op 4829) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 4829) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 4829) least significant
s_mul_hi_u32 s6, s40, s34 // (op 4829) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4829) most significant: sum
s_add_u32 s19, s19, s6 // (op 4829) most significant: sum


s_mov_b32 s76, s42 // (op 4829) reg expression
s_mov_b32 s77, s43 // (op 4829) reg expression
s_mov_b32 s78, s18 // (op 4829) reg expression
s_mov_b32 s79, 131072 // (op 4829) call()

v_and_b32 v16, -64, v1 // (op 4955) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4955) 
v_sub_u32 v27, v1, v16 // (op 4955) 

s_and_b32 s15, s18, 4294967294 // (op 4955) 
v_lshrrev_b32 v16, 6, v1 // (op 4955) 
s_add_u32 s6, s0, s15 // (op 4955) 

s_mul_i32 s15, s16, s52 // (op 4955) 
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

s_mul_i32 s8, s7, s18 // (op 4955) 


v_lshl_add_u32 v29, s15, 6, v27 // (op 4955) 


s_sub_u32 s7, s6, s8 // (op 4955) 


v_lshl_add_u32 v27, s7, 2, v16 // (op 4955) 


v_mul_lo_u32 v16, v27, s44 // (op 4955) 
v_add_u32 v16, v16, v29 // (op 4955) 


s_mul_i32 s19, s44, s49 // (op 4957) most significant: low of low * high
s_mul_i32 s15, s45, s48 // (op 4957) most significant: low of high * low
s_mul_i32 s18, s44, s48 // (op 4957) least significant
s_mul_hi_u32 s6, s44, s48 // (op 4957) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4957) most significant: sum
s_add_u32 s19, s19, s6 // (op 4957) most significant: sum


s_mov_b32 s80, s46 // (op 4957) reg expression
s_mov_b32 s81, s47 // (op 4957) reg expression
s_mov_b32 s82, s18 // (op 4957) reg expression
s_mov_b32 s83, 131072 // (op 4957) call()

v_and_b32 v27, -8, v1 // (op 5502) 
v_sub_u32 v29, v1, v27 // (op 5502) 

v_lshlrev_b32 v27, 4, v29 // (op 5502) 

v_lshrrev_b32 v29, 3, v1 // (op 5502) 
v_lshl_add_u32 v64, v29, 7, v27 // (op 5502) 


s_nop 0
v_readfirstlane_b32 s15, v64 // (op 5502) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -4, v1 // (op 5577) 
v_sub_u32 v29, v1, v27 // (op 5577) 

v_lshlrev_b32 v27, 4, v29 // (op 5577) 

v_lshrrev_b32 v29, 2, v1 // (op 5577) 
v_lshl_add_u32 v64, v29, 6, v27 // (op 5577) 


s_nop 0
v_readfirstlane_b32 s6, v64 // (op 5577) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -4, v1 // (op 5652) 
v_sub_u32 v29, v1, v27 // (op 5652) 

v_and_b32 v27, 4294967292, v1 // (op 5652) 
v_add_u32 v64, v27, v29 // (op 5652) 


v_and_b32 v27, -64, v1 // (op 5728) 
v_sub_u32 v29, v1, v27 // (op 5728) 

v_and_b32 v27, 4294967232, v1 // (op 5728) 
v_add_u32 v65, v27, v29 // (op 5728) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 880) 
s_and_b32 s7, s18, 4294967294 // (op 880) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 879) 
s_mov_b32 s8, s18 // (op 879) convert

s_cmp_lt_u32 s7, s8 // (op 879) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_879 // (op 879) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_879 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_879:
s_barrier  // (op 6340) 
s_add_u32 s7, s7, 1 // (op 881) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 4472) 
v_mov_b64 v[20:21], s[18:19] // (op 4472) convert


v_mov_b64 v[24:25], v[30:31] // (op 4472) 
v_mov_b64 v[28:29], v[24:25] // (op 4472) 
s_add_u32 s8, 8192, s15 // (op 4472) 
s_mov_b32 m0, s8 // (op 4472) reg expression
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 4472) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v20 // (op 4472) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 4472) most significant half

s_add_u32 m0, m0, 4096 // (op 4472) 
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 4472) Load value direct to lds




v_add_co_u32 v30, vcc, 128, v30 // (op 4547) least significant half
v_addc_co_u32 v31, vcc, 0, v31, vcc // (op 4547) most significant half
s_barrier  // (op 6359) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 4478) 
v_mov_b64 v[20:21], s[18:19] // (op 4478) convert


v_mov_b64 v[24:25], v[22:23] // (op 4478) 
v_mov_b64 v[28:29], v[24:25] // (op 4478) reg expression
s_mov_b32 m0, s6 // (op 4478) reg expression
buffer_load_dwordx4 v28, s[72:75], 0 offen offset: 0 lds // (op 4478) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v20 // (op 4478) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 4478) most significant half

s_add_u32 m0, m0, 4096 // (op 4478) 
buffer_load_dwordx4 v28, s[72:75], 0 offen offset: 0 lds // (op 4478) Load value direct to lds



s_lshl_b64 s[18:19], s[36:37], 7 // (op 4691) 
v_mov_b32 v15, s18 // (op 4691) 
v_mov_b32 v20, s19 // (op 4691) 
v_add_co_u32 v22, s[28:29], v22, v15 // (op 4691) least significant half
v_addc_co_u32 v23, s[28:29], v23, v20, s[28:29] // (op 4691) most significant half






v_mov_b32 v15, v26 // (op 477) 

buffer_load_ubyte v20, v15, s[76:79], 0 offen offset: 0 // (op 477) Load value

v_add_u32 v26, 4, v26 // (op 4819) 


s_barrier  // (op 6379) 
v_mov_b32 v15, v16 // (op 478) 



v_mov_b32 v21, v15 // (op 478) reg expression

buffer_load_ubyte v24, v21, s[80:83], 0 offen offset: 0 // (op 478) Load value


s_lshl_b32 s8, s44, 2 // (op 4947) 
v_add_u32 v16, v16, s8 // (op 4947) 

s_waitcnt vmcnt(1)
ds_write_b8 v64, v20 offset:33024 // (op 474) Store local data 

s_barrier  // (op 6346) 



s_barrier  // (op 6400) 
ds_read_b128 v[32:35], v7 offset:8192 // (op 465) Load local data 



ds_read_b128 v[36:39], v7 offset:8256 // (op 465) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v65, v24 offset:32768 // (op 475) Store local data 

s_barrier  // (op 6366) 



ds_read_b64_tr_b8 v[40:41], v8 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v8 offset:512 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v8 offset:4096 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v8 offset:4608 // (op 467) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 6387) 



ds_read_u8 v15, v6 offset:33024 // (op 469) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 6407) 


ds_read_u8 v20, v9 offset:32768 // (op 471) Load local data 



ds_read_b64_tr_b8 v[48:49], v8 offset:32 // (op 636) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v8 offset:544 // (op 636) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[40:47], a[0:3], v15, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 472) 
ds_read_b64_tr_b8 v[52:53], v8 offset:4128 // (op 636) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v8 offset:4640 // (op 636) Transpose load local data 


ds_read_u8 v21, v9 offset:32800 // (op 640) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:39], v[48:55], a[4:7], v15, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 641) 
ds_read_b128 v[56:59], v7 offset:12288 // (op 854) Load local data 


ds_read_b128 v[60:63], v7 offset:12352 // (op 854) Load local data 



ds_read_u8 v15, v6 offset:33152 // (op 857) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:63], v[40:47], a[8:11], v15, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 859) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:63], v[48:55], a[12:15], v15, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 890) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 879) 
s_mov_b32 s8, s18 // (op 879) convert

s_cmp_lt_u32 s7, s8 // (op 879) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_879 // (op 879) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_KLoopTail_879 if true)
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_KLoopTail_879:









s_barrier  // (op 6569) 




v_cmp_eq_f32 vcc, 0.00000, v18 // (op 238) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_238 // (op 238) Condition: False, jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_238
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

buffer_store_dwordx2 v[20:21], v0, s[64:67], 0 offen // (op 249) Store value



v_accvgpr_read v14, a4 // (op 1303) 
v_accvgpr_read v15, a5 // (op 1303) 
v_accvgpr_read v16, a6 // (op 1303) 
v_accvgpr_read v17, a7 // (op 1303) 
v_mul_f32 v20, v19, v14 // (op 1303) 
v_mul_f32 v21, v19, v15 // (op 1303) 
v_mul_f32 v22, v19, v16 // (op 1303) 
v_mul_f32 v23, v19, v17 // (op 1303) 

v_cvt_f16_f32 v14, v20 // (op 1305) 
v_cvt_f16_f32 v15, v21 // (op 1305) 
v_cvt_f16_f32 v16, v22 // (op 1305) 
v_cvt_f16_f32 v17, v23 // (op 1305) 


v_pack_b32_f16 v20, v14, v15 // (op 1306) 
v_pack_b32_f16 v21, v16, v17 // (op 1306) 

s_lshl_b32 s15, s24, 6 // (op 1306) 
v_add_u32 v0, v5, s15 // (op 1306) 


buffer_store_dwordx2 v[20:21], v0, s[64:67], 0 offen // (op 1306) Store value



v_accvgpr_read v14, a8 // (op 1329) 
v_accvgpr_read v15, a9 // (op 1329) 
v_accvgpr_read v16, a10 // (op 1329) 
v_accvgpr_read v17, a11 // (op 1329) 
v_mul_f32 v20, v19, v14 // (op 1329) 
v_mul_f32 v21, v19, v15 // (op 1329) 
v_mul_f32 v22, v19, v16 // (op 1329) 
v_mul_f32 v23, v19, v17 // (op 1329) 

v_cvt_f16_f32 v14, v20 // (op 1331) 
v_cvt_f16_f32 v15, v21 // (op 1331) 
v_cvt_f16_f32 v16, v22 // (op 1331) 
v_cvt_f16_f32 v17, v23 // (op 1331) 


v_pack_b32_f16 v20, v14, v15 // (op 1333) 
v_pack_b32_f16 v21, v16, v17 // (op 1333) 

v_add_u32 v0, 64, v5 // (op 1333) 

buffer_store_dwordx2 v[20:21], v0, s[64:67], 0 offen // (op 1333) Store value



v_accvgpr_read v14, a12 // (op 1339) 
v_accvgpr_read v15, a13 // (op 1339) 
v_accvgpr_read v16, a14 // (op 1339) 
v_accvgpr_read v17, a15 // (op 1339) 
v_mul_f32 v20, v19, v14 // (op 1339) 
v_mul_f32 v21, v19, v15 // (op 1339) 
v_mul_f32 v22, v19, v16 // (op 1339) 
v_mul_f32 v23, v19, v17 // (op 1339) 

v_cvt_f16_f32 v14, v20 // (op 1341) 
v_cvt_f16_f32 v15, v21 // (op 1341) 
v_cvt_f16_f32 v16, v22 // (op 1341) 
v_cvt_f16_f32 v17, v23 // (op 1341) 


v_pack_b32_f16 v20, v14, v15 // (op 1335) 
v_pack_b32_f16 v21, v16, v17 // (op 1335) 

s_lshl_b32 s15, s24, 6 // (op 1335) 
s_add_u32 s15, s15, 64 // (op 1335) 
v_add_u32 v0, v5, s15 // (op 1335) 


buffer_store_dwordx2 v[20:21], v0, s[64:67], 0 offen // (op 1335) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_238 // (op 238) Condition: Done, jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_238
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalFalse_0_beta_238:


v_add_u32 v0, 64, v4 // (op 1425) 

buffer_load_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 1425) Load value

v_accvgpr_read v20, a8 // (op 1416) 
v_accvgpr_read v21, a9 // (op 1416) 
v_accvgpr_read v22, a10 // (op 1416) 
v_accvgpr_read v23, a11 // (op 1416) 
v_mul_f32 v24, v19, v20 // (op 1416) 
v_mul_f32 v25, v19, v21 // (op 1416) 
v_mul_f32 v26, v19, v22 // (op 1416) 
v_mul_f32 v27, v19, v23 // (op 1416) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 1424) 
v_lshrrev_b32 v17, 16, v14 // (op 1424) 
v_cvt_f32_f16 v17, v17 // (op 1424) 
v_mul_f32 v20, v18, v16 // (op 1424) 
v_mul_f32 v21, v18, v17 // (op 1424) 
v_cvt_f32_f16 v16, v15 // (op 1424) 
v_lshrrev_b32 v17, 16, v15 // (op 1424) 
v_cvt_f32_f16 v17, v17 // (op 1424) 
v_mul_f32 v22, v18, v16 // (op 1424) 
v_mul_f32 v23, v18, v17 // (op 1424) 


v_add_f32 v14, v20, v24 // (op 1417) 
v_add_f32 v15, v21, v25 // (op 1417) 
v_add_f32 v16, v22, v26 // (op 1417) 
v_add_f32 v17, v23, v27 // (op 1417) 


v_cvt_f16_f32 v20, v14 // (op 1418) 
v_cvt_f16_f32 v21, v15 // (op 1418) 
v_cvt_f16_f32 v22, v16 // (op 1418) 
v_cvt_f16_f32 v23, v17 // (op 1418) 


v_pack_b32_f16 v14, v20, v21 // (op 1420) 
v_pack_b32_f16 v15, v22, v23 // (op 1420) 

v_add_u32 v0, 64, v5 // (op 1420) 

buffer_store_dwordx2 v[14:15], v0, s[64:67], 0 offen // (op 1420) Store value





s_lshl_b32 s15, s4, 6 // (op 1430) 
s_add_u32 s15, s15, 64 // (op 1430) 
v_add_u32 v0, v4, s15 // (op 1430) 


buffer_load_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 1430) Load value

v_accvgpr_read v20, a12 // (op 1432) 
v_accvgpr_read v21, a13 // (op 1432) 
v_accvgpr_read v22, a14 // (op 1432) 
v_accvgpr_read v23, a15 // (op 1432) 
v_mul_f32 v24, v19, v20 // (op 1432) 
v_mul_f32 v25, v19, v21 // (op 1432) 
v_mul_f32 v26, v19, v22 // (op 1432) 
v_mul_f32 v27, v19, v23 // (op 1432) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 1427) 
v_lshrrev_b32 v17, 16, v14 // (op 1427) 
v_cvt_f32_f16 v17, v17 // (op 1427) 
v_mul_f32 v20, v18, v16 // (op 1427) 
v_mul_f32 v21, v18, v17 // (op 1427) 
v_cvt_f32_f16 v16, v15 // (op 1427) 
v_lshrrev_b32 v17, 16, v15 // (op 1427) 
v_cvt_f32_f16 v17, v17 // (op 1427) 
v_mul_f32 v22, v18, v16 // (op 1427) 
v_mul_f32 v23, v18, v17 // (op 1427) 


v_add_f32 v14, v20, v24 // (op 1428) 
v_add_f32 v15, v21, v25 // (op 1428) 
v_add_f32 v16, v22, v26 // (op 1428) 
v_add_f32 v17, v23, v27 // (op 1428) 


v_cvt_f16_f32 v20, v14 // (op 1429) 
v_cvt_f16_f32 v21, v15 // (op 1429) 
v_cvt_f16_f32 v22, v16 // (op 1429) 
v_cvt_f16_f32 v23, v17 // (op 1429) 


v_pack_b32_f16 v14, v20, v21 // (op 1422) 
v_pack_b32_f16 v15, v22, v23 // (op 1422) 

s_lshl_b32 s15, s24, 6 // (op 1422) 
s_add_u32 s15, s15, 64 // (op 1422) 
v_add_u32 v0, v5, s15 // (op 1422) 


buffer_store_dwordx2 v[14:15], v0, s[64:67], 0 offen // (op 1422) Store value





v_mov_b32 v0, v4 // (op 263) reg expression

buffer_load_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 263) Load value

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

buffer_store_dwordx2 v[14:15], v0, s[64:67], 0 offen // (op 262) Store value





s_lshl_b32 s15, s4, 6 // (op 1385) 
v_add_u32 v0, v4, s15 // (op 1385) 


buffer_load_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 1385) Load value

v_accvgpr_read v20, a4 // (op 1381) 
v_accvgpr_read v21, a5 // (op 1381) 
v_accvgpr_read v22, a6 // (op 1381) 
v_accvgpr_read v23, a7 // (op 1381) 
v_mul_f32 v24, v19, v20 // (op 1381) 
v_mul_f32 v25, v19, v21 // (op 1381) 
v_mul_f32 v26, v19, v22 // (op 1381) 
v_mul_f32 v27, v19, v23 // (op 1381) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v14 // (op 1386) 
v_lshrrev_b32 v17, 16, v14 // (op 1386) 
v_cvt_f32_f16 v17, v17 // (op 1386) 
v_mul_f32 v20, v18, v16 // (op 1386) 
v_mul_f32 v21, v18, v17 // (op 1386) 
v_cvt_f32_f16 v16, v15 // (op 1386) 
v_lshrrev_b32 v17, 16, v15 // (op 1386) 
v_cvt_f32_f16 v17, v17 // (op 1386) 
v_mul_f32 v22, v18, v16 // (op 1386) 
v_mul_f32 v23, v18, v17 // (op 1386) 


v_add_f32 v14, v20, v24 // (op 1382) 
v_add_f32 v15, v21, v25 // (op 1382) 
v_add_f32 v16, v22, v26 // (op 1382) 
v_add_f32 v17, v23, v27 // (op 1382) 


v_cvt_f16_f32 v20, v14 // (op 1383) 
v_cvt_f16_f32 v21, v15 // (op 1383) 
v_cvt_f16_f32 v22, v16 // (op 1383) 
v_cvt_f16_f32 v23, v17 // (op 1383) 


v_pack_b32_f16 v14, v20, v21 // (op 1384) 
v_pack_b32_f16 v15, v22, v23 // (op 1384) 

s_lshl_b32 s15, s24, 6 // (op 1384) 
v_add_u32 v0, v5, s15 // (op 1384) 


buffer_store_dwordx2 v[14:15], v0, s[64:67], 0 offen // (op 1384) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ConditionalBottom_0_beta_238:
s_add_u32 s0, s0, s1 // (op 451) 


s_mul_i32 s15, s52, s16 // (op 449) 
s_cmp_gt_u32 s53, s15 // (op 449) 
s_mov_b32 s6, scc // (op 449) 
s_sub_u32 s7, s53, s15 // (op 449) 

s_cmp_eq_i32 s6, 0 // (op 449) 
s_cselect_b32 s8, 0, s7 // (op 449) 


s_cmp_lt_u32 s52, s8 // (op 449) 
s_mov_b32 s15, scc // (op 449) 
s_cmp_eq_i32 s15, 0 // (op 449) 
s_cselect_b32 s6, s8, s52 // (op 449) 


s_cmp_lt_u32 s0, s6 // (op 449) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_DPStreamTileLoop_449 // (op 449) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopTop_DPStreamTileLoop_449 if true)
RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_449:














































s_endpgm  // End of RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2
.LRR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_end:
.size RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2, .LRR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2_end-RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2
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
    DataType: B8
    DataTypeA: B8
    DataTypeB: B8
    DestDataType: h
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
            { type: int64, semantic: SizeSumScaleA },
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
  - .name: RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2
    .symbol: RR_GEMM_TT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x64x128_UR_2.kd
    .kernarg_segment_size: 192
    .group_segment_fixed_size: 33792
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 92
    .vgpr_count: 108
    .agpr_count: 16
    .max_flat_workgroup_size: 256
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
      - .name: Tensor_2_stride_0_11
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
      - .name: Tensor_7_stride_0_16
        .size: 8
        .offset: 152
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 160
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_17
        .size: 8
        .offset: 168
        .value_kind: by_value
      - .name: SCRATCH_ZeroedBeforeAndAfter
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

