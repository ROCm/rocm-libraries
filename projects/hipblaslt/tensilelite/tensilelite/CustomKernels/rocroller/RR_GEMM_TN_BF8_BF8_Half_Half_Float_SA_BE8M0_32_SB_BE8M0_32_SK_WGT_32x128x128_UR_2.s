// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2,@function
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 4363) 
v_and_b32 v3, 15, v1 // (op 4363) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 4363) 


v_and_b32 v0, 48, v1 // (op 4363) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 4363) 


v_lshrrev_b32 v0, 7, v1 // (op 4621) 
v_and_b32 v4, 15, v1 // (op 4621) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 4621) 


v_and_b32 v0, 48, v1 // (op 4621) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 4621) 


v_lshrrev_b32 v0, 7, v1 // (op 4866) 
v_and_b32 v5, 15, v1 // (op 4866) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 4866) 


v_and_b32 v0, 48, v1 // (op 4866) 
v_lshl_add_u32 v5, v6, 7, v0 // (op 4866) 


v_lshrrev_b32 v0, 7, v1 // (op 4978) 
v_and_b32 v6, 15, v1 // (op 4978) 
v_lshl_add_u32 v7, v0, 4, v6 // (op 4978) 


v_and_b32 v0, 48, v1 // (op 4978) 
v_lshl_add_u32 v6, v7, 7, v0 // (op 4978) 


v_lshrrev_b32 v0, 6, v1 // (op 4433) 
v_and_b32 v7, 1, v0 // (op 4433) 

v_and_b32 v0, 15, v1 // (op 4433) 
v_lshl_add_u32 v8, v7, 4, v0 // (op 4433) 


v_and_b32 v0, 48, v1 // (op 4433) 
v_lshl_add_u32 v7, v8, 7, v0 // (op 4433) 


v_lshrrev_b32 v0, 6, v1 // (op 4509) 
v_and_b32 v8, 1, v0 // (op 4509) 

v_and_b32 v0, 15, v1 // (op 4509) 
v_lshl_add_u32 v9, v8, 4, v0 // (op 4509) 


v_and_b32 v0, 63, v1 // (op 4509) 
v_lshrrev_b32 v8, 4, v0 // (op 4509) 

v_lshl_add_u32 v0, v9, 2, v8 // (op 4509) 


v_lshrrev_b32 v8, 6, v1 // (op 4691) 
v_and_b32 v9, 1, v8 // (op 4691) 

v_and_b32 v8, 15, v1 // (op 4691) 
v_lshl_add_u32 v10, v9, 4, v8 // (op 4691) 


v_and_b32 v8, 48, v1 // (op 4691) 
v_lshl_add_u32 v9, v10, 7, v8 // (op 4691) 


v_lshrrev_b32 v8, 6, v1 // (op 4767) 
v_and_b32 v10, 1, v8 // (op 4767) 

v_and_b32 v8, 15, v1 // (op 4767) 
v_lshl_add_u32 v11, v10, 4, v8 // (op 4767) 


v_and_b32 v8, 63, v1 // (op 4767) 
v_lshrrev_b32 v10, 4, v8 // (op 4767) 

v_lshl_add_u32 v8, v11, 2, v10 // (op 4767) 


v_lshrrev_b32 v10, 6, v1 // (op 4910) 
v_and_b32 v11, 1, v10 // (op 4910) 

v_and_b32 v10, 15, v1 // (op 4910) 
v_lshl_add_u32 v12, v11, 4, v10 // (op 4910) 


v_and_b32 v10, 48, v1 // (op 4910) 
v_lshl_add_u32 v11, v12, 7, v10 // (op 4910) 


v_lshrrev_b32 v10, 6, v1 // (op 4939) 
v_and_b32 v12, 1, v10 // (op 4939) 

v_and_b32 v10, 15, v1 // (op 4939) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 4939) 


v_and_b32 v10, 63, v1 // (op 4939) 
v_lshrrev_b32 v12, 4, v10 // (op 4939) 

v_lshl_add_u32 v10, v13, 2, v12 // (op 4939) 


v_lshrrev_b32 v12, 6, v1 // (op 5022) 
v_and_b32 v13, 1, v12 // (op 5022) 

v_and_b32 v12, 15, v1 // (op 5022) 
v_lshl_add_u32 v14, v13, 4, v12 // (op 5022) 


v_and_b32 v12, 48, v1 // (op 5022) 
v_lshl_add_u32 v13, v14, 7, v12 // (op 5022) 


v_lshrrev_b32 v12, 6, v1 // (op 5051) 
v_and_b32 v14, 1, v12 // (op 5051) 

v_and_b32 v12, 15, v1 // (op 5051) 
v_lshl_add_u32 v15, v14, 4, v12 // (op 5051) 


v_and_b32 v12, 63, v1 // (op 5051) 
v_lshrrev_b32 v14, 4, v12 // (op 5051) 

v_lshl_add_u32 v12, v15, 2, v14 // (op 5051) 


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
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_278 // (op 278) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_278 if false)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_SKStreamTileLoop_278:
v_lshrrev_b32 v16, 7, v1 // (op 4284) 
s_mul_i32 s52, s16, s8 // (op 4284) 
s_add_u32 s52, s52, s0 // (op 4284) 
v_and_b32 v17, 63, v1 // (op 4284) 
s_mul_hi_u32 s53, s52, s13 // (op 4284) 
v_lshrrev_b32 v18, 4, v17 // (op 4284) 
s_sub_u32 s54, s52, s53 // (op 4284) 
v_lshlrev_b32 v19, 2, v18 // (op 4284) 

v_lshl_add_u32 v18, v16, 4, v19 // (op 4284) 


s_lshr_b32 s55, s54, 1 // (op 4284) 

s_add_u32 s54, s55, s53 // (op 4284) 


v_lshrrev_b32 v16, 2, v17 // (op 4284) 

s_lshr_b32 s53, s54, s11 // (op 4284) 

v_and_b32 v17, 3, v16 // (op 4284) 

s_and_b32 s54, s11, 2147483648 // (op 4284) 
v_and_b32 v16, 3, v1 // (op 4284) 
s_cmp_eq_i32 s54, 0 // (op 4284) 
s_mov_b32 s55, scc // (op 4284) 

v_lshl_add_u32 v19, v17, 2, v16 // (op 4284) 


s_cmp_eq_i32 s55, 0 // (op 4284) 
s_cselect_b32 s54, s52, s53 // (op 4284) 



v_lshrrev_b32 v16, 6, v1 // (op 4284) 
s_mul_hi_u32 s52, s54, s12 // (op 4284) 
v_and_b32 v17, 1, v16 // (op 4284) 

s_sub_u32 s53, s54, s52 // (op 4284) 
v_lshl_add_u32 v16, v17, 4, v19 // (op 4284) 


s_lshr_b32 s55, s53, 1 // (op 4284) 

s_add_u32 s53, s55, s52 // (op 4284) 


s_lshr_b32 s52, s53, s10 // (op 4284) 

s_and_b32 s53, s10, 2147483648 // (op 4284) 
s_cmp_eq_i32 s53, 0 // (op 4284) 
s_mov_b32 s55, scc // (op 4284) 

s_cmp_eq_i32 s55, 0 // (op 4284) 
s_cselect_b32 s53, s54, s52 // (op 4284) 


s_mul_i32 s52, s53, s14 // (op 4284) 
v_lshl_add_u32 v17, s53, 5, v18 // (op 4284) 


s_sub_u32 s53, s54, s52 // (op 4284) 


v_lshl_add_u32 v18, s53, 7, v16 // (op 4284) 


v_mul_lo_u32 v16, v18, s4 // (op 4284) 

v_add_lshl_u32 v18, v17, v16, 1 // (op 4284) 


s_mul_i32 s53, s4, s23 // (op 4286) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 4286) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 4286) least significant
s_mul_hi_u32 s55, s4, s22 // (op 4286) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4286) most significant: sum
s_add_u32 s53, s53, s55 // (op 4286) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 4286) 

s_mov_b32 s56, s20 // (op 4286) reg expression
s_mov_b32 s57, s21 // (op 4286) reg expression
s_mov_b32 s58, s54 // (op 4286) reg expression
s_mov_b32 s59, 131072 // (op 4286) call()

v_lshrrev_b32 v16, 7, v1 // (op 4375) 
v_and_b32 v17, 15, v1 // (op 4375) 
v_lshl_add_u32 v19, v16, 4, v17 // (op 4375) 


v_and_b32 v16, 48, v1 // (op 4375) 
v_lshl_add_u32 v17, v19, 7, v16 // (op 4375) 


v_lshrrev_b32 v16, 6, v1 // (op 4450) 
v_and_b32 v19, 1, v16 // (op 4450) 

v_and_b32 v16, 15, v1 // (op 4450) 
v_lshl_add_u32 v20, v19, 4, v16 // (op 4450) 


v_and_b32 v16, 48, v1 // (op 4450) 
v_lshl_add_u32 v19, v20, 7, v16 // (op 4450) 


v_lshrrev_b32 v16, 6, v1 // (op 4526) 
v_and_b32 v20, 1, v16 // (op 4526) 

v_and_b32 v16, 15, v1 // (op 4526) 
v_lshl_add_u32 v21, v20, 4, v16 // (op 4526) 


v_and_b32 v16, 63, v1 // (op 4526) 
v_lshrrev_b32 v20, 4, v16 // (op 4526) 

v_lshl_add_u32 v16, v21, 2, v20 // (op 4526) 


v_lshrrev_b32 v20, 7, v1 // (op 4544) 
s_mul_i32 s52, s16, s8 // (op 4544) 
s_add_u32 s52, s52, s0 // (op 4544) 
v_and_b32 v21, 63, v1 // (op 4544) 
s_mul_hi_u32 s53, s52, s13 // (op 4544) 
v_lshrrev_b32 v22, 4, v21 // (op 4544) 
s_sub_u32 s54, s52, s53 // (op 4544) 
v_lshlrev_b32 v23, 2, v22 // (op 4544) 

s_lshr_b32 s55, s54, 1 // (op 4544) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 4544) 


s_add_u32 s54, s55, s53 // (op 4544) 


v_lshrrev_b32 v20, 2, v21 // (op 4544) 

s_lshr_b32 s53, s54, s11 // (op 4544) 

v_and_b32 v21, 3, v20 // (op 4544) 

s_and_b32 s54, s11, 2147483648 // (op 4544) 
v_and_b32 v20, 3, v1 // (op 4544) 
s_cmp_eq_i32 s54, 0 // (op 4544) 
s_mov_b32 s55, scc // (op 4544) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 4544) 


s_cmp_eq_i32 s55, 0 // (op 4544) 
s_cselect_b32 s54, s52, s53 // (op 4544) 



v_lshrrev_b32 v20, 6, v1 // (op 4544) 
s_mul_hi_u32 s55, s54, s12 // (op 4544) 
v_and_b32 v21, 1, v20 // (op 4544) 

s_sub_u32 s52, s54, s55 // (op 4544) 
v_lshl_add_u32 v20, v21, 4, v23 // (op 4544) 


s_lshr_b32 s53, s52, 1 // (op 4544) 

s_add_u32 s52, s53, s55 // (op 4544) 


s_lshr_b32 s53, s52, s10 // (op 4544) 

s_and_b32 s52, s10, 2147483648 // (op 4544) 
s_cmp_eq_i32 s52, 0 // (op 4544) 
s_mov_b32 s55, scc // (op 4544) 

s_cmp_eq_i32 s55, 0 // (op 4544) 
s_cselect_b32 s52, s54, s53 // (op 4544) 


s_mul_i32 s53, s52, s14 // (op 4544) 
v_lshl_add_u32 v21, s52, 5, v22 // (op 4544) 


s_sub_u32 s52, s54, s53 // (op 4544) 


v_lshl_add_u32 v22, s52, 7, v20 // (op 4544) 


v_mul_lo_u32 v20, v22, s24 // (op 4544) 

v_add_lshl_u32 v22, v21, v20, 1 // (op 4544) 


s_mul_i32 s53, s24, s23 // (op 4546) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 4546) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 4546) least significant
s_mul_hi_u32 s55, s24, s22 // (op 4546) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4546) most significant: sum
s_add_u32 s53, s53, s55 // (op 4546) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 4546) 

s_mov_b32 s60, s26 // (op 4546) reg expression
s_mov_b32 s61, s27 // (op 4546) reg expression
s_mov_b32 s62, s54 // (op 4546) reg expression
s_mov_b32 s63, 131072 // (op 4546) call()

v_lshrrev_b32 v20, 5, v1 // (op 4802) 
v_lshlrev_b32 v21, 2, v20 // (op 4802) 

v_lshl_add_u32 v20, s16, 5, v21 // (op 4802) 

v_and_b32 v21, -32, v1 // (op 4802) 
v_sub_u32 v23, v1, v21 // (op 4802) 

v_lshl_add_u32 v21, v20, 7, v23 // (op 4802) 


v_lshlrev_b32 v20, 2, v21 // (op 4802) 

s_lshl_b32 s52, s15, 14 // (op 4804) 
s_mov_b32 s64, s28 // (op 4804) reg expression
s_mov_b32 s65, s29 // (op 4804) reg expression
s_mov_b32 s66, s52 // (op 4804) reg expression
s_mov_b32 s67, 131072 // (op 4804) call()

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
v_accvgpr_write a4, 0.00000 // (op 1425) call()
v_accvgpr_write a5, 0.00000 // (op 1425) call()
v_accvgpr_write a6, 0.00000 // (op 1425) call()
v_accvgpr_write a7, 0.00000 // (op 1425) call()
v_accvgpr_write a8, 0.00000 // (op 1581) call()
v_accvgpr_write a9, 0.00000 // (op 1581) call()
v_accvgpr_write a10, 0.00000 // (op 1581) call()
v_accvgpr_write a11, 0.00000 // (op 1581) call()
v_accvgpr_write a12, 0.00000 // (op 1737) call()
v_accvgpr_write a13, 0.00000 // (op 1737) call()
v_accvgpr_write a14, 0.00000 // (op 1737) call()
v_accvgpr_write a15, 0.00000 // (op 1737) call()
v_lshrrev_b32 v21, 3, v1 // (op 3856) 
s_mul_i32 s52, s16, s8 // (op 3856) 
s_add_u32 s52, s52, s0 // (op 3856) 
v_and_b32 v23, -8, v1 // (op 3856) 
s_mul_hi_u32 s53, s52, s13 // (op 3856) 
v_sub_u32 v24, v1, v23 // (op 3856) 

s_sub_u32 s54, s52, s53 // (op 3856) 
v_lshlrev_b32 v23, 4, v24 // (op 3856) 

s_lshr_b32 s55, s54, 1 // (op 3856) 

s_add_u32 s54, s55, s53 // (op 3856) 


s_lshr_b32 s53, s54, s11 // (op 3856) 

s_and_b32 s54, s11, 2147483648 // (op 3856) 
s_cmp_eq_i32 s54, 0 // (op 3856) 
s_mov_b32 s55, scc // (op 3856) 

s_cmp_eq_i32 s55, 0 // (op 3856) 
s_cselect_b32 s54, s52, s53 // (op 3856) 


s_mul_hi_u32 s53, s54, s12 // (op 3856) 
s_sub_u32 s55, s54, s53 // (op 3856) 
s_lshr_b32 s68, s55, 1 // (op 3856) 

s_add_u32 s55, s68, s53 // (op 3856) 


s_lshr_b32 s53, s55, s10 // (op 3856) 

s_and_b32 s55, s10, 2147483648 // (op 3856) 
s_cmp_eq_i32 s55, 0 // (op 3856) 
s_mov_b32 s68, scc // (op 3856) 

s_cmp_eq_i32 s68, 0 // (op 3856) 
s_cselect_b32 s55, s54, s53 // (op 3856) 


v_lshl_add_u32 v24, s55, 5, v21 // (op 3856) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 3856) 
s_mul_i32 s53, s54, s68 // (op 3856) 


s_sub_u32 s54, s52, s53 // (op 3856) 


v_lshl_add_u32 v21, s54, 7, v23 // (op 3856) 


v_mov_b32 v27, 0 // (op 3856) convert
v_mov_b32 v26, v21 // (op 3856) convert

v_mul_lo_u32 v29, v24, s31 // (op 3856) most significant: low of low * high
v_mul_hi_u32 v21, v24, s30 // (op 3856) most significant: high of low * low
v_mul_lo_u32 v28, v24, s30 // (op 3856) least significant: low of low * low
v_add_u32 v29, v29, v21 // (op 3856) most significant: sum

v_add_co_u32 v28, s[52:53], v28, v26 // (op 3856) least significant half
v_addc_co_u32 v29, s[52:53], v29, v27, s[52:53] // (op 3856) most significant half



s_mul_i32 s53, s30, s35 // (op 3858) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 3858) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 3858) least significant
s_mul_hi_u32 s55, s30, s34 // (op 3858) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3858) most significant: sum
s_add_u32 s53, s53, s55 // (op 3858) most significant: sum


s_mov_b32 s68, s32 // (op 3858) reg expression
s_mov_b32 s69, s33 // (op 3858) reg expression
s_mov_b32 s70, s52 // (op 3858) reg expression
s_mov_b32 s71, 131072 // (op 3858) call()

v_lshrrev_b32 v21, 3, v1 // (op 3875) 
s_mul_i32 s52, s16, s8 // (op 3875) 
s_add_u32 s52, s52, s0 // (op 3875) 
v_and_b32 v23, -8, v1 // (op 3875) 
s_mul_hi_u32 s53, s52, s13 // (op 3875) 
v_sub_u32 v24, v1, v23 // (op 3875) 

s_sub_u32 s54, s52, s53 // (op 3875) 
v_lshlrev_b32 v23, 4, v24 // (op 3875) 

s_lshr_b32 s55, s54, 1 // (op 3875) 

s_add_u32 s54, s55, s53 // (op 3875) 


s_lshr_b32 s53, s54, s11 // (op 3875) 

s_and_b32 s54, s11, 2147483648 // (op 3875) 
s_cmp_eq_i32 s54, 0 // (op 3875) 
s_mov_b32 s55, scc // (op 3875) 

s_cmp_eq_i32 s55, 0 // (op 3875) 
s_cselect_b32 s54, s52, s53 // (op 3875) 


s_mul_hi_u32 s53, s54, s12 // (op 3875) 
s_sub_u32 s55, s54, s53 // (op 3875) 
s_lshr_b32 s72, s55, 1 // (op 3875) 

s_add_u32 s55, s72, s53 // (op 3875) 


s_lshr_b32 s53, s55, s10 // (op 3875) 

s_and_b32 s55, s10, 2147483648 // (op 3875) 
s_cmp_eq_i32 s55, 0 // (op 3875) 
s_mov_b32 s72, scc // (op 3875) 

s_cmp_eq_i32 s72, 0 // (op 3875) 
s_cselect_b32 s55, s54, s53 // (op 3875) 


v_lshl_add_u32 v24, s55, 5, v21 // (op 3875) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 3875) 
s_mul_i32 s53, s54, s72 // (op 3875) 


s_sub_u32 s54, s52, s53 // (op 3875) 


v_lshl_add_u32 v21, s54, 7, v23 // (op 3875) 


v_mov_b32 v27, 0 // (op 3875) convert
v_mov_b32 v26, v21 // (op 3875) convert

v_mul_lo_u32 v31, v24, s31 // (op 3875) most significant: low of low * high
v_mul_hi_u32 v21, v24, s30 // (op 3875) most significant: high of low * low
v_mul_lo_u32 v30, v24, s30 // (op 3875) least significant: low of low * low
v_add_u32 v31, v31, v21 // (op 3875) most significant: sum

v_add_co_u32 v30, s[52:53], v30, v26 // (op 3875) least significant half
v_addc_co_u32 v31, s[52:53], v31, v27, s[52:53] // (op 3875) most significant half



s_mul_i32 s53, s30, s35 // (op 3877) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 3877) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 3877) least significant
s_mul_hi_u32 s55, s30, s34 // (op 3877) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3877) most significant: sum
s_add_u32 s53, s53, s55 // (op 3877) most significant: sum


s_mov_b32 s68, s32 // (op 3877) reg expression
s_mov_b32 s69, s33 // (op 3877) reg expression
s_mov_b32 s70, s52 // (op 3877) reg expression
s_mov_b32 s71, 131072 // (op 3877) call()

v_lshrrev_b32 v21, 3, v1 // (op 3984) 
s_mul_i32 s52, s16, s8 // (op 3984) 
s_add_u32 s52, s52, s0 // (op 3984) 
v_and_b32 v23, -8, v1 // (op 3984) 
s_mul_hi_u32 s53, s52, s13 // (op 3984) 
v_sub_u32 v24, v1, v23 // (op 3984) 

s_sub_u32 s54, s52, s53 // (op 3984) 
v_lshlrev_b32 v23, 4, v24 // (op 3984) 

s_lshr_b32 s55, s54, 1 // (op 3984) 

s_add_u32 s54, s55, s53 // (op 3984) 


s_lshr_b32 s53, s54, s11 // (op 3984) 

s_and_b32 s54, s11, 2147483648 // (op 3984) 
s_cmp_eq_i32 s54, 0 // (op 3984) 
s_mov_b32 s55, scc // (op 3984) 

s_cmp_eq_i32 s55, 0 // (op 3984) 
s_cselect_b32 s54, s52, s53 // (op 3984) 


s_mul_hi_u32 s53, s54, s12 // (op 3984) 
s_sub_u32 s55, s54, s53 // (op 3984) 
s_lshr_b32 s72, s55, 1 // (op 3984) 

s_add_u32 s55, s72, s53 // (op 3984) 


s_lshr_b32 s53, s55, s10 // (op 3984) 

s_and_b32 s55, s10, 2147483648 // (op 3984) 
s_cmp_eq_i32 s55, 0 // (op 3984) 
s_mov_b32 s72, scc // (op 3984) 

s_cmp_eq_i32 s72, 0 // (op 3984) 
s_cselect_b32 s55, s54, s53 // (op 3984) 


s_mul_i32 s53, s55, s14 // (op 3984) 

s_sub_u32 s55, s54, s53 // (op 3984) 

v_lshl_add_u32 v24, s55, 7, v21 // (op 3984) 


v_mul_lo_u32 v27, v24, s37 // (op 3984) most significant: low of low * high
v_mul_hi_u32 v21, v24, s36 // (op 3984) most significant: high of low * low
v_mul_lo_u32 v26, v24, s36 // (op 3984) least significant: low of low * low
v_add_u32 v27, v27, v21 // (op 3984) most significant: sum


s_ashr_i64 s[72:73], s[2:3], 7 // (op 3984) 
s_mul_i32 s53, s54, s72 // (op 3984) 


s_sub_u32 s54, s52, s53 // (op 3984) 


v_lshl_add_u32 v21, s54, 7, v23 // (op 3984) 


v_mov_b32 v25, 0 // (op 3984) convert
v_mov_b32 v24, v21 // (op 3984) convert

v_add_co_u32 v32, s[52:53], v24, v26 // (op 3984) least significant half
v_addc_co_u32 v33, s[52:53], v25, v27, s[52:53] // (op 3984) most significant half



s_mul_i32 s53, s36, s23 // (op 3986) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 3986) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 3986) least significant
s_mul_hi_u32 s55, s36, s22 // (op 3986) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3986) most significant: sum
s_add_u32 s53, s53, s55 // (op 3986) most significant: sum


s_mov_b32 s72, s38 // (op 3986) reg expression
s_mov_b32 s73, s39 // (op 3986) reg expression
s_mov_b32 s74, s52 // (op 3986) reg expression
s_mov_b32 s75, 131072 // (op 3986) call()

v_lshrrev_b32 v21, 3, v1 // (op 4003) 
s_mul_i32 s52, s16, s8 // (op 4003) 
s_add_u32 s52, s52, s0 // (op 4003) 
v_and_b32 v23, -8, v1 // (op 4003) 
s_mul_hi_u32 s53, s52, s13 // (op 4003) 
v_sub_u32 v24, v1, v23 // (op 4003) 

s_sub_u32 s54, s52, s53 // (op 4003) 
v_lshlrev_b32 v23, 4, v24 // (op 4003) 

s_lshr_b32 s55, s54, 1 // (op 4003) 

s_add_u32 s54, s55, s53 // (op 4003) 


s_lshr_b32 s53, s54, s11 // (op 4003) 

s_and_b32 s54, s11, 2147483648 // (op 4003) 
s_cmp_eq_i32 s54, 0 // (op 4003) 
s_mov_b32 s55, scc // (op 4003) 

s_cmp_eq_i32 s55, 0 // (op 4003) 
s_cselect_b32 s54, s52, s53 // (op 4003) 


s_mul_hi_u32 s53, s54, s12 // (op 4003) 
s_sub_u32 s55, s54, s53 // (op 4003) 
s_lshr_b32 s76, s55, 1 // (op 4003) 

s_add_u32 s55, s76, s53 // (op 4003) 


s_lshr_b32 s53, s55, s10 // (op 4003) 

s_and_b32 s55, s10, 2147483648 // (op 4003) 
s_cmp_eq_i32 s55, 0 // (op 4003) 
s_mov_b32 s76, scc // (op 4003) 

s_cmp_eq_i32 s76, 0 // (op 4003) 
s_cselect_b32 s55, s54, s53 // (op 4003) 


s_mul_i32 s53, s55, s14 // (op 4003) 

s_sub_u32 s55, s54, s53 // (op 4003) 

v_lshl_add_u32 v24, s55, 7, v21 // (op 4003) 


v_mul_lo_u32 v27, v24, s37 // (op 4003) most significant: low of low * high
v_mul_hi_u32 v21, v24, s36 // (op 4003) most significant: high of low * low
v_mul_lo_u32 v26, v24, s36 // (op 4003) least significant: low of low * low
v_add_u32 v27, v27, v21 // (op 4003) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 4003) 
s_mul_i32 s53, s54, s76 // (op 4003) 


s_sub_u32 s54, s52, s53 // (op 4003) 


v_lshl_add_u32 v21, s54, 7, v23 // (op 4003) 


v_mov_b32 v25, 0 // (op 4003) convert
v_mov_b32 v24, v21 // (op 4003) convert

v_add_co_u32 v34, s[52:53], v24, v26 // (op 4003) least significant half
v_addc_co_u32 v35, s[52:53], v25, v27, s[52:53] // (op 4003) most significant half



s_mul_i32 s53, s36, s23 // (op 4005) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 4005) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 4005) least significant
s_mul_hi_u32 s55, s36, s22 // (op 4005) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4005) most significant: sum
s_add_u32 s53, s53, s55 // (op 4005) most significant: sum


s_mov_b32 s72, s38 // (op 4005) reg expression
s_mov_b32 s73, s39 // (op 4005) reg expression
s_mov_b32 s74, s52 // (op 4005) reg expression
s_mov_b32 s75, 131072 // (op 4005) call()

v_lshrrev_b32 v21, 7, v1 // (op 4112) 
s_mul_i32 s52, s16, s8 // (op 4112) 
s_add_u32 s52, s52, s0 // (op 4112) 
v_and_b32 v23, 15, v1 // (op 4112) 
s_mul_hi_u32 s53, s52, s13 // (op 4112) 
v_lshl_add_u32 v24, v21, 4, v23 // (op 4112) 


s_sub_u32 s54, s52, s53 // (op 4112) 
v_and_b32 v21, 63, v1 // (op 4112) 
s_lshr_b32 s55, s54, 1 // (op 4112) 

v_lshrrev_b32 v23, 4, v21 // (op 4112) 

s_add_u32 s54, s55, s53 // (op 4112) 


s_lshr_b32 s53, s54, s11 // (op 4112) 

s_and_b32 s54, s11, 2147483648 // (op 4112) 
s_cmp_eq_i32 s54, 0 // (op 4112) 
s_mov_b32 s55, scc // (op 4112) 

s_cmp_eq_i32 s55, 0 // (op 4112) 
s_cselect_b32 s54, s52, s53 // (op 4112) 


s_mul_hi_u32 s53, s54, s12 // (op 4112) 
s_sub_u32 s55, s54, s53 // (op 4112) 
s_lshr_b32 s76, s55, 1 // (op 4112) 

s_add_u32 s55, s76, s53 // (op 4112) 


s_lshr_b32 s53, s55, s10 // (op 4112) 

s_and_b32 s55, s10, 2147483648 // (op 4112) 
s_cmp_eq_i32 s55, 0 // (op 4112) 
s_mov_b32 s76, scc // (op 4112) 

s_cmp_eq_i32 s76, 0 // (op 4112) 
s_cselect_b32 s55, s54, s53 // (op 4112) 


v_lshl_add_u32 v21, s55, 5, v24 // (op 4112) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 4112) 
s_mul_i32 s53, s54, s76 // (op 4112) 


s_sub_u32 s54, s52, s53 // (op 4112) 


v_lshl_add_u32 v24, s54, 2, v23 // (op 4112) 


v_mul_lo_u32 v23, v21, s40 // (op 4112) 
v_add_u32 v23, v23, v24 // (op 4112) 


s_mul_i32 s53, s40, s35 // (op 4114) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 4114) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 4114) least significant
s_mul_hi_u32 s55, s40, s34 // (op 4114) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4114) most significant: sum
s_add_u32 s53, s53, s55 // (op 4114) most significant: sum


s_mov_b32 s76, s42 // (op 4114) reg expression
s_mov_b32 s77, s43 // (op 4114) reg expression
s_mov_b32 s78, s52 // (op 4114) reg expression
s_mov_b32 s79, 131072 // (op 4114) call()

v_lshrrev_b32 v21, 7, v1 // (op 4131) 
s_mul_i32 s52, s16, s8 // (op 4131) 
s_add_u32 s52, s52, s0 // (op 4131) 
v_and_b32 v24, 15, v1 // (op 4131) 
s_mul_hi_u32 s53, s52, s13 // (op 4131) 
v_lshl_add_u32 v25, v21, 4, v24 // (op 4131) 


s_sub_u32 s54, s52, s53 // (op 4131) 
v_and_b32 v21, 63, v1 // (op 4131) 
s_lshr_b32 s55, s54, 1 // (op 4131) 

v_lshrrev_b32 v24, 4, v21 // (op 4131) 

s_add_u32 s54, s55, s53 // (op 4131) 


s_lshr_b32 s53, s54, s11 // (op 4131) 

s_and_b32 s54, s11, 2147483648 // (op 4131) 
s_cmp_eq_i32 s54, 0 // (op 4131) 
s_mov_b32 s55, scc // (op 4131) 

s_cmp_eq_i32 s55, 0 // (op 4131) 
s_cselect_b32 s54, s52, s53 // (op 4131) 


s_mul_hi_u32 s53, s54, s12 // (op 4131) 
s_sub_u32 s55, s54, s53 // (op 4131) 
s_lshr_b32 s80, s55, 1 // (op 4131) 

s_add_u32 s55, s80, s53 // (op 4131) 


s_lshr_b32 s53, s55, s10 // (op 4131) 

s_and_b32 s55, s10, 2147483648 // (op 4131) 
s_cmp_eq_i32 s55, 0 // (op 4131) 
s_mov_b32 s80, scc // (op 4131) 

s_cmp_eq_i32 s80, 0 // (op 4131) 
s_cselect_b32 s55, s54, s53 // (op 4131) 


v_lshl_add_u32 v21, s55, 5, v25 // (op 4131) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 4131) 
s_mul_i32 s53, s54, s80 // (op 4131) 


s_sub_u32 s54, s52, s53 // (op 4131) 


v_lshl_add_u32 v25, s54, 2, v24 // (op 4131) 


v_mul_lo_u32 v24, v21, s40 // (op 4131) 
v_add_u32 v24, v24, v25 // (op 4131) 


s_mul_i32 s53, s40, s35 // (op 4133) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 4133) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 4133) least significant
s_mul_hi_u32 s55, s40, s34 // (op 4133) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4133) most significant: sum
s_add_u32 s53, s53, s55 // (op 4133) most significant: sum


s_mov_b32 s76, s42 // (op 4133) reg expression
s_mov_b32 s77, s43 // (op 4133) reg expression
s_mov_b32 s78, s52 // (op 4133) reg expression
s_mov_b32 s79, 131072 // (op 4133) call()

v_lshrrev_b32 v21, 2, v1 // (op 4240) 
s_mul_i32 s52, s16, s8 // (op 4240) 
s_add_u32 s52, s52, s0 // (op 4240) 
v_lshlrev_b32 v25, 1, v21 // (op 4240) 

s_mul_hi_u32 s53, s52, s13 // (op 4240) 
v_and_b32 v21, -4, v1 // (op 4240) 
s_sub_u32 s54, s52, s53 // (op 4240) 
v_sub_u32 v26, v1, v21 // (op 4240) 

s_lshr_b32 s55, s54, 1 // (op 4240) 

s_add_u32 s54, s55, s53 // (op 4240) 


s_lshr_b32 s53, s54, s11 // (op 4240) 

s_and_b32 s54, s11, 2147483648 // (op 4240) 
s_cmp_eq_i32 s54, 0 // (op 4240) 
s_mov_b32 s55, scc // (op 4240) 

s_cmp_eq_i32 s55, 0 // (op 4240) 
s_cselect_b32 s54, s52, s53 // (op 4240) 


s_mul_hi_u32 s53, s54, s12 // (op 4240) 
s_sub_u32 s55, s54, s53 // (op 4240) 
s_lshr_b32 s80, s55, 1 // (op 4240) 

s_add_u32 s55, s80, s53 // (op 4240) 


s_lshr_b32 s53, s55, s10 // (op 4240) 

s_and_b32 s55, s10, 2147483648 // (op 4240) 
s_cmp_eq_i32 s55, 0 // (op 4240) 
s_mov_b32 s80, scc // (op 4240) 

s_cmp_eq_i32 s80, 0 // (op 4240) 
s_cselect_b32 s55, s54, s53 // (op 4240) 


s_mul_i32 s53, s55, s14 // (op 4240) 

s_sub_u32 s55, s54, s53 // (op 4240) 

v_lshl_add_u32 v21, s55, 7, v25 // (op 4240) 


v_mul_lo_u32 v25, v21, s44 // (op 4240) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 4240) 
s_mul_i32 s53, s54, s80 // (op 4240) 


s_sub_u32 s54, s52, s53 // (op 4240) 


v_lshl_add_u32 v21, s54, 2, v26 // (op 4240) 


v_add_u32 v26, v21, v25 // (op 4240) 


s_mul_i32 s53, s44, s23 // (op 4242) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 4242) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 4242) least significant
s_mul_hi_u32 s55, s44, s22 // (op 4242) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4242) most significant: sum
s_add_u32 s53, s53, s55 // (op 4242) most significant: sum


s_mov_b32 s80, s46 // (op 4242) reg expression
s_mov_b32 s81, s47 // (op 4242) reg expression
s_mov_b32 s82, s52 // (op 4242) reg expression
s_mov_b32 s83, 131072 // (op 4242) call()

v_lshrrev_b32 v21, 2, v1 // (op 4259) 
s_mul_i32 s52, s16, s8 // (op 4259) 
s_add_u32 s52, s52, s0 // (op 4259) 
v_lshlrev_b32 v25, 1, v21 // (op 4259) 

s_mul_hi_u32 s53, s52, s13 // (op 4259) 
v_and_b32 v21, -4, v1 // (op 4259) 
s_sub_u32 s54, s52, s53 // (op 4259) 
v_sub_u32 v27, v1, v21 // (op 4259) 

s_lshr_b32 s55, s54, 1 // (op 4259) 

s_add_u32 s54, s55, s53 // (op 4259) 


s_lshr_b32 s53, s54, s11 // (op 4259) 

s_and_b32 s54, s11, 2147483648 // (op 4259) 
s_cmp_eq_i32 s54, 0 // (op 4259) 
s_mov_b32 s55, scc // (op 4259) 

s_cmp_eq_i32 s55, 0 // (op 4259) 
s_cselect_b32 s54, s52, s53 // (op 4259) 


s_mul_hi_u32 s53, s54, s12 // (op 4259) 
s_sub_u32 s55, s54, s53 // (op 4259) 
s_lshr_b32 s84, s55, 1 // (op 4259) 

s_add_u32 s55, s84, s53 // (op 4259) 


s_lshr_b32 s53, s55, s10 // (op 4259) 

s_and_b32 s55, s10, 2147483648 // (op 4259) 
s_cmp_eq_i32 s55, 0 // (op 4259) 
s_mov_b32 s84, scc // (op 4259) 

s_cmp_eq_i32 s84, 0 // (op 4259) 
s_cselect_b32 s55, s54, s53 // (op 4259) 


s_mul_i32 s53, s55, s14 // (op 4259) 

s_sub_u32 s55, s54, s53 // (op 4259) 

v_lshl_add_u32 v21, s55, 7, v25 // (op 4259) 


v_mul_lo_u32 v25, v21, s44 // (op 4259) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 4259) 
s_mul_i32 s53, s54, s84 // (op 4259) 


s_sub_u32 s54, s52, s53 // (op 4259) 


v_lshl_add_u32 v21, s54, 2, v27 // (op 4259) 


v_add_u32 v27, v21, v25 // (op 4259) 


s_mul_i32 s53, s44, s23 // (op 4261) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 4261) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 4261) least significant
s_mul_hi_u32 s55, s44, s22 // (op 4261) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4261) most significant: sum
s_add_u32 s53, s53, s55 // (op 4261) most significant: sum


s_mov_b32 s80, s46 // (op 4261) reg expression
s_mov_b32 s81, s47 // (op 4261) reg expression
s_mov_b32 s82, s52 // (op 4261) reg expression
s_mov_b32 s83, 131072 // (op 4261) call()




buffer_load_ubyte v21, v23, s[76:79], 0 offen offset: 0 // (op 3501) Load value
v_and_b32 v25, -8, v1 // (op 4416) 
v_sub_u32 v36, v1, v25 // (op 4416) 

v_lshlrev_b32 v25, 4, v36 // (op 4416) 

v_lshrrev_b32 v36, 3, v1 // (op 4416) 
v_lshl_add_u32 v37, v36, 7, v25 // (op 4416) 


s_nop 0
v_readfirstlane_b32 s52, v37 // (op 4416) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5163) 

s_lshl_b64 s[54:55], s[36:37], 5 // (op 3730) 
v_mov_b64 v[36:37], s[54:55] // (op 3730) convert


v_mov_b64 v[38:39], v[32:33] // (op 3730) 
s_mov_b32 m0, s52 // (op 3730) reg expression
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 3730) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 3730) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 3730) most significant half

s_add_u32 m0, m0, 4096 // (op 3730) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 3730) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 3730) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 3730) most significant half

s_add_u32 m0, m0, 4096 // (op 3730) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 3730) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 3730) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 3730) most significant half

s_add_u32 m0, m0, 4096 // (op 3730) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 3730) Load value direct to lds




v_mov_b64 v[36:37], s[44:45] // (op 3508) convert

v_mov_b32 v25, v26 // (op 3508) 
buffer_load_ubyte v38, v25, s[80:83], 0 offen offset: 0 // (op 3508) Load value
v_add_u32 v25, v25, v36 // (op 3508) 
buffer_load_ubyte v39, v25, s[80:83], 0 offen offset: 0 // (op 3508) Load value


v_and_b32 v25, -8, v1 // (op 4351) 
v_sub_u32 v36, v1, v25 // (op 4351) 

v_lshlrev_b32 v25, 4, v36 // (op 4351) 

v_lshrrev_b32 v36, 3, v1 // (op 4351) 
v_lshl_add_u32 v37, v36, 7, v25 // (op 4351) 


s_nop 0
v_readfirstlane_b32 s52, v37 // (op 4351) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5144) 


s_nop 0
s_add_u32 s53, 16384, s52 // (op 3735) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s53 // (op 3735) reg expression
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 3735) Load value direct to lds





buffer_load_ubyte v25, v23, s[76:79], 0 offen offset:4 // (op 3515) Load value

v_and_b32 v23, -8, v1 // (op 5005) 
v_sub_u32 v36, v1, v23 // (op 5005) 

v_lshlrev_b32 v23, 4, v36 // (op 5005) 

v_lshrrev_b32 v36, 3, v1 // (op 5005) 
v_lshl_add_u32 v37, v36, 7, v23 // (op 5005) 


s_nop 0
v_readfirstlane_b32 s52, v37 // (op 5005) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5293) 

s_lshl_b64 s[54:55], s[36:37], 5 // (op 3740) 
v_mov_b64 v[36:37], s[54:55] // (op 3740) convert


v_mov_b64 v[40:41], v[32:33] // (op 3740) 
s_add_u32 s53, 20480, s52 // (op 3740) 
s_mov_b32 m0, s53 // (op 3740) reg expression
s_mov_b32 s54, 128 // (op 3740) call()
buffer_load_dwordx4 v40, s[72:75], s54 offen offset: 0 lds // (op 3740) Load value direct to lds
v_add_co_u32 v40, s[84:85], v40, v36 // (op 3740) least significant half
v_addc_co_u32 v41, s[84:85], 0, v41, s[84:85] // (op 3740) most significant half

s_add_u32 m0, m0, 4096 // (op 3740) 
buffer_load_dwordx4 v40, s[72:75], s54 offen offset: 0 lds // (op 3740) Load value direct to lds
v_add_co_u32 v40, s[84:85], v40, v36 // (op 3740) least significant half
v_addc_co_u32 v41, s[84:85], 0, v41, s[84:85] // (op 3740) most significant half

s_add_u32 m0, m0, 4096 // (op 3740) 
buffer_load_dwordx4 v40, s[72:75], s54 offen offset: 0 lds // (op 3740) Load value direct to lds
v_add_co_u32 v40, s[84:85], v40, v36 // (op 3740) least significant half
v_addc_co_u32 v41, s[84:85], 0, v41, s[84:85] // (op 3740) most significant half

s_add_u32 m0, m0, 4096 // (op 3740) 
buffer_load_dwordx4 v40, s[72:75], s54 offen offset: 0 lds // (op 3740) Load value direct to lds







v_mov_b64 v[32:33], s[44:45] // (op 3522) convert

v_mov_b32 v23, v26 // (op 3522) 
buffer_load_ubyte v36, v23, s[80:83], 0 offen offset:4 // (op 3522) Load value
v_add_u32 v23, v23, v32 // (op 3522) 
buffer_load_ubyte v37, v23, s[80:83], 0 offen offset:4 // (op 3522) Load value



v_and_b32 v23, -8, v1 // (op 4966) 
v_sub_u32 v26, v1, v23 // (op 4966) 

v_lshlrev_b32 v23, 4, v26 // (op 4966) 

v_lshrrev_b32 v26, 3, v1 // (op 4966) 
v_lshl_add_u32 v32, v26, 7, v23 // (op 4966) 


s_nop 0
v_readfirstlane_b32 s52, v32 // (op 4966) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5284) 


s_nop 0
s_add_u32 s53, 36864, s52 // (op 3742) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s53 // (op 3742) reg expression
s_mov_b32 s54, 128 // (op 3742) call()
buffer_load_dwordx4 v28, s[68:71], s54 offen offset: 0 lds // (op 3742) Load value direct to lds




v_and_b32 v23, -4, v1 // (op 4492) 
v_sub_u32 v26, v1, v23 // (op 4492) 

v_lshrrev_b32 v23, 2, v1 // (op 4492) 
v_lshl_add_u32 v28, v23, 3, v26 // (op 4492) 


s_barrier  // (op 5182) 



s_waitcnt vmcnt(10)
ds_write_b8 v28, v38 offset:40960 // (op 3528) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v28, v39 offset:40964 // (op 3528) Store local data 

s_barrier  // (op 5160) 



ds_read_b128 v[40:43], v19 // (op 3546) Load local data 
ds_read_b128 v[44:47], v19 offset:64 // (op 3546) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 5179) 



ds_read_u8 v23, v16 offset:40960 // (op 3566) Load local data 



ds_read_b128 v[48:51], v19 offset:4096 // (op 3551) Load local data 
ds_read_b128 v[52:55], v19 offset:4160 // (op 3551) Load local data 



ds_read_u8 v26, v16 offset:41088 // (op 3571) Load local data 



ds_read_b128 v[56:59], v19 offset:8192 // (op 3556) Load local data 
ds_read_b128 v[60:63], v19 offset:8256 // (op 3556) Load local data 



ds_read_u8 v28, v16 offset:41216 // (op 3576) Load local data 



ds_read_b128 v[64:67], v19 offset:12288 // (op 3561) Load local data 
ds_read_b128 v[68:71], v19 offset:12352 // (op 3561) Load local data 



ds_read_u8 v29, v16 offset:41344 // (op 3581) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 5141) 



ds_read_b128 v[72:75], v17 offset:16384 // (op 3542) Load local data 
ds_read_b128 v[76:79], v17 offset:16448 // (op 3542) Load local data 
v_and_b32 v32, -8, v1 // (op 4323) 
v_sub_u32 v33, v1, v32 // (op 4323) 

v_lshlrev_b32 v32, 4, v33 // (op 4323) 

v_lshrrev_b32 v33, 3, v1 // (op 4323) 
v_lshl_add_u32 v80, v33, 7, v32 // (op 4323) 


s_nop 0
v_readfirstlane_b32 s52, v80 // (op 4323) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v32, -8, v1 // (op 4388) 
v_sub_u32 v33, v1, v32 // (op 4388) 

v_lshlrev_b32 v32, 4, v33 // (op 4388) 

v_lshrrev_b32 v33, 3, v1 // (op 4388) 
v_lshl_add_u32 v80, v33, 7, v32 // (op 4388) 


s_nop 0
v_readfirstlane_b32 s53, v80 // (op 4388) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v32, -4, v1 // (op 4463) 
v_sub_u32 v33, v1, v32 // (op 4463) 

v_lshrrev_b32 v32, 2, v1 // (op 4463) 
v_lshl_add_u32 v80, v32, 3, v33 // (op 4463) 


v_and_b32 v32, -8, v1 // (op 4951) 
v_sub_u32 v33, v1, v32 // (op 4951) 

v_lshlrev_b32 v32, 4, v33 // (op 4951) 

v_lshrrev_b32 v33, 3, v1 // (op 4951) 
v_lshl_add_u32 v81, v33, 7, v32 // (op 4951) 


s_nop 0
v_readfirstlane_b32 s54, v81 // (op 4951) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v32, -8, v1 // (op 4990) 
v_sub_u32 v33, v1, v32 // (op 4990) 

v_lshlrev_b32 v32, 4, v33 // (op 4990) 

v_lshrrev_b32 v33, 3, v1 // (op 4990) 
v_lshl_add_u32 v81, v33, 7, v32 // (op 4990) 


s_nop 0
v_readfirstlane_b32 s55, v81 // (op 4990) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v32, -4, v1 // (op 5034) 
v_sub_u32 v33, v1, v32 // (op 5034) 

v_lshrrev_b32 v32, 2, v1 // (op 5034) 
v_lshl_add_u32 v81, v32, 3, v33 // (op 5034) 


s_mov_b32 s1, 0 // (op 1739) call()
s_and_b32 s84, s7, 4294967294 // (op 1738) 
s_cmp_lt_u32 s1, s84 // (op 1738) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_1738 // (op 1738) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_1738 if false)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_1738:
s_add_u32 s1, s1, 2 // (op 1740) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:79], v[40:47], a[0:3], v21, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 84) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:79], v[48:55], a[4:7], v21, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1462) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:79], v[56:63], a[8:11], v21, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1618) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:79], v[64:71], a[12:15], v21, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1774) 


v_mov_b32 v32, v24 // (op 77) 

buffer_load_ubyte v21, v32, s[76:79], 0 offen offset:8 // (op 77) Load value
s_barrier  // (op 5166) 

s_lshl_b64 s[84:85], s[36:37], 5 // (op 3655) 
v_mov_b64 v[82:83], s[84:85] // (op 3655) convert


v_mov_b64 v[84:85], v[34:35] // (op 3655) 
v_mov_b64 v[86:87], v[84:85] // (op 3655) 
s_mov_b32 m0, s53 // (op 3655) reg expression
s_mov_b32 s84, 256 // (op 3655) call()
buffer_load_dwordx4 v86, s[72:75], s84 offen offset: 0 lds // (op 3655) Load value direct to lds
v_add_co_u32 v86, s[86:87], v86, v82 // (op 3655) least significant half
v_addc_co_u32 v87, s[86:87], 0, v87, s[86:87] // (op 3655) most significant half

s_add_u32 m0, m0, 4096 // (op 3655) 
buffer_load_dwordx4 v86, s[72:75], s84 offen offset: 0 lds // (op 3655) Load value direct to lds
v_add_co_u32 v86, s[86:87], v86, v82 // (op 3655) least significant half
v_addc_co_u32 v87, s[86:87], 0, v87, s[86:87] // (op 3655) most significant half

s_add_u32 m0, m0, 4096 // (op 3655) 
buffer_load_dwordx4 v86, s[72:75], s84 offen offset: 0 lds // (op 3655) Load value direct to lds
v_add_co_u32 v86, s[86:87], v86, v82 // (op 3655) least significant half
v_addc_co_u32 v87, s[86:87], 0, v87, s[86:87] // (op 3655) most significant half

s_add_u32 m0, m0, 4096 // (op 3655) 
buffer_load_dwordx4 v86, s[72:75], s84 offen offset: 0 lds // (op 3655) Load value direct to lds




v_mov_b64 v[82:83], s[44:45] // (op 21) convert

v_mov_b32 v33, v27 // (op 21) 
v_mov_b32 v86, v33 // (op 21) 
buffer_load_ubyte v38, v86, s[80:83], 0 offen offset:8 // (op 21) Load value
v_add_u32 v86, v86, v82 // (op 21) 
buffer_load_ubyte v39, v86, s[80:83], 0 offen offset:8 // (op 21) Load value


s_barrier  // (op 5147) 


v_mov_b64 v[82:83], v[30:31] // (op 3650) 
s_add_u32 s84, 16384, s52 // (op 3650) 

s_mov_b32 m0, s84 // (op 3650) reg expression
s_mov_b32 s85, 256 // (op 3650) call()
buffer_load_dwordx4 v82, s[68:71], s85 offen offset: 0 lds // (op 3650) Load value direct to lds


s_barrier  // (op 5302) 



s_waitcnt vmcnt(10)
ds_write_b8 v81, v36 offset:41472 // (op 1355) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v81, v37 offset:41476 // (op 1355) Store local data 
s_barrier  // (op 5290) 



ds_read_b128 v[40:43], v13 offset:20480 // (op 1348) Load local data 
ds_read_b128 v[44:47], v13 offset:20544 // (op 1348) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 5299) 



ds_read_u8 v23, v12 offset:41472 // (op 1352) Load local data 



ds_read_b128 v[48:51], v13 offset:24576 // (op 1467) Load local data 
ds_read_b128 v[52:55], v13 offset:24640 // (op 1467) Load local data 



ds_read_u8 v26, v12 offset:41600 // (op 1470) Load local data 



ds_read_b128 v[56:59], v13 offset:28672 // (op 1623) Load local data 
ds_read_b128 v[60:63], v13 offset:28736 // (op 1623) Load local data 



ds_read_u8 v28, v12 offset:41728 // (op 1626) Load local data 



ds_read_b128 v[64:67], v13 offset:32768 // (op 1779) Load local data 
ds_read_b128 v[68:71], v13 offset:32832 // (op 1779) Load local data 



ds_read_u8 v29, v12 offset:41856 // (op 1782) Load local data 
s_barrier  // (op 5497) 
s_waitcnt vmcnt(8)
s_barrier  // (op 5281) 



ds_read_b128 v[72:75], v6 offset:36864 // (op 1346) Load local data 
ds_read_b128 v[76:79], v6 offset:36928 // (op 1346) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:79], v[40:47], a[0:3], v25, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1353) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:79], v[48:55], a[4:7], v25, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1463) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:79], v[56:63], a[8:11], v25, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1619) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:79], v[64:71], a[12:15], v25, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1775) 



buffer_load_ubyte v25, v32, s[76:79], 0 offen offset:12 // (op 1350) Load value

s_barrier  // (op 5296) 
v_add_u32 v24, 8, v24 // (op 4123) 

s_lshl_b64 s[84:85], s[36:37], 5 // (op 3711) 
v_mov_b64 v[86:87], s[84:85] // (op 3711) convert


v_mov_b64 v[88:89], v[84:85] // (op 3711) 
s_add_u32 s84, 20480, s55 // (op 3711) 
s_mov_b32 m0, s84 // (op 3711) reg expression
s_mov_b32 s85, 384 // (op 3711) call()
buffer_load_dwordx4 v88, s[72:75], s85 offen offset: 0 lds // (op 3711) Load value direct to lds
v_add_co_u32 v88, s[86:87], v88, v86 // (op 3711) least significant half
v_addc_co_u32 v89, s[86:87], 0, v89, s[86:87] // (op 3711) most significant half

s_add_u32 m0, m0, 4096 // (op 3711) 
buffer_load_dwordx4 v88, s[72:75], s85 offen offset: 0 lds // (op 3711) Load value direct to lds
v_add_co_u32 v88, s[86:87], v88, v86 // (op 3711) least significant half
v_addc_co_u32 v89, s[86:87], 0, v89, s[86:87] // (op 3711) most significant half

s_add_u32 m0, m0, 4096 // (op 3711) 
buffer_load_dwordx4 v88, s[72:75], s85 offen offset: 0 lds // (op 3711) Load value direct to lds
v_add_co_u32 v88, s[86:87], v88, v86 // (op 3711) least significant half
v_addc_co_u32 v89, s[86:87], 0, v89, s[86:87] // (op 3711) most significant half

s_add_u32 m0, m0, 4096 // (op 3711) 
buffer_load_dwordx4 v88, s[72:75], s85 offen offset: 0 lds // (op 3711) Load value direct to lds






v_add_co_u32 v34, vcc, 256, v34 // (op 3995) least significant half
v_addc_co_u32 v35, vcc, 0, v35, vcc // (op 3995) most significant half
v_mov_b64 v[84:85], s[44:45] // (op 1357) convert

v_mov_b32 v32, v33 // (op 1357) 
buffer_load_ubyte v36, v32, s[80:83], 0 offen offset:12 // (op 1357) Load value
v_add_u32 v32, v32, v84 // (op 1357) 
buffer_load_ubyte v37, v32, s[80:83], 0 offen offset:12 // (op 1357) Load value



s_barrier  // (op 5287) 
v_add_u32 v27, 8, v27 // (op 4251) 


s_add_u32 s84, 36864, s54 // (op 3706) 

s_mov_b32 m0, s84 // (op 3706) reg expression
s_mov_b32 s85, 384 // (op 3706) call()
buffer_load_dwordx4 v82, s[68:71], s85 offen offset: 0 lds // (op 3706) Load value direct to lds



s_barrier  // (op 5185) 
v_add_co_u32 v30, vcc, 256, v30 // (op 3867) least significant half



v_addc_co_u32 v31, vcc, 0, v31, vcc // (op 3867) most significant half
s_waitcnt vmcnt(10)
ds_write_b8 v80, v38 offset:40960 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v80, v39 offset:40964 // (op 26) Store local data 
s_barrier  // (op 5172) 



ds_read_b128 v[40:43], v7 // (op 72) Load local data 
ds_read_b128 v[44:47], v7 offset:64 // (op 72) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 5192) 



ds_read_u8 v23, v0 offset:40960 // (op 81) Load local data 



ds_read_b128 v[48:51], v7 offset:4096 // (op 1458) Load local data 
ds_read_b128 v[52:55], v7 offset:4160 // (op 1458) Load local data 



ds_read_u8 v26, v0 offset:41088 // (op 1461) Load local data 



ds_read_b128 v[56:59], v7 offset:8192 // (op 1614) Load local data 
ds_read_b128 v[60:63], v7 offset:8256 // (op 1614) Load local data 



ds_read_u8 v28, v0 offset:41216 // (op 1617) Load local data 



ds_read_b128 v[64:67], v7 offset:12288 // (op 1770) Load local data 
ds_read_b128 v[68:71], v7 offset:12352 // (op 1770) Load local data 



ds_read_u8 v29, v0 offset:41344 // (op 1773) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 5153) 



ds_read_b128 v[72:75], v3 offset:16384 // (op 67) Load local data 
ds_read_b128 v[76:79], v3 offset:16448 // (op 67) Load local data 
s_and_b32 s84, s7, 4294967294 // (op 1738) 
s_cmp_lt_u32 s1, s84 // (op 1738) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_1738 // (op 1738) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_1738 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_1738:












s_barrier  // (op 5481) 
s_and_b32 s52, s7, 4294967294 // (op 3779) 
v_lshrrev_b32 v21, 3, v1 // (op 3779) 
s_add_u32 s53, s0, s52 // (op 3779) 

v_and_b32 v24, -8, v1 // (op 3779) 
v_sub_u32 v27, v1, v24 // (op 3779) 

s_mul_i32 s52, s16, s8 // (op 3779) 
s_add_u32 s52, s52, s53 // (op 3779) 

v_lshlrev_b32 v24, 4, v27 // (op 3779) 

s_mul_hi_u32 s53, s52, s13 // (op 3779) 
s_sub_u32 s54, s52, s53 // (op 3779) 
s_lshr_b32 s55, s54, 1 // (op 3779) 

s_add_u32 s54, s55, s53 // (op 3779) 


s_lshr_b32 s53, s54, s11 // (op 3779) 

s_and_b32 s54, s11, 2147483648 // (op 3779) 
s_cmp_eq_i32 s54, 0 // (op 3779) 
s_mov_b32 s55, scc // (op 3779) 

s_cmp_eq_i32 s55, 0 // (op 3779) 
s_cselect_b32 s54, s52, s53 // (op 3779) 


s_mul_hi_u32 s53, s54, s12 // (op 3779) 
s_sub_u32 s55, s54, s53 // (op 3779) 
s_lshr_b32 s84, s55, 1 // (op 3779) 

s_add_u32 s55, s84, s53 // (op 3779) 


s_lshr_b32 s53, s55, s10 // (op 3779) 

s_and_b32 s55, s10, 2147483648 // (op 3779) 
s_cmp_eq_i32 s55, 0 // (op 3779) 
s_mov_b32 s84, scc // (op 3779) 

s_cmp_eq_i32 s84, 0 // (op 3779) 
s_cselect_b32 s55, s54, s53 // (op 3779) 


v_lshl_add_u32 v27, s55, 5, v21 // (op 3779) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 3779) 
s_mul_i32 s53, s54, s84 // (op 3779) 


s_sub_u32 s54, s52, s53 // (op 3779) 


v_lshl_add_u32 v21, s54, 7, v24 // (op 3779) 


v_mov_b32 v39, 0 // (op 3779) convert
v_mov_b32 v38, v21 // (op 3779) convert

v_mul_lo_u32 v31, v27, s31 // (op 3779) most significant: low of low * high
v_mul_hi_u32 v21, v27, s30 // (op 3779) most significant: high of low * low
v_mul_lo_u32 v30, v27, s30 // (op 3779) least significant: low of low * low
v_add_u32 v31, v31, v21 // (op 3779) most significant: sum

v_add_co_u32 v30, s[52:53], v30, v38 // (op 3779) least significant half
v_addc_co_u32 v31, s[52:53], v31, v39, s[52:53] // (op 3779) most significant half



s_mul_i32 s53, s30, s35 // (op 3781) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 3781) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 3781) least significant
s_mul_hi_u32 s55, s30, s34 // (op 3781) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3781) most significant: sum
s_add_u32 s53, s53, s55 // (op 3781) most significant: sum


s_mov_b32 s68, s32 // (op 3781) reg expression
s_mov_b32 s69, s33 // (op 3781) reg expression
s_mov_b32 s70, s52 // (op 3781) reg expression
s_mov_b32 s71, 131072 // (op 3781) call()

s_and_b32 s52, s7, 4294967294 // (op 3919) 
v_lshrrev_b32 v21, 3, v1 // (op 3919) 
s_add_u32 s53, s0, s52 // (op 3919) 

v_and_b32 v24, -8, v1 // (op 3919) 
v_sub_u32 v27, v1, v24 // (op 3919) 

s_mul_i32 s52, s16, s8 // (op 3919) 
s_add_u32 s52, s52, s53 // (op 3919) 

v_lshlrev_b32 v24, 4, v27 // (op 3919) 

s_mul_hi_u32 s53, s52, s13 // (op 3919) 
s_sub_u32 s54, s52, s53 // (op 3919) 
s_lshr_b32 s55, s54, 1 // (op 3919) 

s_add_u32 s54, s55, s53 // (op 3919) 


s_lshr_b32 s53, s54, s11 // (op 3919) 

s_and_b32 s54, s11, 2147483648 // (op 3919) 
s_cmp_eq_i32 s54, 0 // (op 3919) 
s_mov_b32 s55, scc // (op 3919) 

s_cmp_eq_i32 s55, 0 // (op 3919) 
s_cselect_b32 s54, s52, s53 // (op 3919) 


s_mul_hi_u32 s53, s54, s12 // (op 3919) 
s_sub_u32 s55, s54, s53 // (op 3919) 
s_lshr_b32 s84, s55, 1 // (op 3919) 

s_add_u32 s55, s84, s53 // (op 3919) 


s_lshr_b32 s53, s55, s10 // (op 3919) 

s_and_b32 s55, s10, 2147483648 // (op 3919) 
s_cmp_eq_i32 s55, 0 // (op 3919) 
s_mov_b32 s84, scc // (op 3919) 

s_cmp_eq_i32 s84, 0 // (op 3919) 
s_cselect_b32 s55, s54, s53 // (op 3919) 


s_mul_i32 s53, s55, s14 // (op 3919) 

s_sub_u32 s55, s54, s53 // (op 3919) 

v_lshl_add_u32 v27, s55, 7, v21 // (op 3919) 


v_mul_lo_u32 v39, v27, s37 // (op 3919) most significant: low of low * high
v_mul_hi_u32 v21, v27, s36 // (op 3919) most significant: high of low * low
v_mul_lo_u32 v38, v27, s36 // (op 3919) least significant: low of low * low
v_add_u32 v39, v39, v21 // (op 3919) most significant: sum


s_ashr_i64 s[84:85], s[2:3], 7 // (op 3919) 
s_mul_i32 s53, s54, s84 // (op 3919) 


s_sub_u32 s54, s52, s53 // (op 3919) 


v_lshl_add_u32 v21, s54, 7, v24 // (op 3919) 


v_mov_b32 v33, 0 // (op 3919) convert
v_mov_b32 v32, v21 // (op 3919) convert

v_add_co_u32 v34, s[52:53], v32, v38 // (op 3919) least significant half
v_addc_co_u32 v35, s[52:53], v33, v39, s[52:53] // (op 3919) most significant half



s_mul_i32 s53, s36, s23 // (op 3921) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 3921) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 3921) least significant
s_mul_hi_u32 s55, s36, s22 // (op 3921) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3921) most significant: sum
s_add_u32 s53, s53, s55 // (op 3921) most significant: sum


s_mov_b32 s72, s38 // (op 3921) reg expression
s_mov_b32 s73, s39 // (op 3921) reg expression
s_mov_b32 s74, s52 // (op 3921) reg expression
s_mov_b32 s75, 131072 // (op 3921) call()

s_and_b32 s52, s7, 4294967294 // (op 4047) 
v_lshrrev_b32 v21, 7, v1 // (op 4047) 
s_add_u32 s53, s0, s52 // (op 4047) 

v_and_b32 v24, 15, v1 // (op 4047) 
v_lshl_add_u32 v27, v21, 4, v24 // (op 4047) 


s_mul_i32 s52, s16, s8 // (op 4047) 
s_add_u32 s52, s52, s53 // (op 4047) 

v_and_b32 v21, 63, v1 // (op 4047) 
s_mul_hi_u32 s53, s52, s13 // (op 4047) 
v_lshrrev_b32 v24, 4, v21 // (op 4047) 

s_sub_u32 s54, s52, s53 // (op 4047) 
s_lshr_b32 s55, s54, 1 // (op 4047) 

s_add_u32 s54, s55, s53 // (op 4047) 


s_lshr_b32 s53, s54, s11 // (op 4047) 

s_and_b32 s54, s11, 2147483648 // (op 4047) 
s_cmp_eq_i32 s54, 0 // (op 4047) 
s_mov_b32 s55, scc // (op 4047) 

s_cmp_eq_i32 s55, 0 // (op 4047) 
s_cselect_b32 s54, s52, s53 // (op 4047) 


s_mul_hi_u32 s53, s54, s12 // (op 4047) 
s_sub_u32 s55, s54, s53 // (op 4047) 
s_lshr_b32 s84, s55, 1 // (op 4047) 

s_add_u32 s55, s84, s53 // (op 4047) 


s_lshr_b32 s53, s55, s10 // (op 4047) 

s_and_b32 s55, s10, 2147483648 // (op 4047) 
s_cmp_eq_i32 s55, 0 // (op 4047) 
s_mov_b32 s84, scc // (op 4047) 

s_cmp_eq_i32 s84, 0 // (op 4047) 
s_cselect_b32 s55, s54, s53 // (op 4047) 


v_lshl_add_u32 v21, s55, 5, v27 // (op 4047) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 4047) 
s_mul_i32 s53, s54, s84 // (op 4047) 


s_sub_u32 s54, s52, s53 // (op 4047) 


v_lshl_add_u32 v27, s54, 2, v24 // (op 4047) 


v_mul_lo_u32 v24, v21, s40 // (op 4047) 
v_add_u32 v24, v24, v27 // (op 4047) 


s_mul_i32 s53, s40, s35 // (op 4049) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 4049) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 4049) least significant
s_mul_hi_u32 s55, s40, s34 // (op 4049) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4049) most significant: sum
s_add_u32 s53, s53, s55 // (op 4049) most significant: sum


s_mov_b32 s76, s42 // (op 4049) reg expression
s_mov_b32 s77, s43 // (op 4049) reg expression
s_mov_b32 s78, s52 // (op 4049) reg expression
s_mov_b32 s79, 131072 // (op 4049) call()

s_and_b32 s52, s7, 4294967294 // (op 4175) 
v_lshrrev_b32 v21, 2, v1 // (op 4175) 
s_add_u32 s53, s0, s52 // (op 4175) 

v_lshlrev_b32 v27, 1, v21 // (op 4175) 

v_and_b32 v21, -4, v1 // (op 4175) 
s_mul_i32 s52, s16, s8 // (op 4175) 
s_add_u32 s52, s52, s53 // (op 4175) 

v_sub_u32 v32, v1, v21 // (op 4175) 

s_mul_hi_u32 s53, s52, s13 // (op 4175) 
s_sub_u32 s54, s52, s53 // (op 4175) 
s_lshr_b32 s55, s54, 1 // (op 4175) 

s_add_u32 s54, s55, s53 // (op 4175) 


s_lshr_b32 s53, s54, s11 // (op 4175) 

s_and_b32 s54, s11, 2147483648 // (op 4175) 
s_cmp_eq_i32 s54, 0 // (op 4175) 
s_mov_b32 s55, scc // (op 4175) 

s_cmp_eq_i32 s55, 0 // (op 4175) 
s_cselect_b32 s54, s52, s53 // (op 4175) 


s_mul_hi_u32 s53, s54, s12 // (op 4175) 
s_sub_u32 s55, s54, s53 // (op 4175) 
s_lshr_b32 s84, s55, 1 // (op 4175) 

s_add_u32 s55, s84, s53 // (op 4175) 


s_lshr_b32 s53, s55, s10 // (op 4175) 

s_and_b32 s55, s10, 2147483648 // (op 4175) 
s_cmp_eq_i32 s55, 0 // (op 4175) 
s_mov_b32 s84, scc // (op 4175) 

s_cmp_eq_i32 s84, 0 // (op 4175) 
s_cselect_b32 s55, s54, s53 // (op 4175) 


s_mul_i32 s53, s55, s14 // (op 4175) 

s_sub_u32 s55, s54, s53 // (op 4175) 

v_lshl_add_u32 v21, s55, 7, v27 // (op 4175) 


v_mul_lo_u32 v27, v21, s44 // (op 4175) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 4175) 
s_mul_i32 s53, s54, s84 // (op 4175) 


s_sub_u32 s54, s52, s53 // (op 4175) 


v_lshl_add_u32 v21, s54, 2, v32 // (op 4175) 


v_add_u32 v32, v21, v27 // (op 4175) 


s_mul_i32 s53, s44, s23 // (op 4177) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 4177) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 4177) least significant
s_mul_hi_u32 s55, s44, s22 // (op 4177) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4177) most significant: sum
s_add_u32 s53, s53, s55 // (op 4177) most significant: sum


s_mov_b32 s80, s46 // (op 4177) reg expression
s_mov_b32 s81, s47 // (op 4177) reg expression
s_mov_b32 s82, s52 // (op 4177) reg expression
s_mov_b32 s83, 131072 // (op 4177) call()

v_and_b32 v21, -8, v1 // (op 4336) 
v_sub_u32 v27, v1, v21 // (op 4336) 

v_lshlrev_b32 v21, 4, v27 // (op 4336) 

v_lshrrev_b32 v27, 3, v1 // (op 4336) 
v_lshl_add_u32 v33, v27, 7, v21 // (op 4336) 


s_nop 0
v_readfirstlane_b32 s52, v33 // (op 4336) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -8, v1 // (op 4401) 
v_sub_u32 v27, v1, v21 // (op 4401) 

v_lshlrev_b32 v21, 4, v27 // (op 4401) 

v_lshrrev_b32 v27, 3, v1 // (op 4401) 
v_lshl_add_u32 v33, v27, 7, v21 // (op 4401) 


s_nop 0
v_readfirstlane_b32 s53, v33 // (op 4401) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -4, v1 // (op 4476) 
v_sub_u32 v27, v1, v21 // (op 4476) 

v_lshrrev_b32 v21, 2, v1 // (op 4476) 
v_lshl_add_u32 v33, v21, 3, v27 // (op 4476) 


s_and_b32 s54, s7, 4294967294 // (op 1744) 
s_cmp_lt_u32 s54, s7 // (op 1743) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_1743 // (op 1743) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_1743 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_1743:
s_barrier  // (op 5150) 
s_add_u32 s54, s54, 1 // (op 1745) 


v_mov_b64 v[28:29], v[30:31] // (op 3693) 
s_add_u32 s55, 16384, s52 // (op 3693) 

s_mov_b32 m0, s55 // (op 3693) reg expression
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 3693) Load value direct to lds


v_add_co_u32 v30, vcc, 128, v30 // (op 3769) least significant half
v_addc_co_u32 v31, vcc, 0, v31, vcc // (op 3769) most significant half
s_barrier  // (op 5169) 

s_lshl_b64 s[84:85], s[36:37], 5 // (op 3699) 
v_mov_b64 v[28:29], s[84:85] // (op 3699) convert


v_mov_b64 v[26:27], v[34:35] // (op 3699) 
v_mov_b64 v[36:37], v[26:27] // (op 3699) 
s_mov_b32 m0, s53 // (op 3699) reg expression
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 3699) Load value direct to lds
v_add_co_u32 v36, s[84:85], v36, v28 // (op 3699) least significant half
v_addc_co_u32 v37, s[84:85], 0, v37, s[84:85] // (op 3699) most significant half

s_add_u32 m0, m0, 4096 // (op 3699) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 3699) Load value direct to lds
v_add_co_u32 v36, s[84:85], v36, v28 // (op 3699) least significant half
v_addc_co_u32 v37, s[84:85], 0, v37, s[84:85] // (op 3699) most significant half

s_add_u32 m0, m0, 4096 // (op 3699) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 3699) Load value direct to lds
v_add_co_u32 v36, s[84:85], v36, v28 // (op 3699) least significant half
v_addc_co_u32 v37, s[84:85], 0, v37, s[84:85] // (op 3699) most significant half

s_add_u32 m0, m0, 4096 // (op 3699) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 3699) Load value direct to lds



v_add_co_u32 v34, vcc, 128, v34 // (op 3911) least significant half
v_addc_co_u32 v35, vcc, 0, v35, vcc // (op 3911) most significant half

v_mov_b64 v[28:29], s[44:45] // (op 1299) convert

v_mov_b32 v21, v32 // (op 1299) 
v_mov_b32 v23, v21 // (op 1299) 
buffer_load_ubyte v26, v23, s[80:83], 0 offen offset: 0 // (op 1299) Load value
v_add_u32 v23, v23, v28 // (op 1299) 
buffer_load_ubyte v27, v23, s[80:83], 0 offen offset: 0 // (op 1299) Load value



v_add_u32 v32, 4, v32 // (op 4167) 
s_barrier  // (op 5188) 
s_waitcnt vmcnt(6)
s_barrier  // (op 5156) 






ds_read_b128 v[36:39], v3 offset:16384 // (op 1288) Load local data 
ds_read_b128 v[40:43], v3 offset:16448 // (op 1288) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v33, v26 offset:40960 // (op 1297) Store local data 
s_barrier  // (op 5175) 
s_waitcnt vmcnt(0)
ds_write_b8 v33, v27 offset:40964 // (op 1297) Store local data 




ds_read_b128 v[44:47], v7 // (op 1290) Load local data 
ds_read_b128 v[48:51], v7 offset:64 // (op 1290) Load local data 


v_mov_b32 v21, v24 // (op 1292) 

buffer_load_ubyte v23, v21, s[76:79], 0 offen offset: 0 // (op 1292) Load value

v_add_u32 v24, 4, v24 // (op 4039) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 5195) 



ds_read_u8 v21, v0 offset:40960 // (op 1294) Load local data 



ds_read_b128 v[52:55], v7 offset:4096 // (op 1439) Load local data 
ds_read_b128 v[56:59], v7 offset:4160 // (op 1439) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:51], a[0:3], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1295) 





ds_read_u8 v21, v0 offset:41088 // (op 1442) Load local data 



ds_read_b128 v[44:47], v7 offset:8192 // (op 1595) Load local data 
ds_read_b128 v[48:51], v7 offset:8256 // (op 1595) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:43], v[52:59], a[4:7], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1443) 





ds_read_u8 v21, v0 offset:41216 // (op 1598) Load local data 



ds_read_b128 v[52:55], v7 offset:12288 // (op 1751) Load local data 
ds_read_b128 v[56:59], v7 offset:12352 // (op 1751) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[36:43], v[44:51], a[8:11], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1599) 





ds_read_u8 v21, v0 offset:41344 // (op 1754) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[36:43], v[52:59], a[12:15], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1755) 




s_cmp_lt_u32 s54, s7 // (op 1743) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_1743 // (op 1743) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_1743 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_1743:








s_barrier  // (op 5485) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 280) 
s_cmp_lg_u32 s18, 0 // (op 306) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Send_Tile_306 // (op 306) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Send_Tile_306



buffer_store_dword a0, v20, s[64:67], 0 offen sc1 // (op 303) Store value
buffer_store_dword a1, v20, s[64:67], 0 offen offset:512 sc1 // (op 303) Store value
buffer_store_dword a2, v20, s[64:67], 0 offen offset:1024 sc1 // (op 303) Store value
buffer_store_dword a3, v20, s[64:67], 0 offen offset:1536 sc1 // (op 303) Store value



buffer_store_dword a4, v20, s[64:67], 0 offen offset:128 sc1 // (op 1978) Store value
buffer_store_dword a5, v20, s[64:67], 0 offen offset:640 sc1 // (op 1978) Store value
buffer_store_dword a6, v20, s[64:67], 0 offen offset:1152 sc1 // (op 1978) Store value
buffer_store_dword a7, v20, s[64:67], 0 offen offset:1664 sc1 // (op 1978) Store value



buffer_store_dword a8, v20, s[64:67], 0 offen offset:256 sc1 // (op 1979) Store value
buffer_store_dword a9, v20, s[64:67], 0 offen offset:768 sc1 // (op 1979) Store value
buffer_store_dword a10, v20, s[64:67], 0 offen offset:1280 sc1 // (op 1979) Store value
buffer_store_dword a11, v20, s[64:67], 0 offen offset:1792 sc1 // (op 1979) Store value



buffer_store_dword a12, v20, s[64:67], 0 offen offset:384 sc1 // (op 1980) Store value
buffer_store_dword a13, v20, s[64:67], 0 offen offset:896 sc1 // (op 1980) Store value
buffer_store_dword a14, v20, s[64:67], 0 offen offset:1408 sc1 // (op 1980) Store value
buffer_store_dword a15, v20, s[64:67], 0 offen offset:1920 sc1 // (op 1980) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 307) 
v_cmp_eq_i32 vcc, v1, 0 // (op 311) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_311 // (op 311) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_311
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
s_branch RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_311 // (op 311) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_311
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_311:
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_311:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Send_Tile_306 // (op 306) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Send_Tile_306
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Send_Tile_306:
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Send_Tile_306:

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
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Receive_Tile_322 // (op 322) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Receive_Tile_322
s_mov_b32 s52, 0 // (op 288) call()
s_mul_i32 s53, s52, s8 // (op 287) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 287) 
s_mov_b32 s84, s54 // (op 287) convert

s_sub_u32 s54, s84, s19 // (op 287) 

s_sub_u32 s55, s54, 1 // (op 287) 

s_cmp_lt_u32 s53, s55 // (op 287) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_287 // (op 287) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_287 if false)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_ReceiveTileLoop_287:
s_add_u32 s53, s16, 1 // (op 4824) 
v_lshrrev_b32 v16, 5, v1 // (op 4824) 
s_add_u32 s54, s53, s52 // (op 4824) 
s_lshl_b32 s54, s54, 5 // (op 4824) 

v_lshlrev_b32 v17, 2, v16 // (op 4824) 

v_add_lshl_u32 v16, s54, v17, 7 // (op 4824) 


v_and_b32 v17, -32, v1 // (op 4824) 
v_sub_u32 v19, v1, v17 // (op 4824) 

v_add_lshl_u32 v17, v16, v19, 2 // (op 4824) 


s_lshl_b32 s53, s15, 14 // (op 4826) 
s_mov_b32 s64, s28 // (op 4826) reg expression
s_mov_b32 s65, s29 // (op 4826) reg expression
s_mov_b32 s66, s53 // (op 4826) reg expression
s_mov_b32 s67, 131072 // (op 4826) call()

RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_326:
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
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_326 // (op 326) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_326 if true)
s_barrier  // (op 330) 
v_cmp_eq_i32 vcc, v1, 0 // (op 329) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_329 // (op 329) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_329
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
s_branch RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_329 // (op 329) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_329
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_329:
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_329:




buffer_load_dword v24, v17, s[64:67], 0 offen // (op 341) Load value
buffer_load_dword v25, v17, s[64:67], 0 offen offset:512 // (op 341) Load value
buffer_load_dword v26, v17, s[64:67], 0 offen offset:1024 // (op 341) Load value
buffer_load_dword v27, v17, s[64:67], 0 offen offset:1536 // (op 341) Load value
v_accvgpr_read v28, a0 // (op 344) reg expression
v_accvgpr_read v29, a1 // (op 344) reg expression
v_accvgpr_read v30, a2 // (op 344) reg expression
v_accvgpr_read v31, a3 // (op 344) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 346) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 346) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 346) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 346) 

v_accvgpr_write a0, v28 // (op 348) reg expression
v_accvgpr_write a1, v29 // (op 348) reg expression
v_accvgpr_write a2, v30 // (op 348) reg expression
v_accvgpr_write a3, v31 // (op 348) reg expression




buffer_load_dword v24, v17, s[64:67], 0 offen offset:128 // (op 2000) Load value
buffer_load_dword v25, v17, s[64:67], 0 offen offset:640 // (op 2000) Load value
buffer_load_dword v26, v17, s[64:67], 0 offen offset:1152 // (op 2000) Load value
buffer_load_dword v27, v17, s[64:67], 0 offen offset:1664 // (op 2000) Load value
v_accvgpr_read v28, a4 // (op 2001) reg expression
v_accvgpr_read v29, a5 // (op 2001) reg expression
v_accvgpr_read v30, a6 // (op 2001) reg expression
v_accvgpr_read v31, a7 // (op 2001) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 2002) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 2002) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 2002) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 2002) 

v_accvgpr_write a4, v28 // (op 2003) reg expression
v_accvgpr_write a5, v29 // (op 2003) reg expression
v_accvgpr_write a6, v30 // (op 2003) reg expression
v_accvgpr_write a7, v31 // (op 2003) reg expression




buffer_load_dword v24, v17, s[64:67], 0 offen offset:256 // (op 2007) Load value
buffer_load_dword v25, v17, s[64:67], 0 offen offset:768 // (op 2007) Load value
buffer_load_dword v26, v17, s[64:67], 0 offen offset:1280 // (op 2007) Load value
buffer_load_dword v27, v17, s[64:67], 0 offen offset:1792 // (op 2007) Load value
v_accvgpr_read v28, a8 // (op 2008) reg expression
v_accvgpr_read v29, a9 // (op 2008) reg expression
v_accvgpr_read v30, a10 // (op 2008) reg expression
v_accvgpr_read v31, a11 // (op 2008) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 2009) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 2009) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 2009) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 2009) 

v_accvgpr_write a8, v28 // (op 2010) reg expression
v_accvgpr_write a9, v29 // (op 2010) reg expression
v_accvgpr_write a10, v30 // (op 2010) reg expression
v_accvgpr_write a11, v31 // (op 2010) reg expression




buffer_load_dword v24, v17, s[64:67], 0 offen offset:384 // (op 2014) Load value
buffer_load_dword v25, v17, s[64:67], 0 offen offset:896 // (op 2014) Load value
buffer_load_dword v26, v17, s[64:67], 0 offen offset:1408 // (op 2014) Load value
buffer_load_dword v27, v17, s[64:67], 0 offen offset:1920 // (op 2014) Load value
v_accvgpr_read v28, a12 // (op 2015) reg expression
v_accvgpr_read v29, a13 // (op 2015) reg expression
v_accvgpr_read v30, a14 // (op 2015) reg expression
v_accvgpr_read v31, a15 // (op 2015) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 2016) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 2016) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 2016) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 2016) 

v_accvgpr_write a12, v28 // (op 2017) reg expression
v_accvgpr_write a13, v29 // (op 2017) reg expression
v_accvgpr_write a14, v30 // (op 2017) reg expression
v_accvgpr_write a15, v31 // (op 2017) reg expression

s_add_u32 s52, s52, 1 // (op 289) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 287) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 287) 
s_mov_b32 s84, s54 // (op 287) convert

s_sub_u32 s54, s84, s19 // (op 287) 

s_sub_u32 s55, s54, 1 // (op 287) 

s_cmp_lt_u32 s53, s55 // (op 287) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_ReceiveTileLoop_287 // (op 287) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_ReceiveTileLoop_287 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_287:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Receive_Tile_322 // (op 322) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Receive_Tile_322
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Receive_Tile_322:
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Receive_Tile_322:

s_cmp_eq_i32 s18, 0 // (op 365) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_365 // (op 365) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_365
v_cmp_eq_f32 vcc, 0.00000, v14 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_125
v_accvgpr_read v24, a0 // (op 46) 
v_accvgpr_read v25, a1 // (op 46) 
v_accvgpr_read v26, a2 // (op 46) 
v_accvgpr_read v27, a3 // (op 46) 
v_mul_f32 v28, v15, v24 // (op 46) 
v_mul_f32 v29, v15, v25 // (op 46) 
v_mul_f32 v30, v15, v26 // (op 46) 
v_mul_f32 v31, v15, v27 // (op 46) 

v_cvt_f16_f32 v24, v28 // (op 52) 
v_cvt_f16_f32 v25, v29 // (op 52) 
v_cvt_f16_f32 v26, v30 // (op 52) 
v_cvt_f16_f32 v27, v31 // (op 52) 


v_pack_b32_f16 v16, v24, v25 // (op 54) 
v_pack_b32_f16 v17, v26, v27 // (op 54) 

v_mov_b32 v23, v22 // (op 54) reg expression

buffer_store_dwordx2 v[16:17], v23, s[60:63], 0 offen // (op 54) Store value



v_accvgpr_read v24, a4 // (op 2045) 
v_accvgpr_read v25, a5 // (op 2045) 
v_accvgpr_read v26, a6 // (op 2045) 
v_accvgpr_read v27, a7 // (op 2045) 
v_mul_f32 v28, v15, v24 // (op 2045) 
v_mul_f32 v29, v15, v25 // (op 2045) 
v_mul_f32 v30, v15, v26 // (op 2045) 
v_mul_f32 v31, v15, v27 // (op 2045) 

v_cvt_f16_f32 v24, v28 // (op 2047) 
v_cvt_f16_f32 v25, v29 // (op 2047) 
v_cvt_f16_f32 v26, v30 // (op 2047) 
v_cvt_f16_f32 v27, v31 // (op 2047) 


v_pack_b32_f16 v16, v24, v25 // (op 2048) 
v_pack_b32_f16 v17, v26, v27 // (op 2048) 

s_lshl_b32 s52, s24, 6 // (op 2048) 
v_add_u32 v23, v22, s52 // (op 2048) 


buffer_store_dwordx2 v[16:17], v23, s[60:63], 0 offen // (op 2048) Store value



v_accvgpr_read v24, a8 // (op 2059) 
v_accvgpr_read v25, a9 // (op 2059) 
v_accvgpr_read v26, a10 // (op 2059) 
v_accvgpr_read v27, a11 // (op 2059) 
v_mul_f32 v28, v15, v24 // (op 2059) 
v_mul_f32 v29, v15, v25 // (op 2059) 
v_mul_f32 v30, v15, v26 // (op 2059) 
v_mul_f32 v31, v15, v27 // (op 2059) 

v_cvt_f16_f32 v24, v28 // (op 2061) 
v_cvt_f16_f32 v25, v29 // (op 2061) 
v_cvt_f16_f32 v26, v30 // (op 2061) 
v_cvt_f16_f32 v27, v31 // (op 2061) 


v_pack_b32_f16 v16, v24, v25 // (op 2062) 
v_pack_b32_f16 v17, v26, v27 // (op 2062) 

s_lshl_b32 s52, s24, 7 // (op 2062) 
v_add_u32 v23, v22, s52 // (op 2062) 


buffer_store_dwordx2 v[16:17], v23, s[60:63], 0 offen // (op 2062) Store value



v_accvgpr_read v24, a12 // (op 2073) 
v_accvgpr_read v25, a13 // (op 2073) 
v_accvgpr_read v26, a14 // (op 2073) 
v_accvgpr_read v27, a15 // (op 2073) 
v_mul_f32 v28, v15, v24 // (op 2073) 
v_mul_f32 v29, v15, v25 // (op 2073) 
v_mul_f32 v30, v15, v26 // (op 2073) 
v_mul_f32 v31, v15, v27 // (op 2073) 

v_cvt_f16_f32 v24, v28 // (op 2075) 
v_cvt_f16_f32 v25, v29 // (op 2075) 
v_cvt_f16_f32 v26, v30 // (op 2075) 
v_cvt_f16_f32 v27, v31 // (op 2075) 


v_pack_b32_f16 v16, v24, v25 // (op 2076) 
v_pack_b32_f16 v17, v26, v27 // (op 2076) 

s_lshl_b32 s52, s24, 6 // (op 2076) 
s_mul_i32 s53, 3, s52 // (op 2076) 

v_add_u32 v23, v22, s53 // (op 2076) 


buffer_store_dwordx2 v[16:17], v23, s[60:63], 0 offen // (op 2076) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_125:


v_mov_b32 v16, v18 // (op 138) reg expression

buffer_load_dwordx2 v[20:21], v16, s[56:59], 0 offen // (op 138) Load value

v_accvgpr_read v24, a0 // (op 134) 
v_accvgpr_read v25, a1 // (op 134) 
v_accvgpr_read v26, a2 // (op 134) 
v_accvgpr_read v27, a3 // (op 134) 
v_mul_f32 v28, v15, v24 // (op 134) 
v_mul_f32 v29, v15, v25 // (op 134) 
v_mul_f32 v30, v15, v26 // (op 134) 
v_mul_f32 v31, v15, v27 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v20 // (op 139) 
v_lshrrev_b32 v17, 16, v20 // (op 139) 
v_cvt_f32_f16 v17, v17 // (op 139) 
v_mul_f32 v24, v14, v16 // (op 139) 
v_mul_f32 v25, v14, v17 // (op 139) 
v_cvt_f32_f16 v16, v21 // (op 139) 
v_lshrrev_b32 v17, 16, v21 // (op 139) 
v_cvt_f32_f16 v17, v17 // (op 139) 
v_mul_f32 v26, v14, v16 // (op 139) 
v_mul_f32 v27, v14, v17 // (op 139) 


v_add_f32 v32, v24, v28 // (op 135) 
v_add_f32 v33, v25, v29 // (op 135) 
v_add_f32 v34, v26, v30 // (op 135) 
v_add_f32 v35, v27, v31 // (op 135) 


v_cvt_f16_f32 v28, v32 // (op 136) 
v_cvt_f16_f32 v29, v33 // (op 136) 
v_cvt_f16_f32 v30, v34 // (op 136) 
v_cvt_f16_f32 v31, v35 // (op 136) 


v_pack_b32_f16 v16, v28, v29 // (op 137) 
v_pack_b32_f16 v17, v30, v31 // (op 137) 

v_mov_b32 v19, v22 // (op 137) reg expression

buffer_store_dwordx2 v[16:17], v19, s[60:63], 0 offen // (op 137) Store value





s_lshl_b32 s52, s4, 6 // (op 2110) 
v_add_u32 v16, v18, s52 // (op 2110) 


buffer_load_dwordx2 v[20:21], v16, s[56:59], 0 offen // (op 2110) Load value

v_accvgpr_read v24, a4 // (op 2106) 
v_accvgpr_read v25, a5 // (op 2106) 
v_accvgpr_read v26, a6 // (op 2106) 
v_accvgpr_read v27, a7 // (op 2106) 
v_mul_f32 v28, v15, v24 // (op 2106) 
v_mul_f32 v29, v15, v25 // (op 2106) 
v_mul_f32 v30, v15, v26 // (op 2106) 
v_mul_f32 v31, v15, v27 // (op 2106) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v20 // (op 2111) 
v_lshrrev_b32 v17, 16, v20 // (op 2111) 
v_cvt_f32_f16 v17, v17 // (op 2111) 
v_mul_f32 v24, v14, v16 // (op 2111) 
v_mul_f32 v25, v14, v17 // (op 2111) 
v_cvt_f32_f16 v16, v21 // (op 2111) 
v_lshrrev_b32 v17, 16, v21 // (op 2111) 
v_cvt_f32_f16 v17, v17 // (op 2111) 
v_mul_f32 v26, v14, v16 // (op 2111) 
v_mul_f32 v27, v14, v17 // (op 2111) 


v_add_f32 v32, v24, v28 // (op 2107) 
v_add_f32 v33, v25, v29 // (op 2107) 
v_add_f32 v34, v26, v30 // (op 2107) 
v_add_f32 v35, v27, v31 // (op 2107) 


v_cvt_f16_f32 v28, v32 // (op 2108) 
v_cvt_f16_f32 v29, v33 // (op 2108) 
v_cvt_f16_f32 v30, v34 // (op 2108) 
v_cvt_f16_f32 v31, v35 // (op 2108) 


v_pack_b32_f16 v16, v28, v29 // (op 2109) 
v_pack_b32_f16 v17, v30, v31 // (op 2109) 

s_lshl_b32 s52, s24, 6 // (op 2109) 
v_add_u32 v19, v22, s52 // (op 2109) 


buffer_store_dwordx2 v[16:17], v19, s[60:63], 0 offen // (op 2109) Store value





s_lshl_b32 s52, s4, 7 // (op 2124) 
v_add_u32 v16, v18, s52 // (op 2124) 


buffer_load_dwordx2 v[20:21], v16, s[56:59], 0 offen // (op 2124) Load value

v_accvgpr_read v24, a8 // (op 2120) 
v_accvgpr_read v25, a9 // (op 2120) 
v_accvgpr_read v26, a10 // (op 2120) 
v_accvgpr_read v27, a11 // (op 2120) 
v_mul_f32 v28, v15, v24 // (op 2120) 
v_mul_f32 v29, v15, v25 // (op 2120) 
v_mul_f32 v30, v15, v26 // (op 2120) 
v_mul_f32 v31, v15, v27 // (op 2120) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v20 // (op 2125) 
v_lshrrev_b32 v17, 16, v20 // (op 2125) 
v_cvt_f32_f16 v17, v17 // (op 2125) 
v_mul_f32 v24, v14, v16 // (op 2125) 
v_mul_f32 v25, v14, v17 // (op 2125) 
v_cvt_f32_f16 v16, v21 // (op 2125) 
v_lshrrev_b32 v17, 16, v21 // (op 2125) 
v_cvt_f32_f16 v17, v17 // (op 2125) 
v_mul_f32 v26, v14, v16 // (op 2125) 
v_mul_f32 v27, v14, v17 // (op 2125) 


v_add_f32 v32, v24, v28 // (op 2121) 
v_add_f32 v33, v25, v29 // (op 2121) 
v_add_f32 v34, v26, v30 // (op 2121) 
v_add_f32 v35, v27, v31 // (op 2121) 


v_cvt_f16_f32 v28, v32 // (op 2122) 
v_cvt_f16_f32 v29, v33 // (op 2122) 
v_cvt_f16_f32 v30, v34 // (op 2122) 
v_cvt_f16_f32 v31, v35 // (op 2122) 


v_pack_b32_f16 v16, v28, v29 // (op 2123) 
v_pack_b32_f16 v17, v30, v31 // (op 2123) 

s_lshl_b32 s52, s24, 7 // (op 2123) 
v_add_u32 v19, v22, s52 // (op 2123) 


buffer_store_dwordx2 v[16:17], v19, s[60:63], 0 offen // (op 2123) Store value





s_lshl_b32 s52, s4, 6 // (op 2138) 
s_mul_i32 s53, 3, s52 // (op 2138) 

v_add_u32 v16, v18, s53 // (op 2138) 


buffer_load_dwordx2 v[20:21], v16, s[56:59], 0 offen // (op 2138) Load value

v_accvgpr_read v24, a12 // (op 2134) 
v_accvgpr_read v25, a13 // (op 2134) 
v_accvgpr_read v26, a14 // (op 2134) 
v_accvgpr_read v27, a15 // (op 2134) 
v_mul_f32 v28, v15, v24 // (op 2134) 
v_mul_f32 v29, v15, v25 // (op 2134) 
v_mul_f32 v30, v15, v26 // (op 2134) 
v_mul_f32 v31, v15, v27 // (op 2134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v20 // (op 2139) 
v_lshrrev_b32 v17, 16, v20 // (op 2139) 
v_cvt_f32_f16 v17, v17 // (op 2139) 
v_mul_f32 v24, v14, v16 // (op 2139) 
v_mul_f32 v25, v14, v17 // (op 2139) 
v_cvt_f32_f16 v16, v21 // (op 2139) 
v_lshrrev_b32 v17, 16, v21 // (op 2139) 
v_cvt_f32_f16 v17, v17 // (op 2139) 
v_mul_f32 v26, v14, v16 // (op 2139) 
v_mul_f32 v27, v14, v17 // (op 2139) 


v_add_f32 v32, v24, v28 // (op 2135) 
v_add_f32 v33, v25, v29 // (op 2135) 
v_add_f32 v34, v26, v30 // (op 2135) 
v_add_f32 v35, v27, v31 // (op 2135) 


v_cvt_f16_f32 v28, v32 // (op 2136) 
v_cvt_f16_f32 v29, v33 // (op 2136) 
v_cvt_f16_f32 v30, v34 // (op 2136) 
v_cvt_f16_f32 v31, v35 // (op 2136) 


v_pack_b32_f16 v16, v28, v29 // (op 2137) 
v_pack_b32_f16 v17, v30, v31 // (op 2137) 

s_lshl_b32 s52, s24, 6 // (op 2137) 
s_mul_i32 s53, 3, s52 // (op 2137) 

v_add_u32 v19, v22, s53 // (op 2137) 


buffer_store_dwordx2 v[16:17], v19, s[60:63], 0 offen // (op 2137) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_125:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_365 // (op 365) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_365
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_365:
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_365:


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
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_SKStreamTileLoop_278 // (op 278) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_SKStreamTileLoop_278 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_278:















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
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_433 // (op 433) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_433 if false)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_DPStreamTileLoop_433:
v_lshrrev_b32 v0, 7, v1 // (op 4306) 
s_mul_i32 s15, s16, s50 // (op 4306) 
s_add_u32 s15, s15, s0 // (op 4306) 
v_and_b32 v3, 63, v1 // (op 4306) 
s_add_u32 s6, s15, s9 // (op 4306) 

v_lshrrev_b32 v6, 4, v3 // (op 4306) 
s_mul_hi_u32 s15, s6, s13 // (op 4306) 
v_lshlrev_b32 v7, 2, v6 // (op 4306) 

s_sub_u32 s7, s6, s15 // (op 4306) 
v_lshl_add_u32 v6, v0, 4, v7 // (op 4306) 


s_lshr_b32 s8, s7, 1 // (op 4306) 

v_lshrrev_b32 v0, 2, v3 // (op 4306) 

s_add_u32 s7, s8, s15 // (op 4306) 


v_and_b32 v3, 3, v0 // (op 4306) 

s_lshr_b32 s8, s7, s11 // (op 4306) 

v_and_b32 v0, 3, v1 // (op 4306) 
s_and_b32 s7, s11, 2147483648 // (op 4306) 
v_lshl_add_u32 v7, v3, 2, v0 // (op 4306) 


s_cmp_eq_i32 s7, 0 // (op 4306) 
s_mov_b32 s15, scc // (op 4306) 

v_lshrrev_b32 v0, 6, v1 // (op 4306) 
s_cmp_eq_i32 s15, 0 // (op 4306) 
s_cselect_b32 s7, s6, s8 // (op 4306) 



v_and_b32 v3, 1, v0 // (op 4306) 

s_mul_hi_u32 s6, s7, s12 // (op 4306) 
v_lshl_add_u32 v0, v3, 4, v7 // (op 4306) 


s_sub_u32 s8, s7, s6 // (op 4306) 
s_lshr_b32 s15, s8, 1 // (op 4306) 

s_add_u32 s8, s15, s6 // (op 4306) 


s_lshr_b32 s6, s8, s10 // (op 4306) 

s_and_b32 s8, s10, 2147483648 // (op 4306) 
s_cmp_eq_i32 s8, 0 // (op 4306) 
s_mov_b32 s15, scc // (op 4306) 

s_cmp_eq_i32 s15, 0 // (op 4306) 
s_cselect_b32 s8, s7, s6 // (op 4306) 


s_mul_i32 s6, s8, s14 // (op 4306) 
v_lshl_add_u32 v3, s8, 5, v6 // (op 4306) 


s_sub_u32 s8, s7, s6 // (op 4306) 


v_lshl_add_u32 v6, s8, 7, v0 // (op 4306) 


v_mul_lo_u32 v0, v6, s4 // (op 4306) 

v_add_lshl_u32 v6, v3, v0, 1 // (op 4306) 


s_mul_i32 s19, s4, s23 // (op 4308) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 4308) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 4308) least significant
s_mul_hi_u32 s6, s4, s22 // (op 4308) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4308) most significant: sum
s_add_u32 s19, s19, s6 // (op 4308) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 4308) 

s_mov_b32 s56, s20 // (op 4308) reg expression
s_mov_b32 s57, s21 // (op 4308) reg expression
s_mov_b32 s58, s28 // (op 4308) reg expression
s_mov_b32 s59, 131072 // (op 4308) call()

v_lshrrev_b32 v0, 7, v1 // (op 4564) 
s_mul_i32 s15, s16, s50 // (op 4564) 
s_add_u32 s15, s15, s0 // (op 4564) 
v_and_b32 v3, 63, v1 // (op 4564) 
s_add_u32 s6, s15, s9 // (op 4564) 

v_lshrrev_b32 v7, 4, v3 // (op 4564) 
s_mul_hi_u32 s15, s6, s13 // (op 4564) 
v_lshlrev_b32 v12, 2, v7 // (op 4564) 

s_sub_u32 s7, s6, s15 // (op 4564) 
v_lshl_add_u32 v7, v0, 4, v12 // (op 4564) 


s_lshr_b32 s8, s7, 1 // (op 4564) 

v_lshrrev_b32 v0, 2, v3 // (op 4564) 

s_add_u32 s7, s8, s15 // (op 4564) 


v_and_b32 v3, 3, v0 // (op 4564) 

s_lshr_b32 s8, s7, s11 // (op 4564) 

v_and_b32 v0, 3, v1 // (op 4564) 
s_and_b32 s7, s11, 2147483648 // (op 4564) 
v_lshl_add_u32 v12, v3, 2, v0 // (op 4564) 


s_cmp_eq_i32 s7, 0 // (op 4564) 
s_mov_b32 s15, scc // (op 4564) 

v_lshrrev_b32 v0, 6, v1 // (op 4564) 
s_cmp_eq_i32 s15, 0 // (op 4564) 
s_cselect_b32 s7, s6, s8 // (op 4564) 



v_and_b32 v3, 1, v0 // (op 4564) 

s_mul_hi_u32 s6, s7, s12 // (op 4564) 
v_lshl_add_u32 v0, v3, 4, v12 // (op 4564) 


s_sub_u32 s8, s7, s6 // (op 4564) 
s_lshr_b32 s15, s8, 1 // (op 4564) 

s_add_u32 s8, s15, s6 // (op 4564) 


s_lshr_b32 s6, s8, s10 // (op 4564) 

s_and_b32 s8, s10, 2147483648 // (op 4564) 
s_cmp_eq_i32 s8, 0 // (op 4564) 
s_mov_b32 s15, scc // (op 4564) 

s_cmp_eq_i32 s15, 0 // (op 4564) 
s_cselect_b32 s8, s7, s6 // (op 4564) 


s_mul_i32 s6, s8, s14 // (op 4564) 
v_lshl_add_u32 v3, s8, 5, v7 // (op 4564) 


s_sub_u32 s8, s7, s6 // (op 4564) 


v_lshl_add_u32 v7, s8, 7, v0 // (op 4564) 


v_mul_lo_u32 v0, v7, s24 // (op 4564) 

v_add_lshl_u32 v7, v3, v0, 1 // (op 4564) 


s_mul_i32 s19, s24, s23 // (op 4566) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 4566) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 4566) least significant
s_mul_hi_u32 s6, s24, s22 // (op 4566) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4566) most significant: sum
s_add_u32 s19, s19, s6 // (op 4566) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 4566) 

s_mov_b32 s60, s26 // (op 4566) reg expression
s_mov_b32 s61, s27 // (op 4566) reg expression
s_mov_b32 s62, s28 // (op 4566) reg expression
s_mov_b32 s63, 131072 // (op 4566) call()

v_lshrrev_b32 v0, 7, v1 // (op 4633) 
v_and_b32 v3, 15, v1 // (op 4633) 
v_lshl_add_u32 v12, v0, 4, v3 // (op 4633) 


v_and_b32 v0, 48, v1 // (op 4633) 
v_lshl_add_u32 v3, v12, 7, v0 // (op 4633) 


v_lshrrev_b32 v0, 6, v1 // (op 4708) 
v_and_b32 v12, 1, v0 // (op 4708) 

v_and_b32 v0, 15, v1 // (op 4708) 
v_lshl_add_u32 v13, v12, 4, v0 // (op 4708) 


v_and_b32 v0, 48, v1 // (op 4708) 
v_lshl_add_u32 v12, v13, 7, v0 // (op 4708) 


v_lshrrev_b32 v0, 6, v1 // (op 4784) 
v_and_b32 v13, 1, v0 // (op 4784) 

v_and_b32 v0, 15, v1 // (op 4784) 
v_lshl_add_u32 v16, v13, 4, v0 // (op 4784) 


v_and_b32 v0, 63, v1 // (op 4784) 
v_lshrrev_b32 v13, 4, v0 // (op 4784) 

v_lshl_add_u32 v0, v16, 2, v13 // (op 4784) 


v_accvgpr_write a0, 0.00000 // (op 174) call()
v_accvgpr_write a1, 0.00000 // (op 174) call()
v_accvgpr_write a2, 0.00000 // (op 174) call()
v_accvgpr_write a3, 0.00000 // (op 174) call()
v_accvgpr_write a4, 0.00000 // (op 586) call()
v_accvgpr_write a5, 0.00000 // (op 586) call()
v_accvgpr_write a6, 0.00000 // (op 586) call()
v_accvgpr_write a7, 0.00000 // (op 586) call()
v_accvgpr_write a8, 0.00000 // (op 742) call()
v_accvgpr_write a9, 0.00000 // (op 742) call()
v_accvgpr_write a10, 0.00000 // (op 742) call()
v_accvgpr_write a11, 0.00000 // (op 742) call()
v_accvgpr_write a12, 0.00000 // (op 898) call()
v_accvgpr_write a13, 0.00000 // (op 898) call()
v_accvgpr_write a14, 0.00000 // (op 898) call()
v_accvgpr_write a15, 0.00000 // (op 898) call()
v_lshrrev_b32 v13, 3, v1 // (op 3808) 
s_mul_i32 s15, s16, s50 // (op 3808) 
s_add_u32 s15, s15, s0 // (op 3808) 
v_and_b32 v16, -8, v1 // (op 3808) 
s_add_u32 s6, s15, s9 // (op 3808) 

v_sub_u32 v17, v1, v16 // (op 3808) 

s_mul_hi_u32 s15, s6, s13 // (op 3808) 
v_lshlrev_b32 v16, 4, v17 // (op 3808) 

s_sub_u32 s7, s6, s15 // (op 3808) 
s_lshr_b32 s8, s7, 1 // (op 3808) 

s_add_u32 s7, s8, s15 // (op 3808) 


s_lshr_b32 s8, s7, s11 // (op 3808) 

s_and_b32 s7, s11, 2147483648 // (op 3808) 
s_cmp_eq_i32 s7, 0 // (op 3808) 
s_mov_b32 s15, scc // (op 3808) 

s_cmp_eq_i32 s15, 0 // (op 3808) 
s_cselect_b32 s7, s6, s8 // (op 3808) 


s_mul_hi_u32 s8, s7, s12 // (op 3808) 
s_sub_u32 s15, s7, s8 // (op 3808) 
s_lshr_b32 s18, s15, 1 // (op 3808) 

s_add_u32 s15, s18, s8 // (op 3808) 


s_lshr_b32 s8, s15, s10 // (op 3808) 

s_and_b32 s15, s10, 2147483648 // (op 3808) 
s_cmp_eq_i32 s15, 0 // (op 3808) 
s_mov_b32 s18, scc // (op 3808) 

s_cmp_eq_i32 s18, 0 // (op 3808) 
s_cselect_b32 s15, s7, s8 // (op 3808) 


v_lshl_add_u32 v17, s15, 5, v13 // (op 3808) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3808) 
s_mul_i32 s8, s7, s18 // (op 3808) 


s_sub_u32 s7, s6, s8 // (op 3808) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 3808) 


v_mov_b32 v19, 0 // (op 3808) convert
v_mov_b32 v18, v13 // (op 3808) convert

v_mul_lo_u32 v21, v17, s31 // (op 3808) most significant: low of low * high
v_mul_hi_u32 v13, v17, s30 // (op 3808) most significant: high of low * low
v_mul_lo_u32 v20, v17, s30 // (op 3808) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 3808) most significant: sum

v_add_co_u32 v20, s[18:19], v20, v18 // (op 3808) least significant half
v_addc_co_u32 v21, s[18:19], v21, v19, s[18:19] // (op 3808) most significant half



s_mul_i32 s19, s30, s35 // (op 3810) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 3810) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 3810) least significant
s_mul_hi_u32 s6, s30, s34 // (op 3810) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3810) most significant: sum
s_add_u32 s19, s19, s6 // (op 3810) most significant: sum


s_mov_b32 s68, s32 // (op 3810) reg expression
s_mov_b32 s69, s33 // (op 3810) reg expression
s_mov_b32 s70, s18 // (op 3810) reg expression
s_mov_b32 s71, 131072 // (op 3810) call()

v_lshrrev_b32 v13, 3, v1 // (op 3827) 
s_mul_i32 s15, s16, s50 // (op 3827) 
s_add_u32 s15, s15, s0 // (op 3827) 
v_and_b32 v16, -8, v1 // (op 3827) 
s_add_u32 s6, s15, s9 // (op 3827) 

v_sub_u32 v17, v1, v16 // (op 3827) 

s_mul_hi_u32 s15, s6, s13 // (op 3827) 
v_lshlrev_b32 v16, 4, v17 // (op 3827) 

s_sub_u32 s7, s6, s15 // (op 3827) 
s_lshr_b32 s8, s7, 1 // (op 3827) 

s_add_u32 s7, s8, s15 // (op 3827) 


s_lshr_b32 s8, s7, s11 // (op 3827) 

s_and_b32 s7, s11, 2147483648 // (op 3827) 
s_cmp_eq_i32 s7, 0 // (op 3827) 
s_mov_b32 s15, scc // (op 3827) 

s_cmp_eq_i32 s15, 0 // (op 3827) 
s_cselect_b32 s7, s6, s8 // (op 3827) 


s_mul_hi_u32 s8, s7, s12 // (op 3827) 
s_sub_u32 s15, s7, s8 // (op 3827) 
s_lshr_b32 s18, s15, 1 // (op 3827) 

s_add_u32 s15, s18, s8 // (op 3827) 


s_lshr_b32 s8, s15, s10 // (op 3827) 

s_and_b32 s15, s10, 2147483648 // (op 3827) 
s_cmp_eq_i32 s15, 0 // (op 3827) 
s_mov_b32 s18, scc // (op 3827) 

s_cmp_eq_i32 s18, 0 // (op 3827) 
s_cselect_b32 s15, s7, s8 // (op 3827) 


v_lshl_add_u32 v17, s15, 5, v13 // (op 3827) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3827) 
s_mul_i32 s8, s7, s18 // (op 3827) 


s_sub_u32 s7, s6, s8 // (op 3827) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 3827) 


v_mov_b32 v19, 0 // (op 3827) convert
v_mov_b32 v18, v13 // (op 3827) convert

v_mul_lo_u32 v23, v17, s31 // (op 3827) most significant: low of low * high
v_mul_hi_u32 v13, v17, s30 // (op 3827) most significant: high of low * low
v_mul_lo_u32 v22, v17, s30 // (op 3827) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 3827) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v18 // (op 3827) least significant half
v_addc_co_u32 v23, s[18:19], v23, v19, s[18:19] // (op 3827) most significant half



s_mul_i32 s19, s30, s35 // (op 3829) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 3829) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 3829) least significant
s_mul_hi_u32 s6, s30, s34 // (op 3829) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3829) most significant: sum
s_add_u32 s19, s19, s6 // (op 3829) most significant: sum


s_mov_b32 s68, s32 // (op 3829) reg expression
s_mov_b32 s69, s33 // (op 3829) reg expression
s_mov_b32 s70, s18 // (op 3829) reg expression
s_mov_b32 s71, 131072 // (op 3829) call()

v_lshrrev_b32 v13, 3, v1 // (op 3942) 
s_mul_i32 s15, s16, s50 // (op 3942) 
s_add_u32 s15, s15, s0 // (op 3942) 
v_and_b32 v16, -8, v1 // (op 3942) 
s_add_u32 s6, s15, s9 // (op 3942) 

v_sub_u32 v17, v1, v16 // (op 3942) 

s_mul_hi_u32 s15, s6, s13 // (op 3942) 
v_lshlrev_b32 v16, 4, v17 // (op 3942) 

s_sub_u32 s7, s6, s15 // (op 3942) 
s_lshr_b32 s8, s7, 1 // (op 3942) 

s_add_u32 s7, s8, s15 // (op 3942) 


s_lshr_b32 s8, s7, s11 // (op 3942) 

s_and_b32 s7, s11, 2147483648 // (op 3942) 
s_cmp_eq_i32 s7, 0 // (op 3942) 
s_mov_b32 s15, scc // (op 3942) 

s_cmp_eq_i32 s15, 0 // (op 3942) 
s_cselect_b32 s7, s6, s8 // (op 3942) 


s_mul_hi_u32 s8, s7, s12 // (op 3942) 
s_sub_u32 s15, s7, s8 // (op 3942) 
s_lshr_b32 s18, s15, 1 // (op 3942) 

s_add_u32 s15, s18, s8 // (op 3942) 


s_lshr_b32 s8, s15, s10 // (op 3942) 

s_and_b32 s15, s10, 2147483648 // (op 3942) 
s_cmp_eq_i32 s15, 0 // (op 3942) 
s_mov_b32 s18, scc // (op 3942) 

s_cmp_eq_i32 s18, 0 // (op 3942) 
s_cselect_b32 s15, s7, s8 // (op 3942) 


s_mul_i32 s8, s15, s14 // (op 3942) 

s_sub_u32 s15, s7, s8 // (op 3942) 

v_lshl_add_u32 v17, s15, 7, v13 // (op 3942) 


v_mul_lo_u32 v19, v17, s37 // (op 3942) most significant: low of low * high
v_mul_hi_u32 v13, v17, s36 // (op 3942) most significant: high of low * low
v_mul_lo_u32 v18, v17, s36 // (op 3942) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 3942) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3942) 
s_mul_i32 s8, s7, s18 // (op 3942) 


s_sub_u32 s7, s6, s8 // (op 3942) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 3942) 


v_mov_b32 v17, 0 // (op 3942) convert
v_mov_b32 v16, v13 // (op 3942) convert

v_add_co_u32 v24, s[18:19], v16, v18 // (op 3942) least significant half
v_addc_co_u32 v25, s[18:19], v17, v19, s[18:19] // (op 3942) most significant half



s_mul_i32 s19, s36, s23 // (op 3944) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 3944) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 3944) least significant
s_mul_hi_u32 s6, s36, s22 // (op 3944) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3944) most significant: sum
s_add_u32 s19, s19, s6 // (op 3944) most significant: sum


s_mov_b32 s72, s38 // (op 3944) reg expression
s_mov_b32 s73, s39 // (op 3944) reg expression
s_mov_b32 s74, s18 // (op 3944) reg expression
s_mov_b32 s75, 131072 // (op 3944) call()

v_lshrrev_b32 v13, 3, v1 // (op 3961) 
s_mul_i32 s15, s16, s50 // (op 3961) 
s_add_u32 s15, s15, s0 // (op 3961) 
v_and_b32 v16, -8, v1 // (op 3961) 
s_add_u32 s6, s15, s9 // (op 3961) 

v_sub_u32 v17, v1, v16 // (op 3961) 

s_mul_hi_u32 s15, s6, s13 // (op 3961) 
v_lshlrev_b32 v16, 4, v17 // (op 3961) 

s_sub_u32 s7, s6, s15 // (op 3961) 
s_lshr_b32 s8, s7, 1 // (op 3961) 

s_add_u32 s7, s8, s15 // (op 3961) 


s_lshr_b32 s8, s7, s11 // (op 3961) 

s_and_b32 s7, s11, 2147483648 // (op 3961) 
s_cmp_eq_i32 s7, 0 // (op 3961) 
s_mov_b32 s15, scc // (op 3961) 

s_cmp_eq_i32 s15, 0 // (op 3961) 
s_cselect_b32 s7, s6, s8 // (op 3961) 


s_mul_hi_u32 s8, s7, s12 // (op 3961) 
s_sub_u32 s15, s7, s8 // (op 3961) 
s_lshr_b32 s18, s15, 1 // (op 3961) 

s_add_u32 s15, s18, s8 // (op 3961) 


s_lshr_b32 s8, s15, s10 // (op 3961) 

s_and_b32 s15, s10, 2147483648 // (op 3961) 
s_cmp_eq_i32 s15, 0 // (op 3961) 
s_mov_b32 s18, scc // (op 3961) 

s_cmp_eq_i32 s18, 0 // (op 3961) 
s_cselect_b32 s15, s7, s8 // (op 3961) 


s_mul_i32 s8, s15, s14 // (op 3961) 

s_sub_u32 s15, s7, s8 // (op 3961) 

v_lshl_add_u32 v17, s15, 7, v13 // (op 3961) 


v_mul_lo_u32 v19, v17, s37 // (op 3961) most significant: low of low * high
v_mul_hi_u32 v13, v17, s36 // (op 3961) most significant: high of low * low
v_mul_lo_u32 v18, v17, s36 // (op 3961) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 3961) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3961) 
s_mul_i32 s8, s7, s18 // (op 3961) 


s_sub_u32 s7, s6, s8 // (op 3961) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 3961) 


v_mov_b32 v17, 0 // (op 3961) convert
v_mov_b32 v16, v13 // (op 3961) convert

v_add_co_u32 v26, s[18:19], v16, v18 // (op 3961) least significant half
v_addc_co_u32 v27, s[18:19], v17, v19, s[18:19] // (op 3961) most significant half



s_mul_i32 s19, s36, s23 // (op 3963) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 3963) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 3963) least significant
s_mul_hi_u32 s6, s36, s22 // (op 3963) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3963) most significant: sum
s_add_u32 s19, s19, s6 // (op 3963) most significant: sum


s_mov_b32 s72, s38 // (op 3963) reg expression
s_mov_b32 s73, s39 // (op 3963) reg expression
s_mov_b32 s74, s18 // (op 3963) reg expression
s_mov_b32 s75, 131072 // (op 3963) call()

v_lshrrev_b32 v13, 7, v1 // (op 4070) 
s_mul_i32 s15, s16, s50 // (op 4070) 
s_add_u32 s15, s15, s0 // (op 4070) 
v_and_b32 v16, 15, v1 // (op 4070) 
s_add_u32 s6, s15, s9 // (op 4070) 

v_lshl_add_u32 v17, v13, 4, v16 // (op 4070) 


s_mul_hi_u32 s15, s6, s13 // (op 4070) 
v_and_b32 v13, 63, v1 // (op 4070) 
s_sub_u32 s7, s6, s15 // (op 4070) 
v_lshrrev_b32 v16, 4, v13 // (op 4070) 

s_lshr_b32 s8, s7, 1 // (op 4070) 

s_add_u32 s7, s8, s15 // (op 4070) 


s_lshr_b32 s8, s7, s11 // (op 4070) 

s_and_b32 s7, s11, 2147483648 // (op 4070) 
s_cmp_eq_i32 s7, 0 // (op 4070) 
s_mov_b32 s15, scc // (op 4070) 

s_cmp_eq_i32 s15, 0 // (op 4070) 
s_cselect_b32 s7, s6, s8 // (op 4070) 


s_mul_hi_u32 s8, s7, s12 // (op 4070) 
s_sub_u32 s15, s7, s8 // (op 4070) 
s_lshr_b32 s18, s15, 1 // (op 4070) 

s_add_u32 s15, s18, s8 // (op 4070) 


s_lshr_b32 s8, s15, s10 // (op 4070) 

s_and_b32 s15, s10, 2147483648 // (op 4070) 
s_cmp_eq_i32 s15, 0 // (op 4070) 
s_mov_b32 s18, scc // (op 4070) 

s_cmp_eq_i32 s18, 0 // (op 4070) 
s_cselect_b32 s15, s7, s8 // (op 4070) 


v_lshl_add_u32 v13, s15, 5, v17 // (op 4070) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4070) 
s_mul_i32 s8, s7, s18 // (op 4070) 


s_sub_u32 s7, s6, s8 // (op 4070) 


v_lshl_add_u32 v17, s7, 2, v16 // (op 4070) 


v_mul_lo_u32 v16, v13, s40 // (op 4070) 
v_add_u32 v16, v16, v17 // (op 4070) 


s_mul_i32 s19, s40, s35 // (op 4072) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 4072) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 4072) least significant
s_mul_hi_u32 s6, s40, s34 // (op 4072) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4072) most significant: sum
s_add_u32 s19, s19, s6 // (op 4072) most significant: sum


s_mov_b32 s76, s42 // (op 4072) reg expression
s_mov_b32 s77, s43 // (op 4072) reg expression
s_mov_b32 s78, s18 // (op 4072) reg expression
s_mov_b32 s79, 131072 // (op 4072) call()

v_lshrrev_b32 v13, 7, v1 // (op 4089) 
s_mul_i32 s15, s16, s50 // (op 4089) 
s_add_u32 s15, s15, s0 // (op 4089) 
v_and_b32 v17, 15, v1 // (op 4089) 
s_add_u32 s6, s15, s9 // (op 4089) 

v_lshl_add_u32 v18, v13, 4, v17 // (op 4089) 


s_mul_hi_u32 s15, s6, s13 // (op 4089) 
v_and_b32 v13, 63, v1 // (op 4089) 
s_sub_u32 s7, s6, s15 // (op 4089) 
v_lshrrev_b32 v17, 4, v13 // (op 4089) 

s_lshr_b32 s8, s7, 1 // (op 4089) 

s_add_u32 s7, s8, s15 // (op 4089) 


s_lshr_b32 s8, s7, s11 // (op 4089) 

s_and_b32 s7, s11, 2147483648 // (op 4089) 
s_cmp_eq_i32 s7, 0 // (op 4089) 
s_mov_b32 s15, scc // (op 4089) 

s_cmp_eq_i32 s15, 0 // (op 4089) 
s_cselect_b32 s7, s6, s8 // (op 4089) 


s_mul_hi_u32 s8, s7, s12 // (op 4089) 
s_sub_u32 s15, s7, s8 // (op 4089) 
s_lshr_b32 s18, s15, 1 // (op 4089) 

s_add_u32 s15, s18, s8 // (op 4089) 


s_lshr_b32 s8, s15, s10 // (op 4089) 

s_and_b32 s15, s10, 2147483648 // (op 4089) 
s_cmp_eq_i32 s15, 0 // (op 4089) 
s_mov_b32 s18, scc // (op 4089) 

s_cmp_eq_i32 s18, 0 // (op 4089) 
s_cselect_b32 s15, s7, s8 // (op 4089) 


v_lshl_add_u32 v13, s15, 5, v18 // (op 4089) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4089) 
s_mul_i32 s8, s7, s18 // (op 4089) 


s_sub_u32 s7, s6, s8 // (op 4089) 


v_lshl_add_u32 v18, s7, 2, v17 // (op 4089) 


v_mul_lo_u32 v17, v13, s40 // (op 4089) 
v_add_u32 v17, v17, v18 // (op 4089) 


s_mul_i32 s19, s40, s35 // (op 4091) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 4091) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 4091) least significant
s_mul_hi_u32 s6, s40, s34 // (op 4091) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4091) most significant: sum
s_add_u32 s19, s19, s6 // (op 4091) most significant: sum


s_mov_b32 s76, s42 // (op 4091) reg expression
s_mov_b32 s77, s43 // (op 4091) reg expression
s_mov_b32 s78, s18 // (op 4091) reg expression
s_mov_b32 s79, 131072 // (op 4091) call()

v_lshrrev_b32 v13, 2, v1 // (op 4198) 
s_mul_i32 s15, s16, s50 // (op 4198) 
s_add_u32 s15, s15, s0 // (op 4198) 
v_lshlrev_b32 v18, 1, v13 // (op 4198) 

s_add_u32 s6, s15, s9 // (op 4198) 

v_and_b32 v13, -4, v1 // (op 4198) 
s_mul_hi_u32 s15, s6, s13 // (op 4198) 
v_sub_u32 v19, v1, v13 // (op 4198) 

s_sub_u32 s7, s6, s15 // (op 4198) 
s_lshr_b32 s8, s7, 1 // (op 4198) 

s_add_u32 s7, s8, s15 // (op 4198) 


s_lshr_b32 s8, s7, s11 // (op 4198) 

s_and_b32 s7, s11, 2147483648 // (op 4198) 
s_cmp_eq_i32 s7, 0 // (op 4198) 
s_mov_b32 s15, scc // (op 4198) 

s_cmp_eq_i32 s15, 0 // (op 4198) 
s_cselect_b32 s7, s6, s8 // (op 4198) 


s_mul_hi_u32 s8, s7, s12 // (op 4198) 
s_sub_u32 s15, s7, s8 // (op 4198) 
s_lshr_b32 s18, s15, 1 // (op 4198) 

s_add_u32 s15, s18, s8 // (op 4198) 


s_lshr_b32 s8, s15, s10 // (op 4198) 

s_and_b32 s15, s10, 2147483648 // (op 4198) 
s_cmp_eq_i32 s15, 0 // (op 4198) 
s_mov_b32 s18, scc // (op 4198) 

s_cmp_eq_i32 s18, 0 // (op 4198) 
s_cselect_b32 s15, s7, s8 // (op 4198) 


s_mul_i32 s8, s15, s14 // (op 4198) 

s_sub_u32 s15, s7, s8 // (op 4198) 

v_lshl_add_u32 v13, s15, 7, v18 // (op 4198) 


v_mul_lo_u32 v18, v13, s44 // (op 4198) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 4198) 
s_mul_i32 s8, s7, s18 // (op 4198) 


s_sub_u32 s7, s6, s8 // (op 4198) 


v_lshl_add_u32 v13, s7, 2, v19 // (op 4198) 


v_add_u32 v19, v13, v18 // (op 4198) 


s_mul_i32 s19, s44, s23 // (op 4200) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 4200) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 4200) least significant
s_mul_hi_u32 s6, s44, s22 // (op 4200) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4200) most significant: sum
s_add_u32 s19, s19, s6 // (op 4200) most significant: sum


s_mov_b32 s80, s46 // (op 4200) reg expression
s_mov_b32 s81, s47 // (op 4200) reg expression
s_mov_b32 s82, s18 // (op 4200) reg expression
s_mov_b32 s83, 131072 // (op 4200) call()

v_lshrrev_b32 v13, 2, v1 // (op 4217) 
s_mul_i32 s15, s16, s50 // (op 4217) 
s_add_u32 s15, s15, s0 // (op 4217) 
v_lshlrev_b32 v18, 1, v13 // (op 4217) 

s_add_u32 s6, s15, s9 // (op 4217) 

v_and_b32 v13, -4, v1 // (op 4217) 
s_mul_hi_u32 s15, s6, s13 // (op 4217) 
v_sub_u32 v28, v1, v13 // (op 4217) 

s_sub_u32 s7, s6, s15 // (op 4217) 
s_lshr_b32 s8, s7, 1 // (op 4217) 

s_add_u32 s7, s8, s15 // (op 4217) 


s_lshr_b32 s8, s7, s11 // (op 4217) 

s_and_b32 s7, s11, 2147483648 // (op 4217) 
s_cmp_eq_i32 s7, 0 // (op 4217) 
s_mov_b32 s15, scc // (op 4217) 

s_cmp_eq_i32 s15, 0 // (op 4217) 
s_cselect_b32 s7, s6, s8 // (op 4217) 


s_mul_hi_u32 s8, s7, s12 // (op 4217) 
s_sub_u32 s15, s7, s8 // (op 4217) 
s_lshr_b32 s18, s15, 1 // (op 4217) 

s_add_u32 s15, s18, s8 // (op 4217) 


s_lshr_b32 s8, s15, s10 // (op 4217) 

s_and_b32 s15, s10, 2147483648 // (op 4217) 
s_cmp_eq_i32 s15, 0 // (op 4217) 
s_mov_b32 s18, scc // (op 4217) 

s_cmp_eq_i32 s18, 0 // (op 4217) 
s_cselect_b32 s15, s7, s8 // (op 4217) 


s_mul_i32 s8, s15, s14 // (op 4217) 

s_sub_u32 s15, s7, s8 // (op 4217) 

v_lshl_add_u32 v13, s15, 7, v18 // (op 4217) 


v_mul_lo_u32 v18, v13, s44 // (op 4217) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 4217) 
s_mul_i32 s8, s7, s18 // (op 4217) 


s_sub_u32 s7, s6, s8 // (op 4217) 


v_lshl_add_u32 v13, s7, 2, v28 // (op 4217) 


v_add_u32 v28, v13, v18 // (op 4217) 


s_mul_i32 s19, s44, s23 // (op 4219) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 4219) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 4219) least significant
s_mul_hi_u32 s6, s44, s22 // (op 4219) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4219) most significant: sum
s_add_u32 s19, s19, s6 // (op 4219) most significant: sum


s_mov_b32 s80, s46 // (op 4219) reg expression
s_mov_b32 s81, s47 // (op 4219) reg expression
s_mov_b32 s82, s18 // (op 4219) reg expression
s_mov_b32 s83, 131072 // (op 4219) call()




buffer_load_ubyte v13, v16, s[76:79], 0 offen offset: 0 // (op 3342) Load value
v_and_b32 v18, -8, v1 // (op 4674) 
v_sub_u32 v29, v1, v18 // (op 4674) 

v_lshlrev_b32 v18, 4, v29 // (op 4674) 

v_lshrrev_b32 v29, 3, v1 // (op 4674) 
v_lshl_add_u32 v30, v29, 7, v18 // (op 4674) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 4674) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5221) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 3716) 
v_mov_b64 v[30:31], s[18:19] // (op 3716) convert


v_mov_b64 v[32:33], v[24:25] // (op 3716) 
s_mov_b32 m0, s15 // (op 3716) reg expression
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 3716) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 3716) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 3716) most significant half

s_add_u32 m0, m0, 4096 // (op 3716) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 3716) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 3716) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 3716) most significant half

s_add_u32 m0, m0, 4096 // (op 3716) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 3716) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 3716) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 3716) most significant half

s_add_u32 m0, m0, 4096 // (op 3716) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 3716) Load value direct to lds




v_mov_b64 v[30:31], s[44:45] // (op 3349) convert

v_mov_b32 v18, v19 // (op 3349) 
buffer_load_ubyte v32, v18, s[80:83], 0 offen offset: 0 // (op 3349) Load value
v_add_u32 v18, v18, v30 // (op 3349) 
buffer_load_ubyte v33, v18, s[80:83], 0 offen offset: 0 // (op 3349) Load value


v_and_b32 v18, -8, v1 // (op 4609) 
v_sub_u32 v29, v1, v18 // (op 4609) 

v_lshlrev_b32 v18, 4, v29 // (op 4609) 

v_lshrrev_b32 v29, 3, v1 // (op 4609) 
v_lshl_add_u32 v30, v29, 7, v18 // (op 4609) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 4609) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5202) 


s_nop 0
s_add_u32 s6, 16384, s15 // (op 3721) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s6 // (op 3721) reg expression
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 3721) Load value direct to lds





buffer_load_ubyte v18, v16, s[76:79], 0 offen offset:4 // (op 3356) Load value

v_and_b32 v16, -8, v1 // (op 4893) 
v_sub_u32 v29, v1, v16 // (op 4893) 

v_lshlrev_b32 v16, 4, v29 // (op 4893) 

v_lshrrev_b32 v29, 3, v1 // (op 4893) 
v_lshl_add_u32 v30, v29, 7, v16 // (op 4893) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 4893) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5269) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 3726) 
v_mov_b64 v[30:31], s[18:19] // (op 3726) convert


v_mov_b64 v[34:35], v[24:25] // (op 3726) 
s_add_u32 s6, 20480, s15 // (op 3726) 
s_mov_b32 m0, s6 // (op 3726) reg expression
s_mov_b32 s7, 128 // (op 3726) call()
buffer_load_dwordx4 v34, s[72:75], s7 offen offset: 0 lds // (op 3726) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v30 // (op 3726) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 3726) most significant half

s_add_u32 m0, m0, 4096 // (op 3726) 
buffer_load_dwordx4 v34, s[72:75], s7 offen offset: 0 lds // (op 3726) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v30 // (op 3726) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 3726) most significant half

s_add_u32 m0, m0, 4096 // (op 3726) 
buffer_load_dwordx4 v34, s[72:75], s7 offen offset: 0 lds // (op 3726) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v30 // (op 3726) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 3726) most significant half

s_add_u32 m0, m0, 4096 // (op 3726) 
buffer_load_dwordx4 v34, s[72:75], s7 offen offset: 0 lds // (op 3726) Load value direct to lds







v_mov_b64 v[24:25], s[44:45] // (op 3363) convert

v_mov_b32 v16, v19 // (op 3363) 
buffer_load_ubyte v30, v16, s[80:83], 0 offen offset:4 // (op 3363) Load value
v_add_u32 v16, v16, v24 // (op 3363) 
buffer_load_ubyte v31, v16, s[80:83], 0 offen offset:4 // (op 3363) Load value



v_and_b32 v16, -8, v1 // (op 4854) 
v_sub_u32 v19, v1, v16 // (op 4854) 

v_lshlrev_b32 v16, 4, v19 // (op 4854) 

v_lshrrev_b32 v19, 3, v1 // (op 4854) 
v_lshl_add_u32 v29, v19, 7, v16 // (op 4854) 


s_nop 0
v_readfirstlane_b32 s15, v29 // (op 4854) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5260) 


s_nop 0
s_add_u32 s6, 36864, s15 // (op 3728) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s6 // (op 3728) reg expression
s_mov_b32 s7, 128 // (op 3728) call()
buffer_load_dwordx4 v20, s[68:71], s7 offen offset: 0 lds // (op 3728) Load value direct to lds




v_and_b32 v16, -4, v1 // (op 4750) 
v_sub_u32 v29, v1, v16 // (op 4750) 

v_lshrrev_b32 v16, 2, v1 // (op 4750) 
v_lshl_add_u32 v19, v16, 3, v29 // (op 4750) 


s_barrier  // (op 5240) 



s_waitcnt vmcnt(10)
ds_write_b8 v19, v32 offset:40960 // (op 3369) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v19, v33 offset:40964 // (op 3369) Store local data 

s_barrier  // (op 5218) 



ds_read_b128 v[34:37], v12 // (op 3387) Load local data 
ds_read_b128 v[38:41], v12 offset:64 // (op 3387) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 5237) 



ds_read_u8 v16, v0 offset:40960 // (op 3407) Load local data 



ds_read_b128 v[42:45], v12 offset:4096 // (op 3392) Load local data 
ds_read_b128 v[46:49], v12 offset:4160 // (op 3392) Load local data 



ds_read_u8 v29, v0 offset:41088 // (op 3412) Load local data 



ds_read_b128 v[50:53], v12 offset:8192 // (op 3397) Load local data 
ds_read_b128 v[54:57], v12 offset:8256 // (op 3397) Load local data 



ds_read_u8 v19, v0 offset:41216 // (op 3417) Load local data 



ds_read_b128 v[58:61], v12 offset:12288 // (op 3402) Load local data 
ds_read_b128 v[62:65], v12 offset:12352 // (op 3402) Load local data 



ds_read_u8 v20, v0 offset:41344 // (op 3422) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 5199) 



ds_read_b128 v[66:69], v3 offset:16384 // (op 3383) Load local data 
ds_read_b128 v[70:73], v3 offset:16448 // (op 3383) Load local data 
v_and_b32 v21, -8, v1 // (op 4581) 
v_sub_u32 v24, v1, v21 // (op 4581) 

v_lshlrev_b32 v21, 4, v24 // (op 4581) 

v_lshrrev_b32 v24, 3, v1 // (op 4581) 
v_lshl_add_u32 v25, v24, 7, v21 // (op 4581) 


s_nop 0
v_readfirstlane_b32 s15, v25 // (op 4581) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -8, v1 // (op 4646) 
v_sub_u32 v24, v1, v21 // (op 4646) 

v_lshlrev_b32 v21, 4, v24 // (op 4646) 

v_lshrrev_b32 v24, 3, v1 // (op 4646) 
v_lshl_add_u32 v25, v24, 7, v21 // (op 4646) 


s_nop 0
v_readfirstlane_b32 s6, v25 // (op 4646) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -4, v1 // (op 4721) 
v_sub_u32 v24, v1, v21 // (op 4721) 

v_lshrrev_b32 v21, 2, v1 // (op 4721) 
v_lshl_add_u32 v25, v21, 3, v24 // (op 4721) 


v_and_b32 v21, -8, v1 // (op 4839) 
v_sub_u32 v24, v1, v21 // (op 4839) 

v_lshlrev_b32 v21, 4, v24 // (op 4839) 

v_lshrrev_b32 v24, 3, v1 // (op 4839) 
v_lshl_add_u32 v74, v24, 7, v21 // (op 4839) 


s_nop 0
v_readfirstlane_b32 s7, v74 // (op 4839) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -8, v1 // (op 4878) 
v_sub_u32 v24, v1, v21 // (op 4878) 

v_lshlrev_b32 v21, 4, v24 // (op 4878) 

v_lshrrev_b32 v24, 3, v1 // (op 4878) 
v_lshl_add_u32 v74, v24, 7, v21 // (op 4878) 


s_nop 0
v_readfirstlane_b32 s8, v74 // (op 4878) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -4, v1 // (op 4922) 
v_sub_u32 v24, v1, v21 // (op 4922) 

v_lshrrev_b32 v21, 2, v1 // (op 4922) 
v_lshl_add_u32 v74, v21, 3, v24 // (op 4922) 


s_mov_b32 s1, 0 // (op 900) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 899) 
s_mov_b32 s28, s18 // (op 899) convert

s_and_b32 s29, s28, 4294967294 // (op 899) 

s_cmp_lt_u32 s1, s29 // (op 899) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_899 // (op 899) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_899 if false)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_899:
s_add_u32 s1, s1, 2 // (op 901) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:73], v[34:41], a[0:3], v13, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 188) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:73], v[42:49], a[4:7], v13, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 623) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:73], v[50:57], a[8:11], v13, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 779) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:73], v[58:65], a[12:15], v13, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 935) 


v_mov_b32 v21, v17 // (op 185) 

buffer_load_ubyte v13, v21, s[76:79], 0 offen offset:8 // (op 185) Load value
s_barrier  // (op 5224) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 3665) 
v_mov_b64 v[76:77], s[18:19] // (op 3665) convert


v_mov_b64 v[78:79], v[26:27] // (op 3665) 
v_mov_b64 v[80:81], v[78:79] // (op 3665) 
s_mov_b32 m0, s6 // (op 3665) reg expression
s_mov_b32 s18, 256 // (op 3665) call()
buffer_load_dwordx4 v80, s[72:75], s18 offen offset: 0 lds // (op 3665) Load value direct to lds
v_add_co_u32 v80, s[28:29], v80, v76 // (op 3665) least significant half
v_addc_co_u32 v81, s[28:29], 0, v81, s[28:29] // (op 3665) most significant half

s_add_u32 m0, m0, 4096 // (op 3665) 
buffer_load_dwordx4 v80, s[72:75], s18 offen offset: 0 lds // (op 3665) Load value direct to lds
v_add_co_u32 v80, s[28:29], v80, v76 // (op 3665) least significant half
v_addc_co_u32 v81, s[28:29], 0, v81, s[28:29] // (op 3665) most significant half

s_add_u32 m0, m0, 4096 // (op 3665) 
buffer_load_dwordx4 v80, s[72:75], s18 offen offset: 0 lds // (op 3665) Load value direct to lds
v_add_co_u32 v80, s[28:29], v80, v76 // (op 3665) least significant half
v_addc_co_u32 v81, s[28:29], 0, v81, s[28:29] // (op 3665) most significant half

s_add_u32 m0, m0, 4096 // (op 3665) 
buffer_load_dwordx4 v80, s[72:75], s18 offen offset: 0 lds // (op 3665) Load value direct to lds




v_mov_b64 v[76:77], s[44:45] // (op 192) convert

v_mov_b32 v24, v28 // (op 192) 
v_mov_b32 v75, v24 // (op 192) 
buffer_load_ubyte v32, v75, s[80:83], 0 offen offset:8 // (op 192) Load value
v_add_u32 v75, v75, v76 // (op 192) 
buffer_load_ubyte v33, v75, s[80:83], 0 offen offset:8 // (op 192) Load value


s_barrier  // (op 5205) 


v_mov_b64 v[76:77], v[22:23] // (op 3660) 
s_add_u32 s18, 16384, s15 // (op 3660) 

s_mov_b32 m0, s18 // (op 3660) reg expression
s_mov_b32 s19, 256 // (op 3660) call()
buffer_load_dwordx4 v76, s[68:71], s19 offen offset: 0 lds // (op 3660) Load value direct to lds


s_barrier  // (op 5278) 



s_waitcnt vmcnt(10)
ds_write_b8 v74, v30 offset:41472 // (op 516) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v74, v31 offset:41476 // (op 516) Store local data 
s_barrier  // (op 5266) 



ds_read_b128 v[34:37], v11 offset:20480 // (op 509) Load local data 
ds_read_b128 v[38:41], v11 offset:20544 // (op 509) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 5275) 



ds_read_u8 v16, v10 offset:41472 // (op 513) Load local data 



ds_read_b128 v[42:45], v11 offset:24576 // (op 628) Load local data 
ds_read_b128 v[46:49], v11 offset:24640 // (op 628) Load local data 



ds_read_u8 v29, v10 offset:41600 // (op 631) Load local data 



ds_read_b128 v[50:53], v11 offset:28672 // (op 784) Load local data 
ds_read_b128 v[54:57], v11 offset:28736 // (op 784) Load local data 



ds_read_u8 v19, v10 offset:41728 // (op 787) Load local data 



ds_read_b128 v[58:61], v11 offset:32768 // (op 940) Load local data 
ds_read_b128 v[62:65], v11 offset:32832 // (op 940) Load local data 



ds_read_u8 v20, v10 offset:41856 // (op 943) Load local data 
s_barrier  // (op 5405) 
s_waitcnt vmcnt(8)
s_barrier  // (op 5257) 



ds_read_b128 v[66:69], v5 offset:36864 // (op 507) Load local data 
ds_read_b128 v[70:73], v5 offset:36928 // (op 507) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:73], v[34:41], a[0:3], v18, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 514) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:73], v[42:49], a[4:7], v18, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 624) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:73], v[50:57], a[8:11], v18, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 780) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:73], v[58:65], a[12:15], v18, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 936) 



buffer_load_ubyte v18, v21, s[76:79], 0 offen offset:12 // (op 511) Load value

s_barrier  // (op 5272) 
v_add_u32 v17, 8, v17 // (op 4081) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 3688) 
v_mov_b64 v[80:81], s[18:19] // (op 3688) convert


v_mov_b64 v[82:83], v[78:79] // (op 3688) 
s_add_u32 s18, 20480, s8 // (op 3688) 
s_mov_b32 m0, s18 // (op 3688) reg expression
s_mov_b32 s19, 384 // (op 3688) call()
buffer_load_dwordx4 v82, s[72:75], s19 offen offset: 0 lds // (op 3688) Load value direct to lds
v_add_co_u32 v82, s[28:29], v82, v80 // (op 3688) least significant half
v_addc_co_u32 v83, s[28:29], 0, v83, s[28:29] // (op 3688) most significant half

s_add_u32 m0, m0, 4096 // (op 3688) 
buffer_load_dwordx4 v82, s[72:75], s19 offen offset: 0 lds // (op 3688) Load value direct to lds
v_add_co_u32 v82, s[28:29], v82, v80 // (op 3688) least significant half
v_addc_co_u32 v83, s[28:29], 0, v83, s[28:29] // (op 3688) most significant half

s_add_u32 m0, m0, 4096 // (op 3688) 
buffer_load_dwordx4 v82, s[72:75], s19 offen offset: 0 lds // (op 3688) Load value direct to lds
v_add_co_u32 v82, s[28:29], v82, v80 // (op 3688) least significant half
v_addc_co_u32 v83, s[28:29], 0, v83, s[28:29] // (op 3688) most significant half

s_add_u32 m0, m0, 4096 // (op 3688) 
buffer_load_dwordx4 v82, s[72:75], s19 offen offset: 0 lds // (op 3688) Load value direct to lds






v_add_co_u32 v26, vcc, 256, v26 // (op 3953) least significant half
v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 3953) most significant half
v_mov_b64 v[78:79], s[44:45] // (op 518) convert

v_mov_b32 v21, v24 // (op 518) 
buffer_load_ubyte v30, v21, s[80:83], 0 offen offset:12 // (op 518) Load value
v_add_u32 v21, v21, v78 // (op 518) 
buffer_load_ubyte v31, v21, s[80:83], 0 offen offset:12 // (op 518) Load value



s_barrier  // (op 5263) 
v_add_u32 v28, 8, v28 // (op 4209) 


s_add_u32 s18, 36864, s7 // (op 3683) 

s_mov_b32 m0, s18 // (op 3683) reg expression
s_mov_b32 s19, 384 // (op 3683) call()
buffer_load_dwordx4 v76, s[68:71], s19 offen offset: 0 lds // (op 3683) Load value direct to lds



s_barrier  // (op 5243) 
v_add_co_u32 v22, vcc, 256, v22 // (op 3819) least significant half



v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 3819) most significant half
s_waitcnt vmcnt(10)
ds_write_b8 v25, v32 offset:40960 // (op 190) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v25, v33 offset:40964 // (op 190) Store local data 
s_barrier  // (op 5230) 



ds_read_b128 v[34:37], v9 // (op 183) Load local data 
ds_read_b128 v[38:41], v9 offset:64 // (op 183) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 5250) 



ds_read_u8 v16, v8 offset:40960 // (op 187) Load local data 



ds_read_b128 v[42:45], v9 offset:4096 // (op 619) Load local data 
ds_read_b128 v[46:49], v9 offset:4160 // (op 619) Load local data 



ds_read_u8 v29, v8 offset:41088 // (op 622) Load local data 



ds_read_b128 v[50:53], v9 offset:8192 // (op 775) Load local data 
ds_read_b128 v[54:57], v9 offset:8256 // (op 775) Load local data 



ds_read_u8 v19, v8 offset:41216 // (op 778) Load local data 



ds_read_b128 v[58:61], v9 offset:12288 // (op 931) Load local data 
ds_read_b128 v[62:65], v9 offset:12352 // (op 931) Load local data 



ds_read_u8 v20, v8 offset:41344 // (op 934) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 5211) 



ds_read_b128 v[66:69], v4 offset:16384 // (op 181) Load local data 
ds_read_b128 v[70:73], v4 offset:16448 // (op 181) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 899) 
s_mov_b32 s28, s18 // (op 899) convert

s_and_b32 s29, s28, 4294967294 // (op 899) 

s_cmp_lt_u32 s1, s29 // (op 899) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_899 // (op 899) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_899 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_899:












s_barrier  // (op 5389) 
v_lshrrev_b32 v13, 3, v1 // (op 3755) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 3755) 
v_and_b32 v17, -8, v1 // (op 3755) 
s_and_b32 s15, s18, 4294967294 // (op 3755) 
v_sub_u32 v21, v1, v17 // (op 3755) 

s_add_u32 s6, s0, s15 // (op 3755) 

v_lshlrev_b32 v17, 4, v21 // (op 3755) 

s_mul_i32 s15, s16, s50 // (op 3755) 
s_add_u32 s15, s15, s6 // (op 3755) 

s_add_u32 s6, s15, s9 // (op 3755) 

s_mul_hi_u32 s15, s6, s13 // (op 3755) 
s_sub_u32 s7, s6, s15 // (op 3755) 
s_lshr_b32 s8, s7, 1 // (op 3755) 

s_add_u32 s7, s8, s15 // (op 3755) 


s_lshr_b32 s8, s7, s11 // (op 3755) 

s_and_b32 s7, s11, 2147483648 // (op 3755) 
s_cmp_eq_i32 s7, 0 // (op 3755) 
s_mov_b32 s15, scc // (op 3755) 

s_cmp_eq_i32 s15, 0 // (op 3755) 
s_cselect_b32 s7, s6, s8 // (op 3755) 


s_mul_hi_u32 s8, s7, s12 // (op 3755) 
s_sub_u32 s15, s7, s8 // (op 3755) 
s_lshr_b32 s28, s15, 1 // (op 3755) 

s_add_u32 s15, s28, s8 // (op 3755) 


s_lshr_b32 s8, s15, s10 // (op 3755) 

s_and_b32 s15, s10, 2147483648 // (op 3755) 
s_cmp_eq_i32 s15, 0 // (op 3755) 
s_mov_b32 s28, scc // (op 3755) 

s_cmp_eq_i32 s28, 0 // (op 3755) 
s_cselect_b32 s15, s7, s8 // (op 3755) 


s_mul_i32 s8, s7, s18 // (op 3755) 


v_lshl_add_u32 v21, s15, 5, v13 // (op 3755) 


s_sub_u32 s7, s6, s8 // (op 3755) 


v_lshl_add_u32 v13, s7, 7, v17 // (op 3755) 


v_mov_b32 v33, 0 // (op 3755) convert
v_mov_b32 v32, v13 // (op 3755) convert

v_mul_lo_u32 v23, v21, s31 // (op 3755) most significant: low of low * high
v_mul_hi_u32 v13, v21, s30 // (op 3755) most significant: high of low * low
v_mul_lo_u32 v22, v21, s30 // (op 3755) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 3755) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v32 // (op 3755) least significant half
v_addc_co_u32 v23, s[18:19], v23, v33, s[18:19] // (op 3755) most significant half



s_mul_i32 s19, s30, s35 // (op 3757) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 3757) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 3757) least significant
s_mul_hi_u32 s6, s30, s34 // (op 3757) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3757) most significant: sum
s_add_u32 s19, s19, s6 // (op 3757) most significant: sum


s_mov_b32 s68, s32 // (op 3757) reg expression
s_mov_b32 s69, s33 // (op 3757) reg expression
s_mov_b32 s70, s18 // (op 3757) reg expression
s_mov_b32 s71, 131072 // (op 3757) call()

v_lshrrev_b32 v13, 3, v1 // (op 3897) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 3897) 
v_and_b32 v17, -8, v1 // (op 3897) 
s_and_b32 s15, s18, 4294967294 // (op 3897) 
v_sub_u32 v21, v1, v17 // (op 3897) 

s_add_u32 s6, s0, s15 // (op 3897) 

v_lshlrev_b32 v17, 4, v21 // (op 3897) 

s_mul_i32 s15, s16, s50 // (op 3897) 
s_add_u32 s15, s15, s6 // (op 3897) 

s_add_u32 s6, s15, s9 // (op 3897) 

s_mul_hi_u32 s15, s6, s13 // (op 3897) 
s_sub_u32 s7, s6, s15 // (op 3897) 
s_lshr_b32 s8, s7, 1 // (op 3897) 

s_add_u32 s7, s8, s15 // (op 3897) 


s_lshr_b32 s8, s7, s11 // (op 3897) 

s_and_b32 s7, s11, 2147483648 // (op 3897) 
s_cmp_eq_i32 s7, 0 // (op 3897) 
s_mov_b32 s15, scc // (op 3897) 

s_cmp_eq_i32 s15, 0 // (op 3897) 
s_cselect_b32 s7, s6, s8 // (op 3897) 


s_mul_hi_u32 s8, s7, s12 // (op 3897) 
s_sub_u32 s15, s7, s8 // (op 3897) 
s_lshr_b32 s28, s15, 1 // (op 3897) 

s_add_u32 s15, s28, s8 // (op 3897) 


s_lshr_b32 s8, s15, s10 // (op 3897) 

s_and_b32 s15, s10, 2147483648 // (op 3897) 
s_cmp_eq_i32 s15, 0 // (op 3897) 
s_mov_b32 s28, scc // (op 3897) 

s_cmp_eq_i32 s28, 0 // (op 3897) 
s_cselect_b32 s15, s7, s8 // (op 3897) 


s_mul_i32 s8, s15, s14 // (op 3897) 

s_sub_u32 s15, s7, s8 // (op 3897) 

s_mul_i32 s8, s7, s18 // (op 3897) 


v_lshl_add_u32 v21, s15, 7, v13 // (op 3897) 


s_sub_u32 s7, s6, s8 // (op 3897) 


v_mul_lo_u32 v33, v21, s37 // (op 3897) most significant: low of low * high
v_mul_hi_u32 v13, v21, s36 // (op 3897) most significant: high of low * low
v_mul_lo_u32 v32, v21, s36 // (op 3897) least significant: low of low * low
v_add_u32 v33, v33, v13 // (op 3897) most significant: sum


v_lshl_add_u32 v13, s7, 7, v17 // (op 3897) 


v_mov_b32 v25, 0 // (op 3897) convert
v_mov_b32 v24, v13 // (op 3897) convert

v_add_co_u32 v26, s[18:19], v24, v32 // (op 3897) least significant half
v_addc_co_u32 v27, s[18:19], v25, v33, s[18:19] // (op 3897) most significant half



s_mul_i32 s19, s36, s23 // (op 3899) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 3899) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 3899) least significant
s_mul_hi_u32 s6, s36, s22 // (op 3899) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3899) most significant: sum
s_add_u32 s19, s19, s6 // (op 3899) most significant: sum


s_mov_b32 s72, s38 // (op 3899) reg expression
s_mov_b32 s73, s39 // (op 3899) reg expression
s_mov_b32 s74, s18 // (op 3899) reg expression
s_mov_b32 s75, 131072 // (op 3899) call()

v_lshrrev_b32 v13, 7, v1 // (op 4025) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4025) 
v_and_b32 v17, 15, v1 // (op 4025) 
s_and_b32 s15, s18, 4294967294 // (op 4025) 
v_lshl_add_u32 v21, v13, 4, v17 // (op 4025) 


s_add_u32 s6, s0, s15 // (op 4025) 

v_and_b32 v13, 63, v1 // (op 4025) 
v_lshrrev_b32 v17, 4, v13 // (op 4025) 

s_mul_i32 s15, s16, s50 // (op 4025) 
s_add_u32 s15, s15, s6 // (op 4025) 

s_add_u32 s6, s15, s9 // (op 4025) 

s_mul_hi_u32 s15, s6, s13 // (op 4025) 
s_sub_u32 s7, s6, s15 // (op 4025) 
s_lshr_b32 s8, s7, 1 // (op 4025) 

s_add_u32 s7, s8, s15 // (op 4025) 


s_lshr_b32 s8, s7, s11 // (op 4025) 

s_and_b32 s7, s11, 2147483648 // (op 4025) 
s_cmp_eq_i32 s7, 0 // (op 4025) 
s_mov_b32 s15, scc // (op 4025) 

s_cmp_eq_i32 s15, 0 // (op 4025) 
s_cselect_b32 s7, s6, s8 // (op 4025) 


s_mul_hi_u32 s8, s7, s12 // (op 4025) 
s_sub_u32 s15, s7, s8 // (op 4025) 
s_lshr_b32 s28, s15, 1 // (op 4025) 

s_add_u32 s15, s28, s8 // (op 4025) 


s_lshr_b32 s8, s15, s10 // (op 4025) 

s_and_b32 s15, s10, 2147483648 // (op 4025) 
s_cmp_eq_i32 s15, 0 // (op 4025) 
s_mov_b32 s28, scc // (op 4025) 

s_cmp_eq_i32 s28, 0 // (op 4025) 
s_cselect_b32 s15, s7, s8 // (op 4025) 


s_mul_i32 s8, s7, s18 // (op 4025) 


v_lshl_add_u32 v13, s15, 5, v21 // (op 4025) 


s_sub_u32 s7, s6, s8 // (op 4025) 


v_lshl_add_u32 v21, s7, 2, v17 // (op 4025) 


v_mul_lo_u32 v17, v13, s40 // (op 4025) 
v_add_u32 v17, v17, v21 // (op 4025) 


s_mul_i32 s19, s40, s35 // (op 4027) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 4027) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 4027) least significant
s_mul_hi_u32 s6, s40, s34 // (op 4027) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4027) most significant: sum
s_add_u32 s19, s19, s6 // (op 4027) most significant: sum


s_mov_b32 s76, s42 // (op 4027) reg expression
s_mov_b32 s77, s43 // (op 4027) reg expression
s_mov_b32 s78, s18 // (op 4027) reg expression
s_mov_b32 s79, 131072 // (op 4027) call()

v_lshrrev_b32 v13, 2, v1 // (op 4153) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4153) 
v_lshlrev_b32 v21, 1, v13 // (op 4153) 

s_and_b32 s15, s18, 4294967294 // (op 4153) 
v_and_b32 v13, -4, v1 // (op 4153) 
s_add_u32 s6, s0, s15 // (op 4153) 

v_sub_u32 v28, v1, v13 // (op 4153) 

s_mul_i32 s15, s16, s50 // (op 4153) 
s_add_u32 s15, s15, s6 // (op 4153) 

s_add_u32 s6, s15, s9 // (op 4153) 

s_mul_hi_u32 s15, s6, s13 // (op 4153) 
s_sub_u32 s7, s6, s15 // (op 4153) 
s_lshr_b32 s8, s7, 1 // (op 4153) 

s_add_u32 s7, s8, s15 // (op 4153) 


s_lshr_b32 s8, s7, s11 // (op 4153) 

s_and_b32 s7, s11, 2147483648 // (op 4153) 
s_cmp_eq_i32 s7, 0 // (op 4153) 
s_mov_b32 s15, scc // (op 4153) 

s_cmp_eq_i32 s15, 0 // (op 4153) 
s_cselect_b32 s7, s6, s8 // (op 4153) 


s_mul_hi_u32 s8, s7, s12 // (op 4153) 
s_sub_u32 s15, s7, s8 // (op 4153) 
s_lshr_b32 s28, s15, 1 // (op 4153) 

s_add_u32 s15, s28, s8 // (op 4153) 


s_lshr_b32 s8, s15, s10 // (op 4153) 

s_and_b32 s15, s10, 2147483648 // (op 4153) 
s_cmp_eq_i32 s15, 0 // (op 4153) 
s_mov_b32 s28, scc // (op 4153) 

s_cmp_eq_i32 s28, 0 // (op 4153) 
s_cselect_b32 s15, s7, s8 // (op 4153) 


s_mul_i32 s8, s15, s14 // (op 4153) 

s_sub_u32 s15, s7, s8 // (op 4153) 

v_lshl_add_u32 v13, s15, 7, v21 // (op 4153) 


s_mul_i32 s8, s7, s18 // (op 4153) 


v_mul_lo_u32 v21, v13, s44 // (op 4153) 

s_sub_u32 s7, s6, s8 // (op 4153) 


v_lshl_add_u32 v13, s7, 2, v28 // (op 4153) 


v_add_u32 v28, v13, v21 // (op 4153) 


s_mul_i32 s19, s44, s23 // (op 4155) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 4155) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 4155) least significant
s_mul_hi_u32 s6, s44, s22 // (op 4155) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4155) most significant: sum
s_add_u32 s19, s19, s6 // (op 4155) most significant: sum


s_mov_b32 s80, s46 // (op 4155) reg expression
s_mov_b32 s81, s47 // (op 4155) reg expression
s_mov_b32 s82, s18 // (op 4155) reg expression
s_mov_b32 s83, 131072 // (op 4155) call()

v_and_b32 v13, -8, v1 // (op 4594) 
v_sub_u32 v21, v1, v13 // (op 4594) 

v_lshlrev_b32 v13, 4, v21 // (op 4594) 

v_lshrrev_b32 v21, 3, v1 // (op 4594) 
v_lshl_add_u32 v24, v21, 7, v13 // (op 4594) 


s_nop 0
v_readfirstlane_b32 s15, v24 // (op 4594) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -8, v1 // (op 4659) 
v_sub_u32 v21, v1, v13 // (op 4659) 

v_lshlrev_b32 v13, 4, v21 // (op 4659) 

v_lshrrev_b32 v21, 3, v1 // (op 4659) 
v_lshl_add_u32 v24, v21, 7, v13 // (op 4659) 


s_nop 0
v_readfirstlane_b32 s6, v24 // (op 4659) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -4, v1 // (op 4734) 
v_sub_u32 v21, v1, v13 // (op 4734) 

v_lshrrev_b32 v13, 2, v1 // (op 4734) 
v_lshl_add_u32 v24, v13, 3, v21 // (op 4734) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 905) 
s_and_b32 s7, s18, 4294967294 // (op 905) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 904) 
s_mov_b32 s8, s18 // (op 904) convert

s_cmp_lt_u32 s7, s8 // (op 904) 












s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_904 // (op 904) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_904 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_904:
s_barrier  // (op 5208) 
s_add_u32 s7, s7, 1 // (op 906) 


v_mov_b64 v[18:19], v[22:23] // (op 3670) 
s_add_u32 s8, 16384, s15 // (op 3670) 

s_mov_b32 m0, s8 // (op 3670) reg expression
buffer_load_dwordx4 v18, s[68:71], 0 offen offset: 0 lds // (op 3670) Load value direct to lds


v_add_co_u32 v22, vcc, 128, v22 // (op 3745) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 3745) most significant half
s_barrier  // (op 5227) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 3676) 
v_mov_b64 v[18:19], s[18:19] // (op 3676) convert


v_mov_b64 v[20:21], v[26:27] // (op 3676) 
v_mov_b64 v[30:31], v[20:21] // (op 3676) 
s_mov_b32 m0, s6 // (op 3676) reg expression
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 3676) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v18 // (op 3676) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 3676) most significant half

s_add_u32 m0, m0, 4096 // (op 3676) 
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 3676) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v18 // (op 3676) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 3676) most significant half

s_add_u32 m0, m0, 4096 // (op 3676) 
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 3676) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v18 // (op 3676) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 3676) most significant half

s_add_u32 m0, m0, 4096 // (op 3676) 
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 3676) Load value direct to lds



v_add_co_u32 v26, vcc, 128, v26 // (op 3889) least significant half
v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 3889) most significant half

v_mov_b64 v[18:19], s[44:45] // (op 460) convert

v_mov_b32 v13, v28 // (op 460) 
v_mov_b32 v16, v13 // (op 460) 
buffer_load_ubyte v20, v16, s[80:83], 0 offen offset: 0 // (op 460) Load value
v_add_u32 v16, v16, v18 // (op 460) 
buffer_load_ubyte v21, v16, s[80:83], 0 offen offset: 0 // (op 460) Load value



v_add_u32 v28, 4, v28 // (op 4145) 
s_barrier  // (op 5246) 
s_waitcnt vmcnt(6)
s_barrier  // (op 5214) 






ds_read_b128 v[30:33], v4 offset:16384 // (op 449) Load local data 
ds_read_b128 v[34:37], v4 offset:16448 // (op 449) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v20 offset:40960 // (op 458) Store local data 
s_barrier  // (op 5233) 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v21 offset:40964 // (op 458) Store local data 




ds_read_b128 v[38:41], v9 // (op 451) Load local data 
ds_read_b128 v[42:45], v9 offset:64 // (op 451) Load local data 


v_mov_b32 v13, v17 // (op 453) 

buffer_load_ubyte v16, v13, s[76:79], 0 offen offset: 0 // (op 453) Load value

v_add_u32 v17, 4, v17 // (op 4017) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 5253) 



ds_read_u8 v13, v8 offset:40960 // (op 455) Load local data 



ds_read_b128 v[46:49], v9 offset:4096 // (op 600) Load local data 
ds_read_b128 v[50:53], v9 offset:4160 // (op 600) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:45], a[0:3], v16, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 456) 





ds_read_u8 v13, v8 offset:41088 // (op 603) Load local data 



ds_read_b128 v[38:41], v9 offset:8192 // (op 756) Load local data 
ds_read_b128 v[42:45], v9 offset:8256 // (op 756) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:37], v[46:53], a[4:7], v16, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 604) 





ds_read_u8 v13, v8 offset:41216 // (op 759) Load local data 



ds_read_b128 v[46:49], v9 offset:12288 // (op 912) Load local data 
ds_read_b128 v[50:53], v9 offset:12352 // (op 912) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[30:37], v[38:45], a[8:11], v16, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 760) 





ds_read_u8 v13, v8 offset:41344 // (op 915) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[30:37], v[46:53], a[12:15], v16, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 916) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 904) 
s_mov_b32 s8, s18 // (op 904) convert

s_cmp_lt_u32 s7, s8 // (op 904) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_904 // (op 904) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_904 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_904:








s_barrier  // (op 5393) 



v_cmp_eq_f32 vcc, 0.00000, v14 // (op 222) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_222 // (op 222) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_222
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

v_mov_b32 v0, v7 // (op 233) reg expression

buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 233) Store value



v_accvgpr_read v16, a4 // (op 1140) 
v_accvgpr_read v17, a5 // (op 1140) 
v_accvgpr_read v18, a6 // (op 1140) 
v_accvgpr_read v19, a7 // (op 1140) 
v_mul_f32 v20, v15, v16 // (op 1140) 
v_mul_f32 v21, v15, v17 // (op 1140) 
v_mul_f32 v22, v15, v18 // (op 1140) 
v_mul_f32 v23, v15, v19 // (op 1140) 

v_cvt_f16_f32 v16, v20 // (op 1142) 
v_cvt_f16_f32 v17, v21 // (op 1142) 
v_cvt_f16_f32 v18, v22 // (op 1142) 
v_cvt_f16_f32 v19, v23 // (op 1142) 


v_pack_b32_f16 v12, v16, v17 // (op 1143) 
v_pack_b32_f16 v13, v18, v19 // (op 1143) 

s_lshl_b32 s15, s24, 6 // (op 1143) 
v_add_u32 v0, v7, s15 // (op 1143) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 1143) Store value



v_accvgpr_read v16, a8 // (op 1154) 
v_accvgpr_read v17, a9 // (op 1154) 
v_accvgpr_read v18, a10 // (op 1154) 
v_accvgpr_read v19, a11 // (op 1154) 
v_mul_f32 v20, v15, v16 // (op 1154) 
v_mul_f32 v21, v15, v17 // (op 1154) 
v_mul_f32 v22, v15, v18 // (op 1154) 
v_mul_f32 v23, v15, v19 // (op 1154) 

v_cvt_f16_f32 v16, v20 // (op 1156) 
v_cvt_f16_f32 v17, v21 // (op 1156) 
v_cvt_f16_f32 v18, v22 // (op 1156) 
v_cvt_f16_f32 v19, v23 // (op 1156) 


v_pack_b32_f16 v12, v16, v17 // (op 1157) 
v_pack_b32_f16 v13, v18, v19 // (op 1157) 

s_lshl_b32 s15, s24, 7 // (op 1157) 
v_add_u32 v0, v7, s15 // (op 1157) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 1157) Store value



v_accvgpr_read v16, a12 // (op 1168) 
v_accvgpr_read v17, a13 // (op 1168) 
v_accvgpr_read v18, a14 // (op 1168) 
v_accvgpr_read v19, a15 // (op 1168) 
v_mul_f32 v20, v15, v16 // (op 1168) 
v_mul_f32 v21, v15, v17 // (op 1168) 
v_mul_f32 v22, v15, v18 // (op 1168) 
v_mul_f32 v23, v15, v19 // (op 1168) 

v_cvt_f16_f32 v16, v20 // (op 1170) 
v_cvt_f16_f32 v17, v21 // (op 1170) 
v_cvt_f16_f32 v18, v22 // (op 1170) 
v_cvt_f16_f32 v19, v23 // (op 1170) 


v_pack_b32_f16 v12, v16, v17 // (op 1171) 
v_pack_b32_f16 v13, v18, v19 // (op 1171) 

s_lshl_b32 s15, s24, 6 // (op 1171) 
s_mul_i32 s6, 3, s15 // (op 1171) 

v_add_u32 v0, v7, s6 // (op 1171) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 1171) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_222 // (op 222) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_222
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_222:


v_mov_b32 v0, v6 // (op 247) reg expression

buffer_load_dwordx2 v[12:13], v0, s[56:59], 0 offen // (op 247) Load value

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

v_mov_b32 v0, v7 // (op 246) reg expression

buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 246) Store value





s_lshl_b32 s15, s4, 6 // (op 1205) 
v_add_u32 v0, v6, s15 // (op 1205) 


buffer_load_dwordx2 v[12:13], v0, s[56:59], 0 offen // (op 1205) Load value

v_accvgpr_read v16, a4 // (op 1201) 
v_accvgpr_read v17, a5 // (op 1201) 
v_accvgpr_read v18, a6 // (op 1201) 
v_accvgpr_read v19, a7 // (op 1201) 
v_mul_f32 v20, v15, v16 // (op 1201) 
v_mul_f32 v21, v15, v17 // (op 1201) 
v_mul_f32 v22, v15, v18 // (op 1201) 
v_mul_f32 v23, v15, v19 // (op 1201) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v12 // (op 1206) 
v_lshrrev_b32 v17, 16, v12 // (op 1206) 
v_cvt_f32_f16 v17, v17 // (op 1206) 
v_mul_f32 v24, v14, v16 // (op 1206) 
v_mul_f32 v25, v14, v17 // (op 1206) 
v_cvt_f32_f16 v16, v13 // (op 1206) 
v_lshrrev_b32 v17, 16, v13 // (op 1206) 
v_cvt_f32_f16 v17, v17 // (op 1206) 
v_mul_f32 v26, v14, v16 // (op 1206) 
v_mul_f32 v27, v14, v17 // (op 1206) 


v_add_f32 v16, v24, v20 // (op 1202) 
v_add_f32 v17, v25, v21 // (op 1202) 
v_add_f32 v18, v26, v22 // (op 1202) 
v_add_f32 v19, v27, v23 // (op 1202) 


v_cvt_f16_f32 v20, v16 // (op 1203) 
v_cvt_f16_f32 v21, v17 // (op 1203) 
v_cvt_f16_f32 v22, v18 // (op 1203) 
v_cvt_f16_f32 v23, v19 // (op 1203) 


v_pack_b32_f16 v12, v20, v21 // (op 1204) 
v_pack_b32_f16 v13, v22, v23 // (op 1204) 

s_lshl_b32 s15, s24, 6 // (op 1204) 
v_add_u32 v0, v7, s15 // (op 1204) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 1204) Store value





s_lshl_b32 s15, s4, 7 // (op 1219) 
v_add_u32 v0, v6, s15 // (op 1219) 


buffer_load_dwordx2 v[12:13], v0, s[56:59], 0 offen // (op 1219) Load value

v_accvgpr_read v16, a8 // (op 1215) 
v_accvgpr_read v17, a9 // (op 1215) 
v_accvgpr_read v18, a10 // (op 1215) 
v_accvgpr_read v19, a11 // (op 1215) 
v_mul_f32 v20, v15, v16 // (op 1215) 
v_mul_f32 v21, v15, v17 // (op 1215) 
v_mul_f32 v22, v15, v18 // (op 1215) 
v_mul_f32 v23, v15, v19 // (op 1215) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v12 // (op 1220) 
v_lshrrev_b32 v17, 16, v12 // (op 1220) 
v_cvt_f32_f16 v17, v17 // (op 1220) 
v_mul_f32 v24, v14, v16 // (op 1220) 
v_mul_f32 v25, v14, v17 // (op 1220) 
v_cvt_f32_f16 v16, v13 // (op 1220) 
v_lshrrev_b32 v17, 16, v13 // (op 1220) 
v_cvt_f32_f16 v17, v17 // (op 1220) 
v_mul_f32 v26, v14, v16 // (op 1220) 
v_mul_f32 v27, v14, v17 // (op 1220) 


v_add_f32 v16, v24, v20 // (op 1216) 
v_add_f32 v17, v25, v21 // (op 1216) 
v_add_f32 v18, v26, v22 // (op 1216) 
v_add_f32 v19, v27, v23 // (op 1216) 


v_cvt_f16_f32 v20, v16 // (op 1217) 
v_cvt_f16_f32 v21, v17 // (op 1217) 
v_cvt_f16_f32 v22, v18 // (op 1217) 
v_cvt_f16_f32 v23, v19 // (op 1217) 


v_pack_b32_f16 v12, v20, v21 // (op 1218) 
v_pack_b32_f16 v13, v22, v23 // (op 1218) 

s_lshl_b32 s15, s24, 7 // (op 1218) 
v_add_u32 v0, v7, s15 // (op 1218) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 1218) Store value





s_lshl_b32 s15, s4, 6 // (op 1233) 
s_mul_i32 s6, 3, s15 // (op 1233) 

v_add_u32 v0, v6, s6 // (op 1233) 


buffer_load_dwordx2 v[12:13], v0, s[56:59], 0 offen // (op 1233) Load value

v_accvgpr_read v16, a12 // (op 1229) 
v_accvgpr_read v17, a13 // (op 1229) 
v_accvgpr_read v18, a14 // (op 1229) 
v_accvgpr_read v19, a15 // (op 1229) 
v_mul_f32 v20, v15, v16 // (op 1229) 
v_mul_f32 v21, v15, v17 // (op 1229) 
v_mul_f32 v22, v15, v18 // (op 1229) 
v_mul_f32 v23, v15, v19 // (op 1229) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v12 // (op 1234) 
v_lshrrev_b32 v17, 16, v12 // (op 1234) 
v_cvt_f32_f16 v17, v17 // (op 1234) 
v_mul_f32 v24, v14, v16 // (op 1234) 
v_mul_f32 v25, v14, v17 // (op 1234) 
v_cvt_f32_f16 v16, v13 // (op 1234) 
v_lshrrev_b32 v17, 16, v13 // (op 1234) 
v_cvt_f32_f16 v17, v17 // (op 1234) 
v_mul_f32 v26, v14, v16 // (op 1234) 
v_mul_f32 v27, v14, v17 // (op 1234) 


v_add_f32 v16, v24, v20 // (op 1230) 
v_add_f32 v17, v25, v21 // (op 1230) 
v_add_f32 v18, v26, v22 // (op 1230) 
v_add_f32 v19, v27, v23 // (op 1230) 


v_cvt_f16_f32 v20, v16 // (op 1231) 
v_cvt_f16_f32 v21, v17 // (op 1231) 
v_cvt_f16_f32 v22, v18 // (op 1231) 
v_cvt_f16_f32 v23, v19 // (op 1231) 


v_pack_b32_f16 v12, v20, v21 // (op 1232) 
v_pack_b32_f16 v13, v22, v23 // (op 1232) 

s_lshl_b32 s15, s24, 6 // (op 1232) 
s_mul_i32 s6, 3, s15 // (op 1232) 

v_add_u32 v0, v7, s6 // (op 1232) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 1232) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_222:
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
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_DPStreamTileLoop_433 // (op 433) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_DPStreamTileLoop_433 if true)
RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_433:











































s_endpgm  // End of RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
.LRR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_end:
.size RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2, .LRR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_end-RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
  .amdhsa_next_free_vgpr 108
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 41984 // lds bytes
  .amdhsa_accum_offset 92
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
    macrotile: [32, 128, 128]
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
  MIWaveTile: [1, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
    .symbol: RR_GEMM_TN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 41984
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 88
    .vgpr_count: 90
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

