// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2
.p2align 8
.type RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2,@function
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 2635) 
v_and_b32 v3, 15, v1 // (op 2635) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2635) 


v_and_b32 v0, 48, v1 // (op 2635) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 2635) 


v_lshrrev_b32 v0, 7, v1 // (op 2893) 
v_and_b32 v4, 15, v1 // (op 2893) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 2893) 


v_and_b32 v0, 48, v1 // (op 2893) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 2893) 


v_lshrrev_b32 v0, 7, v1 // (op 3138) 
v_and_b32 v5, 15, v1 // (op 3138) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 3138) 


v_and_b32 v0, 48, v1 // (op 3138) 
v_lshl_add_u32 v5, v6, 7, v0 // (op 3138) 


v_lshrrev_b32 v0, 7, v1 // (op 3250) 
v_and_b32 v6, 15, v1 // (op 3250) 
v_lshl_add_u32 v7, v0, 4, v6 // (op 3250) 


v_and_b32 v0, 48, v1 // (op 3250) 
v_lshl_add_u32 v6, v7, 7, v0 // (op 3250) 


v_and_b32 v0, 48, v1 // (op 2705) 
v_and_b32 v7, 63, v1 // (op 2705) 
v_sub_u32 v8, v7, v0 // (op 2705) 

v_lshrrev_b32 v7, 1, v8 // (op 2705) 
v_add_lshl_u32 v9, v0, v7, 6 // (op 2705) 


v_and_b32 v0, 1, v8 // (op 2705) 

v_lshlrev_b32 v7, 3, v0 // (op 2705) 

v_lshrrev_b32 v0, 6, v1 // (op 2705) 
v_and_b32 v8, 1, v0 // (op 2705) 

v_lshl_add_u32 v0, v8, 4, v7 // (op 2705) 


v_add_u32 v7, v9, v0 // (op 2705) 


v_lshrrev_b32 v0, 6, v1 // (op 2781) 
v_and_b32 v8, 1, v0 // (op 2781) 

v_and_b32 v0, 15, v1 // (op 2781) 
v_lshl_add_u32 v9, v8, 4, v0 // (op 2781) 


v_and_b32 v0, 63, v1 // (op 2781) 
v_lshrrev_b32 v8, 4, v0 // (op 2781) 

v_lshl_add_u32 v0, v8, 6, v9 // (op 2781) 


v_and_b32 v8, 48, v1 // (op 2963) 
v_and_b32 v9, 63, v1 // (op 2963) 
v_sub_u32 v10, v9, v8 // (op 2963) 

v_lshrrev_b32 v9, 1, v10 // (op 2963) 
v_add_lshl_u32 v11, v8, v9, 6 // (op 2963) 


v_and_b32 v8, 1, v10 // (op 2963) 

v_lshlrev_b32 v9, 3, v8 // (op 2963) 

v_lshrrev_b32 v8, 6, v1 // (op 2963) 
v_and_b32 v10, 1, v8 // (op 2963) 

v_lshl_add_u32 v8, v10, 4, v9 // (op 2963) 


v_add_u32 v9, v11, v8 // (op 2963) 


v_lshrrev_b32 v8, 6, v1 // (op 3039) 
v_and_b32 v10, 1, v8 // (op 3039) 

v_and_b32 v8, 15, v1 // (op 3039) 
v_lshl_add_u32 v11, v10, 4, v8 // (op 3039) 


v_and_b32 v8, 63, v1 // (op 3039) 
v_lshrrev_b32 v10, 4, v8 // (op 3039) 

v_lshl_add_u32 v8, v10, 6, v11 // (op 3039) 


v_and_b32 v10, 48, v1 // (op 3182) 
v_and_b32 v11, 63, v1 // (op 3182) 
v_sub_u32 v12, v11, v10 // (op 3182) 

v_lshrrev_b32 v11, 1, v12 // (op 3182) 
v_add_lshl_u32 v13, v10, v11, 6 // (op 3182) 


v_and_b32 v10, 1, v12 // (op 3182) 

v_lshlrev_b32 v11, 3, v10 // (op 3182) 

v_lshrrev_b32 v10, 6, v1 // (op 3182) 
v_and_b32 v12, 1, v10 // (op 3182) 

v_lshl_add_u32 v10, v12, 4, v11 // (op 3182) 


v_add_u32 v11, v13, v10 // (op 3182) 


v_lshrrev_b32 v10, 6, v1 // (op 3211) 
v_and_b32 v12, 1, v10 // (op 3211) 

v_and_b32 v10, 15, v1 // (op 3211) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 3211) 


v_and_b32 v10, 63, v1 // (op 3211) 
v_lshrrev_b32 v12, 4, v10 // (op 3211) 

v_lshl_add_u32 v10, v12, 6, v13 // (op 3211) 


v_and_b32 v12, 48, v1 // (op 3294) 
v_and_b32 v13, 63, v1 // (op 3294) 
v_sub_u32 v14, v13, v12 // (op 3294) 

v_lshrrev_b32 v13, 1, v14 // (op 3294) 
v_add_lshl_u32 v15, v12, v13, 6 // (op 3294) 


v_and_b32 v12, 1, v14 // (op 3294) 

v_lshlrev_b32 v13, 3, v12 // (op 3294) 

v_lshrrev_b32 v12, 6, v1 // (op 3294) 
v_and_b32 v14, 1, v12 // (op 3294) 

v_lshl_add_u32 v12, v14, 4, v13 // (op 3294) 


v_add_u32 v13, v15, v12 // (op 3294) 


v_lshrrev_b32 v12, 6, v1 // (op 3323) 
v_and_b32 v14, 1, v12 // (op 3323) 

v_and_b32 v12, 15, v1 // (op 3323) 
v_lshl_add_u32 v15, v14, 4, v12 // (op 3323) 


v_and_b32 v12, 63, v1 // (op 3323) 
v_lshrrev_b32 v14, 4, v12 // (op 3323) 

v_lshl_add_u32 v12, v14, 6, v15 // (op 3323) 


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
s_mul_i32 s54, s8, s16 // (op 278) 
s_cmp_gt_u32 s9, s54 // (op 278) 
s_mov_b32 s55, scc // (op 278) 
s_sub_u32 s56, s9, s54 // (op 278) 

s_cmp_eq_i32 s55, 0 // (op 278) 
s_cselect_b32 s54, 0, s56 // (op 278) 


s_cmp_lt_u32 s8, s54 // (op 278) 
s_mov_b32 s55, scc // (op 278) 
s_cmp_eq_i32 s55, 0 // (op 278) 
s_cselect_b32 s56, s54, s8 // (op 278) 


s_cmp_lt_u32 s0, s56 // (op 278) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_278 // (op 278) Condition: Top (jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_278 if false)
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_SKStreamTileLoop_278:
v_lshrrev_b32 v16, 7, v1 // (op 2556) 
s_mul_i32 s54, s16, s8 // (op 2556) 
s_add_u32 s54, s54, s0 // (op 2556) 
v_and_b32 v17, 63, v1 // (op 2556) 
s_mul_hi_u32 s55, s54, s13 // (op 2556) 
v_lshrrev_b32 v18, 4, v17 // (op 2556) 
s_sub_u32 s56, s54, s55 // (op 2556) 
v_lshlrev_b32 v19, 2, v18 // (op 2556) 

v_lshl_add_u32 v18, v16, 4, v19 // (op 2556) 


s_lshr_b32 s57, s56, 1 // (op 2556) 

s_add_u32 s56, s57, s55 // (op 2556) 


v_lshrrev_b32 v16, 2, v17 // (op 2556) 

s_lshr_b32 s55, s56, s11 // (op 2556) 

v_and_b32 v17, 3, v16 // (op 2556) 

s_and_b32 s56, s11, 2147483648 // (op 2556) 
v_and_b32 v16, 3, v1 // (op 2556) 
s_cmp_eq_i32 s56, 0 // (op 2556) 
s_mov_b32 s57, scc // (op 2556) 

v_lshl_add_u32 v19, v17, 2, v16 // (op 2556) 


s_cmp_eq_i32 s57, 0 // (op 2556) 
s_cselect_b32 s56, s54, s55 // (op 2556) 



v_lshrrev_b32 v16, 6, v1 // (op 2556) 
s_mul_hi_u32 s54, s56, s12 // (op 2556) 
v_and_b32 v17, 1, v16 // (op 2556) 

s_sub_u32 s55, s56, s54 // (op 2556) 
v_lshl_add_u32 v16, v17, 4, v19 // (op 2556) 


s_lshr_b32 s57, s55, 1 // (op 2556) 

s_add_u32 s55, s57, s54 // (op 2556) 


s_lshr_b32 s54, s55, s10 // (op 2556) 

s_and_b32 s55, s10, 2147483648 // (op 2556) 
s_cmp_eq_i32 s55, 0 // (op 2556) 
s_mov_b32 s57, scc // (op 2556) 

s_cmp_eq_i32 s57, 0 // (op 2556) 
s_cselect_b32 s55, s56, s54 // (op 2556) 


s_mul_i32 s54, s55, s14 // (op 2556) 
v_lshl_add_u32 v17, s55, 5, v18 // (op 2556) 


s_sub_u32 s55, s56, s54 // (op 2556) 


v_lshl_add_u32 v18, s55, 6, v16 // (op 2556) 


v_mul_lo_u32 v16, v18, s4 // (op 2556) 

v_add_lshl_u32 v18, v17, v16, 1 // (op 2556) 


s_mul_i32 s55, s4, s23 // (op 2558) most significant: low of low * high
s_mul_i32 s56, s5, s22 // (op 2558) most significant: low of high * low
s_mul_i32 s54, s4, s22 // (op 2558) least significant
s_mul_hi_u32 s57, s4, s22 // (op 2558) most significant: high of low * low
s_add_u32 s55, s55, s56 // (op 2558) most significant: sum
s_add_u32 s55, s55, s57 // (op 2558) most significant: sum


s_lshl_b64 s[56:57], s[54:55], 1 // (op 2558) 

s_mov_b32 s60, s20 // (op 2558) reg expression
s_mov_b32 s61, s21 // (op 2558) reg expression
s_mov_b32 s62, s56 // (op 2558) reg expression
s_mov_b32 s63, 131072 // (op 2558) call()

v_lshrrev_b32 v16, 7, v1 // (op 2647) 
v_and_b32 v17, 15, v1 // (op 2647) 
v_lshl_add_u32 v19, v16, 4, v17 // (op 2647) 


v_and_b32 v16, 48, v1 // (op 2647) 
v_lshl_add_u32 v17, v19, 7, v16 // (op 2647) 


v_and_b32 v16, 48, v1 // (op 2722) 
v_and_b32 v19, 63, v1 // (op 2722) 
v_sub_u32 v20, v19, v16 // (op 2722) 

v_lshrrev_b32 v19, 1, v20 // (op 2722) 
v_add_lshl_u32 v21, v16, v19, 6 // (op 2722) 


v_and_b32 v16, 1, v20 // (op 2722) 

v_lshlrev_b32 v19, 3, v16 // (op 2722) 

v_lshrrev_b32 v16, 6, v1 // (op 2722) 
v_and_b32 v20, 1, v16 // (op 2722) 

v_lshl_add_u32 v16, v20, 4, v19 // (op 2722) 


v_add_u32 v19, v21, v16 // (op 2722) 


v_lshrrev_b32 v16, 6, v1 // (op 2798) 
v_and_b32 v20, 1, v16 // (op 2798) 

v_and_b32 v16, 15, v1 // (op 2798) 
v_lshl_add_u32 v21, v20, 4, v16 // (op 2798) 


v_and_b32 v16, 63, v1 // (op 2798) 
v_lshrrev_b32 v20, 4, v16 // (op 2798) 

v_lshl_add_u32 v16, v20, 6, v21 // (op 2798) 


v_lshrrev_b32 v20, 7, v1 // (op 2816) 
s_mul_i32 s54, s16, s8 // (op 2816) 
s_add_u32 s54, s54, s0 // (op 2816) 
v_and_b32 v21, 63, v1 // (op 2816) 
s_mul_hi_u32 s55, s54, s13 // (op 2816) 
v_lshrrev_b32 v22, 4, v21 // (op 2816) 
s_sub_u32 s56, s54, s55 // (op 2816) 
v_lshlrev_b32 v23, 2, v22 // (op 2816) 

s_lshr_b32 s57, s56, 1 // (op 2816) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 2816) 


s_add_u32 s56, s57, s55 // (op 2816) 


v_lshrrev_b32 v20, 2, v21 // (op 2816) 

s_lshr_b32 s55, s56, s11 // (op 2816) 

v_and_b32 v21, 3, v20 // (op 2816) 

s_and_b32 s56, s11, 2147483648 // (op 2816) 
v_and_b32 v20, 3, v1 // (op 2816) 
s_cmp_eq_i32 s56, 0 // (op 2816) 
s_mov_b32 s57, scc // (op 2816) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 2816) 


s_cmp_eq_i32 s57, 0 // (op 2816) 
s_cselect_b32 s56, s54, s55 // (op 2816) 



v_lshrrev_b32 v20, 6, v1 // (op 2816) 
s_mul_hi_u32 s54, s56, s12 // (op 2816) 
v_and_b32 v21, 1, v20 // (op 2816) 

s_sub_u32 s55, s56, s54 // (op 2816) 
v_lshl_add_u32 v20, v21, 4, v23 // (op 2816) 


s_lshr_b32 s57, s55, 1 // (op 2816) 

s_add_u32 s55, s57, s54 // (op 2816) 


s_lshr_b32 s54, s55, s10 // (op 2816) 

s_and_b32 s55, s10, 2147483648 // (op 2816) 
s_cmp_eq_i32 s55, 0 // (op 2816) 
s_mov_b32 s57, scc // (op 2816) 

s_cmp_eq_i32 s57, 0 // (op 2816) 
s_cselect_b32 s55, s56, s54 // (op 2816) 


s_mul_i32 s54, s55, s14 // (op 2816) 
v_lshl_add_u32 v21, s55, 5, v22 // (op 2816) 


s_sub_u32 s55, s56, s54 // (op 2816) 


v_lshl_add_u32 v22, s55, 6, v20 // (op 2816) 


v_mul_lo_u32 v20, v22, s24 // (op 2816) 

v_add_lshl_u32 v22, v21, v20, 1 // (op 2816) 


s_mul_i32 s55, s24, s23 // (op 2818) most significant: low of low * high
s_mul_i32 s56, s25, s22 // (op 2818) most significant: low of high * low
s_mul_i32 s54, s24, s22 // (op 2818) least significant
s_mul_hi_u32 s57, s24, s22 // (op 2818) most significant: high of low * low
s_add_u32 s55, s55, s56 // (op 2818) most significant: sum
s_add_u32 s55, s55, s57 // (op 2818) most significant: sum


s_lshl_b64 s[56:57], s[54:55], 1 // (op 2818) 

s_mov_b32 s64, s26 // (op 2818) reg expression
s_mov_b32 s65, s27 // (op 2818) reg expression
s_mov_b32 s66, s56 // (op 2818) reg expression
s_mov_b32 s67, 131072 // (op 2818) call()

v_lshrrev_b32 v20, 5, v1 // (op 3074) 
v_lshlrev_b32 v21, 2, v20 // (op 3074) 

v_lshl_add_u32 v20, s16, 5, v21 // (op 3074) 

v_and_b32 v21, -32, v1 // (op 3074) 
v_sub_u32 v23, v1, v21 // (op 3074) 

v_lshl_add_u32 v21, v20, 6, v23 // (op 3074) 


v_lshlrev_b32 v20, 2, v21 // (op 3074) 

s_lshl_b32 s54, s15, 13 // (op 3076) 
s_mov_b32 s56, s28 // (op 3076) reg expression
s_mov_b32 s57, s29 // (op 3076) reg expression
s_mov_b32 s58, s54 // (op 3076) reg expression
s_mov_b32 s59, 131072 // (op 3076) call()

s_mul_i32 s6, s8, s16 // (op 283) 
s_add_u32 s6, s6, s0 // (op 283) 
s_mul_hi_u32 s54, s6, s13 // (op 284) 
s_sub_u32 s55, s6, s54 // (op 284) 
s_lshr_b32 s68, s55, 1 // (op 284) 

s_add_u32 s55, s68, s54 // (op 284) 


s_lshr_b32 s54, s55, s11 // (op 284) 

s_and_b32 s55, s11, 2147483648 // (op 284) 
s_cmp_eq_i32 s55, 0 // (op 284) 
s_mov_b32 s68, scc // (op 284) 

s_cmp_eq_i32 s68, 0 // (op 284) 
s_cselect_b32 s55, s6, s54 // (op 284) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 284) 
s_mul_i32 s54, s55, s68 // (op 284) 


s_sub_u32 s18, s6, s54 // (op 284) 

s_ashr_i64 s[54:55], s[2:3], 7 // (op 285) 
s_cmp_gt_u32 s54, s18 // (op 285) 
s_mov_b32 s68, scc // (op 285) 
s_sub_u32 s69, s54, s18 // (op 285) 

s_cmp_eq_i32 s68, 0 // (op 285) 
s_cselect_b32 s54, 0, s69 // (op 285) 


s_cmp_gt_u32 s8, s0 // (op 285) 
s_mov_b32 s55, scc // (op 285) 
s_sub_u32 s68, s8, s0 // (op 285) 
s_cmp_eq_i32 s55, 0 // (op 285) 
s_cselect_b32 s69, 0, s68 // (op 285) 


s_cmp_lt_u32 s54, s69 // (op 285) 
s_mov_b32 s55, scc // (op 285) 
s_cmp_eq_i32 s55, 0 // (op 285) 
s_cselect_b32 s68, s69, s54 // (op 285) 



s_cmp_gt_u32 s9, s6 // (op 285) 
s_mov_b32 s54, scc // (op 285) 
s_sub_u32 s55, s9, s6 // (op 285) 
s_cmp_eq_i32 s54, 0 // (op 285) 
s_cselect_b32 s69, 0, s55 // (op 285) 


s_cmp_lt_u32 s68, s69 // (op 285) 
s_mov_b32 s54, scc // (op 285) 
s_cmp_eq_i32 s54, 0 // (op 285) 
s_cselect_b32 s7, s69, s68 // (op 285) 



s_add_u32 s54, s18, s7 // (op 286) 
s_sub_u32 s19, s54, 1 // (op 286) 

v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 983) call()
v_accvgpr_write a5, 0.00000 // (op 983) call()
v_accvgpr_write a6, 0.00000 // (op 983) call()
v_accvgpr_write a7, 0.00000 // (op 983) call()
v_lshrrev_b32 v21, 3, v1 // (op 2128) 
s_mul_i32 s54, s16, s8 // (op 2128) 
s_add_u32 s54, s54, s0 // (op 2128) 
v_and_b32 v23, -8, v1 // (op 2128) 
s_mul_hi_u32 s55, s54, s13 // (op 2128) 
v_sub_u32 v24, v1, v23 // (op 2128) 

s_sub_u32 s68, s54, s55 // (op 2128) 
v_lshlrev_b32 v23, 4, v24 // (op 2128) 

s_lshr_b32 s69, s68, 1 // (op 2128) 

s_add_u32 s68, s69, s55 // (op 2128) 


s_lshr_b32 s55, s68, s11 // (op 2128) 

s_and_b32 s68, s11, 2147483648 // (op 2128) 
s_cmp_eq_i32 s68, 0 // (op 2128) 
s_mov_b32 s69, scc // (op 2128) 

s_cmp_eq_i32 s69, 0 // (op 2128) 
s_cselect_b32 s68, s54, s55 // (op 2128) 


s_mul_hi_u32 s55, s68, s12 // (op 2128) 
s_sub_u32 s69, s68, s55 // (op 2128) 
s_lshr_b32 s70, s69, 1 // (op 2128) 

s_add_u32 s69, s70, s55 // (op 2128) 


s_lshr_b32 s55, s69, s10 // (op 2128) 

s_and_b32 s69, s10, 2147483648 // (op 2128) 
s_cmp_eq_i32 s69, 0 // (op 2128) 
s_mov_b32 s70, scc // (op 2128) 

s_cmp_eq_i32 s70, 0 // (op 2128) 
s_cselect_b32 s69, s68, s55 // (op 2128) 


v_lshl_add_u32 v24, s69, 5, v21 // (op 2128) 


s_ashr_i64 s[70:71], s[2:3], 7 // (op 2128) 
s_mul_i32 s55, s68, s70 // (op 2128) 


s_sub_u32 s68, s54, s55 // (op 2128) 


v_lshl_add_u32 v21, s68, 7, v23 // (op 2128) 


v_mov_b32 v27, 0 // (op 2128) convert
v_mov_b32 v26, v21 // (op 2128) convert

v_mul_lo_u32 v29, v24, s31 // (op 2128) most significant: low of low * high
v_mul_hi_u32 v21, v24, s30 // (op 2128) most significant: high of low * low
v_mul_lo_u32 v28, v24, s30 // (op 2128) least significant: low of low * low
v_add_u32 v29, v29, v21 // (op 2128) most significant: sum

v_add_co_u32 v28, s[54:55], v28, v26 // (op 2128) least significant half
v_addc_co_u32 v29, s[54:55], v29, v27, s[54:55] // (op 2128) most significant half



s_mul_i32 s55, s30, s35 // (op 2130) most significant: low of low * high
s_mul_i32 s68, s31, s34 // (op 2130) most significant: low of high * low
s_mul_i32 s54, s30, s34 // (op 2130) least significant
s_mul_hi_u32 s69, s30, s34 // (op 2130) most significant: high of low * low
s_add_u32 s55, s55, s68 // (op 2130) most significant: sum
s_add_u32 s55, s55, s69 // (op 2130) most significant: sum


s_mov_b32 s68, s32 // (op 2130) reg expression
s_mov_b32 s69, s33 // (op 2130) reg expression
s_mov_b32 s70, s54 // (op 2130) reg expression
s_mov_b32 s71, 131072 // (op 2130) call()

v_lshrrev_b32 v21, 3, v1 // (op 2147) 
s_mul_i32 s54, s16, s8 // (op 2147) 
s_add_u32 s54, s54, s0 // (op 2147) 
v_and_b32 v23, -8, v1 // (op 2147) 
s_mul_hi_u32 s55, s54, s13 // (op 2147) 
v_sub_u32 v24, v1, v23 // (op 2147) 

v_lshlrev_b32 v23, 4, v24 // (op 2147) 

s_sub_u32 s72, s54, s55 // (op 2147) 
s_lshr_b32 s73, s72, 1 // (op 2147) 

s_add_u32 s72, s73, s55 // (op 2147) 


s_lshr_b32 s55, s72, s11 // (op 2147) 

s_and_b32 s72, s11, 2147483648 // (op 2147) 
s_cmp_eq_i32 s72, 0 // (op 2147) 
s_mov_b32 s73, scc // (op 2147) 

s_cmp_eq_i32 s73, 0 // (op 2147) 
s_cselect_b32 s72, s54, s55 // (op 2147) 


s_mul_hi_u32 s55, s72, s12 // (op 2147) 
s_sub_u32 s73, s72, s55 // (op 2147) 
s_lshr_b32 s74, s73, 1 // (op 2147) 

s_add_u32 s73, s74, s55 // (op 2147) 


s_lshr_b32 s55, s73, s10 // (op 2147) 

s_and_b32 s73, s10, 2147483648 // (op 2147) 
s_cmp_eq_i32 s73, 0 // (op 2147) 
s_mov_b32 s74, scc // (op 2147) 

s_cmp_eq_i32 s74, 0 // (op 2147) 
s_cselect_b32 s73, s72, s55 // (op 2147) 


v_lshl_add_u32 v24, s73, 5, v21 // (op 2147) 


s_ashr_i64 s[74:75], s[2:3], 7 // (op 2147) 
s_mul_i32 s55, s72, s74 // (op 2147) 


s_sub_u32 s72, s54, s55 // (op 2147) 


v_lshl_add_u32 v21, s72, 7, v23 // (op 2147) 


v_mov_b32 v27, 0 // (op 2147) convert
v_mov_b32 v26, v21 // (op 2147) convert

v_mul_lo_u32 v31, v24, s31 // (op 2147) most significant: low of low * high
v_mul_hi_u32 v21, v24, s30 // (op 2147) most significant: high of low * low
v_mul_lo_u32 v30, v24, s30 // (op 2147) least significant: low of low * low
v_add_u32 v31, v31, v21 // (op 2147) most significant: sum

v_add_co_u32 v30, s[54:55], v30, v26 // (op 2147) least significant half
v_addc_co_u32 v31, s[54:55], v31, v27, s[54:55] // (op 2147) most significant half



s_mul_i32 s55, s30, s35 // (op 2149) most significant: low of low * high
s_mul_i32 s72, s31, s34 // (op 2149) most significant: low of high * low
s_mul_i32 s54, s30, s34 // (op 2149) least significant
s_mul_hi_u32 s73, s30, s34 // (op 2149) most significant: high of low * low
s_add_u32 s55, s55, s72 // (op 2149) most significant: sum
s_add_u32 s55, s55, s73 // (op 2149) most significant: sum


s_mov_b32 s68, s32 // (op 2149) reg expression
s_mov_b32 s69, s33 // (op 2149) reg expression
s_mov_b32 s70, s54 // (op 2149) reg expression
s_mov_b32 s71, 131072 // (op 2149) call()

v_and_b32 v21, -4, v1 // (op 2256) 
s_mul_i32 s54, s16, s8 // (op 2256) 
s_add_u32 s54, s54, s0 // (op 2256) 
v_sub_u32 v23, v1, v21 // (op 2256) 

s_mul_hi_u32 s55, s54, s13 // (op 2256) 
v_lshlrev_b32 v21, 4, v23 // (op 2256) 

s_sub_u32 s72, s54, s55 // (op 2256) 
v_lshrrev_b32 v23, 2, v1 // (op 2256) 
s_lshr_b32 s73, s72, 1 // (op 2256) 

s_add_u32 s72, s73, s55 // (op 2256) 


s_lshr_b32 s55, s72, s11 // (op 2256) 

s_and_b32 s72, s11, 2147483648 // (op 2256) 
s_cmp_eq_i32 s72, 0 // (op 2256) 
s_mov_b32 s73, scc // (op 2256) 

s_cmp_eq_i32 s73, 0 // (op 2256) 
s_cselect_b32 s72, s54, s55 // (op 2256) 


s_mul_hi_u32 s55, s72, s12 // (op 2256) 
s_sub_u32 s73, s72, s55 // (op 2256) 
s_lshr_b32 s74, s73, 1 // (op 2256) 

s_add_u32 s73, s74, s55 // (op 2256) 


s_lshr_b32 s55, s73, s10 // (op 2256) 

s_and_b32 s73, s10, 2147483648 // (op 2256) 
s_cmp_eq_i32 s73, 0 // (op 2256) 
s_mov_b32 s74, scc // (op 2256) 

s_cmp_eq_i32 s74, 0 // (op 2256) 
s_cselect_b32 s73, s72, s55 // (op 2256) 


s_mul_i32 s55, s73, s14 // (op 2256) 

s_sub_u32 s73, s72, s55 // (op 2256) 

v_lshl_add_u32 v24, s73, 6, v21 // (op 2256) 


s_ashr_i64 s[74:75], s[2:3], 7 // (op 2256) 
v_mov_b32 v27, 0 // (op 2256) convert
v_mov_b32 v26, v24 // (op 2256) convert

s_mul_i32 s55, s72, s74 // (op 2256) 


s_sub_u32 s72, s54, s55 // (op 2256) 


v_lshl_add_u32 v21, s72, 7, v23 // (op 2256) 


v_mul_lo_u32 v25, v21, s37 // (op 2256) most significant: low of low * high
v_mul_hi_u32 v23, v21, s36 // (op 2256) most significant: high of low * low
v_mul_lo_u32 v24, v21, s36 // (op 2256) least significant: low of low * low
v_add_u32 v25, v25, v23 // (op 2256) most significant: sum

v_add_co_u32 v24, s[54:55], v24, v26 // (op 2256) least significant half
v_addc_co_u32 v25, s[54:55], v25, v27, s[54:55] // (op 2256) most significant half



s_mul_i32 s55, s36, s3 // (op 2258) most significant: low of low * high
s_mul_i32 s72, s37, s2 // (op 2258) most significant: low of high * low
s_mul_i32 s54, s36, s2 // (op 2258) least significant
s_mul_hi_u32 s73, s36, s2 // (op 2258) most significant: high of low * low
s_add_u32 s55, s55, s72 // (op 2258) most significant: sum
s_add_u32 s55, s55, s73 // (op 2258) most significant: sum


s_mov_b32 s72, s38 // (op 2258) reg expression
s_mov_b32 s73, s39 // (op 2258) reg expression
s_mov_b32 s74, s54 // (op 2258) reg expression
s_mov_b32 s75, 131072 // (op 2258) call()

v_and_b32 v21, -4, v1 // (op 2275) 
s_mul_i32 s54, s16, s8 // (op 2275) 
s_add_u32 s54, s54, s0 // (op 2275) 
v_sub_u32 v23, v1, v21 // (op 2275) 

s_mul_hi_u32 s55, s54, s13 // (op 2275) 
v_lshlrev_b32 v21, 4, v23 // (op 2275) 

v_lshrrev_b32 v23, 2, v1 // (op 2275) 
s_sub_u32 s76, s54, s55 // (op 2275) 
s_lshr_b32 s77, s76, 1 // (op 2275) 

s_add_u32 s76, s77, s55 // (op 2275) 


s_lshr_b32 s55, s76, s11 // (op 2275) 

s_and_b32 s76, s11, 2147483648 // (op 2275) 
s_cmp_eq_i32 s76, 0 // (op 2275) 
s_mov_b32 s77, scc // (op 2275) 

s_cmp_eq_i32 s77, 0 // (op 2275) 
s_cselect_b32 s76, s54, s55 // (op 2275) 


s_mul_hi_u32 s55, s76, s12 // (op 2275) 
s_sub_u32 s77, s76, s55 // (op 2275) 
s_lshr_b32 s78, s77, 1 // (op 2275) 

s_add_u32 s77, s78, s55 // (op 2275) 


s_lshr_b32 s55, s77, s10 // (op 2275) 

s_and_b32 s77, s10, 2147483648 // (op 2275) 
s_cmp_eq_i32 s77, 0 // (op 2275) 
s_mov_b32 s78, scc // (op 2275) 

s_cmp_eq_i32 s78, 0 // (op 2275) 
s_cselect_b32 s77, s76, s55 // (op 2275) 


s_mul_i32 s55, s77, s14 // (op 2275) 

s_sub_u32 s77, s76, s55 // (op 2275) 

v_lshl_add_u32 v26, s77, 6, v21 // (op 2275) 


s_ashr_i64 s[78:79], s[2:3], 7 // (op 2275) 
v_mov_b32 v33, 0 // (op 2275) convert
v_mov_b32 v32, v26 // (op 2275) convert

s_mul_i32 s55, s76, s78 // (op 2275) 


s_sub_u32 s76, s54, s55 // (op 2275) 


v_lshl_add_u32 v21, s76, 7, v23 // (op 2275) 


v_mul_lo_u32 v27, v21, s37 // (op 2275) most significant: low of low * high
v_mul_hi_u32 v23, v21, s36 // (op 2275) most significant: high of low * low
v_mul_lo_u32 v26, v21, s36 // (op 2275) least significant: low of low * low
v_add_u32 v27, v27, v23 // (op 2275) most significant: sum

v_add_co_u32 v26, s[54:55], v26, v32 // (op 2275) least significant half
v_addc_co_u32 v27, s[54:55], v27, v33, s[54:55] // (op 2275) most significant half



s_mul_i32 s55, s36, s3 // (op 2277) most significant: low of low * high
s_mul_i32 s76, s37, s2 // (op 2277) most significant: low of high * low
s_mul_i32 s54, s36, s2 // (op 2277) least significant
s_mul_hi_u32 s77, s36, s2 // (op 2277) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 2277) most significant: sum
s_add_u32 s55, s55, s77 // (op 2277) most significant: sum


s_mov_b32 s72, s38 // (op 2277) reg expression
s_mov_b32 s73, s39 // (op 2277) reg expression
s_mov_b32 s74, s54 // (op 2277) reg expression
s_mov_b32 s75, 131072 // (op 2277) call()

v_lshrrev_b32 v21, 7, v1 // (op 2384) 
s_mul_i32 s54, s16, s8 // (op 2384) 
s_add_u32 s54, s54, s0 // (op 2384) 
v_and_b32 v23, 15, v1 // (op 2384) 
s_mul_hi_u32 s55, s54, s13 // (op 2384) 
v_lshl_add_u32 v32, v21, 4, v23 // (op 2384) 


s_sub_u32 s76, s54, s55 // (op 2384) 
v_and_b32 v21, 63, v1 // (op 2384) 
s_lshr_b32 s77, s76, 1 // (op 2384) 

v_lshrrev_b32 v23, 4, v21 // (op 2384) 

s_add_u32 s76, s77, s55 // (op 2384) 


s_lshr_b32 s55, s76, s11 // (op 2384) 

s_and_b32 s76, s11, 2147483648 // (op 2384) 
s_cmp_eq_i32 s76, 0 // (op 2384) 
s_mov_b32 s77, scc // (op 2384) 

s_cmp_eq_i32 s77, 0 // (op 2384) 
s_cselect_b32 s76, s54, s55 // (op 2384) 


s_mul_hi_u32 s55, s76, s12 // (op 2384) 
s_sub_u32 s77, s76, s55 // (op 2384) 
s_lshr_b32 s78, s77, 1 // (op 2384) 

s_add_u32 s77, s78, s55 // (op 2384) 


s_lshr_b32 s55, s77, s10 // (op 2384) 

s_and_b32 s77, s10, 2147483648 // (op 2384) 
s_cmp_eq_i32 s77, 0 // (op 2384) 
s_mov_b32 s78, scc // (op 2384) 

s_cmp_eq_i32 s78, 0 // (op 2384) 
s_cselect_b32 s77, s76, s55 // (op 2384) 


v_lshl_add_u32 v21, s77, 5, v32 // (op 2384) 


s_ashr_i64 s[78:79], s[2:3], 7 // (op 2384) 
s_mul_i32 s55, s76, s78 // (op 2384) 


s_sub_u32 s76, s54, s55 // (op 2384) 


v_lshl_add_u32 v32, s76, 2, v23 // (op 2384) 


v_mul_lo_u32 v23, v21, s40 // (op 2384) 
v_add_u32 v23, v23, v32 // (op 2384) 


s_mul_i32 s55, s40, s35 // (op 2386) most significant: low of low * high
s_mul_i32 s76, s41, s34 // (op 2386) most significant: low of high * low
s_mul_i32 s54, s40, s34 // (op 2386) least significant
s_mul_hi_u32 s77, s40, s34 // (op 2386) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 2386) most significant: sum
s_add_u32 s55, s55, s77 // (op 2386) most significant: sum


s_mov_b32 s76, s42 // (op 2386) reg expression
s_mov_b32 s77, s43 // (op 2386) reg expression
s_mov_b32 s78, s54 // (op 2386) reg expression
s_mov_b32 s79, 131072 // (op 2386) call()

v_lshrrev_b32 v21, 7, v1 // (op 2403) 
s_mul_i32 s54, s16, s8 // (op 2403) 
s_add_u32 s54, s54, s0 // (op 2403) 
v_and_b32 v32, 15, v1 // (op 2403) 
s_mul_hi_u32 s55, s54, s13 // (op 2403) 
v_lshl_add_u32 v33, v21, 4, v32 // (op 2403) 


v_and_b32 v21, 63, v1 // (op 2403) 
s_sub_u32 s80, s54, s55 // (op 2403) 
v_lshrrev_b32 v32, 4, v21 // (op 2403) 

s_lshr_b32 s81, s80, 1 // (op 2403) 

s_add_u32 s80, s81, s55 // (op 2403) 


s_lshr_b32 s55, s80, s11 // (op 2403) 

s_and_b32 s80, s11, 2147483648 // (op 2403) 
s_cmp_eq_i32 s80, 0 // (op 2403) 
s_mov_b32 s81, scc // (op 2403) 

s_cmp_eq_i32 s81, 0 // (op 2403) 
s_cselect_b32 s80, s54, s55 // (op 2403) 


s_mul_hi_u32 s55, s80, s12 // (op 2403) 
s_sub_u32 s81, s80, s55 // (op 2403) 
s_lshr_b32 s82, s81, 1 // (op 2403) 

s_add_u32 s81, s82, s55 // (op 2403) 


s_lshr_b32 s55, s81, s10 // (op 2403) 

s_and_b32 s81, s10, 2147483648 // (op 2403) 
s_cmp_eq_i32 s81, 0 // (op 2403) 
s_mov_b32 s82, scc // (op 2403) 

s_cmp_eq_i32 s82, 0 // (op 2403) 
s_cselect_b32 s81, s80, s55 // (op 2403) 


v_lshl_add_u32 v21, s81, 5, v33 // (op 2403) 


s_ashr_i64 s[82:83], s[2:3], 7 // (op 2403) 
s_mul_i32 s55, s80, s82 // (op 2403) 


s_sub_u32 s80, s54, s55 // (op 2403) 


v_lshl_add_u32 v33, s80, 2, v32 // (op 2403) 


v_mul_lo_u32 v32, v21, s40 // (op 2403) 
v_add_u32 v32, v32, v33 // (op 2403) 


s_mul_i32 s55, s40, s35 // (op 2405) most significant: low of low * high
s_mul_i32 s80, s41, s34 // (op 2405) most significant: low of high * low
s_mul_i32 s54, s40, s34 // (op 2405) least significant
s_mul_hi_u32 s81, s40, s34 // (op 2405) most significant: high of low * low
s_add_u32 s55, s55, s80 // (op 2405) most significant: sum
s_add_u32 s55, s55, s81 // (op 2405) most significant: sum


s_mov_b32 s76, s42 // (op 2405) reg expression
s_mov_b32 s77, s43 // (op 2405) reg expression
s_mov_b32 s78, s54 // (op 2405) reg expression
s_mov_b32 s79, 131072 // (op 2405) call()

v_and_b32 v21, -64, v1 // (op 2512) 
s_mul_i32 s54, s16, s8 // (op 2512) 
s_add_u32 s54, s54, s0 // (op 2512) 
v_sub_u32 v33, v1, v21 // (op 2512) 

s_mul_hi_u32 s55, s54, s13 // (op 2512) 
v_lshrrev_b32 v21, 6, v1 // (op 2512) 
s_sub_u32 s80, s54, s55 // (op 2512) 
s_lshr_b32 s81, s80, 1 // (op 2512) 

s_add_u32 s80, s81, s55 // (op 2512) 


s_lshr_b32 s55, s80, s11 // (op 2512) 

s_and_b32 s80, s11, 2147483648 // (op 2512) 
s_cmp_eq_i32 s80, 0 // (op 2512) 
s_mov_b32 s81, scc // (op 2512) 

s_cmp_eq_i32 s81, 0 // (op 2512) 
s_cselect_b32 s80, s54, s55 // (op 2512) 


s_mul_hi_u32 s55, s80, s12 // (op 2512) 
s_sub_u32 s81, s80, s55 // (op 2512) 
s_lshr_b32 s82, s81, 1 // (op 2512) 

s_add_u32 s81, s82, s55 // (op 2512) 


s_lshr_b32 s55, s81, s10 // (op 2512) 

s_and_b32 s81, s10, 2147483648 // (op 2512) 
s_cmp_eq_i32 s81, 0 // (op 2512) 
s_mov_b32 s82, scc // (op 2512) 

s_cmp_eq_i32 s82, 0 // (op 2512) 
s_cselect_b32 s81, s80, s55 // (op 2512) 


s_mul_i32 s55, s81, s14 // (op 2512) 

s_sub_u32 s81, s80, s55 // (op 2512) 

s_ashr_i64 s[82:83], s[2:3], 7 // (op 2512) 
v_lshl_add_u32 v34, s81, 6, v33 // (op 2512) 


s_mul_i32 s55, s80, s82 // (op 2512) 


s_sub_u32 s80, s54, s55 // (op 2512) 


v_lshl_add_u32 v33, s80, 2, v21 // (op 2512) 


v_mul_lo_u32 v21, v33, s44 // (op 2512) 
v_add_u32 v21, v21, v34 // (op 2512) 


s_mul_i32 s55, s44, s49 // (op 2514) most significant: low of low * high
s_mul_i32 s80, s45, s48 // (op 2514) most significant: low of high * low
s_mul_i32 s54, s44, s48 // (op 2514) least significant
s_mul_hi_u32 s81, s44, s48 // (op 2514) most significant: high of low * low
s_add_u32 s55, s55, s80 // (op 2514) most significant: sum
s_add_u32 s55, s55, s81 // (op 2514) most significant: sum


s_mov_b32 s80, s46 // (op 2514) reg expression
s_mov_b32 s81, s47 // (op 2514) reg expression
s_mov_b32 s82, s54 // (op 2514) reg expression
s_mov_b32 s83, 131072 // (op 2514) call()

v_and_b32 v33, -64, v1 // (op 2531) 
s_mul_i32 s54, s16, s8 // (op 2531) 
s_add_u32 s54, s54, s0 // (op 2531) 
v_sub_u32 v34, v1, v33 // (op 2531) 

s_mul_hi_u32 s55, s54, s13 // (op 2531) 
v_lshrrev_b32 v33, 6, v1 // (op 2531) 
s_sub_u32 s84, s54, s55 // (op 2531) 
s_lshr_b32 s85, s84, 1 // (op 2531) 

s_add_u32 s84, s85, s55 // (op 2531) 


s_lshr_b32 s55, s84, s11 // (op 2531) 

s_and_b32 s84, s11, 2147483648 // (op 2531) 
s_cmp_eq_i32 s84, 0 // (op 2531) 
s_mov_b32 s85, scc // (op 2531) 

s_cmp_eq_i32 s85, 0 // (op 2531) 
s_cselect_b32 s84, s54, s55 // (op 2531) 


s_mul_hi_u32 s55, s84, s12 // (op 2531) 
s_sub_u32 s85, s84, s55 // (op 2531) 
s_lshr_b32 s86, s85, 1 // (op 2531) 

s_add_u32 s85, s86, s55 // (op 2531) 


s_lshr_b32 s55, s85, s10 // (op 2531) 

s_and_b32 s85, s10, 2147483648 // (op 2531) 
s_cmp_eq_i32 s85, 0 // (op 2531) 
s_mov_b32 s86, scc // (op 2531) 

s_cmp_eq_i32 s86, 0 // (op 2531) 
s_cselect_b32 s85, s84, s55 // (op 2531) 


s_mul_i32 s55, s85, s14 // (op 2531) 

s_sub_u32 s85, s84, s55 // (op 2531) 

v_lshl_add_u32 v35, s85, 6, v34 // (op 2531) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 2531) 
s_mul_i32 s55, s84, s86 // (op 2531) 


s_sub_u32 s84, s54, s55 // (op 2531) 


v_lshl_add_u32 v34, s84, 2, v33 // (op 2531) 


v_mul_lo_u32 v33, v34, s44 // (op 2531) 
v_add_u32 v33, v33, v35 // (op 2531) 


s_mul_i32 s55, s44, s49 // (op 2533) most significant: low of low * high
s_mul_i32 s84, s45, s48 // (op 2533) most significant: low of high * low
s_mul_i32 s54, s44, s48 // (op 2533) least significant
s_mul_hi_u32 s85, s44, s48 // (op 2533) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 2533) most significant: sum
s_add_u32 s55, s55, s85 // (op 2533) most significant: sum


s_mov_b32 s80, s46 // (op 2533) reg expression
s_mov_b32 s81, s47 // (op 2533) reg expression
s_mov_b32 s82, s54 // (op 2533) reg expression
s_mov_b32 s83, 131072 // (op 2533) call()




buffer_load_ubyte v34, v23, s[76:79], 0 offen offset: 0 // (op 1817) Load value
v_and_b32 v35, -4, v1 // (op 2688) 
v_sub_u32 v36, v1, v35 // (op 2688) 

v_lshlrev_b32 v35, 4, v36 // (op 2688) 

v_lshrrev_b32 v36, 2, v1 // (op 2688) 
v_lshl_add_u32 v37, v36, 6, v35 // (op 2688) 


s_nop 0
v_readfirstlane_b32 s54, v37 // (op 2688) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3423) 

s_lshl_b64 s[84:85], s[36:37], 6 // (op 2006) 
v_mov_b64 v[36:37], s[84:85] // (op 2006) convert


v_mov_b64 v[38:39], v[24:25] // (op 2006) reg expression
s_mov_b32 m0, s54 // (op 2006) reg expression
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 2006) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v36 // (op 2006) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 2006) most significant half

s_add_u32 m0, m0, 4096 // (op 2006) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 2006) Load value direct to lds





v_mov_b32 v35, v21 // (op 1824) reg expression

buffer_load_ubyte v36, v35, s[80:83], 0 offen offset: 0 // (op 1824) Load value

v_and_b32 v35, -8, v1 // (op 2623) 
v_sub_u32 v37, v1, v35 // (op 2623) 

v_lshlrev_b32 v35, 4, v37 // (op 2623) 

v_lshrrev_b32 v37, 3, v1 // (op 2623) 
v_lshl_add_u32 v38, v37, 7, v35 // (op 2623) 


s_nop 0
v_readfirstlane_b32 s54, v38 // (op 2623) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3404) 


s_nop 0
s_add_u32 s55, 8192, s54 // (op 2011) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s55 // (op 2011) reg expression
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 2011) Load value direct to lds





buffer_load_ubyte v35, v23, s[76:79], 0 offen offset:4 // (op 1831) Load value

v_and_b32 v23, -4, v1 // (op 3277) 
v_sub_u32 v37, v1, v23 // (op 3277) 

v_lshlrev_b32 v23, 4, v37 // (op 3277) 

v_lshrrev_b32 v37, 2, v1 // (op 3277) 
v_lshl_add_u32 v38, v37, 6, v23 // (op 3277) 


s_nop 0
v_readfirstlane_b32 s54, v38 // (op 3277) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3553) 

s_lshl_b64 s[84:85], s[36:37], 6 // (op 2016) 
v_mov_b64 v[38:39], s[84:85] // (op 2016) convert


s_lshl_b64 s[84:85], s[36:37], 7 // (op 2016) 
v_mov_b32 v23, s84 // (op 2016) 
v_mov_b32 v37, s85 // (op 2016) 
v_add_co_u32 v40, s[86:87], v24, v23 // (op 2016) least significant half
v_addc_co_u32 v41, s[86:87], v25, v37, s[86:87] // (op 2016) most significant half




s_add_u32 s55, 12288, s54 // (op 2016) 
s_mov_b32 m0, s55 // (op 2016) reg expression
buffer_load_dwordx4 v40, s[72:75], 0 offen offset: 0 lds // (op 2016) Load value direct to lds
v_add_co_u32 v40, s[84:85], v40, v38 // (op 2016) least significant half
v_addc_co_u32 v41, s[84:85], 0, v41, s[84:85] // (op 2016) most significant half

s_add_u32 m0, m0, 4096 // (op 2016) 
buffer_load_dwordx4 v40, s[72:75], 0 offen offset: 0 lds // (op 2016) Load value direct to lds







s_lshl_b32 s54, s44, 2 // (op 1838) 
v_add_u32 v23, v21, s54 // (op 1838) 


buffer_load_ubyte v24, v23, s[80:83], 0 offen offset: 0 // (op 1838) Load value


v_and_b32 v21, -8, v1 // (op 3238) 
v_sub_u32 v23, v1, v21 // (op 3238) 

v_lshlrev_b32 v21, 4, v23 // (op 3238) 

v_lshrrev_b32 v23, 3, v1 // (op 3238) 
v_lshl_add_u32 v25, v23, 7, v21 // (op 3238) 


s_nop 0
v_readfirstlane_b32 s54, v25 // (op 3238) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3544) 


s_nop 0
s_add_u32 s55, 20480, s54 // (op 2018) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s55 // (op 2018) reg expression
s_mov_b32 s84, 128 // (op 2018) call()
buffer_load_dwordx4 v28, s[68:71], s84 offen offset: 0 lds // (op 2018) Load value direct to lds




v_and_b32 v21, -64, v1 // (op 2764) 
v_sub_u32 v23, v1, v21 // (op 2764) 

v_and_b32 v21, 4294967232, v1 // (op 2764) 
v_add_u32 v25, v21, v23 // (op 2764) 


s_barrier  // (op 3442) 



s_waitcnt vmcnt(6)
ds_write_b8 v25, v36 offset:24576 // (op 1844) Store local data 

s_barrier  // (op 3420) 



ds_read_b64_tr_b8 v[38:39], v19 // (op 1862) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v19 offset:512 // (op 1862) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v19 offset:4096 // (op 1862) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v19 offset:4608 // (op 1862) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3439) 


ds_read_u8 v21, v16 offset:24576 // (op 1872) Load local data 



ds_read_b64_tr_b8 v[46:47], v19 offset:32 // (op 1867) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v19 offset:544 // (op 1867) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v19 offset:4128 // (op 1867) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v19 offset:4640 // (op 1867) Transpose load local data 


ds_read_u8 v23, v16 offset:24608 // (op 1877) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 3401) 



ds_read_b128 v[54:57], v17 offset:8192 // (op 1858) Load local data 
ds_read_b128 v[58:61], v17 offset:8256 // (op 1858) Load local data 
v_and_b32 v25, -8, v1 // (op 2595) 
v_sub_u32 v37, v1, v25 // (op 2595) 

v_lshlrev_b32 v25, 4, v37 // (op 2595) 

v_lshrrev_b32 v37, 3, v1 // (op 2595) 
v_lshl_add_u32 v28, v37, 7, v25 // (op 2595) 


s_nop 0
v_readfirstlane_b32 s54, v28 // (op 2595) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -4, v1 // (op 2660) 
v_sub_u32 v37, v1, v25 // (op 2660) 

v_lshlrev_b32 v25, 4, v37 // (op 2660) 

v_lshrrev_b32 v37, 2, v1 // (op 2660) 
v_lshl_add_u32 v28, v37, 6, v25 // (op 2660) 


s_nop 0
v_readfirstlane_b32 s55, v28 // (op 2660) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -64, v1 // (op 2735) 
v_sub_u32 v37, v1, v25 // (op 2735) 

v_and_b32 v25, 4294967232, v1 // (op 2735) 
v_add_u32 v28, v25, v37 // (op 2735) 


v_and_b32 v25, -8, v1 // (op 3223) 
v_sub_u32 v29, v1, v25 // (op 3223) 

v_lshlrev_b32 v25, 4, v29 // (op 3223) 

v_lshrrev_b32 v29, 3, v1 // (op 3223) 
v_lshl_add_u32 v37, v29, 7, v25 // (op 3223) 


s_nop 0
v_readfirstlane_b32 s84, v37 // (op 3223) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -4, v1 // (op 3262) 
v_sub_u32 v29, v1, v25 // (op 3262) 

v_lshlrev_b32 v25, 4, v29 // (op 3262) 

v_lshrrev_b32 v29, 2, v1 // (op 3262) 
v_lshl_add_u32 v37, v29, 6, v25 // (op 3262) 


s_nop 0
v_readfirstlane_b32 s85, v37 // (op 3262) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -64, v1 // (op 3306) 
v_sub_u32 v29, v1, v25 // (op 3306) 

v_and_b32 v25, 4294967232, v1 // (op 3306) 
v_add_u32 v37, v25, v29 // (op 3306) 


s_mov_b32 s1, 0 // (op 985) call()
s_and_b32 s86, s7, 4294967294 // (op 984) 
s_cmp_lt_u32 s1, s86 // (op 984) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_984 // (op 984) Condition: Top (jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_984 if false)
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoop_984:
s_add_u32 s1, s1, 2 // (op 986) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:61], v[38:45], a[0:3], v34, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 84) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:61], v[46:53], a[4:7], v34, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1020) 


v_mov_b32 v25, v32 // (op 77) 

buffer_load_ubyte v34, v25, s[76:79], 0 offen offset:8 // (op 77) Load value
s_barrier  // (op 3426) 

s_lshl_b64 s[86:87], s[36:37], 6 // (op 1931) 
v_mov_b64 v[62:63], s[86:87] // (op 1931) convert


v_mov_b64 v[64:65], v[26:27] // (op 1931) 
s_lshl_b64 s[86:87], s[36:37], 7 // (op 1931) 
s_lshl_b64 s[88:89], s[86:87], 1 // (op 1931) 

v_mov_b32 v29, s88 // (op 1931) 
v_mov_b32 v66, s89 // (op 1931) 
v_add_co_u32 v68, s[86:87], v64, v29 // (op 1931) least significant half
v_addc_co_u32 v69, s[86:87], v65, v66, s[86:87] // (op 1931) most significant half




s_mov_b32 m0, s55 // (op 1931) reg expression
buffer_load_dwordx4 v68, s[72:75], 0 offen offset: 0 lds // (op 1931) Load value direct to lds
v_add_co_u32 v68, s[86:87], v68, v62 // (op 1931) least significant half
v_addc_co_u32 v69, s[86:87], 0, v69, s[86:87] // (op 1931) most significant half

s_add_u32 m0, m0, 4096 // (op 1931) 
buffer_load_dwordx4 v68, s[72:75], 0 offen offset: 0 lds // (op 1931) Load value direct to lds




v_mov_b32 v29, v33 // (op 21) 
s_lshl_b32 s86, s44, 3 // (op 21) 
v_add_u32 v62, v29, s86 // (op 21) 


buffer_load_ubyte v36, v62, s[80:83], 0 offen offset: 0 // (op 21) Load value

s_barrier  // (op 3407) 


v_mov_b64 v[62:63], v[30:31] // (op 1926) 
s_add_u32 s86, 8192, s54 // (op 1926) 

s_mov_b32 m0, s86 // (op 1926) reg expression
s_mov_b32 s87, 256 // (op 1926) call()
buffer_load_dwordx4 v62, s[68:71], s87 offen offset: 0 lds // (op 1926) Load value direct to lds


s_barrier  // (op 3562) 



s_waitcnt vmcnt(6)
ds_write_b8 v37, v24 offset:24832 // (op 913) Store local data 
s_barrier  // (op 3550) 



ds_read_b64_tr_b8 v[38:39], v13 offset:12288 // (op 906) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v13 offset:12800 // (op 906) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v13 offset:16384 // (op 906) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v13 offset:16896 // (op 906) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3559) 


ds_read_u8 v21, v12 offset:24832 // (op 910) Load local data 



ds_read_b64_tr_b8 v[46:47], v13 offset:12320 // (op 1025) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v13 offset:12832 // (op 1025) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v13 offset:16416 // (op 1025) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v13 offset:16928 // (op 1025) Transpose load local data 


ds_read_u8 v23, v12 offset:24864 // (op 1028) Load local data 
s_barrier  // (op 3717) 
s_waitcnt vmcnt(5)
s_barrier  // (op 3541) 



ds_read_b128 v[54:57], v6 offset:20480 // (op 904) Load local data 
ds_read_b128 v[58:61], v6 offset:20544 // (op 904) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:61], v[38:45], a[0:3], v35, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 911) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:61], v[46:53], a[4:7], v35, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1021) 



buffer_load_ubyte v35, v25, s[76:79], 0 offen offset:12 // (op 908) Load value

s_barrier  // (op 3556) 
v_add_u32 v32, 8, v32 // (op 2395) 

s_lshl_b64 s[86:87], s[36:37], 6 // (op 1987) 
v_mov_b64 v[66:67], s[86:87] // (op 1987) convert


s_lshl_b64 s[86:87], s[36:37], 7 // (op 1987) 
s_mul_i32 s89, 3, s87 // (op 1987) most significant: low of low * high
s_mul_i32 s88, 3, s86 // (op 1987) least significant
s_mul_hi_u32 s90, 3, s86 // (op 1987) most significant: high of low * low
s_add_u32 s89, s89, s90 // (op 1987) most significant: sum


v_mov_b32 v25, s88 // (op 1987) 
v_mov_b32 v68, s89 // (op 1987) 
v_add_co_u32 v70, s[86:87], v64, v25 // (op 1987) least significant half
v_addc_co_u32 v71, s[86:87], v65, v68, s[86:87] // (op 1987) most significant half




s_add_u32 s86, 12288, s85 // (op 1987) 
s_nop 0
s_mov_b32 m0, s86 // (op 1987) reg expression
buffer_load_dwordx4 v70, s[72:75], 0 offen offset: 0 lds // (op 1987) Load value direct to lds
v_add_co_u32 v70, s[88:89], v70, v66 // (op 1987) least significant half
v_addc_co_u32 v71, s[88:89], 0, v71, s[88:89] // (op 1987) most significant half

s_add_u32 m0, m0, 4096 // (op 1987) 
buffer_load_dwordx4 v70, s[72:75], 0 offen offset: 0 lds // (op 1987) Load value direct to lds






s_lshl_b32 s86, s44, 2 // (op 915) 
s_mul_i32 s87, 3, s86 // (op 915) 

v_add_u32 v25, v29, s87 // (op 915) 


s_lshl_b64 s[86:87], s[36:37], 7 // (op 2267) 
s_lshl_b64 s[88:89], s[86:87], 1 // (op 2267) 

v_mov_b32 v64, s88 // (op 2267) 
v_mov_b32 v65, s89 // (op 2267) 
v_add_co_u32 v26, s[86:87], v26, v64 // (op 2267) least significant half
v_addc_co_u32 v27, s[86:87], v27, v65, s[86:87] // (op 2267) most significant half




buffer_load_ubyte v24, v25, s[80:83], 0 offen offset: 0 // (op 915) Load value


s_barrier  // (op 3547) 
s_lshl_b32 s86, s44, 3 // (op 2523) 


v_add_u32 v33, v33, s86 // (op 2523) 

s_add_u32 s86, 20480, s84 // (op 1982) 

s_mov_b32 m0, s86 // (op 1982) reg expression
s_mov_b32 s87, 384 // (op 1982) call()
buffer_load_dwordx4 v62, s[68:71], s87 offen offset: 0 lds // (op 1982) Load value direct to lds



s_barrier  // (op 3445) 
v_add_co_u32 v30, vcc, 256, v30 // (op 2139) least significant half



v_addc_co_u32 v31, vcc, 0, v31, vcc // (op 2139) most significant half
s_waitcnt vmcnt(6)
ds_write_b8 v28, v36 offset:24576 // (op 26) Store local data 
s_barrier  // (op 3432) 



ds_read_b64_tr_b8 v[38:39], v7 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v7 offset:512 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v7 offset:4096 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:4608 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3452) 


ds_read_u8 v21, v0 offset:24576 // (op 81) Load local data 



ds_read_b64_tr_b8 v[46:47], v7 offset:32 // (op 1016) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v7 offset:544 // (op 1016) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v7 offset:4128 // (op 1016) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v7 offset:4640 // (op 1016) Transpose load local data 


ds_read_u8 v23, v0 offset:24608 // (op 1019) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 3413) 



ds_read_b128 v[54:57], v3 offset:8192 // (op 67) Load local data 
ds_read_b128 v[58:61], v3 offset:8256 // (op 67) Load local data 
s_and_b32 s86, s7, 4294967294 // (op 984) 
s_cmp_lt_u32 s1, s86 // (op 984) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoop_984 // (op 984) Condition: Bottom (jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoop_984 if true)
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_984:












s_barrier  // (op 3701) 
s_and_b32 s54, s7, 4294967294 // (op 2055) 
v_lshrrev_b32 v25, 3, v1 // (op 2055) 
s_add_u32 s55, s0, s54 // (op 2055) 

v_and_b32 v26, -8, v1 // (op 2055) 
v_sub_u32 v27, v1, v26 // (op 2055) 

s_mul_i32 s54, s16, s8 // (op 2055) 
s_add_u32 s54, s54, s55 // (op 2055) 

v_lshlrev_b32 v26, 4, v27 // (op 2055) 

s_mul_hi_u32 s55, s54, s13 // (op 2055) 
s_sub_u32 s84, s54, s55 // (op 2055) 
s_lshr_b32 s85, s84, 1 // (op 2055) 

s_add_u32 s84, s85, s55 // (op 2055) 


s_lshr_b32 s55, s84, s11 // (op 2055) 

s_and_b32 s84, s11, 2147483648 // (op 2055) 
s_cmp_eq_i32 s84, 0 // (op 2055) 
s_mov_b32 s85, scc // (op 2055) 

s_cmp_eq_i32 s85, 0 // (op 2055) 
s_cselect_b32 s84, s54, s55 // (op 2055) 


s_mul_hi_u32 s55, s84, s12 // (op 2055) 
s_sub_u32 s85, s84, s55 // (op 2055) 
s_lshr_b32 s86, s85, 1 // (op 2055) 

s_add_u32 s85, s86, s55 // (op 2055) 


s_lshr_b32 s55, s85, s10 // (op 2055) 

s_and_b32 s85, s10, 2147483648 // (op 2055) 
s_cmp_eq_i32 s85, 0 // (op 2055) 
s_mov_b32 s86, scc // (op 2055) 

s_cmp_eq_i32 s86, 0 // (op 2055) 
s_cselect_b32 s85, s84, s55 // (op 2055) 


v_lshl_add_u32 v27, s85, 5, v25 // (op 2055) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 2055) 
s_mul_i32 s55, s84, s86 // (op 2055) 


s_sub_u32 s84, s54, s55 // (op 2055) 


v_lshl_add_u32 v25, s84, 7, v26 // (op 2055) 


v_mov_b32 v37, 0 // (op 2055) convert
v_mov_b32 v36, v25 // (op 2055) convert

v_mul_lo_u32 v29, v27, s31 // (op 2055) most significant: low of low * high
v_mul_hi_u32 v25, v27, s30 // (op 2055) most significant: high of low * low
v_mul_lo_u32 v28, v27, s30 // (op 2055) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 2055) most significant: sum

v_add_co_u32 v28, s[54:55], v28, v36 // (op 2055) least significant half
v_addc_co_u32 v29, s[54:55], v29, v37, s[54:55] // (op 2055) most significant half



s_mul_i32 s55, s30, s35 // (op 2057) most significant: low of low * high
s_mul_i32 s84, s31, s34 // (op 2057) most significant: low of high * low
s_mul_i32 s54, s30, s34 // (op 2057) least significant
s_mul_hi_u32 s85, s30, s34 // (op 2057) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 2057) most significant: sum
s_add_u32 s55, s55, s85 // (op 2057) most significant: sum


s_mov_b32 s68, s32 // (op 2057) reg expression
s_mov_b32 s69, s33 // (op 2057) reg expression
s_mov_b32 s70, s54 // (op 2057) reg expression
s_mov_b32 s71, 131072 // (op 2057) call()

s_and_b32 s54, s7, 4294967294 // (op 2191) 
v_and_b32 v25, -4, v1 // (op 2191) 
s_add_u32 s55, s0, s54 // (op 2191) 

v_sub_u32 v26, v1, v25 // (op 2191) 

v_lshlrev_b32 v25, 4, v26 // (op 2191) 

s_mul_i32 s54, s16, s8 // (op 2191) 
s_add_u32 s54, s54, s55 // (op 2191) 

v_lshrrev_b32 v26, 2, v1 // (op 2191) 
s_mul_hi_u32 s55, s54, s13 // (op 2191) 
s_sub_u32 s84, s54, s55 // (op 2191) 
s_lshr_b32 s85, s84, 1 // (op 2191) 

s_add_u32 s84, s85, s55 // (op 2191) 


s_lshr_b32 s55, s84, s11 // (op 2191) 

s_and_b32 s84, s11, 2147483648 // (op 2191) 
s_cmp_eq_i32 s84, 0 // (op 2191) 
s_mov_b32 s85, scc // (op 2191) 

s_cmp_eq_i32 s85, 0 // (op 2191) 
s_cselect_b32 s84, s54, s55 // (op 2191) 


s_mul_hi_u32 s55, s84, s12 // (op 2191) 
s_sub_u32 s85, s84, s55 // (op 2191) 
s_lshr_b32 s86, s85, 1 // (op 2191) 

s_add_u32 s85, s86, s55 // (op 2191) 


s_lshr_b32 s55, s85, s10 // (op 2191) 

s_and_b32 s85, s10, 2147483648 // (op 2191) 
s_cmp_eq_i32 s85, 0 // (op 2191) 
s_mov_b32 s86, scc // (op 2191) 

s_cmp_eq_i32 s86, 0 // (op 2191) 
s_cselect_b32 s85, s84, s55 // (op 2191) 


s_mul_i32 s55, s85, s14 // (op 2191) 

s_sub_u32 s85, s84, s55 // (op 2191) 

v_lshl_add_u32 v27, s85, 6, v25 // (op 2191) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 2191) 
v_mov_b32 v37, 0 // (op 2191) convert
v_mov_b32 v36, v27 // (op 2191) convert

s_mul_i32 s55, s84, s86 // (op 2191) 


s_sub_u32 s84, s54, s55 // (op 2191) 


v_lshl_add_u32 v25, s84, 7, v26 // (op 2191) 


v_mul_lo_u32 v27, v25, s37 // (op 2191) most significant: low of low * high
v_mul_hi_u32 v30, v25, s36 // (op 2191) most significant: high of low * low
v_mul_lo_u32 v26, v25, s36 // (op 2191) least significant: low of low * low
v_add_u32 v27, v27, v30 // (op 2191) most significant: sum

v_add_co_u32 v26, s[54:55], v26, v36 // (op 2191) least significant half
v_addc_co_u32 v27, s[54:55], v27, v37, s[54:55] // (op 2191) most significant half



s_mul_i32 s55, s36, s3 // (op 2193) most significant: low of low * high
s_mul_i32 s84, s37, s2 // (op 2193) most significant: low of high * low
s_mul_i32 s54, s36, s2 // (op 2193) least significant
s_mul_hi_u32 s85, s36, s2 // (op 2193) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 2193) most significant: sum
s_add_u32 s55, s55, s85 // (op 2193) most significant: sum


s_mov_b32 s72, s38 // (op 2193) reg expression
s_mov_b32 s73, s39 // (op 2193) reg expression
s_mov_b32 s74, s54 // (op 2193) reg expression
s_mov_b32 s75, 131072 // (op 2193) call()

s_and_b32 s54, s7, 4294967294 // (op 2319) 
v_lshrrev_b32 v25, 7, v1 // (op 2319) 
s_add_u32 s55, s0, s54 // (op 2319) 

v_and_b32 v30, 15, v1 // (op 2319) 
v_lshl_add_u32 v31, v25, 4, v30 // (op 2319) 


s_mul_i32 s54, s16, s8 // (op 2319) 
s_add_u32 s54, s54, s55 // (op 2319) 

v_and_b32 v25, 63, v1 // (op 2319) 
s_mul_hi_u32 s55, s54, s13 // (op 2319) 
v_lshrrev_b32 v30, 4, v25 // (op 2319) 

s_sub_u32 s84, s54, s55 // (op 2319) 
s_lshr_b32 s85, s84, 1 // (op 2319) 

s_add_u32 s84, s85, s55 // (op 2319) 


s_lshr_b32 s55, s84, s11 // (op 2319) 

s_and_b32 s84, s11, 2147483648 // (op 2319) 
s_cmp_eq_i32 s84, 0 // (op 2319) 
s_mov_b32 s85, scc // (op 2319) 

s_cmp_eq_i32 s85, 0 // (op 2319) 
s_cselect_b32 s84, s54, s55 // (op 2319) 


s_mul_hi_u32 s55, s84, s12 // (op 2319) 
s_sub_u32 s85, s84, s55 // (op 2319) 
s_lshr_b32 s86, s85, 1 // (op 2319) 

s_add_u32 s85, s86, s55 // (op 2319) 


s_lshr_b32 s55, s85, s10 // (op 2319) 

s_and_b32 s85, s10, 2147483648 // (op 2319) 
s_cmp_eq_i32 s85, 0 // (op 2319) 
s_mov_b32 s86, scc // (op 2319) 

s_cmp_eq_i32 s86, 0 // (op 2319) 
s_cselect_b32 s85, s84, s55 // (op 2319) 


v_lshl_add_u32 v25, s85, 5, v31 // (op 2319) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 2319) 
s_mul_i32 s55, s84, s86 // (op 2319) 


s_sub_u32 s84, s54, s55 // (op 2319) 


v_lshl_add_u32 v31, s84, 2, v30 // (op 2319) 


v_mul_lo_u32 v30, v25, s40 // (op 2319) 
v_add_u32 v30, v30, v31 // (op 2319) 


s_mul_i32 s55, s40, s35 // (op 2321) most significant: low of low * high
s_mul_i32 s84, s41, s34 // (op 2321) most significant: low of high * low
s_mul_i32 s54, s40, s34 // (op 2321) least significant
s_mul_hi_u32 s85, s40, s34 // (op 2321) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 2321) most significant: sum
s_add_u32 s55, s55, s85 // (op 2321) most significant: sum


s_mov_b32 s76, s42 // (op 2321) reg expression
s_mov_b32 s77, s43 // (op 2321) reg expression
s_mov_b32 s78, s54 // (op 2321) reg expression
s_mov_b32 s79, 131072 // (op 2321) call()

s_and_b32 s54, s7, 4294967294 // (op 2447) 
v_and_b32 v25, -64, v1 // (op 2447) 
s_add_u32 s55, s0, s54 // (op 2447) 

v_sub_u32 v31, v1, v25 // (op 2447) 

v_lshrrev_b32 v25, 6, v1 // (op 2447) 
s_mul_i32 s54, s16, s8 // (op 2447) 
s_add_u32 s54, s54, s55 // (op 2447) 

s_mul_hi_u32 s55, s54, s13 // (op 2447) 
s_sub_u32 s84, s54, s55 // (op 2447) 
s_lshr_b32 s85, s84, 1 // (op 2447) 

s_add_u32 s84, s85, s55 // (op 2447) 


s_lshr_b32 s55, s84, s11 // (op 2447) 

s_and_b32 s84, s11, 2147483648 // (op 2447) 
s_cmp_eq_i32 s84, 0 // (op 2447) 
s_mov_b32 s85, scc // (op 2447) 

s_cmp_eq_i32 s85, 0 // (op 2447) 
s_cselect_b32 s84, s54, s55 // (op 2447) 


s_mul_hi_u32 s55, s84, s12 // (op 2447) 
s_sub_u32 s85, s84, s55 // (op 2447) 
s_lshr_b32 s86, s85, 1 // (op 2447) 

s_add_u32 s85, s86, s55 // (op 2447) 


s_lshr_b32 s55, s85, s10 // (op 2447) 

s_and_b32 s85, s10, 2147483648 // (op 2447) 
s_cmp_eq_i32 s85, 0 // (op 2447) 
s_mov_b32 s86, scc // (op 2447) 

s_cmp_eq_i32 s86, 0 // (op 2447) 
s_cselect_b32 s85, s84, s55 // (op 2447) 


s_mul_i32 s55, s85, s14 // (op 2447) 

s_sub_u32 s85, s84, s55 // (op 2447) 

v_lshl_add_u32 v32, s85, 6, v31 // (op 2447) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 2447) 
s_mul_i32 s55, s84, s86 // (op 2447) 


s_sub_u32 s84, s54, s55 // (op 2447) 


v_lshl_add_u32 v31, s84, 2, v25 // (op 2447) 


v_mul_lo_u32 v25, v31, s44 // (op 2447) 
v_add_u32 v25, v25, v32 // (op 2447) 


s_mul_i32 s55, s44, s49 // (op 2449) most significant: low of low * high
s_mul_i32 s84, s45, s48 // (op 2449) most significant: low of high * low
s_mul_i32 s54, s44, s48 // (op 2449) least significant
s_mul_hi_u32 s85, s44, s48 // (op 2449) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 2449) most significant: sum
s_add_u32 s55, s55, s85 // (op 2449) most significant: sum


s_mov_b32 s80, s46 // (op 2449) reg expression
s_mov_b32 s81, s47 // (op 2449) reg expression
s_mov_b32 s82, s54 // (op 2449) reg expression
s_mov_b32 s83, 131072 // (op 2449) call()

v_and_b32 v31, -8, v1 // (op 2608) 
v_sub_u32 v32, v1, v31 // (op 2608) 

v_lshlrev_b32 v31, 4, v32 // (op 2608) 

v_lshrrev_b32 v32, 3, v1 // (op 2608) 
v_lshl_add_u32 v33, v32, 7, v31 // (op 2608) 


s_nop 0
v_readfirstlane_b32 s54, v33 // (op 2608) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v31, -4, v1 // (op 2673) 
v_sub_u32 v32, v1, v31 // (op 2673) 

v_lshlrev_b32 v31, 4, v32 // (op 2673) 

v_lshrrev_b32 v32, 2, v1 // (op 2673) 
v_lshl_add_u32 v33, v32, 6, v31 // (op 2673) 


s_nop 0
v_readfirstlane_b32 s55, v33 // (op 2673) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v31, -64, v1 // (op 2748) 
v_sub_u32 v32, v1, v31 // (op 2748) 

v_and_b32 v31, 4294967232, v1 // (op 2748) 
v_add_u32 v33, v31, v32 // (op 2748) 


s_and_b32 s84, s7, 4294967294 // (op 990) 
s_cmp_lt_u32 s84, s7 // (op 989) 







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_989 // (op 989) Condition: Top (jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_989 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_989:
s_barrier  // (op 3410) 
s_add_u32 s84, s84, 1 // (op 991) 


v_mov_b64 v[34:35], v[28:29] // (op 1969) 
s_add_u32 s85, 8192, s54 // (op 1969) 

s_mov_b32 m0, s85 // (op 1969) reg expression
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 1969) Load value direct to lds


v_add_co_u32 v28, vcc, 128, v28 // (op 2045) least significant half
v_addc_co_u32 v29, vcc, 0, v29, vcc // (op 2045) most significant half
s_barrier  // (op 3429) 

s_lshl_b64 s[86:87], s[36:37], 6 // (op 1975) 
v_mov_b64 v[34:35], s[86:87] // (op 1975) convert


v_mov_b64 v[36:37], v[26:27] // (op 1975) 
v_mov_b64 v[38:39], v[36:37] // (op 1975) reg expression
s_mov_b32 m0, s55 // (op 1975) reg expression
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 1975) Load value direct to lds
v_add_co_u32 v38, s[86:87], v38, v34 // (op 1975) least significant half
v_addc_co_u32 v39, s[86:87], 0, v39, s[86:87] // (op 1975) most significant half

s_add_u32 m0, m0, 4096 // (op 1975) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 1975) Load value direct to lds



s_lshl_b64 s[86:87], s[36:37], 7 // (op 2183) 
v_mov_b32 v21, s86 // (op 2183) 
v_mov_b32 v23, s87 // (op 2183) 
v_add_co_u32 v26, s[88:89], v26, v21 // (op 2183) least significant half
v_addc_co_u32 v27, s[88:89], v27, v23, s[88:89] // (op 2183) most significant half






v_mov_b32 v21, v25 // (op 857) 
v_mov_b32 v23, v21 // (op 857) reg expression

buffer_load_ubyte v24, v23, s[80:83], 0 offen offset: 0 // (op 857) Load value


s_lshl_b32 s85, s44, 2 // (op 2439) 
v_add_u32 v25, v25, s85 // (op 2439) 

s_barrier  // (op 3448) 
s_waitcnt vmcnt(3)
s_barrier  // (op 3416) 






ds_read_b128 v[34:37], v3 offset:8192 // (op 846) Load local data 
ds_read_b128 v[38:41], v3 offset:8256 // (op 846) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v33, v24 offset:24576 // (op 855) Store local data 

s_barrier  // (op 3435) 



ds_read_b64_tr_b8 v[42:43], v7 // (op 848) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:512 // (op 848) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:4096 // (op 848) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v7 offset:4608 // (op 848) Transpose load local data 


v_mov_b32 v21, v30 // (op 850) 

buffer_load_ubyte v23, v21, s[76:79], 0 offen offset: 0 // (op 850) Load value

v_add_u32 v30, 4, v30 // (op 2311) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3455) 


ds_read_u8 v21, v0 offset:24576 // (op 852) Load local data 



ds_read_b64_tr_b8 v[50:51], v7 offset:32 // (op 997) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[42:49], a[0:3], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 853) 
ds_read_b64_tr_b8 v[52:53], v7 offset:544 // (op 997) Transpose load local data 


ds_read_b64_tr_b8 v[54:55], v7 offset:4128 // (op 997) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v7 offset:4640 // (op 997) Transpose load local data 


ds_read_u8 v21, v0 offset:24608 // (op 1000) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:41], v[50:57], a[4:7], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1001) 




s_cmp_lt_u32 s84, s7 // (op 989) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_989 // (op 989) Condition: Bottom (jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_989 if true)
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_989:








s_barrier  // (op 3705) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 280) 
s_cmp_lg_u32 s18, 0 // (op 306) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Send_Tile_306 // (op 306) Condition: False, jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Send_Tile_306



buffer_store_dword a0, v20, s[56:59], 0 offen sc1 // (op 303) Store value
buffer_store_dword a1, v20, s[56:59], 0 offen offset:256 sc1 // (op 303) Store value
buffer_store_dword a2, v20, s[56:59], 0 offen offset:512 sc1 // (op 303) Store value
buffer_store_dword a3, v20, s[56:59], 0 offen offset:768 sc1 // (op 303) Store value



buffer_store_dword a4, v20, s[56:59], 0 offen offset:128 sc1 // (op 1180) Store value
buffer_store_dword a5, v20, s[56:59], 0 offen offset:384 sc1 // (op 1180) Store value
buffer_store_dword a6, v20, s[56:59], 0 offen offset:640 sc1 // (op 1180) Store value
buffer_store_dword a7, v20, s[56:59], 0 offen offset:896 sc1 // (op 1180) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 307) 
v_cmp_eq_i32 vcc, v1, 0 // (op 311) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_311 // (op 311) Condition: False, jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_311
s_mov_b32 s54, 1 // (op 309) call()
s_mov_b32 s85, 0 // (op 310) convert
s_mov_b32 s84, s16 // (op 310) convert
s_lshl_b64 s[84:85], s[84:85], 2 // (op 310) 
s_add_u32 s86, s50, s84 // (op 310) Start of Int64 add, locking SCC
s_addc_u32 s87, s51, s85 // (op 310) End of Int64 add, unlocking SCC
s_store_dword s54, s[86:87], 0 glc // (op 310) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_311 // (op 311) Condition: Done, jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_311
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_311:
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_311:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Send_Tile_306 // (op 306) Condition: Done, jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Send_Tile_306
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Send_Tile_306:
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Send_Tile_306:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 322) 
s_mov_b32 s54, scc // (op 322) 
s_ashr_i64 s[84:85], s[2:3], 7 // (op 322) 
s_mov_b32 s55, s84 // (op 322) convert

s_sub_u32 s84, s55, 1 // (op 322) 

s_cmp_lt_u32 s19, s84 // (op 322) 
s_mov_b32 s55, scc // (op 322) 

s_and_b32 s84, s54, s55 // (op 322) 
s_cmp_lg_u32 s84, 0 // (op 322) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Receive_Tile_322 // (op 322) Condition: False, jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Receive_Tile_322
s_mov_b32 s54, 0 // (op 288) call()
s_mul_i32 s55, s54, s8 // (op 287) 
s_ashr_i64 s[84:85], s[2:3], 7 // (op 287) 
s_mov_b32 s86, s84 // (op 287) convert

s_sub_u32 s84, s86, s19 // (op 287) 

s_sub_u32 s85, s84, 1 // (op 287) 

s_cmp_lt_u32 s55, s85 // (op 287) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_287 // (op 287) Condition: Top (jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_287 if false)
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_ReceiveTileLoop_287:
s_add_u32 s55, s16, 1 // (op 3096) 
v_lshrrev_b32 v16, 5, v1 // (op 3096) 
s_add_u32 s84, s55, s54 // (op 3096) 
s_lshl_b32 s84, s84, 5 // (op 3096) 

v_lshlrev_b32 v17, 2, v16 // (op 3096) 

v_add_lshl_u32 v16, s84, v17, 6 // (op 3096) 


v_and_b32 v17, -32, v1 // (op 3096) 
v_sub_u32 v19, v1, v17 // (op 3096) 

v_add_lshl_u32 v17, v16, v19, 2 // (op 3096) 


s_lshl_b32 s55, s15, 13 // (op 3098) 
s_mov_b32 s56, s28 // (op 3098) reg expression
s_mov_b32 s57, s29 // (op 3098) reg expression
s_mov_b32 s58, s55 // (op 3098) reg expression
s_mov_b32 s59, 131072 // (op 3098) call()

RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_326:
s_add_u32 s55, s16, 1 // (op 324) 
s_add_u32 s84, s55, s54 // (op 324) Start of Int64 add, locking SCC
s_addc_u32 s85, 0, 0 // (op 324) End of Int64 add, unlocking SCC
s_lshl_b64 s[84:85], s[84:85], 2 // (op 324) 

s_add_u32 s86, s50, s84 // (op 324) Start of Int64 add, locking SCC
s_addc_u32 s87, s51, s85 // (op 324) End of Int64 add, unlocking SCC
s_load_dword s55, s[86:87], 0 glc // (op 324) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s55, 0 // (op 326) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_326 // (op 326) Condition: Bottom (jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_326 if true)
s_barrier  // (op 330) 
v_cmp_eq_i32 vcc, v1, 0 // (op 329) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_329 // (op 329) Condition: False, jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_329
s_mov_b32 s55, 0 // (op 327) call()
s_add_u32 s84, s16, 1 // (op 328) 
s_add_u32 s86, s84, s54 // (op 328) Start of Int64 add, locking SCC
s_addc_u32 s87, 0, 0 // (op 328) End of Int64 add, unlocking SCC
s_lshl_b64 s[86:87], s[86:87], 2 // (op 328) 

s_add_u32 s84, s50, s86 // (op 328) Start of Int64 add, locking SCC
s_addc_u32 s85, s51, s87 // (op 328) End of Int64 add, unlocking SCC
s_store_dword s55, s[84:85], 0 glc // (op 328) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_329 // (op 329) Condition: Done, jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_329
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_329:
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_329:




buffer_load_dword v24, v17, s[56:59], 0 offen // (op 341) Load value
buffer_load_dword v25, v17, s[56:59], 0 offen offset:256 // (op 341) Load value
buffer_load_dword v26, v17, s[56:59], 0 offen offset:512 // (op 341) Load value
buffer_load_dword v27, v17, s[56:59], 0 offen offset:768 // (op 341) Load value
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




buffer_load_dword v24, v17, s[56:59], 0 offen offset:128 // (op 1190) Load value
buffer_load_dword v25, v17, s[56:59], 0 offen offset:384 // (op 1190) Load value
buffer_load_dword v26, v17, s[56:59], 0 offen offset:640 // (op 1190) Load value
buffer_load_dword v27, v17, s[56:59], 0 offen offset:896 // (op 1190) Load value
v_accvgpr_read v28, a4 // (op 1191) reg expression
v_accvgpr_read v29, a5 // (op 1191) reg expression
v_accvgpr_read v30, a6 // (op 1191) reg expression
v_accvgpr_read v31, a7 // (op 1191) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 1192) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 1192) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 1192) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 1192) 

v_accvgpr_write a4, v28 // (op 1193) reg expression
v_accvgpr_write a5, v29 // (op 1193) reg expression
v_accvgpr_write a6, v30 // (op 1193) reg expression
v_accvgpr_write a7, v31 // (op 1193) reg expression

s_add_u32 s54, s54, 1 // (op 289) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s55, s54, s8 // (op 287) 
s_ashr_i64 s[84:85], s[2:3], 7 // (op 287) 
s_mov_b32 s86, s84 // (op 287) convert

s_sub_u32 s84, s86, s19 // (op 287) 

s_sub_u32 s85, s84, 1 // (op 287) 

s_cmp_lt_u32 s55, s85 // (op 287) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_ReceiveTileLoop_287 // (op 287) Condition: Bottom (jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_ReceiveTileLoop_287 if true)
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_287:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Receive_Tile_322 // (op 322) Condition: Done, jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Receive_Tile_322
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Receive_Tile_322:
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Receive_Tile_322:

s_cmp_eq_i32 s18, 0 // (op 365) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_365 // (op 365) Condition: False, jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_365
v_cmp_eq_f32 vcc, 0.00000, v14 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_125
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

buffer_store_dwordx2 v[16:17], v23, s[64:67], 0 offen // (op 54) Store value



v_accvgpr_read v24, a4 // (op 1209) 
v_accvgpr_read v25, a5 // (op 1209) 
v_accvgpr_read v26, a6 // (op 1209) 
v_accvgpr_read v27, a7 // (op 1209) 
v_mul_f32 v28, v15, v24 // (op 1209) 
v_mul_f32 v29, v15, v25 // (op 1209) 
v_mul_f32 v30, v15, v26 // (op 1209) 
v_mul_f32 v31, v15, v27 // (op 1209) 

v_cvt_f16_f32 v24, v28 // (op 1211) 
v_cvt_f16_f32 v25, v29 // (op 1211) 
v_cvt_f16_f32 v26, v30 // (op 1211) 
v_cvt_f16_f32 v27, v31 // (op 1211) 


v_pack_b32_f16 v16, v24, v25 // (op 1212) 
v_pack_b32_f16 v17, v26, v27 // (op 1212) 

s_lshl_b32 s54, s24, 6 // (op 1212) 
v_add_u32 v23, v22, s54 // (op 1212) 


buffer_store_dwordx2 v[16:17], v23, s[64:67], 0 offen // (op 1212) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_125:


v_mov_b32 v16, v18 // (op 138) reg expression

buffer_load_dwordx2 v[20:21], v16, s[60:63], 0 offen // (op 138) Load value

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

buffer_store_dwordx2 v[16:17], v19, s[64:67], 0 offen // (op 137) Store value





s_lshl_b32 s54, s4, 6 // (op 1236) 
v_add_u32 v16, v18, s54 // (op 1236) 


buffer_load_dwordx2 v[20:21], v16, s[60:63], 0 offen // (op 1236) Load value

v_accvgpr_read v24, a4 // (op 1232) 
v_accvgpr_read v25, a5 // (op 1232) 
v_accvgpr_read v26, a6 // (op 1232) 
v_accvgpr_read v27, a7 // (op 1232) 
v_mul_f32 v28, v15, v24 // (op 1232) 
v_mul_f32 v29, v15, v25 // (op 1232) 
v_mul_f32 v30, v15, v26 // (op 1232) 
v_mul_f32 v31, v15, v27 // (op 1232) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v20 // (op 1237) 
v_lshrrev_b32 v17, 16, v20 // (op 1237) 
v_cvt_f32_f16 v17, v17 // (op 1237) 
v_mul_f32 v24, v14, v16 // (op 1237) 
v_mul_f32 v25, v14, v17 // (op 1237) 
v_cvt_f32_f16 v16, v21 // (op 1237) 
v_lshrrev_b32 v17, 16, v21 // (op 1237) 
v_cvt_f32_f16 v17, v17 // (op 1237) 
v_mul_f32 v26, v14, v16 // (op 1237) 
v_mul_f32 v27, v14, v17 // (op 1237) 


v_add_f32 v32, v24, v28 // (op 1233) 
v_add_f32 v33, v25, v29 // (op 1233) 
v_add_f32 v34, v26, v30 // (op 1233) 
v_add_f32 v35, v27, v31 // (op 1233) 


v_cvt_f16_f32 v28, v32 // (op 1234) 
v_cvt_f16_f32 v29, v33 // (op 1234) 
v_cvt_f16_f32 v30, v34 // (op 1234) 
v_cvt_f16_f32 v31, v35 // (op 1234) 


v_pack_b32_f16 v16, v28, v29 // (op 1235) 
v_pack_b32_f16 v17, v30, v31 // (op 1235) 

s_lshl_b32 s54, s24, 6 // (op 1235) 
v_add_u32 v19, v22, s54 // (op 1235) 


buffer_store_dwordx2 v[16:17], v19, s[64:67], 0 offen // (op 1235) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_125:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_365 // (op 365) Condition: Done, jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_365
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_365:
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_365:


s_mul_i32 s54, s8, s16 // (op 278) 
s_cmp_gt_u32 s9, s54 // (op 278) 
s_mov_b32 s55, scc // (op 278) 
s_sub_u32 s56, s9, s54 // (op 278) 

s_cmp_eq_i32 s55, 0 // (op 278) 
s_cselect_b32 s54, 0, s56 // (op 278) 


s_cmp_lt_u32 s8, s54 // (op 278) 
s_mov_b32 s55, scc // (op 278) 
s_cmp_eq_i32 s55, 0 // (op 278) 
s_cselect_b32 s56, s54, s8 // (op 278) 


s_cmp_lt_u32 s0, s56 // (op 278) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_SKStreamTileLoop_278 // (op 278) Condition: Bottom (jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_SKStreamTileLoop_278 if true)
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_278:















s_mov_b32 s0, 0 // (op 419) call()
s_mov_b32 s1, 0 // (op 420) call()
s_mov_b32 s0, 0 // (op 434) call()
s_mul_i32 s15, s52, s16 // (op 433) 
s_cmp_gt_u32 s53, s15 // (op 433) 
s_mov_b32 s6, scc // (op 433) 
s_sub_u32 s7, s53, s15 // (op 433) 

s_cmp_eq_i32 s6, 0 // (op 433) 
s_cselect_b32 s8, 0, s7 // (op 433) 


s_cmp_lt_u32 s52, s8 // (op 433) 
s_mov_b32 s15, scc // (op 433) 
s_cmp_eq_i32 s15, 0 // (op 433) 
s_cselect_b32 s6, s8, s52 // (op 433) 


s_cmp_lt_u32 s0, s6 // (op 433) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_433 // (op 433) Condition: Top (jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_433 if false)
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_DPStreamTileLoop_433:
v_lshrrev_b32 v0, 7, v1 // (op 2578) 
s_mul_i32 s15, s16, s52 // (op 2578) 
s_add_u32 s15, s15, s0 // (op 2578) 
v_and_b32 v3, 63, v1 // (op 2578) 
s_add_u32 s6, s15, s9 // (op 2578) 

v_lshrrev_b32 v6, 4, v3 // (op 2578) 
s_mul_hi_u32 s15, s6, s13 // (op 2578) 
v_lshlrev_b32 v7, 2, v6 // (op 2578) 

s_sub_u32 s7, s6, s15 // (op 2578) 
v_lshl_add_u32 v6, v0, 4, v7 // (op 2578) 


s_lshr_b32 s8, s7, 1 // (op 2578) 

v_lshrrev_b32 v0, 2, v3 // (op 2578) 

s_add_u32 s7, s8, s15 // (op 2578) 


v_and_b32 v3, 3, v0 // (op 2578) 

s_lshr_b32 s8, s7, s11 // (op 2578) 

v_and_b32 v0, 3, v1 // (op 2578) 
s_and_b32 s7, s11, 2147483648 // (op 2578) 
v_lshl_add_u32 v7, v3, 2, v0 // (op 2578) 


s_cmp_eq_i32 s7, 0 // (op 2578) 
s_mov_b32 s15, scc // (op 2578) 

v_lshrrev_b32 v0, 6, v1 // (op 2578) 
s_cmp_eq_i32 s15, 0 // (op 2578) 
s_cselect_b32 s7, s6, s8 // (op 2578) 



v_and_b32 v3, 1, v0 // (op 2578) 

s_mul_hi_u32 s6, s7, s12 // (op 2578) 
v_lshl_add_u32 v0, v3, 4, v7 // (op 2578) 


s_sub_u32 s8, s7, s6 // (op 2578) 
s_lshr_b32 s15, s8, 1 // (op 2578) 

s_add_u32 s8, s15, s6 // (op 2578) 


s_lshr_b32 s6, s8, s10 // (op 2578) 

s_and_b32 s8, s10, 2147483648 // (op 2578) 
s_cmp_eq_i32 s8, 0 // (op 2578) 
s_mov_b32 s15, scc // (op 2578) 

s_cmp_eq_i32 s15, 0 // (op 2578) 
s_cselect_b32 s8, s7, s6 // (op 2578) 


s_mul_i32 s6, s8, s14 // (op 2578) 
v_lshl_add_u32 v3, s8, 5, v6 // (op 2578) 


s_sub_u32 s8, s7, s6 // (op 2578) 


v_lshl_add_u32 v6, s8, 6, v0 // (op 2578) 


v_mul_lo_u32 v0, v6, s4 // (op 2578) 

v_add_lshl_u32 v6, v3, v0, 1 // (op 2578) 


s_mul_i32 s19, s4, s23 // (op 2580) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 2580) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 2580) least significant
s_mul_hi_u32 s6, s4, s22 // (op 2580) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2580) most significant: sum
s_add_u32 s19, s19, s6 // (op 2580) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 2580) 

s_mov_b32 s60, s20 // (op 2580) reg expression
s_mov_b32 s61, s21 // (op 2580) reg expression
s_mov_b32 s62, s28 // (op 2580) reg expression
s_mov_b32 s63, 131072 // (op 2580) call()

v_lshrrev_b32 v0, 7, v1 // (op 2836) 
s_mul_i32 s15, s16, s52 // (op 2836) 
s_add_u32 s15, s15, s0 // (op 2836) 
v_and_b32 v3, 63, v1 // (op 2836) 
s_add_u32 s6, s15, s9 // (op 2836) 

v_lshrrev_b32 v7, 4, v3 // (op 2836) 
s_mul_hi_u32 s15, s6, s13 // (op 2836) 
v_lshlrev_b32 v12, 2, v7 // (op 2836) 

s_sub_u32 s7, s6, s15 // (op 2836) 
v_lshl_add_u32 v7, v0, 4, v12 // (op 2836) 


s_lshr_b32 s8, s7, 1 // (op 2836) 

v_lshrrev_b32 v0, 2, v3 // (op 2836) 

s_add_u32 s7, s8, s15 // (op 2836) 


v_and_b32 v3, 3, v0 // (op 2836) 

s_lshr_b32 s8, s7, s11 // (op 2836) 

v_and_b32 v0, 3, v1 // (op 2836) 
s_and_b32 s7, s11, 2147483648 // (op 2836) 
v_lshl_add_u32 v12, v3, 2, v0 // (op 2836) 


s_cmp_eq_i32 s7, 0 // (op 2836) 
s_mov_b32 s15, scc // (op 2836) 

v_lshrrev_b32 v0, 6, v1 // (op 2836) 
s_cmp_eq_i32 s15, 0 // (op 2836) 
s_cselect_b32 s7, s6, s8 // (op 2836) 



v_and_b32 v3, 1, v0 // (op 2836) 

s_mul_hi_u32 s6, s7, s12 // (op 2836) 
v_lshl_add_u32 v0, v3, 4, v12 // (op 2836) 


s_sub_u32 s8, s7, s6 // (op 2836) 
s_lshr_b32 s15, s8, 1 // (op 2836) 

s_add_u32 s8, s15, s6 // (op 2836) 


s_lshr_b32 s6, s8, s10 // (op 2836) 

s_and_b32 s8, s10, 2147483648 // (op 2836) 
s_cmp_eq_i32 s8, 0 // (op 2836) 
s_mov_b32 s15, scc // (op 2836) 

s_cmp_eq_i32 s15, 0 // (op 2836) 
s_cselect_b32 s8, s7, s6 // (op 2836) 


s_mul_i32 s6, s8, s14 // (op 2836) 
v_lshl_add_u32 v3, s8, 5, v7 // (op 2836) 


s_sub_u32 s8, s7, s6 // (op 2836) 


v_lshl_add_u32 v7, s8, 6, v0 // (op 2836) 


v_mul_lo_u32 v0, v7, s24 // (op 2836) 

v_add_lshl_u32 v7, v3, v0, 1 // (op 2836) 


s_mul_i32 s19, s24, s23 // (op 2838) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 2838) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 2838) least significant
s_mul_hi_u32 s6, s24, s22 // (op 2838) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2838) most significant: sum
s_add_u32 s19, s19, s6 // (op 2838) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 2838) 

s_mov_b32 s64, s26 // (op 2838) reg expression
s_mov_b32 s65, s27 // (op 2838) reg expression
s_mov_b32 s66, s28 // (op 2838) reg expression
s_mov_b32 s67, 131072 // (op 2838) call()

v_lshrrev_b32 v0, 7, v1 // (op 2905) 
v_and_b32 v3, 15, v1 // (op 2905) 
v_lshl_add_u32 v12, v0, 4, v3 // (op 2905) 


v_and_b32 v0, 48, v1 // (op 2905) 
v_lshl_add_u32 v3, v12, 7, v0 // (op 2905) 


v_and_b32 v0, 48, v1 // (op 2980) 
v_and_b32 v12, 63, v1 // (op 2980) 
v_sub_u32 v13, v12, v0 // (op 2980) 

v_lshrrev_b32 v12, 1, v13 // (op 2980) 
v_add_lshl_u32 v16, v0, v12, 6 // (op 2980) 


v_and_b32 v0, 1, v13 // (op 2980) 

v_lshlrev_b32 v12, 3, v0 // (op 2980) 

v_lshrrev_b32 v0, 6, v1 // (op 2980) 
v_and_b32 v13, 1, v0 // (op 2980) 

v_lshl_add_u32 v0, v13, 4, v12 // (op 2980) 


v_add_u32 v12, v16, v0 // (op 2980) 


v_lshrrev_b32 v0, 6, v1 // (op 3056) 
v_and_b32 v13, 1, v0 // (op 3056) 

v_and_b32 v0, 15, v1 // (op 3056) 
v_lshl_add_u32 v16, v13, 4, v0 // (op 3056) 


v_and_b32 v0, 63, v1 // (op 3056) 
v_lshrrev_b32 v13, 4, v0 // (op 3056) 

v_lshl_add_u32 v0, v13, 6, v16 // (op 3056) 


v_accvgpr_write a0, 0.00000 // (op 174) call()
v_accvgpr_write a1, 0.00000 // (op 174) call()
v_accvgpr_write a2, 0.00000 // (op 174) call()
v_accvgpr_write a3, 0.00000 // (op 174) call()
v_accvgpr_write a4, 0.00000 // (op 586) call()
v_accvgpr_write a5, 0.00000 // (op 586) call()
v_accvgpr_write a6, 0.00000 // (op 586) call()
v_accvgpr_write a7, 0.00000 // (op 586) call()
v_lshrrev_b32 v13, 3, v1 // (op 2082) 
s_mul_i32 s15, s16, s52 // (op 2082) 
s_add_u32 s15, s15, s0 // (op 2082) 
v_and_b32 v16, -8, v1 // (op 2082) 
s_add_u32 s6, s15, s9 // (op 2082) 

v_sub_u32 v17, v1, v16 // (op 2082) 

s_mul_hi_u32 s15, s6, s13 // (op 2082) 
v_lshlrev_b32 v16, 4, v17 // (op 2082) 

s_sub_u32 s7, s6, s15 // (op 2082) 
s_lshr_b32 s8, s7, 1 // (op 2082) 

s_add_u32 s7, s8, s15 // (op 2082) 


s_lshr_b32 s8, s7, s11 // (op 2082) 

s_and_b32 s7, s11, 2147483648 // (op 2082) 
s_cmp_eq_i32 s7, 0 // (op 2082) 
s_mov_b32 s15, scc // (op 2082) 

s_cmp_eq_i32 s15, 0 // (op 2082) 
s_cselect_b32 s7, s6, s8 // (op 2082) 


s_mul_hi_u32 s8, s7, s12 // (op 2082) 
s_sub_u32 s15, s7, s8 // (op 2082) 
s_lshr_b32 s18, s15, 1 // (op 2082) 

s_add_u32 s15, s18, s8 // (op 2082) 


s_lshr_b32 s8, s15, s10 // (op 2082) 

s_and_b32 s15, s10, 2147483648 // (op 2082) 
s_cmp_eq_i32 s15, 0 // (op 2082) 
s_mov_b32 s18, scc // (op 2082) 

s_cmp_eq_i32 s18, 0 // (op 2082) 
s_cselect_b32 s15, s7, s8 // (op 2082) 


v_lshl_add_u32 v17, s15, 5, v13 // (op 2082) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2082) 
s_mul_i32 s8, s7, s18 // (op 2082) 


s_sub_u32 s7, s6, s8 // (op 2082) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 2082) 


v_mov_b32 v19, 0 // (op 2082) convert
v_mov_b32 v18, v13 // (op 2082) convert

v_mul_lo_u32 v21, v17, s31 // (op 2082) most significant: low of low * high
v_mul_hi_u32 v13, v17, s30 // (op 2082) most significant: high of low * low
v_mul_lo_u32 v20, v17, s30 // (op 2082) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 2082) most significant: sum

v_add_co_u32 v20, s[18:19], v20, v18 // (op 2082) least significant half
v_addc_co_u32 v21, s[18:19], v21, v19, s[18:19] // (op 2082) most significant half



s_mul_i32 s19, s30, s35 // (op 2084) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 2084) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 2084) least significant
s_mul_hi_u32 s6, s30, s34 // (op 2084) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2084) most significant: sum
s_add_u32 s19, s19, s6 // (op 2084) most significant: sum


s_mov_b32 s68, s32 // (op 2084) reg expression
s_mov_b32 s69, s33 // (op 2084) reg expression
s_mov_b32 s70, s18 // (op 2084) reg expression
s_mov_b32 s71, 131072 // (op 2084) call()

v_lshrrev_b32 v13, 3, v1 // (op 2101) 
s_mul_i32 s15, s16, s52 // (op 2101) 
s_add_u32 s15, s15, s0 // (op 2101) 
v_and_b32 v16, -8, v1 // (op 2101) 
s_add_u32 s6, s15, s9 // (op 2101) 

v_sub_u32 v17, v1, v16 // (op 2101) 

s_mul_hi_u32 s15, s6, s13 // (op 2101) 
v_lshlrev_b32 v16, 4, v17 // (op 2101) 

s_sub_u32 s7, s6, s15 // (op 2101) 
s_lshr_b32 s8, s7, 1 // (op 2101) 

s_add_u32 s7, s8, s15 // (op 2101) 


s_lshr_b32 s8, s7, s11 // (op 2101) 

s_and_b32 s7, s11, 2147483648 // (op 2101) 
s_cmp_eq_i32 s7, 0 // (op 2101) 
s_mov_b32 s15, scc // (op 2101) 

s_cmp_eq_i32 s15, 0 // (op 2101) 
s_cselect_b32 s7, s6, s8 // (op 2101) 


s_mul_hi_u32 s8, s7, s12 // (op 2101) 
s_sub_u32 s15, s7, s8 // (op 2101) 
s_lshr_b32 s18, s15, 1 // (op 2101) 

s_add_u32 s15, s18, s8 // (op 2101) 


s_lshr_b32 s8, s15, s10 // (op 2101) 

s_and_b32 s15, s10, 2147483648 // (op 2101) 
s_cmp_eq_i32 s15, 0 // (op 2101) 
s_mov_b32 s18, scc // (op 2101) 

s_cmp_eq_i32 s18, 0 // (op 2101) 
s_cselect_b32 s15, s7, s8 // (op 2101) 


v_lshl_add_u32 v17, s15, 5, v13 // (op 2101) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2101) 
s_mul_i32 s8, s7, s18 // (op 2101) 


s_sub_u32 s7, s6, s8 // (op 2101) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 2101) 


v_mov_b32 v19, 0 // (op 2101) convert
v_mov_b32 v18, v13 // (op 2101) convert

v_mul_lo_u32 v23, v17, s31 // (op 2101) most significant: low of low * high
v_mul_hi_u32 v13, v17, s30 // (op 2101) most significant: high of low * low
v_mul_lo_u32 v22, v17, s30 // (op 2101) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 2101) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v18 // (op 2101) least significant half
v_addc_co_u32 v23, s[18:19], v23, v19, s[18:19] // (op 2101) most significant half



s_mul_i32 s19, s30, s35 // (op 2103) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 2103) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 2103) least significant
s_mul_hi_u32 s6, s30, s34 // (op 2103) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2103) most significant: sum
s_add_u32 s19, s19, s6 // (op 2103) most significant: sum


s_mov_b32 s68, s32 // (op 2103) reg expression
s_mov_b32 s69, s33 // (op 2103) reg expression
s_mov_b32 s70, s18 // (op 2103) reg expression
s_mov_b32 s71, 131072 // (op 2103) call()

v_and_b32 v13, -4, v1 // (op 2214) 
s_mul_i32 s15, s16, s52 // (op 2214) 
s_add_u32 s15, s15, s0 // (op 2214) 
v_sub_u32 v16, v1, v13 // (op 2214) 

s_add_u32 s6, s15, s9 // (op 2214) 

v_lshlrev_b32 v13, 4, v16 // (op 2214) 

s_mul_hi_u32 s15, s6, s13 // (op 2214) 
v_lshrrev_b32 v16, 2, v1 // (op 2214) 
s_sub_u32 s7, s6, s15 // (op 2214) 
s_lshr_b32 s8, s7, 1 // (op 2214) 

s_add_u32 s7, s8, s15 // (op 2214) 


s_lshr_b32 s8, s7, s11 // (op 2214) 

s_and_b32 s7, s11, 2147483648 // (op 2214) 
s_cmp_eq_i32 s7, 0 // (op 2214) 
s_mov_b32 s15, scc // (op 2214) 

s_cmp_eq_i32 s15, 0 // (op 2214) 
s_cselect_b32 s7, s6, s8 // (op 2214) 


s_mul_hi_u32 s8, s7, s12 // (op 2214) 
s_sub_u32 s15, s7, s8 // (op 2214) 
s_lshr_b32 s18, s15, 1 // (op 2214) 

s_add_u32 s15, s18, s8 // (op 2214) 


s_lshr_b32 s8, s15, s10 // (op 2214) 

s_and_b32 s15, s10, 2147483648 // (op 2214) 
s_cmp_eq_i32 s15, 0 // (op 2214) 
s_mov_b32 s18, scc // (op 2214) 

s_cmp_eq_i32 s18, 0 // (op 2214) 
s_cselect_b32 s15, s7, s8 // (op 2214) 


s_mul_i32 s8, s15, s14 // (op 2214) 

s_sub_u32 s15, s7, s8 // (op 2214) 

v_lshl_add_u32 v17, s15, 6, v13 // (op 2214) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2214) 
v_mov_b32 v19, 0 // (op 2214) convert
v_mov_b32 v18, v17 // (op 2214) convert

s_mul_i32 s8, s7, s18 // (op 2214) 


s_sub_u32 s7, s6, s8 // (op 2214) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 2214) 


v_mul_lo_u32 v17, v13, s37 // (op 2214) most significant: low of low * high
v_mul_hi_u32 v24, v13, s36 // (op 2214) most significant: high of low * low
v_mul_lo_u32 v16, v13, s36 // (op 2214) least significant: low of low * low
v_add_u32 v17, v17, v24 // (op 2214) most significant: sum

v_add_co_u32 v16, s[18:19], v16, v18 // (op 2214) least significant half
v_addc_co_u32 v17, s[18:19], v17, v19, s[18:19] // (op 2214) most significant half



s_mul_i32 s19, s36, s3 // (op 2216) most significant: low of low * high
s_mul_i32 s15, s37, s2 // (op 2216) most significant: low of high * low
s_mul_i32 s18, s36, s2 // (op 2216) least significant
s_mul_hi_u32 s6, s36, s2 // (op 2216) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2216) most significant: sum
s_add_u32 s19, s19, s6 // (op 2216) most significant: sum


s_mov_b32 s72, s38 // (op 2216) reg expression
s_mov_b32 s73, s39 // (op 2216) reg expression
s_mov_b32 s74, s18 // (op 2216) reg expression
s_mov_b32 s75, 131072 // (op 2216) call()

v_and_b32 v13, -4, v1 // (op 2233) 
s_mul_i32 s15, s16, s52 // (op 2233) 
s_add_u32 s15, s15, s0 // (op 2233) 
v_sub_u32 v18, v1, v13 // (op 2233) 

s_add_u32 s6, s15, s9 // (op 2233) 

v_lshlrev_b32 v13, 4, v18 // (op 2233) 

s_mul_hi_u32 s15, s6, s13 // (op 2233) 
v_lshrrev_b32 v18, 2, v1 // (op 2233) 
s_sub_u32 s7, s6, s15 // (op 2233) 
s_lshr_b32 s8, s7, 1 // (op 2233) 

s_add_u32 s7, s8, s15 // (op 2233) 


s_lshr_b32 s8, s7, s11 // (op 2233) 

s_and_b32 s7, s11, 2147483648 // (op 2233) 
s_cmp_eq_i32 s7, 0 // (op 2233) 
s_mov_b32 s15, scc // (op 2233) 

s_cmp_eq_i32 s15, 0 // (op 2233) 
s_cselect_b32 s7, s6, s8 // (op 2233) 


s_mul_hi_u32 s8, s7, s12 // (op 2233) 
s_sub_u32 s15, s7, s8 // (op 2233) 
s_lshr_b32 s18, s15, 1 // (op 2233) 

s_add_u32 s15, s18, s8 // (op 2233) 


s_lshr_b32 s8, s15, s10 // (op 2233) 

s_and_b32 s15, s10, 2147483648 // (op 2233) 
s_cmp_eq_i32 s15, 0 // (op 2233) 
s_mov_b32 s18, scc // (op 2233) 

s_cmp_eq_i32 s18, 0 // (op 2233) 
s_cselect_b32 s15, s7, s8 // (op 2233) 


s_mul_i32 s8, s15, s14 // (op 2233) 

s_sub_u32 s15, s7, s8 // (op 2233) 

v_lshl_add_u32 v19, s15, 6, v13 // (op 2233) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2233) 
v_mov_b32 v25, 0 // (op 2233) convert
v_mov_b32 v24, v19 // (op 2233) convert

s_mul_i32 s8, s7, s18 // (op 2233) 


s_sub_u32 s7, s6, s8 // (op 2233) 


v_lshl_add_u32 v13, s7, 7, v18 // (op 2233) 


v_mul_lo_u32 v19, v13, s37 // (op 2233) most significant: low of low * high
v_mul_hi_u32 v26, v13, s36 // (op 2233) most significant: high of low * low
v_mul_lo_u32 v18, v13, s36 // (op 2233) least significant: low of low * low
v_add_u32 v19, v19, v26 // (op 2233) most significant: sum

v_add_co_u32 v18, s[18:19], v18, v24 // (op 2233) least significant half
v_addc_co_u32 v19, s[18:19], v19, v25, s[18:19] // (op 2233) most significant half



s_mul_i32 s19, s36, s3 // (op 2235) most significant: low of low * high
s_mul_i32 s15, s37, s2 // (op 2235) most significant: low of high * low
s_mul_i32 s18, s36, s2 // (op 2235) least significant
s_mul_hi_u32 s6, s36, s2 // (op 2235) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2235) most significant: sum
s_add_u32 s19, s19, s6 // (op 2235) most significant: sum


s_mov_b32 s72, s38 // (op 2235) reg expression
s_mov_b32 s73, s39 // (op 2235) reg expression
s_mov_b32 s74, s18 // (op 2235) reg expression
s_mov_b32 s75, 131072 // (op 2235) call()

v_lshrrev_b32 v13, 7, v1 // (op 2342) 
s_mul_i32 s15, s16, s52 // (op 2342) 
s_add_u32 s15, s15, s0 // (op 2342) 
v_and_b32 v24, 15, v1 // (op 2342) 
s_add_u32 s6, s15, s9 // (op 2342) 

v_lshl_add_u32 v25, v13, 4, v24 // (op 2342) 


s_mul_hi_u32 s15, s6, s13 // (op 2342) 
v_and_b32 v13, 63, v1 // (op 2342) 
s_sub_u32 s7, s6, s15 // (op 2342) 
v_lshrrev_b32 v24, 4, v13 // (op 2342) 

s_lshr_b32 s8, s7, 1 // (op 2342) 

s_add_u32 s7, s8, s15 // (op 2342) 


s_lshr_b32 s8, s7, s11 // (op 2342) 

s_and_b32 s7, s11, 2147483648 // (op 2342) 
s_cmp_eq_i32 s7, 0 // (op 2342) 
s_mov_b32 s15, scc // (op 2342) 

s_cmp_eq_i32 s15, 0 // (op 2342) 
s_cselect_b32 s7, s6, s8 // (op 2342) 


s_mul_hi_u32 s8, s7, s12 // (op 2342) 
s_sub_u32 s15, s7, s8 // (op 2342) 
s_lshr_b32 s18, s15, 1 // (op 2342) 

s_add_u32 s15, s18, s8 // (op 2342) 


s_lshr_b32 s8, s15, s10 // (op 2342) 

s_and_b32 s15, s10, 2147483648 // (op 2342) 
s_cmp_eq_i32 s15, 0 // (op 2342) 
s_mov_b32 s18, scc // (op 2342) 

s_cmp_eq_i32 s18, 0 // (op 2342) 
s_cselect_b32 s15, s7, s8 // (op 2342) 


v_lshl_add_u32 v13, s15, 5, v25 // (op 2342) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2342) 
s_mul_i32 s8, s7, s18 // (op 2342) 


s_sub_u32 s7, s6, s8 // (op 2342) 


v_lshl_add_u32 v25, s7, 2, v24 // (op 2342) 


v_mul_lo_u32 v24, v13, s40 // (op 2342) 
v_add_u32 v24, v24, v25 // (op 2342) 


s_mul_i32 s19, s40, s35 // (op 2344) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 2344) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 2344) least significant
s_mul_hi_u32 s6, s40, s34 // (op 2344) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2344) most significant: sum
s_add_u32 s19, s19, s6 // (op 2344) most significant: sum


s_mov_b32 s76, s42 // (op 2344) reg expression
s_mov_b32 s77, s43 // (op 2344) reg expression
s_mov_b32 s78, s18 // (op 2344) reg expression
s_mov_b32 s79, 131072 // (op 2344) call()

v_lshrrev_b32 v13, 7, v1 // (op 2361) 
s_mul_i32 s15, s16, s52 // (op 2361) 
s_add_u32 s15, s15, s0 // (op 2361) 
v_and_b32 v25, 15, v1 // (op 2361) 
s_add_u32 s6, s15, s9 // (op 2361) 

v_lshl_add_u32 v26, v13, 4, v25 // (op 2361) 


s_mul_hi_u32 s15, s6, s13 // (op 2361) 
v_and_b32 v13, 63, v1 // (op 2361) 
s_sub_u32 s7, s6, s15 // (op 2361) 
v_lshrrev_b32 v25, 4, v13 // (op 2361) 

s_lshr_b32 s8, s7, 1 // (op 2361) 

s_add_u32 s7, s8, s15 // (op 2361) 


s_lshr_b32 s8, s7, s11 // (op 2361) 

s_and_b32 s7, s11, 2147483648 // (op 2361) 
s_cmp_eq_i32 s7, 0 // (op 2361) 
s_mov_b32 s15, scc // (op 2361) 

s_cmp_eq_i32 s15, 0 // (op 2361) 
s_cselect_b32 s7, s6, s8 // (op 2361) 


s_mul_hi_u32 s8, s7, s12 // (op 2361) 
s_sub_u32 s15, s7, s8 // (op 2361) 
s_lshr_b32 s18, s15, 1 // (op 2361) 

s_add_u32 s15, s18, s8 // (op 2361) 


s_lshr_b32 s8, s15, s10 // (op 2361) 

s_and_b32 s15, s10, 2147483648 // (op 2361) 
s_cmp_eq_i32 s15, 0 // (op 2361) 
s_mov_b32 s18, scc // (op 2361) 

s_cmp_eq_i32 s18, 0 // (op 2361) 
s_cselect_b32 s15, s7, s8 // (op 2361) 


v_lshl_add_u32 v13, s15, 5, v26 // (op 2361) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2361) 
s_mul_i32 s8, s7, s18 // (op 2361) 


s_sub_u32 s7, s6, s8 // (op 2361) 


v_lshl_add_u32 v26, s7, 2, v25 // (op 2361) 


v_mul_lo_u32 v25, v13, s40 // (op 2361) 
v_add_u32 v25, v25, v26 // (op 2361) 


s_mul_i32 s19, s40, s35 // (op 2363) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 2363) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 2363) least significant
s_mul_hi_u32 s6, s40, s34 // (op 2363) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2363) most significant: sum
s_add_u32 s19, s19, s6 // (op 2363) most significant: sum


s_mov_b32 s76, s42 // (op 2363) reg expression
s_mov_b32 s77, s43 // (op 2363) reg expression
s_mov_b32 s78, s18 // (op 2363) reg expression
s_mov_b32 s79, 131072 // (op 2363) call()

v_and_b32 v13, -64, v1 // (op 2470) 
s_mul_i32 s15, s16, s52 // (op 2470) 
s_add_u32 s15, s15, s0 // (op 2470) 
v_sub_u32 v26, v1, v13 // (op 2470) 

s_add_u32 s6, s15, s9 // (op 2470) 

v_lshrrev_b32 v13, 6, v1 // (op 2470) 
s_mul_hi_u32 s15, s6, s13 // (op 2470) 
s_sub_u32 s7, s6, s15 // (op 2470) 
s_lshr_b32 s8, s7, 1 // (op 2470) 

s_add_u32 s7, s8, s15 // (op 2470) 


s_lshr_b32 s8, s7, s11 // (op 2470) 

s_and_b32 s7, s11, 2147483648 // (op 2470) 
s_cmp_eq_i32 s7, 0 // (op 2470) 
s_mov_b32 s15, scc // (op 2470) 

s_cmp_eq_i32 s15, 0 // (op 2470) 
s_cselect_b32 s7, s6, s8 // (op 2470) 


s_mul_hi_u32 s8, s7, s12 // (op 2470) 
s_sub_u32 s15, s7, s8 // (op 2470) 
s_lshr_b32 s18, s15, 1 // (op 2470) 

s_add_u32 s15, s18, s8 // (op 2470) 


s_lshr_b32 s8, s15, s10 // (op 2470) 

s_and_b32 s15, s10, 2147483648 // (op 2470) 
s_cmp_eq_i32 s15, 0 // (op 2470) 
s_mov_b32 s18, scc // (op 2470) 

s_cmp_eq_i32 s18, 0 // (op 2470) 
s_cselect_b32 s15, s7, s8 // (op 2470) 


s_mul_i32 s8, s15, s14 // (op 2470) 

s_sub_u32 s15, s7, s8 // (op 2470) 

v_lshl_add_u32 v27, s15, 6, v26 // (op 2470) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2470) 
s_mul_i32 s8, s7, s18 // (op 2470) 


s_sub_u32 s7, s6, s8 // (op 2470) 


v_lshl_add_u32 v26, s7, 2, v13 // (op 2470) 


v_mul_lo_u32 v13, v26, s44 // (op 2470) 
v_add_u32 v13, v13, v27 // (op 2470) 


s_mul_i32 s19, s44, s49 // (op 2472) most significant: low of low * high
s_mul_i32 s15, s45, s48 // (op 2472) most significant: low of high * low
s_mul_i32 s18, s44, s48 // (op 2472) least significant
s_mul_hi_u32 s6, s44, s48 // (op 2472) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2472) most significant: sum
s_add_u32 s19, s19, s6 // (op 2472) most significant: sum


s_mov_b32 s80, s46 // (op 2472) reg expression
s_mov_b32 s81, s47 // (op 2472) reg expression
s_mov_b32 s82, s18 // (op 2472) reg expression
s_mov_b32 s83, 131072 // (op 2472) call()

v_and_b32 v26, -64, v1 // (op 2489) 
s_mul_i32 s15, s16, s52 // (op 2489) 
s_add_u32 s15, s15, s0 // (op 2489) 
v_sub_u32 v27, v1, v26 // (op 2489) 

s_add_u32 s6, s15, s9 // (op 2489) 

v_lshrrev_b32 v26, 6, v1 // (op 2489) 
s_mul_hi_u32 s15, s6, s13 // (op 2489) 
s_sub_u32 s7, s6, s15 // (op 2489) 
s_lshr_b32 s8, s7, 1 // (op 2489) 

s_add_u32 s7, s8, s15 // (op 2489) 


s_lshr_b32 s8, s7, s11 // (op 2489) 

s_and_b32 s7, s11, 2147483648 // (op 2489) 
s_cmp_eq_i32 s7, 0 // (op 2489) 
s_mov_b32 s15, scc // (op 2489) 

s_cmp_eq_i32 s15, 0 // (op 2489) 
s_cselect_b32 s7, s6, s8 // (op 2489) 


s_mul_hi_u32 s8, s7, s12 // (op 2489) 
s_sub_u32 s15, s7, s8 // (op 2489) 
s_lshr_b32 s18, s15, 1 // (op 2489) 

s_add_u32 s15, s18, s8 // (op 2489) 


s_lshr_b32 s8, s15, s10 // (op 2489) 

s_and_b32 s15, s10, 2147483648 // (op 2489) 
s_cmp_eq_i32 s15, 0 // (op 2489) 
s_mov_b32 s18, scc // (op 2489) 

s_cmp_eq_i32 s18, 0 // (op 2489) 
s_cselect_b32 s15, s7, s8 // (op 2489) 


s_mul_i32 s8, s15, s14 // (op 2489) 

s_sub_u32 s15, s7, s8 // (op 2489) 

v_lshl_add_u32 v28, s15, 6, v27 // (op 2489) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2489) 
s_mul_i32 s8, s7, s18 // (op 2489) 


s_sub_u32 s7, s6, s8 // (op 2489) 


v_lshl_add_u32 v27, s7, 2, v26 // (op 2489) 


v_mul_lo_u32 v26, v27, s44 // (op 2489) 
v_add_u32 v26, v26, v28 // (op 2489) 


s_mul_i32 s19, s44, s49 // (op 2491) most significant: low of low * high
s_mul_i32 s15, s45, s48 // (op 2491) most significant: low of high * low
s_mul_i32 s18, s44, s48 // (op 2491) least significant
s_mul_hi_u32 s6, s44, s48 // (op 2491) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2491) most significant: sum
s_add_u32 s19, s19, s6 // (op 2491) most significant: sum


s_mov_b32 s80, s46 // (op 2491) reg expression
s_mov_b32 s81, s47 // (op 2491) reg expression
s_mov_b32 s82, s18 // (op 2491) reg expression
s_mov_b32 s83, 131072 // (op 2491) call()




buffer_load_ubyte v27, v24, s[76:79], 0 offen offset: 0 // (op 1700) Load value
v_and_b32 v28, -4, v1 // (op 2946) 
v_sub_u32 v29, v1, v28 // (op 2946) 

v_lshlrev_b32 v28, 4, v29 // (op 2946) 

v_lshrrev_b32 v29, 2, v1 // (op 2946) 
v_lshl_add_u32 v30, v29, 6, v28 // (op 2946) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 2946) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3481) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 1992) 
v_mov_b64 v[28:29], s[18:19] // (op 1992) convert


v_mov_b64 v[30:31], v[16:17] // (op 1992) reg expression
s_mov_b32 m0, s15 // (op 1992) reg expression
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 1992) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 1992) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 1992) most significant half

s_add_u32 m0, m0, 4096 // (op 1992) 
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 1992) Load value direct to lds





v_mov_b32 v28, v13 // (op 1707) reg expression

buffer_load_ubyte v29, v28, s[80:83], 0 offen offset: 0 // (op 1707) Load value

v_and_b32 v28, -8, v1 // (op 2881) 
v_sub_u32 v30, v1, v28 // (op 2881) 

v_lshlrev_b32 v28, 4, v30 // (op 2881) 

v_lshrrev_b32 v30, 3, v1 // (op 2881) 
v_lshl_add_u32 v31, v30, 7, v28 // (op 2881) 


s_nop 0
v_readfirstlane_b32 s15, v31 // (op 2881) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3462) 


s_nop 0
s_add_u32 s6, 8192, s15 // (op 1997) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s6 // (op 1997) reg expression
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 1997) Load value direct to lds





buffer_load_ubyte v28, v24, s[76:79], 0 offen offset:4 // (op 1714) Load value

v_and_b32 v24, -4, v1 // (op 3165) 
v_sub_u32 v30, v1, v24 // (op 3165) 

v_lshlrev_b32 v24, 4, v30 // (op 3165) 

v_lshrrev_b32 v30, 2, v1 // (op 3165) 
v_lshl_add_u32 v31, v30, 6, v24 // (op 3165) 


s_nop 0
v_readfirstlane_b32 s15, v31 // (op 3165) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3529) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 2002) 
v_mov_b64 v[30:31], s[18:19] // (op 2002) convert


s_lshl_b64 s[18:19], s[36:37], 7 // (op 2002) 
v_mov_b32 v24, s18 // (op 2002) 
v_mov_b32 v32, s19 // (op 2002) 
v_add_co_u32 v34, s[28:29], v16, v24 // (op 2002) least significant half
v_addc_co_u32 v35, s[28:29], v17, v32, s[28:29] // (op 2002) most significant half




s_add_u32 s6, 12288, s15 // (op 2002) 
s_mov_b32 m0, s6 // (op 2002) reg expression
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2002) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v30 // (op 2002) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2002) most significant half

s_add_u32 m0, m0, 4096 // (op 2002) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2002) Load value direct to lds







s_lshl_b32 s15, s44, 2 // (op 1721) 
v_add_u32 v24, v13, s15 // (op 1721) 


buffer_load_ubyte v16, v24, s[80:83], 0 offen offset: 0 // (op 1721) Load value


v_and_b32 v13, -8, v1 // (op 3126) 
v_sub_u32 v17, v1, v13 // (op 3126) 

v_lshlrev_b32 v13, 4, v17 // (op 3126) 

v_lshrrev_b32 v17, 3, v1 // (op 3126) 
v_lshl_add_u32 v24, v17, 7, v13 // (op 3126) 


s_nop 0
v_readfirstlane_b32 s15, v24 // (op 3126) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 3520) 


s_nop 0
s_add_u32 s6, 20480, s15 // (op 2004) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s6 // (op 2004) reg expression
s_mov_b32 s7, 128 // (op 2004) call()
buffer_load_dwordx4 v20, s[68:71], s7 offen offset: 0 lds // (op 2004) Load value direct to lds




v_and_b32 v13, -64, v1 // (op 3022) 
v_sub_u32 v17, v1, v13 // (op 3022) 

v_and_b32 v13, 4294967232, v1 // (op 3022) 
v_add_u32 v24, v13, v17 // (op 3022) 


s_barrier  // (op 3500) 



s_waitcnt vmcnt(6)
ds_write_b8 v24, v29 offset:24576 // (op 1727) Store local data 

s_barrier  // (op 3478) 



ds_read_b64_tr_b8 v[30:31], v12 // (op 1745) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v12 offset:512 // (op 1745) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v12 offset:4096 // (op 1745) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v12 offset:4608 // (op 1745) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3497) 


ds_read_u8 v13, v0 offset:24576 // (op 1755) Load local data 



ds_read_b64_tr_b8 v[38:39], v12 offset:32 // (op 1750) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v12 offset:544 // (op 1750) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v12 offset:4128 // (op 1750) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v12 offset:4640 // (op 1750) Transpose load local data 


ds_read_u8 v17, v0 offset:24608 // (op 1760) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 3459) 



ds_read_b128 v[46:49], v3 offset:8192 // (op 1741) Load local data 
ds_read_b128 v[50:53], v3 offset:8256 // (op 1741) Load local data 
v_and_b32 v24, -8, v1 // (op 2853) 
v_sub_u32 v20, v1, v24 // (op 2853) 

v_lshlrev_b32 v21, 4, v20 // (op 2853) 

v_lshrrev_b32 v20, 3, v1 // (op 2853) 
v_lshl_add_u32 v24, v20, 7, v21 // (op 2853) 


s_nop 0
v_readfirstlane_b32 s15, v24 // (op 2853) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -4, v1 // (op 2918) 
v_sub_u32 v20, v1, v24 // (op 2918) 

v_lshlrev_b32 v21, 4, v20 // (op 2918) 

v_lshrrev_b32 v20, 2, v1 // (op 2918) 
v_lshl_add_u32 v24, v20, 6, v21 // (op 2918) 


s_nop 0
v_readfirstlane_b32 s6, v24 // (op 2918) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -64, v1 // (op 2993) 
v_sub_u32 v20, v1, v24 // (op 2993) 

v_and_b32 v21, 4294967232, v1 // (op 2993) 
v_add_u32 v24, v21, v20 // (op 2993) 


v_and_b32 v20, -8, v1 // (op 3111) 
v_sub_u32 v21, v1, v20 // (op 3111) 

v_lshlrev_b32 v20, 4, v21 // (op 3111) 

v_lshrrev_b32 v21, 3, v1 // (op 3111) 
v_lshl_add_u32 v54, v21, 7, v20 // (op 3111) 


s_nop 0
v_readfirstlane_b32 s7, v54 // (op 3111) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -4, v1 // (op 3150) 
v_sub_u32 v21, v1, v20 // (op 3150) 

v_lshlrev_b32 v20, 4, v21 // (op 3150) 

v_lshrrev_b32 v21, 2, v1 // (op 3150) 
v_lshl_add_u32 v54, v21, 6, v20 // (op 3150) 


s_nop 0
v_readfirstlane_b32 s8, v54 // (op 3150) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -64, v1 // (op 3194) 
v_sub_u32 v21, v1, v20 // (op 3194) 

v_and_b32 v20, 4294967232, v1 // (op 3194) 
v_add_u32 v54, v20, v21 // (op 3194) 


s_mov_b32 s1, 0 // (op 588) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 587) 
s_mov_b32 s28, s18 // (op 587) convert

s_and_b32 s29, s28, 4294967294 // (op 587) 

s_cmp_lt_u32 s1, s29 // (op 587) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_587 // (op 587) Condition: Top (jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_587 if false)
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoop_587:
s_add_u32 s1, s1, 2 // (op 589) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:53], v[30:37], a[0:3], v27, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 188) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[38:45], a[4:7], v27, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 623) 


v_mov_b32 v20, v25 // (op 185) 

buffer_load_ubyte v27, v20, s[76:79], 0 offen offset:8 // (op 185) Load value
s_barrier  // (op 3484) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 1941) 
v_mov_b64 v[56:57], s[18:19] // (op 1941) convert


v_mov_b64 v[58:59], v[18:19] // (op 1941) 
s_lshl_b64 s[18:19], s[36:37], 7 // (op 1941) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 1941) 

v_mov_b32 v21, s28 // (op 1941) 
v_mov_b32 v55, s29 // (op 1941) 
v_add_co_u32 v60, s[18:19], v58, v21 // (op 1941) least significant half
v_addc_co_u32 v61, s[18:19], v59, v55, s[18:19] // (op 1941) most significant half




s_mov_b32 m0, s6 // (op 1941) reg expression
buffer_load_dwordx4 v60, s[72:75], 0 offen offset: 0 lds // (op 1941) Load value direct to lds
v_add_co_u32 v60, s[18:19], v60, v56 // (op 1941) least significant half
v_addc_co_u32 v61, s[18:19], 0, v61, s[18:19] // (op 1941) most significant half

s_add_u32 m0, m0, 4096 // (op 1941) 
buffer_load_dwordx4 v60, s[72:75], 0 offen offset: 0 lds // (op 1941) Load value direct to lds




v_mov_b32 v21, v26 // (op 192) 
s_lshl_b32 s18, s44, 3 // (op 192) 
v_add_u32 v55, v21, s18 // (op 192) 


buffer_load_ubyte v29, v55, s[80:83], 0 offen offset: 0 // (op 192) Load value

s_barrier  // (op 3465) 


v_mov_b64 v[56:57], v[22:23] // (op 1936) 
s_add_u32 s18, 8192, s15 // (op 1936) 

s_mov_b32 m0, s18 // (op 1936) reg expression
s_mov_b32 s19, 256 // (op 1936) call()
buffer_load_dwordx4 v56, s[68:71], s19 offen offset: 0 lds // (op 1936) Load value direct to lds


s_barrier  // (op 3538) 



s_waitcnt vmcnt(6)
ds_write_b8 v54, v16 offset:24832 // (op 516) Store local data 
s_barrier  // (op 3526) 



ds_read_b64_tr_b8 v[30:31], v11 offset:12288 // (op 509) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v11 offset:12800 // (op 509) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v11 offset:16384 // (op 509) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v11 offset:16896 // (op 509) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3535) 


ds_read_u8 v13, v10 offset:24832 // (op 513) Load local data 



ds_read_b64_tr_b8 v[38:39], v11 offset:12320 // (op 628) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v11 offset:12832 // (op 628) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v11 offset:16416 // (op 628) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v11 offset:16928 // (op 628) Transpose load local data 


ds_read_u8 v17, v10 offset:24864 // (op 631) Load local data 
s_barrier  // (op 3657) 
s_waitcnt vmcnt(5)
s_barrier  // (op 3517) 



ds_read_b128 v[46:49], v5 offset:20480 // (op 507) Load local data 
ds_read_b128 v[50:53], v5 offset:20544 // (op 507) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:53], v[30:37], a[0:3], v28, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 514) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[38:45], a[4:7], v28, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 624) 



buffer_load_ubyte v28, v20, s[76:79], 0 offen offset:12 // (op 511) Load value

s_barrier  // (op 3532) 
v_add_u32 v25, 8, v25 // (op 2353) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 1964) 
v_mov_b64 v[60:61], s[18:19] // (op 1964) convert


s_lshl_b64 s[18:19], s[36:37], 7 // (op 1964) 
s_mul_i32 s29, 3, s19 // (op 1964) most significant: low of low * high
s_mul_i32 s28, 3, s18 // (op 1964) least significant
s_mul_hi_u32 s50, 3, s18 // (op 1964) most significant: high of low * low
s_add_u32 s29, s29, s50 // (op 1964) most significant: sum


v_mov_b32 v20, s28 // (op 1964) 
v_mov_b32 v55, s29 // (op 1964) 
v_add_co_u32 v62, s[18:19], v58, v20 // (op 1964) least significant half
v_addc_co_u32 v63, s[18:19], v59, v55, s[18:19] // (op 1964) most significant half




s_add_u32 s18, 12288, s8 // (op 1964) 
s_nop 0
s_mov_b32 m0, s18 // (op 1964) reg expression
buffer_load_dwordx4 v62, s[72:75], 0 offen offset: 0 lds // (op 1964) Load value direct to lds
v_add_co_u32 v62, s[28:29], v62, v60 // (op 1964) least significant half
v_addc_co_u32 v63, s[28:29], 0, v63, s[28:29] // (op 1964) most significant half

s_add_u32 m0, m0, 4096 // (op 1964) 
buffer_load_dwordx4 v62, s[72:75], 0 offen offset: 0 lds // (op 1964) Load value direct to lds






s_lshl_b32 s18, s44, 2 // (op 518) 
s_mul_i32 s19, 3, s18 // (op 518) 

v_add_u32 v20, v21, s19 // (op 518) 


s_lshl_b64 s[18:19], s[36:37], 7 // (op 2225) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 2225) 

v_mov_b32 v55, s28 // (op 2225) 
v_mov_b32 v58, s29 // (op 2225) 
v_add_co_u32 v18, s[18:19], v18, v55 // (op 2225) least significant half
v_addc_co_u32 v19, s[18:19], v19, v58, s[18:19] // (op 2225) most significant half




buffer_load_ubyte v16, v20, s[80:83], 0 offen offset: 0 // (op 518) Load value


s_barrier  // (op 3523) 
s_lshl_b32 s18, s44, 3 // (op 2481) 


v_add_u32 v26, v26, s18 // (op 2481) 

s_add_u32 s18, 20480, s7 // (op 1959) 

s_mov_b32 m0, s18 // (op 1959) reg expression
s_mov_b32 s19, 384 // (op 1959) call()
buffer_load_dwordx4 v56, s[68:71], s19 offen offset: 0 lds // (op 1959) Load value direct to lds



s_barrier  // (op 3503) 
v_add_co_u32 v22, vcc, 256, v22 // (op 2093) least significant half



v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 2093) most significant half
s_waitcnt vmcnt(6)
ds_write_b8 v24, v29 offset:24576 // (op 190) Store local data 
s_barrier  // (op 3490) 



ds_read_b64_tr_b8 v[30:31], v9 // (op 183) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v9 offset:512 // (op 183) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v9 offset:4096 // (op 183) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v9 offset:4608 // (op 183) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3510) 


ds_read_u8 v13, v8 offset:24576 // (op 187) Load local data 



ds_read_b64_tr_b8 v[38:39], v9 offset:32 // (op 619) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v9 offset:544 // (op 619) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v9 offset:4128 // (op 619) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v9 offset:4640 // (op 619) Transpose load local data 


ds_read_u8 v17, v8 offset:24608 // (op 622) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 3471) 



ds_read_b128 v[46:49], v4 offset:8192 // (op 181) Load local data 
ds_read_b128 v[50:53], v4 offset:8256 // (op 181) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 587) 
s_mov_b32 s28, s18 // (op 587) convert

s_and_b32 s29, s28, 4294967294 // (op 587) 

s_cmp_lt_u32 s1, s29 // (op 587) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoop_587 // (op 587) Condition: Bottom (jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoop_587 if true)
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_587:












s_barrier  // (op 3641) 
v_lshrrev_b32 v29, 3, v1 // (op 2031) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2031) 
v_and_b32 v18, -8, v1 // (op 2031) 
s_and_b32 s15, s18, 4294967294 // (op 2031) 
v_sub_u32 v19, v1, v18 // (op 2031) 

s_add_u32 s6, s0, s15 // (op 2031) 

v_lshlrev_b32 v18, 4, v19 // (op 2031) 

s_mul_i32 s15, s16, s52 // (op 2031) 
s_add_u32 s15, s15, s6 // (op 2031) 

s_add_u32 s6, s15, s9 // (op 2031) 

s_mul_hi_u32 s15, s6, s13 // (op 2031) 
s_sub_u32 s7, s6, s15 // (op 2031) 
s_lshr_b32 s8, s7, 1 // (op 2031) 

s_add_u32 s7, s8, s15 // (op 2031) 


s_lshr_b32 s8, s7, s11 // (op 2031) 

s_and_b32 s7, s11, 2147483648 // (op 2031) 
s_cmp_eq_i32 s7, 0 // (op 2031) 
s_mov_b32 s15, scc // (op 2031) 

s_cmp_eq_i32 s15, 0 // (op 2031) 
s_cselect_b32 s7, s6, s8 // (op 2031) 


s_mul_hi_u32 s8, s7, s12 // (op 2031) 
s_sub_u32 s15, s7, s8 // (op 2031) 
s_lshr_b32 s28, s15, 1 // (op 2031) 

s_add_u32 s15, s28, s8 // (op 2031) 


s_lshr_b32 s8, s15, s10 // (op 2031) 

s_and_b32 s15, s10, 2147483648 // (op 2031) 
s_cmp_eq_i32 s15, 0 // (op 2031) 
s_mov_b32 s28, scc // (op 2031) 

s_cmp_eq_i32 s28, 0 // (op 2031) 
s_cselect_b32 s15, s7, s8 // (op 2031) 


s_mul_i32 s8, s7, s18 // (op 2031) 


v_lshl_add_u32 v19, s15, 5, v29 // (op 2031) 


s_sub_u32 s7, s6, s8 // (op 2031) 


v_lshl_add_u32 v29, s7, 7, v18 // (op 2031) 


v_mov_b32 v21, 0 // (op 2031) convert
v_mov_b32 v20, v29 // (op 2031) convert

v_mul_lo_u32 v23, v19, s31 // (op 2031) most significant: low of low * high
v_mul_hi_u32 v18, v19, s30 // (op 2031) most significant: high of low * low
v_mul_lo_u32 v22, v19, s30 // (op 2031) least significant: low of low * low
v_add_u32 v23, v23, v18 // (op 2031) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v20 // (op 2031) least significant half
v_addc_co_u32 v23, s[18:19], v23, v21, s[18:19] // (op 2031) most significant half



s_mul_i32 s19, s30, s35 // (op 2033) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 2033) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 2033) least significant
s_mul_hi_u32 s6, s30, s34 // (op 2033) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2033) most significant: sum
s_add_u32 s19, s19, s6 // (op 2033) most significant: sum


s_mov_b32 s68, s32 // (op 2033) reg expression
s_mov_b32 s69, s33 // (op 2033) reg expression
s_mov_b32 s70, s18 // (op 2033) reg expression
s_mov_b32 s71, 131072 // (op 2033) call()

v_and_b32 v29, -4, v1 // (op 2169) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2169) 
v_sub_u32 v18, v1, v29 // (op 2169) 

s_and_b32 s15, s18, 4294967294 // (op 2169) 
v_lshlrev_b32 v29, 4, v18 // (op 2169) 

s_add_u32 s6, s0, s15 // (op 2169) 

v_lshrrev_b32 v18, 2, v1 // (op 2169) 
s_mul_i32 s15, s16, s52 // (op 2169) 
s_add_u32 s15, s15, s6 // (op 2169) 

s_add_u32 s6, s15, s9 // (op 2169) 

s_mul_hi_u32 s15, s6, s13 // (op 2169) 
s_sub_u32 s7, s6, s15 // (op 2169) 
s_lshr_b32 s8, s7, 1 // (op 2169) 

s_add_u32 s7, s8, s15 // (op 2169) 


s_lshr_b32 s8, s7, s11 // (op 2169) 

s_and_b32 s7, s11, 2147483648 // (op 2169) 
s_cmp_eq_i32 s7, 0 // (op 2169) 
s_mov_b32 s15, scc // (op 2169) 

s_cmp_eq_i32 s15, 0 // (op 2169) 
s_cselect_b32 s7, s6, s8 // (op 2169) 


s_mul_hi_u32 s8, s7, s12 // (op 2169) 
s_sub_u32 s15, s7, s8 // (op 2169) 
s_lshr_b32 s28, s15, 1 // (op 2169) 

s_add_u32 s15, s28, s8 // (op 2169) 


s_lshr_b32 s8, s15, s10 // (op 2169) 

s_and_b32 s15, s10, 2147483648 // (op 2169) 
s_cmp_eq_i32 s15, 0 // (op 2169) 
s_mov_b32 s28, scc // (op 2169) 

s_cmp_eq_i32 s28, 0 // (op 2169) 
s_cselect_b32 s15, s7, s8 // (op 2169) 


s_mul_i32 s8, s15, s14 // (op 2169) 

s_sub_u32 s15, s7, s8 // (op 2169) 

s_mul_i32 s8, s7, s18 // (op 2169) 


v_lshl_add_u32 v19, s15, 6, v29 // (op 2169) 


s_sub_u32 s7, s6, s8 // (op 2169) 


v_mov_b32 v21, 0 // (op 2169) convert
v_mov_b32 v20, v19 // (op 2169) convert

v_lshl_add_u32 v19, s7, 7, v18 // (op 2169) 


v_mul_lo_u32 v25, v19, s37 // (op 2169) most significant: low of low * high
v_mul_hi_u32 v18, v19, s36 // (op 2169) most significant: high of low * low
v_mul_lo_u32 v24, v19, s36 // (op 2169) least significant: low of low * low
v_add_u32 v25, v25, v18 // (op 2169) most significant: sum

v_add_co_u32 v24, s[18:19], v24, v20 // (op 2169) least significant half
v_addc_co_u32 v25, s[18:19], v25, v21, s[18:19] // (op 2169) most significant half



s_mul_i32 s19, s36, s3 // (op 2171) most significant: low of low * high
s_mul_i32 s15, s37, s2 // (op 2171) most significant: low of high * low
s_mul_i32 s18, s36, s2 // (op 2171) least significant
s_mul_hi_u32 s6, s36, s2 // (op 2171) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2171) most significant: sum
s_add_u32 s19, s19, s6 // (op 2171) most significant: sum


s_mov_b32 s72, s38 // (op 2171) reg expression
s_mov_b32 s73, s39 // (op 2171) reg expression
s_mov_b32 s74, s18 // (op 2171) reg expression
s_mov_b32 s75, 131072 // (op 2171) call()

v_lshrrev_b32 v29, 7, v1 // (op 2297) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2297) 
v_and_b32 v18, 15, v1 // (op 2297) 
s_and_b32 s15, s18, 4294967294 // (op 2297) 
v_lshl_add_u32 v19, v29, 4, v18 // (op 2297) 


s_add_u32 s6, s0, s15 // (op 2297) 

v_and_b32 v18, 63, v1 // (op 2297) 
v_lshrrev_b32 v29, 4, v18 // (op 2297) 

s_mul_i32 s15, s16, s52 // (op 2297) 
s_add_u32 s15, s15, s6 // (op 2297) 

s_add_u32 s6, s15, s9 // (op 2297) 

s_mul_hi_u32 s15, s6, s13 // (op 2297) 
s_sub_u32 s7, s6, s15 // (op 2297) 
s_lshr_b32 s8, s7, 1 // (op 2297) 

s_add_u32 s7, s8, s15 // (op 2297) 


s_lshr_b32 s8, s7, s11 // (op 2297) 

s_and_b32 s7, s11, 2147483648 // (op 2297) 
s_cmp_eq_i32 s7, 0 // (op 2297) 
s_mov_b32 s15, scc // (op 2297) 

s_cmp_eq_i32 s15, 0 // (op 2297) 
s_cselect_b32 s7, s6, s8 // (op 2297) 


s_mul_hi_u32 s8, s7, s12 // (op 2297) 
s_sub_u32 s15, s7, s8 // (op 2297) 
s_lshr_b32 s28, s15, 1 // (op 2297) 

s_add_u32 s15, s28, s8 // (op 2297) 


s_lshr_b32 s8, s15, s10 // (op 2297) 

s_and_b32 s15, s10, 2147483648 // (op 2297) 
s_cmp_eq_i32 s15, 0 // (op 2297) 
s_mov_b32 s28, scc // (op 2297) 

s_cmp_eq_i32 s28, 0 // (op 2297) 
s_cselect_b32 s15, s7, s8 // (op 2297) 


s_mul_i32 s8, s7, s18 // (op 2297) 


v_lshl_add_u32 v18, s15, 5, v19 // (op 2297) 


s_sub_u32 s7, s6, s8 // (op 2297) 


v_lshl_add_u32 v19, s7, 2, v29 // (op 2297) 


v_mul_lo_u32 v29, v18, s40 // (op 2297) 
v_add_u32 v29, v29, v19 // (op 2297) 


s_mul_i32 s19, s40, s35 // (op 2299) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 2299) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 2299) least significant
s_mul_hi_u32 s6, s40, s34 // (op 2299) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2299) most significant: sum
s_add_u32 s19, s19, s6 // (op 2299) most significant: sum


s_mov_b32 s76, s42 // (op 2299) reg expression
s_mov_b32 s77, s43 // (op 2299) reg expression
s_mov_b32 s78, s18 // (op 2299) reg expression
s_mov_b32 s79, 131072 // (op 2299) call()

v_and_b32 v18, -64, v1 // (op 2425) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2425) 
v_sub_u32 v19, v1, v18 // (op 2425) 

s_and_b32 s15, s18, 4294967294 // (op 2425) 
v_lshrrev_b32 v18, 6, v1 // (op 2425) 
s_add_u32 s6, s0, s15 // (op 2425) 

s_mul_i32 s15, s16, s52 // (op 2425) 
s_add_u32 s15, s15, s6 // (op 2425) 

s_add_u32 s6, s15, s9 // (op 2425) 

s_mul_hi_u32 s15, s6, s13 // (op 2425) 
s_sub_u32 s7, s6, s15 // (op 2425) 
s_lshr_b32 s8, s7, 1 // (op 2425) 

s_add_u32 s7, s8, s15 // (op 2425) 


s_lshr_b32 s8, s7, s11 // (op 2425) 

s_and_b32 s7, s11, 2147483648 // (op 2425) 
s_cmp_eq_i32 s7, 0 // (op 2425) 
s_mov_b32 s15, scc // (op 2425) 

s_cmp_eq_i32 s15, 0 // (op 2425) 
s_cselect_b32 s7, s6, s8 // (op 2425) 


s_mul_hi_u32 s8, s7, s12 // (op 2425) 
s_sub_u32 s15, s7, s8 // (op 2425) 
s_lshr_b32 s28, s15, 1 // (op 2425) 

s_add_u32 s15, s28, s8 // (op 2425) 


s_lshr_b32 s8, s15, s10 // (op 2425) 

s_and_b32 s15, s10, 2147483648 // (op 2425) 
s_cmp_eq_i32 s15, 0 // (op 2425) 
s_mov_b32 s28, scc // (op 2425) 

s_cmp_eq_i32 s28, 0 // (op 2425) 
s_cselect_b32 s15, s7, s8 // (op 2425) 


s_mul_i32 s8, s15, s14 // (op 2425) 

s_sub_u32 s15, s7, s8 // (op 2425) 

s_mul_i32 s8, s7, s18 // (op 2425) 


v_lshl_add_u32 v20, s15, 6, v19 // (op 2425) 


s_sub_u32 s7, s6, s8 // (op 2425) 


v_lshl_add_u32 v19, s7, 2, v18 // (op 2425) 


v_mul_lo_u32 v18, v19, s44 // (op 2425) 
v_add_u32 v18, v18, v20 // (op 2425) 


s_mul_i32 s19, s44, s49 // (op 2427) most significant: low of low * high
s_mul_i32 s15, s45, s48 // (op 2427) most significant: low of high * low
s_mul_i32 s18, s44, s48 // (op 2427) least significant
s_mul_hi_u32 s6, s44, s48 // (op 2427) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2427) most significant: sum
s_add_u32 s19, s19, s6 // (op 2427) most significant: sum


s_mov_b32 s80, s46 // (op 2427) reg expression
s_mov_b32 s81, s47 // (op 2427) reg expression
s_mov_b32 s82, s18 // (op 2427) reg expression
s_mov_b32 s83, 131072 // (op 2427) call()

v_and_b32 v19, -8, v1 // (op 2866) 
v_sub_u32 v20, v1, v19 // (op 2866) 

v_lshlrev_b32 v19, 4, v20 // (op 2866) 

v_lshrrev_b32 v20, 3, v1 // (op 2866) 
v_lshl_add_u32 v21, v20, 7, v19 // (op 2866) 


s_nop 0
v_readfirstlane_b32 s15, v21 // (op 2866) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -4, v1 // (op 2931) 
v_sub_u32 v20, v1, v19 // (op 2931) 

v_lshlrev_b32 v19, 4, v20 // (op 2931) 

v_lshrrev_b32 v20, 2, v1 // (op 2931) 
v_lshl_add_u32 v21, v20, 6, v19 // (op 2931) 


s_nop 0
v_readfirstlane_b32 s6, v21 // (op 2931) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -64, v1 // (op 3006) 
v_sub_u32 v20, v1, v19 // (op 3006) 

v_and_b32 v19, 4294967232, v1 // (op 3006) 
v_add_u32 v21, v19, v20 // (op 3006) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 593) 
s_and_b32 s7, s18, 4294967294 // (op 593) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 592) 
s_mov_b32 s8, s18 // (op 592) convert

s_cmp_lt_u32 s7, s8 // (op 592) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_592 // (op 592) Condition: Top (jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_592 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_592:
s_barrier  // (op 3468) 
s_add_u32 s7, s7, 1 // (op 594) 


v_mov_b64 v[16:17], v[22:23] // (op 1946) 
s_add_u32 s8, 8192, s15 // (op 1946) 

s_mov_b32 m0, s8 // (op 1946) reg expression
buffer_load_dwordx4 v16, s[68:71], 0 offen offset: 0 lds // (op 1946) Load value direct to lds


v_add_co_u32 v22, vcc, 128, v22 // (op 2021) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 2021) most significant half
s_barrier  // (op 3487) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 1952) 
v_mov_b64 v[16:17], s[18:19] // (op 1952) convert


v_mov_b64 v[26:27], v[24:25] // (op 1952) 
v_mov_b64 v[30:31], v[26:27] // (op 1952) reg expression
s_mov_b32 m0, s6 // (op 1952) reg expression
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 1952) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v16 // (op 1952) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 1952) most significant half

s_add_u32 m0, m0, 4096 // (op 1952) 
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 1952) Load value direct to lds



s_lshl_b64 s[18:19], s[36:37], 7 // (op 2161) 
v_mov_b32 v13, s18 // (op 2161) 
v_mov_b32 v16, s19 // (op 2161) 
v_add_co_u32 v24, s[28:29], v24, v13 // (op 2161) least significant half
v_addc_co_u32 v25, s[28:29], v25, v16, s[28:29] // (op 2161) most significant half






v_mov_b32 v13, v18 // (op 460) 
v_mov_b32 v16, v13 // (op 460) reg expression

buffer_load_ubyte v17, v16, s[80:83], 0 offen offset: 0 // (op 460) Load value


s_lshl_b32 s8, s44, 2 // (op 2417) 
v_add_u32 v18, v18, s8 // (op 2417) 

s_barrier  // (op 3506) 
s_waitcnt vmcnt(3)
s_barrier  // (op 3474) 






ds_read_b128 v[30:33], v4 offset:8192 // (op 449) Load local data 
ds_read_b128 v[34:37], v4 offset:8256 // (op 449) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v21, v17 offset:24576 // (op 458) Store local data 

s_barrier  // (op 3493) 



ds_read_b64_tr_b8 v[38:39], v9 // (op 451) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v9 offset:512 // (op 451) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v9 offset:4096 // (op 451) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v9 offset:4608 // (op 451) Transpose load local data 


v_mov_b32 v13, v29 // (op 453) 

buffer_load_ubyte v16, v13, s[76:79], 0 offen offset: 0 // (op 453) Load value

v_add_u32 v29, 4, v29 // (op 2289) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3513) 


ds_read_u8 v13, v8 offset:24576 // (op 455) Load local data 



ds_read_b64_tr_b8 v[46:47], v9 offset:32 // (op 600) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:45], a[0:3], v16, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 456) 
ds_read_b64_tr_b8 v[48:49], v9 offset:544 // (op 600) Transpose load local data 


ds_read_b64_tr_b8 v[50:51], v9 offset:4128 // (op 600) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v9 offset:4640 // (op 600) Transpose load local data 


ds_read_u8 v13, v8 offset:24608 // (op 603) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:37], v[46:53], a[4:7], v16, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 604) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 592) 
s_mov_b32 s8, s18 // (op 592) convert

s_cmp_lt_u32 s7, s8 // (op 592) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_592 // (op 592) Condition: Bottom (jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_592 if true)
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_592:








s_barrier  // (op 3645) 



v_cmp_eq_f32 vcc, 0.00000, v14 // (op 222) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_222 // (op 222) Condition: False, jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_222
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

buffer_store_dwordx2 v[12:13], v0, s[64:67], 0 offen // (op 233) Store value



v_accvgpr_read v16, a4 // (op 784) 
v_accvgpr_read v17, a5 // (op 784) 
v_accvgpr_read v18, a6 // (op 784) 
v_accvgpr_read v19, a7 // (op 784) 
v_mul_f32 v20, v15, v16 // (op 784) 
v_mul_f32 v21, v15, v17 // (op 784) 
v_mul_f32 v22, v15, v18 // (op 784) 
v_mul_f32 v23, v15, v19 // (op 784) 

v_cvt_f16_f32 v16, v20 // (op 786) 
v_cvt_f16_f32 v17, v21 // (op 786) 
v_cvt_f16_f32 v18, v22 // (op 786) 
v_cvt_f16_f32 v19, v23 // (op 786) 


v_pack_b32_f16 v12, v16, v17 // (op 787) 
v_pack_b32_f16 v13, v18, v19 // (op 787) 

s_lshl_b32 s15, s24, 6 // (op 787) 
v_add_u32 v0, v7, s15 // (op 787) 


buffer_store_dwordx2 v[12:13], v0, s[64:67], 0 offen // (op 787) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_222 // (op 222) Condition: Done, jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_222
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_222:


v_mov_b32 v0, v6 // (op 247) reg expression

buffer_load_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 247) Load value

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

buffer_store_dwordx2 v[12:13], v0, s[64:67], 0 offen // (op 246) Store value





s_lshl_b32 s15, s4, 6 // (op 811) 
v_add_u32 v0, v6, s15 // (op 811) 


buffer_load_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 811) Load value

v_accvgpr_read v16, a4 // (op 807) 
v_accvgpr_read v17, a5 // (op 807) 
v_accvgpr_read v18, a6 // (op 807) 
v_accvgpr_read v19, a7 // (op 807) 
v_mul_f32 v20, v15, v16 // (op 807) 
v_mul_f32 v21, v15, v17 // (op 807) 
v_mul_f32 v22, v15, v18 // (op 807) 
v_mul_f32 v23, v15, v19 // (op 807) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v12 // (op 812) 
v_lshrrev_b32 v17, 16, v12 // (op 812) 
v_cvt_f32_f16 v17, v17 // (op 812) 
v_mul_f32 v24, v14, v16 // (op 812) 
v_mul_f32 v25, v14, v17 // (op 812) 
v_cvt_f32_f16 v16, v13 // (op 812) 
v_lshrrev_b32 v17, 16, v13 // (op 812) 
v_cvt_f32_f16 v17, v17 // (op 812) 
v_mul_f32 v26, v14, v16 // (op 812) 
v_mul_f32 v27, v14, v17 // (op 812) 


v_add_f32 v16, v24, v20 // (op 808) 
v_add_f32 v17, v25, v21 // (op 808) 
v_add_f32 v18, v26, v22 // (op 808) 
v_add_f32 v19, v27, v23 // (op 808) 


v_cvt_f16_f32 v20, v16 // (op 809) 
v_cvt_f16_f32 v21, v17 // (op 809) 
v_cvt_f16_f32 v22, v18 // (op 809) 
v_cvt_f16_f32 v23, v19 // (op 809) 


v_pack_b32_f16 v12, v20, v21 // (op 810) 
v_pack_b32_f16 v13, v22, v23 // (op 810) 

s_lshl_b32 s15, s24, 6 // (op 810) 
v_add_u32 v0, v7, s15 // (op 810) 


buffer_store_dwordx2 v[12:13], v0, s[64:67], 0 offen // (op 810) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_222:
s_add_u32 s0, s0, s1 // (op 435) 


s_mul_i32 s15, s52, s16 // (op 433) 
s_cmp_gt_u32 s53, s15 // (op 433) 
s_mov_b32 s6, scc // (op 433) 
s_sub_u32 s7, s53, s15 // (op 433) 

s_cmp_eq_i32 s6, 0 // (op 433) 
s_cselect_b32 s8, 0, s7 // (op 433) 


s_cmp_lt_u32 s52, s8 // (op 433) 
s_mov_b32 s15, scc // (op 433) 
s_cmp_eq_i32 s15, 0 // (op 433) 
s_cselect_b32 s6, s8, s52 // (op 433) 


s_cmp_lt_u32 s0, s6 // (op 433) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_DPStreamTileLoop_433 // (op 433) Condition: Bottom (jump to RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopTop_DPStreamTileLoop_433 if true)
RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_433:










































s_endpgm  // End of RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2
.LRR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_end:
.size RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2, .LRR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2_end-RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2
  .amdhsa_next_free_vgpr 80
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 25088 // lds bytes
  .amdhsa_accum_offset 72
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
    macrotile: [32, 64, 128]
    threads: [256, 1, 1]
    grid: [RocRollerStreamK, One, One]
    occupancy: 6
    workspaceType: RocRollerStreamK
    workspaceSizePerElemC: 4
    workspaceSizePerElemBias: 0
    unroll: 2
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2
    .symbol: RR_GEMM_TT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x64x128_UR_2.kd
    .kernarg_segment_size: 192
    .group_segment_fixed_size: 25088
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 91
    .vgpr_count: 72
    .agpr_count: 8
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

