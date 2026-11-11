// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2,@function
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 3414) 
v_and_b32 v3, 1, v0 // (op 3414) 

v_and_b32 v0, 15, v1 // (op 3414) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 3414) 


v_and_b32 v0, 63, v1 // (op 3414) 
v_lshrrev_b32 v3, 4, v0 // (op 3414) 

v_lshl_add_u32 v0, v4, 2, v3 // (op 3414) 


v_lshrrev_b32 v3, 6, v1 // (op 3514) 
v_and_b32 v4, 1, v3 // (op 3514) 

v_and_b32 v3, 15, v1 // (op 3514) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 3514) 


v_and_b32 v3, 63, v1 // (op 3514) 
v_lshrrev_b32 v4, 4, v3 // (op 3514) 

v_lshl_add_u32 v3, v5, 2, v4 // (op 3514) 


v_lshrrev_b32 v4, 6, v1 // (op 3585) 
v_and_b32 v5, 1, v4 // (op 3585) 

v_and_b32 v4, 15, v1 // (op 3585) 
v_lshl_add_u32 v6, v5, 4, v4 // (op 3585) 


v_and_b32 v4, 63, v1 // (op 3585) 
v_lshrrev_b32 v5, 4, v4 // (op 3585) 

v_lshl_add_u32 v4, v6, 2, v5 // (op 3585) 


v_lshrrev_b32 v5, 6, v1 // (op 3614) 
v_and_b32 v6, 1, v5 // (op 3614) 

v_and_b32 v5, 15, v1 // (op 3614) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 3614) 


v_and_b32 v5, 63, v1 // (op 3614) 
v_lshrrev_b32 v6, 4, v5 // (op 3614) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 3614) 


v_mov_b32 v6, s6 // (op 2) Move value
v_mov_b32 v7, s7 // (op 31) Move value


s_mov_b32 s0, 0 // (op 342) call()
s_mov_b32 s1, 0 // (op 343) call()
s_mov_b32 s6, 0 // (op 344) call()
s_mov_b32 s7, 0 // (op 345) call()
s_mov_b32 s18, 0 // (op 346) call()
s_mov_b32 s19, 0 // (op 347) call()
s_mov_b32 s0, 0 // (op 242) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 241) 
s_cmp_gt_u32 s9, s52 // (op 241) 
s_mov_b32 s53, scc // (op 241) 
s_sub_u32 s54, s9, s52 // (op 241) 

s_cmp_eq_i32 s53, 0 // (op 241) 
s_cselect_b32 s52, 0, s54 // (op 241) 


s_cmp_lt_u32 s8, s52 // (op 241) 
s_mov_b32 s53, scc // (op 241) 
s_cmp_eq_i32 s53, 0 // (op 241) 
s_cselect_b32 s54, s52, s8 // (op 241) 


s_cmp_lt_u32 s0, s54 // (op 241) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_241 // (op 241) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_241 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_SKStreamTileLoop_241:
v_lshrrev_b32 v8, 7, v1 // (op 3329) 
s_mul_i32 s52, s16, s8 // (op 3329) 
s_add_u32 s52, s52, s0 // (op 3329) 
v_and_b32 v9, 63, v1 // (op 3329) 
s_mul_hi_u32 s53, s52, s13 // (op 3329) 
v_lshrrev_b32 v10, 4, v9 // (op 3329) 
s_sub_u32 s54, s52, s53 // (op 3329) 
v_lshlrev_b32 v11, 2, v10 // (op 3329) 

v_lshl_add_u32 v10, v8, 4, v11 // (op 3329) 


s_lshr_b32 s55, s54, 1 // (op 3329) 

s_add_u32 s54, s55, s53 // (op 3329) 


v_lshrrev_b32 v8, 2, v9 // (op 3329) 

s_lshr_b32 s53, s54, s11 // (op 3329) 

v_and_b32 v9, 3, v8 // (op 3329) 

s_and_b32 s54, s11, 2147483648 // (op 3329) 
v_and_b32 v8, 3, v1 // (op 3329) 
s_cmp_eq_i32 s54, 0 // (op 3329) 
s_mov_b32 s55, scc // (op 3329) 

v_lshl_add_u32 v11, v9, 2, v8 // (op 3329) 


s_cmp_eq_i32 s55, 0 // (op 3329) 
s_cselect_b32 s54, s52, s53 // (op 3329) 



v_lshrrev_b32 v8, 6, v1 // (op 3329) 
s_mul_hi_u32 s52, s54, s12 // (op 3329) 
v_and_b32 v9, 1, v8 // (op 3329) 

s_sub_u32 s53, s54, s52 // (op 3329) 
v_lshl_add_u32 v8, v9, 4, v11 // (op 3329) 


s_lshr_b32 s55, s53, 1 // (op 3329) 

s_add_u32 s53, s55, s52 // (op 3329) 


s_lshr_b32 s52, s53, s10 // (op 3329) 

s_and_b32 s53, s10, 2147483648 // (op 3329) 
s_cmp_eq_i32 s53, 0 // (op 3329) 
s_mov_b32 s55, scc // (op 3329) 

s_cmp_eq_i32 s55, 0 // (op 3329) 
s_cselect_b32 s53, s54, s52 // (op 3329) 


s_mul_i32 s52, s53, s14 // (op 3329) 
v_lshl_add_u32 v9, s53, 5, v10 // (op 3329) 


s_sub_u32 s53, s54, s52 // (op 3329) 


v_lshl_add_u32 v10, s53, 7, v8 // (op 3329) 


v_mul_lo_u32 v8, v10, s4 // (op 3329) 

v_add_lshl_u32 v10, v9, v8, 2 // (op 3329) 


s_mul_i32 s53, s4, s23 // (op 3331) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 3331) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 3331) least significant
s_mul_hi_u32 s55, s4, s22 // (op 3331) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3331) most significant: sum
s_add_u32 s53, s53, s55 // (op 3331) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 3331) 

s_mov_b32 s56, s20 // (op 3331) reg expression
s_mov_b32 s57, s21 // (op 3331) reg expression
s_mov_b32 s58, s54 // (op 3331) reg expression
s_mov_b32 s59, 131072 // (op 3331) call()

v_lshrrev_b32 v8, 7, v1 // (op 3431) 
s_mul_i32 s52, s16, s8 // (op 3431) 
s_add_u32 s52, s52, s0 // (op 3431) 
v_and_b32 v9, 63, v1 // (op 3431) 
s_mul_hi_u32 s53, s52, s13 // (op 3431) 
v_lshrrev_b32 v11, 4, v9 // (op 3431) 
s_sub_u32 s54, s52, s53 // (op 3431) 
v_lshlrev_b32 v12, 2, v11 // (op 3431) 

s_lshr_b32 s55, s54, 1 // (op 3431) 

v_lshl_add_u32 v11, v8, 4, v12 // (op 3431) 


s_add_u32 s54, s55, s53 // (op 3431) 


v_lshrrev_b32 v8, 2, v9 // (op 3431) 

s_lshr_b32 s53, s54, s11 // (op 3431) 

v_and_b32 v9, 3, v8 // (op 3431) 

s_and_b32 s54, s11, 2147483648 // (op 3431) 
v_and_b32 v8, 3, v1 // (op 3431) 
s_cmp_eq_i32 s54, 0 // (op 3431) 
s_mov_b32 s55, scc // (op 3431) 

v_lshl_add_u32 v12, v9, 2, v8 // (op 3431) 


s_cmp_eq_i32 s55, 0 // (op 3431) 
s_cselect_b32 s54, s52, s53 // (op 3431) 



v_lshrrev_b32 v8, 6, v1 // (op 3431) 
s_mul_hi_u32 s55, s54, s12 // (op 3431) 
v_and_b32 v9, 1, v8 // (op 3431) 

s_sub_u32 s52, s54, s55 // (op 3431) 
v_lshl_add_u32 v8, v9, 4, v12 // (op 3431) 


s_lshr_b32 s53, s52, 1 // (op 3431) 

s_add_u32 s52, s53, s55 // (op 3431) 


s_lshr_b32 s53, s52, s10 // (op 3431) 

s_and_b32 s52, s10, 2147483648 // (op 3431) 
s_cmp_eq_i32 s52, 0 // (op 3431) 
s_mov_b32 s55, scc // (op 3431) 

s_cmp_eq_i32 s55, 0 // (op 3431) 
s_cselect_b32 s52, s54, s53 // (op 3431) 


s_mul_i32 s53, s52, s14 // (op 3431) 
v_lshl_add_u32 v9, s52, 5, v11 // (op 3431) 


s_sub_u32 s52, s54, s53 // (op 3431) 


v_lshl_add_u32 v11, s52, 7, v8 // (op 3431) 


v_mul_lo_u32 v8, v11, s24 // (op 3431) 

v_add_lshl_u32 v11, v9, v8, 2 // (op 3431) 


s_mul_i32 s53, s24, s23 // (op 3433) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 3433) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 3433) least significant
s_mul_hi_u32 s55, s24, s22 // (op 3433) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3433) most significant: sum
s_add_u32 s53, s53, s55 // (op 3433) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 3433) 

s_mov_b32 s60, s26 // (op 3433) reg expression
s_mov_b32 s61, s27 // (op 3433) reg expression
s_mov_b32 s62, s54 // (op 3433) reg expression
s_mov_b32 s63, 131072 // (op 3433) call()

v_lshrrev_b32 v8, 5, v1 // (op 3531) 
v_lshlrev_b32 v9, 2, v8 // (op 3531) 

v_lshl_add_u32 v8, s16, 5, v9 // (op 3531) 

v_and_b32 v9, -32, v1 // (op 3531) 
v_sub_u32 v12, v1, v9 // (op 3531) 

v_lshl_add_u32 v9, v8, 7, v12 // (op 3531) 


v_lshlrev_b32 v8, 2, v9 // (op 3531) 

s_lshl_b32 s52, s15, 14 // (op 3533) 
s_mov_b32 s64, s28 // (op 3533) reg expression
s_mov_b32 s65, s29 // (op 3533) reg expression
s_mov_b32 s66, s52 // (op 3533) reg expression
s_mov_b32 s67, 131072 // (op 3533) call()

s_mul_i32 s6, s8, s16 // (op 246) 
s_add_u32 s6, s6, s0 // (op 246) 
s_mul_hi_u32 s52, s6, s13 // (op 247) 
s_sub_u32 s53, s6, s52 // (op 247) 
s_lshr_b32 s54, s53, 1 // (op 247) 

s_add_u32 s53, s54, s52 // (op 247) 


s_lshr_b32 s52, s53, s11 // (op 247) 

s_and_b32 s53, s11, 2147483648 // (op 247) 
s_cmp_eq_i32 s53, 0 // (op 247) 
s_mov_b32 s54, scc // (op 247) 

s_cmp_eq_i32 s54, 0 // (op 247) 
s_cselect_b32 s53, s6, s52 // (op 247) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 247) 
s_mul_i32 s52, s53, s54 // (op 247) 


s_sub_u32 s18, s6, s52 // (op 247) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 248) 
s_cmp_gt_u32 s52, s18 // (op 248) 
s_mov_b32 s54, scc // (op 248) 
s_sub_u32 s55, s52, s18 // (op 248) 

s_cmp_eq_i32 s54, 0 // (op 248) 
s_cselect_b32 s52, 0, s55 // (op 248) 


s_cmp_gt_u32 s8, s0 // (op 248) 
s_mov_b32 s53, scc // (op 248) 
s_sub_u32 s54, s8, s0 // (op 248) 
s_cmp_eq_i32 s53, 0 // (op 248) 
s_cselect_b32 s55, 0, s54 // (op 248) 


s_cmp_lt_u32 s52, s55 // (op 248) 
s_mov_b32 s53, scc // (op 248) 
s_cmp_eq_i32 s53, 0 // (op 248) 
s_cselect_b32 s54, s55, s52 // (op 248) 



s_cmp_gt_u32 s9, s6 // (op 248) 
s_mov_b32 s55, scc // (op 248) 
s_sub_u32 s52, s9, s6 // (op 248) 
s_cmp_eq_i32 s55, 0 // (op 248) 
s_cselect_b32 s53, 0, s52 // (op 248) 


s_cmp_lt_u32 s54, s53 // (op 248) 
s_mov_b32 s52, scc // (op 248) 
s_cmp_eq_i32 s52, 0 // (op 248) 
s_cselect_b32 s7, s53, s54 // (op 248) 



s_add_u32 s52, s18, s7 // (op 249) 
s_sub_u32 s19, s52, 1 // (op 249) 

v_accvgpr_write a0, 0.00000 // (op 51) call()
v_accvgpr_write a1, 0.00000 // (op 51) call()
v_accvgpr_write a2, 0.00000 // (op 51) call()
v_accvgpr_write a3, 0.00000 // (op 51) call()
v_accvgpr_write a4, 0.00000 // (op 1176) call()
v_accvgpr_write a5, 0.00000 // (op 1176) call()
v_accvgpr_write a6, 0.00000 // (op 1176) call()
v_accvgpr_write a7, 0.00000 // (op 1176) call()
v_accvgpr_write a8, 0.00000 // (op 1292) call()
v_accvgpr_write a9, 0.00000 // (op 1292) call()
v_accvgpr_write a10, 0.00000 // (op 1292) call()
v_accvgpr_write a11, 0.00000 // (op 1292) call()
v_accvgpr_write a12, 0.00000 // (op 1408) call()
v_accvgpr_write a13, 0.00000 // (op 1408) call()
v_accvgpr_write a14, 0.00000 // (op 1408) call()
v_accvgpr_write a15, 0.00000 // (op 1408) call()
v_lshrrev_b32 v9, 7, v1 // (op 2871) 
s_mul_i32 s52, s16, s8 // (op 2871) 
s_add_u32 s52, s52, s0 // (op 2871) 
v_and_b32 v12, 15, v1 // (op 2871) 
s_mul_hi_u32 s53, s52, s13 // (op 2871) 
v_lshl_add_u32 v13, v9, 4, v12 // (op 2871) 


s_sub_u32 s54, s52, s53 // (op 2871) 
v_and_b32 v9, 63, v1 // (op 2871) 
s_lshr_b32 s55, s54, 1 // (op 2871) 

v_lshrrev_b32 v12, 4, v9 // (op 2871) 

s_add_u32 s54, s55, s53 // (op 2871) 


v_lshlrev_b32 v9, 5, v12 // (op 2871) 

s_lshr_b32 s53, s54, s11 // (op 2871) 

s_and_b32 s54, s11, 2147483648 // (op 2871) 
s_cmp_eq_i32 s54, 0 // (op 2871) 
s_mov_b32 s55, scc // (op 2871) 

s_cmp_eq_i32 s55, 0 // (op 2871) 
s_cselect_b32 s54, s52, s53 // (op 2871) 


s_mul_hi_u32 s53, s54, s12 // (op 2871) 
s_sub_u32 s55, s54, s53 // (op 2871) 
s_lshr_b32 s68, s55, 1 // (op 2871) 

s_add_u32 s55, s68, s53 // (op 2871) 


s_lshr_b32 s53, s55, s10 // (op 2871) 

s_and_b32 s55, s10, 2147483648 // (op 2871) 
s_cmp_eq_i32 s55, 0 // (op 2871) 
s_mov_b32 s68, scc // (op 2871) 

s_cmp_eq_i32 s68, 0 // (op 2871) 
s_cselect_b32 s55, s54, s53 // (op 2871) 


v_lshl_add_u32 v12, s55, 5, v13 // (op 2871) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 2871) 
s_mul_i32 s53, s54, s68 // (op 2871) 


s_sub_u32 s54, s52, s53 // (op 2871) 


v_lshl_add_u32 v13, s54, 7, v9 // (op 2871) 


v_mov_b32 v15, 0 // (op 2871) convert
v_mov_b32 v14, v13 // (op 2871) convert

v_mul_lo_u32 v17, v12, s31 // (op 2871) most significant: low of low * high
v_mul_hi_u32 v9, v12, s30 // (op 2871) most significant: high of low * low
v_mul_lo_u32 v16, v12, s30 // (op 2871) least significant: low of low * low
v_add_u32 v17, v17, v9 // (op 2871) most significant: sum

v_add_co_u32 v16, s[52:53], v16, v14 // (op 2871) least significant half
v_addc_co_u32 v17, s[52:53], v17, v15, s[52:53] // (op 2871) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 2871) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 2871) 

v_mov_b32 v9, v14 // (op 2871) convert

s_mul_i32 s53, s30, s35 // (op 2873) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 2873) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 2873) least significant
s_mul_hi_u32 s55, s30, s34 // (op 2873) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2873) most significant: sum
s_add_u32 s53, s53, s55 // (op 2873) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 2873) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 2873) 

s_mov_b32 s68, s32 // (op 2873) reg expression
s_mov_b32 s69, s33 // (op 2873) reg expression
s_mov_b32 s70, s52 // (op 2873) reg expression
s_mov_b32 s71, 131072 // (op 2873) call()

v_lshrrev_b32 v12, 7, v1 // (op 2890) 
s_mul_i32 s52, s16, s8 // (op 2890) 
s_add_u32 s52, s52, s0 // (op 2890) 
v_and_b32 v13, 15, v1 // (op 2890) 
s_mul_hi_u32 s53, s52, s13 // (op 2890) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 2890) 


s_sub_u32 s54, s52, s53 // (op 2890) 
v_and_b32 v12, 63, v1 // (op 2890) 
s_lshr_b32 s55, s54, 1 // (op 2890) 

v_lshrrev_b32 v13, 4, v12 // (op 2890) 

s_add_u32 s54, s55, s53 // (op 2890) 


v_lshlrev_b32 v12, 5, v13 // (op 2890) 

s_lshr_b32 s53, s54, s11 // (op 2890) 

s_and_b32 s54, s11, 2147483648 // (op 2890) 
s_cmp_eq_i32 s54, 0 // (op 2890) 
s_mov_b32 s55, scc // (op 2890) 

s_cmp_eq_i32 s55, 0 // (op 2890) 
s_cselect_b32 s54, s52, s53 // (op 2890) 


s_mul_hi_u32 s53, s54, s12 // (op 2890) 
s_sub_u32 s55, s54, s53 // (op 2890) 
s_lshr_b32 s72, s55, 1 // (op 2890) 

s_add_u32 s55, s72, s53 // (op 2890) 


s_lshr_b32 s53, s55, s10 // (op 2890) 

s_and_b32 s55, s10, 2147483648 // (op 2890) 
s_cmp_eq_i32 s55, 0 // (op 2890) 
s_mov_b32 s72, scc // (op 2890) 

s_cmp_eq_i32 s72, 0 // (op 2890) 
s_cselect_b32 s55, s54, s53 // (op 2890) 


v_lshl_add_u32 v13, s55, 5, v14 // (op 2890) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 2890) 
s_mul_i32 s53, s54, s72 // (op 2890) 


s_sub_u32 s54, s52, s53 // (op 2890) 


v_lshl_add_u32 v14, s54, 7, v12 // (op 2890) 


v_mov_b32 v17, 0 // (op 2890) convert
v_mov_b32 v16, v14 // (op 2890) convert

v_mul_lo_u32 v15, v13, s31 // (op 2890) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 2890) most significant: high of low * low
v_mul_lo_u32 v14, v13, s30 // (op 2890) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 2890) most significant: sum

v_add_co_u32 v14, s[52:53], v14, v16 // (op 2890) least significant half
v_addc_co_u32 v15, s[52:53], v15, v17, s[52:53] // (op 2890) most significant half



v_lshlrev_b64 v[12:13], 2, v[14:15] // (op 2890) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 2890) 

v_mov_b32 v12, v14 // (op 2890) convert

s_mul_i32 s53, s30, s35 // (op 2892) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 2892) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 2892) least significant
s_mul_hi_u32 s55, s30, s34 // (op 2892) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2892) most significant: sum
s_add_u32 s53, s53, s55 // (op 2892) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 2892) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 2892) 

s_mov_b32 s68, s32 // (op 2892) reg expression
s_mov_b32 s69, s33 // (op 2892) reg expression
s_mov_b32 s70, s52 // (op 2892) reg expression
s_mov_b32 s71, 131072 // (op 2892) call()

v_lshrrev_b32 v13, 6, v1 // (op 3024) 
s_mul_i32 s52, s16, s8 // (op 3024) 
s_add_u32 s52, s52, s0 // (op 3024) 
v_and_b32 v14, 1, v13 // (op 3024) 

s_mul_hi_u32 s53, s52, s13 // (op 3024) 
v_and_b32 v13, 15, v1 // (op 3024) 
s_sub_u32 s54, s52, s53 // (op 3024) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 3024) 


s_lshr_b32 s55, s54, 1 // (op 3024) 

v_and_b32 v13, 63, v1 // (op 3024) 
s_add_u32 s54, s55, s53 // (op 3024) 


v_lshrrev_b32 v14, 4, v13 // (op 3024) 

s_lshr_b32 s53, s54, s11 // (op 3024) 

v_lshlrev_b32 v13, 5, v14 // (op 3024) 

s_and_b32 s54, s11, 2147483648 // (op 3024) 
s_cmp_eq_i32 s54, 0 // (op 3024) 
s_mov_b32 s55, scc // (op 3024) 

s_cmp_eq_i32 s55, 0 // (op 3024) 
s_cselect_b32 s54, s52, s53 // (op 3024) 


s_mul_hi_u32 s53, s54, s12 // (op 3024) 
s_sub_u32 s55, s54, s53 // (op 3024) 
s_lshr_b32 s72, s55, 1 // (op 3024) 

s_add_u32 s55, s72, s53 // (op 3024) 


s_lshr_b32 s53, s55, s10 // (op 3024) 

s_and_b32 s55, s10, 2147483648 // (op 3024) 
s_cmp_eq_i32 s55, 0 // (op 3024) 
s_mov_b32 s72, scc // (op 3024) 

s_cmp_eq_i32 s72, 0 // (op 3024) 
s_cselect_b32 s55, s54, s53 // (op 3024) 


s_mul_i32 s53, s55, s14 // (op 3024) 

s_sub_u32 s55, s54, s53 // (op 3024) 

v_lshl_add_u32 v14, s55, 7, v15 // (op 3024) 


v_mul_lo_u32 v17, v14, s37 // (op 3024) most significant: low of low * high
v_mul_hi_u32 v15, v14, s36 // (op 3024) most significant: high of low * low
v_mul_lo_u32 v16, v14, s36 // (op 3024) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 3024) most significant: sum


s_ashr_i64 s[72:73], s[2:3], 7 // (op 3024) 
s_mul_i32 s53, s54, s72 // (op 3024) 


s_sub_u32 s54, s52, s53 // (op 3024) 


v_lshl_add_u32 v14, s54, 7, v13 // (op 3024) 


v_mov_b32 v19, 0 // (op 3024) convert
v_mov_b32 v18, v14 // (op 3024) convert

v_add_co_u32 v14, s[52:53], v18, v16 // (op 3024) least significant half
v_addc_co_u32 v15, s[52:53], v19, v17, s[52:53] // (op 3024) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 3024) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 3024) 

v_mov_b32 v13, v16 // (op 3024) convert

s_mul_i32 s53, s36, s23 // (op 3026) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 3026) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 3026) least significant
s_mul_hi_u32 s55, s36, s22 // (op 3026) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3026) most significant: sum
s_add_u32 s53, s53, s55 // (op 3026) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 3026) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 3026) 

s_mov_b32 s72, s38 // (op 3026) reg expression
s_mov_b32 s73, s39 // (op 3026) reg expression
s_mov_b32 s74, s52 // (op 3026) reg expression
s_mov_b32 s75, 131072 // (op 3026) call()

v_lshrrev_b32 v14, 6, v1 // (op 3045) 
s_mul_i32 s52, s16, s8 // (op 3045) 
s_add_u32 s52, s52, s0 // (op 3045) 
v_and_b32 v15, 1, v14 // (op 3045) 

s_mul_hi_u32 s53, s52, s13 // (op 3045) 
v_and_b32 v14, 15, v1 // (op 3045) 
s_sub_u32 s54, s52, s53 // (op 3045) 
v_lshl_add_u32 v16, v15, 4, v14 // (op 3045) 


s_lshr_b32 s55, s54, 1 // (op 3045) 

v_and_b32 v14, 63, v1 // (op 3045) 
s_add_u32 s54, s55, s53 // (op 3045) 


v_lshrrev_b32 v15, 4, v14 // (op 3045) 

s_lshr_b32 s53, s54, s11 // (op 3045) 

v_lshlrev_b32 v14, 5, v15 // (op 3045) 

s_and_b32 s54, s11, 2147483648 // (op 3045) 
s_cmp_eq_i32 s54, 0 // (op 3045) 
s_mov_b32 s55, scc // (op 3045) 

s_cmp_eq_i32 s55, 0 // (op 3045) 
s_cselect_b32 s54, s52, s53 // (op 3045) 


s_mul_hi_u32 s53, s54, s12 // (op 3045) 
s_sub_u32 s55, s54, s53 // (op 3045) 
s_lshr_b32 s76, s55, 1 // (op 3045) 

s_add_u32 s55, s76, s53 // (op 3045) 


s_lshr_b32 s53, s55, s10 // (op 3045) 

s_and_b32 s55, s10, 2147483648 // (op 3045) 
s_cmp_eq_i32 s55, 0 // (op 3045) 
s_mov_b32 s76, scc // (op 3045) 

s_cmp_eq_i32 s76, 0 // (op 3045) 
s_cselect_b32 s55, s54, s53 // (op 3045) 


s_mul_i32 s53, s55, s14 // (op 3045) 

s_sub_u32 s55, s54, s53 // (op 3045) 

v_lshl_add_u32 v15, s55, 7, v16 // (op 3045) 


v_mul_lo_u32 v17, v15, s37 // (op 3045) most significant: low of low * high
v_mul_hi_u32 v18, v15, s36 // (op 3045) most significant: high of low * low
v_mul_lo_u32 v16, v15, s36 // (op 3045) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 3045) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 3045) 
s_mul_i32 s53, s54, s76 // (op 3045) 


s_sub_u32 s54, s52, s53 // (op 3045) 


v_lshl_add_u32 v15, s54, 7, v14 // (op 3045) 


v_mov_b32 v19, 0 // (op 3045) convert
v_mov_b32 v18, v15 // (op 3045) convert

v_add_co_u32 v14, s[52:53], v18, v16 // (op 3045) least significant half
v_addc_co_u32 v15, s[52:53], v19, v17, s[52:53] // (op 3045) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 3045) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 3045) 

v_mov_b32 v14, v16 // (op 3045) convert

s_mul_i32 s53, s36, s23 // (op 3047) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 3047) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 3047) least significant
s_mul_hi_u32 s55, s36, s22 // (op 3047) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3047) most significant: sum
s_add_u32 s53, s53, s55 // (op 3047) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 3047) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 3047) 

s_mov_b32 s72, s38 // (op 3047) reg expression
s_mov_b32 s73, s39 // (op 3047) reg expression
s_mov_b32 s74, s52 // (op 3047) reg expression
s_mov_b32 s75, 131072 // (op 3047) call()

v_lshrrev_b32 v15, 7, v1 // (op 3157) 
s_mul_i32 s52, s16, s8 // (op 3157) 
s_add_u32 s52, s52, s0 // (op 3157) 
v_and_b32 v16, 15, v1 // (op 3157) 
s_mul_hi_u32 s53, s52, s13 // (op 3157) 
v_lshl_add_u32 v17, v15, 4, v16 // (op 3157) 


s_sub_u32 s54, s52, s53 // (op 3157) 
v_and_b32 v15, 63, v1 // (op 3157) 
s_lshr_b32 s55, s54, 1 // (op 3157) 

v_lshrrev_b32 v16, 4, v15 // (op 3157) 

s_add_u32 s54, s55, s53 // (op 3157) 


s_lshr_b32 s53, s54, s11 // (op 3157) 

s_and_b32 s54, s11, 2147483648 // (op 3157) 
s_cmp_eq_i32 s54, 0 // (op 3157) 
s_mov_b32 s55, scc // (op 3157) 

s_cmp_eq_i32 s55, 0 // (op 3157) 
s_cselect_b32 s54, s52, s53 // (op 3157) 


s_mul_hi_u32 s53, s54, s12 // (op 3157) 
s_sub_u32 s55, s54, s53 // (op 3157) 
s_lshr_b32 s76, s55, 1 // (op 3157) 

s_add_u32 s55, s76, s53 // (op 3157) 


s_lshr_b32 s53, s55, s10 // (op 3157) 

s_and_b32 s55, s10, 2147483648 // (op 3157) 
s_cmp_eq_i32 s55, 0 // (op 3157) 
s_mov_b32 s76, scc // (op 3157) 

s_cmp_eq_i32 s76, 0 // (op 3157) 
s_cselect_b32 s55, s54, s53 // (op 3157) 


v_lshl_add_u32 v15, s55, 5, v17 // (op 3157) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 3157) 
s_mul_i32 s53, s54, s76 // (op 3157) 


s_sub_u32 s54, s52, s53 // (op 3157) 


v_lshl_add_u32 v17, s54, 2, v16 // (op 3157) 


v_mul_lo_u32 v16, v15, s40 // (op 3157) 
v_add_u32 v16, v16, v17 // (op 3157) 


s_mul_i32 s53, s40, s35 // (op 3159) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 3159) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 3159) least significant
s_mul_hi_u32 s55, s40, s34 // (op 3159) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3159) most significant: sum
s_add_u32 s53, s53, s55 // (op 3159) most significant: sum


s_mov_b32 s76, s42 // (op 3159) reg expression
s_mov_b32 s77, s43 // (op 3159) reg expression
s_mov_b32 s78, s52 // (op 3159) reg expression
s_mov_b32 s79, 131072 // (op 3159) call()

v_lshrrev_b32 v15, 7, v1 // (op 3176) 
s_mul_i32 s52, s16, s8 // (op 3176) 
s_add_u32 s52, s52, s0 // (op 3176) 
v_and_b32 v17, 15, v1 // (op 3176) 
s_mul_hi_u32 s53, s52, s13 // (op 3176) 
v_lshl_add_u32 v18, v15, 4, v17 // (op 3176) 


s_sub_u32 s54, s52, s53 // (op 3176) 
v_and_b32 v15, 63, v1 // (op 3176) 
s_lshr_b32 s55, s54, 1 // (op 3176) 

v_lshrrev_b32 v17, 4, v15 // (op 3176) 

s_add_u32 s54, s55, s53 // (op 3176) 


s_lshr_b32 s53, s54, s11 // (op 3176) 

s_and_b32 s54, s11, 2147483648 // (op 3176) 
s_cmp_eq_i32 s54, 0 // (op 3176) 
s_mov_b32 s55, scc // (op 3176) 

s_cmp_eq_i32 s55, 0 // (op 3176) 
s_cselect_b32 s54, s52, s53 // (op 3176) 


s_mul_hi_u32 s53, s54, s12 // (op 3176) 
s_sub_u32 s55, s54, s53 // (op 3176) 
s_lshr_b32 s80, s55, 1 // (op 3176) 

s_add_u32 s55, s80, s53 // (op 3176) 


s_lshr_b32 s53, s55, s10 // (op 3176) 

s_and_b32 s55, s10, 2147483648 // (op 3176) 
s_cmp_eq_i32 s55, 0 // (op 3176) 
s_mov_b32 s80, scc // (op 3176) 

s_cmp_eq_i32 s80, 0 // (op 3176) 
s_cselect_b32 s55, s54, s53 // (op 3176) 


v_lshl_add_u32 v15, s55, 5, v18 // (op 3176) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 3176) 
s_mul_i32 s53, s54, s80 // (op 3176) 


s_sub_u32 s54, s52, s53 // (op 3176) 


v_lshl_add_u32 v18, s54, 2, v17 // (op 3176) 


v_mul_lo_u32 v17, v15, s40 // (op 3176) 
v_add_u32 v17, v17, v18 // (op 3176) 


s_mul_i32 s53, s40, s35 // (op 3178) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 3178) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 3178) least significant
s_mul_hi_u32 s55, s40, s34 // (op 3178) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3178) most significant: sum
s_add_u32 s53, s53, s55 // (op 3178) most significant: sum


s_mov_b32 s76, s42 // (op 3178) reg expression
s_mov_b32 s77, s43 // (op 3178) reg expression
s_mov_b32 s78, s52 // (op 3178) reg expression
s_mov_b32 s79, 131072 // (op 3178) call()

v_lshrrev_b32 v15, 2, v1 // (op 3285) 
s_mul_i32 s52, s16, s8 // (op 3285) 
s_add_u32 s52, s52, s0 // (op 3285) 
v_lshlrev_b32 v18, 1, v15 // (op 3285) 

s_mul_hi_u32 s53, s52, s13 // (op 3285) 
v_and_b32 v15, -4, v1 // (op 3285) 
s_sub_u32 s54, s52, s53 // (op 3285) 
v_sub_u32 v19, v1, v15 // (op 3285) 

s_lshr_b32 s55, s54, 1 // (op 3285) 

s_add_u32 s54, s55, s53 // (op 3285) 


s_lshr_b32 s53, s54, s11 // (op 3285) 

s_and_b32 s54, s11, 2147483648 // (op 3285) 
s_cmp_eq_i32 s54, 0 // (op 3285) 
s_mov_b32 s55, scc // (op 3285) 

s_cmp_eq_i32 s55, 0 // (op 3285) 
s_cselect_b32 s54, s52, s53 // (op 3285) 


s_mul_hi_u32 s53, s54, s12 // (op 3285) 
s_sub_u32 s55, s54, s53 // (op 3285) 
s_lshr_b32 s80, s55, 1 // (op 3285) 

s_add_u32 s55, s80, s53 // (op 3285) 


s_lshr_b32 s53, s55, s10 // (op 3285) 

s_and_b32 s55, s10, 2147483648 // (op 3285) 
s_cmp_eq_i32 s55, 0 // (op 3285) 
s_mov_b32 s80, scc // (op 3285) 

s_cmp_eq_i32 s80, 0 // (op 3285) 
s_cselect_b32 s55, s54, s53 // (op 3285) 


s_mul_i32 s53, s55, s14 // (op 3285) 

s_sub_u32 s55, s54, s53 // (op 3285) 

v_lshl_add_u32 v15, s55, 7, v18 // (op 3285) 


v_mul_lo_u32 v18, v15, s44 // (op 3285) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 3285) 
s_mul_i32 s53, s54, s80 // (op 3285) 


s_sub_u32 s54, s52, s53 // (op 3285) 


v_lshl_add_u32 v15, s54, 2, v19 // (op 3285) 


v_add_u32 v19, v15, v18 // (op 3285) 


s_mul_i32 s53, s44, s23 // (op 3287) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 3287) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 3287) least significant
s_mul_hi_u32 s55, s44, s22 // (op 3287) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3287) most significant: sum
s_add_u32 s53, s53, s55 // (op 3287) most significant: sum


s_mov_b32 s80, s46 // (op 3287) reg expression
s_mov_b32 s81, s47 // (op 3287) reg expression
s_mov_b32 s82, s52 // (op 3287) reg expression
s_mov_b32 s83, 131072 // (op 3287) call()

v_lshrrev_b32 v15, 2, v1 // (op 3304) 
s_mul_i32 s52, s16, s8 // (op 3304) 
s_add_u32 s52, s52, s0 // (op 3304) 
v_lshlrev_b32 v18, 1, v15 // (op 3304) 

s_mul_hi_u32 s53, s52, s13 // (op 3304) 
v_and_b32 v15, -4, v1 // (op 3304) 
s_sub_u32 s54, s52, s53 // (op 3304) 
v_sub_u32 v20, v1, v15 // (op 3304) 

s_lshr_b32 s55, s54, 1 // (op 3304) 

s_add_u32 s54, s55, s53 // (op 3304) 


s_lshr_b32 s53, s54, s11 // (op 3304) 

s_and_b32 s54, s11, 2147483648 // (op 3304) 
s_cmp_eq_i32 s54, 0 // (op 3304) 
s_mov_b32 s55, scc // (op 3304) 

s_cmp_eq_i32 s55, 0 // (op 3304) 
s_cselect_b32 s54, s52, s53 // (op 3304) 


s_mul_hi_u32 s53, s54, s12 // (op 3304) 
s_sub_u32 s55, s54, s53 // (op 3304) 
s_lshr_b32 s84, s55, 1 // (op 3304) 

s_add_u32 s55, s84, s53 // (op 3304) 


s_lshr_b32 s53, s55, s10 // (op 3304) 

s_and_b32 s55, s10, 2147483648 // (op 3304) 
s_cmp_eq_i32 s55, 0 // (op 3304) 
s_mov_b32 s84, scc // (op 3304) 

s_cmp_eq_i32 s84, 0 // (op 3304) 
s_cselect_b32 s55, s54, s53 // (op 3304) 


s_mul_i32 s53, s55, s14 // (op 3304) 

s_sub_u32 s55, s54, s53 // (op 3304) 

v_lshl_add_u32 v15, s55, 7, v18 // (op 3304) 


v_mul_lo_u32 v18, v15, s44 // (op 3304) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 3304) 
s_mul_i32 s53, s54, s84 // (op 3304) 


s_sub_u32 s54, s52, s53 // (op 3304) 


v_lshl_add_u32 v15, s54, 2, v20 // (op 3304) 


v_add_u32 v20, v15, v18 // (op 3304) 


s_mul_i32 s53, s44, s23 // (op 3306) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 3306) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 3306) least significant
s_mul_hi_u32 s55, s44, s22 // (op 3306) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3306) most significant: sum
s_add_u32 s53, s53, s55 // (op 3306) most significant: sum


s_mov_b32 s80, s46 // (op 3306) reg expression
s_mov_b32 s81, s47 // (op 3306) reg expression
s_mov_b32 s82, s52 // (op 3306) reg expression
s_mov_b32 s83, 131072 // (op 3306) call()




buffer_load_dwordx4 v[22:25], v9, s[68:71], 0 offen // (op 2602) Load value


v_mov_b32 v15, v13 // (op 2608) reg expression

buffer_load_dwordx4 v[26:29], v15, s[72:75], 0 offen // (op 2608) Load value




buffer_load_ubyte v15, v16, s[76:79], 0 offen offset: 0 // (op 2614) Load value


s_lshl_b64 s[52:53], s[36:37], 5 // (op 2620) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2620) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2620) 

v_add_u32 v18, v13, s52 // (op 2620) 


buffer_load_dwordx4 v[30:33], v18, s[72:75], 0 offen // (op 2620) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 2627) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2627) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2627) 

s_lshl_b32 s54, s52, 1 // (op 2627) 

v_add_u32 v18, v13, s54 // (op 2627) 


buffer_load_dwordx4 v[34:37], v18, s[72:75], 0 offen // (op 2627) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 2634) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2634) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2634) 

s_mul_i32 s54, 3, s52 // (op 2634) 

v_add_u32 v18, v13, s54 // (op 2634) 


buffer_load_dwordx4 v[38:41], v18, s[72:75], 0 offen // (op 2634) Load value


v_mov_b64 v[42:43], s[44:45] // (op 2639) convert

v_mov_b32 v18, v19 // (op 2639) 
buffer_load_ubyte v44, v18, s[80:83], 0 offen offset: 0 // (op 2639) Load value
v_add_u32 v18, v18, v42 // (op 2639) 
buffer_load_ubyte v45, v18, s[80:83], 0 offen offset: 0 // (op 2639) Load value





buffer_load_dwordx4 v[46:49], v9, s[68:71], 0 offen offset:64 // (op 2643) Load value



v_add_u32 v9, 64, v13 // (op 2649) 

buffer_load_dwordx4 v[50:53], v9, s[72:75], 0 offen // (op 2649) Load value




buffer_load_ubyte v9, v16, s[76:79], 0 offen offset:4 // (op 2655) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 2661) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2661) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2661) 

s_add_u32 s54, s52, 64 // (op 2661) 

v_add_u32 v16, v13, s54 // (op 2661) 


buffer_load_dwordx4 v[54:57], v16, s[72:75], 0 offen // (op 2661) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 2668) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2668) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2668) 

s_lshl_b32 s54, s52, 1 // (op 2668) 
s_add_u32 s54, s54, 64 // (op 2668) 

v_add_u32 v16, v13, s54 // (op 2668) 


buffer_load_dwordx4 v[58:61], v16, s[72:75], 0 offen // (op 2668) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 2675) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2675) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2675) 

s_mul_i32 s54, 3, s52 // (op 2675) 
s_add_u32 s54, s54, 64 // (op 2675) 

v_add_u32 v16, v13, s54 // (op 2675) 


buffer_load_dwordx4 v[62:65], v16, s[72:75], 0 offen // (op 2675) Load value



v_mov_b64 v[42:43], s[44:45] // (op 2680) convert

v_mov_b32 v13, v19 // (op 2680) 
buffer_load_ubyte v66, v13, s[80:83], 0 offen offset:4 // (op 2680) Load value
v_add_u32 v13, v13, v42 // (op 2680) 
buffer_load_ubyte v67, v13, s[80:83], 0 offen offset:4 // (op 2680) Load value



v_and_b32 v13, -4, v1 // (op 3397) 
v_sub_u32 v16, v1, v13 // (op 3397) 

v_lshrrev_b32 v13, 2, v1 // (op 3397) 
v_lshl_add_u32 v21, v13, 3, v16 // (op 3397) 


s_barrier  // (op 3709) 



s_waitcnt vmcnt(9)
ds_write_b8 v21, v44 // (op 2682) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v21, v45 offset:4 // (op 2682) Store local data 

v_and_b32 v13, -4, v1 // (op 3368) 
v_sub_u32 v16, v1, v13 // (op 3368) 

v_lshrrev_b32 v13, 2, v1 // (op 3368) 
v_lshl_add_u32 v21, v13, 3, v16 // (op 3368) 


v_and_b32 v13, -4, v1 // (op 3597) 
v_sub_u32 v16, v1, v13 // (op 3597) 

v_lshrrev_b32 v13, 2, v1 // (op 3597) 
v_lshl_add_u32 v18, v13, 3, v16 // (op 3597) 


s_mov_b32 s1, 0 // (op 1410) call()
s_and_b32 s52, s7, 4294967294 // (op 1409) 
s_cmp_lt_u32 s1, s52 // (op 1409) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_1409 // (op 1409) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_1409 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_1409:
s_add_u32 s1, s1, 2 // (op 1411) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3706) 



ds_read_u8 v13, v0 // (op 67) Load local data 



ds_read_u8 v16, v0 offset:128 // (op 1203) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[26:29], a[0:3], v15, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 70) 




ds_read_u8 v13, v0 offset:256 // (op 1319) Load local data 



ds_read_u8 v19, v0 offset:384 // (op 1435) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:25], v[30:33], a[4:7], v15, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1204) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:25], v[34:37], a[8:11], v15, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1320) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[22:25], v[38:41], a[12:15], v15, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1436) 



v_mov_b32 v13, v12 // (op 55) 

buffer_load_dwordx4 v[22:25], v13, s[68:71], 0 offen offset:128 // (op 55) Load value


v_mov_b32 v16, v14 // (op 59) 
v_add_u32 v19, 128, v16 // (op 59) 

buffer_load_dwordx4 v[26:29], v19, s[72:75], 0 offen // (op 59) Load value



v_mov_b32 v19, v17 // (op 63) 

buffer_load_ubyte v15, v19, s[76:79], 0 offen offset:8 // (op 63) Load value


s_lshl_b64 s[52:53], s[36:37], 5 // (op 1226) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1226) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1226) 

s_add_u32 s54, s52, 128 // (op 1226) 

v_add_u32 v42, v16, s54 // (op 1226) 


buffer_load_dwordx4 v[30:33], v42, s[72:75], 0 offen // (op 1226) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 1342) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1342) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1342) 

s_lshl_b32 s54, s52, 1 // (op 1342) 
s_add_u32 s54, s54, 128 // (op 1342) 

v_add_u32 v42, v16, s54 // (op 1342) 


buffer_load_dwordx4 v[34:37], v42, s[72:75], 0 offen // (op 1342) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 1458) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1458) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1458) 

s_mul_i32 s54, 3, s52 // (op 1458) 
s_add_u32 s54, s54, 128 // (op 1458) 

v_add_u32 v42, v16, s54 // (op 1458) 


buffer_load_dwordx4 v[38:41], v42, s[72:75], 0 offen // (op 1458) Load value


v_mov_b64 v[42:43], s[44:45] // (op 13) convert

v_mov_b32 v68, v20 // (op 13) 
v_mov_b32 v69, v68 // (op 13) 
buffer_load_ubyte v44, v69, s[80:83], 0 offen offset:8 // (op 13) Load value
v_add_u32 v69, v69, v42 // (op 13) 
buffer_load_ubyte v45, v69, s[80:83], 0 offen offset:8 // (op 13) Load value


s_barrier  // (op 3747) 



s_waitcnt vmcnt(9)
ds_write_b8 v18, v66 offset:512 // (op 1119) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v18, v67 offset:516 // (op 1119) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3744) 



ds_read_u8 v42, v5 offset:512 // (op 1121) Load local data 



ds_read_u8 v43, v5 offset:640 // (op 1207) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:49], v[50:53], a[0:3], v9, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1122) 




ds_read_u8 v42, v5 offset:768 // (op 1323) Load local data 



ds_read_u8 v69, v5 offset:896 // (op 1439) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:49], v[54:57], a[4:7], v9, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1205) 

s_barrier  // (op 3932) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[46:49], v[58:61], a[8:11], v9, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1321) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[46:49], v[62:65], a[12:15], v9, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1437) 




buffer_load_dwordx4 v[46:49], v13, s[68:71], 0 offen offset:192 // (op 1124) Load value



v_add_u32 v12, 128, v12 // (op 2882) 
v_add_u32 v13, 192, v16 // (op 1126) 

buffer_load_dwordx4 v[50:53], v13, s[72:75], 0 offen // (op 1126) Load value




buffer_load_ubyte v9, v19, s[76:79], 0 offen offset:12 // (op 1128) Load value



v_add_u32 v17, 8, v17 // (op 3168) 
s_lshl_b64 s[52:53], s[36:37], 5 // (op 1221) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1221) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1221) 

s_add_u32 s54, s52, 192 // (op 1221) 

v_add_u32 v13, v16, s54 // (op 1221) 


buffer_load_dwordx4 v[54:57], v13, s[72:75], 0 offen // (op 1221) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 1337) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1337) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1337) 

s_lshl_b32 s54, s52, 1 // (op 1337) 
s_add_u32 s54, s54, 192 // (op 1337) 

v_add_u32 v13, v16, s54 // (op 1337) 


buffer_load_dwordx4 v[58:61], v13, s[72:75], 0 offen // (op 1337) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 1453) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1453) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1453) 

s_mul_i32 s54, 3, s52 // (op 1453) 
s_add_u32 s54, s54, 192 // (op 1453) 

v_add_u32 v13, v16, s54 // (op 1453) 


buffer_load_dwordx4 v[62:65], v13, s[72:75], 0 offen // (op 1453) Load value



v_add_u32 v14, 128, v14 // (op 3035) 
v_mov_b64 v[42:43], s[44:45] // (op 1118) convert

v_mov_b32 v13, v68 // (op 1118) 
buffer_load_ubyte v66, v13, s[80:83], 0 offen offset:12 // (op 1118) Load value
v_add_u32 v13, v13, v42 // (op 1118) 
buffer_load_ubyte v67, v13, s[80:83], 0 offen offset:12 // (op 1118) Load value



s_barrier  // (op 3712) 
v_add_u32 v20, 8, v20 // (op 3296) 



s_waitcnt vmcnt(9)
ds_write_b8 v21, v44 // (op 18) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v21, v45 offset:4 // (op 18) Store local data 
s_and_b32 s52, s7, 4294967294 // (op 1409) 
s_cmp_lt_u32 s1, s52 // (op 1409) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_1409 // (op 1409) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_1409 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_1409:













s_and_b32 s52, s7, 4294967294 // (op 2794) 
v_lshrrev_b32 v12, 7, v1 // (op 2794) 
s_add_u32 s53, s0, s52 // (op 2794) 

v_and_b32 v13, 15, v1 // (op 2794) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 2794) 


s_mul_i32 s52, s16, s8 // (op 2794) 
s_add_u32 s52, s52, s53 // (op 2794) 

v_and_b32 v12, 63, v1 // (op 2794) 
s_mul_hi_u32 s53, s52, s13 // (op 2794) 
v_lshrrev_b32 v13, 4, v12 // (op 2794) 

s_sub_u32 s54, s52, s53 // (op 2794) 
v_lshlrev_b32 v12, 5, v13 // (op 2794) 

s_lshr_b32 s55, s54, 1 // (op 2794) 

s_add_u32 s54, s55, s53 // (op 2794) 


s_lshr_b32 s53, s54, s11 // (op 2794) 

s_and_b32 s54, s11, 2147483648 // (op 2794) 
s_cmp_eq_i32 s54, 0 // (op 2794) 
s_mov_b32 s55, scc // (op 2794) 

s_cmp_eq_i32 s55, 0 // (op 2794) 
s_cselect_b32 s54, s52, s53 // (op 2794) 


s_mul_hi_u32 s53, s54, s12 // (op 2794) 
s_sub_u32 s55, s54, s53 // (op 2794) 
s_lshr_b32 s84, s55, 1 // (op 2794) 

s_add_u32 s55, s84, s53 // (op 2794) 


s_lshr_b32 s53, s55, s10 // (op 2794) 

s_and_b32 s55, s10, 2147483648 // (op 2794) 
s_cmp_eq_i32 s55, 0 // (op 2794) 
s_mov_b32 s84, scc // (op 2794) 

s_cmp_eq_i32 s84, 0 // (op 2794) 
s_cselect_b32 s55, s54, s53 // (op 2794) 


v_lshl_add_u32 v13, s55, 5, v14 // (op 2794) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 2794) 
s_mul_i32 s53, s54, s84 // (op 2794) 


s_sub_u32 s54, s52, s53 // (op 2794) 


v_lshl_add_u32 v14, s54, 7, v12 // (op 2794) 


v_mov_b32 v45, 0 // (op 2794) convert
v_mov_b32 v44, v14 // (op 2794) convert

v_mul_lo_u32 v15, v13, s31 // (op 2794) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 2794) most significant: high of low * low
v_mul_lo_u32 v14, v13, s30 // (op 2794) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 2794) most significant: sum

v_add_co_u32 v14, s[52:53], v14, v44 // (op 2794) least significant half
v_addc_co_u32 v15, s[52:53], v15, v45, s[52:53] // (op 2794) most significant half



v_lshlrev_b64 v[12:13], 2, v[14:15] // (op 2794) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 2794) 

v_mov_b32 v12, v14 // (op 2794) convert

s_mul_i32 s53, s30, s35 // (op 2796) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 2796) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 2796) least significant
s_mul_hi_u32 s55, s30, s34 // (op 2796) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2796) most significant: sum
s_add_u32 s53, s53, s55 // (op 2796) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 2796) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 2796) 

s_mov_b32 s68, s32 // (op 2796) reg expression
s_mov_b32 s69, s33 // (op 2796) reg expression
s_mov_b32 s70, s52 // (op 2796) reg expression
s_mov_b32 s71, 131072 // (op 2796) call()

s_and_b32 s52, s7, 4294967294 // (op 2941) 
v_lshrrev_b32 v13, 6, v1 // (op 2941) 
s_add_u32 s53, s0, s52 // (op 2941) 

v_and_b32 v14, 1, v13 // (op 2941) 

v_and_b32 v13, 15, v1 // (op 2941) 
s_mul_i32 s52, s16, s8 // (op 2941) 
s_add_u32 s52, s52, s53 // (op 2941) 

v_lshl_add_u32 v15, v14, 4, v13 // (op 2941) 


s_mul_hi_u32 s53, s52, s13 // (op 2941) 
v_and_b32 v13, 63, v1 // (op 2941) 
s_sub_u32 s54, s52, s53 // (op 2941) 
v_lshrrev_b32 v14, 4, v13 // (op 2941) 

s_lshr_b32 s55, s54, 1 // (op 2941) 

v_lshlrev_b32 v13, 5, v14 // (op 2941) 

s_add_u32 s54, s55, s53 // (op 2941) 


s_lshr_b32 s53, s54, s11 // (op 2941) 

s_and_b32 s54, s11, 2147483648 // (op 2941) 
s_cmp_eq_i32 s54, 0 // (op 2941) 
s_mov_b32 s55, scc // (op 2941) 

s_cmp_eq_i32 s55, 0 // (op 2941) 
s_cselect_b32 s54, s52, s53 // (op 2941) 


s_mul_hi_u32 s53, s54, s12 // (op 2941) 
s_sub_u32 s55, s54, s53 // (op 2941) 
s_lshr_b32 s84, s55, 1 // (op 2941) 

s_add_u32 s55, s84, s53 // (op 2941) 


s_lshr_b32 s53, s55, s10 // (op 2941) 

s_and_b32 s55, s10, 2147483648 // (op 2941) 
s_cmp_eq_i32 s55, 0 // (op 2941) 
s_mov_b32 s84, scc // (op 2941) 

s_cmp_eq_i32 s84, 0 // (op 2941) 
s_cselect_b32 s55, s54, s53 // (op 2941) 


s_mul_i32 s53, s55, s14 // (op 2941) 

s_sub_u32 s55, s54, s53 // (op 2941) 

v_lshl_add_u32 v14, s55, 7, v15 // (op 2941) 


v_mul_lo_u32 v45, v14, s37 // (op 2941) most significant: low of low * high
v_mul_hi_u32 v15, v14, s36 // (op 2941) most significant: high of low * low
v_mul_lo_u32 v44, v14, s36 // (op 2941) least significant: low of low * low
v_add_u32 v45, v45, v15 // (op 2941) most significant: sum


s_ashr_i64 s[84:85], s[2:3], 7 // (op 2941) 
s_mul_i32 s53, s54, s84 // (op 2941) 


s_sub_u32 s54, s52, s53 // (op 2941) 


v_lshl_add_u32 v14, s54, 7, v13 // (op 2941) 


v_mov_b32 v43, 0 // (op 2941) convert
v_mov_b32 v42, v14 // (op 2941) convert

v_add_co_u32 v40, s[52:53], v42, v44 // (op 2941) least significant half
v_addc_co_u32 v41, s[52:53], v43, v45, s[52:53] // (op 2941) most significant half

v_lshlrev_b64 v[40:41], 2, v[40:41] // (op 2941) 


v_lshrrev_b64 v[38:39], 3, v[40:41] // (op 2941) 

v_mov_b32 v13, v38 // (op 2941) convert

s_mul_i32 s53, s36, s23 // (op 2943) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 2943) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 2943) least significant
s_mul_hi_u32 s55, s36, s22 // (op 2943) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2943) most significant: sum
s_add_u32 s53, s53, s55 // (op 2943) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 2943) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 2943) 

s_mov_b32 s72, s38 // (op 2943) reg expression
s_mov_b32 s73, s39 // (op 2943) reg expression
s_mov_b32 s74, s52 // (op 2943) reg expression
s_mov_b32 s75, 131072 // (op 2943) call()

s_and_b32 s52, s7, 4294967294 // (op 3092) 
v_lshrrev_b32 v14, 7, v1 // (op 3092) 
s_add_u32 s53, s0, s52 // (op 3092) 

v_and_b32 v15, 15, v1 // (op 3092) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 3092) 


s_mul_i32 s52, s16, s8 // (op 3092) 
s_add_u32 s52, s52, s53 // (op 3092) 

v_and_b32 v14, 63, v1 // (op 3092) 
s_mul_hi_u32 s53, s52, s13 // (op 3092) 
v_lshrrev_b32 v15, 4, v14 // (op 3092) 

s_sub_u32 s54, s52, s53 // (op 3092) 
s_lshr_b32 s55, s54, 1 // (op 3092) 

s_add_u32 s54, s55, s53 // (op 3092) 


s_lshr_b32 s53, s54, s11 // (op 3092) 

s_and_b32 s54, s11, 2147483648 // (op 3092) 
s_cmp_eq_i32 s54, 0 // (op 3092) 
s_mov_b32 s55, scc // (op 3092) 

s_cmp_eq_i32 s55, 0 // (op 3092) 
s_cselect_b32 s54, s52, s53 // (op 3092) 


s_mul_hi_u32 s53, s54, s12 // (op 3092) 
s_sub_u32 s55, s54, s53 // (op 3092) 
s_lshr_b32 s84, s55, 1 // (op 3092) 

s_add_u32 s55, s84, s53 // (op 3092) 


s_lshr_b32 s53, s55, s10 // (op 3092) 

s_and_b32 s55, s10, 2147483648 // (op 3092) 
s_cmp_eq_i32 s55, 0 // (op 3092) 
s_mov_b32 s84, scc // (op 3092) 

s_cmp_eq_i32 s84, 0 // (op 3092) 
s_cselect_b32 s55, s54, s53 // (op 3092) 


v_lshl_add_u32 v14, s55, 5, v16 // (op 3092) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 3092) 
s_mul_i32 s53, s54, s84 // (op 3092) 


s_sub_u32 s54, s52, s53 // (op 3092) 


v_lshl_add_u32 v16, s54, 2, v15 // (op 3092) 


v_mul_lo_u32 v15, v14, s40 // (op 3092) 
v_add_u32 v15, v15, v16 // (op 3092) 


s_mul_i32 s53, s40, s35 // (op 3094) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 3094) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 3094) least significant
s_mul_hi_u32 s55, s40, s34 // (op 3094) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3094) most significant: sum
s_add_u32 s53, s53, s55 // (op 3094) most significant: sum


s_mov_b32 s76, s42 // (op 3094) reg expression
s_mov_b32 s77, s43 // (op 3094) reg expression
s_mov_b32 s78, s52 // (op 3094) reg expression
s_mov_b32 s79, 131072 // (op 3094) call()

s_and_b32 s52, s7, 4294967294 // (op 3220) 
v_lshrrev_b32 v14, 2, v1 // (op 3220) 
s_add_u32 s53, s0, s52 // (op 3220) 

v_lshlrev_b32 v16, 1, v14 // (op 3220) 

v_and_b32 v14, -4, v1 // (op 3220) 
s_mul_i32 s52, s16, s8 // (op 3220) 
s_add_u32 s52, s52, s53 // (op 3220) 

v_sub_u32 v17, v1, v14 // (op 3220) 

s_mul_hi_u32 s53, s52, s13 // (op 3220) 
s_sub_u32 s54, s52, s53 // (op 3220) 
s_lshr_b32 s55, s54, 1 // (op 3220) 

s_add_u32 s54, s55, s53 // (op 3220) 


s_lshr_b32 s53, s54, s11 // (op 3220) 

s_and_b32 s54, s11, 2147483648 // (op 3220) 
s_cmp_eq_i32 s54, 0 // (op 3220) 
s_mov_b32 s55, scc // (op 3220) 

s_cmp_eq_i32 s55, 0 // (op 3220) 
s_cselect_b32 s54, s52, s53 // (op 3220) 


s_mul_hi_u32 s53, s54, s12 // (op 3220) 
s_sub_u32 s55, s54, s53 // (op 3220) 
s_lshr_b32 s84, s55, 1 // (op 3220) 

s_add_u32 s55, s84, s53 // (op 3220) 


s_lshr_b32 s53, s55, s10 // (op 3220) 

s_and_b32 s55, s10, 2147483648 // (op 3220) 
s_cmp_eq_i32 s55, 0 // (op 3220) 
s_mov_b32 s84, scc // (op 3220) 

s_cmp_eq_i32 s84, 0 // (op 3220) 
s_cselect_b32 s55, s54, s53 // (op 3220) 


s_mul_i32 s53, s55, s14 // (op 3220) 

s_sub_u32 s55, s54, s53 // (op 3220) 

v_lshl_add_u32 v14, s55, 7, v16 // (op 3220) 


v_mul_lo_u32 v16, v14, s44 // (op 3220) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 3220) 
s_mul_i32 s53, s54, s84 // (op 3220) 


s_sub_u32 s54, s52, s53 // (op 3220) 


v_lshl_add_u32 v14, s54, 2, v17 // (op 3220) 


v_add_u32 v17, v14, v16 // (op 3220) 


s_mul_i32 s53, s44, s23 // (op 3222) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 3222) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 3222) least significant
s_mul_hi_u32 s55, s44, s22 // (op 3222) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3222) most significant: sum
s_add_u32 s53, s53, s55 // (op 3222) most significant: sum


s_mov_b32 s80, s46 // (op 3222) reg expression
s_mov_b32 s81, s47 // (op 3222) reg expression
s_mov_b32 s82, s52 // (op 3222) reg expression
s_mov_b32 s83, 131072 // (op 3222) call()

v_and_b32 v14, -4, v1 // (op 3381) 
v_sub_u32 v16, v1, v14 // (op 3381) 

v_lshrrev_b32 v14, 2, v1 // (op 3381) 
v_lshl_add_u32 v18, v14, 3, v16 // (op 3381) 


s_and_b32 s52, s7, 4294967294 // (op 1415) 
s_cmp_lt_u32 s52, s7 // (op 1414) 







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_1414 // (op 1414) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_1414 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_1414:

s_add_u32 s52, s52, 1 // (op 1416) 
v_mov_b64 v[20:21], s[44:45] // (op 1076) convert

v_mov_b32 v9, v17 // (op 1076) 
v_mov_b32 v14, v9 // (op 1076) 
buffer_load_ubyte v22, v14, s[80:83], 0 offen offset: 0 // (op 1076) Load value
v_add_u32 v14, v14, v20 // (op 1076) 
buffer_load_ubyte v23, v14, s[80:83], 0 offen offset: 0 // (op 1076) Load value



v_add_u32 v17, 4, v17 // (op 3212) 


s_barrier  // (op 3715) 
v_mov_b32 v9, v12 // (op 1082) 




buffer_load_dwordx4 v[24:27], v9, s[68:71], 0 offen // (op 1082) Load value

v_add_u32 v12, 64, v12 // (op 2784) 
s_waitcnt vmcnt(2)
ds_write_b8 v18, v22 // (op 1077) Store local data 


v_mov_b32 v9, v13 // (op 1084) 
v_mov_b32 v14, v9 // (op 1084) reg expression

buffer_load_dwordx4 v[28:31], v14, s[72:75], 0 offen // (op 1084) Load value

s_waitcnt vmcnt(2)
ds_write_b8 v18, v23 offset:4 // (op 1077) Store local data 



v_mov_b32 v14, v15 // (op 1086) 

buffer_load_ubyte v16, v14, s[76:79], 0 offen offset: 0 // (op 1086) Load value

v_add_u32 v15, 4, v15 // (op 3084) 


s_lshl_b64 s[54:55], s[36:37], 5 // (op 1196) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 1196) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 1196) 

v_add_u32 v14, v9, s54 // (op 1196) 


buffer_load_dwordx4 v[20:23], v14, s[72:75], 0 offen // (op 1196) Load value

s_waitcnt lgkmcnt(0)
s_barrier  // (op 3718) 





ds_read_u8 v14, v0 // (op 1079) Load local data 
s_lshl_b64 s[54:55], s[36:37], 5 // (op 1312) 



ds_read_u8 v19, v0 offset:128 // (op 1188) Load local data 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 1312) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 1312) 

s_lshl_b32 s53, s54, 1 // (op 1312) 

v_add_u32 v32, v9, s53 // (op 1312) 


s_waitcnt vmcnt(1) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:27], v[28:31], a[0:3], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1080) 
buffer_load_dwordx4 v[34:37], v32, s[72:75], 0 offen // (op 1312) Load value








ds_read_u8 v14, v0 offset:256 // (op 1304) Load local data 
s_lshl_b64 s[54:55], s[36:37], 5 // (op 1428) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 1428) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 1428) 

s_mul_i32 s53, 3, s54 // (op 1428) 

v_add_u32 v28, v9, s53 // (op 1428) 


buffer_load_dwordx4 v[30:33], v28, s[72:75], 0 offen // (op 1428) Load value

s_waitcnt vmcnt(2) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:27], v[20:23], a[4:7], v16, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1189) 






v_add_u32 v13, 64, v13 // (op 2931) 
ds_read_u8 v9, v0 offset:384 // (op 1420) Load local data 
s_waitcnt vmcnt(1) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[24:27], v[34:37], a[8:11], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1305) 


s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:27], v[30:33], a[12:15], v16, v9 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1421) 




s_cmp_lt_u32 s52, s7 // (op 1414) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_1414 // (op 1414) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_1414 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_1414:






s_barrier  // (op 3918) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 243) 
s_cmp_lg_u32 s18, 0 // (op 269) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Send_Tile_269 // (op 269) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Send_Tile_269



buffer_store_dword a0, v8, s[64:67], 0 offen sc1 // (op 266) Store value
buffer_store_dword a1, v8, s[64:67], 0 offen offset:512 sc1 // (op 266) Store value
buffer_store_dword a2, v8, s[64:67], 0 offen offset:1024 sc1 // (op 266) Store value
buffer_store_dword a3, v8, s[64:67], 0 offen offset:1536 sc1 // (op 266) Store value



buffer_store_dword a4, v8, s[64:67], 0 offen offset:128 sc1 // (op 1609) Store value
buffer_store_dword a5, v8, s[64:67], 0 offen offset:640 sc1 // (op 1609) Store value
buffer_store_dword a6, v8, s[64:67], 0 offen offset:1152 sc1 // (op 1609) Store value
buffer_store_dword a7, v8, s[64:67], 0 offen offset:1664 sc1 // (op 1609) Store value



buffer_store_dword a8, v8, s[64:67], 0 offen offset:256 sc1 // (op 1610) Store value
buffer_store_dword a9, v8, s[64:67], 0 offen offset:768 sc1 // (op 1610) Store value
buffer_store_dword a10, v8, s[64:67], 0 offen offset:1280 sc1 // (op 1610) Store value
buffer_store_dword a11, v8, s[64:67], 0 offen offset:1792 sc1 // (op 1610) Store value



buffer_store_dword a12, v8, s[64:67], 0 offen offset:384 sc1 // (op 1611) Store value
buffer_store_dword a13, v8, s[64:67], 0 offen offset:896 sc1 // (op 1611) Store value
buffer_store_dword a14, v8, s[64:67], 0 offen offset:1408 sc1 // (op 1611) Store value
buffer_store_dword a15, v8, s[64:67], 0 offen offset:1920 sc1 // (op 1611) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 270) 
v_cmp_eq_i32 vcc, v1, 0 // (op 274) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_274 // (op 274) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_274
s_mov_b32 s52, 1 // (op 272) call()
s_mov_b32 s55, 0 // (op 273) convert
s_mov_b32 s54, s16 // (op 273) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 273) 
s_add_u32 s84, s48, s54 // (op 273) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 273) End of Int64 add, unlocking SCC
s_store_dword s52, s[84:85], 0 glc // (op 273) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_274 // (op 274) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_274
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_274:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_274:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Send_Tile_269 // (op 269) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Send_Tile_269
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Send_Tile_269:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Send_Tile_269:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 285) 
s_mov_b32 s52, scc // (op 285) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 285) 
s_mov_b32 s53, s54 // (op 285) convert

s_sub_u32 s54, s53, 1 // (op 285) 

s_cmp_lt_u32 s19, s54 // (op 285) 
s_mov_b32 s53, scc // (op 285) 

s_and_b32 s54, s52, s53 // (op 285) 
s_cmp_lg_u32 s54, 0 // (op 285) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Receive_Tile_285 // (op 285) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Receive_Tile_285
s_mov_b32 s52, 0 // (op 251) call()
s_mul_i32 s53, s52, s8 // (op 250) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 250) 
s_mov_b32 s84, s54 // (op 250) convert

s_sub_u32 s54, s84, s19 // (op 250) 

s_sub_u32 s55, s54, 1 // (op 250) 

s_cmp_lt_u32 s53, s55 // (op 250) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_250 // (op 250) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_250 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_ReceiveTileLoop_250:
s_add_u32 s53, s16, 1 // (op 3553) 
v_lshrrev_b32 v8, 5, v1 // (op 3553) 
s_add_u32 s54, s53, s52 // (op 3553) 
s_lshl_b32 s54, s54, 5 // (op 3553) 

v_lshlrev_b32 v9, 2, v8 // (op 3553) 

v_add_lshl_u32 v8, s54, v9, 7 // (op 3553) 


v_and_b32 v9, -32, v1 // (op 3553) 
v_sub_u32 v12, v1, v9 // (op 3553) 

v_add_lshl_u32 v9, v8, v12, 2 // (op 3553) 


s_lshl_b32 s53, s15, 14 // (op 3555) 
s_mov_b32 s64, s28 // (op 3555) reg expression
s_mov_b32 s65, s29 // (op 3555) reg expression
s_mov_b32 s66, s53 // (op 3555) reg expression
s_mov_b32 s67, 131072 // (op 3555) call()

RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_289:
s_add_u32 s53, s16, 1 // (op 287) 
s_add_u32 s54, s53, s52 // (op 287) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 287) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 287) 

s_add_u32 s84, s48, s54 // (op 287) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 287) End of Int64 add, unlocking SCC
s_load_dword s53, s[84:85], 0 glc // (op 287) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 289) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_289 // (op 289) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_289 if true)
s_barrier  // (op 293) 
v_cmp_eq_i32 vcc, v1, 0 // (op 292) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_292 // (op 292) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_292
s_mov_b32 s53, 0 // (op 290) call()
s_add_u32 s54, s16, 1 // (op 291) 
s_add_u32 s84, s54, s52 // (op 291) Start of Int64 add, locking SCC
s_addc_u32 s85, 0, 0 // (op 291) End of Int64 add, unlocking SCC
s_lshl_b64 s[84:85], s[84:85], 2 // (op 291) 

s_add_u32 s54, s48, s84 // (op 291) Start of Int64 add, locking SCC
s_addc_u32 s55, s49, s85 // (op 291) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 291) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_292 // (op 292) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_292
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_292:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_292:




buffer_load_dword v12, v9, s[64:67], 0 offen // (op 304) Load value
buffer_load_dword v13, v9, s[64:67], 0 offen offset:512 // (op 304) Load value
buffer_load_dword v14, v9, s[64:67], 0 offen offset:1024 // (op 304) Load value
buffer_load_dword v15, v9, s[64:67], 0 offen offset:1536 // (op 304) Load value
v_accvgpr_read v16, a0 // (op 307) reg expression
v_accvgpr_read v17, a1 // (op 307) reg expression
v_accvgpr_read v18, a2 // (op 307) reg expression
v_accvgpr_read v19, a3 // (op 307) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v16, v16, v12 // (op 309) 
s_waitcnt vmcnt(2)
v_add_f32 v17, v17, v13 // (op 309) 
s_waitcnt vmcnt(1)
v_add_f32 v18, v18, v14 // (op 309) 
s_waitcnt vmcnt(0)
v_add_f32 v19, v19, v15 // (op 309) 

v_accvgpr_write a0, v16 // (op 311) reg expression
v_accvgpr_write a1, v17 // (op 311) reg expression
v_accvgpr_write a2, v18 // (op 311) reg expression
v_accvgpr_write a3, v19 // (op 311) reg expression




buffer_load_dword v12, v9, s[64:67], 0 offen offset:128 // (op 1631) Load value
buffer_load_dword v13, v9, s[64:67], 0 offen offset:640 // (op 1631) Load value
buffer_load_dword v14, v9, s[64:67], 0 offen offset:1152 // (op 1631) Load value
buffer_load_dword v15, v9, s[64:67], 0 offen offset:1664 // (op 1631) Load value
v_accvgpr_read v16, a4 // (op 1632) reg expression
v_accvgpr_read v17, a5 // (op 1632) reg expression
v_accvgpr_read v18, a6 // (op 1632) reg expression
v_accvgpr_read v19, a7 // (op 1632) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v16, v16, v12 // (op 1633) 
s_waitcnt vmcnt(2)
v_add_f32 v17, v17, v13 // (op 1633) 
s_waitcnt vmcnt(1)
v_add_f32 v18, v18, v14 // (op 1633) 
s_waitcnt vmcnt(0)
v_add_f32 v19, v19, v15 // (op 1633) 

v_accvgpr_write a4, v16 // (op 1634) reg expression
v_accvgpr_write a5, v17 // (op 1634) reg expression
v_accvgpr_write a6, v18 // (op 1634) reg expression
v_accvgpr_write a7, v19 // (op 1634) reg expression




buffer_load_dword v12, v9, s[64:67], 0 offen offset:256 // (op 1638) Load value
buffer_load_dword v13, v9, s[64:67], 0 offen offset:768 // (op 1638) Load value
buffer_load_dword v14, v9, s[64:67], 0 offen offset:1280 // (op 1638) Load value
buffer_load_dword v15, v9, s[64:67], 0 offen offset:1792 // (op 1638) Load value
v_accvgpr_read v16, a8 // (op 1639) reg expression
v_accvgpr_read v17, a9 // (op 1639) reg expression
v_accvgpr_read v18, a10 // (op 1639) reg expression
v_accvgpr_read v19, a11 // (op 1639) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v16, v16, v12 // (op 1640) 
s_waitcnt vmcnt(2)
v_add_f32 v17, v17, v13 // (op 1640) 
s_waitcnt vmcnt(1)
v_add_f32 v18, v18, v14 // (op 1640) 
s_waitcnt vmcnt(0)
v_add_f32 v19, v19, v15 // (op 1640) 

v_accvgpr_write a8, v16 // (op 1641) reg expression
v_accvgpr_write a9, v17 // (op 1641) reg expression
v_accvgpr_write a10, v18 // (op 1641) reg expression
v_accvgpr_write a11, v19 // (op 1641) reg expression




buffer_load_dword v12, v9, s[64:67], 0 offen offset:384 // (op 1645) Load value
buffer_load_dword v13, v9, s[64:67], 0 offen offset:896 // (op 1645) Load value
buffer_load_dword v14, v9, s[64:67], 0 offen offset:1408 // (op 1645) Load value
buffer_load_dword v15, v9, s[64:67], 0 offen offset:1920 // (op 1645) Load value
v_accvgpr_read v16, a12 // (op 1646) reg expression
v_accvgpr_read v17, a13 // (op 1646) reg expression
v_accvgpr_read v18, a14 // (op 1646) reg expression
v_accvgpr_read v19, a15 // (op 1646) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v16, v16, v12 // (op 1647) 
s_waitcnt vmcnt(2)
v_add_f32 v17, v17, v13 // (op 1647) 
s_waitcnt vmcnt(1)
v_add_f32 v18, v18, v14 // (op 1647) 
s_waitcnt vmcnt(0)
v_add_f32 v19, v19, v15 // (op 1647) 

v_accvgpr_write a12, v16 // (op 1648) reg expression
v_accvgpr_write a13, v17 // (op 1648) reg expression
v_accvgpr_write a14, v18 // (op 1648) reg expression
v_accvgpr_write a15, v19 // (op 1648) reg expression

s_add_u32 s52, s52, 1 // (op 252) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 250) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 250) 
s_mov_b32 s84, s54 // (op 250) convert

s_sub_u32 s54, s84, s19 // (op 250) 

s_sub_u32 s55, s54, 1 // (op 250) 

s_cmp_lt_u32 s53, s55 // (op 250) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_ReceiveTileLoop_250 // (op 250) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_ReceiveTileLoop_250 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_250:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Receive_Tile_285 // (op 285) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Receive_Tile_285
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Receive_Tile_285:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Receive_Tile_285:

s_cmp_eq_i32 s18, 0 // (op 328) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_328 // (op 328) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_328
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 106) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_106 // (op 106) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_106
v_accvgpr_read v12, a0 // (op 38) 
v_accvgpr_read v13, a1 // (op 38) 
v_accvgpr_read v14, a2 // (op 38) 
v_accvgpr_read v15, a3 // (op 38) 
v_mul_f32 v16, v7, v12 // (op 38) 
v_mul_f32 v17, v7, v13 // (op 38) 
v_mul_f32 v18, v7, v14 // (op 38) 
v_mul_f32 v19, v7, v15 // (op 38) 



v_mov_b32 v8, v11 // (op 44) reg expression

buffer_store_dwordx4 v[16:19], v8, s[60:63], 0 offen // (op 44) Store value


v_accvgpr_read v12, a4 // (op 1676) 
v_accvgpr_read v13, a5 // (op 1676) 
v_accvgpr_read v14, a6 // (op 1676) 
v_accvgpr_read v15, a7 // (op 1676) 
v_mul_f32 v16, v7, v12 // (op 1676) 
v_mul_f32 v17, v7, v13 // (op 1676) 
v_mul_f32 v18, v7, v14 // (op 1676) 
v_mul_f32 v19, v7, v15 // (op 1676) 



s_lshl_b32 s52, s24, 7 // (op 1678) 
v_add_u32 v8, v11, s52 // (op 1678) 


buffer_store_dwordx4 v[16:19], v8, s[60:63], 0 offen // (op 1678) Store value


v_accvgpr_read v12, a8 // (op 1688) 
v_accvgpr_read v13, a9 // (op 1688) 
v_accvgpr_read v14, a10 // (op 1688) 
v_accvgpr_read v15, a11 // (op 1688) 
v_mul_f32 v16, v7, v12 // (op 1688) 
v_mul_f32 v17, v7, v13 // (op 1688) 
v_mul_f32 v18, v7, v14 // (op 1688) 
v_mul_f32 v19, v7, v15 // (op 1688) 



s_lshl_b32 s52, s24, 8 // (op 1690) 
v_add_u32 v8, v11, s52 // (op 1690) 


buffer_store_dwordx4 v[16:19], v8, s[60:63], 0 offen // (op 1690) Store value


v_accvgpr_read v12, a12 // (op 1700) 
v_accvgpr_read v13, a13 // (op 1700) 
v_accvgpr_read v14, a14 // (op 1700) 
v_accvgpr_read v15, a15 // (op 1700) 
v_mul_f32 v16, v7, v12 // (op 1700) 
v_mul_f32 v17, v7, v13 // (op 1700) 
v_mul_f32 v18, v7, v14 // (op 1700) 
v_mul_f32 v19, v7, v15 // (op 1700) 



s_lshl_b32 s52, s24, 7 // (op 1702) 
s_mul_i32 s53, 3, s52 // (op 1702) 

v_add_u32 v8, v11, s53 // (op 1702) 


buffer_store_dwordx4 v[16:19], v8, s[60:63], 0 offen // (op 1702) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_106 // (op 106) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_106
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_106:


v_mov_b32 v8, v10 // (op 118) reg expression

buffer_load_dwordx4 v[12:15], v8, s[56:59], 0 offen // (op 118) Load value

v_accvgpr_read v16, a0 // (op 115) 
v_accvgpr_read v17, a1 // (op 115) 
v_accvgpr_read v18, a2 // (op 115) 
v_accvgpr_read v19, a3 // (op 115) 
v_mul_f32 v20, v7, v16 // (op 115) 
v_mul_f32 v21, v7, v17 // (op 115) 
v_mul_f32 v22, v7, v18 // (op 115) 
v_mul_f32 v23, v7, v19 // (op 115) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v6, v12 // (op 119) 
v_mul_f32 v17, v6, v13 // (op 119) 
v_mul_f32 v18, v6, v14 // (op 119) 
v_mul_f32 v19, v6, v15 // (op 119) 

v_add_f32 v12, v16, v20 // (op 116) 
v_add_f32 v13, v17, v21 // (op 116) 
v_add_f32 v14, v18, v22 // (op 116) 
v_add_f32 v15, v19, v23 // (op 116) 




v_mov_b32 v8, v11 // (op 117) reg expression

buffer_store_dwordx4 v[12:15], v8, s[60:63], 0 offen // (op 117) Store value




s_lshl_b32 s52, s4, 7 // (op 1734) 
v_add_u32 v8, v10, s52 // (op 1734) 


buffer_load_dwordx4 v[12:15], v8, s[56:59], 0 offen // (op 1734) Load value

v_accvgpr_read v16, a4 // (op 1731) 
v_accvgpr_read v17, a5 // (op 1731) 
v_accvgpr_read v18, a6 // (op 1731) 
v_accvgpr_read v19, a7 // (op 1731) 
v_mul_f32 v20, v7, v16 // (op 1731) 
v_mul_f32 v21, v7, v17 // (op 1731) 
v_mul_f32 v22, v7, v18 // (op 1731) 
v_mul_f32 v23, v7, v19 // (op 1731) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v6, v12 // (op 1735) 
v_mul_f32 v17, v6, v13 // (op 1735) 
v_mul_f32 v18, v6, v14 // (op 1735) 
v_mul_f32 v19, v6, v15 // (op 1735) 

v_add_f32 v12, v16, v20 // (op 1732) 
v_add_f32 v13, v17, v21 // (op 1732) 
v_add_f32 v14, v18, v22 // (op 1732) 
v_add_f32 v15, v19, v23 // (op 1732) 




s_lshl_b32 s52, s24, 7 // (op 1733) 
v_add_u32 v8, v11, s52 // (op 1733) 


buffer_store_dwordx4 v[12:15], v8, s[60:63], 0 offen // (op 1733) Store value




s_lshl_b32 s52, s4, 8 // (op 1746) 
v_add_u32 v8, v10, s52 // (op 1746) 


buffer_load_dwordx4 v[12:15], v8, s[56:59], 0 offen // (op 1746) Load value

v_accvgpr_read v16, a8 // (op 1743) 
v_accvgpr_read v17, a9 // (op 1743) 
v_accvgpr_read v18, a10 // (op 1743) 
v_accvgpr_read v19, a11 // (op 1743) 
v_mul_f32 v20, v7, v16 // (op 1743) 
v_mul_f32 v21, v7, v17 // (op 1743) 
v_mul_f32 v22, v7, v18 // (op 1743) 
v_mul_f32 v23, v7, v19 // (op 1743) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v6, v12 // (op 1747) 
v_mul_f32 v17, v6, v13 // (op 1747) 
v_mul_f32 v18, v6, v14 // (op 1747) 
v_mul_f32 v19, v6, v15 // (op 1747) 

v_add_f32 v12, v16, v20 // (op 1744) 
v_add_f32 v13, v17, v21 // (op 1744) 
v_add_f32 v14, v18, v22 // (op 1744) 
v_add_f32 v15, v19, v23 // (op 1744) 




s_lshl_b32 s52, s24, 8 // (op 1745) 
v_add_u32 v8, v11, s52 // (op 1745) 


buffer_store_dwordx4 v[12:15], v8, s[60:63], 0 offen // (op 1745) Store value




s_lshl_b32 s52, s4, 7 // (op 1758) 
s_mul_i32 s53, 3, s52 // (op 1758) 

v_add_u32 v8, v10, s53 // (op 1758) 


buffer_load_dwordx4 v[12:15], v8, s[56:59], 0 offen // (op 1758) Load value

v_accvgpr_read v16, a12 // (op 1755) 
v_accvgpr_read v17, a13 // (op 1755) 
v_accvgpr_read v18, a14 // (op 1755) 
v_accvgpr_read v19, a15 // (op 1755) 
v_mul_f32 v20, v7, v16 // (op 1755) 
v_mul_f32 v21, v7, v17 // (op 1755) 
v_mul_f32 v22, v7, v18 // (op 1755) 
v_mul_f32 v23, v7, v19 // (op 1755) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v6, v12 // (op 1759) 
v_mul_f32 v17, v6, v13 // (op 1759) 
v_mul_f32 v18, v6, v14 // (op 1759) 
v_mul_f32 v19, v6, v15 // (op 1759) 

v_add_f32 v12, v16, v20 // (op 1756) 
v_add_f32 v13, v17, v21 // (op 1756) 
v_add_f32 v14, v18, v22 // (op 1756) 
v_add_f32 v15, v19, v23 // (op 1756) 




s_lshl_b32 s52, s24, 7 // (op 1757) 
s_mul_i32 s53, 3, s52 // (op 1757) 

v_add_u32 v8, v11, s53 // (op 1757) 


buffer_store_dwordx4 v[12:15], v8, s[60:63], 0 offen // (op 1757) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_106:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_328 // (op 328) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_328
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_328:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_328:


s_mul_i32 s52, s8, s16 // (op 241) 
s_cmp_gt_u32 s9, s52 // (op 241) 
s_mov_b32 s53, scc // (op 241) 
s_sub_u32 s54, s9, s52 // (op 241) 

s_cmp_eq_i32 s53, 0 // (op 241) 
s_cselect_b32 s52, 0, s54 // (op 241) 


s_cmp_lt_u32 s8, s52 // (op 241) 
s_mov_b32 s53, scc // (op 241) 
s_cmp_eq_i32 s53, 0 // (op 241) 
s_cselect_b32 s54, s52, s8 // (op 241) 


s_cmp_lt_u32 s0, s54 // (op 241) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_SKStreamTileLoop_241 // (op 241) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_SKStreamTileLoop_241 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_241:











s_mov_b32 s0, 0 // (op 382) call()
s_mov_b32 s1, 0 // (op 383) call()
s_mov_b32 s0, 0 // (op 397) call()
s_mul_i32 s15, s50, s16 // (op 396) 
s_cmp_gt_u32 s51, s15 // (op 396) 
s_mov_b32 s6, scc // (op 396) 
s_sub_u32 s7, s51, s15 // (op 396) 

s_cmp_eq_i32 s6, 0 // (op 396) 
s_cselect_b32 s8, 0, s7 // (op 396) 


s_cmp_lt_u32 s50, s8 // (op 396) 
s_mov_b32 s15, scc // (op 396) 
s_cmp_eq_i32 s15, 0 // (op 396) 
s_cselect_b32 s6, s8, s50 // (op 396) 


s_cmp_lt_u32 s0, s6 // (op 396) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_396 // (op 396) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_396 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_DPStreamTileLoop_396:
v_lshrrev_b32 v0, 7, v1 // (op 3351) 
s_mul_i32 s15, s16, s50 // (op 3351) 
s_add_u32 s15, s15, s0 // (op 3351) 
v_and_b32 v5, 63, v1 // (op 3351) 
s_add_u32 s6, s15, s9 // (op 3351) 

v_lshrrev_b32 v8, 4, v5 // (op 3351) 
s_mul_hi_u32 s15, s6, s13 // (op 3351) 
v_lshlrev_b32 v9, 2, v8 // (op 3351) 

s_sub_u32 s7, s6, s15 // (op 3351) 
v_lshl_add_u32 v8, v0, 4, v9 // (op 3351) 


s_lshr_b32 s8, s7, 1 // (op 3351) 

v_lshrrev_b32 v0, 2, v5 // (op 3351) 

s_add_u32 s7, s8, s15 // (op 3351) 


v_and_b32 v5, 3, v0 // (op 3351) 

s_lshr_b32 s8, s7, s11 // (op 3351) 

v_and_b32 v0, 3, v1 // (op 3351) 
s_and_b32 s7, s11, 2147483648 // (op 3351) 
v_lshl_add_u32 v9, v5, 2, v0 // (op 3351) 


s_cmp_eq_i32 s7, 0 // (op 3351) 
s_mov_b32 s15, scc // (op 3351) 

v_lshrrev_b32 v0, 6, v1 // (op 3351) 
s_cmp_eq_i32 s15, 0 // (op 3351) 
s_cselect_b32 s7, s6, s8 // (op 3351) 



v_and_b32 v5, 1, v0 // (op 3351) 

s_mul_hi_u32 s6, s7, s12 // (op 3351) 
v_lshl_add_u32 v0, v5, 4, v9 // (op 3351) 


s_sub_u32 s8, s7, s6 // (op 3351) 
s_lshr_b32 s15, s8, 1 // (op 3351) 

s_add_u32 s8, s15, s6 // (op 3351) 


s_lshr_b32 s6, s8, s10 // (op 3351) 

s_and_b32 s8, s10, 2147483648 // (op 3351) 
s_cmp_eq_i32 s8, 0 // (op 3351) 
s_mov_b32 s15, scc // (op 3351) 

s_cmp_eq_i32 s15, 0 // (op 3351) 
s_cselect_b32 s8, s7, s6 // (op 3351) 


s_mul_i32 s6, s8, s14 // (op 3351) 
v_lshl_add_u32 v5, s8, 5, v8 // (op 3351) 


s_sub_u32 s8, s7, s6 // (op 3351) 


v_lshl_add_u32 v8, s8, 7, v0 // (op 3351) 


v_mul_lo_u32 v0, v8, s4 // (op 3351) 

v_add_lshl_u32 v8, v5, v0, 2 // (op 3351) 


s_mul_i32 s19, s4, s23 // (op 3353) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 3353) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 3353) least significant
s_mul_hi_u32 s6, s4, s22 // (op 3353) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3353) most significant: sum
s_add_u32 s19, s19, s6 // (op 3353) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 3353) 

s_mov_b32 s56, s20 // (op 3353) reg expression
s_mov_b32 s57, s21 // (op 3353) reg expression
s_mov_b32 s58, s28 // (op 3353) reg expression
s_mov_b32 s59, 131072 // (op 3353) call()

v_lshrrev_b32 v0, 7, v1 // (op 3451) 
s_mul_i32 s15, s16, s50 // (op 3451) 
s_add_u32 s15, s15, s0 // (op 3451) 
v_and_b32 v5, 63, v1 // (op 3451) 
s_add_u32 s6, s15, s9 // (op 3451) 

v_lshrrev_b32 v9, 4, v5 // (op 3451) 
s_mul_hi_u32 s15, s6, s13 // (op 3451) 
v_lshlrev_b32 v10, 2, v9 // (op 3451) 

s_sub_u32 s7, s6, s15 // (op 3451) 
v_lshl_add_u32 v9, v0, 4, v10 // (op 3451) 


s_lshr_b32 s8, s7, 1 // (op 3451) 

v_lshrrev_b32 v0, 2, v5 // (op 3451) 

s_add_u32 s7, s8, s15 // (op 3451) 


v_and_b32 v5, 3, v0 // (op 3451) 

s_lshr_b32 s8, s7, s11 // (op 3451) 

v_and_b32 v0, 3, v1 // (op 3451) 
s_and_b32 s7, s11, 2147483648 // (op 3451) 
v_lshl_add_u32 v10, v5, 2, v0 // (op 3451) 


s_cmp_eq_i32 s7, 0 // (op 3451) 
s_mov_b32 s15, scc // (op 3451) 

v_lshrrev_b32 v0, 6, v1 // (op 3451) 
s_cmp_eq_i32 s15, 0 // (op 3451) 
s_cselect_b32 s7, s6, s8 // (op 3451) 



v_and_b32 v5, 1, v0 // (op 3451) 

s_mul_hi_u32 s6, s7, s12 // (op 3451) 
v_lshl_add_u32 v0, v5, 4, v10 // (op 3451) 


s_sub_u32 s8, s7, s6 // (op 3451) 
s_lshr_b32 s15, s8, 1 // (op 3451) 

s_add_u32 s8, s15, s6 // (op 3451) 


s_lshr_b32 s6, s8, s10 // (op 3451) 

s_and_b32 s8, s10, 2147483648 // (op 3451) 
s_cmp_eq_i32 s8, 0 // (op 3451) 
s_mov_b32 s15, scc // (op 3451) 

s_cmp_eq_i32 s15, 0 // (op 3451) 
s_cselect_b32 s8, s7, s6 // (op 3451) 


s_mul_i32 s6, s8, s14 // (op 3451) 
v_lshl_add_u32 v5, s8, 5, v9 // (op 3451) 


s_sub_u32 s8, s7, s6 // (op 3451) 


v_lshl_add_u32 v9, s8, 7, v0 // (op 3451) 


v_mul_lo_u32 v0, v9, s24 // (op 3451) 

v_add_lshl_u32 v9, v5, v0, 2 // (op 3451) 


s_mul_i32 s19, s24, s23 // (op 3453) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 3453) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 3453) least significant
s_mul_hi_u32 s6, s24, s22 // (op 3453) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3453) most significant: sum
s_add_u32 s19, s19, s6 // (op 3453) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 3453) 

s_mov_b32 s60, s26 // (op 3453) reg expression
s_mov_b32 s61, s27 // (op 3453) reg expression
s_mov_b32 s62, s28 // (op 3453) reg expression
s_mov_b32 s63, 131072 // (op 3453) call()

v_accvgpr_write a0, 0.00000 // (op 153) call()
v_accvgpr_write a1, 0.00000 // (op 153) call()
v_accvgpr_write a2, 0.00000 // (op 153) call()
v_accvgpr_write a3, 0.00000 // (op 153) call()
v_accvgpr_write a4, 0.00000 // (op 509) call()
v_accvgpr_write a5, 0.00000 // (op 509) call()
v_accvgpr_write a6, 0.00000 // (op 509) call()
v_accvgpr_write a7, 0.00000 // (op 509) call()
v_accvgpr_write a8, 0.00000 // (op 625) call()
v_accvgpr_write a9, 0.00000 // (op 625) call()
v_accvgpr_write a10, 0.00000 // (op 625) call()
v_accvgpr_write a11, 0.00000 // (op 625) call()
v_accvgpr_write a12, 0.00000 // (op 741) call()
v_accvgpr_write a13, 0.00000 // (op 741) call()
v_accvgpr_write a14, 0.00000 // (op 741) call()
v_accvgpr_write a15, 0.00000 // (op 741) call()
v_lshrrev_b32 v0, 7, v1 // (op 2823) 
s_mul_i32 s15, s16, s50 // (op 2823) 
s_add_u32 s15, s15, s0 // (op 2823) 
v_and_b32 v5, 15, v1 // (op 2823) 
s_add_u32 s6, s15, s9 // (op 2823) 

v_lshl_add_u32 v10, v0, 4, v5 // (op 2823) 


s_mul_hi_u32 s15, s6, s13 // (op 2823) 
v_and_b32 v0, 63, v1 // (op 2823) 
s_sub_u32 s7, s6, s15 // (op 2823) 
v_lshrrev_b32 v5, 4, v0 // (op 2823) 

s_lshr_b32 s8, s7, 1 // (op 2823) 

v_lshlrev_b32 v0, 5, v5 // (op 2823) 

s_add_u32 s7, s8, s15 // (op 2823) 


s_lshr_b32 s8, s7, s11 // (op 2823) 

s_and_b32 s7, s11, 2147483648 // (op 2823) 
s_cmp_eq_i32 s7, 0 // (op 2823) 
s_mov_b32 s15, scc // (op 2823) 

s_cmp_eq_i32 s15, 0 // (op 2823) 
s_cselect_b32 s7, s6, s8 // (op 2823) 


s_mul_hi_u32 s8, s7, s12 // (op 2823) 
s_sub_u32 s15, s7, s8 // (op 2823) 
s_lshr_b32 s18, s15, 1 // (op 2823) 

s_add_u32 s15, s18, s8 // (op 2823) 


s_lshr_b32 s8, s15, s10 // (op 2823) 

s_and_b32 s15, s10, 2147483648 // (op 2823) 
s_cmp_eq_i32 s15, 0 // (op 2823) 
s_mov_b32 s18, scc // (op 2823) 

s_cmp_eq_i32 s18, 0 // (op 2823) 
s_cselect_b32 s15, s7, s8 // (op 2823) 


v_lshl_add_u32 v5, s15, 5, v10 // (op 2823) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2823) 
s_mul_i32 s8, s7, s18 // (op 2823) 


s_sub_u32 s7, s6, s8 // (op 2823) 


v_lshl_add_u32 v10, s7, 7, v0 // (op 2823) 


v_mov_b32 v13, 0 // (op 2823) convert
v_mov_b32 v12, v10 // (op 2823) convert

v_mul_lo_u32 v11, v5, s31 // (op 2823) most significant: low of low * high
v_mul_hi_u32 v0, v5, s30 // (op 2823) most significant: high of low * low
v_mul_lo_u32 v10, v5, s30 // (op 2823) least significant: low of low * low
v_add_u32 v11, v11, v0 // (op 2823) most significant: sum

v_add_co_u32 v10, s[18:19], v10, v12 // (op 2823) least significant half
v_addc_co_u32 v11, s[18:19], v11, v13, s[18:19] // (op 2823) most significant half



v_lshlrev_b64 v[12:13], 2, v[10:11] // (op 2823) 

v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 2823) 

v_mov_b32 v0, v10 // (op 2823) convert

s_mul_i32 s19, s30, s35 // (op 2825) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 2825) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 2825) least significant
s_mul_hi_u32 s6, s30, s34 // (op 2825) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2825) most significant: sum
s_add_u32 s19, s19, s6 // (op 2825) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 2825) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 2825) 

s_mov_b32 s68, s32 // (op 2825) reg expression
s_mov_b32 s69, s33 // (op 2825) reg expression
s_mov_b32 s70, s18 // (op 2825) reg expression
s_mov_b32 s71, 131072 // (op 2825) call()

v_lshrrev_b32 v5, 7, v1 // (op 2842) 
s_mul_i32 s15, s16, s50 // (op 2842) 
s_add_u32 s15, s15, s0 // (op 2842) 
v_and_b32 v10, 15, v1 // (op 2842) 
s_add_u32 s6, s15, s9 // (op 2842) 

v_lshl_add_u32 v11, v5, 4, v10 // (op 2842) 


s_mul_hi_u32 s15, s6, s13 // (op 2842) 
v_and_b32 v5, 63, v1 // (op 2842) 
s_sub_u32 s7, s6, s15 // (op 2842) 
v_lshrrev_b32 v10, 4, v5 // (op 2842) 

s_lshr_b32 s8, s7, 1 // (op 2842) 

v_lshlrev_b32 v5, 5, v10 // (op 2842) 

s_add_u32 s7, s8, s15 // (op 2842) 


s_lshr_b32 s8, s7, s11 // (op 2842) 

s_and_b32 s7, s11, 2147483648 // (op 2842) 
s_cmp_eq_i32 s7, 0 // (op 2842) 
s_mov_b32 s15, scc // (op 2842) 

s_cmp_eq_i32 s15, 0 // (op 2842) 
s_cselect_b32 s7, s6, s8 // (op 2842) 


s_mul_hi_u32 s8, s7, s12 // (op 2842) 
s_sub_u32 s15, s7, s8 // (op 2842) 
s_lshr_b32 s18, s15, 1 // (op 2842) 

s_add_u32 s15, s18, s8 // (op 2842) 


s_lshr_b32 s8, s15, s10 // (op 2842) 

s_and_b32 s15, s10, 2147483648 // (op 2842) 
s_cmp_eq_i32 s15, 0 // (op 2842) 
s_mov_b32 s18, scc // (op 2842) 

s_cmp_eq_i32 s18, 0 // (op 2842) 
s_cselect_b32 s15, s7, s8 // (op 2842) 


v_lshl_add_u32 v10, s15, 5, v11 // (op 2842) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2842) 
s_mul_i32 s8, s7, s18 // (op 2842) 


s_sub_u32 s7, s6, s8 // (op 2842) 


v_lshl_add_u32 v11, s7, 7, v5 // (op 2842) 


v_mov_b32 v13, 0 // (op 2842) convert
v_mov_b32 v12, v11 // (op 2842) convert

v_mul_lo_u32 v15, v10, s31 // (op 2842) most significant: low of low * high
v_mul_hi_u32 v5, v10, s30 // (op 2842) most significant: high of low * low
v_mul_lo_u32 v14, v10, s30 // (op 2842) least significant: low of low * low
v_add_u32 v15, v15, v5 // (op 2842) most significant: sum

v_add_co_u32 v14, s[18:19], v14, v12 // (op 2842) least significant half
v_addc_co_u32 v15, s[18:19], v15, v13, s[18:19] // (op 2842) most significant half



v_lshlrev_b64 v[10:11], 2, v[14:15] // (op 2842) 

v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 2842) 

v_mov_b32 v5, v12 // (op 2842) convert

s_mul_i32 s19, s30, s35 // (op 2844) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 2844) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 2844) least significant
s_mul_hi_u32 s6, s30, s34 // (op 2844) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2844) most significant: sum
s_add_u32 s19, s19, s6 // (op 2844) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 2844) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 2844) 

s_mov_b32 s68, s32 // (op 2844) reg expression
s_mov_b32 s69, s33 // (op 2844) reg expression
s_mov_b32 s70, s18 // (op 2844) reg expression
s_mov_b32 s71, 131072 // (op 2844) call()

v_lshrrev_b32 v10, 6, v1 // (op 2972) 
s_mul_i32 s15, s16, s50 // (op 2972) 
s_add_u32 s15, s15, s0 // (op 2972) 
v_and_b32 v11, 1, v10 // (op 2972) 

s_add_u32 s6, s15, s9 // (op 2972) 

v_and_b32 v10, 15, v1 // (op 2972) 
s_mul_hi_u32 s15, s6, s13 // (op 2972) 
v_lshl_add_u32 v12, v11, 4, v10 // (op 2972) 


s_sub_u32 s7, s6, s15 // (op 2972) 
v_and_b32 v10, 63, v1 // (op 2972) 
s_lshr_b32 s8, s7, 1 // (op 2972) 

v_lshrrev_b32 v11, 4, v10 // (op 2972) 

s_add_u32 s7, s8, s15 // (op 2972) 


v_lshlrev_b32 v10, 5, v11 // (op 2972) 

s_lshr_b32 s8, s7, s11 // (op 2972) 

s_and_b32 s7, s11, 2147483648 // (op 2972) 
s_cmp_eq_i32 s7, 0 // (op 2972) 
s_mov_b32 s15, scc // (op 2972) 

s_cmp_eq_i32 s15, 0 // (op 2972) 
s_cselect_b32 s7, s6, s8 // (op 2972) 


s_mul_hi_u32 s8, s7, s12 // (op 2972) 
s_sub_u32 s15, s7, s8 // (op 2972) 
s_lshr_b32 s18, s15, 1 // (op 2972) 

s_add_u32 s15, s18, s8 // (op 2972) 


s_lshr_b32 s8, s15, s10 // (op 2972) 

s_and_b32 s15, s10, 2147483648 // (op 2972) 
s_cmp_eq_i32 s15, 0 // (op 2972) 
s_mov_b32 s18, scc // (op 2972) 

s_cmp_eq_i32 s18, 0 // (op 2972) 
s_cselect_b32 s15, s7, s8 // (op 2972) 


s_mul_i32 s8, s15, s14 // (op 2972) 

s_sub_u32 s15, s7, s8 // (op 2972) 

v_lshl_add_u32 v11, s15, 7, v12 // (op 2972) 


v_mul_lo_u32 v13, v11, s37 // (op 2972) most significant: low of low * high
v_mul_hi_u32 v14, v11, s36 // (op 2972) most significant: high of low * low
v_mul_lo_u32 v12, v11, s36 // (op 2972) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 2972) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2972) 
s_mul_i32 s8, s7, s18 // (op 2972) 


s_sub_u32 s7, s6, s8 // (op 2972) 


v_lshl_add_u32 v11, s7, 7, v10 // (op 2972) 


v_mov_b32 v15, 0 // (op 2972) convert
v_mov_b32 v14, v11 // (op 2972) convert

v_add_co_u32 v10, s[18:19], v14, v12 // (op 2972) least significant half
v_addc_co_u32 v11, s[18:19], v15, v13, s[18:19] // (op 2972) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 2972) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 2972) 

v_mov_b32 v10, v12 // (op 2972) convert

s_mul_i32 s19, s36, s23 // (op 2974) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 2974) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 2974) least significant
s_mul_hi_u32 s6, s36, s22 // (op 2974) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2974) most significant: sum
s_add_u32 s19, s19, s6 // (op 2974) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 2974) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 2974) 

s_mov_b32 s72, s38 // (op 2974) reg expression
s_mov_b32 s73, s39 // (op 2974) reg expression
s_mov_b32 s74, s18 // (op 2974) reg expression
s_mov_b32 s75, 131072 // (op 2974) call()

v_lshrrev_b32 v11, 6, v1 // (op 2993) 
s_mul_i32 s15, s16, s50 // (op 2993) 
s_add_u32 s15, s15, s0 // (op 2993) 
v_and_b32 v12, 1, v11 // (op 2993) 

s_add_u32 s6, s15, s9 // (op 2993) 

v_and_b32 v11, 15, v1 // (op 2993) 
s_mul_hi_u32 s15, s6, s13 // (op 2993) 
v_lshl_add_u32 v13, v12, 4, v11 // (op 2993) 


s_sub_u32 s7, s6, s15 // (op 2993) 
v_and_b32 v11, 63, v1 // (op 2993) 
s_lshr_b32 s8, s7, 1 // (op 2993) 

v_lshrrev_b32 v12, 4, v11 // (op 2993) 

s_add_u32 s7, s8, s15 // (op 2993) 


v_lshlrev_b32 v11, 5, v12 // (op 2993) 

s_lshr_b32 s8, s7, s11 // (op 2993) 

s_and_b32 s7, s11, 2147483648 // (op 2993) 
s_cmp_eq_i32 s7, 0 // (op 2993) 
s_mov_b32 s15, scc // (op 2993) 

s_cmp_eq_i32 s15, 0 // (op 2993) 
s_cselect_b32 s7, s6, s8 // (op 2993) 


s_mul_hi_u32 s8, s7, s12 // (op 2993) 
s_sub_u32 s15, s7, s8 // (op 2993) 
s_lshr_b32 s18, s15, 1 // (op 2993) 

s_add_u32 s15, s18, s8 // (op 2993) 


s_lshr_b32 s8, s15, s10 // (op 2993) 

s_and_b32 s15, s10, 2147483648 // (op 2993) 
s_cmp_eq_i32 s15, 0 // (op 2993) 
s_mov_b32 s18, scc // (op 2993) 

s_cmp_eq_i32 s18, 0 // (op 2993) 
s_cselect_b32 s15, s7, s8 // (op 2993) 


s_mul_i32 s8, s15, s14 // (op 2993) 

s_sub_u32 s15, s7, s8 // (op 2993) 

v_lshl_add_u32 v12, s15, 7, v13 // (op 2993) 


v_mul_lo_u32 v15, v12, s37 // (op 2993) most significant: low of low * high
v_mul_hi_u32 v13, v12, s36 // (op 2993) most significant: high of low * low
v_mul_lo_u32 v14, v12, s36 // (op 2993) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 2993) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2993) 
s_mul_i32 s8, s7, s18 // (op 2993) 


s_sub_u32 s7, s6, s8 // (op 2993) 


v_lshl_add_u32 v12, s7, 7, v11 // (op 2993) 


v_mov_b32 v17, 0 // (op 2993) convert
v_mov_b32 v16, v12 // (op 2993) convert

v_add_co_u32 v12, s[18:19], v16, v14 // (op 2993) least significant half
v_addc_co_u32 v13, s[18:19], v17, v15, s[18:19] // (op 2993) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 2993) 


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 2993) 

v_mov_b32 v11, v14 // (op 2993) convert

s_mul_i32 s19, s36, s23 // (op 2995) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 2995) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 2995) least significant
s_mul_hi_u32 s6, s36, s22 // (op 2995) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2995) most significant: sum
s_add_u32 s19, s19, s6 // (op 2995) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 2995) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 2995) 

s_mov_b32 s72, s38 // (op 2995) reg expression
s_mov_b32 s73, s39 // (op 2995) reg expression
s_mov_b32 s74, s18 // (op 2995) reg expression
s_mov_b32 s75, 131072 // (op 2995) call()

v_lshrrev_b32 v12, 7, v1 // (op 3115) 
s_mul_i32 s15, s16, s50 // (op 3115) 
s_add_u32 s15, s15, s0 // (op 3115) 
v_and_b32 v13, 15, v1 // (op 3115) 
s_add_u32 s6, s15, s9 // (op 3115) 

v_lshl_add_u32 v14, v12, 4, v13 // (op 3115) 


s_mul_hi_u32 s15, s6, s13 // (op 3115) 
v_and_b32 v12, 63, v1 // (op 3115) 
s_sub_u32 s7, s6, s15 // (op 3115) 
v_lshrrev_b32 v13, 4, v12 // (op 3115) 

s_lshr_b32 s8, s7, 1 // (op 3115) 

s_add_u32 s7, s8, s15 // (op 3115) 


s_lshr_b32 s8, s7, s11 // (op 3115) 

s_and_b32 s7, s11, 2147483648 // (op 3115) 
s_cmp_eq_i32 s7, 0 // (op 3115) 
s_mov_b32 s15, scc // (op 3115) 

s_cmp_eq_i32 s15, 0 // (op 3115) 
s_cselect_b32 s7, s6, s8 // (op 3115) 


s_mul_hi_u32 s8, s7, s12 // (op 3115) 
s_sub_u32 s15, s7, s8 // (op 3115) 
s_lshr_b32 s18, s15, 1 // (op 3115) 

s_add_u32 s15, s18, s8 // (op 3115) 


s_lshr_b32 s8, s15, s10 // (op 3115) 

s_and_b32 s15, s10, 2147483648 // (op 3115) 
s_cmp_eq_i32 s15, 0 // (op 3115) 
s_mov_b32 s18, scc // (op 3115) 

s_cmp_eq_i32 s18, 0 // (op 3115) 
s_cselect_b32 s15, s7, s8 // (op 3115) 


v_lshl_add_u32 v12, s15, 5, v14 // (op 3115) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3115) 
s_mul_i32 s8, s7, s18 // (op 3115) 


s_sub_u32 s7, s6, s8 // (op 3115) 


v_lshl_add_u32 v14, s7, 2, v13 // (op 3115) 


v_mul_lo_u32 v13, v12, s40 // (op 3115) 
v_add_u32 v13, v13, v14 // (op 3115) 


s_mul_i32 s19, s40, s35 // (op 3117) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 3117) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 3117) least significant
s_mul_hi_u32 s6, s40, s34 // (op 3117) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3117) most significant: sum
s_add_u32 s19, s19, s6 // (op 3117) most significant: sum


s_mov_b32 s76, s42 // (op 3117) reg expression
s_mov_b32 s77, s43 // (op 3117) reg expression
s_mov_b32 s78, s18 // (op 3117) reg expression
s_mov_b32 s79, 131072 // (op 3117) call()

v_lshrrev_b32 v12, 7, v1 // (op 3134) 
s_mul_i32 s15, s16, s50 // (op 3134) 
s_add_u32 s15, s15, s0 // (op 3134) 
v_and_b32 v14, 15, v1 // (op 3134) 
s_add_u32 s6, s15, s9 // (op 3134) 

v_lshl_add_u32 v15, v12, 4, v14 // (op 3134) 


s_mul_hi_u32 s15, s6, s13 // (op 3134) 
v_and_b32 v12, 63, v1 // (op 3134) 
s_sub_u32 s7, s6, s15 // (op 3134) 
v_lshrrev_b32 v14, 4, v12 // (op 3134) 

s_lshr_b32 s8, s7, 1 // (op 3134) 

s_add_u32 s7, s8, s15 // (op 3134) 


s_lshr_b32 s8, s7, s11 // (op 3134) 

s_and_b32 s7, s11, 2147483648 // (op 3134) 
s_cmp_eq_i32 s7, 0 // (op 3134) 
s_mov_b32 s15, scc // (op 3134) 

s_cmp_eq_i32 s15, 0 // (op 3134) 
s_cselect_b32 s7, s6, s8 // (op 3134) 


s_mul_hi_u32 s8, s7, s12 // (op 3134) 
s_sub_u32 s15, s7, s8 // (op 3134) 
s_lshr_b32 s18, s15, 1 // (op 3134) 

s_add_u32 s15, s18, s8 // (op 3134) 


s_lshr_b32 s8, s15, s10 // (op 3134) 

s_and_b32 s15, s10, 2147483648 // (op 3134) 
s_cmp_eq_i32 s15, 0 // (op 3134) 
s_mov_b32 s18, scc // (op 3134) 

s_cmp_eq_i32 s18, 0 // (op 3134) 
s_cselect_b32 s15, s7, s8 // (op 3134) 


v_lshl_add_u32 v12, s15, 5, v15 // (op 3134) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3134) 
s_mul_i32 s8, s7, s18 // (op 3134) 


s_sub_u32 s7, s6, s8 // (op 3134) 


v_lshl_add_u32 v15, s7, 2, v14 // (op 3134) 


v_mul_lo_u32 v14, v12, s40 // (op 3134) 
v_add_u32 v14, v14, v15 // (op 3134) 


s_mul_i32 s19, s40, s35 // (op 3136) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 3136) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 3136) least significant
s_mul_hi_u32 s6, s40, s34 // (op 3136) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3136) most significant: sum
s_add_u32 s19, s19, s6 // (op 3136) most significant: sum


s_mov_b32 s76, s42 // (op 3136) reg expression
s_mov_b32 s77, s43 // (op 3136) reg expression
s_mov_b32 s78, s18 // (op 3136) reg expression
s_mov_b32 s79, 131072 // (op 3136) call()

v_lshrrev_b32 v12, 2, v1 // (op 3243) 
s_mul_i32 s15, s16, s50 // (op 3243) 
s_add_u32 s15, s15, s0 // (op 3243) 
v_lshlrev_b32 v15, 1, v12 // (op 3243) 

s_add_u32 s6, s15, s9 // (op 3243) 

v_and_b32 v12, -4, v1 // (op 3243) 
s_mul_hi_u32 s15, s6, s13 // (op 3243) 
v_sub_u32 v16, v1, v12 // (op 3243) 

s_sub_u32 s7, s6, s15 // (op 3243) 
s_lshr_b32 s8, s7, 1 // (op 3243) 

s_add_u32 s7, s8, s15 // (op 3243) 


s_lshr_b32 s8, s7, s11 // (op 3243) 

s_and_b32 s7, s11, 2147483648 // (op 3243) 
s_cmp_eq_i32 s7, 0 // (op 3243) 
s_mov_b32 s15, scc // (op 3243) 

s_cmp_eq_i32 s15, 0 // (op 3243) 
s_cselect_b32 s7, s6, s8 // (op 3243) 


s_mul_hi_u32 s8, s7, s12 // (op 3243) 
s_sub_u32 s15, s7, s8 // (op 3243) 
s_lshr_b32 s18, s15, 1 // (op 3243) 

s_add_u32 s15, s18, s8 // (op 3243) 


s_lshr_b32 s8, s15, s10 // (op 3243) 

s_and_b32 s15, s10, 2147483648 // (op 3243) 
s_cmp_eq_i32 s15, 0 // (op 3243) 
s_mov_b32 s18, scc // (op 3243) 

s_cmp_eq_i32 s18, 0 // (op 3243) 
s_cselect_b32 s15, s7, s8 // (op 3243) 


s_mul_i32 s8, s15, s14 // (op 3243) 

s_sub_u32 s15, s7, s8 // (op 3243) 

v_lshl_add_u32 v12, s15, 7, v15 // (op 3243) 


v_mul_lo_u32 v15, v12, s44 // (op 3243) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 3243) 
s_mul_i32 s8, s7, s18 // (op 3243) 


s_sub_u32 s7, s6, s8 // (op 3243) 


v_lshl_add_u32 v12, s7, 2, v16 // (op 3243) 


v_add_u32 v16, v12, v15 // (op 3243) 


s_mul_i32 s19, s44, s23 // (op 3245) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 3245) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 3245) least significant
s_mul_hi_u32 s6, s44, s22 // (op 3245) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3245) most significant: sum
s_add_u32 s19, s19, s6 // (op 3245) most significant: sum


s_mov_b32 s80, s46 // (op 3245) reg expression
s_mov_b32 s81, s47 // (op 3245) reg expression
s_mov_b32 s82, s18 // (op 3245) reg expression
s_mov_b32 s83, 131072 // (op 3245) call()

v_lshrrev_b32 v12, 2, v1 // (op 3262) 
s_mul_i32 s15, s16, s50 // (op 3262) 
s_add_u32 s15, s15, s0 // (op 3262) 
v_lshlrev_b32 v15, 1, v12 // (op 3262) 

s_add_u32 s6, s15, s9 // (op 3262) 

v_and_b32 v12, -4, v1 // (op 3262) 
s_mul_hi_u32 s15, s6, s13 // (op 3262) 
v_sub_u32 v17, v1, v12 // (op 3262) 

s_sub_u32 s7, s6, s15 // (op 3262) 
s_lshr_b32 s8, s7, 1 // (op 3262) 

s_add_u32 s7, s8, s15 // (op 3262) 


s_lshr_b32 s8, s7, s11 // (op 3262) 

s_and_b32 s7, s11, 2147483648 // (op 3262) 
s_cmp_eq_i32 s7, 0 // (op 3262) 
s_mov_b32 s15, scc // (op 3262) 

s_cmp_eq_i32 s15, 0 // (op 3262) 
s_cselect_b32 s7, s6, s8 // (op 3262) 


s_mul_hi_u32 s8, s7, s12 // (op 3262) 
s_sub_u32 s15, s7, s8 // (op 3262) 
s_lshr_b32 s18, s15, 1 // (op 3262) 

s_add_u32 s15, s18, s8 // (op 3262) 


s_lshr_b32 s8, s15, s10 // (op 3262) 

s_and_b32 s15, s10, 2147483648 // (op 3262) 
s_cmp_eq_i32 s15, 0 // (op 3262) 
s_mov_b32 s18, scc // (op 3262) 

s_cmp_eq_i32 s18, 0 // (op 3262) 
s_cselect_b32 s15, s7, s8 // (op 3262) 


s_mul_i32 s8, s15, s14 // (op 3262) 

s_sub_u32 s15, s7, s8 // (op 3262) 

v_lshl_add_u32 v12, s15, 7, v15 // (op 3262) 


v_mul_lo_u32 v15, v12, s44 // (op 3262) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 3262) 
s_mul_i32 s8, s7, s18 // (op 3262) 


s_sub_u32 s7, s6, s8 // (op 3262) 


v_lshl_add_u32 v12, s7, 2, v17 // (op 3262) 


v_add_u32 v17, v12, v15 // (op 3262) 


s_mul_i32 s19, s44, s23 // (op 3264) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 3264) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 3264) least significant
s_mul_hi_u32 s6, s44, s22 // (op 3264) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3264) most significant: sum
s_add_u32 s19, s19, s6 // (op 3264) most significant: sum


s_mov_b32 s80, s46 // (op 3264) reg expression
s_mov_b32 s81, s47 // (op 3264) reg expression
s_mov_b32 s82, s18 // (op 3264) reg expression
s_mov_b32 s83, 131072 // (op 3264) call()




buffer_load_dwordx4 v[18:21], v0, s[68:71], 0 offen // (op 2435) Load value


v_mov_b32 v12, v10 // (op 2441) reg expression

buffer_load_dwordx4 v[22:25], v12, s[72:75], 0 offen // (op 2441) Load value




buffer_load_ubyte v12, v13, s[76:79], 0 offen offset: 0 // (op 2447) Load value


s_lshl_b64 s[18:19], s[36:37], 5 // (op 2453) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2453) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2453) 

v_add_u32 v15, v10, s18 // (op 2453) 


buffer_load_dwordx4 v[26:29], v15, s[72:75], 0 offen // (op 2453) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 2460) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2460) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2460) 

s_lshl_b32 s15, s18, 1 // (op 2460) 

v_add_u32 v15, v10, s15 // (op 2460) 


buffer_load_dwordx4 v[30:33], v15, s[72:75], 0 offen // (op 2460) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 2467) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2467) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2467) 

s_mul_i32 s15, 3, s18 // (op 2467) 

v_add_u32 v15, v10, s15 // (op 2467) 


buffer_load_dwordx4 v[34:37], v15, s[72:75], 0 offen // (op 2467) Load value


v_mov_b64 v[38:39], s[44:45] // (op 2472) convert

v_mov_b32 v15, v16 // (op 2472) 
buffer_load_ubyte v40, v15, s[80:83], 0 offen offset: 0 // (op 2472) Load value
v_add_u32 v15, v15, v38 // (op 2472) 
buffer_load_ubyte v41, v15, s[80:83], 0 offen offset: 0 // (op 2472) Load value





buffer_load_dwordx4 v[42:45], v0, s[68:71], 0 offen offset:64 // (op 2476) Load value



v_add_u32 v0, 64, v10 // (op 2482) 

buffer_load_dwordx4 v[46:49], v0, s[72:75], 0 offen // (op 2482) Load value




buffer_load_ubyte v0, v13, s[76:79], 0 offen offset:4 // (op 2488) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 2494) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2494) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2494) 

s_add_u32 s15, s18, 64 // (op 2494) 

v_add_u32 v13, v10, s15 // (op 2494) 


buffer_load_dwordx4 v[50:53], v13, s[72:75], 0 offen // (op 2494) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 2501) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2501) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2501) 

s_lshl_b32 s15, s18, 1 // (op 2501) 
s_add_u32 s15, s15, 64 // (op 2501) 

v_add_u32 v13, v10, s15 // (op 2501) 


buffer_load_dwordx4 v[54:57], v13, s[72:75], 0 offen // (op 2501) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 2508) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2508) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2508) 

s_mul_i32 s15, 3, s18 // (op 2508) 
s_add_u32 s15, s15, 64 // (op 2508) 

v_add_u32 v13, v10, s15 // (op 2508) 


buffer_load_dwordx4 v[58:61], v13, s[72:75], 0 offen // (op 2508) Load value



v_mov_b64 v[38:39], s[44:45] // (op 2513) convert

v_mov_b32 v10, v16 // (op 2513) 
buffer_load_ubyte v62, v10, s[80:83], 0 offen offset:4 // (op 2513) Load value
v_add_u32 v10, v10, v38 // (op 2513) 
buffer_load_ubyte v63, v10, s[80:83], 0 offen offset:4 // (op 2513) Load value



v_and_b32 v10, -4, v1 // (op 3497) 
v_sub_u32 v13, v1, v10 // (op 3497) 

v_lshrrev_b32 v10, 2, v1 // (op 3497) 
v_lshl_add_u32 v15, v10, 3, v13 // (op 3497) 


s_barrier  // (op 3725) 



s_waitcnt vmcnt(9)
ds_write_b8 v15, v40 // (op 2515) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v15, v41 offset:4 // (op 2515) Store local data 

v_and_b32 v10, -4, v1 // (op 3468) 
v_sub_u32 v13, v1, v10 // (op 3468) 

v_lshrrev_b32 v10, 2, v1 // (op 3468) 
v_lshl_add_u32 v15, v10, 3, v13 // (op 3468) 


v_and_b32 v10, -4, v1 // (op 3568) 
v_sub_u32 v13, v1, v10 // (op 3568) 

v_lshrrev_b32 v10, 2, v1 // (op 3568) 
v_lshl_add_u32 v16, v10, 3, v13 // (op 3568) 


s_mov_b32 s1, 0 // (op 743) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 742) 
s_mov_b32 s15, s18 // (op 742) convert

s_and_b32 s6, s15, 4294967294 // (op 742) 

s_cmp_lt_u32 s1, s6 // (op 742) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_742 // (op 742) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_742 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_742:
s_add_u32 s1, s1, 2 // (op 744) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3722) 



ds_read_u8 v10, v3 // (op 160) Load local data 



ds_read_u8 v13, v3 offset:128 // (op 536) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[22:25], a[0:3], v12, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 161) 




ds_read_u8 v10, v3 offset:256 // (op 652) Load local data 



ds_read_u8 v38, v3 offset:384 // (op 768) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[26:29], a[4:7], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 537) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[30:33], a[8:11], v12, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 653) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[34:37], a[12:15], v12, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 769) 



v_mov_b32 v10, v5 // (op 163) 

buffer_load_dwordx4 v[18:21], v10, s[68:71], 0 offen offset:128 // (op 163) Load value


v_mov_b32 v13, v11 // (op 165) 
v_add_u32 v38, 128, v13 // (op 165) 

buffer_load_dwordx4 v[22:25], v38, s[72:75], 0 offen // (op 165) Load value



v_mov_b32 v38, v14 // (op 167) 

buffer_load_ubyte v12, v38, s[76:79], 0 offen offset:8 // (op 167) Load value


s_lshl_b64 s[18:19], s[36:37], 5 // (op 559) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 559) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 559) 

s_add_u32 s15, s18, 128 // (op 559) 

v_add_u32 v39, v13, s15 // (op 559) 


buffer_load_dwordx4 v[26:29], v39, s[72:75], 0 offen // (op 559) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 675) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 675) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 675) 

s_lshl_b32 s15, s18, 1 // (op 675) 
s_add_u32 s15, s15, 128 // (op 675) 

v_add_u32 v39, v13, s15 // (op 675) 


buffer_load_dwordx4 v[30:33], v39, s[72:75], 0 offen // (op 675) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 791) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 791) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 791) 

s_mul_i32 s15, 3, s18 // (op 791) 
s_add_u32 s15, s15, 128 // (op 791) 

v_add_u32 v39, v13, s15 // (op 791) 


buffer_load_dwordx4 v[34:37], v39, s[72:75], 0 offen // (op 791) Load value


v_mov_b64 v[64:65], s[44:45] // (op 157) convert

v_mov_b32 v39, v17 // (op 157) 
v_mov_b32 v66, v39 // (op 157) 
buffer_load_ubyte v40, v66, s[80:83], 0 offen offset:8 // (op 157) Load value
v_add_u32 v66, v66, v64 // (op 157) 
buffer_load_ubyte v41, v66, s[80:83], 0 offen offset:8 // (op 157) Load value


s_barrier  // (op 3741) 



s_waitcnt vmcnt(9)
ds_write_b8 v16, v62 offset:512 // (op 452) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v16, v63 offset:516 // (op 452) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3738) 



ds_read_u8 v64, v4 offset:512 // (op 454) Load local data 



ds_read_u8 v65, v4 offset:640 // (op 540) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:45], v[46:49], a[0:3], v0, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 455) 




ds_read_u8 v64, v4 offset:768 // (op 656) Load local data 



ds_read_u8 v66, v4 offset:896 // (op 772) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:45], v[50:53], a[4:7], v0, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 538) 

s_barrier  // (op 3848) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[42:45], v[54:57], a[8:11], v0, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 654) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:45], v[58:61], a[12:15], v0, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 770) 




buffer_load_dwordx4 v[42:45], v10, s[68:71], 0 offen offset:192 // (op 457) Load value



v_add_u32 v5, 128, v5 // (op 2834) 
v_add_u32 v10, 192, v13 // (op 459) 

buffer_load_dwordx4 v[46:49], v10, s[72:75], 0 offen // (op 459) Load value




buffer_load_ubyte v0, v38, s[76:79], 0 offen offset:12 // (op 461) Load value



v_add_u32 v14, 8, v14 // (op 3126) 
s_lshl_b64 s[18:19], s[36:37], 5 // (op 554) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 554) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 554) 

s_add_u32 s15, s18, 192 // (op 554) 

v_add_u32 v10, v13, s15 // (op 554) 


buffer_load_dwordx4 v[50:53], v10, s[72:75], 0 offen // (op 554) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 670) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 670) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 670) 

s_lshl_b32 s15, s18, 1 // (op 670) 
s_add_u32 s15, s15, 192 // (op 670) 

v_add_u32 v10, v13, s15 // (op 670) 


buffer_load_dwordx4 v[54:57], v10, s[72:75], 0 offen // (op 670) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 786) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 786) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 786) 

s_mul_i32 s15, 3, s18 // (op 786) 
s_add_u32 s15, s15, 192 // (op 786) 

v_add_u32 v10, v13, s15 // (op 786) 


buffer_load_dwordx4 v[58:61], v10, s[72:75], 0 offen // (op 786) Load value



v_add_u32 v11, 128, v11 // (op 2983) 
v_mov_b64 v[64:65], s[44:45] // (op 451) convert

v_mov_b32 v10, v39 // (op 451) 
buffer_load_ubyte v62, v10, s[80:83], 0 offen offset:12 // (op 451) Load value
v_add_u32 v10, v10, v64 // (op 451) 
buffer_load_ubyte v63, v10, s[80:83], 0 offen offset:12 // (op 451) Load value



s_barrier  // (op 3728) 
v_add_u32 v17, 8, v17 // (op 3254) 



s_waitcnt vmcnt(9)
ds_write_b8 v15, v40 // (op 158) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v15, v41 offset:4 // (op 158) Store local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 742) 
s_mov_b32 s15, s18 // (op 742) convert

s_and_b32 s6, s15, 4294967294 // (op 742) 

s_cmp_lt_u32 s1, s6 // (op 742) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_742 // (op 742) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_742 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_742:













v_lshrrev_b32 v5, 7, v1 // (op 2770) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2770) 
v_and_b32 v10, 15, v1 // (op 2770) 
s_and_b32 s15, s18, 4294967294 // (op 2770) 
v_lshl_add_u32 v11, v5, 4, v10 // (op 2770) 


s_add_u32 s6, s0, s15 // (op 2770) 

v_and_b32 v5, 63, v1 // (op 2770) 
v_lshrrev_b32 v10, 4, v5 // (op 2770) 

s_mul_i32 s15, s16, s50 // (op 2770) 
s_add_u32 s15, s15, s6 // (op 2770) 

v_lshlrev_b32 v5, 5, v10 // (op 2770) 

s_add_u32 s6, s15, s9 // (op 2770) 

s_mul_hi_u32 s15, s6, s13 // (op 2770) 
s_sub_u32 s7, s6, s15 // (op 2770) 
s_lshr_b32 s8, s7, 1 // (op 2770) 

s_add_u32 s7, s8, s15 // (op 2770) 


s_lshr_b32 s8, s7, s11 // (op 2770) 

s_and_b32 s7, s11, 2147483648 // (op 2770) 
s_cmp_eq_i32 s7, 0 // (op 2770) 
s_mov_b32 s15, scc // (op 2770) 

s_cmp_eq_i32 s15, 0 // (op 2770) 
s_cselect_b32 s7, s6, s8 // (op 2770) 


s_mul_hi_u32 s8, s7, s12 // (op 2770) 
s_sub_u32 s15, s7, s8 // (op 2770) 
s_lshr_b32 s28, s15, 1 // (op 2770) 

s_add_u32 s15, s28, s8 // (op 2770) 


s_lshr_b32 s8, s15, s10 // (op 2770) 

s_and_b32 s15, s10, 2147483648 // (op 2770) 
s_cmp_eq_i32 s15, 0 // (op 2770) 
s_mov_b32 s28, scc // (op 2770) 

s_cmp_eq_i32 s28, 0 // (op 2770) 
s_cselect_b32 s15, s7, s8 // (op 2770) 


s_mul_i32 s8, s7, s18 // (op 2770) 


v_lshl_add_u32 v10, s15, 5, v11 // (op 2770) 


s_sub_u32 s7, s6, s8 // (op 2770) 


v_lshl_add_u32 v11, s7, 7, v5 // (op 2770) 


v_mov_b32 v13, 0 // (op 2770) convert
v_mov_b32 v12, v11 // (op 2770) convert

v_mul_lo_u32 v15, v10, s31 // (op 2770) most significant: low of low * high
v_mul_hi_u32 v5, v10, s30 // (op 2770) most significant: high of low * low
v_mul_lo_u32 v14, v10, s30 // (op 2770) least significant: low of low * low
v_add_u32 v15, v15, v5 // (op 2770) most significant: sum

v_add_co_u32 v14, s[18:19], v14, v12 // (op 2770) least significant half
v_addc_co_u32 v15, s[18:19], v15, v13, s[18:19] // (op 2770) most significant half



v_lshlrev_b64 v[10:11], 2, v[14:15] // (op 2770) 

v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 2770) 

v_mov_b32 v5, v12 // (op 2770) convert

s_mul_i32 s19, s30, s35 // (op 2772) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 2772) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 2772) least significant
s_mul_hi_u32 s6, s30, s34 // (op 2772) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2772) most significant: sum
s_add_u32 s19, s19, s6 // (op 2772) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 2772) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 2772) 

s_mov_b32 s68, s32 // (op 2772) reg expression
s_mov_b32 s69, s33 // (op 2772) reg expression
s_mov_b32 s70, s18 // (op 2772) reg expression
s_mov_b32 s71, 131072 // (op 2772) call()

v_lshrrev_b32 v10, 6, v1 // (op 2914) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2914) 
v_and_b32 v11, 1, v10 // (op 2914) 

s_and_b32 s15, s18, 4294967294 // (op 2914) 
v_and_b32 v10, 15, v1 // (op 2914) 
s_add_u32 s6, s0, s15 // (op 2914) 

v_lshl_add_u32 v12, v11, 4, v10 // (op 2914) 


v_and_b32 v10, 63, v1 // (op 2914) 
s_mul_i32 s15, s16, s50 // (op 2914) 
s_add_u32 s15, s15, s6 // (op 2914) 

v_lshrrev_b32 v11, 4, v10 // (op 2914) 

s_add_u32 s6, s15, s9 // (op 2914) 

v_lshlrev_b32 v10, 5, v11 // (op 2914) 

s_mul_hi_u32 s15, s6, s13 // (op 2914) 
s_sub_u32 s7, s6, s15 // (op 2914) 
s_lshr_b32 s8, s7, 1 // (op 2914) 

s_add_u32 s7, s8, s15 // (op 2914) 


s_lshr_b32 s8, s7, s11 // (op 2914) 

s_and_b32 s7, s11, 2147483648 // (op 2914) 
s_cmp_eq_i32 s7, 0 // (op 2914) 
s_mov_b32 s15, scc // (op 2914) 

s_cmp_eq_i32 s15, 0 // (op 2914) 
s_cselect_b32 s7, s6, s8 // (op 2914) 


s_mul_hi_u32 s8, s7, s12 // (op 2914) 
s_sub_u32 s15, s7, s8 // (op 2914) 
s_lshr_b32 s28, s15, 1 // (op 2914) 

s_add_u32 s15, s28, s8 // (op 2914) 


s_lshr_b32 s8, s15, s10 // (op 2914) 

s_and_b32 s15, s10, 2147483648 // (op 2914) 
s_cmp_eq_i32 s15, 0 // (op 2914) 
s_mov_b32 s28, scc // (op 2914) 

s_cmp_eq_i32 s28, 0 // (op 2914) 
s_cselect_b32 s15, s7, s8 // (op 2914) 


s_mul_i32 s8, s15, s14 // (op 2914) 

s_sub_u32 s15, s7, s8 // (op 2914) 

s_mul_i32 s8, s7, s18 // (op 2914) 


v_lshl_add_u32 v11, s15, 7, v12 // (op 2914) 


s_sub_u32 s7, s6, s8 // (op 2914) 


v_mul_lo_u32 v13, v11, s37 // (op 2914) most significant: low of low * high
v_mul_hi_u32 v14, v11, s36 // (op 2914) most significant: high of low * low
v_mul_lo_u32 v12, v11, s36 // (op 2914) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 2914) most significant: sum


v_lshl_add_u32 v11, s7, 7, v10 // (op 2914) 


v_mov_b32 v15, 0 // (op 2914) convert
v_mov_b32 v14, v11 // (op 2914) convert

v_add_co_u32 v10, s[18:19], v14, v12 // (op 2914) least significant half
v_addc_co_u32 v11, s[18:19], v15, v13, s[18:19] // (op 2914) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 2914) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 2914) 

v_mov_b32 v10, v12 // (op 2914) convert

s_mul_i32 s19, s36, s23 // (op 2916) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 2916) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 2916) least significant
s_mul_hi_u32 s6, s36, s22 // (op 2916) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2916) most significant: sum
s_add_u32 s19, s19, s6 // (op 2916) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 2916) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 2916) 

s_mov_b32 s72, s38 // (op 2916) reg expression
s_mov_b32 s73, s39 // (op 2916) reg expression
s_mov_b32 s74, s18 // (op 2916) reg expression
s_mov_b32 s75, 131072 // (op 2916) call()

v_lshrrev_b32 v11, 7, v1 // (op 3070) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 3070) 
v_and_b32 v12, 15, v1 // (op 3070) 
s_and_b32 s15, s18, 4294967294 // (op 3070) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 3070) 


s_add_u32 s6, s0, s15 // (op 3070) 

v_and_b32 v11, 63, v1 // (op 3070) 
v_lshrrev_b32 v12, 4, v11 // (op 3070) 

s_mul_i32 s15, s16, s50 // (op 3070) 
s_add_u32 s15, s15, s6 // (op 3070) 

s_add_u32 s6, s15, s9 // (op 3070) 

s_mul_hi_u32 s15, s6, s13 // (op 3070) 
s_sub_u32 s7, s6, s15 // (op 3070) 
s_lshr_b32 s8, s7, 1 // (op 3070) 

s_add_u32 s7, s8, s15 // (op 3070) 


s_lshr_b32 s8, s7, s11 // (op 3070) 

s_and_b32 s7, s11, 2147483648 // (op 3070) 
s_cmp_eq_i32 s7, 0 // (op 3070) 
s_mov_b32 s15, scc // (op 3070) 

s_cmp_eq_i32 s15, 0 // (op 3070) 
s_cselect_b32 s7, s6, s8 // (op 3070) 


s_mul_hi_u32 s8, s7, s12 // (op 3070) 
s_sub_u32 s15, s7, s8 // (op 3070) 
s_lshr_b32 s28, s15, 1 // (op 3070) 

s_add_u32 s15, s28, s8 // (op 3070) 


s_lshr_b32 s8, s15, s10 // (op 3070) 

s_and_b32 s15, s10, 2147483648 // (op 3070) 
s_cmp_eq_i32 s15, 0 // (op 3070) 
s_mov_b32 s28, scc // (op 3070) 

s_cmp_eq_i32 s28, 0 // (op 3070) 
s_cselect_b32 s15, s7, s8 // (op 3070) 


s_mul_i32 s8, s7, s18 // (op 3070) 


v_lshl_add_u32 v11, s15, 5, v13 // (op 3070) 


s_sub_u32 s7, s6, s8 // (op 3070) 


v_lshl_add_u32 v13, s7, 2, v12 // (op 3070) 


v_mul_lo_u32 v12, v11, s40 // (op 3070) 
v_add_u32 v12, v12, v13 // (op 3070) 


s_mul_i32 s19, s40, s35 // (op 3072) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 3072) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 3072) least significant
s_mul_hi_u32 s6, s40, s34 // (op 3072) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3072) most significant: sum
s_add_u32 s19, s19, s6 // (op 3072) most significant: sum


s_mov_b32 s76, s42 // (op 3072) reg expression
s_mov_b32 s77, s43 // (op 3072) reg expression
s_mov_b32 s78, s18 // (op 3072) reg expression
s_mov_b32 s79, 131072 // (op 3072) call()

v_lshrrev_b32 v11, 2, v1 // (op 3198) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 3198) 
v_lshlrev_b32 v13, 1, v11 // (op 3198) 

s_and_b32 s15, s18, 4294967294 // (op 3198) 
v_and_b32 v11, -4, v1 // (op 3198) 
s_add_u32 s6, s0, s15 // (op 3198) 

v_sub_u32 v14, v1, v11 // (op 3198) 

s_mul_i32 s15, s16, s50 // (op 3198) 
s_add_u32 s15, s15, s6 // (op 3198) 

s_add_u32 s6, s15, s9 // (op 3198) 

s_mul_hi_u32 s15, s6, s13 // (op 3198) 
s_sub_u32 s7, s6, s15 // (op 3198) 
s_lshr_b32 s8, s7, 1 // (op 3198) 

s_add_u32 s7, s8, s15 // (op 3198) 


s_lshr_b32 s8, s7, s11 // (op 3198) 

s_and_b32 s7, s11, 2147483648 // (op 3198) 
s_cmp_eq_i32 s7, 0 // (op 3198) 
s_mov_b32 s15, scc // (op 3198) 

s_cmp_eq_i32 s15, 0 // (op 3198) 
s_cselect_b32 s7, s6, s8 // (op 3198) 


s_mul_hi_u32 s8, s7, s12 // (op 3198) 
s_sub_u32 s15, s7, s8 // (op 3198) 
s_lshr_b32 s28, s15, 1 // (op 3198) 

s_add_u32 s15, s28, s8 // (op 3198) 


s_lshr_b32 s8, s15, s10 // (op 3198) 

s_and_b32 s15, s10, 2147483648 // (op 3198) 
s_cmp_eq_i32 s15, 0 // (op 3198) 
s_mov_b32 s28, scc // (op 3198) 

s_cmp_eq_i32 s28, 0 // (op 3198) 
s_cselect_b32 s15, s7, s8 // (op 3198) 


s_mul_i32 s8, s15, s14 // (op 3198) 

s_sub_u32 s15, s7, s8 // (op 3198) 

v_lshl_add_u32 v11, s15, 7, v13 // (op 3198) 


s_mul_i32 s8, s7, s18 // (op 3198) 


v_mul_lo_u32 v13, v11, s44 // (op 3198) 

s_sub_u32 s7, s6, s8 // (op 3198) 


v_lshl_add_u32 v11, s7, 2, v14 // (op 3198) 


v_add_u32 v14, v11, v13 // (op 3198) 


s_mul_i32 s19, s44, s23 // (op 3200) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 3200) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 3200) least significant
s_mul_hi_u32 s6, s44, s22 // (op 3200) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3200) most significant: sum
s_add_u32 s19, s19, s6 // (op 3200) most significant: sum


s_mov_b32 s80, s46 // (op 3200) reg expression
s_mov_b32 s81, s47 // (op 3200) reg expression
s_mov_b32 s82, s18 // (op 3200) reg expression
s_mov_b32 s83, 131072 // (op 3200) call()

v_and_b32 v11, -4, v1 // (op 3481) 
v_sub_u32 v13, v1, v11 // (op 3481) 

v_lshrrev_b32 v11, 2, v1 // (op 3481) 
v_lshl_add_u32 v15, v11, 3, v13 // (op 3481) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 748) 
s_and_b32 s15, s18, 4294967294 // (op 748) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 747) 
s_mov_b32 s6, s18 // (op 747) convert

s_cmp_lt_u32 s15, s6 // (op 747) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_747 // (op 747) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_747 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_747:

s_add_u32 s15, s15, 1 // (op 749) 
v_mov_b64 v[16:17], s[44:45] // (op 409) convert

v_mov_b32 v0, v14 // (op 409) 
v_mov_b32 v11, v0 // (op 409) 
buffer_load_ubyte v18, v11, s[80:83], 0 offen offset: 0 // (op 409) Load value
v_add_u32 v11, v11, v16 // (op 409) 
buffer_load_ubyte v19, v11, s[80:83], 0 offen offset: 0 // (op 409) Load value



v_add_u32 v14, 4, v14 // (op 3190) 


s_barrier  // (op 3731) 
v_mov_b32 v0, v5 // (op 415) 




buffer_load_dwordx4 v[20:23], v0, s[68:71], 0 offen // (op 415) Load value

v_add_u32 v5, 64, v5 // (op 2760) 
s_waitcnt vmcnt(2)
ds_write_b8 v15, v18 // (op 410) Store local data 


v_mov_b32 v0, v10 // (op 417) 
v_mov_b32 v11, v0 // (op 417) reg expression

buffer_load_dwordx4 v[24:27], v11, s[72:75], 0 offen // (op 417) Load value

s_waitcnt vmcnt(2)
ds_write_b8 v15, v19 offset:4 // (op 410) Store local data 



v_mov_b32 v11, v12 // (op 419) 

buffer_load_ubyte v13, v11, s[76:79], 0 offen offset: 0 // (op 419) Load value

v_add_u32 v12, 4, v12 // (op 3062) 


s_lshl_b64 s[18:19], s[36:37], 5 // (op 529) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 529) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 529) 

v_add_u32 v11, v0, s18 // (op 529) 


buffer_load_dwordx4 v[16:19], v11, s[72:75], 0 offen // (op 529) Load value

s_waitcnt lgkmcnt(0)
s_barrier  // (op 3734) 





ds_read_u8 v11, v3 // (op 412) Load local data 
s_lshl_b64 s[18:19], s[36:37], 5 // (op 645) 



ds_read_u8 v28, v3 offset:128 // (op 521) Load local data 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 645) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 645) 

s_lshl_b32 s6, s18, 1 // (op 645) 

v_add_u32 v29, v0, s6 // (op 645) 


s_waitcnt vmcnt(1) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:27], a[0:3], v13, v11 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 413) 
buffer_load_dwordx4 v[30:33], v29, s[72:75], 0 offen // (op 645) Load value








ds_read_u8 v11, v3 offset:256 // (op 637) Load local data 
s_lshl_b64 s[18:19], s[36:37], 5 // (op 761) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 761) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 761) 

s_mul_i32 s6, 3, s18 // (op 761) 

v_add_u32 v29, v0, s6 // (op 761) 


buffer_load_dwordx4 v[24:27], v29, s[72:75], 0 offen // (op 761) Load value

s_waitcnt vmcnt(2) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:23], v[16:19], a[4:7], v13, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 522) 






v_add_u32 v10, 64, v10 // (op 2904) 
ds_read_u8 v0, v3 offset:384 // (op 753) Load local data 
s_waitcnt vmcnt(1) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:23], v[30:33], a[8:11], v13, v11 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 638) 


s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:23], v[24:27], a[12:15], v13, v0 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 754) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 747) 
s_mov_b32 s6, s18 // (op 747) convert

s_cmp_lt_u32 s15, s6 // (op 747) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_747 // (op 747) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_747 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_747:






s_barrier  // (op 3834) 
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 189) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_189 // (op 189) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_189
v_accvgpr_read v10, a0 // (op 197) 
v_accvgpr_read v11, a1 // (op 197) 
v_accvgpr_read v12, a2 // (op 197) 
v_accvgpr_read v13, a3 // (op 197) 
v_mul_f32 v14, v7, v10 // (op 197) 
v_mul_f32 v15, v7, v11 // (op 197) 
v_mul_f32 v16, v7, v12 // (op 197) 
v_mul_f32 v17, v7, v13 // (op 197) 



v_mov_b32 v0, v9 // (op 199) reg expression

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 199) Store value


v_accvgpr_read v10, a4 // (op 943) 
v_accvgpr_read v11, a5 // (op 943) 
v_accvgpr_read v12, a6 // (op 943) 
v_accvgpr_read v13, a7 // (op 943) 
v_mul_f32 v14, v7, v10 // (op 943) 
v_mul_f32 v15, v7, v11 // (op 943) 
v_mul_f32 v16, v7, v12 // (op 943) 
v_mul_f32 v17, v7, v13 // (op 943) 



s_lshl_b32 s15, s24, 7 // (op 945) 
v_add_u32 v0, v9, s15 // (op 945) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 945) Store value


v_accvgpr_read v10, a8 // (op 955) 
v_accvgpr_read v11, a9 // (op 955) 
v_accvgpr_read v12, a10 // (op 955) 
v_accvgpr_read v13, a11 // (op 955) 
v_mul_f32 v14, v7, v10 // (op 955) 
v_mul_f32 v15, v7, v11 // (op 955) 
v_mul_f32 v16, v7, v12 // (op 955) 
v_mul_f32 v17, v7, v13 // (op 955) 



s_lshl_b32 s15, s24, 8 // (op 957) 
v_add_u32 v0, v9, s15 // (op 957) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 957) Store value


v_accvgpr_read v10, a12 // (op 967) 
v_accvgpr_read v11, a13 // (op 967) 
v_accvgpr_read v12, a14 // (op 967) 
v_accvgpr_read v13, a15 // (op 967) 
v_mul_f32 v14, v7, v10 // (op 967) 
v_mul_f32 v15, v7, v11 // (op 967) 
v_mul_f32 v16, v7, v12 // (op 967) 
v_mul_f32 v17, v7, v13 // (op 967) 



s_lshl_b32 s15, s24, 7 // (op 969) 
s_mul_i32 s6, 3, s15 // (op 969) 

v_add_u32 v0, v9, s6 // (op 969) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 969) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_189 // (op 189) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_189
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_189:


v_mov_b32 v0, v8 // (op 212) reg expression

buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 212) Load value

v_accvgpr_read v14, a0 // (op 209) 
v_accvgpr_read v15, a1 // (op 209) 
v_accvgpr_read v16, a2 // (op 209) 
v_accvgpr_read v17, a3 // (op 209) 
v_mul_f32 v18, v7, v14 // (op 209) 
v_mul_f32 v19, v7, v15 // (op 209) 
v_mul_f32 v20, v7, v16 // (op 209) 
v_mul_f32 v21, v7, v17 // (op 209) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v6, v10 // (op 213) 
v_mul_f32 v15, v6, v11 // (op 213) 
v_mul_f32 v16, v6, v12 // (op 213) 
v_mul_f32 v17, v6, v13 // (op 213) 

v_add_f32 v10, v14, v18 // (op 210) 
v_add_f32 v11, v15, v19 // (op 210) 
v_add_f32 v12, v16, v20 // (op 210) 
v_add_f32 v13, v17, v21 // (op 210) 




v_mov_b32 v0, v9 // (op 211) reg expression

buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 211) Store value




s_lshl_b32 s15, s4, 7 // (op 1001) 
v_add_u32 v0, v8, s15 // (op 1001) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 1001) Load value

v_accvgpr_read v14, a4 // (op 998) 
v_accvgpr_read v15, a5 // (op 998) 
v_accvgpr_read v16, a6 // (op 998) 
v_accvgpr_read v17, a7 // (op 998) 
v_mul_f32 v18, v7, v14 // (op 998) 
v_mul_f32 v19, v7, v15 // (op 998) 
v_mul_f32 v20, v7, v16 // (op 998) 
v_mul_f32 v21, v7, v17 // (op 998) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v6, v10 // (op 1002) 
v_mul_f32 v15, v6, v11 // (op 1002) 
v_mul_f32 v16, v6, v12 // (op 1002) 
v_mul_f32 v17, v6, v13 // (op 1002) 

v_add_f32 v10, v14, v18 // (op 999) 
v_add_f32 v11, v15, v19 // (op 999) 
v_add_f32 v12, v16, v20 // (op 999) 
v_add_f32 v13, v17, v21 // (op 999) 




s_lshl_b32 s15, s24, 7 // (op 1000) 
v_add_u32 v0, v9, s15 // (op 1000) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1000) Store value




s_lshl_b32 s15, s4, 8 // (op 1013) 
v_add_u32 v0, v8, s15 // (op 1013) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 1013) Load value

v_accvgpr_read v14, a8 // (op 1010) 
v_accvgpr_read v15, a9 // (op 1010) 
v_accvgpr_read v16, a10 // (op 1010) 
v_accvgpr_read v17, a11 // (op 1010) 
v_mul_f32 v18, v7, v14 // (op 1010) 
v_mul_f32 v19, v7, v15 // (op 1010) 
v_mul_f32 v20, v7, v16 // (op 1010) 
v_mul_f32 v21, v7, v17 // (op 1010) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v6, v10 // (op 1014) 
v_mul_f32 v15, v6, v11 // (op 1014) 
v_mul_f32 v16, v6, v12 // (op 1014) 
v_mul_f32 v17, v6, v13 // (op 1014) 

v_add_f32 v10, v14, v18 // (op 1011) 
v_add_f32 v11, v15, v19 // (op 1011) 
v_add_f32 v12, v16, v20 // (op 1011) 
v_add_f32 v13, v17, v21 // (op 1011) 




s_lshl_b32 s15, s24, 8 // (op 1012) 
v_add_u32 v0, v9, s15 // (op 1012) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1012) Store value




s_lshl_b32 s15, s4, 7 // (op 1025) 
s_mul_i32 s6, 3, s15 // (op 1025) 

v_add_u32 v0, v8, s6 // (op 1025) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 1025) Load value

v_accvgpr_read v14, a12 // (op 1022) 
v_accvgpr_read v15, a13 // (op 1022) 
v_accvgpr_read v16, a14 // (op 1022) 
v_accvgpr_read v17, a15 // (op 1022) 
v_mul_f32 v18, v7, v14 // (op 1022) 
v_mul_f32 v19, v7, v15 // (op 1022) 
v_mul_f32 v20, v7, v16 // (op 1022) 
v_mul_f32 v21, v7, v17 // (op 1022) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v6, v10 // (op 1026) 
v_mul_f32 v15, v6, v11 // (op 1026) 
v_mul_f32 v16, v6, v12 // (op 1026) 
v_mul_f32 v17, v6, v13 // (op 1026) 

v_add_f32 v10, v14, v18 // (op 1023) 
v_add_f32 v11, v15, v19 // (op 1023) 
v_add_f32 v12, v16, v20 // (op 1023) 
v_add_f32 v13, v17, v21 // (op 1023) 




s_lshl_b32 s15, s24, 7 // (op 1024) 
s_mul_i32 s6, 3, s15 // (op 1024) 

v_add_u32 v0, v9, s6 // (op 1024) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1024) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_189:
s_add_u32 s0, s0, s1 // (op 398) 


s_mul_i32 s15, s50, s16 // (op 396) 
s_cmp_gt_u32 s51, s15 // (op 396) 
s_mov_b32 s6, scc // (op 396) 
s_sub_u32 s7, s51, s15 // (op 396) 

s_cmp_eq_i32 s6, 0 // (op 396) 
s_cselect_b32 s8, 0, s7 // (op 396) 


s_cmp_lt_u32 s50, s8 // (op 396) 
s_mov_b32 s15, scc // (op 396) 
s_cmp_eq_i32 s15, 0 // (op 396) 
s_cselect_b32 s6, s8, s50 // (op 396) 


s_cmp_lt_u32 s0, s6 // (op 396) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_DPStreamTileLoop_396 // (op 396) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_DPStreamTileLoop_396 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_396:







































s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
.LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_end:
.size RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2, .LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_end-RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
  .amdhsa_next_free_vgpr 88
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 1024 // lds bytes
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
    DataType: F4
    DataTypeA: F4
    DataTypeB: F4
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
    macrotile: [32, 128, 128]
    threads: [256, 1, 1]
    grid: [RocRollerStreamK, One, One]
    occupancy: 7
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
  - .name: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
    .symbol: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 1024
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 86
    .vgpr_count: 70
    .agpr_count: 16
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_size_1_23
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

