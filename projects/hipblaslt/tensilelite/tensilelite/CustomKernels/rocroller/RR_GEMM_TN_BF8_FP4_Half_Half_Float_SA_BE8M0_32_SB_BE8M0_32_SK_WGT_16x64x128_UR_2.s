// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2,@function
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 2194) 
v_and_b32 v3, 1, v0 // (op 2194) 

v_and_b32 v0, 15, v1 // (op 2194) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 2194) 


v_and_b32 v0, 63, v1 // (op 2194) 
v_lshrrev_b32 v3, 4, v0 // (op 2194) 

v_lshl_add_u32 v0, v4, 2, v3 // (op 2194) 


v_lshrrev_b32 v3, 6, v1 // (op 2294) 
v_and_b32 v4, 1, v3 // (op 2294) 

v_and_b32 v3, 15, v1 // (op 2294) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 2294) 


v_and_b32 v3, 63, v1 // (op 2294) 
v_lshrrev_b32 v4, 4, v3 // (op 2294) 

v_lshl_add_u32 v3, v5, 2, v4 // (op 2294) 


v_lshrrev_b32 v4, 6, v1 // (op 2365) 
v_and_b32 v5, 1, v4 // (op 2365) 

v_and_b32 v4, 15, v1 // (op 2365) 
v_lshl_add_u32 v6, v5, 4, v4 // (op 2365) 


v_and_b32 v4, 63, v1 // (op 2365) 
v_lshrrev_b32 v5, 4, v4 // (op 2365) 

v_lshl_add_u32 v4, v6, 2, v5 // (op 2365) 


v_lshrrev_b32 v5, 6, v1 // (op 2394) 
v_and_b32 v6, 1, v5 // (op 2394) 

v_and_b32 v5, 15, v1 // (op 2394) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 2394) 


v_and_b32 v5, 63, v1 // (op 2394) 
v_lshrrev_b32 v6, 4, v5 // (op 2394) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 2394) 


v_mov_b32 v6, s6 // (op 2) Move value
v_mov_b32 v7, s7 // (op 31) Move value


s_mov_b32 s0, 0 // (op 350) call()
s_mov_b32 s1, 0 // (op 351) call()
s_mov_b32 s6, 0 // (op 352) call()
s_mov_b32 s7, 0 // (op 353) call()
s_mov_b32 s18, 0 // (op 354) call()
s_mov_b32 s19, 0 // (op 355) call()
s_mov_b32 s0, 0 // (op 250) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 249) 
s_cmp_gt_u32 s9, s52 // (op 249) 
s_mov_b32 s53, scc // (op 249) 
s_sub_u32 s54, s9, s52 // (op 249) 

s_cmp_eq_i32 s53, 0 // (op 249) 
s_cselect_b32 s52, 0, s54 // (op 249) 


s_cmp_lt_u32 s8, s52 // (op 249) 
s_mov_b32 s53, scc // (op 249) 
s_cmp_eq_i32 s53, 0 // (op 249) 
s_cselect_b32 s54, s52, s8 // (op 249) 


s_cmp_lt_u32 s0, s54 // (op 249) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_249 // (op 249) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_249 if false)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_SKStreamTileLoop_249:
v_lshrrev_b32 v8, 7, v1 // (op 2109) 
s_mul_i32 s52, s16, s8 // (op 2109) 
s_add_u32 s52, s52, s0 // (op 2109) 
v_and_b32 v9, 63, v1 // (op 2109) 
s_mul_hi_u32 s53, s52, s13 // (op 2109) 
v_lshrrev_b32 v10, 4, v9 // (op 2109) 
s_sub_u32 s54, s52, s53 // (op 2109) 
v_lshlrev_b32 v11, 2, v10 // (op 2109) 

v_lshl_add_u32 v10, v8, 4, v11 // (op 2109) 


s_lshr_b32 s55, s54, 1 // (op 2109) 

s_add_u32 s54, s55, s53 // (op 2109) 


v_lshrrev_b32 v8, 2, v9 // (op 2109) 

s_lshr_b32 s53, s54, s11 // (op 2109) 

v_and_b32 v9, 3, v8 // (op 2109) 

s_and_b32 s54, s11, 2147483648 // (op 2109) 
v_and_b32 v8, 3, v1 // (op 2109) 
s_cmp_eq_i32 s54, 0 // (op 2109) 
s_mov_b32 s55, scc // (op 2109) 

v_lshl_add_u32 v11, v9, 2, v8 // (op 2109) 


s_cmp_eq_i32 s55, 0 // (op 2109) 
s_cselect_b32 s54, s52, s53 // (op 2109) 



v_lshrrev_b32 v8, 6, v1 // (op 2109) 
s_mul_hi_u32 s52, s54, s12 // (op 2109) 
v_and_b32 v9, 1, v8 // (op 2109) 

s_sub_u32 s53, s54, s52 // (op 2109) 
v_lshl_add_u32 v8, v9, 4, v11 // (op 2109) 


s_lshr_b32 s55, s53, 1 // (op 2109) 

s_add_u32 s53, s55, s52 // (op 2109) 


s_lshr_b32 s52, s53, s10 // (op 2109) 

s_and_b32 s53, s10, 2147483648 // (op 2109) 
s_cmp_eq_i32 s53, 0 // (op 2109) 
s_mov_b32 s55, scc // (op 2109) 

s_cmp_eq_i32 s55, 0 // (op 2109) 
s_cselect_b32 s53, s54, s52 // (op 2109) 


s_mul_i32 s52, s53, s14 // (op 2109) 
v_lshl_add_u32 v9, s53, 4, v10 // (op 2109) 


s_sub_u32 s53, s54, s52 // (op 2109) 


v_lshl_add_u32 v10, s53, 6, v8 // (op 2109) 


v_mul_lo_u32 v8, v10, s4 // (op 2109) 

v_add_lshl_u32 v10, v9, v8, 1 // (op 2109) 


s_mul_i32 s53, s4, s23 // (op 2111) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 2111) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 2111) least significant
s_mul_hi_u32 s55, s4, s22 // (op 2111) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2111) most significant: sum
s_add_u32 s53, s53, s55 // (op 2111) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 2111) 

s_mov_b32 s56, s20 // (op 2111) reg expression
s_mov_b32 s57, s21 // (op 2111) reg expression
s_mov_b32 s58, s54 // (op 2111) reg expression
s_mov_b32 s59, 131072 // (op 2111) call()

v_lshrrev_b32 v8, 7, v1 // (op 2211) 
s_mul_i32 s52, s16, s8 // (op 2211) 
s_add_u32 s52, s52, s0 // (op 2211) 
v_and_b32 v9, 63, v1 // (op 2211) 
s_mul_hi_u32 s53, s52, s13 // (op 2211) 
v_lshrrev_b32 v11, 4, v9 // (op 2211) 
s_sub_u32 s54, s52, s53 // (op 2211) 
v_lshlrev_b32 v12, 2, v11 // (op 2211) 

s_lshr_b32 s55, s54, 1 // (op 2211) 

v_lshl_add_u32 v11, v8, 4, v12 // (op 2211) 


s_add_u32 s54, s55, s53 // (op 2211) 


v_lshrrev_b32 v8, 2, v9 // (op 2211) 

s_lshr_b32 s53, s54, s11 // (op 2211) 

v_and_b32 v9, 3, v8 // (op 2211) 

s_and_b32 s54, s11, 2147483648 // (op 2211) 
v_and_b32 v8, 3, v1 // (op 2211) 
s_cmp_eq_i32 s54, 0 // (op 2211) 
s_mov_b32 s55, scc // (op 2211) 

v_lshl_add_u32 v12, v9, 2, v8 // (op 2211) 


s_cmp_eq_i32 s55, 0 // (op 2211) 
s_cselect_b32 s54, s52, s53 // (op 2211) 



v_lshrrev_b32 v8, 6, v1 // (op 2211) 
s_mul_hi_u32 s55, s54, s12 // (op 2211) 
v_and_b32 v9, 1, v8 // (op 2211) 

s_sub_u32 s52, s54, s55 // (op 2211) 
v_lshl_add_u32 v8, v9, 4, v12 // (op 2211) 


s_lshr_b32 s53, s52, 1 // (op 2211) 

s_add_u32 s52, s53, s55 // (op 2211) 


s_lshr_b32 s53, s52, s10 // (op 2211) 

s_and_b32 s52, s10, 2147483648 // (op 2211) 
s_cmp_eq_i32 s52, 0 // (op 2211) 
s_mov_b32 s55, scc // (op 2211) 

s_cmp_eq_i32 s55, 0 // (op 2211) 
s_cselect_b32 s52, s54, s53 // (op 2211) 


s_mul_i32 s53, s52, s14 // (op 2211) 
v_lshl_add_u32 v9, s52, 4, v11 // (op 2211) 


s_sub_u32 s52, s54, s53 // (op 2211) 


v_lshl_add_u32 v11, s52, 6, v8 // (op 2211) 


v_mul_lo_u32 v8, v11, s24 // (op 2211) 

v_add_lshl_u32 v11, v9, v8, 1 // (op 2211) 


s_mul_i32 s53, s24, s23 // (op 2213) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 2213) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 2213) least significant
s_mul_hi_u32 s55, s24, s22 // (op 2213) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2213) most significant: sum
s_add_u32 s53, s53, s55 // (op 2213) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 2213) 

s_mov_b32 s60, s26 // (op 2213) reg expression
s_mov_b32 s61, s27 // (op 2213) reg expression
s_mov_b32 s62, s54 // (op 2213) reg expression
s_mov_b32 s63, 131072 // (op 2213) call()

v_lshrrev_b32 v8, 5, v1 // (op 2311) 
v_lshlrev_b32 v9, 2, v8 // (op 2311) 

v_lshl_add_u32 v8, s16, 4, v9 // (op 2311) 

v_and_b32 v9, -32, v1 // (op 2311) 
v_sub_u32 v12, v1, v9 // (op 2311) 

v_lshl_add_u32 v9, v8, 6, v12 // (op 2311) 


v_lshlrev_b32 v8, 2, v9 // (op 2311) 

s_lshl_b32 s52, s15, 12 // (op 2313) 
s_mov_b32 s64, s28 // (op 2313) reg expression
s_mov_b32 s65, s29 // (op 2313) reg expression
s_mov_b32 s66, s52 // (op 2313) reg expression
s_mov_b32 s67, 131072 // (op 2313) call()

s_mul_i32 s6, s8, s16 // (op 254) 
s_add_u32 s6, s6, s0 // (op 254) 
s_mul_hi_u32 s52, s6, s13 // (op 255) 
s_sub_u32 s53, s6, s52 // (op 255) 
s_lshr_b32 s54, s53, 1 // (op 255) 

s_add_u32 s53, s54, s52 // (op 255) 


s_lshr_b32 s52, s53, s11 // (op 255) 

s_and_b32 s53, s11, 2147483648 // (op 255) 
s_cmp_eq_i32 s53, 0 // (op 255) 
s_mov_b32 s54, scc // (op 255) 

s_cmp_eq_i32 s54, 0 // (op 255) 
s_cselect_b32 s53, s6, s52 // (op 255) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 255) 
s_mul_i32 s52, s53, s54 // (op 255) 


s_sub_u32 s18, s6, s52 // (op 255) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 256) 
s_cmp_gt_u32 s52, s18 // (op 256) 
s_mov_b32 s54, scc // (op 256) 
s_sub_u32 s55, s52, s18 // (op 256) 

s_cmp_eq_i32 s54, 0 // (op 256) 
s_cselect_b32 s52, 0, s55 // (op 256) 


s_cmp_gt_u32 s8, s0 // (op 256) 
s_mov_b32 s53, scc // (op 256) 
s_sub_u32 s54, s8, s0 // (op 256) 
s_cmp_eq_i32 s53, 0 // (op 256) 
s_cselect_b32 s55, 0, s54 // (op 256) 


s_cmp_lt_u32 s52, s55 // (op 256) 
s_mov_b32 s53, scc // (op 256) 
s_cmp_eq_i32 s53, 0 // (op 256) 
s_cselect_b32 s54, s55, s52 // (op 256) 



s_cmp_gt_u32 s9, s6 // (op 256) 
s_mov_b32 s55, scc // (op 256) 
s_sub_u32 s52, s9, s6 // (op 256) 
s_cmp_eq_i32 s55, 0 // (op 256) 
s_cselect_b32 s53, 0, s52 // (op 256) 


s_cmp_lt_u32 s54, s53 // (op 256) 
s_mov_b32 s52, scc // (op 256) 
s_cmp_eq_i32 s52, 0 // (op 256) 
s_cselect_b32 s7, s53, s54 // (op 256) 



s_add_u32 s52, s18, s7 // (op 257) 
s_sub_u32 s19, s52, 1 // (op 257) 

v_accvgpr_write a0, 0.00000 // (op 53) call()
v_accvgpr_write a1, 0.00000 // (op 53) call()
v_accvgpr_write a2, 0.00000 // (op 53) call()
v_accvgpr_write a3, 0.00000 // (op 53) call()
v_accvgpr_write a4, 0.00000 // (op 834) call()
v_accvgpr_write a5, 0.00000 // (op 834) call()
v_accvgpr_write a6, 0.00000 // (op 834) call()
v_accvgpr_write a7, 0.00000 // (op 834) call()
v_lshrrev_b32 v9, 7, v1 // (op 1651) 
s_mul_i32 s52, s16, s8 // (op 1651) 
s_add_u32 s52, s52, s0 // (op 1651) 
v_and_b32 v12, 15, v1 // (op 1651) 
s_mul_hi_u32 s53, s52, s13 // (op 1651) 
v_lshl_add_u32 v13, v9, 4, v12 // (op 1651) 


s_sub_u32 s54, s52, s53 // (op 1651) 
v_and_b32 v9, 48, v1 // (op 1651) 
s_lshr_b32 s55, s54, 1 // (op 1651) 

s_add_u32 s54, s55, s53 // (op 1651) 


s_lshr_b32 s53, s54, s11 // (op 1651) 

s_and_b32 s54, s11, 2147483648 // (op 1651) 
s_cmp_eq_i32 s54, 0 // (op 1651) 
s_mov_b32 s55, scc // (op 1651) 

s_cmp_eq_i32 s55, 0 // (op 1651) 
s_cselect_b32 s54, s52, s53 // (op 1651) 


s_mul_hi_u32 s53, s54, s12 // (op 1651) 
s_sub_u32 s55, s54, s53 // (op 1651) 
s_lshr_b32 s68, s55, 1 // (op 1651) 

s_add_u32 s55, s68, s53 // (op 1651) 


s_lshr_b32 s53, s55, s10 // (op 1651) 

s_and_b32 s55, s10, 2147483648 // (op 1651) 
s_cmp_eq_i32 s55, 0 // (op 1651) 
s_mov_b32 s68, scc // (op 1651) 

s_cmp_eq_i32 s68, 0 // (op 1651) 
s_cselect_b32 s55, s54, s53 // (op 1651) 


v_lshl_add_u32 v12, s55, 4, v13 // (op 1651) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 1651) 
s_mul_i32 s53, s54, s68 // (op 1651) 


s_sub_u32 s54, s52, s53 // (op 1651) 


v_lshl_add_u32 v13, s54, 7, v9 // (op 1651) 


v_mul_lo_u32 v9, v12, s30 // (op 1651) 
v_add_u32 v9, v9, v13 // (op 1651) 


s_mul_i32 s53, s30, s35 // (op 1653) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 1653) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 1653) least significant
s_mul_hi_u32 s55, s30, s34 // (op 1653) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1653) most significant: sum
s_add_u32 s53, s53, s55 // (op 1653) most significant: sum


s_mov_b32 s68, s32 // (op 1653) reg expression
s_mov_b32 s69, s33 // (op 1653) reg expression
s_mov_b32 s70, s52 // (op 1653) reg expression
s_mov_b32 s71, 131072 // (op 1653) call()

v_lshrrev_b32 v12, 7, v1 // (op 1670) 
s_mul_i32 s52, s16, s8 // (op 1670) 
s_add_u32 s52, s52, s0 // (op 1670) 
v_and_b32 v13, 15, v1 // (op 1670) 
s_mul_hi_u32 s53, s52, s13 // (op 1670) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 1670) 


s_sub_u32 s54, s52, s53 // (op 1670) 
v_and_b32 v12, 48, v1 // (op 1670) 
s_lshr_b32 s55, s54, 1 // (op 1670) 

s_add_u32 s54, s55, s53 // (op 1670) 


s_lshr_b32 s53, s54, s11 // (op 1670) 

s_and_b32 s54, s11, 2147483648 // (op 1670) 
s_cmp_eq_i32 s54, 0 // (op 1670) 
s_mov_b32 s55, scc // (op 1670) 

s_cmp_eq_i32 s55, 0 // (op 1670) 
s_cselect_b32 s54, s52, s53 // (op 1670) 


s_mul_hi_u32 s53, s54, s12 // (op 1670) 
s_sub_u32 s55, s54, s53 // (op 1670) 
s_lshr_b32 s72, s55, 1 // (op 1670) 

s_add_u32 s55, s72, s53 // (op 1670) 


s_lshr_b32 s53, s55, s10 // (op 1670) 

s_and_b32 s55, s10, 2147483648 // (op 1670) 
s_cmp_eq_i32 s55, 0 // (op 1670) 
s_mov_b32 s72, scc // (op 1670) 

s_cmp_eq_i32 s72, 0 // (op 1670) 
s_cselect_b32 s55, s54, s53 // (op 1670) 


v_lshl_add_u32 v13, s55, 4, v14 // (op 1670) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 1670) 
s_mul_i32 s53, s54, s72 // (op 1670) 


s_sub_u32 s54, s52, s53 // (op 1670) 


v_lshl_add_u32 v14, s54, 7, v12 // (op 1670) 


v_mul_lo_u32 v12, v13, s30 // (op 1670) 
v_add_u32 v12, v12, v14 // (op 1670) 


s_mul_i32 s53, s30, s35 // (op 1672) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 1672) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 1672) least significant
s_mul_hi_u32 s55, s30, s34 // (op 1672) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1672) most significant: sum
s_add_u32 s53, s53, s55 // (op 1672) most significant: sum


s_mov_b32 s68, s32 // (op 1672) reg expression
s_mov_b32 s69, s33 // (op 1672) reg expression
s_mov_b32 s70, s52 // (op 1672) reg expression
s_mov_b32 s71, 131072 // (op 1672) call()

v_lshrrev_b32 v13, 6, v1 // (op 1804) 
s_mul_i32 s52, s16, s8 // (op 1804) 
s_add_u32 s52, s52, s0 // (op 1804) 
v_and_b32 v14, 1, v13 // (op 1804) 

s_mul_hi_u32 s53, s52, s13 // (op 1804) 
v_and_b32 v13, 15, v1 // (op 1804) 
s_sub_u32 s54, s52, s53 // (op 1804) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 1804) 


s_lshr_b32 s55, s54, 1 // (op 1804) 

v_and_b32 v13, 63, v1 // (op 1804) 
s_add_u32 s54, s55, s53 // (op 1804) 


v_lshrrev_b32 v14, 4, v13 // (op 1804) 

s_lshr_b32 s53, s54, s11 // (op 1804) 

v_lshlrev_b32 v13, 5, v14 // (op 1804) 

s_and_b32 s54, s11, 2147483648 // (op 1804) 
s_cmp_eq_i32 s54, 0 // (op 1804) 
s_mov_b32 s55, scc // (op 1804) 

s_cmp_eq_i32 s55, 0 // (op 1804) 
s_cselect_b32 s54, s52, s53 // (op 1804) 


s_mul_hi_u32 s53, s54, s12 // (op 1804) 
s_sub_u32 s55, s54, s53 // (op 1804) 
s_lshr_b32 s72, s55, 1 // (op 1804) 

s_add_u32 s55, s72, s53 // (op 1804) 


s_lshr_b32 s53, s55, s10 // (op 1804) 

s_and_b32 s55, s10, 2147483648 // (op 1804) 
s_cmp_eq_i32 s55, 0 // (op 1804) 
s_mov_b32 s72, scc // (op 1804) 

s_cmp_eq_i32 s72, 0 // (op 1804) 
s_cselect_b32 s55, s54, s53 // (op 1804) 


s_mul_i32 s53, s55, s14 // (op 1804) 

s_sub_u32 s55, s54, s53 // (op 1804) 

v_lshl_add_u32 v14, s55, 6, v15 // (op 1804) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 1804) 
v_mul_lo_u32 v17, v14, s37 // (op 1804) most significant: low of low * high
v_mul_hi_u32 v15, v14, s36 // (op 1804) most significant: high of low * low
v_mul_lo_u32 v16, v14, s36 // (op 1804) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 1804) most significant: sum


s_mul_i32 s53, s54, s72 // (op 1804) 


s_sub_u32 s54, s52, s53 // (op 1804) 


v_lshl_add_u32 v14, s54, 7, v13 // (op 1804) 


v_mov_b32 v19, 0 // (op 1804) convert
v_mov_b32 v18, v14 // (op 1804) convert

v_add_co_u32 v14, s[52:53], v18, v16 // (op 1804) least significant half
v_addc_co_u32 v15, s[52:53], v19, v17, s[52:53] // (op 1804) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 1804) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 1804) 

v_mov_b32 v13, v16 // (op 1804) convert

s_mul_i32 s53, s36, s23 // (op 1806) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 1806) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 1806) least significant
s_mul_hi_u32 s55, s36, s22 // (op 1806) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1806) most significant: sum
s_add_u32 s53, s53, s55 // (op 1806) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1806) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1806) 

s_mov_b32 s72, s38 // (op 1806) reg expression
s_mov_b32 s73, s39 // (op 1806) reg expression
s_mov_b32 s74, s52 // (op 1806) reg expression
s_mov_b32 s75, 131072 // (op 1806) call()

v_lshrrev_b32 v14, 6, v1 // (op 1825) 
s_mul_i32 s52, s16, s8 // (op 1825) 
s_add_u32 s52, s52, s0 // (op 1825) 
v_and_b32 v15, 1, v14 // (op 1825) 

s_mul_hi_u32 s53, s52, s13 // (op 1825) 
v_and_b32 v14, 15, v1 // (op 1825) 
s_sub_u32 s54, s52, s53 // (op 1825) 
v_lshl_add_u32 v16, v15, 4, v14 // (op 1825) 


s_lshr_b32 s55, s54, 1 // (op 1825) 

v_and_b32 v14, 63, v1 // (op 1825) 
s_add_u32 s54, s55, s53 // (op 1825) 


v_lshrrev_b32 v15, 4, v14 // (op 1825) 

s_lshr_b32 s53, s54, s11 // (op 1825) 

v_lshlrev_b32 v14, 5, v15 // (op 1825) 

s_and_b32 s54, s11, 2147483648 // (op 1825) 
s_cmp_eq_i32 s54, 0 // (op 1825) 
s_mov_b32 s55, scc // (op 1825) 

s_cmp_eq_i32 s55, 0 // (op 1825) 
s_cselect_b32 s54, s52, s53 // (op 1825) 


s_mul_hi_u32 s53, s54, s12 // (op 1825) 
s_sub_u32 s55, s54, s53 // (op 1825) 
s_lshr_b32 s76, s55, 1 // (op 1825) 

s_add_u32 s55, s76, s53 // (op 1825) 


s_lshr_b32 s53, s55, s10 // (op 1825) 

s_and_b32 s55, s10, 2147483648 // (op 1825) 
s_cmp_eq_i32 s55, 0 // (op 1825) 
s_mov_b32 s76, scc // (op 1825) 

s_cmp_eq_i32 s76, 0 // (op 1825) 
s_cselect_b32 s55, s54, s53 // (op 1825) 


s_mul_i32 s53, s55, s14 // (op 1825) 

s_sub_u32 s55, s54, s53 // (op 1825) 

v_lshl_add_u32 v15, s55, 6, v16 // (op 1825) 


v_mul_lo_u32 v17, v15, s37 // (op 1825) most significant: low of low * high
v_mul_hi_u32 v18, v15, s36 // (op 1825) most significant: high of low * low
v_mul_lo_u32 v16, v15, s36 // (op 1825) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 1825) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1825) 
s_mul_i32 s53, s54, s76 // (op 1825) 


s_sub_u32 s54, s52, s53 // (op 1825) 


v_lshl_add_u32 v15, s54, 7, v14 // (op 1825) 


v_mov_b32 v19, 0 // (op 1825) convert
v_mov_b32 v18, v15 // (op 1825) convert

v_add_co_u32 v14, s[52:53], v18, v16 // (op 1825) least significant half
v_addc_co_u32 v15, s[52:53], v19, v17, s[52:53] // (op 1825) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 1825) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 1825) 

v_mov_b32 v14, v16 // (op 1825) convert

s_mul_i32 s53, s36, s23 // (op 1827) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 1827) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 1827) least significant
s_mul_hi_u32 s55, s36, s22 // (op 1827) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1827) most significant: sum
s_add_u32 s53, s53, s55 // (op 1827) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1827) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1827) 

s_mov_b32 s72, s38 // (op 1827) reg expression
s_mov_b32 s73, s39 // (op 1827) reg expression
s_mov_b32 s74, s52 // (op 1827) reg expression
s_mov_b32 s75, 131072 // (op 1827) call()

v_lshrrev_b32 v15, 7, v1 // (op 1937) 
s_mul_i32 s52, s16, s8 // (op 1937) 
s_add_u32 s52, s52, s0 // (op 1937) 
v_and_b32 v16, 15, v1 // (op 1937) 
s_mul_hi_u32 s53, s52, s13 // (op 1937) 
v_lshl_add_u32 v17, v15, 4, v16 // (op 1937) 


s_sub_u32 s54, s52, s53 // (op 1937) 
v_and_b32 v15, 63, v1 // (op 1937) 
s_lshr_b32 s55, s54, 1 // (op 1937) 

v_lshrrev_b32 v16, 4, v15 // (op 1937) 

s_add_u32 s54, s55, s53 // (op 1937) 


s_lshr_b32 s53, s54, s11 // (op 1937) 

s_and_b32 s54, s11, 2147483648 // (op 1937) 
s_cmp_eq_i32 s54, 0 // (op 1937) 
s_mov_b32 s55, scc // (op 1937) 

s_cmp_eq_i32 s55, 0 // (op 1937) 
s_cselect_b32 s54, s52, s53 // (op 1937) 


s_mul_hi_u32 s53, s54, s12 // (op 1937) 
s_sub_u32 s55, s54, s53 // (op 1937) 
s_lshr_b32 s76, s55, 1 // (op 1937) 

s_add_u32 s55, s76, s53 // (op 1937) 


s_lshr_b32 s53, s55, s10 // (op 1937) 

s_and_b32 s55, s10, 2147483648 // (op 1937) 
s_cmp_eq_i32 s55, 0 // (op 1937) 
s_mov_b32 s76, scc // (op 1937) 

s_cmp_eq_i32 s76, 0 // (op 1937) 
s_cselect_b32 s55, s54, s53 // (op 1937) 


v_lshl_add_u32 v15, s55, 4, v17 // (op 1937) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1937) 
s_mul_i32 s53, s54, s76 // (op 1937) 


s_sub_u32 s54, s52, s53 // (op 1937) 


v_lshl_add_u32 v17, s54, 2, v16 // (op 1937) 


v_mul_lo_u32 v16, v15, s40 // (op 1937) 
v_add_u32 v16, v16, v17 // (op 1937) 


s_mul_i32 s53, s40, s35 // (op 1939) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 1939) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 1939) least significant
s_mul_hi_u32 s55, s40, s34 // (op 1939) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1939) most significant: sum
s_add_u32 s53, s53, s55 // (op 1939) most significant: sum


s_mov_b32 s76, s42 // (op 1939) reg expression
s_mov_b32 s77, s43 // (op 1939) reg expression
s_mov_b32 s78, s52 // (op 1939) reg expression
s_mov_b32 s79, 131072 // (op 1939) call()

v_lshrrev_b32 v15, 7, v1 // (op 1956) 
s_mul_i32 s52, s16, s8 // (op 1956) 
s_add_u32 s52, s52, s0 // (op 1956) 
v_and_b32 v17, 15, v1 // (op 1956) 
s_mul_hi_u32 s53, s52, s13 // (op 1956) 
v_lshl_add_u32 v18, v15, 4, v17 // (op 1956) 


s_sub_u32 s54, s52, s53 // (op 1956) 
v_and_b32 v15, 63, v1 // (op 1956) 
s_lshr_b32 s55, s54, 1 // (op 1956) 

v_lshrrev_b32 v17, 4, v15 // (op 1956) 

s_add_u32 s54, s55, s53 // (op 1956) 


s_lshr_b32 s53, s54, s11 // (op 1956) 

s_and_b32 s54, s11, 2147483648 // (op 1956) 
s_cmp_eq_i32 s54, 0 // (op 1956) 
s_mov_b32 s55, scc // (op 1956) 

s_cmp_eq_i32 s55, 0 // (op 1956) 
s_cselect_b32 s54, s52, s53 // (op 1956) 


s_mul_hi_u32 s53, s54, s12 // (op 1956) 
s_sub_u32 s55, s54, s53 // (op 1956) 
s_lshr_b32 s80, s55, 1 // (op 1956) 

s_add_u32 s55, s80, s53 // (op 1956) 


s_lshr_b32 s53, s55, s10 // (op 1956) 

s_and_b32 s55, s10, 2147483648 // (op 1956) 
s_cmp_eq_i32 s55, 0 // (op 1956) 
s_mov_b32 s80, scc // (op 1956) 

s_cmp_eq_i32 s80, 0 // (op 1956) 
s_cselect_b32 s55, s54, s53 // (op 1956) 


v_lshl_add_u32 v15, s55, 4, v18 // (op 1956) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 1956) 
s_mul_i32 s53, s54, s80 // (op 1956) 


s_sub_u32 s54, s52, s53 // (op 1956) 


v_lshl_add_u32 v18, s54, 2, v17 // (op 1956) 


v_mul_lo_u32 v17, v15, s40 // (op 1956) 
v_add_u32 v17, v17, v18 // (op 1956) 


s_mul_i32 s53, s40, s35 // (op 1958) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 1958) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 1958) least significant
s_mul_hi_u32 s55, s40, s34 // (op 1958) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1958) most significant: sum
s_add_u32 s53, s53, s55 // (op 1958) most significant: sum


s_mov_b32 s76, s42 // (op 1958) reg expression
s_mov_b32 s77, s43 // (op 1958) reg expression
s_mov_b32 s78, s52 // (op 1958) reg expression
s_mov_b32 s79, 131072 // (op 1958) call()

v_lshrrev_b32 v15, 2, v1 // (op 2065) 
s_mul_i32 s52, s16, s8 // (op 2065) 
s_add_u32 s52, s52, s0 // (op 2065) 
v_lshlrev_b32 v18, 1, v15 // (op 2065) 

s_mul_hi_u32 s53, s52, s13 // (op 2065) 
v_and_b32 v15, -4, v1 // (op 2065) 
s_sub_u32 s54, s52, s53 // (op 2065) 
v_sub_u32 v19, v1, v15 // (op 2065) 

s_lshr_b32 s55, s54, 1 // (op 2065) 

s_add_u32 s54, s55, s53 // (op 2065) 


s_lshr_b32 s53, s54, s11 // (op 2065) 

s_and_b32 s54, s11, 2147483648 // (op 2065) 
s_cmp_eq_i32 s54, 0 // (op 2065) 
s_mov_b32 s55, scc // (op 2065) 

s_cmp_eq_i32 s55, 0 // (op 2065) 
s_cselect_b32 s54, s52, s53 // (op 2065) 


s_mul_hi_u32 s53, s54, s12 // (op 2065) 
s_sub_u32 s55, s54, s53 // (op 2065) 
s_lshr_b32 s80, s55, 1 // (op 2065) 

s_add_u32 s55, s80, s53 // (op 2065) 


s_lshr_b32 s53, s55, s10 // (op 2065) 

s_and_b32 s55, s10, 2147483648 // (op 2065) 
s_cmp_eq_i32 s55, 0 // (op 2065) 
s_mov_b32 s80, scc // (op 2065) 

s_cmp_eq_i32 s80, 0 // (op 2065) 
s_cselect_b32 s55, s54, s53 // (op 2065) 


s_mul_i32 s53, s55, s14 // (op 2065) 

s_sub_u32 s55, s54, s53 // (op 2065) 

v_lshl_add_u32 v15, s55, 6, v18 // (op 2065) 


v_mul_lo_u32 v18, v15, s44 // (op 2065) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 2065) 
s_mul_i32 s53, s54, s80 // (op 2065) 


s_sub_u32 s54, s52, s53 // (op 2065) 


v_lshl_add_u32 v15, s54, 2, v19 // (op 2065) 


v_add_u32 v19, v15, v18 // (op 2065) 


s_mul_i32 s53, s44, s23 // (op 2067) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 2067) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 2067) least significant
s_mul_hi_u32 s55, s44, s22 // (op 2067) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2067) most significant: sum
s_add_u32 s53, s53, s55 // (op 2067) most significant: sum


s_mov_b32 s80, s46 // (op 2067) reg expression
s_mov_b32 s81, s47 // (op 2067) reg expression
s_mov_b32 s82, s52 // (op 2067) reg expression
s_mov_b32 s83, 131072 // (op 2067) call()

v_lshrrev_b32 v15, 2, v1 // (op 2084) 
s_mul_i32 s52, s16, s8 // (op 2084) 
s_add_u32 s52, s52, s0 // (op 2084) 
v_lshlrev_b32 v18, 1, v15 // (op 2084) 

s_mul_hi_u32 s53, s52, s13 // (op 2084) 
v_and_b32 v15, -4, v1 // (op 2084) 
s_sub_u32 s54, s52, s53 // (op 2084) 
v_sub_u32 v20, v1, v15 // (op 2084) 

s_lshr_b32 s55, s54, 1 // (op 2084) 

s_add_u32 s54, s55, s53 // (op 2084) 


s_lshr_b32 s53, s54, s11 // (op 2084) 

s_and_b32 s54, s11, 2147483648 // (op 2084) 
s_cmp_eq_i32 s54, 0 // (op 2084) 
s_mov_b32 s55, scc // (op 2084) 

s_cmp_eq_i32 s55, 0 // (op 2084) 
s_cselect_b32 s54, s52, s53 // (op 2084) 


s_mul_hi_u32 s53, s54, s12 // (op 2084) 
s_sub_u32 s55, s54, s53 // (op 2084) 
s_lshr_b32 s84, s55, 1 // (op 2084) 

s_add_u32 s55, s84, s53 // (op 2084) 


s_lshr_b32 s53, s55, s10 // (op 2084) 

s_and_b32 s55, s10, 2147483648 // (op 2084) 
s_cmp_eq_i32 s55, 0 // (op 2084) 
s_mov_b32 s84, scc // (op 2084) 

s_cmp_eq_i32 s84, 0 // (op 2084) 
s_cselect_b32 s55, s54, s53 // (op 2084) 


s_mul_i32 s53, s55, s14 // (op 2084) 

s_sub_u32 s55, s54, s53 // (op 2084) 

v_lshl_add_u32 v15, s55, 6, v18 // (op 2084) 


v_mul_lo_u32 v18, v15, s44 // (op 2084) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 2084) 
s_mul_i32 s53, s54, s84 // (op 2084) 


s_sub_u32 s54, s52, s53 // (op 2084) 


v_lshl_add_u32 v15, s54, 2, v20 // (op 2084) 


v_add_u32 v20, v15, v18 // (op 2084) 


s_mul_i32 s53, s44, s23 // (op 2086) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 2086) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 2086) least significant
s_mul_hi_u32 s55, s44, s22 // (op 2086) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2086) most significant: sum
s_add_u32 s53, s53, s55 // (op 2086) most significant: sum


s_mov_b32 s80, s46 // (op 2086) reg expression
s_mov_b32 s81, s47 // (op 2086) reg expression
s_mov_b32 s82, s52 // (op 2086) reg expression
s_mov_b32 s83, 131072 // (op 2086) call()




buffer_load_dwordx4 v[22:25], v9, s[68:71], 0 offen // (op 1444) Load value
buffer_load_dwordx4 v[26:29], v9, s[68:71], 0 offen offset:64 // (op 1444) Load value


v_mov_b32 v15, v13 // (op 1450) reg expression

buffer_load_dwordx4 v[30:33], v15, s[72:75], 0 offen // (op 1450) Load value




buffer_load_ubyte v15, v16, s[76:79], 0 offen offset: 0 // (op 1456) Load value


s_lshl_b64 s[52:53], s[36:37], 5 // (op 1462) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1462) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1462) 

v_add_u32 v18, v13, s52 // (op 1462) 


buffer_load_dwordx4 v[34:37], v18, s[72:75], 0 offen // (op 1462) Load value


v_mov_b64 v[38:39], s[44:45] // (op 1467) convert

v_mov_b32 v18, v19 // (op 1467) 
buffer_load_ubyte v40, v18, s[80:83], 0 offen offset: 0 // (op 1467) Load value
v_add_u32 v18, v18, v38 // (op 1467) 
buffer_load_ubyte v41, v18, s[80:83], 0 offen offset: 0 // (op 1467) Load value





buffer_load_dwordx4 v[42:45], v9, s[68:71], 0 offen offset:128 // (op 1471) Load value
buffer_load_dwordx4 v[46:49], v9, s[68:71], 0 offen offset:192 // (op 1471) Load value



v_add_u32 v9, 64, v13 // (op 1477) 

buffer_load_dwordx4 v[50:53], v9, s[72:75], 0 offen // (op 1477) Load value




buffer_load_ubyte v9, v16, s[76:79], 0 offen offset:4 // (op 1483) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 1489) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1489) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1489) 

s_add_u32 s54, s52, 64 // (op 1489) 

v_add_u32 v16, v13, s54 // (op 1489) 


buffer_load_dwordx4 v[54:57], v16, s[72:75], 0 offen // (op 1489) Load value



v_mov_b64 v[38:39], s[44:45] // (op 1494) convert

v_mov_b32 v13, v19 // (op 1494) 
buffer_load_ubyte v58, v13, s[80:83], 0 offen offset:4 // (op 1494) Load value
v_add_u32 v13, v13, v38 // (op 1494) 
buffer_load_ubyte v59, v13, s[80:83], 0 offen offset:4 // (op 1494) Load value



v_and_b32 v13, -4, v1 // (op 2177) 
v_sub_u32 v16, v1, v13 // (op 2177) 

v_lshrrev_b32 v13, 2, v1 // (op 2177) 
v_lshl_add_u32 v21, v13, 3, v16 // (op 2177) 


s_barrier  // (op 2465) 



s_waitcnt vmcnt(8)
ds_write_b8 v21, v40 // (op 1496) Store local data 
s_waitcnt vmcnt(7)
ds_write_b8 v21, v41 offset:4 // (op 1496) Store local data 

v_and_b32 v13, -4, v1 // (op 2148) 
v_sub_u32 v16, v1, v13 // (op 2148) 

v_lshrrev_b32 v13, 2, v1 // (op 2148) 
v_lshl_add_u32 v21, v13, 3, v16 // (op 2148) 


v_and_b32 v13, -4, v1 // (op 2377) 
v_sub_u32 v16, v1, v13 // (op 2377) 

v_lshrrev_b32 v13, 2, v1 // (op 2377) 
v_lshl_add_u32 v18, v13, 3, v16 // (op 2377) 


s_mov_b32 s1, 0 // (op 836) call()
s_and_b32 s52, s7, 4294967294 // (op 835) 
s_cmp_lt_u32 s1, s52 // (op 835) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_835 // (op 835) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_835 if false)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoop_835:
s_add_u32 s1, s1, 2 // (op 837) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2462) 



ds_read_u8 v13, v0 // (op 69) Load local data 



ds_read_u8 v16, v0 offset:128 // (op 861) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:33], a[0:3], v15, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 72) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:29], v[34:37], a[4:7], v15, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 862) 



v_mov_b32 v13, v12 // (op 57) 

buffer_load_dwordx4 v[22:25], v13, s[68:71], 0 offen offset:256 // (op 57) Load value
buffer_load_dwordx4 v[26:29], v13, s[68:71], 0 offen offset:320 // (op 57) Load value


v_mov_b32 v16, v14 // (op 61) 
v_add_u32 v19, 128, v16 // (op 61) 

buffer_load_dwordx4 v[30:33], v19, s[72:75], 0 offen // (op 61) Load value



v_mov_b32 v19, v17 // (op 65) 

buffer_load_ubyte v15, v19, s[76:79], 0 offen offset:8 // (op 65) Load value


s_lshl_b64 s[52:53], s[36:37], 5 // (op 884) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 884) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 884) 

s_add_u32 s54, s52, 128 // (op 884) 

v_add_u32 v38, v16, s54 // (op 884) 


buffer_load_dwordx4 v[34:37], v38, s[72:75], 0 offen // (op 884) Load value


v_mov_b64 v[38:39], s[44:45] // (op 13) convert

v_mov_b32 v60, v20 // (op 13) 
v_mov_b32 v61, v60 // (op 13) 
buffer_load_ubyte v40, v61, s[80:83], 0 offen offset:8 // (op 13) Load value
v_add_u32 v61, v61, v38 // (op 13) 
buffer_load_ubyte v41, v61, s[80:83], 0 offen offset:8 // (op 13) Load value


s_barrier  // (op 2503) 



s_waitcnt vmcnt(8)
ds_write_b8 v18, v58 offset:256 // (op 777) Store local data 
s_waitcnt vmcnt(7)
ds_write_b8 v18, v59 offset:260 // (op 777) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2500) 



ds_read_u8 v38, v5 offset:256 // (op 779) Load local data 



ds_read_u8 v39, v5 offset:384 // (op 865) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[50:53], a[0:3], v9, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 780) 

s_barrier  // (op 2660) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:49], v[54:57], a[4:7], v9, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 863) 




buffer_load_dwordx4 v[42:45], v13, s[68:71], 0 offen offset:384 // (op 782) Load value
buffer_load_dwordx4 v[46:49], v13, s[68:71], 0 offen offset:448 // (op 782) Load value



v_add_u32 v12, 256, v12 // (op 1662) 
v_add_u32 v13, 192, v16 // (op 784) 

buffer_load_dwordx4 v[50:53], v13, s[72:75], 0 offen // (op 784) Load value




buffer_load_ubyte v9, v19, s[76:79], 0 offen offset:12 // (op 786) Load value



v_add_u32 v17, 8, v17 // (op 1948) 
s_lshl_b64 s[52:53], s[36:37], 5 // (op 879) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 879) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 879) 

s_add_u32 s54, s52, 192 // (op 879) 

v_add_u32 v13, v16, s54 // (op 879) 


buffer_load_dwordx4 v[54:57], v13, s[72:75], 0 offen // (op 879) Load value



v_add_u32 v14, 128, v14 // (op 1815) 
v_mov_b64 v[38:39], s[44:45] // (op 776) convert

v_mov_b32 v13, v60 // (op 776) 
buffer_load_ubyte v58, v13, s[80:83], 0 offen offset:12 // (op 776) Load value
v_add_u32 v13, v13, v38 // (op 776) 
buffer_load_ubyte v59, v13, s[80:83], 0 offen offset:12 // (op 776) Load value



s_barrier  // (op 2468) 
v_add_u32 v20, 8, v20 // (op 2076) 



s_waitcnt vmcnt(8)
ds_write_b8 v21, v40 // (op 18) Store local data 
s_waitcnt vmcnt(7)
ds_write_b8 v21, v41 offset:4 // (op 18) Store local data 
s_and_b32 s52, s7, 4294967294 // (op 835) 
s_cmp_lt_u32 s1, s52 // (op 835) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoop_835 // (op 835) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoop_835 if true)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_835:











s_and_b32 s52, s7, 4294967294 // (op 1578) 
v_lshrrev_b32 v12, 7, v1 // (op 1578) 
s_add_u32 s53, s0, s52 // (op 1578) 

v_and_b32 v13, 15, v1 // (op 1578) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 1578) 


s_mul_i32 s52, s16, s8 // (op 1578) 
s_add_u32 s52, s52, s53 // (op 1578) 

v_and_b32 v12, 48, v1 // (op 1578) 
s_mul_hi_u32 s53, s52, s13 // (op 1578) 
s_sub_u32 s54, s52, s53 // (op 1578) 
s_lshr_b32 s55, s54, 1 // (op 1578) 

s_add_u32 s54, s55, s53 // (op 1578) 


s_lshr_b32 s53, s54, s11 // (op 1578) 

s_and_b32 s54, s11, 2147483648 // (op 1578) 
s_cmp_eq_i32 s54, 0 // (op 1578) 
s_mov_b32 s55, scc // (op 1578) 

s_cmp_eq_i32 s55, 0 // (op 1578) 
s_cselect_b32 s54, s52, s53 // (op 1578) 


s_mul_hi_u32 s53, s54, s12 // (op 1578) 
s_sub_u32 s55, s54, s53 // (op 1578) 
s_lshr_b32 s84, s55, 1 // (op 1578) 

s_add_u32 s55, s84, s53 // (op 1578) 


s_lshr_b32 s53, s55, s10 // (op 1578) 

s_and_b32 s55, s10, 2147483648 // (op 1578) 
s_cmp_eq_i32 s55, 0 // (op 1578) 
s_mov_b32 s84, scc // (op 1578) 

s_cmp_eq_i32 s84, 0 // (op 1578) 
s_cselect_b32 s55, s54, s53 // (op 1578) 


v_lshl_add_u32 v13, s55, 4, v14 // (op 1578) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 1578) 
s_mul_i32 s53, s54, s84 // (op 1578) 


s_sub_u32 s54, s52, s53 // (op 1578) 


v_lshl_add_u32 v14, s54, 7, v12 // (op 1578) 


v_mul_lo_u32 v12, v13, s30 // (op 1578) 
v_add_u32 v12, v12, v14 // (op 1578) 


s_mul_i32 s53, s30, s35 // (op 1580) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 1580) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 1580) least significant
s_mul_hi_u32 s55, s30, s34 // (op 1580) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1580) most significant: sum
s_add_u32 s53, s53, s55 // (op 1580) most significant: sum


s_mov_b32 s68, s32 // (op 1580) reg expression
s_mov_b32 s69, s33 // (op 1580) reg expression
s_mov_b32 s70, s52 // (op 1580) reg expression
s_mov_b32 s71, 131072 // (op 1580) call()

s_and_b32 s52, s7, 4294967294 // (op 1721) 
v_lshrrev_b32 v13, 6, v1 // (op 1721) 
s_add_u32 s53, s0, s52 // (op 1721) 

v_and_b32 v14, 1, v13 // (op 1721) 

v_and_b32 v13, 15, v1 // (op 1721) 
s_mul_i32 s52, s16, s8 // (op 1721) 
s_add_u32 s52, s52, s53 // (op 1721) 

v_lshl_add_u32 v15, v14, 4, v13 // (op 1721) 


s_mul_hi_u32 s53, s52, s13 // (op 1721) 
v_and_b32 v13, 63, v1 // (op 1721) 
s_sub_u32 s54, s52, s53 // (op 1721) 
v_lshrrev_b32 v14, 4, v13 // (op 1721) 

s_lshr_b32 s55, s54, 1 // (op 1721) 

v_lshlrev_b32 v13, 5, v14 // (op 1721) 

s_add_u32 s54, s55, s53 // (op 1721) 


s_lshr_b32 s53, s54, s11 // (op 1721) 

s_and_b32 s54, s11, 2147483648 // (op 1721) 
s_cmp_eq_i32 s54, 0 // (op 1721) 
s_mov_b32 s55, scc // (op 1721) 

s_cmp_eq_i32 s55, 0 // (op 1721) 
s_cselect_b32 s54, s52, s53 // (op 1721) 


s_mul_hi_u32 s53, s54, s12 // (op 1721) 
s_sub_u32 s55, s54, s53 // (op 1721) 
s_lshr_b32 s84, s55, 1 // (op 1721) 

s_add_u32 s55, s84, s53 // (op 1721) 


s_lshr_b32 s53, s55, s10 // (op 1721) 

s_and_b32 s55, s10, 2147483648 // (op 1721) 
s_cmp_eq_i32 s55, 0 // (op 1721) 
s_mov_b32 s84, scc // (op 1721) 

s_cmp_eq_i32 s84, 0 // (op 1721) 
s_cselect_b32 s55, s54, s53 // (op 1721) 


s_mul_i32 s53, s55, s14 // (op 1721) 

s_sub_u32 s55, s54, s53 // (op 1721) 

v_lshl_add_u32 v14, s55, 6, v15 // (op 1721) 


v_mul_lo_u32 v41, v14, s37 // (op 1721) most significant: low of low * high
v_mul_hi_u32 v15, v14, s36 // (op 1721) most significant: high of low * low
v_mul_lo_u32 v40, v14, s36 // (op 1721) least significant: low of low * low
v_add_u32 v41, v41, v15 // (op 1721) most significant: sum


s_ashr_i64 s[84:85], s[2:3], 7 // (op 1721) 
s_mul_i32 s53, s54, s84 // (op 1721) 


s_sub_u32 s54, s52, s53 // (op 1721) 


v_lshl_add_u32 v14, s54, 7, v13 // (op 1721) 


v_mov_b32 v39, 0 // (op 1721) convert
v_mov_b32 v38, v14 // (op 1721) convert

v_add_co_u32 v36, s[52:53], v38, v40 // (op 1721) least significant half
v_addc_co_u32 v37, s[52:53], v39, v41, s[52:53] // (op 1721) most significant half

v_lshlrev_b64 v[36:37], 2, v[36:37] // (op 1721) 


v_lshrrev_b64 v[34:35], 3, v[36:37] // (op 1721) 

v_mov_b32 v13, v34 // (op 1721) convert

s_mul_i32 s53, s36, s23 // (op 1723) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 1723) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 1723) least significant
s_mul_hi_u32 s55, s36, s22 // (op 1723) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1723) most significant: sum
s_add_u32 s53, s53, s55 // (op 1723) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1723) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1723) 

s_mov_b32 s72, s38 // (op 1723) reg expression
s_mov_b32 s73, s39 // (op 1723) reg expression
s_mov_b32 s74, s52 // (op 1723) reg expression
s_mov_b32 s75, 131072 // (op 1723) call()

s_and_b32 s52, s7, 4294967294 // (op 1872) 
v_lshrrev_b32 v14, 7, v1 // (op 1872) 
s_add_u32 s53, s0, s52 // (op 1872) 

v_and_b32 v15, 15, v1 // (op 1872) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 1872) 


s_mul_i32 s52, s16, s8 // (op 1872) 
s_add_u32 s52, s52, s53 // (op 1872) 

v_and_b32 v14, 63, v1 // (op 1872) 
s_mul_hi_u32 s53, s52, s13 // (op 1872) 
v_lshrrev_b32 v15, 4, v14 // (op 1872) 

s_sub_u32 s54, s52, s53 // (op 1872) 
s_lshr_b32 s55, s54, 1 // (op 1872) 

s_add_u32 s54, s55, s53 // (op 1872) 


s_lshr_b32 s53, s54, s11 // (op 1872) 

s_and_b32 s54, s11, 2147483648 // (op 1872) 
s_cmp_eq_i32 s54, 0 // (op 1872) 
s_mov_b32 s55, scc // (op 1872) 

s_cmp_eq_i32 s55, 0 // (op 1872) 
s_cselect_b32 s54, s52, s53 // (op 1872) 


s_mul_hi_u32 s53, s54, s12 // (op 1872) 
s_sub_u32 s55, s54, s53 // (op 1872) 
s_lshr_b32 s84, s55, 1 // (op 1872) 

s_add_u32 s55, s84, s53 // (op 1872) 


s_lshr_b32 s53, s55, s10 // (op 1872) 

s_and_b32 s55, s10, 2147483648 // (op 1872) 
s_cmp_eq_i32 s55, 0 // (op 1872) 
s_mov_b32 s84, scc // (op 1872) 

s_cmp_eq_i32 s84, 0 // (op 1872) 
s_cselect_b32 s55, s54, s53 // (op 1872) 


v_lshl_add_u32 v14, s55, 4, v16 // (op 1872) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 1872) 
s_mul_i32 s53, s54, s84 // (op 1872) 


s_sub_u32 s54, s52, s53 // (op 1872) 


v_lshl_add_u32 v16, s54, 2, v15 // (op 1872) 


v_mul_lo_u32 v15, v14, s40 // (op 1872) 
v_add_u32 v15, v15, v16 // (op 1872) 


s_mul_i32 s53, s40, s35 // (op 1874) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 1874) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 1874) least significant
s_mul_hi_u32 s55, s40, s34 // (op 1874) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1874) most significant: sum
s_add_u32 s53, s53, s55 // (op 1874) most significant: sum


s_mov_b32 s76, s42 // (op 1874) reg expression
s_mov_b32 s77, s43 // (op 1874) reg expression
s_mov_b32 s78, s52 // (op 1874) reg expression
s_mov_b32 s79, 131072 // (op 1874) call()

s_and_b32 s52, s7, 4294967294 // (op 2000) 
v_lshrrev_b32 v14, 2, v1 // (op 2000) 
s_add_u32 s53, s0, s52 // (op 2000) 

v_lshlrev_b32 v16, 1, v14 // (op 2000) 

v_and_b32 v14, -4, v1 // (op 2000) 
s_mul_i32 s52, s16, s8 // (op 2000) 
s_add_u32 s52, s52, s53 // (op 2000) 

v_sub_u32 v17, v1, v14 // (op 2000) 

s_mul_hi_u32 s53, s52, s13 // (op 2000) 
s_sub_u32 s54, s52, s53 // (op 2000) 
s_lshr_b32 s55, s54, 1 // (op 2000) 

s_add_u32 s54, s55, s53 // (op 2000) 


s_lshr_b32 s53, s54, s11 // (op 2000) 

s_and_b32 s54, s11, 2147483648 // (op 2000) 
s_cmp_eq_i32 s54, 0 // (op 2000) 
s_mov_b32 s55, scc // (op 2000) 

s_cmp_eq_i32 s55, 0 // (op 2000) 
s_cselect_b32 s54, s52, s53 // (op 2000) 


s_mul_hi_u32 s53, s54, s12 // (op 2000) 
s_sub_u32 s55, s54, s53 // (op 2000) 
s_lshr_b32 s84, s55, 1 // (op 2000) 

s_add_u32 s55, s84, s53 // (op 2000) 


s_lshr_b32 s53, s55, s10 // (op 2000) 

s_and_b32 s55, s10, 2147483648 // (op 2000) 
s_cmp_eq_i32 s55, 0 // (op 2000) 
s_mov_b32 s84, scc // (op 2000) 

s_cmp_eq_i32 s84, 0 // (op 2000) 
s_cselect_b32 s55, s54, s53 // (op 2000) 


s_mul_i32 s53, s55, s14 // (op 2000) 

s_sub_u32 s55, s54, s53 // (op 2000) 

v_lshl_add_u32 v14, s55, 6, v16 // (op 2000) 


v_mul_lo_u32 v16, v14, s44 // (op 2000) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 2000) 
s_mul_i32 s53, s54, s84 // (op 2000) 


s_sub_u32 s54, s52, s53 // (op 2000) 


v_lshl_add_u32 v14, s54, 2, v17 // (op 2000) 


v_add_u32 v17, v14, v16 // (op 2000) 


s_mul_i32 s53, s44, s23 // (op 2002) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 2002) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 2002) least significant
s_mul_hi_u32 s55, s44, s22 // (op 2002) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2002) most significant: sum
s_add_u32 s53, s53, s55 // (op 2002) most significant: sum


s_mov_b32 s80, s46 // (op 2002) reg expression
s_mov_b32 s81, s47 // (op 2002) reg expression
s_mov_b32 s82, s52 // (op 2002) reg expression
s_mov_b32 s83, 131072 // (op 2002) call()

v_and_b32 v14, -4, v1 // (op 2161) 
v_sub_u32 v16, v1, v14 // (op 2161) 

v_lshrrev_b32 v14, 2, v1 // (op 2161) 
v_lshl_add_u32 v18, v14, 3, v16 // (op 2161) 


s_and_b32 s52, s7, 4294967294 // (op 841) 
s_cmp_lt_u32 s52, s7 // (op 840) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_840 // (op 840) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_840 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_840:

s_add_u32 s52, s52, 1 // (op 842) 
v_mov_b64 v[20:21], s[44:45] // (op 734) convert

v_mov_b32 v9, v17 // (op 734) 
v_mov_b32 v14, v9 // (op 734) 
buffer_load_ubyte v22, v14, s[80:83], 0 offen offset: 0 // (op 734) Load value
v_add_u32 v14, v14, v20 // (op 734) 
buffer_load_ubyte v23, v14, s[80:83], 0 offen offset: 0 // (op 734) Load value



v_add_u32 v17, 4, v17 // (op 1992) 


s_barrier  // (op 2471) 
v_mov_b32 v9, v12 // (op 740) 




buffer_load_dwordx4 v[24:27], v9, s[68:71], 0 offen // (op 740) Load value
s_waitcnt vmcnt(2)
ds_write_b8 v18, v22 // (op 735) Store local data 
buffer_load_dwordx4 v[28:31], v9, s[68:71], 0 offen offset:64 // (op 740) Load value

v_add_u32 v12, 128, v12 // (op 1568) 
s_waitcnt vmcnt(2)
ds_write_b8 v18, v23 offset:4 // (op 735) Store local data 



v_mov_b32 v9, v13 // (op 742) 
v_mov_b32 v14, v9 // (op 742) reg expression

buffer_load_dwordx4 v[20:23], v14, s[72:75], 0 offen // (op 742) Load value



v_mov_b32 v14, v15 // (op 744) 

buffer_load_ubyte v16, v14, s[76:79], 0 offen offset: 0 // (op 744) Load value

v_add_u32 v15, 4, v15 // (op 1864) 


s_lshl_b64 s[54:55], s[36:37], 5 // (op 854) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 854) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 854) 

v_add_u32 v14, v9, s54 // (op 854) 


s_waitcnt lgkmcnt(0)
s_barrier  // (op 2474) 
buffer_load_dwordx4 v[32:35], v14, s[72:75], 0 offen // (op 854) Load value





ds_read_u8 v9, v0 // (op 737) Load local data 



v_add_u32 v13, 64, v13 // (op 1711) 
ds_read_u8 v14, v0 offset:128 // (op 846) Load local data 
s_waitcnt vmcnt(1) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:31], v[20:23], a[0:3], v16, v9 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 738) 


s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:31], v[32:35], a[4:7], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 847) 




s_cmp_lt_u32 s52, s7 // (op 840) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_840 // (op 840) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_840 if true)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_840:






s_barrier  // (op 2646) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 251) 
s_cmp_lg_u32 s18, 0 // (op 277) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Send_Tile_277 // (op 277) Condition: False, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Send_Tile_277



s_nop 0
buffer_store_dword a0, v8, s[64:67], 0 offen sc1 // (op 274) Store value
buffer_store_dword a1, v8, s[64:67], 0 offen offset:256 sc1 // (op 274) Store value
buffer_store_dword a2, v8, s[64:67], 0 offen offset:512 sc1 // (op 274) Store value
buffer_store_dword a3, v8, s[64:67], 0 offen offset:768 sc1 // (op 274) Store value



buffer_store_dword a4, v8, s[64:67], 0 offen offset:128 sc1 // (op 991) Store value
buffer_store_dword a5, v8, s[64:67], 0 offen offset:384 sc1 // (op 991) Store value
buffer_store_dword a6, v8, s[64:67], 0 offen offset:640 sc1 // (op 991) Store value
buffer_store_dword a7, v8, s[64:67], 0 offen offset:896 sc1 // (op 991) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 278) 
v_cmp_eq_i32 vcc, v1, 0 // (op 282) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_282 // (op 282) Condition: False, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_282
s_mov_b32 s52, 1 // (op 280) call()
s_mov_b32 s55, 0 // (op 281) convert
s_mov_b32 s54, s16 // (op 281) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 281) 
s_add_u32 s84, s48, s54 // (op 281) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 281) End of Int64 add, unlocking SCC
s_store_dword s52, s[84:85], 0 glc // (op 281) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_282 // (op 282) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_282
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_282:
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_282:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Send_Tile_277 // (op 277) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Send_Tile_277
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Send_Tile_277:
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Send_Tile_277:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 293) 
s_mov_b32 s52, scc // (op 293) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 293) 
s_mov_b32 s53, s54 // (op 293) convert

s_sub_u32 s54, s53, 1 // (op 293) 

s_cmp_lt_u32 s19, s54 // (op 293) 
s_mov_b32 s53, scc // (op 293) 

s_and_b32 s54, s52, s53 // (op 293) 
s_cmp_lg_u32 s54, 0 // (op 293) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Receive_Tile_293 // (op 293) Condition: False, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Receive_Tile_293
s_mov_b32 s52, 0 // (op 259) call()
s_mul_i32 s53, s52, s8 // (op 258) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 258) 
s_mov_b32 s84, s54 // (op 258) convert

s_sub_u32 s54, s84, s19 // (op 258) 

s_sub_u32 s55, s54, 1 // (op 258) 

s_cmp_lt_u32 s53, s55 // (op 258) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_258 // (op 258) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_258 if false)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_ReceiveTileLoop_258:
s_add_u32 s53, s16, 1 // (op 2333) 
v_lshrrev_b32 v8, 5, v1 // (op 2333) 
s_add_u32 s54, s53, s52 // (op 2333) 
s_lshl_b32 s54, s54, 4 // (op 2333) 

v_lshlrev_b32 v9, 2, v8 // (op 2333) 

v_add_lshl_u32 v8, s54, v9, 6 // (op 2333) 


v_and_b32 v9, -32, v1 // (op 2333) 
v_sub_u32 v12, v1, v9 // (op 2333) 

v_add_lshl_u32 v9, v8, v12, 2 // (op 2333) 


s_lshl_b32 s53, s15, 12 // (op 2335) 
s_mov_b32 s64, s28 // (op 2335) reg expression
s_mov_b32 s65, s29 // (op 2335) reg expression
s_mov_b32 s66, s53 // (op 2335) reg expression
s_mov_b32 s67, 131072 // (op 2335) call()

RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_297:
s_add_u32 s53, s16, 1 // (op 295) 
s_add_u32 s54, s53, s52 // (op 295) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 295) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 295) 

s_add_u32 s84, s48, s54 // (op 295) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 295) End of Int64 add, unlocking SCC
s_load_dword s53, s[84:85], 0 glc // (op 295) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 297) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_297 // (op 297) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_297 if true)
s_barrier  // (op 301) 
v_cmp_eq_i32 vcc, v1, 0 // (op 300) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_300 // (op 300) Condition: False, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_300
s_mov_b32 s53, 0 // (op 298) call()
s_add_u32 s54, s16, 1 // (op 299) 
s_add_u32 s84, s54, s52 // (op 299) Start of Int64 add, locking SCC
s_addc_u32 s85, 0, 0 // (op 299) End of Int64 add, unlocking SCC
s_lshl_b64 s[84:85], s[84:85], 2 // (op 299) 

s_add_u32 s54, s48, s84 // (op 299) Start of Int64 add, locking SCC
s_addc_u32 s55, s49, s85 // (op 299) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 299) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_300 // (op 300) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_300
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_300:
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_300:




buffer_load_dword v12, v9, s[64:67], 0 offen // (op 312) Load value
buffer_load_dword v13, v9, s[64:67], 0 offen offset:256 // (op 312) Load value
buffer_load_dword v14, v9, s[64:67], 0 offen offset:512 // (op 312) Load value
buffer_load_dword v15, v9, s[64:67], 0 offen offset:768 // (op 312) Load value
v_accvgpr_read v16, a0 // (op 315) reg expression
v_accvgpr_read v17, a1 // (op 315) reg expression
v_accvgpr_read v18, a2 // (op 315) reg expression
v_accvgpr_read v19, a3 // (op 315) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v16, v16, v12 // (op 317) 
s_waitcnt vmcnt(2)
v_add_f32 v17, v17, v13 // (op 317) 
s_waitcnt vmcnt(1)
v_add_f32 v18, v18, v14 // (op 317) 
s_waitcnt vmcnt(0)
v_add_f32 v19, v19, v15 // (op 317) 

v_accvgpr_write a0, v16 // (op 319) reg expression
v_accvgpr_write a1, v17 // (op 319) reg expression
v_accvgpr_write a2, v18 // (op 319) reg expression
v_accvgpr_write a3, v19 // (op 319) reg expression




buffer_load_dword v12, v9, s[64:67], 0 offen offset:128 // (op 1001) Load value
buffer_load_dword v13, v9, s[64:67], 0 offen offset:384 // (op 1001) Load value
buffer_load_dword v14, v9, s[64:67], 0 offen offset:640 // (op 1001) Load value
buffer_load_dword v15, v9, s[64:67], 0 offen offset:896 // (op 1001) Load value
v_accvgpr_read v16, a4 // (op 1002) reg expression
v_accvgpr_read v17, a5 // (op 1002) reg expression
v_accvgpr_read v18, a6 // (op 1002) reg expression
v_accvgpr_read v19, a7 // (op 1002) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v16, v16, v12 // (op 1003) 
s_waitcnt vmcnt(2)
v_add_f32 v17, v17, v13 // (op 1003) 
s_waitcnt vmcnt(1)
v_add_f32 v18, v18, v14 // (op 1003) 
s_waitcnt vmcnt(0)
v_add_f32 v19, v19, v15 // (op 1003) 

v_accvgpr_write a4, v16 // (op 1004) reg expression
v_accvgpr_write a5, v17 // (op 1004) reg expression
v_accvgpr_write a6, v18 // (op 1004) reg expression
v_accvgpr_write a7, v19 // (op 1004) reg expression

s_add_u32 s52, s52, 1 // (op 260) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 258) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 258) 
s_mov_b32 s84, s54 // (op 258) convert

s_sub_u32 s54, s84, s19 // (op 258) 

s_sub_u32 s55, s54, 1 // (op 258) 

s_cmp_lt_u32 s53, s55 // (op 258) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_ReceiveTileLoop_258 // (op 258) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_ReceiveTileLoop_258 if true)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_258:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Receive_Tile_293 // (op 293) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Receive_Tile_293
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Receive_Tile_293:
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Receive_Tile_293:

s_cmp_eq_i32 s18, 0 // (op 336) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_336 // (op 336) Condition: False, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_336
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 108) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_108 // (op 108) Condition: False, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_108
v_accvgpr_read v12, a0 // (op 38) 
v_accvgpr_read v13, a1 // (op 38) 
v_accvgpr_read v14, a2 // (op 38) 
v_accvgpr_read v15, a3 // (op 38) 
v_mul_f32 v16, v7, v12 // (op 38) 
v_mul_f32 v17, v7, v13 // (op 38) 
v_mul_f32 v18, v7, v14 // (op 38) 
v_mul_f32 v19, v7, v15 // (op 38) 

v_cvt_f16_f32 v12, v16 // (op 44) 
v_cvt_f16_f32 v13, v17 // (op 44) 
v_cvt_f16_f32 v14, v18 // (op 44) 
v_cvt_f16_f32 v15, v19 // (op 44) 


v_pack_b32_f16 v8, v12, v13 // (op 46) 
v_pack_b32_f16 v9, v14, v15 // (op 46) 

v_mov_b32 v16, v11 // (op 46) reg expression

buffer_store_dwordx2 v[8:9], v16, s[60:63], 0 offen // (op 46) Store value



v_accvgpr_read v12, a4 // (op 1020) 
v_accvgpr_read v13, a5 // (op 1020) 
v_accvgpr_read v14, a6 // (op 1020) 
v_accvgpr_read v15, a7 // (op 1020) 
v_mul_f32 v16, v7, v12 // (op 1020) 
v_mul_f32 v17, v7, v13 // (op 1020) 
v_mul_f32 v18, v7, v14 // (op 1020) 
v_mul_f32 v19, v7, v15 // (op 1020) 

v_cvt_f16_f32 v12, v16 // (op 1022) 
v_cvt_f16_f32 v13, v17 // (op 1022) 
v_cvt_f16_f32 v14, v18 // (op 1022) 
v_cvt_f16_f32 v15, v19 // (op 1022) 


v_pack_b32_f16 v8, v12, v13 // (op 1023) 
v_pack_b32_f16 v9, v14, v15 // (op 1023) 

s_lshl_b32 s52, s24, 6 // (op 1023) 
v_add_u32 v16, v11, s52 // (op 1023) 


buffer_store_dwordx2 v[8:9], v16, s[60:63], 0 offen // (op 1023) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_108 // (op 108) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_108
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_108:


v_mov_b32 v8, v10 // (op 121) reg expression

buffer_load_dwordx2 v[12:13], v8, s[56:59], 0 offen // (op 121) Load value

v_accvgpr_read v14, a0 // (op 117) 
v_accvgpr_read v15, a1 // (op 117) 
v_accvgpr_read v16, a2 // (op 117) 
v_accvgpr_read v17, a3 // (op 117) 
v_mul_f32 v18, v7, v14 // (op 117) 
v_mul_f32 v19, v7, v15 // (op 117) 
v_mul_f32 v20, v7, v16 // (op 117) 
v_mul_f32 v21, v7, v17 // (op 117) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v8, v12 // (op 122) 
v_lshrrev_b32 v9, 16, v12 // (op 122) 
v_cvt_f32_f16 v9, v9 // (op 122) 
v_mul_f32 v14, v6, v8 // (op 122) 
v_mul_f32 v15, v6, v9 // (op 122) 
v_cvt_f32_f16 v8, v13 // (op 122) 
v_lshrrev_b32 v9, 16, v13 // (op 122) 
v_cvt_f32_f16 v9, v9 // (op 122) 
v_mul_f32 v16, v6, v8 // (op 122) 
v_mul_f32 v17, v6, v9 // (op 122) 


v_add_f32 v22, v14, v18 // (op 118) 
v_add_f32 v23, v15, v19 // (op 118) 
v_add_f32 v24, v16, v20 // (op 118) 
v_add_f32 v25, v17, v21 // (op 118) 


v_cvt_f16_f32 v12, v22 // (op 119) 
v_cvt_f16_f32 v13, v23 // (op 119) 
v_cvt_f16_f32 v14, v24 // (op 119) 
v_cvt_f16_f32 v15, v25 // (op 119) 


v_pack_b32_f16 v8, v12, v13 // (op 120) 
v_pack_b32_f16 v9, v14, v15 // (op 120) 

v_mov_b32 v16, v11 // (op 120) reg expression

buffer_store_dwordx2 v[8:9], v16, s[60:63], 0 offen // (op 120) Store value





s_lshl_b32 s52, s4, 6 // (op 1047) 
v_add_u32 v8, v10, s52 // (op 1047) 


buffer_load_dwordx2 v[12:13], v8, s[56:59], 0 offen // (op 1047) Load value

v_accvgpr_read v14, a4 // (op 1043) 
v_accvgpr_read v15, a5 // (op 1043) 
v_accvgpr_read v16, a6 // (op 1043) 
v_accvgpr_read v17, a7 // (op 1043) 
v_mul_f32 v18, v7, v14 // (op 1043) 
v_mul_f32 v19, v7, v15 // (op 1043) 
v_mul_f32 v20, v7, v16 // (op 1043) 
v_mul_f32 v21, v7, v17 // (op 1043) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v8, v12 // (op 1048) 
v_lshrrev_b32 v9, 16, v12 // (op 1048) 
v_cvt_f32_f16 v9, v9 // (op 1048) 
v_mul_f32 v14, v6, v8 // (op 1048) 
v_mul_f32 v15, v6, v9 // (op 1048) 
v_cvt_f32_f16 v8, v13 // (op 1048) 
v_lshrrev_b32 v9, 16, v13 // (op 1048) 
v_cvt_f32_f16 v9, v9 // (op 1048) 
v_mul_f32 v16, v6, v8 // (op 1048) 
v_mul_f32 v17, v6, v9 // (op 1048) 


v_add_f32 v22, v14, v18 // (op 1044) 
v_add_f32 v23, v15, v19 // (op 1044) 
v_add_f32 v24, v16, v20 // (op 1044) 
v_add_f32 v25, v17, v21 // (op 1044) 


v_cvt_f16_f32 v12, v22 // (op 1045) 
v_cvt_f16_f32 v13, v23 // (op 1045) 
v_cvt_f16_f32 v14, v24 // (op 1045) 
v_cvt_f16_f32 v15, v25 // (op 1045) 


v_pack_b32_f16 v8, v12, v13 // (op 1046) 
v_pack_b32_f16 v9, v14, v15 // (op 1046) 

s_lshl_b32 s52, s24, 6 // (op 1046) 
v_add_u32 v16, v11, s52 // (op 1046) 


buffer_store_dwordx2 v[8:9], v16, s[60:63], 0 offen // (op 1046) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_108:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_336 // (op 336) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_336
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_336:
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_336:


s_mul_i32 s52, s8, s16 // (op 249) 
s_cmp_gt_u32 s9, s52 // (op 249) 
s_mov_b32 s53, scc // (op 249) 
s_sub_u32 s54, s9, s52 // (op 249) 

s_cmp_eq_i32 s53, 0 // (op 249) 
s_cselect_b32 s52, 0, s54 // (op 249) 


s_cmp_lt_u32 s8, s52 // (op 249) 
s_mov_b32 s53, scc // (op 249) 
s_cmp_eq_i32 s53, 0 // (op 249) 
s_cselect_b32 s54, s52, s8 // (op 249) 


s_cmp_lt_u32 s0, s54 // (op 249) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_SKStreamTileLoop_249 // (op 249) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_SKStreamTileLoop_249 if true)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_249:











s_mov_b32 s0, 0 // (op 390) call()
s_mov_b32 s1, 0 // (op 391) call()
s_mov_b32 s0, 0 // (op 405) call()
s_mul_i32 s15, s50, s16 // (op 404) 
s_cmp_gt_u32 s51, s15 // (op 404) 
s_mov_b32 s6, scc // (op 404) 
s_sub_u32 s7, s51, s15 // (op 404) 

s_cmp_eq_i32 s6, 0 // (op 404) 
s_cselect_b32 s8, 0, s7 // (op 404) 


s_cmp_lt_u32 s50, s8 // (op 404) 
s_mov_b32 s15, scc // (op 404) 
s_cmp_eq_i32 s15, 0 // (op 404) 
s_cselect_b32 s6, s8, s50 // (op 404) 


s_cmp_lt_u32 s0, s6 // (op 404) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_404 // (op 404) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_404 if false)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_DPStreamTileLoop_404:
v_lshrrev_b32 v0, 7, v1 // (op 2131) 
s_mul_i32 s15, s16, s50 // (op 2131) 
s_add_u32 s15, s15, s0 // (op 2131) 
v_and_b32 v5, 63, v1 // (op 2131) 
s_add_u32 s6, s15, s9 // (op 2131) 

v_lshrrev_b32 v8, 4, v5 // (op 2131) 
s_mul_hi_u32 s15, s6, s13 // (op 2131) 
v_lshlrev_b32 v9, 2, v8 // (op 2131) 

s_sub_u32 s7, s6, s15 // (op 2131) 
v_lshl_add_u32 v8, v0, 4, v9 // (op 2131) 


s_lshr_b32 s8, s7, 1 // (op 2131) 

v_lshrrev_b32 v0, 2, v5 // (op 2131) 

s_add_u32 s7, s8, s15 // (op 2131) 


v_and_b32 v5, 3, v0 // (op 2131) 

s_lshr_b32 s8, s7, s11 // (op 2131) 

v_and_b32 v0, 3, v1 // (op 2131) 
s_and_b32 s7, s11, 2147483648 // (op 2131) 
v_lshl_add_u32 v9, v5, 2, v0 // (op 2131) 


s_cmp_eq_i32 s7, 0 // (op 2131) 
s_mov_b32 s15, scc // (op 2131) 

v_lshrrev_b32 v0, 6, v1 // (op 2131) 
s_cmp_eq_i32 s15, 0 // (op 2131) 
s_cselect_b32 s7, s6, s8 // (op 2131) 



v_and_b32 v5, 1, v0 // (op 2131) 

s_mul_hi_u32 s6, s7, s12 // (op 2131) 
v_lshl_add_u32 v0, v5, 4, v9 // (op 2131) 


s_sub_u32 s8, s7, s6 // (op 2131) 
s_lshr_b32 s15, s8, 1 // (op 2131) 

s_add_u32 s8, s15, s6 // (op 2131) 


s_lshr_b32 s6, s8, s10 // (op 2131) 

s_and_b32 s8, s10, 2147483648 // (op 2131) 
s_cmp_eq_i32 s8, 0 // (op 2131) 
s_mov_b32 s15, scc // (op 2131) 

s_cmp_eq_i32 s15, 0 // (op 2131) 
s_cselect_b32 s8, s7, s6 // (op 2131) 


s_mul_i32 s6, s8, s14 // (op 2131) 
v_lshl_add_u32 v5, s8, 4, v8 // (op 2131) 


s_sub_u32 s8, s7, s6 // (op 2131) 


v_lshl_add_u32 v8, s8, 6, v0 // (op 2131) 


v_mul_lo_u32 v0, v8, s4 // (op 2131) 

v_add_lshl_u32 v8, v5, v0, 1 // (op 2131) 


s_mul_i32 s19, s4, s23 // (op 2133) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 2133) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 2133) least significant
s_mul_hi_u32 s6, s4, s22 // (op 2133) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2133) most significant: sum
s_add_u32 s19, s19, s6 // (op 2133) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 2133) 

s_mov_b32 s56, s20 // (op 2133) reg expression
s_mov_b32 s57, s21 // (op 2133) reg expression
s_mov_b32 s58, s28 // (op 2133) reg expression
s_mov_b32 s59, 131072 // (op 2133) call()

v_lshrrev_b32 v0, 7, v1 // (op 2231) 
s_mul_i32 s15, s16, s50 // (op 2231) 
s_add_u32 s15, s15, s0 // (op 2231) 
v_and_b32 v5, 63, v1 // (op 2231) 
s_add_u32 s6, s15, s9 // (op 2231) 

v_lshrrev_b32 v9, 4, v5 // (op 2231) 
s_mul_hi_u32 s15, s6, s13 // (op 2231) 
v_lshlrev_b32 v10, 2, v9 // (op 2231) 

s_sub_u32 s7, s6, s15 // (op 2231) 
v_lshl_add_u32 v9, v0, 4, v10 // (op 2231) 


s_lshr_b32 s8, s7, 1 // (op 2231) 

v_lshrrev_b32 v0, 2, v5 // (op 2231) 

s_add_u32 s7, s8, s15 // (op 2231) 


v_and_b32 v5, 3, v0 // (op 2231) 

s_lshr_b32 s8, s7, s11 // (op 2231) 

v_and_b32 v0, 3, v1 // (op 2231) 
s_and_b32 s7, s11, 2147483648 // (op 2231) 
v_lshl_add_u32 v10, v5, 2, v0 // (op 2231) 


s_cmp_eq_i32 s7, 0 // (op 2231) 
s_mov_b32 s15, scc // (op 2231) 

v_lshrrev_b32 v0, 6, v1 // (op 2231) 
s_cmp_eq_i32 s15, 0 // (op 2231) 
s_cselect_b32 s7, s6, s8 // (op 2231) 



v_and_b32 v5, 1, v0 // (op 2231) 

s_mul_hi_u32 s6, s7, s12 // (op 2231) 
v_lshl_add_u32 v0, v5, 4, v10 // (op 2231) 


s_sub_u32 s8, s7, s6 // (op 2231) 
s_lshr_b32 s15, s8, 1 // (op 2231) 

s_add_u32 s8, s15, s6 // (op 2231) 


s_lshr_b32 s6, s8, s10 // (op 2231) 

s_and_b32 s8, s10, 2147483648 // (op 2231) 
s_cmp_eq_i32 s8, 0 // (op 2231) 
s_mov_b32 s15, scc // (op 2231) 

s_cmp_eq_i32 s15, 0 // (op 2231) 
s_cselect_b32 s8, s7, s6 // (op 2231) 


s_mul_i32 s6, s8, s14 // (op 2231) 
v_lshl_add_u32 v5, s8, 4, v9 // (op 2231) 


s_sub_u32 s8, s7, s6 // (op 2231) 


v_lshl_add_u32 v9, s8, 6, v0 // (op 2231) 


v_mul_lo_u32 v0, v9, s24 // (op 2231) 

v_add_lshl_u32 v9, v5, v0, 1 // (op 2231) 


s_mul_i32 s19, s24, s23 // (op 2233) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 2233) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 2233) least significant
s_mul_hi_u32 s6, s24, s22 // (op 2233) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2233) most significant: sum
s_add_u32 s19, s19, s6 // (op 2233) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 2233) 

s_mov_b32 s60, s26 // (op 2233) reg expression
s_mov_b32 s61, s27 // (op 2233) reg expression
s_mov_b32 s62, s28 // (op 2233) reg expression
s_mov_b32 s63, 131072 // (op 2233) call()

v_accvgpr_write a0, 0.00000 // (op 157) call()
v_accvgpr_write a1, 0.00000 // (op 157) call()
v_accvgpr_write a2, 0.00000 // (op 157) call()
v_accvgpr_write a3, 0.00000 // (op 157) call()
v_accvgpr_write a4, 0.00000 // (op 517) call()
v_accvgpr_write a5, 0.00000 // (op 517) call()
v_accvgpr_write a6, 0.00000 // (op 517) call()
v_accvgpr_write a7, 0.00000 // (op 517) call()
v_lshrrev_b32 v0, 7, v1 // (op 1605) 
s_mul_i32 s15, s16, s50 // (op 1605) 
s_add_u32 s15, s15, s0 // (op 1605) 
v_and_b32 v5, 15, v1 // (op 1605) 
s_add_u32 s6, s15, s9 // (op 1605) 

v_lshl_add_u32 v10, v0, 4, v5 // (op 1605) 


s_mul_hi_u32 s15, s6, s13 // (op 1605) 
v_and_b32 v0, 48, v1 // (op 1605) 
s_sub_u32 s7, s6, s15 // (op 1605) 
s_lshr_b32 s8, s7, 1 // (op 1605) 

s_add_u32 s7, s8, s15 // (op 1605) 


s_lshr_b32 s8, s7, s11 // (op 1605) 

s_and_b32 s7, s11, 2147483648 // (op 1605) 
s_cmp_eq_i32 s7, 0 // (op 1605) 
s_mov_b32 s15, scc // (op 1605) 

s_cmp_eq_i32 s15, 0 // (op 1605) 
s_cselect_b32 s7, s6, s8 // (op 1605) 


s_mul_hi_u32 s8, s7, s12 // (op 1605) 
s_sub_u32 s15, s7, s8 // (op 1605) 
s_lshr_b32 s18, s15, 1 // (op 1605) 

s_add_u32 s15, s18, s8 // (op 1605) 


s_lshr_b32 s8, s15, s10 // (op 1605) 

s_and_b32 s15, s10, 2147483648 // (op 1605) 
s_cmp_eq_i32 s15, 0 // (op 1605) 
s_mov_b32 s18, scc // (op 1605) 

s_cmp_eq_i32 s18, 0 // (op 1605) 
s_cselect_b32 s15, s7, s8 // (op 1605) 


v_lshl_add_u32 v5, s15, 4, v10 // (op 1605) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1605) 
s_mul_i32 s8, s7, s18 // (op 1605) 


s_sub_u32 s7, s6, s8 // (op 1605) 


v_lshl_add_u32 v10, s7, 7, v0 // (op 1605) 


v_mul_lo_u32 v0, v5, s30 // (op 1605) 
v_add_u32 v0, v0, v10 // (op 1605) 


s_mul_i32 s19, s30, s35 // (op 1607) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 1607) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 1607) least significant
s_mul_hi_u32 s6, s30, s34 // (op 1607) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1607) most significant: sum
s_add_u32 s19, s19, s6 // (op 1607) most significant: sum


s_mov_b32 s68, s32 // (op 1607) reg expression
s_mov_b32 s69, s33 // (op 1607) reg expression
s_mov_b32 s70, s18 // (op 1607) reg expression
s_mov_b32 s71, 131072 // (op 1607) call()

v_lshrrev_b32 v5, 7, v1 // (op 1624) 
s_mul_i32 s15, s16, s50 // (op 1624) 
s_add_u32 s15, s15, s0 // (op 1624) 
v_and_b32 v10, 15, v1 // (op 1624) 
s_add_u32 s6, s15, s9 // (op 1624) 

v_lshl_add_u32 v11, v5, 4, v10 // (op 1624) 


s_mul_hi_u32 s15, s6, s13 // (op 1624) 
v_and_b32 v5, 48, v1 // (op 1624) 
s_sub_u32 s7, s6, s15 // (op 1624) 
s_lshr_b32 s8, s7, 1 // (op 1624) 

s_add_u32 s7, s8, s15 // (op 1624) 


s_lshr_b32 s8, s7, s11 // (op 1624) 

s_and_b32 s7, s11, 2147483648 // (op 1624) 
s_cmp_eq_i32 s7, 0 // (op 1624) 
s_mov_b32 s15, scc // (op 1624) 

s_cmp_eq_i32 s15, 0 // (op 1624) 
s_cselect_b32 s7, s6, s8 // (op 1624) 


s_mul_hi_u32 s8, s7, s12 // (op 1624) 
s_sub_u32 s15, s7, s8 // (op 1624) 
s_lshr_b32 s18, s15, 1 // (op 1624) 

s_add_u32 s15, s18, s8 // (op 1624) 


s_lshr_b32 s8, s15, s10 // (op 1624) 

s_and_b32 s15, s10, 2147483648 // (op 1624) 
s_cmp_eq_i32 s15, 0 // (op 1624) 
s_mov_b32 s18, scc // (op 1624) 

s_cmp_eq_i32 s18, 0 // (op 1624) 
s_cselect_b32 s15, s7, s8 // (op 1624) 


v_lshl_add_u32 v10, s15, 4, v11 // (op 1624) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1624) 
s_mul_i32 s8, s7, s18 // (op 1624) 


s_sub_u32 s7, s6, s8 // (op 1624) 


v_lshl_add_u32 v11, s7, 7, v5 // (op 1624) 


v_mul_lo_u32 v5, v10, s30 // (op 1624) 
v_add_u32 v5, v5, v11 // (op 1624) 


s_mul_i32 s19, s30, s35 // (op 1626) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 1626) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 1626) least significant
s_mul_hi_u32 s6, s30, s34 // (op 1626) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1626) most significant: sum
s_add_u32 s19, s19, s6 // (op 1626) most significant: sum


s_mov_b32 s68, s32 // (op 1626) reg expression
s_mov_b32 s69, s33 // (op 1626) reg expression
s_mov_b32 s70, s18 // (op 1626) reg expression
s_mov_b32 s71, 131072 // (op 1626) call()

v_lshrrev_b32 v10, 6, v1 // (op 1752) 
s_mul_i32 s15, s16, s50 // (op 1752) 
s_add_u32 s15, s15, s0 // (op 1752) 
v_and_b32 v11, 1, v10 // (op 1752) 

s_add_u32 s6, s15, s9 // (op 1752) 

v_and_b32 v10, 15, v1 // (op 1752) 
s_mul_hi_u32 s15, s6, s13 // (op 1752) 
v_lshl_add_u32 v12, v11, 4, v10 // (op 1752) 


s_sub_u32 s7, s6, s15 // (op 1752) 
v_and_b32 v10, 63, v1 // (op 1752) 
s_lshr_b32 s8, s7, 1 // (op 1752) 

v_lshrrev_b32 v11, 4, v10 // (op 1752) 

s_add_u32 s7, s8, s15 // (op 1752) 


v_lshlrev_b32 v10, 5, v11 // (op 1752) 

s_lshr_b32 s8, s7, s11 // (op 1752) 

s_and_b32 s7, s11, 2147483648 // (op 1752) 
s_cmp_eq_i32 s7, 0 // (op 1752) 
s_mov_b32 s15, scc // (op 1752) 

s_cmp_eq_i32 s15, 0 // (op 1752) 
s_cselect_b32 s7, s6, s8 // (op 1752) 


s_mul_hi_u32 s8, s7, s12 // (op 1752) 
s_sub_u32 s15, s7, s8 // (op 1752) 
s_lshr_b32 s18, s15, 1 // (op 1752) 

s_add_u32 s15, s18, s8 // (op 1752) 


s_lshr_b32 s8, s15, s10 // (op 1752) 

s_and_b32 s15, s10, 2147483648 // (op 1752) 
s_cmp_eq_i32 s15, 0 // (op 1752) 
s_mov_b32 s18, scc // (op 1752) 

s_cmp_eq_i32 s18, 0 // (op 1752) 
s_cselect_b32 s15, s7, s8 // (op 1752) 


s_mul_i32 s8, s15, s14 // (op 1752) 

s_sub_u32 s15, s7, s8 // (op 1752) 

v_lshl_add_u32 v11, s15, 6, v12 // (op 1752) 


v_mul_lo_u32 v13, v11, s37 // (op 1752) most significant: low of low * high
v_mul_hi_u32 v14, v11, s36 // (op 1752) most significant: high of low * low
v_mul_lo_u32 v12, v11, s36 // (op 1752) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 1752) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1752) 
s_mul_i32 s8, s7, s18 // (op 1752) 


s_sub_u32 s7, s6, s8 // (op 1752) 


v_lshl_add_u32 v11, s7, 7, v10 // (op 1752) 


v_mov_b32 v15, 0 // (op 1752) convert
v_mov_b32 v14, v11 // (op 1752) convert

v_add_co_u32 v10, s[18:19], v14, v12 // (op 1752) least significant half
v_addc_co_u32 v11, s[18:19], v15, v13, s[18:19] // (op 1752) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 1752) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 1752) 

v_mov_b32 v10, v12 // (op 1752) convert

s_mul_i32 s19, s36, s23 // (op 1754) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 1754) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 1754) least significant
s_mul_hi_u32 s6, s36, s22 // (op 1754) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1754) most significant: sum
s_add_u32 s19, s19, s6 // (op 1754) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 1754) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 1754) 

s_mov_b32 s72, s38 // (op 1754) reg expression
s_mov_b32 s73, s39 // (op 1754) reg expression
s_mov_b32 s74, s18 // (op 1754) reg expression
s_mov_b32 s75, 131072 // (op 1754) call()

v_lshrrev_b32 v11, 6, v1 // (op 1773) 
s_mul_i32 s15, s16, s50 // (op 1773) 
s_add_u32 s15, s15, s0 // (op 1773) 
v_and_b32 v12, 1, v11 // (op 1773) 

s_add_u32 s6, s15, s9 // (op 1773) 

v_and_b32 v11, 15, v1 // (op 1773) 
s_mul_hi_u32 s15, s6, s13 // (op 1773) 
v_lshl_add_u32 v13, v12, 4, v11 // (op 1773) 


s_sub_u32 s7, s6, s15 // (op 1773) 
v_and_b32 v11, 63, v1 // (op 1773) 
s_lshr_b32 s8, s7, 1 // (op 1773) 

v_lshrrev_b32 v12, 4, v11 // (op 1773) 

s_add_u32 s7, s8, s15 // (op 1773) 


v_lshlrev_b32 v11, 5, v12 // (op 1773) 

s_lshr_b32 s8, s7, s11 // (op 1773) 

s_and_b32 s7, s11, 2147483648 // (op 1773) 
s_cmp_eq_i32 s7, 0 // (op 1773) 
s_mov_b32 s15, scc // (op 1773) 

s_cmp_eq_i32 s15, 0 // (op 1773) 
s_cselect_b32 s7, s6, s8 // (op 1773) 


s_mul_hi_u32 s8, s7, s12 // (op 1773) 
s_sub_u32 s15, s7, s8 // (op 1773) 
s_lshr_b32 s18, s15, 1 // (op 1773) 

s_add_u32 s15, s18, s8 // (op 1773) 


s_lshr_b32 s8, s15, s10 // (op 1773) 

s_and_b32 s15, s10, 2147483648 // (op 1773) 
s_cmp_eq_i32 s15, 0 // (op 1773) 
s_mov_b32 s18, scc // (op 1773) 

s_cmp_eq_i32 s18, 0 // (op 1773) 
s_cselect_b32 s15, s7, s8 // (op 1773) 


s_mul_i32 s8, s15, s14 // (op 1773) 

s_sub_u32 s15, s7, s8 // (op 1773) 

v_lshl_add_u32 v12, s15, 6, v13 // (op 1773) 


v_mul_lo_u32 v15, v12, s37 // (op 1773) most significant: low of low * high
v_mul_hi_u32 v13, v12, s36 // (op 1773) most significant: high of low * low
v_mul_lo_u32 v14, v12, s36 // (op 1773) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 1773) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1773) 
s_mul_i32 s8, s7, s18 // (op 1773) 


s_sub_u32 s7, s6, s8 // (op 1773) 


v_lshl_add_u32 v12, s7, 7, v11 // (op 1773) 


v_mov_b32 v17, 0 // (op 1773) convert
v_mov_b32 v16, v12 // (op 1773) convert

v_add_co_u32 v12, s[18:19], v16, v14 // (op 1773) least significant half
v_addc_co_u32 v13, s[18:19], v17, v15, s[18:19] // (op 1773) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 1773) 


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1773) 

v_mov_b32 v11, v14 // (op 1773) convert

s_mul_i32 s19, s36, s23 // (op 1775) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 1775) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 1775) least significant
s_mul_hi_u32 s6, s36, s22 // (op 1775) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1775) most significant: sum
s_add_u32 s19, s19, s6 // (op 1775) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 1775) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 1775) 

s_mov_b32 s72, s38 // (op 1775) reg expression
s_mov_b32 s73, s39 // (op 1775) reg expression
s_mov_b32 s74, s18 // (op 1775) reg expression
s_mov_b32 s75, 131072 // (op 1775) call()

v_lshrrev_b32 v12, 7, v1 // (op 1895) 
s_mul_i32 s15, s16, s50 // (op 1895) 
s_add_u32 s15, s15, s0 // (op 1895) 
v_and_b32 v13, 15, v1 // (op 1895) 
s_add_u32 s6, s15, s9 // (op 1895) 

v_lshl_add_u32 v14, v12, 4, v13 // (op 1895) 


s_mul_hi_u32 s15, s6, s13 // (op 1895) 
v_and_b32 v12, 63, v1 // (op 1895) 
s_sub_u32 s7, s6, s15 // (op 1895) 
v_lshrrev_b32 v13, 4, v12 // (op 1895) 

s_lshr_b32 s8, s7, 1 // (op 1895) 

s_add_u32 s7, s8, s15 // (op 1895) 


s_lshr_b32 s8, s7, s11 // (op 1895) 

s_and_b32 s7, s11, 2147483648 // (op 1895) 
s_cmp_eq_i32 s7, 0 // (op 1895) 
s_mov_b32 s15, scc // (op 1895) 

s_cmp_eq_i32 s15, 0 // (op 1895) 
s_cselect_b32 s7, s6, s8 // (op 1895) 


s_mul_hi_u32 s8, s7, s12 // (op 1895) 
s_sub_u32 s15, s7, s8 // (op 1895) 
s_lshr_b32 s18, s15, 1 // (op 1895) 

s_add_u32 s15, s18, s8 // (op 1895) 


s_lshr_b32 s8, s15, s10 // (op 1895) 

s_and_b32 s15, s10, 2147483648 // (op 1895) 
s_cmp_eq_i32 s15, 0 // (op 1895) 
s_mov_b32 s18, scc // (op 1895) 

s_cmp_eq_i32 s18, 0 // (op 1895) 
s_cselect_b32 s15, s7, s8 // (op 1895) 


v_lshl_add_u32 v12, s15, 4, v14 // (op 1895) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1895) 
s_mul_i32 s8, s7, s18 // (op 1895) 


s_sub_u32 s7, s6, s8 // (op 1895) 


v_lshl_add_u32 v14, s7, 2, v13 // (op 1895) 


v_mul_lo_u32 v13, v12, s40 // (op 1895) 
v_add_u32 v13, v13, v14 // (op 1895) 


s_mul_i32 s19, s40, s35 // (op 1897) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 1897) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 1897) least significant
s_mul_hi_u32 s6, s40, s34 // (op 1897) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1897) most significant: sum
s_add_u32 s19, s19, s6 // (op 1897) most significant: sum


s_mov_b32 s76, s42 // (op 1897) reg expression
s_mov_b32 s77, s43 // (op 1897) reg expression
s_mov_b32 s78, s18 // (op 1897) reg expression
s_mov_b32 s79, 131072 // (op 1897) call()

v_lshrrev_b32 v12, 7, v1 // (op 1914) 
s_mul_i32 s15, s16, s50 // (op 1914) 
s_add_u32 s15, s15, s0 // (op 1914) 
v_and_b32 v14, 15, v1 // (op 1914) 
s_add_u32 s6, s15, s9 // (op 1914) 

v_lshl_add_u32 v15, v12, 4, v14 // (op 1914) 


s_mul_hi_u32 s15, s6, s13 // (op 1914) 
v_and_b32 v12, 63, v1 // (op 1914) 
s_sub_u32 s7, s6, s15 // (op 1914) 
v_lshrrev_b32 v14, 4, v12 // (op 1914) 

s_lshr_b32 s8, s7, 1 // (op 1914) 

s_add_u32 s7, s8, s15 // (op 1914) 


s_lshr_b32 s8, s7, s11 // (op 1914) 

s_and_b32 s7, s11, 2147483648 // (op 1914) 
s_cmp_eq_i32 s7, 0 // (op 1914) 
s_mov_b32 s15, scc // (op 1914) 

s_cmp_eq_i32 s15, 0 // (op 1914) 
s_cselect_b32 s7, s6, s8 // (op 1914) 


s_mul_hi_u32 s8, s7, s12 // (op 1914) 
s_sub_u32 s15, s7, s8 // (op 1914) 
s_lshr_b32 s18, s15, 1 // (op 1914) 

s_add_u32 s15, s18, s8 // (op 1914) 


s_lshr_b32 s8, s15, s10 // (op 1914) 

s_and_b32 s15, s10, 2147483648 // (op 1914) 
s_cmp_eq_i32 s15, 0 // (op 1914) 
s_mov_b32 s18, scc // (op 1914) 

s_cmp_eq_i32 s18, 0 // (op 1914) 
s_cselect_b32 s15, s7, s8 // (op 1914) 


v_lshl_add_u32 v12, s15, 4, v15 // (op 1914) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1914) 
s_mul_i32 s8, s7, s18 // (op 1914) 


s_sub_u32 s7, s6, s8 // (op 1914) 


v_lshl_add_u32 v15, s7, 2, v14 // (op 1914) 


v_mul_lo_u32 v14, v12, s40 // (op 1914) 
v_add_u32 v14, v14, v15 // (op 1914) 


s_mul_i32 s19, s40, s35 // (op 1916) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 1916) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 1916) least significant
s_mul_hi_u32 s6, s40, s34 // (op 1916) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1916) most significant: sum
s_add_u32 s19, s19, s6 // (op 1916) most significant: sum


s_mov_b32 s76, s42 // (op 1916) reg expression
s_mov_b32 s77, s43 // (op 1916) reg expression
s_mov_b32 s78, s18 // (op 1916) reg expression
s_mov_b32 s79, 131072 // (op 1916) call()

v_lshrrev_b32 v12, 2, v1 // (op 2023) 
s_mul_i32 s15, s16, s50 // (op 2023) 
s_add_u32 s15, s15, s0 // (op 2023) 
v_lshlrev_b32 v15, 1, v12 // (op 2023) 

s_add_u32 s6, s15, s9 // (op 2023) 

v_and_b32 v12, -4, v1 // (op 2023) 
s_mul_hi_u32 s15, s6, s13 // (op 2023) 
v_sub_u32 v16, v1, v12 // (op 2023) 

s_sub_u32 s7, s6, s15 // (op 2023) 
s_lshr_b32 s8, s7, 1 // (op 2023) 

s_add_u32 s7, s8, s15 // (op 2023) 


s_lshr_b32 s8, s7, s11 // (op 2023) 

s_and_b32 s7, s11, 2147483648 // (op 2023) 
s_cmp_eq_i32 s7, 0 // (op 2023) 
s_mov_b32 s15, scc // (op 2023) 

s_cmp_eq_i32 s15, 0 // (op 2023) 
s_cselect_b32 s7, s6, s8 // (op 2023) 


s_mul_hi_u32 s8, s7, s12 // (op 2023) 
s_sub_u32 s15, s7, s8 // (op 2023) 
s_lshr_b32 s18, s15, 1 // (op 2023) 

s_add_u32 s15, s18, s8 // (op 2023) 


s_lshr_b32 s8, s15, s10 // (op 2023) 

s_and_b32 s15, s10, 2147483648 // (op 2023) 
s_cmp_eq_i32 s15, 0 // (op 2023) 
s_mov_b32 s18, scc // (op 2023) 

s_cmp_eq_i32 s18, 0 // (op 2023) 
s_cselect_b32 s15, s7, s8 // (op 2023) 


s_mul_i32 s8, s15, s14 // (op 2023) 

s_sub_u32 s15, s7, s8 // (op 2023) 

v_lshl_add_u32 v12, s15, 6, v15 // (op 2023) 


v_mul_lo_u32 v15, v12, s44 // (op 2023) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 2023) 
s_mul_i32 s8, s7, s18 // (op 2023) 


s_sub_u32 s7, s6, s8 // (op 2023) 


v_lshl_add_u32 v12, s7, 2, v16 // (op 2023) 


v_add_u32 v16, v12, v15 // (op 2023) 


s_mul_i32 s19, s44, s23 // (op 2025) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 2025) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 2025) least significant
s_mul_hi_u32 s6, s44, s22 // (op 2025) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2025) most significant: sum
s_add_u32 s19, s19, s6 // (op 2025) most significant: sum


s_mov_b32 s80, s46 // (op 2025) reg expression
s_mov_b32 s81, s47 // (op 2025) reg expression
s_mov_b32 s82, s18 // (op 2025) reg expression
s_mov_b32 s83, 131072 // (op 2025) call()

v_lshrrev_b32 v12, 2, v1 // (op 2042) 
s_mul_i32 s15, s16, s50 // (op 2042) 
s_add_u32 s15, s15, s0 // (op 2042) 
v_lshlrev_b32 v15, 1, v12 // (op 2042) 

s_add_u32 s6, s15, s9 // (op 2042) 

v_and_b32 v12, -4, v1 // (op 2042) 
s_mul_hi_u32 s15, s6, s13 // (op 2042) 
v_sub_u32 v17, v1, v12 // (op 2042) 

s_sub_u32 s7, s6, s15 // (op 2042) 
s_lshr_b32 s8, s7, 1 // (op 2042) 

s_add_u32 s7, s8, s15 // (op 2042) 


s_lshr_b32 s8, s7, s11 // (op 2042) 

s_and_b32 s7, s11, 2147483648 // (op 2042) 
s_cmp_eq_i32 s7, 0 // (op 2042) 
s_mov_b32 s15, scc // (op 2042) 

s_cmp_eq_i32 s15, 0 // (op 2042) 
s_cselect_b32 s7, s6, s8 // (op 2042) 


s_mul_hi_u32 s8, s7, s12 // (op 2042) 
s_sub_u32 s15, s7, s8 // (op 2042) 
s_lshr_b32 s18, s15, 1 // (op 2042) 

s_add_u32 s15, s18, s8 // (op 2042) 


s_lshr_b32 s8, s15, s10 // (op 2042) 

s_and_b32 s15, s10, 2147483648 // (op 2042) 
s_cmp_eq_i32 s15, 0 // (op 2042) 
s_mov_b32 s18, scc // (op 2042) 

s_cmp_eq_i32 s18, 0 // (op 2042) 
s_cselect_b32 s15, s7, s8 // (op 2042) 


s_mul_i32 s8, s15, s14 // (op 2042) 

s_sub_u32 s15, s7, s8 // (op 2042) 

v_lshl_add_u32 v12, s15, 6, v15 // (op 2042) 


v_mul_lo_u32 v15, v12, s44 // (op 2042) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 2042) 
s_mul_i32 s8, s7, s18 // (op 2042) 


s_sub_u32 s7, s6, s8 // (op 2042) 


v_lshl_add_u32 v12, s7, 2, v17 // (op 2042) 


v_add_u32 v17, v12, v15 // (op 2042) 


s_mul_i32 s19, s44, s23 // (op 2044) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 2044) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 2044) least significant
s_mul_hi_u32 s6, s44, s22 // (op 2044) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2044) most significant: sum
s_add_u32 s19, s19, s6 // (op 2044) most significant: sum


s_mov_b32 s80, s46 // (op 2044) reg expression
s_mov_b32 s81, s47 // (op 2044) reg expression
s_mov_b32 s82, s18 // (op 2044) reg expression
s_mov_b32 s83, 131072 // (op 2044) call()




buffer_load_dwordx4 v[18:21], v0, s[68:71], 0 offen // (op 1337) Load value
buffer_load_dwordx4 v[22:25], v0, s[68:71], 0 offen offset:64 // (op 1337) Load value


v_mov_b32 v12, v10 // (op 1343) reg expression

buffer_load_dwordx4 v[26:29], v12, s[72:75], 0 offen // (op 1343) Load value




buffer_load_ubyte v12, v13, s[76:79], 0 offen offset: 0 // (op 1349) Load value


s_lshl_b64 s[18:19], s[36:37], 5 // (op 1355) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 1355) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 1355) 

v_add_u32 v15, v10, s18 // (op 1355) 


buffer_load_dwordx4 v[30:33], v15, s[72:75], 0 offen // (op 1355) Load value


v_mov_b64 v[34:35], s[44:45] // (op 1360) convert

v_mov_b32 v15, v16 // (op 1360) 
buffer_load_ubyte v36, v15, s[80:83], 0 offen offset: 0 // (op 1360) Load value
v_add_u32 v15, v15, v34 // (op 1360) 
buffer_load_ubyte v37, v15, s[80:83], 0 offen offset: 0 // (op 1360) Load value





buffer_load_dwordx4 v[38:41], v0, s[68:71], 0 offen offset:128 // (op 1364) Load value
buffer_load_dwordx4 v[42:45], v0, s[68:71], 0 offen offset:192 // (op 1364) Load value



v_add_u32 v0, 64, v10 // (op 1370) 

buffer_load_dwordx4 v[46:49], v0, s[72:75], 0 offen // (op 1370) Load value




buffer_load_ubyte v0, v13, s[76:79], 0 offen offset:4 // (op 1376) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 1382) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 1382) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 1382) 

s_add_u32 s15, s18, 64 // (op 1382) 

v_add_u32 v13, v10, s15 // (op 1382) 


buffer_load_dwordx4 v[50:53], v13, s[72:75], 0 offen // (op 1382) Load value



v_mov_b64 v[34:35], s[44:45] // (op 1387) convert

v_mov_b32 v10, v16 // (op 1387) 
buffer_load_ubyte v54, v10, s[80:83], 0 offen offset:4 // (op 1387) Load value
v_add_u32 v10, v10, v34 // (op 1387) 
buffer_load_ubyte v55, v10, s[80:83], 0 offen offset:4 // (op 1387) Load value



v_and_b32 v10, -4, v1 // (op 2277) 
v_sub_u32 v13, v1, v10 // (op 2277) 

v_lshrrev_b32 v10, 2, v1 // (op 2277) 
v_lshl_add_u32 v15, v10, 3, v13 // (op 2277) 


s_barrier  // (op 2481) 



s_waitcnt vmcnt(8)
ds_write_b8 v15, v36 // (op 1389) Store local data 
s_waitcnt vmcnt(7)
ds_write_b8 v15, v37 offset:4 // (op 1389) Store local data 

v_and_b32 v10, -4, v1 // (op 2248) 
v_sub_u32 v13, v1, v10 // (op 2248) 

v_lshrrev_b32 v10, 2, v1 // (op 2248) 
v_lshl_add_u32 v15, v10, 3, v13 // (op 2248) 


v_and_b32 v10, -4, v1 // (op 2348) 
v_sub_u32 v13, v1, v10 // (op 2348) 

v_lshrrev_b32 v10, 2, v1 // (op 2348) 
v_lshl_add_u32 v16, v10, 3, v13 // (op 2348) 


s_mov_b32 s1, 0 // (op 519) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 518) 
s_mov_b32 s15, s18 // (op 518) convert

s_and_b32 s6, s15, 4294967294 // (op 518) 

s_cmp_lt_u32 s1, s6 // (op 518) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_518 // (op 518) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_518 if false)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoop_518:
s_add_u32 s1, s1, 2 // (op 520) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2478) 



ds_read_u8 v10, v3 // (op 164) Load local data 



ds_read_u8 v13, v3 offset:128 // (op 544) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:25], v[26:29], a[0:3], v12, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 165) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:25], v[30:33], a[4:7], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 545) 



v_mov_b32 v10, v5 // (op 167) 

buffer_load_dwordx4 v[18:21], v10, s[68:71], 0 offen offset:256 // (op 167) Load value
buffer_load_dwordx4 v[22:25], v10, s[68:71], 0 offen offset:320 // (op 167) Load value


v_mov_b32 v13, v11 // (op 169) 
v_add_u32 v34, 128, v13 // (op 169) 

buffer_load_dwordx4 v[26:29], v34, s[72:75], 0 offen // (op 169) Load value



v_mov_b32 v34, v14 // (op 171) 

buffer_load_ubyte v12, v34, s[76:79], 0 offen offset:8 // (op 171) Load value


s_lshl_b64 s[18:19], s[36:37], 5 // (op 567) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 567) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 567) 

s_add_u32 s15, s18, 128 // (op 567) 

v_add_u32 v35, v13, s15 // (op 567) 


buffer_load_dwordx4 v[30:33], v35, s[72:75], 0 offen // (op 567) Load value


v_mov_b64 v[56:57], s[44:45] // (op 161) convert

v_mov_b32 v35, v17 // (op 161) 
v_mov_b32 v58, v35 // (op 161) 
buffer_load_ubyte v36, v58, s[80:83], 0 offen offset:8 // (op 161) Load value
v_add_u32 v58, v58, v56 // (op 161) 
buffer_load_ubyte v37, v58, s[80:83], 0 offen offset:8 // (op 161) Load value


s_barrier  // (op 2497) 



s_waitcnt vmcnt(8)
ds_write_b8 v16, v54 offset:256 // (op 460) Store local data 
s_waitcnt vmcnt(7)
ds_write_b8 v16, v55 offset:260 // (op 460) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2494) 



ds_read_u8 v56, v4 offset:256 // (op 462) Load local data 



ds_read_u8 v57, v4 offset:384 // (op 548) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[46:49], a[0:3], v0, v56 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 463) 

s_barrier  // (op 2600) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:45], v[50:53], a[4:7], v0, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 546) 




buffer_load_dwordx4 v[38:41], v10, s[68:71], 0 offen offset:384 // (op 465) Load value
buffer_load_dwordx4 v[42:45], v10, s[68:71], 0 offen offset:448 // (op 465) Load value



v_add_u32 v5, 256, v5 // (op 1616) 
v_add_u32 v10, 192, v13 // (op 467) 

buffer_load_dwordx4 v[46:49], v10, s[72:75], 0 offen // (op 467) Load value




buffer_load_ubyte v0, v34, s[76:79], 0 offen offset:12 // (op 469) Load value



v_add_u32 v14, 8, v14 // (op 1906) 
s_lshl_b64 s[18:19], s[36:37], 5 // (op 562) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 562) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 562) 

s_add_u32 s15, s18, 192 // (op 562) 

v_add_u32 v10, v13, s15 // (op 562) 


buffer_load_dwordx4 v[50:53], v10, s[72:75], 0 offen // (op 562) Load value



v_add_u32 v11, 128, v11 // (op 1763) 
v_mov_b64 v[56:57], s[44:45] // (op 459) convert

v_mov_b32 v10, v35 // (op 459) 
buffer_load_ubyte v54, v10, s[80:83], 0 offen offset:12 // (op 459) Load value
v_add_u32 v10, v10, v56 // (op 459) 
buffer_load_ubyte v55, v10, s[80:83], 0 offen offset:12 // (op 459) Load value



s_barrier  // (op 2484) 
v_add_u32 v17, 8, v17 // (op 2034) 



s_waitcnt vmcnt(8)
ds_write_b8 v15, v36 // (op 162) Store local data 
s_waitcnt vmcnt(7)
ds_write_b8 v15, v37 offset:4 // (op 162) Store local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 518) 
s_mov_b32 s15, s18 // (op 518) convert

s_and_b32 s6, s15, 4294967294 // (op 518) 

s_cmp_lt_u32 s1, s6 // (op 518) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoop_518 // (op 518) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoop_518 if true)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_518:











v_lshrrev_b32 v5, 7, v1 // (op 1554) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1554) 
v_and_b32 v10, 15, v1 // (op 1554) 
s_and_b32 s15, s18, 4294967294 // (op 1554) 
v_lshl_add_u32 v11, v5, 4, v10 // (op 1554) 


s_add_u32 s6, s0, s15 // (op 1554) 

v_and_b32 v5, 48, v1 // (op 1554) 
s_mul_i32 s15, s16, s50 // (op 1554) 
s_add_u32 s15, s15, s6 // (op 1554) 

s_add_u32 s6, s15, s9 // (op 1554) 

s_mul_hi_u32 s15, s6, s13 // (op 1554) 
s_sub_u32 s7, s6, s15 // (op 1554) 
s_lshr_b32 s8, s7, 1 // (op 1554) 

s_add_u32 s7, s8, s15 // (op 1554) 


s_lshr_b32 s8, s7, s11 // (op 1554) 

s_and_b32 s7, s11, 2147483648 // (op 1554) 
s_cmp_eq_i32 s7, 0 // (op 1554) 
s_mov_b32 s15, scc // (op 1554) 

s_cmp_eq_i32 s15, 0 // (op 1554) 
s_cselect_b32 s7, s6, s8 // (op 1554) 


s_mul_hi_u32 s8, s7, s12 // (op 1554) 
s_sub_u32 s15, s7, s8 // (op 1554) 
s_lshr_b32 s28, s15, 1 // (op 1554) 

s_add_u32 s15, s28, s8 // (op 1554) 


s_lshr_b32 s8, s15, s10 // (op 1554) 

s_and_b32 s15, s10, 2147483648 // (op 1554) 
s_cmp_eq_i32 s15, 0 // (op 1554) 
s_mov_b32 s28, scc // (op 1554) 

s_cmp_eq_i32 s28, 0 // (op 1554) 
s_cselect_b32 s15, s7, s8 // (op 1554) 


s_mul_i32 s8, s7, s18 // (op 1554) 


v_lshl_add_u32 v10, s15, 4, v11 // (op 1554) 


s_sub_u32 s7, s6, s8 // (op 1554) 


v_lshl_add_u32 v11, s7, 7, v5 // (op 1554) 


v_mul_lo_u32 v5, v10, s30 // (op 1554) 
v_add_u32 v5, v5, v11 // (op 1554) 


s_mul_i32 s19, s30, s35 // (op 1556) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 1556) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 1556) least significant
s_mul_hi_u32 s6, s30, s34 // (op 1556) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1556) most significant: sum
s_add_u32 s19, s19, s6 // (op 1556) most significant: sum


s_mov_b32 s68, s32 // (op 1556) reg expression
s_mov_b32 s69, s33 // (op 1556) reg expression
s_mov_b32 s70, s18 // (op 1556) reg expression
s_mov_b32 s71, 131072 // (op 1556) call()

v_lshrrev_b32 v10, 6, v1 // (op 1694) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1694) 
v_and_b32 v11, 1, v10 // (op 1694) 

s_and_b32 s15, s18, 4294967294 // (op 1694) 
v_and_b32 v10, 15, v1 // (op 1694) 
s_add_u32 s6, s0, s15 // (op 1694) 

v_lshl_add_u32 v12, v11, 4, v10 // (op 1694) 


v_and_b32 v10, 63, v1 // (op 1694) 
s_mul_i32 s15, s16, s50 // (op 1694) 
s_add_u32 s15, s15, s6 // (op 1694) 

v_lshrrev_b32 v11, 4, v10 // (op 1694) 

s_add_u32 s6, s15, s9 // (op 1694) 

v_lshlrev_b32 v10, 5, v11 // (op 1694) 

s_mul_hi_u32 s15, s6, s13 // (op 1694) 
s_sub_u32 s7, s6, s15 // (op 1694) 
s_lshr_b32 s8, s7, 1 // (op 1694) 

s_add_u32 s7, s8, s15 // (op 1694) 


s_lshr_b32 s8, s7, s11 // (op 1694) 

s_and_b32 s7, s11, 2147483648 // (op 1694) 
s_cmp_eq_i32 s7, 0 // (op 1694) 
s_mov_b32 s15, scc // (op 1694) 

s_cmp_eq_i32 s15, 0 // (op 1694) 
s_cselect_b32 s7, s6, s8 // (op 1694) 


s_mul_hi_u32 s8, s7, s12 // (op 1694) 
s_sub_u32 s15, s7, s8 // (op 1694) 
s_lshr_b32 s28, s15, 1 // (op 1694) 

s_add_u32 s15, s28, s8 // (op 1694) 


s_lshr_b32 s8, s15, s10 // (op 1694) 

s_and_b32 s15, s10, 2147483648 // (op 1694) 
s_cmp_eq_i32 s15, 0 // (op 1694) 
s_mov_b32 s28, scc // (op 1694) 

s_cmp_eq_i32 s28, 0 // (op 1694) 
s_cselect_b32 s15, s7, s8 // (op 1694) 


s_mul_i32 s8, s15, s14 // (op 1694) 

s_sub_u32 s15, s7, s8 // (op 1694) 

s_mul_i32 s8, s7, s18 // (op 1694) 


v_lshl_add_u32 v11, s15, 6, v12 // (op 1694) 


s_sub_u32 s7, s6, s8 // (op 1694) 


v_mul_lo_u32 v13, v11, s37 // (op 1694) most significant: low of low * high
v_mul_hi_u32 v14, v11, s36 // (op 1694) most significant: high of low * low
v_mul_lo_u32 v12, v11, s36 // (op 1694) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 1694) most significant: sum


v_lshl_add_u32 v11, s7, 7, v10 // (op 1694) 


v_mov_b32 v15, 0 // (op 1694) convert
v_mov_b32 v14, v11 // (op 1694) convert

v_add_co_u32 v10, s[18:19], v14, v12 // (op 1694) least significant half
v_addc_co_u32 v11, s[18:19], v15, v13, s[18:19] // (op 1694) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 1694) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 1694) 

v_mov_b32 v10, v12 // (op 1694) convert

s_mul_i32 s19, s36, s23 // (op 1696) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 1696) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 1696) least significant
s_mul_hi_u32 s6, s36, s22 // (op 1696) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1696) most significant: sum
s_add_u32 s19, s19, s6 // (op 1696) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 1696) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 1696) 

s_mov_b32 s72, s38 // (op 1696) reg expression
s_mov_b32 s73, s39 // (op 1696) reg expression
s_mov_b32 s74, s18 // (op 1696) reg expression
s_mov_b32 s75, 131072 // (op 1696) call()

v_lshrrev_b32 v11, 7, v1 // (op 1850) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1850) 
v_and_b32 v12, 15, v1 // (op 1850) 
s_and_b32 s15, s18, 4294967294 // (op 1850) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 1850) 


s_add_u32 s6, s0, s15 // (op 1850) 

v_and_b32 v11, 63, v1 // (op 1850) 
v_lshrrev_b32 v12, 4, v11 // (op 1850) 

s_mul_i32 s15, s16, s50 // (op 1850) 
s_add_u32 s15, s15, s6 // (op 1850) 

s_add_u32 s6, s15, s9 // (op 1850) 

s_mul_hi_u32 s15, s6, s13 // (op 1850) 
s_sub_u32 s7, s6, s15 // (op 1850) 
s_lshr_b32 s8, s7, 1 // (op 1850) 

s_add_u32 s7, s8, s15 // (op 1850) 


s_lshr_b32 s8, s7, s11 // (op 1850) 

s_and_b32 s7, s11, 2147483648 // (op 1850) 
s_cmp_eq_i32 s7, 0 // (op 1850) 
s_mov_b32 s15, scc // (op 1850) 

s_cmp_eq_i32 s15, 0 // (op 1850) 
s_cselect_b32 s7, s6, s8 // (op 1850) 


s_mul_hi_u32 s8, s7, s12 // (op 1850) 
s_sub_u32 s15, s7, s8 // (op 1850) 
s_lshr_b32 s28, s15, 1 // (op 1850) 

s_add_u32 s15, s28, s8 // (op 1850) 


s_lshr_b32 s8, s15, s10 // (op 1850) 

s_and_b32 s15, s10, 2147483648 // (op 1850) 
s_cmp_eq_i32 s15, 0 // (op 1850) 
s_mov_b32 s28, scc // (op 1850) 

s_cmp_eq_i32 s28, 0 // (op 1850) 
s_cselect_b32 s15, s7, s8 // (op 1850) 


s_mul_i32 s8, s7, s18 // (op 1850) 


v_lshl_add_u32 v11, s15, 4, v13 // (op 1850) 


s_sub_u32 s7, s6, s8 // (op 1850) 


v_lshl_add_u32 v13, s7, 2, v12 // (op 1850) 


v_mul_lo_u32 v12, v11, s40 // (op 1850) 
v_add_u32 v12, v12, v13 // (op 1850) 


s_mul_i32 s19, s40, s35 // (op 1852) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 1852) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 1852) least significant
s_mul_hi_u32 s6, s40, s34 // (op 1852) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1852) most significant: sum
s_add_u32 s19, s19, s6 // (op 1852) most significant: sum


s_mov_b32 s76, s42 // (op 1852) reg expression
s_mov_b32 s77, s43 // (op 1852) reg expression
s_mov_b32 s78, s18 // (op 1852) reg expression
s_mov_b32 s79, 131072 // (op 1852) call()

v_lshrrev_b32 v11, 2, v1 // (op 1978) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1978) 
v_lshlrev_b32 v13, 1, v11 // (op 1978) 

s_and_b32 s15, s18, 4294967294 // (op 1978) 
v_and_b32 v11, -4, v1 // (op 1978) 
s_add_u32 s6, s0, s15 // (op 1978) 

v_sub_u32 v14, v1, v11 // (op 1978) 

s_mul_i32 s15, s16, s50 // (op 1978) 
s_add_u32 s15, s15, s6 // (op 1978) 

s_add_u32 s6, s15, s9 // (op 1978) 

s_mul_hi_u32 s15, s6, s13 // (op 1978) 
s_sub_u32 s7, s6, s15 // (op 1978) 
s_lshr_b32 s8, s7, 1 // (op 1978) 

s_add_u32 s7, s8, s15 // (op 1978) 


s_lshr_b32 s8, s7, s11 // (op 1978) 

s_and_b32 s7, s11, 2147483648 // (op 1978) 
s_cmp_eq_i32 s7, 0 // (op 1978) 
s_mov_b32 s15, scc // (op 1978) 

s_cmp_eq_i32 s15, 0 // (op 1978) 
s_cselect_b32 s7, s6, s8 // (op 1978) 


s_mul_hi_u32 s8, s7, s12 // (op 1978) 
s_sub_u32 s15, s7, s8 // (op 1978) 
s_lshr_b32 s28, s15, 1 // (op 1978) 

s_add_u32 s15, s28, s8 // (op 1978) 


s_lshr_b32 s8, s15, s10 // (op 1978) 

s_and_b32 s15, s10, 2147483648 // (op 1978) 
s_cmp_eq_i32 s15, 0 // (op 1978) 
s_mov_b32 s28, scc // (op 1978) 

s_cmp_eq_i32 s28, 0 // (op 1978) 
s_cselect_b32 s15, s7, s8 // (op 1978) 


s_mul_i32 s8, s15, s14 // (op 1978) 

s_sub_u32 s15, s7, s8 // (op 1978) 

v_lshl_add_u32 v11, s15, 6, v13 // (op 1978) 


s_mul_i32 s8, s7, s18 // (op 1978) 


v_mul_lo_u32 v13, v11, s44 // (op 1978) 

s_sub_u32 s7, s6, s8 // (op 1978) 


v_lshl_add_u32 v11, s7, 2, v14 // (op 1978) 


v_add_u32 v14, v11, v13 // (op 1978) 


s_mul_i32 s19, s44, s23 // (op 1980) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 1980) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 1980) least significant
s_mul_hi_u32 s6, s44, s22 // (op 1980) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1980) most significant: sum
s_add_u32 s19, s19, s6 // (op 1980) most significant: sum


s_mov_b32 s80, s46 // (op 1980) reg expression
s_mov_b32 s81, s47 // (op 1980) reg expression
s_mov_b32 s82, s18 // (op 1980) reg expression
s_mov_b32 s83, 131072 // (op 1980) call()

v_and_b32 v11, -4, v1 // (op 2261) 
v_sub_u32 v13, v1, v11 // (op 2261) 

v_lshrrev_b32 v11, 2, v1 // (op 2261) 
v_lshl_add_u32 v15, v11, 3, v13 // (op 2261) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 524) 
s_and_b32 s15, s18, 4294967294 // (op 524) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 523) 
s_mov_b32 s6, s18 // (op 523) convert

s_cmp_lt_u32 s15, s6 // (op 523) 






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_523 // (op 523) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_523 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_523:

s_add_u32 s15, s15, 1 // (op 525) 
v_mov_b64 v[16:17], s[44:45] // (op 417) convert

v_mov_b32 v0, v14 // (op 417) 
v_mov_b32 v11, v0 // (op 417) 
buffer_load_ubyte v18, v11, s[80:83], 0 offen offset: 0 // (op 417) Load value
v_add_u32 v11, v11, v16 // (op 417) 
buffer_load_ubyte v19, v11, s[80:83], 0 offen offset: 0 // (op 417) Load value



v_add_u32 v14, 4, v14 // (op 1970) 


s_barrier  // (op 2487) 
v_mov_b32 v0, v5 // (op 423) 




buffer_load_dwordx4 v[20:23], v0, s[68:71], 0 offen // (op 423) Load value
s_waitcnt vmcnt(2)
ds_write_b8 v15, v18 // (op 418) Store local data 
buffer_load_dwordx4 v[24:27], v0, s[68:71], 0 offen offset:64 // (op 423) Load value

v_add_u32 v5, 128, v5 // (op 1544) 
s_waitcnt vmcnt(2)
ds_write_b8 v15, v19 offset:4 // (op 418) Store local data 



v_mov_b32 v0, v10 // (op 425) 
v_mov_b32 v11, v0 // (op 425) reg expression

buffer_load_dwordx4 v[16:19], v11, s[72:75], 0 offen // (op 425) Load value



v_mov_b32 v11, v12 // (op 427) 

buffer_load_ubyte v13, v11, s[76:79], 0 offen offset: 0 // (op 427) Load value

v_add_u32 v12, 4, v12 // (op 1842) 


s_lshl_b64 s[18:19], s[36:37], 5 // (op 537) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 537) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 537) 

v_add_u32 v11, v0, s18 // (op 537) 


s_waitcnt lgkmcnt(0)
s_barrier  // (op 2490) 
buffer_load_dwordx4 v[28:31], v11, s[72:75], 0 offen // (op 537) Load value





ds_read_u8 v0, v3 // (op 420) Load local data 



v_add_u32 v10, 64, v10 // (op 1684) 
ds_read_u8 v11, v3 offset:128 // (op 529) Load local data 
s_waitcnt vmcnt(1) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:27], v[16:19], a[0:3], v13, v0 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 421) 


s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:27], v[28:31], a[4:7], v13, v11 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 530) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 523) 
s_mov_b32 s6, s18 // (op 523) convert

s_cmp_lt_u32 s15, s6 // (op 523) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_523 // (op 523) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_523 if true)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_523:






s_barrier  // (op 2586) 
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 193) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_193 // (op 193) Condition: False, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_193
v_accvgpr_read v10, a0 // (op 201) 
v_accvgpr_read v11, a1 // (op 201) 
v_accvgpr_read v12, a2 // (op 201) 
v_accvgpr_read v13, a3 // (op 201) 
v_mul_f32 v14, v7, v10 // (op 201) 
v_mul_f32 v15, v7, v11 // (op 201) 
v_mul_f32 v16, v7, v12 // (op 201) 
v_mul_f32 v17, v7, v13 // (op 201) 

v_cvt_f16_f32 v10, v14 // (op 203) 
v_cvt_f16_f32 v11, v15 // (op 203) 
v_cvt_f16_f32 v12, v16 // (op 203) 
v_cvt_f16_f32 v13, v17 // (op 203) 


v_pack_b32_f16 v14, v10, v11 // (op 204) 
v_pack_b32_f16 v15, v12, v13 // (op 204) 

v_mov_b32 v0, v9 // (op 204) reg expression

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 204) Store value



v_accvgpr_read v10, a4 // (op 675) 
v_accvgpr_read v11, a5 // (op 675) 
v_accvgpr_read v12, a6 // (op 675) 
v_accvgpr_read v13, a7 // (op 675) 
v_mul_f32 v14, v7, v10 // (op 675) 
v_mul_f32 v15, v7, v11 // (op 675) 
v_mul_f32 v16, v7, v12 // (op 675) 
v_mul_f32 v17, v7, v13 // (op 675) 

v_cvt_f16_f32 v10, v14 // (op 677) 
v_cvt_f16_f32 v11, v15 // (op 677) 
v_cvt_f16_f32 v12, v16 // (op 677) 
v_cvt_f16_f32 v13, v17 // (op 677) 


v_pack_b32_f16 v14, v10, v11 // (op 678) 
v_pack_b32_f16 v15, v12, v13 // (op 678) 

s_lshl_b32 s15, s24, 6 // (op 678) 
v_add_u32 v0, v9, s15 // (op 678) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 678) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_193 // (op 193) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_193
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_193:


v_mov_b32 v0, v8 // (op 218) reg expression

buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 218) Load value

v_accvgpr_read v12, a0 // (op 214) 
v_accvgpr_read v13, a1 // (op 214) 
v_accvgpr_read v14, a2 // (op 214) 
v_accvgpr_read v15, a3 // (op 214) 
v_mul_f32 v16, v7, v12 // (op 214) 
v_mul_f32 v17, v7, v13 // (op 214) 
v_mul_f32 v18, v7, v14 // (op 214) 
v_mul_f32 v19, v7, v15 // (op 214) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 219) 
v_lshrrev_b32 v13, 16, v10 // (op 219) 
v_cvt_f32_f16 v13, v13 // (op 219) 
v_mul_f32 v20, v6, v12 // (op 219) 
v_mul_f32 v21, v6, v13 // (op 219) 
v_cvt_f32_f16 v12, v11 // (op 219) 
v_lshrrev_b32 v13, 16, v11 // (op 219) 
v_cvt_f32_f16 v13, v13 // (op 219) 
v_mul_f32 v22, v6, v12 // (op 219) 
v_mul_f32 v23, v6, v13 // (op 219) 


v_add_f32 v10, v20, v16 // (op 215) 
v_add_f32 v11, v21, v17 // (op 215) 
v_add_f32 v12, v22, v18 // (op 215) 
v_add_f32 v13, v23, v19 // (op 215) 


v_cvt_f16_f32 v14, v10 // (op 216) 
v_cvt_f16_f32 v15, v11 // (op 216) 
v_cvt_f16_f32 v16, v12 // (op 216) 
v_cvt_f16_f32 v17, v13 // (op 216) 


v_pack_b32_f16 v10, v14, v15 // (op 217) 
v_pack_b32_f16 v11, v16, v17 // (op 217) 

v_mov_b32 v0, v9 // (op 217) reg expression

buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 217) Store value





s_lshl_b32 s15, s4, 6 // (op 702) 
v_add_u32 v0, v8, s15 // (op 702) 


buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 702) Load value

v_accvgpr_read v12, a4 // (op 698) 
v_accvgpr_read v13, a5 // (op 698) 
v_accvgpr_read v14, a6 // (op 698) 
v_accvgpr_read v15, a7 // (op 698) 
v_mul_f32 v16, v7, v12 // (op 698) 
v_mul_f32 v17, v7, v13 // (op 698) 
v_mul_f32 v18, v7, v14 // (op 698) 
v_mul_f32 v19, v7, v15 // (op 698) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 703) 
v_lshrrev_b32 v13, 16, v10 // (op 703) 
v_cvt_f32_f16 v13, v13 // (op 703) 
v_mul_f32 v20, v6, v12 // (op 703) 
v_mul_f32 v21, v6, v13 // (op 703) 
v_cvt_f32_f16 v12, v11 // (op 703) 
v_lshrrev_b32 v13, 16, v11 // (op 703) 
v_cvt_f32_f16 v13, v13 // (op 703) 
v_mul_f32 v22, v6, v12 // (op 703) 
v_mul_f32 v23, v6, v13 // (op 703) 


v_add_f32 v10, v20, v16 // (op 699) 
v_add_f32 v11, v21, v17 // (op 699) 
v_add_f32 v12, v22, v18 // (op 699) 
v_add_f32 v13, v23, v19 // (op 699) 


v_cvt_f16_f32 v14, v10 // (op 700) 
v_cvt_f16_f32 v15, v11 // (op 700) 
v_cvt_f16_f32 v16, v12 // (op 700) 
v_cvt_f16_f32 v17, v13 // (op 700) 


v_pack_b32_f16 v10, v14, v15 // (op 701) 
v_pack_b32_f16 v11, v16, v17 // (op 701) 

s_lshl_b32 s15, s24, 6 // (op 701) 
v_add_u32 v0, v9, s15 // (op 701) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 701) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_193:
s_add_u32 s0, s0, s1 // (op 406) 


s_mul_i32 s15, s50, s16 // (op 404) 
s_cmp_gt_u32 s51, s15 // (op 404) 
s_mov_b32 s6, scc // (op 404) 
s_sub_u32 s7, s51, s15 // (op 404) 

s_cmp_eq_i32 s6, 0 // (op 404) 
s_cselect_b32 s8, 0, s7 // (op 404) 


s_cmp_lt_u32 s50, s8 // (op 404) 
s_mov_b32 s15, scc // (op 404) 
s_cmp_eq_i32 s15, 0 // (op 404) 
s_cselect_b32 s6, s8, s50 // (op 404) 


s_cmp_lt_u32 s0, s6 // (op 404) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_DPStreamTileLoop_404 // (op 404) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_DPStreamTileLoop_404 if true)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_404:





































s_endpgm  // End of RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2
.LRR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_end:
.size RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2, .LRR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_end-RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 512 // lds bytes
  .amdhsa_accum_offset 64
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
    DataTypeB: F4
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
    macrotile: [16, 64, 128]
    threads: [128, 1, 1]
    grid: [RocRollerStreamK, One, One]
    occupancy: 16
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
  - .name: RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2
    .symbol: RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 512
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 86
    .vgpr_count: 62
    .agpr_count: 8
    .max_flat_workgroup_size: 128
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

