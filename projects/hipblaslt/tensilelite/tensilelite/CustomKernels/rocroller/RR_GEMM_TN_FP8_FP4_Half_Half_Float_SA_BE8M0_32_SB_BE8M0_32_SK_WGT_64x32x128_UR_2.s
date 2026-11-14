// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2,@function
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 2259) 
v_and_b32 v3, 15, v1 // (op 2259) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2259) 


v_and_b32 v0, 63, v1 // (op 2259) 
v_lshrrev_b32 v3, 4, v0 // (op 2259) 

v_lshl_add_u32 v0, v4, 2, v3 // (op 2259) 


v_lshrrev_b32 v3, 7, v1 // (op 2359) 
v_and_b32 v4, 15, v1 // (op 2359) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 2359) 


v_and_b32 v3, 63, v1 // (op 2359) 
v_lshrrev_b32 v4, 4, v3 // (op 2359) 

v_lshl_add_u32 v3, v5, 2, v4 // (op 2359) 


v_lshrrev_b32 v4, 7, v1 // (op 2430) 
v_and_b32 v5, 15, v1 // (op 2430) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 2430) 


v_and_b32 v4, 63, v1 // (op 2430) 
v_lshrrev_b32 v5, 4, v4 // (op 2430) 

v_lshl_add_u32 v4, v6, 2, v5 // (op 2430) 


v_lshrrev_b32 v5, 7, v1 // (op 2459) 
v_and_b32 v6, 15, v1 // (op 2459) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 2459) 


v_and_b32 v5, 63, v1 // (op 2459) 
v_lshrrev_b32 v6, 4, v5 // (op 2459) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 2459) 


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
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_249 // (op 249) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_249 if false)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_SKStreamTileLoop_249:
v_lshrrev_b32 v8, 7, v1 // (op 2174) 
s_mul_i32 s52, s16, s8 // (op 2174) 
s_add_u32 s52, s52, s0 // (op 2174) 
v_and_b32 v9, 63, v1 // (op 2174) 
s_mul_hi_u32 s53, s52, s13 // (op 2174) 
v_lshrrev_b32 v10, 4, v9 // (op 2174) 
s_sub_u32 s54, s52, s53 // (op 2174) 
v_lshlrev_b32 v11, 2, v10 // (op 2174) 

v_lshl_add_u32 v10, v8, 4, v11 // (op 2174) 


s_lshr_b32 s55, s54, 1 // (op 2174) 

s_add_u32 s54, s55, s53 // (op 2174) 


v_lshrrev_b32 v8, 2, v9 // (op 2174) 

s_lshr_b32 s53, s54, s11 // (op 2174) 

v_and_b32 v9, 3, v8 // (op 2174) 

s_and_b32 s54, s11, 2147483648 // (op 2174) 
v_and_b32 v8, 3, v1 // (op 2174) 
s_cmp_eq_i32 s54, 0 // (op 2174) 
s_mov_b32 s55, scc // (op 2174) 

v_lshl_add_u32 v11, v9, 2, v8 // (op 2174) 


s_cmp_eq_i32 s55, 0 // (op 2174) 
s_cselect_b32 s54, s52, s53 // (op 2174) 



v_lshrrev_b32 v8, 6, v1 // (op 2174) 
s_mul_hi_u32 s52, s54, s12 // (op 2174) 
v_and_b32 v9, 1, v8 // (op 2174) 

s_sub_u32 s53, s54, s52 // (op 2174) 
v_lshl_add_u32 v8, v9, 4, v11 // (op 2174) 


s_lshr_b32 s55, s53, 1 // (op 2174) 

s_add_u32 s53, s55, s52 // (op 2174) 


s_lshr_b32 s52, s53, s10 // (op 2174) 

s_and_b32 s53, s10, 2147483648 // (op 2174) 
s_cmp_eq_i32 s53, 0 // (op 2174) 
s_mov_b32 s55, scc // (op 2174) 

s_cmp_eq_i32 s55, 0 // (op 2174) 
s_cselect_b32 s53, s54, s52 // (op 2174) 


s_mul_i32 s52, s53, s14 // (op 2174) 
v_lshl_add_u32 v9, s53, 6, v10 // (op 2174) 


s_sub_u32 s53, s54, s52 // (op 2174) 


v_lshl_add_u32 v10, s53, 5, v8 // (op 2174) 


v_mul_lo_u32 v8, v10, s4 // (op 2174) 

v_add_lshl_u32 v10, v9, v8, 1 // (op 2174) 


s_mul_i32 s53, s4, s23 // (op 2176) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 2176) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 2176) least significant
s_mul_hi_u32 s55, s4, s22 // (op 2176) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2176) most significant: sum
s_add_u32 s53, s53, s55 // (op 2176) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 2176) 

s_mov_b32 s56, s20 // (op 2176) reg expression
s_mov_b32 s57, s21 // (op 2176) reg expression
s_mov_b32 s58, s54 // (op 2176) reg expression
s_mov_b32 s59, 131072 // (op 2176) call()

v_lshrrev_b32 v8, 7, v1 // (op 2276) 
s_mul_i32 s52, s16, s8 // (op 2276) 
s_add_u32 s52, s52, s0 // (op 2276) 
v_and_b32 v9, 63, v1 // (op 2276) 
s_mul_hi_u32 s53, s52, s13 // (op 2276) 
v_lshrrev_b32 v11, 4, v9 // (op 2276) 
s_sub_u32 s54, s52, s53 // (op 2276) 
v_lshlrev_b32 v12, 2, v11 // (op 2276) 

s_lshr_b32 s55, s54, 1 // (op 2276) 

v_lshl_add_u32 v11, v8, 4, v12 // (op 2276) 


s_add_u32 s54, s55, s53 // (op 2276) 


v_lshrrev_b32 v8, 2, v9 // (op 2276) 

s_lshr_b32 s53, s54, s11 // (op 2276) 

v_and_b32 v9, 3, v8 // (op 2276) 

s_and_b32 s54, s11, 2147483648 // (op 2276) 
v_and_b32 v8, 3, v1 // (op 2276) 
s_cmp_eq_i32 s54, 0 // (op 2276) 
s_mov_b32 s55, scc // (op 2276) 

v_lshl_add_u32 v12, v9, 2, v8 // (op 2276) 


s_cmp_eq_i32 s55, 0 // (op 2276) 
s_cselect_b32 s54, s52, s53 // (op 2276) 



v_lshrrev_b32 v8, 6, v1 // (op 2276) 
s_mul_hi_u32 s55, s54, s12 // (op 2276) 
v_and_b32 v9, 1, v8 // (op 2276) 

s_sub_u32 s52, s54, s55 // (op 2276) 
v_lshl_add_u32 v8, v9, 4, v12 // (op 2276) 


s_lshr_b32 s53, s52, 1 // (op 2276) 

s_add_u32 s52, s53, s55 // (op 2276) 


s_lshr_b32 s53, s52, s10 // (op 2276) 

s_and_b32 s52, s10, 2147483648 // (op 2276) 
s_cmp_eq_i32 s52, 0 // (op 2276) 
s_mov_b32 s55, scc // (op 2276) 

s_cmp_eq_i32 s55, 0 // (op 2276) 
s_cselect_b32 s52, s54, s53 // (op 2276) 


s_mul_i32 s53, s52, s14 // (op 2276) 
v_lshl_add_u32 v9, s52, 6, v11 // (op 2276) 


s_sub_u32 s52, s54, s53 // (op 2276) 


v_lshl_add_u32 v11, s52, 5, v8 // (op 2276) 


v_mul_lo_u32 v8, v11, s24 // (op 2276) 

v_add_lshl_u32 v11, v9, v8, 1 // (op 2276) 


s_mul_i32 s53, s24, s23 // (op 2278) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 2278) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 2278) least significant
s_mul_hi_u32 s55, s24, s22 // (op 2278) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2278) most significant: sum
s_add_u32 s53, s53, s55 // (op 2278) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 2278) 

s_mov_b32 s60, s26 // (op 2278) reg expression
s_mov_b32 s61, s27 // (op 2278) reg expression
s_mov_b32 s62, s54 // (op 2278) reg expression
s_mov_b32 s63, 131072 // (op 2278) call()

v_lshrrev_b32 v8, 5, v1 // (op 2376) 
v_lshlrev_b32 v9, 2, v8 // (op 2376) 

v_lshl_add_u32 v8, s16, 6, v9 // (op 2376) 

v_and_b32 v9, -32, v1 // (op 2376) 
v_sub_u32 v12, v1, v9 // (op 2376) 

v_lshl_add_u32 v9, v8, 5, v12 // (op 2376) 


v_lshlrev_b32 v8, 2, v9 // (op 2376) 

s_lshl_b32 s52, s15, 13 // (op 2378) 
s_mov_b32 s64, s28 // (op 2378) reg expression
s_mov_b32 s65, s29 // (op 2378) reg expression
s_mov_b32 s66, s52 // (op 2378) reg expression
s_mov_b32 s67, 131072 // (op 2378) call()

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
v_accvgpr_write a4, 0.00000 // (op 855) call()
v_accvgpr_write a5, 0.00000 // (op 855) call()
v_accvgpr_write a6, 0.00000 // (op 855) call()
v_accvgpr_write a7, 0.00000 // (op 855) call()
v_lshrrev_b32 v9, 7, v1 // (op 1741) 
s_mul_i32 s52, s16, s8 // (op 1741) 
s_add_u32 s52, s52, s0 // (op 1741) 
v_and_b32 v12, 15, v1 // (op 1741) 
s_mul_hi_u32 s53, s52, s13 // (op 1741) 
v_lshl_add_u32 v13, v9, 4, v12 // (op 1741) 


s_sub_u32 s54, s52, s53 // (op 1741) 
v_and_b32 v9, 48, v1 // (op 1741) 
s_lshr_b32 s55, s54, 1 // (op 1741) 

s_add_u32 s54, s55, s53 // (op 1741) 


s_lshr_b32 s53, s54, s11 // (op 1741) 

s_and_b32 s54, s11, 2147483648 // (op 1741) 
s_cmp_eq_i32 s54, 0 // (op 1741) 
s_mov_b32 s55, scc // (op 1741) 

s_cmp_eq_i32 s55, 0 // (op 1741) 
s_cselect_b32 s54, s52, s53 // (op 1741) 


s_mul_hi_u32 s53, s54, s12 // (op 1741) 
s_sub_u32 s55, s54, s53 // (op 1741) 
s_lshr_b32 s68, s55, 1 // (op 1741) 

s_add_u32 s55, s68, s53 // (op 1741) 


s_lshr_b32 s53, s55, s10 // (op 1741) 

s_and_b32 s55, s10, 2147483648 // (op 1741) 
s_cmp_eq_i32 s55, 0 // (op 1741) 
s_mov_b32 s68, scc // (op 1741) 

s_cmp_eq_i32 s68, 0 // (op 1741) 
s_cselect_b32 s55, s54, s53 // (op 1741) 


v_lshl_add_u32 v12, s55, 6, v13 // (op 1741) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 1741) 
s_mul_i32 s53, s54, s68 // (op 1741) 


s_sub_u32 s54, s52, s53 // (op 1741) 


v_lshl_add_u32 v13, s54, 7, v9 // (op 1741) 


v_mul_lo_u32 v9, v12, s30 // (op 1741) 
v_add_u32 v9, v9, v13 // (op 1741) 


s_mul_i32 s53, s30, s35 // (op 1743) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 1743) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 1743) least significant
s_mul_hi_u32 s55, s30, s34 // (op 1743) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1743) most significant: sum
s_add_u32 s53, s53, s55 // (op 1743) most significant: sum


s_mov_b32 s68, s32 // (op 1743) reg expression
s_mov_b32 s69, s33 // (op 1743) reg expression
s_mov_b32 s70, s52 // (op 1743) reg expression
s_mov_b32 s71, 131072 // (op 1743) call()

v_lshrrev_b32 v12, 7, v1 // (op 1762) 
s_mul_i32 s52, s16, s8 // (op 1762) 
s_add_u32 s52, s52, s0 // (op 1762) 
v_and_b32 v13, 15, v1 // (op 1762) 
s_mul_hi_u32 s53, s52, s13 // (op 1762) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 1762) 


s_sub_u32 s54, s52, s53 // (op 1762) 
v_and_b32 v12, 48, v1 // (op 1762) 
s_lshr_b32 s55, s54, 1 // (op 1762) 

s_add_u32 s54, s55, s53 // (op 1762) 


s_lshr_b32 s53, s54, s11 // (op 1762) 

s_and_b32 s54, s11, 2147483648 // (op 1762) 
s_cmp_eq_i32 s54, 0 // (op 1762) 
s_mov_b32 s55, scc // (op 1762) 

s_cmp_eq_i32 s55, 0 // (op 1762) 
s_cselect_b32 s54, s52, s53 // (op 1762) 


s_mul_hi_u32 s53, s54, s12 // (op 1762) 
s_sub_u32 s55, s54, s53 // (op 1762) 
s_lshr_b32 s72, s55, 1 // (op 1762) 

s_add_u32 s55, s72, s53 // (op 1762) 


s_lshr_b32 s53, s55, s10 // (op 1762) 

s_and_b32 s55, s10, 2147483648 // (op 1762) 
s_cmp_eq_i32 s55, 0 // (op 1762) 
s_mov_b32 s72, scc // (op 1762) 

s_cmp_eq_i32 s72, 0 // (op 1762) 
s_cselect_b32 s55, s54, s53 // (op 1762) 


v_lshl_add_u32 v13, s55, 6, v14 // (op 1762) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 1762) 
s_mul_i32 s53, s54, s72 // (op 1762) 


s_sub_u32 s54, s52, s53 // (op 1762) 


v_lshl_add_u32 v14, s54, 7, v12 // (op 1762) 


v_mul_lo_u32 v12, v13, s30 // (op 1762) 
v_add_u32 v12, v12, v14 // (op 1762) 


s_mul_i32 s53, s30, s35 // (op 1764) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 1764) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 1764) least significant
s_mul_hi_u32 s55, s30, s34 // (op 1764) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1764) most significant: sum
s_add_u32 s53, s53, s55 // (op 1764) most significant: sum


s_mov_b32 s68, s32 // (op 1764) reg expression
s_mov_b32 s69, s33 // (op 1764) reg expression
s_mov_b32 s70, s52 // (op 1764) reg expression
s_mov_b32 s71, 131072 // (op 1764) call()

v_lshrrev_b32 v13, 6, v1 // (op 1874) 
s_mul_i32 s52, s16, s8 // (op 1874) 
s_add_u32 s52, s52, s0 // (op 1874) 
v_and_b32 v14, 1, v13 // (op 1874) 

s_mul_hi_u32 s53, s52, s13 // (op 1874) 
v_and_b32 v13, 15, v1 // (op 1874) 
s_sub_u32 s54, s52, s53 // (op 1874) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 1874) 


s_lshr_b32 s55, s54, 1 // (op 1874) 

v_and_b32 v13, 63, v1 // (op 1874) 
s_add_u32 s54, s55, s53 // (op 1874) 


v_lshrrev_b32 v14, 4, v13 // (op 1874) 

s_lshr_b32 s53, s54, s11 // (op 1874) 

v_lshlrev_b32 v13, 5, v14 // (op 1874) 

s_and_b32 s54, s11, 2147483648 // (op 1874) 
s_cmp_eq_i32 s54, 0 // (op 1874) 
s_mov_b32 s55, scc // (op 1874) 

s_cmp_eq_i32 s55, 0 // (op 1874) 
s_cselect_b32 s54, s52, s53 // (op 1874) 


s_mul_hi_u32 s53, s54, s12 // (op 1874) 
s_sub_u32 s55, s54, s53 // (op 1874) 
s_lshr_b32 s72, s55, 1 // (op 1874) 

s_add_u32 s55, s72, s53 // (op 1874) 


s_lshr_b32 s53, s55, s10 // (op 1874) 

s_and_b32 s55, s10, 2147483648 // (op 1874) 
s_cmp_eq_i32 s55, 0 // (op 1874) 
s_mov_b32 s72, scc // (op 1874) 

s_cmp_eq_i32 s72, 0 // (op 1874) 
s_cselect_b32 s55, s54, s53 // (op 1874) 


s_mul_i32 s53, s55, s14 // (op 1874) 

s_sub_u32 s55, s54, s53 // (op 1874) 

v_lshl_add_u32 v14, s55, 5, v15 // (op 1874) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 1874) 
v_mul_lo_u32 v17, v14, s37 // (op 1874) most significant: low of low * high
v_mul_hi_u32 v15, v14, s36 // (op 1874) most significant: high of low * low
v_mul_lo_u32 v16, v14, s36 // (op 1874) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 1874) most significant: sum


s_mul_i32 s53, s54, s72 // (op 1874) 


s_sub_u32 s54, s52, s53 // (op 1874) 


v_lshl_add_u32 v14, s54, 7, v13 // (op 1874) 


v_mov_b32 v19, 0 // (op 1874) convert
v_mov_b32 v18, v14 // (op 1874) convert

v_add_co_u32 v14, s[52:53], v18, v16 // (op 1874) least significant half
v_addc_co_u32 v15, s[52:53], v19, v17, s[52:53] // (op 1874) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 1874) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 1874) 

v_mov_b32 v13, v16 // (op 1874) convert

s_mul_i32 s53, s36, s23 // (op 1876) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 1876) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 1876) least significant
s_mul_hi_u32 s55, s36, s22 // (op 1876) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1876) most significant: sum
s_add_u32 s53, s53, s55 // (op 1876) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1876) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1876) 

s_mov_b32 s72, s38 // (op 1876) reg expression
s_mov_b32 s73, s39 // (op 1876) reg expression
s_mov_b32 s74, s52 // (op 1876) reg expression
s_mov_b32 s75, 131072 // (op 1876) call()

v_lshrrev_b32 v14, 6, v1 // (op 1893) 
s_mul_i32 s52, s16, s8 // (op 1893) 
s_add_u32 s52, s52, s0 // (op 1893) 
v_and_b32 v15, 1, v14 // (op 1893) 

s_mul_hi_u32 s53, s52, s13 // (op 1893) 
v_and_b32 v14, 15, v1 // (op 1893) 
s_sub_u32 s54, s52, s53 // (op 1893) 
v_lshl_add_u32 v16, v15, 4, v14 // (op 1893) 


s_lshr_b32 s55, s54, 1 // (op 1893) 

v_and_b32 v14, 63, v1 // (op 1893) 
s_add_u32 s54, s55, s53 // (op 1893) 


v_lshrrev_b32 v15, 4, v14 // (op 1893) 

s_lshr_b32 s53, s54, s11 // (op 1893) 

v_lshlrev_b32 v14, 5, v15 // (op 1893) 

s_and_b32 s54, s11, 2147483648 // (op 1893) 
s_cmp_eq_i32 s54, 0 // (op 1893) 
s_mov_b32 s55, scc // (op 1893) 

s_cmp_eq_i32 s55, 0 // (op 1893) 
s_cselect_b32 s54, s52, s53 // (op 1893) 


s_mul_hi_u32 s53, s54, s12 // (op 1893) 
s_sub_u32 s55, s54, s53 // (op 1893) 
s_lshr_b32 s76, s55, 1 // (op 1893) 

s_add_u32 s55, s76, s53 // (op 1893) 


s_lshr_b32 s53, s55, s10 // (op 1893) 

s_and_b32 s55, s10, 2147483648 // (op 1893) 
s_cmp_eq_i32 s55, 0 // (op 1893) 
s_mov_b32 s76, scc // (op 1893) 

s_cmp_eq_i32 s76, 0 // (op 1893) 
s_cselect_b32 s55, s54, s53 // (op 1893) 


s_mul_i32 s53, s55, s14 // (op 1893) 

s_sub_u32 s55, s54, s53 // (op 1893) 

v_lshl_add_u32 v15, s55, 5, v16 // (op 1893) 


v_mul_lo_u32 v17, v15, s37 // (op 1893) most significant: low of low * high
v_mul_hi_u32 v18, v15, s36 // (op 1893) most significant: high of low * low
v_mul_lo_u32 v16, v15, s36 // (op 1893) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 1893) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1893) 
s_mul_i32 s53, s54, s76 // (op 1893) 


s_sub_u32 s54, s52, s53 // (op 1893) 


v_lshl_add_u32 v15, s54, 7, v14 // (op 1893) 


v_mov_b32 v19, 0 // (op 1893) convert
v_mov_b32 v18, v15 // (op 1893) convert

v_add_co_u32 v14, s[52:53], v18, v16 // (op 1893) least significant half
v_addc_co_u32 v15, s[52:53], v19, v17, s[52:53] // (op 1893) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 1893) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 1893) 

v_mov_b32 v14, v16 // (op 1893) convert

s_mul_i32 s53, s36, s23 // (op 1895) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 1895) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 1895) least significant
s_mul_hi_u32 s55, s36, s22 // (op 1895) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1895) most significant: sum
s_add_u32 s53, s53, s55 // (op 1895) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1895) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1895) 

s_mov_b32 s72, s38 // (op 1895) reg expression
s_mov_b32 s73, s39 // (op 1895) reg expression
s_mov_b32 s74, s52 // (op 1895) reg expression
s_mov_b32 s75, 131072 // (op 1895) call()

v_lshrrev_b32 v15, 2, v1 // (op 2002) 
s_mul_i32 s52, s16, s8 // (op 2002) 
s_add_u32 s52, s52, s0 // (op 2002) 
v_and_b32 v16, -4, v1 // (op 2002) 
s_mul_hi_u32 s53, s52, s13 // (op 2002) 
v_sub_u32 v17, v1, v16 // (op 2002) 

s_sub_u32 s54, s52, s53 // (op 2002) 
s_lshr_b32 s55, s54, 1 // (op 2002) 

s_add_u32 s54, s55, s53 // (op 2002) 


s_lshr_b32 s53, s54, s11 // (op 2002) 

s_and_b32 s54, s11, 2147483648 // (op 2002) 
s_cmp_eq_i32 s54, 0 // (op 2002) 
s_mov_b32 s55, scc // (op 2002) 

s_cmp_eq_i32 s55, 0 // (op 2002) 
s_cselect_b32 s54, s52, s53 // (op 2002) 


s_mul_hi_u32 s53, s54, s12 // (op 2002) 
s_sub_u32 s55, s54, s53 // (op 2002) 
s_lshr_b32 s76, s55, 1 // (op 2002) 

s_add_u32 s55, s76, s53 // (op 2002) 


s_lshr_b32 s53, s55, s10 // (op 2002) 

s_and_b32 s55, s10, 2147483648 // (op 2002) 
s_cmp_eq_i32 s55, 0 // (op 2002) 
s_mov_b32 s76, scc // (op 2002) 

s_cmp_eq_i32 s76, 0 // (op 2002) 
s_cselect_b32 s55, s54, s53 // (op 2002) 


v_lshl_add_u32 v16, s55, 6, v15 // (op 2002) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 2002) 
s_mul_i32 s53, s54, s76 // (op 2002) 


s_sub_u32 s54, s52, s53 // (op 2002) 


v_lshl_add_u32 v15, s54, 2, v17 // (op 2002) 


v_mul_lo_u32 v17, v16, s40 // (op 2002) 
v_add_u32 v17, v17, v15 // (op 2002) 


s_mul_i32 s53, s40, s35 // (op 2004) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 2004) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 2004) least significant
s_mul_hi_u32 s55, s40, s34 // (op 2004) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2004) most significant: sum
s_add_u32 s53, s53, s55 // (op 2004) most significant: sum


s_mov_b32 s76, s42 // (op 2004) reg expression
s_mov_b32 s77, s43 // (op 2004) reg expression
s_mov_b32 s78, s52 // (op 2004) reg expression
s_mov_b32 s79, 131072 // (op 2004) call()

v_lshrrev_b32 v15, 2, v1 // (op 2021) 
s_mul_i32 s52, s16, s8 // (op 2021) 
s_add_u32 s52, s52, s0 // (op 2021) 
v_and_b32 v16, -4, v1 // (op 2021) 
s_mul_hi_u32 s53, s52, s13 // (op 2021) 
v_sub_u32 v18, v1, v16 // (op 2021) 

s_sub_u32 s54, s52, s53 // (op 2021) 
s_lshr_b32 s55, s54, 1 // (op 2021) 

s_add_u32 s54, s55, s53 // (op 2021) 


s_lshr_b32 s53, s54, s11 // (op 2021) 

s_and_b32 s54, s11, 2147483648 // (op 2021) 
s_cmp_eq_i32 s54, 0 // (op 2021) 
s_mov_b32 s55, scc // (op 2021) 

s_cmp_eq_i32 s55, 0 // (op 2021) 
s_cselect_b32 s54, s52, s53 // (op 2021) 


s_mul_hi_u32 s53, s54, s12 // (op 2021) 
s_sub_u32 s55, s54, s53 // (op 2021) 
s_lshr_b32 s80, s55, 1 // (op 2021) 

s_add_u32 s55, s80, s53 // (op 2021) 


s_lshr_b32 s53, s55, s10 // (op 2021) 

s_and_b32 s55, s10, 2147483648 // (op 2021) 
s_cmp_eq_i32 s55, 0 // (op 2021) 
s_mov_b32 s80, scc // (op 2021) 

s_cmp_eq_i32 s80, 0 // (op 2021) 
s_cselect_b32 s55, s54, s53 // (op 2021) 


v_lshl_add_u32 v16, s55, 6, v15 // (op 2021) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 2021) 
s_mul_i32 s53, s54, s80 // (op 2021) 


s_sub_u32 s54, s52, s53 // (op 2021) 


v_lshl_add_u32 v15, s54, 2, v18 // (op 2021) 


v_mul_lo_u32 v18, v16, s40 // (op 2021) 
v_add_u32 v18, v18, v15 // (op 2021) 


s_mul_i32 s53, s40, s35 // (op 2023) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 2023) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 2023) least significant
s_mul_hi_u32 s55, s40, s34 // (op 2023) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2023) most significant: sum
s_add_u32 s53, s53, s55 // (op 2023) most significant: sum


s_mov_b32 s76, s42 // (op 2023) reg expression
s_mov_b32 s77, s43 // (op 2023) reg expression
s_mov_b32 s78, s52 // (op 2023) reg expression
s_mov_b32 s79, 131072 // (op 2023) call()

v_lshrrev_b32 v15, 6, v1 // (op 2130) 
s_mul_i32 s52, s16, s8 // (op 2130) 
s_add_u32 s52, s52, s0 // (op 2130) 
v_and_b32 v16, 1, v15 // (op 2130) 

s_mul_hi_u32 s53, s52, s13 // (op 2130) 
v_and_b32 v15, 15, v1 // (op 2130) 
s_sub_u32 s54, s52, s53 // (op 2130) 
v_lshl_add_u32 v19, v16, 4, v15 // (op 2130) 


s_lshr_b32 s55, s54, 1 // (op 2130) 

v_and_b32 v15, 63, v1 // (op 2130) 
s_add_u32 s54, s55, s53 // (op 2130) 


v_lshrrev_b32 v16, 4, v15 // (op 2130) 

s_lshr_b32 s53, s54, s11 // (op 2130) 

s_and_b32 s54, s11, 2147483648 // (op 2130) 
s_cmp_eq_i32 s54, 0 // (op 2130) 
s_mov_b32 s55, scc // (op 2130) 

s_cmp_eq_i32 s55, 0 // (op 2130) 
s_cselect_b32 s54, s52, s53 // (op 2130) 


s_mul_hi_u32 s53, s54, s12 // (op 2130) 
s_sub_u32 s55, s54, s53 // (op 2130) 
s_lshr_b32 s80, s55, 1 // (op 2130) 

s_add_u32 s55, s80, s53 // (op 2130) 


s_lshr_b32 s53, s55, s10 // (op 2130) 

s_and_b32 s55, s10, 2147483648 // (op 2130) 
s_cmp_eq_i32 s55, 0 // (op 2130) 
s_mov_b32 s80, scc // (op 2130) 

s_cmp_eq_i32 s80, 0 // (op 2130) 
s_cselect_b32 s55, s54, s53 // (op 2130) 


s_mul_i32 s53, s55, s14 // (op 2130) 

s_sub_u32 s55, s54, s53 // (op 2130) 

v_lshl_add_u32 v15, s55, 5, v19 // (op 2130) 


v_mul_lo_u32 v19, v15, s44 // (op 2130) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 2130) 
s_mul_i32 s53, s54, s80 // (op 2130) 


s_sub_u32 s54, s52, s53 // (op 2130) 


v_lshl_add_u32 v15, s54, 2, v16 // (op 2130) 


v_add_u32 v16, v15, v19 // (op 2130) 


s_mul_i32 s53, s44, s23 // (op 2132) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 2132) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 2132) least significant
s_mul_hi_u32 s55, s44, s22 // (op 2132) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2132) most significant: sum
s_add_u32 s53, s53, s55 // (op 2132) most significant: sum


s_mov_b32 s80, s46 // (op 2132) reg expression
s_mov_b32 s81, s47 // (op 2132) reg expression
s_mov_b32 s82, s52 // (op 2132) reg expression
s_mov_b32 s83, 131072 // (op 2132) call()

v_lshrrev_b32 v15, 6, v1 // (op 2149) 
s_mul_i32 s52, s16, s8 // (op 2149) 
s_add_u32 s52, s52, s0 // (op 2149) 
v_and_b32 v19, 1, v15 // (op 2149) 

s_mul_hi_u32 s53, s52, s13 // (op 2149) 
v_and_b32 v15, 15, v1 // (op 2149) 
s_sub_u32 s54, s52, s53 // (op 2149) 
v_lshl_add_u32 v20, v19, 4, v15 // (op 2149) 


s_lshr_b32 s55, s54, 1 // (op 2149) 

v_and_b32 v15, 63, v1 // (op 2149) 
s_add_u32 s54, s55, s53 // (op 2149) 


v_lshrrev_b32 v19, 4, v15 // (op 2149) 

s_lshr_b32 s53, s54, s11 // (op 2149) 

s_and_b32 s54, s11, 2147483648 // (op 2149) 
s_cmp_eq_i32 s54, 0 // (op 2149) 
s_mov_b32 s55, scc // (op 2149) 

s_cmp_eq_i32 s55, 0 // (op 2149) 
s_cselect_b32 s54, s52, s53 // (op 2149) 


s_mul_hi_u32 s53, s54, s12 // (op 2149) 
s_sub_u32 s55, s54, s53 // (op 2149) 
s_lshr_b32 s84, s55, 1 // (op 2149) 

s_add_u32 s55, s84, s53 // (op 2149) 


s_lshr_b32 s53, s55, s10 // (op 2149) 

s_and_b32 s55, s10, 2147483648 // (op 2149) 
s_cmp_eq_i32 s55, 0 // (op 2149) 
s_mov_b32 s84, scc // (op 2149) 

s_cmp_eq_i32 s84, 0 // (op 2149) 
s_cselect_b32 s55, s54, s53 // (op 2149) 


s_mul_i32 s53, s55, s14 // (op 2149) 

s_sub_u32 s55, s54, s53 // (op 2149) 

v_lshl_add_u32 v15, s55, 5, v20 // (op 2149) 


v_mul_lo_u32 v20, v15, s44 // (op 2149) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 2149) 
s_mul_i32 s53, s54, s84 // (op 2149) 


s_sub_u32 s54, s52, s53 // (op 2149) 


v_lshl_add_u32 v15, s54, 2, v19 // (op 2149) 


v_add_u32 v19, v15, v20 // (op 2149) 


s_mul_i32 s53, s44, s23 // (op 2151) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 2151) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 2151) least significant
s_mul_hi_u32 s55, s44, s22 // (op 2151) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2151) most significant: sum
s_add_u32 s53, s53, s55 // (op 2151) most significant: sum


s_mov_b32 s80, s46 // (op 2151) reg expression
s_mov_b32 s81, s47 // (op 2151) reg expression
s_mov_b32 s82, s52 // (op 2151) reg expression
s_mov_b32 s83, 131072 // (op 2151) call()



v_mov_b32 v15, v9 // (op 1508) reg expression

buffer_load_dwordx4 v[20:23], v15, s[68:71], 0 offen // (op 1508) Load value
buffer_load_dwordx4 v[24:27], v15, s[68:71], 0 offen offset:64 // (op 1508) Load value




buffer_load_dwordx4 v[28:31], v13, s[72:75], 0 offen // (op 1514) Load value



buffer_load_ubyte v15, v16, s[80:83], 0 offen offset: 0 // (op 1519) Load value


s_lshl_b32 s52, s30, 5 // (op 1525) 
v_add_u32 v32, v9, s52 // (op 1525) 


buffer_load_dwordx4 v[34:37], v32, s[68:71], 0 offen // (op 1525) Load value
buffer_load_dwordx4 v[38:41], v32, s[68:71], 0 offen offset:64 // (op 1525) Load value




buffer_load_ubyte v32, v17, s[76:79], 0 offen offset: 0 // (op 1530) Load value



buffer_load_ubyte v33, v16, s[80:83], 0 offen offset:4 // (op 1534) Load value



v_add_u32 v16, 128, v9 // (op 1540) 

buffer_load_dwordx4 v[42:45], v16, s[68:71], 0 offen // (op 1540) Load value
buffer_load_dwordx4 v[46:49], v16, s[68:71], 0 offen offset:64 // (op 1540) Load value




buffer_load_dwordx4 v[50:53], v13, s[72:75], 0 offen offset:64 // (op 1546) Load value



s_lshl_b32 s52, s30, 5 // (op 1552) 
s_add_u32 s52, s52, 128 // (op 1552) 
v_add_u32 v13, v9, s52 // (op 1552) 


buffer_load_dwordx4 v[54:57], v13, s[68:71], 0 offen // (op 1552) Load value
buffer_load_dwordx4 v[58:61], v13, s[68:71], 0 offen offset:64 // (op 1552) Load value





buffer_load_ubyte v9, v17, s[76:79], 0 offen offset:4 // (op 1557) Load value

v_and_b32 v13, -4, v1 // (op 2242) 
v_sub_u32 v16, v1, v13 // (op 2242) 

v_and_b32 v13, 4294967292, v1 // (op 2242) 
v_add_u32 v17, v13, v16 // (op 2242) 


s_barrier  // (op 2530) 



s_waitcnt vmcnt(7)
ds_write_b8 v17, v32 // (op 1559) Store local data 

v_and_b32 v13, -4, v1 // (op 2213) 
v_sub_u32 v16, v1, v13 // (op 2213) 

v_and_b32 v13, 4294967292, v1 // (op 2213) 
v_add_u32 v17, v13, v16 // (op 2213) 


v_and_b32 v13, -4, v1 // (op 2442) 
v_sub_u32 v16, v1, v13 // (op 2442) 

v_and_b32 v13, 4294967292, v1 // (op 2442) 
v_add_u32 v62, v13, v16 // (op 2442) 


s_mov_b32 s1, 0 // (op 857) call()
s_and_b32 s52, s7, 4294967294 // (op 856) 
s_cmp_lt_u32 s1, s52 // (op 856) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_856 // (op 856) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_856 if false)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoop_856:
s_add_u32 s1, s1, 2 // (op 858) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2527) 



ds_read_u8 v13, v0 // (op 65) Load local data 



ds_read_u8 v16, v0 offset:128 // (op 882) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:27], v[28:31], a[0:3], v13, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 72) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:41], v[28:31], a[4:7], v16, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 884) 



v_mov_b32 v13, v12 // (op 57) 
v_add_u32 v16, 256, v13 // (op 57) 

buffer_load_dwordx4 v[20:23], v16, s[68:71], 0 offen // (op 57) Load value
buffer_load_dwordx4 v[24:27], v16, s[68:71], 0 offen offset:64 // (op 57) Load value



v_mov_b32 v16, v14 // (op 61) 

buffer_load_dwordx4 v[28:31], v16, s[72:75], 0 offen offset:128 // (op 61) Load value


v_mov_b32 v63, v19 // (op 70) 

buffer_load_ubyte v15, v63, s[80:83], 0 offen offset:8 // (op 70) Load value


s_lshl_b32 s52, s30, 5 // (op 907) 
s_add_u32 s52, s52, 256 // (op 907) 
v_add_u32 v64, v13, s52 // (op 907) 


buffer_load_dwordx4 v[34:37], v64, s[68:71], 0 offen // (op 907) Load value
buffer_load_dwordx4 v[38:41], v64, s[68:71], 0 offen offset:64 // (op 907) Load value



v_mov_b32 v64, v18 // (op 9) 

buffer_load_ubyte v32, v64, s[76:79], 0 offen offset:8 // (op 9) Load value
s_barrier  // (op 2568) 



s_waitcnt vmcnt(7)
ds_write_b8 v62, v9 offset:256 // (op 795) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2565) 



ds_read_u8 v65, v5 offset:256 // (op 797) Load local data 



ds_read_u8 v66, v5 offset:384 // (op 891) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[50:53], a[0:3], v65, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 800) 

s_barrier  // (op 2725) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:61], v[50:53], a[4:7], v66, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 885) 




buffer_load_ubyte v33, v63, s[80:83], 0 offen offset:12 // (op 799) Load value



v_add_u32 v19, 8, v19 // (op 2141) 
v_add_u32 v63, 384, v13 // (op 802) 

buffer_load_dwordx4 v[42:45], v63, s[68:71], 0 offen // (op 802) Load value
buffer_load_dwordx4 v[46:49], v63, s[68:71], 0 offen offset:64 // (op 802) Load value




buffer_load_dwordx4 v[50:53], v16, s[72:75], 0 offen offset:192 // (op 804) Load value



v_add_u32 v14, 128, v14 // (op 1885) 
s_lshl_b32 s52, s30, 5 // (op 900) 
s_add_u32 s52, s52, 384 // (op 900) 
v_add_u32 v16, v13, s52 // (op 900) 


buffer_load_dwordx4 v[54:57], v16, s[68:71], 0 offen // (op 900) Load value
buffer_load_dwordx4 v[58:61], v16, s[68:71], 0 offen offset:64 // (op 900) Load value





v_add_u32 v12, 256, v12 // (op 1752) 
buffer_load_ubyte v9, v64, s[76:79], 0 offen offset:12 // (op 794) Load value

s_barrier  // (op 2533) 
v_add_u32 v18, 8, v18 // (op 2013) 



s_waitcnt vmcnt(7)
ds_write_b8 v17, v32 // (op 13) Store local data 
s_and_b32 s52, s7, 4294967294 // (op 856) 
s_cmp_lt_u32 s1, s52 // (op 856) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoop_856 // (op 856) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoop_856 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_856:











s_and_b32 s52, s7, 4294967294 // (op 1650) 
v_lshrrev_b32 v12, 7, v1 // (op 1650) 
s_add_u32 s53, s0, s52 // (op 1650) 

v_and_b32 v13, 15, v1 // (op 1650) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 1650) 


s_mul_i32 s52, s16, s8 // (op 1650) 
s_add_u32 s52, s52, s53 // (op 1650) 

v_and_b32 v12, 48, v1 // (op 1650) 
s_mul_hi_u32 s53, s52, s13 // (op 1650) 
s_sub_u32 s54, s52, s53 // (op 1650) 
s_lshr_b32 s55, s54, 1 // (op 1650) 

s_add_u32 s54, s55, s53 // (op 1650) 


s_lshr_b32 s53, s54, s11 // (op 1650) 

s_and_b32 s54, s11, 2147483648 // (op 1650) 
s_cmp_eq_i32 s54, 0 // (op 1650) 
s_mov_b32 s55, scc // (op 1650) 

s_cmp_eq_i32 s55, 0 // (op 1650) 
s_cselect_b32 s54, s52, s53 // (op 1650) 


s_mul_hi_u32 s53, s54, s12 // (op 1650) 
s_sub_u32 s55, s54, s53 // (op 1650) 
s_lshr_b32 s84, s55, 1 // (op 1650) 

s_add_u32 s55, s84, s53 // (op 1650) 


s_lshr_b32 s53, s55, s10 // (op 1650) 

s_and_b32 s55, s10, 2147483648 // (op 1650) 
s_cmp_eq_i32 s55, 0 // (op 1650) 
s_mov_b32 s84, scc // (op 1650) 

s_cmp_eq_i32 s84, 0 // (op 1650) 
s_cselect_b32 s55, s54, s53 // (op 1650) 


v_lshl_add_u32 v13, s55, 6, v14 // (op 1650) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 1650) 
s_mul_i32 s53, s54, s84 // (op 1650) 


s_sub_u32 s54, s52, s53 // (op 1650) 


v_lshl_add_u32 v14, s54, 7, v12 // (op 1650) 


v_mul_lo_u32 v12, v13, s30 // (op 1650) 
v_add_u32 v12, v12, v14 // (op 1650) 


s_mul_i32 s53, s30, s35 // (op 1652) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 1652) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 1652) least significant
s_mul_hi_u32 s55, s30, s34 // (op 1652) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1652) most significant: sum
s_add_u32 s53, s53, s55 // (op 1652) most significant: sum


s_mov_b32 s68, s32 // (op 1652) reg expression
s_mov_b32 s69, s33 // (op 1652) reg expression
s_mov_b32 s70, s52 // (op 1652) reg expression
s_mov_b32 s71, 131072 // (op 1652) call()

s_and_b32 s52, s7, 4294967294 // (op 1809) 
v_lshrrev_b32 v13, 6, v1 // (op 1809) 
s_add_u32 s53, s0, s52 // (op 1809) 

v_and_b32 v14, 1, v13 // (op 1809) 

v_and_b32 v13, 15, v1 // (op 1809) 
s_mul_i32 s52, s16, s8 // (op 1809) 
s_add_u32 s52, s52, s53 // (op 1809) 

v_lshl_add_u32 v15, v14, 4, v13 // (op 1809) 


s_mul_hi_u32 s53, s52, s13 // (op 1809) 
v_and_b32 v13, 63, v1 // (op 1809) 
s_sub_u32 s54, s52, s53 // (op 1809) 
v_lshrrev_b32 v14, 4, v13 // (op 1809) 

s_lshr_b32 s55, s54, 1 // (op 1809) 

v_lshlrev_b32 v13, 5, v14 // (op 1809) 

s_add_u32 s54, s55, s53 // (op 1809) 


s_lshr_b32 s53, s54, s11 // (op 1809) 

s_and_b32 s54, s11, 2147483648 // (op 1809) 
s_cmp_eq_i32 s54, 0 // (op 1809) 
s_mov_b32 s55, scc // (op 1809) 

s_cmp_eq_i32 s55, 0 // (op 1809) 
s_cselect_b32 s54, s52, s53 // (op 1809) 


s_mul_hi_u32 s53, s54, s12 // (op 1809) 
s_sub_u32 s55, s54, s53 // (op 1809) 
s_lshr_b32 s84, s55, 1 // (op 1809) 

s_add_u32 s55, s84, s53 // (op 1809) 


s_lshr_b32 s53, s55, s10 // (op 1809) 

s_and_b32 s55, s10, 2147483648 // (op 1809) 
s_cmp_eq_i32 s55, 0 // (op 1809) 
s_mov_b32 s84, scc // (op 1809) 

s_cmp_eq_i32 s84, 0 // (op 1809) 
s_cselect_b32 s55, s54, s53 // (op 1809) 


s_mul_i32 s53, s55, s14 // (op 1809) 

s_sub_u32 s55, s54, s53 // (op 1809) 

v_lshl_add_u32 v14, s55, 5, v15 // (op 1809) 


v_mul_lo_u32 v35, v14, s37 // (op 1809) most significant: low of low * high
v_mul_hi_u32 v15, v14, s36 // (op 1809) most significant: high of low * low
v_mul_lo_u32 v34, v14, s36 // (op 1809) least significant: low of low * low
v_add_u32 v35, v35, v15 // (op 1809) most significant: sum


s_ashr_i64 s[84:85], s[2:3], 7 // (op 1809) 
s_mul_i32 s53, s54, s84 // (op 1809) 


s_sub_u32 s54, s52, s53 // (op 1809) 


v_lshl_add_u32 v14, s54, 7, v13 // (op 1809) 


v_mov_b32 v37, 0 // (op 1809) convert
v_mov_b32 v36, v14 // (op 1809) convert

v_add_co_u32 v38, s[52:53], v36, v34 // (op 1809) least significant half
v_addc_co_u32 v39, s[52:53], v37, v35, s[52:53] // (op 1809) most significant half

v_lshlrev_b64 v[38:39], 2, v[38:39] // (op 1809) 


v_lshrrev_b64 v[40:41], 3, v[38:39] // (op 1809) 

v_mov_b32 v13, v40 // (op 1809) convert

s_mul_i32 s53, s36, s23 // (op 1811) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 1811) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 1811) least significant
s_mul_hi_u32 s55, s36, s22 // (op 1811) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1811) most significant: sum
s_add_u32 s53, s53, s55 // (op 1811) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1811) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1811) 

s_mov_b32 s72, s38 // (op 1811) reg expression
s_mov_b32 s73, s39 // (op 1811) reg expression
s_mov_b32 s74, s52 // (op 1811) reg expression
s_mov_b32 s75, 131072 // (op 1811) call()

s_and_b32 s52, s7, 4294967294 // (op 1937) 
v_lshrrev_b32 v14, 2, v1 // (op 1937) 
s_add_u32 s53, s0, s52 // (op 1937) 

v_and_b32 v15, -4, v1 // (op 1937) 
v_sub_u32 v16, v1, v15 // (op 1937) 

s_mul_i32 s52, s16, s8 // (op 1937) 
s_add_u32 s52, s52, s53 // (op 1937) 

s_mul_hi_u32 s53, s52, s13 // (op 1937) 
s_sub_u32 s54, s52, s53 // (op 1937) 
s_lshr_b32 s55, s54, 1 // (op 1937) 

s_add_u32 s54, s55, s53 // (op 1937) 


s_lshr_b32 s53, s54, s11 // (op 1937) 

s_and_b32 s54, s11, 2147483648 // (op 1937) 
s_cmp_eq_i32 s54, 0 // (op 1937) 
s_mov_b32 s55, scc // (op 1937) 

s_cmp_eq_i32 s55, 0 // (op 1937) 
s_cselect_b32 s54, s52, s53 // (op 1937) 


s_mul_hi_u32 s53, s54, s12 // (op 1937) 
s_sub_u32 s55, s54, s53 // (op 1937) 
s_lshr_b32 s84, s55, 1 // (op 1937) 

s_add_u32 s55, s84, s53 // (op 1937) 


s_lshr_b32 s53, s55, s10 // (op 1937) 

s_and_b32 s55, s10, 2147483648 // (op 1937) 
s_cmp_eq_i32 s55, 0 // (op 1937) 
s_mov_b32 s84, scc // (op 1937) 

s_cmp_eq_i32 s84, 0 // (op 1937) 
s_cselect_b32 s55, s54, s53 // (op 1937) 


v_lshl_add_u32 v15, s55, 6, v14 // (op 1937) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 1937) 
s_mul_i32 s53, s54, s84 // (op 1937) 


s_sub_u32 s54, s52, s53 // (op 1937) 


v_lshl_add_u32 v14, s54, 2, v16 // (op 1937) 


v_mul_lo_u32 v16, v15, s40 // (op 1937) 
v_add_u32 v16, v16, v14 // (op 1937) 


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

s_and_b32 s52, s7, 4294967294 // (op 2065) 
v_lshrrev_b32 v14, 6, v1 // (op 2065) 
s_add_u32 s53, s0, s52 // (op 2065) 

v_and_b32 v15, 1, v14 // (op 2065) 

v_and_b32 v14, 15, v1 // (op 2065) 
s_mul_i32 s52, s16, s8 // (op 2065) 
s_add_u32 s52, s52, s53 // (op 2065) 

v_lshl_add_u32 v17, v15, 4, v14 // (op 2065) 


s_mul_hi_u32 s53, s52, s13 // (op 2065) 
v_and_b32 v14, 63, v1 // (op 2065) 
s_sub_u32 s54, s52, s53 // (op 2065) 
v_lshrrev_b32 v15, 4, v14 // (op 2065) 

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
s_lshr_b32 s84, s55, 1 // (op 2065) 

s_add_u32 s55, s84, s53 // (op 2065) 


s_lshr_b32 s53, s55, s10 // (op 2065) 

s_and_b32 s55, s10, 2147483648 // (op 2065) 
s_cmp_eq_i32 s55, 0 // (op 2065) 
s_mov_b32 s84, scc // (op 2065) 

s_cmp_eq_i32 s84, 0 // (op 2065) 
s_cselect_b32 s55, s54, s53 // (op 2065) 


s_mul_i32 s53, s55, s14 // (op 2065) 

s_sub_u32 s55, s54, s53 // (op 2065) 

v_lshl_add_u32 v14, s55, 5, v17 // (op 2065) 


v_mul_lo_u32 v17, v14, s44 // (op 2065) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 2065) 
s_mul_i32 s53, s54, s84 // (op 2065) 


s_sub_u32 s54, s52, s53 // (op 2065) 


v_lshl_add_u32 v14, s54, 2, v15 // (op 2065) 


v_add_u32 v15, v14, v17 // (op 2065) 


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

v_and_b32 v14, -4, v1 // (op 2226) 
v_sub_u32 v17, v1, v14 // (op 2226) 

v_and_b32 v14, 4294967292, v1 // (op 2226) 
v_add_u32 v18, v14, v17 // (op 2226) 


s_and_b32 s52, s7, 4294967294 // (op 862) 
s_cmp_lt_u32 s52, s7 // (op 861) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_861 // (op 861) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_861 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_861:


s_add_u32 s52, s52, 1 // (op 863) 
v_mov_b32 v9, v16 // (op 752) 

buffer_load_ubyte v14, v9, s[76:79], 0 offen offset: 0 // (op 752) Load value

v_add_u32 v16, 4, v16 // (op 1929) 


s_barrier  // (op 2536) 
v_mov_b32 v9, v12 // (op 760) 



v_mov_b32 v17, v9 // (op 760) reg expression

buffer_load_dwordx4 v[20:23], v17, s[68:71], 0 offen // (op 760) Load value
buffer_load_dwordx4 v[24:27], v17, s[68:71], 0 offen offset:64 // (op 760) Load value

s_waitcnt vmcnt(2)
ds_write_b8 v18, v14 // (op 753) Store local data 



v_mov_b32 v14, v13 // (op 762) 

buffer_load_dwordx4 v[28:31], v14, s[72:75], 0 offen // (op 762) Load value

v_add_u32 v13, 64, v13 // (op 1801) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2539) 



ds_read_u8 v14, v0 // (op 755) Load local data 


v_mov_b32 v17, v15 // (op 757) 

buffer_load_ubyte v19, v17, s[80:83], 0 offen offset: 0 // (op 757) Load value

v_add_u32 v15, 4, v15 // (op 2057) 


s_lshl_b32 s53, s30, 5 // (op 877) 
v_add_u32 v17, v9, s53 // (op 877) 


buffer_load_dwordx4 v[32:35], v17, s[68:71], 0 offen // (op 877) Load value
s_waitcnt vmcnt(1) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:27], v[28:31], a[0:3], v14, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 758) 
buffer_load_dwordx4 v[36:39], v17, s[68:71], 0 offen offset:64 // (op 877) Load value







v_add_u32 v12, 128, v12 // (op 1638) 
ds_read_u8 v9, v0 offset:128 // (op 867) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:39], v[28:31], a[4:7], v9, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 869) 




s_cmp_lt_u32 s52, s7 // (op 861) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_861 // (op 861) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_861 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_861:






s_barrier  // (op 2711) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 251) 
s_cmp_lg_u32 s18, 0 // (op 277) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Send_Tile_277 // (op 277) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Send_Tile_277



v_add_u32 v9, 4096, v8 // (op 1018) 
s_nop 0
buffer_store_dword a4, v9, s[64:67], 0 offen sc1 // (op 1018) Store value

v_add_u32 v9, 4224, v8 // (op 1018) 
buffer_store_dword a5, v9, s[64:67], 0 offen sc1 // (op 1018) Store value

v_add_u32 v9, 4352, v8 // (op 1018) 
buffer_store_dword a6, v9, s[64:67], 0 offen sc1 // (op 1018) Store value

v_add_u32 v9, 4480, v8 // (op 1018) 
buffer_store_dword a7, v9, s[64:67], 0 offen sc1 // (op 1018) Store value




buffer_store_dword a0, v8, s[64:67], 0 offen sc1 // (op 274) Store value
buffer_store_dword a1, v8, s[64:67], 0 offen offset:128 sc1 // (op 274) Store value
buffer_store_dword a2, v8, s[64:67], 0 offen offset:256 sc1 // (op 274) Store value
buffer_store_dword a3, v8, s[64:67], 0 offen offset:384 sc1 // (op 274) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 278) 
v_cmp_eq_i32 vcc, v1, 0 // (op 282) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_282 // (op 282) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_282
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
s_branch RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_282 // (op 282) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_282
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_282:
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_282:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Send_Tile_277 // (op 277) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Send_Tile_277
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Send_Tile_277:
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Send_Tile_277:

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
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Receive_Tile_293 // (op 293) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Receive_Tile_293
s_mov_b32 s52, 0 // (op 259) call()
s_mul_i32 s53, s52, s8 // (op 258) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 258) 
s_mov_b32 s84, s54 // (op 258) convert

s_sub_u32 s54, s84, s19 // (op 258) 

s_sub_u32 s55, s54, 1 // (op 258) 

s_cmp_lt_u32 s53, s55 // (op 258) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_258 // (op 258) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_258 if false)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_ReceiveTileLoop_258:
s_add_u32 s53, s16, 1 // (op 2398) 
v_lshrrev_b32 v8, 5, v1 // (op 2398) 
s_add_u32 s54, s53, s52 // (op 2398) 
s_lshl_b32 s54, s54, 6 // (op 2398) 

v_lshlrev_b32 v9, 2, v8 // (op 2398) 

v_add_lshl_u32 v8, s54, v9, 5 // (op 2398) 


v_and_b32 v9, -32, v1 // (op 2398) 
v_sub_u32 v12, v1, v9 // (op 2398) 

v_add_lshl_u32 v9, v8, v12, 2 // (op 2398) 


s_lshl_b32 s53, s15, 13 // (op 2400) 
s_mov_b32 s64, s28 // (op 2400) reg expression
s_mov_b32 s65, s29 // (op 2400) reg expression
s_mov_b32 s66, s53 // (op 2400) reg expression
s_mov_b32 s67, 131072 // (op 2400) call()

RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_297:
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
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_297 // (op 297) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_297 if true)
s_barrier  // (op 301) 
v_cmp_eq_i32 vcc, v1, 0 // (op 300) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_300 // (op 300) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_300
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
s_branch RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_300 // (op 300) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_300
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_300:
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_300:




buffer_load_dword v12, v9, s[64:67], 0 offen // (op 312) Load value
buffer_load_dword v13, v9, s[64:67], 0 offen offset:128 // (op 312) Load value
buffer_load_dword v14, v9, s[64:67], 0 offen offset:256 // (op 312) Load value
buffer_load_dword v15, v9, s[64:67], 0 offen offset:384 // (op 312) Load value
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




v_mov_b32 v8, 4096 // (op 1034) 
v_add_i32 v12, v9, v8 // (op 1034) 

buffer_load_dword v14, v12, s[64:67], 0 offen // (op 1034) Load value

v_mov_b32 v8, 4224 // (op 1034) 
v_add_i32 v12, v9, v8 // (op 1034) 

buffer_load_dword v15, v12, s[64:67], 0 offen // (op 1034) Load value

v_mov_b32 v8, 4352 // (op 1034) 
v_add_i32 v12, v9, v8 // (op 1034) 

buffer_load_dword v16, v12, s[64:67], 0 offen // (op 1034) Load value

v_mov_b32 v8, 4480 // (op 1034) 
v_add_i32 v12, v9, v8 // (op 1034) 

buffer_load_dword v17, v12, s[64:67], 0 offen // (op 1034) Load value

v_accvgpr_read v18, a4 // (op 1035) reg expression
v_accvgpr_read v19, a5 // (op 1035) reg expression
v_accvgpr_read v20, a6 // (op 1035) reg expression
v_accvgpr_read v21, a7 // (op 1035) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v18, v18, v14 // (op 1036) 
s_waitcnt vmcnt(2)
v_add_f32 v19, v19, v15 // (op 1036) 
s_waitcnt vmcnt(1)
v_add_f32 v20, v20, v16 // (op 1036) 
s_waitcnt vmcnt(0)
v_add_f32 v21, v21, v17 // (op 1036) 

v_accvgpr_write a4, v18 // (op 1037) reg expression
v_accvgpr_write a5, v19 // (op 1037) reg expression
v_accvgpr_write a6, v20 // (op 1037) reg expression
v_accvgpr_write a7, v21 // (op 1037) reg expression

s_add_u32 s52, s52, 1 // (op 260) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 258) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 258) 
s_mov_b32 s84, s54 // (op 258) convert

s_sub_u32 s54, s84, s19 // (op 258) 

s_sub_u32 s55, s54, 1 // (op 258) 

s_cmp_lt_u32 s53, s55 // (op 258) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_ReceiveTileLoop_258 // (op 258) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_ReceiveTileLoop_258 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_258:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Receive_Tile_293 // (op 293) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Receive_Tile_293
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Receive_Tile_293:
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Receive_Tile_293:

s_cmp_eq_i32 s18, 0 // (op 336) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_336 // (op 336) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_336
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 108) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_108 // (op 108) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_108
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


buffer_store_dwordx2 v[8:9], v11, s[60:63], 0 offen // (op 46) Store value


v_accvgpr_read v12, a4 // (op 1059) 
v_accvgpr_read v13, a5 // (op 1059) 
v_accvgpr_read v14, a6 // (op 1059) 
v_accvgpr_read v15, a7 // (op 1059) 
v_mul_f32 v16, v7, v12 // (op 1059) 
v_mul_f32 v17, v7, v13 // (op 1059) 
v_mul_f32 v18, v7, v14 // (op 1059) 
v_mul_f32 v19, v7, v15 // (op 1059) 

v_cvt_f16_f32 v12, v16 // (op 1061) 
v_cvt_f16_f32 v13, v17 // (op 1061) 
v_cvt_f16_f32 v14, v18 // (op 1061) 
v_cvt_f16_f32 v15, v19 // (op 1061) 


v_pack_b32_f16 v8, v12, v13 // (op 1062) 
v_pack_b32_f16 v9, v14, v15 // (op 1062) 


buffer_store_dwordx2 v[8:9], v11, s[60:63], 0 offen offset:64 // (op 1062) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_108 // (op 108) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_108
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_108:



buffer_load_dwordx2 v[8:9], v10, s[56:59], 0 offen offset:64 // (op 1092) Load value
v_accvgpr_read v12, a4 // (op 1088) 
v_accvgpr_read v13, a5 // (op 1088) 
v_accvgpr_read v14, a6 // (op 1088) 
v_accvgpr_read v15, a7 // (op 1088) 
v_mul_f32 v16, v7, v12 // (op 1088) 
v_mul_f32 v17, v7, v13 // (op 1088) 
v_mul_f32 v18, v7, v14 // (op 1088) 
v_mul_f32 v19, v7, v15 // (op 1088) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v8 // (op 1093) 
v_lshrrev_b32 v13, 16, v8 // (op 1093) 
v_cvt_f32_f16 v13, v13 // (op 1093) 
v_mul_f32 v20, v6, v12 // (op 1093) 
v_mul_f32 v21, v6, v13 // (op 1093) 
v_cvt_f32_f16 v12, v9 // (op 1093) 
v_lshrrev_b32 v13, 16, v9 // (op 1093) 
v_cvt_f32_f16 v13, v13 // (op 1093) 
v_mul_f32 v22, v6, v12 // (op 1093) 
v_mul_f32 v23, v6, v13 // (op 1093) 


v_add_f32 v12, v20, v16 // (op 1089) 
v_add_f32 v13, v21, v17 // (op 1089) 
v_add_f32 v14, v22, v18 // (op 1089) 
v_add_f32 v15, v23, v19 // (op 1089) 


v_cvt_f16_f32 v16, v12 // (op 1090) 
v_cvt_f16_f32 v17, v13 // (op 1090) 
v_cvt_f16_f32 v18, v14 // (op 1090) 
v_cvt_f16_f32 v19, v15 // (op 1090) 


v_pack_b32_f16 v8, v16, v17 // (op 1091) 
v_pack_b32_f16 v9, v18, v19 // (op 1091) 


buffer_store_dwordx2 v[8:9], v11, s[60:63], 0 offen offset:64 // (op 1091) Store value





buffer_load_dwordx2 v[8:9], v10, s[56:59], 0 offen // (op 121) Load value
v_accvgpr_read v12, a0 // (op 117) 
v_accvgpr_read v13, a1 // (op 117) 
v_accvgpr_read v14, a2 // (op 117) 
v_accvgpr_read v15, a3 // (op 117) 
v_mul_f32 v16, v7, v12 // (op 117) 
v_mul_f32 v17, v7, v13 // (op 117) 
v_mul_f32 v18, v7, v14 // (op 117) 
v_mul_f32 v19, v7, v15 // (op 117) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v8 // (op 122) 
v_lshrrev_b32 v13, 16, v8 // (op 122) 
v_cvt_f32_f16 v13, v13 // (op 122) 
v_mul_f32 v20, v6, v12 // (op 122) 
v_mul_f32 v21, v6, v13 // (op 122) 
v_cvt_f32_f16 v12, v9 // (op 122) 
v_lshrrev_b32 v13, 16, v9 // (op 122) 
v_cvt_f32_f16 v13, v13 // (op 122) 
v_mul_f32 v22, v6, v12 // (op 122) 
v_mul_f32 v23, v6, v13 // (op 122) 


v_add_f32 v12, v20, v16 // (op 118) 
v_add_f32 v13, v21, v17 // (op 118) 
v_add_f32 v14, v22, v18 // (op 118) 
v_add_f32 v15, v23, v19 // (op 118) 


v_cvt_f16_f32 v16, v12 // (op 119) 
v_cvt_f16_f32 v17, v13 // (op 119) 
v_cvt_f16_f32 v18, v14 // (op 119) 
v_cvt_f16_f32 v19, v15 // (op 119) 


v_pack_b32_f16 v8, v16, v17 // (op 120) 
v_pack_b32_f16 v9, v18, v19 // (op 120) 


buffer_store_dwordx2 v[8:9], v11, s[60:63], 0 offen // (op 120) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_108:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_336 // (op 336) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_336
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_336:
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_336:


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
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_SKStreamTileLoop_249 // (op 249) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_SKStreamTileLoop_249 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_249:











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
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_404 // (op 404) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_404 if false)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_DPStreamTileLoop_404:
v_lshrrev_b32 v0, 7, v1 // (op 2196) 
s_mul_i32 s15, s16, s50 // (op 2196) 
s_add_u32 s15, s15, s0 // (op 2196) 
v_and_b32 v5, 63, v1 // (op 2196) 
s_add_u32 s6, s15, s9 // (op 2196) 

v_lshrrev_b32 v8, 4, v5 // (op 2196) 
s_mul_hi_u32 s15, s6, s13 // (op 2196) 
v_lshlrev_b32 v9, 2, v8 // (op 2196) 

s_sub_u32 s7, s6, s15 // (op 2196) 
v_lshl_add_u32 v8, v0, 4, v9 // (op 2196) 


s_lshr_b32 s8, s7, 1 // (op 2196) 

v_lshrrev_b32 v0, 2, v5 // (op 2196) 

s_add_u32 s7, s8, s15 // (op 2196) 


v_and_b32 v5, 3, v0 // (op 2196) 

s_lshr_b32 s8, s7, s11 // (op 2196) 

v_and_b32 v0, 3, v1 // (op 2196) 
s_and_b32 s7, s11, 2147483648 // (op 2196) 
v_lshl_add_u32 v9, v5, 2, v0 // (op 2196) 


s_cmp_eq_i32 s7, 0 // (op 2196) 
s_mov_b32 s15, scc // (op 2196) 

v_lshrrev_b32 v0, 6, v1 // (op 2196) 
s_cmp_eq_i32 s15, 0 // (op 2196) 
s_cselect_b32 s7, s6, s8 // (op 2196) 



v_and_b32 v5, 1, v0 // (op 2196) 

s_mul_hi_u32 s6, s7, s12 // (op 2196) 
v_lshl_add_u32 v0, v5, 4, v9 // (op 2196) 


s_sub_u32 s8, s7, s6 // (op 2196) 
s_lshr_b32 s15, s8, 1 // (op 2196) 

s_add_u32 s8, s15, s6 // (op 2196) 


s_lshr_b32 s6, s8, s10 // (op 2196) 

s_and_b32 s8, s10, 2147483648 // (op 2196) 
s_cmp_eq_i32 s8, 0 // (op 2196) 
s_mov_b32 s15, scc // (op 2196) 

s_cmp_eq_i32 s15, 0 // (op 2196) 
s_cselect_b32 s8, s7, s6 // (op 2196) 


s_mul_i32 s6, s8, s14 // (op 2196) 
v_lshl_add_u32 v5, s8, 6, v8 // (op 2196) 


s_sub_u32 s8, s7, s6 // (op 2196) 


v_lshl_add_u32 v8, s8, 5, v0 // (op 2196) 


v_mul_lo_u32 v0, v8, s4 // (op 2196) 

v_add_lshl_u32 v8, v5, v0, 1 // (op 2196) 


s_mul_i32 s19, s4, s23 // (op 2198) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 2198) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 2198) least significant
s_mul_hi_u32 s6, s4, s22 // (op 2198) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2198) most significant: sum
s_add_u32 s19, s19, s6 // (op 2198) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 2198) 

s_mov_b32 s56, s20 // (op 2198) reg expression
s_mov_b32 s57, s21 // (op 2198) reg expression
s_mov_b32 s58, s28 // (op 2198) reg expression
s_mov_b32 s59, 131072 // (op 2198) call()

v_lshrrev_b32 v0, 7, v1 // (op 2296) 
s_mul_i32 s15, s16, s50 // (op 2296) 
s_add_u32 s15, s15, s0 // (op 2296) 
v_and_b32 v5, 63, v1 // (op 2296) 
s_add_u32 s6, s15, s9 // (op 2296) 

v_lshrrev_b32 v9, 4, v5 // (op 2296) 
s_mul_hi_u32 s15, s6, s13 // (op 2296) 
v_lshlrev_b32 v10, 2, v9 // (op 2296) 

s_sub_u32 s7, s6, s15 // (op 2296) 
v_lshl_add_u32 v9, v0, 4, v10 // (op 2296) 


s_lshr_b32 s8, s7, 1 // (op 2296) 

v_lshrrev_b32 v0, 2, v5 // (op 2296) 

s_add_u32 s7, s8, s15 // (op 2296) 


v_and_b32 v5, 3, v0 // (op 2296) 

s_lshr_b32 s8, s7, s11 // (op 2296) 

v_and_b32 v0, 3, v1 // (op 2296) 
s_and_b32 s7, s11, 2147483648 // (op 2296) 
v_lshl_add_u32 v10, v5, 2, v0 // (op 2296) 


s_cmp_eq_i32 s7, 0 // (op 2296) 
s_mov_b32 s15, scc // (op 2296) 

v_lshrrev_b32 v0, 6, v1 // (op 2296) 
s_cmp_eq_i32 s15, 0 // (op 2296) 
s_cselect_b32 s7, s6, s8 // (op 2296) 



v_and_b32 v5, 1, v0 // (op 2296) 

s_mul_hi_u32 s6, s7, s12 // (op 2296) 
v_lshl_add_u32 v0, v5, 4, v10 // (op 2296) 


s_sub_u32 s8, s7, s6 // (op 2296) 
s_lshr_b32 s15, s8, 1 // (op 2296) 

s_add_u32 s8, s15, s6 // (op 2296) 


s_lshr_b32 s6, s8, s10 // (op 2296) 

s_and_b32 s8, s10, 2147483648 // (op 2296) 
s_cmp_eq_i32 s8, 0 // (op 2296) 
s_mov_b32 s15, scc // (op 2296) 

s_cmp_eq_i32 s15, 0 // (op 2296) 
s_cselect_b32 s8, s7, s6 // (op 2296) 


s_mul_i32 s6, s8, s14 // (op 2296) 
v_lshl_add_u32 v5, s8, 6, v9 // (op 2296) 


s_sub_u32 s8, s7, s6 // (op 2296) 


v_lshl_add_u32 v9, s8, 5, v0 // (op 2296) 


v_mul_lo_u32 v0, v9, s24 // (op 2296) 

v_add_lshl_u32 v9, v5, v0, 1 // (op 2296) 


s_mul_i32 s19, s24, s23 // (op 2298) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 2298) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 2298) least significant
s_mul_hi_u32 s6, s24, s22 // (op 2298) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2298) most significant: sum
s_add_u32 s19, s19, s6 // (op 2298) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 2298) 

s_mov_b32 s60, s26 // (op 2298) reg expression
s_mov_b32 s61, s27 // (op 2298) reg expression
s_mov_b32 s62, s28 // (op 2298) reg expression
s_mov_b32 s63, 131072 // (op 2298) call()

v_accvgpr_write a0, 0.00000 // (op 157) call()
v_accvgpr_write a1, 0.00000 // (op 157) call()
v_accvgpr_write a2, 0.00000 // (op 157) call()
v_accvgpr_write a3, 0.00000 // (op 157) call()
v_accvgpr_write a4, 0.00000 // (op 520) call()
v_accvgpr_write a5, 0.00000 // (op 520) call()
v_accvgpr_write a6, 0.00000 // (op 520) call()
v_accvgpr_write a7, 0.00000 // (op 520) call()
v_lshrrev_b32 v0, 7, v1 // (op 1685) 
s_mul_i32 s15, s16, s50 // (op 1685) 
s_add_u32 s15, s15, s0 // (op 1685) 
v_and_b32 v5, 15, v1 // (op 1685) 
s_add_u32 s6, s15, s9 // (op 1685) 

v_lshl_add_u32 v10, v0, 4, v5 // (op 1685) 


s_mul_hi_u32 s15, s6, s13 // (op 1685) 
v_and_b32 v0, 48, v1 // (op 1685) 
s_sub_u32 s7, s6, s15 // (op 1685) 
s_lshr_b32 s8, s7, 1 // (op 1685) 

s_add_u32 s7, s8, s15 // (op 1685) 


s_lshr_b32 s8, s7, s11 // (op 1685) 

s_and_b32 s7, s11, 2147483648 // (op 1685) 
s_cmp_eq_i32 s7, 0 // (op 1685) 
s_mov_b32 s15, scc // (op 1685) 

s_cmp_eq_i32 s15, 0 // (op 1685) 
s_cselect_b32 s7, s6, s8 // (op 1685) 


s_mul_hi_u32 s8, s7, s12 // (op 1685) 
s_sub_u32 s15, s7, s8 // (op 1685) 
s_lshr_b32 s18, s15, 1 // (op 1685) 

s_add_u32 s15, s18, s8 // (op 1685) 


s_lshr_b32 s8, s15, s10 // (op 1685) 

s_and_b32 s15, s10, 2147483648 // (op 1685) 
s_cmp_eq_i32 s15, 0 // (op 1685) 
s_mov_b32 s18, scc // (op 1685) 

s_cmp_eq_i32 s18, 0 // (op 1685) 
s_cselect_b32 s15, s7, s8 // (op 1685) 


v_lshl_add_u32 v5, s15, 6, v10 // (op 1685) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1685) 
s_mul_i32 s8, s7, s18 // (op 1685) 


s_sub_u32 s7, s6, s8 // (op 1685) 


v_lshl_add_u32 v10, s7, 7, v0 // (op 1685) 


v_mul_lo_u32 v0, v5, s30 // (op 1685) 
v_add_u32 v0, v0, v10 // (op 1685) 


s_mul_i32 s19, s30, s35 // (op 1687) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 1687) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 1687) least significant
s_mul_hi_u32 s6, s30, s34 // (op 1687) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1687) most significant: sum
s_add_u32 s19, s19, s6 // (op 1687) most significant: sum


s_mov_b32 s68, s32 // (op 1687) reg expression
s_mov_b32 s69, s33 // (op 1687) reg expression
s_mov_b32 s70, s18 // (op 1687) reg expression
s_mov_b32 s71, 131072 // (op 1687) call()

v_lshrrev_b32 v5, 7, v1 // (op 1706) 
s_mul_i32 s15, s16, s50 // (op 1706) 
s_add_u32 s15, s15, s0 // (op 1706) 
v_and_b32 v10, 15, v1 // (op 1706) 
s_add_u32 s6, s15, s9 // (op 1706) 

v_lshl_add_u32 v11, v5, 4, v10 // (op 1706) 


s_mul_hi_u32 s15, s6, s13 // (op 1706) 
v_and_b32 v5, 48, v1 // (op 1706) 
s_sub_u32 s7, s6, s15 // (op 1706) 
s_lshr_b32 s8, s7, 1 // (op 1706) 

s_add_u32 s7, s8, s15 // (op 1706) 


s_lshr_b32 s8, s7, s11 // (op 1706) 

s_and_b32 s7, s11, 2147483648 // (op 1706) 
s_cmp_eq_i32 s7, 0 // (op 1706) 
s_mov_b32 s15, scc // (op 1706) 

s_cmp_eq_i32 s15, 0 // (op 1706) 
s_cselect_b32 s7, s6, s8 // (op 1706) 


s_mul_hi_u32 s8, s7, s12 // (op 1706) 
s_sub_u32 s15, s7, s8 // (op 1706) 
s_lshr_b32 s18, s15, 1 // (op 1706) 

s_add_u32 s15, s18, s8 // (op 1706) 


s_lshr_b32 s8, s15, s10 // (op 1706) 

s_and_b32 s15, s10, 2147483648 // (op 1706) 
s_cmp_eq_i32 s15, 0 // (op 1706) 
s_mov_b32 s18, scc // (op 1706) 

s_cmp_eq_i32 s18, 0 // (op 1706) 
s_cselect_b32 s15, s7, s8 // (op 1706) 


v_lshl_add_u32 v10, s15, 6, v11 // (op 1706) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1706) 
s_mul_i32 s8, s7, s18 // (op 1706) 


s_sub_u32 s7, s6, s8 // (op 1706) 


v_lshl_add_u32 v11, s7, 7, v5 // (op 1706) 


v_mul_lo_u32 v5, v10, s30 // (op 1706) 
v_add_u32 v5, v5, v11 // (op 1706) 


s_mul_i32 s19, s30, s35 // (op 1708) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 1708) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 1708) least significant
s_mul_hi_u32 s6, s30, s34 // (op 1708) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1708) most significant: sum
s_add_u32 s19, s19, s6 // (op 1708) most significant: sum


s_mov_b32 s68, s32 // (op 1708) reg expression
s_mov_b32 s69, s33 // (op 1708) reg expression
s_mov_b32 s70, s18 // (op 1708) reg expression
s_mov_b32 s71, 131072 // (op 1708) call()

v_lshrrev_b32 v10, 6, v1 // (op 1832) 
s_mul_i32 s15, s16, s50 // (op 1832) 
s_add_u32 s15, s15, s0 // (op 1832) 
v_and_b32 v11, 1, v10 // (op 1832) 

s_add_u32 s6, s15, s9 // (op 1832) 

v_and_b32 v10, 15, v1 // (op 1832) 
s_mul_hi_u32 s15, s6, s13 // (op 1832) 
v_lshl_add_u32 v12, v11, 4, v10 // (op 1832) 


s_sub_u32 s7, s6, s15 // (op 1832) 
v_and_b32 v10, 63, v1 // (op 1832) 
s_lshr_b32 s8, s7, 1 // (op 1832) 

v_lshrrev_b32 v11, 4, v10 // (op 1832) 

s_add_u32 s7, s8, s15 // (op 1832) 


v_lshlrev_b32 v10, 5, v11 // (op 1832) 

s_lshr_b32 s8, s7, s11 // (op 1832) 

s_and_b32 s7, s11, 2147483648 // (op 1832) 
s_cmp_eq_i32 s7, 0 // (op 1832) 
s_mov_b32 s15, scc // (op 1832) 

s_cmp_eq_i32 s15, 0 // (op 1832) 
s_cselect_b32 s7, s6, s8 // (op 1832) 


s_mul_hi_u32 s8, s7, s12 // (op 1832) 
s_sub_u32 s15, s7, s8 // (op 1832) 
s_lshr_b32 s18, s15, 1 // (op 1832) 

s_add_u32 s15, s18, s8 // (op 1832) 


s_lshr_b32 s8, s15, s10 // (op 1832) 

s_and_b32 s15, s10, 2147483648 // (op 1832) 
s_cmp_eq_i32 s15, 0 // (op 1832) 
s_mov_b32 s18, scc // (op 1832) 

s_cmp_eq_i32 s18, 0 // (op 1832) 
s_cselect_b32 s15, s7, s8 // (op 1832) 


s_mul_i32 s8, s15, s14 // (op 1832) 

s_sub_u32 s15, s7, s8 // (op 1832) 

v_lshl_add_u32 v11, s15, 5, v12 // (op 1832) 


v_mul_lo_u32 v13, v11, s37 // (op 1832) most significant: low of low * high
v_mul_hi_u32 v14, v11, s36 // (op 1832) most significant: high of low * low
v_mul_lo_u32 v12, v11, s36 // (op 1832) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 1832) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1832) 
s_mul_i32 s8, s7, s18 // (op 1832) 


s_sub_u32 s7, s6, s8 // (op 1832) 


v_lshl_add_u32 v11, s7, 7, v10 // (op 1832) 


v_mov_b32 v15, 0 // (op 1832) convert
v_mov_b32 v14, v11 // (op 1832) convert

v_add_co_u32 v10, s[18:19], v14, v12 // (op 1832) least significant half
v_addc_co_u32 v11, s[18:19], v15, v13, s[18:19] // (op 1832) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 1832) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 1832) 

v_mov_b32 v10, v12 // (op 1832) convert

s_mul_i32 s19, s36, s23 // (op 1834) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 1834) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 1834) least significant
s_mul_hi_u32 s6, s36, s22 // (op 1834) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1834) most significant: sum
s_add_u32 s19, s19, s6 // (op 1834) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 1834) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 1834) 

s_mov_b32 s72, s38 // (op 1834) reg expression
s_mov_b32 s73, s39 // (op 1834) reg expression
s_mov_b32 s74, s18 // (op 1834) reg expression
s_mov_b32 s75, 131072 // (op 1834) call()

v_lshrrev_b32 v11, 6, v1 // (op 1851) 
s_mul_i32 s15, s16, s50 // (op 1851) 
s_add_u32 s15, s15, s0 // (op 1851) 
v_and_b32 v12, 1, v11 // (op 1851) 

s_add_u32 s6, s15, s9 // (op 1851) 

v_and_b32 v11, 15, v1 // (op 1851) 
s_mul_hi_u32 s15, s6, s13 // (op 1851) 
v_lshl_add_u32 v13, v12, 4, v11 // (op 1851) 


s_sub_u32 s7, s6, s15 // (op 1851) 
v_and_b32 v11, 63, v1 // (op 1851) 
s_lshr_b32 s8, s7, 1 // (op 1851) 

v_lshrrev_b32 v12, 4, v11 // (op 1851) 

s_add_u32 s7, s8, s15 // (op 1851) 


v_lshlrev_b32 v11, 5, v12 // (op 1851) 

s_lshr_b32 s8, s7, s11 // (op 1851) 

s_and_b32 s7, s11, 2147483648 // (op 1851) 
s_cmp_eq_i32 s7, 0 // (op 1851) 
s_mov_b32 s15, scc // (op 1851) 

s_cmp_eq_i32 s15, 0 // (op 1851) 
s_cselect_b32 s7, s6, s8 // (op 1851) 


s_mul_hi_u32 s8, s7, s12 // (op 1851) 
s_sub_u32 s15, s7, s8 // (op 1851) 
s_lshr_b32 s18, s15, 1 // (op 1851) 

s_add_u32 s15, s18, s8 // (op 1851) 


s_lshr_b32 s8, s15, s10 // (op 1851) 

s_and_b32 s15, s10, 2147483648 // (op 1851) 
s_cmp_eq_i32 s15, 0 // (op 1851) 
s_mov_b32 s18, scc // (op 1851) 

s_cmp_eq_i32 s18, 0 // (op 1851) 
s_cselect_b32 s15, s7, s8 // (op 1851) 


s_mul_i32 s8, s15, s14 // (op 1851) 

s_sub_u32 s15, s7, s8 // (op 1851) 

v_lshl_add_u32 v12, s15, 5, v13 // (op 1851) 


v_mul_lo_u32 v15, v12, s37 // (op 1851) most significant: low of low * high
v_mul_hi_u32 v13, v12, s36 // (op 1851) most significant: high of low * low
v_mul_lo_u32 v14, v12, s36 // (op 1851) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 1851) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1851) 
s_mul_i32 s8, s7, s18 // (op 1851) 


s_sub_u32 s7, s6, s8 // (op 1851) 


v_lshl_add_u32 v12, s7, 7, v11 // (op 1851) 


v_mov_b32 v17, 0 // (op 1851) convert
v_mov_b32 v16, v12 // (op 1851) convert

v_add_co_u32 v12, s[18:19], v16, v14 // (op 1851) least significant half
v_addc_co_u32 v13, s[18:19], v17, v15, s[18:19] // (op 1851) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 1851) 


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1851) 

v_mov_b32 v11, v14 // (op 1851) convert

s_mul_i32 s19, s36, s23 // (op 1853) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 1853) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 1853) least significant
s_mul_hi_u32 s6, s36, s22 // (op 1853) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1853) most significant: sum
s_add_u32 s19, s19, s6 // (op 1853) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 1853) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 1853) 

s_mov_b32 s72, s38 // (op 1853) reg expression
s_mov_b32 s73, s39 // (op 1853) reg expression
s_mov_b32 s74, s18 // (op 1853) reg expression
s_mov_b32 s75, 131072 // (op 1853) call()

v_lshrrev_b32 v12, 2, v1 // (op 1960) 
s_mul_i32 s15, s16, s50 // (op 1960) 
s_add_u32 s15, s15, s0 // (op 1960) 
v_and_b32 v13, -4, v1 // (op 1960) 
s_add_u32 s6, s15, s9 // (op 1960) 

v_sub_u32 v14, v1, v13 // (op 1960) 

s_mul_hi_u32 s15, s6, s13 // (op 1960) 
s_sub_u32 s7, s6, s15 // (op 1960) 
s_lshr_b32 s8, s7, 1 // (op 1960) 

s_add_u32 s7, s8, s15 // (op 1960) 


s_lshr_b32 s8, s7, s11 // (op 1960) 

s_and_b32 s7, s11, 2147483648 // (op 1960) 
s_cmp_eq_i32 s7, 0 // (op 1960) 
s_mov_b32 s15, scc // (op 1960) 

s_cmp_eq_i32 s15, 0 // (op 1960) 
s_cselect_b32 s7, s6, s8 // (op 1960) 


s_mul_hi_u32 s8, s7, s12 // (op 1960) 
s_sub_u32 s15, s7, s8 // (op 1960) 
s_lshr_b32 s18, s15, 1 // (op 1960) 

s_add_u32 s15, s18, s8 // (op 1960) 


s_lshr_b32 s8, s15, s10 // (op 1960) 

s_and_b32 s15, s10, 2147483648 // (op 1960) 
s_cmp_eq_i32 s15, 0 // (op 1960) 
s_mov_b32 s18, scc // (op 1960) 

s_cmp_eq_i32 s18, 0 // (op 1960) 
s_cselect_b32 s15, s7, s8 // (op 1960) 


v_lshl_add_u32 v13, s15, 6, v12 // (op 1960) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1960) 
s_mul_i32 s8, s7, s18 // (op 1960) 


s_sub_u32 s7, s6, s8 // (op 1960) 


v_lshl_add_u32 v12, s7, 2, v14 // (op 1960) 


v_mul_lo_u32 v14, v13, s40 // (op 1960) 
v_add_u32 v14, v14, v12 // (op 1960) 


s_mul_i32 s19, s40, s35 // (op 1962) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 1962) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 1962) least significant
s_mul_hi_u32 s6, s40, s34 // (op 1962) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1962) most significant: sum
s_add_u32 s19, s19, s6 // (op 1962) most significant: sum


s_mov_b32 s76, s42 // (op 1962) reg expression
s_mov_b32 s77, s43 // (op 1962) reg expression
s_mov_b32 s78, s18 // (op 1962) reg expression
s_mov_b32 s79, 131072 // (op 1962) call()

v_lshrrev_b32 v12, 2, v1 // (op 1979) 
s_mul_i32 s15, s16, s50 // (op 1979) 
s_add_u32 s15, s15, s0 // (op 1979) 
v_and_b32 v13, -4, v1 // (op 1979) 
s_add_u32 s6, s15, s9 // (op 1979) 

v_sub_u32 v15, v1, v13 // (op 1979) 

s_mul_hi_u32 s15, s6, s13 // (op 1979) 
s_sub_u32 s7, s6, s15 // (op 1979) 
s_lshr_b32 s8, s7, 1 // (op 1979) 

s_add_u32 s7, s8, s15 // (op 1979) 


s_lshr_b32 s8, s7, s11 // (op 1979) 

s_and_b32 s7, s11, 2147483648 // (op 1979) 
s_cmp_eq_i32 s7, 0 // (op 1979) 
s_mov_b32 s15, scc // (op 1979) 

s_cmp_eq_i32 s15, 0 // (op 1979) 
s_cselect_b32 s7, s6, s8 // (op 1979) 


s_mul_hi_u32 s8, s7, s12 // (op 1979) 
s_sub_u32 s15, s7, s8 // (op 1979) 
s_lshr_b32 s18, s15, 1 // (op 1979) 

s_add_u32 s15, s18, s8 // (op 1979) 


s_lshr_b32 s8, s15, s10 // (op 1979) 

s_and_b32 s15, s10, 2147483648 // (op 1979) 
s_cmp_eq_i32 s15, 0 // (op 1979) 
s_mov_b32 s18, scc // (op 1979) 

s_cmp_eq_i32 s18, 0 // (op 1979) 
s_cselect_b32 s15, s7, s8 // (op 1979) 


v_lshl_add_u32 v13, s15, 6, v12 // (op 1979) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1979) 
s_mul_i32 s8, s7, s18 // (op 1979) 


s_sub_u32 s7, s6, s8 // (op 1979) 


v_lshl_add_u32 v12, s7, 2, v15 // (op 1979) 


v_mul_lo_u32 v15, v13, s40 // (op 1979) 
v_add_u32 v15, v15, v12 // (op 1979) 


s_mul_i32 s19, s40, s35 // (op 1981) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 1981) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 1981) least significant
s_mul_hi_u32 s6, s40, s34 // (op 1981) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1981) most significant: sum
s_add_u32 s19, s19, s6 // (op 1981) most significant: sum


s_mov_b32 s76, s42 // (op 1981) reg expression
s_mov_b32 s77, s43 // (op 1981) reg expression
s_mov_b32 s78, s18 // (op 1981) reg expression
s_mov_b32 s79, 131072 // (op 1981) call()

v_lshrrev_b32 v12, 6, v1 // (op 2088) 
s_mul_i32 s15, s16, s50 // (op 2088) 
s_add_u32 s15, s15, s0 // (op 2088) 
v_and_b32 v13, 1, v12 // (op 2088) 

s_add_u32 s6, s15, s9 // (op 2088) 

v_and_b32 v12, 15, v1 // (op 2088) 
s_mul_hi_u32 s15, s6, s13 // (op 2088) 
v_lshl_add_u32 v16, v13, 4, v12 // (op 2088) 


s_sub_u32 s7, s6, s15 // (op 2088) 
v_and_b32 v12, 63, v1 // (op 2088) 
s_lshr_b32 s8, s7, 1 // (op 2088) 

v_lshrrev_b32 v13, 4, v12 // (op 2088) 

s_add_u32 s7, s8, s15 // (op 2088) 


s_lshr_b32 s8, s7, s11 // (op 2088) 

s_and_b32 s7, s11, 2147483648 // (op 2088) 
s_cmp_eq_i32 s7, 0 // (op 2088) 
s_mov_b32 s15, scc // (op 2088) 

s_cmp_eq_i32 s15, 0 // (op 2088) 
s_cselect_b32 s7, s6, s8 // (op 2088) 


s_mul_hi_u32 s8, s7, s12 // (op 2088) 
s_sub_u32 s15, s7, s8 // (op 2088) 
s_lshr_b32 s18, s15, 1 // (op 2088) 

s_add_u32 s15, s18, s8 // (op 2088) 


s_lshr_b32 s8, s15, s10 // (op 2088) 

s_and_b32 s15, s10, 2147483648 // (op 2088) 
s_cmp_eq_i32 s15, 0 // (op 2088) 
s_mov_b32 s18, scc // (op 2088) 

s_cmp_eq_i32 s18, 0 // (op 2088) 
s_cselect_b32 s15, s7, s8 // (op 2088) 


s_mul_i32 s8, s15, s14 // (op 2088) 

s_sub_u32 s15, s7, s8 // (op 2088) 

v_lshl_add_u32 v12, s15, 5, v16 // (op 2088) 


v_mul_lo_u32 v16, v12, s44 // (op 2088) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 2088) 
s_mul_i32 s8, s7, s18 // (op 2088) 


s_sub_u32 s7, s6, s8 // (op 2088) 


v_lshl_add_u32 v12, s7, 2, v13 // (op 2088) 


v_add_u32 v13, v12, v16 // (op 2088) 


s_mul_i32 s19, s44, s23 // (op 2090) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 2090) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 2090) least significant
s_mul_hi_u32 s6, s44, s22 // (op 2090) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2090) most significant: sum
s_add_u32 s19, s19, s6 // (op 2090) most significant: sum


s_mov_b32 s80, s46 // (op 2090) reg expression
s_mov_b32 s81, s47 // (op 2090) reg expression
s_mov_b32 s82, s18 // (op 2090) reg expression
s_mov_b32 s83, 131072 // (op 2090) call()

v_lshrrev_b32 v12, 6, v1 // (op 2107) 
s_mul_i32 s15, s16, s50 // (op 2107) 
s_add_u32 s15, s15, s0 // (op 2107) 
v_and_b32 v16, 1, v12 // (op 2107) 

s_add_u32 s6, s15, s9 // (op 2107) 

v_and_b32 v12, 15, v1 // (op 2107) 
s_mul_hi_u32 s15, s6, s13 // (op 2107) 
v_lshl_add_u32 v17, v16, 4, v12 // (op 2107) 


s_sub_u32 s7, s6, s15 // (op 2107) 
v_and_b32 v12, 63, v1 // (op 2107) 
s_lshr_b32 s8, s7, 1 // (op 2107) 

v_lshrrev_b32 v16, 4, v12 // (op 2107) 

s_add_u32 s7, s8, s15 // (op 2107) 


s_lshr_b32 s8, s7, s11 // (op 2107) 

s_and_b32 s7, s11, 2147483648 // (op 2107) 
s_cmp_eq_i32 s7, 0 // (op 2107) 
s_mov_b32 s15, scc // (op 2107) 

s_cmp_eq_i32 s15, 0 // (op 2107) 
s_cselect_b32 s7, s6, s8 // (op 2107) 


s_mul_hi_u32 s8, s7, s12 // (op 2107) 
s_sub_u32 s15, s7, s8 // (op 2107) 
s_lshr_b32 s18, s15, 1 // (op 2107) 

s_add_u32 s15, s18, s8 // (op 2107) 


s_lshr_b32 s8, s15, s10 // (op 2107) 

s_and_b32 s15, s10, 2147483648 // (op 2107) 
s_cmp_eq_i32 s15, 0 // (op 2107) 
s_mov_b32 s18, scc // (op 2107) 

s_cmp_eq_i32 s18, 0 // (op 2107) 
s_cselect_b32 s15, s7, s8 // (op 2107) 


s_mul_i32 s8, s15, s14 // (op 2107) 

s_sub_u32 s15, s7, s8 // (op 2107) 

v_lshl_add_u32 v12, s15, 5, v17 // (op 2107) 


v_mul_lo_u32 v17, v12, s44 // (op 2107) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 2107) 
s_mul_i32 s8, s7, s18 // (op 2107) 


s_sub_u32 s7, s6, s8 // (op 2107) 


v_lshl_add_u32 v12, s7, 2, v16 // (op 2107) 


v_add_u32 v16, v12, v17 // (op 2107) 


s_mul_i32 s19, s44, s23 // (op 2109) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 2109) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 2109) least significant
s_mul_hi_u32 s6, s44, s22 // (op 2109) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2109) most significant: sum
s_add_u32 s19, s19, s6 // (op 2109) most significant: sum


s_mov_b32 s80, s46 // (op 2109) reg expression
s_mov_b32 s81, s47 // (op 2109) reg expression
s_mov_b32 s82, s18 // (op 2109) reg expression
s_mov_b32 s83, 131072 // (op 2109) call()




buffer_load_ubyte v12, v13, s[80:83], 0 offen offset: 0 // (op 1398) Load value


v_mov_b32 v17, v0 // (op 1404) reg expression

buffer_load_dwordx4 v[18:21], v17, s[68:71], 0 offen // (op 1404) Load value
buffer_load_dwordx4 v[22:25], v17, s[68:71], 0 offen offset:64 // (op 1404) Load value




buffer_load_dwordx4 v[26:29], v10, s[72:75], 0 offen // (op 1410) Load value


s_lshl_b32 s15, s30, 5 // (op 1416) 
v_add_u32 v17, v0, s15 // (op 1416) 


buffer_load_dwordx4 v[30:33], v17, s[68:71], 0 offen // (op 1416) Load value
buffer_load_dwordx4 v[34:37], v17, s[68:71], 0 offen offset:64 // (op 1416) Load value




buffer_load_ubyte v17, v14, s[76:79], 0 offen offset: 0 // (op 1421) Load value



buffer_load_ubyte v38, v13, s[80:83], 0 offen offset:4 // (op 1425) Load value



v_add_u32 v13, 128, v0 // (op 1431) 

buffer_load_dwordx4 v[40:43], v13, s[68:71], 0 offen // (op 1431) Load value
buffer_load_dwordx4 v[44:47], v13, s[68:71], 0 offen offset:64 // (op 1431) Load value




buffer_load_dwordx4 v[48:51], v10, s[72:75], 0 offen offset:64 // (op 1437) Load value



s_lshl_b32 s15, s30, 5 // (op 1443) 
s_add_u32 s15, s15, 128 // (op 1443) 
v_add_u32 v10, v0, s15 // (op 1443) 


buffer_load_dwordx4 v[52:55], v10, s[68:71], 0 offen // (op 1443) Load value
buffer_load_dwordx4 v[56:59], v10, s[68:71], 0 offen offset:64 // (op 1443) Load value





buffer_load_ubyte v0, v14, s[76:79], 0 offen offset:4 // (op 1448) Load value

v_and_b32 v10, -4, v1 // (op 2342) 
v_sub_u32 v39, v1, v10 // (op 2342) 

v_and_b32 v10, 4294967292, v1 // (op 2342) 
v_add_u32 v13, v10, v39 // (op 2342) 


s_barrier  // (op 2546) 



s_waitcnt vmcnt(7)
ds_write_b8 v13, v17 // (op 1450) Store local data 

v_and_b32 v10, -4, v1 // (op 2313) 
v_sub_u32 v39, v1, v10 // (op 2313) 

v_and_b32 v10, 4294967292, v1 // (op 2313) 
v_add_u32 v13, v10, v39 // (op 2313) 


v_and_b32 v10, -4, v1 // (op 2413) 
v_sub_u32 v14, v1, v10 // (op 2413) 

v_and_b32 v10, 4294967292, v1 // (op 2413) 
v_add_u32 v39, v10, v14 // (op 2413) 


s_mov_b32 s1, 0 // (op 522) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 521) 
s_mov_b32 s15, s18 // (op 521) convert

s_and_b32 s6, s15, 4294967294 // (op 521) 

s_cmp_lt_u32 s1, s6 // (op 521) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_521 // (op 521) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_521 if false)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoop_521:
s_add_u32 s1, s1, 2 // (op 523) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2543) 



ds_read_u8 v10, v3 // (op 164) Load local data 



ds_read_u8 v14, v3 offset:128 // (op 547) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:25], v[26:29], a[0:3], v10, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 167) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:37], v[26:29], a[4:7], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 549) 



v_mov_b32 v10, v16 // (op 166) 

buffer_load_ubyte v12, v10, s[80:83], 0 offen offset:8 // (op 166) Load value


v_mov_b32 v14, v5 // (op 169) 
v_add_u32 v60, 256, v14 // (op 169) 

buffer_load_dwordx4 v[18:21], v60, s[68:71], 0 offen // (op 169) Load value
buffer_load_dwordx4 v[22:25], v60, s[68:71], 0 offen offset:64 // (op 169) Load value



v_mov_b32 v60, v11 // (op 171) 

buffer_load_dwordx4 v[26:29], v60, s[72:75], 0 offen offset:128 // (op 171) Load value


s_lshl_b32 s15, s30, 5 // (op 572) 
s_add_u32 s15, s15, 256 // (op 572) 
v_add_u32 v61, v14, s15 // (op 572) 


buffer_load_dwordx4 v[30:33], v61, s[68:71], 0 offen // (op 572) Load value
buffer_load_dwordx4 v[34:37], v61, s[68:71], 0 offen offset:64 // (op 572) Load value



v_mov_b32 v61, v15 // (op 161) 

buffer_load_ubyte v17, v61, s[76:79], 0 offen offset:8 // (op 161) Load value
s_barrier  // (op 2562) 



s_waitcnt vmcnt(7)
ds_write_b8 v39, v0 offset:256 // (op 460) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2559) 



ds_read_u8 v62, v4 offset:256 // (op 462) Load local data 



ds_read_u8 v63, v4 offset:384 // (op 556) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[48:51], a[0:3], v62, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 465) 

s_barrier  // (op 2665) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:59], v[48:51], a[4:7], v63, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 550) 




buffer_load_ubyte v38, v10, s[80:83], 0 offen offset:12 // (op 464) Load value



v_add_u32 v16, 8, v16 // (op 2099) 
v_add_u32 v10, 384, v14 // (op 467) 

buffer_load_dwordx4 v[40:43], v10, s[68:71], 0 offen // (op 467) Load value
buffer_load_dwordx4 v[44:47], v10, s[68:71], 0 offen offset:64 // (op 467) Load value




buffer_load_dwordx4 v[48:51], v60, s[72:75], 0 offen offset:192 // (op 469) Load value



v_add_u32 v11, 128, v11 // (op 1843) 
s_lshl_b32 s15, s30, 5 // (op 565) 
s_add_u32 s15, s15, 384 // (op 565) 
v_add_u32 v10, v14, s15 // (op 565) 


buffer_load_dwordx4 v[52:55], v10, s[68:71], 0 offen // (op 565) Load value
buffer_load_dwordx4 v[56:59], v10, s[68:71], 0 offen offset:64 // (op 565) Load value





v_add_u32 v5, 256, v5 // (op 1696) 
buffer_load_ubyte v0, v61, s[76:79], 0 offen offset:12 // (op 459) Load value

s_barrier  // (op 2549) 
v_add_u32 v15, 8, v15 // (op 1971) 



s_waitcnt vmcnt(7)
ds_write_b8 v13, v17 // (op 162) Store local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 521) 
s_mov_b32 s15, s18 // (op 521) convert

s_and_b32 s6, s15, 4294967294 // (op 521) 

s_cmp_lt_u32 s1, s6 // (op 521) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoop_521 // (op 521) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoop_521 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoop_521:











v_lshrrev_b32 v5, 7, v1 // (op 1621) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1621) 
v_and_b32 v39, 15, v1 // (op 1621) 
s_and_b32 s15, s18, 4294967294 // (op 1621) 
v_lshl_add_u32 v10, v5, 4, v39 // (op 1621) 


s_add_u32 s6, s0, s15 // (op 1621) 

v_and_b32 v5, 48, v1 // (op 1621) 
s_mul_i32 s15, s16, s50 // (op 1621) 
s_add_u32 s15, s15, s6 // (op 1621) 

s_add_u32 s6, s15, s9 // (op 1621) 

s_mul_hi_u32 s15, s6, s13 // (op 1621) 
s_sub_u32 s7, s6, s15 // (op 1621) 
s_lshr_b32 s8, s7, 1 // (op 1621) 

s_add_u32 s7, s8, s15 // (op 1621) 


s_lshr_b32 s8, s7, s11 // (op 1621) 

s_and_b32 s7, s11, 2147483648 // (op 1621) 
s_cmp_eq_i32 s7, 0 // (op 1621) 
s_mov_b32 s15, scc // (op 1621) 

s_cmp_eq_i32 s15, 0 // (op 1621) 
s_cselect_b32 s7, s6, s8 // (op 1621) 


s_mul_hi_u32 s8, s7, s12 // (op 1621) 
s_sub_u32 s15, s7, s8 // (op 1621) 
s_lshr_b32 s28, s15, 1 // (op 1621) 

s_add_u32 s15, s28, s8 // (op 1621) 


s_lshr_b32 s8, s15, s10 // (op 1621) 

s_and_b32 s15, s10, 2147483648 // (op 1621) 
s_cmp_eq_i32 s15, 0 // (op 1621) 
s_mov_b32 s28, scc // (op 1621) 

s_cmp_eq_i32 s28, 0 // (op 1621) 
s_cselect_b32 s15, s7, s8 // (op 1621) 


s_mul_i32 s8, s7, s18 // (op 1621) 


v_lshl_add_u32 v39, s15, 6, v10 // (op 1621) 


s_sub_u32 s7, s6, s8 // (op 1621) 


v_lshl_add_u32 v10, s7, 7, v5 // (op 1621) 


v_mul_lo_u32 v5, v39, s30 // (op 1621) 
v_add_u32 v5, v5, v10 // (op 1621) 


s_mul_i32 s19, s30, s35 // (op 1623) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 1623) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 1623) least significant
s_mul_hi_u32 s6, s30, s34 // (op 1623) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1623) most significant: sum
s_add_u32 s19, s19, s6 // (op 1623) most significant: sum


s_mov_b32 s68, s32 // (op 1623) reg expression
s_mov_b32 s69, s33 // (op 1623) reg expression
s_mov_b32 s70, s18 // (op 1623) reg expression
s_mov_b32 s71, 131072 // (op 1623) call()

v_lshrrev_b32 v39, 6, v1 // (op 1787) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1787) 
v_and_b32 v10, 1, v39 // (op 1787) 

s_and_b32 s15, s18, 4294967294 // (op 1787) 
v_and_b32 v39, 15, v1 // (op 1787) 
s_add_u32 s6, s0, s15 // (op 1787) 

v_lshl_add_u32 v11, v10, 4, v39 // (op 1787) 


v_and_b32 v10, 63, v1 // (op 1787) 
s_mul_i32 s15, s16, s50 // (op 1787) 
s_add_u32 s15, s15, s6 // (op 1787) 

v_lshrrev_b32 v39, 4, v10 // (op 1787) 

s_add_u32 s6, s15, s9 // (op 1787) 

v_lshlrev_b32 v10, 5, v39 // (op 1787) 

s_mul_hi_u32 s15, s6, s13 // (op 1787) 
s_sub_u32 s7, s6, s15 // (op 1787) 
s_lshr_b32 s8, s7, 1 // (op 1787) 

s_add_u32 s7, s8, s15 // (op 1787) 


s_lshr_b32 s8, s7, s11 // (op 1787) 

s_and_b32 s7, s11, 2147483648 // (op 1787) 
s_cmp_eq_i32 s7, 0 // (op 1787) 
s_mov_b32 s15, scc // (op 1787) 

s_cmp_eq_i32 s15, 0 // (op 1787) 
s_cselect_b32 s7, s6, s8 // (op 1787) 


s_mul_hi_u32 s8, s7, s12 // (op 1787) 
s_sub_u32 s15, s7, s8 // (op 1787) 
s_lshr_b32 s28, s15, 1 // (op 1787) 

s_add_u32 s15, s28, s8 // (op 1787) 


s_lshr_b32 s8, s15, s10 // (op 1787) 

s_and_b32 s15, s10, 2147483648 // (op 1787) 
s_cmp_eq_i32 s15, 0 // (op 1787) 
s_mov_b32 s28, scc // (op 1787) 

s_cmp_eq_i32 s28, 0 // (op 1787) 
s_cselect_b32 s15, s7, s8 // (op 1787) 


s_mul_i32 s8, s15, s14 // (op 1787) 

s_sub_u32 s15, s7, s8 // (op 1787) 

s_mul_i32 s8, s7, s18 // (op 1787) 


v_lshl_add_u32 v39, s15, 5, v11 // (op 1787) 


s_sub_u32 s7, s6, s8 // (op 1787) 


v_mul_lo_u32 v37, v39, s37 // (op 1787) most significant: low of low * high
v_mul_hi_u32 v11, v39, s36 // (op 1787) most significant: high of low * low
v_mul_lo_u32 v36, v39, s36 // (op 1787) least significant: low of low * low
v_add_u32 v37, v37, v11 // (op 1787) most significant: sum


v_lshl_add_u32 v39, s7, 7, v10 // (op 1787) 


v_mov_b32 v11, 0 // (op 1787) convert
v_mov_b32 v10, v39 // (op 1787) convert

v_add_co_u32 v12, s[18:19], v10, v36 // (op 1787) least significant half
v_addc_co_u32 v13, s[18:19], v11, v37, s[18:19] // (op 1787) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 1787) 


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 1787) 

v_mov_b32 v39, v10 // (op 1787) convert

s_mul_i32 s19, s36, s23 // (op 1789) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 1789) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 1789) least significant
s_mul_hi_u32 s6, s36, s22 // (op 1789) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1789) most significant: sum
s_add_u32 s19, s19, s6 // (op 1789) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 1789) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 1789) 

s_mov_b32 s72, s38 // (op 1789) reg expression
s_mov_b32 s73, s39 // (op 1789) reg expression
s_mov_b32 s74, s18 // (op 1789) reg expression
s_mov_b32 s75, 131072 // (op 1789) call()

v_lshrrev_b32 v10, 2, v1 // (op 1915) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1915) 
v_and_b32 v11, -4, v1 // (op 1915) 
s_and_b32 s15, s18, 4294967294 // (op 1915) 
v_sub_u32 v12, v1, v11 // (op 1915) 

s_add_u32 s6, s0, s15 // (op 1915) 

s_mul_i32 s15, s16, s50 // (op 1915) 
s_add_u32 s15, s15, s6 // (op 1915) 

s_add_u32 s6, s15, s9 // (op 1915) 

s_mul_hi_u32 s15, s6, s13 // (op 1915) 
s_sub_u32 s7, s6, s15 // (op 1915) 
s_lshr_b32 s8, s7, 1 // (op 1915) 

s_add_u32 s7, s8, s15 // (op 1915) 


s_lshr_b32 s8, s7, s11 // (op 1915) 

s_and_b32 s7, s11, 2147483648 // (op 1915) 
s_cmp_eq_i32 s7, 0 // (op 1915) 
s_mov_b32 s15, scc // (op 1915) 

s_cmp_eq_i32 s15, 0 // (op 1915) 
s_cselect_b32 s7, s6, s8 // (op 1915) 


s_mul_hi_u32 s8, s7, s12 // (op 1915) 
s_sub_u32 s15, s7, s8 // (op 1915) 
s_lshr_b32 s28, s15, 1 // (op 1915) 

s_add_u32 s15, s28, s8 // (op 1915) 


s_lshr_b32 s8, s15, s10 // (op 1915) 

s_and_b32 s15, s10, 2147483648 // (op 1915) 
s_cmp_eq_i32 s15, 0 // (op 1915) 
s_mov_b32 s28, scc // (op 1915) 

s_cmp_eq_i32 s28, 0 // (op 1915) 
s_cselect_b32 s15, s7, s8 // (op 1915) 


s_mul_i32 s8, s7, s18 // (op 1915) 


v_lshl_add_u32 v11, s15, 6, v10 // (op 1915) 


s_sub_u32 s7, s6, s8 // (op 1915) 


v_lshl_add_u32 v10, s7, 2, v12 // (op 1915) 


v_mul_lo_u32 v12, v11, s40 // (op 1915) 
v_add_u32 v12, v12, v10 // (op 1915) 


s_mul_i32 s19, s40, s35 // (op 1917) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 1917) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 1917) least significant
s_mul_hi_u32 s6, s40, s34 // (op 1917) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1917) most significant: sum
s_add_u32 s19, s19, s6 // (op 1917) most significant: sum


s_mov_b32 s76, s42 // (op 1917) reg expression
s_mov_b32 s77, s43 // (op 1917) reg expression
s_mov_b32 s78, s18 // (op 1917) reg expression
s_mov_b32 s79, 131072 // (op 1917) call()

v_lshrrev_b32 v10, 6, v1 // (op 2043) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2043) 
v_and_b32 v11, 1, v10 // (op 2043) 

s_and_b32 s15, s18, 4294967294 // (op 2043) 
v_and_b32 v10, 15, v1 // (op 2043) 
s_add_u32 s6, s0, s15 // (op 2043) 

v_lshl_add_u32 v13, v11, 4, v10 // (op 2043) 


v_and_b32 v10, 63, v1 // (op 2043) 
s_mul_i32 s15, s16, s50 // (op 2043) 
s_add_u32 s15, s15, s6 // (op 2043) 

v_lshrrev_b32 v11, 4, v10 // (op 2043) 

s_add_u32 s6, s15, s9 // (op 2043) 

s_mul_hi_u32 s15, s6, s13 // (op 2043) 
s_sub_u32 s7, s6, s15 // (op 2043) 
s_lshr_b32 s8, s7, 1 // (op 2043) 

s_add_u32 s7, s8, s15 // (op 2043) 


s_lshr_b32 s8, s7, s11 // (op 2043) 

s_and_b32 s7, s11, 2147483648 // (op 2043) 
s_cmp_eq_i32 s7, 0 // (op 2043) 
s_mov_b32 s15, scc // (op 2043) 

s_cmp_eq_i32 s15, 0 // (op 2043) 
s_cselect_b32 s7, s6, s8 // (op 2043) 


s_mul_hi_u32 s8, s7, s12 // (op 2043) 
s_sub_u32 s15, s7, s8 // (op 2043) 
s_lshr_b32 s28, s15, 1 // (op 2043) 

s_add_u32 s15, s28, s8 // (op 2043) 


s_lshr_b32 s8, s15, s10 // (op 2043) 

s_and_b32 s15, s10, 2147483648 // (op 2043) 
s_cmp_eq_i32 s15, 0 // (op 2043) 
s_mov_b32 s28, scc // (op 2043) 

s_cmp_eq_i32 s28, 0 // (op 2043) 
s_cselect_b32 s15, s7, s8 // (op 2043) 


s_mul_i32 s8, s15, s14 // (op 2043) 

s_sub_u32 s15, s7, s8 // (op 2043) 

v_lshl_add_u32 v10, s15, 5, v13 // (op 2043) 


s_mul_i32 s8, s7, s18 // (op 2043) 


v_mul_lo_u32 v13, v10, s44 // (op 2043) 

s_sub_u32 s7, s6, s8 // (op 2043) 


v_lshl_add_u32 v10, s7, 2, v11 // (op 2043) 


v_add_u32 v11, v10, v13 // (op 2043) 


s_mul_i32 s19, s44, s23 // (op 2045) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 2045) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 2045) least significant
s_mul_hi_u32 s6, s44, s22 // (op 2045) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2045) most significant: sum
s_add_u32 s19, s19, s6 // (op 2045) most significant: sum


s_mov_b32 s80, s46 // (op 2045) reg expression
s_mov_b32 s81, s47 // (op 2045) reg expression
s_mov_b32 s82, s18 // (op 2045) reg expression
s_mov_b32 s83, 131072 // (op 2045) call()

v_and_b32 v10, -4, v1 // (op 2326) 
v_sub_u32 v13, v1, v10 // (op 2326) 

v_and_b32 v10, 4294967292, v1 // (op 2326) 
v_add_u32 v14, v10, v13 // (op 2326) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 527) 
s_and_b32 s15, s18, 4294967294 // (op 527) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 526) 
s_mov_b32 s6, s18 // (op 526) convert

s_cmp_lt_u32 s15, s6 // (op 526) 






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_526 // (op 526) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_526 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_526:


s_add_u32 s15, s15, 1 // (op 528) 
v_mov_b32 v0, v12 // (op 417) 

buffer_load_ubyte v10, v0, s[76:79], 0 offen offset: 0 // (op 417) Load value

v_add_u32 v12, 4, v12 // (op 1907) 


s_barrier  // (op 2552) 
v_mov_b32 v0, v5 // (op 425) 



v_mov_b32 v13, v0 // (op 425) reg expression

buffer_load_dwordx4 v[16:19], v13, s[68:71], 0 offen // (op 425) Load value
buffer_load_dwordx4 v[20:23], v13, s[68:71], 0 offen offset:64 // (op 425) Load value

s_waitcnt vmcnt(2)
ds_write_b8 v14, v10 // (op 418) Store local data 



v_mov_b32 v10, v39 // (op 427) 

buffer_load_dwordx4 v[24:27], v10, s[72:75], 0 offen // (op 427) Load value

v_add_u32 v39, 64, v39 // (op 1779) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2555) 



ds_read_u8 v10, v3 // (op 420) Load local data 


v_mov_b32 v13, v11 // (op 422) 

buffer_load_ubyte v15, v13, s[80:83], 0 offen offset: 0 // (op 422) Load value

v_add_u32 v11, 4, v11 // (op 2035) 


s_lshl_b32 s6, s30, 5 // (op 542) 
v_add_u32 v13, v0, s6 // (op 542) 


buffer_load_dwordx4 v[28:31], v13, s[68:71], 0 offen // (op 542) Load value
s_waitcnt vmcnt(1) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:23], v[24:27], a[0:3], v10, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 423) 
buffer_load_dwordx4 v[32:35], v13, s[68:71], 0 offen offset:64 // (op 542) Load value







v_add_u32 v5, 128, v5 // (op 1609) 
ds_read_u8 v0, v3 offset:128 // (op 532) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[28:35], v[24:27], a[4:7], v0, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 534) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 526) 
s_mov_b32 s6, s18 // (op 526) convert

s_cmp_lt_u32 s15, s6 // (op 526) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_526 // (op 526) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_KLoopTail_526 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_KLoopTail_526:






s_barrier  // (op 2651) 
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 193) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_193 // (op 193) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_193
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


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen // (op 204) Store value


v_accvgpr_read v10, a4 // (op 684) 
v_accvgpr_read v11, a5 // (op 684) 
v_accvgpr_read v12, a6 // (op 684) 
v_accvgpr_read v13, a7 // (op 684) 
v_mul_f32 v14, v7, v10 // (op 684) 
v_mul_f32 v15, v7, v11 // (op 684) 
v_mul_f32 v16, v7, v12 // (op 684) 
v_mul_f32 v17, v7, v13 // (op 684) 

v_cvt_f16_f32 v10, v14 // (op 686) 
v_cvt_f16_f32 v11, v15 // (op 686) 
v_cvt_f16_f32 v12, v16 // (op 686) 
v_cvt_f16_f32 v13, v17 // (op 686) 


v_pack_b32_f16 v14, v10, v11 // (op 687) 
v_pack_b32_f16 v15, v12, v13 // (op 687) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen offset:64 // (op 687) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_193 // (op 193) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_193
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalFalse_0_beta_193:



buffer_load_dwordx2 v[10:11], v8, s[56:59], 0 offen offset:64 // (op 717) Load value
v_accvgpr_read v12, a4 // (op 713) 
v_accvgpr_read v13, a5 // (op 713) 
v_accvgpr_read v14, a6 // (op 713) 
v_accvgpr_read v15, a7 // (op 713) 
v_mul_f32 v16, v7, v12 // (op 713) 
v_mul_f32 v17, v7, v13 // (op 713) 
v_mul_f32 v18, v7, v14 // (op 713) 
v_mul_f32 v19, v7, v15 // (op 713) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 718) 
v_lshrrev_b32 v13, 16, v10 // (op 718) 
v_cvt_f32_f16 v13, v13 // (op 718) 
v_mul_f32 v20, v6, v12 // (op 718) 
v_mul_f32 v21, v6, v13 // (op 718) 
v_cvt_f32_f16 v12, v11 // (op 718) 
v_lshrrev_b32 v13, 16, v11 // (op 718) 
v_cvt_f32_f16 v13, v13 // (op 718) 
v_mul_f32 v22, v6, v12 // (op 718) 
v_mul_f32 v23, v6, v13 // (op 718) 


v_add_f32 v10, v20, v16 // (op 714) 
v_add_f32 v11, v21, v17 // (op 714) 
v_add_f32 v12, v22, v18 // (op 714) 
v_add_f32 v13, v23, v19 // (op 714) 


v_cvt_f16_f32 v14, v10 // (op 715) 
v_cvt_f16_f32 v15, v11 // (op 715) 
v_cvt_f16_f32 v16, v12 // (op 715) 
v_cvt_f16_f32 v17, v13 // (op 715) 


v_pack_b32_f16 v10, v14, v15 // (op 716) 
v_pack_b32_f16 v11, v16, v17 // (op 716) 


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen offset:64 // (op 716) Store value





buffer_load_dwordx2 v[10:11], v8, s[56:59], 0 offen // (op 218) Load value
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


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen // (op 217) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ConditionalBottom_0_beta_193:
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
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_DPStreamTileLoop_404 // (op 404) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopTop_DPStreamTileLoop_404 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_404:





































s_endpgm  // End of RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2
.LRR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_end:
.size RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2, .LRR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2_end-RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2
  .amdhsa_next_free_vgpr 76
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 512 // lds bytes
  .amdhsa_accum_offset 68
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
    macrotile: [64, 32, 128]
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
  MIWaveTile: [2, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2
    .symbol: RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x32x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 512
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 86
    .vgpr_count: 67
    .agpr_count: 8
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

