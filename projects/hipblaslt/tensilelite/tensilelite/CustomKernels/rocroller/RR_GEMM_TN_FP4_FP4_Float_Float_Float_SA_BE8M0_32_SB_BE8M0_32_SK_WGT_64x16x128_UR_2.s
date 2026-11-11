// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2,@function
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 2243) 
v_and_b32 v3, 15, v1 // (op 2243) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2243) 


v_and_b32 v0, 63, v1 // (op 2243) 
v_lshrrev_b32 v3, 4, v0 // (op 2243) 

v_lshl_add_u32 v0, v4, 2, v3 // (op 2243) 


v_lshrrev_b32 v3, 6, v1 // (op 2343) 
v_and_b32 v4, 15, v1 // (op 2343) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 2343) 


v_and_b32 v3, 63, v1 // (op 2343) 
v_lshrrev_b32 v4, 4, v3 // (op 2343) 

v_lshl_add_u32 v3, v5, 2, v4 // (op 2343) 


v_lshrrev_b32 v4, 6, v1 // (op 2414) 
v_and_b32 v5, 15, v1 // (op 2414) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 2414) 


v_and_b32 v4, 63, v1 // (op 2414) 
v_lshrrev_b32 v5, 4, v4 // (op 2414) 

v_lshl_add_u32 v4, v6, 2, v5 // (op 2414) 


v_lshrrev_b32 v5, 6, v1 // (op 2443) 
v_and_b32 v6, 15, v1 // (op 2443) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 2443) 


v_and_b32 v5, 63, v1 // (op 2443) 
v_lshrrev_b32 v6, 4, v5 // (op 2443) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 2443) 


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
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_241 // (op 241) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_241 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_SKStreamTileLoop_241:
v_lshrrev_b32 v8, 6, v1 // (op 2158) 
s_mul_i32 s52, s16, s8 // (op 2158) 
s_add_u32 s52, s52, s0 // (op 2158) 
v_and_b32 v9, 63, v1 // (op 2158) 
s_mul_hi_u32 s53, s52, s13 // (op 2158) 
v_lshrrev_b32 v10, 4, v9 // (op 2158) 
s_sub_u32 s54, s52, s53 // (op 2158) 
v_lshlrev_b32 v11, 2, v10 // (op 2158) 

v_lshl_add_u32 v10, v8, 4, v11 // (op 2158) 


s_lshr_b32 s55, s54, 1 // (op 2158) 

s_add_u32 s54, s55, s53 // (op 2158) 


v_lshrrev_b32 v8, 2, v9 // (op 2158) 

s_lshr_b32 s53, s54, s11 // (op 2158) 

v_and_b32 v9, 3, v8 // (op 2158) 

s_and_b32 s54, s11, 2147483648 // (op 2158) 
v_and_b32 v8, 3, v1 // (op 2158) 
s_cmp_eq_i32 s54, 0 // (op 2158) 
s_mov_b32 s55, scc // (op 2158) 

v_lshl_add_u32 v11, v9, 2, v8 // (op 2158) 


s_cmp_eq_i32 s55, 0 // (op 2158) 
s_cselect_b32 s54, s52, s53 // (op 2158) 



s_mul_hi_u32 s52, s54, s12 // (op 2158) 
s_sub_u32 s53, s54, s52 // (op 2158) 
s_lshr_b32 s55, s53, 1 // (op 2158) 

s_add_u32 s53, s55, s52 // (op 2158) 


s_lshr_b32 s52, s53, s10 // (op 2158) 

s_and_b32 s53, s10, 2147483648 // (op 2158) 
s_cmp_eq_i32 s53, 0 // (op 2158) 
s_mov_b32 s55, scc // (op 2158) 

s_cmp_eq_i32 s55, 0 // (op 2158) 
s_cselect_b32 s53, s54, s52 // (op 2158) 


s_mul_i32 s52, s53, s14 // (op 2158) 
v_lshl_add_u32 v8, s53, 6, v10 // (op 2158) 


s_sub_u32 s53, s54, s52 // (op 2158) 


v_lshl_add_u32 v9, s53, 4, v11 // (op 2158) 


v_mul_lo_u32 v10, v9, s4 // (op 2158) 

v_add_lshl_u32 v9, v8, v10, 2 // (op 2158) 


s_mul_i32 s53, s4, s23 // (op 2160) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 2160) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 2160) least significant
s_mul_hi_u32 s55, s4, s22 // (op 2160) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2160) most significant: sum
s_add_u32 s53, s53, s55 // (op 2160) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 2160) 

s_mov_b32 s56, s20 // (op 2160) reg expression
s_mov_b32 s57, s21 // (op 2160) reg expression
s_mov_b32 s58, s54 // (op 2160) reg expression
s_mov_b32 s59, 131072 // (op 2160) call()

v_lshrrev_b32 v8, 6, v1 // (op 2260) 
s_mul_i32 s52, s16, s8 // (op 2260) 
s_add_u32 s52, s52, s0 // (op 2260) 
v_and_b32 v10, 63, v1 // (op 2260) 
s_mul_hi_u32 s53, s52, s13 // (op 2260) 
v_lshrrev_b32 v11, 4, v10 // (op 2260) 
s_sub_u32 s54, s52, s53 // (op 2260) 
v_lshlrev_b32 v12, 2, v11 // (op 2260) 

s_lshr_b32 s55, s54, 1 // (op 2260) 

v_lshl_add_u32 v11, v8, 4, v12 // (op 2260) 


s_add_u32 s54, s55, s53 // (op 2260) 


v_lshrrev_b32 v8, 2, v10 // (op 2260) 

s_lshr_b32 s53, s54, s11 // (op 2260) 

v_and_b32 v10, 3, v8 // (op 2260) 

s_and_b32 s54, s11, 2147483648 // (op 2260) 
v_and_b32 v8, 3, v1 // (op 2260) 
s_cmp_eq_i32 s54, 0 // (op 2260) 
s_mov_b32 s55, scc // (op 2260) 

v_lshl_add_u32 v12, v10, 2, v8 // (op 2260) 


s_cmp_eq_i32 s55, 0 // (op 2260) 
s_cselect_b32 s54, s52, s53 // (op 2260) 



s_mul_hi_u32 s55, s54, s12 // (op 2260) 
s_sub_u32 s52, s54, s55 // (op 2260) 
s_lshr_b32 s53, s52, 1 // (op 2260) 

s_add_u32 s52, s53, s55 // (op 2260) 


s_lshr_b32 s53, s52, s10 // (op 2260) 

s_and_b32 s52, s10, 2147483648 // (op 2260) 
s_cmp_eq_i32 s52, 0 // (op 2260) 
s_mov_b32 s55, scc // (op 2260) 

s_cmp_eq_i32 s55, 0 // (op 2260) 
s_cselect_b32 s52, s54, s53 // (op 2260) 


s_mul_i32 s53, s52, s14 // (op 2260) 
v_lshl_add_u32 v8, s52, 6, v11 // (op 2260) 


s_sub_u32 s52, s54, s53 // (op 2260) 


v_lshl_add_u32 v10, s52, 4, v12 // (op 2260) 


v_mul_lo_u32 v11, v10, s24 // (op 2260) 

v_add_lshl_u32 v10, v8, v11, 2 // (op 2260) 


s_mul_i32 s53, s24, s23 // (op 2262) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 2262) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 2262) least significant
s_mul_hi_u32 s55, s24, s22 // (op 2262) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2262) most significant: sum
s_add_u32 s53, s53, s55 // (op 2262) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 2262) 

s_mov_b32 s60, s26 // (op 2262) reg expression
s_mov_b32 s61, s27 // (op 2262) reg expression
s_mov_b32 s62, s54 // (op 2262) reg expression
s_mov_b32 s63, 131072 // (op 2262) call()

v_lshrrev_b32 v8, 4, v1 // (op 2360) 
v_lshlrev_b32 v11, 2, v8 // (op 2360) 

v_lshl_add_u32 v8, s16, 6, v11 // (op 2360) 

v_and_b32 v11, -16, v1 // (op 2360) 
v_sub_u32 v12, v1, v11 // (op 2360) 

v_lshl_add_u32 v11, v8, 4, v12 // (op 2360) 


v_lshlrev_b32 v8, 2, v11 // (op 2360) 

s_lshl_b32 s52, s15, 12 // (op 2362) 
s_mov_b32 s64, s28 // (op 2362) reg expression
s_mov_b32 s65, s29 // (op 2362) reg expression
s_mov_b32 s66, s52 // (op 2362) reg expression
s_mov_b32 s67, 131072 // (op 2362) call()

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
v_accvgpr_write a4, 0.00000 // (op 843) call()
v_accvgpr_write a5, 0.00000 // (op 843) call()
v_accvgpr_write a6, 0.00000 // (op 843) call()
v_accvgpr_write a7, 0.00000 // (op 843) call()
v_lshrrev_b32 v11, 6, v1 // (op 1725) 
s_mul_i32 s52, s16, s8 // (op 1725) 
s_add_u32 s52, s52, s0 // (op 1725) 
v_and_b32 v12, 15, v1 // (op 1725) 
s_mul_hi_u32 s53, s52, s13 // (op 1725) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 1725) 


s_sub_u32 s54, s52, s53 // (op 1725) 
v_and_b32 v11, 63, v1 // (op 1725) 
s_lshr_b32 s55, s54, 1 // (op 1725) 

v_lshrrev_b32 v12, 4, v11 // (op 1725) 

s_add_u32 s54, s55, s53 // (op 1725) 


v_lshlrev_b32 v11, 5, v12 // (op 1725) 

s_lshr_b32 s53, s54, s11 // (op 1725) 

s_and_b32 s54, s11, 2147483648 // (op 1725) 
s_cmp_eq_i32 s54, 0 // (op 1725) 
s_mov_b32 s55, scc // (op 1725) 

s_cmp_eq_i32 s55, 0 // (op 1725) 
s_cselect_b32 s54, s52, s53 // (op 1725) 


s_mul_hi_u32 s53, s54, s12 // (op 1725) 
s_sub_u32 s55, s54, s53 // (op 1725) 
s_lshr_b32 s68, s55, 1 // (op 1725) 

s_add_u32 s55, s68, s53 // (op 1725) 


s_lshr_b32 s53, s55, s10 // (op 1725) 

s_and_b32 s55, s10, 2147483648 // (op 1725) 
s_cmp_eq_i32 s55, 0 // (op 1725) 
s_mov_b32 s68, scc // (op 1725) 

s_cmp_eq_i32 s68, 0 // (op 1725) 
s_cselect_b32 s55, s54, s53 // (op 1725) 


v_lshl_add_u32 v12, s55, 6, v13 // (op 1725) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 1725) 
s_mul_i32 s53, s54, s68 // (op 1725) 


s_sub_u32 s54, s52, s53 // (op 1725) 


v_lshl_add_u32 v13, s54, 7, v11 // (op 1725) 


v_mov_b32 v15, 0 // (op 1725) convert
v_mov_b32 v14, v13 // (op 1725) convert

v_mul_lo_u32 v17, v12, s31 // (op 1725) most significant: low of low * high
v_mul_hi_u32 v11, v12, s30 // (op 1725) most significant: high of low * low
v_mul_lo_u32 v16, v12, s30 // (op 1725) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 1725) most significant: sum

v_add_co_u32 v16, s[52:53], v16, v14 // (op 1725) least significant half
v_addc_co_u32 v17, s[52:53], v17, v15, s[52:53] // (op 1725) most significant half



v_lshlrev_b64 v[14:15], 2, v[16:17] // (op 1725) 

v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 1725) 

v_mov_b32 v11, v12 // (op 1725) convert

s_mul_i32 s53, s30, s35 // (op 1727) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 1727) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 1727) least significant
s_mul_hi_u32 s55, s30, s34 // (op 1727) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1727) most significant: sum
s_add_u32 s53, s53, s55 // (op 1727) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1727) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1727) 

s_mov_b32 s68, s32 // (op 1727) reg expression
s_mov_b32 s69, s33 // (op 1727) reg expression
s_mov_b32 s70, s52 // (op 1727) reg expression
s_mov_b32 s71, 131072 // (op 1727) call()

v_lshrrev_b32 v12, 6, v1 // (op 1746) 
s_mul_i32 s52, s16, s8 // (op 1746) 
s_add_u32 s52, s52, s0 // (op 1746) 
v_and_b32 v13, 15, v1 // (op 1746) 
s_mul_hi_u32 s53, s52, s13 // (op 1746) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 1746) 


s_sub_u32 s54, s52, s53 // (op 1746) 
v_and_b32 v12, 63, v1 // (op 1746) 
s_lshr_b32 s55, s54, 1 // (op 1746) 

v_lshrrev_b32 v13, 4, v12 // (op 1746) 

s_add_u32 s54, s55, s53 // (op 1746) 


v_lshlrev_b32 v12, 5, v13 // (op 1746) 

s_lshr_b32 s53, s54, s11 // (op 1746) 

s_and_b32 s54, s11, 2147483648 // (op 1746) 
s_cmp_eq_i32 s54, 0 // (op 1746) 
s_mov_b32 s55, scc // (op 1746) 

s_cmp_eq_i32 s55, 0 // (op 1746) 
s_cselect_b32 s54, s52, s53 // (op 1746) 


s_mul_hi_u32 s53, s54, s12 // (op 1746) 
s_sub_u32 s55, s54, s53 // (op 1746) 
s_lshr_b32 s72, s55, 1 // (op 1746) 

s_add_u32 s55, s72, s53 // (op 1746) 


s_lshr_b32 s53, s55, s10 // (op 1746) 

s_and_b32 s55, s10, 2147483648 // (op 1746) 
s_cmp_eq_i32 s55, 0 // (op 1746) 
s_mov_b32 s72, scc // (op 1746) 

s_cmp_eq_i32 s72, 0 // (op 1746) 
s_cselect_b32 s55, s54, s53 // (op 1746) 


v_lshl_add_u32 v13, s55, 6, v14 // (op 1746) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 1746) 
s_mul_i32 s53, s54, s72 // (op 1746) 


s_sub_u32 s54, s52, s53 // (op 1746) 


v_lshl_add_u32 v14, s54, 7, v12 // (op 1746) 


v_mov_b32 v17, 0 // (op 1746) convert
v_mov_b32 v16, v14 // (op 1746) convert

v_mul_lo_u32 v15, v13, s31 // (op 1746) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 1746) most significant: high of low * low
v_mul_lo_u32 v14, v13, s30 // (op 1746) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 1746) most significant: sum

v_add_co_u32 v14, s[52:53], v14, v16 // (op 1746) least significant half
v_addc_co_u32 v15, s[52:53], v15, v17, s[52:53] // (op 1746) most significant half



v_lshlrev_b64 v[12:13], 2, v[14:15] // (op 1746) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1746) 

v_mov_b32 v12, v14 // (op 1746) convert

s_mul_i32 s53, s30, s35 // (op 1748) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 1748) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 1748) least significant
s_mul_hi_u32 s55, s30, s34 // (op 1748) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1748) most significant: sum
s_add_u32 s53, s53, s55 // (op 1748) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1748) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1748) 

s_mov_b32 s68, s32 // (op 1748) reg expression
s_mov_b32 s69, s33 // (op 1748) reg expression
s_mov_b32 s70, s52 // (op 1748) reg expression
s_mov_b32 s71, 131072 // (op 1748) call()

v_and_b32 v13, 15, v1 // (op 1858) 
s_mul_i32 s52, s16, s8 // (op 1858) 
s_add_u32 s52, s52, s0 // (op 1858) 
v_and_b32 v14, 63, v1 // (op 1858) 
s_mul_hi_u32 s53, s52, s13 // (op 1858) 
v_lshrrev_b32 v15, 4, v14 // (op 1858) 

s_sub_u32 s54, s52, s53 // (op 1858) 
v_lshlrev_b32 v14, 5, v15 // (op 1858) 

s_lshr_b32 s55, s54, 1 // (op 1858) 

s_add_u32 s54, s55, s53 // (op 1858) 


s_lshr_b32 s53, s54, s11 // (op 1858) 

s_and_b32 s54, s11, 2147483648 // (op 1858) 
s_cmp_eq_i32 s54, 0 // (op 1858) 
s_mov_b32 s55, scc // (op 1858) 

s_cmp_eq_i32 s55, 0 // (op 1858) 
s_cselect_b32 s54, s52, s53 // (op 1858) 


s_mul_hi_u32 s53, s54, s12 // (op 1858) 
s_sub_u32 s55, s54, s53 // (op 1858) 
s_lshr_b32 s72, s55, 1 // (op 1858) 

s_add_u32 s55, s72, s53 // (op 1858) 


s_lshr_b32 s53, s55, s10 // (op 1858) 

s_and_b32 s55, s10, 2147483648 // (op 1858) 
s_cmp_eq_i32 s55, 0 // (op 1858) 
s_mov_b32 s72, scc // (op 1858) 

s_cmp_eq_i32 s72, 0 // (op 1858) 
s_cselect_b32 s55, s54, s53 // (op 1858) 


s_mul_i32 s53, s55, s14 // (op 1858) 

s_sub_u32 s55, s54, s53 // (op 1858) 

v_lshl_add_u32 v15, s55, 4, v13 // (op 1858) 


v_mul_lo_u32 v17, v15, s37 // (op 1858) most significant: low of low * high
v_mul_hi_u32 v13, v15, s36 // (op 1858) most significant: high of low * low
v_mul_lo_u32 v16, v15, s36 // (op 1858) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 1858) most significant: sum


s_ashr_i64 s[72:73], s[2:3], 7 // (op 1858) 
s_mul_i32 s53, s54, s72 // (op 1858) 


s_sub_u32 s54, s52, s53 // (op 1858) 


v_lshl_add_u32 v13, s54, 7, v14 // (op 1858) 


v_mov_b32 v15, 0 // (op 1858) convert
v_mov_b32 v14, v13 // (op 1858) convert

v_add_co_u32 v18, s[52:53], v14, v16 // (op 1858) least significant half
v_addc_co_u32 v19, s[52:53], v15, v17, s[52:53] // (op 1858) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 1858) 


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 1858) 

v_mov_b32 v13, v16 // (op 1858) convert

s_mul_i32 s53, s36, s23 // (op 1860) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 1860) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 1860) least significant
s_mul_hi_u32 s55, s36, s22 // (op 1860) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1860) most significant: sum
s_add_u32 s53, s53, s55 // (op 1860) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1860) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1860) 

s_mov_b32 s72, s38 // (op 1860) reg expression
s_mov_b32 s73, s39 // (op 1860) reg expression
s_mov_b32 s74, s52 // (op 1860) reg expression
s_mov_b32 s75, 131072 // (op 1860) call()

v_and_b32 v14, 15, v1 // (op 1877) 
s_mul_i32 s52, s16, s8 // (op 1877) 
s_add_u32 s52, s52, s0 // (op 1877) 
v_and_b32 v15, 63, v1 // (op 1877) 
s_mul_hi_u32 s53, s52, s13 // (op 1877) 
v_lshrrev_b32 v16, 4, v15 // (op 1877) 

s_sub_u32 s54, s52, s53 // (op 1877) 
v_lshlrev_b32 v15, 5, v16 // (op 1877) 

s_lshr_b32 s55, s54, 1 // (op 1877) 

s_add_u32 s54, s55, s53 // (op 1877) 


s_lshr_b32 s53, s54, s11 // (op 1877) 

s_and_b32 s54, s11, 2147483648 // (op 1877) 
s_cmp_eq_i32 s54, 0 // (op 1877) 
s_mov_b32 s55, scc // (op 1877) 

s_cmp_eq_i32 s55, 0 // (op 1877) 
s_cselect_b32 s54, s52, s53 // (op 1877) 


s_mul_hi_u32 s53, s54, s12 // (op 1877) 
s_sub_u32 s55, s54, s53 // (op 1877) 
s_lshr_b32 s76, s55, 1 // (op 1877) 

s_add_u32 s55, s76, s53 // (op 1877) 


s_lshr_b32 s53, s55, s10 // (op 1877) 

s_and_b32 s55, s10, 2147483648 // (op 1877) 
s_cmp_eq_i32 s55, 0 // (op 1877) 
s_mov_b32 s76, scc // (op 1877) 

s_cmp_eq_i32 s76, 0 // (op 1877) 
s_cselect_b32 s55, s54, s53 // (op 1877) 


s_mul_i32 s53, s55, s14 // (op 1877) 

s_sub_u32 s55, s54, s53 // (op 1877) 

v_lshl_add_u32 v16, s55, 4, v14 // (op 1877) 


v_mul_lo_u32 v19, v16, s37 // (op 1877) most significant: low of low * high
v_mul_hi_u32 v14, v16, s36 // (op 1877) most significant: high of low * low
v_mul_lo_u32 v18, v16, s36 // (op 1877) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 1877) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1877) 
s_mul_i32 s53, s54, s76 // (op 1877) 


s_sub_u32 s54, s52, s53 // (op 1877) 


v_lshl_add_u32 v14, s54, 7, v15 // (op 1877) 


v_mov_b32 v17, 0 // (op 1877) convert
v_mov_b32 v16, v14 // (op 1877) convert

v_add_co_u32 v14, s[52:53], v16, v18 // (op 1877) least significant half
v_addc_co_u32 v15, s[52:53], v17, v19, s[52:53] // (op 1877) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 1877) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 1877) 

v_mov_b32 v14, v16 // (op 1877) convert

s_mul_i32 s53, s36, s23 // (op 1879) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 1879) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 1879) least significant
s_mul_hi_u32 s55, s36, s22 // (op 1879) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1879) most significant: sum
s_add_u32 s53, s53, s55 // (op 1879) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1879) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1879) 

s_mov_b32 s72, s38 // (op 1879) reg expression
s_mov_b32 s73, s39 // (op 1879) reg expression
s_mov_b32 s74, s52 // (op 1879) reg expression
s_mov_b32 s75, 131072 // (op 1879) call()

v_lshrrev_b32 v15, 2, v1 // (op 1986) 
s_mul_i32 s52, s16, s8 // (op 1986) 
s_add_u32 s52, s52, s0 // (op 1986) 
v_lshlrev_b32 v16, 1, v15 // (op 1986) 

s_mul_hi_u32 s53, s52, s13 // (op 1986) 
v_and_b32 v15, -4, v1 // (op 1986) 
s_sub_u32 s54, s52, s53 // (op 1986) 
v_sub_u32 v17, v1, v15 // (op 1986) 

s_lshr_b32 s55, s54, 1 // (op 1986) 

s_add_u32 s54, s55, s53 // (op 1986) 


s_lshr_b32 s53, s54, s11 // (op 1986) 

s_and_b32 s54, s11, 2147483648 // (op 1986) 
s_cmp_eq_i32 s54, 0 // (op 1986) 
s_mov_b32 s55, scc // (op 1986) 

s_cmp_eq_i32 s55, 0 // (op 1986) 
s_cselect_b32 s54, s52, s53 // (op 1986) 


s_mul_hi_u32 s53, s54, s12 // (op 1986) 
s_sub_u32 s55, s54, s53 // (op 1986) 
s_lshr_b32 s76, s55, 1 // (op 1986) 

s_add_u32 s55, s76, s53 // (op 1986) 


s_lshr_b32 s53, s55, s10 // (op 1986) 

s_and_b32 s55, s10, 2147483648 // (op 1986) 
s_cmp_eq_i32 s55, 0 // (op 1986) 
s_mov_b32 s76, scc // (op 1986) 

s_cmp_eq_i32 s76, 0 // (op 1986) 
s_cselect_b32 s55, s54, s53 // (op 1986) 


v_lshl_add_u32 v15, s55, 6, v16 // (op 1986) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1986) 
s_mul_i32 s53, s54, s76 // (op 1986) 


s_sub_u32 s54, s52, s53 // (op 1986) 


v_lshl_add_u32 v16, s54, 2, v17 // (op 1986) 


v_mul_lo_u32 v17, v15, s40 // (op 1986) 
v_add_u32 v17, v17, v16 // (op 1986) 


s_mul_i32 s53, s40, s35 // (op 1988) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 1988) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 1988) least significant
s_mul_hi_u32 s55, s40, s34 // (op 1988) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1988) most significant: sum
s_add_u32 s53, s53, s55 // (op 1988) most significant: sum


s_mov_b32 s76, s42 // (op 1988) reg expression
s_mov_b32 s77, s43 // (op 1988) reg expression
s_mov_b32 s78, s52 // (op 1988) reg expression
s_mov_b32 s79, 131072 // (op 1988) call()

v_lshrrev_b32 v15, 2, v1 // (op 2005) 
s_mul_i32 s52, s16, s8 // (op 2005) 
s_add_u32 s52, s52, s0 // (op 2005) 
v_lshlrev_b32 v16, 1, v15 // (op 2005) 

s_mul_hi_u32 s53, s52, s13 // (op 2005) 
v_and_b32 v15, -4, v1 // (op 2005) 
s_sub_u32 s54, s52, s53 // (op 2005) 
v_sub_u32 v18, v1, v15 // (op 2005) 

s_lshr_b32 s55, s54, 1 // (op 2005) 

s_add_u32 s54, s55, s53 // (op 2005) 


s_lshr_b32 s53, s54, s11 // (op 2005) 

s_and_b32 s54, s11, 2147483648 // (op 2005) 
s_cmp_eq_i32 s54, 0 // (op 2005) 
s_mov_b32 s55, scc // (op 2005) 

s_cmp_eq_i32 s55, 0 // (op 2005) 
s_cselect_b32 s54, s52, s53 // (op 2005) 


s_mul_hi_u32 s53, s54, s12 // (op 2005) 
s_sub_u32 s55, s54, s53 // (op 2005) 
s_lshr_b32 s80, s55, 1 // (op 2005) 

s_add_u32 s55, s80, s53 // (op 2005) 


s_lshr_b32 s53, s55, s10 // (op 2005) 

s_and_b32 s55, s10, 2147483648 // (op 2005) 
s_cmp_eq_i32 s55, 0 // (op 2005) 
s_mov_b32 s80, scc // (op 2005) 

s_cmp_eq_i32 s80, 0 // (op 2005) 
s_cselect_b32 s55, s54, s53 // (op 2005) 


v_lshl_add_u32 v15, s55, 6, v16 // (op 2005) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 2005) 
s_mul_i32 s53, s54, s80 // (op 2005) 


s_sub_u32 s54, s52, s53 // (op 2005) 


v_lshl_add_u32 v16, s54, 2, v18 // (op 2005) 


v_mul_lo_u32 v18, v15, s40 // (op 2005) 
v_add_u32 v18, v18, v16 // (op 2005) 


s_mul_i32 s53, s40, s35 // (op 2007) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 2007) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 2007) least significant
s_mul_hi_u32 s55, s40, s34 // (op 2007) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2007) most significant: sum
s_add_u32 s53, s53, s55 // (op 2007) most significant: sum


s_mov_b32 s76, s42 // (op 2007) reg expression
s_mov_b32 s77, s43 // (op 2007) reg expression
s_mov_b32 s78, s52 // (op 2007) reg expression
s_mov_b32 s79, 131072 // (op 2007) call()

v_and_b32 v15, 15, v1 // (op 2114) 
s_mul_i32 s52, s16, s8 // (op 2114) 
s_add_u32 s52, s52, s0 // (op 2114) 
v_and_b32 v16, 63, v1 // (op 2114) 
s_mul_hi_u32 s53, s52, s13 // (op 2114) 
v_lshrrev_b32 v19, 4, v16 // (op 2114) 

s_sub_u32 s54, s52, s53 // (op 2114) 
s_lshr_b32 s55, s54, 1 // (op 2114) 

s_add_u32 s54, s55, s53 // (op 2114) 


s_lshr_b32 s53, s54, s11 // (op 2114) 

s_and_b32 s54, s11, 2147483648 // (op 2114) 
s_cmp_eq_i32 s54, 0 // (op 2114) 
s_mov_b32 s55, scc // (op 2114) 

s_cmp_eq_i32 s55, 0 // (op 2114) 
s_cselect_b32 s54, s52, s53 // (op 2114) 


s_mul_hi_u32 s53, s54, s12 // (op 2114) 
s_sub_u32 s55, s54, s53 // (op 2114) 
s_lshr_b32 s80, s55, 1 // (op 2114) 

s_add_u32 s55, s80, s53 // (op 2114) 


s_lshr_b32 s53, s55, s10 // (op 2114) 

s_and_b32 s55, s10, 2147483648 // (op 2114) 
s_cmp_eq_i32 s55, 0 // (op 2114) 
s_mov_b32 s80, scc // (op 2114) 

s_cmp_eq_i32 s80, 0 // (op 2114) 
s_cselect_b32 s55, s54, s53 // (op 2114) 


s_mul_i32 s53, s55, s14 // (op 2114) 

s_sub_u32 s55, s54, s53 // (op 2114) 

v_lshl_add_u32 v16, s55, 4, v15 // (op 2114) 


v_mul_lo_u32 v15, v16, s44 // (op 2114) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 2114) 
s_mul_i32 s53, s54, s80 // (op 2114) 


s_sub_u32 s54, s52, s53 // (op 2114) 


v_lshl_add_u32 v16, s54, 2, v19 // (op 2114) 


v_add_u32 v19, v16, v15 // (op 2114) 


s_mul_i32 s53, s44, s23 // (op 2116) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 2116) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 2116) least significant
s_mul_hi_u32 s55, s44, s22 // (op 2116) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2116) most significant: sum
s_add_u32 s53, s53, s55 // (op 2116) most significant: sum


s_mov_b32 s80, s46 // (op 2116) reg expression
s_mov_b32 s81, s47 // (op 2116) reg expression
s_mov_b32 s82, s52 // (op 2116) reg expression
s_mov_b32 s83, 131072 // (op 2116) call()

v_and_b32 v15, 15, v1 // (op 2133) 
s_mul_i32 s52, s16, s8 // (op 2133) 
s_add_u32 s52, s52, s0 // (op 2133) 
v_and_b32 v16, 63, v1 // (op 2133) 
s_mul_hi_u32 s53, s52, s13 // (op 2133) 
v_lshrrev_b32 v20, 4, v16 // (op 2133) 

s_sub_u32 s54, s52, s53 // (op 2133) 
s_lshr_b32 s55, s54, 1 // (op 2133) 

s_add_u32 s54, s55, s53 // (op 2133) 


s_lshr_b32 s53, s54, s11 // (op 2133) 

s_and_b32 s54, s11, 2147483648 // (op 2133) 
s_cmp_eq_i32 s54, 0 // (op 2133) 
s_mov_b32 s55, scc // (op 2133) 

s_cmp_eq_i32 s55, 0 // (op 2133) 
s_cselect_b32 s54, s52, s53 // (op 2133) 


s_mul_hi_u32 s53, s54, s12 // (op 2133) 
s_sub_u32 s55, s54, s53 // (op 2133) 
s_lshr_b32 s84, s55, 1 // (op 2133) 

s_add_u32 s55, s84, s53 // (op 2133) 


s_lshr_b32 s53, s55, s10 // (op 2133) 

s_and_b32 s55, s10, 2147483648 // (op 2133) 
s_cmp_eq_i32 s55, 0 // (op 2133) 
s_mov_b32 s84, scc // (op 2133) 

s_cmp_eq_i32 s84, 0 // (op 2133) 
s_cselect_b32 s55, s54, s53 // (op 2133) 


s_mul_i32 s53, s55, s14 // (op 2133) 

s_sub_u32 s55, s54, s53 // (op 2133) 

v_lshl_add_u32 v16, s55, 4, v15 // (op 2133) 


v_mul_lo_u32 v15, v16, s44 // (op 2133) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 2133) 
s_mul_i32 s53, s54, s84 // (op 2133) 


s_sub_u32 s54, s52, s53 // (op 2133) 


v_lshl_add_u32 v16, s54, 2, v20 // (op 2133) 


v_add_u32 v20, v16, v15 // (op 2133) 


s_mul_i32 s53, s44, s23 // (op 2135) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 2135) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 2135) least significant
s_mul_hi_u32 s55, s44, s22 // (op 2135) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2135) most significant: sum
s_add_u32 s53, s53, s55 // (op 2135) most significant: sum


s_mov_b32 s80, s46 // (op 2135) reg expression
s_mov_b32 s81, s47 // (op 2135) reg expression
s_mov_b32 s82, s52 // (op 2135) reg expression
s_mov_b32 s83, 131072 // (op 2135) call()



v_mov_b32 v15, v11 // (op 1492) reg expression

buffer_load_dwordx4 v[22:25], v15, s[68:71], 0 offen // (op 1492) Load value




buffer_load_dwordx4 v[26:29], v13, s[72:75], 0 offen // (op 1498) Load value



buffer_load_ubyte v21, v19, s[80:83], 0 offen offset: 0 // (op 1503) Load value


s_lshl_b64 s[52:53], s[30:31], 5 // (op 1509) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1509) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1509) 

v_add_u32 v15, v11, s52 // (op 1509) 


buffer_load_dwordx4 v[30:33], v15, s[68:71], 0 offen // (op 1509) Load value


v_mov_b64 v[34:35], s[40:41] // (op 1514) convert

v_mov_b32 v15, v17 // (op 1514) 
buffer_load_ubyte v36, v15, s[76:79], 0 offen offset: 0 // (op 1514) Load value
v_add_u32 v15, v15, v34 // (op 1514) 
buffer_load_ubyte v37, v15, s[76:79], 0 offen offset: 0 // (op 1514) Load value





buffer_load_ubyte v15, v19, s[80:83], 0 offen offset:4 // (op 1518) Load value



v_add_u32 v16, 64, v11 // (op 1524) 

buffer_load_dwordx4 v[38:41], v16, s[68:71], 0 offen // (op 1524) Load value




buffer_load_dwordx4 v[42:45], v13, s[72:75], 0 offen offset:64 // (op 1530) Load value



s_lshl_b64 s[52:53], s[30:31], 5 // (op 1536) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1536) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1536) 

s_add_u32 s54, s52, 64 // (op 1536) 

v_add_u32 v13, v11, s54 // (op 1536) 


buffer_load_dwordx4 v[46:49], v13, s[68:71], 0 offen // (op 1536) Load value



v_mov_b64 v[34:35], s[40:41] // (op 1541) convert

v_mov_b32 v11, v17 // (op 1541) 
buffer_load_ubyte v50, v11, s[76:79], 0 offen offset:4 // (op 1541) Load value
v_add_u32 v11, v11, v34 // (op 1541) 
buffer_load_ubyte v51, v11, s[76:79], 0 offen offset:4 // (op 1541) Load value



v_and_b32 v11, -4, v1 // (op 2226) 
v_sub_u32 v13, v1, v11 // (op 2226) 

v_lshrrev_b32 v11, 2, v1 // (op 2226) 
v_lshl_add_u32 v19, v11, 3, v13 // (op 2226) 


s_barrier  // (op 2514) 



s_waitcnt vmcnt(7)
ds_write_b8 v19, v36 // (op 1543) Store local data 
s_waitcnt vmcnt(6)
ds_write_b8 v19, v37 offset:4 // (op 1543) Store local data 

v_and_b32 v11, -4, v1 // (op 2197) 
v_sub_u32 v13, v1, v11 // (op 2197) 

v_lshrrev_b32 v11, 2, v1 // (op 2197) 
v_lshl_add_u32 v19, v11, 3, v13 // (op 2197) 


v_and_b32 v11, -4, v1 // (op 2426) 
v_sub_u32 v13, v1, v11 // (op 2426) 

v_lshrrev_b32 v11, 2, v1 // (op 2426) 
v_lshl_add_u32 v16, v11, 3, v13 // (op 2426) 


s_mov_b32 s1, 0 // (op 845) call()
s_and_b32 s52, s7, 4294967294 // (op 844) 
s_cmp_lt_u32 s1, s52 // (op 844) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_844 // (op 844) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_844 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoop_844:
s_add_u32 s1, s1, 2 // (op 846) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2511) 



ds_read_u8 v11, v0 // (op 63) Load local data 



ds_read_u8 v13, v0 offset:128 // (op 870) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[26:29], a[0:3], v11, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 70) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:33], v[26:29], a[4:7], v13, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 872) 



v_mov_b32 v11, v12 // (op 55) 
v_add_u32 v13, 128, v11 // (op 55) 

buffer_load_dwordx4 v[22:25], v13, s[68:71], 0 offen // (op 55) Load value



v_mov_b32 v13, v14 // (op 59) 

buffer_load_dwordx4 v[26:29], v13, s[72:75], 0 offen offset:128 // (op 59) Load value


v_mov_b32 v17, v20 // (op 68) 

buffer_load_ubyte v21, v17, s[80:83], 0 offen offset:8 // (op 68) Load value


s_lshl_b64 s[52:53], s[30:31], 5 // (op 895) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 895) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 895) 

s_add_u32 s54, s52, 128 // (op 895) 

v_add_u32 v34, v11, s54 // (op 895) 


buffer_load_dwordx4 v[30:33], v34, s[68:71], 0 offen // (op 895) Load value


v_mov_b64 v[34:35], s[40:41] // (op 9) convert

v_mov_b32 v52, v18 // (op 9) 
v_mov_b32 v53, v52 // (op 9) 
buffer_load_ubyte v36, v53, s[76:79], 0 offen offset:8 // (op 9) Load value
v_add_u32 v53, v53, v34 // (op 9) 
buffer_load_ubyte v37, v53, s[76:79], 0 offen offset:8 // (op 9) Load value


s_barrier  // (op 2552) 



s_waitcnt vmcnt(7)
ds_write_b8 v16, v50 offset:256 // (op 783) Store local data 
s_waitcnt vmcnt(6)
ds_write_b8 v16, v51 offset:260 // (op 783) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2549) 



ds_read_u8 v34, v5 offset:256 // (op 785) Load local data 



ds_read_u8 v35, v5 offset:384 // (op 879) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:41], v[42:45], a[0:3], v34, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 788) 

s_barrier  // (op 2697) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:49], v[42:45], a[4:7], v35, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 873) 




buffer_load_ubyte v15, v17, s[80:83], 0 offen offset:12 // (op 787) Load value



v_add_u32 v20, 8, v20 // (op 2125) 
v_add_u32 v17, 192, v11 // (op 790) 

buffer_load_dwordx4 v[38:41], v17, s[68:71], 0 offen // (op 790) Load value




buffer_load_dwordx4 v[42:45], v13, s[72:75], 0 offen offset:192 // (op 792) Load value



v_add_u32 v14, 128, v14 // (op 1869) 
s_lshl_b64 s[52:53], s[30:31], 5 // (op 888) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 888) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 888) 

s_add_u32 s54, s52, 192 // (op 888) 

v_add_u32 v13, v11, s54 // (op 888) 


buffer_load_dwordx4 v[46:49], v13, s[68:71], 0 offen // (op 888) Load value



v_add_u32 v12, 128, v12 // (op 1736) 
v_mov_b64 v[34:35], s[40:41] // (op 782) convert

v_mov_b32 v11, v52 // (op 782) 
buffer_load_ubyte v50, v11, s[76:79], 0 offen offset:12 // (op 782) Load value
v_add_u32 v11, v11, v34 // (op 782) 
buffer_load_ubyte v51, v11, s[76:79], 0 offen offset:12 // (op 782) Load value



s_barrier  // (op 2517) 
v_add_u32 v18, 8, v18 // (op 1997) 



s_waitcnt vmcnt(7)
ds_write_b8 v19, v36 // (op 13) Store local data 
s_waitcnt vmcnt(6)
ds_write_b8 v19, v37 offset:4 // (op 13) Store local data 
s_and_b32 s52, s7, 4294967294 // (op 844) 
s_cmp_lt_u32 s1, s52 // (op 844) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoop_844 // (op 844) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoop_844 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_844:











s_and_b32 s52, s7, 4294967294 // (op 1634) 
v_lshrrev_b32 v11, 6, v1 // (op 1634) 
s_add_u32 s53, s0, s52 // (op 1634) 

v_and_b32 v12, 15, v1 // (op 1634) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 1634) 


s_mul_i32 s52, s16, s8 // (op 1634) 
s_add_u32 s52, s52, s53 // (op 1634) 

v_and_b32 v14, 63, v1 // (op 1634) 
s_mul_hi_u32 s53, s52, s13 // (op 1634) 
v_lshrrev_b32 v11, 4, v14 // (op 1634) 

s_sub_u32 s54, s52, s53 // (op 1634) 
v_lshlrev_b32 v12, 5, v11 // (op 1634) 

s_lshr_b32 s55, s54, 1 // (op 1634) 

s_add_u32 s54, s55, s53 // (op 1634) 


s_lshr_b32 s53, s54, s11 // (op 1634) 

s_and_b32 s54, s11, 2147483648 // (op 1634) 
s_cmp_eq_i32 s54, 0 // (op 1634) 
s_mov_b32 s55, scc // (op 1634) 

s_cmp_eq_i32 s55, 0 // (op 1634) 
s_cselect_b32 s54, s52, s53 // (op 1634) 


s_mul_hi_u32 s53, s54, s12 // (op 1634) 
s_sub_u32 s55, s54, s53 // (op 1634) 
s_lshr_b32 s84, s55, 1 // (op 1634) 

s_add_u32 s55, s84, s53 // (op 1634) 


s_lshr_b32 s53, s55, s10 // (op 1634) 

s_and_b32 s55, s10, 2147483648 // (op 1634) 
s_cmp_eq_i32 s55, 0 // (op 1634) 
s_mov_b32 s84, scc // (op 1634) 

s_cmp_eq_i32 s84, 0 // (op 1634) 
s_cselect_b32 s55, s54, s53 // (op 1634) 


v_lshl_add_u32 v11, s55, 6, v13 // (op 1634) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 1634) 
s_mul_i32 s53, s54, s84 // (op 1634) 


s_sub_u32 s54, s52, s53 // (op 1634) 


v_lshl_add_u32 v13, s54, 7, v12 // (op 1634) 


v_mov_b32 v17, 0 // (op 1634) convert
v_mov_b32 v16, v13 // (op 1634) convert

v_mul_lo_u32 v13, v11, s31 // (op 1634) most significant: low of low * high
v_mul_hi_u32 v14, v11, s30 // (op 1634) most significant: high of low * low
v_mul_lo_u32 v12, v11, s30 // (op 1634) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 1634) most significant: sum

v_add_co_u32 v12, s[52:53], v12, v16 // (op 1634) least significant half
v_addc_co_u32 v13, s[52:53], v13, v17, s[52:53] // (op 1634) most significant half



v_lshlrev_b64 v[16:17], 2, v[12:13] // (op 1634) 

v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 1634) 

v_mov_b32 v11, v18 // (op 1634) convert

s_mul_i32 s53, s30, s35 // (op 1636) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 1636) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 1636) least significant
s_mul_hi_u32 s55, s30, s34 // (op 1636) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1636) most significant: sum
s_add_u32 s53, s53, s55 // (op 1636) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1636) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1636) 

s_mov_b32 s68, s32 // (op 1636) reg expression
s_mov_b32 s69, s33 // (op 1636) reg expression
s_mov_b32 s70, s52 // (op 1636) reg expression
s_mov_b32 s71, 131072 // (op 1636) call()

s_and_b32 s52, s7, 4294967294 // (op 1793) 
v_and_b32 v12, 15, v1 // (op 1793) 
s_add_u32 s53, s0, s52 // (op 1793) 

v_and_b32 v13, 63, v1 // (op 1793) 
v_lshrrev_b32 v14, 4, v13 // (op 1793) 

s_mul_i32 s52, s16, s8 // (op 1793) 
s_add_u32 s52, s52, s53 // (op 1793) 

v_lshlrev_b32 v13, 5, v14 // (op 1793) 

s_mul_hi_u32 s53, s52, s13 // (op 1793) 
s_sub_u32 s54, s52, s53 // (op 1793) 
s_lshr_b32 s55, s54, 1 // (op 1793) 

s_add_u32 s54, s55, s53 // (op 1793) 


s_lshr_b32 s53, s54, s11 // (op 1793) 

s_and_b32 s54, s11, 2147483648 // (op 1793) 
s_cmp_eq_i32 s54, 0 // (op 1793) 
s_mov_b32 s55, scc // (op 1793) 

s_cmp_eq_i32 s55, 0 // (op 1793) 
s_cselect_b32 s54, s52, s53 // (op 1793) 


s_mul_hi_u32 s53, s54, s12 // (op 1793) 
s_sub_u32 s55, s54, s53 // (op 1793) 
s_lshr_b32 s84, s55, 1 // (op 1793) 

s_add_u32 s55, s84, s53 // (op 1793) 


s_lshr_b32 s53, s55, s10 // (op 1793) 

s_and_b32 s55, s10, 2147483648 // (op 1793) 
s_cmp_eq_i32 s55, 0 // (op 1793) 
s_mov_b32 s84, scc // (op 1793) 

s_cmp_eq_i32 s84, 0 // (op 1793) 
s_cselect_b32 s55, s54, s53 // (op 1793) 


s_mul_i32 s53, s55, s14 // (op 1793) 

s_sub_u32 s55, s54, s53 // (op 1793) 

v_lshl_add_u32 v14, s55, 4, v12 // (op 1793) 


v_mul_lo_u32 v17, v14, s37 // (op 1793) most significant: low of low * high
v_mul_hi_u32 v12, v14, s36 // (op 1793) most significant: high of low * low
v_mul_lo_u32 v16, v14, s36 // (op 1793) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 1793) most significant: sum


s_ashr_i64 s[84:85], s[2:3], 7 // (op 1793) 
s_mul_i32 s53, s54, s84 // (op 1793) 


s_sub_u32 s54, s52, s53 // (op 1793) 


v_lshl_add_u32 v12, s54, 7, v13 // (op 1793) 


v_mov_b32 v19, 0 // (op 1793) convert
v_mov_b32 v18, v12 // (op 1793) convert

v_add_co_u32 v12, s[52:53], v18, v16 // (op 1793) least significant half
v_addc_co_u32 v13, s[52:53], v19, v17, s[52:53] // (op 1793) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 1793) 


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 1793) 

v_mov_b32 v12, v16 // (op 1793) convert

s_mul_i32 s53, s36, s23 // (op 1795) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 1795) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 1795) least significant
s_mul_hi_u32 s55, s36, s22 // (op 1795) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1795) most significant: sum
s_add_u32 s53, s53, s55 // (op 1795) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1795) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1795) 

s_mov_b32 s72, s38 // (op 1795) reg expression
s_mov_b32 s73, s39 // (op 1795) reg expression
s_mov_b32 s74, s52 // (op 1795) reg expression
s_mov_b32 s75, 131072 // (op 1795) call()

s_and_b32 s52, s7, 4294967294 // (op 1921) 
v_lshrrev_b32 v13, 2, v1 // (op 1921) 
s_add_u32 s53, s0, s52 // (op 1921) 

v_lshlrev_b32 v14, 1, v13 // (op 1921) 

v_and_b32 v13, -4, v1 // (op 1921) 
s_mul_i32 s52, s16, s8 // (op 1921) 
s_add_u32 s52, s52, s53 // (op 1921) 

v_sub_u32 v16, v1, v13 // (op 1921) 

s_mul_hi_u32 s53, s52, s13 // (op 1921) 
s_sub_u32 s54, s52, s53 // (op 1921) 
s_lshr_b32 s55, s54, 1 // (op 1921) 

s_add_u32 s54, s55, s53 // (op 1921) 


s_lshr_b32 s53, s54, s11 // (op 1921) 

s_and_b32 s54, s11, 2147483648 // (op 1921) 
s_cmp_eq_i32 s54, 0 // (op 1921) 
s_mov_b32 s55, scc // (op 1921) 

s_cmp_eq_i32 s55, 0 // (op 1921) 
s_cselect_b32 s54, s52, s53 // (op 1921) 


s_mul_hi_u32 s53, s54, s12 // (op 1921) 
s_sub_u32 s55, s54, s53 // (op 1921) 
s_lshr_b32 s84, s55, 1 // (op 1921) 

s_add_u32 s55, s84, s53 // (op 1921) 


s_lshr_b32 s53, s55, s10 // (op 1921) 

s_and_b32 s55, s10, 2147483648 // (op 1921) 
s_cmp_eq_i32 s55, 0 // (op 1921) 
s_mov_b32 s84, scc // (op 1921) 

s_cmp_eq_i32 s84, 0 // (op 1921) 
s_cselect_b32 s55, s54, s53 // (op 1921) 


v_lshl_add_u32 v13, s55, 6, v14 // (op 1921) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 1921) 
s_mul_i32 s53, s54, s84 // (op 1921) 


s_sub_u32 s54, s52, s53 // (op 1921) 


v_lshl_add_u32 v14, s54, 2, v16 // (op 1921) 


v_mul_lo_u32 v16, v13, s40 // (op 1921) 
v_add_u32 v16, v16, v14 // (op 1921) 


s_mul_i32 s53, s40, s35 // (op 1923) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 1923) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 1923) least significant
s_mul_hi_u32 s55, s40, s34 // (op 1923) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1923) most significant: sum
s_add_u32 s53, s53, s55 // (op 1923) most significant: sum


s_mov_b32 s76, s42 // (op 1923) reg expression
s_mov_b32 s77, s43 // (op 1923) reg expression
s_mov_b32 s78, s52 // (op 1923) reg expression
s_mov_b32 s79, 131072 // (op 1923) call()

s_and_b32 s52, s7, 4294967294 // (op 2049) 
v_and_b32 v13, 15, v1 // (op 2049) 
s_add_u32 s53, s0, s52 // (op 2049) 

v_and_b32 v14, 63, v1 // (op 2049) 
v_lshrrev_b32 v17, 4, v14 // (op 2049) 

s_mul_i32 s52, s16, s8 // (op 2049) 
s_add_u32 s52, s52, s53 // (op 2049) 

s_mul_hi_u32 s53, s52, s13 // (op 2049) 
s_sub_u32 s54, s52, s53 // (op 2049) 
s_lshr_b32 s55, s54, 1 // (op 2049) 

s_add_u32 s54, s55, s53 // (op 2049) 


s_lshr_b32 s53, s54, s11 // (op 2049) 

s_and_b32 s54, s11, 2147483648 // (op 2049) 
s_cmp_eq_i32 s54, 0 // (op 2049) 
s_mov_b32 s55, scc // (op 2049) 

s_cmp_eq_i32 s55, 0 // (op 2049) 
s_cselect_b32 s54, s52, s53 // (op 2049) 


s_mul_hi_u32 s53, s54, s12 // (op 2049) 
s_sub_u32 s55, s54, s53 // (op 2049) 
s_lshr_b32 s84, s55, 1 // (op 2049) 

s_add_u32 s55, s84, s53 // (op 2049) 


s_lshr_b32 s53, s55, s10 // (op 2049) 

s_and_b32 s55, s10, 2147483648 // (op 2049) 
s_cmp_eq_i32 s55, 0 // (op 2049) 
s_mov_b32 s84, scc // (op 2049) 

s_cmp_eq_i32 s84, 0 // (op 2049) 
s_cselect_b32 s55, s54, s53 // (op 2049) 


s_mul_i32 s53, s55, s14 // (op 2049) 

s_sub_u32 s55, s54, s53 // (op 2049) 

v_lshl_add_u32 v14, s55, 4, v13 // (op 2049) 


v_mul_lo_u32 v13, v14, s44 // (op 2049) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 2049) 
s_mul_i32 s53, s54, s84 // (op 2049) 


s_sub_u32 s54, s52, s53 // (op 2049) 


v_lshl_add_u32 v14, s54, 2, v17 // (op 2049) 


v_add_u32 v17, v14, v13 // (op 2049) 


s_mul_i32 s53, s44, s23 // (op 2051) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 2051) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 2051) least significant
s_mul_hi_u32 s55, s44, s22 // (op 2051) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2051) most significant: sum
s_add_u32 s53, s53, s55 // (op 2051) most significant: sum


s_mov_b32 s80, s46 // (op 2051) reg expression
s_mov_b32 s81, s47 // (op 2051) reg expression
s_mov_b32 s82, s52 // (op 2051) reg expression
s_mov_b32 s83, 131072 // (op 2051) call()

v_and_b32 v13, -4, v1 // (op 2210) 
v_sub_u32 v14, v1, v13 // (op 2210) 

v_lshrrev_b32 v13, 2, v1 // (op 2210) 
v_lshl_add_u32 v18, v13, 3, v14 // (op 2210) 


s_and_b32 s52, s7, 4294967294 // (op 850) 
s_cmp_lt_u32 s52, s7 // (op 849) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_849 // (op 849) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_849 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_849:

s_add_u32 s52, s52, 1 // (op 851) 
v_mov_b64 v[14:15], s[40:41] // (op 740) convert

v_mov_b32 v13, v16 // (op 740) 
v_mov_b32 v19, v13 // (op 740) 
buffer_load_ubyte v20, v19, s[76:79], 0 offen offset: 0 // (op 740) Load value
v_add_u32 v19, v19, v14 // (op 740) 
buffer_load_ubyte v21, v19, s[76:79], 0 offen offset: 0 // (op 740) Load value



v_add_u32 v16, 4, v16 // (op 1913) 


s_barrier  // (op 2520) 
v_mov_b32 v19, v11 // (op 748) 



v_mov_b32 v13, v19 // (op 748) reg expression

buffer_load_dwordx4 v[22:25], v13, s[68:71], 0 offen // (op 748) Load value

s_waitcnt vmcnt(2)
ds_write_b8 v18, v20 // (op 741) Store local data 


v_mov_b32 v13, v12 // (op 750) 

buffer_load_dwordx4 v[26:29], v13, s[72:75], 0 offen // (op 750) Load value

v_add_u32 v12, 64, v12 // (op 1785) 
s_waitcnt vmcnt(2)
ds_write_b8 v18, v21 offset:4 // (op 741) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 2523) 



ds_read_u8 v13, v0 // (op 743) Load local data 


v_mov_b32 v14, v17 // (op 745) 

buffer_load_ubyte v15, v14, s[80:83], 0 offen offset: 0 // (op 745) Load value

v_add_u32 v17, 4, v17 // (op 2041) 


s_lshl_b64 s[54:55], s[30:31], 5 // (op 865) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 865) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 865) 

v_add_u32 v14, v19, s54 // (op 865) 


buffer_load_dwordx4 v[30:33], v14, s[68:71], 0 offen // (op 865) Load value

s_waitcnt vmcnt(1) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[26:29], a[0:3], v13, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 746) 






v_add_u32 v11, 64, v11 // (op 1622) 
ds_read_u8 v13, v0 offset:128 // (op 855) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:33], v[26:29], a[4:7], v13, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 857) 




s_cmp_lt_u32 s52, s7 // (op 849) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_849 // (op 849) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_849 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_849:






s_barrier  // (op 2683) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 243) 
s_cmp_lg_u32 s18, 0 // (op 269) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Send_Tile_269 // (op 269) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Send_Tile_269



s_nop 1
buffer_store_dword a4, v8, s[64:67], 0 offen offset:2048 sc1 // (op 1006) Store value
buffer_store_dword a5, v8, s[64:67], 0 offen offset:2112 sc1 // (op 1006) Store value
buffer_store_dword a6, v8, s[64:67], 0 offen offset:2176 sc1 // (op 1006) Store value
buffer_store_dword a7, v8, s[64:67], 0 offen offset:2240 sc1 // (op 1006) Store value



buffer_store_dword a0, v8, s[64:67], 0 offen sc1 // (op 266) Store value
buffer_store_dword a1, v8, s[64:67], 0 offen offset:64 sc1 // (op 266) Store value
buffer_store_dword a2, v8, s[64:67], 0 offen offset:128 sc1 // (op 266) Store value
buffer_store_dword a3, v8, s[64:67], 0 offen offset:192 sc1 // (op 266) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 270) 
v_cmp_eq_i32 vcc, v1, 0 // (op 274) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_274 // (op 274) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_274
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
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_274 // (op 274) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_274
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_274:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_274:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Send_Tile_269 // (op 269) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Send_Tile_269
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Send_Tile_269:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Send_Tile_269:

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
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Receive_Tile_285 // (op 285) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Receive_Tile_285
s_mov_b32 s52, 0 // (op 251) call()
s_mul_i32 s53, s52, s8 // (op 250) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 250) 
s_mov_b32 s84, s54 // (op 250) convert

s_sub_u32 s54, s84, s19 // (op 250) 

s_sub_u32 s55, s54, 1 // (op 250) 

s_cmp_lt_u32 s53, s55 // (op 250) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_250 // (op 250) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_250 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_ReceiveTileLoop_250:
s_add_u32 s53, s16, 1 // (op 2382) 
v_lshrrev_b32 v8, 4, v1 // (op 2382) 
s_add_u32 s54, s53, s52 // (op 2382) 
s_lshl_b32 s54, s54, 6 // (op 2382) 

v_lshlrev_b32 v11, 2, v8 // (op 2382) 

v_add_lshl_u32 v8, s54, v11, 4 // (op 2382) 


v_and_b32 v11, -16, v1 // (op 2382) 
v_sub_u32 v12, v1, v11 // (op 2382) 

v_add_lshl_u32 v11, v8, v12, 2 // (op 2382) 


s_lshl_b32 s53, s15, 12 // (op 2384) 
s_mov_b32 s64, s28 // (op 2384) reg expression
s_mov_b32 s65, s29 // (op 2384) reg expression
s_mov_b32 s66, s53 // (op 2384) reg expression
s_mov_b32 s67, 131072 // (op 2384) call()

RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_289:
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
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_289 // (op 289) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_289 if true)
s_barrier  // (op 293) 
v_cmp_eq_i32 vcc, v1, 0 // (op 292) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_292 // (op 292) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_292
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
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_292 // (op 292) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_292
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_292:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_292:




buffer_load_dword v12, v11, s[64:67], 0 offen // (op 304) Load value
buffer_load_dword v13, v11, s[64:67], 0 offen offset:64 // (op 304) Load value
buffer_load_dword v14, v11, s[64:67], 0 offen offset:128 // (op 304) Load value
buffer_load_dword v15, v11, s[64:67], 0 offen offset:192 // (op 304) Load value
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




buffer_load_dword v12, v11, s[64:67], 0 offen offset:2048 // (op 1022) Load value
buffer_load_dword v13, v11, s[64:67], 0 offen offset:2112 // (op 1022) Load value
buffer_load_dword v14, v11, s[64:67], 0 offen offset:2176 // (op 1022) Load value
buffer_load_dword v15, v11, s[64:67], 0 offen offset:2240 // (op 1022) Load value
v_accvgpr_read v16, a4 // (op 1023) reg expression
v_accvgpr_read v17, a5 // (op 1023) reg expression
v_accvgpr_read v18, a6 // (op 1023) reg expression
v_accvgpr_read v19, a7 // (op 1023) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v16, v16, v12 // (op 1024) 
s_waitcnt vmcnt(2)
v_add_f32 v17, v17, v13 // (op 1024) 
s_waitcnt vmcnt(1)
v_add_f32 v18, v18, v14 // (op 1024) 
s_waitcnt vmcnt(0)
v_add_f32 v19, v19, v15 // (op 1024) 

v_accvgpr_write a4, v16 // (op 1025) reg expression
v_accvgpr_write a5, v17 // (op 1025) reg expression
v_accvgpr_write a6, v18 // (op 1025) reg expression
v_accvgpr_write a7, v19 // (op 1025) reg expression

s_add_u32 s52, s52, 1 // (op 252) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 250) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 250) 
s_mov_b32 s84, s54 // (op 250) convert

s_sub_u32 s54, s84, s19 // (op 250) 

s_sub_u32 s55, s54, 1 // (op 250) 

s_cmp_lt_u32 s53, s55 // (op 250) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_ReceiveTileLoop_250 // (op 250) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_ReceiveTileLoop_250 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_250:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Receive_Tile_285 // (op 285) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Receive_Tile_285
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Receive_Tile_285:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Receive_Tile_285:

s_cmp_eq_i32 s18, 0 // (op 328) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_328 // (op 328) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_328
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 106) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_106 // (op 106) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_106
v_accvgpr_read v12, a0 // (op 38) 
v_accvgpr_read v13, a1 // (op 38) 
v_accvgpr_read v14, a2 // (op 38) 
v_accvgpr_read v15, a3 // (op 38) 
v_mul_f32 v16, v7, v12 // (op 38) 
v_mul_f32 v17, v7, v13 // (op 38) 
v_mul_f32 v18, v7, v14 // (op 38) 
v_mul_f32 v19, v7, v15 // (op 38) 




buffer_store_dwordx4 v[16:19], v10, s[60:63], 0 offen // (op 44) Store value

v_accvgpr_read v12, a4 // (op 1047) 
v_accvgpr_read v13, a5 // (op 1047) 
v_accvgpr_read v14, a6 // (op 1047) 
v_accvgpr_read v15, a7 // (op 1047) 
v_mul_f32 v16, v7, v12 // (op 1047) 
v_mul_f32 v17, v7, v13 // (op 1047) 
v_mul_f32 v18, v7, v14 // (op 1047) 
v_mul_f32 v19, v7, v15 // (op 1047) 




buffer_store_dwordx4 v[16:19], v10, s[60:63], 0 offen offset:128 // (op 1049) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_106 // (op 106) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_106
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_106:



buffer_load_dwordx4 v[12:15], v9, s[56:59], 0 offen offset:128 // (op 1077) Load value
v_accvgpr_read v16, a4 // (op 1074) 
v_accvgpr_read v17, a5 // (op 1074) 
v_accvgpr_read v18, a6 // (op 1074) 
v_accvgpr_read v19, a7 // (op 1074) 
v_mul_f32 v20, v7, v16 // (op 1074) 
v_mul_f32 v21, v7, v17 // (op 1074) 
v_mul_f32 v22, v7, v18 // (op 1074) 
v_mul_f32 v23, v7, v19 // (op 1074) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v6, v12 // (op 1078) 
v_mul_f32 v17, v6, v13 // (op 1078) 
v_mul_f32 v18, v6, v14 // (op 1078) 
v_mul_f32 v19, v6, v15 // (op 1078) 

v_add_f32 v12, v16, v20 // (op 1075) 
v_add_f32 v13, v17, v21 // (op 1075) 
v_add_f32 v14, v18, v22 // (op 1075) 
v_add_f32 v15, v19, v23 // (op 1075) 





buffer_store_dwordx4 v[12:15], v10, s[60:63], 0 offen offset:128 // (op 1076) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v9, s[56:59], 0 offen // (op 118) Load value
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





buffer_store_dwordx4 v[12:15], v10, s[60:63], 0 offen // (op 117) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_106:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_328 // (op 328) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_328
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_328:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_328:


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
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_SKStreamTileLoop_241 // (op 241) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_SKStreamTileLoop_241 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_241:











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
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_396 // (op 396) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_396 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_DPStreamTileLoop_396:
v_lshrrev_b32 v0, 6, v1 // (op 2180) 
s_mul_i32 s15, s16, s50 // (op 2180) 
s_add_u32 s15, s15, s0 // (op 2180) 
v_and_b32 v5, 63, v1 // (op 2180) 
s_add_u32 s6, s15, s9 // (op 2180) 

v_lshrrev_b32 v8, 4, v5 // (op 2180) 
s_mul_hi_u32 s15, s6, s13 // (op 2180) 
v_lshlrev_b32 v9, 2, v8 // (op 2180) 

s_sub_u32 s7, s6, s15 // (op 2180) 
v_lshl_add_u32 v8, v0, 4, v9 // (op 2180) 


s_lshr_b32 s8, s7, 1 // (op 2180) 

v_lshrrev_b32 v0, 2, v5 // (op 2180) 

s_add_u32 s7, s8, s15 // (op 2180) 


v_and_b32 v5, 3, v0 // (op 2180) 

s_lshr_b32 s8, s7, s11 // (op 2180) 

v_and_b32 v0, 3, v1 // (op 2180) 
s_and_b32 s7, s11, 2147483648 // (op 2180) 
v_lshl_add_u32 v9, v5, 2, v0 // (op 2180) 


s_cmp_eq_i32 s7, 0 // (op 2180) 
s_mov_b32 s15, scc // (op 2180) 

s_cmp_eq_i32 s15, 0 // (op 2180) 
s_cselect_b32 s7, s6, s8 // (op 2180) 



s_mul_hi_u32 s6, s7, s12 // (op 2180) 
s_sub_u32 s8, s7, s6 // (op 2180) 
s_lshr_b32 s15, s8, 1 // (op 2180) 

s_add_u32 s8, s15, s6 // (op 2180) 


s_lshr_b32 s6, s8, s10 // (op 2180) 

s_and_b32 s8, s10, 2147483648 // (op 2180) 
s_cmp_eq_i32 s8, 0 // (op 2180) 
s_mov_b32 s15, scc // (op 2180) 

s_cmp_eq_i32 s15, 0 // (op 2180) 
s_cselect_b32 s8, s7, s6 // (op 2180) 


s_mul_i32 s6, s8, s14 // (op 2180) 
v_lshl_add_u32 v0, s8, 6, v8 // (op 2180) 


s_sub_u32 s8, s7, s6 // (op 2180) 


v_lshl_add_u32 v5, s8, 4, v9 // (op 2180) 


v_mul_lo_u32 v8, v5, s4 // (op 2180) 

v_add_lshl_u32 v5, v0, v8, 2 // (op 2180) 


s_mul_i32 s19, s4, s23 // (op 2182) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 2182) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 2182) least significant
s_mul_hi_u32 s6, s4, s22 // (op 2182) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2182) most significant: sum
s_add_u32 s19, s19, s6 // (op 2182) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 2182) 

s_mov_b32 s56, s20 // (op 2182) reg expression
s_mov_b32 s57, s21 // (op 2182) reg expression
s_mov_b32 s58, s28 // (op 2182) reg expression
s_mov_b32 s59, 131072 // (op 2182) call()

v_lshrrev_b32 v0, 6, v1 // (op 2280) 
s_mul_i32 s15, s16, s50 // (op 2280) 
s_add_u32 s15, s15, s0 // (op 2280) 
v_and_b32 v8, 63, v1 // (op 2280) 
s_add_u32 s6, s15, s9 // (op 2280) 

v_lshrrev_b32 v9, 4, v8 // (op 2280) 
s_mul_hi_u32 s15, s6, s13 // (op 2280) 
v_lshlrev_b32 v10, 2, v9 // (op 2280) 

s_sub_u32 s7, s6, s15 // (op 2280) 
v_lshl_add_u32 v9, v0, 4, v10 // (op 2280) 


s_lshr_b32 s8, s7, 1 // (op 2280) 

v_lshrrev_b32 v0, 2, v8 // (op 2280) 

s_add_u32 s7, s8, s15 // (op 2280) 


v_and_b32 v8, 3, v0 // (op 2280) 

s_lshr_b32 s8, s7, s11 // (op 2280) 

v_and_b32 v0, 3, v1 // (op 2280) 
s_and_b32 s7, s11, 2147483648 // (op 2280) 
v_lshl_add_u32 v10, v8, 2, v0 // (op 2280) 


s_cmp_eq_i32 s7, 0 // (op 2280) 
s_mov_b32 s15, scc // (op 2280) 

s_cmp_eq_i32 s15, 0 // (op 2280) 
s_cselect_b32 s7, s6, s8 // (op 2280) 



s_mul_hi_u32 s6, s7, s12 // (op 2280) 
s_sub_u32 s8, s7, s6 // (op 2280) 
s_lshr_b32 s15, s8, 1 // (op 2280) 

s_add_u32 s8, s15, s6 // (op 2280) 


s_lshr_b32 s6, s8, s10 // (op 2280) 

s_and_b32 s8, s10, 2147483648 // (op 2280) 
s_cmp_eq_i32 s8, 0 // (op 2280) 
s_mov_b32 s15, scc // (op 2280) 

s_cmp_eq_i32 s15, 0 // (op 2280) 
s_cselect_b32 s8, s7, s6 // (op 2280) 


s_mul_i32 s6, s8, s14 // (op 2280) 
v_lshl_add_u32 v0, s8, 6, v9 // (op 2280) 


s_sub_u32 s8, s7, s6 // (op 2280) 


v_lshl_add_u32 v8, s8, 4, v10 // (op 2280) 


v_mul_lo_u32 v9, v8, s24 // (op 2280) 

v_add_lshl_u32 v8, v0, v9, 2 // (op 2280) 


s_mul_i32 s19, s24, s23 // (op 2282) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 2282) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 2282) least significant
s_mul_hi_u32 s6, s24, s22 // (op 2282) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2282) most significant: sum
s_add_u32 s19, s19, s6 // (op 2282) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 2282) 

s_mov_b32 s60, s26 // (op 2282) reg expression
s_mov_b32 s61, s27 // (op 2282) reg expression
s_mov_b32 s62, s28 // (op 2282) reg expression
s_mov_b32 s63, 131072 // (op 2282) call()

v_accvgpr_write a0, 0.00000 // (op 153) call()
v_accvgpr_write a1, 0.00000 // (op 153) call()
v_accvgpr_write a2, 0.00000 // (op 153) call()
v_accvgpr_write a3, 0.00000 // (op 153) call()
v_accvgpr_write a4, 0.00000 // (op 512) call()
v_accvgpr_write a5, 0.00000 // (op 512) call()
v_accvgpr_write a6, 0.00000 // (op 512) call()
v_accvgpr_write a7, 0.00000 // (op 512) call()
v_lshrrev_b32 v0, 6, v1 // (op 1669) 
s_mul_i32 s15, s16, s50 // (op 1669) 
s_add_u32 s15, s15, s0 // (op 1669) 
v_and_b32 v9, 15, v1 // (op 1669) 
s_add_u32 s6, s15, s9 // (op 1669) 

v_lshl_add_u32 v10, v0, 4, v9 // (op 1669) 


s_mul_hi_u32 s15, s6, s13 // (op 1669) 
v_and_b32 v0, 63, v1 // (op 1669) 
s_sub_u32 s7, s6, s15 // (op 1669) 
v_lshrrev_b32 v9, 4, v0 // (op 1669) 

s_lshr_b32 s8, s7, 1 // (op 1669) 

v_lshlrev_b32 v0, 5, v9 // (op 1669) 

s_add_u32 s7, s8, s15 // (op 1669) 


s_lshr_b32 s8, s7, s11 // (op 1669) 

s_and_b32 s7, s11, 2147483648 // (op 1669) 
s_cmp_eq_i32 s7, 0 // (op 1669) 
s_mov_b32 s15, scc // (op 1669) 

s_cmp_eq_i32 s15, 0 // (op 1669) 
s_cselect_b32 s7, s6, s8 // (op 1669) 


s_mul_hi_u32 s8, s7, s12 // (op 1669) 
s_sub_u32 s15, s7, s8 // (op 1669) 
s_lshr_b32 s18, s15, 1 // (op 1669) 

s_add_u32 s15, s18, s8 // (op 1669) 


s_lshr_b32 s8, s15, s10 // (op 1669) 

s_and_b32 s15, s10, 2147483648 // (op 1669) 
s_cmp_eq_i32 s15, 0 // (op 1669) 
s_mov_b32 s18, scc // (op 1669) 

s_cmp_eq_i32 s18, 0 // (op 1669) 
s_cselect_b32 s15, s7, s8 // (op 1669) 


v_lshl_add_u32 v9, s15, 6, v10 // (op 1669) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1669) 
s_mul_i32 s8, s7, s18 // (op 1669) 


s_sub_u32 s7, s6, s8 // (op 1669) 


v_lshl_add_u32 v10, s7, 7, v0 // (op 1669) 


v_mov_b32 v13, 0 // (op 1669) convert
v_mov_b32 v12, v10 // (op 1669) convert

v_mul_lo_u32 v11, v9, s31 // (op 1669) most significant: low of low * high
v_mul_hi_u32 v0, v9, s30 // (op 1669) most significant: high of low * low
v_mul_lo_u32 v10, v9, s30 // (op 1669) least significant: low of low * low
v_add_u32 v11, v11, v0 // (op 1669) most significant: sum

v_add_co_u32 v10, s[18:19], v10, v12 // (op 1669) least significant half
v_addc_co_u32 v11, s[18:19], v11, v13, s[18:19] // (op 1669) most significant half



v_lshlrev_b64 v[12:13], 2, v[10:11] // (op 1669) 

v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 1669) 

v_mov_b32 v0, v10 // (op 1669) convert

s_mul_i32 s19, s30, s35 // (op 1671) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 1671) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 1671) least significant
s_mul_hi_u32 s6, s30, s34 // (op 1671) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1671) most significant: sum
s_add_u32 s19, s19, s6 // (op 1671) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 1671) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 1671) 

s_mov_b32 s68, s32 // (op 1671) reg expression
s_mov_b32 s69, s33 // (op 1671) reg expression
s_mov_b32 s70, s18 // (op 1671) reg expression
s_mov_b32 s71, 131072 // (op 1671) call()

v_lshrrev_b32 v9, 6, v1 // (op 1690) 
s_mul_i32 s15, s16, s50 // (op 1690) 
s_add_u32 s15, s15, s0 // (op 1690) 
v_and_b32 v10, 15, v1 // (op 1690) 
s_add_u32 s6, s15, s9 // (op 1690) 

v_lshl_add_u32 v11, v9, 4, v10 // (op 1690) 


s_mul_hi_u32 s15, s6, s13 // (op 1690) 
v_and_b32 v9, 63, v1 // (op 1690) 
s_sub_u32 s7, s6, s15 // (op 1690) 
v_lshrrev_b32 v10, 4, v9 // (op 1690) 

s_lshr_b32 s8, s7, 1 // (op 1690) 

v_lshlrev_b32 v9, 5, v10 // (op 1690) 

s_add_u32 s7, s8, s15 // (op 1690) 


s_lshr_b32 s8, s7, s11 // (op 1690) 

s_and_b32 s7, s11, 2147483648 // (op 1690) 
s_cmp_eq_i32 s7, 0 // (op 1690) 
s_mov_b32 s15, scc // (op 1690) 

s_cmp_eq_i32 s15, 0 // (op 1690) 
s_cselect_b32 s7, s6, s8 // (op 1690) 


s_mul_hi_u32 s8, s7, s12 // (op 1690) 
s_sub_u32 s15, s7, s8 // (op 1690) 
s_lshr_b32 s18, s15, 1 // (op 1690) 

s_add_u32 s15, s18, s8 // (op 1690) 


s_lshr_b32 s8, s15, s10 // (op 1690) 

s_and_b32 s15, s10, 2147483648 // (op 1690) 
s_cmp_eq_i32 s15, 0 // (op 1690) 
s_mov_b32 s18, scc // (op 1690) 

s_cmp_eq_i32 s18, 0 // (op 1690) 
s_cselect_b32 s15, s7, s8 // (op 1690) 


v_lshl_add_u32 v10, s15, 6, v11 // (op 1690) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1690) 
s_mul_i32 s8, s7, s18 // (op 1690) 


s_sub_u32 s7, s6, s8 // (op 1690) 


v_lshl_add_u32 v11, s7, 7, v9 // (op 1690) 


v_mov_b32 v13, 0 // (op 1690) convert
v_mov_b32 v12, v11 // (op 1690) convert

v_mul_lo_u32 v15, v10, s31 // (op 1690) most significant: low of low * high
v_mul_hi_u32 v9, v10, s30 // (op 1690) most significant: high of low * low
v_mul_lo_u32 v14, v10, s30 // (op 1690) least significant: low of low * low
v_add_u32 v15, v15, v9 // (op 1690) most significant: sum

v_add_co_u32 v14, s[18:19], v14, v12 // (op 1690) least significant half
v_addc_co_u32 v15, s[18:19], v15, v13, s[18:19] // (op 1690) most significant half



v_lshlrev_b64 v[12:13], 2, v[14:15] // (op 1690) 

v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 1690) 

v_mov_b32 v9, v10 // (op 1690) convert

s_mul_i32 s19, s30, s35 // (op 1692) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 1692) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 1692) least significant
s_mul_hi_u32 s6, s30, s34 // (op 1692) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1692) most significant: sum
s_add_u32 s19, s19, s6 // (op 1692) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 1692) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 1692) 

s_mov_b32 s68, s32 // (op 1692) reg expression
s_mov_b32 s69, s33 // (op 1692) reg expression
s_mov_b32 s70, s18 // (op 1692) reg expression
s_mov_b32 s71, 131072 // (op 1692) call()

v_and_b32 v10, 15, v1 // (op 1816) 
s_mul_i32 s15, s16, s50 // (op 1816) 
s_add_u32 s15, s15, s0 // (op 1816) 
v_and_b32 v11, 63, v1 // (op 1816) 
s_add_u32 s6, s15, s9 // (op 1816) 

v_lshrrev_b32 v12, 4, v11 // (op 1816) 

s_mul_hi_u32 s15, s6, s13 // (op 1816) 
v_lshlrev_b32 v11, 5, v12 // (op 1816) 

s_sub_u32 s7, s6, s15 // (op 1816) 
s_lshr_b32 s8, s7, 1 // (op 1816) 

s_add_u32 s7, s8, s15 // (op 1816) 


s_lshr_b32 s8, s7, s11 // (op 1816) 

s_and_b32 s7, s11, 2147483648 // (op 1816) 
s_cmp_eq_i32 s7, 0 // (op 1816) 
s_mov_b32 s15, scc // (op 1816) 

s_cmp_eq_i32 s15, 0 // (op 1816) 
s_cselect_b32 s7, s6, s8 // (op 1816) 


s_mul_hi_u32 s8, s7, s12 // (op 1816) 
s_sub_u32 s15, s7, s8 // (op 1816) 
s_lshr_b32 s18, s15, 1 // (op 1816) 

s_add_u32 s15, s18, s8 // (op 1816) 


s_lshr_b32 s8, s15, s10 // (op 1816) 

s_and_b32 s15, s10, 2147483648 // (op 1816) 
s_cmp_eq_i32 s15, 0 // (op 1816) 
s_mov_b32 s18, scc // (op 1816) 

s_cmp_eq_i32 s18, 0 // (op 1816) 
s_cselect_b32 s15, s7, s8 // (op 1816) 


s_mul_i32 s8, s15, s14 // (op 1816) 

s_sub_u32 s15, s7, s8 // (op 1816) 

v_lshl_add_u32 v12, s15, 4, v10 // (op 1816) 


v_mul_lo_u32 v15, v12, s37 // (op 1816) most significant: low of low * high
v_mul_hi_u32 v10, v12, s36 // (op 1816) most significant: high of low * low
v_mul_lo_u32 v14, v12, s36 // (op 1816) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 1816) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1816) 
s_mul_i32 s8, s7, s18 // (op 1816) 


s_sub_u32 s7, s6, s8 // (op 1816) 


v_lshl_add_u32 v10, s7, 7, v11 // (op 1816) 


v_mov_b32 v13, 0 // (op 1816) convert
v_mov_b32 v12, v10 // (op 1816) convert

v_add_co_u32 v10, s[18:19], v12, v14 // (op 1816) least significant half
v_addc_co_u32 v11, s[18:19], v13, v15, s[18:19] // (op 1816) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 1816) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 1816) 

v_mov_b32 v10, v12 // (op 1816) convert

s_mul_i32 s19, s36, s23 // (op 1818) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 1818) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 1818) least significant
s_mul_hi_u32 s6, s36, s22 // (op 1818) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1818) most significant: sum
s_add_u32 s19, s19, s6 // (op 1818) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 1818) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 1818) 

s_mov_b32 s72, s38 // (op 1818) reg expression
s_mov_b32 s73, s39 // (op 1818) reg expression
s_mov_b32 s74, s18 // (op 1818) reg expression
s_mov_b32 s75, 131072 // (op 1818) call()

v_and_b32 v11, 15, v1 // (op 1835) 
s_mul_i32 s15, s16, s50 // (op 1835) 
s_add_u32 s15, s15, s0 // (op 1835) 
v_and_b32 v12, 63, v1 // (op 1835) 
s_add_u32 s6, s15, s9 // (op 1835) 

v_lshrrev_b32 v13, 4, v12 // (op 1835) 

s_mul_hi_u32 s15, s6, s13 // (op 1835) 
v_lshlrev_b32 v12, 5, v13 // (op 1835) 

s_sub_u32 s7, s6, s15 // (op 1835) 
s_lshr_b32 s8, s7, 1 // (op 1835) 

s_add_u32 s7, s8, s15 // (op 1835) 


s_lshr_b32 s8, s7, s11 // (op 1835) 

s_and_b32 s7, s11, 2147483648 // (op 1835) 
s_cmp_eq_i32 s7, 0 // (op 1835) 
s_mov_b32 s15, scc // (op 1835) 

s_cmp_eq_i32 s15, 0 // (op 1835) 
s_cselect_b32 s7, s6, s8 // (op 1835) 


s_mul_hi_u32 s8, s7, s12 // (op 1835) 
s_sub_u32 s15, s7, s8 // (op 1835) 
s_lshr_b32 s18, s15, 1 // (op 1835) 

s_add_u32 s15, s18, s8 // (op 1835) 


s_lshr_b32 s8, s15, s10 // (op 1835) 

s_and_b32 s15, s10, 2147483648 // (op 1835) 
s_cmp_eq_i32 s15, 0 // (op 1835) 
s_mov_b32 s18, scc // (op 1835) 

s_cmp_eq_i32 s18, 0 // (op 1835) 
s_cselect_b32 s15, s7, s8 // (op 1835) 


s_mul_i32 s8, s15, s14 // (op 1835) 

s_sub_u32 s15, s7, s8 // (op 1835) 

v_lshl_add_u32 v13, s15, 4, v11 // (op 1835) 


v_mul_lo_u32 v15, v13, s37 // (op 1835) most significant: low of low * high
v_mul_hi_u32 v11, v13, s36 // (op 1835) most significant: high of low * low
v_mul_lo_u32 v14, v13, s36 // (op 1835) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 1835) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1835) 
s_mul_i32 s8, s7, s18 // (op 1835) 


s_sub_u32 s7, s6, s8 // (op 1835) 


v_lshl_add_u32 v11, s7, 7, v12 // (op 1835) 


v_mov_b32 v13, 0 // (op 1835) convert
v_mov_b32 v12, v11 // (op 1835) convert

v_add_co_u32 v16, s[18:19], v12, v14 // (op 1835) least significant half
v_addc_co_u32 v17, s[18:19], v13, v15, s[18:19] // (op 1835) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 1835) 


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 1835) 

v_mov_b32 v11, v14 // (op 1835) convert

s_mul_i32 s19, s36, s23 // (op 1837) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 1837) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 1837) least significant
s_mul_hi_u32 s6, s36, s22 // (op 1837) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1837) most significant: sum
s_add_u32 s19, s19, s6 // (op 1837) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 1837) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 1837) 

s_mov_b32 s72, s38 // (op 1837) reg expression
s_mov_b32 s73, s39 // (op 1837) reg expression
s_mov_b32 s74, s18 // (op 1837) reg expression
s_mov_b32 s75, 131072 // (op 1837) call()

v_lshrrev_b32 v12, 2, v1 // (op 1944) 
s_mul_i32 s15, s16, s50 // (op 1944) 
s_add_u32 s15, s15, s0 // (op 1944) 
v_lshlrev_b32 v13, 1, v12 // (op 1944) 

s_add_u32 s6, s15, s9 // (op 1944) 

v_and_b32 v12, -4, v1 // (op 1944) 
s_mul_hi_u32 s15, s6, s13 // (op 1944) 
v_sub_u32 v14, v1, v12 // (op 1944) 

s_sub_u32 s7, s6, s15 // (op 1944) 
s_lshr_b32 s8, s7, 1 // (op 1944) 

s_add_u32 s7, s8, s15 // (op 1944) 


s_lshr_b32 s8, s7, s11 // (op 1944) 

s_and_b32 s7, s11, 2147483648 // (op 1944) 
s_cmp_eq_i32 s7, 0 // (op 1944) 
s_mov_b32 s15, scc // (op 1944) 

s_cmp_eq_i32 s15, 0 // (op 1944) 
s_cselect_b32 s7, s6, s8 // (op 1944) 


s_mul_hi_u32 s8, s7, s12 // (op 1944) 
s_sub_u32 s15, s7, s8 // (op 1944) 
s_lshr_b32 s18, s15, 1 // (op 1944) 

s_add_u32 s15, s18, s8 // (op 1944) 


s_lshr_b32 s8, s15, s10 // (op 1944) 

s_and_b32 s15, s10, 2147483648 // (op 1944) 
s_cmp_eq_i32 s15, 0 // (op 1944) 
s_mov_b32 s18, scc // (op 1944) 

s_cmp_eq_i32 s18, 0 // (op 1944) 
s_cselect_b32 s15, s7, s8 // (op 1944) 


v_lshl_add_u32 v12, s15, 6, v13 // (op 1944) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1944) 
s_mul_i32 s8, s7, s18 // (op 1944) 


s_sub_u32 s7, s6, s8 // (op 1944) 


v_lshl_add_u32 v13, s7, 2, v14 // (op 1944) 


v_mul_lo_u32 v14, v12, s40 // (op 1944) 
v_add_u32 v14, v14, v13 // (op 1944) 


s_mul_i32 s19, s40, s35 // (op 1946) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 1946) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 1946) least significant
s_mul_hi_u32 s6, s40, s34 // (op 1946) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1946) most significant: sum
s_add_u32 s19, s19, s6 // (op 1946) most significant: sum


s_mov_b32 s76, s42 // (op 1946) reg expression
s_mov_b32 s77, s43 // (op 1946) reg expression
s_mov_b32 s78, s18 // (op 1946) reg expression
s_mov_b32 s79, 131072 // (op 1946) call()

v_lshrrev_b32 v12, 2, v1 // (op 1963) 
s_mul_i32 s15, s16, s50 // (op 1963) 
s_add_u32 s15, s15, s0 // (op 1963) 
v_lshlrev_b32 v13, 1, v12 // (op 1963) 

s_add_u32 s6, s15, s9 // (op 1963) 

v_and_b32 v12, -4, v1 // (op 1963) 
s_mul_hi_u32 s15, s6, s13 // (op 1963) 
v_sub_u32 v15, v1, v12 // (op 1963) 

s_sub_u32 s7, s6, s15 // (op 1963) 
s_lshr_b32 s8, s7, 1 // (op 1963) 

s_add_u32 s7, s8, s15 // (op 1963) 


s_lshr_b32 s8, s7, s11 // (op 1963) 

s_and_b32 s7, s11, 2147483648 // (op 1963) 
s_cmp_eq_i32 s7, 0 // (op 1963) 
s_mov_b32 s15, scc // (op 1963) 

s_cmp_eq_i32 s15, 0 // (op 1963) 
s_cselect_b32 s7, s6, s8 // (op 1963) 


s_mul_hi_u32 s8, s7, s12 // (op 1963) 
s_sub_u32 s15, s7, s8 // (op 1963) 
s_lshr_b32 s18, s15, 1 // (op 1963) 

s_add_u32 s15, s18, s8 // (op 1963) 


s_lshr_b32 s8, s15, s10 // (op 1963) 

s_and_b32 s15, s10, 2147483648 // (op 1963) 
s_cmp_eq_i32 s15, 0 // (op 1963) 
s_mov_b32 s18, scc // (op 1963) 

s_cmp_eq_i32 s18, 0 // (op 1963) 
s_cselect_b32 s15, s7, s8 // (op 1963) 


v_lshl_add_u32 v12, s15, 6, v13 // (op 1963) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1963) 
s_mul_i32 s8, s7, s18 // (op 1963) 


s_sub_u32 s7, s6, s8 // (op 1963) 


v_lshl_add_u32 v13, s7, 2, v15 // (op 1963) 


v_mul_lo_u32 v15, v12, s40 // (op 1963) 
v_add_u32 v15, v15, v13 // (op 1963) 


s_mul_i32 s19, s40, s35 // (op 1965) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 1965) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 1965) least significant
s_mul_hi_u32 s6, s40, s34 // (op 1965) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1965) most significant: sum
s_add_u32 s19, s19, s6 // (op 1965) most significant: sum


s_mov_b32 s76, s42 // (op 1965) reg expression
s_mov_b32 s77, s43 // (op 1965) reg expression
s_mov_b32 s78, s18 // (op 1965) reg expression
s_mov_b32 s79, 131072 // (op 1965) call()

v_and_b32 v12, 15, v1 // (op 2072) 
s_mul_i32 s15, s16, s50 // (op 2072) 
s_add_u32 s15, s15, s0 // (op 2072) 
v_and_b32 v13, 63, v1 // (op 2072) 
s_add_u32 s6, s15, s9 // (op 2072) 

v_lshrrev_b32 v16, 4, v13 // (op 2072) 

s_mul_hi_u32 s15, s6, s13 // (op 2072) 
s_sub_u32 s7, s6, s15 // (op 2072) 
s_lshr_b32 s8, s7, 1 // (op 2072) 

s_add_u32 s7, s8, s15 // (op 2072) 


s_lshr_b32 s8, s7, s11 // (op 2072) 

s_and_b32 s7, s11, 2147483648 // (op 2072) 
s_cmp_eq_i32 s7, 0 // (op 2072) 
s_mov_b32 s15, scc // (op 2072) 

s_cmp_eq_i32 s15, 0 // (op 2072) 
s_cselect_b32 s7, s6, s8 // (op 2072) 


s_mul_hi_u32 s8, s7, s12 // (op 2072) 
s_sub_u32 s15, s7, s8 // (op 2072) 
s_lshr_b32 s18, s15, 1 // (op 2072) 

s_add_u32 s15, s18, s8 // (op 2072) 


s_lshr_b32 s8, s15, s10 // (op 2072) 

s_and_b32 s15, s10, 2147483648 // (op 2072) 
s_cmp_eq_i32 s15, 0 // (op 2072) 
s_mov_b32 s18, scc // (op 2072) 

s_cmp_eq_i32 s18, 0 // (op 2072) 
s_cselect_b32 s15, s7, s8 // (op 2072) 


s_mul_i32 s8, s15, s14 // (op 2072) 

s_sub_u32 s15, s7, s8 // (op 2072) 

v_lshl_add_u32 v13, s15, 4, v12 // (op 2072) 


v_mul_lo_u32 v12, v13, s44 // (op 2072) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 2072) 
s_mul_i32 s8, s7, s18 // (op 2072) 


s_sub_u32 s7, s6, s8 // (op 2072) 


v_lshl_add_u32 v13, s7, 2, v16 // (op 2072) 


v_add_u32 v16, v13, v12 // (op 2072) 


s_mul_i32 s19, s44, s23 // (op 2074) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 2074) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 2074) least significant
s_mul_hi_u32 s6, s44, s22 // (op 2074) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2074) most significant: sum
s_add_u32 s19, s19, s6 // (op 2074) most significant: sum


s_mov_b32 s80, s46 // (op 2074) reg expression
s_mov_b32 s81, s47 // (op 2074) reg expression
s_mov_b32 s82, s18 // (op 2074) reg expression
s_mov_b32 s83, 131072 // (op 2074) call()

v_and_b32 v12, 15, v1 // (op 2091) 
s_mul_i32 s15, s16, s50 // (op 2091) 
s_add_u32 s15, s15, s0 // (op 2091) 
v_and_b32 v13, 63, v1 // (op 2091) 
s_add_u32 s6, s15, s9 // (op 2091) 

v_lshrrev_b32 v17, 4, v13 // (op 2091) 

s_mul_hi_u32 s15, s6, s13 // (op 2091) 
s_sub_u32 s7, s6, s15 // (op 2091) 
s_lshr_b32 s8, s7, 1 // (op 2091) 

s_add_u32 s7, s8, s15 // (op 2091) 


s_lshr_b32 s8, s7, s11 // (op 2091) 

s_and_b32 s7, s11, 2147483648 // (op 2091) 
s_cmp_eq_i32 s7, 0 // (op 2091) 
s_mov_b32 s15, scc // (op 2091) 

s_cmp_eq_i32 s15, 0 // (op 2091) 
s_cselect_b32 s7, s6, s8 // (op 2091) 


s_mul_hi_u32 s8, s7, s12 // (op 2091) 
s_sub_u32 s15, s7, s8 // (op 2091) 
s_lshr_b32 s18, s15, 1 // (op 2091) 

s_add_u32 s15, s18, s8 // (op 2091) 


s_lshr_b32 s8, s15, s10 // (op 2091) 

s_and_b32 s15, s10, 2147483648 // (op 2091) 
s_cmp_eq_i32 s15, 0 // (op 2091) 
s_mov_b32 s18, scc // (op 2091) 

s_cmp_eq_i32 s18, 0 // (op 2091) 
s_cselect_b32 s15, s7, s8 // (op 2091) 


s_mul_i32 s8, s15, s14 // (op 2091) 

s_sub_u32 s15, s7, s8 // (op 2091) 

v_lshl_add_u32 v13, s15, 4, v12 // (op 2091) 


v_mul_lo_u32 v12, v13, s44 // (op 2091) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 2091) 
s_mul_i32 s8, s7, s18 // (op 2091) 


s_sub_u32 s7, s6, s8 // (op 2091) 


v_lshl_add_u32 v13, s7, 2, v17 // (op 2091) 


v_add_u32 v17, v13, v12 // (op 2091) 


s_mul_i32 s19, s44, s23 // (op 2093) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 2093) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 2093) least significant
s_mul_hi_u32 s6, s44, s22 // (op 2093) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2093) most significant: sum
s_add_u32 s19, s19, s6 // (op 2093) most significant: sum


s_mov_b32 s80, s46 // (op 2093) reg expression
s_mov_b32 s81, s47 // (op 2093) reg expression
s_mov_b32 s82, s18 // (op 2093) reg expression
s_mov_b32 s83, 131072 // (op 2093) call()




buffer_load_ubyte v12, v16, s[80:83], 0 offen offset: 0 // (op 1382) Load value


v_mov_b32 v13, v0 // (op 1388) reg expression

buffer_load_dwordx4 v[18:21], v13, s[68:71], 0 offen // (op 1388) Load value




buffer_load_dwordx4 v[22:25], v10, s[72:75], 0 offen // (op 1394) Load value


s_lshl_b64 s[18:19], s[30:31], 5 // (op 1400) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 1400) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 1400) 

v_add_u32 v13, v0, s18 // (op 1400) 


buffer_load_dwordx4 v[26:29], v13, s[68:71], 0 offen // (op 1400) Load value


v_mov_b64 v[30:31], s[40:41] // (op 1405) convert

v_mov_b32 v13, v14 // (op 1405) 
buffer_load_ubyte v32, v13, s[76:79], 0 offen offset: 0 // (op 1405) Load value
v_add_u32 v13, v13, v30 // (op 1405) 
buffer_load_ubyte v33, v13, s[76:79], 0 offen offset: 0 // (op 1405) Load value





buffer_load_ubyte v13, v16, s[80:83], 0 offen offset:4 // (op 1409) Load value



v_add_u32 v16, 64, v0 // (op 1415) 

buffer_load_dwordx4 v[34:37], v16, s[68:71], 0 offen // (op 1415) Load value




buffer_load_dwordx4 v[38:41], v10, s[72:75], 0 offen offset:64 // (op 1421) Load value



s_lshl_b64 s[18:19], s[30:31], 5 // (op 1427) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 1427) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 1427) 

s_add_u32 s15, s18, 64 // (op 1427) 

v_add_u32 v10, v0, s15 // (op 1427) 


buffer_load_dwordx4 v[42:45], v10, s[68:71], 0 offen // (op 1427) Load value



v_mov_b64 v[30:31], s[40:41] // (op 1432) convert

v_mov_b32 v0, v14 // (op 1432) 
buffer_load_ubyte v46, v0, s[76:79], 0 offen offset:4 // (op 1432) Load value
v_add_u32 v0, v0, v30 // (op 1432) 
buffer_load_ubyte v47, v0, s[76:79], 0 offen offset:4 // (op 1432) Load value



v_and_b32 v0, -4, v1 // (op 2326) 
v_sub_u32 v10, v1, v0 // (op 2326) 

v_lshrrev_b32 v0, 2, v1 // (op 2326) 
v_lshl_add_u32 v14, v0, 3, v10 // (op 2326) 


s_barrier  // (op 2530) 



s_waitcnt vmcnt(7)
ds_write_b8 v14, v32 // (op 1434) Store local data 
s_waitcnt vmcnt(6)
ds_write_b8 v14, v33 offset:4 // (op 1434) Store local data 

v_and_b32 v0, -4, v1 // (op 2297) 
v_sub_u32 v10, v1, v0 // (op 2297) 

v_lshrrev_b32 v0, 2, v1 // (op 2297) 
v_lshl_add_u32 v14, v0, 3, v10 // (op 2297) 


v_and_b32 v0, -4, v1 // (op 2397) 
v_sub_u32 v10, v1, v0 // (op 2397) 

v_lshrrev_b32 v0, 2, v1 // (op 2397) 
v_lshl_add_u32 v16, v0, 3, v10 // (op 2397) 


s_mov_b32 s1, 0 // (op 514) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 513) 
s_mov_b32 s15, s18 // (op 513) convert

s_and_b32 s6, s15, 4294967294 // (op 513) 

s_cmp_lt_u32 s1, s6 // (op 513) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_513 // (op 513) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_513 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoop_513:
s_add_u32 s1, s1, 2 // (op 515) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2527) 



ds_read_u8 v0, v3 // (op 160) Load local data 



ds_read_u8 v10, v3 offset:128 // (op 539) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[22:25], a[0:3], v0, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 163) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:29], v[22:25], a[4:7], v10, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 541) 



v_mov_b32 v0, v17 // (op 162) 

buffer_load_ubyte v12, v0, s[80:83], 0 offen offset:8 // (op 162) Load value


v_mov_b32 v10, v9 // (op 165) 
v_add_u32 v30, 128, v10 // (op 165) 

buffer_load_dwordx4 v[18:21], v30, s[68:71], 0 offen // (op 165) Load value



v_mov_b32 v30, v11 // (op 167) 

buffer_load_dwordx4 v[22:25], v30, s[72:75], 0 offen offset:128 // (op 167) Load value


s_lshl_b64 s[18:19], s[30:31], 5 // (op 564) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 564) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 564) 

s_add_u32 s15, s18, 128 // (op 564) 

v_add_u32 v31, v10, s15 // (op 564) 


buffer_load_dwordx4 v[26:29], v31, s[68:71], 0 offen // (op 564) Load value


v_mov_b64 v[48:49], s[40:41] // (op 157) convert

v_mov_b32 v31, v15 // (op 157) 
v_mov_b32 v50, v31 // (op 157) 
buffer_load_ubyte v32, v50, s[76:79], 0 offen offset:8 // (op 157) Load value
v_add_u32 v50, v50, v48 // (op 157) 
buffer_load_ubyte v33, v50, s[76:79], 0 offen offset:8 // (op 157) Load value


s_barrier  // (op 2546) 



s_waitcnt vmcnt(7)
ds_write_b8 v16, v46 offset:256 // (op 452) Store local data 
s_waitcnt vmcnt(6)
ds_write_b8 v16, v47 offset:260 // (op 452) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2543) 



ds_read_u8 v48, v4 offset:256 // (op 454) Load local data 



ds_read_u8 v49, v4 offset:384 // (op 548) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[38:41], a[0:3], v48, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 457) 

s_barrier  // (op 2641) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:45], v[38:41], a[4:7], v49, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 542) 




buffer_load_ubyte v13, v0, s[80:83], 0 offen offset:12 // (op 456) Load value



v_add_u32 v17, 8, v17 // (op 2083) 
v_add_u32 v0, 192, v10 // (op 459) 

buffer_load_dwordx4 v[34:37], v0, s[68:71], 0 offen // (op 459) Load value




buffer_load_dwordx4 v[38:41], v30, s[72:75], 0 offen offset:192 // (op 461) Load value



v_add_u32 v11, 128, v11 // (op 1827) 
s_lshl_b64 s[18:19], s[30:31], 5 // (op 557) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 557) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 557) 

s_add_u32 s15, s18, 192 // (op 557) 

v_add_u32 v0, v10, s15 // (op 557) 


buffer_load_dwordx4 v[42:45], v0, s[68:71], 0 offen // (op 557) Load value



v_add_u32 v9, 128, v9 // (op 1680) 
v_mov_b64 v[48:49], s[40:41] // (op 451) convert

v_mov_b32 v0, v31 // (op 451) 
buffer_load_ubyte v46, v0, s[76:79], 0 offen offset:12 // (op 451) Load value
v_add_u32 v0, v0, v48 // (op 451) 
buffer_load_ubyte v47, v0, s[76:79], 0 offen offset:12 // (op 451) Load value



s_barrier  // (op 2533) 
v_add_u32 v15, 8, v15 // (op 1955) 



s_waitcnt vmcnt(7)
ds_write_b8 v14, v32 // (op 158) Store local data 
s_waitcnt vmcnt(6)
ds_write_b8 v14, v33 offset:4 // (op 158) Store local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 513) 
s_mov_b32 s15, s18 // (op 513) convert

s_and_b32 s6, s15, 4294967294 // (op 513) 

s_cmp_lt_u32 s1, s6 // (op 513) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoop_513 // (op 513) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoop_513 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_513:











v_lshrrev_b32 v0, 6, v1 // (op 1605) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1605) 
v_and_b32 v9, 15, v1 // (op 1605) 
s_and_b32 s15, s18, 4294967294 // (op 1605) 
v_lshl_add_u32 v10, v0, 4, v9 // (op 1605) 


s_add_u32 s6, s0, s15 // (op 1605) 

v_and_b32 v0, 63, v1 // (op 1605) 
v_lshrrev_b32 v9, 4, v0 // (op 1605) 

s_mul_i32 s15, s16, s50 // (op 1605) 
s_add_u32 s15, s15, s6 // (op 1605) 

v_lshlrev_b32 v0, 5, v9 // (op 1605) 

s_add_u32 s6, s15, s9 // (op 1605) 

s_mul_hi_u32 s15, s6, s13 // (op 1605) 
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
s_lshr_b32 s28, s15, 1 // (op 1605) 

s_add_u32 s15, s28, s8 // (op 1605) 


s_lshr_b32 s8, s15, s10 // (op 1605) 

s_and_b32 s15, s10, 2147483648 // (op 1605) 
s_cmp_eq_i32 s15, 0 // (op 1605) 
s_mov_b32 s28, scc // (op 1605) 

s_cmp_eq_i32 s28, 0 // (op 1605) 
s_cselect_b32 s15, s7, s8 // (op 1605) 


s_mul_i32 s8, s7, s18 // (op 1605) 


v_lshl_add_u32 v9, s15, 6, v10 // (op 1605) 


s_sub_u32 s7, s6, s8 // (op 1605) 


v_lshl_add_u32 v10, s7, 7, v0 // (op 1605) 


v_mov_b32 v15, 0 // (op 1605) convert
v_mov_b32 v14, v10 // (op 1605) convert

v_mul_lo_u32 v11, v9, s31 // (op 1605) most significant: low of low * high
v_mul_hi_u32 v0, v9, s30 // (op 1605) most significant: high of low * low
v_mul_lo_u32 v10, v9, s30 // (op 1605) least significant: low of low * low
v_add_u32 v11, v11, v0 // (op 1605) most significant: sum

v_add_co_u32 v10, s[18:19], v10, v14 // (op 1605) least significant half
v_addc_co_u32 v11, s[18:19], v11, v15, s[18:19] // (op 1605) most significant half



v_lshlrev_b64 v[14:15], 2, v[10:11] // (op 1605) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 1605) 

v_mov_b32 v0, v16 // (op 1605) convert

s_mul_i32 s19, s30, s35 // (op 1607) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 1607) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 1607) least significant
s_mul_hi_u32 s6, s30, s34 // (op 1607) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1607) most significant: sum
s_add_u32 s19, s19, s6 // (op 1607) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 1607) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 1607) 

s_mov_b32 s68, s32 // (op 1607) reg expression
s_mov_b32 s69, s33 // (op 1607) reg expression
s_mov_b32 s70, s18 // (op 1607) reg expression
s_mov_b32 s71, 131072 // (op 1607) call()

v_and_b32 v9, 15, v1 // (op 1771) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1771) 
v_and_b32 v10, 63, v1 // (op 1771) 
s_and_b32 s15, s18, 4294967294 // (op 1771) 
v_lshrrev_b32 v11, 4, v10 // (op 1771) 

s_add_u32 s6, s0, s15 // (op 1771) 

v_lshlrev_b32 v10, 5, v11 // (op 1771) 

s_mul_i32 s15, s16, s50 // (op 1771) 
s_add_u32 s15, s15, s6 // (op 1771) 

s_add_u32 s6, s15, s9 // (op 1771) 

s_mul_hi_u32 s15, s6, s13 // (op 1771) 
s_sub_u32 s7, s6, s15 // (op 1771) 
s_lshr_b32 s8, s7, 1 // (op 1771) 

s_add_u32 s7, s8, s15 // (op 1771) 


s_lshr_b32 s8, s7, s11 // (op 1771) 

s_and_b32 s7, s11, 2147483648 // (op 1771) 
s_cmp_eq_i32 s7, 0 // (op 1771) 
s_mov_b32 s15, scc // (op 1771) 

s_cmp_eq_i32 s15, 0 // (op 1771) 
s_cselect_b32 s7, s6, s8 // (op 1771) 


s_mul_hi_u32 s8, s7, s12 // (op 1771) 
s_sub_u32 s15, s7, s8 // (op 1771) 
s_lshr_b32 s28, s15, 1 // (op 1771) 

s_add_u32 s15, s28, s8 // (op 1771) 


s_lshr_b32 s8, s15, s10 // (op 1771) 

s_and_b32 s15, s10, 2147483648 // (op 1771) 
s_cmp_eq_i32 s15, 0 // (op 1771) 
s_mov_b32 s28, scc // (op 1771) 

s_cmp_eq_i32 s28, 0 // (op 1771) 
s_cselect_b32 s15, s7, s8 // (op 1771) 


s_mul_i32 s8, s15, s14 // (op 1771) 

s_sub_u32 s15, s7, s8 // (op 1771) 

s_mul_i32 s8, s7, s18 // (op 1771) 


v_lshl_add_u32 v11, s15, 4, v9 // (op 1771) 


s_sub_u32 s7, s6, s8 // (op 1771) 


v_mul_lo_u32 v15, v11, s37 // (op 1771) most significant: low of low * high
v_mul_hi_u32 v9, v11, s36 // (op 1771) most significant: high of low * low
v_mul_lo_u32 v14, v11, s36 // (op 1771) least significant: low of low * low
v_add_u32 v15, v15, v9 // (op 1771) most significant: sum


v_lshl_add_u32 v9, s7, 7, v10 // (op 1771) 


v_mov_b32 v11, 0 // (op 1771) convert
v_mov_b32 v10, v9 // (op 1771) convert

v_add_co_u32 v16, s[18:19], v10, v14 // (op 1771) least significant half
v_addc_co_u32 v17, s[18:19], v11, v15, s[18:19] // (op 1771) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 1771) 


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 1771) 

v_mov_b32 v9, v14 // (op 1771) convert

s_mul_i32 s19, s36, s23 // (op 1773) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 1773) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 1773) least significant
s_mul_hi_u32 s6, s36, s22 // (op 1773) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1773) most significant: sum
s_add_u32 s19, s19, s6 // (op 1773) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 1773) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 1773) 

s_mov_b32 s72, s38 // (op 1773) reg expression
s_mov_b32 s73, s39 // (op 1773) reg expression
s_mov_b32 s74, s18 // (op 1773) reg expression
s_mov_b32 s75, 131072 // (op 1773) call()

v_lshrrev_b32 v10, 2, v1 // (op 1899) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1899) 
v_lshlrev_b32 v11, 1, v10 // (op 1899) 

s_and_b32 s15, s18, 4294967294 // (op 1899) 
v_and_b32 v10, -4, v1 // (op 1899) 
s_add_u32 s6, s0, s15 // (op 1899) 

v_sub_u32 v12, v1, v10 // (op 1899) 

s_mul_i32 s15, s16, s50 // (op 1899) 
s_add_u32 s15, s15, s6 // (op 1899) 

s_add_u32 s6, s15, s9 // (op 1899) 

s_mul_hi_u32 s15, s6, s13 // (op 1899) 
s_sub_u32 s7, s6, s15 // (op 1899) 
s_lshr_b32 s8, s7, 1 // (op 1899) 

s_add_u32 s7, s8, s15 // (op 1899) 


s_lshr_b32 s8, s7, s11 // (op 1899) 

s_and_b32 s7, s11, 2147483648 // (op 1899) 
s_cmp_eq_i32 s7, 0 // (op 1899) 
s_mov_b32 s15, scc // (op 1899) 

s_cmp_eq_i32 s15, 0 // (op 1899) 
s_cselect_b32 s7, s6, s8 // (op 1899) 


s_mul_hi_u32 s8, s7, s12 // (op 1899) 
s_sub_u32 s15, s7, s8 // (op 1899) 
s_lshr_b32 s28, s15, 1 // (op 1899) 

s_add_u32 s15, s28, s8 // (op 1899) 


s_lshr_b32 s8, s15, s10 // (op 1899) 

s_and_b32 s15, s10, 2147483648 // (op 1899) 
s_cmp_eq_i32 s15, 0 // (op 1899) 
s_mov_b32 s28, scc // (op 1899) 

s_cmp_eq_i32 s28, 0 // (op 1899) 
s_cselect_b32 s15, s7, s8 // (op 1899) 


s_mul_i32 s8, s7, s18 // (op 1899) 


v_lshl_add_u32 v10, s15, 6, v11 // (op 1899) 


s_sub_u32 s7, s6, s8 // (op 1899) 


v_lshl_add_u32 v11, s7, 2, v12 // (op 1899) 


v_mul_lo_u32 v12, v10, s40 // (op 1899) 
v_add_u32 v12, v12, v11 // (op 1899) 


s_mul_i32 s19, s40, s35 // (op 1901) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 1901) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 1901) least significant
s_mul_hi_u32 s6, s40, s34 // (op 1901) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1901) most significant: sum
s_add_u32 s19, s19, s6 // (op 1901) most significant: sum


s_mov_b32 s76, s42 // (op 1901) reg expression
s_mov_b32 s77, s43 // (op 1901) reg expression
s_mov_b32 s78, s18 // (op 1901) reg expression
s_mov_b32 s79, 131072 // (op 1901) call()

v_and_b32 v10, 15, v1 // (op 2027) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2027) 
v_and_b32 v11, 63, v1 // (op 2027) 
s_and_b32 s15, s18, 4294967294 // (op 2027) 
v_lshrrev_b32 v14, 4, v11 // (op 2027) 

s_add_u32 s6, s0, s15 // (op 2027) 

s_mul_i32 s15, s16, s50 // (op 2027) 
s_add_u32 s15, s15, s6 // (op 2027) 

s_add_u32 s6, s15, s9 // (op 2027) 

s_mul_hi_u32 s15, s6, s13 // (op 2027) 
s_sub_u32 s7, s6, s15 // (op 2027) 
s_lshr_b32 s8, s7, 1 // (op 2027) 

s_add_u32 s7, s8, s15 // (op 2027) 


s_lshr_b32 s8, s7, s11 // (op 2027) 

s_and_b32 s7, s11, 2147483648 // (op 2027) 
s_cmp_eq_i32 s7, 0 // (op 2027) 
s_mov_b32 s15, scc // (op 2027) 

s_cmp_eq_i32 s15, 0 // (op 2027) 
s_cselect_b32 s7, s6, s8 // (op 2027) 


s_mul_hi_u32 s8, s7, s12 // (op 2027) 
s_sub_u32 s15, s7, s8 // (op 2027) 
s_lshr_b32 s28, s15, 1 // (op 2027) 

s_add_u32 s15, s28, s8 // (op 2027) 


s_lshr_b32 s8, s15, s10 // (op 2027) 

s_and_b32 s15, s10, 2147483648 // (op 2027) 
s_cmp_eq_i32 s15, 0 // (op 2027) 
s_mov_b32 s28, scc // (op 2027) 

s_cmp_eq_i32 s28, 0 // (op 2027) 
s_cselect_b32 s15, s7, s8 // (op 2027) 


s_mul_i32 s8, s15, s14 // (op 2027) 

s_sub_u32 s15, s7, s8 // (op 2027) 

v_lshl_add_u32 v11, s15, 4, v10 // (op 2027) 


s_mul_i32 s8, s7, s18 // (op 2027) 


v_mul_lo_u32 v10, v11, s44 // (op 2027) 

s_sub_u32 s7, s6, s8 // (op 2027) 


v_lshl_add_u32 v11, s7, 2, v14 // (op 2027) 


v_add_u32 v14, v11, v10 // (op 2027) 


s_mul_i32 s19, s44, s23 // (op 2029) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 2029) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 2029) least significant
s_mul_hi_u32 s6, s44, s22 // (op 2029) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2029) most significant: sum
s_add_u32 s19, s19, s6 // (op 2029) most significant: sum


s_mov_b32 s80, s46 // (op 2029) reg expression
s_mov_b32 s81, s47 // (op 2029) reg expression
s_mov_b32 s82, s18 // (op 2029) reg expression
s_mov_b32 s83, 131072 // (op 2029) call()

v_and_b32 v10, -4, v1 // (op 2310) 
v_sub_u32 v11, v1, v10 // (op 2310) 

v_lshrrev_b32 v10, 2, v1 // (op 2310) 
v_lshl_add_u32 v15, v10, 3, v11 // (op 2310) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 519) 
s_and_b32 s15, s18, 4294967294 // (op 519) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 518) 
s_mov_b32 s6, s18 // (op 518) convert

s_cmp_lt_u32 s15, s6 // (op 518) 






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_518 // (op 518) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_518 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_518:

s_add_u32 s15, s15, 1 // (op 520) 
v_mov_b64 v[10:11], s[40:41] // (op 409) convert

v_mov_b32 v13, v12 // (op 409) 
v_mov_b32 v16, v13 // (op 409) 
buffer_load_ubyte v18, v16, s[76:79], 0 offen offset: 0 // (op 409) Load value
v_add_u32 v16, v16, v10 // (op 409) 
buffer_load_ubyte v19, v16, s[76:79], 0 offen offset: 0 // (op 409) Load value



v_add_u32 v12, 4, v12 // (op 1891) 


s_barrier  // (op 2536) 
v_mov_b32 v13, v0 // (op 417) 



v_mov_b32 v10, v13 // (op 417) reg expression

buffer_load_dwordx4 v[20:23], v10, s[68:71], 0 offen // (op 417) Load value

s_waitcnt vmcnt(2)
ds_write_b8 v15, v18 // (op 410) Store local data 


v_mov_b32 v10, v9 // (op 419) 

buffer_load_dwordx4 v[24:27], v10, s[72:75], 0 offen // (op 419) Load value

v_add_u32 v9, 64, v9 // (op 1763) 
s_waitcnt vmcnt(2)
ds_write_b8 v15, v19 offset:4 // (op 410) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 2539) 



ds_read_u8 v10, v3 // (op 412) Load local data 


v_mov_b32 v11, v14 // (op 414) 

buffer_load_ubyte v16, v11, s[80:83], 0 offen offset: 0 // (op 414) Load value

v_add_u32 v14, 4, v14 // (op 2019) 


s_lshl_b64 s[18:19], s[30:31], 5 // (op 534) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 534) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 534) 

v_add_u32 v11, v13, s18 // (op 534) 


buffer_load_dwordx4 v[28:31], v11, s[68:71], 0 offen // (op 534) Load value

s_waitcnt vmcnt(1) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:27], a[0:3], v10, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 415) 






v_add_u32 v0, 64, v0 // (op 1593) 
ds_read_u8 v13, v3 offset:128 // (op 524) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[28:31], v[24:27], a[4:7], v13, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 526) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 518) 
s_mov_b32 s6, s18 // (op 518) convert

s_cmp_lt_u32 s15, s6 // (op 518) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_518 // (op 518) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_518 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_518:






s_barrier  // (op 2627) 
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 189) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_189 // (op 189) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_189
v_accvgpr_read v10, a0 // (op 197) 
v_accvgpr_read v11, a1 // (op 197) 
v_accvgpr_read v12, a2 // (op 197) 
v_accvgpr_read v13, a3 // (op 197) 
v_mul_f32 v14, v7, v10 // (op 197) 
v_mul_f32 v15, v7, v11 // (op 197) 
v_mul_f32 v16, v7, v12 // (op 197) 
v_mul_f32 v17, v7, v13 // (op 197) 




buffer_store_dwordx4 v[14:17], v8, s[60:63], 0 offen // (op 199) Store value

v_accvgpr_read v10, a4 // (op 676) 
v_accvgpr_read v11, a5 // (op 676) 
v_accvgpr_read v12, a6 // (op 676) 
v_accvgpr_read v13, a7 // (op 676) 
v_mul_f32 v14, v7, v10 // (op 676) 
v_mul_f32 v15, v7, v11 // (op 676) 
v_mul_f32 v16, v7, v12 // (op 676) 
v_mul_f32 v17, v7, v13 // (op 676) 




buffer_store_dwordx4 v[14:17], v8, s[60:63], 0 offen offset:128 // (op 678) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_189 // (op 189) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_189
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_189:



buffer_load_dwordx4 v[10:13], v5, s[56:59], 0 offen offset:128 // (op 706) Load value
v_accvgpr_read v14, a4 // (op 703) 
v_accvgpr_read v15, a5 // (op 703) 
v_accvgpr_read v16, a6 // (op 703) 
v_accvgpr_read v17, a7 // (op 703) 
v_mul_f32 v18, v7, v14 // (op 703) 
v_mul_f32 v19, v7, v15 // (op 703) 
v_mul_f32 v20, v7, v16 // (op 703) 
v_mul_f32 v21, v7, v17 // (op 703) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v6, v10 // (op 707) 
v_mul_f32 v15, v6, v11 // (op 707) 
v_mul_f32 v16, v6, v12 // (op 707) 
v_mul_f32 v17, v6, v13 // (op 707) 

v_add_f32 v10, v14, v18 // (op 704) 
v_add_f32 v11, v15, v19 // (op 704) 
v_add_f32 v12, v16, v20 // (op 704) 
v_add_f32 v13, v17, v21 // (op 704) 





buffer_store_dwordx4 v[10:13], v8, s[60:63], 0 offen offset:128 // (op 705) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v5, s[56:59], 0 offen // (op 212) Load value
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





buffer_store_dwordx4 v[10:13], v8, s[60:63], 0 offen // (op 211) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_189:
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
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_DPStreamTileLoop_396 // (op 396) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopTop_DPStreamTileLoop_396 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_396:





































s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2
.LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_end:
.size RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2, .LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2_end-RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2
  .amdhsa_next_free_vgpr 64
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 512 // lds bytes
  .amdhsa_accum_offset 56
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
    macrotile: [64, 16, 128]
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
  MIWaveTile: [2, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2
    .symbol: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_64x16x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 512
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 86
    .vgpr_count: 54
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

