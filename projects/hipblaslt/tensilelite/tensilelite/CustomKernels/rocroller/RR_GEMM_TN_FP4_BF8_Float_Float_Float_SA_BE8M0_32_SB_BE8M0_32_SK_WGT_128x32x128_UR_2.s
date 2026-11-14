// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2,@function
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 3630) 
v_and_b32 v3, 15, v1 // (op 3630) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 3630) 


v_and_b32 v0, 63, v1 // (op 3630) 
v_lshrrev_b32 v3, 4, v0 // (op 3630) 

v_lshl_add_u32 v0, v4, 2, v3 // (op 3630) 


v_lshrrev_b32 v3, 7, v1 // (op 3730) 
v_and_b32 v4, 15, v1 // (op 3730) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 3730) 


v_and_b32 v3, 63, v1 // (op 3730) 
v_lshrrev_b32 v4, 4, v3 // (op 3730) 

v_lshl_add_u32 v3, v5, 2, v4 // (op 3730) 


v_lshrrev_b32 v4, 7, v1 // (op 3801) 
v_and_b32 v5, 15, v1 // (op 3801) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 3801) 


v_and_b32 v4, 63, v1 // (op 3801) 
v_lshrrev_b32 v5, 4, v4 // (op 3801) 

v_lshl_add_u32 v4, v6, 2, v5 // (op 3801) 


v_lshrrev_b32 v5, 7, v1 // (op 3830) 
v_and_b32 v6, 15, v1 // (op 3830) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 3830) 


v_and_b32 v5, 63, v1 // (op 3830) 
v_lshrrev_b32 v6, 4, v5 // (op 3830) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 3830) 


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
s_cbranch_scc0 RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_241 // (op 241) Condition: Top (jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_241 if false)
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_SKStreamTileLoop_241:
v_lshrrev_b32 v8, 7, v1 // (op 3545) 
s_mul_i32 s52, s16, s8 // (op 3545) 
s_add_u32 s52, s52, s0 // (op 3545) 
v_and_b32 v9, 63, v1 // (op 3545) 
s_mul_hi_u32 s53, s52, s13 // (op 3545) 
v_lshrrev_b32 v10, 4, v9 // (op 3545) 
s_sub_u32 s54, s52, s53 // (op 3545) 
v_lshlrev_b32 v11, 2, v10 // (op 3545) 

v_lshl_add_u32 v10, v8, 4, v11 // (op 3545) 


s_lshr_b32 s55, s54, 1 // (op 3545) 

s_add_u32 s54, s55, s53 // (op 3545) 


v_lshrrev_b32 v8, 2, v9 // (op 3545) 

s_lshr_b32 s53, s54, s11 // (op 3545) 

v_and_b32 v9, 3, v8 // (op 3545) 

s_and_b32 s54, s11, 2147483648 // (op 3545) 
v_and_b32 v8, 3, v1 // (op 3545) 
s_cmp_eq_i32 s54, 0 // (op 3545) 
s_mov_b32 s55, scc // (op 3545) 

v_lshl_add_u32 v11, v9, 2, v8 // (op 3545) 


s_cmp_eq_i32 s55, 0 // (op 3545) 
s_cselect_b32 s54, s52, s53 // (op 3545) 



v_lshrrev_b32 v8, 6, v1 // (op 3545) 
s_mul_hi_u32 s52, s54, s12 // (op 3545) 
v_and_b32 v9, 1, v8 // (op 3545) 

s_sub_u32 s53, s54, s52 // (op 3545) 
v_lshl_add_u32 v8, v9, 4, v11 // (op 3545) 


s_lshr_b32 s55, s53, 1 // (op 3545) 

s_add_u32 s53, s55, s52 // (op 3545) 


s_lshr_b32 s52, s53, s10 // (op 3545) 

s_and_b32 s53, s10, 2147483648 // (op 3545) 
s_cmp_eq_i32 s53, 0 // (op 3545) 
s_mov_b32 s55, scc // (op 3545) 

s_cmp_eq_i32 s55, 0 // (op 3545) 
s_cselect_b32 s53, s54, s52 // (op 3545) 


s_mul_i32 s52, s53, s14 // (op 3545) 
v_lshl_add_u32 v9, s53, 7, v10 // (op 3545) 


s_sub_u32 s53, s54, s52 // (op 3545) 


v_lshl_add_u32 v10, s53, 5, v8 // (op 3545) 


v_mul_lo_u32 v8, v10, s4 // (op 3545) 

v_add_lshl_u32 v10, v9, v8, 2 // (op 3545) 


s_mul_i32 s53, s4, s23 // (op 3547) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 3547) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 3547) least significant
s_mul_hi_u32 s55, s4, s22 // (op 3547) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3547) most significant: sum
s_add_u32 s53, s53, s55 // (op 3547) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 3547) 

s_mov_b32 s56, s20 // (op 3547) reg expression
s_mov_b32 s57, s21 // (op 3547) reg expression
s_mov_b32 s58, s54 // (op 3547) reg expression
s_mov_b32 s59, 131072 // (op 3547) call()

v_lshrrev_b32 v8, 7, v1 // (op 3647) 
s_mul_i32 s52, s16, s8 // (op 3647) 
s_add_u32 s52, s52, s0 // (op 3647) 
v_and_b32 v9, 63, v1 // (op 3647) 
s_mul_hi_u32 s53, s52, s13 // (op 3647) 
v_lshrrev_b32 v11, 4, v9 // (op 3647) 
s_sub_u32 s54, s52, s53 // (op 3647) 
v_lshlrev_b32 v12, 2, v11 // (op 3647) 

s_lshr_b32 s55, s54, 1 // (op 3647) 

v_lshl_add_u32 v11, v8, 4, v12 // (op 3647) 


s_add_u32 s54, s55, s53 // (op 3647) 


v_lshrrev_b32 v8, 2, v9 // (op 3647) 

s_lshr_b32 s53, s54, s11 // (op 3647) 

v_and_b32 v9, 3, v8 // (op 3647) 

s_and_b32 s54, s11, 2147483648 // (op 3647) 
v_and_b32 v8, 3, v1 // (op 3647) 
s_cmp_eq_i32 s54, 0 // (op 3647) 
s_mov_b32 s55, scc // (op 3647) 

v_lshl_add_u32 v12, v9, 2, v8 // (op 3647) 


s_cmp_eq_i32 s55, 0 // (op 3647) 
s_cselect_b32 s54, s52, s53 // (op 3647) 



v_lshrrev_b32 v8, 6, v1 // (op 3647) 
s_mul_hi_u32 s55, s54, s12 // (op 3647) 
v_and_b32 v9, 1, v8 // (op 3647) 

s_sub_u32 s52, s54, s55 // (op 3647) 
v_lshl_add_u32 v8, v9, 4, v12 // (op 3647) 


s_lshr_b32 s53, s52, 1 // (op 3647) 

s_add_u32 s52, s53, s55 // (op 3647) 


s_lshr_b32 s53, s52, s10 // (op 3647) 

s_and_b32 s52, s10, 2147483648 // (op 3647) 
s_cmp_eq_i32 s52, 0 // (op 3647) 
s_mov_b32 s55, scc // (op 3647) 

s_cmp_eq_i32 s55, 0 // (op 3647) 
s_cselect_b32 s52, s54, s53 // (op 3647) 


s_mul_i32 s53, s52, s14 // (op 3647) 
v_lshl_add_u32 v9, s52, 7, v11 // (op 3647) 


s_sub_u32 s52, s54, s53 // (op 3647) 


v_lshl_add_u32 v11, s52, 5, v8 // (op 3647) 


v_mul_lo_u32 v8, v11, s24 // (op 3647) 

v_add_lshl_u32 v11, v9, v8, 2 // (op 3647) 


s_mul_i32 s53, s24, s23 // (op 3649) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 3649) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 3649) least significant
s_mul_hi_u32 s55, s24, s22 // (op 3649) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3649) most significant: sum
s_add_u32 s53, s53, s55 // (op 3649) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 3649) 

s_mov_b32 s60, s26 // (op 3649) reg expression
s_mov_b32 s61, s27 // (op 3649) reg expression
s_mov_b32 s62, s54 // (op 3649) reg expression
s_mov_b32 s63, 131072 // (op 3649) call()

v_lshrrev_b32 v8, 5, v1 // (op 3747) 
v_lshlrev_b32 v9, 2, v8 // (op 3747) 

v_lshl_add_u32 v8, s16, 7, v9 // (op 3747) 

v_and_b32 v9, -32, v1 // (op 3747) 
v_sub_u32 v12, v1, v9 // (op 3747) 

v_lshl_add_u32 v9, v8, 5, v12 // (op 3747) 


v_lshlrev_b32 v8, 2, v9 // (op 3747) 

s_lshl_b32 s52, s15, 14 // (op 3749) 
s_mov_b32 s64, s28 // (op 3749) reg expression
s_mov_b32 s65, s29 // (op 3749) reg expression
s_mov_b32 s66, s52 // (op 3749) reg expression
s_mov_b32 s67, 131072 // (op 3749) call()

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
v_accvgpr_write a4, 0.00000 // (op 1233) call()
v_accvgpr_write a5, 0.00000 // (op 1233) call()
v_accvgpr_write a6, 0.00000 // (op 1233) call()
v_accvgpr_write a7, 0.00000 // (op 1233) call()
v_accvgpr_write a8, 0.00000 // (op 1355) call()
v_accvgpr_write a9, 0.00000 // (op 1355) call()
v_accvgpr_write a10, 0.00000 // (op 1355) call()
v_accvgpr_write a11, 0.00000 // (op 1355) call()
v_accvgpr_write a12, 0.00000 // (op 1477) call()
v_accvgpr_write a13, 0.00000 // (op 1477) call()
v_accvgpr_write a14, 0.00000 // (op 1477) call()
v_accvgpr_write a15, 0.00000 // (op 1477) call()
v_lshrrev_b32 v9, 7, v1 // (op 3112) 
s_mul_i32 s52, s16, s8 // (op 3112) 
s_add_u32 s52, s52, s0 // (op 3112) 
v_and_b32 v12, 15, v1 // (op 3112) 
s_mul_hi_u32 s53, s52, s13 // (op 3112) 
v_lshl_add_u32 v13, v9, 4, v12 // (op 3112) 


s_sub_u32 s54, s52, s53 // (op 3112) 
v_and_b32 v9, 63, v1 // (op 3112) 
s_lshr_b32 s55, s54, 1 // (op 3112) 

v_lshrrev_b32 v12, 4, v9 // (op 3112) 

s_add_u32 s54, s55, s53 // (op 3112) 


v_lshlrev_b32 v9, 5, v12 // (op 3112) 

s_lshr_b32 s53, s54, s11 // (op 3112) 

s_and_b32 s54, s11, 2147483648 // (op 3112) 
s_cmp_eq_i32 s54, 0 // (op 3112) 
s_mov_b32 s55, scc // (op 3112) 

s_cmp_eq_i32 s55, 0 // (op 3112) 
s_cselect_b32 s54, s52, s53 // (op 3112) 


s_mul_hi_u32 s53, s54, s12 // (op 3112) 
s_sub_u32 s55, s54, s53 // (op 3112) 
s_lshr_b32 s68, s55, 1 // (op 3112) 

s_add_u32 s55, s68, s53 // (op 3112) 


s_lshr_b32 s53, s55, s10 // (op 3112) 

s_and_b32 s55, s10, 2147483648 // (op 3112) 
s_cmp_eq_i32 s55, 0 // (op 3112) 
s_mov_b32 s68, scc // (op 3112) 

s_cmp_eq_i32 s68, 0 // (op 3112) 
s_cselect_b32 s55, s54, s53 // (op 3112) 


v_lshl_add_u32 v12, s55, 7, v13 // (op 3112) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 3112) 
s_mul_i32 s53, s54, s68 // (op 3112) 


s_sub_u32 s54, s52, s53 // (op 3112) 


v_lshl_add_u32 v13, s54, 7, v9 // (op 3112) 


v_mov_b32 v15, 0 // (op 3112) convert
v_mov_b32 v14, v13 // (op 3112) convert

v_mul_lo_u32 v17, v12, s31 // (op 3112) most significant: low of low * high
v_mul_hi_u32 v9, v12, s30 // (op 3112) most significant: high of low * low
v_mul_lo_u32 v16, v12, s30 // (op 3112) least significant: low of low * low
v_add_u32 v17, v17, v9 // (op 3112) most significant: sum

v_add_co_u32 v16, s[52:53], v16, v14 // (op 3112) least significant half
v_addc_co_u32 v17, s[52:53], v17, v15, s[52:53] // (op 3112) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 3112) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 3112) 

v_mov_b32 v9, v14 // (op 3112) convert

s_mul_i32 s53, s30, s35 // (op 3114) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 3114) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 3114) least significant
s_mul_hi_u32 s55, s30, s34 // (op 3114) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3114) most significant: sum
s_add_u32 s53, s53, s55 // (op 3114) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 3114) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 3114) 

s_mov_b32 s68, s32 // (op 3114) reg expression
s_mov_b32 s69, s33 // (op 3114) reg expression
s_mov_b32 s70, s52 // (op 3114) reg expression
s_mov_b32 s71, 131072 // (op 3114) call()

v_lshrrev_b32 v12, 7, v1 // (op 3133) 
s_mul_i32 s52, s16, s8 // (op 3133) 
s_add_u32 s52, s52, s0 // (op 3133) 
v_and_b32 v13, 15, v1 // (op 3133) 
s_mul_hi_u32 s53, s52, s13 // (op 3133) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 3133) 


s_sub_u32 s54, s52, s53 // (op 3133) 
v_and_b32 v12, 63, v1 // (op 3133) 
s_lshr_b32 s55, s54, 1 // (op 3133) 

v_lshrrev_b32 v13, 4, v12 // (op 3133) 

s_add_u32 s54, s55, s53 // (op 3133) 


v_lshlrev_b32 v12, 5, v13 // (op 3133) 

s_lshr_b32 s53, s54, s11 // (op 3133) 

s_and_b32 s54, s11, 2147483648 // (op 3133) 
s_cmp_eq_i32 s54, 0 // (op 3133) 
s_mov_b32 s55, scc // (op 3133) 

s_cmp_eq_i32 s55, 0 // (op 3133) 
s_cselect_b32 s54, s52, s53 // (op 3133) 


s_mul_hi_u32 s53, s54, s12 // (op 3133) 
s_sub_u32 s55, s54, s53 // (op 3133) 
s_lshr_b32 s72, s55, 1 // (op 3133) 

s_add_u32 s55, s72, s53 // (op 3133) 


s_lshr_b32 s53, s55, s10 // (op 3133) 

s_and_b32 s55, s10, 2147483648 // (op 3133) 
s_cmp_eq_i32 s55, 0 // (op 3133) 
s_mov_b32 s72, scc // (op 3133) 

s_cmp_eq_i32 s72, 0 // (op 3133) 
s_cselect_b32 s55, s54, s53 // (op 3133) 


v_lshl_add_u32 v13, s55, 7, v14 // (op 3133) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 3133) 
s_mul_i32 s53, s54, s72 // (op 3133) 


s_sub_u32 s54, s52, s53 // (op 3133) 


v_lshl_add_u32 v14, s54, 7, v12 // (op 3133) 


v_mov_b32 v17, 0 // (op 3133) convert
v_mov_b32 v16, v14 // (op 3133) convert

v_mul_lo_u32 v15, v13, s31 // (op 3133) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 3133) most significant: high of low * low
v_mul_lo_u32 v14, v13, s30 // (op 3133) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 3133) most significant: sum

v_add_co_u32 v14, s[52:53], v14, v16 // (op 3133) least significant half
v_addc_co_u32 v15, s[52:53], v15, v17, s[52:53] // (op 3133) most significant half



v_lshlrev_b64 v[12:13], 2, v[14:15] // (op 3133) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 3133) 

v_mov_b32 v12, v14 // (op 3133) convert

s_mul_i32 s53, s30, s35 // (op 3135) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 3135) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 3135) least significant
s_mul_hi_u32 s55, s30, s34 // (op 3135) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3135) most significant: sum
s_add_u32 s53, s53, s55 // (op 3135) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 3135) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 3135) 

s_mov_b32 s68, s32 // (op 3135) reg expression
s_mov_b32 s69, s33 // (op 3135) reg expression
s_mov_b32 s70, s52 // (op 3135) reg expression
s_mov_b32 s71, 131072 // (op 3135) call()

v_lshrrev_b32 v13, 6, v1 // (op 3245) 
s_mul_i32 s52, s16, s8 // (op 3245) 
s_add_u32 s52, s52, s0 // (op 3245) 
v_and_b32 v14, 1, v13 // (op 3245) 

s_mul_hi_u32 s53, s52, s13 // (op 3245) 
v_and_b32 v13, 15, v1 // (op 3245) 
s_sub_u32 s54, s52, s53 // (op 3245) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 3245) 


s_lshr_b32 s55, s54, 1 // (op 3245) 

v_and_b32 v13, 48, v1 // (op 3245) 
s_add_u32 s54, s55, s53 // (op 3245) 


s_lshr_b32 s53, s54, s11 // (op 3245) 

s_and_b32 s54, s11, 2147483648 // (op 3245) 
s_cmp_eq_i32 s54, 0 // (op 3245) 
s_mov_b32 s55, scc // (op 3245) 

s_cmp_eq_i32 s55, 0 // (op 3245) 
s_cselect_b32 s54, s52, s53 // (op 3245) 


s_mul_hi_u32 s53, s54, s12 // (op 3245) 
s_sub_u32 s55, s54, s53 // (op 3245) 
s_lshr_b32 s72, s55, 1 // (op 3245) 

s_add_u32 s55, s72, s53 // (op 3245) 


s_lshr_b32 s53, s55, s10 // (op 3245) 

s_and_b32 s55, s10, 2147483648 // (op 3245) 
s_cmp_eq_i32 s55, 0 // (op 3245) 
s_mov_b32 s72, scc // (op 3245) 

s_cmp_eq_i32 s72, 0 // (op 3245) 
s_cselect_b32 s55, s54, s53 // (op 3245) 


s_mul_i32 s53, s55, s14 // (op 3245) 

s_sub_u32 s55, s54, s53 // (op 3245) 

v_lshl_add_u32 v14, s55, 5, v15 // (op 3245) 


v_mul_lo_u32 v15, v14, s36 // (op 3245) 

s_ashr_i64 s[72:73], s[2:3], 7 // (op 3245) 
s_mul_i32 s53, s54, s72 // (op 3245) 


s_sub_u32 s54, s52, s53 // (op 3245) 


v_lshl_add_u32 v14, s54, 7, v13 // (op 3245) 


v_add_u32 v13, v14, v15 // (op 3245) 


s_mul_i32 s53, s36, s23 // (op 3247) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 3247) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 3247) least significant
s_mul_hi_u32 s55, s36, s22 // (op 3247) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3247) most significant: sum
s_add_u32 s53, s53, s55 // (op 3247) most significant: sum


s_mov_b32 s72, s38 // (op 3247) reg expression
s_mov_b32 s73, s39 // (op 3247) reg expression
s_mov_b32 s74, s52 // (op 3247) reg expression
s_mov_b32 s75, 131072 // (op 3247) call()

v_lshrrev_b32 v14, 6, v1 // (op 3264) 
s_mul_i32 s52, s16, s8 // (op 3264) 
s_add_u32 s52, s52, s0 // (op 3264) 
v_and_b32 v15, 1, v14 // (op 3264) 

s_mul_hi_u32 s53, s52, s13 // (op 3264) 
v_and_b32 v14, 15, v1 // (op 3264) 
s_sub_u32 s54, s52, s53 // (op 3264) 
v_lshl_add_u32 v16, v15, 4, v14 // (op 3264) 


s_lshr_b32 s55, s54, 1 // (op 3264) 

v_and_b32 v14, 48, v1 // (op 3264) 
s_add_u32 s54, s55, s53 // (op 3264) 


s_lshr_b32 s53, s54, s11 // (op 3264) 

s_and_b32 s54, s11, 2147483648 // (op 3264) 
s_cmp_eq_i32 s54, 0 // (op 3264) 
s_mov_b32 s55, scc // (op 3264) 

s_cmp_eq_i32 s55, 0 // (op 3264) 
s_cselect_b32 s54, s52, s53 // (op 3264) 


s_mul_hi_u32 s53, s54, s12 // (op 3264) 
s_sub_u32 s55, s54, s53 // (op 3264) 
s_lshr_b32 s76, s55, 1 // (op 3264) 

s_add_u32 s55, s76, s53 // (op 3264) 


s_lshr_b32 s53, s55, s10 // (op 3264) 

s_and_b32 s55, s10, 2147483648 // (op 3264) 
s_cmp_eq_i32 s55, 0 // (op 3264) 
s_mov_b32 s76, scc // (op 3264) 

s_cmp_eq_i32 s76, 0 // (op 3264) 
s_cselect_b32 s55, s54, s53 // (op 3264) 


s_mul_i32 s53, s55, s14 // (op 3264) 

s_sub_u32 s55, s54, s53 // (op 3264) 

v_lshl_add_u32 v15, s55, 5, v16 // (op 3264) 


v_mul_lo_u32 v16, v15, s36 // (op 3264) 

s_ashr_i64 s[76:77], s[2:3], 7 // (op 3264) 
s_mul_i32 s53, s54, s76 // (op 3264) 


s_sub_u32 s54, s52, s53 // (op 3264) 


v_lshl_add_u32 v15, s54, 7, v14 // (op 3264) 


v_add_u32 v14, v15, v16 // (op 3264) 


s_mul_i32 s53, s36, s23 // (op 3266) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 3266) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 3266) least significant
s_mul_hi_u32 s55, s36, s22 // (op 3266) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3266) most significant: sum
s_add_u32 s53, s53, s55 // (op 3266) most significant: sum


s_mov_b32 s72, s38 // (op 3266) reg expression
s_mov_b32 s73, s39 // (op 3266) reg expression
s_mov_b32 s74, s52 // (op 3266) reg expression
s_mov_b32 s75, 131072 // (op 3266) call()

v_lshrrev_b32 v15, 2, v1 // (op 3373) 
s_mul_i32 s52, s16, s8 // (op 3373) 
s_add_u32 s52, s52, s0 // (op 3373) 
v_lshlrev_b32 v16, 1, v15 // (op 3373) 

s_mul_hi_u32 s53, s52, s13 // (op 3373) 
v_and_b32 v15, -4, v1 // (op 3373) 
s_sub_u32 s54, s52, s53 // (op 3373) 
v_sub_u32 v17, v1, v15 // (op 3373) 

s_lshr_b32 s55, s54, 1 // (op 3373) 

s_add_u32 s54, s55, s53 // (op 3373) 


s_lshr_b32 s53, s54, s11 // (op 3373) 

s_and_b32 s54, s11, 2147483648 // (op 3373) 
s_cmp_eq_i32 s54, 0 // (op 3373) 
s_mov_b32 s55, scc // (op 3373) 

s_cmp_eq_i32 s55, 0 // (op 3373) 
s_cselect_b32 s54, s52, s53 // (op 3373) 


s_mul_hi_u32 s53, s54, s12 // (op 3373) 
s_sub_u32 s55, s54, s53 // (op 3373) 
s_lshr_b32 s76, s55, 1 // (op 3373) 

s_add_u32 s55, s76, s53 // (op 3373) 


s_lshr_b32 s53, s55, s10 // (op 3373) 

s_and_b32 s55, s10, 2147483648 // (op 3373) 
s_cmp_eq_i32 s55, 0 // (op 3373) 
s_mov_b32 s76, scc // (op 3373) 

s_cmp_eq_i32 s76, 0 // (op 3373) 
s_cselect_b32 s55, s54, s53 // (op 3373) 


v_lshl_add_u32 v15, s55, 7, v16 // (op 3373) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 3373) 
s_mul_i32 s53, s54, s76 // (op 3373) 


s_sub_u32 s54, s52, s53 // (op 3373) 


v_lshl_add_u32 v16, s54, 2, v17 // (op 3373) 


v_mul_lo_u32 v17, v15, s40 // (op 3373) 
v_add_u32 v17, v17, v16 // (op 3373) 


s_mul_i32 s53, s40, s35 // (op 3375) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 3375) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 3375) least significant
s_mul_hi_u32 s55, s40, s34 // (op 3375) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3375) most significant: sum
s_add_u32 s53, s53, s55 // (op 3375) most significant: sum


s_mov_b32 s76, s42 // (op 3375) reg expression
s_mov_b32 s77, s43 // (op 3375) reg expression
s_mov_b32 s78, s52 // (op 3375) reg expression
s_mov_b32 s79, 131072 // (op 3375) call()

v_lshrrev_b32 v15, 2, v1 // (op 3392) 
s_mul_i32 s52, s16, s8 // (op 3392) 
s_add_u32 s52, s52, s0 // (op 3392) 
v_lshlrev_b32 v16, 1, v15 // (op 3392) 

s_mul_hi_u32 s53, s52, s13 // (op 3392) 
v_and_b32 v15, -4, v1 // (op 3392) 
s_sub_u32 s54, s52, s53 // (op 3392) 
v_sub_u32 v18, v1, v15 // (op 3392) 

s_lshr_b32 s55, s54, 1 // (op 3392) 

s_add_u32 s54, s55, s53 // (op 3392) 


s_lshr_b32 s53, s54, s11 // (op 3392) 

s_and_b32 s54, s11, 2147483648 // (op 3392) 
s_cmp_eq_i32 s54, 0 // (op 3392) 
s_mov_b32 s55, scc // (op 3392) 

s_cmp_eq_i32 s55, 0 // (op 3392) 
s_cselect_b32 s54, s52, s53 // (op 3392) 


s_mul_hi_u32 s53, s54, s12 // (op 3392) 
s_sub_u32 s55, s54, s53 // (op 3392) 
s_lshr_b32 s80, s55, 1 // (op 3392) 

s_add_u32 s55, s80, s53 // (op 3392) 


s_lshr_b32 s53, s55, s10 // (op 3392) 

s_and_b32 s55, s10, 2147483648 // (op 3392) 
s_cmp_eq_i32 s55, 0 // (op 3392) 
s_mov_b32 s80, scc // (op 3392) 

s_cmp_eq_i32 s80, 0 // (op 3392) 
s_cselect_b32 s55, s54, s53 // (op 3392) 


v_lshl_add_u32 v15, s55, 7, v16 // (op 3392) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 3392) 
s_mul_i32 s53, s54, s80 // (op 3392) 


s_sub_u32 s54, s52, s53 // (op 3392) 


v_lshl_add_u32 v16, s54, 2, v18 // (op 3392) 


v_mul_lo_u32 v18, v15, s40 // (op 3392) 
v_add_u32 v18, v18, v16 // (op 3392) 


s_mul_i32 s53, s40, s35 // (op 3394) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 3394) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 3394) least significant
s_mul_hi_u32 s55, s40, s34 // (op 3394) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3394) most significant: sum
s_add_u32 s53, s53, s55 // (op 3394) most significant: sum


s_mov_b32 s76, s42 // (op 3394) reg expression
s_mov_b32 s77, s43 // (op 3394) reg expression
s_mov_b32 s78, s52 // (op 3394) reg expression
s_mov_b32 s79, 131072 // (op 3394) call()

v_lshrrev_b32 v15, 6, v1 // (op 3501) 
s_mul_i32 s52, s16, s8 // (op 3501) 
s_add_u32 s52, s52, s0 // (op 3501) 
v_and_b32 v16, 1, v15 // (op 3501) 

s_mul_hi_u32 s53, s52, s13 // (op 3501) 
v_and_b32 v15, 15, v1 // (op 3501) 
s_sub_u32 s54, s52, s53 // (op 3501) 
v_lshl_add_u32 v19, v16, 4, v15 // (op 3501) 


s_lshr_b32 s55, s54, 1 // (op 3501) 

v_and_b32 v15, 63, v1 // (op 3501) 
s_add_u32 s54, s55, s53 // (op 3501) 


v_lshrrev_b32 v16, 4, v15 // (op 3501) 

s_lshr_b32 s53, s54, s11 // (op 3501) 

s_and_b32 s54, s11, 2147483648 // (op 3501) 
s_cmp_eq_i32 s54, 0 // (op 3501) 
s_mov_b32 s55, scc // (op 3501) 

s_cmp_eq_i32 s55, 0 // (op 3501) 
s_cselect_b32 s54, s52, s53 // (op 3501) 


s_mul_hi_u32 s53, s54, s12 // (op 3501) 
s_sub_u32 s55, s54, s53 // (op 3501) 
s_lshr_b32 s80, s55, 1 // (op 3501) 

s_add_u32 s55, s80, s53 // (op 3501) 


s_lshr_b32 s53, s55, s10 // (op 3501) 

s_and_b32 s55, s10, 2147483648 // (op 3501) 
s_cmp_eq_i32 s55, 0 // (op 3501) 
s_mov_b32 s80, scc // (op 3501) 

s_cmp_eq_i32 s80, 0 // (op 3501) 
s_cselect_b32 s55, s54, s53 // (op 3501) 


s_mul_i32 s53, s55, s14 // (op 3501) 

s_sub_u32 s55, s54, s53 // (op 3501) 

v_lshl_add_u32 v15, s55, 5, v19 // (op 3501) 


v_mul_lo_u32 v19, v15, s44 // (op 3501) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 3501) 
s_mul_i32 s53, s54, s80 // (op 3501) 


s_sub_u32 s54, s52, s53 // (op 3501) 


v_lshl_add_u32 v15, s54, 2, v16 // (op 3501) 


v_add_u32 v16, v15, v19 // (op 3501) 


s_mul_i32 s53, s44, s23 // (op 3503) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 3503) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 3503) least significant
s_mul_hi_u32 s55, s44, s22 // (op 3503) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3503) most significant: sum
s_add_u32 s53, s53, s55 // (op 3503) most significant: sum


s_mov_b32 s80, s46 // (op 3503) reg expression
s_mov_b32 s81, s47 // (op 3503) reg expression
s_mov_b32 s82, s52 // (op 3503) reg expression
s_mov_b32 s83, 131072 // (op 3503) call()

v_lshrrev_b32 v15, 6, v1 // (op 3520) 
s_mul_i32 s52, s16, s8 // (op 3520) 
s_add_u32 s52, s52, s0 // (op 3520) 
v_and_b32 v19, 1, v15 // (op 3520) 

s_mul_hi_u32 s53, s52, s13 // (op 3520) 
v_and_b32 v15, 15, v1 // (op 3520) 
s_sub_u32 s54, s52, s53 // (op 3520) 
v_lshl_add_u32 v20, v19, 4, v15 // (op 3520) 


s_lshr_b32 s55, s54, 1 // (op 3520) 

v_and_b32 v15, 63, v1 // (op 3520) 
s_add_u32 s54, s55, s53 // (op 3520) 


v_lshrrev_b32 v19, 4, v15 // (op 3520) 

s_lshr_b32 s53, s54, s11 // (op 3520) 

s_and_b32 s54, s11, 2147483648 // (op 3520) 
s_cmp_eq_i32 s54, 0 // (op 3520) 
s_mov_b32 s55, scc // (op 3520) 

s_cmp_eq_i32 s55, 0 // (op 3520) 
s_cselect_b32 s54, s52, s53 // (op 3520) 


s_mul_hi_u32 s53, s54, s12 // (op 3520) 
s_sub_u32 s55, s54, s53 // (op 3520) 
s_lshr_b32 s84, s55, 1 // (op 3520) 

s_add_u32 s55, s84, s53 // (op 3520) 


s_lshr_b32 s53, s55, s10 // (op 3520) 

s_and_b32 s55, s10, 2147483648 // (op 3520) 
s_cmp_eq_i32 s55, 0 // (op 3520) 
s_mov_b32 s84, scc // (op 3520) 

s_cmp_eq_i32 s84, 0 // (op 3520) 
s_cselect_b32 s55, s54, s53 // (op 3520) 


s_mul_i32 s53, s55, s14 // (op 3520) 

s_sub_u32 s55, s54, s53 // (op 3520) 

v_lshl_add_u32 v15, s55, 5, v20 // (op 3520) 


v_mul_lo_u32 v20, v15, s44 // (op 3520) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 3520) 
s_mul_i32 s53, s54, s84 // (op 3520) 


s_sub_u32 s54, s52, s53 // (op 3520) 


v_lshl_add_u32 v15, s54, 2, v19 // (op 3520) 


v_add_u32 v19, v15, v20 // (op 3520) 


s_mul_i32 s53, s44, s23 // (op 3522) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 3522) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 3522) least significant
s_mul_hi_u32 s55, s44, s22 // (op 3522) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3522) most significant: sum
s_add_u32 s53, s53, s55 // (op 3522) most significant: sum


s_mov_b32 s80, s46 // (op 3522) reg expression
s_mov_b32 s81, s47 // (op 3522) reg expression
s_mov_b32 s82, s52 // (op 3522) reg expression
s_mov_b32 s83, 131072 // (op 3522) call()



v_mov_b32 v15, v9 // (op 2817) reg expression

buffer_load_dwordx4 v[20:23], v15, s[68:71], 0 offen // (op 2817) Load value




buffer_load_dwordx4 v[24:27], v13, s[72:75], 0 offen // (op 2823) Load value
buffer_load_dwordx4 v[28:31], v13, s[72:75], 0 offen offset:64 // (op 2823) Load value



buffer_load_ubyte v15, v16, s[80:83], 0 offen offset: 0 // (op 2828) Load value


s_lshl_b64 s[52:53], s[30:31], 5 // (op 2834) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2834) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2834) 

v_add_u32 v32, v9, s52 // (op 2834) 


buffer_load_dwordx4 v[34:37], v32, s[68:71], 0 offen // (op 2834) Load value



s_lshl_b64 s[52:53], s[30:31], 5 // (op 2841) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2841) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2841) 

s_lshl_b32 s54, s52, 1 // (op 2841) 

v_add_u32 v32, v9, s54 // (op 2841) 


buffer_load_dwordx4 v[38:41], v32, s[68:71], 0 offen // (op 2841) Load value



s_lshl_b64 s[52:53], s[30:31], 5 // (op 2848) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2848) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2848) 

s_mul_i32 s54, 3, s52 // (op 2848) 

v_add_u32 v32, v9, s54 // (op 2848) 


buffer_load_dwordx4 v[42:45], v32, s[68:71], 0 offen // (op 2848) Load value


v_mov_b64 v[32:33], s[40:41] // (op 2853) convert

v_mov_b32 v46, v17 // (op 2853) 
buffer_load_ubyte v48, v46, s[76:79], 0 offen offset: 0 // (op 2853) Load value
v_add_u32 v46, v46, v32 // (op 2853) 
buffer_load_ubyte v49, v46, s[76:79], 0 offen offset: 0 // (op 2853) Load value





buffer_load_ubyte v32, v16, s[80:83], 0 offen offset:4 // (op 2857) Load value



v_add_u32 v16, 64, v9 // (op 2863) 

buffer_load_dwordx4 v[50:53], v16, s[68:71], 0 offen // (op 2863) Load value




buffer_load_dwordx4 v[54:57], v13, s[72:75], 0 offen offset:128 // (op 2869) Load value
buffer_load_dwordx4 v[58:61], v13, s[72:75], 0 offen offset:192 // (op 2869) Load value



s_lshl_b64 s[52:53], s[30:31], 5 // (op 2875) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2875) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2875) 

s_add_u32 s54, s52, 64 // (op 2875) 

v_add_u32 v13, v9, s54 // (op 2875) 


buffer_load_dwordx4 v[62:65], v13, s[68:71], 0 offen // (op 2875) Load value



s_lshl_b64 s[52:53], s[30:31], 5 // (op 2882) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2882) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2882) 

s_lshl_b32 s54, s52, 1 // (op 2882) 
s_add_u32 s54, s54, 64 // (op 2882) 

v_add_u32 v13, v9, s54 // (op 2882) 


buffer_load_dwordx4 v[66:69], v13, s[68:71], 0 offen // (op 2882) Load value



s_lshl_b64 s[52:53], s[30:31], 5 // (op 2889) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2889) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2889) 

s_mul_i32 s54, 3, s52 // (op 2889) 
s_add_u32 s54, s54, 64 // (op 2889) 

v_add_u32 v13, v9, s54 // (op 2889) 


buffer_load_dwordx4 v[70:73], v13, s[68:71], 0 offen // (op 2889) Load value



v_mov_b64 v[46:47], s[40:41] // (op 2894) convert

v_mov_b32 v9, v17 // (op 2894) 
buffer_load_ubyte v74, v9, s[76:79], 0 offen offset:4 // (op 2894) Load value
v_add_u32 v9, v9, v46 // (op 2894) 
buffer_load_ubyte v75, v9, s[76:79], 0 offen offset:4 // (op 2894) Load value



v_and_b32 v9, -4, v1 // (op 3613) 
v_sub_u32 v13, v1, v9 // (op 3613) 

v_lshrrev_b32 v9, 2, v1 // (op 3613) 
v_lshl_add_u32 v33, v9, 3, v13 // (op 3613) 


s_barrier  // (op 3925) 



s_waitcnt vmcnt(10)
ds_write_b8 v33, v48 // (op 2896) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v33, v49 offset:4 // (op 2896) Store local data 

v_and_b32 v9, -4, v1 // (op 3584) 
v_sub_u32 v13, v1, v9 // (op 3584) 

v_lshrrev_b32 v9, 2, v1 // (op 3584) 
v_lshl_add_u32 v33, v9, 3, v13 // (op 3584) 


v_and_b32 v9, -4, v1 // (op 3813) 
v_sub_u32 v13, v1, v9 // (op 3813) 

v_lshrrev_b32 v9, 2, v1 // (op 3813) 
v_lshl_add_u32 v16, v9, 3, v13 // (op 3813) 


s_mov_b32 s1, 0 // (op 1479) call()
s_and_b32 s52, s7, 4294967294 // (op 1478) 
s_cmp_lt_u32 s1, s52 // (op 1478) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_1478 // (op 1478) Condition: Top (jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_1478 if false)
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_1478:
s_add_u32 s1, s1, 2 // (op 1480) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3922) 



ds_read_u8 v9, v0 // (op 63) Load local data 



ds_read_u8 v13, v0 offset:128 // (op 1260) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:31], a[0:3], v9, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 70) 




ds_read_u8 v9, v0 offset:256 // (op 1382) Load local data 



ds_read_u8 v17, v0 offset:384 // (op 1504) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[24:31], a[4:7], v13, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1262) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[38:41], v[24:31], a[8:11], v9, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1384) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:45], v[24:31], a[12:15], v17, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1506) 



v_mov_b32 v9, v12 // (op 55) 
v_add_u32 v13, 128, v9 // (op 55) 

buffer_load_dwordx4 v[20:23], v13, s[68:71], 0 offen // (op 55) Load value



v_mov_b32 v13, v14 // (op 59) 

buffer_load_dwordx4 v[24:27], v13, s[72:75], 0 offen offset:256 // (op 59) Load value
buffer_load_dwordx4 v[28:31], v13, s[72:75], 0 offen offset:320 // (op 59) Load value


v_mov_b32 v17, v19 // (op 68) 

buffer_load_ubyte v15, v17, s[80:83], 0 offen offset:8 // (op 68) Load value


s_lshl_b64 s[52:53], s[30:31], 5 // (op 1285) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1285) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1285) 

s_add_u32 s54, s52, 128 // (op 1285) 

v_add_u32 v46, v9, s54 // (op 1285) 


buffer_load_dwordx4 v[34:37], v46, s[68:71], 0 offen // (op 1285) Load value



s_lshl_b64 s[52:53], s[30:31], 5 // (op 1407) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1407) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1407) 

s_lshl_b32 s54, s52, 1 // (op 1407) 
s_add_u32 s54, s54, 128 // (op 1407) 

v_add_u32 v46, v9, s54 // (op 1407) 


buffer_load_dwordx4 v[38:41], v46, s[68:71], 0 offen // (op 1407) Load value



s_lshl_b64 s[52:53], s[30:31], 5 // (op 1529) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1529) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1529) 

s_mul_i32 s54, 3, s52 // (op 1529) 
s_add_u32 s54, s54, 128 // (op 1529) 

v_add_u32 v46, v9, s54 // (op 1529) 


buffer_load_dwordx4 v[42:45], v46, s[68:71], 0 offen // (op 1529) Load value


v_mov_b64 v[46:47], s[40:41] // (op 9) convert

v_mov_b32 v76, v18 // (op 9) 
v_mov_b32 v77, v76 // (op 9) 
buffer_load_ubyte v48, v77, s[76:79], 0 offen offset:8 // (op 9) Load value
v_add_u32 v77, v77, v46 // (op 9) 
buffer_load_ubyte v49, v77, s[76:79], 0 offen offset:8 // (op 9) Load value


s_barrier  // (op 3963) 



s_waitcnt vmcnt(10)
ds_write_b8 v16, v74 offset:512 // (op 1173) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v16, v75 offset:516 // (op 1173) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3960) 



ds_read_u8 v46, v5 offset:512 // (op 1175) Load local data 



ds_read_u8 v47, v5 offset:640 // (op 1269) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:53], v[54:61], a[0:3], v46, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1178) 




ds_read_u8 v46, v5 offset:768 // (op 1391) Load local data 



ds_read_u8 v77, v5 offset:896 // (op 1513) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[54:61], a[4:7], v47, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1263) 

s_barrier  // (op 4148) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[54:61], a[8:11], v46, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1385) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:73], v[54:61], a[12:15], v77, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1507) 




buffer_load_ubyte v32, v17, s[80:83], 0 offen offset:12 // (op 1177) Load value



v_add_u32 v19, 8, v19 // (op 3512) 
v_add_u32 v17, 192, v9 // (op 1180) 

buffer_load_dwordx4 v[50:53], v17, s[68:71], 0 offen // (op 1180) Load value




buffer_load_dwordx4 v[54:57], v13, s[72:75], 0 offen offset:384 // (op 1182) Load value
buffer_load_dwordx4 v[58:61], v13, s[72:75], 0 offen offset:448 // (op 1182) Load value



v_add_u32 v14, 256, v14 // (op 3256) 
s_lshl_b64 s[52:53], s[30:31], 5 // (op 1278) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1278) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1278) 

s_add_u32 s54, s52, 192 // (op 1278) 

v_add_u32 v13, v9, s54 // (op 1278) 


buffer_load_dwordx4 v[62:65], v13, s[68:71], 0 offen // (op 1278) Load value



s_lshl_b64 s[52:53], s[30:31], 5 // (op 1400) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1400) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1400) 

s_lshl_b32 s54, s52, 1 // (op 1400) 
s_add_u32 s54, s54, 192 // (op 1400) 

v_add_u32 v13, v9, s54 // (op 1400) 


buffer_load_dwordx4 v[66:69], v13, s[68:71], 0 offen // (op 1400) Load value



s_lshl_b64 s[52:53], s[30:31], 5 // (op 1522) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1522) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1522) 

s_mul_i32 s54, 3, s52 // (op 1522) 
s_add_u32 s54, s54, 192 // (op 1522) 

v_add_u32 v13, v9, s54 // (op 1522) 


buffer_load_dwordx4 v[70:73], v13, s[68:71], 0 offen // (op 1522) Load value



v_add_u32 v12, 128, v12 // (op 3123) 
v_mov_b64 v[46:47], s[40:41] // (op 1172) convert

v_mov_b32 v9, v76 // (op 1172) 
buffer_load_ubyte v74, v9, s[76:79], 0 offen offset:12 // (op 1172) Load value
v_add_u32 v9, v9, v46 // (op 1172) 
buffer_load_ubyte v75, v9, s[76:79], 0 offen offset:12 // (op 1172) Load value



s_barrier  // (op 3928) 
v_add_u32 v18, 8, v18 // (op 3384) 



s_waitcnt vmcnt(10)
ds_write_b8 v33, v48 // (op 13) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v33, v49 offset:4 // (op 13) Store local data 
s_and_b32 s52, s7, 4294967294 // (op 1478) 
s_cmp_lt_u32 s1, s52 // (op 1478) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_1478 // (op 1478) Condition: Bottom (jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_1478 if true)
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_1478:













s_and_b32 s52, s7, 4294967294 // (op 3017) 
v_lshrrev_b32 v9, 7, v1 // (op 3017) 
s_add_u32 s53, s0, s52 // (op 3017) 

v_and_b32 v12, 15, v1 // (op 3017) 
v_lshl_add_u32 v13, v9, 4, v12 // (op 3017) 


s_mul_i32 s52, s16, s8 // (op 3017) 
s_add_u32 s52, s52, s53 // (op 3017) 

v_and_b32 v9, 63, v1 // (op 3017) 
s_mul_hi_u32 s53, s52, s13 // (op 3017) 
v_lshrrev_b32 v12, 4, v9 // (op 3017) 

s_sub_u32 s54, s52, s53 // (op 3017) 
v_lshlrev_b32 v9, 5, v12 // (op 3017) 

s_lshr_b32 s55, s54, 1 // (op 3017) 

s_add_u32 s54, s55, s53 // (op 3017) 


s_lshr_b32 s53, s54, s11 // (op 3017) 

s_and_b32 s54, s11, 2147483648 // (op 3017) 
s_cmp_eq_i32 s54, 0 // (op 3017) 
s_mov_b32 s55, scc // (op 3017) 

s_cmp_eq_i32 s55, 0 // (op 3017) 
s_cselect_b32 s54, s52, s53 // (op 3017) 


s_mul_hi_u32 s53, s54, s12 // (op 3017) 
s_sub_u32 s55, s54, s53 // (op 3017) 
s_lshr_b32 s84, s55, 1 // (op 3017) 

s_add_u32 s55, s84, s53 // (op 3017) 


s_lshr_b32 s53, s55, s10 // (op 3017) 

s_and_b32 s55, s10, 2147483648 // (op 3017) 
s_cmp_eq_i32 s55, 0 // (op 3017) 
s_mov_b32 s84, scc // (op 3017) 

s_cmp_eq_i32 s84, 0 // (op 3017) 
s_cselect_b32 s55, s54, s53 // (op 3017) 


v_lshl_add_u32 v12, s55, 7, v13 // (op 3017) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 3017) 
s_mul_i32 s53, s54, s84 // (op 3017) 


s_sub_u32 s54, s52, s53 // (op 3017) 


v_lshl_add_u32 v13, s54, 7, v9 // (op 3017) 


v_mov_b32 v15, 0 // (op 3017) convert
v_mov_b32 v14, v13 // (op 3017) convert

v_mul_lo_u32 v17, v12, s31 // (op 3017) most significant: low of low * high
v_mul_hi_u32 v9, v12, s30 // (op 3017) most significant: high of low * low
v_mul_lo_u32 v16, v12, s30 // (op 3017) least significant: low of low * low
v_add_u32 v17, v17, v9 // (op 3017) most significant: sum

v_add_co_u32 v16, s[52:53], v16, v14 // (op 3017) least significant half
v_addc_co_u32 v17, s[52:53], v17, v15, s[52:53] // (op 3017) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 3017) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 3017) 

v_mov_b32 v9, v14 // (op 3017) convert

s_mul_i32 s53, s30, s35 // (op 3019) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 3019) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 3019) least significant
s_mul_hi_u32 s55, s30, s34 // (op 3019) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3019) most significant: sum
s_add_u32 s53, s53, s55 // (op 3019) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 3019) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 3019) 

s_mov_b32 s68, s32 // (op 3019) reg expression
s_mov_b32 s69, s33 // (op 3019) reg expression
s_mov_b32 s70, s52 // (op 3019) reg expression
s_mov_b32 s71, 131072 // (op 3019) call()

s_and_b32 s52, s7, 4294967294 // (op 3180) 
v_lshrrev_b32 v12, 6, v1 // (op 3180) 
s_add_u32 s53, s0, s52 // (op 3180) 

v_and_b32 v13, 1, v12 // (op 3180) 

v_and_b32 v12, 15, v1 // (op 3180) 
s_mul_i32 s52, s16, s8 // (op 3180) 
s_add_u32 s52, s52, s53 // (op 3180) 

v_lshl_add_u32 v14, v13, 4, v12 // (op 3180) 


s_mul_hi_u32 s53, s52, s13 // (op 3180) 
v_and_b32 v12, 48, v1 // (op 3180) 
s_sub_u32 s54, s52, s53 // (op 3180) 
s_lshr_b32 s55, s54, 1 // (op 3180) 

s_add_u32 s54, s55, s53 // (op 3180) 


s_lshr_b32 s53, s54, s11 // (op 3180) 

s_and_b32 s54, s11, 2147483648 // (op 3180) 
s_cmp_eq_i32 s54, 0 // (op 3180) 
s_mov_b32 s55, scc // (op 3180) 

s_cmp_eq_i32 s55, 0 // (op 3180) 
s_cselect_b32 s54, s52, s53 // (op 3180) 


s_mul_hi_u32 s53, s54, s12 // (op 3180) 
s_sub_u32 s55, s54, s53 // (op 3180) 
s_lshr_b32 s84, s55, 1 // (op 3180) 

s_add_u32 s55, s84, s53 // (op 3180) 


s_lshr_b32 s53, s55, s10 // (op 3180) 

s_and_b32 s55, s10, 2147483648 // (op 3180) 
s_cmp_eq_i32 s55, 0 // (op 3180) 
s_mov_b32 s84, scc // (op 3180) 

s_cmp_eq_i32 s84, 0 // (op 3180) 
s_cselect_b32 s55, s54, s53 // (op 3180) 


s_mul_i32 s53, s55, s14 // (op 3180) 

s_sub_u32 s55, s54, s53 // (op 3180) 

v_lshl_add_u32 v13, s55, 5, v14 // (op 3180) 


v_mul_lo_u32 v14, v13, s36 // (op 3180) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 3180) 
s_mul_i32 s53, s54, s84 // (op 3180) 


s_sub_u32 s54, s52, s53 // (op 3180) 


v_lshl_add_u32 v13, s54, 7, v12 // (op 3180) 


v_add_u32 v12, v13, v14 // (op 3180) 


s_mul_i32 s53, s36, s23 // (op 3182) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 3182) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 3182) least significant
s_mul_hi_u32 s55, s36, s22 // (op 3182) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3182) most significant: sum
s_add_u32 s53, s53, s55 // (op 3182) most significant: sum


s_mov_b32 s72, s38 // (op 3182) reg expression
s_mov_b32 s73, s39 // (op 3182) reg expression
s_mov_b32 s74, s52 // (op 3182) reg expression
s_mov_b32 s75, 131072 // (op 3182) call()

s_and_b32 s52, s7, 4294967294 // (op 3308) 
v_lshrrev_b32 v13, 2, v1 // (op 3308) 
s_add_u32 s53, s0, s52 // (op 3308) 

v_lshlrev_b32 v14, 1, v13 // (op 3308) 

v_and_b32 v13, -4, v1 // (op 3308) 
s_mul_i32 s52, s16, s8 // (op 3308) 
s_add_u32 s52, s52, s53 // (op 3308) 

v_sub_u32 v15, v1, v13 // (op 3308) 

s_mul_hi_u32 s53, s52, s13 // (op 3308) 
s_sub_u32 s54, s52, s53 // (op 3308) 
s_lshr_b32 s55, s54, 1 // (op 3308) 

s_add_u32 s54, s55, s53 // (op 3308) 


s_lshr_b32 s53, s54, s11 // (op 3308) 

s_and_b32 s54, s11, 2147483648 // (op 3308) 
s_cmp_eq_i32 s54, 0 // (op 3308) 
s_mov_b32 s55, scc // (op 3308) 

s_cmp_eq_i32 s55, 0 // (op 3308) 
s_cselect_b32 s54, s52, s53 // (op 3308) 


s_mul_hi_u32 s53, s54, s12 // (op 3308) 
s_sub_u32 s55, s54, s53 // (op 3308) 
s_lshr_b32 s84, s55, 1 // (op 3308) 

s_add_u32 s55, s84, s53 // (op 3308) 


s_lshr_b32 s53, s55, s10 // (op 3308) 

s_and_b32 s55, s10, 2147483648 // (op 3308) 
s_cmp_eq_i32 s55, 0 // (op 3308) 
s_mov_b32 s84, scc // (op 3308) 

s_cmp_eq_i32 s84, 0 // (op 3308) 
s_cselect_b32 s55, s54, s53 // (op 3308) 


v_lshl_add_u32 v13, s55, 7, v14 // (op 3308) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 3308) 
s_mul_i32 s53, s54, s84 // (op 3308) 


s_sub_u32 s54, s52, s53 // (op 3308) 


v_lshl_add_u32 v14, s54, 2, v15 // (op 3308) 


v_mul_lo_u32 v15, v13, s40 // (op 3308) 
v_add_u32 v15, v15, v14 // (op 3308) 


s_mul_i32 s53, s40, s35 // (op 3310) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 3310) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 3310) least significant
s_mul_hi_u32 s55, s40, s34 // (op 3310) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3310) most significant: sum
s_add_u32 s53, s53, s55 // (op 3310) most significant: sum


s_mov_b32 s76, s42 // (op 3310) reg expression
s_mov_b32 s77, s43 // (op 3310) reg expression
s_mov_b32 s78, s52 // (op 3310) reg expression
s_mov_b32 s79, 131072 // (op 3310) call()

s_and_b32 s52, s7, 4294967294 // (op 3436) 
v_lshrrev_b32 v13, 6, v1 // (op 3436) 
s_add_u32 s53, s0, s52 // (op 3436) 

v_and_b32 v14, 1, v13 // (op 3436) 

v_and_b32 v13, 15, v1 // (op 3436) 
s_mul_i32 s52, s16, s8 // (op 3436) 
s_add_u32 s52, s52, s53 // (op 3436) 

v_lshl_add_u32 v16, v14, 4, v13 // (op 3436) 


s_mul_hi_u32 s53, s52, s13 // (op 3436) 
v_and_b32 v13, 63, v1 // (op 3436) 
s_sub_u32 s54, s52, s53 // (op 3436) 
v_lshrrev_b32 v14, 4, v13 // (op 3436) 

s_lshr_b32 s55, s54, 1 // (op 3436) 

s_add_u32 s54, s55, s53 // (op 3436) 


s_lshr_b32 s53, s54, s11 // (op 3436) 

s_and_b32 s54, s11, 2147483648 // (op 3436) 
s_cmp_eq_i32 s54, 0 // (op 3436) 
s_mov_b32 s55, scc // (op 3436) 

s_cmp_eq_i32 s55, 0 // (op 3436) 
s_cselect_b32 s54, s52, s53 // (op 3436) 


s_mul_hi_u32 s53, s54, s12 // (op 3436) 
s_sub_u32 s55, s54, s53 // (op 3436) 
s_lshr_b32 s84, s55, 1 // (op 3436) 

s_add_u32 s55, s84, s53 // (op 3436) 


s_lshr_b32 s53, s55, s10 // (op 3436) 

s_and_b32 s55, s10, 2147483648 // (op 3436) 
s_cmp_eq_i32 s55, 0 // (op 3436) 
s_mov_b32 s84, scc // (op 3436) 

s_cmp_eq_i32 s84, 0 // (op 3436) 
s_cselect_b32 s55, s54, s53 // (op 3436) 


s_mul_i32 s53, s55, s14 // (op 3436) 

s_sub_u32 s55, s54, s53 // (op 3436) 

v_lshl_add_u32 v13, s55, 5, v16 // (op 3436) 


v_mul_lo_u32 v16, v13, s44 // (op 3436) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 3436) 
s_mul_i32 s53, s54, s84 // (op 3436) 


s_sub_u32 s54, s52, s53 // (op 3436) 


v_lshl_add_u32 v13, s54, 2, v14 // (op 3436) 


v_add_u32 v14, v13, v16 // (op 3436) 


s_mul_i32 s53, s44, s23 // (op 3438) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 3438) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 3438) least significant
s_mul_hi_u32 s55, s44, s22 // (op 3438) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3438) most significant: sum
s_add_u32 s53, s53, s55 // (op 3438) most significant: sum


s_mov_b32 s80, s46 // (op 3438) reg expression
s_mov_b32 s81, s47 // (op 3438) reg expression
s_mov_b32 s82, s52 // (op 3438) reg expression
s_mov_b32 s83, 131072 // (op 3438) call()

v_and_b32 v13, -4, v1 // (op 3597) 
v_sub_u32 v16, v1, v13 // (op 3597) 

v_lshrrev_b32 v13, 2, v1 // (op 3597) 
v_lshl_add_u32 v17, v13, 3, v16 // (op 3597) 


s_and_b32 s52, s7, 4294967294 // (op 1484) 
s_cmp_lt_u32 s52, s7 // (op 1483) 







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_1483 // (op 1483) Condition: Top (jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_1483 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_1483:

s_add_u32 s52, s52, 1 // (op 1485) 
v_mov_b64 v[18:19], s[40:41] // (op 1130) convert

v_mov_b32 v13, v15 // (op 1130) 
v_mov_b32 v16, v13 // (op 1130) 
buffer_load_ubyte v20, v16, s[76:79], 0 offen offset: 0 // (op 1130) Load value
v_add_u32 v16, v16, v18 // (op 1130) 
buffer_load_ubyte v21, v16, s[76:79], 0 offen offset: 0 // (op 1130) Load value



v_add_u32 v15, 4, v15 // (op 3300) 


s_barrier  // (op 3931) 
v_mov_b32 v13, v9 // (op 1138) 



v_mov_b32 v16, v13 // (op 1138) reg expression

buffer_load_dwordx4 v[22:25], v16, s[68:71], 0 offen // (op 1138) Load value

s_waitcnt vmcnt(2)
ds_write_b8 v17, v20 // (op 1131) Store local data 


v_mov_b32 v16, v12 // (op 1140) 

buffer_load_dwordx4 v[26:29], v16, s[72:75], 0 offen // (op 1140) Load value
buffer_load_dwordx4 v[30:33], v16, s[72:75], 0 offen offset:64 // (op 1140) Load value

v_add_u32 v12, 128, v12 // (op 3172) 
s_waitcnt vmcnt(3)
ds_write_b8 v17, v21 offset:4 // (op 1131) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 3934) 



ds_read_u8 v16, v0 // (op 1133) Load local data 


v_mov_b32 v18, v14 // (op 1135) 

buffer_load_ubyte v19, v18, s[80:83], 0 offen offset: 0 // (op 1135) Load value

v_add_u32 v14, 4, v14 // (op 3428) 


s_lshl_b64 s[54:55], s[30:31], 5 // (op 1255) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 1255) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 1255) 

v_add_u32 v18, v13, s54 // (op 1255) 


s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[26:33], a[0:3], v16, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1136) 
buffer_load_dwordx4 v[34:37], v18, s[68:71], 0 offen // (op 1255) Load value






ds_read_u8 v16, v0 offset:128 // (op 1245) Load local data 


s_lshl_b64 s[54:55], s[30:31], 5 // (op 1377) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 1377) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 1377) 

s_lshl_b32 s53, s54, 1 // (op 1377) 

v_add_u32 v18, v13, s53 // (op 1377) 


buffer_load_dwordx4 v[20:23], v18, s[68:71], 0 offen // (op 1377) Load value

s_waitcnt vmcnt(1) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[26:33], a[4:7], v16, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1247) 





ds_read_u8 v16, v0 offset:256 // (op 1367) Load local data 


s_lshl_b64 s[54:55], s[30:31], 5 // (op 1499) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 1499) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 1499) 

s_mul_i32 s53, 3, s54 // (op 1499) 

v_add_u32 v18, v13, s53 // (op 1499) 


buffer_load_dwordx4 v[34:37], v18, s[68:71], 0 offen // (op 1499) Load value

s_waitcnt vmcnt(1) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:23], v[26:33], a[8:11], v16, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1369) 






v_add_u32 v9, 64, v9 // (op 3005) 
ds_read_u8 v13, v0 offset:384 // (op 1489) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:37], v[26:33], a[12:15], v13, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1491) 




s_cmp_lt_u32 s52, s7 // (op 1483) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_1483 // (op 1483) Condition: Bottom (jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_1483 if true)
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_1483:






s_barrier  // (op 4134) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 243) 
s_cmp_lg_u32 s18, 0 // (op 269) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Send_Tile_269 // (op 269) Condition: False, jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Send_Tile_269



v_add_u32 v9, 12288, v8 // (op 1698) 
s_nop 0
buffer_store_dword a12, v9, s[64:67], 0 offen sc1 // (op 1698) Store value

v_add_u32 v9, 12416, v8 // (op 1698) 
buffer_store_dword a13, v9, s[64:67], 0 offen sc1 // (op 1698) Store value

v_add_u32 v9, 12544, v8 // (op 1698) 
buffer_store_dword a14, v9, s[64:67], 0 offen sc1 // (op 1698) Store value

v_add_u32 v9, 12672, v8 // (op 1698) 
buffer_store_dword a15, v9, s[64:67], 0 offen sc1 // (op 1698) Store value




v_add_u32 v9, 8192, v8 // (op 1691) 
buffer_store_dword a8, v9, s[64:67], 0 offen sc1 // (op 1691) Store value

v_add_u32 v9, 8320, v8 // (op 1691) 
buffer_store_dword a9, v9, s[64:67], 0 offen sc1 // (op 1691) Store value

v_add_u32 v9, 8448, v8 // (op 1691) 
buffer_store_dword a10, v9, s[64:67], 0 offen sc1 // (op 1691) Store value

v_add_u32 v9, 8576, v8 // (op 1691) 
buffer_store_dword a11, v9, s[64:67], 0 offen sc1 // (op 1691) Store value




v_add_u32 v9, 4096, v8 // (op 1684) 
buffer_store_dword a4, v9, s[64:67], 0 offen sc1 // (op 1684) Store value

v_add_u32 v9, 4224, v8 // (op 1684) 
buffer_store_dword a5, v9, s[64:67], 0 offen sc1 // (op 1684) Store value

v_add_u32 v9, 4352, v8 // (op 1684) 
buffer_store_dword a6, v9, s[64:67], 0 offen sc1 // (op 1684) Store value

v_add_u32 v9, 4480, v8 // (op 1684) 
buffer_store_dword a7, v9, s[64:67], 0 offen sc1 // (op 1684) Store value




buffer_store_dword a0, v8, s[64:67], 0 offen sc1 // (op 266) Store value
buffer_store_dword a1, v8, s[64:67], 0 offen offset:128 sc1 // (op 266) Store value
buffer_store_dword a2, v8, s[64:67], 0 offen offset:256 sc1 // (op 266) Store value
buffer_store_dword a3, v8, s[64:67], 0 offen offset:384 sc1 // (op 266) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 270) 
v_cmp_eq_i32 vcc, v1, 0 // (op 274) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_274 // (op 274) Condition: False, jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_274
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
s_branch RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_274 // (op 274) Condition: Done, jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_274
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_274:
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_274:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Send_Tile_269 // (op 269) Condition: Done, jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Send_Tile_269
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Send_Tile_269:
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Send_Tile_269:

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
s_cbranch_scc0 RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Receive_Tile_285 // (op 285) Condition: False, jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Receive_Tile_285
s_mov_b32 s52, 0 // (op 251) call()
s_mul_i32 s53, s52, s8 // (op 250) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 250) 
s_mov_b32 s84, s54 // (op 250) convert

s_sub_u32 s54, s84, s19 // (op 250) 

s_sub_u32 s55, s54, 1 // (op 250) 

s_cmp_lt_u32 s53, s55 // (op 250) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_250 // (op 250) Condition: Top (jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_250 if false)
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_ReceiveTileLoop_250:
s_add_u32 s53, s16, 1 // (op 3769) 
v_lshrrev_b32 v8, 5, v1 // (op 3769) 
s_add_u32 s54, s53, s52 // (op 3769) 
s_lshl_b32 s54, s54, 7 // (op 3769) 

v_lshlrev_b32 v9, 2, v8 // (op 3769) 

v_add_lshl_u32 v8, s54, v9, 5 // (op 3769) 


v_and_b32 v9, -32, v1 // (op 3769) 
v_sub_u32 v12, v1, v9 // (op 3769) 

v_add_lshl_u32 v9, v8, v12, 2 // (op 3769) 


s_lshl_b32 s53, s15, 14 // (op 3771) 
s_mov_b32 s64, s28 // (op 3771) reg expression
s_mov_b32 s65, s29 // (op 3771) reg expression
s_mov_b32 s66, s53 // (op 3771) reg expression
s_mov_b32 s67, 131072 // (op 3771) call()

RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_289:
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
s_cbranch_scc1 RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_289 // (op 289) Condition: Bottom (jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_289 if true)
s_barrier  // (op 293) 
v_cmp_eq_i32 vcc, v1, 0 // (op 292) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_292 // (op 292) Condition: False, jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_292
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
s_branch RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_292 // (op 292) Condition: Done, jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_292
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_292:
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_292:




buffer_load_dword v12, v9, s[64:67], 0 offen // (op 304) Load value
buffer_load_dword v13, v9, s[64:67], 0 offen offset:128 // (op 304) Load value
buffer_load_dword v14, v9, s[64:67], 0 offen offset:256 // (op 304) Load value
buffer_load_dword v15, v9, s[64:67], 0 offen offset:384 // (op 304) Load value
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




v_mov_b32 v8, 4096 // (op 1724) 
v_add_i32 v12, v9, v8 // (op 1724) 

buffer_load_dword v14, v12, s[64:67], 0 offen // (op 1724) Load value

v_mov_b32 v8, 4224 // (op 1724) 
v_add_i32 v12, v9, v8 // (op 1724) 

buffer_load_dword v15, v12, s[64:67], 0 offen // (op 1724) Load value

v_mov_b32 v8, 4352 // (op 1724) 
v_add_i32 v12, v9, v8 // (op 1724) 

buffer_load_dword v16, v12, s[64:67], 0 offen // (op 1724) Load value

v_mov_b32 v8, 4480 // (op 1724) 
v_add_i32 v12, v9, v8 // (op 1724) 

buffer_load_dword v17, v12, s[64:67], 0 offen // (op 1724) Load value

v_accvgpr_read v18, a4 // (op 1725) reg expression
v_accvgpr_read v19, a5 // (op 1725) reg expression
v_accvgpr_read v20, a6 // (op 1725) reg expression
v_accvgpr_read v21, a7 // (op 1725) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v18, v18, v14 // (op 1726) 
s_waitcnt vmcnt(2)
v_add_f32 v19, v19, v15 // (op 1726) 
s_waitcnt vmcnt(1)
v_add_f32 v20, v20, v16 // (op 1726) 
s_waitcnt vmcnt(0)
v_add_f32 v21, v21, v17 // (op 1726) 

v_accvgpr_write a4, v18 // (op 1727) reg expression
v_accvgpr_write a5, v19 // (op 1727) reg expression
v_accvgpr_write a6, v20 // (op 1727) reg expression
v_accvgpr_write a7, v21 // (op 1727) reg expression




v_mov_b32 v8, 8192 // (op 1737) 
v_add_i32 v12, v9, v8 // (op 1737) 

buffer_load_dword v14, v12, s[64:67], 0 offen // (op 1737) Load value

v_mov_b32 v8, 8320 // (op 1737) 
v_add_i32 v12, v9, v8 // (op 1737) 

buffer_load_dword v15, v12, s[64:67], 0 offen // (op 1737) Load value

v_mov_b32 v8, 8448 // (op 1737) 
v_add_i32 v12, v9, v8 // (op 1737) 

buffer_load_dword v16, v12, s[64:67], 0 offen // (op 1737) Load value

v_mov_b32 v8, 8576 // (op 1737) 
v_add_i32 v12, v9, v8 // (op 1737) 

buffer_load_dword v17, v12, s[64:67], 0 offen // (op 1737) Load value

v_accvgpr_read v18, a8 // (op 1738) reg expression
v_accvgpr_read v19, a9 // (op 1738) reg expression
v_accvgpr_read v20, a10 // (op 1738) reg expression
v_accvgpr_read v21, a11 // (op 1738) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v18, v18, v14 // (op 1739) 
s_waitcnt vmcnt(2)
v_add_f32 v19, v19, v15 // (op 1739) 
s_waitcnt vmcnt(1)
v_add_f32 v20, v20, v16 // (op 1739) 
s_waitcnt vmcnt(0)
v_add_f32 v21, v21, v17 // (op 1739) 

v_accvgpr_write a8, v18 // (op 1740) reg expression
v_accvgpr_write a9, v19 // (op 1740) reg expression
v_accvgpr_write a10, v20 // (op 1740) reg expression
v_accvgpr_write a11, v21 // (op 1740) reg expression




v_mov_b32 v8, 12288 // (op 1750) 
v_add_i32 v12, v9, v8 // (op 1750) 

buffer_load_dword v14, v12, s[64:67], 0 offen // (op 1750) Load value

v_mov_b32 v8, 12416 // (op 1750) 
v_add_i32 v12, v9, v8 // (op 1750) 

buffer_load_dword v15, v12, s[64:67], 0 offen // (op 1750) Load value

v_mov_b32 v8, 12544 // (op 1750) 
v_add_i32 v12, v9, v8 // (op 1750) 

buffer_load_dword v16, v12, s[64:67], 0 offen // (op 1750) Load value

v_mov_b32 v8, 12672 // (op 1750) 
v_add_i32 v12, v9, v8 // (op 1750) 

buffer_load_dword v17, v12, s[64:67], 0 offen // (op 1750) Load value

v_accvgpr_read v18, a12 // (op 1751) reg expression
v_accvgpr_read v19, a13 // (op 1751) reg expression
v_accvgpr_read v20, a14 // (op 1751) reg expression
v_accvgpr_read v21, a15 // (op 1751) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v18, v18, v14 // (op 1752) 
s_waitcnt vmcnt(2)
v_add_f32 v19, v19, v15 // (op 1752) 
s_waitcnt vmcnt(1)
v_add_f32 v20, v20, v16 // (op 1752) 
s_waitcnt vmcnt(0)
v_add_f32 v21, v21, v17 // (op 1752) 

v_accvgpr_write a12, v18 // (op 1753) reg expression
v_accvgpr_write a13, v19 // (op 1753) reg expression
v_accvgpr_write a14, v20 // (op 1753) reg expression
v_accvgpr_write a15, v21 // (op 1753) reg expression

s_add_u32 s52, s52, 1 // (op 252) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 250) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 250) 
s_mov_b32 s84, s54 // (op 250) convert

s_sub_u32 s54, s84, s19 // (op 250) 

s_sub_u32 s55, s54, 1 // (op 250) 

s_cmp_lt_u32 s53, s55 // (op 250) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_ReceiveTileLoop_250 // (op 250) Condition: Bottom (jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_ReceiveTileLoop_250 if true)
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_250:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Receive_Tile_285 // (op 285) Condition: Done, jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Receive_Tile_285
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Receive_Tile_285:
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Receive_Tile_285:

s_cmp_eq_i32 s18, 0 // (op 328) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_328 // (op 328) Condition: False, jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_328
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 106) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_106 // (op 106) Condition: False, jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_106
v_accvgpr_read v12, a0 // (op 38) 
v_accvgpr_read v13, a1 // (op 38) 
v_accvgpr_read v14, a2 // (op 38) 
v_accvgpr_read v15, a3 // (op 38) 
v_mul_f32 v16, v7, v12 // (op 38) 
v_mul_f32 v17, v7, v13 // (op 38) 
v_mul_f32 v18, v7, v14 // (op 38) 
v_mul_f32 v19, v7, v15 // (op 38) 




buffer_store_dwordx4 v[16:19], v11, s[60:63], 0 offen // (op 44) Store value

v_accvgpr_read v12, a4 // (op 1787) 
v_accvgpr_read v13, a5 // (op 1787) 
v_accvgpr_read v14, a6 // (op 1787) 
v_accvgpr_read v15, a7 // (op 1787) 
v_mul_f32 v16, v7, v12 // (op 1787) 
v_mul_f32 v17, v7, v13 // (op 1787) 
v_mul_f32 v18, v7, v14 // (op 1787) 
v_mul_f32 v19, v7, v15 // (op 1787) 




buffer_store_dwordx4 v[16:19], v11, s[60:63], 0 offen offset:128 // (op 1789) Store value

v_accvgpr_read v12, a8 // (op 1805) 
v_accvgpr_read v13, a9 // (op 1805) 
v_accvgpr_read v14, a10 // (op 1805) 
v_accvgpr_read v15, a11 // (op 1805) 
v_mul_f32 v16, v7, v12 // (op 1805) 
v_mul_f32 v17, v7, v13 // (op 1805) 
v_mul_f32 v18, v7, v14 // (op 1805) 
v_mul_f32 v19, v7, v15 // (op 1805) 




buffer_store_dwordx4 v[16:19], v11, s[60:63], 0 offen offset:256 // (op 1807) Store value

v_accvgpr_read v12, a12 // (op 1823) 
v_accvgpr_read v13, a13 // (op 1823) 
v_accvgpr_read v14, a14 // (op 1823) 
v_accvgpr_read v15, a15 // (op 1823) 
v_mul_f32 v16, v7, v12 // (op 1823) 
v_mul_f32 v17, v7, v13 // (op 1823) 
v_mul_f32 v18, v7, v14 // (op 1823) 
v_mul_f32 v19, v7, v15 // (op 1823) 




buffer_store_dwordx4 v[16:19], v11, s[60:63], 0 offen offset:384 // (op 1825) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_106 // (op 106) Condition: Done, jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_106
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_106:



buffer_load_dwordx4 v[12:15], v10, s[56:59], 0 offen offset:256 // (op 1881) Load value
v_accvgpr_read v16, a8 // (op 1878) 
v_accvgpr_read v17, a9 // (op 1878) 
v_accvgpr_read v18, a10 // (op 1878) 
v_accvgpr_read v19, a11 // (op 1878) 
v_mul_f32 v20, v7, v16 // (op 1878) 
v_mul_f32 v21, v7, v17 // (op 1878) 
v_mul_f32 v22, v7, v18 // (op 1878) 
v_mul_f32 v23, v7, v19 // (op 1878) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v6, v12 // (op 1882) 
v_mul_f32 v17, v6, v13 // (op 1882) 
v_mul_f32 v18, v6, v14 // (op 1882) 
v_mul_f32 v19, v6, v15 // (op 1882) 

v_add_f32 v12, v16, v20 // (op 1879) 
v_add_f32 v13, v17, v21 // (op 1879) 
v_add_f32 v14, v18, v22 // (op 1879) 
v_add_f32 v15, v19, v23 // (op 1879) 





buffer_store_dwordx4 v[12:15], v11, s[60:63], 0 offen offset:256 // (op 1880) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v10, s[56:59], 0 offen offset:128 // (op 1863) Load value
v_accvgpr_read v16, a4 // (op 1860) 
v_accvgpr_read v17, a5 // (op 1860) 
v_accvgpr_read v18, a6 // (op 1860) 
v_accvgpr_read v19, a7 // (op 1860) 
v_mul_f32 v20, v7, v16 // (op 1860) 
v_mul_f32 v21, v7, v17 // (op 1860) 
v_mul_f32 v22, v7, v18 // (op 1860) 
v_mul_f32 v23, v7, v19 // (op 1860) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v6, v12 // (op 1864) 
v_mul_f32 v17, v6, v13 // (op 1864) 
v_mul_f32 v18, v6, v14 // (op 1864) 
v_mul_f32 v19, v6, v15 // (op 1864) 

v_add_f32 v12, v16, v20 // (op 1861) 
v_add_f32 v13, v17, v21 // (op 1861) 
v_add_f32 v14, v18, v22 // (op 1861) 
v_add_f32 v15, v19, v23 // (op 1861) 





buffer_store_dwordx4 v[12:15], v11, s[60:63], 0 offen offset:128 // (op 1862) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v10, s[56:59], 0 offen offset:384 // (op 1899) Load value
v_accvgpr_read v16, a12 // (op 1896) 
v_accvgpr_read v17, a13 // (op 1896) 
v_accvgpr_read v18, a14 // (op 1896) 
v_accvgpr_read v19, a15 // (op 1896) 
v_mul_f32 v20, v7, v16 // (op 1896) 
v_mul_f32 v21, v7, v17 // (op 1896) 
v_mul_f32 v22, v7, v18 // (op 1896) 
v_mul_f32 v23, v7, v19 // (op 1896) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v6, v12 // (op 1900) 
v_mul_f32 v17, v6, v13 // (op 1900) 
v_mul_f32 v18, v6, v14 // (op 1900) 
v_mul_f32 v19, v6, v15 // (op 1900) 

v_add_f32 v12, v16, v20 // (op 1897) 
v_add_f32 v13, v17, v21 // (op 1897) 
v_add_f32 v14, v18, v22 // (op 1897) 
v_add_f32 v15, v19, v23 // (op 1897) 





buffer_store_dwordx4 v[12:15], v11, s[60:63], 0 offen offset:384 // (op 1898) Store value




s_nop 1
buffer_load_dwordx4 v[12:15], v10, s[56:59], 0 offen // (op 118) Load value
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





buffer_store_dwordx4 v[12:15], v11, s[60:63], 0 offen // (op 117) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_106:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_328 // (op 328) Condition: Done, jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_328
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_328:
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_328:


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
s_cbranch_scc1 RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_SKStreamTileLoop_241 // (op 241) Condition: Bottom (jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_SKStreamTileLoop_241 if true)
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_241:











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
s_cbranch_scc0 RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_396 // (op 396) Condition: Top (jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_396 if false)
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_DPStreamTileLoop_396:
v_lshrrev_b32 v0, 7, v1 // (op 3567) 
s_mul_i32 s15, s16, s50 // (op 3567) 
s_add_u32 s15, s15, s0 // (op 3567) 
v_and_b32 v5, 63, v1 // (op 3567) 
s_add_u32 s6, s15, s9 // (op 3567) 

v_lshrrev_b32 v8, 4, v5 // (op 3567) 
s_mul_hi_u32 s15, s6, s13 // (op 3567) 
v_lshlrev_b32 v9, 2, v8 // (op 3567) 

s_sub_u32 s7, s6, s15 // (op 3567) 
v_lshl_add_u32 v8, v0, 4, v9 // (op 3567) 


s_lshr_b32 s8, s7, 1 // (op 3567) 

v_lshrrev_b32 v0, 2, v5 // (op 3567) 

s_add_u32 s7, s8, s15 // (op 3567) 


v_and_b32 v5, 3, v0 // (op 3567) 

s_lshr_b32 s8, s7, s11 // (op 3567) 

v_and_b32 v0, 3, v1 // (op 3567) 
s_and_b32 s7, s11, 2147483648 // (op 3567) 
v_lshl_add_u32 v9, v5, 2, v0 // (op 3567) 


s_cmp_eq_i32 s7, 0 // (op 3567) 
s_mov_b32 s15, scc // (op 3567) 

v_lshrrev_b32 v0, 6, v1 // (op 3567) 
s_cmp_eq_i32 s15, 0 // (op 3567) 
s_cselect_b32 s7, s6, s8 // (op 3567) 



v_and_b32 v5, 1, v0 // (op 3567) 

s_mul_hi_u32 s6, s7, s12 // (op 3567) 
v_lshl_add_u32 v0, v5, 4, v9 // (op 3567) 


s_sub_u32 s8, s7, s6 // (op 3567) 
s_lshr_b32 s15, s8, 1 // (op 3567) 

s_add_u32 s8, s15, s6 // (op 3567) 


s_lshr_b32 s6, s8, s10 // (op 3567) 

s_and_b32 s8, s10, 2147483648 // (op 3567) 
s_cmp_eq_i32 s8, 0 // (op 3567) 
s_mov_b32 s15, scc // (op 3567) 

s_cmp_eq_i32 s15, 0 // (op 3567) 
s_cselect_b32 s8, s7, s6 // (op 3567) 


s_mul_i32 s6, s8, s14 // (op 3567) 
v_lshl_add_u32 v5, s8, 7, v8 // (op 3567) 


s_sub_u32 s8, s7, s6 // (op 3567) 


v_lshl_add_u32 v8, s8, 5, v0 // (op 3567) 


v_mul_lo_u32 v0, v8, s4 // (op 3567) 

v_add_lshl_u32 v8, v5, v0, 2 // (op 3567) 


s_mul_i32 s19, s4, s23 // (op 3569) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 3569) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 3569) least significant
s_mul_hi_u32 s6, s4, s22 // (op 3569) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3569) most significant: sum
s_add_u32 s19, s19, s6 // (op 3569) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 3569) 

s_mov_b32 s56, s20 // (op 3569) reg expression
s_mov_b32 s57, s21 // (op 3569) reg expression
s_mov_b32 s58, s28 // (op 3569) reg expression
s_mov_b32 s59, 131072 // (op 3569) call()

v_lshrrev_b32 v0, 7, v1 // (op 3667) 
s_mul_i32 s15, s16, s50 // (op 3667) 
s_add_u32 s15, s15, s0 // (op 3667) 
v_and_b32 v5, 63, v1 // (op 3667) 
s_add_u32 s6, s15, s9 // (op 3667) 

v_lshrrev_b32 v9, 4, v5 // (op 3667) 
s_mul_hi_u32 s15, s6, s13 // (op 3667) 
v_lshlrev_b32 v10, 2, v9 // (op 3667) 

s_sub_u32 s7, s6, s15 // (op 3667) 
v_lshl_add_u32 v9, v0, 4, v10 // (op 3667) 


s_lshr_b32 s8, s7, 1 // (op 3667) 

v_lshrrev_b32 v0, 2, v5 // (op 3667) 

s_add_u32 s7, s8, s15 // (op 3667) 


v_and_b32 v5, 3, v0 // (op 3667) 

s_lshr_b32 s8, s7, s11 // (op 3667) 

v_and_b32 v0, 3, v1 // (op 3667) 
s_and_b32 s7, s11, 2147483648 // (op 3667) 
v_lshl_add_u32 v10, v5, 2, v0 // (op 3667) 


s_cmp_eq_i32 s7, 0 // (op 3667) 
s_mov_b32 s15, scc // (op 3667) 

v_lshrrev_b32 v0, 6, v1 // (op 3667) 
s_cmp_eq_i32 s15, 0 // (op 3667) 
s_cselect_b32 s7, s6, s8 // (op 3667) 



v_and_b32 v5, 1, v0 // (op 3667) 

s_mul_hi_u32 s6, s7, s12 // (op 3667) 
v_lshl_add_u32 v0, v5, 4, v10 // (op 3667) 


s_sub_u32 s8, s7, s6 // (op 3667) 
s_lshr_b32 s15, s8, 1 // (op 3667) 

s_add_u32 s8, s15, s6 // (op 3667) 


s_lshr_b32 s6, s8, s10 // (op 3667) 

s_and_b32 s8, s10, 2147483648 // (op 3667) 
s_cmp_eq_i32 s8, 0 // (op 3667) 
s_mov_b32 s15, scc // (op 3667) 

s_cmp_eq_i32 s15, 0 // (op 3667) 
s_cselect_b32 s8, s7, s6 // (op 3667) 


s_mul_i32 s6, s8, s14 // (op 3667) 
v_lshl_add_u32 v5, s8, 7, v9 // (op 3667) 


s_sub_u32 s8, s7, s6 // (op 3667) 


v_lshl_add_u32 v9, s8, 5, v0 // (op 3667) 


v_mul_lo_u32 v0, v9, s24 // (op 3667) 

v_add_lshl_u32 v9, v5, v0, 2 // (op 3667) 


s_mul_i32 s19, s24, s23 // (op 3669) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 3669) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 3669) least significant
s_mul_hi_u32 s6, s24, s22 // (op 3669) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3669) most significant: sum
s_add_u32 s19, s19, s6 // (op 3669) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 3669) 

s_mov_b32 s60, s26 // (op 3669) reg expression
s_mov_b32 s61, s27 // (op 3669) reg expression
s_mov_b32 s62, s28 // (op 3669) reg expression
s_mov_b32 s63, 131072 // (op 3669) call()

v_accvgpr_write a0, 0.00000 // (op 153) call()
v_accvgpr_write a1, 0.00000 // (op 153) call()
v_accvgpr_write a2, 0.00000 // (op 153) call()
v_accvgpr_write a3, 0.00000 // (op 153) call()
v_accvgpr_write a4, 0.00000 // (op 512) call()
v_accvgpr_write a5, 0.00000 // (op 512) call()
v_accvgpr_write a6, 0.00000 // (op 512) call()
v_accvgpr_write a7, 0.00000 // (op 512) call()
v_accvgpr_write a8, 0.00000 // (op 634) call()
v_accvgpr_write a9, 0.00000 // (op 634) call()
v_accvgpr_write a10, 0.00000 // (op 634) call()
v_accvgpr_write a11, 0.00000 // (op 634) call()
v_accvgpr_write a12, 0.00000 // (op 756) call()
v_accvgpr_write a13, 0.00000 // (op 756) call()
v_accvgpr_write a14, 0.00000 // (op 756) call()
v_accvgpr_write a15, 0.00000 // (op 756) call()
v_lshrrev_b32 v0, 7, v1 // (op 3054) 
s_mul_i32 s15, s16, s50 // (op 3054) 
s_add_u32 s15, s15, s0 // (op 3054) 
v_and_b32 v5, 15, v1 // (op 3054) 
s_add_u32 s6, s15, s9 // (op 3054) 

v_lshl_add_u32 v10, v0, 4, v5 // (op 3054) 


s_mul_hi_u32 s15, s6, s13 // (op 3054) 
v_and_b32 v0, 63, v1 // (op 3054) 
s_sub_u32 s7, s6, s15 // (op 3054) 
v_lshrrev_b32 v5, 4, v0 // (op 3054) 

s_lshr_b32 s8, s7, 1 // (op 3054) 

v_lshlrev_b32 v0, 5, v5 // (op 3054) 

s_add_u32 s7, s8, s15 // (op 3054) 


s_lshr_b32 s8, s7, s11 // (op 3054) 

s_and_b32 s7, s11, 2147483648 // (op 3054) 
s_cmp_eq_i32 s7, 0 // (op 3054) 
s_mov_b32 s15, scc // (op 3054) 

s_cmp_eq_i32 s15, 0 // (op 3054) 
s_cselect_b32 s7, s6, s8 // (op 3054) 


s_mul_hi_u32 s8, s7, s12 // (op 3054) 
s_sub_u32 s15, s7, s8 // (op 3054) 
s_lshr_b32 s18, s15, 1 // (op 3054) 

s_add_u32 s15, s18, s8 // (op 3054) 


s_lshr_b32 s8, s15, s10 // (op 3054) 

s_and_b32 s15, s10, 2147483648 // (op 3054) 
s_cmp_eq_i32 s15, 0 // (op 3054) 
s_mov_b32 s18, scc // (op 3054) 

s_cmp_eq_i32 s18, 0 // (op 3054) 
s_cselect_b32 s15, s7, s8 // (op 3054) 


v_lshl_add_u32 v5, s15, 7, v10 // (op 3054) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3054) 
s_mul_i32 s8, s7, s18 // (op 3054) 


s_sub_u32 s7, s6, s8 // (op 3054) 


v_lshl_add_u32 v10, s7, 7, v0 // (op 3054) 


v_mov_b32 v13, 0 // (op 3054) convert
v_mov_b32 v12, v10 // (op 3054) convert

v_mul_lo_u32 v11, v5, s31 // (op 3054) most significant: low of low * high
v_mul_hi_u32 v0, v5, s30 // (op 3054) most significant: high of low * low
v_mul_lo_u32 v10, v5, s30 // (op 3054) least significant: low of low * low
v_add_u32 v11, v11, v0 // (op 3054) most significant: sum

v_add_co_u32 v10, s[18:19], v10, v12 // (op 3054) least significant half
v_addc_co_u32 v11, s[18:19], v11, v13, s[18:19] // (op 3054) most significant half



v_lshlrev_b64 v[12:13], 2, v[10:11] // (op 3054) 

v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 3054) 

v_mov_b32 v0, v10 // (op 3054) convert

s_mul_i32 s19, s30, s35 // (op 3056) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 3056) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 3056) least significant
s_mul_hi_u32 s6, s30, s34 // (op 3056) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3056) most significant: sum
s_add_u32 s19, s19, s6 // (op 3056) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 3056) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 3056) 

s_mov_b32 s68, s32 // (op 3056) reg expression
s_mov_b32 s69, s33 // (op 3056) reg expression
s_mov_b32 s70, s18 // (op 3056) reg expression
s_mov_b32 s71, 131072 // (op 3056) call()

v_lshrrev_b32 v5, 7, v1 // (op 3075) 
s_mul_i32 s15, s16, s50 // (op 3075) 
s_add_u32 s15, s15, s0 // (op 3075) 
v_and_b32 v10, 15, v1 // (op 3075) 
s_add_u32 s6, s15, s9 // (op 3075) 

v_lshl_add_u32 v11, v5, 4, v10 // (op 3075) 


s_mul_hi_u32 s15, s6, s13 // (op 3075) 
v_and_b32 v5, 63, v1 // (op 3075) 
s_sub_u32 s7, s6, s15 // (op 3075) 
v_lshrrev_b32 v10, 4, v5 // (op 3075) 

s_lshr_b32 s8, s7, 1 // (op 3075) 

v_lshlrev_b32 v5, 5, v10 // (op 3075) 

s_add_u32 s7, s8, s15 // (op 3075) 


s_lshr_b32 s8, s7, s11 // (op 3075) 

s_and_b32 s7, s11, 2147483648 // (op 3075) 
s_cmp_eq_i32 s7, 0 // (op 3075) 
s_mov_b32 s15, scc // (op 3075) 

s_cmp_eq_i32 s15, 0 // (op 3075) 
s_cselect_b32 s7, s6, s8 // (op 3075) 


s_mul_hi_u32 s8, s7, s12 // (op 3075) 
s_sub_u32 s15, s7, s8 // (op 3075) 
s_lshr_b32 s18, s15, 1 // (op 3075) 

s_add_u32 s15, s18, s8 // (op 3075) 


s_lshr_b32 s8, s15, s10 // (op 3075) 

s_and_b32 s15, s10, 2147483648 // (op 3075) 
s_cmp_eq_i32 s15, 0 // (op 3075) 
s_mov_b32 s18, scc // (op 3075) 

s_cmp_eq_i32 s18, 0 // (op 3075) 
s_cselect_b32 s15, s7, s8 // (op 3075) 


v_lshl_add_u32 v10, s15, 7, v11 // (op 3075) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3075) 
s_mul_i32 s8, s7, s18 // (op 3075) 


s_sub_u32 s7, s6, s8 // (op 3075) 


v_lshl_add_u32 v11, s7, 7, v5 // (op 3075) 


v_mov_b32 v13, 0 // (op 3075) convert
v_mov_b32 v12, v11 // (op 3075) convert

v_mul_lo_u32 v15, v10, s31 // (op 3075) most significant: low of low * high
v_mul_hi_u32 v5, v10, s30 // (op 3075) most significant: high of low * low
v_mul_lo_u32 v14, v10, s30 // (op 3075) least significant: low of low * low
v_add_u32 v15, v15, v5 // (op 3075) most significant: sum

v_add_co_u32 v14, s[18:19], v14, v12 // (op 3075) least significant half
v_addc_co_u32 v15, s[18:19], v15, v13, s[18:19] // (op 3075) most significant half



v_lshlrev_b64 v[10:11], 2, v[14:15] // (op 3075) 

v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 3075) 

v_mov_b32 v5, v12 // (op 3075) convert

s_mul_i32 s19, s30, s35 // (op 3077) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 3077) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 3077) least significant
s_mul_hi_u32 s6, s30, s34 // (op 3077) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3077) most significant: sum
s_add_u32 s19, s19, s6 // (op 3077) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 3077) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 3077) 

s_mov_b32 s68, s32 // (op 3077) reg expression
s_mov_b32 s69, s33 // (op 3077) reg expression
s_mov_b32 s70, s18 // (op 3077) reg expression
s_mov_b32 s71, 131072 // (op 3077) call()

v_lshrrev_b32 v10, 6, v1 // (op 3203) 
s_mul_i32 s15, s16, s50 // (op 3203) 
s_add_u32 s15, s15, s0 // (op 3203) 
v_and_b32 v11, 1, v10 // (op 3203) 

s_add_u32 s6, s15, s9 // (op 3203) 

v_and_b32 v10, 15, v1 // (op 3203) 
s_mul_hi_u32 s15, s6, s13 // (op 3203) 
v_lshl_add_u32 v12, v11, 4, v10 // (op 3203) 


s_sub_u32 s7, s6, s15 // (op 3203) 
v_and_b32 v10, 48, v1 // (op 3203) 
s_lshr_b32 s8, s7, 1 // (op 3203) 

s_add_u32 s7, s8, s15 // (op 3203) 


s_lshr_b32 s8, s7, s11 // (op 3203) 

s_and_b32 s7, s11, 2147483648 // (op 3203) 
s_cmp_eq_i32 s7, 0 // (op 3203) 
s_mov_b32 s15, scc // (op 3203) 

s_cmp_eq_i32 s15, 0 // (op 3203) 
s_cselect_b32 s7, s6, s8 // (op 3203) 


s_mul_hi_u32 s8, s7, s12 // (op 3203) 
s_sub_u32 s15, s7, s8 // (op 3203) 
s_lshr_b32 s18, s15, 1 // (op 3203) 

s_add_u32 s15, s18, s8 // (op 3203) 


s_lshr_b32 s8, s15, s10 // (op 3203) 

s_and_b32 s15, s10, 2147483648 // (op 3203) 
s_cmp_eq_i32 s15, 0 // (op 3203) 
s_mov_b32 s18, scc // (op 3203) 

s_cmp_eq_i32 s18, 0 // (op 3203) 
s_cselect_b32 s15, s7, s8 // (op 3203) 


s_mul_i32 s8, s15, s14 // (op 3203) 

s_sub_u32 s15, s7, s8 // (op 3203) 

v_lshl_add_u32 v11, s15, 5, v12 // (op 3203) 


v_mul_lo_u32 v12, v11, s36 // (op 3203) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 3203) 
s_mul_i32 s8, s7, s18 // (op 3203) 


s_sub_u32 s7, s6, s8 // (op 3203) 


v_lshl_add_u32 v11, s7, 7, v10 // (op 3203) 


v_add_u32 v10, v11, v12 // (op 3203) 


s_mul_i32 s19, s36, s23 // (op 3205) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 3205) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 3205) least significant
s_mul_hi_u32 s6, s36, s22 // (op 3205) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3205) most significant: sum
s_add_u32 s19, s19, s6 // (op 3205) most significant: sum


s_mov_b32 s72, s38 // (op 3205) reg expression
s_mov_b32 s73, s39 // (op 3205) reg expression
s_mov_b32 s74, s18 // (op 3205) reg expression
s_mov_b32 s75, 131072 // (op 3205) call()

v_lshrrev_b32 v11, 6, v1 // (op 3222) 
s_mul_i32 s15, s16, s50 // (op 3222) 
s_add_u32 s15, s15, s0 // (op 3222) 
v_and_b32 v12, 1, v11 // (op 3222) 

s_add_u32 s6, s15, s9 // (op 3222) 

v_and_b32 v11, 15, v1 // (op 3222) 
s_mul_hi_u32 s15, s6, s13 // (op 3222) 
v_lshl_add_u32 v13, v12, 4, v11 // (op 3222) 


s_sub_u32 s7, s6, s15 // (op 3222) 
v_and_b32 v11, 48, v1 // (op 3222) 
s_lshr_b32 s8, s7, 1 // (op 3222) 

s_add_u32 s7, s8, s15 // (op 3222) 


s_lshr_b32 s8, s7, s11 // (op 3222) 

s_and_b32 s7, s11, 2147483648 // (op 3222) 
s_cmp_eq_i32 s7, 0 // (op 3222) 
s_mov_b32 s15, scc // (op 3222) 

s_cmp_eq_i32 s15, 0 // (op 3222) 
s_cselect_b32 s7, s6, s8 // (op 3222) 


s_mul_hi_u32 s8, s7, s12 // (op 3222) 
s_sub_u32 s15, s7, s8 // (op 3222) 
s_lshr_b32 s18, s15, 1 // (op 3222) 

s_add_u32 s15, s18, s8 // (op 3222) 


s_lshr_b32 s8, s15, s10 // (op 3222) 

s_and_b32 s15, s10, 2147483648 // (op 3222) 
s_cmp_eq_i32 s15, 0 // (op 3222) 
s_mov_b32 s18, scc // (op 3222) 

s_cmp_eq_i32 s18, 0 // (op 3222) 
s_cselect_b32 s15, s7, s8 // (op 3222) 


s_mul_i32 s8, s15, s14 // (op 3222) 

s_sub_u32 s15, s7, s8 // (op 3222) 

v_lshl_add_u32 v12, s15, 5, v13 // (op 3222) 


v_mul_lo_u32 v13, v12, s36 // (op 3222) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 3222) 
s_mul_i32 s8, s7, s18 // (op 3222) 


s_sub_u32 s7, s6, s8 // (op 3222) 


v_lshl_add_u32 v12, s7, 7, v11 // (op 3222) 


v_add_u32 v11, v12, v13 // (op 3222) 


s_mul_i32 s19, s36, s23 // (op 3224) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 3224) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 3224) least significant
s_mul_hi_u32 s6, s36, s22 // (op 3224) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3224) most significant: sum
s_add_u32 s19, s19, s6 // (op 3224) most significant: sum


s_mov_b32 s72, s38 // (op 3224) reg expression
s_mov_b32 s73, s39 // (op 3224) reg expression
s_mov_b32 s74, s18 // (op 3224) reg expression
s_mov_b32 s75, 131072 // (op 3224) call()

v_lshrrev_b32 v12, 2, v1 // (op 3331) 
s_mul_i32 s15, s16, s50 // (op 3331) 
s_add_u32 s15, s15, s0 // (op 3331) 
v_lshlrev_b32 v13, 1, v12 // (op 3331) 

s_add_u32 s6, s15, s9 // (op 3331) 

v_and_b32 v12, -4, v1 // (op 3331) 
s_mul_hi_u32 s15, s6, s13 // (op 3331) 
v_sub_u32 v14, v1, v12 // (op 3331) 

s_sub_u32 s7, s6, s15 // (op 3331) 
s_lshr_b32 s8, s7, 1 // (op 3331) 

s_add_u32 s7, s8, s15 // (op 3331) 


s_lshr_b32 s8, s7, s11 // (op 3331) 

s_and_b32 s7, s11, 2147483648 // (op 3331) 
s_cmp_eq_i32 s7, 0 // (op 3331) 
s_mov_b32 s15, scc // (op 3331) 

s_cmp_eq_i32 s15, 0 // (op 3331) 
s_cselect_b32 s7, s6, s8 // (op 3331) 


s_mul_hi_u32 s8, s7, s12 // (op 3331) 
s_sub_u32 s15, s7, s8 // (op 3331) 
s_lshr_b32 s18, s15, 1 // (op 3331) 

s_add_u32 s15, s18, s8 // (op 3331) 


s_lshr_b32 s8, s15, s10 // (op 3331) 

s_and_b32 s15, s10, 2147483648 // (op 3331) 
s_cmp_eq_i32 s15, 0 // (op 3331) 
s_mov_b32 s18, scc // (op 3331) 

s_cmp_eq_i32 s18, 0 // (op 3331) 
s_cselect_b32 s15, s7, s8 // (op 3331) 


v_lshl_add_u32 v12, s15, 7, v13 // (op 3331) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3331) 
s_mul_i32 s8, s7, s18 // (op 3331) 


s_sub_u32 s7, s6, s8 // (op 3331) 


v_lshl_add_u32 v13, s7, 2, v14 // (op 3331) 


v_mul_lo_u32 v14, v12, s40 // (op 3331) 
v_add_u32 v14, v14, v13 // (op 3331) 


s_mul_i32 s19, s40, s35 // (op 3333) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 3333) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 3333) least significant
s_mul_hi_u32 s6, s40, s34 // (op 3333) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3333) most significant: sum
s_add_u32 s19, s19, s6 // (op 3333) most significant: sum


s_mov_b32 s76, s42 // (op 3333) reg expression
s_mov_b32 s77, s43 // (op 3333) reg expression
s_mov_b32 s78, s18 // (op 3333) reg expression
s_mov_b32 s79, 131072 // (op 3333) call()

v_lshrrev_b32 v12, 2, v1 // (op 3350) 
s_mul_i32 s15, s16, s50 // (op 3350) 
s_add_u32 s15, s15, s0 // (op 3350) 
v_lshlrev_b32 v13, 1, v12 // (op 3350) 

s_add_u32 s6, s15, s9 // (op 3350) 

v_and_b32 v12, -4, v1 // (op 3350) 
s_mul_hi_u32 s15, s6, s13 // (op 3350) 
v_sub_u32 v15, v1, v12 // (op 3350) 

s_sub_u32 s7, s6, s15 // (op 3350) 
s_lshr_b32 s8, s7, 1 // (op 3350) 

s_add_u32 s7, s8, s15 // (op 3350) 


s_lshr_b32 s8, s7, s11 // (op 3350) 

s_and_b32 s7, s11, 2147483648 // (op 3350) 
s_cmp_eq_i32 s7, 0 // (op 3350) 
s_mov_b32 s15, scc // (op 3350) 

s_cmp_eq_i32 s15, 0 // (op 3350) 
s_cselect_b32 s7, s6, s8 // (op 3350) 


s_mul_hi_u32 s8, s7, s12 // (op 3350) 
s_sub_u32 s15, s7, s8 // (op 3350) 
s_lshr_b32 s18, s15, 1 // (op 3350) 

s_add_u32 s15, s18, s8 // (op 3350) 


s_lshr_b32 s8, s15, s10 // (op 3350) 

s_and_b32 s15, s10, 2147483648 // (op 3350) 
s_cmp_eq_i32 s15, 0 // (op 3350) 
s_mov_b32 s18, scc // (op 3350) 

s_cmp_eq_i32 s18, 0 // (op 3350) 
s_cselect_b32 s15, s7, s8 // (op 3350) 


v_lshl_add_u32 v12, s15, 7, v13 // (op 3350) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3350) 
s_mul_i32 s8, s7, s18 // (op 3350) 


s_sub_u32 s7, s6, s8 // (op 3350) 


v_lshl_add_u32 v13, s7, 2, v15 // (op 3350) 


v_mul_lo_u32 v15, v12, s40 // (op 3350) 
v_add_u32 v15, v15, v13 // (op 3350) 


s_mul_i32 s19, s40, s35 // (op 3352) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 3352) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 3352) least significant
s_mul_hi_u32 s6, s40, s34 // (op 3352) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3352) most significant: sum
s_add_u32 s19, s19, s6 // (op 3352) most significant: sum


s_mov_b32 s76, s42 // (op 3352) reg expression
s_mov_b32 s77, s43 // (op 3352) reg expression
s_mov_b32 s78, s18 // (op 3352) reg expression
s_mov_b32 s79, 131072 // (op 3352) call()

v_lshrrev_b32 v12, 6, v1 // (op 3459) 
s_mul_i32 s15, s16, s50 // (op 3459) 
s_add_u32 s15, s15, s0 // (op 3459) 
v_and_b32 v13, 1, v12 // (op 3459) 

s_add_u32 s6, s15, s9 // (op 3459) 

v_and_b32 v12, 15, v1 // (op 3459) 
s_mul_hi_u32 s15, s6, s13 // (op 3459) 
v_lshl_add_u32 v16, v13, 4, v12 // (op 3459) 


s_sub_u32 s7, s6, s15 // (op 3459) 
v_and_b32 v12, 63, v1 // (op 3459) 
s_lshr_b32 s8, s7, 1 // (op 3459) 

v_lshrrev_b32 v13, 4, v12 // (op 3459) 

s_add_u32 s7, s8, s15 // (op 3459) 


s_lshr_b32 s8, s7, s11 // (op 3459) 

s_and_b32 s7, s11, 2147483648 // (op 3459) 
s_cmp_eq_i32 s7, 0 // (op 3459) 
s_mov_b32 s15, scc // (op 3459) 

s_cmp_eq_i32 s15, 0 // (op 3459) 
s_cselect_b32 s7, s6, s8 // (op 3459) 


s_mul_hi_u32 s8, s7, s12 // (op 3459) 
s_sub_u32 s15, s7, s8 // (op 3459) 
s_lshr_b32 s18, s15, 1 // (op 3459) 

s_add_u32 s15, s18, s8 // (op 3459) 


s_lshr_b32 s8, s15, s10 // (op 3459) 

s_and_b32 s15, s10, 2147483648 // (op 3459) 
s_cmp_eq_i32 s15, 0 // (op 3459) 
s_mov_b32 s18, scc // (op 3459) 

s_cmp_eq_i32 s18, 0 // (op 3459) 
s_cselect_b32 s15, s7, s8 // (op 3459) 


s_mul_i32 s8, s15, s14 // (op 3459) 

s_sub_u32 s15, s7, s8 // (op 3459) 

v_lshl_add_u32 v12, s15, 5, v16 // (op 3459) 


v_mul_lo_u32 v16, v12, s44 // (op 3459) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 3459) 
s_mul_i32 s8, s7, s18 // (op 3459) 


s_sub_u32 s7, s6, s8 // (op 3459) 


v_lshl_add_u32 v12, s7, 2, v13 // (op 3459) 


v_add_u32 v13, v12, v16 // (op 3459) 


s_mul_i32 s19, s44, s23 // (op 3461) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 3461) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 3461) least significant
s_mul_hi_u32 s6, s44, s22 // (op 3461) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3461) most significant: sum
s_add_u32 s19, s19, s6 // (op 3461) most significant: sum


s_mov_b32 s80, s46 // (op 3461) reg expression
s_mov_b32 s81, s47 // (op 3461) reg expression
s_mov_b32 s82, s18 // (op 3461) reg expression
s_mov_b32 s83, 131072 // (op 3461) call()

v_lshrrev_b32 v12, 6, v1 // (op 3478) 
s_mul_i32 s15, s16, s50 // (op 3478) 
s_add_u32 s15, s15, s0 // (op 3478) 
v_and_b32 v16, 1, v12 // (op 3478) 

s_add_u32 s6, s15, s9 // (op 3478) 

v_and_b32 v12, 15, v1 // (op 3478) 
s_mul_hi_u32 s15, s6, s13 // (op 3478) 
v_lshl_add_u32 v17, v16, 4, v12 // (op 3478) 


s_sub_u32 s7, s6, s15 // (op 3478) 
v_and_b32 v12, 63, v1 // (op 3478) 
s_lshr_b32 s8, s7, 1 // (op 3478) 

v_lshrrev_b32 v16, 4, v12 // (op 3478) 

s_add_u32 s7, s8, s15 // (op 3478) 


s_lshr_b32 s8, s7, s11 // (op 3478) 

s_and_b32 s7, s11, 2147483648 // (op 3478) 
s_cmp_eq_i32 s7, 0 // (op 3478) 
s_mov_b32 s15, scc // (op 3478) 

s_cmp_eq_i32 s15, 0 // (op 3478) 
s_cselect_b32 s7, s6, s8 // (op 3478) 


s_mul_hi_u32 s8, s7, s12 // (op 3478) 
s_sub_u32 s15, s7, s8 // (op 3478) 
s_lshr_b32 s18, s15, 1 // (op 3478) 

s_add_u32 s15, s18, s8 // (op 3478) 


s_lshr_b32 s8, s15, s10 // (op 3478) 

s_and_b32 s15, s10, 2147483648 // (op 3478) 
s_cmp_eq_i32 s15, 0 // (op 3478) 
s_mov_b32 s18, scc // (op 3478) 

s_cmp_eq_i32 s18, 0 // (op 3478) 
s_cselect_b32 s15, s7, s8 // (op 3478) 


s_mul_i32 s8, s15, s14 // (op 3478) 

s_sub_u32 s15, s7, s8 // (op 3478) 

v_lshl_add_u32 v12, s15, 5, v17 // (op 3478) 


v_mul_lo_u32 v17, v12, s44 // (op 3478) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 3478) 
s_mul_i32 s8, s7, s18 // (op 3478) 


s_sub_u32 s7, s6, s8 // (op 3478) 


v_lshl_add_u32 v12, s7, 2, v16 // (op 3478) 


v_add_u32 v16, v12, v17 // (op 3478) 


s_mul_i32 s19, s44, s23 // (op 3480) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 3480) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 3480) least significant
s_mul_hi_u32 s6, s44, s22 // (op 3480) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3480) most significant: sum
s_add_u32 s19, s19, s6 // (op 3480) most significant: sum


s_mov_b32 s80, s46 // (op 3480) reg expression
s_mov_b32 s81, s47 // (op 3480) reg expression
s_mov_b32 s82, s18 // (op 3480) reg expression
s_mov_b32 s83, 131072 // (op 3480) call()




buffer_load_ubyte v12, v13, s[80:83], 0 offen offset: 0 // (op 2647) Load value


v_mov_b32 v17, v0 // (op 2653) reg expression

buffer_load_dwordx4 v[18:21], v17, s[68:71], 0 offen // (op 2653) Load value




buffer_load_dwordx4 v[22:25], v10, s[72:75], 0 offen // (op 2659) Load value
buffer_load_dwordx4 v[26:29], v10, s[72:75], 0 offen offset:64 // (op 2659) Load value


s_lshl_b64 s[18:19], s[30:31], 5 // (op 2665) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2665) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2665) 

v_add_u32 v17, v0, s18 // (op 2665) 


buffer_load_dwordx4 v[30:33], v17, s[68:71], 0 offen // (op 2665) Load value



s_lshl_b64 s[18:19], s[30:31], 5 // (op 2672) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2672) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2672) 

s_lshl_b32 s15, s18, 1 // (op 2672) 

v_add_u32 v17, v0, s15 // (op 2672) 


buffer_load_dwordx4 v[34:37], v17, s[68:71], 0 offen // (op 2672) Load value



s_lshl_b64 s[18:19], s[30:31], 5 // (op 2679) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2679) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2679) 

s_mul_i32 s15, 3, s18 // (op 2679) 

v_add_u32 v17, v0, s15 // (op 2679) 


buffer_load_dwordx4 v[38:41], v17, s[68:71], 0 offen // (op 2679) Load value


v_mov_b64 v[42:43], s[40:41] // (op 2684) convert

v_mov_b32 v17, v14 // (op 2684) 
buffer_load_ubyte v44, v17, s[76:79], 0 offen offset: 0 // (op 2684) Load value
v_add_u32 v17, v17, v42 // (op 2684) 
buffer_load_ubyte v45, v17, s[76:79], 0 offen offset: 0 // (op 2684) Load value





buffer_load_ubyte v17, v13, s[80:83], 0 offen offset:4 // (op 2688) Load value



v_add_u32 v13, 64, v0 // (op 2694) 

buffer_load_dwordx4 v[46:49], v13, s[68:71], 0 offen // (op 2694) Load value




buffer_load_dwordx4 v[50:53], v10, s[72:75], 0 offen offset:128 // (op 2700) Load value
buffer_load_dwordx4 v[54:57], v10, s[72:75], 0 offen offset:192 // (op 2700) Load value



s_lshl_b64 s[18:19], s[30:31], 5 // (op 2706) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2706) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2706) 

s_add_u32 s15, s18, 64 // (op 2706) 

v_add_u32 v10, v0, s15 // (op 2706) 


buffer_load_dwordx4 v[58:61], v10, s[68:71], 0 offen // (op 2706) Load value



s_lshl_b64 s[18:19], s[30:31], 5 // (op 2713) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2713) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2713) 

s_lshl_b32 s15, s18, 1 // (op 2713) 
s_add_u32 s15, s15, 64 // (op 2713) 

v_add_u32 v10, v0, s15 // (op 2713) 


buffer_load_dwordx4 v[62:65], v10, s[68:71], 0 offen // (op 2713) Load value



s_lshl_b64 s[18:19], s[30:31], 5 // (op 2720) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2720) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2720) 

s_mul_i32 s15, 3, s18 // (op 2720) 
s_add_u32 s15, s15, 64 // (op 2720) 

v_add_u32 v10, v0, s15 // (op 2720) 


buffer_load_dwordx4 v[66:69], v10, s[68:71], 0 offen // (op 2720) Load value



v_mov_b64 v[42:43], s[40:41] // (op 2725) convert

v_mov_b32 v0, v14 // (op 2725) 
buffer_load_ubyte v70, v0, s[76:79], 0 offen offset:4 // (op 2725) Load value
v_add_u32 v0, v0, v42 // (op 2725) 
buffer_load_ubyte v71, v0, s[76:79], 0 offen offset:4 // (op 2725) Load value



v_and_b32 v0, -4, v1 // (op 3713) 
v_sub_u32 v10, v1, v0 // (op 3713) 

v_lshrrev_b32 v0, 2, v1 // (op 3713) 
v_lshl_add_u32 v13, v0, 3, v10 // (op 3713) 


s_barrier  // (op 3941) 



s_waitcnt vmcnt(10)
ds_write_b8 v13, v44 // (op 2727) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v13, v45 offset:4 // (op 2727) Store local data 

v_and_b32 v0, -4, v1 // (op 3684) 
v_sub_u32 v10, v1, v0 // (op 3684) 

v_lshrrev_b32 v0, 2, v1 // (op 3684) 
v_lshl_add_u32 v13, v0, 3, v10 // (op 3684) 


v_and_b32 v0, -4, v1 // (op 3784) 
v_sub_u32 v10, v1, v0 // (op 3784) 

v_lshrrev_b32 v0, 2, v1 // (op 3784) 
v_lshl_add_u32 v14, v0, 3, v10 // (op 3784) 


s_mov_b32 s1, 0 // (op 758) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 757) 
s_mov_b32 s15, s18 // (op 757) convert

s_and_b32 s6, s15, 4294967294 // (op 757) 

s_cmp_lt_u32 s1, s6 // (op 757) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_757 // (op 757) Condition: Top (jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_757 if false)
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_757:
s_add_u32 s1, s1, 2 // (op 759) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3938) 



ds_read_u8 v0, v3 // (op 160) Load local data 



ds_read_u8 v10, v3 offset:128 // (op 539) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[22:29], a[0:3], v0, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 163) 




ds_read_u8 v0, v3 offset:256 // (op 661) Load local data 



ds_read_u8 v42, v3 offset:384 // (op 783) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:33], v[22:29], a[4:7], v10, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 541) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[34:37], v[22:29], a[8:11], v0, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 663) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[38:41], v[22:29], a[12:15], v42, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 785) 



v_mov_b32 v0, v16 // (op 162) 

buffer_load_ubyte v12, v0, s[80:83], 0 offen offset:8 // (op 162) Load value


v_mov_b32 v10, v5 // (op 165) 
v_add_u32 v42, 128, v10 // (op 165) 

buffer_load_dwordx4 v[18:21], v42, s[68:71], 0 offen // (op 165) Load value



v_mov_b32 v42, v11 // (op 167) 

buffer_load_dwordx4 v[22:25], v42, s[72:75], 0 offen offset:256 // (op 167) Load value
buffer_load_dwordx4 v[26:29], v42, s[72:75], 0 offen offset:320 // (op 167) Load value


s_lshl_b64 s[18:19], s[30:31], 5 // (op 564) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 564) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 564) 

s_add_u32 s15, s18, 128 // (op 564) 

v_add_u32 v43, v10, s15 // (op 564) 


buffer_load_dwordx4 v[30:33], v43, s[68:71], 0 offen // (op 564) Load value



s_lshl_b64 s[18:19], s[30:31], 5 // (op 686) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 686) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 686) 

s_lshl_b32 s15, s18, 1 // (op 686) 
s_add_u32 s15, s15, 128 // (op 686) 

v_add_u32 v43, v10, s15 // (op 686) 


buffer_load_dwordx4 v[34:37], v43, s[68:71], 0 offen // (op 686) Load value



s_lshl_b64 s[18:19], s[30:31], 5 // (op 808) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 808) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 808) 

s_mul_i32 s15, 3, s18 // (op 808) 
s_add_u32 s15, s15, 128 // (op 808) 

v_add_u32 v43, v10, s15 // (op 808) 


buffer_load_dwordx4 v[38:41], v43, s[68:71], 0 offen // (op 808) Load value


v_mov_b64 v[72:73], s[40:41] // (op 157) convert

v_mov_b32 v43, v15 // (op 157) 
v_mov_b32 v74, v43 // (op 157) 
buffer_load_ubyte v44, v74, s[76:79], 0 offen offset:8 // (op 157) Load value
v_add_u32 v74, v74, v72 // (op 157) 
buffer_load_ubyte v45, v74, s[76:79], 0 offen offset:8 // (op 157) Load value


s_barrier  // (op 3957) 



s_waitcnt vmcnt(10)
ds_write_b8 v14, v70 offset:512 // (op 452) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v14, v71 offset:516 // (op 452) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3954) 



ds_read_u8 v72, v4 offset:512 // (op 454) Load local data 



ds_read_u8 v73, v4 offset:640 // (op 548) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:49], v[50:57], a[0:3], v72, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 457) 




ds_read_u8 v72, v4 offset:768 // (op 670) Load local data 



ds_read_u8 v74, v4 offset:896 // (op 792) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:61], v[50:57], a[4:7], v73, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 542) 

s_barrier  // (op 4064) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[50:57], a[8:11], v72, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 664) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[50:57], a[12:15], v74, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 786) 




buffer_load_ubyte v17, v0, s[80:83], 0 offen offset:12 // (op 456) Load value



v_add_u32 v16, 8, v16 // (op 3470) 
v_add_u32 v0, 192, v10 // (op 459) 

buffer_load_dwordx4 v[46:49], v0, s[68:71], 0 offen // (op 459) Load value




buffer_load_dwordx4 v[50:53], v42, s[72:75], 0 offen offset:384 // (op 461) Load value
buffer_load_dwordx4 v[54:57], v42, s[72:75], 0 offen offset:448 // (op 461) Load value



v_add_u32 v11, 256, v11 // (op 3214) 
s_lshl_b64 s[18:19], s[30:31], 5 // (op 557) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 557) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 557) 

s_add_u32 s15, s18, 192 // (op 557) 

v_add_u32 v0, v10, s15 // (op 557) 


buffer_load_dwordx4 v[58:61], v0, s[68:71], 0 offen // (op 557) Load value



s_lshl_b64 s[18:19], s[30:31], 5 // (op 679) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 679) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 679) 

s_lshl_b32 s15, s18, 1 // (op 679) 
s_add_u32 s15, s15, 192 // (op 679) 

v_add_u32 v0, v10, s15 // (op 679) 


buffer_load_dwordx4 v[62:65], v0, s[68:71], 0 offen // (op 679) Load value



s_lshl_b64 s[18:19], s[30:31], 5 // (op 801) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 801) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 801) 

s_mul_i32 s15, 3, s18 // (op 801) 
s_add_u32 s15, s15, 192 // (op 801) 

v_add_u32 v0, v10, s15 // (op 801) 


buffer_load_dwordx4 v[66:69], v0, s[68:71], 0 offen // (op 801) Load value



v_add_u32 v5, 128, v5 // (op 3065) 
v_mov_b64 v[72:73], s[40:41] // (op 451) convert

v_mov_b32 v0, v43 // (op 451) 
buffer_load_ubyte v70, v0, s[76:79], 0 offen offset:12 // (op 451) Load value
v_add_u32 v0, v0, v72 // (op 451) 
buffer_load_ubyte v71, v0, s[76:79], 0 offen offset:12 // (op 451) Load value



s_barrier  // (op 3944) 
v_add_u32 v15, 8, v15 // (op 3342) 



s_waitcnt vmcnt(10)
ds_write_b8 v13, v44 // (op 158) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v13, v45 offset:4 // (op 158) Store local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 757) 
s_mov_b32 s15, s18 // (op 757) convert

s_and_b32 s6, s15, 4294967294 // (op 757) 

s_cmp_lt_u32 s1, s6 // (op 757) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_757 // (op 757) Condition: Bottom (jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_757 if true)
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_757:













v_lshrrev_b32 v0, 7, v1 // (op 2988) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2988) 
v_and_b32 v5, 15, v1 // (op 2988) 
s_and_b32 s15, s18, 4294967294 // (op 2988) 
v_lshl_add_u32 v10, v0, 4, v5 // (op 2988) 


s_add_u32 s6, s0, s15 // (op 2988) 

v_and_b32 v0, 63, v1 // (op 2988) 
v_lshrrev_b32 v5, 4, v0 // (op 2988) 

s_mul_i32 s15, s16, s50 // (op 2988) 
s_add_u32 s15, s15, s6 // (op 2988) 

v_lshlrev_b32 v0, 5, v5 // (op 2988) 

s_add_u32 s6, s15, s9 // (op 2988) 

s_mul_hi_u32 s15, s6, s13 // (op 2988) 
s_sub_u32 s7, s6, s15 // (op 2988) 
s_lshr_b32 s8, s7, 1 // (op 2988) 

s_add_u32 s7, s8, s15 // (op 2988) 


s_lshr_b32 s8, s7, s11 // (op 2988) 

s_and_b32 s7, s11, 2147483648 // (op 2988) 
s_cmp_eq_i32 s7, 0 // (op 2988) 
s_mov_b32 s15, scc // (op 2988) 

s_cmp_eq_i32 s15, 0 // (op 2988) 
s_cselect_b32 s7, s6, s8 // (op 2988) 


s_mul_hi_u32 s8, s7, s12 // (op 2988) 
s_sub_u32 s15, s7, s8 // (op 2988) 
s_lshr_b32 s28, s15, 1 // (op 2988) 

s_add_u32 s15, s28, s8 // (op 2988) 


s_lshr_b32 s8, s15, s10 // (op 2988) 

s_and_b32 s15, s10, 2147483648 // (op 2988) 
s_cmp_eq_i32 s15, 0 // (op 2988) 
s_mov_b32 s28, scc // (op 2988) 

s_cmp_eq_i32 s28, 0 // (op 2988) 
s_cselect_b32 s15, s7, s8 // (op 2988) 


s_mul_i32 s8, s7, s18 // (op 2988) 


v_lshl_add_u32 v5, s15, 7, v10 // (op 2988) 


s_sub_u32 s7, s6, s8 // (op 2988) 


v_lshl_add_u32 v10, s7, 7, v0 // (op 2988) 


v_mov_b32 v19, 0 // (op 2988) convert
v_mov_b32 v18, v10 // (op 2988) convert

v_mul_lo_u32 v11, v5, s31 // (op 2988) most significant: low of low * high
v_mul_hi_u32 v0, v5, s30 // (op 2988) most significant: high of low * low
v_mul_lo_u32 v10, v5, s30 // (op 2988) least significant: low of low * low
v_add_u32 v11, v11, v0 // (op 2988) most significant: sum

v_add_co_u32 v10, s[18:19], v10, v18 // (op 2988) least significant half
v_addc_co_u32 v11, s[18:19], v11, v19, s[18:19] // (op 2988) most significant half



v_lshlrev_b64 v[12:13], 2, v[10:11] // (op 2988) 

v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 2988) 

v_mov_b32 v0, v10 // (op 2988) convert

s_mul_i32 s19, s30, s35 // (op 2990) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 2990) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 2990) least significant
s_mul_hi_u32 s6, s30, s34 // (op 2990) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2990) most significant: sum
s_add_u32 s19, s19, s6 // (op 2990) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 2990) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 2990) 

s_mov_b32 s68, s32 // (op 2990) reg expression
s_mov_b32 s69, s33 // (op 2990) reg expression
s_mov_b32 s70, s18 // (op 2990) reg expression
s_mov_b32 s71, 131072 // (op 2990) call()

v_lshrrev_b32 v5, 6, v1 // (op 3158) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 3158) 
v_and_b32 v10, 1, v5 // (op 3158) 

s_and_b32 s15, s18, 4294967294 // (op 3158) 
v_and_b32 v5, 15, v1 // (op 3158) 
s_add_u32 s6, s0, s15 // (op 3158) 

v_lshl_add_u32 v11, v10, 4, v5 // (op 3158) 


v_and_b32 v5, 48, v1 // (op 3158) 
s_mul_i32 s15, s16, s50 // (op 3158) 
s_add_u32 s15, s15, s6 // (op 3158) 

s_add_u32 s6, s15, s9 // (op 3158) 

s_mul_hi_u32 s15, s6, s13 // (op 3158) 
s_sub_u32 s7, s6, s15 // (op 3158) 
s_lshr_b32 s8, s7, 1 // (op 3158) 

s_add_u32 s7, s8, s15 // (op 3158) 


s_lshr_b32 s8, s7, s11 // (op 3158) 

s_and_b32 s7, s11, 2147483648 // (op 3158) 
s_cmp_eq_i32 s7, 0 // (op 3158) 
s_mov_b32 s15, scc // (op 3158) 

s_cmp_eq_i32 s15, 0 // (op 3158) 
s_cselect_b32 s7, s6, s8 // (op 3158) 


s_mul_hi_u32 s8, s7, s12 // (op 3158) 
s_sub_u32 s15, s7, s8 // (op 3158) 
s_lshr_b32 s28, s15, 1 // (op 3158) 

s_add_u32 s15, s28, s8 // (op 3158) 


s_lshr_b32 s8, s15, s10 // (op 3158) 

s_and_b32 s15, s10, 2147483648 // (op 3158) 
s_cmp_eq_i32 s15, 0 // (op 3158) 
s_mov_b32 s28, scc // (op 3158) 

s_cmp_eq_i32 s28, 0 // (op 3158) 
s_cselect_b32 s15, s7, s8 // (op 3158) 


s_mul_i32 s8, s15, s14 // (op 3158) 

s_sub_u32 s15, s7, s8 // (op 3158) 

v_lshl_add_u32 v10, s15, 5, v11 // (op 3158) 


s_mul_i32 s8, s7, s18 // (op 3158) 


v_mul_lo_u32 v11, v10, s36 // (op 3158) 

s_sub_u32 s7, s6, s8 // (op 3158) 


v_lshl_add_u32 v10, s7, 7, v5 // (op 3158) 


v_add_u32 v5, v10, v11 // (op 3158) 


s_mul_i32 s19, s36, s23 // (op 3160) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 3160) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 3160) least significant
s_mul_hi_u32 s6, s36, s22 // (op 3160) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3160) most significant: sum
s_add_u32 s19, s19, s6 // (op 3160) most significant: sum


s_mov_b32 s72, s38 // (op 3160) reg expression
s_mov_b32 s73, s39 // (op 3160) reg expression
s_mov_b32 s74, s18 // (op 3160) reg expression
s_mov_b32 s75, 131072 // (op 3160) call()

v_lshrrev_b32 v10, 2, v1 // (op 3286) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 3286) 
v_lshlrev_b32 v11, 1, v10 // (op 3286) 

s_and_b32 s15, s18, 4294967294 // (op 3286) 
v_and_b32 v10, -4, v1 // (op 3286) 
s_add_u32 s6, s0, s15 // (op 3286) 

v_sub_u32 v12, v1, v10 // (op 3286) 

s_mul_i32 s15, s16, s50 // (op 3286) 
s_add_u32 s15, s15, s6 // (op 3286) 

s_add_u32 s6, s15, s9 // (op 3286) 

s_mul_hi_u32 s15, s6, s13 // (op 3286) 
s_sub_u32 s7, s6, s15 // (op 3286) 
s_lshr_b32 s8, s7, 1 // (op 3286) 

s_add_u32 s7, s8, s15 // (op 3286) 


s_lshr_b32 s8, s7, s11 // (op 3286) 

s_and_b32 s7, s11, 2147483648 // (op 3286) 
s_cmp_eq_i32 s7, 0 // (op 3286) 
s_mov_b32 s15, scc // (op 3286) 

s_cmp_eq_i32 s15, 0 // (op 3286) 
s_cselect_b32 s7, s6, s8 // (op 3286) 


s_mul_hi_u32 s8, s7, s12 // (op 3286) 
s_sub_u32 s15, s7, s8 // (op 3286) 
s_lshr_b32 s28, s15, 1 // (op 3286) 

s_add_u32 s15, s28, s8 // (op 3286) 


s_lshr_b32 s8, s15, s10 // (op 3286) 

s_and_b32 s15, s10, 2147483648 // (op 3286) 
s_cmp_eq_i32 s15, 0 // (op 3286) 
s_mov_b32 s28, scc // (op 3286) 

s_cmp_eq_i32 s28, 0 // (op 3286) 
s_cselect_b32 s15, s7, s8 // (op 3286) 


s_mul_i32 s8, s7, s18 // (op 3286) 


v_lshl_add_u32 v10, s15, 7, v11 // (op 3286) 


s_sub_u32 s7, s6, s8 // (op 3286) 


v_lshl_add_u32 v11, s7, 2, v12 // (op 3286) 


v_mul_lo_u32 v12, v10, s40 // (op 3286) 
v_add_u32 v12, v12, v11 // (op 3286) 


s_mul_i32 s19, s40, s35 // (op 3288) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 3288) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 3288) least significant
s_mul_hi_u32 s6, s40, s34 // (op 3288) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3288) most significant: sum
s_add_u32 s19, s19, s6 // (op 3288) most significant: sum


s_mov_b32 s76, s42 // (op 3288) reg expression
s_mov_b32 s77, s43 // (op 3288) reg expression
s_mov_b32 s78, s18 // (op 3288) reg expression
s_mov_b32 s79, 131072 // (op 3288) call()

v_lshrrev_b32 v10, 6, v1 // (op 3414) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 3414) 
v_and_b32 v11, 1, v10 // (op 3414) 

s_and_b32 s15, s18, 4294967294 // (op 3414) 
v_and_b32 v10, 15, v1 // (op 3414) 
s_add_u32 s6, s0, s15 // (op 3414) 

v_lshl_add_u32 v13, v11, 4, v10 // (op 3414) 


v_and_b32 v10, 63, v1 // (op 3414) 
s_mul_i32 s15, s16, s50 // (op 3414) 
s_add_u32 s15, s15, s6 // (op 3414) 

v_lshrrev_b32 v11, 4, v10 // (op 3414) 

s_add_u32 s6, s15, s9 // (op 3414) 

s_mul_hi_u32 s15, s6, s13 // (op 3414) 
s_sub_u32 s7, s6, s15 // (op 3414) 
s_lshr_b32 s8, s7, 1 // (op 3414) 

s_add_u32 s7, s8, s15 // (op 3414) 


s_lshr_b32 s8, s7, s11 // (op 3414) 

s_and_b32 s7, s11, 2147483648 // (op 3414) 
s_cmp_eq_i32 s7, 0 // (op 3414) 
s_mov_b32 s15, scc // (op 3414) 

s_cmp_eq_i32 s15, 0 // (op 3414) 
s_cselect_b32 s7, s6, s8 // (op 3414) 


s_mul_hi_u32 s8, s7, s12 // (op 3414) 
s_sub_u32 s15, s7, s8 // (op 3414) 
s_lshr_b32 s28, s15, 1 // (op 3414) 

s_add_u32 s15, s28, s8 // (op 3414) 


s_lshr_b32 s8, s15, s10 // (op 3414) 

s_and_b32 s15, s10, 2147483648 // (op 3414) 
s_cmp_eq_i32 s15, 0 // (op 3414) 
s_mov_b32 s28, scc // (op 3414) 

s_cmp_eq_i32 s28, 0 // (op 3414) 
s_cselect_b32 s15, s7, s8 // (op 3414) 


s_mul_i32 s8, s15, s14 // (op 3414) 

s_sub_u32 s15, s7, s8 // (op 3414) 

v_lshl_add_u32 v10, s15, 5, v13 // (op 3414) 


s_mul_i32 s8, s7, s18 // (op 3414) 


v_mul_lo_u32 v13, v10, s44 // (op 3414) 

s_sub_u32 s7, s6, s8 // (op 3414) 


v_lshl_add_u32 v10, s7, 2, v11 // (op 3414) 


v_add_u32 v11, v10, v13 // (op 3414) 


s_mul_i32 s19, s44, s23 // (op 3416) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 3416) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 3416) least significant
s_mul_hi_u32 s6, s44, s22 // (op 3416) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3416) most significant: sum
s_add_u32 s19, s19, s6 // (op 3416) most significant: sum


s_mov_b32 s80, s46 // (op 3416) reg expression
s_mov_b32 s81, s47 // (op 3416) reg expression
s_mov_b32 s82, s18 // (op 3416) reg expression
s_mov_b32 s83, 131072 // (op 3416) call()

v_and_b32 v10, -4, v1 // (op 3697) 
v_sub_u32 v13, v1, v10 // (op 3697) 

v_lshrrev_b32 v10, 2, v1 // (op 3697) 
v_lshl_add_u32 v14, v10, 3, v13 // (op 3697) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 763) 
s_and_b32 s15, s18, 4294967294 // (op 763) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 762) 
s_mov_b32 s6, s18 // (op 762) convert

s_cmp_lt_u32 s15, s6 // (op 762) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_762 // (op 762) Condition: Top (jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_762 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_762:

s_add_u32 s15, s15, 1 // (op 764) 
v_mov_b64 v[16:17], s[40:41] // (op 409) convert

v_mov_b32 v10, v12 // (op 409) 
v_mov_b32 v13, v10 // (op 409) 
buffer_load_ubyte v18, v13, s[76:79], 0 offen offset: 0 // (op 409) Load value
v_add_u32 v13, v13, v16 // (op 409) 
buffer_load_ubyte v19, v13, s[76:79], 0 offen offset: 0 // (op 409) Load value



v_add_u32 v12, 4, v12 // (op 3278) 


s_barrier  // (op 3947) 
v_mov_b32 v10, v0 // (op 417) 



v_mov_b32 v13, v10 // (op 417) reg expression

buffer_load_dwordx4 v[20:23], v13, s[68:71], 0 offen // (op 417) Load value

s_waitcnt vmcnt(2)
ds_write_b8 v14, v18 // (op 410) Store local data 


v_mov_b32 v13, v5 // (op 419) 

buffer_load_dwordx4 v[24:27], v13, s[72:75], 0 offen // (op 419) Load value
buffer_load_dwordx4 v[28:31], v13, s[72:75], 0 offen offset:64 // (op 419) Load value

v_add_u32 v5, 128, v5 // (op 3150) 
s_waitcnt vmcnt(3)
ds_write_b8 v14, v19 offset:4 // (op 410) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 3950) 



ds_read_u8 v13, v3 // (op 412) Load local data 


v_mov_b32 v15, v11 // (op 414) 

buffer_load_ubyte v16, v15, s[80:83], 0 offen offset: 0 // (op 414) Load value

v_add_u32 v11, 4, v11 // (op 3406) 


s_lshl_b64 s[18:19], s[30:31], 5 // (op 534) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 534) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 534) 

v_add_u32 v15, v10, s18 // (op 534) 


s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:31], a[0:3], v13, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 415) 
buffer_load_dwordx4 v[32:35], v15, s[68:71], 0 offen // (op 534) Load value






ds_read_u8 v13, v3 offset:128 // (op 524) Load local data 


s_lshl_b64 s[18:19], s[30:31], 5 // (op 656) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 656) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 656) 

s_lshl_b32 s6, s18, 1 // (op 656) 

v_add_u32 v15, v10, s6 // (op 656) 


buffer_load_dwordx4 v[20:23], v15, s[68:71], 0 offen // (op 656) Load value

s_waitcnt vmcnt(1) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:35], v[24:31], a[4:7], v13, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 526) 





ds_read_u8 v13, v3 offset:256 // (op 646) Load local data 


s_lshl_b64 s[18:19], s[30:31], 5 // (op 778) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 778) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 778) 

s_mul_i32 s6, 3, s18 // (op 778) 

v_add_u32 v15, v10, s6 // (op 778) 


buffer_load_dwordx4 v[32:35], v15, s[68:71], 0 offen // (op 778) Load value

s_waitcnt vmcnt(1) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:23], v[24:31], a[8:11], v13, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 648) 






v_add_u32 v0, 64, v0 // (op 2976) 
ds_read_u8 v10, v3 offset:384 // (op 768) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[32:35], v[24:31], a[12:15], v10, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 770) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 762) 
s_mov_b32 s6, s18 // (op 762) convert

s_cmp_lt_u32 s15, s6 // (op 762) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_762 // (op 762) Condition: Bottom (jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_762 if true)
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_762:






s_barrier  // (op 4050) 
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 189) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_189 // (op 189) Condition: False, jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_189
v_accvgpr_read v10, a0 // (op 197) 
v_accvgpr_read v11, a1 // (op 197) 
v_accvgpr_read v12, a2 // (op 197) 
v_accvgpr_read v13, a3 // (op 197) 
v_mul_f32 v14, v7, v10 // (op 197) 
v_mul_f32 v15, v7, v11 // (op 197) 
v_mul_f32 v16, v7, v12 // (op 197) 
v_mul_f32 v17, v7, v13 // (op 197) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen // (op 199) Store value

v_accvgpr_read v10, a4 // (op 964) 
v_accvgpr_read v11, a5 // (op 964) 
v_accvgpr_read v12, a6 // (op 964) 
v_accvgpr_read v13, a7 // (op 964) 
v_mul_f32 v14, v7, v10 // (op 964) 
v_mul_f32 v15, v7, v11 // (op 964) 
v_mul_f32 v16, v7, v12 // (op 964) 
v_mul_f32 v17, v7, v13 // (op 964) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen offset:128 // (op 966) Store value

v_accvgpr_read v10, a8 // (op 982) 
v_accvgpr_read v11, a9 // (op 982) 
v_accvgpr_read v12, a10 // (op 982) 
v_accvgpr_read v13, a11 // (op 982) 
v_mul_f32 v14, v7, v10 // (op 982) 
v_mul_f32 v15, v7, v11 // (op 982) 
v_mul_f32 v16, v7, v12 // (op 982) 
v_mul_f32 v17, v7, v13 // (op 982) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen offset:256 // (op 984) Store value

v_accvgpr_read v10, a12 // (op 1000) 
v_accvgpr_read v11, a13 // (op 1000) 
v_accvgpr_read v12, a14 // (op 1000) 
v_accvgpr_read v13, a15 // (op 1000) 
v_mul_f32 v14, v7, v10 // (op 1000) 
v_mul_f32 v15, v7, v11 // (op 1000) 
v_mul_f32 v16, v7, v12 // (op 1000) 
v_mul_f32 v17, v7, v13 // (op 1000) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen offset:384 // (op 1002) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_189 // (op 189) Condition: Done, jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_189
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_189:



buffer_load_dwordx4 v[10:13], v8, s[56:59], 0 offen offset:384 // (op 1076) Load value
v_accvgpr_read v14, a12 // (op 1073) 
v_accvgpr_read v15, a13 // (op 1073) 
v_accvgpr_read v16, a14 // (op 1073) 
v_accvgpr_read v17, a15 // (op 1073) 
v_mul_f32 v18, v7, v14 // (op 1073) 
v_mul_f32 v19, v7, v15 // (op 1073) 
v_mul_f32 v20, v7, v16 // (op 1073) 
v_mul_f32 v21, v7, v17 // (op 1073) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v6, v10 // (op 1077) 
v_mul_f32 v15, v6, v11 // (op 1077) 
v_mul_f32 v16, v6, v12 // (op 1077) 
v_mul_f32 v17, v6, v13 // (op 1077) 

v_add_f32 v10, v14, v18 // (op 1074) 
v_add_f32 v11, v15, v19 // (op 1074) 
v_add_f32 v12, v16, v20 // (op 1074) 
v_add_f32 v13, v17, v21 // (op 1074) 





buffer_store_dwordx4 v[10:13], v9, s[60:63], 0 offen offset:384 // (op 1075) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v8, s[56:59], 0 offen offset:256 // (op 1058) Load value
v_accvgpr_read v14, a8 // (op 1055) 
v_accvgpr_read v15, a9 // (op 1055) 
v_accvgpr_read v16, a10 // (op 1055) 
v_accvgpr_read v17, a11 // (op 1055) 
v_mul_f32 v18, v7, v14 // (op 1055) 
v_mul_f32 v19, v7, v15 // (op 1055) 
v_mul_f32 v20, v7, v16 // (op 1055) 
v_mul_f32 v21, v7, v17 // (op 1055) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v6, v10 // (op 1059) 
v_mul_f32 v15, v6, v11 // (op 1059) 
v_mul_f32 v16, v6, v12 // (op 1059) 
v_mul_f32 v17, v6, v13 // (op 1059) 

v_add_f32 v10, v14, v18 // (op 1056) 
v_add_f32 v11, v15, v19 // (op 1056) 
v_add_f32 v12, v16, v20 // (op 1056) 
v_add_f32 v13, v17, v21 // (op 1056) 





buffer_store_dwordx4 v[10:13], v9, s[60:63], 0 offen offset:256 // (op 1057) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v8, s[56:59], 0 offen offset:128 // (op 1040) Load value
v_accvgpr_read v14, a4 // (op 1037) 
v_accvgpr_read v15, a5 // (op 1037) 
v_accvgpr_read v16, a6 // (op 1037) 
v_accvgpr_read v17, a7 // (op 1037) 
v_mul_f32 v18, v7, v14 // (op 1037) 
v_mul_f32 v19, v7, v15 // (op 1037) 
v_mul_f32 v20, v7, v16 // (op 1037) 
v_mul_f32 v21, v7, v17 // (op 1037) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v6, v10 // (op 1041) 
v_mul_f32 v15, v6, v11 // (op 1041) 
v_mul_f32 v16, v6, v12 // (op 1041) 
v_mul_f32 v17, v6, v13 // (op 1041) 

v_add_f32 v10, v14, v18 // (op 1038) 
v_add_f32 v11, v15, v19 // (op 1038) 
v_add_f32 v12, v16, v20 // (op 1038) 
v_add_f32 v13, v17, v21 // (op 1038) 





buffer_store_dwordx4 v[10:13], v9, s[60:63], 0 offen offset:128 // (op 1039) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v8, s[56:59], 0 offen // (op 212) Load value
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





buffer_store_dwordx4 v[10:13], v9, s[60:63], 0 offen // (op 211) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_189:
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
s_cbranch_scc1 RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_DPStreamTileLoop_396 // (op 396) Condition: Bottom (jump to RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_DPStreamTileLoop_396 if true)
RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_396:







































s_endpgm  // End of RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
.LRR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_end:
.size RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2, .LRR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_end-RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
  .amdhsa_next_free_vgpr 96
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 1024 // lds bytes
  .amdhsa_accum_offset 80
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
    macrotile: [128, 32, 128]
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
  MIWaveTile: [4, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
    .symbol: RR_GEMM_TN_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 1024
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 86
    .vgpr_count: 78
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

