// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2,@function
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 2178) 
v_and_b32 v3, 1, v0 // (op 2178) 

v_and_b32 v0, 15, v1 // (op 2178) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 2178) 


v_and_b32 v0, 63, v1 // (op 2178) 
v_lshrrev_b32 v3, 4, v0 // (op 2178) 

v_lshl_add_u32 v0, v4, 2, v3 // (op 2178) 


v_lshrrev_b32 v3, 6, v1 // (op 2278) 
v_and_b32 v4, 1, v3 // (op 2278) 

v_and_b32 v3, 15, v1 // (op 2278) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 2278) 


v_and_b32 v3, 63, v1 // (op 2278) 
v_lshrrev_b32 v4, 4, v3 // (op 2278) 

v_lshl_add_u32 v3, v5, 2, v4 // (op 2278) 


v_lshrrev_b32 v4, 6, v1 // (op 2349) 
v_and_b32 v5, 1, v4 // (op 2349) 

v_and_b32 v4, 15, v1 // (op 2349) 
v_lshl_add_u32 v6, v5, 4, v4 // (op 2349) 


v_and_b32 v4, 63, v1 // (op 2349) 
v_lshrrev_b32 v5, 4, v4 // (op 2349) 

v_lshl_add_u32 v4, v6, 2, v5 // (op 2349) 


v_lshrrev_b32 v5, 6, v1 // (op 2378) 
v_and_b32 v6, 1, v5 // (op 2378) 

v_and_b32 v5, 15, v1 // (op 2378) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 2378) 


v_and_b32 v5, 63, v1 // (op 2378) 
v_lshrrev_b32 v6, 4, v5 // (op 2378) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 2378) 


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
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_241 // (op 241) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_241 if false)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_SKStreamTileLoop_241:
v_lshrrev_b32 v8, 7, v1 // (op 2093) 
s_mul_i32 s52, s16, s8 // (op 2093) 
s_add_u32 s52, s52, s0 // (op 2093) 
v_and_b32 v9, 63, v1 // (op 2093) 
s_mul_hi_u32 s53, s52, s13 // (op 2093) 
v_lshrrev_b32 v10, 4, v9 // (op 2093) 
s_sub_u32 s54, s52, s53 // (op 2093) 
v_lshlrev_b32 v11, 2, v10 // (op 2093) 

v_lshl_add_u32 v10, v8, 4, v11 // (op 2093) 


s_lshr_b32 s55, s54, 1 // (op 2093) 

s_add_u32 s54, s55, s53 // (op 2093) 


v_lshrrev_b32 v8, 2, v9 // (op 2093) 

s_lshr_b32 s53, s54, s11 // (op 2093) 

v_and_b32 v9, 3, v8 // (op 2093) 

s_and_b32 s54, s11, 2147483648 // (op 2093) 
v_and_b32 v8, 3, v1 // (op 2093) 
s_cmp_eq_i32 s54, 0 // (op 2093) 
s_mov_b32 s55, scc // (op 2093) 

v_lshl_add_u32 v11, v9, 2, v8 // (op 2093) 


s_cmp_eq_i32 s55, 0 // (op 2093) 
s_cselect_b32 s54, s52, s53 // (op 2093) 



v_lshrrev_b32 v8, 6, v1 // (op 2093) 
s_mul_hi_u32 s52, s54, s12 // (op 2093) 
v_and_b32 v9, 1, v8 // (op 2093) 

s_sub_u32 s53, s54, s52 // (op 2093) 
v_lshl_add_u32 v8, v9, 4, v11 // (op 2093) 


s_lshr_b32 s55, s53, 1 // (op 2093) 

s_add_u32 s53, s55, s52 // (op 2093) 


s_lshr_b32 s52, s53, s10 // (op 2093) 

s_and_b32 s53, s10, 2147483648 // (op 2093) 
s_cmp_eq_i32 s53, 0 // (op 2093) 
s_mov_b32 s55, scc // (op 2093) 

s_cmp_eq_i32 s55, 0 // (op 2093) 
s_cselect_b32 s53, s54, s52 // (op 2093) 


s_mul_i32 s52, s53, s14 // (op 2093) 
v_lshl_add_u32 v9, s53, 4, v10 // (op 2093) 


s_sub_u32 s53, s54, s52 // (op 2093) 


v_lshl_add_u32 v10, s53, 6, v8 // (op 2093) 


v_mul_lo_u32 v8, v10, s4 // (op 2093) 

v_add_lshl_u32 v10, v9, v8, 2 // (op 2093) 


s_mul_i32 s53, s4, s23 // (op 2095) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 2095) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 2095) least significant
s_mul_hi_u32 s55, s4, s22 // (op 2095) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2095) most significant: sum
s_add_u32 s53, s53, s55 // (op 2095) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 2095) 

s_mov_b32 s56, s20 // (op 2095) reg expression
s_mov_b32 s57, s21 // (op 2095) reg expression
s_mov_b32 s58, s54 // (op 2095) reg expression
s_mov_b32 s59, 131072 // (op 2095) call()

v_lshrrev_b32 v8, 7, v1 // (op 2195) 
s_mul_i32 s52, s16, s8 // (op 2195) 
s_add_u32 s52, s52, s0 // (op 2195) 
v_and_b32 v9, 63, v1 // (op 2195) 
s_mul_hi_u32 s53, s52, s13 // (op 2195) 
v_lshrrev_b32 v11, 4, v9 // (op 2195) 
s_sub_u32 s54, s52, s53 // (op 2195) 
v_lshlrev_b32 v12, 2, v11 // (op 2195) 

s_lshr_b32 s55, s54, 1 // (op 2195) 

v_lshl_add_u32 v11, v8, 4, v12 // (op 2195) 


s_add_u32 s54, s55, s53 // (op 2195) 


v_lshrrev_b32 v8, 2, v9 // (op 2195) 

s_lshr_b32 s53, s54, s11 // (op 2195) 

v_and_b32 v9, 3, v8 // (op 2195) 

s_and_b32 s54, s11, 2147483648 // (op 2195) 
v_and_b32 v8, 3, v1 // (op 2195) 
s_cmp_eq_i32 s54, 0 // (op 2195) 
s_mov_b32 s55, scc // (op 2195) 

v_lshl_add_u32 v12, v9, 2, v8 // (op 2195) 


s_cmp_eq_i32 s55, 0 // (op 2195) 
s_cselect_b32 s54, s52, s53 // (op 2195) 



v_lshrrev_b32 v8, 6, v1 // (op 2195) 
s_mul_hi_u32 s55, s54, s12 // (op 2195) 
v_and_b32 v9, 1, v8 // (op 2195) 

s_sub_u32 s52, s54, s55 // (op 2195) 
v_lshl_add_u32 v8, v9, 4, v12 // (op 2195) 


s_lshr_b32 s53, s52, 1 // (op 2195) 

s_add_u32 s52, s53, s55 // (op 2195) 


s_lshr_b32 s53, s52, s10 // (op 2195) 

s_and_b32 s52, s10, 2147483648 // (op 2195) 
s_cmp_eq_i32 s52, 0 // (op 2195) 
s_mov_b32 s55, scc // (op 2195) 

s_cmp_eq_i32 s55, 0 // (op 2195) 
s_cselect_b32 s52, s54, s53 // (op 2195) 


s_mul_i32 s53, s52, s14 // (op 2195) 
v_lshl_add_u32 v9, s52, 4, v11 // (op 2195) 


s_sub_u32 s52, s54, s53 // (op 2195) 


v_lshl_add_u32 v11, s52, 6, v8 // (op 2195) 


v_mul_lo_u32 v8, v11, s24 // (op 2195) 

v_add_lshl_u32 v11, v9, v8, 2 // (op 2195) 


s_mul_i32 s53, s24, s23 // (op 2197) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 2197) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 2197) least significant
s_mul_hi_u32 s55, s24, s22 // (op 2197) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2197) most significant: sum
s_add_u32 s53, s53, s55 // (op 2197) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 2197) 

s_mov_b32 s60, s26 // (op 2197) reg expression
s_mov_b32 s61, s27 // (op 2197) reg expression
s_mov_b32 s62, s54 // (op 2197) reg expression
s_mov_b32 s63, 131072 // (op 2197) call()

v_lshrrev_b32 v8, 5, v1 // (op 2295) 
v_lshlrev_b32 v9, 2, v8 // (op 2295) 

v_lshl_add_u32 v8, s16, 4, v9 // (op 2295) 

v_and_b32 v9, -32, v1 // (op 2295) 
v_sub_u32 v12, v1, v9 // (op 2295) 

v_lshl_add_u32 v9, v8, 6, v12 // (op 2295) 


v_lshlrev_b32 v8, 2, v9 // (op 2295) 

s_lshl_b32 s52, s15, 12 // (op 2297) 
s_mov_b32 s64, s28 // (op 2297) reg expression
s_mov_b32 s65, s29 // (op 2297) reg expression
s_mov_b32 s66, s52 // (op 2297) reg expression
s_mov_b32 s67, 131072 // (op 2297) call()

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
v_accvgpr_write a4, 0.00000 // (op 822) call()
v_accvgpr_write a5, 0.00000 // (op 822) call()
v_accvgpr_write a6, 0.00000 // (op 822) call()
v_accvgpr_write a7, 0.00000 // (op 822) call()
v_lshrrev_b32 v9, 7, v1 // (op 1635) 
s_mul_i32 s52, s16, s8 // (op 1635) 
s_add_u32 s52, s52, s0 // (op 1635) 
v_and_b32 v12, 15, v1 // (op 1635) 
s_mul_hi_u32 s53, s52, s13 // (op 1635) 
v_lshl_add_u32 v13, v9, 4, v12 // (op 1635) 


s_sub_u32 s54, s52, s53 // (op 1635) 
v_and_b32 v9, 48, v1 // (op 1635) 
s_lshr_b32 s55, s54, 1 // (op 1635) 

s_add_u32 s54, s55, s53 // (op 1635) 


s_lshr_b32 s53, s54, s11 // (op 1635) 

s_and_b32 s54, s11, 2147483648 // (op 1635) 
s_cmp_eq_i32 s54, 0 // (op 1635) 
s_mov_b32 s55, scc // (op 1635) 

s_cmp_eq_i32 s55, 0 // (op 1635) 
s_cselect_b32 s54, s52, s53 // (op 1635) 


s_mul_hi_u32 s53, s54, s12 // (op 1635) 
s_sub_u32 s55, s54, s53 // (op 1635) 
s_lshr_b32 s68, s55, 1 // (op 1635) 

s_add_u32 s55, s68, s53 // (op 1635) 


s_lshr_b32 s53, s55, s10 // (op 1635) 

s_and_b32 s55, s10, 2147483648 // (op 1635) 
s_cmp_eq_i32 s55, 0 // (op 1635) 
s_mov_b32 s68, scc // (op 1635) 

s_cmp_eq_i32 s68, 0 // (op 1635) 
s_cselect_b32 s55, s54, s53 // (op 1635) 


v_lshl_add_u32 v12, s55, 4, v13 // (op 1635) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 1635) 
s_mul_i32 s53, s54, s68 // (op 1635) 


s_sub_u32 s54, s52, s53 // (op 1635) 


v_lshl_add_u32 v13, s54, 7, v9 // (op 1635) 


v_mul_lo_u32 v9, v12, s30 // (op 1635) 
v_add_u32 v9, v9, v13 // (op 1635) 


s_mul_i32 s53, s30, s35 // (op 1637) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 1637) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 1637) least significant
s_mul_hi_u32 s55, s30, s34 // (op 1637) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1637) most significant: sum
s_add_u32 s53, s53, s55 // (op 1637) most significant: sum


s_mov_b32 s68, s32 // (op 1637) reg expression
s_mov_b32 s69, s33 // (op 1637) reg expression
s_mov_b32 s70, s52 // (op 1637) reg expression
s_mov_b32 s71, 131072 // (op 1637) call()

v_lshrrev_b32 v12, 7, v1 // (op 1654) 
s_mul_i32 s52, s16, s8 // (op 1654) 
s_add_u32 s52, s52, s0 // (op 1654) 
v_and_b32 v13, 15, v1 // (op 1654) 
s_mul_hi_u32 s53, s52, s13 // (op 1654) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 1654) 


s_sub_u32 s54, s52, s53 // (op 1654) 
v_and_b32 v12, 48, v1 // (op 1654) 
s_lshr_b32 s55, s54, 1 // (op 1654) 

s_add_u32 s54, s55, s53 // (op 1654) 


s_lshr_b32 s53, s54, s11 // (op 1654) 

s_and_b32 s54, s11, 2147483648 // (op 1654) 
s_cmp_eq_i32 s54, 0 // (op 1654) 
s_mov_b32 s55, scc // (op 1654) 

s_cmp_eq_i32 s55, 0 // (op 1654) 
s_cselect_b32 s54, s52, s53 // (op 1654) 


s_mul_hi_u32 s53, s54, s12 // (op 1654) 
s_sub_u32 s55, s54, s53 // (op 1654) 
s_lshr_b32 s72, s55, 1 // (op 1654) 

s_add_u32 s55, s72, s53 // (op 1654) 


s_lshr_b32 s53, s55, s10 // (op 1654) 

s_and_b32 s55, s10, 2147483648 // (op 1654) 
s_cmp_eq_i32 s55, 0 // (op 1654) 
s_mov_b32 s72, scc // (op 1654) 

s_cmp_eq_i32 s72, 0 // (op 1654) 
s_cselect_b32 s55, s54, s53 // (op 1654) 


v_lshl_add_u32 v13, s55, 4, v14 // (op 1654) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 1654) 
s_mul_i32 s53, s54, s72 // (op 1654) 


s_sub_u32 s54, s52, s53 // (op 1654) 


v_lshl_add_u32 v14, s54, 7, v12 // (op 1654) 


v_mul_lo_u32 v12, v13, s30 // (op 1654) 
v_add_u32 v12, v12, v14 // (op 1654) 


s_mul_i32 s53, s30, s35 // (op 1656) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 1656) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 1656) least significant
s_mul_hi_u32 s55, s30, s34 // (op 1656) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1656) most significant: sum
s_add_u32 s53, s53, s55 // (op 1656) most significant: sum


s_mov_b32 s68, s32 // (op 1656) reg expression
s_mov_b32 s69, s33 // (op 1656) reg expression
s_mov_b32 s70, s52 // (op 1656) reg expression
s_mov_b32 s71, 131072 // (op 1656) call()

v_lshrrev_b32 v13, 6, v1 // (op 1788) 
s_mul_i32 s52, s16, s8 // (op 1788) 
s_add_u32 s52, s52, s0 // (op 1788) 
v_and_b32 v14, 1, v13 // (op 1788) 

s_mul_hi_u32 s53, s52, s13 // (op 1788) 
v_and_b32 v13, 15, v1 // (op 1788) 
s_sub_u32 s54, s52, s53 // (op 1788) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 1788) 


s_lshr_b32 s55, s54, 1 // (op 1788) 

v_and_b32 v13, 63, v1 // (op 1788) 
s_add_u32 s54, s55, s53 // (op 1788) 


v_lshrrev_b32 v14, 4, v13 // (op 1788) 

s_lshr_b32 s53, s54, s11 // (op 1788) 

v_lshlrev_b32 v13, 5, v14 // (op 1788) 

s_and_b32 s54, s11, 2147483648 // (op 1788) 
s_cmp_eq_i32 s54, 0 // (op 1788) 
s_mov_b32 s55, scc // (op 1788) 

s_cmp_eq_i32 s55, 0 // (op 1788) 
s_cselect_b32 s54, s52, s53 // (op 1788) 


s_mul_hi_u32 s53, s54, s12 // (op 1788) 
s_sub_u32 s55, s54, s53 // (op 1788) 
s_lshr_b32 s72, s55, 1 // (op 1788) 

s_add_u32 s55, s72, s53 // (op 1788) 


s_lshr_b32 s53, s55, s10 // (op 1788) 

s_and_b32 s55, s10, 2147483648 // (op 1788) 
s_cmp_eq_i32 s55, 0 // (op 1788) 
s_mov_b32 s72, scc // (op 1788) 

s_cmp_eq_i32 s72, 0 // (op 1788) 
s_cselect_b32 s55, s54, s53 // (op 1788) 


s_mul_i32 s53, s55, s14 // (op 1788) 

s_sub_u32 s55, s54, s53 // (op 1788) 

v_lshl_add_u32 v14, s55, 6, v15 // (op 1788) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 1788) 
v_mul_lo_u32 v17, v14, s37 // (op 1788) most significant: low of low * high
v_mul_hi_u32 v15, v14, s36 // (op 1788) most significant: high of low * low
v_mul_lo_u32 v16, v14, s36 // (op 1788) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 1788) most significant: sum


s_mul_i32 s53, s54, s72 // (op 1788) 


s_sub_u32 s54, s52, s53 // (op 1788) 


v_lshl_add_u32 v14, s54, 7, v13 // (op 1788) 


v_mov_b32 v19, 0 // (op 1788) convert
v_mov_b32 v18, v14 // (op 1788) convert

v_add_co_u32 v14, s[52:53], v18, v16 // (op 1788) least significant half
v_addc_co_u32 v15, s[52:53], v19, v17, s[52:53] // (op 1788) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 1788) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 1788) 

v_mov_b32 v13, v16 // (op 1788) convert

s_mul_i32 s53, s36, s23 // (op 1790) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 1790) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 1790) least significant
s_mul_hi_u32 s55, s36, s22 // (op 1790) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1790) most significant: sum
s_add_u32 s53, s53, s55 // (op 1790) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1790) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1790) 

s_mov_b32 s72, s38 // (op 1790) reg expression
s_mov_b32 s73, s39 // (op 1790) reg expression
s_mov_b32 s74, s52 // (op 1790) reg expression
s_mov_b32 s75, 131072 // (op 1790) call()

v_lshrrev_b32 v14, 6, v1 // (op 1809) 
s_mul_i32 s52, s16, s8 // (op 1809) 
s_add_u32 s52, s52, s0 // (op 1809) 
v_and_b32 v15, 1, v14 // (op 1809) 

s_mul_hi_u32 s53, s52, s13 // (op 1809) 
v_and_b32 v14, 15, v1 // (op 1809) 
s_sub_u32 s54, s52, s53 // (op 1809) 
v_lshl_add_u32 v16, v15, 4, v14 // (op 1809) 


s_lshr_b32 s55, s54, 1 // (op 1809) 

v_and_b32 v14, 63, v1 // (op 1809) 
s_add_u32 s54, s55, s53 // (op 1809) 


v_lshrrev_b32 v15, 4, v14 // (op 1809) 

s_lshr_b32 s53, s54, s11 // (op 1809) 

v_lshlrev_b32 v14, 5, v15 // (op 1809) 

s_and_b32 s54, s11, 2147483648 // (op 1809) 
s_cmp_eq_i32 s54, 0 // (op 1809) 
s_mov_b32 s55, scc // (op 1809) 

s_cmp_eq_i32 s55, 0 // (op 1809) 
s_cselect_b32 s54, s52, s53 // (op 1809) 


s_mul_hi_u32 s53, s54, s12 // (op 1809) 
s_sub_u32 s55, s54, s53 // (op 1809) 
s_lshr_b32 s76, s55, 1 // (op 1809) 

s_add_u32 s55, s76, s53 // (op 1809) 


s_lshr_b32 s53, s55, s10 // (op 1809) 

s_and_b32 s55, s10, 2147483648 // (op 1809) 
s_cmp_eq_i32 s55, 0 // (op 1809) 
s_mov_b32 s76, scc // (op 1809) 

s_cmp_eq_i32 s76, 0 // (op 1809) 
s_cselect_b32 s55, s54, s53 // (op 1809) 


s_mul_i32 s53, s55, s14 // (op 1809) 

s_sub_u32 s55, s54, s53 // (op 1809) 

v_lshl_add_u32 v15, s55, 6, v16 // (op 1809) 


v_mul_lo_u32 v17, v15, s37 // (op 1809) most significant: low of low * high
v_mul_hi_u32 v18, v15, s36 // (op 1809) most significant: high of low * low
v_mul_lo_u32 v16, v15, s36 // (op 1809) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 1809) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1809) 
s_mul_i32 s53, s54, s76 // (op 1809) 


s_sub_u32 s54, s52, s53 // (op 1809) 


v_lshl_add_u32 v15, s54, 7, v14 // (op 1809) 


v_mov_b32 v19, 0 // (op 1809) convert
v_mov_b32 v18, v15 // (op 1809) convert

v_add_co_u32 v14, s[52:53], v18, v16 // (op 1809) least significant half
v_addc_co_u32 v15, s[52:53], v19, v17, s[52:53] // (op 1809) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 1809) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 1809) 

v_mov_b32 v14, v16 // (op 1809) convert

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

v_lshrrev_b32 v15, 7, v1 // (op 1921) 
s_mul_i32 s52, s16, s8 // (op 1921) 
s_add_u32 s52, s52, s0 // (op 1921) 
v_and_b32 v16, 15, v1 // (op 1921) 
s_mul_hi_u32 s53, s52, s13 // (op 1921) 
v_lshl_add_u32 v17, v15, 4, v16 // (op 1921) 


s_sub_u32 s54, s52, s53 // (op 1921) 
v_and_b32 v15, 63, v1 // (op 1921) 
s_lshr_b32 s55, s54, 1 // (op 1921) 

v_lshrrev_b32 v16, 4, v15 // (op 1921) 

s_add_u32 s54, s55, s53 // (op 1921) 


s_lshr_b32 s53, s54, s11 // (op 1921) 

s_and_b32 s54, s11, 2147483648 // (op 1921) 
s_cmp_eq_i32 s54, 0 // (op 1921) 
s_mov_b32 s55, scc // (op 1921) 

s_cmp_eq_i32 s55, 0 // (op 1921) 
s_cselect_b32 s54, s52, s53 // (op 1921) 


s_mul_hi_u32 s53, s54, s12 // (op 1921) 
s_sub_u32 s55, s54, s53 // (op 1921) 
s_lshr_b32 s76, s55, 1 // (op 1921) 

s_add_u32 s55, s76, s53 // (op 1921) 


s_lshr_b32 s53, s55, s10 // (op 1921) 

s_and_b32 s55, s10, 2147483648 // (op 1921) 
s_cmp_eq_i32 s55, 0 // (op 1921) 
s_mov_b32 s76, scc // (op 1921) 

s_cmp_eq_i32 s76, 0 // (op 1921) 
s_cselect_b32 s55, s54, s53 // (op 1921) 


v_lshl_add_u32 v15, s55, 4, v17 // (op 1921) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1921) 
s_mul_i32 s53, s54, s76 // (op 1921) 


s_sub_u32 s54, s52, s53 // (op 1921) 


v_lshl_add_u32 v17, s54, 2, v16 // (op 1921) 


v_mul_lo_u32 v16, v15, s40 // (op 1921) 
v_add_u32 v16, v16, v17 // (op 1921) 


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

v_lshrrev_b32 v15, 7, v1 // (op 1940) 
s_mul_i32 s52, s16, s8 // (op 1940) 
s_add_u32 s52, s52, s0 // (op 1940) 
v_and_b32 v17, 15, v1 // (op 1940) 
s_mul_hi_u32 s53, s52, s13 // (op 1940) 
v_lshl_add_u32 v18, v15, 4, v17 // (op 1940) 


s_sub_u32 s54, s52, s53 // (op 1940) 
v_and_b32 v15, 63, v1 // (op 1940) 
s_lshr_b32 s55, s54, 1 // (op 1940) 

v_lshrrev_b32 v17, 4, v15 // (op 1940) 

s_add_u32 s54, s55, s53 // (op 1940) 


s_lshr_b32 s53, s54, s11 // (op 1940) 

s_and_b32 s54, s11, 2147483648 // (op 1940) 
s_cmp_eq_i32 s54, 0 // (op 1940) 
s_mov_b32 s55, scc // (op 1940) 

s_cmp_eq_i32 s55, 0 // (op 1940) 
s_cselect_b32 s54, s52, s53 // (op 1940) 


s_mul_hi_u32 s53, s54, s12 // (op 1940) 
s_sub_u32 s55, s54, s53 // (op 1940) 
s_lshr_b32 s80, s55, 1 // (op 1940) 

s_add_u32 s55, s80, s53 // (op 1940) 


s_lshr_b32 s53, s55, s10 // (op 1940) 

s_and_b32 s55, s10, 2147483648 // (op 1940) 
s_cmp_eq_i32 s55, 0 // (op 1940) 
s_mov_b32 s80, scc // (op 1940) 

s_cmp_eq_i32 s80, 0 // (op 1940) 
s_cselect_b32 s55, s54, s53 // (op 1940) 


v_lshl_add_u32 v15, s55, 4, v18 // (op 1940) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 1940) 
s_mul_i32 s53, s54, s80 // (op 1940) 


s_sub_u32 s54, s52, s53 // (op 1940) 


v_lshl_add_u32 v18, s54, 2, v17 // (op 1940) 


v_mul_lo_u32 v17, v15, s40 // (op 1940) 
v_add_u32 v17, v17, v18 // (op 1940) 


s_mul_i32 s53, s40, s35 // (op 1942) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 1942) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 1942) least significant
s_mul_hi_u32 s55, s40, s34 // (op 1942) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1942) most significant: sum
s_add_u32 s53, s53, s55 // (op 1942) most significant: sum


s_mov_b32 s76, s42 // (op 1942) reg expression
s_mov_b32 s77, s43 // (op 1942) reg expression
s_mov_b32 s78, s52 // (op 1942) reg expression
s_mov_b32 s79, 131072 // (op 1942) call()

v_lshrrev_b32 v15, 2, v1 // (op 2049) 
s_mul_i32 s52, s16, s8 // (op 2049) 
s_add_u32 s52, s52, s0 // (op 2049) 
v_lshlrev_b32 v18, 1, v15 // (op 2049) 

s_mul_hi_u32 s53, s52, s13 // (op 2049) 
v_and_b32 v15, -4, v1 // (op 2049) 
s_sub_u32 s54, s52, s53 // (op 2049) 
v_sub_u32 v19, v1, v15 // (op 2049) 

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
s_lshr_b32 s80, s55, 1 // (op 2049) 

s_add_u32 s55, s80, s53 // (op 2049) 


s_lshr_b32 s53, s55, s10 // (op 2049) 

s_and_b32 s55, s10, 2147483648 // (op 2049) 
s_cmp_eq_i32 s55, 0 // (op 2049) 
s_mov_b32 s80, scc // (op 2049) 

s_cmp_eq_i32 s80, 0 // (op 2049) 
s_cselect_b32 s55, s54, s53 // (op 2049) 


s_mul_i32 s53, s55, s14 // (op 2049) 

s_sub_u32 s55, s54, s53 // (op 2049) 

v_lshl_add_u32 v15, s55, 6, v18 // (op 2049) 


v_mul_lo_u32 v18, v15, s44 // (op 2049) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 2049) 
s_mul_i32 s53, s54, s80 // (op 2049) 


s_sub_u32 s54, s52, s53 // (op 2049) 


v_lshl_add_u32 v15, s54, 2, v19 // (op 2049) 


v_add_u32 v19, v15, v18 // (op 2049) 


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

v_lshrrev_b32 v15, 2, v1 // (op 2068) 
s_mul_i32 s52, s16, s8 // (op 2068) 
s_add_u32 s52, s52, s0 // (op 2068) 
v_lshlrev_b32 v18, 1, v15 // (op 2068) 

s_mul_hi_u32 s53, s52, s13 // (op 2068) 
v_and_b32 v15, -4, v1 // (op 2068) 
s_sub_u32 s54, s52, s53 // (op 2068) 
v_sub_u32 v20, v1, v15 // (op 2068) 

s_lshr_b32 s55, s54, 1 // (op 2068) 

s_add_u32 s54, s55, s53 // (op 2068) 


s_lshr_b32 s53, s54, s11 // (op 2068) 

s_and_b32 s54, s11, 2147483648 // (op 2068) 
s_cmp_eq_i32 s54, 0 // (op 2068) 
s_mov_b32 s55, scc // (op 2068) 

s_cmp_eq_i32 s55, 0 // (op 2068) 
s_cselect_b32 s54, s52, s53 // (op 2068) 


s_mul_hi_u32 s53, s54, s12 // (op 2068) 
s_sub_u32 s55, s54, s53 // (op 2068) 
s_lshr_b32 s84, s55, 1 // (op 2068) 

s_add_u32 s55, s84, s53 // (op 2068) 


s_lshr_b32 s53, s55, s10 // (op 2068) 

s_and_b32 s55, s10, 2147483648 // (op 2068) 
s_cmp_eq_i32 s55, 0 // (op 2068) 
s_mov_b32 s84, scc // (op 2068) 

s_cmp_eq_i32 s84, 0 // (op 2068) 
s_cselect_b32 s55, s54, s53 // (op 2068) 


s_mul_i32 s53, s55, s14 // (op 2068) 

s_sub_u32 s55, s54, s53 // (op 2068) 

v_lshl_add_u32 v15, s55, 6, v18 // (op 2068) 


v_mul_lo_u32 v18, v15, s44 // (op 2068) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 2068) 
s_mul_i32 s53, s54, s84 // (op 2068) 


s_sub_u32 s54, s52, s53 // (op 2068) 


v_lshl_add_u32 v15, s54, 2, v20 // (op 2068) 


v_add_u32 v20, v15, v18 // (op 2068) 


s_mul_i32 s53, s44, s23 // (op 2070) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 2070) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 2070) least significant
s_mul_hi_u32 s55, s44, s22 // (op 2070) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2070) most significant: sum
s_add_u32 s53, s53, s55 // (op 2070) most significant: sum


s_mov_b32 s80, s46 // (op 2070) reg expression
s_mov_b32 s81, s47 // (op 2070) reg expression
s_mov_b32 s82, s52 // (op 2070) reg expression
s_mov_b32 s83, 131072 // (op 2070) call()




buffer_load_dwordx4 v[22:25], v9, s[68:71], 0 offen // (op 1428) Load value
buffer_load_dwordx4 v[26:29], v9, s[68:71], 0 offen offset:64 // (op 1428) Load value


v_mov_b32 v15, v13 // (op 1434) reg expression

buffer_load_dwordx4 v[30:33], v15, s[72:75], 0 offen // (op 1434) Load value




buffer_load_ubyte v15, v16, s[76:79], 0 offen offset: 0 // (op 1440) Load value


s_lshl_b64 s[52:53], s[36:37], 5 // (op 1446) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1446) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1446) 

v_add_u32 v18, v13, s52 // (op 1446) 


buffer_load_dwordx4 v[34:37], v18, s[72:75], 0 offen // (op 1446) Load value


v_mov_b64 v[38:39], s[44:45] // (op 1451) convert

v_mov_b32 v18, v19 // (op 1451) 
buffer_load_ubyte v40, v18, s[80:83], 0 offen offset: 0 // (op 1451) Load value
v_add_u32 v18, v18, v38 // (op 1451) 
buffer_load_ubyte v41, v18, s[80:83], 0 offen offset: 0 // (op 1451) Load value





buffer_load_dwordx4 v[42:45], v9, s[68:71], 0 offen offset:128 // (op 1455) Load value
buffer_load_dwordx4 v[46:49], v9, s[68:71], 0 offen offset:192 // (op 1455) Load value



v_add_u32 v9, 64, v13 // (op 1461) 

buffer_load_dwordx4 v[50:53], v9, s[72:75], 0 offen // (op 1461) Load value




buffer_load_ubyte v9, v16, s[76:79], 0 offen offset:4 // (op 1467) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 1473) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1473) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1473) 

s_add_u32 s54, s52, 64 // (op 1473) 

v_add_u32 v16, v13, s54 // (op 1473) 


buffer_load_dwordx4 v[54:57], v16, s[72:75], 0 offen // (op 1473) Load value



v_mov_b64 v[38:39], s[44:45] // (op 1478) convert

v_mov_b32 v13, v19 // (op 1478) 
buffer_load_ubyte v58, v13, s[80:83], 0 offen offset:4 // (op 1478) Load value
v_add_u32 v13, v13, v38 // (op 1478) 
buffer_load_ubyte v59, v13, s[80:83], 0 offen offset:4 // (op 1478) Load value



v_and_b32 v13, -4, v1 // (op 2161) 
v_sub_u32 v16, v1, v13 // (op 2161) 

v_lshrrev_b32 v13, 2, v1 // (op 2161) 
v_lshl_add_u32 v21, v13, 3, v16 // (op 2161) 


s_barrier  // (op 2449) 



s_waitcnt vmcnt(8)
ds_write_b8 v21, v40 // (op 1480) Store local data 
s_waitcnt vmcnt(7)
ds_write_b8 v21, v41 offset:4 // (op 1480) Store local data 

v_and_b32 v13, -4, v1 // (op 2132) 
v_sub_u32 v16, v1, v13 // (op 2132) 

v_lshrrev_b32 v13, 2, v1 // (op 2132) 
v_lshl_add_u32 v21, v13, 3, v16 // (op 2132) 


v_and_b32 v13, -4, v1 // (op 2361) 
v_sub_u32 v16, v1, v13 // (op 2361) 

v_lshrrev_b32 v13, 2, v1 // (op 2361) 
v_lshl_add_u32 v18, v13, 3, v16 // (op 2361) 


s_mov_b32 s1, 0 // (op 824) call()
s_and_b32 s52, s7, 4294967294 // (op 823) 
s_cmp_lt_u32 s1, s52 // (op 823) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_823 // (op 823) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_823 if false)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoop_823:
s_add_u32 s1, s1, 2 // (op 825) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2446) 



ds_read_u8 v13, v0 // (op 67) Load local data 



ds_read_u8 v16, v0 offset:128 // (op 849) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:33], a[0:3], v15, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 70) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:29], v[34:37], a[4:7], v15, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 850) 



v_mov_b32 v13, v12 // (op 55) 

buffer_load_dwordx4 v[22:25], v13, s[68:71], 0 offen offset:256 // (op 55) Load value
buffer_load_dwordx4 v[26:29], v13, s[68:71], 0 offen offset:320 // (op 55) Load value


v_mov_b32 v16, v14 // (op 59) 
v_add_u32 v19, 128, v16 // (op 59) 

buffer_load_dwordx4 v[30:33], v19, s[72:75], 0 offen // (op 59) Load value



v_mov_b32 v19, v17 // (op 63) 

buffer_load_ubyte v15, v19, s[76:79], 0 offen offset:8 // (op 63) Load value


s_lshl_b64 s[52:53], s[36:37], 5 // (op 872) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 872) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 872) 

s_add_u32 s54, s52, 128 // (op 872) 

v_add_u32 v38, v16, s54 // (op 872) 


buffer_load_dwordx4 v[34:37], v38, s[72:75], 0 offen // (op 872) Load value


v_mov_b64 v[38:39], s[44:45] // (op 13) convert

v_mov_b32 v60, v20 // (op 13) 
v_mov_b32 v61, v60 // (op 13) 
buffer_load_ubyte v40, v61, s[80:83], 0 offen offset:8 // (op 13) Load value
v_add_u32 v61, v61, v38 // (op 13) 
buffer_load_ubyte v41, v61, s[80:83], 0 offen offset:8 // (op 13) Load value


s_barrier  // (op 2487) 



s_waitcnt vmcnt(8)
ds_write_b8 v18, v58 offset:256 // (op 765) Store local data 
s_waitcnt vmcnt(7)
ds_write_b8 v18, v59 offset:260 // (op 765) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2484) 



ds_read_u8 v38, v5 offset:256 // (op 767) Load local data 



ds_read_u8 v39, v5 offset:384 // (op 853) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[50:53], a[0:3], v9, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 768) 

s_barrier  // (op 2632) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:49], v[54:57], a[4:7], v9, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 851) 




buffer_load_dwordx4 v[42:45], v13, s[68:71], 0 offen offset:384 // (op 770) Load value
buffer_load_dwordx4 v[46:49], v13, s[68:71], 0 offen offset:448 // (op 770) Load value



v_add_u32 v12, 256, v12 // (op 1646) 
v_add_u32 v13, 192, v16 // (op 772) 

buffer_load_dwordx4 v[50:53], v13, s[72:75], 0 offen // (op 772) Load value




buffer_load_ubyte v9, v19, s[76:79], 0 offen offset:12 // (op 774) Load value



v_add_u32 v17, 8, v17 // (op 1932) 
s_lshl_b64 s[52:53], s[36:37], 5 // (op 867) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 867) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 867) 

s_add_u32 s54, s52, 192 // (op 867) 

v_add_u32 v13, v16, s54 // (op 867) 


buffer_load_dwordx4 v[54:57], v13, s[72:75], 0 offen // (op 867) Load value



v_add_u32 v14, 128, v14 // (op 1799) 
v_mov_b64 v[38:39], s[44:45] // (op 764) convert

v_mov_b32 v13, v60 // (op 764) 
buffer_load_ubyte v58, v13, s[80:83], 0 offen offset:12 // (op 764) Load value
v_add_u32 v13, v13, v38 // (op 764) 
buffer_load_ubyte v59, v13, s[80:83], 0 offen offset:12 // (op 764) Load value



s_barrier  // (op 2452) 
v_add_u32 v20, 8, v20 // (op 2060) 



s_waitcnt vmcnt(8)
ds_write_b8 v21, v40 // (op 18) Store local data 
s_waitcnt vmcnt(7)
ds_write_b8 v21, v41 offset:4 // (op 18) Store local data 
s_and_b32 s52, s7, 4294967294 // (op 823) 
s_cmp_lt_u32 s1, s52 // (op 823) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoop_823 // (op 823) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoop_823 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_823:











s_and_b32 s52, s7, 4294967294 // (op 1562) 
v_lshrrev_b32 v12, 7, v1 // (op 1562) 
s_add_u32 s53, s0, s52 // (op 1562) 

v_and_b32 v13, 15, v1 // (op 1562) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 1562) 


s_mul_i32 s52, s16, s8 // (op 1562) 
s_add_u32 s52, s52, s53 // (op 1562) 

v_and_b32 v12, 48, v1 // (op 1562) 
s_mul_hi_u32 s53, s52, s13 // (op 1562) 
s_sub_u32 s54, s52, s53 // (op 1562) 
s_lshr_b32 s55, s54, 1 // (op 1562) 

s_add_u32 s54, s55, s53 // (op 1562) 


s_lshr_b32 s53, s54, s11 // (op 1562) 

s_and_b32 s54, s11, 2147483648 // (op 1562) 
s_cmp_eq_i32 s54, 0 // (op 1562) 
s_mov_b32 s55, scc // (op 1562) 

s_cmp_eq_i32 s55, 0 // (op 1562) 
s_cselect_b32 s54, s52, s53 // (op 1562) 


s_mul_hi_u32 s53, s54, s12 // (op 1562) 
s_sub_u32 s55, s54, s53 // (op 1562) 
s_lshr_b32 s84, s55, 1 // (op 1562) 

s_add_u32 s55, s84, s53 // (op 1562) 


s_lshr_b32 s53, s55, s10 // (op 1562) 

s_and_b32 s55, s10, 2147483648 // (op 1562) 
s_cmp_eq_i32 s55, 0 // (op 1562) 
s_mov_b32 s84, scc // (op 1562) 

s_cmp_eq_i32 s84, 0 // (op 1562) 
s_cselect_b32 s55, s54, s53 // (op 1562) 


v_lshl_add_u32 v13, s55, 4, v14 // (op 1562) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 1562) 
s_mul_i32 s53, s54, s84 // (op 1562) 


s_sub_u32 s54, s52, s53 // (op 1562) 


v_lshl_add_u32 v14, s54, 7, v12 // (op 1562) 


v_mul_lo_u32 v12, v13, s30 // (op 1562) 
v_add_u32 v12, v12, v14 // (op 1562) 


s_mul_i32 s53, s30, s35 // (op 1564) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 1564) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 1564) least significant
s_mul_hi_u32 s55, s30, s34 // (op 1564) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1564) most significant: sum
s_add_u32 s53, s53, s55 // (op 1564) most significant: sum


s_mov_b32 s68, s32 // (op 1564) reg expression
s_mov_b32 s69, s33 // (op 1564) reg expression
s_mov_b32 s70, s52 // (op 1564) reg expression
s_mov_b32 s71, 131072 // (op 1564) call()

s_and_b32 s52, s7, 4294967294 // (op 1705) 
v_lshrrev_b32 v13, 6, v1 // (op 1705) 
s_add_u32 s53, s0, s52 // (op 1705) 

v_and_b32 v14, 1, v13 // (op 1705) 

v_and_b32 v13, 15, v1 // (op 1705) 
s_mul_i32 s52, s16, s8 // (op 1705) 
s_add_u32 s52, s52, s53 // (op 1705) 

v_lshl_add_u32 v15, v14, 4, v13 // (op 1705) 


s_mul_hi_u32 s53, s52, s13 // (op 1705) 
v_and_b32 v13, 63, v1 // (op 1705) 
s_sub_u32 s54, s52, s53 // (op 1705) 
v_lshrrev_b32 v14, 4, v13 // (op 1705) 

s_lshr_b32 s55, s54, 1 // (op 1705) 

v_lshlrev_b32 v13, 5, v14 // (op 1705) 

s_add_u32 s54, s55, s53 // (op 1705) 


s_lshr_b32 s53, s54, s11 // (op 1705) 

s_and_b32 s54, s11, 2147483648 // (op 1705) 
s_cmp_eq_i32 s54, 0 // (op 1705) 
s_mov_b32 s55, scc // (op 1705) 

s_cmp_eq_i32 s55, 0 // (op 1705) 
s_cselect_b32 s54, s52, s53 // (op 1705) 


s_mul_hi_u32 s53, s54, s12 // (op 1705) 
s_sub_u32 s55, s54, s53 // (op 1705) 
s_lshr_b32 s84, s55, 1 // (op 1705) 

s_add_u32 s55, s84, s53 // (op 1705) 


s_lshr_b32 s53, s55, s10 // (op 1705) 

s_and_b32 s55, s10, 2147483648 // (op 1705) 
s_cmp_eq_i32 s55, 0 // (op 1705) 
s_mov_b32 s84, scc // (op 1705) 

s_cmp_eq_i32 s84, 0 // (op 1705) 
s_cselect_b32 s55, s54, s53 // (op 1705) 


s_mul_i32 s53, s55, s14 // (op 1705) 

s_sub_u32 s55, s54, s53 // (op 1705) 

v_lshl_add_u32 v14, s55, 6, v15 // (op 1705) 


v_mul_lo_u32 v41, v14, s37 // (op 1705) most significant: low of low * high
v_mul_hi_u32 v15, v14, s36 // (op 1705) most significant: high of low * low
v_mul_lo_u32 v40, v14, s36 // (op 1705) least significant: low of low * low
v_add_u32 v41, v41, v15 // (op 1705) most significant: sum


s_ashr_i64 s[84:85], s[2:3], 7 // (op 1705) 
s_mul_i32 s53, s54, s84 // (op 1705) 


s_sub_u32 s54, s52, s53 // (op 1705) 


v_lshl_add_u32 v14, s54, 7, v13 // (op 1705) 


v_mov_b32 v39, 0 // (op 1705) convert
v_mov_b32 v38, v14 // (op 1705) convert

v_add_co_u32 v36, s[52:53], v38, v40 // (op 1705) least significant half
v_addc_co_u32 v37, s[52:53], v39, v41, s[52:53] // (op 1705) most significant half

v_lshlrev_b64 v[36:37], 2, v[36:37] // (op 1705) 


v_lshrrev_b64 v[34:35], 3, v[36:37] // (op 1705) 

v_mov_b32 v13, v34 // (op 1705) convert

s_mul_i32 s53, s36, s23 // (op 1707) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 1707) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 1707) least significant
s_mul_hi_u32 s55, s36, s22 // (op 1707) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1707) most significant: sum
s_add_u32 s53, s53, s55 // (op 1707) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1707) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1707) 

s_mov_b32 s72, s38 // (op 1707) reg expression
s_mov_b32 s73, s39 // (op 1707) reg expression
s_mov_b32 s74, s52 // (op 1707) reg expression
s_mov_b32 s75, 131072 // (op 1707) call()

s_and_b32 s52, s7, 4294967294 // (op 1856) 
v_lshrrev_b32 v14, 7, v1 // (op 1856) 
s_add_u32 s53, s0, s52 // (op 1856) 

v_and_b32 v15, 15, v1 // (op 1856) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 1856) 


s_mul_i32 s52, s16, s8 // (op 1856) 
s_add_u32 s52, s52, s53 // (op 1856) 

v_and_b32 v14, 63, v1 // (op 1856) 
s_mul_hi_u32 s53, s52, s13 // (op 1856) 
v_lshrrev_b32 v15, 4, v14 // (op 1856) 

s_sub_u32 s54, s52, s53 // (op 1856) 
s_lshr_b32 s55, s54, 1 // (op 1856) 

s_add_u32 s54, s55, s53 // (op 1856) 


s_lshr_b32 s53, s54, s11 // (op 1856) 

s_and_b32 s54, s11, 2147483648 // (op 1856) 
s_cmp_eq_i32 s54, 0 // (op 1856) 
s_mov_b32 s55, scc // (op 1856) 

s_cmp_eq_i32 s55, 0 // (op 1856) 
s_cselect_b32 s54, s52, s53 // (op 1856) 


s_mul_hi_u32 s53, s54, s12 // (op 1856) 
s_sub_u32 s55, s54, s53 // (op 1856) 
s_lshr_b32 s84, s55, 1 // (op 1856) 

s_add_u32 s55, s84, s53 // (op 1856) 


s_lshr_b32 s53, s55, s10 // (op 1856) 

s_and_b32 s55, s10, 2147483648 // (op 1856) 
s_cmp_eq_i32 s55, 0 // (op 1856) 
s_mov_b32 s84, scc // (op 1856) 

s_cmp_eq_i32 s84, 0 // (op 1856) 
s_cselect_b32 s55, s54, s53 // (op 1856) 


v_lshl_add_u32 v14, s55, 4, v16 // (op 1856) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 1856) 
s_mul_i32 s53, s54, s84 // (op 1856) 


s_sub_u32 s54, s52, s53 // (op 1856) 


v_lshl_add_u32 v16, s54, 2, v15 // (op 1856) 


v_mul_lo_u32 v15, v14, s40 // (op 1856) 
v_add_u32 v15, v15, v16 // (op 1856) 


s_mul_i32 s53, s40, s35 // (op 1858) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 1858) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 1858) least significant
s_mul_hi_u32 s55, s40, s34 // (op 1858) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1858) most significant: sum
s_add_u32 s53, s53, s55 // (op 1858) most significant: sum


s_mov_b32 s76, s42 // (op 1858) reg expression
s_mov_b32 s77, s43 // (op 1858) reg expression
s_mov_b32 s78, s52 // (op 1858) reg expression
s_mov_b32 s79, 131072 // (op 1858) call()

s_and_b32 s52, s7, 4294967294 // (op 1984) 
v_lshrrev_b32 v14, 2, v1 // (op 1984) 
s_add_u32 s53, s0, s52 // (op 1984) 

v_lshlrev_b32 v16, 1, v14 // (op 1984) 

v_and_b32 v14, -4, v1 // (op 1984) 
s_mul_i32 s52, s16, s8 // (op 1984) 
s_add_u32 s52, s52, s53 // (op 1984) 

v_sub_u32 v17, v1, v14 // (op 1984) 

s_mul_hi_u32 s53, s52, s13 // (op 1984) 
s_sub_u32 s54, s52, s53 // (op 1984) 
s_lshr_b32 s55, s54, 1 // (op 1984) 

s_add_u32 s54, s55, s53 // (op 1984) 


s_lshr_b32 s53, s54, s11 // (op 1984) 

s_and_b32 s54, s11, 2147483648 // (op 1984) 
s_cmp_eq_i32 s54, 0 // (op 1984) 
s_mov_b32 s55, scc // (op 1984) 

s_cmp_eq_i32 s55, 0 // (op 1984) 
s_cselect_b32 s54, s52, s53 // (op 1984) 


s_mul_hi_u32 s53, s54, s12 // (op 1984) 
s_sub_u32 s55, s54, s53 // (op 1984) 
s_lshr_b32 s84, s55, 1 // (op 1984) 

s_add_u32 s55, s84, s53 // (op 1984) 


s_lshr_b32 s53, s55, s10 // (op 1984) 

s_and_b32 s55, s10, 2147483648 // (op 1984) 
s_cmp_eq_i32 s55, 0 // (op 1984) 
s_mov_b32 s84, scc // (op 1984) 

s_cmp_eq_i32 s84, 0 // (op 1984) 
s_cselect_b32 s55, s54, s53 // (op 1984) 


s_mul_i32 s53, s55, s14 // (op 1984) 

s_sub_u32 s55, s54, s53 // (op 1984) 

v_lshl_add_u32 v14, s55, 6, v16 // (op 1984) 


v_mul_lo_u32 v16, v14, s44 // (op 1984) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 1984) 
s_mul_i32 s53, s54, s84 // (op 1984) 


s_sub_u32 s54, s52, s53 // (op 1984) 


v_lshl_add_u32 v14, s54, 2, v17 // (op 1984) 


v_add_u32 v17, v14, v16 // (op 1984) 


s_mul_i32 s53, s44, s23 // (op 1986) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 1986) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 1986) least significant
s_mul_hi_u32 s55, s44, s22 // (op 1986) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1986) most significant: sum
s_add_u32 s53, s53, s55 // (op 1986) most significant: sum


s_mov_b32 s80, s46 // (op 1986) reg expression
s_mov_b32 s81, s47 // (op 1986) reg expression
s_mov_b32 s82, s52 // (op 1986) reg expression
s_mov_b32 s83, 131072 // (op 1986) call()

v_and_b32 v14, -4, v1 // (op 2145) 
v_sub_u32 v16, v1, v14 // (op 2145) 

v_lshrrev_b32 v14, 2, v1 // (op 2145) 
v_lshl_add_u32 v18, v14, 3, v16 // (op 2145) 


s_and_b32 s52, s7, 4294967294 // (op 829) 
s_cmp_lt_u32 s52, s7 // (op 828) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_828 // (op 828) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_828 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_828:

s_add_u32 s52, s52, 1 // (op 830) 
v_mov_b64 v[20:21], s[44:45] // (op 722) convert

v_mov_b32 v9, v17 // (op 722) 
v_mov_b32 v14, v9 // (op 722) 
buffer_load_ubyte v22, v14, s[80:83], 0 offen offset: 0 // (op 722) Load value
v_add_u32 v14, v14, v20 // (op 722) 
buffer_load_ubyte v23, v14, s[80:83], 0 offen offset: 0 // (op 722) Load value



v_add_u32 v17, 4, v17 // (op 1976) 


s_barrier  // (op 2455) 
v_mov_b32 v9, v12 // (op 728) 




buffer_load_dwordx4 v[24:27], v9, s[68:71], 0 offen // (op 728) Load value
s_waitcnt vmcnt(2)
ds_write_b8 v18, v22 // (op 723) Store local data 
buffer_load_dwordx4 v[28:31], v9, s[68:71], 0 offen offset:64 // (op 728) Load value

v_add_u32 v12, 128, v12 // (op 1552) 
s_waitcnt vmcnt(2)
ds_write_b8 v18, v23 offset:4 // (op 723) Store local data 



v_mov_b32 v9, v13 // (op 730) 
v_mov_b32 v14, v9 // (op 730) reg expression

buffer_load_dwordx4 v[20:23], v14, s[72:75], 0 offen // (op 730) Load value



v_mov_b32 v14, v15 // (op 732) 

buffer_load_ubyte v16, v14, s[76:79], 0 offen offset: 0 // (op 732) Load value

v_add_u32 v15, 4, v15 // (op 1848) 


s_lshl_b64 s[54:55], s[36:37], 5 // (op 842) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 842) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 842) 

v_add_u32 v14, v9, s54 // (op 842) 


s_waitcnt lgkmcnt(0)
s_barrier  // (op 2458) 
buffer_load_dwordx4 v[32:35], v14, s[72:75], 0 offen // (op 842) Load value





ds_read_u8 v9, v0 // (op 725) Load local data 



v_add_u32 v13, 64, v13 // (op 1695) 
ds_read_u8 v14, v0 offset:128 // (op 834) Load local data 
s_waitcnt vmcnt(1) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:31], v[20:23], a[0:3], v16, v9 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 726) 


s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:31], v[32:35], a[4:7], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 835) 




s_cmp_lt_u32 s52, s7 // (op 828) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_828 // (op 828) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_828 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_828:






s_barrier  // (op 2618) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 243) 
s_cmp_lg_u32 s18, 0 // (op 269) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Send_Tile_269 // (op 269) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Send_Tile_269



s_nop 0
buffer_store_dword a0, v8, s[64:67], 0 offen sc1 // (op 266) Store value
buffer_store_dword a1, v8, s[64:67], 0 offen offset:256 sc1 // (op 266) Store value
buffer_store_dword a2, v8, s[64:67], 0 offen offset:512 sc1 // (op 266) Store value
buffer_store_dword a3, v8, s[64:67], 0 offen offset:768 sc1 // (op 266) Store value



buffer_store_dword a4, v8, s[64:67], 0 offen offset:128 sc1 // (op 979) Store value
buffer_store_dword a5, v8, s[64:67], 0 offen offset:384 sc1 // (op 979) Store value
buffer_store_dword a6, v8, s[64:67], 0 offen offset:640 sc1 // (op 979) Store value
buffer_store_dword a7, v8, s[64:67], 0 offen offset:896 sc1 // (op 979) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 270) 
v_cmp_eq_i32 vcc, v1, 0 // (op 274) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_274 // (op 274) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_274
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
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_274 // (op 274) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_274
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Wave0_Store_Flag_274:
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Wave0_Store_Flag_274:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Send_Tile_269 // (op 269) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Send_Tile_269
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Send_Tile_269:
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Send_Tile_269:

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
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Receive_Tile_285 // (op 285) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Receive_Tile_285
s_mov_b32 s52, 0 // (op 251) call()
s_mul_i32 s53, s52, s8 // (op 250) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 250) 
s_mov_b32 s84, s54 // (op 250) convert

s_sub_u32 s54, s84, s19 // (op 250) 

s_sub_u32 s55, s54, 1 // (op 250) 

s_cmp_lt_u32 s53, s55 // (op 250) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_250 // (op 250) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_250 if false)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_ReceiveTileLoop_250:
s_add_u32 s53, s16, 1 // (op 2317) 
v_lshrrev_b32 v8, 5, v1 // (op 2317) 
s_add_u32 s54, s53, s52 // (op 2317) 
s_lshl_b32 s54, s54, 4 // (op 2317) 

v_lshlrev_b32 v9, 2, v8 // (op 2317) 

v_add_lshl_u32 v8, s54, v9, 6 // (op 2317) 


v_and_b32 v9, -32, v1 // (op 2317) 
v_sub_u32 v12, v1, v9 // (op 2317) 

v_add_lshl_u32 v9, v8, v12, 2 // (op 2317) 


s_lshl_b32 s53, s15, 12 // (op 2319) 
s_mov_b32 s64, s28 // (op 2319) reg expression
s_mov_b32 s65, s29 // (op 2319) reg expression
s_mov_b32 s66, s53 // (op 2319) reg expression
s_mov_b32 s67, 131072 // (op 2319) call()

RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_289:
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
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_289 // (op 289) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_DoWhileTop_Global_sync_spin_loop_289 if true)
s_barrier  // (op 293) 
v_cmp_eq_i32 vcc, v1, 0 // (op 292) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_292 // (op 292) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_292
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
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_292 // (op 292) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_292
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_292:
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_292:




buffer_load_dword v12, v9, s[64:67], 0 offen // (op 304) Load value
buffer_load_dword v13, v9, s[64:67], 0 offen offset:256 // (op 304) Load value
buffer_load_dword v14, v9, s[64:67], 0 offen offset:512 // (op 304) Load value
buffer_load_dword v15, v9, s[64:67], 0 offen offset:768 // (op 304) Load value
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




buffer_load_dword v12, v9, s[64:67], 0 offen offset:128 // (op 989) Load value
buffer_load_dword v13, v9, s[64:67], 0 offen offset:384 // (op 989) Load value
buffer_load_dword v14, v9, s[64:67], 0 offen offset:640 // (op 989) Load value
buffer_load_dword v15, v9, s[64:67], 0 offen offset:896 // (op 989) Load value
v_accvgpr_read v16, a4 // (op 990) reg expression
v_accvgpr_read v17, a5 // (op 990) reg expression
v_accvgpr_read v18, a6 // (op 990) reg expression
v_accvgpr_read v19, a7 // (op 990) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v16, v16, v12 // (op 991) 
s_waitcnt vmcnt(2)
v_add_f32 v17, v17, v13 // (op 991) 
s_waitcnt vmcnt(1)
v_add_f32 v18, v18, v14 // (op 991) 
s_waitcnt vmcnt(0)
v_add_f32 v19, v19, v15 // (op 991) 

v_accvgpr_write a4, v16 // (op 992) reg expression
v_accvgpr_write a5, v17 // (op 992) reg expression
v_accvgpr_write a6, v18 // (op 992) reg expression
v_accvgpr_write a7, v19 // (op 992) reg expression

s_add_u32 s52, s52, 1 // (op 252) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 250) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 250) 
s_mov_b32 s84, s54 // (op 250) convert

s_sub_u32 s54, s84, s19 // (op 250) 

s_sub_u32 s55, s54, 1 // (op 250) 

s_cmp_lt_u32 s53, s55 // (op 250) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_ReceiveTileLoop_250 // (op 250) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_ReceiveTileLoop_250 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_ReceiveTileLoop_250:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Receive_Tile_285 // (op 285) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Receive_Tile_285
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Receive_Tile_285:
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Receive_Tile_285:

s_cmp_eq_i32 s18, 0 // (op 328) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_328 // (op 328) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_328
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 106) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_106 // (op 106) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_106
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


v_accvgpr_read v12, a4 // (op 1008) 
v_accvgpr_read v13, a5 // (op 1008) 
v_accvgpr_read v14, a6 // (op 1008) 
v_accvgpr_read v15, a7 // (op 1008) 
v_mul_f32 v16, v7, v12 // (op 1008) 
v_mul_f32 v17, v7, v13 // (op 1008) 
v_mul_f32 v18, v7, v14 // (op 1008) 
v_mul_f32 v19, v7, v15 // (op 1008) 



s_lshl_b32 s52, s24, 7 // (op 1010) 
v_add_u32 v8, v11, s52 // (op 1010) 


buffer_store_dwordx4 v[16:19], v8, s[60:63], 0 offen // (op 1010) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_106 // (op 106) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_106
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_106:


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




s_lshl_b32 s52, s4, 7 // (op 1032) 
v_add_u32 v8, v10, s52 // (op 1032) 


buffer_load_dwordx4 v[12:15], v8, s[56:59], 0 offen // (op 1032) Load value

v_accvgpr_read v16, a4 // (op 1029) 
v_accvgpr_read v17, a5 // (op 1029) 
v_accvgpr_read v18, a6 // (op 1029) 
v_accvgpr_read v19, a7 // (op 1029) 
v_mul_f32 v20, v7, v16 // (op 1029) 
v_mul_f32 v21, v7, v17 // (op 1029) 
v_mul_f32 v22, v7, v18 // (op 1029) 
v_mul_f32 v23, v7, v19 // (op 1029) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v6, v12 // (op 1033) 
v_mul_f32 v17, v6, v13 // (op 1033) 
v_mul_f32 v18, v6, v14 // (op 1033) 
v_mul_f32 v19, v6, v15 // (op 1033) 

v_add_f32 v12, v16, v20 // (op 1030) 
v_add_f32 v13, v17, v21 // (op 1030) 
v_add_f32 v14, v18, v22 // (op 1030) 
v_add_f32 v15, v19, v23 // (op 1030) 




s_lshl_b32 s52, s24, 7 // (op 1031) 
v_add_u32 v8, v11, s52 // (op 1031) 


buffer_store_dwordx4 v[12:15], v8, s[60:63], 0 offen // (op 1031) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_106:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_328 // (op 328) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_328
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_328:
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_328:


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
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_SKStreamTileLoop_241 // (op 241) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_SKStreamTileLoop_241 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_SKStreamTileLoop_241:











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
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_396 // (op 396) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_396 if false)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_DPStreamTileLoop_396:
v_lshrrev_b32 v0, 7, v1 // (op 2115) 
s_mul_i32 s15, s16, s50 // (op 2115) 
s_add_u32 s15, s15, s0 // (op 2115) 
v_and_b32 v5, 63, v1 // (op 2115) 
s_add_u32 s6, s15, s9 // (op 2115) 

v_lshrrev_b32 v8, 4, v5 // (op 2115) 
s_mul_hi_u32 s15, s6, s13 // (op 2115) 
v_lshlrev_b32 v9, 2, v8 // (op 2115) 

s_sub_u32 s7, s6, s15 // (op 2115) 
v_lshl_add_u32 v8, v0, 4, v9 // (op 2115) 


s_lshr_b32 s8, s7, 1 // (op 2115) 

v_lshrrev_b32 v0, 2, v5 // (op 2115) 

s_add_u32 s7, s8, s15 // (op 2115) 


v_and_b32 v5, 3, v0 // (op 2115) 

s_lshr_b32 s8, s7, s11 // (op 2115) 

v_and_b32 v0, 3, v1 // (op 2115) 
s_and_b32 s7, s11, 2147483648 // (op 2115) 
v_lshl_add_u32 v9, v5, 2, v0 // (op 2115) 


s_cmp_eq_i32 s7, 0 // (op 2115) 
s_mov_b32 s15, scc // (op 2115) 

v_lshrrev_b32 v0, 6, v1 // (op 2115) 
s_cmp_eq_i32 s15, 0 // (op 2115) 
s_cselect_b32 s7, s6, s8 // (op 2115) 



v_and_b32 v5, 1, v0 // (op 2115) 

s_mul_hi_u32 s6, s7, s12 // (op 2115) 
v_lshl_add_u32 v0, v5, 4, v9 // (op 2115) 


s_sub_u32 s8, s7, s6 // (op 2115) 
s_lshr_b32 s15, s8, 1 // (op 2115) 

s_add_u32 s8, s15, s6 // (op 2115) 


s_lshr_b32 s6, s8, s10 // (op 2115) 

s_and_b32 s8, s10, 2147483648 // (op 2115) 
s_cmp_eq_i32 s8, 0 // (op 2115) 
s_mov_b32 s15, scc // (op 2115) 

s_cmp_eq_i32 s15, 0 // (op 2115) 
s_cselect_b32 s8, s7, s6 // (op 2115) 


s_mul_i32 s6, s8, s14 // (op 2115) 
v_lshl_add_u32 v5, s8, 4, v8 // (op 2115) 


s_sub_u32 s8, s7, s6 // (op 2115) 


v_lshl_add_u32 v8, s8, 6, v0 // (op 2115) 


v_mul_lo_u32 v0, v8, s4 // (op 2115) 

v_add_lshl_u32 v8, v5, v0, 2 // (op 2115) 


s_mul_i32 s19, s4, s23 // (op 2117) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 2117) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 2117) least significant
s_mul_hi_u32 s6, s4, s22 // (op 2117) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2117) most significant: sum
s_add_u32 s19, s19, s6 // (op 2117) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 2117) 

s_mov_b32 s56, s20 // (op 2117) reg expression
s_mov_b32 s57, s21 // (op 2117) reg expression
s_mov_b32 s58, s28 // (op 2117) reg expression
s_mov_b32 s59, 131072 // (op 2117) call()

v_lshrrev_b32 v0, 7, v1 // (op 2215) 
s_mul_i32 s15, s16, s50 // (op 2215) 
s_add_u32 s15, s15, s0 // (op 2215) 
v_and_b32 v5, 63, v1 // (op 2215) 
s_add_u32 s6, s15, s9 // (op 2215) 

v_lshrrev_b32 v9, 4, v5 // (op 2215) 
s_mul_hi_u32 s15, s6, s13 // (op 2215) 
v_lshlrev_b32 v10, 2, v9 // (op 2215) 

s_sub_u32 s7, s6, s15 // (op 2215) 
v_lshl_add_u32 v9, v0, 4, v10 // (op 2215) 


s_lshr_b32 s8, s7, 1 // (op 2215) 

v_lshrrev_b32 v0, 2, v5 // (op 2215) 

s_add_u32 s7, s8, s15 // (op 2215) 


v_and_b32 v5, 3, v0 // (op 2215) 

s_lshr_b32 s8, s7, s11 // (op 2215) 

v_and_b32 v0, 3, v1 // (op 2215) 
s_and_b32 s7, s11, 2147483648 // (op 2215) 
v_lshl_add_u32 v10, v5, 2, v0 // (op 2215) 


s_cmp_eq_i32 s7, 0 // (op 2215) 
s_mov_b32 s15, scc // (op 2215) 

v_lshrrev_b32 v0, 6, v1 // (op 2215) 
s_cmp_eq_i32 s15, 0 // (op 2215) 
s_cselect_b32 s7, s6, s8 // (op 2215) 



v_and_b32 v5, 1, v0 // (op 2215) 

s_mul_hi_u32 s6, s7, s12 // (op 2215) 
v_lshl_add_u32 v0, v5, 4, v10 // (op 2215) 


s_sub_u32 s8, s7, s6 // (op 2215) 
s_lshr_b32 s15, s8, 1 // (op 2215) 

s_add_u32 s8, s15, s6 // (op 2215) 


s_lshr_b32 s6, s8, s10 // (op 2215) 

s_and_b32 s8, s10, 2147483648 // (op 2215) 
s_cmp_eq_i32 s8, 0 // (op 2215) 
s_mov_b32 s15, scc // (op 2215) 

s_cmp_eq_i32 s15, 0 // (op 2215) 
s_cselect_b32 s8, s7, s6 // (op 2215) 


s_mul_i32 s6, s8, s14 // (op 2215) 
v_lshl_add_u32 v5, s8, 4, v9 // (op 2215) 


s_sub_u32 s8, s7, s6 // (op 2215) 


v_lshl_add_u32 v9, s8, 6, v0 // (op 2215) 


v_mul_lo_u32 v0, v9, s24 // (op 2215) 

v_add_lshl_u32 v9, v5, v0, 2 // (op 2215) 


s_mul_i32 s19, s24, s23 // (op 2217) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 2217) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 2217) least significant
s_mul_hi_u32 s6, s24, s22 // (op 2217) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2217) most significant: sum
s_add_u32 s19, s19, s6 // (op 2217) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 2217) 

s_mov_b32 s60, s26 // (op 2217) reg expression
s_mov_b32 s61, s27 // (op 2217) reg expression
s_mov_b32 s62, s28 // (op 2217) reg expression
s_mov_b32 s63, 131072 // (op 2217) call()

v_accvgpr_write a0, 0.00000 // (op 153) call()
v_accvgpr_write a1, 0.00000 // (op 153) call()
v_accvgpr_write a2, 0.00000 // (op 153) call()
v_accvgpr_write a3, 0.00000 // (op 153) call()
v_accvgpr_write a4, 0.00000 // (op 509) call()
v_accvgpr_write a5, 0.00000 // (op 509) call()
v_accvgpr_write a6, 0.00000 // (op 509) call()
v_accvgpr_write a7, 0.00000 // (op 509) call()
v_lshrrev_b32 v0, 7, v1 // (op 1589) 
s_mul_i32 s15, s16, s50 // (op 1589) 
s_add_u32 s15, s15, s0 // (op 1589) 
v_and_b32 v5, 15, v1 // (op 1589) 
s_add_u32 s6, s15, s9 // (op 1589) 

v_lshl_add_u32 v10, v0, 4, v5 // (op 1589) 


s_mul_hi_u32 s15, s6, s13 // (op 1589) 
v_and_b32 v0, 48, v1 // (op 1589) 
s_sub_u32 s7, s6, s15 // (op 1589) 
s_lshr_b32 s8, s7, 1 // (op 1589) 

s_add_u32 s7, s8, s15 // (op 1589) 


s_lshr_b32 s8, s7, s11 // (op 1589) 

s_and_b32 s7, s11, 2147483648 // (op 1589) 
s_cmp_eq_i32 s7, 0 // (op 1589) 
s_mov_b32 s15, scc // (op 1589) 

s_cmp_eq_i32 s15, 0 // (op 1589) 
s_cselect_b32 s7, s6, s8 // (op 1589) 


s_mul_hi_u32 s8, s7, s12 // (op 1589) 
s_sub_u32 s15, s7, s8 // (op 1589) 
s_lshr_b32 s18, s15, 1 // (op 1589) 

s_add_u32 s15, s18, s8 // (op 1589) 


s_lshr_b32 s8, s15, s10 // (op 1589) 

s_and_b32 s15, s10, 2147483648 // (op 1589) 
s_cmp_eq_i32 s15, 0 // (op 1589) 
s_mov_b32 s18, scc // (op 1589) 

s_cmp_eq_i32 s18, 0 // (op 1589) 
s_cselect_b32 s15, s7, s8 // (op 1589) 


v_lshl_add_u32 v5, s15, 4, v10 // (op 1589) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1589) 
s_mul_i32 s8, s7, s18 // (op 1589) 


s_sub_u32 s7, s6, s8 // (op 1589) 


v_lshl_add_u32 v10, s7, 7, v0 // (op 1589) 


v_mul_lo_u32 v0, v5, s30 // (op 1589) 
v_add_u32 v0, v0, v10 // (op 1589) 


s_mul_i32 s19, s30, s35 // (op 1591) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 1591) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 1591) least significant
s_mul_hi_u32 s6, s30, s34 // (op 1591) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1591) most significant: sum
s_add_u32 s19, s19, s6 // (op 1591) most significant: sum


s_mov_b32 s68, s32 // (op 1591) reg expression
s_mov_b32 s69, s33 // (op 1591) reg expression
s_mov_b32 s70, s18 // (op 1591) reg expression
s_mov_b32 s71, 131072 // (op 1591) call()

v_lshrrev_b32 v5, 7, v1 // (op 1608) 
s_mul_i32 s15, s16, s50 // (op 1608) 
s_add_u32 s15, s15, s0 // (op 1608) 
v_and_b32 v10, 15, v1 // (op 1608) 
s_add_u32 s6, s15, s9 // (op 1608) 

v_lshl_add_u32 v11, v5, 4, v10 // (op 1608) 


s_mul_hi_u32 s15, s6, s13 // (op 1608) 
v_and_b32 v5, 48, v1 // (op 1608) 
s_sub_u32 s7, s6, s15 // (op 1608) 
s_lshr_b32 s8, s7, 1 // (op 1608) 

s_add_u32 s7, s8, s15 // (op 1608) 


s_lshr_b32 s8, s7, s11 // (op 1608) 

s_and_b32 s7, s11, 2147483648 // (op 1608) 
s_cmp_eq_i32 s7, 0 // (op 1608) 
s_mov_b32 s15, scc // (op 1608) 

s_cmp_eq_i32 s15, 0 // (op 1608) 
s_cselect_b32 s7, s6, s8 // (op 1608) 


s_mul_hi_u32 s8, s7, s12 // (op 1608) 
s_sub_u32 s15, s7, s8 // (op 1608) 
s_lshr_b32 s18, s15, 1 // (op 1608) 

s_add_u32 s15, s18, s8 // (op 1608) 


s_lshr_b32 s8, s15, s10 // (op 1608) 

s_and_b32 s15, s10, 2147483648 // (op 1608) 
s_cmp_eq_i32 s15, 0 // (op 1608) 
s_mov_b32 s18, scc // (op 1608) 

s_cmp_eq_i32 s18, 0 // (op 1608) 
s_cselect_b32 s15, s7, s8 // (op 1608) 


v_lshl_add_u32 v10, s15, 4, v11 // (op 1608) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1608) 
s_mul_i32 s8, s7, s18 // (op 1608) 


s_sub_u32 s7, s6, s8 // (op 1608) 


v_lshl_add_u32 v11, s7, 7, v5 // (op 1608) 


v_mul_lo_u32 v5, v10, s30 // (op 1608) 
v_add_u32 v5, v5, v11 // (op 1608) 


s_mul_i32 s19, s30, s35 // (op 1610) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 1610) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 1610) least significant
s_mul_hi_u32 s6, s30, s34 // (op 1610) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1610) most significant: sum
s_add_u32 s19, s19, s6 // (op 1610) most significant: sum


s_mov_b32 s68, s32 // (op 1610) reg expression
s_mov_b32 s69, s33 // (op 1610) reg expression
s_mov_b32 s70, s18 // (op 1610) reg expression
s_mov_b32 s71, 131072 // (op 1610) call()

v_lshrrev_b32 v10, 6, v1 // (op 1736) 
s_mul_i32 s15, s16, s50 // (op 1736) 
s_add_u32 s15, s15, s0 // (op 1736) 
v_and_b32 v11, 1, v10 // (op 1736) 

s_add_u32 s6, s15, s9 // (op 1736) 

v_and_b32 v10, 15, v1 // (op 1736) 
s_mul_hi_u32 s15, s6, s13 // (op 1736) 
v_lshl_add_u32 v12, v11, 4, v10 // (op 1736) 


s_sub_u32 s7, s6, s15 // (op 1736) 
v_and_b32 v10, 63, v1 // (op 1736) 
s_lshr_b32 s8, s7, 1 // (op 1736) 

v_lshrrev_b32 v11, 4, v10 // (op 1736) 

s_add_u32 s7, s8, s15 // (op 1736) 


v_lshlrev_b32 v10, 5, v11 // (op 1736) 

s_lshr_b32 s8, s7, s11 // (op 1736) 

s_and_b32 s7, s11, 2147483648 // (op 1736) 
s_cmp_eq_i32 s7, 0 // (op 1736) 
s_mov_b32 s15, scc // (op 1736) 

s_cmp_eq_i32 s15, 0 // (op 1736) 
s_cselect_b32 s7, s6, s8 // (op 1736) 


s_mul_hi_u32 s8, s7, s12 // (op 1736) 
s_sub_u32 s15, s7, s8 // (op 1736) 
s_lshr_b32 s18, s15, 1 // (op 1736) 

s_add_u32 s15, s18, s8 // (op 1736) 


s_lshr_b32 s8, s15, s10 // (op 1736) 

s_and_b32 s15, s10, 2147483648 // (op 1736) 
s_cmp_eq_i32 s15, 0 // (op 1736) 
s_mov_b32 s18, scc // (op 1736) 

s_cmp_eq_i32 s18, 0 // (op 1736) 
s_cselect_b32 s15, s7, s8 // (op 1736) 


s_mul_i32 s8, s15, s14 // (op 1736) 

s_sub_u32 s15, s7, s8 // (op 1736) 

v_lshl_add_u32 v11, s15, 6, v12 // (op 1736) 


v_mul_lo_u32 v13, v11, s37 // (op 1736) most significant: low of low * high
v_mul_hi_u32 v14, v11, s36 // (op 1736) most significant: high of low * low
v_mul_lo_u32 v12, v11, s36 // (op 1736) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 1736) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1736) 
s_mul_i32 s8, s7, s18 // (op 1736) 


s_sub_u32 s7, s6, s8 // (op 1736) 


v_lshl_add_u32 v11, s7, 7, v10 // (op 1736) 


v_mov_b32 v15, 0 // (op 1736) convert
v_mov_b32 v14, v11 // (op 1736) convert

v_add_co_u32 v10, s[18:19], v14, v12 // (op 1736) least significant half
v_addc_co_u32 v11, s[18:19], v15, v13, s[18:19] // (op 1736) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 1736) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 1736) 

v_mov_b32 v10, v12 // (op 1736) convert

s_mul_i32 s19, s36, s23 // (op 1738) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 1738) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 1738) least significant
s_mul_hi_u32 s6, s36, s22 // (op 1738) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1738) most significant: sum
s_add_u32 s19, s19, s6 // (op 1738) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 1738) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 1738) 

s_mov_b32 s72, s38 // (op 1738) reg expression
s_mov_b32 s73, s39 // (op 1738) reg expression
s_mov_b32 s74, s18 // (op 1738) reg expression
s_mov_b32 s75, 131072 // (op 1738) call()

v_lshrrev_b32 v11, 6, v1 // (op 1757) 
s_mul_i32 s15, s16, s50 // (op 1757) 
s_add_u32 s15, s15, s0 // (op 1757) 
v_and_b32 v12, 1, v11 // (op 1757) 

s_add_u32 s6, s15, s9 // (op 1757) 

v_and_b32 v11, 15, v1 // (op 1757) 
s_mul_hi_u32 s15, s6, s13 // (op 1757) 
v_lshl_add_u32 v13, v12, 4, v11 // (op 1757) 


s_sub_u32 s7, s6, s15 // (op 1757) 
v_and_b32 v11, 63, v1 // (op 1757) 
s_lshr_b32 s8, s7, 1 // (op 1757) 

v_lshrrev_b32 v12, 4, v11 // (op 1757) 

s_add_u32 s7, s8, s15 // (op 1757) 


v_lshlrev_b32 v11, 5, v12 // (op 1757) 

s_lshr_b32 s8, s7, s11 // (op 1757) 

s_and_b32 s7, s11, 2147483648 // (op 1757) 
s_cmp_eq_i32 s7, 0 // (op 1757) 
s_mov_b32 s15, scc // (op 1757) 

s_cmp_eq_i32 s15, 0 // (op 1757) 
s_cselect_b32 s7, s6, s8 // (op 1757) 


s_mul_hi_u32 s8, s7, s12 // (op 1757) 
s_sub_u32 s15, s7, s8 // (op 1757) 
s_lshr_b32 s18, s15, 1 // (op 1757) 

s_add_u32 s15, s18, s8 // (op 1757) 


s_lshr_b32 s8, s15, s10 // (op 1757) 

s_and_b32 s15, s10, 2147483648 // (op 1757) 
s_cmp_eq_i32 s15, 0 // (op 1757) 
s_mov_b32 s18, scc // (op 1757) 

s_cmp_eq_i32 s18, 0 // (op 1757) 
s_cselect_b32 s15, s7, s8 // (op 1757) 


s_mul_i32 s8, s15, s14 // (op 1757) 

s_sub_u32 s15, s7, s8 // (op 1757) 

v_lshl_add_u32 v12, s15, 6, v13 // (op 1757) 


v_mul_lo_u32 v15, v12, s37 // (op 1757) most significant: low of low * high
v_mul_hi_u32 v13, v12, s36 // (op 1757) most significant: high of low * low
v_mul_lo_u32 v14, v12, s36 // (op 1757) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 1757) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1757) 
s_mul_i32 s8, s7, s18 // (op 1757) 


s_sub_u32 s7, s6, s8 // (op 1757) 


v_lshl_add_u32 v12, s7, 7, v11 // (op 1757) 


v_mov_b32 v17, 0 // (op 1757) convert
v_mov_b32 v16, v12 // (op 1757) convert

v_add_co_u32 v12, s[18:19], v16, v14 // (op 1757) least significant half
v_addc_co_u32 v13, s[18:19], v17, v15, s[18:19] // (op 1757) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 1757) 


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1757) 

v_mov_b32 v11, v14 // (op 1757) convert

s_mul_i32 s19, s36, s23 // (op 1759) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 1759) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 1759) least significant
s_mul_hi_u32 s6, s36, s22 // (op 1759) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1759) most significant: sum
s_add_u32 s19, s19, s6 // (op 1759) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 1759) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 1759) 

s_mov_b32 s72, s38 // (op 1759) reg expression
s_mov_b32 s73, s39 // (op 1759) reg expression
s_mov_b32 s74, s18 // (op 1759) reg expression
s_mov_b32 s75, 131072 // (op 1759) call()

v_lshrrev_b32 v12, 7, v1 // (op 1879) 
s_mul_i32 s15, s16, s50 // (op 1879) 
s_add_u32 s15, s15, s0 // (op 1879) 
v_and_b32 v13, 15, v1 // (op 1879) 
s_add_u32 s6, s15, s9 // (op 1879) 

v_lshl_add_u32 v14, v12, 4, v13 // (op 1879) 


s_mul_hi_u32 s15, s6, s13 // (op 1879) 
v_and_b32 v12, 63, v1 // (op 1879) 
s_sub_u32 s7, s6, s15 // (op 1879) 
v_lshrrev_b32 v13, 4, v12 // (op 1879) 

s_lshr_b32 s8, s7, 1 // (op 1879) 

s_add_u32 s7, s8, s15 // (op 1879) 


s_lshr_b32 s8, s7, s11 // (op 1879) 

s_and_b32 s7, s11, 2147483648 // (op 1879) 
s_cmp_eq_i32 s7, 0 // (op 1879) 
s_mov_b32 s15, scc // (op 1879) 

s_cmp_eq_i32 s15, 0 // (op 1879) 
s_cselect_b32 s7, s6, s8 // (op 1879) 


s_mul_hi_u32 s8, s7, s12 // (op 1879) 
s_sub_u32 s15, s7, s8 // (op 1879) 
s_lshr_b32 s18, s15, 1 // (op 1879) 

s_add_u32 s15, s18, s8 // (op 1879) 


s_lshr_b32 s8, s15, s10 // (op 1879) 

s_and_b32 s15, s10, 2147483648 // (op 1879) 
s_cmp_eq_i32 s15, 0 // (op 1879) 
s_mov_b32 s18, scc // (op 1879) 

s_cmp_eq_i32 s18, 0 // (op 1879) 
s_cselect_b32 s15, s7, s8 // (op 1879) 


v_lshl_add_u32 v12, s15, 4, v14 // (op 1879) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1879) 
s_mul_i32 s8, s7, s18 // (op 1879) 


s_sub_u32 s7, s6, s8 // (op 1879) 


v_lshl_add_u32 v14, s7, 2, v13 // (op 1879) 


v_mul_lo_u32 v13, v12, s40 // (op 1879) 
v_add_u32 v13, v13, v14 // (op 1879) 


s_mul_i32 s19, s40, s35 // (op 1881) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 1881) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 1881) least significant
s_mul_hi_u32 s6, s40, s34 // (op 1881) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1881) most significant: sum
s_add_u32 s19, s19, s6 // (op 1881) most significant: sum


s_mov_b32 s76, s42 // (op 1881) reg expression
s_mov_b32 s77, s43 // (op 1881) reg expression
s_mov_b32 s78, s18 // (op 1881) reg expression
s_mov_b32 s79, 131072 // (op 1881) call()

v_lshrrev_b32 v12, 7, v1 // (op 1898) 
s_mul_i32 s15, s16, s50 // (op 1898) 
s_add_u32 s15, s15, s0 // (op 1898) 
v_and_b32 v14, 15, v1 // (op 1898) 
s_add_u32 s6, s15, s9 // (op 1898) 

v_lshl_add_u32 v15, v12, 4, v14 // (op 1898) 


s_mul_hi_u32 s15, s6, s13 // (op 1898) 
v_and_b32 v12, 63, v1 // (op 1898) 
s_sub_u32 s7, s6, s15 // (op 1898) 
v_lshrrev_b32 v14, 4, v12 // (op 1898) 

s_lshr_b32 s8, s7, 1 // (op 1898) 

s_add_u32 s7, s8, s15 // (op 1898) 


s_lshr_b32 s8, s7, s11 // (op 1898) 

s_and_b32 s7, s11, 2147483648 // (op 1898) 
s_cmp_eq_i32 s7, 0 // (op 1898) 
s_mov_b32 s15, scc // (op 1898) 

s_cmp_eq_i32 s15, 0 // (op 1898) 
s_cselect_b32 s7, s6, s8 // (op 1898) 


s_mul_hi_u32 s8, s7, s12 // (op 1898) 
s_sub_u32 s15, s7, s8 // (op 1898) 
s_lshr_b32 s18, s15, 1 // (op 1898) 

s_add_u32 s15, s18, s8 // (op 1898) 


s_lshr_b32 s8, s15, s10 // (op 1898) 

s_and_b32 s15, s10, 2147483648 // (op 1898) 
s_cmp_eq_i32 s15, 0 // (op 1898) 
s_mov_b32 s18, scc // (op 1898) 

s_cmp_eq_i32 s18, 0 // (op 1898) 
s_cselect_b32 s15, s7, s8 // (op 1898) 


v_lshl_add_u32 v12, s15, 4, v15 // (op 1898) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1898) 
s_mul_i32 s8, s7, s18 // (op 1898) 


s_sub_u32 s7, s6, s8 // (op 1898) 


v_lshl_add_u32 v15, s7, 2, v14 // (op 1898) 


v_mul_lo_u32 v14, v12, s40 // (op 1898) 
v_add_u32 v14, v14, v15 // (op 1898) 


s_mul_i32 s19, s40, s35 // (op 1900) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 1900) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 1900) least significant
s_mul_hi_u32 s6, s40, s34 // (op 1900) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1900) most significant: sum
s_add_u32 s19, s19, s6 // (op 1900) most significant: sum


s_mov_b32 s76, s42 // (op 1900) reg expression
s_mov_b32 s77, s43 // (op 1900) reg expression
s_mov_b32 s78, s18 // (op 1900) reg expression
s_mov_b32 s79, 131072 // (op 1900) call()

v_lshrrev_b32 v12, 2, v1 // (op 2007) 
s_mul_i32 s15, s16, s50 // (op 2007) 
s_add_u32 s15, s15, s0 // (op 2007) 
v_lshlrev_b32 v15, 1, v12 // (op 2007) 

s_add_u32 s6, s15, s9 // (op 2007) 

v_and_b32 v12, -4, v1 // (op 2007) 
s_mul_hi_u32 s15, s6, s13 // (op 2007) 
v_sub_u32 v16, v1, v12 // (op 2007) 

s_sub_u32 s7, s6, s15 // (op 2007) 
s_lshr_b32 s8, s7, 1 // (op 2007) 

s_add_u32 s7, s8, s15 // (op 2007) 


s_lshr_b32 s8, s7, s11 // (op 2007) 

s_and_b32 s7, s11, 2147483648 // (op 2007) 
s_cmp_eq_i32 s7, 0 // (op 2007) 
s_mov_b32 s15, scc // (op 2007) 

s_cmp_eq_i32 s15, 0 // (op 2007) 
s_cselect_b32 s7, s6, s8 // (op 2007) 


s_mul_hi_u32 s8, s7, s12 // (op 2007) 
s_sub_u32 s15, s7, s8 // (op 2007) 
s_lshr_b32 s18, s15, 1 // (op 2007) 

s_add_u32 s15, s18, s8 // (op 2007) 


s_lshr_b32 s8, s15, s10 // (op 2007) 

s_and_b32 s15, s10, 2147483648 // (op 2007) 
s_cmp_eq_i32 s15, 0 // (op 2007) 
s_mov_b32 s18, scc // (op 2007) 

s_cmp_eq_i32 s18, 0 // (op 2007) 
s_cselect_b32 s15, s7, s8 // (op 2007) 


s_mul_i32 s8, s15, s14 // (op 2007) 

s_sub_u32 s15, s7, s8 // (op 2007) 

v_lshl_add_u32 v12, s15, 6, v15 // (op 2007) 


v_mul_lo_u32 v15, v12, s44 // (op 2007) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 2007) 
s_mul_i32 s8, s7, s18 // (op 2007) 


s_sub_u32 s7, s6, s8 // (op 2007) 


v_lshl_add_u32 v12, s7, 2, v16 // (op 2007) 


v_add_u32 v16, v12, v15 // (op 2007) 


s_mul_i32 s19, s44, s23 // (op 2009) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 2009) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 2009) least significant
s_mul_hi_u32 s6, s44, s22 // (op 2009) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2009) most significant: sum
s_add_u32 s19, s19, s6 // (op 2009) most significant: sum


s_mov_b32 s80, s46 // (op 2009) reg expression
s_mov_b32 s81, s47 // (op 2009) reg expression
s_mov_b32 s82, s18 // (op 2009) reg expression
s_mov_b32 s83, 131072 // (op 2009) call()

v_lshrrev_b32 v12, 2, v1 // (op 2026) 
s_mul_i32 s15, s16, s50 // (op 2026) 
s_add_u32 s15, s15, s0 // (op 2026) 
v_lshlrev_b32 v15, 1, v12 // (op 2026) 

s_add_u32 s6, s15, s9 // (op 2026) 

v_and_b32 v12, -4, v1 // (op 2026) 
s_mul_hi_u32 s15, s6, s13 // (op 2026) 
v_sub_u32 v17, v1, v12 // (op 2026) 

s_sub_u32 s7, s6, s15 // (op 2026) 
s_lshr_b32 s8, s7, 1 // (op 2026) 

s_add_u32 s7, s8, s15 // (op 2026) 


s_lshr_b32 s8, s7, s11 // (op 2026) 

s_and_b32 s7, s11, 2147483648 // (op 2026) 
s_cmp_eq_i32 s7, 0 // (op 2026) 
s_mov_b32 s15, scc // (op 2026) 

s_cmp_eq_i32 s15, 0 // (op 2026) 
s_cselect_b32 s7, s6, s8 // (op 2026) 


s_mul_hi_u32 s8, s7, s12 // (op 2026) 
s_sub_u32 s15, s7, s8 // (op 2026) 
s_lshr_b32 s18, s15, 1 // (op 2026) 

s_add_u32 s15, s18, s8 // (op 2026) 


s_lshr_b32 s8, s15, s10 // (op 2026) 

s_and_b32 s15, s10, 2147483648 // (op 2026) 
s_cmp_eq_i32 s15, 0 // (op 2026) 
s_mov_b32 s18, scc // (op 2026) 

s_cmp_eq_i32 s18, 0 // (op 2026) 
s_cselect_b32 s15, s7, s8 // (op 2026) 


s_mul_i32 s8, s15, s14 // (op 2026) 

s_sub_u32 s15, s7, s8 // (op 2026) 

v_lshl_add_u32 v12, s15, 6, v15 // (op 2026) 


v_mul_lo_u32 v15, v12, s44 // (op 2026) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 2026) 
s_mul_i32 s8, s7, s18 // (op 2026) 


s_sub_u32 s7, s6, s8 // (op 2026) 


v_lshl_add_u32 v12, s7, 2, v17 // (op 2026) 


v_add_u32 v17, v12, v15 // (op 2026) 


s_mul_i32 s19, s44, s23 // (op 2028) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 2028) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 2028) least significant
s_mul_hi_u32 s6, s44, s22 // (op 2028) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2028) most significant: sum
s_add_u32 s19, s19, s6 // (op 2028) most significant: sum


s_mov_b32 s80, s46 // (op 2028) reg expression
s_mov_b32 s81, s47 // (op 2028) reg expression
s_mov_b32 s82, s18 // (op 2028) reg expression
s_mov_b32 s83, 131072 // (op 2028) call()




buffer_load_dwordx4 v[18:21], v0, s[68:71], 0 offen // (op 1321) Load value
buffer_load_dwordx4 v[22:25], v0, s[68:71], 0 offen offset:64 // (op 1321) Load value


v_mov_b32 v12, v10 // (op 1327) reg expression

buffer_load_dwordx4 v[26:29], v12, s[72:75], 0 offen // (op 1327) Load value




buffer_load_ubyte v12, v13, s[76:79], 0 offen offset: 0 // (op 1333) Load value


s_lshl_b64 s[18:19], s[36:37], 5 // (op 1339) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 1339) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 1339) 

v_add_u32 v15, v10, s18 // (op 1339) 


buffer_load_dwordx4 v[30:33], v15, s[72:75], 0 offen // (op 1339) Load value


v_mov_b64 v[34:35], s[44:45] // (op 1344) convert

v_mov_b32 v15, v16 // (op 1344) 
buffer_load_ubyte v36, v15, s[80:83], 0 offen offset: 0 // (op 1344) Load value
v_add_u32 v15, v15, v34 // (op 1344) 
buffer_load_ubyte v37, v15, s[80:83], 0 offen offset: 0 // (op 1344) Load value





buffer_load_dwordx4 v[38:41], v0, s[68:71], 0 offen offset:128 // (op 1348) Load value
buffer_load_dwordx4 v[42:45], v0, s[68:71], 0 offen offset:192 // (op 1348) Load value



v_add_u32 v0, 64, v10 // (op 1354) 

buffer_load_dwordx4 v[46:49], v0, s[72:75], 0 offen // (op 1354) Load value




buffer_load_ubyte v0, v13, s[76:79], 0 offen offset:4 // (op 1360) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 1366) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 1366) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 1366) 

s_add_u32 s15, s18, 64 // (op 1366) 

v_add_u32 v13, v10, s15 // (op 1366) 


buffer_load_dwordx4 v[50:53], v13, s[72:75], 0 offen // (op 1366) Load value



v_mov_b64 v[34:35], s[44:45] // (op 1371) convert

v_mov_b32 v10, v16 // (op 1371) 
buffer_load_ubyte v54, v10, s[80:83], 0 offen offset:4 // (op 1371) Load value
v_add_u32 v10, v10, v34 // (op 1371) 
buffer_load_ubyte v55, v10, s[80:83], 0 offen offset:4 // (op 1371) Load value



v_and_b32 v10, -4, v1 // (op 2261) 
v_sub_u32 v13, v1, v10 // (op 2261) 

v_lshrrev_b32 v10, 2, v1 // (op 2261) 
v_lshl_add_u32 v15, v10, 3, v13 // (op 2261) 


s_barrier  // (op 2465) 



s_waitcnt vmcnt(8)
ds_write_b8 v15, v36 // (op 1373) Store local data 
s_waitcnt vmcnt(7)
ds_write_b8 v15, v37 offset:4 // (op 1373) Store local data 

v_and_b32 v10, -4, v1 // (op 2232) 
v_sub_u32 v13, v1, v10 // (op 2232) 

v_lshrrev_b32 v10, 2, v1 // (op 2232) 
v_lshl_add_u32 v15, v10, 3, v13 // (op 2232) 


v_and_b32 v10, -4, v1 // (op 2332) 
v_sub_u32 v13, v1, v10 // (op 2332) 

v_lshrrev_b32 v10, 2, v1 // (op 2332) 
v_lshl_add_u32 v16, v10, 3, v13 // (op 2332) 


s_mov_b32 s1, 0 // (op 511) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 510) 
s_mov_b32 s15, s18 // (op 510) convert

s_and_b32 s6, s15, 4294967294 // (op 510) 

s_cmp_lt_u32 s1, s6 // (op 510) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_510 // (op 510) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_510 if false)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoop_510:
s_add_u32 s1, s1, 2 // (op 512) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2462) 



ds_read_u8 v10, v3 // (op 160) Load local data 



ds_read_u8 v13, v3 offset:128 // (op 536) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:25], v[26:29], a[0:3], v12, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 161) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:25], v[30:33], a[4:7], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 537) 



v_mov_b32 v10, v5 // (op 163) 

buffer_load_dwordx4 v[18:21], v10, s[68:71], 0 offen offset:256 // (op 163) Load value
buffer_load_dwordx4 v[22:25], v10, s[68:71], 0 offen offset:320 // (op 163) Load value


v_mov_b32 v13, v11 // (op 165) 
v_add_u32 v34, 128, v13 // (op 165) 

buffer_load_dwordx4 v[26:29], v34, s[72:75], 0 offen // (op 165) Load value



v_mov_b32 v34, v14 // (op 167) 

buffer_load_ubyte v12, v34, s[76:79], 0 offen offset:8 // (op 167) Load value


s_lshl_b64 s[18:19], s[36:37], 5 // (op 559) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 559) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 559) 

s_add_u32 s15, s18, 128 // (op 559) 

v_add_u32 v35, v13, s15 // (op 559) 


buffer_load_dwordx4 v[30:33], v35, s[72:75], 0 offen // (op 559) Load value


v_mov_b64 v[56:57], s[44:45] // (op 157) convert

v_mov_b32 v35, v17 // (op 157) 
v_mov_b32 v58, v35 // (op 157) 
buffer_load_ubyte v36, v58, s[80:83], 0 offen offset:8 // (op 157) Load value
v_add_u32 v58, v58, v56 // (op 157) 
buffer_load_ubyte v37, v58, s[80:83], 0 offen offset:8 // (op 157) Load value


s_barrier  // (op 2481) 



s_waitcnt vmcnt(8)
ds_write_b8 v16, v54 offset:256 // (op 452) Store local data 
s_waitcnt vmcnt(7)
ds_write_b8 v16, v55 offset:260 // (op 452) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2478) 



ds_read_u8 v56, v4 offset:256 // (op 454) Load local data 



ds_read_u8 v57, v4 offset:384 // (op 540) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[46:49], a[0:3], v0, v56 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 455) 

s_barrier  // (op 2576) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:45], v[50:53], a[4:7], v0, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 538) 




buffer_load_dwordx4 v[38:41], v10, s[68:71], 0 offen offset:384 // (op 457) Load value
buffer_load_dwordx4 v[42:45], v10, s[68:71], 0 offen offset:448 // (op 457) Load value



v_add_u32 v5, 256, v5 // (op 1600) 
v_add_u32 v10, 192, v13 // (op 459) 

buffer_load_dwordx4 v[46:49], v10, s[72:75], 0 offen // (op 459) Load value




buffer_load_ubyte v0, v34, s[76:79], 0 offen offset:12 // (op 461) Load value



v_add_u32 v14, 8, v14 // (op 1890) 
s_lshl_b64 s[18:19], s[36:37], 5 // (op 554) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 554) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 554) 

s_add_u32 s15, s18, 192 // (op 554) 

v_add_u32 v10, v13, s15 // (op 554) 


buffer_load_dwordx4 v[50:53], v10, s[72:75], 0 offen // (op 554) Load value



v_add_u32 v11, 128, v11 // (op 1747) 
v_mov_b64 v[56:57], s[44:45] // (op 451) convert

v_mov_b32 v10, v35 // (op 451) 
buffer_load_ubyte v54, v10, s[80:83], 0 offen offset:12 // (op 451) Load value
v_add_u32 v10, v10, v56 // (op 451) 
buffer_load_ubyte v55, v10, s[80:83], 0 offen offset:12 // (op 451) Load value



s_barrier  // (op 2468) 
v_add_u32 v17, 8, v17 // (op 2018) 



s_waitcnt vmcnt(8)
ds_write_b8 v15, v36 // (op 158) Store local data 
s_waitcnt vmcnt(7)
ds_write_b8 v15, v37 offset:4 // (op 158) Store local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 510) 
s_mov_b32 s15, s18 // (op 510) convert

s_and_b32 s6, s15, 4294967294 // (op 510) 

s_cmp_lt_u32 s1, s6 // (op 510) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoop_510 // (op 510) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoop_510 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_510:











v_lshrrev_b32 v5, 7, v1 // (op 1538) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1538) 
v_and_b32 v10, 15, v1 // (op 1538) 
s_and_b32 s15, s18, 4294967294 // (op 1538) 
v_lshl_add_u32 v11, v5, 4, v10 // (op 1538) 


s_add_u32 s6, s0, s15 // (op 1538) 

v_and_b32 v5, 48, v1 // (op 1538) 
s_mul_i32 s15, s16, s50 // (op 1538) 
s_add_u32 s15, s15, s6 // (op 1538) 

s_add_u32 s6, s15, s9 // (op 1538) 

s_mul_hi_u32 s15, s6, s13 // (op 1538) 
s_sub_u32 s7, s6, s15 // (op 1538) 
s_lshr_b32 s8, s7, 1 // (op 1538) 

s_add_u32 s7, s8, s15 // (op 1538) 


s_lshr_b32 s8, s7, s11 // (op 1538) 

s_and_b32 s7, s11, 2147483648 // (op 1538) 
s_cmp_eq_i32 s7, 0 // (op 1538) 
s_mov_b32 s15, scc // (op 1538) 

s_cmp_eq_i32 s15, 0 // (op 1538) 
s_cselect_b32 s7, s6, s8 // (op 1538) 


s_mul_hi_u32 s8, s7, s12 // (op 1538) 
s_sub_u32 s15, s7, s8 // (op 1538) 
s_lshr_b32 s28, s15, 1 // (op 1538) 

s_add_u32 s15, s28, s8 // (op 1538) 


s_lshr_b32 s8, s15, s10 // (op 1538) 

s_and_b32 s15, s10, 2147483648 // (op 1538) 
s_cmp_eq_i32 s15, 0 // (op 1538) 
s_mov_b32 s28, scc // (op 1538) 

s_cmp_eq_i32 s28, 0 // (op 1538) 
s_cselect_b32 s15, s7, s8 // (op 1538) 


s_mul_i32 s8, s7, s18 // (op 1538) 


v_lshl_add_u32 v10, s15, 4, v11 // (op 1538) 


s_sub_u32 s7, s6, s8 // (op 1538) 


v_lshl_add_u32 v11, s7, 7, v5 // (op 1538) 


v_mul_lo_u32 v5, v10, s30 // (op 1538) 
v_add_u32 v5, v5, v11 // (op 1538) 


s_mul_i32 s19, s30, s35 // (op 1540) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 1540) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 1540) least significant
s_mul_hi_u32 s6, s30, s34 // (op 1540) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1540) most significant: sum
s_add_u32 s19, s19, s6 // (op 1540) most significant: sum


s_mov_b32 s68, s32 // (op 1540) reg expression
s_mov_b32 s69, s33 // (op 1540) reg expression
s_mov_b32 s70, s18 // (op 1540) reg expression
s_mov_b32 s71, 131072 // (op 1540) call()

v_lshrrev_b32 v10, 6, v1 // (op 1678) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1678) 
v_and_b32 v11, 1, v10 // (op 1678) 

s_and_b32 s15, s18, 4294967294 // (op 1678) 
v_and_b32 v10, 15, v1 // (op 1678) 
s_add_u32 s6, s0, s15 // (op 1678) 

v_lshl_add_u32 v12, v11, 4, v10 // (op 1678) 


v_and_b32 v10, 63, v1 // (op 1678) 
s_mul_i32 s15, s16, s50 // (op 1678) 
s_add_u32 s15, s15, s6 // (op 1678) 

v_lshrrev_b32 v11, 4, v10 // (op 1678) 

s_add_u32 s6, s15, s9 // (op 1678) 

v_lshlrev_b32 v10, 5, v11 // (op 1678) 

s_mul_hi_u32 s15, s6, s13 // (op 1678) 
s_sub_u32 s7, s6, s15 // (op 1678) 
s_lshr_b32 s8, s7, 1 // (op 1678) 

s_add_u32 s7, s8, s15 // (op 1678) 


s_lshr_b32 s8, s7, s11 // (op 1678) 

s_and_b32 s7, s11, 2147483648 // (op 1678) 
s_cmp_eq_i32 s7, 0 // (op 1678) 
s_mov_b32 s15, scc // (op 1678) 

s_cmp_eq_i32 s15, 0 // (op 1678) 
s_cselect_b32 s7, s6, s8 // (op 1678) 


s_mul_hi_u32 s8, s7, s12 // (op 1678) 
s_sub_u32 s15, s7, s8 // (op 1678) 
s_lshr_b32 s28, s15, 1 // (op 1678) 

s_add_u32 s15, s28, s8 // (op 1678) 


s_lshr_b32 s8, s15, s10 // (op 1678) 

s_and_b32 s15, s10, 2147483648 // (op 1678) 
s_cmp_eq_i32 s15, 0 // (op 1678) 
s_mov_b32 s28, scc // (op 1678) 

s_cmp_eq_i32 s28, 0 // (op 1678) 
s_cselect_b32 s15, s7, s8 // (op 1678) 


s_mul_i32 s8, s15, s14 // (op 1678) 

s_sub_u32 s15, s7, s8 // (op 1678) 

s_mul_i32 s8, s7, s18 // (op 1678) 


v_lshl_add_u32 v11, s15, 6, v12 // (op 1678) 


s_sub_u32 s7, s6, s8 // (op 1678) 


v_mul_lo_u32 v13, v11, s37 // (op 1678) most significant: low of low * high
v_mul_hi_u32 v14, v11, s36 // (op 1678) most significant: high of low * low
v_mul_lo_u32 v12, v11, s36 // (op 1678) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 1678) most significant: sum


v_lshl_add_u32 v11, s7, 7, v10 // (op 1678) 


v_mov_b32 v15, 0 // (op 1678) convert
v_mov_b32 v14, v11 // (op 1678) convert

v_add_co_u32 v10, s[18:19], v14, v12 // (op 1678) least significant half
v_addc_co_u32 v11, s[18:19], v15, v13, s[18:19] // (op 1678) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 1678) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 1678) 

v_mov_b32 v10, v12 // (op 1678) convert

s_mul_i32 s19, s36, s23 // (op 1680) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 1680) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 1680) least significant
s_mul_hi_u32 s6, s36, s22 // (op 1680) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1680) most significant: sum
s_add_u32 s19, s19, s6 // (op 1680) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 1680) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 1680) 

s_mov_b32 s72, s38 // (op 1680) reg expression
s_mov_b32 s73, s39 // (op 1680) reg expression
s_mov_b32 s74, s18 // (op 1680) reg expression
s_mov_b32 s75, 131072 // (op 1680) call()

v_lshrrev_b32 v11, 7, v1 // (op 1834) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1834) 
v_and_b32 v12, 15, v1 // (op 1834) 
s_and_b32 s15, s18, 4294967294 // (op 1834) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 1834) 


s_add_u32 s6, s0, s15 // (op 1834) 

v_and_b32 v11, 63, v1 // (op 1834) 
v_lshrrev_b32 v12, 4, v11 // (op 1834) 

s_mul_i32 s15, s16, s50 // (op 1834) 
s_add_u32 s15, s15, s6 // (op 1834) 

s_add_u32 s6, s15, s9 // (op 1834) 

s_mul_hi_u32 s15, s6, s13 // (op 1834) 
s_sub_u32 s7, s6, s15 // (op 1834) 
s_lshr_b32 s8, s7, 1 // (op 1834) 

s_add_u32 s7, s8, s15 // (op 1834) 


s_lshr_b32 s8, s7, s11 // (op 1834) 

s_and_b32 s7, s11, 2147483648 // (op 1834) 
s_cmp_eq_i32 s7, 0 // (op 1834) 
s_mov_b32 s15, scc // (op 1834) 

s_cmp_eq_i32 s15, 0 // (op 1834) 
s_cselect_b32 s7, s6, s8 // (op 1834) 


s_mul_hi_u32 s8, s7, s12 // (op 1834) 
s_sub_u32 s15, s7, s8 // (op 1834) 
s_lshr_b32 s28, s15, 1 // (op 1834) 

s_add_u32 s15, s28, s8 // (op 1834) 


s_lshr_b32 s8, s15, s10 // (op 1834) 

s_and_b32 s15, s10, 2147483648 // (op 1834) 
s_cmp_eq_i32 s15, 0 // (op 1834) 
s_mov_b32 s28, scc // (op 1834) 

s_cmp_eq_i32 s28, 0 // (op 1834) 
s_cselect_b32 s15, s7, s8 // (op 1834) 


s_mul_i32 s8, s7, s18 // (op 1834) 


v_lshl_add_u32 v11, s15, 4, v13 // (op 1834) 


s_sub_u32 s7, s6, s8 // (op 1834) 


v_lshl_add_u32 v13, s7, 2, v12 // (op 1834) 


v_mul_lo_u32 v12, v11, s40 // (op 1834) 
v_add_u32 v12, v12, v13 // (op 1834) 


s_mul_i32 s19, s40, s35 // (op 1836) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 1836) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 1836) least significant
s_mul_hi_u32 s6, s40, s34 // (op 1836) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1836) most significant: sum
s_add_u32 s19, s19, s6 // (op 1836) most significant: sum


s_mov_b32 s76, s42 // (op 1836) reg expression
s_mov_b32 s77, s43 // (op 1836) reg expression
s_mov_b32 s78, s18 // (op 1836) reg expression
s_mov_b32 s79, 131072 // (op 1836) call()

v_lshrrev_b32 v11, 2, v1 // (op 1962) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1962) 
v_lshlrev_b32 v13, 1, v11 // (op 1962) 

s_and_b32 s15, s18, 4294967294 // (op 1962) 
v_and_b32 v11, -4, v1 // (op 1962) 
s_add_u32 s6, s0, s15 // (op 1962) 

v_sub_u32 v14, v1, v11 // (op 1962) 

s_mul_i32 s15, s16, s50 // (op 1962) 
s_add_u32 s15, s15, s6 // (op 1962) 

s_add_u32 s6, s15, s9 // (op 1962) 

s_mul_hi_u32 s15, s6, s13 // (op 1962) 
s_sub_u32 s7, s6, s15 // (op 1962) 
s_lshr_b32 s8, s7, 1 // (op 1962) 

s_add_u32 s7, s8, s15 // (op 1962) 


s_lshr_b32 s8, s7, s11 // (op 1962) 

s_and_b32 s7, s11, 2147483648 // (op 1962) 
s_cmp_eq_i32 s7, 0 // (op 1962) 
s_mov_b32 s15, scc // (op 1962) 

s_cmp_eq_i32 s15, 0 // (op 1962) 
s_cselect_b32 s7, s6, s8 // (op 1962) 


s_mul_hi_u32 s8, s7, s12 // (op 1962) 
s_sub_u32 s15, s7, s8 // (op 1962) 
s_lshr_b32 s28, s15, 1 // (op 1962) 

s_add_u32 s15, s28, s8 // (op 1962) 


s_lshr_b32 s8, s15, s10 // (op 1962) 

s_and_b32 s15, s10, 2147483648 // (op 1962) 
s_cmp_eq_i32 s15, 0 // (op 1962) 
s_mov_b32 s28, scc // (op 1962) 

s_cmp_eq_i32 s28, 0 // (op 1962) 
s_cselect_b32 s15, s7, s8 // (op 1962) 


s_mul_i32 s8, s15, s14 // (op 1962) 

s_sub_u32 s15, s7, s8 // (op 1962) 

v_lshl_add_u32 v11, s15, 6, v13 // (op 1962) 


s_mul_i32 s8, s7, s18 // (op 1962) 


v_mul_lo_u32 v13, v11, s44 // (op 1962) 

s_sub_u32 s7, s6, s8 // (op 1962) 


v_lshl_add_u32 v11, s7, 2, v14 // (op 1962) 


v_add_u32 v14, v11, v13 // (op 1962) 


s_mul_i32 s19, s44, s23 // (op 1964) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 1964) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 1964) least significant
s_mul_hi_u32 s6, s44, s22 // (op 1964) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1964) most significant: sum
s_add_u32 s19, s19, s6 // (op 1964) most significant: sum


s_mov_b32 s80, s46 // (op 1964) reg expression
s_mov_b32 s81, s47 // (op 1964) reg expression
s_mov_b32 s82, s18 // (op 1964) reg expression
s_mov_b32 s83, 131072 // (op 1964) call()

v_and_b32 v11, -4, v1 // (op 2245) 
v_sub_u32 v13, v1, v11 // (op 2245) 

v_lshrrev_b32 v11, 2, v1 // (op 2245) 
v_lshl_add_u32 v15, v11, 3, v13 // (op 2245) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 516) 
s_and_b32 s15, s18, 4294967294 // (op 516) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 515) 
s_mov_b32 s6, s18 // (op 515) convert

s_cmp_lt_u32 s15, s6 // (op 515) 






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_515 // (op 515) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_515 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_515:

s_add_u32 s15, s15, 1 // (op 517) 
v_mov_b64 v[16:17], s[44:45] // (op 409) convert

v_mov_b32 v0, v14 // (op 409) 
v_mov_b32 v11, v0 // (op 409) 
buffer_load_ubyte v18, v11, s[80:83], 0 offen offset: 0 // (op 409) Load value
v_add_u32 v11, v11, v16 // (op 409) 
buffer_load_ubyte v19, v11, s[80:83], 0 offen offset: 0 // (op 409) Load value



v_add_u32 v14, 4, v14 // (op 1954) 


s_barrier  // (op 2471) 
v_mov_b32 v0, v5 // (op 415) 




buffer_load_dwordx4 v[20:23], v0, s[68:71], 0 offen // (op 415) Load value
s_waitcnt vmcnt(2)
ds_write_b8 v15, v18 // (op 410) Store local data 
buffer_load_dwordx4 v[24:27], v0, s[68:71], 0 offen offset:64 // (op 415) Load value

v_add_u32 v5, 128, v5 // (op 1528) 
s_waitcnt vmcnt(2)
ds_write_b8 v15, v19 offset:4 // (op 410) Store local data 



v_mov_b32 v0, v10 // (op 417) 
v_mov_b32 v11, v0 // (op 417) reg expression

buffer_load_dwordx4 v[16:19], v11, s[72:75], 0 offen // (op 417) Load value



v_mov_b32 v11, v12 // (op 419) 

buffer_load_ubyte v13, v11, s[76:79], 0 offen offset: 0 // (op 419) Load value

v_add_u32 v12, 4, v12 // (op 1826) 


s_lshl_b64 s[18:19], s[36:37], 5 // (op 529) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 529) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 529) 

v_add_u32 v11, v0, s18 // (op 529) 


s_waitcnt lgkmcnt(0)
s_barrier  // (op 2474) 
buffer_load_dwordx4 v[28:31], v11, s[72:75], 0 offen // (op 529) Load value





ds_read_u8 v0, v3 // (op 412) Load local data 



v_add_u32 v10, 64, v10 // (op 1668) 
ds_read_u8 v11, v3 offset:128 // (op 521) Load local data 
s_waitcnt vmcnt(1) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:27], v[16:19], a[0:3], v13, v0 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 413) 


s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:27], v[28:31], a[4:7], v13, v11 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 522) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 515) 
s_mov_b32 s6, s18 // (op 515) convert

s_cmp_lt_u32 s15, s6 // (op 515) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_515 // (op 515) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_515 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_515:






s_barrier  // (op 2562) 
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 189) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_189 // (op 189) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_189
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


v_accvgpr_read v10, a4 // (op 667) 
v_accvgpr_read v11, a5 // (op 667) 
v_accvgpr_read v12, a6 // (op 667) 
v_accvgpr_read v13, a7 // (op 667) 
v_mul_f32 v14, v7, v10 // (op 667) 
v_mul_f32 v15, v7, v11 // (op 667) 
v_mul_f32 v16, v7, v12 // (op 667) 
v_mul_f32 v17, v7, v13 // (op 667) 



s_lshl_b32 s15, s24, 7 // (op 669) 
v_add_u32 v0, v9, s15 // (op 669) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 669) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_189 // (op 189) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_189
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_189:


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




s_lshl_b32 s15, s4, 7 // (op 691) 
v_add_u32 v0, v8, s15 // (op 691) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 691) Load value

v_accvgpr_read v14, a4 // (op 688) 
v_accvgpr_read v15, a5 // (op 688) 
v_accvgpr_read v16, a6 // (op 688) 
v_accvgpr_read v17, a7 // (op 688) 
v_mul_f32 v18, v7, v14 // (op 688) 
v_mul_f32 v19, v7, v15 // (op 688) 
v_mul_f32 v20, v7, v16 // (op 688) 
v_mul_f32 v21, v7, v17 // (op 688) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v6, v10 // (op 692) 
v_mul_f32 v15, v6, v11 // (op 692) 
v_mul_f32 v16, v6, v12 // (op 692) 
v_mul_f32 v17, v6, v13 // (op 692) 

v_add_f32 v10, v14, v18 // (op 689) 
v_add_f32 v11, v15, v19 // (op 689) 
v_add_f32 v12, v16, v20 // (op 689) 
v_add_f32 v13, v17, v21 // (op 689) 




s_lshl_b32 s15, s24, 7 // (op 690) 
v_add_u32 v0, v9, s15 // (op 690) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 690) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_189:
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
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_DPStreamTileLoop_396 // (op 396) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopTop_DPStreamTileLoop_396 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_ForLoopBottom_DPStreamTileLoop_396:





































s_endpgm  // End of RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2
.LRR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_end:
.size RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2, .LRR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2_end-RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2
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
    DataType: F8
    DataTypeA: F8
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
  - .name: RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2
    .symbol: RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x128_UR_2.kd
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

