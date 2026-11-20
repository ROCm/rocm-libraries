// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2,@function
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 14558) 
v_and_b32 v3, 15, v1 // (op 14558) 
v_lshl_add_u32 v4, v0, 6, v3 // (op 14558) 


v_and_b32 v0, 48, v1 // (op 14558) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 14558) 


v_lshrrev_b32 v0, 6, v1 // (op 14633) 
v_and_b32 v4, 1, v0 // (op 14633) 

v_and_b32 v0, 15, v1 // (op 14633) 
v_lshl_add_u32 v5, v4, 6, v0 // (op 14633) 


v_and_b32 v0, 63, v1 // (op 14633) 
v_lshrrev_b32 v4, 4, v0 // (op 14633) 

v_lshlrev_b32 v0, 5, v4 // (op 14633) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 14633) 


v_lshlrev_b32 v0, 2, v4 // (op 14633) 

v_lshrrev_b32 v4, 3, v0 // (op 14633) 

v_lshrrev_b32 v0, 7, v1 // (op 14760) 
v_and_b32 v5, 15, v1 // (op 14760) 
v_lshl_add_u32 v6, v0, 6, v5 // (op 14760) 


v_and_b32 v0, 48, v1 // (op 14760) 
v_lshl_add_u32 v5, v6, 7, v0 // (op 14760) 


v_lshrrev_b32 v0, 6, v1 // (op 14835) 
v_and_b32 v6, 1, v0 // (op 14835) 

v_and_b32 v0, 15, v1 // (op 14835) 
v_lshl_add_u32 v7, v6, 6, v0 // (op 14835) 


v_and_b32 v0, 63, v1 // (op 14835) 
v_lshrrev_b32 v6, 4, v0 // (op 14835) 

v_lshlrev_b32 v0, 5, v6 // (op 14835) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 14835) 


v_lshlrev_b32 v0, 2, v6 // (op 14835) 

v_lshrrev_b32 v6, 3, v0 // (op 14835) 

v_lshrrev_b32 v0, 7, v1 // (op 14949) 
v_and_b32 v7, 15, v1 // (op 14949) 
v_lshl_add_u32 v8, v0, 6, v7 // (op 14949) 


v_and_b32 v0, 48, v1 // (op 14949) 
v_lshl_add_u32 v7, v8, 7, v0 // (op 14949) 


v_lshrrev_b32 v0, 6, v1 // (op 14993) 
v_and_b32 v8, 1, v0 // (op 14993) 

v_and_b32 v0, 15, v1 // (op 14993) 
v_lshl_add_u32 v9, v8, 6, v0 // (op 14993) 


v_and_b32 v0, 63, v1 // (op 14993) 
v_lshrrev_b32 v8, 4, v0 // (op 14993) 

v_lshlrev_b32 v0, 5, v8 // (op 14993) 

v_lshl_add_u32 v8, v9, 7, v0 // (op 14993) 


v_lshlrev_b32 v0, 2, v8 // (op 14993) 

v_lshrrev_b32 v8, 3, v0 // (op 14993) 

v_lshrrev_b32 v0, 7, v1 // (op 15037) 
v_and_b32 v9, 15, v1 // (op 15037) 
v_lshl_add_u32 v10, v0, 6, v9 // (op 15037) 


v_and_b32 v0, 48, v1 // (op 15037) 
v_lshl_add_u32 v9, v10, 7, v0 // (op 15037) 


v_lshrrev_b32 v0, 6, v1 // (op 15081) 
v_and_b32 v10, 1, v0 // (op 15081) 

v_and_b32 v0, 15, v1 // (op 15081) 
v_lshl_add_u32 v11, v10, 6, v0 // (op 15081) 


v_and_b32 v0, 63, v1 // (op 15081) 
v_lshrrev_b32 v10, 4, v0 // (op 15081) 

v_lshlrev_b32 v0, 5, v10 // (op 15081) 

v_lshl_add_u32 v10, v11, 7, v0 // (op 15081) 


v_lshlrev_b32 v0, 2, v10 // (op 15081) 

v_lshrrev_b32 v10, 3, v0 // (op 15081) 

v_mov_b32 v0, s6 // (op 2) Move value
v_mov_b32 v11, s7 // (op 35) Move value


s_mov_b32 s0, 0 // (op 364) call()
s_mov_b32 s1, 0 // (op 365) call()
s_mov_b32 s6, 0 // (op 366) call()
s_mov_b32 s7, 0 // (op 367) call()
s_mov_b32 s18, 0 // (op 368) call()
s_mov_b32 s19, 0 // (op 369) call()
s_mov_b32 s0, 0 // (op 264) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 263) 
s_cmp_gt_u32 s9, s52 // (op 263) 
s_mov_b32 s53, scc // (op 263) 
s_sub_u32 s54, s9, s52 // (op 263) 

s_cmp_eq_i32 s53, 0 // (op 263) 
s_cselect_b32 s52, 0, s54 // (op 263) 


s_cmp_lt_u32 s8, s52 // (op 263) 
s_mov_b32 s53, scc // (op 263) 
s_cmp_eq_i32 s53, 0 // (op 263) 
s_cselect_b32 s54, s52, s8 // (op 263) 


s_cmp_lt_u32 s0, s54 // (op 263) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_263 // (op 263) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_263 if false)
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_SKStreamTileLoop_263:
v_lshrrev_b32 v12, 7, v1 // (op 14469) 
s_mul_i32 s52, s16, s8 // (op 14469) 
s_add_u32 s52, s52, s0 // (op 14469) 
v_and_b32 v13, 63, v1 // (op 14469) 
s_mul_hi_u32 s53, s52, s13 // (op 14469) 
v_lshrrev_b32 v14, 4, v13 // (op 14469) 
s_sub_u32 s54, s52, s53 // (op 14469) 
v_lshlrev_b32 v15, 2, v14 // (op 14469) 

v_lshl_add_u32 v14, v12, 6, v15 // (op 14469) 


s_lshr_b32 s55, s54, 1 // (op 14469) 

s_add_u32 s54, s55, s53 // (op 14469) 


v_lshrrev_b32 v12, 2, v13 // (op 14469) 

s_lshr_b32 s53, s54, s11 // (op 14469) 

v_and_b32 v13, 3, v12 // (op 14469) 

s_and_b32 s54, s11, 2147483648 // (op 14469) 
v_and_b32 v12, 3, v1 // (op 14469) 
s_cmp_eq_i32 s54, 0 // (op 14469) 
s_mov_b32 s55, scc // (op 14469) 

v_lshl_add_u32 v15, v13, 2, v12 // (op 14469) 


s_cmp_eq_i32 s55, 0 // (op 14469) 
s_cselect_b32 s54, s52, s53 // (op 14469) 



v_lshrrev_b32 v12, 6, v1 // (op 14469) 
s_mul_hi_u32 s52, s54, s12 // (op 14469) 
v_and_b32 v13, 1, v12 // (op 14469) 

s_sub_u32 s53, s54, s52 // (op 14469) 
v_lshl_add_u32 v12, v13, 6, v15 // (op 14469) 


s_lshr_b32 s55, s53, 1 // (op 14469) 

s_add_u32 s53, s55, s52 // (op 14469) 


s_lshr_b32 s52, s53, s10 // (op 14469) 

s_and_b32 s53, s10, 2147483648 // (op 14469) 
s_cmp_eq_i32 s53, 0 // (op 14469) 
s_mov_b32 s55, scc // (op 14469) 

s_cmp_eq_i32 s55, 0 // (op 14469) 
s_cselect_b32 s53, s54, s52 // (op 14469) 


s_mul_i32 s52, s53, s14 // (op 14469) 
v_lshl_add_u32 v13, s53, 7, v14 // (op 14469) 


s_sub_u32 s53, s54, s52 // (op 14469) 


v_lshl_add_u32 v14, s53, 7, v12 // (op 14469) 


v_mul_lo_u32 v12, v14, s4 // (op 14469) 

v_add_lshl_u32 v14, v13, v12, 1 // (op 14469) 


s_mul_i32 s53, s4, s23 // (op 14471) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 14471) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 14471) least significant
s_mul_hi_u32 s55, s4, s22 // (op 14471) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 14471) most significant: sum
s_add_u32 s53, s53, s55 // (op 14471) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 14471) 

s_mov_b32 s56, s20 // (op 14471) reg expression
s_mov_b32 s57, s21 // (op 14471) reg expression
s_mov_b32 s58, s54 // (op 14471) reg expression
s_mov_b32 s59, 131072 // (op 14471) call()

v_lshrrev_b32 v12, 7, v1 // (op 14575) 
v_and_b32 v13, 15, v1 // (op 14575) 
v_lshl_add_u32 v15, v12, 6, v13 // (op 14575) 


v_and_b32 v12, 48, v1 // (op 14575) 
v_lshl_add_u32 v13, v15, 7, v12 // (op 14575) 


v_lshrrev_b32 v12, 6, v1 // (op 14650) 
v_and_b32 v15, 1, v12 // (op 14650) 

v_and_b32 v12, 15, v1 // (op 14650) 
v_lshl_add_u32 v16, v15, 6, v12 // (op 14650) 


v_and_b32 v12, 63, v1 // (op 14650) 
v_lshrrev_b32 v15, 4, v12 // (op 14650) 

v_lshlrev_b32 v12, 5, v15 // (op 14650) 

v_lshl_add_u32 v15, v16, 7, v12 // (op 14650) 


v_lshlrev_b32 v12, 2, v15 // (op 14650) 

v_lshrrev_b32 v15, 3, v12 // (op 14650) 

v_lshrrev_b32 v12, 7, v1 // (op 14673) 
s_mul_i32 s52, s16, s8 // (op 14673) 
s_add_u32 s52, s52, s0 // (op 14673) 
v_and_b32 v16, 63, v1 // (op 14673) 
s_mul_hi_u32 s53, s52, s13 // (op 14673) 
v_lshrrev_b32 v17, 4, v16 // (op 14673) 
s_sub_u32 s54, s52, s53 // (op 14673) 
v_lshlrev_b32 v18, 2, v17 // (op 14673) 

s_lshr_b32 s55, s54, 1 // (op 14673) 

v_lshl_add_u32 v17, v12, 6, v18 // (op 14673) 


s_add_u32 s54, s55, s53 // (op 14673) 


v_lshrrev_b32 v12, 2, v16 // (op 14673) 

s_lshr_b32 s53, s54, s11 // (op 14673) 

v_and_b32 v16, 3, v12 // (op 14673) 

s_and_b32 s54, s11, 2147483648 // (op 14673) 
v_and_b32 v12, 3, v1 // (op 14673) 
s_cmp_eq_i32 s54, 0 // (op 14673) 
s_mov_b32 s55, scc // (op 14673) 

v_lshl_add_u32 v18, v16, 2, v12 // (op 14673) 


s_cmp_eq_i32 s55, 0 // (op 14673) 
s_cselect_b32 s54, s52, s53 // (op 14673) 



v_lshrrev_b32 v12, 6, v1 // (op 14673) 
s_mul_hi_u32 s55, s54, s12 // (op 14673) 
v_and_b32 v16, 1, v12 // (op 14673) 

s_sub_u32 s52, s54, s55 // (op 14673) 
v_lshl_add_u32 v12, v16, 6, v18 // (op 14673) 


s_lshr_b32 s53, s52, 1 // (op 14673) 

s_add_u32 s52, s53, s55 // (op 14673) 


s_lshr_b32 s53, s52, s10 // (op 14673) 

s_and_b32 s52, s10, 2147483648 // (op 14673) 
s_cmp_eq_i32 s52, 0 // (op 14673) 
s_mov_b32 s55, scc // (op 14673) 

s_cmp_eq_i32 s55, 0 // (op 14673) 
s_cselect_b32 s52, s54, s53 // (op 14673) 


s_mul_i32 s53, s52, s14 // (op 14673) 
v_lshl_add_u32 v16, s52, 7, v17 // (op 14673) 


s_sub_u32 s52, s54, s53 // (op 14673) 


v_lshl_add_u32 v17, s52, 7, v12 // (op 14673) 


v_mul_lo_u32 v12, v17, s24 // (op 14673) 

v_add_lshl_u32 v17, v16, v12, 1 // (op 14673) 


s_mul_i32 s53, s24, s23 // (op 14675) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 14675) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 14675) least significant
s_mul_hi_u32 s55, s24, s22 // (op 14675) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 14675) most significant: sum
s_add_u32 s53, s53, s55 // (op 14675) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 14675) 

s_mov_b32 s60, s26 // (op 14675) reg expression
s_mov_b32 s61, s27 // (op 14675) reg expression
s_mov_b32 s62, s54 // (op 14675) reg expression
s_mov_b32 s63, 131072 // (op 14675) call()

v_lshrrev_b32 v12, 5, v1 // (op 14875) 
v_lshlrev_b32 v16, 2, v12 // (op 14875) 

v_lshl_add_u32 v12, s16, 7, v16 // (op 14875) 

v_and_b32 v16, -32, v1 // (op 14875) 
v_sub_u32 v18, v1, v16 // (op 14875) 

v_lshl_add_u32 v16, v12, 7, v18 // (op 14875) 


v_lshlrev_b32 v12, 2, v16 // (op 14875) 

s_lshl_b32 s52, s15, 16 // (op 14877) 
s_mov_b32 s64, s28 // (op 14877) reg expression
s_mov_b32 s65, s29 // (op 14877) reg expression
s_mov_b32 s66, s52 // (op 14877) reg expression
s_mov_b32 s67, 131072 // (op 14877) call()

s_mul_i32 s6, s8, s16 // (op 268) 
s_add_u32 s6, s6, s0 // (op 268) 
s_mul_hi_u32 s52, s6, s13 // (op 269) 
s_sub_u32 s53, s6, s52 // (op 269) 
s_lshr_b32 s54, s53, 1 // (op 269) 

s_add_u32 s53, s54, s52 // (op 269) 


s_lshr_b32 s52, s53, s11 // (op 269) 

s_and_b32 s53, s11, 2147483648 // (op 269) 
s_cmp_eq_i32 s53, 0 // (op 269) 
s_mov_b32 s54, scc // (op 269) 

s_cmp_eq_i32 s54, 0 // (op 269) 
s_cselect_b32 s53, s6, s52 // (op 269) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 269) 
s_mul_i32 s52, s53, s54 // (op 269) 


s_sub_u32 s18, s6, s52 // (op 269) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 270) 
s_cmp_gt_u32 s52, s18 // (op 270) 
s_mov_b32 s54, scc // (op 270) 
s_sub_u32 s55, s52, s18 // (op 270) 

s_cmp_eq_i32 s54, 0 // (op 270) 
s_cselect_b32 s52, 0, s55 // (op 270) 


s_cmp_gt_u32 s8, s0 // (op 270) 
s_mov_b32 s53, scc // (op 270) 
s_sub_u32 s54, s8, s0 // (op 270) 
s_cmp_eq_i32 s53, 0 // (op 270) 
s_cselect_b32 s55, 0, s54 // (op 270) 


s_cmp_lt_u32 s52, s55 // (op 270) 
s_mov_b32 s53, scc // (op 270) 
s_cmp_eq_i32 s53, 0 // (op 270) 
s_cselect_b32 s54, s55, s52 // (op 270) 



s_cmp_gt_u32 s9, s6 // (op 270) 
s_mov_b32 s55, scc // (op 270) 
s_sub_u32 s52, s9, s6 // (op 270) 
s_cmp_eq_i32 s55, 0 // (op 270) 
s_cselect_b32 s53, 0, s52 // (op 270) 


s_cmp_lt_u32 s54, s53 // (op 270) 
s_mov_b32 s52, scc // (op 270) 
s_cmp_eq_i32 s52, 0 // (op 270) 
s_cselect_b32 s7, s53, s54 // (op 270) 



s_add_u32 s52, s18, s7 // (op 271) 
s_sub_u32 s19, s52, 1 // (op 271) 

v_accvgpr_write a0, 0.00000 // (op 57) call()
v_accvgpr_write a1, 0.00000 // (op 57) call()
v_accvgpr_write a2, 0.00000 // (op 57) call()
v_accvgpr_write a3, 0.00000 // (op 57) call()
v_accvgpr_write a4, 0.00000 // (op 4091) call()
v_accvgpr_write a5, 0.00000 // (op 4091) call()
v_accvgpr_write a6, 0.00000 // (op 4091) call()
v_accvgpr_write a7, 0.00000 // (op 4091) call()
v_accvgpr_write a8, 0.00000 // (op 4227) call()
v_accvgpr_write a9, 0.00000 // (op 4227) call()
v_accvgpr_write a10, 0.00000 // (op 4227) call()
v_accvgpr_write a11, 0.00000 // (op 4227) call()
v_accvgpr_write a12, 0.00000 // (op 4363) call()
v_accvgpr_write a13, 0.00000 // (op 4363) call()
v_accvgpr_write a14, 0.00000 // (op 4363) call()
v_accvgpr_write a15, 0.00000 // (op 4363) call()
v_accvgpr_write a16, 0.00000 // (op 4587) call()
v_accvgpr_write a17, 0.00000 // (op 4587) call()
v_accvgpr_write a18, 0.00000 // (op 4587) call()
v_accvgpr_write a19, 0.00000 // (op 4587) call()
v_accvgpr_write a20, 0.00000 // (op 4848) call()
v_accvgpr_write a21, 0.00000 // (op 4848) call()
v_accvgpr_write a22, 0.00000 // (op 4848) call()
v_accvgpr_write a23, 0.00000 // (op 4848) call()
v_accvgpr_write a24, 0.00000 // (op 4849) call()
v_accvgpr_write a25, 0.00000 // (op 4849) call()
v_accvgpr_write a26, 0.00000 // (op 4849) call()
v_accvgpr_write a27, 0.00000 // (op 4849) call()
v_accvgpr_write a28, 0.00000 // (op 4850) call()
v_accvgpr_write a29, 0.00000 // (op 4850) call()
v_accvgpr_write a30, 0.00000 // (op 4850) call()
v_accvgpr_write a31, 0.00000 // (op 4850) call()
v_accvgpr_write a32, 0.00000 // (op 5197) call()
v_accvgpr_write a33, 0.00000 // (op 5197) call()
v_accvgpr_write a34, 0.00000 // (op 5197) call()
v_accvgpr_write a35, 0.00000 // (op 5197) call()
v_accvgpr_write a36, 0.00000 // (op 5458) call()
v_accvgpr_write a37, 0.00000 // (op 5458) call()
v_accvgpr_write a38, 0.00000 // (op 5458) call()
v_accvgpr_write a39, 0.00000 // (op 5458) call()
v_accvgpr_write a40, 0.00000 // (op 5459) call()
v_accvgpr_write a41, 0.00000 // (op 5459) call()
v_accvgpr_write a42, 0.00000 // (op 5459) call()
v_accvgpr_write a43, 0.00000 // (op 5459) call()
v_accvgpr_write a44, 0.00000 // (op 5460) call()
v_accvgpr_write a45, 0.00000 // (op 5460) call()
v_accvgpr_write a46, 0.00000 // (op 5460) call()
v_accvgpr_write a47, 0.00000 // (op 5460) call()
v_accvgpr_write a48, 0.00000 // (op 5807) call()
v_accvgpr_write a49, 0.00000 // (op 5807) call()
v_accvgpr_write a50, 0.00000 // (op 5807) call()
v_accvgpr_write a51, 0.00000 // (op 5807) call()
v_accvgpr_write a52, 0.00000 // (op 6068) call()
v_accvgpr_write a53, 0.00000 // (op 6068) call()
v_accvgpr_write a54, 0.00000 // (op 6068) call()
v_accvgpr_write a55, 0.00000 // (op 6068) call()
v_accvgpr_write a56, 0.00000 // (op 6069) call()
v_accvgpr_write a57, 0.00000 // (op 6069) call()
v_accvgpr_write a58, 0.00000 // (op 6069) call()
v_accvgpr_write a59, 0.00000 // (op 6069) call()
v_accvgpr_write a60, 0.00000 // (op 6070) call()
v_accvgpr_write a61, 0.00000 // (op 6070) call()
v_accvgpr_write a62, 0.00000 // (op 6070) call()
v_accvgpr_write a63, 0.00000 // (op 6070) call()
v_lshrrev_b32 v16, 3, v1 // (op 14292) 
s_mul_i32 s52, s16, s8 // (op 14292) 
s_add_u32 s52, s52, s0 // (op 14292) 
v_and_b32 v18, -8, v1 // (op 14292) 
s_mul_hi_u32 s53, s52, s13 // (op 14292) 
v_sub_u32 v19, v1, v18 // (op 14292) 

s_sub_u32 s54, s52, s53 // (op 14292) 
v_lshlrev_b32 v18, 4, v19 // (op 14292) 

s_lshr_b32 s55, s54, 1 // (op 14292) 

s_add_u32 s54, s55, s53 // (op 14292) 


s_lshr_b32 s53, s54, s11 // (op 14292) 

s_and_b32 s54, s11, 2147483648 // (op 14292) 
s_cmp_eq_i32 s54, 0 // (op 14292) 
s_mov_b32 s55, scc // (op 14292) 

s_cmp_eq_i32 s55, 0 // (op 14292) 
s_cselect_b32 s54, s52, s53 // (op 14292) 


s_mul_hi_u32 s53, s54, s12 // (op 14292) 
s_sub_u32 s55, s54, s53 // (op 14292) 
s_lshr_b32 s68, s55, 1 // (op 14292) 

s_add_u32 s55, s68, s53 // (op 14292) 


s_lshr_b32 s53, s55, s10 // (op 14292) 

s_and_b32 s55, s10, 2147483648 // (op 14292) 
s_cmp_eq_i32 s55, 0 // (op 14292) 
s_mov_b32 s68, scc // (op 14292) 

s_cmp_eq_i32 s68, 0 // (op 14292) 
s_cselect_b32 s55, s54, s53 // (op 14292) 


v_lshl_add_u32 v19, s55, 7, v16 // (op 14292) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 14292) 
s_mul_i32 s53, s54, s68 // (op 14292) 


s_sub_u32 s54, s52, s53 // (op 14292) 


v_lshl_add_u32 v16, s54, 7, v18 // (op 14292) 


v_mov_b32 v21, 0 // (op 14292) convert
v_mov_b32 v20, v16 // (op 14292) convert

v_mul_lo_u32 v23, v19, s31 // (op 14292) most significant: low of low * high
v_mul_hi_u32 v16, v19, s30 // (op 14292) most significant: high of low * low
v_mul_lo_u32 v22, v19, s30 // (op 14292) least significant: low of low * low
v_add_u32 v23, v23, v16 // (op 14292) most significant: sum

v_add_co_u32 v22, s[52:53], v22, v20 // (op 14292) least significant half
v_addc_co_u32 v23, s[52:53], v23, v21, s[52:53] // (op 14292) most significant half



s_mul_i32 s53, s30, s35 // (op 14294) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 14294) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 14294) least significant
s_mul_hi_u32 s55, s30, s34 // (op 14294) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 14294) most significant: sum
s_add_u32 s53, s53, s55 // (op 14294) most significant: sum


s_mov_b32 s68, s32 // (op 14294) reg expression
s_mov_b32 s69, s33 // (op 14294) reg expression
s_mov_b32 s70, s52 // (op 14294) reg expression
s_mov_b32 s71, 131072 // (op 14294) call()

v_lshrrev_b32 v16, 3, v1 // (op 14311) 
s_mul_i32 s52, s16, s8 // (op 14311) 
s_add_u32 s52, s52, s0 // (op 14311) 
v_and_b32 v18, -8, v1 // (op 14311) 
s_mul_hi_u32 s53, s52, s13 // (op 14311) 
v_sub_u32 v19, v1, v18 // (op 14311) 

s_sub_u32 s54, s52, s53 // (op 14311) 
v_lshlrev_b32 v18, 4, v19 // (op 14311) 

s_lshr_b32 s55, s54, 1 // (op 14311) 

s_add_u32 s54, s55, s53 // (op 14311) 


s_lshr_b32 s53, s54, s11 // (op 14311) 

s_and_b32 s54, s11, 2147483648 // (op 14311) 
s_cmp_eq_i32 s54, 0 // (op 14311) 
s_mov_b32 s55, scc // (op 14311) 

s_cmp_eq_i32 s55, 0 // (op 14311) 
s_cselect_b32 s54, s52, s53 // (op 14311) 


s_mul_hi_u32 s53, s54, s12 // (op 14311) 
s_sub_u32 s55, s54, s53 // (op 14311) 
s_lshr_b32 s72, s55, 1 // (op 14311) 

s_add_u32 s55, s72, s53 // (op 14311) 


s_lshr_b32 s53, s55, s10 // (op 14311) 

s_and_b32 s55, s10, 2147483648 // (op 14311) 
s_cmp_eq_i32 s55, 0 // (op 14311) 
s_mov_b32 s72, scc // (op 14311) 

s_cmp_eq_i32 s72, 0 // (op 14311) 
s_cselect_b32 s55, s54, s53 // (op 14311) 


v_lshl_add_u32 v19, s55, 7, v16 // (op 14311) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 14311) 
s_mul_i32 s53, s54, s72 // (op 14311) 


s_sub_u32 s54, s52, s53 // (op 14311) 


v_lshl_add_u32 v16, s54, 7, v18 // (op 14311) 


v_mov_b32 v21, 0 // (op 14311) convert
v_mov_b32 v20, v16 // (op 14311) convert

v_mul_lo_u32 v25, v19, s31 // (op 14311) most significant: low of low * high
v_mul_hi_u32 v16, v19, s30 // (op 14311) most significant: high of low * low
v_mul_lo_u32 v24, v19, s30 // (op 14311) least significant: low of low * low
v_add_u32 v25, v25, v16 // (op 14311) most significant: sum

v_add_co_u32 v24, s[52:53], v24, v20 // (op 14311) least significant half
v_addc_co_u32 v25, s[52:53], v25, v21, s[52:53] // (op 14311) most significant half



s_mul_i32 s53, s30, s35 // (op 14313) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 14313) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 14313) least significant
s_mul_hi_u32 s55, s30, s34 // (op 14313) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 14313) most significant: sum
s_add_u32 s53, s53, s55 // (op 14313) most significant: sum


s_mov_b32 s68, s32 // (op 14313) reg expression
s_mov_b32 s69, s33 // (op 14313) reg expression
s_mov_b32 s70, s52 // (op 14313) reg expression
s_mov_b32 s71, 131072 // (op 14313) call()

v_lshrrev_b32 v16, 2, v1 // (op 14420) 
s_mul_i32 s52, s16, s8 // (op 14420) 
s_add_u32 s52, s52, s0 // (op 14420) 
v_and_b32 v18, -4, v1 // (op 14420) 
s_mul_hi_u32 s53, s52, s13 // (op 14420) 
v_sub_u32 v19, v1, v18 // (op 14420) 

s_sub_u32 s54, s52, s53 // (op 14420) 
v_lshlrev_b32 v18, 5, v19 // (op 14420) 

s_lshr_b32 s55, s54, 1 // (op 14420) 

s_add_u32 s54, s55, s53 // (op 14420) 


s_lshr_b32 s53, s54, s11 // (op 14420) 

s_and_b32 s54, s11, 2147483648 // (op 14420) 
s_cmp_eq_i32 s54, 0 // (op 14420) 
s_mov_b32 s55, scc // (op 14420) 

s_cmp_eq_i32 s55, 0 // (op 14420) 
s_cselect_b32 s54, s52, s53 // (op 14420) 


s_mul_hi_u32 s53, s54, s12 // (op 14420) 
s_sub_u32 s55, s54, s53 // (op 14420) 
s_lshr_b32 s72, s55, 1 // (op 14420) 

s_add_u32 s55, s72, s53 // (op 14420) 


s_lshr_b32 s53, s55, s10 // (op 14420) 

s_and_b32 s55, s10, 2147483648 // (op 14420) 
s_cmp_eq_i32 s55, 0 // (op 14420) 
s_mov_b32 s72, scc // (op 14420) 

s_cmp_eq_i32 s72, 0 // (op 14420) 
s_cselect_b32 s55, s54, s53 // (op 14420) 


s_mul_i32 s53, s55, s14 // (op 14420) 

s_sub_u32 s55, s54, s53 // (op 14420) 

v_lshl_add_u32 v19, s55, 7, v16 // (op 14420) 


v_mul_lo_u32 v21, v19, s37 // (op 14420) most significant: low of low * high
v_mul_hi_u32 v16, v19, s36 // (op 14420) most significant: high of low * low
v_mul_lo_u32 v20, v19, s36 // (op 14420) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 14420) most significant: sum


s_ashr_i64 s[72:73], s[2:3], 7 // (op 14420) 
s_mul_i32 s53, s54, s72 // (op 14420) 


s_sub_u32 s54, s52, s53 // (op 14420) 


v_lshl_add_u32 v16, s54, 7, v18 // (op 14420) 


v_mov_b32 v19, 0 // (op 14420) convert
v_mov_b32 v18, v16 // (op 14420) convert

v_add_co_u32 v26, s[52:53], v18, v20 // (op 14420) least significant half
v_addc_co_u32 v27, s[52:53], v19, v21, s[52:53] // (op 14420) most significant half

v_lshlrev_b64 v[26:27], 2, v[26:27] // (op 14420) 


v_lshrrev_b64 v[18:19], 3, v[26:27] // (op 14420) 

s_mul_i32 s53, s36, s23 // (op 14422) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 14422) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 14422) least significant
s_mul_hi_u32 s55, s36, s22 // (op 14422) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 14422) most significant: sum
s_add_u32 s53, s53, s55 // (op 14422) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 14422) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 14422) 

s_mov_b32 s72, s38 // (op 14422) reg expression
s_mov_b32 s73, s39 // (op 14422) reg expression
s_mov_b32 s74, s52 // (op 14422) reg expression
s_mov_b32 s75, 131072 // (op 14422) call()

v_lshrrev_b32 v16, 2, v1 // (op 14439) 
s_mul_i32 s52, s16, s8 // (op 14439) 
s_add_u32 s52, s52, s0 // (op 14439) 
v_and_b32 v20, -4, v1 // (op 14439) 
s_mul_hi_u32 s53, s52, s13 // (op 14439) 
v_sub_u32 v21, v1, v20 // (op 14439) 

s_sub_u32 s54, s52, s53 // (op 14439) 
v_lshlrev_b32 v20, 5, v21 // (op 14439) 

s_lshr_b32 s55, s54, 1 // (op 14439) 

s_add_u32 s54, s55, s53 // (op 14439) 


s_lshr_b32 s53, s54, s11 // (op 14439) 

s_and_b32 s54, s11, 2147483648 // (op 14439) 
s_cmp_eq_i32 s54, 0 // (op 14439) 
s_mov_b32 s55, scc // (op 14439) 

s_cmp_eq_i32 s55, 0 // (op 14439) 
s_cselect_b32 s54, s52, s53 // (op 14439) 


s_mul_hi_u32 s53, s54, s12 // (op 14439) 
s_sub_u32 s55, s54, s53 // (op 14439) 
s_lshr_b32 s76, s55, 1 // (op 14439) 

s_add_u32 s55, s76, s53 // (op 14439) 


s_lshr_b32 s53, s55, s10 // (op 14439) 

s_and_b32 s55, s10, 2147483648 // (op 14439) 
s_cmp_eq_i32 s55, 0 // (op 14439) 
s_mov_b32 s76, scc // (op 14439) 

s_cmp_eq_i32 s76, 0 // (op 14439) 
s_cselect_b32 s55, s54, s53 // (op 14439) 


s_mul_i32 s53, s55, s14 // (op 14439) 

s_sub_u32 s55, s54, s53 // (op 14439) 

v_lshl_add_u32 v21, s55, 7, v16 // (op 14439) 


v_mul_lo_u32 v27, v21, s37 // (op 14439) most significant: low of low * high
v_mul_hi_u32 v16, v21, s36 // (op 14439) most significant: high of low * low
v_mul_lo_u32 v26, v21, s36 // (op 14439) least significant: low of low * low
v_add_u32 v27, v27, v16 // (op 14439) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 14439) 
s_mul_i32 s53, s54, s76 // (op 14439) 


s_sub_u32 s54, s52, s53 // (op 14439) 


v_lshl_add_u32 v16, s54, 7, v20 // (op 14439) 


v_mov_b32 v21, 0 // (op 14439) convert
v_mov_b32 v20, v16 // (op 14439) convert

v_add_co_u32 v28, s[52:53], v20, v26 // (op 14439) least significant half
v_addc_co_u32 v29, s[52:53], v21, v27, s[52:53] // (op 14439) most significant half

v_lshlrev_b64 v[28:29], 2, v[28:29] // (op 14439) 


v_lshrrev_b64 v[20:21], 3, v[28:29] // (op 14439) 

s_mul_i32 s53, s36, s23 // (op 14441) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 14441) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 14441) least significant
s_mul_hi_u32 s55, s36, s22 // (op 14441) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 14441) most significant: sum
s_add_u32 s53, s53, s55 // (op 14441) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 14441) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 14441) 

s_mov_b32 s72, s38 // (op 14441) reg expression
s_mov_b32 s73, s39 // (op 14441) reg expression
s_mov_b32 s74, s52 // (op 14441) reg expression
s_mov_b32 s75, 131072 // (op 14441) call()

v_and_b32 v16, 63, v1 // (op 15257) 
s_mul_i32 s52, s16, s8 // (op 15257) 
s_add_u32 s52, s52, s0 // (op 15257) 
v_lshrrev_b32 v26, 4, v16 // (op 15257) 
s_mul_hi_u32 s53, s52, s13 // (op 15257) 
v_and_b32 v27, 3, v26 // (op 15257) 

s_sub_u32 s54, s52, s53 // (op 15257) 
v_and_b32 v26, 15, v1 // (op 15257) 
s_lshr_b32 s55, s54, 1 // (op 15257) 

v_lshl_add_u32 v28, v27, 4, v26 // (op 15257) 


s_add_u32 s54, s55, s53 // (op 15257) 


v_lshrrev_b32 v26, 7, v1 // (op 15257) 
s_lshr_b32 s53, s54, s11 // (op 15257) 

v_lshl_add_u32 v27, v26, 6, v28 // (op 15257) 


s_and_b32 s54, s11, 2147483648 // (op 15257) 
v_lshrrev_b32 v26, 6, v16 // (op 15257) 

s_cmp_eq_i32 s54, 0 // (op 15257) 
s_mov_b32 s55, scc // (op 15257) 

v_lshlrev_b32 v16, 2, v26 // (op 15257) 

s_cmp_eq_i32 s55, 0 // (op 15257) 
s_cselect_b32 s54, s52, s53 // (op 15257) 


s_mul_hi_u32 s53, s54, s12 // (op 15257) 
s_sub_u32 s55, s54, s53 // (op 15257) 
s_lshr_b32 s76, s55, 1 // (op 15257) 

s_add_u32 s55, s76, s53 // (op 15257) 


s_lshr_b32 s53, s55, s10 // (op 15257) 

s_and_b32 s55, s10, 2147483648 // (op 15257) 
s_cmp_eq_i32 s55, 0 // (op 15257) 
s_mov_b32 s76, scc // (op 15257) 

s_cmp_eq_i32 s76, 0 // (op 15257) 
s_cselect_b32 s55, s54, s53 // (op 15257) 


v_lshl_add_u32 v26, s55, 7, v27 // (op 15257) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 15257) 
s_mul_i32 s53, s54, s76 // (op 15257) 


s_sub_u32 s54, s52, s53 // (op 15257) 


v_lshl_add_u32 v27, s54, 2, v16 // (op 15257) 


v_mul_lo_u32 v16, v26, s40 // (op 15257) 
v_add_u32 v16, v16, v27 // (op 15257) 


s_mul_i32 s53, s40, s35 // (op 15259) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 15259) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 15259) least significant
s_mul_hi_u32 s55, s40, s34 // (op 15259) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 15259) most significant: sum
s_add_u32 s53, s53, s55 // (op 15259) most significant: sum


s_mov_b32 s76, s42 // (op 15259) reg expression
s_mov_b32 s77, s43 // (op 15259) reg expression
s_mov_b32 s78, s52 // (op 15259) reg expression
s_mov_b32 s79, 131072 // (op 15259) call()

v_and_b32 v26, 63, v1 // (op 15284) 
s_mul_i32 s52, s16, s8 // (op 15284) 
s_add_u32 s52, s52, s0 // (op 15284) 
v_lshrrev_b32 v27, 4, v26 // (op 15284) 
s_mul_hi_u32 s53, s52, s13 // (op 15284) 
v_and_b32 v28, 3, v27 // (op 15284) 

s_sub_u32 s54, s52, s53 // (op 15284) 
v_and_b32 v27, 15, v1 // (op 15284) 
s_lshr_b32 s55, s54, 1 // (op 15284) 

v_lshl_add_u32 v29, v28, 4, v27 // (op 15284) 


s_add_u32 s54, s55, s53 // (op 15284) 


v_lshrrev_b32 v27, 6, v1 // (op 15284) 
s_lshr_b32 s53, s54, s11 // (op 15284) 

v_and_b32 v28, 1, v27 // (op 15284) 

s_and_b32 s54, s11, 2147483648 // (op 15284) 
v_lshl_add_u32 v27, v28, 6, v29 // (op 15284) 


s_cmp_eq_i32 s54, 0 // (op 15284) 
s_mov_b32 s55, scc // (op 15284) 

v_lshrrev_b32 v28, 6, v26 // (op 15284) 

s_cmp_eq_i32 s55, 0 // (op 15284) 
s_cselect_b32 s54, s52, s53 // (op 15284) 


v_lshlrev_b32 v26, 2, v28 // (op 15284) 

s_mul_hi_u32 s53, s54, s12 // (op 15284) 
s_sub_u32 s55, s54, s53 // (op 15284) 
s_lshr_b32 s80, s55, 1 // (op 15284) 

s_add_u32 s55, s80, s53 // (op 15284) 


s_lshr_b32 s53, s55, s10 // (op 15284) 

s_and_b32 s55, s10, 2147483648 // (op 15284) 
s_cmp_eq_i32 s55, 0 // (op 15284) 
s_mov_b32 s80, scc // (op 15284) 

s_cmp_eq_i32 s80, 0 // (op 15284) 
s_cselect_b32 s55, s54, s53 // (op 15284) 


s_mul_i32 s53, s55, s14 // (op 15284) 

s_sub_u32 s55, s54, s53 // (op 15284) 

v_lshl_add_u32 v28, s55, 7, v27 // (op 15284) 


v_mul_lo_u32 v27, v28, s44 // (op 15284) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 15284) 
s_mul_i32 s53, s54, s80 // (op 15284) 


s_sub_u32 s54, s52, s53 // (op 15284) 


v_lshl_add_u32 v28, s54, 2, v26 // (op 15284) 


v_add_u32 v26, v28, v27 // (op 15284) 


s_mul_i32 s53, s44, s23 // (op 15286) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 15286) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 15286) least significant
s_mul_hi_u32 s55, s44, s22 // (op 15286) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 15286) most significant: sum
s_add_u32 s53, s53, s55 // (op 15286) most significant: sum


s_mov_b32 s80, s46 // (op 15286) reg expression
s_mov_b32 s81, s47 // (op 15286) reg expression
s_mov_b32 s82, s52 // (op 15286) reg expression
s_mov_b32 s83, 131072 // (op 15286) call()

v_and_b32 v27, -4, v1 // (op 14616) 
v_sub_u32 v28, v1, v27 // (op 14616) 

v_lshlrev_b32 v27, 5, v28 // (op 14616) 

v_lshrrev_b32 v28, 2, v1 // (op 14616) 
v_lshl_add_u32 v29, v28, 7, v27 // (op 14616) 


v_lshlrev_b32 v27, 2, v29 // (op 14616) 

v_lshrrev_b32 v28, 3, v27 // (op 14616) 

s_nop 0
v_readfirstlane_b32 s52, v28 // (op 14616) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 15585) 

s_lshl_b64 s[54:55], s[36:37], 6 // (op 14142) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 14142) 

v_lshrrev_b64 v[28:29], 3, s[84:85] // (op 14142) 


v_mov_b64 v[30:31], v[18:19] // (op 14142) 
s_mov_b32 m0, s52 // (op 14142) reg expression
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 14142) Load value direct to lds
v_add_co_u32 v30, s[54:55], v30, v28 // (op 14142) least significant half
v_addc_co_u32 v31, s[54:55], 0, v31, s[54:55] // (op 14142) most significant half

s_add_u32 m0, m0, 4096 // (op 14142) 
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 14142) Load value direct to lds



v_and_b32 v27, -8, v1 // (op 14541) 
v_sub_u32 v28, v1, v27 // (op 14541) 

v_lshlrev_b32 v27, 4, v28 // (op 14541) 

v_lshrrev_b32 v28, 3, v1 // (op 14541) 
v_lshl_add_u32 v29, v28, 7, v27 // (op 14541) 


s_nop 0
v_readfirstlane_b32 s52, v29 // (op 14541) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 15566) 

s_lshl_b64 s[54:55], s[30:31], 5 // (op 14147) 
v_mov_b64 v[28:29], s[54:55] // (op 14147) convert


v_mov_b64 v[30:31], v[22:23] // (op 14147) 
s_add_u32 s53, 8192, s52 // (op 14147) 
s_mov_b32 m0, s53 // (op 14147) reg expression
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 14147) Load value direct to lds
v_add_co_u32 v30, s[54:55], v30, v28 // (op 14147) least significant half
v_addc_co_u32 v31, s[54:55], 0, v31, s[54:55] // (op 14147) most significant half

s_add_u32 m0, m0, 4096 // (op 14147) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 14147) Load value direct to lds
v_add_co_u32 v30, s[54:55], v30, v28 // (op 14147) least significant half
v_addc_co_u32 v31, s[54:55], 0, v31, s[54:55] // (op 14147) most significant half

s_add_u32 m0, m0, 4096 // (op 14147) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 14147) Load value direct to lds
v_add_co_u32 v30, s[54:55], v30, v28 // (op 14147) least significant half
v_addc_co_u32 v31, s[54:55], 0, v31, s[54:55] // (op 14147) most significant half

s_add_u32 m0, m0, 4096 // (op 14147) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 14147) Load value direct to lds




v_and_b32 v27, -4, v1 // (op 15064) 
v_sub_u32 v28, v1, v27 // (op 15064) 

v_lshlrev_b32 v27, 5, v28 // (op 15064) 

v_lshrrev_b32 v28, 2, v1 // (op 15064) 
v_lshl_add_u32 v29, v28, 7, v27 // (op 15064) 


v_lshlrev_b32 v27, 2, v29 // (op 15064) 

v_lshrrev_b32 v28, 3, v27 // (op 15064) 

s_nop 0
v_readfirstlane_b32 s52, v28 // (op 15064) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 15672) 

s_lshl_b64 s[54:55], s[36:37], 6 // (op 14152) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 14152) 

v_lshrrev_b64 v[28:29], 3, s[84:85] // (op 14152) 


v_mov_b64 v[30:31], v[18:19] // (op 14152) 
s_add_u32 s53, 24576, s52 // (op 14152) 
s_mov_b32 m0, s53 // (op 14152) reg expression
buffer_load_dwordx4 v30, s[72:75], 64 offen offset: 0 lds // (op 14152) Load value direct to lds
v_add_co_u32 v30, s[54:55], v30, v28 // (op 14152) least significant half
v_addc_co_u32 v31, s[54:55], 0, v31, s[54:55] // (op 14152) most significant half

s_add_u32 m0, m0, 4096 // (op 14152) 
buffer_load_dwordx4 v30, s[72:75], 64 offen offset: 0 lds // (op 14152) Load value direct to lds





v_and_b32 v18, -8, v1 // (op 15020) 
v_sub_u32 v19, v1, v18 // (op 15020) 

v_lshlrev_b32 v18, 4, v19 // (op 15020) 

v_lshrrev_b32 v19, 3, v1 // (op 15020) 
v_lshl_add_u32 v27, v19, 7, v18 // (op 15020) 


s_nop 0
v_readfirstlane_b32 s52, v27 // (op 15020) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 15662) 

s_lshl_b64 s[54:55], s[30:31], 5 // (op 14154) 
v_mov_b64 v[18:19], s[54:55] // (op 14154) convert


v_mov_b64 v[28:29], v[22:23] // (op 14154) 
s_add_u32 s53, 32768, s52 // (op 14154) 
s_mov_b32 m0, s53 // (op 14154) reg expression
s_mov_b32 s54, 128 // (op 14154) call()
buffer_load_dwordx4 v28, s[68:71], s54 offen offset: 0 lds // (op 14154) Load value direct to lds
v_add_co_u32 v28, s[84:85], v28, v18 // (op 14154) least significant half
v_addc_co_u32 v29, s[84:85], 0, v29, s[84:85] // (op 14154) most significant half

s_add_u32 m0, m0, 4096 // (op 14154) 
buffer_load_dwordx4 v28, s[68:71], s54 offen offset: 0 lds // (op 14154) Load value direct to lds
v_add_co_u32 v28, s[84:85], v28, v18 // (op 14154) least significant half
v_addc_co_u32 v29, s[84:85], 0, v29, s[84:85] // (op 14154) most significant half

s_add_u32 m0, m0, 4096 // (op 14154) 
buffer_load_dwordx4 v28, s[68:71], s54 offen offset: 0 lds // (op 14154) Load value direct to lds
v_add_co_u32 v28, s[84:85], v28, v18 // (op 14154) least significant half
v_addc_co_u32 v29, s[84:85], 0, v29, s[84:85] // (op 14154) most significant half

s_add_u32 m0, m0, 4096 // (op 14154) 
buffer_load_dwordx4 v28, s[68:71], s54 offen offset: 0 lds // (op 14154) Load value direct to lds






s_waitcnt vmcnt(10)
s_barrier  // (op 15582) 



ds_read_b128 v[28:31], v15 // (op 13842) Load local data 



ds_read_b128 v[32:35], v15 offset:1024 // (op 13847) Load local data 



ds_read_b128 v[36:39], v15 offset:2048 // (op 13852) Load local data 



ds_read_b128 v[40:43], v15 offset:3072 // (op 13857) Load local data 
s_waitcnt vmcnt(6)
s_barrier  // (op 15563) 



ds_read_b128 v[44:47], v13 offset:8192 // (op 13822) Load local data 
ds_read_b128 v[48:51], v13 offset:8256 // (op 13822) Load local data 



ds_read_b128 v[52:55], v13 offset:10240 // (op 13827) Load local data 
ds_read_b128 v[56:59], v13 offset:10304 // (op 13827) Load local data 



ds_read_b128 v[60:63], v13 offset:12288 // (op 13832) Load local data 
ds_read_b128 v[64:67], v13 offset:12352 // (op 13832) Load local data 



ds_read_b128 v[68:71], v13 offset:14336 // (op 13837) Load local data 
ds_read_b128 v[72:75], v13 offset:14400 // (op 13837) Load local data 
v_and_b32 v27, -8, v1 // (op 14526) 
v_sub_u32 v18, v1, v27 // (op 14526) 

v_lshlrev_b32 v19, 4, v18 // (op 14526) 

v_lshrrev_b32 v18, 3, v1 // (op 14526) 
v_lshl_add_u32 v27, v18, 7, v19 // (op 14526) 


s_nop 0
v_readfirstlane_b32 s52, v27 // (op 14526) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -4, v1 // (op 14601) 
v_sub_u32 v18, v1, v27 // (op 14601) 

v_lshlrev_b32 v19, 5, v18 // (op 14601) 

v_lshrrev_b32 v18, 2, v1 // (op 14601) 
v_lshl_add_u32 v27, v18, 7, v19 // (op 14601) 


v_lshlrev_b32 v18, 2, v27 // (op 14601) 

v_lshrrev_b32 v19, 3, v18 // (op 14601) 

s_nop 0
v_readfirstlane_b32 s53, v19 // (op 14601) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -8, v1 // (op 15005) 
v_sub_u32 v18, v1, v27 // (op 15005) 

v_lshlrev_b32 v19, 4, v18 // (op 15005) 

v_lshrrev_b32 v18, 3, v1 // (op 15005) 
v_lshl_add_u32 v27, v18, 7, v19 // (op 15005) 


s_nop 0
v_readfirstlane_b32 s54, v27 // (op 15005) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -4, v1 // (op 15049) 
v_sub_u32 v18, v1, v27 // (op 15049) 

v_lshlrev_b32 v19, 5, v18 // (op 15049) 

v_lshrrev_b32 v18, 2, v1 // (op 15049) 
v_lshl_add_u32 v27, v18, 7, v19 // (op 15049) 


v_lshlrev_b32 v18, 2, v27 // (op 15049) 

v_lshrrev_b32 v19, 3, v18 // (op 15049) 

s_nop 0
v_readfirstlane_b32 s55, v19 // (op 15049) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s1, 0 // (op 5890) call()
s_and_b32 s84, s7, 4294967294 // (op 5889) 
s_cmp_lt_u32 s1, s84 // (op 5889) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_5889 // (op 5889) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_5889 if false)
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_5889:


s_add_u32 s1, s1, 2 // (op 5891) 
v_mov_b32 v27, v16 // (op 72) 
v_mov_b32 v18, v27 // (op 72) reg expression

buffer_load_dwordx2 v[22:23], v18, s[76:79], 0 offen // (op 72) Load value


v_add_u32 v16, 8, v16 // (op 15247) 


v_mov_b32 v27, v26 // (op 76) 
v_mov_b32 v18, v27 // (op 76) reg expression

buffer_load_dwordx2 v[76:77], v18, s[80:83], 0 offen // (op 76) Load value


v_add_u32 v26, 8, v26 // (op 15274) 
s_waitcnt vmcnt(1)
v_bfe_u32 v78, v22, 0, 8 // (op 13073) 
v_bfe_u32 v79, v22, 8, 8 // (op 13073) 
v_bfe_u32 v80, v22, 16, 8 // (op 13073) 
v_bfe_u32 v81, v22, 24, 8 // (op 13073) 
v_permlane16_swap_b32 v78, v79 // (op 13073) 
s_nop 0
v_permlane16_swap_b32 v80, v81 // (op 13073) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v78, v80 // (op 13073) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v79, v81 // (op 13073) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 15588) 

s_lshl_b64 s[84:85], s[36:37], 6 // (op 14069) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 14069) 

v_lshrrev_b64 v[18:19], 3, s[86:87] // (op 14069) 


v_mov_b64 v[82:83], v[20:21] // (op 14069) 
v_mov_b64 v[84:85], v[82:83] // (op 14069) 
s_mov_b32 m0, s53 // (op 14069) reg expression
s_mov_b32 s84, 128 // (op 14069) call()
buffer_load_dwordx4 v84, s[72:75], s84 offen offset: 0 lds // (op 14069) Load value direct to lds
v_add_co_u32 v84, s[86:87], v84, v18 // (op 14069) least significant half
v_addc_co_u32 v85, s[86:87], 0, v85, s[86:87] // (op 14069) most significant half

s_add_u32 m0, m0, 4096 // (op 14069) 
buffer_load_dwordx4 v84, s[72:75], s84 offen offset: 0 lds // (op 14069) Load value direct to lds



s_waitcnt vmcnt(2)
v_bfe_u32 v84, v76, 0, 8 // (op 13136) 
v_bfe_u32 v85, v76, 8, 8 // (op 13136) 
v_bfe_u32 v86, v76, 16, 8 // (op 13136) 
v_bfe_u32 v87, v76, 24, 8 // (op 13136) 
v_permlane16_swap_b32 v84, v85 // (op 13136) 
s_nop 0
v_permlane16_swap_b32 v86, v87 // (op 13136) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v84, v86 // (op 13136) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v85, v87 // (op 13136) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 15569) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[28:31], a[0:3], v78, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 78) 
s_lshl_b64 s[84:85], s[30:31], 5 // (op 14064) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[32:35], a[4:7], v78, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4125) 
v_mov_b64 v[18:19], s[84:85] // (op 14064) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[44:51], v[36:39], a[8:11], v78, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4261) 
v_mov_b64 v[88:89], v[24:25] // (op 14064) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[44:51], v[40:43], a[12:15], v78, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4397) 
v_mov_b64 v[90:91], v[88:89] // (op 14064) 
s_add_u32 s84, 8192, s52 // (op 14064) 
s_barrier  // (op 15668) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[52:59], v[28:31], a[16:19], v79, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4817) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[52:59], v[32:35], a[20:23], v79, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4779) 
s_mov_b32 m0, s84 // (op 14064) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[52:59], v[36:39], a[24:27], v79, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4741) 
s_mov_b32 s85, 256 // (op 14064) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[52:59], v[40:43], a[28:31], v79, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4703) 
buffer_load_dwordx4 v90, s[68:71], s85 offen offset: 0 lds // (op 14064) Load value direct to lds
v_add_co_u32 v90, s[86:87], v90, v18 // (op 14064) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[60:67], v[28:31], a[32:35], v80, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5427) 
v_addc_co_u32 v91, s[86:87], 0, v91, s[86:87] // (op 14064) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[60:67], v[32:35], a[36:39], v80, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5389) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[60:67], v[36:39], a[40:43], v80, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5351) 
s_add_u32 m0, m0, 4096 // (op 14064) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[60:67], v[40:43], a[44:47], v80, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5313) 
ds_read_b128 v[92:95], v10 offset:24576 // (op 4026) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:75], v[28:31], a[48:51], v81, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6037) 
buffer_load_dwordx4 v90, s[68:71], s85 offen offset: 0 lds // (op 14064) Load value direct to lds
v_add_co_u32 v90, s[86:87], v90, v18 // (op 14064) least significant half



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:75], v[32:35], a[52:55], v81, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5999) 
v_addc_co_u32 v91, s[86:87], 0, v91, s[86:87] // (op 14064) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:75], v[36:39], a[56:59], v81, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5961) 
s_add_u32 m0, m0, 4096 // (op 14064) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:75], v[40:43], a[60:63], v81, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5923) 
buffer_load_dwordx4 v90, s[68:71], s85 offen offset: 0 lds // (op 14064) Load value direct to lds


ds_read_b128 v[78:81], v10 offset:25600 // (op 4134) Load local data 
v_add_co_u32 v90, s[86:87], v90, v18 // (op 14064) least significant half



v_addc_co_u32 v91, s[86:87], 0, v91, s[86:87] // (op 14064) most significant half

s_add_u32 m0, m0, 4096 // (op 14064) 
ds_read_b128 v[84:87], v10 offset:26624 // (op 4270) Load local data 
buffer_load_dwordx4 v90, s[68:71], s85 offen offset: 0 lds // (op 14064) Load value direct to lds







ds_read_b128 v[96:99], v10 offset:27648 // (op 4406) Load local data 
s_barrier  // (op 15659) 



ds_read_b128 v[44:47], v9 offset:32768 // (op 4024) Load local data 
ds_read_b128 v[48:51], v9 offset:32832 // (op 4024) Load local data 



ds_read_b128 v[52:55], v9 offset:34816 // (op 4711) Load local data 
ds_read_b128 v[56:59], v9 offset:34880 // (op 4711) Load local data 



ds_read_b128 v[60:63], v9 offset:36864 // (op 5321) Load local data 
ds_read_b128 v[64:67], v9 offset:36928 // (op 5321) Load local data 



ds_read_b128 v[100:103], v9 offset:38912 // (op 5931) Load local data 
ds_read_b128 v[104:107], v9 offset:38976 // (op 5931) Load local data 
v_bfe_u32 v108, v23, 0, 8 // (op 13088) 
v_bfe_u32 v109, v23, 8, 8 // (op 13088) 
v_bfe_u32 v110, v23, 16, 8 // (op 13088) 
v_bfe_u32 v111, v23, 24, 8 // (op 13088) 
v_permlane16_swap_b32 v108, v109 // (op 13088) 
s_nop 0
v_permlane16_swap_b32 v110, v111 // (op 13088) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v108, v110 // (op 13088) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v109, v111 // (op 13088) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 15675) 


s_lshl_b64 s[84:85], s[36:37], 6 // (op 14123) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 14123) 

v_lshrrev_b64 v[18:19], 3, s[86:87] // (op 14123) 


v_mov_b64 v[22:23], v[82:83] // (op 14123) 
s_add_u32 s84, 24576, s55 // (op 14123) 
s_mov_b32 m0, s84 // (op 14123) reg expression
s_mov_b32 s85, 192 // (op 14123) call()
v_bfe_u32 v112, v77, 0, 8 // (op 13151) 
v_bfe_u32 v113, v77, 8, 8 // (op 13151) 
v_bfe_u32 v114, v77, 16, 8 // (op 13151) 
v_bfe_u32 v115, v77, 24, 8 // (op 13151) 
v_permlane16_swap_b32 v112, v113 // (op 13151) 
s_nop 0
v_permlane16_swap_b32 v114, v115 // (op 13151) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v112, v114 // (op 13151) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v113, v115 // (op 13151) 
buffer_load_dwordx4 v22, s[72:75], s85 offen offset: 0 lds // (op 14123) Load value direct to lds

v_add_co_u32 v22, s[86:87], v22, v18 // (op 14123) least significant half
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[92:95], a[0:3], v108, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4031) 
v_addc_co_u32 v23, s[86:87], 0, v23, s[86:87] // (op 14123) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[78:81], a[4:7], v108, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4126) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[44:51], v[84:87], a[8:11], v108, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4262) 
s_add_u32 m0, m0, 4096 // (op 14123) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[44:51], v[96:99], a[12:15], v108, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4398) 
buffer_load_dwordx4 v22, s[72:75], s85 offen offset: 0 lds // (op 14123) Load value direct to lds





s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[52:59], v[92:95], a[16:19], v109, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4818) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[52:59], v[78:81], a[20:23], v109, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4780) 
v_add_co_u32 v20, vcc, 128, v20 // (op 14431) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[52:59], v[84:87], a[24:27], v109, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4742) 
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 14431) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[52:59], v[96:99], a[28:31], v109, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4704) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 15665) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[60:67], v[92:95], a[32:35], v110, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5428) 
s_lshl_b64 s[84:85], s[30:31], 5 // (op 14118) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[60:67], v[78:81], a[36:39], v110, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5390) 
v_mov_b64 v[18:19], s[84:85] // (op 14118) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[60:67], v[84:87], a[40:43], v110, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5352) 
v_mov_b64 v[22:23], v[88:89] // (op 14118) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[60:67], v[96:99], a[44:47], v110, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5314) 
s_add_u32 s84, 32768, s54 // (op 14118) 
s_mov_b32 m0, s84 // (op 14118) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[100:107], v[92:95], a[48:51], v111, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6038) 
s_mov_b32 s85, 384 // (op 14118) call()

buffer_load_dwordx4 v22, s[68:71], s85 offen offset: 0 lds // (op 14118) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[100:107], v[78:81], a[52:55], v111, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6000) 
v_add_co_u32 v22, s[86:87], v22, v18 // (op 14118) least significant half

v_addc_co_u32 v23, s[86:87], 0, v23, s[86:87] // (op 14118) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[100:107], v[84:87], a[56:59], v111, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5962) 

s_add_u32 m0, m0, 4096 // (op 14118) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[100:107], v[96:99], a[60:63], v111, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5924) 




s_waitcnt vmcnt(7)
s_barrier  // (op 15594) 
buffer_load_dwordx4 v22, s[68:71], s85 offen offset: 0 lds // (op 14118) Load value direct to lds



ds_read_b128 v[28:31], v4 // (op 67) Load local data 
v_add_co_u32 v22, s[86:87], v22, v18 // (op 14118) least significant half



ds_read_b128 v[32:35], v4 offset:1024 // (op 4122) Load local data 
v_addc_co_u32 v23, s[86:87], 0, v23, s[86:87] // (op 14118) most significant half




s_add_u32 m0, m0, 4096 // (op 14118) 
ds_read_b128 v[36:39], v4 offset:2048 // (op 4258) Load local data 
buffer_load_dwordx4 v22, s[68:71], s85 offen offset: 0 lds // (op 14118) Load value direct to lds



v_add_co_u32 v22, s[86:87], v22, v18 // (op 14118) least significant half
v_addc_co_u32 v23, s[86:87], 0, v23, s[86:87] // (op 14118) most significant half

s_add_u32 m0, m0, 4096 // (op 14118) 
ds_read_b128 v[40:43], v4 offset:3072 // (op 4394) Load local data 
buffer_load_dwordx4 v22, s[68:71], s85 offen offset: 0 lds // (op 14118) Load value direct to lds





s_waitcnt vmcnt(6)
s_barrier  // (op 15575) 



ds_read_b128 v[44:47], v3 offset:8192 // (op 62) Load local data 
ds_read_b128 v[48:51], v3 offset:8256 // (op 62) Load local data 
v_add_co_u32 v24, vcc, 256, v24 // (op 14303) least significant half



v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 14303) most significant half
ds_read_b128 v[52:55], v3 offset:10240 // (op 4699) Load local data 
ds_read_b128 v[56:59], v3 offset:10304 // (op 4699) Load local data 



ds_read_b128 v[60:63], v3 offset:12288 // (op 5309) Load local data 
ds_read_b128 v[64:67], v3 offset:12352 // (op 5309) Load local data 



ds_read_b128 v[68:71], v3 offset:14336 // (op 5919) Load local data 
ds_read_b128 v[72:75], v3 offset:14400 // (op 5919) Load local data 
s_and_b32 s84, s7, 4294967294 // (op 5889) 
s_cmp_lt_u32 s1, s84 // (op 5889) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_5889 // (op 5889) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_5889 if true)
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_5889:










s_barrier  // (op 16030) 
s_and_b32 s52, s7, 4294967294 // (op 14191) 
v_lshrrev_b32 v16, 3, v1 // (op 14191) 
s_add_u32 s53, s0, s52 // (op 14191) 

v_and_b32 v18, -8, v1 // (op 14191) 
v_sub_u32 v19, v1, v18 // (op 14191) 

s_mul_i32 s52, s16, s8 // (op 14191) 
s_add_u32 s52, s52, s53 // (op 14191) 

v_lshlrev_b32 v18, 4, v19 // (op 14191) 

s_mul_hi_u32 s53, s52, s13 // (op 14191) 
s_sub_u32 s54, s52, s53 // (op 14191) 
s_lshr_b32 s55, s54, 1 // (op 14191) 

s_add_u32 s54, s55, s53 // (op 14191) 


s_lshr_b32 s53, s54, s11 // (op 14191) 

s_and_b32 s54, s11, 2147483648 // (op 14191) 
s_cmp_eq_i32 s54, 0 // (op 14191) 
s_mov_b32 s55, scc // (op 14191) 

s_cmp_eq_i32 s55, 0 // (op 14191) 
s_cselect_b32 s54, s52, s53 // (op 14191) 


s_mul_hi_u32 s53, s54, s12 // (op 14191) 
s_sub_u32 s55, s54, s53 // (op 14191) 
s_lshr_b32 s76, s55, 1 // (op 14191) 

s_add_u32 s55, s76, s53 // (op 14191) 


s_lshr_b32 s53, s55, s10 // (op 14191) 

s_and_b32 s55, s10, 2147483648 // (op 14191) 
s_cmp_eq_i32 s55, 0 // (op 14191) 
s_mov_b32 s76, scc // (op 14191) 

s_cmp_eq_i32 s76, 0 // (op 14191) 
s_cselect_b32 s55, s54, s53 // (op 14191) 


v_lshl_add_u32 v19, s55, 7, v16 // (op 14191) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 14191) 
s_mul_i32 s53, s54, s76 // (op 14191) 


s_sub_u32 s54, s52, s53 // (op 14191) 


v_lshl_add_u32 v16, s54, 7, v18 // (op 14191) 


v_mov_b32 v21, 0 // (op 14191) convert
v_mov_b32 v20, v16 // (op 14191) convert

v_mul_lo_u32 v23, v19, s31 // (op 14191) most significant: low of low * high
v_mul_hi_u32 v16, v19, s30 // (op 14191) most significant: high of low * low
v_mul_lo_u32 v22, v19, s30 // (op 14191) least significant: low of low * low
v_add_u32 v23, v23, v16 // (op 14191) most significant: sum

v_add_co_u32 v22, s[52:53], v22, v20 // (op 14191) least significant half
v_addc_co_u32 v23, s[52:53], v23, v21, s[52:53] // (op 14191) most significant half



s_mul_i32 s53, s30, s35 // (op 14193) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 14193) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 14193) least significant
s_mul_hi_u32 s55, s30, s34 // (op 14193) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 14193) most significant: sum
s_add_u32 s53, s53, s55 // (op 14193) most significant: sum


s_mov_b32 s68, s32 // (op 14193) reg expression
s_mov_b32 s69, s33 // (op 14193) reg expression
s_mov_b32 s70, s52 // (op 14193) reg expression
s_mov_b32 s71, 131072 // (op 14193) call()

s_and_b32 s52, s7, 4294967294 // (op 14355) 
v_lshrrev_b32 v16, 2, v1 // (op 14355) 
s_add_u32 s53, s0, s52 // (op 14355) 

v_and_b32 v18, -4, v1 // (op 14355) 
v_sub_u32 v19, v1, v18 // (op 14355) 

s_mul_i32 s52, s16, s8 // (op 14355) 
s_add_u32 s52, s52, s53 // (op 14355) 

v_lshlrev_b32 v18, 5, v19 // (op 14355) 

s_mul_hi_u32 s53, s52, s13 // (op 14355) 
s_sub_u32 s54, s52, s53 // (op 14355) 
s_lshr_b32 s55, s54, 1 // (op 14355) 

s_add_u32 s54, s55, s53 // (op 14355) 


s_lshr_b32 s53, s54, s11 // (op 14355) 

s_and_b32 s54, s11, 2147483648 // (op 14355) 
s_cmp_eq_i32 s54, 0 // (op 14355) 
s_mov_b32 s55, scc // (op 14355) 

s_cmp_eq_i32 s55, 0 // (op 14355) 
s_cselect_b32 s54, s52, s53 // (op 14355) 


s_mul_hi_u32 s53, s54, s12 // (op 14355) 
s_sub_u32 s55, s54, s53 // (op 14355) 
s_lshr_b32 s76, s55, 1 // (op 14355) 

s_add_u32 s55, s76, s53 // (op 14355) 


s_lshr_b32 s53, s55, s10 // (op 14355) 

s_and_b32 s55, s10, 2147483648 // (op 14355) 
s_cmp_eq_i32 s55, 0 // (op 14355) 
s_mov_b32 s76, scc // (op 14355) 

s_cmp_eq_i32 s76, 0 // (op 14355) 
s_cselect_b32 s55, s54, s53 // (op 14355) 


s_mul_i32 s53, s55, s14 // (op 14355) 

s_sub_u32 s55, s54, s53 // (op 14355) 

v_lshl_add_u32 v19, s55, 7, v16 // (op 14355) 


v_mul_lo_u32 v21, v19, s37 // (op 14355) most significant: low of low * high
v_mul_hi_u32 v16, v19, s36 // (op 14355) most significant: high of low * low
v_mul_lo_u32 v20, v19, s36 // (op 14355) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 14355) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 14355) 
s_mul_i32 s53, s54, s76 // (op 14355) 


s_sub_u32 s54, s52, s53 // (op 14355) 


v_lshl_add_u32 v16, s54, 7, v18 // (op 14355) 


v_mov_b32 v19, 0 // (op 14355) convert
v_mov_b32 v18, v16 // (op 14355) convert

v_add_co_u32 v24, s[52:53], v18, v20 // (op 14355) least significant half
v_addc_co_u32 v25, s[52:53], v19, v21, s[52:53] // (op 14355) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 14355) 


v_lshrrev_b64 v[26:27], 3, v[24:25] // (op 14355) 

s_mul_i32 s53, s36, s23 // (op 14357) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 14357) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 14357) least significant
s_mul_hi_u32 s55, s36, s22 // (op 14357) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 14357) most significant: sum
s_add_u32 s53, s53, s55 // (op 14357) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 14357) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 14357) 

s_mov_b32 s72, s38 // (op 14357) reg expression
s_mov_b32 s73, s39 // (op 14357) reg expression
s_mov_b32 s74, s52 // (op 14357) reg expression
s_mov_b32 s75, 131072 // (op 14357) call()

v_and_b32 v16, -8, v1 // (op 14511) 
v_sub_u32 v18, v1, v16 // (op 14511) 

v_lshlrev_b32 v16, 4, v18 // (op 14511) 

v_lshrrev_b32 v18, 3, v1 // (op 14511) 
v_lshl_add_u32 v19, v18, 7, v16 // (op 14511) 


s_nop 0
v_readfirstlane_b32 s52, v19 // (op 14511) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 14588) 
v_sub_u32 v18, v1, v16 // (op 14588) 

v_lshlrev_b32 v16, 5, v18 // (op 14588) 

v_lshrrev_b32 v18, 2, v1 // (op 14588) 
v_lshl_add_u32 v19, v18, 7, v16 // (op 14588) 


v_lshlrev_b32 v16, 2, v19 // (op 14588) 

v_lshrrev_b32 v18, 3, v16 // (op 14588) 

s_nop 0
v_readfirstlane_b32 s53, v18 // (op 14588) Wait state hazard: VALU Write Readlane Hazard

s_and_b32 s54, s7, 4294967294 // (op 15311) 
v_and_b32 v16, 63, v1 // (op 15311) 
s_add_u32 s55, s0, s54 // (op 15311) 

v_lshrrev_b32 v18, 4, v16 // (op 15311) 
v_and_b32 v19, 3, v18 // (op 15311) 

s_mul_i32 s54, s16, s8 // (op 15311) 
s_add_u32 s54, s54, s55 // (op 15311) 

v_and_b32 v18, 15, v1 // (op 15311) 
s_mul_hi_u32 s55, s54, s13 // (op 15311) 
v_lshl_add_u32 v20, v19, 4, v18 // (op 15311) 


s_sub_u32 s76, s54, s55 // (op 15311) 
v_lshrrev_b32 v21, 7, v1 // (op 15311) 
s_lshr_b32 s77, s76, 1 // (op 15311) 

v_lshl_add_u32 v18, v21, 6, v20 // (op 15311) 


s_add_u32 s76, s77, s55 // (op 15311) 


v_lshrrev_b32 v19, 6, v16 // (op 15311) 

s_lshr_b32 s55, s76, s11 // (op 15311) 

v_lshlrev_b32 v16, 2, v19 // (op 15311) 

s_and_b32 s76, s11, 2147483648 // (op 15311) 
s_cmp_eq_i32 s76, 0 // (op 15311) 
s_mov_b32 s77, scc // (op 15311) 

s_cmp_eq_i32 s77, 0 // (op 15311) 
s_cselect_b32 s76, s54, s55 // (op 15311) 


s_mul_hi_u32 s55, s76, s12 // (op 15311) 
s_sub_u32 s77, s76, s55 // (op 15311) 
s_lshr_b32 s78, s77, 1 // (op 15311) 

s_add_u32 s77, s78, s55 // (op 15311) 


s_lshr_b32 s55, s77, s10 // (op 15311) 

s_and_b32 s77, s10, 2147483648 // (op 15311) 
s_cmp_eq_i32 s77, 0 // (op 15311) 
s_mov_b32 s78, scc // (op 15311) 

s_cmp_eq_i32 s78, 0 // (op 15311) 
s_cselect_b32 s77, s76, s55 // (op 15311) 


v_lshl_add_u32 v19, s77, 7, v18 // (op 15311) 


s_ashr_i64 s[78:79], s[2:3], 7 // (op 15311) 
s_mul_i32 s55, s76, s78 // (op 15311) 


s_sub_u32 s76, s54, s55 // (op 15311) 


v_lshl_add_u32 v18, s76, 2, v16 // (op 15311) 


v_mul_lo_u32 v16, v19, s40 // (op 15311) 
v_add_u32 v16, v16, v18 // (op 15311) 


s_mul_i32 s55, s40, s35 // (op 15313) most significant: low of low * high
s_mul_i32 s76, s41, s34 // (op 15313) most significant: low of high * low
s_mul_i32 s54, s40, s34 // (op 15313) least significant
s_mul_hi_u32 s77, s40, s34 // (op 15313) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 15313) most significant: sum
s_add_u32 s55, s55, s77 // (op 15313) most significant: sum


s_mov_b32 s76, s42 // (op 15313) reg expression
s_mov_b32 s77, s43 // (op 15313) reg expression
s_mov_b32 s78, s54 // (op 15313) reg expression
s_mov_b32 s79, 131072 // (op 15313) call()

s_and_b32 s54, s7, 4294967294 // (op 15338) 
v_and_b32 v18, 63, v1 // (op 15338) 
s_add_u32 s55, s0, s54 // (op 15338) 

v_lshrrev_b32 v19, 4, v18 // (op 15338) 
v_and_b32 v20, 3, v19 // (op 15338) 

s_mul_i32 s54, s16, s8 // (op 15338) 
s_add_u32 s54, s54, s55 // (op 15338) 

v_and_b32 v19, 15, v1 // (op 15338) 
s_mul_hi_u32 s55, s54, s13 // (op 15338) 
v_lshl_add_u32 v21, v20, 4, v19 // (op 15338) 


s_sub_u32 s80, s54, s55 // (op 15338) 
v_lshrrev_b32 v19, 6, v1 // (op 15338) 
s_lshr_b32 s81, s80, 1 // (op 15338) 

v_and_b32 v20, 1, v19 // (op 15338) 

s_add_u32 s80, s81, s55 // (op 15338) 


v_lshl_add_u32 v19, v20, 6, v21 // (op 15338) 


s_lshr_b32 s55, s80, s11 // (op 15338) 

v_lshrrev_b32 v20, 6, v18 // (op 15338) 

s_and_b32 s80, s11, 2147483648 // (op 15338) 
v_lshlrev_b32 v18, 2, v20 // (op 15338) 

s_cmp_eq_i32 s80, 0 // (op 15338) 
s_mov_b32 s81, scc // (op 15338) 

s_cmp_eq_i32 s81, 0 // (op 15338) 
s_cselect_b32 s80, s54, s55 // (op 15338) 


s_mul_hi_u32 s55, s80, s12 // (op 15338) 
s_sub_u32 s81, s80, s55 // (op 15338) 
s_lshr_b32 s82, s81, 1 // (op 15338) 

s_add_u32 s81, s82, s55 // (op 15338) 


s_lshr_b32 s55, s81, s10 // (op 15338) 

s_and_b32 s81, s10, 2147483648 // (op 15338) 
s_cmp_eq_i32 s81, 0 // (op 15338) 
s_mov_b32 s82, scc // (op 15338) 

s_cmp_eq_i32 s82, 0 // (op 15338) 
s_cselect_b32 s81, s80, s55 // (op 15338) 


s_mul_i32 s55, s81, s14 // (op 15338) 

s_sub_u32 s81, s80, s55 // (op 15338) 

v_lshl_add_u32 v20, s81, 7, v19 // (op 15338) 


v_mul_lo_u32 v19, v20, s44 // (op 15338) 

s_ashr_i64 s[82:83], s[2:3], 7 // (op 15338) 
s_mul_i32 s55, s80, s82 // (op 15338) 


s_sub_u32 s80, s54, s55 // (op 15338) 


v_lshl_add_u32 v20, s80, 2, v18 // (op 15338) 


v_add_u32 v18, v20, v19 // (op 15338) 


s_mul_i32 s55, s44, s23 // (op 15340) most significant: low of low * high
s_mul_i32 s80, s45, s22 // (op 15340) most significant: low of high * low
s_mul_i32 s54, s44, s22 // (op 15340) least significant
s_mul_hi_u32 s81, s44, s22 // (op 15340) most significant: high of low * low
s_add_u32 s55, s55, s80 // (op 15340) most significant: sum
s_add_u32 s55, s55, s81 // (op 15340) most significant: sum


s_mov_b32 s80, s46 // (op 15340) reg expression
s_mov_b32 s81, s47 // (op 15340) reg expression
s_mov_b32 s82, s54 // (op 15340) reg expression
s_mov_b32 s83, 131072 // (op 15340) call()

s_and_b32 s54, s7, 4294967294 // (op 3967) 
s_cmp_lt_u32 s54, s7 // (op 3966) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_3966 // (op 3966) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_3966 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_3966:
s_barrier  // (op 15572) 
s_add_u32 s54, s54, 1 // (op 3968) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 14106) 
v_mov_b64 v[20:21], s[84:85] // (op 14106) convert


v_mov_b64 v[24:25], v[22:23] // (op 14106) 
v_mov_b64 v[28:29], v[24:25] // (op 14106) 
s_add_u32 s55, 8192, s52 // (op 14106) 
s_mov_b32 m0, s55 // (op 14106) reg expression
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 14106) Load value direct to lds
v_add_co_u32 v28, s[84:85], v28, v20 // (op 14106) least significant half
v_addc_co_u32 v29, s[84:85], 0, v29, s[84:85] // (op 14106) most significant half

s_add_u32 m0, m0, 4096 // (op 14106) 
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 14106) Load value direct to lds
v_add_co_u32 v28, s[84:85], v28, v20 // (op 14106) least significant half
v_addc_co_u32 v29, s[84:85], 0, v29, s[84:85] // (op 14106) most significant half

s_add_u32 m0, m0, 4096 // (op 14106) 
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 14106) Load value direct to lds
v_add_co_u32 v28, s[84:85], v28, v20 // (op 14106) least significant half
v_addc_co_u32 v29, s[84:85], 0, v29, s[84:85] // (op 14106) most significant half

s_add_u32 m0, m0, 4096 // (op 14106) 
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 14106) Load value direct to lds




v_add_co_u32 v22, vcc, 128, v22 // (op 14181) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 14181) most significant half
s_barrier  // (op 15591) 

s_lshl_b64 s[84:85], s[36:37], 6 // (op 14112) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 14112) 

v_lshrrev_b64 v[20:21], 3, s[86:87] // (op 14112) 


v_mov_b64 v[24:25], v[26:27] // (op 14112) 
v_mov_b64 v[28:29], v[24:25] // (op 14112) 
s_mov_b32 m0, s53 // (op 14112) reg expression
buffer_load_dwordx4 v28, s[72:75], 0 offen offset: 0 lds // (op 14112) Load value direct to lds
v_add_co_u32 v28, s[84:85], v28, v20 // (op 14112) least significant half
v_addc_co_u32 v29, s[84:85], 0, v29, s[84:85] // (op 14112) most significant half

s_add_u32 m0, m0, 4096 // (op 14112) 
buffer_load_dwordx4 v28, s[72:75], 0 offen offset: 0 lds // (op 14112) Load value direct to lds



v_add_co_u32 v26, s[84:85], 64, v26 // (op 14347) least significant half
v_addc_co_u32 v27, s[84:85], 0, v27, s[84:85] // (op 14347) most significant half

s_waitcnt vmcnt(2)
s_barrier  // (op 15578) 



ds_read_b128 v[28:31], v3 offset:8192 // (op 3974) Load local data 
ds_read_b128 v[32:35], v3 offset:8256 // (op 3974) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 15597) 



ds_read_b128 v[36:39], v4 // (op 3976) Load local data 


v_mov_b32 v19, v16 // (op 3978) 
v_mov_b32 v20, v19 // (op 3978) reg expression

buffer_load_dwordx2 v[24:25], v20, s[76:79], 0 offen // (op 3978) Load value


v_add_u32 v16, 4, v16 // (op 15301) 


v_mov_b32 v19, v18 // (op 3980) 
v_mov_b32 v20, v19 // (op 3980) reg expression

buffer_load_dwordx2 v[40:41], v20, s[80:83], 0 offen // (op 3980) Load value


v_add_u32 v18, 4, v18 // (op 15328) 
s_waitcnt vmcnt(1)
v_bfe_u32 v42, v24, 0, 8 // (op 13200) 



v_bfe_u32 v43, v24, 8, 8 // (op 13200) 
v_bfe_u32 v44, v24, 16, 8 // (op 13200) 
v_bfe_u32 v45, v24, 24, 8 // (op 13200) 
v_permlane16_swap_b32 v42, v43 // (op 13200) 
s_nop 0
v_permlane16_swap_b32 v44, v45 // (op 13200) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v42, v44 // (op 13200) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v43, v45 // (op 13200) 
ds_read_b128 v[46:49], v4 offset:1024 // (op 4105) Load local data 




ds_read_b128 v[50:53], v4 offset:2048 // (op 4241) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v54, v40, 0, 8 // (op 13230) 



v_bfe_u32 v55, v40, 8, 8 // (op 13230) 
v_bfe_u32 v56, v40, 16, 8 // (op 13230) 
v_bfe_u32 v57, v40, 24, 8 // (op 13230) 
v_permlane16_swap_b32 v54, v55 // (op 13230) 
ds_read_b128 v[58:61], v4 offset:3072 // (op 4377) Load local data 
v_permlane16_swap_b32 v56, v57 // (op 13230) 



s_nop 1
v_permlane32_swap_b32 v54, v56 // (op 13230) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v55, v57 // (op 13230) 
ds_read_b128 v[62:65], v3 offset:10240 // (op 4599) Load local data 

ds_read_b128 v[66:69], v3 offset:10304 // (op 4599) Load local data 



s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[36:39], a[0:3], v42, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 3981) 
ds_read_b128 v[70:73], v3 offset:12288 // (op 5209) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[28:35], v[46:49], a[4:7], v42, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4108) 
ds_read_b128 v[74:77], v3 offset:12352 // (op 5209) Load local data 



s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[28:35], v[50:53], a[8:11], v42, v56 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4244) 
ds_read_b128 v[78:81], v3 offset:14336 // (op 5819) Load local data 
ds_read_b128 v[82:85], v3 offset:14400 // (op 5819) Load local data 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[28:35], v[58:61], a[12:15], v42, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4380) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[62:69], v[36:39], a[16:19], v43, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4603) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[62:69], v[50:53], a[24:27], v43, v56 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4657) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[62:69], v[58:61], a[28:31], v43, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4684) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[62:69], v[46:49], a[20:23], v43, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4630) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[70:77], v[36:39], a[32:35], v44, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5213) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[70:77], v[50:53], a[40:43], v44, v56 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5267) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[70:77], v[58:61], a[44:47], v44, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5294) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:77], v[46:49], a[36:39], v44, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5240) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[78:85], v[36:39], a[48:51], v45, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5823) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[78:85], v[50:53], a[56:59], v45, v56 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5877) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[78:85], v[58:61], a[60:63], v45, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5904) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[78:85], v[46:49], a[52:55], v45, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5850) 




s_cmp_lt_u32 s54, s7 // (op 3966) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_3966 // (op 3966) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_3966 if true)
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_3966:









s_barrier  // (op 15996) 


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 265) 
s_cmp_lg_u32 s18, 0 // (op 291) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Send_Tile_291 // (op 291) Condition: False, jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Send_Tile_291



v_add_u32 v13, 49152, v12 // (op 6789) 
buffer_store_dword a48, v13, s[64:67], 0 offen sc1 // (op 6789) Store value

v_add_u32 v13, 49664, v12 // (op 6789) 
buffer_store_dword a49, v13, s[64:67], 0 offen sc1 // (op 6789) Store value

v_add_u32 v13, 50176, v12 // (op 6789) 
buffer_store_dword a50, v13, s[64:67], 0 offen sc1 // (op 6789) Store value

v_add_u32 v13, 50688, v12 // (op 6789) 
buffer_store_dword a51, v13, s[64:67], 0 offen sc1 // (op 6789) Store value




v_add_u32 v13, 49280, v12 // (op 6791) 
buffer_store_dword a52, v13, s[64:67], 0 offen sc1 // (op 6791) Store value

v_add_u32 v13, 49792, v12 // (op 6791) 
buffer_store_dword a53, v13, s[64:67], 0 offen sc1 // (op 6791) Store value

v_add_u32 v13, 50304, v12 // (op 6791) 
buffer_store_dword a54, v13, s[64:67], 0 offen sc1 // (op 6791) Store value

v_add_u32 v13, 50816, v12 // (op 6791) 
buffer_store_dword a55, v13, s[64:67], 0 offen sc1 // (op 6791) Store value




v_add_u32 v13, 49408, v12 // (op 6793) 
buffer_store_dword a56, v13, s[64:67], 0 offen sc1 // (op 6793) Store value

v_add_u32 v13, 49920, v12 // (op 6793) 
buffer_store_dword a57, v13, s[64:67], 0 offen sc1 // (op 6793) Store value

v_add_u32 v13, 50432, v12 // (op 6793) 
buffer_store_dword a58, v13, s[64:67], 0 offen sc1 // (op 6793) Store value

v_add_u32 v13, 50944, v12 // (op 6793) 
buffer_store_dword a59, v13, s[64:67], 0 offen sc1 // (op 6793) Store value




v_add_u32 v13, 49536, v12 // (op 6795) 
buffer_store_dword a60, v13, s[64:67], 0 offen sc1 // (op 6795) Store value

v_add_u32 v13, 50048, v12 // (op 6795) 
buffer_store_dword a61, v13, s[64:67], 0 offen sc1 // (op 6795) Store value

v_add_u32 v13, 50560, v12 // (op 6795) 
buffer_store_dword a62, v13, s[64:67], 0 offen sc1 // (op 6795) Store value

v_add_u32 v13, 51072, v12 // (op 6795) 
buffer_store_dword a63, v13, s[64:67], 0 offen sc1 // (op 6795) Store value




v_add_u32 v13, 32768, v12 // (op 6765) 
buffer_store_dword a32, v13, s[64:67], 0 offen sc1 // (op 6765) Store value

v_add_u32 v13, 33280, v12 // (op 6765) 
buffer_store_dword a33, v13, s[64:67], 0 offen sc1 // (op 6765) Store value

v_add_u32 v13, 33792, v12 // (op 6765) 
buffer_store_dword a34, v13, s[64:67], 0 offen sc1 // (op 6765) Store value

v_add_u32 v13, 34304, v12 // (op 6765) 
buffer_store_dword a35, v13, s[64:67], 0 offen sc1 // (op 6765) Store value




v_add_u32 v13, 32896, v12 // (op 6767) 
buffer_store_dword a36, v13, s[64:67], 0 offen sc1 // (op 6767) Store value

v_add_u32 v13, 33408, v12 // (op 6767) 
buffer_store_dword a37, v13, s[64:67], 0 offen sc1 // (op 6767) Store value

v_add_u32 v13, 33920, v12 // (op 6767) 
buffer_store_dword a38, v13, s[64:67], 0 offen sc1 // (op 6767) Store value

v_add_u32 v13, 34432, v12 // (op 6767) 
buffer_store_dword a39, v13, s[64:67], 0 offen sc1 // (op 6767) Store value




v_add_u32 v13, 33024, v12 // (op 6769) 
buffer_store_dword a40, v13, s[64:67], 0 offen sc1 // (op 6769) Store value

v_add_u32 v13, 33536, v12 // (op 6769) 
buffer_store_dword a41, v13, s[64:67], 0 offen sc1 // (op 6769) Store value

v_add_u32 v13, 34048, v12 // (op 6769) 
buffer_store_dword a42, v13, s[64:67], 0 offen sc1 // (op 6769) Store value

v_add_u32 v13, 34560, v12 // (op 6769) 
buffer_store_dword a43, v13, s[64:67], 0 offen sc1 // (op 6769) Store value




v_add_u32 v13, 33152, v12 // (op 6771) 
buffer_store_dword a44, v13, s[64:67], 0 offen sc1 // (op 6771) Store value

v_add_u32 v13, 33664, v12 // (op 6771) 
buffer_store_dword a45, v13, s[64:67], 0 offen sc1 // (op 6771) Store value

v_add_u32 v13, 34176, v12 // (op 6771) 
buffer_store_dword a46, v13, s[64:67], 0 offen sc1 // (op 6771) Store value

v_add_u32 v13, 34688, v12 // (op 6771) 
buffer_store_dword a47, v13, s[64:67], 0 offen sc1 // (op 6771) Store value




v_add_u32 v13, 16384, v12 // (op 6741) 
buffer_store_dword a16, v13, s[64:67], 0 offen sc1 // (op 6741) Store value

v_add_u32 v13, 16896, v12 // (op 6741) 
buffer_store_dword a17, v13, s[64:67], 0 offen sc1 // (op 6741) Store value

v_add_u32 v13, 17408, v12 // (op 6741) 
buffer_store_dword a18, v13, s[64:67], 0 offen sc1 // (op 6741) Store value

v_add_u32 v13, 17920, v12 // (op 6741) 
buffer_store_dword a19, v13, s[64:67], 0 offen sc1 // (op 6741) Store value




v_add_u32 v13, 16512, v12 // (op 6743) 
buffer_store_dword a20, v13, s[64:67], 0 offen sc1 // (op 6743) Store value

v_add_u32 v13, 17024, v12 // (op 6743) 
buffer_store_dword a21, v13, s[64:67], 0 offen sc1 // (op 6743) Store value

v_add_u32 v13, 17536, v12 // (op 6743) 
buffer_store_dword a22, v13, s[64:67], 0 offen sc1 // (op 6743) Store value

v_add_u32 v13, 18048, v12 // (op 6743) 
buffer_store_dword a23, v13, s[64:67], 0 offen sc1 // (op 6743) Store value




v_add_u32 v13, 16640, v12 // (op 6745) 
buffer_store_dword a24, v13, s[64:67], 0 offen sc1 // (op 6745) Store value

v_add_u32 v13, 17152, v12 // (op 6745) 
buffer_store_dword a25, v13, s[64:67], 0 offen sc1 // (op 6745) Store value

v_add_u32 v13, 17664, v12 // (op 6745) 
buffer_store_dword a26, v13, s[64:67], 0 offen sc1 // (op 6745) Store value

v_add_u32 v13, 18176, v12 // (op 6745) 
buffer_store_dword a27, v13, s[64:67], 0 offen sc1 // (op 6745) Store value




v_add_u32 v13, 16768, v12 // (op 6747) 
buffer_store_dword a28, v13, s[64:67], 0 offen sc1 // (op 6747) Store value

v_add_u32 v13, 17280, v12 // (op 6747) 
buffer_store_dword a29, v13, s[64:67], 0 offen sc1 // (op 6747) Store value

v_add_u32 v13, 17792, v12 // (op 6747) 
buffer_store_dword a30, v13, s[64:67], 0 offen sc1 // (op 6747) Store value

v_add_u32 v13, 18304, v12 // (op 6747) 
buffer_store_dword a31, v13, s[64:67], 0 offen sc1 // (op 6747) Store value




buffer_store_dword a0, v12, s[64:67], 0 offen sc1 // (op 288) Store value
buffer_store_dword a1, v12, s[64:67], 0 offen offset:512 sc1 // (op 288) Store value
buffer_store_dword a2, v12, s[64:67], 0 offen offset:1024 sc1 // (op 288) Store value
buffer_store_dword a3, v12, s[64:67], 0 offen offset:1536 sc1 // (op 288) Store value



buffer_store_dword a4, v12, s[64:67], 0 offen offset:128 sc1 // (op 6715) Store value
buffer_store_dword a5, v12, s[64:67], 0 offen offset:640 sc1 // (op 6715) Store value
buffer_store_dword a6, v12, s[64:67], 0 offen offset:1152 sc1 // (op 6715) Store value
buffer_store_dword a7, v12, s[64:67], 0 offen offset:1664 sc1 // (op 6715) Store value



buffer_store_dword a8, v12, s[64:67], 0 offen offset:256 sc1 // (op 6716) Store value
buffer_store_dword a9, v12, s[64:67], 0 offen offset:768 sc1 // (op 6716) Store value
buffer_store_dword a10, v12, s[64:67], 0 offen offset:1280 sc1 // (op 6716) Store value
buffer_store_dword a11, v12, s[64:67], 0 offen offset:1792 sc1 // (op 6716) Store value



buffer_store_dword a12, v12, s[64:67], 0 offen offset:384 sc1 // (op 6717) Store value
buffer_store_dword a13, v12, s[64:67], 0 offen offset:896 sc1 // (op 6717) Store value
buffer_store_dword a14, v12, s[64:67], 0 offen offset:1408 sc1 // (op 6717) Store value
buffer_store_dword a15, v12, s[64:67], 0 offen offset:1920 sc1 // (op 6717) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 292) 
v_cmp_eq_i32 vcc, v1, 0 // (op 296) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_296 // (op 296) Condition: False, jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_296
s_mov_b32 s52, 1 // (op 294) call()
s_mov_b32 s55, 0 // (op 295) convert
s_mov_b32 s54, s16 // (op 295) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 295) 
s_add_u32 s76, s48, s54 // (op 295) Start of Int64 add, locking SCC
s_addc_u32 s77, s49, s55 // (op 295) End of Int64 add, unlocking SCC
s_store_dword s52, s[76:77], 0 glc // (op 295) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_296 // (op 296) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_296
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_296:
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_296:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Send_Tile_291 // (op 291) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Send_Tile_291
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Send_Tile_291:
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Send_Tile_291:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 307) 
s_mov_b32 s52, scc // (op 307) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 307) 
s_mov_b32 s53, s54 // (op 307) convert

s_sub_u32 s54, s53, 1 // (op 307) 

s_cmp_lt_u32 s19, s54 // (op 307) 
s_mov_b32 s53, scc // (op 307) 

s_and_b32 s54, s52, s53 // (op 307) 
s_cmp_lg_u32 s54, 0 // (op 307) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Receive_Tile_307 // (op 307) Condition: False, jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Receive_Tile_307
s_mov_b32 s52, 0 // (op 273) call()
s_mul_i32 s53, s52, s8 // (op 272) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 272) 
s_mov_b32 s76, s54 // (op 272) convert

s_sub_u32 s54, s76, s19 // (op 272) 

s_sub_u32 s55, s54, 1 // (op 272) 

s_cmp_lt_u32 s53, s55 // (op 272) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_272 // (op 272) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_272 if false)
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_ReceiveTileLoop_272:
s_add_u32 s53, s16, 1 // (op 14902) 
v_lshrrev_b32 v12, 5, v1 // (op 14902) 
s_add_u32 s54, s53, s52 // (op 14902) 
s_lshl_b32 s54, s54, 7 // (op 14902) 

v_lshlrev_b32 v13, 2, v12 // (op 14902) 

v_add_lshl_u32 v12, s54, v13, 7 // (op 14902) 


v_and_b32 v13, -32, v1 // (op 14902) 
v_sub_u32 v15, v1, v13 // (op 14902) 

v_add_lshl_u32 v13, v12, v15, 2 // (op 14902) 


s_lshl_b32 s53, s15, 16 // (op 14904) 
s_mov_b32 s64, s28 // (op 14904) reg expression
s_mov_b32 s65, s29 // (op 14904) reg expression
s_mov_b32 s66, s53 // (op 14904) reg expression
s_mov_b32 s67, 131072 // (op 14904) call()

RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_311:
s_add_u32 s53, s16, 1 // (op 309) 
s_add_u32 s54, s53, s52 // (op 309) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 309) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 309) 

s_add_u32 s76, s48, s54 // (op 309) Start of Int64 add, locking SCC
s_addc_u32 s77, s49, s55 // (op 309) End of Int64 add, unlocking SCC
s_load_dword s53, s[76:77], 0 glc // (op 309) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 311) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_311 // (op 311) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_311 if true)
s_barrier  // (op 315) 
v_cmp_eq_i32 vcc, v1, 0 // (op 314) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_314 // (op 314) Condition: False, jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_314
s_mov_b32 s53, 0 // (op 312) call()
s_add_u32 s54, s16, 1 // (op 313) 
s_add_u32 s76, s54, s52 // (op 313) Start of Int64 add, locking SCC
s_addc_u32 s77, 0, 0 // (op 313) End of Int64 add, unlocking SCC
s_lshl_b64 s[76:77], s[76:77], 2 // (op 313) 

s_add_u32 s54, s48, s76 // (op 313) Start of Int64 add, locking SCC
s_addc_u32 s55, s49, s77 // (op 313) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 313) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_314 // (op 314) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_314
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_314:
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_314:




buffer_load_dword v18, v13, s[64:67], 0 offen // (op 326) Load value
buffer_load_dword v19, v13, s[64:67], 0 offen offset:512 // (op 326) Load value
buffer_load_dword v20, v13, s[64:67], 0 offen offset:1024 // (op 326) Load value
buffer_load_dword v21, v13, s[64:67], 0 offen offset:1536 // (op 326) Load value
v_accvgpr_read v22, a0 // (op 329) reg expression
v_accvgpr_read v23, a1 // (op 329) reg expression
v_accvgpr_read v24, a2 // (op 329) reg expression
v_accvgpr_read v25, a3 // (op 329) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 331) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 331) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 331) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 331) 

v_accvgpr_write a0, v22 // (op 333) reg expression
v_accvgpr_write a1, v23 // (op 333) reg expression
v_accvgpr_write a2, v24 // (op 333) reg expression
v_accvgpr_write a3, v25 // (op 333) reg expression




buffer_load_dword v18, v13, s[64:67], 0 offen offset:128 // (op 6864) Load value
buffer_load_dword v19, v13, s[64:67], 0 offen offset:640 // (op 6864) Load value
buffer_load_dword v20, v13, s[64:67], 0 offen offset:1152 // (op 6864) Load value
buffer_load_dword v21, v13, s[64:67], 0 offen offset:1664 // (op 6864) Load value
v_accvgpr_read v22, a4 // (op 6865) reg expression
v_accvgpr_read v23, a5 // (op 6865) reg expression
v_accvgpr_read v24, a6 // (op 6865) reg expression
v_accvgpr_read v25, a7 // (op 6865) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6866) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6866) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6866) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6866) 

v_accvgpr_write a4, v22 // (op 6867) reg expression
v_accvgpr_write a5, v23 // (op 6867) reg expression
v_accvgpr_write a6, v24 // (op 6867) reg expression
v_accvgpr_write a7, v25 // (op 6867) reg expression




buffer_load_dword v18, v13, s[64:67], 0 offen offset:256 // (op 6871) Load value
buffer_load_dword v19, v13, s[64:67], 0 offen offset:768 // (op 6871) Load value
buffer_load_dword v20, v13, s[64:67], 0 offen offset:1280 // (op 6871) Load value
buffer_load_dword v21, v13, s[64:67], 0 offen offset:1792 // (op 6871) Load value
v_accvgpr_read v22, a8 // (op 6872) reg expression
v_accvgpr_read v23, a9 // (op 6872) reg expression
v_accvgpr_read v24, a10 // (op 6872) reg expression
v_accvgpr_read v25, a11 // (op 6872) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6873) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6873) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6873) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6873) 

v_accvgpr_write a8, v22 // (op 6874) reg expression
v_accvgpr_write a9, v23 // (op 6874) reg expression
v_accvgpr_write a10, v24 // (op 6874) reg expression
v_accvgpr_write a11, v25 // (op 6874) reg expression




buffer_load_dword v18, v13, s[64:67], 0 offen offset:384 // (op 6878) Load value
buffer_load_dword v19, v13, s[64:67], 0 offen offset:896 // (op 6878) Load value
buffer_load_dword v20, v13, s[64:67], 0 offen offset:1408 // (op 6878) Load value
buffer_load_dword v21, v13, s[64:67], 0 offen offset:1920 // (op 6878) Load value
v_accvgpr_read v22, a12 // (op 6879) reg expression
v_accvgpr_read v23, a13 // (op 6879) reg expression
v_accvgpr_read v24, a14 // (op 6879) reg expression
v_accvgpr_read v25, a15 // (op 6879) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6880) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6880) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6880) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6880) 

v_accvgpr_write a12, v22 // (op 6881) reg expression
v_accvgpr_write a13, v23 // (op 6881) reg expression
v_accvgpr_write a14, v24 // (op 6881) reg expression
v_accvgpr_write a15, v25 // (op 6881) reg expression




v_mov_b32 v12, 16384 // (op 6915) 
v_add_i32 v15, v13, v12 // (op 6915) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6915) Load value

v_mov_b32 v12, 16896 // (op 6915) 
v_add_i32 v15, v13, v12 // (op 6915) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6915) Load value

v_mov_b32 v12, 17408 // (op 6915) 
v_add_i32 v15, v13, v12 // (op 6915) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6915) Load value

v_mov_b32 v12, 17920 // (op 6915) 
v_add_i32 v15, v13, v12 // (op 6915) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6915) Load value

v_accvgpr_read v22, a16 // (op 6912) reg expression
v_accvgpr_read v23, a17 // (op 6912) reg expression
v_accvgpr_read v24, a18 // (op 6912) reg expression
v_accvgpr_read v25, a19 // (op 6912) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6913) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6913) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6913) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6913) 

v_accvgpr_write a16, v22 // (op 6914) reg expression
v_accvgpr_write a17, v23 // (op 6914) reg expression
v_accvgpr_write a18, v24 // (op 6914) reg expression
v_accvgpr_write a19, v25 // (op 6914) reg expression




v_mov_b32 v12, 16512 // (op 6920) 
v_add_i32 v15, v13, v12 // (op 6920) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6920) Load value

v_mov_b32 v12, 17024 // (op 6920) 
v_add_i32 v15, v13, v12 // (op 6920) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6920) Load value

v_mov_b32 v12, 17536 // (op 6920) 
v_add_i32 v15, v13, v12 // (op 6920) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6920) Load value

v_mov_b32 v12, 18048 // (op 6920) 
v_add_i32 v15, v13, v12 // (op 6920) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6920) Load value

v_accvgpr_read v22, a20 // (op 6917) reg expression
v_accvgpr_read v23, a21 // (op 6917) reg expression
v_accvgpr_read v24, a22 // (op 6917) reg expression
v_accvgpr_read v25, a23 // (op 6917) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6918) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6918) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6918) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6918) 

v_accvgpr_write a20, v22 // (op 6919) reg expression
v_accvgpr_write a21, v23 // (op 6919) reg expression
v_accvgpr_write a22, v24 // (op 6919) reg expression
v_accvgpr_write a23, v25 // (op 6919) reg expression




v_mov_b32 v12, 16640 // (op 6925) 
v_add_i32 v15, v13, v12 // (op 6925) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6925) Load value

v_mov_b32 v12, 17152 // (op 6925) 
v_add_i32 v15, v13, v12 // (op 6925) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6925) Load value

v_mov_b32 v12, 17664 // (op 6925) 
v_add_i32 v15, v13, v12 // (op 6925) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6925) Load value

v_mov_b32 v12, 18176 // (op 6925) 
v_add_i32 v15, v13, v12 // (op 6925) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6925) Load value

v_accvgpr_read v22, a24 // (op 6922) reg expression
v_accvgpr_read v23, a25 // (op 6922) reg expression
v_accvgpr_read v24, a26 // (op 6922) reg expression
v_accvgpr_read v25, a27 // (op 6922) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6923) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6923) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6923) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6923) 

v_accvgpr_write a24, v22 // (op 6924) reg expression
v_accvgpr_write a25, v23 // (op 6924) reg expression
v_accvgpr_write a26, v24 // (op 6924) reg expression
v_accvgpr_write a27, v25 // (op 6924) reg expression




v_mov_b32 v12, 16768 // (op 6930) 
v_add_i32 v15, v13, v12 // (op 6930) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6930) Load value

v_mov_b32 v12, 17280 // (op 6930) 
v_add_i32 v15, v13, v12 // (op 6930) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6930) Load value

v_mov_b32 v12, 17792 // (op 6930) 
v_add_i32 v15, v13, v12 // (op 6930) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6930) Load value

v_mov_b32 v12, 18304 // (op 6930) 
v_add_i32 v15, v13, v12 // (op 6930) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6930) Load value

v_accvgpr_read v22, a28 // (op 6927) reg expression
v_accvgpr_read v23, a29 // (op 6927) reg expression
v_accvgpr_read v24, a30 // (op 6927) reg expression
v_accvgpr_read v25, a31 // (op 6927) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6928) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6928) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6928) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6928) 

v_accvgpr_write a28, v22 // (op 6929) reg expression
v_accvgpr_write a29, v23 // (op 6929) reg expression
v_accvgpr_write a30, v24 // (op 6929) reg expression
v_accvgpr_write a31, v25 // (op 6929) reg expression




v_mov_b32 v12, 32768 // (op 6963) 
v_add_i32 v15, v13, v12 // (op 6963) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6963) Load value

v_mov_b32 v12, 33280 // (op 6963) 
v_add_i32 v15, v13, v12 // (op 6963) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6963) Load value

v_mov_b32 v12, 33792 // (op 6963) 
v_add_i32 v15, v13, v12 // (op 6963) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6963) Load value

v_mov_b32 v12, 34304 // (op 6963) 
v_add_i32 v15, v13, v12 // (op 6963) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6963) Load value

v_accvgpr_read v22, a32 // (op 6960) reg expression
v_accvgpr_read v23, a33 // (op 6960) reg expression
v_accvgpr_read v24, a34 // (op 6960) reg expression
v_accvgpr_read v25, a35 // (op 6960) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6961) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6961) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6961) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6961) 

v_accvgpr_write a32, v22 // (op 6962) reg expression
v_accvgpr_write a33, v23 // (op 6962) reg expression
v_accvgpr_write a34, v24 // (op 6962) reg expression
v_accvgpr_write a35, v25 // (op 6962) reg expression




v_mov_b32 v12, 32896 // (op 6968) 
v_add_i32 v15, v13, v12 // (op 6968) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6968) Load value

v_mov_b32 v12, 33408 // (op 6968) 
v_add_i32 v15, v13, v12 // (op 6968) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6968) Load value

v_mov_b32 v12, 33920 // (op 6968) 
v_add_i32 v15, v13, v12 // (op 6968) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6968) Load value

v_mov_b32 v12, 34432 // (op 6968) 
v_add_i32 v15, v13, v12 // (op 6968) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6968) Load value

v_accvgpr_read v22, a36 // (op 6965) reg expression
v_accvgpr_read v23, a37 // (op 6965) reg expression
v_accvgpr_read v24, a38 // (op 6965) reg expression
v_accvgpr_read v25, a39 // (op 6965) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6966) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6966) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6966) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6966) 

v_accvgpr_write a36, v22 // (op 6967) reg expression
v_accvgpr_write a37, v23 // (op 6967) reg expression
v_accvgpr_write a38, v24 // (op 6967) reg expression
v_accvgpr_write a39, v25 // (op 6967) reg expression




v_mov_b32 v12, 33024 // (op 6973) 
v_add_i32 v15, v13, v12 // (op 6973) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6973) Load value

v_mov_b32 v12, 33536 // (op 6973) 
v_add_i32 v15, v13, v12 // (op 6973) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6973) Load value

v_mov_b32 v12, 34048 // (op 6973) 
v_add_i32 v15, v13, v12 // (op 6973) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6973) Load value

v_mov_b32 v12, 34560 // (op 6973) 
v_add_i32 v15, v13, v12 // (op 6973) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6973) Load value

v_accvgpr_read v22, a40 // (op 6970) reg expression
v_accvgpr_read v23, a41 // (op 6970) reg expression
v_accvgpr_read v24, a42 // (op 6970) reg expression
v_accvgpr_read v25, a43 // (op 6970) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6971) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6971) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6971) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6971) 

v_accvgpr_write a40, v22 // (op 6972) reg expression
v_accvgpr_write a41, v23 // (op 6972) reg expression
v_accvgpr_write a42, v24 // (op 6972) reg expression
v_accvgpr_write a43, v25 // (op 6972) reg expression




v_mov_b32 v12, 33152 // (op 6978) 
v_add_i32 v15, v13, v12 // (op 6978) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6978) Load value

v_mov_b32 v12, 33664 // (op 6978) 
v_add_i32 v15, v13, v12 // (op 6978) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6978) Load value

v_mov_b32 v12, 34176 // (op 6978) 
v_add_i32 v15, v13, v12 // (op 6978) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6978) Load value

v_mov_b32 v12, 34688 // (op 6978) 
v_add_i32 v15, v13, v12 // (op 6978) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6978) Load value

v_accvgpr_read v22, a44 // (op 6975) reg expression
v_accvgpr_read v23, a45 // (op 6975) reg expression
v_accvgpr_read v24, a46 // (op 6975) reg expression
v_accvgpr_read v25, a47 // (op 6975) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6976) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6976) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6976) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6976) 

v_accvgpr_write a44, v22 // (op 6977) reg expression
v_accvgpr_write a45, v23 // (op 6977) reg expression
v_accvgpr_write a46, v24 // (op 6977) reg expression
v_accvgpr_write a47, v25 // (op 6977) reg expression




v_mov_b32 v12, 49152 // (op 7011) 
v_add_i32 v15, v13, v12 // (op 7011) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 7011) Load value

v_mov_b32 v12, 49664 // (op 7011) 
v_add_i32 v15, v13, v12 // (op 7011) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 7011) Load value

v_mov_b32 v12, 50176 // (op 7011) 
v_add_i32 v15, v13, v12 // (op 7011) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 7011) Load value

v_mov_b32 v12, 50688 // (op 7011) 
v_add_i32 v15, v13, v12 // (op 7011) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 7011) Load value

v_accvgpr_read v22, a48 // (op 7008) reg expression
v_accvgpr_read v23, a49 // (op 7008) reg expression
v_accvgpr_read v24, a50 // (op 7008) reg expression
v_accvgpr_read v25, a51 // (op 7008) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 7009) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 7009) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 7009) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 7009) 

v_accvgpr_write a48, v22 // (op 7010) reg expression
v_accvgpr_write a49, v23 // (op 7010) reg expression
v_accvgpr_write a50, v24 // (op 7010) reg expression
v_accvgpr_write a51, v25 // (op 7010) reg expression




v_mov_b32 v12, 49280 // (op 7016) 
v_add_i32 v15, v13, v12 // (op 7016) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 7016) Load value

v_mov_b32 v12, 49792 // (op 7016) 
v_add_i32 v15, v13, v12 // (op 7016) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 7016) Load value

v_mov_b32 v12, 50304 // (op 7016) 
v_add_i32 v15, v13, v12 // (op 7016) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 7016) Load value

v_mov_b32 v12, 50816 // (op 7016) 
v_add_i32 v15, v13, v12 // (op 7016) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 7016) Load value

v_accvgpr_read v22, a52 // (op 7013) reg expression
v_accvgpr_read v23, a53 // (op 7013) reg expression
v_accvgpr_read v24, a54 // (op 7013) reg expression
v_accvgpr_read v25, a55 // (op 7013) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 7014) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 7014) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 7014) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 7014) 

v_accvgpr_write a52, v22 // (op 7015) reg expression
v_accvgpr_write a53, v23 // (op 7015) reg expression
v_accvgpr_write a54, v24 // (op 7015) reg expression
v_accvgpr_write a55, v25 // (op 7015) reg expression




v_mov_b32 v12, 49408 // (op 7021) 
v_add_i32 v15, v13, v12 // (op 7021) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 7021) Load value

v_mov_b32 v12, 49920 // (op 7021) 
v_add_i32 v15, v13, v12 // (op 7021) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 7021) Load value

v_mov_b32 v12, 50432 // (op 7021) 
v_add_i32 v15, v13, v12 // (op 7021) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 7021) Load value

v_mov_b32 v12, 50944 // (op 7021) 
v_add_i32 v15, v13, v12 // (op 7021) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 7021) Load value

v_accvgpr_read v22, a56 // (op 7018) reg expression
v_accvgpr_read v23, a57 // (op 7018) reg expression
v_accvgpr_read v24, a58 // (op 7018) reg expression
v_accvgpr_read v25, a59 // (op 7018) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 7019) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 7019) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 7019) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 7019) 

v_accvgpr_write a56, v22 // (op 7020) reg expression
v_accvgpr_write a57, v23 // (op 7020) reg expression
v_accvgpr_write a58, v24 // (op 7020) reg expression
v_accvgpr_write a59, v25 // (op 7020) reg expression




v_mov_b32 v12, 49536 // (op 7026) 
v_add_i32 v15, v13, v12 // (op 7026) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 7026) Load value

v_mov_b32 v12, 50048 // (op 7026) 
v_add_i32 v15, v13, v12 // (op 7026) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 7026) Load value

v_mov_b32 v12, 50560 // (op 7026) 
v_add_i32 v15, v13, v12 // (op 7026) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 7026) Load value

v_mov_b32 v12, 51072 // (op 7026) 
v_add_i32 v15, v13, v12 // (op 7026) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 7026) Load value

v_accvgpr_read v22, a60 // (op 7023) reg expression
v_accvgpr_read v23, a61 // (op 7023) reg expression
v_accvgpr_read v24, a62 // (op 7023) reg expression
v_accvgpr_read v25, a63 // (op 7023) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 7024) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 7024) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 7024) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 7024) 

v_accvgpr_write a60, v22 // (op 7025) reg expression
v_accvgpr_write a61, v23 // (op 7025) reg expression
v_accvgpr_write a62, v24 // (op 7025) reg expression
v_accvgpr_write a63, v25 // (op 7025) reg expression

s_add_u32 s52, s52, 1 // (op 274) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 272) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 272) 
s_mov_b32 s76, s54 // (op 272) convert

s_sub_u32 s54, s76, s19 // (op 272) 

s_sub_u32 s55, s54, 1 // (op 272) 

s_cmp_lt_u32 s53, s55 // (op 272) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_ReceiveTileLoop_272 // (op 272) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_ReceiveTileLoop_272 if true)
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_272:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Receive_Tile_307 // (op 307) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Receive_Tile_307
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Receive_Tile_307:
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Receive_Tile_307:

s_cmp_eq_i32 s18, 0 // (op 350) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_350 // (op 350) Condition: False, jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_350
v_cmp_eq_f32 vcc, 0.00000, v0 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_116
v_accvgpr_read v18, a0 // (op 42) 
v_accvgpr_read v19, a1 // (op 42) 
v_accvgpr_read v20, a2 // (op 42) 
v_accvgpr_read v21, a3 // (op 42) 
v_mul_f32 v22, v11, v18 // (op 42) 
v_mul_f32 v23, v11, v19 // (op 42) 
v_mul_f32 v24, v11, v20 // (op 42) 
v_mul_f32 v25, v11, v21 // (op 42) 

v_lshrrev_b32 v18, 16, v22 // (op 48) 
v_lshrrev_b32 v19, 16, v23 // (op 48) 
v_lshrrev_b32 v20, 16, v24 // (op 48) 
v_lshrrev_b32 v21, 16, v25 // (op 48) 


v_pack_b32_f16 v12, v18, v19 // (op 50) 
v_pack_b32_f16 v13, v20, v21 // (op 50) 

v_mov_b32 v15, v17 // (op 50) reg expression

buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 50) Store value



v_accvgpr_read v18, a4 // (op 7108) 
v_accvgpr_read v19, a5 // (op 7108) 
v_accvgpr_read v20, a6 // (op 7108) 
v_accvgpr_read v21, a7 // (op 7108) 
v_mul_f32 v22, v11, v18 // (op 7108) 
v_mul_f32 v23, v11, v19 // (op 7108) 
v_mul_f32 v24, v11, v20 // (op 7108) 
v_mul_f32 v25, v11, v21 // (op 7108) 

v_lshrrev_b32 v18, 16, v22 // (op 7110) 
v_lshrrev_b32 v19, 16, v23 // (op 7110) 
v_lshrrev_b32 v20, 16, v24 // (op 7110) 
v_lshrrev_b32 v21, 16, v25 // (op 7110) 


v_pack_b32_f16 v12, v18, v19 // (op 7111) 
v_pack_b32_f16 v13, v20, v21 // (op 7111) 

s_lshl_b32 s52, s24, 5 // (op 7111) 
v_add_u32 v15, v17, s52 // (op 7111) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7111) Store value



v_accvgpr_read v18, a8 // (op 7122) 
v_accvgpr_read v19, a9 // (op 7122) 
v_accvgpr_read v20, a10 // (op 7122) 
v_accvgpr_read v21, a11 // (op 7122) 
v_mul_f32 v22, v11, v18 // (op 7122) 
v_mul_f32 v23, v11, v19 // (op 7122) 
v_mul_f32 v24, v11, v20 // (op 7122) 
v_mul_f32 v25, v11, v21 // (op 7122) 

v_lshrrev_b32 v18, 16, v22 // (op 7124) 
v_lshrrev_b32 v19, 16, v23 // (op 7124) 
v_lshrrev_b32 v20, 16, v24 // (op 7124) 
v_lshrrev_b32 v21, 16, v25 // (op 7124) 


v_pack_b32_f16 v12, v18, v19 // (op 7125) 
v_pack_b32_f16 v13, v20, v21 // (op 7125) 

s_lshl_b32 s52, s24, 6 // (op 7125) 
v_add_u32 v15, v17, s52 // (op 7125) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7125) Store value



v_accvgpr_read v18, a12 // (op 7136) 
v_accvgpr_read v19, a13 // (op 7136) 
v_accvgpr_read v20, a14 // (op 7136) 
v_accvgpr_read v21, a15 // (op 7136) 
v_mul_f32 v22, v11, v18 // (op 7136) 
v_mul_f32 v23, v11, v19 // (op 7136) 
v_mul_f32 v24, v11, v20 // (op 7136) 
v_mul_f32 v25, v11, v21 // (op 7136) 

v_lshrrev_b32 v18, 16, v22 // (op 7138) 
v_lshrrev_b32 v19, 16, v23 // (op 7138) 
v_lshrrev_b32 v20, 16, v24 // (op 7138) 
v_lshrrev_b32 v21, 16, v25 // (op 7138) 


v_pack_b32_f16 v12, v18, v19 // (op 7139) 
v_pack_b32_f16 v13, v20, v21 // (op 7139) 

s_lshl_b32 s52, s24, 5 // (op 7139) 
s_mul_i32 s53, 3, s52 // (op 7139) 

v_add_u32 v15, v17, s53 // (op 7139) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7139) Store value



v_accvgpr_read v18, a16 // (op 7172) 
v_accvgpr_read v19, a17 // (op 7172) 
v_accvgpr_read v20, a18 // (op 7172) 
v_accvgpr_read v21, a19 // (op 7172) 
v_mul_f32 v22, v11, v18 // (op 7172) 
v_mul_f32 v23, v11, v19 // (op 7172) 
v_mul_f32 v24, v11, v20 // (op 7172) 
v_mul_f32 v25, v11, v21 // (op 7172) 

v_lshrrev_b32 v18, 16, v22 // (op 7174) 
v_lshrrev_b32 v19, 16, v23 // (op 7174) 
v_lshrrev_b32 v20, 16, v24 // (op 7174) 
v_lshrrev_b32 v21, 16, v25 // (op 7174) 


v_pack_b32_f16 v12, v18, v19 // (op 7176) 
v_pack_b32_f16 v13, v20, v21 // (op 7176) 

v_add_u32 v15, 32, v17 // (op 7176) 

buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7176) Store value



v_accvgpr_read v18, a20 // (op 7186) 
v_accvgpr_read v19, a21 // (op 7186) 
v_accvgpr_read v20, a22 // (op 7186) 
v_accvgpr_read v21, a23 // (op 7186) 
v_mul_f32 v22, v11, v18 // (op 7186) 
v_mul_f32 v23, v11, v19 // (op 7186) 
v_mul_f32 v24, v11, v20 // (op 7186) 
v_mul_f32 v25, v11, v21 // (op 7186) 

v_lshrrev_b32 v18, 16, v22 // (op 7188) 
v_lshrrev_b32 v19, 16, v23 // (op 7188) 
v_lshrrev_b32 v20, 16, v24 // (op 7188) 
v_lshrrev_b32 v21, 16, v25 // (op 7188) 


v_pack_b32_f16 v12, v18, v19 // (op 7178) 
v_pack_b32_f16 v13, v20, v21 // (op 7178) 

s_lshl_b32 s52, s24, 5 // (op 7178) 
s_add_u32 s52, s52, 32 // (op 7178) 
v_add_u32 v15, v17, s52 // (op 7178) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7178) Store value



v_accvgpr_read v18, a24 // (op 7192) 
v_accvgpr_read v19, a25 // (op 7192) 
v_accvgpr_read v20, a26 // (op 7192) 
v_accvgpr_read v21, a27 // (op 7192) 
v_mul_f32 v22, v11, v18 // (op 7192) 
v_mul_f32 v23, v11, v19 // (op 7192) 
v_mul_f32 v24, v11, v20 // (op 7192) 
v_mul_f32 v25, v11, v21 // (op 7192) 

v_lshrrev_b32 v18, 16, v22 // (op 7194) 
v_lshrrev_b32 v19, 16, v23 // (op 7194) 
v_lshrrev_b32 v20, 16, v24 // (op 7194) 
v_lshrrev_b32 v21, 16, v25 // (op 7194) 


v_pack_b32_f16 v12, v18, v19 // (op 7180) 
v_pack_b32_f16 v13, v20, v21 // (op 7180) 

s_lshl_b32 s52, s24, 6 // (op 7180) 
s_add_u32 s52, s52, 32 // (op 7180) 
v_add_u32 v15, v17, s52 // (op 7180) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7180) Store value



v_accvgpr_read v18, a28 // (op 7198) 
v_accvgpr_read v19, a29 // (op 7198) 
v_accvgpr_read v20, a30 // (op 7198) 
v_accvgpr_read v21, a31 // (op 7198) 
v_mul_f32 v22, v11, v18 // (op 7198) 
v_mul_f32 v23, v11, v19 // (op 7198) 
v_mul_f32 v24, v11, v20 // (op 7198) 
v_mul_f32 v25, v11, v21 // (op 7198) 

v_lshrrev_b32 v18, 16, v22 // (op 7200) 
v_lshrrev_b32 v19, 16, v23 // (op 7200) 
v_lshrrev_b32 v20, 16, v24 // (op 7200) 
v_lshrrev_b32 v21, 16, v25 // (op 7200) 


v_pack_b32_f16 v12, v18, v19 // (op 7182) 
v_pack_b32_f16 v13, v20, v21 // (op 7182) 

s_lshl_b32 s52, s24, 5 // (op 7182) 
s_mul_i32 s53, 3, s52 // (op 7182) 
s_add_u32 s53, s53, 32 // (op 7182) 

v_add_u32 v15, v17, s53 // (op 7182) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7182) Store value



v_accvgpr_read v18, a32 // (op 7248) 
v_accvgpr_read v19, a33 // (op 7248) 
v_accvgpr_read v20, a34 // (op 7248) 
v_accvgpr_read v21, a35 // (op 7248) 
v_mul_f32 v22, v11, v18 // (op 7248) 
v_mul_f32 v23, v11, v19 // (op 7248) 
v_mul_f32 v24, v11, v20 // (op 7248) 
v_mul_f32 v25, v11, v21 // (op 7248) 

v_lshrrev_b32 v18, 16, v22 // (op 7250) 
v_lshrrev_b32 v19, 16, v23 // (op 7250) 
v_lshrrev_b32 v20, 16, v24 // (op 7250) 
v_lshrrev_b32 v21, 16, v25 // (op 7250) 


v_pack_b32_f16 v12, v18, v19 // (op 7252) 
v_pack_b32_f16 v13, v20, v21 // (op 7252) 

v_add_u32 v15, 64, v17 // (op 7252) 

buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7252) Store value



v_accvgpr_read v18, a36 // (op 7262) 
v_accvgpr_read v19, a37 // (op 7262) 
v_accvgpr_read v20, a38 // (op 7262) 
v_accvgpr_read v21, a39 // (op 7262) 
v_mul_f32 v22, v11, v18 // (op 7262) 
v_mul_f32 v23, v11, v19 // (op 7262) 
v_mul_f32 v24, v11, v20 // (op 7262) 
v_mul_f32 v25, v11, v21 // (op 7262) 

v_lshrrev_b32 v18, 16, v22 // (op 7264) 
v_lshrrev_b32 v19, 16, v23 // (op 7264) 
v_lshrrev_b32 v20, 16, v24 // (op 7264) 
v_lshrrev_b32 v21, 16, v25 // (op 7264) 


v_pack_b32_f16 v12, v18, v19 // (op 7254) 
v_pack_b32_f16 v13, v20, v21 // (op 7254) 

s_lshl_b32 s52, s24, 5 // (op 7254) 
s_add_u32 s52, s52, 64 // (op 7254) 
v_add_u32 v15, v17, s52 // (op 7254) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7254) Store value



v_accvgpr_read v18, a40 // (op 7268) 
v_accvgpr_read v19, a41 // (op 7268) 
v_accvgpr_read v20, a42 // (op 7268) 
v_accvgpr_read v21, a43 // (op 7268) 
v_mul_f32 v22, v11, v18 // (op 7268) 
v_mul_f32 v23, v11, v19 // (op 7268) 
v_mul_f32 v24, v11, v20 // (op 7268) 
v_mul_f32 v25, v11, v21 // (op 7268) 

v_lshrrev_b32 v18, 16, v22 // (op 7270) 
v_lshrrev_b32 v19, 16, v23 // (op 7270) 
v_lshrrev_b32 v20, 16, v24 // (op 7270) 
v_lshrrev_b32 v21, 16, v25 // (op 7270) 


v_pack_b32_f16 v12, v18, v19 // (op 7256) 
v_pack_b32_f16 v13, v20, v21 // (op 7256) 

s_lshl_b32 s52, s24, 6 // (op 7256) 
s_add_u32 s52, s52, 64 // (op 7256) 
v_add_u32 v15, v17, s52 // (op 7256) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7256) Store value



v_accvgpr_read v18, a44 // (op 7274) 
v_accvgpr_read v19, a45 // (op 7274) 
v_accvgpr_read v20, a46 // (op 7274) 
v_accvgpr_read v21, a47 // (op 7274) 
v_mul_f32 v22, v11, v18 // (op 7274) 
v_mul_f32 v23, v11, v19 // (op 7274) 
v_mul_f32 v24, v11, v20 // (op 7274) 
v_mul_f32 v25, v11, v21 // (op 7274) 

v_lshrrev_b32 v18, 16, v22 // (op 7276) 
v_lshrrev_b32 v19, 16, v23 // (op 7276) 
v_lshrrev_b32 v20, 16, v24 // (op 7276) 
v_lshrrev_b32 v21, 16, v25 // (op 7276) 


v_pack_b32_f16 v12, v18, v19 // (op 7258) 
v_pack_b32_f16 v13, v20, v21 // (op 7258) 

s_lshl_b32 s52, s24, 5 // (op 7258) 
s_mul_i32 s53, 3, s52 // (op 7258) 
s_add_u32 s53, s53, 64 // (op 7258) 

v_add_u32 v15, v17, s53 // (op 7258) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7258) Store value



v_accvgpr_read v18, a48 // (op 7324) 
v_accvgpr_read v19, a49 // (op 7324) 
v_accvgpr_read v20, a50 // (op 7324) 
v_accvgpr_read v21, a51 // (op 7324) 
v_mul_f32 v22, v11, v18 // (op 7324) 
v_mul_f32 v23, v11, v19 // (op 7324) 
v_mul_f32 v24, v11, v20 // (op 7324) 
v_mul_f32 v25, v11, v21 // (op 7324) 

v_lshrrev_b32 v18, 16, v22 // (op 7326) 
v_lshrrev_b32 v19, 16, v23 // (op 7326) 
v_lshrrev_b32 v20, 16, v24 // (op 7326) 
v_lshrrev_b32 v21, 16, v25 // (op 7326) 


v_pack_b32_f16 v12, v18, v19 // (op 7328) 
v_pack_b32_f16 v13, v20, v21 // (op 7328) 

v_add_u32 v15, 96, v17 // (op 7328) 

buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7328) Store value



v_accvgpr_read v18, a52 // (op 7338) 
v_accvgpr_read v19, a53 // (op 7338) 
v_accvgpr_read v20, a54 // (op 7338) 
v_accvgpr_read v21, a55 // (op 7338) 
v_mul_f32 v22, v11, v18 // (op 7338) 
v_mul_f32 v23, v11, v19 // (op 7338) 
v_mul_f32 v24, v11, v20 // (op 7338) 
v_mul_f32 v25, v11, v21 // (op 7338) 

v_lshrrev_b32 v18, 16, v22 // (op 7340) 
v_lshrrev_b32 v19, 16, v23 // (op 7340) 
v_lshrrev_b32 v20, 16, v24 // (op 7340) 
v_lshrrev_b32 v21, 16, v25 // (op 7340) 


v_pack_b32_f16 v12, v18, v19 // (op 7330) 
v_pack_b32_f16 v13, v20, v21 // (op 7330) 

s_lshl_b32 s52, s24, 5 // (op 7330) 
s_add_u32 s52, s52, 96 // (op 7330) 
v_add_u32 v15, v17, s52 // (op 7330) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7330) Store value



v_accvgpr_read v18, a56 // (op 7344) 
v_accvgpr_read v19, a57 // (op 7344) 
v_accvgpr_read v20, a58 // (op 7344) 
v_accvgpr_read v21, a59 // (op 7344) 
v_mul_f32 v22, v11, v18 // (op 7344) 
v_mul_f32 v23, v11, v19 // (op 7344) 
v_mul_f32 v24, v11, v20 // (op 7344) 
v_mul_f32 v25, v11, v21 // (op 7344) 

v_lshrrev_b32 v18, 16, v22 // (op 7346) 
v_lshrrev_b32 v19, 16, v23 // (op 7346) 
v_lshrrev_b32 v20, 16, v24 // (op 7346) 
v_lshrrev_b32 v21, 16, v25 // (op 7346) 


v_pack_b32_f16 v12, v18, v19 // (op 7332) 
v_pack_b32_f16 v13, v20, v21 // (op 7332) 

s_lshl_b32 s52, s24, 6 // (op 7332) 
s_add_u32 s52, s52, 96 // (op 7332) 
v_add_u32 v15, v17, s52 // (op 7332) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7332) Store value



v_accvgpr_read v18, a60 // (op 7350) 
v_accvgpr_read v19, a61 // (op 7350) 
v_accvgpr_read v20, a62 // (op 7350) 
v_accvgpr_read v21, a63 // (op 7350) 
v_mul_f32 v22, v11, v18 // (op 7350) 
v_mul_f32 v23, v11, v19 // (op 7350) 
v_mul_f32 v24, v11, v20 // (op 7350) 
v_mul_f32 v25, v11, v21 // (op 7350) 

v_lshrrev_b32 v18, 16, v22 // (op 7352) 
v_lshrrev_b32 v19, 16, v23 // (op 7352) 
v_lshrrev_b32 v20, 16, v24 // (op 7352) 
v_lshrrev_b32 v21, 16, v25 // (op 7352) 


v_pack_b32_f16 v12, v18, v19 // (op 7334) 
v_pack_b32_f16 v13, v20, v21 // (op 7334) 

s_lshl_b32 s52, s24, 5 // (op 7334) 
s_mul_i32 s53, 3, s52 // (op 7334) 
s_add_u32 s53, s53, 96 // (op 7334) 

v_add_u32 v15, v17, s53 // (op 7334) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7334) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_116
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_116:


v_add_u32 v12, 96, v14 // (op 7722) 

buffer_load_dwordx2 v[18:19], v12, s[56:59], 0 offen // (op 7722) Load value

v_accvgpr_read v20, a48 // (op 7709) 
v_accvgpr_read v21, a49 // (op 7709) 
v_accvgpr_read v22, a50 // (op 7709) 
v_accvgpr_read v23, a51 // (op 7709) 
v_mul_f32 v24, v11, v20 // (op 7709) 
v_mul_f32 v25, v11, v21 // (op 7709) 
v_mul_f32 v26, v11, v22 // (op 7709) 
v_mul_f32 v27, v11, v23 // (op 7709) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v18 // (op 7721) 
v_lshrrev_b32 v13, 16, v18 // (op 7721) 
v_lshlrev_b32 v12, 16, v12 // (op 7721) 
v_lshlrev_b32 v13, 16, v13 // (op 7721) 
v_mul_f32 v20, v0, v12 // (op 7721) 
v_mul_f32 v21, v0, v13 // (op 7721) 
v_and_b32 v12, 65535, v19 // (op 7721) 
v_lshrrev_b32 v13, 16, v19 // (op 7721) 
v_lshlrev_b32 v12, 16, v12 // (op 7721) 
v_lshlrev_b32 v13, 16, v13 // (op 7721) 
v_mul_f32 v22, v0, v12 // (op 7721) 
v_mul_f32 v23, v0, v13 // (op 7721) 


v_add_f32 v28, v20, v24 // (op 7710) 
v_add_f32 v29, v21, v25 // (op 7710) 
v_add_f32 v30, v22, v26 // (op 7710) 
v_add_f32 v31, v23, v27 // (op 7710) 


v_lshrrev_b32 v18, 16, v28 // (op 7711) 
v_lshrrev_b32 v19, 16, v29 // (op 7711) 
v_lshrrev_b32 v20, 16, v30 // (op 7711) 
v_lshrrev_b32 v21, 16, v31 // (op 7711) 


v_pack_b32_f16 v12, v18, v19 // (op 7713) 
v_pack_b32_f16 v13, v20, v21 // (op 7713) 

v_add_u32 v15, 96, v17 // (op 7713) 

buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7713) Store value





s_lshl_b32 s52, s4, 5 // (op 7727) 
s_add_u32 s52, s52, 96 // (op 7727) 
v_add_u32 v12, v14, s52 // (op 7727) 


buffer_load_dwordx2 v[18:19], v12, s[56:59], 0 offen // (op 7727) Load value

v_accvgpr_read v20, a52 // (op 7739) 
v_accvgpr_read v21, a53 // (op 7739) 
v_accvgpr_read v22, a54 // (op 7739) 
v_accvgpr_read v23, a55 // (op 7739) 
v_mul_f32 v24, v11, v20 // (op 7739) 
v_mul_f32 v25, v11, v21 // (op 7739) 
v_mul_f32 v26, v11, v22 // (op 7739) 
v_mul_f32 v27, v11, v23 // (op 7739) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v18 // (op 7724) 
v_lshrrev_b32 v13, 16, v18 // (op 7724) 
v_lshlrev_b32 v12, 16, v12 // (op 7724) 
v_lshlrev_b32 v13, 16, v13 // (op 7724) 
v_mul_f32 v20, v0, v12 // (op 7724) 
v_mul_f32 v21, v0, v13 // (op 7724) 
v_and_b32 v12, 65535, v19 // (op 7724) 
v_lshrrev_b32 v13, 16, v19 // (op 7724) 
v_lshlrev_b32 v12, 16, v12 // (op 7724) 
v_lshlrev_b32 v13, 16, v13 // (op 7724) 
v_mul_f32 v22, v0, v12 // (op 7724) 
v_mul_f32 v23, v0, v13 // (op 7724) 


v_add_f32 v28, v20, v24 // (op 7725) 
v_add_f32 v29, v21, v25 // (op 7725) 
v_add_f32 v30, v22, v26 // (op 7725) 
v_add_f32 v31, v23, v27 // (op 7725) 


v_lshrrev_b32 v18, 16, v28 // (op 7726) 
v_lshrrev_b32 v19, 16, v29 // (op 7726) 
v_lshrrev_b32 v20, 16, v30 // (op 7726) 
v_lshrrev_b32 v21, 16, v31 // (op 7726) 


v_pack_b32_f16 v12, v18, v19 // (op 7715) 
v_pack_b32_f16 v13, v20, v21 // (op 7715) 

s_lshl_b32 s52, s24, 5 // (op 7715) 
s_add_u32 s52, s52, 96 // (op 7715) 
v_add_u32 v15, v17, s52 // (op 7715) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7715) Store value





s_lshl_b32 s52, s4, 6 // (op 7732) 
s_add_u32 s52, s52, 96 // (op 7732) 
v_add_u32 v12, v14, s52 // (op 7732) 


buffer_load_dwordx2 v[18:19], v12, s[56:59], 0 offen // (op 7732) Load value

v_accvgpr_read v20, a56 // (op 7741) 
v_accvgpr_read v21, a57 // (op 7741) 
v_accvgpr_read v22, a58 // (op 7741) 
v_accvgpr_read v23, a59 // (op 7741) 
v_mul_f32 v24, v11, v20 // (op 7741) 
v_mul_f32 v25, v11, v21 // (op 7741) 
v_mul_f32 v26, v11, v22 // (op 7741) 
v_mul_f32 v27, v11, v23 // (op 7741) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v18 // (op 7729) 
v_lshrrev_b32 v13, 16, v18 // (op 7729) 
v_lshlrev_b32 v12, 16, v12 // (op 7729) 
v_lshlrev_b32 v13, 16, v13 // (op 7729) 
v_mul_f32 v20, v0, v12 // (op 7729) 
v_mul_f32 v21, v0, v13 // (op 7729) 
v_and_b32 v12, 65535, v19 // (op 7729) 
v_lshrrev_b32 v13, 16, v19 // (op 7729) 
v_lshlrev_b32 v12, 16, v12 // (op 7729) 
v_lshlrev_b32 v13, 16, v13 // (op 7729) 
v_mul_f32 v22, v0, v12 // (op 7729) 
v_mul_f32 v23, v0, v13 // (op 7729) 


v_add_f32 v28, v20, v24 // (op 7730) 
v_add_f32 v29, v21, v25 // (op 7730) 
v_add_f32 v30, v22, v26 // (op 7730) 
v_add_f32 v31, v23, v27 // (op 7730) 


v_lshrrev_b32 v18, 16, v28 // (op 7731) 
v_lshrrev_b32 v19, 16, v29 // (op 7731) 
v_lshrrev_b32 v20, 16, v30 // (op 7731) 
v_lshrrev_b32 v21, 16, v31 // (op 7731) 


v_pack_b32_f16 v12, v18, v19 // (op 7717) 
v_pack_b32_f16 v13, v20, v21 // (op 7717) 

s_lshl_b32 s52, s24, 6 // (op 7717) 
s_add_u32 s52, s52, 96 // (op 7717) 
v_add_u32 v15, v17, s52 // (op 7717) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7717) Store value





s_lshl_b32 s52, s4, 5 // (op 7737) 
s_mul_i32 s53, 3, s52 // (op 7737) 
s_add_u32 s53, s53, 96 // (op 7737) 

v_add_u32 v12, v14, s53 // (op 7737) 


buffer_load_dwordx2 v[18:19], v12, s[56:59], 0 offen // (op 7737) Load value

v_accvgpr_read v20, a60 // (op 7743) 
v_accvgpr_read v21, a61 // (op 7743) 
v_accvgpr_read v22, a62 // (op 7743) 
v_accvgpr_read v23, a63 // (op 7743) 
v_mul_f32 v24, v11, v20 // (op 7743) 
v_mul_f32 v25, v11, v21 // (op 7743) 
v_mul_f32 v26, v11, v22 // (op 7743) 
v_mul_f32 v27, v11, v23 // (op 7743) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v18 // (op 7734) 
v_lshrrev_b32 v13, 16, v18 // (op 7734) 
v_lshlrev_b32 v12, 16, v12 // (op 7734) 
v_lshlrev_b32 v13, 16, v13 // (op 7734) 
v_mul_f32 v20, v0, v12 // (op 7734) 
v_mul_f32 v21, v0, v13 // (op 7734) 
v_and_b32 v12, 65535, v19 // (op 7734) 
v_lshrrev_b32 v13, 16, v19 // (op 7734) 
v_lshlrev_b32 v12, 16, v12 // (op 7734) 
v_lshlrev_b32 v13, 16, v13 // (op 7734) 
v_mul_f32 v22, v0, v12 // (op 7734) 
v_mul_f32 v23, v0, v13 // (op 7734) 


v_add_f32 v28, v20, v24 // (op 7735) 
v_add_f32 v29, v21, v25 // (op 7735) 
v_add_f32 v30, v22, v26 // (op 7735) 
v_add_f32 v31, v23, v27 // (op 7735) 


v_lshrrev_b32 v18, 16, v28 // (op 7736) 
v_lshrrev_b32 v19, 16, v29 // (op 7736) 
v_lshrrev_b32 v20, 16, v30 // (op 7736) 
v_lshrrev_b32 v21, 16, v31 // (op 7736) 


v_pack_b32_f16 v12, v18, v19 // (op 7719) 
v_pack_b32_f16 v13, v20, v21 // (op 7719) 

s_lshl_b32 s52, s24, 5 // (op 7719) 
s_mul_i32 s53, 3, s52 // (op 7719) 
s_add_u32 s53, s53, 96 // (op 7719) 

v_add_u32 v15, v17, s53 // (op 7719) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7719) Store value





v_add_u32 v12, 64, v14 // (op 7635) 

buffer_load_dwordx2 v[18:19], v12, s[56:59], 0 offen // (op 7635) Load value

v_accvgpr_read v20, a32 // (op 7622) 
v_accvgpr_read v21, a33 // (op 7622) 
v_accvgpr_read v22, a34 // (op 7622) 
v_accvgpr_read v23, a35 // (op 7622) 
v_mul_f32 v24, v11, v20 // (op 7622) 
v_mul_f32 v25, v11, v21 // (op 7622) 
v_mul_f32 v26, v11, v22 // (op 7622) 
v_mul_f32 v27, v11, v23 // (op 7622) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v18 // (op 7634) 
v_lshrrev_b32 v13, 16, v18 // (op 7634) 
v_lshlrev_b32 v12, 16, v12 // (op 7634) 
v_lshlrev_b32 v13, 16, v13 // (op 7634) 
v_mul_f32 v20, v0, v12 // (op 7634) 
v_mul_f32 v21, v0, v13 // (op 7634) 
v_and_b32 v12, 65535, v19 // (op 7634) 
v_lshrrev_b32 v13, 16, v19 // (op 7634) 
v_lshlrev_b32 v12, 16, v12 // (op 7634) 
v_lshlrev_b32 v13, 16, v13 // (op 7634) 
v_mul_f32 v22, v0, v12 // (op 7634) 
v_mul_f32 v23, v0, v13 // (op 7634) 


v_add_f32 v28, v20, v24 // (op 7623) 
v_add_f32 v29, v21, v25 // (op 7623) 
v_add_f32 v30, v22, v26 // (op 7623) 
v_add_f32 v31, v23, v27 // (op 7623) 


v_lshrrev_b32 v18, 16, v28 // (op 7624) 
v_lshrrev_b32 v19, 16, v29 // (op 7624) 
v_lshrrev_b32 v20, 16, v30 // (op 7624) 
v_lshrrev_b32 v21, 16, v31 // (op 7624) 


v_pack_b32_f16 v12, v18, v19 // (op 7626) 
v_pack_b32_f16 v13, v20, v21 // (op 7626) 

v_add_u32 v15, 64, v17 // (op 7626) 

buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7626) Store value





s_lshl_b32 s52, s4, 5 // (op 7640) 
s_add_u32 s52, s52, 64 // (op 7640) 
v_add_u32 v12, v14, s52 // (op 7640) 


buffer_load_dwordx2 v[18:19], v12, s[56:59], 0 offen // (op 7640) Load value

v_accvgpr_read v20, a36 // (op 7652) 
v_accvgpr_read v21, a37 // (op 7652) 
v_accvgpr_read v22, a38 // (op 7652) 
v_accvgpr_read v23, a39 // (op 7652) 
v_mul_f32 v24, v11, v20 // (op 7652) 
v_mul_f32 v25, v11, v21 // (op 7652) 
v_mul_f32 v26, v11, v22 // (op 7652) 
v_mul_f32 v27, v11, v23 // (op 7652) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v18 // (op 7637) 
v_lshrrev_b32 v13, 16, v18 // (op 7637) 
v_lshlrev_b32 v12, 16, v12 // (op 7637) 
v_lshlrev_b32 v13, 16, v13 // (op 7637) 
v_mul_f32 v20, v0, v12 // (op 7637) 
v_mul_f32 v21, v0, v13 // (op 7637) 
v_and_b32 v12, 65535, v19 // (op 7637) 
v_lshrrev_b32 v13, 16, v19 // (op 7637) 
v_lshlrev_b32 v12, 16, v12 // (op 7637) 
v_lshlrev_b32 v13, 16, v13 // (op 7637) 
v_mul_f32 v22, v0, v12 // (op 7637) 
v_mul_f32 v23, v0, v13 // (op 7637) 


v_add_f32 v28, v20, v24 // (op 7638) 
v_add_f32 v29, v21, v25 // (op 7638) 
v_add_f32 v30, v22, v26 // (op 7638) 
v_add_f32 v31, v23, v27 // (op 7638) 


v_lshrrev_b32 v18, 16, v28 // (op 7639) 
v_lshrrev_b32 v19, 16, v29 // (op 7639) 
v_lshrrev_b32 v20, 16, v30 // (op 7639) 
v_lshrrev_b32 v21, 16, v31 // (op 7639) 


v_pack_b32_f16 v12, v18, v19 // (op 7628) 
v_pack_b32_f16 v13, v20, v21 // (op 7628) 

s_lshl_b32 s52, s24, 5 // (op 7628) 
s_add_u32 s52, s52, 64 // (op 7628) 
v_add_u32 v15, v17, s52 // (op 7628) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7628) Store value





s_lshl_b32 s52, s4, 6 // (op 7645) 
s_add_u32 s52, s52, 64 // (op 7645) 
v_add_u32 v12, v14, s52 // (op 7645) 


buffer_load_dwordx2 v[18:19], v12, s[56:59], 0 offen // (op 7645) Load value

v_accvgpr_read v20, a40 // (op 7654) 
v_accvgpr_read v21, a41 // (op 7654) 
v_accvgpr_read v22, a42 // (op 7654) 
v_accvgpr_read v23, a43 // (op 7654) 
v_mul_f32 v24, v11, v20 // (op 7654) 
v_mul_f32 v25, v11, v21 // (op 7654) 
v_mul_f32 v26, v11, v22 // (op 7654) 
v_mul_f32 v27, v11, v23 // (op 7654) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v18 // (op 7642) 
v_lshrrev_b32 v13, 16, v18 // (op 7642) 
v_lshlrev_b32 v12, 16, v12 // (op 7642) 
v_lshlrev_b32 v13, 16, v13 // (op 7642) 
v_mul_f32 v20, v0, v12 // (op 7642) 
v_mul_f32 v21, v0, v13 // (op 7642) 
v_and_b32 v12, 65535, v19 // (op 7642) 
v_lshrrev_b32 v13, 16, v19 // (op 7642) 
v_lshlrev_b32 v12, 16, v12 // (op 7642) 
v_lshlrev_b32 v13, 16, v13 // (op 7642) 
v_mul_f32 v22, v0, v12 // (op 7642) 
v_mul_f32 v23, v0, v13 // (op 7642) 


v_add_f32 v28, v20, v24 // (op 7643) 
v_add_f32 v29, v21, v25 // (op 7643) 
v_add_f32 v30, v22, v26 // (op 7643) 
v_add_f32 v31, v23, v27 // (op 7643) 


v_lshrrev_b32 v18, 16, v28 // (op 7644) 
v_lshrrev_b32 v19, 16, v29 // (op 7644) 
v_lshrrev_b32 v20, 16, v30 // (op 7644) 
v_lshrrev_b32 v21, 16, v31 // (op 7644) 


v_pack_b32_f16 v12, v18, v19 // (op 7630) 
v_pack_b32_f16 v13, v20, v21 // (op 7630) 

s_lshl_b32 s52, s24, 6 // (op 7630) 
s_add_u32 s52, s52, 64 // (op 7630) 
v_add_u32 v15, v17, s52 // (op 7630) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7630) Store value





s_lshl_b32 s52, s4, 5 // (op 7650) 
s_mul_i32 s53, 3, s52 // (op 7650) 
s_add_u32 s53, s53, 64 // (op 7650) 

v_add_u32 v12, v14, s53 // (op 7650) 


buffer_load_dwordx2 v[18:19], v12, s[56:59], 0 offen // (op 7650) Load value

v_accvgpr_read v20, a44 // (op 7656) 
v_accvgpr_read v21, a45 // (op 7656) 
v_accvgpr_read v22, a46 // (op 7656) 
v_accvgpr_read v23, a47 // (op 7656) 
v_mul_f32 v24, v11, v20 // (op 7656) 
v_mul_f32 v25, v11, v21 // (op 7656) 
v_mul_f32 v26, v11, v22 // (op 7656) 
v_mul_f32 v27, v11, v23 // (op 7656) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v18 // (op 7647) 
v_lshrrev_b32 v13, 16, v18 // (op 7647) 
v_lshlrev_b32 v12, 16, v12 // (op 7647) 
v_lshlrev_b32 v13, 16, v13 // (op 7647) 
v_mul_f32 v20, v0, v12 // (op 7647) 
v_mul_f32 v21, v0, v13 // (op 7647) 
v_and_b32 v12, 65535, v19 // (op 7647) 
v_lshrrev_b32 v13, 16, v19 // (op 7647) 
v_lshlrev_b32 v12, 16, v12 // (op 7647) 
v_lshlrev_b32 v13, 16, v13 // (op 7647) 
v_mul_f32 v22, v0, v12 // (op 7647) 
v_mul_f32 v23, v0, v13 // (op 7647) 


v_add_f32 v28, v20, v24 // (op 7648) 
v_add_f32 v29, v21, v25 // (op 7648) 
v_add_f32 v30, v22, v26 // (op 7648) 
v_add_f32 v31, v23, v27 // (op 7648) 


v_lshrrev_b32 v18, 16, v28 // (op 7649) 
v_lshrrev_b32 v19, 16, v29 // (op 7649) 
v_lshrrev_b32 v20, 16, v30 // (op 7649) 
v_lshrrev_b32 v21, 16, v31 // (op 7649) 


v_pack_b32_f16 v12, v18, v19 // (op 7632) 
v_pack_b32_f16 v13, v20, v21 // (op 7632) 

s_lshl_b32 s52, s24, 5 // (op 7632) 
s_mul_i32 s53, 3, s52 // (op 7632) 
s_add_u32 s53, s53, 64 // (op 7632) 

v_add_u32 v15, v17, s53 // (op 7632) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7632) Store value





v_add_u32 v12, 32, v14 // (op 7548) 

buffer_load_dwordx2 v[18:19], v12, s[56:59], 0 offen // (op 7548) Load value

v_accvgpr_read v20, a16 // (op 7535) 
v_accvgpr_read v21, a17 // (op 7535) 
v_accvgpr_read v22, a18 // (op 7535) 
v_accvgpr_read v23, a19 // (op 7535) 
v_mul_f32 v24, v11, v20 // (op 7535) 
v_mul_f32 v25, v11, v21 // (op 7535) 
v_mul_f32 v26, v11, v22 // (op 7535) 
v_mul_f32 v27, v11, v23 // (op 7535) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v18 // (op 7547) 
v_lshrrev_b32 v13, 16, v18 // (op 7547) 
v_lshlrev_b32 v12, 16, v12 // (op 7547) 
v_lshlrev_b32 v13, 16, v13 // (op 7547) 
v_mul_f32 v20, v0, v12 // (op 7547) 
v_mul_f32 v21, v0, v13 // (op 7547) 
v_and_b32 v12, 65535, v19 // (op 7547) 
v_lshrrev_b32 v13, 16, v19 // (op 7547) 
v_lshlrev_b32 v12, 16, v12 // (op 7547) 
v_lshlrev_b32 v13, 16, v13 // (op 7547) 
v_mul_f32 v22, v0, v12 // (op 7547) 
v_mul_f32 v23, v0, v13 // (op 7547) 


v_add_f32 v28, v20, v24 // (op 7536) 
v_add_f32 v29, v21, v25 // (op 7536) 
v_add_f32 v30, v22, v26 // (op 7536) 
v_add_f32 v31, v23, v27 // (op 7536) 


v_lshrrev_b32 v18, 16, v28 // (op 7537) 
v_lshrrev_b32 v19, 16, v29 // (op 7537) 
v_lshrrev_b32 v20, 16, v30 // (op 7537) 
v_lshrrev_b32 v21, 16, v31 // (op 7537) 


v_pack_b32_f16 v12, v18, v19 // (op 7539) 
v_pack_b32_f16 v13, v20, v21 // (op 7539) 

v_add_u32 v15, 32, v17 // (op 7539) 

buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7539) Store value





s_lshl_b32 s52, s4, 5 // (op 7553) 
s_add_u32 s52, s52, 32 // (op 7553) 
v_add_u32 v12, v14, s52 // (op 7553) 


buffer_load_dwordx2 v[18:19], v12, s[56:59], 0 offen // (op 7553) Load value

v_accvgpr_read v20, a20 // (op 7565) 
v_accvgpr_read v21, a21 // (op 7565) 
v_accvgpr_read v22, a22 // (op 7565) 
v_accvgpr_read v23, a23 // (op 7565) 
v_mul_f32 v24, v11, v20 // (op 7565) 
v_mul_f32 v25, v11, v21 // (op 7565) 
v_mul_f32 v26, v11, v22 // (op 7565) 
v_mul_f32 v27, v11, v23 // (op 7565) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v18 // (op 7550) 
v_lshrrev_b32 v13, 16, v18 // (op 7550) 
v_lshlrev_b32 v12, 16, v12 // (op 7550) 
v_lshlrev_b32 v13, 16, v13 // (op 7550) 
v_mul_f32 v20, v0, v12 // (op 7550) 
v_mul_f32 v21, v0, v13 // (op 7550) 
v_and_b32 v12, 65535, v19 // (op 7550) 
v_lshrrev_b32 v13, 16, v19 // (op 7550) 
v_lshlrev_b32 v12, 16, v12 // (op 7550) 
v_lshlrev_b32 v13, 16, v13 // (op 7550) 
v_mul_f32 v22, v0, v12 // (op 7550) 
v_mul_f32 v23, v0, v13 // (op 7550) 


v_add_f32 v28, v20, v24 // (op 7551) 
v_add_f32 v29, v21, v25 // (op 7551) 
v_add_f32 v30, v22, v26 // (op 7551) 
v_add_f32 v31, v23, v27 // (op 7551) 


v_lshrrev_b32 v18, 16, v28 // (op 7552) 
v_lshrrev_b32 v19, 16, v29 // (op 7552) 
v_lshrrev_b32 v20, 16, v30 // (op 7552) 
v_lshrrev_b32 v21, 16, v31 // (op 7552) 


v_pack_b32_f16 v12, v18, v19 // (op 7541) 
v_pack_b32_f16 v13, v20, v21 // (op 7541) 

s_lshl_b32 s52, s24, 5 // (op 7541) 
s_add_u32 s52, s52, 32 // (op 7541) 
v_add_u32 v15, v17, s52 // (op 7541) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7541) Store value





s_lshl_b32 s52, s4, 6 // (op 7558) 
s_add_u32 s52, s52, 32 // (op 7558) 
v_add_u32 v12, v14, s52 // (op 7558) 


buffer_load_dwordx2 v[18:19], v12, s[56:59], 0 offen // (op 7558) Load value

v_accvgpr_read v20, a24 // (op 7567) 
v_accvgpr_read v21, a25 // (op 7567) 
v_accvgpr_read v22, a26 // (op 7567) 
v_accvgpr_read v23, a27 // (op 7567) 
v_mul_f32 v24, v11, v20 // (op 7567) 
v_mul_f32 v25, v11, v21 // (op 7567) 
v_mul_f32 v26, v11, v22 // (op 7567) 
v_mul_f32 v27, v11, v23 // (op 7567) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v18 // (op 7555) 
v_lshrrev_b32 v13, 16, v18 // (op 7555) 
v_lshlrev_b32 v12, 16, v12 // (op 7555) 
v_lshlrev_b32 v13, 16, v13 // (op 7555) 
v_mul_f32 v20, v0, v12 // (op 7555) 
v_mul_f32 v21, v0, v13 // (op 7555) 
v_and_b32 v12, 65535, v19 // (op 7555) 
v_lshrrev_b32 v13, 16, v19 // (op 7555) 
v_lshlrev_b32 v12, 16, v12 // (op 7555) 
v_lshlrev_b32 v13, 16, v13 // (op 7555) 
v_mul_f32 v22, v0, v12 // (op 7555) 
v_mul_f32 v23, v0, v13 // (op 7555) 


v_add_f32 v28, v20, v24 // (op 7556) 
v_add_f32 v29, v21, v25 // (op 7556) 
v_add_f32 v30, v22, v26 // (op 7556) 
v_add_f32 v31, v23, v27 // (op 7556) 


v_lshrrev_b32 v18, 16, v28 // (op 7557) 
v_lshrrev_b32 v19, 16, v29 // (op 7557) 
v_lshrrev_b32 v20, 16, v30 // (op 7557) 
v_lshrrev_b32 v21, 16, v31 // (op 7557) 


v_pack_b32_f16 v12, v18, v19 // (op 7543) 
v_pack_b32_f16 v13, v20, v21 // (op 7543) 

s_lshl_b32 s52, s24, 6 // (op 7543) 
s_add_u32 s52, s52, 32 // (op 7543) 
v_add_u32 v15, v17, s52 // (op 7543) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7543) Store value





s_lshl_b32 s52, s4, 5 // (op 7563) 
s_mul_i32 s53, 3, s52 // (op 7563) 
s_add_u32 s53, s53, 32 // (op 7563) 

v_add_u32 v12, v14, s53 // (op 7563) 


buffer_load_dwordx2 v[18:19], v12, s[56:59], 0 offen // (op 7563) Load value

v_accvgpr_read v20, a28 // (op 7569) 
v_accvgpr_read v21, a29 // (op 7569) 
v_accvgpr_read v22, a30 // (op 7569) 
v_accvgpr_read v23, a31 // (op 7569) 
v_mul_f32 v24, v11, v20 // (op 7569) 
v_mul_f32 v25, v11, v21 // (op 7569) 
v_mul_f32 v26, v11, v22 // (op 7569) 
v_mul_f32 v27, v11, v23 // (op 7569) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v18 // (op 7560) 
v_lshrrev_b32 v13, 16, v18 // (op 7560) 
v_lshlrev_b32 v12, 16, v12 // (op 7560) 
v_lshlrev_b32 v13, 16, v13 // (op 7560) 
v_mul_f32 v20, v0, v12 // (op 7560) 
v_mul_f32 v21, v0, v13 // (op 7560) 
v_and_b32 v12, 65535, v19 // (op 7560) 
v_lshrrev_b32 v13, 16, v19 // (op 7560) 
v_lshlrev_b32 v12, 16, v12 // (op 7560) 
v_lshlrev_b32 v13, 16, v13 // (op 7560) 
v_mul_f32 v22, v0, v12 // (op 7560) 
v_mul_f32 v23, v0, v13 // (op 7560) 


v_add_f32 v28, v20, v24 // (op 7561) 
v_add_f32 v29, v21, v25 // (op 7561) 
v_add_f32 v30, v22, v26 // (op 7561) 
v_add_f32 v31, v23, v27 // (op 7561) 


v_lshrrev_b32 v18, 16, v28 // (op 7562) 
v_lshrrev_b32 v19, 16, v29 // (op 7562) 
v_lshrrev_b32 v20, 16, v30 // (op 7562) 
v_lshrrev_b32 v21, 16, v31 // (op 7562) 


v_pack_b32_f16 v12, v18, v19 // (op 7545) 
v_pack_b32_f16 v13, v20, v21 // (op 7545) 

s_lshl_b32 s52, s24, 5 // (op 7545) 
s_mul_i32 s53, 3, s52 // (op 7545) 
s_add_u32 s53, s53, 32 // (op 7545) 

v_add_u32 v15, v17, s53 // (op 7545) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7545) Store value





v_mov_b32 v12, v14 // (op 129) reg expression

buffer_load_dwordx2 v[18:19], v12, s[56:59], 0 offen // (op 129) Load value

v_accvgpr_read v20, a0 // (op 125) 
v_accvgpr_read v21, a1 // (op 125) 
v_accvgpr_read v22, a2 // (op 125) 
v_accvgpr_read v23, a3 // (op 125) 
v_mul_f32 v24, v11, v20 // (op 125) 
v_mul_f32 v25, v11, v21 // (op 125) 
v_mul_f32 v26, v11, v22 // (op 125) 
v_mul_f32 v27, v11, v23 // (op 125) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v18 // (op 130) 
v_lshrrev_b32 v13, 16, v18 // (op 130) 
v_lshlrev_b32 v12, 16, v12 // (op 130) 
v_lshlrev_b32 v13, 16, v13 // (op 130) 
v_mul_f32 v20, v0, v12 // (op 130) 
v_mul_f32 v21, v0, v13 // (op 130) 
v_and_b32 v12, 65535, v19 // (op 130) 
v_lshrrev_b32 v13, 16, v19 // (op 130) 
v_lshlrev_b32 v12, 16, v12 // (op 130) 
v_lshlrev_b32 v13, 16, v13 // (op 130) 
v_mul_f32 v22, v0, v12 // (op 130) 
v_mul_f32 v23, v0, v13 // (op 130) 


v_add_f32 v28, v20, v24 // (op 126) 
v_add_f32 v29, v21, v25 // (op 126) 
v_add_f32 v30, v22, v26 // (op 126) 
v_add_f32 v31, v23, v27 // (op 126) 


v_lshrrev_b32 v18, 16, v28 // (op 127) 
v_lshrrev_b32 v19, 16, v29 // (op 127) 
v_lshrrev_b32 v20, 16, v30 // (op 127) 
v_lshrrev_b32 v21, 16, v31 // (op 127) 


v_pack_b32_f16 v12, v18, v19 // (op 128) 
v_pack_b32_f16 v13, v20, v21 // (op 128) 

v_mov_b32 v15, v17 // (op 128) reg expression

buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 128) Store value





s_lshl_b32 s52, s4, 5 // (op 7456) 
v_add_u32 v12, v14, s52 // (op 7456) 


buffer_load_dwordx2 v[18:19], v12, s[56:59], 0 offen // (op 7456) Load value

v_accvgpr_read v20, a4 // (op 7452) 
v_accvgpr_read v21, a5 // (op 7452) 
v_accvgpr_read v22, a6 // (op 7452) 
v_accvgpr_read v23, a7 // (op 7452) 
v_mul_f32 v24, v11, v20 // (op 7452) 
v_mul_f32 v25, v11, v21 // (op 7452) 
v_mul_f32 v26, v11, v22 // (op 7452) 
v_mul_f32 v27, v11, v23 // (op 7452) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v18 // (op 7457) 
v_lshrrev_b32 v13, 16, v18 // (op 7457) 
v_lshlrev_b32 v12, 16, v12 // (op 7457) 
v_lshlrev_b32 v13, 16, v13 // (op 7457) 
v_mul_f32 v20, v0, v12 // (op 7457) 
v_mul_f32 v21, v0, v13 // (op 7457) 
v_and_b32 v12, 65535, v19 // (op 7457) 
v_lshrrev_b32 v13, 16, v19 // (op 7457) 
v_lshlrev_b32 v12, 16, v12 // (op 7457) 
v_lshlrev_b32 v13, 16, v13 // (op 7457) 
v_mul_f32 v22, v0, v12 // (op 7457) 
v_mul_f32 v23, v0, v13 // (op 7457) 


v_add_f32 v28, v20, v24 // (op 7453) 
v_add_f32 v29, v21, v25 // (op 7453) 
v_add_f32 v30, v22, v26 // (op 7453) 
v_add_f32 v31, v23, v27 // (op 7453) 


v_lshrrev_b32 v18, 16, v28 // (op 7454) 
v_lshrrev_b32 v19, 16, v29 // (op 7454) 
v_lshrrev_b32 v20, 16, v30 // (op 7454) 
v_lshrrev_b32 v21, 16, v31 // (op 7454) 


v_pack_b32_f16 v12, v18, v19 // (op 7455) 
v_pack_b32_f16 v13, v20, v21 // (op 7455) 

s_lshl_b32 s52, s24, 5 // (op 7455) 
v_add_u32 v15, v17, s52 // (op 7455) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7455) Store value





s_lshl_b32 s52, s4, 6 // (op 7470) 
v_add_u32 v12, v14, s52 // (op 7470) 


buffer_load_dwordx2 v[18:19], v12, s[56:59], 0 offen // (op 7470) Load value

v_accvgpr_read v20, a8 // (op 7466) 
v_accvgpr_read v21, a9 // (op 7466) 
v_accvgpr_read v22, a10 // (op 7466) 
v_accvgpr_read v23, a11 // (op 7466) 
v_mul_f32 v24, v11, v20 // (op 7466) 
v_mul_f32 v25, v11, v21 // (op 7466) 
v_mul_f32 v26, v11, v22 // (op 7466) 
v_mul_f32 v27, v11, v23 // (op 7466) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v18 // (op 7471) 
v_lshrrev_b32 v13, 16, v18 // (op 7471) 
v_lshlrev_b32 v12, 16, v12 // (op 7471) 
v_lshlrev_b32 v13, 16, v13 // (op 7471) 
v_mul_f32 v20, v0, v12 // (op 7471) 
v_mul_f32 v21, v0, v13 // (op 7471) 
v_and_b32 v12, 65535, v19 // (op 7471) 
v_lshrrev_b32 v13, 16, v19 // (op 7471) 
v_lshlrev_b32 v12, 16, v12 // (op 7471) 
v_lshlrev_b32 v13, 16, v13 // (op 7471) 
v_mul_f32 v22, v0, v12 // (op 7471) 
v_mul_f32 v23, v0, v13 // (op 7471) 


v_add_f32 v28, v20, v24 // (op 7467) 
v_add_f32 v29, v21, v25 // (op 7467) 
v_add_f32 v30, v22, v26 // (op 7467) 
v_add_f32 v31, v23, v27 // (op 7467) 


v_lshrrev_b32 v18, 16, v28 // (op 7468) 
v_lshrrev_b32 v19, 16, v29 // (op 7468) 
v_lshrrev_b32 v20, 16, v30 // (op 7468) 
v_lshrrev_b32 v21, 16, v31 // (op 7468) 


v_pack_b32_f16 v12, v18, v19 // (op 7469) 
v_pack_b32_f16 v13, v20, v21 // (op 7469) 

s_lshl_b32 s52, s24, 6 // (op 7469) 
v_add_u32 v15, v17, s52 // (op 7469) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7469) Store value





s_lshl_b32 s52, s4, 5 // (op 7484) 
s_mul_i32 s53, 3, s52 // (op 7484) 

v_add_u32 v12, v14, s53 // (op 7484) 


buffer_load_dwordx2 v[18:19], v12, s[56:59], 0 offen // (op 7484) Load value

v_accvgpr_read v20, a12 // (op 7480) 
v_accvgpr_read v21, a13 // (op 7480) 
v_accvgpr_read v22, a14 // (op 7480) 
v_accvgpr_read v23, a15 // (op 7480) 
v_mul_f32 v24, v11, v20 // (op 7480) 
v_mul_f32 v25, v11, v21 // (op 7480) 
v_mul_f32 v26, v11, v22 // (op 7480) 
v_mul_f32 v27, v11, v23 // (op 7480) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v18 // (op 7485) 
v_lshrrev_b32 v13, 16, v18 // (op 7485) 
v_lshlrev_b32 v12, 16, v12 // (op 7485) 
v_lshlrev_b32 v13, 16, v13 // (op 7485) 
v_mul_f32 v20, v0, v12 // (op 7485) 
v_mul_f32 v21, v0, v13 // (op 7485) 
v_and_b32 v12, 65535, v19 // (op 7485) 
v_lshrrev_b32 v13, 16, v19 // (op 7485) 
v_lshlrev_b32 v12, 16, v12 // (op 7485) 
v_lshlrev_b32 v13, 16, v13 // (op 7485) 
v_mul_f32 v22, v0, v12 // (op 7485) 
v_mul_f32 v23, v0, v13 // (op 7485) 


v_add_f32 v28, v20, v24 // (op 7481) 
v_add_f32 v29, v21, v25 // (op 7481) 
v_add_f32 v30, v22, v26 // (op 7481) 
v_add_f32 v31, v23, v27 // (op 7481) 


v_lshrrev_b32 v18, 16, v28 // (op 7482) 
v_lshrrev_b32 v19, 16, v29 // (op 7482) 
v_lshrrev_b32 v20, 16, v30 // (op 7482) 
v_lshrrev_b32 v21, 16, v31 // (op 7482) 


v_pack_b32_f16 v12, v18, v19 // (op 7483) 
v_pack_b32_f16 v13, v20, v21 // (op 7483) 

s_lshl_b32 s52, s24, 5 // (op 7483) 
s_mul_i32 s53, 3, s52 // (op 7483) 

v_add_u32 v15, v17, s53 // (op 7483) 


buffer_store_dwordx2 v[12:13], v15, s[60:63], 0 offen // (op 7483) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_116:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_350 // (op 350) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_350
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_350:
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_350:


s_mul_i32 s52, s8, s16 // (op 263) 
s_cmp_gt_u32 s9, s52 // (op 263) 
s_mov_b32 s53, scc // (op 263) 
s_sub_u32 s54, s9, s52 // (op 263) 

s_cmp_eq_i32 s53, 0 // (op 263) 
s_cselect_b32 s52, 0, s54 // (op 263) 


s_cmp_lt_u32 s8, s52 // (op 263) 
s_mov_b32 s53, scc // (op 263) 
s_cmp_eq_i32 s53, 0 // (op 263) 
s_cselect_b32 s54, s52, s8 // (op 263) 


s_cmp_lt_u32 s0, s54 // (op 263) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_SKStreamTileLoop_263 // (op 263) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_SKStreamTileLoop_263 if true)
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_263:













s_mov_b32 s0, 0 // (op 404) call()
s_mov_b32 s1, 0 // (op 405) call()
s_mov_b32 s0, 0 // (op 419) call()
s_mul_i32 s15, s50, s16 // (op 418) 
s_cmp_gt_u32 s51, s15 // (op 418) 
s_mov_b32 s6, scc // (op 418) 
s_sub_u32 s7, s51, s15 // (op 418) 

s_cmp_eq_i32 s6, 0 // (op 418) 
s_cselect_b32 s8, 0, s7 // (op 418) 


s_cmp_lt_u32 s50, s8 // (op 418) 
s_mov_b32 s15, scc // (op 418) 
s_cmp_eq_i32 s15, 0 // (op 418) 
s_cselect_b32 s6, s8, s50 // (op 418) 


s_cmp_lt_u32 s0, s6 // (op 418) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_418 // (op 418) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_418 if false)
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_DPStreamTileLoop_418:
v_lshrrev_b32 v3, 7, v1 // (op 14496) 
s_mul_i32 s15, s16, s50 // (op 14496) 
s_add_u32 s15, s15, s0 // (op 14496) 
v_and_b32 v4, 63, v1 // (op 14496) 
s_add_u32 s6, s15, s9 // (op 14496) 

v_lshrrev_b32 v9, 4, v4 // (op 14496) 
s_mul_hi_u32 s15, s6, s13 // (op 14496) 
v_lshlrev_b32 v10, 2, v9 // (op 14496) 

s_sub_u32 s7, s6, s15 // (op 14496) 
v_lshl_add_u32 v9, v3, 6, v10 // (op 14496) 


s_lshr_b32 s8, s7, 1 // (op 14496) 

v_lshrrev_b32 v3, 2, v4 // (op 14496) 

s_add_u32 s7, s8, s15 // (op 14496) 


v_and_b32 v4, 3, v3 // (op 14496) 

s_lshr_b32 s8, s7, s11 // (op 14496) 

v_and_b32 v3, 3, v1 // (op 14496) 
s_and_b32 s7, s11, 2147483648 // (op 14496) 
v_lshl_add_u32 v10, v4, 2, v3 // (op 14496) 


s_cmp_eq_i32 s7, 0 // (op 14496) 
s_mov_b32 s15, scc // (op 14496) 

v_lshrrev_b32 v3, 6, v1 // (op 14496) 
s_cmp_eq_i32 s15, 0 // (op 14496) 
s_cselect_b32 s7, s6, s8 // (op 14496) 



v_and_b32 v4, 1, v3 // (op 14496) 

s_mul_hi_u32 s6, s7, s12 // (op 14496) 
v_lshl_add_u32 v3, v4, 6, v10 // (op 14496) 


s_sub_u32 s8, s7, s6 // (op 14496) 
s_lshr_b32 s15, s8, 1 // (op 14496) 

s_add_u32 s8, s15, s6 // (op 14496) 


s_lshr_b32 s6, s8, s10 // (op 14496) 

s_and_b32 s8, s10, 2147483648 // (op 14496) 
s_cmp_eq_i32 s8, 0 // (op 14496) 
s_mov_b32 s15, scc // (op 14496) 

s_cmp_eq_i32 s15, 0 // (op 14496) 
s_cselect_b32 s8, s7, s6 // (op 14496) 


s_mul_i32 s6, s8, s14 // (op 14496) 
v_lshl_add_u32 v4, s8, 7, v9 // (op 14496) 


s_sub_u32 s8, s7, s6 // (op 14496) 


v_lshl_add_u32 v9, s8, 7, v3 // (op 14496) 


v_mul_lo_u32 v3, v9, s4 // (op 14496) 

v_add_lshl_u32 v9, v4, v3, 1 // (op 14496) 


s_mul_i32 s19, s4, s23 // (op 14498) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 14498) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 14498) least significant
s_mul_hi_u32 s6, s4, s22 // (op 14498) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 14498) most significant: sum
s_add_u32 s19, s19, s6 // (op 14498) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 14498) 

s_mov_b32 s56, s20 // (op 14498) reg expression
s_mov_b32 s57, s21 // (op 14498) reg expression
s_mov_b32 s58, s28 // (op 14498) reg expression
s_mov_b32 s59, 131072 // (op 14498) call()

v_lshrrev_b32 v10, 7, v1 // (op 14698) 
s_mul_i32 s15, s16, s50 // (op 14698) 
s_add_u32 s15, s15, s0 // (op 14698) 
v_and_b32 v3, 63, v1 // (op 14698) 
s_add_u32 s6, s15, s9 // (op 14698) 

v_lshrrev_b32 v4, 4, v3 // (op 14698) 
s_mul_hi_u32 s15, s6, s13 // (op 14698) 
v_lshlrev_b32 v12, 2, v4 // (op 14698) 

s_sub_u32 s7, s6, s15 // (op 14698) 
v_lshl_add_u32 v4, v10, 6, v12 // (op 14698) 


s_lshr_b32 s8, s7, 1 // (op 14698) 

v_lshrrev_b32 v10, 2, v3 // (op 14698) 

s_add_u32 s7, s8, s15 // (op 14698) 


v_and_b32 v3, 3, v10 // (op 14698) 

s_lshr_b32 s8, s7, s11 // (op 14698) 

v_and_b32 v10, 3, v1 // (op 14698) 
s_and_b32 s7, s11, 2147483648 // (op 14698) 
v_lshl_add_u32 v12, v3, 2, v10 // (op 14698) 


s_cmp_eq_i32 s7, 0 // (op 14698) 
s_mov_b32 s15, scc // (op 14698) 

v_lshrrev_b32 v3, 6, v1 // (op 14698) 
s_cmp_eq_i32 s15, 0 // (op 14698) 
s_cselect_b32 s7, s6, s8 // (op 14698) 



v_and_b32 v10, 1, v3 // (op 14698) 

s_mul_hi_u32 s6, s7, s12 // (op 14698) 
v_lshl_add_u32 v3, v10, 6, v12 // (op 14698) 


s_sub_u32 s8, s7, s6 // (op 14698) 
s_lshr_b32 s15, s8, 1 // (op 14698) 

s_add_u32 s8, s15, s6 // (op 14698) 


s_lshr_b32 s6, s8, s10 // (op 14698) 

s_and_b32 s8, s10, 2147483648 // (op 14698) 
s_cmp_eq_i32 s8, 0 // (op 14698) 
s_mov_b32 s15, scc // (op 14698) 

s_cmp_eq_i32 s15, 0 // (op 14698) 
s_cselect_b32 s8, s7, s6 // (op 14698) 


s_mul_i32 s6, s8, s14 // (op 14698) 
v_lshl_add_u32 v10, s8, 7, v4 // (op 14698) 


s_sub_u32 s8, s7, s6 // (op 14698) 


v_lshl_add_u32 v4, s8, 7, v3 // (op 14698) 


v_mul_lo_u32 v3, v4, s24 // (op 14698) 

v_add_lshl_u32 v4, v10, v3, 1 // (op 14698) 


s_mul_i32 s19, s24, s23 // (op 14700) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 14700) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 14700) least significant
s_mul_hi_u32 s6, s24, s22 // (op 14700) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 14700) most significant: sum
s_add_u32 s19, s19, s6 // (op 14700) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 14700) 

s_mov_b32 s60, s26 // (op 14700) reg expression
s_mov_b32 s61, s27 // (op 14700) reg expression
s_mov_b32 s62, s28 // (op 14700) reg expression
s_mov_b32 s63, 131072 // (op 14700) call()

v_lshrrev_b32 v3, 7, v1 // (op 14777) 
v_and_b32 v10, 15, v1 // (op 14777) 
v_lshl_add_u32 v12, v3, 6, v10 // (op 14777) 


v_and_b32 v3, 48, v1 // (op 14777) 
v_lshl_add_u32 v10, v12, 7, v3 // (op 14777) 


v_lshrrev_b32 v3, 6, v1 // (op 14852) 
v_and_b32 v12, 1, v3 // (op 14852) 

v_and_b32 v3, 15, v1 // (op 14852) 
v_lshl_add_u32 v13, v12, 6, v3 // (op 14852) 


v_and_b32 v3, 63, v1 // (op 14852) 
v_lshrrev_b32 v12, 4, v3 // (op 14852) 

v_lshlrev_b32 v3, 5, v12 // (op 14852) 

v_lshl_add_u32 v12, v13, 7, v3 // (op 14852) 


v_lshlrev_b32 v3, 2, v12 // (op 14852) 

v_lshrrev_b32 v12, 3, v3 // (op 14852) 

v_accvgpr_write a0, 0.00000 // (op 165) call()
v_accvgpr_write a1, 0.00000 // (op 165) call()
v_accvgpr_write a2, 0.00000 // (op 165) call()
v_accvgpr_write a3, 0.00000 // (op 165) call()
v_accvgpr_write a4, 0.00000 // (op 551) call()
v_accvgpr_write a5, 0.00000 // (op 551) call()
v_accvgpr_write a6, 0.00000 // (op 551) call()
v_accvgpr_write a7, 0.00000 // (op 551) call()
v_accvgpr_write a8, 0.00000 // (op 687) call()
v_accvgpr_write a9, 0.00000 // (op 687) call()
v_accvgpr_write a10, 0.00000 // (op 687) call()
v_accvgpr_write a11, 0.00000 // (op 687) call()
v_accvgpr_write a12, 0.00000 // (op 823) call()
v_accvgpr_write a13, 0.00000 // (op 823) call()
v_accvgpr_write a14, 0.00000 // (op 823) call()
v_accvgpr_write a15, 0.00000 // (op 823) call()
v_accvgpr_write a16, 0.00000 // (op 1047) call()
v_accvgpr_write a17, 0.00000 // (op 1047) call()
v_accvgpr_write a18, 0.00000 // (op 1047) call()
v_accvgpr_write a19, 0.00000 // (op 1047) call()
v_accvgpr_write a20, 0.00000 // (op 1308) call()
v_accvgpr_write a21, 0.00000 // (op 1308) call()
v_accvgpr_write a22, 0.00000 // (op 1308) call()
v_accvgpr_write a23, 0.00000 // (op 1308) call()
v_accvgpr_write a24, 0.00000 // (op 1309) call()
v_accvgpr_write a25, 0.00000 // (op 1309) call()
v_accvgpr_write a26, 0.00000 // (op 1309) call()
v_accvgpr_write a27, 0.00000 // (op 1309) call()
v_accvgpr_write a28, 0.00000 // (op 1310) call()
v_accvgpr_write a29, 0.00000 // (op 1310) call()
v_accvgpr_write a30, 0.00000 // (op 1310) call()
v_accvgpr_write a31, 0.00000 // (op 1310) call()
v_accvgpr_write a32, 0.00000 // (op 1657) call()
v_accvgpr_write a33, 0.00000 // (op 1657) call()
v_accvgpr_write a34, 0.00000 // (op 1657) call()
v_accvgpr_write a35, 0.00000 // (op 1657) call()
v_accvgpr_write a36, 0.00000 // (op 1918) call()
v_accvgpr_write a37, 0.00000 // (op 1918) call()
v_accvgpr_write a38, 0.00000 // (op 1918) call()
v_accvgpr_write a39, 0.00000 // (op 1918) call()
v_accvgpr_write a40, 0.00000 // (op 1919) call()
v_accvgpr_write a41, 0.00000 // (op 1919) call()
v_accvgpr_write a42, 0.00000 // (op 1919) call()
v_accvgpr_write a43, 0.00000 // (op 1919) call()
v_accvgpr_write a44, 0.00000 // (op 1920) call()
v_accvgpr_write a45, 0.00000 // (op 1920) call()
v_accvgpr_write a46, 0.00000 // (op 1920) call()
v_accvgpr_write a47, 0.00000 // (op 1920) call()
v_accvgpr_write a48, 0.00000 // (op 2267) call()
v_accvgpr_write a49, 0.00000 // (op 2267) call()
v_accvgpr_write a50, 0.00000 // (op 2267) call()
v_accvgpr_write a51, 0.00000 // (op 2267) call()
v_accvgpr_write a52, 0.00000 // (op 2528) call()
v_accvgpr_write a53, 0.00000 // (op 2528) call()
v_accvgpr_write a54, 0.00000 // (op 2528) call()
v_accvgpr_write a55, 0.00000 // (op 2528) call()
v_accvgpr_write a56, 0.00000 // (op 2529) call()
v_accvgpr_write a57, 0.00000 // (op 2529) call()
v_accvgpr_write a58, 0.00000 // (op 2529) call()
v_accvgpr_write a59, 0.00000 // (op 2529) call()
v_accvgpr_write a60, 0.00000 // (op 2530) call()
v_accvgpr_write a61, 0.00000 // (op 2530) call()
v_accvgpr_write a62, 0.00000 // (op 2530) call()
v_accvgpr_write a63, 0.00000 // (op 2530) call()
v_lshrrev_b32 v3, 3, v1 // (op 14232) 
s_mul_i32 s15, s16, s50 // (op 14232) 
s_add_u32 s15, s15, s0 // (op 14232) 
v_and_b32 v13, -8, v1 // (op 14232) 
s_add_u32 s6, s15, s9 // (op 14232) 

v_sub_u32 v14, v1, v13 // (op 14232) 

s_mul_hi_u32 s15, s6, s13 // (op 14232) 
v_lshlrev_b32 v13, 4, v14 // (op 14232) 

s_sub_u32 s7, s6, s15 // (op 14232) 
s_lshr_b32 s8, s7, 1 // (op 14232) 

s_add_u32 s7, s8, s15 // (op 14232) 


s_lshr_b32 s8, s7, s11 // (op 14232) 

s_and_b32 s7, s11, 2147483648 // (op 14232) 
s_cmp_eq_i32 s7, 0 // (op 14232) 
s_mov_b32 s15, scc // (op 14232) 

s_cmp_eq_i32 s15, 0 // (op 14232) 
s_cselect_b32 s7, s6, s8 // (op 14232) 


s_mul_hi_u32 s8, s7, s12 // (op 14232) 
s_sub_u32 s15, s7, s8 // (op 14232) 
s_lshr_b32 s18, s15, 1 // (op 14232) 

s_add_u32 s15, s18, s8 // (op 14232) 


s_lshr_b32 s8, s15, s10 // (op 14232) 

s_and_b32 s15, s10, 2147483648 // (op 14232) 
s_cmp_eq_i32 s15, 0 // (op 14232) 
s_mov_b32 s18, scc // (op 14232) 

s_cmp_eq_i32 s18, 0 // (op 14232) 
s_cselect_b32 s15, s7, s8 // (op 14232) 


v_lshl_add_u32 v14, s15, 7, v3 // (op 14232) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 14232) 
s_mul_i32 s8, s7, s18 // (op 14232) 


s_sub_u32 s7, s6, s8 // (op 14232) 


v_lshl_add_u32 v3, s7, 7, v13 // (op 14232) 


v_mov_b32 v17, 0 // (op 14232) convert
v_mov_b32 v16, v3 // (op 14232) convert

v_mul_lo_u32 v19, v14, s31 // (op 14232) most significant: low of low * high
v_mul_hi_u32 v3, v14, s30 // (op 14232) most significant: high of low * low
v_mul_lo_u32 v18, v14, s30 // (op 14232) least significant: low of low * low
v_add_u32 v19, v19, v3 // (op 14232) most significant: sum

v_add_co_u32 v18, s[18:19], v18, v16 // (op 14232) least significant half
v_addc_co_u32 v19, s[18:19], v19, v17, s[18:19] // (op 14232) most significant half



s_mul_i32 s19, s30, s35 // (op 14234) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 14234) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 14234) least significant
s_mul_hi_u32 s6, s30, s34 // (op 14234) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 14234) most significant: sum
s_add_u32 s19, s19, s6 // (op 14234) most significant: sum


s_mov_b32 s68, s32 // (op 14234) reg expression
s_mov_b32 s69, s33 // (op 14234) reg expression
s_mov_b32 s70, s18 // (op 14234) reg expression
s_mov_b32 s71, 131072 // (op 14234) call()

v_lshrrev_b32 v3, 3, v1 // (op 14251) 
s_mul_i32 s15, s16, s50 // (op 14251) 
s_add_u32 s15, s15, s0 // (op 14251) 
v_and_b32 v13, -8, v1 // (op 14251) 
s_add_u32 s6, s15, s9 // (op 14251) 

v_sub_u32 v14, v1, v13 // (op 14251) 

s_mul_hi_u32 s15, s6, s13 // (op 14251) 
v_lshlrev_b32 v13, 4, v14 // (op 14251) 

s_sub_u32 s7, s6, s15 // (op 14251) 
s_lshr_b32 s8, s7, 1 // (op 14251) 

s_add_u32 s7, s8, s15 // (op 14251) 


s_lshr_b32 s8, s7, s11 // (op 14251) 

s_and_b32 s7, s11, 2147483648 // (op 14251) 
s_cmp_eq_i32 s7, 0 // (op 14251) 
s_mov_b32 s15, scc // (op 14251) 

s_cmp_eq_i32 s15, 0 // (op 14251) 
s_cselect_b32 s7, s6, s8 // (op 14251) 


s_mul_hi_u32 s8, s7, s12 // (op 14251) 
s_sub_u32 s15, s7, s8 // (op 14251) 
s_lshr_b32 s18, s15, 1 // (op 14251) 

s_add_u32 s15, s18, s8 // (op 14251) 


s_lshr_b32 s8, s15, s10 // (op 14251) 

s_and_b32 s15, s10, 2147483648 // (op 14251) 
s_cmp_eq_i32 s15, 0 // (op 14251) 
s_mov_b32 s18, scc // (op 14251) 

s_cmp_eq_i32 s18, 0 // (op 14251) 
s_cselect_b32 s15, s7, s8 // (op 14251) 


v_lshl_add_u32 v14, s15, 7, v3 // (op 14251) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 14251) 
s_mul_i32 s8, s7, s18 // (op 14251) 


s_sub_u32 s7, s6, s8 // (op 14251) 


v_lshl_add_u32 v3, s7, 7, v13 // (op 14251) 


v_mov_b32 v17, 0 // (op 14251) convert
v_mov_b32 v16, v3 // (op 14251) convert

v_mul_lo_u32 v21, v14, s31 // (op 14251) most significant: low of low * high
v_mul_hi_u32 v3, v14, s30 // (op 14251) most significant: high of low * low
v_mul_lo_u32 v20, v14, s30 // (op 14251) least significant: low of low * low
v_add_u32 v21, v21, v3 // (op 14251) most significant: sum

v_add_co_u32 v20, s[18:19], v20, v16 // (op 14251) least significant half
v_addc_co_u32 v21, s[18:19], v21, v17, s[18:19] // (op 14251) most significant half



s_mul_i32 s19, s30, s35 // (op 14253) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 14253) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 14253) least significant
s_mul_hi_u32 s6, s30, s34 // (op 14253) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 14253) most significant: sum
s_add_u32 s19, s19, s6 // (op 14253) most significant: sum


s_mov_b32 s68, s32 // (op 14253) reg expression
s_mov_b32 s69, s33 // (op 14253) reg expression
s_mov_b32 s70, s18 // (op 14253) reg expression
s_mov_b32 s71, 131072 // (op 14253) call()

v_lshrrev_b32 v3, 2, v1 // (op 14378) 
s_mul_i32 s15, s16, s50 // (op 14378) 
s_add_u32 s15, s15, s0 // (op 14378) 
v_and_b32 v13, -4, v1 // (op 14378) 
s_add_u32 s6, s15, s9 // (op 14378) 

v_sub_u32 v14, v1, v13 // (op 14378) 

s_mul_hi_u32 s15, s6, s13 // (op 14378) 
v_lshlrev_b32 v13, 5, v14 // (op 14378) 

s_sub_u32 s7, s6, s15 // (op 14378) 
s_lshr_b32 s8, s7, 1 // (op 14378) 

s_add_u32 s7, s8, s15 // (op 14378) 


s_lshr_b32 s8, s7, s11 // (op 14378) 

s_and_b32 s7, s11, 2147483648 // (op 14378) 
s_cmp_eq_i32 s7, 0 // (op 14378) 
s_mov_b32 s15, scc // (op 14378) 

s_cmp_eq_i32 s15, 0 // (op 14378) 
s_cselect_b32 s7, s6, s8 // (op 14378) 


s_mul_hi_u32 s8, s7, s12 // (op 14378) 
s_sub_u32 s15, s7, s8 // (op 14378) 
s_lshr_b32 s18, s15, 1 // (op 14378) 

s_add_u32 s15, s18, s8 // (op 14378) 


s_lshr_b32 s8, s15, s10 // (op 14378) 

s_and_b32 s15, s10, 2147483648 // (op 14378) 
s_cmp_eq_i32 s15, 0 // (op 14378) 
s_mov_b32 s18, scc // (op 14378) 

s_cmp_eq_i32 s18, 0 // (op 14378) 
s_cselect_b32 s15, s7, s8 // (op 14378) 


s_mul_i32 s8, s15, s14 // (op 14378) 

s_sub_u32 s15, s7, s8 // (op 14378) 

v_lshl_add_u32 v14, s15, 7, v3 // (op 14378) 


v_mul_lo_u32 v17, v14, s37 // (op 14378) most significant: low of low * high
v_mul_hi_u32 v3, v14, s36 // (op 14378) most significant: high of low * low
v_mul_lo_u32 v16, v14, s36 // (op 14378) least significant: low of low * low
v_add_u32 v17, v17, v3 // (op 14378) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 14378) 
s_mul_i32 s8, s7, s18 // (op 14378) 


s_sub_u32 s7, s6, s8 // (op 14378) 


v_lshl_add_u32 v3, s7, 7, v13 // (op 14378) 


v_mov_b32 v15, 0 // (op 14378) convert
v_mov_b32 v14, v3 // (op 14378) convert

v_add_co_u32 v22, s[18:19], v14, v16 // (op 14378) least significant half
v_addc_co_u32 v23, s[18:19], v15, v17, s[18:19] // (op 14378) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 14378) 


v_lshrrev_b64 v[16:17], 3, v[22:23] // (op 14378) 

s_mul_i32 s19, s36, s23 // (op 14380) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 14380) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 14380) least significant
s_mul_hi_u32 s6, s36, s22 // (op 14380) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 14380) most significant: sum
s_add_u32 s19, s19, s6 // (op 14380) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 14380) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 14380) 

s_mov_b32 s72, s38 // (op 14380) reg expression
s_mov_b32 s73, s39 // (op 14380) reg expression
s_mov_b32 s74, s18 // (op 14380) reg expression
s_mov_b32 s75, 131072 // (op 14380) call()

v_lshrrev_b32 v3, 2, v1 // (op 14397) 
s_mul_i32 s15, s16, s50 // (op 14397) 
s_add_u32 s15, s15, s0 // (op 14397) 
v_and_b32 v13, -4, v1 // (op 14397) 
s_add_u32 s6, s15, s9 // (op 14397) 

v_sub_u32 v14, v1, v13 // (op 14397) 

s_mul_hi_u32 s15, s6, s13 // (op 14397) 
v_lshlrev_b32 v13, 5, v14 // (op 14397) 

s_sub_u32 s7, s6, s15 // (op 14397) 
s_lshr_b32 s8, s7, 1 // (op 14397) 

s_add_u32 s7, s8, s15 // (op 14397) 


s_lshr_b32 s8, s7, s11 // (op 14397) 

s_and_b32 s7, s11, 2147483648 // (op 14397) 
s_cmp_eq_i32 s7, 0 // (op 14397) 
s_mov_b32 s15, scc // (op 14397) 

s_cmp_eq_i32 s15, 0 // (op 14397) 
s_cselect_b32 s7, s6, s8 // (op 14397) 


s_mul_hi_u32 s8, s7, s12 // (op 14397) 
s_sub_u32 s15, s7, s8 // (op 14397) 
s_lshr_b32 s18, s15, 1 // (op 14397) 

s_add_u32 s15, s18, s8 // (op 14397) 


s_lshr_b32 s8, s15, s10 // (op 14397) 

s_and_b32 s15, s10, 2147483648 // (op 14397) 
s_cmp_eq_i32 s15, 0 // (op 14397) 
s_mov_b32 s18, scc // (op 14397) 

s_cmp_eq_i32 s18, 0 // (op 14397) 
s_cselect_b32 s15, s7, s8 // (op 14397) 


s_mul_i32 s8, s15, s14 // (op 14397) 

s_sub_u32 s15, s7, s8 // (op 14397) 

v_lshl_add_u32 v14, s15, 7, v3 // (op 14397) 


v_mul_lo_u32 v23, v14, s37 // (op 14397) most significant: low of low * high
v_mul_hi_u32 v3, v14, s36 // (op 14397) most significant: high of low * low
v_mul_lo_u32 v22, v14, s36 // (op 14397) least significant: low of low * low
v_add_u32 v23, v23, v3 // (op 14397) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 14397) 
s_mul_i32 s8, s7, s18 // (op 14397) 


s_sub_u32 s7, s6, s8 // (op 14397) 


v_lshl_add_u32 v3, s7, 7, v13 // (op 14397) 


v_mov_b32 v15, 0 // (op 14397) convert
v_mov_b32 v14, v3 // (op 14397) convert

v_add_co_u32 v24, s[18:19], v14, v22 // (op 14397) least significant half
v_addc_co_u32 v25, s[18:19], v15, v23, s[18:19] // (op 14397) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 14397) 


v_lshrrev_b64 v[14:15], 3, v[24:25] // (op 14397) 

s_mul_i32 s19, s36, s23 // (op 14399) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 14399) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 14399) least significant
s_mul_hi_u32 s6, s36, s22 // (op 14399) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 14399) most significant: sum
s_add_u32 s19, s19, s6 // (op 14399) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 14399) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 14399) 

s_mov_b32 s72, s38 // (op 14399) reg expression
s_mov_b32 s73, s39 // (op 14399) reg expression
s_mov_b32 s74, s18 // (op 14399) reg expression
s_mov_b32 s75, 131072 // (op 14399) call()

v_and_b32 v3, 63, v1 // (op 15103) 
s_mul_i32 s15, s16, s50 // (op 15103) 
s_add_u32 s15, s15, s0 // (op 15103) 
v_lshrrev_b32 v13, 4, v3 // (op 15103) 
s_add_u32 s6, s15, s9 // (op 15103) 

v_and_b32 v22, 3, v13 // (op 15103) 

s_mul_hi_u32 s15, s6, s13 // (op 15103) 
v_and_b32 v13, 15, v1 // (op 15103) 
s_sub_u32 s7, s6, s15 // (op 15103) 
v_lshl_add_u32 v23, v22, 4, v13 // (op 15103) 


s_lshr_b32 s8, s7, 1 // (op 15103) 

v_lshrrev_b32 v13, 7, v1 // (op 15103) 
s_add_u32 s7, s8, s15 // (op 15103) 


v_lshl_add_u32 v22, v13, 6, v23 // (op 15103) 


s_lshr_b32 s8, s7, s11 // (op 15103) 

v_lshrrev_b32 v13, 6, v3 // (op 15103) 

s_and_b32 s7, s11, 2147483648 // (op 15103) 
v_lshlrev_b32 v3, 2, v13 // (op 15103) 

s_cmp_eq_i32 s7, 0 // (op 15103) 
s_mov_b32 s15, scc // (op 15103) 

s_cmp_eq_i32 s15, 0 // (op 15103) 
s_cselect_b32 s7, s6, s8 // (op 15103) 


s_mul_hi_u32 s8, s7, s12 // (op 15103) 
s_sub_u32 s15, s7, s8 // (op 15103) 
s_lshr_b32 s18, s15, 1 // (op 15103) 

s_add_u32 s15, s18, s8 // (op 15103) 


s_lshr_b32 s8, s15, s10 // (op 15103) 

s_and_b32 s15, s10, 2147483648 // (op 15103) 
s_cmp_eq_i32 s15, 0 // (op 15103) 
s_mov_b32 s18, scc // (op 15103) 

s_cmp_eq_i32 s18, 0 // (op 15103) 
s_cselect_b32 s15, s7, s8 // (op 15103) 


v_lshl_add_u32 v13, s15, 7, v22 // (op 15103) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 15103) 
s_mul_i32 s8, s7, s18 // (op 15103) 


s_sub_u32 s7, s6, s8 // (op 15103) 


v_lshl_add_u32 v22, s7, 2, v3 // (op 15103) 


v_mul_lo_u32 v3, v13, s40 // (op 15103) 
v_add_u32 v3, v3, v22 // (op 15103) 


s_mul_i32 s19, s40, s35 // (op 15105) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 15105) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 15105) least significant
s_mul_hi_u32 s6, s40, s34 // (op 15105) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 15105) most significant: sum
s_add_u32 s19, s19, s6 // (op 15105) most significant: sum


s_mov_b32 s52, s42 // (op 15105) reg expression
s_mov_b32 s53, s43 // (op 15105) reg expression
s_mov_b32 s54, s18 // (op 15105) reg expression
s_mov_b32 s55, 131072 // (op 15105) call()

v_and_b32 v13, 63, v1 // (op 15124) 
s_mul_i32 s15, s16, s50 // (op 15124) 
s_add_u32 s15, s15, s0 // (op 15124) 
v_lshrrev_b32 v22, 4, v13 // (op 15124) 
s_add_u32 s6, s15, s9 // (op 15124) 

v_and_b32 v23, 3, v22 // (op 15124) 

s_mul_hi_u32 s15, s6, s13 // (op 15124) 
v_and_b32 v22, 15, v1 // (op 15124) 
s_sub_u32 s7, s6, s15 // (op 15124) 
v_lshl_add_u32 v24, v23, 4, v22 // (op 15124) 


s_lshr_b32 s8, s7, 1 // (op 15124) 

v_lshrrev_b32 v22, 7, v1 // (op 15124) 
s_add_u32 s7, s8, s15 // (op 15124) 


v_lshl_add_u32 v23, v22, 6, v24 // (op 15124) 


s_lshr_b32 s8, s7, s11 // (op 15124) 

v_lshrrev_b32 v22, 6, v13 // (op 15124) 

s_and_b32 s7, s11, 2147483648 // (op 15124) 
v_lshlrev_b32 v13, 2, v22 // (op 15124) 

s_cmp_eq_i32 s7, 0 // (op 15124) 
s_mov_b32 s15, scc // (op 15124) 

s_cmp_eq_i32 s15, 0 // (op 15124) 
s_cselect_b32 s7, s6, s8 // (op 15124) 


s_mul_hi_u32 s8, s7, s12 // (op 15124) 
s_sub_u32 s15, s7, s8 // (op 15124) 
s_lshr_b32 s18, s15, 1 // (op 15124) 

s_add_u32 s15, s18, s8 // (op 15124) 


s_lshr_b32 s8, s15, s10 // (op 15124) 

s_and_b32 s15, s10, 2147483648 // (op 15124) 
s_cmp_eq_i32 s15, 0 // (op 15124) 
s_mov_b32 s18, scc // (op 15124) 

s_cmp_eq_i32 s18, 0 // (op 15124) 
s_cselect_b32 s15, s7, s8 // (op 15124) 


v_lshl_add_u32 v22, s15, 7, v23 // (op 15124) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 15124) 
s_mul_i32 s8, s7, s18 // (op 15124) 


s_sub_u32 s7, s6, s8 // (op 15124) 


v_lshl_add_u32 v23, s7, 2, v13 // (op 15124) 


v_mul_lo_u32 v13, v22, s40 // (op 15124) 
v_add_u32 v13, v13, v23 // (op 15124) 


s_mul_i32 s19, s40, s35 // (op 15126) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 15126) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 15126) least significant
s_mul_hi_u32 s6, s40, s34 // (op 15126) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 15126) most significant: sum
s_add_u32 s19, s19, s6 // (op 15126) most significant: sum


s_mov_b32 s52, s42 // (op 15126) reg expression
s_mov_b32 s53, s43 // (op 15126) reg expression
s_mov_b32 s54, s18 // (op 15126) reg expression
s_mov_b32 s55, 131072 // (op 15126) call()

v_and_b32 v22, 63, v1 // (op 15155) 
s_mul_i32 s15, s16, s50 // (op 15155) 
s_add_u32 s15, s15, s0 // (op 15155) 
v_lshrrev_b32 v23, 4, v22 // (op 15155) 
s_add_u32 s6, s15, s9 // (op 15155) 

v_and_b32 v24, 3, v23 // (op 15155) 

s_mul_hi_u32 s15, s6, s13 // (op 15155) 
v_and_b32 v23, 15, v1 // (op 15155) 
s_sub_u32 s7, s6, s15 // (op 15155) 
v_lshl_add_u32 v25, v24, 4, v23 // (op 15155) 


s_lshr_b32 s8, s7, 1 // (op 15155) 

v_lshrrev_b32 v23, 6, v1 // (op 15155) 
s_add_u32 s7, s8, s15 // (op 15155) 


v_and_b32 v24, 1, v23 // (op 15155) 

s_lshr_b32 s8, s7, s11 // (op 15155) 

v_lshl_add_u32 v23, v24, 6, v25 // (op 15155) 


s_and_b32 s7, s11, 2147483648 // (op 15155) 
v_lshrrev_b32 v24, 6, v22 // (op 15155) 

s_cmp_eq_i32 s7, 0 // (op 15155) 
s_mov_b32 s15, scc // (op 15155) 

v_lshlrev_b32 v22, 2, v24 // (op 15155) 

s_cmp_eq_i32 s15, 0 // (op 15155) 
s_cselect_b32 s7, s6, s8 // (op 15155) 


s_mul_hi_u32 s8, s7, s12 // (op 15155) 
s_sub_u32 s15, s7, s8 // (op 15155) 
s_lshr_b32 s18, s15, 1 // (op 15155) 

s_add_u32 s15, s18, s8 // (op 15155) 


s_lshr_b32 s8, s15, s10 // (op 15155) 

s_and_b32 s15, s10, 2147483648 // (op 15155) 
s_cmp_eq_i32 s15, 0 // (op 15155) 
s_mov_b32 s18, scc // (op 15155) 

s_cmp_eq_i32 s18, 0 // (op 15155) 
s_cselect_b32 s15, s7, s8 // (op 15155) 


s_mul_i32 s8, s15, s14 // (op 15155) 

s_sub_u32 s15, s7, s8 // (op 15155) 

v_lshl_add_u32 v24, s15, 7, v23 // (op 15155) 


v_mul_lo_u32 v23, v24, s44 // (op 15155) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 15155) 
s_mul_i32 s8, s7, s18 // (op 15155) 


s_sub_u32 s7, s6, s8 // (op 15155) 


v_lshl_add_u32 v24, s7, 2, v22 // (op 15155) 


v_add_u32 v22, v24, v23 // (op 15155) 


s_mul_i32 s19, s44, s23 // (op 15157) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 15157) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 15157) least significant
s_mul_hi_u32 s6, s44, s22 // (op 15157) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 15157) most significant: sum
s_add_u32 s19, s19, s6 // (op 15157) most significant: sum


s_mov_b32 s64, s46 // (op 15157) reg expression
s_mov_b32 s65, s47 // (op 15157) reg expression
s_mov_b32 s66, s18 // (op 15157) reg expression
s_mov_b32 s67, 131072 // (op 15157) call()

v_and_b32 v23, 63, v1 // (op 15176) 
s_mul_i32 s15, s16, s50 // (op 15176) 
s_add_u32 s15, s15, s0 // (op 15176) 
v_lshrrev_b32 v24, 4, v23 // (op 15176) 
s_add_u32 s6, s15, s9 // (op 15176) 

v_and_b32 v25, 3, v24 // (op 15176) 

s_mul_hi_u32 s15, s6, s13 // (op 15176) 
v_and_b32 v24, 15, v1 // (op 15176) 
s_sub_u32 s7, s6, s15 // (op 15176) 
v_lshl_add_u32 v26, v25, 4, v24 // (op 15176) 


s_lshr_b32 s8, s7, 1 // (op 15176) 

v_lshrrev_b32 v24, 6, v1 // (op 15176) 
s_add_u32 s7, s8, s15 // (op 15176) 


v_and_b32 v25, 1, v24 // (op 15176) 

s_lshr_b32 s8, s7, s11 // (op 15176) 

v_lshl_add_u32 v24, v25, 6, v26 // (op 15176) 


s_and_b32 s7, s11, 2147483648 // (op 15176) 
v_lshrrev_b32 v25, 6, v23 // (op 15176) 

s_cmp_eq_i32 s7, 0 // (op 15176) 
s_mov_b32 s15, scc // (op 15176) 

v_lshlrev_b32 v23, 2, v25 // (op 15176) 

s_cmp_eq_i32 s15, 0 // (op 15176) 
s_cselect_b32 s7, s6, s8 // (op 15176) 


s_mul_hi_u32 s8, s7, s12 // (op 15176) 
s_sub_u32 s15, s7, s8 // (op 15176) 
s_lshr_b32 s18, s15, 1 // (op 15176) 

s_add_u32 s15, s18, s8 // (op 15176) 


s_lshr_b32 s8, s15, s10 // (op 15176) 

s_and_b32 s15, s10, 2147483648 // (op 15176) 
s_cmp_eq_i32 s15, 0 // (op 15176) 
s_mov_b32 s18, scc // (op 15176) 

s_cmp_eq_i32 s18, 0 // (op 15176) 
s_cselect_b32 s15, s7, s8 // (op 15176) 


s_mul_i32 s8, s15, s14 // (op 15176) 

s_sub_u32 s15, s7, s8 // (op 15176) 

v_lshl_add_u32 v25, s15, 7, v24 // (op 15176) 


v_mul_lo_u32 v24, v25, s44 // (op 15176) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 15176) 
s_mul_i32 s8, s7, s18 // (op 15176) 


s_sub_u32 s7, s6, s8 // (op 15176) 


v_lshl_add_u32 v25, s7, 2, v23 // (op 15176) 


v_add_u32 v23, v25, v24 // (op 15176) 


s_mul_i32 s19, s44, s23 // (op 15178) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 15178) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 15178) least significant
s_mul_hi_u32 s6, s44, s22 // (op 15178) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 15178) most significant: sum
s_add_u32 s19, s19, s6 // (op 15178) most significant: sum


s_mov_b32 s64, s46 // (op 15178) reg expression
s_mov_b32 s65, s47 // (op 15178) reg expression
s_mov_b32 s66, s18 // (op 15178) reg expression
s_mov_b32 s67, 131072 // (op 15178) call()

v_and_b32 v24, -4, v1 // (op 14818) 
v_sub_u32 v25, v1, v24 // (op 14818) 

v_lshlrev_b32 v24, 5, v25 // (op 14818) 

v_lshrrev_b32 v25, 2, v1 // (op 14818) 
v_lshl_add_u32 v26, v25, 7, v24 // (op 14818) 


v_lshlrev_b32 v24, 2, v26 // (op 14818) 

v_lshrrev_b32 v25, 3, v24 // (op 14818) 

s_nop 0
v_readfirstlane_b32 s15, v25 // (op 14818) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 15623) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 14128) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 14128) 

v_lshrrev_b64 v[24:25], 3, s[28:29] // (op 14128) 


v_mov_b64 v[26:27], v[16:17] // (op 14128) 
s_mov_b32 m0, s15 // (op 14128) reg expression
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 14128) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 14128) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 14128) most significant half

s_add_u32 m0, m0, 4096 // (op 14128) 
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 14128) Load value direct to lds



v_and_b32 v24, -8, v1 // (op 14743) 
v_sub_u32 v25, v1, v24 // (op 14743) 

v_lshlrev_b32 v24, 4, v25 // (op 14743) 

v_lshrrev_b32 v25, 3, v1 // (op 14743) 
v_lshl_add_u32 v26, v25, 7, v24 // (op 14743) 


s_nop 0
v_readfirstlane_b32 s15, v26 // (op 14743) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 15604) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 14133) 
v_mov_b64 v[24:25], s[18:19] // (op 14133) convert


v_mov_b64 v[26:27], v[18:19] // (op 14133) 
s_add_u32 s6, 8192, s15 // (op 14133) 
s_mov_b32 m0, s6 // (op 14133) reg expression
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 14133) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 14133) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 14133) most significant half

s_add_u32 m0, m0, 4096 // (op 14133) 
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 14133) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 14133) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 14133) most significant half

s_add_u32 m0, m0, 4096 // (op 14133) 
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 14133) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 14133) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 14133) most significant half

s_add_u32 m0, m0, 4096 // (op 14133) 
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 14133) Load value direct to lds






v_mov_b32 v24, v22 // (op 178) reg expression

buffer_load_dwordx2 v[26:27], v24, s[64:67], 0 offen // (op 178) Load value




v_mov_b32 v22, v3 // (op 176) reg expression

buffer_load_dwordx2 v[24:25], v22, s[52:55], 0 offen // (op 176) Load value


v_and_b32 v3, -4, v1 // (op 14976) 
v_sub_u32 v22, v1, v3 // (op 14976) 

v_lshlrev_b32 v3, 5, v22 // (op 14976) 

v_lshrrev_b32 v22, 2, v1 // (op 14976) 
v_lshl_add_u32 v28, v22, 7, v3 // (op 14976) 


v_lshlrev_b32 v3, 2, v28 // (op 14976) 

v_lshrrev_b32 v22, 3, v3 // (op 14976) 

s_nop 0
v_readfirstlane_b32 s15, v22 // (op 14976) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 15652) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 14138) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 14138) 

v_lshrrev_b64 v[28:29], 3, s[28:29] // (op 14138) 


v_mov_b64 v[30:31], v[16:17] // (op 14138) 
s_add_u32 s6, 24576, s15 // (op 14138) 
s_mov_b32 m0, s6 // (op 14138) reg expression
buffer_load_dwordx4 v30, s[72:75], 64 offen offset: 0 lds // (op 14138) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 14138) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 14138) most significant half

s_add_u32 m0, m0, 4096 // (op 14138) 
buffer_load_dwordx4 v30, s[72:75], 64 offen offset: 0 lds // (op 14138) Load value direct to lds





v_and_b32 v3, -8, v1 // (op 14932) 
v_sub_u32 v22, v1, v3 // (op 14932) 

v_lshlrev_b32 v3, 4, v22 // (op 14932) 

v_lshrrev_b32 v22, 3, v1 // (op 14932) 
v_lshl_add_u32 v16, v22, 7, v3 // (op 14932) 


s_nop 0
v_readfirstlane_b32 s15, v16 // (op 14932) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 15642) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 14140) 
v_mov_b64 v[16:17], s[18:19] // (op 14140) convert


v_mov_b64 v[28:29], v[18:19] // (op 14140) 
s_add_u32 s6, 32768, s15 // (op 14140) 
s_mov_b32 m0, s6 // (op 14140) reg expression
s_mov_b32 s7, 128 // (op 14140) call()
buffer_load_dwordx4 v28, s[68:71], s7 offen offset: 0 lds // (op 14140) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v16 // (op 14140) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 14140) most significant half

s_add_u32 m0, m0, 4096 // (op 14140) 
buffer_load_dwordx4 v28, s[68:71], s7 offen offset: 0 lds // (op 14140) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v16 // (op 14140) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 14140) most significant half

s_add_u32 m0, m0, 4096 // (op 14140) 
buffer_load_dwordx4 v28, s[68:71], s7 offen offset: 0 lds // (op 14140) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v16 // (op 14140) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 14140) most significant half

s_add_u32 m0, m0, 4096 // (op 14140) 
buffer_load_dwordx4 v28, s[68:71], s7 offen offset: 0 lds // (op 14140) Load value direct to lds






s_waitcnt vmcnt(12)
s_barrier  // (op 15620) 



ds_read_b128 v[16:19], v12 // (op 13622) Load local data 



ds_read_b128 v[28:31], v12 offset:1024 // (op 13627) Load local data 



ds_read_b128 v[32:35], v12 offset:2048 // (op 13632) Load local data 



ds_read_b128 v[36:39], v12 offset:3072 // (op 13637) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 15601) 



ds_read_b128 v[40:43], v10 offset:8192 // (op 13602) Load local data 
ds_read_b128 v[44:47], v10 offset:8256 // (op 13602) Load local data 



ds_read_b128 v[48:51], v10 offset:10240 // (op 13607) Load local data 
ds_read_b128 v[52:55], v10 offset:10304 // (op 13607) Load local data 



ds_read_b128 v[56:59], v10 offset:12288 // (op 13612) Load local data 
ds_read_b128 v[60:63], v10 offset:12352 // (op 13612) Load local data 



ds_read_b128 v[64:67], v10 offset:14336 // (op 13617) Load local data 
ds_read_b128 v[68:71], v10 offset:14400 // (op 13617) Load local data 
v_and_b32 v3, -8, v1 // (op 14728) 
v_sub_u32 v22, v1, v3 // (op 14728) 

v_lshlrev_b32 v3, 4, v22 // (op 14728) 

v_lshrrev_b32 v22, 3, v1 // (op 14728) 
v_lshl_add_u32 v72, v22, 7, v3 // (op 14728) 


s_nop 0
v_readfirstlane_b32 s15, v72 // (op 14728) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 14803) 
v_sub_u32 v22, v1, v3 // (op 14803) 

v_lshlrev_b32 v3, 5, v22 // (op 14803) 

v_lshrrev_b32 v22, 2, v1 // (op 14803) 
v_lshl_add_u32 v72, v22, 7, v3 // (op 14803) 


v_lshlrev_b32 v3, 2, v72 // (op 14803) 

v_lshrrev_b32 v22, 3, v3 // (op 14803) 

s_nop 0
v_readfirstlane_b32 s6, v22 // (op 14803) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -8, v1 // (op 14917) 
v_sub_u32 v22, v1, v3 // (op 14917) 

v_lshlrev_b32 v3, 4, v22 // (op 14917) 

v_lshrrev_b32 v22, 3, v1 // (op 14917) 
v_lshl_add_u32 v72, v22, 7, v3 // (op 14917) 


s_nop 0
v_readfirstlane_b32 s7, v72 // (op 14917) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 14961) 
v_sub_u32 v22, v1, v3 // (op 14961) 

v_lshlrev_b32 v3, 5, v22 // (op 14961) 

v_lshrrev_b32 v22, 2, v1 // (op 14961) 
v_lshl_add_u32 v72, v22, 7, v3 // (op 14961) 


v_lshlrev_b32 v3, 2, v72 // (op 14961) 

v_lshrrev_b32 v22, 3, v3 // (op 14961) 

s_nop 0
v_readfirstlane_b32 s8, v22 // (op 14961) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s1, 0 // (op 1130) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1129) 
s_mov_b32 s28, s18 // (op 1129) convert

s_and_b32 s29, s28, 4294967294 // (op 1129) 

s_cmp_lt_u32 s1, s29 // (op 1129) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_1129 // (op 1129) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_1129 if false)
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_1129:
s_add_u32 s1, s1, 2 // (op 1131) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 15626) 
s_waitcnt vmcnt(7)
v_mov_b32 v72, v26 // (op 14018) reg expression

v_mov_b32 v73, v27 // (op 14018) reg expression
s_lshl_b64 s[18:19], s[36:37], 6 // (op 14079) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 14079) 

v_lshrrev_b64 v[74:75], 3, s[28:29] // (op 14079) 


v_mov_b64 v[76:77], v[14:15] // (op 14079) 
v_mov_b64 v[78:79], v[76:77] // (op 14079) 
s_mov_b32 m0, s6 // (op 14079) reg expression
s_mov_b32 s18, 128 // (op 14079) call()
buffer_load_dwordx4 v78, s[72:75], s18 offen offset: 0 lds // (op 14079) Load value direct to lds
v_add_co_u32 v78, s[28:29], v78, v74 // (op 14079) least significant half
v_addc_co_u32 v79, s[28:29], 0, v79, s[28:29] // (op 14079) most significant half

s_add_u32 m0, m0, 4096 // (op 14079) 
v_bfe_u32 v80, v73, 0, 8 // (op 12964) 
v_bfe_u32 v81, v73, 8, 8 // (op 12964) 
v_bfe_u32 v82, v73, 16, 8 // (op 12964) 
v_bfe_u32 v83, v73, 24, 8 // (op 12964) 
v_permlane16_swap_b32 v80, v81 // (op 12964) 
s_nop 0
v_permlane16_swap_b32 v82, v83 // (op 12964) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v80, v82 // (op 12964) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v81, v83 // (op 12964) 
s_waitcnt vmcnt(7)
v_mov_b32 v84, v24 // (op 14032) reg expression
v_mov_b32 v85, v25 // (op 14032) reg expression
buffer_load_dwordx4 v78, s[72:75], s18 offen offset: 0 lds // (op 14079) Load value direct to lds



v_bfe_u32 v86, v84, 0, 8 // (op 12886) 
v_bfe_u32 v87, v84, 8, 8 // (op 12886) 
v_bfe_u32 v88, v84, 16, 8 // (op 12886) 
v_bfe_u32 v89, v84, 24, 8 // (op 12886) 
v_permlane16_swap_b32 v86, v87 // (op 12886) 
v_bfe_u32 v90, v85, 0, 8 // (op 12901) 
v_permlane16_swap_b32 v88, v89 // (op 12886) 
v_bfe_u32 v91, v85, 8, 8 // (op 12901) 
v_bfe_u32 v92, v85, 16, 8 // (op 12901) 
v_permlane32_swap_b32 v86, v88 // (op 12886) 
v_permlane32_swap_b32 v87, v89 // (op 12886) 
v_bfe_u32 v93, v85, 24, 8 // (op 12901) 
v_permlane16_swap_b32 v90, v91 // (op 12901) 
s_nop 0
v_permlane16_swap_b32 v92, v93 // (op 12901) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v90, v92 // (op 12901) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v91, v93 // (op 12901) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 15607) 


s_lshl_b64 s[18:19], s[30:31], 5 // (op 14074) 
v_mov_b64 v[74:75], s[18:19] // (op 14074) convert


v_mov_b64 v[78:79], v[20:21] // (op 14074) 
v_mov_b64 v[84:85], v[78:79] // (op 14074) 
s_add_u32 s18, 8192, s15 // (op 14074) 
s_mov_b32 m0, s18 // (op 14074) reg expression
s_mov_b32 s19, 256 // (op 14074) call()
buffer_load_dwordx4 v84, s[68:71], s19 offen offset: 0 lds // (op 14074) Load value direct to lds
v_add_co_u32 v84, s[28:29], v84, v74 // (op 14074) least significant half
v_addc_co_u32 v85, s[28:29], 0, v85, s[28:29] // (op 14074) most significant half

s_add_u32 m0, m0, 4096 // (op 14074) 
v_bfe_u32 v94, v72, 0, 8 // (op 12949) 
v_bfe_u32 v95, v72, 8, 8 // (op 12949) 
v_bfe_u32 v96, v72, 16, 8 // (op 12949) 
v_bfe_u32 v97, v72, 24, 8 // (op 12949) 
v_permlane16_swap_b32 v94, v95 // (op 12949) 
s_nop 0
v_permlane16_swap_b32 v96, v97 // (op 12949) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v94, v96 // (op 12949) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v95, v97 // (op 12949) 
buffer_load_dwordx4 v84, s[68:71], s19 offen offset: 0 lds // (op 14074) Load value direct to lds

v_add_co_u32 v84, s[28:29], v84, v74 // (op 14074) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[16:19], a[0:3], v86, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 179) 
v_addc_co_u32 v85, s[28:29], 0, v85, s[28:29] // (op 14074) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[40:47], v[28:31], a[4:7], v86, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 585) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[40:47], v[32:35], a[8:11], v86, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 721) 
s_add_u32 m0, m0, 4096 // (op 14074) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[40:47], v[36:39], a[12:15], v86, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 857) 
buffer_load_dwordx4 v84, s[68:71], s19 offen offset: 0 lds // (op 14074) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[48:55], v[16:19], a[16:19], v87, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1277) 
v_add_co_u32 v84, s[28:29], v84, v74 // (op 14074) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[48:55], v[28:31], a[20:23], v87, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1239) 
v_addc_co_u32 v85, s[28:29], 0, v85, s[28:29] // (op 14074) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[48:55], v[32:35], a[24:27], v87, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1201) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:55], v[36:39], a[28:31], v87, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1163) 
s_add_u32 m0, m0, 4096 // (op 14074) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[56:63], v[16:19], a[32:35], v88, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1887) 
buffer_load_dwordx4 v84, s[68:71], s19 offen offset: 0 lds // (op 14074) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[56:63], v[28:31], a[36:39], v88, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1849) 
v_mov_b32 v3, v23 // (op 14027) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[56:63], v[32:35], a[40:43], v88, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1811) 
v_add_u32 v22, 8, v3 // (op 14027) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[56:63], v[36:39], a[44:47], v88, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1773) 
s_waitcnt vmcnt(10)
s_barrier  // (op 15648) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[64:71], v[16:19], a[48:51], v89, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2497) 
ds_read_b128 v[72:75], v8 offset:24576 // (op 486) Load local data 
buffer_load_dwordx2 v[26:27], v22, s[64:67], 0 offen // (op 14027) Load value





v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[64:71], v[28:31], a[52:55], v89, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2459) 
v_add_u32 v23, 8, v23 // (op 15166) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[64:71], v[32:35], a[56:59], v89, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2421) 
v_mov_b32 v3, v13 // (op 14041) 
v_add_u32 v22, 8, v3 // (op 14041) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[64:71], v[36:39], a[60:63], v89, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2383) 
ds_read_b128 v[98:101], v8 offset:25600 // (op 594) Load local data 





ds_read_b128 v[94:97], v8 offset:26624 // (op 730) Load local data 
buffer_load_dwordx2 v[24:25], v22, s[52:55], 0 offen // (op 14041) Load value





v_add_u32 v13, 8, v13 // (op 15114) 
ds_read_b128 v[84:87], v8 offset:27648 // (op 866) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 15639) 



ds_read_b128 v[40:43], v7 offset:32768 // (op 484) Load local data 
ds_read_b128 v[44:47], v7 offset:32832 // (op 484) Load local data 



ds_read_b128 v[48:51], v7 offset:34816 // (op 1171) Load local data 
ds_read_b128 v[52:55], v7 offset:34880 // (op 1171) Load local data 



ds_read_b128 v[56:59], v7 offset:36864 // (op 1781) Load local data 
ds_read_b128 v[60:63], v7 offset:36928 // (op 1781) Load local data 



ds_read_b128 v[102:105], v7 offset:38912 // (op 2391) Load local data 
ds_read_b128 v[106:109], v7 offset:38976 // (op 2391) Load local data 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[72:75], a[0:3], v90, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 491) 
s_barrier  // (op 15655) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[40:47], v[98:101], a[4:7], v90, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 586) 
s_lshl_b64 s[18:19], s[36:37], 6 // (op 14101) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[40:47], v[84:87], a[12:15], v90, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 858) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 14101) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[40:47], v[94:97], a[8:11], v90, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 722) 
v_lshrrev_b64 v[88:89], 3, s[28:29] // (op 14101) 


v_mov_b64 v[110:111], v[76:77] // (op 14101) 
s_add_u32 s18, 24576, s8 // (op 14101) 
s_mov_b32 m0, s18 // (op 14101) reg expression
s_mov_b32 s19, 192 // (op 14101) call()
buffer_load_dwordx4 v110, s[72:75], s19 offen offset: 0 lds // (op 14101) Load value direct to lds
v_add_co_u32 v110, s[28:29], v110, v88 // (op 14101) least significant half
v_addc_co_u32 v111, s[28:29], 0, v111, s[28:29] // (op 14101) most significant half

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[48:55], v[72:75], a[16:19], v91, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1278) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[48:55], v[98:101], a[20:23], v91, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1240) 
s_add_u32 m0, m0, 4096 // (op 14101) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[48:55], v[94:97], a[24:27], v91, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1202) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:55], v[84:87], a[28:31], v91, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1164) 
s_waitcnt vmcnt(7)
s_barrier  // (op 15632) 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[56:63], v[72:75], a[32:35], v92, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1888) 
ds_read_b128 v[16:19], v6 // (op 174) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[56:63], v[98:101], a[36:39], v92, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1850) 
buffer_load_dwordx4 v110, s[72:75], s19 offen offset: 0 lds // (op 14101) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[56:63], v[94:97], a[40:43], v92, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1812) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[56:63], v[84:87], a[44:47], v92, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1774) 
v_add_co_u32 v14, vcc, 128, v14 // (op 14389) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 14389) most significant half
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[102:109], v[72:75], a[48:51], v93, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2498) 
s_barrier  // (op 15645) 





v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[102:109], v[98:101], a[52:55], v93, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2460) 
ds_read_b128 v[28:31], v6 offset:1024 // (op 582) Load local data 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 14096) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[102:109], v[94:97], a[56:59], v93, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2422) 
ds_read_b128 v[32:35], v6 offset:2048 // (op 718) Load local data 

v_mov_b64 v[88:89], s[18:19] // (op 14096) convert





v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[102:109], v[84:87], a[60:63], v93, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2384) 
ds_read_b128 v[36:39], v6 offset:3072 // (op 854) Load local data 




v_mov_b64 v[72:73], v[78:79] // (op 14096) 
s_add_u32 s18, 32768, s7 // (op 14096) 
s_mov_b32 m0, s18 // (op 14096) reg expression
s_mov_b32 s19, 384 // (op 14096) call()
s_waitcnt vmcnt(4)
s_barrier  // (op 15613) 
buffer_load_dwordx4 v72, s[68:71], s19 offen offset: 0 lds // (op 14096) Load value direct to lds



v_add_co_u32 v72, s[28:29], v72, v88 // (op 14096) least significant half
v_addc_co_u32 v73, s[28:29], 0, v73, s[28:29] // (op 14096) most significant half

s_add_u32 m0, m0, 4096 // (op 14096) 
ds_read_b128 v[40:43], v5 offset:8192 // (op 172) Load local data 
ds_read_b128 v[44:47], v5 offset:8256 // (op 172) Load local data 
buffer_load_dwordx4 v72, s[68:71], s19 offen offset: 0 lds // (op 14096) Load value direct to lds



ds_read_b128 v[48:51], v5 offset:10240 // (op 1159) Load local data 
ds_read_b128 v[52:55], v5 offset:10304 // (op 1159) Load local data 
v_add_co_u32 v72, s[28:29], v72, v88 // (op 14096) least significant half



ds_read_b128 v[56:59], v5 offset:12288 // (op 1769) Load local data 
v_addc_co_u32 v73, s[28:29], 0, v73, s[28:29] // (op 14096) most significant half

s_add_u32 m0, m0, 4096 // (op 14096) 
ds_read_b128 v[60:63], v5 offset:12352 // (op 1769) Load local data 
buffer_load_dwordx4 v72, s[68:71], s19 offen offset: 0 lds // (op 14096) Load value direct to lds



ds_read_b128 v[64:67], v5 offset:14336 // (op 2379) Load local data 
ds_read_b128 v[68:71], v5 offset:14400 // (op 2379) Load local data 
v_add_co_u32 v72, s[28:29], v72, v88 // (op 14096) least significant half
v_addc_co_u32 v73, s[28:29], 0, v73, s[28:29] // (op 14096) most significant half

s_add_u32 m0, m0, 4096 // (op 14096) 
buffer_load_dwordx4 v72, s[68:71], s19 offen offset: 0 lds // (op 14096) Load value direct to lds





v_add_co_u32 v20, vcc, 256, v20 // (op 14243) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 14243) most significant half
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1129) 
s_mov_b32 s28, s18 // (op 1129) convert

s_and_b32 s29, s28, 4294967294 // (op 1129) 

s_cmp_lt_u32 s1, s29 // (op 1129) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_1129 // (op 1129) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_1129 if true)
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_1129:










s_barrier  // (op 15764) 
v_lshrrev_b32 v3, 3, v1 // (op 14167) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 14167) 
v_and_b32 v13, -8, v1 // (op 14167) 
s_and_b32 s15, s18, 4294967294 // (op 14167) 
v_sub_u32 v14, v1, v13 // (op 14167) 

s_add_u32 s6, s0, s15 // (op 14167) 

v_lshlrev_b32 v13, 4, v14 // (op 14167) 

s_mul_i32 s15, s16, s50 // (op 14167) 
s_add_u32 s15, s15, s6 // (op 14167) 

s_add_u32 s6, s15, s9 // (op 14167) 

s_mul_hi_u32 s15, s6, s13 // (op 14167) 
s_sub_u32 s7, s6, s15 // (op 14167) 
s_lshr_b32 s8, s7, 1 // (op 14167) 

s_add_u32 s7, s8, s15 // (op 14167) 


s_lshr_b32 s8, s7, s11 // (op 14167) 

s_and_b32 s7, s11, 2147483648 // (op 14167) 
s_cmp_eq_i32 s7, 0 // (op 14167) 
s_mov_b32 s15, scc // (op 14167) 

s_cmp_eq_i32 s15, 0 // (op 14167) 
s_cselect_b32 s7, s6, s8 // (op 14167) 


s_mul_hi_u32 s8, s7, s12 // (op 14167) 
s_sub_u32 s15, s7, s8 // (op 14167) 
s_lshr_b32 s28, s15, 1 // (op 14167) 

s_add_u32 s15, s28, s8 // (op 14167) 


s_lshr_b32 s8, s15, s10 // (op 14167) 

s_and_b32 s15, s10, 2147483648 // (op 14167) 
s_cmp_eq_i32 s15, 0 // (op 14167) 
s_mov_b32 s28, scc // (op 14167) 

s_cmp_eq_i32 s28, 0 // (op 14167) 
s_cselect_b32 s15, s7, s8 // (op 14167) 


s_mul_i32 s8, s7, s18 // (op 14167) 


v_lshl_add_u32 v14, s15, 7, v3 // (op 14167) 


s_sub_u32 s7, s6, s8 // (op 14167) 


v_lshl_add_u32 v3, s7, 7, v13 // (op 14167) 


v_mov_b32 v21, 0 // (op 14167) convert
v_mov_b32 v20, v3 // (op 14167) convert

v_mul_lo_u32 v23, v14, s31 // (op 14167) most significant: low of low * high
v_mul_hi_u32 v3, v14, s30 // (op 14167) most significant: high of low * low
v_mul_lo_u32 v22, v14, s30 // (op 14167) least significant: low of low * low
v_add_u32 v23, v23, v3 // (op 14167) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v20 // (op 14167) least significant half
v_addc_co_u32 v23, s[18:19], v23, v21, s[18:19] // (op 14167) most significant half



s_mul_i32 s19, s30, s35 // (op 14169) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 14169) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 14169) least significant
s_mul_hi_u32 s6, s30, s34 // (op 14169) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 14169) most significant: sum
s_add_u32 s19, s19, s6 // (op 14169) most significant: sum


s_mov_b32 s68, s32 // (op 14169) reg expression
s_mov_b32 s69, s33 // (op 14169) reg expression
s_mov_b32 s70, s18 // (op 14169) reg expression
s_mov_b32 s71, 131072 // (op 14169) call()

v_lshrrev_b32 v3, 2, v1 // (op 14333) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 14333) 
v_and_b32 v13, -4, v1 // (op 14333) 
s_and_b32 s15, s18, 4294967294 // (op 14333) 
v_sub_u32 v14, v1, v13 // (op 14333) 

s_add_u32 s6, s0, s15 // (op 14333) 

v_lshlrev_b32 v13, 5, v14 // (op 14333) 

s_mul_i32 s15, s16, s50 // (op 14333) 
s_add_u32 s15, s15, s6 // (op 14333) 

s_add_u32 s6, s15, s9 // (op 14333) 

s_mul_hi_u32 s15, s6, s13 // (op 14333) 
s_sub_u32 s7, s6, s15 // (op 14333) 
s_lshr_b32 s8, s7, 1 // (op 14333) 

s_add_u32 s7, s8, s15 // (op 14333) 


s_lshr_b32 s8, s7, s11 // (op 14333) 

s_and_b32 s7, s11, 2147483648 // (op 14333) 
s_cmp_eq_i32 s7, 0 // (op 14333) 
s_mov_b32 s15, scc // (op 14333) 

s_cmp_eq_i32 s15, 0 // (op 14333) 
s_cselect_b32 s7, s6, s8 // (op 14333) 


s_mul_hi_u32 s8, s7, s12 // (op 14333) 
s_sub_u32 s15, s7, s8 // (op 14333) 
s_lshr_b32 s28, s15, 1 // (op 14333) 

s_add_u32 s15, s28, s8 // (op 14333) 


s_lshr_b32 s8, s15, s10 // (op 14333) 

s_and_b32 s15, s10, 2147483648 // (op 14333) 
s_cmp_eq_i32 s15, 0 // (op 14333) 
s_mov_b32 s28, scc // (op 14333) 

s_cmp_eq_i32 s28, 0 // (op 14333) 
s_cselect_b32 s15, s7, s8 // (op 14333) 


s_mul_i32 s8, s15, s14 // (op 14333) 

s_sub_u32 s15, s7, s8 // (op 14333) 

s_mul_i32 s8, s7, s18 // (op 14333) 


v_lshl_add_u32 v14, s15, 7, v3 // (op 14333) 


s_sub_u32 s7, s6, s8 // (op 14333) 


v_mul_lo_u32 v21, v14, s37 // (op 14333) most significant: low of low * high
v_mul_hi_u32 v3, v14, s36 // (op 14333) most significant: high of low * low
v_mul_lo_u32 v20, v14, s36 // (op 14333) least significant: low of low * low
v_add_u32 v21, v21, v3 // (op 14333) most significant: sum


v_lshl_add_u32 v3, s7, 7, v13 // (op 14333) 


v_mov_b32 v15, 0 // (op 14333) convert
v_mov_b32 v14, v3 // (op 14333) convert

v_add_co_u32 v72, s[18:19], v14, v20 // (op 14333) least significant half
v_addc_co_u32 v73, s[18:19], v15, v21, s[18:19] // (op 14333) most significant half

v_lshlrev_b64 v[72:73], 2, v[72:73] // (op 14333) 


v_lshrrev_b64 v[14:15], 3, v[72:73] // (op 14333) 

s_mul_i32 s19, s36, s23 // (op 14335) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 14335) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 14335) least significant
s_mul_hi_u32 s6, s36, s22 // (op 14335) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 14335) most significant: sum
s_add_u32 s19, s19, s6 // (op 14335) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 14335) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 14335) 

s_mov_b32 s72, s38 // (op 14335) reg expression
s_mov_b32 s73, s39 // (op 14335) reg expression
s_mov_b32 s74, s18 // (op 14335) reg expression
s_mov_b32 s75, 131072 // (op 14335) call()

v_and_b32 v3, -8, v1 // (op 14713) 
v_sub_u32 v13, v1, v3 // (op 14713) 

v_lshlrev_b32 v3, 4, v13 // (op 14713) 

v_lshrrev_b32 v13, 3, v1 // (op 14713) 
v_lshl_add_u32 v20, v13, 7, v3 // (op 14713) 


s_nop 0
v_readfirstlane_b32 s15, v20 // (op 14713) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 14790) 
v_sub_u32 v13, v1, v3 // (op 14790) 

v_lshlrev_b32 v3, 5, v13 // (op 14790) 

v_lshrrev_b32 v13, 2, v1 // (op 14790) 
v_lshl_add_u32 v20, v13, 7, v3 // (op 14790) 


v_lshlrev_b32 v3, 2, v20 // (op 14790) 

v_lshrrev_b32 v13, 3, v3 // (op 14790) 

s_nop 0
v_readfirstlane_b32 s6, v13 // (op 14790) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, 63, v1 // (op 15203) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 15203) 
v_lshrrev_b32 v13, 4, v3 // (op 15203) 
s_and_b32 s7, s18, 4294967294 // (op 15203) 
v_and_b32 v20, 3, v13 // (op 15203) 

s_add_u32 s8, s0, s7 // (op 15203) 

v_and_b32 v13, 15, v1 // (op 15203) 
v_lshl_add_u32 v21, v20, 4, v13 // (op 15203) 


s_mul_i32 s7, s16, s50 // (op 15203) 
s_add_u32 s7, s7, s8 // (op 15203) 

v_lshrrev_b32 v13, 7, v1 // (op 15203) 
s_add_u32 s8, s7, s9 // (op 15203) 

v_lshl_add_u32 v20, v13, 6, v21 // (op 15203) 


s_mul_hi_u32 s7, s8, s13 // (op 15203) 
v_lshrrev_b32 v13, 6, v3 // (op 15203) 

s_sub_u32 s28, s8, s7 // (op 15203) 
v_lshlrev_b32 v3, 2, v13 // (op 15203) 

s_lshr_b32 s29, s28, 1 // (op 15203) 

s_add_u32 s28, s29, s7 // (op 15203) 


s_lshr_b32 s7, s28, s11 // (op 15203) 

s_and_b32 s28, s11, 2147483648 // (op 15203) 
s_cmp_eq_i32 s28, 0 // (op 15203) 
s_mov_b32 s29, scc // (op 15203) 

s_cmp_eq_i32 s29, 0 // (op 15203) 
s_cselect_b32 s28, s8, s7 // (op 15203) 


s_mul_hi_u32 s7, s28, s12 // (op 15203) 
s_sub_u32 s29, s28, s7 // (op 15203) 
s_lshr_b32 s48, s29, 1 // (op 15203) 

s_add_u32 s29, s48, s7 // (op 15203) 


s_lshr_b32 s7, s29, s10 // (op 15203) 

s_and_b32 s29, s10, 2147483648 // (op 15203) 
s_cmp_eq_i32 s29, 0 // (op 15203) 
s_mov_b32 s48, scc // (op 15203) 

s_cmp_eq_i32 s48, 0 // (op 15203) 
s_cselect_b32 s29, s28, s7 // (op 15203) 


s_mul_i32 s7, s28, s18 // (op 15203) 


v_lshl_add_u32 v13, s29, 7, v20 // (op 15203) 


s_sub_u32 s18, s8, s7 // (op 15203) 


v_lshl_add_u32 v20, s18, 2, v3 // (op 15203) 


v_mul_lo_u32 v3, v13, s40 // (op 15203) 
v_add_u32 v3, v3, v20 // (op 15203) 


s_mul_i32 s19, s40, s35 // (op 15205) most significant: low of low * high
s_mul_i32 s7, s41, s34 // (op 15205) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 15205) least significant
s_mul_hi_u32 s8, s40, s34 // (op 15205) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 15205) most significant: sum
s_add_u32 s19, s19, s8 // (op 15205) most significant: sum


s_mov_b32 s52, s42 // (op 15205) reg expression
s_mov_b32 s53, s43 // (op 15205) reg expression
s_mov_b32 s54, s18 // (op 15205) reg expression
s_mov_b32 s55, 131072 // (op 15205) call()

v_and_b32 v13, 63, v1 // (op 15230) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 15230) 
v_lshrrev_b32 v20, 4, v13 // (op 15230) 
s_and_b32 s7, s18, 4294967294 // (op 15230) 
v_and_b32 v21, 3, v20 // (op 15230) 

s_add_u32 s8, s0, s7 // (op 15230) 

v_and_b32 v20, 15, v1 // (op 15230) 
v_lshl_add_u32 v72, v21, 4, v20 // (op 15230) 


s_mul_i32 s7, s16, s50 // (op 15230) 
s_add_u32 s7, s7, s8 // (op 15230) 

v_lshrrev_b32 v20, 6, v1 // (op 15230) 
s_add_u32 s8, s7, s9 // (op 15230) 

v_and_b32 v21, 1, v20 // (op 15230) 

s_mul_hi_u32 s7, s8, s13 // (op 15230) 
v_lshl_add_u32 v20, v21, 6, v72 // (op 15230) 


s_sub_u32 s28, s8, s7 // (op 15230) 
v_lshrrev_b32 v21, 6, v13 // (op 15230) 

s_lshr_b32 s29, s28, 1 // (op 15230) 

v_lshlrev_b32 v13, 2, v21 // (op 15230) 

s_add_u32 s28, s29, s7 // (op 15230) 


s_lshr_b32 s7, s28, s11 // (op 15230) 

s_and_b32 s28, s11, 2147483648 // (op 15230) 
s_cmp_eq_i32 s28, 0 // (op 15230) 
s_mov_b32 s29, scc // (op 15230) 

s_cmp_eq_i32 s29, 0 // (op 15230) 
s_cselect_b32 s28, s8, s7 // (op 15230) 


s_mul_hi_u32 s7, s28, s12 // (op 15230) 
s_sub_u32 s29, s28, s7 // (op 15230) 
s_lshr_b32 s48, s29, 1 // (op 15230) 

s_add_u32 s29, s48, s7 // (op 15230) 


s_lshr_b32 s7, s29, s10 // (op 15230) 

s_and_b32 s29, s10, 2147483648 // (op 15230) 
s_cmp_eq_i32 s29, 0 // (op 15230) 
s_mov_b32 s48, scc // (op 15230) 

s_cmp_eq_i32 s48, 0 // (op 15230) 
s_cselect_b32 s29, s28, s7 // (op 15230) 


s_mul_i32 s7, s29, s14 // (op 15230) 

s_sub_u32 s29, s28, s7 // (op 15230) 

v_lshl_add_u32 v21, s29, 7, v20 // (op 15230) 


s_mul_i32 s7, s28, s18 // (op 15230) 


v_mul_lo_u32 v20, v21, s44 // (op 15230) 

s_sub_u32 s18, s8, s7 // (op 15230) 


v_lshl_add_u32 v21, s18, 2, v13 // (op 15230) 


v_add_u32 v13, v21, v20 // (op 15230) 


s_mul_i32 s19, s44, s23 // (op 15232) most significant: low of low * high
s_mul_i32 s7, s45, s22 // (op 15232) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 15232) least significant
s_mul_hi_u32 s8, s44, s22 // (op 15232) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 15232) most significant: sum
s_add_u32 s19, s19, s8 // (op 15232) most significant: sum


s_mov_b32 s64, s46 // (op 15232) reg expression
s_mov_b32 s65, s47 // (op 15232) reg expression
s_mov_b32 s66, s18 // (op 15232) reg expression
s_mov_b32 s67, 131072 // (op 15232) call()

s_ashr_i64 s[18:19], s[2:3], 7 // (op 427) 
s_and_b32 s7, s18, 4294967294 // (op 427) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 426) 
s_mov_b32 s8, s18 // (op 426) convert

s_cmp_lt_u32 s7, s8 // (op 426) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_426 // (op 426) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_426 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_426:
s_barrier  // (op 15610) 
s_add_u32 s7, s7, 1 // (op 428) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 14084) 
v_mov_b64 v[16:17], s[18:19] // (op 14084) convert


v_mov_b64 v[18:19], v[22:23] // (op 14084) 
v_mov_b64 v[20:21], v[18:19] // (op 14084) 
s_add_u32 s8, 8192, s15 // (op 14084) 
s_mov_b32 m0, s8 // (op 14084) reg expression
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 14084) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 14084) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 14084) most significant half

s_add_u32 m0, m0, 4096 // (op 14084) 
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 14084) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 14084) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 14084) most significant half

s_add_u32 m0, m0, 4096 // (op 14084) 
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 14084) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 14084) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 14084) most significant half

s_add_u32 m0, m0, 4096 // (op 14084) 
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 14084) Load value direct to lds




v_add_co_u32 v22, vcc, 128, v22 // (op 14157) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 14157) most significant half
s_barrier  // (op 15629) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 14090) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 14090) 

v_lshrrev_b64 v[16:17], 3, s[28:29] // (op 14090) 


v_mov_b64 v[18:19], v[14:15] // (op 14090) 
v_mov_b64 v[20:21], v[18:19] // (op 14090) 
s_mov_b32 m0, s6 // (op 14090) reg expression
buffer_load_dwordx4 v20, s[72:75], 0 offen offset: 0 lds // (op 14090) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 14090) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 14090) most significant half

s_add_u32 m0, m0, 4096 // (op 14090) 
buffer_load_dwordx4 v20, s[72:75], 0 offen offset: 0 lds // (op 14090) Load value direct to lds



v_add_co_u32 v14, s[18:19], 64, v14 // (op 14325) least significant half
v_addc_co_u32 v15, s[18:19], 0, v15, s[18:19] // (op 14325) most significant half

s_waitcnt vmcnt(2)
s_barrier  // (op 15616) 



ds_read_b128 v[24:27], v5 offset:8192 // (op 434) Load local data 
ds_read_b128 v[28:31], v5 offset:8256 // (op 434) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 15635) 



ds_read_b128 v[16:19], v6 // (op 436) Load local data 


v_mov_b32 v20, v3 // (op 438) 
v_mov_b32 v21, v20 // (op 438) reg expression

buffer_load_dwordx2 v[32:33], v21, s[52:55], 0 offen // (op 438) Load value


v_add_u32 v3, 4, v3 // (op 15193) 


v_mov_b32 v20, v13 // (op 440) 
v_mov_b32 v21, v20 // (op 440) reg expression

buffer_load_dwordx2 v[34:35], v21, s[64:67], 0 offen // (op 440) Load value


v_add_u32 v13, 4, v13 // (op 15220) 
s_waitcnt vmcnt(1)
v_bfe_u32 v36, v32, 0, 8 // (op 13013) 



v_bfe_u32 v37, v32, 8, 8 // (op 13013) 
v_bfe_u32 v38, v32, 16, 8 // (op 13013) 
v_bfe_u32 v39, v32, 24, 8 // (op 13013) 
v_permlane16_swap_b32 v36, v37 // (op 13013) 
s_nop 0
v_permlane16_swap_b32 v38, v39 // (op 13013) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v36, v38 // (op 13013) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v37, v39 // (op 13013) 
ds_read_b128 v[40:43], v6 offset:1024 // (op 565) Load local data 




ds_read_b128 v[44:47], v6 offset:2048 // (op 701) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v48, v34, 0, 8 // (op 13043) 



v_bfe_u32 v49, v34, 8, 8 // (op 13043) 
v_bfe_u32 v50, v34, 16, 8 // (op 13043) 
v_bfe_u32 v51, v34, 24, 8 // (op 13043) 
v_permlane16_swap_b32 v48, v49 // (op 13043) 
ds_read_b128 v[52:55], v6 offset:3072 // (op 837) Load local data 
v_permlane16_swap_b32 v50, v51 // (op 13043) 



s_nop 1
v_permlane32_swap_b32 v48, v50 // (op 13043) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v49, v51 // (op 13043) 
ds_read_b128 v[56:59], v5 offset:10240 // (op 1059) Load local data 

ds_read_b128 v[60:63], v5 offset:10304 // (op 1059) Load local data 



s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:31], v[16:19], a[0:3], v36, v48 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 441) 
ds_read_b128 v[64:67], v5 offset:12288 // (op 1669) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:31], v[40:43], a[4:7], v36, v49 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 568) 
ds_read_b128 v[68:71], v5 offset:12352 // (op 1669) Load local data 



s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[24:31], v[44:47], a[8:11], v36, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 704) 
ds_read_b128 v[72:75], v5 offset:14336 // (op 2279) Load local data 
ds_read_b128 v[76:79], v5 offset:14400 // (op 2279) Load local data 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:31], v[52:55], a[12:15], v36, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 840) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:63], v[16:19], a[16:19], v37, v48 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1063) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:63], v[44:47], a[24:27], v37, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1117) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:63], v[52:55], a[28:31], v37, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1144) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:63], v[40:43], a[20:23], v37, v49 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1090) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:71], v[16:19], a[32:35], v38, v48 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1673) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[64:71], v[44:47], a[40:43], v38, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1727) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[64:71], v[52:55], a[44:47], v38, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1754) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:71], v[40:43], a[36:39], v38, v49 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1700) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[72:79], v[16:19], a[48:51], v39, v48 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2283) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[72:79], v[44:47], a[56:59], v39, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2337) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[72:79], v[52:55], a[60:63], v39, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2364) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[72:79], v[40:43], a[52:55], v39, v49 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2310) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 426) 
s_mov_b32 s8, s18 // (op 426) convert

s_cmp_lt_u32 s7, s8 // (op 426) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_426 // (op 426) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_426 if true)
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_426:









s_barrier  // (op 15746) 


v_cmp_eq_f32 vcc, 0.00000, v0 // (op 207) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_207 // (op 207) Condition: False, jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_207
v_accvgpr_read v12, a0 // (op 215) 
v_accvgpr_read v13, a1 // (op 215) 
v_accvgpr_read v14, a2 // (op 215) 
v_accvgpr_read v15, a3 // (op 215) 
v_mul_f32 v16, v11, v12 // (op 215) 
v_mul_f32 v17, v11, v13 // (op 215) 
v_mul_f32 v18, v11, v14 // (op 215) 
v_mul_f32 v19, v11, v15 // (op 215) 

v_lshrrev_b32 v12, 16, v16 // (op 217) 
v_lshrrev_b32 v13, 16, v17 // (op 217) 
v_lshrrev_b32 v14, 16, v18 // (op 217) 
v_lshrrev_b32 v15, 16, v19 // (op 217) 


v_pack_b32_f16 v16, v12, v13 // (op 218) 
v_pack_b32_f16 v17, v14, v15 // (op 218) 

v_mov_b32 v3, v4 // (op 218) reg expression

buffer_store_dwordx2 v[16:17], v3, s[60:63], 0 offen // (op 218) Store value



v_accvgpr_read v12, a4 // (op 3176) 
v_accvgpr_read v13, a5 // (op 3176) 
v_accvgpr_read v14, a6 // (op 3176) 
v_accvgpr_read v15, a7 // (op 3176) 
v_mul_f32 v16, v11, v12 // (op 3176) 
v_mul_f32 v17, v11, v13 // (op 3176) 
v_mul_f32 v18, v11, v14 // (op 3176) 
v_mul_f32 v19, v11, v15 // (op 3176) 

v_lshrrev_b32 v12, 16, v16 // (op 3178) 
v_lshrrev_b32 v13, 16, v17 // (op 3178) 
v_lshrrev_b32 v14, 16, v18 // (op 3178) 
v_lshrrev_b32 v15, 16, v19 // (op 3178) 


v_pack_b32_f16 v16, v12, v13 // (op 3179) 
v_pack_b32_f16 v17, v14, v15 // (op 3179) 

s_lshl_b32 s15, s24, 5 // (op 3179) 
v_add_u32 v3, v4, s15 // (op 3179) 


buffer_store_dwordx2 v[16:17], v3, s[60:63], 0 offen // (op 3179) Store value



v_accvgpr_read v12, a8 // (op 3190) 
v_accvgpr_read v13, a9 // (op 3190) 
v_accvgpr_read v14, a10 // (op 3190) 
v_accvgpr_read v15, a11 // (op 3190) 
v_mul_f32 v16, v11, v12 // (op 3190) 
v_mul_f32 v17, v11, v13 // (op 3190) 
v_mul_f32 v18, v11, v14 // (op 3190) 
v_mul_f32 v19, v11, v15 // (op 3190) 

v_lshrrev_b32 v12, 16, v16 // (op 3192) 
v_lshrrev_b32 v13, 16, v17 // (op 3192) 
v_lshrrev_b32 v14, 16, v18 // (op 3192) 
v_lshrrev_b32 v15, 16, v19 // (op 3192) 


v_pack_b32_f16 v16, v12, v13 // (op 3193) 
v_pack_b32_f16 v17, v14, v15 // (op 3193) 

s_lshl_b32 s15, s24, 6 // (op 3193) 
v_add_u32 v3, v4, s15 // (op 3193) 


buffer_store_dwordx2 v[16:17], v3, s[60:63], 0 offen // (op 3193) Store value



v_accvgpr_read v12, a12 // (op 3204) 
v_accvgpr_read v13, a13 // (op 3204) 
v_accvgpr_read v14, a14 // (op 3204) 
v_accvgpr_read v15, a15 // (op 3204) 
v_mul_f32 v16, v11, v12 // (op 3204) 
v_mul_f32 v17, v11, v13 // (op 3204) 
v_mul_f32 v18, v11, v14 // (op 3204) 
v_mul_f32 v19, v11, v15 // (op 3204) 

v_lshrrev_b32 v12, 16, v16 // (op 3206) 
v_lshrrev_b32 v13, 16, v17 // (op 3206) 
v_lshrrev_b32 v14, 16, v18 // (op 3206) 
v_lshrrev_b32 v15, 16, v19 // (op 3206) 


v_pack_b32_f16 v16, v12, v13 // (op 3207) 
v_pack_b32_f16 v17, v14, v15 // (op 3207) 

s_lshl_b32 s15, s24, 5 // (op 3207) 
s_mul_i32 s6, 3, s15 // (op 3207) 

v_add_u32 v3, v4, s6 // (op 3207) 


buffer_store_dwordx2 v[16:17], v3, s[60:63], 0 offen // (op 3207) Store value



v_accvgpr_read v12, a16 // (op 3240) 
v_accvgpr_read v13, a17 // (op 3240) 
v_accvgpr_read v14, a18 // (op 3240) 
v_accvgpr_read v15, a19 // (op 3240) 
v_mul_f32 v16, v11, v12 // (op 3240) 
v_mul_f32 v17, v11, v13 // (op 3240) 
v_mul_f32 v18, v11, v14 // (op 3240) 
v_mul_f32 v19, v11, v15 // (op 3240) 

v_lshrrev_b32 v12, 16, v16 // (op 3242) 
v_lshrrev_b32 v13, 16, v17 // (op 3242) 
v_lshrrev_b32 v14, 16, v18 // (op 3242) 
v_lshrrev_b32 v15, 16, v19 // (op 3242) 


v_pack_b32_f16 v16, v12, v13 // (op 3244) 
v_pack_b32_f16 v17, v14, v15 // (op 3244) 

v_add_u32 v3, 32, v4 // (op 3244) 

buffer_store_dwordx2 v[16:17], v3, s[60:63], 0 offen // (op 3244) Store value



v_accvgpr_read v12, a20 // (op 3254) 
v_accvgpr_read v13, a21 // (op 3254) 
v_accvgpr_read v14, a22 // (op 3254) 
v_accvgpr_read v15, a23 // (op 3254) 
v_mul_f32 v16, v11, v12 // (op 3254) 
v_mul_f32 v17, v11, v13 // (op 3254) 
v_mul_f32 v18, v11, v14 // (op 3254) 
v_mul_f32 v19, v11, v15 // (op 3254) 

v_lshrrev_b32 v12, 16, v16 // (op 3256) 
v_lshrrev_b32 v13, 16, v17 // (op 3256) 
v_lshrrev_b32 v14, 16, v18 // (op 3256) 
v_lshrrev_b32 v15, 16, v19 // (op 3256) 


v_pack_b32_f16 v16, v12, v13 // (op 3246) 
v_pack_b32_f16 v17, v14, v15 // (op 3246) 

s_lshl_b32 s15, s24, 5 // (op 3246) 
s_add_u32 s15, s15, 32 // (op 3246) 
v_add_u32 v3, v4, s15 // (op 3246) 


buffer_store_dwordx2 v[16:17], v3, s[60:63], 0 offen // (op 3246) Store value



v_accvgpr_read v12, a24 // (op 3260) 
v_accvgpr_read v13, a25 // (op 3260) 
v_accvgpr_read v14, a26 // (op 3260) 
v_accvgpr_read v15, a27 // (op 3260) 
v_mul_f32 v16, v11, v12 // (op 3260) 
v_mul_f32 v17, v11, v13 // (op 3260) 
v_mul_f32 v18, v11, v14 // (op 3260) 
v_mul_f32 v19, v11, v15 // (op 3260) 

v_lshrrev_b32 v12, 16, v16 // (op 3262) 
v_lshrrev_b32 v13, 16, v17 // (op 3262) 
v_lshrrev_b32 v14, 16, v18 // (op 3262) 
v_lshrrev_b32 v15, 16, v19 // (op 3262) 


v_pack_b32_f16 v16, v12, v13 // (op 3248) 
v_pack_b32_f16 v17, v14, v15 // (op 3248) 

s_lshl_b32 s15, s24, 6 // (op 3248) 
s_add_u32 s15, s15, 32 // (op 3248) 
v_add_u32 v3, v4, s15 // (op 3248) 


buffer_store_dwordx2 v[16:17], v3, s[60:63], 0 offen // (op 3248) Store value



v_accvgpr_read v12, a28 // (op 3266) 
v_accvgpr_read v13, a29 // (op 3266) 
v_accvgpr_read v14, a30 // (op 3266) 
v_accvgpr_read v15, a31 // (op 3266) 
v_mul_f32 v16, v11, v12 // (op 3266) 
v_mul_f32 v17, v11, v13 // (op 3266) 
v_mul_f32 v18, v11, v14 // (op 3266) 
v_mul_f32 v19, v11, v15 // (op 3266) 

v_lshrrev_b32 v12, 16, v16 // (op 3268) 
v_lshrrev_b32 v13, 16, v17 // (op 3268) 
v_lshrrev_b32 v14, 16, v18 // (op 3268) 
v_lshrrev_b32 v15, 16, v19 // (op 3268) 


v_pack_b32_f16 v16, v12, v13 // (op 3250) 
v_pack_b32_f16 v17, v14, v15 // (op 3250) 

s_lshl_b32 s15, s24, 5 // (op 3250) 
s_mul_i32 s6, 3, s15 // (op 3250) 
s_add_u32 s6, s6, 32 // (op 3250) 

v_add_u32 v3, v4, s6 // (op 3250) 


buffer_store_dwordx2 v[16:17], v3, s[60:63], 0 offen // (op 3250) Store value



v_accvgpr_read v12, a32 // (op 3316) 
v_accvgpr_read v13, a33 // (op 3316) 
v_accvgpr_read v14, a34 // (op 3316) 
v_accvgpr_read v15, a35 // (op 3316) 
v_mul_f32 v16, v11, v12 // (op 3316) 
v_mul_f32 v17, v11, v13 // (op 3316) 
v_mul_f32 v18, v11, v14 // (op 3316) 
v_mul_f32 v19, v11, v15 // (op 3316) 

v_lshrrev_b32 v12, 16, v16 // (op 3318) 
v_lshrrev_b32 v13, 16, v17 // (op 3318) 
v_lshrrev_b32 v14, 16, v18 // (op 3318) 
v_lshrrev_b32 v15, 16, v19 // (op 3318) 


v_pack_b32_f16 v16, v12, v13 // (op 3320) 
v_pack_b32_f16 v17, v14, v15 // (op 3320) 

v_add_u32 v3, 64, v4 // (op 3320) 

buffer_store_dwordx2 v[16:17], v3, s[60:63], 0 offen // (op 3320) Store value



v_accvgpr_read v12, a36 // (op 3330) 
v_accvgpr_read v13, a37 // (op 3330) 
v_accvgpr_read v14, a38 // (op 3330) 
v_accvgpr_read v15, a39 // (op 3330) 
v_mul_f32 v16, v11, v12 // (op 3330) 
v_mul_f32 v17, v11, v13 // (op 3330) 
v_mul_f32 v18, v11, v14 // (op 3330) 
v_mul_f32 v19, v11, v15 // (op 3330) 

v_lshrrev_b32 v12, 16, v16 // (op 3332) 
v_lshrrev_b32 v13, 16, v17 // (op 3332) 
v_lshrrev_b32 v14, 16, v18 // (op 3332) 
v_lshrrev_b32 v15, 16, v19 // (op 3332) 


v_pack_b32_f16 v16, v12, v13 // (op 3322) 
v_pack_b32_f16 v17, v14, v15 // (op 3322) 

s_lshl_b32 s15, s24, 5 // (op 3322) 
s_add_u32 s15, s15, 64 // (op 3322) 
v_add_u32 v3, v4, s15 // (op 3322) 


buffer_store_dwordx2 v[16:17], v3, s[60:63], 0 offen // (op 3322) Store value



v_accvgpr_read v12, a40 // (op 3336) 
v_accvgpr_read v13, a41 // (op 3336) 
v_accvgpr_read v14, a42 // (op 3336) 
v_accvgpr_read v15, a43 // (op 3336) 
v_mul_f32 v16, v11, v12 // (op 3336) 
v_mul_f32 v17, v11, v13 // (op 3336) 
v_mul_f32 v18, v11, v14 // (op 3336) 
v_mul_f32 v19, v11, v15 // (op 3336) 

v_lshrrev_b32 v12, 16, v16 // (op 3338) 
v_lshrrev_b32 v13, 16, v17 // (op 3338) 
v_lshrrev_b32 v14, 16, v18 // (op 3338) 
v_lshrrev_b32 v15, 16, v19 // (op 3338) 


v_pack_b32_f16 v16, v12, v13 // (op 3324) 
v_pack_b32_f16 v17, v14, v15 // (op 3324) 

s_lshl_b32 s15, s24, 6 // (op 3324) 
s_add_u32 s15, s15, 64 // (op 3324) 
v_add_u32 v3, v4, s15 // (op 3324) 


buffer_store_dwordx2 v[16:17], v3, s[60:63], 0 offen // (op 3324) Store value



v_accvgpr_read v12, a44 // (op 3342) 
v_accvgpr_read v13, a45 // (op 3342) 
v_accvgpr_read v14, a46 // (op 3342) 
v_accvgpr_read v15, a47 // (op 3342) 
v_mul_f32 v16, v11, v12 // (op 3342) 
v_mul_f32 v17, v11, v13 // (op 3342) 
v_mul_f32 v18, v11, v14 // (op 3342) 
v_mul_f32 v19, v11, v15 // (op 3342) 

v_lshrrev_b32 v12, 16, v16 // (op 3344) 
v_lshrrev_b32 v13, 16, v17 // (op 3344) 
v_lshrrev_b32 v14, 16, v18 // (op 3344) 
v_lshrrev_b32 v15, 16, v19 // (op 3344) 


v_pack_b32_f16 v16, v12, v13 // (op 3326) 
v_pack_b32_f16 v17, v14, v15 // (op 3326) 

s_lshl_b32 s15, s24, 5 // (op 3326) 
s_mul_i32 s6, 3, s15 // (op 3326) 
s_add_u32 s6, s6, 64 // (op 3326) 

v_add_u32 v3, v4, s6 // (op 3326) 


buffer_store_dwordx2 v[16:17], v3, s[60:63], 0 offen // (op 3326) Store value



v_accvgpr_read v12, a48 // (op 3392) 
v_accvgpr_read v13, a49 // (op 3392) 
v_accvgpr_read v14, a50 // (op 3392) 
v_accvgpr_read v15, a51 // (op 3392) 
v_mul_f32 v16, v11, v12 // (op 3392) 
v_mul_f32 v17, v11, v13 // (op 3392) 
v_mul_f32 v18, v11, v14 // (op 3392) 
v_mul_f32 v19, v11, v15 // (op 3392) 

v_lshrrev_b32 v12, 16, v16 // (op 3394) 
v_lshrrev_b32 v13, 16, v17 // (op 3394) 
v_lshrrev_b32 v14, 16, v18 // (op 3394) 
v_lshrrev_b32 v15, 16, v19 // (op 3394) 


v_pack_b32_f16 v16, v12, v13 // (op 3396) 
v_pack_b32_f16 v17, v14, v15 // (op 3396) 

v_add_u32 v3, 96, v4 // (op 3396) 

buffer_store_dwordx2 v[16:17], v3, s[60:63], 0 offen // (op 3396) Store value



v_accvgpr_read v12, a52 // (op 3406) 
v_accvgpr_read v13, a53 // (op 3406) 
v_accvgpr_read v14, a54 // (op 3406) 
v_accvgpr_read v15, a55 // (op 3406) 
v_mul_f32 v16, v11, v12 // (op 3406) 
v_mul_f32 v17, v11, v13 // (op 3406) 
v_mul_f32 v18, v11, v14 // (op 3406) 
v_mul_f32 v19, v11, v15 // (op 3406) 

v_lshrrev_b32 v12, 16, v16 // (op 3408) 
v_lshrrev_b32 v13, 16, v17 // (op 3408) 
v_lshrrev_b32 v14, 16, v18 // (op 3408) 
v_lshrrev_b32 v15, 16, v19 // (op 3408) 


v_pack_b32_f16 v16, v12, v13 // (op 3398) 
v_pack_b32_f16 v17, v14, v15 // (op 3398) 

s_lshl_b32 s15, s24, 5 // (op 3398) 
s_add_u32 s15, s15, 96 // (op 3398) 
v_add_u32 v3, v4, s15 // (op 3398) 


buffer_store_dwordx2 v[16:17], v3, s[60:63], 0 offen // (op 3398) Store value



v_accvgpr_read v12, a56 // (op 3412) 
v_accvgpr_read v13, a57 // (op 3412) 
v_accvgpr_read v14, a58 // (op 3412) 
v_accvgpr_read v15, a59 // (op 3412) 
v_mul_f32 v16, v11, v12 // (op 3412) 
v_mul_f32 v17, v11, v13 // (op 3412) 
v_mul_f32 v18, v11, v14 // (op 3412) 
v_mul_f32 v19, v11, v15 // (op 3412) 

v_lshrrev_b32 v12, 16, v16 // (op 3414) 
v_lshrrev_b32 v13, 16, v17 // (op 3414) 
v_lshrrev_b32 v14, 16, v18 // (op 3414) 
v_lshrrev_b32 v15, 16, v19 // (op 3414) 


v_pack_b32_f16 v16, v12, v13 // (op 3400) 
v_pack_b32_f16 v17, v14, v15 // (op 3400) 

s_lshl_b32 s15, s24, 6 // (op 3400) 
s_add_u32 s15, s15, 96 // (op 3400) 
v_add_u32 v3, v4, s15 // (op 3400) 


buffer_store_dwordx2 v[16:17], v3, s[60:63], 0 offen // (op 3400) Store value



v_accvgpr_read v12, a60 // (op 3418) 
v_accvgpr_read v13, a61 // (op 3418) 
v_accvgpr_read v14, a62 // (op 3418) 
v_accvgpr_read v15, a63 // (op 3418) 
v_mul_f32 v16, v11, v12 // (op 3418) 
v_mul_f32 v17, v11, v13 // (op 3418) 
v_mul_f32 v18, v11, v14 // (op 3418) 
v_mul_f32 v19, v11, v15 // (op 3418) 

v_lshrrev_b32 v12, 16, v16 // (op 3420) 
v_lshrrev_b32 v13, 16, v17 // (op 3420) 
v_lshrrev_b32 v14, 16, v18 // (op 3420) 
v_lshrrev_b32 v15, 16, v19 // (op 3420) 


v_pack_b32_f16 v16, v12, v13 // (op 3402) 
v_pack_b32_f16 v17, v14, v15 // (op 3402) 

s_lshl_b32 s15, s24, 5 // (op 3402) 
s_mul_i32 s6, 3, s15 // (op 3402) 
s_add_u32 s6, s6, 96 // (op 3402) 

v_add_u32 v3, v4, s6 // (op 3402) 


buffer_store_dwordx2 v[16:17], v3, s[60:63], 0 offen // (op 3402) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_207 // (op 207) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_207
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_207:


v_add_u32 v3, 64, v9 // (op 3703) 

buffer_load_dwordx2 v[12:13], v3, s[56:59], 0 offen // (op 3703) Load value

v_accvgpr_read v14, a32 // (op 3690) 
v_accvgpr_read v15, a33 // (op 3690) 
v_accvgpr_read v16, a34 // (op 3690) 
v_accvgpr_read v17, a35 // (op 3690) 
v_mul_f32 v18, v11, v14 // (op 3690) 
v_mul_f32 v19, v11, v15 // (op 3690) 
v_mul_f32 v20, v11, v16 // (op 3690) 
v_mul_f32 v21, v11, v17 // (op 3690) 

s_waitcnt vmcnt(0)
v_and_b32 v14, 65535, v12 // (op 3702) 
v_lshrrev_b32 v15, 16, v12 // (op 3702) 
v_lshlrev_b32 v14, 16, v14 // (op 3702) 
v_lshlrev_b32 v15, 16, v15 // (op 3702) 
v_mul_f32 v22, v0, v14 // (op 3702) 
v_mul_f32 v23, v0, v15 // (op 3702) 
v_and_b32 v14, 65535, v13 // (op 3702) 
v_lshrrev_b32 v15, 16, v13 // (op 3702) 
v_lshlrev_b32 v14, 16, v14 // (op 3702) 
v_lshlrev_b32 v15, 16, v15 // (op 3702) 
v_mul_f32 v24, v0, v14 // (op 3702) 
v_mul_f32 v25, v0, v15 // (op 3702) 


v_add_f32 v12, v22, v18 // (op 3691) 
v_add_f32 v13, v23, v19 // (op 3691) 
v_add_f32 v14, v24, v20 // (op 3691) 
v_add_f32 v15, v25, v21 // (op 3691) 


v_lshrrev_b32 v16, 16, v12 // (op 3692) 
v_lshrrev_b32 v17, 16, v13 // (op 3692) 
v_lshrrev_b32 v18, 16, v14 // (op 3692) 
v_lshrrev_b32 v19, 16, v15 // (op 3692) 


v_pack_b32_f16 v12, v16, v17 // (op 3694) 
v_pack_b32_f16 v13, v18, v19 // (op 3694) 

v_add_u32 v3, 64, v4 // (op 3694) 

buffer_store_dwordx2 v[12:13], v3, s[60:63], 0 offen // (op 3694) Store value





s_lshl_b32 s15, s4, 5 // (op 3708) 
s_add_u32 s15, s15, 64 // (op 3708) 
v_add_u32 v3, v9, s15 // (op 3708) 


buffer_load_dwordx2 v[12:13], v3, s[56:59], 0 offen // (op 3708) Load value

v_accvgpr_read v14, a36 // (op 3720) 
v_accvgpr_read v15, a37 // (op 3720) 
v_accvgpr_read v16, a38 // (op 3720) 
v_accvgpr_read v17, a39 // (op 3720) 
v_mul_f32 v18, v11, v14 // (op 3720) 
v_mul_f32 v19, v11, v15 // (op 3720) 
v_mul_f32 v20, v11, v16 // (op 3720) 
v_mul_f32 v21, v11, v17 // (op 3720) 

s_waitcnt vmcnt(0)
v_and_b32 v14, 65535, v12 // (op 3705) 
v_lshrrev_b32 v15, 16, v12 // (op 3705) 
v_lshlrev_b32 v14, 16, v14 // (op 3705) 
v_lshlrev_b32 v15, 16, v15 // (op 3705) 
v_mul_f32 v22, v0, v14 // (op 3705) 
v_mul_f32 v23, v0, v15 // (op 3705) 
v_and_b32 v14, 65535, v13 // (op 3705) 
v_lshrrev_b32 v15, 16, v13 // (op 3705) 
v_lshlrev_b32 v14, 16, v14 // (op 3705) 
v_lshlrev_b32 v15, 16, v15 // (op 3705) 
v_mul_f32 v24, v0, v14 // (op 3705) 
v_mul_f32 v25, v0, v15 // (op 3705) 


v_add_f32 v12, v22, v18 // (op 3706) 
v_add_f32 v13, v23, v19 // (op 3706) 
v_add_f32 v14, v24, v20 // (op 3706) 
v_add_f32 v15, v25, v21 // (op 3706) 


v_lshrrev_b32 v16, 16, v12 // (op 3707) 
v_lshrrev_b32 v17, 16, v13 // (op 3707) 
v_lshrrev_b32 v18, 16, v14 // (op 3707) 
v_lshrrev_b32 v19, 16, v15 // (op 3707) 


v_pack_b32_f16 v12, v16, v17 // (op 3696) 
v_pack_b32_f16 v13, v18, v19 // (op 3696) 

s_lshl_b32 s15, s24, 5 // (op 3696) 
s_add_u32 s15, s15, 64 // (op 3696) 
v_add_u32 v3, v4, s15 // (op 3696) 


buffer_store_dwordx2 v[12:13], v3, s[60:63], 0 offen // (op 3696) Store value





s_lshl_b32 s15, s4, 6 // (op 3713) 
s_add_u32 s15, s15, 64 // (op 3713) 
v_add_u32 v3, v9, s15 // (op 3713) 


buffer_load_dwordx2 v[12:13], v3, s[56:59], 0 offen // (op 3713) Load value

v_accvgpr_read v14, a40 // (op 3722) 
v_accvgpr_read v15, a41 // (op 3722) 
v_accvgpr_read v16, a42 // (op 3722) 
v_accvgpr_read v17, a43 // (op 3722) 
v_mul_f32 v18, v11, v14 // (op 3722) 
v_mul_f32 v19, v11, v15 // (op 3722) 
v_mul_f32 v20, v11, v16 // (op 3722) 
v_mul_f32 v21, v11, v17 // (op 3722) 

s_waitcnt vmcnt(0)
v_and_b32 v14, 65535, v12 // (op 3710) 
v_lshrrev_b32 v15, 16, v12 // (op 3710) 
v_lshlrev_b32 v14, 16, v14 // (op 3710) 
v_lshlrev_b32 v15, 16, v15 // (op 3710) 
v_mul_f32 v22, v0, v14 // (op 3710) 
v_mul_f32 v23, v0, v15 // (op 3710) 
v_and_b32 v14, 65535, v13 // (op 3710) 
v_lshrrev_b32 v15, 16, v13 // (op 3710) 
v_lshlrev_b32 v14, 16, v14 // (op 3710) 
v_lshlrev_b32 v15, 16, v15 // (op 3710) 
v_mul_f32 v24, v0, v14 // (op 3710) 
v_mul_f32 v25, v0, v15 // (op 3710) 


v_add_f32 v12, v22, v18 // (op 3711) 
v_add_f32 v13, v23, v19 // (op 3711) 
v_add_f32 v14, v24, v20 // (op 3711) 
v_add_f32 v15, v25, v21 // (op 3711) 


v_lshrrev_b32 v16, 16, v12 // (op 3712) 
v_lshrrev_b32 v17, 16, v13 // (op 3712) 
v_lshrrev_b32 v18, 16, v14 // (op 3712) 
v_lshrrev_b32 v19, 16, v15 // (op 3712) 


v_pack_b32_f16 v12, v16, v17 // (op 3698) 
v_pack_b32_f16 v13, v18, v19 // (op 3698) 

s_lshl_b32 s15, s24, 6 // (op 3698) 
s_add_u32 s15, s15, 64 // (op 3698) 
v_add_u32 v3, v4, s15 // (op 3698) 


buffer_store_dwordx2 v[12:13], v3, s[60:63], 0 offen // (op 3698) Store value





s_lshl_b32 s15, s4, 5 // (op 3718) 
s_mul_i32 s6, 3, s15 // (op 3718) 
s_add_u32 s6, s6, 64 // (op 3718) 

v_add_u32 v3, v9, s6 // (op 3718) 


buffer_load_dwordx2 v[12:13], v3, s[56:59], 0 offen // (op 3718) Load value

v_accvgpr_read v14, a44 // (op 3724) 
v_accvgpr_read v15, a45 // (op 3724) 
v_accvgpr_read v16, a46 // (op 3724) 
v_accvgpr_read v17, a47 // (op 3724) 
v_mul_f32 v18, v11, v14 // (op 3724) 
v_mul_f32 v19, v11, v15 // (op 3724) 
v_mul_f32 v20, v11, v16 // (op 3724) 
v_mul_f32 v21, v11, v17 // (op 3724) 

s_waitcnt vmcnt(0)
v_and_b32 v14, 65535, v12 // (op 3715) 
v_lshrrev_b32 v15, 16, v12 // (op 3715) 
v_lshlrev_b32 v14, 16, v14 // (op 3715) 
v_lshlrev_b32 v15, 16, v15 // (op 3715) 
v_mul_f32 v22, v0, v14 // (op 3715) 
v_mul_f32 v23, v0, v15 // (op 3715) 
v_and_b32 v14, 65535, v13 // (op 3715) 
v_lshrrev_b32 v15, 16, v13 // (op 3715) 
v_lshlrev_b32 v14, 16, v14 // (op 3715) 
v_lshlrev_b32 v15, 16, v15 // (op 3715) 
v_mul_f32 v24, v0, v14 // (op 3715) 
v_mul_f32 v25, v0, v15 // (op 3715) 


v_add_f32 v12, v22, v18 // (op 3716) 
v_add_f32 v13, v23, v19 // (op 3716) 
v_add_f32 v14, v24, v20 // (op 3716) 
v_add_f32 v15, v25, v21 // (op 3716) 


v_lshrrev_b32 v16, 16, v12 // (op 3717) 
v_lshrrev_b32 v17, 16, v13 // (op 3717) 
v_lshrrev_b32 v18, 16, v14 // (op 3717) 
v_lshrrev_b32 v19, 16, v15 // (op 3717) 


v_pack_b32_f16 v12, v16, v17 // (op 3700) 
v_pack_b32_f16 v13, v18, v19 // (op 3700) 

s_lshl_b32 s15, s24, 5 // (op 3700) 
s_mul_i32 s6, 3, s15 // (op 3700) 
s_add_u32 s6, s6, 64 // (op 3700) 

v_add_u32 v3, v4, s6 // (op 3700) 


buffer_store_dwordx2 v[12:13], v3, s[60:63], 0 offen // (op 3700) Store value





v_add_u32 v3, 32, v9 // (op 3616) 

buffer_load_dwordx2 v[12:13], v3, s[56:59], 0 offen // (op 3616) Load value

v_accvgpr_read v14, a16 // (op 3603) 
v_accvgpr_read v15, a17 // (op 3603) 
v_accvgpr_read v16, a18 // (op 3603) 
v_accvgpr_read v17, a19 // (op 3603) 
v_mul_f32 v18, v11, v14 // (op 3603) 
v_mul_f32 v19, v11, v15 // (op 3603) 
v_mul_f32 v20, v11, v16 // (op 3603) 
v_mul_f32 v21, v11, v17 // (op 3603) 

s_waitcnt vmcnt(0)
v_and_b32 v14, 65535, v12 // (op 3615) 
v_lshrrev_b32 v15, 16, v12 // (op 3615) 
v_lshlrev_b32 v14, 16, v14 // (op 3615) 
v_lshlrev_b32 v15, 16, v15 // (op 3615) 
v_mul_f32 v22, v0, v14 // (op 3615) 
v_mul_f32 v23, v0, v15 // (op 3615) 
v_and_b32 v14, 65535, v13 // (op 3615) 
v_lshrrev_b32 v15, 16, v13 // (op 3615) 
v_lshlrev_b32 v14, 16, v14 // (op 3615) 
v_lshlrev_b32 v15, 16, v15 // (op 3615) 
v_mul_f32 v24, v0, v14 // (op 3615) 
v_mul_f32 v25, v0, v15 // (op 3615) 


v_add_f32 v12, v22, v18 // (op 3604) 
v_add_f32 v13, v23, v19 // (op 3604) 
v_add_f32 v14, v24, v20 // (op 3604) 
v_add_f32 v15, v25, v21 // (op 3604) 


v_lshrrev_b32 v16, 16, v12 // (op 3605) 
v_lshrrev_b32 v17, 16, v13 // (op 3605) 
v_lshrrev_b32 v18, 16, v14 // (op 3605) 
v_lshrrev_b32 v19, 16, v15 // (op 3605) 


v_pack_b32_f16 v12, v16, v17 // (op 3607) 
v_pack_b32_f16 v13, v18, v19 // (op 3607) 

v_add_u32 v3, 32, v4 // (op 3607) 

buffer_store_dwordx2 v[12:13], v3, s[60:63], 0 offen // (op 3607) Store value





s_lshl_b32 s15, s4, 5 // (op 3621) 
s_add_u32 s15, s15, 32 // (op 3621) 
v_add_u32 v3, v9, s15 // (op 3621) 


buffer_load_dwordx2 v[12:13], v3, s[56:59], 0 offen // (op 3621) Load value

v_accvgpr_read v14, a20 // (op 3633) 
v_accvgpr_read v15, a21 // (op 3633) 
v_accvgpr_read v16, a22 // (op 3633) 
v_accvgpr_read v17, a23 // (op 3633) 
v_mul_f32 v18, v11, v14 // (op 3633) 
v_mul_f32 v19, v11, v15 // (op 3633) 
v_mul_f32 v20, v11, v16 // (op 3633) 
v_mul_f32 v21, v11, v17 // (op 3633) 

s_waitcnt vmcnt(0)
v_and_b32 v14, 65535, v12 // (op 3618) 
v_lshrrev_b32 v15, 16, v12 // (op 3618) 
v_lshlrev_b32 v14, 16, v14 // (op 3618) 
v_lshlrev_b32 v15, 16, v15 // (op 3618) 
v_mul_f32 v22, v0, v14 // (op 3618) 
v_mul_f32 v23, v0, v15 // (op 3618) 
v_and_b32 v14, 65535, v13 // (op 3618) 
v_lshrrev_b32 v15, 16, v13 // (op 3618) 
v_lshlrev_b32 v14, 16, v14 // (op 3618) 
v_lshlrev_b32 v15, 16, v15 // (op 3618) 
v_mul_f32 v24, v0, v14 // (op 3618) 
v_mul_f32 v25, v0, v15 // (op 3618) 


v_add_f32 v12, v22, v18 // (op 3619) 
v_add_f32 v13, v23, v19 // (op 3619) 
v_add_f32 v14, v24, v20 // (op 3619) 
v_add_f32 v15, v25, v21 // (op 3619) 


v_lshrrev_b32 v16, 16, v12 // (op 3620) 
v_lshrrev_b32 v17, 16, v13 // (op 3620) 
v_lshrrev_b32 v18, 16, v14 // (op 3620) 
v_lshrrev_b32 v19, 16, v15 // (op 3620) 


v_pack_b32_f16 v12, v16, v17 // (op 3609) 
v_pack_b32_f16 v13, v18, v19 // (op 3609) 

s_lshl_b32 s15, s24, 5 // (op 3609) 
s_add_u32 s15, s15, 32 // (op 3609) 
v_add_u32 v3, v4, s15 // (op 3609) 


buffer_store_dwordx2 v[12:13], v3, s[60:63], 0 offen // (op 3609) Store value





s_lshl_b32 s15, s4, 6 // (op 3626) 
s_add_u32 s15, s15, 32 // (op 3626) 
v_add_u32 v3, v9, s15 // (op 3626) 


buffer_load_dwordx2 v[12:13], v3, s[56:59], 0 offen // (op 3626) Load value

v_accvgpr_read v14, a24 // (op 3635) 
v_accvgpr_read v15, a25 // (op 3635) 
v_accvgpr_read v16, a26 // (op 3635) 
v_accvgpr_read v17, a27 // (op 3635) 
v_mul_f32 v18, v11, v14 // (op 3635) 
v_mul_f32 v19, v11, v15 // (op 3635) 
v_mul_f32 v20, v11, v16 // (op 3635) 
v_mul_f32 v21, v11, v17 // (op 3635) 

s_waitcnt vmcnt(0)
v_and_b32 v14, 65535, v12 // (op 3623) 
v_lshrrev_b32 v15, 16, v12 // (op 3623) 
v_lshlrev_b32 v14, 16, v14 // (op 3623) 
v_lshlrev_b32 v15, 16, v15 // (op 3623) 
v_mul_f32 v22, v0, v14 // (op 3623) 
v_mul_f32 v23, v0, v15 // (op 3623) 
v_and_b32 v14, 65535, v13 // (op 3623) 
v_lshrrev_b32 v15, 16, v13 // (op 3623) 
v_lshlrev_b32 v14, 16, v14 // (op 3623) 
v_lshlrev_b32 v15, 16, v15 // (op 3623) 
v_mul_f32 v24, v0, v14 // (op 3623) 
v_mul_f32 v25, v0, v15 // (op 3623) 


v_add_f32 v12, v22, v18 // (op 3624) 
v_add_f32 v13, v23, v19 // (op 3624) 
v_add_f32 v14, v24, v20 // (op 3624) 
v_add_f32 v15, v25, v21 // (op 3624) 


v_lshrrev_b32 v16, 16, v12 // (op 3625) 
v_lshrrev_b32 v17, 16, v13 // (op 3625) 
v_lshrrev_b32 v18, 16, v14 // (op 3625) 
v_lshrrev_b32 v19, 16, v15 // (op 3625) 


v_pack_b32_f16 v12, v16, v17 // (op 3611) 
v_pack_b32_f16 v13, v18, v19 // (op 3611) 

s_lshl_b32 s15, s24, 6 // (op 3611) 
s_add_u32 s15, s15, 32 // (op 3611) 
v_add_u32 v3, v4, s15 // (op 3611) 


buffer_store_dwordx2 v[12:13], v3, s[60:63], 0 offen // (op 3611) Store value





s_lshl_b32 s15, s4, 5 // (op 3631) 
s_mul_i32 s6, 3, s15 // (op 3631) 
s_add_u32 s6, s6, 32 // (op 3631) 

v_add_u32 v3, v9, s6 // (op 3631) 


buffer_load_dwordx2 v[12:13], v3, s[56:59], 0 offen // (op 3631) Load value

v_accvgpr_read v14, a28 // (op 3637) 
v_accvgpr_read v15, a29 // (op 3637) 
v_accvgpr_read v16, a30 // (op 3637) 
v_accvgpr_read v17, a31 // (op 3637) 
v_mul_f32 v18, v11, v14 // (op 3637) 
v_mul_f32 v19, v11, v15 // (op 3637) 
v_mul_f32 v20, v11, v16 // (op 3637) 
v_mul_f32 v21, v11, v17 // (op 3637) 

s_waitcnt vmcnt(0)
v_and_b32 v14, 65535, v12 // (op 3628) 
v_lshrrev_b32 v15, 16, v12 // (op 3628) 
v_lshlrev_b32 v14, 16, v14 // (op 3628) 
v_lshlrev_b32 v15, 16, v15 // (op 3628) 
v_mul_f32 v22, v0, v14 // (op 3628) 
v_mul_f32 v23, v0, v15 // (op 3628) 
v_and_b32 v14, 65535, v13 // (op 3628) 
v_lshrrev_b32 v15, 16, v13 // (op 3628) 
v_lshlrev_b32 v14, 16, v14 // (op 3628) 
v_lshlrev_b32 v15, 16, v15 // (op 3628) 
v_mul_f32 v24, v0, v14 // (op 3628) 
v_mul_f32 v25, v0, v15 // (op 3628) 


v_add_f32 v12, v22, v18 // (op 3629) 
v_add_f32 v13, v23, v19 // (op 3629) 
v_add_f32 v14, v24, v20 // (op 3629) 
v_add_f32 v15, v25, v21 // (op 3629) 


v_lshrrev_b32 v16, 16, v12 // (op 3630) 
v_lshrrev_b32 v17, 16, v13 // (op 3630) 
v_lshrrev_b32 v18, 16, v14 // (op 3630) 
v_lshrrev_b32 v19, 16, v15 // (op 3630) 


v_pack_b32_f16 v12, v16, v17 // (op 3613) 
v_pack_b32_f16 v13, v18, v19 // (op 3613) 

s_lshl_b32 s15, s24, 5 // (op 3613) 
s_mul_i32 s6, 3, s15 // (op 3613) 
s_add_u32 s6, s6, 32 // (op 3613) 

v_add_u32 v3, v4, s6 // (op 3613) 


buffer_store_dwordx2 v[12:13], v3, s[60:63], 0 offen // (op 3613) Store value





v_add_u32 v3, 96, v9 // (op 3790) 

buffer_load_dwordx2 v[12:13], v3, s[56:59], 0 offen // (op 3790) Load value

v_accvgpr_read v14, a48 // (op 3777) 
v_accvgpr_read v15, a49 // (op 3777) 
v_accvgpr_read v16, a50 // (op 3777) 
v_accvgpr_read v17, a51 // (op 3777) 
v_mul_f32 v18, v11, v14 // (op 3777) 
v_mul_f32 v19, v11, v15 // (op 3777) 
v_mul_f32 v20, v11, v16 // (op 3777) 
v_mul_f32 v21, v11, v17 // (op 3777) 

s_waitcnt vmcnt(0)
v_and_b32 v14, 65535, v12 // (op 3789) 
v_lshrrev_b32 v15, 16, v12 // (op 3789) 
v_lshlrev_b32 v14, 16, v14 // (op 3789) 
v_lshlrev_b32 v15, 16, v15 // (op 3789) 
v_mul_f32 v22, v0, v14 // (op 3789) 
v_mul_f32 v23, v0, v15 // (op 3789) 
v_and_b32 v14, 65535, v13 // (op 3789) 
v_lshrrev_b32 v15, 16, v13 // (op 3789) 
v_lshlrev_b32 v14, 16, v14 // (op 3789) 
v_lshlrev_b32 v15, 16, v15 // (op 3789) 
v_mul_f32 v24, v0, v14 // (op 3789) 
v_mul_f32 v25, v0, v15 // (op 3789) 


v_add_f32 v12, v22, v18 // (op 3778) 
v_add_f32 v13, v23, v19 // (op 3778) 
v_add_f32 v14, v24, v20 // (op 3778) 
v_add_f32 v15, v25, v21 // (op 3778) 


v_lshrrev_b32 v16, 16, v12 // (op 3779) 
v_lshrrev_b32 v17, 16, v13 // (op 3779) 
v_lshrrev_b32 v18, 16, v14 // (op 3779) 
v_lshrrev_b32 v19, 16, v15 // (op 3779) 


v_pack_b32_f16 v12, v16, v17 // (op 3781) 
v_pack_b32_f16 v13, v18, v19 // (op 3781) 

v_add_u32 v3, 96, v4 // (op 3781) 

buffer_store_dwordx2 v[12:13], v3, s[60:63], 0 offen // (op 3781) Store value





s_lshl_b32 s15, s4, 5 // (op 3795) 
s_add_u32 s15, s15, 96 // (op 3795) 
v_add_u32 v3, v9, s15 // (op 3795) 


buffer_load_dwordx2 v[12:13], v3, s[56:59], 0 offen // (op 3795) Load value

v_accvgpr_read v14, a52 // (op 3807) 
v_accvgpr_read v15, a53 // (op 3807) 
v_accvgpr_read v16, a54 // (op 3807) 
v_accvgpr_read v17, a55 // (op 3807) 
v_mul_f32 v18, v11, v14 // (op 3807) 
v_mul_f32 v19, v11, v15 // (op 3807) 
v_mul_f32 v20, v11, v16 // (op 3807) 
v_mul_f32 v21, v11, v17 // (op 3807) 

s_waitcnt vmcnt(0)
v_and_b32 v14, 65535, v12 // (op 3792) 
v_lshrrev_b32 v15, 16, v12 // (op 3792) 
v_lshlrev_b32 v14, 16, v14 // (op 3792) 
v_lshlrev_b32 v15, 16, v15 // (op 3792) 
v_mul_f32 v22, v0, v14 // (op 3792) 
v_mul_f32 v23, v0, v15 // (op 3792) 
v_and_b32 v14, 65535, v13 // (op 3792) 
v_lshrrev_b32 v15, 16, v13 // (op 3792) 
v_lshlrev_b32 v14, 16, v14 // (op 3792) 
v_lshlrev_b32 v15, 16, v15 // (op 3792) 
v_mul_f32 v24, v0, v14 // (op 3792) 
v_mul_f32 v25, v0, v15 // (op 3792) 


v_add_f32 v12, v22, v18 // (op 3793) 
v_add_f32 v13, v23, v19 // (op 3793) 
v_add_f32 v14, v24, v20 // (op 3793) 
v_add_f32 v15, v25, v21 // (op 3793) 


v_lshrrev_b32 v16, 16, v12 // (op 3794) 
v_lshrrev_b32 v17, 16, v13 // (op 3794) 
v_lshrrev_b32 v18, 16, v14 // (op 3794) 
v_lshrrev_b32 v19, 16, v15 // (op 3794) 


v_pack_b32_f16 v12, v16, v17 // (op 3783) 
v_pack_b32_f16 v13, v18, v19 // (op 3783) 

s_lshl_b32 s15, s24, 5 // (op 3783) 
s_add_u32 s15, s15, 96 // (op 3783) 
v_add_u32 v3, v4, s15 // (op 3783) 


buffer_store_dwordx2 v[12:13], v3, s[60:63], 0 offen // (op 3783) Store value





s_lshl_b32 s15, s4, 6 // (op 3800) 
s_add_u32 s15, s15, 96 // (op 3800) 
v_add_u32 v3, v9, s15 // (op 3800) 


buffer_load_dwordx2 v[12:13], v3, s[56:59], 0 offen // (op 3800) Load value

v_accvgpr_read v14, a56 // (op 3809) 
v_accvgpr_read v15, a57 // (op 3809) 
v_accvgpr_read v16, a58 // (op 3809) 
v_accvgpr_read v17, a59 // (op 3809) 
v_mul_f32 v18, v11, v14 // (op 3809) 
v_mul_f32 v19, v11, v15 // (op 3809) 
v_mul_f32 v20, v11, v16 // (op 3809) 
v_mul_f32 v21, v11, v17 // (op 3809) 

s_waitcnt vmcnt(0)
v_and_b32 v14, 65535, v12 // (op 3797) 
v_lshrrev_b32 v15, 16, v12 // (op 3797) 
v_lshlrev_b32 v14, 16, v14 // (op 3797) 
v_lshlrev_b32 v15, 16, v15 // (op 3797) 
v_mul_f32 v22, v0, v14 // (op 3797) 
v_mul_f32 v23, v0, v15 // (op 3797) 
v_and_b32 v14, 65535, v13 // (op 3797) 
v_lshrrev_b32 v15, 16, v13 // (op 3797) 
v_lshlrev_b32 v14, 16, v14 // (op 3797) 
v_lshlrev_b32 v15, 16, v15 // (op 3797) 
v_mul_f32 v24, v0, v14 // (op 3797) 
v_mul_f32 v25, v0, v15 // (op 3797) 


v_add_f32 v12, v22, v18 // (op 3798) 
v_add_f32 v13, v23, v19 // (op 3798) 
v_add_f32 v14, v24, v20 // (op 3798) 
v_add_f32 v15, v25, v21 // (op 3798) 


v_lshrrev_b32 v16, 16, v12 // (op 3799) 
v_lshrrev_b32 v17, 16, v13 // (op 3799) 
v_lshrrev_b32 v18, 16, v14 // (op 3799) 
v_lshrrev_b32 v19, 16, v15 // (op 3799) 


v_pack_b32_f16 v12, v16, v17 // (op 3785) 
v_pack_b32_f16 v13, v18, v19 // (op 3785) 

s_lshl_b32 s15, s24, 6 // (op 3785) 
s_add_u32 s15, s15, 96 // (op 3785) 
v_add_u32 v3, v4, s15 // (op 3785) 


buffer_store_dwordx2 v[12:13], v3, s[60:63], 0 offen // (op 3785) Store value





s_lshl_b32 s15, s4, 5 // (op 3805) 
s_mul_i32 s6, 3, s15 // (op 3805) 
s_add_u32 s6, s6, 96 // (op 3805) 

v_add_u32 v3, v9, s6 // (op 3805) 


buffer_load_dwordx2 v[12:13], v3, s[56:59], 0 offen // (op 3805) Load value

v_accvgpr_read v14, a60 // (op 3811) 
v_accvgpr_read v15, a61 // (op 3811) 
v_accvgpr_read v16, a62 // (op 3811) 
v_accvgpr_read v17, a63 // (op 3811) 
v_mul_f32 v18, v11, v14 // (op 3811) 
v_mul_f32 v19, v11, v15 // (op 3811) 
v_mul_f32 v20, v11, v16 // (op 3811) 
v_mul_f32 v21, v11, v17 // (op 3811) 

s_waitcnt vmcnt(0)
v_and_b32 v14, 65535, v12 // (op 3802) 
v_lshrrev_b32 v15, 16, v12 // (op 3802) 
v_lshlrev_b32 v14, 16, v14 // (op 3802) 
v_lshlrev_b32 v15, 16, v15 // (op 3802) 
v_mul_f32 v22, v0, v14 // (op 3802) 
v_mul_f32 v23, v0, v15 // (op 3802) 
v_and_b32 v14, 65535, v13 // (op 3802) 
v_lshrrev_b32 v15, 16, v13 // (op 3802) 
v_lshlrev_b32 v14, 16, v14 // (op 3802) 
v_lshlrev_b32 v15, 16, v15 // (op 3802) 
v_mul_f32 v24, v0, v14 // (op 3802) 
v_mul_f32 v25, v0, v15 // (op 3802) 


v_add_f32 v12, v22, v18 // (op 3803) 
v_add_f32 v13, v23, v19 // (op 3803) 
v_add_f32 v14, v24, v20 // (op 3803) 
v_add_f32 v15, v25, v21 // (op 3803) 


v_lshrrev_b32 v16, 16, v12 // (op 3804) 
v_lshrrev_b32 v17, 16, v13 // (op 3804) 
v_lshrrev_b32 v18, 16, v14 // (op 3804) 
v_lshrrev_b32 v19, 16, v15 // (op 3804) 


v_pack_b32_f16 v12, v16, v17 // (op 3787) 
v_pack_b32_f16 v13, v18, v19 // (op 3787) 

s_lshl_b32 s15, s24, 5 // (op 3787) 
s_mul_i32 s6, 3, s15 // (op 3787) 
s_add_u32 s6, s6, 96 // (op 3787) 

v_add_u32 v3, v4, s6 // (op 3787) 


buffer_store_dwordx2 v[12:13], v3, s[60:63], 0 offen // (op 3787) Store value





v_mov_b32 v3, v9 // (op 232) reg expression

buffer_load_dwordx2 v[12:13], v3, s[56:59], 0 offen // (op 232) Load value

v_accvgpr_read v14, a0 // (op 228) 
v_accvgpr_read v15, a1 // (op 228) 
v_accvgpr_read v16, a2 // (op 228) 
v_accvgpr_read v17, a3 // (op 228) 
v_mul_f32 v18, v11, v14 // (op 228) 
v_mul_f32 v19, v11, v15 // (op 228) 
v_mul_f32 v20, v11, v16 // (op 228) 
v_mul_f32 v21, v11, v17 // (op 228) 

s_waitcnt vmcnt(0)
v_and_b32 v14, 65535, v12 // (op 233) 
v_lshrrev_b32 v15, 16, v12 // (op 233) 
v_lshlrev_b32 v14, 16, v14 // (op 233) 
v_lshlrev_b32 v15, 16, v15 // (op 233) 
v_mul_f32 v22, v0, v14 // (op 233) 
v_mul_f32 v23, v0, v15 // (op 233) 
v_and_b32 v14, 65535, v13 // (op 233) 
v_lshrrev_b32 v15, 16, v13 // (op 233) 
v_lshlrev_b32 v14, 16, v14 // (op 233) 
v_lshlrev_b32 v15, 16, v15 // (op 233) 
v_mul_f32 v24, v0, v14 // (op 233) 
v_mul_f32 v25, v0, v15 // (op 233) 


v_add_f32 v12, v22, v18 // (op 229) 
v_add_f32 v13, v23, v19 // (op 229) 
v_add_f32 v14, v24, v20 // (op 229) 
v_add_f32 v15, v25, v21 // (op 229) 


v_lshrrev_b32 v16, 16, v12 // (op 230) 
v_lshrrev_b32 v17, 16, v13 // (op 230) 
v_lshrrev_b32 v18, 16, v14 // (op 230) 
v_lshrrev_b32 v19, 16, v15 // (op 230) 


v_pack_b32_f16 v12, v16, v17 // (op 231) 
v_pack_b32_f16 v13, v18, v19 // (op 231) 

v_mov_b32 v3, v4 // (op 231) reg expression

buffer_store_dwordx2 v[12:13], v3, s[60:63], 0 offen // (op 231) Store value





s_lshl_b32 s15, s4, 5 // (op 3524) 
v_add_u32 v3, v9, s15 // (op 3524) 


buffer_load_dwordx2 v[12:13], v3, s[56:59], 0 offen // (op 3524) Load value

v_accvgpr_read v14, a4 // (op 3520) 
v_accvgpr_read v15, a5 // (op 3520) 
v_accvgpr_read v16, a6 // (op 3520) 
v_accvgpr_read v17, a7 // (op 3520) 
v_mul_f32 v18, v11, v14 // (op 3520) 
v_mul_f32 v19, v11, v15 // (op 3520) 
v_mul_f32 v20, v11, v16 // (op 3520) 
v_mul_f32 v21, v11, v17 // (op 3520) 

s_waitcnt vmcnt(0)
v_and_b32 v14, 65535, v12 // (op 3525) 
v_lshrrev_b32 v15, 16, v12 // (op 3525) 
v_lshlrev_b32 v14, 16, v14 // (op 3525) 
v_lshlrev_b32 v15, 16, v15 // (op 3525) 
v_mul_f32 v22, v0, v14 // (op 3525) 
v_mul_f32 v23, v0, v15 // (op 3525) 
v_and_b32 v14, 65535, v13 // (op 3525) 
v_lshrrev_b32 v15, 16, v13 // (op 3525) 
v_lshlrev_b32 v14, 16, v14 // (op 3525) 
v_lshlrev_b32 v15, 16, v15 // (op 3525) 
v_mul_f32 v24, v0, v14 // (op 3525) 
v_mul_f32 v25, v0, v15 // (op 3525) 


v_add_f32 v12, v22, v18 // (op 3521) 
v_add_f32 v13, v23, v19 // (op 3521) 
v_add_f32 v14, v24, v20 // (op 3521) 
v_add_f32 v15, v25, v21 // (op 3521) 


v_lshrrev_b32 v16, 16, v12 // (op 3522) 
v_lshrrev_b32 v17, 16, v13 // (op 3522) 
v_lshrrev_b32 v18, 16, v14 // (op 3522) 
v_lshrrev_b32 v19, 16, v15 // (op 3522) 


v_pack_b32_f16 v12, v16, v17 // (op 3523) 
v_pack_b32_f16 v13, v18, v19 // (op 3523) 

s_lshl_b32 s15, s24, 5 // (op 3523) 
v_add_u32 v3, v4, s15 // (op 3523) 


buffer_store_dwordx2 v[12:13], v3, s[60:63], 0 offen // (op 3523) Store value





s_lshl_b32 s15, s4, 6 // (op 3538) 
v_add_u32 v3, v9, s15 // (op 3538) 


buffer_load_dwordx2 v[12:13], v3, s[56:59], 0 offen // (op 3538) Load value

v_accvgpr_read v14, a8 // (op 3534) 
v_accvgpr_read v15, a9 // (op 3534) 
v_accvgpr_read v16, a10 // (op 3534) 
v_accvgpr_read v17, a11 // (op 3534) 
v_mul_f32 v18, v11, v14 // (op 3534) 
v_mul_f32 v19, v11, v15 // (op 3534) 
v_mul_f32 v20, v11, v16 // (op 3534) 
v_mul_f32 v21, v11, v17 // (op 3534) 

s_waitcnt vmcnt(0)
v_and_b32 v14, 65535, v12 // (op 3539) 
v_lshrrev_b32 v15, 16, v12 // (op 3539) 
v_lshlrev_b32 v14, 16, v14 // (op 3539) 
v_lshlrev_b32 v15, 16, v15 // (op 3539) 
v_mul_f32 v22, v0, v14 // (op 3539) 
v_mul_f32 v23, v0, v15 // (op 3539) 
v_and_b32 v14, 65535, v13 // (op 3539) 
v_lshrrev_b32 v15, 16, v13 // (op 3539) 
v_lshlrev_b32 v14, 16, v14 // (op 3539) 
v_lshlrev_b32 v15, 16, v15 // (op 3539) 
v_mul_f32 v24, v0, v14 // (op 3539) 
v_mul_f32 v25, v0, v15 // (op 3539) 


v_add_f32 v12, v22, v18 // (op 3535) 
v_add_f32 v13, v23, v19 // (op 3535) 
v_add_f32 v14, v24, v20 // (op 3535) 
v_add_f32 v15, v25, v21 // (op 3535) 


v_lshrrev_b32 v16, 16, v12 // (op 3536) 
v_lshrrev_b32 v17, 16, v13 // (op 3536) 
v_lshrrev_b32 v18, 16, v14 // (op 3536) 
v_lshrrev_b32 v19, 16, v15 // (op 3536) 


v_pack_b32_f16 v12, v16, v17 // (op 3537) 
v_pack_b32_f16 v13, v18, v19 // (op 3537) 

s_lshl_b32 s15, s24, 6 // (op 3537) 
v_add_u32 v3, v4, s15 // (op 3537) 


buffer_store_dwordx2 v[12:13], v3, s[60:63], 0 offen // (op 3537) Store value





s_lshl_b32 s15, s4, 5 // (op 3552) 
s_mul_i32 s6, 3, s15 // (op 3552) 

v_add_u32 v3, v9, s6 // (op 3552) 


buffer_load_dwordx2 v[12:13], v3, s[56:59], 0 offen // (op 3552) Load value

v_accvgpr_read v14, a12 // (op 3548) 
v_accvgpr_read v15, a13 // (op 3548) 
v_accvgpr_read v16, a14 // (op 3548) 
v_accvgpr_read v17, a15 // (op 3548) 
v_mul_f32 v18, v11, v14 // (op 3548) 
v_mul_f32 v19, v11, v15 // (op 3548) 
v_mul_f32 v20, v11, v16 // (op 3548) 
v_mul_f32 v21, v11, v17 // (op 3548) 

s_waitcnt vmcnt(0)
v_and_b32 v14, 65535, v12 // (op 3553) 
v_lshrrev_b32 v15, 16, v12 // (op 3553) 
v_lshlrev_b32 v14, 16, v14 // (op 3553) 
v_lshlrev_b32 v15, 16, v15 // (op 3553) 
v_mul_f32 v22, v0, v14 // (op 3553) 
v_mul_f32 v23, v0, v15 // (op 3553) 
v_and_b32 v14, 65535, v13 // (op 3553) 
v_lshrrev_b32 v15, 16, v13 // (op 3553) 
v_lshlrev_b32 v14, 16, v14 // (op 3553) 
v_lshlrev_b32 v15, 16, v15 // (op 3553) 
v_mul_f32 v24, v0, v14 // (op 3553) 
v_mul_f32 v25, v0, v15 // (op 3553) 


v_add_f32 v12, v22, v18 // (op 3549) 
v_add_f32 v13, v23, v19 // (op 3549) 
v_add_f32 v14, v24, v20 // (op 3549) 
v_add_f32 v15, v25, v21 // (op 3549) 


v_lshrrev_b32 v16, 16, v12 // (op 3550) 
v_lshrrev_b32 v17, 16, v13 // (op 3550) 
v_lshrrev_b32 v18, 16, v14 // (op 3550) 
v_lshrrev_b32 v19, 16, v15 // (op 3550) 


v_pack_b32_f16 v12, v16, v17 // (op 3551) 
v_pack_b32_f16 v13, v18, v19 // (op 3551) 

s_lshl_b32 s15, s24, 5 // (op 3551) 
s_mul_i32 s6, 3, s15 // (op 3551) 

v_add_u32 v3, v4, s6 // (op 3551) 


buffer_store_dwordx2 v[12:13], v3, s[60:63], 0 offen // (op 3551) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_207:
s_add_u32 s0, s0, s1 // (op 420) 


s_mul_i32 s15, s50, s16 // (op 418) 
s_cmp_gt_u32 s51, s15 // (op 418) 
s_mov_b32 s6, scc // (op 418) 
s_sub_u32 s7, s51, s15 // (op 418) 

s_cmp_eq_i32 s6, 0 // (op 418) 
s_cselect_b32 s8, 0, s7 // (op 418) 


s_cmp_lt_u32 s50, s8 // (op 418) 
s_mov_b32 s15, scc // (op 418) 
s_cmp_eq_i32 s15, 0 // (op 418) 
s_cselect_b32 s6, s8, s50 // (op 418) 


s_cmp_lt_u32 s0, s6 // (op 418) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_DPStreamTileLoop_418 // (op 418) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_DPStreamTileLoop_418 if true)
RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_418:



















































s_endpgm  // End of RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
.LRR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_end:
.size RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2, .LRR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_end-RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
  .amdhsa_next_free_vgpr 180
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 49152 // lds bytes
  .amdhsa_accum_offset 116
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
    DestDataType: b
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
    macrotile: [128, 128, 128]
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
  MIWaveTile: [4, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
    .symbol: RR_GEMM_TN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 49152
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 88
    .vgpr_count: 116
    .agpr_count: 64
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

