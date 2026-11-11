// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2,@function
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 1446) 
v_and_b32 v3, 15, v1 // (op 1446) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1446) 


v_and_b32 v0, 63, v1 // (op 1446) 
v_lshrrev_b32 v3, 4, v0 // (op 1446) 

v_lshl_add_u32 v0, v4, 2, v3 // (op 1446) 


v_lshrrev_b32 v3, 6, v1 // (op 1569) 
v_and_b32 v4, 15, v1 // (op 1569) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 1569) 


v_and_b32 v3, 63, v1 // (op 1569) 
v_lshrrev_b32 v4, 4, v3 // (op 1569) 

v_lshl_add_u32 v3, v5, 2, v4 // (op 1569) 


v_lshrrev_b32 v4, 6, v1 // (op 1627) 
v_and_b32 v5, 15, v1 // (op 1627) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 1627) 


v_and_b32 v4, 63, v1 // (op 1627) 
v_lshrrev_b32 v5, 4, v4 // (op 1627) 

v_lshl_add_u32 v4, v6, 2, v5 // (op 1627) 


v_lshrrev_b32 v5, 6, v1 // (op 1651) 
v_and_b32 v6, 15, v1 // (op 1651) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 1651) 


v_and_b32 v5, 63, v1 // (op 1651) 
v_lshrrev_b32 v6, 4, v5 // (op 1651) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 1651) 


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
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_249 // (op 249) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_249 if false)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_SKStreamTileLoop_249:
v_lshrrev_b32 v8, 6, v1 // (op 1371) 
s_mul_i32 s52, s16, s8 // (op 1371) 
s_add_u32 s52, s52, s0 // (op 1371) 
v_and_b32 v9, 63, v1 // (op 1371) 
s_mul_hi_u32 s53, s52, s13 // (op 1371) 
v_lshrrev_b32 v10, 4, v9 // (op 1371) 
s_sub_u32 s54, s52, s53 // (op 1371) 
v_lshlrev_b32 v11, 2, v10 // (op 1371) 

v_lshl_add_u32 v10, v8, 4, v11 // (op 1371) 


s_lshr_b32 s55, s54, 1 // (op 1371) 

s_add_u32 s54, s55, s53 // (op 1371) 


v_lshrrev_b32 v8, 2, v9 // (op 1371) 

s_lshr_b32 s53, s54, s11 // (op 1371) 

v_and_b32 v9, 3, v8 // (op 1371) 

s_and_b32 s54, s11, 2147483648 // (op 1371) 
v_and_b32 v8, 3, v1 // (op 1371) 
s_cmp_eq_i32 s54, 0 // (op 1371) 
s_mov_b32 s55, scc // (op 1371) 

v_lshl_add_u32 v11, v9, 2, v8 // (op 1371) 


s_cmp_eq_i32 s55, 0 // (op 1371) 
s_cselect_b32 s54, s52, s53 // (op 1371) 



s_mul_hi_u32 s52, s54, s12 // (op 1371) 
s_sub_u32 s53, s54, s52 // (op 1371) 
s_lshr_b32 s55, s53, 1 // (op 1371) 

s_add_u32 s53, s55, s52 // (op 1371) 


s_lshr_b32 s52, s53, s10 // (op 1371) 

s_and_b32 s53, s10, 2147483648 // (op 1371) 
s_cmp_eq_i32 s53, 0 // (op 1371) 
s_mov_b32 s55, scc // (op 1371) 

s_cmp_eq_i32 s55, 0 // (op 1371) 
s_cselect_b32 s53, s54, s52 // (op 1371) 


s_mul_i32 s52, s53, s14 // (op 1371) 
v_lshl_add_u32 v8, s53, 5, v10 // (op 1371) 


s_sub_u32 s53, s54, s52 // (op 1371) 


v_lshl_add_u32 v9, s53, 4, v11 // (op 1371) 


v_mul_lo_u32 v10, v9, s4 // (op 1371) 

v_add_lshl_u32 v9, v8, v10, 1 // (op 1371) 


s_mul_i32 s53, s4, s23 // (op 1373) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 1373) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 1373) least significant
s_mul_hi_u32 s55, s4, s22 // (op 1373) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1373) most significant: sum
s_add_u32 s53, s53, s55 // (op 1373) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 1373) 

s_mov_b32 s56, s20 // (op 1373) reg expression
s_mov_b32 s57, s21 // (op 1373) reg expression
s_mov_b32 s58, s54 // (op 1373) reg expression
s_mov_b32 s59, 131072 // (op 1373) call()

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
v_lshrrev_b32 v8, 6, v1 // (op 903) 
s_mul_i32 s52, s16, s8 // (op 903) 
s_add_u32 s52, s52, s0 // (op 903) 
v_and_b32 v10, 15, v1 // (op 903) 
s_mul_hi_u32 s53, s52, s13 // (op 903) 
v_lshl_add_u32 v11, v8, 4, v10 // (op 903) 


s_sub_u32 s54, s52, s53 // (op 903) 
v_and_b32 v8, 63, v1 // (op 903) 
s_lshr_b32 s55, s54, 1 // (op 903) 

v_lshrrev_b32 v10, 4, v8 // (op 903) 

s_add_u32 s54, s55, s53 // (op 903) 


v_lshlrev_b32 v8, 5, v10 // (op 903) 

s_lshr_b32 s53, s54, s11 // (op 903) 

s_and_b32 s54, s11, 2147483648 // (op 903) 
s_cmp_eq_i32 s54, 0 // (op 903) 
s_mov_b32 s55, scc // (op 903) 

s_cmp_eq_i32 s55, 0 // (op 903) 
s_cselect_b32 s54, s52, s53 // (op 903) 


s_mul_hi_u32 s53, s54, s12 // (op 903) 
s_sub_u32 s55, s54, s53 // (op 903) 
s_lshr_b32 s60, s55, 1 // (op 903) 

s_add_u32 s55, s60, s53 // (op 903) 


s_lshr_b32 s53, s55, s10 // (op 903) 

s_and_b32 s55, s10, 2147483648 // (op 903) 
s_cmp_eq_i32 s55, 0 // (op 903) 
s_mov_b32 s60, scc // (op 903) 

s_cmp_eq_i32 s60, 0 // (op 903) 
s_cselect_b32 s55, s54, s53 // (op 903) 


v_lshl_add_u32 v10, s55, 5, v11 // (op 903) 


s_ashr_i64 s[60:61], s[2:3], 7 // (op 903) 
s_mul_i32 s53, s54, s60 // (op 903) 


s_sub_u32 s54, s52, s53 // (op 903) 


v_lshl_add_u32 v11, s54, 7, v8 // (op 903) 


v_mov_b32 v13, 0 // (op 903) convert
v_mov_b32 v12, v11 // (op 903) convert

v_mul_lo_u32 v15, v10, s25 // (op 903) most significant: low of low * high
v_mul_hi_u32 v8, v10, s24 // (op 903) most significant: high of low * low
v_mul_lo_u32 v14, v10, s24 // (op 903) least significant: low of low * low
v_add_u32 v15, v15, v8 // (op 903) most significant: sum

v_add_co_u32 v14, s[52:53], v14, v12 // (op 903) least significant half
v_addc_co_u32 v15, s[52:53], v15, v13, s[52:53] // (op 903) most significant half



v_lshlrev_b64 v[10:11], 2, v[14:15] // (op 903) 

v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 903) 

v_mov_b32 v8, v12 // (op 903) convert

s_mul_i32 s53, s24, s29 // (op 905) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 905) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 905) least significant
s_mul_hi_u32 s55, s24, s28 // (op 905) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 905) most significant: sum
s_add_u32 s53, s53, s55 // (op 905) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 905) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 905) 

s_mov_b32 s60, s26 // (op 905) reg expression
s_mov_b32 s61, s27 // (op 905) reg expression
s_mov_b32 s62, s52 // (op 905) reg expression
s_mov_b32 s63, 131072 // (op 905) call()

v_lshrrev_b32 v10, 6, v1 // (op 922) 
s_mul_i32 s52, s16, s8 // (op 922) 
s_add_u32 s52, s52, s0 // (op 922) 
v_and_b32 v11, 15, v1 // (op 922) 
s_mul_hi_u32 s53, s52, s13 // (op 922) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 922) 


s_sub_u32 s54, s52, s53 // (op 922) 
v_and_b32 v10, 63, v1 // (op 922) 
s_lshr_b32 s55, s54, 1 // (op 922) 

v_lshrrev_b32 v11, 4, v10 // (op 922) 

s_add_u32 s54, s55, s53 // (op 922) 


v_lshlrev_b32 v10, 5, v11 // (op 922) 

s_lshr_b32 s53, s54, s11 // (op 922) 

s_and_b32 s54, s11, 2147483648 // (op 922) 
s_cmp_eq_i32 s54, 0 // (op 922) 
s_mov_b32 s55, scc // (op 922) 

s_cmp_eq_i32 s55, 0 // (op 922) 
s_cselect_b32 s54, s52, s53 // (op 922) 


s_mul_hi_u32 s53, s54, s12 // (op 922) 
s_sub_u32 s55, s54, s53 // (op 922) 
s_lshr_b32 s64, s55, 1 // (op 922) 

s_add_u32 s55, s64, s53 // (op 922) 


s_lshr_b32 s53, s55, s10 // (op 922) 

s_and_b32 s55, s10, 2147483648 // (op 922) 
s_cmp_eq_i32 s55, 0 // (op 922) 
s_mov_b32 s64, scc // (op 922) 

s_cmp_eq_i32 s64, 0 // (op 922) 
s_cselect_b32 s55, s54, s53 // (op 922) 


v_lshl_add_u32 v11, s55, 5, v12 // (op 922) 


s_ashr_i64 s[64:65], s[2:3], 7 // (op 922) 
s_mul_i32 s53, s54, s64 // (op 922) 


s_sub_u32 s54, s52, s53 // (op 922) 


v_lshl_add_u32 v12, s54, 7, v10 // (op 922) 


v_mov_b32 v15, 0 // (op 922) convert
v_mov_b32 v14, v12 // (op 922) convert

v_mul_lo_u32 v13, v11, s25 // (op 922) most significant: low of low * high
v_mul_hi_u32 v10, v11, s24 // (op 922) most significant: high of low * low
v_mul_lo_u32 v12, v11, s24 // (op 922) least significant: low of low * low
v_add_u32 v13, v13, v10 // (op 922) most significant: sum

v_add_co_u32 v12, s[52:53], v12, v14 // (op 922) least significant half
v_addc_co_u32 v13, s[52:53], v13, v15, s[52:53] // (op 922) most significant half



v_lshlrev_b64 v[10:11], 2, v[12:13] // (op 922) 

v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 922) 

v_mov_b32 v10, v12 // (op 922) convert

s_mul_i32 s53, s24, s29 // (op 924) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 924) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 924) least significant
s_mul_hi_u32 s55, s24, s28 // (op 924) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 924) most significant: sum
s_add_u32 s53, s53, s55 // (op 924) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 924) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 924) 

s_mov_b32 s60, s26 // (op 924) reg expression
s_mov_b32 s61, s27 // (op 924) reg expression
s_mov_b32 s62, s52 // (op 924) reg expression
s_mov_b32 s63, 131072 // (op 924) call()

v_and_b32 v11, 15, v1 // (op 1034) 
s_mul_i32 s52, s16, s8 // (op 1034) 
s_add_u32 s52, s52, s0 // (op 1034) 
v_and_b32 v12, 48, v1 // (op 1034) 
s_mul_hi_u32 s53, s52, s13 // (op 1034) 
s_sub_u32 s54, s52, s53 // (op 1034) 
s_lshr_b32 s55, s54, 1 // (op 1034) 

s_add_u32 s54, s55, s53 // (op 1034) 


s_lshr_b32 s53, s54, s11 // (op 1034) 

s_and_b32 s54, s11, 2147483648 // (op 1034) 
s_cmp_eq_i32 s54, 0 // (op 1034) 
s_mov_b32 s55, scc // (op 1034) 

s_cmp_eq_i32 s55, 0 // (op 1034) 
s_cselect_b32 s54, s52, s53 // (op 1034) 


s_mul_hi_u32 s53, s54, s12 // (op 1034) 
s_sub_u32 s55, s54, s53 // (op 1034) 
s_lshr_b32 s64, s55, 1 // (op 1034) 

s_add_u32 s55, s64, s53 // (op 1034) 


s_lshr_b32 s53, s55, s10 // (op 1034) 

s_and_b32 s55, s10, 2147483648 // (op 1034) 
s_cmp_eq_i32 s55, 0 // (op 1034) 
s_mov_b32 s64, scc // (op 1034) 

s_cmp_eq_i32 s64, 0 // (op 1034) 
s_cselect_b32 s55, s54, s53 // (op 1034) 


s_mul_i32 s53, s55, s14 // (op 1034) 

s_sub_u32 s55, s54, s53 // (op 1034) 

v_lshl_add_u32 v13, s55, 4, v11 // (op 1034) 


v_mul_lo_u32 v11, v13, s30 // (op 1034) 

s_ashr_i64 s[64:65], s[2:3], 7 // (op 1034) 
s_mul_i32 s53, s54, s64 // (op 1034) 


s_sub_u32 s54, s52, s53 // (op 1034) 


v_lshl_add_u32 v13, s54, 7, v12 // (op 1034) 


v_add_u32 v12, v13, v11 // (op 1034) 


s_mul_i32 s53, s30, s23 // (op 1036) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 1036) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 1036) least significant
s_mul_hi_u32 s55, s30, s22 // (op 1036) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1036) most significant: sum
s_add_u32 s53, s53, s55 // (op 1036) most significant: sum


s_mov_b32 s64, s32 // (op 1036) reg expression
s_mov_b32 s65, s33 // (op 1036) reg expression
s_mov_b32 s66, s52 // (op 1036) reg expression
s_mov_b32 s67, 131072 // (op 1036) call()

v_and_b32 v11, 15, v1 // (op 1053) 
s_mul_i32 s52, s16, s8 // (op 1053) 
s_add_u32 s52, s52, s0 // (op 1053) 
v_and_b32 v13, 48, v1 // (op 1053) 
s_mul_hi_u32 s53, s52, s13 // (op 1053) 
s_sub_u32 s54, s52, s53 // (op 1053) 
s_lshr_b32 s55, s54, 1 // (op 1053) 

s_add_u32 s54, s55, s53 // (op 1053) 


s_lshr_b32 s53, s54, s11 // (op 1053) 

s_and_b32 s54, s11, 2147483648 // (op 1053) 
s_cmp_eq_i32 s54, 0 // (op 1053) 
s_mov_b32 s55, scc // (op 1053) 

s_cmp_eq_i32 s55, 0 // (op 1053) 
s_cselect_b32 s54, s52, s53 // (op 1053) 


s_mul_hi_u32 s53, s54, s12 // (op 1053) 
s_sub_u32 s55, s54, s53 // (op 1053) 
s_lshr_b32 s68, s55, 1 // (op 1053) 

s_add_u32 s55, s68, s53 // (op 1053) 


s_lshr_b32 s53, s55, s10 // (op 1053) 

s_and_b32 s55, s10, 2147483648 // (op 1053) 
s_cmp_eq_i32 s55, 0 // (op 1053) 
s_mov_b32 s68, scc // (op 1053) 

s_cmp_eq_i32 s68, 0 // (op 1053) 
s_cselect_b32 s55, s54, s53 // (op 1053) 


s_mul_i32 s53, s55, s14 // (op 1053) 

s_sub_u32 s55, s54, s53 // (op 1053) 

v_lshl_add_u32 v14, s55, 4, v11 // (op 1053) 


v_mul_lo_u32 v11, v14, s30 // (op 1053) 

s_ashr_i64 s[68:69], s[2:3], 7 // (op 1053) 
s_mul_i32 s53, s54, s68 // (op 1053) 


s_sub_u32 s54, s52, s53 // (op 1053) 


v_lshl_add_u32 v14, s54, 7, v13 // (op 1053) 


v_add_u32 v13, v14, v11 // (op 1053) 


s_mul_i32 s53, s30, s23 // (op 1055) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 1055) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 1055) least significant
s_mul_hi_u32 s55, s30, s22 // (op 1055) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1055) most significant: sum
s_add_u32 s53, s53, s55 // (op 1055) most significant: sum


s_mov_b32 s64, s32 // (op 1055) reg expression
s_mov_b32 s65, s33 // (op 1055) reg expression
s_mov_b32 s66, s52 // (op 1055) reg expression
s_mov_b32 s67, 131072 // (op 1055) call()

v_lshrrev_b32 v11, 2, v1 // (op 1162) 
s_mul_i32 s52, s16, s8 // (op 1162) 
s_add_u32 s52, s52, s0 // (op 1162) 
v_and_b32 v14, -4, v1 // (op 1162) 
s_mul_hi_u32 s53, s52, s13 // (op 1162) 
v_sub_u32 v15, v1, v14 // (op 1162) 

s_sub_u32 s54, s52, s53 // (op 1162) 
s_lshr_b32 s55, s54, 1 // (op 1162) 

s_add_u32 s54, s55, s53 // (op 1162) 


s_lshr_b32 s53, s54, s11 // (op 1162) 

s_and_b32 s54, s11, 2147483648 // (op 1162) 
s_cmp_eq_i32 s54, 0 // (op 1162) 
s_mov_b32 s55, scc // (op 1162) 

s_cmp_eq_i32 s55, 0 // (op 1162) 
s_cselect_b32 s54, s52, s53 // (op 1162) 


s_mul_hi_u32 s53, s54, s12 // (op 1162) 
s_sub_u32 s55, s54, s53 // (op 1162) 
s_lshr_b32 s68, s55, 1 // (op 1162) 

s_add_u32 s55, s68, s53 // (op 1162) 


s_lshr_b32 s53, s55, s10 // (op 1162) 

s_and_b32 s55, s10, 2147483648 // (op 1162) 
s_cmp_eq_i32 s55, 0 // (op 1162) 
s_mov_b32 s68, scc // (op 1162) 

s_cmp_eq_i32 s68, 0 // (op 1162) 
s_cselect_b32 s55, s54, s53 // (op 1162) 


v_lshl_add_u32 v14, s55, 5, v11 // (op 1162) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 1162) 
s_mul_i32 s53, s54, s68 // (op 1162) 


s_sub_u32 s54, s52, s53 // (op 1162) 


v_lshl_add_u32 v11, s54, 2, v15 // (op 1162) 


v_mul_lo_u32 v15, v14, s34 // (op 1162) 
v_add_u32 v15, v15, v11 // (op 1162) 


s_mul_i32 s53, s34, s29 // (op 1164) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 1164) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 1164) least significant
s_mul_hi_u32 s55, s34, s28 // (op 1164) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1164) most significant: sum
s_add_u32 s53, s53, s55 // (op 1164) most significant: sum


s_mov_b32 s68, s36 // (op 1164) reg expression
s_mov_b32 s69, s37 // (op 1164) reg expression
s_mov_b32 s70, s52 // (op 1164) reg expression
s_mov_b32 s71, 131072 // (op 1164) call()

v_lshrrev_b32 v11, 2, v1 // (op 1181) 
s_mul_i32 s52, s16, s8 // (op 1181) 
s_add_u32 s52, s52, s0 // (op 1181) 
v_and_b32 v14, -4, v1 // (op 1181) 
s_mul_hi_u32 s53, s52, s13 // (op 1181) 
v_sub_u32 v16, v1, v14 // (op 1181) 

s_sub_u32 s54, s52, s53 // (op 1181) 
s_lshr_b32 s55, s54, 1 // (op 1181) 

s_add_u32 s54, s55, s53 // (op 1181) 


s_lshr_b32 s53, s54, s11 // (op 1181) 

s_and_b32 s54, s11, 2147483648 // (op 1181) 
s_cmp_eq_i32 s54, 0 // (op 1181) 
s_mov_b32 s55, scc // (op 1181) 

s_cmp_eq_i32 s55, 0 // (op 1181) 
s_cselect_b32 s54, s52, s53 // (op 1181) 


s_mul_hi_u32 s53, s54, s12 // (op 1181) 
s_sub_u32 s55, s54, s53 // (op 1181) 
s_lshr_b32 s72, s55, 1 // (op 1181) 

s_add_u32 s55, s72, s53 // (op 1181) 


s_lshr_b32 s53, s55, s10 // (op 1181) 

s_and_b32 s55, s10, 2147483648 // (op 1181) 
s_cmp_eq_i32 s55, 0 // (op 1181) 
s_mov_b32 s72, scc // (op 1181) 

s_cmp_eq_i32 s72, 0 // (op 1181) 
s_cselect_b32 s55, s54, s53 // (op 1181) 


v_lshl_add_u32 v14, s55, 5, v11 // (op 1181) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 1181) 
s_mul_i32 s53, s54, s72 // (op 1181) 


s_sub_u32 s54, s52, s53 // (op 1181) 


v_lshl_add_u32 v11, s54, 2, v16 // (op 1181) 


v_mul_lo_u32 v16, v14, s34 // (op 1181) 
v_add_u32 v16, v16, v11 // (op 1181) 


s_mul_i32 s53, s34, s29 // (op 1183) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 1183) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 1183) least significant
s_mul_hi_u32 s55, s34, s28 // (op 1183) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1183) most significant: sum
s_add_u32 s53, s53, s55 // (op 1183) most significant: sum


s_mov_b32 s68, s36 // (op 1183) reg expression
s_mov_b32 s69, s37 // (op 1183) reg expression
s_mov_b32 s70, s52 // (op 1183) reg expression
s_mov_b32 s71, 131072 // (op 1183) call()

v_and_b32 v11, 15, v1 // (op 1290) 
s_mul_i32 s52, s16, s8 // (op 1290) 
s_add_u32 s52, s52, s0 // (op 1290) 
v_and_b32 v14, 63, v1 // (op 1290) 
s_mul_hi_u32 s53, s52, s13 // (op 1290) 
v_lshrrev_b32 v17, 4, v14 // (op 1290) 

s_sub_u32 s54, s52, s53 // (op 1290) 
s_lshr_b32 s55, s54, 1 // (op 1290) 

s_add_u32 s54, s55, s53 // (op 1290) 


s_lshr_b32 s53, s54, s11 // (op 1290) 

s_and_b32 s54, s11, 2147483648 // (op 1290) 
s_cmp_eq_i32 s54, 0 // (op 1290) 
s_mov_b32 s55, scc // (op 1290) 

s_cmp_eq_i32 s55, 0 // (op 1290) 
s_cselect_b32 s54, s52, s53 // (op 1290) 


s_mul_hi_u32 s53, s54, s12 // (op 1290) 
s_sub_u32 s55, s54, s53 // (op 1290) 
s_lshr_b32 s72, s55, 1 // (op 1290) 

s_add_u32 s55, s72, s53 // (op 1290) 


s_lshr_b32 s53, s55, s10 // (op 1290) 

s_and_b32 s55, s10, 2147483648 // (op 1290) 
s_cmp_eq_i32 s55, 0 // (op 1290) 
s_mov_b32 s72, scc // (op 1290) 

s_cmp_eq_i32 s72, 0 // (op 1290) 
s_cselect_b32 s55, s54, s53 // (op 1290) 


s_mul_i32 s53, s55, s14 // (op 1290) 

s_sub_u32 s55, s54, s53 // (op 1290) 

v_lshl_add_u32 v14, s55, 4, v11 // (op 1290) 


v_mul_lo_u32 v11, v14, s38 // (op 1290) 

s_ashr_i64 s[72:73], s[2:3], 7 // (op 1290) 
s_mul_i32 s53, s54, s72 // (op 1290) 


s_sub_u32 s54, s52, s53 // (op 1290) 


v_lshl_add_u32 v14, s54, 2, v17 // (op 1290) 


v_add_u32 v17, v14, v11 // (op 1290) 


s_mul_i32 s53, s38, s23 // (op 1292) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 1292) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 1292) least significant
s_mul_hi_u32 s55, s38, s22 // (op 1292) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1292) most significant: sum
s_add_u32 s53, s53, s55 // (op 1292) most significant: sum


s_mov_b32 s72, s40 // (op 1292) reg expression
s_mov_b32 s73, s41 // (op 1292) reg expression
s_mov_b32 s74, s52 // (op 1292) reg expression
s_mov_b32 s75, 131072 // (op 1292) call()

v_and_b32 v11, 15, v1 // (op 1309) 
s_mul_i32 s52, s16, s8 // (op 1309) 
s_add_u32 s52, s52, s0 // (op 1309) 
v_and_b32 v14, 63, v1 // (op 1309) 
s_mul_hi_u32 s53, s52, s13 // (op 1309) 
v_lshrrev_b32 v18, 4, v14 // (op 1309) 

s_sub_u32 s54, s52, s53 // (op 1309) 
s_lshr_b32 s55, s54, 1 // (op 1309) 

s_add_u32 s54, s55, s53 // (op 1309) 


s_lshr_b32 s53, s54, s11 // (op 1309) 

s_and_b32 s54, s11, 2147483648 // (op 1309) 
s_cmp_eq_i32 s54, 0 // (op 1309) 
s_mov_b32 s55, scc // (op 1309) 

s_cmp_eq_i32 s55, 0 // (op 1309) 
s_cselect_b32 s54, s52, s53 // (op 1309) 


s_mul_hi_u32 s53, s54, s12 // (op 1309) 
s_sub_u32 s55, s54, s53 // (op 1309) 
s_lshr_b32 s76, s55, 1 // (op 1309) 

s_add_u32 s55, s76, s53 // (op 1309) 


s_lshr_b32 s53, s55, s10 // (op 1309) 

s_and_b32 s55, s10, 2147483648 // (op 1309) 
s_cmp_eq_i32 s55, 0 // (op 1309) 
s_mov_b32 s76, scc // (op 1309) 

s_cmp_eq_i32 s76, 0 // (op 1309) 
s_cselect_b32 s55, s54, s53 // (op 1309) 


s_mul_i32 s53, s55, s14 // (op 1309) 

s_sub_u32 s55, s54, s53 // (op 1309) 

v_lshl_add_u32 v14, s55, 4, v11 // (op 1309) 


v_mul_lo_u32 v11, v14, s38 // (op 1309) 

s_ashr_i64 s[76:77], s[2:3], 7 // (op 1309) 
s_mul_i32 s53, s54, s76 // (op 1309) 


s_sub_u32 s54, s52, s53 // (op 1309) 


v_lshl_add_u32 v14, s54, 2, v18 // (op 1309) 


v_add_u32 v18, v14, v11 // (op 1309) 


s_mul_i32 s53, s38, s23 // (op 1311) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 1311) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 1311) least significant
s_mul_hi_u32 s55, s38, s22 // (op 1311) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1311) most significant: sum
s_add_u32 s53, s53, s55 // (op 1311) most significant: sum


s_mov_b32 s72, s40 // (op 1311) reg expression
s_mov_b32 s73, s41 // (op 1311) reg expression
s_mov_b32 s74, s52 // (op 1311) reg expression
s_mov_b32 s75, 131072 // (op 1311) call()




buffer_load_dwordx4 v[20:23], v8, s[60:63], 0 offen // (op 693) Load value



buffer_load_dwordx4 v[24:27], v12, s[64:67], 0 offen // (op 698) Load value
buffer_load_dwordx4 v[28:31], v12, s[64:67], 0 offen offset:64 // (op 698) Load value



buffer_load_ubyte v11, v17, s[72:75], 0 offen offset: 0 // (op 703) Load value



buffer_load_ubyte v14, v15, s[68:71], 0 offen offset: 0 // (op 707) Load value



buffer_load_ubyte v19, v17, s[72:75], 0 offen offset:4 // (op 711) Load value




buffer_load_dwordx4 v[32:35], v8, s[60:63], 0 offen offset:64 // (op 716) Load value




buffer_load_dwordx4 v[36:39], v12, s[64:67], 0 offen offset:128 // (op 721) Load value
buffer_load_dwordx4 v[40:43], v12, s[64:67], 0 offen offset:192 // (op 721) Load value




buffer_load_ubyte v8, v15, s[68:71], 0 offen offset:4 // (op 725) Load value

v_and_b32 v12, -4, v1 // (op 1434) 
v_sub_u32 v15, v1, v12 // (op 1434) 

v_and_b32 v12, 4294967292, v1 // (op 1434) 
v_add_u32 v17, v12, v15 // (op 1434) 


s_barrier  // (op 1708) 



s_waitcnt vmcnt(5)
ds_write_b8 v17, v14 // (op 727) Store local data 

v_and_b32 v12, -4, v1 // (op 1405) 
v_sub_u32 v15, v1, v12 // (op 1405) 

v_and_b32 v12, 4294967292, v1 // (op 1405) 
v_add_u32 v17, v12, v15 // (op 1405) 


v_and_b32 v12, -4, v1 // (op 1639) 
v_sub_u32 v15, v1, v12 // (op 1639) 

v_and_b32 v12, 4294967292, v1 // (op 1639) 
v_add_u32 v44, v12, v15 // (op 1639) 


s_mov_b32 s1, 0 // (op 49) call()
s_and_b32 s52, s7, 4294967294 // (op 48) 
s_cmp_lt_u32 s1, s52 // (op 48) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_48 // (op 48) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_48 if false)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoop_48:
s_add_u32 s1, s1, 2 // (op 50) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1705) 



ds_read_u8 v12, v0 // (op 65) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:31], a[0:3], v12, v11 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 72) 



v_mov_b32 v12, v10 // (op 57) 

buffer_load_dwordx4 v[20:23], v12, s[60:63], 0 offen offset:128 // (op 57) Load value


v_mov_b32 v15, v13 // (op 61) 

buffer_load_dwordx4 v[24:27], v15, s[64:67], 0 offen offset:256 // (op 61) Load value
buffer_load_dwordx4 v[28:31], v15, s[64:67], 0 offen offset:320 // (op 61) Load value


v_mov_b32 v45, v18 // (op 70) 

buffer_load_ubyte v11, v45, s[72:75], 0 offen offset:8 // (op 70) Load value


v_mov_b32 v46, v16 // (op 9) 

buffer_load_ubyte v14, v46, s[68:71], 0 offen offset:8 // (op 9) Load value
s_barrier  // (op 1746) 



s_waitcnt vmcnt(5)
ds_write_b8 v44, v8 offset:128 // (op 565) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1743) 



ds_read_u8 v47, v5 offset:128 // (op 567) Load local data 
s_barrier  // (op 1879) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:35], v[36:43], a[0:3], v47, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 570) 




buffer_load_ubyte v19, v45, s[72:75], 0 offen offset:12 // (op 569) Load value




v_add_u32 v18, 8, v18 // (op 1301) 
buffer_load_dwordx4 v[32:35], v12, s[60:63], 0 offen offset:192 // (op 572) Load value




v_add_u32 v10, 128, v10 // (op 914) 
buffer_load_dwordx4 v[36:39], v15, s[64:67], 0 offen offset:384 // (op 574) Load value
buffer_load_dwordx4 v[40:43], v15, s[64:67], 0 offen offset:448 // (op 574) Load value




v_add_u32 v13, 256, v13 // (op 1045) 
buffer_load_ubyte v8, v46, s[68:71], 0 offen offset:12 // (op 564) Load value

s_barrier  // (op 1711) 
v_add_u32 v16, 8, v16 // (op 1173) 



s_waitcnt vmcnt(5)
ds_write_b8 v17, v14 // (op 13) Store local data 
s_and_b32 s52, s7, 4294967294 // (op 48) 
s_cmp_lt_u32 s1, s52 // (op 48) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoop_48 // (op 48) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoop_48 if true)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_48:










s_and_b32 s52, s7, 4294967294 // (op 877) 
v_lshrrev_b32 v10, 6, v1 // (op 877) 
s_add_u32 s53, s0, s52 // (op 877) 

v_and_b32 v11, 15, v1 // (op 877) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 877) 


s_mul_i32 s52, s16, s8 // (op 877) 
s_add_u32 s52, s52, s53 // (op 877) 

v_and_b32 v10, 63, v1 // (op 877) 
s_mul_hi_u32 s53, s52, s13 // (op 877) 
v_lshrrev_b32 v11, 4, v10 // (op 877) 

s_sub_u32 s54, s52, s53 // (op 877) 
v_lshlrev_b32 v10, 5, v11 // (op 877) 

s_lshr_b32 s55, s54, 1 // (op 877) 

s_add_u32 s54, s55, s53 // (op 877) 


s_lshr_b32 s53, s54, s11 // (op 877) 

s_and_b32 s54, s11, 2147483648 // (op 877) 
s_cmp_eq_i32 s54, 0 // (op 877) 
s_mov_b32 s55, scc // (op 877) 

s_cmp_eq_i32 s55, 0 // (op 877) 
s_cselect_b32 s54, s52, s53 // (op 877) 


s_mul_hi_u32 s53, s54, s12 // (op 877) 
s_sub_u32 s55, s54, s53 // (op 877) 
s_lshr_b32 s76, s55, 1 // (op 877) 

s_add_u32 s55, s76, s53 // (op 877) 


s_lshr_b32 s53, s55, s10 // (op 877) 

s_and_b32 s55, s10, 2147483648 // (op 877) 
s_cmp_eq_i32 s55, 0 // (op 877) 
s_mov_b32 s76, scc // (op 877) 

s_cmp_eq_i32 s76, 0 // (op 877) 
s_cselect_b32 s55, s54, s53 // (op 877) 


v_lshl_add_u32 v11, s55, 5, v12 // (op 877) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 877) 
s_mul_i32 s53, s54, s76 // (op 877) 


s_sub_u32 s54, s52, s53 // (op 877) 


v_lshl_add_u32 v12, s54, 7, v10 // (op 877) 


v_mov_b32 v21, 0 // (op 877) convert
v_mov_b32 v20, v12 // (op 877) convert

v_mul_lo_u32 v13, v11, s25 // (op 877) most significant: low of low * high
v_mul_hi_u32 v10, v11, s24 // (op 877) most significant: high of low * low
v_mul_lo_u32 v12, v11, s24 // (op 877) least significant: low of low * low
v_add_u32 v13, v13, v10 // (op 877) most significant: sum

v_add_co_u32 v12, s[52:53], v12, v20 // (op 877) least significant half
v_addc_co_u32 v13, s[52:53], v13, v21, s[52:53] // (op 877) most significant half



v_lshlrev_b64 v[10:11], 2, v[12:13] // (op 877) 

v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 877) 

v_mov_b32 v10, v12 // (op 877) convert

s_mul_i32 s53, s24, s29 // (op 879) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 879) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 879) least significant
s_mul_hi_u32 s55, s24, s28 // (op 879) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 879) most significant: sum
s_add_u32 s53, s53, s55 // (op 879) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 879) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 879) 

s_mov_b32 s60, s26 // (op 879) reg expression
s_mov_b32 s61, s27 // (op 879) reg expression
s_mov_b32 s62, s52 // (op 879) reg expression
s_mov_b32 s63, 131072 // (op 879) call()

s_and_b32 s52, s7, 4294967294 // (op 1011) 
v_and_b32 v11, 15, v1 // (op 1011) 
s_add_u32 s53, s0, s52 // (op 1011) 

v_and_b32 v12, 48, v1 // (op 1011) 
s_mul_i32 s52, s16, s8 // (op 1011) 
s_add_u32 s52, s52, s53 // (op 1011) 

s_mul_hi_u32 s53, s52, s13 // (op 1011) 
s_sub_u32 s54, s52, s53 // (op 1011) 
s_lshr_b32 s55, s54, 1 // (op 1011) 

s_add_u32 s54, s55, s53 // (op 1011) 


s_lshr_b32 s53, s54, s11 // (op 1011) 

s_and_b32 s54, s11, 2147483648 // (op 1011) 
s_cmp_eq_i32 s54, 0 // (op 1011) 
s_mov_b32 s55, scc // (op 1011) 

s_cmp_eq_i32 s55, 0 // (op 1011) 
s_cselect_b32 s54, s52, s53 // (op 1011) 


s_mul_hi_u32 s53, s54, s12 // (op 1011) 
s_sub_u32 s55, s54, s53 // (op 1011) 
s_lshr_b32 s76, s55, 1 // (op 1011) 

s_add_u32 s55, s76, s53 // (op 1011) 


s_lshr_b32 s53, s55, s10 // (op 1011) 

s_and_b32 s55, s10, 2147483648 // (op 1011) 
s_cmp_eq_i32 s55, 0 // (op 1011) 
s_mov_b32 s76, scc // (op 1011) 

s_cmp_eq_i32 s76, 0 // (op 1011) 
s_cselect_b32 s55, s54, s53 // (op 1011) 


s_mul_i32 s53, s55, s14 // (op 1011) 

s_sub_u32 s55, s54, s53 // (op 1011) 

v_lshl_add_u32 v13, s55, 4, v11 // (op 1011) 


v_mul_lo_u32 v11, v13, s30 // (op 1011) 

s_ashr_i64 s[76:77], s[2:3], 7 // (op 1011) 
s_mul_i32 s53, s54, s76 // (op 1011) 


s_sub_u32 s54, s52, s53 // (op 1011) 


v_lshl_add_u32 v13, s54, 7, v12 // (op 1011) 


v_add_u32 v12, v13, v11 // (op 1011) 


s_mul_i32 s53, s30, s23 // (op 1013) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 1013) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 1013) least significant
s_mul_hi_u32 s55, s30, s22 // (op 1013) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1013) most significant: sum
s_add_u32 s53, s53, s55 // (op 1013) most significant: sum


s_mov_b32 s64, s32 // (op 1013) reg expression
s_mov_b32 s65, s33 // (op 1013) reg expression
s_mov_b32 s66, s52 // (op 1013) reg expression
s_mov_b32 s67, 131072 // (op 1013) call()

s_and_b32 s52, s7, 4294967294 // (op 1139) 
v_lshrrev_b32 v11, 2, v1 // (op 1139) 
s_add_u32 s53, s0, s52 // (op 1139) 

v_and_b32 v13, -4, v1 // (op 1139) 
v_sub_u32 v14, v1, v13 // (op 1139) 

s_mul_i32 s52, s16, s8 // (op 1139) 
s_add_u32 s52, s52, s53 // (op 1139) 

s_mul_hi_u32 s53, s52, s13 // (op 1139) 
s_sub_u32 s54, s52, s53 // (op 1139) 
s_lshr_b32 s55, s54, 1 // (op 1139) 

s_add_u32 s54, s55, s53 // (op 1139) 


s_lshr_b32 s53, s54, s11 // (op 1139) 

s_and_b32 s54, s11, 2147483648 // (op 1139) 
s_cmp_eq_i32 s54, 0 // (op 1139) 
s_mov_b32 s55, scc // (op 1139) 

s_cmp_eq_i32 s55, 0 // (op 1139) 
s_cselect_b32 s54, s52, s53 // (op 1139) 


s_mul_hi_u32 s53, s54, s12 // (op 1139) 
s_sub_u32 s55, s54, s53 // (op 1139) 
s_lshr_b32 s76, s55, 1 // (op 1139) 

s_add_u32 s55, s76, s53 // (op 1139) 


s_lshr_b32 s53, s55, s10 // (op 1139) 

s_and_b32 s55, s10, 2147483648 // (op 1139) 
s_cmp_eq_i32 s55, 0 // (op 1139) 
s_mov_b32 s76, scc // (op 1139) 

s_cmp_eq_i32 s76, 0 // (op 1139) 
s_cselect_b32 s55, s54, s53 // (op 1139) 


v_lshl_add_u32 v13, s55, 5, v11 // (op 1139) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1139) 
s_mul_i32 s53, s54, s76 // (op 1139) 


s_sub_u32 s54, s52, s53 // (op 1139) 


v_lshl_add_u32 v11, s54, 2, v14 // (op 1139) 


v_mul_lo_u32 v14, v13, s34 // (op 1139) 
v_add_u32 v14, v14, v11 // (op 1139) 


s_mul_i32 s53, s34, s29 // (op 1141) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 1141) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 1141) least significant
s_mul_hi_u32 s55, s34, s28 // (op 1141) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1141) most significant: sum
s_add_u32 s53, s53, s55 // (op 1141) most significant: sum


s_mov_b32 s68, s36 // (op 1141) reg expression
s_mov_b32 s69, s37 // (op 1141) reg expression
s_mov_b32 s70, s52 // (op 1141) reg expression
s_mov_b32 s71, 131072 // (op 1141) call()

s_and_b32 s52, s7, 4294967294 // (op 1267) 
v_and_b32 v11, 15, v1 // (op 1267) 
s_add_u32 s53, s0, s52 // (op 1267) 

v_and_b32 v13, 63, v1 // (op 1267) 
v_lshrrev_b32 v15, 4, v13 // (op 1267) 

s_mul_i32 s52, s16, s8 // (op 1267) 
s_add_u32 s52, s52, s53 // (op 1267) 

s_mul_hi_u32 s53, s52, s13 // (op 1267) 
s_sub_u32 s54, s52, s53 // (op 1267) 
s_lshr_b32 s55, s54, 1 // (op 1267) 

s_add_u32 s54, s55, s53 // (op 1267) 


s_lshr_b32 s53, s54, s11 // (op 1267) 

s_and_b32 s54, s11, 2147483648 // (op 1267) 
s_cmp_eq_i32 s54, 0 // (op 1267) 
s_mov_b32 s55, scc // (op 1267) 

s_cmp_eq_i32 s55, 0 // (op 1267) 
s_cselect_b32 s54, s52, s53 // (op 1267) 


s_mul_hi_u32 s53, s54, s12 // (op 1267) 
s_sub_u32 s55, s54, s53 // (op 1267) 
s_lshr_b32 s76, s55, 1 // (op 1267) 

s_add_u32 s55, s76, s53 // (op 1267) 


s_lshr_b32 s53, s55, s10 // (op 1267) 

s_and_b32 s55, s10, 2147483648 // (op 1267) 
s_cmp_eq_i32 s55, 0 // (op 1267) 
s_mov_b32 s76, scc // (op 1267) 

s_cmp_eq_i32 s76, 0 // (op 1267) 
s_cselect_b32 s55, s54, s53 // (op 1267) 


s_mul_i32 s53, s55, s14 // (op 1267) 

s_sub_u32 s55, s54, s53 // (op 1267) 

v_lshl_add_u32 v13, s55, 4, v11 // (op 1267) 


v_mul_lo_u32 v11, v13, s38 // (op 1267) 

s_ashr_i64 s[76:77], s[2:3], 7 // (op 1267) 
s_mul_i32 s53, s54, s76 // (op 1267) 


s_sub_u32 s54, s52, s53 // (op 1267) 


v_lshl_add_u32 v13, s54, 2, v15 // (op 1267) 


v_add_u32 v15, v13, v11 // (op 1267) 


s_mul_i32 s53, s38, s23 // (op 1269) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 1269) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 1269) least significant
s_mul_hi_u32 s55, s38, s22 // (op 1269) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1269) most significant: sum
s_add_u32 s53, s53, s55 // (op 1269) most significant: sum


s_mov_b32 s72, s40 // (op 1269) reg expression
s_mov_b32 s73, s41 // (op 1269) reg expression
s_mov_b32 s74, s52 // (op 1269) reg expression
s_mov_b32 s75, 131072 // (op 1269) call()

v_and_b32 v11, -4, v1 // (op 1418) 
v_sub_u32 v13, v1, v11 // (op 1418) 

v_and_b32 v11, 4294967292, v1 // (op 1418) 
v_add_u32 v16, v11, v13 // (op 1418) 


s_and_b32 s52, s7, 4294967294 // (op 518) 
s_cmp_lt_u32 s52, s7 // (op 517) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_517 // (op 517) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_517 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_517:


s_add_u32 s52, s52, 1 // (op 519) 
v_mov_b32 v8, v14 // (op 522) 

buffer_load_ubyte v11, v8, s[68:71], 0 offen offset: 0 // (op 522) Load value

v_add_u32 v14, 4, v14 // (op 1131) 


s_barrier  // (op 1714) 
v_mov_b32 v8, v10 // (op 530) 




buffer_load_dwordx4 v[18:21], v8, s[60:63], 0 offen // (op 530) Load value

v_add_u32 v10, 64, v10 // (op 867) 
s_waitcnt vmcnt(1)
ds_write_b8 v16, v11 // (op 523) Store local data 



v_mov_b32 v8, v12 // (op 532) 

buffer_load_dwordx4 v[22:25], v8, s[64:67], 0 offen // (op 532) Load value
buffer_load_dwordx4 v[26:29], v8, s[64:67], 0 offen offset:64 // (op 532) Load value

v_add_u32 v12, 128, v12 // (op 1003) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1717) 



ds_read_u8 v8, v0 // (op 525) Load local data 


v_mov_b32 v11, v15 // (op 527) 

buffer_load_ubyte v13, v11, s[72:75], 0 offen offset: 0 // (op 527) Load value

v_add_u32 v15, 4, v15 // (op 1259) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[22:29], a[0:3], v8, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 528) 




s_cmp_lt_u32 s52, s7 // (op 517) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_517 // (op 517) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_517 if true)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_517:






s_barrier  // (op 1873) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_lg_u32 s18, 0 // (op 277) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Send_Tile_277 // (op 277) Condition: False, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Send_Tile_277
v_lshrrev_b32 v8, 4, v1 // (op 1583) 
v_lshlrev_b32 v10, 2, v8 // (op 1583) 

v_lshl_add_u32 v8, s16, 5, v10 // (op 1583) 

v_and_b32 v10, -16, v1 // (op 1583) 
v_sub_u32 v11, v1, v10 // (op 1583) 

v_lshl_add_u32 v10, v8, 4, v11 // (op 1583) 


v_lshlrev_b32 v8, 2, v10 // (op 1583) 

s_lshl_b32 s52, s15, 11 // (op 1585) 
s_mov_b32 s76, s42 // (op 1585) reg expression
s_mov_b32 s77, s43 // (op 1585) reg expression
s_mov_b32 s78, s52 // (op 1585) reg expression
s_mov_b32 s79, 131072 // (op 1585) call()




buffer_store_dword a0, v8, s[76:79], 0 offen sc1 // (op 274) Store value
buffer_store_dword a1, v8, s[76:79], 0 offen offset:64 sc1 // (op 274) Store value
buffer_store_dword a2, v8, s[76:79], 0 offen offset:128 sc1 // (op 274) Store value
buffer_store_dword a3, v8, s[76:79], 0 offen offset:192 sc1 // (op 274) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 278) 
v_cmp_eq_i32 vcc, v1, 0 // (op 282) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_282 // (op 282) Condition: False, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_282
s_mov_b32 s52, 1 // (op 280) call()
s_mov_b32 s55, 0 // (op 281) convert
s_mov_b32 s54, s16 // (op 281) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 281) 
s_add_u32 s80, s44, s54 // (op 281) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 281) End of Int64 add, unlocking SCC
s_store_dword s52, s[80:81], 0 glc // (op 281) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_282 // (op 282) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_282
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_282:
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_282:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Send_Tile_277 // (op 277) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Send_Tile_277
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Send_Tile_277:
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Send_Tile_277:
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
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Receive_Tile_293 // (op 293) Condition: False, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Receive_Tile_293
s_mov_b32 s52, 0 // (op 259) call()
s_mul_i32 s53, s52, s8 // (op 258) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 258) 
s_mov_b32 s80, s54 // (op 258) convert

s_sub_u32 s54, s80, s19 // (op 258) 

s_sub_u32 s55, s54, 1 // (op 258) 

s_cmp_lt_u32 s53, s55 // (op 258) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_258 // (op 258) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_258 if false)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_ReceiveTileLoop_258:
s_add_u32 s53, s16, 1 // (op 1600) 
v_lshrrev_b32 v8, 4, v1 // (op 1600) 
s_add_u32 s54, s53, s52 // (op 1600) 
s_lshl_b32 s54, s54, 5 // (op 1600) 

v_lshlrev_b32 v10, 2, v8 // (op 1600) 

v_add_lshl_u32 v8, s54, v10, 4 // (op 1600) 


v_and_b32 v10, -16, v1 // (op 1600) 
v_sub_u32 v11, v1, v10 // (op 1600) 

v_add_lshl_u32 v10, v8, v11, 2 // (op 1600) 


s_lshl_b32 s53, s15, 11 // (op 1602) 
s_mov_b32 s76, s42 // (op 1602) reg expression
s_mov_b32 s77, s43 // (op 1602) reg expression
s_mov_b32 s78, s53 // (op 1602) reg expression
s_mov_b32 s79, 131072 // (op 1602) call()

RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_297:
s_add_u32 s53, s16, 1 // (op 295) 
s_add_u32 s54, s53, s52 // (op 295) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 295) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 295) 

s_add_u32 s80, s44, s54 // (op 295) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 295) End of Int64 add, unlocking SCC
s_load_dword s53, s[80:81], 0 glc // (op 295) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 297) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_297 // (op 297) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_297 if true)
s_barrier  // (op 301) 
v_cmp_eq_i32 vcc, v1, 0 // (op 300) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_300 // (op 300) Condition: False, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_300
s_mov_b32 s53, 0 // (op 298) call()
s_add_u32 s54, s16, 1 // (op 299) 
s_add_u32 s80, s54, s52 // (op 299) Start of Int64 add, locking SCC
s_addc_u32 s81, 0, 0 // (op 299) End of Int64 add, unlocking SCC
s_lshl_b64 s[80:81], s[80:81], 2 // (op 299) 

s_add_u32 s54, s44, s80 // (op 299) Start of Int64 add, locking SCC
s_addc_u32 s55, s45, s81 // (op 299) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 299) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_300 // (op 300) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_300
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_300:
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_300:




buffer_load_dword v12, v10, s[76:79], 0 offen // (op 312) Load value
buffer_load_dword v13, v10, s[76:79], 0 offen offset:64 // (op 312) Load value
buffer_load_dword v14, v10, s[76:79], 0 offen offset:128 // (op 312) Load value
buffer_load_dword v15, v10, s[76:79], 0 offen offset:192 // (op 312) Load value
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

s_add_u32 s52, s52, 1 // (op 260) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 258) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 258) 
s_mov_b32 s80, s54 // (op 258) convert

s_sub_u32 s54, s80, s19 // (op 258) 

s_sub_u32 s55, s54, 1 // (op 258) 

s_cmp_lt_u32 s53, s55 // (op 258) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_ReceiveTileLoop_258 // (op 258) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_ReceiveTileLoop_258 if true)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_258:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Receive_Tile_293 // (op 293) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Receive_Tile_293
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Receive_Tile_293:
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Receive_Tile_293:

s_cmp_eq_i32 s18, 0 // (op 336) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_336 // (op 336) Condition: False, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_336
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 108) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_108 // (op 108) Condition: False, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_108
v_accvgpr_read v10, a0 // (op 38) 
v_accvgpr_read v11, a1 // (op 38) 
v_accvgpr_read v12, a2 // (op 38) 
v_accvgpr_read v13, a3 // (op 38) 
v_mul_f32 v14, v7, v10 // (op 38) 
v_mul_f32 v15, v7, v11 // (op 38) 
v_mul_f32 v16, v7, v12 // (op 38) 
v_mul_f32 v17, v7, v13 // (op 38) 

v_lshrrev_b32 v10, 16, v14 // (op 44) 
v_lshrrev_b32 v11, 16, v15 // (op 44) 
v_lshrrev_b32 v12, 16, v16 // (op 44) 
v_lshrrev_b32 v13, 16, v17 // (op 44) 

v_lshrrev_b32 v8, 6, v1 // (op 1460) 
s_mul_i32 s52, s16, s8 // (op 1460) 
s_add_u32 s52, s52, s0 // (op 1460) 
v_and_b32 v14, 63, v1 // (op 1460) 
s_mul_hi_u32 s53, s52, s13 // (op 1460) 
v_lshrrev_b32 v15, 4, v14 // (op 1460) 
s_sub_u32 s54, s52, s53 // (op 1460) 
v_lshlrev_b32 v16, 2, v15 // (op 1460) 

s_lshr_b32 s55, s54, 1 // (op 1460) 

v_lshl_add_u32 v15, v8, 4, v16 // (op 1460) 


s_add_u32 s54, s55, s53 // (op 1460) 


v_lshrrev_b32 v8, 2, v14 // (op 1460) 

s_lshr_b32 s53, s54, s11 // (op 1460) 

v_and_b32 v14, 3, v8 // (op 1460) 

s_and_b32 s54, s11, 2147483648 // (op 1460) 
v_and_b32 v8, 3, v1 // (op 1460) 
s_cmp_eq_i32 s54, 0 // (op 1460) 
s_mov_b32 s55, scc // (op 1460) 

v_lshl_add_u32 v16, v14, 2, v8 // (op 1460) 


s_cmp_eq_i32 s55, 0 // (op 1460) 
s_cselect_b32 s54, s52, s53 // (op 1460) 



s_mul_hi_u32 s55, s54, s12 // (op 1460) 
s_sub_u32 s52, s54, s55 // (op 1460) 
s_lshr_b32 s53, s52, 1 // (op 1460) 

s_add_u32 s52, s53, s55 // (op 1460) 


s_lshr_b32 s53, s52, s10 // (op 1460) 

s_and_b32 s52, s10, 2147483648 // (op 1460) 
s_cmp_eq_i32 s52, 0 // (op 1460) 
s_mov_b32 s55, scc // (op 1460) 

s_cmp_eq_i32 s55, 0 // (op 1460) 
s_cselect_b32 s52, s54, s53 // (op 1460) 


s_mul_i32 s53, s52, s14 // (op 1460) 
v_lshl_add_u32 v8, s52, 5, v15 // (op 1460) 


s_sub_u32 s52, s54, s53 // (op 1460) 


v_lshl_add_u32 v14, s52, 4, v16 // (op 1460) 


v_mul_lo_u32 v15, v14, s46 // (op 1460) 

v_add_lshl_u32 v14, v8, v15, 1 // (op 1460) 


s_mul_i32 s53, s46, s23 // (op 1462) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1462) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1462) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1462) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1462) most significant: sum
s_add_u32 s53, s53, s55 // (op 1462) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 1462) 

s_mov_b32 s76, s48 // (op 1462) reg expression
s_mov_b32 s77, s49 // (op 1462) reg expression
s_mov_b32 s78, s54 // (op 1462) reg expression
s_mov_b32 s79, 131072 // (op 1462) call()


v_pack_b32_f16 v16, v10, v11 // (op 46) 
v_pack_b32_f16 v17, v12, v13 // (op 46) 


buffer_store_dwordx2 v[16:17], v14, s[76:79], 0 offen // (op 46) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_108 // (op 108) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_108
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_108:



buffer_load_dwordx2 v[10:11], v9, s[56:59], 0 offen // (op 121) Load value
v_accvgpr_read v12, a0 // (op 117) 
v_accvgpr_read v13, a1 // (op 117) 
v_accvgpr_read v14, a2 // (op 117) 
v_accvgpr_read v15, a3 // (op 117) 
v_mul_f32 v16, v7, v12 // (op 117) 
v_mul_f32 v17, v7, v13 // (op 117) 
v_mul_f32 v18, v7, v14 // (op 117) 
v_mul_f32 v19, v7, v15 // (op 117) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 122) 
v_lshrrev_b32 v13, 16, v10 // (op 122) 
v_lshlrev_b32 v12, 16, v12 // (op 122) 
v_lshlrev_b32 v13, 16, v13 // (op 122) 
v_mul_f32 v20, v6, v12 // (op 122) 
v_mul_f32 v21, v6, v13 // (op 122) 
v_and_b32 v12, 65535, v11 // (op 122) 
v_lshrrev_b32 v13, 16, v11 // (op 122) 
v_lshlrev_b32 v12, 16, v12 // (op 122) 
v_lshlrev_b32 v13, 16, v13 // (op 122) 
v_mul_f32 v22, v6, v12 // (op 122) 
v_mul_f32 v23, v6, v13 // (op 122) 


v_add_f32 v10, v20, v16 // (op 118) 
v_add_f32 v11, v21, v17 // (op 118) 
v_add_f32 v12, v22, v18 // (op 118) 
v_add_f32 v13, v23, v19 // (op 118) 


v_lshrrev_b32 v14, 16, v10 // (op 119) 
v_lshrrev_b32 v15, 16, v11 // (op 119) 
v_lshrrev_b32 v16, 16, v12 // (op 119) 
v_lshrrev_b32 v17, 16, v13 // (op 119) 

v_lshrrev_b32 v8, 6, v1 // (op 1477) 
s_mul_i32 s52, s16, s8 // (op 1477) 
s_add_u32 s52, s52, s0 // (op 1477) 
v_and_b32 v10, 63, v1 // (op 1477) 
s_mul_hi_u32 s53, s52, s13 // (op 1477) 
v_lshrrev_b32 v11, 4, v10 // (op 1477) 
s_sub_u32 s54, s52, s53 // (op 1477) 
v_lshlrev_b32 v12, 2, v11 // (op 1477) 

s_lshr_b32 s55, s54, 1 // (op 1477) 

v_lshl_add_u32 v11, v8, 4, v12 // (op 1477) 


s_add_u32 s54, s55, s53 // (op 1477) 


v_lshrrev_b32 v8, 2, v10 // (op 1477) 

s_lshr_b32 s53, s54, s11 // (op 1477) 

v_and_b32 v10, 3, v8 // (op 1477) 

s_and_b32 s54, s11, 2147483648 // (op 1477) 
v_and_b32 v8, 3, v1 // (op 1477) 
s_cmp_eq_i32 s54, 0 // (op 1477) 
s_mov_b32 s55, scc // (op 1477) 

v_lshl_add_u32 v12, v10, 2, v8 // (op 1477) 


s_cmp_eq_i32 s55, 0 // (op 1477) 
s_cselect_b32 s54, s52, s53 // (op 1477) 



s_mul_hi_u32 s55, s54, s12 // (op 1477) 
s_sub_u32 s52, s54, s55 // (op 1477) 
s_lshr_b32 s53, s52, 1 // (op 1477) 

s_add_u32 s52, s53, s55 // (op 1477) 


s_lshr_b32 s53, s52, s10 // (op 1477) 

s_and_b32 s52, s10, 2147483648 // (op 1477) 
s_cmp_eq_i32 s52, 0 // (op 1477) 
s_mov_b32 s55, scc // (op 1477) 

s_cmp_eq_i32 s55, 0 // (op 1477) 
s_cselect_b32 s52, s54, s53 // (op 1477) 


s_mul_i32 s53, s52, s14 // (op 1477) 
v_lshl_add_u32 v8, s52, 5, v11 // (op 1477) 


s_sub_u32 s52, s54, s53 // (op 1477) 


v_lshl_add_u32 v13, s52, 4, v12 // (op 1477) 


v_mul_lo_u32 v10, v13, s46 // (op 1477) 

v_add_lshl_u32 v11, v8, v10, 1 // (op 1477) 


s_mul_i32 s53, s46, s23 // (op 1479) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1479) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1479) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1479) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1479) most significant: sum
s_add_u32 s53, s53, s55 // (op 1479) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 1479) 

s_mov_b32 s76, s48 // (op 1479) reg expression
s_mov_b32 s77, s49 // (op 1479) reg expression
s_mov_b32 s78, s54 // (op 1479) reg expression
s_mov_b32 s79, 131072 // (op 1479) call()


v_pack_b32_f16 v12, v14, v15 // (op 120) 
v_pack_b32_f16 v13, v16, v17 // (op 120) 


buffer_store_dwordx2 v[12:13], v11, s[76:79], 0 offen // (op 120) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_108:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_336 // (op 336) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_336
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_336:
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_336:

s_add_u32 s0, s0, s7 // (op 251) 
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
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_SKStreamTileLoop_249 // (op 249) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_SKStreamTileLoop_249 if true)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_249:











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
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_404 // (op 404) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_404 if false)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_DPStreamTileLoop_404:
v_lshrrev_b32 v0, 6, v1 // (op 1388) 
s_mul_i32 s15, s16, s50 // (op 1388) 
s_add_u32 s15, s15, s0 // (op 1388) 
v_and_b32 v5, 63, v1 // (op 1388) 
s_add_u32 s6, s15, s9 // (op 1388) 

v_lshrrev_b32 v8, 4, v5 // (op 1388) 
s_mul_hi_u32 s15, s6, s13 // (op 1388) 
v_lshlrev_b32 v9, 2, v8 // (op 1388) 

s_sub_u32 s7, s6, s15 // (op 1388) 
v_lshl_add_u32 v8, v0, 4, v9 // (op 1388) 


s_lshr_b32 s8, s7, 1 // (op 1388) 

v_lshrrev_b32 v0, 2, v5 // (op 1388) 

s_add_u32 s7, s8, s15 // (op 1388) 


v_and_b32 v5, 3, v0 // (op 1388) 

s_lshr_b32 s8, s7, s11 // (op 1388) 

v_and_b32 v0, 3, v1 // (op 1388) 
s_and_b32 s7, s11, 2147483648 // (op 1388) 
v_lshl_add_u32 v9, v5, 2, v0 // (op 1388) 


s_cmp_eq_i32 s7, 0 // (op 1388) 
s_mov_b32 s15, scc // (op 1388) 

s_cmp_eq_i32 s15, 0 // (op 1388) 
s_cselect_b32 s7, s6, s8 // (op 1388) 



s_mul_hi_u32 s6, s7, s12 // (op 1388) 
s_sub_u32 s8, s7, s6 // (op 1388) 
s_lshr_b32 s15, s8, 1 // (op 1388) 

s_add_u32 s8, s15, s6 // (op 1388) 


s_lshr_b32 s6, s8, s10 // (op 1388) 

s_and_b32 s8, s10, 2147483648 // (op 1388) 
s_cmp_eq_i32 s8, 0 // (op 1388) 
s_mov_b32 s15, scc // (op 1388) 

s_cmp_eq_i32 s15, 0 // (op 1388) 
s_cselect_b32 s8, s7, s6 // (op 1388) 


s_mul_i32 s6, s8, s14 // (op 1388) 
v_lshl_add_u32 v0, s8, 5, v8 // (op 1388) 


s_sub_u32 s8, s7, s6 // (op 1388) 


v_lshl_add_u32 v5, s8, 4, v9 // (op 1388) 


v_mul_lo_u32 v8, v5, s4 // (op 1388) 

v_add_lshl_u32 v5, v0, v8, 1 // (op 1388) 


s_mul_i32 s19, s4, s23 // (op 1390) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 1390) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 1390) least significant
s_mul_hi_u32 s6, s4, s22 // (op 1390) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1390) most significant: sum
s_add_u32 s19, s19, s6 // (op 1390) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 1 // (op 1390) 

s_mov_b32 s56, s20 // (op 1390) reg expression
s_mov_b32 s57, s21 // (op 1390) reg expression
s_mov_b32 s58, s6 // (op 1390) reg expression
s_mov_b32 s59, 131072 // (op 1390) call()

v_accvgpr_write a0, 0.00000 // (op 157) call()
v_accvgpr_write a1, 0.00000 // (op 157) call()
v_accvgpr_write a2, 0.00000 // (op 157) call()
v_accvgpr_write a3, 0.00000 // (op 157) call()
v_lshrrev_b32 v0, 6, v1 // (op 948) 
s_mul_i32 s15, s16, s50 // (op 948) 
s_add_u32 s15, s15, s0 // (op 948) 
v_and_b32 v8, 15, v1 // (op 948) 
s_add_u32 s6, s15, s9 // (op 948) 

v_lshl_add_u32 v9, v0, 4, v8 // (op 948) 


s_mul_hi_u32 s15, s6, s13 // (op 948) 
v_and_b32 v0, 63, v1 // (op 948) 
s_sub_u32 s7, s6, s15 // (op 948) 
v_lshrrev_b32 v8, 4, v0 // (op 948) 

s_lshr_b32 s8, s7, 1 // (op 948) 

v_lshlrev_b32 v0, 5, v8 // (op 948) 

s_add_u32 s7, s8, s15 // (op 948) 


s_lshr_b32 s8, s7, s11 // (op 948) 

s_and_b32 s7, s11, 2147483648 // (op 948) 
s_cmp_eq_i32 s7, 0 // (op 948) 
s_mov_b32 s15, scc // (op 948) 

s_cmp_eq_i32 s15, 0 // (op 948) 
s_cselect_b32 s7, s6, s8 // (op 948) 


s_mul_hi_u32 s8, s7, s12 // (op 948) 
s_sub_u32 s15, s7, s8 // (op 948) 
s_lshr_b32 s18, s15, 1 // (op 948) 

s_add_u32 s15, s18, s8 // (op 948) 


s_lshr_b32 s8, s15, s10 // (op 948) 

s_and_b32 s15, s10, 2147483648 // (op 948) 
s_cmp_eq_i32 s15, 0 // (op 948) 
s_mov_b32 s18, scc // (op 948) 

s_cmp_eq_i32 s18, 0 // (op 948) 
s_cselect_b32 s15, s7, s8 // (op 948) 


v_lshl_add_u32 v8, s15, 5, v9 // (op 948) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 948) 
s_mul_i32 s8, s7, s18 // (op 948) 


s_sub_u32 s7, s6, s8 // (op 948) 


v_lshl_add_u32 v9, s7, 7, v0 // (op 948) 


v_mov_b32 v11, 0 // (op 948) convert
v_mov_b32 v10, v9 // (op 948) convert

v_mul_lo_u32 v13, v8, s25 // (op 948) most significant: low of low * high
v_mul_hi_u32 v0, v8, s24 // (op 948) most significant: high of low * low
v_mul_lo_u32 v12, v8, s24 // (op 948) least significant: low of low * low
v_add_u32 v13, v13, v0 // (op 948) most significant: sum

v_add_co_u32 v12, s[18:19], v12, v10 // (op 948) least significant half
v_addc_co_u32 v13, s[18:19], v13, v11, s[18:19] // (op 948) most significant half



v_lshlrev_b64 v[8:9], 2, v[12:13] // (op 948) 

v_lshrrev_b64 v[10:11], 3, v[8:9] // (op 948) 

v_mov_b32 v0, v10 // (op 948) convert

s_mul_i32 s19, s24, s29 // (op 950) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 950) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 950) least significant
s_mul_hi_u32 s6, s24, s28 // (op 950) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 950) most significant: sum
s_add_u32 s19, s19, s6 // (op 950) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 950) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 950) 

s_mov_b32 s60, s26 // (op 950) reg expression
s_mov_b32 s61, s27 // (op 950) reg expression
s_mov_b32 s62, s18 // (op 950) reg expression
s_mov_b32 s63, 131072 // (op 950) call()

v_lshrrev_b32 v8, 6, v1 // (op 967) 
s_mul_i32 s15, s16, s50 // (op 967) 
s_add_u32 s15, s15, s0 // (op 967) 
v_and_b32 v9, 15, v1 // (op 967) 
s_add_u32 s6, s15, s9 // (op 967) 

v_lshl_add_u32 v10, v8, 4, v9 // (op 967) 


s_mul_hi_u32 s15, s6, s13 // (op 967) 
v_and_b32 v8, 63, v1 // (op 967) 
s_sub_u32 s7, s6, s15 // (op 967) 
v_lshrrev_b32 v9, 4, v8 // (op 967) 

s_lshr_b32 s8, s7, 1 // (op 967) 

v_lshlrev_b32 v8, 5, v9 // (op 967) 

s_add_u32 s7, s8, s15 // (op 967) 


s_lshr_b32 s8, s7, s11 // (op 967) 

s_and_b32 s7, s11, 2147483648 // (op 967) 
s_cmp_eq_i32 s7, 0 // (op 967) 
s_mov_b32 s15, scc // (op 967) 

s_cmp_eq_i32 s15, 0 // (op 967) 
s_cselect_b32 s7, s6, s8 // (op 967) 


s_mul_hi_u32 s8, s7, s12 // (op 967) 
s_sub_u32 s15, s7, s8 // (op 967) 
s_lshr_b32 s18, s15, 1 // (op 967) 

s_add_u32 s15, s18, s8 // (op 967) 


s_lshr_b32 s8, s15, s10 // (op 967) 

s_and_b32 s15, s10, 2147483648 // (op 967) 
s_cmp_eq_i32 s15, 0 // (op 967) 
s_mov_b32 s18, scc // (op 967) 

s_cmp_eq_i32 s18, 0 // (op 967) 
s_cselect_b32 s15, s7, s8 // (op 967) 


v_lshl_add_u32 v9, s15, 5, v10 // (op 967) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 967) 
s_mul_i32 s8, s7, s18 // (op 967) 


s_sub_u32 s7, s6, s8 // (op 967) 


v_lshl_add_u32 v10, s7, 7, v8 // (op 967) 


v_mov_b32 v13, 0 // (op 967) convert
v_mov_b32 v12, v10 // (op 967) convert

v_mul_lo_u32 v11, v9, s25 // (op 967) most significant: low of low * high
v_mul_hi_u32 v8, v9, s24 // (op 967) most significant: high of low * low
v_mul_lo_u32 v10, v9, s24 // (op 967) least significant: low of low * low
v_add_u32 v11, v11, v8 // (op 967) most significant: sum

v_add_co_u32 v10, s[18:19], v10, v12 // (op 967) least significant half
v_addc_co_u32 v11, s[18:19], v11, v13, s[18:19] // (op 967) most significant half



v_lshlrev_b64 v[8:9], 2, v[10:11] // (op 967) 

v_lshrrev_b64 v[10:11], 3, v[8:9] // (op 967) 

v_mov_b32 v8, v10 // (op 967) convert

s_mul_i32 s19, s24, s29 // (op 969) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 969) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 969) least significant
s_mul_hi_u32 s6, s24, s28 // (op 969) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 969) most significant: sum
s_add_u32 s19, s19, s6 // (op 969) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 969) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 969) 

s_mov_b32 s60, s26 // (op 969) reg expression
s_mov_b32 s61, s27 // (op 969) reg expression
s_mov_b32 s62, s18 // (op 969) reg expression
s_mov_b32 s63, 131072 // (op 969) call()

v_and_b32 v9, 15, v1 // (op 1076) 
s_mul_i32 s15, s16, s50 // (op 1076) 
s_add_u32 s15, s15, s0 // (op 1076) 
v_and_b32 v10, 48, v1 // (op 1076) 
s_add_u32 s6, s15, s9 // (op 1076) 

s_mul_hi_u32 s15, s6, s13 // (op 1076) 
s_sub_u32 s7, s6, s15 // (op 1076) 
s_lshr_b32 s8, s7, 1 // (op 1076) 

s_add_u32 s7, s8, s15 // (op 1076) 


s_lshr_b32 s8, s7, s11 // (op 1076) 

s_and_b32 s7, s11, 2147483648 // (op 1076) 
s_cmp_eq_i32 s7, 0 // (op 1076) 
s_mov_b32 s15, scc // (op 1076) 

s_cmp_eq_i32 s15, 0 // (op 1076) 
s_cselect_b32 s7, s6, s8 // (op 1076) 


s_mul_hi_u32 s8, s7, s12 // (op 1076) 
s_sub_u32 s15, s7, s8 // (op 1076) 
s_lshr_b32 s18, s15, 1 // (op 1076) 

s_add_u32 s15, s18, s8 // (op 1076) 


s_lshr_b32 s8, s15, s10 // (op 1076) 

s_and_b32 s15, s10, 2147483648 // (op 1076) 
s_cmp_eq_i32 s15, 0 // (op 1076) 
s_mov_b32 s18, scc // (op 1076) 

s_cmp_eq_i32 s18, 0 // (op 1076) 
s_cselect_b32 s15, s7, s8 // (op 1076) 


s_mul_i32 s8, s15, s14 // (op 1076) 

s_sub_u32 s15, s7, s8 // (op 1076) 

v_lshl_add_u32 v11, s15, 4, v9 // (op 1076) 


v_mul_lo_u32 v9, v11, s30 // (op 1076) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1076) 
s_mul_i32 s8, s7, s18 // (op 1076) 


s_sub_u32 s7, s6, s8 // (op 1076) 


v_lshl_add_u32 v11, s7, 7, v10 // (op 1076) 


v_add_u32 v10, v11, v9 // (op 1076) 


s_mul_i32 s19, s30, s23 // (op 1078) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 1078) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 1078) least significant
s_mul_hi_u32 s6, s30, s22 // (op 1078) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1078) most significant: sum
s_add_u32 s19, s19, s6 // (op 1078) most significant: sum


s_mov_b32 s64, s32 // (op 1078) reg expression
s_mov_b32 s65, s33 // (op 1078) reg expression
s_mov_b32 s66, s18 // (op 1078) reg expression
s_mov_b32 s67, 131072 // (op 1078) call()

v_and_b32 v9, 15, v1 // (op 1095) 
s_mul_i32 s15, s16, s50 // (op 1095) 
s_add_u32 s15, s15, s0 // (op 1095) 
v_and_b32 v11, 48, v1 // (op 1095) 
s_add_u32 s6, s15, s9 // (op 1095) 

s_mul_hi_u32 s15, s6, s13 // (op 1095) 
s_sub_u32 s7, s6, s15 // (op 1095) 
s_lshr_b32 s8, s7, 1 // (op 1095) 

s_add_u32 s7, s8, s15 // (op 1095) 


s_lshr_b32 s8, s7, s11 // (op 1095) 

s_and_b32 s7, s11, 2147483648 // (op 1095) 
s_cmp_eq_i32 s7, 0 // (op 1095) 
s_mov_b32 s15, scc // (op 1095) 

s_cmp_eq_i32 s15, 0 // (op 1095) 
s_cselect_b32 s7, s6, s8 // (op 1095) 


s_mul_hi_u32 s8, s7, s12 // (op 1095) 
s_sub_u32 s15, s7, s8 // (op 1095) 
s_lshr_b32 s18, s15, 1 // (op 1095) 

s_add_u32 s15, s18, s8 // (op 1095) 


s_lshr_b32 s8, s15, s10 // (op 1095) 

s_and_b32 s15, s10, 2147483648 // (op 1095) 
s_cmp_eq_i32 s15, 0 // (op 1095) 
s_mov_b32 s18, scc // (op 1095) 

s_cmp_eq_i32 s18, 0 // (op 1095) 
s_cselect_b32 s15, s7, s8 // (op 1095) 


s_mul_i32 s8, s15, s14 // (op 1095) 

s_sub_u32 s15, s7, s8 // (op 1095) 

v_lshl_add_u32 v12, s15, 4, v9 // (op 1095) 


v_mul_lo_u32 v9, v12, s30 // (op 1095) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1095) 
s_mul_i32 s8, s7, s18 // (op 1095) 


s_sub_u32 s7, s6, s8 // (op 1095) 


v_lshl_add_u32 v12, s7, 7, v11 // (op 1095) 


v_add_u32 v11, v12, v9 // (op 1095) 


s_mul_i32 s19, s30, s23 // (op 1097) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 1097) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 1097) least significant
s_mul_hi_u32 s6, s30, s22 // (op 1097) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1097) most significant: sum
s_add_u32 s19, s19, s6 // (op 1097) most significant: sum


s_mov_b32 s64, s32 // (op 1097) reg expression
s_mov_b32 s65, s33 // (op 1097) reg expression
s_mov_b32 s66, s18 // (op 1097) reg expression
s_mov_b32 s67, 131072 // (op 1097) call()

v_lshrrev_b32 v9, 2, v1 // (op 1204) 
s_mul_i32 s15, s16, s50 // (op 1204) 
s_add_u32 s15, s15, s0 // (op 1204) 
v_and_b32 v12, -4, v1 // (op 1204) 
s_add_u32 s6, s15, s9 // (op 1204) 

v_sub_u32 v13, v1, v12 // (op 1204) 

s_mul_hi_u32 s15, s6, s13 // (op 1204) 
s_sub_u32 s7, s6, s15 // (op 1204) 
s_lshr_b32 s8, s7, 1 // (op 1204) 

s_add_u32 s7, s8, s15 // (op 1204) 


s_lshr_b32 s8, s7, s11 // (op 1204) 

s_and_b32 s7, s11, 2147483648 // (op 1204) 
s_cmp_eq_i32 s7, 0 // (op 1204) 
s_mov_b32 s15, scc // (op 1204) 

s_cmp_eq_i32 s15, 0 // (op 1204) 
s_cselect_b32 s7, s6, s8 // (op 1204) 


s_mul_hi_u32 s8, s7, s12 // (op 1204) 
s_sub_u32 s15, s7, s8 // (op 1204) 
s_lshr_b32 s18, s15, 1 // (op 1204) 

s_add_u32 s15, s18, s8 // (op 1204) 


s_lshr_b32 s8, s15, s10 // (op 1204) 

s_and_b32 s15, s10, 2147483648 // (op 1204) 
s_cmp_eq_i32 s15, 0 // (op 1204) 
s_mov_b32 s18, scc // (op 1204) 

s_cmp_eq_i32 s18, 0 // (op 1204) 
s_cselect_b32 s15, s7, s8 // (op 1204) 


v_lshl_add_u32 v12, s15, 5, v9 // (op 1204) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1204) 
s_mul_i32 s8, s7, s18 // (op 1204) 


s_sub_u32 s7, s6, s8 // (op 1204) 


v_lshl_add_u32 v9, s7, 2, v13 // (op 1204) 


v_mul_lo_u32 v13, v12, s34 // (op 1204) 
v_add_u32 v13, v13, v9 // (op 1204) 


s_mul_i32 s19, s34, s29 // (op 1206) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1206) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1206) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1206) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1206) most significant: sum
s_add_u32 s19, s19, s6 // (op 1206) most significant: sum


s_mov_b32 s68, s36 // (op 1206) reg expression
s_mov_b32 s69, s37 // (op 1206) reg expression
s_mov_b32 s70, s18 // (op 1206) reg expression
s_mov_b32 s71, 131072 // (op 1206) call()

v_lshrrev_b32 v9, 2, v1 // (op 1223) 
s_mul_i32 s15, s16, s50 // (op 1223) 
s_add_u32 s15, s15, s0 // (op 1223) 
v_and_b32 v12, -4, v1 // (op 1223) 
s_add_u32 s6, s15, s9 // (op 1223) 

v_sub_u32 v14, v1, v12 // (op 1223) 

s_mul_hi_u32 s15, s6, s13 // (op 1223) 
s_sub_u32 s7, s6, s15 // (op 1223) 
s_lshr_b32 s8, s7, 1 // (op 1223) 

s_add_u32 s7, s8, s15 // (op 1223) 


s_lshr_b32 s8, s7, s11 // (op 1223) 

s_and_b32 s7, s11, 2147483648 // (op 1223) 
s_cmp_eq_i32 s7, 0 // (op 1223) 
s_mov_b32 s15, scc // (op 1223) 

s_cmp_eq_i32 s15, 0 // (op 1223) 
s_cselect_b32 s7, s6, s8 // (op 1223) 


s_mul_hi_u32 s8, s7, s12 // (op 1223) 
s_sub_u32 s15, s7, s8 // (op 1223) 
s_lshr_b32 s18, s15, 1 // (op 1223) 

s_add_u32 s15, s18, s8 // (op 1223) 


s_lshr_b32 s8, s15, s10 // (op 1223) 

s_and_b32 s15, s10, 2147483648 // (op 1223) 
s_cmp_eq_i32 s15, 0 // (op 1223) 
s_mov_b32 s18, scc // (op 1223) 

s_cmp_eq_i32 s18, 0 // (op 1223) 
s_cselect_b32 s15, s7, s8 // (op 1223) 


v_lshl_add_u32 v12, s15, 5, v9 // (op 1223) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1223) 
s_mul_i32 s8, s7, s18 // (op 1223) 


s_sub_u32 s7, s6, s8 // (op 1223) 


v_lshl_add_u32 v9, s7, 2, v14 // (op 1223) 


v_mul_lo_u32 v14, v12, s34 // (op 1223) 
v_add_u32 v14, v14, v9 // (op 1223) 


s_mul_i32 s19, s34, s29 // (op 1225) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1225) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1225) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1225) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1225) most significant: sum
s_add_u32 s19, s19, s6 // (op 1225) most significant: sum


s_mov_b32 s68, s36 // (op 1225) reg expression
s_mov_b32 s69, s37 // (op 1225) reg expression
s_mov_b32 s70, s18 // (op 1225) reg expression
s_mov_b32 s71, 131072 // (op 1225) call()

v_and_b32 v9, 15, v1 // (op 1332) 
s_mul_i32 s15, s16, s50 // (op 1332) 
s_add_u32 s15, s15, s0 // (op 1332) 
v_and_b32 v12, 63, v1 // (op 1332) 
s_add_u32 s6, s15, s9 // (op 1332) 

v_lshrrev_b32 v15, 4, v12 // (op 1332) 

s_mul_hi_u32 s15, s6, s13 // (op 1332) 
s_sub_u32 s7, s6, s15 // (op 1332) 
s_lshr_b32 s8, s7, 1 // (op 1332) 

s_add_u32 s7, s8, s15 // (op 1332) 


s_lshr_b32 s8, s7, s11 // (op 1332) 

s_and_b32 s7, s11, 2147483648 // (op 1332) 
s_cmp_eq_i32 s7, 0 // (op 1332) 
s_mov_b32 s15, scc // (op 1332) 

s_cmp_eq_i32 s15, 0 // (op 1332) 
s_cselect_b32 s7, s6, s8 // (op 1332) 


s_mul_hi_u32 s8, s7, s12 // (op 1332) 
s_sub_u32 s15, s7, s8 // (op 1332) 
s_lshr_b32 s18, s15, 1 // (op 1332) 

s_add_u32 s15, s18, s8 // (op 1332) 


s_lshr_b32 s8, s15, s10 // (op 1332) 

s_and_b32 s15, s10, 2147483648 // (op 1332) 
s_cmp_eq_i32 s15, 0 // (op 1332) 
s_mov_b32 s18, scc // (op 1332) 

s_cmp_eq_i32 s18, 0 // (op 1332) 
s_cselect_b32 s15, s7, s8 // (op 1332) 


s_mul_i32 s8, s15, s14 // (op 1332) 

s_sub_u32 s15, s7, s8 // (op 1332) 

v_lshl_add_u32 v12, s15, 4, v9 // (op 1332) 


v_mul_lo_u32 v9, v12, s38 // (op 1332) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1332) 
s_mul_i32 s8, s7, s18 // (op 1332) 


s_sub_u32 s7, s6, s8 // (op 1332) 


v_lshl_add_u32 v12, s7, 2, v15 // (op 1332) 


v_add_u32 v15, v12, v9 // (op 1332) 


s_mul_i32 s19, s38, s23 // (op 1334) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 1334) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 1334) least significant
s_mul_hi_u32 s6, s38, s22 // (op 1334) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1334) most significant: sum
s_add_u32 s19, s19, s6 // (op 1334) most significant: sum


s_mov_b32 s72, s40 // (op 1334) reg expression
s_mov_b32 s73, s41 // (op 1334) reg expression
s_mov_b32 s74, s18 // (op 1334) reg expression
s_mov_b32 s75, 131072 // (op 1334) call()

v_and_b32 v9, 15, v1 // (op 1351) 
s_mul_i32 s15, s16, s50 // (op 1351) 
s_add_u32 s15, s15, s0 // (op 1351) 
v_and_b32 v12, 63, v1 // (op 1351) 
s_add_u32 s6, s15, s9 // (op 1351) 

v_lshrrev_b32 v16, 4, v12 // (op 1351) 

s_mul_hi_u32 s15, s6, s13 // (op 1351) 
s_sub_u32 s7, s6, s15 // (op 1351) 
s_lshr_b32 s8, s7, 1 // (op 1351) 

s_add_u32 s7, s8, s15 // (op 1351) 


s_lshr_b32 s8, s7, s11 // (op 1351) 

s_and_b32 s7, s11, 2147483648 // (op 1351) 
s_cmp_eq_i32 s7, 0 // (op 1351) 
s_mov_b32 s15, scc // (op 1351) 

s_cmp_eq_i32 s15, 0 // (op 1351) 
s_cselect_b32 s7, s6, s8 // (op 1351) 


s_mul_hi_u32 s8, s7, s12 // (op 1351) 
s_sub_u32 s15, s7, s8 // (op 1351) 
s_lshr_b32 s18, s15, 1 // (op 1351) 

s_add_u32 s15, s18, s8 // (op 1351) 


s_lshr_b32 s8, s15, s10 // (op 1351) 

s_and_b32 s15, s10, 2147483648 // (op 1351) 
s_cmp_eq_i32 s15, 0 // (op 1351) 
s_mov_b32 s18, scc // (op 1351) 

s_cmp_eq_i32 s18, 0 // (op 1351) 
s_cselect_b32 s15, s7, s8 // (op 1351) 


s_mul_i32 s8, s15, s14 // (op 1351) 

s_sub_u32 s15, s7, s8 // (op 1351) 

v_lshl_add_u32 v12, s15, 4, v9 // (op 1351) 


v_mul_lo_u32 v9, v12, s38 // (op 1351) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1351) 
s_mul_i32 s8, s7, s18 // (op 1351) 


s_sub_u32 s7, s6, s8 // (op 1351) 


v_lshl_add_u32 v12, s7, 2, v16 // (op 1351) 


v_add_u32 v16, v12, v9 // (op 1351) 


s_mul_i32 s19, s38, s23 // (op 1353) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 1353) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 1353) least significant
s_mul_hi_u32 s6, s38, s22 // (op 1353) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1353) most significant: sum
s_add_u32 s19, s19, s6 // (op 1353) most significant: sum


s_mov_b32 s72, s40 // (op 1353) reg expression
s_mov_b32 s73, s41 // (op 1353) reg expression
s_mov_b32 s74, s18 // (op 1353) reg expression
s_mov_b32 s75, 131072 // (op 1353) call()




buffer_load_ubyte v9, v15, s[72:75], 0 offen offset: 0 // (op 771) Load value



buffer_load_dwordx4 v[18:21], v0, s[60:63], 0 offen // (op 776) Load value



buffer_load_dwordx4 v[22:25], v10, s[64:67], 0 offen // (op 781) Load value
buffer_load_dwordx4 v[26:29], v10, s[64:67], 0 offen offset:64 // (op 781) Load value



buffer_load_ubyte v12, v13, s[68:71], 0 offen offset: 0 // (op 785) Load value



buffer_load_ubyte v17, v15, s[72:75], 0 offen offset:4 // (op 789) Load value




buffer_load_dwordx4 v[30:33], v0, s[60:63], 0 offen offset:64 // (op 794) Load value




buffer_load_dwordx4 v[34:37], v10, s[64:67], 0 offen offset:128 // (op 799) Load value
buffer_load_dwordx4 v[38:41], v10, s[64:67], 0 offen offset:192 // (op 799) Load value




buffer_load_ubyte v0, v13, s[68:71], 0 offen offset:4 // (op 803) Load value

v_and_b32 v10, -4, v1 // (op 1557) 
v_sub_u32 v13, v1, v10 // (op 1557) 

v_and_b32 v10, 4294967292, v1 // (op 1557) 
v_add_u32 v15, v10, v13 // (op 1557) 


s_barrier  // (op 1724) 



s_waitcnt vmcnt(5)
ds_write_b8 v15, v12 // (op 805) Store local data 

v_and_b32 v10, -4, v1 // (op 1528) 
v_sub_u32 v13, v1, v10 // (op 1528) 

v_and_b32 v10, 4294967292, v1 // (op 1528) 
v_add_u32 v15, v10, v13 // (op 1528) 


v_and_b32 v10, -4, v1 // (op 1615) 
v_sub_u32 v13, v1, v10 // (op 1615) 

v_and_b32 v10, 4294967292, v1 // (op 1615) 
v_add_u32 v42, v10, v13 // (op 1615) 


s_mov_b32 s1, 0 // (op 159) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 158) 
s_mov_b32 s15, s18 // (op 158) convert

s_and_b32 s6, s15, 4294967294 // (op 158) 

s_cmp_lt_u32 s1, s6 // (op 158) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_158 // (op 158) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_158 if false)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoop_158:
s_add_u32 s1, s1, 2 // (op 160) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1721) 



ds_read_u8 v10, v3 // (op 164) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[22:29], a[0:3], v10, v9 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 167) 



v_mov_b32 v10, v16 // (op 166) 

buffer_load_ubyte v9, v10, s[72:75], 0 offen offset:8 // (op 166) Load value


v_mov_b32 v13, v8 // (op 169) 

buffer_load_dwordx4 v[18:21], v13, s[60:63], 0 offen offset:128 // (op 169) Load value


v_mov_b32 v43, v11 // (op 171) 

buffer_load_dwordx4 v[22:25], v43, s[64:67], 0 offen offset:256 // (op 171) Load value
buffer_load_dwordx4 v[26:29], v43, s[64:67], 0 offen offset:320 // (op 171) Load value


v_mov_b32 v44, v14 // (op 161) 

buffer_load_ubyte v12, v44, s[68:71], 0 offen offset:8 // (op 161) Load value
s_barrier  // (op 1740) 



s_waitcnt vmcnt(5)
ds_write_b8 v42, v0 offset:128 // (op 460) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1737) 



ds_read_u8 v45, v4 offset:128 // (op 462) Load local data 
s_barrier  // (op 1843) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:33], v[34:41], a[0:3], v45, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 465) 




buffer_load_ubyte v17, v10, s[72:75], 0 offen offset:12 // (op 464) Load value




v_add_u32 v16, 8, v16 // (op 1343) 
buffer_load_dwordx4 v[30:33], v13, s[60:63], 0 offen offset:192 // (op 467) Load value




v_add_u32 v8, 128, v8 // (op 959) 
buffer_load_dwordx4 v[34:37], v43, s[64:67], 0 offen offset:384 // (op 469) Load value
buffer_load_dwordx4 v[38:41], v43, s[64:67], 0 offen offset:448 // (op 469) Load value




v_add_u32 v11, 256, v11 // (op 1087) 
buffer_load_ubyte v0, v44, s[68:71], 0 offen offset:12 // (op 459) Load value

s_barrier  // (op 1727) 
v_add_u32 v14, 8, v14 // (op 1215) 



s_waitcnt vmcnt(5)
ds_write_b8 v15, v12 // (op 162) Store local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 158) 
s_mov_b32 s15, s18 // (op 158) convert

s_and_b32 s6, s15, 4294967294 // (op 158) 

s_cmp_lt_u32 s1, s6 // (op 158) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoop_158 // (op 158) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoop_158 if true)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_158:










v_lshrrev_b32 v8, 6, v1 // (op 853) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 853) 
v_and_b32 v9, 15, v1 // (op 853) 
s_and_b32 s15, s18, 4294967294 // (op 853) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 853) 


s_add_u32 s6, s0, s15 // (op 853) 

v_and_b32 v8, 63, v1 // (op 853) 
v_lshrrev_b32 v9, 4, v8 // (op 853) 

s_mul_i32 s15, s16, s50 // (op 853) 
s_add_u32 s15, s15, s6 // (op 853) 

v_lshlrev_b32 v8, 5, v9 // (op 853) 

s_add_u32 s6, s15, s9 // (op 853) 

s_mul_hi_u32 s15, s6, s13 // (op 853) 
s_sub_u32 s7, s6, s15 // (op 853) 
s_lshr_b32 s8, s7, 1 // (op 853) 

s_add_u32 s7, s8, s15 // (op 853) 


s_lshr_b32 s8, s7, s11 // (op 853) 

s_and_b32 s7, s11, 2147483648 // (op 853) 
s_cmp_eq_i32 s7, 0 // (op 853) 
s_mov_b32 s15, scc // (op 853) 

s_cmp_eq_i32 s15, 0 // (op 853) 
s_cselect_b32 s7, s6, s8 // (op 853) 


s_mul_hi_u32 s8, s7, s12 // (op 853) 
s_sub_u32 s15, s7, s8 // (op 853) 
s_lshr_b32 s42, s15, 1 // (op 853) 

s_add_u32 s15, s42, s8 // (op 853) 


s_lshr_b32 s8, s15, s10 // (op 853) 

s_and_b32 s15, s10, 2147483648 // (op 853) 
s_cmp_eq_i32 s15, 0 // (op 853) 
s_mov_b32 s42, scc // (op 853) 

s_cmp_eq_i32 s42, 0 // (op 853) 
s_cselect_b32 s15, s7, s8 // (op 853) 


s_mul_i32 s8, s7, s18 // (op 853) 


v_lshl_add_u32 v9, s15, 5, v10 // (op 853) 


s_sub_u32 s7, s6, s8 // (op 853) 


v_lshl_add_u32 v10, s7, 7, v8 // (op 853) 


v_mov_b32 v19, 0 // (op 853) convert
v_mov_b32 v18, v10 // (op 853) convert

v_mul_lo_u32 v11, v9, s25 // (op 853) most significant: low of low * high
v_mul_hi_u32 v8, v9, s24 // (op 853) most significant: high of low * low
v_mul_lo_u32 v10, v9, s24 // (op 853) least significant: low of low * low
v_add_u32 v11, v11, v8 // (op 853) most significant: sum

v_add_co_u32 v10, s[18:19], v10, v18 // (op 853) least significant half
v_addc_co_u32 v11, s[18:19], v11, v19, s[18:19] // (op 853) most significant half



v_lshlrev_b64 v[8:9], 2, v[10:11] // (op 853) 

v_lshrrev_b64 v[10:11], 3, v[8:9] // (op 853) 

v_mov_b32 v8, v10 // (op 853) convert

s_mul_i32 s19, s24, s29 // (op 855) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 855) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 855) least significant
s_mul_hi_u32 s6, s24, s28 // (op 855) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 855) most significant: sum
s_add_u32 s19, s19, s6 // (op 855) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 855) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 855) 

s_mov_b32 s60, s26 // (op 855) reg expression
s_mov_b32 s61, s27 // (op 855) reg expression
s_mov_b32 s62, s18 // (op 855) reg expression
s_mov_b32 s63, 131072 // (op 855) call()

v_and_b32 v9, 15, v1 // (op 989) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 989) 
v_and_b32 v10, 48, v1 // (op 989) 
s_and_b32 s15, s18, 4294967294 // (op 989) 
s_add_u32 s6, s0, s15 // (op 989) 

s_mul_i32 s15, s16, s50 // (op 989) 
s_add_u32 s15, s15, s6 // (op 989) 

s_add_u32 s6, s15, s9 // (op 989) 

s_mul_hi_u32 s15, s6, s13 // (op 989) 
s_sub_u32 s7, s6, s15 // (op 989) 
s_lshr_b32 s8, s7, 1 // (op 989) 

s_add_u32 s7, s8, s15 // (op 989) 


s_lshr_b32 s8, s7, s11 // (op 989) 

s_and_b32 s7, s11, 2147483648 // (op 989) 
s_cmp_eq_i32 s7, 0 // (op 989) 
s_mov_b32 s15, scc // (op 989) 

s_cmp_eq_i32 s15, 0 // (op 989) 
s_cselect_b32 s7, s6, s8 // (op 989) 


s_mul_hi_u32 s8, s7, s12 // (op 989) 
s_sub_u32 s15, s7, s8 // (op 989) 
s_lshr_b32 s42, s15, 1 // (op 989) 

s_add_u32 s15, s42, s8 // (op 989) 


s_lshr_b32 s8, s15, s10 // (op 989) 

s_and_b32 s15, s10, 2147483648 // (op 989) 
s_cmp_eq_i32 s15, 0 // (op 989) 
s_mov_b32 s42, scc // (op 989) 

s_cmp_eq_i32 s42, 0 // (op 989) 
s_cselect_b32 s15, s7, s8 // (op 989) 


s_mul_i32 s8, s15, s14 // (op 989) 

s_sub_u32 s15, s7, s8 // (op 989) 

v_lshl_add_u32 v11, s15, 4, v9 // (op 989) 


s_mul_i32 s8, s7, s18 // (op 989) 


v_mul_lo_u32 v9, v11, s30 // (op 989) 

s_sub_u32 s7, s6, s8 // (op 989) 


v_lshl_add_u32 v11, s7, 7, v10 // (op 989) 


v_add_u32 v10, v11, v9 // (op 989) 


s_mul_i32 s19, s30, s23 // (op 991) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 991) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 991) least significant
s_mul_hi_u32 s6, s30, s22 // (op 991) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 991) most significant: sum
s_add_u32 s19, s19, s6 // (op 991) most significant: sum


s_mov_b32 s64, s32 // (op 991) reg expression
s_mov_b32 s65, s33 // (op 991) reg expression
s_mov_b32 s66, s18 // (op 991) reg expression
s_mov_b32 s67, 131072 // (op 991) call()

v_lshrrev_b32 v9, 2, v1 // (op 1117) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1117) 
v_and_b32 v11, -4, v1 // (op 1117) 
s_and_b32 s15, s18, 4294967294 // (op 1117) 
v_sub_u32 v12, v1, v11 // (op 1117) 

s_add_u32 s6, s0, s15 // (op 1117) 

s_mul_i32 s15, s16, s50 // (op 1117) 
s_add_u32 s15, s15, s6 // (op 1117) 

s_add_u32 s6, s15, s9 // (op 1117) 

s_mul_hi_u32 s15, s6, s13 // (op 1117) 
s_sub_u32 s7, s6, s15 // (op 1117) 
s_lshr_b32 s8, s7, 1 // (op 1117) 

s_add_u32 s7, s8, s15 // (op 1117) 


s_lshr_b32 s8, s7, s11 // (op 1117) 

s_and_b32 s7, s11, 2147483648 // (op 1117) 
s_cmp_eq_i32 s7, 0 // (op 1117) 
s_mov_b32 s15, scc // (op 1117) 

s_cmp_eq_i32 s15, 0 // (op 1117) 
s_cselect_b32 s7, s6, s8 // (op 1117) 


s_mul_hi_u32 s8, s7, s12 // (op 1117) 
s_sub_u32 s15, s7, s8 // (op 1117) 
s_lshr_b32 s42, s15, 1 // (op 1117) 

s_add_u32 s15, s42, s8 // (op 1117) 


s_lshr_b32 s8, s15, s10 // (op 1117) 

s_and_b32 s15, s10, 2147483648 // (op 1117) 
s_cmp_eq_i32 s15, 0 // (op 1117) 
s_mov_b32 s42, scc // (op 1117) 

s_cmp_eq_i32 s42, 0 // (op 1117) 
s_cselect_b32 s15, s7, s8 // (op 1117) 


s_mul_i32 s8, s7, s18 // (op 1117) 


v_lshl_add_u32 v11, s15, 5, v9 // (op 1117) 


s_sub_u32 s7, s6, s8 // (op 1117) 


v_lshl_add_u32 v9, s7, 2, v12 // (op 1117) 


v_mul_lo_u32 v12, v11, s34 // (op 1117) 
v_add_u32 v12, v12, v9 // (op 1117) 


s_mul_i32 s19, s34, s29 // (op 1119) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1119) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1119) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1119) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1119) most significant: sum
s_add_u32 s19, s19, s6 // (op 1119) most significant: sum


s_mov_b32 s68, s36 // (op 1119) reg expression
s_mov_b32 s69, s37 // (op 1119) reg expression
s_mov_b32 s70, s18 // (op 1119) reg expression
s_mov_b32 s71, 131072 // (op 1119) call()

v_and_b32 v9, 15, v1 // (op 1245) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1245) 
v_and_b32 v11, 63, v1 // (op 1245) 
s_and_b32 s15, s18, 4294967294 // (op 1245) 
v_lshrrev_b32 v13, 4, v11 // (op 1245) 

s_add_u32 s6, s0, s15 // (op 1245) 

s_mul_i32 s15, s16, s50 // (op 1245) 
s_add_u32 s15, s15, s6 // (op 1245) 

s_add_u32 s6, s15, s9 // (op 1245) 

s_mul_hi_u32 s15, s6, s13 // (op 1245) 
s_sub_u32 s7, s6, s15 // (op 1245) 
s_lshr_b32 s8, s7, 1 // (op 1245) 

s_add_u32 s7, s8, s15 // (op 1245) 


s_lshr_b32 s8, s7, s11 // (op 1245) 

s_and_b32 s7, s11, 2147483648 // (op 1245) 
s_cmp_eq_i32 s7, 0 // (op 1245) 
s_mov_b32 s15, scc // (op 1245) 

s_cmp_eq_i32 s15, 0 // (op 1245) 
s_cselect_b32 s7, s6, s8 // (op 1245) 


s_mul_hi_u32 s8, s7, s12 // (op 1245) 
s_sub_u32 s15, s7, s8 // (op 1245) 
s_lshr_b32 s42, s15, 1 // (op 1245) 

s_add_u32 s15, s42, s8 // (op 1245) 


s_lshr_b32 s8, s15, s10 // (op 1245) 

s_and_b32 s15, s10, 2147483648 // (op 1245) 
s_cmp_eq_i32 s15, 0 // (op 1245) 
s_mov_b32 s42, scc // (op 1245) 

s_cmp_eq_i32 s42, 0 // (op 1245) 
s_cselect_b32 s15, s7, s8 // (op 1245) 


s_mul_i32 s8, s15, s14 // (op 1245) 

s_sub_u32 s15, s7, s8 // (op 1245) 

v_lshl_add_u32 v11, s15, 4, v9 // (op 1245) 


s_mul_i32 s8, s7, s18 // (op 1245) 


v_mul_lo_u32 v9, v11, s38 // (op 1245) 

s_sub_u32 s7, s6, s8 // (op 1245) 


v_lshl_add_u32 v11, s7, 2, v13 // (op 1245) 


v_add_u32 v13, v11, v9 // (op 1245) 


s_mul_i32 s19, s38, s23 // (op 1247) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 1247) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 1247) least significant
s_mul_hi_u32 s6, s38, s22 // (op 1247) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1247) most significant: sum
s_add_u32 s19, s19, s6 // (op 1247) most significant: sum


s_mov_b32 s72, s40 // (op 1247) reg expression
s_mov_b32 s73, s41 // (op 1247) reg expression
s_mov_b32 s74, s18 // (op 1247) reg expression
s_mov_b32 s75, 131072 // (op 1247) call()

v_and_b32 v9, -4, v1 // (op 1541) 
v_sub_u32 v11, v1, v9 // (op 1541) 

v_and_b32 v9, 4294967292, v1 // (op 1541) 
v_add_u32 v14, v9, v11 // (op 1541) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 413) 
s_and_b32 s15, s18, 4294967294 // (op 413) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 412) 
s_mov_b32 s6, s18 // (op 412) convert

s_cmp_lt_u32 s15, s6 // (op 412) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_412 // (op 412) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_412 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_412:


s_add_u32 s15, s15, 1 // (op 414) 
v_mov_b32 v0, v12 // (op 417) 

buffer_load_ubyte v9, v0, s[68:71], 0 offen offset: 0 // (op 417) Load value

v_add_u32 v12, 4, v12 // (op 1109) 


s_barrier  // (op 1730) 
v_mov_b32 v0, v8 // (op 425) 




buffer_load_dwordx4 v[16:19], v0, s[60:63], 0 offen // (op 425) Load value

v_add_u32 v8, 64, v8 // (op 843) 
s_waitcnt vmcnt(1)
ds_write_b8 v14, v9 // (op 418) Store local data 



v_mov_b32 v0, v10 // (op 427) 

buffer_load_dwordx4 v[20:23], v0, s[64:67], 0 offen // (op 427) Load value
buffer_load_dwordx4 v[24:27], v0, s[64:67], 0 offen offset:64 // (op 427) Load value

v_add_u32 v10, 128, v10 // (op 981) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1733) 



ds_read_u8 v0, v3 // (op 420) Load local data 


v_mov_b32 v9, v13 // (op 422) 

buffer_load_ubyte v11, v9, s[72:75], 0 offen offset: 0 // (op 422) Load value

v_add_u32 v13, 4, v13 // (op 1237) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:19], v[20:27], a[0:3], v0, v11 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 423) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 412) 
s_mov_b32 s6, s18 // (op 412) convert

s_cmp_lt_u32 s15, s6 // (op 412) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_412 // (op 412) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_412 if true)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_412:






s_barrier  // (op 1837) 
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 193) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_193 // (op 193) Condition: False, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_193
s_nop 0
v_accvgpr_read v8, a0 // (op 201) Wait state hazard: XDL Write Hazard
v_accvgpr_read v9, a1 // (op 201) 
v_accvgpr_read v10, a2 // (op 201) 
v_accvgpr_read v11, a3 // (op 201) 
v_mul_f32 v12, v7, v8 // (op 201) 
v_mul_f32 v13, v7, v9 // (op 201) 
v_mul_f32 v14, v7, v10 // (op 201) 
v_mul_f32 v15, v7, v11 // (op 201) 

v_lshrrev_b32 v8, 16, v12 // (op 203) 
v_lshrrev_b32 v9, 16, v13 // (op 203) 
v_lshrrev_b32 v10, 16, v14 // (op 203) 
v_lshrrev_b32 v11, 16, v15 // (op 203) 

v_lshrrev_b32 v0, 6, v1 // (op 1494) 
s_mul_i32 s15, s16, s50 // (op 1494) 
s_add_u32 s15, s15, s0 // (op 1494) 
v_and_b32 v12, 63, v1 // (op 1494) 
s_add_u32 s6, s15, s9 // (op 1494) 

v_lshrrev_b32 v13, 4, v12 // (op 1494) 
s_mul_hi_u32 s15, s6, s13 // (op 1494) 
v_lshlrev_b32 v14, 2, v13 // (op 1494) 

s_sub_u32 s7, s6, s15 // (op 1494) 
v_lshl_add_u32 v13, v0, 4, v14 // (op 1494) 


s_lshr_b32 s8, s7, 1 // (op 1494) 

v_lshrrev_b32 v0, 2, v12 // (op 1494) 

s_add_u32 s7, s8, s15 // (op 1494) 


v_and_b32 v12, 3, v0 // (op 1494) 

s_lshr_b32 s8, s7, s11 // (op 1494) 

v_and_b32 v0, 3, v1 // (op 1494) 
s_and_b32 s7, s11, 2147483648 // (op 1494) 
v_lshl_add_u32 v14, v12, 2, v0 // (op 1494) 


s_cmp_eq_i32 s7, 0 // (op 1494) 
s_mov_b32 s15, scc // (op 1494) 

s_cmp_eq_i32 s15, 0 // (op 1494) 
s_cselect_b32 s7, s6, s8 // (op 1494) 



s_mul_hi_u32 s6, s7, s12 // (op 1494) 
s_sub_u32 s8, s7, s6 // (op 1494) 
s_lshr_b32 s15, s8, 1 // (op 1494) 

s_add_u32 s8, s15, s6 // (op 1494) 


s_lshr_b32 s6, s8, s10 // (op 1494) 

s_and_b32 s8, s10, 2147483648 // (op 1494) 
s_cmp_eq_i32 s8, 0 // (op 1494) 
s_mov_b32 s15, scc // (op 1494) 

s_cmp_eq_i32 s15, 0 // (op 1494) 
s_cselect_b32 s8, s7, s6 // (op 1494) 


s_mul_i32 s6, s8, s14 // (op 1494) 
v_lshl_add_u32 v0, s8, 5, v13 // (op 1494) 


s_sub_u32 s8, s7, s6 // (op 1494) 


v_lshl_add_u32 v12, s8, 4, v14 // (op 1494) 


v_mul_lo_u32 v13, v12, s46 // (op 1494) 

v_add_lshl_u32 v12, v0, v13, 1 // (op 1494) 


s_mul_i32 s19, s46, s23 // (op 1496) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1496) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1496) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1496) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1496) most significant: sum
s_add_u32 s19, s19, s6 // (op 1496) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 1 // (op 1496) 

s_mov_b32 s76, s48 // (op 1496) reg expression
s_mov_b32 s77, s49 // (op 1496) reg expression
s_mov_b32 s78, s6 // (op 1496) reg expression
s_mov_b32 s79, 131072 // (op 1496) call()


v_pack_b32_f16 v14, v8, v9 // (op 204) 
v_pack_b32_f16 v15, v10, v11 // (op 204) 


buffer_store_dwordx2 v[14:15], v12, s[76:79], 0 offen // (op 204) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_193 // (op 193) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_193
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_193:



buffer_load_dwordx2 v[8:9], v5, s[56:59], 0 offen // (op 218) Load value
v_accvgpr_read v10, a0 // (op 214) 
v_accvgpr_read v11, a1 // (op 214) 
v_accvgpr_read v12, a2 // (op 214) 
v_accvgpr_read v13, a3 // (op 214) 
v_mul_f32 v14, v7, v10 // (op 214) 
v_mul_f32 v15, v7, v11 // (op 214) 
v_mul_f32 v16, v7, v12 // (op 214) 
v_mul_f32 v17, v7, v13 // (op 214) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v8 // (op 219) 
v_lshrrev_b32 v11, 16, v8 // (op 219) 
v_lshlrev_b32 v10, 16, v10 // (op 219) 
v_lshlrev_b32 v11, 16, v11 // (op 219) 
v_mul_f32 v18, v6, v10 // (op 219) 
v_mul_f32 v19, v6, v11 // (op 219) 
v_and_b32 v10, 65535, v9 // (op 219) 
v_lshrrev_b32 v11, 16, v9 // (op 219) 
v_lshlrev_b32 v10, 16, v10 // (op 219) 
v_lshlrev_b32 v11, 16, v11 // (op 219) 
v_mul_f32 v20, v6, v10 // (op 219) 
v_mul_f32 v21, v6, v11 // (op 219) 


v_add_f32 v8, v18, v14 // (op 215) 
v_add_f32 v9, v19, v15 // (op 215) 
v_add_f32 v10, v20, v16 // (op 215) 
v_add_f32 v11, v21, v17 // (op 215) 


v_lshrrev_b32 v12, 16, v8 // (op 216) 
v_lshrrev_b32 v13, 16, v9 // (op 216) 
v_lshrrev_b32 v14, 16, v10 // (op 216) 
v_lshrrev_b32 v15, 16, v11 // (op 216) 

v_lshrrev_b32 v0, 6, v1 // (op 1511) 
s_mul_i32 s15, s16, s50 // (op 1511) 
s_add_u32 s15, s15, s0 // (op 1511) 
v_and_b32 v8, 63, v1 // (op 1511) 
s_add_u32 s6, s15, s9 // (op 1511) 

v_lshrrev_b32 v9, 4, v8 // (op 1511) 
s_mul_hi_u32 s15, s6, s13 // (op 1511) 
v_lshlrev_b32 v10, 2, v9 // (op 1511) 

s_sub_u32 s7, s6, s15 // (op 1511) 
v_lshl_add_u32 v9, v0, 4, v10 // (op 1511) 


s_lshr_b32 s8, s7, 1 // (op 1511) 

v_lshrrev_b32 v0, 2, v8 // (op 1511) 

s_add_u32 s7, s8, s15 // (op 1511) 


v_and_b32 v8, 3, v0 // (op 1511) 

s_lshr_b32 s8, s7, s11 // (op 1511) 

v_and_b32 v0, 3, v1 // (op 1511) 
s_and_b32 s7, s11, 2147483648 // (op 1511) 
v_lshl_add_u32 v10, v8, 2, v0 // (op 1511) 


s_cmp_eq_i32 s7, 0 // (op 1511) 
s_mov_b32 s15, scc // (op 1511) 

s_cmp_eq_i32 s15, 0 // (op 1511) 
s_cselect_b32 s7, s6, s8 // (op 1511) 



s_mul_hi_u32 s6, s7, s12 // (op 1511) 
s_sub_u32 s8, s7, s6 // (op 1511) 
s_lshr_b32 s15, s8, 1 // (op 1511) 

s_add_u32 s8, s15, s6 // (op 1511) 


s_lshr_b32 s6, s8, s10 // (op 1511) 

s_and_b32 s8, s10, 2147483648 // (op 1511) 
s_cmp_eq_i32 s8, 0 // (op 1511) 
s_mov_b32 s15, scc // (op 1511) 

s_cmp_eq_i32 s15, 0 // (op 1511) 
s_cselect_b32 s8, s7, s6 // (op 1511) 


s_mul_i32 s6, s8, s14 // (op 1511) 
v_lshl_add_u32 v0, s8, 5, v9 // (op 1511) 


s_sub_u32 s8, s7, s6 // (op 1511) 


v_lshl_add_u32 v11, s8, 4, v10 // (op 1511) 


v_mul_lo_u32 v8, v11, s46 // (op 1511) 

v_add_lshl_u32 v9, v0, v8, 1 // (op 1511) 


s_mul_i32 s19, s46, s23 // (op 1513) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1513) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1513) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1513) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1513) most significant: sum
s_add_u32 s19, s19, s6 // (op 1513) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 1 // (op 1513) 

s_mov_b32 s76, s48 // (op 1513) reg expression
s_mov_b32 s77, s49 // (op 1513) reg expression
s_mov_b32 s78, s6 // (op 1513) reg expression
s_mov_b32 s79, 131072 // (op 1513) call()


v_pack_b32_f16 v10, v12, v13 // (op 217) 
v_pack_b32_f16 v11, v14, v15 // (op 217) 


buffer_store_dwordx2 v[10:11], v9, s[76:79], 0 offen // (op 217) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_193:

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
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_DPStreamTileLoop_404 // (op 404) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_DPStreamTileLoop_404 if true)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_404:




































s_endpgm  // End of RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2
.LRR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_end:
.size RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2, .LRR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_end-RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2
  .amdhsa_next_free_vgpr 52
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 256 // lds bytes
  .amdhsa_accum_offset 48
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
    DataTypeB: F8
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
            { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: address, semantic: RR_SKAddressPartials },
            { type: address, semantic: RR_SKAddressFlags },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: uint32, semantic: RR_SKNumDPTilesPerWG },
            { type: uint32, semantic: RR_SKNumDPTiles } ]
    macrotile: [32, 16, 128]
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
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2
    .symbol: RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 256
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 82
    .vgpr_count: 48
    .agpr_count: 4
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
      - .name: Tensor_0_stride_0_8
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_9
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_2_stride_1_11
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_14
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_16
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: SCRATCH_None
        .size: 8
        .offset: 144
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: SCRATCH_ZeroedBeforeAndAfter
        .size: 8
        .offset: 152
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_stride_1_24
        .size: 8
        .offset: 160
        .value_kind: by_value
      - .name: Tensor_21_pointer
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

