// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2,@function
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 1434) 
v_and_b32 v3, 1, v0 // (op 1434) 

v_and_b32 v0, 15, v1 // (op 1434) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 1434) 


v_and_b32 v0, 63, v1 // (op 1434) 
v_lshrrev_b32 v3, 4, v0 // (op 1434) 

v_lshl_add_u32 v0, v4, 2, v3 // (op 1434) 


v_lshrrev_b32 v3, 6, v1 // (op 1557) 
v_and_b32 v4, 1, v3 // (op 1557) 

v_and_b32 v3, 15, v1 // (op 1557) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 1557) 


v_and_b32 v3, 63, v1 // (op 1557) 
v_lshrrev_b32 v4, 4, v3 // (op 1557) 

v_lshl_add_u32 v3, v5, 2, v4 // (op 1557) 


v_lshrrev_b32 v4, 6, v1 // (op 1615) 
v_and_b32 v5, 1, v4 // (op 1615) 

v_and_b32 v4, 15, v1 // (op 1615) 
v_lshl_add_u32 v6, v5, 4, v4 // (op 1615) 


v_and_b32 v4, 63, v1 // (op 1615) 
v_lshrrev_b32 v5, 4, v4 // (op 1615) 

v_lshl_add_u32 v4, v6, 2, v5 // (op 1615) 


v_lshrrev_b32 v5, 6, v1 // (op 1639) 
v_and_b32 v6, 1, v5 // (op 1639) 

v_and_b32 v5, 15, v1 // (op 1639) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 1639) 


v_and_b32 v5, 63, v1 // (op 1639) 
v_lshrrev_b32 v6, 4, v5 // (op 1639) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 1639) 


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
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_241 // (op 241) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_241 if false)
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_SKStreamTileLoop_241:
v_lshrrev_b32 v8, 7, v1 // (op 1359) 
s_mul_i32 s52, s16, s8 // (op 1359) 
s_add_u32 s52, s52, s0 // (op 1359) 
v_and_b32 v9, 63, v1 // (op 1359) 
s_mul_hi_u32 s53, s52, s13 // (op 1359) 
v_lshrrev_b32 v10, 4, v9 // (op 1359) 
s_sub_u32 s54, s52, s53 // (op 1359) 
v_lshlrev_b32 v11, 2, v10 // (op 1359) 

v_lshl_add_u32 v10, v8, 4, v11 // (op 1359) 


s_lshr_b32 s55, s54, 1 // (op 1359) 

s_add_u32 s54, s55, s53 // (op 1359) 


v_lshrrev_b32 v8, 2, v9 // (op 1359) 

s_lshr_b32 s53, s54, s11 // (op 1359) 

v_and_b32 v9, 3, v8 // (op 1359) 

s_and_b32 s54, s11, 2147483648 // (op 1359) 
v_and_b32 v8, 3, v1 // (op 1359) 
s_cmp_eq_i32 s54, 0 // (op 1359) 
s_mov_b32 s55, scc // (op 1359) 

v_lshl_add_u32 v11, v9, 2, v8 // (op 1359) 


s_cmp_eq_i32 s55, 0 // (op 1359) 
s_cselect_b32 s54, s52, s53 // (op 1359) 



v_lshrrev_b32 v8, 6, v1 // (op 1359) 
s_mul_hi_u32 s52, s54, s12 // (op 1359) 
v_and_b32 v9, 1, v8 // (op 1359) 

s_sub_u32 s53, s54, s52 // (op 1359) 
v_lshl_add_u32 v8, v9, 4, v11 // (op 1359) 


s_lshr_b32 s55, s53, 1 // (op 1359) 

s_add_u32 s53, s55, s52 // (op 1359) 


s_lshr_b32 s52, s53, s10 // (op 1359) 

s_and_b32 s53, s10, 2147483648 // (op 1359) 
s_cmp_eq_i32 s53, 0 // (op 1359) 
s_mov_b32 s55, scc // (op 1359) 

s_cmp_eq_i32 s55, 0 // (op 1359) 
s_cselect_b32 s53, s54, s52 // (op 1359) 


s_mul_i32 s52, s53, s14 // (op 1359) 
v_lshl_add_u32 v9, s53, 4, v10 // (op 1359) 


s_sub_u32 s53, s54, s52 // (op 1359) 


v_lshl_add_u32 v10, s53, 5, v8 // (op 1359) 


v_mul_lo_u32 v8, v10, s4 // (op 1359) 

v_add_lshl_u32 v10, v9, v8, 2 // (op 1359) 


s_mul_i32 s53, s4, s23 // (op 1361) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 1361) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 1361) least significant
s_mul_hi_u32 s55, s4, s22 // (op 1361) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1361) most significant: sum
s_add_u32 s53, s53, s55 // (op 1361) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1361) 

s_mov_b32 s56, s20 // (op 1361) reg expression
s_mov_b32 s57, s21 // (op 1361) reg expression
s_mov_b32 s58, s54 // (op 1361) reg expression
s_mov_b32 s59, 131072 // (op 1361) call()

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
v_lshrrev_b32 v8, 7, v1 // (op 891) 
s_mul_i32 s52, s16, s8 // (op 891) 
s_add_u32 s52, s52, s0 // (op 891) 
v_and_b32 v9, 15, v1 // (op 891) 
s_mul_hi_u32 s53, s52, s13 // (op 891) 
v_lshl_add_u32 v11, v8, 4, v9 // (op 891) 


s_sub_u32 s54, s52, s53 // (op 891) 
v_and_b32 v8, 63, v1 // (op 891) 
s_lshr_b32 s55, s54, 1 // (op 891) 

v_lshrrev_b32 v9, 4, v8 // (op 891) 

s_add_u32 s54, s55, s53 // (op 891) 


v_lshlrev_b32 v8, 5, v9 // (op 891) 

s_lshr_b32 s53, s54, s11 // (op 891) 

s_and_b32 s54, s11, 2147483648 // (op 891) 
s_cmp_eq_i32 s54, 0 // (op 891) 
s_mov_b32 s55, scc // (op 891) 

s_cmp_eq_i32 s55, 0 // (op 891) 
s_cselect_b32 s54, s52, s53 // (op 891) 


s_mul_hi_u32 s53, s54, s12 // (op 891) 
s_sub_u32 s55, s54, s53 // (op 891) 
s_lshr_b32 s60, s55, 1 // (op 891) 

s_add_u32 s55, s60, s53 // (op 891) 


s_lshr_b32 s53, s55, s10 // (op 891) 

s_and_b32 s55, s10, 2147483648 // (op 891) 
s_cmp_eq_i32 s55, 0 // (op 891) 
s_mov_b32 s60, scc // (op 891) 

s_cmp_eq_i32 s60, 0 // (op 891) 
s_cselect_b32 s55, s54, s53 // (op 891) 


v_lshl_add_u32 v9, s55, 4, v11 // (op 891) 


s_ashr_i64 s[60:61], s[2:3], 7 // (op 891) 
s_mul_i32 s53, s54, s60 // (op 891) 


s_sub_u32 s54, s52, s53 // (op 891) 


v_lshl_add_u32 v11, s54, 7, v8 // (op 891) 


v_mov_b32 v13, 0 // (op 891) convert
v_mov_b32 v12, v11 // (op 891) convert

v_mul_lo_u32 v15, v9, s25 // (op 891) most significant: low of low * high
v_mul_hi_u32 v8, v9, s24 // (op 891) most significant: high of low * low
v_mul_lo_u32 v14, v9, s24 // (op 891) least significant: low of low * low
v_add_u32 v15, v15, v8 // (op 891) most significant: sum

v_add_co_u32 v14, s[52:53], v14, v12 // (op 891) least significant half
v_addc_co_u32 v15, s[52:53], v15, v13, s[52:53] // (op 891) most significant half



v_lshlrev_b64 v[8:9], 2, v[14:15] // (op 891) 

v_lshrrev_b64 v[12:13], 3, v[8:9] // (op 891) 

v_mov_b32 v11, v12 // (op 891) convert

s_mul_i32 s53, s24, s29 // (op 893) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 893) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 893) least significant
s_mul_hi_u32 s55, s24, s28 // (op 893) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 893) most significant: sum
s_add_u32 s53, s53, s55 // (op 893) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 893) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 893) 

s_mov_b32 s60, s26 // (op 893) reg expression
s_mov_b32 s61, s27 // (op 893) reg expression
s_mov_b32 s62, s52 // (op 893) reg expression
s_mov_b32 s63, 131072 // (op 893) call()

v_lshrrev_b32 v8, 7, v1 // (op 910) 
s_mul_i32 s52, s16, s8 // (op 910) 
s_add_u32 s52, s52, s0 // (op 910) 
v_and_b32 v9, 15, v1 // (op 910) 
s_mul_hi_u32 s53, s52, s13 // (op 910) 
v_lshl_add_u32 v12, v8, 4, v9 // (op 910) 


s_sub_u32 s54, s52, s53 // (op 910) 
v_and_b32 v8, 63, v1 // (op 910) 
s_lshr_b32 s55, s54, 1 // (op 910) 

v_lshrrev_b32 v9, 4, v8 // (op 910) 

s_add_u32 s54, s55, s53 // (op 910) 


v_lshlrev_b32 v8, 5, v9 // (op 910) 

s_lshr_b32 s53, s54, s11 // (op 910) 

s_and_b32 s54, s11, 2147483648 // (op 910) 
s_cmp_eq_i32 s54, 0 // (op 910) 
s_mov_b32 s55, scc // (op 910) 

s_cmp_eq_i32 s55, 0 // (op 910) 
s_cselect_b32 s54, s52, s53 // (op 910) 


s_mul_hi_u32 s53, s54, s12 // (op 910) 
s_sub_u32 s55, s54, s53 // (op 910) 
s_lshr_b32 s64, s55, 1 // (op 910) 

s_add_u32 s55, s64, s53 // (op 910) 


s_lshr_b32 s53, s55, s10 // (op 910) 

s_and_b32 s55, s10, 2147483648 // (op 910) 
s_cmp_eq_i32 s55, 0 // (op 910) 
s_mov_b32 s64, scc // (op 910) 

s_cmp_eq_i32 s64, 0 // (op 910) 
s_cselect_b32 s55, s54, s53 // (op 910) 


v_lshl_add_u32 v9, s55, 4, v12 // (op 910) 


s_ashr_i64 s[64:65], s[2:3], 7 // (op 910) 
s_mul_i32 s53, s54, s64 // (op 910) 


s_sub_u32 s54, s52, s53 // (op 910) 


v_lshl_add_u32 v12, s54, 7, v8 // (op 910) 


v_mov_b32 v15, 0 // (op 910) convert
v_mov_b32 v14, v12 // (op 910) convert

v_mul_lo_u32 v13, v9, s25 // (op 910) most significant: low of low * high
v_mul_hi_u32 v8, v9, s24 // (op 910) most significant: high of low * low
v_mul_lo_u32 v12, v9, s24 // (op 910) least significant: low of low * low
v_add_u32 v13, v13, v8 // (op 910) most significant: sum

v_add_co_u32 v12, s[52:53], v12, v14 // (op 910) least significant half
v_addc_co_u32 v13, s[52:53], v13, v15, s[52:53] // (op 910) most significant half



v_lshlrev_b64 v[8:9], 2, v[12:13] // (op 910) 

v_lshrrev_b64 v[12:13], 3, v[8:9] // (op 910) 

v_mov_b32 v8, v12 // (op 910) convert

s_mul_i32 s53, s24, s29 // (op 912) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 912) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 912) least significant
s_mul_hi_u32 s55, s24, s28 // (op 912) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 912) most significant: sum
s_add_u32 s53, s53, s55 // (op 912) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 912) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 912) 

s_mov_b32 s60, s26 // (op 912) reg expression
s_mov_b32 s61, s27 // (op 912) reg expression
s_mov_b32 s62, s52 // (op 912) reg expression
s_mov_b32 s63, 131072 // (op 912) call()

v_lshrrev_b32 v9, 6, v1 // (op 1022) 
s_mul_i32 s52, s16, s8 // (op 1022) 
s_add_u32 s52, s52, s0 // (op 1022) 
v_and_b32 v12, 1, v9 // (op 1022) 

s_mul_hi_u32 s53, s52, s13 // (op 1022) 
v_and_b32 v9, 15, v1 // (op 1022) 
s_sub_u32 s54, s52, s53 // (op 1022) 
v_lshl_add_u32 v13, v12, 4, v9 // (op 1022) 


s_lshr_b32 s55, s54, 1 // (op 1022) 

v_and_b32 v9, 48, v1 // (op 1022) 
s_add_u32 s54, s55, s53 // (op 1022) 


s_lshr_b32 s53, s54, s11 // (op 1022) 

s_and_b32 s54, s11, 2147483648 // (op 1022) 
s_cmp_eq_i32 s54, 0 // (op 1022) 
s_mov_b32 s55, scc // (op 1022) 

s_cmp_eq_i32 s55, 0 // (op 1022) 
s_cselect_b32 s54, s52, s53 // (op 1022) 


s_mul_hi_u32 s53, s54, s12 // (op 1022) 
s_sub_u32 s55, s54, s53 // (op 1022) 
s_lshr_b32 s64, s55, 1 // (op 1022) 

s_add_u32 s55, s64, s53 // (op 1022) 


s_lshr_b32 s53, s55, s10 // (op 1022) 

s_and_b32 s55, s10, 2147483648 // (op 1022) 
s_cmp_eq_i32 s55, 0 // (op 1022) 
s_mov_b32 s64, scc // (op 1022) 

s_cmp_eq_i32 s64, 0 // (op 1022) 
s_cselect_b32 s55, s54, s53 // (op 1022) 


s_mul_i32 s53, s55, s14 // (op 1022) 

s_sub_u32 s55, s54, s53 // (op 1022) 

v_lshl_add_u32 v12, s55, 5, v13 // (op 1022) 


v_mul_lo_u32 v13, v12, s30 // (op 1022) 

s_ashr_i64 s[64:65], s[2:3], 7 // (op 1022) 
s_mul_i32 s53, s54, s64 // (op 1022) 


s_sub_u32 s54, s52, s53 // (op 1022) 


v_lshl_add_u32 v12, s54, 7, v9 // (op 1022) 


v_add_u32 v9, v12, v13 // (op 1022) 


s_mul_i32 s53, s30, s23 // (op 1024) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 1024) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 1024) least significant
s_mul_hi_u32 s55, s30, s22 // (op 1024) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1024) most significant: sum
s_add_u32 s53, s53, s55 // (op 1024) most significant: sum


s_mov_b32 s64, s32 // (op 1024) reg expression
s_mov_b32 s65, s33 // (op 1024) reg expression
s_mov_b32 s66, s52 // (op 1024) reg expression
s_mov_b32 s67, 131072 // (op 1024) call()

v_lshrrev_b32 v12, 6, v1 // (op 1041) 
s_mul_i32 s52, s16, s8 // (op 1041) 
s_add_u32 s52, s52, s0 // (op 1041) 
v_and_b32 v13, 1, v12 // (op 1041) 

s_mul_hi_u32 s53, s52, s13 // (op 1041) 
v_and_b32 v12, 15, v1 // (op 1041) 
s_sub_u32 s54, s52, s53 // (op 1041) 
v_lshl_add_u32 v14, v13, 4, v12 // (op 1041) 


s_lshr_b32 s55, s54, 1 // (op 1041) 

v_and_b32 v12, 48, v1 // (op 1041) 
s_add_u32 s54, s55, s53 // (op 1041) 


s_lshr_b32 s53, s54, s11 // (op 1041) 

s_and_b32 s54, s11, 2147483648 // (op 1041) 
s_cmp_eq_i32 s54, 0 // (op 1041) 
s_mov_b32 s55, scc // (op 1041) 

s_cmp_eq_i32 s55, 0 // (op 1041) 
s_cselect_b32 s54, s52, s53 // (op 1041) 


s_mul_hi_u32 s53, s54, s12 // (op 1041) 
s_sub_u32 s55, s54, s53 // (op 1041) 
s_lshr_b32 s68, s55, 1 // (op 1041) 

s_add_u32 s55, s68, s53 // (op 1041) 


s_lshr_b32 s53, s55, s10 // (op 1041) 

s_and_b32 s55, s10, 2147483648 // (op 1041) 
s_cmp_eq_i32 s55, 0 // (op 1041) 
s_mov_b32 s68, scc // (op 1041) 

s_cmp_eq_i32 s68, 0 // (op 1041) 
s_cselect_b32 s55, s54, s53 // (op 1041) 


s_mul_i32 s53, s55, s14 // (op 1041) 

s_sub_u32 s55, s54, s53 // (op 1041) 

v_lshl_add_u32 v13, s55, 5, v14 // (op 1041) 


v_mul_lo_u32 v14, v13, s30 // (op 1041) 

s_ashr_i64 s[68:69], s[2:3], 7 // (op 1041) 
s_mul_i32 s53, s54, s68 // (op 1041) 


s_sub_u32 s54, s52, s53 // (op 1041) 


v_lshl_add_u32 v13, s54, 7, v12 // (op 1041) 


v_add_u32 v12, v13, v14 // (op 1041) 


s_mul_i32 s53, s30, s23 // (op 1043) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 1043) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 1043) least significant
s_mul_hi_u32 s55, s30, s22 // (op 1043) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1043) most significant: sum
s_add_u32 s53, s53, s55 // (op 1043) most significant: sum


s_mov_b32 s64, s32 // (op 1043) reg expression
s_mov_b32 s65, s33 // (op 1043) reg expression
s_mov_b32 s66, s52 // (op 1043) reg expression
s_mov_b32 s67, 131072 // (op 1043) call()

v_lshrrev_b32 v13, 7, v1 // (op 1150) 
s_mul_i32 s52, s16, s8 // (op 1150) 
s_add_u32 s52, s52, s0 // (op 1150) 
v_and_b32 v14, 15, v1 // (op 1150) 
s_mul_hi_u32 s53, s52, s13 // (op 1150) 
v_lshl_add_u32 v15, v13, 4, v14 // (op 1150) 


s_sub_u32 s54, s52, s53 // (op 1150) 
v_and_b32 v13, 63, v1 // (op 1150) 
s_lshr_b32 s55, s54, 1 // (op 1150) 

v_lshrrev_b32 v14, 4, v13 // (op 1150) 

s_add_u32 s54, s55, s53 // (op 1150) 


s_lshr_b32 s53, s54, s11 // (op 1150) 

s_and_b32 s54, s11, 2147483648 // (op 1150) 
s_cmp_eq_i32 s54, 0 // (op 1150) 
s_mov_b32 s55, scc // (op 1150) 

s_cmp_eq_i32 s55, 0 // (op 1150) 
s_cselect_b32 s54, s52, s53 // (op 1150) 


s_mul_hi_u32 s53, s54, s12 // (op 1150) 
s_sub_u32 s55, s54, s53 // (op 1150) 
s_lshr_b32 s68, s55, 1 // (op 1150) 

s_add_u32 s55, s68, s53 // (op 1150) 


s_lshr_b32 s53, s55, s10 // (op 1150) 

s_and_b32 s55, s10, 2147483648 // (op 1150) 
s_cmp_eq_i32 s55, 0 // (op 1150) 
s_mov_b32 s68, scc // (op 1150) 

s_cmp_eq_i32 s68, 0 // (op 1150) 
s_cselect_b32 s55, s54, s53 // (op 1150) 


v_lshl_add_u32 v13, s55, 4, v15 // (op 1150) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 1150) 
s_mul_i32 s53, s54, s68 // (op 1150) 


s_sub_u32 s54, s52, s53 // (op 1150) 


v_lshl_add_u32 v15, s54, 2, v14 // (op 1150) 


v_mul_lo_u32 v14, v13, s34 // (op 1150) 
v_add_u32 v14, v14, v15 // (op 1150) 


s_mul_i32 s53, s34, s29 // (op 1152) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 1152) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 1152) least significant
s_mul_hi_u32 s55, s34, s28 // (op 1152) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1152) most significant: sum
s_add_u32 s53, s53, s55 // (op 1152) most significant: sum


s_mov_b32 s68, s36 // (op 1152) reg expression
s_mov_b32 s69, s37 // (op 1152) reg expression
s_mov_b32 s70, s52 // (op 1152) reg expression
s_mov_b32 s71, 131072 // (op 1152) call()

v_lshrrev_b32 v13, 7, v1 // (op 1169) 
s_mul_i32 s52, s16, s8 // (op 1169) 
s_add_u32 s52, s52, s0 // (op 1169) 
v_and_b32 v15, 15, v1 // (op 1169) 
s_mul_hi_u32 s53, s52, s13 // (op 1169) 
v_lshl_add_u32 v16, v13, 4, v15 // (op 1169) 


s_sub_u32 s54, s52, s53 // (op 1169) 
v_and_b32 v13, 63, v1 // (op 1169) 
s_lshr_b32 s55, s54, 1 // (op 1169) 

v_lshrrev_b32 v15, 4, v13 // (op 1169) 

s_add_u32 s54, s55, s53 // (op 1169) 


s_lshr_b32 s53, s54, s11 // (op 1169) 

s_and_b32 s54, s11, 2147483648 // (op 1169) 
s_cmp_eq_i32 s54, 0 // (op 1169) 
s_mov_b32 s55, scc // (op 1169) 

s_cmp_eq_i32 s55, 0 // (op 1169) 
s_cselect_b32 s54, s52, s53 // (op 1169) 


s_mul_hi_u32 s53, s54, s12 // (op 1169) 
s_sub_u32 s55, s54, s53 // (op 1169) 
s_lshr_b32 s72, s55, 1 // (op 1169) 

s_add_u32 s55, s72, s53 // (op 1169) 


s_lshr_b32 s53, s55, s10 // (op 1169) 

s_and_b32 s55, s10, 2147483648 // (op 1169) 
s_cmp_eq_i32 s55, 0 // (op 1169) 
s_mov_b32 s72, scc // (op 1169) 

s_cmp_eq_i32 s72, 0 // (op 1169) 
s_cselect_b32 s55, s54, s53 // (op 1169) 


v_lshl_add_u32 v13, s55, 4, v16 // (op 1169) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 1169) 
s_mul_i32 s53, s54, s72 // (op 1169) 


s_sub_u32 s54, s52, s53 // (op 1169) 


v_lshl_add_u32 v16, s54, 2, v15 // (op 1169) 


v_mul_lo_u32 v15, v13, s34 // (op 1169) 
v_add_u32 v15, v15, v16 // (op 1169) 


s_mul_i32 s53, s34, s29 // (op 1171) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 1171) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 1171) least significant
s_mul_hi_u32 s55, s34, s28 // (op 1171) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1171) most significant: sum
s_add_u32 s53, s53, s55 // (op 1171) most significant: sum


s_mov_b32 s68, s36 // (op 1171) reg expression
s_mov_b32 s69, s37 // (op 1171) reg expression
s_mov_b32 s70, s52 // (op 1171) reg expression
s_mov_b32 s71, 131072 // (op 1171) call()

v_lshrrev_b32 v13, 2, v1 // (op 1278) 
s_mul_i32 s52, s16, s8 // (op 1278) 
s_add_u32 s52, s52, s0 // (op 1278) 
v_and_b32 v16, -4, v1 // (op 1278) 
s_mul_hi_u32 s53, s52, s13 // (op 1278) 
v_sub_u32 v17, v1, v16 // (op 1278) 

s_sub_u32 s54, s52, s53 // (op 1278) 
s_lshr_b32 s55, s54, 1 // (op 1278) 

s_add_u32 s54, s55, s53 // (op 1278) 


s_lshr_b32 s53, s54, s11 // (op 1278) 

s_and_b32 s54, s11, 2147483648 // (op 1278) 
s_cmp_eq_i32 s54, 0 // (op 1278) 
s_mov_b32 s55, scc // (op 1278) 

s_cmp_eq_i32 s55, 0 // (op 1278) 
s_cselect_b32 s54, s52, s53 // (op 1278) 


s_mul_hi_u32 s53, s54, s12 // (op 1278) 
s_sub_u32 s55, s54, s53 // (op 1278) 
s_lshr_b32 s72, s55, 1 // (op 1278) 

s_add_u32 s55, s72, s53 // (op 1278) 


s_lshr_b32 s53, s55, s10 // (op 1278) 

s_and_b32 s55, s10, 2147483648 // (op 1278) 
s_cmp_eq_i32 s55, 0 // (op 1278) 
s_mov_b32 s72, scc // (op 1278) 

s_cmp_eq_i32 s72, 0 // (op 1278) 
s_cselect_b32 s55, s54, s53 // (op 1278) 


s_mul_i32 s53, s55, s14 // (op 1278) 

s_sub_u32 s55, s54, s53 // (op 1278) 

v_lshl_add_u32 v16, s55, 5, v13 // (op 1278) 


v_mul_lo_u32 v13, v16, s38 // (op 1278) 

s_ashr_i64 s[72:73], s[2:3], 7 // (op 1278) 
s_mul_i32 s53, s54, s72 // (op 1278) 


s_sub_u32 s54, s52, s53 // (op 1278) 


v_lshl_add_u32 v16, s54, 2, v17 // (op 1278) 


v_add_u32 v17, v16, v13 // (op 1278) 


s_mul_i32 s53, s38, s23 // (op 1280) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 1280) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 1280) least significant
s_mul_hi_u32 s55, s38, s22 // (op 1280) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1280) most significant: sum
s_add_u32 s53, s53, s55 // (op 1280) most significant: sum


s_mov_b32 s72, s40 // (op 1280) reg expression
s_mov_b32 s73, s41 // (op 1280) reg expression
s_mov_b32 s74, s52 // (op 1280) reg expression
s_mov_b32 s75, 131072 // (op 1280) call()

v_lshrrev_b32 v13, 2, v1 // (op 1297) 
s_mul_i32 s52, s16, s8 // (op 1297) 
s_add_u32 s52, s52, s0 // (op 1297) 
v_and_b32 v16, -4, v1 // (op 1297) 
s_mul_hi_u32 s53, s52, s13 // (op 1297) 
v_sub_u32 v18, v1, v16 // (op 1297) 

s_sub_u32 s54, s52, s53 // (op 1297) 
s_lshr_b32 s55, s54, 1 // (op 1297) 

s_add_u32 s54, s55, s53 // (op 1297) 


s_lshr_b32 s53, s54, s11 // (op 1297) 

s_and_b32 s54, s11, 2147483648 // (op 1297) 
s_cmp_eq_i32 s54, 0 // (op 1297) 
s_mov_b32 s55, scc // (op 1297) 

s_cmp_eq_i32 s55, 0 // (op 1297) 
s_cselect_b32 s54, s52, s53 // (op 1297) 


s_mul_hi_u32 s53, s54, s12 // (op 1297) 
s_sub_u32 s55, s54, s53 // (op 1297) 
s_lshr_b32 s76, s55, 1 // (op 1297) 

s_add_u32 s55, s76, s53 // (op 1297) 


s_lshr_b32 s53, s55, s10 // (op 1297) 

s_and_b32 s55, s10, 2147483648 // (op 1297) 
s_cmp_eq_i32 s55, 0 // (op 1297) 
s_mov_b32 s76, scc // (op 1297) 

s_cmp_eq_i32 s76, 0 // (op 1297) 
s_cselect_b32 s55, s54, s53 // (op 1297) 


s_mul_i32 s53, s55, s14 // (op 1297) 

s_sub_u32 s55, s54, s53 // (op 1297) 

v_lshl_add_u32 v16, s55, 5, v13 // (op 1297) 


v_mul_lo_u32 v13, v16, s38 // (op 1297) 

s_ashr_i64 s[76:77], s[2:3], 7 // (op 1297) 
s_mul_i32 s53, s54, s76 // (op 1297) 


s_sub_u32 s54, s52, s53 // (op 1297) 


v_lshl_add_u32 v16, s54, 2, v18 // (op 1297) 


v_add_u32 v18, v16, v13 // (op 1297) 


s_mul_i32 s53, s38, s23 // (op 1299) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 1299) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 1299) least significant
s_mul_hi_u32 s55, s38, s22 // (op 1299) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1299) most significant: sum
s_add_u32 s53, s53, s55 // (op 1299) most significant: sum


s_mov_b32 s72, s40 // (op 1299) reg expression
s_mov_b32 s73, s41 // (op 1299) reg expression
s_mov_b32 s74, s52 // (op 1299) reg expression
s_mov_b32 s75, 131072 // (op 1299) call()




buffer_load_dwordx4 v[20:23], v11, s[60:63], 0 offen // (op 685) Load value



buffer_load_dwordx4 v[24:27], v9, s[64:67], 0 offen // (op 690) Load value
buffer_load_dwordx4 v[28:31], v9, s[64:67], 0 offen offset:64 // (op 690) Load value



buffer_load_ubyte v13, v14, s[68:71], 0 offen offset: 0 // (op 695) Load value



buffer_load_ubyte v16, v17, s[72:75], 0 offen offset: 0 // (op 699) Load value



buffer_load_dwordx4 v[32:35], v11, s[60:63], 0 offen offset:64 // (op 703) Load value




buffer_load_dwordx4 v[36:39], v9, s[64:67], 0 offen offset:128 // (op 708) Load value
buffer_load_dwordx4 v[40:43], v9, s[64:67], 0 offen offset:192 // (op 708) Load value




buffer_load_ubyte v9, v14, s[68:71], 0 offen offset:4 // (op 713) Load value




buffer_load_ubyte v11, v17, s[72:75], 0 offen offset:4 // (op 717) Load value

v_and_b32 v14, -4, v1 // (op 1422) 
v_sub_u32 v17, v1, v14 // (op 1422) 

v_and_b32 v14, 4294967292, v1 // (op 1422) 
v_add_u32 v19, v14, v17 // (op 1422) 


s_barrier  // (op 1696) 



s_waitcnt vmcnt(5)
ds_write_b8 v19, v16 // (op 719) Store local data 

v_and_b32 v14, -4, v1 // (op 1393) 
v_sub_u32 v17, v1, v14 // (op 1393) 

v_and_b32 v14, 4294967292, v1 // (op 1393) 
v_add_u32 v19, v14, v17 // (op 1393) 


v_and_b32 v14, -4, v1 // (op 1627) 
v_sub_u32 v17, v1, v14 // (op 1627) 

v_and_b32 v14, 4294967292, v1 // (op 1627) 
v_add_u32 v44, v14, v17 // (op 1627) 


s_mov_b32 s1, 0 // (op 47) call()
s_and_b32 s52, s7, 4294967294 // (op 46) 
s_cmp_lt_u32 s1, s52 // (op 46) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_46 // (op 46) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_46 if false)
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoop_46:
s_add_u32 s1, s1, 2 // (op 48) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1693) 



ds_read_u8 v14, v0 // (op 67) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:31], a[0:3], v13, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 70) 



v_mov_b32 v14, v8 // (op 55) 

buffer_load_dwordx4 v[20:23], v14, s[60:63], 0 offen offset:128 // (op 55) Load value


v_mov_b32 v17, v12 // (op 59) 

buffer_load_dwordx4 v[24:27], v17, s[64:67], 0 offen offset:256 // (op 59) Load value
buffer_load_dwordx4 v[28:31], v17, s[64:67], 0 offen offset:320 // (op 59) Load value


v_mov_b32 v45, v15 // (op 63) 

buffer_load_ubyte v13, v45, s[68:71], 0 offen offset:8 // (op 63) Load value


v_mov_b32 v46, v18 // (op 13) 

buffer_load_ubyte v16, v46, s[72:75], 0 offen offset:8 // (op 13) Load value
s_barrier  // (op 1734) 



s_waitcnt vmcnt(5)
ds_write_b8 v44, v11 offset:128 // (op 557) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1731) 



ds_read_u8 v47, v5 offset:128 // (op 559) Load local data 
s_barrier  // (op 1859) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:35], v[36:43], a[0:3], v9, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 560) 




buffer_load_dwordx4 v[32:35], v14, s[60:63], 0 offen offset:192 // (op 562) Load value




v_add_u32 v8, 128, v8 // (op 902) 
buffer_load_dwordx4 v[36:39], v17, s[64:67], 0 offen offset:384 // (op 564) Load value
buffer_load_dwordx4 v[40:43], v17, s[64:67], 0 offen offset:448 // (op 564) Load value




v_add_u32 v12, 256, v12 // (op 1033) 
buffer_load_ubyte v9, v45, s[68:71], 0 offen offset:12 // (op 566) Load value




v_add_u32 v15, 8, v15 // (op 1161) 
buffer_load_ubyte v11, v46, s[72:75], 0 offen offset:12 // (op 556) Load value

s_barrier  // (op 1699) 
v_add_u32 v18, 8, v18 // (op 1289) 



s_waitcnt vmcnt(5)
ds_write_b8 v19, v16 // (op 18) Store local data 
s_and_b32 s52, s7, 4294967294 // (op 46) 
s_cmp_lt_u32 s1, s52 // (op 46) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoop_46 // (op 46) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoop_46 if true)
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_46:










s_and_b32 s52, s7, 4294967294 // (op 865) 
v_lshrrev_b32 v8, 7, v1 // (op 865) 
s_add_u32 s53, s0, s52 // (op 865) 

v_and_b32 v12, 15, v1 // (op 865) 
v_lshl_add_u32 v13, v8, 4, v12 // (op 865) 


s_mul_i32 s52, s16, s8 // (op 865) 
s_add_u32 s52, s52, s53 // (op 865) 

v_and_b32 v8, 63, v1 // (op 865) 
s_mul_hi_u32 s53, s52, s13 // (op 865) 
v_lshrrev_b32 v12, 4, v8 // (op 865) 

s_sub_u32 s54, s52, s53 // (op 865) 
v_lshlrev_b32 v8, 5, v12 // (op 865) 

s_lshr_b32 s55, s54, 1 // (op 865) 

s_add_u32 s54, s55, s53 // (op 865) 


s_lshr_b32 s53, s54, s11 // (op 865) 

s_and_b32 s54, s11, 2147483648 // (op 865) 
s_cmp_eq_i32 s54, 0 // (op 865) 
s_mov_b32 s55, scc // (op 865) 

s_cmp_eq_i32 s55, 0 // (op 865) 
s_cselect_b32 s54, s52, s53 // (op 865) 


s_mul_hi_u32 s53, s54, s12 // (op 865) 
s_sub_u32 s55, s54, s53 // (op 865) 
s_lshr_b32 s76, s55, 1 // (op 865) 

s_add_u32 s55, s76, s53 // (op 865) 


s_lshr_b32 s53, s55, s10 // (op 865) 

s_and_b32 s55, s10, 2147483648 // (op 865) 
s_cmp_eq_i32 s55, 0 // (op 865) 
s_mov_b32 s76, scc // (op 865) 

s_cmp_eq_i32 s76, 0 // (op 865) 
s_cselect_b32 s55, s54, s53 // (op 865) 


v_lshl_add_u32 v12, s55, 4, v13 // (op 865) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 865) 
s_mul_i32 s53, s54, s76 // (op 865) 


s_sub_u32 s54, s52, s53 // (op 865) 


v_lshl_add_u32 v13, s54, 7, v8 // (op 865) 


v_mov_b32 v15, 0 // (op 865) convert
v_mov_b32 v14, v13 // (op 865) convert

v_mul_lo_u32 v17, v12, s25 // (op 865) most significant: low of low * high
v_mul_hi_u32 v8, v12, s24 // (op 865) most significant: high of low * low
v_mul_lo_u32 v16, v12, s24 // (op 865) least significant: low of low * low
v_add_u32 v17, v17, v8 // (op 865) most significant: sum

v_add_co_u32 v16, s[52:53], v16, v14 // (op 865) least significant half
v_addc_co_u32 v17, s[52:53], v17, v15, s[52:53] // (op 865) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 865) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 865) 

v_mov_b32 v8, v14 // (op 865) convert

s_mul_i32 s53, s24, s29 // (op 867) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 867) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 867) least significant
s_mul_hi_u32 s55, s24, s28 // (op 867) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 867) most significant: sum
s_add_u32 s53, s53, s55 // (op 867) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 867) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 867) 

s_mov_b32 s60, s26 // (op 867) reg expression
s_mov_b32 s61, s27 // (op 867) reg expression
s_mov_b32 s62, s52 // (op 867) reg expression
s_mov_b32 s63, 131072 // (op 867) call()

s_and_b32 s52, s7, 4294967294 // (op 999) 
v_lshrrev_b32 v12, 6, v1 // (op 999) 
s_add_u32 s53, s0, s52 // (op 999) 

v_and_b32 v13, 1, v12 // (op 999) 

v_and_b32 v12, 15, v1 // (op 999) 
s_mul_i32 s52, s16, s8 // (op 999) 
s_add_u32 s52, s52, s53 // (op 999) 

v_lshl_add_u32 v14, v13, 4, v12 // (op 999) 


s_mul_hi_u32 s53, s52, s13 // (op 999) 
v_and_b32 v12, 48, v1 // (op 999) 
s_sub_u32 s54, s52, s53 // (op 999) 
s_lshr_b32 s55, s54, 1 // (op 999) 

s_add_u32 s54, s55, s53 // (op 999) 


s_lshr_b32 s53, s54, s11 // (op 999) 

s_and_b32 s54, s11, 2147483648 // (op 999) 
s_cmp_eq_i32 s54, 0 // (op 999) 
s_mov_b32 s55, scc // (op 999) 

s_cmp_eq_i32 s55, 0 // (op 999) 
s_cselect_b32 s54, s52, s53 // (op 999) 


s_mul_hi_u32 s53, s54, s12 // (op 999) 
s_sub_u32 s55, s54, s53 // (op 999) 
s_lshr_b32 s76, s55, 1 // (op 999) 

s_add_u32 s55, s76, s53 // (op 999) 


s_lshr_b32 s53, s55, s10 // (op 999) 

s_and_b32 s55, s10, 2147483648 // (op 999) 
s_cmp_eq_i32 s55, 0 // (op 999) 
s_mov_b32 s76, scc // (op 999) 

s_cmp_eq_i32 s76, 0 // (op 999) 
s_cselect_b32 s55, s54, s53 // (op 999) 


s_mul_i32 s53, s55, s14 // (op 999) 

s_sub_u32 s55, s54, s53 // (op 999) 

v_lshl_add_u32 v13, s55, 5, v14 // (op 999) 


v_mul_lo_u32 v14, v13, s30 // (op 999) 

s_ashr_i64 s[76:77], s[2:3], 7 // (op 999) 
s_mul_i32 s53, s54, s76 // (op 999) 


s_sub_u32 s54, s52, s53 // (op 999) 


v_lshl_add_u32 v13, s54, 7, v12 // (op 999) 


v_add_u32 v12, v13, v14 // (op 999) 


s_mul_i32 s53, s30, s23 // (op 1001) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 1001) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 1001) least significant
s_mul_hi_u32 s55, s30, s22 // (op 1001) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1001) most significant: sum
s_add_u32 s53, s53, s55 // (op 1001) most significant: sum


s_mov_b32 s64, s32 // (op 1001) reg expression
s_mov_b32 s65, s33 // (op 1001) reg expression
s_mov_b32 s66, s52 // (op 1001) reg expression
s_mov_b32 s67, 131072 // (op 1001) call()

s_and_b32 s52, s7, 4294967294 // (op 1127) 
v_lshrrev_b32 v13, 7, v1 // (op 1127) 
s_add_u32 s53, s0, s52 // (op 1127) 

v_and_b32 v14, 15, v1 // (op 1127) 
v_lshl_add_u32 v15, v13, 4, v14 // (op 1127) 


s_mul_i32 s52, s16, s8 // (op 1127) 
s_add_u32 s52, s52, s53 // (op 1127) 

v_and_b32 v13, 63, v1 // (op 1127) 
s_mul_hi_u32 s53, s52, s13 // (op 1127) 
v_lshrrev_b32 v14, 4, v13 // (op 1127) 

s_sub_u32 s54, s52, s53 // (op 1127) 
s_lshr_b32 s55, s54, 1 // (op 1127) 

s_add_u32 s54, s55, s53 // (op 1127) 


s_lshr_b32 s53, s54, s11 // (op 1127) 

s_and_b32 s54, s11, 2147483648 // (op 1127) 
s_cmp_eq_i32 s54, 0 // (op 1127) 
s_mov_b32 s55, scc // (op 1127) 

s_cmp_eq_i32 s55, 0 // (op 1127) 
s_cselect_b32 s54, s52, s53 // (op 1127) 


s_mul_hi_u32 s53, s54, s12 // (op 1127) 
s_sub_u32 s55, s54, s53 // (op 1127) 
s_lshr_b32 s76, s55, 1 // (op 1127) 

s_add_u32 s55, s76, s53 // (op 1127) 


s_lshr_b32 s53, s55, s10 // (op 1127) 

s_and_b32 s55, s10, 2147483648 // (op 1127) 
s_cmp_eq_i32 s55, 0 // (op 1127) 
s_mov_b32 s76, scc // (op 1127) 

s_cmp_eq_i32 s76, 0 // (op 1127) 
s_cselect_b32 s55, s54, s53 // (op 1127) 


v_lshl_add_u32 v13, s55, 4, v15 // (op 1127) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1127) 
s_mul_i32 s53, s54, s76 // (op 1127) 


s_sub_u32 s54, s52, s53 // (op 1127) 


v_lshl_add_u32 v15, s54, 2, v14 // (op 1127) 


v_mul_lo_u32 v14, v13, s34 // (op 1127) 
v_add_u32 v14, v14, v15 // (op 1127) 


s_mul_i32 s53, s34, s29 // (op 1129) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 1129) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 1129) least significant
s_mul_hi_u32 s55, s34, s28 // (op 1129) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1129) most significant: sum
s_add_u32 s53, s53, s55 // (op 1129) most significant: sum


s_mov_b32 s68, s36 // (op 1129) reg expression
s_mov_b32 s69, s37 // (op 1129) reg expression
s_mov_b32 s70, s52 // (op 1129) reg expression
s_mov_b32 s71, 131072 // (op 1129) call()

s_and_b32 s52, s7, 4294967294 // (op 1255) 
v_lshrrev_b32 v13, 2, v1 // (op 1255) 
s_add_u32 s53, s0, s52 // (op 1255) 

v_and_b32 v15, -4, v1 // (op 1255) 
v_sub_u32 v16, v1, v15 // (op 1255) 

s_mul_i32 s52, s16, s8 // (op 1255) 
s_add_u32 s52, s52, s53 // (op 1255) 

s_mul_hi_u32 s53, s52, s13 // (op 1255) 
s_sub_u32 s54, s52, s53 // (op 1255) 
s_lshr_b32 s55, s54, 1 // (op 1255) 

s_add_u32 s54, s55, s53 // (op 1255) 


s_lshr_b32 s53, s54, s11 // (op 1255) 

s_and_b32 s54, s11, 2147483648 // (op 1255) 
s_cmp_eq_i32 s54, 0 // (op 1255) 
s_mov_b32 s55, scc // (op 1255) 

s_cmp_eq_i32 s55, 0 // (op 1255) 
s_cselect_b32 s54, s52, s53 // (op 1255) 


s_mul_hi_u32 s53, s54, s12 // (op 1255) 
s_sub_u32 s55, s54, s53 // (op 1255) 
s_lshr_b32 s76, s55, 1 // (op 1255) 

s_add_u32 s55, s76, s53 // (op 1255) 


s_lshr_b32 s53, s55, s10 // (op 1255) 

s_and_b32 s55, s10, 2147483648 // (op 1255) 
s_cmp_eq_i32 s55, 0 // (op 1255) 
s_mov_b32 s76, scc // (op 1255) 

s_cmp_eq_i32 s76, 0 // (op 1255) 
s_cselect_b32 s55, s54, s53 // (op 1255) 


s_mul_i32 s53, s55, s14 // (op 1255) 

s_sub_u32 s55, s54, s53 // (op 1255) 

v_lshl_add_u32 v15, s55, 5, v13 // (op 1255) 


v_mul_lo_u32 v13, v15, s38 // (op 1255) 

s_ashr_i64 s[76:77], s[2:3], 7 // (op 1255) 
s_mul_i32 s53, s54, s76 // (op 1255) 


s_sub_u32 s54, s52, s53 // (op 1255) 


v_lshl_add_u32 v15, s54, 2, v16 // (op 1255) 


v_add_u32 v16, v15, v13 // (op 1255) 


s_mul_i32 s53, s38, s23 // (op 1257) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 1257) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 1257) least significant
s_mul_hi_u32 s55, s38, s22 // (op 1257) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1257) most significant: sum
s_add_u32 s53, s53, s55 // (op 1257) most significant: sum


s_mov_b32 s72, s40 // (op 1257) reg expression
s_mov_b32 s73, s41 // (op 1257) reg expression
s_mov_b32 s74, s52 // (op 1257) reg expression
s_mov_b32 s75, 131072 // (op 1257) call()

v_and_b32 v13, -4, v1 // (op 1406) 
v_sub_u32 v15, v1, v13 // (op 1406) 

v_and_b32 v13, 4294967292, v1 // (op 1406) 
v_add_u32 v17, v13, v15 // (op 1406) 


s_and_b32 s52, s7, 4294967294 // (op 510) 
s_cmp_lt_u32 s52, s7 // (op 509) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_509 // (op 509) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_509 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_509:


s_add_u32 s52, s52, 1 // (op 511) 
v_mov_b32 v9, v16 // (op 514) 

buffer_load_ubyte v11, v9, s[72:75], 0 offen offset: 0 // (op 514) Load value

v_add_u32 v16, 4, v16 // (op 1247) 


s_barrier  // (op 1702) 
v_mov_b32 v9, v8 // (op 520) 




buffer_load_dwordx4 v[18:21], v9, s[60:63], 0 offen // (op 520) Load value

v_add_u32 v8, 64, v8 // (op 855) 
s_waitcnt vmcnt(1)
ds_write_b8 v17, v11 // (op 515) Store local data 



v_mov_b32 v9, v12 // (op 522) 

buffer_load_dwordx4 v[22:25], v9, s[64:67], 0 offen // (op 522) Load value
buffer_load_dwordx4 v[26:29], v9, s[64:67], 0 offen offset:64 // (op 522) Load value

v_add_u32 v12, 128, v12 // (op 991) 


v_mov_b32 v9, v14 // (op 524) 

buffer_load_ubyte v11, v9, s[68:71], 0 offen offset: 0 // (op 524) Load value

v_add_u32 v14, 4, v14 // (op 1119) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1705) 



ds_read_u8 v9, v0 // (op 517) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[22:29], a[0:3], v11, v9 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 518) 




s_cmp_lt_u32 s52, s7 // (op 509) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_509 // (op 509) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_509 if true)
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_509:






s_barrier  // (op 1853) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_lg_u32 s18, 0 // (op 269) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Send_Tile_269 // (op 269) Condition: False, jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Send_Tile_269
v_lshrrev_b32 v8, 5, v1 // (op 1571) 
v_lshlrev_b32 v9, 2, v8 // (op 1571) 

v_lshl_add_u32 v8, s16, 4, v9 // (op 1571) 

v_and_b32 v9, -32, v1 // (op 1571) 
v_sub_u32 v11, v1, v9 // (op 1571) 

v_lshl_add_u32 v9, v8, 5, v11 // (op 1571) 


v_lshlrev_b32 v8, 2, v9 // (op 1571) 

s_lshl_b32 s52, s15, 11 // (op 1573) 
s_mov_b32 s76, s42 // (op 1573) reg expression
s_mov_b32 s77, s43 // (op 1573) reg expression
s_mov_b32 s78, s52 // (op 1573) reg expression
s_mov_b32 s79, 131072 // (op 1573) call()




buffer_store_dword a0, v8, s[76:79], 0 offen sc1 // (op 266) Store value
buffer_store_dword a1, v8, s[76:79], 0 offen offset:128 sc1 // (op 266) Store value
buffer_store_dword a2, v8, s[76:79], 0 offen offset:256 sc1 // (op 266) Store value
buffer_store_dword a3, v8, s[76:79], 0 offen offset:384 sc1 // (op 266) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 270) 
v_cmp_eq_i32 vcc, v1, 0 // (op 274) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_274 // (op 274) Condition: False, jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_274
s_mov_b32 s52, 1 // (op 272) call()
s_mov_b32 s55, 0 // (op 273) convert
s_mov_b32 s54, s16 // (op 273) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 273) 
s_add_u32 s80, s44, s54 // (op 273) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 273) End of Int64 add, unlocking SCC
s_store_dword s52, s[80:81], 0 glc // (op 273) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_274 // (op 274) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_274
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_274:
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_274:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Send_Tile_269 // (op 269) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Send_Tile_269
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Send_Tile_269:
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Send_Tile_269:
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
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Receive_Tile_285 // (op 285) Condition: False, jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Receive_Tile_285
s_mov_b32 s52, 0 // (op 251) call()
s_mul_i32 s53, s52, s8 // (op 250) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 250) 
s_mov_b32 s80, s54 // (op 250) convert

s_sub_u32 s54, s80, s19 // (op 250) 

s_sub_u32 s55, s54, 1 // (op 250) 

s_cmp_lt_u32 s53, s55 // (op 250) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_250 // (op 250) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_250 if false)
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_ReceiveTileLoop_250:
s_add_u32 s53, s16, 1 // (op 1588) 
v_lshrrev_b32 v8, 5, v1 // (op 1588) 
s_add_u32 s54, s53, s52 // (op 1588) 
s_lshl_b32 s54, s54, 4 // (op 1588) 

v_lshlrev_b32 v9, 2, v8 // (op 1588) 

v_add_lshl_u32 v8, s54, v9, 5 // (op 1588) 


v_and_b32 v9, -32, v1 // (op 1588) 
v_sub_u32 v11, v1, v9 // (op 1588) 

v_add_lshl_u32 v9, v8, v11, 2 // (op 1588) 


s_lshl_b32 s53, s15, 11 // (op 1590) 
s_mov_b32 s76, s42 // (op 1590) reg expression
s_mov_b32 s77, s43 // (op 1590) reg expression
s_mov_b32 s78, s53 // (op 1590) reg expression
s_mov_b32 s79, 131072 // (op 1590) call()

RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_289:
s_add_u32 s53, s16, 1 // (op 287) 
s_add_u32 s54, s53, s52 // (op 287) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 287) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 287) 

s_add_u32 s80, s44, s54 // (op 287) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 287) End of Int64 add, unlocking SCC
s_load_dword s53, s[80:81], 0 glc // (op 287) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 289) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_289 // (op 289) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_289 if true)
s_barrier  // (op 293) 
v_cmp_eq_i32 vcc, v1, 0 // (op 292) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_292 // (op 292) Condition: False, jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_292
s_mov_b32 s53, 0 // (op 290) call()
s_add_u32 s54, s16, 1 // (op 291) 
s_add_u32 s80, s54, s52 // (op 291) Start of Int64 add, locking SCC
s_addc_u32 s81, 0, 0 // (op 291) End of Int64 add, unlocking SCC
s_lshl_b64 s[80:81], s[80:81], 2 // (op 291) 

s_add_u32 s54, s44, s80 // (op 291) Start of Int64 add, locking SCC
s_addc_u32 s55, s45, s81 // (op 291) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 291) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_292 // (op 292) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_292
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_292:
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_292:




buffer_load_dword v12, v9, s[76:79], 0 offen // (op 304) Load value
buffer_load_dword v13, v9, s[76:79], 0 offen offset:128 // (op 304) Load value
buffer_load_dword v14, v9, s[76:79], 0 offen offset:256 // (op 304) Load value
buffer_load_dword v15, v9, s[76:79], 0 offen offset:384 // (op 304) Load value
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

s_add_u32 s52, s52, 1 // (op 252) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 250) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 250) 
s_mov_b32 s80, s54 // (op 250) convert

s_sub_u32 s54, s80, s19 // (op 250) 

s_sub_u32 s55, s54, 1 // (op 250) 

s_cmp_lt_u32 s53, s55 // (op 250) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_ReceiveTileLoop_250 // (op 250) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_ReceiveTileLoop_250 if true)
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_250:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Receive_Tile_285 // (op 285) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Receive_Tile_285
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Receive_Tile_285:
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Receive_Tile_285:

s_cmp_eq_i32 s18, 0 // (op 328) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_328 // (op 328) Condition: False, jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_328
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 106) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_106 // (op 106) Condition: False, jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_106
v_accvgpr_read v12, a0 // (op 38) 
v_accvgpr_read v13, a1 // (op 38) 
v_accvgpr_read v14, a2 // (op 38) 
v_accvgpr_read v15, a3 // (op 38) 
v_mul_f32 v16, v7, v12 // (op 38) 
v_mul_f32 v17, v7, v13 // (op 38) 
v_mul_f32 v18, v7, v14 // (op 38) 
v_mul_f32 v19, v7, v15 // (op 38) 

v_lshrrev_b32 v8, 7, v1 // (op 1448) 
s_mul_i32 s52, s16, s8 // (op 1448) 
s_add_u32 s52, s52, s0 // (op 1448) 
v_and_b32 v9, 63, v1 // (op 1448) 
s_mul_hi_u32 s53, s52, s13 // (op 1448) 
v_lshrrev_b32 v11, 4, v9 // (op 1448) 
s_sub_u32 s54, s52, s53 // (op 1448) 
v_lshlrev_b32 v12, 2, v11 // (op 1448) 

s_lshr_b32 s55, s54, 1 // (op 1448) 

v_lshl_add_u32 v11, v8, 4, v12 // (op 1448) 


s_add_u32 s54, s55, s53 // (op 1448) 


v_lshrrev_b32 v8, 2, v9 // (op 1448) 

s_lshr_b32 s53, s54, s11 // (op 1448) 

v_and_b32 v9, 3, v8 // (op 1448) 

s_and_b32 s54, s11, 2147483648 // (op 1448) 
v_and_b32 v8, 3, v1 // (op 1448) 
s_cmp_eq_i32 s54, 0 // (op 1448) 
s_mov_b32 s55, scc // (op 1448) 

v_lshl_add_u32 v12, v9, 2, v8 // (op 1448) 


s_cmp_eq_i32 s55, 0 // (op 1448) 
s_cselect_b32 s54, s52, s53 // (op 1448) 



v_lshrrev_b32 v8, 6, v1 // (op 1448) 
s_mul_hi_u32 s55, s54, s12 // (op 1448) 
v_and_b32 v9, 1, v8 // (op 1448) 

s_sub_u32 s52, s54, s55 // (op 1448) 
v_lshl_add_u32 v8, v9, 4, v12 // (op 1448) 


s_lshr_b32 s53, s52, 1 // (op 1448) 

s_add_u32 s52, s53, s55 // (op 1448) 


s_lshr_b32 s53, s52, s10 // (op 1448) 

s_and_b32 s52, s10, 2147483648 // (op 1448) 
s_cmp_eq_i32 s52, 0 // (op 1448) 
s_mov_b32 s55, scc // (op 1448) 

s_cmp_eq_i32 s55, 0 // (op 1448) 
s_cselect_b32 s52, s54, s53 // (op 1448) 


s_mul_i32 s53, s52, s14 // (op 1448) 
v_lshl_add_u32 v9, s52, 4, v11 // (op 1448) 


s_sub_u32 s52, s54, s53 // (op 1448) 


v_lshl_add_u32 v11, s52, 5, v8 // (op 1448) 


v_mul_lo_u32 v8, v11, s46 // (op 1448) 

v_add_lshl_u32 v11, v9, v8, 2 // (op 1448) 


s_mul_i32 s53, s46, s23 // (op 1450) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1450) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1450) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1450) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1450) most significant: sum
s_add_u32 s53, s53, s55 // (op 1450) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1450) 

s_mov_b32 s76, s48 // (op 1450) reg expression
s_mov_b32 s77, s49 // (op 1450) reg expression
s_mov_b32 s78, s54 // (op 1450) reg expression
s_mov_b32 s79, 131072 // (op 1450) call()




buffer_store_dwordx4 v[16:19], v11, s[76:79], 0 offen // (op 44) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_106 // (op 106) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_106
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_106:



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


v_lshrrev_b32 v11, 7, v1 // (op 1465) 
s_mul_i32 s52, s16, s8 // (op 1465) 
s_add_u32 s52, s52, s0 // (op 1465) 
v_and_b32 v8, 63, v1 // (op 1465) 
s_mul_hi_u32 s53, s52, s13 // (op 1465) 
v_lshrrev_b32 v9, 4, v8 // (op 1465) 
s_sub_u32 s54, s52, s53 // (op 1465) 
v_lshlrev_b32 v16, 2, v9 // (op 1465) 

s_lshr_b32 s55, s54, 1 // (op 1465) 

v_lshl_add_u32 v9, v11, 4, v16 // (op 1465) 


s_add_u32 s54, s55, s53 // (op 1465) 


v_lshrrev_b32 v11, 2, v8 // (op 1465) 

s_lshr_b32 s53, s54, s11 // (op 1465) 

v_and_b32 v8, 3, v11 // (op 1465) 

s_and_b32 s54, s11, 2147483648 // (op 1465) 
v_and_b32 v11, 3, v1 // (op 1465) 
s_cmp_eq_i32 s54, 0 // (op 1465) 
s_mov_b32 s55, scc // (op 1465) 

v_lshl_add_u32 v16, v8, 2, v11 // (op 1465) 


s_cmp_eq_i32 s55, 0 // (op 1465) 
s_cselect_b32 s54, s52, s53 // (op 1465) 



v_lshrrev_b32 v8, 6, v1 // (op 1465) 
s_mul_hi_u32 s55, s54, s12 // (op 1465) 
v_and_b32 v11, 1, v8 // (op 1465) 

s_sub_u32 s52, s54, s55 // (op 1465) 
v_lshl_add_u32 v8, v11, 4, v16 // (op 1465) 


s_lshr_b32 s53, s52, 1 // (op 1465) 

s_add_u32 s52, s53, s55 // (op 1465) 


s_lshr_b32 s53, s52, s10 // (op 1465) 

s_and_b32 s52, s10, 2147483648 // (op 1465) 
s_cmp_eq_i32 s52, 0 // (op 1465) 
s_mov_b32 s55, scc // (op 1465) 

s_cmp_eq_i32 s55, 0 // (op 1465) 
s_cselect_b32 s52, s54, s53 // (op 1465) 


s_mul_i32 s53, s52, s14 // (op 1465) 
v_lshl_add_u32 v11, s52, 4, v9 // (op 1465) 


s_sub_u32 s52, s54, s53 // (op 1465) 


v_lshl_add_u32 v9, s52, 5, v8 // (op 1465) 


v_mul_lo_u32 v8, v9, s46 // (op 1465) 

v_add_lshl_u32 v9, v11, v8, 2 // (op 1465) 


s_mul_i32 s53, s46, s23 // (op 1467) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1467) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1467) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1467) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1467) most significant: sum
s_add_u32 s53, s53, s55 // (op 1467) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1467) 

s_mov_b32 s76, s48 // (op 1467) reg expression
s_mov_b32 s77, s49 // (op 1467) reg expression
s_mov_b32 s78, s54 // (op 1467) reg expression
s_mov_b32 s79, 131072 // (op 1467) call()




buffer_store_dwordx4 v[12:15], v9, s[76:79], 0 offen // (op 117) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_106:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_328 // (op 328) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_328
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_328:
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_328:

s_add_u32 s0, s0, s7 // (op 243) 
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
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_SKStreamTileLoop_241 // (op 241) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_SKStreamTileLoop_241 if true)
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_241:











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
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_396 // (op 396) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_396 if false)
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_DPStreamTileLoop_396:
v_lshrrev_b32 v0, 7, v1 // (op 1376) 
s_mul_i32 s15, s16, s50 // (op 1376) 
s_add_u32 s15, s15, s0 // (op 1376) 
v_and_b32 v5, 63, v1 // (op 1376) 
s_add_u32 s6, s15, s9 // (op 1376) 

v_lshrrev_b32 v8, 4, v5 // (op 1376) 
s_mul_hi_u32 s15, s6, s13 // (op 1376) 
v_lshlrev_b32 v9, 2, v8 // (op 1376) 

s_sub_u32 s7, s6, s15 // (op 1376) 
v_lshl_add_u32 v8, v0, 4, v9 // (op 1376) 


s_lshr_b32 s8, s7, 1 // (op 1376) 

v_lshrrev_b32 v0, 2, v5 // (op 1376) 

s_add_u32 s7, s8, s15 // (op 1376) 


v_and_b32 v5, 3, v0 // (op 1376) 

s_lshr_b32 s8, s7, s11 // (op 1376) 

v_and_b32 v0, 3, v1 // (op 1376) 
s_and_b32 s7, s11, 2147483648 // (op 1376) 
v_lshl_add_u32 v9, v5, 2, v0 // (op 1376) 


s_cmp_eq_i32 s7, 0 // (op 1376) 
s_mov_b32 s15, scc // (op 1376) 

v_lshrrev_b32 v0, 6, v1 // (op 1376) 
s_cmp_eq_i32 s15, 0 // (op 1376) 
s_cselect_b32 s7, s6, s8 // (op 1376) 



v_and_b32 v5, 1, v0 // (op 1376) 

s_mul_hi_u32 s6, s7, s12 // (op 1376) 
v_lshl_add_u32 v0, v5, 4, v9 // (op 1376) 


s_sub_u32 s8, s7, s6 // (op 1376) 
s_lshr_b32 s15, s8, 1 // (op 1376) 

s_add_u32 s8, s15, s6 // (op 1376) 


s_lshr_b32 s6, s8, s10 // (op 1376) 

s_and_b32 s8, s10, 2147483648 // (op 1376) 
s_cmp_eq_i32 s8, 0 // (op 1376) 
s_mov_b32 s15, scc // (op 1376) 

s_cmp_eq_i32 s15, 0 // (op 1376) 
s_cselect_b32 s8, s7, s6 // (op 1376) 


s_mul_i32 s6, s8, s14 // (op 1376) 
v_lshl_add_u32 v5, s8, 4, v8 // (op 1376) 


s_sub_u32 s8, s7, s6 // (op 1376) 


v_lshl_add_u32 v8, s8, 5, v0 // (op 1376) 


v_mul_lo_u32 v0, v8, s4 // (op 1376) 

v_add_lshl_u32 v8, v5, v0, 2 // (op 1376) 


s_mul_i32 s19, s4, s23 // (op 1378) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 1378) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 1378) least significant
s_mul_hi_u32 s6, s4, s22 // (op 1378) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1378) most significant: sum
s_add_u32 s19, s19, s6 // (op 1378) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1378) 

s_mov_b32 s56, s20 // (op 1378) reg expression
s_mov_b32 s57, s21 // (op 1378) reg expression
s_mov_b32 s58, s6 // (op 1378) reg expression
s_mov_b32 s59, 131072 // (op 1378) call()

v_accvgpr_write a0, 0.00000 // (op 153) call()
v_accvgpr_write a1, 0.00000 // (op 153) call()
v_accvgpr_write a2, 0.00000 // (op 153) call()
v_accvgpr_write a3, 0.00000 // (op 153) call()
v_lshrrev_b32 v0, 7, v1 // (op 936) 
s_mul_i32 s15, s16, s50 // (op 936) 
s_add_u32 s15, s15, s0 // (op 936) 
v_and_b32 v5, 15, v1 // (op 936) 
s_add_u32 s6, s15, s9 // (op 936) 

v_lshl_add_u32 v9, v0, 4, v5 // (op 936) 


s_mul_hi_u32 s15, s6, s13 // (op 936) 
v_and_b32 v0, 63, v1 // (op 936) 
s_sub_u32 s7, s6, s15 // (op 936) 
v_lshrrev_b32 v5, 4, v0 // (op 936) 

s_lshr_b32 s8, s7, 1 // (op 936) 

v_lshlrev_b32 v0, 5, v5 // (op 936) 

s_add_u32 s7, s8, s15 // (op 936) 


s_lshr_b32 s8, s7, s11 // (op 936) 

s_and_b32 s7, s11, 2147483648 // (op 936) 
s_cmp_eq_i32 s7, 0 // (op 936) 
s_mov_b32 s15, scc // (op 936) 

s_cmp_eq_i32 s15, 0 // (op 936) 
s_cselect_b32 s7, s6, s8 // (op 936) 


s_mul_hi_u32 s8, s7, s12 // (op 936) 
s_sub_u32 s15, s7, s8 // (op 936) 
s_lshr_b32 s18, s15, 1 // (op 936) 

s_add_u32 s15, s18, s8 // (op 936) 


s_lshr_b32 s8, s15, s10 // (op 936) 

s_and_b32 s15, s10, 2147483648 // (op 936) 
s_cmp_eq_i32 s15, 0 // (op 936) 
s_mov_b32 s18, scc // (op 936) 

s_cmp_eq_i32 s18, 0 // (op 936) 
s_cselect_b32 s15, s7, s8 // (op 936) 


v_lshl_add_u32 v5, s15, 4, v9 // (op 936) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 936) 
s_mul_i32 s8, s7, s18 // (op 936) 


s_sub_u32 s7, s6, s8 // (op 936) 


v_lshl_add_u32 v9, s7, 7, v0 // (op 936) 


v_mov_b32 v11, 0 // (op 936) convert
v_mov_b32 v10, v9 // (op 936) convert

v_mul_lo_u32 v13, v5, s25 // (op 936) most significant: low of low * high
v_mul_hi_u32 v0, v5, s24 // (op 936) most significant: high of low * low
v_mul_lo_u32 v12, v5, s24 // (op 936) least significant: low of low * low
v_add_u32 v13, v13, v0 // (op 936) most significant: sum

v_add_co_u32 v12, s[18:19], v12, v10 // (op 936) least significant half
v_addc_co_u32 v13, s[18:19], v13, v11, s[18:19] // (op 936) most significant half



v_lshlrev_b64 v[10:11], 2, v[12:13] // (op 936) 

v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 936) 

v_mov_b32 v0, v12 // (op 936) convert

s_mul_i32 s19, s24, s29 // (op 938) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 938) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 938) least significant
s_mul_hi_u32 s6, s24, s28 // (op 938) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 938) most significant: sum
s_add_u32 s19, s19, s6 // (op 938) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 938) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 938) 

s_mov_b32 s60, s26 // (op 938) reg expression
s_mov_b32 s61, s27 // (op 938) reg expression
s_mov_b32 s62, s18 // (op 938) reg expression
s_mov_b32 s63, 131072 // (op 938) call()

v_lshrrev_b32 v5, 7, v1 // (op 955) 
s_mul_i32 s15, s16, s50 // (op 955) 
s_add_u32 s15, s15, s0 // (op 955) 
v_and_b32 v9, 15, v1 // (op 955) 
s_add_u32 s6, s15, s9 // (op 955) 

v_lshl_add_u32 v10, v5, 4, v9 // (op 955) 


s_mul_hi_u32 s15, s6, s13 // (op 955) 
v_and_b32 v5, 63, v1 // (op 955) 
s_sub_u32 s7, s6, s15 // (op 955) 
v_lshrrev_b32 v9, 4, v5 // (op 955) 

s_lshr_b32 s8, s7, 1 // (op 955) 

v_lshlrev_b32 v5, 5, v9 // (op 955) 

s_add_u32 s7, s8, s15 // (op 955) 


s_lshr_b32 s8, s7, s11 // (op 955) 

s_and_b32 s7, s11, 2147483648 // (op 955) 
s_cmp_eq_i32 s7, 0 // (op 955) 
s_mov_b32 s15, scc // (op 955) 

s_cmp_eq_i32 s15, 0 // (op 955) 
s_cselect_b32 s7, s6, s8 // (op 955) 


s_mul_hi_u32 s8, s7, s12 // (op 955) 
s_sub_u32 s15, s7, s8 // (op 955) 
s_lshr_b32 s18, s15, 1 // (op 955) 

s_add_u32 s15, s18, s8 // (op 955) 


s_lshr_b32 s8, s15, s10 // (op 955) 

s_and_b32 s15, s10, 2147483648 // (op 955) 
s_cmp_eq_i32 s15, 0 // (op 955) 
s_mov_b32 s18, scc // (op 955) 

s_cmp_eq_i32 s18, 0 // (op 955) 
s_cselect_b32 s15, s7, s8 // (op 955) 


v_lshl_add_u32 v9, s15, 4, v10 // (op 955) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 955) 
s_mul_i32 s8, s7, s18 // (op 955) 


s_sub_u32 s7, s6, s8 // (op 955) 


v_lshl_add_u32 v10, s7, 7, v5 // (op 955) 


v_mov_b32 v13, 0 // (op 955) convert
v_mov_b32 v12, v10 // (op 955) convert

v_mul_lo_u32 v11, v9, s25 // (op 955) most significant: low of low * high
v_mul_hi_u32 v5, v9, s24 // (op 955) most significant: high of low * low
v_mul_lo_u32 v10, v9, s24 // (op 955) least significant: low of low * low
v_add_u32 v11, v11, v5 // (op 955) most significant: sum

v_add_co_u32 v10, s[18:19], v10, v12 // (op 955) least significant half
v_addc_co_u32 v11, s[18:19], v11, v13, s[18:19] // (op 955) most significant half



v_lshlrev_b64 v[12:13], 2, v[10:11] // (op 955) 

v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 955) 

v_mov_b32 v5, v10 // (op 955) convert

s_mul_i32 s19, s24, s29 // (op 957) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 957) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 957) least significant
s_mul_hi_u32 s6, s24, s28 // (op 957) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 957) most significant: sum
s_add_u32 s19, s19, s6 // (op 957) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 957) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 957) 

s_mov_b32 s60, s26 // (op 957) reg expression
s_mov_b32 s61, s27 // (op 957) reg expression
s_mov_b32 s62, s18 // (op 957) reg expression
s_mov_b32 s63, 131072 // (op 957) call()

v_lshrrev_b32 v9, 6, v1 // (op 1064) 
s_mul_i32 s15, s16, s50 // (op 1064) 
s_add_u32 s15, s15, s0 // (op 1064) 
v_and_b32 v10, 1, v9 // (op 1064) 

s_add_u32 s6, s15, s9 // (op 1064) 

v_and_b32 v9, 15, v1 // (op 1064) 
s_mul_hi_u32 s15, s6, s13 // (op 1064) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 1064) 


s_sub_u32 s7, s6, s15 // (op 1064) 
v_and_b32 v9, 48, v1 // (op 1064) 
s_lshr_b32 s8, s7, 1 // (op 1064) 

s_add_u32 s7, s8, s15 // (op 1064) 


s_lshr_b32 s8, s7, s11 // (op 1064) 

s_and_b32 s7, s11, 2147483648 // (op 1064) 
s_cmp_eq_i32 s7, 0 // (op 1064) 
s_mov_b32 s15, scc // (op 1064) 

s_cmp_eq_i32 s15, 0 // (op 1064) 
s_cselect_b32 s7, s6, s8 // (op 1064) 


s_mul_hi_u32 s8, s7, s12 // (op 1064) 
s_sub_u32 s15, s7, s8 // (op 1064) 
s_lshr_b32 s18, s15, 1 // (op 1064) 

s_add_u32 s15, s18, s8 // (op 1064) 


s_lshr_b32 s8, s15, s10 // (op 1064) 

s_and_b32 s15, s10, 2147483648 // (op 1064) 
s_cmp_eq_i32 s15, 0 // (op 1064) 
s_mov_b32 s18, scc // (op 1064) 

s_cmp_eq_i32 s18, 0 // (op 1064) 
s_cselect_b32 s15, s7, s8 // (op 1064) 


s_mul_i32 s8, s15, s14 // (op 1064) 

s_sub_u32 s15, s7, s8 // (op 1064) 

v_lshl_add_u32 v10, s15, 5, v11 // (op 1064) 


v_mul_lo_u32 v11, v10, s30 // (op 1064) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1064) 
s_mul_i32 s8, s7, s18 // (op 1064) 


s_sub_u32 s7, s6, s8 // (op 1064) 


v_lshl_add_u32 v10, s7, 7, v9 // (op 1064) 


v_add_u32 v9, v10, v11 // (op 1064) 


s_mul_i32 s19, s30, s23 // (op 1066) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 1066) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 1066) least significant
s_mul_hi_u32 s6, s30, s22 // (op 1066) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1066) most significant: sum
s_add_u32 s19, s19, s6 // (op 1066) most significant: sum


s_mov_b32 s64, s32 // (op 1066) reg expression
s_mov_b32 s65, s33 // (op 1066) reg expression
s_mov_b32 s66, s18 // (op 1066) reg expression
s_mov_b32 s67, 131072 // (op 1066) call()

v_lshrrev_b32 v10, 6, v1 // (op 1083) 
s_mul_i32 s15, s16, s50 // (op 1083) 
s_add_u32 s15, s15, s0 // (op 1083) 
v_and_b32 v11, 1, v10 // (op 1083) 

s_add_u32 s6, s15, s9 // (op 1083) 

v_and_b32 v10, 15, v1 // (op 1083) 
s_mul_hi_u32 s15, s6, s13 // (op 1083) 
v_lshl_add_u32 v12, v11, 4, v10 // (op 1083) 


s_sub_u32 s7, s6, s15 // (op 1083) 
v_and_b32 v10, 48, v1 // (op 1083) 
s_lshr_b32 s8, s7, 1 // (op 1083) 

s_add_u32 s7, s8, s15 // (op 1083) 


s_lshr_b32 s8, s7, s11 // (op 1083) 

s_and_b32 s7, s11, 2147483648 // (op 1083) 
s_cmp_eq_i32 s7, 0 // (op 1083) 
s_mov_b32 s15, scc // (op 1083) 

s_cmp_eq_i32 s15, 0 // (op 1083) 
s_cselect_b32 s7, s6, s8 // (op 1083) 


s_mul_hi_u32 s8, s7, s12 // (op 1083) 
s_sub_u32 s15, s7, s8 // (op 1083) 
s_lshr_b32 s18, s15, 1 // (op 1083) 

s_add_u32 s15, s18, s8 // (op 1083) 


s_lshr_b32 s8, s15, s10 // (op 1083) 

s_and_b32 s15, s10, 2147483648 // (op 1083) 
s_cmp_eq_i32 s15, 0 // (op 1083) 
s_mov_b32 s18, scc // (op 1083) 

s_cmp_eq_i32 s18, 0 // (op 1083) 
s_cselect_b32 s15, s7, s8 // (op 1083) 


s_mul_i32 s8, s15, s14 // (op 1083) 

s_sub_u32 s15, s7, s8 // (op 1083) 

v_lshl_add_u32 v11, s15, 5, v12 // (op 1083) 


v_mul_lo_u32 v12, v11, s30 // (op 1083) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1083) 
s_mul_i32 s8, s7, s18 // (op 1083) 


s_sub_u32 s7, s6, s8 // (op 1083) 


v_lshl_add_u32 v11, s7, 7, v10 // (op 1083) 


v_add_u32 v10, v11, v12 // (op 1083) 


s_mul_i32 s19, s30, s23 // (op 1085) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 1085) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 1085) least significant
s_mul_hi_u32 s6, s30, s22 // (op 1085) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1085) most significant: sum
s_add_u32 s19, s19, s6 // (op 1085) most significant: sum


s_mov_b32 s64, s32 // (op 1085) reg expression
s_mov_b32 s65, s33 // (op 1085) reg expression
s_mov_b32 s66, s18 // (op 1085) reg expression
s_mov_b32 s67, 131072 // (op 1085) call()

v_lshrrev_b32 v11, 7, v1 // (op 1192) 
s_mul_i32 s15, s16, s50 // (op 1192) 
s_add_u32 s15, s15, s0 // (op 1192) 
v_and_b32 v12, 15, v1 // (op 1192) 
s_add_u32 s6, s15, s9 // (op 1192) 

v_lshl_add_u32 v13, v11, 4, v12 // (op 1192) 


s_mul_hi_u32 s15, s6, s13 // (op 1192) 
v_and_b32 v11, 63, v1 // (op 1192) 
s_sub_u32 s7, s6, s15 // (op 1192) 
v_lshrrev_b32 v12, 4, v11 // (op 1192) 

s_lshr_b32 s8, s7, 1 // (op 1192) 

s_add_u32 s7, s8, s15 // (op 1192) 


s_lshr_b32 s8, s7, s11 // (op 1192) 

s_and_b32 s7, s11, 2147483648 // (op 1192) 
s_cmp_eq_i32 s7, 0 // (op 1192) 
s_mov_b32 s15, scc // (op 1192) 

s_cmp_eq_i32 s15, 0 // (op 1192) 
s_cselect_b32 s7, s6, s8 // (op 1192) 


s_mul_hi_u32 s8, s7, s12 // (op 1192) 
s_sub_u32 s15, s7, s8 // (op 1192) 
s_lshr_b32 s18, s15, 1 // (op 1192) 

s_add_u32 s15, s18, s8 // (op 1192) 


s_lshr_b32 s8, s15, s10 // (op 1192) 

s_and_b32 s15, s10, 2147483648 // (op 1192) 
s_cmp_eq_i32 s15, 0 // (op 1192) 
s_mov_b32 s18, scc // (op 1192) 

s_cmp_eq_i32 s18, 0 // (op 1192) 
s_cselect_b32 s15, s7, s8 // (op 1192) 


v_lshl_add_u32 v11, s15, 4, v13 // (op 1192) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1192) 
s_mul_i32 s8, s7, s18 // (op 1192) 


s_sub_u32 s7, s6, s8 // (op 1192) 


v_lshl_add_u32 v13, s7, 2, v12 // (op 1192) 


v_mul_lo_u32 v12, v11, s34 // (op 1192) 
v_add_u32 v12, v12, v13 // (op 1192) 


s_mul_i32 s19, s34, s29 // (op 1194) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1194) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1194) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1194) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1194) most significant: sum
s_add_u32 s19, s19, s6 // (op 1194) most significant: sum


s_mov_b32 s68, s36 // (op 1194) reg expression
s_mov_b32 s69, s37 // (op 1194) reg expression
s_mov_b32 s70, s18 // (op 1194) reg expression
s_mov_b32 s71, 131072 // (op 1194) call()

v_lshrrev_b32 v11, 7, v1 // (op 1211) 
s_mul_i32 s15, s16, s50 // (op 1211) 
s_add_u32 s15, s15, s0 // (op 1211) 
v_and_b32 v13, 15, v1 // (op 1211) 
s_add_u32 s6, s15, s9 // (op 1211) 

v_lshl_add_u32 v14, v11, 4, v13 // (op 1211) 


s_mul_hi_u32 s15, s6, s13 // (op 1211) 
v_and_b32 v11, 63, v1 // (op 1211) 
s_sub_u32 s7, s6, s15 // (op 1211) 
v_lshrrev_b32 v13, 4, v11 // (op 1211) 

s_lshr_b32 s8, s7, 1 // (op 1211) 

s_add_u32 s7, s8, s15 // (op 1211) 


s_lshr_b32 s8, s7, s11 // (op 1211) 

s_and_b32 s7, s11, 2147483648 // (op 1211) 
s_cmp_eq_i32 s7, 0 // (op 1211) 
s_mov_b32 s15, scc // (op 1211) 

s_cmp_eq_i32 s15, 0 // (op 1211) 
s_cselect_b32 s7, s6, s8 // (op 1211) 


s_mul_hi_u32 s8, s7, s12 // (op 1211) 
s_sub_u32 s15, s7, s8 // (op 1211) 
s_lshr_b32 s18, s15, 1 // (op 1211) 

s_add_u32 s15, s18, s8 // (op 1211) 


s_lshr_b32 s8, s15, s10 // (op 1211) 

s_and_b32 s15, s10, 2147483648 // (op 1211) 
s_cmp_eq_i32 s15, 0 // (op 1211) 
s_mov_b32 s18, scc // (op 1211) 

s_cmp_eq_i32 s18, 0 // (op 1211) 
s_cselect_b32 s15, s7, s8 // (op 1211) 


v_lshl_add_u32 v11, s15, 4, v14 // (op 1211) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1211) 
s_mul_i32 s8, s7, s18 // (op 1211) 


s_sub_u32 s7, s6, s8 // (op 1211) 


v_lshl_add_u32 v14, s7, 2, v13 // (op 1211) 


v_mul_lo_u32 v13, v11, s34 // (op 1211) 
v_add_u32 v13, v13, v14 // (op 1211) 


s_mul_i32 s19, s34, s29 // (op 1213) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1213) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1213) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1213) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1213) most significant: sum
s_add_u32 s19, s19, s6 // (op 1213) most significant: sum


s_mov_b32 s68, s36 // (op 1213) reg expression
s_mov_b32 s69, s37 // (op 1213) reg expression
s_mov_b32 s70, s18 // (op 1213) reg expression
s_mov_b32 s71, 131072 // (op 1213) call()

v_lshrrev_b32 v11, 2, v1 // (op 1320) 
s_mul_i32 s15, s16, s50 // (op 1320) 
s_add_u32 s15, s15, s0 // (op 1320) 
v_and_b32 v14, -4, v1 // (op 1320) 
s_add_u32 s6, s15, s9 // (op 1320) 

v_sub_u32 v15, v1, v14 // (op 1320) 

s_mul_hi_u32 s15, s6, s13 // (op 1320) 
s_sub_u32 s7, s6, s15 // (op 1320) 
s_lshr_b32 s8, s7, 1 // (op 1320) 

s_add_u32 s7, s8, s15 // (op 1320) 


s_lshr_b32 s8, s7, s11 // (op 1320) 

s_and_b32 s7, s11, 2147483648 // (op 1320) 
s_cmp_eq_i32 s7, 0 // (op 1320) 
s_mov_b32 s15, scc // (op 1320) 

s_cmp_eq_i32 s15, 0 // (op 1320) 
s_cselect_b32 s7, s6, s8 // (op 1320) 


s_mul_hi_u32 s8, s7, s12 // (op 1320) 
s_sub_u32 s15, s7, s8 // (op 1320) 
s_lshr_b32 s18, s15, 1 // (op 1320) 

s_add_u32 s15, s18, s8 // (op 1320) 


s_lshr_b32 s8, s15, s10 // (op 1320) 

s_and_b32 s15, s10, 2147483648 // (op 1320) 
s_cmp_eq_i32 s15, 0 // (op 1320) 
s_mov_b32 s18, scc // (op 1320) 

s_cmp_eq_i32 s18, 0 // (op 1320) 
s_cselect_b32 s15, s7, s8 // (op 1320) 


s_mul_i32 s8, s15, s14 // (op 1320) 

s_sub_u32 s15, s7, s8 // (op 1320) 

v_lshl_add_u32 v14, s15, 5, v11 // (op 1320) 


v_mul_lo_u32 v11, v14, s38 // (op 1320) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1320) 
s_mul_i32 s8, s7, s18 // (op 1320) 


s_sub_u32 s7, s6, s8 // (op 1320) 


v_lshl_add_u32 v14, s7, 2, v15 // (op 1320) 


v_add_u32 v15, v14, v11 // (op 1320) 


s_mul_i32 s19, s38, s23 // (op 1322) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 1322) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 1322) least significant
s_mul_hi_u32 s6, s38, s22 // (op 1322) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1322) most significant: sum
s_add_u32 s19, s19, s6 // (op 1322) most significant: sum


s_mov_b32 s72, s40 // (op 1322) reg expression
s_mov_b32 s73, s41 // (op 1322) reg expression
s_mov_b32 s74, s18 // (op 1322) reg expression
s_mov_b32 s75, 131072 // (op 1322) call()

v_lshrrev_b32 v11, 2, v1 // (op 1339) 
s_mul_i32 s15, s16, s50 // (op 1339) 
s_add_u32 s15, s15, s0 // (op 1339) 
v_and_b32 v14, -4, v1 // (op 1339) 
s_add_u32 s6, s15, s9 // (op 1339) 

v_sub_u32 v16, v1, v14 // (op 1339) 

s_mul_hi_u32 s15, s6, s13 // (op 1339) 
s_sub_u32 s7, s6, s15 // (op 1339) 
s_lshr_b32 s8, s7, 1 // (op 1339) 

s_add_u32 s7, s8, s15 // (op 1339) 


s_lshr_b32 s8, s7, s11 // (op 1339) 

s_and_b32 s7, s11, 2147483648 // (op 1339) 
s_cmp_eq_i32 s7, 0 // (op 1339) 
s_mov_b32 s15, scc // (op 1339) 

s_cmp_eq_i32 s15, 0 // (op 1339) 
s_cselect_b32 s7, s6, s8 // (op 1339) 


s_mul_hi_u32 s8, s7, s12 // (op 1339) 
s_sub_u32 s15, s7, s8 // (op 1339) 
s_lshr_b32 s18, s15, 1 // (op 1339) 

s_add_u32 s15, s18, s8 // (op 1339) 


s_lshr_b32 s8, s15, s10 // (op 1339) 

s_and_b32 s15, s10, 2147483648 // (op 1339) 
s_cmp_eq_i32 s15, 0 // (op 1339) 
s_mov_b32 s18, scc // (op 1339) 

s_cmp_eq_i32 s18, 0 // (op 1339) 
s_cselect_b32 s15, s7, s8 // (op 1339) 


s_mul_i32 s8, s15, s14 // (op 1339) 

s_sub_u32 s15, s7, s8 // (op 1339) 

v_lshl_add_u32 v14, s15, 5, v11 // (op 1339) 


v_mul_lo_u32 v11, v14, s38 // (op 1339) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1339) 
s_mul_i32 s8, s7, s18 // (op 1339) 


s_sub_u32 s7, s6, s8 // (op 1339) 


v_lshl_add_u32 v14, s7, 2, v16 // (op 1339) 


v_add_u32 v16, v14, v11 // (op 1339) 


s_mul_i32 s19, s38, s23 // (op 1341) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 1341) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 1341) least significant
s_mul_hi_u32 s6, s38, s22 // (op 1341) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1341) most significant: sum
s_add_u32 s19, s19, s6 // (op 1341) most significant: sum


s_mov_b32 s72, s40 // (op 1341) reg expression
s_mov_b32 s73, s41 // (op 1341) reg expression
s_mov_b32 s74, s18 // (op 1341) reg expression
s_mov_b32 s75, 131072 // (op 1341) call()




buffer_load_dwordx4 v[18:21], v0, s[60:63], 0 offen // (op 761) Load value



buffer_load_dwordx4 v[22:25], v9, s[64:67], 0 offen // (op 766) Load value
buffer_load_dwordx4 v[26:29], v9, s[64:67], 0 offen offset:64 // (op 766) Load value



buffer_load_ubyte v11, v12, s[68:71], 0 offen offset: 0 // (op 771) Load value



buffer_load_ubyte v14, v15, s[72:75], 0 offen offset: 0 // (op 775) Load value



buffer_load_dwordx4 v[30:33], v0, s[60:63], 0 offen offset:64 // (op 779) Load value




buffer_load_dwordx4 v[34:37], v9, s[64:67], 0 offen offset:128 // (op 784) Load value
buffer_load_dwordx4 v[38:41], v9, s[64:67], 0 offen offset:192 // (op 784) Load value




buffer_load_ubyte v0, v12, s[68:71], 0 offen offset:4 // (op 789) Load value




buffer_load_ubyte v9, v15, s[72:75], 0 offen offset:4 // (op 793) Load value

v_and_b32 v12, -4, v1 // (op 1545) 
v_sub_u32 v15, v1, v12 // (op 1545) 

v_and_b32 v12, 4294967292, v1 // (op 1545) 
v_add_u32 v17, v12, v15 // (op 1545) 


s_barrier  // (op 1712) 



s_waitcnt vmcnt(5)
ds_write_b8 v17, v14 // (op 795) Store local data 

v_and_b32 v12, -4, v1 // (op 1516) 
v_sub_u32 v15, v1, v12 // (op 1516) 

v_and_b32 v12, 4294967292, v1 // (op 1516) 
v_add_u32 v17, v12, v15 // (op 1516) 


v_and_b32 v12, -4, v1 // (op 1603) 
v_sub_u32 v15, v1, v12 // (op 1603) 

v_and_b32 v12, 4294967292, v1 // (op 1603) 
v_add_u32 v42, v12, v15 // (op 1603) 


s_mov_b32 s1, 0 // (op 155) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 154) 
s_mov_b32 s15, s18 // (op 154) convert

s_and_b32 s6, s15, 4294967294 // (op 154) 

s_cmp_lt_u32 s1, s6 // (op 154) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_154 // (op 154) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_154 if false)
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoop_154:
s_add_u32 s1, s1, 2 // (op 156) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1709) 



ds_read_u8 v12, v3 // (op 160) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[22:29], a[0:3], v11, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 161) 



v_mov_b32 v12, v5 // (op 163) 

buffer_load_dwordx4 v[18:21], v12, s[60:63], 0 offen offset:128 // (op 163) Load value


v_mov_b32 v15, v10 // (op 165) 

buffer_load_dwordx4 v[22:25], v15, s[64:67], 0 offen offset:256 // (op 165) Load value
buffer_load_dwordx4 v[26:29], v15, s[64:67], 0 offen offset:320 // (op 165) Load value


v_mov_b32 v43, v13 // (op 167) 

buffer_load_ubyte v11, v43, s[68:71], 0 offen offset:8 // (op 167) Load value


v_mov_b32 v44, v16 // (op 157) 

buffer_load_ubyte v14, v44, s[72:75], 0 offen offset:8 // (op 157) Load value
s_barrier  // (op 1728) 



s_waitcnt vmcnt(5)
ds_write_b8 v42, v9 offset:128 // (op 452) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1725) 



ds_read_u8 v45, v4 offset:128 // (op 454) Load local data 
s_barrier  // (op 1823) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:33], v[34:41], a[0:3], v0, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 455) 




buffer_load_dwordx4 v[30:33], v12, s[60:63], 0 offen offset:192 // (op 457) Load value




v_add_u32 v5, 128, v5 // (op 947) 
buffer_load_dwordx4 v[34:37], v15, s[64:67], 0 offen offset:384 // (op 459) Load value
buffer_load_dwordx4 v[38:41], v15, s[64:67], 0 offen offset:448 // (op 459) Load value




v_add_u32 v10, 256, v10 // (op 1075) 
buffer_load_ubyte v0, v43, s[68:71], 0 offen offset:12 // (op 461) Load value




v_add_u32 v13, 8, v13 // (op 1203) 
buffer_load_ubyte v9, v44, s[72:75], 0 offen offset:12 // (op 451) Load value

s_barrier  // (op 1715) 
v_add_u32 v16, 8, v16 // (op 1331) 



s_waitcnt vmcnt(5)
ds_write_b8 v17, v14 // (op 158) Store local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 154) 
s_mov_b32 s15, s18 // (op 154) convert

s_and_b32 s6, s15, 4294967294 // (op 154) 

s_cmp_lt_u32 s1, s6 // (op 154) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoop_154 // (op 154) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoop_154 if true)
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_154:










v_lshrrev_b32 v5, 7, v1 // (op 841) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 841) 
v_and_b32 v10, 15, v1 // (op 841) 
s_and_b32 s15, s18, 4294967294 // (op 841) 
v_lshl_add_u32 v11, v5, 4, v10 // (op 841) 


s_add_u32 s6, s0, s15 // (op 841) 

v_and_b32 v5, 63, v1 // (op 841) 
v_lshrrev_b32 v10, 4, v5 // (op 841) 

s_mul_i32 s15, s16, s50 // (op 841) 
s_add_u32 s15, s15, s6 // (op 841) 

v_lshlrev_b32 v5, 5, v10 // (op 841) 

s_add_u32 s6, s15, s9 // (op 841) 

s_mul_hi_u32 s15, s6, s13 // (op 841) 
s_sub_u32 s7, s6, s15 // (op 841) 
s_lshr_b32 s8, s7, 1 // (op 841) 

s_add_u32 s7, s8, s15 // (op 841) 


s_lshr_b32 s8, s7, s11 // (op 841) 

s_and_b32 s7, s11, 2147483648 // (op 841) 
s_cmp_eq_i32 s7, 0 // (op 841) 
s_mov_b32 s15, scc // (op 841) 

s_cmp_eq_i32 s15, 0 // (op 841) 
s_cselect_b32 s7, s6, s8 // (op 841) 


s_mul_hi_u32 s8, s7, s12 // (op 841) 
s_sub_u32 s15, s7, s8 // (op 841) 
s_lshr_b32 s42, s15, 1 // (op 841) 

s_add_u32 s15, s42, s8 // (op 841) 


s_lshr_b32 s8, s15, s10 // (op 841) 

s_and_b32 s15, s10, 2147483648 // (op 841) 
s_cmp_eq_i32 s15, 0 // (op 841) 
s_mov_b32 s42, scc // (op 841) 

s_cmp_eq_i32 s42, 0 // (op 841) 
s_cselect_b32 s15, s7, s8 // (op 841) 


s_mul_i32 s8, s7, s18 // (op 841) 


v_lshl_add_u32 v10, s15, 4, v11 // (op 841) 


s_sub_u32 s7, s6, s8 // (op 841) 


v_lshl_add_u32 v11, s7, 7, v5 // (op 841) 


v_mov_b32 v13, 0 // (op 841) convert
v_mov_b32 v12, v11 // (op 841) convert

v_mul_lo_u32 v15, v10, s25 // (op 841) most significant: low of low * high
v_mul_hi_u32 v5, v10, s24 // (op 841) most significant: high of low * low
v_mul_lo_u32 v14, v10, s24 // (op 841) least significant: low of low * low
v_add_u32 v15, v15, v5 // (op 841) most significant: sum

v_add_co_u32 v14, s[18:19], v14, v12 // (op 841) least significant half
v_addc_co_u32 v15, s[18:19], v15, v13, s[18:19] // (op 841) most significant half



v_lshlrev_b64 v[10:11], 2, v[14:15] // (op 841) 

v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 841) 

v_mov_b32 v5, v12 // (op 841) convert

s_mul_i32 s19, s24, s29 // (op 843) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 843) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 843) least significant
s_mul_hi_u32 s6, s24, s28 // (op 843) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 843) most significant: sum
s_add_u32 s19, s19, s6 // (op 843) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 843) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 843) 

s_mov_b32 s60, s26 // (op 843) reg expression
s_mov_b32 s61, s27 // (op 843) reg expression
s_mov_b32 s62, s18 // (op 843) reg expression
s_mov_b32 s63, 131072 // (op 843) call()

v_lshrrev_b32 v10, 6, v1 // (op 977) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 977) 
v_and_b32 v11, 1, v10 // (op 977) 

s_and_b32 s15, s18, 4294967294 // (op 977) 
v_and_b32 v10, 15, v1 // (op 977) 
s_add_u32 s6, s0, s15 // (op 977) 

v_lshl_add_u32 v12, v11, 4, v10 // (op 977) 


v_and_b32 v10, 48, v1 // (op 977) 
s_mul_i32 s15, s16, s50 // (op 977) 
s_add_u32 s15, s15, s6 // (op 977) 

s_add_u32 s6, s15, s9 // (op 977) 

s_mul_hi_u32 s15, s6, s13 // (op 977) 
s_sub_u32 s7, s6, s15 // (op 977) 
s_lshr_b32 s8, s7, 1 // (op 977) 

s_add_u32 s7, s8, s15 // (op 977) 


s_lshr_b32 s8, s7, s11 // (op 977) 

s_and_b32 s7, s11, 2147483648 // (op 977) 
s_cmp_eq_i32 s7, 0 // (op 977) 
s_mov_b32 s15, scc // (op 977) 

s_cmp_eq_i32 s15, 0 // (op 977) 
s_cselect_b32 s7, s6, s8 // (op 977) 


s_mul_hi_u32 s8, s7, s12 // (op 977) 
s_sub_u32 s15, s7, s8 // (op 977) 
s_lshr_b32 s42, s15, 1 // (op 977) 

s_add_u32 s15, s42, s8 // (op 977) 


s_lshr_b32 s8, s15, s10 // (op 977) 

s_and_b32 s15, s10, 2147483648 // (op 977) 
s_cmp_eq_i32 s15, 0 // (op 977) 
s_mov_b32 s42, scc // (op 977) 

s_cmp_eq_i32 s42, 0 // (op 977) 
s_cselect_b32 s15, s7, s8 // (op 977) 


s_mul_i32 s8, s15, s14 // (op 977) 

s_sub_u32 s15, s7, s8 // (op 977) 

v_lshl_add_u32 v11, s15, 5, v12 // (op 977) 


s_mul_i32 s8, s7, s18 // (op 977) 


v_mul_lo_u32 v12, v11, s30 // (op 977) 

s_sub_u32 s7, s6, s8 // (op 977) 


v_lshl_add_u32 v11, s7, 7, v10 // (op 977) 


v_add_u32 v10, v11, v12 // (op 977) 


s_mul_i32 s19, s30, s23 // (op 979) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 979) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 979) least significant
s_mul_hi_u32 s6, s30, s22 // (op 979) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 979) most significant: sum
s_add_u32 s19, s19, s6 // (op 979) most significant: sum


s_mov_b32 s64, s32 // (op 979) reg expression
s_mov_b32 s65, s33 // (op 979) reg expression
s_mov_b32 s66, s18 // (op 979) reg expression
s_mov_b32 s67, 131072 // (op 979) call()

v_lshrrev_b32 v11, 7, v1 // (op 1105) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1105) 
v_and_b32 v12, 15, v1 // (op 1105) 
s_and_b32 s15, s18, 4294967294 // (op 1105) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 1105) 


s_add_u32 s6, s0, s15 // (op 1105) 

v_and_b32 v11, 63, v1 // (op 1105) 
v_lshrrev_b32 v12, 4, v11 // (op 1105) 

s_mul_i32 s15, s16, s50 // (op 1105) 
s_add_u32 s15, s15, s6 // (op 1105) 

s_add_u32 s6, s15, s9 // (op 1105) 

s_mul_hi_u32 s15, s6, s13 // (op 1105) 
s_sub_u32 s7, s6, s15 // (op 1105) 
s_lshr_b32 s8, s7, 1 // (op 1105) 

s_add_u32 s7, s8, s15 // (op 1105) 


s_lshr_b32 s8, s7, s11 // (op 1105) 

s_and_b32 s7, s11, 2147483648 // (op 1105) 
s_cmp_eq_i32 s7, 0 // (op 1105) 
s_mov_b32 s15, scc // (op 1105) 

s_cmp_eq_i32 s15, 0 // (op 1105) 
s_cselect_b32 s7, s6, s8 // (op 1105) 


s_mul_hi_u32 s8, s7, s12 // (op 1105) 
s_sub_u32 s15, s7, s8 // (op 1105) 
s_lshr_b32 s42, s15, 1 // (op 1105) 

s_add_u32 s15, s42, s8 // (op 1105) 


s_lshr_b32 s8, s15, s10 // (op 1105) 

s_and_b32 s15, s10, 2147483648 // (op 1105) 
s_cmp_eq_i32 s15, 0 // (op 1105) 
s_mov_b32 s42, scc // (op 1105) 

s_cmp_eq_i32 s42, 0 // (op 1105) 
s_cselect_b32 s15, s7, s8 // (op 1105) 


s_mul_i32 s8, s7, s18 // (op 1105) 


v_lshl_add_u32 v11, s15, 4, v13 // (op 1105) 


s_sub_u32 s7, s6, s8 // (op 1105) 


v_lshl_add_u32 v13, s7, 2, v12 // (op 1105) 


v_mul_lo_u32 v12, v11, s34 // (op 1105) 
v_add_u32 v12, v12, v13 // (op 1105) 


s_mul_i32 s19, s34, s29 // (op 1107) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1107) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1107) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1107) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1107) most significant: sum
s_add_u32 s19, s19, s6 // (op 1107) most significant: sum


s_mov_b32 s68, s36 // (op 1107) reg expression
s_mov_b32 s69, s37 // (op 1107) reg expression
s_mov_b32 s70, s18 // (op 1107) reg expression
s_mov_b32 s71, 131072 // (op 1107) call()

v_lshrrev_b32 v11, 2, v1 // (op 1233) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1233) 
v_and_b32 v13, -4, v1 // (op 1233) 
s_and_b32 s15, s18, 4294967294 // (op 1233) 
v_sub_u32 v14, v1, v13 // (op 1233) 

s_add_u32 s6, s0, s15 // (op 1233) 

s_mul_i32 s15, s16, s50 // (op 1233) 
s_add_u32 s15, s15, s6 // (op 1233) 

s_add_u32 s6, s15, s9 // (op 1233) 

s_mul_hi_u32 s15, s6, s13 // (op 1233) 
s_sub_u32 s7, s6, s15 // (op 1233) 
s_lshr_b32 s8, s7, 1 // (op 1233) 

s_add_u32 s7, s8, s15 // (op 1233) 


s_lshr_b32 s8, s7, s11 // (op 1233) 

s_and_b32 s7, s11, 2147483648 // (op 1233) 
s_cmp_eq_i32 s7, 0 // (op 1233) 
s_mov_b32 s15, scc // (op 1233) 

s_cmp_eq_i32 s15, 0 // (op 1233) 
s_cselect_b32 s7, s6, s8 // (op 1233) 


s_mul_hi_u32 s8, s7, s12 // (op 1233) 
s_sub_u32 s15, s7, s8 // (op 1233) 
s_lshr_b32 s42, s15, 1 // (op 1233) 

s_add_u32 s15, s42, s8 // (op 1233) 


s_lshr_b32 s8, s15, s10 // (op 1233) 

s_and_b32 s15, s10, 2147483648 // (op 1233) 
s_cmp_eq_i32 s15, 0 // (op 1233) 
s_mov_b32 s42, scc // (op 1233) 

s_cmp_eq_i32 s42, 0 // (op 1233) 
s_cselect_b32 s15, s7, s8 // (op 1233) 


s_mul_i32 s8, s15, s14 // (op 1233) 

s_sub_u32 s15, s7, s8 // (op 1233) 

v_lshl_add_u32 v13, s15, 5, v11 // (op 1233) 


s_mul_i32 s8, s7, s18 // (op 1233) 


v_mul_lo_u32 v11, v13, s38 // (op 1233) 

s_sub_u32 s7, s6, s8 // (op 1233) 


v_lshl_add_u32 v13, s7, 2, v14 // (op 1233) 


v_add_u32 v14, v13, v11 // (op 1233) 


s_mul_i32 s19, s38, s23 // (op 1235) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 1235) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 1235) least significant
s_mul_hi_u32 s6, s38, s22 // (op 1235) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1235) most significant: sum
s_add_u32 s19, s19, s6 // (op 1235) most significant: sum


s_mov_b32 s72, s40 // (op 1235) reg expression
s_mov_b32 s73, s41 // (op 1235) reg expression
s_mov_b32 s74, s18 // (op 1235) reg expression
s_mov_b32 s75, 131072 // (op 1235) call()

v_and_b32 v11, -4, v1 // (op 1529) 
v_sub_u32 v13, v1, v11 // (op 1529) 

v_and_b32 v11, 4294967292, v1 // (op 1529) 
v_add_u32 v15, v11, v13 // (op 1529) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 405) 
s_and_b32 s15, s18, 4294967294 // (op 405) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 404) 
s_mov_b32 s6, s18 // (op 404) convert

s_cmp_lt_u32 s15, s6 // (op 404) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_404 // (op 404) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_404 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_404:


s_add_u32 s15, s15, 1 // (op 406) 
v_mov_b32 v0, v14 // (op 409) 

buffer_load_ubyte v9, v0, s[72:75], 0 offen offset: 0 // (op 409) Load value

v_add_u32 v14, 4, v14 // (op 1225) 


s_barrier  // (op 1718) 
v_mov_b32 v0, v5 // (op 415) 




buffer_load_dwordx4 v[16:19], v0, s[60:63], 0 offen // (op 415) Load value

v_add_u32 v5, 64, v5 // (op 831) 
s_waitcnt vmcnt(1)
ds_write_b8 v15, v9 // (op 410) Store local data 



v_mov_b32 v0, v10 // (op 417) 

buffer_load_dwordx4 v[20:23], v0, s[64:67], 0 offen // (op 417) Load value
buffer_load_dwordx4 v[24:27], v0, s[64:67], 0 offen offset:64 // (op 417) Load value

v_add_u32 v10, 128, v10 // (op 969) 


v_mov_b32 v0, v12 // (op 419) 

buffer_load_ubyte v9, v0, s[68:71], 0 offen offset: 0 // (op 419) Load value

v_add_u32 v12, 4, v12 // (op 1097) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1721) 



ds_read_u8 v0, v3 // (op 412) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:19], v[20:27], a[0:3], v9, v0 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 413) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 404) 
s_mov_b32 s6, s18 // (op 404) convert

s_cmp_lt_u32 s15, s6 // (op 404) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_404 // (op 404) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_404 if true)
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_404:






s_barrier  // (op 1817) 
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 189) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_189 // (op 189) Condition: False, jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_189
s_nop 0
v_accvgpr_read v10, a0 // (op 197) Wait state hazard: XDL Write Hazard
v_accvgpr_read v11, a1 // (op 197) 
v_accvgpr_read v12, a2 // (op 197) 
v_accvgpr_read v13, a3 // (op 197) 
v_mul_f32 v14, v7, v10 // (op 197) 
v_mul_f32 v15, v7, v11 // (op 197) 
v_mul_f32 v16, v7, v12 // (op 197) 
v_mul_f32 v17, v7, v13 // (op 197) 

v_lshrrev_b32 v0, 7, v1 // (op 1482) 
s_mul_i32 s15, s16, s50 // (op 1482) 
s_add_u32 s15, s15, s0 // (op 1482) 
v_and_b32 v5, 63, v1 // (op 1482) 
s_add_u32 s6, s15, s9 // (op 1482) 

v_lshrrev_b32 v9, 4, v5 // (op 1482) 
s_mul_hi_u32 s15, s6, s13 // (op 1482) 
v_lshlrev_b32 v10, 2, v9 // (op 1482) 

s_sub_u32 s7, s6, s15 // (op 1482) 
v_lshl_add_u32 v9, v0, 4, v10 // (op 1482) 


s_lshr_b32 s8, s7, 1 // (op 1482) 

v_lshrrev_b32 v0, 2, v5 // (op 1482) 

s_add_u32 s7, s8, s15 // (op 1482) 


v_and_b32 v5, 3, v0 // (op 1482) 

s_lshr_b32 s8, s7, s11 // (op 1482) 

v_and_b32 v0, 3, v1 // (op 1482) 
s_and_b32 s7, s11, 2147483648 // (op 1482) 
v_lshl_add_u32 v10, v5, 2, v0 // (op 1482) 


s_cmp_eq_i32 s7, 0 // (op 1482) 
s_mov_b32 s15, scc // (op 1482) 

v_lshrrev_b32 v0, 6, v1 // (op 1482) 
s_cmp_eq_i32 s15, 0 // (op 1482) 
s_cselect_b32 s7, s6, s8 // (op 1482) 



v_and_b32 v5, 1, v0 // (op 1482) 

s_mul_hi_u32 s6, s7, s12 // (op 1482) 
v_lshl_add_u32 v0, v5, 4, v10 // (op 1482) 


s_sub_u32 s8, s7, s6 // (op 1482) 
s_lshr_b32 s15, s8, 1 // (op 1482) 

s_add_u32 s8, s15, s6 // (op 1482) 


s_lshr_b32 s6, s8, s10 // (op 1482) 

s_and_b32 s8, s10, 2147483648 // (op 1482) 
s_cmp_eq_i32 s8, 0 // (op 1482) 
s_mov_b32 s15, scc // (op 1482) 

s_cmp_eq_i32 s15, 0 // (op 1482) 
s_cselect_b32 s8, s7, s6 // (op 1482) 


s_mul_i32 s6, s8, s14 // (op 1482) 
v_lshl_add_u32 v5, s8, 4, v9 // (op 1482) 


s_sub_u32 s8, s7, s6 // (op 1482) 


v_lshl_add_u32 v9, s8, 5, v0 // (op 1482) 


v_mul_lo_u32 v0, v9, s46 // (op 1482) 

v_add_lshl_u32 v9, v5, v0, 2 // (op 1482) 


s_mul_i32 s19, s46, s23 // (op 1484) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1484) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1484) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1484) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1484) most significant: sum
s_add_u32 s19, s19, s6 // (op 1484) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1484) 

s_mov_b32 s76, s48 // (op 1484) reg expression
s_mov_b32 s77, s49 // (op 1484) reg expression
s_mov_b32 s78, s6 // (op 1484) reg expression
s_mov_b32 s79, 131072 // (op 1484) call()




buffer_store_dwordx4 v[14:17], v9, s[76:79], 0 offen // (op 199) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_189 // (op 189) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_189
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_189:



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


v_lshrrev_b32 v0, 7, v1 // (op 1499) 
s_mul_i32 s15, s16, s50 // (op 1499) 
s_add_u32 s15, s15, s0 // (op 1499) 
v_and_b32 v5, 63, v1 // (op 1499) 
s_add_u32 s6, s15, s9 // (op 1499) 

v_lshrrev_b32 v9, 4, v5 // (op 1499) 
s_mul_hi_u32 s15, s6, s13 // (op 1499) 
v_lshlrev_b32 v14, 2, v9 // (op 1499) 

s_sub_u32 s7, s6, s15 // (op 1499) 
v_lshl_add_u32 v9, v0, 4, v14 // (op 1499) 


s_lshr_b32 s8, s7, 1 // (op 1499) 

v_lshrrev_b32 v0, 2, v5 // (op 1499) 

s_add_u32 s7, s8, s15 // (op 1499) 


v_and_b32 v5, 3, v0 // (op 1499) 

s_lshr_b32 s8, s7, s11 // (op 1499) 

v_and_b32 v0, 3, v1 // (op 1499) 
s_and_b32 s7, s11, 2147483648 // (op 1499) 
v_lshl_add_u32 v14, v5, 2, v0 // (op 1499) 


s_cmp_eq_i32 s7, 0 // (op 1499) 
s_mov_b32 s15, scc // (op 1499) 

v_lshrrev_b32 v0, 6, v1 // (op 1499) 
s_cmp_eq_i32 s15, 0 // (op 1499) 
s_cselect_b32 s7, s6, s8 // (op 1499) 



v_and_b32 v5, 1, v0 // (op 1499) 

s_mul_hi_u32 s6, s7, s12 // (op 1499) 
v_lshl_add_u32 v0, v5, 4, v14 // (op 1499) 


s_sub_u32 s8, s7, s6 // (op 1499) 
s_lshr_b32 s15, s8, 1 // (op 1499) 

s_add_u32 s8, s15, s6 // (op 1499) 


s_lshr_b32 s6, s8, s10 // (op 1499) 

s_and_b32 s8, s10, 2147483648 // (op 1499) 
s_cmp_eq_i32 s8, 0 // (op 1499) 
s_mov_b32 s15, scc // (op 1499) 

s_cmp_eq_i32 s15, 0 // (op 1499) 
s_cselect_b32 s8, s7, s6 // (op 1499) 


s_mul_i32 s6, s8, s14 // (op 1499) 
v_lshl_add_u32 v5, s8, 4, v9 // (op 1499) 


s_sub_u32 s8, s7, s6 // (op 1499) 


v_lshl_add_u32 v9, s8, 5, v0 // (op 1499) 


v_mul_lo_u32 v0, v9, s46 // (op 1499) 

v_add_lshl_u32 v9, v5, v0, 2 // (op 1499) 


s_mul_i32 s19, s46, s23 // (op 1501) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1501) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1501) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1501) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1501) most significant: sum
s_add_u32 s19, s19, s6 // (op 1501) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1501) 

s_mov_b32 s76, s48 // (op 1501) reg expression
s_mov_b32 s77, s49 // (op 1501) reg expression
s_mov_b32 s78, s6 // (op 1501) reg expression
s_mov_b32 s79, 131072 // (op 1501) call()




buffer_store_dwordx4 v[10:13], v9, s[76:79], 0 offen // (op 211) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_189:

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
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_DPStreamTileLoop_396 // (op 396) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_DPStreamTileLoop_396 if true)
RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_396:




































s_endpgm  // End of RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2
.LRR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_end:
.size RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2, .LRR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_end-RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2
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
    macrotile: [16, 32, 128]
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
  - .name: RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2
    .symbol: RR_GEMM_TN_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2.kd
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

