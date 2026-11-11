// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2,@function
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 1607) 
v_and_b32 v3, 15, v1 // (op 1607) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1607) 


v_and_b32 v0, 48, v1 // (op 1607) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 1607) 


v_lshrrev_b32 v0, 6, v1 // (op 1672) 
v_and_b32 v4, 1, v0 // (op 1672) 

v_and_b32 v0, 15, v1 // (op 1672) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 1672) 


v_and_b32 v0, 48, v1 // (op 1672) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 1672) 


v_lshrrev_b32 v0, 7, v1 // (op 1807) 
v_and_b32 v5, 15, v1 // (op 1807) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 1807) 


v_and_b32 v0, 48, v1 // (op 1807) 
v_lshl_add_u32 v5, v6, 7, v0 // (op 1807) 


v_lshrrev_b32 v0, 6, v1 // (op 1872) 
v_and_b32 v6, 1, v0 // (op 1872) 

v_and_b32 v0, 15, v1 // (op 1872) 
v_lshl_add_u32 v7, v6, 4, v0 // (op 1872) 


v_and_b32 v0, 48, v1 // (op 1872) 
v_lshl_add_u32 v6, v7, 7, v0 // (op 1872) 


v_lshrrev_b32 v0, 7, v1 // (op 1958) 
v_and_b32 v7, 15, v1 // (op 1958) 
v_lshl_add_u32 v8, v0, 4, v7 // (op 1958) 


v_and_b32 v0, 48, v1 // (op 1958) 
v_lshl_add_u32 v7, v8, 7, v0 // (op 1958) 


v_lshrrev_b32 v0, 6, v1 // (op 1997) 
v_and_b32 v8, 1, v0 // (op 1997) 

v_and_b32 v0, 15, v1 // (op 1997) 
v_lshl_add_u32 v9, v8, 4, v0 // (op 1997) 


v_and_b32 v0, 48, v1 // (op 1997) 
v_lshl_add_u32 v8, v9, 7, v0 // (op 1997) 


v_lshrrev_b32 v0, 7, v1 // (op 2036) 
v_and_b32 v9, 15, v1 // (op 2036) 
v_lshl_add_u32 v10, v0, 4, v9 // (op 2036) 


v_and_b32 v0, 48, v1 // (op 2036) 
v_lshl_add_u32 v9, v10, 7, v0 // (op 2036) 


v_lshrrev_b32 v0, 6, v1 // (op 2075) 
v_and_b32 v10, 1, v0 // (op 2075) 

v_and_b32 v0, 15, v1 // (op 2075) 
v_lshl_add_u32 v11, v10, 4, v0 // (op 2075) 


v_and_b32 v0, 48, v1 // (op 2075) 
v_lshl_add_u32 v10, v11, 7, v0 // (op 2075) 


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
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_263 // (op 263) Condition: Top (jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_263 if false)
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_SKStreamTileLoop_263:
v_lshrrev_b32 v12, 7, v1 // (op 1533) 
s_mul_i32 s52, s16, s8 // (op 1533) 
s_add_u32 s52, s52, s0 // (op 1533) 
v_and_b32 v13, 63, v1 // (op 1533) 
s_mul_hi_u32 s53, s52, s13 // (op 1533) 
v_lshrrev_b32 v14, 4, v13 // (op 1533) 
s_sub_u32 s54, s52, s53 // (op 1533) 
v_lshlrev_b32 v15, 2, v14 // (op 1533) 

v_lshl_add_u32 v14, v12, 4, v15 // (op 1533) 


s_lshr_b32 s55, s54, 1 // (op 1533) 

s_add_u32 s54, s55, s53 // (op 1533) 


v_lshrrev_b32 v12, 2, v13 // (op 1533) 

s_lshr_b32 s53, s54, s11 // (op 1533) 

v_and_b32 v13, 3, v12 // (op 1533) 

s_and_b32 s54, s11, 2147483648 // (op 1533) 
v_and_b32 v12, 3, v1 // (op 1533) 
s_cmp_eq_i32 s54, 0 // (op 1533) 
s_mov_b32 s55, scc // (op 1533) 

v_lshl_add_u32 v15, v13, 2, v12 // (op 1533) 


s_cmp_eq_i32 s55, 0 // (op 1533) 
s_cselect_b32 s54, s52, s53 // (op 1533) 



v_lshrrev_b32 v12, 6, v1 // (op 1533) 
s_mul_hi_u32 s52, s54, s12 // (op 1533) 
v_and_b32 v13, 1, v12 // (op 1533) 

s_sub_u32 s53, s54, s52 // (op 1533) 
v_lshl_add_u32 v12, v13, 4, v15 // (op 1533) 


s_lshr_b32 s55, s53, 1 // (op 1533) 

s_add_u32 s53, s55, s52 // (op 1533) 


s_lshr_b32 s52, s53, s10 // (op 1533) 

s_and_b32 s53, s10, 2147483648 // (op 1533) 
s_cmp_eq_i32 s53, 0 // (op 1533) 
s_mov_b32 s55, scc // (op 1533) 

s_cmp_eq_i32 s55, 0 // (op 1533) 
s_cselect_b32 s53, s54, s52 // (op 1533) 


s_mul_i32 s52, s53, s14 // (op 1533) 
v_lshl_add_u32 v13, s53, 5, v14 // (op 1533) 


s_sub_u32 s53, s54, s52 // (op 1533) 


v_lshl_add_u32 v14, s53, 5, v12 // (op 1533) 


v_mul_lo_u32 v12, v14, s4 // (op 1533) 

v_add_lshl_u32 v14, v13, v12, 1 // (op 1533) 


s_mul_i32 s53, s4, s23 // (op 1535) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 1535) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 1535) least significant
s_mul_hi_u32 s55, s4, s22 // (op 1535) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1535) most significant: sum
s_add_u32 s53, s53, s55 // (op 1535) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 1535) 

s_mov_b32 s56, s20 // (op 1535) reg expression
s_mov_b32 s57, s21 // (op 1535) reg expression
s_mov_b32 s58, s54 // (op 1535) reg expression
s_mov_b32 s59, 131072 // (op 1535) call()

v_lshrrev_b32 v12, 7, v1 // (op 1619) 
v_and_b32 v13, 15, v1 // (op 1619) 
v_lshl_add_u32 v15, v12, 4, v13 // (op 1619) 


v_and_b32 v12, 48, v1 // (op 1619) 
v_lshl_add_u32 v13, v15, 7, v12 // (op 1619) 


v_lshrrev_b32 v12, 6, v1 // (op 1684) 
v_and_b32 v15, 1, v12 // (op 1684) 

v_and_b32 v12, 15, v1 // (op 1684) 
v_lshl_add_u32 v16, v15, 4, v12 // (op 1684) 


v_and_b32 v12, 48, v1 // (op 1684) 
v_lshl_add_u32 v15, v16, 7, v12 // (op 1684) 


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
v_lshrrev_b32 v12, 3, v1 // (op 1065) 
s_mul_i32 s52, s16, s8 // (op 1065) 
s_add_u32 s52, s52, s0 // (op 1065) 
v_and_b32 v16, -8, v1 // (op 1065) 
s_mul_hi_u32 s53, s52, s13 // (op 1065) 
v_sub_u32 v17, v1, v16 // (op 1065) 

s_sub_u32 s54, s52, s53 // (op 1065) 
v_lshlrev_b32 v16, 4, v17 // (op 1065) 

s_lshr_b32 s55, s54, 1 // (op 1065) 

s_add_u32 s54, s55, s53 // (op 1065) 


s_lshr_b32 s53, s54, s11 // (op 1065) 

s_and_b32 s54, s11, 2147483648 // (op 1065) 
s_cmp_eq_i32 s54, 0 // (op 1065) 
s_mov_b32 s55, scc // (op 1065) 

s_cmp_eq_i32 s55, 0 // (op 1065) 
s_cselect_b32 s54, s52, s53 // (op 1065) 


s_mul_hi_u32 s53, s54, s12 // (op 1065) 
s_sub_u32 s55, s54, s53 // (op 1065) 
s_lshr_b32 s60, s55, 1 // (op 1065) 

s_add_u32 s55, s60, s53 // (op 1065) 


s_lshr_b32 s53, s55, s10 // (op 1065) 

s_and_b32 s55, s10, 2147483648 // (op 1065) 
s_cmp_eq_i32 s55, 0 // (op 1065) 
s_mov_b32 s60, scc // (op 1065) 

s_cmp_eq_i32 s60, 0 // (op 1065) 
s_cselect_b32 s55, s54, s53 // (op 1065) 


v_lshl_add_u32 v17, s55, 5, v12 // (op 1065) 


s_ashr_i64 s[60:61], s[2:3], 7 // (op 1065) 
s_mul_i32 s53, s54, s60 // (op 1065) 


s_sub_u32 s54, s52, s53 // (op 1065) 


v_lshl_add_u32 v12, s54, 7, v16 // (op 1065) 


v_mov_b32 v19, 0 // (op 1065) convert
v_mov_b32 v18, v12 // (op 1065) convert

v_mul_lo_u32 v21, v17, s25 // (op 1065) most significant: low of low * high
v_mul_hi_u32 v12, v17, s24 // (op 1065) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 1065) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 1065) most significant: sum

v_add_co_u32 v20, s[52:53], v20, v18 // (op 1065) least significant half
v_addc_co_u32 v21, s[52:53], v21, v19, s[52:53] // (op 1065) most significant half



s_mul_i32 s53, s24, s29 // (op 1067) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 1067) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 1067) least significant
s_mul_hi_u32 s55, s24, s28 // (op 1067) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1067) most significant: sum
s_add_u32 s53, s53, s55 // (op 1067) most significant: sum


s_mov_b32 s60, s26 // (op 1067) reg expression
s_mov_b32 s61, s27 // (op 1067) reg expression
s_mov_b32 s62, s52 // (op 1067) reg expression
s_mov_b32 s63, 131072 // (op 1067) call()

v_lshrrev_b32 v12, 3, v1 // (op 1084) 
s_mul_i32 s52, s16, s8 // (op 1084) 
s_add_u32 s52, s52, s0 // (op 1084) 
v_and_b32 v16, -8, v1 // (op 1084) 
s_mul_hi_u32 s53, s52, s13 // (op 1084) 
v_sub_u32 v17, v1, v16 // (op 1084) 

s_sub_u32 s54, s52, s53 // (op 1084) 
v_lshlrev_b32 v16, 4, v17 // (op 1084) 

s_lshr_b32 s55, s54, 1 // (op 1084) 

s_add_u32 s54, s55, s53 // (op 1084) 


s_lshr_b32 s53, s54, s11 // (op 1084) 

s_and_b32 s54, s11, 2147483648 // (op 1084) 
s_cmp_eq_i32 s54, 0 // (op 1084) 
s_mov_b32 s55, scc // (op 1084) 

s_cmp_eq_i32 s55, 0 // (op 1084) 
s_cselect_b32 s54, s52, s53 // (op 1084) 


s_mul_hi_u32 s53, s54, s12 // (op 1084) 
s_sub_u32 s55, s54, s53 // (op 1084) 
s_lshr_b32 s64, s55, 1 // (op 1084) 

s_add_u32 s55, s64, s53 // (op 1084) 


s_lshr_b32 s53, s55, s10 // (op 1084) 

s_and_b32 s55, s10, 2147483648 // (op 1084) 
s_cmp_eq_i32 s55, 0 // (op 1084) 
s_mov_b32 s64, scc // (op 1084) 

s_cmp_eq_i32 s64, 0 // (op 1084) 
s_cselect_b32 s55, s54, s53 // (op 1084) 


v_lshl_add_u32 v17, s55, 5, v12 // (op 1084) 


s_ashr_i64 s[64:65], s[2:3], 7 // (op 1084) 
s_mul_i32 s53, s54, s64 // (op 1084) 


s_sub_u32 s54, s52, s53 // (op 1084) 


v_lshl_add_u32 v12, s54, 7, v16 // (op 1084) 


v_mov_b32 v19, 0 // (op 1084) convert
v_mov_b32 v18, v12 // (op 1084) convert

v_mul_lo_u32 v23, v17, s25 // (op 1084) most significant: low of low * high
v_mul_hi_u32 v12, v17, s24 // (op 1084) most significant: high of low * low
v_mul_lo_u32 v22, v17, s24 // (op 1084) least significant: low of low * low
v_add_u32 v23, v23, v12 // (op 1084) most significant: sum

v_add_co_u32 v22, s[52:53], v22, v18 // (op 1084) least significant half
v_addc_co_u32 v23, s[52:53], v23, v19, s[52:53] // (op 1084) most significant half



s_mul_i32 s53, s24, s29 // (op 1086) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 1086) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 1086) least significant
s_mul_hi_u32 s55, s24, s28 // (op 1086) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1086) most significant: sum
s_add_u32 s53, s53, s55 // (op 1086) most significant: sum


s_mov_b32 s60, s26 // (op 1086) reg expression
s_mov_b32 s61, s27 // (op 1086) reg expression
s_mov_b32 s62, s52 // (op 1086) reg expression
s_mov_b32 s63, 131072 // (op 1086) call()

v_lshrrev_b32 v12, 3, v1 // (op 1196) 
s_mul_i32 s52, s16, s8 // (op 1196) 
s_add_u32 s52, s52, s0 // (op 1196) 
v_and_b32 v16, -8, v1 // (op 1196) 
s_mul_hi_u32 s53, s52, s13 // (op 1196) 
v_sub_u32 v17, v1, v16 // (op 1196) 

s_sub_u32 s54, s52, s53 // (op 1196) 
v_lshlrev_b32 v16, 4, v17 // (op 1196) 

s_lshr_b32 s55, s54, 1 // (op 1196) 

s_add_u32 s54, s55, s53 // (op 1196) 


s_lshr_b32 s53, s54, s11 // (op 1196) 

s_and_b32 s54, s11, 2147483648 // (op 1196) 
s_cmp_eq_i32 s54, 0 // (op 1196) 
s_mov_b32 s55, scc // (op 1196) 

s_cmp_eq_i32 s55, 0 // (op 1196) 
s_cselect_b32 s54, s52, s53 // (op 1196) 


s_mul_hi_u32 s53, s54, s12 // (op 1196) 
s_sub_u32 s55, s54, s53 // (op 1196) 
s_lshr_b32 s64, s55, 1 // (op 1196) 

s_add_u32 s55, s64, s53 // (op 1196) 


s_lshr_b32 s53, s55, s10 // (op 1196) 

s_and_b32 s55, s10, 2147483648 // (op 1196) 
s_cmp_eq_i32 s55, 0 // (op 1196) 
s_mov_b32 s64, scc // (op 1196) 

s_cmp_eq_i32 s64, 0 // (op 1196) 
s_cselect_b32 s55, s54, s53 // (op 1196) 


s_mul_i32 s53, s55, s14 // (op 1196) 

s_sub_u32 s55, s54, s53 // (op 1196) 

v_lshl_add_u32 v17, s55, 5, v12 // (op 1196) 


v_mul_lo_u32 v19, v17, s31 // (op 1196) most significant: low of low * high
v_mul_hi_u32 v12, v17, s30 // (op 1196) most significant: high of low * low
v_mul_lo_u32 v18, v17, s30 // (op 1196) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 1196) most significant: sum


s_ashr_i64 s[64:65], s[2:3], 7 // (op 1196) 
s_mul_i32 s53, s54, s64 // (op 1196) 


s_sub_u32 s54, s52, s53 // (op 1196) 


v_lshl_add_u32 v12, s54, 7, v16 // (op 1196) 


v_mov_b32 v17, 0 // (op 1196) convert
v_mov_b32 v16, v12 // (op 1196) convert

v_add_co_u32 v24, s[52:53], v16, v18 // (op 1196) least significant half
v_addc_co_u32 v25, s[52:53], v17, v19, s[52:53] // (op 1196) most significant half



s_mul_i32 s53, s30, s23 // (op 1198) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 1198) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 1198) least significant
s_mul_hi_u32 s55, s30, s22 // (op 1198) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1198) most significant: sum
s_add_u32 s53, s53, s55 // (op 1198) most significant: sum


s_mov_b32 s64, s32 // (op 1198) reg expression
s_mov_b32 s65, s33 // (op 1198) reg expression
s_mov_b32 s66, s52 // (op 1198) reg expression
s_mov_b32 s67, 131072 // (op 1198) call()

v_lshrrev_b32 v12, 3, v1 // (op 1215) 
s_mul_i32 s52, s16, s8 // (op 1215) 
s_add_u32 s52, s52, s0 // (op 1215) 
v_and_b32 v16, -8, v1 // (op 1215) 
s_mul_hi_u32 s53, s52, s13 // (op 1215) 
v_sub_u32 v17, v1, v16 // (op 1215) 

s_sub_u32 s54, s52, s53 // (op 1215) 
v_lshlrev_b32 v16, 4, v17 // (op 1215) 

s_lshr_b32 s55, s54, 1 // (op 1215) 

s_add_u32 s54, s55, s53 // (op 1215) 


s_lshr_b32 s53, s54, s11 // (op 1215) 

s_and_b32 s54, s11, 2147483648 // (op 1215) 
s_cmp_eq_i32 s54, 0 // (op 1215) 
s_mov_b32 s55, scc // (op 1215) 

s_cmp_eq_i32 s55, 0 // (op 1215) 
s_cselect_b32 s54, s52, s53 // (op 1215) 


s_mul_hi_u32 s53, s54, s12 // (op 1215) 
s_sub_u32 s55, s54, s53 // (op 1215) 
s_lshr_b32 s68, s55, 1 // (op 1215) 

s_add_u32 s55, s68, s53 // (op 1215) 


s_lshr_b32 s53, s55, s10 // (op 1215) 

s_and_b32 s55, s10, 2147483648 // (op 1215) 
s_cmp_eq_i32 s55, 0 // (op 1215) 
s_mov_b32 s68, scc // (op 1215) 

s_cmp_eq_i32 s68, 0 // (op 1215) 
s_cselect_b32 s55, s54, s53 // (op 1215) 


s_mul_i32 s53, s55, s14 // (op 1215) 

s_sub_u32 s55, s54, s53 // (op 1215) 

v_lshl_add_u32 v17, s55, 5, v12 // (op 1215) 


v_mul_lo_u32 v19, v17, s31 // (op 1215) most significant: low of low * high
v_mul_hi_u32 v12, v17, s30 // (op 1215) most significant: high of low * low
v_mul_lo_u32 v18, v17, s30 // (op 1215) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 1215) most significant: sum


s_ashr_i64 s[68:69], s[2:3], 7 // (op 1215) 
s_mul_i32 s53, s54, s68 // (op 1215) 


s_sub_u32 s54, s52, s53 // (op 1215) 


v_lshl_add_u32 v12, s54, 7, v16 // (op 1215) 


v_mov_b32 v17, 0 // (op 1215) convert
v_mov_b32 v16, v12 // (op 1215) convert

v_add_co_u32 v26, s[52:53], v16, v18 // (op 1215) least significant half
v_addc_co_u32 v27, s[52:53], v17, v19, s[52:53] // (op 1215) most significant half



s_mul_i32 s53, s30, s23 // (op 1217) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 1217) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 1217) least significant
s_mul_hi_u32 s55, s30, s22 // (op 1217) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1217) most significant: sum
s_add_u32 s53, s53, s55 // (op 1217) most significant: sum


s_mov_b32 s64, s32 // (op 1217) reg expression
s_mov_b32 s65, s33 // (op 1217) reg expression
s_mov_b32 s66, s52 // (op 1217) reg expression
s_mov_b32 s67, 131072 // (op 1217) call()

v_lshrrev_b32 v12, 7, v1 // (op 1324) 
s_mul_i32 s52, s16, s8 // (op 1324) 
s_add_u32 s52, s52, s0 // (op 1324) 
v_and_b32 v16, 15, v1 // (op 1324) 
s_mul_hi_u32 s53, s52, s13 // (op 1324) 
v_lshl_add_u32 v17, v12, 4, v16 // (op 1324) 


s_sub_u32 s54, s52, s53 // (op 1324) 
v_and_b32 v12, 63, v1 // (op 1324) 
s_lshr_b32 s55, s54, 1 // (op 1324) 

v_lshrrev_b32 v16, 4, v12 // (op 1324) 

s_add_u32 s54, s55, s53 // (op 1324) 


s_lshr_b32 s53, s54, s11 // (op 1324) 

s_and_b32 s54, s11, 2147483648 // (op 1324) 
s_cmp_eq_i32 s54, 0 // (op 1324) 
s_mov_b32 s55, scc // (op 1324) 

s_cmp_eq_i32 s55, 0 // (op 1324) 
s_cselect_b32 s54, s52, s53 // (op 1324) 


s_mul_hi_u32 s53, s54, s12 // (op 1324) 
s_sub_u32 s55, s54, s53 // (op 1324) 
s_lshr_b32 s68, s55, 1 // (op 1324) 

s_add_u32 s55, s68, s53 // (op 1324) 


s_lshr_b32 s53, s55, s10 // (op 1324) 

s_and_b32 s55, s10, 2147483648 // (op 1324) 
s_cmp_eq_i32 s55, 0 // (op 1324) 
s_mov_b32 s68, scc // (op 1324) 

s_cmp_eq_i32 s68, 0 // (op 1324) 
s_cselect_b32 s55, s54, s53 // (op 1324) 


v_lshl_add_u32 v12, s55, 5, v17 // (op 1324) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 1324) 
s_mul_i32 s53, s54, s68 // (op 1324) 


s_sub_u32 s54, s52, s53 // (op 1324) 


v_lshl_add_u32 v17, s54, 2, v16 // (op 1324) 


v_mul_lo_u32 v16, v12, s34 // (op 1324) 
v_add_u32 v16, v16, v17 // (op 1324) 


s_mul_i32 s53, s34, s29 // (op 1326) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 1326) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 1326) least significant
s_mul_hi_u32 s55, s34, s28 // (op 1326) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1326) most significant: sum
s_add_u32 s53, s53, s55 // (op 1326) most significant: sum


s_mov_b32 s68, s36 // (op 1326) reg expression
s_mov_b32 s69, s37 // (op 1326) reg expression
s_mov_b32 s70, s52 // (op 1326) reg expression
s_mov_b32 s71, 131072 // (op 1326) call()

v_lshrrev_b32 v12, 7, v1 // (op 1343) 
s_mul_i32 s52, s16, s8 // (op 1343) 
s_add_u32 s52, s52, s0 // (op 1343) 
v_and_b32 v17, 15, v1 // (op 1343) 
s_mul_hi_u32 s53, s52, s13 // (op 1343) 
v_lshl_add_u32 v18, v12, 4, v17 // (op 1343) 


s_sub_u32 s54, s52, s53 // (op 1343) 
v_and_b32 v12, 63, v1 // (op 1343) 
s_lshr_b32 s55, s54, 1 // (op 1343) 

v_lshrrev_b32 v17, 4, v12 // (op 1343) 

s_add_u32 s54, s55, s53 // (op 1343) 


s_lshr_b32 s53, s54, s11 // (op 1343) 

s_and_b32 s54, s11, 2147483648 // (op 1343) 
s_cmp_eq_i32 s54, 0 // (op 1343) 
s_mov_b32 s55, scc // (op 1343) 

s_cmp_eq_i32 s55, 0 // (op 1343) 
s_cselect_b32 s54, s52, s53 // (op 1343) 


s_mul_hi_u32 s53, s54, s12 // (op 1343) 
s_sub_u32 s55, s54, s53 // (op 1343) 
s_lshr_b32 s72, s55, 1 // (op 1343) 

s_add_u32 s55, s72, s53 // (op 1343) 


s_lshr_b32 s53, s55, s10 // (op 1343) 

s_and_b32 s55, s10, 2147483648 // (op 1343) 
s_cmp_eq_i32 s55, 0 // (op 1343) 
s_mov_b32 s72, scc // (op 1343) 

s_cmp_eq_i32 s72, 0 // (op 1343) 
s_cselect_b32 s55, s54, s53 // (op 1343) 


v_lshl_add_u32 v12, s55, 5, v18 // (op 1343) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 1343) 
s_mul_i32 s53, s54, s72 // (op 1343) 


s_sub_u32 s54, s52, s53 // (op 1343) 


v_lshl_add_u32 v18, s54, 2, v17 // (op 1343) 


v_mul_lo_u32 v17, v12, s34 // (op 1343) 
v_add_u32 v17, v17, v18 // (op 1343) 


s_mul_i32 s53, s34, s29 // (op 1345) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 1345) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 1345) least significant
s_mul_hi_u32 s55, s34, s28 // (op 1345) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1345) most significant: sum
s_add_u32 s53, s53, s55 // (op 1345) most significant: sum


s_mov_b32 s68, s36 // (op 1345) reg expression
s_mov_b32 s69, s37 // (op 1345) reg expression
s_mov_b32 s70, s52 // (op 1345) reg expression
s_mov_b32 s71, 131072 // (op 1345) call()

v_lshrrev_b32 v12, 6, v1 // (op 1452) 
s_mul_i32 s52, s16, s8 // (op 1452) 
s_add_u32 s52, s52, s0 // (op 1452) 
v_and_b32 v18, 1, v12 // (op 1452) 

s_mul_hi_u32 s53, s52, s13 // (op 1452) 
v_and_b32 v12, 15, v1 // (op 1452) 
s_sub_u32 s54, s52, s53 // (op 1452) 
v_lshl_add_u32 v19, v18, 4, v12 // (op 1452) 


s_lshr_b32 s55, s54, 1 // (op 1452) 

v_and_b32 v12, 63, v1 // (op 1452) 
s_add_u32 s54, s55, s53 // (op 1452) 


v_lshrrev_b32 v18, 4, v12 // (op 1452) 

s_lshr_b32 s53, s54, s11 // (op 1452) 

s_and_b32 s54, s11, 2147483648 // (op 1452) 
s_cmp_eq_i32 s54, 0 // (op 1452) 
s_mov_b32 s55, scc // (op 1452) 

s_cmp_eq_i32 s55, 0 // (op 1452) 
s_cselect_b32 s54, s52, s53 // (op 1452) 


s_mul_hi_u32 s53, s54, s12 // (op 1452) 
s_sub_u32 s55, s54, s53 // (op 1452) 
s_lshr_b32 s72, s55, 1 // (op 1452) 

s_add_u32 s55, s72, s53 // (op 1452) 


s_lshr_b32 s53, s55, s10 // (op 1452) 

s_and_b32 s55, s10, 2147483648 // (op 1452) 
s_cmp_eq_i32 s55, 0 // (op 1452) 
s_mov_b32 s72, scc // (op 1452) 

s_cmp_eq_i32 s72, 0 // (op 1452) 
s_cselect_b32 s55, s54, s53 // (op 1452) 


s_mul_i32 s53, s55, s14 // (op 1452) 

s_sub_u32 s55, s54, s53 // (op 1452) 

v_lshl_add_u32 v12, s55, 5, v19 // (op 1452) 


v_mul_lo_u32 v19, v12, s38 // (op 1452) 

s_ashr_i64 s[72:73], s[2:3], 7 // (op 1452) 
s_mul_i32 s53, s54, s72 // (op 1452) 


s_sub_u32 s54, s52, s53 // (op 1452) 


v_lshl_add_u32 v12, s54, 2, v18 // (op 1452) 


v_add_u32 v18, v12, v19 // (op 1452) 


s_mul_i32 s53, s38, s23 // (op 1454) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 1454) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 1454) least significant
s_mul_hi_u32 s55, s38, s22 // (op 1454) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1454) most significant: sum
s_add_u32 s53, s53, s55 // (op 1454) most significant: sum


s_mov_b32 s72, s40 // (op 1454) reg expression
s_mov_b32 s73, s41 // (op 1454) reg expression
s_mov_b32 s74, s52 // (op 1454) reg expression
s_mov_b32 s75, 131072 // (op 1454) call()

v_lshrrev_b32 v12, 6, v1 // (op 1471) 
s_mul_i32 s52, s16, s8 // (op 1471) 
s_add_u32 s52, s52, s0 // (op 1471) 
v_and_b32 v19, 1, v12 // (op 1471) 

s_mul_hi_u32 s53, s52, s13 // (op 1471) 
v_and_b32 v12, 15, v1 // (op 1471) 
s_sub_u32 s54, s52, s53 // (op 1471) 
v_lshl_add_u32 v28, v19, 4, v12 // (op 1471) 


s_lshr_b32 s55, s54, 1 // (op 1471) 

v_and_b32 v12, 63, v1 // (op 1471) 
s_add_u32 s54, s55, s53 // (op 1471) 


v_lshrrev_b32 v19, 4, v12 // (op 1471) 

s_lshr_b32 s53, s54, s11 // (op 1471) 

s_and_b32 s54, s11, 2147483648 // (op 1471) 
s_cmp_eq_i32 s54, 0 // (op 1471) 
s_mov_b32 s55, scc // (op 1471) 

s_cmp_eq_i32 s55, 0 // (op 1471) 
s_cselect_b32 s54, s52, s53 // (op 1471) 


s_mul_hi_u32 s53, s54, s12 // (op 1471) 
s_sub_u32 s55, s54, s53 // (op 1471) 
s_lshr_b32 s76, s55, 1 // (op 1471) 

s_add_u32 s55, s76, s53 // (op 1471) 


s_lshr_b32 s53, s55, s10 // (op 1471) 

s_and_b32 s55, s10, 2147483648 // (op 1471) 
s_cmp_eq_i32 s55, 0 // (op 1471) 
s_mov_b32 s76, scc // (op 1471) 

s_cmp_eq_i32 s76, 0 // (op 1471) 
s_cselect_b32 s55, s54, s53 // (op 1471) 


s_mul_i32 s53, s55, s14 // (op 1471) 

s_sub_u32 s55, s54, s53 // (op 1471) 

v_lshl_add_u32 v12, s55, 5, v28 // (op 1471) 


v_mul_lo_u32 v28, v12, s38 // (op 1471) 

s_ashr_i64 s[76:77], s[2:3], 7 // (op 1471) 
s_mul_i32 s53, s54, s76 // (op 1471) 


s_sub_u32 s54, s52, s53 // (op 1471) 


v_lshl_add_u32 v12, s54, 2, v19 // (op 1471) 


v_add_u32 v19, v12, v28 // (op 1471) 


s_mul_i32 s53, s38, s23 // (op 1473) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 1473) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 1473) least significant
s_mul_hi_u32 s55, s38, s22 // (op 1473) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1473) most significant: sum
s_add_u32 s53, s53, s55 // (op 1473) most significant: sum


s_mov_b32 s72, s40 // (op 1473) reg expression
s_mov_b32 s73, s41 // (op 1473) reg expression
s_mov_b32 s74, s52 // (op 1473) reg expression
s_mov_b32 s75, 131072 // (op 1473) call()




buffer_load_ubyte v12, v16, s[68:71], 0 offen offset: 0 // (op 747) Load value



buffer_load_ubyte v28, v18, s[72:75], 0 offen offset: 0 // (op 752) Load value
v_and_b32 v29, -8, v1 // (op 1660) 
v_sub_u32 v30, v1, v29 // (op 1660) 

v_lshlrev_b32 v29, 4, v30 // (op 1660) 

v_lshrrev_b32 v30, 3, v1 // (op 1660) 
v_lshl_add_u32 v31, v30, 7, v29 // (op 1660) 


s_nop 0
v_readfirstlane_b32 s52, v31 // (op 1660) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2159) 



s_nop 0
s_mov_b32 m0, s52 // (op 976) reg expression
buffer_load_dwordx4 v24, s[64:67], 0 offen offset: 0 lds // (op 976) Load value direct to lds

v_and_b32 v29, -8, v1 // (op 1595) 
v_sub_u32 v30, v1, v29 // (op 1595) 

v_lshlrev_b32 v29, 4, v30 // (op 1595) 

v_lshrrev_b32 v30, 3, v1 // (op 1595) 
v_lshl_add_u32 v31, v30, 7, v29 // (op 1595) 


s_nop 0
v_readfirstlane_b32 s52, v31 // (op 1595) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2140) 


s_nop 0
s_add_u32 s53, 4096, s52 // (op 981) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s53 // (op 981) reg expression
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 981) Load value direct to lds





buffer_load_ubyte v29, v16, s[68:71], 0 offen offset:4 // (op 763) Load value




buffer_load_ubyte v16, v18, s[72:75], 0 offen offset:4 // (op 768) Load value

v_and_b32 v18, -8, v1 // (op 2063) 
v_sub_u32 v30, v1, v18 // (op 2063) 

v_lshlrev_b32 v18, 4, v30 // (op 2063) 

v_lshrrev_b32 v30, 3, v1 // (op 2063) 
v_lshl_add_u32 v31, v30, 7, v18 // (op 2063) 


s_nop 0
v_readfirstlane_b32 s52, v31 // (op 2063) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2243) 


s_nop 0
s_add_u32 s53, 8192, s52 // (op 986) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s53 // (op 986) reg expression
s_mov_b32 s54, 128 // (op 986) call()
buffer_load_dwordx4 v24, s[64:67], s54 offen offset: 0 lds // (op 986) Load value direct to lds




v_and_b32 v18, -8, v1 // (op 2024) 
v_sub_u32 v24, v1, v18 // (op 2024) 

v_lshlrev_b32 v18, 4, v24 // (op 2024) 

v_lshrrev_b32 v24, 3, v1 // (op 2024) 
v_lshl_add_u32 v25, v24, 7, v18 // (op 2024) 


s_nop 0
v_readfirstlane_b32 s52, v25 // (op 2024) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2234) 


s_nop 0
s_add_u32 s53, 12288, s52 // (op 988) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s53 // (op 988) reg expression
s_mov_b32 s54, 128 // (op 988) call()
buffer_load_dwordx4 v20, s[60:63], s54 offen offset: 0 lds // (op 988) Load value direct to lds




s_waitcnt vmcnt(5)
s_barrier  // (op 2156) 



ds_read_b128 v[30:33], v15 // (op 793) Load local data 
ds_read_b128 v[34:37], v15 offset:64 // (op 793) Load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 2137) 



ds_read_b128 v[38:41], v13 offset:4096 // (op 790) Load local data 
ds_read_b128 v[42:45], v13 offset:4160 // (op 790) Load local data 
v_and_b32 v18, -8, v1 // (op 1567) 
v_sub_u32 v20, v1, v18 // (op 1567) 

v_lshlrev_b32 v18, 4, v20 // (op 1567) 

v_lshrrev_b32 v20, 3, v1 // (op 1567) 
v_lshl_add_u32 v21, v20, 7, v18 // (op 1567) 


s_nop 0
v_readfirstlane_b32 s52, v21 // (op 1567) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -8, v1 // (op 1632) 
v_sub_u32 v20, v1, v18 // (op 1632) 

v_lshlrev_b32 v18, 4, v20 // (op 1632) 

v_lshrrev_b32 v20, 3, v1 // (op 1632) 
v_lshl_add_u32 v21, v20, 7, v18 // (op 1632) 


s_nop 0
v_readfirstlane_b32 s53, v21 // (op 1632) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -8, v1 // (op 2009) 
v_sub_u32 v20, v1, v18 // (op 2009) 

v_lshlrev_b32 v18, 4, v20 // (op 2009) 

v_lshrrev_b32 v20, 3, v1 // (op 2009) 
v_lshl_add_u32 v21, v20, 7, v18 // (op 2009) 


s_nop 0
v_readfirstlane_b32 s54, v21 // (op 2009) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -8, v1 // (op 2048) 
v_sub_u32 v20, v1, v18 // (op 2048) 

v_lshlrev_b32 v18, 4, v20 // (op 2048) 

v_lshrrev_b32 v20, 3, v1 // (op 2048) 
v_lshl_add_u32 v21, v20, 7, v18 // (op 2048) 


s_nop 0
v_readfirstlane_b32 s55, v21 // (op 2048) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s1, 0 // (op 53) call()
s_and_b32 s76, s7, 4294967294 // (op 52) 
s_cmp_lt_u32 s1, s76 // (op 52) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_52 // (op 52) Condition: Top (jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_52 if false)
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_52:
s_add_u32 s1, s1, 2 // (op 54) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v12, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 78) 


v_mov_b32 v18, v17 // (op 72) 

buffer_load_ubyte v12, v18, s[68:71], 0 offen offset:8 // (op 72) Load value


v_mov_b32 v20, v19 // (op 76) 

buffer_load_ubyte v28, v20, s[72:75], 0 offen offset:8 // (op 76) Load value
s_barrier  // (op 2162) 


v_mov_b64 v[24:25], v[26:27] // (op 917) 

s_mov_b32 m0, s53 // (op 917) reg expression
s_mov_b32 s76, 256 // (op 917) call()
buffer_load_dwordx4 v24, s[64:67], s76 offen offset: 0 lds // (op 917) Load value direct to lds

s_barrier  // (op 2143) 


v_mov_b64 v[46:47], v[22:23] // (op 912) 
s_add_u32 s76, 4096, s52 // (op 912) 

s_mov_b32 m0, s76 // (op 912) reg expression
s_mov_b32 s77, 256 // (op 912) call()
buffer_load_dwordx4 v46, s[60:63], s77 offen offset: 0 lds // (op 912) Load value direct to lds


s_waitcnt vmcnt(5)
s_barrier  // (op 2240) 



ds_read_b128 v[30:33], v10 offset:8192 // (op 611) Load local data 
ds_read_b128 v[34:37], v10 offset:8256 // (op 611) Load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 2231) 



ds_read_b128 v[38:41], v9 offset:12288 // (op 609) Load local data 
ds_read_b128 v[42:45], v9 offset:12352 // (op 609) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v29, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 616) 



buffer_load_ubyte v29, v18, s[68:71], 0 offen offset:12 // (op 613) Load value




v_add_u32 v17, 8, v17 // (op 1335) 
buffer_load_ubyte v16, v20, s[72:75], 0 offen offset:12 // (op 615) Load value

s_barrier  // (op 2246) 
v_add_u32 v19, 8, v19 // (op 1463) 


s_add_u32 s76, 8192, s55 // (op 971) 

s_mov_b32 m0, s76 // (op 971) reg expression
s_mov_b32 s77, 384 // (op 971) call()
buffer_load_dwordx4 v24, s[64:67], s77 offen offset: 0 lds // (op 971) Load value direct to lds



s_barrier  // (op 2237) 
v_add_co_u32 v26, vcc, 256, v26 // (op 1207) least significant half


v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 1207) most significant half
s_add_u32 s76, 12288, s54 // (op 966) 

s_mov_b32 m0, s76 // (op 966) reg expression
s_mov_b32 s77, 384 // (op 966) call()
buffer_load_dwordx4 v46, s[60:63], s77 offen offset: 0 lds // (op 966) Load value direct to lds



v_add_co_u32 v22, vcc, 256, v22 // (op 1076) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 1076) most significant half
s_waitcnt vmcnt(5)
s_barrier  // (op 2168) 



ds_read_b128 v[30:33], v4 // (op 67) Load local data 
ds_read_b128 v[34:37], v4 offset:64 // (op 67) Load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 2149) 



ds_read_b128 v[38:41], v3 offset:4096 // (op 62) Load local data 
ds_read_b128 v[42:45], v3 offset:4160 // (op 62) Load local data 
s_and_b32 s76, s7, 4294967294 // (op 52) 
s_cmp_lt_u32 s1, s76 // (op 52) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_52 // (op 52) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_52 if true)
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_52:










s_barrier  // (op 2255) 
s_and_b32 s52, s7, 4294967294 // (op 1039) 
v_lshrrev_b32 v12, 3, v1 // (op 1039) 
s_add_u32 s53, s0, s52 // (op 1039) 

v_and_b32 v17, -8, v1 // (op 1039) 
v_sub_u32 v18, v1, v17 // (op 1039) 

s_mul_i32 s52, s16, s8 // (op 1039) 
s_add_u32 s52, s52, s53 // (op 1039) 

v_lshlrev_b32 v17, 4, v18 // (op 1039) 

s_mul_hi_u32 s53, s52, s13 // (op 1039) 
s_sub_u32 s54, s52, s53 // (op 1039) 
s_lshr_b32 s55, s54, 1 // (op 1039) 

s_add_u32 s54, s55, s53 // (op 1039) 


s_lshr_b32 s53, s54, s11 // (op 1039) 

s_and_b32 s54, s11, 2147483648 // (op 1039) 
s_cmp_eq_i32 s54, 0 // (op 1039) 
s_mov_b32 s55, scc // (op 1039) 

s_cmp_eq_i32 s55, 0 // (op 1039) 
s_cselect_b32 s54, s52, s53 // (op 1039) 


s_mul_hi_u32 s53, s54, s12 // (op 1039) 
s_sub_u32 s55, s54, s53 // (op 1039) 
s_lshr_b32 s76, s55, 1 // (op 1039) 

s_add_u32 s55, s76, s53 // (op 1039) 


s_lshr_b32 s53, s55, s10 // (op 1039) 

s_and_b32 s55, s10, 2147483648 // (op 1039) 
s_cmp_eq_i32 s55, 0 // (op 1039) 
s_mov_b32 s76, scc // (op 1039) 

s_cmp_eq_i32 s76, 0 // (op 1039) 
s_cselect_b32 s55, s54, s53 // (op 1039) 


v_lshl_add_u32 v18, s55, 5, v12 // (op 1039) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1039) 
s_mul_i32 s53, s54, s76 // (op 1039) 


s_sub_u32 s54, s52, s53 // (op 1039) 


v_lshl_add_u32 v12, s54, 7, v17 // (op 1039) 


v_mov_b32 v21, 0 // (op 1039) convert
v_mov_b32 v20, v12 // (op 1039) convert

v_mul_lo_u32 v23, v18, s25 // (op 1039) most significant: low of low * high
v_mul_hi_u32 v12, v18, s24 // (op 1039) most significant: high of low * low
v_mul_lo_u32 v22, v18, s24 // (op 1039) least significant: low of low * low
v_add_u32 v23, v23, v12 // (op 1039) most significant: sum

v_add_co_u32 v22, s[52:53], v22, v20 // (op 1039) least significant half
v_addc_co_u32 v23, s[52:53], v23, v21, s[52:53] // (op 1039) most significant half



s_mul_i32 s53, s24, s29 // (op 1041) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 1041) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 1041) least significant
s_mul_hi_u32 s55, s24, s28 // (op 1041) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1041) most significant: sum
s_add_u32 s53, s53, s55 // (op 1041) most significant: sum


s_mov_b32 s60, s26 // (op 1041) reg expression
s_mov_b32 s61, s27 // (op 1041) reg expression
s_mov_b32 s62, s52 // (op 1041) reg expression
s_mov_b32 s63, 131072 // (op 1041) call()

s_and_b32 s52, s7, 4294967294 // (op 1173) 
v_lshrrev_b32 v12, 3, v1 // (op 1173) 
s_add_u32 s53, s0, s52 // (op 1173) 

v_and_b32 v17, -8, v1 // (op 1173) 
v_sub_u32 v18, v1, v17 // (op 1173) 

s_mul_i32 s52, s16, s8 // (op 1173) 
s_add_u32 s52, s52, s53 // (op 1173) 

v_lshlrev_b32 v17, 4, v18 // (op 1173) 

s_mul_hi_u32 s53, s52, s13 // (op 1173) 
s_sub_u32 s54, s52, s53 // (op 1173) 
s_lshr_b32 s55, s54, 1 // (op 1173) 

s_add_u32 s54, s55, s53 // (op 1173) 


s_lshr_b32 s53, s54, s11 // (op 1173) 

s_and_b32 s54, s11, 2147483648 // (op 1173) 
s_cmp_eq_i32 s54, 0 // (op 1173) 
s_mov_b32 s55, scc // (op 1173) 

s_cmp_eq_i32 s55, 0 // (op 1173) 
s_cselect_b32 s54, s52, s53 // (op 1173) 


s_mul_hi_u32 s53, s54, s12 // (op 1173) 
s_sub_u32 s55, s54, s53 // (op 1173) 
s_lshr_b32 s76, s55, 1 // (op 1173) 

s_add_u32 s55, s76, s53 // (op 1173) 


s_lshr_b32 s53, s55, s10 // (op 1173) 

s_and_b32 s55, s10, 2147483648 // (op 1173) 
s_cmp_eq_i32 s55, 0 // (op 1173) 
s_mov_b32 s76, scc // (op 1173) 

s_cmp_eq_i32 s76, 0 // (op 1173) 
s_cselect_b32 s55, s54, s53 // (op 1173) 


s_mul_i32 s53, s55, s14 // (op 1173) 

s_sub_u32 s55, s54, s53 // (op 1173) 

v_lshl_add_u32 v18, s55, 5, v12 // (op 1173) 


v_mul_lo_u32 v21, v18, s31 // (op 1173) most significant: low of low * high
v_mul_hi_u32 v12, v18, s30 // (op 1173) most significant: high of low * low
v_mul_lo_u32 v20, v18, s30 // (op 1173) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 1173) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1173) 
s_mul_i32 s53, s54, s76 // (op 1173) 


s_sub_u32 s54, s52, s53 // (op 1173) 


v_lshl_add_u32 v12, s54, 7, v17 // (op 1173) 


v_mov_b32 v19, 0 // (op 1173) convert
v_mov_b32 v18, v12 // (op 1173) convert

v_add_co_u32 v24, s[52:53], v18, v20 // (op 1173) least significant half
v_addc_co_u32 v25, s[52:53], v19, v21, s[52:53] // (op 1173) most significant half



s_mul_i32 s53, s30, s23 // (op 1175) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 1175) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 1175) least significant
s_mul_hi_u32 s55, s30, s22 // (op 1175) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1175) most significant: sum
s_add_u32 s53, s53, s55 // (op 1175) most significant: sum


s_mov_b32 s64, s32 // (op 1175) reg expression
s_mov_b32 s65, s33 // (op 1175) reg expression
s_mov_b32 s66, s52 // (op 1175) reg expression
s_mov_b32 s67, 131072 // (op 1175) call()

s_and_b32 s52, s7, 4294967294 // (op 1301) 
v_lshrrev_b32 v12, 7, v1 // (op 1301) 
s_add_u32 s53, s0, s52 // (op 1301) 

v_and_b32 v17, 15, v1 // (op 1301) 
v_lshl_add_u32 v18, v12, 4, v17 // (op 1301) 


s_mul_i32 s52, s16, s8 // (op 1301) 
s_add_u32 s52, s52, s53 // (op 1301) 

v_and_b32 v12, 63, v1 // (op 1301) 
s_mul_hi_u32 s53, s52, s13 // (op 1301) 
v_lshrrev_b32 v17, 4, v12 // (op 1301) 

s_sub_u32 s54, s52, s53 // (op 1301) 
s_lshr_b32 s55, s54, 1 // (op 1301) 

s_add_u32 s54, s55, s53 // (op 1301) 


s_lshr_b32 s53, s54, s11 // (op 1301) 

s_and_b32 s54, s11, 2147483648 // (op 1301) 
s_cmp_eq_i32 s54, 0 // (op 1301) 
s_mov_b32 s55, scc // (op 1301) 

s_cmp_eq_i32 s55, 0 // (op 1301) 
s_cselect_b32 s54, s52, s53 // (op 1301) 


s_mul_hi_u32 s53, s54, s12 // (op 1301) 
s_sub_u32 s55, s54, s53 // (op 1301) 
s_lshr_b32 s76, s55, 1 // (op 1301) 

s_add_u32 s55, s76, s53 // (op 1301) 


s_lshr_b32 s53, s55, s10 // (op 1301) 

s_and_b32 s55, s10, 2147483648 // (op 1301) 
s_cmp_eq_i32 s55, 0 // (op 1301) 
s_mov_b32 s76, scc // (op 1301) 

s_cmp_eq_i32 s76, 0 // (op 1301) 
s_cselect_b32 s55, s54, s53 // (op 1301) 


v_lshl_add_u32 v12, s55, 5, v18 // (op 1301) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1301) 
s_mul_i32 s53, s54, s76 // (op 1301) 


s_sub_u32 s54, s52, s53 // (op 1301) 


v_lshl_add_u32 v18, s54, 2, v17 // (op 1301) 


v_mul_lo_u32 v17, v12, s34 // (op 1301) 
v_add_u32 v17, v17, v18 // (op 1301) 


s_mul_i32 s53, s34, s29 // (op 1303) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 1303) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 1303) least significant
s_mul_hi_u32 s55, s34, s28 // (op 1303) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1303) most significant: sum
s_add_u32 s53, s53, s55 // (op 1303) most significant: sum


s_mov_b32 s68, s36 // (op 1303) reg expression
s_mov_b32 s69, s37 // (op 1303) reg expression
s_mov_b32 s70, s52 // (op 1303) reg expression
s_mov_b32 s71, 131072 // (op 1303) call()

s_and_b32 s52, s7, 4294967294 // (op 1429) 
v_lshrrev_b32 v12, 6, v1 // (op 1429) 
s_add_u32 s53, s0, s52 // (op 1429) 

v_and_b32 v18, 1, v12 // (op 1429) 

v_and_b32 v12, 15, v1 // (op 1429) 
s_mul_i32 s52, s16, s8 // (op 1429) 
s_add_u32 s52, s52, s53 // (op 1429) 

v_lshl_add_u32 v19, v18, 4, v12 // (op 1429) 


s_mul_hi_u32 s53, s52, s13 // (op 1429) 
v_and_b32 v12, 63, v1 // (op 1429) 
s_sub_u32 s54, s52, s53 // (op 1429) 
v_lshrrev_b32 v18, 4, v12 // (op 1429) 

s_lshr_b32 s55, s54, 1 // (op 1429) 

s_add_u32 s54, s55, s53 // (op 1429) 


s_lshr_b32 s53, s54, s11 // (op 1429) 

s_and_b32 s54, s11, 2147483648 // (op 1429) 
s_cmp_eq_i32 s54, 0 // (op 1429) 
s_mov_b32 s55, scc // (op 1429) 

s_cmp_eq_i32 s55, 0 // (op 1429) 
s_cselect_b32 s54, s52, s53 // (op 1429) 


s_mul_hi_u32 s53, s54, s12 // (op 1429) 
s_sub_u32 s55, s54, s53 // (op 1429) 
s_lshr_b32 s76, s55, 1 // (op 1429) 

s_add_u32 s55, s76, s53 // (op 1429) 


s_lshr_b32 s53, s55, s10 // (op 1429) 

s_and_b32 s55, s10, 2147483648 // (op 1429) 
s_cmp_eq_i32 s55, 0 // (op 1429) 
s_mov_b32 s76, scc // (op 1429) 

s_cmp_eq_i32 s76, 0 // (op 1429) 
s_cselect_b32 s55, s54, s53 // (op 1429) 


s_mul_i32 s53, s55, s14 // (op 1429) 

s_sub_u32 s55, s54, s53 // (op 1429) 

v_lshl_add_u32 v12, s55, 5, v19 // (op 1429) 


v_mul_lo_u32 v19, v12, s38 // (op 1429) 

s_ashr_i64 s[76:77], s[2:3], 7 // (op 1429) 
s_mul_i32 s53, s54, s76 // (op 1429) 


s_sub_u32 s54, s52, s53 // (op 1429) 


v_lshl_add_u32 v12, s54, 2, v18 // (op 1429) 


v_add_u32 v18, v12, v19 // (op 1429) 


s_mul_i32 s53, s38, s23 // (op 1431) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 1431) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 1431) least significant
s_mul_hi_u32 s55, s38, s22 // (op 1431) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1431) most significant: sum
s_add_u32 s53, s53, s55 // (op 1431) most significant: sum


s_mov_b32 s72, s40 // (op 1431) reg expression
s_mov_b32 s73, s41 // (op 1431) reg expression
s_mov_b32 s74, s52 // (op 1431) reg expression
s_mov_b32 s75, 131072 // (op 1431) call()

v_and_b32 v12, -8, v1 // (op 1580) 
v_sub_u32 v19, v1, v12 // (op 1580) 

v_lshlrev_b32 v12, 4, v19 // (op 1580) 

v_lshrrev_b32 v19, 3, v1 // (op 1580) 
v_lshl_add_u32 v20, v19, 7, v12 // (op 1580) 


s_nop 0
v_readfirstlane_b32 s52, v20 // (op 1580) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -8, v1 // (op 1645) 
v_sub_u32 v19, v1, v12 // (op 1645) 

v_lshlrev_b32 v12, 4, v19 // (op 1645) 

v_lshrrev_b32 v19, 3, v1 // (op 1645) 
v_lshl_add_u32 v20, v19, 7, v12 // (op 1645) 


s_nop 0
v_readfirstlane_b32 s53, v20 // (op 1645) Wait state hazard: VALU Write Readlane Hazard

s_and_b32 s54, s7, 4294967294 // (op 552) 
s_cmp_lt_u32 s54, s7 // (op 551) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_551 // (op 551) Condition: Top (jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_551 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_551:
s_barrier  // (op 2146) 
s_add_u32 s54, s54, 1 // (op 553) 


v_mov_b64 v[20:21], v[22:23] // (op 954) 
s_add_u32 s55, 4096, s52 // (op 954) 

s_mov_b32 m0, s55 // (op 954) reg expression
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 954) Load value direct to lds


v_add_co_u32 v22, vcc, 128, v22 // (op 1029) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 1029) most significant half
s_barrier  // (op 2165) 


v_mov_b64 v[20:21], v[24:25] // (op 960) 

s_mov_b32 m0, s53 // (op 960) reg expression
buffer_load_dwordx4 v20, s[64:67], 0 offen offset: 0 lds // (op 960) Load value direct to lds

v_add_co_u32 v24, vcc, 128, v24 // (op 1165) least significant half
v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 1165) most significant half
s_waitcnt vmcnt(1)
s_barrier  // (op 2152) 



ds_read_b128 v[26:29], v3 offset:4096 // (op 559) Load local data 
ds_read_b128 v[30:33], v3 offset:4160 // (op 559) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 2171) 



ds_read_b128 v[34:37], v4 // (op 561) Load local data 
ds_read_b128 v[38:41], v4 offset:64 // (op 561) Load local data 


v_mov_b32 v12, v17 // (op 563) 

buffer_load_ubyte v16, v12, s[68:71], 0 offen offset: 0 // (op 563) Load value

v_add_u32 v17, 4, v17 // (op 1293) 


v_mov_b32 v12, v18 // (op 565) 

buffer_load_ubyte v19, v12, s[72:75], 0 offen offset: 0 // (op 565) Load value

v_add_u32 v18, 4, v18 // (op 1421) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v16, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 566) 




s_cmp_lt_u32 s54, s7 // (op 551) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_551 // (op 551) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_551 if true)
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_551:







s_barrier  // (op 2353) 


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_lg_u32 s18, 0 // (op 291) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Send_Tile_291 // (op 291) Condition: False, jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Send_Tile_291
v_lshrrev_b32 v12, 5, v1 // (op 1899) 
v_lshlrev_b32 v13, 2, v12 // (op 1899) 

v_lshl_add_u32 v12, s16, 5, v13 // (op 1899) 

v_and_b32 v13, -32, v1 // (op 1899) 
v_sub_u32 v15, v1, v13 // (op 1899) 

v_lshl_add_u32 v13, v12, 5, v15 // (op 1899) 


v_lshlrev_b32 v12, 2, v13 // (op 1899) 

s_lshl_b32 s52, s15, 12 // (op 1901) 
s_mov_b32 s76, s42 // (op 1901) reg expression
s_mov_b32 s77, s43 // (op 1901) reg expression
s_mov_b32 s78, s52 // (op 1901) reg expression
s_mov_b32 s79, 131072 // (op 1901) call()




buffer_store_dword a0, v12, s[76:79], 0 offen sc1 // (op 288) Store value
buffer_store_dword a1, v12, s[76:79], 0 offen offset:128 sc1 // (op 288) Store value
buffer_store_dword a2, v12, s[76:79], 0 offen offset:256 sc1 // (op 288) Store value
buffer_store_dword a3, v12, s[76:79], 0 offen offset:384 sc1 // (op 288) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 292) 
v_cmp_eq_i32 vcc, v1, 0 // (op 296) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_296 // (op 296) Condition: False, jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_296
s_mov_b32 s52, 1 // (op 294) call()
s_mov_b32 s55, 0 // (op 295) convert
s_mov_b32 s54, s16 // (op 295) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 295) 
s_add_u32 s80, s44, s54 // (op 295) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 295) End of Int64 add, unlocking SCC
s_store_dword s52, s[80:81], 0 glc // (op 295) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_296 // (op 296) Condition: Done, jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_296
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_296:
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_296:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Send_Tile_291 // (op 291) Condition: Done, jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Send_Tile_291
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Send_Tile_291:
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Send_Tile_291:
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
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Receive_Tile_307 // (op 307) Condition: False, jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Receive_Tile_307
s_mov_b32 s52, 0 // (op 273) call()
s_mul_i32 s53, s52, s8 // (op 272) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 272) 
s_mov_b32 s80, s54 // (op 272) convert

s_sub_u32 s54, s80, s19 // (op 272) 

s_sub_u32 s55, s54, 1 // (op 272) 

s_cmp_lt_u32 s53, s55 // (op 272) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_272 // (op 272) Condition: Top (jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_272 if false)
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_ReceiveTileLoop_272:
s_add_u32 s53, s16, 1 // (op 1916) 
v_lshrrev_b32 v12, 5, v1 // (op 1916) 
s_add_u32 s54, s53, s52 // (op 1916) 
s_lshl_b32 s54, s54, 5 // (op 1916) 

v_lshlrev_b32 v13, 2, v12 // (op 1916) 

v_add_lshl_u32 v12, s54, v13, 5 // (op 1916) 


v_and_b32 v13, -32, v1 // (op 1916) 
v_sub_u32 v15, v1, v13 // (op 1916) 

v_add_lshl_u32 v13, v12, v15, 2 // (op 1916) 


s_lshl_b32 s53, s15, 12 // (op 1918) 
s_mov_b32 s76, s42 // (op 1918) reg expression
s_mov_b32 s77, s43 // (op 1918) reg expression
s_mov_b32 s78, s53 // (op 1918) reg expression
s_mov_b32 s79, 131072 // (op 1918) call()

RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_311:
s_add_u32 s53, s16, 1 // (op 309) 
s_add_u32 s54, s53, s52 // (op 309) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 309) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 309) 

s_add_u32 s80, s44, s54 // (op 309) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 309) End of Int64 add, unlocking SCC
s_load_dword s53, s[80:81], 0 glc // (op 309) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 311) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_311 // (op 311) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_311 if true)
s_barrier  // (op 315) 
v_cmp_eq_i32 vcc, v1, 0 // (op 314) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_314 // (op 314) Condition: False, jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_314
s_mov_b32 s53, 0 // (op 312) call()
s_add_u32 s54, s16, 1 // (op 313) 
s_add_u32 s80, s54, s52 // (op 313) Start of Int64 add, locking SCC
s_addc_u32 s81, 0, 0 // (op 313) End of Int64 add, unlocking SCC
s_lshl_b64 s[80:81], s[80:81], 2 // (op 313) 

s_add_u32 s54, s44, s80 // (op 313) Start of Int64 add, locking SCC
s_addc_u32 s55, s45, s81 // (op 313) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 313) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_314 // (op 314) Condition: Done, jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_314
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_314:
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_314:




buffer_load_dword v16, v13, s[76:79], 0 offen // (op 326) Load value
buffer_load_dword v17, v13, s[76:79], 0 offen offset:128 // (op 326) Load value
buffer_load_dword v18, v13, s[76:79], 0 offen offset:256 // (op 326) Load value
buffer_load_dword v19, v13, s[76:79], 0 offen offset:384 // (op 326) Load value
v_accvgpr_read v20, a0 // (op 329) reg expression
v_accvgpr_read v21, a1 // (op 329) reg expression
v_accvgpr_read v22, a2 // (op 329) reg expression
v_accvgpr_read v23, a3 // (op 329) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v20, v20, v16 // (op 331) 
s_waitcnt vmcnt(2)
v_add_f32 v21, v21, v17 // (op 331) 
s_waitcnt vmcnt(1)
v_add_f32 v22, v22, v18 // (op 331) 
s_waitcnt vmcnt(0)
v_add_f32 v23, v23, v19 // (op 331) 

v_accvgpr_write a0, v20 // (op 333) reg expression
v_accvgpr_write a1, v21 // (op 333) reg expression
v_accvgpr_write a2, v22 // (op 333) reg expression
v_accvgpr_write a3, v23 // (op 333) reg expression

s_add_u32 s52, s52, 1 // (op 274) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 272) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 272) 
s_mov_b32 s80, s54 // (op 272) convert

s_sub_u32 s54, s80, s19 // (op 272) 

s_sub_u32 s55, s54, 1 // (op 272) 

s_cmp_lt_u32 s53, s55 // (op 272) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_ReceiveTileLoop_272 // (op 272) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_ReceiveTileLoop_272 if true)
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_272:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Receive_Tile_307 // (op 307) Condition: Done, jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Receive_Tile_307
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Receive_Tile_307:
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Receive_Tile_307:

s_cmp_eq_i32 s18, 0 // (op 350) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_350 // (op 350) Condition: False, jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_350
v_cmp_eq_f32 vcc, 0.00000, v0 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_116
v_accvgpr_read v16, a0 // (op 42) 
v_accvgpr_read v17, a1 // (op 42) 
v_accvgpr_read v18, a2 // (op 42) 
v_accvgpr_read v19, a3 // (op 42) 
v_mul_f32 v20, v11, v16 // (op 42) 
v_mul_f32 v21, v11, v17 // (op 42) 
v_mul_f32 v22, v11, v18 // (op 42) 
v_mul_f32 v23, v11, v19 // (op 42) 

v_lshrrev_b32 v16, 16, v20 // (op 48) 
v_lshrrev_b32 v17, 16, v21 // (op 48) 
v_lshrrev_b32 v18, 16, v22 // (op 48) 
v_lshrrev_b32 v19, 16, v23 // (op 48) 

v_lshrrev_b32 v15, 7, v1 // (op 1699) 
s_mul_i32 s52, s16, s8 // (op 1699) 
s_add_u32 s52, s52, s0 // (op 1699) 
v_and_b32 v12, 63, v1 // (op 1699) 
s_mul_hi_u32 s53, s52, s13 // (op 1699) 
v_lshrrev_b32 v13, 4, v12 // (op 1699) 
s_sub_u32 s54, s52, s53 // (op 1699) 
v_lshlrev_b32 v20, 2, v13 // (op 1699) 

s_lshr_b32 s55, s54, 1 // (op 1699) 

v_lshl_add_u32 v13, v15, 4, v20 // (op 1699) 


s_add_u32 s54, s55, s53 // (op 1699) 


v_lshrrev_b32 v15, 2, v12 // (op 1699) 

s_lshr_b32 s53, s54, s11 // (op 1699) 

v_and_b32 v12, 3, v15 // (op 1699) 

s_and_b32 s54, s11, 2147483648 // (op 1699) 
v_and_b32 v15, 3, v1 // (op 1699) 
s_cmp_eq_i32 s54, 0 // (op 1699) 
s_mov_b32 s55, scc // (op 1699) 

v_lshl_add_u32 v20, v12, 2, v15 // (op 1699) 


s_cmp_eq_i32 s55, 0 // (op 1699) 
s_cselect_b32 s54, s52, s53 // (op 1699) 



v_lshrrev_b32 v12, 6, v1 // (op 1699) 
s_mul_hi_u32 s55, s54, s12 // (op 1699) 
v_and_b32 v15, 1, v12 // (op 1699) 

s_sub_u32 s52, s54, s55 // (op 1699) 
v_lshl_add_u32 v12, v15, 4, v20 // (op 1699) 


s_lshr_b32 s53, s52, 1 // (op 1699) 

s_add_u32 s52, s53, s55 // (op 1699) 


s_lshr_b32 s53, s52, s10 // (op 1699) 

s_and_b32 s52, s10, 2147483648 // (op 1699) 
s_cmp_eq_i32 s52, 0 // (op 1699) 
s_mov_b32 s55, scc // (op 1699) 

s_cmp_eq_i32 s55, 0 // (op 1699) 
s_cselect_b32 s52, s54, s53 // (op 1699) 


s_mul_i32 s53, s52, s14 // (op 1699) 
v_lshl_add_u32 v15, s52, 5, v13 // (op 1699) 


s_sub_u32 s52, s54, s53 // (op 1699) 


v_lshl_add_u32 v13, s52, 5, v12 // (op 1699) 


v_mul_lo_u32 v12, v13, s46 // (op 1699) 

v_add_lshl_u32 v13, v15, v12, 1 // (op 1699) 


s_mul_i32 s53, s46, s23 // (op 1701) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1701) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1701) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1701) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1701) most significant: sum
s_add_u32 s53, s53, s55 // (op 1701) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 1701) 

s_mov_b32 s76, s48 // (op 1701) reg expression
s_mov_b32 s77, s49 // (op 1701) reg expression
s_mov_b32 s78, s54 // (op 1701) reg expression
s_mov_b32 s79, 131072 // (op 1701) call()


v_pack_b32_f16 v20, v16, v17 // (op 50) 
v_pack_b32_f16 v21, v18, v19 // (op 50) 


buffer_store_dwordx2 v[20:21], v13, s[76:79], 0 offen // (op 50) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_116
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_116:



buffer_load_dwordx2 v[12:13], v14, s[56:59], 0 offen // (op 129) Load value
v_accvgpr_read v16, a0 // (op 125) 
v_accvgpr_read v17, a1 // (op 125) 
v_accvgpr_read v18, a2 // (op 125) 
v_accvgpr_read v19, a3 // (op 125) 
v_mul_f32 v20, v11, v16 // (op 125) 
v_mul_f32 v21, v11, v17 // (op 125) 
v_mul_f32 v22, v11, v18 // (op 125) 
v_mul_f32 v23, v11, v19 // (op 125) 

s_waitcnt vmcnt(0)
v_and_b32 v18, 65535, v12 // (op 130) 
v_lshrrev_b32 v19, 16, v12 // (op 130) 
v_lshlrev_b32 v18, 16, v18 // (op 130) 
v_lshlrev_b32 v19, 16, v19 // (op 130) 
v_mul_f32 v24, v0, v18 // (op 130) 
v_mul_f32 v25, v0, v19 // (op 130) 
v_and_b32 v18, 65535, v13 // (op 130) 
v_lshrrev_b32 v19, 16, v13 // (op 130) 
v_lshlrev_b32 v18, 16, v18 // (op 130) 
v_lshlrev_b32 v19, 16, v19 // (op 130) 
v_mul_f32 v26, v0, v18 // (op 130) 
v_mul_f32 v27, v0, v19 // (op 130) 


v_add_f32 v16, v24, v20 // (op 126) 
v_add_f32 v17, v25, v21 // (op 126) 
v_add_f32 v18, v26, v22 // (op 126) 
v_add_f32 v19, v27, v23 // (op 126) 


v_lshrrev_b32 v20, 16, v16 // (op 127) 
v_lshrrev_b32 v21, 16, v17 // (op 127) 
v_lshrrev_b32 v22, 16, v18 // (op 127) 
v_lshrrev_b32 v23, 16, v19 // (op 127) 

v_lshrrev_b32 v12, 7, v1 // (op 1716) 
s_mul_i32 s52, s16, s8 // (op 1716) 
s_add_u32 s52, s52, s0 // (op 1716) 
v_and_b32 v13, 63, v1 // (op 1716) 
s_mul_hi_u32 s53, s52, s13 // (op 1716) 
v_lshrrev_b32 v15, 4, v13 // (op 1716) 
s_sub_u32 s54, s52, s53 // (op 1716) 
v_lshlrev_b32 v16, 2, v15 // (op 1716) 

s_lshr_b32 s55, s54, 1 // (op 1716) 

v_lshl_add_u32 v15, v12, 4, v16 // (op 1716) 


s_add_u32 s54, s55, s53 // (op 1716) 


v_lshrrev_b32 v12, 2, v13 // (op 1716) 

s_lshr_b32 s53, s54, s11 // (op 1716) 

v_and_b32 v13, 3, v12 // (op 1716) 

s_and_b32 s54, s11, 2147483648 // (op 1716) 
v_and_b32 v12, 3, v1 // (op 1716) 
s_cmp_eq_i32 s54, 0 // (op 1716) 
s_mov_b32 s55, scc // (op 1716) 

v_lshl_add_u32 v16, v13, 2, v12 // (op 1716) 


s_cmp_eq_i32 s55, 0 // (op 1716) 
s_cselect_b32 s54, s52, s53 // (op 1716) 



v_lshrrev_b32 v12, 6, v1 // (op 1716) 
s_mul_hi_u32 s55, s54, s12 // (op 1716) 
v_and_b32 v13, 1, v12 // (op 1716) 

s_sub_u32 s52, s54, s55 // (op 1716) 
v_lshl_add_u32 v12, v13, 4, v16 // (op 1716) 


s_lshr_b32 s53, s52, 1 // (op 1716) 

s_add_u32 s52, s53, s55 // (op 1716) 


s_lshr_b32 s53, s52, s10 // (op 1716) 

s_and_b32 s52, s10, 2147483648 // (op 1716) 
s_cmp_eq_i32 s52, 0 // (op 1716) 
s_mov_b32 s55, scc // (op 1716) 

s_cmp_eq_i32 s55, 0 // (op 1716) 
s_cselect_b32 s52, s54, s53 // (op 1716) 


s_mul_i32 s53, s52, s14 // (op 1716) 
v_lshl_add_u32 v13, s52, 5, v15 // (op 1716) 


s_sub_u32 s52, s54, s53 // (op 1716) 


v_lshl_add_u32 v15, s52, 5, v12 // (op 1716) 


v_mul_lo_u32 v12, v15, s46 // (op 1716) 

v_add_lshl_u32 v15, v13, v12, 1 // (op 1716) 


s_mul_i32 s53, s46, s23 // (op 1718) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1718) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1718) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1718) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1718) most significant: sum
s_add_u32 s53, s53, s55 // (op 1718) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 1718) 

s_mov_b32 s76, s48 // (op 1718) reg expression
s_mov_b32 s77, s49 // (op 1718) reg expression
s_mov_b32 s78, s54 // (op 1718) reg expression
s_mov_b32 s79, 131072 // (op 1718) call()


v_pack_b32_f16 v12, v20, v21 // (op 128) 
v_pack_b32_f16 v13, v22, v23 // (op 128) 


buffer_store_dwordx2 v[12:13], v15, s[76:79], 0 offen // (op 128) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_116:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_350 // (op 350) Condition: Done, jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_350
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_350:
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_350:

s_add_u32 s0, s0, s7 // (op 265) 
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
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_SKStreamTileLoop_263 // (op 263) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_SKStreamTileLoop_263 if true)
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_263:













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
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_418 // (op 418) Condition: Top (jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_418 if false)
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_DPStreamTileLoop_418:
v_lshrrev_b32 v3, 7, v1 // (op 1550) 
s_mul_i32 s15, s16, s50 // (op 1550) 
s_add_u32 s15, s15, s0 // (op 1550) 
v_and_b32 v4, 63, v1 // (op 1550) 
s_add_u32 s6, s15, s9 // (op 1550) 

v_lshrrev_b32 v9, 4, v4 // (op 1550) 
s_mul_hi_u32 s15, s6, s13 // (op 1550) 
v_lshlrev_b32 v10, 2, v9 // (op 1550) 

s_sub_u32 s7, s6, s15 // (op 1550) 
v_lshl_add_u32 v9, v3, 4, v10 // (op 1550) 


s_lshr_b32 s8, s7, 1 // (op 1550) 

v_lshrrev_b32 v3, 2, v4 // (op 1550) 

s_add_u32 s7, s8, s15 // (op 1550) 


v_and_b32 v4, 3, v3 // (op 1550) 

s_lshr_b32 s8, s7, s11 // (op 1550) 

v_and_b32 v3, 3, v1 // (op 1550) 
s_and_b32 s7, s11, 2147483648 // (op 1550) 
v_lshl_add_u32 v10, v4, 2, v3 // (op 1550) 


s_cmp_eq_i32 s7, 0 // (op 1550) 
s_mov_b32 s15, scc // (op 1550) 

v_lshrrev_b32 v3, 6, v1 // (op 1550) 
s_cmp_eq_i32 s15, 0 // (op 1550) 
s_cselect_b32 s7, s6, s8 // (op 1550) 



v_and_b32 v4, 1, v3 // (op 1550) 

s_mul_hi_u32 s6, s7, s12 // (op 1550) 
v_lshl_add_u32 v3, v4, 4, v10 // (op 1550) 


s_sub_u32 s8, s7, s6 // (op 1550) 
s_lshr_b32 s15, s8, 1 // (op 1550) 

s_add_u32 s8, s15, s6 // (op 1550) 


s_lshr_b32 s6, s8, s10 // (op 1550) 

s_and_b32 s8, s10, 2147483648 // (op 1550) 
s_cmp_eq_i32 s8, 0 // (op 1550) 
s_mov_b32 s15, scc // (op 1550) 

s_cmp_eq_i32 s15, 0 // (op 1550) 
s_cselect_b32 s8, s7, s6 // (op 1550) 


s_mul_i32 s6, s8, s14 // (op 1550) 
v_lshl_add_u32 v4, s8, 5, v9 // (op 1550) 


s_sub_u32 s8, s7, s6 // (op 1550) 


v_lshl_add_u32 v9, s8, 5, v3 // (op 1550) 


v_mul_lo_u32 v3, v9, s4 // (op 1550) 

v_add_lshl_u32 v9, v4, v3, 1 // (op 1550) 


s_mul_i32 s19, s4, s23 // (op 1552) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 1552) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 1552) least significant
s_mul_hi_u32 s6, s4, s22 // (op 1552) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1552) most significant: sum
s_add_u32 s19, s19, s6 // (op 1552) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 1 // (op 1552) 

s_mov_b32 s56, s20 // (op 1552) reg expression
s_mov_b32 s57, s21 // (op 1552) reg expression
s_mov_b32 s58, s6 // (op 1552) reg expression
s_mov_b32 s59, 131072 // (op 1552) call()

v_lshrrev_b32 v10, 7, v1 // (op 1819) 
v_and_b32 v3, 15, v1 // (op 1819) 
v_lshl_add_u32 v4, v10, 4, v3 // (op 1819) 


v_and_b32 v3, 48, v1 // (op 1819) 
v_lshl_add_u32 v10, v4, 7, v3 // (op 1819) 


v_lshrrev_b32 v3, 6, v1 // (op 1884) 
v_and_b32 v4, 1, v3 // (op 1884) 

v_and_b32 v3, 15, v1 // (op 1884) 
v_lshl_add_u32 v12, v4, 4, v3 // (op 1884) 


v_and_b32 v3, 48, v1 // (op 1884) 
v_lshl_add_u32 v4, v12, 7, v3 // (op 1884) 


v_accvgpr_write a0, 0.00000 // (op 165) call()
v_accvgpr_write a1, 0.00000 // (op 165) call()
v_accvgpr_write a2, 0.00000 // (op 165) call()
v_accvgpr_write a3, 0.00000 // (op 165) call()
v_lshrrev_b32 v3, 3, v1 // (op 1110) 
s_mul_i32 s15, s16, s50 // (op 1110) 
s_add_u32 s15, s15, s0 // (op 1110) 
v_and_b32 v12, -8, v1 // (op 1110) 
s_add_u32 s6, s15, s9 // (op 1110) 

v_sub_u32 v13, v1, v12 // (op 1110) 

s_mul_hi_u32 s15, s6, s13 // (op 1110) 
v_lshlrev_b32 v12, 4, v13 // (op 1110) 

s_sub_u32 s7, s6, s15 // (op 1110) 
s_lshr_b32 s8, s7, 1 // (op 1110) 

s_add_u32 s7, s8, s15 // (op 1110) 


s_lshr_b32 s8, s7, s11 // (op 1110) 

s_and_b32 s7, s11, 2147483648 // (op 1110) 
s_cmp_eq_i32 s7, 0 // (op 1110) 
s_mov_b32 s15, scc // (op 1110) 

s_cmp_eq_i32 s15, 0 // (op 1110) 
s_cselect_b32 s7, s6, s8 // (op 1110) 


s_mul_hi_u32 s8, s7, s12 // (op 1110) 
s_sub_u32 s15, s7, s8 // (op 1110) 
s_lshr_b32 s18, s15, 1 // (op 1110) 

s_add_u32 s15, s18, s8 // (op 1110) 


s_lshr_b32 s8, s15, s10 // (op 1110) 

s_and_b32 s15, s10, 2147483648 // (op 1110) 
s_cmp_eq_i32 s15, 0 // (op 1110) 
s_mov_b32 s18, scc // (op 1110) 

s_cmp_eq_i32 s18, 0 // (op 1110) 
s_cselect_b32 s15, s7, s8 // (op 1110) 


v_lshl_add_u32 v13, s15, 5, v3 // (op 1110) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1110) 
s_mul_i32 s8, s7, s18 // (op 1110) 


s_sub_u32 s7, s6, s8 // (op 1110) 


v_lshl_add_u32 v3, s7, 7, v12 // (op 1110) 


v_mov_b32 v15, 0 // (op 1110) convert
v_mov_b32 v14, v3 // (op 1110) convert

v_mul_lo_u32 v17, v13, s25 // (op 1110) most significant: low of low * high
v_mul_hi_u32 v3, v13, s24 // (op 1110) most significant: high of low * low
v_mul_lo_u32 v16, v13, s24 // (op 1110) least significant: low of low * low
v_add_u32 v17, v17, v3 // (op 1110) most significant: sum

v_add_co_u32 v16, s[18:19], v16, v14 // (op 1110) least significant half
v_addc_co_u32 v17, s[18:19], v17, v15, s[18:19] // (op 1110) most significant half



s_mul_i32 s19, s24, s29 // (op 1112) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 1112) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 1112) least significant
s_mul_hi_u32 s6, s24, s28 // (op 1112) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1112) most significant: sum
s_add_u32 s19, s19, s6 // (op 1112) most significant: sum


s_mov_b32 s60, s26 // (op 1112) reg expression
s_mov_b32 s61, s27 // (op 1112) reg expression
s_mov_b32 s62, s18 // (op 1112) reg expression
s_mov_b32 s63, 131072 // (op 1112) call()

v_lshrrev_b32 v3, 3, v1 // (op 1129) 
s_mul_i32 s15, s16, s50 // (op 1129) 
s_add_u32 s15, s15, s0 // (op 1129) 
v_and_b32 v12, -8, v1 // (op 1129) 
s_add_u32 s6, s15, s9 // (op 1129) 

v_sub_u32 v13, v1, v12 // (op 1129) 

s_mul_hi_u32 s15, s6, s13 // (op 1129) 
v_lshlrev_b32 v12, 4, v13 // (op 1129) 

s_sub_u32 s7, s6, s15 // (op 1129) 
s_lshr_b32 s8, s7, 1 // (op 1129) 

s_add_u32 s7, s8, s15 // (op 1129) 


s_lshr_b32 s8, s7, s11 // (op 1129) 

s_and_b32 s7, s11, 2147483648 // (op 1129) 
s_cmp_eq_i32 s7, 0 // (op 1129) 
s_mov_b32 s15, scc // (op 1129) 

s_cmp_eq_i32 s15, 0 // (op 1129) 
s_cselect_b32 s7, s6, s8 // (op 1129) 


s_mul_hi_u32 s8, s7, s12 // (op 1129) 
s_sub_u32 s15, s7, s8 // (op 1129) 
s_lshr_b32 s18, s15, 1 // (op 1129) 

s_add_u32 s15, s18, s8 // (op 1129) 


s_lshr_b32 s8, s15, s10 // (op 1129) 

s_and_b32 s15, s10, 2147483648 // (op 1129) 
s_cmp_eq_i32 s15, 0 // (op 1129) 
s_mov_b32 s18, scc // (op 1129) 

s_cmp_eq_i32 s18, 0 // (op 1129) 
s_cselect_b32 s15, s7, s8 // (op 1129) 


v_lshl_add_u32 v13, s15, 5, v3 // (op 1129) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1129) 
s_mul_i32 s8, s7, s18 // (op 1129) 


s_sub_u32 s7, s6, s8 // (op 1129) 


v_lshl_add_u32 v3, s7, 7, v12 // (op 1129) 


v_mov_b32 v15, 0 // (op 1129) convert
v_mov_b32 v14, v3 // (op 1129) convert

v_mul_lo_u32 v19, v13, s25 // (op 1129) most significant: low of low * high
v_mul_hi_u32 v3, v13, s24 // (op 1129) most significant: high of low * low
v_mul_lo_u32 v18, v13, s24 // (op 1129) least significant: low of low * low
v_add_u32 v19, v19, v3 // (op 1129) most significant: sum

v_add_co_u32 v18, s[18:19], v18, v14 // (op 1129) least significant half
v_addc_co_u32 v19, s[18:19], v19, v15, s[18:19] // (op 1129) most significant half



s_mul_i32 s19, s24, s29 // (op 1131) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 1131) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 1131) least significant
s_mul_hi_u32 s6, s24, s28 // (op 1131) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1131) most significant: sum
s_add_u32 s19, s19, s6 // (op 1131) most significant: sum


s_mov_b32 s60, s26 // (op 1131) reg expression
s_mov_b32 s61, s27 // (op 1131) reg expression
s_mov_b32 s62, s18 // (op 1131) reg expression
s_mov_b32 s63, 131072 // (op 1131) call()

v_lshrrev_b32 v3, 3, v1 // (op 1238) 
s_mul_i32 s15, s16, s50 // (op 1238) 
s_add_u32 s15, s15, s0 // (op 1238) 
v_and_b32 v12, -8, v1 // (op 1238) 
s_add_u32 s6, s15, s9 // (op 1238) 

v_sub_u32 v13, v1, v12 // (op 1238) 

s_mul_hi_u32 s15, s6, s13 // (op 1238) 
v_lshlrev_b32 v12, 4, v13 // (op 1238) 

s_sub_u32 s7, s6, s15 // (op 1238) 
s_lshr_b32 s8, s7, 1 // (op 1238) 

s_add_u32 s7, s8, s15 // (op 1238) 


s_lshr_b32 s8, s7, s11 // (op 1238) 

s_and_b32 s7, s11, 2147483648 // (op 1238) 
s_cmp_eq_i32 s7, 0 // (op 1238) 
s_mov_b32 s15, scc // (op 1238) 

s_cmp_eq_i32 s15, 0 // (op 1238) 
s_cselect_b32 s7, s6, s8 // (op 1238) 


s_mul_hi_u32 s8, s7, s12 // (op 1238) 
s_sub_u32 s15, s7, s8 // (op 1238) 
s_lshr_b32 s18, s15, 1 // (op 1238) 

s_add_u32 s15, s18, s8 // (op 1238) 


s_lshr_b32 s8, s15, s10 // (op 1238) 

s_and_b32 s15, s10, 2147483648 // (op 1238) 
s_cmp_eq_i32 s15, 0 // (op 1238) 
s_mov_b32 s18, scc // (op 1238) 

s_cmp_eq_i32 s18, 0 // (op 1238) 
s_cselect_b32 s15, s7, s8 // (op 1238) 


s_mul_i32 s8, s15, s14 // (op 1238) 

s_sub_u32 s15, s7, s8 // (op 1238) 

v_lshl_add_u32 v13, s15, 5, v3 // (op 1238) 


v_mul_lo_u32 v15, v13, s31 // (op 1238) most significant: low of low * high
v_mul_hi_u32 v3, v13, s30 // (op 1238) most significant: high of low * low
v_mul_lo_u32 v14, v13, s30 // (op 1238) least significant: low of low * low
v_add_u32 v15, v15, v3 // (op 1238) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1238) 
s_mul_i32 s8, s7, s18 // (op 1238) 


s_sub_u32 s7, s6, s8 // (op 1238) 


v_lshl_add_u32 v3, s7, 7, v12 // (op 1238) 


v_mov_b32 v13, 0 // (op 1238) convert
v_mov_b32 v12, v3 // (op 1238) convert

v_add_co_u32 v20, s[18:19], v12, v14 // (op 1238) least significant half
v_addc_co_u32 v21, s[18:19], v13, v15, s[18:19] // (op 1238) most significant half



s_mul_i32 s19, s30, s23 // (op 1240) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 1240) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 1240) least significant
s_mul_hi_u32 s6, s30, s22 // (op 1240) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1240) most significant: sum
s_add_u32 s19, s19, s6 // (op 1240) most significant: sum


s_mov_b32 s64, s32 // (op 1240) reg expression
s_mov_b32 s65, s33 // (op 1240) reg expression
s_mov_b32 s66, s18 // (op 1240) reg expression
s_mov_b32 s67, 131072 // (op 1240) call()

v_lshrrev_b32 v3, 3, v1 // (op 1257) 
s_mul_i32 s15, s16, s50 // (op 1257) 
s_add_u32 s15, s15, s0 // (op 1257) 
v_and_b32 v12, -8, v1 // (op 1257) 
s_add_u32 s6, s15, s9 // (op 1257) 

v_sub_u32 v13, v1, v12 // (op 1257) 

s_mul_hi_u32 s15, s6, s13 // (op 1257) 
v_lshlrev_b32 v12, 4, v13 // (op 1257) 

s_sub_u32 s7, s6, s15 // (op 1257) 
s_lshr_b32 s8, s7, 1 // (op 1257) 

s_add_u32 s7, s8, s15 // (op 1257) 


s_lshr_b32 s8, s7, s11 // (op 1257) 

s_and_b32 s7, s11, 2147483648 // (op 1257) 
s_cmp_eq_i32 s7, 0 // (op 1257) 
s_mov_b32 s15, scc // (op 1257) 

s_cmp_eq_i32 s15, 0 // (op 1257) 
s_cselect_b32 s7, s6, s8 // (op 1257) 


s_mul_hi_u32 s8, s7, s12 // (op 1257) 
s_sub_u32 s15, s7, s8 // (op 1257) 
s_lshr_b32 s18, s15, 1 // (op 1257) 

s_add_u32 s15, s18, s8 // (op 1257) 


s_lshr_b32 s8, s15, s10 // (op 1257) 

s_and_b32 s15, s10, 2147483648 // (op 1257) 
s_cmp_eq_i32 s15, 0 // (op 1257) 
s_mov_b32 s18, scc // (op 1257) 

s_cmp_eq_i32 s18, 0 // (op 1257) 
s_cselect_b32 s15, s7, s8 // (op 1257) 


s_mul_i32 s8, s15, s14 // (op 1257) 

s_sub_u32 s15, s7, s8 // (op 1257) 

v_lshl_add_u32 v13, s15, 5, v3 // (op 1257) 


v_mul_lo_u32 v15, v13, s31 // (op 1257) most significant: low of low * high
v_mul_hi_u32 v3, v13, s30 // (op 1257) most significant: high of low * low
v_mul_lo_u32 v14, v13, s30 // (op 1257) least significant: low of low * low
v_add_u32 v15, v15, v3 // (op 1257) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1257) 
s_mul_i32 s8, s7, s18 // (op 1257) 


s_sub_u32 s7, s6, s8 // (op 1257) 


v_lshl_add_u32 v3, s7, 7, v12 // (op 1257) 


v_mov_b32 v13, 0 // (op 1257) convert
v_mov_b32 v12, v3 // (op 1257) convert

v_add_co_u32 v22, s[18:19], v12, v14 // (op 1257) least significant half
v_addc_co_u32 v23, s[18:19], v13, v15, s[18:19] // (op 1257) most significant half



s_mul_i32 s19, s30, s23 // (op 1259) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 1259) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 1259) least significant
s_mul_hi_u32 s6, s30, s22 // (op 1259) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1259) most significant: sum
s_add_u32 s19, s19, s6 // (op 1259) most significant: sum


s_mov_b32 s64, s32 // (op 1259) reg expression
s_mov_b32 s65, s33 // (op 1259) reg expression
s_mov_b32 s66, s18 // (op 1259) reg expression
s_mov_b32 s67, 131072 // (op 1259) call()

v_lshrrev_b32 v3, 7, v1 // (op 1366) 
s_mul_i32 s15, s16, s50 // (op 1366) 
s_add_u32 s15, s15, s0 // (op 1366) 
v_and_b32 v12, 15, v1 // (op 1366) 
s_add_u32 s6, s15, s9 // (op 1366) 

v_lshl_add_u32 v13, v3, 4, v12 // (op 1366) 


s_mul_hi_u32 s15, s6, s13 // (op 1366) 
v_and_b32 v3, 63, v1 // (op 1366) 
s_sub_u32 s7, s6, s15 // (op 1366) 
v_lshrrev_b32 v12, 4, v3 // (op 1366) 

s_lshr_b32 s8, s7, 1 // (op 1366) 

s_add_u32 s7, s8, s15 // (op 1366) 


s_lshr_b32 s8, s7, s11 // (op 1366) 

s_and_b32 s7, s11, 2147483648 // (op 1366) 
s_cmp_eq_i32 s7, 0 // (op 1366) 
s_mov_b32 s15, scc // (op 1366) 

s_cmp_eq_i32 s15, 0 // (op 1366) 
s_cselect_b32 s7, s6, s8 // (op 1366) 


s_mul_hi_u32 s8, s7, s12 // (op 1366) 
s_sub_u32 s15, s7, s8 // (op 1366) 
s_lshr_b32 s18, s15, 1 // (op 1366) 

s_add_u32 s15, s18, s8 // (op 1366) 


s_lshr_b32 s8, s15, s10 // (op 1366) 

s_and_b32 s15, s10, 2147483648 // (op 1366) 
s_cmp_eq_i32 s15, 0 // (op 1366) 
s_mov_b32 s18, scc // (op 1366) 

s_cmp_eq_i32 s18, 0 // (op 1366) 
s_cselect_b32 s15, s7, s8 // (op 1366) 


v_lshl_add_u32 v3, s15, 5, v13 // (op 1366) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1366) 
s_mul_i32 s8, s7, s18 // (op 1366) 


s_sub_u32 s7, s6, s8 // (op 1366) 


v_lshl_add_u32 v13, s7, 2, v12 // (op 1366) 


v_mul_lo_u32 v12, v3, s34 // (op 1366) 
v_add_u32 v12, v12, v13 // (op 1366) 


s_mul_i32 s19, s34, s29 // (op 1368) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1368) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1368) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1368) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1368) most significant: sum
s_add_u32 s19, s19, s6 // (op 1368) most significant: sum


s_mov_b32 s68, s36 // (op 1368) reg expression
s_mov_b32 s69, s37 // (op 1368) reg expression
s_mov_b32 s70, s18 // (op 1368) reg expression
s_mov_b32 s71, 131072 // (op 1368) call()

v_lshrrev_b32 v3, 7, v1 // (op 1385) 
s_mul_i32 s15, s16, s50 // (op 1385) 
s_add_u32 s15, s15, s0 // (op 1385) 
v_and_b32 v13, 15, v1 // (op 1385) 
s_add_u32 s6, s15, s9 // (op 1385) 

v_lshl_add_u32 v14, v3, 4, v13 // (op 1385) 


s_mul_hi_u32 s15, s6, s13 // (op 1385) 
v_and_b32 v3, 63, v1 // (op 1385) 
s_sub_u32 s7, s6, s15 // (op 1385) 
v_lshrrev_b32 v13, 4, v3 // (op 1385) 

s_lshr_b32 s8, s7, 1 // (op 1385) 

s_add_u32 s7, s8, s15 // (op 1385) 


s_lshr_b32 s8, s7, s11 // (op 1385) 

s_and_b32 s7, s11, 2147483648 // (op 1385) 
s_cmp_eq_i32 s7, 0 // (op 1385) 
s_mov_b32 s15, scc // (op 1385) 

s_cmp_eq_i32 s15, 0 // (op 1385) 
s_cselect_b32 s7, s6, s8 // (op 1385) 


s_mul_hi_u32 s8, s7, s12 // (op 1385) 
s_sub_u32 s15, s7, s8 // (op 1385) 
s_lshr_b32 s18, s15, 1 // (op 1385) 

s_add_u32 s15, s18, s8 // (op 1385) 


s_lshr_b32 s8, s15, s10 // (op 1385) 

s_and_b32 s15, s10, 2147483648 // (op 1385) 
s_cmp_eq_i32 s15, 0 // (op 1385) 
s_mov_b32 s18, scc // (op 1385) 

s_cmp_eq_i32 s18, 0 // (op 1385) 
s_cselect_b32 s15, s7, s8 // (op 1385) 


v_lshl_add_u32 v3, s15, 5, v14 // (op 1385) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1385) 
s_mul_i32 s8, s7, s18 // (op 1385) 


s_sub_u32 s7, s6, s8 // (op 1385) 


v_lshl_add_u32 v14, s7, 2, v13 // (op 1385) 


v_mul_lo_u32 v13, v3, s34 // (op 1385) 
v_add_u32 v13, v13, v14 // (op 1385) 


s_mul_i32 s19, s34, s29 // (op 1387) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1387) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1387) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1387) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1387) most significant: sum
s_add_u32 s19, s19, s6 // (op 1387) most significant: sum


s_mov_b32 s68, s36 // (op 1387) reg expression
s_mov_b32 s69, s37 // (op 1387) reg expression
s_mov_b32 s70, s18 // (op 1387) reg expression
s_mov_b32 s71, 131072 // (op 1387) call()

v_lshrrev_b32 v3, 6, v1 // (op 1494) 
s_mul_i32 s15, s16, s50 // (op 1494) 
s_add_u32 s15, s15, s0 // (op 1494) 
v_and_b32 v14, 1, v3 // (op 1494) 

s_add_u32 s6, s15, s9 // (op 1494) 

v_and_b32 v3, 15, v1 // (op 1494) 
s_mul_hi_u32 s15, s6, s13 // (op 1494) 
v_lshl_add_u32 v15, v14, 4, v3 // (op 1494) 


s_sub_u32 s7, s6, s15 // (op 1494) 
v_and_b32 v3, 63, v1 // (op 1494) 
s_lshr_b32 s8, s7, 1 // (op 1494) 

v_lshrrev_b32 v14, 4, v3 // (op 1494) 

s_add_u32 s7, s8, s15 // (op 1494) 


s_lshr_b32 s8, s7, s11 // (op 1494) 

s_and_b32 s7, s11, 2147483648 // (op 1494) 
s_cmp_eq_i32 s7, 0 // (op 1494) 
s_mov_b32 s15, scc // (op 1494) 

s_cmp_eq_i32 s15, 0 // (op 1494) 
s_cselect_b32 s7, s6, s8 // (op 1494) 


s_mul_hi_u32 s8, s7, s12 // (op 1494) 
s_sub_u32 s15, s7, s8 // (op 1494) 
s_lshr_b32 s18, s15, 1 // (op 1494) 

s_add_u32 s15, s18, s8 // (op 1494) 


s_lshr_b32 s8, s15, s10 // (op 1494) 

s_and_b32 s15, s10, 2147483648 // (op 1494) 
s_cmp_eq_i32 s15, 0 // (op 1494) 
s_mov_b32 s18, scc // (op 1494) 

s_cmp_eq_i32 s18, 0 // (op 1494) 
s_cselect_b32 s15, s7, s8 // (op 1494) 


s_mul_i32 s8, s15, s14 // (op 1494) 

s_sub_u32 s15, s7, s8 // (op 1494) 

v_lshl_add_u32 v3, s15, 5, v15 // (op 1494) 


v_mul_lo_u32 v15, v3, s38 // (op 1494) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1494) 
s_mul_i32 s8, s7, s18 // (op 1494) 


s_sub_u32 s7, s6, s8 // (op 1494) 


v_lshl_add_u32 v3, s7, 2, v14 // (op 1494) 


v_add_u32 v14, v3, v15 // (op 1494) 


s_mul_i32 s19, s38, s23 // (op 1496) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 1496) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 1496) least significant
s_mul_hi_u32 s6, s38, s22 // (op 1496) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1496) most significant: sum
s_add_u32 s19, s19, s6 // (op 1496) most significant: sum


s_mov_b32 s72, s40 // (op 1496) reg expression
s_mov_b32 s73, s41 // (op 1496) reg expression
s_mov_b32 s74, s18 // (op 1496) reg expression
s_mov_b32 s75, 131072 // (op 1496) call()

v_lshrrev_b32 v3, 6, v1 // (op 1513) 
s_mul_i32 s15, s16, s50 // (op 1513) 
s_add_u32 s15, s15, s0 // (op 1513) 
v_and_b32 v15, 1, v3 // (op 1513) 

s_add_u32 s6, s15, s9 // (op 1513) 

v_and_b32 v3, 15, v1 // (op 1513) 
s_mul_hi_u32 s15, s6, s13 // (op 1513) 
v_lshl_add_u32 v24, v15, 4, v3 // (op 1513) 


s_sub_u32 s7, s6, s15 // (op 1513) 
v_and_b32 v3, 63, v1 // (op 1513) 
s_lshr_b32 s8, s7, 1 // (op 1513) 

v_lshrrev_b32 v15, 4, v3 // (op 1513) 

s_add_u32 s7, s8, s15 // (op 1513) 


s_lshr_b32 s8, s7, s11 // (op 1513) 

s_and_b32 s7, s11, 2147483648 // (op 1513) 
s_cmp_eq_i32 s7, 0 // (op 1513) 
s_mov_b32 s15, scc // (op 1513) 

s_cmp_eq_i32 s15, 0 // (op 1513) 
s_cselect_b32 s7, s6, s8 // (op 1513) 


s_mul_hi_u32 s8, s7, s12 // (op 1513) 
s_sub_u32 s15, s7, s8 // (op 1513) 
s_lshr_b32 s18, s15, 1 // (op 1513) 

s_add_u32 s15, s18, s8 // (op 1513) 


s_lshr_b32 s8, s15, s10 // (op 1513) 

s_and_b32 s15, s10, 2147483648 // (op 1513) 
s_cmp_eq_i32 s15, 0 // (op 1513) 
s_mov_b32 s18, scc // (op 1513) 

s_cmp_eq_i32 s18, 0 // (op 1513) 
s_cselect_b32 s15, s7, s8 // (op 1513) 


s_mul_i32 s8, s15, s14 // (op 1513) 

s_sub_u32 s15, s7, s8 // (op 1513) 

v_lshl_add_u32 v3, s15, 5, v24 // (op 1513) 


v_mul_lo_u32 v24, v3, s38 // (op 1513) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1513) 
s_mul_i32 s8, s7, s18 // (op 1513) 


s_sub_u32 s7, s6, s8 // (op 1513) 


v_lshl_add_u32 v3, s7, 2, v15 // (op 1513) 


v_add_u32 v15, v3, v24 // (op 1513) 


s_mul_i32 s19, s38, s23 // (op 1515) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 1515) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 1515) least significant
s_mul_hi_u32 s6, s38, s22 // (op 1515) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1515) most significant: sum
s_add_u32 s19, s19, s6 // (op 1515) most significant: sum


s_mov_b32 s72, s40 // (op 1515) reg expression
s_mov_b32 s73, s41 // (op 1515) reg expression
s_mov_b32 s74, s18 // (op 1515) reg expression
s_mov_b32 s75, 131072 // (op 1515) call()




buffer_load_ubyte v3, v12, s[68:71], 0 offen offset: 0 // (op 833) Load value



buffer_load_ubyte v24, v14, s[72:75], 0 offen offset: 0 // (op 838) Load value
v_and_b32 v25, -8, v1 // (op 1860) 
v_sub_u32 v26, v1, v25 // (op 1860) 

v_lshlrev_b32 v25, 4, v26 // (op 1860) 

v_lshrrev_b32 v26, 3, v1 // (op 1860) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 1860) 


s_nop 0
v_readfirstlane_b32 s15, v27 // (op 1860) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2197) 



s_nop 0
s_mov_b32 m0, s15 // (op 990) reg expression
buffer_load_dwordx4 v20, s[64:67], 0 offen offset: 0 lds // (op 990) Load value direct to lds

v_and_b32 v25, -8, v1 // (op 1795) 
v_sub_u32 v26, v1, v25 // (op 1795) 

v_lshlrev_b32 v25, 4, v26 // (op 1795) 

v_lshrrev_b32 v26, 3, v1 // (op 1795) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 1795) 


s_nop 0
v_readfirstlane_b32 s15, v27 // (op 1795) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2178) 


s_nop 0
s_add_u32 s6, 4096, s15 // (op 995) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s6 // (op 995) reg expression
buffer_load_dwordx4 v16, s[60:63], 0 offen offset: 0 lds // (op 995) Load value direct to lds





buffer_load_ubyte v25, v12, s[68:71], 0 offen offset:4 // (op 849) Load value




buffer_load_ubyte v12, v14, s[72:75], 0 offen offset:4 // (op 854) Load value

v_and_b32 v14, -8, v1 // (op 1985) 
v_sub_u32 v26, v1, v14 // (op 1985) 

v_lshlrev_b32 v14, 4, v26 // (op 1985) 

v_lshrrev_b32 v26, 3, v1 // (op 1985) 
v_lshl_add_u32 v27, v26, 7, v14 // (op 1985) 


s_nop 0
v_readfirstlane_b32 s15, v27 // (op 1985) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2225) 


s_nop 0
s_add_u32 s6, 8192, s15 // (op 1000) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s6 // (op 1000) reg expression
s_mov_b32 s7, 128 // (op 1000) call()
buffer_load_dwordx4 v20, s[64:67], s7 offen offset: 0 lds // (op 1000) Load value direct to lds




v_and_b32 v14, -8, v1 // (op 1946) 
v_sub_u32 v20, v1, v14 // (op 1946) 

v_lshlrev_b32 v14, 4, v20 // (op 1946) 

v_lshrrev_b32 v20, 3, v1 // (op 1946) 
v_lshl_add_u32 v21, v20, 7, v14 // (op 1946) 


s_nop 0
v_readfirstlane_b32 s15, v21 // (op 1946) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2216) 


s_nop 0
s_add_u32 s6, 12288, s15 // (op 1002) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s6 // (op 1002) reg expression
s_mov_b32 s7, 128 // (op 1002) call()
buffer_load_dwordx4 v16, s[60:63], s7 offen offset: 0 lds // (op 1002) Load value direct to lds




s_waitcnt vmcnt(5)
s_barrier  // (op 2194) 



ds_read_b128 v[26:29], v4 // (op 879) Load local data 
ds_read_b128 v[30:33], v4 offset:64 // (op 879) Load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 2175) 



ds_read_b128 v[34:37], v10 offset:4096 // (op 876) Load local data 
ds_read_b128 v[38:41], v10 offset:4160 // (op 876) Load local data 
v_and_b32 v14, -8, v1 // (op 1767) 
v_sub_u32 v16, v1, v14 // (op 1767) 

v_lshlrev_b32 v14, 4, v16 // (op 1767) 

v_lshrrev_b32 v16, 3, v1 // (op 1767) 
v_lshl_add_u32 v17, v16, 7, v14 // (op 1767) 


s_nop 0
v_readfirstlane_b32 s15, v17 // (op 1767) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -8, v1 // (op 1832) 
v_sub_u32 v16, v1, v14 // (op 1832) 

v_lshlrev_b32 v14, 4, v16 // (op 1832) 

v_lshrrev_b32 v16, 3, v1 // (op 1832) 
v_lshl_add_u32 v17, v16, 7, v14 // (op 1832) 


s_nop 0
v_readfirstlane_b32 s6, v17 // (op 1832) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -8, v1 // (op 1931) 
v_sub_u32 v16, v1, v14 // (op 1931) 

v_lshlrev_b32 v14, 4, v16 // (op 1931) 

v_lshrrev_b32 v16, 3, v1 // (op 1931) 
v_lshl_add_u32 v17, v16, 7, v14 // (op 1931) 


s_nop 0
v_readfirstlane_b32 s7, v17 // (op 1931) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -8, v1 // (op 1970) 
v_sub_u32 v16, v1, v14 // (op 1970) 

v_lshlrev_b32 v14, 4, v16 // (op 1970) 

v_lshrrev_b32 v16, 3, v1 // (op 1970) 
v_lshl_add_u32 v17, v16, 7, v14 // (op 1970) 


s_nop 0
v_readfirstlane_b32 s8, v17 // (op 1970) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s1, 0 // (op 167) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 166) 
s_mov_b32 s42, s18 // (op 166) convert

s_and_b32 s18, s42, 4294967294 // (op 166) 

s_cmp_lt_u32 s1, s18 // (op 166) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_166 // (op 166) Condition: Top (jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_166 if false)
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_166:
s_add_u32 s1, s1, 2 // (op 168) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[26:33], a[0:3], v3, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 179) 


v_mov_b32 v14, v13 // (op 176) 

buffer_load_ubyte v3, v14, s[68:71], 0 offen offset:8 // (op 176) Load value


v_mov_b32 v16, v15 // (op 178) 

buffer_load_ubyte v24, v16, s[72:75], 0 offen offset:8 // (op 178) Load value
s_barrier  // (op 2200) 


v_mov_b64 v[20:21], v[22:23] // (op 927) 

s_mov_b32 m0, s6 // (op 927) reg expression
s_mov_b32 s18, 256 // (op 927) call()
buffer_load_dwordx4 v20, s[64:67], s18 offen offset: 0 lds // (op 927) Load value direct to lds

s_barrier  // (op 2181) 


v_mov_b64 v[42:43], v[18:19] // (op 922) 
s_add_u32 s18, 4096, s15 // (op 922) 

s_mov_b32 m0, s18 // (op 922) reg expression
s_mov_b32 s19, 256 // (op 922) call()
buffer_load_dwordx4 v42, s[60:63], s19 offen offset: 0 lds // (op 922) Load value direct to lds


s_waitcnt vmcnt(5)
s_barrier  // (op 2222) 



ds_read_b128 v[26:29], v8 offset:8192 // (op 486) Load local data 
ds_read_b128 v[30:33], v8 offset:8256 // (op 486) Load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 2213) 



ds_read_b128 v[34:37], v7 offset:12288 // (op 484) Load local data 
ds_read_b128 v[38:41], v7 offset:12352 // (op 484) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[26:33], a[0:3], v25, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 491) 



buffer_load_ubyte v25, v14, s[68:71], 0 offen offset:12 // (op 488) Load value




v_add_u32 v13, 8, v13 // (op 1377) 
buffer_load_ubyte v12, v16, s[72:75], 0 offen offset:12 // (op 490) Load value

s_barrier  // (op 2228) 
v_add_u32 v15, 8, v15 // (op 1505) 


s_add_u32 s18, 8192, s8 // (op 949) 

s_mov_b32 m0, s18 // (op 949) reg expression
s_mov_b32 s19, 384 // (op 949) call()
buffer_load_dwordx4 v20, s[64:67], s19 offen offset: 0 lds // (op 949) Load value direct to lds



s_barrier  // (op 2219) 
v_add_co_u32 v22, vcc, 256, v22 // (op 1249) least significant half


v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 1249) most significant half
s_add_u32 s18, 12288, s7 // (op 944) 

s_mov_b32 m0, s18 // (op 944) reg expression
s_mov_b32 s19, 384 // (op 944) call()
buffer_load_dwordx4 v42, s[60:63], s19 offen offset: 0 lds // (op 944) Load value direct to lds



v_add_co_u32 v18, vcc, 256, v18 // (op 1121) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 1121) most significant half
s_waitcnt vmcnt(5)
s_barrier  // (op 2206) 



ds_read_b128 v[26:29], v6 // (op 174) Load local data 
ds_read_b128 v[30:33], v6 offset:64 // (op 174) Load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 2187) 



ds_read_b128 v[34:37], v5 offset:4096 // (op 172) Load local data 
ds_read_b128 v[38:41], v5 offset:4160 // (op 172) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 166) 
s_mov_b32 s42, s18 // (op 166) convert

s_and_b32 s18, s42, 4294967294 // (op 166) 

s_cmp_lt_u32 s1, s18 // (op 166) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_166 // (op 166) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_166 if true)
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_166:










s_barrier  // (op 2269) 
v_lshrrev_b32 v3, 3, v1 // (op 1015) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1015) 
v_and_b32 v13, -8, v1 // (op 1015) 
s_and_b32 s15, s18, 4294967294 // (op 1015) 
v_sub_u32 v14, v1, v13 // (op 1015) 

s_add_u32 s6, s0, s15 // (op 1015) 

v_lshlrev_b32 v13, 4, v14 // (op 1015) 

s_mul_i32 s15, s16, s50 // (op 1015) 
s_add_u32 s15, s15, s6 // (op 1015) 

s_add_u32 s6, s15, s9 // (op 1015) 

s_mul_hi_u32 s15, s6, s13 // (op 1015) 
s_sub_u32 s7, s6, s15 // (op 1015) 
s_lshr_b32 s8, s7, 1 // (op 1015) 

s_add_u32 s7, s8, s15 // (op 1015) 


s_lshr_b32 s8, s7, s11 // (op 1015) 

s_and_b32 s7, s11, 2147483648 // (op 1015) 
s_cmp_eq_i32 s7, 0 // (op 1015) 
s_mov_b32 s15, scc // (op 1015) 

s_cmp_eq_i32 s15, 0 // (op 1015) 
s_cselect_b32 s7, s6, s8 // (op 1015) 


s_mul_hi_u32 s8, s7, s12 // (op 1015) 
s_sub_u32 s15, s7, s8 // (op 1015) 
s_lshr_b32 s42, s15, 1 // (op 1015) 

s_add_u32 s15, s42, s8 // (op 1015) 


s_lshr_b32 s8, s15, s10 // (op 1015) 

s_and_b32 s15, s10, 2147483648 // (op 1015) 
s_cmp_eq_i32 s15, 0 // (op 1015) 
s_mov_b32 s42, scc // (op 1015) 

s_cmp_eq_i32 s42, 0 // (op 1015) 
s_cselect_b32 s15, s7, s8 // (op 1015) 


s_mul_i32 s8, s7, s18 // (op 1015) 


v_lshl_add_u32 v14, s15, 5, v3 // (op 1015) 


s_sub_u32 s7, s6, s8 // (op 1015) 


v_lshl_add_u32 v3, s7, 7, v13 // (op 1015) 


v_mov_b32 v17, 0 // (op 1015) convert
v_mov_b32 v16, v3 // (op 1015) convert

v_mul_lo_u32 v19, v14, s25 // (op 1015) most significant: low of low * high
v_mul_hi_u32 v3, v14, s24 // (op 1015) most significant: high of low * low
v_mul_lo_u32 v18, v14, s24 // (op 1015) least significant: low of low * low
v_add_u32 v19, v19, v3 // (op 1015) most significant: sum

v_add_co_u32 v18, s[18:19], v18, v16 // (op 1015) least significant half
v_addc_co_u32 v19, s[18:19], v19, v17, s[18:19] // (op 1015) most significant half



s_mul_i32 s19, s24, s29 // (op 1017) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 1017) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 1017) least significant
s_mul_hi_u32 s6, s24, s28 // (op 1017) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1017) most significant: sum
s_add_u32 s19, s19, s6 // (op 1017) most significant: sum


s_mov_b32 s60, s26 // (op 1017) reg expression
s_mov_b32 s61, s27 // (op 1017) reg expression
s_mov_b32 s62, s18 // (op 1017) reg expression
s_mov_b32 s63, 131072 // (op 1017) call()

v_lshrrev_b32 v3, 3, v1 // (op 1151) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1151) 
v_and_b32 v13, -8, v1 // (op 1151) 
s_and_b32 s15, s18, 4294967294 // (op 1151) 
v_sub_u32 v14, v1, v13 // (op 1151) 

s_add_u32 s6, s0, s15 // (op 1151) 

v_lshlrev_b32 v13, 4, v14 // (op 1151) 

s_mul_i32 s15, s16, s50 // (op 1151) 
s_add_u32 s15, s15, s6 // (op 1151) 

s_add_u32 s6, s15, s9 // (op 1151) 

s_mul_hi_u32 s15, s6, s13 // (op 1151) 
s_sub_u32 s7, s6, s15 // (op 1151) 
s_lshr_b32 s8, s7, 1 // (op 1151) 

s_add_u32 s7, s8, s15 // (op 1151) 


s_lshr_b32 s8, s7, s11 // (op 1151) 

s_and_b32 s7, s11, 2147483648 // (op 1151) 
s_cmp_eq_i32 s7, 0 // (op 1151) 
s_mov_b32 s15, scc // (op 1151) 

s_cmp_eq_i32 s15, 0 // (op 1151) 
s_cselect_b32 s7, s6, s8 // (op 1151) 


s_mul_hi_u32 s8, s7, s12 // (op 1151) 
s_sub_u32 s15, s7, s8 // (op 1151) 
s_lshr_b32 s42, s15, 1 // (op 1151) 

s_add_u32 s15, s42, s8 // (op 1151) 


s_lshr_b32 s8, s15, s10 // (op 1151) 

s_and_b32 s15, s10, 2147483648 // (op 1151) 
s_cmp_eq_i32 s15, 0 // (op 1151) 
s_mov_b32 s42, scc // (op 1151) 

s_cmp_eq_i32 s42, 0 // (op 1151) 
s_cselect_b32 s15, s7, s8 // (op 1151) 


s_mul_i32 s8, s15, s14 // (op 1151) 

s_sub_u32 s15, s7, s8 // (op 1151) 

s_mul_i32 s8, s7, s18 // (op 1151) 


v_lshl_add_u32 v14, s15, 5, v3 // (op 1151) 


s_sub_u32 s7, s6, s8 // (op 1151) 


v_mul_lo_u32 v17, v14, s31 // (op 1151) most significant: low of low * high
v_mul_hi_u32 v3, v14, s30 // (op 1151) most significant: high of low * low
v_mul_lo_u32 v16, v14, s30 // (op 1151) least significant: low of low * low
v_add_u32 v17, v17, v3 // (op 1151) most significant: sum


v_lshl_add_u32 v3, s7, 7, v13 // (op 1151) 


v_mov_b32 v15, 0 // (op 1151) convert
v_mov_b32 v14, v3 // (op 1151) convert

v_add_co_u32 v20, s[18:19], v14, v16 // (op 1151) least significant half
v_addc_co_u32 v21, s[18:19], v15, v17, s[18:19] // (op 1151) most significant half



s_mul_i32 s19, s30, s23 // (op 1153) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 1153) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 1153) least significant
s_mul_hi_u32 s6, s30, s22 // (op 1153) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1153) most significant: sum
s_add_u32 s19, s19, s6 // (op 1153) most significant: sum


s_mov_b32 s64, s32 // (op 1153) reg expression
s_mov_b32 s65, s33 // (op 1153) reg expression
s_mov_b32 s66, s18 // (op 1153) reg expression
s_mov_b32 s67, 131072 // (op 1153) call()

v_lshrrev_b32 v3, 7, v1 // (op 1279) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1279) 
v_and_b32 v13, 15, v1 // (op 1279) 
s_and_b32 s15, s18, 4294967294 // (op 1279) 
v_lshl_add_u32 v14, v3, 4, v13 // (op 1279) 


s_add_u32 s6, s0, s15 // (op 1279) 

v_and_b32 v3, 63, v1 // (op 1279) 
v_lshrrev_b32 v13, 4, v3 // (op 1279) 

s_mul_i32 s15, s16, s50 // (op 1279) 
s_add_u32 s15, s15, s6 // (op 1279) 

s_add_u32 s6, s15, s9 // (op 1279) 

s_mul_hi_u32 s15, s6, s13 // (op 1279) 
s_sub_u32 s7, s6, s15 // (op 1279) 
s_lshr_b32 s8, s7, 1 // (op 1279) 

s_add_u32 s7, s8, s15 // (op 1279) 


s_lshr_b32 s8, s7, s11 // (op 1279) 

s_and_b32 s7, s11, 2147483648 // (op 1279) 
s_cmp_eq_i32 s7, 0 // (op 1279) 
s_mov_b32 s15, scc // (op 1279) 

s_cmp_eq_i32 s15, 0 // (op 1279) 
s_cselect_b32 s7, s6, s8 // (op 1279) 


s_mul_hi_u32 s8, s7, s12 // (op 1279) 
s_sub_u32 s15, s7, s8 // (op 1279) 
s_lshr_b32 s42, s15, 1 // (op 1279) 

s_add_u32 s15, s42, s8 // (op 1279) 


s_lshr_b32 s8, s15, s10 // (op 1279) 

s_and_b32 s15, s10, 2147483648 // (op 1279) 
s_cmp_eq_i32 s15, 0 // (op 1279) 
s_mov_b32 s42, scc // (op 1279) 

s_cmp_eq_i32 s42, 0 // (op 1279) 
s_cselect_b32 s15, s7, s8 // (op 1279) 


s_mul_i32 s8, s7, s18 // (op 1279) 


v_lshl_add_u32 v3, s15, 5, v14 // (op 1279) 


s_sub_u32 s7, s6, s8 // (op 1279) 


v_lshl_add_u32 v14, s7, 2, v13 // (op 1279) 


v_mul_lo_u32 v13, v3, s34 // (op 1279) 
v_add_u32 v13, v13, v14 // (op 1279) 


s_mul_i32 s19, s34, s29 // (op 1281) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1281) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1281) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1281) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1281) most significant: sum
s_add_u32 s19, s19, s6 // (op 1281) most significant: sum


s_mov_b32 s68, s36 // (op 1281) reg expression
s_mov_b32 s69, s37 // (op 1281) reg expression
s_mov_b32 s70, s18 // (op 1281) reg expression
s_mov_b32 s71, 131072 // (op 1281) call()

v_lshrrev_b32 v3, 6, v1 // (op 1407) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1407) 
v_and_b32 v14, 1, v3 // (op 1407) 

s_and_b32 s15, s18, 4294967294 // (op 1407) 
v_and_b32 v3, 15, v1 // (op 1407) 
s_add_u32 s6, s0, s15 // (op 1407) 

v_lshl_add_u32 v15, v14, 4, v3 // (op 1407) 


v_and_b32 v3, 63, v1 // (op 1407) 
s_mul_i32 s15, s16, s50 // (op 1407) 
s_add_u32 s15, s15, s6 // (op 1407) 

v_lshrrev_b32 v14, 4, v3 // (op 1407) 

s_add_u32 s6, s15, s9 // (op 1407) 

s_mul_hi_u32 s15, s6, s13 // (op 1407) 
s_sub_u32 s7, s6, s15 // (op 1407) 
s_lshr_b32 s8, s7, 1 // (op 1407) 

s_add_u32 s7, s8, s15 // (op 1407) 


s_lshr_b32 s8, s7, s11 // (op 1407) 

s_and_b32 s7, s11, 2147483648 // (op 1407) 
s_cmp_eq_i32 s7, 0 // (op 1407) 
s_mov_b32 s15, scc // (op 1407) 

s_cmp_eq_i32 s15, 0 // (op 1407) 
s_cselect_b32 s7, s6, s8 // (op 1407) 


s_mul_hi_u32 s8, s7, s12 // (op 1407) 
s_sub_u32 s15, s7, s8 // (op 1407) 
s_lshr_b32 s42, s15, 1 // (op 1407) 

s_add_u32 s15, s42, s8 // (op 1407) 


s_lshr_b32 s8, s15, s10 // (op 1407) 

s_and_b32 s15, s10, 2147483648 // (op 1407) 
s_cmp_eq_i32 s15, 0 // (op 1407) 
s_mov_b32 s42, scc // (op 1407) 

s_cmp_eq_i32 s42, 0 // (op 1407) 
s_cselect_b32 s15, s7, s8 // (op 1407) 


s_mul_i32 s8, s15, s14 // (op 1407) 

s_sub_u32 s15, s7, s8 // (op 1407) 

v_lshl_add_u32 v3, s15, 5, v15 // (op 1407) 


s_mul_i32 s8, s7, s18 // (op 1407) 


v_mul_lo_u32 v15, v3, s38 // (op 1407) 

s_sub_u32 s7, s6, s8 // (op 1407) 


v_lshl_add_u32 v3, s7, 2, v14 // (op 1407) 


v_add_u32 v14, v3, v15 // (op 1407) 


s_mul_i32 s19, s38, s23 // (op 1409) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 1409) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 1409) least significant
s_mul_hi_u32 s6, s38, s22 // (op 1409) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1409) most significant: sum
s_add_u32 s19, s19, s6 // (op 1409) most significant: sum


s_mov_b32 s72, s40 // (op 1409) reg expression
s_mov_b32 s73, s41 // (op 1409) reg expression
s_mov_b32 s74, s18 // (op 1409) reg expression
s_mov_b32 s75, 131072 // (op 1409) call()

v_and_b32 v3, -8, v1 // (op 1780) 
v_sub_u32 v15, v1, v3 // (op 1780) 

v_lshlrev_b32 v3, 4, v15 // (op 1780) 

v_lshrrev_b32 v15, 3, v1 // (op 1780) 
v_lshl_add_u32 v16, v15, 7, v3 // (op 1780) 


s_nop 0
v_readfirstlane_b32 s15, v16 // (op 1780) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -8, v1 // (op 1845) 
v_sub_u32 v15, v1, v3 // (op 1845) 

v_lshlrev_b32 v3, 4, v15 // (op 1845) 

v_lshrrev_b32 v15, 3, v1 // (op 1845) 
v_lshl_add_u32 v16, v15, 7, v3 // (op 1845) 


s_nop 0
v_readfirstlane_b32 s6, v16 // (op 1845) Wait state hazard: VALU Write Readlane Hazard

s_ashr_i64 s[18:19], s[2:3], 7 // (op 427) 
s_and_b32 s7, s18, 4294967294 // (op 427) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 426) 
s_mov_b32 s8, s18 // (op 426) convert

s_cmp_lt_u32 s7, s8 // (op 426) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_426 // (op 426) Condition: Top (jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_426 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_426:
s_barrier  // (op 2184) 
s_add_u32 s7, s7, 1 // (op 428) 


v_mov_b64 v[16:17], v[18:19] // (op 932) 
s_add_u32 s8, 4096, s15 // (op 932) 

s_mov_b32 m0, s8 // (op 932) reg expression
buffer_load_dwordx4 v16, s[60:63], 0 offen offset: 0 lds // (op 932) Load value direct to lds


v_add_co_u32 v18, vcc, 128, v18 // (op 1005) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 1005) most significant half
s_barrier  // (op 2203) 


v_mov_b64 v[16:17], v[20:21] // (op 938) 

s_mov_b32 m0, s6 // (op 938) reg expression
buffer_load_dwordx4 v16, s[64:67], 0 offen offset: 0 lds // (op 938) Load value direct to lds

v_add_co_u32 v20, vcc, 128, v20 // (op 1143) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 1143) most significant half
s_waitcnt vmcnt(1)
s_barrier  // (op 2190) 



ds_read_b128 v[22:25], v5 offset:4096 // (op 434) Load local data 
ds_read_b128 v[26:29], v5 offset:4160 // (op 434) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 2209) 



ds_read_b128 v[30:33], v6 // (op 436) Load local data 
ds_read_b128 v[34:37], v6 offset:64 // (op 436) Load local data 


v_mov_b32 v3, v13 // (op 438) 

buffer_load_ubyte v12, v3, s[68:71], 0 offen offset: 0 // (op 438) Load value

v_add_u32 v13, 4, v13 // (op 1271) 


v_mov_b32 v3, v14 // (op 440) 

buffer_load_ubyte v15, v3, s[72:75], 0 offen offset: 0 // (op 440) Load value

v_add_u32 v14, 4, v14 // (op 1399) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:37], a[0:3], v12, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 441) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 426) 
s_mov_b32 s8, s18 // (op 426) convert

s_cmp_lt_u32 s7, s8 // (op 426) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_426 // (op 426) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_426 if true)
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_426:







s_barrier  // (op 2327) 


v_cmp_eq_f32 vcc, 0.00000, v0 // (op 207) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_207 // (op 207) Condition: False, jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_207
s_nop 0
v_accvgpr_read v12, a0 // (op 215) Wait state hazard: XDL Write Hazard
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

v_lshrrev_b32 v10, 7, v1 // (op 1733) 
s_mul_i32 s15, s16, s50 // (op 1733) 
s_add_u32 s15, s15, s0 // (op 1733) 
v_and_b32 v3, 63, v1 // (op 1733) 
s_add_u32 s6, s15, s9 // (op 1733) 

v_lshrrev_b32 v4, 4, v3 // (op 1733) 
s_mul_hi_u32 s15, s6, s13 // (op 1733) 
v_lshlrev_b32 v16, 2, v4 // (op 1733) 

s_sub_u32 s7, s6, s15 // (op 1733) 
v_lshl_add_u32 v4, v10, 4, v16 // (op 1733) 


s_lshr_b32 s8, s7, 1 // (op 1733) 

v_lshrrev_b32 v10, 2, v3 // (op 1733) 

s_add_u32 s7, s8, s15 // (op 1733) 


v_and_b32 v3, 3, v10 // (op 1733) 

s_lshr_b32 s8, s7, s11 // (op 1733) 

v_and_b32 v10, 3, v1 // (op 1733) 
s_and_b32 s7, s11, 2147483648 // (op 1733) 
v_lshl_add_u32 v16, v3, 2, v10 // (op 1733) 


s_cmp_eq_i32 s7, 0 // (op 1733) 
s_mov_b32 s15, scc // (op 1733) 

v_lshrrev_b32 v3, 6, v1 // (op 1733) 
s_cmp_eq_i32 s15, 0 // (op 1733) 
s_cselect_b32 s7, s6, s8 // (op 1733) 



v_and_b32 v10, 1, v3 // (op 1733) 

s_mul_hi_u32 s6, s7, s12 // (op 1733) 
v_lshl_add_u32 v3, v10, 4, v16 // (op 1733) 


s_sub_u32 s8, s7, s6 // (op 1733) 
s_lshr_b32 s15, s8, 1 // (op 1733) 

s_add_u32 s8, s15, s6 // (op 1733) 


s_lshr_b32 s6, s8, s10 // (op 1733) 

s_and_b32 s8, s10, 2147483648 // (op 1733) 
s_cmp_eq_i32 s8, 0 // (op 1733) 
s_mov_b32 s15, scc // (op 1733) 

s_cmp_eq_i32 s15, 0 // (op 1733) 
s_cselect_b32 s8, s7, s6 // (op 1733) 


s_mul_i32 s6, s8, s14 // (op 1733) 
v_lshl_add_u32 v10, s8, 5, v4 // (op 1733) 


s_sub_u32 s8, s7, s6 // (op 1733) 


v_lshl_add_u32 v4, s8, 5, v3 // (op 1733) 


v_mul_lo_u32 v3, v4, s46 // (op 1733) 

v_add_lshl_u32 v4, v10, v3, 1 // (op 1733) 


s_mul_i32 s19, s46, s23 // (op 1735) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1735) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1735) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1735) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1735) most significant: sum
s_add_u32 s19, s19, s6 // (op 1735) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 1 // (op 1735) 

s_mov_b32 s76, s48 // (op 1735) reg expression
s_mov_b32 s77, s49 // (op 1735) reg expression
s_mov_b32 s78, s6 // (op 1735) reg expression
s_mov_b32 s79, 131072 // (op 1735) call()


v_pack_b32_f16 v16, v12, v13 // (op 218) 
v_pack_b32_f16 v17, v14, v15 // (op 218) 


buffer_store_dwordx2 v[16:17], v4, s[76:79], 0 offen // (op 218) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_207 // (op 207) Condition: Done, jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_207
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_207:



buffer_load_dwordx2 v[12:13], v9, s[56:59], 0 offen // (op 232) Load value
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

v_lshrrev_b32 v10, 7, v1 // (op 1750) 
s_mul_i32 s15, s16, s50 // (op 1750) 
s_add_u32 s15, s15, s0 // (op 1750) 
v_and_b32 v3, 63, v1 // (op 1750) 
s_add_u32 s6, s15, s9 // (op 1750) 

v_lshrrev_b32 v4, 4, v3 // (op 1750) 
s_mul_hi_u32 s15, s6, s13 // (op 1750) 
v_lshlrev_b32 v12, 2, v4 // (op 1750) 

s_sub_u32 s7, s6, s15 // (op 1750) 
v_lshl_add_u32 v4, v10, 4, v12 // (op 1750) 


s_lshr_b32 s8, s7, 1 // (op 1750) 

v_lshrrev_b32 v10, 2, v3 // (op 1750) 

s_add_u32 s7, s8, s15 // (op 1750) 


v_and_b32 v3, 3, v10 // (op 1750) 

s_lshr_b32 s8, s7, s11 // (op 1750) 

v_and_b32 v10, 3, v1 // (op 1750) 
s_and_b32 s7, s11, 2147483648 // (op 1750) 
v_lshl_add_u32 v12, v3, 2, v10 // (op 1750) 


s_cmp_eq_i32 s7, 0 // (op 1750) 
s_mov_b32 s15, scc // (op 1750) 

v_lshrrev_b32 v3, 6, v1 // (op 1750) 
s_cmp_eq_i32 s15, 0 // (op 1750) 
s_cselect_b32 s7, s6, s8 // (op 1750) 



v_and_b32 v10, 1, v3 // (op 1750) 

s_mul_hi_u32 s6, s7, s12 // (op 1750) 
v_lshl_add_u32 v3, v10, 4, v12 // (op 1750) 


s_sub_u32 s8, s7, s6 // (op 1750) 
s_lshr_b32 s15, s8, 1 // (op 1750) 

s_add_u32 s8, s15, s6 // (op 1750) 


s_lshr_b32 s6, s8, s10 // (op 1750) 

s_and_b32 s8, s10, 2147483648 // (op 1750) 
s_cmp_eq_i32 s8, 0 // (op 1750) 
s_mov_b32 s15, scc // (op 1750) 

s_cmp_eq_i32 s15, 0 // (op 1750) 
s_cselect_b32 s8, s7, s6 // (op 1750) 


s_mul_i32 s6, s8, s14 // (op 1750) 
v_lshl_add_u32 v10, s8, 5, v4 // (op 1750) 


s_sub_u32 s8, s7, s6 // (op 1750) 


v_lshl_add_u32 v4, s8, 5, v3 // (op 1750) 


v_mul_lo_u32 v3, v4, s46 // (op 1750) 

v_add_lshl_u32 v4, v10, v3, 1 // (op 1750) 


s_mul_i32 s19, s46, s23 // (op 1752) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1752) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1752) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1752) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1752) most significant: sum
s_add_u32 s19, s19, s6 // (op 1752) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 1 // (op 1752) 

s_mov_b32 s76, s48 // (op 1752) reg expression
s_mov_b32 s77, s49 // (op 1752) reg expression
s_mov_b32 s78, s6 // (op 1752) reg expression
s_mov_b32 s79, 131072 // (op 1752) call()


v_pack_b32_f16 v12, v16, v17 // (op 231) 
v_pack_b32_f16 v13, v18, v19 // (op 231) 


buffer_store_dwordx2 v[12:13], v4, s[76:79], 0 offen // (op 231) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_207:

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
s_cbranch_scc1 RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_DPStreamTileLoop_418 // (op 418) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_DPStreamTileLoop_418 if true)
RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_418:






































s_endpgm  // End of RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
.LRR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_end:
.size RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2, .LRR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_end-RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
  .amdhsa_next_free_vgpr 52
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16384 // lds bytes
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: B8
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
    macrotile: [32, 32, 128]
    threads: [256, 1, 1]
    grid: [RocRollerStreamK, One, One]
    occupancy: 8
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
  - .name: RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
    .symbol: RR_GEMM_TN_FP8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 82
    .vgpr_count: 48
    .agpr_count: 4
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

