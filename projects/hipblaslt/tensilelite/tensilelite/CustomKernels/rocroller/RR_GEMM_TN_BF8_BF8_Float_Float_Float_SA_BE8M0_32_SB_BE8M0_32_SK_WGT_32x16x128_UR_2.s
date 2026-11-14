// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2,@function
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 1668) 
v_and_b32 v3, 15, v1 // (op 1668) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1668) 


v_and_b32 v0, 48, v1 // (op 1668) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 1668) 


v_and_b32 v0, 15, v1 // (op 1733) 
v_and_b32 v4, 48, v1 // (op 1733) 
v_lshl_add_u32 v5, v0, 7, v4 // (op 1733) 


v_lshrrev_b32 v0, 6, v1 // (op 1799) 
v_and_b32 v4, 15, v1 // (op 1799) 
v_lshl_add_u32 v6, v0, 4, v4 // (op 1799) 


v_and_b32 v0, 63, v1 // (op 1799) 
v_lshrrev_b32 v4, 4, v0 // (op 1799) 

v_lshl_add_u32 v0, v6, 2, v4 // (op 1799) 


v_lshrrev_b32 v4, 6, v1 // (op 1934) 
v_and_b32 v6, 15, v1 // (op 1934) 
v_lshl_add_u32 v7, v4, 4, v6 // (op 1934) 


v_and_b32 v4, 48, v1 // (op 1934) 
v_lshl_add_u32 v6, v7, 7, v4 // (op 1934) 


v_and_b32 v4, 15, v1 // (op 1999) 
v_and_b32 v7, 48, v1 // (op 1999) 
v_lshl_add_u32 v8, v4, 7, v7 // (op 1999) 


v_lshrrev_b32 v4, 6, v1 // (op 2065) 
v_and_b32 v7, 15, v1 // (op 2065) 
v_lshl_add_u32 v9, v4, 4, v7 // (op 2065) 


v_and_b32 v4, 63, v1 // (op 2065) 
v_lshrrev_b32 v7, 4, v4 // (op 2065) 

v_lshl_add_u32 v4, v9, 2, v7 // (op 2065) 


v_lshrrev_b32 v7, 6, v1 // (op 2151) 
v_and_b32 v9, 15, v1 // (op 2151) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 2151) 


v_and_b32 v7, 48, v1 // (op 2151) 
v_lshl_add_u32 v9, v10, 7, v7 // (op 2151) 


v_and_b32 v7, 15, v1 // (op 2190) 
v_and_b32 v10, 48, v1 // (op 2190) 
v_lshl_add_u32 v11, v7, 7, v10 // (op 2190) 


v_lshrrev_b32 v7, 6, v1 // (op 2214) 
v_and_b32 v10, 15, v1 // (op 2214) 
v_lshl_add_u32 v12, v7, 4, v10 // (op 2214) 


v_and_b32 v7, 63, v1 // (op 2214) 
v_lshrrev_b32 v10, 4, v7 // (op 2214) 

v_lshl_add_u32 v7, v12, 2, v10 // (op 2214) 


v_lshrrev_b32 v10, 6, v1 // (op 2253) 
v_and_b32 v12, 15, v1 // (op 2253) 
v_lshl_add_u32 v13, v10, 4, v12 // (op 2253) 


v_and_b32 v10, 48, v1 // (op 2253) 
v_lshl_add_u32 v12, v13, 7, v10 // (op 2253) 


v_and_b32 v10, 15, v1 // (op 2292) 
v_and_b32 v13, 48, v1 // (op 2292) 
v_lshl_add_u32 v14, v10, 7, v13 // (op 2292) 


v_lshrrev_b32 v10, 6, v1 // (op 2316) 
v_and_b32 v13, 15, v1 // (op 2316) 
v_lshl_add_u32 v15, v10, 4, v13 // (op 2316) 


v_and_b32 v10, 63, v1 // (op 2316) 
v_lshrrev_b32 v13, 4, v10 // (op 2316) 

v_lshl_add_u32 v10, v15, 2, v13 // (op 2316) 


v_mov_b32 v13, s6 // (op 2) Move value
v_mov_b32 v15, s7 // (op 39) Move value


s_mov_b32 s0, 0 // (op 371) call()
s_mov_b32 s1, 0 // (op 372) call()
s_mov_b32 s6, 0 // (op 373) call()
s_mov_b32 s7, 0 // (op 374) call()
s_mov_b32 s18, 0 // (op 375) call()
s_mov_b32 s19, 0 // (op 376) call()
s_mov_b32 s0, 0 // (op 271) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 270) 
s_cmp_gt_u32 s9, s52 // (op 270) 
s_mov_b32 s53, scc // (op 270) 
s_sub_u32 s54, s9, s52 // (op 270) 

s_cmp_eq_i32 s53, 0 // (op 270) 
s_cselect_b32 s52, 0, s54 // (op 270) 


s_cmp_lt_u32 s8, s52 // (op 270) 
s_mov_b32 s53, scc // (op 270) 
s_cmp_eq_i32 s53, 0 // (op 270) 
s_cselect_b32 s54, s52, s8 // (op 270) 


s_cmp_lt_u32 s0, s54 // (op 270) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_270 // (op 270) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_270 if false)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_SKStreamTileLoop_270:
v_lshrrev_b32 v16, 6, v1 // (op 1594) 
s_mul_i32 s52, s16, s8 // (op 1594) 
s_add_u32 s52, s52, s0 // (op 1594) 
v_and_b32 v17, 63, v1 // (op 1594) 
s_mul_hi_u32 s53, s52, s13 // (op 1594) 
v_lshrrev_b32 v18, 4, v17 // (op 1594) 
s_sub_u32 s54, s52, s53 // (op 1594) 
v_lshlrev_b32 v19, 2, v18 // (op 1594) 

v_lshl_add_u32 v18, v16, 4, v19 // (op 1594) 


s_lshr_b32 s55, s54, 1 // (op 1594) 

s_add_u32 s54, s55, s53 // (op 1594) 


v_lshrrev_b32 v16, 2, v17 // (op 1594) 

s_lshr_b32 s53, s54, s11 // (op 1594) 

v_and_b32 v17, 3, v16 // (op 1594) 

s_and_b32 s54, s11, 2147483648 // (op 1594) 
v_and_b32 v16, 3, v1 // (op 1594) 
s_cmp_eq_i32 s54, 0 // (op 1594) 
s_mov_b32 s55, scc // (op 1594) 

v_lshl_add_u32 v19, v17, 2, v16 // (op 1594) 


s_cmp_eq_i32 s55, 0 // (op 1594) 
s_cselect_b32 s54, s52, s53 // (op 1594) 



s_mul_hi_u32 s52, s54, s12 // (op 1594) 
s_sub_u32 s53, s54, s52 // (op 1594) 
s_lshr_b32 s55, s53, 1 // (op 1594) 

s_add_u32 s53, s55, s52 // (op 1594) 


s_lshr_b32 s52, s53, s10 // (op 1594) 

s_and_b32 s53, s10, 2147483648 // (op 1594) 
s_cmp_eq_i32 s53, 0 // (op 1594) 
s_mov_b32 s55, scc // (op 1594) 

s_cmp_eq_i32 s55, 0 // (op 1594) 
s_cselect_b32 s53, s54, s52 // (op 1594) 


s_mul_i32 s52, s53, s14 // (op 1594) 
v_lshl_add_u32 v16, s53, 5, v18 // (op 1594) 


s_sub_u32 s53, s54, s52 // (op 1594) 


v_lshl_add_u32 v17, s53, 4, v19 // (op 1594) 


v_mul_lo_u32 v18, v17, s4 // (op 1594) 

v_add_lshl_u32 v17, v16, v18, 2 // (op 1594) 


s_mul_i32 s53, s4, s23 // (op 1596) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 1596) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 1596) least significant
s_mul_hi_u32 s55, s4, s22 // (op 1596) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1596) most significant: sum
s_add_u32 s53, s53, s55 // (op 1596) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1596) 

s_mov_b32 s56, s20 // (op 1596) reg expression
s_mov_b32 s57, s21 // (op 1596) reg expression
s_mov_b32 s58, s54 // (op 1596) reg expression
s_mov_b32 s59, 131072 // (op 1596) call()

v_lshrrev_b32 v16, 6, v1 // (op 1680) 
v_and_b32 v18, 15, v1 // (op 1680) 
v_lshl_add_u32 v19, v16, 4, v18 // (op 1680) 


v_and_b32 v16, 48, v1 // (op 1680) 
v_lshl_add_u32 v18, v19, 7, v16 // (op 1680) 


v_and_b32 v16, 15, v1 // (op 1745) 
v_and_b32 v19, 48, v1 // (op 1745) 
v_lshl_add_u32 v20, v16, 7, v19 // (op 1745) 


v_lshrrev_b32 v16, 6, v1 // (op 1811) 
v_and_b32 v19, 15, v1 // (op 1811) 
v_lshl_add_u32 v21, v16, 4, v19 // (op 1811) 


v_and_b32 v16, 63, v1 // (op 1811) 
v_lshrrev_b32 v19, 4, v16 // (op 1811) 

v_lshl_add_u32 v16, v21, 2, v19 // (op 1811) 


s_mul_i32 s6, s8, s16 // (op 275) 
s_add_u32 s6, s6, s0 // (op 275) 
s_mul_hi_u32 s52, s6, s13 // (op 276) 
s_sub_u32 s53, s6, s52 // (op 276) 
s_lshr_b32 s54, s53, 1 // (op 276) 

s_add_u32 s53, s54, s52 // (op 276) 


s_lshr_b32 s52, s53, s11 // (op 276) 

s_and_b32 s53, s11, 2147483648 // (op 276) 
s_cmp_eq_i32 s53, 0 // (op 276) 
s_mov_b32 s54, scc // (op 276) 

s_cmp_eq_i32 s54, 0 // (op 276) 
s_cselect_b32 s53, s6, s52 // (op 276) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 276) 
s_mul_i32 s52, s53, s54 // (op 276) 


s_sub_u32 s18, s6, s52 // (op 276) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 277) 
s_cmp_gt_u32 s52, s18 // (op 277) 
s_mov_b32 s54, scc // (op 277) 
s_sub_u32 s55, s52, s18 // (op 277) 

s_cmp_eq_i32 s54, 0 // (op 277) 
s_cselect_b32 s52, 0, s55 // (op 277) 


s_cmp_gt_u32 s8, s0 // (op 277) 
s_mov_b32 s53, scc // (op 277) 
s_sub_u32 s54, s8, s0 // (op 277) 
s_cmp_eq_i32 s53, 0 // (op 277) 
s_cselect_b32 s55, 0, s54 // (op 277) 


s_cmp_lt_u32 s52, s55 // (op 277) 
s_mov_b32 s53, scc // (op 277) 
s_cmp_eq_i32 s53, 0 // (op 277) 
s_cselect_b32 s54, s55, s52 // (op 277) 



s_cmp_gt_u32 s9, s6 // (op 277) 
s_mov_b32 s55, scc // (op 277) 
s_sub_u32 s52, s9, s6 // (op 277) 
s_cmp_eq_i32 s55, 0 // (op 277) 
s_cselect_b32 s53, 0, s52 // (op 277) 


s_cmp_lt_u32 s54, s53 // (op 277) 
s_mov_b32 s52, scc // (op 277) 
s_cmp_eq_i32 s52, 0 // (op 277) 
s_cselect_b32 s7, s53, s54 // (op 277) 



s_add_u32 s52, s18, s7 // (op 278) 
s_sub_u32 s19, s52, 1 // (op 278) 

v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_lshrrev_b32 v19, 3, v1 // (op 1126) 
s_mul_i32 s52, s16, s8 // (op 1126) 
s_add_u32 s52, s52, s0 // (op 1126) 
v_and_b32 v21, -8, v1 // (op 1126) 
s_mul_hi_u32 s53, s52, s13 // (op 1126) 
v_sub_u32 v22, v1, v21 // (op 1126) 

s_sub_u32 s54, s52, s53 // (op 1126) 
v_lshlrev_b32 v21, 4, v22 // (op 1126) 

s_lshr_b32 s55, s54, 1 // (op 1126) 

s_add_u32 s54, s55, s53 // (op 1126) 


s_lshr_b32 s53, s54, s11 // (op 1126) 

s_and_b32 s54, s11, 2147483648 // (op 1126) 
s_cmp_eq_i32 s54, 0 // (op 1126) 
s_mov_b32 s55, scc // (op 1126) 

s_cmp_eq_i32 s55, 0 // (op 1126) 
s_cselect_b32 s54, s52, s53 // (op 1126) 


s_mul_hi_u32 s53, s54, s12 // (op 1126) 
s_sub_u32 s55, s54, s53 // (op 1126) 
s_lshr_b32 s60, s55, 1 // (op 1126) 

s_add_u32 s55, s60, s53 // (op 1126) 


s_lshr_b32 s53, s55, s10 // (op 1126) 

s_and_b32 s55, s10, 2147483648 // (op 1126) 
s_cmp_eq_i32 s55, 0 // (op 1126) 
s_mov_b32 s60, scc // (op 1126) 

s_cmp_eq_i32 s60, 0 // (op 1126) 
s_cselect_b32 s55, s54, s53 // (op 1126) 


v_lshl_add_u32 v22, s55, 5, v19 // (op 1126) 


s_ashr_i64 s[60:61], s[2:3], 7 // (op 1126) 
s_mul_i32 s53, s54, s60 // (op 1126) 


s_sub_u32 s54, s52, s53 // (op 1126) 


v_lshl_add_u32 v19, s54, 7, v21 // (op 1126) 


v_mov_b32 v25, 0 // (op 1126) convert
v_mov_b32 v24, v19 // (op 1126) convert

v_mul_lo_u32 v27, v22, s25 // (op 1126) most significant: low of low * high
v_mul_hi_u32 v19, v22, s24 // (op 1126) most significant: high of low * low
v_mul_lo_u32 v26, v22, s24 // (op 1126) least significant: low of low * low
v_add_u32 v27, v27, v19 // (op 1126) most significant: sum

v_add_co_u32 v26, s[52:53], v26, v24 // (op 1126) least significant half
v_addc_co_u32 v27, s[52:53], v27, v25, s[52:53] // (op 1126) most significant half



s_mul_i32 s53, s24, s29 // (op 1128) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 1128) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 1128) least significant
s_mul_hi_u32 s55, s24, s28 // (op 1128) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1128) most significant: sum
s_add_u32 s53, s53, s55 // (op 1128) most significant: sum


s_mov_b32 s60, s26 // (op 1128) reg expression
s_mov_b32 s61, s27 // (op 1128) reg expression
s_mov_b32 s62, s52 // (op 1128) reg expression
s_mov_b32 s63, 131072 // (op 1128) call()

v_lshrrev_b32 v19, 3, v1 // (op 1145) 
s_mul_i32 s52, s16, s8 // (op 1145) 
s_add_u32 s52, s52, s0 // (op 1145) 
v_and_b32 v21, -8, v1 // (op 1145) 
s_mul_hi_u32 s53, s52, s13 // (op 1145) 
v_sub_u32 v22, v1, v21 // (op 1145) 

s_sub_u32 s54, s52, s53 // (op 1145) 
v_lshlrev_b32 v21, 4, v22 // (op 1145) 

s_lshr_b32 s55, s54, 1 // (op 1145) 

s_add_u32 s54, s55, s53 // (op 1145) 


s_lshr_b32 s53, s54, s11 // (op 1145) 

s_and_b32 s54, s11, 2147483648 // (op 1145) 
s_cmp_eq_i32 s54, 0 // (op 1145) 
s_mov_b32 s55, scc // (op 1145) 

s_cmp_eq_i32 s55, 0 // (op 1145) 
s_cselect_b32 s54, s52, s53 // (op 1145) 


s_mul_hi_u32 s53, s54, s12 // (op 1145) 
s_sub_u32 s55, s54, s53 // (op 1145) 
s_lshr_b32 s64, s55, 1 // (op 1145) 

s_add_u32 s55, s64, s53 // (op 1145) 


s_lshr_b32 s53, s55, s10 // (op 1145) 

s_and_b32 s55, s10, 2147483648 // (op 1145) 
s_cmp_eq_i32 s55, 0 // (op 1145) 
s_mov_b32 s64, scc // (op 1145) 

s_cmp_eq_i32 s64, 0 // (op 1145) 
s_cselect_b32 s55, s54, s53 // (op 1145) 


v_lshl_add_u32 v22, s55, 5, v19 // (op 1145) 


s_ashr_i64 s[64:65], s[2:3], 7 // (op 1145) 
s_mul_i32 s53, s54, s64 // (op 1145) 


s_sub_u32 s54, s52, s53 // (op 1145) 


v_lshl_add_u32 v19, s54, 7, v21 // (op 1145) 


v_mov_b32 v25, 0 // (op 1145) convert
v_mov_b32 v24, v19 // (op 1145) convert

v_mul_lo_u32 v29, v22, s25 // (op 1145) most significant: low of low * high
v_mul_hi_u32 v19, v22, s24 // (op 1145) most significant: high of low * low
v_mul_lo_u32 v28, v22, s24 // (op 1145) least significant: low of low * low
v_add_u32 v29, v29, v19 // (op 1145) most significant: sum

v_add_co_u32 v28, s[52:53], v28, v24 // (op 1145) least significant half
v_addc_co_u32 v29, s[52:53], v29, v25, s[52:53] // (op 1145) most significant half



s_mul_i32 s53, s24, s29 // (op 1147) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 1147) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 1147) least significant
s_mul_hi_u32 s55, s24, s28 // (op 1147) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1147) most significant: sum
s_add_u32 s53, s53, s55 // (op 1147) most significant: sum


s_mov_b32 s60, s26 // (op 1147) reg expression
s_mov_b32 s61, s27 // (op 1147) reg expression
s_mov_b32 s62, s52 // (op 1147) reg expression
s_mov_b32 s63, 131072 // (op 1147) call()

v_lshrrev_b32 v19, 3, v1 // (op 1257) 
s_mul_i32 s52, s16, s8 // (op 1257) 
s_add_u32 s52, s52, s0 // (op 1257) 
v_and_b32 v21, -8, v1 // (op 1257) 
s_mul_hi_u32 s53, s52, s13 // (op 1257) 
v_sub_u32 v22, v1, v21 // (op 1257) 

s_sub_u32 s54, s52, s53 // (op 1257) 
v_lshlrev_b32 v21, 4, v22 // (op 1257) 

s_lshr_b32 s55, s54, 1 // (op 1257) 

s_add_u32 s54, s55, s53 // (op 1257) 


s_lshr_b32 s53, s54, s11 // (op 1257) 

s_and_b32 s54, s11, 2147483648 // (op 1257) 
s_cmp_eq_i32 s54, 0 // (op 1257) 
s_mov_b32 s55, scc // (op 1257) 

s_cmp_eq_i32 s55, 0 // (op 1257) 
s_cselect_b32 s54, s52, s53 // (op 1257) 


s_mul_hi_u32 s53, s54, s12 // (op 1257) 
s_sub_u32 s55, s54, s53 // (op 1257) 
s_lshr_b32 s64, s55, 1 // (op 1257) 

s_add_u32 s55, s64, s53 // (op 1257) 


s_lshr_b32 s53, s55, s10 // (op 1257) 

s_and_b32 s55, s10, 2147483648 // (op 1257) 
s_cmp_eq_i32 s55, 0 // (op 1257) 
s_mov_b32 s64, scc // (op 1257) 

s_cmp_eq_i32 s64, 0 // (op 1257) 
s_cselect_b32 s55, s54, s53 // (op 1257) 


s_mul_i32 s53, s55, s14 // (op 1257) 

s_sub_u32 s55, s54, s53 // (op 1257) 

v_lshl_add_u32 v22, s55, 4, v19 // (op 1257) 


v_mul_lo_u32 v25, v22, s31 // (op 1257) most significant: low of low * high
v_mul_hi_u32 v19, v22, s30 // (op 1257) most significant: high of low * low
v_mul_lo_u32 v24, v22, s30 // (op 1257) least significant: low of low * low
v_add_u32 v25, v25, v19 // (op 1257) most significant: sum


s_ashr_i64 s[64:65], s[2:3], 7 // (op 1257) 
s_mul_i32 s53, s54, s64 // (op 1257) 


s_sub_u32 s54, s52, s53 // (op 1257) 


v_lshl_add_u32 v19, s54, 7, v21 // (op 1257) 


v_mov_b32 v23, 0 // (op 1257) convert
v_mov_b32 v22, v19 // (op 1257) convert

v_add_co_u32 v30, s[52:53], v22, v24 // (op 1257) least significant half
v_addc_co_u32 v31, s[52:53], v23, v25, s[52:53] // (op 1257) most significant half



s_mul_i32 s53, s30, s23 // (op 1259) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 1259) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 1259) least significant
s_mul_hi_u32 s55, s30, s22 // (op 1259) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1259) most significant: sum
s_add_u32 s53, s53, s55 // (op 1259) most significant: sum


s_mov_b32 s64, s32 // (op 1259) reg expression
s_mov_b32 s65, s33 // (op 1259) reg expression
s_mov_b32 s66, s52 // (op 1259) reg expression
s_mov_b32 s67, 131072 // (op 1259) call()

v_lshrrev_b32 v19, 3, v1 // (op 1276) 
s_mul_i32 s52, s16, s8 // (op 1276) 
s_add_u32 s52, s52, s0 // (op 1276) 
v_and_b32 v21, -8, v1 // (op 1276) 
s_mul_hi_u32 s53, s52, s13 // (op 1276) 
v_sub_u32 v22, v1, v21 // (op 1276) 

s_sub_u32 s54, s52, s53 // (op 1276) 
v_lshlrev_b32 v21, 4, v22 // (op 1276) 

s_lshr_b32 s55, s54, 1 // (op 1276) 

s_add_u32 s54, s55, s53 // (op 1276) 


s_lshr_b32 s53, s54, s11 // (op 1276) 

s_and_b32 s54, s11, 2147483648 // (op 1276) 
s_cmp_eq_i32 s54, 0 // (op 1276) 
s_mov_b32 s55, scc // (op 1276) 

s_cmp_eq_i32 s55, 0 // (op 1276) 
s_cselect_b32 s54, s52, s53 // (op 1276) 


s_mul_hi_u32 s53, s54, s12 // (op 1276) 
s_sub_u32 s55, s54, s53 // (op 1276) 
s_lshr_b32 s68, s55, 1 // (op 1276) 

s_add_u32 s55, s68, s53 // (op 1276) 


s_lshr_b32 s53, s55, s10 // (op 1276) 

s_and_b32 s55, s10, 2147483648 // (op 1276) 
s_cmp_eq_i32 s55, 0 // (op 1276) 
s_mov_b32 s68, scc // (op 1276) 

s_cmp_eq_i32 s68, 0 // (op 1276) 
s_cselect_b32 s55, s54, s53 // (op 1276) 


s_mul_i32 s53, s55, s14 // (op 1276) 

s_sub_u32 s55, s54, s53 // (op 1276) 

v_lshl_add_u32 v22, s55, 4, v19 // (op 1276) 


v_mul_lo_u32 v25, v22, s31 // (op 1276) most significant: low of low * high
v_mul_hi_u32 v19, v22, s30 // (op 1276) most significant: high of low * low
v_mul_lo_u32 v24, v22, s30 // (op 1276) least significant: low of low * low
v_add_u32 v25, v25, v19 // (op 1276) most significant: sum


s_ashr_i64 s[68:69], s[2:3], 7 // (op 1276) 
s_mul_i32 s53, s54, s68 // (op 1276) 


s_sub_u32 s54, s52, s53 // (op 1276) 


v_lshl_add_u32 v19, s54, 7, v21 // (op 1276) 


v_mov_b32 v23, 0 // (op 1276) convert
v_mov_b32 v22, v19 // (op 1276) convert

v_add_co_u32 v32, s[52:53], v22, v24 // (op 1276) least significant half
v_addc_co_u32 v33, s[52:53], v23, v25, s[52:53] // (op 1276) most significant half



s_mul_i32 s53, s30, s23 // (op 1278) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 1278) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 1278) least significant
s_mul_hi_u32 s55, s30, s22 // (op 1278) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1278) most significant: sum
s_add_u32 s53, s53, s55 // (op 1278) most significant: sum


s_mov_b32 s64, s32 // (op 1278) reg expression
s_mov_b32 s65, s33 // (op 1278) reg expression
s_mov_b32 s66, s52 // (op 1278) reg expression
s_mov_b32 s67, 131072 // (op 1278) call()

v_lshrrev_b32 v19, 2, v1 // (op 1385) 
s_mul_i32 s52, s16, s8 // (op 1385) 
s_add_u32 s52, s52, s0 // (op 1385) 
v_and_b32 v21, -4, v1 // (op 1385) 
s_mul_hi_u32 s53, s52, s13 // (op 1385) 
v_sub_u32 v22, v1, v21 // (op 1385) 

s_sub_u32 s54, s52, s53 // (op 1385) 
s_lshr_b32 s55, s54, 1 // (op 1385) 

s_add_u32 s54, s55, s53 // (op 1385) 


s_lshr_b32 s53, s54, s11 // (op 1385) 

s_and_b32 s54, s11, 2147483648 // (op 1385) 
s_cmp_eq_i32 s54, 0 // (op 1385) 
s_mov_b32 s55, scc // (op 1385) 

s_cmp_eq_i32 s55, 0 // (op 1385) 
s_cselect_b32 s54, s52, s53 // (op 1385) 


s_mul_hi_u32 s53, s54, s12 // (op 1385) 
s_sub_u32 s55, s54, s53 // (op 1385) 
s_lshr_b32 s68, s55, 1 // (op 1385) 

s_add_u32 s55, s68, s53 // (op 1385) 


s_lshr_b32 s53, s55, s10 // (op 1385) 

s_and_b32 s55, s10, 2147483648 // (op 1385) 
s_cmp_eq_i32 s55, 0 // (op 1385) 
s_mov_b32 s68, scc // (op 1385) 

s_cmp_eq_i32 s68, 0 // (op 1385) 
s_cselect_b32 s55, s54, s53 // (op 1385) 


v_lshl_add_u32 v21, s55, 5, v19 // (op 1385) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 1385) 
s_mul_i32 s53, s54, s68 // (op 1385) 


s_sub_u32 s54, s52, s53 // (op 1385) 


v_lshl_add_u32 v19, s54, 2, v22 // (op 1385) 


v_mul_lo_u32 v22, v21, s34 // (op 1385) 
v_add_u32 v22, v22, v19 // (op 1385) 


s_mul_i32 s53, s34, s29 // (op 1387) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 1387) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 1387) least significant
s_mul_hi_u32 s55, s34, s28 // (op 1387) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1387) most significant: sum
s_add_u32 s53, s53, s55 // (op 1387) most significant: sum


s_mov_b32 s68, s36 // (op 1387) reg expression
s_mov_b32 s69, s37 // (op 1387) reg expression
s_mov_b32 s70, s52 // (op 1387) reg expression
s_mov_b32 s71, 131072 // (op 1387) call()

v_lshrrev_b32 v19, 2, v1 // (op 1404) 
s_mul_i32 s52, s16, s8 // (op 1404) 
s_add_u32 s52, s52, s0 // (op 1404) 
v_and_b32 v21, -4, v1 // (op 1404) 
s_mul_hi_u32 s53, s52, s13 // (op 1404) 
v_sub_u32 v23, v1, v21 // (op 1404) 

s_sub_u32 s54, s52, s53 // (op 1404) 
s_lshr_b32 s55, s54, 1 // (op 1404) 

s_add_u32 s54, s55, s53 // (op 1404) 


s_lshr_b32 s53, s54, s11 // (op 1404) 

s_and_b32 s54, s11, 2147483648 // (op 1404) 
s_cmp_eq_i32 s54, 0 // (op 1404) 
s_mov_b32 s55, scc // (op 1404) 

s_cmp_eq_i32 s55, 0 // (op 1404) 
s_cselect_b32 s54, s52, s53 // (op 1404) 


s_mul_hi_u32 s53, s54, s12 // (op 1404) 
s_sub_u32 s55, s54, s53 // (op 1404) 
s_lshr_b32 s72, s55, 1 // (op 1404) 

s_add_u32 s55, s72, s53 // (op 1404) 


s_lshr_b32 s53, s55, s10 // (op 1404) 

s_and_b32 s55, s10, 2147483648 // (op 1404) 
s_cmp_eq_i32 s55, 0 // (op 1404) 
s_mov_b32 s72, scc // (op 1404) 

s_cmp_eq_i32 s72, 0 // (op 1404) 
s_cselect_b32 s55, s54, s53 // (op 1404) 


v_lshl_add_u32 v21, s55, 5, v19 // (op 1404) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 1404) 
s_mul_i32 s53, s54, s72 // (op 1404) 


s_sub_u32 s54, s52, s53 // (op 1404) 


v_lshl_add_u32 v19, s54, 2, v23 // (op 1404) 


v_mul_lo_u32 v23, v21, s34 // (op 1404) 
v_add_u32 v23, v23, v19 // (op 1404) 


s_mul_i32 s53, s34, s29 // (op 1406) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 1406) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 1406) least significant
s_mul_hi_u32 s55, s34, s28 // (op 1406) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1406) most significant: sum
s_add_u32 s53, s53, s55 // (op 1406) most significant: sum


s_mov_b32 s68, s36 // (op 1406) reg expression
s_mov_b32 s69, s37 // (op 1406) reg expression
s_mov_b32 s70, s52 // (op 1406) reg expression
s_mov_b32 s71, 131072 // (op 1406) call()

v_and_b32 v19, 15, v1 // (op 1513) 
s_mul_i32 s52, s16, s8 // (op 1513) 
s_add_u32 s52, s52, s0 // (op 1513) 
v_and_b32 v21, 63, v1 // (op 1513) 
s_mul_hi_u32 s53, s52, s13 // (op 1513) 
v_lshrrev_b32 v24, 4, v21 // (op 1513) 

s_sub_u32 s54, s52, s53 // (op 1513) 
s_lshr_b32 s55, s54, 1 // (op 1513) 

s_add_u32 s54, s55, s53 // (op 1513) 


s_lshr_b32 s53, s54, s11 // (op 1513) 

s_and_b32 s54, s11, 2147483648 // (op 1513) 
s_cmp_eq_i32 s54, 0 // (op 1513) 
s_mov_b32 s55, scc // (op 1513) 

s_cmp_eq_i32 s55, 0 // (op 1513) 
s_cselect_b32 s54, s52, s53 // (op 1513) 


s_mul_hi_u32 s53, s54, s12 // (op 1513) 
s_sub_u32 s55, s54, s53 // (op 1513) 
s_lshr_b32 s72, s55, 1 // (op 1513) 

s_add_u32 s55, s72, s53 // (op 1513) 


s_lshr_b32 s53, s55, s10 // (op 1513) 

s_and_b32 s55, s10, 2147483648 // (op 1513) 
s_cmp_eq_i32 s55, 0 // (op 1513) 
s_mov_b32 s72, scc // (op 1513) 

s_cmp_eq_i32 s72, 0 // (op 1513) 
s_cselect_b32 s55, s54, s53 // (op 1513) 


s_mul_i32 s53, s55, s14 // (op 1513) 

s_sub_u32 s55, s54, s53 // (op 1513) 

v_lshl_add_u32 v21, s55, 4, v19 // (op 1513) 


v_mul_lo_u32 v19, v21, s38 // (op 1513) 

s_ashr_i64 s[72:73], s[2:3], 7 // (op 1513) 
s_mul_i32 s53, s54, s72 // (op 1513) 


s_sub_u32 s54, s52, s53 // (op 1513) 


v_lshl_add_u32 v21, s54, 2, v24 // (op 1513) 


v_add_u32 v24, v21, v19 // (op 1513) 


s_mul_i32 s53, s38, s23 // (op 1515) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 1515) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 1515) least significant
s_mul_hi_u32 s55, s38, s22 // (op 1515) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1515) most significant: sum
s_add_u32 s53, s53, s55 // (op 1515) most significant: sum


s_mov_b32 s72, s40 // (op 1515) reg expression
s_mov_b32 s73, s41 // (op 1515) reg expression
s_mov_b32 s74, s52 // (op 1515) reg expression
s_mov_b32 s75, 131072 // (op 1515) call()

v_and_b32 v19, 15, v1 // (op 1532) 
s_mul_i32 s52, s16, s8 // (op 1532) 
s_add_u32 s52, s52, s0 // (op 1532) 
v_and_b32 v21, 63, v1 // (op 1532) 
s_mul_hi_u32 s53, s52, s13 // (op 1532) 
v_lshrrev_b32 v25, 4, v21 // (op 1532) 

s_sub_u32 s54, s52, s53 // (op 1532) 
s_lshr_b32 s55, s54, 1 // (op 1532) 

s_add_u32 s54, s55, s53 // (op 1532) 


s_lshr_b32 s53, s54, s11 // (op 1532) 

s_and_b32 s54, s11, 2147483648 // (op 1532) 
s_cmp_eq_i32 s54, 0 // (op 1532) 
s_mov_b32 s55, scc // (op 1532) 

s_cmp_eq_i32 s55, 0 // (op 1532) 
s_cselect_b32 s54, s52, s53 // (op 1532) 


s_mul_hi_u32 s53, s54, s12 // (op 1532) 
s_sub_u32 s55, s54, s53 // (op 1532) 
s_lshr_b32 s76, s55, 1 // (op 1532) 

s_add_u32 s55, s76, s53 // (op 1532) 


s_lshr_b32 s53, s55, s10 // (op 1532) 

s_and_b32 s55, s10, 2147483648 // (op 1532) 
s_cmp_eq_i32 s55, 0 // (op 1532) 
s_mov_b32 s76, scc // (op 1532) 

s_cmp_eq_i32 s76, 0 // (op 1532) 
s_cselect_b32 s55, s54, s53 // (op 1532) 


s_mul_i32 s53, s55, s14 // (op 1532) 

s_sub_u32 s55, s54, s53 // (op 1532) 

v_lshl_add_u32 v21, s55, 4, v19 // (op 1532) 


v_mul_lo_u32 v19, v21, s38 // (op 1532) 

s_ashr_i64 s[76:77], s[2:3], 7 // (op 1532) 
s_mul_i32 s53, s54, s76 // (op 1532) 


s_sub_u32 s54, s52, s53 // (op 1532) 


v_lshl_add_u32 v21, s54, 2, v25 // (op 1532) 


v_add_u32 v25, v21, v19 // (op 1532) 


s_mul_i32 s53, s38, s23 // (op 1534) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 1534) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 1534) least significant
s_mul_hi_u32 s55, s38, s22 // (op 1534) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1534) most significant: sum
s_add_u32 s53, s53, s55 // (op 1534) most significant: sum


s_mov_b32 s72, s40 // (op 1534) reg expression
s_mov_b32 s73, s41 // (op 1534) reg expression
s_mov_b32 s74, s52 // (op 1534) reg expression
s_mov_b32 s75, 131072 // (op 1534) call()




buffer_load_ubyte v19, v24, s[72:75], 0 offen offset: 0 // (op 794) Load value
v_and_b32 v21, -8, v1 // (op 1721) 
v_sub_u32 v34, v1, v21 // (op 1721) 

v_lshlrev_b32 v21, 4, v34 // (op 1721) 

v_lshrrev_b32 v34, 3, v1 // (op 1721) 
v_lshl_add_u32 v35, v34, 7, v21 // (op 1721) 


s_nop 0
v_readfirstlane_b32 s52, v35 // (op 1721) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2408) 



s_nop 0
s_mov_b32 m0, s52 // (op 1037) reg expression
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 1037) Load value direct to lds

v_and_b32 v21, -8, v1 // (op 1656) 
v_sub_u32 v34, v1, v21 // (op 1656) 

v_lshlrev_b32 v21, 4, v34 // (op 1656) 

v_lshrrev_b32 v34, 3, v1 // (op 1656) 
v_lshl_add_u32 v35, v34, 7, v21 // (op 1656) 


s_nop 0
v_readfirstlane_b32 s52, v35 // (op 1656) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2389) 

s_lshl_b64 s[54:55], s[24:25], 4 // (op 1042) 
v_mov_b64 v[34:35], s[54:55] // (op 1042) convert


v_mov_b64 v[36:37], v[26:27] // (op 1042) 
s_add_u32 s53, 2048, s52 // (op 1042) 
s_mov_b32 m0, s53 // (op 1042) reg expression
buffer_load_dwordx4 v36, s[60:63], 0 offen offset: 0 lds // (op 1042) Load value direct to lds
v_add_co_u32 v36, s[54:55], v36, v34 // (op 1042) least significant half
v_addc_co_u32 v37, s[54:55], 0, v37, s[54:55] // (op 1042) most significant half

s_add_u32 m0, m0, 2048 // (op 1042) 
buffer_load_dwordx4 v36, s[60:63], 0 offen offset: 0 lds // (op 1042) Load value direct to lds







buffer_load_ubyte v21, v22, s[68:71], 0 offen offset: 0 // (op 804) Load value



buffer_load_ubyte v34, v24, s[72:75], 0 offen offset:4 // (op 808) Load value

v_and_b32 v24, -8, v1 // (op 2280) 
v_sub_u32 v35, v1, v24 // (op 2280) 

v_lshlrev_b32 v24, 4, v35 // (op 2280) 

v_lshrrev_b32 v35, 3, v1 // (op 2280) 
v_lshl_add_u32 v36, v35, 7, v24 // (op 2280) 


s_nop 0
v_readfirstlane_b32 s52, v36 // (op 2280) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2538) 


s_nop 0
s_add_u32 s53, 6144, s52 // (op 1047) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s53 // (op 1047) reg expression
s_mov_b32 s54, 128 // (op 1047) call()
buffer_load_dwordx4 v30, s[64:67], s54 offen offset: 0 lds // (op 1047) Load value direct to lds




v_and_b32 v24, -8, v1 // (op 2241) 
v_sub_u32 v30, v1, v24 // (op 2241) 

v_lshlrev_b32 v24, 4, v30 // (op 2241) 

v_lshrrev_b32 v30, 3, v1 // (op 2241) 
v_lshl_add_u32 v31, v30, 7, v24 // (op 2241) 


s_nop 0
v_readfirstlane_b32 s52, v31 // (op 2241) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2529) 

s_lshl_b64 s[54:55], s[24:25], 4 // (op 1049) 
v_mov_b64 v[30:31], s[54:55] // (op 1049) convert


v_mov_b64 v[36:37], v[26:27] // (op 1049) 
s_add_u32 s53, 8192, s52 // (op 1049) 
s_mov_b32 m0, s53 // (op 1049) reg expression
s_mov_b32 s54, 128 // (op 1049) call()
buffer_load_dwordx4 v36, s[60:63], s54 offen offset: 0 lds // (op 1049) Load value direct to lds
v_add_co_u32 v36, s[76:77], v36, v30 // (op 1049) least significant half
v_addc_co_u32 v37, s[76:77], 0, v37, s[76:77] // (op 1049) most significant half

s_add_u32 m0, m0, 2048 // (op 1049) 
buffer_load_dwordx4 v36, s[60:63], s54 offen offset: 0 lds // (op 1049) Load value direct to lds









buffer_load_ubyte v24, v22, s[68:71], 0 offen offset:4 // (op 818) Load value

v_and_b32 v22, -4, v1 // (op 1787) 
v_sub_u32 v26, v1, v22 // (op 1787) 

v_and_b32 v22, 4294967292, v1 // (op 1787) 
v_add_u32 v27, v22, v26 // (op 1787) 


s_barrier  // (op 2427) 



s_waitcnt vmcnt(5)
ds_write_b8 v27, v21 offset:12288 // (op 822) Store local data 

s_barrier  // (op 2405) 



ds_read_b128 v[36:39], v20 // (op 838) Load local data 
ds_read_b128 v[40:43], v20 offset:64 // (op 838) Load local data 
s_barrier  // (op 2386) 



ds_read_b128 v[44:47], v18 offset:2048 // (op 835) Load local data 
ds_read_b128 v[48:51], v18 offset:2112 // (op 835) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2424) 



ds_read_u8 v22, v16 offset:12288 // (op 841) Load local data 
v_and_b32 v35, -8, v1 // (op 1628) 
v_sub_u32 v26, v1, v35 // (op 1628) 

v_lshlrev_b32 v27, 4, v26 // (op 1628) 

v_lshrrev_b32 v26, 3, v1 // (op 1628) 
v_lshl_add_u32 v35, v26, 7, v27 // (op 1628) 


s_nop 0
v_readfirstlane_b32 s52, v35 // (op 1628) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -8, v1 // (op 1693) 
v_sub_u32 v26, v1, v35 // (op 1693) 

v_lshlrev_b32 v27, 4, v26 // (op 1693) 

v_lshrrev_b32 v26, 3, v1 // (op 1693) 
v_lshl_add_u32 v35, v26, 7, v27 // (op 1693) 


s_nop 0
v_readfirstlane_b32 s53, v35 // (op 1693) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -4, v1 // (op 1758) 
v_sub_u32 v26, v1, v35 // (op 1758) 

v_and_b32 v27, 4294967292, v1 // (op 1758) 
v_add_u32 v35, v27, v26 // (op 1758) 


v_and_b32 v26, -8, v1 // (op 2226) 
v_sub_u32 v27, v1, v26 // (op 2226) 

v_lshlrev_b32 v26, 4, v27 // (op 2226) 

v_lshrrev_b32 v27, 3, v1 // (op 2226) 
v_lshl_add_u32 v30, v27, 7, v26 // (op 2226) 


s_nop 0
v_readfirstlane_b32 s54, v30 // (op 2226) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -8, v1 // (op 2265) 
v_sub_u32 v27, v1, v26 // (op 2265) 

v_lshlrev_b32 v26, 4, v27 // (op 2265) 

v_lshrrev_b32 v27, 3, v1 // (op 2265) 
v_lshl_add_u32 v30, v27, 7, v26 // (op 2265) 


s_nop 0
v_readfirstlane_b32 s55, v30 // (op 2265) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 2304) 
v_sub_u32 v27, v1, v26 // (op 2304) 

v_and_b32 v26, 4294967292, v1 // (op 2304) 
v_add_u32 v30, v26, v27 // (op 2304) 


s_mov_b32 s1, 0 // (op 55) call()
s_and_b32 s76, s7, 4294967294 // (op 54) 
s_cmp_lt_u32 s1, s76 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54 if false)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54:
s_add_u32 s1, s1, 2 // (op 56) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[36:43], a[0:3], v22, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 82) 


v_mov_b32 v31, v25 // (op 80) 

buffer_load_ubyte v19, v31, s[72:75], 0 offen offset:8 // (op 80) Load value
s_barrier  // (op 2411) 


v_mov_b64 v[26:27], v[32:33] // (op 976) 

s_mov_b32 m0, s53 // (op 976) reg expression
s_mov_b32 s76, 256 // (op 976) call()
buffer_load_dwordx4 v26, s[64:67], s76 offen offset: 0 lds // (op 976) Load value direct to lds

s_barrier  // (op 2392) 

s_lshl_b64 s[76:77], s[24:25], 4 // (op 971) 
v_mov_b64 v[52:53], s[76:77] // (op 971) convert


v_mov_b64 v[54:55], v[28:29] // (op 971) 
v_mov_b64 v[56:57], v[54:55] // (op 971) 
s_add_u32 s76, 2048, s52 // (op 971) 
s_mov_b32 m0, s76 // (op 971) reg expression
s_mov_b32 s77, 256 // (op 971) call()
buffer_load_dwordx4 v56, s[60:63], s77 offen offset: 0 lds // (op 971) Load value direct to lds
v_add_co_u32 v56, s[78:79], v56, v52 // (op 971) least significant half
v_addc_co_u32 v57, s[78:79], 0, v57, s[78:79] // (op 971) most significant half

s_add_u32 m0, m0, 2048 // (op 971) 
buffer_load_dwordx4 v56, s[60:63], s77 offen offset: 0 lds // (op 971) Load value direct to lds






v_mov_b32 v52, v23 // (op 17) 

buffer_load_ubyte v21, v52, s[68:71], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 2547) 



s_waitcnt vmcnt(5)
ds_write_b8 v30, v24 offset:12416 // (op 653) Store local data 
s_barrier  // (op 2535) 



ds_read_b128 v[36:39], v14 offset:6144 // (op 646) Load local data 
ds_read_b128 v[40:43], v14 offset:6208 // (op 646) Load local data 
s_barrier  // (op 2526) 



ds_read_b128 v[44:47], v12 offset:8192 // (op 644) Load local data 
ds_read_b128 v[48:51], v12 offset:8256 // (op 644) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2544) 



ds_read_u8 v22, v10 offset:12416 // (op 648) Load local data 
s_barrier  // (op 2670) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[36:43], a[0:3], v22, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 651) 



buffer_load_ubyte v34, v31, s[72:75], 0 offen offset:12 // (op 650) Load value

s_barrier  // (op 2541) 
v_add_u32 v25, 8, v25 // (op 1524) 


s_add_u32 s76, 6144, s55 // (op 1032) 

s_mov_b32 m0, s76 // (op 1032) reg expression
s_mov_b32 s77, 384 // (op 1032) call()
buffer_load_dwordx4 v26, s[64:67], s77 offen offset: 0 lds // (op 1032) Load value direct to lds



s_barrier  // (op 2532) 
v_add_co_u32 v32, vcc, 256, v32 // (op 1268) least significant half

v_addc_co_u32 v33, vcc, 0, v33, vcc // (op 1268) most significant half
s_lshl_b64 s[76:77], s[24:25], 4 // (op 1027) 
v_mov_b64 v[26:27], s[76:77] // (op 1027) convert


v_mov_b64 v[56:57], v[54:55] // (op 1027) 
s_add_u32 s76, 8192, s54 // (op 1027) 
s_mov_b32 m0, s76 // (op 1027) reg expression
s_mov_b32 s77, 384 // (op 1027) call()
buffer_load_dwordx4 v56, s[60:63], s77 offen offset: 0 lds // (op 1027) Load value direct to lds
v_add_co_u32 v56, s[78:79], v56, v26 // (op 1027) least significant half
v_addc_co_u32 v57, s[78:79], 0, v57, s[78:79] // (op 1027) most significant half

s_add_u32 m0, m0, 2048 // (op 1027) 
buffer_load_dwordx4 v56, s[60:63], s77 offen offset: 0 lds // (op 1027) Load value direct to lds








v_add_co_u32 v28, vcc, 256, v28 // (op 1137) least significant half
v_addc_co_u32 v29, vcc, 0, v29, vcc // (op 1137) most significant half
buffer_load_ubyte v24, v52, s[68:71], 0 offen offset:12 // (op 655) Load value

v_add_u32 v23, 8, v23 // (op 1396) 
s_barrier  // (op 2430) 



s_waitcnt vmcnt(5)
ds_write_b8 v35, v21 offset:12288 // (op 21) Store local data 
s_barrier  // (op 2417) 



ds_read_b128 v[36:39], v5 // (op 70) Load local data 
ds_read_b128 v[40:43], v5 offset:64 // (op 70) Load local data 
s_barrier  // (op 2398) 



ds_read_b128 v[44:47], v3 offset:2048 // (op 65) Load local data 
ds_read_b128 v[48:51], v3 offset:2112 // (op 65) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2437) 



ds_read_u8 v22, v0 offset:12288 // (op 75) Load local data 
s_and_b32 s76, s7, 4294967294 // (op 54) 
s_cmp_lt_u32 s1, s76 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54:












s_barrier  // (op 2556) 
s_and_b32 s52, s7, 4294967294 // (op 1100) 
v_lshrrev_b32 v19, 3, v1 // (op 1100) 
s_add_u32 s53, s0, s52 // (op 1100) 

v_and_b32 v21, -8, v1 // (op 1100) 
v_sub_u32 v23, v1, v21 // (op 1100) 

s_mul_i32 s52, s16, s8 // (op 1100) 
s_add_u32 s52, s52, s53 // (op 1100) 

v_lshlrev_b32 v21, 4, v23 // (op 1100) 

s_mul_hi_u32 s53, s52, s13 // (op 1100) 
s_sub_u32 s54, s52, s53 // (op 1100) 
s_lshr_b32 s55, s54, 1 // (op 1100) 

s_add_u32 s54, s55, s53 // (op 1100) 


s_lshr_b32 s53, s54, s11 // (op 1100) 

s_and_b32 s54, s11, 2147483648 // (op 1100) 
s_cmp_eq_i32 s54, 0 // (op 1100) 
s_mov_b32 s55, scc // (op 1100) 

s_cmp_eq_i32 s55, 0 // (op 1100) 
s_cselect_b32 s54, s52, s53 // (op 1100) 


s_mul_hi_u32 s53, s54, s12 // (op 1100) 
s_sub_u32 s55, s54, s53 // (op 1100) 
s_lshr_b32 s76, s55, 1 // (op 1100) 

s_add_u32 s55, s76, s53 // (op 1100) 


s_lshr_b32 s53, s55, s10 // (op 1100) 

s_and_b32 s55, s10, 2147483648 // (op 1100) 
s_cmp_eq_i32 s55, 0 // (op 1100) 
s_mov_b32 s76, scc // (op 1100) 

s_cmp_eq_i32 s76, 0 // (op 1100) 
s_cselect_b32 s55, s54, s53 // (op 1100) 


v_lshl_add_u32 v23, s55, 5, v19 // (op 1100) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1100) 
s_mul_i32 s53, s54, s76 // (op 1100) 


s_sub_u32 s54, s52, s53 // (op 1100) 


v_lshl_add_u32 v19, s54, 7, v21 // (op 1100) 


v_mov_b32 v33, 0 // (op 1100) convert
v_mov_b32 v32, v19 // (op 1100) convert

v_mul_lo_u32 v31, v23, s25 // (op 1100) most significant: low of low * high
v_mul_hi_u32 v19, v23, s24 // (op 1100) most significant: high of low * low
v_mul_lo_u32 v30, v23, s24 // (op 1100) least significant: low of low * low
v_add_u32 v31, v31, v19 // (op 1100) most significant: sum

v_add_co_u32 v30, s[52:53], v30, v32 // (op 1100) least significant half
v_addc_co_u32 v31, s[52:53], v31, v33, s[52:53] // (op 1100) most significant half



s_mul_i32 s53, s24, s29 // (op 1102) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 1102) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 1102) least significant
s_mul_hi_u32 s55, s24, s28 // (op 1102) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1102) most significant: sum
s_add_u32 s53, s53, s55 // (op 1102) most significant: sum


s_mov_b32 s60, s26 // (op 1102) reg expression
s_mov_b32 s61, s27 // (op 1102) reg expression
s_mov_b32 s62, s52 // (op 1102) reg expression
s_mov_b32 s63, 131072 // (op 1102) call()

s_and_b32 s52, s7, 4294967294 // (op 1234) 
v_lshrrev_b32 v19, 3, v1 // (op 1234) 
s_add_u32 s53, s0, s52 // (op 1234) 

v_and_b32 v21, -8, v1 // (op 1234) 
v_sub_u32 v23, v1, v21 // (op 1234) 

s_mul_i32 s52, s16, s8 // (op 1234) 
s_add_u32 s52, s52, s53 // (op 1234) 

v_lshlrev_b32 v21, 4, v23 // (op 1234) 

s_mul_hi_u32 s53, s52, s13 // (op 1234) 
s_sub_u32 s54, s52, s53 // (op 1234) 
s_lshr_b32 s55, s54, 1 // (op 1234) 

s_add_u32 s54, s55, s53 // (op 1234) 


s_lshr_b32 s53, s54, s11 // (op 1234) 

s_and_b32 s54, s11, 2147483648 // (op 1234) 
s_cmp_eq_i32 s54, 0 // (op 1234) 
s_mov_b32 s55, scc // (op 1234) 

s_cmp_eq_i32 s55, 0 // (op 1234) 
s_cselect_b32 s54, s52, s53 // (op 1234) 


s_mul_hi_u32 s53, s54, s12 // (op 1234) 
s_sub_u32 s55, s54, s53 // (op 1234) 
s_lshr_b32 s76, s55, 1 // (op 1234) 

s_add_u32 s55, s76, s53 // (op 1234) 


s_lshr_b32 s53, s55, s10 // (op 1234) 

s_and_b32 s55, s10, 2147483648 // (op 1234) 
s_cmp_eq_i32 s55, 0 // (op 1234) 
s_mov_b32 s76, scc // (op 1234) 

s_cmp_eq_i32 s76, 0 // (op 1234) 
s_cselect_b32 s55, s54, s53 // (op 1234) 


s_mul_i32 s53, s55, s14 // (op 1234) 

s_sub_u32 s55, s54, s53 // (op 1234) 

v_lshl_add_u32 v23, s55, 4, v19 // (op 1234) 


v_mul_lo_u32 v33, v23, s31 // (op 1234) most significant: low of low * high
v_mul_hi_u32 v19, v23, s30 // (op 1234) most significant: high of low * low
v_mul_lo_u32 v32, v23, s30 // (op 1234) least significant: low of low * low
v_add_u32 v33, v33, v19 // (op 1234) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1234) 
s_mul_i32 s53, s54, s76 // (op 1234) 


s_sub_u32 s54, s52, s53 // (op 1234) 


v_lshl_add_u32 v19, s54, 7, v21 // (op 1234) 


v_mov_b32 v29, 0 // (op 1234) convert
v_mov_b32 v28, v19 // (op 1234) convert

v_add_co_u32 v26, s[52:53], v28, v32 // (op 1234) least significant half
v_addc_co_u32 v27, s[52:53], v29, v33, s[52:53] // (op 1234) most significant half



s_mul_i32 s53, s30, s23 // (op 1236) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 1236) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 1236) least significant
s_mul_hi_u32 s55, s30, s22 // (op 1236) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1236) most significant: sum
s_add_u32 s53, s53, s55 // (op 1236) most significant: sum


s_mov_b32 s64, s32 // (op 1236) reg expression
s_mov_b32 s65, s33 // (op 1236) reg expression
s_mov_b32 s66, s52 // (op 1236) reg expression
s_mov_b32 s67, 131072 // (op 1236) call()

s_and_b32 s52, s7, 4294967294 // (op 1362) 
v_lshrrev_b32 v19, 2, v1 // (op 1362) 
s_add_u32 s53, s0, s52 // (op 1362) 

v_and_b32 v21, -4, v1 // (op 1362) 
v_sub_u32 v23, v1, v21 // (op 1362) 

s_mul_i32 s52, s16, s8 // (op 1362) 
s_add_u32 s52, s52, s53 // (op 1362) 

s_mul_hi_u32 s53, s52, s13 // (op 1362) 
s_sub_u32 s54, s52, s53 // (op 1362) 
s_lshr_b32 s55, s54, 1 // (op 1362) 

s_add_u32 s54, s55, s53 // (op 1362) 


s_lshr_b32 s53, s54, s11 // (op 1362) 

s_and_b32 s54, s11, 2147483648 // (op 1362) 
s_cmp_eq_i32 s54, 0 // (op 1362) 
s_mov_b32 s55, scc // (op 1362) 

s_cmp_eq_i32 s55, 0 // (op 1362) 
s_cselect_b32 s54, s52, s53 // (op 1362) 


s_mul_hi_u32 s53, s54, s12 // (op 1362) 
s_sub_u32 s55, s54, s53 // (op 1362) 
s_lshr_b32 s76, s55, 1 // (op 1362) 

s_add_u32 s55, s76, s53 // (op 1362) 


s_lshr_b32 s53, s55, s10 // (op 1362) 

s_and_b32 s55, s10, 2147483648 // (op 1362) 
s_cmp_eq_i32 s55, 0 // (op 1362) 
s_mov_b32 s76, scc // (op 1362) 

s_cmp_eq_i32 s76, 0 // (op 1362) 
s_cselect_b32 s55, s54, s53 // (op 1362) 


v_lshl_add_u32 v21, s55, 5, v19 // (op 1362) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1362) 
s_mul_i32 s53, s54, s76 // (op 1362) 


s_sub_u32 s54, s52, s53 // (op 1362) 


v_lshl_add_u32 v19, s54, 2, v23 // (op 1362) 


v_mul_lo_u32 v23, v21, s34 // (op 1362) 
v_add_u32 v23, v23, v19 // (op 1362) 


s_mul_i32 s53, s34, s29 // (op 1364) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 1364) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 1364) least significant
s_mul_hi_u32 s55, s34, s28 // (op 1364) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1364) most significant: sum
s_add_u32 s53, s53, s55 // (op 1364) most significant: sum


s_mov_b32 s68, s36 // (op 1364) reg expression
s_mov_b32 s69, s37 // (op 1364) reg expression
s_mov_b32 s70, s52 // (op 1364) reg expression
s_mov_b32 s71, 131072 // (op 1364) call()

s_and_b32 s52, s7, 4294967294 // (op 1490) 
v_and_b32 v19, 15, v1 // (op 1490) 
s_add_u32 s53, s0, s52 // (op 1490) 

v_and_b32 v21, 63, v1 // (op 1490) 
v_lshrrev_b32 v25, 4, v21 // (op 1490) 

s_mul_i32 s52, s16, s8 // (op 1490) 
s_add_u32 s52, s52, s53 // (op 1490) 

s_mul_hi_u32 s53, s52, s13 // (op 1490) 
s_sub_u32 s54, s52, s53 // (op 1490) 
s_lshr_b32 s55, s54, 1 // (op 1490) 

s_add_u32 s54, s55, s53 // (op 1490) 


s_lshr_b32 s53, s54, s11 // (op 1490) 

s_and_b32 s54, s11, 2147483648 // (op 1490) 
s_cmp_eq_i32 s54, 0 // (op 1490) 
s_mov_b32 s55, scc // (op 1490) 

s_cmp_eq_i32 s55, 0 // (op 1490) 
s_cselect_b32 s54, s52, s53 // (op 1490) 


s_mul_hi_u32 s53, s54, s12 // (op 1490) 
s_sub_u32 s55, s54, s53 // (op 1490) 
s_lshr_b32 s76, s55, 1 // (op 1490) 

s_add_u32 s55, s76, s53 // (op 1490) 


s_lshr_b32 s53, s55, s10 // (op 1490) 

s_and_b32 s55, s10, 2147483648 // (op 1490) 
s_cmp_eq_i32 s55, 0 // (op 1490) 
s_mov_b32 s76, scc // (op 1490) 

s_cmp_eq_i32 s76, 0 // (op 1490) 
s_cselect_b32 s55, s54, s53 // (op 1490) 


s_mul_i32 s53, s55, s14 // (op 1490) 

s_sub_u32 s55, s54, s53 // (op 1490) 

v_lshl_add_u32 v21, s55, 4, v19 // (op 1490) 


v_mul_lo_u32 v19, v21, s38 // (op 1490) 

s_ashr_i64 s[76:77], s[2:3], 7 // (op 1490) 
s_mul_i32 s53, s54, s76 // (op 1490) 


s_sub_u32 s54, s52, s53 // (op 1490) 


v_lshl_add_u32 v21, s54, 2, v25 // (op 1490) 


v_add_u32 v25, v21, v19 // (op 1490) 


s_mul_i32 s53, s38, s23 // (op 1492) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 1492) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 1492) least significant
s_mul_hi_u32 s55, s38, s22 // (op 1492) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1492) most significant: sum
s_add_u32 s53, s53, s55 // (op 1492) most significant: sum


s_mov_b32 s72, s40 // (op 1492) reg expression
s_mov_b32 s73, s41 // (op 1492) reg expression
s_mov_b32 s74, s52 // (op 1492) reg expression
s_mov_b32 s75, 131072 // (op 1492) call()

v_and_b32 v19, -8, v1 // (op 1641) 
v_sub_u32 v21, v1, v19 // (op 1641) 

v_lshlrev_b32 v19, 4, v21 // (op 1641) 

v_lshrrev_b32 v21, 3, v1 // (op 1641) 
v_lshl_add_u32 v35, v21, 7, v19 // (op 1641) 


s_nop 0
v_readfirstlane_b32 s52, v35 // (op 1641) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -8, v1 // (op 1706) 
v_sub_u32 v21, v1, v19 // (op 1706) 

v_lshlrev_b32 v19, 4, v21 // (op 1706) 

v_lshrrev_b32 v21, 3, v1 // (op 1706) 
v_lshl_add_u32 v35, v21, 7, v19 // (op 1706) 


s_nop 0
v_readfirstlane_b32 s53, v35 // (op 1706) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -4, v1 // (op 1771) 
v_sub_u32 v21, v1, v19 // (op 1771) 

v_and_b32 v19, 4294967292, v1 // (op 1771) 
v_add_u32 v35, v19, v21 // (op 1771) 


s_and_b32 s54, s7, 4294967294 // (op 579) 
s_cmp_lt_u32 s54, s7 // (op 578) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_578 // (op 578) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_578 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_578:
s_barrier  // (op 2395) 
s_add_u32 s54, s54, 1 // (op 580) 

s_lshl_b64 s[76:77], s[24:25], 4 // (op 1014) 
v_mov_b64 v[28:29], s[76:77] // (op 1014) convert


v_mov_b64 v[32:33], v[30:31] // (op 1014) 
v_mov_b64 v[36:37], v[32:33] // (op 1014) 
s_add_u32 s55, 2048, s52 // (op 1014) 
s_mov_b32 m0, s55 // (op 1014) reg expression
buffer_load_dwordx4 v36, s[60:63], 0 offen offset: 0 lds // (op 1014) Load value direct to lds
v_add_co_u32 v36, s[76:77], v36, v28 // (op 1014) least significant half
v_addc_co_u32 v37, s[76:77], 0, v37, s[76:77] // (op 1014) most significant half

s_add_u32 m0, m0, 2048 // (op 1014) 
buffer_load_dwordx4 v36, s[60:63], 0 offen offset: 0 lds // (op 1014) Load value direct to lds




v_add_co_u32 v30, vcc, 128, v30 // (op 1090) least significant half
v_addc_co_u32 v31, vcc, 0, v31, vcc // (op 1090) most significant half
s_barrier  // (op 2414) 


v_mov_b64 v[28:29], v[26:27] // (op 1020) 

s_mov_b32 m0, s53 // (op 1020) reg expression
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 1020) Load value direct to lds

v_add_co_u32 v26, vcc, 128, v26 // (op 1226) least significant half
v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 1226) most significant half


v_mov_b32 v19, v23 // (op 597) 

buffer_load_ubyte v24, v19, s[68:71], 0 offen offset: 0 // (op 597) Load value

v_add_u32 v23, 4, v23 // (op 1354) 
s_barrier  // (op 2433) 
s_waitcnt vmcnt(2)
s_barrier  // (op 2401) 






ds_read_b128 v[36:39], v3 offset:2048 // (op 586) Load local data 
ds_read_b128 v[40:43], v3 offset:2112 // (op 586) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v35, v24 offset:12288 // (op 595) Store local data 

s_barrier  // (op 2420) 



ds_read_b128 v[44:47], v5 // (op 588) Load local data 
ds_read_b128 v[48:51], v5 offset:64 // (op 588) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2440) 



ds_read_u8 v19, v0 offset:12288 // (op 590) Load local data 


v_mov_b32 v24, v25 // (op 592) 

buffer_load_ubyte v21, v24, s[72:75], 0 offen offset: 0 // (op 592) Load value

v_add_u32 v25, 4, v25 // (op 1482) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:51], a[0:3], v19, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 593) 




s_cmp_lt_u32 s54, s7 // (op 578) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_578 // (op 578) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_578 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_578:








s_barrier  // (op 2662) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_lg_u32 s18, 0 // (op 298) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Send_Tile_298 // (op 298) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Send_Tile_298
v_lshrrev_b32 v16, 4, v1 // (op 2092) 
v_lshlrev_b32 v18, 2, v16 // (op 2092) 

v_lshl_add_u32 v16, s16, 5, v18 // (op 2092) 

v_and_b32 v18, -16, v1 // (op 2092) 
v_sub_u32 v19, v1, v18 // (op 2092) 

v_lshl_add_u32 v18, v16, 4, v19 // (op 2092) 


v_lshlrev_b32 v16, 2, v18 // (op 2092) 

s_lshl_b32 s52, s15, 11 // (op 2094) 
s_mov_b32 s76, s42 // (op 2094) reg expression
s_mov_b32 s77, s43 // (op 2094) reg expression
s_mov_b32 s78, s52 // (op 2094) reg expression
s_mov_b32 s79, 131072 // (op 2094) call()




buffer_store_dword a0, v16, s[76:79], 0 offen sc1 // (op 295) Store value
buffer_store_dword a1, v16, s[76:79], 0 offen offset:64 sc1 // (op 295) Store value
buffer_store_dword a2, v16, s[76:79], 0 offen offset:128 sc1 // (op 295) Store value
buffer_store_dword a3, v16, s[76:79], 0 offen offset:192 sc1 // (op 295) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 299) 
v_cmp_eq_i32 vcc, v1, 0 // (op 303) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_303 // (op 303) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_303
s_mov_b32 s52, 1 // (op 301) call()
s_mov_b32 s55, 0 // (op 302) convert
s_mov_b32 s54, s16 // (op 302) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 302) 
s_add_u32 s80, s44, s54 // (op 302) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 302) End of Int64 add, unlocking SCC
s_store_dword s52, s[80:81], 0 glc // (op 302) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_303 // (op 303) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_303
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_303:
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_303:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Send_Tile_298 // (op 298) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Send_Tile_298
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Send_Tile_298:
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Send_Tile_298:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 314) 
s_mov_b32 s52, scc // (op 314) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 314) 
s_mov_b32 s53, s54 // (op 314) convert

s_sub_u32 s54, s53, 1 // (op 314) 

s_cmp_lt_u32 s19, s54 // (op 314) 
s_mov_b32 s53, scc // (op 314) 

s_and_b32 s54, s52, s53 // (op 314) 
s_cmp_lg_u32 s54, 0 // (op 314) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Receive_Tile_314 // (op 314) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Receive_Tile_314
s_mov_b32 s52, 0 // (op 280) call()
s_mul_i32 s53, s52, s8 // (op 279) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 279) 
s_mov_b32 s80, s54 // (op 279) convert

s_sub_u32 s54, s80, s19 // (op 279) 

s_sub_u32 s55, s54, 1 // (op 279) 

s_cmp_lt_u32 s53, s55 // (op 279) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_279 // (op 279) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_279 if false)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_ReceiveTileLoop_279:
s_add_u32 s53, s16, 1 // (op 2109) 
v_lshrrev_b32 v16, 4, v1 // (op 2109) 
s_add_u32 s54, s53, s52 // (op 2109) 
s_lshl_b32 s54, s54, 5 // (op 2109) 

v_lshlrev_b32 v18, 2, v16 // (op 2109) 

v_add_lshl_u32 v16, s54, v18, 4 // (op 2109) 


v_and_b32 v18, -16, v1 // (op 2109) 
v_sub_u32 v19, v1, v18 // (op 2109) 

v_add_lshl_u32 v18, v16, v19, 2 // (op 2109) 


s_lshl_b32 s53, s15, 11 // (op 2111) 
s_mov_b32 s76, s42 // (op 2111) reg expression
s_mov_b32 s77, s43 // (op 2111) reg expression
s_mov_b32 s78, s53 // (op 2111) reg expression
s_mov_b32 s79, 131072 // (op 2111) call()

RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_318:
s_add_u32 s53, s16, 1 // (op 316) 
s_add_u32 s54, s53, s52 // (op 316) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 316) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 316) 

s_add_u32 s80, s44, s54 // (op 316) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 316) End of Int64 add, unlocking SCC
s_load_dword s53, s[80:81], 0 glc // (op 316) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 318) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_318 // (op 318) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_318 if true)
s_barrier  // (op 322) 
v_cmp_eq_i32 vcc, v1, 0 // (op 321) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_321 // (op 321) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_321
s_mov_b32 s53, 0 // (op 319) call()
s_add_u32 s54, s16, 1 // (op 320) 
s_add_u32 s80, s54, s52 // (op 320) Start of Int64 add, locking SCC
s_addc_u32 s81, 0, 0 // (op 320) End of Int64 add, unlocking SCC
s_lshl_b64 s[80:81], s[80:81], 2 // (op 320) 

s_add_u32 s54, s44, s80 // (op 320) Start of Int64 add, locking SCC
s_addc_u32 s55, s45, s81 // (op 320) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 320) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_321 // (op 321) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_321
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_321:
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_321:




buffer_load_dword v20, v18, s[76:79], 0 offen // (op 333) Load value
buffer_load_dword v21, v18, s[76:79], 0 offen offset:64 // (op 333) Load value
buffer_load_dword v22, v18, s[76:79], 0 offen offset:128 // (op 333) Load value
buffer_load_dword v23, v18, s[76:79], 0 offen offset:192 // (op 333) Load value
v_accvgpr_read v24, a0 // (op 336) reg expression
v_accvgpr_read v25, a1 // (op 336) reg expression
v_accvgpr_read v26, a2 // (op 336) reg expression
v_accvgpr_read v27, a3 // (op 336) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v24, v24, v20 // (op 338) 
s_waitcnt vmcnt(2)
v_add_f32 v25, v25, v21 // (op 338) 
s_waitcnt vmcnt(1)
v_add_f32 v26, v26, v22 // (op 338) 
s_waitcnt vmcnt(0)
v_add_f32 v27, v27, v23 // (op 338) 

v_accvgpr_write a0, v24 // (op 340) reg expression
v_accvgpr_write a1, v25 // (op 340) reg expression
v_accvgpr_write a2, v26 // (op 340) reg expression
v_accvgpr_write a3, v27 // (op 340) reg expression

s_add_u32 s52, s52, 1 // (op 281) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 279) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 279) 
s_mov_b32 s80, s54 // (op 279) convert

s_sub_u32 s54, s80, s19 // (op 279) 

s_sub_u32 s55, s54, 1 // (op 279) 

s_cmp_lt_u32 s53, s55 // (op 279) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_ReceiveTileLoop_279 // (op 279) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_ReceiveTileLoop_279 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_279:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Receive_Tile_314 // (op 314) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Receive_Tile_314
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Receive_Tile_314:
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Receive_Tile_314:

s_cmp_eq_i32 s18, 0 // (op 357) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_357 // (op 357) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_357
v_cmp_eq_f32 vcc, 0.00000, v13 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123
v_accvgpr_read v18, a0 // (op 46) 
v_accvgpr_read v19, a1 // (op 46) 
v_accvgpr_read v20, a2 // (op 46) 
v_accvgpr_read v21, a3 // (op 46) 
v_mul_f32 v22, v15, v18 // (op 46) 
v_mul_f32 v23, v15, v19 // (op 46) 
v_mul_f32 v24, v15, v20 // (op 46) 
v_mul_f32 v25, v15, v21 // (op 46) 

v_lshrrev_b32 v16, 6, v1 // (op 1826) 
s_mul_i32 s52, s16, s8 // (op 1826) 
s_add_u32 s52, s52, s0 // (op 1826) 
v_and_b32 v18, 63, v1 // (op 1826) 
s_mul_hi_u32 s53, s52, s13 // (op 1826) 
v_lshrrev_b32 v19, 4, v18 // (op 1826) 
s_sub_u32 s54, s52, s53 // (op 1826) 
v_lshlrev_b32 v20, 2, v19 // (op 1826) 

s_lshr_b32 s55, s54, 1 // (op 1826) 

v_lshl_add_u32 v19, v16, 4, v20 // (op 1826) 


s_add_u32 s54, s55, s53 // (op 1826) 


v_lshrrev_b32 v16, 2, v18 // (op 1826) 

s_lshr_b32 s53, s54, s11 // (op 1826) 

v_and_b32 v18, 3, v16 // (op 1826) 

s_and_b32 s54, s11, 2147483648 // (op 1826) 
v_and_b32 v16, 3, v1 // (op 1826) 
s_cmp_eq_i32 s54, 0 // (op 1826) 
s_mov_b32 s55, scc // (op 1826) 

v_lshl_add_u32 v20, v18, 2, v16 // (op 1826) 


s_cmp_eq_i32 s55, 0 // (op 1826) 
s_cselect_b32 s54, s52, s53 // (op 1826) 



s_mul_hi_u32 s55, s54, s12 // (op 1826) 
s_sub_u32 s52, s54, s55 // (op 1826) 
s_lshr_b32 s53, s52, 1 // (op 1826) 

s_add_u32 s52, s53, s55 // (op 1826) 


s_lshr_b32 s53, s52, s10 // (op 1826) 

s_and_b32 s52, s10, 2147483648 // (op 1826) 
s_cmp_eq_i32 s52, 0 // (op 1826) 
s_mov_b32 s55, scc // (op 1826) 

s_cmp_eq_i32 s55, 0 // (op 1826) 
s_cselect_b32 s52, s54, s53 // (op 1826) 


s_mul_i32 s53, s52, s14 // (op 1826) 
v_lshl_add_u32 v16, s52, 5, v19 // (op 1826) 


s_sub_u32 s52, s54, s53 // (op 1826) 


v_lshl_add_u32 v21, s52, 4, v20 // (op 1826) 


v_mul_lo_u32 v18, v21, s46 // (op 1826) 

v_add_lshl_u32 v19, v16, v18, 2 // (op 1826) 


s_mul_i32 s53, s46, s23 // (op 1828) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1828) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1828) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1828) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1828) most significant: sum
s_add_u32 s53, s53, s55 // (op 1828) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1828) 

s_mov_b32 s76, s48 // (op 1828) reg expression
s_mov_b32 s77, s49 // (op 1828) reg expression
s_mov_b32 s78, s54 // (op 1828) reg expression
s_mov_b32 s79, 131072 // (op 1828) call()




buffer_store_dwordx4 v[22:25], v19, s[76:79], 0 offen // (op 52) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123:



buffer_load_dwordx4 v[18:21], v17, s[56:59], 0 offen // (op 135) Load value
v_accvgpr_read v22, a0 // (op 132) 
v_accvgpr_read v23, a1 // (op 132) 
v_accvgpr_read v24, a2 // (op 132) 
v_accvgpr_read v25, a3 // (op 132) 
v_mul_f32 v26, v15, v22 // (op 132) 
v_mul_f32 v27, v15, v23 // (op 132) 
v_mul_f32 v28, v15, v24 // (op 132) 
v_mul_f32 v29, v15, v25 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v22, v13, v18 // (op 136) 
v_mul_f32 v23, v13, v19 // (op 136) 
v_mul_f32 v24, v13, v20 // (op 136) 
v_mul_f32 v25, v13, v21 // (op 136) 

v_add_f32 v18, v22, v26 // (op 133) 
v_add_f32 v19, v23, v27 // (op 133) 
v_add_f32 v20, v24, v28 // (op 133) 
v_add_f32 v21, v25, v29 // (op 133) 


v_lshrrev_b32 v16, 6, v1 // (op 1843) 
s_mul_i32 s52, s16, s8 // (op 1843) 
s_add_u32 s52, s52, s0 // (op 1843) 
v_and_b32 v22, 63, v1 // (op 1843) 
s_mul_hi_u32 s53, s52, s13 // (op 1843) 
v_lshrrev_b32 v23, 4, v22 // (op 1843) 
s_sub_u32 s54, s52, s53 // (op 1843) 
v_lshlrev_b32 v24, 2, v23 // (op 1843) 

s_lshr_b32 s55, s54, 1 // (op 1843) 

v_lshl_add_u32 v23, v16, 4, v24 // (op 1843) 


s_add_u32 s54, s55, s53 // (op 1843) 


v_lshrrev_b32 v16, 2, v22 // (op 1843) 

s_lshr_b32 s53, s54, s11 // (op 1843) 

v_and_b32 v22, 3, v16 // (op 1843) 

s_and_b32 s54, s11, 2147483648 // (op 1843) 
v_and_b32 v16, 3, v1 // (op 1843) 
s_cmp_eq_i32 s54, 0 // (op 1843) 
s_mov_b32 s55, scc // (op 1843) 

v_lshl_add_u32 v24, v22, 2, v16 // (op 1843) 


s_cmp_eq_i32 s55, 0 // (op 1843) 
s_cselect_b32 s54, s52, s53 // (op 1843) 



s_mul_hi_u32 s55, s54, s12 // (op 1843) 
s_sub_u32 s52, s54, s55 // (op 1843) 
s_lshr_b32 s53, s52, 1 // (op 1843) 

s_add_u32 s52, s53, s55 // (op 1843) 


s_lshr_b32 s53, s52, s10 // (op 1843) 

s_and_b32 s52, s10, 2147483648 // (op 1843) 
s_cmp_eq_i32 s52, 0 // (op 1843) 
s_mov_b32 s55, scc // (op 1843) 

s_cmp_eq_i32 s55, 0 // (op 1843) 
s_cselect_b32 s52, s54, s53 // (op 1843) 


s_mul_i32 s53, s52, s14 // (op 1843) 
v_lshl_add_u32 v16, s52, 5, v23 // (op 1843) 


s_sub_u32 s52, s54, s53 // (op 1843) 


v_lshl_add_u32 v22, s52, 4, v24 // (op 1843) 


v_mul_lo_u32 v23, v22, s46 // (op 1843) 

v_add_lshl_u32 v22, v16, v23, 2 // (op 1843) 


s_mul_i32 s53, s46, s23 // (op 1845) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1845) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1845) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1845) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1845) most significant: sum
s_add_u32 s53, s53, s55 // (op 1845) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1845) 

s_mov_b32 s76, s48 // (op 1845) reg expression
s_mov_b32 s77, s49 // (op 1845) reg expression
s_mov_b32 s78, s54 // (op 1845) reg expression
s_mov_b32 s79, 131072 // (op 1845) call()




buffer_store_dwordx4 v[18:21], v22, s[76:79], 0 offen // (op 134) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_357 // (op 357) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_357
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_357:
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_357:

s_add_u32 s0, s0, s7 // (op 272) 
s_mul_i32 s52, s8, s16 // (op 270) 
s_cmp_gt_u32 s9, s52 // (op 270) 
s_mov_b32 s53, scc // (op 270) 
s_sub_u32 s54, s9, s52 // (op 270) 

s_cmp_eq_i32 s53, 0 // (op 270) 
s_cselect_b32 s52, 0, s54 // (op 270) 


s_cmp_lt_u32 s8, s52 // (op 270) 
s_mov_b32 s53, scc // (op 270) 
s_cmp_eq_i32 s53, 0 // (op 270) 
s_cselect_b32 s54, s52, s8 // (op 270) 


s_cmp_lt_u32 s0, s54 // (op 270) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_SKStreamTileLoop_270 // (op 270) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_SKStreamTileLoop_270 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_270:















s_mov_b32 s0, 0 // (op 411) call()
s_mov_b32 s1, 0 // (op 412) call()
s_mov_b32 s0, 0 // (op 426) call()
s_mul_i32 s15, s50, s16 // (op 425) 
s_cmp_gt_u32 s51, s15 // (op 425) 
s_mov_b32 s6, scc // (op 425) 
s_sub_u32 s7, s51, s15 // (op 425) 

s_cmp_eq_i32 s6, 0 // (op 425) 
s_cselect_b32 s8, 0, s7 // (op 425) 


s_cmp_lt_u32 s50, s8 // (op 425) 
s_mov_b32 s15, scc // (op 425) 
s_cmp_eq_i32 s15, 0 // (op 425) 
s_cselect_b32 s6, s8, s50 // (op 425) 


s_cmp_lt_u32 s0, s6 // (op 425) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_425 // (op 425) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_425 if false)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_DPStreamTileLoop_425:
v_lshrrev_b32 v0, 6, v1 // (op 1611) 
s_mul_i32 s15, s16, s50 // (op 1611) 
s_add_u32 s15, s15, s0 // (op 1611) 
v_and_b32 v3, 63, v1 // (op 1611) 
s_add_u32 s6, s15, s9 // (op 1611) 

v_lshrrev_b32 v5, 4, v3 // (op 1611) 
s_mul_hi_u32 s15, s6, s13 // (op 1611) 
v_lshlrev_b32 v10, 2, v5 // (op 1611) 

s_sub_u32 s7, s6, s15 // (op 1611) 
v_lshl_add_u32 v5, v0, 4, v10 // (op 1611) 


s_lshr_b32 s8, s7, 1 // (op 1611) 

v_lshrrev_b32 v0, 2, v3 // (op 1611) 

s_add_u32 s7, s8, s15 // (op 1611) 


v_and_b32 v3, 3, v0 // (op 1611) 

s_lshr_b32 s8, s7, s11 // (op 1611) 

v_and_b32 v0, 3, v1 // (op 1611) 
s_and_b32 s7, s11, 2147483648 // (op 1611) 
v_lshl_add_u32 v10, v3, 2, v0 // (op 1611) 


s_cmp_eq_i32 s7, 0 // (op 1611) 
s_mov_b32 s15, scc // (op 1611) 

s_cmp_eq_i32 s15, 0 // (op 1611) 
s_cselect_b32 s7, s6, s8 // (op 1611) 



s_mul_hi_u32 s6, s7, s12 // (op 1611) 
s_sub_u32 s8, s7, s6 // (op 1611) 
s_lshr_b32 s15, s8, 1 // (op 1611) 

s_add_u32 s8, s15, s6 // (op 1611) 


s_lshr_b32 s6, s8, s10 // (op 1611) 

s_and_b32 s8, s10, 2147483648 // (op 1611) 
s_cmp_eq_i32 s8, 0 // (op 1611) 
s_mov_b32 s15, scc // (op 1611) 

s_cmp_eq_i32 s15, 0 // (op 1611) 
s_cselect_b32 s8, s7, s6 // (op 1611) 


s_mul_i32 s6, s8, s14 // (op 1611) 
v_lshl_add_u32 v0, s8, 5, v5 // (op 1611) 


s_sub_u32 s8, s7, s6 // (op 1611) 


v_lshl_add_u32 v3, s8, 4, v10 // (op 1611) 


v_mul_lo_u32 v5, v3, s4 // (op 1611) 

v_add_lshl_u32 v3, v0, v5, 2 // (op 1611) 


s_mul_i32 s19, s4, s23 // (op 1613) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 1613) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 1613) least significant
s_mul_hi_u32 s6, s4, s22 // (op 1613) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1613) most significant: sum
s_add_u32 s19, s19, s6 // (op 1613) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1613) 

s_mov_b32 s56, s20 // (op 1613) reg expression
s_mov_b32 s57, s21 // (op 1613) reg expression
s_mov_b32 s58, s6 // (op 1613) reg expression
s_mov_b32 s59, 131072 // (op 1613) call()

v_lshrrev_b32 v0, 6, v1 // (op 1946) 
v_and_b32 v5, 15, v1 // (op 1946) 
v_lshl_add_u32 v10, v0, 4, v5 // (op 1946) 


v_and_b32 v0, 48, v1 // (op 1946) 
v_lshl_add_u32 v5, v10, 7, v0 // (op 1946) 


v_and_b32 v0, 15, v1 // (op 2011) 
v_and_b32 v10, 48, v1 // (op 2011) 
v_lshl_add_u32 v12, v0, 7, v10 // (op 2011) 


v_lshrrev_b32 v0, 6, v1 // (op 2077) 
v_and_b32 v10, 15, v1 // (op 2077) 
v_lshl_add_u32 v14, v0, 4, v10 // (op 2077) 


v_and_b32 v0, 63, v1 // (op 2077) 
v_lshrrev_b32 v10, 4, v0 // (op 2077) 

v_lshl_add_u32 v0, v14, 2, v10 // (op 2077) 


v_accvgpr_write a0, 0.00000 // (op 170) call()
v_accvgpr_write a1, 0.00000 // (op 170) call()
v_accvgpr_write a2, 0.00000 // (op 170) call()
v_accvgpr_write a3, 0.00000 // (op 170) call()
v_lshrrev_b32 v10, 3, v1 // (op 1171) 
s_mul_i32 s15, s16, s50 // (op 1171) 
s_add_u32 s15, s15, s0 // (op 1171) 
v_and_b32 v14, -8, v1 // (op 1171) 
s_add_u32 s6, s15, s9 // (op 1171) 

v_sub_u32 v16, v1, v14 // (op 1171) 

s_mul_hi_u32 s15, s6, s13 // (op 1171) 
v_lshlrev_b32 v14, 4, v16 // (op 1171) 

s_sub_u32 s7, s6, s15 // (op 1171) 
s_lshr_b32 s8, s7, 1 // (op 1171) 

s_add_u32 s7, s8, s15 // (op 1171) 


s_lshr_b32 s8, s7, s11 // (op 1171) 

s_and_b32 s7, s11, 2147483648 // (op 1171) 
s_cmp_eq_i32 s7, 0 // (op 1171) 
s_mov_b32 s15, scc // (op 1171) 

s_cmp_eq_i32 s15, 0 // (op 1171) 
s_cselect_b32 s7, s6, s8 // (op 1171) 


s_mul_hi_u32 s8, s7, s12 // (op 1171) 
s_sub_u32 s15, s7, s8 // (op 1171) 
s_lshr_b32 s18, s15, 1 // (op 1171) 

s_add_u32 s15, s18, s8 // (op 1171) 


s_lshr_b32 s8, s15, s10 // (op 1171) 

s_and_b32 s15, s10, 2147483648 // (op 1171) 
s_cmp_eq_i32 s15, 0 // (op 1171) 
s_mov_b32 s18, scc // (op 1171) 

s_cmp_eq_i32 s18, 0 // (op 1171) 
s_cselect_b32 s15, s7, s8 // (op 1171) 


v_lshl_add_u32 v16, s15, 5, v10 // (op 1171) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1171) 
s_mul_i32 s8, s7, s18 // (op 1171) 


s_sub_u32 s7, s6, s8 // (op 1171) 


v_lshl_add_u32 v10, s7, 7, v14 // (op 1171) 


v_mov_b32 v19, 0 // (op 1171) convert
v_mov_b32 v18, v10 // (op 1171) convert

v_mul_lo_u32 v21, v16, s25 // (op 1171) most significant: low of low * high
v_mul_hi_u32 v10, v16, s24 // (op 1171) most significant: high of low * low
v_mul_lo_u32 v20, v16, s24 // (op 1171) least significant: low of low * low
v_add_u32 v21, v21, v10 // (op 1171) most significant: sum

v_add_co_u32 v20, s[18:19], v20, v18 // (op 1171) least significant half
v_addc_co_u32 v21, s[18:19], v21, v19, s[18:19] // (op 1171) most significant half



s_mul_i32 s19, s24, s29 // (op 1173) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 1173) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 1173) least significant
s_mul_hi_u32 s6, s24, s28 // (op 1173) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1173) most significant: sum
s_add_u32 s19, s19, s6 // (op 1173) most significant: sum


s_mov_b32 s60, s26 // (op 1173) reg expression
s_mov_b32 s61, s27 // (op 1173) reg expression
s_mov_b32 s62, s18 // (op 1173) reg expression
s_mov_b32 s63, 131072 // (op 1173) call()

v_lshrrev_b32 v10, 3, v1 // (op 1190) 
s_mul_i32 s15, s16, s50 // (op 1190) 
s_add_u32 s15, s15, s0 // (op 1190) 
v_and_b32 v14, -8, v1 // (op 1190) 
s_add_u32 s6, s15, s9 // (op 1190) 

v_sub_u32 v16, v1, v14 // (op 1190) 

s_mul_hi_u32 s15, s6, s13 // (op 1190) 
v_lshlrev_b32 v14, 4, v16 // (op 1190) 

s_sub_u32 s7, s6, s15 // (op 1190) 
s_lshr_b32 s8, s7, 1 // (op 1190) 

s_add_u32 s7, s8, s15 // (op 1190) 


s_lshr_b32 s8, s7, s11 // (op 1190) 

s_and_b32 s7, s11, 2147483648 // (op 1190) 
s_cmp_eq_i32 s7, 0 // (op 1190) 
s_mov_b32 s15, scc // (op 1190) 

s_cmp_eq_i32 s15, 0 // (op 1190) 
s_cselect_b32 s7, s6, s8 // (op 1190) 


s_mul_hi_u32 s8, s7, s12 // (op 1190) 
s_sub_u32 s15, s7, s8 // (op 1190) 
s_lshr_b32 s18, s15, 1 // (op 1190) 

s_add_u32 s15, s18, s8 // (op 1190) 


s_lshr_b32 s8, s15, s10 // (op 1190) 

s_and_b32 s15, s10, 2147483648 // (op 1190) 
s_cmp_eq_i32 s15, 0 // (op 1190) 
s_mov_b32 s18, scc // (op 1190) 

s_cmp_eq_i32 s18, 0 // (op 1190) 
s_cselect_b32 s15, s7, s8 // (op 1190) 


v_lshl_add_u32 v16, s15, 5, v10 // (op 1190) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1190) 
s_mul_i32 s8, s7, s18 // (op 1190) 


s_sub_u32 s7, s6, s8 // (op 1190) 


v_lshl_add_u32 v10, s7, 7, v14 // (op 1190) 


v_mov_b32 v19, 0 // (op 1190) convert
v_mov_b32 v18, v10 // (op 1190) convert

v_mul_lo_u32 v23, v16, s25 // (op 1190) most significant: low of low * high
v_mul_hi_u32 v10, v16, s24 // (op 1190) most significant: high of low * low
v_mul_lo_u32 v22, v16, s24 // (op 1190) least significant: low of low * low
v_add_u32 v23, v23, v10 // (op 1190) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v18 // (op 1190) least significant half
v_addc_co_u32 v23, s[18:19], v23, v19, s[18:19] // (op 1190) most significant half



s_mul_i32 s19, s24, s29 // (op 1192) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 1192) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 1192) least significant
s_mul_hi_u32 s6, s24, s28 // (op 1192) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1192) most significant: sum
s_add_u32 s19, s19, s6 // (op 1192) most significant: sum


s_mov_b32 s60, s26 // (op 1192) reg expression
s_mov_b32 s61, s27 // (op 1192) reg expression
s_mov_b32 s62, s18 // (op 1192) reg expression
s_mov_b32 s63, 131072 // (op 1192) call()

v_lshrrev_b32 v10, 3, v1 // (op 1299) 
s_mul_i32 s15, s16, s50 // (op 1299) 
s_add_u32 s15, s15, s0 // (op 1299) 
v_and_b32 v14, -8, v1 // (op 1299) 
s_add_u32 s6, s15, s9 // (op 1299) 

v_sub_u32 v16, v1, v14 // (op 1299) 

s_mul_hi_u32 s15, s6, s13 // (op 1299) 
v_lshlrev_b32 v14, 4, v16 // (op 1299) 

s_sub_u32 s7, s6, s15 // (op 1299) 
s_lshr_b32 s8, s7, 1 // (op 1299) 

s_add_u32 s7, s8, s15 // (op 1299) 


s_lshr_b32 s8, s7, s11 // (op 1299) 

s_and_b32 s7, s11, 2147483648 // (op 1299) 
s_cmp_eq_i32 s7, 0 // (op 1299) 
s_mov_b32 s15, scc // (op 1299) 

s_cmp_eq_i32 s15, 0 // (op 1299) 
s_cselect_b32 s7, s6, s8 // (op 1299) 


s_mul_hi_u32 s8, s7, s12 // (op 1299) 
s_sub_u32 s15, s7, s8 // (op 1299) 
s_lshr_b32 s18, s15, 1 // (op 1299) 

s_add_u32 s15, s18, s8 // (op 1299) 


s_lshr_b32 s8, s15, s10 // (op 1299) 

s_and_b32 s15, s10, 2147483648 // (op 1299) 
s_cmp_eq_i32 s15, 0 // (op 1299) 
s_mov_b32 s18, scc // (op 1299) 

s_cmp_eq_i32 s18, 0 // (op 1299) 
s_cselect_b32 s15, s7, s8 // (op 1299) 


s_mul_i32 s8, s15, s14 // (op 1299) 

s_sub_u32 s15, s7, s8 // (op 1299) 

v_lshl_add_u32 v16, s15, 4, v10 // (op 1299) 


v_mul_lo_u32 v19, v16, s31 // (op 1299) most significant: low of low * high
v_mul_hi_u32 v10, v16, s30 // (op 1299) most significant: high of low * low
v_mul_lo_u32 v18, v16, s30 // (op 1299) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 1299) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1299) 
s_mul_i32 s8, s7, s18 // (op 1299) 


s_sub_u32 s7, s6, s8 // (op 1299) 


v_lshl_add_u32 v10, s7, 7, v14 // (op 1299) 


v_mov_b32 v17, 0 // (op 1299) convert
v_mov_b32 v16, v10 // (op 1299) convert

v_add_co_u32 v24, s[18:19], v16, v18 // (op 1299) least significant half
v_addc_co_u32 v25, s[18:19], v17, v19, s[18:19] // (op 1299) most significant half



s_mul_i32 s19, s30, s23 // (op 1301) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 1301) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 1301) least significant
s_mul_hi_u32 s6, s30, s22 // (op 1301) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1301) most significant: sum
s_add_u32 s19, s19, s6 // (op 1301) most significant: sum


s_mov_b32 s64, s32 // (op 1301) reg expression
s_mov_b32 s65, s33 // (op 1301) reg expression
s_mov_b32 s66, s18 // (op 1301) reg expression
s_mov_b32 s67, 131072 // (op 1301) call()

v_lshrrev_b32 v10, 3, v1 // (op 1318) 
s_mul_i32 s15, s16, s50 // (op 1318) 
s_add_u32 s15, s15, s0 // (op 1318) 
v_and_b32 v14, -8, v1 // (op 1318) 
s_add_u32 s6, s15, s9 // (op 1318) 

v_sub_u32 v16, v1, v14 // (op 1318) 

s_mul_hi_u32 s15, s6, s13 // (op 1318) 
v_lshlrev_b32 v14, 4, v16 // (op 1318) 

s_sub_u32 s7, s6, s15 // (op 1318) 
s_lshr_b32 s8, s7, 1 // (op 1318) 

s_add_u32 s7, s8, s15 // (op 1318) 


s_lshr_b32 s8, s7, s11 // (op 1318) 

s_and_b32 s7, s11, 2147483648 // (op 1318) 
s_cmp_eq_i32 s7, 0 // (op 1318) 
s_mov_b32 s15, scc // (op 1318) 

s_cmp_eq_i32 s15, 0 // (op 1318) 
s_cselect_b32 s7, s6, s8 // (op 1318) 


s_mul_hi_u32 s8, s7, s12 // (op 1318) 
s_sub_u32 s15, s7, s8 // (op 1318) 
s_lshr_b32 s18, s15, 1 // (op 1318) 

s_add_u32 s15, s18, s8 // (op 1318) 


s_lshr_b32 s8, s15, s10 // (op 1318) 

s_and_b32 s15, s10, 2147483648 // (op 1318) 
s_cmp_eq_i32 s15, 0 // (op 1318) 
s_mov_b32 s18, scc // (op 1318) 

s_cmp_eq_i32 s18, 0 // (op 1318) 
s_cselect_b32 s15, s7, s8 // (op 1318) 


s_mul_i32 s8, s15, s14 // (op 1318) 

s_sub_u32 s15, s7, s8 // (op 1318) 

v_lshl_add_u32 v16, s15, 4, v10 // (op 1318) 


v_mul_lo_u32 v19, v16, s31 // (op 1318) most significant: low of low * high
v_mul_hi_u32 v10, v16, s30 // (op 1318) most significant: high of low * low
v_mul_lo_u32 v18, v16, s30 // (op 1318) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 1318) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1318) 
s_mul_i32 s8, s7, s18 // (op 1318) 


s_sub_u32 s7, s6, s8 // (op 1318) 


v_lshl_add_u32 v10, s7, 7, v14 // (op 1318) 


v_mov_b32 v17, 0 // (op 1318) convert
v_mov_b32 v16, v10 // (op 1318) convert

v_add_co_u32 v26, s[18:19], v16, v18 // (op 1318) least significant half
v_addc_co_u32 v27, s[18:19], v17, v19, s[18:19] // (op 1318) most significant half



s_mul_i32 s19, s30, s23 // (op 1320) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 1320) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 1320) least significant
s_mul_hi_u32 s6, s30, s22 // (op 1320) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1320) most significant: sum
s_add_u32 s19, s19, s6 // (op 1320) most significant: sum


s_mov_b32 s64, s32 // (op 1320) reg expression
s_mov_b32 s65, s33 // (op 1320) reg expression
s_mov_b32 s66, s18 // (op 1320) reg expression
s_mov_b32 s67, 131072 // (op 1320) call()

v_lshrrev_b32 v10, 2, v1 // (op 1427) 
s_mul_i32 s15, s16, s50 // (op 1427) 
s_add_u32 s15, s15, s0 // (op 1427) 
v_and_b32 v14, -4, v1 // (op 1427) 
s_add_u32 s6, s15, s9 // (op 1427) 

v_sub_u32 v16, v1, v14 // (op 1427) 

s_mul_hi_u32 s15, s6, s13 // (op 1427) 
s_sub_u32 s7, s6, s15 // (op 1427) 
s_lshr_b32 s8, s7, 1 // (op 1427) 

s_add_u32 s7, s8, s15 // (op 1427) 


s_lshr_b32 s8, s7, s11 // (op 1427) 

s_and_b32 s7, s11, 2147483648 // (op 1427) 
s_cmp_eq_i32 s7, 0 // (op 1427) 
s_mov_b32 s15, scc // (op 1427) 

s_cmp_eq_i32 s15, 0 // (op 1427) 
s_cselect_b32 s7, s6, s8 // (op 1427) 


s_mul_hi_u32 s8, s7, s12 // (op 1427) 
s_sub_u32 s15, s7, s8 // (op 1427) 
s_lshr_b32 s18, s15, 1 // (op 1427) 

s_add_u32 s15, s18, s8 // (op 1427) 


s_lshr_b32 s8, s15, s10 // (op 1427) 

s_and_b32 s15, s10, 2147483648 // (op 1427) 
s_cmp_eq_i32 s15, 0 // (op 1427) 
s_mov_b32 s18, scc // (op 1427) 

s_cmp_eq_i32 s18, 0 // (op 1427) 
s_cselect_b32 s15, s7, s8 // (op 1427) 


v_lshl_add_u32 v14, s15, 5, v10 // (op 1427) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1427) 
s_mul_i32 s8, s7, s18 // (op 1427) 


s_sub_u32 s7, s6, s8 // (op 1427) 


v_lshl_add_u32 v10, s7, 2, v16 // (op 1427) 


v_mul_lo_u32 v16, v14, s34 // (op 1427) 
v_add_u32 v16, v16, v10 // (op 1427) 


s_mul_i32 s19, s34, s29 // (op 1429) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1429) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1429) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1429) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1429) most significant: sum
s_add_u32 s19, s19, s6 // (op 1429) most significant: sum


s_mov_b32 s68, s36 // (op 1429) reg expression
s_mov_b32 s69, s37 // (op 1429) reg expression
s_mov_b32 s70, s18 // (op 1429) reg expression
s_mov_b32 s71, 131072 // (op 1429) call()

v_lshrrev_b32 v10, 2, v1 // (op 1446) 
s_mul_i32 s15, s16, s50 // (op 1446) 
s_add_u32 s15, s15, s0 // (op 1446) 
v_and_b32 v14, -4, v1 // (op 1446) 
s_add_u32 s6, s15, s9 // (op 1446) 

v_sub_u32 v17, v1, v14 // (op 1446) 

s_mul_hi_u32 s15, s6, s13 // (op 1446) 
s_sub_u32 s7, s6, s15 // (op 1446) 
s_lshr_b32 s8, s7, 1 // (op 1446) 

s_add_u32 s7, s8, s15 // (op 1446) 


s_lshr_b32 s8, s7, s11 // (op 1446) 

s_and_b32 s7, s11, 2147483648 // (op 1446) 
s_cmp_eq_i32 s7, 0 // (op 1446) 
s_mov_b32 s15, scc // (op 1446) 

s_cmp_eq_i32 s15, 0 // (op 1446) 
s_cselect_b32 s7, s6, s8 // (op 1446) 


s_mul_hi_u32 s8, s7, s12 // (op 1446) 
s_sub_u32 s15, s7, s8 // (op 1446) 
s_lshr_b32 s18, s15, 1 // (op 1446) 

s_add_u32 s15, s18, s8 // (op 1446) 


s_lshr_b32 s8, s15, s10 // (op 1446) 

s_and_b32 s15, s10, 2147483648 // (op 1446) 
s_cmp_eq_i32 s15, 0 // (op 1446) 
s_mov_b32 s18, scc // (op 1446) 

s_cmp_eq_i32 s18, 0 // (op 1446) 
s_cselect_b32 s15, s7, s8 // (op 1446) 


v_lshl_add_u32 v14, s15, 5, v10 // (op 1446) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1446) 
s_mul_i32 s8, s7, s18 // (op 1446) 


s_sub_u32 s7, s6, s8 // (op 1446) 


v_lshl_add_u32 v10, s7, 2, v17 // (op 1446) 


v_mul_lo_u32 v17, v14, s34 // (op 1446) 
v_add_u32 v17, v17, v10 // (op 1446) 


s_mul_i32 s19, s34, s29 // (op 1448) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1448) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1448) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1448) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1448) most significant: sum
s_add_u32 s19, s19, s6 // (op 1448) most significant: sum


s_mov_b32 s68, s36 // (op 1448) reg expression
s_mov_b32 s69, s37 // (op 1448) reg expression
s_mov_b32 s70, s18 // (op 1448) reg expression
s_mov_b32 s71, 131072 // (op 1448) call()

v_and_b32 v10, 15, v1 // (op 1555) 
s_mul_i32 s15, s16, s50 // (op 1555) 
s_add_u32 s15, s15, s0 // (op 1555) 
v_and_b32 v14, 63, v1 // (op 1555) 
s_add_u32 s6, s15, s9 // (op 1555) 

v_lshrrev_b32 v18, 4, v14 // (op 1555) 

s_mul_hi_u32 s15, s6, s13 // (op 1555) 
s_sub_u32 s7, s6, s15 // (op 1555) 
s_lshr_b32 s8, s7, 1 // (op 1555) 

s_add_u32 s7, s8, s15 // (op 1555) 


s_lshr_b32 s8, s7, s11 // (op 1555) 

s_and_b32 s7, s11, 2147483648 // (op 1555) 
s_cmp_eq_i32 s7, 0 // (op 1555) 
s_mov_b32 s15, scc // (op 1555) 

s_cmp_eq_i32 s15, 0 // (op 1555) 
s_cselect_b32 s7, s6, s8 // (op 1555) 


s_mul_hi_u32 s8, s7, s12 // (op 1555) 
s_sub_u32 s15, s7, s8 // (op 1555) 
s_lshr_b32 s18, s15, 1 // (op 1555) 

s_add_u32 s15, s18, s8 // (op 1555) 


s_lshr_b32 s8, s15, s10 // (op 1555) 

s_and_b32 s15, s10, 2147483648 // (op 1555) 
s_cmp_eq_i32 s15, 0 // (op 1555) 
s_mov_b32 s18, scc // (op 1555) 

s_cmp_eq_i32 s18, 0 // (op 1555) 
s_cselect_b32 s15, s7, s8 // (op 1555) 


s_mul_i32 s8, s15, s14 // (op 1555) 

s_sub_u32 s15, s7, s8 // (op 1555) 

v_lshl_add_u32 v14, s15, 4, v10 // (op 1555) 


v_mul_lo_u32 v10, v14, s38 // (op 1555) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1555) 
s_mul_i32 s8, s7, s18 // (op 1555) 


s_sub_u32 s7, s6, s8 // (op 1555) 


v_lshl_add_u32 v14, s7, 2, v18 // (op 1555) 


v_add_u32 v18, v14, v10 // (op 1555) 


s_mul_i32 s19, s38, s23 // (op 1557) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 1557) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 1557) least significant
s_mul_hi_u32 s6, s38, s22 // (op 1557) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1557) most significant: sum
s_add_u32 s19, s19, s6 // (op 1557) most significant: sum


s_mov_b32 s72, s40 // (op 1557) reg expression
s_mov_b32 s73, s41 // (op 1557) reg expression
s_mov_b32 s74, s18 // (op 1557) reg expression
s_mov_b32 s75, 131072 // (op 1557) call()

v_and_b32 v10, 15, v1 // (op 1574) 
s_mul_i32 s15, s16, s50 // (op 1574) 
s_add_u32 s15, s15, s0 // (op 1574) 
v_and_b32 v14, 63, v1 // (op 1574) 
s_add_u32 s6, s15, s9 // (op 1574) 

v_lshrrev_b32 v19, 4, v14 // (op 1574) 

s_mul_hi_u32 s15, s6, s13 // (op 1574) 
s_sub_u32 s7, s6, s15 // (op 1574) 
s_lshr_b32 s8, s7, 1 // (op 1574) 

s_add_u32 s7, s8, s15 // (op 1574) 


s_lshr_b32 s8, s7, s11 // (op 1574) 

s_and_b32 s7, s11, 2147483648 // (op 1574) 
s_cmp_eq_i32 s7, 0 // (op 1574) 
s_mov_b32 s15, scc // (op 1574) 

s_cmp_eq_i32 s15, 0 // (op 1574) 
s_cselect_b32 s7, s6, s8 // (op 1574) 


s_mul_hi_u32 s8, s7, s12 // (op 1574) 
s_sub_u32 s15, s7, s8 // (op 1574) 
s_lshr_b32 s18, s15, 1 // (op 1574) 

s_add_u32 s15, s18, s8 // (op 1574) 


s_lshr_b32 s8, s15, s10 // (op 1574) 

s_and_b32 s15, s10, 2147483648 // (op 1574) 
s_cmp_eq_i32 s15, 0 // (op 1574) 
s_mov_b32 s18, scc // (op 1574) 

s_cmp_eq_i32 s18, 0 // (op 1574) 
s_cselect_b32 s15, s7, s8 // (op 1574) 


s_mul_i32 s8, s15, s14 // (op 1574) 

s_sub_u32 s15, s7, s8 // (op 1574) 

v_lshl_add_u32 v14, s15, 4, v10 // (op 1574) 


v_mul_lo_u32 v10, v14, s38 // (op 1574) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1574) 
s_mul_i32 s8, s7, s18 // (op 1574) 


s_sub_u32 s7, s6, s8 // (op 1574) 


v_lshl_add_u32 v14, s7, 2, v19 // (op 1574) 


v_add_u32 v19, v14, v10 // (op 1574) 


s_mul_i32 s19, s38, s23 // (op 1576) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 1576) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 1576) least significant
s_mul_hi_u32 s6, s38, s22 // (op 1576) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1576) most significant: sum
s_add_u32 s19, s19, s6 // (op 1576) most significant: sum


s_mov_b32 s72, s40 // (op 1576) reg expression
s_mov_b32 s73, s41 // (op 1576) reg expression
s_mov_b32 s74, s18 // (op 1576) reg expression
s_mov_b32 s75, 131072 // (op 1576) call()




buffer_load_ubyte v10, v18, s[72:75], 0 offen offset: 0 // (op 886) Load value
v_and_b32 v14, -8, v1 // (op 1987) 
v_sub_u32 v28, v1, v14 // (op 1987) 

v_lshlrev_b32 v14, 4, v28 // (op 1987) 

v_lshrrev_b32 v28, 3, v1 // (op 1987) 
v_lshl_add_u32 v29, v28, 7, v14 // (op 1987) 


s_nop 0
v_readfirstlane_b32 s15, v29 // (op 1987) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2466) 



s_nop 0
s_mov_b32 m0, s15 // (op 1051) reg expression
buffer_load_dwordx4 v24, s[64:67], 0 offen offset: 0 lds // (op 1051) Load value direct to lds

v_and_b32 v14, -8, v1 // (op 1922) 
v_sub_u32 v28, v1, v14 // (op 1922) 

v_lshlrev_b32 v14, 4, v28 // (op 1922) 

v_lshrrev_b32 v28, 3, v1 // (op 1922) 
v_lshl_add_u32 v29, v28, 7, v14 // (op 1922) 


s_nop 0
v_readfirstlane_b32 s15, v29 // (op 1922) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2447) 

s_lshl_b64 s[18:19], s[24:25], 4 // (op 1056) 
v_mov_b64 v[28:29], s[18:19] // (op 1056) convert


v_mov_b64 v[30:31], v[20:21] // (op 1056) 
s_add_u32 s6, 2048, s15 // (op 1056) 
s_mov_b32 m0, s6 // (op 1056) reg expression
buffer_load_dwordx4 v30, s[60:63], 0 offen offset: 0 lds // (op 1056) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 1056) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 1056) most significant half

s_add_u32 m0, m0, 2048 // (op 1056) 
buffer_load_dwordx4 v30, s[60:63], 0 offen offset: 0 lds // (op 1056) Load value direct to lds







buffer_load_ubyte v14, v16, s[68:71], 0 offen offset: 0 // (op 896) Load value



buffer_load_ubyte v28, v18, s[72:75], 0 offen offset:4 // (op 900) Load value

v_and_b32 v18, -8, v1 // (op 2178) 
v_sub_u32 v29, v1, v18 // (op 2178) 

v_lshlrev_b32 v18, 4, v29 // (op 2178) 

v_lshrrev_b32 v29, 3, v1 // (op 2178) 
v_lshl_add_u32 v30, v29, 7, v18 // (op 2178) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 2178) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2514) 


s_nop 0
s_add_u32 s6, 6144, s15 // (op 1061) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s6 // (op 1061) reg expression
s_mov_b32 s7, 128 // (op 1061) call()
buffer_load_dwordx4 v24, s[64:67], s7 offen offset: 0 lds // (op 1061) Load value direct to lds




v_and_b32 v18, -8, v1 // (op 2139) 
v_sub_u32 v24, v1, v18 // (op 2139) 

v_lshlrev_b32 v18, 4, v24 // (op 2139) 

v_lshrrev_b32 v24, 3, v1 // (op 2139) 
v_lshl_add_u32 v25, v24, 7, v18 // (op 2139) 


s_nop 0
v_readfirstlane_b32 s15, v25 // (op 2139) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2505) 

s_lshl_b64 s[18:19], s[24:25], 4 // (op 1063) 
v_mov_b64 v[24:25], s[18:19] // (op 1063) convert


v_mov_b64 v[30:31], v[20:21] // (op 1063) 
s_add_u32 s6, 8192, s15 // (op 1063) 
s_mov_b32 m0, s6 // (op 1063) reg expression
s_mov_b32 s7, 128 // (op 1063) call()
buffer_load_dwordx4 v30, s[60:63], s7 offen offset: 0 lds // (op 1063) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v24 // (op 1063) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 1063) most significant half

s_add_u32 m0, m0, 2048 // (op 1063) 
buffer_load_dwordx4 v30, s[60:63], s7 offen offset: 0 lds // (op 1063) Load value direct to lds









buffer_load_ubyte v18, v16, s[68:71], 0 offen offset:4 // (op 910) Load value

v_and_b32 v16, -4, v1 // (op 2053) 
v_sub_u32 v20, v1, v16 // (op 2053) 

v_and_b32 v16, 4294967292, v1 // (op 2053) 
v_add_u32 v21, v16, v20 // (op 2053) 


s_barrier  // (op 2485) 



s_waitcnt vmcnt(5)
ds_write_b8 v21, v14 offset:12288 // (op 914) Store local data 

s_barrier  // (op 2463) 



ds_read_b128 v[30:33], v12 // (op 930) Load local data 
ds_read_b128 v[34:37], v12 offset:64 // (op 930) Load local data 
s_barrier  // (op 2444) 



ds_read_b128 v[38:41], v5 offset:2048 // (op 927) Load local data 
ds_read_b128 v[42:45], v5 offset:2112 // (op 927) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2482) 



ds_read_u8 v16, v0 offset:12288 // (op 933) Load local data 
v_and_b32 v29, -8, v1 // (op 1894) 
v_sub_u32 v20, v1, v29 // (op 1894) 

v_lshlrev_b32 v21, 4, v20 // (op 1894) 

v_lshrrev_b32 v20, 3, v1 // (op 1894) 
v_lshl_add_u32 v29, v20, 7, v21 // (op 1894) 


s_nop 0
v_readfirstlane_b32 s15, v29 // (op 1894) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -8, v1 // (op 1959) 
v_sub_u32 v20, v1, v29 // (op 1959) 

v_lshlrev_b32 v21, 4, v20 // (op 1959) 

v_lshrrev_b32 v20, 3, v1 // (op 1959) 
v_lshl_add_u32 v29, v20, 7, v21 // (op 1959) 


s_nop 0
v_readfirstlane_b32 s6, v29 // (op 1959) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -4, v1 // (op 2024) 
v_sub_u32 v20, v1, v29 // (op 2024) 

v_and_b32 v21, 4294967292, v1 // (op 2024) 
v_add_u32 v29, v21, v20 // (op 2024) 


v_and_b32 v20, -8, v1 // (op 2124) 
v_sub_u32 v21, v1, v20 // (op 2124) 

v_lshlrev_b32 v20, 4, v21 // (op 2124) 

v_lshrrev_b32 v21, 3, v1 // (op 2124) 
v_lshl_add_u32 v24, v21, 7, v20 // (op 2124) 


s_nop 0
v_readfirstlane_b32 s7, v24 // (op 2124) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -8, v1 // (op 2163) 
v_sub_u32 v21, v1, v20 // (op 2163) 

v_lshlrev_b32 v20, 4, v21 // (op 2163) 

v_lshrrev_b32 v21, 3, v1 // (op 2163) 
v_lshl_add_u32 v24, v21, 7, v20 // (op 2163) 


s_nop 0
v_readfirstlane_b32 s8, v24 // (op 2163) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -4, v1 // (op 2202) 
v_sub_u32 v21, v1, v20 // (op 2202) 

v_and_b32 v20, 4294967292, v1 // (op 2202) 
v_add_u32 v24, v20, v21 // (op 2202) 


s_mov_b32 s1, 0 // (op 172) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 171) 
s_mov_b32 s42, s18 // (op 171) convert

s_and_b32 s18, s42, 4294967294 // (op 171) 

s_cmp_lt_u32 s1, s18 // (op 171) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_171 // (op 171) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_171 if false)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoop_171:
s_add_u32 s1, s1, 2 // (op 173) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v16, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 184) 


v_mov_b32 v25, v19 // (op 183) 

buffer_load_ubyte v10, v25, s[72:75], 0 offen offset:8 // (op 183) Load value
s_barrier  // (op 2469) 


v_mov_b64 v[20:21], v[26:27] // (op 986) 

s_mov_b32 m0, s6 // (op 986) reg expression
s_mov_b32 s18, 256 // (op 986) call()
buffer_load_dwordx4 v20, s[64:67], s18 offen offset: 0 lds // (op 986) Load value direct to lds

s_barrier  // (op 2450) 

s_lshl_b64 s[18:19], s[24:25], 4 // (op 981) 
v_mov_b64 v[46:47], s[18:19] // (op 981) convert


v_mov_b64 v[48:49], v[22:23] // (op 981) 
v_mov_b64 v[50:51], v[48:49] // (op 981) 
s_add_u32 s18, 2048, s15 // (op 981) 
s_mov_b32 m0, s18 // (op 981) reg expression
s_mov_b32 s19, 256 // (op 981) call()
buffer_load_dwordx4 v50, s[60:63], s19 offen offset: 0 lds // (op 981) Load value direct to lds
v_add_co_u32 v50, s[42:43], v50, v46 // (op 981) least significant half
v_addc_co_u32 v51, s[42:43], 0, v51, s[42:43] // (op 981) most significant half

s_add_u32 m0, m0, 2048 // (op 981) 
buffer_load_dwordx4 v50, s[60:63], s19 offen offset: 0 lds // (op 981) Load value direct to lds






v_mov_b32 v46, v17 // (op 188) 

buffer_load_ubyte v14, v46, s[68:71], 0 offen offset:8 // (op 188) Load value
s_barrier  // (op 2523) 



s_waitcnt vmcnt(5)
ds_write_b8 v24, v18 offset:12416 // (op 508) Store local data 
s_barrier  // (op 2511) 



ds_read_b128 v[30:33], v11 offset:6144 // (op 501) Load local data 
ds_read_b128 v[34:37], v11 offset:6208 // (op 501) Load local data 
s_barrier  // (op 2502) 



ds_read_b128 v[38:41], v9 offset:8192 // (op 499) Load local data 
ds_read_b128 v[42:45], v9 offset:8256 // (op 499) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2520) 



ds_read_u8 v16, v7 offset:12416 // (op 503) Load local data 
s_barrier  // (op 2634) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v16, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 506) 



buffer_load_ubyte v28, v25, s[72:75], 0 offen offset:12 // (op 505) Load value

s_barrier  // (op 2517) 
v_add_u32 v19, 8, v19 // (op 1566) 


s_add_u32 s18, 6144, s8 // (op 1009) 

s_mov_b32 m0, s18 // (op 1009) reg expression
s_mov_b32 s19, 384 // (op 1009) call()
buffer_load_dwordx4 v20, s[64:67], s19 offen offset: 0 lds // (op 1009) Load value direct to lds



s_barrier  // (op 2508) 
v_add_co_u32 v26, vcc, 256, v26 // (op 1310) least significant half

v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 1310) most significant half
s_lshl_b64 s[18:19], s[24:25], 4 // (op 1004) 
v_mov_b64 v[20:21], s[18:19] // (op 1004) convert


v_mov_b64 v[50:51], v[48:49] // (op 1004) 
s_add_u32 s18, 8192, s7 // (op 1004) 
s_mov_b32 m0, s18 // (op 1004) reg expression
s_mov_b32 s19, 384 // (op 1004) call()
buffer_load_dwordx4 v50, s[60:63], s19 offen offset: 0 lds // (op 1004) Load value direct to lds
v_add_co_u32 v50, s[42:43], v50, v20 // (op 1004) least significant half
v_addc_co_u32 v51, s[42:43], 0, v51, s[42:43] // (op 1004) most significant half

s_add_u32 m0, m0, 2048 // (op 1004) 
buffer_load_dwordx4 v50, s[60:63], s19 offen offset: 0 lds // (op 1004) Load value direct to lds








v_add_co_u32 v22, vcc, 256, v22 // (op 1182) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 1182) most significant half
buffer_load_ubyte v18, v46, s[68:71], 0 offen offset:12 // (op 510) Load value

v_add_u32 v17, 8, v17 // (op 1438) 
s_barrier  // (op 2488) 



s_waitcnt vmcnt(5)
ds_write_b8 v29, v14 offset:12288 // (op 186) Store local data 
s_barrier  // (op 2475) 



ds_read_b128 v[30:33], v8 // (op 179) Load local data 
ds_read_b128 v[34:37], v8 offset:64 // (op 179) Load local data 
s_barrier  // (op 2456) 



ds_read_b128 v[38:41], v6 offset:2048 // (op 177) Load local data 
ds_read_b128 v[42:45], v6 offset:2112 // (op 177) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2495) 



ds_read_u8 v16, v4 offset:12288 // (op 181) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 171) 
s_mov_b32 s42, s18 // (op 171) convert

s_and_b32 s18, s42, 4294967294 // (op 171) 

s_cmp_lt_u32 s1, s18 // (op 171) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoop_171 // (op 171) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoop_171 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_171:












s_barrier  // (op 2568) 
v_lshrrev_b32 v10, 3, v1 // (op 1076) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1076) 
v_and_b32 v14, -8, v1 // (op 1076) 
s_and_b32 s15, s18, 4294967294 // (op 1076) 
v_sub_u32 v17, v1, v14 // (op 1076) 

s_add_u32 s6, s0, s15 // (op 1076) 

v_lshlrev_b32 v14, 4, v17 // (op 1076) 

s_mul_i32 s15, s16, s50 // (op 1076) 
s_add_u32 s15, s15, s6 // (op 1076) 

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
s_lshr_b32 s42, s15, 1 // (op 1076) 

s_add_u32 s15, s42, s8 // (op 1076) 


s_lshr_b32 s8, s15, s10 // (op 1076) 

s_and_b32 s15, s10, 2147483648 // (op 1076) 
s_cmp_eq_i32 s15, 0 // (op 1076) 
s_mov_b32 s42, scc // (op 1076) 

s_cmp_eq_i32 s42, 0 // (op 1076) 
s_cselect_b32 s15, s7, s8 // (op 1076) 


s_mul_i32 s8, s7, s18 // (op 1076) 


v_lshl_add_u32 v17, s15, 5, v10 // (op 1076) 


s_sub_u32 s7, s6, s8 // (op 1076) 


v_lshl_add_u32 v10, s7, 7, v14 // (op 1076) 


v_mov_b32 v27, 0 // (op 1076) convert
v_mov_b32 v26, v10 // (op 1076) convert

v_mul_lo_u32 v25, v17, s25 // (op 1076) most significant: low of low * high
v_mul_hi_u32 v10, v17, s24 // (op 1076) most significant: high of low * low
v_mul_lo_u32 v24, v17, s24 // (op 1076) least significant: low of low * low
v_add_u32 v25, v25, v10 // (op 1076) most significant: sum

v_add_co_u32 v24, s[18:19], v24, v26 // (op 1076) least significant half
v_addc_co_u32 v25, s[18:19], v25, v27, s[18:19] // (op 1076) most significant half



s_mul_i32 s19, s24, s29 // (op 1078) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 1078) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 1078) least significant
s_mul_hi_u32 s6, s24, s28 // (op 1078) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1078) most significant: sum
s_add_u32 s19, s19, s6 // (op 1078) most significant: sum


s_mov_b32 s60, s26 // (op 1078) reg expression
s_mov_b32 s61, s27 // (op 1078) reg expression
s_mov_b32 s62, s18 // (op 1078) reg expression
s_mov_b32 s63, 131072 // (op 1078) call()

v_lshrrev_b32 v10, 3, v1 // (op 1212) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1212) 
v_and_b32 v14, -8, v1 // (op 1212) 
s_and_b32 s15, s18, 4294967294 // (op 1212) 
v_sub_u32 v17, v1, v14 // (op 1212) 

s_add_u32 s6, s0, s15 // (op 1212) 

v_lshlrev_b32 v14, 4, v17 // (op 1212) 

s_mul_i32 s15, s16, s50 // (op 1212) 
s_add_u32 s15, s15, s6 // (op 1212) 

s_add_u32 s6, s15, s9 // (op 1212) 

s_mul_hi_u32 s15, s6, s13 // (op 1212) 
s_sub_u32 s7, s6, s15 // (op 1212) 
s_lshr_b32 s8, s7, 1 // (op 1212) 

s_add_u32 s7, s8, s15 // (op 1212) 


s_lshr_b32 s8, s7, s11 // (op 1212) 

s_and_b32 s7, s11, 2147483648 // (op 1212) 
s_cmp_eq_i32 s7, 0 // (op 1212) 
s_mov_b32 s15, scc // (op 1212) 

s_cmp_eq_i32 s15, 0 // (op 1212) 
s_cselect_b32 s7, s6, s8 // (op 1212) 


s_mul_hi_u32 s8, s7, s12 // (op 1212) 
s_sub_u32 s15, s7, s8 // (op 1212) 
s_lshr_b32 s42, s15, 1 // (op 1212) 

s_add_u32 s15, s42, s8 // (op 1212) 


s_lshr_b32 s8, s15, s10 // (op 1212) 

s_and_b32 s15, s10, 2147483648 // (op 1212) 
s_cmp_eq_i32 s15, 0 // (op 1212) 
s_mov_b32 s42, scc // (op 1212) 

s_cmp_eq_i32 s42, 0 // (op 1212) 
s_cselect_b32 s15, s7, s8 // (op 1212) 


s_mul_i32 s8, s15, s14 // (op 1212) 

s_sub_u32 s15, s7, s8 // (op 1212) 

s_mul_i32 s8, s7, s18 // (op 1212) 


v_lshl_add_u32 v17, s15, 4, v10 // (op 1212) 


s_sub_u32 s7, s6, s8 // (op 1212) 


v_mul_lo_u32 v27, v17, s31 // (op 1212) most significant: low of low * high
v_mul_hi_u32 v10, v17, s30 // (op 1212) most significant: high of low * low
v_mul_lo_u32 v26, v17, s30 // (op 1212) least significant: low of low * low
v_add_u32 v27, v27, v10 // (op 1212) most significant: sum


v_lshl_add_u32 v10, s7, 7, v14 // (op 1212) 


v_mov_b32 v23, 0 // (op 1212) convert
v_mov_b32 v22, v10 // (op 1212) convert

v_add_co_u32 v20, s[18:19], v22, v26 // (op 1212) least significant half
v_addc_co_u32 v21, s[18:19], v23, v27, s[18:19] // (op 1212) most significant half



s_mul_i32 s19, s30, s23 // (op 1214) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 1214) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 1214) least significant
s_mul_hi_u32 s6, s30, s22 // (op 1214) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1214) most significant: sum
s_add_u32 s19, s19, s6 // (op 1214) most significant: sum


s_mov_b32 s64, s32 // (op 1214) reg expression
s_mov_b32 s65, s33 // (op 1214) reg expression
s_mov_b32 s66, s18 // (op 1214) reg expression
s_mov_b32 s67, 131072 // (op 1214) call()

v_lshrrev_b32 v10, 2, v1 // (op 1340) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1340) 
v_and_b32 v14, -4, v1 // (op 1340) 
s_and_b32 s15, s18, 4294967294 // (op 1340) 
v_sub_u32 v17, v1, v14 // (op 1340) 

s_add_u32 s6, s0, s15 // (op 1340) 

s_mul_i32 s15, s16, s50 // (op 1340) 
s_add_u32 s15, s15, s6 // (op 1340) 

s_add_u32 s6, s15, s9 // (op 1340) 

s_mul_hi_u32 s15, s6, s13 // (op 1340) 
s_sub_u32 s7, s6, s15 // (op 1340) 
s_lshr_b32 s8, s7, 1 // (op 1340) 

s_add_u32 s7, s8, s15 // (op 1340) 


s_lshr_b32 s8, s7, s11 // (op 1340) 

s_and_b32 s7, s11, 2147483648 // (op 1340) 
s_cmp_eq_i32 s7, 0 // (op 1340) 
s_mov_b32 s15, scc // (op 1340) 

s_cmp_eq_i32 s15, 0 // (op 1340) 
s_cselect_b32 s7, s6, s8 // (op 1340) 


s_mul_hi_u32 s8, s7, s12 // (op 1340) 
s_sub_u32 s15, s7, s8 // (op 1340) 
s_lshr_b32 s42, s15, 1 // (op 1340) 

s_add_u32 s15, s42, s8 // (op 1340) 


s_lshr_b32 s8, s15, s10 // (op 1340) 

s_and_b32 s15, s10, 2147483648 // (op 1340) 
s_cmp_eq_i32 s15, 0 // (op 1340) 
s_mov_b32 s42, scc // (op 1340) 

s_cmp_eq_i32 s42, 0 // (op 1340) 
s_cselect_b32 s15, s7, s8 // (op 1340) 


s_mul_i32 s8, s7, s18 // (op 1340) 


v_lshl_add_u32 v14, s15, 5, v10 // (op 1340) 


s_sub_u32 s7, s6, s8 // (op 1340) 


v_lshl_add_u32 v10, s7, 2, v17 // (op 1340) 


v_mul_lo_u32 v17, v14, s34 // (op 1340) 
v_add_u32 v17, v17, v10 // (op 1340) 


s_mul_i32 s19, s34, s29 // (op 1342) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1342) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1342) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1342) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1342) most significant: sum
s_add_u32 s19, s19, s6 // (op 1342) most significant: sum


s_mov_b32 s68, s36 // (op 1342) reg expression
s_mov_b32 s69, s37 // (op 1342) reg expression
s_mov_b32 s70, s18 // (op 1342) reg expression
s_mov_b32 s71, 131072 // (op 1342) call()

v_and_b32 v10, 15, v1 // (op 1468) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1468) 
v_and_b32 v14, 63, v1 // (op 1468) 
s_and_b32 s15, s18, 4294967294 // (op 1468) 
v_lshrrev_b32 v19, 4, v14 // (op 1468) 

s_add_u32 s6, s0, s15 // (op 1468) 

s_mul_i32 s15, s16, s50 // (op 1468) 
s_add_u32 s15, s15, s6 // (op 1468) 

s_add_u32 s6, s15, s9 // (op 1468) 

s_mul_hi_u32 s15, s6, s13 // (op 1468) 
s_sub_u32 s7, s6, s15 // (op 1468) 
s_lshr_b32 s8, s7, 1 // (op 1468) 

s_add_u32 s7, s8, s15 // (op 1468) 


s_lshr_b32 s8, s7, s11 // (op 1468) 

s_and_b32 s7, s11, 2147483648 // (op 1468) 
s_cmp_eq_i32 s7, 0 // (op 1468) 
s_mov_b32 s15, scc // (op 1468) 

s_cmp_eq_i32 s15, 0 // (op 1468) 
s_cselect_b32 s7, s6, s8 // (op 1468) 


s_mul_hi_u32 s8, s7, s12 // (op 1468) 
s_sub_u32 s15, s7, s8 // (op 1468) 
s_lshr_b32 s42, s15, 1 // (op 1468) 

s_add_u32 s15, s42, s8 // (op 1468) 


s_lshr_b32 s8, s15, s10 // (op 1468) 

s_and_b32 s15, s10, 2147483648 // (op 1468) 
s_cmp_eq_i32 s15, 0 // (op 1468) 
s_mov_b32 s42, scc // (op 1468) 

s_cmp_eq_i32 s42, 0 // (op 1468) 
s_cselect_b32 s15, s7, s8 // (op 1468) 


s_mul_i32 s8, s15, s14 // (op 1468) 

s_sub_u32 s15, s7, s8 // (op 1468) 

v_lshl_add_u32 v14, s15, 4, v10 // (op 1468) 


s_mul_i32 s8, s7, s18 // (op 1468) 


v_mul_lo_u32 v10, v14, s38 // (op 1468) 

s_sub_u32 s7, s6, s8 // (op 1468) 


v_lshl_add_u32 v14, s7, 2, v19 // (op 1468) 


v_add_u32 v19, v14, v10 // (op 1468) 


s_mul_i32 s19, s38, s23 // (op 1470) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 1470) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 1470) least significant
s_mul_hi_u32 s6, s38, s22 // (op 1470) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1470) most significant: sum
s_add_u32 s19, s19, s6 // (op 1470) most significant: sum


s_mov_b32 s72, s40 // (op 1470) reg expression
s_mov_b32 s73, s41 // (op 1470) reg expression
s_mov_b32 s74, s18 // (op 1470) reg expression
s_mov_b32 s75, 131072 // (op 1470) call()

v_and_b32 v10, -8, v1 // (op 1907) 
v_sub_u32 v14, v1, v10 // (op 1907) 

v_lshlrev_b32 v10, 4, v14 // (op 1907) 

v_lshrrev_b32 v14, 3, v1 // (op 1907) 
v_lshl_add_u32 v29, v14, 7, v10 // (op 1907) 


s_nop 0
v_readfirstlane_b32 s15, v29 // (op 1907) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -8, v1 // (op 1972) 
v_sub_u32 v14, v1, v10 // (op 1972) 

v_lshlrev_b32 v10, 4, v14 // (op 1972) 

v_lshrrev_b32 v14, 3, v1 // (op 1972) 
v_lshl_add_u32 v29, v14, 7, v10 // (op 1972) 


s_nop 0
v_readfirstlane_b32 s6, v29 // (op 1972) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -4, v1 // (op 2037) 
v_sub_u32 v14, v1, v10 // (op 2037) 

v_and_b32 v10, 4294967292, v1 // (op 2037) 
v_add_u32 v29, v10, v14 // (op 2037) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 434) 
s_and_b32 s7, s18, 4294967294 // (op 434) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 433) 
s_mov_b32 s8, s18 // (op 433) convert

s_cmp_lt_u32 s7, s8 // (op 433) 






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_433 // (op 433) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_433 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_433:
s_barrier  // (op 2453) 
s_add_u32 s7, s7, 1 // (op 435) 

s_lshl_b64 s[18:19], s[24:25], 4 // (op 991) 
v_mov_b64 v[22:23], s[18:19] // (op 991) convert


v_mov_b64 v[26:27], v[24:25] // (op 991) 
v_mov_b64 v[30:31], v[26:27] // (op 991) 
s_add_u32 s8, 2048, s15 // (op 991) 
s_mov_b32 m0, s8 // (op 991) reg expression
buffer_load_dwordx4 v30, s[60:63], 0 offen offset: 0 lds // (op 991) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v22 // (op 991) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 991) most significant half

s_add_u32 m0, m0, 2048 // (op 991) 
buffer_load_dwordx4 v30, s[60:63], 0 offen offset: 0 lds // (op 991) Load value direct to lds




v_add_co_u32 v24, vcc, 128, v24 // (op 1066) least significant half
v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 1066) most significant half
s_barrier  // (op 2472) 


v_mov_b64 v[22:23], v[20:21] // (op 997) 

s_mov_b32 m0, s6 // (op 997) reg expression
buffer_load_dwordx4 v22, s[64:67], 0 offen offset: 0 lds // (op 997) Load value direct to lds

v_add_co_u32 v20, vcc, 128, v20 // (op 1204) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 1204) most significant half


v_mov_b32 v10, v17 // (op 452) 

buffer_load_ubyte v14, v10, s[68:71], 0 offen offset: 0 // (op 452) Load value

v_add_u32 v17, 4, v17 // (op 1332) 
s_barrier  // (op 2491) 
s_waitcnt vmcnt(2)
s_barrier  // (op 2459) 






ds_read_b128 v[30:33], v6 offset:2048 // (op 441) Load local data 
ds_read_b128 v[34:37], v6 offset:2112 // (op 441) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v29, v14 offset:12288 // (op 450) Store local data 

s_barrier  // (op 2478) 



ds_read_b128 v[38:41], v8 // (op 443) Load local data 
ds_read_b128 v[42:45], v8 offset:64 // (op 443) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2498) 



ds_read_u8 v10, v4 offset:12288 // (op 445) Load local data 


v_mov_b32 v14, v19 // (op 447) 

buffer_load_ubyte v16, v14, s[72:75], 0 offen offset: 0 // (op 447) Load value

v_add_u32 v19, 4, v19 // (op 1460) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:45], a[0:3], v10, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 448) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 433) 
s_mov_b32 s8, s18 // (op 433) convert

s_cmp_lt_u32 s7, s8 // (op 433) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_433 // (op 433) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_433 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_433:








s_barrier  // (op 2626) 



v_cmp_eq_f32 vcc, 0.00000, v13 // (op 218) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_218 // (op 218) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_218
s_nop 0
v_accvgpr_read v16, a0 // (op 226) Wait state hazard: XDL Write Hazard
v_accvgpr_read v17, a1 // (op 226) 
v_accvgpr_read v18, a2 // (op 226) 
v_accvgpr_read v19, a3 // (op 226) 
v_mul_f32 v20, v15, v16 // (op 226) 
v_mul_f32 v21, v15, v17 // (op 226) 
v_mul_f32 v22, v15, v18 // (op 226) 
v_mul_f32 v23, v15, v19 // (op 226) 

v_lshrrev_b32 v0, 6, v1 // (op 1860) 
s_mul_i32 s15, s16, s50 // (op 1860) 
s_add_u32 s15, s15, s0 // (op 1860) 
v_and_b32 v5, 63, v1 // (op 1860) 
s_add_u32 s6, s15, s9 // (op 1860) 

v_lshrrev_b32 v10, 4, v5 // (op 1860) 
s_mul_hi_u32 s15, s6, s13 // (op 1860) 
v_lshlrev_b32 v12, 2, v10 // (op 1860) 

s_sub_u32 s7, s6, s15 // (op 1860) 
v_lshl_add_u32 v10, v0, 4, v12 // (op 1860) 


s_lshr_b32 s8, s7, 1 // (op 1860) 

v_lshrrev_b32 v0, 2, v5 // (op 1860) 

s_add_u32 s7, s8, s15 // (op 1860) 


v_and_b32 v5, 3, v0 // (op 1860) 

s_lshr_b32 s8, s7, s11 // (op 1860) 

v_and_b32 v0, 3, v1 // (op 1860) 
s_and_b32 s7, s11, 2147483648 // (op 1860) 
v_lshl_add_u32 v12, v5, 2, v0 // (op 1860) 


s_cmp_eq_i32 s7, 0 // (op 1860) 
s_mov_b32 s15, scc // (op 1860) 

s_cmp_eq_i32 s15, 0 // (op 1860) 
s_cselect_b32 s7, s6, s8 // (op 1860) 



s_mul_hi_u32 s6, s7, s12 // (op 1860) 
s_sub_u32 s8, s7, s6 // (op 1860) 
s_lshr_b32 s15, s8, 1 // (op 1860) 

s_add_u32 s8, s15, s6 // (op 1860) 


s_lshr_b32 s6, s8, s10 // (op 1860) 

s_and_b32 s8, s10, 2147483648 // (op 1860) 
s_cmp_eq_i32 s8, 0 // (op 1860) 
s_mov_b32 s15, scc // (op 1860) 

s_cmp_eq_i32 s15, 0 // (op 1860) 
s_cselect_b32 s8, s7, s6 // (op 1860) 


s_mul_i32 s6, s8, s14 // (op 1860) 
v_lshl_add_u32 v0, s8, 5, v10 // (op 1860) 


s_sub_u32 s8, s7, s6 // (op 1860) 


v_lshl_add_u32 v5, s8, 4, v12 // (op 1860) 


v_mul_lo_u32 v10, v5, s46 // (op 1860) 

v_add_lshl_u32 v5, v0, v10, 2 // (op 1860) 


s_mul_i32 s19, s46, s23 // (op 1862) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1862) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1862) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1862) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1862) most significant: sum
s_add_u32 s19, s19, s6 // (op 1862) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1862) 

s_mov_b32 s76, s48 // (op 1862) reg expression
s_mov_b32 s77, s49 // (op 1862) reg expression
s_mov_b32 s78, s6 // (op 1862) reg expression
s_mov_b32 s79, 131072 // (op 1862) call()




buffer_store_dwordx4 v[20:23], v5, s[76:79], 0 offen // (op 228) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_218 // (op 218) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_218
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_218:



buffer_load_dwordx4 v[16:19], v3, s[56:59], 0 offen // (op 241) Load value
v_accvgpr_read v20, a0 // (op 238) 
v_accvgpr_read v21, a1 // (op 238) 
v_accvgpr_read v22, a2 // (op 238) 
v_accvgpr_read v23, a3 // (op 238) 
v_mul_f32 v24, v15, v20 // (op 238) 
v_mul_f32 v25, v15, v21 // (op 238) 
v_mul_f32 v26, v15, v22 // (op 238) 
v_mul_f32 v27, v15, v23 // (op 238) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v13, v16 // (op 242) 
v_mul_f32 v21, v13, v17 // (op 242) 
v_mul_f32 v22, v13, v18 // (op 242) 
v_mul_f32 v23, v13, v19 // (op 242) 

v_add_f32 v16, v20, v24 // (op 239) 
v_add_f32 v17, v21, v25 // (op 239) 
v_add_f32 v18, v22, v26 // (op 239) 
v_add_f32 v19, v23, v27 // (op 239) 


v_lshrrev_b32 v0, 6, v1 // (op 1877) 
s_mul_i32 s15, s16, s50 // (op 1877) 
s_add_u32 s15, s15, s0 // (op 1877) 
v_and_b32 v5, 63, v1 // (op 1877) 
s_add_u32 s6, s15, s9 // (op 1877) 

v_lshrrev_b32 v10, 4, v5 // (op 1877) 
s_mul_hi_u32 s15, s6, s13 // (op 1877) 
v_lshlrev_b32 v12, 2, v10 // (op 1877) 

s_sub_u32 s7, s6, s15 // (op 1877) 
v_lshl_add_u32 v10, v0, 4, v12 // (op 1877) 


s_lshr_b32 s8, s7, 1 // (op 1877) 

v_lshrrev_b32 v0, 2, v5 // (op 1877) 

s_add_u32 s7, s8, s15 // (op 1877) 


v_and_b32 v5, 3, v0 // (op 1877) 

s_lshr_b32 s8, s7, s11 // (op 1877) 

v_and_b32 v0, 3, v1 // (op 1877) 
s_and_b32 s7, s11, 2147483648 // (op 1877) 
v_lshl_add_u32 v12, v5, 2, v0 // (op 1877) 


s_cmp_eq_i32 s7, 0 // (op 1877) 
s_mov_b32 s15, scc // (op 1877) 

s_cmp_eq_i32 s15, 0 // (op 1877) 
s_cselect_b32 s7, s6, s8 // (op 1877) 



s_mul_hi_u32 s6, s7, s12 // (op 1877) 
s_sub_u32 s8, s7, s6 // (op 1877) 
s_lshr_b32 s15, s8, 1 // (op 1877) 

s_add_u32 s8, s15, s6 // (op 1877) 


s_lshr_b32 s6, s8, s10 // (op 1877) 

s_and_b32 s8, s10, 2147483648 // (op 1877) 
s_cmp_eq_i32 s8, 0 // (op 1877) 
s_mov_b32 s15, scc // (op 1877) 

s_cmp_eq_i32 s15, 0 // (op 1877) 
s_cselect_b32 s8, s7, s6 // (op 1877) 


s_mul_i32 s6, s8, s14 // (op 1877) 
v_lshl_add_u32 v0, s8, 5, v10 // (op 1877) 


s_sub_u32 s8, s7, s6 // (op 1877) 


v_lshl_add_u32 v5, s8, 4, v12 // (op 1877) 


v_mul_lo_u32 v10, v5, s46 // (op 1877) 

v_add_lshl_u32 v5, v0, v10, 2 // (op 1877) 


s_mul_i32 s19, s46, s23 // (op 1879) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1879) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1879) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1879) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1879) most significant: sum
s_add_u32 s19, s19, s6 // (op 1879) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1879) 

s_mov_b32 s76, s48 // (op 1879) reg expression
s_mov_b32 s77, s49 // (op 1879) reg expression
s_mov_b32 s78, s6 // (op 1879) reg expression
s_mov_b32 s79, 131072 // (op 1879) call()




buffer_store_dwordx4 v[16:19], v5, s[76:79], 0 offen // (op 240) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_218:

s_add_u32 s0, s0, s1 // (op 427) 
s_mul_i32 s15, s50, s16 // (op 425) 
s_cmp_gt_u32 s51, s15 // (op 425) 
s_mov_b32 s6, scc // (op 425) 
s_sub_u32 s7, s51, s15 // (op 425) 

s_cmp_eq_i32 s6, 0 // (op 425) 
s_cselect_b32 s8, 0, s7 // (op 425) 


s_cmp_lt_u32 s50, s8 // (op 425) 
s_mov_b32 s15, scc // (op 425) 
s_cmp_eq_i32 s15, 0 // (op 425) 
s_cselect_b32 s6, s8, s50 // (op 425) 


s_cmp_lt_u32 s0, s6 // (op 425) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_DPStreamTileLoop_425 // (op 425) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_DPStreamTileLoop_425 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_425:








































s_endpgm  // End of RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2
.LRR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_end:
.size RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2, .LRR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_end-RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2
  .amdhsa_next_free_vgpr 64
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 12544 // lds bytes
  .amdhsa_accum_offset 60
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
    occupancy: 13
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
  - .name: RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2
    .symbol: RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 12544
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 82
    .vgpr_count: 58
    .agpr_count: 4
    .max_flat_workgroup_size: 128
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

