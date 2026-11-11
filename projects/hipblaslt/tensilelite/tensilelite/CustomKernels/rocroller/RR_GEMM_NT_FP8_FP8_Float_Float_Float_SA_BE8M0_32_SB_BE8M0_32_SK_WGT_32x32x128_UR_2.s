// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
.p2align 8
.type RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2,@function
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 1599) 
v_and_b32 v3, 63, v1 // (op 1599) 
v_sub_u32 v4, v3, v0 // (op 1599) 

v_lshrrev_b32 v3, 1, v4 // (op 1599) 
v_add_lshl_u32 v5, v0, v3, 5 // (op 1599) 


v_and_b32 v0, 1, v4 // (op 1599) 

v_lshlrev_b32 v3, 3, v0 // (op 1599) 

v_lshrrev_b32 v0, 7, v1 // (op 1599) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1599) 


v_add_u32 v0, v5, v4 // (op 1599) 


v_and_b32 v3, 48, v1 // (op 1664) 
v_and_b32 v4, 63, v1 // (op 1664) 
v_sub_u32 v5, v4, v3 // (op 1664) 

v_lshrrev_b32 v4, 1, v5 // (op 1664) 
v_add_lshl_u32 v6, v3, v4, 5 // (op 1664) 


v_and_b32 v3, 1, v5 // (op 1664) 

v_lshlrev_b32 v4, 3, v3 // (op 1664) 

v_lshrrev_b32 v3, 6, v1 // (op 1664) 
v_and_b32 v5, 1, v3 // (op 1664) 

v_lshl_add_u32 v3, v5, 4, v4 // (op 1664) 


v_add_u32 v4, v6, v3 // (op 1664) 


v_and_b32 v3, 48, v1 // (op 1799) 
v_and_b32 v5, 63, v1 // (op 1799) 
v_sub_u32 v6, v5, v3 // (op 1799) 

v_lshrrev_b32 v5, 1, v6 // (op 1799) 
v_add_lshl_u32 v7, v3, v5, 5 // (op 1799) 


v_and_b32 v3, 1, v6 // (op 1799) 

v_lshlrev_b32 v5, 3, v3 // (op 1799) 

v_lshrrev_b32 v3, 7, v1 // (op 1799) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 1799) 


v_add_u32 v3, v7, v6 // (op 1799) 


v_and_b32 v5, 48, v1 // (op 1864) 
v_and_b32 v6, 63, v1 // (op 1864) 
v_sub_u32 v7, v6, v5 // (op 1864) 

v_lshrrev_b32 v6, 1, v7 // (op 1864) 
v_add_lshl_u32 v8, v5, v6, 5 // (op 1864) 


v_and_b32 v5, 1, v7 // (op 1864) 

v_lshlrev_b32 v6, 3, v5 // (op 1864) 

v_lshrrev_b32 v5, 6, v1 // (op 1864) 
v_and_b32 v7, 1, v5 // (op 1864) 

v_lshl_add_u32 v5, v7, 4, v6 // (op 1864) 


v_add_u32 v6, v8, v5 // (op 1864) 


v_and_b32 v5, 48, v1 // (op 1950) 
v_and_b32 v7, 63, v1 // (op 1950) 
v_sub_u32 v8, v7, v5 // (op 1950) 

v_lshrrev_b32 v7, 1, v8 // (op 1950) 
v_add_lshl_u32 v9, v5, v7, 5 // (op 1950) 


v_and_b32 v5, 1, v8 // (op 1950) 

v_lshlrev_b32 v7, 3, v5 // (op 1950) 

v_lshrrev_b32 v5, 7, v1 // (op 1950) 
v_lshl_add_u32 v8, v5, 4, v7 // (op 1950) 


v_add_u32 v5, v9, v8 // (op 1950) 


v_and_b32 v7, 48, v1 // (op 1989) 
v_and_b32 v8, 63, v1 // (op 1989) 
v_sub_u32 v9, v8, v7 // (op 1989) 

v_lshrrev_b32 v8, 1, v9 // (op 1989) 
v_add_lshl_u32 v10, v7, v8, 5 // (op 1989) 


v_and_b32 v7, 1, v9 // (op 1989) 

v_lshlrev_b32 v8, 3, v7 // (op 1989) 

v_lshrrev_b32 v7, 6, v1 // (op 1989) 
v_and_b32 v9, 1, v7 // (op 1989) 

v_lshl_add_u32 v7, v9, 4, v8 // (op 1989) 


v_add_u32 v8, v10, v7 // (op 1989) 


v_and_b32 v7, 48, v1 // (op 2028) 
v_and_b32 v9, 63, v1 // (op 2028) 
v_sub_u32 v10, v9, v7 // (op 2028) 

v_lshrrev_b32 v9, 1, v10 // (op 2028) 
v_add_lshl_u32 v11, v7, v9, 5 // (op 2028) 


v_and_b32 v7, 1, v10 // (op 2028) 

v_lshlrev_b32 v9, 3, v7 // (op 2028) 

v_lshrrev_b32 v7, 7, v1 // (op 2028) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 2028) 


v_add_u32 v7, v11, v10 // (op 2028) 


v_and_b32 v9, 48, v1 // (op 2067) 
v_and_b32 v10, 63, v1 // (op 2067) 
v_sub_u32 v11, v10, v9 // (op 2067) 

v_lshrrev_b32 v10, 1, v11 // (op 2067) 
v_add_lshl_u32 v12, v9, v10, 5 // (op 2067) 


v_and_b32 v9, 1, v11 // (op 2067) 

v_lshlrev_b32 v10, 3, v9 // (op 2067) 

v_lshrrev_b32 v9, 6, v1 // (op 2067) 
v_and_b32 v11, 1, v9 // (op 2067) 

v_lshl_add_u32 v9, v11, 4, v10 // (op 2067) 


v_add_u32 v10, v12, v9 // (op 2067) 


v_mov_b32 v9, s6 // (op 2) Move value
v_mov_b32 v11, s7 // (op 35) Move value


s_mov_b32 s0, 0 // (op 356) call()
s_mov_b32 s1, 0 // (op 357) call()
s_mov_b32 s6, 0 // (op 358) call()
s_mov_b32 s7, 0 // (op 359) call()
s_mov_b32 s18, 0 // (op 360) call()
s_mov_b32 s19, 0 // (op 361) call()
s_mov_b32 s0, 0 // (op 256) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 255) 
s_cmp_gt_u32 s9, s52 // (op 255) 
s_mov_b32 s53, scc // (op 255) 
s_sub_u32 s54, s9, s52 // (op 255) 

s_cmp_eq_i32 s53, 0 // (op 255) 
s_cselect_b32 s52, 0, s54 // (op 255) 


s_cmp_lt_u32 s8, s52 // (op 255) 
s_mov_b32 s53, scc // (op 255) 
s_cmp_eq_i32 s53, 0 // (op 255) 
s_cselect_b32 s54, s52, s8 // (op 255) 


s_cmp_lt_u32 s0, s54 // (op 255) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_255 // (op 255) Condition: Top (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_255 if false)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_SKStreamTileLoop_255:
v_lshrrev_b32 v12, 7, v1 // (op 1525) 
s_mul_i32 s52, s16, s8 // (op 1525) 
s_add_u32 s52, s52, s0 // (op 1525) 
v_and_b32 v13, 63, v1 // (op 1525) 
s_mul_hi_u32 s53, s52, s13 // (op 1525) 
v_lshrrev_b32 v14, 4, v13 // (op 1525) 
s_sub_u32 s54, s52, s53 // (op 1525) 
v_lshlrev_b32 v15, 2, v14 // (op 1525) 

v_lshl_add_u32 v14, v12, 4, v15 // (op 1525) 


s_lshr_b32 s55, s54, 1 // (op 1525) 

s_add_u32 s54, s55, s53 // (op 1525) 


v_lshrrev_b32 v12, 2, v13 // (op 1525) 

s_lshr_b32 s53, s54, s11 // (op 1525) 

v_and_b32 v13, 3, v12 // (op 1525) 

s_and_b32 s54, s11, 2147483648 // (op 1525) 
v_and_b32 v12, 3, v1 // (op 1525) 
s_cmp_eq_i32 s54, 0 // (op 1525) 
s_mov_b32 s55, scc // (op 1525) 

v_lshl_add_u32 v15, v13, 2, v12 // (op 1525) 


s_cmp_eq_i32 s55, 0 // (op 1525) 
s_cselect_b32 s54, s52, s53 // (op 1525) 



v_lshrrev_b32 v12, 6, v1 // (op 1525) 
s_mul_hi_u32 s52, s54, s12 // (op 1525) 
v_and_b32 v13, 1, v12 // (op 1525) 

s_sub_u32 s53, s54, s52 // (op 1525) 
v_lshl_add_u32 v12, v13, 4, v15 // (op 1525) 


s_lshr_b32 s55, s53, 1 // (op 1525) 

s_add_u32 s53, s55, s52 // (op 1525) 


s_lshr_b32 s52, s53, s10 // (op 1525) 

s_and_b32 s53, s10, 2147483648 // (op 1525) 
s_cmp_eq_i32 s53, 0 // (op 1525) 
s_mov_b32 s55, scc // (op 1525) 

s_cmp_eq_i32 s55, 0 // (op 1525) 
s_cselect_b32 s53, s54, s52 // (op 1525) 


s_mul_i32 s52, s53, s14 // (op 1525) 
v_lshl_add_u32 v13, s53, 5, v14 // (op 1525) 


s_sub_u32 s53, s54, s52 // (op 1525) 


v_lshl_add_u32 v14, s53, 5, v12 // (op 1525) 


v_mul_lo_u32 v12, v14, s4 // (op 1525) 

v_add_lshl_u32 v14, v13, v12, 2 // (op 1525) 


s_mul_i32 s53, s4, s23 // (op 1527) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 1527) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 1527) least significant
s_mul_hi_u32 s55, s4, s22 // (op 1527) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1527) most significant: sum
s_add_u32 s53, s53, s55 // (op 1527) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1527) 

s_mov_b32 s56, s20 // (op 1527) reg expression
s_mov_b32 s57, s21 // (op 1527) reg expression
s_mov_b32 s58, s54 // (op 1527) reg expression
s_mov_b32 s59, 131072 // (op 1527) call()

v_and_b32 v12, 48, v1 // (op 1611) 
v_and_b32 v13, 63, v1 // (op 1611) 
v_sub_u32 v15, v13, v12 // (op 1611) 

v_lshrrev_b32 v13, 1, v15 // (op 1611) 
v_add_lshl_u32 v16, v12, v13, 5 // (op 1611) 


v_and_b32 v12, 1, v15 // (op 1611) 

v_lshlrev_b32 v13, 3, v12 // (op 1611) 

v_lshrrev_b32 v12, 7, v1 // (op 1611) 
v_lshl_add_u32 v15, v12, 4, v13 // (op 1611) 


v_add_u32 v12, v16, v15 // (op 1611) 


v_and_b32 v13, 48, v1 // (op 1676) 
v_and_b32 v15, 63, v1 // (op 1676) 
v_sub_u32 v16, v15, v13 // (op 1676) 

v_lshrrev_b32 v15, 1, v16 // (op 1676) 
v_add_lshl_u32 v17, v13, v15, 5 // (op 1676) 


v_and_b32 v13, 1, v16 // (op 1676) 

v_lshlrev_b32 v15, 3, v13 // (op 1676) 

v_lshrrev_b32 v13, 6, v1 // (op 1676) 
v_and_b32 v16, 1, v13 // (op 1676) 

v_lshl_add_u32 v13, v16, 4, v15 // (op 1676) 


v_add_u32 v15, v17, v13 // (op 1676) 


s_mul_i32 s6, s8, s16 // (op 260) 
s_add_u32 s6, s6, s0 // (op 260) 
s_mul_hi_u32 s52, s6, s13 // (op 261) 
s_sub_u32 s53, s6, s52 // (op 261) 
s_lshr_b32 s54, s53, 1 // (op 261) 

s_add_u32 s53, s54, s52 // (op 261) 


s_lshr_b32 s52, s53, s11 // (op 261) 

s_and_b32 s53, s11, 2147483648 // (op 261) 
s_cmp_eq_i32 s53, 0 // (op 261) 
s_mov_b32 s54, scc // (op 261) 

s_cmp_eq_i32 s54, 0 // (op 261) 
s_cselect_b32 s53, s6, s52 // (op 261) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 261) 
s_mul_i32 s52, s53, s54 // (op 261) 


s_sub_u32 s18, s6, s52 // (op 261) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 262) 
s_cmp_gt_u32 s52, s18 // (op 262) 
s_mov_b32 s54, scc // (op 262) 
s_sub_u32 s55, s52, s18 // (op 262) 

s_cmp_eq_i32 s54, 0 // (op 262) 
s_cselect_b32 s52, 0, s55 // (op 262) 


s_cmp_gt_u32 s8, s0 // (op 262) 
s_mov_b32 s53, scc // (op 262) 
s_sub_u32 s54, s8, s0 // (op 262) 
s_cmp_eq_i32 s53, 0 // (op 262) 
s_cselect_b32 s55, 0, s54 // (op 262) 


s_cmp_lt_u32 s52, s55 // (op 262) 
s_mov_b32 s53, scc // (op 262) 
s_cmp_eq_i32 s53, 0 // (op 262) 
s_cselect_b32 s54, s55, s52 // (op 262) 



s_cmp_gt_u32 s9, s6 // (op 262) 
s_mov_b32 s55, scc // (op 262) 
s_sub_u32 s52, s9, s6 // (op 262) 
s_cmp_eq_i32 s55, 0 // (op 262) 
s_cselect_b32 s53, 0, s52 // (op 262) 


s_cmp_lt_u32 s54, s53 // (op 262) 
s_mov_b32 s52, scc // (op 262) 
s_cmp_eq_i32 s52, 0 // (op 262) 
s_cselect_b32 s7, s53, s54 // (op 262) 



s_add_u32 s52, s18, s7 // (op 263) 
s_sub_u32 s19, s52, 1 // (op 263) 

v_accvgpr_write a0, 0.00000 // (op 55) call()
v_accvgpr_write a1, 0.00000 // (op 55) call()
v_accvgpr_write a2, 0.00000 // (op 55) call()
v_accvgpr_write a3, 0.00000 // (op 55) call()
v_and_b32 v13, -2, v1 // (op 1057) 
s_mul_i32 s52, s16, s8 // (op 1057) 
s_add_u32 s52, s52, s0 // (op 1057) 
v_sub_u32 v16, v1, v13 // (op 1057) 

s_mul_hi_u32 s53, s52, s13 // (op 1057) 
v_lshlrev_b32 v13, 4, v16 // (op 1057) 

s_sub_u32 s54, s52, s53 // (op 1057) 
v_lshrrev_b32 v16, 1, v1 // (op 1057) 
s_lshr_b32 s55, s54, 1 // (op 1057) 

s_add_u32 s54, s55, s53 // (op 1057) 


s_lshr_b32 s53, s54, s11 // (op 1057) 

s_and_b32 s54, s11, 2147483648 // (op 1057) 
s_cmp_eq_i32 s54, 0 // (op 1057) 
s_mov_b32 s55, scc // (op 1057) 

s_cmp_eq_i32 s55, 0 // (op 1057) 
s_cselect_b32 s54, s52, s53 // (op 1057) 


s_mul_hi_u32 s53, s54, s12 // (op 1057) 
s_sub_u32 s55, s54, s53 // (op 1057) 
s_lshr_b32 s60, s55, 1 // (op 1057) 

s_add_u32 s55, s60, s53 // (op 1057) 


s_lshr_b32 s53, s55, s10 // (op 1057) 

s_and_b32 s55, s10, 2147483648 // (op 1057) 
s_cmp_eq_i32 s55, 0 // (op 1057) 
s_mov_b32 s60, scc // (op 1057) 

s_cmp_eq_i32 s60, 0 // (op 1057) 
s_cselect_b32 s55, s54, s53 // (op 1057) 


v_lshl_add_u32 v17, s55, 5, v13 // (op 1057) 


s_ashr_i64 s[60:61], s[2:3], 7 // (op 1057) 
v_mov_b32 v19, 0 // (op 1057) convert
v_mov_b32 v18, v17 // (op 1057) convert

s_mul_i32 s53, s54, s60 // (op 1057) 


s_sub_u32 s54, s52, s53 // (op 1057) 


v_lshl_add_u32 v13, s54, 7, v16 // (op 1057) 


v_mul_lo_u32 v17, v13, s25 // (op 1057) most significant: low of low * high
v_mul_hi_u32 v20, v13, s24 // (op 1057) most significant: high of low * low
v_mul_lo_u32 v16, v13, s24 // (op 1057) least significant: low of low * low
v_add_u32 v17, v17, v20 // (op 1057) most significant: sum


v_add_co_u32 v20, s[52:53], v18, v16 // (op 1057) least significant half
v_addc_co_u32 v21, s[52:53], v19, v17, s[52:53] // (op 1057) most significant half



s_mul_i32 s53, s24, s3 // (op 1059) most significant: low of low * high
s_mul_i32 s54, s25, s2 // (op 1059) most significant: low of high * low
s_mul_i32 s52, s24, s2 // (op 1059) least significant
s_mul_hi_u32 s55, s24, s2 // (op 1059) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1059) most significant: sum
s_add_u32 s53, s53, s55 // (op 1059) most significant: sum


s_mov_b32 s60, s26 // (op 1059) reg expression
s_mov_b32 s61, s27 // (op 1059) reg expression
s_mov_b32 s62, s52 // (op 1059) reg expression
s_mov_b32 s63, 131072 // (op 1059) call()

v_and_b32 v13, -2, v1 // (op 1076) 
s_mul_i32 s52, s16, s8 // (op 1076) 
s_add_u32 s52, s52, s0 // (op 1076) 
v_sub_u32 v16, v1, v13 // (op 1076) 

s_mul_hi_u32 s53, s52, s13 // (op 1076) 
v_lshlrev_b32 v13, 4, v16 // (op 1076) 

s_sub_u32 s54, s52, s53 // (op 1076) 
v_lshrrev_b32 v16, 1, v1 // (op 1076) 
s_lshr_b32 s55, s54, 1 // (op 1076) 

s_add_u32 s54, s55, s53 // (op 1076) 


s_lshr_b32 s53, s54, s11 // (op 1076) 

s_and_b32 s54, s11, 2147483648 // (op 1076) 
s_cmp_eq_i32 s54, 0 // (op 1076) 
s_mov_b32 s55, scc // (op 1076) 

s_cmp_eq_i32 s55, 0 // (op 1076) 
s_cselect_b32 s54, s52, s53 // (op 1076) 


s_mul_hi_u32 s53, s54, s12 // (op 1076) 
s_sub_u32 s55, s54, s53 // (op 1076) 
s_lshr_b32 s64, s55, 1 // (op 1076) 

s_add_u32 s55, s64, s53 // (op 1076) 


s_lshr_b32 s53, s55, s10 // (op 1076) 

s_and_b32 s55, s10, 2147483648 // (op 1076) 
s_cmp_eq_i32 s55, 0 // (op 1076) 
s_mov_b32 s64, scc // (op 1076) 

s_cmp_eq_i32 s64, 0 // (op 1076) 
s_cselect_b32 s55, s54, s53 // (op 1076) 


v_lshl_add_u32 v17, s55, 5, v13 // (op 1076) 


s_ashr_i64 s[64:65], s[2:3], 7 // (op 1076) 
v_mov_b32 v19, 0 // (op 1076) convert
v_mov_b32 v18, v17 // (op 1076) convert

s_mul_i32 s53, s54, s64 // (op 1076) 


s_sub_u32 s54, s52, s53 // (op 1076) 


v_lshl_add_u32 v13, s54, 7, v16 // (op 1076) 


v_mul_lo_u32 v17, v13, s25 // (op 1076) most significant: low of low * high
v_mul_hi_u32 v22, v13, s24 // (op 1076) most significant: high of low * low
v_mul_lo_u32 v16, v13, s24 // (op 1076) least significant: low of low * low
v_add_u32 v17, v17, v22 // (op 1076) most significant: sum


v_add_co_u32 v22, s[52:53], v18, v16 // (op 1076) least significant half
v_addc_co_u32 v23, s[52:53], v19, v17, s[52:53] // (op 1076) most significant half



s_mul_i32 s53, s24, s3 // (op 1078) most significant: low of low * high
s_mul_i32 s54, s25, s2 // (op 1078) most significant: low of high * low
s_mul_i32 s52, s24, s2 // (op 1078) least significant
s_mul_hi_u32 s55, s24, s2 // (op 1078) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1078) most significant: sum
s_add_u32 s53, s53, s55 // (op 1078) most significant: sum


s_mov_b32 s60, s26 // (op 1078) reg expression
s_mov_b32 s61, s27 // (op 1078) reg expression
s_mov_b32 s62, s52 // (op 1078) reg expression
s_mov_b32 s63, 131072 // (op 1078) call()

v_and_b32 v13, -2, v1 // (op 1188) 
s_mul_i32 s52, s16, s8 // (op 1188) 
s_add_u32 s52, s52, s0 // (op 1188) 
v_sub_u32 v16, v1, v13 // (op 1188) 

s_mul_hi_u32 s53, s52, s13 // (op 1188) 
v_lshlrev_b32 v13, 4, v16 // (op 1188) 

s_sub_u32 s54, s52, s53 // (op 1188) 
v_lshrrev_b32 v16, 1, v1 // (op 1188) 
s_lshr_b32 s55, s54, 1 // (op 1188) 

s_add_u32 s54, s55, s53 // (op 1188) 


s_lshr_b32 s53, s54, s11 // (op 1188) 

s_and_b32 s54, s11, 2147483648 // (op 1188) 
s_cmp_eq_i32 s54, 0 // (op 1188) 
s_mov_b32 s55, scc // (op 1188) 

s_cmp_eq_i32 s55, 0 // (op 1188) 
s_cselect_b32 s54, s52, s53 // (op 1188) 


s_mul_hi_u32 s53, s54, s12 // (op 1188) 
s_sub_u32 s55, s54, s53 // (op 1188) 
s_lshr_b32 s64, s55, 1 // (op 1188) 

s_add_u32 s55, s64, s53 // (op 1188) 


s_lshr_b32 s53, s55, s10 // (op 1188) 

s_and_b32 s55, s10, 2147483648 // (op 1188) 
s_cmp_eq_i32 s55, 0 // (op 1188) 
s_mov_b32 s64, scc // (op 1188) 

s_cmp_eq_i32 s64, 0 // (op 1188) 
s_cselect_b32 s55, s54, s53 // (op 1188) 


s_mul_i32 s53, s55, s14 // (op 1188) 

s_sub_u32 s55, s54, s53 // (op 1188) 

v_lshl_add_u32 v17, s55, 5, v13 // (op 1188) 


s_ashr_i64 s[64:65], s[2:3], 7 // (op 1188) 
v_mov_b32 v19, 0 // (op 1188) convert
v_mov_b32 v18, v17 // (op 1188) convert

s_mul_i32 s53, s54, s64 // (op 1188) 


s_sub_u32 s54, s52, s53 // (op 1188) 


v_lshl_add_u32 v13, s54, 7, v16 // (op 1188) 


v_mul_lo_u32 v17, v13, s29 // (op 1188) most significant: low of low * high
v_mul_hi_u32 v24, v13, s28 // (op 1188) most significant: high of low * low
v_mul_lo_u32 v16, v13, s28 // (op 1188) least significant: low of low * low
v_add_u32 v17, v17, v24 // (op 1188) most significant: sum

v_add_co_u32 v16, s[52:53], v16, v18 // (op 1188) least significant half
v_addc_co_u32 v17, s[52:53], v17, v19, s[52:53] // (op 1188) most significant half



s_mul_i32 s53, s28, s3 // (op 1190) most significant: low of low * high
s_mul_i32 s54, s29, s2 // (op 1190) most significant: low of high * low
s_mul_i32 s52, s28, s2 // (op 1190) least significant
s_mul_hi_u32 s55, s28, s2 // (op 1190) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1190) most significant: sum
s_add_u32 s53, s53, s55 // (op 1190) most significant: sum


s_mov_b32 s64, s30 // (op 1190) reg expression
s_mov_b32 s65, s31 // (op 1190) reg expression
s_mov_b32 s66, s52 // (op 1190) reg expression
s_mov_b32 s67, 131072 // (op 1190) call()

v_and_b32 v13, -2, v1 // (op 1207) 
s_mul_i32 s52, s16, s8 // (op 1207) 
s_add_u32 s52, s52, s0 // (op 1207) 
v_sub_u32 v18, v1, v13 // (op 1207) 

s_mul_hi_u32 s53, s52, s13 // (op 1207) 
v_lshlrev_b32 v13, 4, v18 // (op 1207) 

s_sub_u32 s54, s52, s53 // (op 1207) 
v_lshrrev_b32 v18, 1, v1 // (op 1207) 
s_lshr_b32 s55, s54, 1 // (op 1207) 

s_add_u32 s54, s55, s53 // (op 1207) 


s_lshr_b32 s53, s54, s11 // (op 1207) 

s_and_b32 s54, s11, 2147483648 // (op 1207) 
s_cmp_eq_i32 s54, 0 // (op 1207) 
s_mov_b32 s55, scc // (op 1207) 

s_cmp_eq_i32 s55, 0 // (op 1207) 
s_cselect_b32 s54, s52, s53 // (op 1207) 


s_mul_hi_u32 s53, s54, s12 // (op 1207) 
s_sub_u32 s55, s54, s53 // (op 1207) 
s_lshr_b32 s68, s55, 1 // (op 1207) 

s_add_u32 s55, s68, s53 // (op 1207) 


s_lshr_b32 s53, s55, s10 // (op 1207) 

s_and_b32 s55, s10, 2147483648 // (op 1207) 
s_cmp_eq_i32 s55, 0 // (op 1207) 
s_mov_b32 s68, scc // (op 1207) 

s_cmp_eq_i32 s68, 0 // (op 1207) 
s_cselect_b32 s55, s54, s53 // (op 1207) 


s_mul_i32 s53, s55, s14 // (op 1207) 

s_sub_u32 s55, s54, s53 // (op 1207) 

v_lshl_add_u32 v19, s55, 5, v13 // (op 1207) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 1207) 
v_mov_b32 v25, 0 // (op 1207) convert
v_mov_b32 v24, v19 // (op 1207) convert

s_mul_i32 s53, s54, s68 // (op 1207) 


s_sub_u32 s54, s52, s53 // (op 1207) 


v_lshl_add_u32 v13, s54, 7, v18 // (op 1207) 


v_mul_lo_u32 v19, v13, s29 // (op 1207) most significant: low of low * high
v_mul_hi_u32 v26, v13, s28 // (op 1207) most significant: high of low * low
v_mul_lo_u32 v18, v13, s28 // (op 1207) least significant: low of low * low
v_add_u32 v19, v19, v26 // (op 1207) most significant: sum

v_add_co_u32 v18, s[52:53], v18, v24 // (op 1207) least significant half
v_addc_co_u32 v19, s[52:53], v19, v25, s[52:53] // (op 1207) most significant half



s_mul_i32 s53, s28, s3 // (op 1209) most significant: low of low * high
s_mul_i32 s54, s29, s2 // (op 1209) most significant: low of high * low
s_mul_i32 s52, s28, s2 // (op 1209) least significant
s_mul_hi_u32 s55, s28, s2 // (op 1209) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1209) most significant: sum
s_add_u32 s53, s53, s55 // (op 1209) most significant: sum


s_mov_b32 s64, s30 // (op 1209) reg expression
s_mov_b32 s65, s31 // (op 1209) reg expression
s_mov_b32 s66, s52 // (op 1209) reg expression
s_mov_b32 s67, 131072 // (op 1209) call()

v_lshrrev_b32 v13, 7, v1 // (op 1316) 
s_mul_i32 s52, s16, s8 // (op 1316) 
s_add_u32 s52, s52, s0 // (op 1316) 
v_and_b32 v24, 15, v1 // (op 1316) 
s_mul_hi_u32 s53, s52, s13 // (op 1316) 
v_lshl_add_u32 v25, v13, 4, v24 // (op 1316) 


s_sub_u32 s54, s52, s53 // (op 1316) 
v_and_b32 v13, 63, v1 // (op 1316) 
s_lshr_b32 s55, s54, 1 // (op 1316) 

v_lshrrev_b32 v24, 4, v13 // (op 1316) 

s_add_u32 s54, s55, s53 // (op 1316) 


s_lshr_b32 s53, s54, s11 // (op 1316) 

s_and_b32 s54, s11, 2147483648 // (op 1316) 
s_cmp_eq_i32 s54, 0 // (op 1316) 
s_mov_b32 s55, scc // (op 1316) 

s_cmp_eq_i32 s55, 0 // (op 1316) 
s_cselect_b32 s54, s52, s53 // (op 1316) 


s_mul_hi_u32 s53, s54, s12 // (op 1316) 
s_sub_u32 s55, s54, s53 // (op 1316) 
s_lshr_b32 s68, s55, 1 // (op 1316) 

s_add_u32 s55, s68, s53 // (op 1316) 


s_lshr_b32 s53, s55, s10 // (op 1316) 

s_and_b32 s55, s10, 2147483648 // (op 1316) 
s_cmp_eq_i32 s55, 0 // (op 1316) 
s_mov_b32 s68, scc // (op 1316) 

s_cmp_eq_i32 s68, 0 // (op 1316) 
s_cselect_b32 s55, s54, s53 // (op 1316) 


v_lshl_add_u32 v13, s55, 5, v25 // (op 1316) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 1316) 
s_mul_i32 s53, s54, s68 // (op 1316) 


s_sub_u32 s54, s52, s53 // (op 1316) 


v_lshl_add_u32 v25, s54, 2, v24 // (op 1316) 


v_mul_lo_u32 v24, v25, s32 // (op 1316) 

v_add_u32 v25, v13, v24 // (op 1316) 


s_mul_i32 s53, s32, s37 // (op 1318) most significant: low of low * high
s_mul_i32 s54, s33, s36 // (op 1318) most significant: low of high * low
s_mul_i32 s52, s32, s36 // (op 1318) least significant
s_mul_hi_u32 s55, s32, s36 // (op 1318) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1318) most significant: sum
s_add_u32 s53, s53, s55 // (op 1318) most significant: sum


s_mov_b32 s68, s34 // (op 1318) reg expression
s_mov_b32 s69, s35 // (op 1318) reg expression
s_mov_b32 s70, s52 // (op 1318) reg expression
s_mov_b32 s71, 131072 // (op 1318) call()

v_lshrrev_b32 v13, 7, v1 // (op 1335) 
s_mul_i32 s52, s16, s8 // (op 1335) 
s_add_u32 s52, s52, s0 // (op 1335) 
v_and_b32 v24, 15, v1 // (op 1335) 
s_mul_hi_u32 s53, s52, s13 // (op 1335) 
v_lshl_add_u32 v26, v13, 4, v24 // (op 1335) 


s_sub_u32 s54, s52, s53 // (op 1335) 
v_and_b32 v13, 63, v1 // (op 1335) 
s_lshr_b32 s55, s54, 1 // (op 1335) 

v_lshrrev_b32 v24, 4, v13 // (op 1335) 

s_add_u32 s54, s55, s53 // (op 1335) 


s_lshr_b32 s53, s54, s11 // (op 1335) 

s_and_b32 s54, s11, 2147483648 // (op 1335) 
s_cmp_eq_i32 s54, 0 // (op 1335) 
s_mov_b32 s55, scc // (op 1335) 

s_cmp_eq_i32 s55, 0 // (op 1335) 
s_cselect_b32 s54, s52, s53 // (op 1335) 


s_mul_hi_u32 s53, s54, s12 // (op 1335) 
s_sub_u32 s55, s54, s53 // (op 1335) 
s_lshr_b32 s72, s55, 1 // (op 1335) 

s_add_u32 s55, s72, s53 // (op 1335) 


s_lshr_b32 s53, s55, s10 // (op 1335) 

s_and_b32 s55, s10, 2147483648 // (op 1335) 
s_cmp_eq_i32 s55, 0 // (op 1335) 
s_mov_b32 s72, scc // (op 1335) 

s_cmp_eq_i32 s72, 0 // (op 1335) 
s_cselect_b32 s55, s54, s53 // (op 1335) 


v_lshl_add_u32 v13, s55, 5, v26 // (op 1335) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 1335) 
s_mul_i32 s53, s54, s72 // (op 1335) 


s_sub_u32 s54, s52, s53 // (op 1335) 


v_lshl_add_u32 v26, s54, 2, v24 // (op 1335) 


v_mul_lo_u32 v24, v26, s32 // (op 1335) 

v_add_u32 v26, v13, v24 // (op 1335) 


s_mul_i32 s53, s32, s37 // (op 1337) most significant: low of low * high
s_mul_i32 s54, s33, s36 // (op 1337) most significant: low of high * low
s_mul_i32 s52, s32, s36 // (op 1337) least significant
s_mul_hi_u32 s55, s32, s36 // (op 1337) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1337) most significant: sum
s_add_u32 s53, s53, s55 // (op 1337) most significant: sum


s_mov_b32 s68, s34 // (op 1337) reg expression
s_mov_b32 s69, s35 // (op 1337) reg expression
s_mov_b32 s70, s52 // (op 1337) reg expression
s_mov_b32 s71, 131072 // (op 1337) call()

v_lshrrev_b32 v13, 6, v1 // (op 1444) 
s_mul_i32 s52, s16, s8 // (op 1444) 
s_add_u32 s52, s52, s0 // (op 1444) 
v_and_b32 v24, 1, v13 // (op 1444) 

s_mul_hi_u32 s53, s52, s13 // (op 1444) 
v_and_b32 v13, 15, v1 // (op 1444) 
s_sub_u32 s54, s52, s53 // (op 1444) 
v_lshl_add_u32 v27, v24, 4, v13 // (op 1444) 


s_lshr_b32 s55, s54, 1 // (op 1444) 

v_and_b32 v13, 63, v1 // (op 1444) 
s_add_u32 s54, s55, s53 // (op 1444) 


v_lshrrev_b32 v24, 4, v13 // (op 1444) 

s_lshr_b32 s53, s54, s11 // (op 1444) 

s_and_b32 s54, s11, 2147483648 // (op 1444) 
s_cmp_eq_i32 s54, 0 // (op 1444) 
s_mov_b32 s55, scc // (op 1444) 

s_cmp_eq_i32 s55, 0 // (op 1444) 
s_cselect_b32 s54, s52, s53 // (op 1444) 


s_mul_hi_u32 s53, s54, s12 // (op 1444) 
s_sub_u32 s55, s54, s53 // (op 1444) 
s_lshr_b32 s72, s55, 1 // (op 1444) 

s_add_u32 s55, s72, s53 // (op 1444) 


s_lshr_b32 s53, s55, s10 // (op 1444) 

s_and_b32 s55, s10, 2147483648 // (op 1444) 
s_cmp_eq_i32 s55, 0 // (op 1444) 
s_mov_b32 s72, scc // (op 1444) 

s_cmp_eq_i32 s72, 0 // (op 1444) 
s_cselect_b32 s55, s54, s53 // (op 1444) 


s_mul_i32 s53, s55, s14 // (op 1444) 

s_sub_u32 s55, s54, s53 // (op 1444) 

v_lshl_add_u32 v13, s55, 5, v27 // (op 1444) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 1444) 
s_mul_i32 s53, s54, s72 // (op 1444) 


s_sub_u32 s54, s52, s53 // (op 1444) 


v_lshl_add_u32 v27, s54, 2, v24 // (op 1444) 


v_mul_lo_u32 v24, v27, s38 // (op 1444) 
v_add_u32 v24, v24, v13 // (op 1444) 


s_mul_i32 s53, s38, s37 // (op 1446) most significant: low of low * high
s_mul_i32 s54, s39, s36 // (op 1446) most significant: low of high * low
s_mul_i32 s52, s38, s36 // (op 1446) least significant
s_mul_hi_u32 s55, s38, s36 // (op 1446) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1446) most significant: sum
s_add_u32 s53, s53, s55 // (op 1446) most significant: sum


s_mov_b32 s72, s40 // (op 1446) reg expression
s_mov_b32 s73, s41 // (op 1446) reg expression
s_mov_b32 s74, s52 // (op 1446) reg expression
s_mov_b32 s75, 131072 // (op 1446) call()

v_lshrrev_b32 v13, 6, v1 // (op 1463) 
s_mul_i32 s52, s16, s8 // (op 1463) 
s_add_u32 s52, s52, s0 // (op 1463) 
v_and_b32 v27, 1, v13 // (op 1463) 

s_mul_hi_u32 s53, s52, s13 // (op 1463) 
v_and_b32 v13, 15, v1 // (op 1463) 
s_sub_u32 s54, s52, s53 // (op 1463) 
v_lshl_add_u32 v28, v27, 4, v13 // (op 1463) 


s_lshr_b32 s55, s54, 1 // (op 1463) 

v_and_b32 v13, 63, v1 // (op 1463) 
s_add_u32 s54, s55, s53 // (op 1463) 


v_lshrrev_b32 v27, 4, v13 // (op 1463) 

s_lshr_b32 s53, s54, s11 // (op 1463) 

s_and_b32 s54, s11, 2147483648 // (op 1463) 
s_cmp_eq_i32 s54, 0 // (op 1463) 
s_mov_b32 s55, scc // (op 1463) 

s_cmp_eq_i32 s55, 0 // (op 1463) 
s_cselect_b32 s54, s52, s53 // (op 1463) 


s_mul_hi_u32 s53, s54, s12 // (op 1463) 
s_sub_u32 s55, s54, s53 // (op 1463) 
s_lshr_b32 s76, s55, 1 // (op 1463) 

s_add_u32 s55, s76, s53 // (op 1463) 


s_lshr_b32 s53, s55, s10 // (op 1463) 

s_and_b32 s55, s10, 2147483648 // (op 1463) 
s_cmp_eq_i32 s55, 0 // (op 1463) 
s_mov_b32 s76, scc // (op 1463) 

s_cmp_eq_i32 s76, 0 // (op 1463) 
s_cselect_b32 s55, s54, s53 // (op 1463) 


s_mul_i32 s53, s55, s14 // (op 1463) 

s_sub_u32 s55, s54, s53 // (op 1463) 

v_lshl_add_u32 v13, s55, 5, v28 // (op 1463) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1463) 
s_mul_i32 s53, s54, s76 // (op 1463) 


s_sub_u32 s54, s52, s53 // (op 1463) 


v_lshl_add_u32 v28, s54, 2, v27 // (op 1463) 


v_mul_lo_u32 v27, v28, s38 // (op 1463) 
v_add_u32 v27, v27, v13 // (op 1463) 


s_mul_i32 s53, s38, s37 // (op 1465) most significant: low of low * high
s_mul_i32 s54, s39, s36 // (op 1465) most significant: low of high * low
s_mul_i32 s52, s38, s36 // (op 1465) least significant
s_mul_hi_u32 s55, s38, s36 // (op 1465) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1465) most significant: sum
s_add_u32 s53, s53, s55 // (op 1465) most significant: sum


s_mov_b32 s72, s40 // (op 1465) reg expression
s_mov_b32 s73, s41 // (op 1465) reg expression
s_mov_b32 s74, s52 // (op 1465) reg expression
s_mov_b32 s75, 131072 // (op 1465) call()


v_mov_b64 v[28:29], s[32:33] // (op 739) convert

v_mov_b32 v13, v25 // (op 739) reg expression
v_mov_b32 v30, v13 // (op 739) 
buffer_load_ubyte v31, v30, s[68:71], 0 offen offset: 0 // (op 739) Load value




v_mov_b64 v[28:29], s[38:39] // (op 744) convert

v_mov_b32 v13, v24 // (op 744) reg expression
v_mov_b32 v30, v13 // (op 744) 
buffer_load_ubyte v32, v30, s[72:75], 0 offen offset: 0 // (op 744) Load value



v_and_b32 v13, -2, v1 // (op 1652) 
v_sub_u32 v28, v1, v13 // (op 1652) 

v_lshlrev_b32 v13, 4, v28 // (op 1652) 

v_lshrrev_b32 v28, 1, v1 // (op 1652) 
v_lshl_add_u32 v29, v28, 5, v13 // (op 1652) 


s_nop 0
v_readfirstlane_b32 s52, v29 // (op 1652) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2151) 


v_mov_b64 v[28:29], v[16:17] // (op 968) reg expression

s_mov_b32 m0, s52 // (op 968) reg expression
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 968) Load value direct to lds


v_and_b32 v13, -2, v1 // (op 1587) 
v_sub_u32 v28, v1, v13 // (op 1587) 

v_lshlrev_b32 v13, 4, v28 // (op 1587) 

v_lshrrev_b32 v28, 1, v1 // (op 1587) 
v_lshl_add_u32 v29, v28, 5, v13 // (op 1587) 


s_nop 0
v_readfirstlane_b32 s52, v29 // (op 1587) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2132) 


v_mov_b64 v[28:29], v[20:21] // (op 973) reg expression
s_add_u32 s53, 4096, s52 // (op 973) 

s_mov_b32 m0, s53 // (op 973) reg expression
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 973) Load value direct to lds




v_mov_b64 v[28:29], s[32:33] // (op 755) convert

s_lshl_b32 s52, s32, 2 // (op 755) 
v_add_u32 v13, v25, s52 // (op 755) 

v_mov_b32 v30, v13 // (op 755) 
buffer_load_ubyte v33, v30, s[68:71], 0 offen offset: 0 // (op 755) Load value





v_mov_b64 v[28:29], s[38:39] // (op 760) convert

s_lshl_b32 s52, s38, 2 // (op 760) 
v_add_u32 v13, v24, s52 // (op 760) 

v_mov_b32 v25, v13 // (op 760) 
buffer_load_ubyte v30, v25, s[72:75], 0 offen offset: 0 // (op 760) Load value




v_and_b32 v13, -2, v1 // (op 2055) 
v_sub_u32 v24, v1, v13 // (op 2055) 

v_lshlrev_b32 v13, 4, v24 // (op 2055) 

v_lshrrev_b32 v24, 1, v1 // (op 2055) 
v_lshl_add_u32 v25, v24, 5, v13 // (op 2055) 


s_nop 0
v_readfirstlane_b32 s52, v25 // (op 2055) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2235) 


s_lshl_b64 s[54:55], s[28:29], 7 // (op 978) 
v_mov_b32 v13, s54 // (op 978) 
v_mov_b32 v24, s55 // (op 978) 
v_add_co_u32 v28, s[76:77], v16, v13 // (op 978) least significant half
v_addc_co_u32 v29, s[76:77], v17, v24, s[76:77] // (op 978) most significant half




s_add_u32 s53, 8192, s52 // (op 978) 

s_mov_b32 m0, s53 // (op 978) reg expression
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 978) Load value direct to lds




v_and_b32 v13, -2, v1 // (op 2016) 
v_sub_u32 v16, v1, v13 // (op 2016) 

v_lshlrev_b32 v13, 4, v16 // (op 2016) 

v_lshrrev_b32 v16, 1, v1 // (op 2016) 
v_lshl_add_u32 v17, v16, 5, v13 // (op 2016) 


s_nop 0
v_readfirstlane_b32 s52, v17 // (op 2016) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2226) 


s_lshl_b64 s[54:55], s[24:25], 7 // (op 980) 
v_mov_b32 v13, s54 // (op 980) 
v_mov_b32 v16, s55 // (op 980) 
v_add_co_u32 v24, s[76:77], v20, v13 // (op 980) least significant half
v_addc_co_u32 v25, s[76:77], v21, v16, s[76:77] // (op 980) most significant half




s_add_u32 s53, 12288, s52 // (op 980) 

s_mov_b32 m0, s53 // (op 980) reg expression
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 980) Load value direct to lds




s_waitcnt vmcnt(5)
s_barrier  // (op 2148) 



ds_read_b64_tr_b8 v[34:35], v15 // (op 785) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v15 offset:256 // (op 785) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v15 offset:2048 // (op 785) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v15 offset:2304 // (op 785) Transpose load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 2129) 



ds_read_b64_tr_b8 v[42:43], v12 offset:4096 // (op 782) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v12 offset:4352 // (op 782) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v12 offset:6144 // (op 782) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v12 offset:6400 // (op 782) Transpose load local data 
v_and_b32 v13, -2, v1 // (op 1559) 
v_sub_u32 v16, v1, v13 // (op 1559) 

v_lshlrev_b32 v13, 4, v16 // (op 1559) 

v_lshrrev_b32 v16, 1, v1 // (op 1559) 
v_lshl_add_u32 v17, v16, 5, v13 // (op 1559) 


s_nop 0
v_readfirstlane_b32 s52, v17 // (op 1559) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -2, v1 // (op 1624) 
v_sub_u32 v16, v1, v13 // (op 1624) 

v_lshlrev_b32 v13, 4, v16 // (op 1624) 

v_lshrrev_b32 v16, 1, v1 // (op 1624) 
v_lshl_add_u32 v17, v16, 5, v13 // (op 1624) 


s_nop 0
v_readfirstlane_b32 s53, v17 // (op 1624) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -2, v1 // (op 2001) 
v_sub_u32 v16, v1, v13 // (op 2001) 

v_lshlrev_b32 v13, 4, v16 // (op 2001) 

v_lshrrev_b32 v16, 1, v1 // (op 2001) 
v_lshl_add_u32 v17, v16, 5, v13 // (op 2001) 


s_nop 0
v_readfirstlane_b32 s54, v17 // (op 2001) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -2, v1 // (op 2040) 
v_sub_u32 v16, v1, v13 // (op 2040) 

v_lshlrev_b32 v13, 4, v16 // (op 2040) 

v_lshrrev_b32 v16, 1, v1 // (op 2040) 
v_lshl_add_u32 v17, v16, 5, v13 // (op 2040) 


s_nop 0
v_readfirstlane_b32 s55, v17 // (op 2040) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s1, 0 // (op 51) call()
s_and_b32 s76, s7, 4294967294 // (op 50) 
s_cmp_lt_u32 s1, s76 // (op 50) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_50 // (op 50) Condition: Top (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_50 if false)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_50:
s_add_u32 s1, s1, 2 // (op 52) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[34:41], a[0:3], v31, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 76) 

v_mov_b64 v[16:17], s[32:33] // (op 70) convert

v_mov_b32 v13, v26 // (op 70) 
s_lshl_b32 s76, s32, 3 // (op 70) 
v_add_u32 v20, v13, s76 // (op 70) 

v_mov_b32 v21, v20 // (op 70) 
buffer_load_ubyte v31, v21, s[68:71], 0 offen offset: 0 // (op 70) Load value




v_mov_b64 v[16:17], s[38:39] // (op 74) convert

v_mov_b32 v20, v27 // (op 74) 
s_lshl_b32 s76, s38, 3 // (op 74) 
v_add_u32 v21, v20, s76 // (op 74) 

v_mov_b32 v24, v21 // (op 74) 
buffer_load_ubyte v32, v24, s[72:75], 0 offen offset: 0 // (op 74) Load value



s_barrier  // (op 2154) 


v_mov_b64 v[16:17], v[18:19] // (op 909) 
s_lshl_b64 s[76:77], s[28:29], 7 // (op 909) 
s_lshl_b64 s[78:79], s[76:77], 1 // (op 909) 

v_mov_b32 v21, s78 // (op 909) 
v_mov_b32 v24, s79 // (op 909) 
v_add_co_u32 v28, s[76:77], v16, v21 // (op 909) least significant half
v_addc_co_u32 v29, s[76:77], v17, v24, s[76:77] // (op 909) most significant half





s_mov_b32 m0, s53 // (op 909) reg expression
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 909) Load value direct to lds

s_barrier  // (op 2135) 


v_mov_b64 v[24:25], v[22:23] // (op 904) 
s_lshl_b64 s[76:77], s[24:25], 7 // (op 904) 
s_lshl_b64 s[78:79], s[76:77], 1 // (op 904) 

v_mov_b32 v21, s78 // (op 904) 
v_mov_b32 v28, s79 // (op 904) 
v_add_co_u32 v50, s[76:77], v24, v21 // (op 904) least significant half
v_addc_co_u32 v51, s[76:77], v25, v28, s[76:77] // (op 904) most significant half




s_add_u32 s76, 4096, s52 // (op 904) 

s_nop 0
s_mov_b32 m0, s76 // (op 904) reg expression
buffer_load_dwordx4 v50, s[60:63], 0 offen offset: 0 lds // (op 904) Load value direct to lds


s_waitcnt vmcnt(5)
s_barrier  // (op 2232) 



ds_read_b64_tr_b8 v[34:35], v10 offset:8192 // (op 603) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v10 offset:8448 // (op 603) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v10 offset:10240 // (op 603) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v10 offset:10496 // (op 603) Transpose load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 2223) 



ds_read_b64_tr_b8 v[42:43], v7 offset:12288 // (op 601) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:12544 // (op 601) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:14336 // (op 601) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v7 offset:14592 // (op 601) Transpose load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[34:41], a[0:3], v33, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 608) 

v_mov_b64 v[28:29], s[32:33] // (op 605) convert

s_lshl_b32 s76, s32, 2 // (op 605) 
s_mul_i32 s77, 3, s76 // (op 605) 

v_add_u32 v21, v13, s77 // (op 605) 

v_mov_b32 v50, v21 // (op 605) 
buffer_load_ubyte v33, v50, s[68:71], 0 offen offset: 0 // (op 605) Load value





s_lshl_b32 s76, s32, 3 // (op 1327) 
v_add_u32 v26, v26, s76 // (op 1327) 

v_mov_b64 v[28:29], s[38:39] // (op 607) convert

s_lshl_b32 s76, s38, 2 // (op 607) 
s_mul_i32 s77, 3, s76 // (op 607) 

v_add_u32 v13, v20, s77 // (op 607) 

v_mov_b32 v21, v13 // (op 607) 
buffer_load_ubyte v30, v21, s[72:75], 0 offen offset: 0 // (op 607) Load value




s_barrier  // (op 2238) 
s_lshl_b32 s76, s38, 3 // (op 1455) 


v_add_u32 v27, v27, s76 // (op 1455) 

s_lshl_b64 s[76:77], s[28:29], 7 // (op 963) 
s_mul_i32 s79, 3, s77 // (op 963) most significant: low of low * high
s_mul_i32 s78, 3, s76 // (op 963) least significant
s_mul_hi_u32 s80, 3, s76 // (op 963) most significant: high of low * low
s_add_u32 s79, s79, s80 // (op 963) most significant: sum


v_mov_b32 v13, s78 // (op 963) 
v_mov_b32 v20, s79 // (op 963) 
v_add_co_u32 v28, s[76:77], v16, v13 // (op 963) least significant half
v_addc_co_u32 v29, s[76:77], v17, v20, s[76:77] // (op 963) most significant half




s_add_u32 s76, 8192, s55 // (op 963) 

s_nop 0
s_mov_b32 m0, s76 // (op 963) reg expression
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 963) Load value direct to lds



s_barrier  // (op 2229) 
s_lshl_b64 s[76:77], s[28:29], 7 // (op 1199) 


s_lshl_b64 s[78:79], s[76:77], 1 // (op 1199) 

v_mov_b32 v13, s78 // (op 1199) 
v_mov_b32 v16, s79 // (op 1199) 
s_lshl_b64 s[76:77], s[24:25], 7 // (op 958) 
s_mul_i32 s81, 3, s77 // (op 958) most significant: low of low * high
s_mul_i32 s80, 3, s76 // (op 958) least significant
s_mul_hi_u32 s82, 3, s76 // (op 958) most significant: high of low * low
s_add_u32 s81, s81, s82 // (op 958) most significant: sum


v_mov_b32 v17, s80 // (op 958) 
v_mov_b32 v20, s81 // (op 958) 
v_add_co_u32 v18, s[76:77], v18, v13 // (op 1199) least significant half
v_addc_co_u32 v19, s[76:77], v19, v16, s[76:77] // (op 1199) most significant half




v_add_co_u32 v28, s[76:77], v24, v17 // (op 958) least significant half
v_addc_co_u32 v29, s[76:77], v25, v20, s[76:77] // (op 958) most significant half




s_add_u32 s76, 12288, s54 // (op 958) 

s_nop 0
s_mov_b32 m0, s76 // (op 958) reg expression
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 958) Load value direct to lds



s_lshl_b64 s[76:77], s[24:25], 7 // (op 1068) 
s_lshl_b64 s[78:79], s[76:77], 1 // (op 1068) 

v_mov_b32 v13, s78 // (op 1068) 
v_mov_b32 v16, s79 // (op 1068) 
v_add_co_u32 v22, s[76:77], v22, v13 // (op 1068) least significant half
v_addc_co_u32 v23, s[76:77], v23, v16, s[76:77] // (op 1068) most significant half




s_waitcnt vmcnt(5)
s_barrier  // (op 2160) 



ds_read_b64_tr_b8 v[34:35], v4 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:256 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v4 offset:2048 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:2304 // (op 65) Transpose load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 2141) 



ds_read_b64_tr_b8 v[42:43], v0 offset:4096 // (op 60) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:4352 // (op 60) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:6144 // (op 60) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:6400 // (op 60) Transpose load local data 
s_and_b32 s76, s7, 4294967294 // (op 50) 
s_cmp_lt_u32 s1, s76 // (op 50) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_50 // (op 50) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_50 if true)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_50:










s_barrier  // (op 2245) 
s_and_b32 s52, s7, 4294967294 // (op 1031) 
v_and_b32 v13, -2, v1 // (op 1031) 
s_add_u32 s53, s0, s52 // (op 1031) 

v_sub_u32 v16, v1, v13 // (op 1031) 

v_lshlrev_b32 v13, 4, v16 // (op 1031) 

s_mul_i32 s52, s16, s8 // (op 1031) 
s_add_u32 s52, s52, s53 // (op 1031) 

v_lshrrev_b32 v16, 1, v1 // (op 1031) 
s_mul_hi_u32 s53, s52, s13 // (op 1031) 
s_sub_u32 s54, s52, s53 // (op 1031) 
s_lshr_b32 s55, s54, 1 // (op 1031) 

s_add_u32 s54, s55, s53 // (op 1031) 


s_lshr_b32 s53, s54, s11 // (op 1031) 

s_and_b32 s54, s11, 2147483648 // (op 1031) 
s_cmp_eq_i32 s54, 0 // (op 1031) 
s_mov_b32 s55, scc // (op 1031) 

s_cmp_eq_i32 s55, 0 // (op 1031) 
s_cselect_b32 s54, s52, s53 // (op 1031) 


s_mul_hi_u32 s53, s54, s12 // (op 1031) 
s_sub_u32 s55, s54, s53 // (op 1031) 
s_lshr_b32 s76, s55, 1 // (op 1031) 

s_add_u32 s55, s76, s53 // (op 1031) 


s_lshr_b32 s53, s55, s10 // (op 1031) 

s_and_b32 s55, s10, 2147483648 // (op 1031) 
s_cmp_eq_i32 s55, 0 // (op 1031) 
s_mov_b32 s76, scc // (op 1031) 

s_cmp_eq_i32 s76, 0 // (op 1031) 
s_cselect_b32 s55, s54, s53 // (op 1031) 


v_lshl_add_u32 v17, s55, 5, v13 // (op 1031) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1031) 
v_mov_b32 v19, 0 // (op 1031) convert
v_mov_b32 v18, v17 // (op 1031) convert

s_mul_i32 s53, s54, s76 // (op 1031) 


s_sub_u32 s54, s52, s53 // (op 1031) 


v_lshl_add_u32 v13, s54, 7, v16 // (op 1031) 


v_mul_lo_u32 v17, v13, s25 // (op 1031) most significant: low of low * high
v_mul_hi_u32 v20, v13, s24 // (op 1031) most significant: high of low * low
v_mul_lo_u32 v16, v13, s24 // (op 1031) least significant: low of low * low
v_add_u32 v17, v17, v20 // (op 1031) most significant: sum


v_add_co_u32 v20, s[52:53], v18, v16 // (op 1031) least significant half
v_addc_co_u32 v21, s[52:53], v19, v17, s[52:53] // (op 1031) most significant half



s_mul_i32 s53, s24, s3 // (op 1033) most significant: low of low * high
s_mul_i32 s54, s25, s2 // (op 1033) most significant: low of high * low
s_mul_i32 s52, s24, s2 // (op 1033) least significant
s_mul_hi_u32 s55, s24, s2 // (op 1033) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1033) most significant: sum
s_add_u32 s53, s53, s55 // (op 1033) most significant: sum


s_mov_b32 s60, s26 // (op 1033) reg expression
s_mov_b32 s61, s27 // (op 1033) reg expression
s_mov_b32 s62, s52 // (op 1033) reg expression
s_mov_b32 s63, 131072 // (op 1033) call()

s_and_b32 s52, s7, 4294967294 // (op 1165) 
v_and_b32 v13, -2, v1 // (op 1165) 
s_add_u32 s53, s0, s52 // (op 1165) 

v_sub_u32 v16, v1, v13 // (op 1165) 

v_lshlrev_b32 v13, 4, v16 // (op 1165) 

s_mul_i32 s52, s16, s8 // (op 1165) 
s_add_u32 s52, s52, s53 // (op 1165) 

v_lshrrev_b32 v16, 1, v1 // (op 1165) 
s_mul_hi_u32 s53, s52, s13 // (op 1165) 
s_sub_u32 s54, s52, s53 // (op 1165) 
s_lshr_b32 s55, s54, 1 // (op 1165) 

s_add_u32 s54, s55, s53 // (op 1165) 


s_lshr_b32 s53, s54, s11 // (op 1165) 

s_and_b32 s54, s11, 2147483648 // (op 1165) 
s_cmp_eq_i32 s54, 0 // (op 1165) 
s_mov_b32 s55, scc // (op 1165) 

s_cmp_eq_i32 s55, 0 // (op 1165) 
s_cselect_b32 s54, s52, s53 // (op 1165) 


s_mul_hi_u32 s53, s54, s12 // (op 1165) 
s_sub_u32 s55, s54, s53 // (op 1165) 
s_lshr_b32 s76, s55, 1 // (op 1165) 

s_add_u32 s55, s76, s53 // (op 1165) 


s_lshr_b32 s53, s55, s10 // (op 1165) 

s_and_b32 s55, s10, 2147483648 // (op 1165) 
s_cmp_eq_i32 s55, 0 // (op 1165) 
s_mov_b32 s76, scc // (op 1165) 

s_cmp_eq_i32 s76, 0 // (op 1165) 
s_cselect_b32 s55, s54, s53 // (op 1165) 


s_mul_i32 s53, s55, s14 // (op 1165) 

s_sub_u32 s55, s54, s53 // (op 1165) 

v_lshl_add_u32 v17, s55, 5, v13 // (op 1165) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1165) 
v_mov_b32 v19, 0 // (op 1165) convert
v_mov_b32 v18, v17 // (op 1165) convert

s_mul_i32 s53, s54, s76 // (op 1165) 


s_sub_u32 s54, s52, s53 // (op 1165) 


v_lshl_add_u32 v13, s54, 7, v16 // (op 1165) 


v_mul_lo_u32 v17, v13, s29 // (op 1165) most significant: low of low * high
v_mul_hi_u32 v22, v13, s28 // (op 1165) most significant: high of low * low
v_mul_lo_u32 v16, v13, s28 // (op 1165) least significant: low of low * low
v_add_u32 v17, v17, v22 // (op 1165) most significant: sum

v_add_co_u32 v16, s[52:53], v16, v18 // (op 1165) least significant half
v_addc_co_u32 v17, s[52:53], v17, v19, s[52:53] // (op 1165) most significant half



s_mul_i32 s53, s28, s3 // (op 1167) most significant: low of low * high
s_mul_i32 s54, s29, s2 // (op 1167) most significant: low of high * low
s_mul_i32 s52, s28, s2 // (op 1167) least significant
s_mul_hi_u32 s55, s28, s2 // (op 1167) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1167) most significant: sum
s_add_u32 s53, s53, s55 // (op 1167) most significant: sum


s_mov_b32 s64, s30 // (op 1167) reg expression
s_mov_b32 s65, s31 // (op 1167) reg expression
s_mov_b32 s66, s52 // (op 1167) reg expression
s_mov_b32 s67, 131072 // (op 1167) call()

s_and_b32 s52, s7, 4294967294 // (op 1293) 
v_lshrrev_b32 v13, 7, v1 // (op 1293) 
s_add_u32 s53, s0, s52 // (op 1293) 

v_and_b32 v18, 15, v1 // (op 1293) 
v_lshl_add_u32 v19, v13, 4, v18 // (op 1293) 


s_mul_i32 s52, s16, s8 // (op 1293) 
s_add_u32 s52, s52, s53 // (op 1293) 

v_and_b32 v13, 63, v1 // (op 1293) 
s_mul_hi_u32 s53, s52, s13 // (op 1293) 
v_lshrrev_b32 v18, 4, v13 // (op 1293) 

s_sub_u32 s54, s52, s53 // (op 1293) 
s_lshr_b32 s55, s54, 1 // (op 1293) 

s_add_u32 s54, s55, s53 // (op 1293) 


s_lshr_b32 s53, s54, s11 // (op 1293) 

s_and_b32 s54, s11, 2147483648 // (op 1293) 
s_cmp_eq_i32 s54, 0 // (op 1293) 
s_mov_b32 s55, scc // (op 1293) 

s_cmp_eq_i32 s55, 0 // (op 1293) 
s_cselect_b32 s54, s52, s53 // (op 1293) 


s_mul_hi_u32 s53, s54, s12 // (op 1293) 
s_sub_u32 s55, s54, s53 // (op 1293) 
s_lshr_b32 s76, s55, 1 // (op 1293) 

s_add_u32 s55, s76, s53 // (op 1293) 


s_lshr_b32 s53, s55, s10 // (op 1293) 

s_and_b32 s55, s10, 2147483648 // (op 1293) 
s_cmp_eq_i32 s55, 0 // (op 1293) 
s_mov_b32 s76, scc // (op 1293) 

s_cmp_eq_i32 s76, 0 // (op 1293) 
s_cselect_b32 s55, s54, s53 // (op 1293) 


v_lshl_add_u32 v13, s55, 5, v19 // (op 1293) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1293) 
s_mul_i32 s53, s54, s76 // (op 1293) 


s_sub_u32 s54, s52, s53 // (op 1293) 


v_lshl_add_u32 v19, s54, 2, v18 // (op 1293) 


v_mul_lo_u32 v18, v19, s32 // (op 1293) 

v_add_u32 v19, v13, v18 // (op 1293) 


s_mul_i32 s53, s32, s37 // (op 1295) most significant: low of low * high
s_mul_i32 s54, s33, s36 // (op 1295) most significant: low of high * low
s_mul_i32 s52, s32, s36 // (op 1295) least significant
s_mul_hi_u32 s55, s32, s36 // (op 1295) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1295) most significant: sum
s_add_u32 s53, s53, s55 // (op 1295) most significant: sum


s_mov_b32 s68, s34 // (op 1295) reg expression
s_mov_b32 s69, s35 // (op 1295) reg expression
s_mov_b32 s70, s52 // (op 1295) reg expression
s_mov_b32 s71, 131072 // (op 1295) call()

s_and_b32 s52, s7, 4294967294 // (op 1421) 
v_lshrrev_b32 v13, 6, v1 // (op 1421) 
s_add_u32 s53, s0, s52 // (op 1421) 

v_and_b32 v18, 1, v13 // (op 1421) 

v_and_b32 v13, 15, v1 // (op 1421) 
s_mul_i32 s52, s16, s8 // (op 1421) 
s_add_u32 s52, s52, s53 // (op 1421) 

v_lshl_add_u32 v22, v18, 4, v13 // (op 1421) 


s_mul_hi_u32 s53, s52, s13 // (op 1421) 
v_and_b32 v13, 63, v1 // (op 1421) 
s_sub_u32 s54, s52, s53 // (op 1421) 
v_lshrrev_b32 v18, 4, v13 // (op 1421) 

s_lshr_b32 s55, s54, 1 // (op 1421) 

s_add_u32 s54, s55, s53 // (op 1421) 


s_lshr_b32 s53, s54, s11 // (op 1421) 

s_and_b32 s54, s11, 2147483648 // (op 1421) 
s_cmp_eq_i32 s54, 0 // (op 1421) 
s_mov_b32 s55, scc // (op 1421) 

s_cmp_eq_i32 s55, 0 // (op 1421) 
s_cselect_b32 s54, s52, s53 // (op 1421) 


s_mul_hi_u32 s53, s54, s12 // (op 1421) 
s_sub_u32 s55, s54, s53 // (op 1421) 
s_lshr_b32 s76, s55, 1 // (op 1421) 

s_add_u32 s55, s76, s53 // (op 1421) 


s_lshr_b32 s53, s55, s10 // (op 1421) 

s_and_b32 s55, s10, 2147483648 // (op 1421) 
s_cmp_eq_i32 s55, 0 // (op 1421) 
s_mov_b32 s76, scc // (op 1421) 

s_cmp_eq_i32 s76, 0 // (op 1421) 
s_cselect_b32 s55, s54, s53 // (op 1421) 


s_mul_i32 s53, s55, s14 // (op 1421) 

s_sub_u32 s55, s54, s53 // (op 1421) 

v_lshl_add_u32 v13, s55, 5, v22 // (op 1421) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1421) 
s_mul_i32 s53, s54, s76 // (op 1421) 


s_sub_u32 s54, s52, s53 // (op 1421) 


v_lshl_add_u32 v22, s54, 2, v18 // (op 1421) 


v_mul_lo_u32 v18, v22, s38 // (op 1421) 
v_add_u32 v18, v18, v13 // (op 1421) 


s_mul_i32 s53, s38, s37 // (op 1423) most significant: low of low * high
s_mul_i32 s54, s39, s36 // (op 1423) most significant: low of high * low
s_mul_i32 s52, s38, s36 // (op 1423) least significant
s_mul_hi_u32 s55, s38, s36 // (op 1423) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1423) most significant: sum
s_add_u32 s53, s53, s55 // (op 1423) most significant: sum


s_mov_b32 s72, s40 // (op 1423) reg expression
s_mov_b32 s73, s41 // (op 1423) reg expression
s_mov_b32 s74, s52 // (op 1423) reg expression
s_mov_b32 s75, 131072 // (op 1423) call()

v_and_b32 v13, -2, v1 // (op 1572) 
v_sub_u32 v22, v1, v13 // (op 1572) 

v_lshlrev_b32 v13, 4, v22 // (op 1572) 

v_lshrrev_b32 v22, 1, v1 // (op 1572) 
v_lshl_add_u32 v23, v22, 5, v13 // (op 1572) 


s_nop 0
v_readfirstlane_b32 s52, v23 // (op 1572) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -2, v1 // (op 1637) 
v_sub_u32 v22, v1, v13 // (op 1637) 

v_lshlrev_b32 v13, 4, v22 // (op 1637) 

v_lshrrev_b32 v22, 1, v1 // (op 1637) 
v_lshl_add_u32 v23, v22, 5, v13 // (op 1637) 


s_nop 0
v_readfirstlane_b32 s53, v23 // (op 1637) Wait state hazard: VALU Write Readlane Hazard

s_and_b32 s54, s7, 4294967294 // (op 544) 
s_cmp_lt_u32 s54, s7 // (op 543) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_543 // (op 543) Condition: Top (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_543 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_543:
s_barrier  // (op 2138) 
s_add_u32 s54, s54, 1 // (op 545) 


v_mov_b64 v[22:23], v[20:21] // (op 946) 
v_mov_b64 v[24:25], v[22:23] // (op 946) reg expression
s_add_u32 s55, 4096, s52 // (op 946) 

s_mov_b32 m0, s55 // (op 946) reg expression
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 946) Load value direct to lds



s_lshl_b64 s[76:77], s[24:25], 7 // (op 1021) 
v_mov_b32 v13, s76 // (op 1021) 
v_mov_b32 v22, s77 // (op 1021) 
v_add_co_u32 v20, s[78:79], v20, v13 // (op 1021) least significant half
v_addc_co_u32 v21, s[78:79], v21, v22, s[78:79] // (op 1021) most significant half




s_barrier  // (op 2157) 


v_mov_b64 v[22:23], v[16:17] // (op 952) 
v_mov_b64 v[24:25], v[22:23] // (op 952) reg expression

s_mov_b32 m0, s53 // (op 952) reg expression
buffer_load_dwordx4 v24, s[64:67], 0 offen offset: 0 lds // (op 952) Load value direct to lds


s_lshl_b64 s[76:77], s[28:29], 7 // (op 1157) 
v_mov_b32 v13, s76 // (op 1157) 
v_mov_b32 v22, s77 // (op 1157) 
v_add_co_u32 v16, s[78:79], v16, v13 // (op 1157) least significant half
v_addc_co_u32 v17, s[78:79], v17, v22, s[78:79] // (op 1157) most significant half




s_waitcnt vmcnt(1)
s_barrier  // (op 2144) 



ds_read_b64_tr_b8 v[22:23], v0 offset:4096 // (op 551) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v0 offset:4352 // (op 551) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:6144 // (op 551) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v0 offset:6400 // (op 551) Transpose load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 2163) 



ds_read_b64_tr_b8 v[30:31], v4 // (op 553) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v4 offset:256 // (op 553) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v4 offset:2048 // (op 553) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:2304 // (op 553) Transpose load local data 

v_mov_b64 v[38:39], s[32:33] // (op 555) convert

v_mov_b32 v13, v19 // (op 555) 
v_mov_b32 v40, v13 // (op 555) reg expression
v_mov_b32 v41, v40 // (op 555) 
buffer_load_ubyte v42, v41, s[68:71], 0 offen offset: 0 // (op 555) Load value




s_lshl_b32 s55, s32, 2 // (op 1285) 
v_add_u32 v19, v19, s55 // (op 1285) 


v_mov_b64 v[38:39], s[38:39] // (op 557) convert

v_mov_b32 v13, v18 // (op 557) 
v_mov_b32 v40, v13 // (op 557) reg expression
v_mov_b32 v41, v40 // (op 557) 
buffer_load_ubyte v43, v41, s[72:75], 0 offen offset: 0 // (op 557) Load value




s_lshl_b32 s55, s38, 2 // (op 1413) 
v_add_u32 v18, v18, s55 // (op 1413) 

s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:37], a[0:3], v42, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 558) 




s_cmp_lt_u32 s54, s7 // (op 543) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_543 // (op 543) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_543 if true)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_543:







s_barrier  // (op 2337) 


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_lg_u32 s18, 0 // (op 283) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Send_Tile_283 // (op 283) Condition: False, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Send_Tile_283
v_lshrrev_b32 v12, 5, v1 // (op 1891) 
v_lshlrev_b32 v13, 2, v12 // (op 1891) 

v_lshl_add_u32 v12, s16, 5, v13 // (op 1891) 

v_and_b32 v13, -32, v1 // (op 1891) 
v_sub_u32 v15, v1, v13 // (op 1891) 

v_lshl_add_u32 v13, v12, 5, v15 // (op 1891) 


v_lshlrev_b32 v12, 2, v13 // (op 1891) 

s_lshl_b32 s52, s15, 12 // (op 1893) 
s_mov_b32 s76, s42 // (op 1893) reg expression
s_mov_b32 s77, s43 // (op 1893) reg expression
s_mov_b32 s78, s52 // (op 1893) reg expression
s_mov_b32 s79, 131072 // (op 1893) call()




buffer_store_dword a0, v12, s[76:79], 0 offen sc1 // (op 280) Store value
buffer_store_dword a1, v12, s[76:79], 0 offen offset:128 sc1 // (op 280) Store value
buffer_store_dword a2, v12, s[76:79], 0 offen offset:256 sc1 // (op 280) Store value
buffer_store_dword a3, v12, s[76:79], 0 offen offset:384 sc1 // (op 280) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 284) 
v_cmp_eq_i32 vcc, v1, 0 // (op 288) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_288 // (op 288) Condition: False, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_288
s_mov_b32 s52, 1 // (op 286) call()
s_mov_b32 s55, 0 // (op 287) convert
s_mov_b32 s54, s16 // (op 287) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 287) 
s_add_u32 s80, s44, s54 // (op 287) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 287) End of Int64 add, unlocking SCC
s_store_dword s52, s[80:81], 0 glc // (op 287) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_288 // (op 288) Condition: Done, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_288
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_288:
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_288:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Send_Tile_283 // (op 283) Condition: Done, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Send_Tile_283
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Send_Tile_283:
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Send_Tile_283:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 299) 
s_mov_b32 s52, scc // (op 299) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 299) 
s_mov_b32 s53, s54 // (op 299) convert

s_sub_u32 s54, s53, 1 // (op 299) 

s_cmp_lt_u32 s19, s54 // (op 299) 
s_mov_b32 s53, scc // (op 299) 

s_and_b32 s54, s52, s53 // (op 299) 
s_cmp_lg_u32 s54, 0 // (op 299) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Receive_Tile_299 // (op 299) Condition: False, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Receive_Tile_299
s_mov_b32 s52, 0 // (op 265) call()
s_mul_i32 s53, s52, s8 // (op 264) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 264) 
s_mov_b32 s80, s54 // (op 264) convert

s_sub_u32 s54, s80, s19 // (op 264) 

s_sub_u32 s55, s54, 1 // (op 264) 

s_cmp_lt_u32 s53, s55 // (op 264) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_264 // (op 264) Condition: Top (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_264 if false)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_ReceiveTileLoop_264:
s_add_u32 s53, s16, 1 // (op 1908) 
v_lshrrev_b32 v12, 5, v1 // (op 1908) 
s_add_u32 s54, s53, s52 // (op 1908) 
s_lshl_b32 s54, s54, 5 // (op 1908) 

v_lshlrev_b32 v13, 2, v12 // (op 1908) 

v_add_lshl_u32 v12, s54, v13, 5 // (op 1908) 


v_and_b32 v13, -32, v1 // (op 1908) 
v_sub_u32 v15, v1, v13 // (op 1908) 

v_add_lshl_u32 v13, v12, v15, 2 // (op 1908) 


s_lshl_b32 s53, s15, 12 // (op 1910) 
s_mov_b32 s76, s42 // (op 1910) reg expression
s_mov_b32 s77, s43 // (op 1910) reg expression
s_mov_b32 s78, s53 // (op 1910) reg expression
s_mov_b32 s79, 131072 // (op 1910) call()

RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_303:
s_add_u32 s53, s16, 1 // (op 301) 
s_add_u32 s54, s53, s52 // (op 301) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 301) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 301) 

s_add_u32 s80, s44, s54 // (op 301) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 301) End of Int64 add, unlocking SCC
s_load_dword s53, s[80:81], 0 glc // (op 301) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 303) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_303 // (op 303) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_303 if true)
s_barrier  // (op 307) 
v_cmp_eq_i32 vcc, v1, 0 // (op 306) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_306 // (op 306) Condition: False, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_306
s_mov_b32 s53, 0 // (op 304) call()
s_add_u32 s54, s16, 1 // (op 305) 
s_add_u32 s80, s54, s52 // (op 305) Start of Int64 add, locking SCC
s_addc_u32 s81, 0, 0 // (op 305) End of Int64 add, unlocking SCC
s_lshl_b64 s[80:81], s[80:81], 2 // (op 305) 

s_add_u32 s54, s44, s80 // (op 305) Start of Int64 add, locking SCC
s_addc_u32 s55, s45, s81 // (op 305) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 305) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_306 // (op 306) Condition: Done, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_306
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_306:
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_306:




buffer_load_dword v16, v13, s[76:79], 0 offen // (op 318) Load value
buffer_load_dword v17, v13, s[76:79], 0 offen offset:128 // (op 318) Load value
buffer_load_dword v18, v13, s[76:79], 0 offen offset:256 // (op 318) Load value
buffer_load_dword v19, v13, s[76:79], 0 offen offset:384 // (op 318) Load value
v_accvgpr_read v20, a0 // (op 321) reg expression
v_accvgpr_read v21, a1 // (op 321) reg expression
v_accvgpr_read v22, a2 // (op 321) reg expression
v_accvgpr_read v23, a3 // (op 321) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v20, v20, v16 // (op 323) 
s_waitcnt vmcnt(2)
v_add_f32 v21, v21, v17 // (op 323) 
s_waitcnt vmcnt(1)
v_add_f32 v22, v22, v18 // (op 323) 
s_waitcnt vmcnt(0)
v_add_f32 v23, v23, v19 // (op 323) 

v_accvgpr_write a0, v20 // (op 325) reg expression
v_accvgpr_write a1, v21 // (op 325) reg expression
v_accvgpr_write a2, v22 // (op 325) reg expression
v_accvgpr_write a3, v23 // (op 325) reg expression

s_add_u32 s52, s52, 1 // (op 266) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 264) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 264) 
s_mov_b32 s80, s54 // (op 264) convert

s_sub_u32 s54, s80, s19 // (op 264) 

s_sub_u32 s55, s54, 1 // (op 264) 

s_cmp_lt_u32 s53, s55 // (op 264) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_ReceiveTileLoop_264 // (op 264) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_ReceiveTileLoop_264 if true)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_264:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Receive_Tile_299 // (op 299) Condition: Done, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Receive_Tile_299
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Receive_Tile_299:
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Receive_Tile_299:

s_cmp_eq_i32 s18, 0 // (op 342) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_342 // (op 342) Condition: False, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_342
v_cmp_eq_f32 vcc, 0.00000, v9 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_114
v_accvgpr_read v16, a0 // (op 42) 
v_accvgpr_read v17, a1 // (op 42) 
v_accvgpr_read v18, a2 // (op 42) 
v_accvgpr_read v19, a3 // (op 42) 
v_mul_f32 v20, v11, v16 // (op 42) 
v_mul_f32 v21, v11, v17 // (op 42) 
v_mul_f32 v22, v11, v18 // (op 42) 
v_mul_f32 v23, v11, v19 // (op 42) 

v_lshrrev_b32 v12, 7, v1 // (op 1691) 
s_mul_i32 s52, s16, s8 // (op 1691) 
s_add_u32 s52, s52, s0 // (op 1691) 
v_and_b32 v13, 63, v1 // (op 1691) 
s_mul_hi_u32 s53, s52, s13 // (op 1691) 
v_lshrrev_b32 v15, 4, v13 // (op 1691) 
s_sub_u32 s54, s52, s53 // (op 1691) 
v_lshlrev_b32 v16, 2, v15 // (op 1691) 

s_lshr_b32 s55, s54, 1 // (op 1691) 

v_lshl_add_u32 v15, v12, 4, v16 // (op 1691) 


s_add_u32 s54, s55, s53 // (op 1691) 


v_lshrrev_b32 v12, 2, v13 // (op 1691) 

s_lshr_b32 s53, s54, s11 // (op 1691) 

v_and_b32 v13, 3, v12 // (op 1691) 

s_and_b32 s54, s11, 2147483648 // (op 1691) 
v_and_b32 v12, 3, v1 // (op 1691) 
s_cmp_eq_i32 s54, 0 // (op 1691) 
s_mov_b32 s55, scc // (op 1691) 

v_lshl_add_u32 v16, v13, 2, v12 // (op 1691) 


s_cmp_eq_i32 s55, 0 // (op 1691) 
s_cselect_b32 s54, s52, s53 // (op 1691) 



v_lshrrev_b32 v12, 6, v1 // (op 1691) 
s_mul_hi_u32 s55, s54, s12 // (op 1691) 
v_and_b32 v13, 1, v12 // (op 1691) 

s_sub_u32 s52, s54, s55 // (op 1691) 
v_lshl_add_u32 v12, v13, 4, v16 // (op 1691) 


s_lshr_b32 s53, s52, 1 // (op 1691) 

s_add_u32 s52, s53, s55 // (op 1691) 


s_lshr_b32 s53, s52, s10 // (op 1691) 

s_and_b32 s52, s10, 2147483648 // (op 1691) 
s_cmp_eq_i32 s52, 0 // (op 1691) 
s_mov_b32 s55, scc // (op 1691) 

s_cmp_eq_i32 s55, 0 // (op 1691) 
s_cselect_b32 s52, s54, s53 // (op 1691) 


s_mul_i32 s53, s52, s14 // (op 1691) 
v_lshl_add_u32 v13, s52, 5, v15 // (op 1691) 


s_sub_u32 s52, s54, s53 // (op 1691) 


v_lshl_add_u32 v15, s52, 5, v12 // (op 1691) 


v_mul_lo_u32 v12, v15, s46 // (op 1691) 

v_add_lshl_u32 v15, v13, v12, 2 // (op 1691) 


s_mul_i32 s53, s46, s23 // (op 1693) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1693) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1693) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1693) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1693) most significant: sum
s_add_u32 s53, s53, s55 // (op 1693) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1693) 

s_mov_b32 s76, s48 // (op 1693) reg expression
s_mov_b32 s77, s49 // (op 1693) reg expression
s_mov_b32 s78, s54 // (op 1693) reg expression
s_mov_b32 s79, 131072 // (op 1693) call()




buffer_store_dwordx4 v[20:23], v15, s[76:79], 0 offen // (op 48) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_114
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_114:



buffer_load_dwordx4 v[16:19], v14, s[56:59], 0 offen // (op 126) Load value
v_accvgpr_read v20, a0 // (op 123) 
v_accvgpr_read v21, a1 // (op 123) 
v_accvgpr_read v22, a2 // (op 123) 
v_accvgpr_read v23, a3 // (op 123) 
v_mul_f32 v24, v11, v20 // (op 123) 
v_mul_f32 v25, v11, v21 // (op 123) 
v_mul_f32 v26, v11, v22 // (op 123) 
v_mul_f32 v27, v11, v23 // (op 123) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v9, v16 // (op 127) 
v_mul_f32 v21, v9, v17 // (op 127) 
v_mul_f32 v22, v9, v18 // (op 127) 
v_mul_f32 v23, v9, v19 // (op 127) 

v_add_f32 v16, v20, v24 // (op 124) 
v_add_f32 v17, v21, v25 // (op 124) 
v_add_f32 v18, v22, v26 // (op 124) 
v_add_f32 v19, v23, v27 // (op 124) 


v_lshrrev_b32 v15, 7, v1 // (op 1708) 
s_mul_i32 s52, s16, s8 // (op 1708) 
s_add_u32 s52, s52, s0 // (op 1708) 
v_and_b32 v12, 63, v1 // (op 1708) 
s_mul_hi_u32 s53, s52, s13 // (op 1708) 
v_lshrrev_b32 v13, 4, v12 // (op 1708) 
s_sub_u32 s54, s52, s53 // (op 1708) 
v_lshlrev_b32 v20, 2, v13 // (op 1708) 

s_lshr_b32 s55, s54, 1 // (op 1708) 

v_lshl_add_u32 v13, v15, 4, v20 // (op 1708) 


s_add_u32 s54, s55, s53 // (op 1708) 


v_lshrrev_b32 v15, 2, v12 // (op 1708) 

s_lshr_b32 s53, s54, s11 // (op 1708) 

v_and_b32 v12, 3, v15 // (op 1708) 

s_and_b32 s54, s11, 2147483648 // (op 1708) 
v_and_b32 v15, 3, v1 // (op 1708) 
s_cmp_eq_i32 s54, 0 // (op 1708) 
s_mov_b32 s55, scc // (op 1708) 

v_lshl_add_u32 v20, v12, 2, v15 // (op 1708) 


s_cmp_eq_i32 s55, 0 // (op 1708) 
s_cselect_b32 s54, s52, s53 // (op 1708) 



v_lshrrev_b32 v12, 6, v1 // (op 1708) 
s_mul_hi_u32 s55, s54, s12 // (op 1708) 
v_and_b32 v15, 1, v12 // (op 1708) 

s_sub_u32 s52, s54, s55 // (op 1708) 
v_lshl_add_u32 v12, v15, 4, v20 // (op 1708) 


s_lshr_b32 s53, s52, 1 // (op 1708) 

s_add_u32 s52, s53, s55 // (op 1708) 


s_lshr_b32 s53, s52, s10 // (op 1708) 

s_and_b32 s52, s10, 2147483648 // (op 1708) 
s_cmp_eq_i32 s52, 0 // (op 1708) 
s_mov_b32 s55, scc // (op 1708) 

s_cmp_eq_i32 s55, 0 // (op 1708) 
s_cselect_b32 s52, s54, s53 // (op 1708) 


s_mul_i32 s53, s52, s14 // (op 1708) 
v_lshl_add_u32 v15, s52, 5, v13 // (op 1708) 


s_sub_u32 s52, s54, s53 // (op 1708) 


v_lshl_add_u32 v13, s52, 5, v12 // (op 1708) 


v_mul_lo_u32 v12, v13, s46 // (op 1708) 

v_add_lshl_u32 v13, v15, v12, 2 // (op 1708) 


s_mul_i32 s53, s46, s23 // (op 1710) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1710) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1710) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1710) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1710) most significant: sum
s_add_u32 s53, s53, s55 // (op 1710) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1710) 

s_mov_b32 s76, s48 // (op 1710) reg expression
s_mov_b32 s77, s49 // (op 1710) reg expression
s_mov_b32 s78, s54 // (op 1710) reg expression
s_mov_b32 s79, 131072 // (op 1710) call()




buffer_store_dwordx4 v[16:19], v13, s[76:79], 0 offen // (op 125) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_114:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_342 // (op 342) Condition: Done, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_342
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_342:
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_342:

s_add_u32 s0, s0, s7 // (op 257) 
s_mul_i32 s52, s8, s16 // (op 255) 
s_cmp_gt_u32 s9, s52 // (op 255) 
s_mov_b32 s53, scc // (op 255) 
s_sub_u32 s54, s9, s52 // (op 255) 

s_cmp_eq_i32 s53, 0 // (op 255) 
s_cselect_b32 s52, 0, s54 // (op 255) 


s_cmp_lt_u32 s8, s52 // (op 255) 
s_mov_b32 s53, scc // (op 255) 
s_cmp_eq_i32 s53, 0 // (op 255) 
s_cselect_b32 s54, s52, s8 // (op 255) 


s_cmp_lt_u32 s0, s54 // (op 255) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_SKStreamTileLoop_255 // (op 255) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_SKStreamTileLoop_255 if true)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_255:













s_mov_b32 s0, 0 // (op 396) call()
s_mov_b32 s1, 0 // (op 397) call()
s_mov_b32 s0, 0 // (op 411) call()
s_mul_i32 s15, s50, s16 // (op 410) 
s_cmp_gt_u32 s51, s15 // (op 410) 
s_mov_b32 s6, scc // (op 410) 
s_sub_u32 s7, s51, s15 // (op 410) 

s_cmp_eq_i32 s6, 0 // (op 410) 
s_cselect_b32 s8, 0, s7 // (op 410) 


s_cmp_lt_u32 s50, s8 // (op 410) 
s_mov_b32 s15, scc // (op 410) 
s_cmp_eq_i32 s15, 0 // (op 410) 
s_cselect_b32 s6, s8, s50 // (op 410) 


s_cmp_lt_u32 s0, s6 // (op 410) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_410 // (op 410) Condition: Top (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_410 if false)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_DPStreamTileLoop_410:
v_lshrrev_b32 v0, 7, v1 // (op 1542) 
s_mul_i32 s15, s16, s50 // (op 1542) 
s_add_u32 s15, s15, s0 // (op 1542) 
v_and_b32 v4, 63, v1 // (op 1542) 
s_add_u32 s6, s15, s9 // (op 1542) 

v_lshrrev_b32 v7, 4, v4 // (op 1542) 
s_mul_hi_u32 s15, s6, s13 // (op 1542) 
v_lshlrev_b32 v10, 2, v7 // (op 1542) 

s_sub_u32 s7, s6, s15 // (op 1542) 
v_lshl_add_u32 v7, v0, 4, v10 // (op 1542) 


s_lshr_b32 s8, s7, 1 // (op 1542) 

v_lshrrev_b32 v0, 2, v4 // (op 1542) 

s_add_u32 s7, s8, s15 // (op 1542) 


v_and_b32 v4, 3, v0 // (op 1542) 

s_lshr_b32 s8, s7, s11 // (op 1542) 

v_and_b32 v0, 3, v1 // (op 1542) 
s_and_b32 s7, s11, 2147483648 // (op 1542) 
v_lshl_add_u32 v10, v4, 2, v0 // (op 1542) 


s_cmp_eq_i32 s7, 0 // (op 1542) 
s_mov_b32 s15, scc // (op 1542) 

v_lshrrev_b32 v0, 6, v1 // (op 1542) 
s_cmp_eq_i32 s15, 0 // (op 1542) 
s_cselect_b32 s7, s6, s8 // (op 1542) 



v_and_b32 v4, 1, v0 // (op 1542) 

s_mul_hi_u32 s6, s7, s12 // (op 1542) 
v_lshl_add_u32 v0, v4, 4, v10 // (op 1542) 


s_sub_u32 s8, s7, s6 // (op 1542) 
s_lshr_b32 s15, s8, 1 // (op 1542) 

s_add_u32 s8, s15, s6 // (op 1542) 


s_lshr_b32 s6, s8, s10 // (op 1542) 

s_and_b32 s8, s10, 2147483648 // (op 1542) 
s_cmp_eq_i32 s8, 0 // (op 1542) 
s_mov_b32 s15, scc // (op 1542) 

s_cmp_eq_i32 s15, 0 // (op 1542) 
s_cselect_b32 s8, s7, s6 // (op 1542) 


s_mul_i32 s6, s8, s14 // (op 1542) 
v_lshl_add_u32 v4, s8, 5, v7 // (op 1542) 


s_sub_u32 s8, s7, s6 // (op 1542) 


v_lshl_add_u32 v7, s8, 5, v0 // (op 1542) 


v_mul_lo_u32 v0, v7, s4 // (op 1542) 

v_add_lshl_u32 v7, v4, v0, 2 // (op 1542) 


s_mul_i32 s19, s4, s23 // (op 1544) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 1544) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 1544) least significant
s_mul_hi_u32 s6, s4, s22 // (op 1544) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1544) most significant: sum
s_add_u32 s19, s19, s6 // (op 1544) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1544) 

s_mov_b32 s56, s20 // (op 1544) reg expression
s_mov_b32 s57, s21 // (op 1544) reg expression
s_mov_b32 s58, s6 // (op 1544) reg expression
s_mov_b32 s59, 131072 // (op 1544) call()

v_and_b32 v0, 48, v1 // (op 1811) 
v_and_b32 v4, 63, v1 // (op 1811) 
v_sub_u32 v10, v4, v0 // (op 1811) 

v_lshrrev_b32 v4, 1, v10 // (op 1811) 
v_add_lshl_u32 v12, v0, v4, 5 // (op 1811) 


v_and_b32 v0, 1, v10 // (op 1811) 

v_lshlrev_b32 v4, 3, v0 // (op 1811) 

v_lshrrev_b32 v0, 7, v1 // (op 1811) 
v_lshl_add_u32 v10, v0, 4, v4 // (op 1811) 


v_add_u32 v0, v12, v10 // (op 1811) 


v_and_b32 v4, 48, v1 // (op 1876) 
v_and_b32 v10, 63, v1 // (op 1876) 
v_sub_u32 v12, v10, v4 // (op 1876) 

v_lshrrev_b32 v10, 1, v12 // (op 1876) 
v_add_lshl_u32 v13, v4, v10, 5 // (op 1876) 


v_and_b32 v4, 1, v12 // (op 1876) 

v_lshlrev_b32 v10, 3, v4 // (op 1876) 

v_lshrrev_b32 v4, 6, v1 // (op 1876) 
v_and_b32 v12, 1, v4 // (op 1876) 

v_lshl_add_u32 v4, v12, 4, v10 // (op 1876) 


v_add_u32 v10, v13, v4 // (op 1876) 


v_accvgpr_write a0, 0.00000 // (op 161) call()
v_accvgpr_write a1, 0.00000 // (op 161) call()
v_accvgpr_write a2, 0.00000 // (op 161) call()
v_accvgpr_write a3, 0.00000 // (op 161) call()
v_and_b32 v4, -2, v1 // (op 1102) 
s_mul_i32 s15, s16, s50 // (op 1102) 
s_add_u32 s15, s15, s0 // (op 1102) 
v_sub_u32 v12, v1, v4 // (op 1102) 

s_add_u32 s6, s15, s9 // (op 1102) 

v_lshlrev_b32 v4, 4, v12 // (op 1102) 

s_mul_hi_u32 s15, s6, s13 // (op 1102) 
v_lshrrev_b32 v12, 1, v1 // (op 1102) 
s_sub_u32 s7, s6, s15 // (op 1102) 
s_lshr_b32 s8, s7, 1 // (op 1102) 

s_add_u32 s7, s8, s15 // (op 1102) 


s_lshr_b32 s8, s7, s11 // (op 1102) 

s_and_b32 s7, s11, 2147483648 // (op 1102) 
s_cmp_eq_i32 s7, 0 // (op 1102) 
s_mov_b32 s15, scc // (op 1102) 

s_cmp_eq_i32 s15, 0 // (op 1102) 
s_cselect_b32 s7, s6, s8 // (op 1102) 


s_mul_hi_u32 s8, s7, s12 // (op 1102) 
s_sub_u32 s15, s7, s8 // (op 1102) 
s_lshr_b32 s18, s15, 1 // (op 1102) 

s_add_u32 s15, s18, s8 // (op 1102) 


s_lshr_b32 s8, s15, s10 // (op 1102) 

s_and_b32 s15, s10, 2147483648 // (op 1102) 
s_cmp_eq_i32 s15, 0 // (op 1102) 
s_mov_b32 s18, scc // (op 1102) 

s_cmp_eq_i32 s18, 0 // (op 1102) 
s_cselect_b32 s15, s7, s8 // (op 1102) 


v_lshl_add_u32 v13, s15, 5, v4 // (op 1102) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1102) 
v_mov_b32 v15, 0 // (op 1102) convert
v_mov_b32 v14, v13 // (op 1102) convert

s_mul_i32 s8, s7, s18 // (op 1102) 


s_sub_u32 s7, s6, s8 // (op 1102) 


v_lshl_add_u32 v4, s7, 7, v12 // (op 1102) 


v_mul_lo_u32 v13, v4, s25 // (op 1102) most significant: low of low * high
v_mul_hi_u32 v16, v4, s24 // (op 1102) most significant: high of low * low
v_mul_lo_u32 v12, v4, s24 // (op 1102) least significant: low of low * low
v_add_u32 v13, v13, v16 // (op 1102) most significant: sum


v_add_co_u32 v16, s[18:19], v14, v12 // (op 1102) least significant half
v_addc_co_u32 v17, s[18:19], v15, v13, s[18:19] // (op 1102) most significant half



s_mul_i32 s19, s24, s3 // (op 1104) most significant: low of low * high
s_mul_i32 s15, s25, s2 // (op 1104) most significant: low of high * low
s_mul_i32 s18, s24, s2 // (op 1104) least significant
s_mul_hi_u32 s6, s24, s2 // (op 1104) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1104) most significant: sum
s_add_u32 s19, s19, s6 // (op 1104) most significant: sum


s_mov_b32 s60, s26 // (op 1104) reg expression
s_mov_b32 s61, s27 // (op 1104) reg expression
s_mov_b32 s62, s18 // (op 1104) reg expression
s_mov_b32 s63, 131072 // (op 1104) call()

v_and_b32 v4, -2, v1 // (op 1121) 
s_mul_i32 s15, s16, s50 // (op 1121) 
s_add_u32 s15, s15, s0 // (op 1121) 
v_sub_u32 v12, v1, v4 // (op 1121) 

s_add_u32 s6, s15, s9 // (op 1121) 

v_lshlrev_b32 v4, 4, v12 // (op 1121) 

s_mul_hi_u32 s15, s6, s13 // (op 1121) 
v_lshrrev_b32 v12, 1, v1 // (op 1121) 
s_sub_u32 s7, s6, s15 // (op 1121) 
s_lshr_b32 s8, s7, 1 // (op 1121) 

s_add_u32 s7, s8, s15 // (op 1121) 


s_lshr_b32 s8, s7, s11 // (op 1121) 

s_and_b32 s7, s11, 2147483648 // (op 1121) 
s_cmp_eq_i32 s7, 0 // (op 1121) 
s_mov_b32 s15, scc // (op 1121) 

s_cmp_eq_i32 s15, 0 // (op 1121) 
s_cselect_b32 s7, s6, s8 // (op 1121) 


s_mul_hi_u32 s8, s7, s12 // (op 1121) 
s_sub_u32 s15, s7, s8 // (op 1121) 
s_lshr_b32 s18, s15, 1 // (op 1121) 

s_add_u32 s15, s18, s8 // (op 1121) 


s_lshr_b32 s8, s15, s10 // (op 1121) 

s_and_b32 s15, s10, 2147483648 // (op 1121) 
s_cmp_eq_i32 s15, 0 // (op 1121) 
s_mov_b32 s18, scc // (op 1121) 

s_cmp_eq_i32 s18, 0 // (op 1121) 
s_cselect_b32 s15, s7, s8 // (op 1121) 


v_lshl_add_u32 v13, s15, 5, v4 // (op 1121) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1121) 
v_mov_b32 v15, 0 // (op 1121) convert
v_mov_b32 v14, v13 // (op 1121) convert

s_mul_i32 s8, s7, s18 // (op 1121) 


s_sub_u32 s7, s6, s8 // (op 1121) 


v_lshl_add_u32 v4, s7, 7, v12 // (op 1121) 


v_mul_lo_u32 v13, v4, s25 // (op 1121) most significant: low of low * high
v_mul_hi_u32 v18, v4, s24 // (op 1121) most significant: high of low * low
v_mul_lo_u32 v12, v4, s24 // (op 1121) least significant: low of low * low
v_add_u32 v13, v13, v18 // (op 1121) most significant: sum


v_add_co_u32 v18, s[18:19], v14, v12 // (op 1121) least significant half
v_addc_co_u32 v19, s[18:19], v15, v13, s[18:19] // (op 1121) most significant half



s_mul_i32 s19, s24, s3 // (op 1123) most significant: low of low * high
s_mul_i32 s15, s25, s2 // (op 1123) most significant: low of high * low
s_mul_i32 s18, s24, s2 // (op 1123) least significant
s_mul_hi_u32 s6, s24, s2 // (op 1123) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1123) most significant: sum
s_add_u32 s19, s19, s6 // (op 1123) most significant: sum


s_mov_b32 s60, s26 // (op 1123) reg expression
s_mov_b32 s61, s27 // (op 1123) reg expression
s_mov_b32 s62, s18 // (op 1123) reg expression
s_mov_b32 s63, 131072 // (op 1123) call()

v_and_b32 v4, -2, v1 // (op 1230) 
s_mul_i32 s15, s16, s50 // (op 1230) 
s_add_u32 s15, s15, s0 // (op 1230) 
v_sub_u32 v12, v1, v4 // (op 1230) 

s_add_u32 s6, s15, s9 // (op 1230) 

v_lshlrev_b32 v4, 4, v12 // (op 1230) 

s_mul_hi_u32 s15, s6, s13 // (op 1230) 
v_lshrrev_b32 v12, 1, v1 // (op 1230) 
s_sub_u32 s7, s6, s15 // (op 1230) 
s_lshr_b32 s8, s7, 1 // (op 1230) 

s_add_u32 s7, s8, s15 // (op 1230) 


s_lshr_b32 s8, s7, s11 // (op 1230) 

s_and_b32 s7, s11, 2147483648 // (op 1230) 
s_cmp_eq_i32 s7, 0 // (op 1230) 
s_mov_b32 s15, scc // (op 1230) 

s_cmp_eq_i32 s15, 0 // (op 1230) 
s_cselect_b32 s7, s6, s8 // (op 1230) 


s_mul_hi_u32 s8, s7, s12 // (op 1230) 
s_sub_u32 s15, s7, s8 // (op 1230) 
s_lshr_b32 s18, s15, 1 // (op 1230) 

s_add_u32 s15, s18, s8 // (op 1230) 


s_lshr_b32 s8, s15, s10 // (op 1230) 

s_and_b32 s15, s10, 2147483648 // (op 1230) 
s_cmp_eq_i32 s15, 0 // (op 1230) 
s_mov_b32 s18, scc // (op 1230) 

s_cmp_eq_i32 s18, 0 // (op 1230) 
s_cselect_b32 s15, s7, s8 // (op 1230) 


s_mul_i32 s8, s15, s14 // (op 1230) 

s_sub_u32 s15, s7, s8 // (op 1230) 

v_lshl_add_u32 v13, s15, 5, v4 // (op 1230) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1230) 
v_mov_b32 v15, 0 // (op 1230) convert
v_mov_b32 v14, v13 // (op 1230) convert

s_mul_i32 s8, s7, s18 // (op 1230) 


s_sub_u32 s7, s6, s8 // (op 1230) 


v_lshl_add_u32 v4, s7, 7, v12 // (op 1230) 


v_mul_lo_u32 v13, v4, s29 // (op 1230) most significant: low of low * high
v_mul_hi_u32 v20, v4, s28 // (op 1230) most significant: high of low * low
v_mul_lo_u32 v12, v4, s28 // (op 1230) least significant: low of low * low
v_add_u32 v13, v13, v20 // (op 1230) most significant: sum

v_add_co_u32 v12, s[18:19], v12, v14 // (op 1230) least significant half
v_addc_co_u32 v13, s[18:19], v13, v15, s[18:19] // (op 1230) most significant half



s_mul_i32 s19, s28, s3 // (op 1232) most significant: low of low * high
s_mul_i32 s15, s29, s2 // (op 1232) most significant: low of high * low
s_mul_i32 s18, s28, s2 // (op 1232) least significant
s_mul_hi_u32 s6, s28, s2 // (op 1232) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1232) most significant: sum
s_add_u32 s19, s19, s6 // (op 1232) most significant: sum


s_mov_b32 s64, s30 // (op 1232) reg expression
s_mov_b32 s65, s31 // (op 1232) reg expression
s_mov_b32 s66, s18 // (op 1232) reg expression
s_mov_b32 s67, 131072 // (op 1232) call()

v_and_b32 v4, -2, v1 // (op 1249) 
s_mul_i32 s15, s16, s50 // (op 1249) 
s_add_u32 s15, s15, s0 // (op 1249) 
v_sub_u32 v14, v1, v4 // (op 1249) 

s_add_u32 s6, s15, s9 // (op 1249) 

v_lshlrev_b32 v4, 4, v14 // (op 1249) 

s_mul_hi_u32 s15, s6, s13 // (op 1249) 
v_lshrrev_b32 v14, 1, v1 // (op 1249) 
s_sub_u32 s7, s6, s15 // (op 1249) 
s_lshr_b32 s8, s7, 1 // (op 1249) 

s_add_u32 s7, s8, s15 // (op 1249) 


s_lshr_b32 s8, s7, s11 // (op 1249) 

s_and_b32 s7, s11, 2147483648 // (op 1249) 
s_cmp_eq_i32 s7, 0 // (op 1249) 
s_mov_b32 s15, scc // (op 1249) 

s_cmp_eq_i32 s15, 0 // (op 1249) 
s_cselect_b32 s7, s6, s8 // (op 1249) 


s_mul_hi_u32 s8, s7, s12 // (op 1249) 
s_sub_u32 s15, s7, s8 // (op 1249) 
s_lshr_b32 s18, s15, 1 // (op 1249) 

s_add_u32 s15, s18, s8 // (op 1249) 


s_lshr_b32 s8, s15, s10 // (op 1249) 

s_and_b32 s15, s10, 2147483648 // (op 1249) 
s_cmp_eq_i32 s15, 0 // (op 1249) 
s_mov_b32 s18, scc // (op 1249) 

s_cmp_eq_i32 s18, 0 // (op 1249) 
s_cselect_b32 s15, s7, s8 // (op 1249) 


s_mul_i32 s8, s15, s14 // (op 1249) 

s_sub_u32 s15, s7, s8 // (op 1249) 

v_lshl_add_u32 v15, s15, 5, v4 // (op 1249) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1249) 
v_mov_b32 v21, 0 // (op 1249) convert
v_mov_b32 v20, v15 // (op 1249) convert

s_mul_i32 s8, s7, s18 // (op 1249) 


s_sub_u32 s7, s6, s8 // (op 1249) 


v_lshl_add_u32 v4, s7, 7, v14 // (op 1249) 


v_mul_lo_u32 v15, v4, s29 // (op 1249) most significant: low of low * high
v_mul_hi_u32 v22, v4, s28 // (op 1249) most significant: high of low * low
v_mul_lo_u32 v14, v4, s28 // (op 1249) least significant: low of low * low
v_add_u32 v15, v15, v22 // (op 1249) most significant: sum

v_add_co_u32 v14, s[18:19], v14, v20 // (op 1249) least significant half
v_addc_co_u32 v15, s[18:19], v15, v21, s[18:19] // (op 1249) most significant half



s_mul_i32 s19, s28, s3 // (op 1251) most significant: low of low * high
s_mul_i32 s15, s29, s2 // (op 1251) most significant: low of high * low
s_mul_i32 s18, s28, s2 // (op 1251) least significant
s_mul_hi_u32 s6, s28, s2 // (op 1251) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1251) most significant: sum
s_add_u32 s19, s19, s6 // (op 1251) most significant: sum


s_mov_b32 s64, s30 // (op 1251) reg expression
s_mov_b32 s65, s31 // (op 1251) reg expression
s_mov_b32 s66, s18 // (op 1251) reg expression
s_mov_b32 s67, 131072 // (op 1251) call()

v_lshrrev_b32 v4, 7, v1 // (op 1358) 
s_mul_i32 s15, s16, s50 // (op 1358) 
s_add_u32 s15, s15, s0 // (op 1358) 
v_and_b32 v20, 15, v1 // (op 1358) 
s_add_u32 s6, s15, s9 // (op 1358) 

v_lshl_add_u32 v21, v4, 4, v20 // (op 1358) 


s_mul_hi_u32 s15, s6, s13 // (op 1358) 
v_and_b32 v4, 63, v1 // (op 1358) 
s_sub_u32 s7, s6, s15 // (op 1358) 
v_lshrrev_b32 v20, 4, v4 // (op 1358) 

s_lshr_b32 s8, s7, 1 // (op 1358) 

s_add_u32 s7, s8, s15 // (op 1358) 


s_lshr_b32 s8, s7, s11 // (op 1358) 

s_and_b32 s7, s11, 2147483648 // (op 1358) 
s_cmp_eq_i32 s7, 0 // (op 1358) 
s_mov_b32 s15, scc // (op 1358) 

s_cmp_eq_i32 s15, 0 // (op 1358) 
s_cselect_b32 s7, s6, s8 // (op 1358) 


s_mul_hi_u32 s8, s7, s12 // (op 1358) 
s_sub_u32 s15, s7, s8 // (op 1358) 
s_lshr_b32 s18, s15, 1 // (op 1358) 

s_add_u32 s15, s18, s8 // (op 1358) 


s_lshr_b32 s8, s15, s10 // (op 1358) 

s_and_b32 s15, s10, 2147483648 // (op 1358) 
s_cmp_eq_i32 s15, 0 // (op 1358) 
s_mov_b32 s18, scc // (op 1358) 

s_cmp_eq_i32 s18, 0 // (op 1358) 
s_cselect_b32 s15, s7, s8 // (op 1358) 


v_lshl_add_u32 v4, s15, 5, v21 // (op 1358) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1358) 
s_mul_i32 s8, s7, s18 // (op 1358) 


s_sub_u32 s7, s6, s8 // (op 1358) 


v_lshl_add_u32 v21, s7, 2, v20 // (op 1358) 


v_mul_lo_u32 v20, v21, s32 // (op 1358) 

v_add_u32 v21, v4, v20 // (op 1358) 


s_mul_i32 s19, s32, s37 // (op 1360) most significant: low of low * high
s_mul_i32 s15, s33, s36 // (op 1360) most significant: low of high * low
s_mul_i32 s18, s32, s36 // (op 1360) least significant
s_mul_hi_u32 s6, s32, s36 // (op 1360) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1360) most significant: sum
s_add_u32 s19, s19, s6 // (op 1360) most significant: sum


s_mov_b32 s68, s34 // (op 1360) reg expression
s_mov_b32 s69, s35 // (op 1360) reg expression
s_mov_b32 s70, s18 // (op 1360) reg expression
s_mov_b32 s71, 131072 // (op 1360) call()

v_lshrrev_b32 v4, 7, v1 // (op 1377) 
s_mul_i32 s15, s16, s50 // (op 1377) 
s_add_u32 s15, s15, s0 // (op 1377) 
v_and_b32 v20, 15, v1 // (op 1377) 
s_add_u32 s6, s15, s9 // (op 1377) 

v_lshl_add_u32 v22, v4, 4, v20 // (op 1377) 


s_mul_hi_u32 s15, s6, s13 // (op 1377) 
v_and_b32 v4, 63, v1 // (op 1377) 
s_sub_u32 s7, s6, s15 // (op 1377) 
v_lshrrev_b32 v20, 4, v4 // (op 1377) 

s_lshr_b32 s8, s7, 1 // (op 1377) 

s_add_u32 s7, s8, s15 // (op 1377) 


s_lshr_b32 s8, s7, s11 // (op 1377) 

s_and_b32 s7, s11, 2147483648 // (op 1377) 
s_cmp_eq_i32 s7, 0 // (op 1377) 
s_mov_b32 s15, scc // (op 1377) 

s_cmp_eq_i32 s15, 0 // (op 1377) 
s_cselect_b32 s7, s6, s8 // (op 1377) 


s_mul_hi_u32 s8, s7, s12 // (op 1377) 
s_sub_u32 s15, s7, s8 // (op 1377) 
s_lshr_b32 s18, s15, 1 // (op 1377) 

s_add_u32 s15, s18, s8 // (op 1377) 


s_lshr_b32 s8, s15, s10 // (op 1377) 

s_and_b32 s15, s10, 2147483648 // (op 1377) 
s_cmp_eq_i32 s15, 0 // (op 1377) 
s_mov_b32 s18, scc // (op 1377) 

s_cmp_eq_i32 s18, 0 // (op 1377) 
s_cselect_b32 s15, s7, s8 // (op 1377) 


v_lshl_add_u32 v4, s15, 5, v22 // (op 1377) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1377) 
s_mul_i32 s8, s7, s18 // (op 1377) 


s_sub_u32 s7, s6, s8 // (op 1377) 


v_lshl_add_u32 v22, s7, 2, v20 // (op 1377) 


v_mul_lo_u32 v20, v22, s32 // (op 1377) 

v_add_u32 v22, v4, v20 // (op 1377) 


s_mul_i32 s19, s32, s37 // (op 1379) most significant: low of low * high
s_mul_i32 s15, s33, s36 // (op 1379) most significant: low of high * low
s_mul_i32 s18, s32, s36 // (op 1379) least significant
s_mul_hi_u32 s6, s32, s36 // (op 1379) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1379) most significant: sum
s_add_u32 s19, s19, s6 // (op 1379) most significant: sum


s_mov_b32 s68, s34 // (op 1379) reg expression
s_mov_b32 s69, s35 // (op 1379) reg expression
s_mov_b32 s70, s18 // (op 1379) reg expression
s_mov_b32 s71, 131072 // (op 1379) call()

v_lshrrev_b32 v4, 6, v1 // (op 1486) 
s_mul_i32 s15, s16, s50 // (op 1486) 
s_add_u32 s15, s15, s0 // (op 1486) 
v_and_b32 v20, 1, v4 // (op 1486) 

s_add_u32 s6, s15, s9 // (op 1486) 

v_and_b32 v4, 15, v1 // (op 1486) 
s_mul_hi_u32 s15, s6, s13 // (op 1486) 
v_lshl_add_u32 v23, v20, 4, v4 // (op 1486) 


s_sub_u32 s7, s6, s15 // (op 1486) 
v_and_b32 v4, 63, v1 // (op 1486) 
s_lshr_b32 s8, s7, 1 // (op 1486) 

v_lshrrev_b32 v20, 4, v4 // (op 1486) 

s_add_u32 s7, s8, s15 // (op 1486) 


s_lshr_b32 s8, s7, s11 // (op 1486) 

s_and_b32 s7, s11, 2147483648 // (op 1486) 
s_cmp_eq_i32 s7, 0 // (op 1486) 
s_mov_b32 s15, scc // (op 1486) 

s_cmp_eq_i32 s15, 0 // (op 1486) 
s_cselect_b32 s7, s6, s8 // (op 1486) 


s_mul_hi_u32 s8, s7, s12 // (op 1486) 
s_sub_u32 s15, s7, s8 // (op 1486) 
s_lshr_b32 s18, s15, 1 // (op 1486) 

s_add_u32 s15, s18, s8 // (op 1486) 


s_lshr_b32 s8, s15, s10 // (op 1486) 

s_and_b32 s15, s10, 2147483648 // (op 1486) 
s_cmp_eq_i32 s15, 0 // (op 1486) 
s_mov_b32 s18, scc // (op 1486) 

s_cmp_eq_i32 s18, 0 // (op 1486) 
s_cselect_b32 s15, s7, s8 // (op 1486) 


s_mul_i32 s8, s15, s14 // (op 1486) 

s_sub_u32 s15, s7, s8 // (op 1486) 

v_lshl_add_u32 v4, s15, 5, v23 // (op 1486) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1486) 
s_mul_i32 s8, s7, s18 // (op 1486) 


s_sub_u32 s7, s6, s8 // (op 1486) 


v_lshl_add_u32 v23, s7, 2, v20 // (op 1486) 


v_mul_lo_u32 v20, v23, s38 // (op 1486) 
v_add_u32 v20, v20, v4 // (op 1486) 


s_mul_i32 s19, s38, s37 // (op 1488) most significant: low of low * high
s_mul_i32 s15, s39, s36 // (op 1488) most significant: low of high * low
s_mul_i32 s18, s38, s36 // (op 1488) least significant
s_mul_hi_u32 s6, s38, s36 // (op 1488) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1488) most significant: sum
s_add_u32 s19, s19, s6 // (op 1488) most significant: sum


s_mov_b32 s72, s40 // (op 1488) reg expression
s_mov_b32 s73, s41 // (op 1488) reg expression
s_mov_b32 s74, s18 // (op 1488) reg expression
s_mov_b32 s75, 131072 // (op 1488) call()

v_lshrrev_b32 v4, 6, v1 // (op 1505) 
s_mul_i32 s15, s16, s50 // (op 1505) 
s_add_u32 s15, s15, s0 // (op 1505) 
v_and_b32 v23, 1, v4 // (op 1505) 

s_add_u32 s6, s15, s9 // (op 1505) 

v_and_b32 v4, 15, v1 // (op 1505) 
s_mul_hi_u32 s15, s6, s13 // (op 1505) 
v_lshl_add_u32 v24, v23, 4, v4 // (op 1505) 


s_sub_u32 s7, s6, s15 // (op 1505) 
v_and_b32 v4, 63, v1 // (op 1505) 
s_lshr_b32 s8, s7, 1 // (op 1505) 

v_lshrrev_b32 v23, 4, v4 // (op 1505) 

s_add_u32 s7, s8, s15 // (op 1505) 


s_lshr_b32 s8, s7, s11 // (op 1505) 

s_and_b32 s7, s11, 2147483648 // (op 1505) 
s_cmp_eq_i32 s7, 0 // (op 1505) 
s_mov_b32 s15, scc // (op 1505) 

s_cmp_eq_i32 s15, 0 // (op 1505) 
s_cselect_b32 s7, s6, s8 // (op 1505) 


s_mul_hi_u32 s8, s7, s12 // (op 1505) 
s_sub_u32 s15, s7, s8 // (op 1505) 
s_lshr_b32 s18, s15, 1 // (op 1505) 

s_add_u32 s15, s18, s8 // (op 1505) 


s_lshr_b32 s8, s15, s10 // (op 1505) 

s_and_b32 s15, s10, 2147483648 // (op 1505) 
s_cmp_eq_i32 s15, 0 // (op 1505) 
s_mov_b32 s18, scc // (op 1505) 

s_cmp_eq_i32 s18, 0 // (op 1505) 
s_cselect_b32 s15, s7, s8 // (op 1505) 


s_mul_i32 s8, s15, s14 // (op 1505) 

s_sub_u32 s15, s7, s8 // (op 1505) 

v_lshl_add_u32 v4, s15, 5, v24 // (op 1505) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1505) 
s_mul_i32 s8, s7, s18 // (op 1505) 


s_sub_u32 s7, s6, s8 // (op 1505) 


v_lshl_add_u32 v24, s7, 2, v23 // (op 1505) 


v_mul_lo_u32 v23, v24, s38 // (op 1505) 
v_add_u32 v23, v23, v4 // (op 1505) 


s_mul_i32 s19, s38, s37 // (op 1507) most significant: low of low * high
s_mul_i32 s15, s39, s36 // (op 1507) most significant: low of high * low
s_mul_i32 s18, s38, s36 // (op 1507) least significant
s_mul_hi_u32 s6, s38, s36 // (op 1507) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1507) most significant: sum
s_add_u32 s19, s19, s6 // (op 1507) most significant: sum


s_mov_b32 s72, s40 // (op 1507) reg expression
s_mov_b32 s73, s41 // (op 1507) reg expression
s_mov_b32 s74, s18 // (op 1507) reg expression
s_mov_b32 s75, 131072 // (op 1507) call()


v_mov_b64 v[24:25], s[32:33] // (op 825) convert

v_mov_b32 v4, v21 // (op 825) reg expression
v_mov_b32 v26, v4 // (op 825) 
buffer_load_ubyte v27, v26, s[68:71], 0 offen offset: 0 // (op 825) Load value




v_mov_b64 v[24:25], s[38:39] // (op 830) convert

v_mov_b32 v4, v20 // (op 830) reg expression
v_mov_b32 v26, v4 // (op 830) 
buffer_load_ubyte v28, v26, s[72:75], 0 offen offset: 0 // (op 830) Load value



v_and_b32 v4, -2, v1 // (op 1852) 
v_sub_u32 v24, v1, v4 // (op 1852) 

v_lshlrev_b32 v4, 4, v24 // (op 1852) 

v_lshrrev_b32 v24, 1, v1 // (op 1852) 
v_lshl_add_u32 v25, v24, 5, v4 // (op 1852) 


s_nop 0
v_readfirstlane_b32 s15, v25 // (op 1852) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2189) 


v_mov_b64 v[24:25], v[12:13] // (op 982) reg expression

s_mov_b32 m0, s15 // (op 982) reg expression
buffer_load_dwordx4 v24, s[64:67], 0 offen offset: 0 lds // (op 982) Load value direct to lds


v_and_b32 v4, -2, v1 // (op 1787) 
v_sub_u32 v24, v1, v4 // (op 1787) 

v_lshlrev_b32 v4, 4, v24 // (op 1787) 

v_lshrrev_b32 v24, 1, v1 // (op 1787) 
v_lshl_add_u32 v25, v24, 5, v4 // (op 1787) 


s_nop 0
v_readfirstlane_b32 s15, v25 // (op 1787) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2170) 


v_mov_b64 v[24:25], v[16:17] // (op 987) reg expression
s_add_u32 s6, 4096, s15 // (op 987) 

s_mov_b32 m0, s6 // (op 987) reg expression
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 987) Load value direct to lds




v_mov_b64 v[24:25], s[32:33] // (op 841) convert

s_lshl_b32 s15, s32, 2 // (op 841) 
v_add_u32 v4, v21, s15 // (op 841) 

v_mov_b32 v26, v4 // (op 841) 
buffer_load_ubyte v29, v26, s[68:71], 0 offen offset: 0 // (op 841) Load value





v_mov_b64 v[24:25], s[38:39] // (op 846) convert

s_lshl_b32 s15, s38, 2 // (op 846) 
v_add_u32 v4, v20, s15 // (op 846) 

v_mov_b32 v21, v4 // (op 846) 
buffer_load_ubyte v26, v21, s[72:75], 0 offen offset: 0 // (op 846) Load value




v_and_b32 v4, -2, v1 // (op 1977) 
v_sub_u32 v20, v1, v4 // (op 1977) 

v_lshlrev_b32 v4, 4, v20 // (op 1977) 

v_lshrrev_b32 v20, 1, v1 // (op 1977) 
v_lshl_add_u32 v21, v20, 5, v4 // (op 1977) 


s_nop 0
v_readfirstlane_b32 s15, v21 // (op 1977) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2217) 


s_lshl_b64 s[18:19], s[28:29], 7 // (op 992) 
v_mov_b32 v4, s18 // (op 992) 
v_mov_b32 v20, s19 // (op 992) 
v_add_co_u32 v24, s[6:7], v12, v4 // (op 992) least significant half
v_addc_co_u32 v25, s[6:7], v13, v20, s[6:7] // (op 992) most significant half




s_add_u32 s6, 8192, s15 // (op 992) 

s_nop 0
s_mov_b32 m0, s6 // (op 992) reg expression
buffer_load_dwordx4 v24, s[64:67], 0 offen offset: 0 lds // (op 992) Load value direct to lds




v_and_b32 v4, -2, v1 // (op 1938) 
v_sub_u32 v12, v1, v4 // (op 1938) 

v_lshlrev_b32 v4, 4, v12 // (op 1938) 

v_lshrrev_b32 v12, 1, v1 // (op 1938) 
v_lshl_add_u32 v13, v12, 5, v4 // (op 1938) 


s_nop 0
v_readfirstlane_b32 s15, v13 // (op 1938) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2208) 


s_lshl_b64 s[18:19], s[24:25], 7 // (op 994) 
v_mov_b32 v4, s18 // (op 994) 
v_mov_b32 v12, s19 // (op 994) 
v_add_co_u32 v20, s[6:7], v16, v4 // (op 994) least significant half
v_addc_co_u32 v21, s[6:7], v17, v12, s[6:7] // (op 994) most significant half




s_add_u32 s6, 12288, s15 // (op 994) 

s_nop 0
s_mov_b32 m0, s6 // (op 994) reg expression
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 994) Load value direct to lds




s_waitcnt vmcnt(5)
s_barrier  // (op 2186) 



ds_read_b64_tr_b8 v[30:31], v10 // (op 871) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v10 offset:256 // (op 871) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v10 offset:2048 // (op 871) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v10 offset:2304 // (op 871) Transpose load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 2167) 



ds_read_b64_tr_b8 v[38:39], v0 offset:4096 // (op 868) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:4352 // (op 868) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:6144 // (op 868) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:6400 // (op 868) Transpose load local data 
v_and_b32 v4, -2, v1 // (op 1759) 
v_sub_u32 v12, v1, v4 // (op 1759) 

v_lshlrev_b32 v4, 4, v12 // (op 1759) 

v_lshrrev_b32 v12, 1, v1 // (op 1759) 
v_lshl_add_u32 v13, v12, 5, v4 // (op 1759) 


s_nop 0
v_readfirstlane_b32 s15, v13 // (op 1759) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v4, -2, v1 // (op 1824) 
v_sub_u32 v12, v1, v4 // (op 1824) 

v_lshlrev_b32 v4, 4, v12 // (op 1824) 

v_lshrrev_b32 v12, 1, v1 // (op 1824) 
v_lshl_add_u32 v13, v12, 5, v4 // (op 1824) 


s_nop 0
v_readfirstlane_b32 s6, v13 // (op 1824) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v4, -2, v1 // (op 1923) 
v_sub_u32 v12, v1, v4 // (op 1923) 

v_lshlrev_b32 v4, 4, v12 // (op 1923) 

v_lshrrev_b32 v12, 1, v1 // (op 1923) 
v_lshl_add_u32 v13, v12, 5, v4 // (op 1923) 


s_nop 0
v_readfirstlane_b32 s7, v13 // (op 1923) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v4, -2, v1 // (op 1962) 
v_sub_u32 v12, v1, v4 // (op 1962) 

v_lshlrev_b32 v4, 4, v12 // (op 1962) 

v_lshrrev_b32 v12, 1, v1 // (op 1962) 
v_lshl_add_u32 v13, v12, 5, v4 // (op 1962) 


s_nop 0
v_readfirstlane_b32 s8, v13 // (op 1962) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s1, 0 // (op 163) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 162) 
s_mov_b32 s42, s18 // (op 162) convert

s_and_b32 s18, s42, 4294967294 // (op 162) 

s_cmp_lt_u32 s1, s18 // (op 162) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_162 // (op 162) Condition: Top (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_162 if false)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_162:
s_add_u32 s1, s1, 2 // (op 164) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v27, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 175) 

v_mov_b64 v[12:13], s[32:33] // (op 172) convert

v_mov_b32 v4, v22 // (op 172) 
s_lshl_b32 s18, s32, 3 // (op 172) 
v_add_u32 v16, v4, s18 // (op 172) 

v_mov_b32 v17, v16 // (op 172) 
buffer_load_ubyte v27, v17, s[68:71], 0 offen offset: 0 // (op 172) Load value




v_mov_b64 v[12:13], s[38:39] // (op 174) convert

v_mov_b32 v16, v23 // (op 174) 
s_lshl_b32 s18, s38, 3 // (op 174) 
v_add_u32 v17, v16, s18 // (op 174) 

v_mov_b32 v20, v17 // (op 174) 
buffer_load_ubyte v28, v20, s[72:75], 0 offen offset: 0 // (op 174) Load value



s_barrier  // (op 2192) 


v_mov_b64 v[12:13], v[14:15] // (op 919) 
s_lshl_b64 s[18:19], s[28:29], 7 // (op 919) 
s_lshl_b64 s[42:43], s[18:19], 1 // (op 919) 

v_mov_b32 v17, s42 // (op 919) 
v_mov_b32 v20, s43 // (op 919) 
v_add_co_u32 v24, s[18:19], v12, v17 // (op 919) least significant half
v_addc_co_u32 v25, s[18:19], v13, v20, s[18:19] // (op 919) most significant half





s_mov_b32 m0, s6 // (op 919) reg expression
buffer_load_dwordx4 v24, s[64:67], 0 offen offset: 0 lds // (op 919) Load value direct to lds

s_barrier  // (op 2173) 


v_mov_b64 v[20:21], v[18:19] // (op 914) 
s_lshl_b64 s[18:19], s[24:25], 7 // (op 914) 
s_lshl_b64 s[42:43], s[18:19], 1 // (op 914) 

v_mov_b32 v17, s42 // (op 914) 
v_mov_b32 v24, s43 // (op 914) 
v_add_co_u32 v46, s[18:19], v20, v17 // (op 914) least significant half
v_addc_co_u32 v47, s[18:19], v21, v24, s[18:19] // (op 914) most significant half




s_add_u32 s18, 4096, s15 // (op 914) 

s_nop 0
s_mov_b32 m0, s18 // (op 914) reg expression
buffer_load_dwordx4 v46, s[60:63], 0 offen offset: 0 lds // (op 914) Load value direct to lds


s_waitcnt vmcnt(5)
s_barrier  // (op 2214) 



ds_read_b64_tr_b8 v[30:31], v8 offset:8192 // (op 478) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v8 offset:8448 // (op 478) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v8 offset:10240 // (op 478) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v8 offset:10496 // (op 478) Transpose load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 2205) 



ds_read_b64_tr_b8 v[38:39], v5 offset:12288 // (op 476) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v5 offset:12544 // (op 476) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v5 offset:14336 // (op 476) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v5 offset:14592 // (op 476) Transpose load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v29, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 483) 

v_mov_b64 v[24:25], s[32:33] // (op 480) convert

s_lshl_b32 s18, s32, 2 // (op 480) 
s_mul_i32 s19, 3, s18 // (op 480) 

v_add_u32 v17, v4, s19 // (op 480) 

v_mov_b32 v46, v17 // (op 480) 
buffer_load_ubyte v29, v46, s[68:71], 0 offen offset: 0 // (op 480) Load value





s_lshl_b32 s18, s32, 3 // (op 1369) 
v_add_u32 v22, v22, s18 // (op 1369) 

v_mov_b64 v[24:25], s[38:39] // (op 482) convert

s_lshl_b32 s18, s38, 2 // (op 482) 
s_mul_i32 s19, 3, s18 // (op 482) 

v_add_u32 v4, v16, s19 // (op 482) 

v_mov_b32 v17, v4 // (op 482) 
buffer_load_ubyte v26, v17, s[72:75], 0 offen offset: 0 // (op 482) Load value




s_barrier  // (op 2220) 
s_lshl_b32 s18, s38, 3 // (op 1497) 


v_add_u32 v23, v23, s18 // (op 1497) 

s_lshl_b64 s[18:19], s[28:29], 7 // (op 941) 
s_mul_i32 s43, 3, s19 // (op 941) most significant: low of low * high
s_mul_i32 s42, 3, s18 // (op 941) least significant
s_mul_hi_u32 s44, 3, s18 // (op 941) most significant: high of low * low
s_add_u32 s43, s43, s44 // (op 941) most significant: sum


v_mov_b32 v4, s42 // (op 941) 
v_mov_b32 v16, s43 // (op 941) 
v_add_co_u32 v24, s[18:19], v12, v4 // (op 941) least significant half
v_addc_co_u32 v25, s[18:19], v13, v16, s[18:19] // (op 941) most significant half




s_add_u32 s18, 8192, s8 // (op 941) 

s_nop 0
s_mov_b32 m0, s18 // (op 941) reg expression
buffer_load_dwordx4 v24, s[64:67], 0 offen offset: 0 lds // (op 941) Load value direct to lds



s_barrier  // (op 2211) 
s_lshl_b64 s[18:19], s[28:29], 7 // (op 1241) 


s_lshl_b64 s[42:43], s[18:19], 1 // (op 1241) 

v_mov_b32 v4, s42 // (op 1241) 
v_mov_b32 v12, s43 // (op 1241) 
s_lshl_b64 s[18:19], s[24:25], 7 // (op 936) 
s_mul_i32 s45, 3, s19 // (op 936) most significant: low of low * high
s_mul_i32 s44, 3, s18 // (op 936) least significant
s_mul_hi_u32 s52, 3, s18 // (op 936) most significant: high of low * low
s_add_u32 s45, s45, s52 // (op 936) most significant: sum


v_mov_b32 v13, s44 // (op 936) 
v_mov_b32 v16, s45 // (op 936) 
v_add_co_u32 v14, s[18:19], v14, v4 // (op 1241) least significant half
v_addc_co_u32 v15, s[18:19], v15, v12, s[18:19] // (op 1241) most significant half




v_add_co_u32 v24, s[18:19], v20, v13 // (op 936) least significant half
v_addc_co_u32 v25, s[18:19], v21, v16, s[18:19] // (op 936) most significant half




s_add_u32 s18, 12288, s7 // (op 936) 

s_nop 0
s_mov_b32 m0, s18 // (op 936) reg expression
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 936) Load value direct to lds



s_lshl_b64 s[18:19], s[24:25], 7 // (op 1113) 
s_lshl_b64 s[42:43], s[18:19], 1 // (op 1113) 

v_mov_b32 v4, s42 // (op 1113) 
v_mov_b32 v12, s43 // (op 1113) 
v_add_co_u32 v18, s[18:19], v18, v4 // (op 1113) least significant half
v_addc_co_u32 v19, s[18:19], v19, v12, s[18:19] // (op 1113) most significant half




s_waitcnt vmcnt(5)
s_barrier  // (op 2198) 



ds_read_b64_tr_b8 v[30:31], v6 // (op 170) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v6 offset:256 // (op 170) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v6 offset:2048 // (op 170) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v6 offset:2304 // (op 170) Transpose load local data 
s_waitcnt vmcnt(4)
s_barrier  // (op 2179) 



ds_read_b64_tr_b8 v[38:39], v3 offset:4096 // (op 168) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v3 offset:4352 // (op 168) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v3 offset:6144 // (op 168) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v3 offset:6400 // (op 168) Transpose load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 162) 
s_mov_b32 s42, s18 // (op 162) convert

s_and_b32 s18, s42, 4294967294 // (op 162) 

s_cmp_lt_u32 s1, s18 // (op 162) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_162 // (op 162) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_162 if true)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_162:










s_barrier  // (op 2257) 
v_and_b32 v4, -2, v1 // (op 1007) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1007) 
v_sub_u32 v12, v1, v4 // (op 1007) 

s_and_b32 s15, s18, 4294967294 // (op 1007) 
v_lshlrev_b32 v4, 4, v12 // (op 1007) 

s_add_u32 s6, s0, s15 // (op 1007) 

v_lshrrev_b32 v12, 1, v1 // (op 1007) 
s_mul_i32 s15, s16, s50 // (op 1007) 
s_add_u32 s15, s15, s6 // (op 1007) 

s_add_u32 s6, s15, s9 // (op 1007) 

s_mul_hi_u32 s15, s6, s13 // (op 1007) 
s_sub_u32 s7, s6, s15 // (op 1007) 
s_lshr_b32 s8, s7, 1 // (op 1007) 

s_add_u32 s7, s8, s15 // (op 1007) 


s_lshr_b32 s8, s7, s11 // (op 1007) 

s_and_b32 s7, s11, 2147483648 // (op 1007) 
s_cmp_eq_i32 s7, 0 // (op 1007) 
s_mov_b32 s15, scc // (op 1007) 

s_cmp_eq_i32 s15, 0 // (op 1007) 
s_cselect_b32 s7, s6, s8 // (op 1007) 


s_mul_hi_u32 s8, s7, s12 // (op 1007) 
s_sub_u32 s15, s7, s8 // (op 1007) 
s_lshr_b32 s42, s15, 1 // (op 1007) 

s_add_u32 s15, s42, s8 // (op 1007) 


s_lshr_b32 s8, s15, s10 // (op 1007) 

s_and_b32 s15, s10, 2147483648 // (op 1007) 
s_cmp_eq_i32 s15, 0 // (op 1007) 
s_mov_b32 s42, scc // (op 1007) 

s_cmp_eq_i32 s42, 0 // (op 1007) 
s_cselect_b32 s15, s7, s8 // (op 1007) 


s_mul_i32 s8, s7, s18 // (op 1007) 


v_lshl_add_u32 v13, s15, 5, v4 // (op 1007) 


s_sub_u32 s7, s6, s8 // (op 1007) 


v_mov_b32 v15, 0 // (op 1007) convert
v_mov_b32 v14, v13 // (op 1007) convert

v_lshl_add_u32 v4, s7, 7, v12 // (op 1007) 


v_mul_lo_u32 v13, v4, s25 // (op 1007) most significant: low of low * high
v_mul_hi_u32 v16, v4, s24 // (op 1007) most significant: high of low * low
v_mul_lo_u32 v12, v4, s24 // (op 1007) least significant: low of low * low
v_add_u32 v13, v13, v16 // (op 1007) most significant: sum


v_add_co_u32 v16, s[18:19], v14, v12 // (op 1007) least significant half
v_addc_co_u32 v17, s[18:19], v15, v13, s[18:19] // (op 1007) most significant half



s_mul_i32 s19, s24, s3 // (op 1009) most significant: low of low * high
s_mul_i32 s15, s25, s2 // (op 1009) most significant: low of high * low
s_mul_i32 s18, s24, s2 // (op 1009) least significant
s_mul_hi_u32 s6, s24, s2 // (op 1009) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1009) most significant: sum
s_add_u32 s19, s19, s6 // (op 1009) most significant: sum


s_mov_b32 s60, s26 // (op 1009) reg expression
s_mov_b32 s61, s27 // (op 1009) reg expression
s_mov_b32 s62, s18 // (op 1009) reg expression
s_mov_b32 s63, 131072 // (op 1009) call()

v_and_b32 v4, -2, v1 // (op 1143) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1143) 
v_sub_u32 v12, v1, v4 // (op 1143) 

s_and_b32 s15, s18, 4294967294 // (op 1143) 
v_lshlrev_b32 v4, 4, v12 // (op 1143) 

s_add_u32 s6, s0, s15 // (op 1143) 

v_lshrrev_b32 v12, 1, v1 // (op 1143) 
s_mul_i32 s15, s16, s50 // (op 1143) 
s_add_u32 s15, s15, s6 // (op 1143) 

s_add_u32 s6, s15, s9 // (op 1143) 

s_mul_hi_u32 s15, s6, s13 // (op 1143) 
s_sub_u32 s7, s6, s15 // (op 1143) 
s_lshr_b32 s8, s7, 1 // (op 1143) 

s_add_u32 s7, s8, s15 // (op 1143) 


s_lshr_b32 s8, s7, s11 // (op 1143) 

s_and_b32 s7, s11, 2147483648 // (op 1143) 
s_cmp_eq_i32 s7, 0 // (op 1143) 
s_mov_b32 s15, scc // (op 1143) 

s_cmp_eq_i32 s15, 0 // (op 1143) 
s_cselect_b32 s7, s6, s8 // (op 1143) 


s_mul_hi_u32 s8, s7, s12 // (op 1143) 
s_sub_u32 s15, s7, s8 // (op 1143) 
s_lshr_b32 s42, s15, 1 // (op 1143) 

s_add_u32 s15, s42, s8 // (op 1143) 


s_lshr_b32 s8, s15, s10 // (op 1143) 

s_and_b32 s15, s10, 2147483648 // (op 1143) 
s_cmp_eq_i32 s15, 0 // (op 1143) 
s_mov_b32 s42, scc // (op 1143) 

s_cmp_eq_i32 s42, 0 // (op 1143) 
s_cselect_b32 s15, s7, s8 // (op 1143) 


s_mul_i32 s8, s15, s14 // (op 1143) 

s_sub_u32 s15, s7, s8 // (op 1143) 

s_mul_i32 s8, s7, s18 // (op 1143) 


v_lshl_add_u32 v13, s15, 5, v4 // (op 1143) 


s_sub_u32 s7, s6, s8 // (op 1143) 


v_mov_b32 v15, 0 // (op 1143) convert
v_mov_b32 v14, v13 // (op 1143) convert

v_lshl_add_u32 v4, s7, 7, v12 // (op 1143) 


v_mul_lo_u32 v13, v4, s29 // (op 1143) most significant: low of low * high
v_mul_hi_u32 v18, v4, s28 // (op 1143) most significant: high of low * low
v_mul_lo_u32 v12, v4, s28 // (op 1143) least significant: low of low * low
v_add_u32 v13, v13, v18 // (op 1143) most significant: sum

v_add_co_u32 v12, s[18:19], v12, v14 // (op 1143) least significant half
v_addc_co_u32 v13, s[18:19], v13, v15, s[18:19] // (op 1143) most significant half



s_mul_i32 s19, s28, s3 // (op 1145) most significant: low of low * high
s_mul_i32 s15, s29, s2 // (op 1145) most significant: low of high * low
s_mul_i32 s18, s28, s2 // (op 1145) least significant
s_mul_hi_u32 s6, s28, s2 // (op 1145) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1145) most significant: sum
s_add_u32 s19, s19, s6 // (op 1145) most significant: sum


s_mov_b32 s64, s30 // (op 1145) reg expression
s_mov_b32 s65, s31 // (op 1145) reg expression
s_mov_b32 s66, s18 // (op 1145) reg expression
s_mov_b32 s67, 131072 // (op 1145) call()

v_lshrrev_b32 v4, 7, v1 // (op 1271) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1271) 
v_and_b32 v14, 15, v1 // (op 1271) 
s_and_b32 s15, s18, 4294967294 // (op 1271) 
v_lshl_add_u32 v15, v4, 4, v14 // (op 1271) 


s_add_u32 s6, s0, s15 // (op 1271) 

v_and_b32 v4, 63, v1 // (op 1271) 
v_lshrrev_b32 v14, 4, v4 // (op 1271) 

s_mul_i32 s15, s16, s50 // (op 1271) 
s_add_u32 s15, s15, s6 // (op 1271) 

s_add_u32 s6, s15, s9 // (op 1271) 

s_mul_hi_u32 s15, s6, s13 // (op 1271) 
s_sub_u32 s7, s6, s15 // (op 1271) 
s_lshr_b32 s8, s7, 1 // (op 1271) 

s_add_u32 s7, s8, s15 // (op 1271) 


s_lshr_b32 s8, s7, s11 // (op 1271) 

s_and_b32 s7, s11, 2147483648 // (op 1271) 
s_cmp_eq_i32 s7, 0 // (op 1271) 
s_mov_b32 s15, scc // (op 1271) 

s_cmp_eq_i32 s15, 0 // (op 1271) 
s_cselect_b32 s7, s6, s8 // (op 1271) 


s_mul_hi_u32 s8, s7, s12 // (op 1271) 
s_sub_u32 s15, s7, s8 // (op 1271) 
s_lshr_b32 s42, s15, 1 // (op 1271) 

s_add_u32 s15, s42, s8 // (op 1271) 


s_lshr_b32 s8, s15, s10 // (op 1271) 

s_and_b32 s15, s10, 2147483648 // (op 1271) 
s_cmp_eq_i32 s15, 0 // (op 1271) 
s_mov_b32 s42, scc // (op 1271) 

s_cmp_eq_i32 s42, 0 // (op 1271) 
s_cselect_b32 s15, s7, s8 // (op 1271) 


s_mul_i32 s8, s7, s18 // (op 1271) 


v_lshl_add_u32 v4, s15, 5, v15 // (op 1271) 


s_sub_u32 s7, s6, s8 // (op 1271) 


v_lshl_add_u32 v15, s7, 2, v14 // (op 1271) 


v_mul_lo_u32 v14, v15, s32 // (op 1271) 

v_add_u32 v15, v4, v14 // (op 1271) 


s_mul_i32 s19, s32, s37 // (op 1273) most significant: low of low * high
s_mul_i32 s15, s33, s36 // (op 1273) most significant: low of high * low
s_mul_i32 s18, s32, s36 // (op 1273) least significant
s_mul_hi_u32 s6, s32, s36 // (op 1273) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1273) most significant: sum
s_add_u32 s19, s19, s6 // (op 1273) most significant: sum


s_mov_b32 s68, s34 // (op 1273) reg expression
s_mov_b32 s69, s35 // (op 1273) reg expression
s_mov_b32 s70, s18 // (op 1273) reg expression
s_mov_b32 s71, 131072 // (op 1273) call()

v_lshrrev_b32 v4, 6, v1 // (op 1399) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1399) 
v_and_b32 v14, 1, v4 // (op 1399) 

s_and_b32 s15, s18, 4294967294 // (op 1399) 
v_and_b32 v4, 15, v1 // (op 1399) 
s_add_u32 s6, s0, s15 // (op 1399) 

v_lshl_add_u32 v18, v14, 4, v4 // (op 1399) 


v_and_b32 v4, 63, v1 // (op 1399) 
s_mul_i32 s15, s16, s50 // (op 1399) 
s_add_u32 s15, s15, s6 // (op 1399) 

v_lshrrev_b32 v14, 4, v4 // (op 1399) 

s_add_u32 s6, s15, s9 // (op 1399) 

s_mul_hi_u32 s15, s6, s13 // (op 1399) 
s_sub_u32 s7, s6, s15 // (op 1399) 
s_lshr_b32 s8, s7, 1 // (op 1399) 

s_add_u32 s7, s8, s15 // (op 1399) 


s_lshr_b32 s8, s7, s11 // (op 1399) 

s_and_b32 s7, s11, 2147483648 // (op 1399) 
s_cmp_eq_i32 s7, 0 // (op 1399) 
s_mov_b32 s15, scc // (op 1399) 

s_cmp_eq_i32 s15, 0 // (op 1399) 
s_cselect_b32 s7, s6, s8 // (op 1399) 


s_mul_hi_u32 s8, s7, s12 // (op 1399) 
s_sub_u32 s15, s7, s8 // (op 1399) 
s_lshr_b32 s42, s15, 1 // (op 1399) 

s_add_u32 s15, s42, s8 // (op 1399) 


s_lshr_b32 s8, s15, s10 // (op 1399) 

s_and_b32 s15, s10, 2147483648 // (op 1399) 
s_cmp_eq_i32 s15, 0 // (op 1399) 
s_mov_b32 s42, scc // (op 1399) 

s_cmp_eq_i32 s42, 0 // (op 1399) 
s_cselect_b32 s15, s7, s8 // (op 1399) 


s_mul_i32 s8, s15, s14 // (op 1399) 

s_sub_u32 s15, s7, s8 // (op 1399) 

s_mul_i32 s8, s7, s18 // (op 1399) 


v_lshl_add_u32 v4, s15, 5, v18 // (op 1399) 


s_sub_u32 s7, s6, s8 // (op 1399) 


v_lshl_add_u32 v18, s7, 2, v14 // (op 1399) 


v_mul_lo_u32 v14, v18, s38 // (op 1399) 
v_add_u32 v14, v14, v4 // (op 1399) 


s_mul_i32 s19, s38, s37 // (op 1401) most significant: low of low * high
s_mul_i32 s15, s39, s36 // (op 1401) most significant: low of high * low
s_mul_i32 s18, s38, s36 // (op 1401) least significant
s_mul_hi_u32 s6, s38, s36 // (op 1401) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1401) most significant: sum
s_add_u32 s19, s19, s6 // (op 1401) most significant: sum


s_mov_b32 s72, s40 // (op 1401) reg expression
s_mov_b32 s73, s41 // (op 1401) reg expression
s_mov_b32 s74, s18 // (op 1401) reg expression
s_mov_b32 s75, 131072 // (op 1401) call()

v_and_b32 v4, -2, v1 // (op 1772) 
v_sub_u32 v18, v1, v4 // (op 1772) 

v_lshlrev_b32 v4, 4, v18 // (op 1772) 

v_lshrrev_b32 v18, 1, v1 // (op 1772) 
v_lshl_add_u32 v19, v18, 5, v4 // (op 1772) 


s_nop 0
v_readfirstlane_b32 s15, v19 // (op 1772) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v4, -2, v1 // (op 1837) 
v_sub_u32 v18, v1, v4 // (op 1837) 

v_lshlrev_b32 v4, 4, v18 // (op 1837) 

v_lshrrev_b32 v18, 1, v1 // (op 1837) 
v_lshl_add_u32 v19, v18, 5, v4 // (op 1837) 


s_nop 0
v_readfirstlane_b32 s6, v19 // (op 1837) Wait state hazard: VALU Write Readlane Hazard

s_ashr_i64 s[18:19], s[2:3], 7 // (op 419) 
s_and_b32 s7, s18, 4294967294 // (op 419) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 418) 
s_mov_b32 s8, s18 // (op 418) convert

s_cmp_lt_u32 s7, s8 // (op 418) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_418 // (op 418) Condition: Top (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_418 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_418:
s_barrier  // (op 2176) 
s_add_u32 s7, s7, 1 // (op 420) 


v_mov_b64 v[18:19], v[16:17] // (op 924) 
v_mov_b64 v[20:21], v[18:19] // (op 924) reg expression
s_add_u32 s8, 4096, s15 // (op 924) 

s_mov_b32 m0, s8 // (op 924) reg expression
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 924) Load value direct to lds



s_lshl_b64 s[18:19], s[24:25], 7 // (op 997) 
v_mov_b32 v4, s18 // (op 997) 
v_mov_b32 v18, s19 // (op 997) 
v_add_co_u32 v16, s[42:43], v16, v4 // (op 997) least significant half
v_addc_co_u32 v17, s[42:43], v17, v18, s[42:43] // (op 997) most significant half




s_barrier  // (op 2195) 


v_mov_b64 v[18:19], v[12:13] // (op 930) 
v_mov_b64 v[20:21], v[18:19] // (op 930) reg expression

s_mov_b32 m0, s6 // (op 930) reg expression
buffer_load_dwordx4 v20, s[64:67], 0 offen offset: 0 lds // (op 930) Load value direct to lds


s_lshl_b64 s[18:19], s[28:29], 7 // (op 1135) 
v_mov_b32 v4, s18 // (op 1135) 
v_mov_b32 v18, s19 // (op 1135) 
v_add_co_u32 v12, s[42:43], v12, v4 // (op 1135) least significant half
v_addc_co_u32 v13, s[42:43], v13, v18, s[42:43] // (op 1135) most significant half




s_waitcnt vmcnt(1)
s_barrier  // (op 2182) 



ds_read_b64_tr_b8 v[18:19], v3 offset:4096 // (op 426) Transpose load local data 
ds_read_b64_tr_b8 v[20:21], v3 offset:4352 // (op 426) Transpose load local data 
ds_read_b64_tr_b8 v[22:23], v3 offset:6144 // (op 426) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v3 offset:6400 // (op 426) Transpose load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 2201) 



ds_read_b64_tr_b8 v[26:27], v6 // (op 428) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v6 offset:256 // (op 428) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v6 offset:2048 // (op 428) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v6 offset:2304 // (op 428) Transpose load local data 

v_mov_b64 v[34:35], s[32:33] // (op 430) convert

v_mov_b32 v4, v15 // (op 430) 
v_mov_b32 v36, v4 // (op 430) reg expression
v_mov_b32 v37, v36 // (op 430) 
buffer_load_ubyte v38, v37, s[68:71], 0 offen offset: 0 // (op 430) Load value




s_lshl_b32 s8, s32, 2 // (op 1263) 
v_add_u32 v15, v15, s8 // (op 1263) 


v_mov_b64 v[34:35], s[38:39] // (op 432) convert

v_mov_b32 v4, v14 // (op 432) 
v_mov_b32 v36, v4 // (op 432) reg expression
v_mov_b32 v37, v36 // (op 432) 
buffer_load_ubyte v39, v37, s[72:75], 0 offen offset: 0 // (op 432) Load value




s_lshl_b32 s8, s38, 2 // (op 1391) 
v_add_u32 v14, v14, s8 // (op 1391) 

s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:25], v[26:33], a[0:3], v38, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 433) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 418) 
s_mov_b32 s8, s18 // (op 418) convert

s_cmp_lt_u32 s7, s8 // (op 418) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_418 // (op 418) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_418 if true)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_418:







s_barrier  // (op 2311) 


v_cmp_eq_f32 vcc, 0.00000, v9 // (op 203) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_203 // (op 203) Condition: False, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_203
s_nop 0
v_accvgpr_read v12, a0 // (op 211) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 211) 
v_accvgpr_read v14, a2 // (op 211) 
v_accvgpr_read v15, a3 // (op 211) 
v_mul_f32 v16, v11, v12 // (op 211) 
v_mul_f32 v17, v11, v13 // (op 211) 
v_mul_f32 v18, v11, v14 // (op 211) 
v_mul_f32 v19, v11, v15 // (op 211) 

v_lshrrev_b32 v0, 7, v1 // (op 1725) 
s_mul_i32 s15, s16, s50 // (op 1725) 
s_add_u32 s15, s15, s0 // (op 1725) 
v_and_b32 v4, 63, v1 // (op 1725) 
s_add_u32 s6, s15, s9 // (op 1725) 

v_lshrrev_b32 v10, 4, v4 // (op 1725) 
s_mul_hi_u32 s15, s6, s13 // (op 1725) 
v_lshlrev_b32 v12, 2, v10 // (op 1725) 

s_sub_u32 s7, s6, s15 // (op 1725) 
v_lshl_add_u32 v10, v0, 4, v12 // (op 1725) 


s_lshr_b32 s8, s7, 1 // (op 1725) 

v_lshrrev_b32 v0, 2, v4 // (op 1725) 

s_add_u32 s7, s8, s15 // (op 1725) 


v_and_b32 v4, 3, v0 // (op 1725) 

s_lshr_b32 s8, s7, s11 // (op 1725) 

v_and_b32 v0, 3, v1 // (op 1725) 
s_and_b32 s7, s11, 2147483648 // (op 1725) 
v_lshl_add_u32 v12, v4, 2, v0 // (op 1725) 


s_cmp_eq_i32 s7, 0 // (op 1725) 
s_mov_b32 s15, scc // (op 1725) 

v_lshrrev_b32 v0, 6, v1 // (op 1725) 
s_cmp_eq_i32 s15, 0 // (op 1725) 
s_cselect_b32 s7, s6, s8 // (op 1725) 



v_and_b32 v4, 1, v0 // (op 1725) 

s_mul_hi_u32 s6, s7, s12 // (op 1725) 
v_lshl_add_u32 v0, v4, 4, v12 // (op 1725) 


s_sub_u32 s8, s7, s6 // (op 1725) 
s_lshr_b32 s15, s8, 1 // (op 1725) 

s_add_u32 s8, s15, s6 // (op 1725) 


s_lshr_b32 s6, s8, s10 // (op 1725) 

s_and_b32 s8, s10, 2147483648 // (op 1725) 
s_cmp_eq_i32 s8, 0 // (op 1725) 
s_mov_b32 s15, scc // (op 1725) 

s_cmp_eq_i32 s15, 0 // (op 1725) 
s_cselect_b32 s8, s7, s6 // (op 1725) 


s_mul_i32 s6, s8, s14 // (op 1725) 
v_lshl_add_u32 v4, s8, 5, v10 // (op 1725) 


s_sub_u32 s8, s7, s6 // (op 1725) 


v_lshl_add_u32 v10, s8, 5, v0 // (op 1725) 


v_mul_lo_u32 v0, v10, s46 // (op 1725) 

v_add_lshl_u32 v10, v4, v0, 2 // (op 1725) 


s_mul_i32 s19, s46, s23 // (op 1727) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1727) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1727) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1727) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1727) most significant: sum
s_add_u32 s19, s19, s6 // (op 1727) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1727) 

s_mov_b32 s76, s48 // (op 1727) reg expression
s_mov_b32 s77, s49 // (op 1727) reg expression
s_mov_b32 s78, s6 // (op 1727) reg expression
s_mov_b32 s79, 131072 // (op 1727) call()




buffer_store_dwordx4 v[16:19], v10, s[76:79], 0 offen // (op 213) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_203 // (op 203) Condition: Done, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_203
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_203:



buffer_load_dwordx4 v[12:15], v7, s[56:59], 0 offen // (op 226) Load value
v_accvgpr_read v16, a0 // (op 223) 
v_accvgpr_read v17, a1 // (op 223) 
v_accvgpr_read v18, a2 // (op 223) 
v_accvgpr_read v19, a3 // (op 223) 
v_mul_f32 v20, v11, v16 // (op 223) 
v_mul_f32 v21, v11, v17 // (op 223) 
v_mul_f32 v22, v11, v18 // (op 223) 
v_mul_f32 v23, v11, v19 // (op 223) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v9, v12 // (op 227) 
v_mul_f32 v17, v9, v13 // (op 227) 
v_mul_f32 v18, v9, v14 // (op 227) 
v_mul_f32 v19, v9, v15 // (op 227) 

v_add_f32 v12, v16, v20 // (op 224) 
v_add_f32 v13, v17, v21 // (op 224) 
v_add_f32 v14, v18, v22 // (op 224) 
v_add_f32 v15, v19, v23 // (op 224) 


v_lshrrev_b32 v0, 7, v1 // (op 1742) 
s_mul_i32 s15, s16, s50 // (op 1742) 
s_add_u32 s15, s15, s0 // (op 1742) 
v_and_b32 v4, 63, v1 // (op 1742) 
s_add_u32 s6, s15, s9 // (op 1742) 

v_lshrrev_b32 v10, 4, v4 // (op 1742) 
s_mul_hi_u32 s15, s6, s13 // (op 1742) 
v_lshlrev_b32 v16, 2, v10 // (op 1742) 

s_sub_u32 s7, s6, s15 // (op 1742) 
v_lshl_add_u32 v10, v0, 4, v16 // (op 1742) 


s_lshr_b32 s8, s7, 1 // (op 1742) 

v_lshrrev_b32 v0, 2, v4 // (op 1742) 

s_add_u32 s7, s8, s15 // (op 1742) 


v_and_b32 v4, 3, v0 // (op 1742) 

s_lshr_b32 s8, s7, s11 // (op 1742) 

v_and_b32 v0, 3, v1 // (op 1742) 
s_and_b32 s7, s11, 2147483648 // (op 1742) 
v_lshl_add_u32 v16, v4, 2, v0 // (op 1742) 


s_cmp_eq_i32 s7, 0 // (op 1742) 
s_mov_b32 s15, scc // (op 1742) 

v_lshrrev_b32 v0, 6, v1 // (op 1742) 
s_cmp_eq_i32 s15, 0 // (op 1742) 
s_cselect_b32 s7, s6, s8 // (op 1742) 



v_and_b32 v4, 1, v0 // (op 1742) 

s_mul_hi_u32 s6, s7, s12 // (op 1742) 
v_lshl_add_u32 v0, v4, 4, v16 // (op 1742) 


s_sub_u32 s8, s7, s6 // (op 1742) 
s_lshr_b32 s15, s8, 1 // (op 1742) 

s_add_u32 s8, s15, s6 // (op 1742) 


s_lshr_b32 s6, s8, s10 // (op 1742) 

s_and_b32 s8, s10, 2147483648 // (op 1742) 
s_cmp_eq_i32 s8, 0 // (op 1742) 
s_mov_b32 s15, scc // (op 1742) 

s_cmp_eq_i32 s15, 0 // (op 1742) 
s_cselect_b32 s8, s7, s6 // (op 1742) 


s_mul_i32 s6, s8, s14 // (op 1742) 
v_lshl_add_u32 v4, s8, 5, v10 // (op 1742) 


s_sub_u32 s8, s7, s6 // (op 1742) 


v_lshl_add_u32 v10, s8, 5, v0 // (op 1742) 


v_mul_lo_u32 v0, v10, s46 // (op 1742) 

v_add_lshl_u32 v10, v4, v0, 2 // (op 1742) 


s_mul_i32 s19, s46, s23 // (op 1744) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1744) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1744) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1744) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1744) most significant: sum
s_add_u32 s19, s19, s6 // (op 1744) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1744) 

s_mov_b32 s76, s48 // (op 1744) reg expression
s_mov_b32 s77, s49 // (op 1744) reg expression
s_mov_b32 s78, s6 // (op 1744) reg expression
s_mov_b32 s79, 131072 // (op 1744) call()




buffer_store_dwordx4 v[12:15], v10, s[76:79], 0 offen // (op 225) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_203:

s_add_u32 s0, s0, s1 // (op 412) 
s_mul_i32 s15, s50, s16 // (op 410) 
s_cmp_gt_u32 s51, s15 // (op 410) 
s_mov_b32 s6, scc // (op 410) 
s_sub_u32 s7, s51, s15 // (op 410) 

s_cmp_eq_i32 s6, 0 // (op 410) 
s_cselect_b32 s8, 0, s7 // (op 410) 


s_cmp_lt_u32 s50, s8 // (op 410) 
s_mov_b32 s15, scc // (op 410) 
s_cmp_eq_i32 s15, 0 // (op 410) 
s_cselect_b32 s6, s8, s50 // (op 410) 


s_cmp_lt_u32 s0, s6 // (op 410) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_DPStreamTileLoop_410 // (op 410) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_DPStreamTileLoop_410 if true)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_410:






































s_endpgm  // End of RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
.LRR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_end:
.size RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2, .LRR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_end-RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
  .amdhsa_next_free_vgpr 56
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16384 // lds bytes
  .amdhsa_accum_offset 52
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
    DataTypeB: F8
    DestDataType: s
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: False
    TransposeB: True
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
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
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
  - .name: RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
    .symbol: RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 83
    .vgpr_count: 52
    .agpr_count: 4
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_size_1_9
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
      - .name: Tensor_2_size_1_19
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_0_stride_1_8
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_0_11
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_13
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 112
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_14
        .size: 8
        .offset: 120
        .value_kind: by_value
      - .name: Tensor_7_stride_0_16
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

