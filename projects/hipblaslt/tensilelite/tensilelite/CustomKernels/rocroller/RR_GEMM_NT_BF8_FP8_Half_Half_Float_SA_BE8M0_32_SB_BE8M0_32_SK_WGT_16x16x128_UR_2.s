// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2
.p2align 8
.type RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2,@function
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 1736) 
v_and_b32 v3, 63, v1 // (op 1736) 
v_sub_u32 v4, v3, v0 // (op 1736) 

v_lshrrev_b32 v3, 1, v4 // (op 1736) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 1736) 


v_and_b32 v0, 1, v4 // (op 1736) 

v_lshlrev_b32 v3, 3, v0 // (op 1736) 

v_lshrrev_b32 v0, 6, v1 // (op 1736) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1736) 


v_add_u32 v0, v5, v4 // (op 1736) 


v_and_b32 v3, 48, v1 // (op 1801) 
v_and_b32 v4, 63, v1 // (op 1801) 
v_sub_u32 v5, v4, v3 // (op 1801) 

v_lshrrev_b32 v4, 1, v5 // (op 1801) 
v_add_lshl_u32 v6, v3, v4, 4 // (op 1801) 


v_and_b32 v3, 1, v5 // (op 1801) 

v_lshlrev_b32 v4, 3, v3 // (op 1801) 

v_add_u32 v3, v6, v4 // (op 1801) 


v_lshrrev_b32 v4, 6, v1 // (op 1867) 
v_and_b32 v5, 15, v1 // (op 1867) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 1867) 


v_and_b32 v4, 48, v1 // (op 1867) 
v_add_u32 v5, v4, v6 // (op 1867) 


v_and_b32 v4, 48, v1 // (op 1933) 
v_and_b32 v6, 15, v1 // (op 1933) 
v_add_u32 v7, v4, v6 // (op 1933) 


v_and_b32 v4, 48, v1 // (op 2068) 
v_and_b32 v6, 63, v1 // (op 2068) 
v_sub_u32 v8, v6, v4 // (op 2068) 

v_lshrrev_b32 v6, 1, v8 // (op 2068) 
v_add_lshl_u32 v9, v4, v6, 4 // (op 2068) 


v_and_b32 v4, 1, v8 // (op 2068) 

v_lshlrev_b32 v6, 3, v4 // (op 2068) 

v_lshrrev_b32 v4, 6, v1 // (op 2068) 
v_lshl_add_u32 v8, v4, 4, v6 // (op 2068) 


v_add_u32 v4, v9, v8 // (op 2068) 


v_and_b32 v6, 48, v1 // (op 2133) 
v_and_b32 v8, 63, v1 // (op 2133) 
v_sub_u32 v9, v8, v6 // (op 2133) 

v_lshrrev_b32 v8, 1, v9 // (op 2133) 
v_add_lshl_u32 v10, v6, v8, 4 // (op 2133) 


v_and_b32 v6, 1, v9 // (op 2133) 

v_lshlrev_b32 v8, 3, v6 // (op 2133) 

v_add_u32 v6, v10, v8 // (op 2133) 


v_lshrrev_b32 v8, 6, v1 // (op 2199) 
v_and_b32 v9, 15, v1 // (op 2199) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 2199) 


v_and_b32 v8, 48, v1 // (op 2199) 
v_add_u32 v9, v8, v10 // (op 2199) 


v_and_b32 v8, 48, v1 // (op 2265) 
v_and_b32 v10, 15, v1 // (op 2265) 
v_add_u32 v11, v8, v10 // (op 2265) 


v_and_b32 v8, 48, v1 // (op 2351) 
v_and_b32 v10, 63, v1 // (op 2351) 
v_sub_u32 v12, v10, v8 // (op 2351) 

v_lshrrev_b32 v10, 1, v12 // (op 2351) 
v_add_lshl_u32 v13, v8, v10, 4 // (op 2351) 


v_and_b32 v8, 1, v12 // (op 2351) 

v_lshlrev_b32 v10, 3, v8 // (op 2351) 

v_lshrrev_b32 v8, 6, v1 // (op 2351) 
v_lshl_add_u32 v12, v8, 4, v10 // (op 2351) 


v_add_u32 v8, v13, v12 // (op 2351) 


v_and_b32 v10, 48, v1 // (op 2390) 
v_and_b32 v12, 63, v1 // (op 2390) 
v_sub_u32 v13, v12, v10 // (op 2390) 

v_lshrrev_b32 v12, 1, v13 // (op 2390) 
v_add_lshl_u32 v14, v10, v12, 4 // (op 2390) 


v_and_b32 v10, 1, v13 // (op 2390) 

v_lshlrev_b32 v12, 3, v10 // (op 2390) 

v_add_u32 v10, v14, v12 // (op 2390) 


v_lshrrev_b32 v12, 6, v1 // (op 2414) 
v_and_b32 v13, 15, v1 // (op 2414) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 2414) 


v_and_b32 v12, 48, v1 // (op 2414) 
v_add_u32 v13, v12, v14 // (op 2414) 


v_and_b32 v12, 48, v1 // (op 2438) 
v_and_b32 v14, 15, v1 // (op 2438) 
v_add_u32 v15, v12, v14 // (op 2438) 


v_and_b32 v12, 48, v1 // (op 2477) 
v_and_b32 v14, 63, v1 // (op 2477) 
v_sub_u32 v16, v14, v12 // (op 2477) 

v_lshrrev_b32 v14, 1, v16 // (op 2477) 
v_add_lshl_u32 v17, v12, v14, 4 // (op 2477) 


v_and_b32 v12, 1, v16 // (op 2477) 

v_lshlrev_b32 v14, 3, v12 // (op 2477) 

v_lshrrev_b32 v12, 6, v1 // (op 2477) 
v_lshl_add_u32 v16, v12, 4, v14 // (op 2477) 


v_add_u32 v12, v17, v16 // (op 2477) 


v_and_b32 v14, 48, v1 // (op 2516) 
v_and_b32 v16, 63, v1 // (op 2516) 
v_sub_u32 v17, v16, v14 // (op 2516) 

v_lshrrev_b32 v16, 1, v17 // (op 2516) 
v_add_lshl_u32 v18, v14, v16, 4 // (op 2516) 


v_and_b32 v14, 1, v17 // (op 2516) 

v_lshlrev_b32 v16, 3, v14 // (op 2516) 

v_add_u32 v14, v18, v16 // (op 2516) 


v_lshrrev_b32 v16, 6, v1 // (op 2540) 
v_and_b32 v17, 15, v1 // (op 2540) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 2540) 


v_and_b32 v16, 48, v1 // (op 2540) 
v_add_u32 v17, v16, v18 // (op 2540) 


v_and_b32 v16, 48, v1 // (op 2564) 
v_and_b32 v18, 15, v1 // (op 2564) 
v_add_u32 v19, v16, v18 // (op 2564) 


v_mov_b32 v16, s6 // (op 2) Move value
v_mov_b32 v18, s7 // (op 43) Move value


s_mov_b32 s0, 0 // (op 395) call()
s_mov_b32 s1, 0 // (op 396) call()
s_mov_b32 s6, 0 // (op 397) call()
s_mov_b32 s7, 0 // (op 398) call()
s_mov_b32 s18, 0 // (op 399) call()
s_mov_b32 s19, 0 // (op 400) call()
s_mov_b32 s0, 0 // (op 295) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 294) 
s_cmp_gt_u32 s9, s52 // (op 294) 
s_mov_b32 s53, scc // (op 294) 
s_sub_u32 s54, s9, s52 // (op 294) 

s_cmp_eq_i32 s53, 0 // (op 294) 
s_cselect_b32 s52, 0, s54 // (op 294) 


s_cmp_lt_u32 s8, s52 // (op 294) 
s_mov_b32 s53, scc // (op 294) 
s_cmp_eq_i32 s53, 0 // (op 294) 
s_cselect_b32 s54, s52, s8 // (op 294) 


s_cmp_lt_u32 s0, s54 // (op 294) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_294 // (op 294) Condition: Top (jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_294 if false)
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_SKStreamTileLoop_294:
v_lshrrev_b32 v20, 6, v1 // (op 1662) 
s_mul_i32 s52, s16, s8 // (op 1662) 
s_add_u32 s52, s52, s0 // (op 1662) 
v_and_b32 v21, 63, v1 // (op 1662) 
s_mul_hi_u32 s53, s52, s13 // (op 1662) 
v_lshrrev_b32 v22, 4, v21 // (op 1662) 
s_sub_u32 s54, s52, s53 // (op 1662) 
v_lshlrev_b32 v23, 2, v22 // (op 1662) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 1662) 


s_lshr_b32 s55, s54, 1 // (op 1662) 

s_add_u32 s54, s55, s53 // (op 1662) 


v_lshrrev_b32 v20, 2, v21 // (op 1662) 

s_lshr_b32 s53, s54, s11 // (op 1662) 

v_and_b32 v21, 3, v20 // (op 1662) 

s_and_b32 s54, s11, 2147483648 // (op 1662) 
v_and_b32 v20, 3, v1 // (op 1662) 
s_cmp_eq_i32 s54, 0 // (op 1662) 
s_mov_b32 s55, scc // (op 1662) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 1662) 


s_cmp_eq_i32 s55, 0 // (op 1662) 
s_cselect_b32 s54, s52, s53 // (op 1662) 



s_mul_hi_u32 s52, s54, s12 // (op 1662) 
s_sub_u32 s53, s54, s52 // (op 1662) 
s_lshr_b32 s55, s53, 1 // (op 1662) 

s_add_u32 s53, s55, s52 // (op 1662) 


s_lshr_b32 s52, s53, s10 // (op 1662) 

s_and_b32 s53, s10, 2147483648 // (op 1662) 
s_cmp_eq_i32 s53, 0 // (op 1662) 
s_mov_b32 s55, scc // (op 1662) 

s_cmp_eq_i32 s55, 0 // (op 1662) 
s_cselect_b32 s53, s54, s52 // (op 1662) 


s_mul_i32 s52, s53, s14 // (op 1662) 
v_lshl_add_u32 v20, s53, 4, v22 // (op 1662) 


s_sub_u32 s53, s54, s52 // (op 1662) 


v_lshl_add_u32 v21, s53, 4, v23 // (op 1662) 


v_mul_lo_u32 v22, v21, s4 // (op 1662) 

v_add_lshl_u32 v21, v20, v22, 1 // (op 1662) 


s_mul_i32 s53, s4, s23 // (op 1664) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 1664) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 1664) least significant
s_mul_hi_u32 s55, s4, s22 // (op 1664) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1664) most significant: sum
s_add_u32 s53, s53, s55 // (op 1664) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 1664) 

s_mov_b32 s56, s20 // (op 1664) reg expression
s_mov_b32 s57, s21 // (op 1664) reg expression
s_mov_b32 s58, s54 // (op 1664) reg expression
s_mov_b32 s59, 131072 // (op 1664) call()

v_and_b32 v20, 48, v1 // (op 1748) 
v_and_b32 v22, 63, v1 // (op 1748) 
v_sub_u32 v23, v22, v20 // (op 1748) 

v_lshrrev_b32 v22, 1, v23 // (op 1748) 
v_add_lshl_u32 v24, v20, v22, 4 // (op 1748) 


v_and_b32 v20, 1, v23 // (op 1748) 

v_lshlrev_b32 v22, 3, v20 // (op 1748) 

v_lshrrev_b32 v20, 6, v1 // (op 1748) 
v_lshl_add_u32 v23, v20, 4, v22 // (op 1748) 


v_add_u32 v20, v24, v23 // (op 1748) 


v_and_b32 v22, 48, v1 // (op 1813) 
v_and_b32 v23, 63, v1 // (op 1813) 
v_sub_u32 v24, v23, v22 // (op 1813) 

v_lshrrev_b32 v23, 1, v24 // (op 1813) 
v_add_lshl_u32 v25, v22, v23, 4 // (op 1813) 


v_and_b32 v22, 1, v24 // (op 1813) 

v_lshlrev_b32 v23, 3, v22 // (op 1813) 

v_add_u32 v22, v25, v23 // (op 1813) 


v_lshrrev_b32 v23, 6, v1 // (op 1879) 
v_and_b32 v24, 15, v1 // (op 1879) 
v_lshl_add_u32 v25, v23, 4, v24 // (op 1879) 


v_and_b32 v23, 48, v1 // (op 1879) 
v_add_u32 v24, v23, v25 // (op 1879) 


v_and_b32 v23, 48, v1 // (op 1945) 
v_and_b32 v25, 15, v1 // (op 1945) 
v_add_u32 v26, v23, v25 // (op 1945) 


s_mul_i32 s6, s8, s16 // (op 299) 
s_add_u32 s6, s6, s0 // (op 299) 
s_mul_hi_u32 s52, s6, s13 // (op 300) 
s_sub_u32 s53, s6, s52 // (op 300) 
s_lshr_b32 s54, s53, 1 // (op 300) 

s_add_u32 s53, s54, s52 // (op 300) 


s_lshr_b32 s52, s53, s11 // (op 300) 

s_and_b32 s53, s11, 2147483648 // (op 300) 
s_cmp_eq_i32 s53, 0 // (op 300) 
s_mov_b32 s54, scc // (op 300) 

s_cmp_eq_i32 s54, 0 // (op 300) 
s_cselect_b32 s53, s6, s52 // (op 300) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 300) 
s_mul_i32 s52, s53, s54 // (op 300) 


s_sub_u32 s18, s6, s52 // (op 300) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 301) 
s_cmp_gt_u32 s52, s18 // (op 301) 
s_mov_b32 s54, scc // (op 301) 
s_sub_u32 s55, s52, s18 // (op 301) 

s_cmp_eq_i32 s54, 0 // (op 301) 
s_cselect_b32 s52, 0, s55 // (op 301) 


s_cmp_gt_u32 s8, s0 // (op 301) 
s_mov_b32 s53, scc // (op 301) 
s_sub_u32 s54, s8, s0 // (op 301) 
s_cmp_eq_i32 s53, 0 // (op 301) 
s_cselect_b32 s55, 0, s54 // (op 301) 


s_cmp_lt_u32 s52, s55 // (op 301) 
s_mov_b32 s53, scc // (op 301) 
s_cmp_eq_i32 s53, 0 // (op 301) 
s_cselect_b32 s54, s55, s52 // (op 301) 



s_cmp_gt_u32 s9, s6 // (op 301) 
s_mov_b32 s55, scc // (op 301) 
s_sub_u32 s52, s9, s6 // (op 301) 
s_cmp_eq_i32 s55, 0 // (op 301) 
s_cselect_b32 s53, 0, s52 // (op 301) 


s_cmp_lt_u32 s54, s53 // (op 301) 
s_mov_b32 s52, scc // (op 301) 
s_cmp_eq_i32 s52, 0 // (op 301) 
s_cselect_b32 s7, s53, s54 // (op 301) 



s_add_u32 s52, s18, s7 // (op 302) 
s_sub_u32 s19, s52, 1 // (op 302) 

v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
s_mul_i32 s52, s16, s8 // (op 1194) 
s_add_u32 s52, s52, s0 // (op 1194) 
s_mul_hi_u32 s53, s52, s13 // (op 1194) 
s_sub_u32 s54, s52, s53 // (op 1194) 
s_lshr_b32 s55, s54, 1 // (op 1194) 

s_add_u32 s54, s55, s53 // (op 1194) 


s_lshr_b32 s53, s54, s11 // (op 1194) 

s_and_b32 s54, s11, 2147483648 // (op 1194) 
s_cmp_eq_i32 s54, 0 // (op 1194) 
s_mov_b32 s55, scc // (op 1194) 

s_cmp_eq_i32 s55, 0 // (op 1194) 
s_cselect_b32 s54, s52, s53 // (op 1194) 


s_mul_hi_u32 s53, s54, s12 // (op 1194) 
s_sub_u32 s55, s54, s53 // (op 1194) 
s_lshr_b32 s60, s55, 1 // (op 1194) 

s_add_u32 s55, s60, s53 // (op 1194) 


s_lshr_b32 s53, s55, s10 // (op 1194) 

s_and_b32 s55, s10, 2147483648 // (op 1194) 
s_cmp_eq_i32 s55, 0 // (op 1194) 
s_mov_b32 s60, scc // (op 1194) 

s_cmp_eq_i32 s60, 0 // (op 1194) 
s_cselect_b32 s55, s54, s53 // (op 1194) 


s_lshl_b32 s53, s55, 4 // (op 1194) 

s_mov_b32 s61, 0 // (op 1194) convert
s_mov_b32 s60, s53 // (op 1194) convert

s_ashr_i64 s[62:63], s[2:3], 7 // (op 1194) 
s_mul_i32 s53, s54, s62 // (op 1194) 


s_sub_u32 s54, s52, s53 // (op 1194) 


v_lshl_add_u32 v23, s54, 7, v1 // (op 1194) 

v_mul_lo_u32 v29, v23, s25 // (op 1194) most significant: low of low * high
v_mul_hi_u32 v25, v23, s24 // (op 1194) most significant: high of low * low
v_mul_lo_u32 v28, v23, s24 // (op 1194) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 1194) most significant: sum


v_mov_b32 v23, s60 // (op 1194) 
v_mov_b32 v25, s61 // (op 1194) 
v_add_co_u32 v30, s[52:53], v23, v28 // (op 1194) least significant half
v_addc_co_u32 v31, s[52:53], v25, v29, s[52:53] // (op 1194) most significant half





s_mul_i32 s53, s24, s3 // (op 1196) most significant: low of low * high
s_mul_i32 s54, s25, s2 // (op 1196) most significant: low of high * low
s_mul_i32 s52, s24, s2 // (op 1196) least significant
s_mul_hi_u32 s55, s24, s2 // (op 1196) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1196) most significant: sum
s_add_u32 s53, s53, s55 // (op 1196) most significant: sum


s_mov_b32 s60, s26 // (op 1196) reg expression
s_mov_b32 s61, s27 // (op 1196) reg expression
s_mov_b32 s62, s52 // (op 1196) reg expression
s_mov_b32 s63, 131072 // (op 1196) call()

s_mul_i32 s52, s16, s8 // (op 1213) 
s_add_u32 s52, s52, s0 // (op 1213) 
s_mul_hi_u32 s53, s52, s13 // (op 1213) 
s_sub_u32 s54, s52, s53 // (op 1213) 
s_lshr_b32 s55, s54, 1 // (op 1213) 

s_add_u32 s54, s55, s53 // (op 1213) 


s_lshr_b32 s53, s54, s11 // (op 1213) 

s_and_b32 s54, s11, 2147483648 // (op 1213) 
s_cmp_eq_i32 s54, 0 // (op 1213) 
s_mov_b32 s55, scc // (op 1213) 

s_cmp_eq_i32 s55, 0 // (op 1213) 
s_cselect_b32 s54, s52, s53 // (op 1213) 


s_mul_hi_u32 s53, s54, s12 // (op 1213) 
s_sub_u32 s55, s54, s53 // (op 1213) 
s_lshr_b32 s64, s55, 1 // (op 1213) 

s_add_u32 s55, s64, s53 // (op 1213) 


s_lshr_b32 s53, s55, s10 // (op 1213) 

s_and_b32 s55, s10, 2147483648 // (op 1213) 
s_cmp_eq_i32 s55, 0 // (op 1213) 
s_mov_b32 s64, scc // (op 1213) 

s_cmp_eq_i32 s64, 0 // (op 1213) 
s_cselect_b32 s55, s54, s53 // (op 1213) 


s_lshl_b32 s53, s55, 4 // (op 1213) 

s_mov_b32 s65, 0 // (op 1213) convert
s_mov_b32 s64, s53 // (op 1213) convert

s_ashr_i64 s[66:67], s[2:3], 7 // (op 1213) 
s_mul_i32 s53, s54, s66 // (op 1213) 


s_sub_u32 s54, s52, s53 // (op 1213) 


v_lshl_add_u32 v23, s54, 7, v1 // (op 1213) 

v_mul_lo_u32 v29, v23, s25 // (op 1213) most significant: low of low * high
v_mul_hi_u32 v25, v23, s24 // (op 1213) most significant: high of low * low
v_mul_lo_u32 v28, v23, s24 // (op 1213) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 1213) most significant: sum


v_mov_b32 v23, s64 // (op 1213) 
v_mov_b32 v25, s65 // (op 1213) 
v_add_co_u32 v32, s[52:53], v23, v28 // (op 1213) least significant half
v_addc_co_u32 v33, s[52:53], v25, v29, s[52:53] // (op 1213) most significant half





s_mul_i32 s53, s24, s3 // (op 1215) most significant: low of low * high
s_mul_i32 s54, s25, s2 // (op 1215) most significant: low of high * low
s_mul_i32 s52, s24, s2 // (op 1215) least significant
s_mul_hi_u32 s55, s24, s2 // (op 1215) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1215) most significant: sum
s_add_u32 s53, s53, s55 // (op 1215) most significant: sum


s_mov_b32 s60, s26 // (op 1215) reg expression
s_mov_b32 s61, s27 // (op 1215) reg expression
s_mov_b32 s62, s52 // (op 1215) reg expression
s_mov_b32 s63, 131072 // (op 1215) call()

s_mul_i32 s52, s16, s8 // (op 1325) 
s_add_u32 s52, s52, s0 // (op 1325) 
s_mul_hi_u32 s53, s52, s13 // (op 1325) 
s_sub_u32 s54, s52, s53 // (op 1325) 
s_lshr_b32 s55, s54, 1 // (op 1325) 

s_add_u32 s54, s55, s53 // (op 1325) 


s_lshr_b32 s53, s54, s11 // (op 1325) 

s_and_b32 s54, s11, 2147483648 // (op 1325) 
s_cmp_eq_i32 s54, 0 // (op 1325) 
s_mov_b32 s55, scc // (op 1325) 

s_cmp_eq_i32 s55, 0 // (op 1325) 
s_cselect_b32 s54, s52, s53 // (op 1325) 


s_mul_hi_u32 s53, s54, s12 // (op 1325) 
s_sub_u32 s55, s54, s53 // (op 1325) 
s_lshr_b32 s64, s55, 1 // (op 1325) 

s_add_u32 s55, s64, s53 // (op 1325) 


s_lshr_b32 s53, s55, s10 // (op 1325) 

s_and_b32 s55, s10, 2147483648 // (op 1325) 
s_cmp_eq_i32 s55, 0 // (op 1325) 
s_mov_b32 s64, scc // (op 1325) 

s_cmp_eq_i32 s64, 0 // (op 1325) 
s_cselect_b32 s55, s54, s53 // (op 1325) 


s_mul_i32 s53, s55, s14 // (op 1325) 

s_sub_u32 s55, s54, s53 // (op 1325) 

s_lshl_b32 s53, s55, 4 // (op 1325) 

s_mov_b32 s65, 0 // (op 1325) convert
s_mov_b32 s64, s53 // (op 1325) convert

s_ashr_i64 s[66:67], s[2:3], 7 // (op 1325) 
s_mul_i32 s53, s54, s66 // (op 1325) 


s_sub_u32 s54, s52, s53 // (op 1325) 


v_lshl_add_u32 v23, s54, 7, v1 // (op 1325) 

v_mul_lo_u32 v29, v23, s29 // (op 1325) most significant: low of low * high
v_mul_hi_u32 v25, v23, s28 // (op 1325) most significant: high of low * low
v_mul_lo_u32 v28, v23, s28 // (op 1325) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 1325) most significant: sum

v_mov_b32 v25, s64 // (op 1325) 
v_mov_b32 v27, s65 // (op 1325) 
v_add_co_u32 v28, s[52:53], v28, v25 // (op 1325) least significant half
v_addc_co_u32 v29, s[52:53], v29, v27, s[52:53] // (op 1325) most significant half





s_mul_i32 s53, s28, s3 // (op 1327) most significant: low of low * high
s_mul_i32 s54, s29, s2 // (op 1327) most significant: low of high * low
s_mul_i32 s52, s28, s2 // (op 1327) least significant
s_mul_hi_u32 s55, s28, s2 // (op 1327) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1327) most significant: sum
s_add_u32 s53, s53, s55 // (op 1327) most significant: sum


s_mov_b32 s64, s30 // (op 1327) reg expression
s_mov_b32 s65, s31 // (op 1327) reg expression
s_mov_b32 s66, s52 // (op 1327) reg expression
s_mov_b32 s67, 131072 // (op 1327) call()

s_mul_i32 s52, s16, s8 // (op 1344) 
s_add_u32 s52, s52, s0 // (op 1344) 
s_mul_hi_u32 s53, s52, s13 // (op 1344) 
s_sub_u32 s54, s52, s53 // (op 1344) 
s_lshr_b32 s55, s54, 1 // (op 1344) 

s_add_u32 s54, s55, s53 // (op 1344) 


s_lshr_b32 s53, s54, s11 // (op 1344) 

s_and_b32 s54, s11, 2147483648 // (op 1344) 
s_cmp_eq_i32 s54, 0 // (op 1344) 
s_mov_b32 s55, scc // (op 1344) 

s_cmp_eq_i32 s55, 0 // (op 1344) 
s_cselect_b32 s54, s52, s53 // (op 1344) 


s_mul_hi_u32 s53, s54, s12 // (op 1344) 
s_sub_u32 s55, s54, s53 // (op 1344) 
s_lshr_b32 s68, s55, 1 // (op 1344) 

s_add_u32 s55, s68, s53 // (op 1344) 


s_lshr_b32 s53, s55, s10 // (op 1344) 

s_and_b32 s55, s10, 2147483648 // (op 1344) 
s_cmp_eq_i32 s55, 0 // (op 1344) 
s_mov_b32 s68, scc // (op 1344) 

s_cmp_eq_i32 s68, 0 // (op 1344) 
s_cselect_b32 s55, s54, s53 // (op 1344) 


s_mul_i32 s53, s55, s14 // (op 1344) 

s_sub_u32 s55, s54, s53 // (op 1344) 

s_lshl_b32 s53, s55, 4 // (op 1344) 

s_mov_b32 s69, 0 // (op 1344) convert
s_mov_b32 s68, s53 // (op 1344) convert

s_ashr_i64 s[70:71], s[2:3], 7 // (op 1344) 
s_mul_i32 s53, s54, s70 // (op 1344) 


s_sub_u32 s54, s52, s53 // (op 1344) 


v_lshl_add_u32 v23, s54, 7, v1 // (op 1344) 

v_mul_lo_u32 v35, v23, s29 // (op 1344) most significant: low of low * high
v_mul_hi_u32 v25, v23, s28 // (op 1344) most significant: high of low * low
v_mul_lo_u32 v34, v23, s28 // (op 1344) least significant: low of low * low
v_add_u32 v35, v35, v25 // (op 1344) most significant: sum

v_mov_b32 v25, s68 // (op 1344) 
v_mov_b32 v27, s69 // (op 1344) 
v_add_co_u32 v34, s[52:53], v34, v25 // (op 1344) least significant half
v_addc_co_u32 v35, s[52:53], v35, v27, s[52:53] // (op 1344) most significant half





s_mul_i32 s53, s28, s3 // (op 1346) most significant: low of low * high
s_mul_i32 s54, s29, s2 // (op 1346) most significant: low of high * low
s_mul_i32 s52, s28, s2 // (op 1346) least significant
s_mul_hi_u32 s55, s28, s2 // (op 1346) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1346) most significant: sum
s_add_u32 s53, s53, s55 // (op 1346) most significant: sum


s_mov_b32 s64, s30 // (op 1346) reg expression
s_mov_b32 s65, s31 // (op 1346) reg expression
s_mov_b32 s66, s52 // (op 1346) reg expression
s_mov_b32 s67, 131072 // (op 1346) call()

v_and_b32 v23, -16, v1 // (op 1453) 
s_mul_i32 s52, s16, s8 // (op 1453) 
s_add_u32 s52, s52, s0 // (op 1453) 
v_sub_u32 v25, v1, v23 // (op 1453) 

s_mul_hi_u32 s53, s52, s13 // (op 1453) 
v_lshrrev_b32 v23, 4, v1 // (op 1453) 
s_sub_u32 s54, s52, s53 // (op 1453) 
s_lshr_b32 s55, s54, 1 // (op 1453) 

s_add_u32 s54, s55, s53 // (op 1453) 


s_lshr_b32 s53, s54, s11 // (op 1453) 

s_and_b32 s54, s11, 2147483648 // (op 1453) 
s_cmp_eq_i32 s54, 0 // (op 1453) 
s_mov_b32 s55, scc // (op 1453) 

s_cmp_eq_i32 s55, 0 // (op 1453) 
s_cselect_b32 s54, s52, s53 // (op 1453) 


s_mul_hi_u32 s53, s54, s12 // (op 1453) 
s_sub_u32 s55, s54, s53 // (op 1453) 
s_lshr_b32 s68, s55, 1 // (op 1453) 

s_add_u32 s55, s68, s53 // (op 1453) 


s_lshr_b32 s53, s55, s10 // (op 1453) 

s_and_b32 s55, s10, 2147483648 // (op 1453) 
s_cmp_eq_i32 s55, 0 // (op 1453) 
s_mov_b32 s68, scc // (op 1453) 

s_cmp_eq_i32 s68, 0 // (op 1453) 
s_cselect_b32 s55, s54, s53 // (op 1453) 


v_lshl_add_u32 v27, s55, 4, v25 // (op 1453) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 1453) 
s_mul_i32 s53, s54, s68 // (op 1453) 


s_sub_u32 s54, s52, s53 // (op 1453) 


v_lshl_add_u32 v25, s54, 2, v23 // (op 1453) 


v_mul_lo_u32 v23, v25, s32 // (op 1453) 

v_add_u32 v25, v27, v23 // (op 1453) 


s_mul_i32 s53, s32, s37 // (op 1455) most significant: low of low * high
s_mul_i32 s54, s33, s36 // (op 1455) most significant: low of high * low
s_mul_i32 s52, s32, s36 // (op 1455) least significant
s_mul_hi_u32 s55, s32, s36 // (op 1455) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1455) most significant: sum
s_add_u32 s53, s53, s55 // (op 1455) most significant: sum


s_mov_b32 s68, s34 // (op 1455) reg expression
s_mov_b32 s69, s35 // (op 1455) reg expression
s_mov_b32 s70, s52 // (op 1455) reg expression
s_mov_b32 s71, 131072 // (op 1455) call()

v_and_b32 v23, -16, v1 // (op 1472) 
s_mul_i32 s52, s16, s8 // (op 1472) 
s_add_u32 s52, s52, s0 // (op 1472) 
v_sub_u32 v27, v1, v23 // (op 1472) 

s_mul_hi_u32 s53, s52, s13 // (op 1472) 
v_lshrrev_b32 v23, 4, v1 // (op 1472) 
s_sub_u32 s54, s52, s53 // (op 1472) 
s_lshr_b32 s55, s54, 1 // (op 1472) 

s_add_u32 s54, s55, s53 // (op 1472) 


s_lshr_b32 s53, s54, s11 // (op 1472) 

s_and_b32 s54, s11, 2147483648 // (op 1472) 
s_cmp_eq_i32 s54, 0 // (op 1472) 
s_mov_b32 s55, scc // (op 1472) 

s_cmp_eq_i32 s55, 0 // (op 1472) 
s_cselect_b32 s54, s52, s53 // (op 1472) 


s_mul_hi_u32 s53, s54, s12 // (op 1472) 
s_sub_u32 s55, s54, s53 // (op 1472) 
s_lshr_b32 s72, s55, 1 // (op 1472) 

s_add_u32 s55, s72, s53 // (op 1472) 


s_lshr_b32 s53, s55, s10 // (op 1472) 

s_and_b32 s55, s10, 2147483648 // (op 1472) 
s_cmp_eq_i32 s55, 0 // (op 1472) 
s_mov_b32 s72, scc // (op 1472) 

s_cmp_eq_i32 s72, 0 // (op 1472) 
s_cselect_b32 s55, s54, s53 // (op 1472) 


v_lshl_add_u32 v36, s55, 4, v27 // (op 1472) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 1472) 
s_mul_i32 s53, s54, s72 // (op 1472) 


s_sub_u32 s54, s52, s53 // (op 1472) 


v_lshl_add_u32 v27, s54, 2, v23 // (op 1472) 


v_mul_lo_u32 v23, v27, s32 // (op 1472) 

v_add_u32 v27, v36, v23 // (op 1472) 


s_mul_i32 s53, s32, s37 // (op 1474) most significant: low of low * high
s_mul_i32 s54, s33, s36 // (op 1474) most significant: low of high * low
s_mul_i32 s52, s32, s36 // (op 1474) least significant
s_mul_hi_u32 s55, s32, s36 // (op 1474) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1474) most significant: sum
s_add_u32 s53, s53, s55 // (op 1474) most significant: sum


s_mov_b32 s68, s34 // (op 1474) reg expression
s_mov_b32 s69, s35 // (op 1474) reg expression
s_mov_b32 s70, s52 // (op 1474) reg expression
s_mov_b32 s71, 131072 // (op 1474) call()

v_and_b32 v23, -16, v1 // (op 1581) 
s_mul_i32 s52, s16, s8 // (op 1581) 
s_add_u32 s52, s52, s0 // (op 1581) 
v_sub_u32 v36, v1, v23 // (op 1581) 

s_mul_hi_u32 s53, s52, s13 // (op 1581) 
v_lshrrev_b32 v23, 4, v1 // (op 1581) 
s_sub_u32 s54, s52, s53 // (op 1581) 
s_lshr_b32 s55, s54, 1 // (op 1581) 

s_add_u32 s54, s55, s53 // (op 1581) 


s_lshr_b32 s53, s54, s11 // (op 1581) 

s_and_b32 s54, s11, 2147483648 // (op 1581) 
s_cmp_eq_i32 s54, 0 // (op 1581) 
s_mov_b32 s55, scc // (op 1581) 

s_cmp_eq_i32 s55, 0 // (op 1581) 
s_cselect_b32 s54, s52, s53 // (op 1581) 


s_mul_hi_u32 s53, s54, s12 // (op 1581) 
s_sub_u32 s55, s54, s53 // (op 1581) 
s_lshr_b32 s72, s55, 1 // (op 1581) 

s_add_u32 s55, s72, s53 // (op 1581) 


s_lshr_b32 s53, s55, s10 // (op 1581) 

s_and_b32 s55, s10, 2147483648 // (op 1581) 
s_cmp_eq_i32 s55, 0 // (op 1581) 
s_mov_b32 s72, scc // (op 1581) 

s_cmp_eq_i32 s72, 0 // (op 1581) 
s_cselect_b32 s55, s54, s53 // (op 1581) 


s_mul_i32 s53, s55, s14 // (op 1581) 

s_sub_u32 s55, s54, s53 // (op 1581) 

s_ashr_i64 s[72:73], s[2:3], 7 // (op 1581) 
v_lshl_add_u32 v37, s55, 4, v36 // (op 1581) 


s_mul_i32 s53, s54, s72 // (op 1581) 


s_sub_u32 s54, s52, s53 // (op 1581) 


v_lshl_add_u32 v36, s54, 2, v23 // (op 1581) 


v_mul_lo_u32 v23, v36, s38 // (op 1581) 
v_add_u32 v23, v23, v37 // (op 1581) 


s_mul_i32 s53, s38, s37 // (op 1583) most significant: low of low * high
s_mul_i32 s54, s39, s36 // (op 1583) most significant: low of high * low
s_mul_i32 s52, s38, s36 // (op 1583) least significant
s_mul_hi_u32 s55, s38, s36 // (op 1583) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1583) most significant: sum
s_add_u32 s53, s53, s55 // (op 1583) most significant: sum


s_mov_b32 s72, s40 // (op 1583) reg expression
s_mov_b32 s73, s41 // (op 1583) reg expression
s_mov_b32 s74, s52 // (op 1583) reg expression
s_mov_b32 s75, 131072 // (op 1583) call()

v_and_b32 v36, -16, v1 // (op 1600) 
s_mul_i32 s52, s16, s8 // (op 1600) 
s_add_u32 s52, s52, s0 // (op 1600) 
v_sub_u32 v37, v1, v36 // (op 1600) 

s_mul_hi_u32 s53, s52, s13 // (op 1600) 
v_lshrrev_b32 v36, 4, v1 // (op 1600) 
s_sub_u32 s54, s52, s53 // (op 1600) 
s_lshr_b32 s55, s54, 1 // (op 1600) 

s_add_u32 s54, s55, s53 // (op 1600) 


s_lshr_b32 s53, s54, s11 // (op 1600) 

s_and_b32 s54, s11, 2147483648 // (op 1600) 
s_cmp_eq_i32 s54, 0 // (op 1600) 
s_mov_b32 s55, scc // (op 1600) 

s_cmp_eq_i32 s55, 0 // (op 1600) 
s_cselect_b32 s54, s52, s53 // (op 1600) 


s_mul_hi_u32 s53, s54, s12 // (op 1600) 
s_sub_u32 s55, s54, s53 // (op 1600) 
s_lshr_b32 s76, s55, 1 // (op 1600) 

s_add_u32 s55, s76, s53 // (op 1600) 


s_lshr_b32 s53, s55, s10 // (op 1600) 

s_and_b32 s55, s10, 2147483648 // (op 1600) 
s_cmp_eq_i32 s55, 0 // (op 1600) 
s_mov_b32 s76, scc // (op 1600) 

s_cmp_eq_i32 s76, 0 // (op 1600) 
s_cselect_b32 s55, s54, s53 // (op 1600) 


s_mul_i32 s53, s55, s14 // (op 1600) 

s_sub_u32 s55, s54, s53 // (op 1600) 

v_lshl_add_u32 v38, s55, 4, v37 // (op 1600) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1600) 
s_mul_i32 s53, s54, s76 // (op 1600) 


s_sub_u32 s54, s52, s53 // (op 1600) 


v_lshl_add_u32 v37, s54, 2, v36 // (op 1600) 


v_mul_lo_u32 v36, v37, s38 // (op 1600) 
v_add_u32 v36, v36, v38 // (op 1600) 


s_mul_i32 s53, s38, s37 // (op 1602) most significant: low of low * high
s_mul_i32 s54, s39, s36 // (op 1602) most significant: low of high * low
s_mul_i32 s52, s38, s36 // (op 1602) least significant
s_mul_hi_u32 s55, s38, s36 // (op 1602) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1602) most significant: sum
s_add_u32 s53, s53, s55 // (op 1602) most significant: sum


s_mov_b32 s72, s40 // (op 1602) reg expression
s_mov_b32 s73, s41 // (op 1602) reg expression
s_mov_b32 s74, s52 // (op 1602) reg expression
s_mov_b32 s75, 131072 // (op 1602) call()

v_lshlrev_b32 v37, 4, v1 // (op 1789) 
s_nop 0
v_readfirstlane_b32 s52, v37 // (op 1789) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2677) 

s_lshl_b64 s[54:55], s[28:29], 6 // (op 1105) 
v_mov_b64 v[38:39], s[54:55] // (op 1105) convert


v_mov_b64 v[40:41], v[28:29] // (op 1105) reg expression
s_mov_b32 m0, s52 // (op 1105) reg expression
buffer_load_dwordx4 v40, s[64:67], 0 offen offset: 0 lds // (op 1105) Load value direct to lds
v_add_co_u32 v40, s[54:55], v40, v38 // (op 1105) least significant half
v_addc_co_u32 v41, s[54:55], 0, v41, s[54:55] // (op 1105) most significant half

s_add_u32 m0, m0, 1024 // (op 1105) 
buffer_load_dwordx4 v40, s[64:67], 0 offen offset: 0 lds // (op 1105) Load value direct to lds





v_mov_b32 v37, v23 // (op 860) reg expression

buffer_load_ubyte v38, v37, s[72:75], 0 offen offset: 0 // (op 860) Load value

v_lshlrev_b32 v37, 4, v1 // (op 1724) 
s_nop 0
v_readfirstlane_b32 s52, v37 // (op 1724) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2658) 

s_lshl_b64 s[54:55], s[24:25], 6 // (op 1110) 
v_mov_b64 v[40:41], s[54:55] // (op 1110) convert


v_mov_b64 v[42:43], v[30:31] // (op 1110) reg expression
s_add_u32 s53, 2048, s52 // (op 1110) 
s_mov_b32 m0, s53 // (op 1110) reg expression
buffer_load_dwordx4 v42, s[60:63], 0 offen offset: 0 lds // (op 1110) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 1110) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 1110) most significant half

s_add_u32 m0, m0, 1024 // (op 1110) 
buffer_load_dwordx4 v42, s[60:63], 0 offen offset: 0 lds // (op 1110) Load value direct to lds






v_mov_b32 v37, v25 // (op 866) reg expression

buffer_load_ubyte v39, v37, s[68:71], 0 offen offset: 0 // (op 866) Load value

v_lshlrev_b32 v37, 4, v1 // (op 2504) 
s_nop 0
v_readfirstlane_b32 s52, v37 // (op 2504) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2857) 

s_lshl_b64 s[54:55], s[28:29], 6 // (op 1115) 
v_mov_b64 v[40:41], s[54:55] // (op 1115) convert


s_lshl_b64 s[54:55], s[28:29], 7 // (op 1115) 
v_mov_b32 v37, s54 // (op 1115) 
v_mov_b32 v42, s55 // (op 1115) 
v_add_co_u32 v44, s[76:77], v28, v37 // (op 1115) least significant half
v_addc_co_u32 v45, s[76:77], v29, v42, s[76:77] // (op 1115) most significant half




s_add_u32 s53, 4096, s52 // (op 1115) 
s_mov_b32 m0, s53 // (op 1115) reg expression
buffer_load_dwordx4 v44, s[64:67], 0 offen offset: 0 lds // (op 1115) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v40 // (op 1115) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 1115) most significant half

s_add_u32 m0, m0, 1024 // (op 1115) 
buffer_load_dwordx4 v44, s[64:67], 0 offen offset: 0 lds // (op 1115) Load value direct to lds







s_lshl_b32 s52, s38, 2 // (op 872) 
v_add_u32 v37, v23, s52 // (op 872) 


buffer_load_ubyte v28, v37, s[72:75], 0 offen offset: 0 // (op 872) Load value


v_lshlrev_b32 v23, 4, v1 // (op 2465) 
s_nop 0
v_readfirstlane_b32 s52, v23 // (op 2465) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2847) 

s_lshl_b64 s[54:55], s[24:25], 6 // (op 1117) 
v_mov_b64 v[40:41], s[54:55] // (op 1117) convert


s_lshl_b64 s[54:55], s[24:25], 7 // (op 1117) 
v_mov_b32 v23, s54 // (op 1117) 
v_mov_b32 v29, s55 // (op 1117) 
v_add_co_u32 v42, s[76:77], v30, v23 // (op 1117) least significant half
v_addc_co_u32 v43, s[76:77], v31, v29, s[76:77] // (op 1117) most significant half




s_add_u32 s53, 6144, s52 // (op 1117) 
s_mov_b32 m0, s53 // (op 1117) reg expression
buffer_load_dwordx4 v42, s[60:63], 0 offen offset: 0 lds // (op 1117) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 1117) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 1117) most significant half

s_add_u32 m0, m0, 1024 // (op 1117) 
buffer_load_dwordx4 v42, s[60:63], 0 offen offset: 0 lds // (op 1117) Load value direct to lds







s_lshl_b32 s52, s32, 2 // (op 878) 
v_add_u32 v23, v25, s52 // (op 878) 


buffer_load_ubyte v37, v23, s[68:71], 0 offen offset: 0 // (op 878) Load value


v_and_b32 v23, -16, v1 // (op 1921) 
v_sub_u32 v25, v1, v23 // (op 1921) 

v_and_b32 v23, 4294967280, v1 // (op 1921) 
v_add_u32 v29, v23, v25 // (op 1921) 


s_barrier  // (op 2717) 



s_waitcnt vmcnt(9)
ds_write_b8 v29, v38 offset:8192 // (op 881) Store local data 

v_and_b32 v23, -16, v1 // (op 1855) 
v_sub_u32 v25, v1, v23 // (op 1855) 

v_and_b32 v23, 4294967280, v1 // (op 1855) 
v_add_u32 v29, v23, v25 // (op 1855) 


s_barrier  // (op 2697) 



s_waitcnt vmcnt(6)
ds_write_b8 v29, v39 offset:8256 // (op 883) Store local data 

s_barrier  // (op 2674) 



ds_read_b64_tr_b8 v[40:41], v22 // (op 900) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v22 offset:128 // (op 900) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v22 offset:1024 // (op 900) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v22 offset:1152 // (op 900) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2714) 


ds_read_u8 v23, v26 offset:8192 // (op 906) Load local data 
s_barrier  // (op 2655) 



ds_read_b64_tr_b8 v[48:49], v20 offset:2048 // (op 897) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v20 offset:2176 // (op 897) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v20 offset:3072 // (op 897) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v20 offset:3200 // (op 897) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2694) 


ds_read_u8 v25, v24 offset:8256 // (op 903) Load local data 
v_lshlrev_b32 v29, 4, v1 // (op 1696) 
s_nop 0
v_readfirstlane_b32 s52, v29 // (op 1696) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v29, 4, v1 // (op 1761) 
s_nop 0
v_readfirstlane_b32 s53, v29 // (op 1761) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -16, v1 // (op 1826) 
v_sub_u32 v30, v1, v29 // (op 1826) 

v_and_b32 v29, 4294967280, v1 // (op 1826) 
v_add_u32 v31, v29, v30 // (op 1826) 


v_and_b32 v29, -16, v1 // (op 1892) 
v_sub_u32 v30, v1, v29 // (op 1892) 

v_and_b32 v29, 4294967280, v1 // (op 1892) 
v_add_u32 v56, v29, v30 // (op 1892) 


v_lshlrev_b32 v29, 4, v1 // (op 2450) 
s_nop 0
v_readfirstlane_b32 s54, v29 // (op 2450) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v29, 4, v1 // (op 2489) 
s_nop 0
v_readfirstlane_b32 s55, v29 // (op 2489) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -16, v1 // (op 2528) 
v_sub_u32 v30, v1, v29 // (op 2528) 

v_and_b32 v29, 4294967280, v1 // (op 2528) 
v_add_u32 v57, v29, v30 // (op 2528) 


v_and_b32 v29, -16, v1 // (op 2552) 
v_sub_u32 v30, v1, v29 // (op 2552) 

v_and_b32 v29, 4294967280, v1 // (op 2552) 
v_add_u32 v58, v29, v30 // (op 2552) 


s_mov_b32 s1, 0 // (op 61) call()
s_and_b32 s76, s7, 4294967294 // (op 60) 
s_cmp_lt_u32 s1, s76 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_60 if false)
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoop_60:
s_add_u32 s1, s1, 2 // (op 62) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[40:47], a[0:3], v25, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 90) 
s_barrier  // (op 2680) 

s_lshl_b64 s[76:77], s[28:29], 6 // (op 1044) 
v_mov_b64 v[60:61], s[76:77] // (op 1044) convert


v_mov_b64 v[62:63], v[34:35] // (op 1044) 
s_lshl_b64 s[76:77], s[28:29], 7 // (op 1044) 
s_lshl_b64 s[78:79], s[76:77], 1 // (op 1044) 

v_mov_b32 v59, s78 // (op 1044) 
v_mov_b32 v29, s79 // (op 1044) 
v_add_co_u32 v64, s[76:77], v62, v59 // (op 1044) least significant half
v_addc_co_u32 v65, s[76:77], v63, v29, s[76:77] // (op 1044) most significant half




s_mov_b32 m0, s53 // (op 1044) reg expression
buffer_load_dwordx4 v64, s[64:67], 0 offen offset: 0 lds // (op 1044) Load value direct to lds
v_add_co_u32 v64, s[76:77], v64, v60 // (op 1044) least significant half
v_addc_co_u32 v65, s[76:77], 0, v65, s[76:77] // (op 1044) most significant half

s_add_u32 m0, m0, 1024 // (op 1044) 
buffer_load_dwordx4 v64, s[64:67], 0 offen offset: 0 lds // (op 1044) Load value direct to lds




v_mov_b32 v29, v36 // (op 25) 
s_lshl_b32 s76, s38, 3 // (op 25) 
v_add_u32 v30, v29, s76 // (op 25) 


buffer_load_ubyte v38, v30, s[72:75], 0 offen offset: 0 // (op 25) Load value

s_barrier  // (op 2661) 

s_lshl_b64 s[76:77], s[24:25], 6 // (op 1039) 
v_mov_b64 v[60:61], s[76:77] // (op 1039) convert


v_mov_b64 v[64:65], v[32:33] // (op 1039) 
s_lshl_b64 s[76:77], s[24:25], 7 // (op 1039) 
s_lshl_b64 s[78:79], s[76:77], 1 // (op 1039) 

v_mov_b32 v30, s78 // (op 1039) 
v_mov_b32 v59, s79 // (op 1039) 
v_add_co_u32 v66, s[76:77], v64, v30 // (op 1039) least significant half
v_addc_co_u32 v67, s[76:77], v65, v59, s[76:77] // (op 1039) most significant half




s_add_u32 s76, 2048, s52 // (op 1039) 
s_nop 0
s_mov_b32 m0, s76 // (op 1039) reg expression
buffer_load_dwordx4 v66, s[60:63], 0 offen offset: 0 lds // (op 1039) Load value direct to lds
v_add_co_u32 v66, s[78:79], v66, v60 // (op 1039) least significant half
v_addc_co_u32 v67, s[78:79], 0, v67, s[78:79] // (op 1039) most significant half

s_add_u32 m0, m0, 1024 // (op 1039) 
buffer_load_dwordx4 v66, s[60:63], 0 offen offset: 0 lds // (op 1039) Load value direct to lds





v_mov_b32 v30, v27 // (op 17) 
s_lshl_b32 s76, s32, 3 // (op 17) 
v_add_u32 v59, v30, s76 // (op 17) 


buffer_load_ubyte v39, v59, s[68:71], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 2872) 



s_waitcnt vmcnt(9)
ds_write_b8 v58, v28 offset:8320 // (op 706) Store local data 
s_barrier  // (op 2866) 



s_waitcnt vmcnt(6)
ds_write_b8 v57, v37 offset:8384 // (op 705) Store local data 
s_barrier  // (op 2853) 



ds_read_b64_tr_b8 v[40:41], v14 offset:4096 // (op 698) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v14 offset:4224 // (op 698) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v14 offset:5120 // (op 698) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v14 offset:5248 // (op 698) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2869) 


ds_read_u8 v23, v19 offset:8320 // (op 702) Load local data 
s_barrier  // (op 2844) 
s_barrier  // (op 3025) 



ds_read_b64_tr_b8 v[48:49], v12 offset:6144 // (op 696) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v12 offset:6272 // (op 696) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v12 offset:7168 // (op 696) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v12 offset:7296 // (op 696) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2863) 


ds_read_u8 v25, v17 offset:8384 // (op 700) Load local data 
s_barrier  // (op 3019) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[40:47], a[0:3], v25, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 703) 
s_barrier  // (op 2860) 

s_lshl_b64 s[76:77], s[28:29], 6 // (op 1100) 
v_mov_b64 v[60:61], s[76:77] // (op 1100) convert


s_lshl_b64 s[76:77], s[28:29], 7 // (op 1100) 
s_mul_i32 s79, 3, s77 // (op 1100) most significant: low of low * high
s_mul_i32 s78, 3, s76 // (op 1100) least significant
s_mul_hi_u32 s80, 3, s76 // (op 1100) most significant: high of low * low
s_add_u32 s79, s79, s80 // (op 1100) most significant: sum


v_mov_b32 v59, s78 // (op 1100) 
v_mov_b32 v66, s79 // (op 1100) 
v_add_co_u32 v68, s[76:77], v62, v59 // (op 1100) least significant half
v_addc_co_u32 v69, s[76:77], v63, v66, s[76:77] // (op 1100) most significant half




s_add_u32 s76, 4096, s55 // (op 1100) 
s_nop 0
s_mov_b32 m0, s76 // (op 1100) reg expression
buffer_load_dwordx4 v68, s[64:67], 0 offen offset: 0 lds // (op 1100) Load value direct to lds
v_add_co_u32 v68, s[78:79], v68, v60 // (op 1100) least significant half
v_addc_co_u32 v69, s[78:79], 0, v69, s[78:79] // (op 1100) most significant half

s_add_u32 m0, m0, 1024 // (op 1100) 
buffer_load_dwordx4 v68, s[64:67], 0 offen offset: 0 lds // (op 1100) Load value direct to lds






s_lshl_b32 s76, s38, 2 // (op 709) 
s_mul_i32 s77, 3, s76 // (op 709) 

v_add_u32 v59, v29, s77 // (op 709) 


buffer_load_ubyte v28, v59, s[72:75], 0 offen offset: 0 // (op 709) Load value

s_lshl_b64 s[76:77], s[28:29], 7 // (op 1336) 

s_lshl_b64 s[78:79], s[76:77], 1 // (op 1336) 

s_barrier  // (op 2850) 
s_lshl_b32 s76, s38, 3 // (op 1592) 

v_add_u32 v36, v36, s76 // (op 1592) 

v_mov_b32 v29, s78 // (op 1336) 
v_mov_b32 v59, s79 // (op 1336) 
s_lshl_b64 s[76:77], s[24:25], 6 // (op 1095) 
v_mov_b64 v[60:61], s[76:77] // (op 1095) convert


s_lshl_b64 s[76:77], s[24:25], 7 // (op 1095) 
s_mul_i32 s81, 3, s77 // (op 1095) most significant: low of low * high
s_mul_i32 s80, 3, s76 // (op 1095) least significant
s_mul_hi_u32 s82, 3, s76 // (op 1095) most significant: high of low * low
s_add_u32 s81, s81, s82 // (op 1095) most significant: sum


v_mov_b32 v62, s80 // (op 1095) 
v_mov_b32 v63, s81 // (op 1095) 
v_add_co_u32 v34, s[76:77], v34, v29 // (op 1336) least significant half
v_addc_co_u32 v35, s[76:77], v35, v59, s[76:77] // (op 1336) most significant half




v_add_co_u32 v66, s[76:77], v64, v62 // (op 1095) least significant half
v_addc_co_u32 v67, s[76:77], v65, v63, s[76:77] // (op 1095) most significant half




s_add_u32 s76, 6144, s54 // (op 1095) 
s_nop 0
s_mov_b32 m0, s76 // (op 1095) reg expression
buffer_load_dwordx4 v66, s[60:63], 0 offen offset: 0 lds // (op 1095) Load value direct to lds
v_add_co_u32 v66, s[78:79], v66, v60 // (op 1095) least significant half
v_addc_co_u32 v67, s[78:79], 0, v67, s[78:79] // (op 1095) most significant half

s_add_u32 m0, m0, 1024 // (op 1095) 
buffer_load_dwordx4 v66, s[60:63], 0 offen offset: 0 lds // (op 1095) Load value direct to lds






s_lshl_b32 s76, s32, 2 // (op 708) 
s_mul_i32 s77, 3, s76 // (op 708) 

v_add_u32 v29, v30, s77 // (op 708) 


buffer_load_ubyte v37, v29, s[68:71], 0 offen offset: 0 // (op 708) Load value

s_lshl_b64 s[76:77], s[24:25], 7 // (op 1205) 

s_lshl_b64 s[78:79], s[76:77], 1 // (op 1205) 

s_barrier  // (op 2720) 
s_lshl_b32 s76, s32, 3 // (op 1464) 



v_add_u32 v27, v27, s76 // (op 1464) 

v_mov_b32 v29, s78 // (op 1205) 
v_mov_b32 v30, s79 // (op 1205) 
v_add_co_u32 v32, s[76:77], v32, v29 // (op 1205) least significant half
v_addc_co_u32 v33, s[76:77], v33, v30, s[76:77] // (op 1205) most significant half




s_waitcnt vmcnt(9)
ds_write_b8 v56, v38 offset:8192 // (op 30) Store local data 
s_barrier  // (op 2700) 



s_waitcnt vmcnt(6)
ds_write_b8 v31, v39 offset:8256 // (op 21) Store local data 
s_barrier  // (op 2686) 



ds_read_b64_tr_b8 v[40:41], v3 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v3 offset:128 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v3 offset:1024 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v3 offset:1152 // (op 77) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2727) 


ds_read_u8 v23, v7 offset:8192 // (op 87) Load local data 
s_barrier  // (op 2667) 



ds_read_b64_tr_b8 v[48:49], v0 offset:2048 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:2176 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:3072 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:3200 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2707) 


ds_read_u8 v25, v5 offset:8256 // (op 82) Load local data 
s_and_b32 s76, s7, 4294967294 // (op 60) 
s_cmp_lt_u32 s1, s76 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoop_60 if true)
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_60:














s_barrier  // (op 2885) 
s_and_b32 s52, s7, 4294967294 // (op 1168) 
s_add_u32 s53, s0, s52 // (op 1168) 

s_mul_i32 s52, s16, s8 // (op 1168) 
s_add_u32 s52, s52, s53 // (op 1168) 

s_mul_hi_u32 s53, s52, s13 // (op 1168) 
s_sub_u32 s54, s52, s53 // (op 1168) 
s_lshr_b32 s55, s54, 1 // (op 1168) 

s_add_u32 s54, s55, s53 // (op 1168) 


s_lshr_b32 s53, s54, s11 // (op 1168) 

s_and_b32 s54, s11, 2147483648 // (op 1168) 
s_cmp_eq_i32 s54, 0 // (op 1168) 
s_mov_b32 s55, scc // (op 1168) 

s_cmp_eq_i32 s55, 0 // (op 1168) 
s_cselect_b32 s54, s52, s53 // (op 1168) 


s_mul_hi_u32 s53, s54, s12 // (op 1168) 
s_sub_u32 s55, s54, s53 // (op 1168) 
s_lshr_b32 s76, s55, 1 // (op 1168) 

s_add_u32 s55, s76, s53 // (op 1168) 


s_lshr_b32 s53, s55, s10 // (op 1168) 

s_and_b32 s55, s10, 2147483648 // (op 1168) 
s_cmp_eq_i32 s55, 0 // (op 1168) 
s_mov_b32 s76, scc // (op 1168) 

s_cmp_eq_i32 s76, 0 // (op 1168) 
s_cselect_b32 s55, s54, s53 // (op 1168) 


s_lshl_b32 s53, s55, 4 // (op 1168) 

s_mov_b32 s77, 0 // (op 1168) convert
s_mov_b32 s76, s53 // (op 1168) convert

s_ashr_i64 s[78:79], s[2:3], 7 // (op 1168) 
s_mul_i32 s53, s54, s78 // (op 1168) 


s_sub_u32 s54, s52, s53 // (op 1168) 


v_lshl_add_u32 v27, s54, 7, v1 // (op 1168) 

v_mul_lo_u32 v39, v27, s25 // (op 1168) most significant: low of low * high
v_mul_hi_u32 v29, v27, s24 // (op 1168) most significant: high of low * low
v_mul_lo_u32 v38, v27, s24 // (op 1168) least significant: low of low * low
v_add_u32 v39, v39, v29 // (op 1168) most significant: sum


v_mov_b32 v27, s76 // (op 1168) 
v_mov_b32 v29, s77 // (op 1168) 
v_add_co_u32 v30, s[52:53], v27, v38 // (op 1168) least significant half
v_addc_co_u32 v31, s[52:53], v29, v39, s[52:53] // (op 1168) most significant half





s_mul_i32 s53, s24, s3 // (op 1170) most significant: low of low * high
s_mul_i32 s54, s25, s2 // (op 1170) most significant: low of high * low
s_mul_i32 s52, s24, s2 // (op 1170) least significant
s_mul_hi_u32 s55, s24, s2 // (op 1170) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1170) most significant: sum
s_add_u32 s53, s53, s55 // (op 1170) most significant: sum


s_mov_b32 s60, s26 // (op 1170) reg expression
s_mov_b32 s61, s27 // (op 1170) reg expression
s_mov_b32 s62, s52 // (op 1170) reg expression
s_mov_b32 s63, 131072 // (op 1170) call()

s_and_b32 s52, s7, 4294967294 // (op 1302) 
s_add_u32 s53, s0, s52 // (op 1302) 

s_mul_i32 s52, s16, s8 // (op 1302) 
s_add_u32 s52, s52, s53 // (op 1302) 

s_mul_hi_u32 s53, s52, s13 // (op 1302) 
s_sub_u32 s54, s52, s53 // (op 1302) 
s_lshr_b32 s55, s54, 1 // (op 1302) 

s_add_u32 s54, s55, s53 // (op 1302) 


s_lshr_b32 s53, s54, s11 // (op 1302) 

s_and_b32 s54, s11, 2147483648 // (op 1302) 
s_cmp_eq_i32 s54, 0 // (op 1302) 
s_mov_b32 s55, scc // (op 1302) 

s_cmp_eq_i32 s55, 0 // (op 1302) 
s_cselect_b32 s54, s52, s53 // (op 1302) 


s_mul_hi_u32 s53, s54, s12 // (op 1302) 
s_sub_u32 s55, s54, s53 // (op 1302) 
s_lshr_b32 s76, s55, 1 // (op 1302) 

s_add_u32 s55, s76, s53 // (op 1302) 


s_lshr_b32 s53, s55, s10 // (op 1302) 

s_and_b32 s55, s10, 2147483648 // (op 1302) 
s_cmp_eq_i32 s55, 0 // (op 1302) 
s_mov_b32 s76, scc // (op 1302) 

s_cmp_eq_i32 s76, 0 // (op 1302) 
s_cselect_b32 s55, s54, s53 // (op 1302) 


s_mul_i32 s53, s55, s14 // (op 1302) 

s_sub_u32 s55, s54, s53 // (op 1302) 

s_lshl_b32 s53, s55, 4 // (op 1302) 

s_mov_b32 s77, 0 // (op 1302) convert
s_mov_b32 s76, s53 // (op 1302) convert

s_ashr_i64 s[78:79], s[2:3], 7 // (op 1302) 
s_mul_i32 s53, s54, s78 // (op 1302) 


s_sub_u32 s54, s52, s53 // (op 1302) 


v_lshl_add_u32 v27, s54, 7, v1 // (op 1302) 

v_mul_lo_u32 v39, v27, s29 // (op 1302) most significant: low of low * high
v_mul_hi_u32 v29, v27, s28 // (op 1302) most significant: high of low * low
v_mul_lo_u32 v38, v27, s28 // (op 1302) least significant: low of low * low
v_add_u32 v39, v39, v29 // (op 1302) most significant: sum

v_mov_b32 v29, s76 // (op 1302) 
v_mov_b32 v32, s77 // (op 1302) 
v_add_co_u32 v38, s[52:53], v38, v29 // (op 1302) least significant half
v_addc_co_u32 v39, s[52:53], v39, v32, s[52:53] // (op 1302) most significant half





s_mul_i32 s53, s28, s3 // (op 1304) most significant: low of low * high
s_mul_i32 s54, s29, s2 // (op 1304) most significant: low of high * low
s_mul_i32 s52, s28, s2 // (op 1304) least significant
s_mul_hi_u32 s55, s28, s2 // (op 1304) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1304) most significant: sum
s_add_u32 s53, s53, s55 // (op 1304) most significant: sum


s_mov_b32 s64, s30 // (op 1304) reg expression
s_mov_b32 s65, s31 // (op 1304) reg expression
s_mov_b32 s66, s52 // (op 1304) reg expression
s_mov_b32 s67, 131072 // (op 1304) call()

s_and_b32 s52, s7, 4294967294 // (op 1430) 
v_and_b32 v27, -16, v1 // (op 1430) 
s_add_u32 s53, s0, s52 // (op 1430) 

v_sub_u32 v29, v1, v27 // (op 1430) 

v_lshrrev_b32 v27, 4, v1 // (op 1430) 
s_mul_i32 s52, s16, s8 // (op 1430) 
s_add_u32 s52, s52, s53 // (op 1430) 

s_mul_hi_u32 s53, s52, s13 // (op 1430) 
s_sub_u32 s54, s52, s53 // (op 1430) 
s_lshr_b32 s55, s54, 1 // (op 1430) 

s_add_u32 s54, s55, s53 // (op 1430) 


s_lshr_b32 s53, s54, s11 // (op 1430) 

s_and_b32 s54, s11, 2147483648 // (op 1430) 
s_cmp_eq_i32 s54, 0 // (op 1430) 
s_mov_b32 s55, scc // (op 1430) 

s_cmp_eq_i32 s55, 0 // (op 1430) 
s_cselect_b32 s54, s52, s53 // (op 1430) 


s_mul_hi_u32 s53, s54, s12 // (op 1430) 
s_sub_u32 s55, s54, s53 // (op 1430) 
s_lshr_b32 s76, s55, 1 // (op 1430) 

s_add_u32 s55, s76, s53 // (op 1430) 


s_lshr_b32 s53, s55, s10 // (op 1430) 

s_and_b32 s55, s10, 2147483648 // (op 1430) 
s_cmp_eq_i32 s55, 0 // (op 1430) 
s_mov_b32 s76, scc // (op 1430) 

s_cmp_eq_i32 s76, 0 // (op 1430) 
s_cselect_b32 s55, s54, s53 // (op 1430) 


v_lshl_add_u32 v32, s55, 4, v29 // (op 1430) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1430) 
s_mul_i32 s53, s54, s76 // (op 1430) 


s_sub_u32 s54, s52, s53 // (op 1430) 


v_lshl_add_u32 v29, s54, 2, v27 // (op 1430) 


v_mul_lo_u32 v27, v29, s32 // (op 1430) 

v_add_u32 v29, v32, v27 // (op 1430) 


s_mul_i32 s53, s32, s37 // (op 1432) most significant: low of low * high
s_mul_i32 s54, s33, s36 // (op 1432) most significant: low of high * low
s_mul_i32 s52, s32, s36 // (op 1432) least significant
s_mul_hi_u32 s55, s32, s36 // (op 1432) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1432) most significant: sum
s_add_u32 s53, s53, s55 // (op 1432) most significant: sum


s_mov_b32 s68, s34 // (op 1432) reg expression
s_mov_b32 s69, s35 // (op 1432) reg expression
s_mov_b32 s70, s52 // (op 1432) reg expression
s_mov_b32 s71, 131072 // (op 1432) call()

s_and_b32 s52, s7, 4294967294 // (op 1558) 
v_and_b32 v27, -16, v1 // (op 1558) 
s_add_u32 s53, s0, s52 // (op 1558) 

v_sub_u32 v32, v1, v27 // (op 1558) 

v_lshrrev_b32 v27, 4, v1 // (op 1558) 
s_mul_i32 s52, s16, s8 // (op 1558) 
s_add_u32 s52, s52, s53 // (op 1558) 

s_mul_hi_u32 s53, s52, s13 // (op 1558) 
s_sub_u32 s54, s52, s53 // (op 1558) 
s_lshr_b32 s55, s54, 1 // (op 1558) 

s_add_u32 s54, s55, s53 // (op 1558) 


s_lshr_b32 s53, s54, s11 // (op 1558) 

s_and_b32 s54, s11, 2147483648 // (op 1558) 
s_cmp_eq_i32 s54, 0 // (op 1558) 
s_mov_b32 s55, scc // (op 1558) 

s_cmp_eq_i32 s55, 0 // (op 1558) 
s_cselect_b32 s54, s52, s53 // (op 1558) 


s_mul_hi_u32 s53, s54, s12 // (op 1558) 
s_sub_u32 s55, s54, s53 // (op 1558) 
s_lshr_b32 s76, s55, 1 // (op 1558) 

s_add_u32 s55, s76, s53 // (op 1558) 


s_lshr_b32 s53, s55, s10 // (op 1558) 

s_and_b32 s55, s10, 2147483648 // (op 1558) 
s_cmp_eq_i32 s55, 0 // (op 1558) 
s_mov_b32 s76, scc // (op 1558) 

s_cmp_eq_i32 s76, 0 // (op 1558) 
s_cselect_b32 s55, s54, s53 // (op 1558) 


s_mul_i32 s53, s55, s14 // (op 1558) 

s_sub_u32 s55, s54, s53 // (op 1558) 

v_lshl_add_u32 v33, s55, 4, v32 // (op 1558) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1558) 
s_mul_i32 s53, s54, s76 // (op 1558) 


s_sub_u32 s54, s52, s53 // (op 1558) 


v_lshl_add_u32 v32, s54, 2, v27 // (op 1558) 


v_mul_lo_u32 v27, v32, s38 // (op 1558) 
v_add_u32 v27, v27, v33 // (op 1558) 


s_mul_i32 s53, s38, s37 // (op 1560) most significant: low of low * high
s_mul_i32 s54, s39, s36 // (op 1560) most significant: low of high * low
s_mul_i32 s52, s38, s36 // (op 1560) least significant
s_mul_hi_u32 s55, s38, s36 // (op 1560) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1560) most significant: sum
s_add_u32 s53, s53, s55 // (op 1560) most significant: sum


s_mov_b32 s72, s40 // (op 1560) reg expression
s_mov_b32 s73, s41 // (op 1560) reg expression
s_mov_b32 s74, s52 // (op 1560) reg expression
s_mov_b32 s75, 131072 // (op 1560) call()

v_lshlrev_b32 v32, 4, v1 // (op 1709) 
s_nop 0
v_readfirstlane_b32 s52, v32 // (op 1709) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v32, 4, v1 // (op 1774) 
s_nop 0
v_readfirstlane_b32 s53, v32 // (op 1774) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v32, -16, v1 // (op 1839) 
v_sub_u32 v33, v1, v32 // (op 1839) 

v_and_b32 v32, 4294967280, v1 // (op 1839) 
v_add_u32 v34, v32, v33 // (op 1839) 


v_and_b32 v32, -16, v1 // (op 1905) 
v_sub_u32 v33, v1, v32 // (op 1905) 

v_and_b32 v32, 4294967280, v1 // (op 1905) 
v_add_u32 v35, v32, v33 // (op 1905) 


s_and_b32 s54, s7, 4294967294 // (op 623) 
s_cmp_lt_u32 s54, s7 // (op 622) 






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_622 // (op 622) Condition: Top (jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_622 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_622:
s_barrier  // (op 2664) 
s_add_u32 s54, s54, 1 // (op 624) 

s_lshl_b64 s[76:77], s[24:25], 6 // (op 1082) 
v_mov_b64 v[32:33], s[76:77] // (op 1082) convert


v_mov_b64 v[36:37], v[30:31] // (op 1082) 
v_mov_b64 v[40:41], v[36:37] // (op 1082) reg expression
s_add_u32 s55, 2048, s52 // (op 1082) 
s_mov_b32 m0, s55 // (op 1082) reg expression
buffer_load_dwordx4 v40, s[60:63], 0 offen offset: 0 lds // (op 1082) Load value direct to lds
v_add_co_u32 v40, s[76:77], v40, v32 // (op 1082) least significant half
v_addc_co_u32 v41, s[76:77], 0, v41, s[76:77] // (op 1082) most significant half

s_add_u32 m0, m0, 1024 // (op 1082) 
buffer_load_dwordx4 v40, s[60:63], 0 offen offset: 0 lds // (op 1082) Load value direct to lds




s_lshl_b64 s[76:77], s[24:25], 7 // (op 1158) 
v_mov_b32 v23, s76 // (op 1158) 
v_mov_b32 v25, s77 // (op 1158) 
v_add_co_u32 v30, s[78:79], v30, v23 // (op 1158) least significant half
v_addc_co_u32 v31, s[78:79], v31, v25, s[78:79] // (op 1158) most significant half




s_barrier  // (op 2683) 

s_lshl_b64 s[76:77], s[28:29], 6 // (op 1088) 
v_mov_b64 v[32:33], s[76:77] // (op 1088) convert


v_mov_b64 v[36:37], v[38:39] // (op 1088) 
v_mov_b64 v[40:41], v[36:37] // (op 1088) reg expression
s_mov_b32 m0, s53 // (op 1088) reg expression
buffer_load_dwordx4 v40, s[64:67], 0 offen offset: 0 lds // (op 1088) Load value direct to lds
v_add_co_u32 v40, s[76:77], v40, v32 // (op 1088) least significant half
v_addc_co_u32 v41, s[76:77], 0, v41, s[76:77] // (op 1088) most significant half

s_add_u32 m0, m0, 1024 // (op 1088) 
buffer_load_dwordx4 v40, s[64:67], 0 offen offset: 0 lds // (op 1088) Load value direct to lds



s_lshl_b64 s[76:77], s[28:29], 7 // (op 1294) 
v_mov_b32 v23, s76 // (op 1294) 
v_mov_b32 v25, s77 // (op 1294) 
v_add_co_u32 v38, s[78:79], v38, v23 // (op 1294) least significant half
v_addc_co_u32 v39, s[78:79], v39, v25, s[78:79] // (op 1294) most significant half






v_mov_b32 v23, v29 // (op 642) 
v_mov_b32 v25, v23 // (op 642) reg expression

buffer_load_ubyte v28, v25, s[68:71], 0 offen offset: 0 // (op 642) Load value


s_lshl_b32 s55, s32, 2 // (op 1422) 
v_add_u32 v29, v29, s55 // (op 1422) 



s_barrier  // (op 2703) 
v_mov_b32 v23, v27 // (op 643) 



v_mov_b32 v25, v23 // (op 643) reg expression

buffer_load_ubyte v32, v25, s[72:75], 0 offen offset: 0 // (op 643) Load value


s_lshl_b32 s55, s38, 2 // (op 1550) 
v_add_u32 v27, v27, s55 // (op 1550) 

s_waitcnt vmcnt(1)
ds_write_b8 v34, v28 offset:8256 // (op 639) Store local data 

s_barrier  // (op 2670) 



s_barrier  // (op 2723) 
ds_read_b64_tr_b8 v[40:41], v0 offset:2048 // (op 630) Transpose load local data 



ds_read_b64_tr_b8 v[42:43], v0 offset:2176 // (op 630) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:3072 // (op 630) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:3200 // (op 630) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v35, v32 offset:8192 // (op 640) Store local data 

s_barrier  // (op 2690) 



ds_read_b64_tr_b8 v[48:49], v3 // (op 632) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v3 offset:128 // (op 632) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v3 offset:1024 // (op 632) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v3 offset:1152 // (op 632) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2710) 


ds_read_u8 v23, v5 offset:8256 // (op 634) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2730) 


ds_read_u8 v25, v7 offset:8192 // (op 636) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[48:55], a[0:3], v23, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 637) 




s_cmp_lt_u32 s54, s7 // (op 622) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_622 // (op 622) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_622 if true)
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_622:









s_barrier  // (op 3011) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_lg_u32 s18, 0 // (op 322) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Send_Tile_322 // (op 322) Condition: False, jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Send_Tile_322
v_lshrrev_b32 v20, 4, v1 // (op 2292) 
v_lshlrev_b32 v22, 2, v20 // (op 2292) 

v_lshl_add_u32 v20, s16, 4, v22 // (op 2292) 

v_and_b32 v22, -16, v1 // (op 2292) 
v_sub_u32 v23, v1, v22 // (op 2292) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 2292) 


v_lshlrev_b32 v20, 2, v22 // (op 2292) 

s_lshl_b32 s52, s15, 10 // (op 2294) 
s_mov_b32 s76, s42 // (op 2294) reg expression
s_mov_b32 s77, s43 // (op 2294) reg expression
s_mov_b32 s78, s52 // (op 2294) reg expression
s_mov_b32 s79, 131072 // (op 2294) call()




buffer_store_dword a0, v20, s[76:79], 0 offen sc1 // (op 319) Store value
buffer_store_dword a1, v20, s[76:79], 0 offen offset:64 sc1 // (op 319) Store value
buffer_store_dword a2, v20, s[76:79], 0 offen offset:128 sc1 // (op 319) Store value
buffer_store_dword a3, v20, s[76:79], 0 offen offset:192 sc1 // (op 319) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 323) 
v_cmp_eq_i32 vcc, v1, 0 // (op 327) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_327 // (op 327) Condition: False, jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_327
s_mov_b32 s52, 1 // (op 325) call()
s_mov_b32 s55, 0 // (op 326) convert
s_mov_b32 s54, s16 // (op 326) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 326) 
s_add_u32 s80, s44, s54 // (op 326) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 326) End of Int64 add, unlocking SCC
s_store_dword s52, s[80:81], 0 glc // (op 326) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_327 // (op 327) Condition: Done, jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_327
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_327:
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_327:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Send_Tile_322 // (op 322) Condition: Done, jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Send_Tile_322
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Send_Tile_322:
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Send_Tile_322:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 338) 
s_mov_b32 s52, scc // (op 338) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 338) 
s_mov_b32 s53, s54 // (op 338) convert

s_sub_u32 s54, s53, 1 // (op 338) 

s_cmp_lt_u32 s19, s54 // (op 338) 
s_mov_b32 s53, scc // (op 338) 

s_and_b32 s54, s52, s53 // (op 338) 
s_cmp_lg_u32 s54, 0 // (op 338) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Receive_Tile_338 // (op 338) Condition: False, jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Receive_Tile_338
s_mov_b32 s52, 0 // (op 304) call()
s_mul_i32 s53, s52, s8 // (op 303) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 303) 
s_mov_b32 s80, s54 // (op 303) convert

s_sub_u32 s54, s80, s19 // (op 303) 

s_sub_u32 s55, s54, 1 // (op 303) 

s_cmp_lt_u32 s53, s55 // (op 303) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_303 // (op 303) Condition: Top (jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_303 if false)
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_ReceiveTileLoop_303:
s_add_u32 s53, s16, 1 // (op 2309) 
v_lshrrev_b32 v20, 4, v1 // (op 2309) 
s_add_u32 s54, s53, s52 // (op 2309) 
s_lshl_b32 s54, s54, 4 // (op 2309) 

v_lshlrev_b32 v22, 2, v20 // (op 2309) 

v_add_lshl_u32 v20, s54, v22, 4 // (op 2309) 


v_and_b32 v22, -16, v1 // (op 2309) 
v_sub_u32 v23, v1, v22 // (op 2309) 

v_add_lshl_u32 v22, v20, v23, 2 // (op 2309) 


s_lshl_b32 s53, s15, 10 // (op 2311) 
s_mov_b32 s76, s42 // (op 2311) reg expression
s_mov_b32 s77, s43 // (op 2311) reg expression
s_mov_b32 s78, s53 // (op 2311) reg expression
s_mov_b32 s79, 131072 // (op 2311) call()

RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_342:
s_add_u32 s53, s16, 1 // (op 340) 
s_add_u32 s54, s53, s52 // (op 340) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 340) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 340) 

s_add_u32 s80, s44, s54 // (op 340) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 340) End of Int64 add, unlocking SCC
s_load_dword s53, s[80:81], 0 glc // (op 340) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 342) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_342 // (op 342) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_342 if true)
s_barrier  // (op 346) 
v_cmp_eq_i32 vcc, v1, 0 // (op 345) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_345 // (op 345) Condition: False, jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_345
s_mov_b32 s53, 0 // (op 343) call()
s_add_u32 s54, s16, 1 // (op 344) 
s_add_u32 s80, s54, s52 // (op 344) Start of Int64 add, locking SCC
s_addc_u32 s81, 0, 0 // (op 344) End of Int64 add, unlocking SCC
s_lshl_b64 s[80:81], s[80:81], 2 // (op 344) 

s_add_u32 s54, s44, s80 // (op 344) Start of Int64 add, locking SCC
s_addc_u32 s55, s45, s81 // (op 344) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 344) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_345 // (op 345) Condition: Done, jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_345
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_345:
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_345:




buffer_load_dword v24, v22, s[76:79], 0 offen // (op 357) Load value
buffer_load_dword v25, v22, s[76:79], 0 offen offset:64 // (op 357) Load value
buffer_load_dword v26, v22, s[76:79], 0 offen offset:128 // (op 357) Load value
buffer_load_dword v27, v22, s[76:79], 0 offen offset:192 // (op 357) Load value
v_accvgpr_read v28, a0 // (op 360) reg expression
v_accvgpr_read v29, a1 // (op 360) reg expression
v_accvgpr_read v30, a2 // (op 360) reg expression
v_accvgpr_read v31, a3 // (op 360) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 362) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 362) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 362) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 362) 

v_accvgpr_write a0, v28 // (op 364) reg expression
v_accvgpr_write a1, v29 // (op 364) reg expression
v_accvgpr_write a2, v30 // (op 364) reg expression
v_accvgpr_write a3, v31 // (op 364) reg expression

s_add_u32 s52, s52, 1 // (op 305) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 303) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 303) 
s_mov_b32 s80, s54 // (op 303) convert

s_sub_u32 s54, s80, s19 // (op 303) 

s_sub_u32 s55, s54, 1 // (op 303) 

s_cmp_lt_u32 s53, s55 // (op 303) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_ReceiveTileLoop_303 // (op 303) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_ReceiveTileLoop_303 if true)
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_303:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Receive_Tile_338 // (op 338) Condition: Done, jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Receive_Tile_338
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Receive_Tile_338:
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Receive_Tile_338:

s_cmp_eq_i32 s18, 0 // (op 381) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_381 // (op 381) Condition: False, jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_381
v_cmp_eq_f32 vcc, 0.00000, v16 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_135
v_accvgpr_read v22, a0 // (op 50) 
v_accvgpr_read v23, a1 // (op 50) 
v_accvgpr_read v24, a2 // (op 50) 
v_accvgpr_read v25, a3 // (op 50) 
v_mul_f32 v26, v18, v22 // (op 50) 
v_mul_f32 v27, v18, v23 // (op 50) 
v_mul_f32 v28, v18, v24 // (op 50) 
v_mul_f32 v29, v18, v25 // (op 50) 

v_cvt_f16_f32 v22, v26 // (op 56) 
v_cvt_f16_f32 v23, v27 // (op 56) 
v_cvt_f16_f32 v24, v28 // (op 56) 
v_cvt_f16_f32 v25, v29 // (op 56) 

v_lshrrev_b32 v20, 6, v1 // (op 1960) 
s_mul_i32 s52, s16, s8 // (op 1960) 
s_add_u32 s52, s52, s0 // (op 1960) 
v_and_b32 v26, 63, v1 // (op 1960) 
s_mul_hi_u32 s53, s52, s13 // (op 1960) 
v_lshrrev_b32 v27, 4, v26 // (op 1960) 
s_sub_u32 s54, s52, s53 // (op 1960) 
v_lshlrev_b32 v28, 2, v27 // (op 1960) 

s_lshr_b32 s55, s54, 1 // (op 1960) 

v_lshl_add_u32 v27, v20, 4, v28 // (op 1960) 


s_add_u32 s54, s55, s53 // (op 1960) 


v_lshrrev_b32 v20, 2, v26 // (op 1960) 

s_lshr_b32 s53, s54, s11 // (op 1960) 

v_and_b32 v26, 3, v20 // (op 1960) 

s_and_b32 s54, s11, 2147483648 // (op 1960) 
v_and_b32 v20, 3, v1 // (op 1960) 
s_cmp_eq_i32 s54, 0 // (op 1960) 
s_mov_b32 s55, scc // (op 1960) 

v_lshl_add_u32 v28, v26, 2, v20 // (op 1960) 


s_cmp_eq_i32 s55, 0 // (op 1960) 
s_cselect_b32 s54, s52, s53 // (op 1960) 



s_mul_hi_u32 s55, s54, s12 // (op 1960) 
s_sub_u32 s52, s54, s55 // (op 1960) 
s_lshr_b32 s53, s52, 1 // (op 1960) 

s_add_u32 s52, s53, s55 // (op 1960) 


s_lshr_b32 s53, s52, s10 // (op 1960) 

s_and_b32 s52, s10, 2147483648 // (op 1960) 
s_cmp_eq_i32 s52, 0 // (op 1960) 
s_mov_b32 s55, scc // (op 1960) 

s_cmp_eq_i32 s55, 0 // (op 1960) 
s_cselect_b32 s52, s54, s53 // (op 1960) 


s_mul_i32 s53, s52, s14 // (op 1960) 
v_lshl_add_u32 v20, s52, 4, v27 // (op 1960) 


s_sub_u32 s52, s54, s53 // (op 1960) 


v_lshl_add_u32 v26, s52, 4, v28 // (op 1960) 


v_mul_lo_u32 v27, v26, s46 // (op 1960) 

v_add_lshl_u32 v26, v20, v27, 1 // (op 1960) 


s_mul_i32 s53, s46, s23 // (op 1962) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1962) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1962) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1962) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1962) most significant: sum
s_add_u32 s53, s53, s55 // (op 1962) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 1962) 

s_mov_b32 s76, s48 // (op 1962) reg expression
s_mov_b32 s77, s49 // (op 1962) reg expression
s_mov_b32 s78, s54 // (op 1962) reg expression
s_mov_b32 s79, 131072 // (op 1962) call()


v_pack_b32_f16 v28, v22, v23 // (op 58) 
v_pack_b32_f16 v29, v24, v25 // (op 58) 


buffer_store_dwordx2 v[28:29], v26, s[76:79], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_135
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_135:



buffer_load_dwordx2 v[22:23], v21, s[56:59], 0 offen // (op 148) Load value
v_accvgpr_read v24, a0 // (op 144) 
v_accvgpr_read v25, a1 // (op 144) 
v_accvgpr_read v26, a2 // (op 144) 
v_accvgpr_read v27, a3 // (op 144) 
v_mul_f32 v28, v18, v24 // (op 144) 
v_mul_f32 v29, v18, v25 // (op 144) 
v_mul_f32 v30, v18, v26 // (op 144) 
v_mul_f32 v31, v18, v27 // (op 144) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v24, v22 // (op 149) 
v_lshrrev_b32 v25, 16, v22 // (op 149) 
v_cvt_f32_f16 v25, v25 // (op 149) 
v_mul_f32 v32, v16, v24 // (op 149) 
v_mul_f32 v33, v16, v25 // (op 149) 
v_cvt_f32_f16 v24, v23 // (op 149) 
v_lshrrev_b32 v25, 16, v23 // (op 149) 
v_cvt_f32_f16 v25, v25 // (op 149) 
v_mul_f32 v34, v16, v24 // (op 149) 
v_mul_f32 v35, v16, v25 // (op 149) 


v_add_f32 v22, v32, v28 // (op 145) 
v_add_f32 v23, v33, v29 // (op 145) 
v_add_f32 v24, v34, v30 // (op 145) 
v_add_f32 v25, v35, v31 // (op 145) 


v_cvt_f16_f32 v26, v22 // (op 146) 
v_cvt_f16_f32 v27, v23 // (op 146) 
v_cvt_f16_f32 v28, v24 // (op 146) 
v_cvt_f16_f32 v29, v25 // (op 146) 

v_lshrrev_b32 v20, 6, v1 // (op 1977) 
s_mul_i32 s52, s16, s8 // (op 1977) 
s_add_u32 s52, s52, s0 // (op 1977) 
v_and_b32 v22, 63, v1 // (op 1977) 
s_mul_hi_u32 s53, s52, s13 // (op 1977) 
v_lshrrev_b32 v23, 4, v22 // (op 1977) 
s_sub_u32 s54, s52, s53 // (op 1977) 
v_lshlrev_b32 v24, 2, v23 // (op 1977) 

s_lshr_b32 s55, s54, 1 // (op 1977) 

v_lshl_add_u32 v23, v20, 4, v24 // (op 1977) 


s_add_u32 s54, s55, s53 // (op 1977) 


v_lshrrev_b32 v20, 2, v22 // (op 1977) 

s_lshr_b32 s53, s54, s11 // (op 1977) 

v_and_b32 v22, 3, v20 // (op 1977) 

s_and_b32 s54, s11, 2147483648 // (op 1977) 
v_and_b32 v20, 3, v1 // (op 1977) 
s_cmp_eq_i32 s54, 0 // (op 1977) 
s_mov_b32 s55, scc // (op 1977) 

v_lshl_add_u32 v24, v22, 2, v20 // (op 1977) 


s_cmp_eq_i32 s55, 0 // (op 1977) 
s_cselect_b32 s54, s52, s53 // (op 1977) 



s_mul_hi_u32 s55, s54, s12 // (op 1977) 
s_sub_u32 s52, s54, s55 // (op 1977) 
s_lshr_b32 s53, s52, 1 // (op 1977) 

s_add_u32 s52, s53, s55 // (op 1977) 


s_lshr_b32 s53, s52, s10 // (op 1977) 

s_and_b32 s52, s10, 2147483648 // (op 1977) 
s_cmp_eq_i32 s52, 0 // (op 1977) 
s_mov_b32 s55, scc // (op 1977) 

s_cmp_eq_i32 s55, 0 // (op 1977) 
s_cselect_b32 s52, s54, s53 // (op 1977) 


s_mul_i32 s53, s52, s14 // (op 1977) 
v_lshl_add_u32 v20, s52, 4, v23 // (op 1977) 


s_sub_u32 s52, s54, s53 // (op 1977) 


v_lshl_add_u32 v25, s52, 4, v24 // (op 1977) 


v_mul_lo_u32 v22, v25, s46 // (op 1977) 

v_add_lshl_u32 v23, v20, v22, 1 // (op 1977) 


s_mul_i32 s53, s46, s23 // (op 1979) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1979) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1979) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1979) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1979) most significant: sum
s_add_u32 s53, s53, s55 // (op 1979) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 1979) 

s_mov_b32 s76, s48 // (op 1979) reg expression
s_mov_b32 s77, s49 // (op 1979) reg expression
s_mov_b32 s78, s54 // (op 1979) reg expression
s_mov_b32 s79, 131072 // (op 1979) call()


v_pack_b32_f16 v24, v26, v27 // (op 147) 
v_pack_b32_f16 v25, v28, v29 // (op 147) 


buffer_store_dwordx2 v[24:25], v23, s[76:79], 0 offen // (op 147) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_135:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_381 // (op 381) Condition: Done, jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_381
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_381:
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_381:

s_add_u32 s0, s0, s7 // (op 296) 
s_mul_i32 s52, s8, s16 // (op 294) 
s_cmp_gt_u32 s9, s52 // (op 294) 
s_mov_b32 s53, scc // (op 294) 
s_sub_u32 s54, s9, s52 // (op 294) 

s_cmp_eq_i32 s53, 0 // (op 294) 
s_cselect_b32 s52, 0, s54 // (op 294) 


s_cmp_lt_u32 s8, s52 // (op 294) 
s_mov_b32 s53, scc // (op 294) 
s_cmp_eq_i32 s53, 0 // (op 294) 
s_cselect_b32 s54, s52, s8 // (op 294) 


s_cmp_lt_u32 s0, s54 // (op 294) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_SKStreamTileLoop_294 // (op 294) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_SKStreamTileLoop_294 if true)
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_294:

















s_mov_b32 s0, 0 // (op 435) call()
s_mov_b32 s1, 0 // (op 436) call()
s_mov_b32 s0, 0 // (op 450) call()
s_mul_i32 s15, s50, s16 // (op 449) 
s_cmp_gt_u32 s51, s15 // (op 449) 
s_mov_b32 s6, scc // (op 449) 
s_sub_u32 s7, s51, s15 // (op 449) 

s_cmp_eq_i32 s6, 0 // (op 449) 
s_cselect_b32 s8, 0, s7 // (op 449) 


s_cmp_lt_u32 s50, s8 // (op 449) 
s_mov_b32 s15, scc // (op 449) 
s_cmp_eq_i32 s15, 0 // (op 449) 
s_cselect_b32 s6, s8, s50 // (op 449) 


s_cmp_lt_u32 s0, s6 // (op 449) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_449 // (op 449) Condition: Top (jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_449 if false)
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_DPStreamTileLoop_449:
v_lshrrev_b32 v0, 6, v1 // (op 1679) 
s_mul_i32 s15, s16, s50 // (op 1679) 
s_add_u32 s15, s15, s0 // (op 1679) 
v_and_b32 v3, 63, v1 // (op 1679) 
s_add_u32 s6, s15, s9 // (op 1679) 

v_lshrrev_b32 v5, 4, v3 // (op 1679) 
s_mul_hi_u32 s15, s6, s13 // (op 1679) 
v_lshlrev_b32 v7, 2, v5 // (op 1679) 

s_sub_u32 s7, s6, s15 // (op 1679) 
v_lshl_add_u32 v5, v0, 4, v7 // (op 1679) 


s_lshr_b32 s8, s7, 1 // (op 1679) 

v_lshrrev_b32 v0, 2, v3 // (op 1679) 

s_add_u32 s7, s8, s15 // (op 1679) 


v_and_b32 v3, 3, v0 // (op 1679) 

s_lshr_b32 s8, s7, s11 // (op 1679) 

v_and_b32 v0, 3, v1 // (op 1679) 
s_and_b32 s7, s11, 2147483648 // (op 1679) 
v_lshl_add_u32 v7, v3, 2, v0 // (op 1679) 


s_cmp_eq_i32 s7, 0 // (op 1679) 
s_mov_b32 s15, scc // (op 1679) 

s_cmp_eq_i32 s15, 0 // (op 1679) 
s_cselect_b32 s7, s6, s8 // (op 1679) 



s_mul_hi_u32 s6, s7, s12 // (op 1679) 
s_sub_u32 s8, s7, s6 // (op 1679) 
s_lshr_b32 s15, s8, 1 // (op 1679) 

s_add_u32 s8, s15, s6 // (op 1679) 


s_lshr_b32 s6, s8, s10 // (op 1679) 

s_and_b32 s8, s10, 2147483648 // (op 1679) 
s_cmp_eq_i32 s8, 0 // (op 1679) 
s_mov_b32 s15, scc // (op 1679) 

s_cmp_eq_i32 s15, 0 // (op 1679) 
s_cselect_b32 s8, s7, s6 // (op 1679) 


s_mul_i32 s6, s8, s14 // (op 1679) 
v_lshl_add_u32 v0, s8, 4, v5 // (op 1679) 


s_sub_u32 s8, s7, s6 // (op 1679) 


v_lshl_add_u32 v3, s8, 4, v7 // (op 1679) 


v_mul_lo_u32 v5, v3, s4 // (op 1679) 

v_add_lshl_u32 v3, v0, v5, 1 // (op 1679) 


s_mul_i32 s19, s4, s23 // (op 1681) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 1681) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 1681) least significant
s_mul_hi_u32 s6, s4, s22 // (op 1681) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1681) most significant: sum
s_add_u32 s19, s19, s6 // (op 1681) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 1 // (op 1681) 

s_mov_b32 s56, s20 // (op 1681) reg expression
s_mov_b32 s57, s21 // (op 1681) reg expression
s_mov_b32 s58, s6 // (op 1681) reg expression
s_mov_b32 s59, 131072 // (op 1681) call()

v_and_b32 v0, 48, v1 // (op 2080) 
v_and_b32 v5, 63, v1 // (op 2080) 
v_sub_u32 v7, v5, v0 // (op 2080) 

v_lshrrev_b32 v5, 1, v7 // (op 2080) 
v_add_lshl_u32 v12, v0, v5, 4 // (op 2080) 


v_and_b32 v0, 1, v7 // (op 2080) 

v_lshlrev_b32 v5, 3, v0 // (op 2080) 

v_lshrrev_b32 v0, 6, v1 // (op 2080) 
v_lshl_add_u32 v7, v0, 4, v5 // (op 2080) 


v_add_u32 v0, v12, v7 // (op 2080) 


v_and_b32 v5, 48, v1 // (op 2145) 
v_and_b32 v7, 63, v1 // (op 2145) 
v_sub_u32 v12, v7, v5 // (op 2145) 

v_lshrrev_b32 v7, 1, v12 // (op 2145) 
v_add_lshl_u32 v14, v5, v7, 4 // (op 2145) 


v_and_b32 v5, 1, v12 // (op 2145) 

v_lshlrev_b32 v7, 3, v5 // (op 2145) 

v_add_u32 v5, v14, v7 // (op 2145) 


v_lshrrev_b32 v7, 6, v1 // (op 2211) 
v_and_b32 v12, 15, v1 // (op 2211) 
v_lshl_add_u32 v14, v7, 4, v12 // (op 2211) 


v_and_b32 v7, 48, v1 // (op 2211) 
v_add_u32 v12, v7, v14 // (op 2211) 


v_and_b32 v7, 48, v1 // (op 2277) 
v_and_b32 v14, 15, v1 // (op 2277) 
v_add_u32 v17, v7, v14 // (op 2277) 


v_accvgpr_write a0, 0.00000 // (op 184) call()
v_accvgpr_write a1, 0.00000 // (op 184) call()
v_accvgpr_write a2, 0.00000 // (op 184) call()
v_accvgpr_write a3, 0.00000 // (op 184) call()
s_mul_i32 s15, s16, s50 // (op 1239) 
s_add_u32 s15, s15, s0 // (op 1239) 
s_add_u32 s6, s15, s9 // (op 1239) 

s_mul_hi_u32 s15, s6, s13 // (op 1239) 
s_sub_u32 s7, s6, s15 // (op 1239) 
s_lshr_b32 s8, s7, 1 // (op 1239) 

s_add_u32 s7, s8, s15 // (op 1239) 


s_lshr_b32 s8, s7, s11 // (op 1239) 

s_and_b32 s7, s11, 2147483648 // (op 1239) 
s_cmp_eq_i32 s7, 0 // (op 1239) 
s_mov_b32 s15, scc // (op 1239) 

s_cmp_eq_i32 s15, 0 // (op 1239) 
s_cselect_b32 s7, s6, s8 // (op 1239) 


s_mul_hi_u32 s8, s7, s12 // (op 1239) 
s_sub_u32 s15, s7, s8 // (op 1239) 
s_lshr_b32 s18, s15, 1 // (op 1239) 

s_add_u32 s15, s18, s8 // (op 1239) 


s_lshr_b32 s8, s15, s10 // (op 1239) 

s_and_b32 s15, s10, 2147483648 // (op 1239) 
s_cmp_eq_i32 s15, 0 // (op 1239) 
s_mov_b32 s18, scc // (op 1239) 

s_cmp_eq_i32 s18, 0 // (op 1239) 
s_cselect_b32 s15, s7, s8 // (op 1239) 


s_lshl_b32 s8, s15, 4 // (op 1239) 

s_mov_b32 s19, 0 // (op 1239) convert
s_mov_b32 s18, s8 // (op 1239) convert

s_ashr_i64 s[42:43], s[2:3], 7 // (op 1239) 
s_mul_i32 s8, s7, s42 // (op 1239) 


s_sub_u32 s7, s6, s8 // (op 1239) 


v_lshl_add_u32 v7, s7, 7, v1 // (op 1239) 

v_mul_lo_u32 v21, v7, s25 // (op 1239) most significant: low of low * high
v_mul_hi_u32 v14, v7, s24 // (op 1239) most significant: high of low * low
v_mul_lo_u32 v20, v7, s24 // (op 1239) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 1239) most significant: sum


v_mov_b32 v7, s18 // (op 1239) 
v_mov_b32 v14, s19 // (op 1239) 
v_add_co_u32 v22, s[6:7], v7, v20 // (op 1239) least significant half
v_addc_co_u32 v23, s[6:7], v14, v21, s[6:7] // (op 1239) most significant half





s_mul_i32 s19, s24, s3 // (op 1241) most significant: low of low * high
s_mul_i32 s15, s25, s2 // (op 1241) most significant: low of high * low
s_mul_i32 s18, s24, s2 // (op 1241) least significant
s_mul_hi_u32 s6, s24, s2 // (op 1241) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1241) most significant: sum
s_add_u32 s19, s19, s6 // (op 1241) most significant: sum


s_mov_b32 s60, s26 // (op 1241) reg expression
s_mov_b32 s61, s27 // (op 1241) reg expression
s_mov_b32 s62, s18 // (op 1241) reg expression
s_mov_b32 s63, 131072 // (op 1241) call()

s_mul_i32 s15, s16, s50 // (op 1258) 
s_add_u32 s15, s15, s0 // (op 1258) 
s_add_u32 s6, s15, s9 // (op 1258) 

s_mul_hi_u32 s15, s6, s13 // (op 1258) 
s_sub_u32 s7, s6, s15 // (op 1258) 
s_lshr_b32 s8, s7, 1 // (op 1258) 

s_add_u32 s7, s8, s15 // (op 1258) 


s_lshr_b32 s8, s7, s11 // (op 1258) 

s_and_b32 s7, s11, 2147483648 // (op 1258) 
s_cmp_eq_i32 s7, 0 // (op 1258) 
s_mov_b32 s15, scc // (op 1258) 

s_cmp_eq_i32 s15, 0 // (op 1258) 
s_cselect_b32 s7, s6, s8 // (op 1258) 


s_mul_hi_u32 s8, s7, s12 // (op 1258) 
s_sub_u32 s15, s7, s8 // (op 1258) 
s_lshr_b32 s18, s15, 1 // (op 1258) 

s_add_u32 s15, s18, s8 // (op 1258) 


s_lshr_b32 s8, s15, s10 // (op 1258) 

s_and_b32 s15, s10, 2147483648 // (op 1258) 
s_cmp_eq_i32 s15, 0 // (op 1258) 
s_mov_b32 s18, scc // (op 1258) 

s_cmp_eq_i32 s18, 0 // (op 1258) 
s_cselect_b32 s15, s7, s8 // (op 1258) 


s_lshl_b32 s8, s15, 4 // (op 1258) 

s_mov_b32 s19, 0 // (op 1258) convert
s_mov_b32 s18, s8 // (op 1258) convert

s_ashr_i64 s[42:43], s[2:3], 7 // (op 1258) 
s_mul_i32 s8, s7, s42 // (op 1258) 


s_sub_u32 s7, s6, s8 // (op 1258) 


v_lshl_add_u32 v7, s7, 7, v1 // (op 1258) 

v_mul_lo_u32 v21, v7, s25 // (op 1258) most significant: low of low * high
v_mul_hi_u32 v14, v7, s24 // (op 1258) most significant: high of low * low
v_mul_lo_u32 v20, v7, s24 // (op 1258) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 1258) most significant: sum


v_mov_b32 v7, s18 // (op 1258) 
v_mov_b32 v14, s19 // (op 1258) 
v_add_co_u32 v24, s[6:7], v7, v20 // (op 1258) least significant half
v_addc_co_u32 v25, s[6:7], v14, v21, s[6:7] // (op 1258) most significant half





s_mul_i32 s19, s24, s3 // (op 1260) most significant: low of low * high
s_mul_i32 s15, s25, s2 // (op 1260) most significant: low of high * low
s_mul_i32 s18, s24, s2 // (op 1260) least significant
s_mul_hi_u32 s6, s24, s2 // (op 1260) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1260) most significant: sum
s_add_u32 s19, s19, s6 // (op 1260) most significant: sum


s_mov_b32 s60, s26 // (op 1260) reg expression
s_mov_b32 s61, s27 // (op 1260) reg expression
s_mov_b32 s62, s18 // (op 1260) reg expression
s_mov_b32 s63, 131072 // (op 1260) call()

s_mul_i32 s15, s16, s50 // (op 1367) 
s_add_u32 s15, s15, s0 // (op 1367) 
s_add_u32 s6, s15, s9 // (op 1367) 

s_mul_hi_u32 s15, s6, s13 // (op 1367) 
s_sub_u32 s7, s6, s15 // (op 1367) 
s_lshr_b32 s8, s7, 1 // (op 1367) 

s_add_u32 s7, s8, s15 // (op 1367) 


s_lshr_b32 s8, s7, s11 // (op 1367) 

s_and_b32 s7, s11, 2147483648 // (op 1367) 
s_cmp_eq_i32 s7, 0 // (op 1367) 
s_mov_b32 s15, scc // (op 1367) 

s_cmp_eq_i32 s15, 0 // (op 1367) 
s_cselect_b32 s7, s6, s8 // (op 1367) 


s_mul_hi_u32 s8, s7, s12 // (op 1367) 
s_sub_u32 s15, s7, s8 // (op 1367) 
s_lshr_b32 s18, s15, 1 // (op 1367) 

s_add_u32 s15, s18, s8 // (op 1367) 


s_lshr_b32 s8, s15, s10 // (op 1367) 

s_and_b32 s15, s10, 2147483648 // (op 1367) 
s_cmp_eq_i32 s15, 0 // (op 1367) 
s_mov_b32 s18, scc // (op 1367) 

s_cmp_eq_i32 s18, 0 // (op 1367) 
s_cselect_b32 s15, s7, s8 // (op 1367) 


s_mul_i32 s8, s15, s14 // (op 1367) 

s_sub_u32 s15, s7, s8 // (op 1367) 

s_lshl_b32 s8, s15, 4 // (op 1367) 

s_mov_b32 s19, 0 // (op 1367) convert
s_mov_b32 s18, s8 // (op 1367) convert

s_ashr_i64 s[42:43], s[2:3], 7 // (op 1367) 
s_mul_i32 s8, s7, s42 // (op 1367) 


s_sub_u32 s7, s6, s8 // (op 1367) 


v_lshl_add_u32 v7, s7, 7, v1 // (op 1367) 

v_mul_lo_u32 v21, v7, s29 // (op 1367) most significant: low of low * high
v_mul_hi_u32 v14, v7, s28 // (op 1367) most significant: high of low * low
v_mul_lo_u32 v20, v7, s28 // (op 1367) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 1367) most significant: sum

v_mov_b32 v14, s18 // (op 1367) 
v_mov_b32 v19, s19 // (op 1367) 
v_add_co_u32 v20, s[6:7], v20, v14 // (op 1367) least significant half
v_addc_co_u32 v21, s[6:7], v21, v19, s[6:7] // (op 1367) most significant half





s_mul_i32 s19, s28, s3 // (op 1369) most significant: low of low * high
s_mul_i32 s15, s29, s2 // (op 1369) most significant: low of high * low
s_mul_i32 s18, s28, s2 // (op 1369) least significant
s_mul_hi_u32 s6, s28, s2 // (op 1369) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1369) most significant: sum
s_add_u32 s19, s19, s6 // (op 1369) most significant: sum


s_mov_b32 s64, s30 // (op 1369) reg expression
s_mov_b32 s65, s31 // (op 1369) reg expression
s_mov_b32 s66, s18 // (op 1369) reg expression
s_mov_b32 s67, 131072 // (op 1369) call()

s_mul_i32 s15, s16, s50 // (op 1386) 
s_add_u32 s15, s15, s0 // (op 1386) 
s_add_u32 s6, s15, s9 // (op 1386) 

s_mul_hi_u32 s15, s6, s13 // (op 1386) 
s_sub_u32 s7, s6, s15 // (op 1386) 
s_lshr_b32 s8, s7, 1 // (op 1386) 

s_add_u32 s7, s8, s15 // (op 1386) 


s_lshr_b32 s8, s7, s11 // (op 1386) 

s_and_b32 s7, s11, 2147483648 // (op 1386) 
s_cmp_eq_i32 s7, 0 // (op 1386) 
s_mov_b32 s15, scc // (op 1386) 

s_cmp_eq_i32 s15, 0 // (op 1386) 
s_cselect_b32 s7, s6, s8 // (op 1386) 


s_mul_hi_u32 s8, s7, s12 // (op 1386) 
s_sub_u32 s15, s7, s8 // (op 1386) 
s_lshr_b32 s18, s15, 1 // (op 1386) 

s_add_u32 s15, s18, s8 // (op 1386) 


s_lshr_b32 s8, s15, s10 // (op 1386) 

s_and_b32 s15, s10, 2147483648 // (op 1386) 
s_cmp_eq_i32 s15, 0 // (op 1386) 
s_mov_b32 s18, scc // (op 1386) 

s_cmp_eq_i32 s18, 0 // (op 1386) 
s_cselect_b32 s15, s7, s8 // (op 1386) 


s_mul_i32 s8, s15, s14 // (op 1386) 

s_sub_u32 s15, s7, s8 // (op 1386) 

s_lshl_b32 s8, s15, 4 // (op 1386) 

s_mov_b32 s19, 0 // (op 1386) convert
s_mov_b32 s18, s8 // (op 1386) convert

s_ashr_i64 s[42:43], s[2:3], 7 // (op 1386) 
s_mul_i32 s8, s7, s42 // (op 1386) 


s_sub_u32 s7, s6, s8 // (op 1386) 


v_lshl_add_u32 v7, s7, 7, v1 // (op 1386) 

v_mul_lo_u32 v27, v7, s29 // (op 1386) most significant: low of low * high
v_mul_hi_u32 v14, v7, s28 // (op 1386) most significant: high of low * low
v_mul_lo_u32 v26, v7, s28 // (op 1386) least significant: low of low * low
v_add_u32 v27, v27, v14 // (op 1386) most significant: sum

v_mov_b32 v14, s18 // (op 1386) 
v_mov_b32 v19, s19 // (op 1386) 
v_add_co_u32 v26, s[6:7], v26, v14 // (op 1386) least significant half
v_addc_co_u32 v27, s[6:7], v27, v19, s[6:7] // (op 1386) most significant half





s_mul_i32 s19, s28, s3 // (op 1388) most significant: low of low * high
s_mul_i32 s15, s29, s2 // (op 1388) most significant: low of high * low
s_mul_i32 s18, s28, s2 // (op 1388) least significant
s_mul_hi_u32 s6, s28, s2 // (op 1388) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1388) most significant: sum
s_add_u32 s19, s19, s6 // (op 1388) most significant: sum


s_mov_b32 s64, s30 // (op 1388) reg expression
s_mov_b32 s65, s31 // (op 1388) reg expression
s_mov_b32 s66, s18 // (op 1388) reg expression
s_mov_b32 s67, 131072 // (op 1388) call()

v_and_b32 v7, -16, v1 // (op 1495) 
s_mul_i32 s15, s16, s50 // (op 1495) 
s_add_u32 s15, s15, s0 // (op 1495) 
v_sub_u32 v14, v1, v7 // (op 1495) 

s_add_u32 s6, s15, s9 // (op 1495) 

v_lshrrev_b32 v7, 4, v1 // (op 1495) 
s_mul_hi_u32 s15, s6, s13 // (op 1495) 
s_sub_u32 s7, s6, s15 // (op 1495) 
s_lshr_b32 s8, s7, 1 // (op 1495) 

s_add_u32 s7, s8, s15 // (op 1495) 


s_lshr_b32 s8, s7, s11 // (op 1495) 

s_and_b32 s7, s11, 2147483648 // (op 1495) 
s_cmp_eq_i32 s7, 0 // (op 1495) 
s_mov_b32 s15, scc // (op 1495) 

s_cmp_eq_i32 s15, 0 // (op 1495) 
s_cselect_b32 s7, s6, s8 // (op 1495) 


s_mul_hi_u32 s8, s7, s12 // (op 1495) 
s_sub_u32 s15, s7, s8 // (op 1495) 
s_lshr_b32 s18, s15, 1 // (op 1495) 

s_add_u32 s15, s18, s8 // (op 1495) 


s_lshr_b32 s8, s15, s10 // (op 1495) 

s_and_b32 s15, s10, 2147483648 // (op 1495) 
s_cmp_eq_i32 s15, 0 // (op 1495) 
s_mov_b32 s18, scc // (op 1495) 

s_cmp_eq_i32 s18, 0 // (op 1495) 
s_cselect_b32 s15, s7, s8 // (op 1495) 


v_lshl_add_u32 v19, s15, 4, v14 // (op 1495) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1495) 
s_mul_i32 s8, s7, s18 // (op 1495) 


s_sub_u32 s7, s6, s8 // (op 1495) 


v_lshl_add_u32 v14, s7, 2, v7 // (op 1495) 


v_mul_lo_u32 v7, v14, s32 // (op 1495) 

v_add_u32 v14, v19, v7 // (op 1495) 


s_mul_i32 s19, s32, s37 // (op 1497) most significant: low of low * high
s_mul_i32 s15, s33, s36 // (op 1497) most significant: low of high * low
s_mul_i32 s18, s32, s36 // (op 1497) least significant
s_mul_hi_u32 s6, s32, s36 // (op 1497) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1497) most significant: sum
s_add_u32 s19, s19, s6 // (op 1497) most significant: sum


s_mov_b32 s68, s34 // (op 1497) reg expression
s_mov_b32 s69, s35 // (op 1497) reg expression
s_mov_b32 s70, s18 // (op 1497) reg expression
s_mov_b32 s71, 131072 // (op 1497) call()

v_and_b32 v7, -16, v1 // (op 1514) 
s_mul_i32 s15, s16, s50 // (op 1514) 
s_add_u32 s15, s15, s0 // (op 1514) 
v_sub_u32 v19, v1, v7 // (op 1514) 

s_add_u32 s6, s15, s9 // (op 1514) 

v_lshrrev_b32 v7, 4, v1 // (op 1514) 
s_mul_hi_u32 s15, s6, s13 // (op 1514) 
s_sub_u32 s7, s6, s15 // (op 1514) 
s_lshr_b32 s8, s7, 1 // (op 1514) 

s_add_u32 s7, s8, s15 // (op 1514) 


s_lshr_b32 s8, s7, s11 // (op 1514) 

s_and_b32 s7, s11, 2147483648 // (op 1514) 
s_cmp_eq_i32 s7, 0 // (op 1514) 
s_mov_b32 s15, scc // (op 1514) 

s_cmp_eq_i32 s15, 0 // (op 1514) 
s_cselect_b32 s7, s6, s8 // (op 1514) 


s_mul_hi_u32 s8, s7, s12 // (op 1514) 
s_sub_u32 s15, s7, s8 // (op 1514) 
s_lshr_b32 s18, s15, 1 // (op 1514) 

s_add_u32 s15, s18, s8 // (op 1514) 


s_lshr_b32 s8, s15, s10 // (op 1514) 

s_and_b32 s15, s10, 2147483648 // (op 1514) 
s_cmp_eq_i32 s15, 0 // (op 1514) 
s_mov_b32 s18, scc // (op 1514) 

s_cmp_eq_i32 s18, 0 // (op 1514) 
s_cselect_b32 s15, s7, s8 // (op 1514) 


v_lshl_add_u32 v28, s15, 4, v19 // (op 1514) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1514) 
s_mul_i32 s8, s7, s18 // (op 1514) 


s_sub_u32 s7, s6, s8 // (op 1514) 


v_lshl_add_u32 v19, s7, 2, v7 // (op 1514) 


v_mul_lo_u32 v7, v19, s32 // (op 1514) 

v_add_u32 v19, v28, v7 // (op 1514) 


s_mul_i32 s19, s32, s37 // (op 1516) most significant: low of low * high
s_mul_i32 s15, s33, s36 // (op 1516) most significant: low of high * low
s_mul_i32 s18, s32, s36 // (op 1516) least significant
s_mul_hi_u32 s6, s32, s36 // (op 1516) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1516) most significant: sum
s_add_u32 s19, s19, s6 // (op 1516) most significant: sum


s_mov_b32 s68, s34 // (op 1516) reg expression
s_mov_b32 s69, s35 // (op 1516) reg expression
s_mov_b32 s70, s18 // (op 1516) reg expression
s_mov_b32 s71, 131072 // (op 1516) call()

v_and_b32 v7, -16, v1 // (op 1623) 
s_mul_i32 s15, s16, s50 // (op 1623) 
s_add_u32 s15, s15, s0 // (op 1623) 
v_sub_u32 v28, v1, v7 // (op 1623) 

s_add_u32 s6, s15, s9 // (op 1623) 

v_lshrrev_b32 v7, 4, v1 // (op 1623) 
s_mul_hi_u32 s15, s6, s13 // (op 1623) 
s_sub_u32 s7, s6, s15 // (op 1623) 
s_lshr_b32 s8, s7, 1 // (op 1623) 

s_add_u32 s7, s8, s15 // (op 1623) 


s_lshr_b32 s8, s7, s11 // (op 1623) 

s_and_b32 s7, s11, 2147483648 // (op 1623) 
s_cmp_eq_i32 s7, 0 // (op 1623) 
s_mov_b32 s15, scc // (op 1623) 

s_cmp_eq_i32 s15, 0 // (op 1623) 
s_cselect_b32 s7, s6, s8 // (op 1623) 


s_mul_hi_u32 s8, s7, s12 // (op 1623) 
s_sub_u32 s15, s7, s8 // (op 1623) 
s_lshr_b32 s18, s15, 1 // (op 1623) 

s_add_u32 s15, s18, s8 // (op 1623) 


s_lshr_b32 s8, s15, s10 // (op 1623) 

s_and_b32 s15, s10, 2147483648 // (op 1623) 
s_cmp_eq_i32 s15, 0 // (op 1623) 
s_mov_b32 s18, scc // (op 1623) 

s_cmp_eq_i32 s18, 0 // (op 1623) 
s_cselect_b32 s15, s7, s8 // (op 1623) 


s_mul_i32 s8, s15, s14 // (op 1623) 

s_sub_u32 s15, s7, s8 // (op 1623) 

v_lshl_add_u32 v29, s15, 4, v28 // (op 1623) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1623) 
s_mul_i32 s8, s7, s18 // (op 1623) 


s_sub_u32 s7, s6, s8 // (op 1623) 


v_lshl_add_u32 v28, s7, 2, v7 // (op 1623) 


v_mul_lo_u32 v7, v28, s38 // (op 1623) 
v_add_u32 v7, v7, v29 // (op 1623) 


s_mul_i32 s19, s38, s37 // (op 1625) most significant: low of low * high
s_mul_i32 s15, s39, s36 // (op 1625) most significant: low of high * low
s_mul_i32 s18, s38, s36 // (op 1625) least significant
s_mul_hi_u32 s6, s38, s36 // (op 1625) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1625) most significant: sum
s_add_u32 s19, s19, s6 // (op 1625) most significant: sum


s_mov_b32 s72, s40 // (op 1625) reg expression
s_mov_b32 s73, s41 // (op 1625) reg expression
s_mov_b32 s74, s18 // (op 1625) reg expression
s_mov_b32 s75, 131072 // (op 1625) call()

v_and_b32 v28, -16, v1 // (op 1642) 
s_mul_i32 s15, s16, s50 // (op 1642) 
s_add_u32 s15, s15, s0 // (op 1642) 
v_sub_u32 v29, v1, v28 // (op 1642) 

s_add_u32 s6, s15, s9 // (op 1642) 

v_lshrrev_b32 v28, 4, v1 // (op 1642) 
s_mul_hi_u32 s15, s6, s13 // (op 1642) 
s_sub_u32 s7, s6, s15 // (op 1642) 
s_lshr_b32 s8, s7, 1 // (op 1642) 

s_add_u32 s7, s8, s15 // (op 1642) 


s_lshr_b32 s8, s7, s11 // (op 1642) 

s_and_b32 s7, s11, 2147483648 // (op 1642) 
s_cmp_eq_i32 s7, 0 // (op 1642) 
s_mov_b32 s15, scc // (op 1642) 

s_cmp_eq_i32 s15, 0 // (op 1642) 
s_cselect_b32 s7, s6, s8 // (op 1642) 


s_mul_hi_u32 s8, s7, s12 // (op 1642) 
s_sub_u32 s15, s7, s8 // (op 1642) 
s_lshr_b32 s18, s15, 1 // (op 1642) 

s_add_u32 s15, s18, s8 // (op 1642) 


s_lshr_b32 s8, s15, s10 // (op 1642) 

s_and_b32 s15, s10, 2147483648 // (op 1642) 
s_cmp_eq_i32 s15, 0 // (op 1642) 
s_mov_b32 s18, scc // (op 1642) 

s_cmp_eq_i32 s18, 0 // (op 1642) 
s_cselect_b32 s15, s7, s8 // (op 1642) 


s_mul_i32 s8, s15, s14 // (op 1642) 

s_sub_u32 s15, s7, s8 // (op 1642) 

v_lshl_add_u32 v30, s15, 4, v29 // (op 1642) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1642) 
s_mul_i32 s8, s7, s18 // (op 1642) 


s_sub_u32 s7, s6, s8 // (op 1642) 


v_lshl_add_u32 v29, s7, 2, v28 // (op 1642) 


v_mul_lo_u32 v28, v29, s38 // (op 1642) 
v_add_u32 v28, v28, v30 // (op 1642) 


s_mul_i32 s19, s38, s37 // (op 1644) most significant: low of low * high
s_mul_i32 s15, s39, s36 // (op 1644) most significant: low of high * low
s_mul_i32 s18, s38, s36 // (op 1644) least significant
s_mul_hi_u32 s6, s38, s36 // (op 1644) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1644) most significant: sum
s_add_u32 s19, s19, s6 // (op 1644) most significant: sum


s_mov_b32 s72, s40 // (op 1644) reg expression
s_mov_b32 s73, s41 // (op 1644) reg expression
s_mov_b32 s74, s18 // (op 1644) reg expression
s_mov_b32 s75, 131072 // (op 1644) call()

v_lshlrev_b32 v29, 4, v1 // (op 2121) 
s_nop 0
v_readfirstlane_b32 s15, v29 // (op 2121) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2756) 

s_lshl_b64 s[18:19], s[28:29], 6 // (op 1119) 
v_mov_b64 v[30:31], s[18:19] // (op 1119) convert


v_mov_b64 v[32:33], v[20:21] // (op 1119) reg expression
s_mov_b32 m0, s15 // (op 1119) reg expression
buffer_load_dwordx4 v32, s[64:67], 0 offen offset: 0 lds // (op 1119) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 1119) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 1119) most significant half

s_add_u32 m0, m0, 1024 // (op 1119) 
buffer_load_dwordx4 v32, s[64:67], 0 offen offset: 0 lds // (op 1119) Load value direct to lds





v_mov_b32 v29, v7 // (op 954) reg expression

buffer_load_ubyte v30, v29, s[72:75], 0 offen offset: 0 // (op 954) Load value

v_lshlrev_b32 v29, 4, v1 // (op 2056) 
s_nop 0
v_readfirstlane_b32 s15, v29 // (op 2056) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2737) 

s_lshl_b64 s[18:19], s[24:25], 6 // (op 1124) 
v_mov_b64 v[32:33], s[18:19] // (op 1124) convert


v_mov_b64 v[34:35], v[22:23] // (op 1124) reg expression
s_add_u32 s6, 2048, s15 // (op 1124) 
s_mov_b32 m0, s6 // (op 1124) reg expression
buffer_load_dwordx4 v34, s[60:63], 0 offen offset: 0 lds // (op 1124) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 1124) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 1124) most significant half

s_add_u32 m0, m0, 1024 // (op 1124) 
buffer_load_dwordx4 v34, s[60:63], 0 offen offset: 0 lds // (op 1124) Load value direct to lds






v_mov_b32 v29, v14 // (op 960) reg expression

buffer_load_ubyte v31, v29, s[68:71], 0 offen offset: 0 // (op 960) Load value

v_lshlrev_b32 v29, 4, v1 // (op 2378) 
s_nop 0
v_readfirstlane_b32 s15, v29 // (op 2378) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2826) 

s_lshl_b64 s[18:19], s[28:29], 6 // (op 1129) 
v_mov_b64 v[32:33], s[18:19] // (op 1129) convert


s_lshl_b64 s[18:19], s[28:29], 7 // (op 1129) 
v_mov_b32 v29, s18 // (op 1129) 
v_mov_b32 v34, s19 // (op 1129) 
v_add_co_u32 v36, s[6:7], v20, v29 // (op 1129) least significant half
v_addc_co_u32 v37, s[6:7], v21, v34, s[6:7] // (op 1129) most significant half




s_add_u32 s6, 4096, s15 // (op 1129) 
s_nop 0
s_mov_b32 m0, s6 // (op 1129) reg expression
buffer_load_dwordx4 v36, s[64:67], 0 offen offset: 0 lds // (op 1129) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 1129) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1129) most significant half

s_add_u32 m0, m0, 1024 // (op 1129) 
buffer_load_dwordx4 v36, s[64:67], 0 offen offset: 0 lds // (op 1129) Load value direct to lds







s_lshl_b32 s15, s38, 2 // (op 966) 
v_add_u32 v29, v7, s15 // (op 966) 


buffer_load_ubyte v20, v29, s[72:75], 0 offen offset: 0 // (op 966) Load value


v_lshlrev_b32 v7, 4, v1 // (op 2339) 
s_nop 0
v_readfirstlane_b32 s15, v7 // (op 2339) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2816) 

s_lshl_b64 s[18:19], s[24:25], 6 // (op 1131) 
v_mov_b64 v[32:33], s[18:19] // (op 1131) convert


s_lshl_b64 s[18:19], s[24:25], 7 // (op 1131) 
v_mov_b32 v7, s18 // (op 1131) 
v_mov_b32 v21, s19 // (op 1131) 
v_add_co_u32 v34, s[6:7], v22, v7 // (op 1131) least significant half
v_addc_co_u32 v35, s[6:7], v23, v21, s[6:7] // (op 1131) most significant half




s_add_u32 s6, 6144, s15 // (op 1131) 
s_nop 0
s_mov_b32 m0, s6 // (op 1131) reg expression
buffer_load_dwordx4 v34, s[60:63], 0 offen offset: 0 lds // (op 1131) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 1131) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 1131) most significant half

s_add_u32 m0, m0, 1024 // (op 1131) 
buffer_load_dwordx4 v34, s[60:63], 0 offen offset: 0 lds // (op 1131) Load value direct to lds







s_lshl_b32 s15, s32, 2 // (op 972) 
v_add_u32 v7, v14, s15 // (op 972) 


buffer_load_ubyte v29, v7, s[68:71], 0 offen offset: 0 // (op 972) Load value


v_and_b32 v7, -16, v1 // (op 2253) 
v_sub_u32 v14, v1, v7 // (op 2253) 

v_and_b32 v7, 4294967280, v1 // (op 2253) 
v_add_u32 v21, v7, v14 // (op 2253) 


s_barrier  // (op 2796) 



s_waitcnt vmcnt(9)
ds_write_b8 v21, v30 offset:8192 // (op 975) Store local data 

v_and_b32 v7, -16, v1 // (op 2187) 
v_sub_u32 v14, v1, v7 // (op 2187) 

v_and_b32 v7, 4294967280, v1 // (op 2187) 
v_add_u32 v21, v7, v14 // (op 2187) 


s_barrier  // (op 2776) 



s_waitcnt vmcnt(6)
ds_write_b8 v21, v31 offset:8256 // (op 977) Store local data 

s_barrier  // (op 2753) 



ds_read_b64_tr_b8 v[32:33], v5 // (op 994) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v5 offset:128 // (op 994) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v5 offset:1024 // (op 994) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v5 offset:1152 // (op 994) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2793) 


ds_read_u8 v7, v17 offset:8192 // (op 1000) Load local data 
s_barrier  // (op 2734) 



ds_read_b64_tr_b8 v[40:41], v0 offset:2048 // (op 991) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:2176 // (op 991) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:3072 // (op 991) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:3200 // (op 991) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2773) 


ds_read_u8 v14, v12 offset:8256 // (op 997) Load local data 
v_lshlrev_b32 v21, 4, v1 // (op 2028) 
s_nop 0
v_readfirstlane_b32 s15, v21 // (op 2028) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v21, 4, v1 // (op 2093) 
s_nop 0
v_readfirstlane_b32 s6, v21 // (op 2093) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -16, v1 // (op 2158) 
v_sub_u32 v22, v1, v21 // (op 2158) 

v_and_b32 v21, 4294967280, v1 // (op 2158) 
v_add_u32 v23, v21, v22 // (op 2158) 


v_and_b32 v21, -16, v1 // (op 2224) 
v_sub_u32 v22, v1, v21 // (op 2224) 

v_and_b32 v21, 4294967280, v1 // (op 2224) 
v_add_u32 v48, v21, v22 // (op 2224) 


v_lshlrev_b32 v21, 4, v1 // (op 2324) 
s_nop 0
v_readfirstlane_b32 s7, v21 // (op 2324) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v21, 4, v1 // (op 2363) 
s_nop 0
v_readfirstlane_b32 s8, v21 // (op 2363) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -16, v1 // (op 2402) 
v_sub_u32 v22, v1, v21 // (op 2402) 

v_and_b32 v21, 4294967280, v1 // (op 2402) 
v_add_u32 v49, v21, v22 // (op 2402) 


v_and_b32 v21, -16, v1 // (op 2426) 
v_sub_u32 v22, v1, v21 // (op 2426) 

v_and_b32 v21, 4294967280, v1 // (op 2426) 
v_add_u32 v50, v21, v22 // (op 2426) 


s_mov_b32 s1, 0 // (op 186) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 185) 
s_mov_b32 s42, s18 // (op 185) convert

s_and_b32 s18, s42, 4294967294 // (op 185) 

s_cmp_lt_u32 s1, s18 // (op 185) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_185 // (op 185) Condition: Top (jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_185 if false)
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoop_185:
s_add_u32 s1, s1, 2 // (op 187) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[32:39], a[0:3], v14, v7 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 198) 
s_barrier  // (op 2759) 

s_lshl_b64 s[18:19], s[28:29], 6 // (op 1054) 
v_mov_b64 v[52:53], s[18:19] // (op 1054) convert


v_mov_b64 v[54:55], v[26:27] // (op 1054) 
s_lshl_b64 s[18:19], s[28:29], 7 // (op 1054) 
s_lshl_b64 s[42:43], s[18:19], 1 // (op 1054) 

v_mov_b32 v51, s42 // (op 1054) 
v_mov_b32 v21, s43 // (op 1054) 
v_add_co_u32 v56, s[18:19], v54, v51 // (op 1054) least significant half
v_addc_co_u32 v57, s[18:19], v55, v21, s[18:19] // (op 1054) most significant half




s_mov_b32 m0, s6 // (op 1054) reg expression
buffer_load_dwordx4 v56, s[64:67], 0 offen offset: 0 lds // (op 1054) Load value direct to lds
v_add_co_u32 v56, s[18:19], v56, v52 // (op 1054) least significant half
v_addc_co_u32 v57, s[18:19], 0, v57, s[18:19] // (op 1054) most significant half

s_add_u32 m0, m0, 1024 // (op 1054) 
buffer_load_dwordx4 v56, s[64:67], 0 offen offset: 0 lds // (op 1054) Load value direct to lds




v_mov_b32 v21, v28 // (op 204) 
s_lshl_b32 s18, s38, 3 // (op 204) 
v_add_u32 v22, v21, s18 // (op 204) 


buffer_load_ubyte v30, v22, s[72:75], 0 offen offset: 0 // (op 204) Load value

s_barrier  // (op 2740) 

s_lshl_b64 s[18:19], s[24:25], 6 // (op 1049) 
v_mov_b64 v[52:53], s[18:19] // (op 1049) convert


v_mov_b64 v[56:57], v[24:25] // (op 1049) 
s_lshl_b64 s[18:19], s[24:25], 7 // (op 1049) 
s_lshl_b64 s[42:43], s[18:19], 1 // (op 1049) 

v_mov_b32 v22, s42 // (op 1049) 
v_mov_b32 v51, s43 // (op 1049) 
v_add_co_u32 v58, s[18:19], v56, v22 // (op 1049) least significant half
v_addc_co_u32 v59, s[18:19], v57, v51, s[18:19] // (op 1049) most significant half




s_add_u32 s18, 2048, s15 // (op 1049) 
s_nop 0
s_mov_b32 m0, s18 // (op 1049) reg expression
buffer_load_dwordx4 v58, s[60:63], 0 offen offset: 0 lds // (op 1049) Load value direct to lds
v_add_co_u32 v58, s[42:43], v58, v52 // (op 1049) least significant half
v_addc_co_u32 v59, s[42:43], 0, v59, s[42:43] // (op 1049) most significant half

s_add_u32 m0, m0, 1024 // (op 1049) 
buffer_load_dwordx4 v58, s[60:63], 0 offen offset: 0 lds // (op 1049) Load value direct to lds





v_mov_b32 v22, v19 // (op 203) 
s_lshl_b32 s18, s32, 3 // (op 203) 
v_add_u32 v51, v22, s18 // (op 203) 


buffer_load_ubyte v31, v51, s[68:71], 0 offen offset: 0 // (op 203) Load value

s_barrier  // (op 2841) 



s_waitcnt vmcnt(9)
ds_write_b8 v50, v20 offset:8320 // (op 541) Store local data 
s_barrier  // (op 2835) 



s_waitcnt vmcnt(6)
ds_write_b8 v49, v29 offset:8384 // (op 540) Store local data 
s_barrier  // (op 2822) 



ds_read_b64_tr_b8 v[32:33], v10 offset:4096 // (op 533) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v10 offset:4224 // (op 533) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v10 offset:5120 // (op 533) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v10 offset:5248 // (op 533) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2838) 


ds_read_u8 v7, v15 offset:8320 // (op 537) Load local data 
s_barrier  // (op 2813) 
s_barrier  // (op 2979) 



ds_read_b64_tr_b8 v[40:41], v8 offset:6144 // (op 531) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v8 offset:6272 // (op 531) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v8 offset:7168 // (op 531) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v8 offset:7296 // (op 531) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2832) 


ds_read_u8 v14, v13 offset:8384 // (op 535) Load local data 
s_barrier  // (op 2973) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[32:39], a[0:3], v14, v7 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 538) 
s_barrier  // (op 2829) 

s_lshl_b64 s[18:19], s[28:29], 6 // (op 1077) 
v_mov_b64 v[52:53], s[18:19] // (op 1077) convert


s_lshl_b64 s[18:19], s[28:29], 7 // (op 1077) 
s_mul_i32 s43, 3, s19 // (op 1077) most significant: low of low * high
s_mul_i32 s42, 3, s18 // (op 1077) least significant
s_mul_hi_u32 s44, 3, s18 // (op 1077) most significant: high of low * low
s_add_u32 s43, s43, s44 // (op 1077) most significant: sum


v_mov_b32 v51, s42 // (op 1077) 
v_mov_b32 v58, s43 // (op 1077) 
v_add_co_u32 v60, s[18:19], v54, v51 // (op 1077) least significant half
v_addc_co_u32 v61, s[18:19], v55, v58, s[18:19] // (op 1077) most significant half




s_add_u32 s18, 4096, s8 // (op 1077) 
s_nop 0
s_mov_b32 m0, s18 // (op 1077) reg expression
buffer_load_dwordx4 v60, s[64:67], 0 offen offset: 0 lds // (op 1077) Load value direct to lds
v_add_co_u32 v60, s[42:43], v60, v52 // (op 1077) least significant half
v_addc_co_u32 v61, s[42:43], 0, v61, s[42:43] // (op 1077) most significant half

s_add_u32 m0, m0, 1024 // (op 1077) 
buffer_load_dwordx4 v60, s[64:67], 0 offen offset: 0 lds // (op 1077) Load value direct to lds






s_lshl_b32 s18, s38, 2 // (op 544) 
s_mul_i32 s19, 3, s18 // (op 544) 

v_add_u32 v51, v21, s19 // (op 544) 


buffer_load_ubyte v20, v51, s[72:75], 0 offen offset: 0 // (op 544) Load value

s_lshl_b64 s[18:19], s[28:29], 7 // (op 1378) 

s_lshl_b64 s[42:43], s[18:19], 1 // (op 1378) 

s_barrier  // (op 2819) 
s_lshl_b32 s18, s38, 3 // (op 1634) 

v_add_u32 v28, v28, s18 // (op 1634) 

v_mov_b32 v21, s42 // (op 1378) 
v_mov_b32 v51, s43 // (op 1378) 
s_lshl_b64 s[18:19], s[24:25], 6 // (op 1072) 
v_mov_b64 v[52:53], s[18:19] // (op 1072) convert


s_lshl_b64 s[18:19], s[24:25], 7 // (op 1072) 
s_mul_i32 s45, 3, s19 // (op 1072) most significant: low of low * high
s_mul_i32 s44, 3, s18 // (op 1072) least significant
s_mul_hi_u32 s52, 3, s18 // (op 1072) most significant: high of low * low
s_add_u32 s45, s45, s52 // (op 1072) most significant: sum


v_mov_b32 v54, s44 // (op 1072) 
v_mov_b32 v55, s45 // (op 1072) 
v_add_co_u32 v26, s[18:19], v26, v21 // (op 1378) least significant half
v_addc_co_u32 v27, s[18:19], v27, v51, s[18:19] // (op 1378) most significant half




v_add_co_u32 v58, s[18:19], v56, v54 // (op 1072) least significant half
v_addc_co_u32 v59, s[18:19], v57, v55, s[18:19] // (op 1072) most significant half




s_add_u32 s18, 6144, s7 // (op 1072) 
s_nop 0
s_mov_b32 m0, s18 // (op 1072) reg expression
buffer_load_dwordx4 v58, s[60:63], 0 offen offset: 0 lds // (op 1072) Load value direct to lds
v_add_co_u32 v58, s[42:43], v58, v52 // (op 1072) least significant half
v_addc_co_u32 v59, s[42:43], 0, v59, s[42:43] // (op 1072) most significant half

s_add_u32 m0, m0, 1024 // (op 1072) 
buffer_load_dwordx4 v58, s[60:63], 0 offen offset: 0 lds // (op 1072) Load value direct to lds






s_lshl_b32 s18, s32, 2 // (op 543) 
s_mul_i32 s19, 3, s18 // (op 543) 

v_add_u32 v21, v22, s19 // (op 543) 


buffer_load_ubyte v29, v21, s[68:71], 0 offen offset: 0 // (op 543) Load value

s_lshl_b64 s[18:19], s[24:25], 7 // (op 1250) 

s_lshl_b64 s[42:43], s[18:19], 1 // (op 1250) 

s_barrier  // (op 2799) 
s_lshl_b32 s18, s32, 3 // (op 1506) 



v_add_u32 v19, v19, s18 // (op 1506) 

v_mov_b32 v21, s42 // (op 1250) 
v_mov_b32 v22, s43 // (op 1250) 
v_add_co_u32 v24, s[18:19], v24, v21 // (op 1250) least significant half
v_addc_co_u32 v25, s[18:19], v25, v22, s[18:19] // (op 1250) most significant half




s_waitcnt vmcnt(9)
ds_write_b8 v48, v30 offset:8192 // (op 201) Store local data 
s_barrier  // (op 2779) 



s_waitcnt vmcnt(6)
ds_write_b8 v23, v31 offset:8256 // (op 200) Store local data 
s_barrier  // (op 2765) 



ds_read_b64_tr_b8 v[32:33], v6 // (op 193) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v6 offset:128 // (op 193) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v6 offset:1024 // (op 193) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v6 offset:1152 // (op 193) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2806) 


ds_read_u8 v7, v11 offset:8192 // (op 197) Load local data 
s_barrier  // (op 2746) 



ds_read_b64_tr_b8 v[40:41], v4 offset:2048 // (op 191) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:2176 // (op 191) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:3072 // (op 191) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v4 offset:3200 // (op 191) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2786) 


ds_read_u8 v14, v9 offset:8256 // (op 195) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 185) 
s_mov_b32 s42, s18 // (op 185) convert

s_and_b32 s18, s42, 4294967294 // (op 185) 

s_cmp_lt_u32 s1, s18 // (op 185) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoop_185 // (op 185) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoop_185 if true)
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_185:














s_barrier  // (op 2899) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1144) 
s_and_b32 s15, s18, 4294967294 // (op 1144) 
s_add_u32 s6, s0, s15 // (op 1144) 

s_mul_i32 s15, s16, s50 // (op 1144) 
s_add_u32 s15, s15, s6 // (op 1144) 

s_add_u32 s6, s15, s9 // (op 1144) 

s_mul_hi_u32 s15, s6, s13 // (op 1144) 
s_sub_u32 s7, s6, s15 // (op 1144) 
s_lshr_b32 s8, s7, 1 // (op 1144) 

s_add_u32 s7, s8, s15 // (op 1144) 


s_lshr_b32 s8, s7, s11 // (op 1144) 

s_and_b32 s7, s11, 2147483648 // (op 1144) 
s_cmp_eq_i32 s7, 0 // (op 1144) 
s_mov_b32 s15, scc // (op 1144) 

s_cmp_eq_i32 s15, 0 // (op 1144) 
s_cselect_b32 s7, s6, s8 // (op 1144) 


s_mul_hi_u32 s8, s7, s12 // (op 1144) 
s_sub_u32 s15, s7, s8 // (op 1144) 
s_lshr_b32 s42, s15, 1 // (op 1144) 

s_add_u32 s15, s42, s8 // (op 1144) 


s_lshr_b32 s8, s15, s10 // (op 1144) 

s_and_b32 s15, s10, 2147483648 // (op 1144) 
s_cmp_eq_i32 s15, 0 // (op 1144) 
s_mov_b32 s42, scc // (op 1144) 

s_cmp_eq_i32 s42, 0 // (op 1144) 
s_cselect_b32 s15, s7, s8 // (op 1144) 


s_lshl_b32 s8, s15, 4 // (op 1144) 

s_mov_b32 s43, 0 // (op 1144) convert
s_mov_b32 s42, s8 // (op 1144) convert

s_mul_i32 s8, s7, s18 // (op 1144) 


s_sub_u32 s7, s6, s8 // (op 1144) 


v_lshl_add_u32 v19, s7, 7, v1 // (op 1144) 

v_mul_lo_u32 v31, v19, s25 // (op 1144) most significant: low of low * high
v_mul_hi_u32 v21, v19, s24 // (op 1144) most significant: high of low * low
v_mul_lo_u32 v30, v19, s24 // (op 1144) least significant: low of low * low
v_add_u32 v31, v31, v21 // (op 1144) most significant: sum


v_mov_b32 v19, s42 // (op 1144) 
v_mov_b32 v21, s43 // (op 1144) 
v_add_co_u32 v22, s[18:19], v19, v30 // (op 1144) least significant half
v_addc_co_u32 v23, s[18:19], v21, v31, s[18:19] // (op 1144) most significant half





s_mul_i32 s19, s24, s3 // (op 1146) most significant: low of low * high
s_mul_i32 s15, s25, s2 // (op 1146) most significant: low of high * low
s_mul_i32 s18, s24, s2 // (op 1146) least significant
s_mul_hi_u32 s6, s24, s2 // (op 1146) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1146) most significant: sum
s_add_u32 s19, s19, s6 // (op 1146) most significant: sum


s_mov_b32 s60, s26 // (op 1146) reg expression
s_mov_b32 s61, s27 // (op 1146) reg expression
s_mov_b32 s62, s18 // (op 1146) reg expression
s_mov_b32 s63, 131072 // (op 1146) call()

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1280) 
s_and_b32 s15, s18, 4294967294 // (op 1280) 
s_add_u32 s6, s0, s15 // (op 1280) 

s_mul_i32 s15, s16, s50 // (op 1280) 
s_add_u32 s15, s15, s6 // (op 1280) 

s_add_u32 s6, s15, s9 // (op 1280) 

s_mul_hi_u32 s15, s6, s13 // (op 1280) 
s_sub_u32 s7, s6, s15 // (op 1280) 
s_lshr_b32 s8, s7, 1 // (op 1280) 

s_add_u32 s7, s8, s15 // (op 1280) 


s_lshr_b32 s8, s7, s11 // (op 1280) 

s_and_b32 s7, s11, 2147483648 // (op 1280) 
s_cmp_eq_i32 s7, 0 // (op 1280) 
s_mov_b32 s15, scc // (op 1280) 

s_cmp_eq_i32 s15, 0 // (op 1280) 
s_cselect_b32 s7, s6, s8 // (op 1280) 


s_mul_hi_u32 s8, s7, s12 // (op 1280) 
s_sub_u32 s15, s7, s8 // (op 1280) 
s_lshr_b32 s42, s15, 1 // (op 1280) 

s_add_u32 s15, s42, s8 // (op 1280) 


s_lshr_b32 s8, s15, s10 // (op 1280) 

s_and_b32 s15, s10, 2147483648 // (op 1280) 
s_cmp_eq_i32 s15, 0 // (op 1280) 
s_mov_b32 s42, scc // (op 1280) 

s_cmp_eq_i32 s42, 0 // (op 1280) 
s_cselect_b32 s15, s7, s8 // (op 1280) 


s_mul_i32 s8, s15, s14 // (op 1280) 

s_sub_u32 s15, s7, s8 // (op 1280) 

s_lshl_b32 s8, s15, 4 // (op 1280) 

s_mov_b32 s43, 0 // (op 1280) convert
s_mov_b32 s42, s8 // (op 1280) convert

s_mul_i32 s8, s7, s18 // (op 1280) 


s_sub_u32 s7, s6, s8 // (op 1280) 


v_lshl_add_u32 v19, s7, 7, v1 // (op 1280) 

v_mul_lo_u32 v31, v19, s29 // (op 1280) most significant: low of low * high
v_mul_hi_u32 v21, v19, s28 // (op 1280) most significant: high of low * low
v_mul_lo_u32 v30, v19, s28 // (op 1280) least significant: low of low * low
v_add_u32 v31, v31, v21 // (op 1280) most significant: sum

v_mov_b32 v21, s42 // (op 1280) 
v_mov_b32 v24, s43 // (op 1280) 
v_add_co_u32 v30, s[18:19], v30, v21 // (op 1280) least significant half
v_addc_co_u32 v31, s[18:19], v31, v24, s[18:19] // (op 1280) most significant half





s_mul_i32 s19, s28, s3 // (op 1282) most significant: low of low * high
s_mul_i32 s15, s29, s2 // (op 1282) most significant: low of high * low
s_mul_i32 s18, s28, s2 // (op 1282) least significant
s_mul_hi_u32 s6, s28, s2 // (op 1282) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1282) most significant: sum
s_add_u32 s19, s19, s6 // (op 1282) most significant: sum


s_mov_b32 s64, s30 // (op 1282) reg expression
s_mov_b32 s65, s31 // (op 1282) reg expression
s_mov_b32 s66, s18 // (op 1282) reg expression
s_mov_b32 s67, 131072 // (op 1282) call()

v_and_b32 v19, -16, v1 // (op 1408) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1408) 
v_sub_u32 v21, v1, v19 // (op 1408) 

s_and_b32 s15, s18, 4294967294 // (op 1408) 
v_lshrrev_b32 v19, 4, v1 // (op 1408) 
s_add_u32 s6, s0, s15 // (op 1408) 

s_mul_i32 s15, s16, s50 // (op 1408) 
s_add_u32 s15, s15, s6 // (op 1408) 

s_add_u32 s6, s15, s9 // (op 1408) 

s_mul_hi_u32 s15, s6, s13 // (op 1408) 
s_sub_u32 s7, s6, s15 // (op 1408) 
s_lshr_b32 s8, s7, 1 // (op 1408) 

s_add_u32 s7, s8, s15 // (op 1408) 


s_lshr_b32 s8, s7, s11 // (op 1408) 

s_and_b32 s7, s11, 2147483648 // (op 1408) 
s_cmp_eq_i32 s7, 0 // (op 1408) 
s_mov_b32 s15, scc // (op 1408) 

s_cmp_eq_i32 s15, 0 // (op 1408) 
s_cselect_b32 s7, s6, s8 // (op 1408) 


s_mul_hi_u32 s8, s7, s12 // (op 1408) 
s_sub_u32 s15, s7, s8 // (op 1408) 
s_lshr_b32 s42, s15, 1 // (op 1408) 

s_add_u32 s15, s42, s8 // (op 1408) 


s_lshr_b32 s8, s15, s10 // (op 1408) 

s_and_b32 s15, s10, 2147483648 // (op 1408) 
s_cmp_eq_i32 s15, 0 // (op 1408) 
s_mov_b32 s42, scc // (op 1408) 

s_cmp_eq_i32 s42, 0 // (op 1408) 
s_cselect_b32 s15, s7, s8 // (op 1408) 


s_mul_i32 s8, s7, s18 // (op 1408) 


v_lshl_add_u32 v24, s15, 4, v21 // (op 1408) 


s_sub_u32 s7, s6, s8 // (op 1408) 


v_lshl_add_u32 v21, s7, 2, v19 // (op 1408) 


v_mul_lo_u32 v19, v21, s32 // (op 1408) 

v_add_u32 v21, v24, v19 // (op 1408) 


s_mul_i32 s19, s32, s37 // (op 1410) most significant: low of low * high
s_mul_i32 s15, s33, s36 // (op 1410) most significant: low of high * low
s_mul_i32 s18, s32, s36 // (op 1410) least significant
s_mul_hi_u32 s6, s32, s36 // (op 1410) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1410) most significant: sum
s_add_u32 s19, s19, s6 // (op 1410) most significant: sum


s_mov_b32 s68, s34 // (op 1410) reg expression
s_mov_b32 s69, s35 // (op 1410) reg expression
s_mov_b32 s70, s18 // (op 1410) reg expression
s_mov_b32 s71, 131072 // (op 1410) call()

v_and_b32 v19, -16, v1 // (op 1536) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1536) 
v_sub_u32 v24, v1, v19 // (op 1536) 

s_and_b32 s15, s18, 4294967294 // (op 1536) 
v_lshrrev_b32 v19, 4, v1 // (op 1536) 
s_add_u32 s6, s0, s15 // (op 1536) 

s_mul_i32 s15, s16, s50 // (op 1536) 
s_add_u32 s15, s15, s6 // (op 1536) 

s_add_u32 s6, s15, s9 // (op 1536) 

s_mul_hi_u32 s15, s6, s13 // (op 1536) 
s_sub_u32 s7, s6, s15 // (op 1536) 
s_lshr_b32 s8, s7, 1 // (op 1536) 

s_add_u32 s7, s8, s15 // (op 1536) 


s_lshr_b32 s8, s7, s11 // (op 1536) 

s_and_b32 s7, s11, 2147483648 // (op 1536) 
s_cmp_eq_i32 s7, 0 // (op 1536) 
s_mov_b32 s15, scc // (op 1536) 

s_cmp_eq_i32 s15, 0 // (op 1536) 
s_cselect_b32 s7, s6, s8 // (op 1536) 


s_mul_hi_u32 s8, s7, s12 // (op 1536) 
s_sub_u32 s15, s7, s8 // (op 1536) 
s_lshr_b32 s42, s15, 1 // (op 1536) 

s_add_u32 s15, s42, s8 // (op 1536) 


s_lshr_b32 s8, s15, s10 // (op 1536) 

s_and_b32 s15, s10, 2147483648 // (op 1536) 
s_cmp_eq_i32 s15, 0 // (op 1536) 
s_mov_b32 s42, scc // (op 1536) 

s_cmp_eq_i32 s42, 0 // (op 1536) 
s_cselect_b32 s15, s7, s8 // (op 1536) 


s_mul_i32 s8, s15, s14 // (op 1536) 

s_sub_u32 s15, s7, s8 // (op 1536) 

s_mul_i32 s8, s7, s18 // (op 1536) 


v_lshl_add_u32 v25, s15, 4, v24 // (op 1536) 


s_sub_u32 s7, s6, s8 // (op 1536) 


v_lshl_add_u32 v24, s7, 2, v19 // (op 1536) 


v_mul_lo_u32 v19, v24, s38 // (op 1536) 
v_add_u32 v19, v19, v25 // (op 1536) 


s_mul_i32 s19, s38, s37 // (op 1538) most significant: low of low * high
s_mul_i32 s15, s39, s36 // (op 1538) most significant: low of high * low
s_mul_i32 s18, s38, s36 // (op 1538) least significant
s_mul_hi_u32 s6, s38, s36 // (op 1538) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1538) most significant: sum
s_add_u32 s19, s19, s6 // (op 1538) most significant: sum


s_mov_b32 s72, s40 // (op 1538) reg expression
s_mov_b32 s73, s41 // (op 1538) reg expression
s_mov_b32 s74, s18 // (op 1538) reg expression
s_mov_b32 s75, 131072 // (op 1538) call()

v_lshlrev_b32 v24, 4, v1 // (op 2041) 
s_nop 0
v_readfirstlane_b32 s15, v24 // (op 2041) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v24, 4, v1 // (op 2106) 
s_nop 0
v_readfirstlane_b32 s6, v24 // (op 2106) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -16, v1 // (op 2171) 
v_sub_u32 v25, v1, v24 // (op 2171) 

v_and_b32 v24, 4294967280, v1 // (op 2171) 
v_add_u32 v26, v24, v25 // (op 2171) 


v_and_b32 v24, -16, v1 // (op 2237) 
v_sub_u32 v25, v1, v24 // (op 2237) 

v_and_b32 v24, 4294967280, v1 // (op 2237) 
v_add_u32 v27, v24, v25 // (op 2237) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 458) 
s_and_b32 s7, s18, 4294967294 // (op 458) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 457) 
s_mov_b32 s8, s18 // (op 457) convert

s_cmp_lt_u32 s7, s8 // (op 457) 







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_457 // (op 457) Condition: Top (jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_457 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_457:
s_barrier  // (op 2743) 
s_add_u32 s7, s7, 1 // (op 459) 

s_lshl_b64 s[18:19], s[24:25], 6 // (op 1059) 
v_mov_b64 v[24:25], s[18:19] // (op 1059) convert


v_mov_b64 v[28:29], v[22:23] // (op 1059) 
v_mov_b64 v[32:33], v[28:29] // (op 1059) reg expression
s_add_u32 s8, 2048, s15 // (op 1059) 
s_mov_b32 m0, s8 // (op 1059) reg expression
buffer_load_dwordx4 v32, s[60:63], 0 offen offset: 0 lds // (op 1059) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v24 // (op 1059) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 1059) most significant half

s_add_u32 m0, m0, 1024 // (op 1059) 
buffer_load_dwordx4 v32, s[60:63], 0 offen offset: 0 lds // (op 1059) Load value direct to lds




s_lshl_b64 s[18:19], s[24:25], 7 // (op 1134) 
v_mov_b32 v7, s18 // (op 1134) 
v_mov_b32 v14, s19 // (op 1134) 
v_add_co_u32 v22, s[42:43], v22, v7 // (op 1134) least significant half
v_addc_co_u32 v23, s[42:43], v23, v14, s[42:43] // (op 1134) most significant half




s_barrier  // (op 2762) 

s_lshl_b64 s[18:19], s[28:29], 6 // (op 1065) 
v_mov_b64 v[24:25], s[18:19] // (op 1065) convert


v_mov_b64 v[28:29], v[30:31] // (op 1065) 
v_mov_b64 v[32:33], v[28:29] // (op 1065) reg expression
s_mov_b32 m0, s6 // (op 1065) reg expression
buffer_load_dwordx4 v32, s[64:67], 0 offen offset: 0 lds // (op 1065) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v24 // (op 1065) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 1065) most significant half

s_add_u32 m0, m0, 1024 // (op 1065) 
buffer_load_dwordx4 v32, s[64:67], 0 offen offset: 0 lds // (op 1065) Load value direct to lds



s_lshl_b64 s[18:19], s[28:29], 7 // (op 1272) 
v_mov_b32 v7, s18 // (op 1272) 
v_mov_b32 v14, s19 // (op 1272) 
v_add_co_u32 v30, s[42:43], v30, v7 // (op 1272) least significant half
v_addc_co_u32 v31, s[42:43], v31, v14, s[42:43] // (op 1272) most significant half






v_mov_b32 v7, v21 // (op 477) 
v_mov_b32 v14, v7 // (op 477) reg expression

buffer_load_ubyte v20, v14, s[68:71], 0 offen offset: 0 // (op 477) Load value


s_lshl_b32 s8, s32, 2 // (op 1400) 
v_add_u32 v21, v21, s8 // (op 1400) 



s_barrier  // (op 2782) 
v_mov_b32 v7, v19 // (op 478) 



v_mov_b32 v14, v7 // (op 478) reg expression

buffer_load_ubyte v24, v14, s[72:75], 0 offen offset: 0 // (op 478) Load value


s_lshl_b32 s8, s38, 2 // (op 1528) 
v_add_u32 v19, v19, s8 // (op 1528) 

s_waitcnt vmcnt(1)
ds_write_b8 v26, v20 offset:8256 // (op 474) Store local data 

s_barrier  // (op 2749) 



s_barrier  // (op 2802) 
ds_read_b64_tr_b8 v[32:33], v4 offset:2048 // (op 465) Transpose load local data 



ds_read_b64_tr_b8 v[34:35], v4 offset:2176 // (op 465) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:3072 // (op 465) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v4 offset:3200 // (op 465) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v27, v24 offset:8192 // (op 475) Store local data 

s_barrier  // (op 2769) 



ds_read_b64_tr_b8 v[40:41], v6 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v6 offset:128 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v6 offset:1024 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v6 offset:1152 // (op 467) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2789) 


ds_read_u8 v7, v9 offset:8256 // (op 469) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2809) 


ds_read_u8 v14, v11 offset:8192 // (op 471) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[40:47], a[0:3], v7, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 472) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 457) 
s_mov_b32 s8, s18 // (op 457) convert

s_cmp_lt_u32 s7, s8 // (op 457) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_457 // (op 457) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_457 if true)
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_457:









s_barrier  // (op 2965) 




v_cmp_eq_f32 vcc, 0.00000, v16 // (op 238) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_238 // (op 238) Condition: False, jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_238
s_nop 0
v_accvgpr_read v20, a0 // (op 246) Wait state hazard: XDL Write Hazard
v_accvgpr_read v21, a1 // (op 246) 
v_accvgpr_read v22, a2 // (op 246) 
v_accvgpr_read v23, a3 // (op 246) 
v_mul_f32 v24, v18, v20 // (op 246) 
v_mul_f32 v25, v18, v21 // (op 246) 
v_mul_f32 v26, v18, v22 // (op 246) 
v_mul_f32 v27, v18, v23 // (op 246) 

v_cvt_f16_f32 v20, v24 // (op 248) 
v_cvt_f16_f32 v21, v25 // (op 248) 
v_cvt_f16_f32 v22, v26 // (op 248) 
v_cvt_f16_f32 v23, v27 // (op 248) 

v_lshrrev_b32 v0, 6, v1 // (op 1994) 
s_mul_i32 s15, s16, s50 // (op 1994) 
s_add_u32 s15, s15, s0 // (op 1994) 
v_and_b32 v5, 63, v1 // (op 1994) 
s_add_u32 s6, s15, s9 // (op 1994) 

v_lshrrev_b32 v7, 4, v5 // (op 1994) 
s_mul_hi_u32 s15, s6, s13 // (op 1994) 
v_lshlrev_b32 v12, 2, v7 // (op 1994) 

s_sub_u32 s7, s6, s15 // (op 1994) 
v_lshl_add_u32 v7, v0, 4, v12 // (op 1994) 


s_lshr_b32 s8, s7, 1 // (op 1994) 

v_lshrrev_b32 v0, 2, v5 // (op 1994) 

s_add_u32 s7, s8, s15 // (op 1994) 


v_and_b32 v5, 3, v0 // (op 1994) 

s_lshr_b32 s8, s7, s11 // (op 1994) 

v_and_b32 v0, 3, v1 // (op 1994) 
s_and_b32 s7, s11, 2147483648 // (op 1994) 
v_lshl_add_u32 v12, v5, 2, v0 // (op 1994) 


s_cmp_eq_i32 s7, 0 // (op 1994) 
s_mov_b32 s15, scc // (op 1994) 

s_cmp_eq_i32 s15, 0 // (op 1994) 
s_cselect_b32 s7, s6, s8 // (op 1994) 



s_mul_hi_u32 s6, s7, s12 // (op 1994) 
s_sub_u32 s8, s7, s6 // (op 1994) 
s_lshr_b32 s15, s8, 1 // (op 1994) 

s_add_u32 s8, s15, s6 // (op 1994) 


s_lshr_b32 s6, s8, s10 // (op 1994) 

s_and_b32 s8, s10, 2147483648 // (op 1994) 
s_cmp_eq_i32 s8, 0 // (op 1994) 
s_mov_b32 s15, scc // (op 1994) 

s_cmp_eq_i32 s15, 0 // (op 1994) 
s_cselect_b32 s8, s7, s6 // (op 1994) 


s_mul_i32 s6, s8, s14 // (op 1994) 
v_lshl_add_u32 v0, s8, 4, v7 // (op 1994) 


s_sub_u32 s8, s7, s6 // (op 1994) 


v_lshl_add_u32 v5, s8, 4, v12 // (op 1994) 


v_mul_lo_u32 v7, v5, s46 // (op 1994) 

v_add_lshl_u32 v5, v0, v7, 1 // (op 1994) 


s_mul_i32 s19, s46, s23 // (op 1996) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1996) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1996) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1996) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1996) most significant: sum
s_add_u32 s19, s19, s6 // (op 1996) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 1 // (op 1996) 

s_mov_b32 s76, s48 // (op 1996) reg expression
s_mov_b32 s77, s49 // (op 1996) reg expression
s_mov_b32 s78, s6 // (op 1996) reg expression
s_mov_b32 s79, 131072 // (op 1996) call()


v_pack_b32_f16 v24, v20, v21 // (op 249) 
v_pack_b32_f16 v25, v22, v23 // (op 249) 


buffer_store_dwordx2 v[24:25], v5, s[76:79], 0 offen // (op 249) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_238 // (op 238) Condition: Done, jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_238
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_238:



buffer_load_dwordx2 v[20:21], v3, s[56:59], 0 offen // (op 263) Load value
v_accvgpr_read v22, a0 // (op 259) 
v_accvgpr_read v23, a1 // (op 259) 
v_accvgpr_read v24, a2 // (op 259) 
v_accvgpr_read v25, a3 // (op 259) 
v_mul_f32 v26, v18, v22 // (op 259) 
v_mul_f32 v27, v18, v23 // (op 259) 
v_mul_f32 v28, v18, v24 // (op 259) 
v_mul_f32 v29, v18, v25 // (op 259) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v22, v20 // (op 264) 
v_lshrrev_b32 v23, 16, v20 // (op 264) 
v_cvt_f32_f16 v23, v23 // (op 264) 
v_mul_f32 v30, v16, v22 // (op 264) 
v_mul_f32 v31, v16, v23 // (op 264) 
v_cvt_f32_f16 v22, v21 // (op 264) 
v_lshrrev_b32 v23, 16, v21 // (op 264) 
v_cvt_f32_f16 v23, v23 // (op 264) 
v_mul_f32 v32, v16, v22 // (op 264) 
v_mul_f32 v33, v16, v23 // (op 264) 


v_add_f32 v22, v30, v26 // (op 260) 
v_add_f32 v23, v31, v27 // (op 260) 
v_add_f32 v24, v32, v28 // (op 260) 
v_add_f32 v25, v33, v29 // (op 260) 


v_cvt_f16_f32 v26, v22 // (op 261) 
v_cvt_f16_f32 v27, v23 // (op 261) 
v_cvt_f16_f32 v28, v24 // (op 261) 
v_cvt_f16_f32 v29, v25 // (op 261) 

v_lshrrev_b32 v0, 6, v1 // (op 2011) 
s_mul_i32 s15, s16, s50 // (op 2011) 
s_add_u32 s15, s15, s0 // (op 2011) 
v_and_b32 v5, 63, v1 // (op 2011) 
s_add_u32 s6, s15, s9 // (op 2011) 

v_lshrrev_b32 v7, 4, v5 // (op 2011) 
s_mul_hi_u32 s15, s6, s13 // (op 2011) 
v_lshlrev_b32 v12, 2, v7 // (op 2011) 

s_sub_u32 s7, s6, s15 // (op 2011) 
v_lshl_add_u32 v7, v0, 4, v12 // (op 2011) 


s_lshr_b32 s8, s7, 1 // (op 2011) 

v_lshrrev_b32 v0, 2, v5 // (op 2011) 

s_add_u32 s7, s8, s15 // (op 2011) 


v_and_b32 v5, 3, v0 // (op 2011) 

s_lshr_b32 s8, s7, s11 // (op 2011) 

v_and_b32 v0, 3, v1 // (op 2011) 
s_and_b32 s7, s11, 2147483648 // (op 2011) 
v_lshl_add_u32 v12, v5, 2, v0 // (op 2011) 


s_cmp_eq_i32 s7, 0 // (op 2011) 
s_mov_b32 s15, scc // (op 2011) 

s_cmp_eq_i32 s15, 0 // (op 2011) 
s_cselect_b32 s7, s6, s8 // (op 2011) 



s_mul_hi_u32 s6, s7, s12 // (op 2011) 
s_sub_u32 s8, s7, s6 // (op 2011) 
s_lshr_b32 s15, s8, 1 // (op 2011) 

s_add_u32 s8, s15, s6 // (op 2011) 


s_lshr_b32 s6, s8, s10 // (op 2011) 

s_and_b32 s8, s10, 2147483648 // (op 2011) 
s_cmp_eq_i32 s8, 0 // (op 2011) 
s_mov_b32 s15, scc // (op 2011) 

s_cmp_eq_i32 s15, 0 // (op 2011) 
s_cselect_b32 s8, s7, s6 // (op 2011) 


s_mul_i32 s6, s8, s14 // (op 2011) 
v_lshl_add_u32 v0, s8, 4, v7 // (op 2011) 


s_sub_u32 s8, s7, s6 // (op 2011) 


v_lshl_add_u32 v5, s8, 4, v12 // (op 2011) 


v_mul_lo_u32 v7, v5, s46 // (op 2011) 

v_add_lshl_u32 v5, v0, v7, 1 // (op 2011) 


s_mul_i32 s19, s46, s23 // (op 2013) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 2013) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 2013) least significant
s_mul_hi_u32 s6, s46, s22 // (op 2013) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2013) most significant: sum
s_add_u32 s19, s19, s6 // (op 2013) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 1 // (op 2013) 

s_mov_b32 s76, s48 // (op 2013) reg expression
s_mov_b32 s77, s49 // (op 2013) reg expression
s_mov_b32 s78, s6 // (op 2013) reg expression
s_mov_b32 s79, 131072 // (op 2013) call()


v_pack_b32_f16 v24, v26, v27 // (op 262) 
v_pack_b32_f16 v25, v28, v29 // (op 262) 


buffer_store_dwordx2 v[24:25], v5, s[76:79], 0 offen // (op 262) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_238:

s_add_u32 s0, s0, s1 // (op 451) 
s_mul_i32 s15, s50, s16 // (op 449) 
s_cmp_gt_u32 s51, s15 // (op 449) 
s_mov_b32 s6, scc // (op 449) 
s_sub_u32 s7, s51, s15 // (op 449) 

s_cmp_eq_i32 s6, 0 // (op 449) 
s_cselect_b32 s8, 0, s7 // (op 449) 


s_cmp_lt_u32 s50, s8 // (op 449) 
s_mov_b32 s15, scc // (op 449) 
s_cmp_eq_i32 s15, 0 // (op 449) 
s_cselect_b32 s6, s8, s50 // (op 449) 


s_cmp_lt_u32 s0, s6 // (op 449) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_DPStreamTileLoop_449 // (op 449) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_DPStreamTileLoop_449 if true)
RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_449:










































s_endpgm  // End of RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2
.LRR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_end:
.size RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2, .LRR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_end-RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2
  .amdhsa_next_free_vgpr 76
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 8448 // lds bytes
  .amdhsa_accum_offset 72
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
    DataTypeB: F8
    DestDataType: h
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
    macrotile: [16, 16, 128]
    threads: [64, 1, 1]
    grid: [RocRollerStreamK, One, One]
    occupancy: 19
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
  - .name: RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2
    .symbol: RR_GEMM_NT_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 8448
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 83
    .vgpr_count: 70
    .agpr_count: 4
    .max_flat_workgroup_size: 64
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

