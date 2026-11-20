// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2,@function
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 1438) 
v_and_b32 v3, 15, v1 // (op 1438) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1438) 


v_and_b32 v0, 63, v1 // (op 1438) 
v_lshrrev_b32 v3, 4, v0 // (op 1438) 

v_lshl_add_u32 v0, v4, 2, v3 // (op 1438) 


v_lshrrev_b32 v3, 6, v1 // (op 1561) 
v_and_b32 v4, 15, v1 // (op 1561) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 1561) 


v_and_b32 v3, 63, v1 // (op 1561) 
v_lshrrev_b32 v4, 4, v3 // (op 1561) 

v_lshl_add_u32 v3, v5, 2, v4 // (op 1561) 


v_lshrrev_b32 v4, 6, v1 // (op 1619) 
v_and_b32 v5, 15, v1 // (op 1619) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 1619) 


v_and_b32 v4, 63, v1 // (op 1619) 
v_lshrrev_b32 v5, 4, v4 // (op 1619) 

v_lshl_add_u32 v4, v6, 2, v5 // (op 1619) 


v_lshrrev_b32 v5, 6, v1 // (op 1643) 
v_and_b32 v6, 15, v1 // (op 1643) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 1643) 


v_and_b32 v5, 63, v1 // (op 1643) 
v_lshrrev_b32 v6, 4, v5 // (op 1643) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 1643) 


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
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_241 // (op 241) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_241 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_SKStreamTileLoop_241:
v_lshrrev_b32 v8, 6, v1 // (op 1363) 
s_mul_i32 s52, s16, s8 // (op 1363) 
s_add_u32 s52, s52, s0 // (op 1363) 
v_and_b32 v9, 63, v1 // (op 1363) 
s_mul_hi_u32 s53, s52, s13 // (op 1363) 
v_lshrrev_b32 v10, 4, v9 // (op 1363) 
s_sub_u32 s54, s52, s53 // (op 1363) 
v_lshlrev_b32 v11, 2, v10 // (op 1363) 

v_lshl_add_u32 v10, v8, 4, v11 // (op 1363) 


s_lshr_b32 s55, s54, 1 // (op 1363) 

s_add_u32 s54, s55, s53 // (op 1363) 


v_lshrrev_b32 v8, 2, v9 // (op 1363) 

s_lshr_b32 s53, s54, s11 // (op 1363) 

v_and_b32 v9, 3, v8 // (op 1363) 

s_and_b32 s54, s11, 2147483648 // (op 1363) 
v_and_b32 v8, 3, v1 // (op 1363) 
s_cmp_eq_i32 s54, 0 // (op 1363) 
s_mov_b32 s55, scc // (op 1363) 

v_lshl_add_u32 v11, v9, 2, v8 // (op 1363) 


s_cmp_eq_i32 s55, 0 // (op 1363) 
s_cselect_b32 s54, s52, s53 // (op 1363) 



s_mul_hi_u32 s52, s54, s12 // (op 1363) 
s_sub_u32 s53, s54, s52 // (op 1363) 
s_lshr_b32 s55, s53, 1 // (op 1363) 

s_add_u32 s53, s55, s52 // (op 1363) 


s_lshr_b32 s52, s53, s10 // (op 1363) 

s_and_b32 s53, s10, 2147483648 // (op 1363) 
s_cmp_eq_i32 s53, 0 // (op 1363) 
s_mov_b32 s55, scc // (op 1363) 

s_cmp_eq_i32 s55, 0 // (op 1363) 
s_cselect_b32 s53, s54, s52 // (op 1363) 


s_mul_i32 s52, s53, s14 // (op 1363) 
v_lshl_add_u32 v8, s53, 5, v10 // (op 1363) 


s_sub_u32 s53, s54, s52 // (op 1363) 


v_lshl_add_u32 v9, s53, 4, v11 // (op 1363) 


v_mul_lo_u32 v10, v9, s4 // (op 1363) 

v_add_lshl_u32 v9, v8, v10, 2 // (op 1363) 


s_mul_i32 s53, s4, s23 // (op 1365) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 1365) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 1365) least significant
s_mul_hi_u32 s55, s4, s22 // (op 1365) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1365) most significant: sum
s_add_u32 s53, s53, s55 // (op 1365) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1365) 

s_mov_b32 s56, s20 // (op 1365) reg expression
s_mov_b32 s57, s21 // (op 1365) reg expression
s_mov_b32 s58, s54 // (op 1365) reg expression
s_mov_b32 s59, 131072 // (op 1365) call()

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
v_lshrrev_b32 v8, 6, v1 // (op 895) 
s_mul_i32 s52, s16, s8 // (op 895) 
s_add_u32 s52, s52, s0 // (op 895) 
v_and_b32 v10, 15, v1 // (op 895) 
s_mul_hi_u32 s53, s52, s13 // (op 895) 
v_lshl_add_u32 v11, v8, 4, v10 // (op 895) 


s_sub_u32 s54, s52, s53 // (op 895) 
v_and_b32 v8, 63, v1 // (op 895) 
s_lshr_b32 s55, s54, 1 // (op 895) 

v_lshrrev_b32 v10, 4, v8 // (op 895) 

s_add_u32 s54, s55, s53 // (op 895) 


v_lshlrev_b32 v8, 5, v10 // (op 895) 

s_lshr_b32 s53, s54, s11 // (op 895) 

s_and_b32 s54, s11, 2147483648 // (op 895) 
s_cmp_eq_i32 s54, 0 // (op 895) 
s_mov_b32 s55, scc // (op 895) 

s_cmp_eq_i32 s55, 0 // (op 895) 
s_cselect_b32 s54, s52, s53 // (op 895) 


s_mul_hi_u32 s53, s54, s12 // (op 895) 
s_sub_u32 s55, s54, s53 // (op 895) 
s_lshr_b32 s60, s55, 1 // (op 895) 

s_add_u32 s55, s60, s53 // (op 895) 


s_lshr_b32 s53, s55, s10 // (op 895) 

s_and_b32 s55, s10, 2147483648 // (op 895) 
s_cmp_eq_i32 s55, 0 // (op 895) 
s_mov_b32 s60, scc // (op 895) 

s_cmp_eq_i32 s60, 0 // (op 895) 
s_cselect_b32 s55, s54, s53 // (op 895) 


v_lshl_add_u32 v10, s55, 5, v11 // (op 895) 


s_ashr_i64 s[60:61], s[2:3], 7 // (op 895) 
s_mul_i32 s53, s54, s60 // (op 895) 


s_sub_u32 s54, s52, s53 // (op 895) 


v_lshl_add_u32 v11, s54, 7, v8 // (op 895) 


v_mov_b32 v13, 0 // (op 895) convert
v_mov_b32 v12, v11 // (op 895) convert

v_mul_lo_u32 v15, v10, s25 // (op 895) most significant: low of low * high
v_mul_hi_u32 v8, v10, s24 // (op 895) most significant: high of low * low
v_mul_lo_u32 v14, v10, s24 // (op 895) least significant: low of low * low
v_add_u32 v15, v15, v8 // (op 895) most significant: sum

v_add_co_u32 v14, s[52:53], v14, v12 // (op 895) least significant half
v_addc_co_u32 v15, s[52:53], v15, v13, s[52:53] // (op 895) most significant half



v_lshlrev_b64 v[10:11], 2, v[14:15] // (op 895) 

v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 895) 

v_mov_b32 v8, v12 // (op 895) convert

s_mul_i32 s53, s24, s29 // (op 897) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 897) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 897) least significant
s_mul_hi_u32 s55, s24, s28 // (op 897) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 897) most significant: sum
s_add_u32 s53, s53, s55 // (op 897) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 897) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 897) 

s_mov_b32 s60, s26 // (op 897) reg expression
s_mov_b32 s61, s27 // (op 897) reg expression
s_mov_b32 s62, s52 // (op 897) reg expression
s_mov_b32 s63, 131072 // (op 897) call()

v_lshrrev_b32 v10, 6, v1 // (op 914) 
s_mul_i32 s52, s16, s8 // (op 914) 
s_add_u32 s52, s52, s0 // (op 914) 
v_and_b32 v11, 15, v1 // (op 914) 
s_mul_hi_u32 s53, s52, s13 // (op 914) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 914) 


s_sub_u32 s54, s52, s53 // (op 914) 
v_and_b32 v10, 63, v1 // (op 914) 
s_lshr_b32 s55, s54, 1 // (op 914) 

v_lshrrev_b32 v11, 4, v10 // (op 914) 

s_add_u32 s54, s55, s53 // (op 914) 


v_lshlrev_b32 v10, 5, v11 // (op 914) 

s_lshr_b32 s53, s54, s11 // (op 914) 

s_and_b32 s54, s11, 2147483648 // (op 914) 
s_cmp_eq_i32 s54, 0 // (op 914) 
s_mov_b32 s55, scc // (op 914) 

s_cmp_eq_i32 s55, 0 // (op 914) 
s_cselect_b32 s54, s52, s53 // (op 914) 


s_mul_hi_u32 s53, s54, s12 // (op 914) 
s_sub_u32 s55, s54, s53 // (op 914) 
s_lshr_b32 s64, s55, 1 // (op 914) 

s_add_u32 s55, s64, s53 // (op 914) 


s_lshr_b32 s53, s55, s10 // (op 914) 

s_and_b32 s55, s10, 2147483648 // (op 914) 
s_cmp_eq_i32 s55, 0 // (op 914) 
s_mov_b32 s64, scc // (op 914) 

s_cmp_eq_i32 s64, 0 // (op 914) 
s_cselect_b32 s55, s54, s53 // (op 914) 


v_lshl_add_u32 v11, s55, 5, v12 // (op 914) 


s_ashr_i64 s[64:65], s[2:3], 7 // (op 914) 
s_mul_i32 s53, s54, s64 // (op 914) 


s_sub_u32 s54, s52, s53 // (op 914) 


v_lshl_add_u32 v12, s54, 7, v10 // (op 914) 


v_mov_b32 v15, 0 // (op 914) convert
v_mov_b32 v14, v12 // (op 914) convert

v_mul_lo_u32 v13, v11, s25 // (op 914) most significant: low of low * high
v_mul_hi_u32 v10, v11, s24 // (op 914) most significant: high of low * low
v_mul_lo_u32 v12, v11, s24 // (op 914) least significant: low of low * low
v_add_u32 v13, v13, v10 // (op 914) most significant: sum

v_add_co_u32 v12, s[52:53], v12, v14 // (op 914) least significant half
v_addc_co_u32 v13, s[52:53], v13, v15, s[52:53] // (op 914) most significant half



v_lshlrev_b64 v[10:11], 2, v[12:13] // (op 914) 

v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 914) 

v_mov_b32 v10, v12 // (op 914) convert

s_mul_i32 s53, s24, s29 // (op 916) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 916) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 916) least significant
s_mul_hi_u32 s55, s24, s28 // (op 916) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 916) most significant: sum
s_add_u32 s53, s53, s55 // (op 916) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 916) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 916) 

s_mov_b32 s60, s26 // (op 916) reg expression
s_mov_b32 s61, s27 // (op 916) reg expression
s_mov_b32 s62, s52 // (op 916) reg expression
s_mov_b32 s63, 131072 // (op 916) call()

v_and_b32 v11, 15, v1 // (op 1026) 
s_mul_i32 s52, s16, s8 // (op 1026) 
s_add_u32 s52, s52, s0 // (op 1026) 
v_and_b32 v12, 63, v1 // (op 1026) 
s_mul_hi_u32 s53, s52, s13 // (op 1026) 
v_lshrrev_b32 v13, 4, v12 // (op 1026) 

s_sub_u32 s54, s52, s53 // (op 1026) 
v_lshlrev_b32 v12, 5, v13 // (op 1026) 

s_lshr_b32 s55, s54, 1 // (op 1026) 

s_add_u32 s54, s55, s53 // (op 1026) 


s_lshr_b32 s53, s54, s11 // (op 1026) 

s_and_b32 s54, s11, 2147483648 // (op 1026) 
s_cmp_eq_i32 s54, 0 // (op 1026) 
s_mov_b32 s55, scc // (op 1026) 

s_cmp_eq_i32 s55, 0 // (op 1026) 
s_cselect_b32 s54, s52, s53 // (op 1026) 


s_mul_hi_u32 s53, s54, s12 // (op 1026) 
s_sub_u32 s55, s54, s53 // (op 1026) 
s_lshr_b32 s64, s55, 1 // (op 1026) 

s_add_u32 s55, s64, s53 // (op 1026) 


s_lshr_b32 s53, s55, s10 // (op 1026) 

s_and_b32 s55, s10, 2147483648 // (op 1026) 
s_cmp_eq_i32 s55, 0 // (op 1026) 
s_mov_b32 s64, scc // (op 1026) 

s_cmp_eq_i32 s64, 0 // (op 1026) 
s_cselect_b32 s55, s54, s53 // (op 1026) 


s_mul_i32 s53, s55, s14 // (op 1026) 

s_sub_u32 s55, s54, s53 // (op 1026) 

v_lshl_add_u32 v13, s55, 4, v11 // (op 1026) 


v_mul_lo_u32 v15, v13, s31 // (op 1026) most significant: low of low * high
v_mul_hi_u32 v11, v13, s30 // (op 1026) most significant: high of low * low
v_mul_lo_u32 v14, v13, s30 // (op 1026) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 1026) most significant: sum


s_ashr_i64 s[64:65], s[2:3], 7 // (op 1026) 
s_mul_i32 s53, s54, s64 // (op 1026) 


s_sub_u32 s54, s52, s53 // (op 1026) 


v_lshl_add_u32 v11, s54, 7, v12 // (op 1026) 


v_mov_b32 v13, 0 // (op 1026) convert
v_mov_b32 v12, v11 // (op 1026) convert

v_add_co_u32 v16, s[52:53], v12, v14 // (op 1026) least significant half
v_addc_co_u32 v17, s[52:53], v13, v15, s[52:53] // (op 1026) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 1026) 


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 1026) 

v_mov_b32 v11, v14 // (op 1026) convert

s_mul_i32 s53, s30, s23 // (op 1028) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 1028) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 1028) least significant
s_mul_hi_u32 s55, s30, s22 // (op 1028) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1028) most significant: sum
s_add_u32 s53, s53, s55 // (op 1028) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1028) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1028) 

s_mov_b32 s64, s32 // (op 1028) reg expression
s_mov_b32 s65, s33 // (op 1028) reg expression
s_mov_b32 s66, s52 // (op 1028) reg expression
s_mov_b32 s67, 131072 // (op 1028) call()

v_and_b32 v12, 15, v1 // (op 1045) 
s_mul_i32 s52, s16, s8 // (op 1045) 
s_add_u32 s52, s52, s0 // (op 1045) 
v_and_b32 v13, 63, v1 // (op 1045) 
s_mul_hi_u32 s53, s52, s13 // (op 1045) 
v_lshrrev_b32 v14, 4, v13 // (op 1045) 

s_sub_u32 s54, s52, s53 // (op 1045) 
v_lshlrev_b32 v13, 5, v14 // (op 1045) 

s_lshr_b32 s55, s54, 1 // (op 1045) 

s_add_u32 s54, s55, s53 // (op 1045) 


s_lshr_b32 s53, s54, s11 // (op 1045) 

s_and_b32 s54, s11, 2147483648 // (op 1045) 
s_cmp_eq_i32 s54, 0 // (op 1045) 
s_mov_b32 s55, scc // (op 1045) 

s_cmp_eq_i32 s55, 0 // (op 1045) 
s_cselect_b32 s54, s52, s53 // (op 1045) 


s_mul_hi_u32 s53, s54, s12 // (op 1045) 
s_sub_u32 s55, s54, s53 // (op 1045) 
s_lshr_b32 s68, s55, 1 // (op 1045) 

s_add_u32 s55, s68, s53 // (op 1045) 


s_lshr_b32 s53, s55, s10 // (op 1045) 

s_and_b32 s55, s10, 2147483648 // (op 1045) 
s_cmp_eq_i32 s55, 0 // (op 1045) 
s_mov_b32 s68, scc // (op 1045) 

s_cmp_eq_i32 s68, 0 // (op 1045) 
s_cselect_b32 s55, s54, s53 // (op 1045) 


s_mul_i32 s53, s55, s14 // (op 1045) 

s_sub_u32 s55, s54, s53 // (op 1045) 

v_lshl_add_u32 v14, s55, 4, v12 // (op 1045) 


v_mul_lo_u32 v17, v14, s31 // (op 1045) most significant: low of low * high
v_mul_hi_u32 v12, v14, s30 // (op 1045) most significant: high of low * low
v_mul_lo_u32 v16, v14, s30 // (op 1045) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 1045) most significant: sum


s_ashr_i64 s[68:69], s[2:3], 7 // (op 1045) 
s_mul_i32 s53, s54, s68 // (op 1045) 


s_sub_u32 s54, s52, s53 // (op 1045) 


v_lshl_add_u32 v12, s54, 7, v13 // (op 1045) 


v_mov_b32 v15, 0 // (op 1045) convert
v_mov_b32 v14, v12 // (op 1045) convert

v_add_co_u32 v12, s[52:53], v14, v16 // (op 1045) least significant half
v_addc_co_u32 v13, s[52:53], v15, v17, s[52:53] // (op 1045) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 1045) 


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1045) 

v_mov_b32 v12, v14 // (op 1045) convert

s_mul_i32 s53, s30, s23 // (op 1047) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 1047) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 1047) least significant
s_mul_hi_u32 s55, s30, s22 // (op 1047) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1047) most significant: sum
s_add_u32 s53, s53, s55 // (op 1047) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1047) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1047) 

s_mov_b32 s64, s32 // (op 1047) reg expression
s_mov_b32 s65, s33 // (op 1047) reg expression
s_mov_b32 s66, s52 // (op 1047) reg expression
s_mov_b32 s67, 131072 // (op 1047) call()

v_lshrrev_b32 v13, 2, v1 // (op 1154) 
s_mul_i32 s52, s16, s8 // (op 1154) 
s_add_u32 s52, s52, s0 // (op 1154) 
v_and_b32 v14, -4, v1 // (op 1154) 
s_mul_hi_u32 s53, s52, s13 // (op 1154) 
v_sub_u32 v15, v1, v14 // (op 1154) 

s_sub_u32 s54, s52, s53 // (op 1154) 
s_lshr_b32 s55, s54, 1 // (op 1154) 

s_add_u32 s54, s55, s53 // (op 1154) 


s_lshr_b32 s53, s54, s11 // (op 1154) 

s_and_b32 s54, s11, 2147483648 // (op 1154) 
s_cmp_eq_i32 s54, 0 // (op 1154) 
s_mov_b32 s55, scc // (op 1154) 

s_cmp_eq_i32 s55, 0 // (op 1154) 
s_cselect_b32 s54, s52, s53 // (op 1154) 


s_mul_hi_u32 s53, s54, s12 // (op 1154) 
s_sub_u32 s55, s54, s53 // (op 1154) 
s_lshr_b32 s68, s55, 1 // (op 1154) 

s_add_u32 s55, s68, s53 // (op 1154) 


s_lshr_b32 s53, s55, s10 // (op 1154) 

s_and_b32 s55, s10, 2147483648 // (op 1154) 
s_cmp_eq_i32 s55, 0 // (op 1154) 
s_mov_b32 s68, scc // (op 1154) 

s_cmp_eq_i32 s68, 0 // (op 1154) 
s_cselect_b32 s55, s54, s53 // (op 1154) 


v_lshl_add_u32 v14, s55, 5, v13 // (op 1154) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 1154) 
s_mul_i32 s53, s54, s68 // (op 1154) 


s_sub_u32 s54, s52, s53 // (op 1154) 


v_lshl_add_u32 v13, s54, 2, v15 // (op 1154) 


v_mul_lo_u32 v15, v14, s34 // (op 1154) 
v_add_u32 v15, v15, v13 // (op 1154) 


s_mul_i32 s53, s34, s29 // (op 1156) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 1156) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 1156) least significant
s_mul_hi_u32 s55, s34, s28 // (op 1156) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1156) most significant: sum
s_add_u32 s53, s53, s55 // (op 1156) most significant: sum


s_mov_b32 s68, s36 // (op 1156) reg expression
s_mov_b32 s69, s37 // (op 1156) reg expression
s_mov_b32 s70, s52 // (op 1156) reg expression
s_mov_b32 s71, 131072 // (op 1156) call()

v_lshrrev_b32 v13, 2, v1 // (op 1173) 
s_mul_i32 s52, s16, s8 // (op 1173) 
s_add_u32 s52, s52, s0 // (op 1173) 
v_and_b32 v14, -4, v1 // (op 1173) 
s_mul_hi_u32 s53, s52, s13 // (op 1173) 
v_sub_u32 v16, v1, v14 // (op 1173) 

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
s_lshr_b32 s72, s55, 1 // (op 1173) 

s_add_u32 s55, s72, s53 // (op 1173) 


s_lshr_b32 s53, s55, s10 // (op 1173) 

s_and_b32 s55, s10, 2147483648 // (op 1173) 
s_cmp_eq_i32 s55, 0 // (op 1173) 
s_mov_b32 s72, scc // (op 1173) 

s_cmp_eq_i32 s72, 0 // (op 1173) 
s_cselect_b32 s55, s54, s53 // (op 1173) 


v_lshl_add_u32 v14, s55, 5, v13 // (op 1173) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 1173) 
s_mul_i32 s53, s54, s72 // (op 1173) 


s_sub_u32 s54, s52, s53 // (op 1173) 


v_lshl_add_u32 v13, s54, 2, v16 // (op 1173) 


v_mul_lo_u32 v16, v14, s34 // (op 1173) 
v_add_u32 v16, v16, v13 // (op 1173) 


s_mul_i32 s53, s34, s29 // (op 1175) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 1175) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 1175) least significant
s_mul_hi_u32 s55, s34, s28 // (op 1175) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1175) most significant: sum
s_add_u32 s53, s53, s55 // (op 1175) most significant: sum


s_mov_b32 s68, s36 // (op 1175) reg expression
s_mov_b32 s69, s37 // (op 1175) reg expression
s_mov_b32 s70, s52 // (op 1175) reg expression
s_mov_b32 s71, 131072 // (op 1175) call()

v_and_b32 v13, 15, v1 // (op 1282) 
s_mul_i32 s52, s16, s8 // (op 1282) 
s_add_u32 s52, s52, s0 // (op 1282) 
v_and_b32 v14, 63, v1 // (op 1282) 
s_mul_hi_u32 s53, s52, s13 // (op 1282) 
v_lshrrev_b32 v17, 4, v14 // (op 1282) 

s_sub_u32 s54, s52, s53 // (op 1282) 
s_lshr_b32 s55, s54, 1 // (op 1282) 

s_add_u32 s54, s55, s53 // (op 1282) 


s_lshr_b32 s53, s54, s11 // (op 1282) 

s_and_b32 s54, s11, 2147483648 // (op 1282) 
s_cmp_eq_i32 s54, 0 // (op 1282) 
s_mov_b32 s55, scc // (op 1282) 

s_cmp_eq_i32 s55, 0 // (op 1282) 
s_cselect_b32 s54, s52, s53 // (op 1282) 


s_mul_hi_u32 s53, s54, s12 // (op 1282) 
s_sub_u32 s55, s54, s53 // (op 1282) 
s_lshr_b32 s72, s55, 1 // (op 1282) 

s_add_u32 s55, s72, s53 // (op 1282) 


s_lshr_b32 s53, s55, s10 // (op 1282) 

s_and_b32 s55, s10, 2147483648 // (op 1282) 
s_cmp_eq_i32 s55, 0 // (op 1282) 
s_mov_b32 s72, scc // (op 1282) 

s_cmp_eq_i32 s72, 0 // (op 1282) 
s_cselect_b32 s55, s54, s53 // (op 1282) 


s_mul_i32 s53, s55, s14 // (op 1282) 

s_sub_u32 s55, s54, s53 // (op 1282) 

v_lshl_add_u32 v14, s55, 4, v13 // (op 1282) 


v_mul_lo_u32 v13, v14, s38 // (op 1282) 

s_ashr_i64 s[72:73], s[2:3], 7 // (op 1282) 
s_mul_i32 s53, s54, s72 // (op 1282) 


s_sub_u32 s54, s52, s53 // (op 1282) 


v_lshl_add_u32 v14, s54, 2, v17 // (op 1282) 


v_add_u32 v17, v14, v13 // (op 1282) 


s_mul_i32 s53, s38, s23 // (op 1284) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 1284) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 1284) least significant
s_mul_hi_u32 s55, s38, s22 // (op 1284) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1284) most significant: sum
s_add_u32 s53, s53, s55 // (op 1284) most significant: sum


s_mov_b32 s72, s40 // (op 1284) reg expression
s_mov_b32 s73, s41 // (op 1284) reg expression
s_mov_b32 s74, s52 // (op 1284) reg expression
s_mov_b32 s75, 131072 // (op 1284) call()

v_and_b32 v13, 15, v1 // (op 1301) 
s_mul_i32 s52, s16, s8 // (op 1301) 
s_add_u32 s52, s52, s0 // (op 1301) 
v_and_b32 v14, 63, v1 // (op 1301) 
s_mul_hi_u32 s53, s52, s13 // (op 1301) 
v_lshrrev_b32 v18, 4, v14 // (op 1301) 

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


s_mul_i32 s53, s55, s14 // (op 1301) 

s_sub_u32 s55, s54, s53 // (op 1301) 

v_lshl_add_u32 v14, s55, 4, v13 // (op 1301) 


v_mul_lo_u32 v13, v14, s38 // (op 1301) 

s_ashr_i64 s[76:77], s[2:3], 7 // (op 1301) 
s_mul_i32 s53, s54, s76 // (op 1301) 


s_sub_u32 s54, s52, s53 // (op 1301) 


v_lshl_add_u32 v14, s54, 2, v18 // (op 1301) 


v_add_u32 v18, v14, v13 // (op 1301) 


s_mul_i32 s53, s38, s23 // (op 1303) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 1303) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 1303) least significant
s_mul_hi_u32 s55, s38, s22 // (op 1303) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1303) most significant: sum
s_add_u32 s53, s53, s55 // (op 1303) most significant: sum


s_mov_b32 s72, s40 // (op 1303) reg expression
s_mov_b32 s73, s41 // (op 1303) reg expression
s_mov_b32 s74, s52 // (op 1303) reg expression
s_mov_b32 s75, 131072 // (op 1303) call()




buffer_load_dwordx4 v[20:23], v8, s[60:63], 0 offen // (op 685) Load value



buffer_load_dwordx4 v[24:27], v11, s[64:67], 0 offen // (op 690) Load value



buffer_load_ubyte v19, v17, s[72:75], 0 offen offset: 0 // (op 695) Load value



buffer_load_ubyte v13, v15, s[68:71], 0 offen offset: 0 // (op 699) Load value



buffer_load_ubyte v14, v17, s[72:75], 0 offen offset:4 // (op 703) Load value




buffer_load_dwordx4 v[28:31], v8, s[60:63], 0 offen offset:64 // (op 708) Load value




buffer_load_dwordx4 v[32:35], v11, s[64:67], 0 offen offset:64 // (op 713) Load value




buffer_load_ubyte v8, v15, s[68:71], 0 offen offset:4 // (op 717) Load value

v_and_b32 v11, -4, v1 // (op 1426) 
v_sub_u32 v15, v1, v11 // (op 1426) 

v_and_b32 v11, 4294967292, v1 // (op 1426) 
v_add_u32 v17, v11, v15 // (op 1426) 


s_barrier  // (op 1700) 



s_waitcnt vmcnt(4)
ds_write_b8 v17, v13 // (op 719) Store local data 

v_and_b32 v11, -4, v1 // (op 1397) 
v_sub_u32 v15, v1, v11 // (op 1397) 

v_and_b32 v11, 4294967292, v1 // (op 1397) 
v_add_u32 v17, v11, v15 // (op 1397) 


v_and_b32 v11, -4, v1 // (op 1631) 
v_sub_u32 v15, v1, v11 // (op 1631) 

v_and_b32 v11, 4294967292, v1 // (op 1631) 
v_add_u32 v36, v11, v15 // (op 1631) 


s_mov_b32 s1, 0 // (op 47) call()
s_and_b32 s52, s7, 4294967294 // (op 46) 
s_cmp_lt_u32 s1, s52 // (op 46) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_46 // (op 46) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_46 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoop_46:
s_add_u32 s1, s1, 2 // (op 48) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1697) 



ds_read_u8 v11, v0 // (op 63) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:27], a[0:3], v11, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 70) 



v_mov_b32 v11, v10 // (op 55) 

buffer_load_dwordx4 v[20:23], v11, s[60:63], 0 offen offset:128 // (op 55) Load value


v_mov_b32 v15, v12 // (op 59) 

buffer_load_dwordx4 v[24:27], v15, s[64:67], 0 offen offset:128 // (op 59) Load value


v_mov_b32 v37, v18 // (op 68) 

buffer_load_ubyte v19, v37, s[72:75], 0 offen offset:8 // (op 68) Load value


v_mov_b32 v38, v16 // (op 9) 

buffer_load_ubyte v13, v38, s[68:71], 0 offen offset:8 // (op 9) Load value
s_barrier  // (op 1738) 



s_waitcnt vmcnt(4)
ds_write_b8 v36, v8 offset:128 // (op 557) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1735) 



ds_read_u8 v39, v5 offset:128 // (op 559) Load local data 
s_barrier  // (op 1863) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:31], v[32:35], a[0:3], v39, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 562) 




buffer_load_ubyte v14, v37, s[72:75], 0 offen offset:12 // (op 561) Load value




v_add_u32 v18, 8, v18 // (op 1293) 
buffer_load_dwordx4 v[28:31], v11, s[60:63], 0 offen offset:192 // (op 564) Load value




v_add_u32 v10, 128, v10 // (op 906) 
buffer_load_dwordx4 v[32:35], v15, s[64:67], 0 offen offset:192 // (op 566) Load value




v_add_u32 v12, 128, v12 // (op 1037) 
buffer_load_ubyte v8, v38, s[68:71], 0 offen offset:12 // (op 556) Load value

s_barrier  // (op 1703) 
v_add_u32 v16, 8, v16 // (op 1165) 



s_waitcnt vmcnt(4)
ds_write_b8 v17, v13 // (op 13) Store local data 
s_and_b32 s52, s7, 4294967294 // (op 46) 
s_cmp_lt_u32 s1, s52 // (op 46) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoop_46 // (op 46) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoop_46 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_46:










s_and_b32 s52, s7, 4294967294 // (op 869) 
v_lshrrev_b32 v10, 6, v1 // (op 869) 
s_add_u32 s53, s0, s52 // (op 869) 

v_and_b32 v11, 15, v1 // (op 869) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 869) 


s_mul_i32 s52, s16, s8 // (op 869) 
s_add_u32 s52, s52, s53 // (op 869) 

v_and_b32 v13, 63, v1 // (op 869) 
s_mul_hi_u32 s53, s52, s13 // (op 869) 
v_lshrrev_b32 v10, 4, v13 // (op 869) 

s_sub_u32 s54, s52, s53 // (op 869) 
v_lshlrev_b32 v11, 5, v10 // (op 869) 

s_lshr_b32 s55, s54, 1 // (op 869) 

s_add_u32 s54, s55, s53 // (op 869) 


s_lshr_b32 s53, s54, s11 // (op 869) 

s_and_b32 s54, s11, 2147483648 // (op 869) 
s_cmp_eq_i32 s54, 0 // (op 869) 
s_mov_b32 s55, scc // (op 869) 

s_cmp_eq_i32 s55, 0 // (op 869) 
s_cselect_b32 s54, s52, s53 // (op 869) 


s_mul_hi_u32 s53, s54, s12 // (op 869) 
s_sub_u32 s55, s54, s53 // (op 869) 
s_lshr_b32 s76, s55, 1 // (op 869) 

s_add_u32 s55, s76, s53 // (op 869) 


s_lshr_b32 s53, s55, s10 // (op 869) 

s_and_b32 s55, s10, 2147483648 // (op 869) 
s_cmp_eq_i32 s55, 0 // (op 869) 
s_mov_b32 s76, scc // (op 869) 

s_cmp_eq_i32 s76, 0 // (op 869) 
s_cselect_b32 s55, s54, s53 // (op 869) 


v_lshl_add_u32 v10, s55, 5, v12 // (op 869) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 869) 
s_mul_i32 s53, s54, s76 // (op 869) 


s_sub_u32 s54, s52, s53 // (op 869) 


v_lshl_add_u32 v12, s54, 7, v11 // (op 869) 


v_mov_b32 v27, 0 // (op 869) convert
v_mov_b32 v26, v12 // (op 869) convert

v_mul_lo_u32 v13, v10, s25 // (op 869) most significant: low of low * high
v_mul_hi_u32 v11, v10, s24 // (op 869) most significant: high of low * low
v_mul_lo_u32 v12, v10, s24 // (op 869) least significant: low of low * low
v_add_u32 v13, v13, v11 // (op 869) most significant: sum

v_add_co_u32 v12, s[52:53], v12, v26 // (op 869) least significant half
v_addc_co_u32 v13, s[52:53], v13, v27, s[52:53] // (op 869) most significant half



v_lshlrev_b64 v[10:11], 2, v[12:13] // (op 869) 

v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 869) 

v_mov_b32 v10, v12 // (op 869) convert

s_mul_i32 s53, s24, s29 // (op 871) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 871) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 871) least significant
s_mul_hi_u32 s55, s24, s28 // (op 871) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 871) most significant: sum
s_add_u32 s53, s53, s55 // (op 871) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 871) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 871) 

s_mov_b32 s60, s26 // (op 871) reg expression
s_mov_b32 s61, s27 // (op 871) reg expression
s_mov_b32 s62, s52 // (op 871) reg expression
s_mov_b32 s63, 131072 // (op 871) call()

s_and_b32 s52, s7, 4294967294 // (op 1003) 
v_and_b32 v11, 15, v1 // (op 1003) 
s_add_u32 s53, s0, s52 // (op 1003) 

v_and_b32 v12, 63, v1 // (op 1003) 
v_lshrrev_b32 v13, 4, v12 // (op 1003) 

s_mul_i32 s52, s16, s8 // (op 1003) 
s_add_u32 s52, s52, s53 // (op 1003) 

v_lshlrev_b32 v12, 5, v13 // (op 1003) 

s_mul_hi_u32 s53, s52, s13 // (op 1003) 
s_sub_u32 s54, s52, s53 // (op 1003) 
s_lshr_b32 s55, s54, 1 // (op 1003) 

s_add_u32 s54, s55, s53 // (op 1003) 


s_lshr_b32 s53, s54, s11 // (op 1003) 

s_and_b32 s54, s11, 2147483648 // (op 1003) 
s_cmp_eq_i32 s54, 0 // (op 1003) 
s_mov_b32 s55, scc // (op 1003) 

s_cmp_eq_i32 s55, 0 // (op 1003) 
s_cselect_b32 s54, s52, s53 // (op 1003) 


s_mul_hi_u32 s53, s54, s12 // (op 1003) 
s_sub_u32 s55, s54, s53 // (op 1003) 
s_lshr_b32 s76, s55, 1 // (op 1003) 

s_add_u32 s55, s76, s53 // (op 1003) 


s_lshr_b32 s53, s55, s10 // (op 1003) 

s_and_b32 s55, s10, 2147483648 // (op 1003) 
s_cmp_eq_i32 s55, 0 // (op 1003) 
s_mov_b32 s76, scc // (op 1003) 

s_cmp_eq_i32 s76, 0 // (op 1003) 
s_cselect_b32 s55, s54, s53 // (op 1003) 


s_mul_i32 s53, s55, s14 // (op 1003) 

s_sub_u32 s55, s54, s53 // (op 1003) 

v_lshl_add_u32 v13, s55, 4, v11 // (op 1003) 


v_mul_lo_u32 v27, v13, s31 // (op 1003) most significant: low of low * high
v_mul_hi_u32 v11, v13, s30 // (op 1003) most significant: high of low * low
v_mul_lo_u32 v26, v13, s30 // (op 1003) least significant: low of low * low
v_add_u32 v27, v27, v11 // (op 1003) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1003) 
s_mul_i32 s53, s54, s76 // (op 1003) 


s_sub_u32 s54, s52, s53 // (op 1003) 


v_lshl_add_u32 v11, s54, 7, v12 // (op 1003) 


v_mov_b32 v13, 0 // (op 1003) convert
v_mov_b32 v12, v11 // (op 1003) convert

v_add_co_u32 v24, s[52:53], v12, v26 // (op 1003) least significant half
v_addc_co_u32 v25, s[52:53], v13, v27, s[52:53] // (op 1003) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 1003) 


v_lshrrev_b64 v[12:13], 3, v[24:25] // (op 1003) 

v_mov_b32 v11, v12 // (op 1003) convert

s_mul_i32 s53, s30, s23 // (op 1005) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 1005) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 1005) least significant
s_mul_hi_u32 s55, s30, s22 // (op 1005) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1005) most significant: sum
s_add_u32 s53, s53, s55 // (op 1005) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1005) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1005) 

s_mov_b32 s64, s32 // (op 1005) reg expression
s_mov_b32 s65, s33 // (op 1005) reg expression
s_mov_b32 s66, s52 // (op 1005) reg expression
s_mov_b32 s67, 131072 // (op 1005) call()

s_and_b32 s52, s7, 4294967294 // (op 1131) 
v_lshrrev_b32 v12, 2, v1 // (op 1131) 
s_add_u32 s53, s0, s52 // (op 1131) 

v_and_b32 v13, -4, v1 // (op 1131) 
v_sub_u32 v15, v1, v13 // (op 1131) 

s_mul_i32 s52, s16, s8 // (op 1131) 
s_add_u32 s52, s52, s53 // (op 1131) 

s_mul_hi_u32 s53, s52, s13 // (op 1131) 
s_sub_u32 s54, s52, s53 // (op 1131) 
s_lshr_b32 s55, s54, 1 // (op 1131) 

s_add_u32 s54, s55, s53 // (op 1131) 


s_lshr_b32 s53, s54, s11 // (op 1131) 

s_and_b32 s54, s11, 2147483648 // (op 1131) 
s_cmp_eq_i32 s54, 0 // (op 1131) 
s_mov_b32 s55, scc // (op 1131) 

s_cmp_eq_i32 s55, 0 // (op 1131) 
s_cselect_b32 s54, s52, s53 // (op 1131) 


s_mul_hi_u32 s53, s54, s12 // (op 1131) 
s_sub_u32 s55, s54, s53 // (op 1131) 
s_lshr_b32 s76, s55, 1 // (op 1131) 

s_add_u32 s55, s76, s53 // (op 1131) 


s_lshr_b32 s53, s55, s10 // (op 1131) 

s_and_b32 s55, s10, 2147483648 // (op 1131) 
s_cmp_eq_i32 s55, 0 // (op 1131) 
s_mov_b32 s76, scc // (op 1131) 

s_cmp_eq_i32 s76, 0 // (op 1131) 
s_cselect_b32 s55, s54, s53 // (op 1131) 


v_lshl_add_u32 v13, s55, 5, v12 // (op 1131) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1131) 
s_mul_i32 s53, s54, s76 // (op 1131) 


s_sub_u32 s54, s52, s53 // (op 1131) 


v_lshl_add_u32 v12, s54, 2, v15 // (op 1131) 


v_mul_lo_u32 v15, v13, s34 // (op 1131) 
v_add_u32 v15, v15, v12 // (op 1131) 


s_mul_i32 s53, s34, s29 // (op 1133) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 1133) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 1133) least significant
s_mul_hi_u32 s55, s34, s28 // (op 1133) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1133) most significant: sum
s_add_u32 s53, s53, s55 // (op 1133) most significant: sum


s_mov_b32 s68, s36 // (op 1133) reg expression
s_mov_b32 s69, s37 // (op 1133) reg expression
s_mov_b32 s70, s52 // (op 1133) reg expression
s_mov_b32 s71, 131072 // (op 1133) call()

s_and_b32 s52, s7, 4294967294 // (op 1259) 
v_and_b32 v12, 15, v1 // (op 1259) 
s_add_u32 s53, s0, s52 // (op 1259) 

v_and_b32 v13, 63, v1 // (op 1259) 
v_lshrrev_b32 v16, 4, v13 // (op 1259) 

s_mul_i32 s52, s16, s8 // (op 1259) 
s_add_u32 s52, s52, s53 // (op 1259) 

s_mul_hi_u32 s53, s52, s13 // (op 1259) 
s_sub_u32 s54, s52, s53 // (op 1259) 
s_lshr_b32 s55, s54, 1 // (op 1259) 

s_add_u32 s54, s55, s53 // (op 1259) 


s_lshr_b32 s53, s54, s11 // (op 1259) 

s_and_b32 s54, s11, 2147483648 // (op 1259) 
s_cmp_eq_i32 s54, 0 // (op 1259) 
s_mov_b32 s55, scc // (op 1259) 

s_cmp_eq_i32 s55, 0 // (op 1259) 
s_cselect_b32 s54, s52, s53 // (op 1259) 


s_mul_hi_u32 s53, s54, s12 // (op 1259) 
s_sub_u32 s55, s54, s53 // (op 1259) 
s_lshr_b32 s76, s55, 1 // (op 1259) 

s_add_u32 s55, s76, s53 // (op 1259) 


s_lshr_b32 s53, s55, s10 // (op 1259) 

s_and_b32 s55, s10, 2147483648 // (op 1259) 
s_cmp_eq_i32 s55, 0 // (op 1259) 
s_mov_b32 s76, scc // (op 1259) 

s_cmp_eq_i32 s76, 0 // (op 1259) 
s_cselect_b32 s55, s54, s53 // (op 1259) 


s_mul_i32 s53, s55, s14 // (op 1259) 

s_sub_u32 s55, s54, s53 // (op 1259) 

v_lshl_add_u32 v13, s55, 4, v12 // (op 1259) 


v_mul_lo_u32 v12, v13, s38 // (op 1259) 

s_ashr_i64 s[76:77], s[2:3], 7 // (op 1259) 
s_mul_i32 s53, s54, s76 // (op 1259) 


s_sub_u32 s54, s52, s53 // (op 1259) 


v_lshl_add_u32 v13, s54, 2, v16 // (op 1259) 


v_add_u32 v16, v13, v12 // (op 1259) 


s_mul_i32 s53, s38, s23 // (op 1261) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 1261) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 1261) least significant
s_mul_hi_u32 s55, s38, s22 // (op 1261) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1261) most significant: sum
s_add_u32 s53, s53, s55 // (op 1261) most significant: sum


s_mov_b32 s72, s40 // (op 1261) reg expression
s_mov_b32 s73, s41 // (op 1261) reg expression
s_mov_b32 s74, s52 // (op 1261) reg expression
s_mov_b32 s75, 131072 // (op 1261) call()

v_and_b32 v12, -4, v1 // (op 1410) 
v_sub_u32 v13, v1, v12 // (op 1410) 

v_and_b32 v12, 4294967292, v1 // (op 1410) 
v_add_u32 v17, v12, v13 // (op 1410) 


s_and_b32 s52, s7, 4294967294 // (op 510) 
s_cmp_lt_u32 s52, s7 // (op 509) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_509 // (op 509) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_509 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_509:


s_add_u32 s52, s52, 1 // (op 511) 
v_mov_b32 v8, v15 // (op 514) 

buffer_load_ubyte v12, v8, s[68:71], 0 offen offset: 0 // (op 514) Load value

v_add_u32 v15, 4, v15 // (op 1123) 


s_barrier  // (op 1706) 
v_mov_b32 v8, v10 // (op 522) 




buffer_load_dwordx4 v[18:21], v8, s[60:63], 0 offen // (op 522) Load value

v_add_u32 v10, 64, v10 // (op 859) 
s_waitcnt vmcnt(1)
ds_write_b8 v17, v12 // (op 515) Store local data 



v_mov_b32 v8, v11 // (op 524) 

buffer_load_dwordx4 v[22:25], v8, s[64:67], 0 offen // (op 524) Load value

v_add_u32 v11, 64, v11 // (op 995) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1709) 



ds_read_u8 v8, v0 // (op 517) Load local data 


v_mov_b32 v12, v16 // (op 519) 

buffer_load_ubyte v13, v12, s[72:75], 0 offen offset: 0 // (op 519) Load value

v_add_u32 v16, 4, v16 // (op 1251) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[22:25], a[0:3], v8, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 520) 




s_cmp_lt_u32 s52, s7 // (op 509) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_509 // (op 509) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_509 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_509:






s_barrier  // (op 1857) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_lg_u32 s18, 0 // (op 269) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Send_Tile_269 // (op 269) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Send_Tile_269
v_lshrrev_b32 v8, 4, v1 // (op 1575) 
v_lshlrev_b32 v10, 2, v8 // (op 1575) 

v_lshl_add_u32 v8, s16, 5, v10 // (op 1575) 

v_and_b32 v10, -16, v1 // (op 1575) 
v_sub_u32 v11, v1, v10 // (op 1575) 

v_lshl_add_u32 v10, v8, 4, v11 // (op 1575) 


v_lshlrev_b32 v8, 2, v10 // (op 1575) 

s_lshl_b32 s52, s15, 11 // (op 1577) 
s_mov_b32 s76, s42 // (op 1577) reg expression
s_mov_b32 s77, s43 // (op 1577) reg expression
s_mov_b32 s78, s52 // (op 1577) reg expression
s_mov_b32 s79, 131072 // (op 1577) call()




buffer_store_dword a0, v8, s[76:79], 0 offen sc1 // (op 266) Store value
buffer_store_dword a1, v8, s[76:79], 0 offen offset:64 sc1 // (op 266) Store value
buffer_store_dword a2, v8, s[76:79], 0 offen offset:128 sc1 // (op 266) Store value
buffer_store_dword a3, v8, s[76:79], 0 offen offset:192 sc1 // (op 266) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 270) 
v_cmp_eq_i32 vcc, v1, 0 // (op 274) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_274 // (op 274) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_274
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
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_274 // (op 274) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_274
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_274:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_274:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Send_Tile_269 // (op 269) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Send_Tile_269
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Send_Tile_269:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Send_Tile_269:
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
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Receive_Tile_285 // (op 285) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Receive_Tile_285
s_mov_b32 s52, 0 // (op 251) call()
s_mul_i32 s53, s52, s8 // (op 250) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 250) 
s_mov_b32 s80, s54 // (op 250) convert

s_sub_u32 s54, s80, s19 // (op 250) 

s_sub_u32 s55, s54, 1 // (op 250) 

s_cmp_lt_u32 s53, s55 // (op 250) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_250 // (op 250) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_250 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_ReceiveTileLoop_250:
s_add_u32 s53, s16, 1 // (op 1592) 
v_lshrrev_b32 v8, 4, v1 // (op 1592) 
s_add_u32 s54, s53, s52 // (op 1592) 
s_lshl_b32 s54, s54, 5 // (op 1592) 

v_lshlrev_b32 v10, 2, v8 // (op 1592) 

v_add_lshl_u32 v8, s54, v10, 4 // (op 1592) 


v_and_b32 v10, -16, v1 // (op 1592) 
v_sub_u32 v11, v1, v10 // (op 1592) 

v_add_lshl_u32 v10, v8, v11, 2 // (op 1592) 


s_lshl_b32 s53, s15, 11 // (op 1594) 
s_mov_b32 s76, s42 // (op 1594) reg expression
s_mov_b32 s77, s43 // (op 1594) reg expression
s_mov_b32 s78, s53 // (op 1594) reg expression
s_mov_b32 s79, 131072 // (op 1594) call()

RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_289:
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
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_289 // (op 289) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_289 if true)
s_barrier  // (op 293) 
v_cmp_eq_i32 vcc, v1, 0 // (op 292) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_292 // (op 292) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_292
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
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_292 // (op 292) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_292
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_292:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_292:




buffer_load_dword v12, v10, s[76:79], 0 offen // (op 304) Load value
buffer_load_dword v13, v10, s[76:79], 0 offen offset:64 // (op 304) Load value
buffer_load_dword v14, v10, s[76:79], 0 offen offset:128 // (op 304) Load value
buffer_load_dword v15, v10, s[76:79], 0 offen offset:192 // (op 304) Load value
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
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_ReceiveTileLoop_250 // (op 250) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_ReceiveTileLoop_250 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_250:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Receive_Tile_285 // (op 285) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Receive_Tile_285
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Receive_Tile_285:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Receive_Tile_285:

s_cmp_eq_i32 s18, 0 // (op 328) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_328 // (op 328) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_328
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 106) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_106 // (op 106) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_106
v_accvgpr_read v10, a0 // (op 38) 
v_accvgpr_read v11, a1 // (op 38) 
v_accvgpr_read v12, a2 // (op 38) 
v_accvgpr_read v13, a3 // (op 38) 
v_mul_f32 v14, v7, v10 // (op 38) 
v_mul_f32 v15, v7, v11 // (op 38) 
v_mul_f32 v16, v7, v12 // (op 38) 
v_mul_f32 v17, v7, v13 // (op 38) 

v_lshrrev_b32 v8, 6, v1 // (op 1452) 
s_mul_i32 s52, s16, s8 // (op 1452) 
s_add_u32 s52, s52, s0 // (op 1452) 
v_and_b32 v10, 63, v1 // (op 1452) 
s_mul_hi_u32 s53, s52, s13 // (op 1452) 
v_lshrrev_b32 v11, 4, v10 // (op 1452) 
s_sub_u32 s54, s52, s53 // (op 1452) 
v_lshlrev_b32 v12, 2, v11 // (op 1452) 

s_lshr_b32 s55, s54, 1 // (op 1452) 

v_lshl_add_u32 v11, v8, 4, v12 // (op 1452) 


s_add_u32 s54, s55, s53 // (op 1452) 


v_lshrrev_b32 v8, 2, v10 // (op 1452) 

s_lshr_b32 s53, s54, s11 // (op 1452) 

v_and_b32 v10, 3, v8 // (op 1452) 

s_and_b32 s54, s11, 2147483648 // (op 1452) 
v_and_b32 v8, 3, v1 // (op 1452) 
s_cmp_eq_i32 s54, 0 // (op 1452) 
s_mov_b32 s55, scc // (op 1452) 

v_lshl_add_u32 v12, v10, 2, v8 // (op 1452) 


s_cmp_eq_i32 s55, 0 // (op 1452) 
s_cselect_b32 s54, s52, s53 // (op 1452) 



s_mul_hi_u32 s55, s54, s12 // (op 1452) 
s_sub_u32 s52, s54, s55 // (op 1452) 
s_lshr_b32 s53, s52, 1 // (op 1452) 

s_add_u32 s52, s53, s55 // (op 1452) 


s_lshr_b32 s53, s52, s10 // (op 1452) 

s_and_b32 s52, s10, 2147483648 // (op 1452) 
s_cmp_eq_i32 s52, 0 // (op 1452) 
s_mov_b32 s55, scc // (op 1452) 

s_cmp_eq_i32 s55, 0 // (op 1452) 
s_cselect_b32 s52, s54, s53 // (op 1452) 


s_mul_i32 s53, s52, s14 // (op 1452) 
v_lshl_add_u32 v8, s52, 5, v11 // (op 1452) 


s_sub_u32 s52, s54, s53 // (op 1452) 


v_lshl_add_u32 v13, s52, 4, v12 // (op 1452) 


v_mul_lo_u32 v10, v13, s46 // (op 1452) 

v_add_lshl_u32 v11, v8, v10, 2 // (op 1452) 


s_mul_i32 s53, s46, s23 // (op 1454) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1454) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1454) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1454) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1454) most significant: sum
s_add_u32 s53, s53, s55 // (op 1454) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1454) 

s_mov_b32 s76, s48 // (op 1454) reg expression
s_mov_b32 s77, s49 // (op 1454) reg expression
s_mov_b32 s78, s54 // (op 1454) reg expression
s_mov_b32 s79, 131072 // (op 1454) call()




buffer_store_dwordx4 v[14:17], v11, s[76:79], 0 offen // (op 44) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_106 // (op 106) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_106
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_106:



buffer_load_dwordx4 v[10:13], v9, s[56:59], 0 offen // (op 118) Load value
v_accvgpr_read v14, a0 // (op 115) 
v_accvgpr_read v15, a1 // (op 115) 
v_accvgpr_read v16, a2 // (op 115) 
v_accvgpr_read v17, a3 // (op 115) 
v_mul_f32 v18, v7, v14 // (op 115) 
v_mul_f32 v19, v7, v15 // (op 115) 
v_mul_f32 v20, v7, v16 // (op 115) 
v_mul_f32 v21, v7, v17 // (op 115) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v6, v10 // (op 119) 
v_mul_f32 v15, v6, v11 // (op 119) 
v_mul_f32 v16, v6, v12 // (op 119) 
v_mul_f32 v17, v6, v13 // (op 119) 

v_add_f32 v10, v14, v18 // (op 116) 
v_add_f32 v11, v15, v19 // (op 116) 
v_add_f32 v12, v16, v20 // (op 116) 
v_add_f32 v13, v17, v21 // (op 116) 


v_lshrrev_b32 v8, 6, v1 // (op 1469) 
s_mul_i32 s52, s16, s8 // (op 1469) 
s_add_u32 s52, s52, s0 // (op 1469) 
v_and_b32 v14, 63, v1 // (op 1469) 
s_mul_hi_u32 s53, s52, s13 // (op 1469) 
v_lshrrev_b32 v15, 4, v14 // (op 1469) 
s_sub_u32 s54, s52, s53 // (op 1469) 
v_lshlrev_b32 v16, 2, v15 // (op 1469) 

s_lshr_b32 s55, s54, 1 // (op 1469) 

v_lshl_add_u32 v15, v8, 4, v16 // (op 1469) 


s_add_u32 s54, s55, s53 // (op 1469) 


v_lshrrev_b32 v8, 2, v14 // (op 1469) 

s_lshr_b32 s53, s54, s11 // (op 1469) 

v_and_b32 v14, 3, v8 // (op 1469) 

s_and_b32 s54, s11, 2147483648 // (op 1469) 
v_and_b32 v8, 3, v1 // (op 1469) 
s_cmp_eq_i32 s54, 0 // (op 1469) 
s_mov_b32 s55, scc // (op 1469) 

v_lshl_add_u32 v16, v14, 2, v8 // (op 1469) 


s_cmp_eq_i32 s55, 0 // (op 1469) 
s_cselect_b32 s54, s52, s53 // (op 1469) 



s_mul_hi_u32 s55, s54, s12 // (op 1469) 
s_sub_u32 s52, s54, s55 // (op 1469) 
s_lshr_b32 s53, s52, 1 // (op 1469) 

s_add_u32 s52, s53, s55 // (op 1469) 


s_lshr_b32 s53, s52, s10 // (op 1469) 

s_and_b32 s52, s10, 2147483648 // (op 1469) 
s_cmp_eq_i32 s52, 0 // (op 1469) 
s_mov_b32 s55, scc // (op 1469) 

s_cmp_eq_i32 s55, 0 // (op 1469) 
s_cselect_b32 s52, s54, s53 // (op 1469) 


s_mul_i32 s53, s52, s14 // (op 1469) 
v_lshl_add_u32 v8, s52, 5, v15 // (op 1469) 


s_sub_u32 s52, s54, s53 // (op 1469) 


v_lshl_add_u32 v14, s52, 4, v16 // (op 1469) 


v_mul_lo_u32 v15, v14, s46 // (op 1469) 

v_add_lshl_u32 v14, v8, v15, 2 // (op 1469) 


s_mul_i32 s53, s46, s23 // (op 1471) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1471) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1471) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1471) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1471) most significant: sum
s_add_u32 s53, s53, s55 // (op 1471) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1471) 

s_mov_b32 s76, s48 // (op 1471) reg expression
s_mov_b32 s77, s49 // (op 1471) reg expression
s_mov_b32 s78, s54 // (op 1471) reg expression
s_mov_b32 s79, 131072 // (op 1471) call()




buffer_store_dwordx4 v[10:13], v14, s[76:79], 0 offen // (op 117) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_106:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_328 // (op 328) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_328
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_328:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_328:

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
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_SKStreamTileLoop_241 // (op 241) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_SKStreamTileLoop_241 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_241:











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
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_396 // (op 396) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_396 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_DPStreamTileLoop_396:
v_lshrrev_b32 v0, 6, v1 // (op 1380) 
s_mul_i32 s15, s16, s50 // (op 1380) 
s_add_u32 s15, s15, s0 // (op 1380) 
v_and_b32 v5, 63, v1 // (op 1380) 
s_add_u32 s6, s15, s9 // (op 1380) 

v_lshrrev_b32 v8, 4, v5 // (op 1380) 
s_mul_hi_u32 s15, s6, s13 // (op 1380) 
v_lshlrev_b32 v9, 2, v8 // (op 1380) 

s_sub_u32 s7, s6, s15 // (op 1380) 
v_lshl_add_u32 v8, v0, 4, v9 // (op 1380) 


s_lshr_b32 s8, s7, 1 // (op 1380) 

v_lshrrev_b32 v0, 2, v5 // (op 1380) 

s_add_u32 s7, s8, s15 // (op 1380) 


v_and_b32 v5, 3, v0 // (op 1380) 

s_lshr_b32 s8, s7, s11 // (op 1380) 

v_and_b32 v0, 3, v1 // (op 1380) 
s_and_b32 s7, s11, 2147483648 // (op 1380) 
v_lshl_add_u32 v9, v5, 2, v0 // (op 1380) 


s_cmp_eq_i32 s7, 0 // (op 1380) 
s_mov_b32 s15, scc // (op 1380) 

s_cmp_eq_i32 s15, 0 // (op 1380) 
s_cselect_b32 s7, s6, s8 // (op 1380) 



s_mul_hi_u32 s6, s7, s12 // (op 1380) 
s_sub_u32 s8, s7, s6 // (op 1380) 
s_lshr_b32 s15, s8, 1 // (op 1380) 

s_add_u32 s8, s15, s6 // (op 1380) 


s_lshr_b32 s6, s8, s10 // (op 1380) 

s_and_b32 s8, s10, 2147483648 // (op 1380) 
s_cmp_eq_i32 s8, 0 // (op 1380) 
s_mov_b32 s15, scc // (op 1380) 

s_cmp_eq_i32 s15, 0 // (op 1380) 
s_cselect_b32 s8, s7, s6 // (op 1380) 


s_mul_i32 s6, s8, s14 // (op 1380) 
v_lshl_add_u32 v0, s8, 5, v8 // (op 1380) 


s_sub_u32 s8, s7, s6 // (op 1380) 


v_lshl_add_u32 v5, s8, 4, v9 // (op 1380) 


v_mul_lo_u32 v8, v5, s4 // (op 1380) 

v_add_lshl_u32 v5, v0, v8, 2 // (op 1380) 


s_mul_i32 s19, s4, s23 // (op 1382) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 1382) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 1382) least significant
s_mul_hi_u32 s6, s4, s22 // (op 1382) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1382) most significant: sum
s_add_u32 s19, s19, s6 // (op 1382) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1382) 

s_mov_b32 s56, s20 // (op 1382) reg expression
s_mov_b32 s57, s21 // (op 1382) reg expression
s_mov_b32 s58, s6 // (op 1382) reg expression
s_mov_b32 s59, 131072 // (op 1382) call()

v_accvgpr_write a0, 0.00000 // (op 153) call()
v_accvgpr_write a1, 0.00000 // (op 153) call()
v_accvgpr_write a2, 0.00000 // (op 153) call()
v_accvgpr_write a3, 0.00000 // (op 153) call()
v_lshrrev_b32 v0, 6, v1 // (op 940) 
s_mul_i32 s15, s16, s50 // (op 940) 
s_add_u32 s15, s15, s0 // (op 940) 
v_and_b32 v8, 15, v1 // (op 940) 
s_add_u32 s6, s15, s9 // (op 940) 

v_lshl_add_u32 v9, v0, 4, v8 // (op 940) 


s_mul_hi_u32 s15, s6, s13 // (op 940) 
v_and_b32 v0, 63, v1 // (op 940) 
s_sub_u32 s7, s6, s15 // (op 940) 
v_lshrrev_b32 v8, 4, v0 // (op 940) 

s_lshr_b32 s8, s7, 1 // (op 940) 

v_lshlrev_b32 v0, 5, v8 // (op 940) 

s_add_u32 s7, s8, s15 // (op 940) 


s_lshr_b32 s8, s7, s11 // (op 940) 

s_and_b32 s7, s11, 2147483648 // (op 940) 
s_cmp_eq_i32 s7, 0 // (op 940) 
s_mov_b32 s15, scc // (op 940) 

s_cmp_eq_i32 s15, 0 // (op 940) 
s_cselect_b32 s7, s6, s8 // (op 940) 


s_mul_hi_u32 s8, s7, s12 // (op 940) 
s_sub_u32 s15, s7, s8 // (op 940) 
s_lshr_b32 s18, s15, 1 // (op 940) 

s_add_u32 s15, s18, s8 // (op 940) 


s_lshr_b32 s8, s15, s10 // (op 940) 

s_and_b32 s15, s10, 2147483648 // (op 940) 
s_cmp_eq_i32 s15, 0 // (op 940) 
s_mov_b32 s18, scc // (op 940) 

s_cmp_eq_i32 s18, 0 // (op 940) 
s_cselect_b32 s15, s7, s8 // (op 940) 


v_lshl_add_u32 v8, s15, 5, v9 // (op 940) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 940) 
s_mul_i32 s8, s7, s18 // (op 940) 


s_sub_u32 s7, s6, s8 // (op 940) 


v_lshl_add_u32 v9, s7, 7, v0 // (op 940) 


v_mov_b32 v11, 0 // (op 940) convert
v_mov_b32 v10, v9 // (op 940) convert

v_mul_lo_u32 v13, v8, s25 // (op 940) most significant: low of low * high
v_mul_hi_u32 v0, v8, s24 // (op 940) most significant: high of low * low
v_mul_lo_u32 v12, v8, s24 // (op 940) least significant: low of low * low
v_add_u32 v13, v13, v0 // (op 940) most significant: sum

v_add_co_u32 v12, s[18:19], v12, v10 // (op 940) least significant half
v_addc_co_u32 v13, s[18:19], v13, v11, s[18:19] // (op 940) most significant half



v_lshlrev_b64 v[8:9], 2, v[12:13] // (op 940) 

v_lshrrev_b64 v[10:11], 3, v[8:9] // (op 940) 

v_mov_b32 v0, v10 // (op 940) convert

s_mul_i32 s19, s24, s29 // (op 942) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 942) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 942) least significant
s_mul_hi_u32 s6, s24, s28 // (op 942) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 942) most significant: sum
s_add_u32 s19, s19, s6 // (op 942) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 942) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 942) 

s_mov_b32 s60, s26 // (op 942) reg expression
s_mov_b32 s61, s27 // (op 942) reg expression
s_mov_b32 s62, s18 // (op 942) reg expression
s_mov_b32 s63, 131072 // (op 942) call()

v_lshrrev_b32 v8, 6, v1 // (op 959) 
s_mul_i32 s15, s16, s50 // (op 959) 
s_add_u32 s15, s15, s0 // (op 959) 
v_and_b32 v9, 15, v1 // (op 959) 
s_add_u32 s6, s15, s9 // (op 959) 

v_lshl_add_u32 v10, v8, 4, v9 // (op 959) 


s_mul_hi_u32 s15, s6, s13 // (op 959) 
v_and_b32 v8, 63, v1 // (op 959) 
s_sub_u32 s7, s6, s15 // (op 959) 
v_lshrrev_b32 v9, 4, v8 // (op 959) 

s_lshr_b32 s8, s7, 1 // (op 959) 

v_lshlrev_b32 v8, 5, v9 // (op 959) 

s_add_u32 s7, s8, s15 // (op 959) 


s_lshr_b32 s8, s7, s11 // (op 959) 

s_and_b32 s7, s11, 2147483648 // (op 959) 
s_cmp_eq_i32 s7, 0 // (op 959) 
s_mov_b32 s15, scc // (op 959) 

s_cmp_eq_i32 s15, 0 // (op 959) 
s_cselect_b32 s7, s6, s8 // (op 959) 


s_mul_hi_u32 s8, s7, s12 // (op 959) 
s_sub_u32 s15, s7, s8 // (op 959) 
s_lshr_b32 s18, s15, 1 // (op 959) 

s_add_u32 s15, s18, s8 // (op 959) 


s_lshr_b32 s8, s15, s10 // (op 959) 

s_and_b32 s15, s10, 2147483648 // (op 959) 
s_cmp_eq_i32 s15, 0 // (op 959) 
s_mov_b32 s18, scc // (op 959) 

s_cmp_eq_i32 s18, 0 // (op 959) 
s_cselect_b32 s15, s7, s8 // (op 959) 


v_lshl_add_u32 v9, s15, 5, v10 // (op 959) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 959) 
s_mul_i32 s8, s7, s18 // (op 959) 


s_sub_u32 s7, s6, s8 // (op 959) 


v_lshl_add_u32 v10, s7, 7, v8 // (op 959) 


v_mov_b32 v13, 0 // (op 959) convert
v_mov_b32 v12, v10 // (op 959) convert

v_mul_lo_u32 v11, v9, s25 // (op 959) most significant: low of low * high
v_mul_hi_u32 v8, v9, s24 // (op 959) most significant: high of low * low
v_mul_lo_u32 v10, v9, s24 // (op 959) least significant: low of low * low
v_add_u32 v11, v11, v8 // (op 959) most significant: sum

v_add_co_u32 v10, s[18:19], v10, v12 // (op 959) least significant half
v_addc_co_u32 v11, s[18:19], v11, v13, s[18:19] // (op 959) most significant half



v_lshlrev_b64 v[8:9], 2, v[10:11] // (op 959) 

v_lshrrev_b64 v[10:11], 3, v[8:9] // (op 959) 

v_mov_b32 v8, v10 // (op 959) convert

s_mul_i32 s19, s24, s29 // (op 961) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 961) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 961) least significant
s_mul_hi_u32 s6, s24, s28 // (op 961) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 961) most significant: sum
s_add_u32 s19, s19, s6 // (op 961) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 961) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 961) 

s_mov_b32 s60, s26 // (op 961) reg expression
s_mov_b32 s61, s27 // (op 961) reg expression
s_mov_b32 s62, s18 // (op 961) reg expression
s_mov_b32 s63, 131072 // (op 961) call()

v_and_b32 v9, 15, v1 // (op 1068) 
s_mul_i32 s15, s16, s50 // (op 1068) 
s_add_u32 s15, s15, s0 // (op 1068) 
v_and_b32 v10, 63, v1 // (op 1068) 
s_add_u32 s6, s15, s9 // (op 1068) 

v_lshrrev_b32 v11, 4, v10 // (op 1068) 

s_mul_hi_u32 s15, s6, s13 // (op 1068) 
v_lshlrev_b32 v10, 5, v11 // (op 1068) 

s_sub_u32 s7, s6, s15 // (op 1068) 
s_lshr_b32 s8, s7, 1 // (op 1068) 

s_add_u32 s7, s8, s15 // (op 1068) 


s_lshr_b32 s8, s7, s11 // (op 1068) 

s_and_b32 s7, s11, 2147483648 // (op 1068) 
s_cmp_eq_i32 s7, 0 // (op 1068) 
s_mov_b32 s15, scc // (op 1068) 

s_cmp_eq_i32 s15, 0 // (op 1068) 
s_cselect_b32 s7, s6, s8 // (op 1068) 


s_mul_hi_u32 s8, s7, s12 // (op 1068) 
s_sub_u32 s15, s7, s8 // (op 1068) 
s_lshr_b32 s18, s15, 1 // (op 1068) 

s_add_u32 s15, s18, s8 // (op 1068) 


s_lshr_b32 s8, s15, s10 // (op 1068) 

s_and_b32 s15, s10, 2147483648 // (op 1068) 
s_cmp_eq_i32 s15, 0 // (op 1068) 
s_mov_b32 s18, scc // (op 1068) 

s_cmp_eq_i32 s18, 0 // (op 1068) 
s_cselect_b32 s15, s7, s8 // (op 1068) 


s_mul_i32 s8, s15, s14 // (op 1068) 

s_sub_u32 s15, s7, s8 // (op 1068) 

v_lshl_add_u32 v11, s15, 4, v9 // (op 1068) 


v_mul_lo_u32 v13, v11, s31 // (op 1068) most significant: low of low * high
v_mul_hi_u32 v9, v11, s30 // (op 1068) most significant: high of low * low
v_mul_lo_u32 v12, v11, s30 // (op 1068) least significant: low of low * low
v_add_u32 v13, v13, v9 // (op 1068) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1068) 
s_mul_i32 s8, s7, s18 // (op 1068) 


s_sub_u32 s7, s6, s8 // (op 1068) 


v_lshl_add_u32 v9, s7, 7, v10 // (op 1068) 


v_mov_b32 v11, 0 // (op 1068) convert
v_mov_b32 v10, v9 // (op 1068) convert

v_add_co_u32 v14, s[18:19], v10, v12 // (op 1068) least significant half
v_addc_co_u32 v15, s[18:19], v11, v13, s[18:19] // (op 1068) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 1068) 


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 1068) 

v_mov_b32 v9, v12 // (op 1068) convert

s_mul_i32 s19, s30, s23 // (op 1070) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 1070) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 1070) least significant
s_mul_hi_u32 s6, s30, s22 // (op 1070) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1070) most significant: sum
s_add_u32 s19, s19, s6 // (op 1070) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1070) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 1070) 

s_mov_b32 s64, s32 // (op 1070) reg expression
s_mov_b32 s65, s33 // (op 1070) reg expression
s_mov_b32 s66, s18 // (op 1070) reg expression
s_mov_b32 s67, 131072 // (op 1070) call()

v_and_b32 v10, 15, v1 // (op 1087) 
s_mul_i32 s15, s16, s50 // (op 1087) 
s_add_u32 s15, s15, s0 // (op 1087) 
v_and_b32 v11, 63, v1 // (op 1087) 
s_add_u32 s6, s15, s9 // (op 1087) 

v_lshrrev_b32 v12, 4, v11 // (op 1087) 

s_mul_hi_u32 s15, s6, s13 // (op 1087) 
v_lshlrev_b32 v11, 5, v12 // (op 1087) 

s_sub_u32 s7, s6, s15 // (op 1087) 
s_lshr_b32 s8, s7, 1 // (op 1087) 

s_add_u32 s7, s8, s15 // (op 1087) 


s_lshr_b32 s8, s7, s11 // (op 1087) 

s_and_b32 s7, s11, 2147483648 // (op 1087) 
s_cmp_eq_i32 s7, 0 // (op 1087) 
s_mov_b32 s15, scc // (op 1087) 

s_cmp_eq_i32 s15, 0 // (op 1087) 
s_cselect_b32 s7, s6, s8 // (op 1087) 


s_mul_hi_u32 s8, s7, s12 // (op 1087) 
s_sub_u32 s15, s7, s8 // (op 1087) 
s_lshr_b32 s18, s15, 1 // (op 1087) 

s_add_u32 s15, s18, s8 // (op 1087) 


s_lshr_b32 s8, s15, s10 // (op 1087) 

s_and_b32 s15, s10, 2147483648 // (op 1087) 
s_cmp_eq_i32 s15, 0 // (op 1087) 
s_mov_b32 s18, scc // (op 1087) 

s_cmp_eq_i32 s18, 0 // (op 1087) 
s_cselect_b32 s15, s7, s8 // (op 1087) 


s_mul_i32 s8, s15, s14 // (op 1087) 

s_sub_u32 s15, s7, s8 // (op 1087) 

v_lshl_add_u32 v12, s15, 4, v10 // (op 1087) 


v_mul_lo_u32 v15, v12, s31 // (op 1087) most significant: low of low * high
v_mul_hi_u32 v10, v12, s30 // (op 1087) most significant: high of low * low
v_mul_lo_u32 v14, v12, s30 // (op 1087) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 1087) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1087) 
s_mul_i32 s8, s7, s18 // (op 1087) 


s_sub_u32 s7, s6, s8 // (op 1087) 


v_lshl_add_u32 v10, s7, 7, v11 // (op 1087) 


v_mov_b32 v13, 0 // (op 1087) convert
v_mov_b32 v12, v10 // (op 1087) convert

v_add_co_u32 v10, s[18:19], v12, v14 // (op 1087) least significant half
v_addc_co_u32 v11, s[18:19], v13, v15, s[18:19] // (op 1087) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 1087) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 1087) 

v_mov_b32 v10, v12 // (op 1087) convert

s_mul_i32 s19, s30, s23 // (op 1089) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 1089) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 1089) least significant
s_mul_hi_u32 s6, s30, s22 // (op 1089) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1089) most significant: sum
s_add_u32 s19, s19, s6 // (op 1089) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1089) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 1089) 

s_mov_b32 s64, s32 // (op 1089) reg expression
s_mov_b32 s65, s33 // (op 1089) reg expression
s_mov_b32 s66, s18 // (op 1089) reg expression
s_mov_b32 s67, 131072 // (op 1089) call()

v_lshrrev_b32 v11, 2, v1 // (op 1196) 
s_mul_i32 s15, s16, s50 // (op 1196) 
s_add_u32 s15, s15, s0 // (op 1196) 
v_and_b32 v12, -4, v1 // (op 1196) 
s_add_u32 s6, s15, s9 // (op 1196) 

v_sub_u32 v13, v1, v12 // (op 1196) 

s_mul_hi_u32 s15, s6, s13 // (op 1196) 
s_sub_u32 s7, s6, s15 // (op 1196) 
s_lshr_b32 s8, s7, 1 // (op 1196) 

s_add_u32 s7, s8, s15 // (op 1196) 


s_lshr_b32 s8, s7, s11 // (op 1196) 

s_and_b32 s7, s11, 2147483648 // (op 1196) 
s_cmp_eq_i32 s7, 0 // (op 1196) 
s_mov_b32 s15, scc // (op 1196) 

s_cmp_eq_i32 s15, 0 // (op 1196) 
s_cselect_b32 s7, s6, s8 // (op 1196) 


s_mul_hi_u32 s8, s7, s12 // (op 1196) 
s_sub_u32 s15, s7, s8 // (op 1196) 
s_lshr_b32 s18, s15, 1 // (op 1196) 

s_add_u32 s15, s18, s8 // (op 1196) 


s_lshr_b32 s8, s15, s10 // (op 1196) 

s_and_b32 s15, s10, 2147483648 // (op 1196) 
s_cmp_eq_i32 s15, 0 // (op 1196) 
s_mov_b32 s18, scc // (op 1196) 

s_cmp_eq_i32 s18, 0 // (op 1196) 
s_cselect_b32 s15, s7, s8 // (op 1196) 


v_lshl_add_u32 v12, s15, 5, v11 // (op 1196) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1196) 
s_mul_i32 s8, s7, s18 // (op 1196) 


s_sub_u32 s7, s6, s8 // (op 1196) 


v_lshl_add_u32 v11, s7, 2, v13 // (op 1196) 


v_mul_lo_u32 v13, v12, s34 // (op 1196) 
v_add_u32 v13, v13, v11 // (op 1196) 


s_mul_i32 s19, s34, s29 // (op 1198) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1198) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1198) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1198) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1198) most significant: sum
s_add_u32 s19, s19, s6 // (op 1198) most significant: sum


s_mov_b32 s68, s36 // (op 1198) reg expression
s_mov_b32 s69, s37 // (op 1198) reg expression
s_mov_b32 s70, s18 // (op 1198) reg expression
s_mov_b32 s71, 131072 // (op 1198) call()

v_lshrrev_b32 v11, 2, v1 // (op 1215) 
s_mul_i32 s15, s16, s50 // (op 1215) 
s_add_u32 s15, s15, s0 // (op 1215) 
v_and_b32 v12, -4, v1 // (op 1215) 
s_add_u32 s6, s15, s9 // (op 1215) 

v_sub_u32 v14, v1, v12 // (op 1215) 

s_mul_hi_u32 s15, s6, s13 // (op 1215) 
s_sub_u32 s7, s6, s15 // (op 1215) 
s_lshr_b32 s8, s7, 1 // (op 1215) 

s_add_u32 s7, s8, s15 // (op 1215) 


s_lshr_b32 s8, s7, s11 // (op 1215) 

s_and_b32 s7, s11, 2147483648 // (op 1215) 
s_cmp_eq_i32 s7, 0 // (op 1215) 
s_mov_b32 s15, scc // (op 1215) 

s_cmp_eq_i32 s15, 0 // (op 1215) 
s_cselect_b32 s7, s6, s8 // (op 1215) 


s_mul_hi_u32 s8, s7, s12 // (op 1215) 
s_sub_u32 s15, s7, s8 // (op 1215) 
s_lshr_b32 s18, s15, 1 // (op 1215) 

s_add_u32 s15, s18, s8 // (op 1215) 


s_lshr_b32 s8, s15, s10 // (op 1215) 

s_and_b32 s15, s10, 2147483648 // (op 1215) 
s_cmp_eq_i32 s15, 0 // (op 1215) 
s_mov_b32 s18, scc // (op 1215) 

s_cmp_eq_i32 s18, 0 // (op 1215) 
s_cselect_b32 s15, s7, s8 // (op 1215) 


v_lshl_add_u32 v12, s15, 5, v11 // (op 1215) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1215) 
s_mul_i32 s8, s7, s18 // (op 1215) 


s_sub_u32 s7, s6, s8 // (op 1215) 


v_lshl_add_u32 v11, s7, 2, v14 // (op 1215) 


v_mul_lo_u32 v14, v12, s34 // (op 1215) 
v_add_u32 v14, v14, v11 // (op 1215) 


s_mul_i32 s19, s34, s29 // (op 1217) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1217) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1217) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1217) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1217) most significant: sum
s_add_u32 s19, s19, s6 // (op 1217) most significant: sum


s_mov_b32 s68, s36 // (op 1217) reg expression
s_mov_b32 s69, s37 // (op 1217) reg expression
s_mov_b32 s70, s18 // (op 1217) reg expression
s_mov_b32 s71, 131072 // (op 1217) call()

v_and_b32 v11, 15, v1 // (op 1324) 
s_mul_i32 s15, s16, s50 // (op 1324) 
s_add_u32 s15, s15, s0 // (op 1324) 
v_and_b32 v12, 63, v1 // (op 1324) 
s_add_u32 s6, s15, s9 // (op 1324) 

v_lshrrev_b32 v15, 4, v12 // (op 1324) 

s_mul_hi_u32 s15, s6, s13 // (op 1324) 
s_sub_u32 s7, s6, s15 // (op 1324) 
s_lshr_b32 s8, s7, 1 // (op 1324) 

s_add_u32 s7, s8, s15 // (op 1324) 


s_lshr_b32 s8, s7, s11 // (op 1324) 

s_and_b32 s7, s11, 2147483648 // (op 1324) 
s_cmp_eq_i32 s7, 0 // (op 1324) 
s_mov_b32 s15, scc // (op 1324) 

s_cmp_eq_i32 s15, 0 // (op 1324) 
s_cselect_b32 s7, s6, s8 // (op 1324) 


s_mul_hi_u32 s8, s7, s12 // (op 1324) 
s_sub_u32 s15, s7, s8 // (op 1324) 
s_lshr_b32 s18, s15, 1 // (op 1324) 

s_add_u32 s15, s18, s8 // (op 1324) 


s_lshr_b32 s8, s15, s10 // (op 1324) 

s_and_b32 s15, s10, 2147483648 // (op 1324) 
s_cmp_eq_i32 s15, 0 // (op 1324) 
s_mov_b32 s18, scc // (op 1324) 

s_cmp_eq_i32 s18, 0 // (op 1324) 
s_cselect_b32 s15, s7, s8 // (op 1324) 


s_mul_i32 s8, s15, s14 // (op 1324) 

s_sub_u32 s15, s7, s8 // (op 1324) 

v_lshl_add_u32 v12, s15, 4, v11 // (op 1324) 


v_mul_lo_u32 v11, v12, s38 // (op 1324) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1324) 
s_mul_i32 s8, s7, s18 // (op 1324) 


s_sub_u32 s7, s6, s8 // (op 1324) 


v_lshl_add_u32 v12, s7, 2, v15 // (op 1324) 


v_add_u32 v15, v12, v11 // (op 1324) 


s_mul_i32 s19, s38, s23 // (op 1326) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 1326) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 1326) least significant
s_mul_hi_u32 s6, s38, s22 // (op 1326) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1326) most significant: sum
s_add_u32 s19, s19, s6 // (op 1326) most significant: sum


s_mov_b32 s72, s40 // (op 1326) reg expression
s_mov_b32 s73, s41 // (op 1326) reg expression
s_mov_b32 s74, s18 // (op 1326) reg expression
s_mov_b32 s75, 131072 // (op 1326) call()

v_and_b32 v11, 15, v1 // (op 1343) 
s_mul_i32 s15, s16, s50 // (op 1343) 
s_add_u32 s15, s15, s0 // (op 1343) 
v_and_b32 v12, 63, v1 // (op 1343) 
s_add_u32 s6, s15, s9 // (op 1343) 

v_lshrrev_b32 v16, 4, v12 // (op 1343) 

s_mul_hi_u32 s15, s6, s13 // (op 1343) 
s_sub_u32 s7, s6, s15 // (op 1343) 
s_lshr_b32 s8, s7, 1 // (op 1343) 

s_add_u32 s7, s8, s15 // (op 1343) 


s_lshr_b32 s8, s7, s11 // (op 1343) 

s_and_b32 s7, s11, 2147483648 // (op 1343) 
s_cmp_eq_i32 s7, 0 // (op 1343) 
s_mov_b32 s15, scc // (op 1343) 

s_cmp_eq_i32 s15, 0 // (op 1343) 
s_cselect_b32 s7, s6, s8 // (op 1343) 


s_mul_hi_u32 s8, s7, s12 // (op 1343) 
s_sub_u32 s15, s7, s8 // (op 1343) 
s_lshr_b32 s18, s15, 1 // (op 1343) 

s_add_u32 s15, s18, s8 // (op 1343) 


s_lshr_b32 s8, s15, s10 // (op 1343) 

s_and_b32 s15, s10, 2147483648 // (op 1343) 
s_cmp_eq_i32 s15, 0 // (op 1343) 
s_mov_b32 s18, scc // (op 1343) 

s_cmp_eq_i32 s18, 0 // (op 1343) 
s_cselect_b32 s15, s7, s8 // (op 1343) 


s_mul_i32 s8, s15, s14 // (op 1343) 

s_sub_u32 s15, s7, s8 // (op 1343) 

v_lshl_add_u32 v12, s15, 4, v11 // (op 1343) 


v_mul_lo_u32 v11, v12, s38 // (op 1343) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1343) 
s_mul_i32 s8, s7, s18 // (op 1343) 


s_sub_u32 s7, s6, s8 // (op 1343) 


v_lshl_add_u32 v12, s7, 2, v16 // (op 1343) 


v_add_u32 v16, v12, v11 // (op 1343) 


s_mul_i32 s19, s38, s23 // (op 1345) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 1345) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 1345) least significant
s_mul_hi_u32 s6, s38, s22 // (op 1345) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1345) most significant: sum
s_add_u32 s19, s19, s6 // (op 1345) most significant: sum


s_mov_b32 s72, s40 // (op 1345) reg expression
s_mov_b32 s73, s41 // (op 1345) reg expression
s_mov_b32 s74, s18 // (op 1345) reg expression
s_mov_b32 s75, 131072 // (op 1345) call()




buffer_load_ubyte v11, v15, s[72:75], 0 offen offset: 0 // (op 763) Load value



buffer_load_dwordx4 v[18:21], v0, s[60:63], 0 offen // (op 768) Load value



buffer_load_dwordx4 v[22:25], v9, s[64:67], 0 offen // (op 773) Load value



buffer_load_ubyte v12, v13, s[68:71], 0 offen offset: 0 // (op 777) Load value



buffer_load_ubyte v17, v15, s[72:75], 0 offen offset:4 // (op 781) Load value




buffer_load_dwordx4 v[26:29], v0, s[60:63], 0 offen offset:64 // (op 786) Load value




buffer_load_dwordx4 v[30:33], v9, s[64:67], 0 offen offset:64 // (op 791) Load value




buffer_load_ubyte v0, v13, s[68:71], 0 offen offset:4 // (op 795) Load value

v_and_b32 v9, -4, v1 // (op 1549) 
v_sub_u32 v13, v1, v9 // (op 1549) 

v_and_b32 v9, 4294967292, v1 // (op 1549) 
v_add_u32 v15, v9, v13 // (op 1549) 


s_barrier  // (op 1716) 



s_waitcnt vmcnt(4)
ds_write_b8 v15, v12 // (op 797) Store local data 

v_and_b32 v9, -4, v1 // (op 1520) 
v_sub_u32 v13, v1, v9 // (op 1520) 

v_and_b32 v9, 4294967292, v1 // (op 1520) 
v_add_u32 v15, v9, v13 // (op 1520) 


v_and_b32 v9, -4, v1 // (op 1607) 
v_sub_u32 v13, v1, v9 // (op 1607) 

v_and_b32 v9, 4294967292, v1 // (op 1607) 
v_add_u32 v34, v9, v13 // (op 1607) 


s_mov_b32 s1, 0 // (op 155) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 154) 
s_mov_b32 s15, s18 // (op 154) convert

s_and_b32 s6, s15, 4294967294 // (op 154) 

s_cmp_lt_u32 s1, s6 // (op 154) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_154 // (op 154) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_154 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoop_154:
s_add_u32 s1, s1, 2 // (op 156) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1713) 



ds_read_u8 v9, v3 // (op 160) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[22:25], a[0:3], v9, v11 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 163) 



v_mov_b32 v9, v16 // (op 162) 

buffer_load_ubyte v11, v9, s[72:75], 0 offen offset:8 // (op 162) Load value


v_mov_b32 v13, v8 // (op 165) 

buffer_load_dwordx4 v[18:21], v13, s[60:63], 0 offen offset:128 // (op 165) Load value


v_mov_b32 v35, v10 // (op 167) 

buffer_load_dwordx4 v[22:25], v35, s[64:67], 0 offen offset:128 // (op 167) Load value


v_mov_b32 v36, v14 // (op 157) 

buffer_load_ubyte v12, v36, s[68:71], 0 offen offset:8 // (op 157) Load value
s_barrier  // (op 1732) 



s_waitcnt vmcnt(4)
ds_write_b8 v34, v0 offset:128 // (op 452) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1729) 



ds_read_u8 v37, v4 offset:128 // (op 454) Load local data 
s_barrier  // (op 1827) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[30:33], a[0:3], v37, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 457) 




buffer_load_ubyte v17, v9, s[72:75], 0 offen offset:12 // (op 456) Load value




v_add_u32 v16, 8, v16 // (op 1335) 
buffer_load_dwordx4 v[26:29], v13, s[60:63], 0 offen offset:192 // (op 459) Load value




v_add_u32 v8, 128, v8 // (op 951) 
buffer_load_dwordx4 v[30:33], v35, s[64:67], 0 offen offset:192 // (op 461) Load value




v_add_u32 v10, 128, v10 // (op 1079) 
buffer_load_ubyte v0, v36, s[68:71], 0 offen offset:12 // (op 451) Load value

s_barrier  // (op 1719) 
v_add_u32 v14, 8, v14 // (op 1207) 



s_waitcnt vmcnt(4)
ds_write_b8 v15, v12 // (op 158) Store local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 154) 
s_mov_b32 s15, s18 // (op 154) convert

s_and_b32 s6, s15, 4294967294 // (op 154) 

s_cmp_lt_u32 s1, s6 // (op 154) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoop_154 // (op 154) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoop_154 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_154:










v_lshrrev_b32 v8, 6, v1 // (op 845) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 845) 
v_and_b32 v9, 15, v1 // (op 845) 
s_and_b32 s15, s18, 4294967294 // (op 845) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 845) 


s_add_u32 s6, s0, s15 // (op 845) 

v_and_b32 v8, 63, v1 // (op 845) 
v_lshrrev_b32 v9, 4, v8 // (op 845) 

s_mul_i32 s15, s16, s50 // (op 845) 
s_add_u32 s15, s15, s6 // (op 845) 

v_lshlrev_b32 v8, 5, v9 // (op 845) 

s_add_u32 s6, s15, s9 // (op 845) 

s_mul_hi_u32 s15, s6, s13 // (op 845) 
s_sub_u32 s7, s6, s15 // (op 845) 
s_lshr_b32 s8, s7, 1 // (op 845) 

s_add_u32 s7, s8, s15 // (op 845) 


s_lshr_b32 s8, s7, s11 // (op 845) 

s_and_b32 s7, s11, 2147483648 // (op 845) 
s_cmp_eq_i32 s7, 0 // (op 845) 
s_mov_b32 s15, scc // (op 845) 

s_cmp_eq_i32 s15, 0 // (op 845) 
s_cselect_b32 s7, s6, s8 // (op 845) 


s_mul_hi_u32 s8, s7, s12 // (op 845) 
s_sub_u32 s15, s7, s8 // (op 845) 
s_lshr_b32 s42, s15, 1 // (op 845) 

s_add_u32 s15, s42, s8 // (op 845) 


s_lshr_b32 s8, s15, s10 // (op 845) 

s_and_b32 s15, s10, 2147483648 // (op 845) 
s_cmp_eq_i32 s15, 0 // (op 845) 
s_mov_b32 s42, scc // (op 845) 

s_cmp_eq_i32 s42, 0 // (op 845) 
s_cselect_b32 s15, s7, s8 // (op 845) 


s_mul_i32 s8, s7, s18 // (op 845) 


v_lshl_add_u32 v9, s15, 5, v10 // (op 845) 


s_sub_u32 s7, s6, s8 // (op 845) 


v_lshl_add_u32 v10, s7, 7, v8 // (op 845) 


v_mov_b32 v19, 0 // (op 845) convert
v_mov_b32 v18, v10 // (op 845) convert

v_mul_lo_u32 v11, v9, s25 // (op 845) most significant: low of low * high
v_mul_hi_u32 v8, v9, s24 // (op 845) most significant: high of low * low
v_mul_lo_u32 v10, v9, s24 // (op 845) least significant: low of low * low
v_add_u32 v11, v11, v8 // (op 845) most significant: sum

v_add_co_u32 v10, s[18:19], v10, v18 // (op 845) least significant half
v_addc_co_u32 v11, s[18:19], v11, v19, s[18:19] // (op 845) most significant half



v_lshlrev_b64 v[8:9], 2, v[10:11] // (op 845) 

v_lshrrev_b64 v[10:11], 3, v[8:9] // (op 845) 

v_mov_b32 v8, v10 // (op 845) convert

s_mul_i32 s19, s24, s29 // (op 847) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 847) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 847) least significant
s_mul_hi_u32 s6, s24, s28 // (op 847) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 847) most significant: sum
s_add_u32 s19, s19, s6 // (op 847) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 847) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 847) 

s_mov_b32 s60, s26 // (op 847) reg expression
s_mov_b32 s61, s27 // (op 847) reg expression
s_mov_b32 s62, s18 // (op 847) reg expression
s_mov_b32 s63, 131072 // (op 847) call()

v_and_b32 v9, 15, v1 // (op 981) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 981) 
v_and_b32 v10, 63, v1 // (op 981) 
s_and_b32 s15, s18, 4294967294 // (op 981) 
v_lshrrev_b32 v11, 4, v10 // (op 981) 

s_add_u32 s6, s0, s15 // (op 981) 

v_lshlrev_b32 v10, 5, v11 // (op 981) 

s_mul_i32 s15, s16, s50 // (op 981) 
s_add_u32 s15, s15, s6 // (op 981) 

s_add_u32 s6, s15, s9 // (op 981) 

s_mul_hi_u32 s15, s6, s13 // (op 981) 
s_sub_u32 s7, s6, s15 // (op 981) 
s_lshr_b32 s8, s7, 1 // (op 981) 

s_add_u32 s7, s8, s15 // (op 981) 


s_lshr_b32 s8, s7, s11 // (op 981) 

s_and_b32 s7, s11, 2147483648 // (op 981) 
s_cmp_eq_i32 s7, 0 // (op 981) 
s_mov_b32 s15, scc // (op 981) 

s_cmp_eq_i32 s15, 0 // (op 981) 
s_cselect_b32 s7, s6, s8 // (op 981) 


s_mul_hi_u32 s8, s7, s12 // (op 981) 
s_sub_u32 s15, s7, s8 // (op 981) 
s_lshr_b32 s42, s15, 1 // (op 981) 

s_add_u32 s15, s42, s8 // (op 981) 


s_lshr_b32 s8, s15, s10 // (op 981) 

s_and_b32 s15, s10, 2147483648 // (op 981) 
s_cmp_eq_i32 s15, 0 // (op 981) 
s_mov_b32 s42, scc // (op 981) 

s_cmp_eq_i32 s42, 0 // (op 981) 
s_cselect_b32 s15, s7, s8 // (op 981) 


s_mul_i32 s8, s15, s14 // (op 981) 

s_sub_u32 s15, s7, s8 // (op 981) 

s_mul_i32 s8, s7, s18 // (op 981) 


v_lshl_add_u32 v11, s15, 4, v9 // (op 981) 


s_sub_u32 s7, s6, s8 // (op 981) 


v_mul_lo_u32 v13, v11, s31 // (op 981) most significant: low of low * high
v_mul_hi_u32 v9, v11, s30 // (op 981) most significant: high of low * low
v_mul_lo_u32 v12, v11, s30 // (op 981) least significant: low of low * low
v_add_u32 v13, v13, v9 // (op 981) most significant: sum


v_lshl_add_u32 v9, s7, 7, v10 // (op 981) 


v_mov_b32 v11, 0 // (op 981) convert
v_mov_b32 v10, v9 // (op 981) convert

v_add_co_u32 v14, s[18:19], v10, v12 // (op 981) least significant half
v_addc_co_u32 v15, s[18:19], v11, v13, s[18:19] // (op 981) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 981) 


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 981) 

v_mov_b32 v9, v12 // (op 981) convert

s_mul_i32 s19, s30, s23 // (op 983) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 983) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 983) least significant
s_mul_hi_u32 s6, s30, s22 // (op 983) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 983) most significant: sum
s_add_u32 s19, s19, s6 // (op 983) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 983) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 983) 

s_mov_b32 s64, s32 // (op 983) reg expression
s_mov_b32 s65, s33 // (op 983) reg expression
s_mov_b32 s66, s18 // (op 983) reg expression
s_mov_b32 s67, 131072 // (op 983) call()

v_lshrrev_b32 v10, 2, v1 // (op 1109) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1109) 
v_and_b32 v11, -4, v1 // (op 1109) 
s_and_b32 s15, s18, 4294967294 // (op 1109) 
v_sub_u32 v12, v1, v11 // (op 1109) 

s_add_u32 s6, s0, s15 // (op 1109) 

s_mul_i32 s15, s16, s50 // (op 1109) 
s_add_u32 s15, s15, s6 // (op 1109) 

s_add_u32 s6, s15, s9 // (op 1109) 

s_mul_hi_u32 s15, s6, s13 // (op 1109) 
s_sub_u32 s7, s6, s15 // (op 1109) 
s_lshr_b32 s8, s7, 1 // (op 1109) 

s_add_u32 s7, s8, s15 // (op 1109) 


s_lshr_b32 s8, s7, s11 // (op 1109) 

s_and_b32 s7, s11, 2147483648 // (op 1109) 
s_cmp_eq_i32 s7, 0 // (op 1109) 
s_mov_b32 s15, scc // (op 1109) 

s_cmp_eq_i32 s15, 0 // (op 1109) 
s_cselect_b32 s7, s6, s8 // (op 1109) 


s_mul_hi_u32 s8, s7, s12 // (op 1109) 
s_sub_u32 s15, s7, s8 // (op 1109) 
s_lshr_b32 s42, s15, 1 // (op 1109) 

s_add_u32 s15, s42, s8 // (op 1109) 


s_lshr_b32 s8, s15, s10 // (op 1109) 

s_and_b32 s15, s10, 2147483648 // (op 1109) 
s_cmp_eq_i32 s15, 0 // (op 1109) 
s_mov_b32 s42, scc // (op 1109) 

s_cmp_eq_i32 s42, 0 // (op 1109) 
s_cselect_b32 s15, s7, s8 // (op 1109) 


s_mul_i32 s8, s7, s18 // (op 1109) 


v_lshl_add_u32 v11, s15, 5, v10 // (op 1109) 


s_sub_u32 s7, s6, s8 // (op 1109) 


v_lshl_add_u32 v10, s7, 2, v12 // (op 1109) 


v_mul_lo_u32 v12, v11, s34 // (op 1109) 
v_add_u32 v12, v12, v10 // (op 1109) 


s_mul_i32 s19, s34, s29 // (op 1111) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1111) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1111) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1111) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1111) most significant: sum
s_add_u32 s19, s19, s6 // (op 1111) most significant: sum


s_mov_b32 s68, s36 // (op 1111) reg expression
s_mov_b32 s69, s37 // (op 1111) reg expression
s_mov_b32 s70, s18 // (op 1111) reg expression
s_mov_b32 s71, 131072 // (op 1111) call()

v_and_b32 v10, 15, v1 // (op 1237) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1237) 
v_and_b32 v11, 63, v1 // (op 1237) 
s_and_b32 s15, s18, 4294967294 // (op 1237) 
v_lshrrev_b32 v13, 4, v11 // (op 1237) 

s_add_u32 s6, s0, s15 // (op 1237) 

s_mul_i32 s15, s16, s50 // (op 1237) 
s_add_u32 s15, s15, s6 // (op 1237) 

s_add_u32 s6, s15, s9 // (op 1237) 

s_mul_hi_u32 s15, s6, s13 // (op 1237) 
s_sub_u32 s7, s6, s15 // (op 1237) 
s_lshr_b32 s8, s7, 1 // (op 1237) 

s_add_u32 s7, s8, s15 // (op 1237) 


s_lshr_b32 s8, s7, s11 // (op 1237) 

s_and_b32 s7, s11, 2147483648 // (op 1237) 
s_cmp_eq_i32 s7, 0 // (op 1237) 
s_mov_b32 s15, scc // (op 1237) 

s_cmp_eq_i32 s15, 0 // (op 1237) 
s_cselect_b32 s7, s6, s8 // (op 1237) 


s_mul_hi_u32 s8, s7, s12 // (op 1237) 
s_sub_u32 s15, s7, s8 // (op 1237) 
s_lshr_b32 s42, s15, 1 // (op 1237) 

s_add_u32 s15, s42, s8 // (op 1237) 


s_lshr_b32 s8, s15, s10 // (op 1237) 

s_and_b32 s15, s10, 2147483648 // (op 1237) 
s_cmp_eq_i32 s15, 0 // (op 1237) 
s_mov_b32 s42, scc // (op 1237) 

s_cmp_eq_i32 s42, 0 // (op 1237) 
s_cselect_b32 s15, s7, s8 // (op 1237) 


s_mul_i32 s8, s15, s14 // (op 1237) 

s_sub_u32 s15, s7, s8 // (op 1237) 

v_lshl_add_u32 v11, s15, 4, v10 // (op 1237) 


s_mul_i32 s8, s7, s18 // (op 1237) 


v_mul_lo_u32 v10, v11, s38 // (op 1237) 

s_sub_u32 s7, s6, s8 // (op 1237) 


v_lshl_add_u32 v11, s7, 2, v13 // (op 1237) 


v_add_u32 v13, v11, v10 // (op 1237) 


s_mul_i32 s19, s38, s23 // (op 1239) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 1239) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 1239) least significant
s_mul_hi_u32 s6, s38, s22 // (op 1239) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1239) most significant: sum
s_add_u32 s19, s19, s6 // (op 1239) most significant: sum


s_mov_b32 s72, s40 // (op 1239) reg expression
s_mov_b32 s73, s41 // (op 1239) reg expression
s_mov_b32 s74, s18 // (op 1239) reg expression
s_mov_b32 s75, 131072 // (op 1239) call()

v_and_b32 v10, -4, v1 // (op 1533) 
v_sub_u32 v11, v1, v10 // (op 1533) 

v_and_b32 v10, 4294967292, v1 // (op 1533) 
v_add_u32 v14, v10, v11 // (op 1533) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 405) 
s_and_b32 s15, s18, 4294967294 // (op 405) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 404) 
s_mov_b32 s6, s18 // (op 404) convert

s_cmp_lt_u32 s15, s6 // (op 404) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_404 // (op 404) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_404 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_404:


s_add_u32 s15, s15, 1 // (op 406) 
v_mov_b32 v0, v12 // (op 409) 

buffer_load_ubyte v10, v0, s[68:71], 0 offen offset: 0 // (op 409) Load value

v_add_u32 v12, 4, v12 // (op 1101) 


s_barrier  // (op 1722) 
v_mov_b32 v0, v8 // (op 417) 




buffer_load_dwordx4 v[16:19], v0, s[60:63], 0 offen // (op 417) Load value

v_add_u32 v8, 64, v8 // (op 835) 
s_waitcnt vmcnt(1)
ds_write_b8 v14, v10 // (op 410) Store local data 



v_mov_b32 v0, v9 // (op 419) 

buffer_load_dwordx4 v[20:23], v0, s[64:67], 0 offen // (op 419) Load value

v_add_u32 v9, 64, v9 // (op 973) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1725) 



ds_read_u8 v0, v3 // (op 412) Load local data 


v_mov_b32 v15, v13 // (op 414) 

buffer_load_ubyte v10, v15, s[72:75], 0 offen offset: 0 // (op 414) Load value

v_add_u32 v13, 4, v13 // (op 1229) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:19], v[20:23], a[0:3], v0, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 415) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 404) 
s_mov_b32 s6, s18 // (op 404) convert

s_cmp_lt_u32 s15, s6 // (op 404) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_404 // (op 404) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_404 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_404:






s_barrier  // (op 1821) 
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 189) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_189 // (op 189) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_189
s_nop 0
v_accvgpr_read v8, a0 // (op 197) Wait state hazard: XDL Write Hazard
v_accvgpr_read v9, a1 // (op 197) 
v_accvgpr_read v10, a2 // (op 197) 
v_accvgpr_read v11, a3 // (op 197) 
v_mul_f32 v12, v7, v8 // (op 197) 
v_mul_f32 v13, v7, v9 // (op 197) 
v_mul_f32 v14, v7, v10 // (op 197) 
v_mul_f32 v15, v7, v11 // (op 197) 

v_lshrrev_b32 v0, 6, v1 // (op 1486) 
s_mul_i32 s15, s16, s50 // (op 1486) 
s_add_u32 s15, s15, s0 // (op 1486) 
v_and_b32 v8, 63, v1 // (op 1486) 
s_add_u32 s6, s15, s9 // (op 1486) 

v_lshrrev_b32 v9, 4, v8 // (op 1486) 
s_mul_hi_u32 s15, s6, s13 // (op 1486) 
v_lshlrev_b32 v10, 2, v9 // (op 1486) 

s_sub_u32 s7, s6, s15 // (op 1486) 
v_lshl_add_u32 v9, v0, 4, v10 // (op 1486) 


s_lshr_b32 s8, s7, 1 // (op 1486) 

v_lshrrev_b32 v0, 2, v8 // (op 1486) 

s_add_u32 s7, s8, s15 // (op 1486) 


v_and_b32 v8, 3, v0 // (op 1486) 

s_lshr_b32 s8, s7, s11 // (op 1486) 

v_and_b32 v0, 3, v1 // (op 1486) 
s_and_b32 s7, s11, 2147483648 // (op 1486) 
v_lshl_add_u32 v10, v8, 2, v0 // (op 1486) 


s_cmp_eq_i32 s7, 0 // (op 1486) 
s_mov_b32 s15, scc // (op 1486) 

s_cmp_eq_i32 s15, 0 // (op 1486) 
s_cselect_b32 s7, s6, s8 // (op 1486) 



s_mul_hi_u32 s6, s7, s12 // (op 1486) 
s_sub_u32 s8, s7, s6 // (op 1486) 
s_lshr_b32 s15, s8, 1 // (op 1486) 

s_add_u32 s8, s15, s6 // (op 1486) 


s_lshr_b32 s6, s8, s10 // (op 1486) 

s_and_b32 s8, s10, 2147483648 // (op 1486) 
s_cmp_eq_i32 s8, 0 // (op 1486) 
s_mov_b32 s15, scc // (op 1486) 

s_cmp_eq_i32 s15, 0 // (op 1486) 
s_cselect_b32 s8, s7, s6 // (op 1486) 


s_mul_i32 s6, s8, s14 // (op 1486) 
v_lshl_add_u32 v0, s8, 5, v9 // (op 1486) 


s_sub_u32 s8, s7, s6 // (op 1486) 


v_lshl_add_u32 v11, s8, 4, v10 // (op 1486) 


v_mul_lo_u32 v8, v11, s46 // (op 1486) 

v_add_lshl_u32 v9, v0, v8, 2 // (op 1486) 


s_mul_i32 s19, s46, s23 // (op 1488) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1488) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1488) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1488) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1488) most significant: sum
s_add_u32 s19, s19, s6 // (op 1488) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1488) 

s_mov_b32 s76, s48 // (op 1488) reg expression
s_mov_b32 s77, s49 // (op 1488) reg expression
s_mov_b32 s78, s6 // (op 1488) reg expression
s_mov_b32 s79, 131072 // (op 1488) call()




buffer_store_dwordx4 v[12:15], v9, s[76:79], 0 offen // (op 199) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_189 // (op 189) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_189
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_189:



buffer_load_dwordx4 v[8:11], v5, s[56:59], 0 offen // (op 212) Load value
v_accvgpr_read v12, a0 // (op 209) 
v_accvgpr_read v13, a1 // (op 209) 
v_accvgpr_read v14, a2 // (op 209) 
v_accvgpr_read v15, a3 // (op 209) 
v_mul_f32 v16, v7, v12 // (op 209) 
v_mul_f32 v17, v7, v13 // (op 209) 
v_mul_f32 v18, v7, v14 // (op 209) 
v_mul_f32 v19, v7, v15 // (op 209) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v6, v8 // (op 213) 
v_mul_f32 v13, v6, v9 // (op 213) 
v_mul_f32 v14, v6, v10 // (op 213) 
v_mul_f32 v15, v6, v11 // (op 213) 

v_add_f32 v8, v12, v16 // (op 210) 
v_add_f32 v9, v13, v17 // (op 210) 
v_add_f32 v10, v14, v18 // (op 210) 
v_add_f32 v11, v15, v19 // (op 210) 


v_lshrrev_b32 v0, 6, v1 // (op 1503) 
s_mul_i32 s15, s16, s50 // (op 1503) 
s_add_u32 s15, s15, s0 // (op 1503) 
v_and_b32 v12, 63, v1 // (op 1503) 
s_add_u32 s6, s15, s9 // (op 1503) 

v_lshrrev_b32 v13, 4, v12 // (op 1503) 
s_mul_hi_u32 s15, s6, s13 // (op 1503) 
v_lshlrev_b32 v14, 2, v13 // (op 1503) 

s_sub_u32 s7, s6, s15 // (op 1503) 
v_lshl_add_u32 v13, v0, 4, v14 // (op 1503) 


s_lshr_b32 s8, s7, 1 // (op 1503) 

v_lshrrev_b32 v0, 2, v12 // (op 1503) 

s_add_u32 s7, s8, s15 // (op 1503) 


v_and_b32 v12, 3, v0 // (op 1503) 

s_lshr_b32 s8, s7, s11 // (op 1503) 

v_and_b32 v0, 3, v1 // (op 1503) 
s_and_b32 s7, s11, 2147483648 // (op 1503) 
v_lshl_add_u32 v14, v12, 2, v0 // (op 1503) 


s_cmp_eq_i32 s7, 0 // (op 1503) 
s_mov_b32 s15, scc // (op 1503) 

s_cmp_eq_i32 s15, 0 // (op 1503) 
s_cselect_b32 s7, s6, s8 // (op 1503) 



s_mul_hi_u32 s6, s7, s12 // (op 1503) 
s_sub_u32 s8, s7, s6 // (op 1503) 
s_lshr_b32 s15, s8, 1 // (op 1503) 

s_add_u32 s8, s15, s6 // (op 1503) 


s_lshr_b32 s6, s8, s10 // (op 1503) 

s_and_b32 s8, s10, 2147483648 // (op 1503) 
s_cmp_eq_i32 s8, 0 // (op 1503) 
s_mov_b32 s15, scc // (op 1503) 

s_cmp_eq_i32 s15, 0 // (op 1503) 
s_cselect_b32 s8, s7, s6 // (op 1503) 


s_mul_i32 s6, s8, s14 // (op 1503) 
v_lshl_add_u32 v0, s8, 5, v13 // (op 1503) 


s_sub_u32 s8, s7, s6 // (op 1503) 


v_lshl_add_u32 v12, s8, 4, v14 // (op 1503) 


v_mul_lo_u32 v13, v12, s46 // (op 1503) 

v_add_lshl_u32 v12, v0, v13, 2 // (op 1503) 


s_mul_i32 s19, s46, s23 // (op 1505) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1505) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1505) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1505) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1505) most significant: sum
s_add_u32 s19, s19, s6 // (op 1505) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1505) 

s_mov_b32 s76, s48 // (op 1505) reg expression
s_mov_b32 s77, s49 // (op 1505) reg expression
s_mov_b32 s78, s6 // (op 1505) reg expression
s_mov_b32 s79, 131072 // (op 1505) call()




buffer_store_dwordx4 v[8:11], v12, s[76:79], 0 offen // (op 211) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_189:

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
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_DPStreamTileLoop_396 // (op 396) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopTop_DPStreamTileLoop_396 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_396:




































s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2
.LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_end:
.size RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2, .LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2_end-RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2
  .amdhsa_next_free_vgpr 44
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 256 // lds bytes
  .amdhsa_accum_offset 40
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
  - .name: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2
    .symbol: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x16x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 256
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 82
    .vgpr_count: 40
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

