// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2
.p2align 8
.type RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2,@function
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 1728) 
v_and_b32 v3, 63, v1 // (op 1728) 
v_sub_u32 v4, v3, v0 // (op 1728) 

v_lshrrev_b32 v3, 1, v4 // (op 1728) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 1728) 


v_and_b32 v0, 1, v4 // (op 1728) 

v_lshlrev_b32 v3, 3, v0 // (op 1728) 

v_lshrrev_b32 v0, 6, v1 // (op 1728) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1728) 


v_add_u32 v0, v5, v4 // (op 1728) 


v_and_b32 v3, 48, v1 // (op 1793) 
v_and_b32 v4, 63, v1 // (op 1793) 
v_sub_u32 v5, v4, v3 // (op 1793) 

v_lshrrev_b32 v4, 1, v5 // (op 1793) 
v_add_lshl_u32 v6, v3, v4, 4 // (op 1793) 


v_and_b32 v3, 1, v5 // (op 1793) 

v_lshlrev_b32 v4, 3, v3 // (op 1793) 

v_add_u32 v3, v6, v4 // (op 1793) 


v_lshrrev_b32 v4, 6, v1 // (op 1859) 
v_and_b32 v5, 15, v1 // (op 1859) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 1859) 


v_and_b32 v4, 48, v1 // (op 1859) 
v_add_u32 v5, v4, v6 // (op 1859) 


v_and_b32 v4, 48, v1 // (op 1925) 
v_and_b32 v6, 15, v1 // (op 1925) 
v_add_u32 v7, v4, v6 // (op 1925) 


v_and_b32 v4, 48, v1 // (op 2060) 
v_and_b32 v6, 63, v1 // (op 2060) 
v_sub_u32 v8, v6, v4 // (op 2060) 

v_lshrrev_b32 v6, 1, v8 // (op 2060) 
v_add_lshl_u32 v9, v4, v6, 4 // (op 2060) 


v_and_b32 v4, 1, v8 // (op 2060) 

v_lshlrev_b32 v6, 3, v4 // (op 2060) 

v_lshrrev_b32 v4, 6, v1 // (op 2060) 
v_lshl_add_u32 v8, v4, 4, v6 // (op 2060) 


v_add_u32 v4, v9, v8 // (op 2060) 


v_and_b32 v6, 48, v1 // (op 2125) 
v_and_b32 v8, 63, v1 // (op 2125) 
v_sub_u32 v9, v8, v6 // (op 2125) 

v_lshrrev_b32 v8, 1, v9 // (op 2125) 
v_add_lshl_u32 v10, v6, v8, 4 // (op 2125) 


v_and_b32 v6, 1, v9 // (op 2125) 

v_lshlrev_b32 v8, 3, v6 // (op 2125) 

v_add_u32 v6, v10, v8 // (op 2125) 


v_lshrrev_b32 v8, 6, v1 // (op 2191) 
v_and_b32 v9, 15, v1 // (op 2191) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 2191) 


v_and_b32 v8, 48, v1 // (op 2191) 
v_add_u32 v9, v8, v10 // (op 2191) 


v_and_b32 v8, 48, v1 // (op 2257) 
v_and_b32 v10, 15, v1 // (op 2257) 
v_add_u32 v11, v8, v10 // (op 2257) 


v_and_b32 v8, 48, v1 // (op 2343) 
v_and_b32 v10, 63, v1 // (op 2343) 
v_sub_u32 v12, v10, v8 // (op 2343) 

v_lshrrev_b32 v10, 1, v12 // (op 2343) 
v_add_lshl_u32 v13, v8, v10, 4 // (op 2343) 


v_and_b32 v8, 1, v12 // (op 2343) 

v_lshlrev_b32 v10, 3, v8 // (op 2343) 

v_lshrrev_b32 v8, 6, v1 // (op 2343) 
v_lshl_add_u32 v12, v8, 4, v10 // (op 2343) 


v_add_u32 v8, v13, v12 // (op 2343) 


v_and_b32 v10, 48, v1 // (op 2382) 
v_and_b32 v12, 63, v1 // (op 2382) 
v_sub_u32 v13, v12, v10 // (op 2382) 

v_lshrrev_b32 v12, 1, v13 // (op 2382) 
v_add_lshl_u32 v14, v10, v12, 4 // (op 2382) 


v_and_b32 v10, 1, v13 // (op 2382) 

v_lshlrev_b32 v12, 3, v10 // (op 2382) 

v_add_u32 v10, v14, v12 // (op 2382) 


v_lshrrev_b32 v12, 6, v1 // (op 2406) 
v_and_b32 v13, 15, v1 // (op 2406) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 2406) 


v_and_b32 v12, 48, v1 // (op 2406) 
v_add_u32 v13, v12, v14 // (op 2406) 


v_and_b32 v12, 48, v1 // (op 2430) 
v_and_b32 v14, 15, v1 // (op 2430) 
v_add_u32 v15, v12, v14 // (op 2430) 


v_and_b32 v12, 48, v1 // (op 2469) 
v_and_b32 v14, 63, v1 // (op 2469) 
v_sub_u32 v16, v14, v12 // (op 2469) 

v_lshrrev_b32 v14, 1, v16 // (op 2469) 
v_add_lshl_u32 v17, v12, v14, 4 // (op 2469) 


v_and_b32 v12, 1, v16 // (op 2469) 

v_lshlrev_b32 v14, 3, v12 // (op 2469) 

v_lshrrev_b32 v12, 6, v1 // (op 2469) 
v_lshl_add_u32 v16, v12, 4, v14 // (op 2469) 


v_add_u32 v12, v17, v16 // (op 2469) 


v_and_b32 v14, 48, v1 // (op 2508) 
v_and_b32 v16, 63, v1 // (op 2508) 
v_sub_u32 v17, v16, v14 // (op 2508) 

v_lshrrev_b32 v16, 1, v17 // (op 2508) 
v_add_lshl_u32 v18, v14, v16, 4 // (op 2508) 


v_and_b32 v14, 1, v17 // (op 2508) 

v_lshlrev_b32 v16, 3, v14 // (op 2508) 

v_add_u32 v14, v18, v16 // (op 2508) 


v_lshrrev_b32 v16, 6, v1 // (op 2532) 
v_and_b32 v17, 15, v1 // (op 2532) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 2532) 


v_and_b32 v16, 48, v1 // (op 2532) 
v_add_u32 v17, v16, v18 // (op 2532) 


v_and_b32 v16, 48, v1 // (op 2556) 
v_and_b32 v18, 15, v1 // (op 2556) 
v_add_u32 v19, v16, v18 // (op 2556) 


v_mov_b32 v16, s6 // (op 2) Move value
v_mov_b32 v18, s7 // (op 43) Move value


s_mov_b32 s0, 0 // (op 387) call()
s_mov_b32 s1, 0 // (op 388) call()
s_mov_b32 s6, 0 // (op 389) call()
s_mov_b32 s7, 0 // (op 390) call()
s_mov_b32 s18, 0 // (op 391) call()
s_mov_b32 s19, 0 // (op 392) call()
s_mov_b32 s0, 0 // (op 287) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 286) 
s_cmp_gt_u32 s9, s52 // (op 286) 
s_mov_b32 s53, scc // (op 286) 
s_sub_u32 s54, s9, s52 // (op 286) 

s_cmp_eq_i32 s53, 0 // (op 286) 
s_cselect_b32 s52, 0, s54 // (op 286) 


s_cmp_lt_u32 s8, s52 // (op 286) 
s_mov_b32 s53, scc // (op 286) 
s_cmp_eq_i32 s53, 0 // (op 286) 
s_cselect_b32 s54, s52, s8 // (op 286) 


s_cmp_lt_u32 s0, s54 // (op 286) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_286 // (op 286) Condition: Top (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_286 if false)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_SKStreamTileLoop_286:
v_lshrrev_b32 v20, 6, v1 // (op 1654) 
s_mul_i32 s52, s16, s8 // (op 1654) 
s_add_u32 s52, s52, s0 // (op 1654) 
v_and_b32 v21, 63, v1 // (op 1654) 
s_mul_hi_u32 s53, s52, s13 // (op 1654) 
v_lshrrev_b32 v22, 4, v21 // (op 1654) 
s_sub_u32 s54, s52, s53 // (op 1654) 
v_lshlrev_b32 v23, 2, v22 // (op 1654) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 1654) 


s_lshr_b32 s55, s54, 1 // (op 1654) 

s_add_u32 s54, s55, s53 // (op 1654) 


v_lshrrev_b32 v20, 2, v21 // (op 1654) 

s_lshr_b32 s53, s54, s11 // (op 1654) 

v_and_b32 v21, 3, v20 // (op 1654) 

s_and_b32 s54, s11, 2147483648 // (op 1654) 
v_and_b32 v20, 3, v1 // (op 1654) 
s_cmp_eq_i32 s54, 0 // (op 1654) 
s_mov_b32 s55, scc // (op 1654) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 1654) 


s_cmp_eq_i32 s55, 0 // (op 1654) 
s_cselect_b32 s54, s52, s53 // (op 1654) 



s_mul_hi_u32 s52, s54, s12 // (op 1654) 
s_sub_u32 s53, s54, s52 // (op 1654) 
s_lshr_b32 s55, s53, 1 // (op 1654) 

s_add_u32 s53, s55, s52 // (op 1654) 


s_lshr_b32 s52, s53, s10 // (op 1654) 

s_and_b32 s53, s10, 2147483648 // (op 1654) 
s_cmp_eq_i32 s53, 0 // (op 1654) 
s_mov_b32 s55, scc // (op 1654) 

s_cmp_eq_i32 s55, 0 // (op 1654) 
s_cselect_b32 s53, s54, s52 // (op 1654) 


s_mul_i32 s52, s53, s14 // (op 1654) 
v_lshl_add_u32 v20, s53, 4, v22 // (op 1654) 


s_sub_u32 s53, s54, s52 // (op 1654) 


v_lshl_add_u32 v21, s53, 4, v23 // (op 1654) 


v_mul_lo_u32 v22, v21, s4 // (op 1654) 

v_add_lshl_u32 v21, v20, v22, 2 // (op 1654) 


s_mul_i32 s53, s4, s23 // (op 1656) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 1656) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 1656) least significant
s_mul_hi_u32 s55, s4, s22 // (op 1656) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1656) most significant: sum
s_add_u32 s53, s53, s55 // (op 1656) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1656) 

s_mov_b32 s56, s20 // (op 1656) reg expression
s_mov_b32 s57, s21 // (op 1656) reg expression
s_mov_b32 s58, s54 // (op 1656) reg expression
s_mov_b32 s59, 131072 // (op 1656) call()

v_and_b32 v20, 48, v1 // (op 1740) 
v_and_b32 v22, 63, v1 // (op 1740) 
v_sub_u32 v23, v22, v20 // (op 1740) 

v_lshrrev_b32 v22, 1, v23 // (op 1740) 
v_add_lshl_u32 v24, v20, v22, 4 // (op 1740) 


v_and_b32 v20, 1, v23 // (op 1740) 

v_lshlrev_b32 v22, 3, v20 // (op 1740) 

v_lshrrev_b32 v20, 6, v1 // (op 1740) 
v_lshl_add_u32 v23, v20, 4, v22 // (op 1740) 


v_add_u32 v20, v24, v23 // (op 1740) 


v_and_b32 v22, 48, v1 // (op 1805) 
v_and_b32 v23, 63, v1 // (op 1805) 
v_sub_u32 v24, v23, v22 // (op 1805) 

v_lshrrev_b32 v23, 1, v24 // (op 1805) 
v_add_lshl_u32 v25, v22, v23, 4 // (op 1805) 


v_and_b32 v22, 1, v24 // (op 1805) 

v_lshlrev_b32 v23, 3, v22 // (op 1805) 

v_add_u32 v22, v25, v23 // (op 1805) 


v_lshrrev_b32 v23, 6, v1 // (op 1871) 
v_and_b32 v24, 15, v1 // (op 1871) 
v_lshl_add_u32 v25, v23, 4, v24 // (op 1871) 


v_and_b32 v23, 48, v1 // (op 1871) 
v_add_u32 v24, v23, v25 // (op 1871) 


v_and_b32 v23, 48, v1 // (op 1937) 
v_and_b32 v25, 15, v1 // (op 1937) 
v_add_u32 v26, v23, v25 // (op 1937) 


s_mul_i32 s6, s8, s16 // (op 291) 
s_add_u32 s6, s6, s0 // (op 291) 
s_mul_hi_u32 s52, s6, s13 // (op 292) 
s_sub_u32 s53, s6, s52 // (op 292) 
s_lshr_b32 s54, s53, 1 // (op 292) 

s_add_u32 s53, s54, s52 // (op 292) 


s_lshr_b32 s52, s53, s11 // (op 292) 

s_and_b32 s53, s11, 2147483648 // (op 292) 
s_cmp_eq_i32 s53, 0 // (op 292) 
s_mov_b32 s54, scc // (op 292) 

s_cmp_eq_i32 s54, 0 // (op 292) 
s_cselect_b32 s53, s6, s52 // (op 292) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 292) 
s_mul_i32 s52, s53, s54 // (op 292) 


s_sub_u32 s18, s6, s52 // (op 292) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 293) 
s_cmp_gt_u32 s52, s18 // (op 293) 
s_mov_b32 s54, scc // (op 293) 
s_sub_u32 s55, s52, s18 // (op 293) 

s_cmp_eq_i32 s54, 0 // (op 293) 
s_cselect_b32 s52, 0, s55 // (op 293) 


s_cmp_gt_u32 s8, s0 // (op 293) 
s_mov_b32 s53, scc // (op 293) 
s_sub_u32 s54, s8, s0 // (op 293) 
s_cmp_eq_i32 s53, 0 // (op 293) 
s_cselect_b32 s55, 0, s54 // (op 293) 


s_cmp_lt_u32 s52, s55 // (op 293) 
s_mov_b32 s53, scc // (op 293) 
s_cmp_eq_i32 s53, 0 // (op 293) 
s_cselect_b32 s54, s55, s52 // (op 293) 



s_cmp_gt_u32 s9, s6 // (op 293) 
s_mov_b32 s55, scc // (op 293) 
s_sub_u32 s52, s9, s6 // (op 293) 
s_cmp_eq_i32 s55, 0 // (op 293) 
s_cselect_b32 s53, 0, s52 // (op 293) 


s_cmp_lt_u32 s54, s53 // (op 293) 
s_mov_b32 s52, scc // (op 293) 
s_cmp_eq_i32 s52, 0 // (op 293) 
s_cselect_b32 s7, s53, s54 // (op 293) 



s_add_u32 s52, s18, s7 // (op 294) 
s_sub_u32 s19, s52, 1 // (op 294) 

v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
s_mul_i32 s52, s16, s8 // (op 1186) 
s_add_u32 s52, s52, s0 // (op 1186) 
s_mul_hi_u32 s53, s52, s13 // (op 1186) 
s_sub_u32 s54, s52, s53 // (op 1186) 
s_lshr_b32 s55, s54, 1 // (op 1186) 

s_add_u32 s54, s55, s53 // (op 1186) 


s_lshr_b32 s53, s54, s11 // (op 1186) 

s_and_b32 s54, s11, 2147483648 // (op 1186) 
s_cmp_eq_i32 s54, 0 // (op 1186) 
s_mov_b32 s55, scc // (op 1186) 

s_cmp_eq_i32 s55, 0 // (op 1186) 
s_cselect_b32 s54, s52, s53 // (op 1186) 


s_mul_hi_u32 s53, s54, s12 // (op 1186) 
s_sub_u32 s55, s54, s53 // (op 1186) 
s_lshr_b32 s60, s55, 1 // (op 1186) 

s_add_u32 s55, s60, s53 // (op 1186) 


s_lshr_b32 s53, s55, s10 // (op 1186) 

s_and_b32 s55, s10, 2147483648 // (op 1186) 
s_cmp_eq_i32 s55, 0 // (op 1186) 
s_mov_b32 s60, scc // (op 1186) 

s_cmp_eq_i32 s60, 0 // (op 1186) 
s_cselect_b32 s55, s54, s53 // (op 1186) 


s_lshl_b32 s53, s55, 4 // (op 1186) 

s_mov_b32 s61, 0 // (op 1186) convert
s_mov_b32 s60, s53 // (op 1186) convert

s_ashr_i64 s[62:63], s[2:3], 7 // (op 1186) 
s_mul_i32 s53, s54, s62 // (op 1186) 


s_sub_u32 s54, s52, s53 // (op 1186) 


v_lshl_add_u32 v23, s54, 7, v1 // (op 1186) 

v_mul_lo_u32 v29, v23, s25 // (op 1186) most significant: low of low * high
v_mul_hi_u32 v25, v23, s24 // (op 1186) most significant: high of low * low
v_mul_lo_u32 v28, v23, s24 // (op 1186) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 1186) most significant: sum


v_mov_b32 v23, s60 // (op 1186) 
v_mov_b32 v25, s61 // (op 1186) 
v_add_co_u32 v30, s[52:53], v23, v28 // (op 1186) least significant half
v_addc_co_u32 v31, s[52:53], v25, v29, s[52:53] // (op 1186) most significant half





s_mul_i32 s53, s24, s3 // (op 1188) most significant: low of low * high
s_mul_i32 s54, s25, s2 // (op 1188) most significant: low of high * low
s_mul_i32 s52, s24, s2 // (op 1188) least significant
s_mul_hi_u32 s55, s24, s2 // (op 1188) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1188) most significant: sum
s_add_u32 s53, s53, s55 // (op 1188) most significant: sum


s_mov_b32 s60, s26 // (op 1188) reg expression
s_mov_b32 s61, s27 // (op 1188) reg expression
s_mov_b32 s62, s52 // (op 1188) reg expression
s_mov_b32 s63, 131072 // (op 1188) call()

s_mul_i32 s52, s16, s8 // (op 1205) 
s_add_u32 s52, s52, s0 // (op 1205) 
s_mul_hi_u32 s53, s52, s13 // (op 1205) 
s_sub_u32 s54, s52, s53 // (op 1205) 
s_lshr_b32 s55, s54, 1 // (op 1205) 

s_add_u32 s54, s55, s53 // (op 1205) 


s_lshr_b32 s53, s54, s11 // (op 1205) 

s_and_b32 s54, s11, 2147483648 // (op 1205) 
s_cmp_eq_i32 s54, 0 // (op 1205) 
s_mov_b32 s55, scc // (op 1205) 

s_cmp_eq_i32 s55, 0 // (op 1205) 
s_cselect_b32 s54, s52, s53 // (op 1205) 


s_mul_hi_u32 s53, s54, s12 // (op 1205) 
s_sub_u32 s55, s54, s53 // (op 1205) 
s_lshr_b32 s64, s55, 1 // (op 1205) 

s_add_u32 s55, s64, s53 // (op 1205) 


s_lshr_b32 s53, s55, s10 // (op 1205) 

s_and_b32 s55, s10, 2147483648 // (op 1205) 
s_cmp_eq_i32 s55, 0 // (op 1205) 
s_mov_b32 s64, scc // (op 1205) 

s_cmp_eq_i32 s64, 0 // (op 1205) 
s_cselect_b32 s55, s54, s53 // (op 1205) 


s_lshl_b32 s53, s55, 4 // (op 1205) 

s_mov_b32 s65, 0 // (op 1205) convert
s_mov_b32 s64, s53 // (op 1205) convert

s_ashr_i64 s[66:67], s[2:3], 7 // (op 1205) 
s_mul_i32 s53, s54, s66 // (op 1205) 


s_sub_u32 s54, s52, s53 // (op 1205) 


v_lshl_add_u32 v23, s54, 7, v1 // (op 1205) 

v_mul_lo_u32 v29, v23, s25 // (op 1205) most significant: low of low * high
v_mul_hi_u32 v25, v23, s24 // (op 1205) most significant: high of low * low
v_mul_lo_u32 v28, v23, s24 // (op 1205) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 1205) most significant: sum


v_mov_b32 v23, s64 // (op 1205) 
v_mov_b32 v25, s65 // (op 1205) 
v_add_co_u32 v32, s[52:53], v23, v28 // (op 1205) least significant half
v_addc_co_u32 v33, s[52:53], v25, v29, s[52:53] // (op 1205) most significant half





s_mul_i32 s53, s24, s3 // (op 1207) most significant: low of low * high
s_mul_i32 s54, s25, s2 // (op 1207) most significant: low of high * low
s_mul_i32 s52, s24, s2 // (op 1207) least significant
s_mul_hi_u32 s55, s24, s2 // (op 1207) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1207) most significant: sum
s_add_u32 s53, s53, s55 // (op 1207) most significant: sum


s_mov_b32 s60, s26 // (op 1207) reg expression
s_mov_b32 s61, s27 // (op 1207) reg expression
s_mov_b32 s62, s52 // (op 1207) reg expression
s_mov_b32 s63, 131072 // (op 1207) call()

s_mul_i32 s52, s16, s8 // (op 1317) 
s_add_u32 s52, s52, s0 // (op 1317) 
s_mul_hi_u32 s53, s52, s13 // (op 1317) 
s_sub_u32 s54, s52, s53 // (op 1317) 
s_lshr_b32 s55, s54, 1 // (op 1317) 

s_add_u32 s54, s55, s53 // (op 1317) 


s_lshr_b32 s53, s54, s11 // (op 1317) 

s_and_b32 s54, s11, 2147483648 // (op 1317) 
s_cmp_eq_i32 s54, 0 // (op 1317) 
s_mov_b32 s55, scc // (op 1317) 

s_cmp_eq_i32 s55, 0 // (op 1317) 
s_cselect_b32 s54, s52, s53 // (op 1317) 


s_mul_hi_u32 s53, s54, s12 // (op 1317) 
s_sub_u32 s55, s54, s53 // (op 1317) 
s_lshr_b32 s64, s55, 1 // (op 1317) 

s_add_u32 s55, s64, s53 // (op 1317) 


s_lshr_b32 s53, s55, s10 // (op 1317) 

s_and_b32 s55, s10, 2147483648 // (op 1317) 
s_cmp_eq_i32 s55, 0 // (op 1317) 
s_mov_b32 s64, scc // (op 1317) 

s_cmp_eq_i32 s64, 0 // (op 1317) 
s_cselect_b32 s55, s54, s53 // (op 1317) 


s_mul_i32 s53, s55, s14 // (op 1317) 

s_sub_u32 s55, s54, s53 // (op 1317) 

s_lshl_b32 s53, s55, 4 // (op 1317) 

s_mov_b32 s65, 0 // (op 1317) convert
s_mov_b32 s64, s53 // (op 1317) convert

s_ashr_i64 s[66:67], s[2:3], 7 // (op 1317) 
s_mul_i32 s53, s54, s66 // (op 1317) 


s_sub_u32 s54, s52, s53 // (op 1317) 


v_lshl_add_u32 v23, s54, 7, v1 // (op 1317) 

v_mul_lo_u32 v29, v23, s29 // (op 1317) most significant: low of low * high
v_mul_hi_u32 v25, v23, s28 // (op 1317) most significant: high of low * low
v_mul_lo_u32 v28, v23, s28 // (op 1317) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 1317) most significant: sum

v_mov_b32 v25, s64 // (op 1317) 
v_mov_b32 v27, s65 // (op 1317) 
v_add_co_u32 v28, s[52:53], v28, v25 // (op 1317) least significant half
v_addc_co_u32 v29, s[52:53], v29, v27, s[52:53] // (op 1317) most significant half





s_mul_i32 s53, s28, s3 // (op 1319) most significant: low of low * high
s_mul_i32 s54, s29, s2 // (op 1319) most significant: low of high * low
s_mul_i32 s52, s28, s2 // (op 1319) least significant
s_mul_hi_u32 s55, s28, s2 // (op 1319) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1319) most significant: sum
s_add_u32 s53, s53, s55 // (op 1319) most significant: sum


s_mov_b32 s64, s30 // (op 1319) reg expression
s_mov_b32 s65, s31 // (op 1319) reg expression
s_mov_b32 s66, s52 // (op 1319) reg expression
s_mov_b32 s67, 131072 // (op 1319) call()

s_mul_i32 s52, s16, s8 // (op 1336) 
s_add_u32 s52, s52, s0 // (op 1336) 
s_mul_hi_u32 s53, s52, s13 // (op 1336) 
s_sub_u32 s54, s52, s53 // (op 1336) 
s_lshr_b32 s55, s54, 1 // (op 1336) 

s_add_u32 s54, s55, s53 // (op 1336) 


s_lshr_b32 s53, s54, s11 // (op 1336) 

s_and_b32 s54, s11, 2147483648 // (op 1336) 
s_cmp_eq_i32 s54, 0 // (op 1336) 
s_mov_b32 s55, scc // (op 1336) 

s_cmp_eq_i32 s55, 0 // (op 1336) 
s_cselect_b32 s54, s52, s53 // (op 1336) 


s_mul_hi_u32 s53, s54, s12 // (op 1336) 
s_sub_u32 s55, s54, s53 // (op 1336) 
s_lshr_b32 s68, s55, 1 // (op 1336) 

s_add_u32 s55, s68, s53 // (op 1336) 


s_lshr_b32 s53, s55, s10 // (op 1336) 

s_and_b32 s55, s10, 2147483648 // (op 1336) 
s_cmp_eq_i32 s55, 0 // (op 1336) 
s_mov_b32 s68, scc // (op 1336) 

s_cmp_eq_i32 s68, 0 // (op 1336) 
s_cselect_b32 s55, s54, s53 // (op 1336) 


s_mul_i32 s53, s55, s14 // (op 1336) 

s_sub_u32 s55, s54, s53 // (op 1336) 

s_lshl_b32 s53, s55, 4 // (op 1336) 

s_mov_b32 s69, 0 // (op 1336) convert
s_mov_b32 s68, s53 // (op 1336) convert

s_ashr_i64 s[70:71], s[2:3], 7 // (op 1336) 
s_mul_i32 s53, s54, s70 // (op 1336) 


s_sub_u32 s54, s52, s53 // (op 1336) 


v_lshl_add_u32 v23, s54, 7, v1 // (op 1336) 

v_mul_lo_u32 v35, v23, s29 // (op 1336) most significant: low of low * high
v_mul_hi_u32 v25, v23, s28 // (op 1336) most significant: high of low * low
v_mul_lo_u32 v34, v23, s28 // (op 1336) least significant: low of low * low
v_add_u32 v35, v35, v25 // (op 1336) most significant: sum

v_mov_b32 v25, s68 // (op 1336) 
v_mov_b32 v27, s69 // (op 1336) 
v_add_co_u32 v34, s[52:53], v34, v25 // (op 1336) least significant half
v_addc_co_u32 v35, s[52:53], v35, v27, s[52:53] // (op 1336) most significant half





s_mul_i32 s53, s28, s3 // (op 1338) most significant: low of low * high
s_mul_i32 s54, s29, s2 // (op 1338) most significant: low of high * low
s_mul_i32 s52, s28, s2 // (op 1338) least significant
s_mul_hi_u32 s55, s28, s2 // (op 1338) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1338) most significant: sum
s_add_u32 s53, s53, s55 // (op 1338) most significant: sum


s_mov_b32 s64, s30 // (op 1338) reg expression
s_mov_b32 s65, s31 // (op 1338) reg expression
s_mov_b32 s66, s52 // (op 1338) reg expression
s_mov_b32 s67, 131072 // (op 1338) call()

v_and_b32 v23, -16, v1 // (op 1445) 
s_mul_i32 s52, s16, s8 // (op 1445) 
s_add_u32 s52, s52, s0 // (op 1445) 
v_sub_u32 v25, v1, v23 // (op 1445) 

s_mul_hi_u32 s53, s52, s13 // (op 1445) 
v_lshrrev_b32 v23, 4, v1 // (op 1445) 
s_sub_u32 s54, s52, s53 // (op 1445) 
s_lshr_b32 s55, s54, 1 // (op 1445) 

s_add_u32 s54, s55, s53 // (op 1445) 


s_lshr_b32 s53, s54, s11 // (op 1445) 

s_and_b32 s54, s11, 2147483648 // (op 1445) 
s_cmp_eq_i32 s54, 0 // (op 1445) 
s_mov_b32 s55, scc // (op 1445) 

s_cmp_eq_i32 s55, 0 // (op 1445) 
s_cselect_b32 s54, s52, s53 // (op 1445) 


s_mul_hi_u32 s53, s54, s12 // (op 1445) 
s_sub_u32 s55, s54, s53 // (op 1445) 
s_lshr_b32 s68, s55, 1 // (op 1445) 

s_add_u32 s55, s68, s53 // (op 1445) 


s_lshr_b32 s53, s55, s10 // (op 1445) 

s_and_b32 s55, s10, 2147483648 // (op 1445) 
s_cmp_eq_i32 s55, 0 // (op 1445) 
s_mov_b32 s68, scc // (op 1445) 

s_cmp_eq_i32 s68, 0 // (op 1445) 
s_cselect_b32 s55, s54, s53 // (op 1445) 


v_lshl_add_u32 v27, s55, 4, v25 // (op 1445) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 1445) 
s_mul_i32 s53, s54, s68 // (op 1445) 


s_sub_u32 s54, s52, s53 // (op 1445) 


v_lshl_add_u32 v25, s54, 2, v23 // (op 1445) 


v_mul_lo_u32 v23, v25, s32 // (op 1445) 

v_add_u32 v25, v27, v23 // (op 1445) 


s_mul_i32 s53, s32, s37 // (op 1447) most significant: low of low * high
s_mul_i32 s54, s33, s36 // (op 1447) most significant: low of high * low
s_mul_i32 s52, s32, s36 // (op 1447) least significant
s_mul_hi_u32 s55, s32, s36 // (op 1447) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1447) most significant: sum
s_add_u32 s53, s53, s55 // (op 1447) most significant: sum


s_mov_b32 s68, s34 // (op 1447) reg expression
s_mov_b32 s69, s35 // (op 1447) reg expression
s_mov_b32 s70, s52 // (op 1447) reg expression
s_mov_b32 s71, 131072 // (op 1447) call()

v_and_b32 v23, -16, v1 // (op 1464) 
s_mul_i32 s52, s16, s8 // (op 1464) 
s_add_u32 s52, s52, s0 // (op 1464) 
v_sub_u32 v27, v1, v23 // (op 1464) 

s_mul_hi_u32 s53, s52, s13 // (op 1464) 
v_lshrrev_b32 v23, 4, v1 // (op 1464) 
s_sub_u32 s54, s52, s53 // (op 1464) 
s_lshr_b32 s55, s54, 1 // (op 1464) 

s_add_u32 s54, s55, s53 // (op 1464) 


s_lshr_b32 s53, s54, s11 // (op 1464) 

s_and_b32 s54, s11, 2147483648 // (op 1464) 
s_cmp_eq_i32 s54, 0 // (op 1464) 
s_mov_b32 s55, scc // (op 1464) 

s_cmp_eq_i32 s55, 0 // (op 1464) 
s_cselect_b32 s54, s52, s53 // (op 1464) 


s_mul_hi_u32 s53, s54, s12 // (op 1464) 
s_sub_u32 s55, s54, s53 // (op 1464) 
s_lshr_b32 s72, s55, 1 // (op 1464) 

s_add_u32 s55, s72, s53 // (op 1464) 


s_lshr_b32 s53, s55, s10 // (op 1464) 

s_and_b32 s55, s10, 2147483648 // (op 1464) 
s_cmp_eq_i32 s55, 0 // (op 1464) 
s_mov_b32 s72, scc // (op 1464) 

s_cmp_eq_i32 s72, 0 // (op 1464) 
s_cselect_b32 s55, s54, s53 // (op 1464) 


v_lshl_add_u32 v36, s55, 4, v27 // (op 1464) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 1464) 
s_mul_i32 s53, s54, s72 // (op 1464) 


s_sub_u32 s54, s52, s53 // (op 1464) 


v_lshl_add_u32 v27, s54, 2, v23 // (op 1464) 


v_mul_lo_u32 v23, v27, s32 // (op 1464) 

v_add_u32 v27, v36, v23 // (op 1464) 


s_mul_i32 s53, s32, s37 // (op 1466) most significant: low of low * high
s_mul_i32 s54, s33, s36 // (op 1466) most significant: low of high * low
s_mul_i32 s52, s32, s36 // (op 1466) least significant
s_mul_hi_u32 s55, s32, s36 // (op 1466) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1466) most significant: sum
s_add_u32 s53, s53, s55 // (op 1466) most significant: sum


s_mov_b32 s68, s34 // (op 1466) reg expression
s_mov_b32 s69, s35 // (op 1466) reg expression
s_mov_b32 s70, s52 // (op 1466) reg expression
s_mov_b32 s71, 131072 // (op 1466) call()

v_and_b32 v23, -16, v1 // (op 1573) 
s_mul_i32 s52, s16, s8 // (op 1573) 
s_add_u32 s52, s52, s0 // (op 1573) 
v_sub_u32 v36, v1, v23 // (op 1573) 

s_mul_hi_u32 s53, s52, s13 // (op 1573) 
v_lshrrev_b32 v23, 4, v1 // (op 1573) 
s_sub_u32 s54, s52, s53 // (op 1573) 
s_lshr_b32 s55, s54, 1 // (op 1573) 

s_add_u32 s54, s55, s53 // (op 1573) 


s_lshr_b32 s53, s54, s11 // (op 1573) 

s_and_b32 s54, s11, 2147483648 // (op 1573) 
s_cmp_eq_i32 s54, 0 // (op 1573) 
s_mov_b32 s55, scc // (op 1573) 

s_cmp_eq_i32 s55, 0 // (op 1573) 
s_cselect_b32 s54, s52, s53 // (op 1573) 


s_mul_hi_u32 s53, s54, s12 // (op 1573) 
s_sub_u32 s55, s54, s53 // (op 1573) 
s_lshr_b32 s72, s55, 1 // (op 1573) 

s_add_u32 s55, s72, s53 // (op 1573) 


s_lshr_b32 s53, s55, s10 // (op 1573) 

s_and_b32 s55, s10, 2147483648 // (op 1573) 
s_cmp_eq_i32 s55, 0 // (op 1573) 
s_mov_b32 s72, scc // (op 1573) 

s_cmp_eq_i32 s72, 0 // (op 1573) 
s_cselect_b32 s55, s54, s53 // (op 1573) 


s_mul_i32 s53, s55, s14 // (op 1573) 

s_sub_u32 s55, s54, s53 // (op 1573) 

s_ashr_i64 s[72:73], s[2:3], 7 // (op 1573) 
v_lshl_add_u32 v37, s55, 4, v36 // (op 1573) 


s_mul_i32 s53, s54, s72 // (op 1573) 


s_sub_u32 s54, s52, s53 // (op 1573) 


v_lshl_add_u32 v36, s54, 2, v23 // (op 1573) 


v_mul_lo_u32 v23, v36, s38 // (op 1573) 
v_add_u32 v23, v23, v37 // (op 1573) 


s_mul_i32 s53, s38, s37 // (op 1575) most significant: low of low * high
s_mul_i32 s54, s39, s36 // (op 1575) most significant: low of high * low
s_mul_i32 s52, s38, s36 // (op 1575) least significant
s_mul_hi_u32 s55, s38, s36 // (op 1575) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1575) most significant: sum
s_add_u32 s53, s53, s55 // (op 1575) most significant: sum


s_mov_b32 s72, s40 // (op 1575) reg expression
s_mov_b32 s73, s41 // (op 1575) reg expression
s_mov_b32 s74, s52 // (op 1575) reg expression
s_mov_b32 s75, 131072 // (op 1575) call()

v_and_b32 v36, -16, v1 // (op 1592) 
s_mul_i32 s52, s16, s8 // (op 1592) 
s_add_u32 s52, s52, s0 // (op 1592) 
v_sub_u32 v37, v1, v36 // (op 1592) 

s_mul_hi_u32 s53, s52, s13 // (op 1592) 
v_lshrrev_b32 v36, 4, v1 // (op 1592) 
s_sub_u32 s54, s52, s53 // (op 1592) 
s_lshr_b32 s55, s54, 1 // (op 1592) 

s_add_u32 s54, s55, s53 // (op 1592) 


s_lshr_b32 s53, s54, s11 // (op 1592) 

s_and_b32 s54, s11, 2147483648 // (op 1592) 
s_cmp_eq_i32 s54, 0 // (op 1592) 
s_mov_b32 s55, scc // (op 1592) 

s_cmp_eq_i32 s55, 0 // (op 1592) 
s_cselect_b32 s54, s52, s53 // (op 1592) 


s_mul_hi_u32 s53, s54, s12 // (op 1592) 
s_sub_u32 s55, s54, s53 // (op 1592) 
s_lshr_b32 s76, s55, 1 // (op 1592) 

s_add_u32 s55, s76, s53 // (op 1592) 


s_lshr_b32 s53, s55, s10 // (op 1592) 

s_and_b32 s55, s10, 2147483648 // (op 1592) 
s_cmp_eq_i32 s55, 0 // (op 1592) 
s_mov_b32 s76, scc // (op 1592) 

s_cmp_eq_i32 s76, 0 // (op 1592) 
s_cselect_b32 s55, s54, s53 // (op 1592) 


s_mul_i32 s53, s55, s14 // (op 1592) 

s_sub_u32 s55, s54, s53 // (op 1592) 

v_lshl_add_u32 v38, s55, 4, v37 // (op 1592) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1592) 
s_mul_i32 s53, s54, s76 // (op 1592) 


s_sub_u32 s54, s52, s53 // (op 1592) 


v_lshl_add_u32 v37, s54, 2, v36 // (op 1592) 


v_mul_lo_u32 v36, v37, s38 // (op 1592) 
v_add_u32 v36, v36, v38 // (op 1592) 


s_mul_i32 s53, s38, s37 // (op 1594) most significant: low of low * high
s_mul_i32 s54, s39, s36 // (op 1594) most significant: low of high * low
s_mul_i32 s52, s38, s36 // (op 1594) least significant
s_mul_hi_u32 s55, s38, s36 // (op 1594) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1594) most significant: sum
s_add_u32 s53, s53, s55 // (op 1594) most significant: sum


s_mov_b32 s72, s40 // (op 1594) reg expression
s_mov_b32 s73, s41 // (op 1594) reg expression
s_mov_b32 s74, s52 // (op 1594) reg expression
s_mov_b32 s75, 131072 // (op 1594) call()

v_lshlrev_b32 v37, 4, v1 // (op 1781) 
s_nop 0
v_readfirstlane_b32 s52, v37 // (op 1781) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2669) 

s_lshl_b64 s[54:55], s[28:29], 6 // (op 1097) 
v_mov_b64 v[38:39], s[54:55] // (op 1097) convert


v_mov_b64 v[40:41], v[28:29] // (op 1097) reg expression
s_mov_b32 m0, s52 // (op 1097) reg expression
buffer_load_dwordx4 v40, s[64:67], 0 offen offset: 0 lds // (op 1097) Load value direct to lds
v_add_co_u32 v40, s[54:55], v40, v38 // (op 1097) least significant half
v_addc_co_u32 v41, s[54:55], 0, v41, s[54:55] // (op 1097) most significant half

s_add_u32 m0, m0, 1024 // (op 1097) 
buffer_load_dwordx4 v40, s[64:67], 0 offen offset: 0 lds // (op 1097) Load value direct to lds





v_mov_b32 v37, v23 // (op 852) reg expression

buffer_load_ubyte v38, v37, s[72:75], 0 offen offset: 0 // (op 852) Load value

v_lshlrev_b32 v37, 4, v1 // (op 1716) 
s_nop 0
v_readfirstlane_b32 s52, v37 // (op 1716) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2650) 

s_lshl_b64 s[54:55], s[24:25], 6 // (op 1102) 
v_mov_b64 v[40:41], s[54:55] // (op 1102) convert


v_mov_b64 v[42:43], v[30:31] // (op 1102) reg expression
s_add_u32 s53, 2048, s52 // (op 1102) 
s_mov_b32 m0, s53 // (op 1102) reg expression
buffer_load_dwordx4 v42, s[60:63], 0 offen offset: 0 lds // (op 1102) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 1102) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 1102) most significant half

s_add_u32 m0, m0, 1024 // (op 1102) 
buffer_load_dwordx4 v42, s[60:63], 0 offen offset: 0 lds // (op 1102) Load value direct to lds






v_mov_b32 v37, v25 // (op 858) reg expression

buffer_load_ubyte v39, v37, s[68:71], 0 offen offset: 0 // (op 858) Load value

v_lshlrev_b32 v37, 4, v1 // (op 2496) 
s_nop 0
v_readfirstlane_b32 s52, v37 // (op 2496) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2849) 

s_lshl_b64 s[54:55], s[28:29], 6 // (op 1107) 
v_mov_b64 v[40:41], s[54:55] // (op 1107) convert


s_lshl_b64 s[54:55], s[28:29], 7 // (op 1107) 
v_mov_b32 v37, s54 // (op 1107) 
v_mov_b32 v42, s55 // (op 1107) 
v_add_co_u32 v44, s[76:77], v28, v37 // (op 1107) least significant half
v_addc_co_u32 v45, s[76:77], v29, v42, s[76:77] // (op 1107) most significant half




s_add_u32 s53, 4096, s52 // (op 1107) 
s_mov_b32 m0, s53 // (op 1107) reg expression
buffer_load_dwordx4 v44, s[64:67], 0 offen offset: 0 lds // (op 1107) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v40 // (op 1107) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 1107) most significant half

s_add_u32 m0, m0, 1024 // (op 1107) 
buffer_load_dwordx4 v44, s[64:67], 0 offen offset: 0 lds // (op 1107) Load value direct to lds







s_lshl_b32 s52, s38, 2 // (op 864) 
v_add_u32 v37, v23, s52 // (op 864) 


buffer_load_ubyte v28, v37, s[72:75], 0 offen offset: 0 // (op 864) Load value


v_lshlrev_b32 v23, 4, v1 // (op 2457) 
s_nop 0
v_readfirstlane_b32 s52, v23 // (op 2457) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2839) 

s_lshl_b64 s[54:55], s[24:25], 6 // (op 1109) 
v_mov_b64 v[40:41], s[54:55] // (op 1109) convert


s_lshl_b64 s[54:55], s[24:25], 7 // (op 1109) 
v_mov_b32 v23, s54 // (op 1109) 
v_mov_b32 v29, s55 // (op 1109) 
v_add_co_u32 v42, s[76:77], v30, v23 // (op 1109) least significant half
v_addc_co_u32 v43, s[76:77], v31, v29, s[76:77] // (op 1109) most significant half




s_add_u32 s53, 6144, s52 // (op 1109) 
s_mov_b32 m0, s53 // (op 1109) reg expression
buffer_load_dwordx4 v42, s[60:63], 0 offen offset: 0 lds // (op 1109) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 1109) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 1109) most significant half

s_add_u32 m0, m0, 1024 // (op 1109) 
buffer_load_dwordx4 v42, s[60:63], 0 offen offset: 0 lds // (op 1109) Load value direct to lds







s_lshl_b32 s52, s32, 2 // (op 870) 
v_add_u32 v23, v25, s52 // (op 870) 


buffer_load_ubyte v37, v23, s[68:71], 0 offen offset: 0 // (op 870) Load value


v_and_b32 v23, -16, v1 // (op 1913) 
v_sub_u32 v25, v1, v23 // (op 1913) 

v_and_b32 v23, 4294967280, v1 // (op 1913) 
v_add_u32 v29, v23, v25 // (op 1913) 


s_barrier  // (op 2709) 



s_waitcnt vmcnt(9)
ds_write_b8 v29, v38 offset:8192 // (op 873) Store local data 

v_and_b32 v23, -16, v1 // (op 1847) 
v_sub_u32 v25, v1, v23 // (op 1847) 

v_and_b32 v23, 4294967280, v1 // (op 1847) 
v_add_u32 v29, v23, v25 // (op 1847) 


s_barrier  // (op 2689) 



s_waitcnt vmcnt(6)
ds_write_b8 v29, v39 offset:8256 // (op 875) Store local data 

s_barrier  // (op 2666) 



ds_read_b64_tr_b8 v[40:41], v22 // (op 892) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v22 offset:128 // (op 892) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v22 offset:1024 // (op 892) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v22 offset:1152 // (op 892) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2706) 


ds_read_u8 v23, v26 offset:8192 // (op 898) Load local data 
s_barrier  // (op 2647) 



ds_read_b64_tr_b8 v[48:49], v20 offset:2048 // (op 889) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v20 offset:2176 // (op 889) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v20 offset:3072 // (op 889) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v20 offset:3200 // (op 889) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2686) 


ds_read_u8 v25, v24 offset:8256 // (op 895) Load local data 
v_lshlrev_b32 v29, 4, v1 // (op 1688) 
s_nop 0
v_readfirstlane_b32 s52, v29 // (op 1688) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v29, 4, v1 // (op 1753) 
s_nop 0
v_readfirstlane_b32 s53, v29 // (op 1753) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -16, v1 // (op 1818) 
v_sub_u32 v30, v1, v29 // (op 1818) 

v_and_b32 v29, 4294967280, v1 // (op 1818) 
v_add_u32 v31, v29, v30 // (op 1818) 


v_and_b32 v29, -16, v1 // (op 1884) 
v_sub_u32 v30, v1, v29 // (op 1884) 

v_and_b32 v29, 4294967280, v1 // (op 1884) 
v_add_u32 v56, v29, v30 // (op 1884) 


v_lshlrev_b32 v29, 4, v1 // (op 2442) 
s_nop 0
v_readfirstlane_b32 s54, v29 // (op 2442) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v29, 4, v1 // (op 2481) 
s_nop 0
v_readfirstlane_b32 s55, v29 // (op 2481) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -16, v1 // (op 2520) 
v_sub_u32 v30, v1, v29 // (op 2520) 

v_and_b32 v29, 4294967280, v1 // (op 2520) 
v_add_u32 v57, v29, v30 // (op 2520) 


v_and_b32 v29, -16, v1 // (op 2544) 
v_sub_u32 v30, v1, v29 // (op 2544) 

v_and_b32 v29, 4294967280, v1 // (op 2544) 
v_add_u32 v58, v29, v30 // (op 2544) 


s_mov_b32 s1, 0 // (op 59) call()
s_and_b32 s76, s7, 4294967294 // (op 58) 
s_cmp_lt_u32 s1, s76 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_58 if false)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoop_58:
s_add_u32 s1, s1, 2 // (op 60) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[40:47], a[0:3], v25, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 88) 
s_barrier  // (op 2672) 

s_lshl_b64 s[76:77], s[28:29], 6 // (op 1036) 
v_mov_b64 v[60:61], s[76:77] // (op 1036) convert


v_mov_b64 v[62:63], v[34:35] // (op 1036) 
s_lshl_b64 s[76:77], s[28:29], 7 // (op 1036) 
s_lshl_b64 s[78:79], s[76:77], 1 // (op 1036) 

v_mov_b32 v59, s78 // (op 1036) 
v_mov_b32 v29, s79 // (op 1036) 
v_add_co_u32 v64, s[76:77], v62, v59 // (op 1036) least significant half
v_addc_co_u32 v65, s[76:77], v63, v29, s[76:77] // (op 1036) most significant half




s_mov_b32 m0, s53 // (op 1036) reg expression
buffer_load_dwordx4 v64, s[64:67], 0 offen offset: 0 lds // (op 1036) Load value direct to lds
v_add_co_u32 v64, s[76:77], v64, v60 // (op 1036) least significant half
v_addc_co_u32 v65, s[76:77], 0, v65, s[76:77] // (op 1036) most significant half

s_add_u32 m0, m0, 1024 // (op 1036) 
buffer_load_dwordx4 v64, s[64:67], 0 offen offset: 0 lds // (op 1036) Load value direct to lds




v_mov_b32 v29, v36 // (op 25) 
s_lshl_b32 s76, s38, 3 // (op 25) 
v_add_u32 v30, v29, s76 // (op 25) 


buffer_load_ubyte v38, v30, s[72:75], 0 offen offset: 0 // (op 25) Load value

s_barrier  // (op 2653) 

s_lshl_b64 s[76:77], s[24:25], 6 // (op 1031) 
v_mov_b64 v[60:61], s[76:77] // (op 1031) convert


v_mov_b64 v[64:65], v[32:33] // (op 1031) 
s_lshl_b64 s[76:77], s[24:25], 7 // (op 1031) 
s_lshl_b64 s[78:79], s[76:77], 1 // (op 1031) 

v_mov_b32 v30, s78 // (op 1031) 
v_mov_b32 v59, s79 // (op 1031) 
v_add_co_u32 v66, s[76:77], v64, v30 // (op 1031) least significant half
v_addc_co_u32 v67, s[76:77], v65, v59, s[76:77] // (op 1031) most significant half




s_add_u32 s76, 2048, s52 // (op 1031) 
s_nop 0
s_mov_b32 m0, s76 // (op 1031) reg expression
buffer_load_dwordx4 v66, s[60:63], 0 offen offset: 0 lds // (op 1031) Load value direct to lds
v_add_co_u32 v66, s[78:79], v66, v60 // (op 1031) least significant half
v_addc_co_u32 v67, s[78:79], 0, v67, s[78:79] // (op 1031) most significant half

s_add_u32 m0, m0, 1024 // (op 1031) 
buffer_load_dwordx4 v66, s[60:63], 0 offen offset: 0 lds // (op 1031) Load value direct to lds





v_mov_b32 v30, v27 // (op 17) 
s_lshl_b32 s76, s32, 3 // (op 17) 
v_add_u32 v59, v30, s76 // (op 17) 


buffer_load_ubyte v39, v59, s[68:71], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 2864) 



s_waitcnt vmcnt(9)
ds_write_b8 v58, v28 offset:8320 // (op 698) Store local data 
s_barrier  // (op 2858) 



s_waitcnt vmcnt(6)
ds_write_b8 v57, v37 offset:8384 // (op 697) Store local data 
s_barrier  // (op 2845) 



ds_read_b64_tr_b8 v[40:41], v14 offset:4096 // (op 690) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v14 offset:4224 // (op 690) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v14 offset:5120 // (op 690) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v14 offset:5248 // (op 690) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2861) 


ds_read_u8 v23, v19 offset:8320 // (op 694) Load local data 
s_barrier  // (op 2836) 
s_barrier  // (op 3009) 



ds_read_b64_tr_b8 v[48:49], v12 offset:6144 // (op 688) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v12 offset:6272 // (op 688) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v12 offset:7168 // (op 688) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v12 offset:7296 // (op 688) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2855) 


ds_read_u8 v25, v17 offset:8384 // (op 692) Load local data 
s_barrier  // (op 3003) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[40:47], a[0:3], v25, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 695) 
s_barrier  // (op 2852) 

s_lshl_b64 s[76:77], s[28:29], 6 // (op 1092) 
v_mov_b64 v[60:61], s[76:77] // (op 1092) convert


s_lshl_b64 s[76:77], s[28:29], 7 // (op 1092) 
s_mul_i32 s79, 3, s77 // (op 1092) most significant: low of low * high
s_mul_i32 s78, 3, s76 // (op 1092) least significant
s_mul_hi_u32 s80, 3, s76 // (op 1092) most significant: high of low * low
s_add_u32 s79, s79, s80 // (op 1092) most significant: sum


v_mov_b32 v59, s78 // (op 1092) 
v_mov_b32 v66, s79 // (op 1092) 
v_add_co_u32 v68, s[76:77], v62, v59 // (op 1092) least significant half
v_addc_co_u32 v69, s[76:77], v63, v66, s[76:77] // (op 1092) most significant half




s_add_u32 s76, 4096, s55 // (op 1092) 
s_nop 0
s_mov_b32 m0, s76 // (op 1092) reg expression
buffer_load_dwordx4 v68, s[64:67], 0 offen offset: 0 lds // (op 1092) Load value direct to lds
v_add_co_u32 v68, s[78:79], v68, v60 // (op 1092) least significant half
v_addc_co_u32 v69, s[78:79], 0, v69, s[78:79] // (op 1092) most significant half

s_add_u32 m0, m0, 1024 // (op 1092) 
buffer_load_dwordx4 v68, s[64:67], 0 offen offset: 0 lds // (op 1092) Load value direct to lds






s_lshl_b32 s76, s38, 2 // (op 701) 
s_mul_i32 s77, 3, s76 // (op 701) 

v_add_u32 v59, v29, s77 // (op 701) 


buffer_load_ubyte v28, v59, s[72:75], 0 offen offset: 0 // (op 701) Load value

s_lshl_b64 s[76:77], s[28:29], 7 // (op 1328) 

s_lshl_b64 s[78:79], s[76:77], 1 // (op 1328) 

s_barrier  // (op 2842) 
s_lshl_b32 s76, s38, 3 // (op 1584) 

v_add_u32 v36, v36, s76 // (op 1584) 

v_mov_b32 v29, s78 // (op 1328) 
v_mov_b32 v59, s79 // (op 1328) 
s_lshl_b64 s[76:77], s[24:25], 6 // (op 1087) 
v_mov_b64 v[60:61], s[76:77] // (op 1087) convert


s_lshl_b64 s[76:77], s[24:25], 7 // (op 1087) 
s_mul_i32 s81, 3, s77 // (op 1087) most significant: low of low * high
s_mul_i32 s80, 3, s76 // (op 1087) least significant
s_mul_hi_u32 s82, 3, s76 // (op 1087) most significant: high of low * low
s_add_u32 s81, s81, s82 // (op 1087) most significant: sum


v_mov_b32 v62, s80 // (op 1087) 
v_mov_b32 v63, s81 // (op 1087) 
v_add_co_u32 v34, s[76:77], v34, v29 // (op 1328) least significant half
v_addc_co_u32 v35, s[76:77], v35, v59, s[76:77] // (op 1328) most significant half




v_add_co_u32 v66, s[76:77], v64, v62 // (op 1087) least significant half
v_addc_co_u32 v67, s[76:77], v65, v63, s[76:77] // (op 1087) most significant half




s_add_u32 s76, 6144, s54 // (op 1087) 
s_nop 0
s_mov_b32 m0, s76 // (op 1087) reg expression
buffer_load_dwordx4 v66, s[60:63], 0 offen offset: 0 lds // (op 1087) Load value direct to lds
v_add_co_u32 v66, s[78:79], v66, v60 // (op 1087) least significant half
v_addc_co_u32 v67, s[78:79], 0, v67, s[78:79] // (op 1087) most significant half

s_add_u32 m0, m0, 1024 // (op 1087) 
buffer_load_dwordx4 v66, s[60:63], 0 offen offset: 0 lds // (op 1087) Load value direct to lds






s_lshl_b32 s76, s32, 2 // (op 700) 
s_mul_i32 s77, 3, s76 // (op 700) 

v_add_u32 v29, v30, s77 // (op 700) 


buffer_load_ubyte v37, v29, s[68:71], 0 offen offset: 0 // (op 700) Load value

s_lshl_b64 s[76:77], s[24:25], 7 // (op 1197) 

s_lshl_b64 s[78:79], s[76:77], 1 // (op 1197) 

s_barrier  // (op 2712) 
s_lshl_b32 s76, s32, 3 // (op 1456) 



v_add_u32 v27, v27, s76 // (op 1456) 

v_mov_b32 v29, s78 // (op 1197) 
v_mov_b32 v30, s79 // (op 1197) 
v_add_co_u32 v32, s[76:77], v32, v29 // (op 1197) least significant half
v_addc_co_u32 v33, s[76:77], v33, v30, s[76:77] // (op 1197) most significant half




s_waitcnt vmcnt(9)
ds_write_b8 v56, v38 offset:8192 // (op 30) Store local data 
s_barrier  // (op 2692) 



s_waitcnt vmcnt(6)
ds_write_b8 v31, v39 offset:8256 // (op 21) Store local data 
s_barrier  // (op 2678) 



ds_read_b64_tr_b8 v[40:41], v3 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v3 offset:128 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v3 offset:1024 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v3 offset:1152 // (op 75) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2719) 


ds_read_u8 v23, v7 offset:8192 // (op 85) Load local data 
s_barrier  // (op 2659) 



ds_read_b64_tr_b8 v[48:49], v0 offset:2048 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:2176 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:3072 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:3200 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2699) 


ds_read_u8 v25, v5 offset:8256 // (op 80) Load local data 
s_and_b32 s76, s7, 4294967294 // (op 58) 
s_cmp_lt_u32 s1, s76 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoop_58 if true)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_58:














s_barrier  // (op 2875) 
s_and_b32 s52, s7, 4294967294 // (op 1160) 
s_add_u32 s53, s0, s52 // (op 1160) 

s_mul_i32 s52, s16, s8 // (op 1160) 
s_add_u32 s52, s52, s53 // (op 1160) 

s_mul_hi_u32 s53, s52, s13 // (op 1160) 
s_sub_u32 s54, s52, s53 // (op 1160) 
s_lshr_b32 s55, s54, 1 // (op 1160) 

s_add_u32 s54, s55, s53 // (op 1160) 


s_lshr_b32 s53, s54, s11 // (op 1160) 

s_and_b32 s54, s11, 2147483648 // (op 1160) 
s_cmp_eq_i32 s54, 0 // (op 1160) 
s_mov_b32 s55, scc // (op 1160) 

s_cmp_eq_i32 s55, 0 // (op 1160) 
s_cselect_b32 s54, s52, s53 // (op 1160) 


s_mul_hi_u32 s53, s54, s12 // (op 1160) 
s_sub_u32 s55, s54, s53 // (op 1160) 
s_lshr_b32 s76, s55, 1 // (op 1160) 

s_add_u32 s55, s76, s53 // (op 1160) 


s_lshr_b32 s53, s55, s10 // (op 1160) 

s_and_b32 s55, s10, 2147483648 // (op 1160) 
s_cmp_eq_i32 s55, 0 // (op 1160) 
s_mov_b32 s76, scc // (op 1160) 

s_cmp_eq_i32 s76, 0 // (op 1160) 
s_cselect_b32 s55, s54, s53 // (op 1160) 


s_lshl_b32 s53, s55, 4 // (op 1160) 

s_mov_b32 s77, 0 // (op 1160) convert
s_mov_b32 s76, s53 // (op 1160) convert

s_ashr_i64 s[78:79], s[2:3], 7 // (op 1160) 
s_mul_i32 s53, s54, s78 // (op 1160) 


s_sub_u32 s54, s52, s53 // (op 1160) 


v_lshl_add_u32 v27, s54, 7, v1 // (op 1160) 

v_mul_lo_u32 v39, v27, s25 // (op 1160) most significant: low of low * high
v_mul_hi_u32 v29, v27, s24 // (op 1160) most significant: high of low * low
v_mul_lo_u32 v38, v27, s24 // (op 1160) least significant: low of low * low
v_add_u32 v39, v39, v29 // (op 1160) most significant: sum


v_mov_b32 v27, s76 // (op 1160) 
v_mov_b32 v29, s77 // (op 1160) 
v_add_co_u32 v30, s[52:53], v27, v38 // (op 1160) least significant half
v_addc_co_u32 v31, s[52:53], v29, v39, s[52:53] // (op 1160) most significant half





s_mul_i32 s53, s24, s3 // (op 1162) most significant: low of low * high
s_mul_i32 s54, s25, s2 // (op 1162) most significant: low of high * low
s_mul_i32 s52, s24, s2 // (op 1162) least significant
s_mul_hi_u32 s55, s24, s2 // (op 1162) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1162) most significant: sum
s_add_u32 s53, s53, s55 // (op 1162) most significant: sum


s_mov_b32 s60, s26 // (op 1162) reg expression
s_mov_b32 s61, s27 // (op 1162) reg expression
s_mov_b32 s62, s52 // (op 1162) reg expression
s_mov_b32 s63, 131072 // (op 1162) call()

s_and_b32 s52, s7, 4294967294 // (op 1294) 
s_add_u32 s53, s0, s52 // (op 1294) 

s_mul_i32 s52, s16, s8 // (op 1294) 
s_add_u32 s52, s52, s53 // (op 1294) 

s_mul_hi_u32 s53, s52, s13 // (op 1294) 
s_sub_u32 s54, s52, s53 // (op 1294) 
s_lshr_b32 s55, s54, 1 // (op 1294) 

s_add_u32 s54, s55, s53 // (op 1294) 


s_lshr_b32 s53, s54, s11 // (op 1294) 

s_and_b32 s54, s11, 2147483648 // (op 1294) 
s_cmp_eq_i32 s54, 0 // (op 1294) 
s_mov_b32 s55, scc // (op 1294) 

s_cmp_eq_i32 s55, 0 // (op 1294) 
s_cselect_b32 s54, s52, s53 // (op 1294) 


s_mul_hi_u32 s53, s54, s12 // (op 1294) 
s_sub_u32 s55, s54, s53 // (op 1294) 
s_lshr_b32 s76, s55, 1 // (op 1294) 

s_add_u32 s55, s76, s53 // (op 1294) 


s_lshr_b32 s53, s55, s10 // (op 1294) 

s_and_b32 s55, s10, 2147483648 // (op 1294) 
s_cmp_eq_i32 s55, 0 // (op 1294) 
s_mov_b32 s76, scc // (op 1294) 

s_cmp_eq_i32 s76, 0 // (op 1294) 
s_cselect_b32 s55, s54, s53 // (op 1294) 


s_mul_i32 s53, s55, s14 // (op 1294) 

s_sub_u32 s55, s54, s53 // (op 1294) 

s_lshl_b32 s53, s55, 4 // (op 1294) 

s_mov_b32 s77, 0 // (op 1294) convert
s_mov_b32 s76, s53 // (op 1294) convert

s_ashr_i64 s[78:79], s[2:3], 7 // (op 1294) 
s_mul_i32 s53, s54, s78 // (op 1294) 


s_sub_u32 s54, s52, s53 // (op 1294) 


v_lshl_add_u32 v27, s54, 7, v1 // (op 1294) 

v_mul_lo_u32 v39, v27, s29 // (op 1294) most significant: low of low * high
v_mul_hi_u32 v29, v27, s28 // (op 1294) most significant: high of low * low
v_mul_lo_u32 v38, v27, s28 // (op 1294) least significant: low of low * low
v_add_u32 v39, v39, v29 // (op 1294) most significant: sum

v_mov_b32 v29, s76 // (op 1294) 
v_mov_b32 v32, s77 // (op 1294) 
v_add_co_u32 v38, s[52:53], v38, v29 // (op 1294) least significant half
v_addc_co_u32 v39, s[52:53], v39, v32, s[52:53] // (op 1294) most significant half





s_mul_i32 s53, s28, s3 // (op 1296) most significant: low of low * high
s_mul_i32 s54, s29, s2 // (op 1296) most significant: low of high * low
s_mul_i32 s52, s28, s2 // (op 1296) least significant
s_mul_hi_u32 s55, s28, s2 // (op 1296) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1296) most significant: sum
s_add_u32 s53, s53, s55 // (op 1296) most significant: sum


s_mov_b32 s64, s30 // (op 1296) reg expression
s_mov_b32 s65, s31 // (op 1296) reg expression
s_mov_b32 s66, s52 // (op 1296) reg expression
s_mov_b32 s67, 131072 // (op 1296) call()

s_and_b32 s52, s7, 4294967294 // (op 1422) 
v_and_b32 v27, -16, v1 // (op 1422) 
s_add_u32 s53, s0, s52 // (op 1422) 

v_sub_u32 v29, v1, v27 // (op 1422) 

v_lshrrev_b32 v27, 4, v1 // (op 1422) 
s_mul_i32 s52, s16, s8 // (op 1422) 
s_add_u32 s52, s52, s53 // (op 1422) 

s_mul_hi_u32 s53, s52, s13 // (op 1422) 
s_sub_u32 s54, s52, s53 // (op 1422) 
s_lshr_b32 s55, s54, 1 // (op 1422) 

s_add_u32 s54, s55, s53 // (op 1422) 


s_lshr_b32 s53, s54, s11 // (op 1422) 

s_and_b32 s54, s11, 2147483648 // (op 1422) 
s_cmp_eq_i32 s54, 0 // (op 1422) 
s_mov_b32 s55, scc // (op 1422) 

s_cmp_eq_i32 s55, 0 // (op 1422) 
s_cselect_b32 s54, s52, s53 // (op 1422) 


s_mul_hi_u32 s53, s54, s12 // (op 1422) 
s_sub_u32 s55, s54, s53 // (op 1422) 
s_lshr_b32 s76, s55, 1 // (op 1422) 

s_add_u32 s55, s76, s53 // (op 1422) 


s_lshr_b32 s53, s55, s10 // (op 1422) 

s_and_b32 s55, s10, 2147483648 // (op 1422) 
s_cmp_eq_i32 s55, 0 // (op 1422) 
s_mov_b32 s76, scc // (op 1422) 

s_cmp_eq_i32 s76, 0 // (op 1422) 
s_cselect_b32 s55, s54, s53 // (op 1422) 


v_lshl_add_u32 v32, s55, 4, v29 // (op 1422) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1422) 
s_mul_i32 s53, s54, s76 // (op 1422) 


s_sub_u32 s54, s52, s53 // (op 1422) 


v_lshl_add_u32 v29, s54, 2, v27 // (op 1422) 


v_mul_lo_u32 v27, v29, s32 // (op 1422) 

v_add_u32 v29, v32, v27 // (op 1422) 


s_mul_i32 s53, s32, s37 // (op 1424) most significant: low of low * high
s_mul_i32 s54, s33, s36 // (op 1424) most significant: low of high * low
s_mul_i32 s52, s32, s36 // (op 1424) least significant
s_mul_hi_u32 s55, s32, s36 // (op 1424) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1424) most significant: sum
s_add_u32 s53, s53, s55 // (op 1424) most significant: sum


s_mov_b32 s68, s34 // (op 1424) reg expression
s_mov_b32 s69, s35 // (op 1424) reg expression
s_mov_b32 s70, s52 // (op 1424) reg expression
s_mov_b32 s71, 131072 // (op 1424) call()

s_and_b32 s52, s7, 4294967294 // (op 1550) 
v_and_b32 v27, -16, v1 // (op 1550) 
s_add_u32 s53, s0, s52 // (op 1550) 

v_sub_u32 v32, v1, v27 // (op 1550) 

v_lshrrev_b32 v27, 4, v1 // (op 1550) 
s_mul_i32 s52, s16, s8 // (op 1550) 
s_add_u32 s52, s52, s53 // (op 1550) 

s_mul_hi_u32 s53, s52, s13 // (op 1550) 
s_sub_u32 s54, s52, s53 // (op 1550) 
s_lshr_b32 s55, s54, 1 // (op 1550) 

s_add_u32 s54, s55, s53 // (op 1550) 


s_lshr_b32 s53, s54, s11 // (op 1550) 

s_and_b32 s54, s11, 2147483648 // (op 1550) 
s_cmp_eq_i32 s54, 0 // (op 1550) 
s_mov_b32 s55, scc // (op 1550) 

s_cmp_eq_i32 s55, 0 // (op 1550) 
s_cselect_b32 s54, s52, s53 // (op 1550) 


s_mul_hi_u32 s53, s54, s12 // (op 1550) 
s_sub_u32 s55, s54, s53 // (op 1550) 
s_lshr_b32 s76, s55, 1 // (op 1550) 

s_add_u32 s55, s76, s53 // (op 1550) 


s_lshr_b32 s53, s55, s10 // (op 1550) 

s_and_b32 s55, s10, 2147483648 // (op 1550) 
s_cmp_eq_i32 s55, 0 // (op 1550) 
s_mov_b32 s76, scc // (op 1550) 

s_cmp_eq_i32 s76, 0 // (op 1550) 
s_cselect_b32 s55, s54, s53 // (op 1550) 


s_mul_i32 s53, s55, s14 // (op 1550) 

s_sub_u32 s55, s54, s53 // (op 1550) 

v_lshl_add_u32 v33, s55, 4, v32 // (op 1550) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1550) 
s_mul_i32 s53, s54, s76 // (op 1550) 


s_sub_u32 s54, s52, s53 // (op 1550) 


v_lshl_add_u32 v32, s54, 2, v27 // (op 1550) 


v_mul_lo_u32 v27, v32, s38 // (op 1550) 
v_add_u32 v27, v27, v33 // (op 1550) 


s_mul_i32 s53, s38, s37 // (op 1552) most significant: low of low * high
s_mul_i32 s54, s39, s36 // (op 1552) most significant: low of high * low
s_mul_i32 s52, s38, s36 // (op 1552) least significant
s_mul_hi_u32 s55, s38, s36 // (op 1552) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1552) most significant: sum
s_add_u32 s53, s53, s55 // (op 1552) most significant: sum


s_mov_b32 s72, s40 // (op 1552) reg expression
s_mov_b32 s73, s41 // (op 1552) reg expression
s_mov_b32 s74, s52 // (op 1552) reg expression
s_mov_b32 s75, 131072 // (op 1552) call()

v_lshlrev_b32 v32, 4, v1 // (op 1701) 
s_nop 0
v_readfirstlane_b32 s52, v32 // (op 1701) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v32, 4, v1 // (op 1766) 
s_nop 0
v_readfirstlane_b32 s53, v32 // (op 1766) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v32, -16, v1 // (op 1831) 
v_sub_u32 v33, v1, v32 // (op 1831) 

v_and_b32 v32, 4294967280, v1 // (op 1831) 
v_add_u32 v34, v32, v33 // (op 1831) 


v_and_b32 v32, -16, v1 // (op 1897) 
v_sub_u32 v33, v1, v32 // (op 1897) 

v_and_b32 v32, 4294967280, v1 // (op 1897) 
v_add_u32 v35, v32, v33 // (op 1897) 


s_and_b32 s54, s7, 4294967294 // (op 615) 
s_cmp_lt_u32 s54, s7 // (op 614) 






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_614 // (op 614) Condition: Top (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_614 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_614:
s_barrier  // (op 2656) 
s_add_u32 s54, s54, 1 // (op 616) 

s_lshl_b64 s[76:77], s[24:25], 6 // (op 1074) 
v_mov_b64 v[32:33], s[76:77] // (op 1074) convert


v_mov_b64 v[36:37], v[30:31] // (op 1074) 
v_mov_b64 v[40:41], v[36:37] // (op 1074) reg expression
s_add_u32 s55, 2048, s52 // (op 1074) 
s_mov_b32 m0, s55 // (op 1074) reg expression
buffer_load_dwordx4 v40, s[60:63], 0 offen offset: 0 lds // (op 1074) Load value direct to lds
v_add_co_u32 v40, s[76:77], v40, v32 // (op 1074) least significant half
v_addc_co_u32 v41, s[76:77], 0, v41, s[76:77] // (op 1074) most significant half

s_add_u32 m0, m0, 1024 // (op 1074) 
buffer_load_dwordx4 v40, s[60:63], 0 offen offset: 0 lds // (op 1074) Load value direct to lds




s_lshl_b64 s[76:77], s[24:25], 7 // (op 1150) 
v_mov_b32 v23, s76 // (op 1150) 
v_mov_b32 v25, s77 // (op 1150) 
v_add_co_u32 v30, s[78:79], v30, v23 // (op 1150) least significant half
v_addc_co_u32 v31, s[78:79], v31, v25, s[78:79] // (op 1150) most significant half




s_barrier  // (op 2675) 

s_lshl_b64 s[76:77], s[28:29], 6 // (op 1080) 
v_mov_b64 v[32:33], s[76:77] // (op 1080) convert


v_mov_b64 v[36:37], v[38:39] // (op 1080) 
v_mov_b64 v[40:41], v[36:37] // (op 1080) reg expression
s_mov_b32 m0, s53 // (op 1080) reg expression
buffer_load_dwordx4 v40, s[64:67], 0 offen offset: 0 lds // (op 1080) Load value direct to lds
v_add_co_u32 v40, s[76:77], v40, v32 // (op 1080) least significant half
v_addc_co_u32 v41, s[76:77], 0, v41, s[76:77] // (op 1080) most significant half

s_add_u32 m0, m0, 1024 // (op 1080) 
buffer_load_dwordx4 v40, s[64:67], 0 offen offset: 0 lds // (op 1080) Load value direct to lds



s_lshl_b64 s[76:77], s[28:29], 7 // (op 1286) 
v_mov_b32 v23, s76 // (op 1286) 
v_mov_b32 v25, s77 // (op 1286) 
v_add_co_u32 v38, s[78:79], v38, v23 // (op 1286) least significant half
v_addc_co_u32 v39, s[78:79], v39, v25, s[78:79] // (op 1286) most significant half






v_mov_b32 v23, v29 // (op 634) 
v_mov_b32 v25, v23 // (op 634) reg expression

buffer_load_ubyte v28, v25, s[68:71], 0 offen offset: 0 // (op 634) Load value


s_lshl_b32 s55, s32, 2 // (op 1414) 
v_add_u32 v29, v29, s55 // (op 1414) 



s_barrier  // (op 2695) 
v_mov_b32 v23, v27 // (op 635) 



v_mov_b32 v25, v23 // (op 635) reg expression

buffer_load_ubyte v32, v25, s[72:75], 0 offen offset: 0 // (op 635) Load value


s_lshl_b32 s55, s38, 2 // (op 1542) 
v_add_u32 v27, v27, s55 // (op 1542) 

s_waitcnt vmcnt(1)
ds_write_b8 v34, v28 offset:8256 // (op 631) Store local data 

s_barrier  // (op 2662) 



s_barrier  // (op 2715) 
ds_read_b64_tr_b8 v[40:41], v0 offset:2048 // (op 622) Transpose load local data 



ds_read_b64_tr_b8 v[42:43], v0 offset:2176 // (op 622) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:3072 // (op 622) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:3200 // (op 622) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v35, v32 offset:8192 // (op 632) Store local data 

s_barrier  // (op 2682) 



ds_read_b64_tr_b8 v[48:49], v3 // (op 624) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v3 offset:128 // (op 624) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v3 offset:1024 // (op 624) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v3 offset:1152 // (op 624) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2702) 


ds_read_u8 v23, v5 offset:8256 // (op 626) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2722) 


ds_read_u8 v25, v7 offset:8192 // (op 628) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[48:55], a[0:3], v23, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 629) 




s_cmp_lt_u32 s54, s7 // (op 614) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_614 // (op 614) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_614 if true)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_614:









s_barrier  // (op 2995) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_lg_u32 s18, 0 // (op 314) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Send_Tile_314 // (op 314) Condition: False, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Send_Tile_314
v_lshrrev_b32 v20, 4, v1 // (op 2284) 
v_lshlrev_b32 v22, 2, v20 // (op 2284) 

v_lshl_add_u32 v20, s16, 4, v22 // (op 2284) 

v_and_b32 v22, -16, v1 // (op 2284) 
v_sub_u32 v23, v1, v22 // (op 2284) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 2284) 


v_lshlrev_b32 v20, 2, v22 // (op 2284) 

s_lshl_b32 s52, s15, 10 // (op 2286) 
s_mov_b32 s76, s42 // (op 2286) reg expression
s_mov_b32 s77, s43 // (op 2286) reg expression
s_mov_b32 s78, s52 // (op 2286) reg expression
s_mov_b32 s79, 131072 // (op 2286) call()




buffer_store_dword a0, v20, s[76:79], 0 offen sc1 // (op 311) Store value
buffer_store_dword a1, v20, s[76:79], 0 offen offset:64 sc1 // (op 311) Store value
buffer_store_dword a2, v20, s[76:79], 0 offen offset:128 sc1 // (op 311) Store value
buffer_store_dword a3, v20, s[76:79], 0 offen offset:192 sc1 // (op 311) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 315) 
v_cmp_eq_i32 vcc, v1, 0 // (op 319) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_319 // (op 319) Condition: False, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_319
s_mov_b32 s52, 1 // (op 317) call()
s_mov_b32 s55, 0 // (op 318) convert
s_mov_b32 s54, s16 // (op 318) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 318) 
s_add_u32 s80, s44, s54 // (op 318) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 318) End of Int64 add, unlocking SCC
s_store_dword s52, s[80:81], 0 glc // (op 318) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_319 // (op 319) Condition: Done, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_319
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Wave0_Store_Flag_319:
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Wave0_Store_Flag_319:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Send_Tile_314 // (op 314) Condition: Done, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Send_Tile_314
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Send_Tile_314:
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Send_Tile_314:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 330) 
s_mov_b32 s52, scc // (op 330) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 330) 
s_mov_b32 s53, s54 // (op 330) convert

s_sub_u32 s54, s53, 1 // (op 330) 

s_cmp_lt_u32 s19, s54 // (op 330) 
s_mov_b32 s53, scc // (op 330) 

s_and_b32 s54, s52, s53 // (op 330) 
s_cmp_lg_u32 s54, 0 // (op 330) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Receive_Tile_330 // (op 330) Condition: False, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Receive_Tile_330
s_mov_b32 s52, 0 // (op 296) call()
s_mul_i32 s53, s52, s8 // (op 295) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 295) 
s_mov_b32 s80, s54 // (op 295) convert

s_sub_u32 s54, s80, s19 // (op 295) 

s_sub_u32 s55, s54, 1 // (op 295) 

s_cmp_lt_u32 s53, s55 // (op 295) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_295 // (op 295) Condition: Top (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_295 if false)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_ReceiveTileLoop_295:
s_add_u32 s53, s16, 1 // (op 2301) 
v_lshrrev_b32 v20, 4, v1 // (op 2301) 
s_add_u32 s54, s53, s52 // (op 2301) 
s_lshl_b32 s54, s54, 4 // (op 2301) 

v_lshlrev_b32 v22, 2, v20 // (op 2301) 

v_add_lshl_u32 v20, s54, v22, 4 // (op 2301) 


v_and_b32 v22, -16, v1 // (op 2301) 
v_sub_u32 v23, v1, v22 // (op 2301) 

v_add_lshl_u32 v22, v20, v23, 2 // (op 2301) 


s_lshl_b32 s53, s15, 10 // (op 2303) 
s_mov_b32 s76, s42 // (op 2303) reg expression
s_mov_b32 s77, s43 // (op 2303) reg expression
s_mov_b32 s78, s53 // (op 2303) reg expression
s_mov_b32 s79, 131072 // (op 2303) call()

RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_334:
s_add_u32 s53, s16, 1 // (op 332) 
s_add_u32 s54, s53, s52 // (op 332) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 332) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 332) 

s_add_u32 s80, s44, s54 // (op 332) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 332) End of Int64 add, unlocking SCC
s_load_dword s53, s[80:81], 0 glc // (op 332) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 334) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_334 // (op 334) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_DoWhileTop_Global_sync_spin_loop_334 if true)
s_barrier  // (op 338) 
v_cmp_eq_i32 vcc, v1, 0 // (op 337) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_337 // (op 337) Condition: False, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_337
s_mov_b32 s53, 0 // (op 335) call()
s_add_u32 s54, s16, 1 // (op 336) 
s_add_u32 s80, s54, s52 // (op 336) Start of Int64 add, locking SCC
s_addc_u32 s81, 0, 0 // (op 336) End of Int64 add, unlocking SCC
s_lshl_b64 s[80:81], s[80:81], 2 // (op 336) 

s_add_u32 s54, s44, s80 // (op 336) Start of Int64 add, locking SCC
s_addc_u32 s55, s45, s81 // (op 336) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 336) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_337 // (op 337) Condition: Done, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_337
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_337:
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_337:




buffer_load_dword v24, v22, s[76:79], 0 offen // (op 349) Load value
buffer_load_dword v25, v22, s[76:79], 0 offen offset:64 // (op 349) Load value
buffer_load_dword v26, v22, s[76:79], 0 offen offset:128 // (op 349) Load value
buffer_load_dword v27, v22, s[76:79], 0 offen offset:192 // (op 349) Load value
v_accvgpr_read v28, a0 // (op 352) reg expression
v_accvgpr_read v29, a1 // (op 352) reg expression
v_accvgpr_read v30, a2 // (op 352) reg expression
v_accvgpr_read v31, a3 // (op 352) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 354) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 354) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 354) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 354) 

v_accvgpr_write a0, v28 // (op 356) reg expression
v_accvgpr_write a1, v29 // (op 356) reg expression
v_accvgpr_write a2, v30 // (op 356) reg expression
v_accvgpr_write a3, v31 // (op 356) reg expression

s_add_u32 s52, s52, 1 // (op 297) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 295) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 295) 
s_mov_b32 s80, s54 // (op 295) convert

s_sub_u32 s54, s80, s19 // (op 295) 

s_sub_u32 s55, s54, 1 // (op 295) 

s_cmp_lt_u32 s53, s55 // (op 295) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_ReceiveTileLoop_295 // (op 295) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_ReceiveTileLoop_295 if true)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_ReceiveTileLoop_295:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Receive_Tile_330 // (op 330) Condition: Done, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Receive_Tile_330
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Receive_Tile_330:
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Receive_Tile_330:

s_cmp_eq_i32 s18, 0 // (op 373) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_373 // (op 373) Condition: False, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_373
v_cmp_eq_f32 vcc, 0.00000, v16 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_133
v_accvgpr_read v22, a0 // (op 50) 
v_accvgpr_read v23, a1 // (op 50) 
v_accvgpr_read v24, a2 // (op 50) 
v_accvgpr_read v25, a3 // (op 50) 
v_mul_f32 v26, v18, v22 // (op 50) 
v_mul_f32 v27, v18, v23 // (op 50) 
v_mul_f32 v28, v18, v24 // (op 50) 
v_mul_f32 v29, v18, v25 // (op 50) 

v_lshrrev_b32 v20, 6, v1 // (op 1952) 
s_mul_i32 s52, s16, s8 // (op 1952) 
s_add_u32 s52, s52, s0 // (op 1952) 
v_and_b32 v22, 63, v1 // (op 1952) 
s_mul_hi_u32 s53, s52, s13 // (op 1952) 
v_lshrrev_b32 v23, 4, v22 // (op 1952) 
s_sub_u32 s54, s52, s53 // (op 1952) 
v_lshlrev_b32 v24, 2, v23 // (op 1952) 

s_lshr_b32 s55, s54, 1 // (op 1952) 

v_lshl_add_u32 v23, v20, 4, v24 // (op 1952) 


s_add_u32 s54, s55, s53 // (op 1952) 


v_lshrrev_b32 v20, 2, v22 // (op 1952) 

s_lshr_b32 s53, s54, s11 // (op 1952) 

v_and_b32 v22, 3, v20 // (op 1952) 

s_and_b32 s54, s11, 2147483648 // (op 1952) 
v_and_b32 v20, 3, v1 // (op 1952) 
s_cmp_eq_i32 s54, 0 // (op 1952) 
s_mov_b32 s55, scc // (op 1952) 

v_lshl_add_u32 v24, v22, 2, v20 // (op 1952) 


s_cmp_eq_i32 s55, 0 // (op 1952) 
s_cselect_b32 s54, s52, s53 // (op 1952) 



s_mul_hi_u32 s55, s54, s12 // (op 1952) 
s_sub_u32 s52, s54, s55 // (op 1952) 
s_lshr_b32 s53, s52, 1 // (op 1952) 

s_add_u32 s52, s53, s55 // (op 1952) 


s_lshr_b32 s53, s52, s10 // (op 1952) 

s_and_b32 s52, s10, 2147483648 // (op 1952) 
s_cmp_eq_i32 s52, 0 // (op 1952) 
s_mov_b32 s55, scc // (op 1952) 

s_cmp_eq_i32 s55, 0 // (op 1952) 
s_cselect_b32 s52, s54, s53 // (op 1952) 


s_mul_i32 s53, s52, s14 // (op 1952) 
v_lshl_add_u32 v20, s52, 4, v23 // (op 1952) 


s_sub_u32 s52, s54, s53 // (op 1952) 


v_lshl_add_u32 v25, s52, 4, v24 // (op 1952) 


v_mul_lo_u32 v22, v25, s46 // (op 1952) 

v_add_lshl_u32 v23, v20, v22, 2 // (op 1952) 


s_mul_i32 s53, s46, s23 // (op 1954) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1954) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1954) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1954) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1954) most significant: sum
s_add_u32 s53, s53, s55 // (op 1954) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1954) 

s_mov_b32 s76, s48 // (op 1954) reg expression
s_mov_b32 s77, s49 // (op 1954) reg expression
s_mov_b32 s78, s54 // (op 1954) reg expression
s_mov_b32 s79, 131072 // (op 1954) call()




buffer_store_dwordx4 v[26:29], v23, s[76:79], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_133
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_133:



buffer_load_dwordx4 v[22:25], v21, s[56:59], 0 offen // (op 145) Load value
v_accvgpr_read v26, a0 // (op 142) 
v_accvgpr_read v27, a1 // (op 142) 
v_accvgpr_read v28, a2 // (op 142) 
v_accvgpr_read v29, a3 // (op 142) 
v_mul_f32 v30, v18, v26 // (op 142) 
v_mul_f32 v31, v18, v27 // (op 142) 
v_mul_f32 v32, v18, v28 // (op 142) 
v_mul_f32 v33, v18, v29 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v26, v16, v22 // (op 146) 
v_mul_f32 v27, v16, v23 // (op 146) 
v_mul_f32 v28, v16, v24 // (op 146) 
v_mul_f32 v29, v16, v25 // (op 146) 

v_add_f32 v22, v26, v30 // (op 143) 
v_add_f32 v23, v27, v31 // (op 143) 
v_add_f32 v24, v28, v32 // (op 143) 
v_add_f32 v25, v29, v33 // (op 143) 


v_lshrrev_b32 v20, 6, v1 // (op 1969) 
s_mul_i32 s52, s16, s8 // (op 1969) 
s_add_u32 s52, s52, s0 // (op 1969) 
v_and_b32 v26, 63, v1 // (op 1969) 
s_mul_hi_u32 s53, s52, s13 // (op 1969) 
v_lshrrev_b32 v27, 4, v26 // (op 1969) 
s_sub_u32 s54, s52, s53 // (op 1969) 
v_lshlrev_b32 v28, 2, v27 // (op 1969) 

s_lshr_b32 s55, s54, 1 // (op 1969) 

v_lshl_add_u32 v27, v20, 4, v28 // (op 1969) 


s_add_u32 s54, s55, s53 // (op 1969) 


v_lshrrev_b32 v20, 2, v26 // (op 1969) 

s_lshr_b32 s53, s54, s11 // (op 1969) 

v_and_b32 v26, 3, v20 // (op 1969) 

s_and_b32 s54, s11, 2147483648 // (op 1969) 
v_and_b32 v20, 3, v1 // (op 1969) 
s_cmp_eq_i32 s54, 0 // (op 1969) 
s_mov_b32 s55, scc // (op 1969) 

v_lshl_add_u32 v28, v26, 2, v20 // (op 1969) 


s_cmp_eq_i32 s55, 0 // (op 1969) 
s_cselect_b32 s54, s52, s53 // (op 1969) 



s_mul_hi_u32 s55, s54, s12 // (op 1969) 
s_sub_u32 s52, s54, s55 // (op 1969) 
s_lshr_b32 s53, s52, 1 // (op 1969) 

s_add_u32 s52, s53, s55 // (op 1969) 


s_lshr_b32 s53, s52, s10 // (op 1969) 

s_and_b32 s52, s10, 2147483648 // (op 1969) 
s_cmp_eq_i32 s52, 0 // (op 1969) 
s_mov_b32 s55, scc // (op 1969) 

s_cmp_eq_i32 s55, 0 // (op 1969) 
s_cselect_b32 s52, s54, s53 // (op 1969) 


s_mul_i32 s53, s52, s14 // (op 1969) 
v_lshl_add_u32 v20, s52, 4, v27 // (op 1969) 


s_sub_u32 s52, s54, s53 // (op 1969) 


v_lshl_add_u32 v26, s52, 4, v28 // (op 1969) 


v_mul_lo_u32 v27, v26, s46 // (op 1969) 

v_add_lshl_u32 v26, v20, v27, 2 // (op 1969) 


s_mul_i32 s53, s46, s23 // (op 1971) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1971) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1971) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1971) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1971) most significant: sum
s_add_u32 s53, s53, s55 // (op 1971) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1971) 

s_mov_b32 s76, s48 // (op 1971) reg expression
s_mov_b32 s77, s49 // (op 1971) reg expression
s_mov_b32 s78, s54 // (op 1971) reg expression
s_mov_b32 s79, 131072 // (op 1971) call()




buffer_store_dwordx4 v[22:25], v26, s[76:79], 0 offen // (op 144) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_133:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_373 // (op 373) Condition: Done, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_373
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_373:
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_373:

s_add_u32 s0, s0, s7 // (op 288) 
s_mul_i32 s52, s8, s16 // (op 286) 
s_cmp_gt_u32 s9, s52 // (op 286) 
s_mov_b32 s53, scc // (op 286) 
s_sub_u32 s54, s9, s52 // (op 286) 

s_cmp_eq_i32 s53, 0 // (op 286) 
s_cselect_b32 s52, 0, s54 // (op 286) 


s_cmp_lt_u32 s8, s52 // (op 286) 
s_mov_b32 s53, scc // (op 286) 
s_cmp_eq_i32 s53, 0 // (op 286) 
s_cselect_b32 s54, s52, s8 // (op 286) 


s_cmp_lt_u32 s0, s54 // (op 286) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_SKStreamTileLoop_286 // (op 286) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_SKStreamTileLoop_286 if true)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_SKStreamTileLoop_286:

















s_mov_b32 s0, 0 // (op 427) call()
s_mov_b32 s1, 0 // (op 428) call()
s_mov_b32 s0, 0 // (op 442) call()
s_mul_i32 s15, s50, s16 // (op 441) 
s_cmp_gt_u32 s51, s15 // (op 441) 
s_mov_b32 s6, scc // (op 441) 
s_sub_u32 s7, s51, s15 // (op 441) 

s_cmp_eq_i32 s6, 0 // (op 441) 
s_cselect_b32 s8, 0, s7 // (op 441) 


s_cmp_lt_u32 s50, s8 // (op 441) 
s_mov_b32 s15, scc // (op 441) 
s_cmp_eq_i32 s15, 0 // (op 441) 
s_cselect_b32 s6, s8, s50 // (op 441) 


s_cmp_lt_u32 s0, s6 // (op 441) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_441 // (op 441) Condition: Top (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_441 if false)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_DPStreamTileLoop_441:
v_lshrrev_b32 v0, 6, v1 // (op 1671) 
s_mul_i32 s15, s16, s50 // (op 1671) 
s_add_u32 s15, s15, s0 // (op 1671) 
v_and_b32 v3, 63, v1 // (op 1671) 
s_add_u32 s6, s15, s9 // (op 1671) 

v_lshrrev_b32 v5, 4, v3 // (op 1671) 
s_mul_hi_u32 s15, s6, s13 // (op 1671) 
v_lshlrev_b32 v7, 2, v5 // (op 1671) 

s_sub_u32 s7, s6, s15 // (op 1671) 
v_lshl_add_u32 v5, v0, 4, v7 // (op 1671) 


s_lshr_b32 s8, s7, 1 // (op 1671) 

v_lshrrev_b32 v0, 2, v3 // (op 1671) 

s_add_u32 s7, s8, s15 // (op 1671) 


v_and_b32 v3, 3, v0 // (op 1671) 

s_lshr_b32 s8, s7, s11 // (op 1671) 

v_and_b32 v0, 3, v1 // (op 1671) 
s_and_b32 s7, s11, 2147483648 // (op 1671) 
v_lshl_add_u32 v7, v3, 2, v0 // (op 1671) 


s_cmp_eq_i32 s7, 0 // (op 1671) 
s_mov_b32 s15, scc // (op 1671) 

s_cmp_eq_i32 s15, 0 // (op 1671) 
s_cselect_b32 s7, s6, s8 // (op 1671) 



s_mul_hi_u32 s6, s7, s12 // (op 1671) 
s_sub_u32 s8, s7, s6 // (op 1671) 
s_lshr_b32 s15, s8, 1 // (op 1671) 

s_add_u32 s8, s15, s6 // (op 1671) 


s_lshr_b32 s6, s8, s10 // (op 1671) 

s_and_b32 s8, s10, 2147483648 // (op 1671) 
s_cmp_eq_i32 s8, 0 // (op 1671) 
s_mov_b32 s15, scc // (op 1671) 

s_cmp_eq_i32 s15, 0 // (op 1671) 
s_cselect_b32 s8, s7, s6 // (op 1671) 


s_mul_i32 s6, s8, s14 // (op 1671) 
v_lshl_add_u32 v0, s8, 4, v5 // (op 1671) 


s_sub_u32 s8, s7, s6 // (op 1671) 


v_lshl_add_u32 v3, s8, 4, v7 // (op 1671) 


v_mul_lo_u32 v5, v3, s4 // (op 1671) 

v_add_lshl_u32 v3, v0, v5, 2 // (op 1671) 


s_mul_i32 s19, s4, s23 // (op 1673) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 1673) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 1673) least significant
s_mul_hi_u32 s6, s4, s22 // (op 1673) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1673) most significant: sum
s_add_u32 s19, s19, s6 // (op 1673) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1673) 

s_mov_b32 s56, s20 // (op 1673) reg expression
s_mov_b32 s57, s21 // (op 1673) reg expression
s_mov_b32 s58, s6 // (op 1673) reg expression
s_mov_b32 s59, 131072 // (op 1673) call()

v_and_b32 v0, 48, v1 // (op 2072) 
v_and_b32 v5, 63, v1 // (op 2072) 
v_sub_u32 v7, v5, v0 // (op 2072) 

v_lshrrev_b32 v5, 1, v7 // (op 2072) 
v_add_lshl_u32 v12, v0, v5, 4 // (op 2072) 


v_and_b32 v0, 1, v7 // (op 2072) 

v_lshlrev_b32 v5, 3, v0 // (op 2072) 

v_lshrrev_b32 v0, 6, v1 // (op 2072) 
v_lshl_add_u32 v7, v0, 4, v5 // (op 2072) 


v_add_u32 v0, v12, v7 // (op 2072) 


v_and_b32 v5, 48, v1 // (op 2137) 
v_and_b32 v7, 63, v1 // (op 2137) 
v_sub_u32 v12, v7, v5 // (op 2137) 

v_lshrrev_b32 v7, 1, v12 // (op 2137) 
v_add_lshl_u32 v14, v5, v7, 4 // (op 2137) 


v_and_b32 v5, 1, v12 // (op 2137) 

v_lshlrev_b32 v7, 3, v5 // (op 2137) 

v_add_u32 v5, v14, v7 // (op 2137) 


v_lshrrev_b32 v7, 6, v1 // (op 2203) 
v_and_b32 v12, 15, v1 // (op 2203) 
v_lshl_add_u32 v14, v7, 4, v12 // (op 2203) 


v_and_b32 v7, 48, v1 // (op 2203) 
v_add_u32 v12, v7, v14 // (op 2203) 


v_and_b32 v7, 48, v1 // (op 2269) 
v_and_b32 v14, 15, v1 // (op 2269) 
v_add_u32 v17, v7, v14 // (op 2269) 


v_accvgpr_write a0, 0.00000 // (op 180) call()
v_accvgpr_write a1, 0.00000 // (op 180) call()
v_accvgpr_write a2, 0.00000 // (op 180) call()
v_accvgpr_write a3, 0.00000 // (op 180) call()
s_mul_i32 s15, s16, s50 // (op 1231) 
s_add_u32 s15, s15, s0 // (op 1231) 
s_add_u32 s6, s15, s9 // (op 1231) 

s_mul_hi_u32 s15, s6, s13 // (op 1231) 
s_sub_u32 s7, s6, s15 // (op 1231) 
s_lshr_b32 s8, s7, 1 // (op 1231) 

s_add_u32 s7, s8, s15 // (op 1231) 


s_lshr_b32 s8, s7, s11 // (op 1231) 

s_and_b32 s7, s11, 2147483648 // (op 1231) 
s_cmp_eq_i32 s7, 0 // (op 1231) 
s_mov_b32 s15, scc // (op 1231) 

s_cmp_eq_i32 s15, 0 // (op 1231) 
s_cselect_b32 s7, s6, s8 // (op 1231) 


s_mul_hi_u32 s8, s7, s12 // (op 1231) 
s_sub_u32 s15, s7, s8 // (op 1231) 
s_lshr_b32 s18, s15, 1 // (op 1231) 

s_add_u32 s15, s18, s8 // (op 1231) 


s_lshr_b32 s8, s15, s10 // (op 1231) 

s_and_b32 s15, s10, 2147483648 // (op 1231) 
s_cmp_eq_i32 s15, 0 // (op 1231) 
s_mov_b32 s18, scc // (op 1231) 

s_cmp_eq_i32 s18, 0 // (op 1231) 
s_cselect_b32 s15, s7, s8 // (op 1231) 


s_lshl_b32 s8, s15, 4 // (op 1231) 

s_mov_b32 s19, 0 // (op 1231) convert
s_mov_b32 s18, s8 // (op 1231) convert

s_ashr_i64 s[42:43], s[2:3], 7 // (op 1231) 
s_mul_i32 s8, s7, s42 // (op 1231) 


s_sub_u32 s7, s6, s8 // (op 1231) 


v_lshl_add_u32 v7, s7, 7, v1 // (op 1231) 

v_mul_lo_u32 v21, v7, s25 // (op 1231) most significant: low of low * high
v_mul_hi_u32 v14, v7, s24 // (op 1231) most significant: high of low * low
v_mul_lo_u32 v20, v7, s24 // (op 1231) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 1231) most significant: sum


v_mov_b32 v7, s18 // (op 1231) 
v_mov_b32 v14, s19 // (op 1231) 
v_add_co_u32 v22, s[6:7], v7, v20 // (op 1231) least significant half
v_addc_co_u32 v23, s[6:7], v14, v21, s[6:7] // (op 1231) most significant half





s_mul_i32 s19, s24, s3 // (op 1233) most significant: low of low * high
s_mul_i32 s15, s25, s2 // (op 1233) most significant: low of high * low
s_mul_i32 s18, s24, s2 // (op 1233) least significant
s_mul_hi_u32 s6, s24, s2 // (op 1233) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1233) most significant: sum
s_add_u32 s19, s19, s6 // (op 1233) most significant: sum


s_mov_b32 s60, s26 // (op 1233) reg expression
s_mov_b32 s61, s27 // (op 1233) reg expression
s_mov_b32 s62, s18 // (op 1233) reg expression
s_mov_b32 s63, 131072 // (op 1233) call()

s_mul_i32 s15, s16, s50 // (op 1250) 
s_add_u32 s15, s15, s0 // (op 1250) 
s_add_u32 s6, s15, s9 // (op 1250) 

s_mul_hi_u32 s15, s6, s13 // (op 1250) 
s_sub_u32 s7, s6, s15 // (op 1250) 
s_lshr_b32 s8, s7, 1 // (op 1250) 

s_add_u32 s7, s8, s15 // (op 1250) 


s_lshr_b32 s8, s7, s11 // (op 1250) 

s_and_b32 s7, s11, 2147483648 // (op 1250) 
s_cmp_eq_i32 s7, 0 // (op 1250) 
s_mov_b32 s15, scc // (op 1250) 

s_cmp_eq_i32 s15, 0 // (op 1250) 
s_cselect_b32 s7, s6, s8 // (op 1250) 


s_mul_hi_u32 s8, s7, s12 // (op 1250) 
s_sub_u32 s15, s7, s8 // (op 1250) 
s_lshr_b32 s18, s15, 1 // (op 1250) 

s_add_u32 s15, s18, s8 // (op 1250) 


s_lshr_b32 s8, s15, s10 // (op 1250) 

s_and_b32 s15, s10, 2147483648 // (op 1250) 
s_cmp_eq_i32 s15, 0 // (op 1250) 
s_mov_b32 s18, scc // (op 1250) 

s_cmp_eq_i32 s18, 0 // (op 1250) 
s_cselect_b32 s15, s7, s8 // (op 1250) 


s_lshl_b32 s8, s15, 4 // (op 1250) 

s_mov_b32 s19, 0 // (op 1250) convert
s_mov_b32 s18, s8 // (op 1250) convert

s_ashr_i64 s[42:43], s[2:3], 7 // (op 1250) 
s_mul_i32 s8, s7, s42 // (op 1250) 


s_sub_u32 s7, s6, s8 // (op 1250) 


v_lshl_add_u32 v7, s7, 7, v1 // (op 1250) 

v_mul_lo_u32 v21, v7, s25 // (op 1250) most significant: low of low * high
v_mul_hi_u32 v14, v7, s24 // (op 1250) most significant: high of low * low
v_mul_lo_u32 v20, v7, s24 // (op 1250) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 1250) most significant: sum


v_mov_b32 v7, s18 // (op 1250) 
v_mov_b32 v14, s19 // (op 1250) 
v_add_co_u32 v24, s[6:7], v7, v20 // (op 1250) least significant half
v_addc_co_u32 v25, s[6:7], v14, v21, s[6:7] // (op 1250) most significant half





s_mul_i32 s19, s24, s3 // (op 1252) most significant: low of low * high
s_mul_i32 s15, s25, s2 // (op 1252) most significant: low of high * low
s_mul_i32 s18, s24, s2 // (op 1252) least significant
s_mul_hi_u32 s6, s24, s2 // (op 1252) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1252) most significant: sum
s_add_u32 s19, s19, s6 // (op 1252) most significant: sum


s_mov_b32 s60, s26 // (op 1252) reg expression
s_mov_b32 s61, s27 // (op 1252) reg expression
s_mov_b32 s62, s18 // (op 1252) reg expression
s_mov_b32 s63, 131072 // (op 1252) call()

s_mul_i32 s15, s16, s50 // (op 1359) 
s_add_u32 s15, s15, s0 // (op 1359) 
s_add_u32 s6, s15, s9 // (op 1359) 

s_mul_hi_u32 s15, s6, s13 // (op 1359) 
s_sub_u32 s7, s6, s15 // (op 1359) 
s_lshr_b32 s8, s7, 1 // (op 1359) 

s_add_u32 s7, s8, s15 // (op 1359) 


s_lshr_b32 s8, s7, s11 // (op 1359) 

s_and_b32 s7, s11, 2147483648 // (op 1359) 
s_cmp_eq_i32 s7, 0 // (op 1359) 
s_mov_b32 s15, scc // (op 1359) 

s_cmp_eq_i32 s15, 0 // (op 1359) 
s_cselect_b32 s7, s6, s8 // (op 1359) 


s_mul_hi_u32 s8, s7, s12 // (op 1359) 
s_sub_u32 s15, s7, s8 // (op 1359) 
s_lshr_b32 s18, s15, 1 // (op 1359) 

s_add_u32 s15, s18, s8 // (op 1359) 


s_lshr_b32 s8, s15, s10 // (op 1359) 

s_and_b32 s15, s10, 2147483648 // (op 1359) 
s_cmp_eq_i32 s15, 0 // (op 1359) 
s_mov_b32 s18, scc // (op 1359) 

s_cmp_eq_i32 s18, 0 // (op 1359) 
s_cselect_b32 s15, s7, s8 // (op 1359) 


s_mul_i32 s8, s15, s14 // (op 1359) 

s_sub_u32 s15, s7, s8 // (op 1359) 

s_lshl_b32 s8, s15, 4 // (op 1359) 

s_mov_b32 s19, 0 // (op 1359) convert
s_mov_b32 s18, s8 // (op 1359) convert

s_ashr_i64 s[42:43], s[2:3], 7 // (op 1359) 
s_mul_i32 s8, s7, s42 // (op 1359) 


s_sub_u32 s7, s6, s8 // (op 1359) 


v_lshl_add_u32 v7, s7, 7, v1 // (op 1359) 

v_mul_lo_u32 v21, v7, s29 // (op 1359) most significant: low of low * high
v_mul_hi_u32 v14, v7, s28 // (op 1359) most significant: high of low * low
v_mul_lo_u32 v20, v7, s28 // (op 1359) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 1359) most significant: sum

v_mov_b32 v14, s18 // (op 1359) 
v_mov_b32 v19, s19 // (op 1359) 
v_add_co_u32 v20, s[6:7], v20, v14 // (op 1359) least significant half
v_addc_co_u32 v21, s[6:7], v21, v19, s[6:7] // (op 1359) most significant half





s_mul_i32 s19, s28, s3 // (op 1361) most significant: low of low * high
s_mul_i32 s15, s29, s2 // (op 1361) most significant: low of high * low
s_mul_i32 s18, s28, s2 // (op 1361) least significant
s_mul_hi_u32 s6, s28, s2 // (op 1361) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1361) most significant: sum
s_add_u32 s19, s19, s6 // (op 1361) most significant: sum


s_mov_b32 s64, s30 // (op 1361) reg expression
s_mov_b32 s65, s31 // (op 1361) reg expression
s_mov_b32 s66, s18 // (op 1361) reg expression
s_mov_b32 s67, 131072 // (op 1361) call()

s_mul_i32 s15, s16, s50 // (op 1378) 
s_add_u32 s15, s15, s0 // (op 1378) 
s_add_u32 s6, s15, s9 // (op 1378) 

s_mul_hi_u32 s15, s6, s13 // (op 1378) 
s_sub_u32 s7, s6, s15 // (op 1378) 
s_lshr_b32 s8, s7, 1 // (op 1378) 

s_add_u32 s7, s8, s15 // (op 1378) 


s_lshr_b32 s8, s7, s11 // (op 1378) 

s_and_b32 s7, s11, 2147483648 // (op 1378) 
s_cmp_eq_i32 s7, 0 // (op 1378) 
s_mov_b32 s15, scc // (op 1378) 

s_cmp_eq_i32 s15, 0 // (op 1378) 
s_cselect_b32 s7, s6, s8 // (op 1378) 


s_mul_hi_u32 s8, s7, s12 // (op 1378) 
s_sub_u32 s15, s7, s8 // (op 1378) 
s_lshr_b32 s18, s15, 1 // (op 1378) 

s_add_u32 s15, s18, s8 // (op 1378) 


s_lshr_b32 s8, s15, s10 // (op 1378) 

s_and_b32 s15, s10, 2147483648 // (op 1378) 
s_cmp_eq_i32 s15, 0 // (op 1378) 
s_mov_b32 s18, scc // (op 1378) 

s_cmp_eq_i32 s18, 0 // (op 1378) 
s_cselect_b32 s15, s7, s8 // (op 1378) 


s_mul_i32 s8, s15, s14 // (op 1378) 

s_sub_u32 s15, s7, s8 // (op 1378) 

s_lshl_b32 s8, s15, 4 // (op 1378) 

s_mov_b32 s19, 0 // (op 1378) convert
s_mov_b32 s18, s8 // (op 1378) convert

s_ashr_i64 s[42:43], s[2:3], 7 // (op 1378) 
s_mul_i32 s8, s7, s42 // (op 1378) 


s_sub_u32 s7, s6, s8 // (op 1378) 


v_lshl_add_u32 v7, s7, 7, v1 // (op 1378) 

v_mul_lo_u32 v27, v7, s29 // (op 1378) most significant: low of low * high
v_mul_hi_u32 v14, v7, s28 // (op 1378) most significant: high of low * low
v_mul_lo_u32 v26, v7, s28 // (op 1378) least significant: low of low * low
v_add_u32 v27, v27, v14 // (op 1378) most significant: sum

v_mov_b32 v14, s18 // (op 1378) 
v_mov_b32 v19, s19 // (op 1378) 
v_add_co_u32 v26, s[6:7], v26, v14 // (op 1378) least significant half
v_addc_co_u32 v27, s[6:7], v27, v19, s[6:7] // (op 1378) most significant half





s_mul_i32 s19, s28, s3 // (op 1380) most significant: low of low * high
s_mul_i32 s15, s29, s2 // (op 1380) most significant: low of high * low
s_mul_i32 s18, s28, s2 // (op 1380) least significant
s_mul_hi_u32 s6, s28, s2 // (op 1380) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1380) most significant: sum
s_add_u32 s19, s19, s6 // (op 1380) most significant: sum


s_mov_b32 s64, s30 // (op 1380) reg expression
s_mov_b32 s65, s31 // (op 1380) reg expression
s_mov_b32 s66, s18 // (op 1380) reg expression
s_mov_b32 s67, 131072 // (op 1380) call()

v_and_b32 v7, -16, v1 // (op 1487) 
s_mul_i32 s15, s16, s50 // (op 1487) 
s_add_u32 s15, s15, s0 // (op 1487) 
v_sub_u32 v14, v1, v7 // (op 1487) 

s_add_u32 s6, s15, s9 // (op 1487) 

v_lshrrev_b32 v7, 4, v1 // (op 1487) 
s_mul_hi_u32 s15, s6, s13 // (op 1487) 
s_sub_u32 s7, s6, s15 // (op 1487) 
s_lshr_b32 s8, s7, 1 // (op 1487) 

s_add_u32 s7, s8, s15 // (op 1487) 


s_lshr_b32 s8, s7, s11 // (op 1487) 

s_and_b32 s7, s11, 2147483648 // (op 1487) 
s_cmp_eq_i32 s7, 0 // (op 1487) 
s_mov_b32 s15, scc // (op 1487) 

s_cmp_eq_i32 s15, 0 // (op 1487) 
s_cselect_b32 s7, s6, s8 // (op 1487) 


s_mul_hi_u32 s8, s7, s12 // (op 1487) 
s_sub_u32 s15, s7, s8 // (op 1487) 
s_lshr_b32 s18, s15, 1 // (op 1487) 

s_add_u32 s15, s18, s8 // (op 1487) 


s_lshr_b32 s8, s15, s10 // (op 1487) 

s_and_b32 s15, s10, 2147483648 // (op 1487) 
s_cmp_eq_i32 s15, 0 // (op 1487) 
s_mov_b32 s18, scc // (op 1487) 

s_cmp_eq_i32 s18, 0 // (op 1487) 
s_cselect_b32 s15, s7, s8 // (op 1487) 


v_lshl_add_u32 v19, s15, 4, v14 // (op 1487) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1487) 
s_mul_i32 s8, s7, s18 // (op 1487) 


s_sub_u32 s7, s6, s8 // (op 1487) 


v_lshl_add_u32 v14, s7, 2, v7 // (op 1487) 


v_mul_lo_u32 v7, v14, s32 // (op 1487) 

v_add_u32 v14, v19, v7 // (op 1487) 


s_mul_i32 s19, s32, s37 // (op 1489) most significant: low of low * high
s_mul_i32 s15, s33, s36 // (op 1489) most significant: low of high * low
s_mul_i32 s18, s32, s36 // (op 1489) least significant
s_mul_hi_u32 s6, s32, s36 // (op 1489) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1489) most significant: sum
s_add_u32 s19, s19, s6 // (op 1489) most significant: sum


s_mov_b32 s68, s34 // (op 1489) reg expression
s_mov_b32 s69, s35 // (op 1489) reg expression
s_mov_b32 s70, s18 // (op 1489) reg expression
s_mov_b32 s71, 131072 // (op 1489) call()

v_and_b32 v7, -16, v1 // (op 1506) 
s_mul_i32 s15, s16, s50 // (op 1506) 
s_add_u32 s15, s15, s0 // (op 1506) 
v_sub_u32 v19, v1, v7 // (op 1506) 

s_add_u32 s6, s15, s9 // (op 1506) 

v_lshrrev_b32 v7, 4, v1 // (op 1506) 
s_mul_hi_u32 s15, s6, s13 // (op 1506) 
s_sub_u32 s7, s6, s15 // (op 1506) 
s_lshr_b32 s8, s7, 1 // (op 1506) 

s_add_u32 s7, s8, s15 // (op 1506) 


s_lshr_b32 s8, s7, s11 // (op 1506) 

s_and_b32 s7, s11, 2147483648 // (op 1506) 
s_cmp_eq_i32 s7, 0 // (op 1506) 
s_mov_b32 s15, scc // (op 1506) 

s_cmp_eq_i32 s15, 0 // (op 1506) 
s_cselect_b32 s7, s6, s8 // (op 1506) 


s_mul_hi_u32 s8, s7, s12 // (op 1506) 
s_sub_u32 s15, s7, s8 // (op 1506) 
s_lshr_b32 s18, s15, 1 // (op 1506) 

s_add_u32 s15, s18, s8 // (op 1506) 


s_lshr_b32 s8, s15, s10 // (op 1506) 

s_and_b32 s15, s10, 2147483648 // (op 1506) 
s_cmp_eq_i32 s15, 0 // (op 1506) 
s_mov_b32 s18, scc // (op 1506) 

s_cmp_eq_i32 s18, 0 // (op 1506) 
s_cselect_b32 s15, s7, s8 // (op 1506) 


v_lshl_add_u32 v28, s15, 4, v19 // (op 1506) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1506) 
s_mul_i32 s8, s7, s18 // (op 1506) 


s_sub_u32 s7, s6, s8 // (op 1506) 


v_lshl_add_u32 v19, s7, 2, v7 // (op 1506) 


v_mul_lo_u32 v7, v19, s32 // (op 1506) 

v_add_u32 v19, v28, v7 // (op 1506) 


s_mul_i32 s19, s32, s37 // (op 1508) most significant: low of low * high
s_mul_i32 s15, s33, s36 // (op 1508) most significant: low of high * low
s_mul_i32 s18, s32, s36 // (op 1508) least significant
s_mul_hi_u32 s6, s32, s36 // (op 1508) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1508) most significant: sum
s_add_u32 s19, s19, s6 // (op 1508) most significant: sum


s_mov_b32 s68, s34 // (op 1508) reg expression
s_mov_b32 s69, s35 // (op 1508) reg expression
s_mov_b32 s70, s18 // (op 1508) reg expression
s_mov_b32 s71, 131072 // (op 1508) call()

v_and_b32 v7, -16, v1 // (op 1615) 
s_mul_i32 s15, s16, s50 // (op 1615) 
s_add_u32 s15, s15, s0 // (op 1615) 
v_sub_u32 v28, v1, v7 // (op 1615) 

s_add_u32 s6, s15, s9 // (op 1615) 

v_lshrrev_b32 v7, 4, v1 // (op 1615) 
s_mul_hi_u32 s15, s6, s13 // (op 1615) 
s_sub_u32 s7, s6, s15 // (op 1615) 
s_lshr_b32 s8, s7, 1 // (op 1615) 

s_add_u32 s7, s8, s15 // (op 1615) 


s_lshr_b32 s8, s7, s11 // (op 1615) 

s_and_b32 s7, s11, 2147483648 // (op 1615) 
s_cmp_eq_i32 s7, 0 // (op 1615) 
s_mov_b32 s15, scc // (op 1615) 

s_cmp_eq_i32 s15, 0 // (op 1615) 
s_cselect_b32 s7, s6, s8 // (op 1615) 


s_mul_hi_u32 s8, s7, s12 // (op 1615) 
s_sub_u32 s15, s7, s8 // (op 1615) 
s_lshr_b32 s18, s15, 1 // (op 1615) 

s_add_u32 s15, s18, s8 // (op 1615) 


s_lshr_b32 s8, s15, s10 // (op 1615) 

s_and_b32 s15, s10, 2147483648 // (op 1615) 
s_cmp_eq_i32 s15, 0 // (op 1615) 
s_mov_b32 s18, scc // (op 1615) 

s_cmp_eq_i32 s18, 0 // (op 1615) 
s_cselect_b32 s15, s7, s8 // (op 1615) 


s_mul_i32 s8, s15, s14 // (op 1615) 

s_sub_u32 s15, s7, s8 // (op 1615) 

v_lshl_add_u32 v29, s15, 4, v28 // (op 1615) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1615) 
s_mul_i32 s8, s7, s18 // (op 1615) 


s_sub_u32 s7, s6, s8 // (op 1615) 


v_lshl_add_u32 v28, s7, 2, v7 // (op 1615) 


v_mul_lo_u32 v7, v28, s38 // (op 1615) 
v_add_u32 v7, v7, v29 // (op 1615) 


s_mul_i32 s19, s38, s37 // (op 1617) most significant: low of low * high
s_mul_i32 s15, s39, s36 // (op 1617) most significant: low of high * low
s_mul_i32 s18, s38, s36 // (op 1617) least significant
s_mul_hi_u32 s6, s38, s36 // (op 1617) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1617) most significant: sum
s_add_u32 s19, s19, s6 // (op 1617) most significant: sum


s_mov_b32 s72, s40 // (op 1617) reg expression
s_mov_b32 s73, s41 // (op 1617) reg expression
s_mov_b32 s74, s18 // (op 1617) reg expression
s_mov_b32 s75, 131072 // (op 1617) call()

v_and_b32 v28, -16, v1 // (op 1634) 
s_mul_i32 s15, s16, s50 // (op 1634) 
s_add_u32 s15, s15, s0 // (op 1634) 
v_sub_u32 v29, v1, v28 // (op 1634) 

s_add_u32 s6, s15, s9 // (op 1634) 

v_lshrrev_b32 v28, 4, v1 // (op 1634) 
s_mul_hi_u32 s15, s6, s13 // (op 1634) 
s_sub_u32 s7, s6, s15 // (op 1634) 
s_lshr_b32 s8, s7, 1 // (op 1634) 

s_add_u32 s7, s8, s15 // (op 1634) 


s_lshr_b32 s8, s7, s11 // (op 1634) 

s_and_b32 s7, s11, 2147483648 // (op 1634) 
s_cmp_eq_i32 s7, 0 // (op 1634) 
s_mov_b32 s15, scc // (op 1634) 

s_cmp_eq_i32 s15, 0 // (op 1634) 
s_cselect_b32 s7, s6, s8 // (op 1634) 


s_mul_hi_u32 s8, s7, s12 // (op 1634) 
s_sub_u32 s15, s7, s8 // (op 1634) 
s_lshr_b32 s18, s15, 1 // (op 1634) 

s_add_u32 s15, s18, s8 // (op 1634) 


s_lshr_b32 s8, s15, s10 // (op 1634) 

s_and_b32 s15, s10, 2147483648 // (op 1634) 
s_cmp_eq_i32 s15, 0 // (op 1634) 
s_mov_b32 s18, scc // (op 1634) 

s_cmp_eq_i32 s18, 0 // (op 1634) 
s_cselect_b32 s15, s7, s8 // (op 1634) 


s_mul_i32 s8, s15, s14 // (op 1634) 

s_sub_u32 s15, s7, s8 // (op 1634) 

v_lshl_add_u32 v30, s15, 4, v29 // (op 1634) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1634) 
s_mul_i32 s8, s7, s18 // (op 1634) 


s_sub_u32 s7, s6, s8 // (op 1634) 


v_lshl_add_u32 v29, s7, 2, v28 // (op 1634) 


v_mul_lo_u32 v28, v29, s38 // (op 1634) 
v_add_u32 v28, v28, v30 // (op 1634) 


s_mul_i32 s19, s38, s37 // (op 1636) most significant: low of low * high
s_mul_i32 s15, s39, s36 // (op 1636) most significant: low of high * low
s_mul_i32 s18, s38, s36 // (op 1636) least significant
s_mul_hi_u32 s6, s38, s36 // (op 1636) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1636) most significant: sum
s_add_u32 s19, s19, s6 // (op 1636) most significant: sum


s_mov_b32 s72, s40 // (op 1636) reg expression
s_mov_b32 s73, s41 // (op 1636) reg expression
s_mov_b32 s74, s18 // (op 1636) reg expression
s_mov_b32 s75, 131072 // (op 1636) call()

v_lshlrev_b32 v29, 4, v1 // (op 2113) 
s_nop 0
v_readfirstlane_b32 s15, v29 // (op 2113) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2748) 

s_lshl_b64 s[18:19], s[28:29], 6 // (op 1111) 
v_mov_b64 v[30:31], s[18:19] // (op 1111) convert


v_mov_b64 v[32:33], v[20:21] // (op 1111) reg expression
s_mov_b32 m0, s15 // (op 1111) reg expression
buffer_load_dwordx4 v32, s[64:67], 0 offen offset: 0 lds // (op 1111) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 1111) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 1111) most significant half

s_add_u32 m0, m0, 1024 // (op 1111) 
buffer_load_dwordx4 v32, s[64:67], 0 offen offset: 0 lds // (op 1111) Load value direct to lds





v_mov_b32 v29, v7 // (op 946) reg expression

buffer_load_ubyte v30, v29, s[72:75], 0 offen offset: 0 // (op 946) Load value

v_lshlrev_b32 v29, 4, v1 // (op 2048) 
s_nop 0
v_readfirstlane_b32 s15, v29 // (op 2048) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2729) 

s_lshl_b64 s[18:19], s[24:25], 6 // (op 1116) 
v_mov_b64 v[32:33], s[18:19] // (op 1116) convert


v_mov_b64 v[34:35], v[22:23] // (op 1116) reg expression
s_add_u32 s6, 2048, s15 // (op 1116) 
s_mov_b32 m0, s6 // (op 1116) reg expression
buffer_load_dwordx4 v34, s[60:63], 0 offen offset: 0 lds // (op 1116) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 1116) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 1116) most significant half

s_add_u32 m0, m0, 1024 // (op 1116) 
buffer_load_dwordx4 v34, s[60:63], 0 offen offset: 0 lds // (op 1116) Load value direct to lds






v_mov_b32 v29, v14 // (op 952) reg expression

buffer_load_ubyte v31, v29, s[68:71], 0 offen offset: 0 // (op 952) Load value

v_lshlrev_b32 v29, 4, v1 // (op 2370) 
s_nop 0
v_readfirstlane_b32 s15, v29 // (op 2370) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2818) 

s_lshl_b64 s[18:19], s[28:29], 6 // (op 1121) 
v_mov_b64 v[32:33], s[18:19] // (op 1121) convert


s_lshl_b64 s[18:19], s[28:29], 7 // (op 1121) 
v_mov_b32 v29, s18 // (op 1121) 
v_mov_b32 v34, s19 // (op 1121) 
v_add_co_u32 v36, s[6:7], v20, v29 // (op 1121) least significant half
v_addc_co_u32 v37, s[6:7], v21, v34, s[6:7] // (op 1121) most significant half




s_add_u32 s6, 4096, s15 // (op 1121) 
s_nop 0
s_mov_b32 m0, s6 // (op 1121) reg expression
buffer_load_dwordx4 v36, s[64:67], 0 offen offset: 0 lds // (op 1121) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v32 // (op 1121) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1121) most significant half

s_add_u32 m0, m0, 1024 // (op 1121) 
buffer_load_dwordx4 v36, s[64:67], 0 offen offset: 0 lds // (op 1121) Load value direct to lds







s_lshl_b32 s15, s38, 2 // (op 958) 
v_add_u32 v29, v7, s15 // (op 958) 


buffer_load_ubyte v20, v29, s[72:75], 0 offen offset: 0 // (op 958) Load value


v_lshlrev_b32 v7, 4, v1 // (op 2331) 
s_nop 0
v_readfirstlane_b32 s15, v7 // (op 2331) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 2808) 

s_lshl_b64 s[18:19], s[24:25], 6 // (op 1123) 
v_mov_b64 v[32:33], s[18:19] // (op 1123) convert


s_lshl_b64 s[18:19], s[24:25], 7 // (op 1123) 
v_mov_b32 v7, s18 // (op 1123) 
v_mov_b32 v21, s19 // (op 1123) 
v_add_co_u32 v34, s[6:7], v22, v7 // (op 1123) least significant half
v_addc_co_u32 v35, s[6:7], v23, v21, s[6:7] // (op 1123) most significant half




s_add_u32 s6, 6144, s15 // (op 1123) 
s_nop 0
s_mov_b32 m0, s6 // (op 1123) reg expression
buffer_load_dwordx4 v34, s[60:63], 0 offen offset: 0 lds // (op 1123) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 1123) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 1123) most significant half

s_add_u32 m0, m0, 1024 // (op 1123) 
buffer_load_dwordx4 v34, s[60:63], 0 offen offset: 0 lds // (op 1123) Load value direct to lds







s_lshl_b32 s15, s32, 2 // (op 964) 
v_add_u32 v7, v14, s15 // (op 964) 


buffer_load_ubyte v29, v7, s[68:71], 0 offen offset: 0 // (op 964) Load value


v_and_b32 v7, -16, v1 // (op 2245) 
v_sub_u32 v14, v1, v7 // (op 2245) 

v_and_b32 v7, 4294967280, v1 // (op 2245) 
v_add_u32 v21, v7, v14 // (op 2245) 


s_barrier  // (op 2788) 



s_waitcnt vmcnt(9)
ds_write_b8 v21, v30 offset:8192 // (op 967) Store local data 

v_and_b32 v7, -16, v1 // (op 2179) 
v_sub_u32 v14, v1, v7 // (op 2179) 

v_and_b32 v7, 4294967280, v1 // (op 2179) 
v_add_u32 v21, v7, v14 // (op 2179) 


s_barrier  // (op 2768) 



s_waitcnt vmcnt(6)
ds_write_b8 v21, v31 offset:8256 // (op 969) Store local data 

s_barrier  // (op 2745) 



ds_read_b64_tr_b8 v[32:33], v5 // (op 986) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v5 offset:128 // (op 986) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v5 offset:1024 // (op 986) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v5 offset:1152 // (op 986) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2785) 


ds_read_u8 v7, v17 offset:8192 // (op 992) Load local data 
s_barrier  // (op 2726) 



ds_read_b64_tr_b8 v[40:41], v0 offset:2048 // (op 983) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:2176 // (op 983) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:3072 // (op 983) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:3200 // (op 983) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2765) 


ds_read_u8 v14, v12 offset:8256 // (op 989) Load local data 
v_lshlrev_b32 v21, 4, v1 // (op 2020) 
s_nop 0
v_readfirstlane_b32 s15, v21 // (op 2020) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v21, 4, v1 // (op 2085) 
s_nop 0
v_readfirstlane_b32 s6, v21 // (op 2085) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -16, v1 // (op 2150) 
v_sub_u32 v22, v1, v21 // (op 2150) 

v_and_b32 v21, 4294967280, v1 // (op 2150) 
v_add_u32 v23, v21, v22 // (op 2150) 


v_and_b32 v21, -16, v1 // (op 2216) 
v_sub_u32 v22, v1, v21 // (op 2216) 

v_and_b32 v21, 4294967280, v1 // (op 2216) 
v_add_u32 v48, v21, v22 // (op 2216) 


v_lshlrev_b32 v21, 4, v1 // (op 2316) 
s_nop 0
v_readfirstlane_b32 s7, v21 // (op 2316) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v21, 4, v1 // (op 2355) 
s_nop 0
v_readfirstlane_b32 s8, v21 // (op 2355) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -16, v1 // (op 2394) 
v_sub_u32 v22, v1, v21 // (op 2394) 

v_and_b32 v21, 4294967280, v1 // (op 2394) 
v_add_u32 v49, v21, v22 // (op 2394) 


v_and_b32 v21, -16, v1 // (op 2418) 
v_sub_u32 v22, v1, v21 // (op 2418) 

v_and_b32 v21, 4294967280, v1 // (op 2418) 
v_add_u32 v50, v21, v22 // (op 2418) 


s_mov_b32 s1, 0 // (op 182) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 181) 
s_mov_b32 s42, s18 // (op 181) convert

s_and_b32 s18, s42, 4294967294 // (op 181) 

s_cmp_lt_u32 s1, s18 // (op 181) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_181 // (op 181) Condition: Top (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_181 if false)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoop_181:
s_add_u32 s1, s1, 2 // (op 183) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[32:39], a[0:3], v14, v7 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 194) 
s_barrier  // (op 2751) 

s_lshl_b64 s[18:19], s[28:29], 6 // (op 1046) 
v_mov_b64 v[52:53], s[18:19] // (op 1046) convert


v_mov_b64 v[54:55], v[26:27] // (op 1046) 
s_lshl_b64 s[18:19], s[28:29], 7 // (op 1046) 
s_lshl_b64 s[42:43], s[18:19], 1 // (op 1046) 

v_mov_b32 v51, s42 // (op 1046) 
v_mov_b32 v21, s43 // (op 1046) 
v_add_co_u32 v56, s[18:19], v54, v51 // (op 1046) least significant half
v_addc_co_u32 v57, s[18:19], v55, v21, s[18:19] // (op 1046) most significant half




s_mov_b32 m0, s6 // (op 1046) reg expression
buffer_load_dwordx4 v56, s[64:67], 0 offen offset: 0 lds // (op 1046) Load value direct to lds
v_add_co_u32 v56, s[18:19], v56, v52 // (op 1046) least significant half
v_addc_co_u32 v57, s[18:19], 0, v57, s[18:19] // (op 1046) most significant half

s_add_u32 m0, m0, 1024 // (op 1046) 
buffer_load_dwordx4 v56, s[64:67], 0 offen offset: 0 lds // (op 1046) Load value direct to lds




v_mov_b32 v21, v28 // (op 200) 
s_lshl_b32 s18, s38, 3 // (op 200) 
v_add_u32 v22, v21, s18 // (op 200) 


buffer_load_ubyte v30, v22, s[72:75], 0 offen offset: 0 // (op 200) Load value

s_barrier  // (op 2732) 

s_lshl_b64 s[18:19], s[24:25], 6 // (op 1041) 
v_mov_b64 v[52:53], s[18:19] // (op 1041) convert


v_mov_b64 v[56:57], v[24:25] // (op 1041) 
s_lshl_b64 s[18:19], s[24:25], 7 // (op 1041) 
s_lshl_b64 s[42:43], s[18:19], 1 // (op 1041) 

v_mov_b32 v22, s42 // (op 1041) 
v_mov_b32 v51, s43 // (op 1041) 
v_add_co_u32 v58, s[18:19], v56, v22 // (op 1041) least significant half
v_addc_co_u32 v59, s[18:19], v57, v51, s[18:19] // (op 1041) most significant half




s_add_u32 s18, 2048, s15 // (op 1041) 
s_nop 0
s_mov_b32 m0, s18 // (op 1041) reg expression
buffer_load_dwordx4 v58, s[60:63], 0 offen offset: 0 lds // (op 1041) Load value direct to lds
v_add_co_u32 v58, s[42:43], v58, v52 // (op 1041) least significant half
v_addc_co_u32 v59, s[42:43], 0, v59, s[42:43] // (op 1041) most significant half

s_add_u32 m0, m0, 1024 // (op 1041) 
buffer_load_dwordx4 v58, s[60:63], 0 offen offset: 0 lds // (op 1041) Load value direct to lds





v_mov_b32 v22, v19 // (op 199) 
s_lshl_b32 s18, s32, 3 // (op 199) 
v_add_u32 v51, v22, s18 // (op 199) 


buffer_load_ubyte v31, v51, s[68:71], 0 offen offset: 0 // (op 199) Load value

s_barrier  // (op 2833) 



s_waitcnt vmcnt(9)
ds_write_b8 v50, v20 offset:8320 // (op 533) Store local data 
s_barrier  // (op 2827) 



s_waitcnt vmcnt(6)
ds_write_b8 v49, v29 offset:8384 // (op 532) Store local data 
s_barrier  // (op 2814) 



ds_read_b64_tr_b8 v[32:33], v10 offset:4096 // (op 525) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v10 offset:4224 // (op 525) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v10 offset:5120 // (op 525) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v10 offset:5248 // (op 525) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2830) 


ds_read_u8 v7, v15 offset:8320 // (op 529) Load local data 
s_barrier  // (op 2805) 
s_barrier  // (op 2963) 



ds_read_b64_tr_b8 v[40:41], v8 offset:6144 // (op 523) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v8 offset:6272 // (op 523) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v8 offset:7168 // (op 523) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v8 offset:7296 // (op 523) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2824) 


ds_read_u8 v14, v13 offset:8384 // (op 527) Load local data 
s_barrier  // (op 2957) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[32:39], a[0:3], v14, v7 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 530) 
s_barrier  // (op 2821) 

s_lshl_b64 s[18:19], s[28:29], 6 // (op 1069) 
v_mov_b64 v[52:53], s[18:19] // (op 1069) convert


s_lshl_b64 s[18:19], s[28:29], 7 // (op 1069) 
s_mul_i32 s43, 3, s19 // (op 1069) most significant: low of low * high
s_mul_i32 s42, 3, s18 // (op 1069) least significant
s_mul_hi_u32 s44, 3, s18 // (op 1069) most significant: high of low * low
s_add_u32 s43, s43, s44 // (op 1069) most significant: sum


v_mov_b32 v51, s42 // (op 1069) 
v_mov_b32 v58, s43 // (op 1069) 
v_add_co_u32 v60, s[18:19], v54, v51 // (op 1069) least significant half
v_addc_co_u32 v61, s[18:19], v55, v58, s[18:19] // (op 1069) most significant half




s_add_u32 s18, 4096, s8 // (op 1069) 
s_nop 0
s_mov_b32 m0, s18 // (op 1069) reg expression
buffer_load_dwordx4 v60, s[64:67], 0 offen offset: 0 lds // (op 1069) Load value direct to lds
v_add_co_u32 v60, s[42:43], v60, v52 // (op 1069) least significant half
v_addc_co_u32 v61, s[42:43], 0, v61, s[42:43] // (op 1069) most significant half

s_add_u32 m0, m0, 1024 // (op 1069) 
buffer_load_dwordx4 v60, s[64:67], 0 offen offset: 0 lds // (op 1069) Load value direct to lds






s_lshl_b32 s18, s38, 2 // (op 536) 
s_mul_i32 s19, 3, s18 // (op 536) 

v_add_u32 v51, v21, s19 // (op 536) 


buffer_load_ubyte v20, v51, s[72:75], 0 offen offset: 0 // (op 536) Load value

s_lshl_b64 s[18:19], s[28:29], 7 // (op 1370) 

s_lshl_b64 s[42:43], s[18:19], 1 // (op 1370) 

s_barrier  // (op 2811) 
s_lshl_b32 s18, s38, 3 // (op 1626) 

v_add_u32 v28, v28, s18 // (op 1626) 

v_mov_b32 v21, s42 // (op 1370) 
v_mov_b32 v51, s43 // (op 1370) 
s_lshl_b64 s[18:19], s[24:25], 6 // (op 1064) 
v_mov_b64 v[52:53], s[18:19] // (op 1064) convert


s_lshl_b64 s[18:19], s[24:25], 7 // (op 1064) 
s_mul_i32 s45, 3, s19 // (op 1064) most significant: low of low * high
s_mul_i32 s44, 3, s18 // (op 1064) least significant
s_mul_hi_u32 s52, 3, s18 // (op 1064) most significant: high of low * low
s_add_u32 s45, s45, s52 // (op 1064) most significant: sum


v_mov_b32 v54, s44 // (op 1064) 
v_mov_b32 v55, s45 // (op 1064) 
v_add_co_u32 v26, s[18:19], v26, v21 // (op 1370) least significant half
v_addc_co_u32 v27, s[18:19], v27, v51, s[18:19] // (op 1370) most significant half




v_add_co_u32 v58, s[18:19], v56, v54 // (op 1064) least significant half
v_addc_co_u32 v59, s[18:19], v57, v55, s[18:19] // (op 1064) most significant half




s_add_u32 s18, 6144, s7 // (op 1064) 
s_nop 0
s_mov_b32 m0, s18 // (op 1064) reg expression
buffer_load_dwordx4 v58, s[60:63], 0 offen offset: 0 lds // (op 1064) Load value direct to lds
v_add_co_u32 v58, s[42:43], v58, v52 // (op 1064) least significant half
v_addc_co_u32 v59, s[42:43], 0, v59, s[42:43] // (op 1064) most significant half

s_add_u32 m0, m0, 1024 // (op 1064) 
buffer_load_dwordx4 v58, s[60:63], 0 offen offset: 0 lds // (op 1064) Load value direct to lds






s_lshl_b32 s18, s32, 2 // (op 535) 
s_mul_i32 s19, 3, s18 // (op 535) 

v_add_u32 v21, v22, s19 // (op 535) 


buffer_load_ubyte v29, v21, s[68:71], 0 offen offset: 0 // (op 535) Load value

s_lshl_b64 s[18:19], s[24:25], 7 // (op 1242) 

s_lshl_b64 s[42:43], s[18:19], 1 // (op 1242) 

s_barrier  // (op 2791) 
s_lshl_b32 s18, s32, 3 // (op 1498) 



v_add_u32 v19, v19, s18 // (op 1498) 

v_mov_b32 v21, s42 // (op 1242) 
v_mov_b32 v22, s43 // (op 1242) 
v_add_co_u32 v24, s[18:19], v24, v21 // (op 1242) least significant half
v_addc_co_u32 v25, s[18:19], v25, v22, s[18:19] // (op 1242) most significant half




s_waitcnt vmcnt(9)
ds_write_b8 v48, v30 offset:8192 // (op 197) Store local data 
s_barrier  // (op 2771) 



s_waitcnt vmcnt(6)
ds_write_b8 v23, v31 offset:8256 // (op 196) Store local data 
s_barrier  // (op 2757) 



ds_read_b64_tr_b8 v[32:33], v6 // (op 189) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v6 offset:128 // (op 189) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v6 offset:1024 // (op 189) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v6 offset:1152 // (op 189) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2798) 


ds_read_u8 v7, v11 offset:8192 // (op 193) Load local data 
s_barrier  // (op 2738) 



ds_read_b64_tr_b8 v[40:41], v4 offset:2048 // (op 187) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:2176 // (op 187) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:3072 // (op 187) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v4 offset:3200 // (op 187) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2778) 


ds_read_u8 v14, v9 offset:8256 // (op 191) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 181) 
s_mov_b32 s42, s18 // (op 181) convert

s_and_b32 s18, s42, 4294967294 // (op 181) 

s_cmp_lt_u32 s1, s18 // (op 181) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoop_181 // (op 181) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoop_181 if true)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_181:














s_barrier  // (op 2887) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1136) 
s_and_b32 s15, s18, 4294967294 // (op 1136) 
s_add_u32 s6, s0, s15 // (op 1136) 

s_mul_i32 s15, s16, s50 // (op 1136) 
s_add_u32 s15, s15, s6 // (op 1136) 

s_add_u32 s6, s15, s9 // (op 1136) 

s_mul_hi_u32 s15, s6, s13 // (op 1136) 
s_sub_u32 s7, s6, s15 // (op 1136) 
s_lshr_b32 s8, s7, 1 // (op 1136) 

s_add_u32 s7, s8, s15 // (op 1136) 


s_lshr_b32 s8, s7, s11 // (op 1136) 

s_and_b32 s7, s11, 2147483648 // (op 1136) 
s_cmp_eq_i32 s7, 0 // (op 1136) 
s_mov_b32 s15, scc // (op 1136) 

s_cmp_eq_i32 s15, 0 // (op 1136) 
s_cselect_b32 s7, s6, s8 // (op 1136) 


s_mul_hi_u32 s8, s7, s12 // (op 1136) 
s_sub_u32 s15, s7, s8 // (op 1136) 
s_lshr_b32 s42, s15, 1 // (op 1136) 

s_add_u32 s15, s42, s8 // (op 1136) 


s_lshr_b32 s8, s15, s10 // (op 1136) 

s_and_b32 s15, s10, 2147483648 // (op 1136) 
s_cmp_eq_i32 s15, 0 // (op 1136) 
s_mov_b32 s42, scc // (op 1136) 

s_cmp_eq_i32 s42, 0 // (op 1136) 
s_cselect_b32 s15, s7, s8 // (op 1136) 


s_lshl_b32 s8, s15, 4 // (op 1136) 

s_mov_b32 s43, 0 // (op 1136) convert
s_mov_b32 s42, s8 // (op 1136) convert

s_mul_i32 s8, s7, s18 // (op 1136) 


s_sub_u32 s7, s6, s8 // (op 1136) 


v_lshl_add_u32 v19, s7, 7, v1 // (op 1136) 

v_mul_lo_u32 v31, v19, s25 // (op 1136) most significant: low of low * high
v_mul_hi_u32 v21, v19, s24 // (op 1136) most significant: high of low * low
v_mul_lo_u32 v30, v19, s24 // (op 1136) least significant: low of low * low
v_add_u32 v31, v31, v21 // (op 1136) most significant: sum


v_mov_b32 v19, s42 // (op 1136) 
v_mov_b32 v21, s43 // (op 1136) 
v_add_co_u32 v22, s[18:19], v19, v30 // (op 1136) least significant half
v_addc_co_u32 v23, s[18:19], v21, v31, s[18:19] // (op 1136) most significant half





s_mul_i32 s19, s24, s3 // (op 1138) most significant: low of low * high
s_mul_i32 s15, s25, s2 // (op 1138) most significant: low of high * low
s_mul_i32 s18, s24, s2 // (op 1138) least significant
s_mul_hi_u32 s6, s24, s2 // (op 1138) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1138) most significant: sum
s_add_u32 s19, s19, s6 // (op 1138) most significant: sum


s_mov_b32 s60, s26 // (op 1138) reg expression
s_mov_b32 s61, s27 // (op 1138) reg expression
s_mov_b32 s62, s18 // (op 1138) reg expression
s_mov_b32 s63, 131072 // (op 1138) call()

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1272) 
s_and_b32 s15, s18, 4294967294 // (op 1272) 
s_add_u32 s6, s0, s15 // (op 1272) 

s_mul_i32 s15, s16, s50 // (op 1272) 
s_add_u32 s15, s15, s6 // (op 1272) 

s_add_u32 s6, s15, s9 // (op 1272) 

s_mul_hi_u32 s15, s6, s13 // (op 1272) 
s_sub_u32 s7, s6, s15 // (op 1272) 
s_lshr_b32 s8, s7, 1 // (op 1272) 

s_add_u32 s7, s8, s15 // (op 1272) 


s_lshr_b32 s8, s7, s11 // (op 1272) 

s_and_b32 s7, s11, 2147483648 // (op 1272) 
s_cmp_eq_i32 s7, 0 // (op 1272) 
s_mov_b32 s15, scc // (op 1272) 

s_cmp_eq_i32 s15, 0 // (op 1272) 
s_cselect_b32 s7, s6, s8 // (op 1272) 


s_mul_hi_u32 s8, s7, s12 // (op 1272) 
s_sub_u32 s15, s7, s8 // (op 1272) 
s_lshr_b32 s42, s15, 1 // (op 1272) 

s_add_u32 s15, s42, s8 // (op 1272) 


s_lshr_b32 s8, s15, s10 // (op 1272) 

s_and_b32 s15, s10, 2147483648 // (op 1272) 
s_cmp_eq_i32 s15, 0 // (op 1272) 
s_mov_b32 s42, scc // (op 1272) 

s_cmp_eq_i32 s42, 0 // (op 1272) 
s_cselect_b32 s15, s7, s8 // (op 1272) 


s_mul_i32 s8, s15, s14 // (op 1272) 

s_sub_u32 s15, s7, s8 // (op 1272) 

s_lshl_b32 s8, s15, 4 // (op 1272) 

s_mov_b32 s43, 0 // (op 1272) convert
s_mov_b32 s42, s8 // (op 1272) convert

s_mul_i32 s8, s7, s18 // (op 1272) 


s_sub_u32 s7, s6, s8 // (op 1272) 


v_lshl_add_u32 v19, s7, 7, v1 // (op 1272) 

v_mul_lo_u32 v31, v19, s29 // (op 1272) most significant: low of low * high
v_mul_hi_u32 v21, v19, s28 // (op 1272) most significant: high of low * low
v_mul_lo_u32 v30, v19, s28 // (op 1272) least significant: low of low * low
v_add_u32 v31, v31, v21 // (op 1272) most significant: sum

v_mov_b32 v21, s42 // (op 1272) 
v_mov_b32 v24, s43 // (op 1272) 
v_add_co_u32 v30, s[18:19], v30, v21 // (op 1272) least significant half
v_addc_co_u32 v31, s[18:19], v31, v24, s[18:19] // (op 1272) most significant half





s_mul_i32 s19, s28, s3 // (op 1274) most significant: low of low * high
s_mul_i32 s15, s29, s2 // (op 1274) most significant: low of high * low
s_mul_i32 s18, s28, s2 // (op 1274) least significant
s_mul_hi_u32 s6, s28, s2 // (op 1274) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1274) most significant: sum
s_add_u32 s19, s19, s6 // (op 1274) most significant: sum


s_mov_b32 s64, s30 // (op 1274) reg expression
s_mov_b32 s65, s31 // (op 1274) reg expression
s_mov_b32 s66, s18 // (op 1274) reg expression
s_mov_b32 s67, 131072 // (op 1274) call()

v_and_b32 v19, -16, v1 // (op 1400) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1400) 
v_sub_u32 v21, v1, v19 // (op 1400) 

s_and_b32 s15, s18, 4294967294 // (op 1400) 
v_lshrrev_b32 v19, 4, v1 // (op 1400) 
s_add_u32 s6, s0, s15 // (op 1400) 

s_mul_i32 s15, s16, s50 // (op 1400) 
s_add_u32 s15, s15, s6 // (op 1400) 

s_add_u32 s6, s15, s9 // (op 1400) 

s_mul_hi_u32 s15, s6, s13 // (op 1400) 
s_sub_u32 s7, s6, s15 // (op 1400) 
s_lshr_b32 s8, s7, 1 // (op 1400) 

s_add_u32 s7, s8, s15 // (op 1400) 


s_lshr_b32 s8, s7, s11 // (op 1400) 

s_and_b32 s7, s11, 2147483648 // (op 1400) 
s_cmp_eq_i32 s7, 0 // (op 1400) 
s_mov_b32 s15, scc // (op 1400) 

s_cmp_eq_i32 s15, 0 // (op 1400) 
s_cselect_b32 s7, s6, s8 // (op 1400) 


s_mul_hi_u32 s8, s7, s12 // (op 1400) 
s_sub_u32 s15, s7, s8 // (op 1400) 
s_lshr_b32 s42, s15, 1 // (op 1400) 

s_add_u32 s15, s42, s8 // (op 1400) 


s_lshr_b32 s8, s15, s10 // (op 1400) 

s_and_b32 s15, s10, 2147483648 // (op 1400) 
s_cmp_eq_i32 s15, 0 // (op 1400) 
s_mov_b32 s42, scc // (op 1400) 

s_cmp_eq_i32 s42, 0 // (op 1400) 
s_cselect_b32 s15, s7, s8 // (op 1400) 


s_mul_i32 s8, s7, s18 // (op 1400) 


v_lshl_add_u32 v24, s15, 4, v21 // (op 1400) 


s_sub_u32 s7, s6, s8 // (op 1400) 


v_lshl_add_u32 v21, s7, 2, v19 // (op 1400) 


v_mul_lo_u32 v19, v21, s32 // (op 1400) 

v_add_u32 v21, v24, v19 // (op 1400) 


s_mul_i32 s19, s32, s37 // (op 1402) most significant: low of low * high
s_mul_i32 s15, s33, s36 // (op 1402) most significant: low of high * low
s_mul_i32 s18, s32, s36 // (op 1402) least significant
s_mul_hi_u32 s6, s32, s36 // (op 1402) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1402) most significant: sum
s_add_u32 s19, s19, s6 // (op 1402) most significant: sum


s_mov_b32 s68, s34 // (op 1402) reg expression
s_mov_b32 s69, s35 // (op 1402) reg expression
s_mov_b32 s70, s18 // (op 1402) reg expression
s_mov_b32 s71, 131072 // (op 1402) call()

v_and_b32 v19, -16, v1 // (op 1528) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1528) 
v_sub_u32 v24, v1, v19 // (op 1528) 

s_and_b32 s15, s18, 4294967294 // (op 1528) 
v_lshrrev_b32 v19, 4, v1 // (op 1528) 
s_add_u32 s6, s0, s15 // (op 1528) 

s_mul_i32 s15, s16, s50 // (op 1528) 
s_add_u32 s15, s15, s6 // (op 1528) 

s_add_u32 s6, s15, s9 // (op 1528) 

s_mul_hi_u32 s15, s6, s13 // (op 1528) 
s_sub_u32 s7, s6, s15 // (op 1528) 
s_lshr_b32 s8, s7, 1 // (op 1528) 

s_add_u32 s7, s8, s15 // (op 1528) 


s_lshr_b32 s8, s7, s11 // (op 1528) 

s_and_b32 s7, s11, 2147483648 // (op 1528) 
s_cmp_eq_i32 s7, 0 // (op 1528) 
s_mov_b32 s15, scc // (op 1528) 

s_cmp_eq_i32 s15, 0 // (op 1528) 
s_cselect_b32 s7, s6, s8 // (op 1528) 


s_mul_hi_u32 s8, s7, s12 // (op 1528) 
s_sub_u32 s15, s7, s8 // (op 1528) 
s_lshr_b32 s42, s15, 1 // (op 1528) 

s_add_u32 s15, s42, s8 // (op 1528) 


s_lshr_b32 s8, s15, s10 // (op 1528) 

s_and_b32 s15, s10, 2147483648 // (op 1528) 
s_cmp_eq_i32 s15, 0 // (op 1528) 
s_mov_b32 s42, scc // (op 1528) 

s_cmp_eq_i32 s42, 0 // (op 1528) 
s_cselect_b32 s15, s7, s8 // (op 1528) 


s_mul_i32 s8, s15, s14 // (op 1528) 

s_sub_u32 s15, s7, s8 // (op 1528) 

s_mul_i32 s8, s7, s18 // (op 1528) 


v_lshl_add_u32 v25, s15, 4, v24 // (op 1528) 


s_sub_u32 s7, s6, s8 // (op 1528) 


v_lshl_add_u32 v24, s7, 2, v19 // (op 1528) 


v_mul_lo_u32 v19, v24, s38 // (op 1528) 
v_add_u32 v19, v19, v25 // (op 1528) 


s_mul_i32 s19, s38, s37 // (op 1530) most significant: low of low * high
s_mul_i32 s15, s39, s36 // (op 1530) most significant: low of high * low
s_mul_i32 s18, s38, s36 // (op 1530) least significant
s_mul_hi_u32 s6, s38, s36 // (op 1530) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1530) most significant: sum
s_add_u32 s19, s19, s6 // (op 1530) most significant: sum


s_mov_b32 s72, s40 // (op 1530) reg expression
s_mov_b32 s73, s41 // (op 1530) reg expression
s_mov_b32 s74, s18 // (op 1530) reg expression
s_mov_b32 s75, 131072 // (op 1530) call()

v_lshlrev_b32 v24, 4, v1 // (op 2033) 
s_nop 0
v_readfirstlane_b32 s15, v24 // (op 2033) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v24, 4, v1 // (op 2098) 
s_nop 0
v_readfirstlane_b32 s6, v24 // (op 2098) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -16, v1 // (op 2163) 
v_sub_u32 v25, v1, v24 // (op 2163) 

v_and_b32 v24, 4294967280, v1 // (op 2163) 
v_add_u32 v26, v24, v25 // (op 2163) 


v_and_b32 v24, -16, v1 // (op 2229) 
v_sub_u32 v25, v1, v24 // (op 2229) 

v_and_b32 v24, 4294967280, v1 // (op 2229) 
v_add_u32 v27, v24, v25 // (op 2229) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 450) 
s_and_b32 s7, s18, 4294967294 // (op 450) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 449) 
s_mov_b32 s8, s18 // (op 449) convert

s_cmp_lt_u32 s7, s8 // (op 449) 







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_449 // (op 449) Condition: Top (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_449 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_449:
s_barrier  // (op 2735) 
s_add_u32 s7, s7, 1 // (op 451) 

s_lshl_b64 s[18:19], s[24:25], 6 // (op 1051) 
v_mov_b64 v[24:25], s[18:19] // (op 1051) convert


v_mov_b64 v[28:29], v[22:23] // (op 1051) 
v_mov_b64 v[32:33], v[28:29] // (op 1051) reg expression
s_add_u32 s8, 2048, s15 // (op 1051) 
s_mov_b32 m0, s8 // (op 1051) reg expression
buffer_load_dwordx4 v32, s[60:63], 0 offen offset: 0 lds // (op 1051) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v24 // (op 1051) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 1051) most significant half

s_add_u32 m0, m0, 1024 // (op 1051) 
buffer_load_dwordx4 v32, s[60:63], 0 offen offset: 0 lds // (op 1051) Load value direct to lds




s_lshl_b64 s[18:19], s[24:25], 7 // (op 1126) 
v_mov_b32 v7, s18 // (op 1126) 
v_mov_b32 v14, s19 // (op 1126) 
v_add_co_u32 v22, s[42:43], v22, v7 // (op 1126) least significant half
v_addc_co_u32 v23, s[42:43], v23, v14, s[42:43] // (op 1126) most significant half




s_barrier  // (op 2754) 

s_lshl_b64 s[18:19], s[28:29], 6 // (op 1057) 
v_mov_b64 v[24:25], s[18:19] // (op 1057) convert


v_mov_b64 v[28:29], v[30:31] // (op 1057) 
v_mov_b64 v[32:33], v[28:29] // (op 1057) reg expression
s_mov_b32 m0, s6 // (op 1057) reg expression
buffer_load_dwordx4 v32, s[64:67], 0 offen offset: 0 lds // (op 1057) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v24 // (op 1057) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 1057) most significant half

s_add_u32 m0, m0, 1024 // (op 1057) 
buffer_load_dwordx4 v32, s[64:67], 0 offen offset: 0 lds // (op 1057) Load value direct to lds



s_lshl_b64 s[18:19], s[28:29], 7 // (op 1264) 
v_mov_b32 v7, s18 // (op 1264) 
v_mov_b32 v14, s19 // (op 1264) 
v_add_co_u32 v30, s[42:43], v30, v7 // (op 1264) least significant half
v_addc_co_u32 v31, s[42:43], v31, v14, s[42:43] // (op 1264) most significant half






v_mov_b32 v7, v21 // (op 469) 
v_mov_b32 v14, v7 // (op 469) reg expression

buffer_load_ubyte v20, v14, s[68:71], 0 offen offset: 0 // (op 469) Load value


s_lshl_b32 s8, s32, 2 // (op 1392) 
v_add_u32 v21, v21, s8 // (op 1392) 



s_barrier  // (op 2774) 
v_mov_b32 v7, v19 // (op 470) 



v_mov_b32 v14, v7 // (op 470) reg expression

buffer_load_ubyte v24, v14, s[72:75], 0 offen offset: 0 // (op 470) Load value


s_lshl_b32 s8, s38, 2 // (op 1520) 
v_add_u32 v19, v19, s8 // (op 1520) 

s_waitcnt vmcnt(1)
ds_write_b8 v26, v20 offset:8256 // (op 466) Store local data 

s_barrier  // (op 2741) 



s_barrier  // (op 2794) 
ds_read_b64_tr_b8 v[32:33], v4 offset:2048 // (op 457) Transpose load local data 



ds_read_b64_tr_b8 v[34:35], v4 offset:2176 // (op 457) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:3072 // (op 457) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v4 offset:3200 // (op 457) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v27, v24 offset:8192 // (op 467) Store local data 

s_barrier  // (op 2761) 



ds_read_b64_tr_b8 v[40:41], v6 // (op 459) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v6 offset:128 // (op 459) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v6 offset:1024 // (op 459) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v6 offset:1152 // (op 459) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2781) 


ds_read_u8 v7, v9 offset:8256 // (op 461) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2801) 


ds_read_u8 v14, v11 offset:8192 // (op 463) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[40:47], a[0:3], v7, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 464) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 449) 
s_mov_b32 s8, s18 // (op 449) convert

s_cmp_lt_u32 s7, s8 // (op 449) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_449 // (op 449) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_449 if true)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_449:









s_barrier  // (op 2949) 




v_cmp_eq_f32 vcc, 0.00000, v16 // (op 234) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_234 // (op 234) Condition: False, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_234
s_nop 0
v_accvgpr_read v20, a0 // (op 242) Wait state hazard: XDL Write Hazard
v_accvgpr_read v21, a1 // (op 242) 
v_accvgpr_read v22, a2 // (op 242) 
v_accvgpr_read v23, a3 // (op 242) 
v_mul_f32 v24, v18, v20 // (op 242) 
v_mul_f32 v25, v18, v21 // (op 242) 
v_mul_f32 v26, v18, v22 // (op 242) 
v_mul_f32 v27, v18, v23 // (op 242) 

v_lshrrev_b32 v0, 6, v1 // (op 1986) 
s_mul_i32 s15, s16, s50 // (op 1986) 
s_add_u32 s15, s15, s0 // (op 1986) 
v_and_b32 v5, 63, v1 // (op 1986) 
s_add_u32 s6, s15, s9 // (op 1986) 

v_lshrrev_b32 v7, 4, v5 // (op 1986) 
s_mul_hi_u32 s15, s6, s13 // (op 1986) 
v_lshlrev_b32 v12, 2, v7 // (op 1986) 

s_sub_u32 s7, s6, s15 // (op 1986) 
v_lshl_add_u32 v7, v0, 4, v12 // (op 1986) 


s_lshr_b32 s8, s7, 1 // (op 1986) 

v_lshrrev_b32 v0, 2, v5 // (op 1986) 

s_add_u32 s7, s8, s15 // (op 1986) 


v_and_b32 v5, 3, v0 // (op 1986) 

s_lshr_b32 s8, s7, s11 // (op 1986) 

v_and_b32 v0, 3, v1 // (op 1986) 
s_and_b32 s7, s11, 2147483648 // (op 1986) 
v_lshl_add_u32 v12, v5, 2, v0 // (op 1986) 


s_cmp_eq_i32 s7, 0 // (op 1986) 
s_mov_b32 s15, scc // (op 1986) 

s_cmp_eq_i32 s15, 0 // (op 1986) 
s_cselect_b32 s7, s6, s8 // (op 1986) 



s_mul_hi_u32 s6, s7, s12 // (op 1986) 
s_sub_u32 s8, s7, s6 // (op 1986) 
s_lshr_b32 s15, s8, 1 // (op 1986) 

s_add_u32 s8, s15, s6 // (op 1986) 


s_lshr_b32 s6, s8, s10 // (op 1986) 

s_and_b32 s8, s10, 2147483648 // (op 1986) 
s_cmp_eq_i32 s8, 0 // (op 1986) 
s_mov_b32 s15, scc // (op 1986) 

s_cmp_eq_i32 s15, 0 // (op 1986) 
s_cselect_b32 s8, s7, s6 // (op 1986) 


s_mul_i32 s6, s8, s14 // (op 1986) 
v_lshl_add_u32 v0, s8, 4, v7 // (op 1986) 


s_sub_u32 s8, s7, s6 // (op 1986) 


v_lshl_add_u32 v5, s8, 4, v12 // (op 1986) 


v_mul_lo_u32 v7, v5, s46 // (op 1986) 

v_add_lshl_u32 v5, v0, v7, 2 // (op 1986) 


s_mul_i32 s19, s46, s23 // (op 1988) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1988) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1988) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1988) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1988) most significant: sum
s_add_u32 s19, s19, s6 // (op 1988) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1988) 

s_mov_b32 s76, s48 // (op 1988) reg expression
s_mov_b32 s77, s49 // (op 1988) reg expression
s_mov_b32 s78, s6 // (op 1988) reg expression
s_mov_b32 s79, 131072 // (op 1988) call()




buffer_store_dwordx4 v[24:27], v5, s[76:79], 0 offen // (op 244) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_234 // (op 234) Condition: Done, jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_234
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_234:



buffer_load_dwordx4 v[20:23], v3, s[56:59], 0 offen // (op 257) Load value
v_accvgpr_read v24, a0 // (op 254) 
v_accvgpr_read v25, a1 // (op 254) 
v_accvgpr_read v26, a2 // (op 254) 
v_accvgpr_read v27, a3 // (op 254) 
v_mul_f32 v28, v18, v24 // (op 254) 
v_mul_f32 v29, v18, v25 // (op 254) 
v_mul_f32 v30, v18, v26 // (op 254) 
v_mul_f32 v31, v18, v27 // (op 254) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v16, v20 // (op 258) 
v_mul_f32 v25, v16, v21 // (op 258) 
v_mul_f32 v26, v16, v22 // (op 258) 
v_mul_f32 v27, v16, v23 // (op 258) 

v_add_f32 v20, v24, v28 // (op 255) 
v_add_f32 v21, v25, v29 // (op 255) 
v_add_f32 v22, v26, v30 // (op 255) 
v_add_f32 v23, v27, v31 // (op 255) 


v_lshrrev_b32 v0, 6, v1 // (op 2003) 
s_mul_i32 s15, s16, s50 // (op 2003) 
s_add_u32 s15, s15, s0 // (op 2003) 
v_and_b32 v5, 63, v1 // (op 2003) 
s_add_u32 s6, s15, s9 // (op 2003) 

v_lshrrev_b32 v7, 4, v5 // (op 2003) 
s_mul_hi_u32 s15, s6, s13 // (op 2003) 
v_lshlrev_b32 v12, 2, v7 // (op 2003) 

s_sub_u32 s7, s6, s15 // (op 2003) 
v_lshl_add_u32 v7, v0, 4, v12 // (op 2003) 


s_lshr_b32 s8, s7, 1 // (op 2003) 

v_lshrrev_b32 v0, 2, v5 // (op 2003) 

s_add_u32 s7, s8, s15 // (op 2003) 


v_and_b32 v5, 3, v0 // (op 2003) 

s_lshr_b32 s8, s7, s11 // (op 2003) 

v_and_b32 v0, 3, v1 // (op 2003) 
s_and_b32 s7, s11, 2147483648 // (op 2003) 
v_lshl_add_u32 v12, v5, 2, v0 // (op 2003) 


s_cmp_eq_i32 s7, 0 // (op 2003) 
s_mov_b32 s15, scc // (op 2003) 

s_cmp_eq_i32 s15, 0 // (op 2003) 
s_cselect_b32 s7, s6, s8 // (op 2003) 



s_mul_hi_u32 s6, s7, s12 // (op 2003) 
s_sub_u32 s8, s7, s6 // (op 2003) 
s_lshr_b32 s15, s8, 1 // (op 2003) 

s_add_u32 s8, s15, s6 // (op 2003) 


s_lshr_b32 s6, s8, s10 // (op 2003) 

s_and_b32 s8, s10, 2147483648 // (op 2003) 
s_cmp_eq_i32 s8, 0 // (op 2003) 
s_mov_b32 s15, scc // (op 2003) 

s_cmp_eq_i32 s15, 0 // (op 2003) 
s_cselect_b32 s8, s7, s6 // (op 2003) 


s_mul_i32 s6, s8, s14 // (op 2003) 
v_lshl_add_u32 v0, s8, 4, v7 // (op 2003) 


s_sub_u32 s8, s7, s6 // (op 2003) 


v_lshl_add_u32 v5, s8, 4, v12 // (op 2003) 


v_mul_lo_u32 v7, v5, s46 // (op 2003) 

v_add_lshl_u32 v5, v0, v7, 2 // (op 2003) 


s_mul_i32 s19, s46, s23 // (op 2005) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 2005) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 2005) least significant
s_mul_hi_u32 s6, s46, s22 // (op 2005) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2005) most significant: sum
s_add_u32 s19, s19, s6 // (op 2005) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 2005) 

s_mov_b32 s76, s48 // (op 2005) reg expression
s_mov_b32 s77, s49 // (op 2005) reg expression
s_mov_b32 s78, s6 // (op 2005) reg expression
s_mov_b32 s79, 131072 // (op 2005) call()




buffer_store_dwordx4 v[20:23], v5, s[76:79], 0 offen // (op 256) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_234:

s_add_u32 s0, s0, s1 // (op 443) 
s_mul_i32 s15, s50, s16 // (op 441) 
s_cmp_gt_u32 s51, s15 // (op 441) 
s_mov_b32 s6, scc // (op 441) 
s_sub_u32 s7, s51, s15 // (op 441) 

s_cmp_eq_i32 s6, 0 // (op 441) 
s_cselect_b32 s8, 0, s7 // (op 441) 


s_cmp_lt_u32 s50, s8 // (op 441) 
s_mov_b32 s15, scc // (op 441) 
s_cmp_eq_i32 s15, 0 // (op 441) 
s_cselect_b32 s6, s8, s50 // (op 441) 


s_cmp_lt_u32 s0, s6 // (op 441) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_DPStreamTileLoop_441 // (op 441) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopTop_DPStreamTileLoop_441 if true)
RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_ForLoopBottom_DPStreamTileLoop_441:










































s_endpgm  // End of RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2
.LRR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_end:
.size RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2, .LRR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2_end-RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2
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
  - .name: RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2
    .symbol: RR_GEMM_NT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x16x128_UR_2.kd
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

