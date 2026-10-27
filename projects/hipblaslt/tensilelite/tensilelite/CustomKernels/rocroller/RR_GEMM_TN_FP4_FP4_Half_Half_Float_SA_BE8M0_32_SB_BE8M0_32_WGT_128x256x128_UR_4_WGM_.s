// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM_
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM_,@function
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__exec_begin:


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
s_load_dword s48, s[0:1], 168 // Load scalar value

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


s_mul_hi_i32 s0, s16, s10 // (op 20656) 
s_mul_hi_i32 s1, s16, s10 // (op 20795) 
v_lshrrev_b32 v0, 7, v1 // (op 20717) 
v_and_b32 v3, 15, v1 // (op 20717) 
v_lshl_add_u32 v4, v0, 6, v3 // (op 20717) 


v_and_b32 v0, 63, v1 // (op 20717) 
v_lshrrev_b32 v3, 4, v0 // (op 20717) 

v_lshlrev_b32 v0, 5, v3 // (op 20717) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 20717) 


v_lshlrev_b32 v0, 2, v3 // (op 20717) 

v_lshrrev_b32 v3, 3, v0 // (op 20717) 

v_lshrrev_b32 v0, 6, v1 // (op 20774) 
v_and_b32 v4, 1, v0 // (op 20774) 

v_and_b32 v0, 15, v1 // (op 20774) 
v_lshl_add_u32 v5, v4, 7, v0 // (op 20774) 


v_and_b32 v0, 63, v1 // (op 20774) 
v_lshrrev_b32 v4, 4, v0 // (op 20774) 

v_lshlrev_b32 v0, 5, v4 // (op 20774) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 20774) 


v_lshlrev_b32 v0, 2, v4 // (op 20774) 

v_lshrrev_b32 v4, 3, v0 // (op 20774) 

v_lshrrev_b32 v0, 7, v1 // (op 20841) 
v_and_b32 v5, 15, v1 // (op 20841) 
v_lshl_add_u32 v6, v0, 6, v5 // (op 20841) 


v_and_b32 v0, 63, v1 // (op 20841) 
v_lshrrev_b32 v5, 4, v0 // (op 20841) 

v_lshlrev_b32 v0, 5, v5 // (op 20841) 

v_lshl_add_u32 v5, v6, 7, v0 // (op 20841) 


v_lshlrev_b32 v0, 2, v5 // (op 20841) 

v_lshrrev_b32 v5, 3, v0 // (op 20841) 

v_lshrrev_b32 v0, 6, v1 // (op 20885) 
v_and_b32 v6, 1, v0 // (op 20885) 

v_and_b32 v0, 15, v1 // (op 20885) 
v_lshl_add_u32 v7, v6, 7, v0 // (op 20885) 


v_and_b32 v0, 63, v1 // (op 20885) 
v_lshrrev_b32 v6, 4, v0 // (op 20885) 

v_lshlrev_b32 v0, 5, v6 // (op 20885) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 20885) 


v_lshlrev_b32 v0, 2, v6 // (op 20885) 

v_lshrrev_b32 v6, 3, v0 // (op 20885) 

v_lshrrev_b32 v0, 7, v1 // (op 20929) 
v_and_b32 v7, 15, v1 // (op 20929) 
v_lshl_add_u32 v8, v0, 6, v7 // (op 20929) 


v_and_b32 v0, 63, v1 // (op 20929) 
v_lshrrev_b32 v7, 4, v0 // (op 20929) 

v_lshlrev_b32 v0, 5, v7 // (op 20929) 

v_lshl_add_u32 v7, v8, 7, v0 // (op 20929) 


v_lshlrev_b32 v0, 2, v7 // (op 20929) 

v_lshrrev_b32 v7, 3, v0 // (op 20929) 

v_lshrrev_b32 v0, 6, v1 // (op 20973) 
v_and_b32 v8, 1, v0 // (op 20973) 

v_and_b32 v0, 15, v1 // (op 20973) 
v_lshl_add_u32 v9, v8, 7, v0 // (op 20973) 


v_and_b32 v0, 63, v1 // (op 20973) 
v_lshrrev_b32 v8, 4, v0 // (op 20973) 

v_lshlrev_b32 v0, 5, v8 // (op 20973) 

v_lshl_add_u32 v8, v9, 7, v0 // (op 20973) 


v_lshlrev_b32 v0, 2, v8 // (op 20973) 

v_lshrrev_b32 v8, 3, v0 // (op 20973) 

v_lshrrev_b32 v0, 7, v1 // (op 21017) 
v_and_b32 v9, 15, v1 // (op 21017) 
v_lshl_add_u32 v10, v0, 6, v9 // (op 21017) 


v_and_b32 v0, 63, v1 // (op 21017) 
v_lshrrev_b32 v9, 4, v0 // (op 21017) 

v_lshlrev_b32 v0, 5, v9 // (op 21017) 

v_lshl_add_u32 v9, v10, 7, v0 // (op 21017) 


v_lshlrev_b32 v0, 2, v9 // (op 21017) 

v_lshrrev_b32 v9, 3, v0 // (op 21017) 

v_lshrrev_b32 v0, 6, v1 // (op 21061) 
v_and_b32 v10, 1, v0 // (op 21061) 

v_and_b32 v0, 15, v1 // (op 21061) 
v_lshl_add_u32 v11, v10, 7, v0 // (op 21061) 


v_and_b32 v0, 63, v1 // (op 21061) 
v_lshrrev_b32 v10, 4, v0 // (op 21061) 

v_lshlrev_b32 v0, 5, v10 // (op 21061) 

v_lshl_add_u32 v10, v11, 7, v0 // (op 21061) 


v_lshlrev_b32 v0, 2, v10 // (op 21061) 

v_lshrrev_b32 v10, 3, v0 // (op 21061) 

v_lshrrev_b32 v0, 7, v1 // (op 20656) 
s_add_i32 s18, s0, s16 // (op 20656) 

v_and_b32 v11, 63, v1 // (op 20656) 
s_ashr_i32 s0, s18, 31 // (op 20656) 
v_lshrrev_b32 v12, 4, v11 // (op 20656) 
s_cmp_eq_i32 s10, 0 // (op 20656) 
s_mov_b32 s19, scc // (op 20656) 
v_lshlrev_b32 v13, 2, v12 // (op 20656) 

s_cmp_eq_i32 s19, 0 // (op 20656) 
s_cselect_b32 s49, 0, -1 // (op 20656) 

v_lshl_add_u32 v12, v0, 6, v13 // (op 20656) 


s_and_b32 s19, s13, 31 // (op 20656) 
v_lshrrev_b32 v0, 2, v11 // (op 20656) 

v_and_b32 v11, 3, v0 // (op 20656) 

v_lshrrev_b32 v0, 7, v1 // (op 20795) 
v_and_b32 v13, 63, v1 // (op 20795) 
s_lshl_b32 s50, 1, s19 // (op 20656) 
v_and_b32 v14, 3, v1 // (op 20656) 
s_add_i32 s51, s50, s49 // (op 20656) 


s_and_b32 s49, s0, s51 // (op 20656) 


s_add_i32 s0, s1, s16 // (op 20795) 

s_ashr_i32 s1, s0, 31 // (op 20795) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 20656) 


s_add_i32 s50, s18, s49 // (op 20656) 


v_lshrrev_b32 v11, 6, v1 // (op 20656) 
s_ashr_i32 s18, s50, s19 // (op 20656) 


v_and_b32 v14, 1, v11 // (op 20656) 

s_lshl_b32 s19, s13, 24 // (op 20656) 
v_lshl_add_u32 v11, v14, 7, v15 // (op 20656) 


s_ashr_i32 s49, s19, 31 // (op 20656) 

s_xor_b32 s19, s18, s49 // (op 20656) 

s_sub_i32 s18, s19, s49 // (op 20656) 


s_mul_hi_i32 s19, s18, s14 // (op 20656) 
s_add_i32 s49, s19, s18 // (op 20656) 

s_ashr_i32 s19, s49, 31 // (op 20656) 
s_cmp_eq_i32 s14, 0 // (op 20656) 
s_mov_b32 s50, scc // (op 20656) 
s_cmp_eq_i32 s50, 0 // (op 20656) 
s_cselect_b32 s51, 0, -1 // (op 20656) 

s_and_b32 s50, s15, 31 // (op 20656) 
v_lshrrev_b32 v14, 4, v13 // (op 20795) 
s_cmp_eq_i32 s10, 0 // (op 20795) 
s_mov_b32 s52, scc // (op 20795) 
v_lshlrev_b32 v15, 2, v14 // (op 20795) 

s_cmp_eq_i32 s52, 0 // (op 20795) 
s_cselect_b32 s53, 0, -1 // (op 20795) 

s_lshl_b32 s52, 1, s50 // (op 20656) 
v_lshl_add_u32 v14, v0, 6, v15 // (op 20795) 


v_lshrrev_b32 v0, 2, v13 // (op 20795) 

s_add_i32 s54, s52, s51 // (op 20656) 


s_and_b32 s51, s19, s54 // (op 20656) 


s_add_i32 s19, s49, s51 // (op 20656) 


s_ashr_i32 s49, s19, s50 // (op 20656) 


s_lshl_b32 s19, s15, 24 // (op 20656) 
s_ashr_i32 s50, s19, 31 // (op 20656) 

s_xor_b32 s19, s49, s50 // (op 20656) 

s_sub_i32 s49, s19, s50 // (op 20656) 


s_mul_i32 s19, s18, s11 // (op 20656) 
s_sub_u32 s50, s16, s19 // (op 20656) 

s_mul_i32 s19, s49, s11 // (op 20656) 
s_add_u32 s19, s19, s50 // (op 20656) 

s_mul_hi_i32 s50, s16, s9 // (op 20656) 
s_add_i32 s51, s50, s16 // (op 20656) 

s_ashr_i32 s50, s51, 31 // (op 20656) 
s_cmp_eq_i32 s9, 0 // (op 20656) 
s_mov_b32 s52, scc // (op 20656) 
s_cmp_eq_i32 s52, 0 // (op 20656) 
s_cselect_b32 s54, 0, -1 // (op 20656) 

s_and_b32 s52, s13, 31 // (op 20795) 
v_and_b32 v13, 3, v0 // (op 20795) 

s_lshl_b32 s55, 1, s52 // (op 20795) 
v_and_b32 v0, 3, v1 // (op 20795) 
s_add_i32 s56, s55, s53 // (op 20795) 


s_and_b32 s53, s1, s56 // (op 20795) 


v_lshl_add_u32 v15, v13, 2, v0 // (op 20795) 


s_add_i32 s1, s0, s53 // (op 20795) 


v_lshrrev_b32 v0, 6, v1 // (op 20795) 
s_ashr_i32 s0, s1, s52 // (op 20795) 


v_and_b32 v13, 1, v0 // (op 20795) 

s_lshl_b32 s1, s13, 24 // (op 20795) 
v_lshl_add_u32 v0, v13, 7, v15 // (op 20795) 


s_ashr_i32 s52, s1, 31 // (op 20795) 

s_xor_b32 s1, s0, s52 // (op 20795) 

s_sub_i32 s0, s1, s52 // (op 20795) 


s_mul_hi_i32 s1, s0, s14 // (op 20795) 
s_add_i32 s52, s1, s0 // (op 20795) 

s_ashr_i32 s1, s52, 31 // (op 20795) 
s_cmp_eq_i32 s14, 0 // (op 20795) 
s_mov_b32 s53, scc // (op 20795) 
s_cmp_eq_i32 s53, 0 // (op 20795) 
s_cselect_b32 s55, 0, -1 // (op 20795) 

s_and_b32 s53, s15, 31 // (op 20795) 
s_lshl_b32 s56, 1, s53 // (op 20795) 
s_add_i32 s57, s56, s55 // (op 20795) 


s_and_b32 s55, s1, s57 // (op 20795) 


s_add_i32 s1, s52, s55 // (op 20795) 


s_ashr_i32 s52, s1, s53 // (op 20795) 


s_lshl_b32 s1, s15, 24 // (op 20795) 
s_ashr_i32 s53, s1, 31 // (op 20795) 

s_xor_b32 s1, s52, s53 // (op 20795) 

s_sub_i32 s52, s1, s53 // (op 20795) 


s_mul_i32 s1, s0, s11 // (op 20795) 
s_sub_u32 s53, s16, s1 // (op 20795) 

s_mul_i32 s1, s52, s11 // (op 20795) 
s_add_u32 s1, s1, s53 // (op 20795) 

s_mul_hi_i32 s53, s16, s9 // (op 20795) 
s_add_i32 s55, s53, s16 // (op 20795) 

s_ashr_i32 s53, s55, 31 // (op 20795) 
s_cmp_eq_i32 s9, 0 // (op 20795) 
s_mov_b32 s56, scc // (op 20795) 
s_cmp_eq_i32 s56, 0 // (op 20795) 
s_cselect_b32 s57, 0, -1 // (op 20795) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 20656) 
s_lshl_b32 s58, 1, s56 // (op 20656) 
s_add_i32 s59, s58, s54 // (op 20656) 


s_and_b32 s54, s50, s59 // (op 20656) 


s_add_i32 s50, s51, s54 // (op 20656) 


s_ashr_i32 s51, s50, s56 // (op 20656) 


s_lshl_b32 s50, s45, 24 // (op 20656) 
s_ashr_i32 s54, s50, 31 // (op 20656) 

s_xor_b32 s50, s51, s54 // (op 20656) 

s_sub_i32 s51, s50, s54 // (op 20656) 


s_and_b32 s50, s45, 31 // (op 20795) 
s_lshl_b32 s54, 1, s50 // (op 20795) 
s_add_i32 s56, s54, s57 // (op 20795) 


s_and_b32 s54, s53, s56 // (op 20795) 


s_add_i32 s53, s55, s54 // (op 20795) 


s_ashr_i32 s54, s53, s50 // (op 20795) 


s_lshl_b32 s50, s45, 24 // (op 20795) 
s_ashr_i32 s53, s50, 31 // (op 20795) 

s_xor_b32 s50, s54, s53 // (op 20795) 

s_sub_i32 s54, s50, s53 // (op 20795) 


s_add_u32 s56, s6, 256 // (op 20656) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 20656) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 20656) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 20656) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 20656) 

s_mul_i32 s50, s11, s56 // (op 20656) 
s_mul_i32 s53, s8, s50 // (op 20656) 

s_mul_i32 s50, s51, s53 // (op 20656) 

s_sub_u32 s53, s16, s50 // (op 20656) 

s_mov_b32 s50, s53 // (op 20656) 
s_mul_hi_i32 s55, s50, s12 // (op 20656) 
s_add_i32 s58, s55, s50 // (op 20656) 


s_ashr_i32 s50, s58, 31 // (op 20656) 
s_cmp_eq_i32 s12, 0 // (op 20656) 
s_mov_b32 s55, scc // (op 20656) 
s_cmp_eq_i32 s55, 0 // (op 20656) 
s_cselect_b32 s59, 0, -1 // (op 20656) 

s_and_b32 s55, s44, 31 // (op 20656) 
s_lshl_b32 s60, 1, s55 // (op 20656) 
s_add_i32 s61, s60, s59 // (op 20656) 


s_and_b32 s59, s50, s61 // (op 20656) 


s_add_i32 s50, s58, s59 // (op 20656) 


s_ashr_i32 s58, s50, s55 // (op 20656) 


s_lshl_b32 s50, s44, 24 // (op 20656) 
s_ashr_i32 s55, s50, 31 // (op 20656) 

s_xor_b32 s50, s58, s55 // (op 20656) 

s_sub_i32 s58, s50, s55 // (op 20656) 


s_mul_i32 s50, s46, s11 // (op 20656) 
s_add_u32 s60, s20, 128 // (op 20656) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 20656) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 20656) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 20656) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 7 // (op 20656) 

s_sub_i32 s55, s60, s50 // (op 20656) 

s_mul_i32 s59, s58, s55 // (op 20656) 

s_sub_u32 s55, s53, s59 // (op 20656) 


s_add_u32 s53, s55, s50 // (op 20656) 


s_cmp_eq_i32 s51, 0 // (op 20656) 
s_mov_b32 s50, scc // (op 20656) 

s_cmp_eq_i32 s50, 0 // (op 20656) 
s_cselect_b32 s51, s53, s19 // (op 20656) 


s_mul_i32 s19, s49, s56 // (op 20656) 


v_lshl_add_u32 v13, s51, 7, v12 // (op 20656) 


s_sub_i32 s49, s18, s19 // (op 20656) 


s_cmp_eq_i32 s50, 0 // (op 20656) 
s_cselect_b32 s51, s58, s49 // (op 20656) 



v_lshl_add_u32 v12, s51, 8, v11 // (op 20656) 


v_mul_lo_u32 v11, v12, s2 // (op 20656) 

v_add_lshl_u32 v12, v13, v11, 1 // (op 20656) 


s_add_u32 s18, s6, 256 // (op 20795) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 20795) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 20795) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 20795) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 8 // (op 20795) 

s_mul_i32 s53, s11, s18 // (op 20795) 
s_mul_i32 s49, s8, s53 // (op 20795) 

s_mul_i32 s53, s54, s49 // (op 20795) 

s_sub_u32 s49, s16, s53 // (op 20795) 

s_mov_b32 s53, s49 // (op 20795) 
s_mul_hi_i32 s50, s53, s12 // (op 20795) 
s_add_i32 s51, s50, s53 // (op 20795) 


s_ashr_i32 s50, s51, 31 // (op 20795) 
s_cmp_eq_i32 s12, 0 // (op 20795) 
s_mov_b32 s53, scc // (op 20795) 
s_cmp_eq_i32 s53, 0 // (op 20795) 
s_cselect_b32 s55, 0, -1 // (op 20795) 

s_and_b32 s53, s44, 31 // (op 20795) 
s_lshl_b32 s56, 1, s53 // (op 20795) 
s_add_i32 s57, s56, s55 // (op 20795) 


s_and_b32 s55, s50, s57 // (op 20795) 


s_add_i32 s50, s51, s55 // (op 20795) 


s_ashr_i32 s51, s50, s53 // (op 20795) 


s_lshl_b32 s50, s44, 24 // (op 20795) 
s_ashr_i32 s53, s50, 31 // (op 20795) 

s_xor_b32 s50, s51, s53 // (op 20795) 

s_sub_i32 s51, s50, s53 // (op 20795) 


s_mul_i32 s50, s46, s11 // (op 20795) 
s_add_u32 s56, s20, 128 // (op 20795) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 20795) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 20795) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 20795) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 20795) 

s_sub_i32 s53, s56, s50 // (op 20795) 

s_mul_i32 s55, s51, s53 // (op 20795) 

s_sub_u32 s53, s49, s55 // (op 20795) 


s_add_u32 s49, s53, s50 // (op 20795) 


s_cmp_eq_i32 s54, 0 // (op 20795) 
s_mov_b32 s50, scc // (op 20795) 

s_cmp_eq_i32 s50, 0 // (op 20795) 
s_cselect_b32 s53, s49, s1 // (op 20795) 


s_mul_i32 s1, s52, s18 // (op 20795) 


v_lshl_add_u32 v11, s53, 7, v14 // (op 20795) 


s_sub_i32 s49, s0, s1 // (op 20795) 


s_cmp_eq_i32 s50, 0 // (op 20795) 
s_cselect_b32 s0, s51, s49 // (op 20795) 



v_lshl_add_u32 v13, s0, 8, v0 // (op 20795) 


v_mul_lo_u32 v0, v13, s4 // (op 20795) 

v_add_lshl_u32 v13, v11, v0, 1 // (op 20795) 


s_mul_i32 s1, s2, s7 // (op 20658) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 20658) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 20658) least significant
s_mul_hi_u32 s19, s2, s6 // (op 20658) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 20658) most significant: sum
s_add_u32 s1, s1, s19 // (op 20658) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 20658) 

s_mul_i32 s1, s4, s7 // (op 20797) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 20797) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 20797) least significant
s_mul_hi_u32 s50, s4, s6 // (op 20797) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 20797) most significant: sum
s_add_u32 s1, s1, s50 // (op 20797) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 20797) 

s_mov_b32 s52, s22 // (op 20658) reg expression
s_mov_b32 s53, s23 // (op 20658) reg expression
s_mov_b32 s54, s18 // (op 20658) reg expression
s_mov_b32 s55, 131072 // (op 20658) call()

s_mov_b32 s56, s24 // (op 20797) reg expression

s_mov_b32 s57, s25 // (op 20797) reg expression
s_mov_b32 s58, s50 // (op 20797) reg expression
s_mov_b32 s59, 131072 // (op 20797) call()


v_mov_b32 v0, s47 // (op 2) Move value
v_mov_b32 v11, s48 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 57) call()
v_accvgpr_write a1, 0.00000 // (op 57) call()
v_accvgpr_write a2, 0.00000 // (op 57) call()
v_accvgpr_write a3, 0.00000 // (op 57) call()
v_accvgpr_write a4, 0.00000 // (op 387) call()
v_accvgpr_write a5, 0.00000 // (op 387) call()
v_accvgpr_write a6, 0.00000 // (op 387) call()
v_accvgpr_write a7, 0.00000 // (op 387) call()
v_accvgpr_write a8, 0.00000 // (op 613) call()
v_accvgpr_write a9, 0.00000 // (op 613) call()
v_accvgpr_write a10, 0.00000 // (op 613) call()
v_accvgpr_write a11, 0.00000 // (op 613) call()
v_accvgpr_write a12, 0.00000 // (op 839) call()
v_accvgpr_write a13, 0.00000 // (op 839) call()
v_accvgpr_write a14, 0.00000 // (op 839) call()
v_accvgpr_write a15, 0.00000 // (op 839) call()
v_accvgpr_write a16, 0.00000 // (op 1065) call()
v_accvgpr_write a17, 0.00000 // (op 1065) call()
v_accvgpr_write a18, 0.00000 // (op 1065) call()
v_accvgpr_write a19, 0.00000 // (op 1065) call()
v_accvgpr_write a20, 0.00000 // (op 1291) call()
v_accvgpr_write a21, 0.00000 // (op 1291) call()
v_accvgpr_write a22, 0.00000 // (op 1291) call()
v_accvgpr_write a23, 0.00000 // (op 1291) call()
v_accvgpr_write a24, 0.00000 // (op 1517) call()
v_accvgpr_write a25, 0.00000 // (op 1517) call()
v_accvgpr_write a26, 0.00000 // (op 1517) call()
v_accvgpr_write a27, 0.00000 // (op 1517) call()
v_accvgpr_write a28, 0.00000 // (op 1743) call()
v_accvgpr_write a29, 0.00000 // (op 1743) call()
v_accvgpr_write a30, 0.00000 // (op 1743) call()
v_accvgpr_write a31, 0.00000 // (op 1743) call()
v_accvgpr_write a32, 0.00000 // (op 2241) call()
v_accvgpr_write a33, 0.00000 // (op 2241) call()
v_accvgpr_write a34, 0.00000 // (op 2241) call()
v_accvgpr_write a35, 0.00000 // (op 2241) call()
v_accvgpr_write a36, 0.00000 // (op 3066) call()
v_accvgpr_write a37, 0.00000 // (op 3066) call()
v_accvgpr_write a38, 0.00000 // (op 3066) call()
v_accvgpr_write a39, 0.00000 // (op 3066) call()
v_accvgpr_write a40, 0.00000 // (op 3067) call()
v_accvgpr_write a41, 0.00000 // (op 3067) call()
v_accvgpr_write a42, 0.00000 // (op 3067) call()
v_accvgpr_write a43, 0.00000 // (op 3067) call()
v_accvgpr_write a44, 0.00000 // (op 3068) call()
v_accvgpr_write a45, 0.00000 // (op 3068) call()
v_accvgpr_write a46, 0.00000 // (op 3068) call()
v_accvgpr_write a47, 0.00000 // (op 3068) call()
v_accvgpr_write a48, 0.00000 // (op 3069) call()
v_accvgpr_write a49, 0.00000 // (op 3069) call()
v_accvgpr_write a50, 0.00000 // (op 3069) call()
v_accvgpr_write a51, 0.00000 // (op 3069) call()
v_accvgpr_write a52, 0.00000 // (op 3070) call()
v_accvgpr_write a53, 0.00000 // (op 3070) call()
v_accvgpr_write a54, 0.00000 // (op 3070) call()
v_accvgpr_write a55, 0.00000 // (op 3070) call()
v_accvgpr_write a56, 0.00000 // (op 3071) call()
v_accvgpr_write a57, 0.00000 // (op 3071) call()
v_accvgpr_write a58, 0.00000 // (op 3071) call()
v_accvgpr_write a59, 0.00000 // (op 3071) call()
v_accvgpr_write a60, 0.00000 // (op 3072) call()
v_accvgpr_write a61, 0.00000 // (op 3072) call()
v_accvgpr_write a62, 0.00000 // (op 3072) call()
v_accvgpr_write a63, 0.00000 // (op 3072) call()
v_accvgpr_write a64, 0.00000 // (op 4243) call()
v_accvgpr_write a65, 0.00000 // (op 4243) call()
v_accvgpr_write a66, 0.00000 // (op 4243) call()
v_accvgpr_write a67, 0.00000 // (op 4243) call()
v_accvgpr_write a68, 0.00000 // (op 5068) call()
v_accvgpr_write a69, 0.00000 // (op 5068) call()
v_accvgpr_write a70, 0.00000 // (op 5068) call()
v_accvgpr_write a71, 0.00000 // (op 5068) call()
v_accvgpr_write a72, 0.00000 // (op 5069) call()
v_accvgpr_write a73, 0.00000 // (op 5069) call()
v_accvgpr_write a74, 0.00000 // (op 5069) call()
v_accvgpr_write a75, 0.00000 // (op 5069) call()
v_accvgpr_write a76, 0.00000 // (op 5070) call()
v_accvgpr_write a77, 0.00000 // (op 5070) call()
v_accvgpr_write a78, 0.00000 // (op 5070) call()
v_accvgpr_write a79, 0.00000 // (op 5070) call()
v_accvgpr_write a80, 0.00000 // (op 5071) call()
v_accvgpr_write a81, 0.00000 // (op 5071) call()
v_accvgpr_write a82, 0.00000 // (op 5071) call()
v_accvgpr_write a83, 0.00000 // (op 5071) call()
v_accvgpr_write a84, 0.00000 // (op 5072) call()
v_accvgpr_write a85, 0.00000 // (op 5072) call()
v_accvgpr_write a86, 0.00000 // (op 5072) call()
v_accvgpr_write a87, 0.00000 // (op 5072) call()
v_accvgpr_write a88, 0.00000 // (op 5073) call()
v_accvgpr_write a89, 0.00000 // (op 5073) call()
v_accvgpr_write a90, 0.00000 // (op 5073) call()
v_accvgpr_write a91, 0.00000 // (op 5073) call()
v_accvgpr_write a92, 0.00000 // (op 5074) call()
v_accvgpr_write a93, 0.00000 // (op 5074) call()
v_accvgpr_write a94, 0.00000 // (op 5074) call()
v_accvgpr_write a95, 0.00000 // (op 5074) call()
v_accvgpr_write a96, 0.00000 // (op 6245) call()
v_accvgpr_write a97, 0.00000 // (op 6245) call()
v_accvgpr_write a98, 0.00000 // (op 6245) call()
v_accvgpr_write a99, 0.00000 // (op 6245) call()
v_accvgpr_write a100, 0.00000 // (op 7070) call()
v_accvgpr_write a101, 0.00000 // (op 7070) call()
v_accvgpr_write a102, 0.00000 // (op 7070) call()
v_accvgpr_write a103, 0.00000 // (op 7070) call()
v_accvgpr_write a104, 0.00000 // (op 7071) call()
v_accvgpr_write a105, 0.00000 // (op 7071) call()
v_accvgpr_write a106, 0.00000 // (op 7071) call()
v_accvgpr_write a107, 0.00000 // (op 7071) call()
v_accvgpr_write a108, 0.00000 // (op 7072) call()
v_accvgpr_write a109, 0.00000 // (op 7072) call()
v_accvgpr_write a110, 0.00000 // (op 7072) call()
v_accvgpr_write a111, 0.00000 // (op 7072) call()
v_accvgpr_write a112, 0.00000 // (op 7073) call()
v_accvgpr_write a113, 0.00000 // (op 7073) call()
v_accvgpr_write a114, 0.00000 // (op 7073) call()
v_accvgpr_write a115, 0.00000 // (op 7073) call()
v_accvgpr_write a116, 0.00000 // (op 7074) call()
v_accvgpr_write a117, 0.00000 // (op 7074) call()
v_accvgpr_write a118, 0.00000 // (op 7074) call()
v_accvgpr_write a119, 0.00000 // (op 7074) call()
v_accvgpr_write a120, 0.00000 // (op 7075) call()
v_accvgpr_write a121, 0.00000 // (op 7075) call()
v_accvgpr_write a122, 0.00000 // (op 7075) call()
v_accvgpr_write a123, 0.00000 // (op 7075) call()
v_accvgpr_write a124, 0.00000 // (op 7076) call()
v_accvgpr_write a125, 0.00000 // (op 7076) call()
v_accvgpr_write a126, 0.00000 // (op 7076) call()
v_accvgpr_write a127, 0.00000 // (op 7076) call()
s_mul_hi_i32 s0, s16, s10 // (op 20546) 
v_lshrrev_b32 v14, 2, v1 // (op 20546) 
s_add_i32 s1, s0, s16 // (op 20546) 

v_and_b32 v15, -4, v1 // (op 20546) 
s_ashr_i32 s0, s1, 31 // (op 20546) 
v_sub_u32 v16, v1, v15 // (op 20546) 

s_cmp_eq_i32 s10, 0 // (op 20546) 
s_mov_b32 s18, scc // (op 20546) 
v_lshlrev_b32 v15, 5, v16 // (op 20546) 

s_cmp_eq_i32 s18, 0 // (op 20546) 
s_cselect_b32 s19, 0, -1 // (op 20546) 

v_mov_b32 v17, 0 // (op 20546) convert
v_mov_b32 v16, v15 // (op 20546) convert

s_and_b32 s18, s13, 31 // (op 20546) 
s_lshl_b32 s22, 1, s18 // (op 20546) 
s_add_i32 s23, s22, s19 // (op 20546) 


s_and_b32 s19, s0, s23 // (op 20546) 


s_add_i32 s0, s1, s19 // (op 20546) 


s_ashr_i32 s1, s0, s18 // (op 20546) 


s_lshl_b32 s0, s13, 24 // (op 20546) 
s_ashr_i32 s18, s0, 31 // (op 20546) 

s_xor_b32 s0, s1, s18 // (op 20546) 

s_sub_i32 s1, s0, s18 // (op 20546) 


s_mul_hi_i32 s0, s1, s14 // (op 20546) 
s_add_i32 s18, s0, s1 // (op 20546) 

s_ashr_i32 s0, s18, 31 // (op 20546) 
s_cmp_eq_i32 s14, 0 // (op 20546) 
s_mov_b32 s19, scc // (op 20546) 
s_cmp_eq_i32 s19, 0 // (op 20546) 
s_cselect_b32 s22, 0, -1 // (op 20546) 

s_and_b32 s19, s15, 31 // (op 20546) 
s_lshl_b32 s23, 1, s19 // (op 20546) 
s_add_i32 s24, s23, s22 // (op 20546) 


s_and_b32 s25, s0, s24 // (op 20546) 


s_add_i32 s0, s18, s25 // (op 20546) 


s_ashr_i32 s18, s0, s19 // (op 20546) 


s_lshl_b32 s0, s15, 24 // (op 20546) 
s_ashr_i32 s19, s0, 31 // (op 20546) 

s_xor_b32 s0, s18, s19 // (op 20546) 

s_sub_i32 s18, s0, s19 // (op 20546) 


s_mul_i32 s0, s1, s11 // (op 20546) 

s_sub_u32 s1, s16, s0 // (op 20546) 

s_mul_i32 s0, s18, s11 // (op 20546) 
s_add_u32 s0, s0, s1 // (op 20546) 


s_mul_hi_i32 s1, s16, s9 // (op 20546) 
s_add_i32 s18, s1, s16 // (op 20546) 

s_ashr_i32 s1, s18, 31 // (op 20546) 
s_cmp_eq_i32 s9, 0 // (op 20546) 
s_mov_b32 s19, scc // (op 20546) 
s_cmp_eq_i32 s19, 0 // (op 20546) 
s_cselect_b32 s22, 0, -1 // (op 20546) 

s_and_b32 s19, s45, 31 // (op 20546) 
s_lshl_b32 s23, 1, s19 // (op 20546) 
s_add_i32 s24, s23, s22 // (op 20546) 


s_and_b32 s25, s1, s24 // (op 20546) 


s_add_i32 s1, s18, s25 // (op 20546) 


s_ashr_i32 s18, s1, s19 // (op 20546) 


s_lshl_b32 s1, s45, 24 // (op 20546) 
s_ashr_i32 s19, s1, 31 // (op 20546) 

s_xor_b32 s1, s18, s19 // (op 20546) 

s_sub_i32 s18, s1, s19 // (op 20546) 


s_add_u32 s22, s6, 256 // (op 20546) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 20546) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 20546) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 20546) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 20546) 

s_mul_i32 s1, s11, s22 // (op 20546) 

s_mul_i32 s19, s8, s1 // (op 20546) 

s_mul_i32 s1, s18, s19 // (op 20546) 

s_sub_u32 s19, s16, s1 // (op 20546) 

s_mov_b32 s1, s19 // (op 20546) 
s_mul_hi_i32 s22, s1, s12 // (op 20546) 
s_add_i32 s23, s22, s1 // (op 20546) 


s_ashr_i32 s1, s23, 31 // (op 20546) 
s_cmp_eq_i32 s12, 0 // (op 20546) 
s_mov_b32 s22, scc // (op 20546) 
s_cmp_eq_i32 s22, 0 // (op 20546) 
s_cselect_b32 s24, 0, -1 // (op 20546) 

s_and_b32 s22, s44, 31 // (op 20546) 
s_lshl_b32 s25, 1, s22 // (op 20546) 
s_add_i32 s47, s25, s24 // (op 20546) 


s_and_b32 s24, s1, s47 // (op 20546) 


s_add_i32 s1, s23, s24 // (op 20546) 


s_ashr_i32 s23, s1, s22 // (op 20546) 


s_lshl_b32 s1, s44, 24 // (op 20546) 
s_ashr_i32 s22, s1, 31 // (op 20546) 

s_xor_b32 s1, s23, s22 // (op 20546) 

s_sub_i32 s23, s1, s22 // (op 20546) 


s_mul_i32 s1, s46, s11 // (op 20546) 
s_add_u32 s24, s20, 128 // (op 20546) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 20546) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 20546) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 20546) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 7 // (op 20546) 

s_sub_i32 s22, s24, s1 // (op 20546) 

s_mul_i32 s24, s23, s22 // (op 20546) 


s_sub_u32 s25, s19, s24 // (op 20546) 


s_add_u32 s19, s25, s1 // (op 20546) 


s_cmp_eq_i32 s18, 0 // (op 20546) 
s_mov_b32 s1, scc // (op 20546) 

s_cmp_eq_i32 s1, 0 // (op 20546) 
s_cselect_b32 s18, s19, s0 // (op 20546) 



v_lshl_add_u32 v15, s18, 7, v14 // (op 20546) 


v_mul_lo_u32 v19, v15, s27 // (op 20546) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 20546) most significant: high of low * low
v_mul_lo_u32 v18, v15, s26 // (op 20546) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 20546) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 20546) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 20546) most significant half



v_lshlrev_b64 v[14:15], 2, v[18:19] // (op 20546) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 20546) 

s_mul_i32 s1, s26, s21 // (op 20548) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 20548) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 20548) least significant
s_mul_hi_u32 s19, s26, s20 // (op 20548) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 20548) most significant: sum
s_add_u32 s1, s1, s19 // (op 20548) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 20548) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 20548) 

s_mov_b32 s48, s28 // (op 20548) reg expression
s_mov_b32 s49, s29 // (op 20548) reg expression
s_mov_b32 s50, s0 // (op 20548) reg expression
s_mov_b32 s51, 131072 // (op 20548) call()

s_mul_hi_i32 s47, s16, s10 // (op 20565) 
v_lshrrev_b32 v14, 2, v1 // (op 20565) 
s_add_i32 s0, s47, s16 // (op 20565) 

v_and_b32 v15, -4, v1 // (op 20565) 
s_ashr_i32 s1, s0, 31 // (op 20565) 
v_sub_u32 v18, v1, v15 // (op 20565) 

s_cmp_eq_i32 s10, 0 // (op 20565) 
s_mov_b32 s47, scc // (op 20565) 
v_lshlrev_b32 v15, 5, v18 // (op 20565) 

s_cmp_eq_i32 s47, 0 // (op 20565) 
s_cselect_b32 s18, 0, -1 // (op 20565) 

v_mov_b32 v19, 0 // (op 20565) convert
v_mov_b32 v18, v15 // (op 20565) convert

s_and_b32 s19, s13, 31 // (op 20565) 
s_lshl_b32 s47, 1, s19 // (op 20565) 
s_add_i32 s22, s47, s18 // (op 20565) 


s_and_b32 s18, s1, s22 // (op 20565) 


s_add_i32 s1, s0, s18 // (op 20565) 


s_ashr_i32 s0, s1, s19 // (op 20565) 


s_lshl_b32 s1, s13, 24 // (op 20565) 
s_ashr_i32 s47, s1, 31 // (op 20565) 

s_xor_b32 s1, s0, s47 // (op 20565) 

s_sub_i32 s0, s1, s47 // (op 20565) 


s_mul_hi_i32 s1, s0, s14 // (op 20565) 
s_add_i32 s47, s1, s0 // (op 20565) 

s_ashr_i32 s1, s47, 31 // (op 20565) 
s_cmp_eq_i32 s14, 0 // (op 20565) 
s_mov_b32 s18, scc // (op 20565) 
s_cmp_eq_i32 s18, 0 // (op 20565) 
s_cselect_b32 s19, 0, -1 // (op 20565) 

s_and_b32 s18, s15, 31 // (op 20565) 
s_lshl_b32 s22, 1, s18 // (op 20565) 
s_add_i32 s23, s22, s19 // (op 20565) 


s_and_b32 s19, s1, s23 // (op 20565) 


s_add_i32 s1, s47, s19 // (op 20565) 


s_ashr_i32 s19, s1, s18 // (op 20565) 


s_lshl_b32 s1, s15, 24 // (op 20565) 
s_ashr_i32 s18, s1, 31 // (op 20565) 

s_xor_b32 s1, s19, s18 // (op 20565) 

s_sub_i32 s19, s1, s18 // (op 20565) 


s_mul_i32 s1, s0, s11 // (op 20565) 

s_sub_u32 s0, s16, s1 // (op 20565) 

s_mul_i32 s1, s19, s11 // (op 20565) 
s_add_u32 s1, s1, s0 // (op 20565) 


s_mul_hi_i32 s0, s16, s9 // (op 20565) 
s_add_i32 s47, s0, s16 // (op 20565) 

s_ashr_i32 s0, s47, 31 // (op 20565) 
s_cmp_eq_i32 s9, 0 // (op 20565) 
s_mov_b32 s18, scc // (op 20565) 
s_cmp_eq_i32 s18, 0 // (op 20565) 
s_cselect_b32 s19, 0, -1 // (op 20565) 

s_and_b32 s18, s45, 31 // (op 20565) 
s_lshl_b32 s22, 1, s18 // (op 20565) 
s_add_i32 s23, s22, s19 // (op 20565) 


s_and_b32 s19, s0, s23 // (op 20565) 


s_add_i32 s0, s47, s19 // (op 20565) 


s_ashr_i32 s19, s0, s18 // (op 20565) 


s_lshl_b32 s0, s45, 24 // (op 20565) 
s_ashr_i32 s18, s0, 31 // (op 20565) 

s_xor_b32 s0, s19, s18 // (op 20565) 

s_sub_i32 s19, s0, s18 // (op 20565) 


s_add_u32 s22, s6, 256 // (op 20565) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 20565) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 20565) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 20565) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 20565) 

s_mul_i32 s0, s11, s22 // (op 20565) 

s_mul_i32 s18, s8, s0 // (op 20565) 

s_mul_i32 s0, s19, s18 // (op 20565) 

s_sub_u32 s18, s16, s0 // (op 20565) 

s_mov_b32 s0, s18 // (op 20565) 
s_mul_hi_i32 s47, s0, s12 // (op 20565) 
s_add_i32 s22, s47, s0 // (op 20565) 


s_ashr_i32 s0, s22, 31 // (op 20565) 
s_cmp_eq_i32 s12, 0 // (op 20565) 
s_mov_b32 s47, scc // (op 20565) 
s_cmp_eq_i32 s47, 0 // (op 20565) 
s_cselect_b32 s23, 0, -1 // (op 20565) 

s_and_b32 s47, s44, 31 // (op 20565) 
s_lshl_b32 s24, 1, s47 // (op 20565) 
s_add_i32 s25, s24, s23 // (op 20565) 


s_and_b32 s23, s0, s25 // (op 20565) 


s_add_i32 s0, s22, s23 // (op 20565) 


s_ashr_i32 s22, s0, s47 // (op 20565) 


s_lshl_b32 s0, s44, 24 // (op 20565) 
s_ashr_i32 s47, s0, 31 // (op 20565) 

s_xor_b32 s0, s22, s47 // (op 20565) 

s_sub_i32 s22, s0, s47 // (op 20565) 


s_mul_i32 s0, s46, s11 // (op 20565) 
s_add_u32 s24, s20, 128 // (op 20565) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 20565) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 20565) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 20565) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 7 // (op 20565) 

s_sub_i32 s23, s24, s0 // (op 20565) 

s_mul_i32 s47, s22, s23 // (op 20565) 


s_sub_u32 s22, s18, s47 // (op 20565) 


s_add_u32 s18, s22, s0 // (op 20565) 


s_cmp_eq_i32 s19, 0 // (op 20565) 
s_mov_b32 s0, scc // (op 20565) 

s_cmp_eq_i32 s0, 0 // (op 20565) 
s_cselect_b32 s19, s18, s1 // (op 20565) 



v_lshl_add_u32 v15, s19, 7, v14 // (op 20565) 


v_mul_lo_u32 v21, v15, s27 // (op 20565) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 20565) most significant: high of low * low
v_mul_lo_u32 v20, v15, s26 // (op 20565) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 20565) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 20565) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 20565) most significant half



v_lshlrev_b64 v[14:15], 2, v[20:21] // (op 20565) 

v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 20565) 

s_mul_i32 s1, s26, s21 // (op 20567) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 20567) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 20567) least significant
s_mul_hi_u32 s18, s26, s20 // (op 20567) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 20567) most significant: sum
s_add_u32 s1, s1, s18 // (op 20567) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 20567) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 20567) 

s_mov_b32 s48, s28 // (op 20567) reg expression
s_mov_b32 s49, s29 // (op 20567) reg expression
s_mov_b32 s50, s0 // (op 20567) reg expression
s_mov_b32 s51, 131072 // (op 20567) call()

s_mul_hi_i32 s47, s16, s10 // (op 20610) 
v_lshrrev_b32 v14, 2, v1 // (op 20610) 
s_add_i32 s0, s47, s16 // (op 20610) 

v_and_b32 v15, -4, v1 // (op 20610) 
s_ashr_i32 s1, s0, 31 // (op 20610) 
v_sub_u32 v20, v1, v15 // (op 20610) 

s_cmp_eq_i32 s10, 0 // (op 20610) 
s_mov_b32 s47, scc // (op 20610) 
v_lshlrev_b32 v15, 5, v20 // (op 20610) 

s_cmp_eq_i32 s47, 0 // (op 20610) 
s_cselect_b32 s18, 0, -1 // (op 20610) 

v_mov_b32 v21, 0 // (op 20610) convert
v_mov_b32 v20, v15 // (op 20610) convert

s_and_b32 s19, s13, 31 // (op 20610) 
s_lshl_b32 s47, 1, s19 // (op 20610) 
s_add_i32 s22, s47, s18 // (op 20610) 


s_and_b32 s18, s1, s22 // (op 20610) 


s_add_i32 s1, s0, s18 // (op 20610) 


s_ashr_i32 s0, s1, s19 // (op 20610) 


s_lshl_b32 s1, s13, 24 // (op 20610) 
s_ashr_i32 s47, s1, 31 // (op 20610) 

s_xor_b32 s1, s0, s47 // (op 20610) 

s_sub_i32 s0, s1, s47 // (op 20610) 


s_mul_hi_i32 s1, s0, s14 // (op 20610) 
s_add_i32 s47, s1, s0 // (op 20610) 

s_ashr_i32 s1, s47, 31 // (op 20610) 
s_cmp_eq_i32 s14, 0 // (op 20610) 
s_mov_b32 s18, scc // (op 20610) 
s_cmp_eq_i32 s18, 0 // (op 20610) 
s_cselect_b32 s19, 0, -1 // (op 20610) 

s_and_b32 s18, s15, 31 // (op 20610) 
s_lshl_b32 s22, 1, s18 // (op 20610) 
s_add_i32 s23, s22, s19 // (op 20610) 


s_and_b32 s19, s1, s23 // (op 20610) 


s_add_i32 s1, s47, s19 // (op 20610) 


s_ashr_i32 s19, s1, s18 // (op 20610) 


s_lshl_b32 s1, s15, 24 // (op 20610) 
s_ashr_i32 s18, s1, 31 // (op 20610) 

s_xor_b32 s1, s19, s18 // (op 20610) 

s_sub_i32 s19, s1, s18 // (op 20610) 


s_add_u32 s22, s6, 256 // (op 20610) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 20610) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 20610) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 20610) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 20610) 

s_mul_i32 s1, s19, s22 // (op 20610) 

s_sub_i32 s47, s0, s1 // (op 20610) 


s_mul_hi_i32 s0, s16, s9 // (op 20610) 
s_add_i32 s1, s0, s16 // (op 20610) 

s_ashr_i32 s0, s1, 31 // (op 20610) 
s_cmp_eq_i32 s9, 0 // (op 20610) 
s_mov_b32 s18, scc // (op 20610) 
s_cmp_eq_i32 s18, 0 // (op 20610) 
s_cselect_b32 s19, 0, -1 // (op 20610) 

s_and_b32 s18, s45, 31 // (op 20610) 
s_lshl_b32 s24, 1, s18 // (op 20610) 
s_add_i32 s25, s24, s19 // (op 20610) 


s_and_b32 s19, s0, s25 // (op 20610) 


s_add_i32 s0, s1, s19 // (op 20610) 


s_ashr_i32 s1, s0, s18 // (op 20610) 


s_lshl_b32 s0, s45, 24 // (op 20610) 
s_ashr_i32 s18, s0, 31 // (op 20610) 

s_xor_b32 s0, s1, s18 // (op 20610) 

s_sub_i32 s1, s0, s18 // (op 20610) 


s_mul_i32 s0, s11, s22 // (op 20610) 

s_mul_i32 s18, s8, s0 // (op 20610) 

s_mul_i32 s0, s1, s18 // (op 20610) 

s_sub_u32 s18, s16, s0 // (op 20610) 

s_mul_hi_i32 s0, s18, s12 // (op 20610) 
s_add_i32 s19, s0, s18 // (op 20610) 


s_ashr_i32 s0, s19, 31 // (op 20610) 
s_cmp_eq_i32 s12, 0 // (op 20610) 
s_mov_b32 s18, scc // (op 20610) 
s_cmp_eq_i32 s18, 0 // (op 20610) 
s_cselect_b32 s22, 0, -1 // (op 20610) 

s_and_b32 s18, s44, 31 // (op 20610) 
s_lshl_b32 s23, 1, s18 // (op 20610) 
s_add_i32 s24, s23, s22 // (op 20610) 


s_and_b32 s25, s0, s24 // (op 20610) 


s_add_i32 s0, s19, s25 // (op 20610) 


s_ashr_i32 s19, s0, s18 // (op 20610) 


s_lshl_b32 s0, s44, 24 // (op 20610) 
s_ashr_i32 s18, s0, 31 // (op 20610) 

s_xor_b32 s0, s19, s18 // (op 20610) 

s_sub_i32 s19, s0, s18 // (op 20610) 


s_cmp_eq_i32 s1, 0 // (op 20610) 
s_mov_b32 s0, scc // (op 20610) 

s_cmp_eq_i32 s0, 0 // (op 20610) 
s_cselect_b32 s1, s19, s47 // (op 20610) 



v_lshl_add_u32 v15, s1, 8, v14 // (op 20610) 


v_mul_lo_u32 v23, v15, s31 // (op 20610) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 20610) most significant: high of low * low
v_mul_lo_u32 v22, v15, s30 // (op 20610) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 20610) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 20610) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 20610) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 20610) 


v_lshrrev_b64 v[20:21], 3, v[14:15] // (op 20610) 

s_mul_i32 s1, s30, s7 // (op 20612) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 20612) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 20612) least significant
s_mul_hi_u32 s18, s30, s6 // (op 20612) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 20612) most significant: sum
s_add_u32 s1, s1, s18 // (op 20612) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 20612) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 20612) 

s_mov_b32 s60, s32 // (op 20612) reg expression
s_mov_b32 s61, s33 // (op 20612) reg expression
s_mov_b32 s62, s0 // (op 20612) reg expression
s_mov_b32 s63, 131072 // (op 20612) call()

s_mul_hi_i32 s47, s16, s10 // (op 20629) 
v_lshrrev_b32 v14, 2, v1 // (op 20629) 
s_add_i32 s0, s47, s16 // (op 20629) 

v_and_b32 v15, -4, v1 // (op 20629) 
s_ashr_i32 s1, s0, 31 // (op 20629) 
v_sub_u32 v22, v1, v15 // (op 20629) 

s_cmp_eq_i32 s10, 0 // (op 20629) 
s_mov_b32 s47, scc // (op 20629) 
v_lshlrev_b32 v15, 5, v22 // (op 20629) 

s_cmp_eq_i32 s47, 0 // (op 20629) 
s_cselect_b32 s18, 0, -1 // (op 20629) 

v_mov_b32 v23, 0 // (op 20629) convert
v_mov_b32 v22, v15 // (op 20629) convert

s_and_b32 s19, s13, 31 // (op 20629) 
s_lshl_b32 s47, 1, s19 // (op 20629) 
s_add_i32 s22, s47, s18 // (op 20629) 


s_and_b32 s18, s1, s22 // (op 20629) 


s_add_i32 s1, s0, s18 // (op 20629) 


s_ashr_i32 s0, s1, s19 // (op 20629) 


s_lshl_b32 s1, s13, 24 // (op 20629) 
s_ashr_i32 s47, s1, 31 // (op 20629) 

s_xor_b32 s1, s0, s47 // (op 20629) 

s_sub_i32 s0, s1, s47 // (op 20629) 


s_mul_hi_i32 s1, s0, s14 // (op 20629) 
s_add_i32 s47, s1, s0 // (op 20629) 

s_ashr_i32 s1, s47, 31 // (op 20629) 
s_cmp_eq_i32 s14, 0 // (op 20629) 
s_mov_b32 s18, scc // (op 20629) 
s_cmp_eq_i32 s18, 0 // (op 20629) 
s_cselect_b32 s19, 0, -1 // (op 20629) 

s_and_b32 s18, s15, 31 // (op 20629) 
s_lshl_b32 s22, 1, s18 // (op 20629) 
s_add_i32 s23, s22, s19 // (op 20629) 


s_and_b32 s19, s1, s23 // (op 20629) 


s_add_i32 s1, s47, s19 // (op 20629) 


s_ashr_i32 s19, s1, s18 // (op 20629) 


s_lshl_b32 s1, s15, 24 // (op 20629) 
s_ashr_i32 s18, s1, 31 // (op 20629) 

s_xor_b32 s1, s19, s18 // (op 20629) 

s_sub_i32 s19, s1, s18 // (op 20629) 


s_add_u32 s22, s6, 256 // (op 20629) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 20629) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 20629) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 20629) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 20629) 

s_mul_i32 s1, s19, s22 // (op 20629) 

s_sub_i32 s47, s0, s1 // (op 20629) 


s_mul_hi_i32 s0, s16, s9 // (op 20629) 
s_add_i32 s1, s0, s16 // (op 20629) 

s_ashr_i32 s0, s1, 31 // (op 20629) 
s_cmp_eq_i32 s9, 0 // (op 20629) 
s_mov_b32 s18, scc // (op 20629) 
s_cmp_eq_i32 s18, 0 // (op 20629) 
s_cselect_b32 s19, 0, -1 // (op 20629) 

s_and_b32 s18, s45, 31 // (op 20629) 
s_lshl_b32 s24, 1, s18 // (op 20629) 
s_add_i32 s25, s24, s19 // (op 20629) 


s_and_b32 s19, s0, s25 // (op 20629) 


s_add_i32 s0, s1, s19 // (op 20629) 


s_ashr_i32 s1, s0, s18 // (op 20629) 


s_lshl_b32 s0, s45, 24 // (op 20629) 
s_ashr_i32 s18, s0, 31 // (op 20629) 

s_xor_b32 s0, s1, s18 // (op 20629) 

s_sub_i32 s1, s0, s18 // (op 20629) 


s_mul_i32 s0, s11, s22 // (op 20629) 

s_mul_i32 s18, s8, s0 // (op 20629) 

s_mul_i32 s0, s1, s18 // (op 20629) 

s_sub_u32 s18, s16, s0 // (op 20629) 

s_mul_hi_i32 s0, s18, s12 // (op 20629) 
s_add_i32 s19, s0, s18 // (op 20629) 


s_ashr_i32 s0, s19, 31 // (op 20629) 
s_cmp_eq_i32 s12, 0 // (op 20629) 
s_mov_b32 s18, scc // (op 20629) 
s_cmp_eq_i32 s18, 0 // (op 20629) 
s_cselect_b32 s22, 0, -1 // (op 20629) 

s_and_b32 s18, s44, 31 // (op 20629) 
s_lshl_b32 s23, 1, s18 // (op 20629) 
s_add_i32 s24, s23, s22 // (op 20629) 


s_and_b32 s25, s0, s24 // (op 20629) 


s_add_i32 s0, s19, s25 // (op 20629) 


s_ashr_i32 s19, s0, s18 // (op 20629) 


s_lshl_b32 s0, s44, 24 // (op 20629) 
s_ashr_i32 s18, s0, 31 // (op 20629) 

s_xor_b32 s0, s19, s18 // (op 20629) 

s_sub_i32 s19, s0, s18 // (op 20629) 


s_cmp_eq_i32 s1, 0 // (op 20629) 
s_mov_b32 s0, scc // (op 20629) 

s_cmp_eq_i32 s0, 0 // (op 20629) 
s_cselect_b32 s1, s19, s47 // (op 20629) 



v_lshl_add_u32 v15, s1, 8, v14 // (op 20629) 


v_mul_lo_u32 v25, v15, s31 // (op 20629) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 20629) most significant: high of low * low
v_mul_lo_u32 v24, v15, s30 // (op 20629) least significant: low of low * low
v_add_u32 v25, v25, v14 // (op 20629) most significant: sum


v_add_co_u32 v14, s[0:1], v22, v24 // (op 20629) least significant half
v_addc_co_u32 v15, s[0:1], v23, v25, s[0:1] // (op 20629) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 20629) 


v_lshrrev_b64 v[22:23], 3, v[14:15] // (op 20629) 

s_mul_i32 s1, s30, s7 // (op 20631) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 20631) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 20631) least significant
s_mul_hi_u32 s18, s30, s6 // (op 20631) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 20631) most significant: sum
s_add_u32 s1, s1, s18 // (op 20631) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 20631) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 20631) 

s_mov_b32 s60, s32 // (op 20631) reg expression
s_mov_b32 s61, s33 // (op 20631) reg expression
s_mov_b32 s62, s0 // (op 20631) reg expression
s_mov_b32 s63, 131072 // (op 20631) call()

s_mul_hi_i32 s47, s16, s10 // (op 21083) 
v_and_b32 v14, 63, v1 // (op 21083) 
s_add_i32 s0, s47, s16 // (op 21083) 

v_lshrrev_b32 v15, 4, v14 // (op 21083) 
s_ashr_i32 s1, s0, 31 // (op 21083) 
v_and_b32 v24, 3, v15 // (op 21083) 

s_cmp_eq_i32 s10, 0 // (op 21083) 
s_mov_b32 s47, scc // (op 21083) 
v_and_b32 v15, 15, v1 // (op 21083) 
s_cmp_eq_i32 s47, 0 // (op 21083) 
s_cselect_b32 s18, 0, -1 // (op 21083) 

v_lshl_add_u32 v25, v24, 4, v15 // (op 21083) 


s_and_b32 s19, s13, 31 // (op 21083) 
v_lshrrev_b32 v15, 7, v1 // (op 21083) 
s_lshl_b32 s47, 1, s19 // (op 21083) 
v_lshl_add_u32 v24, v15, 6, v25 // (op 21083) 


s_add_i32 s22, s47, s18 // (op 21083) 


v_lshrrev_b32 v15, 6, v14 // (op 21083) 

s_and_b32 s18, s1, s22 // (op 21083) 


v_lshlrev_b32 v14, 2, v15 // (op 21083) 

s_add_i32 s1, s0, s18 // (op 21083) 


s_ashr_i32 s0, s1, s19 // (op 21083) 


s_lshl_b32 s1, s13, 24 // (op 21083) 
s_ashr_i32 s47, s1, 31 // (op 21083) 

s_xor_b32 s1, s0, s47 // (op 21083) 

s_sub_i32 s0, s1, s47 // (op 21083) 


s_mul_hi_i32 s1, s0, s14 // (op 21083) 
s_add_i32 s47, s1, s0 // (op 21083) 

s_ashr_i32 s1, s47, 31 // (op 21083) 
s_cmp_eq_i32 s14, 0 // (op 21083) 
s_mov_b32 s18, scc // (op 21083) 
s_cmp_eq_i32 s18, 0 // (op 21083) 
s_cselect_b32 s19, 0, -1 // (op 21083) 

s_and_b32 s18, s15, 31 // (op 21083) 
s_lshl_b32 s22, 1, s18 // (op 21083) 
s_add_i32 s23, s22, s19 // (op 21083) 


s_and_b32 s19, s1, s23 // (op 21083) 


s_add_i32 s1, s47, s19 // (op 21083) 


s_ashr_i32 s19, s1, s18 // (op 21083) 


s_lshl_b32 s1, s15, 24 // (op 21083) 
s_ashr_i32 s18, s1, 31 // (op 21083) 

s_xor_b32 s1, s19, s18 // (op 21083) 

s_sub_i32 s19, s1, s18 // (op 21083) 


s_mul_i32 s1, s0, s11 // (op 21083) 

s_sub_u32 s0, s16, s1 // (op 21083) 

s_mul_i32 s1, s19, s11 // (op 21083) 
s_add_u32 s1, s1, s0 // (op 21083) 


s_mul_hi_i32 s0, s16, s9 // (op 21083) 
s_add_i32 s47, s0, s16 // (op 21083) 

s_ashr_i32 s0, s47, 31 // (op 21083) 
s_cmp_eq_i32 s9, 0 // (op 21083) 
s_mov_b32 s18, scc // (op 21083) 
s_cmp_eq_i32 s18, 0 // (op 21083) 
s_cselect_b32 s19, 0, -1 // (op 21083) 

s_and_b32 s18, s45, 31 // (op 21083) 
s_lshl_b32 s22, 1, s18 // (op 21083) 
s_add_i32 s23, s22, s19 // (op 21083) 


s_and_b32 s19, s0, s23 // (op 21083) 


s_add_i32 s0, s47, s19 // (op 21083) 


s_ashr_i32 s19, s0, s18 // (op 21083) 


s_lshl_b32 s0, s45, 24 // (op 21083) 
s_ashr_i32 s18, s0, 31 // (op 21083) 

s_xor_b32 s0, s19, s18 // (op 21083) 

s_sub_i32 s19, s0, s18 // (op 21083) 


s_add_u32 s22, s6, 256 // (op 21083) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 21083) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 21083) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 21083) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 21083) 

s_mul_i32 s0, s11, s22 // (op 21083) 

s_mul_i32 s18, s8, s0 // (op 21083) 

s_mul_i32 s0, s19, s18 // (op 21083) 

s_sub_u32 s18, s16, s0 // (op 21083) 

s_mov_b32 s0, s18 // (op 21083) 
s_mul_hi_i32 s47, s0, s12 // (op 21083) 
s_add_i32 s22, s47, s0 // (op 21083) 


s_ashr_i32 s0, s22, 31 // (op 21083) 
s_cmp_eq_i32 s12, 0 // (op 21083) 
s_mov_b32 s47, scc // (op 21083) 
s_cmp_eq_i32 s47, 0 // (op 21083) 
s_cselect_b32 s23, 0, -1 // (op 21083) 

s_and_b32 s47, s44, 31 // (op 21083) 
s_lshl_b32 s24, 1, s47 // (op 21083) 
s_add_i32 s25, s24, s23 // (op 21083) 


s_and_b32 s23, s0, s25 // (op 21083) 


s_add_i32 s0, s22, s23 // (op 21083) 


s_ashr_i32 s22, s0, s47 // (op 21083) 


s_lshl_b32 s0, s44, 24 // (op 21083) 
s_ashr_i32 s47, s0, 31 // (op 21083) 

s_xor_b32 s0, s22, s47 // (op 21083) 

s_sub_i32 s22, s0, s47 // (op 21083) 


s_mul_i32 s0, s46, s11 // (op 21083) 
s_add_u32 s24, s20, 128 // (op 21083) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 21083) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 21083) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 21083) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 21083) 

s_sub_i32 s23, s24, s0 // (op 21083) 

s_mul_i32 s47, s22, s23 // (op 21083) 


s_sub_u32 s22, s18, s47 // (op 21083) 


s_add_u32 s18, s22, s0 // (op 21083) 


s_cmp_eq_i32 s19, 0 // (op 21083) 
s_mov_b32 s0, scc // (op 21083) 

s_cmp_eq_i32 s0, 0 // (op 21083) 
s_cselect_b32 s19, s18, s1 // (op 21083) 



v_lshl_add_u32 v15, s19, 7, v24 // (op 21083) 


v_mul_lo_u32 v24, v15, s34 // (op 21083) 
v_add_u32 v24, v24, v14 // (op 21083) 


s_mul_i32 s1, s34, s21 // (op 21085) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 21085) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 21085) least significant
s_mul_hi_u32 s18, s34, s20 // (op 21085) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 21085) most significant: sum
s_add_u32 s1, s1, s18 // (op 21085) most significant: sum


s_mov_b32 s64, s36 // (op 21085) reg expression
s_mov_b32 s65, s37 // (op 21085) reg expression
s_mov_b32 s66, s0 // (op 21085) reg expression
s_mov_b32 s67, 131072 // (op 21085) call()

s_mul_hi_i32 s47, s16, s10 // (op 21104) 
v_and_b32 v14, 63, v1 // (op 21104) 
s_add_i32 s0, s47, s16 // (op 21104) 

v_lshrrev_b32 v15, 4, v14 // (op 21104) 
s_ashr_i32 s1, s0, 31 // (op 21104) 
v_and_b32 v25, 3, v15 // (op 21104) 

s_cmp_eq_i32 s10, 0 // (op 21104) 
s_mov_b32 s47, scc // (op 21104) 
v_and_b32 v15, 15, v1 // (op 21104) 
s_cmp_eq_i32 s47, 0 // (op 21104) 
s_cselect_b32 s18, 0, -1 // (op 21104) 

v_lshl_add_u32 v26, v25, 4, v15 // (op 21104) 


s_and_b32 s19, s13, 31 // (op 21104) 
v_lshrrev_b32 v15, 7, v1 // (op 21104) 
s_lshl_b32 s47, 1, s19 // (op 21104) 
v_lshl_add_u32 v25, v15, 6, v26 // (op 21104) 


s_add_i32 s22, s47, s18 // (op 21104) 


v_lshrrev_b32 v15, 6, v14 // (op 21104) 

s_and_b32 s18, s1, s22 // (op 21104) 


v_lshlrev_b32 v14, 2, v15 // (op 21104) 

s_add_i32 s1, s0, s18 // (op 21104) 


s_ashr_i32 s0, s1, s19 // (op 21104) 


s_lshl_b32 s1, s13, 24 // (op 21104) 
s_ashr_i32 s47, s1, 31 // (op 21104) 

s_xor_b32 s1, s0, s47 // (op 21104) 

s_sub_i32 s0, s1, s47 // (op 21104) 


s_mul_hi_i32 s1, s0, s14 // (op 21104) 
s_add_i32 s47, s1, s0 // (op 21104) 

s_ashr_i32 s1, s47, 31 // (op 21104) 
s_cmp_eq_i32 s14, 0 // (op 21104) 
s_mov_b32 s18, scc // (op 21104) 
s_cmp_eq_i32 s18, 0 // (op 21104) 
s_cselect_b32 s19, 0, -1 // (op 21104) 

s_and_b32 s18, s15, 31 // (op 21104) 
s_lshl_b32 s22, 1, s18 // (op 21104) 
s_add_i32 s23, s22, s19 // (op 21104) 


s_and_b32 s19, s1, s23 // (op 21104) 


s_add_i32 s1, s47, s19 // (op 21104) 


s_ashr_i32 s19, s1, s18 // (op 21104) 


s_lshl_b32 s1, s15, 24 // (op 21104) 
s_ashr_i32 s18, s1, 31 // (op 21104) 

s_xor_b32 s1, s19, s18 // (op 21104) 

s_sub_i32 s19, s1, s18 // (op 21104) 


s_mul_i32 s1, s0, s11 // (op 21104) 

s_sub_u32 s0, s16, s1 // (op 21104) 

s_mul_i32 s1, s19, s11 // (op 21104) 
s_add_u32 s1, s1, s0 // (op 21104) 


s_mul_hi_i32 s0, s16, s9 // (op 21104) 
s_add_i32 s47, s0, s16 // (op 21104) 

s_ashr_i32 s0, s47, 31 // (op 21104) 
s_cmp_eq_i32 s9, 0 // (op 21104) 
s_mov_b32 s18, scc // (op 21104) 
s_cmp_eq_i32 s18, 0 // (op 21104) 
s_cselect_b32 s19, 0, -1 // (op 21104) 

s_and_b32 s18, s45, 31 // (op 21104) 
s_lshl_b32 s22, 1, s18 // (op 21104) 
s_add_i32 s23, s22, s19 // (op 21104) 


s_and_b32 s19, s0, s23 // (op 21104) 


s_add_i32 s0, s47, s19 // (op 21104) 


s_ashr_i32 s19, s0, s18 // (op 21104) 


s_lshl_b32 s0, s45, 24 // (op 21104) 
s_ashr_i32 s18, s0, 31 // (op 21104) 

s_xor_b32 s0, s19, s18 // (op 21104) 

s_sub_i32 s19, s0, s18 // (op 21104) 


s_add_u32 s22, s6, 256 // (op 21104) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 21104) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 21104) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 21104) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 21104) 

s_mul_i32 s0, s11, s22 // (op 21104) 

s_mul_i32 s18, s8, s0 // (op 21104) 

s_mul_i32 s0, s19, s18 // (op 21104) 

s_sub_u32 s18, s16, s0 // (op 21104) 

s_mov_b32 s0, s18 // (op 21104) 
s_mul_hi_i32 s47, s0, s12 // (op 21104) 
s_add_i32 s22, s47, s0 // (op 21104) 


s_ashr_i32 s0, s22, 31 // (op 21104) 
s_cmp_eq_i32 s12, 0 // (op 21104) 
s_mov_b32 s47, scc // (op 21104) 
s_cmp_eq_i32 s47, 0 // (op 21104) 
s_cselect_b32 s23, 0, -1 // (op 21104) 

s_and_b32 s47, s44, 31 // (op 21104) 
s_lshl_b32 s24, 1, s47 // (op 21104) 
s_add_i32 s25, s24, s23 // (op 21104) 


s_and_b32 s23, s0, s25 // (op 21104) 


s_add_i32 s0, s22, s23 // (op 21104) 


s_ashr_i32 s22, s0, s47 // (op 21104) 


s_lshl_b32 s0, s44, 24 // (op 21104) 
s_ashr_i32 s47, s0, 31 // (op 21104) 

s_xor_b32 s0, s22, s47 // (op 21104) 

s_sub_i32 s22, s0, s47 // (op 21104) 


s_mul_i32 s0, s46, s11 // (op 21104) 
s_add_u32 s24, s20, 128 // (op 21104) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 21104) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 21104) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 21104) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 7 // (op 21104) 

s_sub_i32 s23, s24, s0 // (op 21104) 

s_mul_i32 s47, s22, s23 // (op 21104) 


s_sub_u32 s22, s18, s47 // (op 21104) 


s_add_u32 s18, s22, s0 // (op 21104) 


s_cmp_eq_i32 s19, 0 // (op 21104) 
s_mov_b32 s0, scc // (op 21104) 

s_cmp_eq_i32 s0, 0 // (op 21104) 
s_cselect_b32 s19, s18, s1 // (op 21104) 



v_lshl_add_u32 v15, s19, 7, v25 // (op 21104) 


v_mul_lo_u32 v25, v15, s34 // (op 21104) 
v_add_u32 v25, v25, v14 // (op 21104) 


s_mul_i32 s1, s34, s21 // (op 21106) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 21106) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 21106) least significant
s_mul_hi_u32 s18, s34, s20 // (op 21106) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 21106) most significant: sum
s_add_u32 s1, s1, s18 // (op 21106) most significant: sum


s_mov_b32 s64, s36 // (op 21106) reg expression
s_mov_b32 s65, s37 // (op 21106) reg expression
s_mov_b32 s66, s0 // (op 21106) reg expression
s_mov_b32 s67, 131072 // (op 21106) call()

s_mul_hi_i32 s47, s16, s10 // (op 21135) 
v_and_b32 v14, 63, v1 // (op 21135) 
s_add_i32 s0, s47, s16 // (op 21135) 

v_lshrrev_b32 v15, 4, v14 // (op 21135) 
s_ashr_i32 s1, s0, 31 // (op 21135) 
v_and_b32 v26, 3, v15 // (op 21135) 

s_cmp_eq_i32 s10, 0 // (op 21135) 
s_mov_b32 s47, scc // (op 21135) 
v_and_b32 v15, 15, v1 // (op 21135) 
s_cmp_eq_i32 s47, 0 // (op 21135) 
s_cselect_b32 s18, 0, -1 // (op 21135) 

v_lshl_add_u32 v27, v26, 4, v15 // (op 21135) 


s_and_b32 s19, s13, 31 // (op 21135) 
v_lshrrev_b32 v15, 6, v1 // (op 21135) 
s_lshl_b32 s47, 1, s19 // (op 21135) 
v_and_b32 v26, 1, v15 // (op 21135) 

s_add_i32 s22, s47, s18 // (op 21135) 


v_lshl_add_u32 v15, v26, 7, v27 // (op 21135) 


s_and_b32 s18, s1, s22 // (op 21135) 


v_lshrrev_b32 v26, 6, v14 // (op 21135) 

s_add_i32 s1, s0, s18 // (op 21135) 


s_ashr_i32 s0, s1, s19 // (op 21135) 


s_lshl_b32 s1, s13, 24 // (op 21135) 
s_ashr_i32 s47, s1, 31 // (op 21135) 

s_xor_b32 s1, s0, s47 // (op 21135) 

s_sub_i32 s0, s1, s47 // (op 21135) 


s_mul_hi_i32 s1, s0, s14 // (op 21135) 
s_add_i32 s47, s1, s0 // (op 21135) 

s_ashr_i32 s1, s47, 31 // (op 21135) 
s_cmp_eq_i32 s14, 0 // (op 21135) 
s_mov_b32 s18, scc // (op 21135) 
s_cmp_eq_i32 s18, 0 // (op 21135) 
s_cselect_b32 s19, 0, -1 // (op 21135) 

s_and_b32 s18, s15, 31 // (op 21135) 
s_lshl_b32 s22, 1, s18 // (op 21135) 
s_add_i32 s23, s22, s19 // (op 21135) 


s_and_b32 s19, s1, s23 // (op 21135) 


s_add_i32 s1, s47, s19 // (op 21135) 


s_ashr_i32 s19, s1, s18 // (op 21135) 


s_lshl_b32 s1, s15, 24 // (op 21135) 
s_ashr_i32 s18, s1, 31 // (op 21135) 

s_xor_b32 s1, s19, s18 // (op 21135) 

s_sub_i32 s19, s1, s18 // (op 21135) 


s_add_u32 s22, s6, 256 // (op 21135) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 21135) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 21135) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 21135) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 21135) 

s_mul_i32 s1, s19, s22 // (op 21135) 

s_sub_i32 s47, s0, s1 // (op 21135) 


s_mul_hi_i32 s0, s16, s9 // (op 21135) 
s_add_i32 s1, s0, s16 // (op 21135) 

s_ashr_i32 s0, s1, 31 // (op 21135) 
s_cmp_eq_i32 s9, 0 // (op 21135) 
s_mov_b32 s18, scc // (op 21135) 
s_cmp_eq_i32 s18, 0 // (op 21135) 
s_cselect_b32 s19, 0, -1 // (op 21135) 

s_and_b32 s18, s45, 31 // (op 21135) 
s_lshl_b32 s24, 1, s18 // (op 21135) 
s_add_i32 s25, s24, s19 // (op 21135) 


s_and_b32 s19, s0, s25 // (op 21135) 


s_add_i32 s0, s1, s19 // (op 21135) 


s_ashr_i32 s1, s0, s18 // (op 21135) 


s_lshl_b32 s0, s45, 24 // (op 21135) 
s_ashr_i32 s18, s0, 31 // (op 21135) 

s_xor_b32 s0, s1, s18 // (op 21135) 

s_sub_i32 s1, s0, s18 // (op 21135) 


s_mul_i32 s0, s11, s22 // (op 21135) 

s_mul_i32 s18, s8, s0 // (op 21135) 

s_mul_i32 s0, s1, s18 // (op 21135) 

s_sub_u32 s18, s16, s0 // (op 21135) 

s_mul_hi_i32 s0, s18, s12 // (op 21135) 
s_add_i32 s19, s0, s18 // (op 21135) 


s_ashr_i32 s0, s19, 31 // (op 21135) 
s_cmp_eq_i32 s12, 0 // (op 21135) 
s_mov_b32 s18, scc // (op 21135) 
s_cmp_eq_i32 s18, 0 // (op 21135) 
s_cselect_b32 s22, 0, -1 // (op 21135) 

s_and_b32 s18, s44, 31 // (op 21135) 
s_lshl_b32 s23, 1, s18 // (op 21135) 
s_add_i32 s24, s23, s22 // (op 21135) 


s_and_b32 s25, s0, s24 // (op 21135) 


s_add_i32 s0, s19, s25 // (op 21135) 


s_ashr_i32 s19, s0, s18 // (op 21135) 


s_lshl_b32 s0, s44, 24 // (op 21135) 
s_ashr_i32 s18, s0, 31 // (op 21135) 

s_xor_b32 s0, s19, s18 // (op 21135) 

s_sub_i32 s19, s0, s18 // (op 21135) 


s_cmp_eq_i32 s1, 0 // (op 21135) 
s_mov_b32 s0, scc // (op 21135) 

s_cmp_eq_i32 s0, 0 // (op 21135) 
s_cselect_b32 s1, s19, s47 // (op 21135) 



v_lshl_add_u32 v14, s1, 8, v15 // (op 21135) 


v_mul_lo_u32 v15, v14, s38 // (op 21135) 

v_lshl_add_u32 v14, v26, 2, v15 // (op 21135) 


s_mul_i32 s1, s38, s7 // (op 21137) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 21137) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 21137) least significant
s_mul_hi_u32 s18, s38, s6 // (op 21137) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 21137) most significant: sum
s_add_u32 s1, s1, s18 // (op 21137) most significant: sum


s_mov_b32 s68, s40 // (op 21137) reg expression
s_mov_b32 s69, s41 // (op 21137) reg expression
s_mov_b32 s70, s0 // (op 21137) reg expression
s_mov_b32 s71, 131072 // (op 21137) call()

s_mul_hi_i32 s47, s16, s10 // (op 21156) 
v_and_b32 v15, 63, v1 // (op 21156) 
s_add_i32 s0, s47, s16 // (op 21156) 

v_lshrrev_b32 v26, 4, v15 // (op 21156) 
s_ashr_i32 s1, s0, 31 // (op 21156) 
v_and_b32 v27, 3, v26 // (op 21156) 

s_cmp_eq_i32 s10, 0 // (op 21156) 
s_mov_b32 s47, scc // (op 21156) 
v_and_b32 v26, 15, v1 // (op 21156) 
s_cmp_eq_i32 s47, 0 // (op 21156) 
s_cselect_b32 s18, 0, -1 // (op 21156) 

v_lshl_add_u32 v28, v27, 4, v26 // (op 21156) 


s_and_b32 s19, s13, 31 // (op 21156) 
v_lshrrev_b32 v26, 6, v1 // (op 21156) 
s_lshl_b32 s47, 1, s19 // (op 21156) 
v_and_b32 v27, 1, v26 // (op 21156) 

s_add_i32 s22, s47, s18 // (op 21156) 


v_lshl_add_u32 v26, v27, 7, v28 // (op 21156) 


s_and_b32 s18, s1, s22 // (op 21156) 


v_lshrrev_b32 v27, 6, v15 // (op 21156) 

s_add_i32 s1, s0, s18 // (op 21156) 


s_ashr_i32 s0, s1, s19 // (op 21156) 


s_lshl_b32 s1, s13, 24 // (op 21156) 
s_ashr_i32 s47, s1, 31 // (op 21156) 

s_xor_b32 s1, s0, s47 // (op 21156) 

s_sub_i32 s0, s1, s47 // (op 21156) 


s_mul_hi_i32 s1, s0, s14 // (op 21156) 
s_add_i32 s47, s1, s0 // (op 21156) 

s_ashr_i32 s1, s47, 31 // (op 21156) 
s_cmp_eq_i32 s14, 0 // (op 21156) 
s_mov_b32 s18, scc // (op 21156) 
s_cmp_eq_i32 s18, 0 // (op 21156) 
s_cselect_b32 s19, 0, -1 // (op 21156) 

s_and_b32 s18, s15, 31 // (op 21156) 
s_lshl_b32 s22, 1, s18 // (op 21156) 
s_add_i32 s23, s22, s19 // (op 21156) 


s_and_b32 s19, s1, s23 // (op 21156) 


s_add_i32 s1, s47, s19 // (op 21156) 


s_ashr_i32 s19, s1, s18 // (op 21156) 


s_lshl_b32 s1, s15, 24 // (op 21156) 
s_ashr_i32 s18, s1, 31 // (op 21156) 

s_xor_b32 s1, s19, s18 // (op 21156) 

s_sub_i32 s19, s1, s18 // (op 21156) 


s_add_u32 s22, s6, 256 // (op 21156) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 21156) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 21156) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 21156) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 21156) 

s_mul_i32 s1, s19, s22 // (op 21156) 

s_sub_i32 s47, s0, s1 // (op 21156) 


s_mul_hi_i32 s0, s16, s9 // (op 21156) 
s_add_i32 s1, s0, s16 // (op 21156) 

s_ashr_i32 s0, s1, 31 // (op 21156) 
s_cmp_eq_i32 s9, 0 // (op 21156) 
s_mov_b32 s18, scc // (op 21156) 
s_cmp_eq_i32 s18, 0 // (op 21156) 
s_cselect_b32 s19, 0, -1 // (op 21156) 

s_and_b32 s18, s45, 31 // (op 21156) 
s_lshl_b32 s24, 1, s18 // (op 21156) 
s_add_i32 s25, s24, s19 // (op 21156) 


s_and_b32 s19, s0, s25 // (op 21156) 


s_add_i32 s0, s1, s19 // (op 21156) 


s_ashr_i32 s1, s0, s18 // (op 21156) 


s_lshl_b32 s0, s45, 24 // (op 21156) 
s_ashr_i32 s18, s0, 31 // (op 21156) 

s_xor_b32 s0, s1, s18 // (op 21156) 

s_sub_i32 s1, s0, s18 // (op 21156) 


s_mul_i32 s0, s11, s22 // (op 21156) 

s_mul_i32 s18, s8, s0 // (op 21156) 

s_mul_i32 s0, s1, s18 // (op 21156) 

s_sub_u32 s18, s16, s0 // (op 21156) 

s_mul_hi_i32 s0, s18, s12 // (op 21156) 
s_add_i32 s19, s0, s18 // (op 21156) 


s_ashr_i32 s0, s19, 31 // (op 21156) 
s_cmp_eq_i32 s12, 0 // (op 21156) 
s_mov_b32 s18, scc // (op 21156) 
s_cmp_eq_i32 s18, 0 // (op 21156) 
s_cselect_b32 s22, 0, -1 // (op 21156) 

s_and_b32 s18, s44, 31 // (op 21156) 
s_lshl_b32 s23, 1, s18 // (op 21156) 
s_add_i32 s24, s23, s22 // (op 21156) 


s_and_b32 s25, s0, s24 // (op 21156) 


s_add_i32 s0, s19, s25 // (op 21156) 


s_ashr_i32 s19, s0, s18 // (op 21156) 


s_lshl_b32 s0, s44, 24 // (op 21156) 
s_ashr_i32 s18, s0, 31 // (op 21156) 

s_xor_b32 s0, s19, s18 // (op 21156) 

s_sub_i32 s19, s0, s18 // (op 21156) 


s_cmp_eq_i32 s1, 0 // (op 21156) 
s_mov_b32 s0, scc // (op 21156) 

s_cmp_eq_i32 s0, 0 // (op 21156) 
s_cselect_b32 s1, s19, s47 // (op 21156) 



v_lshl_add_u32 v15, s1, 8, v26 // (op 21156) 


v_mul_lo_u32 v26, v15, s38 // (op 21156) 

v_lshl_add_u32 v15, v27, 2, v26 // (op 21156) 


s_mul_i32 s1, s38, s7 // (op 21158) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 21158) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 21158) least significant
s_mul_hi_u32 s18, s38, s6 // (op 21158) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 21158) most significant: sum
s_add_u32 s1, s1, s18 // (op 21158) most significant: sum


s_mov_b32 s68, s40 // (op 21158) reg expression
s_mov_b32 s69, s41 // (op 21158) reg expression
s_mov_b32 s70, s0 // (op 21158) reg expression
s_mov_b32 s71, 131072 // (op 21158) call()

v_and_b32 v26, -4, v1 // (op 20757) 
v_sub_u32 v27, v1, v26 // (op 20757) 

v_lshlrev_b32 v26, 5, v27 // (op 20757) 

v_lshrrev_b32 v27, 2, v1 // (op 20757) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 20757) 


v_lshlrev_b32 v26, 2, v28 // (op 20757) 

v_lshrrev_b32 v27, 3, v26 // (op 20757) 

s_nop 0
v_readfirstlane_b32 s47, v27 // (op 20757) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 20456) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 20456) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 20456) 


v_mov_b64 v[28:29], v[20:21] // (op 20456) 
s_mov_b32 m0, s47 // (op 20456) reg expression
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 20456) Load value direct to lds
v_add_co_u32 v28, s[0:1], v28, v26 // (op 20456) least significant half
v_addc_co_u32 v29, s[0:1], 0, v29, s[0:1] // (op 20456) most significant half

s_add_u32 m0, m0, 4096 // (op 20456) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 20456) Load value direct to lds
v_add_co_u32 v28, s[0:1], v28, v26 // (op 20456) least significant half
v_addc_co_u32 v29, s[0:1], 0, v29, s[0:1] // (op 20456) most significant half

s_add_u32 m0, m0, 4096 // (op 20456) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 20456) Load value direct to lds
v_add_co_u32 v28, s[0:1], v28, v26 // (op 20456) least significant half
v_addc_co_u32 v29, s[0:1], 0, v29, s[0:1] // (op 20456) most significant half

s_add_u32 m0, m0, 4096 // (op 20456) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 20456) Load value direct to lds



v_and_b32 v26, -4, v1 // (op 20700) 
v_sub_u32 v27, v1, v26 // (op 20700) 

v_lshlrev_b32 v26, 5, v27 // (op 20700) 

v_lshrrev_b32 v27, 2, v1 // (op 20700) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 20700) 


v_lshlrev_b32 v26, 2, v28 // (op 20700) 

v_lshrrev_b32 v27, 3, v26 // (op 20700) 

s_nop 0
v_readfirstlane_b32 s47, v27 // (op 20700) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 20461) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 20461) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 20461) 


v_mov_b64 v[28:29], v[16:17] // (op 20461) 
s_add_u32 s0, 16384, s47 // (op 20461) 
s_mov_b32 m0, s0 // (op 20461) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 20461) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 20461) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 20461) most significant half

s_add_u32 m0, m0, 4096 // (op 20461) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 20461) Load value direct to lds






v_mov_b32 v26, v14 // (op 76) reg expression

buffer_load_dwordx4 v[28:31], v26, s[68:71], 0 offen // (op 76) Load value



s_lshl_b32 s47, s38, 6 // (op 1104) 
v_add_u32 v26, v14, s47 // (op 1104) 


buffer_load_dwordx4 v[32:35], v26, s[68:71], 0 offen // (op 1104) Load value




v_mov_b32 v14, v24 // (op 72) reg expression

buffer_load_dwordx4 v[36:39], v14, s[64:67], 0 offen // (op 72) Load value


v_and_b32 v14, -4, v1 // (op 20868) 
v_sub_u32 v24, v1, v14 // (op 20868) 

v_lshlrev_b32 v14, 5, v24 // (op 20868) 

v_lshrrev_b32 v24, 2, v1 // (op 20868) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 20868) 


v_lshlrev_b32 v14, 2, v26 // (op 20868) 

v_lshrrev_b32 v24, 3, v14 // (op 20868) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 20868) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 20466) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 20466) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 20466) 


v_mov_b64 v[40:41], v[20:21] // (op 20466) 
s_add_u32 s0, 24576, s47 // (op 20466) 
s_mov_b32 m0, s0 // (op 20466) reg expression
buffer_load_dwordx4 v40, s[60:63], 64 offen offset: 0 lds // (op 20466) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20466) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20466) most significant half

s_add_u32 m0, m0, 4096 // (op 20466) 
buffer_load_dwordx4 v40, s[60:63], 64 offen offset: 0 lds // (op 20466) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20466) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20466) most significant half

s_add_u32 m0, m0, 4096 // (op 20466) 
buffer_load_dwordx4 v40, s[60:63], 64 offen offset: 0 lds // (op 20466) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20466) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20466) most significant half

s_add_u32 m0, m0, 4096 // (op 20466) 
buffer_load_dwordx4 v40, s[60:63], 64 offen offset: 0 lds // (op 20466) Load value direct to lds




v_and_b32 v14, -4, v1 // (op 20824) 
v_sub_u32 v24, v1, v14 // (op 20824) 

v_lshlrev_b32 v14, 5, v24 // (op 20824) 

v_lshrrev_b32 v24, 2, v1 // (op 20824) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 20824) 


v_lshlrev_b32 v14, 2, v26 // (op 20824) 

v_lshrrev_b32 v24, 3, v14 // (op 20824) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 20824) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 20468) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 20468) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 20468) 


v_mov_b64 v[40:41], v[16:17] // (op 20468) 
s_add_u32 s0, 40960, s47 // (op 20468) 
s_mov_b32 m0, s0 // (op 20468) reg expression
buffer_load_dwordx4 v40, s[48:51], 64 offen offset: 0 lds // (op 20468) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20468) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20468) most significant half

s_add_u32 m0, m0, 4096 // (op 20468) 
buffer_load_dwordx4 v40, s[48:51], 64 offen offset: 0 lds // (op 20468) Load value direct to lds




v_and_b32 v14, -4, v1 // (op 20956) 
v_sub_u32 v24, v1, v14 // (op 20956) 

v_lshlrev_b32 v14, 5, v24 // (op 20956) 

v_lshrrev_b32 v24, 2, v1 // (op 20956) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 20956) 


v_lshlrev_b32 v14, 2, v26 // (op 20956) 

v_lshrrev_b32 v24, 3, v14 // (op 20956) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 20956) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 20470) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 20470) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 20470) 


v_mov_b64 v[40:41], v[20:21] // (op 20470) 
s_add_u32 s0, 49152, s47 // (op 20470) 
s_mov_b32 m0, s0 // (op 20470) reg expression
s_mov_b32 s1, 128 // (op 20470) call()
buffer_load_dwordx4 v40, s[60:63], s1 offen offset: 0 lds // (op 20470) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20470) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20470) most significant half

s_add_u32 m0, m0, 4096 // (op 20470) 
buffer_load_dwordx4 v40, s[60:63], s1 offen offset: 0 lds // (op 20470) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20470) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20470) most significant half

s_add_u32 m0, m0, 4096 // (op 20470) 
buffer_load_dwordx4 v40, s[60:63], s1 offen offset: 0 lds // (op 20470) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20470) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20470) most significant half

s_add_u32 m0, m0, 4096 // (op 20470) 
buffer_load_dwordx4 v40, s[60:63], s1 offen offset: 0 lds // (op 20470) Load value direct to lds





v_and_b32 v14, -4, v1 // (op 20912) 
v_sub_u32 v24, v1, v14 // (op 20912) 

v_lshlrev_b32 v14, 5, v24 // (op 20912) 

v_lshrrev_b32 v24, 2, v1 // (op 20912) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 20912) 


v_lshlrev_b32 v14, 2, v26 // (op 20912) 

v_lshrrev_b32 v24, 3, v14 // (op 20912) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 20912) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 20472) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 20472) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 20472) 


v_mov_b64 v[40:41], v[16:17] // (op 20472) 
s_add_u32 s0, 65536, s47 // (op 20472) 
s_mov_b32 m0, s0 // (op 20472) reg expression
s_mov_b32 s1, 128 // (op 20472) call()
buffer_load_dwordx4 v40, s[48:51], s1 offen offset: 0 lds // (op 20472) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20472) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20472) most significant half

s_add_u32 m0, m0, 4096 // (op 20472) 
buffer_load_dwordx4 v40, s[48:51], s1 offen offset: 0 lds // (op 20472) Load value direct to lds





v_and_b32 v14, -4, v1 // (op 21044) 
v_sub_u32 v24, v1, v14 // (op 21044) 

v_lshlrev_b32 v14, 5, v24 // (op 21044) 

v_lshrrev_b32 v24, 2, v1 // (op 21044) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 21044) 


v_lshlrev_b32 v14, 2, v26 // (op 21044) 

v_lshrrev_b32 v24, 3, v14 // (op 21044) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 21044) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 20474) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 20474) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 20474) 


v_mov_b64 v[40:41], v[20:21] // (op 20474) 
s_add_u32 s0, 73728, s47 // (op 20474) 
s_mov_b32 m0, s0 // (op 20474) reg expression
s_mov_b32 s1, 192 // (op 20474) call()
buffer_load_dwordx4 v40, s[60:63], s1 offen offset: 0 lds // (op 20474) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20474) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20474) most significant half

s_add_u32 m0, m0, 4096 // (op 20474) 
buffer_load_dwordx4 v40, s[60:63], s1 offen offset: 0 lds // (op 20474) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20474) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20474) most significant half

s_add_u32 m0, m0, 4096 // (op 20474) 
buffer_load_dwordx4 v40, s[60:63], s1 offen offset: 0 lds // (op 20474) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20474) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20474) most significant half

s_add_u32 m0, m0, 4096 // (op 20474) 
buffer_load_dwordx4 v40, s[60:63], s1 offen offset: 0 lds // (op 20474) Load value direct to lds






v_and_b32 v14, -4, v1 // (op 21000) 
v_sub_u32 v24, v1, v14 // (op 21000) 

v_lshlrev_b32 v14, 5, v24 // (op 21000) 

v_lshrrev_b32 v24, 2, v1 // (op 21000) 
v_lshl_add_u32 v20, v24, 7, v14 // (op 21000) 


v_lshlrev_b32 v14, 2, v20 // (op 21000) 

v_lshrrev_b32 v24, 3, v14 // (op 21000) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 21000) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 20476) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 20476) 

v_lshrrev_b64 v[20:21], 3, s[18:19] // (op 20476) 


v_mov_b64 v[26:27], v[16:17] // (op 20476) 
s_add_u32 s0, 90112, s47 // (op 20476) 
s_mov_b32 m0, s0 // (op 20476) reg expression
s_mov_b32 s1, 192 // (op 20476) call()
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 20476) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v20 // (op 20476) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 20476) most significant half

s_add_u32 m0, m0, 4096 // (op 20476) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 20476) Load value direct to lds






s_waitcnt vmcnt(23)
s_barrier  // (op 21601) 



ds_read_b128 v[40:43], v4 // (op 19764) Load local data 



ds_read_b128 v[44:47], v4 offset:1024 // (op 19769) Load local data 



ds_read_b128 v[48:51], v4 offset:2048 // (op 19774) Load local data 



ds_read_b128 v[52:55], v4 offset:3072 // (op 19779) Load local data 



ds_read_b128 v[56:59], v4 offset:4096 // (op 19784) Load local data 



ds_read_b128 v[60:63], v4 offset:5120 // (op 19789) Load local data 



ds_read_b128 v[64:67], v4 offset:6144 // (op 19794) Load local data 



ds_read_b128 v[68:71], v4 offset:7168 // (op 19799) Load local data 
s_waitcnt vmcnt(21)
s_barrier  // (op 21584) 



ds_read_b128 v[72:75], v3 offset:16384 // (op 19744) Load local data 



ds_read_b128 v[76:79], v3 offset:17408 // (op 19749) Load local data 



ds_read_b128 v[80:83], v3 offset:18432 // (op 19754) Load local data 



ds_read_b128 v[84:87], v3 offset:19456 // (op 19759) Load local data 
v_and_b32 v14, -4, v1 // (op 20672) 
v_sub_u32 v24, v1, v14 // (op 20672) 

v_lshlrev_b32 v14, 5, v24 // (op 20672) 

v_lshrrev_b32 v24, 2, v1 // (op 20672) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20672) 


v_lshlrev_b32 v14, 2, v16 // (op 20672) 

v_lshrrev_b32 v24, 3, v14 // (op 20672) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 20672) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20729) 
v_sub_u32 v24, v1, v14 // (op 20729) 

v_lshlrev_b32 v14, 5, v24 // (op 20729) 

v_lshrrev_b32 v24, 2, v1 // (op 20729) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20729) 


v_lshlrev_b32 v14, 2, v16 // (op 20729) 

v_lshrrev_b32 v24, 3, v14 // (op 20729) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 20729) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20809) 
v_sub_u32 v24, v1, v14 // (op 20809) 

v_lshlrev_b32 v14, 5, v24 // (op 20809) 

v_lshrrev_b32 v24, 2, v1 // (op 20809) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20809) 


v_lshlrev_b32 v14, 2, v16 // (op 20809) 

v_lshrrev_b32 v24, 3, v14 // (op 20809) 

s_nop 0
v_readfirstlane_b32 s1, v24 // (op 20809) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20853) 
v_sub_u32 v24, v1, v14 // (op 20853) 

v_lshlrev_b32 v14, 5, v24 // (op 20853) 

v_lshrrev_b32 v24, 2, v1 // (op 20853) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20853) 


v_lshlrev_b32 v14, 2, v16 // (op 20853) 

v_lshrrev_b32 v24, 3, v14 // (op 20853) 

s_nop 0
v_readfirstlane_b32 s18, v24 // (op 20853) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20897) 
v_sub_u32 v24, v1, v14 // (op 20897) 

v_lshlrev_b32 v14, 5, v24 // (op 20897) 

v_lshrrev_b32 v24, 2, v1 // (op 20897) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20897) 


v_lshlrev_b32 v14, 2, v16 // (op 20897) 

v_lshrrev_b32 v24, 3, v14 // (op 20897) 

s_nop 0
v_readfirstlane_b32 s19, v24 // (op 20897) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20941) 
v_sub_u32 v24, v1, v14 // (op 20941) 

v_lshlrev_b32 v14, 5, v24 // (op 20941) 

v_lshrrev_b32 v24, 2, v1 // (op 20941) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20941) 


v_lshlrev_b32 v14, 2, v16 // (op 20941) 

v_lshrrev_b32 v24, 3, v14 // (op 20941) 

s_nop 0
v_readfirstlane_b32 s22, v24 // (op 20941) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20985) 
v_sub_u32 v24, v1, v14 // (op 20985) 

v_lshlrev_b32 v14, 5, v24 // (op 20985) 

v_lshrrev_b32 v24, 2, v1 // (op 20985) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20985) 


v_lshlrev_b32 v14, 2, v16 // (op 20985) 

v_lshrrev_b32 v24, 3, v14 // (op 20985) 

s_nop 0
v_readfirstlane_b32 s23, v24 // (op 20985) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 21029) 
v_sub_u32 v24, v1, v14 // (op 21029) 

v_lshlrev_b32 v14, 5, v24 // (op 21029) 

v_lshrrev_b32 v24, 2, v1 // (op 21029) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 21029) 


v_lshlrev_b32 v14, 2, v16 // (op 21029) 

v_lshrrev_b32 v24, 3, v14 // (op 21029) 

s_nop 0
v_readfirstlane_b32 s24, v24 // (op 21029) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s25, 0 // (op 4434) call()
s_ashr_i64 s[72:73], s[42:43], 7 // (op 4433) 
s_mov_b32 s74, s72 // (op 4433) convert

s_and_b32 s72, s74, 4294967292 // (op 4433) 

s_cmp_lt_u32 s25, s72 // (op 4433) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__ForLoopBottom_KLoop_4433 // (op 4433) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__ForLoopBottom_KLoop_4433 if false)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__ForLoopTop_KLoop_4433:
s_add_u32 s25, s25, 4 // (op 4435) 
s_waitcnt vmcnt(20)
v_mov_b32 v88, v28 // (op 20335) reg expression
v_mov_b32 v89, v29 // (op 20335) reg expression
v_mov_b32 v90, v30 // (op 20335) reg expression
v_mov_b32 v91, v31 // (op 20335) reg expression
s_waitcnt lgkmcnt(4)
s_barrier  // (op 21604) 
s_waitcnt vmcnt(19)
v_mov_b32 v92, v32 // (op 20348) reg expression

v_mov_b32 v93, v33 // (op 20348) reg expression
v_mov_b32 v94, v34 // (op 20348) reg expression
v_mov_b32 v95, v35 // (op 20348) reg expression
s_lshl_b64 s[72:73], s[30:31], 6 // (op 20409) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 20409) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 20409) 


v_mov_b64 v[20:21], v[22:23] // (op 20409) 
v_mov_b64 v[26:27], v[20:21] // (op 20409) 
s_mov_b32 m0, s0 // (op 20409) reg expression
s_mov_b32 s72, 256 // (op 20409) call()
s_waitcnt vmcnt(18)
v_mov_b32 v96, v36 // (op 20361) reg expression
v_mov_b32 v97, v37 // (op 20361) reg expression
v_mov_b32 v98, v38 // (op 20361) reg expression
v_mov_b32 v99, v39 // (op 20361) reg expression
buffer_load_dwordx4 v26, s[60:63], s72 offen offset: 0 lds // (op 20409) Load value direct to lds
v_add_co_u32 v26, s[74:75], v26, v16 // (op 20409) least significant half
v_addc_co_u32 v27, s[74:75], 0, v27, s[74:75] // (op 20409) most significant half

s_add_u32 m0, m0, 4096 // (op 20409) 
v_bfe_u32 v100, v96, 0, 8 // (op 18523) 
v_bfe_u32 v101, v96, 8, 8 // (op 18523) 
v_bfe_u32 v102, v96, 16, 8 // (op 18523) 
v_bfe_u32 v103, v96, 24, 8 // (op 18523) 
v_permlane16_swap_b32 v100, v101 // (op 18523) 
s_nop 0
v_permlane16_swap_b32 v102, v103 // (op 18523) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v100, v102 // (op 18523) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v101, v103 // (op 18523) 
buffer_load_dwordx4 v26, s[60:63], s72 offen offset: 0 lds // (op 20409) Load value direct to lds
v_add_co_u32 v26, s[74:75], v26, v16 // (op 20409) least significant half
v_addc_co_u32 v27, s[74:75], 0, v27, s[74:75] // (op 20409) most significant half

s_add_u32 m0, m0, 4096 // (op 20409) 
buffer_load_dwordx4 v26, s[60:63], s72 offen offset: 0 lds // (op 20409) Load value direct to lds
v_add_co_u32 v26, s[74:75], v26, v16 // (op 20409) least significant half
v_addc_co_u32 v27, s[74:75], 0, v27, s[74:75] // (op 20409) most significant half

s_add_u32 m0, m0, 4096 // (op 20409) 
v_bfe_u32 v104, v88, 0, 8 // (op 18716) 
v_bfe_u32 v105, v88, 8, 8 // (op 18716) 
v_bfe_u32 v106, v88, 16, 8 // (op 18716) 
v_bfe_u32 v107, v88, 24, 8 // (op 18716) 
v_permlane16_swap_b32 v104, v105 // (op 18716) 
s_nop 0
v_permlane16_swap_b32 v106, v107 // (op 18716) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v104, v106 // (op 18716) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v105, v107 // (op 18716) 
buffer_load_dwordx4 v26, s[60:63], s72 offen offset: 0 lds // (op 20409) Load value direct to lds



s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:75], v[40:43], a[0:3], v100, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 78) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 21587) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[44:47], a[4:7], v100, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 421) 
s_lshl_b64 s[72:73], s[26:27], 6 // (op 20404) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:75], v[48:51], a[8:11], v100, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 647) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 20404) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:75], v[52:55], a[12:15], v100, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 873) 
v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 20404) 


v_mov_b64 v[26:27], v[18:19] // (op 20404) 
v_mov_b64 v[108:109], v[26:27] // (op 20404) 
s_add_u32 s72, 16384, s47 // (op 20404) 
s_mov_b32 m0, s72 // (op 20404) reg expression
s_mov_b32 s73, 256 // (op 20404) call()
buffer_load_dwordx4 v108, s[48:51], s73 offen offset: 0 lds // (op 20404) Load value direct to lds
v_add_co_u32 v108, s[74:75], v108, v16 // (op 20404) least significant half
v_addc_co_u32 v109, s[74:75], 0, v109, s[74:75] // (op 20404) most significant half

s_add_u32 m0, m0, 4096 // (op 20404) 
buffer_load_dwordx4 v108, s[48:51], s73 offen offset: 0 lds // (op 20404) Load value direct to lds




v_bfe_u32 v108, v92, 0, 8 // (op 18784) 


v_bfe_u32 v109, v92, 8, 8 // (op 18784) 
v_bfe_u32 v110, v92, 16, 8 // (op 18784) 
v_bfe_u32 v111, v92, 24, 8 // (op 18784) 
v_permlane16_swap_b32 v108, v109 // (op 18784) 
v_mov_b32 v14, v15 // (op 20343) 
v_permlane16_swap_b32 v110, v111 // (op 18784) 
v_add_u32 v24, 16, v14 // (op 20343) 

s_nop 0
v_permlane32_swap_b32 v108, v110 // (op 18784) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v109, v111 // (op 18784) 
buffer_load_dwordx4 v[28:31], v24, s[68:71], 0 offen // (op 20343) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:75], v[56:59], a[16:19], v100, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1099) 
s_lshl_b32 s72, s38, 6 // (op 20356) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:75], v[60:63], a[20:23], v100, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1325) 
s_add_u32 s72, s72, 16 // (op 20356) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:75], v[64:67], a[24:27], v100, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1551) 
v_add_u32 v24, v14, s72 // (op 20356) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:75], v[68:71], a[28:31], v100, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1777) 
buffer_load_dwordx4 v[32:35], v24, s[68:71], 0 offen // (op 20356) Load value




v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[40:43], a[32:35], v101, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2997) 
v_add_u32 v15, 16, v15 // (op 21146) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[44:47], a[36:39], v101, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2921) 
v_mov_b32 v14, v25 // (op 20369) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:79], v[48:51], a[40:43], v101, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2845) 
v_add_u32 v24, 16, v14 // (op 20369) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[52:55], a[44:47], v101, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2769) 
buffer_load_dwordx4 v[36:39], v24, s[64:67], 0 offen // (op 20369) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[76:79], v[56:59], a[48:51], v101, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2693) 
v_add_u32 v25, 16, v25 // (op 21094) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:79], v[60:63], a[52:55], v101, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2617) 
s_waitcnt vmcnt(23)
s_barrier  // (op 21624) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[76:79], v[64:67], a[56:59], v101, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2541) 
ds_read_b128 v[112:115], v6 offset:24576 // (op 216) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[76:79], v[68:71], a[60:63], v101, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2465) 
ds_read_b128 v[116:119], v6 offset:25600 // (op 432) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[80:83], v[40:43], a[64:67], v102, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4999) 
ds_read_b128 v[120:123], v6 offset:26624 // (op 658) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[80:83], v[44:47], a[68:71], v102, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4923) 
ds_read_b128 v[124:127], v6 offset:27648 // (op 884) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[80:83], v[48:51], a[72:75], v102, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4847) 
ds_read_b128 v[128:131], v6 offset:28672 // (op 1110) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[80:83], v[52:55], a[76:79], v102, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4771) 
ds_read_b128 v[132:135], v6 offset:29696 // (op 1336) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[80:83], v[56:59], a[80:83], v102, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4695) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[80:83], v[60:63], a[84:87], v102, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4619) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[80:83], v[64:67], a[88:91], v102, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4543) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[80:83], v[68:71], a[92:95], v102, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4467) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[84:87], v[40:43], a[96:99], v103, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7001) 
ds_read_b128 v[136:139], v6 offset:30720 // (op 1562) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:87], v[44:47], a[100:103], v103, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6925) 
ds_read_b128 v[140:143], v6 offset:31744 // (op 1788) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[84:87], v[48:51], a[104:107], v103, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6849) 
s_waitcnt vmcnt(21)
s_barrier  // (op 21617) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[84:87], v[52:55], a[108:111], v103, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6773) 
ds_read_b128 v[72:75], v5 offset:40960 // (op 214) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[84:87], v[56:59], a[112:115], v103, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6697) 
ds_read_b128 v[76:79], v5 offset:41984 // (op 2475) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[84:87], v[60:63], a[116:119], v103, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6621) 
ds_read_b128 v[80:83], v5 offset:43008 // (op 4477) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[84:87], v[64:67], a[120:123], v103, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6545) 
ds_read_b128 v[104:107], v5 offset:44032 // (op 6479) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[84:87], v[68:71], a[124:127], v103, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6469) 


v_bfe_u32 v100, v97, 0, 8 // (op 18538) 
v_bfe_u32 v101, v97, 8, 8 // (op 18538) 
v_bfe_u32 v102, v97, 16, 8 // (op 18538) 
v_bfe_u32 v103, v97, 24, 8 // (op 18538) 
v_permlane16_swap_b32 v100, v101 // (op 18538) 
s_nop 0
v_permlane16_swap_b32 v102, v103 // (op 18538) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v100, v102 // (op 18538) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v101, v103 // (op 18538) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 21628) 

s_lshl_b64 s[72:73], s[30:31], 6 // (op 20431) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 20431) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 20431) 


v_mov_b64 v[108:109], v[20:21] // (op 20431) 
s_add_u32 s72, 24576, s18 // (op 20431) 
s_mov_b32 m0, s72 // (op 20431) reg expression
s_mov_b32 s73, 320 // (op 20431) call()
v_bfe_u32 v144, v89, 0, 8 // (op 18731) 
v_bfe_u32 v145, v89, 8, 8 // (op 18731) 
v_bfe_u32 v146, v89, 16, 8 // (op 18731) 
v_bfe_u32 v147, v89, 24, 8 // (op 18731) 
v_permlane16_swap_b32 v144, v145 // (op 18731) 
s_nop 0
v_permlane16_swap_b32 v146, v147 // (op 18731) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v144, v146 // (op 18731) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v145, v147 // (op 18731) 
buffer_load_dwordx4 v108, s[60:63], s73 offen offset: 0 lds // (op 20431) Load value direct to lds
v_add_co_u32 v108, s[74:75], v108, v16 // (op 20431) least significant half
v_addc_co_u32 v109, s[74:75], 0, v109, s[74:75] // (op 20431) most significant half

s_add_u32 m0, m0, 4096 // (op 20431) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:75], v[112:115], a[0:3], v100, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 221) 
buffer_load_dwordx4 v108, s[60:63], s73 offen offset: 0 lds // (op 20431) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[116:119], a[4:7], v100, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 422) 
v_add_co_u32 v108, s[74:75], v108, v16 // (op 20431) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:75], v[120:123], a[8:11], v100, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 648) 
v_addc_co_u32 v109, s[74:75], 0, v109, s[74:75] // (op 20431) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:75], v[124:127], a[12:15], v100, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 874) 
s_add_u32 m0, m0, 4096 // (op 20431) 
v_bfe_u32 v148, v93, 0, 8 // (op 18799) 
v_bfe_u32 v149, v93, 8, 8 // (op 18799) 
v_bfe_u32 v150, v93, 16, 8 // (op 18799) 
v_bfe_u32 v151, v93, 24, 8 // (op 18799) 
buffer_load_dwordx4 v108, s[60:63], s73 offen offset: 0 lds // (op 20431) Load value direct to lds
v_permlane16_swap_b32 v148, v149 // (op 18799) 
v_permlane16_swap_b32 v150, v151 // (op 18799) 
v_add_co_u32 v108, s[74:75], v108, v16 // (op 20431) least significant half
v_addc_co_u32 v109, s[74:75], 0, v109, s[74:75] // (op 20431) most significant half

s_add_u32 m0, m0, 4096 // (op 20431) 
v_permlane32_swap_b32 v148, v150 // (op 18799) 
v_permlane32_swap_b32 v149, v151 // (op 18799) 
buffer_load_dwordx4 v108, s[60:63], s73 offen offset: 0 lds // (op 20431) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:75], v[128:131], a[16:19], v100, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1100) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 21621) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:75], v[132:135], a[20:23], v100, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1326) 
s_lshl_b64 s[72:73], s[26:27], 6 // (op 20426) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:75], v[136:139], a[24:27], v100, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1552) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 20426) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:75], v[140:143], a[28:31], v100, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1778) 
v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 20426) 


v_mov_b64 v[108:109], v[26:27] // (op 20426) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[112:115], a[32:35], v101, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2998) 
s_add_u32 s72, 40960, s1 // (op 20426) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[116:119], a[36:39], v101, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2922) 
s_mov_b32 m0, s72 // (op 20426) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:79], v[120:123], a[40:43], v101, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2846) 
s_mov_b32 s73, 320 // (op 20426) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[124:127], a[44:47], v101, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2770) 
buffer_load_dwordx4 v108, s[48:51], s73 offen offset: 0 lds // (op 20426) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[76:79], v[128:131], a[48:51], v101, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2694) 
v_add_co_u32 v108, s[74:75], v108, v16 // (op 20426) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:79], v[132:135], a[52:55], v101, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2618) 
v_addc_co_u32 v109, s[74:75], 0, v109, s[74:75] // (op 20426) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[76:79], v[136:139], a[56:59], v101, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2542) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[76:79], v[140:143], a[60:63], v101, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2466) 
s_add_u32 m0, m0, 4096 // (op 20426) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[80:83], v[112:115], a[64:67], v102, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5000) 
buffer_load_dwordx4 v108, s[48:51], s73 offen offset: 0 lds // (op 20426) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[80:83], v[116:119], a[68:71], v102, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4924) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[80:83], v[120:123], a[72:75], v102, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4848) 
s_waitcnt vmcnt(23)
s_barrier  // (op 21639) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[80:83], v[124:127], a[76:79], v102, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4772) 
ds_read_b128 v[40:43], v8 offset:49152 // (op 244) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[80:83], v[128:131], a[80:83], v102, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4696) 
ds_read_b128 v[44:47], v8 offset:50176 // (op 442) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[80:83], v[132:135], a[84:87], v102, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4620) 
ds_read_b128 v[48:51], v8 offset:51200 // (op 668) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[80:83], v[136:139], a[88:91], v102, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4544) 
ds_read_b128 v[52:55], v8 offset:52224 // (op 894) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[80:83], v[140:143], a[92:95], v102, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4468) 
ds_read_b128 v[56:59], v8 offset:53248 // (op 1120) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[104:107], v[112:115], a[96:99], v103, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7002) 
ds_read_b128 v[60:63], v8 offset:54272 // (op 1346) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[104:107], v[116:119], a[100:103], v103, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6926) 
ds_read_b128 v[64:67], v8 offset:55296 // (op 1572) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:107], v[120:123], a[104:107], v103, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6850) 
ds_read_b128 v[108:111], v8 offset:56320 // (op 1798) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[124:127], a[108:111], v103, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6774) 
s_waitcnt vmcnt(21)
s_barrier  // (op 21632) 





v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[104:107], v[128:131], a[112:115], v103, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6698) 
v_add_u32 v14, 65536, v7 // (op 242) 

ds_read_b128 v[72:75], v14 // (op 242) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[104:107], v[132:135], a[116:119], v103, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6622) 
v_add_u32 v14, 66560, v7 // (op 2485) 

ds_read_b128 v[76:79], v14 // (op 2485) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[104:107], v[136:139], a[120:123], v103, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6546) 
v_add_u32 v14, 67584, v7 // (op 4487) 

ds_read_b128 v[80:83], v14 // (op 4487) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[104:107], v[140:143], a[124:127], v103, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6470) 
v_add_u32 v14, 68608, v7 // (op 6489) 




ds_read_b128 v[84:87], v14 // (op 6489) Load local data 

v_bfe_u32 v100, v98, 0, 8 // (op 18555) 
v_bfe_u32 v101, v98, 8, 8 // (op 18555) 
v_bfe_u32 v102, v98, 16, 8 // (op 18555) 
v_bfe_u32 v103, v98, 24, 8 // (op 18555) 
v_permlane16_swap_b32 v100, v101 // (op 18555) 
s_nop 0
v_permlane16_swap_b32 v102, v103 // (op 18555) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v100, v102 // (op 18555) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v101, v103 // (op 18555) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 21642) 

s_lshl_b64 s[72:73], s[30:31], 6 // (op 20441) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 20441) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 20441) 


v_mov_b64 v[104:105], v[20:21] // (op 20441) 
s_add_u32 s72, 49152, s22 // (op 20441) 
s_mov_b32 m0, s72 // (op 20441) reg expression
s_mov_b32 s73, 384 // (op 20441) call()
v_bfe_u32 v112, v90, 0, 8 // (op 18749) 
v_bfe_u32 v113, v90, 8, 8 // (op 18749) 
v_bfe_u32 v114, v90, 16, 8 // (op 18749) 
v_bfe_u32 v115, v90, 24, 8 // (op 18749) 
v_permlane16_swap_b32 v112, v113 // (op 18749) 
s_nop 0
v_permlane16_swap_b32 v114, v115 // (op 18749) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v112, v114 // (op 18749) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v113, v115 // (op 18749) 
buffer_load_dwordx4 v104, s[60:63], s73 offen offset: 0 lds // (op 20441) Load value direct to lds
v_add_co_u32 v104, s[74:75], v104, v16 // (op 20441) least significant half
v_addc_co_u32 v105, s[74:75], 0, v105, s[74:75] // (op 20441) most significant half

s_add_u32 m0, m0, 4096 // (op 20441) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:75], v[40:43], a[0:3], v100, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 249) 
buffer_load_dwordx4 v104, s[60:63], s73 offen offset: 0 lds // (op 20441) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[44:47], a[4:7], v100, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 423) 
v_add_co_u32 v104, s[74:75], v104, v16 // (op 20441) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:75], v[48:51], a[8:11], v100, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 649) 
v_addc_co_u32 v105, s[74:75], 0, v105, s[74:75] // (op 20441) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:75], v[52:55], a[12:15], v100, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 875) 
s_add_u32 m0, m0, 4096 // (op 20441) 
v_bfe_u32 v116, v94, 0, 8 // (op 18817) 
v_bfe_u32 v117, v94, 8, 8 // (op 18817) 
v_bfe_u32 v118, v94, 16, 8 // (op 18817) 
v_bfe_u32 v119, v94, 24, 8 // (op 18817) 
v_permlane16_swap_b32 v116, v117 // (op 18817) 
s_nop 0
v_permlane16_swap_b32 v118, v119 // (op 18817) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v116, v118 // (op 18817) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v117, v119 // (op 18817) 
buffer_load_dwordx4 v104, s[60:63], s73 offen offset: 0 lds // (op 20441) Load value direct to lds
v_add_co_u32 v104, s[74:75], v104, v16 // (op 20441) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:75], v[56:59], a[16:19], v100, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1101) 
v_addc_co_u32 v105, s[74:75], 0, v105, s[74:75] // (op 20441) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:75], v[60:63], a[20:23], v100, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1327) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:75], v[64:67], a[24:27], v100, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1553) 
s_add_u32 m0, m0, 4096 // (op 20441) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:75], v[108:111], a[28:31], v100, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1779) 
buffer_load_dwordx4 v104, s[60:63], s73 offen offset: 0 lds // (op 20441) Load value direct to lds




s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[40:43], a[32:35], v101, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2999) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 21636) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[44:47], a[36:39], v101, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2923) 
s_lshl_b64 s[72:73], s[26:27], 6 // (op 20436) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:79], v[48:51], a[40:43], v101, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2847) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 20436) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[52:55], a[44:47], v101, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2771) 
v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 20436) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[76:79], v[56:59], a[48:51], v101, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2695) 
v_mov_b64 v[104:105], v[26:27] // (op 20436) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:79], v[60:63], a[52:55], v101, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2619) 
s_add_u32 s72, 65536, s19 // (op 20436) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[76:79], v[64:67], a[56:59], v101, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2543) 
s_mov_b32 m0, s72 // (op 20436) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[76:79], v[108:111], a[60:63], v101, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2467) 
s_mov_b32 s73, 384 // (op 20436) call()
buffer_load_dwordx4 v104, s[48:51], s73 offen offset: 0 lds // (op 20436) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[80:83], v[40:43], a[64:67], v102, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5001) 
v_add_co_u32 v104, s[74:75], v104, v16 // (op 20436) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[80:83], v[44:47], a[68:71], v102, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4925) 
v_addc_co_u32 v105, s[74:75], 0, v105, s[74:75] // (op 20436) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[80:83], v[48:51], a[72:75], v102, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4849) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[80:83], v[52:55], a[76:79], v102, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4773) 
s_add_u32 m0, m0, 4096 // (op 20436) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[80:83], v[56:59], a[80:83], v102, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4697) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[80:83], v[60:63], a[84:87], v102, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4621) 
buffer_load_dwordx4 v104, s[48:51], s73 offen offset: 0 lds // (op 20436) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[80:83], v[64:67], a[88:91], v102, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4545) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[80:83], v[108:111], a[92:95], v102, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4469) 
s_waitcnt vmcnt(23)
s_barrier  // (op 21652) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[84:87], v[40:43], a[96:99], v103, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7003) 
v_add_u32 v14, 73728, v10 // (op 272) 
ds_read_b128 v[68:71], v14 // (op 272) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:87], v[44:47], a[100:103], v103, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6927) 
v_add_u32 v14, 74752, v10 // (op 452) 
ds_read_b128 v[104:107], v14 // (op 452) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[84:87], v[48:51], a[104:107], v103, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6851) 
v_add_u32 v14, 75776, v10 // (op 678) 
ds_read_b128 v[120:123], v14 // (op 678) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[84:87], v[52:55], a[108:111], v103, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6775) 
v_add_u32 v14, 76800, v10 // (op 904) 

ds_read_b128 v[112:115], v14 // (op 904) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[84:87], v[56:59], a[112:115], v103, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6699) 
v_add_u32 v14, 77824, v10 // (op 1130) 
ds_read_b128 v[124:127], v14 // (op 1130) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[84:87], v[60:63], a[116:119], v103, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6623) 
v_add_u32 v14, 78848, v10 // (op 1356) 
ds_read_b128 v[128:131], v14 // (op 1356) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[84:87], v[64:67], a[120:123], v103, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6547) 
v_add_u32 v14, 79872, v10 // (op 1582) 
ds_read_b128 v[132:135], v14 // (op 1582) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[84:87], v[108:111], a[124:127], v103, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6471) 
v_add_u32 v14, 80896, v10 // (op 1808) 



ds_read_b128 v[100:103], v14 // (op 1808) Load local data 

s_waitcnt vmcnt(21)
s_barrier  // (op 21645) 



v_add_u32 v14, 90112, v9 // (op 270) 
ds_read_b128 v[72:75], v14 // (op 270) Load local data 




v_add_u32 v14, 91136, v9 // (op 2495) 
ds_read_b128 v[76:79], v14 // (op 2495) Load local data 




v_add_u32 v14, 92160, v9 // (op 4497) 
ds_read_b128 v[80:83], v14 // (op 4497) Load local data 




v_add_u32 v14, 93184, v9 // (op 6499) 
ds_read_b128 v[108:111], v14 // (op 6499) Load local data 

v_bfe_u32 v116, v99, 0, 8 // (op 18572) 
v_bfe_u32 v117, v99, 8, 8 // (op 18572) 
v_bfe_u32 v118, v99, 16, 8 // (op 18572) 
v_bfe_u32 v119, v99, 24, 8 // (op 18572) 
v_permlane16_swap_b32 v116, v117 // (op 18572) 
s_nop 0
v_permlane16_swap_b32 v118, v119 // (op 18572) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v116, v118 // (op 18572) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v117, v119 // (op 18572) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 21655) 


s_lshl_b64 s[72:73], s[30:31], 6 // (op 20451) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 20451) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 20451) 


v_mov_b64 v[96:97], v[20:21] // (op 20451) 
s_add_u32 s72, 73728, s24 // (op 20451) 
s_mov_b32 m0, s72 // (op 20451) reg expression
s_mov_b32 s73, 448 // (op 20451) call()
v_bfe_u32 v136, v91, 0, 8 // (op 18767) 
v_bfe_u32 v137, v91, 8, 8 // (op 18767) 
v_bfe_u32 v138, v91, 16, 8 // (op 18767) 
v_bfe_u32 v139, v91, 24, 8 // (op 18767) 
v_permlane16_swap_b32 v136, v137 // (op 18767) 
s_nop 0
v_permlane16_swap_b32 v138, v139 // (op 18767) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v136, v138 // (op 18767) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v137, v139 // (op 18767) 
buffer_load_dwordx4 v96, s[60:63], s73 offen offset: 0 lds // (op 20451) Load value direct to lds

v_add_co_u32 v96, s[74:75], v96, v16 // (op 20451) least significant half
v_addc_co_u32 v97, s[74:75], 0, v97, s[74:75] // (op 20451) most significant half

s_add_u32 m0, m0, 4096 // (op 20451) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:75], v[68:71], a[0:3], v116, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 277) 
buffer_load_dwordx4 v96, s[60:63], s73 offen offset: 0 lds // (op 20451) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[104:107], a[4:7], v116, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 424) 
v_add_co_u32 v96, s[74:75], v96, v16 // (op 20451) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:75], v[120:123], a[8:11], v116, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 650) 
v_addc_co_u32 v97, s[74:75], 0, v97, s[74:75] // (op 20451) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:75], v[112:115], a[12:15], v116, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 876) 
s_add_u32 m0, m0, 4096 // (op 20451) 
v_bfe_u32 v88, v95, 0, 8 // (op 18835) 
v_bfe_u32 v89, v95, 8, 8 // (op 18835) 
v_bfe_u32 v90, v95, 16, 8 // (op 18835) 
v_bfe_u32 v91, v95, 24, 8 // (op 18835) 
v_permlane16_swap_b32 v88, v89 // (op 18835) 
s_nop 0
v_permlane16_swap_b32 v90, v91 // (op 18835) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v88, v90 // (op 18835) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v89, v91 // (op 18835) 
buffer_load_dwordx4 v96, s[60:63], s73 offen offset: 0 lds // (op 20451) Load value direct to lds

v_add_co_u32 v96, s[74:75], v96, v16 // (op 20451) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:75], v[124:127], a[16:19], v116, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1102) 
v_addc_co_u32 v97, s[74:75], 0, v97, s[74:75] // (op 20451) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:75], v[128:131], a[20:23], v116, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1328) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:75], v[132:135], a[24:27], v116, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1554) 
s_add_u32 m0, m0, 4096 // (op 20451) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:75], v[100:103], a[28:31], v116, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1780) 
buffer_load_dwordx4 v96, s[60:63], s73 offen offset: 0 lds // (op 20451) Load value direct to lds





s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[68:71], a[32:35], v117, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3000) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[104:107], a[36:39], v117, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2924) 
v_add_co_u32 v22, vcc, 256, v22 // (op 20621) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:79], v[120:123], a[40:43], v117, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2848) 
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 20621) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[112:115], a[44:47], v117, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2772) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 21649) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[76:79], v[124:127], a[48:51], v117, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2696) 
s_lshl_b64 s[72:73], s[26:27], 6 // (op 20446) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:79], v[128:131], a[52:55], v117, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2620) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 20446) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[76:79], v[132:135], a[56:59], v117, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2544) 
v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 20446) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[76:79], v[100:103], a[60:63], v117, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2468) 
v_mov_b64 v[20:21], v[26:27] // (op 20446) 
s_add_u32 s72, 90112, s23 // (op 20446) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[80:83], v[68:71], a[64:67], v118, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5002) 
s_mov_b32 m0, s72 // (op 20446) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[80:83], v[104:107], a[68:71], v118, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4926) 
s_mov_b32 s73, 448 // (op 20446) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[80:83], v[120:123], a[72:75], v118, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4850) 
buffer_load_dwordx4 v20, s[48:51], s73 offen offset: 0 lds // (op 20446) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[80:83], v[112:115], a[76:79], v118, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4774) 
v_add_co_u32 v20, s[74:75], v20, v16 // (op 20446) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[80:83], v[124:127], a[80:83], v118, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4698) 
v_addc_co_u32 v21, s[74:75], 0, v21, s[74:75] // (op 20446) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[80:83], v[128:131], a[84:87], v118, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4622) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[80:83], v[132:135], a[88:91], v118, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4546) 
s_add_u32 m0, m0, 4096 // (op 20446) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[80:83], v[100:103], a[92:95], v118, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4470) 
buffer_load_dwordx4 v20, s[48:51], s73 offen offset: 0 lds // (op 20446) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[108:111], v[68:71], a[96:99], v119, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7004) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[108:111], v[104:107], a[100:103], v119, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6928) 
v_add_co_u32 v18, vcc, 256, v18 // (op 20557) least significant half

v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 20557) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[108:111], v[120:123], a[104:107], v119, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6852) 
s_waitcnt vmcnt(23)
s_barrier  // (op 21610) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[108:111], v[112:115], a[108:111], v119, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6776) 
ds_read_b128 v[40:43], v4 // (op 67) Load local data 





v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[108:111], v[124:127], a[112:115], v119, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6700) 
ds_read_b128 v[44:47], v4 offset:1024 // (op 418) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[108:111], v[128:131], a[116:119], v119, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6624) 
ds_read_b128 v[48:51], v4 offset:2048 // (op 644) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[108:111], v[132:135], a[120:123], v119, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6548) 
ds_read_b128 v[52:55], v4 offset:3072 // (op 870) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[108:111], v[100:103], a[124:127], v119, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6472) 
ds_read_b128 v[56:59], v4 offset:4096 // (op 1096) Load local data 







ds_read_b128 v[60:63], v4 offset:5120 // (op 1322) Load local data 



ds_read_b128 v[64:67], v4 offset:6144 // (op 1548) Load local data 



ds_read_b128 v[68:71], v4 offset:7168 // (op 1774) Load local data 
s_waitcnt vmcnt(21)
s_barrier  // (op 21593) 



ds_read_b128 v[72:75], v3 offset:16384 // (op 62) Load local data 



ds_read_b128 v[76:79], v3 offset:17408 // (op 2461) Load local data 



ds_read_b128 v[80:83], v3 offset:18432 // (op 4463) Load local data 



ds_read_b128 v[84:87], v3 offset:19456 // (op 6465) Load local data 
s_ashr_i64 s[72:73], s[42:43], 7 // (op 4433) 
s_mov_b32 s74, s72 // (op 4433) convert

s_and_b32 s72, s74, 4294967292 // (op 4433) 

s_cmp_lt_u32 s25, s72 // (op 4433) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__ForLoopTop_KLoop_4433 // (op 4433) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__ForLoopTop_KLoop_4433 if true)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__ForLoopBottom_KLoop_4433:















s_barrier  // (op 21712) 
s_mul_hi_i32 s47, s16, s10 // (op 20489) 
v_lshrrev_b32 v14, 2, v1 // (op 20489) 
s_add_i32 s0, s47, s16 // (op 20489) 

v_and_b32 v15, -4, v1 // (op 20489) 
s_ashr_i32 s1, s0, 31 // (op 20489) 
v_sub_u32 v16, v1, v15 // (op 20489) 

s_cmp_eq_i32 s10, 0 // (op 20489) 
s_mov_b32 s47, scc // (op 20489) 
v_lshlrev_b32 v15, 5, v16 // (op 20489) 

s_cmp_eq_i32 s47, 0 // (op 20489) 
s_cselect_b32 s18, 0, -1 // (op 20489) 

s_and_b32 s19, s13, 31 // (op 20489) 
s_lshl_b32 s47, 1, s19 // (op 20489) 
s_add_i32 s22, s47, s18 // (op 20489) 


s_and_b32 s18, s1, s22 // (op 20489) 


s_add_i32 s1, s0, s18 // (op 20489) 


s_ashr_i32 s0, s1, s19 // (op 20489) 


s_lshl_b32 s1, s13, 24 // (op 20489) 
s_ashr_i32 s47, s1, 31 // (op 20489) 

s_xor_b32 s1, s0, s47 // (op 20489) 

s_sub_i32 s0, s1, s47 // (op 20489) 


s_mul_hi_i32 s1, s0, s14 // (op 20489) 
s_add_i32 s47, s1, s0 // (op 20489) 

s_ashr_i32 s1, s47, 31 // (op 20489) 
s_cmp_eq_i32 s14, 0 // (op 20489) 
s_mov_b32 s18, scc // (op 20489) 
s_cmp_eq_i32 s18, 0 // (op 20489) 
s_cselect_b32 s19, 0, -1 // (op 20489) 

s_and_b32 s18, s15, 31 // (op 20489) 
s_lshl_b32 s22, 1, s18 // (op 20489) 
s_add_i32 s23, s22, s19 // (op 20489) 


s_and_b32 s19, s1, s23 // (op 20489) 


s_add_i32 s1, s47, s19 // (op 20489) 


s_ashr_i32 s19, s1, s18 // (op 20489) 


s_lshl_b32 s1, s15, 24 // (op 20489) 
s_ashr_i32 s18, s1, 31 // (op 20489) 

s_xor_b32 s1, s19, s18 // (op 20489) 

s_sub_i32 s19, s1, s18 // (op 20489) 


s_mul_i32 s1, s0, s11 // (op 20489) 

s_sub_u32 s0, s16, s1 // (op 20489) 

s_mul_i32 s1, s19, s11 // (op 20489) 
s_add_u32 s1, s1, s0 // (op 20489) 


s_mul_hi_i32 s0, s16, s9 // (op 20489) 
s_add_i32 s47, s0, s16 // (op 20489) 

s_ashr_i32 s0, s47, 31 // (op 20489) 
s_cmp_eq_i32 s9, 0 // (op 20489) 
s_mov_b32 s18, scc // (op 20489) 
s_cmp_eq_i32 s18, 0 // (op 20489) 
s_cselect_b32 s19, 0, -1 // (op 20489) 

s_and_b32 s18, s45, 31 // (op 20489) 
s_lshl_b32 s22, 1, s18 // (op 20489) 
s_add_i32 s23, s22, s19 // (op 20489) 


s_and_b32 s19, s0, s23 // (op 20489) 


s_add_i32 s0, s47, s19 // (op 20489) 


s_ashr_i32 s19, s0, s18 // (op 20489) 


s_lshl_b32 s0, s45, 24 // (op 20489) 
s_ashr_i32 s18, s0, 31 // (op 20489) 

s_xor_b32 s0, s19, s18 // (op 20489) 

s_sub_i32 s19, s0, s18 // (op 20489) 


s_add_u32 s22, s6, 256 // (op 20489) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 20489) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 20489) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 20489) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 20489) 

s_mul_i32 s0, s11, s22 // (op 20489) 

s_mul_i32 s18, s8, s0 // (op 20489) 

s_mul_i32 s0, s19, s18 // (op 20489) 

s_sub_u32 s18, s16, s0 // (op 20489) 

s_mov_b32 s0, s18 // (op 20489) 
s_mul_hi_i32 s47, s0, s12 // (op 20489) 
s_add_i32 s22, s47, s0 // (op 20489) 


s_ashr_i32 s0, s22, 31 // (op 20489) 
s_cmp_eq_i32 s12, 0 // (op 20489) 
s_mov_b32 s47, scc // (op 20489) 
s_cmp_eq_i32 s47, 0 // (op 20489) 
s_cselect_b32 s23, 0, -1 // (op 20489) 

s_and_b32 s47, s44, 31 // (op 20489) 
s_lshl_b32 s24, 1, s47 // (op 20489) 
s_add_i32 s25, s24, s23 // (op 20489) 


s_and_b32 s23, s0, s25 // (op 20489) 


s_add_i32 s0, s22, s23 // (op 20489) 


s_ashr_i32 s22, s0, s47 // (op 20489) 


s_lshl_b32 s0, s44, 24 // (op 20489) 
s_ashr_i32 s47, s0, 31 // (op 20489) 

s_xor_b32 s0, s22, s47 // (op 20489) 

s_sub_i32 s22, s0, s47 // (op 20489) 


s_mul_i32 s0, s46, s11 // (op 20489) 
s_add_u32 s24, s20, 128 // (op 20489) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 20489) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 20489) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 20489) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 20489) 

s_sub_i32 s23, s24, s0 // (op 20489) 

s_mul_i32 s47, s22, s23 // (op 20489) 


s_sub_u32 s22, s18, s47 // (op 20489) 


s_add_u32 s18, s22, s0 // (op 20489) 


s_cmp_eq_i32 s19, 0 // (op 20489) 
s_mov_b32 s0, scc // (op 20489) 

s_cmp_eq_i32 s0, 0 // (op 20489) 
s_cselect_b32 s19, s18, s1 // (op 20489) 



v_lshl_add_u32 v16, s19, 7, v14 // (op 20489) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 20489) 
s_lshr_b32 s47, s0, 2 // (op 20489) 

v_lshl_add_u32 v14, s47, 9, v15 // (op 20489) 


v_mov_b32 v27, 0 // (op 20489) convert
v_mov_b32 v26, v14 // (op 20489) convert

v_mul_lo_u32 v15, v16, s27 // (op 20489) most significant: low of low * high
v_mul_hi_u32 v17, v16, s26 // (op 20489) most significant: high of low * low
v_mul_lo_u32 v14, v16, s26 // (op 20489) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 20489) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v26 // (op 20489) least significant half
v_addc_co_u32 v15, s[0:1], v15, v27, s[0:1] // (op 20489) most significant half



v_lshlrev_b64 v[16:17], 2, v[14:15] // (op 20489) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 20489) 

s_mul_i32 s1, s26, s21 // (op 20491) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 20491) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 20491) least significant
s_mul_hi_u32 s18, s26, s20 // (op 20491) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 20491) most significant: sum
s_add_u32 s1, s1, s18 // (op 20491) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 20491) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 20491) 

s_mov_b32 s48, s28 // (op 20491) reg expression
s_mov_b32 s49, s29 // (op 20491) reg expression
s_mov_b32 s50, s0 // (op 20491) reg expression
s_mov_b32 s51, 131072 // (op 20491) call()

s_mul_hi_i32 s47, s16, s10 // (op 20587) 
v_lshrrev_b32 v16, 2, v1 // (op 20587) 
s_add_i32 s0, s47, s16 // (op 20587) 

v_and_b32 v17, -4, v1 // (op 20587) 
s_ashr_i32 s1, s0, 31 // (op 20587) 
v_sub_u32 v18, v1, v17 // (op 20587) 

s_cmp_eq_i32 s10, 0 // (op 20587) 
s_mov_b32 s47, scc // (op 20587) 
v_lshlrev_b32 v17, 5, v18 // (op 20587) 

s_cmp_eq_i32 s47, 0 // (op 20587) 
s_cselect_b32 s18, 0, -1 // (op 20587) 

s_and_b32 s19, s13, 31 // (op 20587) 
s_lshl_b32 s47, 1, s19 // (op 20587) 
s_add_i32 s22, s47, s18 // (op 20587) 


s_and_b32 s18, s1, s22 // (op 20587) 


s_add_i32 s1, s0, s18 // (op 20587) 


s_ashr_i32 s0, s1, s19 // (op 20587) 


s_lshl_b32 s1, s13, 24 // (op 20587) 
s_ashr_i32 s47, s1, 31 // (op 20587) 

s_xor_b32 s1, s0, s47 // (op 20587) 

s_sub_i32 s0, s1, s47 // (op 20587) 


s_mul_hi_i32 s1, s0, s14 // (op 20587) 
s_add_i32 s47, s1, s0 // (op 20587) 

s_ashr_i32 s1, s47, 31 // (op 20587) 
s_cmp_eq_i32 s14, 0 // (op 20587) 
s_mov_b32 s18, scc // (op 20587) 
s_cmp_eq_i32 s18, 0 // (op 20587) 
s_cselect_b32 s19, 0, -1 // (op 20587) 

s_and_b32 s18, s15, 31 // (op 20587) 
s_lshl_b32 s22, 1, s18 // (op 20587) 
s_add_i32 s23, s22, s19 // (op 20587) 


s_and_b32 s19, s1, s23 // (op 20587) 


s_add_i32 s1, s47, s19 // (op 20587) 


s_ashr_i32 s19, s1, s18 // (op 20587) 


s_lshl_b32 s1, s15, 24 // (op 20587) 
s_ashr_i32 s18, s1, 31 // (op 20587) 

s_xor_b32 s1, s19, s18 // (op 20587) 

s_sub_i32 s19, s1, s18 // (op 20587) 


s_add_u32 s22, s6, 256 // (op 20587) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 20587) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 20587) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 20587) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 20587) 

s_mul_i32 s1, s19, s22 // (op 20587) 

s_sub_i32 s47, s0, s1 // (op 20587) 


s_mul_hi_i32 s0, s16, s9 // (op 20587) 
s_add_i32 s1, s0, s16 // (op 20587) 

s_ashr_i32 s0, s1, 31 // (op 20587) 
s_cmp_eq_i32 s9, 0 // (op 20587) 
s_mov_b32 s18, scc // (op 20587) 
s_cmp_eq_i32 s18, 0 // (op 20587) 
s_cselect_b32 s19, 0, -1 // (op 20587) 

s_and_b32 s18, s45, 31 // (op 20587) 
s_lshl_b32 s24, 1, s18 // (op 20587) 
s_add_i32 s25, s24, s19 // (op 20587) 


s_and_b32 s19, s0, s25 // (op 20587) 


s_add_i32 s0, s1, s19 // (op 20587) 


s_ashr_i32 s1, s0, s18 // (op 20587) 


s_lshl_b32 s0, s45, 24 // (op 20587) 
s_ashr_i32 s18, s0, 31 // (op 20587) 

s_xor_b32 s0, s1, s18 // (op 20587) 

s_sub_i32 s1, s0, s18 // (op 20587) 


s_mul_i32 s0, s11, s22 // (op 20587) 

s_mul_i32 s18, s8, s0 // (op 20587) 

s_mul_i32 s0, s1, s18 // (op 20587) 

s_sub_u32 s18, s16, s0 // (op 20587) 

s_mul_hi_i32 s0, s18, s12 // (op 20587) 
s_add_i32 s19, s0, s18 // (op 20587) 


s_ashr_i32 s0, s19, 31 // (op 20587) 
s_cmp_eq_i32 s12, 0 // (op 20587) 
s_mov_b32 s18, scc // (op 20587) 
s_cmp_eq_i32 s18, 0 // (op 20587) 
s_cselect_b32 s22, 0, -1 // (op 20587) 

s_and_b32 s18, s44, 31 // (op 20587) 
s_lshl_b32 s23, 1, s18 // (op 20587) 
s_add_i32 s24, s23, s22 // (op 20587) 


s_and_b32 s25, s0, s24 // (op 20587) 


s_add_i32 s0, s19, s25 // (op 20587) 


s_ashr_i32 s19, s0, s18 // (op 20587) 


s_lshl_b32 s0, s44, 24 // (op 20587) 
s_ashr_i32 s18, s0, 31 // (op 20587) 

s_xor_b32 s0, s19, s18 // (op 20587) 

s_sub_i32 s19, s0, s18 // (op 20587) 


s_cmp_eq_i32 s1, 0 // (op 20587) 
s_mov_b32 s0, scc // (op 20587) 

s_cmp_eq_i32 s0, 0 // (op 20587) 
s_cselect_b32 s1, s19, s47 // (op 20587) 



v_lshl_add_u32 v18, s1, 8, v16 // (op 20587) 


v_mul_lo_u32 v27, v18, s31 // (op 20587) most significant: low of low * high
v_mul_hi_u32 v16, v18, s30 // (op 20587) most significant: high of low * low
v_mul_lo_u32 v26, v18, s30 // (op 20587) least significant: low of low * low
v_add_u32 v27, v27, v16 // (op 20587) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 20587) 
s_lshr_b32 s47, s0, 2 // (op 20587) 

v_lshl_add_u32 v16, s47, 9, v17 // (op 20587) 


v_mov_b32 v25, 0 // (op 20587) convert
v_mov_b32 v24, v16 // (op 20587) convert

v_add_co_u32 v16, s[0:1], v24, v26 // (op 20587) least significant half
v_addc_co_u32 v17, s[0:1], v25, v27, s[0:1] // (op 20587) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 20587) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 20587) 

s_mul_i32 s1, s30, s7 // (op 20589) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 20589) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 20589) least significant
s_mul_hi_u32 s18, s30, s6 // (op 20589) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 20589) most significant: sum
s_add_u32 s1, s1, s18 // (op 20589) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 20589) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 20589) 

s_mov_b32 s60, s32 // (op 20589) reg expression
s_mov_b32 s61, s33 // (op 20589) reg expression
s_mov_b32 s62, s0 // (op 20589) reg expression
s_mov_b32 s63, 131072 // (op 20589) call()

v_and_b32 v16, -4, v1 // (op 20685) 
v_sub_u32 v17, v1, v16 // (op 20685) 

v_lshlrev_b32 v16, 5, v17 // (op 20685) 

v_lshrrev_b32 v17, 2, v1 // (op 20685) 
v_lshl_add_u32 v20, v17, 7, v16 // (op 20685) 


v_lshlrev_b32 v16, 2, v20 // (op 20685) 

v_lshrrev_b32 v17, 3, v16 // (op 20685) 

s_nop 0
v_readfirstlane_b32 s47, v17 // (op 20685) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 20742) 
v_sub_u32 v17, v1, v16 // (op 20742) 

v_lshlrev_b32 v16, 5, v17 // (op 20742) 

v_lshrrev_b32 v17, 2, v1 // (op 20742) 
v_lshl_add_u32 v20, v17, 7, v16 // (op 20742) 


v_lshlrev_b32 v16, 2, v20 // (op 20742) 

v_lshrrev_b32 v17, 3, v16 // (op 20742) 

s_nop 0
v_readfirstlane_b32 s0, v17 // (op 20742) Wait state hazard: VALU Write Readlane Hazard

s_mul_hi_i32 s1, s16, s10 // (op 21183) 
v_and_b32 v16, 63, v1 // (op 21183) 
s_add_i32 s18, s1, s16 // (op 21183) 

v_lshrrev_b32 v17, 4, v16 // (op 21183) 
s_ashr_i32 s1, s18, 31 // (op 21183) 
v_and_b32 v20, 3, v17 // (op 21183) 

s_cmp_eq_i32 s10, 0 // (op 21183) 
s_mov_b32 s19, scc // (op 21183) 
v_and_b32 v17, 15, v1 // (op 21183) 
s_cmp_eq_i32 s19, 0 // (op 21183) 
s_cselect_b32 s22, 0, -1 // (op 21183) 

v_lshl_add_u32 v21, v20, 4, v17 // (op 21183) 


s_and_b32 s19, s13, 31 // (op 21183) 
v_lshrrev_b32 v17, 7, v1 // (op 21183) 
s_lshl_b32 s23, 1, s19 // (op 21183) 
v_lshl_add_u32 v20, v17, 6, v21 // (op 21183) 


s_add_i32 s24, s23, s22 // (op 21183) 


v_lshrrev_b32 v17, 6, v16 // (op 21183) 

s_and_b32 s25, s1, s24 // (op 21183) 


v_lshlrev_b32 v16, 2, v17 // (op 21183) 

s_add_i32 s1, s18, s25 // (op 21183) 


s_ashr_i32 s18, s1, s19 // (op 21183) 


s_lshl_b32 s1, s13, 24 // (op 21183) 
s_ashr_i32 s19, s1, 31 // (op 21183) 

s_xor_b32 s1, s18, s19 // (op 21183) 

s_sub_i32 s18, s1, s19 // (op 21183) 


s_mul_hi_i32 s1, s18, s14 // (op 21183) 
s_add_i32 s19, s1, s18 // (op 21183) 

s_ashr_i32 s1, s19, 31 // (op 21183) 
s_cmp_eq_i32 s14, 0 // (op 21183) 
s_mov_b32 s22, scc // (op 21183) 
s_cmp_eq_i32 s22, 0 // (op 21183) 
s_cselect_b32 s23, 0, -1 // (op 21183) 

s_and_b32 s22, s15, 31 // (op 21183) 
s_lshl_b32 s24, 1, s22 // (op 21183) 
s_add_i32 s25, s24, s23 // (op 21183) 


s_and_b32 s23, s1, s25 // (op 21183) 


s_add_i32 s1, s19, s23 // (op 21183) 


s_ashr_i32 s19, s1, s22 // (op 21183) 


s_lshl_b32 s1, s15, 24 // (op 21183) 
s_ashr_i32 s22, s1, 31 // (op 21183) 

s_xor_b32 s1, s19, s22 // (op 21183) 

s_sub_i32 s19, s1, s22 // (op 21183) 


s_mul_i32 s1, s18, s11 // (op 21183) 

s_sub_u32 s18, s16, s1 // (op 21183) 

s_mul_i32 s1, s19, s11 // (op 21183) 
s_add_u32 s1, s1, s18 // (op 21183) 


s_mul_hi_i32 s18, s16, s9 // (op 21183) 
s_add_i32 s19, s18, s16 // (op 21183) 

s_ashr_i32 s18, s19, 31 // (op 21183) 
s_cmp_eq_i32 s9, 0 // (op 21183) 
s_mov_b32 s22, scc // (op 21183) 
s_cmp_eq_i32 s22, 0 // (op 21183) 
s_cselect_b32 s23, 0, -1 // (op 21183) 

s_and_b32 s22, s45, 31 // (op 21183) 
s_lshl_b32 s24, 1, s22 // (op 21183) 
s_add_i32 s25, s24, s23 // (op 21183) 


s_and_b32 s23, s18, s25 // (op 21183) 


s_add_i32 s18, s19, s23 // (op 21183) 


s_ashr_i32 s19, s18, s22 // (op 21183) 


s_lshl_b32 s18, s45, 24 // (op 21183) 
s_ashr_i32 s22, s18, 31 // (op 21183) 

s_xor_b32 s18, s19, s22 // (op 21183) 

s_sub_i32 s19, s18, s22 // (op 21183) 


s_add_u32 s22, s6, 256 // (op 21183) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 21183) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 21183) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 21183) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 21183) 

s_mul_i32 s18, s11, s22 // (op 21183) 

s_mul_i32 s22, s8, s18 // (op 21183) 

s_mul_i32 s18, s19, s22 // (op 21183) 

s_sub_u32 s22, s16, s18 // (op 21183) 

s_mov_b32 s18, s22 // (op 21183) 
s_mul_hi_i32 s23, s18, s12 // (op 21183) 
s_add_i32 s24, s23, s18 // (op 21183) 


s_ashr_i32 s18, s24, 31 // (op 21183) 
s_cmp_eq_i32 s12, 0 // (op 21183) 
s_mov_b32 s23, scc // (op 21183) 
s_cmp_eq_i32 s23, 0 // (op 21183) 
s_cselect_b32 s25, 0, -1 // (op 21183) 

s_and_b32 s23, s44, 31 // (op 21183) 
s_lshl_b32 s64, 1, s23 // (op 21183) 
s_add_i32 s65, s64, s25 // (op 21183) 


s_and_b32 s25, s18, s65 // (op 21183) 


s_add_i32 s18, s24, s25 // (op 21183) 


s_ashr_i32 s24, s18, s23 // (op 21183) 


s_lshl_b32 s18, s44, 24 // (op 21183) 
s_ashr_i32 s23, s18, 31 // (op 21183) 

s_xor_b32 s18, s24, s23 // (op 21183) 

s_sub_i32 s24, s18, s23 // (op 21183) 


s_mul_i32 s18, s46, s11 // (op 21183) 
s_add_u32 s64, s20, 128 // (op 21183) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 21183) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 21183) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 21183) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 7 // (op 21183) 

s_sub_i32 s23, s64, s18 // (op 21183) 

s_mul_i32 s25, s24, s23 // (op 21183) 


s_sub_u32 s23, s22, s25 // (op 21183) 


s_add_u32 s22, s23, s18 // (op 21183) 


s_cmp_eq_i32 s19, 0 // (op 21183) 
s_mov_b32 s18, scc // (op 21183) 

s_cmp_eq_i32 s18, 0 // (op 21183) 
s_cselect_b32 s19, s22, s1 // (op 21183) 



v_lshl_add_u32 v17, s19, 7, v20 // (op 21183) 


s_ashr_i64 s[18:19], s[42:43], 7 // (op 21183) 
s_lshr_b32 s1, s18, 2 // (op 21183) 

v_lshl_add_u32 v20, s1, 4, v16 // (op 21183) 


v_mul_lo_u32 v16, v17, s34 // (op 21183) 
v_add_u32 v16, v16, v20 // (op 21183) 


s_mul_i32 s19, s34, s21 // (op 21185) most significant: low of low * high
s_mul_i32 s1, s35, s20 // (op 21185) most significant: low of high * low
s_mul_i32 s18, s34, s20 // (op 21185) least significant
s_mul_hi_u32 s22, s34, s20 // (op 21185) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 21185) most significant: sum
s_add_u32 s19, s19, s22 // (op 21185) most significant: sum


s_mov_b32 s64, s36 // (op 21185) reg expression
s_mov_b32 s65, s37 // (op 21185) reg expression
s_mov_b32 s66, s18 // (op 21185) reg expression
s_mov_b32 s67, 131072 // (op 21185) call()

s_mul_hi_i32 s1, s16, s10 // (op 21210) 
v_and_b32 v17, 63, v1 // (op 21210) 
s_add_i32 s18, s1, s16 // (op 21210) 

v_lshrrev_b32 v20, 4, v17 // (op 21210) 
s_ashr_i32 s1, s18, 31 // (op 21210) 
v_and_b32 v21, 3, v20 // (op 21210) 

s_cmp_eq_i32 s10, 0 // (op 21210) 
s_mov_b32 s19, scc // (op 21210) 
v_and_b32 v20, 15, v1 // (op 21210) 
s_cmp_eq_i32 s19, 0 // (op 21210) 
s_cselect_b32 s22, 0, -1 // (op 21210) 

v_lshl_add_u32 v22, v21, 4, v20 // (op 21210) 


s_and_b32 s19, s13, 31 // (op 21210) 
v_lshrrev_b32 v20, 6, v1 // (op 21210) 
s_lshl_b32 s23, 1, s19 // (op 21210) 
v_and_b32 v21, 1, v20 // (op 21210) 

s_add_i32 s24, s23, s22 // (op 21210) 


v_lshl_add_u32 v20, v21, 7, v22 // (op 21210) 


s_and_b32 s25, s1, s24 // (op 21210) 


v_lshrrev_b32 v21, 6, v17 // (op 21210) 

s_add_i32 s1, s18, s25 // (op 21210) 


v_lshlrev_b32 v17, 2, v21 // (op 21210) 

s_ashr_i32 s18, s1, s19 // (op 21210) 


s_lshl_b32 s1, s13, 24 // (op 21210) 
s_ashr_i32 s19, s1, 31 // (op 21210) 

s_xor_b32 s1, s18, s19 // (op 21210) 

s_sub_i32 s18, s1, s19 // (op 21210) 


s_mul_hi_i32 s1, s18, s14 // (op 21210) 
s_add_i32 s19, s1, s18 // (op 21210) 

s_ashr_i32 s1, s19, 31 // (op 21210) 
s_cmp_eq_i32 s14, 0 // (op 21210) 
s_mov_b32 s22, scc // (op 21210) 
s_cmp_eq_i32 s22, 0 // (op 21210) 
s_cselect_b32 s23, 0, -1 // (op 21210) 

s_and_b32 s22, s15, 31 // (op 21210) 
s_lshl_b32 s24, 1, s22 // (op 21210) 
s_add_i32 s25, s24, s23 // (op 21210) 


s_and_b32 s23, s1, s25 // (op 21210) 


s_add_i32 s1, s19, s23 // (op 21210) 


s_ashr_i32 s19, s1, s22 // (op 21210) 


s_lshl_b32 s1, s15, 24 // (op 21210) 
s_ashr_i32 s22, s1, 31 // (op 21210) 

s_xor_b32 s1, s19, s22 // (op 21210) 

s_sub_i32 s19, s1, s22 // (op 21210) 


s_add_u32 s22, s6, 256 // (op 21210) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 21210) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 21210) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 21210) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 21210) 

s_mul_i32 s1, s19, s22 // (op 21210) 

s_sub_i32 s19, s18, s1 // (op 21210) 


s_mul_hi_i32 s1, s16, s9 // (op 21210) 
s_add_i32 s18, s1, s16 // (op 21210) 

s_ashr_i32 s1, s18, 31 // (op 21210) 
s_cmp_eq_i32 s9, 0 // (op 21210) 
s_mov_b32 s24, scc // (op 21210) 
s_cmp_eq_i32 s24, 0 // (op 21210) 
s_cselect_b32 s25, 0, -1 // (op 21210) 

s_and_b32 s24, s45, 31 // (op 21210) 
s_lshl_b32 s68, 1, s24 // (op 21210) 
s_add_i32 s69, s68, s25 // (op 21210) 


s_and_b32 s25, s1, s69 // (op 21210) 


s_add_i32 s1, s18, s25 // (op 21210) 


s_ashr_i32 s18, s1, s24 // (op 21210) 


s_lshl_b32 s1, s45, 24 // (op 21210) 
s_ashr_i32 s24, s1, 31 // (op 21210) 

s_xor_b32 s1, s18, s24 // (op 21210) 

s_sub_i32 s18, s1, s24 // (op 21210) 


s_mul_i32 s1, s11, s22 // (op 21210) 

s_mul_i32 s22, s8, s1 // (op 21210) 

s_mul_i32 s1, s18, s22 // (op 21210) 

s_sub_u32 s22, s16, s1 // (op 21210) 

s_mul_hi_i32 s1, s22, s12 // (op 21210) 
s_add_i32 s23, s1, s22 // (op 21210) 


s_ashr_i32 s1, s23, 31 // (op 21210) 
s_cmp_eq_i32 s12, 0 // (op 21210) 
s_mov_b32 s22, scc // (op 21210) 
s_cmp_eq_i32 s22, 0 // (op 21210) 
s_cselect_b32 s24, 0, -1 // (op 21210) 

s_and_b32 s22, s44, 31 // (op 21210) 
s_lshl_b32 s25, 1, s22 // (op 21210) 
s_add_i32 s68, s25, s24 // (op 21210) 


s_and_b32 s24, s1, s68 // (op 21210) 


s_add_i32 s1, s23, s24 // (op 21210) 


s_ashr_i32 s23, s1, s22 // (op 21210) 


s_lshl_b32 s1, s44, 24 // (op 21210) 
s_ashr_i32 s22, s1, 31 // (op 21210) 

s_xor_b32 s1, s23, s22 // (op 21210) 

s_sub_i32 s23, s1, s22 // (op 21210) 


s_cmp_eq_i32 s18, 0 // (op 21210) 
s_mov_b32 s1, scc // (op 21210) 

s_cmp_eq_i32 s1, 0 // (op 21210) 
s_cselect_b32 s18, s23, s19 // (op 21210) 



v_lshl_add_u32 v21, s18, 8, v20 // (op 21210) 


v_mul_lo_u32 v20, v21, s38 // (op 21210) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 21210) 
s_lshr_b32 s1, s18, 2 // (op 21210) 

v_lshl_add_u32 v21, s1, 4, v17 // (op 21210) 


v_add_u32 v17, v21, v20 // (op 21210) 


s_mul_i32 s19, s38, s7 // (op 21212) most significant: low of low * high
s_mul_i32 s1, s39, s6 // (op 21212) most significant: low of high * low
s_mul_i32 s18, s38, s6 // (op 21212) least significant
s_mul_hi_u32 s22, s38, s6 // (op 21212) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 21212) most significant: sum
s_add_u32 s19, s19, s22 // (op 21212) most significant: sum


s_mov_b32 s68, s40 // (op 21212) reg expression
s_mov_b32 s69, s41 // (op 21212) reg expression
s_mov_b32 s70, s18 // (op 21212) reg expression
s_mov_b32 s71, 131072 // (op 21212) call()

s_ashr_i64 s[18:19], s[42:43], 7 // (op 6414) 
s_and_b32 s1, s18, 4294967292 // (op 6414) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 6413) 
s_mov_b32 s22, s18 // (op 6413) convert

s_cmp_lt_u32 s1, s22 // (op 6413) 
















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__ForLoopBottom_KLoopTail_6413 // (op 6413) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__ForLoopBottom_KLoopTail_6413 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__ForLoopTop_KLoopTail_6413:
s_barrier  // (op 21590) 
s_add_u32 s1, s1, 1 // (op 6415) 

s_lshl_b64 s[18:19], s[26:27], 6 // (op 20414) 
s_lshl_b64 s[22:23], s[18:19], 2 // (op 20414) 

v_lshrrev_b64 v[20:21], 3, s[22:23] // (op 20414) 


v_mov_b64 v[22:23], v[14:15] // (op 20414) 
v_mov_b64 v[24:25], v[22:23] // (op 20414) 
s_add_u32 s18, 16384, s47 // (op 20414) 
s_mov_b32 m0, s18 // (op 20414) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 20414) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v20 // (op 20414) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 20414) most significant half

s_add_u32 m0, m0, 4096 // (op 20414) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 20414) Load value direct to lds




v_add_co_u32 v14, s[18:19], 64, v14 // (op 20479) least significant half
v_addc_co_u32 v15, s[18:19], 0, v15, s[18:19] // (op 20479) most significant half

s_barrier  // (op 21607) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 20420) 
s_lshl_b64 s[22:23], s[18:19], 2 // (op 20420) 

v_lshrrev_b64 v[20:21], 3, s[22:23] // (op 20420) 


v_mov_b64 v[22:23], v[18:19] // (op 20420) 
v_mov_b64 v[24:25], v[22:23] // (op 20420) 
s_mov_b32 m0, s0 // (op 20420) reg expression
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 20420) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v20 // (op 20420) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 20420) most significant half

s_add_u32 m0, m0, 4096 // (op 20420) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 20420) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v20 // (op 20420) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 20420) most significant half

s_add_u32 m0, m0, 4096 // (op 20420) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 20420) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v20 // (op 20420) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 20420) most significant half

s_add_u32 m0, m0, 4096 // (op 20420) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 20420) Load value direct to lds



v_add_co_u32 v18, s[18:19], 64, v18 // (op 20579) least significant half
v_addc_co_u32 v19, s[18:19], 0, v19, s[18:19] // (op 20579) most significant half

s_waitcnt vmcnt(4)
s_barrier  // (op 21597) 



ds_read_b128 v[20:23], v3 offset:16384 // (op 164) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 21613) 



ds_read_b128 v[24:27], v4 // (op 166) Load local data 


v_mov_b32 v28, v16 // (op 168) 
v_mov_b32 v29, v28 // (op 168) reg expression

buffer_load_dwordx4 v[30:33], v29, s[64:67], 0 offen // (op 168) Load value


v_add_u32 v16, 4, v16 // (op 21173) 


v_mov_b32 v28, v17 // (op 170) 
v_mov_b32 v29, v28 // (op 170) reg expression

buffer_load_dwordx4 v[34:37], v29, s[68:71], 0 offen // (op 170) Load value

s_waitcnt vmcnt(1)
v_bfe_u32 v38, v30, 0, 8 // (op 18980) 



v_bfe_u32 v39, v30, 8, 8 // (op 18980) 
v_bfe_u32 v40, v30, 16, 8 // (op 18980) 
v_bfe_u32 v41, v30, 24, 8 // (op 18980) 
v_permlane16_swap_b32 v38, v39 // (op 18980) 
ds_read_b128 v[42:45], v4 offset:1024 // (op 401) Load local data 
v_permlane16_swap_b32 v40, v41 // (op 18980) 



s_nop 1
v_permlane32_swap_b32 v38, v40 // (op 18980) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v39, v41 // (op 18980) 
ds_read_b128 v[46:49], v4 offset:2048 // (op 627) Load local data 




ds_read_b128 v[30:33], v4 offset:3072 // (op 853) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v50, v34, 0, 8 // (op 19026) 



v_bfe_u32 v51, v34, 8, 8 // (op 19026) 
v_bfe_u32 v52, v34, 16, 8 // (op 19026) 
v_bfe_u32 v53, v34, 24, 8 // (op 19026) 
v_permlane16_swap_b32 v50, v51 // (op 19026) 
s_nop 0
v_permlane16_swap_b32 v52, v53 // (op 19026) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v50, v52 // (op 19026) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v51, v53 // (op 19026) 
ds_read_b128 v[54:57], v4 offset:4096 // (op 1079) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:27], a[0:3], v38, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 171) 
s_lshl_b32 s18, s38, 6 // (op 1084) 
v_add_u32 v29, v28, s18 // (op 1084) 


buffer_load_dwordx4 v[34:37], v29, s[68:71], 0 offen // (op 1084) Load value

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:23], v[42:45], a[4:7], v38, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 404) 




v_add_u32 v17, 4, v17 // (op 21200) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:23], v[46:49], a[8:11], v38, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 630) 
ds_read_b128 v[58:61], v4 offset:5120 // (op 1305) Load local data 



ds_read_b128 v[62:65], v4 offset:6144 // (op 1531) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:23], v[30:33], a[12:15], v38, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 856) 



ds_read_b128 v[66:69], v4 offset:7168 // (op 1757) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v70, v34, 0, 8 // (op 19040) 



v_bfe_u32 v71, v34, 8, 8 // (op 19040) 
v_bfe_u32 v72, v34, 16, 8 // (op 19040) 
v_bfe_u32 v73, v34, 24, 8 // (op 19040) 
v_permlane16_swap_b32 v70, v71 // (op 19040) 
s_nop 0
v_permlane16_swap_b32 v72, v73 // (op 19040) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v70, v72 // (op 19040) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v71, v73 // (op 19040) 
ds_read_b128 v[74:77], v3 offset:17408 // (op 2253) Load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:23], v[54:57], a[16:19], v38, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1082) 
ds_read_b128 v[34:37], v3 offset:18432 // (op 4255) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:23], v[58:61], a[20:23], v38, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1308) 
ds_read_b128 v[78:81], v3 offset:19456 // (op 6257) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:23], v[62:65], a[24:27], v38, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1534) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:23], v[66:69], a[28:31], v38, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1760) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[24:27], a[32:35], v39, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2257) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:77], v[46:49], a[40:43], v39, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2311) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:77], v[30:33], a[44:47], v39, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2338) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:77], v[42:45], a[36:39], v39, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2284) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:77], v[54:57], a[48:51], v39, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2365) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:77], v[62:65], a[56:59], v39, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2419) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:77], v[66:69], a[60:63], v39, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2446) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:77], v[58:61], a[52:55], v39, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2392) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[34:37], v[24:27], a[64:67], v40, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4259) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[34:37], v[46:49], a[72:75], v40, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4313) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[34:37], v[30:33], a[76:79], v40, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4340) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[34:37], v[42:45], a[68:71], v40, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4286) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[34:37], v[54:57], a[80:83], v40, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4367) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[34:37], v[62:65], a[88:91], v40, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4421) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[34:37], v[66:69], a[92:95], v40, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4448) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[34:37], v[58:61], a[84:87], v40, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4394) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[24:27], a[96:99], v41, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6261) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[46:49], a[104:107], v41, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6315) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[30:33], a[108:111], v41, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6342) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[42:45], a[100:103], v41, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6288) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[78:81], v[54:57], a[112:115], v41, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6369) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[78:81], v[62:65], a[120:123], v41, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6423) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[78:81], v[66:69], a[124:127], v41, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6450) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[78:81], v[58:61], a[116:119], v41, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6396) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 6413) 
s_mov_b32 s22, s18 // (op 6413) convert

s_cmp_lt_u32 s1, s22 // (op 6413) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__ForLoopTop_KLoopTail_6413 // (op 6413) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__ForLoopTop_KLoopTail_6413 if true)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__ForLoopBottom_KLoopTail_6413:




















s_barrier  // (op 21736) 





















v_cmp_eq_f32 vcc, 0.00000, v0 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__ConditionalFalse_0_beta_116
v_accvgpr_read v4, a0 // (op 42) 
v_accvgpr_read v5, a1 // (op 42) 
v_accvgpr_read v6, a2 // (op 42) 
v_accvgpr_read v7, a3 // (op 42) 
v_mul_f32 v14, v11, v4 // (op 42) 
v_mul_f32 v15, v11, v5 // (op 42) 
v_mul_f32 v16, v11, v6 // (op 42) 
v_mul_f32 v17, v11, v7 // (op 42) 

v_cvt_f16_f32 v4, v14 // (op 48) 
v_cvt_f16_f32 v5, v15 // (op 48) 
v_cvt_f16_f32 v6, v16 // (op 48) 
v_cvt_f16_f32 v7, v17 // (op 48) 


v_pack_b32_f16 v8, v4, v5 // (op 50) 
v_pack_b32_f16 v9, v6, v7 // (op 50) 

v_mov_b32 v3, v13 // (op 50) reg expression

buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 50) Store value



v_accvgpr_read v4, a4 // (op 9218) 
v_accvgpr_read v5, a5 // (op 9218) 
v_accvgpr_read v6, a6 // (op 9218) 
v_accvgpr_read v7, a7 // (op 9218) 
v_mul_f32 v14, v11, v4 // (op 9218) 
v_mul_f32 v15, v11, v5 // (op 9218) 
v_mul_f32 v16, v11, v6 // (op 9218) 
v_mul_f32 v17, v11, v7 // (op 9218) 

v_cvt_f16_f32 v4, v14 // (op 9220) 
v_cvt_f16_f32 v5, v15 // (op 9220) 
v_cvt_f16_f32 v6, v16 // (op 9220) 
v_cvt_f16_f32 v7, v17 // (op 9220) 


v_pack_b32_f16 v8, v4, v5 // (op 9221) 
v_pack_b32_f16 v9, v6, v7 // (op 9221) 

s_lshl_b32 s0, s4, 5 // (op 9221) 
v_add_u32 v3, v13, s0 // (op 9221) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9221) Store value



v_accvgpr_read v4, a8 // (op 9232) 
v_accvgpr_read v5, a9 // (op 9232) 
v_accvgpr_read v6, a10 // (op 9232) 
v_accvgpr_read v7, a11 // (op 9232) 
v_mul_f32 v14, v11, v4 // (op 9232) 
v_mul_f32 v15, v11, v5 // (op 9232) 
v_mul_f32 v16, v11, v6 // (op 9232) 
v_mul_f32 v17, v11, v7 // (op 9232) 

v_cvt_f16_f32 v4, v14 // (op 9234) 
v_cvt_f16_f32 v5, v15 // (op 9234) 
v_cvt_f16_f32 v6, v16 // (op 9234) 
v_cvt_f16_f32 v7, v17 // (op 9234) 


v_pack_b32_f16 v8, v4, v5 // (op 9235) 
v_pack_b32_f16 v9, v6, v7 // (op 9235) 

s_lshl_b32 s0, s4, 6 // (op 9235) 
v_add_u32 v3, v13, s0 // (op 9235) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9235) Store value



v_accvgpr_read v4, a12 // (op 9246) 
v_accvgpr_read v5, a13 // (op 9246) 
v_accvgpr_read v6, a14 // (op 9246) 
v_accvgpr_read v7, a15 // (op 9246) 
v_mul_f32 v14, v11, v4 // (op 9246) 
v_mul_f32 v15, v11, v5 // (op 9246) 
v_mul_f32 v16, v11, v6 // (op 9246) 
v_mul_f32 v17, v11, v7 // (op 9246) 

v_cvt_f16_f32 v4, v14 // (op 9248) 
v_cvt_f16_f32 v5, v15 // (op 9248) 
v_cvt_f16_f32 v6, v16 // (op 9248) 
v_cvt_f16_f32 v7, v17 // (op 9248) 


v_pack_b32_f16 v8, v4, v5 // (op 9249) 
v_pack_b32_f16 v9, v6, v7 // (op 9249) 

s_lshl_b32 s0, s4, 5 // (op 9249) 
s_mul_i32 s1, 3, s0 // (op 9249) 

v_add_u32 v3, v13, s1 // (op 9249) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9249) Store value



v_accvgpr_read v4, a16 // (op 9260) 
v_accvgpr_read v5, a17 // (op 9260) 
v_accvgpr_read v6, a18 // (op 9260) 
v_accvgpr_read v7, a19 // (op 9260) 
v_mul_f32 v14, v11, v4 // (op 9260) 
v_mul_f32 v15, v11, v5 // (op 9260) 
v_mul_f32 v16, v11, v6 // (op 9260) 
v_mul_f32 v17, v11, v7 // (op 9260) 

v_cvt_f16_f32 v4, v14 // (op 9262) 
v_cvt_f16_f32 v5, v15 // (op 9262) 
v_cvt_f16_f32 v6, v16 // (op 9262) 
v_cvt_f16_f32 v7, v17 // (op 9262) 


v_pack_b32_f16 v8, v4, v5 // (op 9263) 
v_pack_b32_f16 v9, v6, v7 // (op 9263) 

s_lshl_b32 s0, s4, 7 // (op 9263) 
v_add_u32 v3, v13, s0 // (op 9263) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9263) Store value



v_accvgpr_read v4, a20 // (op 9274) 
v_accvgpr_read v5, a21 // (op 9274) 
v_accvgpr_read v6, a22 // (op 9274) 
v_accvgpr_read v7, a23 // (op 9274) 
v_mul_f32 v14, v11, v4 // (op 9274) 
v_mul_f32 v15, v11, v5 // (op 9274) 
v_mul_f32 v16, v11, v6 // (op 9274) 
v_mul_f32 v17, v11, v7 // (op 9274) 

v_cvt_f16_f32 v4, v14 // (op 9276) 
v_cvt_f16_f32 v5, v15 // (op 9276) 
v_cvt_f16_f32 v6, v16 // (op 9276) 
v_cvt_f16_f32 v7, v17 // (op 9276) 


v_pack_b32_f16 v8, v4, v5 // (op 9277) 
v_pack_b32_f16 v9, v6, v7 // (op 9277) 

s_lshl_b32 s0, s4, 5 // (op 9277) 
s_mul_i32 s1, 5, s0 // (op 9277) 

v_add_u32 v3, v13, s1 // (op 9277) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9277) Store value



v_accvgpr_read v4, a24 // (op 9288) 
v_accvgpr_read v5, a25 // (op 9288) 
v_accvgpr_read v6, a26 // (op 9288) 
v_accvgpr_read v7, a27 // (op 9288) 
v_mul_f32 v14, v11, v4 // (op 9288) 
v_mul_f32 v15, v11, v5 // (op 9288) 
v_mul_f32 v16, v11, v6 // (op 9288) 
v_mul_f32 v17, v11, v7 // (op 9288) 

v_cvt_f16_f32 v4, v14 // (op 9290) 
v_cvt_f16_f32 v5, v15 // (op 9290) 
v_cvt_f16_f32 v6, v16 // (op 9290) 
v_cvt_f16_f32 v7, v17 // (op 9290) 


v_pack_b32_f16 v8, v4, v5 // (op 9291) 
v_pack_b32_f16 v9, v6, v7 // (op 9291) 

s_lshl_b32 s0, s4, 5 // (op 9291) 
s_mul_i32 s1, 6, s0 // (op 9291) 

v_add_u32 v3, v13, s1 // (op 9291) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9291) Store value



v_accvgpr_read v4, a28 // (op 9302) 
v_accvgpr_read v5, a29 // (op 9302) 
v_accvgpr_read v6, a30 // (op 9302) 
v_accvgpr_read v7, a31 // (op 9302) 
v_mul_f32 v14, v11, v4 // (op 9302) 
v_mul_f32 v15, v11, v5 // (op 9302) 
v_mul_f32 v16, v11, v6 // (op 9302) 
v_mul_f32 v17, v11, v7 // (op 9302) 

v_cvt_f16_f32 v4, v14 // (op 9304) 
v_cvt_f16_f32 v5, v15 // (op 9304) 
v_cvt_f16_f32 v6, v16 // (op 9304) 
v_cvt_f16_f32 v7, v17 // (op 9304) 


v_pack_b32_f16 v8, v4, v5 // (op 9305) 
v_pack_b32_f16 v9, v6, v7 // (op 9305) 

s_lshl_b32 s0, s4, 5 // (op 9305) 
s_mul_i32 s1, 7, s0 // (op 9305) 

v_add_u32 v3, v13, s1 // (op 9305) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9305) Store value



v_accvgpr_read v4, a32 // (op 9358) 
v_accvgpr_read v5, a33 // (op 9358) 
v_accvgpr_read v6, a34 // (op 9358) 
v_accvgpr_read v7, a35 // (op 9358) 
v_mul_f32 v14, v11, v4 // (op 9358) 
v_mul_f32 v15, v11, v5 // (op 9358) 
v_mul_f32 v16, v11, v6 // (op 9358) 
v_mul_f32 v17, v11, v7 // (op 9358) 

v_cvt_f16_f32 v4, v14 // (op 9360) 
v_cvt_f16_f32 v5, v15 // (op 9360) 
v_cvt_f16_f32 v6, v16 // (op 9360) 
v_cvt_f16_f32 v7, v17 // (op 9360) 


v_pack_b32_f16 v8, v4, v5 // (op 9362) 
v_pack_b32_f16 v9, v6, v7 // (op 9362) 

v_add_u32 v3, 32, v13 // (op 9362) 

buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9362) Store value



v_accvgpr_read v4, a36 // (op 9380) 
v_accvgpr_read v5, a37 // (op 9380) 
v_accvgpr_read v6, a38 // (op 9380) 
v_accvgpr_read v7, a39 // (op 9380) 
v_mul_f32 v14, v11, v4 // (op 9380) 
v_mul_f32 v15, v11, v5 // (op 9380) 
v_mul_f32 v16, v11, v6 // (op 9380) 
v_mul_f32 v17, v11, v7 // (op 9380) 

v_cvt_f16_f32 v4, v14 // (op 9382) 
v_cvt_f16_f32 v5, v15 // (op 9382) 
v_cvt_f16_f32 v6, v16 // (op 9382) 
v_cvt_f16_f32 v7, v17 // (op 9382) 


v_pack_b32_f16 v8, v4, v5 // (op 9364) 
v_pack_b32_f16 v9, v6, v7 // (op 9364) 

s_lshl_b32 s0, s4, 5 // (op 9364) 
s_add_u32 s0, s0, 32 // (op 9364) 
v_add_u32 v3, v13, s0 // (op 9364) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9364) Store value



v_accvgpr_read v4, a40 // (op 9386) 
v_accvgpr_read v5, a41 // (op 9386) 
v_accvgpr_read v6, a42 // (op 9386) 
v_accvgpr_read v7, a43 // (op 9386) 
v_mul_f32 v14, v11, v4 // (op 9386) 
v_mul_f32 v15, v11, v5 // (op 9386) 
v_mul_f32 v16, v11, v6 // (op 9386) 
v_mul_f32 v17, v11, v7 // (op 9386) 

v_cvt_f16_f32 v4, v14 // (op 9388) 
v_cvt_f16_f32 v5, v15 // (op 9388) 
v_cvt_f16_f32 v6, v16 // (op 9388) 
v_cvt_f16_f32 v7, v17 // (op 9388) 


v_pack_b32_f16 v8, v4, v5 // (op 9366) 
v_pack_b32_f16 v9, v6, v7 // (op 9366) 

s_lshl_b32 s0, s4, 6 // (op 9366) 
s_add_u32 s0, s0, 32 // (op 9366) 
v_add_u32 v3, v13, s0 // (op 9366) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9366) Store value



v_accvgpr_read v4, a44 // (op 9392) 
v_accvgpr_read v5, a45 // (op 9392) 
v_accvgpr_read v6, a46 // (op 9392) 
v_accvgpr_read v7, a47 // (op 9392) 
v_mul_f32 v14, v11, v4 // (op 9392) 
v_mul_f32 v15, v11, v5 // (op 9392) 
v_mul_f32 v16, v11, v6 // (op 9392) 
v_mul_f32 v17, v11, v7 // (op 9392) 

v_cvt_f16_f32 v4, v14 // (op 9394) 
v_cvt_f16_f32 v5, v15 // (op 9394) 
v_cvt_f16_f32 v6, v16 // (op 9394) 
v_cvt_f16_f32 v7, v17 // (op 9394) 


v_pack_b32_f16 v8, v4, v5 // (op 9368) 
v_pack_b32_f16 v9, v6, v7 // (op 9368) 

s_lshl_b32 s0, s4, 5 // (op 9368) 
s_mul_i32 s1, 3, s0 // (op 9368) 
s_add_u32 s1, s1, 32 // (op 9368) 

v_add_u32 v3, v13, s1 // (op 9368) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9368) Store value



v_accvgpr_read v4, a48 // (op 9398) 
v_accvgpr_read v5, a49 // (op 9398) 
v_accvgpr_read v6, a50 // (op 9398) 
v_accvgpr_read v7, a51 // (op 9398) 
v_mul_f32 v14, v11, v4 // (op 9398) 
v_mul_f32 v15, v11, v5 // (op 9398) 
v_mul_f32 v16, v11, v6 // (op 9398) 
v_mul_f32 v17, v11, v7 // (op 9398) 

v_cvt_f16_f32 v4, v14 // (op 9400) 
v_cvt_f16_f32 v5, v15 // (op 9400) 
v_cvt_f16_f32 v6, v16 // (op 9400) 
v_cvt_f16_f32 v7, v17 // (op 9400) 


v_pack_b32_f16 v8, v4, v5 // (op 9370) 
v_pack_b32_f16 v9, v6, v7 // (op 9370) 

s_lshl_b32 s0, s4, 7 // (op 9370) 
s_add_u32 s0, s0, 32 // (op 9370) 
v_add_u32 v3, v13, s0 // (op 9370) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9370) Store value



v_accvgpr_read v4, a52 // (op 9404) 
v_accvgpr_read v5, a53 // (op 9404) 
v_accvgpr_read v6, a54 // (op 9404) 
v_accvgpr_read v7, a55 // (op 9404) 
v_mul_f32 v14, v11, v4 // (op 9404) 
v_mul_f32 v15, v11, v5 // (op 9404) 
v_mul_f32 v16, v11, v6 // (op 9404) 
v_mul_f32 v17, v11, v7 // (op 9404) 

v_cvt_f16_f32 v4, v14 // (op 9406) 
v_cvt_f16_f32 v5, v15 // (op 9406) 
v_cvt_f16_f32 v6, v16 // (op 9406) 
v_cvt_f16_f32 v7, v17 // (op 9406) 


v_pack_b32_f16 v8, v4, v5 // (op 9372) 
v_pack_b32_f16 v9, v6, v7 // (op 9372) 

s_lshl_b32 s0, s4, 5 // (op 9372) 
s_mul_i32 s1, 5, s0 // (op 9372) 
s_add_u32 s1, s1, 32 // (op 9372) 

v_add_u32 v3, v13, s1 // (op 9372) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9372) Store value



v_accvgpr_read v4, a56 // (op 9410) 
v_accvgpr_read v5, a57 // (op 9410) 
v_accvgpr_read v6, a58 // (op 9410) 
v_accvgpr_read v7, a59 // (op 9410) 
v_mul_f32 v14, v11, v4 // (op 9410) 
v_mul_f32 v15, v11, v5 // (op 9410) 
v_mul_f32 v16, v11, v6 // (op 9410) 
v_mul_f32 v17, v11, v7 // (op 9410) 

v_cvt_f16_f32 v4, v14 // (op 9412) 
v_cvt_f16_f32 v5, v15 // (op 9412) 
v_cvt_f16_f32 v6, v16 // (op 9412) 
v_cvt_f16_f32 v7, v17 // (op 9412) 


v_pack_b32_f16 v8, v4, v5 // (op 9374) 
v_pack_b32_f16 v9, v6, v7 // (op 9374) 

s_lshl_b32 s0, s4, 5 // (op 9374) 
s_mul_i32 s1, 6, s0 // (op 9374) 
s_add_u32 s1, s1, 32 // (op 9374) 

v_add_u32 v3, v13, s1 // (op 9374) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9374) Store value



v_accvgpr_read v4, a60 // (op 9416) 
v_accvgpr_read v5, a61 // (op 9416) 
v_accvgpr_read v6, a62 // (op 9416) 
v_accvgpr_read v7, a63 // (op 9416) 
v_mul_f32 v14, v11, v4 // (op 9416) 
v_mul_f32 v15, v11, v5 // (op 9416) 
v_mul_f32 v16, v11, v6 // (op 9416) 
v_mul_f32 v17, v11, v7 // (op 9416) 

v_cvt_f16_f32 v4, v14 // (op 9418) 
v_cvt_f16_f32 v5, v15 // (op 9418) 
v_cvt_f16_f32 v6, v16 // (op 9418) 
v_cvt_f16_f32 v7, v17 // (op 9418) 


v_pack_b32_f16 v8, v4, v5 // (op 9376) 
v_pack_b32_f16 v9, v6, v7 // (op 9376) 

s_lshl_b32 s0, s4, 5 // (op 9376) 
s_mul_i32 s1, 7, s0 // (op 9376) 
s_add_u32 s1, s1, 32 // (op 9376) 

v_add_u32 v3, v13, s1 // (op 9376) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9376) Store value



v_accvgpr_read v4, a64 // (op 9506) 
v_accvgpr_read v5, a65 // (op 9506) 
v_accvgpr_read v6, a66 // (op 9506) 
v_accvgpr_read v7, a67 // (op 9506) 
v_mul_f32 v14, v11, v4 // (op 9506) 
v_mul_f32 v15, v11, v5 // (op 9506) 
v_mul_f32 v16, v11, v6 // (op 9506) 
v_mul_f32 v17, v11, v7 // (op 9506) 

v_cvt_f16_f32 v4, v14 // (op 9508) 
v_cvt_f16_f32 v5, v15 // (op 9508) 
v_cvt_f16_f32 v6, v16 // (op 9508) 
v_cvt_f16_f32 v7, v17 // (op 9508) 


v_pack_b32_f16 v8, v4, v5 // (op 9510) 
v_pack_b32_f16 v9, v6, v7 // (op 9510) 

v_add_u32 v3, 64, v13 // (op 9510) 

buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9510) Store value



v_accvgpr_read v4, a68 // (op 9528) 
v_accvgpr_read v5, a69 // (op 9528) 
v_accvgpr_read v6, a70 // (op 9528) 
v_accvgpr_read v7, a71 // (op 9528) 
v_mul_f32 v14, v11, v4 // (op 9528) 
v_mul_f32 v15, v11, v5 // (op 9528) 
v_mul_f32 v16, v11, v6 // (op 9528) 
v_mul_f32 v17, v11, v7 // (op 9528) 

v_cvt_f16_f32 v4, v14 // (op 9530) 
v_cvt_f16_f32 v5, v15 // (op 9530) 
v_cvt_f16_f32 v6, v16 // (op 9530) 
v_cvt_f16_f32 v7, v17 // (op 9530) 


v_pack_b32_f16 v8, v4, v5 // (op 9512) 
v_pack_b32_f16 v9, v6, v7 // (op 9512) 

s_lshl_b32 s0, s4, 5 // (op 9512) 
s_add_u32 s0, s0, 64 // (op 9512) 
v_add_u32 v3, v13, s0 // (op 9512) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9512) Store value



v_accvgpr_read v4, a72 // (op 9534) 
v_accvgpr_read v5, a73 // (op 9534) 
v_accvgpr_read v6, a74 // (op 9534) 
v_accvgpr_read v7, a75 // (op 9534) 
v_mul_f32 v14, v11, v4 // (op 9534) 
v_mul_f32 v15, v11, v5 // (op 9534) 
v_mul_f32 v16, v11, v6 // (op 9534) 
v_mul_f32 v17, v11, v7 // (op 9534) 

v_cvt_f16_f32 v4, v14 // (op 9536) 
v_cvt_f16_f32 v5, v15 // (op 9536) 
v_cvt_f16_f32 v6, v16 // (op 9536) 
v_cvt_f16_f32 v7, v17 // (op 9536) 


v_pack_b32_f16 v8, v4, v5 // (op 9514) 
v_pack_b32_f16 v9, v6, v7 // (op 9514) 

s_lshl_b32 s0, s4, 6 // (op 9514) 
s_add_u32 s0, s0, 64 // (op 9514) 
v_add_u32 v3, v13, s0 // (op 9514) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9514) Store value



v_accvgpr_read v4, a76 // (op 9540) 
v_accvgpr_read v5, a77 // (op 9540) 
v_accvgpr_read v6, a78 // (op 9540) 
v_accvgpr_read v7, a79 // (op 9540) 
v_mul_f32 v14, v11, v4 // (op 9540) 
v_mul_f32 v15, v11, v5 // (op 9540) 
v_mul_f32 v16, v11, v6 // (op 9540) 
v_mul_f32 v17, v11, v7 // (op 9540) 

v_cvt_f16_f32 v4, v14 // (op 9542) 
v_cvt_f16_f32 v5, v15 // (op 9542) 
v_cvt_f16_f32 v6, v16 // (op 9542) 
v_cvt_f16_f32 v7, v17 // (op 9542) 


v_pack_b32_f16 v8, v4, v5 // (op 9516) 
v_pack_b32_f16 v9, v6, v7 // (op 9516) 

s_lshl_b32 s0, s4, 5 // (op 9516) 
s_mul_i32 s1, 3, s0 // (op 9516) 
s_add_u32 s1, s1, 64 // (op 9516) 

v_add_u32 v3, v13, s1 // (op 9516) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9516) Store value



v_accvgpr_read v4, a80 // (op 9546) 
v_accvgpr_read v5, a81 // (op 9546) 
v_accvgpr_read v6, a82 // (op 9546) 
v_accvgpr_read v7, a83 // (op 9546) 
v_mul_f32 v14, v11, v4 // (op 9546) 
v_mul_f32 v15, v11, v5 // (op 9546) 
v_mul_f32 v16, v11, v6 // (op 9546) 
v_mul_f32 v17, v11, v7 // (op 9546) 

v_cvt_f16_f32 v4, v14 // (op 9548) 
v_cvt_f16_f32 v5, v15 // (op 9548) 
v_cvt_f16_f32 v6, v16 // (op 9548) 
v_cvt_f16_f32 v7, v17 // (op 9548) 


v_pack_b32_f16 v8, v4, v5 // (op 9518) 
v_pack_b32_f16 v9, v6, v7 // (op 9518) 

s_lshl_b32 s0, s4, 7 // (op 9518) 
s_add_u32 s0, s0, 64 // (op 9518) 
v_add_u32 v3, v13, s0 // (op 9518) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9518) Store value



v_accvgpr_read v4, a84 // (op 9552) 
v_accvgpr_read v5, a85 // (op 9552) 
v_accvgpr_read v6, a86 // (op 9552) 
v_accvgpr_read v7, a87 // (op 9552) 
v_mul_f32 v14, v11, v4 // (op 9552) 
v_mul_f32 v15, v11, v5 // (op 9552) 
v_mul_f32 v16, v11, v6 // (op 9552) 
v_mul_f32 v17, v11, v7 // (op 9552) 

v_cvt_f16_f32 v4, v14 // (op 9554) 
v_cvt_f16_f32 v5, v15 // (op 9554) 
v_cvt_f16_f32 v6, v16 // (op 9554) 
v_cvt_f16_f32 v7, v17 // (op 9554) 


v_pack_b32_f16 v8, v4, v5 // (op 9520) 
v_pack_b32_f16 v9, v6, v7 // (op 9520) 

s_lshl_b32 s0, s4, 5 // (op 9520) 
s_mul_i32 s1, 5, s0 // (op 9520) 
s_add_u32 s1, s1, 64 // (op 9520) 

v_add_u32 v3, v13, s1 // (op 9520) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9520) Store value



v_accvgpr_read v4, a88 // (op 9558) 
v_accvgpr_read v5, a89 // (op 9558) 
v_accvgpr_read v6, a90 // (op 9558) 
v_accvgpr_read v7, a91 // (op 9558) 
v_mul_f32 v14, v11, v4 // (op 9558) 
v_mul_f32 v15, v11, v5 // (op 9558) 
v_mul_f32 v16, v11, v6 // (op 9558) 
v_mul_f32 v17, v11, v7 // (op 9558) 

v_cvt_f16_f32 v4, v14 // (op 9560) 
v_cvt_f16_f32 v5, v15 // (op 9560) 
v_cvt_f16_f32 v6, v16 // (op 9560) 
v_cvt_f16_f32 v7, v17 // (op 9560) 


v_pack_b32_f16 v8, v4, v5 // (op 9522) 
v_pack_b32_f16 v9, v6, v7 // (op 9522) 

s_lshl_b32 s0, s4, 5 // (op 9522) 
s_mul_i32 s1, 6, s0 // (op 9522) 
s_add_u32 s1, s1, 64 // (op 9522) 

v_add_u32 v3, v13, s1 // (op 9522) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9522) Store value



v_accvgpr_read v4, a92 // (op 9564) 
v_accvgpr_read v5, a93 // (op 9564) 
v_accvgpr_read v6, a94 // (op 9564) 
v_accvgpr_read v7, a95 // (op 9564) 
v_mul_f32 v14, v11, v4 // (op 9564) 
v_mul_f32 v15, v11, v5 // (op 9564) 
v_mul_f32 v16, v11, v6 // (op 9564) 
v_mul_f32 v17, v11, v7 // (op 9564) 

v_cvt_f16_f32 v4, v14 // (op 9566) 
v_cvt_f16_f32 v5, v15 // (op 9566) 
v_cvt_f16_f32 v6, v16 // (op 9566) 
v_cvt_f16_f32 v7, v17 // (op 9566) 


v_pack_b32_f16 v8, v4, v5 // (op 9524) 
v_pack_b32_f16 v9, v6, v7 // (op 9524) 

s_lshl_b32 s0, s4, 5 // (op 9524) 
s_mul_i32 s1, 7, s0 // (op 9524) 
s_add_u32 s1, s1, 64 // (op 9524) 

v_add_u32 v3, v13, s1 // (op 9524) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9524) Store value



v_accvgpr_read v4, a96 // (op 9654) 
v_accvgpr_read v5, a97 // (op 9654) 
v_accvgpr_read v6, a98 // (op 9654) 
v_accvgpr_read v7, a99 // (op 9654) 
v_mul_f32 v14, v11, v4 // (op 9654) 
v_mul_f32 v15, v11, v5 // (op 9654) 
v_mul_f32 v16, v11, v6 // (op 9654) 
v_mul_f32 v17, v11, v7 // (op 9654) 

v_cvt_f16_f32 v4, v14 // (op 9656) 
v_cvt_f16_f32 v5, v15 // (op 9656) 
v_cvt_f16_f32 v6, v16 // (op 9656) 
v_cvt_f16_f32 v7, v17 // (op 9656) 


v_pack_b32_f16 v8, v4, v5 // (op 9658) 
v_pack_b32_f16 v9, v6, v7 // (op 9658) 

v_add_u32 v3, 96, v13 // (op 9658) 

buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9658) Store value



v_accvgpr_read v4, a100 // (op 9676) 
v_accvgpr_read v5, a101 // (op 9676) 
v_accvgpr_read v6, a102 // (op 9676) 
v_accvgpr_read v7, a103 // (op 9676) 
v_mul_f32 v14, v11, v4 // (op 9676) 
v_mul_f32 v15, v11, v5 // (op 9676) 
v_mul_f32 v16, v11, v6 // (op 9676) 
v_mul_f32 v17, v11, v7 // (op 9676) 

v_cvt_f16_f32 v4, v14 // (op 9678) 
v_cvt_f16_f32 v5, v15 // (op 9678) 
v_cvt_f16_f32 v6, v16 // (op 9678) 
v_cvt_f16_f32 v7, v17 // (op 9678) 


v_pack_b32_f16 v8, v4, v5 // (op 9660) 
v_pack_b32_f16 v9, v6, v7 // (op 9660) 

s_lshl_b32 s0, s4, 5 // (op 9660) 
s_add_u32 s0, s0, 96 // (op 9660) 
v_add_u32 v3, v13, s0 // (op 9660) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9660) Store value



v_accvgpr_read v4, a104 // (op 9682) 
v_accvgpr_read v5, a105 // (op 9682) 
v_accvgpr_read v6, a106 // (op 9682) 
v_accvgpr_read v7, a107 // (op 9682) 
v_mul_f32 v14, v11, v4 // (op 9682) 
v_mul_f32 v15, v11, v5 // (op 9682) 
v_mul_f32 v16, v11, v6 // (op 9682) 
v_mul_f32 v17, v11, v7 // (op 9682) 

v_cvt_f16_f32 v4, v14 // (op 9684) 
v_cvt_f16_f32 v5, v15 // (op 9684) 
v_cvt_f16_f32 v6, v16 // (op 9684) 
v_cvt_f16_f32 v7, v17 // (op 9684) 


v_pack_b32_f16 v8, v4, v5 // (op 9662) 
v_pack_b32_f16 v9, v6, v7 // (op 9662) 

s_lshl_b32 s0, s4, 6 // (op 9662) 
s_add_u32 s0, s0, 96 // (op 9662) 
v_add_u32 v3, v13, s0 // (op 9662) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9662) Store value



v_accvgpr_read v4, a108 // (op 9688) 
v_accvgpr_read v5, a109 // (op 9688) 
v_accvgpr_read v6, a110 // (op 9688) 
v_accvgpr_read v7, a111 // (op 9688) 
v_mul_f32 v14, v11, v4 // (op 9688) 
v_mul_f32 v15, v11, v5 // (op 9688) 
v_mul_f32 v16, v11, v6 // (op 9688) 
v_mul_f32 v17, v11, v7 // (op 9688) 

v_cvt_f16_f32 v4, v14 // (op 9690) 
v_cvt_f16_f32 v5, v15 // (op 9690) 
v_cvt_f16_f32 v6, v16 // (op 9690) 
v_cvt_f16_f32 v7, v17 // (op 9690) 


v_pack_b32_f16 v8, v4, v5 // (op 9664) 
v_pack_b32_f16 v9, v6, v7 // (op 9664) 

s_lshl_b32 s0, s4, 5 // (op 9664) 
s_mul_i32 s1, 3, s0 // (op 9664) 
s_add_u32 s1, s1, 96 // (op 9664) 

v_add_u32 v3, v13, s1 // (op 9664) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9664) Store value



v_accvgpr_read v4, a112 // (op 9694) 
v_accvgpr_read v5, a113 // (op 9694) 
v_accvgpr_read v6, a114 // (op 9694) 
v_accvgpr_read v7, a115 // (op 9694) 
v_mul_f32 v14, v11, v4 // (op 9694) 
v_mul_f32 v15, v11, v5 // (op 9694) 
v_mul_f32 v16, v11, v6 // (op 9694) 
v_mul_f32 v17, v11, v7 // (op 9694) 

v_cvt_f16_f32 v4, v14 // (op 9696) 
v_cvt_f16_f32 v5, v15 // (op 9696) 
v_cvt_f16_f32 v6, v16 // (op 9696) 
v_cvt_f16_f32 v7, v17 // (op 9696) 


v_pack_b32_f16 v8, v4, v5 // (op 9666) 
v_pack_b32_f16 v9, v6, v7 // (op 9666) 

s_lshl_b32 s0, s4, 7 // (op 9666) 
s_add_u32 s0, s0, 96 // (op 9666) 
v_add_u32 v3, v13, s0 // (op 9666) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9666) Store value



v_accvgpr_read v4, a116 // (op 9700) 
v_accvgpr_read v5, a117 // (op 9700) 
v_accvgpr_read v6, a118 // (op 9700) 
v_accvgpr_read v7, a119 // (op 9700) 
v_mul_f32 v14, v11, v4 // (op 9700) 
v_mul_f32 v15, v11, v5 // (op 9700) 
v_mul_f32 v16, v11, v6 // (op 9700) 
v_mul_f32 v17, v11, v7 // (op 9700) 

v_cvt_f16_f32 v4, v14 // (op 9702) 
v_cvt_f16_f32 v5, v15 // (op 9702) 
v_cvt_f16_f32 v6, v16 // (op 9702) 
v_cvt_f16_f32 v7, v17 // (op 9702) 


v_pack_b32_f16 v8, v4, v5 // (op 9668) 
v_pack_b32_f16 v9, v6, v7 // (op 9668) 

s_lshl_b32 s0, s4, 5 // (op 9668) 
s_mul_i32 s1, 5, s0 // (op 9668) 
s_add_u32 s1, s1, 96 // (op 9668) 

v_add_u32 v3, v13, s1 // (op 9668) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9668) Store value



v_accvgpr_read v4, a120 // (op 9706) 
v_accvgpr_read v5, a121 // (op 9706) 
v_accvgpr_read v6, a122 // (op 9706) 
v_accvgpr_read v7, a123 // (op 9706) 
v_mul_f32 v14, v11, v4 // (op 9706) 
v_mul_f32 v15, v11, v5 // (op 9706) 
v_mul_f32 v16, v11, v6 // (op 9706) 
v_mul_f32 v17, v11, v7 // (op 9706) 

v_cvt_f16_f32 v4, v14 // (op 9708) 
v_cvt_f16_f32 v5, v15 // (op 9708) 
v_cvt_f16_f32 v6, v16 // (op 9708) 
v_cvt_f16_f32 v7, v17 // (op 9708) 


v_pack_b32_f16 v8, v4, v5 // (op 9670) 
v_pack_b32_f16 v9, v6, v7 // (op 9670) 

s_lshl_b32 s0, s4, 5 // (op 9670) 
s_mul_i32 s1, 6, s0 // (op 9670) 
s_add_u32 s1, s1, 96 // (op 9670) 

v_add_u32 v3, v13, s1 // (op 9670) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9670) Store value



v_accvgpr_read v4, a124 // (op 9712) 
v_accvgpr_read v5, a125 // (op 9712) 
v_accvgpr_read v6, a126 // (op 9712) 
v_accvgpr_read v7, a127 // (op 9712) 
v_mul_f32 v14, v11, v4 // (op 9712) 
v_mul_f32 v15, v11, v5 // (op 9712) 
v_mul_f32 v16, v11, v6 // (op 9712) 
v_mul_f32 v17, v11, v7 // (op 9712) 

v_cvt_f16_f32 v4, v14 // (op 9714) 
v_cvt_f16_f32 v5, v15 // (op 9714) 
v_cvt_f16_f32 v6, v16 // (op 9714) 
v_cvt_f16_f32 v7, v17 // (op 9714) 


v_pack_b32_f16 v8, v4, v5 // (op 9672) 
v_pack_b32_f16 v9, v6, v7 // (op 9672) 

s_lshl_b32 s0, s4, 5 // (op 9672) 
s_mul_i32 s1, 7, s0 // (op 9672) 
s_add_u32 s1, s1, 96 // (op 9672) 

v_add_u32 v3, v13, s1 // (op 9672) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 9672) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__ConditionalBottom_0_beta_116
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__ConditionalFalse_0_beta_116:


v_add_u32 v3, 96, v12 // (op 10444) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10444) Load value

v_accvgpr_read v6, a96 // (op 10423) 
v_accvgpr_read v7, a97 // (op 10423) 
v_accvgpr_read v8, a98 // (op 10423) 
v_accvgpr_read v9, a99 // (op 10423) 
v_mul_f32 v14, v11, v6 // (op 10423) 
v_mul_f32 v15, v11, v7 // (op 10423) 
v_mul_f32 v16, v11, v8 // (op 10423) 
v_mul_f32 v17, v11, v9 // (op 10423) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10443) 
v_lshrrev_b32 v7, 16, v4 // (op 10443) 
v_cvt_f32_f16 v7, v7 // (op 10443) 
v_mul_f32 v18, v0, v6 // (op 10443) 
v_mul_f32 v19, v0, v7 // (op 10443) 
v_cvt_f32_f16 v6, v5 // (op 10443) 
v_lshrrev_b32 v7, 16, v5 // (op 10443) 
v_cvt_f32_f16 v7, v7 // (op 10443) 
v_mul_f32 v20, v0, v6 // (op 10443) 
v_mul_f32 v21, v0, v7 // (op 10443) 


v_add_f32 v4, v18, v14 // (op 10424) 
v_add_f32 v5, v19, v15 // (op 10424) 
v_add_f32 v6, v20, v16 // (op 10424) 
v_add_f32 v7, v21, v17 // (op 10424) 


v_cvt_f16_f32 v14, v4 // (op 10425) 
v_cvt_f16_f32 v15, v5 // (op 10425) 
v_cvt_f16_f32 v16, v6 // (op 10425) 
v_cvt_f16_f32 v17, v7 // (op 10425) 


v_pack_b32_f16 v4, v14, v15 // (op 10427) 
v_pack_b32_f16 v5, v16, v17 // (op 10427) 

v_add_u32 v3, 96, v13 // (op 10427) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10427) Store value





s_lshl_b32 s0, s2, 5 // (op 10449) 
s_add_u32 s0, s0, 96 // (op 10449) 
v_add_u32 v3, v12, s0 // (op 10449) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10449) Load value

v_accvgpr_read v6, a100 // (op 10481) 
v_accvgpr_read v7, a101 // (op 10481) 
v_accvgpr_read v8, a102 // (op 10481) 
v_accvgpr_read v9, a103 // (op 10481) 
v_mul_f32 v14, v11, v6 // (op 10481) 
v_mul_f32 v15, v11, v7 // (op 10481) 
v_mul_f32 v16, v11, v8 // (op 10481) 
v_mul_f32 v17, v11, v9 // (op 10481) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10446) 
v_lshrrev_b32 v7, 16, v4 // (op 10446) 
v_cvt_f32_f16 v7, v7 // (op 10446) 
v_mul_f32 v18, v0, v6 // (op 10446) 
v_mul_f32 v19, v0, v7 // (op 10446) 
v_cvt_f32_f16 v6, v5 // (op 10446) 
v_lshrrev_b32 v7, 16, v5 // (op 10446) 
v_cvt_f32_f16 v7, v7 // (op 10446) 
v_mul_f32 v20, v0, v6 // (op 10446) 
v_mul_f32 v21, v0, v7 // (op 10446) 


v_add_f32 v4, v18, v14 // (op 10447) 
v_add_f32 v5, v19, v15 // (op 10447) 
v_add_f32 v6, v20, v16 // (op 10447) 
v_add_f32 v7, v21, v17 // (op 10447) 


v_cvt_f16_f32 v14, v4 // (op 10448) 
v_cvt_f16_f32 v15, v5 // (op 10448) 
v_cvt_f16_f32 v16, v6 // (op 10448) 
v_cvt_f16_f32 v17, v7 // (op 10448) 


v_pack_b32_f16 v4, v14, v15 // (op 10429) 
v_pack_b32_f16 v5, v16, v17 // (op 10429) 

s_lshl_b32 s0, s4, 5 // (op 10429) 
s_add_u32 s0, s0, 96 // (op 10429) 
v_add_u32 v3, v13, s0 // (op 10429) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10429) Store value





s_lshl_b32 s0, s2, 6 // (op 10454) 
s_add_u32 s0, s0, 96 // (op 10454) 
v_add_u32 v3, v12, s0 // (op 10454) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10454) Load value

v_accvgpr_read v6, a104 // (op 10483) 
v_accvgpr_read v7, a105 // (op 10483) 
v_accvgpr_read v8, a106 // (op 10483) 
v_accvgpr_read v9, a107 // (op 10483) 
v_mul_f32 v14, v11, v6 // (op 10483) 
v_mul_f32 v15, v11, v7 // (op 10483) 
v_mul_f32 v16, v11, v8 // (op 10483) 
v_mul_f32 v17, v11, v9 // (op 10483) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10451) 
v_lshrrev_b32 v7, 16, v4 // (op 10451) 
v_cvt_f32_f16 v7, v7 // (op 10451) 
v_mul_f32 v18, v0, v6 // (op 10451) 
v_mul_f32 v19, v0, v7 // (op 10451) 
v_cvt_f32_f16 v6, v5 // (op 10451) 
v_lshrrev_b32 v7, 16, v5 // (op 10451) 
v_cvt_f32_f16 v7, v7 // (op 10451) 
v_mul_f32 v20, v0, v6 // (op 10451) 
v_mul_f32 v21, v0, v7 // (op 10451) 


v_add_f32 v4, v18, v14 // (op 10452) 
v_add_f32 v5, v19, v15 // (op 10452) 
v_add_f32 v6, v20, v16 // (op 10452) 
v_add_f32 v7, v21, v17 // (op 10452) 


v_cvt_f16_f32 v14, v4 // (op 10453) 
v_cvt_f16_f32 v15, v5 // (op 10453) 
v_cvt_f16_f32 v16, v6 // (op 10453) 
v_cvt_f16_f32 v17, v7 // (op 10453) 


v_pack_b32_f16 v4, v14, v15 // (op 10431) 
v_pack_b32_f16 v5, v16, v17 // (op 10431) 

s_lshl_b32 s0, s4, 6 // (op 10431) 
s_add_u32 s0, s0, 96 // (op 10431) 
v_add_u32 v3, v13, s0 // (op 10431) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10431) Store value





s_lshl_b32 s0, s2, 5 // (op 10459) 
s_mul_i32 s1, 3, s0 // (op 10459) 
s_add_u32 s1, s1, 96 // (op 10459) 

v_add_u32 v3, v12, s1 // (op 10459) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10459) Load value

v_accvgpr_read v6, a108 // (op 10485) 
v_accvgpr_read v7, a109 // (op 10485) 
v_accvgpr_read v8, a110 // (op 10485) 
v_accvgpr_read v9, a111 // (op 10485) 
v_mul_f32 v14, v11, v6 // (op 10485) 
v_mul_f32 v15, v11, v7 // (op 10485) 
v_mul_f32 v16, v11, v8 // (op 10485) 
v_mul_f32 v17, v11, v9 // (op 10485) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10456) 
v_lshrrev_b32 v7, 16, v4 // (op 10456) 
v_cvt_f32_f16 v7, v7 // (op 10456) 
v_mul_f32 v18, v0, v6 // (op 10456) 
v_mul_f32 v19, v0, v7 // (op 10456) 
v_cvt_f32_f16 v6, v5 // (op 10456) 
v_lshrrev_b32 v7, 16, v5 // (op 10456) 
v_cvt_f32_f16 v7, v7 // (op 10456) 
v_mul_f32 v20, v0, v6 // (op 10456) 
v_mul_f32 v21, v0, v7 // (op 10456) 


v_add_f32 v4, v18, v14 // (op 10457) 
v_add_f32 v5, v19, v15 // (op 10457) 
v_add_f32 v6, v20, v16 // (op 10457) 
v_add_f32 v7, v21, v17 // (op 10457) 


v_cvt_f16_f32 v14, v4 // (op 10458) 
v_cvt_f16_f32 v15, v5 // (op 10458) 
v_cvt_f16_f32 v16, v6 // (op 10458) 
v_cvt_f16_f32 v17, v7 // (op 10458) 


v_pack_b32_f16 v4, v14, v15 // (op 10433) 
v_pack_b32_f16 v5, v16, v17 // (op 10433) 

s_lshl_b32 s0, s4, 5 // (op 10433) 
s_mul_i32 s1, 3, s0 // (op 10433) 
s_add_u32 s1, s1, 96 // (op 10433) 

v_add_u32 v3, v13, s1 // (op 10433) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10433) Store value





s_lshl_b32 s0, s2, 7 // (op 10464) 
s_add_u32 s0, s0, 96 // (op 10464) 
v_add_u32 v3, v12, s0 // (op 10464) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10464) Load value

v_accvgpr_read v6, a112 // (op 10487) 
v_accvgpr_read v7, a113 // (op 10487) 
v_accvgpr_read v8, a114 // (op 10487) 
v_accvgpr_read v9, a115 // (op 10487) 
v_mul_f32 v14, v11, v6 // (op 10487) 
v_mul_f32 v15, v11, v7 // (op 10487) 
v_mul_f32 v16, v11, v8 // (op 10487) 
v_mul_f32 v17, v11, v9 // (op 10487) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10461) 
v_lshrrev_b32 v7, 16, v4 // (op 10461) 
v_cvt_f32_f16 v7, v7 // (op 10461) 
v_mul_f32 v18, v0, v6 // (op 10461) 
v_mul_f32 v19, v0, v7 // (op 10461) 
v_cvt_f32_f16 v6, v5 // (op 10461) 
v_lshrrev_b32 v7, 16, v5 // (op 10461) 
v_cvt_f32_f16 v7, v7 // (op 10461) 
v_mul_f32 v20, v0, v6 // (op 10461) 
v_mul_f32 v21, v0, v7 // (op 10461) 


v_add_f32 v4, v18, v14 // (op 10462) 
v_add_f32 v5, v19, v15 // (op 10462) 
v_add_f32 v6, v20, v16 // (op 10462) 
v_add_f32 v7, v21, v17 // (op 10462) 


v_cvt_f16_f32 v14, v4 // (op 10463) 
v_cvt_f16_f32 v15, v5 // (op 10463) 
v_cvt_f16_f32 v16, v6 // (op 10463) 
v_cvt_f16_f32 v17, v7 // (op 10463) 


v_pack_b32_f16 v4, v14, v15 // (op 10435) 
v_pack_b32_f16 v5, v16, v17 // (op 10435) 

s_lshl_b32 s0, s4, 7 // (op 10435) 
s_add_u32 s0, s0, 96 // (op 10435) 
v_add_u32 v3, v13, s0 // (op 10435) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10435) Store value





s_lshl_b32 s0, s2, 5 // (op 10469) 
s_mul_i32 s1, 5, s0 // (op 10469) 
s_add_u32 s1, s1, 96 // (op 10469) 

v_add_u32 v3, v12, s1 // (op 10469) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10469) Load value

v_accvgpr_read v6, a116 // (op 10489) 
v_accvgpr_read v7, a117 // (op 10489) 
v_accvgpr_read v8, a118 // (op 10489) 
v_accvgpr_read v9, a119 // (op 10489) 
v_mul_f32 v14, v11, v6 // (op 10489) 
v_mul_f32 v15, v11, v7 // (op 10489) 
v_mul_f32 v16, v11, v8 // (op 10489) 
v_mul_f32 v17, v11, v9 // (op 10489) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10466) 
v_lshrrev_b32 v7, 16, v4 // (op 10466) 
v_cvt_f32_f16 v7, v7 // (op 10466) 
v_mul_f32 v18, v0, v6 // (op 10466) 
v_mul_f32 v19, v0, v7 // (op 10466) 
v_cvt_f32_f16 v6, v5 // (op 10466) 
v_lshrrev_b32 v7, 16, v5 // (op 10466) 
v_cvt_f32_f16 v7, v7 // (op 10466) 
v_mul_f32 v20, v0, v6 // (op 10466) 
v_mul_f32 v21, v0, v7 // (op 10466) 


v_add_f32 v4, v18, v14 // (op 10467) 
v_add_f32 v5, v19, v15 // (op 10467) 
v_add_f32 v6, v20, v16 // (op 10467) 
v_add_f32 v7, v21, v17 // (op 10467) 


v_cvt_f16_f32 v14, v4 // (op 10468) 
v_cvt_f16_f32 v15, v5 // (op 10468) 
v_cvt_f16_f32 v16, v6 // (op 10468) 
v_cvt_f16_f32 v17, v7 // (op 10468) 


v_pack_b32_f16 v4, v14, v15 // (op 10437) 
v_pack_b32_f16 v5, v16, v17 // (op 10437) 

s_lshl_b32 s0, s4, 5 // (op 10437) 
s_mul_i32 s1, 5, s0 // (op 10437) 
s_add_u32 s1, s1, 96 // (op 10437) 

v_add_u32 v3, v13, s1 // (op 10437) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10437) Store value





s_lshl_b32 s0, s2, 5 // (op 10474) 
s_mul_i32 s1, 6, s0 // (op 10474) 
s_add_u32 s1, s1, 96 // (op 10474) 

v_add_u32 v3, v12, s1 // (op 10474) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10474) Load value

v_accvgpr_read v6, a120 // (op 10491) 
v_accvgpr_read v7, a121 // (op 10491) 
v_accvgpr_read v8, a122 // (op 10491) 
v_accvgpr_read v9, a123 // (op 10491) 
v_mul_f32 v14, v11, v6 // (op 10491) 
v_mul_f32 v15, v11, v7 // (op 10491) 
v_mul_f32 v16, v11, v8 // (op 10491) 
v_mul_f32 v17, v11, v9 // (op 10491) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10471) 
v_lshrrev_b32 v7, 16, v4 // (op 10471) 
v_cvt_f32_f16 v7, v7 // (op 10471) 
v_mul_f32 v18, v0, v6 // (op 10471) 
v_mul_f32 v19, v0, v7 // (op 10471) 
v_cvt_f32_f16 v6, v5 // (op 10471) 
v_lshrrev_b32 v7, 16, v5 // (op 10471) 
v_cvt_f32_f16 v7, v7 // (op 10471) 
v_mul_f32 v20, v0, v6 // (op 10471) 
v_mul_f32 v21, v0, v7 // (op 10471) 


v_add_f32 v4, v18, v14 // (op 10472) 
v_add_f32 v5, v19, v15 // (op 10472) 
v_add_f32 v6, v20, v16 // (op 10472) 
v_add_f32 v7, v21, v17 // (op 10472) 


v_cvt_f16_f32 v14, v4 // (op 10473) 
v_cvt_f16_f32 v15, v5 // (op 10473) 
v_cvt_f16_f32 v16, v6 // (op 10473) 
v_cvt_f16_f32 v17, v7 // (op 10473) 


v_pack_b32_f16 v4, v14, v15 // (op 10439) 
v_pack_b32_f16 v5, v16, v17 // (op 10439) 

s_lshl_b32 s0, s4, 5 // (op 10439) 
s_mul_i32 s1, 6, s0 // (op 10439) 
s_add_u32 s1, s1, 96 // (op 10439) 

v_add_u32 v3, v13, s1 // (op 10439) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10439) Store value





s_lshl_b32 s0, s2, 5 // (op 10479) 
s_mul_i32 s1, 7, s0 // (op 10479) 
s_add_u32 s1, s1, 96 // (op 10479) 

v_add_u32 v3, v12, s1 // (op 10479) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10479) Load value

v_accvgpr_read v6, a124 // (op 10493) 
v_accvgpr_read v7, a125 // (op 10493) 
v_accvgpr_read v8, a126 // (op 10493) 
v_accvgpr_read v9, a127 // (op 10493) 
v_mul_f32 v14, v11, v6 // (op 10493) 
v_mul_f32 v15, v11, v7 // (op 10493) 
v_mul_f32 v16, v11, v8 // (op 10493) 
v_mul_f32 v17, v11, v9 // (op 10493) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10476) 
v_lshrrev_b32 v7, 16, v4 // (op 10476) 
v_cvt_f32_f16 v7, v7 // (op 10476) 
v_mul_f32 v18, v0, v6 // (op 10476) 
v_mul_f32 v19, v0, v7 // (op 10476) 
v_cvt_f32_f16 v6, v5 // (op 10476) 
v_lshrrev_b32 v7, 16, v5 // (op 10476) 
v_cvt_f32_f16 v7, v7 // (op 10476) 
v_mul_f32 v20, v0, v6 // (op 10476) 
v_mul_f32 v21, v0, v7 // (op 10476) 


v_add_f32 v4, v18, v14 // (op 10477) 
v_add_f32 v5, v19, v15 // (op 10477) 
v_add_f32 v6, v20, v16 // (op 10477) 
v_add_f32 v7, v21, v17 // (op 10477) 


v_cvt_f16_f32 v14, v4 // (op 10478) 
v_cvt_f16_f32 v15, v5 // (op 10478) 
v_cvt_f16_f32 v16, v6 // (op 10478) 
v_cvt_f16_f32 v17, v7 // (op 10478) 


v_pack_b32_f16 v4, v14, v15 // (op 10441) 
v_pack_b32_f16 v5, v16, v17 // (op 10441) 

s_lshl_b32 s0, s4, 5 // (op 10441) 
s_mul_i32 s1, 7, s0 // (op 10441) 
s_add_u32 s1, s1, 96 // (op 10441) 

v_add_u32 v3, v13, s1 // (op 10441) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10441) Store value





v_add_u32 v3, 32, v12 // (op 10102) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10102) Load value

v_accvgpr_read v6, a32 // (op 10081) 
v_accvgpr_read v7, a33 // (op 10081) 
v_accvgpr_read v8, a34 // (op 10081) 
v_accvgpr_read v9, a35 // (op 10081) 
v_mul_f32 v14, v11, v6 // (op 10081) 
v_mul_f32 v15, v11, v7 // (op 10081) 
v_mul_f32 v16, v11, v8 // (op 10081) 
v_mul_f32 v17, v11, v9 // (op 10081) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10101) 
v_lshrrev_b32 v7, 16, v4 // (op 10101) 
v_cvt_f32_f16 v7, v7 // (op 10101) 
v_mul_f32 v18, v0, v6 // (op 10101) 
v_mul_f32 v19, v0, v7 // (op 10101) 
v_cvt_f32_f16 v6, v5 // (op 10101) 
v_lshrrev_b32 v7, 16, v5 // (op 10101) 
v_cvt_f32_f16 v7, v7 // (op 10101) 
v_mul_f32 v20, v0, v6 // (op 10101) 
v_mul_f32 v21, v0, v7 // (op 10101) 


v_add_f32 v4, v18, v14 // (op 10082) 
v_add_f32 v5, v19, v15 // (op 10082) 
v_add_f32 v6, v20, v16 // (op 10082) 
v_add_f32 v7, v21, v17 // (op 10082) 


v_cvt_f16_f32 v14, v4 // (op 10083) 
v_cvt_f16_f32 v15, v5 // (op 10083) 
v_cvt_f16_f32 v16, v6 // (op 10083) 
v_cvt_f16_f32 v17, v7 // (op 10083) 


v_pack_b32_f16 v4, v14, v15 // (op 10085) 
v_pack_b32_f16 v5, v16, v17 // (op 10085) 

v_add_u32 v3, 32, v13 // (op 10085) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10085) Store value





s_lshl_b32 s0, s2, 5 // (op 10107) 
s_add_u32 s0, s0, 32 // (op 10107) 
v_add_u32 v3, v12, s0 // (op 10107) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10107) Load value

v_accvgpr_read v6, a36 // (op 10139) 
v_accvgpr_read v7, a37 // (op 10139) 
v_accvgpr_read v8, a38 // (op 10139) 
v_accvgpr_read v9, a39 // (op 10139) 
v_mul_f32 v14, v11, v6 // (op 10139) 
v_mul_f32 v15, v11, v7 // (op 10139) 
v_mul_f32 v16, v11, v8 // (op 10139) 
v_mul_f32 v17, v11, v9 // (op 10139) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10104) 
v_lshrrev_b32 v7, 16, v4 // (op 10104) 
v_cvt_f32_f16 v7, v7 // (op 10104) 
v_mul_f32 v18, v0, v6 // (op 10104) 
v_mul_f32 v19, v0, v7 // (op 10104) 
v_cvt_f32_f16 v6, v5 // (op 10104) 
v_lshrrev_b32 v7, 16, v5 // (op 10104) 
v_cvt_f32_f16 v7, v7 // (op 10104) 
v_mul_f32 v20, v0, v6 // (op 10104) 
v_mul_f32 v21, v0, v7 // (op 10104) 


v_add_f32 v4, v18, v14 // (op 10105) 
v_add_f32 v5, v19, v15 // (op 10105) 
v_add_f32 v6, v20, v16 // (op 10105) 
v_add_f32 v7, v21, v17 // (op 10105) 


v_cvt_f16_f32 v14, v4 // (op 10106) 
v_cvt_f16_f32 v15, v5 // (op 10106) 
v_cvt_f16_f32 v16, v6 // (op 10106) 
v_cvt_f16_f32 v17, v7 // (op 10106) 


v_pack_b32_f16 v4, v14, v15 // (op 10087) 
v_pack_b32_f16 v5, v16, v17 // (op 10087) 

s_lshl_b32 s0, s4, 5 // (op 10087) 
s_add_u32 s0, s0, 32 // (op 10087) 
v_add_u32 v3, v13, s0 // (op 10087) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10087) Store value





s_lshl_b32 s0, s2, 6 // (op 10112) 
s_add_u32 s0, s0, 32 // (op 10112) 
v_add_u32 v3, v12, s0 // (op 10112) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10112) Load value

v_accvgpr_read v6, a40 // (op 10141) 
v_accvgpr_read v7, a41 // (op 10141) 
v_accvgpr_read v8, a42 // (op 10141) 
v_accvgpr_read v9, a43 // (op 10141) 
v_mul_f32 v14, v11, v6 // (op 10141) 
v_mul_f32 v15, v11, v7 // (op 10141) 
v_mul_f32 v16, v11, v8 // (op 10141) 
v_mul_f32 v17, v11, v9 // (op 10141) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10109) 
v_lshrrev_b32 v7, 16, v4 // (op 10109) 
v_cvt_f32_f16 v7, v7 // (op 10109) 
v_mul_f32 v18, v0, v6 // (op 10109) 
v_mul_f32 v19, v0, v7 // (op 10109) 
v_cvt_f32_f16 v6, v5 // (op 10109) 
v_lshrrev_b32 v7, 16, v5 // (op 10109) 
v_cvt_f32_f16 v7, v7 // (op 10109) 
v_mul_f32 v20, v0, v6 // (op 10109) 
v_mul_f32 v21, v0, v7 // (op 10109) 


v_add_f32 v4, v18, v14 // (op 10110) 
v_add_f32 v5, v19, v15 // (op 10110) 
v_add_f32 v6, v20, v16 // (op 10110) 
v_add_f32 v7, v21, v17 // (op 10110) 


v_cvt_f16_f32 v14, v4 // (op 10111) 
v_cvt_f16_f32 v15, v5 // (op 10111) 
v_cvt_f16_f32 v16, v6 // (op 10111) 
v_cvt_f16_f32 v17, v7 // (op 10111) 


v_pack_b32_f16 v4, v14, v15 // (op 10089) 
v_pack_b32_f16 v5, v16, v17 // (op 10089) 

s_lshl_b32 s0, s4, 6 // (op 10089) 
s_add_u32 s0, s0, 32 // (op 10089) 
v_add_u32 v3, v13, s0 // (op 10089) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10089) Store value





s_lshl_b32 s0, s2, 5 // (op 10117) 
s_mul_i32 s1, 3, s0 // (op 10117) 
s_add_u32 s1, s1, 32 // (op 10117) 

v_add_u32 v3, v12, s1 // (op 10117) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10117) Load value

v_accvgpr_read v6, a44 // (op 10143) 
v_accvgpr_read v7, a45 // (op 10143) 
v_accvgpr_read v8, a46 // (op 10143) 
v_accvgpr_read v9, a47 // (op 10143) 
v_mul_f32 v14, v11, v6 // (op 10143) 
v_mul_f32 v15, v11, v7 // (op 10143) 
v_mul_f32 v16, v11, v8 // (op 10143) 
v_mul_f32 v17, v11, v9 // (op 10143) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10114) 
v_lshrrev_b32 v7, 16, v4 // (op 10114) 
v_cvt_f32_f16 v7, v7 // (op 10114) 
v_mul_f32 v18, v0, v6 // (op 10114) 
v_mul_f32 v19, v0, v7 // (op 10114) 
v_cvt_f32_f16 v6, v5 // (op 10114) 
v_lshrrev_b32 v7, 16, v5 // (op 10114) 
v_cvt_f32_f16 v7, v7 // (op 10114) 
v_mul_f32 v20, v0, v6 // (op 10114) 
v_mul_f32 v21, v0, v7 // (op 10114) 


v_add_f32 v4, v18, v14 // (op 10115) 
v_add_f32 v5, v19, v15 // (op 10115) 
v_add_f32 v6, v20, v16 // (op 10115) 
v_add_f32 v7, v21, v17 // (op 10115) 


v_cvt_f16_f32 v14, v4 // (op 10116) 
v_cvt_f16_f32 v15, v5 // (op 10116) 
v_cvt_f16_f32 v16, v6 // (op 10116) 
v_cvt_f16_f32 v17, v7 // (op 10116) 


v_pack_b32_f16 v4, v14, v15 // (op 10091) 
v_pack_b32_f16 v5, v16, v17 // (op 10091) 

s_lshl_b32 s0, s4, 5 // (op 10091) 
s_mul_i32 s1, 3, s0 // (op 10091) 
s_add_u32 s1, s1, 32 // (op 10091) 

v_add_u32 v3, v13, s1 // (op 10091) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10091) Store value





s_lshl_b32 s0, s2, 7 // (op 10122) 
s_add_u32 s0, s0, 32 // (op 10122) 
v_add_u32 v3, v12, s0 // (op 10122) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10122) Load value

v_accvgpr_read v6, a48 // (op 10145) 
v_accvgpr_read v7, a49 // (op 10145) 
v_accvgpr_read v8, a50 // (op 10145) 
v_accvgpr_read v9, a51 // (op 10145) 
v_mul_f32 v14, v11, v6 // (op 10145) 
v_mul_f32 v15, v11, v7 // (op 10145) 
v_mul_f32 v16, v11, v8 // (op 10145) 
v_mul_f32 v17, v11, v9 // (op 10145) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10119) 
v_lshrrev_b32 v7, 16, v4 // (op 10119) 
v_cvt_f32_f16 v7, v7 // (op 10119) 
v_mul_f32 v18, v0, v6 // (op 10119) 
v_mul_f32 v19, v0, v7 // (op 10119) 
v_cvt_f32_f16 v6, v5 // (op 10119) 
v_lshrrev_b32 v7, 16, v5 // (op 10119) 
v_cvt_f32_f16 v7, v7 // (op 10119) 
v_mul_f32 v20, v0, v6 // (op 10119) 
v_mul_f32 v21, v0, v7 // (op 10119) 


v_add_f32 v4, v18, v14 // (op 10120) 
v_add_f32 v5, v19, v15 // (op 10120) 
v_add_f32 v6, v20, v16 // (op 10120) 
v_add_f32 v7, v21, v17 // (op 10120) 


v_cvt_f16_f32 v14, v4 // (op 10121) 
v_cvt_f16_f32 v15, v5 // (op 10121) 
v_cvt_f16_f32 v16, v6 // (op 10121) 
v_cvt_f16_f32 v17, v7 // (op 10121) 


v_pack_b32_f16 v4, v14, v15 // (op 10093) 
v_pack_b32_f16 v5, v16, v17 // (op 10093) 

s_lshl_b32 s0, s4, 7 // (op 10093) 
s_add_u32 s0, s0, 32 // (op 10093) 
v_add_u32 v3, v13, s0 // (op 10093) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10093) Store value





s_lshl_b32 s0, s2, 5 // (op 10127) 
s_mul_i32 s1, 5, s0 // (op 10127) 
s_add_u32 s1, s1, 32 // (op 10127) 

v_add_u32 v3, v12, s1 // (op 10127) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10127) Load value

v_accvgpr_read v6, a52 // (op 10147) 
v_accvgpr_read v7, a53 // (op 10147) 
v_accvgpr_read v8, a54 // (op 10147) 
v_accvgpr_read v9, a55 // (op 10147) 
v_mul_f32 v14, v11, v6 // (op 10147) 
v_mul_f32 v15, v11, v7 // (op 10147) 
v_mul_f32 v16, v11, v8 // (op 10147) 
v_mul_f32 v17, v11, v9 // (op 10147) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10124) 
v_lshrrev_b32 v7, 16, v4 // (op 10124) 
v_cvt_f32_f16 v7, v7 // (op 10124) 
v_mul_f32 v18, v0, v6 // (op 10124) 
v_mul_f32 v19, v0, v7 // (op 10124) 
v_cvt_f32_f16 v6, v5 // (op 10124) 
v_lshrrev_b32 v7, 16, v5 // (op 10124) 
v_cvt_f32_f16 v7, v7 // (op 10124) 
v_mul_f32 v20, v0, v6 // (op 10124) 
v_mul_f32 v21, v0, v7 // (op 10124) 


v_add_f32 v4, v18, v14 // (op 10125) 
v_add_f32 v5, v19, v15 // (op 10125) 
v_add_f32 v6, v20, v16 // (op 10125) 
v_add_f32 v7, v21, v17 // (op 10125) 


v_cvt_f16_f32 v14, v4 // (op 10126) 
v_cvt_f16_f32 v15, v5 // (op 10126) 
v_cvt_f16_f32 v16, v6 // (op 10126) 
v_cvt_f16_f32 v17, v7 // (op 10126) 


v_pack_b32_f16 v4, v14, v15 // (op 10095) 
v_pack_b32_f16 v5, v16, v17 // (op 10095) 

s_lshl_b32 s0, s4, 5 // (op 10095) 
s_mul_i32 s1, 5, s0 // (op 10095) 
s_add_u32 s1, s1, 32 // (op 10095) 

v_add_u32 v3, v13, s1 // (op 10095) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10095) Store value





s_lshl_b32 s0, s2, 5 // (op 10132) 
s_mul_i32 s1, 6, s0 // (op 10132) 
s_add_u32 s1, s1, 32 // (op 10132) 

v_add_u32 v3, v12, s1 // (op 10132) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10132) Load value

v_accvgpr_read v6, a56 // (op 10149) 
v_accvgpr_read v7, a57 // (op 10149) 
v_accvgpr_read v8, a58 // (op 10149) 
v_accvgpr_read v9, a59 // (op 10149) 
v_mul_f32 v14, v11, v6 // (op 10149) 
v_mul_f32 v15, v11, v7 // (op 10149) 
v_mul_f32 v16, v11, v8 // (op 10149) 
v_mul_f32 v17, v11, v9 // (op 10149) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10129) 
v_lshrrev_b32 v7, 16, v4 // (op 10129) 
v_cvt_f32_f16 v7, v7 // (op 10129) 
v_mul_f32 v18, v0, v6 // (op 10129) 
v_mul_f32 v19, v0, v7 // (op 10129) 
v_cvt_f32_f16 v6, v5 // (op 10129) 
v_lshrrev_b32 v7, 16, v5 // (op 10129) 
v_cvt_f32_f16 v7, v7 // (op 10129) 
v_mul_f32 v20, v0, v6 // (op 10129) 
v_mul_f32 v21, v0, v7 // (op 10129) 


v_add_f32 v4, v18, v14 // (op 10130) 
v_add_f32 v5, v19, v15 // (op 10130) 
v_add_f32 v6, v20, v16 // (op 10130) 
v_add_f32 v7, v21, v17 // (op 10130) 


v_cvt_f16_f32 v14, v4 // (op 10131) 
v_cvt_f16_f32 v15, v5 // (op 10131) 
v_cvt_f16_f32 v16, v6 // (op 10131) 
v_cvt_f16_f32 v17, v7 // (op 10131) 


v_pack_b32_f16 v4, v14, v15 // (op 10097) 
v_pack_b32_f16 v5, v16, v17 // (op 10097) 

s_lshl_b32 s0, s4, 5 // (op 10097) 
s_mul_i32 s1, 6, s0 // (op 10097) 
s_add_u32 s1, s1, 32 // (op 10097) 

v_add_u32 v3, v13, s1 // (op 10097) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10097) Store value





s_lshl_b32 s0, s2, 5 // (op 10137) 
s_mul_i32 s1, 7, s0 // (op 10137) 
s_add_u32 s1, s1, 32 // (op 10137) 

v_add_u32 v3, v12, s1 // (op 10137) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10137) Load value

v_accvgpr_read v6, a60 // (op 10151) 
v_accvgpr_read v7, a61 // (op 10151) 
v_accvgpr_read v8, a62 // (op 10151) 
v_accvgpr_read v9, a63 // (op 10151) 
v_mul_f32 v14, v11, v6 // (op 10151) 
v_mul_f32 v15, v11, v7 // (op 10151) 
v_mul_f32 v16, v11, v8 // (op 10151) 
v_mul_f32 v17, v11, v9 // (op 10151) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10134) 
v_lshrrev_b32 v7, 16, v4 // (op 10134) 
v_cvt_f32_f16 v7, v7 // (op 10134) 
v_mul_f32 v18, v0, v6 // (op 10134) 
v_mul_f32 v19, v0, v7 // (op 10134) 
v_cvt_f32_f16 v6, v5 // (op 10134) 
v_lshrrev_b32 v7, 16, v5 // (op 10134) 
v_cvt_f32_f16 v7, v7 // (op 10134) 
v_mul_f32 v20, v0, v6 // (op 10134) 
v_mul_f32 v21, v0, v7 // (op 10134) 


v_add_f32 v4, v18, v14 // (op 10135) 
v_add_f32 v5, v19, v15 // (op 10135) 
v_add_f32 v6, v20, v16 // (op 10135) 
v_add_f32 v7, v21, v17 // (op 10135) 


v_cvt_f16_f32 v14, v4 // (op 10136) 
v_cvt_f16_f32 v15, v5 // (op 10136) 
v_cvt_f16_f32 v16, v6 // (op 10136) 
v_cvt_f16_f32 v17, v7 // (op 10136) 


v_pack_b32_f16 v4, v14, v15 // (op 10099) 
v_pack_b32_f16 v5, v16, v17 // (op 10099) 

s_lshl_b32 s0, s4, 5 // (op 10099) 
s_mul_i32 s1, 7, s0 // (op 10099) 
s_add_u32 s1, s1, 32 // (op 10099) 

v_add_u32 v3, v13, s1 // (op 10099) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10099) Store value





v_add_u32 v3, 64, v12 // (op 10273) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10273) Load value

v_accvgpr_read v6, a64 // (op 10252) 
v_accvgpr_read v7, a65 // (op 10252) 
v_accvgpr_read v8, a66 // (op 10252) 
v_accvgpr_read v9, a67 // (op 10252) 
v_mul_f32 v14, v11, v6 // (op 10252) 
v_mul_f32 v15, v11, v7 // (op 10252) 
v_mul_f32 v16, v11, v8 // (op 10252) 
v_mul_f32 v17, v11, v9 // (op 10252) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10272) 
v_lshrrev_b32 v7, 16, v4 // (op 10272) 
v_cvt_f32_f16 v7, v7 // (op 10272) 
v_mul_f32 v18, v0, v6 // (op 10272) 
v_mul_f32 v19, v0, v7 // (op 10272) 
v_cvt_f32_f16 v6, v5 // (op 10272) 
v_lshrrev_b32 v7, 16, v5 // (op 10272) 
v_cvt_f32_f16 v7, v7 // (op 10272) 
v_mul_f32 v20, v0, v6 // (op 10272) 
v_mul_f32 v21, v0, v7 // (op 10272) 


v_add_f32 v4, v18, v14 // (op 10253) 
v_add_f32 v5, v19, v15 // (op 10253) 
v_add_f32 v6, v20, v16 // (op 10253) 
v_add_f32 v7, v21, v17 // (op 10253) 


v_cvt_f16_f32 v14, v4 // (op 10254) 
v_cvt_f16_f32 v15, v5 // (op 10254) 
v_cvt_f16_f32 v16, v6 // (op 10254) 
v_cvt_f16_f32 v17, v7 // (op 10254) 


v_pack_b32_f16 v4, v14, v15 // (op 10256) 
v_pack_b32_f16 v5, v16, v17 // (op 10256) 

v_add_u32 v3, 64, v13 // (op 10256) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10256) Store value





s_lshl_b32 s0, s2, 5 // (op 10278) 
s_add_u32 s0, s0, 64 // (op 10278) 
v_add_u32 v3, v12, s0 // (op 10278) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10278) Load value

v_accvgpr_read v6, a68 // (op 10310) 
v_accvgpr_read v7, a69 // (op 10310) 
v_accvgpr_read v8, a70 // (op 10310) 
v_accvgpr_read v9, a71 // (op 10310) 
v_mul_f32 v14, v11, v6 // (op 10310) 
v_mul_f32 v15, v11, v7 // (op 10310) 
v_mul_f32 v16, v11, v8 // (op 10310) 
v_mul_f32 v17, v11, v9 // (op 10310) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10275) 
v_lshrrev_b32 v7, 16, v4 // (op 10275) 
v_cvt_f32_f16 v7, v7 // (op 10275) 
v_mul_f32 v18, v0, v6 // (op 10275) 
v_mul_f32 v19, v0, v7 // (op 10275) 
v_cvt_f32_f16 v6, v5 // (op 10275) 
v_lshrrev_b32 v7, 16, v5 // (op 10275) 
v_cvt_f32_f16 v7, v7 // (op 10275) 
v_mul_f32 v20, v0, v6 // (op 10275) 
v_mul_f32 v21, v0, v7 // (op 10275) 


v_add_f32 v4, v18, v14 // (op 10276) 
v_add_f32 v5, v19, v15 // (op 10276) 
v_add_f32 v6, v20, v16 // (op 10276) 
v_add_f32 v7, v21, v17 // (op 10276) 


v_cvt_f16_f32 v14, v4 // (op 10277) 
v_cvt_f16_f32 v15, v5 // (op 10277) 
v_cvt_f16_f32 v16, v6 // (op 10277) 
v_cvt_f16_f32 v17, v7 // (op 10277) 


v_pack_b32_f16 v4, v14, v15 // (op 10258) 
v_pack_b32_f16 v5, v16, v17 // (op 10258) 

s_lshl_b32 s0, s4, 5 // (op 10258) 
s_add_u32 s0, s0, 64 // (op 10258) 
v_add_u32 v3, v13, s0 // (op 10258) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10258) Store value





s_lshl_b32 s0, s2, 6 // (op 10283) 
s_add_u32 s0, s0, 64 // (op 10283) 
v_add_u32 v3, v12, s0 // (op 10283) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10283) Load value

v_accvgpr_read v6, a72 // (op 10312) 
v_accvgpr_read v7, a73 // (op 10312) 
v_accvgpr_read v8, a74 // (op 10312) 
v_accvgpr_read v9, a75 // (op 10312) 
v_mul_f32 v14, v11, v6 // (op 10312) 
v_mul_f32 v15, v11, v7 // (op 10312) 
v_mul_f32 v16, v11, v8 // (op 10312) 
v_mul_f32 v17, v11, v9 // (op 10312) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10280) 
v_lshrrev_b32 v7, 16, v4 // (op 10280) 
v_cvt_f32_f16 v7, v7 // (op 10280) 
v_mul_f32 v18, v0, v6 // (op 10280) 
v_mul_f32 v19, v0, v7 // (op 10280) 
v_cvt_f32_f16 v6, v5 // (op 10280) 
v_lshrrev_b32 v7, 16, v5 // (op 10280) 
v_cvt_f32_f16 v7, v7 // (op 10280) 
v_mul_f32 v20, v0, v6 // (op 10280) 
v_mul_f32 v21, v0, v7 // (op 10280) 


v_add_f32 v4, v18, v14 // (op 10281) 
v_add_f32 v5, v19, v15 // (op 10281) 
v_add_f32 v6, v20, v16 // (op 10281) 
v_add_f32 v7, v21, v17 // (op 10281) 


v_cvt_f16_f32 v14, v4 // (op 10282) 
v_cvt_f16_f32 v15, v5 // (op 10282) 
v_cvt_f16_f32 v16, v6 // (op 10282) 
v_cvt_f16_f32 v17, v7 // (op 10282) 


v_pack_b32_f16 v4, v14, v15 // (op 10260) 
v_pack_b32_f16 v5, v16, v17 // (op 10260) 

s_lshl_b32 s0, s4, 6 // (op 10260) 
s_add_u32 s0, s0, 64 // (op 10260) 
v_add_u32 v3, v13, s0 // (op 10260) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10260) Store value





s_lshl_b32 s0, s2, 5 // (op 10288) 
s_mul_i32 s1, 3, s0 // (op 10288) 
s_add_u32 s1, s1, 64 // (op 10288) 

v_add_u32 v3, v12, s1 // (op 10288) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10288) Load value

v_accvgpr_read v6, a76 // (op 10314) 
v_accvgpr_read v7, a77 // (op 10314) 
v_accvgpr_read v8, a78 // (op 10314) 
v_accvgpr_read v9, a79 // (op 10314) 
v_mul_f32 v14, v11, v6 // (op 10314) 
v_mul_f32 v15, v11, v7 // (op 10314) 
v_mul_f32 v16, v11, v8 // (op 10314) 
v_mul_f32 v17, v11, v9 // (op 10314) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10285) 
v_lshrrev_b32 v7, 16, v4 // (op 10285) 
v_cvt_f32_f16 v7, v7 // (op 10285) 
v_mul_f32 v18, v0, v6 // (op 10285) 
v_mul_f32 v19, v0, v7 // (op 10285) 
v_cvt_f32_f16 v6, v5 // (op 10285) 
v_lshrrev_b32 v7, 16, v5 // (op 10285) 
v_cvt_f32_f16 v7, v7 // (op 10285) 
v_mul_f32 v20, v0, v6 // (op 10285) 
v_mul_f32 v21, v0, v7 // (op 10285) 


v_add_f32 v4, v18, v14 // (op 10286) 
v_add_f32 v5, v19, v15 // (op 10286) 
v_add_f32 v6, v20, v16 // (op 10286) 
v_add_f32 v7, v21, v17 // (op 10286) 


v_cvt_f16_f32 v14, v4 // (op 10287) 
v_cvt_f16_f32 v15, v5 // (op 10287) 
v_cvt_f16_f32 v16, v6 // (op 10287) 
v_cvt_f16_f32 v17, v7 // (op 10287) 


v_pack_b32_f16 v4, v14, v15 // (op 10262) 
v_pack_b32_f16 v5, v16, v17 // (op 10262) 

s_lshl_b32 s0, s4, 5 // (op 10262) 
s_mul_i32 s1, 3, s0 // (op 10262) 
s_add_u32 s1, s1, 64 // (op 10262) 

v_add_u32 v3, v13, s1 // (op 10262) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10262) Store value





s_lshl_b32 s0, s2, 7 // (op 10293) 
s_add_u32 s0, s0, 64 // (op 10293) 
v_add_u32 v3, v12, s0 // (op 10293) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10293) Load value

v_accvgpr_read v6, a80 // (op 10316) 
v_accvgpr_read v7, a81 // (op 10316) 
v_accvgpr_read v8, a82 // (op 10316) 
v_accvgpr_read v9, a83 // (op 10316) 
v_mul_f32 v14, v11, v6 // (op 10316) 
v_mul_f32 v15, v11, v7 // (op 10316) 
v_mul_f32 v16, v11, v8 // (op 10316) 
v_mul_f32 v17, v11, v9 // (op 10316) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10290) 
v_lshrrev_b32 v7, 16, v4 // (op 10290) 
v_cvt_f32_f16 v7, v7 // (op 10290) 
v_mul_f32 v18, v0, v6 // (op 10290) 
v_mul_f32 v19, v0, v7 // (op 10290) 
v_cvt_f32_f16 v6, v5 // (op 10290) 
v_lshrrev_b32 v7, 16, v5 // (op 10290) 
v_cvt_f32_f16 v7, v7 // (op 10290) 
v_mul_f32 v20, v0, v6 // (op 10290) 
v_mul_f32 v21, v0, v7 // (op 10290) 


v_add_f32 v4, v18, v14 // (op 10291) 
v_add_f32 v5, v19, v15 // (op 10291) 
v_add_f32 v6, v20, v16 // (op 10291) 
v_add_f32 v7, v21, v17 // (op 10291) 


v_cvt_f16_f32 v14, v4 // (op 10292) 
v_cvt_f16_f32 v15, v5 // (op 10292) 
v_cvt_f16_f32 v16, v6 // (op 10292) 
v_cvt_f16_f32 v17, v7 // (op 10292) 


v_pack_b32_f16 v4, v14, v15 // (op 10264) 
v_pack_b32_f16 v5, v16, v17 // (op 10264) 

s_lshl_b32 s0, s4, 7 // (op 10264) 
s_add_u32 s0, s0, 64 // (op 10264) 
v_add_u32 v3, v13, s0 // (op 10264) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10264) Store value





s_lshl_b32 s0, s2, 5 // (op 10298) 
s_mul_i32 s1, 5, s0 // (op 10298) 
s_add_u32 s1, s1, 64 // (op 10298) 

v_add_u32 v3, v12, s1 // (op 10298) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10298) Load value

v_accvgpr_read v6, a84 // (op 10318) 
v_accvgpr_read v7, a85 // (op 10318) 
v_accvgpr_read v8, a86 // (op 10318) 
v_accvgpr_read v9, a87 // (op 10318) 
v_mul_f32 v14, v11, v6 // (op 10318) 
v_mul_f32 v15, v11, v7 // (op 10318) 
v_mul_f32 v16, v11, v8 // (op 10318) 
v_mul_f32 v17, v11, v9 // (op 10318) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10295) 
v_lshrrev_b32 v7, 16, v4 // (op 10295) 
v_cvt_f32_f16 v7, v7 // (op 10295) 
v_mul_f32 v18, v0, v6 // (op 10295) 
v_mul_f32 v19, v0, v7 // (op 10295) 
v_cvt_f32_f16 v6, v5 // (op 10295) 
v_lshrrev_b32 v7, 16, v5 // (op 10295) 
v_cvt_f32_f16 v7, v7 // (op 10295) 
v_mul_f32 v20, v0, v6 // (op 10295) 
v_mul_f32 v21, v0, v7 // (op 10295) 


v_add_f32 v4, v18, v14 // (op 10296) 
v_add_f32 v5, v19, v15 // (op 10296) 
v_add_f32 v6, v20, v16 // (op 10296) 
v_add_f32 v7, v21, v17 // (op 10296) 


v_cvt_f16_f32 v14, v4 // (op 10297) 
v_cvt_f16_f32 v15, v5 // (op 10297) 
v_cvt_f16_f32 v16, v6 // (op 10297) 
v_cvt_f16_f32 v17, v7 // (op 10297) 


v_pack_b32_f16 v4, v14, v15 // (op 10266) 
v_pack_b32_f16 v5, v16, v17 // (op 10266) 

s_lshl_b32 s0, s4, 5 // (op 10266) 
s_mul_i32 s1, 5, s0 // (op 10266) 
s_add_u32 s1, s1, 64 // (op 10266) 

v_add_u32 v3, v13, s1 // (op 10266) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10266) Store value





s_lshl_b32 s0, s2, 5 // (op 10303) 
s_mul_i32 s1, 6, s0 // (op 10303) 
s_add_u32 s1, s1, 64 // (op 10303) 

v_add_u32 v3, v12, s1 // (op 10303) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10303) Load value

v_accvgpr_read v6, a88 // (op 10320) 
v_accvgpr_read v7, a89 // (op 10320) 
v_accvgpr_read v8, a90 // (op 10320) 
v_accvgpr_read v9, a91 // (op 10320) 
v_mul_f32 v14, v11, v6 // (op 10320) 
v_mul_f32 v15, v11, v7 // (op 10320) 
v_mul_f32 v16, v11, v8 // (op 10320) 
v_mul_f32 v17, v11, v9 // (op 10320) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10300) 
v_lshrrev_b32 v7, 16, v4 // (op 10300) 
v_cvt_f32_f16 v7, v7 // (op 10300) 
v_mul_f32 v18, v0, v6 // (op 10300) 
v_mul_f32 v19, v0, v7 // (op 10300) 
v_cvt_f32_f16 v6, v5 // (op 10300) 
v_lshrrev_b32 v7, 16, v5 // (op 10300) 
v_cvt_f32_f16 v7, v7 // (op 10300) 
v_mul_f32 v20, v0, v6 // (op 10300) 
v_mul_f32 v21, v0, v7 // (op 10300) 


v_add_f32 v4, v18, v14 // (op 10301) 
v_add_f32 v5, v19, v15 // (op 10301) 
v_add_f32 v6, v20, v16 // (op 10301) 
v_add_f32 v7, v21, v17 // (op 10301) 


v_cvt_f16_f32 v14, v4 // (op 10302) 
v_cvt_f16_f32 v15, v5 // (op 10302) 
v_cvt_f16_f32 v16, v6 // (op 10302) 
v_cvt_f16_f32 v17, v7 // (op 10302) 


v_pack_b32_f16 v4, v14, v15 // (op 10268) 
v_pack_b32_f16 v5, v16, v17 // (op 10268) 

s_lshl_b32 s0, s4, 5 // (op 10268) 
s_mul_i32 s1, 6, s0 // (op 10268) 
s_add_u32 s1, s1, 64 // (op 10268) 

v_add_u32 v3, v13, s1 // (op 10268) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10268) Store value





s_lshl_b32 s0, s2, 5 // (op 10308) 
s_mul_i32 s1, 7, s0 // (op 10308) 
s_add_u32 s1, s1, 64 // (op 10308) 

v_add_u32 v3, v12, s1 // (op 10308) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 10308) Load value

v_accvgpr_read v6, a92 // (op 10322) 
v_accvgpr_read v7, a93 // (op 10322) 
v_accvgpr_read v8, a94 // (op 10322) 
v_accvgpr_read v9, a95 // (op 10322) 
v_mul_f32 v14, v11, v6 // (op 10322) 
v_mul_f32 v15, v11, v7 // (op 10322) 
v_mul_f32 v16, v11, v8 // (op 10322) 
v_mul_f32 v17, v11, v9 // (op 10322) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 10305) 
v_lshrrev_b32 v7, 16, v4 // (op 10305) 
v_cvt_f32_f16 v7, v7 // (op 10305) 
v_mul_f32 v18, v0, v6 // (op 10305) 
v_mul_f32 v19, v0, v7 // (op 10305) 
v_cvt_f32_f16 v6, v5 // (op 10305) 
v_lshrrev_b32 v7, 16, v5 // (op 10305) 
v_cvt_f32_f16 v7, v7 // (op 10305) 
v_mul_f32 v20, v0, v6 // (op 10305) 
v_mul_f32 v21, v0, v7 // (op 10305) 


v_add_f32 v4, v18, v14 // (op 10306) 
v_add_f32 v5, v19, v15 // (op 10306) 
v_add_f32 v6, v20, v16 // (op 10306) 
v_add_f32 v7, v21, v17 // (op 10306) 


v_cvt_f16_f32 v14, v4 // (op 10307) 
v_cvt_f16_f32 v15, v5 // (op 10307) 
v_cvt_f16_f32 v16, v6 // (op 10307) 
v_cvt_f16_f32 v17, v7 // (op 10307) 


v_pack_b32_f16 v4, v14, v15 // (op 10270) 
v_pack_b32_f16 v5, v16, v17 // (op 10270) 

s_lshl_b32 s0, s4, 5 // (op 10270) 
s_mul_i32 s1, 7, s0 // (op 10270) 
s_add_u32 s1, s1, 64 // (op 10270) 

v_add_u32 v3, v13, s1 // (op 10270) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 10270) Store value





v_mov_b32 v3, v12 // (op 129) reg expression

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 129) Load value

v_accvgpr_read v6, a0 // (op 125) 
v_accvgpr_read v7, a1 // (op 125) 
v_accvgpr_read v8, a2 // (op 125) 
v_accvgpr_read v9, a3 // (op 125) 
v_mul_f32 v14, v11, v6 // (op 125) 
v_mul_f32 v15, v11, v7 // (op 125) 
v_mul_f32 v16, v11, v8 // (op 125) 
v_mul_f32 v17, v11, v9 // (op 125) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 130) 
v_lshrrev_b32 v7, 16, v4 // (op 130) 
v_cvt_f32_f16 v7, v7 // (op 130) 
v_mul_f32 v18, v0, v6 // (op 130) 
v_mul_f32 v19, v0, v7 // (op 130) 
v_cvt_f32_f16 v6, v5 // (op 130) 
v_lshrrev_b32 v7, 16, v5 // (op 130) 
v_cvt_f32_f16 v7, v7 // (op 130) 
v_mul_f32 v20, v0, v6 // (op 130) 
v_mul_f32 v21, v0, v7 // (op 130) 


v_add_f32 v4, v18, v14 // (op 126) 
v_add_f32 v5, v19, v15 // (op 126) 
v_add_f32 v6, v20, v16 // (op 126) 
v_add_f32 v7, v21, v17 // (op 126) 


v_cvt_f16_f32 v14, v4 // (op 127) 
v_cvt_f16_f32 v15, v5 // (op 127) 
v_cvt_f16_f32 v16, v6 // (op 127) 
v_cvt_f16_f32 v17, v7 // (op 127) 


v_pack_b32_f16 v4, v14, v15 // (op 128) 
v_pack_b32_f16 v5, v16, v17 // (op 128) 

v_mov_b32 v3, v13 // (op 128) reg expression

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 128) Store value





s_lshl_b32 s0, s2, 5 // (op 9906) 
v_add_u32 v3, v12, s0 // (op 9906) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 9906) Load value

v_accvgpr_read v6, a4 // (op 9902) 
v_accvgpr_read v7, a5 // (op 9902) 
v_accvgpr_read v8, a6 // (op 9902) 
v_accvgpr_read v9, a7 // (op 9902) 
v_mul_f32 v14, v11, v6 // (op 9902) 
v_mul_f32 v15, v11, v7 // (op 9902) 
v_mul_f32 v16, v11, v8 // (op 9902) 
v_mul_f32 v17, v11, v9 // (op 9902) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 9907) 
v_lshrrev_b32 v7, 16, v4 // (op 9907) 
v_cvt_f32_f16 v7, v7 // (op 9907) 
v_mul_f32 v18, v0, v6 // (op 9907) 
v_mul_f32 v19, v0, v7 // (op 9907) 
v_cvt_f32_f16 v6, v5 // (op 9907) 
v_lshrrev_b32 v7, 16, v5 // (op 9907) 
v_cvt_f32_f16 v7, v7 // (op 9907) 
v_mul_f32 v20, v0, v6 // (op 9907) 
v_mul_f32 v21, v0, v7 // (op 9907) 


v_add_f32 v4, v18, v14 // (op 9903) 
v_add_f32 v5, v19, v15 // (op 9903) 
v_add_f32 v6, v20, v16 // (op 9903) 
v_add_f32 v7, v21, v17 // (op 9903) 


v_cvt_f16_f32 v14, v4 // (op 9904) 
v_cvt_f16_f32 v15, v5 // (op 9904) 
v_cvt_f16_f32 v16, v6 // (op 9904) 
v_cvt_f16_f32 v17, v7 // (op 9904) 


v_pack_b32_f16 v4, v14, v15 // (op 9905) 
v_pack_b32_f16 v5, v16, v17 // (op 9905) 

s_lshl_b32 s0, s4, 5 // (op 9905) 
v_add_u32 v3, v13, s0 // (op 9905) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 9905) Store value





s_lshl_b32 s0, s2, 6 // (op 9920) 
v_add_u32 v3, v12, s0 // (op 9920) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 9920) Load value

v_accvgpr_read v6, a8 // (op 9916) 
v_accvgpr_read v7, a9 // (op 9916) 
v_accvgpr_read v8, a10 // (op 9916) 
v_accvgpr_read v9, a11 // (op 9916) 
v_mul_f32 v14, v11, v6 // (op 9916) 
v_mul_f32 v15, v11, v7 // (op 9916) 
v_mul_f32 v16, v11, v8 // (op 9916) 
v_mul_f32 v17, v11, v9 // (op 9916) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 9921) 
v_lshrrev_b32 v7, 16, v4 // (op 9921) 
v_cvt_f32_f16 v7, v7 // (op 9921) 
v_mul_f32 v18, v0, v6 // (op 9921) 
v_mul_f32 v19, v0, v7 // (op 9921) 
v_cvt_f32_f16 v6, v5 // (op 9921) 
v_lshrrev_b32 v7, 16, v5 // (op 9921) 
v_cvt_f32_f16 v7, v7 // (op 9921) 
v_mul_f32 v20, v0, v6 // (op 9921) 
v_mul_f32 v21, v0, v7 // (op 9921) 


v_add_f32 v4, v18, v14 // (op 9917) 
v_add_f32 v5, v19, v15 // (op 9917) 
v_add_f32 v6, v20, v16 // (op 9917) 
v_add_f32 v7, v21, v17 // (op 9917) 


v_cvt_f16_f32 v14, v4 // (op 9918) 
v_cvt_f16_f32 v15, v5 // (op 9918) 
v_cvt_f16_f32 v16, v6 // (op 9918) 
v_cvt_f16_f32 v17, v7 // (op 9918) 


v_pack_b32_f16 v4, v14, v15 // (op 9919) 
v_pack_b32_f16 v5, v16, v17 // (op 9919) 

s_lshl_b32 s0, s4, 6 // (op 9919) 
v_add_u32 v3, v13, s0 // (op 9919) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 9919) Store value





s_lshl_b32 s0, s2, 5 // (op 9934) 
s_mul_i32 s1, 3, s0 // (op 9934) 

v_add_u32 v3, v12, s1 // (op 9934) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 9934) Load value

v_accvgpr_read v6, a12 // (op 9930) 
v_accvgpr_read v7, a13 // (op 9930) 
v_accvgpr_read v8, a14 // (op 9930) 
v_accvgpr_read v9, a15 // (op 9930) 
v_mul_f32 v14, v11, v6 // (op 9930) 
v_mul_f32 v15, v11, v7 // (op 9930) 
v_mul_f32 v16, v11, v8 // (op 9930) 
v_mul_f32 v17, v11, v9 // (op 9930) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 9935) 
v_lshrrev_b32 v7, 16, v4 // (op 9935) 
v_cvt_f32_f16 v7, v7 // (op 9935) 
v_mul_f32 v18, v0, v6 // (op 9935) 
v_mul_f32 v19, v0, v7 // (op 9935) 
v_cvt_f32_f16 v6, v5 // (op 9935) 
v_lshrrev_b32 v7, 16, v5 // (op 9935) 
v_cvt_f32_f16 v7, v7 // (op 9935) 
v_mul_f32 v20, v0, v6 // (op 9935) 
v_mul_f32 v21, v0, v7 // (op 9935) 


v_add_f32 v4, v18, v14 // (op 9931) 
v_add_f32 v5, v19, v15 // (op 9931) 
v_add_f32 v6, v20, v16 // (op 9931) 
v_add_f32 v7, v21, v17 // (op 9931) 


v_cvt_f16_f32 v14, v4 // (op 9932) 
v_cvt_f16_f32 v15, v5 // (op 9932) 
v_cvt_f16_f32 v16, v6 // (op 9932) 
v_cvt_f16_f32 v17, v7 // (op 9932) 


v_pack_b32_f16 v4, v14, v15 // (op 9933) 
v_pack_b32_f16 v5, v16, v17 // (op 9933) 

s_lshl_b32 s0, s4, 5 // (op 9933) 
s_mul_i32 s1, 3, s0 // (op 9933) 

v_add_u32 v3, v13, s1 // (op 9933) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 9933) Store value





s_lshl_b32 s0, s2, 7 // (op 9948) 
v_add_u32 v3, v12, s0 // (op 9948) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 9948) Load value

v_accvgpr_read v6, a16 // (op 9944) 
v_accvgpr_read v7, a17 // (op 9944) 
v_accvgpr_read v8, a18 // (op 9944) 
v_accvgpr_read v9, a19 // (op 9944) 
v_mul_f32 v14, v11, v6 // (op 9944) 
v_mul_f32 v15, v11, v7 // (op 9944) 
v_mul_f32 v16, v11, v8 // (op 9944) 
v_mul_f32 v17, v11, v9 // (op 9944) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 9949) 
v_lshrrev_b32 v7, 16, v4 // (op 9949) 
v_cvt_f32_f16 v7, v7 // (op 9949) 
v_mul_f32 v18, v0, v6 // (op 9949) 
v_mul_f32 v19, v0, v7 // (op 9949) 
v_cvt_f32_f16 v6, v5 // (op 9949) 
v_lshrrev_b32 v7, 16, v5 // (op 9949) 
v_cvt_f32_f16 v7, v7 // (op 9949) 
v_mul_f32 v20, v0, v6 // (op 9949) 
v_mul_f32 v21, v0, v7 // (op 9949) 


v_add_f32 v4, v18, v14 // (op 9945) 
v_add_f32 v5, v19, v15 // (op 9945) 
v_add_f32 v6, v20, v16 // (op 9945) 
v_add_f32 v7, v21, v17 // (op 9945) 


v_cvt_f16_f32 v14, v4 // (op 9946) 
v_cvt_f16_f32 v15, v5 // (op 9946) 
v_cvt_f16_f32 v16, v6 // (op 9946) 
v_cvt_f16_f32 v17, v7 // (op 9946) 


v_pack_b32_f16 v4, v14, v15 // (op 9947) 
v_pack_b32_f16 v5, v16, v17 // (op 9947) 

s_lshl_b32 s0, s4, 7 // (op 9947) 
v_add_u32 v3, v13, s0 // (op 9947) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 9947) Store value





s_lshl_b32 s0, s2, 5 // (op 9962) 
s_mul_i32 s1, 5, s0 // (op 9962) 

v_add_u32 v3, v12, s1 // (op 9962) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 9962) Load value

v_accvgpr_read v6, a20 // (op 9958) 
v_accvgpr_read v7, a21 // (op 9958) 
v_accvgpr_read v8, a22 // (op 9958) 
v_accvgpr_read v9, a23 // (op 9958) 
v_mul_f32 v14, v11, v6 // (op 9958) 
v_mul_f32 v15, v11, v7 // (op 9958) 
v_mul_f32 v16, v11, v8 // (op 9958) 
v_mul_f32 v17, v11, v9 // (op 9958) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 9963) 
v_lshrrev_b32 v7, 16, v4 // (op 9963) 
v_cvt_f32_f16 v7, v7 // (op 9963) 
v_mul_f32 v18, v0, v6 // (op 9963) 
v_mul_f32 v19, v0, v7 // (op 9963) 
v_cvt_f32_f16 v6, v5 // (op 9963) 
v_lshrrev_b32 v7, 16, v5 // (op 9963) 
v_cvt_f32_f16 v7, v7 // (op 9963) 
v_mul_f32 v20, v0, v6 // (op 9963) 
v_mul_f32 v21, v0, v7 // (op 9963) 


v_add_f32 v4, v18, v14 // (op 9959) 
v_add_f32 v5, v19, v15 // (op 9959) 
v_add_f32 v6, v20, v16 // (op 9959) 
v_add_f32 v7, v21, v17 // (op 9959) 


v_cvt_f16_f32 v14, v4 // (op 9960) 
v_cvt_f16_f32 v15, v5 // (op 9960) 
v_cvt_f16_f32 v16, v6 // (op 9960) 
v_cvt_f16_f32 v17, v7 // (op 9960) 


v_pack_b32_f16 v4, v14, v15 // (op 9961) 
v_pack_b32_f16 v5, v16, v17 // (op 9961) 

s_lshl_b32 s0, s4, 5 // (op 9961) 
s_mul_i32 s1, 5, s0 // (op 9961) 

v_add_u32 v3, v13, s1 // (op 9961) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 9961) Store value





s_lshl_b32 s0, s2, 5 // (op 9976) 
s_mul_i32 s1, 6, s0 // (op 9976) 

v_add_u32 v3, v12, s1 // (op 9976) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 9976) Load value

v_accvgpr_read v6, a24 // (op 9972) 
v_accvgpr_read v7, a25 // (op 9972) 
v_accvgpr_read v8, a26 // (op 9972) 
v_accvgpr_read v9, a27 // (op 9972) 
v_mul_f32 v14, v11, v6 // (op 9972) 
v_mul_f32 v15, v11, v7 // (op 9972) 
v_mul_f32 v16, v11, v8 // (op 9972) 
v_mul_f32 v17, v11, v9 // (op 9972) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 9977) 
v_lshrrev_b32 v7, 16, v4 // (op 9977) 
v_cvt_f32_f16 v7, v7 // (op 9977) 
v_mul_f32 v18, v0, v6 // (op 9977) 
v_mul_f32 v19, v0, v7 // (op 9977) 
v_cvt_f32_f16 v6, v5 // (op 9977) 
v_lshrrev_b32 v7, 16, v5 // (op 9977) 
v_cvt_f32_f16 v7, v7 // (op 9977) 
v_mul_f32 v20, v0, v6 // (op 9977) 
v_mul_f32 v21, v0, v7 // (op 9977) 


v_add_f32 v4, v18, v14 // (op 9973) 
v_add_f32 v5, v19, v15 // (op 9973) 
v_add_f32 v6, v20, v16 // (op 9973) 
v_add_f32 v7, v21, v17 // (op 9973) 


v_cvt_f16_f32 v14, v4 // (op 9974) 
v_cvt_f16_f32 v15, v5 // (op 9974) 
v_cvt_f16_f32 v16, v6 // (op 9974) 
v_cvt_f16_f32 v17, v7 // (op 9974) 


v_pack_b32_f16 v4, v14, v15 // (op 9975) 
v_pack_b32_f16 v5, v16, v17 // (op 9975) 

s_lshl_b32 s0, s4, 5 // (op 9975) 
s_mul_i32 s1, 6, s0 // (op 9975) 

v_add_u32 v3, v13, s1 // (op 9975) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 9975) Store value





s_lshl_b32 s0, s2, 5 // (op 9990) 
s_mul_i32 s1, 7, s0 // (op 9990) 

v_add_u32 v3, v12, s1 // (op 9990) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 9990) Load value

v_accvgpr_read v6, a28 // (op 9986) 
v_accvgpr_read v7, a29 // (op 9986) 
v_accvgpr_read v8, a30 // (op 9986) 
v_accvgpr_read v9, a31 // (op 9986) 
v_mul_f32 v14, v11, v6 // (op 9986) 
v_mul_f32 v15, v11, v7 // (op 9986) 
v_mul_f32 v16, v11, v8 // (op 9986) 
v_mul_f32 v17, v11, v9 // (op 9986) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 9991) 
v_lshrrev_b32 v7, 16, v4 // (op 9991) 
v_cvt_f32_f16 v7, v7 // (op 9991) 
v_mul_f32 v18, v0, v6 // (op 9991) 
v_mul_f32 v19, v0, v7 // (op 9991) 
v_cvt_f32_f16 v6, v5 // (op 9991) 
v_lshrrev_b32 v7, 16, v5 // (op 9991) 
v_cvt_f32_f16 v7, v7 // (op 9991) 
v_mul_f32 v20, v0, v6 // (op 9991) 
v_mul_f32 v21, v0, v7 // (op 9991) 


v_add_f32 v4, v18, v14 // (op 9987) 
v_add_f32 v5, v19, v15 // (op 9987) 
v_add_f32 v6, v20, v16 // (op 9987) 
v_add_f32 v7, v21, v17 // (op 9987) 


v_cvt_f16_f32 v14, v4 // (op 9988) 
v_cvt_f16_f32 v15, v5 // (op 9988) 
v_cvt_f16_f32 v16, v6 // (op 9988) 
v_cvt_f16_f32 v17, v7 // (op 9988) 


v_pack_b32_f16 v4, v14, v15 // (op 9989) 
v_pack_b32_f16 v5, v16, v17 // (op 9989) 

s_lshl_b32 s0, s4, 5 // (op 9989) 
s_mul_i32 s1, 7, s0 // (op 9989) 

v_add_u32 v3, v13, s1 // (op 9989) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 9989) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__ConditionalBottom_0_beta_116:








































s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM_
.LRR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__end:
.size RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM_, .LRR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM__end-RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM_
  .amdhsa_next_free_vgpr 280
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 98304 // lds bytes
  .amdhsa_accum_offset 152
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
  ProblemType:
    OperationType: GEMM
    DataType: F4
    DataTypeA: F4
    DataTypeB: F4
    DestDataType: h
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
    args: [ { type: int64, semantic: StrideC0 },
            { type: int64, semantic: StrideD0 },
            { type: int64, semantic: SizeFree1 },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int64, semantic: SizeFree0 },
            { type: address, semantic: AddressC },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSum },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha } ]
    macrotile: [128, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM_
    .symbol: RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 98304
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 152
    .agpr_count: 128
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 24
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 32
        .expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 36
        .expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 40
        .expression: {type: MagicMultiple, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 44
        .expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 48
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_4_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 128
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_7_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 136
        .expression: {type: CommandArgument, size: 8, offset: 120, name: Tensor_7_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 144
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 152
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 156
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 160
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 164
        .expression: {type: CommandArgument, size: 4, offset: 204, name: user_Float_Value_14, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 168
        .expression: {type: CommandArgument, size: 4, offset: 200, name: user_Float_Value_12, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
...

.end_amdgpu_metadata

