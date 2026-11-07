// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT



.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 27067) 
s_mul_hi_i32 s1, s16, s10 // (op 27206) 
v_and_b32 v0, 4294967168, v1 // (op 27128) 
v_and_b32 v3, 15, v1 // (op 27128) 
v_add_lshl_u32 v4, v0, v3, 7 // (op 27128) 


v_and_b32 v0, 48, v1 // (op 27128) 
v_add_u32 v3, v4, v0 // (op 27128) 


v_lshrrev_b32 v0, 6, v1 // (op 27185) 
v_and_b32 v4, 1, v0 // (op 27185) 

v_and_b32 v0, 15, v1 // (op 27185) 
v_lshl_add_u32 v5, v4, 7, v0 // (op 27185) 


v_and_b32 v0, 63, v1 // (op 27185) 
v_lshrrev_b32 v4, 4, v0 // (op 27185) 

v_lshlrev_b32 v0, 5, v4 // (op 27185) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 27185) 


v_lshlrev_b32 v0, 2, v4 // (op 27185) 

v_lshrrev_b32 v4, 3, v0 // (op 27185) 

v_and_b32 v0, 4294967168, v1 // (op 27252) 
v_and_b32 v5, 15, v1 // (op 27252) 
v_add_lshl_u32 v6, v0, v5, 7 // (op 27252) 


v_and_b32 v0, 48, v1 // (op 27252) 
v_add_u32 v5, v6, v0 // (op 27252) 


v_lshrrev_b32 v0, 6, v1 // (op 27296) 
v_and_b32 v6, 1, v0 // (op 27296) 

v_and_b32 v0, 15, v1 // (op 27296) 
v_lshl_add_u32 v7, v6, 7, v0 // (op 27296) 


v_and_b32 v0, 63, v1 // (op 27296) 
v_lshrrev_b32 v6, 4, v0 // (op 27296) 

v_lshlrev_b32 v0, 5, v6 // (op 27296) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 27296) 


v_lshlrev_b32 v0, 2, v6 // (op 27296) 

v_lshrrev_b32 v6, 3, v0 // (op 27296) 

v_and_b32 v0, 4294967168, v1 // (op 27067) 
s_add_i32 s18, s0, s16 // (op 27067) 

v_and_b32 v7, 63, v1 // (op 27067) 
s_ashr_i32 s0, s18, 31 // (op 27067) 
v_lshrrev_b32 v8, 4, v7 // (op 27067) 
s_cmp_eq_i32 s10, 0 // (op 27067) 
s_mov_b32 s19, scc // (op 27067) 
v_lshlrev_b32 v9, 2, v8 // (op 27067) 

s_cmp_eq_i32 s19, 0 // (op 27067) 
s_cselect_b32 s49, 0, -1 // (op 27067) 

v_add_u32 v8, v0, v9 // (op 27067) 


s_and_b32 s19, s13, 31 // (op 27067) 
v_lshrrev_b32 v0, 2, v7 // (op 27067) 

v_and_b32 v7, 3, v0 // (op 27067) 

v_and_b32 v0, 4294967168, v1 // (op 27206) 
v_and_b32 v9, 63, v1 // (op 27206) 
s_lshl_b32 s50, 1, s19 // (op 27067) 
v_and_b32 v10, 3, v1 // (op 27067) 
s_add_i32 s51, s50, s49 // (op 27067) 


s_and_b32 s49, s0, s51 // (op 27067) 


s_add_i32 s0, s1, s16 // (op 27206) 

s_ashr_i32 s1, s0, 31 // (op 27206) 
v_lshl_add_u32 v11, v7, 2, v10 // (op 27067) 


s_add_i32 s50, s18, s49 // (op 27067) 


v_lshrrev_b32 v7, 6, v1 // (op 27067) 
s_ashr_i32 s18, s50, s19 // (op 27067) 


v_and_b32 v10, 1, v7 // (op 27067) 

s_lshl_b32 s19, s13, 24 // (op 27067) 
v_lshl_add_u32 v7, v10, 7, v11 // (op 27067) 


s_ashr_i32 s49, s19, 31 // (op 27067) 

s_xor_b32 s19, s18, s49 // (op 27067) 

s_sub_i32 s18, s19, s49 // (op 27067) 


s_mul_hi_i32 s19, s18, s14 // (op 27067) 
s_add_i32 s49, s19, s18 // (op 27067) 

s_ashr_i32 s19, s49, 31 // (op 27067) 
s_cmp_eq_i32 s14, 0 // (op 27067) 
s_mov_b32 s50, scc // (op 27067) 
s_cmp_eq_i32 s50, 0 // (op 27067) 
s_cselect_b32 s51, 0, -1 // (op 27067) 

s_and_b32 s50, s15, 31 // (op 27067) 
v_lshrrev_b32 v10, 4, v9 // (op 27206) 
s_cmp_eq_i32 s10, 0 // (op 27206) 
s_mov_b32 s52, scc // (op 27206) 
v_lshlrev_b32 v11, 2, v10 // (op 27206) 

s_cmp_eq_i32 s52, 0 // (op 27206) 
s_cselect_b32 s53, 0, -1 // (op 27206) 

s_lshl_b32 s52, 1, s50 // (op 27067) 
v_add_u32 v10, v0, v11 // (op 27206) 


v_lshrrev_b32 v0, 2, v9 // (op 27206) 

s_add_i32 s54, s52, s51 // (op 27067) 


s_and_b32 s51, s19, s54 // (op 27067) 


s_add_i32 s19, s49, s51 // (op 27067) 


s_ashr_i32 s49, s19, s50 // (op 27067) 


s_lshl_b32 s19, s15, 24 // (op 27067) 
s_ashr_i32 s50, s19, 31 // (op 27067) 

s_xor_b32 s19, s49, s50 // (op 27067) 

s_sub_i32 s49, s19, s50 // (op 27067) 


s_mul_i32 s19, s18, s11 // (op 27067) 
s_sub_u32 s50, s16, s19 // (op 27067) 

s_mul_i32 s19, s49, s11 // (op 27067) 
s_add_u32 s19, s19, s50 // (op 27067) 

s_mul_hi_i32 s50, s16, s9 // (op 27067) 
s_add_i32 s51, s50, s16 // (op 27067) 

s_ashr_i32 s50, s51, 31 // (op 27067) 
s_cmp_eq_i32 s9, 0 // (op 27067) 
s_mov_b32 s52, scc // (op 27067) 
s_cmp_eq_i32 s52, 0 // (op 27067) 
s_cselect_b32 s54, 0, -1 // (op 27067) 

s_and_b32 s52, s13, 31 // (op 27206) 
v_and_b32 v9, 3, v0 // (op 27206) 

s_lshl_b32 s55, 1, s52 // (op 27206) 
v_and_b32 v0, 3, v1 // (op 27206) 
s_add_i32 s56, s55, s53 // (op 27206) 


s_and_b32 s53, s1, s56 // (op 27206) 


v_lshl_add_u32 v11, v9, 2, v0 // (op 27206) 


s_add_i32 s1, s0, s53 // (op 27206) 


v_lshrrev_b32 v0, 6, v1 // (op 27206) 
s_ashr_i32 s0, s1, s52 // (op 27206) 


v_and_b32 v9, 1, v0 // (op 27206) 

s_lshl_b32 s1, s13, 24 // (op 27206) 
v_lshl_add_u32 v0, v9, 7, v11 // (op 27206) 


s_ashr_i32 s52, s1, 31 // (op 27206) 

s_xor_b32 s1, s0, s52 // (op 27206) 

s_sub_i32 s0, s1, s52 // (op 27206) 


s_mul_hi_i32 s1, s0, s14 // (op 27206) 
s_add_i32 s52, s1, s0 // (op 27206) 

s_ashr_i32 s1, s52, 31 // (op 27206) 
s_cmp_eq_i32 s14, 0 // (op 27206) 
s_mov_b32 s53, scc // (op 27206) 
s_cmp_eq_i32 s53, 0 // (op 27206) 
s_cselect_b32 s55, 0, -1 // (op 27206) 

s_and_b32 s53, s15, 31 // (op 27206) 
s_lshl_b32 s56, 1, s53 // (op 27206) 
s_add_i32 s57, s56, s55 // (op 27206) 


s_and_b32 s55, s1, s57 // (op 27206) 


s_add_i32 s1, s52, s55 // (op 27206) 


s_ashr_i32 s52, s1, s53 // (op 27206) 


s_lshl_b32 s1, s15, 24 // (op 27206) 
s_ashr_i32 s53, s1, 31 // (op 27206) 

s_xor_b32 s1, s52, s53 // (op 27206) 

s_sub_i32 s52, s1, s53 // (op 27206) 


s_mul_i32 s1, s0, s11 // (op 27206) 
s_sub_u32 s53, s16, s1 // (op 27206) 

s_mul_i32 s1, s52, s11 // (op 27206) 
s_add_u32 s1, s1, s53 // (op 27206) 

s_mul_hi_i32 s53, s16, s9 // (op 27206) 
s_add_i32 s55, s53, s16 // (op 27206) 

s_ashr_i32 s53, s55, 31 // (op 27206) 
s_cmp_eq_i32 s9, 0 // (op 27206) 
s_mov_b32 s56, scc // (op 27206) 
s_cmp_eq_i32 s56, 0 // (op 27206) 
s_cselect_b32 s57, 0, -1 // (op 27206) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 27067) 
s_lshl_b32 s58, 1, s56 // (op 27067) 
s_add_i32 s59, s58, s54 // (op 27067) 


s_and_b32 s54, s50, s59 // (op 27067) 


s_add_i32 s50, s51, s54 // (op 27067) 


s_ashr_i32 s51, s50, s56 // (op 27067) 


s_lshl_b32 s50, s45, 24 // (op 27067) 
s_ashr_i32 s54, s50, 31 // (op 27067) 

s_xor_b32 s50, s51, s54 // (op 27067) 

s_sub_i32 s51, s50, s54 // (op 27067) 


s_and_b32 s50, s45, 31 // (op 27206) 
s_lshl_b32 s54, 1, s50 // (op 27206) 
s_add_i32 s56, s54, s57 // (op 27206) 


s_and_b32 s54, s53, s56 // (op 27206) 


s_add_i32 s53, s55, s54 // (op 27206) 


s_ashr_i32 s54, s53, s50 // (op 27206) 


s_lshl_b32 s50, s45, 24 // (op 27206) 
s_ashr_i32 s53, s50, 31 // (op 27206) 

s_xor_b32 s50, s54, s53 // (op 27206) 

s_sub_i32 s54, s50, s53 // (op 27206) 


s_add_u32 s56, s6, 256 // (op 27067) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 27067) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 27067) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 27067) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 27067) 

s_mul_i32 s50, s11, s56 // (op 27067) 
s_mul_i32 s53, s8, s50 // (op 27067) 

s_mul_i32 s50, s51, s53 // (op 27067) 

s_sub_u32 s53, s16, s50 // (op 27067) 

s_mov_b32 s50, s53 // (op 27067) 
s_mul_hi_i32 s55, s50, s12 // (op 27067) 
s_add_i32 s58, s55, s50 // (op 27067) 


s_ashr_i32 s50, s58, 31 // (op 27067) 
s_cmp_eq_i32 s12, 0 // (op 27067) 
s_mov_b32 s55, scc // (op 27067) 
s_cmp_eq_i32 s55, 0 // (op 27067) 
s_cselect_b32 s59, 0, -1 // (op 27067) 

s_and_b32 s55, s44, 31 // (op 27067) 
s_lshl_b32 s60, 1, s55 // (op 27067) 
s_add_i32 s61, s60, s59 // (op 27067) 


s_and_b32 s59, s50, s61 // (op 27067) 


s_add_i32 s50, s58, s59 // (op 27067) 


s_ashr_i32 s58, s50, s55 // (op 27067) 


s_lshl_b32 s50, s44, 24 // (op 27067) 
s_ashr_i32 s55, s50, 31 // (op 27067) 

s_xor_b32 s50, s58, s55 // (op 27067) 

s_sub_i32 s58, s50, s55 // (op 27067) 


s_mul_i32 s50, s46, s11 // (op 27067) 
s_add_u32 s60, s20, 256 // (op 27067) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 27067) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 27067) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 27067) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 8 // (op 27067) 

s_sub_i32 s55, s60, s50 // (op 27067) 

s_mul_i32 s59, s58, s55 // (op 27067) 

s_sub_u32 s55, s53, s59 // (op 27067) 


s_add_u32 s53, s55, s50 // (op 27067) 


s_cmp_eq_i32 s51, 0 // (op 27067) 
s_mov_b32 s50, scc // (op 27067) 

s_cmp_eq_i32 s50, 0 // (op 27067) 
s_cselect_b32 s51, s53, s19 // (op 27067) 


s_mul_i32 s19, s49, s56 // (op 27067) 


v_lshl_add_u32 v9, s51, 8, v8 // (op 27067) 


s_sub_i32 s49, s18, s19 // (op 27067) 


s_cmp_eq_i32 s50, 0 // (op 27067) 
s_cselect_b32 s51, s58, s49 // (op 27067) 



v_lshl_add_u32 v8, s51, 8, v7 // (op 27067) 


v_mul_lo_u32 v7, v8, s2 // (op 27067) 

v_add_lshl_u32 v8, v9, v7, 1 // (op 27067) 


s_add_u32 s18, s6, 256 // (op 27206) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 27206) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 27206) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 27206) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 8 // (op 27206) 

s_mul_i32 s53, s11, s18 // (op 27206) 
s_mul_i32 s49, s8, s53 // (op 27206) 

s_mul_i32 s53, s54, s49 // (op 27206) 

s_sub_u32 s49, s16, s53 // (op 27206) 

s_mov_b32 s53, s49 // (op 27206) 
s_mul_hi_i32 s50, s53, s12 // (op 27206) 
s_add_i32 s51, s50, s53 // (op 27206) 


s_ashr_i32 s50, s51, 31 // (op 27206) 
s_cmp_eq_i32 s12, 0 // (op 27206) 
s_mov_b32 s53, scc // (op 27206) 
s_cmp_eq_i32 s53, 0 // (op 27206) 
s_cselect_b32 s55, 0, -1 // (op 27206) 

s_and_b32 s53, s44, 31 // (op 27206) 
s_lshl_b32 s56, 1, s53 // (op 27206) 
s_add_i32 s57, s56, s55 // (op 27206) 


s_and_b32 s55, s50, s57 // (op 27206) 


s_add_i32 s50, s51, s55 // (op 27206) 


s_ashr_i32 s51, s50, s53 // (op 27206) 


s_lshl_b32 s50, s44, 24 // (op 27206) 
s_ashr_i32 s53, s50, 31 // (op 27206) 

s_xor_b32 s50, s51, s53 // (op 27206) 

s_sub_i32 s51, s50, s53 // (op 27206) 


s_mul_i32 s50, s46, s11 // (op 27206) 
s_add_u32 s56, s20, 256 // (op 27206) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 27206) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 27206) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 27206) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 27206) 

s_sub_i32 s53, s56, s50 // (op 27206) 

s_mul_i32 s55, s51, s53 // (op 27206) 

s_sub_u32 s53, s49, s55 // (op 27206) 


s_add_u32 s49, s53, s50 // (op 27206) 


s_cmp_eq_i32 s54, 0 // (op 27206) 
s_mov_b32 s50, scc // (op 27206) 

s_cmp_eq_i32 s50, 0 // (op 27206) 
s_cselect_b32 s53, s49, s1 // (op 27206) 


s_mul_i32 s1, s52, s18 // (op 27206) 


v_lshl_add_u32 v7, s53, 8, v10 // (op 27206) 


s_sub_i32 s49, s0, s1 // (op 27206) 


s_cmp_eq_i32 s50, 0 // (op 27206) 
s_cselect_b32 s0, s51, s49 // (op 27206) 



v_lshl_add_u32 v9, s0, 8, v0 // (op 27206) 


v_mul_lo_u32 v0, v9, s4 // (op 27206) 

v_add_lshl_u32 v9, v7, v0, 1 // (op 27206) 


s_mul_i32 s1, s2, s7 // (op 27069) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 27069) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 27069) least significant
s_mul_hi_u32 s19, s2, s6 // (op 27069) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 27069) most significant: sum
s_add_u32 s1, s1, s19 // (op 27069) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 27069) 

s_mul_i32 s1, s4, s7 // (op 27208) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 27208) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 27208) least significant
s_mul_hi_u32 s50, s4, s6 // (op 27208) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 27208) most significant: sum
s_add_u32 s1, s1, s50 // (op 27208) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 27208) 

s_mov_b32 s52, s22 // (op 27069) reg expression
s_mov_b32 s53, s23 // (op 27069) reg expression
s_mov_b32 s54, s18 // (op 27069) reg expression
s_mov_b32 s55, 131072 // (op 27069) call()

s_mov_b32 s56, s24 // (op 27208) reg expression

s_mov_b32 s57, s25 // (op 27208) reg expression
s_mov_b32 s58, s50 // (op 27208) reg expression
s_mov_b32 s59, 131072 // (op 27208) call()


v_mov_b32 v0, s47 // (op 2) Move value
v_mov_b32 v7, s48 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 57) call()
v_accvgpr_write a1, 0.00000 // (op 57) call()
v_accvgpr_write a2, 0.00000 // (op 57) call()
v_accvgpr_write a3, 0.00000 // (op 57) call()
v_accvgpr_write a4, 0.00000 // (op 281) call()
v_accvgpr_write a5, 0.00000 // (op 281) call()
v_accvgpr_write a6, 0.00000 // (op 281) call()
v_accvgpr_write a7, 0.00000 // (op 281) call()
v_accvgpr_write a8, 0.00000 // (op 417) call()
v_accvgpr_write a9, 0.00000 // (op 417) call()
v_accvgpr_write a10, 0.00000 // (op 417) call()
v_accvgpr_write a11, 0.00000 // (op 417) call()
v_accvgpr_write a12, 0.00000 // (op 553) call()
v_accvgpr_write a13, 0.00000 // (op 553) call()
v_accvgpr_write a14, 0.00000 // (op 553) call()
v_accvgpr_write a15, 0.00000 // (op 553) call()
v_accvgpr_write a16, 0.00000 // (op 689) call()
v_accvgpr_write a17, 0.00000 // (op 689) call()
v_accvgpr_write a18, 0.00000 // (op 689) call()
v_accvgpr_write a19, 0.00000 // (op 689) call()
v_accvgpr_write a20, 0.00000 // (op 825) call()
v_accvgpr_write a21, 0.00000 // (op 825) call()
v_accvgpr_write a22, 0.00000 // (op 825) call()
v_accvgpr_write a23, 0.00000 // (op 825) call()
v_accvgpr_write a24, 0.00000 // (op 961) call()
v_accvgpr_write a25, 0.00000 // (op 961) call()
v_accvgpr_write a26, 0.00000 // (op 961) call()
v_accvgpr_write a27, 0.00000 // (op 961) call()
v_accvgpr_write a28, 0.00000 // (op 1097) call()
v_accvgpr_write a29, 0.00000 // (op 1097) call()
v_accvgpr_write a30, 0.00000 // (op 1097) call()
v_accvgpr_write a31, 0.00000 // (op 1097) call()
v_accvgpr_write a32, 0.00000 // (op 1409) call()
v_accvgpr_write a33, 0.00000 // (op 1409) call()
v_accvgpr_write a34, 0.00000 // (op 1409) call()
v_accvgpr_write a35, 0.00000 // (op 1409) call()
v_accvgpr_write a36, 0.00000 // (op 1930) call()
v_accvgpr_write a37, 0.00000 // (op 1930) call()
v_accvgpr_write a38, 0.00000 // (op 1930) call()
v_accvgpr_write a39, 0.00000 // (op 1930) call()
v_accvgpr_write a40, 0.00000 // (op 1931) call()
v_accvgpr_write a41, 0.00000 // (op 1931) call()
v_accvgpr_write a42, 0.00000 // (op 1931) call()
v_accvgpr_write a43, 0.00000 // (op 1931) call()
v_accvgpr_write a44, 0.00000 // (op 1932) call()
v_accvgpr_write a45, 0.00000 // (op 1932) call()
v_accvgpr_write a46, 0.00000 // (op 1932) call()
v_accvgpr_write a47, 0.00000 // (op 1932) call()
v_accvgpr_write a48, 0.00000 // (op 1933) call()
v_accvgpr_write a49, 0.00000 // (op 1933) call()
v_accvgpr_write a50, 0.00000 // (op 1933) call()
v_accvgpr_write a51, 0.00000 // (op 1933) call()
v_accvgpr_write a52, 0.00000 // (op 1934) call()
v_accvgpr_write a53, 0.00000 // (op 1934) call()
v_accvgpr_write a54, 0.00000 // (op 1934) call()
v_accvgpr_write a55, 0.00000 // (op 1934) call()
v_accvgpr_write a56, 0.00000 // (op 1935) call()
v_accvgpr_write a57, 0.00000 // (op 1935) call()
v_accvgpr_write a58, 0.00000 // (op 1935) call()
v_accvgpr_write a59, 0.00000 // (op 1935) call()
v_accvgpr_write a60, 0.00000 // (op 1936) call()
v_accvgpr_write a61, 0.00000 // (op 1936) call()
v_accvgpr_write a62, 0.00000 // (op 1936) call()
v_accvgpr_write a63, 0.00000 // (op 1936) call()
v_accvgpr_write a64, 0.00000 // (op 2627) call()
v_accvgpr_write a65, 0.00000 // (op 2627) call()
v_accvgpr_write a66, 0.00000 // (op 2627) call()
v_accvgpr_write a67, 0.00000 // (op 2627) call()
v_accvgpr_write a68, 0.00000 // (op 3148) call()
v_accvgpr_write a69, 0.00000 // (op 3148) call()
v_accvgpr_write a70, 0.00000 // (op 3148) call()
v_accvgpr_write a71, 0.00000 // (op 3148) call()
v_accvgpr_write a72, 0.00000 // (op 3149) call()
v_accvgpr_write a73, 0.00000 // (op 3149) call()
v_accvgpr_write a74, 0.00000 // (op 3149) call()
v_accvgpr_write a75, 0.00000 // (op 3149) call()
v_accvgpr_write a76, 0.00000 // (op 3150) call()
v_accvgpr_write a77, 0.00000 // (op 3150) call()
v_accvgpr_write a78, 0.00000 // (op 3150) call()
v_accvgpr_write a79, 0.00000 // (op 3150) call()
v_accvgpr_write a80, 0.00000 // (op 3151) call()
v_accvgpr_write a81, 0.00000 // (op 3151) call()
v_accvgpr_write a82, 0.00000 // (op 3151) call()
v_accvgpr_write a83, 0.00000 // (op 3151) call()
v_accvgpr_write a84, 0.00000 // (op 3152) call()
v_accvgpr_write a85, 0.00000 // (op 3152) call()
v_accvgpr_write a86, 0.00000 // (op 3152) call()
v_accvgpr_write a87, 0.00000 // (op 3152) call()
v_accvgpr_write a88, 0.00000 // (op 3153) call()
v_accvgpr_write a89, 0.00000 // (op 3153) call()
v_accvgpr_write a90, 0.00000 // (op 3153) call()
v_accvgpr_write a91, 0.00000 // (op 3153) call()
v_accvgpr_write a92, 0.00000 // (op 3154) call()
v_accvgpr_write a93, 0.00000 // (op 3154) call()
v_accvgpr_write a94, 0.00000 // (op 3154) call()
v_accvgpr_write a95, 0.00000 // (op 3154) call()
v_accvgpr_write a96, 0.00000 // (op 3845) call()
v_accvgpr_write a97, 0.00000 // (op 3845) call()
v_accvgpr_write a98, 0.00000 // (op 3845) call()
v_accvgpr_write a99, 0.00000 // (op 3845) call()
v_accvgpr_write a100, 0.00000 // (op 4366) call()
v_accvgpr_write a101, 0.00000 // (op 4366) call()
v_accvgpr_write a102, 0.00000 // (op 4366) call()
v_accvgpr_write a103, 0.00000 // (op 4366) call()
v_accvgpr_write a104, 0.00000 // (op 4367) call()
v_accvgpr_write a105, 0.00000 // (op 4367) call()
v_accvgpr_write a106, 0.00000 // (op 4367) call()
v_accvgpr_write a107, 0.00000 // (op 4367) call()
v_accvgpr_write a108, 0.00000 // (op 4368) call()
v_accvgpr_write a109, 0.00000 // (op 4368) call()
v_accvgpr_write a110, 0.00000 // (op 4368) call()
v_accvgpr_write a111, 0.00000 // (op 4368) call()
v_accvgpr_write a112, 0.00000 // (op 4369) call()
v_accvgpr_write a113, 0.00000 // (op 4369) call()
v_accvgpr_write a114, 0.00000 // (op 4369) call()
v_accvgpr_write a115, 0.00000 // (op 4369) call()
v_accvgpr_write a116, 0.00000 // (op 4370) call()
v_accvgpr_write a117, 0.00000 // (op 4370) call()
v_accvgpr_write a118, 0.00000 // (op 4370) call()
v_accvgpr_write a119, 0.00000 // (op 4370) call()
v_accvgpr_write a120, 0.00000 // (op 4371) call()
v_accvgpr_write a121, 0.00000 // (op 4371) call()
v_accvgpr_write a122, 0.00000 // (op 4371) call()
v_accvgpr_write a123, 0.00000 // (op 4371) call()
v_accvgpr_write a124, 0.00000 // (op 4372) call()
v_accvgpr_write a125, 0.00000 // (op 4372) call()
v_accvgpr_write a126, 0.00000 // (op 4372) call()
v_accvgpr_write a127, 0.00000 // (op 4372) call()
v_accvgpr_write a128, 0.00000 // (op 5063) call()
v_accvgpr_write a129, 0.00000 // (op 5063) call()
v_accvgpr_write a130, 0.00000 // (op 5063) call()
v_accvgpr_write a131, 0.00000 // (op 5063) call()
v_accvgpr_write a132, 0.00000 // (op 5584) call()
v_accvgpr_write a133, 0.00000 // (op 5584) call()
v_accvgpr_write a134, 0.00000 // (op 5584) call()
v_accvgpr_write a135, 0.00000 // (op 5584) call()
v_accvgpr_write a136, 0.00000 // (op 5585) call()
v_accvgpr_write a137, 0.00000 // (op 5585) call()
v_accvgpr_write a138, 0.00000 // (op 5585) call()
v_accvgpr_write a139, 0.00000 // (op 5585) call()
v_accvgpr_write a140, 0.00000 // (op 5586) call()
v_accvgpr_write a141, 0.00000 // (op 5586) call()
v_accvgpr_write a142, 0.00000 // (op 5586) call()
v_accvgpr_write a143, 0.00000 // (op 5586) call()
v_accvgpr_write a144, 0.00000 // (op 5587) call()
v_accvgpr_write a145, 0.00000 // (op 5587) call()
v_accvgpr_write a146, 0.00000 // (op 5587) call()
v_accvgpr_write a147, 0.00000 // (op 5587) call()
v_accvgpr_write a148, 0.00000 // (op 5588) call()
v_accvgpr_write a149, 0.00000 // (op 5588) call()
v_accvgpr_write a150, 0.00000 // (op 5588) call()
v_accvgpr_write a151, 0.00000 // (op 5588) call()
v_accvgpr_write a152, 0.00000 // (op 5589) call()
v_accvgpr_write a153, 0.00000 // (op 5589) call()
v_accvgpr_write a154, 0.00000 // (op 5589) call()
v_accvgpr_write a155, 0.00000 // (op 5589) call()
v_accvgpr_write a156, 0.00000 // (op 5590) call()
v_accvgpr_write a157, 0.00000 // (op 5590) call()
v_accvgpr_write a158, 0.00000 // (op 5590) call()
v_accvgpr_write a159, 0.00000 // (op 5590) call()
v_accvgpr_write a160, 0.00000 // (op 6281) call()
v_accvgpr_write a161, 0.00000 // (op 6281) call()
v_accvgpr_write a162, 0.00000 // (op 6281) call()
v_accvgpr_write a163, 0.00000 // (op 6281) call()
v_accvgpr_write a164, 0.00000 // (op 6802) call()
v_accvgpr_write a165, 0.00000 // (op 6802) call()
v_accvgpr_write a166, 0.00000 // (op 6802) call()
v_accvgpr_write a167, 0.00000 // (op 6802) call()
v_accvgpr_write a168, 0.00000 // (op 6803) call()
v_accvgpr_write a169, 0.00000 // (op 6803) call()
v_accvgpr_write a170, 0.00000 // (op 6803) call()
v_accvgpr_write a171, 0.00000 // (op 6803) call()
v_accvgpr_write a172, 0.00000 // (op 6804) call()
v_accvgpr_write a173, 0.00000 // (op 6804) call()
v_accvgpr_write a174, 0.00000 // (op 6804) call()
v_accvgpr_write a175, 0.00000 // (op 6804) call()
v_accvgpr_write a176, 0.00000 // (op 6805) call()
v_accvgpr_write a177, 0.00000 // (op 6805) call()
v_accvgpr_write a178, 0.00000 // (op 6805) call()
v_accvgpr_write a179, 0.00000 // (op 6805) call()
v_accvgpr_write a180, 0.00000 // (op 6806) call()
v_accvgpr_write a181, 0.00000 // (op 6806) call()
v_accvgpr_write a182, 0.00000 // (op 6806) call()
v_accvgpr_write a183, 0.00000 // (op 6806) call()
v_accvgpr_write a184, 0.00000 // (op 6807) call()
v_accvgpr_write a185, 0.00000 // (op 6807) call()
v_accvgpr_write a186, 0.00000 // (op 6807) call()
v_accvgpr_write a187, 0.00000 // (op 6807) call()
v_accvgpr_write a188, 0.00000 // (op 6808) call()
v_accvgpr_write a189, 0.00000 // (op 6808) call()
v_accvgpr_write a190, 0.00000 // (op 6808) call()
v_accvgpr_write a191, 0.00000 // (op 6808) call()
v_accvgpr_write a192, 0.00000 // (op 7499) call()
v_accvgpr_write a193, 0.00000 // (op 7499) call()
v_accvgpr_write a194, 0.00000 // (op 7499) call()
v_accvgpr_write a195, 0.00000 // (op 7499) call()
v_accvgpr_write a196, 0.00000 // (op 8020) call()
v_accvgpr_write a197, 0.00000 // (op 8020) call()
v_accvgpr_write a198, 0.00000 // (op 8020) call()
v_accvgpr_write a199, 0.00000 // (op 8020) call()
v_accvgpr_write a200, 0.00000 // (op 8021) call()
v_accvgpr_write a201, 0.00000 // (op 8021) call()
v_accvgpr_write a202, 0.00000 // (op 8021) call()
v_accvgpr_write a203, 0.00000 // (op 8021) call()
v_accvgpr_write a204, 0.00000 // (op 8022) call()
v_accvgpr_write a205, 0.00000 // (op 8022) call()
v_accvgpr_write a206, 0.00000 // (op 8022) call()
v_accvgpr_write a207, 0.00000 // (op 8022) call()
v_accvgpr_write a208, 0.00000 // (op 8023) call()
v_accvgpr_write a209, 0.00000 // (op 8023) call()
v_accvgpr_write a210, 0.00000 // (op 8023) call()
v_accvgpr_write a211, 0.00000 // (op 8023) call()
v_accvgpr_write a212, 0.00000 // (op 8024) call()
v_accvgpr_write a213, 0.00000 // (op 8024) call()
v_accvgpr_write a214, 0.00000 // (op 8024) call()
v_accvgpr_write a215, 0.00000 // (op 8024) call()
v_accvgpr_write a216, 0.00000 // (op 8025) call()
v_accvgpr_write a217, 0.00000 // (op 8025) call()
v_accvgpr_write a218, 0.00000 // (op 8025) call()
v_accvgpr_write a219, 0.00000 // (op 8025) call()
v_accvgpr_write a220, 0.00000 // (op 8026) call()
v_accvgpr_write a221, 0.00000 // (op 8026) call()
v_accvgpr_write a222, 0.00000 // (op 8026) call()
v_accvgpr_write a223, 0.00000 // (op 8026) call()
v_accvgpr_write a224, 0.00000 // (op 8717) call()
v_accvgpr_write a225, 0.00000 // (op 8717) call()
v_accvgpr_write a226, 0.00000 // (op 8717) call()
v_accvgpr_write a227, 0.00000 // (op 8717) call()
v_accvgpr_write a228, 0.00000 // (op 9238) call()
v_accvgpr_write a229, 0.00000 // (op 9238) call()
v_accvgpr_write a230, 0.00000 // (op 9238) call()
v_accvgpr_write a231, 0.00000 // (op 9238) call()
v_accvgpr_write a232, 0.00000 // (op 9239) call()
v_accvgpr_write a233, 0.00000 // (op 9239) call()
v_accvgpr_write a234, 0.00000 // (op 9239) call()
v_accvgpr_write a235, 0.00000 // (op 9239) call()
v_accvgpr_write a236, 0.00000 // (op 9240) call()
v_accvgpr_write a237, 0.00000 // (op 9240) call()
v_accvgpr_write a238, 0.00000 // (op 9240) call()
v_accvgpr_write a239, 0.00000 // (op 9240) call()
v_accvgpr_write a240, 0.00000 // (op 9241) call()
v_accvgpr_write a241, 0.00000 // (op 9241) call()
v_accvgpr_write a242, 0.00000 // (op 9241) call()
v_accvgpr_write a243, 0.00000 // (op 9241) call()
v_accvgpr_write a244, 0.00000 // (op 9242) call()
v_accvgpr_write a245, 0.00000 // (op 9242) call()
v_accvgpr_write a246, 0.00000 // (op 9242) call()
v_accvgpr_write a247, 0.00000 // (op 9242) call()
v_accvgpr_write a248, 0.00000 // (op 9243) call()
v_accvgpr_write a249, 0.00000 // (op 9243) call()
v_accvgpr_write a250, 0.00000 // (op 9243) call()
v_accvgpr_write a251, 0.00000 // (op 9243) call()
v_accvgpr_write a252, 0.00000 // (op 9244) call()
v_accvgpr_write a253, 0.00000 // (op 9244) call()
v_accvgpr_write a254, 0.00000 // (op 9244) call()
v_accvgpr_write a255, 0.00000 // (op 9244) call()
s_mul_hi_i32 s0, s16, s10 // (op 26957) 
v_lshrrev_b32 v10, 3, v1 // (op 26957) 
s_add_i32 s1, s0, s16 // (op 26957) 

v_and_b32 v11, -8, v1 // (op 26957) 
s_ashr_i32 s0, s1, 31 // (op 26957) 
v_sub_u32 v12, v1, v11 // (op 26957) 

s_cmp_eq_i32 s10, 0 // (op 26957) 
s_mov_b32 s18, scc // (op 26957) 
v_lshlrev_b32 v11, 4, v12 // (op 26957) 

s_cmp_eq_i32 s18, 0 // (op 26957) 
s_cselect_b32 s19, 0, -1 // (op 26957) 

v_mov_b32 v13, 0 // (op 26957) convert
v_mov_b32 v12, v11 // (op 26957) convert

s_and_b32 s18, s13, 31 // (op 26957) 
s_lshl_b32 s22, 1, s18 // (op 26957) 
s_add_i32 s23, s22, s19 // (op 26957) 


s_and_b32 s19, s0, s23 // (op 26957) 


s_add_i32 s0, s1, s19 // (op 26957) 


s_ashr_i32 s1, s0, s18 // (op 26957) 


s_lshl_b32 s0, s13, 24 // (op 26957) 
s_ashr_i32 s18, s0, 31 // (op 26957) 

s_xor_b32 s0, s1, s18 // (op 26957) 

s_sub_i32 s1, s0, s18 // (op 26957) 


s_mul_hi_i32 s0, s1, s14 // (op 26957) 
s_add_i32 s18, s0, s1 // (op 26957) 

s_ashr_i32 s0, s18, 31 // (op 26957) 
s_cmp_eq_i32 s14, 0 // (op 26957) 
s_mov_b32 s19, scc // (op 26957) 
s_cmp_eq_i32 s19, 0 // (op 26957) 
s_cselect_b32 s22, 0, -1 // (op 26957) 

s_and_b32 s19, s15, 31 // (op 26957) 
s_lshl_b32 s23, 1, s19 // (op 26957) 
s_add_i32 s24, s23, s22 // (op 26957) 


s_and_b32 s25, s0, s24 // (op 26957) 


s_add_i32 s0, s18, s25 // (op 26957) 


s_ashr_i32 s18, s0, s19 // (op 26957) 


s_lshl_b32 s0, s15, 24 // (op 26957) 
s_ashr_i32 s19, s0, 31 // (op 26957) 

s_xor_b32 s0, s18, s19 // (op 26957) 

s_sub_i32 s18, s0, s19 // (op 26957) 


s_mul_i32 s0, s1, s11 // (op 26957) 

s_sub_u32 s1, s16, s0 // (op 26957) 

s_mul_i32 s0, s18, s11 // (op 26957) 
s_add_u32 s0, s0, s1 // (op 26957) 


s_mul_hi_i32 s1, s16, s9 // (op 26957) 
s_add_i32 s18, s1, s16 // (op 26957) 

s_ashr_i32 s1, s18, 31 // (op 26957) 
s_cmp_eq_i32 s9, 0 // (op 26957) 
s_mov_b32 s19, scc // (op 26957) 
s_cmp_eq_i32 s19, 0 // (op 26957) 
s_cselect_b32 s22, 0, -1 // (op 26957) 

s_and_b32 s19, s45, 31 // (op 26957) 
s_lshl_b32 s23, 1, s19 // (op 26957) 
s_add_i32 s24, s23, s22 // (op 26957) 


s_and_b32 s25, s1, s24 // (op 26957) 


s_add_i32 s1, s18, s25 // (op 26957) 


s_ashr_i32 s18, s1, s19 // (op 26957) 


s_lshl_b32 s1, s45, 24 // (op 26957) 
s_ashr_i32 s19, s1, 31 // (op 26957) 

s_xor_b32 s1, s18, s19 // (op 26957) 

s_sub_i32 s18, s1, s19 // (op 26957) 


s_add_u32 s22, s6, 256 // (op 26957) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 26957) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 26957) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 26957) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 26957) 

s_mul_i32 s1, s11, s22 // (op 26957) 

s_mul_i32 s19, s8, s1 // (op 26957) 

s_mul_i32 s1, s18, s19 // (op 26957) 

s_sub_u32 s19, s16, s1 // (op 26957) 

s_mov_b32 s1, s19 // (op 26957) 
s_mul_hi_i32 s22, s1, s12 // (op 26957) 
s_add_i32 s23, s22, s1 // (op 26957) 


s_ashr_i32 s1, s23, 31 // (op 26957) 
s_cmp_eq_i32 s12, 0 // (op 26957) 
s_mov_b32 s22, scc // (op 26957) 
s_cmp_eq_i32 s22, 0 // (op 26957) 
s_cselect_b32 s24, 0, -1 // (op 26957) 

s_and_b32 s22, s44, 31 // (op 26957) 
s_lshl_b32 s25, 1, s22 // (op 26957) 
s_add_i32 s47, s25, s24 // (op 26957) 


s_and_b32 s24, s1, s47 // (op 26957) 


s_add_i32 s1, s23, s24 // (op 26957) 


s_ashr_i32 s23, s1, s22 // (op 26957) 


s_lshl_b32 s1, s44, 24 // (op 26957) 
s_ashr_i32 s22, s1, 31 // (op 26957) 

s_xor_b32 s1, s23, s22 // (op 26957) 

s_sub_i32 s23, s1, s22 // (op 26957) 


s_mul_i32 s1, s46, s11 // (op 26957) 
s_add_u32 s24, s20, 256 // (op 26957) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 26957) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 26957) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 26957) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 26957) 

s_sub_i32 s22, s24, s1 // (op 26957) 

s_mul_i32 s24, s23, s22 // (op 26957) 


s_sub_u32 s25, s19, s24 // (op 26957) 


s_add_u32 s19, s25, s1 // (op 26957) 


s_cmp_eq_i32 s18, 0 // (op 26957) 
s_mov_b32 s1, scc // (op 26957) 

s_cmp_eq_i32 s1, 0 // (op 26957) 
s_cselect_b32 s18, s19, s0 // (op 26957) 



v_lshl_add_u32 v11, s18, 8, v10 // (op 26957) 


v_mul_lo_u32 v15, v11, s27 // (op 26957) most significant: low of low * high
v_mul_hi_u32 v10, v11, s26 // (op 26957) most significant: high of low * low
v_mul_lo_u32 v14, v11, s26 // (op 26957) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 26957) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 26957) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 26957) most significant half



s_mul_i32 s1, s26, s21 // (op 26959) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 26959) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 26959) least significant
s_mul_hi_u32 s19, s26, s20 // (op 26959) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 26959) most significant: sum
s_add_u32 s1, s1, s19 // (op 26959) most significant: sum


s_mov_b32 s48, s28 // (op 26959) reg expression
s_mov_b32 s49, s29 // (op 26959) reg expression
s_mov_b32 s50, s0 // (op 26959) reg expression
s_mov_b32 s51, 131072 // (op 26959) call()

s_mul_hi_i32 s47, s16, s10 // (op 26976) 
v_lshrrev_b32 v10, 3, v1 // (op 26976) 
s_add_i32 s0, s47, s16 // (op 26976) 

v_and_b32 v11, -8, v1 // (op 26976) 
s_ashr_i32 s1, s0, 31 // (op 26976) 
v_sub_u32 v12, v1, v11 // (op 26976) 

s_cmp_eq_i32 s10, 0 // (op 26976) 
s_mov_b32 s47, scc // (op 26976) 
v_lshlrev_b32 v11, 4, v12 // (op 26976) 

s_cmp_eq_i32 s47, 0 // (op 26976) 
s_cselect_b32 s18, 0, -1 // (op 26976) 

v_mov_b32 v13, 0 // (op 26976) convert
v_mov_b32 v12, v11 // (op 26976) convert

s_and_b32 s19, s13, 31 // (op 26976) 
s_lshl_b32 s47, 1, s19 // (op 26976) 
s_add_i32 s22, s47, s18 // (op 26976) 


s_and_b32 s18, s1, s22 // (op 26976) 


s_add_i32 s1, s0, s18 // (op 26976) 


s_ashr_i32 s0, s1, s19 // (op 26976) 


s_lshl_b32 s1, s13, 24 // (op 26976) 
s_ashr_i32 s47, s1, 31 // (op 26976) 

s_xor_b32 s1, s0, s47 // (op 26976) 

s_sub_i32 s0, s1, s47 // (op 26976) 


s_mul_hi_i32 s1, s0, s14 // (op 26976) 
s_add_i32 s47, s1, s0 // (op 26976) 

s_ashr_i32 s1, s47, 31 // (op 26976) 
s_cmp_eq_i32 s14, 0 // (op 26976) 
s_mov_b32 s18, scc // (op 26976) 
s_cmp_eq_i32 s18, 0 // (op 26976) 
s_cselect_b32 s19, 0, -1 // (op 26976) 

s_and_b32 s18, s15, 31 // (op 26976) 
s_lshl_b32 s22, 1, s18 // (op 26976) 
s_add_i32 s23, s22, s19 // (op 26976) 


s_and_b32 s19, s1, s23 // (op 26976) 


s_add_i32 s1, s47, s19 // (op 26976) 


s_ashr_i32 s19, s1, s18 // (op 26976) 


s_lshl_b32 s1, s15, 24 // (op 26976) 
s_ashr_i32 s18, s1, 31 // (op 26976) 

s_xor_b32 s1, s19, s18 // (op 26976) 

s_sub_i32 s19, s1, s18 // (op 26976) 


s_mul_i32 s1, s0, s11 // (op 26976) 

s_sub_u32 s0, s16, s1 // (op 26976) 

s_mul_i32 s1, s19, s11 // (op 26976) 
s_add_u32 s1, s1, s0 // (op 26976) 


s_mul_hi_i32 s0, s16, s9 // (op 26976) 
s_add_i32 s47, s0, s16 // (op 26976) 

s_ashr_i32 s0, s47, 31 // (op 26976) 
s_cmp_eq_i32 s9, 0 // (op 26976) 
s_mov_b32 s18, scc // (op 26976) 
s_cmp_eq_i32 s18, 0 // (op 26976) 
s_cselect_b32 s19, 0, -1 // (op 26976) 

s_and_b32 s18, s45, 31 // (op 26976) 
s_lshl_b32 s22, 1, s18 // (op 26976) 
s_add_i32 s23, s22, s19 // (op 26976) 


s_and_b32 s19, s0, s23 // (op 26976) 


s_add_i32 s0, s47, s19 // (op 26976) 


s_ashr_i32 s19, s0, s18 // (op 26976) 


s_lshl_b32 s0, s45, 24 // (op 26976) 
s_ashr_i32 s18, s0, 31 // (op 26976) 

s_xor_b32 s0, s19, s18 // (op 26976) 

s_sub_i32 s19, s0, s18 // (op 26976) 


s_add_u32 s22, s6, 256 // (op 26976) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 26976) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 26976) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 26976) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 26976) 

s_mul_i32 s0, s11, s22 // (op 26976) 

s_mul_i32 s18, s8, s0 // (op 26976) 

s_mul_i32 s0, s19, s18 // (op 26976) 

s_sub_u32 s18, s16, s0 // (op 26976) 

s_mov_b32 s0, s18 // (op 26976) 
s_mul_hi_i32 s47, s0, s12 // (op 26976) 
s_add_i32 s22, s47, s0 // (op 26976) 


s_ashr_i32 s0, s22, 31 // (op 26976) 
s_cmp_eq_i32 s12, 0 // (op 26976) 
s_mov_b32 s47, scc // (op 26976) 
s_cmp_eq_i32 s47, 0 // (op 26976) 
s_cselect_b32 s23, 0, -1 // (op 26976) 

s_and_b32 s47, s44, 31 // (op 26976) 
s_lshl_b32 s24, 1, s47 // (op 26976) 
s_add_i32 s25, s24, s23 // (op 26976) 


s_and_b32 s23, s0, s25 // (op 26976) 


s_add_i32 s0, s22, s23 // (op 26976) 


s_ashr_i32 s22, s0, s47 // (op 26976) 


s_lshl_b32 s0, s44, 24 // (op 26976) 
s_ashr_i32 s47, s0, 31 // (op 26976) 

s_xor_b32 s0, s22, s47 // (op 26976) 

s_sub_i32 s22, s0, s47 // (op 26976) 


s_mul_i32 s0, s46, s11 // (op 26976) 
s_add_u32 s24, s20, 256 // (op 26976) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 26976) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 26976) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 26976) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 8 // (op 26976) 

s_sub_i32 s23, s24, s0 // (op 26976) 

s_mul_i32 s47, s22, s23 // (op 26976) 


s_sub_u32 s22, s18, s47 // (op 26976) 


s_add_u32 s18, s22, s0 // (op 26976) 


s_cmp_eq_i32 s19, 0 // (op 26976) 
s_mov_b32 s0, scc // (op 26976) 

s_cmp_eq_i32 s0, 0 // (op 26976) 
s_cselect_b32 s19, s18, s1 // (op 26976) 



v_lshl_add_u32 v11, s19, 8, v10 // (op 26976) 


v_mul_lo_u32 v17, v11, s27 // (op 26976) most significant: low of low * high
v_mul_hi_u32 v10, v11, s26 // (op 26976) most significant: high of low * low
v_mul_lo_u32 v16, v11, s26 // (op 26976) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 26976) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v12 // (op 26976) least significant half
v_addc_co_u32 v17, s[0:1], v17, v13, s[0:1] // (op 26976) most significant half



s_mul_i32 s1, s26, s21 // (op 26978) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 26978) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 26978) least significant
s_mul_hi_u32 s18, s26, s20 // (op 26978) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 26978) most significant: sum
s_add_u32 s1, s1, s18 // (op 26978) most significant: sum


s_mov_b32 s48, s28 // (op 26978) reg expression
s_mov_b32 s49, s29 // (op 26978) reg expression
s_mov_b32 s50, s0 // (op 26978) reg expression
s_mov_b32 s51, 131072 // (op 26978) call()

s_mul_hi_i32 s47, s16, s10 // (op 27021) 
v_lshrrev_b32 v10, 2, v1 // (op 27021) 
s_add_i32 s0, s47, s16 // (op 27021) 

v_and_b32 v11, -4, v1 // (op 27021) 
s_ashr_i32 s1, s0, 31 // (op 27021) 
v_sub_u32 v12, v1, v11 // (op 27021) 

s_cmp_eq_i32 s10, 0 // (op 27021) 
s_mov_b32 s47, scc // (op 27021) 
v_lshlrev_b32 v11, 5, v12 // (op 27021) 

s_cmp_eq_i32 s47, 0 // (op 27021) 
s_cselect_b32 s18, 0, -1 // (op 27021) 

v_mov_b32 v13, 0 // (op 27021) convert
v_mov_b32 v12, v11 // (op 27021) convert

s_and_b32 s19, s13, 31 // (op 27021) 
s_lshl_b32 s47, 1, s19 // (op 27021) 
s_add_i32 s22, s47, s18 // (op 27021) 


s_and_b32 s18, s1, s22 // (op 27021) 


s_add_i32 s1, s0, s18 // (op 27021) 


s_ashr_i32 s0, s1, s19 // (op 27021) 


s_lshl_b32 s1, s13, 24 // (op 27021) 
s_ashr_i32 s47, s1, 31 // (op 27021) 

s_xor_b32 s1, s0, s47 // (op 27021) 

s_sub_i32 s0, s1, s47 // (op 27021) 


s_mul_hi_i32 s1, s0, s14 // (op 27021) 
s_add_i32 s47, s1, s0 // (op 27021) 

s_ashr_i32 s1, s47, 31 // (op 27021) 
s_cmp_eq_i32 s14, 0 // (op 27021) 
s_mov_b32 s18, scc // (op 27021) 
s_cmp_eq_i32 s18, 0 // (op 27021) 
s_cselect_b32 s19, 0, -1 // (op 27021) 

s_and_b32 s18, s15, 31 // (op 27021) 
s_lshl_b32 s22, 1, s18 // (op 27021) 
s_add_i32 s23, s22, s19 // (op 27021) 


s_and_b32 s19, s1, s23 // (op 27021) 


s_add_i32 s1, s47, s19 // (op 27021) 


s_ashr_i32 s19, s1, s18 // (op 27021) 


s_lshl_b32 s1, s15, 24 // (op 27021) 
s_ashr_i32 s18, s1, 31 // (op 27021) 

s_xor_b32 s1, s19, s18 // (op 27021) 

s_sub_i32 s19, s1, s18 // (op 27021) 


s_add_u32 s22, s6, 256 // (op 27021) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27021) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27021) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27021) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27021) 

s_mul_i32 s1, s19, s22 // (op 27021) 

s_sub_i32 s47, s0, s1 // (op 27021) 


s_mul_hi_i32 s0, s16, s9 // (op 27021) 
s_add_i32 s1, s0, s16 // (op 27021) 

s_ashr_i32 s0, s1, 31 // (op 27021) 
s_cmp_eq_i32 s9, 0 // (op 27021) 
s_mov_b32 s18, scc // (op 27021) 
s_cmp_eq_i32 s18, 0 // (op 27021) 
s_cselect_b32 s19, 0, -1 // (op 27021) 

s_and_b32 s18, s45, 31 // (op 27021) 
s_lshl_b32 s24, 1, s18 // (op 27021) 
s_add_i32 s25, s24, s19 // (op 27021) 


s_and_b32 s19, s0, s25 // (op 27021) 


s_add_i32 s0, s1, s19 // (op 27021) 


s_ashr_i32 s1, s0, s18 // (op 27021) 


s_lshl_b32 s0, s45, 24 // (op 27021) 
s_ashr_i32 s18, s0, 31 // (op 27021) 

s_xor_b32 s0, s1, s18 // (op 27021) 

s_sub_i32 s1, s0, s18 // (op 27021) 


s_mul_i32 s0, s11, s22 // (op 27021) 

s_mul_i32 s18, s8, s0 // (op 27021) 

s_mul_i32 s0, s1, s18 // (op 27021) 

s_sub_u32 s18, s16, s0 // (op 27021) 

s_mul_hi_i32 s0, s18, s12 // (op 27021) 
s_add_i32 s19, s0, s18 // (op 27021) 


s_ashr_i32 s0, s19, 31 // (op 27021) 
s_cmp_eq_i32 s12, 0 // (op 27021) 
s_mov_b32 s18, scc // (op 27021) 
s_cmp_eq_i32 s18, 0 // (op 27021) 
s_cselect_b32 s22, 0, -1 // (op 27021) 

s_and_b32 s18, s44, 31 // (op 27021) 
s_lshl_b32 s23, 1, s18 // (op 27021) 
s_add_i32 s24, s23, s22 // (op 27021) 


s_and_b32 s25, s0, s24 // (op 27021) 


s_add_i32 s0, s19, s25 // (op 27021) 


s_ashr_i32 s19, s0, s18 // (op 27021) 


s_lshl_b32 s0, s44, 24 // (op 27021) 
s_ashr_i32 s18, s0, 31 // (op 27021) 

s_xor_b32 s0, s19, s18 // (op 27021) 

s_sub_i32 s19, s0, s18 // (op 27021) 


s_cmp_eq_i32 s1, 0 // (op 27021) 
s_mov_b32 s0, scc // (op 27021) 

s_cmp_eq_i32 s0, 0 // (op 27021) 
s_cselect_b32 s1, s19, s47 // (op 27021) 



v_lshl_add_u32 v11, s1, 8, v10 // (op 27021) 


v_mul_lo_u32 v19, v11, s31 // (op 27021) most significant: low of low * high
v_mul_hi_u32 v10, v11, s30 // (op 27021) most significant: high of low * low
v_mul_lo_u32 v18, v11, s30 // (op 27021) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 27021) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v18 // (op 27021) least significant half
v_addc_co_u32 v11, s[0:1], v13, v19, s[0:1] // (op 27021) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 27021) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 27021) 

s_mul_i32 s1, s30, s7 // (op 27023) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 27023) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 27023) least significant
s_mul_hi_u32 s18, s30, s6 // (op 27023) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27023) most significant: sum
s_add_u32 s1, s1, s18 // (op 27023) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 27023) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 27023) 

s_mov_b32 s60, s32 // (op 27023) reg expression
s_mov_b32 s61, s33 // (op 27023) reg expression
s_mov_b32 s62, s0 // (op 27023) reg expression
s_mov_b32 s63, 131072 // (op 27023) call()

s_mul_hi_i32 s47, s16, s10 // (op 27040) 
v_lshrrev_b32 v10, 2, v1 // (op 27040) 
s_add_i32 s0, s47, s16 // (op 27040) 

v_and_b32 v11, -4, v1 // (op 27040) 
s_ashr_i32 s1, s0, 31 // (op 27040) 
v_sub_u32 v18, v1, v11 // (op 27040) 

s_cmp_eq_i32 s10, 0 // (op 27040) 
s_mov_b32 s47, scc // (op 27040) 
v_lshlrev_b32 v11, 5, v18 // (op 27040) 

s_cmp_eq_i32 s47, 0 // (op 27040) 
s_cselect_b32 s18, 0, -1 // (op 27040) 

v_mov_b32 v19, 0 // (op 27040) convert
v_mov_b32 v18, v11 // (op 27040) convert

s_and_b32 s19, s13, 31 // (op 27040) 
s_lshl_b32 s47, 1, s19 // (op 27040) 
s_add_i32 s22, s47, s18 // (op 27040) 


s_and_b32 s18, s1, s22 // (op 27040) 


s_add_i32 s1, s0, s18 // (op 27040) 


s_ashr_i32 s0, s1, s19 // (op 27040) 


s_lshl_b32 s1, s13, 24 // (op 27040) 
s_ashr_i32 s47, s1, 31 // (op 27040) 

s_xor_b32 s1, s0, s47 // (op 27040) 

s_sub_i32 s0, s1, s47 // (op 27040) 


s_mul_hi_i32 s1, s0, s14 // (op 27040) 
s_add_i32 s47, s1, s0 // (op 27040) 

s_ashr_i32 s1, s47, 31 // (op 27040) 
s_cmp_eq_i32 s14, 0 // (op 27040) 
s_mov_b32 s18, scc // (op 27040) 
s_cmp_eq_i32 s18, 0 // (op 27040) 
s_cselect_b32 s19, 0, -1 // (op 27040) 

s_and_b32 s18, s15, 31 // (op 27040) 
s_lshl_b32 s22, 1, s18 // (op 27040) 
s_add_i32 s23, s22, s19 // (op 27040) 


s_and_b32 s19, s1, s23 // (op 27040) 


s_add_i32 s1, s47, s19 // (op 27040) 


s_ashr_i32 s19, s1, s18 // (op 27040) 


s_lshl_b32 s1, s15, 24 // (op 27040) 
s_ashr_i32 s18, s1, 31 // (op 27040) 

s_xor_b32 s1, s19, s18 // (op 27040) 

s_sub_i32 s19, s1, s18 // (op 27040) 


s_add_u32 s22, s6, 256 // (op 27040) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27040) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27040) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27040) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27040) 

s_mul_i32 s1, s19, s22 // (op 27040) 

s_sub_i32 s47, s0, s1 // (op 27040) 


s_mul_hi_i32 s0, s16, s9 // (op 27040) 
s_add_i32 s1, s0, s16 // (op 27040) 

s_ashr_i32 s0, s1, 31 // (op 27040) 
s_cmp_eq_i32 s9, 0 // (op 27040) 
s_mov_b32 s18, scc // (op 27040) 
s_cmp_eq_i32 s18, 0 // (op 27040) 
s_cselect_b32 s19, 0, -1 // (op 27040) 

s_and_b32 s18, s45, 31 // (op 27040) 
s_lshl_b32 s24, 1, s18 // (op 27040) 
s_add_i32 s25, s24, s19 // (op 27040) 


s_and_b32 s19, s0, s25 // (op 27040) 


s_add_i32 s0, s1, s19 // (op 27040) 


s_ashr_i32 s1, s0, s18 // (op 27040) 


s_lshl_b32 s0, s45, 24 // (op 27040) 
s_ashr_i32 s18, s0, 31 // (op 27040) 

s_xor_b32 s0, s1, s18 // (op 27040) 

s_sub_i32 s1, s0, s18 // (op 27040) 


s_mul_i32 s0, s11, s22 // (op 27040) 

s_mul_i32 s18, s8, s0 // (op 27040) 

s_mul_i32 s0, s1, s18 // (op 27040) 

s_sub_u32 s18, s16, s0 // (op 27040) 

s_mul_hi_i32 s0, s18, s12 // (op 27040) 
s_add_i32 s19, s0, s18 // (op 27040) 


s_ashr_i32 s0, s19, 31 // (op 27040) 
s_cmp_eq_i32 s12, 0 // (op 27040) 
s_mov_b32 s18, scc // (op 27040) 
s_cmp_eq_i32 s18, 0 // (op 27040) 
s_cselect_b32 s22, 0, -1 // (op 27040) 

s_and_b32 s18, s44, 31 // (op 27040) 
s_lshl_b32 s23, 1, s18 // (op 27040) 
s_add_i32 s24, s23, s22 // (op 27040) 


s_and_b32 s25, s0, s24 // (op 27040) 


s_add_i32 s0, s19, s25 // (op 27040) 


s_ashr_i32 s19, s0, s18 // (op 27040) 


s_lshl_b32 s0, s44, 24 // (op 27040) 
s_ashr_i32 s18, s0, 31 // (op 27040) 

s_xor_b32 s0, s19, s18 // (op 27040) 

s_sub_i32 s19, s0, s18 // (op 27040) 


s_cmp_eq_i32 s1, 0 // (op 27040) 
s_mov_b32 s0, scc // (op 27040) 

s_cmp_eq_i32 s0, 0 // (op 27040) 
s_cselect_b32 s1, s19, s47 // (op 27040) 



v_lshl_add_u32 v11, s1, 8, v10 // (op 27040) 


v_mul_lo_u32 v21, v11, s31 // (op 27040) most significant: low of low * high
v_mul_hi_u32 v10, v11, s30 // (op 27040) most significant: high of low * low
v_mul_lo_u32 v20, v11, s30 // (op 27040) least significant: low of low * low
v_add_u32 v21, v21, v10 // (op 27040) most significant: sum


v_add_co_u32 v10, s[0:1], v18, v20 // (op 27040) least significant half
v_addc_co_u32 v11, s[0:1], v19, v21, s[0:1] // (op 27040) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 27040) 


v_lshrrev_b64 v[18:19], 3, v[10:11] // (op 27040) 

s_mul_i32 s1, s30, s7 // (op 27042) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 27042) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 27042) least significant
s_mul_hi_u32 s18, s30, s6 // (op 27042) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27042) most significant: sum
s_add_u32 s1, s1, s18 // (op 27042) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 27042) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 27042) 

s_mov_b32 s60, s32 // (op 27042) reg expression
s_mov_b32 s61, s33 // (op 27042) reg expression
s_mov_b32 s62, s0 // (op 27042) reg expression
s_mov_b32 s63, 131072 // (op 27042) call()

s_mul_hi_i32 s47, s16, s10 // (op 27318) 
v_and_b32 v10, 63, v1 // (op 27318) 
s_add_i32 s0, s47, s16 // (op 27318) 

v_lshrrev_b32 v11, 4, v10 // (op 27318) 
s_ashr_i32 s1, s0, 31 // (op 27318) 
v_and_b32 v20, 3, v11 // (op 27318) 

s_cmp_eq_i32 s10, 0 // (op 27318) 
s_mov_b32 s47, scc // (op 27318) 
v_and_b32 v11, 15, v1 // (op 27318) 
s_cmp_eq_i32 s47, 0 // (op 27318) 
s_cselect_b32 s18, 0, -1 // (op 27318) 

v_lshl_add_u32 v21, v20, 4, v11 // (op 27318) 


s_and_b32 s19, s13, 31 // (op 27318) 
v_and_b32 v11, 4294967168, v1 // (op 27318) 
s_lshl_b32 s47, 1, s19 // (op 27318) 
v_add_u32 v20, v11, v21 // (op 27318) 


s_add_i32 s22, s47, s18 // (op 27318) 


v_lshrrev_b32 v11, 6, v10 // (op 27318) 

s_and_b32 s18, s1, s22 // (op 27318) 


v_lshlrev_b32 v10, 2, v11 // (op 27318) 

s_add_i32 s1, s0, s18 // (op 27318) 


s_ashr_i32 s0, s1, s19 // (op 27318) 


s_lshl_b32 s1, s13, 24 // (op 27318) 
s_ashr_i32 s47, s1, 31 // (op 27318) 

s_xor_b32 s1, s0, s47 // (op 27318) 

s_sub_i32 s0, s1, s47 // (op 27318) 


s_mul_hi_i32 s1, s0, s14 // (op 27318) 
s_add_i32 s47, s1, s0 // (op 27318) 

s_ashr_i32 s1, s47, 31 // (op 27318) 
s_cmp_eq_i32 s14, 0 // (op 27318) 
s_mov_b32 s18, scc // (op 27318) 
s_cmp_eq_i32 s18, 0 // (op 27318) 
s_cselect_b32 s19, 0, -1 // (op 27318) 

s_and_b32 s18, s15, 31 // (op 27318) 
s_lshl_b32 s22, 1, s18 // (op 27318) 
s_add_i32 s23, s22, s19 // (op 27318) 


s_and_b32 s19, s1, s23 // (op 27318) 


s_add_i32 s1, s47, s19 // (op 27318) 


s_ashr_i32 s19, s1, s18 // (op 27318) 


s_lshl_b32 s1, s15, 24 // (op 27318) 
s_ashr_i32 s18, s1, 31 // (op 27318) 

s_xor_b32 s1, s19, s18 // (op 27318) 

s_sub_i32 s19, s1, s18 // (op 27318) 


s_mul_i32 s1, s0, s11 // (op 27318) 

s_sub_u32 s0, s16, s1 // (op 27318) 

s_mul_i32 s1, s19, s11 // (op 27318) 
s_add_u32 s1, s1, s0 // (op 27318) 


s_mul_hi_i32 s0, s16, s9 // (op 27318) 
s_add_i32 s47, s0, s16 // (op 27318) 

s_ashr_i32 s0, s47, 31 // (op 27318) 
s_cmp_eq_i32 s9, 0 // (op 27318) 
s_mov_b32 s18, scc // (op 27318) 
s_cmp_eq_i32 s18, 0 // (op 27318) 
s_cselect_b32 s19, 0, -1 // (op 27318) 

s_and_b32 s18, s45, 31 // (op 27318) 
s_lshl_b32 s22, 1, s18 // (op 27318) 
s_add_i32 s23, s22, s19 // (op 27318) 


s_and_b32 s19, s0, s23 // (op 27318) 


s_add_i32 s0, s47, s19 // (op 27318) 


s_ashr_i32 s19, s0, s18 // (op 27318) 


s_lshl_b32 s0, s45, 24 // (op 27318) 
s_ashr_i32 s18, s0, 31 // (op 27318) 

s_xor_b32 s0, s19, s18 // (op 27318) 

s_sub_i32 s19, s0, s18 // (op 27318) 


s_add_u32 s22, s6, 256 // (op 27318) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27318) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27318) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27318) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27318) 

s_mul_i32 s0, s11, s22 // (op 27318) 

s_mul_i32 s18, s8, s0 // (op 27318) 

s_mul_i32 s0, s19, s18 // (op 27318) 

s_sub_u32 s18, s16, s0 // (op 27318) 

s_mov_b32 s0, s18 // (op 27318) 
s_mul_hi_i32 s47, s0, s12 // (op 27318) 
s_add_i32 s22, s47, s0 // (op 27318) 


s_ashr_i32 s0, s22, 31 // (op 27318) 
s_cmp_eq_i32 s12, 0 // (op 27318) 
s_mov_b32 s47, scc // (op 27318) 
s_cmp_eq_i32 s47, 0 // (op 27318) 
s_cselect_b32 s23, 0, -1 // (op 27318) 

s_and_b32 s47, s44, 31 // (op 27318) 
s_lshl_b32 s24, 1, s47 // (op 27318) 
s_add_i32 s25, s24, s23 // (op 27318) 


s_and_b32 s23, s0, s25 // (op 27318) 


s_add_i32 s0, s22, s23 // (op 27318) 


s_ashr_i32 s22, s0, s47 // (op 27318) 


s_lshl_b32 s0, s44, 24 // (op 27318) 
s_ashr_i32 s47, s0, 31 // (op 27318) 

s_xor_b32 s0, s22, s47 // (op 27318) 

s_sub_i32 s22, s0, s47 // (op 27318) 


s_mul_i32 s0, s46, s11 // (op 27318) 
s_add_u32 s24, s20, 256 // (op 27318) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 27318) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 27318) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 27318) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 27318) 

s_sub_i32 s23, s24, s0 // (op 27318) 

s_mul_i32 s47, s22, s23 // (op 27318) 


s_sub_u32 s22, s18, s47 // (op 27318) 


s_add_u32 s18, s22, s0 // (op 27318) 


s_cmp_eq_i32 s19, 0 // (op 27318) 
s_mov_b32 s0, scc // (op 27318) 

s_cmp_eq_i32 s0, 0 // (op 27318) 
s_cselect_b32 s19, s18, s1 // (op 27318) 



v_lshl_add_u32 v11, s19, 8, v20 // (op 27318) 


v_mul_lo_u32 v20, v11, s34 // (op 27318) 
v_add_u32 v20, v20, v10 // (op 27318) 


s_mul_i32 s1, s34, s21 // (op 27320) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 27320) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 27320) least significant
s_mul_hi_u32 s18, s34, s20 // (op 27320) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27320) most significant: sum
s_add_u32 s1, s1, s18 // (op 27320) most significant: sum


s_mov_b32 s64, s36 // (op 27320) reg expression
s_mov_b32 s65, s37 // (op 27320) reg expression
s_mov_b32 s66, s0 // (op 27320) reg expression
s_mov_b32 s67, 131072 // (op 27320) call()

s_mul_hi_i32 s47, s16, s10 // (op 27339) 
v_and_b32 v10, 63, v1 // (op 27339) 
s_add_i32 s0, s47, s16 // (op 27339) 

v_lshrrev_b32 v11, 4, v10 // (op 27339) 
s_ashr_i32 s1, s0, 31 // (op 27339) 
v_and_b32 v21, 3, v11 // (op 27339) 

s_cmp_eq_i32 s10, 0 // (op 27339) 
s_mov_b32 s47, scc // (op 27339) 
v_and_b32 v11, 15, v1 // (op 27339) 
s_cmp_eq_i32 s47, 0 // (op 27339) 
s_cselect_b32 s18, 0, -1 // (op 27339) 

v_lshl_add_u32 v22, v21, 4, v11 // (op 27339) 


s_and_b32 s19, s13, 31 // (op 27339) 
v_and_b32 v11, 4294967168, v1 // (op 27339) 
s_lshl_b32 s47, 1, s19 // (op 27339) 
v_add_u32 v21, v11, v22 // (op 27339) 


s_add_i32 s22, s47, s18 // (op 27339) 


v_lshrrev_b32 v11, 6, v10 // (op 27339) 

s_and_b32 s18, s1, s22 // (op 27339) 


v_lshlrev_b32 v10, 2, v11 // (op 27339) 

s_add_i32 s1, s0, s18 // (op 27339) 


s_ashr_i32 s0, s1, s19 // (op 27339) 


s_lshl_b32 s1, s13, 24 // (op 27339) 
s_ashr_i32 s47, s1, 31 // (op 27339) 

s_xor_b32 s1, s0, s47 // (op 27339) 

s_sub_i32 s0, s1, s47 // (op 27339) 


s_mul_hi_i32 s1, s0, s14 // (op 27339) 
s_add_i32 s47, s1, s0 // (op 27339) 

s_ashr_i32 s1, s47, 31 // (op 27339) 
s_cmp_eq_i32 s14, 0 // (op 27339) 
s_mov_b32 s18, scc // (op 27339) 
s_cmp_eq_i32 s18, 0 // (op 27339) 
s_cselect_b32 s19, 0, -1 // (op 27339) 

s_and_b32 s18, s15, 31 // (op 27339) 
s_lshl_b32 s22, 1, s18 // (op 27339) 
s_add_i32 s23, s22, s19 // (op 27339) 


s_and_b32 s19, s1, s23 // (op 27339) 


s_add_i32 s1, s47, s19 // (op 27339) 


s_ashr_i32 s19, s1, s18 // (op 27339) 


s_lshl_b32 s1, s15, 24 // (op 27339) 
s_ashr_i32 s18, s1, 31 // (op 27339) 

s_xor_b32 s1, s19, s18 // (op 27339) 

s_sub_i32 s19, s1, s18 // (op 27339) 


s_mul_i32 s1, s0, s11 // (op 27339) 

s_sub_u32 s0, s16, s1 // (op 27339) 

s_mul_i32 s1, s19, s11 // (op 27339) 
s_add_u32 s1, s1, s0 // (op 27339) 


s_mul_hi_i32 s0, s16, s9 // (op 27339) 
s_add_i32 s47, s0, s16 // (op 27339) 

s_ashr_i32 s0, s47, 31 // (op 27339) 
s_cmp_eq_i32 s9, 0 // (op 27339) 
s_mov_b32 s18, scc // (op 27339) 
s_cmp_eq_i32 s18, 0 // (op 27339) 
s_cselect_b32 s19, 0, -1 // (op 27339) 

s_and_b32 s18, s45, 31 // (op 27339) 
s_lshl_b32 s22, 1, s18 // (op 27339) 
s_add_i32 s23, s22, s19 // (op 27339) 


s_and_b32 s19, s0, s23 // (op 27339) 


s_add_i32 s0, s47, s19 // (op 27339) 


s_ashr_i32 s19, s0, s18 // (op 27339) 


s_lshl_b32 s0, s45, 24 // (op 27339) 
s_ashr_i32 s18, s0, 31 // (op 27339) 

s_xor_b32 s0, s19, s18 // (op 27339) 

s_sub_i32 s19, s0, s18 // (op 27339) 


s_add_u32 s22, s6, 256 // (op 27339) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27339) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27339) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27339) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27339) 

s_mul_i32 s0, s11, s22 // (op 27339) 

s_mul_i32 s18, s8, s0 // (op 27339) 

s_mul_i32 s0, s19, s18 // (op 27339) 

s_sub_u32 s18, s16, s0 // (op 27339) 

s_mov_b32 s0, s18 // (op 27339) 
s_mul_hi_i32 s47, s0, s12 // (op 27339) 
s_add_i32 s22, s47, s0 // (op 27339) 


s_ashr_i32 s0, s22, 31 // (op 27339) 
s_cmp_eq_i32 s12, 0 // (op 27339) 
s_mov_b32 s47, scc // (op 27339) 
s_cmp_eq_i32 s47, 0 // (op 27339) 
s_cselect_b32 s23, 0, -1 // (op 27339) 

s_and_b32 s47, s44, 31 // (op 27339) 
s_lshl_b32 s24, 1, s47 // (op 27339) 
s_add_i32 s25, s24, s23 // (op 27339) 


s_and_b32 s23, s0, s25 // (op 27339) 


s_add_i32 s0, s22, s23 // (op 27339) 


s_ashr_i32 s22, s0, s47 // (op 27339) 


s_lshl_b32 s0, s44, 24 // (op 27339) 
s_ashr_i32 s47, s0, 31 // (op 27339) 

s_xor_b32 s0, s22, s47 // (op 27339) 

s_sub_i32 s22, s0, s47 // (op 27339) 


s_mul_i32 s0, s46, s11 // (op 27339) 
s_add_u32 s24, s20, 256 // (op 27339) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 27339) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 27339) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 27339) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 8 // (op 27339) 

s_sub_i32 s23, s24, s0 // (op 27339) 

s_mul_i32 s47, s22, s23 // (op 27339) 


s_sub_u32 s22, s18, s47 // (op 27339) 


s_add_u32 s18, s22, s0 // (op 27339) 


s_cmp_eq_i32 s19, 0 // (op 27339) 
s_mov_b32 s0, scc // (op 27339) 

s_cmp_eq_i32 s0, 0 // (op 27339) 
s_cselect_b32 s19, s18, s1 // (op 27339) 



v_lshl_add_u32 v11, s19, 8, v21 // (op 27339) 


v_mul_lo_u32 v21, v11, s34 // (op 27339) 
v_add_u32 v21, v21, v10 // (op 27339) 


s_mul_i32 s1, s34, s21 // (op 27341) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 27341) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 27341) least significant
s_mul_hi_u32 s18, s34, s20 // (op 27341) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27341) most significant: sum
s_add_u32 s1, s1, s18 // (op 27341) most significant: sum


s_mov_b32 s64, s36 // (op 27341) reg expression
s_mov_b32 s65, s37 // (op 27341) reg expression
s_mov_b32 s66, s0 // (op 27341) reg expression
s_mov_b32 s67, 131072 // (op 27341) call()

s_mul_hi_i32 s47, s16, s10 // (op 27370) 
v_and_b32 v10, 63, v1 // (op 27370) 
s_add_i32 s0, s47, s16 // (op 27370) 

v_lshrrev_b32 v11, 4, v10 // (op 27370) 
s_ashr_i32 s1, s0, 31 // (op 27370) 
v_and_b32 v22, 3, v11 // (op 27370) 

s_cmp_eq_i32 s10, 0 // (op 27370) 
s_mov_b32 s47, scc // (op 27370) 
v_and_b32 v11, 15, v1 // (op 27370) 
s_cmp_eq_i32 s47, 0 // (op 27370) 
s_cselect_b32 s18, 0, -1 // (op 27370) 

v_lshl_add_u32 v23, v22, 4, v11 // (op 27370) 


s_and_b32 s19, s13, 31 // (op 27370) 
v_lshrrev_b32 v11, 6, v1 // (op 27370) 
s_lshl_b32 s47, 1, s19 // (op 27370) 
v_and_b32 v22, 1, v11 // (op 27370) 

s_add_i32 s22, s47, s18 // (op 27370) 


v_lshl_add_u32 v11, v22, 7, v23 // (op 27370) 


s_and_b32 s18, s1, s22 // (op 27370) 


v_lshrrev_b32 v22, 6, v10 // (op 27370) 

s_add_i32 s1, s0, s18 // (op 27370) 


s_ashr_i32 s0, s1, s19 // (op 27370) 


s_lshl_b32 s1, s13, 24 // (op 27370) 
s_ashr_i32 s47, s1, 31 // (op 27370) 

s_xor_b32 s1, s0, s47 // (op 27370) 

s_sub_i32 s0, s1, s47 // (op 27370) 


s_mul_hi_i32 s1, s0, s14 // (op 27370) 
s_add_i32 s47, s1, s0 // (op 27370) 

s_ashr_i32 s1, s47, 31 // (op 27370) 
s_cmp_eq_i32 s14, 0 // (op 27370) 
s_mov_b32 s18, scc // (op 27370) 
s_cmp_eq_i32 s18, 0 // (op 27370) 
s_cselect_b32 s19, 0, -1 // (op 27370) 

s_and_b32 s18, s15, 31 // (op 27370) 
s_lshl_b32 s22, 1, s18 // (op 27370) 
s_add_i32 s23, s22, s19 // (op 27370) 


s_and_b32 s19, s1, s23 // (op 27370) 


s_add_i32 s1, s47, s19 // (op 27370) 


s_ashr_i32 s19, s1, s18 // (op 27370) 


s_lshl_b32 s1, s15, 24 // (op 27370) 
s_ashr_i32 s18, s1, 31 // (op 27370) 

s_xor_b32 s1, s19, s18 // (op 27370) 

s_sub_i32 s19, s1, s18 // (op 27370) 


s_add_u32 s22, s6, 256 // (op 27370) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27370) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27370) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27370) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27370) 

s_mul_i32 s1, s19, s22 // (op 27370) 

s_sub_i32 s47, s0, s1 // (op 27370) 


s_mul_hi_i32 s0, s16, s9 // (op 27370) 
s_add_i32 s1, s0, s16 // (op 27370) 

s_ashr_i32 s0, s1, 31 // (op 27370) 
s_cmp_eq_i32 s9, 0 // (op 27370) 
s_mov_b32 s18, scc // (op 27370) 
s_cmp_eq_i32 s18, 0 // (op 27370) 
s_cselect_b32 s19, 0, -1 // (op 27370) 

s_and_b32 s18, s45, 31 // (op 27370) 
s_lshl_b32 s24, 1, s18 // (op 27370) 
s_add_i32 s25, s24, s19 // (op 27370) 


s_and_b32 s19, s0, s25 // (op 27370) 


s_add_i32 s0, s1, s19 // (op 27370) 


s_ashr_i32 s1, s0, s18 // (op 27370) 


s_lshl_b32 s0, s45, 24 // (op 27370) 
s_ashr_i32 s18, s0, 31 // (op 27370) 

s_xor_b32 s0, s1, s18 // (op 27370) 

s_sub_i32 s1, s0, s18 // (op 27370) 


s_mul_i32 s0, s11, s22 // (op 27370) 

s_mul_i32 s18, s8, s0 // (op 27370) 

s_mul_i32 s0, s1, s18 // (op 27370) 

s_sub_u32 s18, s16, s0 // (op 27370) 

s_mul_hi_i32 s0, s18, s12 // (op 27370) 
s_add_i32 s19, s0, s18 // (op 27370) 


s_ashr_i32 s0, s19, 31 // (op 27370) 
s_cmp_eq_i32 s12, 0 // (op 27370) 
s_mov_b32 s18, scc // (op 27370) 
s_cmp_eq_i32 s18, 0 // (op 27370) 
s_cselect_b32 s22, 0, -1 // (op 27370) 

s_and_b32 s18, s44, 31 // (op 27370) 
s_lshl_b32 s23, 1, s18 // (op 27370) 
s_add_i32 s24, s23, s22 // (op 27370) 


s_and_b32 s25, s0, s24 // (op 27370) 


s_add_i32 s0, s19, s25 // (op 27370) 


s_ashr_i32 s19, s0, s18 // (op 27370) 


s_lshl_b32 s0, s44, 24 // (op 27370) 
s_ashr_i32 s18, s0, 31 // (op 27370) 

s_xor_b32 s0, s19, s18 // (op 27370) 

s_sub_i32 s19, s0, s18 // (op 27370) 


s_cmp_eq_i32 s1, 0 // (op 27370) 
s_mov_b32 s0, scc // (op 27370) 

s_cmp_eq_i32 s0, 0 // (op 27370) 
s_cselect_b32 s1, s19, s47 // (op 27370) 



v_lshl_add_u32 v10, s1, 8, v11 // (op 27370) 


v_mul_lo_u32 v11, v10, s38 // (op 27370) 

v_lshl_add_u32 v10, v22, 2, v11 // (op 27370) 


s_mul_i32 s1, s38, s7 // (op 27372) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 27372) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 27372) least significant
s_mul_hi_u32 s18, s38, s6 // (op 27372) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27372) most significant: sum
s_add_u32 s1, s1, s18 // (op 27372) most significant: sum


s_mov_b32 s68, s40 // (op 27372) reg expression
s_mov_b32 s69, s41 // (op 27372) reg expression
s_mov_b32 s70, s0 // (op 27372) reg expression
s_mov_b32 s71, 131072 // (op 27372) call()

s_mul_hi_i32 s47, s16, s10 // (op 27391) 
v_and_b32 v11, 63, v1 // (op 27391) 
s_add_i32 s0, s47, s16 // (op 27391) 

v_lshrrev_b32 v22, 4, v11 // (op 27391) 
s_ashr_i32 s1, s0, 31 // (op 27391) 
v_and_b32 v23, 3, v22 // (op 27391) 

s_cmp_eq_i32 s10, 0 // (op 27391) 
s_mov_b32 s47, scc // (op 27391) 
v_and_b32 v22, 15, v1 // (op 27391) 
s_cmp_eq_i32 s47, 0 // (op 27391) 
s_cselect_b32 s18, 0, -1 // (op 27391) 

v_lshl_add_u32 v24, v23, 4, v22 // (op 27391) 


s_and_b32 s19, s13, 31 // (op 27391) 
v_lshrrev_b32 v22, 6, v1 // (op 27391) 
s_lshl_b32 s47, 1, s19 // (op 27391) 
v_and_b32 v23, 1, v22 // (op 27391) 

s_add_i32 s22, s47, s18 // (op 27391) 


v_lshl_add_u32 v22, v23, 7, v24 // (op 27391) 


s_and_b32 s18, s1, s22 // (op 27391) 


v_lshrrev_b32 v23, 6, v11 // (op 27391) 

s_add_i32 s1, s0, s18 // (op 27391) 


s_ashr_i32 s0, s1, s19 // (op 27391) 


s_lshl_b32 s1, s13, 24 // (op 27391) 
s_ashr_i32 s47, s1, 31 // (op 27391) 

s_xor_b32 s1, s0, s47 // (op 27391) 

s_sub_i32 s0, s1, s47 // (op 27391) 


s_mul_hi_i32 s1, s0, s14 // (op 27391) 
s_add_i32 s47, s1, s0 // (op 27391) 

s_ashr_i32 s1, s47, 31 // (op 27391) 
s_cmp_eq_i32 s14, 0 // (op 27391) 
s_mov_b32 s18, scc // (op 27391) 
s_cmp_eq_i32 s18, 0 // (op 27391) 
s_cselect_b32 s19, 0, -1 // (op 27391) 

s_and_b32 s18, s15, 31 // (op 27391) 
s_lshl_b32 s22, 1, s18 // (op 27391) 
s_add_i32 s23, s22, s19 // (op 27391) 


s_and_b32 s19, s1, s23 // (op 27391) 


s_add_i32 s1, s47, s19 // (op 27391) 


s_ashr_i32 s19, s1, s18 // (op 27391) 


s_lshl_b32 s1, s15, 24 // (op 27391) 
s_ashr_i32 s18, s1, 31 // (op 27391) 

s_xor_b32 s1, s19, s18 // (op 27391) 

s_sub_i32 s19, s1, s18 // (op 27391) 


s_add_u32 s22, s6, 256 // (op 27391) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27391) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27391) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27391) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27391) 

s_mul_i32 s1, s19, s22 // (op 27391) 

s_sub_i32 s47, s0, s1 // (op 27391) 


s_mul_hi_i32 s0, s16, s9 // (op 27391) 
s_add_i32 s1, s0, s16 // (op 27391) 

s_ashr_i32 s0, s1, 31 // (op 27391) 
s_cmp_eq_i32 s9, 0 // (op 27391) 
s_mov_b32 s18, scc // (op 27391) 
s_cmp_eq_i32 s18, 0 // (op 27391) 
s_cselect_b32 s19, 0, -1 // (op 27391) 

s_and_b32 s18, s45, 31 // (op 27391) 
s_lshl_b32 s24, 1, s18 // (op 27391) 
s_add_i32 s25, s24, s19 // (op 27391) 


s_and_b32 s19, s0, s25 // (op 27391) 


s_add_i32 s0, s1, s19 // (op 27391) 


s_ashr_i32 s1, s0, s18 // (op 27391) 


s_lshl_b32 s0, s45, 24 // (op 27391) 
s_ashr_i32 s18, s0, 31 // (op 27391) 

s_xor_b32 s0, s1, s18 // (op 27391) 

s_sub_i32 s1, s0, s18 // (op 27391) 


s_mul_i32 s0, s11, s22 // (op 27391) 

s_mul_i32 s18, s8, s0 // (op 27391) 

s_mul_i32 s0, s1, s18 // (op 27391) 

s_sub_u32 s18, s16, s0 // (op 27391) 

s_mul_hi_i32 s0, s18, s12 // (op 27391) 
s_add_i32 s19, s0, s18 // (op 27391) 


s_ashr_i32 s0, s19, 31 // (op 27391) 
s_cmp_eq_i32 s12, 0 // (op 27391) 
s_mov_b32 s18, scc // (op 27391) 
s_cmp_eq_i32 s18, 0 // (op 27391) 
s_cselect_b32 s22, 0, -1 // (op 27391) 

s_and_b32 s18, s44, 31 // (op 27391) 
s_lshl_b32 s23, 1, s18 // (op 27391) 
s_add_i32 s24, s23, s22 // (op 27391) 


s_and_b32 s25, s0, s24 // (op 27391) 


s_add_i32 s0, s19, s25 // (op 27391) 


s_ashr_i32 s19, s0, s18 // (op 27391) 


s_lshl_b32 s0, s44, 24 // (op 27391) 
s_ashr_i32 s18, s0, 31 // (op 27391) 

s_xor_b32 s0, s19, s18 // (op 27391) 

s_sub_i32 s19, s0, s18 // (op 27391) 


s_cmp_eq_i32 s1, 0 // (op 27391) 
s_mov_b32 s0, scc // (op 27391) 

s_cmp_eq_i32 s0, 0 // (op 27391) 
s_cselect_b32 s1, s19, s47 // (op 27391) 



v_lshl_add_u32 v11, s1, 8, v22 // (op 27391) 


v_mul_lo_u32 v22, v11, s38 // (op 27391) 

v_lshl_add_u32 v11, v23, 2, v22 // (op 27391) 


s_mul_i32 s1, s38, s7 // (op 27393) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 27393) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 27393) least significant
s_mul_hi_u32 s18, s38, s6 // (op 27393) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27393) most significant: sum
s_add_u32 s1, s1, s18 // (op 27393) most significant: sum


s_mov_b32 s68, s40 // (op 27393) reg expression
s_mov_b32 s69, s41 // (op 27393) reg expression
s_mov_b32 s70, s0 // (op 27393) reg expression
s_mov_b32 s71, 131072 // (op 27393) call()

v_and_b32 v22, -4, v1 // (op 27168) 
v_sub_u32 v23, v1, v22 // (op 27168) 

v_lshlrev_b32 v22, 5, v23 // (op 27168) 

v_lshrrev_b32 v23, 2, v1 // (op 27168) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 27168) 


v_lshlrev_b32 v22, 2, v24 // (op 27168) 

v_lshrrev_b32 v23, 3, v22 // (op 27168) 

s_nop 0
v_readfirstlane_b32 s47, v23 // (op 27168) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 26843) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 26843) 

v_lshrrev_b64 v[22:23], 3, s[18:19] // (op 26843) 


v_mov_b64 v[24:25], v[12:13] // (op 26843) 
s_mov_b32 m0, s47 // (op 26843) reg expression
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 26843) Load value direct to lds
v_add_co_u32 v24, s[0:1], v24, v22 // (op 26843) least significant half
v_addc_co_u32 v25, s[0:1], 0, v25, s[0:1] // (op 26843) most significant half

s_add_u32 m0, m0, 4096 // (op 26843) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 26843) Load value direct to lds
v_add_co_u32 v24, s[0:1], v24, v22 // (op 26843) least significant half
v_addc_co_u32 v25, s[0:1], 0, v25, s[0:1] // (op 26843) most significant half

s_add_u32 m0, m0, 4096 // (op 26843) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 26843) Load value direct to lds
v_add_co_u32 v24, s[0:1], v24, v22 // (op 26843) least significant half
v_addc_co_u32 v25, s[0:1], 0, v25, s[0:1] // (op 26843) most significant half

s_add_u32 m0, m0, 4096 // (op 26843) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 26843) Load value direct to lds



v_and_b32 v22, -8, v1 // (op 27111) 
v_sub_u32 v23, v1, v22 // (op 27111) 

v_lshlrev_b32 v22, 4, v23 // (op 27111) 

v_lshrrev_b32 v23, 3, v1 // (op 27111) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 27111) 


s_nop 0
v_readfirstlane_b32 s47, v24 // (op 27111) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 26848) 
v_mov_b64 v[22:23], s[0:1] // (op 26848) convert


v_mov_b64 v[24:25], v[14:15] // (op 26848) 
s_add_u32 s0, 16384, s47 // (op 26848) 
s_mov_b32 m0, s0 // (op 26848) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26848) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 26848) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 26848) most significant half

s_add_u32 m0, m0, 4096 // (op 26848) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26848) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 26848) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 26848) most significant half

s_add_u32 m0, m0, 4096 // (op 26848) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26848) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 26848) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 26848) most significant half

s_add_u32 m0, m0, 4096 // (op 26848) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26848) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 26848) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 26848) most significant half

s_add_u32 m0, m0, 4096 // (op 26848) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26848) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 26848) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 26848) most significant half

s_add_u32 m0, m0, 4096 // (op 26848) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26848) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 26848) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 26848) most significant half

s_add_u32 m0, m0, 4096 // (op 26848) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26848) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 26848) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 26848) most significant half

s_add_u32 m0, m0, 4096 // (op 26848) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26848) Load value direct to lds






v_mov_b32 v22, v10 // (op 76) reg expression

buffer_load_dwordx2 v[24:25], v22, s[68:71], 0 offen // (op 76) Load value



s_lshl_b32 s47, s38, 6 // (op 726) 
v_add_u32 v22, v10, s47 // (op 726) 


buffer_load_dwordx2 v[26:27], v22, s[68:71], 0 offen // (op 726) Load value




s_lshl_b32 s47, s34, 6 // (op 5311) 
v_add_u32 v10, v20, s47 // (op 5311) 


buffer_load_dwordx2 v[22:23], v10, s[64:67], 0 offen // (op 5311) Load value



v_mov_b32 v10, v20 // (op 72) reg expression

buffer_load_dwordx2 v[28:29], v10, s[64:67], 0 offen // (op 72) Load value


v_and_b32 v10, -4, v1 // (op 27279) 
v_sub_u32 v20, v1, v10 // (op 27279) 

v_lshlrev_b32 v10, 5, v20 // (op 27279) 

v_lshrrev_b32 v20, 2, v1 // (op 27279) 
v_lshl_add_u32 v30, v20, 7, v10 // (op 27279) 


v_lshlrev_b32 v10, 2, v30 // (op 27279) 

v_lshrrev_b32 v20, 3, v10 // (op 27279) 

s_nop 0
v_readfirstlane_b32 s47, v20 // (op 27279) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 26853) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 26853) 

v_lshrrev_b64 v[30:31], 3, s[18:19] // (op 26853) 


v_mov_b64 v[32:33], v[12:13] // (op 26853) 
s_add_u32 s0, 49152, s47 // (op 26853) 
s_mov_b32 m0, s0 // (op 26853) reg expression
buffer_load_dwordx4 v32, s[60:63], 64 offen offset: 0 lds // (op 26853) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 26853) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 26853) most significant half

s_add_u32 m0, m0, 4096 // (op 26853) 
buffer_load_dwordx4 v32, s[60:63], 64 offen offset: 0 lds // (op 26853) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 26853) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 26853) most significant half

s_add_u32 m0, m0, 4096 // (op 26853) 
buffer_load_dwordx4 v32, s[60:63], 64 offen offset: 0 lds // (op 26853) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 26853) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 26853) most significant half

s_add_u32 m0, m0, 4096 // (op 26853) 
buffer_load_dwordx4 v32, s[60:63], 64 offen offset: 0 lds // (op 26853) Load value direct to lds





v_and_b32 v10, -8, v1 // (op 27235) 
v_sub_u32 v20, v1, v10 // (op 27235) 

v_lshlrev_b32 v10, 4, v20 // (op 27235) 

v_lshrrev_b32 v20, 3, v1 // (op 27235) 
v_lshl_add_u32 v12, v20, 7, v10 // (op 27235) 


s_nop 0
v_readfirstlane_b32 s47, v12 // (op 27235) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 26855) 
v_mov_b64 v[12:13], s[0:1] // (op 26855) convert


v_mov_b64 v[30:31], v[14:15] // (op 26855) 
s_add_u32 s0, 65536, s47 // (op 26855) 
s_mov_b32 m0, s0 // (op 26855) reg expression
s_mov_b32 s1, 128 // (op 26855) call()
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 26855) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v12 // (op 26855) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 26855) most significant half

s_add_u32 m0, m0, 4096 // (op 26855) 
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 26855) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v12 // (op 26855) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 26855) most significant half

s_add_u32 m0, m0, 4096 // (op 26855) 
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 26855) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v12 // (op 26855) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 26855) most significant half

s_add_u32 m0, m0, 4096 // (op 26855) 
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 26855) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v12 // (op 26855) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 26855) most significant half

s_add_u32 m0, m0, 4096 // (op 26855) 
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 26855) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v12 // (op 26855) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 26855) most significant half

s_add_u32 m0, m0, 4096 // (op 26855) 
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 26855) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v12 // (op 26855) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 26855) most significant half

s_add_u32 m0, m0, 4096 // (op 26855) 
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 26855) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v12 // (op 26855) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 26855) most significant half

s_add_u32 m0, m0, 4096 // (op 26855) 
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 26855) Load value direct to lds






s_waitcnt vmcnt(24)
s_barrier  // (op 27808) 



ds_read_b128 v[12:15], v4 // (op 26194) Load local data 



ds_read_b128 v[30:33], v4 offset:1024 // (op 26199) Load local data 



ds_read_b128 v[34:37], v4 offset:2048 // (op 26204) Load local data 



ds_read_b128 v[38:41], v4 offset:3072 // (op 26209) Load local data 



ds_read_b128 v[42:45], v4 offset:4096 // (op 26214) Load local data 



ds_read_b128 v[46:49], v4 offset:5120 // (op 26219) Load local data 



ds_read_b128 v[50:53], v4 offset:6144 // (op 26224) Load local data 



ds_read_b128 v[54:57], v4 offset:7168 // (op 26229) Load local data 
s_waitcnt vmcnt(16)
s_barrier  // (op 27792) 



ds_read_b128 v[58:61], v3 offset:16384 // (op 26154) Load local data 
ds_read_b128 v[62:65], v3 offset:16448 // (op 26154) Load local data 



ds_read_b128 v[66:69], v3 offset:18432 // (op 26159) Load local data 
ds_read_b128 v[70:73], v3 offset:18496 // (op 26159) Load local data 



ds_read_b128 v[74:77], v3 offset:20480 // (op 26164) Load local data 
ds_read_b128 v[78:81], v3 offset:20544 // (op 26164) Load local data 



ds_read_b128 v[82:85], v3 offset:22528 // (op 26169) Load local data 
ds_read_b128 v[86:89], v3 offset:22592 // (op 26169) Load local data 



ds_read_b128 v[90:93], v3 offset:24576 // (op 26174) Load local data 
ds_read_b128 v[94:97], v3 offset:24640 // (op 26174) Load local data 



ds_read_b128 v[98:101], v3 offset:26624 // (op 26179) Load local data 
ds_read_b128 v[102:105], v3 offset:26688 // (op 26179) Load local data 



ds_read_b128 v[106:109], v3 offset:28672 // (op 26184) Load local data 
ds_read_b128 v[110:113], v3 offset:28736 // (op 26184) Load local data 



ds_read_b128 v[114:117], v3 offset:30720 // (op 26189) Load local data 
ds_read_b128 v[118:121], v3 offset:30784 // (op 26189) Load local data 
v_and_b32 v10, -8, v1 // (op 27096) 
v_sub_u32 v20, v1, v10 // (op 27096) 

v_lshlrev_b32 v10, 4, v20 // (op 27096) 

v_lshrrev_b32 v20, 3, v1 // (op 27096) 
v_lshl_add_u32 v122, v20, 7, v10 // (op 27096) 


s_nop 0
v_readfirstlane_b32 s47, v122 // (op 27096) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -4, v1 // (op 27153) 
v_sub_u32 v20, v1, v10 // (op 27153) 

v_lshlrev_b32 v10, 5, v20 // (op 27153) 

v_lshrrev_b32 v20, 2, v1 // (op 27153) 
v_lshl_add_u32 v122, v20, 7, v10 // (op 27153) 


v_lshlrev_b32 v10, 2, v122 // (op 27153) 

v_lshrrev_b32 v20, 3, v10 // (op 27153) 

s_nop 0
v_readfirstlane_b32 s0, v20 // (op 27153) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -8, v1 // (op 27220) 
v_sub_u32 v20, v1, v10 // (op 27220) 

v_lshlrev_b32 v10, 4, v20 // (op 27220) 

v_lshrrev_b32 v20, 3, v1 // (op 27220) 
v_lshl_add_u32 v122, v20, 7, v10 // (op 27220) 


s_nop 0
v_readfirstlane_b32 s1, v122 // (op 27220) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -4, v1 // (op 27264) 
v_sub_u32 v20, v1, v10 // (op 27264) 

v_lshlrev_b32 v10, 5, v20 // (op 27264) 

v_lshrrev_b32 v20, 2, v1 // (op 27264) 
v_lshl_add_u32 v122, v20, 7, v10 // (op 27264) 


v_lshlrev_b32 v10, 2, v122 // (op 27264) 

v_lshrrev_b32 v20, 3, v10 // (op 27264) 

s_nop 0
v_readfirstlane_b32 s18, v20 // (op 27264) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s19, 0 // (op 8908) call()
s_ashr_i64 s[22:23], s[42:43], 7 // (op 8907) 
s_mov_b32 s24, s22 // (op 8907) convert

s_and_b32 s25, s24, 4294967294 // (op 8907) 

s_cmp_lt_u32 s19, s25 // (op 8907) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoop_8907 // (op 8907) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoop_8907 if false)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoop_8907:
s_add_u32 s19, s19, 2 // (op 8909) 
s_waitcnt vmcnt(15)
v_mov_b32 v122, v24 // (op 26729) reg expression
v_mov_b32 v123, v25 // (op 26729) reg expression
s_barrier  // (op 27811) 
s_waitcnt vmcnt(14)
v_mov_b32 v124, v26 // (op 26742) reg expression

v_mov_b32 v125, v27 // (op 26742) reg expression
s_lshl_b64 s[22:23], s[30:31], 6 // (op 26816) 
s_lshl_b64 s[24:25], s[22:23], 2 // (op 26816) 

v_lshrrev_b64 v[126:127], 3, s[24:25] // (op 26816) 


v_mov_b64 v[128:129], v[18:19] // (op 26816) 
v_mov_b64 v[130:131], v[128:129] // (op 26816) 
s_mov_b32 m0, s0 // (op 26816) reg expression
s_mov_b32 s22, 128 // (op 26816) call()
buffer_load_dwordx4 v130, s[60:63], s22 offen offset: 0 lds // (op 26816) Load value direct to lds
v_add_co_u32 v130, s[24:25], v130, v126 // (op 26816) least significant half
v_addc_co_u32 v131, s[24:25], 0, v131, s[24:25] // (op 26816) most significant half

s_add_u32 m0, m0, 4096 // (op 26816) 
buffer_load_dwordx4 v130, s[60:63], s22 offen offset: 0 lds // (op 26816) Load value direct to lds
v_add_co_u32 v130, s[24:25], v130, v126 // (op 26816) least significant half
v_addc_co_u32 v131, s[24:25], 0, v131, s[24:25] // (op 26816) most significant half

s_add_u32 m0, m0, 4096 // (op 26816) 
buffer_load_dwordx4 v130, s[60:63], s22 offen offset: 0 lds // (op 26816) Load value direct to lds
v_add_co_u32 v130, s[24:25], v130, v126 // (op 26816) least significant half
v_addc_co_u32 v131, s[24:25], 0, v131, s[24:25] // (op 26816) most significant half

s_add_u32 m0, m0, 4096 // (op 26816) 
s_waitcnt vmcnt(16)
v_mov_b32 v132, v22 // (op 26755) reg expression
v_mov_b32 v133, v23 // (op 26755) reg expression
s_waitcnt vmcnt(15)
v_mov_b32 v134, v28 // (op 26768) reg expression
v_mov_b32 v135, v29 // (op 26768) reg expression
buffer_load_dwordx4 v130, s[60:63], s22 offen offset: 0 lds // (op 26816) Load value direct to lds



v_bfe_u32 v136, v134, 0, 8 // (op 24807) 
v_bfe_u32 v137, v134, 8, 8 // (op 24807) 
v_bfe_u32 v138, v134, 16, 8 // (op 24807) 
v_bfe_u32 v139, v134, 24, 8 // (op 24807) 
v_permlane16_swap_b32 v136, v137 // (op 24807) 
s_nop 0
v_permlane16_swap_b32 v138, v139 // (op 24807) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v136, v138 // (op 24807) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v137, v139 // (op 24807) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 27795) 

s_lshl_b64 s[22:23], s[26:27], 5 // (op 26811) 
v_mov_b64 v[126:127], s[22:23] // (op 26811) convert


v_mov_b64 v[130:131], v[16:17] // (op 26811) 
v_mov_b64 v[140:141], v[130:131] // (op 26811) 
s_add_u32 s22, 16384, s47 // (op 26811) 
s_mov_b32 m0, s22 // (op 26811) reg expression
s_mov_b32 s23, 256 // (op 26811) call()
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 26811) Load value direct to lds
v_add_co_u32 v140, s[24:25], v140, v126 // (op 26811) least significant half
v_addc_co_u32 v141, s[24:25], 0, v141, s[24:25] // (op 26811) most significant half

s_add_u32 m0, m0, 4096 // (op 26811) 
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 26811) Load value direct to lds
v_add_co_u32 v140, s[24:25], v140, v126 // (op 26811) least significant half
v_addc_co_u32 v141, s[24:25], 0, v141, s[24:25] // (op 26811) most significant half

s_add_u32 m0, m0, 4096 // (op 26811) 
v_bfe_u32 v142, v122, 0, 8 // (op 24998) 
v_bfe_u32 v143, v122, 8, 8 // (op 24998) 
v_bfe_u32 v144, v122, 16, 8 // (op 24998) 
v_bfe_u32 v145, v122, 24, 8 // (op 24998) 
v_permlane16_swap_b32 v142, v143 // (op 24998) 
s_nop 0
v_permlane16_swap_b32 v144, v145 // (op 24998) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v142, v144 // (op 24998) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v143, v145 // (op 24998) 
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 26811) Load value direct to lds
v_add_co_u32 v140, s[24:25], v140, v126 // (op 26811) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:65], v[12:15], a[0:3], v136, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 78) 
v_addc_co_u32 v141, s[24:25], 0, v141, s[24:25] // (op 26811) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:65], v[30:33], a[4:7], v136, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 315) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[58:65], v[34:37], a[8:11], v136, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 451) 
s_add_u32 m0, m0, 4096 // (op 26811) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:65], v[38:41], a[12:15], v136, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 587) 
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 26811) Load value direct to lds
v_add_co_u32 v140, s[24:25], v140, v126 // (op 26811) least significant half
v_addc_co_u32 v141, s[24:25], 0, v141, s[24:25] // (op 26811) most significant half

s_add_u32 m0, m0, 4096 // (op 26811) 
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 26811) Load value direct to lds
v_add_co_u32 v140, s[24:25], v140, v126 // (op 26811) least significant half
v_addc_co_u32 v141, s[24:25], 0, v141, s[24:25] // (op 26811) most significant half

s_add_u32 m0, m0, 4096 // (op 26811) 
v_bfe_u32 v146, v124, 0, 8 // (op 25030) 
v_bfe_u32 v147, v124, 8, 8 // (op 25030) 
v_bfe_u32 v148, v124, 16, 8 // (op 25030) 
v_bfe_u32 v149, v124, 24, 8 // (op 25030) 
v_permlane16_swap_b32 v146, v147 // (op 25030) 
s_nop 0
v_permlane16_swap_b32 v148, v149 // (op 25030) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v146, v148 // (op 25030) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v147, v149 // (op 25030) 
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 26811) Load value direct to lds
v_add_co_u32 v140, s[24:25], v140, v126 // (op 26811) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[58:65], v[42:45], a[16:19], v136, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 723) 
v_addc_co_u32 v141, s[24:25], 0, v141, s[24:25] // (op 26811) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:65], v[46:49], a[20:23], v136, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 859) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:65], v[50:53], a[24:27], v136, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 995) 
s_add_u32 m0, m0, 4096 // (op 26811) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:65], v[54:57], a[28:31], v136, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1131) 
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 26811) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:73], v[12:15], a[32:35], v137, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1899) 
v_add_co_u32 v140, s[24:25], v140, v126 // (op 26811) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[66:73], v[30:33], a[36:39], v137, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1861) 
v_addc_co_u32 v141, s[24:25], 0, v141, s[24:25] // (op 26811) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[66:73], v[34:37], a[40:43], v137, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1823) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[66:73], v[38:41], a[44:47], v137, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1785) 
s_add_u32 m0, m0, 4096 // (op 26811) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:73], v[42:45], a[48:51], v137, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1747) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[66:73], v[46:49], a[52:55], v137, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1709) 
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 26811) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[66:73], v[50:53], a[56:59], v137, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1671) 
v_mov_b32 v10, v11 // (op 26737) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[66:73], v[54:57], a[60:63], v137, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1633) 
v_add_u32 v20, 8, v10 // (op 26737) 

v_bfe_u32 v150, v132, 0, 8 // (op 24838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:81], v[12:15], a[64:67], v138, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3117) 
v_bfe_u32 v151, v132, 8, 8 // (op 24838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[74:81], v[30:33], a[68:71], v138, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3079) 
v_bfe_u32 v152, v132, 16, 8 // (op 24838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[74:81], v[34:37], a[72:75], v138, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3041) 
v_bfe_u32 v153, v132, 24, 8 // (op 24838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[74:81], v[38:41], a[76:79], v138, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3003) 
v_permlane16_swap_b32 v150, v151 // (op 24838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:81], v[42:45], a[80:83], v138, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2965) 
v_permlane16_swap_b32 v152, v153 // (op 24838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:81], v[46:49], a[84:87], v138, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2927) 
s_nop 0
v_permlane32_swap_b32 v150, v152 // (op 24838) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:81], v[50:53], a[88:91], v138, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2889) 
v_permlane32_swap_b32 v151, v153 // (op 24838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:81], v[54:57], a[92:95], v138, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2851) 
s_waitcnt vmcnt(20)
s_barrier  // (op 27830) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:89], v[12:15], a[96:99], v139, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4335) 
ds_read_b128 v[154:157], v6 offset:49152 // (op 216) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:89], v[30:33], a[100:103], v139, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4297) 
ds_read_b128 v[158:161], v6 offset:50176 // (op 324) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:89], v[34:37], a[104:107], v139, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4259) 
ds_read_b128 v[162:165], v6 offset:51200 // (op 460) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:89], v[38:41], a[108:111], v139, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4221) 
buffer_load_dwordx2 v[24:25], v20, s[68:71], 0 offen // (op 26737) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[82:89], v[42:45], a[112:115], v139, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4183) 
s_lshl_b32 s22, s38, 6 // (op 26750) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[82:89], v[46:49], a[116:119], v139, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4145) 
s_add_u32 s22, s22, 8 // (op 26750) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[82:89], v[50:53], a[120:123], v139, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4107) 
v_add_u32 v20, v10, s22 // (op 26750) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[82:89], v[54:57], a[124:127], v139, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4069) 
buffer_load_dwordx2 v[26:27], v20, s[68:71], 0 offen // (op 26750) Load value





v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[90:97], v[12:15], a[128:131], v150, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5553) 
v_add_u32 v11, 8, v11 // (op 27381) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[90:97], v[30:33], a[132:135], v150, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5515) 
v_mov_b32 v10, v21 // (op 26763) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[90:97], v[34:37], a[136:139], v150, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5477) 
s_lshl_b32 s22, s34, 6 // (op 26763) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[90:97], v[38:41], a[140:143], v150, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5439) 
s_add_u32 s23, 8, s22 // (op 26763) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[90:97], v[42:45], a[144:147], v150, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5401) 
v_add_u32 v20, v10, s23 // (op 26763) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[90:97], v[46:49], a[148:151], v150, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5363) 
buffer_load_dwordx2 v[22:23], v20, s[64:67], 0 offen // (op 26763) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[90:97], v[50:53], a[152:155], v150, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5325) 
v_add_u32 v20, 8, v10 // (op 26776) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:97], v[54:57], a[156:159], v150, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5287) 
buffer_load_dwordx2 v[28:29], v20, s[64:67], 0 offen // (op 26776) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[98:105], v[12:15], a[160:163], v151, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6771) 
v_add_u32 v21, 8, v21 // (op 27329) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[98:105], v[30:33], a[164:167], v151, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6733) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[98:105], v[34:37], a[168:171], v151, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6695) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[98:105], v[38:41], a[172:175], v151, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6657) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[98:105], v[42:45], a[176:179], v151, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6619) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[98:105], v[46:49], a[180:183], v151, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6581) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[98:105], v[50:53], a[184:187], v151, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6543) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[98:105], v[54:57], a[188:191], v151, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6505) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[106:113], v[12:15], a[192:195], v152, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7989) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[106:113], v[30:33], a[196:199], v152, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7951) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[106:113], v[34:37], a[200:203], v152, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7913) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[106:113], v[38:41], a[204:207], v152, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7875) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[106:113], v[42:45], a[208:211], v152, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7837) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[106:113], v[46:49], a[212:215], v152, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7799) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[106:113], v[50:53], a[216:219], v152, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7761) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[106:113], v[54:57], a[220:223], v152, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7723) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[114:121], v[12:15], a[224:227], v153, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 9207) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[114:121], v[30:33], a[228:231], v153, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 9169) 
ds_read_b128 v[136:139], v6 offset:52224 // (op 596) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[114:121], v[34:37], a[232:235], v153, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 9131) 
ds_read_b128 v[166:169], v6 offset:53248 // (op 732) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:121], v[38:41], a[236:239], v153, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 9093) 
ds_read_b128 v[170:173], v6 offset:54272 // (op 868) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[114:121], v[42:45], a[240:243], v153, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 9055) 
ds_read_b128 v[140:143], v6 offset:55296 // (op 1004) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[114:121], v[46:49], a[244:247], v153, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 9017) 
ds_read_b128 v[174:177], v6 offset:56320 // (op 1140) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[114:121], v[50:53], a[248:251], v153, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 8979) 
s_waitcnt vmcnt(16)
s_barrier  // (op 27824) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[114:121], v[54:57], a[252:255], v153, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 8941) 
v_add_u32 v10, 65536, v5 // (op 214) 


ds_read_b128 v[58:61], v10 // (op 214) Load local data 

v_add_u32 v10, 65600, v5 // (op 214) 
ds_read_b128 v[62:65], v10 // (op 214) Load local data 




v_add_u32 v10, 67584, v5 // (op 1641) 
ds_read_b128 v[66:69], v10 // (op 1641) Load local data 

v_add_u32 v10, 67648, v5 // (op 1641) 
ds_read_b128 v[70:73], v10 // (op 1641) Load local data 




v_add_u32 v10, 69632, v5 // (op 2859) 
ds_read_b128 v[74:77], v10 // (op 2859) Load local data 

v_add_u32 v10, 69696, v5 // (op 2859) 
ds_read_b128 v[78:81], v10 // (op 2859) Load local data 




v_add_u32 v10, 71680, v5 // (op 4077) 
ds_read_b128 v[82:85], v10 // (op 4077) Load local data 

v_add_u32 v10, 71744, v5 // (op 4077) 
ds_read_b128 v[86:89], v10 // (op 4077) Load local data 




v_add_u32 v10, 73728, v5 // (op 5295) 
ds_read_b128 v[90:93], v10 // (op 5295) Load local data 

v_add_u32 v10, 73792, v5 // (op 5295) 
ds_read_b128 v[94:97], v10 // (op 5295) Load local data 




v_add_u32 v10, 75776, v5 // (op 6513) 
ds_read_b128 v[98:101], v10 // (op 6513) Load local data 

v_add_u32 v10, 75840, v5 // (op 6513) 
ds_read_b128 v[102:105], v10 // (op 6513) Load local data 




v_add_u32 v10, 77824, v5 // (op 7731) 
ds_read_b128 v[106:109], v10 // (op 7731) Load local data 

v_add_u32 v10, 77888, v5 // (op 7731) 
ds_read_b128 v[110:113], v10 // (op 7731) Load local data 




v_add_u32 v10, 79872, v5 // (op 8949) 
ds_read_b128 v[144:147], v10 // (op 8949) Load local data 

v_add_u32 v10, 79936, v5 // (op 8949) 
ds_read_b128 v[148:151], v10 // (op 8949) Load local data 

v_bfe_u32 v178, v135, 0, 8 // (op 24822) 
v_bfe_u32 v179, v135, 8, 8 // (op 24822) 
v_bfe_u32 v180, v135, 16, 8 // (op 24822) 
v_bfe_u32 v181, v135, 24, 8 // (op 24822) 
v_permlane16_swap_b32 v178, v179 // (op 24822) 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 27834) 
v_permlane16_swap_b32 v180, v181 // (op 24822) 

s_lshl_b64 s[22:23], s[30:31], 6 // (op 26838) 
s_lshl_b64 s[24:25], s[22:23], 2 // (op 26838) 

v_permlane32_swap_b32 v178, v180 // (op 24822) 
v_permlane32_swap_b32 v179, v181 // (op 24822) 
v_lshrrev_b64 v[126:127], 3, s[24:25] // (op 26838) 



v_mov_b64 v[134:135], v[128:129] // (op 26838) 
s_add_u32 s22, 49152, s18 // (op 26838) 
s_mov_b32 m0, s22 // (op 26838) reg expression
s_mov_b32 s23, 192 // (op 26838) call()
buffer_load_dwordx4 v134, s[60:63], s23 offen offset: 0 lds // (op 26838) Load value direct to lds
v_add_co_u32 v134, s[24:25], v134, v126 // (op 26838) least significant half
v_addc_co_u32 v135, s[24:25], 0, v135, s[24:25] // (op 26838) most significant half

s_add_u32 m0, m0, 4096 // (op 26838) 
buffer_load_dwordx4 v134, s[60:63], s23 offen offset: 0 lds // (op 26838) Load value direct to lds
v_add_co_u32 v134, s[24:25], v134, v126 // (op 26838) least significant half
v_addc_co_u32 v135, s[24:25], 0, v135, s[24:25] // (op 26838) most significant half

s_add_u32 m0, m0, 4096 // (op 26838) 
buffer_load_dwordx4 v134, s[60:63], s23 offen offset: 0 lds // (op 26838) Load value direct to lds
v_add_co_u32 v134, s[24:25], v134, v126 // (op 26838) least significant half
v_addc_co_u32 v135, s[24:25], 0, v135, s[24:25] // (op 26838) most significant half

s_add_u32 m0, m0, 4096 // (op 26838) 
v_bfe_u32 v182, v123, 0, 8 // (op 25013) 
v_bfe_u32 v183, v123, 8, 8 // (op 25013) 
v_bfe_u32 v184, v123, 16, 8 // (op 25013) 
v_bfe_u32 v185, v123, 24, 8 // (op 25013) 
v_permlane16_swap_b32 v182, v183 // (op 25013) 
s_nop 0
v_permlane16_swap_b32 v184, v185 // (op 25013) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v182, v184 // (op 25013) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v183, v185 // (op 25013) 
buffer_load_dwordx4 v134, s[60:63], s23 offen offset: 0 lds // (op 26838) Load value direct to lds






s_waitcnt lgkmcnt(14)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:65], v[154:157], a[0:3], v178, v182 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 221) 
v_add_co_u32 v18, vcc, 128, v18 // (op 27032) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:65], v[158:161], a[4:7], v178, v183 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 316) 
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 27032) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[58:65], v[162:165], a[8:11], v178, v184 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 452) 
v_bfe_u32 v126, v125, 0, 8 // (op 25045) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:65], v[136:139], a[12:15], v178, v185 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 588) 
v_bfe_u32 v127, v125, 8, 8 // (op 25045) 
v_bfe_u32 v128, v125, 16, 8 // (op 25045) 
v_bfe_u32 v129, v125, 24, 8 // (op 25045) 
v_permlane16_swap_b32 v126, v127 // (op 25045) 
s_nop 0
v_permlane16_swap_b32 v128, v129 // (op 25045) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v126, v128 // (op 25045) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v127, v129 // (op 25045) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 27827) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[58:65], v[166:169], a[16:19], v178, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 724) 
s_lshl_b64 s[22:23], s[26:27], 5 // (op 26833) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:65], v[170:173], a[20:23], v178, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 860) 
v_mov_b64 v[134:135], s[22:23] // (op 26833) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:65], v[140:143], a[24:27], v178, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 996) 
v_mov_b64 v[152:153], v[130:131] // (op 26833) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:65], v[174:177], a[28:31], v178, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1132) 
s_add_u32 s22, 65536, s1 // (op 26833) 
s_mov_b32 m0, s22 // (op 26833) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:73], v[154:157], a[32:35], v179, v182 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1900) 
s_mov_b32 s23, 384 // (op 26833) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[66:73], v[158:161], a[36:39], v179, v183 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1862) 
buffer_load_dwordx4 v152, s[48:51], s23 offen offset: 0 lds // (op 26833) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[66:73], v[162:165], a[40:43], v179, v184 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1824) 
v_add_co_u32 v152, s[24:25], v152, v134 // (op 26833) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[66:73], v[136:139], a[44:47], v179, v185 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1786) 
v_addc_co_u32 v153, s[24:25], 0, v153, s[24:25] // (op 26833) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:73], v[166:169], a[48:51], v179, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1748) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[66:73], v[170:173], a[52:55], v179, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1710) 
s_add_u32 m0, m0, 4096 // (op 26833) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[66:73], v[140:143], a[56:59], v179, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1672) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[66:73], v[174:177], a[60:63], v179, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1634) 
buffer_load_dwordx4 v152, s[48:51], s23 offen offset: 0 lds // (op 26833) Load value direct to lds
v_add_co_u32 v152, s[24:25], v152, v134 // (op 26833) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:81], v[154:157], a[64:67], v180, v182 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3118) 
v_addc_co_u32 v153, s[24:25], 0, v153, s[24:25] // (op 26833) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[74:81], v[158:161], a[68:71], v180, v183 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3080) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[74:81], v[162:165], a[72:75], v180, v184 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3042) 
s_add_u32 m0, m0, 4096 // (op 26833) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[74:81], v[136:139], a[76:79], v180, v185 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3004) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:81], v[166:169], a[80:83], v180, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2966) 
v_bfe_u32 v122, v133, 0, 8 // (op 24853) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:81], v[170:173], a[84:87], v180, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2928) 
v_bfe_u32 v123, v133, 8, 8 // (op 24853) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:81], v[140:143], a[88:91], v180, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2890) 
v_bfe_u32 v124, v133, 16, 8 // (op 24853) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:81], v[174:177], a[92:95], v180, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2852) 
v_bfe_u32 v125, v133, 24, 8 // (op 24853) 
v_permlane16_swap_b32 v122, v123 // (op 24853) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:89], v[154:157], a[96:99], v181, v182 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4336) 
v_permlane16_swap_b32 v124, v125 // (op 24853) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:89], v[158:161], a[100:103], v181, v183 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4298) 
s_nop 0
v_permlane32_swap_b32 v122, v124 // (op 24853) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:89], v[162:165], a[104:107], v181, v184 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4260) 
v_permlane32_swap_b32 v123, v125 // (op 24853) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:89], v[136:139], a[108:111], v181, v185 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4222) 
s_waitcnt vmcnt(18)
s_barrier  // (op 27817) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[82:89], v[166:169], a[112:115], v181, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4184) 
ds_read_b128 v[12:15], v4 // (op 67) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[82:89], v[170:173], a[116:119], v181, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4146) 
ds_read_b128 v[30:33], v4 offset:1024 // (op 312) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[82:89], v[140:143], a[120:123], v181, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4108) 
ds_read_b128 v[34:37], v4 offset:2048 // (op 448) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[82:89], v[174:177], a[124:127], v181, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4070) 
buffer_load_dwordx4 v152, s[48:51], s23 offen offset: 0 lds // (op 26833) Load value direct to lds

v_add_co_u32 v152, s[24:25], v152, v134 // (op 26833) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[90:97], v[154:157], a[128:131], v122, v182 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5554) 
v_addc_co_u32 v153, s[24:25], 0, v153, s[24:25] // (op 26833) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[90:97], v[158:161], a[132:135], v122, v183 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5516) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[90:97], v[162:165], a[136:139], v122, v184 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5478) 
s_add_u32 m0, m0, 4096 // (op 26833) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[90:97], v[136:139], a[140:143], v122, v185 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5440) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[90:97], v[166:169], a[144:147], v122, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5402) 
buffer_load_dwordx4 v152, s[48:51], s23 offen offset: 0 lds // (op 26833) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[90:97], v[170:173], a[148:151], v122, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5364) 
v_add_co_u32 v152, s[24:25], v152, v134 // (op 26833) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[90:97], v[140:143], a[152:155], v122, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5326) 
v_addc_co_u32 v153, s[24:25], 0, v153, s[24:25] // (op 26833) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:97], v[174:177], a[156:159], v122, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5288) 
s_add_u32 m0, m0, 4096 // (op 26833) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[98:105], v[154:157], a[160:163], v123, v182 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6772) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[98:105], v[158:161], a[164:167], v123, v183 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6734) 
buffer_load_dwordx4 v152, s[48:51], s23 offen offset: 0 lds // (op 26833) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[98:105], v[162:165], a[168:171], v123, v184 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6696) 
v_add_co_u32 v152, s[24:25], v152, v134 // (op 26833) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[98:105], v[136:139], a[172:175], v123, v185 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6658) 
v_addc_co_u32 v153, s[24:25], 0, v153, s[24:25] // (op 26833) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[98:105], v[166:169], a[176:179], v123, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6620) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[98:105], v[170:173], a[180:183], v123, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6582) 
s_add_u32 m0, m0, 4096 // (op 26833) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[98:105], v[140:143], a[184:187], v123, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6544) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[98:105], v[174:177], a[188:191], v123, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6506) 
buffer_load_dwordx4 v152, s[48:51], s23 offen offset: 0 lds // (op 26833) Load value direct to lds
v_add_co_u32 v152, s[24:25], v152, v134 // (op 26833) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[106:113], v[154:157], a[192:195], v124, v182 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7990) 
v_addc_co_u32 v153, s[24:25], 0, v153, s[24:25] // (op 26833) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[106:113], v[158:161], a[196:199], v124, v183 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7952) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[106:113], v[162:165], a[200:203], v124, v184 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7914) 
s_add_u32 m0, m0, 4096 // (op 26833) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[106:113], v[136:139], a[204:207], v124, v185 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7876) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[106:113], v[166:169], a[208:211], v124, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7838) 
buffer_load_dwordx4 v152, s[48:51], s23 offen offset: 0 lds // (op 26833) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[106:113], v[170:173], a[212:215], v124, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7800) 
v_add_co_u32 v152, s[24:25], v152, v134 // (op 26833) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[106:113], v[140:143], a[216:219], v124, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7762) 
v_addc_co_u32 v153, s[24:25], 0, v153, s[24:25] // (op 26833) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[106:113], v[174:177], a[220:223], v124, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7724) 
s_add_u32 m0, m0, 4096 // (op 26833) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[144:151], v[154:157], a[224:227], v125, v182 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 9208) 

buffer_load_dwordx4 v152, s[48:51], s23 offen offset: 0 lds // (op 26833) Load value direct to lds








v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[144:151], v[158:161], a[228:231], v125, v183 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 9170) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[144:151], v[162:165], a[232:235], v125, v184 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 9132) 
ds_read_b128 v[38:41], v4 offset:3072 // (op 584) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[144:151], v[136:139], a[236:239], v125, v185 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 9094) 
ds_read_b128 v[42:45], v4 offset:4096 // (op 720) Load local data 





v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[144:151], v[166:169], a[240:243], v125, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 9056) 
ds_read_b128 v[46:49], v4 offset:5120 // (op 856) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[144:151], v[170:173], a[244:247], v125, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 9018) 
v_add_co_u32 v16, vcc, 256, v16 // (op 26968) least significant half

v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 26968) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[144:151], v[140:143], a[248:251], v125, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 8980) 
ds_read_b128 v[50:53], v4 offset:6144 // (op 992) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[144:151], v[174:177], a[252:255], v125, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 8942) 
ds_read_b128 v[54:57], v4 offset:7168 // (op 1128) Load local data 




s_waitcnt vmcnt(16)
s_barrier  // (op 27801) 



ds_read_b128 v[58:61], v3 offset:16384 // (op 62) Load local data 
ds_read_b128 v[62:65], v3 offset:16448 // (op 62) Load local data 



ds_read_b128 v[66:69], v3 offset:18432 // (op 1629) Load local data 
ds_read_b128 v[70:73], v3 offset:18496 // (op 1629) Load local data 



ds_read_b128 v[74:77], v3 offset:20480 // (op 2847) Load local data 
ds_read_b128 v[78:81], v3 offset:20544 // (op 2847) Load local data 



ds_read_b128 v[82:85], v3 offset:22528 // (op 4065) Load local data 
ds_read_b128 v[86:89], v3 offset:22592 // (op 4065) Load local data 



ds_read_b128 v[90:93], v3 offset:24576 // (op 5283) Load local data 
ds_read_b128 v[94:97], v3 offset:24640 // (op 5283) Load local data 



ds_read_b128 v[98:101], v3 offset:26624 // (op 6501) Load local data 
ds_read_b128 v[102:105], v3 offset:26688 // (op 6501) Load local data 



ds_read_b128 v[106:109], v3 offset:28672 // (op 7719) Load local data 
ds_read_b128 v[110:113], v3 offset:28736 // (op 7719) Load local data 



ds_read_b128 v[114:117], v3 offset:30720 // (op 8937) Load local data 
ds_read_b128 v[118:121], v3 offset:30784 // (op 8937) Load local data 
s_ashr_i64 s[22:23], s[42:43], 7 // (op 8907) 
s_mov_b32 s24, s22 // (op 8907) convert

s_and_b32 s25, s24, 4294967294 // (op 8907) 

s_cmp_lt_u32 s19, s25 // (op 8907) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoop_8907 // (op 8907) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoop_8907 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoop_8907:



















s_barrier  // (op 27926) 
s_mul_hi_i32 s47, s16, s10 // (op 26868) 
v_lshrrev_b32 v10, 3, v1 // (op 26868) 
s_add_i32 s0, s47, s16 // (op 26868) 

v_and_b32 v11, -8, v1 // (op 26868) 
s_ashr_i32 s1, s0, 31 // (op 26868) 
v_sub_u32 v12, v1, v11 // (op 26868) 

s_cmp_eq_i32 s10, 0 // (op 26868) 
s_mov_b32 s47, scc // (op 26868) 
v_lshlrev_b32 v11, 4, v12 // (op 26868) 

s_cmp_eq_i32 s47, 0 // (op 26868) 
s_cselect_b32 s18, 0, -1 // (op 26868) 

s_and_b32 s19, s13, 31 // (op 26868) 
s_lshl_b32 s47, 1, s19 // (op 26868) 
s_add_i32 s22, s47, s18 // (op 26868) 


s_and_b32 s18, s1, s22 // (op 26868) 


s_add_i32 s1, s0, s18 // (op 26868) 


s_ashr_i32 s0, s1, s19 // (op 26868) 


s_lshl_b32 s1, s13, 24 // (op 26868) 
s_ashr_i32 s47, s1, 31 // (op 26868) 

s_xor_b32 s1, s0, s47 // (op 26868) 

s_sub_i32 s0, s1, s47 // (op 26868) 


s_mul_hi_i32 s1, s0, s14 // (op 26868) 
s_add_i32 s47, s1, s0 // (op 26868) 

s_ashr_i32 s1, s47, 31 // (op 26868) 
s_cmp_eq_i32 s14, 0 // (op 26868) 
s_mov_b32 s18, scc // (op 26868) 
s_cmp_eq_i32 s18, 0 // (op 26868) 
s_cselect_b32 s19, 0, -1 // (op 26868) 

s_and_b32 s18, s15, 31 // (op 26868) 
s_lshl_b32 s22, 1, s18 // (op 26868) 
s_add_i32 s23, s22, s19 // (op 26868) 


s_and_b32 s19, s1, s23 // (op 26868) 


s_add_i32 s1, s47, s19 // (op 26868) 


s_ashr_i32 s19, s1, s18 // (op 26868) 


s_lshl_b32 s1, s15, 24 // (op 26868) 
s_ashr_i32 s18, s1, 31 // (op 26868) 

s_xor_b32 s1, s19, s18 // (op 26868) 

s_sub_i32 s19, s1, s18 // (op 26868) 


s_mul_i32 s1, s0, s11 // (op 26868) 

s_sub_u32 s0, s16, s1 // (op 26868) 

s_mul_i32 s1, s19, s11 // (op 26868) 
s_add_u32 s1, s1, s0 // (op 26868) 


s_mul_hi_i32 s0, s16, s9 // (op 26868) 
s_add_i32 s47, s0, s16 // (op 26868) 

s_ashr_i32 s0, s47, 31 // (op 26868) 
s_cmp_eq_i32 s9, 0 // (op 26868) 
s_mov_b32 s18, scc // (op 26868) 
s_cmp_eq_i32 s18, 0 // (op 26868) 
s_cselect_b32 s19, 0, -1 // (op 26868) 

s_and_b32 s18, s45, 31 // (op 26868) 
s_lshl_b32 s22, 1, s18 // (op 26868) 
s_add_i32 s23, s22, s19 // (op 26868) 


s_and_b32 s19, s0, s23 // (op 26868) 


s_add_i32 s0, s47, s19 // (op 26868) 


s_ashr_i32 s19, s0, s18 // (op 26868) 


s_lshl_b32 s0, s45, 24 // (op 26868) 
s_ashr_i32 s18, s0, 31 // (op 26868) 

s_xor_b32 s0, s19, s18 // (op 26868) 

s_sub_i32 s19, s0, s18 // (op 26868) 


s_add_u32 s22, s6, 256 // (op 26868) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 26868) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 26868) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 26868) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 26868) 

s_mul_i32 s0, s11, s22 // (op 26868) 

s_mul_i32 s18, s8, s0 // (op 26868) 

s_mul_i32 s0, s19, s18 // (op 26868) 

s_sub_u32 s18, s16, s0 // (op 26868) 

s_mov_b32 s0, s18 // (op 26868) 
s_mul_hi_i32 s47, s0, s12 // (op 26868) 
s_add_i32 s22, s47, s0 // (op 26868) 


s_ashr_i32 s0, s22, 31 // (op 26868) 
s_cmp_eq_i32 s12, 0 // (op 26868) 
s_mov_b32 s47, scc // (op 26868) 
s_cmp_eq_i32 s47, 0 // (op 26868) 
s_cselect_b32 s23, 0, -1 // (op 26868) 

s_and_b32 s47, s44, 31 // (op 26868) 
s_lshl_b32 s24, 1, s47 // (op 26868) 
s_add_i32 s25, s24, s23 // (op 26868) 


s_and_b32 s23, s0, s25 // (op 26868) 


s_add_i32 s0, s22, s23 // (op 26868) 


s_ashr_i32 s22, s0, s47 // (op 26868) 


s_lshl_b32 s0, s44, 24 // (op 26868) 
s_ashr_i32 s47, s0, 31 // (op 26868) 

s_xor_b32 s0, s22, s47 // (op 26868) 

s_sub_i32 s22, s0, s47 // (op 26868) 


s_mul_i32 s0, s46, s11 // (op 26868) 
s_add_u32 s24, s20, 256 // (op 26868) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 26868) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 26868) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 26868) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 26868) 

s_sub_i32 s23, s24, s0 // (op 26868) 

s_mul_i32 s47, s22, s23 // (op 26868) 


s_sub_u32 s22, s18, s47 // (op 26868) 


s_add_u32 s18, s22, s0 // (op 26868) 


s_cmp_eq_i32 s19, 0 // (op 26868) 
s_mov_b32 s0, scc // (op 26868) 

s_cmp_eq_i32 s0, 0 // (op 26868) 
s_cselect_b32 s19, s18, s1 // (op 26868) 



v_lshl_add_u32 v12, s19, 8, v10 // (op 26868) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 26868) 
s_lshr_b32 s47, s0, 1 // (op 26868) 

v_lshl_add_u32 v10, s47, 8, v11 // (op 26868) 


v_mov_b32 v21, 0 // (op 26868) convert
v_mov_b32 v20, v10 // (op 26868) convert

v_mul_lo_u32 v11, v12, s27 // (op 26868) most significant: low of low * high
v_mul_hi_u32 v13, v12, s26 // (op 26868) most significant: high of low * low
v_mul_lo_u32 v10, v12, s26 // (op 26868) least significant: low of low * low
v_add_u32 v11, v11, v13 // (op 26868) most significant: sum

v_add_co_u32 v10, s[0:1], v10, v20 // (op 26868) least significant half
v_addc_co_u32 v11, s[0:1], v11, v21, s[0:1] // (op 26868) most significant half



s_mul_i32 s1, s26, s21 // (op 26870) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 26870) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 26870) least significant
s_mul_hi_u32 s18, s26, s20 // (op 26870) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 26870) most significant: sum
s_add_u32 s1, s1, s18 // (op 26870) most significant: sum


s_mov_b32 s48, s28 // (op 26870) reg expression
s_mov_b32 s49, s29 // (op 26870) reg expression
s_mov_b32 s50, s0 // (op 26870) reg expression
s_mov_b32 s51, 131072 // (op 26870) call()

s_mul_hi_i32 s47, s16, s10 // (op 26998) 
v_lshrrev_b32 v12, 2, v1 // (op 26998) 
s_add_i32 s0, s47, s16 // (op 26998) 

v_and_b32 v13, -4, v1 // (op 26998) 
s_ashr_i32 s1, s0, 31 // (op 26998) 
v_sub_u32 v14, v1, v13 // (op 26998) 

s_cmp_eq_i32 s10, 0 // (op 26998) 
s_mov_b32 s47, scc // (op 26998) 
v_lshlrev_b32 v13, 5, v14 // (op 26998) 

s_cmp_eq_i32 s47, 0 // (op 26998) 
s_cselect_b32 s18, 0, -1 // (op 26998) 

s_and_b32 s19, s13, 31 // (op 26998) 
s_lshl_b32 s47, 1, s19 // (op 26998) 
s_add_i32 s22, s47, s18 // (op 26998) 


s_and_b32 s18, s1, s22 // (op 26998) 


s_add_i32 s1, s0, s18 // (op 26998) 


s_ashr_i32 s0, s1, s19 // (op 26998) 


s_lshl_b32 s1, s13, 24 // (op 26998) 
s_ashr_i32 s47, s1, 31 // (op 26998) 

s_xor_b32 s1, s0, s47 // (op 26998) 

s_sub_i32 s0, s1, s47 // (op 26998) 


s_mul_hi_i32 s1, s0, s14 // (op 26998) 
s_add_i32 s47, s1, s0 // (op 26998) 

s_ashr_i32 s1, s47, 31 // (op 26998) 
s_cmp_eq_i32 s14, 0 // (op 26998) 
s_mov_b32 s18, scc // (op 26998) 
s_cmp_eq_i32 s18, 0 // (op 26998) 
s_cselect_b32 s19, 0, -1 // (op 26998) 

s_and_b32 s18, s15, 31 // (op 26998) 
s_lshl_b32 s22, 1, s18 // (op 26998) 
s_add_i32 s23, s22, s19 // (op 26998) 


s_and_b32 s19, s1, s23 // (op 26998) 


s_add_i32 s1, s47, s19 // (op 26998) 


s_ashr_i32 s19, s1, s18 // (op 26998) 


s_lshl_b32 s1, s15, 24 // (op 26998) 
s_ashr_i32 s18, s1, 31 // (op 26998) 

s_xor_b32 s1, s19, s18 // (op 26998) 

s_sub_i32 s19, s1, s18 // (op 26998) 


s_add_u32 s22, s6, 256 // (op 26998) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 26998) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 26998) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 26998) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 26998) 

s_mul_i32 s1, s19, s22 // (op 26998) 

s_sub_i32 s47, s0, s1 // (op 26998) 


s_mul_hi_i32 s0, s16, s9 // (op 26998) 
s_add_i32 s1, s0, s16 // (op 26998) 

s_ashr_i32 s0, s1, 31 // (op 26998) 
s_cmp_eq_i32 s9, 0 // (op 26998) 
s_mov_b32 s18, scc // (op 26998) 
s_cmp_eq_i32 s18, 0 // (op 26998) 
s_cselect_b32 s19, 0, -1 // (op 26998) 

s_and_b32 s18, s45, 31 // (op 26998) 
s_lshl_b32 s24, 1, s18 // (op 26998) 
s_add_i32 s25, s24, s19 // (op 26998) 


s_and_b32 s19, s0, s25 // (op 26998) 


s_add_i32 s0, s1, s19 // (op 26998) 


s_ashr_i32 s1, s0, s18 // (op 26998) 


s_lshl_b32 s0, s45, 24 // (op 26998) 
s_ashr_i32 s18, s0, 31 // (op 26998) 

s_xor_b32 s0, s1, s18 // (op 26998) 

s_sub_i32 s1, s0, s18 // (op 26998) 


s_mul_i32 s0, s11, s22 // (op 26998) 

s_mul_i32 s18, s8, s0 // (op 26998) 

s_mul_i32 s0, s1, s18 // (op 26998) 

s_sub_u32 s18, s16, s0 // (op 26998) 

s_mul_hi_i32 s0, s18, s12 // (op 26998) 
s_add_i32 s19, s0, s18 // (op 26998) 


s_ashr_i32 s0, s19, 31 // (op 26998) 
s_cmp_eq_i32 s12, 0 // (op 26998) 
s_mov_b32 s18, scc // (op 26998) 
s_cmp_eq_i32 s18, 0 // (op 26998) 
s_cselect_b32 s22, 0, -1 // (op 26998) 

s_and_b32 s18, s44, 31 // (op 26998) 
s_lshl_b32 s23, 1, s18 // (op 26998) 
s_add_i32 s24, s23, s22 // (op 26998) 


s_and_b32 s25, s0, s24 // (op 26998) 


s_add_i32 s0, s19, s25 // (op 26998) 


s_ashr_i32 s19, s0, s18 // (op 26998) 


s_lshl_b32 s0, s44, 24 // (op 26998) 
s_ashr_i32 s18, s0, 31 // (op 26998) 

s_xor_b32 s0, s19, s18 // (op 26998) 

s_sub_i32 s19, s0, s18 // (op 26998) 


s_cmp_eq_i32 s1, 0 // (op 26998) 
s_mov_b32 s0, scc // (op 26998) 

s_cmp_eq_i32 s0, 0 // (op 26998) 
s_cselect_b32 s1, s19, s47 // (op 26998) 



v_lshl_add_u32 v14, s1, 8, v12 // (op 26998) 


v_mul_lo_u32 v21, v14, s31 // (op 26998) most significant: low of low * high
v_mul_hi_u32 v12, v14, s30 // (op 26998) most significant: high of low * low
v_mul_lo_u32 v20, v14, s30 // (op 26998) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 26998) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 26998) 
s_lshr_b32 s47, s0, 1 // (op 26998) 

v_lshl_add_u32 v12, s47, 8, v13 // (op 26998) 


v_mov_b32 v19, 0 // (op 26998) convert
v_mov_b32 v18, v12 // (op 26998) convert

v_add_co_u32 v12, s[0:1], v18, v20 // (op 26998) least significant half
v_addc_co_u32 v13, s[0:1], v19, v21, s[0:1] // (op 26998) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 26998) 


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 26998) 

s_mul_i32 s1, s30, s7 // (op 27000) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 27000) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 27000) least significant
s_mul_hi_u32 s18, s30, s6 // (op 27000) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27000) most significant: sum
s_add_u32 s1, s1, s18 // (op 27000) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 27000) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 27000) 

s_mov_b32 s60, s32 // (op 27000) reg expression
s_mov_b32 s61, s33 // (op 27000) reg expression
s_mov_b32 s62, s0 // (op 27000) reg expression
s_mov_b32 s63, 131072 // (op 27000) call()

v_and_b32 v12, -8, v1 // (op 27081) 
v_sub_u32 v13, v1, v12 // (op 27081) 

v_lshlrev_b32 v12, 4, v13 // (op 27081) 

v_lshrrev_b32 v13, 3, v1 // (op 27081) 
v_lshl_add_u32 v16, v13, 7, v12 // (op 27081) 


s_nop 0
v_readfirstlane_b32 s47, v16 // (op 27081) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -4, v1 // (op 27140) 
v_sub_u32 v13, v1, v12 // (op 27140) 

v_lshlrev_b32 v12, 5, v13 // (op 27140) 

v_lshrrev_b32 v13, 2, v1 // (op 27140) 
v_lshl_add_u32 v16, v13, 7, v12 // (op 27140) 


v_lshlrev_b32 v12, 2, v16 // (op 27140) 

v_lshrrev_b32 v13, 3, v12 // (op 27140) 

s_nop 0
v_readfirstlane_b32 s0, v13 // (op 27140) Wait state hazard: VALU Write Readlane Hazard

s_mul_hi_i32 s1, s16, s10 // (op 27418) 
v_and_b32 v12, 63, v1 // (op 27418) 
s_add_i32 s18, s1, s16 // (op 27418) 

v_lshrrev_b32 v13, 4, v12 // (op 27418) 
s_ashr_i32 s1, s18, 31 // (op 27418) 
v_and_b32 v16, 3, v13 // (op 27418) 

s_cmp_eq_i32 s10, 0 // (op 27418) 
s_mov_b32 s19, scc // (op 27418) 
v_and_b32 v13, 15, v1 // (op 27418) 
s_cmp_eq_i32 s19, 0 // (op 27418) 
s_cselect_b32 s22, 0, -1 // (op 27418) 

v_lshl_add_u32 v17, v16, 4, v13 // (op 27418) 


s_and_b32 s19, s13, 31 // (op 27418) 
v_and_b32 v13, 4294967168, v1 // (op 27418) 
s_lshl_b32 s23, 1, s19 // (op 27418) 
v_add_u32 v16, v13, v17 // (op 27418) 


s_add_i32 s24, s23, s22 // (op 27418) 


v_lshrrev_b32 v13, 6, v12 // (op 27418) 

s_and_b32 s25, s1, s24 // (op 27418) 


v_lshlrev_b32 v12, 2, v13 // (op 27418) 

s_add_i32 s1, s18, s25 // (op 27418) 


s_ashr_i32 s18, s1, s19 // (op 27418) 


s_lshl_b32 s1, s13, 24 // (op 27418) 
s_ashr_i32 s19, s1, 31 // (op 27418) 

s_xor_b32 s1, s18, s19 // (op 27418) 

s_sub_i32 s18, s1, s19 // (op 27418) 


s_mul_hi_i32 s1, s18, s14 // (op 27418) 
s_add_i32 s19, s1, s18 // (op 27418) 

s_ashr_i32 s1, s19, 31 // (op 27418) 
s_cmp_eq_i32 s14, 0 // (op 27418) 
s_mov_b32 s22, scc // (op 27418) 
s_cmp_eq_i32 s22, 0 // (op 27418) 
s_cselect_b32 s23, 0, -1 // (op 27418) 

s_and_b32 s22, s15, 31 // (op 27418) 
s_lshl_b32 s24, 1, s22 // (op 27418) 
s_add_i32 s25, s24, s23 // (op 27418) 


s_and_b32 s23, s1, s25 // (op 27418) 


s_add_i32 s1, s19, s23 // (op 27418) 


s_ashr_i32 s19, s1, s22 // (op 27418) 


s_lshl_b32 s1, s15, 24 // (op 27418) 
s_ashr_i32 s22, s1, 31 // (op 27418) 

s_xor_b32 s1, s19, s22 // (op 27418) 

s_sub_i32 s19, s1, s22 // (op 27418) 


s_mul_i32 s1, s18, s11 // (op 27418) 

s_sub_u32 s18, s16, s1 // (op 27418) 

s_mul_i32 s1, s19, s11 // (op 27418) 
s_add_u32 s1, s1, s18 // (op 27418) 


s_mul_hi_i32 s18, s16, s9 // (op 27418) 
s_add_i32 s19, s18, s16 // (op 27418) 

s_ashr_i32 s18, s19, 31 // (op 27418) 
s_cmp_eq_i32 s9, 0 // (op 27418) 
s_mov_b32 s22, scc // (op 27418) 
s_cmp_eq_i32 s22, 0 // (op 27418) 
s_cselect_b32 s23, 0, -1 // (op 27418) 

s_and_b32 s22, s45, 31 // (op 27418) 
s_lshl_b32 s24, 1, s22 // (op 27418) 
s_add_i32 s25, s24, s23 // (op 27418) 


s_and_b32 s23, s18, s25 // (op 27418) 


s_add_i32 s18, s19, s23 // (op 27418) 


s_ashr_i32 s19, s18, s22 // (op 27418) 


s_lshl_b32 s18, s45, 24 // (op 27418) 
s_ashr_i32 s22, s18, 31 // (op 27418) 

s_xor_b32 s18, s19, s22 // (op 27418) 

s_sub_i32 s19, s18, s22 // (op 27418) 


s_add_u32 s22, s6, 256 // (op 27418) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27418) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27418) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27418) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27418) 

s_mul_i32 s18, s11, s22 // (op 27418) 

s_mul_i32 s22, s8, s18 // (op 27418) 

s_mul_i32 s18, s19, s22 // (op 27418) 

s_sub_u32 s22, s16, s18 // (op 27418) 

s_mov_b32 s18, s22 // (op 27418) 
s_mul_hi_i32 s23, s18, s12 // (op 27418) 
s_add_i32 s24, s23, s18 // (op 27418) 


s_ashr_i32 s18, s24, 31 // (op 27418) 
s_cmp_eq_i32 s12, 0 // (op 27418) 
s_mov_b32 s23, scc // (op 27418) 
s_cmp_eq_i32 s23, 0 // (op 27418) 
s_cselect_b32 s25, 0, -1 // (op 27418) 

s_and_b32 s23, s44, 31 // (op 27418) 
s_lshl_b32 s64, 1, s23 // (op 27418) 
s_add_i32 s65, s64, s25 // (op 27418) 


s_and_b32 s25, s18, s65 // (op 27418) 


s_add_i32 s18, s24, s25 // (op 27418) 


s_ashr_i32 s24, s18, s23 // (op 27418) 


s_lshl_b32 s18, s44, 24 // (op 27418) 
s_ashr_i32 s23, s18, 31 // (op 27418) 

s_xor_b32 s18, s24, s23 // (op 27418) 

s_sub_i32 s24, s18, s23 // (op 27418) 


s_mul_i32 s18, s46, s11 // (op 27418) 
s_add_u32 s64, s20, 256 // (op 27418) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 27418) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 27418) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 27418) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 8 // (op 27418) 

s_sub_i32 s23, s64, s18 // (op 27418) 

s_mul_i32 s25, s24, s23 // (op 27418) 


s_sub_u32 s23, s22, s25 // (op 27418) 


s_add_u32 s22, s23, s18 // (op 27418) 


s_cmp_eq_i32 s19, 0 // (op 27418) 
s_mov_b32 s18, scc // (op 27418) 

s_cmp_eq_i32 s18, 0 // (op 27418) 
s_cselect_b32 s19, s22, s1 // (op 27418) 



v_lshl_add_u32 v13, s19, 8, v16 // (op 27418) 


s_ashr_i64 s[18:19], s[42:43], 7 // (op 27418) 
s_lshr_b32 s1, s18, 1 // (op 27418) 

v_lshl_add_u32 v16, s1, 3, v12 // (op 27418) 


v_mul_lo_u32 v12, v13, s34 // (op 27418) 
v_add_u32 v12, v12, v16 // (op 27418) 


s_mul_i32 s19, s34, s21 // (op 27420) most significant: low of low * high
s_mul_i32 s1, s35, s20 // (op 27420) most significant: low of high * low
s_mul_i32 s18, s34, s20 // (op 27420) least significant
s_mul_hi_u32 s22, s34, s20 // (op 27420) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 27420) most significant: sum
s_add_u32 s19, s19, s22 // (op 27420) most significant: sum


s_mov_b32 s64, s36 // (op 27420) reg expression
s_mov_b32 s65, s37 // (op 27420) reg expression
s_mov_b32 s66, s18 // (op 27420) reg expression
s_mov_b32 s67, 131072 // (op 27420) call()

s_mul_hi_i32 s1, s16, s10 // (op 27445) 
v_and_b32 v13, 63, v1 // (op 27445) 
s_add_i32 s18, s1, s16 // (op 27445) 

v_lshrrev_b32 v16, 4, v13 // (op 27445) 
s_ashr_i32 s1, s18, 31 // (op 27445) 
v_and_b32 v17, 3, v16 // (op 27445) 

s_cmp_eq_i32 s10, 0 // (op 27445) 
s_mov_b32 s19, scc // (op 27445) 
v_and_b32 v16, 15, v1 // (op 27445) 
s_cmp_eq_i32 s19, 0 // (op 27445) 
s_cselect_b32 s22, 0, -1 // (op 27445) 

v_lshl_add_u32 v18, v17, 4, v16 // (op 27445) 


s_and_b32 s19, s13, 31 // (op 27445) 
v_lshrrev_b32 v16, 6, v1 // (op 27445) 
s_lshl_b32 s23, 1, s19 // (op 27445) 
v_and_b32 v17, 1, v16 // (op 27445) 

s_add_i32 s24, s23, s22 // (op 27445) 


v_lshl_add_u32 v16, v17, 7, v18 // (op 27445) 


s_and_b32 s25, s1, s24 // (op 27445) 


v_lshrrev_b32 v17, 6, v13 // (op 27445) 

s_add_i32 s1, s18, s25 // (op 27445) 


v_lshlrev_b32 v13, 2, v17 // (op 27445) 

s_ashr_i32 s18, s1, s19 // (op 27445) 


s_lshl_b32 s1, s13, 24 // (op 27445) 
s_ashr_i32 s19, s1, 31 // (op 27445) 

s_xor_b32 s1, s18, s19 // (op 27445) 

s_sub_i32 s18, s1, s19 // (op 27445) 


s_mul_hi_i32 s1, s18, s14 // (op 27445) 
s_add_i32 s19, s1, s18 // (op 27445) 

s_ashr_i32 s1, s19, 31 // (op 27445) 
s_cmp_eq_i32 s14, 0 // (op 27445) 
s_mov_b32 s22, scc // (op 27445) 
s_cmp_eq_i32 s22, 0 // (op 27445) 
s_cselect_b32 s23, 0, -1 // (op 27445) 

s_and_b32 s22, s15, 31 // (op 27445) 
s_lshl_b32 s24, 1, s22 // (op 27445) 
s_add_i32 s25, s24, s23 // (op 27445) 


s_and_b32 s23, s1, s25 // (op 27445) 


s_add_i32 s1, s19, s23 // (op 27445) 


s_ashr_i32 s19, s1, s22 // (op 27445) 


s_lshl_b32 s1, s15, 24 // (op 27445) 
s_ashr_i32 s22, s1, 31 // (op 27445) 

s_xor_b32 s1, s19, s22 // (op 27445) 

s_sub_i32 s19, s1, s22 // (op 27445) 


s_add_u32 s22, s6, 256 // (op 27445) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27445) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27445) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27445) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27445) 

s_mul_i32 s1, s19, s22 // (op 27445) 

s_sub_i32 s19, s18, s1 // (op 27445) 


s_mul_hi_i32 s1, s16, s9 // (op 27445) 
s_add_i32 s18, s1, s16 // (op 27445) 

s_ashr_i32 s1, s18, 31 // (op 27445) 
s_cmp_eq_i32 s9, 0 // (op 27445) 
s_mov_b32 s24, scc // (op 27445) 
s_cmp_eq_i32 s24, 0 // (op 27445) 
s_cselect_b32 s25, 0, -1 // (op 27445) 

s_and_b32 s24, s45, 31 // (op 27445) 
s_lshl_b32 s68, 1, s24 // (op 27445) 
s_add_i32 s69, s68, s25 // (op 27445) 


s_and_b32 s25, s1, s69 // (op 27445) 


s_add_i32 s1, s18, s25 // (op 27445) 


s_ashr_i32 s18, s1, s24 // (op 27445) 


s_lshl_b32 s1, s45, 24 // (op 27445) 
s_ashr_i32 s24, s1, 31 // (op 27445) 

s_xor_b32 s1, s18, s24 // (op 27445) 

s_sub_i32 s18, s1, s24 // (op 27445) 


s_mul_i32 s1, s11, s22 // (op 27445) 

s_mul_i32 s22, s8, s1 // (op 27445) 

s_mul_i32 s1, s18, s22 // (op 27445) 

s_sub_u32 s22, s16, s1 // (op 27445) 

s_mul_hi_i32 s1, s22, s12 // (op 27445) 
s_add_i32 s23, s1, s22 // (op 27445) 


s_ashr_i32 s1, s23, 31 // (op 27445) 
s_cmp_eq_i32 s12, 0 // (op 27445) 
s_mov_b32 s22, scc // (op 27445) 
s_cmp_eq_i32 s22, 0 // (op 27445) 
s_cselect_b32 s24, 0, -1 // (op 27445) 

s_and_b32 s22, s44, 31 // (op 27445) 
s_lshl_b32 s25, 1, s22 // (op 27445) 
s_add_i32 s68, s25, s24 // (op 27445) 


s_and_b32 s24, s1, s68 // (op 27445) 


s_add_i32 s1, s23, s24 // (op 27445) 


s_ashr_i32 s23, s1, s22 // (op 27445) 


s_lshl_b32 s1, s44, 24 // (op 27445) 
s_ashr_i32 s22, s1, 31 // (op 27445) 

s_xor_b32 s1, s23, s22 // (op 27445) 

s_sub_i32 s23, s1, s22 // (op 27445) 


s_cmp_eq_i32 s18, 0 // (op 27445) 
s_mov_b32 s1, scc // (op 27445) 

s_cmp_eq_i32 s1, 0 // (op 27445) 
s_cselect_b32 s18, s23, s19 // (op 27445) 



v_lshl_add_u32 v17, s18, 8, v16 // (op 27445) 


v_mul_lo_u32 v16, v17, s38 // (op 27445) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 27445) 
s_lshr_b32 s1, s18, 1 // (op 27445) 

v_lshl_add_u32 v17, s1, 3, v13 // (op 27445) 


v_add_u32 v13, v17, v16 // (op 27445) 


s_mul_i32 s19, s38, s7 // (op 27447) most significant: low of low * high
s_mul_i32 s1, s39, s6 // (op 27447) most significant: low of high * low
s_mul_i32 s18, s38, s6 // (op 27447) least significant
s_mul_hi_u32 s22, s38, s6 // (op 27447) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 27447) most significant: sum
s_add_u32 s19, s19, s22 // (op 27447) most significant: sum


s_mov_b32 s68, s40 // (op 27447) reg expression
s_mov_b32 s69, s41 // (op 27447) reg expression
s_mov_b32 s70, s18 // (op 27447) reg expression
s_mov_b32 s71, 131072 // (op 27447) call()

s_ashr_i64 s[18:19], s[42:43], 7 // (op 2823) 
s_and_b32 s1, s18, 4294967294 // (op 2823) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 2822) 
s_mov_b32 s22, s18 // (op 2822) convert

s_cmp_lt_u32 s1, s22 // (op 2822) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_2822 // (op 2822) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_2822 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoopTail_2822:
s_barrier  // (op 27798) 
s_add_u32 s1, s1, 1 // (op 2824) 

s_lshl_b64 s[18:19], s[26:27], 5 // (op 26821) 
v_mov_b64 v[16:17], s[18:19] // (op 26821) convert


v_mov_b64 v[18:19], v[10:11] // (op 26821) 
v_mov_b64 v[20:21], v[18:19] // (op 26821) 
s_add_u32 s18, 16384, s47 // (op 26821) 
s_mov_b32 m0, s18 // (op 26821) reg expression
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 26821) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 26821) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 26821) most significant half

s_add_u32 m0, m0, 4096 // (op 26821) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 26821) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 26821) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 26821) most significant half

s_add_u32 m0, m0, 4096 // (op 26821) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 26821) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 26821) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 26821) most significant half

s_add_u32 m0, m0, 4096 // (op 26821) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 26821) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 26821) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 26821) most significant half

s_add_u32 m0, m0, 4096 // (op 26821) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 26821) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 26821) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 26821) most significant half

s_add_u32 m0, m0, 4096 // (op 26821) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 26821) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 26821) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 26821) most significant half

s_add_u32 m0, m0, 4096 // (op 26821) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 26821) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 26821) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 26821) most significant half

s_add_u32 m0, m0, 4096 // (op 26821) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 26821) Load value direct to lds




v_add_co_u32 v10, vcc, 128, v10 // (op 26858) least significant half
v_addc_co_u32 v11, vcc, 0, v11, vcc // (op 26858) most significant half
s_barrier  // (op 27814) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 26827) 
s_lshl_b64 s[22:23], s[18:19], 2 // (op 26827) 

v_lshrrev_b64 v[16:17], 3, s[22:23] // (op 26827) 


v_mov_b64 v[18:19], v[14:15] // (op 26827) 
v_mov_b64 v[20:21], v[18:19] // (op 26827) 
s_mov_b32 m0, s0 // (op 26827) reg expression
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 26827) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 26827) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 26827) most significant half

s_add_u32 m0, m0, 4096 // (op 26827) 
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 26827) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 26827) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 26827) most significant half

s_add_u32 m0, m0, 4096 // (op 26827) 
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 26827) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 26827) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 26827) most significant half

s_add_u32 m0, m0, 4096 // (op 26827) 
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 26827) Load value direct to lds



v_add_co_u32 v14, s[18:19], 64, v14 // (op 26990) least significant half
v_addc_co_u32 v15, s[18:19], 0, v15, s[18:19] // (op 26990) most significant half

s_waitcnt vmcnt(4)
s_barrier  // (op 27804) 



ds_read_b128 v[16:19], v3 offset:16384 // (op 164) Load local data 
ds_read_b128 v[20:23], v3 offset:16448 // (op 164) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 27820) 



ds_read_b128 v[24:27], v4 // (op 166) Load local data 


v_mov_b32 v28, v12 // (op 168) 
v_mov_b32 v29, v28 // (op 168) reg expression

buffer_load_dwordx2 v[30:31], v29, s[64:67], 0 offen // (op 168) Load value



v_mov_b32 v29, v13 // (op 170) 
v_mov_b32 v32, v29 // (op 170) reg expression

buffer_load_dwordx2 v[34:35], v32, s[68:71], 0 offen // (op 170) Load value

s_waitcnt vmcnt(1)
v_bfe_u32 v36, v30, 0, 8 // (op 25190) 



v_bfe_u32 v37, v30, 8, 8 // (op 25190) 
v_bfe_u32 v38, v30, 16, 8 // (op 25190) 
v_bfe_u32 v39, v30, 24, 8 // (op 25190) 
v_permlane16_swap_b32 v36, v37 // (op 25190) 
s_nop 0
v_permlane16_swap_b32 v38, v39 // (op 25190) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v36, v38 // (op 25190) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v37, v39 // (op 25190) 
ds_read_b128 v[40:43], v4 offset:1024 // (op 295) Load local data 




ds_read_b128 v[30:33], v4 offset:2048 // (op 431) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v44, v34, 0, 8 // (op 25282) 



v_bfe_u32 v45, v34, 8, 8 // (op 25282) 
v_bfe_u32 v46, v34, 16, 8 // (op 25282) 
v_bfe_u32 v47, v34, 24, 8 // (op 25282) 
v_permlane16_swap_b32 v44, v45 // (op 25282) 
ds_read_b128 v[48:51], v4 offset:3072 // (op 567) Load local data 
v_permlane16_swap_b32 v46, v47 // (op 25282) 



s_nop 1
v_permlane32_swap_b32 v44, v46 // (op 25282) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v45, v47 // (op 25282) 
ds_read_b128 v[52:55], v4 offset:4096 // (op 703) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:23], v[24:27], a[0:3], v36, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 171) 
s_lshl_b32 s18, s38, 6 // (op 708) 
v_add_u32 v34, v29, s18 // (op 708) 


buffer_load_dwordx2 v[56:57], v34, s[68:71], 0 offen // (op 708) Load value

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[16:23], v[40:43], a[4:7], v36, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 298) 




v_add_u32 v13, 4, v13 // (op 27435) 
ds_read_b128 v[58:61], v4 offset:5120 // (op 839) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[16:23], v[30:33], a[8:11], v36, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 434) 



ds_read_b128 v[62:65], v4 offset:6144 // (op 975) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[16:23], v[48:51], a[12:15], v36, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 570) 



s_waitcnt vmcnt(0)
v_bfe_u32 v66, v56, 0, 8 // (op 25296) 
v_bfe_u32 v67, v56, 8, 8 // (op 25296) 
v_bfe_u32 v68, v56, 16, 8 // (op 25296) 
v_bfe_u32 v69, v56, 24, 8 // (op 25296) 
v_permlane16_swap_b32 v66, v67 // (op 25296) 
s_nop 0
v_permlane16_swap_b32 v68, v69 // (op 25296) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v66, v68 // (op 25296) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v67, v69 // (op 25296) 
ds_read_b128 v[70:73], v4 offset:7168 // (op 1111) Load local data 




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[16:23], v[52:55], a[16:19], v36, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 706) 
ds_read_b128 v[74:77], v3 offset:18432 // (op 1421) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[16:23], v[58:61], a[20:23], v36, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 842) 
ds_read_b128 v[78:81], v3 offset:18496 // (op 1421) Load local data 



s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[16:23], v[62:65], a[24:27], v36, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 978) 
ds_read_b128 v[82:85], v3 offset:20480 // (op 2639) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[16:23], v[70:73], a[28:31], v36, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1114) 
ds_read_b128 v[86:89], v3 offset:20544 // (op 2639) Load local data 




s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:81], v[24:27], a[32:35], v37, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1425) 
ds_read_b128 v[16:19], v3 offset:22528 // (op 3857) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:81], v[30:33], a[40:43], v37, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1479) 
ds_read_b128 v[20:23], v3 offset:22592 // (op 3857) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:81], v[48:51], a[44:47], v37, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1506) 
ds_read_b128 v[90:93], v3 offset:24576 // (op 5075) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:81], v[40:43], a[36:39], v37, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1452) 
ds_read_b128 v[94:97], v3 offset:24640 // (op 5075) Load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:81], v[52:55], a[48:51], v37, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1533) 
s_lshl_b32 s18, s34, 6 // (op 5084) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:81], v[62:65], a[56:59], v37, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1587) 
v_add_u32 v29, v28, s18 // (op 5084) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:81], v[70:73], a[60:63], v37, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1614) 
buffer_load_dwordx2 v[34:35], v29, s[64:67], 0 offen // (op 5084) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:81], v[58:61], a[52:55], v37, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1560) 




v_add_u32 v12, 4, v12 // (op 27408) 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[82:89], v[24:27], a[64:67], v38, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2643) 
ds_read_b128 v[74:77], v3 offset:26624 // (op 6293) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[82:89], v[30:33], a[72:75], v38, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2697) 
ds_read_b128 v[78:81], v3 offset:26688 // (op 6293) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[82:89], v[48:51], a[76:79], v38, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2724) 
ds_read_b128 v[98:101], v3 offset:28672 // (op 7511) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[82:89], v[40:43], a[68:71], v38, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2670) 
s_waitcnt vmcnt(0)
v_bfe_u32 v106, v34, 0, 8 // (op 25204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[82:89], v[52:55], a[80:83], v38, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2751) 
v_bfe_u32 v107, v34, 8, 8 // (op 25204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[82:89], v[62:65], a[88:91], v38, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2805) 
v_bfe_u32 v108, v34, 16, 8 // (op 25204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[82:89], v[70:73], a[92:95], v38, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2832) 
v_bfe_u32 v109, v34, 24, 8 // (op 25204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[82:89], v[58:61], a[84:87], v38, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2778) 
v_permlane16_swap_b32 v106, v107 // (op 25204) 

v_permlane16_swap_b32 v108, v109 // (op 25204) 
s_nop 1
v_permlane32_swap_b32 v106, v108 // (op 25204) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v107, v109 // (op 25204) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[16:23], v[24:27], a[96:99], v39, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3861) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[16:23], v[30:33], a[104:107], v39, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3915) 
ds_read_b128 v[102:105], v3 offset:28736 // (op 7511) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[16:23], v[48:51], a[108:111], v39, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3942) 
ds_read_b128 v[82:85], v3 offset:30720 // (op 8729) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[16:23], v[40:43], a[100:103], v39, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3888) 
ds_read_b128 v[86:89], v3 offset:30784 // (op 8729) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[16:23], v[52:55], a[112:115], v39, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3969) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[16:23], v[62:65], a[120:123], v39, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4023) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[16:23], v[70:73], a[124:127], v39, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4050) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[16:23], v[58:61], a[116:119], v39, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3996) 


s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[90:97], v[40:43], a[132:135], v106, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5106) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[90:97], v[58:61], a[148:151], v106, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5214) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[90:97], v[24:27], a[128:131], v106, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5079) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[90:97], v[30:33], a[136:139], v106, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5133) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[90:97], v[48:51], a[140:143], v106, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5160) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[90:97], v[52:55], a[144:147], v106, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5187) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[90:97], v[62:65], a[152:155], v106, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5241) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:97], v[70:73], a[156:159], v106, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5268) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[74:81], v[40:43], a[164:167], v107, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6324) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[74:81], v[58:61], a[180:183], v107, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6432) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[74:81], v[24:27], a[160:163], v107, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6297) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[74:81], v[30:33], a[168:171], v107, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6351) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[74:81], v[48:51], a[172:175], v107, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6378) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[74:81], v[52:55], a[176:179], v107, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6405) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[74:81], v[62:65], a[184:187], v107, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6459) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[74:81], v[70:73], a[188:191], v107, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 6486) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[98:105], v[40:43], a[196:199], v108, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7542) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[98:105], v[58:61], a[212:215], v108, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7650) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[98:105], v[24:27], a[192:195], v108, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7515) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[98:105], v[30:33], a[200:203], v108, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7569) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[98:105], v[48:51], a[204:207], v108, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7596) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[98:105], v[52:55], a[208:211], v108, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7623) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[98:105], v[62:65], a[216:219], v108, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7677) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[98:105], v[70:73], a[220:223], v108, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 7704) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[82:89], v[40:43], a[228:231], v109, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 8760) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[82:89], v[58:61], a[244:247], v109, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 8868) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[82:89], v[24:27], a[224:227], v109, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 8733) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[82:89], v[30:33], a[232:235], v109, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 8787) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[82:89], v[48:51], a[236:239], v109, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 8814) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[82:89], v[52:55], a[240:243], v109, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 8841) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[82:89], v[62:65], a[248:251], v109, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 8895) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[82:89], v[70:73], a[252:255], v109, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 8922) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 2822) 
s_mov_b32 s22, s18 // (op 2822) convert

s_cmp_lt_u32 s1, s22 // (op 2822) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoopTail_2822 // (op 2822) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoopTail_2822 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_2822:




















s_barrier  // (op 27876) 

















v_cmp_eq_f32 vcc, 0.00000, v0 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalFalse_0_beta_116
v_accvgpr_read v10, a0 // (op 42) 
v_accvgpr_read v11, a1 // (op 42) 
v_accvgpr_read v12, a2 // (op 42) 
v_accvgpr_read v13, a3 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 
v_mul_f32 v16, v7, v12 // (op 42) 
v_mul_f32 v17, v7, v13 // (op 42) 

v_cvt_f16_f32 v10, v14 // (op 48) 
v_cvt_f16_f32 v11, v15 // (op 48) 
v_cvt_f16_f32 v12, v16 // (op 48) 
v_cvt_f16_f32 v13, v17 // (op 48) 


v_pack_b32_f16 v4, v10, v11 // (op 50) 
v_pack_b32_f16 v5, v12, v13 // (op 50) 

v_mov_b32 v3, v9 // (op 50) reg expression

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 50) Store value



v_accvgpr_read v10, a4 // (op 11114) 
v_accvgpr_read v11, a5 // (op 11114) 
v_accvgpr_read v12, a6 // (op 11114) 
v_accvgpr_read v13, a7 // (op 11114) 
v_mul_f32 v14, v7, v10 // (op 11114) 
v_mul_f32 v15, v7, v11 // (op 11114) 
v_mul_f32 v16, v7, v12 // (op 11114) 
v_mul_f32 v17, v7, v13 // (op 11114) 

v_cvt_f16_f32 v10, v14 // (op 11116) 
v_cvt_f16_f32 v11, v15 // (op 11116) 
v_cvt_f16_f32 v12, v16 // (op 11116) 
v_cvt_f16_f32 v13, v17 // (op 11116) 


v_pack_b32_f16 v4, v10, v11 // (op 11117) 
v_pack_b32_f16 v5, v12, v13 // (op 11117) 

s_lshl_b32 s0, s4, 5 // (op 11117) 
v_add_u32 v3, v9, s0 // (op 11117) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11117) Store value



v_accvgpr_read v10, a8 // (op 11128) 
v_accvgpr_read v11, a9 // (op 11128) 
v_accvgpr_read v12, a10 // (op 11128) 
v_accvgpr_read v13, a11 // (op 11128) 
v_mul_f32 v14, v7, v10 // (op 11128) 
v_mul_f32 v15, v7, v11 // (op 11128) 
v_mul_f32 v16, v7, v12 // (op 11128) 
v_mul_f32 v17, v7, v13 // (op 11128) 

v_cvt_f16_f32 v10, v14 // (op 11130) 
v_cvt_f16_f32 v11, v15 // (op 11130) 
v_cvt_f16_f32 v12, v16 // (op 11130) 
v_cvt_f16_f32 v13, v17 // (op 11130) 


v_pack_b32_f16 v4, v10, v11 // (op 11131) 
v_pack_b32_f16 v5, v12, v13 // (op 11131) 

s_lshl_b32 s0, s4, 6 // (op 11131) 
v_add_u32 v3, v9, s0 // (op 11131) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11131) Store value



v_accvgpr_read v10, a12 // (op 11142) 
v_accvgpr_read v11, a13 // (op 11142) 
v_accvgpr_read v12, a14 // (op 11142) 
v_accvgpr_read v13, a15 // (op 11142) 
v_mul_f32 v14, v7, v10 // (op 11142) 
v_mul_f32 v15, v7, v11 // (op 11142) 
v_mul_f32 v16, v7, v12 // (op 11142) 
v_mul_f32 v17, v7, v13 // (op 11142) 

v_cvt_f16_f32 v10, v14 // (op 11144) 
v_cvt_f16_f32 v11, v15 // (op 11144) 
v_cvt_f16_f32 v12, v16 // (op 11144) 
v_cvt_f16_f32 v13, v17 // (op 11144) 


v_pack_b32_f16 v4, v10, v11 // (op 11145) 
v_pack_b32_f16 v5, v12, v13 // (op 11145) 

s_lshl_b32 s0, s4, 5 // (op 11145) 
s_mul_i32 s1, 3, s0 // (op 11145) 

v_add_u32 v3, v9, s1 // (op 11145) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11145) Store value



v_accvgpr_read v10, a16 // (op 11156) 
v_accvgpr_read v11, a17 // (op 11156) 
v_accvgpr_read v12, a18 // (op 11156) 
v_accvgpr_read v13, a19 // (op 11156) 
v_mul_f32 v14, v7, v10 // (op 11156) 
v_mul_f32 v15, v7, v11 // (op 11156) 
v_mul_f32 v16, v7, v12 // (op 11156) 
v_mul_f32 v17, v7, v13 // (op 11156) 

v_cvt_f16_f32 v10, v14 // (op 11158) 
v_cvt_f16_f32 v11, v15 // (op 11158) 
v_cvt_f16_f32 v12, v16 // (op 11158) 
v_cvt_f16_f32 v13, v17 // (op 11158) 


v_pack_b32_f16 v4, v10, v11 // (op 11159) 
v_pack_b32_f16 v5, v12, v13 // (op 11159) 

s_lshl_b32 s0, s4, 7 // (op 11159) 
v_add_u32 v3, v9, s0 // (op 11159) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11159) Store value



v_accvgpr_read v10, a20 // (op 11170) 
v_accvgpr_read v11, a21 // (op 11170) 
v_accvgpr_read v12, a22 // (op 11170) 
v_accvgpr_read v13, a23 // (op 11170) 
v_mul_f32 v14, v7, v10 // (op 11170) 
v_mul_f32 v15, v7, v11 // (op 11170) 
v_mul_f32 v16, v7, v12 // (op 11170) 
v_mul_f32 v17, v7, v13 // (op 11170) 

v_cvt_f16_f32 v10, v14 // (op 11172) 
v_cvt_f16_f32 v11, v15 // (op 11172) 
v_cvt_f16_f32 v12, v16 // (op 11172) 
v_cvt_f16_f32 v13, v17 // (op 11172) 


v_pack_b32_f16 v4, v10, v11 // (op 11173) 
v_pack_b32_f16 v5, v12, v13 // (op 11173) 

s_lshl_b32 s0, s4, 5 // (op 11173) 
s_mul_i32 s1, 5, s0 // (op 11173) 

v_add_u32 v3, v9, s1 // (op 11173) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11173) Store value



v_accvgpr_read v10, a24 // (op 11184) 
v_accvgpr_read v11, a25 // (op 11184) 
v_accvgpr_read v12, a26 // (op 11184) 
v_accvgpr_read v13, a27 // (op 11184) 
v_mul_f32 v14, v7, v10 // (op 11184) 
v_mul_f32 v15, v7, v11 // (op 11184) 
v_mul_f32 v16, v7, v12 // (op 11184) 
v_mul_f32 v17, v7, v13 // (op 11184) 

v_cvt_f16_f32 v10, v14 // (op 11186) 
v_cvt_f16_f32 v11, v15 // (op 11186) 
v_cvt_f16_f32 v12, v16 // (op 11186) 
v_cvt_f16_f32 v13, v17 // (op 11186) 


v_pack_b32_f16 v4, v10, v11 // (op 11187) 
v_pack_b32_f16 v5, v12, v13 // (op 11187) 

s_lshl_b32 s0, s4, 5 // (op 11187) 
s_mul_i32 s1, 6, s0 // (op 11187) 

v_add_u32 v3, v9, s1 // (op 11187) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11187) Store value



v_accvgpr_read v10, a28 // (op 11198) 
v_accvgpr_read v11, a29 // (op 11198) 
v_accvgpr_read v12, a30 // (op 11198) 
v_accvgpr_read v13, a31 // (op 11198) 
v_mul_f32 v14, v7, v10 // (op 11198) 
v_mul_f32 v15, v7, v11 // (op 11198) 
v_mul_f32 v16, v7, v12 // (op 11198) 
v_mul_f32 v17, v7, v13 // (op 11198) 

v_cvt_f16_f32 v10, v14 // (op 11200) 
v_cvt_f16_f32 v11, v15 // (op 11200) 
v_cvt_f16_f32 v12, v16 // (op 11200) 
v_cvt_f16_f32 v13, v17 // (op 11200) 


v_pack_b32_f16 v4, v10, v11 // (op 11201) 
v_pack_b32_f16 v5, v12, v13 // (op 11201) 

s_lshl_b32 s0, s4, 5 // (op 11201) 
s_mul_i32 s1, 7, s0 // (op 11201) 

v_add_u32 v3, v9, s1 // (op 11201) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11201) Store value



v_accvgpr_read v10, a32 // (op 11254) 
v_accvgpr_read v11, a33 // (op 11254) 
v_accvgpr_read v12, a34 // (op 11254) 
v_accvgpr_read v13, a35 // (op 11254) 
v_mul_f32 v14, v7, v10 // (op 11254) 
v_mul_f32 v15, v7, v11 // (op 11254) 
v_mul_f32 v16, v7, v12 // (op 11254) 
v_mul_f32 v17, v7, v13 // (op 11254) 

v_cvt_f16_f32 v10, v14 // (op 11256) 
v_cvt_f16_f32 v11, v15 // (op 11256) 
v_cvt_f16_f32 v12, v16 // (op 11256) 
v_cvt_f16_f32 v13, v17 // (op 11256) 


v_pack_b32_f16 v4, v10, v11 // (op 11258) 
v_pack_b32_f16 v5, v12, v13 // (op 11258) 

v_add_u32 v3, 32, v9 // (op 11258) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11258) Store value



v_accvgpr_read v10, a36 // (op 11276) 
v_accvgpr_read v11, a37 // (op 11276) 
v_accvgpr_read v12, a38 // (op 11276) 
v_accvgpr_read v13, a39 // (op 11276) 
v_mul_f32 v14, v7, v10 // (op 11276) 
v_mul_f32 v15, v7, v11 // (op 11276) 
v_mul_f32 v16, v7, v12 // (op 11276) 
v_mul_f32 v17, v7, v13 // (op 11276) 

v_cvt_f16_f32 v10, v14 // (op 11278) 
v_cvt_f16_f32 v11, v15 // (op 11278) 
v_cvt_f16_f32 v12, v16 // (op 11278) 
v_cvt_f16_f32 v13, v17 // (op 11278) 


v_pack_b32_f16 v4, v10, v11 // (op 11260) 
v_pack_b32_f16 v5, v12, v13 // (op 11260) 

s_lshl_b32 s0, s4, 5 // (op 11260) 
s_add_u32 s0, s0, 32 // (op 11260) 
v_add_u32 v3, v9, s0 // (op 11260) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11260) Store value



v_accvgpr_read v10, a40 // (op 11282) 
v_accvgpr_read v11, a41 // (op 11282) 
v_accvgpr_read v12, a42 // (op 11282) 
v_accvgpr_read v13, a43 // (op 11282) 
v_mul_f32 v14, v7, v10 // (op 11282) 
v_mul_f32 v15, v7, v11 // (op 11282) 
v_mul_f32 v16, v7, v12 // (op 11282) 
v_mul_f32 v17, v7, v13 // (op 11282) 

v_cvt_f16_f32 v10, v14 // (op 11284) 
v_cvt_f16_f32 v11, v15 // (op 11284) 
v_cvt_f16_f32 v12, v16 // (op 11284) 
v_cvt_f16_f32 v13, v17 // (op 11284) 


v_pack_b32_f16 v4, v10, v11 // (op 11262) 
v_pack_b32_f16 v5, v12, v13 // (op 11262) 

s_lshl_b32 s0, s4, 6 // (op 11262) 
s_add_u32 s0, s0, 32 // (op 11262) 
v_add_u32 v3, v9, s0 // (op 11262) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11262) Store value



v_accvgpr_read v10, a44 // (op 11288) 
v_accvgpr_read v11, a45 // (op 11288) 
v_accvgpr_read v12, a46 // (op 11288) 
v_accvgpr_read v13, a47 // (op 11288) 
v_mul_f32 v14, v7, v10 // (op 11288) 
v_mul_f32 v15, v7, v11 // (op 11288) 
v_mul_f32 v16, v7, v12 // (op 11288) 
v_mul_f32 v17, v7, v13 // (op 11288) 

v_cvt_f16_f32 v10, v14 // (op 11290) 
v_cvt_f16_f32 v11, v15 // (op 11290) 
v_cvt_f16_f32 v12, v16 // (op 11290) 
v_cvt_f16_f32 v13, v17 // (op 11290) 


v_pack_b32_f16 v4, v10, v11 // (op 11264) 
v_pack_b32_f16 v5, v12, v13 // (op 11264) 

s_lshl_b32 s0, s4, 5 // (op 11264) 
s_mul_i32 s1, 3, s0 // (op 11264) 
s_add_u32 s1, s1, 32 // (op 11264) 

v_add_u32 v3, v9, s1 // (op 11264) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11264) Store value



v_accvgpr_read v10, a48 // (op 11294) 
v_accvgpr_read v11, a49 // (op 11294) 
v_accvgpr_read v12, a50 // (op 11294) 
v_accvgpr_read v13, a51 // (op 11294) 
v_mul_f32 v14, v7, v10 // (op 11294) 
v_mul_f32 v15, v7, v11 // (op 11294) 
v_mul_f32 v16, v7, v12 // (op 11294) 
v_mul_f32 v17, v7, v13 // (op 11294) 

v_cvt_f16_f32 v10, v14 // (op 11296) 
v_cvt_f16_f32 v11, v15 // (op 11296) 
v_cvt_f16_f32 v12, v16 // (op 11296) 
v_cvt_f16_f32 v13, v17 // (op 11296) 


v_pack_b32_f16 v4, v10, v11 // (op 11266) 
v_pack_b32_f16 v5, v12, v13 // (op 11266) 

s_lshl_b32 s0, s4, 7 // (op 11266) 
s_add_u32 s0, s0, 32 // (op 11266) 
v_add_u32 v3, v9, s0 // (op 11266) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11266) Store value



v_accvgpr_read v10, a52 // (op 11300) 
v_accvgpr_read v11, a53 // (op 11300) 
v_accvgpr_read v12, a54 // (op 11300) 
v_accvgpr_read v13, a55 // (op 11300) 
v_mul_f32 v14, v7, v10 // (op 11300) 
v_mul_f32 v15, v7, v11 // (op 11300) 
v_mul_f32 v16, v7, v12 // (op 11300) 
v_mul_f32 v17, v7, v13 // (op 11300) 

v_cvt_f16_f32 v10, v14 // (op 11302) 
v_cvt_f16_f32 v11, v15 // (op 11302) 
v_cvt_f16_f32 v12, v16 // (op 11302) 
v_cvt_f16_f32 v13, v17 // (op 11302) 


v_pack_b32_f16 v4, v10, v11 // (op 11268) 
v_pack_b32_f16 v5, v12, v13 // (op 11268) 

s_lshl_b32 s0, s4, 5 // (op 11268) 
s_mul_i32 s1, 5, s0 // (op 11268) 
s_add_u32 s1, s1, 32 // (op 11268) 

v_add_u32 v3, v9, s1 // (op 11268) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11268) Store value



v_accvgpr_read v10, a56 // (op 11306) 
v_accvgpr_read v11, a57 // (op 11306) 
v_accvgpr_read v12, a58 // (op 11306) 
v_accvgpr_read v13, a59 // (op 11306) 
v_mul_f32 v14, v7, v10 // (op 11306) 
v_mul_f32 v15, v7, v11 // (op 11306) 
v_mul_f32 v16, v7, v12 // (op 11306) 
v_mul_f32 v17, v7, v13 // (op 11306) 

v_cvt_f16_f32 v10, v14 // (op 11308) 
v_cvt_f16_f32 v11, v15 // (op 11308) 
v_cvt_f16_f32 v12, v16 // (op 11308) 
v_cvt_f16_f32 v13, v17 // (op 11308) 


v_pack_b32_f16 v4, v10, v11 // (op 11270) 
v_pack_b32_f16 v5, v12, v13 // (op 11270) 

s_lshl_b32 s0, s4, 5 // (op 11270) 
s_mul_i32 s1, 6, s0 // (op 11270) 
s_add_u32 s1, s1, 32 // (op 11270) 

v_add_u32 v3, v9, s1 // (op 11270) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11270) Store value



v_accvgpr_read v10, a60 // (op 11312) 
v_accvgpr_read v11, a61 // (op 11312) 
v_accvgpr_read v12, a62 // (op 11312) 
v_accvgpr_read v13, a63 // (op 11312) 
v_mul_f32 v14, v7, v10 // (op 11312) 
v_mul_f32 v15, v7, v11 // (op 11312) 
v_mul_f32 v16, v7, v12 // (op 11312) 
v_mul_f32 v17, v7, v13 // (op 11312) 

v_cvt_f16_f32 v10, v14 // (op 11314) 
v_cvt_f16_f32 v11, v15 // (op 11314) 
v_cvt_f16_f32 v12, v16 // (op 11314) 
v_cvt_f16_f32 v13, v17 // (op 11314) 


v_pack_b32_f16 v4, v10, v11 // (op 11272) 
v_pack_b32_f16 v5, v12, v13 // (op 11272) 

s_lshl_b32 s0, s4, 5 // (op 11272) 
s_mul_i32 s1, 7, s0 // (op 11272) 
s_add_u32 s1, s1, 32 // (op 11272) 

v_add_u32 v3, v9, s1 // (op 11272) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11272) Store value



v_accvgpr_read v10, a64 // (op 11402) 
v_accvgpr_read v11, a65 // (op 11402) 
v_accvgpr_read v12, a66 // (op 11402) 
v_accvgpr_read v13, a67 // (op 11402) 
v_mul_f32 v14, v7, v10 // (op 11402) 
v_mul_f32 v15, v7, v11 // (op 11402) 
v_mul_f32 v16, v7, v12 // (op 11402) 
v_mul_f32 v17, v7, v13 // (op 11402) 

v_cvt_f16_f32 v10, v14 // (op 11404) 
v_cvt_f16_f32 v11, v15 // (op 11404) 
v_cvt_f16_f32 v12, v16 // (op 11404) 
v_cvt_f16_f32 v13, v17 // (op 11404) 


v_pack_b32_f16 v4, v10, v11 // (op 11406) 
v_pack_b32_f16 v5, v12, v13 // (op 11406) 

v_add_u32 v3, 64, v9 // (op 11406) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11406) Store value



v_accvgpr_read v10, a68 // (op 11424) 
v_accvgpr_read v11, a69 // (op 11424) 
v_accvgpr_read v12, a70 // (op 11424) 
v_accvgpr_read v13, a71 // (op 11424) 
v_mul_f32 v14, v7, v10 // (op 11424) 
v_mul_f32 v15, v7, v11 // (op 11424) 
v_mul_f32 v16, v7, v12 // (op 11424) 
v_mul_f32 v17, v7, v13 // (op 11424) 

v_cvt_f16_f32 v10, v14 // (op 11426) 
v_cvt_f16_f32 v11, v15 // (op 11426) 
v_cvt_f16_f32 v12, v16 // (op 11426) 
v_cvt_f16_f32 v13, v17 // (op 11426) 


v_pack_b32_f16 v4, v10, v11 // (op 11408) 
v_pack_b32_f16 v5, v12, v13 // (op 11408) 

s_lshl_b32 s0, s4, 5 // (op 11408) 
s_add_u32 s0, s0, 64 // (op 11408) 
v_add_u32 v3, v9, s0 // (op 11408) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11408) Store value



v_accvgpr_read v10, a72 // (op 11430) 
v_accvgpr_read v11, a73 // (op 11430) 
v_accvgpr_read v12, a74 // (op 11430) 
v_accvgpr_read v13, a75 // (op 11430) 
v_mul_f32 v14, v7, v10 // (op 11430) 
v_mul_f32 v15, v7, v11 // (op 11430) 
v_mul_f32 v16, v7, v12 // (op 11430) 
v_mul_f32 v17, v7, v13 // (op 11430) 

v_cvt_f16_f32 v10, v14 // (op 11432) 
v_cvt_f16_f32 v11, v15 // (op 11432) 
v_cvt_f16_f32 v12, v16 // (op 11432) 
v_cvt_f16_f32 v13, v17 // (op 11432) 


v_pack_b32_f16 v4, v10, v11 // (op 11410) 
v_pack_b32_f16 v5, v12, v13 // (op 11410) 

s_lshl_b32 s0, s4, 6 // (op 11410) 
s_add_u32 s0, s0, 64 // (op 11410) 
v_add_u32 v3, v9, s0 // (op 11410) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11410) Store value



v_accvgpr_read v10, a76 // (op 11436) 
v_accvgpr_read v11, a77 // (op 11436) 
v_accvgpr_read v12, a78 // (op 11436) 
v_accvgpr_read v13, a79 // (op 11436) 
v_mul_f32 v14, v7, v10 // (op 11436) 
v_mul_f32 v15, v7, v11 // (op 11436) 
v_mul_f32 v16, v7, v12 // (op 11436) 
v_mul_f32 v17, v7, v13 // (op 11436) 

v_cvt_f16_f32 v10, v14 // (op 11438) 
v_cvt_f16_f32 v11, v15 // (op 11438) 
v_cvt_f16_f32 v12, v16 // (op 11438) 
v_cvt_f16_f32 v13, v17 // (op 11438) 


v_pack_b32_f16 v4, v10, v11 // (op 11412) 
v_pack_b32_f16 v5, v12, v13 // (op 11412) 

s_lshl_b32 s0, s4, 5 // (op 11412) 
s_mul_i32 s1, 3, s0 // (op 11412) 
s_add_u32 s1, s1, 64 // (op 11412) 

v_add_u32 v3, v9, s1 // (op 11412) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11412) Store value



v_accvgpr_read v10, a80 // (op 11442) 
v_accvgpr_read v11, a81 // (op 11442) 
v_accvgpr_read v12, a82 // (op 11442) 
v_accvgpr_read v13, a83 // (op 11442) 
v_mul_f32 v14, v7, v10 // (op 11442) 
v_mul_f32 v15, v7, v11 // (op 11442) 
v_mul_f32 v16, v7, v12 // (op 11442) 
v_mul_f32 v17, v7, v13 // (op 11442) 

v_cvt_f16_f32 v10, v14 // (op 11444) 
v_cvt_f16_f32 v11, v15 // (op 11444) 
v_cvt_f16_f32 v12, v16 // (op 11444) 
v_cvt_f16_f32 v13, v17 // (op 11444) 


v_pack_b32_f16 v4, v10, v11 // (op 11414) 
v_pack_b32_f16 v5, v12, v13 // (op 11414) 

s_lshl_b32 s0, s4, 7 // (op 11414) 
s_add_u32 s0, s0, 64 // (op 11414) 
v_add_u32 v3, v9, s0 // (op 11414) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11414) Store value



v_accvgpr_read v10, a84 // (op 11448) 
v_accvgpr_read v11, a85 // (op 11448) 
v_accvgpr_read v12, a86 // (op 11448) 
v_accvgpr_read v13, a87 // (op 11448) 
v_mul_f32 v14, v7, v10 // (op 11448) 
v_mul_f32 v15, v7, v11 // (op 11448) 
v_mul_f32 v16, v7, v12 // (op 11448) 
v_mul_f32 v17, v7, v13 // (op 11448) 

v_cvt_f16_f32 v10, v14 // (op 11450) 
v_cvt_f16_f32 v11, v15 // (op 11450) 
v_cvt_f16_f32 v12, v16 // (op 11450) 
v_cvt_f16_f32 v13, v17 // (op 11450) 


v_pack_b32_f16 v4, v10, v11 // (op 11416) 
v_pack_b32_f16 v5, v12, v13 // (op 11416) 

s_lshl_b32 s0, s4, 5 // (op 11416) 
s_mul_i32 s1, 5, s0 // (op 11416) 
s_add_u32 s1, s1, 64 // (op 11416) 

v_add_u32 v3, v9, s1 // (op 11416) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11416) Store value



v_accvgpr_read v10, a88 // (op 11454) 
v_accvgpr_read v11, a89 // (op 11454) 
v_accvgpr_read v12, a90 // (op 11454) 
v_accvgpr_read v13, a91 // (op 11454) 
v_mul_f32 v14, v7, v10 // (op 11454) 
v_mul_f32 v15, v7, v11 // (op 11454) 
v_mul_f32 v16, v7, v12 // (op 11454) 
v_mul_f32 v17, v7, v13 // (op 11454) 

v_cvt_f16_f32 v10, v14 // (op 11456) 
v_cvt_f16_f32 v11, v15 // (op 11456) 
v_cvt_f16_f32 v12, v16 // (op 11456) 
v_cvt_f16_f32 v13, v17 // (op 11456) 


v_pack_b32_f16 v4, v10, v11 // (op 11418) 
v_pack_b32_f16 v5, v12, v13 // (op 11418) 

s_lshl_b32 s0, s4, 5 // (op 11418) 
s_mul_i32 s1, 6, s0 // (op 11418) 
s_add_u32 s1, s1, 64 // (op 11418) 

v_add_u32 v3, v9, s1 // (op 11418) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11418) Store value



v_accvgpr_read v10, a92 // (op 11460) 
v_accvgpr_read v11, a93 // (op 11460) 
v_accvgpr_read v12, a94 // (op 11460) 
v_accvgpr_read v13, a95 // (op 11460) 
v_mul_f32 v14, v7, v10 // (op 11460) 
v_mul_f32 v15, v7, v11 // (op 11460) 
v_mul_f32 v16, v7, v12 // (op 11460) 
v_mul_f32 v17, v7, v13 // (op 11460) 

v_cvt_f16_f32 v10, v14 // (op 11462) 
v_cvt_f16_f32 v11, v15 // (op 11462) 
v_cvt_f16_f32 v12, v16 // (op 11462) 
v_cvt_f16_f32 v13, v17 // (op 11462) 


v_pack_b32_f16 v4, v10, v11 // (op 11420) 
v_pack_b32_f16 v5, v12, v13 // (op 11420) 

s_lshl_b32 s0, s4, 5 // (op 11420) 
s_mul_i32 s1, 7, s0 // (op 11420) 
s_add_u32 s1, s1, 64 // (op 11420) 

v_add_u32 v3, v9, s1 // (op 11420) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11420) Store value



v_accvgpr_read v10, a96 // (op 11550) 
v_accvgpr_read v11, a97 // (op 11550) 
v_accvgpr_read v12, a98 // (op 11550) 
v_accvgpr_read v13, a99 // (op 11550) 
v_mul_f32 v14, v7, v10 // (op 11550) 
v_mul_f32 v15, v7, v11 // (op 11550) 
v_mul_f32 v16, v7, v12 // (op 11550) 
v_mul_f32 v17, v7, v13 // (op 11550) 

v_cvt_f16_f32 v10, v14 // (op 11552) 
v_cvt_f16_f32 v11, v15 // (op 11552) 
v_cvt_f16_f32 v12, v16 // (op 11552) 
v_cvt_f16_f32 v13, v17 // (op 11552) 


v_pack_b32_f16 v4, v10, v11 // (op 11554) 
v_pack_b32_f16 v5, v12, v13 // (op 11554) 

v_add_u32 v3, 96, v9 // (op 11554) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11554) Store value



v_accvgpr_read v10, a100 // (op 11572) 
v_accvgpr_read v11, a101 // (op 11572) 
v_accvgpr_read v12, a102 // (op 11572) 
v_accvgpr_read v13, a103 // (op 11572) 
v_mul_f32 v14, v7, v10 // (op 11572) 
v_mul_f32 v15, v7, v11 // (op 11572) 
v_mul_f32 v16, v7, v12 // (op 11572) 
v_mul_f32 v17, v7, v13 // (op 11572) 

v_cvt_f16_f32 v10, v14 // (op 11574) 
v_cvt_f16_f32 v11, v15 // (op 11574) 
v_cvt_f16_f32 v12, v16 // (op 11574) 
v_cvt_f16_f32 v13, v17 // (op 11574) 


v_pack_b32_f16 v4, v10, v11 // (op 11556) 
v_pack_b32_f16 v5, v12, v13 // (op 11556) 

s_lshl_b32 s0, s4, 5 // (op 11556) 
s_add_u32 s0, s0, 96 // (op 11556) 
v_add_u32 v3, v9, s0 // (op 11556) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11556) Store value



v_accvgpr_read v10, a104 // (op 11578) 
v_accvgpr_read v11, a105 // (op 11578) 
v_accvgpr_read v12, a106 // (op 11578) 
v_accvgpr_read v13, a107 // (op 11578) 
v_mul_f32 v14, v7, v10 // (op 11578) 
v_mul_f32 v15, v7, v11 // (op 11578) 
v_mul_f32 v16, v7, v12 // (op 11578) 
v_mul_f32 v17, v7, v13 // (op 11578) 

v_cvt_f16_f32 v10, v14 // (op 11580) 
v_cvt_f16_f32 v11, v15 // (op 11580) 
v_cvt_f16_f32 v12, v16 // (op 11580) 
v_cvt_f16_f32 v13, v17 // (op 11580) 


v_pack_b32_f16 v4, v10, v11 // (op 11558) 
v_pack_b32_f16 v5, v12, v13 // (op 11558) 

s_lshl_b32 s0, s4, 6 // (op 11558) 
s_add_u32 s0, s0, 96 // (op 11558) 
v_add_u32 v3, v9, s0 // (op 11558) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11558) Store value



v_accvgpr_read v10, a108 // (op 11584) 
v_accvgpr_read v11, a109 // (op 11584) 
v_accvgpr_read v12, a110 // (op 11584) 
v_accvgpr_read v13, a111 // (op 11584) 
v_mul_f32 v14, v7, v10 // (op 11584) 
v_mul_f32 v15, v7, v11 // (op 11584) 
v_mul_f32 v16, v7, v12 // (op 11584) 
v_mul_f32 v17, v7, v13 // (op 11584) 

v_cvt_f16_f32 v10, v14 // (op 11586) 
v_cvt_f16_f32 v11, v15 // (op 11586) 
v_cvt_f16_f32 v12, v16 // (op 11586) 
v_cvt_f16_f32 v13, v17 // (op 11586) 


v_pack_b32_f16 v4, v10, v11 // (op 11560) 
v_pack_b32_f16 v5, v12, v13 // (op 11560) 

s_lshl_b32 s0, s4, 5 // (op 11560) 
s_mul_i32 s1, 3, s0 // (op 11560) 
s_add_u32 s1, s1, 96 // (op 11560) 

v_add_u32 v3, v9, s1 // (op 11560) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11560) Store value



v_accvgpr_read v10, a112 // (op 11590) 
v_accvgpr_read v11, a113 // (op 11590) 
v_accvgpr_read v12, a114 // (op 11590) 
v_accvgpr_read v13, a115 // (op 11590) 
v_mul_f32 v14, v7, v10 // (op 11590) 
v_mul_f32 v15, v7, v11 // (op 11590) 
v_mul_f32 v16, v7, v12 // (op 11590) 
v_mul_f32 v17, v7, v13 // (op 11590) 

v_cvt_f16_f32 v10, v14 // (op 11592) 
v_cvt_f16_f32 v11, v15 // (op 11592) 
v_cvt_f16_f32 v12, v16 // (op 11592) 
v_cvt_f16_f32 v13, v17 // (op 11592) 


v_pack_b32_f16 v4, v10, v11 // (op 11562) 
v_pack_b32_f16 v5, v12, v13 // (op 11562) 

s_lshl_b32 s0, s4, 7 // (op 11562) 
s_add_u32 s0, s0, 96 // (op 11562) 
v_add_u32 v3, v9, s0 // (op 11562) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11562) Store value



v_accvgpr_read v10, a116 // (op 11596) 
v_accvgpr_read v11, a117 // (op 11596) 
v_accvgpr_read v12, a118 // (op 11596) 
v_accvgpr_read v13, a119 // (op 11596) 
v_mul_f32 v14, v7, v10 // (op 11596) 
v_mul_f32 v15, v7, v11 // (op 11596) 
v_mul_f32 v16, v7, v12 // (op 11596) 
v_mul_f32 v17, v7, v13 // (op 11596) 

v_cvt_f16_f32 v10, v14 // (op 11598) 
v_cvt_f16_f32 v11, v15 // (op 11598) 
v_cvt_f16_f32 v12, v16 // (op 11598) 
v_cvt_f16_f32 v13, v17 // (op 11598) 


v_pack_b32_f16 v4, v10, v11 // (op 11564) 
v_pack_b32_f16 v5, v12, v13 // (op 11564) 

s_lshl_b32 s0, s4, 5 // (op 11564) 
s_mul_i32 s1, 5, s0 // (op 11564) 
s_add_u32 s1, s1, 96 // (op 11564) 

v_add_u32 v3, v9, s1 // (op 11564) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11564) Store value



v_accvgpr_read v10, a120 // (op 11602) 
v_accvgpr_read v11, a121 // (op 11602) 
v_accvgpr_read v12, a122 // (op 11602) 
v_accvgpr_read v13, a123 // (op 11602) 
v_mul_f32 v14, v7, v10 // (op 11602) 
v_mul_f32 v15, v7, v11 // (op 11602) 
v_mul_f32 v16, v7, v12 // (op 11602) 
v_mul_f32 v17, v7, v13 // (op 11602) 

v_cvt_f16_f32 v10, v14 // (op 11604) 
v_cvt_f16_f32 v11, v15 // (op 11604) 
v_cvt_f16_f32 v12, v16 // (op 11604) 
v_cvt_f16_f32 v13, v17 // (op 11604) 


v_pack_b32_f16 v4, v10, v11 // (op 11566) 
v_pack_b32_f16 v5, v12, v13 // (op 11566) 

s_lshl_b32 s0, s4, 5 // (op 11566) 
s_mul_i32 s1, 6, s0 // (op 11566) 
s_add_u32 s1, s1, 96 // (op 11566) 

v_add_u32 v3, v9, s1 // (op 11566) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11566) Store value



v_accvgpr_read v10, a124 // (op 11608) 
v_accvgpr_read v11, a125 // (op 11608) 
v_accvgpr_read v12, a126 // (op 11608) 
v_accvgpr_read v13, a127 // (op 11608) 
v_mul_f32 v14, v7, v10 // (op 11608) 
v_mul_f32 v15, v7, v11 // (op 11608) 
v_mul_f32 v16, v7, v12 // (op 11608) 
v_mul_f32 v17, v7, v13 // (op 11608) 

v_cvt_f16_f32 v10, v14 // (op 11610) 
v_cvt_f16_f32 v11, v15 // (op 11610) 
v_cvt_f16_f32 v12, v16 // (op 11610) 
v_cvt_f16_f32 v13, v17 // (op 11610) 


v_pack_b32_f16 v4, v10, v11 // (op 11568) 
v_pack_b32_f16 v5, v12, v13 // (op 11568) 

s_lshl_b32 s0, s4, 5 // (op 11568) 
s_mul_i32 s1, 7, s0 // (op 11568) 
s_add_u32 s1, s1, 96 // (op 11568) 

v_add_u32 v3, v9, s1 // (op 11568) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11568) Store value



v_accvgpr_read v10, a128 // (op 11698) 
v_accvgpr_read v11, a129 // (op 11698) 
v_accvgpr_read v12, a130 // (op 11698) 
v_accvgpr_read v13, a131 // (op 11698) 
v_mul_f32 v14, v7, v10 // (op 11698) 
v_mul_f32 v15, v7, v11 // (op 11698) 
v_mul_f32 v16, v7, v12 // (op 11698) 
v_mul_f32 v17, v7, v13 // (op 11698) 

v_cvt_f16_f32 v10, v14 // (op 11700) 
v_cvt_f16_f32 v11, v15 // (op 11700) 
v_cvt_f16_f32 v12, v16 // (op 11700) 
v_cvt_f16_f32 v13, v17 // (op 11700) 


v_pack_b32_f16 v4, v10, v11 // (op 11702) 
v_pack_b32_f16 v5, v12, v13 // (op 11702) 

v_add_u32 v3, 128, v9 // (op 11702) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11702) Store value



v_accvgpr_read v10, a132 // (op 11720) 
v_accvgpr_read v11, a133 // (op 11720) 
v_accvgpr_read v12, a134 // (op 11720) 
v_accvgpr_read v13, a135 // (op 11720) 
v_mul_f32 v14, v7, v10 // (op 11720) 
v_mul_f32 v15, v7, v11 // (op 11720) 
v_mul_f32 v16, v7, v12 // (op 11720) 
v_mul_f32 v17, v7, v13 // (op 11720) 

v_cvt_f16_f32 v10, v14 // (op 11722) 
v_cvt_f16_f32 v11, v15 // (op 11722) 
v_cvt_f16_f32 v12, v16 // (op 11722) 
v_cvt_f16_f32 v13, v17 // (op 11722) 


v_pack_b32_f16 v4, v10, v11 // (op 11704) 
v_pack_b32_f16 v5, v12, v13 // (op 11704) 

s_lshl_b32 s0, s4, 5 // (op 11704) 
s_add_u32 s0, s0, 128 // (op 11704) 
v_add_u32 v3, v9, s0 // (op 11704) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11704) Store value



v_accvgpr_read v10, a136 // (op 11726) 
v_accvgpr_read v11, a137 // (op 11726) 
v_accvgpr_read v12, a138 // (op 11726) 
v_accvgpr_read v13, a139 // (op 11726) 
v_mul_f32 v14, v7, v10 // (op 11726) 
v_mul_f32 v15, v7, v11 // (op 11726) 
v_mul_f32 v16, v7, v12 // (op 11726) 
v_mul_f32 v17, v7, v13 // (op 11726) 

v_cvt_f16_f32 v10, v14 // (op 11728) 
v_cvt_f16_f32 v11, v15 // (op 11728) 
v_cvt_f16_f32 v12, v16 // (op 11728) 
v_cvt_f16_f32 v13, v17 // (op 11728) 


v_pack_b32_f16 v4, v10, v11 // (op 11706) 
v_pack_b32_f16 v5, v12, v13 // (op 11706) 

s_lshl_b32 s0, s4, 6 // (op 11706) 
s_add_u32 s0, s0, 128 // (op 11706) 
v_add_u32 v3, v9, s0 // (op 11706) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11706) Store value



v_accvgpr_read v10, a140 // (op 11732) 
v_accvgpr_read v11, a141 // (op 11732) 
v_accvgpr_read v12, a142 // (op 11732) 
v_accvgpr_read v13, a143 // (op 11732) 
v_mul_f32 v14, v7, v10 // (op 11732) 
v_mul_f32 v15, v7, v11 // (op 11732) 
v_mul_f32 v16, v7, v12 // (op 11732) 
v_mul_f32 v17, v7, v13 // (op 11732) 

v_cvt_f16_f32 v10, v14 // (op 11734) 
v_cvt_f16_f32 v11, v15 // (op 11734) 
v_cvt_f16_f32 v12, v16 // (op 11734) 
v_cvt_f16_f32 v13, v17 // (op 11734) 


v_pack_b32_f16 v4, v10, v11 // (op 11708) 
v_pack_b32_f16 v5, v12, v13 // (op 11708) 

s_lshl_b32 s0, s4, 5 // (op 11708) 
s_mul_i32 s1, 3, s0 // (op 11708) 
s_add_u32 s1, s1, 128 // (op 11708) 

v_add_u32 v3, v9, s1 // (op 11708) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11708) Store value



v_accvgpr_read v10, a144 // (op 11738) 
v_accvgpr_read v11, a145 // (op 11738) 
v_accvgpr_read v12, a146 // (op 11738) 
v_accvgpr_read v13, a147 // (op 11738) 
v_mul_f32 v14, v7, v10 // (op 11738) 
v_mul_f32 v15, v7, v11 // (op 11738) 
v_mul_f32 v16, v7, v12 // (op 11738) 
v_mul_f32 v17, v7, v13 // (op 11738) 

v_cvt_f16_f32 v10, v14 // (op 11740) 
v_cvt_f16_f32 v11, v15 // (op 11740) 
v_cvt_f16_f32 v12, v16 // (op 11740) 
v_cvt_f16_f32 v13, v17 // (op 11740) 


v_pack_b32_f16 v4, v10, v11 // (op 11710) 
v_pack_b32_f16 v5, v12, v13 // (op 11710) 

s_lshl_b32 s0, s4, 7 // (op 11710) 
s_add_u32 s0, s0, 128 // (op 11710) 
v_add_u32 v3, v9, s0 // (op 11710) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11710) Store value



v_accvgpr_read v10, a148 // (op 11744) 
v_accvgpr_read v11, a149 // (op 11744) 
v_accvgpr_read v12, a150 // (op 11744) 
v_accvgpr_read v13, a151 // (op 11744) 
v_mul_f32 v14, v7, v10 // (op 11744) 
v_mul_f32 v15, v7, v11 // (op 11744) 
v_mul_f32 v16, v7, v12 // (op 11744) 
v_mul_f32 v17, v7, v13 // (op 11744) 

v_cvt_f16_f32 v10, v14 // (op 11746) 
v_cvt_f16_f32 v11, v15 // (op 11746) 
v_cvt_f16_f32 v12, v16 // (op 11746) 
v_cvt_f16_f32 v13, v17 // (op 11746) 


v_pack_b32_f16 v4, v10, v11 // (op 11712) 
v_pack_b32_f16 v5, v12, v13 // (op 11712) 

s_lshl_b32 s0, s4, 5 // (op 11712) 
s_mul_i32 s1, 5, s0 // (op 11712) 
s_add_u32 s1, s1, 128 // (op 11712) 

v_add_u32 v3, v9, s1 // (op 11712) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11712) Store value



v_accvgpr_read v10, a152 // (op 11750) 
v_accvgpr_read v11, a153 // (op 11750) 
v_accvgpr_read v12, a154 // (op 11750) 
v_accvgpr_read v13, a155 // (op 11750) 
v_mul_f32 v14, v7, v10 // (op 11750) 
v_mul_f32 v15, v7, v11 // (op 11750) 
v_mul_f32 v16, v7, v12 // (op 11750) 
v_mul_f32 v17, v7, v13 // (op 11750) 

v_cvt_f16_f32 v10, v14 // (op 11752) 
v_cvt_f16_f32 v11, v15 // (op 11752) 
v_cvt_f16_f32 v12, v16 // (op 11752) 
v_cvt_f16_f32 v13, v17 // (op 11752) 


v_pack_b32_f16 v4, v10, v11 // (op 11714) 
v_pack_b32_f16 v5, v12, v13 // (op 11714) 

s_lshl_b32 s0, s4, 5 // (op 11714) 
s_mul_i32 s1, 6, s0 // (op 11714) 
s_add_u32 s1, s1, 128 // (op 11714) 

v_add_u32 v3, v9, s1 // (op 11714) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11714) Store value



v_accvgpr_read v10, a156 // (op 11756) 
v_accvgpr_read v11, a157 // (op 11756) 
v_accvgpr_read v12, a158 // (op 11756) 
v_accvgpr_read v13, a159 // (op 11756) 
v_mul_f32 v14, v7, v10 // (op 11756) 
v_mul_f32 v15, v7, v11 // (op 11756) 
v_mul_f32 v16, v7, v12 // (op 11756) 
v_mul_f32 v17, v7, v13 // (op 11756) 

v_cvt_f16_f32 v10, v14 // (op 11758) 
v_cvt_f16_f32 v11, v15 // (op 11758) 
v_cvt_f16_f32 v12, v16 // (op 11758) 
v_cvt_f16_f32 v13, v17 // (op 11758) 


v_pack_b32_f16 v4, v10, v11 // (op 11716) 
v_pack_b32_f16 v5, v12, v13 // (op 11716) 

s_lshl_b32 s0, s4, 5 // (op 11716) 
s_mul_i32 s1, 7, s0 // (op 11716) 
s_add_u32 s1, s1, 128 // (op 11716) 

v_add_u32 v3, v9, s1 // (op 11716) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11716) Store value



v_accvgpr_read v10, a160 // (op 11846) 
v_accvgpr_read v11, a161 // (op 11846) 
v_accvgpr_read v12, a162 // (op 11846) 
v_accvgpr_read v13, a163 // (op 11846) 
v_mul_f32 v14, v7, v10 // (op 11846) 
v_mul_f32 v15, v7, v11 // (op 11846) 
v_mul_f32 v16, v7, v12 // (op 11846) 
v_mul_f32 v17, v7, v13 // (op 11846) 

v_cvt_f16_f32 v10, v14 // (op 11848) 
v_cvt_f16_f32 v11, v15 // (op 11848) 
v_cvt_f16_f32 v12, v16 // (op 11848) 
v_cvt_f16_f32 v13, v17 // (op 11848) 


v_pack_b32_f16 v4, v10, v11 // (op 11850) 
v_pack_b32_f16 v5, v12, v13 // (op 11850) 

v_add_u32 v3, 160, v9 // (op 11850) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11850) Store value



v_accvgpr_read v10, a164 // (op 11868) 
v_accvgpr_read v11, a165 // (op 11868) 
v_accvgpr_read v12, a166 // (op 11868) 
v_accvgpr_read v13, a167 // (op 11868) 
v_mul_f32 v14, v7, v10 // (op 11868) 
v_mul_f32 v15, v7, v11 // (op 11868) 
v_mul_f32 v16, v7, v12 // (op 11868) 
v_mul_f32 v17, v7, v13 // (op 11868) 

v_cvt_f16_f32 v10, v14 // (op 11870) 
v_cvt_f16_f32 v11, v15 // (op 11870) 
v_cvt_f16_f32 v12, v16 // (op 11870) 
v_cvt_f16_f32 v13, v17 // (op 11870) 


v_pack_b32_f16 v4, v10, v11 // (op 11852) 
v_pack_b32_f16 v5, v12, v13 // (op 11852) 

s_lshl_b32 s0, s4, 5 // (op 11852) 
s_add_u32 s0, s0, 160 // (op 11852) 
v_add_u32 v3, v9, s0 // (op 11852) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11852) Store value



v_accvgpr_read v10, a168 // (op 11874) 
v_accvgpr_read v11, a169 // (op 11874) 
v_accvgpr_read v12, a170 // (op 11874) 
v_accvgpr_read v13, a171 // (op 11874) 
v_mul_f32 v14, v7, v10 // (op 11874) 
v_mul_f32 v15, v7, v11 // (op 11874) 
v_mul_f32 v16, v7, v12 // (op 11874) 
v_mul_f32 v17, v7, v13 // (op 11874) 

v_cvt_f16_f32 v10, v14 // (op 11876) 
v_cvt_f16_f32 v11, v15 // (op 11876) 
v_cvt_f16_f32 v12, v16 // (op 11876) 
v_cvt_f16_f32 v13, v17 // (op 11876) 


v_pack_b32_f16 v4, v10, v11 // (op 11854) 
v_pack_b32_f16 v5, v12, v13 // (op 11854) 

s_lshl_b32 s0, s4, 6 // (op 11854) 
s_add_u32 s0, s0, 160 // (op 11854) 
v_add_u32 v3, v9, s0 // (op 11854) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11854) Store value



v_accvgpr_read v10, a172 // (op 11880) 
v_accvgpr_read v11, a173 // (op 11880) 
v_accvgpr_read v12, a174 // (op 11880) 
v_accvgpr_read v13, a175 // (op 11880) 
v_mul_f32 v14, v7, v10 // (op 11880) 
v_mul_f32 v15, v7, v11 // (op 11880) 
v_mul_f32 v16, v7, v12 // (op 11880) 
v_mul_f32 v17, v7, v13 // (op 11880) 

v_cvt_f16_f32 v10, v14 // (op 11882) 
v_cvt_f16_f32 v11, v15 // (op 11882) 
v_cvt_f16_f32 v12, v16 // (op 11882) 
v_cvt_f16_f32 v13, v17 // (op 11882) 


v_pack_b32_f16 v4, v10, v11 // (op 11856) 
v_pack_b32_f16 v5, v12, v13 // (op 11856) 

s_lshl_b32 s0, s4, 5 // (op 11856) 
s_mul_i32 s1, 3, s0 // (op 11856) 
s_add_u32 s1, s1, 160 // (op 11856) 

v_add_u32 v3, v9, s1 // (op 11856) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11856) Store value



v_accvgpr_read v10, a176 // (op 11886) 
v_accvgpr_read v11, a177 // (op 11886) 
v_accvgpr_read v12, a178 // (op 11886) 
v_accvgpr_read v13, a179 // (op 11886) 
v_mul_f32 v14, v7, v10 // (op 11886) 
v_mul_f32 v15, v7, v11 // (op 11886) 
v_mul_f32 v16, v7, v12 // (op 11886) 
v_mul_f32 v17, v7, v13 // (op 11886) 

v_cvt_f16_f32 v10, v14 // (op 11888) 
v_cvt_f16_f32 v11, v15 // (op 11888) 
v_cvt_f16_f32 v12, v16 // (op 11888) 
v_cvt_f16_f32 v13, v17 // (op 11888) 


v_pack_b32_f16 v4, v10, v11 // (op 11858) 
v_pack_b32_f16 v5, v12, v13 // (op 11858) 

s_lshl_b32 s0, s4, 7 // (op 11858) 
s_add_u32 s0, s0, 160 // (op 11858) 
v_add_u32 v3, v9, s0 // (op 11858) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11858) Store value



v_accvgpr_read v10, a180 // (op 11892) 
v_accvgpr_read v11, a181 // (op 11892) 
v_accvgpr_read v12, a182 // (op 11892) 
v_accvgpr_read v13, a183 // (op 11892) 
v_mul_f32 v14, v7, v10 // (op 11892) 
v_mul_f32 v15, v7, v11 // (op 11892) 
v_mul_f32 v16, v7, v12 // (op 11892) 
v_mul_f32 v17, v7, v13 // (op 11892) 

v_cvt_f16_f32 v10, v14 // (op 11894) 
v_cvt_f16_f32 v11, v15 // (op 11894) 
v_cvt_f16_f32 v12, v16 // (op 11894) 
v_cvt_f16_f32 v13, v17 // (op 11894) 


v_pack_b32_f16 v4, v10, v11 // (op 11860) 
v_pack_b32_f16 v5, v12, v13 // (op 11860) 

s_lshl_b32 s0, s4, 5 // (op 11860) 
s_mul_i32 s1, 5, s0 // (op 11860) 
s_add_u32 s1, s1, 160 // (op 11860) 

v_add_u32 v3, v9, s1 // (op 11860) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11860) Store value



v_accvgpr_read v10, a184 // (op 11898) 
v_accvgpr_read v11, a185 // (op 11898) 
v_accvgpr_read v12, a186 // (op 11898) 
v_accvgpr_read v13, a187 // (op 11898) 
v_mul_f32 v14, v7, v10 // (op 11898) 
v_mul_f32 v15, v7, v11 // (op 11898) 
v_mul_f32 v16, v7, v12 // (op 11898) 
v_mul_f32 v17, v7, v13 // (op 11898) 

v_cvt_f16_f32 v10, v14 // (op 11900) 
v_cvt_f16_f32 v11, v15 // (op 11900) 
v_cvt_f16_f32 v12, v16 // (op 11900) 
v_cvt_f16_f32 v13, v17 // (op 11900) 


v_pack_b32_f16 v4, v10, v11 // (op 11862) 
v_pack_b32_f16 v5, v12, v13 // (op 11862) 

s_lshl_b32 s0, s4, 5 // (op 11862) 
s_mul_i32 s1, 6, s0 // (op 11862) 
s_add_u32 s1, s1, 160 // (op 11862) 

v_add_u32 v3, v9, s1 // (op 11862) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11862) Store value



v_accvgpr_read v10, a188 // (op 11904) 
v_accvgpr_read v11, a189 // (op 11904) 
v_accvgpr_read v12, a190 // (op 11904) 
v_accvgpr_read v13, a191 // (op 11904) 
v_mul_f32 v14, v7, v10 // (op 11904) 
v_mul_f32 v15, v7, v11 // (op 11904) 
v_mul_f32 v16, v7, v12 // (op 11904) 
v_mul_f32 v17, v7, v13 // (op 11904) 

v_cvt_f16_f32 v10, v14 // (op 11906) 
v_cvt_f16_f32 v11, v15 // (op 11906) 
v_cvt_f16_f32 v12, v16 // (op 11906) 
v_cvt_f16_f32 v13, v17 // (op 11906) 


v_pack_b32_f16 v4, v10, v11 // (op 11864) 
v_pack_b32_f16 v5, v12, v13 // (op 11864) 

s_lshl_b32 s0, s4, 5 // (op 11864) 
s_mul_i32 s1, 7, s0 // (op 11864) 
s_add_u32 s1, s1, 160 // (op 11864) 

v_add_u32 v3, v9, s1 // (op 11864) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11864) Store value



v_accvgpr_read v10, a192 // (op 11994) 
v_accvgpr_read v11, a193 // (op 11994) 
v_accvgpr_read v12, a194 // (op 11994) 
v_accvgpr_read v13, a195 // (op 11994) 
v_mul_f32 v14, v7, v10 // (op 11994) 
v_mul_f32 v15, v7, v11 // (op 11994) 
v_mul_f32 v16, v7, v12 // (op 11994) 
v_mul_f32 v17, v7, v13 // (op 11994) 

v_cvt_f16_f32 v10, v14 // (op 11996) 
v_cvt_f16_f32 v11, v15 // (op 11996) 
v_cvt_f16_f32 v12, v16 // (op 11996) 
v_cvt_f16_f32 v13, v17 // (op 11996) 


v_pack_b32_f16 v4, v10, v11 // (op 11998) 
v_pack_b32_f16 v5, v12, v13 // (op 11998) 

v_add_u32 v3, 192, v9 // (op 11998) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 11998) Store value



v_accvgpr_read v10, a196 // (op 12016) 
v_accvgpr_read v11, a197 // (op 12016) 
v_accvgpr_read v12, a198 // (op 12016) 
v_accvgpr_read v13, a199 // (op 12016) 
v_mul_f32 v14, v7, v10 // (op 12016) 
v_mul_f32 v15, v7, v11 // (op 12016) 
v_mul_f32 v16, v7, v12 // (op 12016) 
v_mul_f32 v17, v7, v13 // (op 12016) 

v_cvt_f16_f32 v10, v14 // (op 12018) 
v_cvt_f16_f32 v11, v15 // (op 12018) 
v_cvt_f16_f32 v12, v16 // (op 12018) 
v_cvt_f16_f32 v13, v17 // (op 12018) 


v_pack_b32_f16 v4, v10, v11 // (op 12000) 
v_pack_b32_f16 v5, v12, v13 // (op 12000) 

s_lshl_b32 s0, s4, 5 // (op 12000) 
s_add_u32 s0, s0, 192 // (op 12000) 
v_add_u32 v3, v9, s0 // (op 12000) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 12000) Store value



v_accvgpr_read v10, a200 // (op 12022) 
v_accvgpr_read v11, a201 // (op 12022) 
v_accvgpr_read v12, a202 // (op 12022) 
v_accvgpr_read v13, a203 // (op 12022) 
v_mul_f32 v14, v7, v10 // (op 12022) 
v_mul_f32 v15, v7, v11 // (op 12022) 
v_mul_f32 v16, v7, v12 // (op 12022) 
v_mul_f32 v17, v7, v13 // (op 12022) 

v_cvt_f16_f32 v10, v14 // (op 12024) 
v_cvt_f16_f32 v11, v15 // (op 12024) 
v_cvt_f16_f32 v12, v16 // (op 12024) 
v_cvt_f16_f32 v13, v17 // (op 12024) 


v_pack_b32_f16 v4, v10, v11 // (op 12002) 
v_pack_b32_f16 v5, v12, v13 // (op 12002) 

s_lshl_b32 s0, s4, 6 // (op 12002) 
s_add_u32 s0, s0, 192 // (op 12002) 
v_add_u32 v3, v9, s0 // (op 12002) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 12002) Store value



v_accvgpr_read v10, a204 // (op 12028) 
v_accvgpr_read v11, a205 // (op 12028) 
v_accvgpr_read v12, a206 // (op 12028) 
v_accvgpr_read v13, a207 // (op 12028) 
v_mul_f32 v14, v7, v10 // (op 12028) 
v_mul_f32 v15, v7, v11 // (op 12028) 
v_mul_f32 v16, v7, v12 // (op 12028) 
v_mul_f32 v17, v7, v13 // (op 12028) 

v_cvt_f16_f32 v10, v14 // (op 12030) 
v_cvt_f16_f32 v11, v15 // (op 12030) 
v_cvt_f16_f32 v12, v16 // (op 12030) 
v_cvt_f16_f32 v13, v17 // (op 12030) 


v_pack_b32_f16 v4, v10, v11 // (op 12004) 
v_pack_b32_f16 v5, v12, v13 // (op 12004) 

s_lshl_b32 s0, s4, 5 // (op 12004) 
s_mul_i32 s1, 3, s0 // (op 12004) 
s_add_u32 s1, s1, 192 // (op 12004) 

v_add_u32 v3, v9, s1 // (op 12004) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 12004) Store value



v_accvgpr_read v10, a208 // (op 12034) 
v_accvgpr_read v11, a209 // (op 12034) 
v_accvgpr_read v12, a210 // (op 12034) 
v_accvgpr_read v13, a211 // (op 12034) 
v_mul_f32 v14, v7, v10 // (op 12034) 
v_mul_f32 v15, v7, v11 // (op 12034) 
v_mul_f32 v16, v7, v12 // (op 12034) 
v_mul_f32 v17, v7, v13 // (op 12034) 

v_cvt_f16_f32 v10, v14 // (op 12036) 
v_cvt_f16_f32 v11, v15 // (op 12036) 
v_cvt_f16_f32 v12, v16 // (op 12036) 
v_cvt_f16_f32 v13, v17 // (op 12036) 


v_pack_b32_f16 v4, v10, v11 // (op 12006) 
v_pack_b32_f16 v5, v12, v13 // (op 12006) 

s_lshl_b32 s0, s4, 7 // (op 12006) 
s_add_u32 s0, s0, 192 // (op 12006) 
v_add_u32 v3, v9, s0 // (op 12006) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 12006) Store value



v_accvgpr_read v10, a212 // (op 12040) 
v_accvgpr_read v11, a213 // (op 12040) 
v_accvgpr_read v12, a214 // (op 12040) 
v_accvgpr_read v13, a215 // (op 12040) 
v_mul_f32 v14, v7, v10 // (op 12040) 
v_mul_f32 v15, v7, v11 // (op 12040) 
v_mul_f32 v16, v7, v12 // (op 12040) 
v_mul_f32 v17, v7, v13 // (op 12040) 

v_cvt_f16_f32 v10, v14 // (op 12042) 
v_cvt_f16_f32 v11, v15 // (op 12042) 
v_cvt_f16_f32 v12, v16 // (op 12042) 
v_cvt_f16_f32 v13, v17 // (op 12042) 


v_pack_b32_f16 v4, v10, v11 // (op 12008) 
v_pack_b32_f16 v5, v12, v13 // (op 12008) 

s_lshl_b32 s0, s4, 5 // (op 12008) 
s_mul_i32 s1, 5, s0 // (op 12008) 
s_add_u32 s1, s1, 192 // (op 12008) 

v_add_u32 v3, v9, s1 // (op 12008) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 12008) Store value



v_accvgpr_read v10, a216 // (op 12046) 
v_accvgpr_read v11, a217 // (op 12046) 
v_accvgpr_read v12, a218 // (op 12046) 
v_accvgpr_read v13, a219 // (op 12046) 
v_mul_f32 v14, v7, v10 // (op 12046) 
v_mul_f32 v15, v7, v11 // (op 12046) 
v_mul_f32 v16, v7, v12 // (op 12046) 
v_mul_f32 v17, v7, v13 // (op 12046) 

v_cvt_f16_f32 v10, v14 // (op 12048) 
v_cvt_f16_f32 v11, v15 // (op 12048) 
v_cvt_f16_f32 v12, v16 // (op 12048) 
v_cvt_f16_f32 v13, v17 // (op 12048) 


v_pack_b32_f16 v4, v10, v11 // (op 12010) 
v_pack_b32_f16 v5, v12, v13 // (op 12010) 

s_lshl_b32 s0, s4, 5 // (op 12010) 
s_mul_i32 s1, 6, s0 // (op 12010) 
s_add_u32 s1, s1, 192 // (op 12010) 

v_add_u32 v3, v9, s1 // (op 12010) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 12010) Store value



v_accvgpr_read v10, a220 // (op 12052) 
v_accvgpr_read v11, a221 // (op 12052) 
v_accvgpr_read v12, a222 // (op 12052) 
v_accvgpr_read v13, a223 // (op 12052) 
v_mul_f32 v14, v7, v10 // (op 12052) 
v_mul_f32 v15, v7, v11 // (op 12052) 
v_mul_f32 v16, v7, v12 // (op 12052) 
v_mul_f32 v17, v7, v13 // (op 12052) 

v_cvt_f16_f32 v10, v14 // (op 12054) 
v_cvt_f16_f32 v11, v15 // (op 12054) 
v_cvt_f16_f32 v12, v16 // (op 12054) 
v_cvt_f16_f32 v13, v17 // (op 12054) 


v_pack_b32_f16 v4, v10, v11 // (op 12012) 
v_pack_b32_f16 v5, v12, v13 // (op 12012) 

s_lshl_b32 s0, s4, 5 // (op 12012) 
s_mul_i32 s1, 7, s0 // (op 12012) 
s_add_u32 s1, s1, 192 // (op 12012) 

v_add_u32 v3, v9, s1 // (op 12012) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 12012) Store value



v_accvgpr_read v10, a224 // (op 12142) 
v_accvgpr_read v11, a225 // (op 12142) 
v_accvgpr_read v12, a226 // (op 12142) 
v_accvgpr_read v13, a227 // (op 12142) 
v_mul_f32 v14, v7, v10 // (op 12142) 
v_mul_f32 v15, v7, v11 // (op 12142) 
v_mul_f32 v16, v7, v12 // (op 12142) 
v_mul_f32 v17, v7, v13 // (op 12142) 

v_cvt_f16_f32 v10, v14 // (op 12144) 
v_cvt_f16_f32 v11, v15 // (op 12144) 
v_cvt_f16_f32 v12, v16 // (op 12144) 
v_cvt_f16_f32 v13, v17 // (op 12144) 


v_pack_b32_f16 v4, v10, v11 // (op 12146) 
v_pack_b32_f16 v5, v12, v13 // (op 12146) 

v_add_u32 v3, 224, v9 // (op 12146) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 12146) Store value



v_accvgpr_read v10, a228 // (op 12164) 
v_accvgpr_read v11, a229 // (op 12164) 
v_accvgpr_read v12, a230 // (op 12164) 
v_accvgpr_read v13, a231 // (op 12164) 
v_mul_f32 v14, v7, v10 // (op 12164) 
v_mul_f32 v15, v7, v11 // (op 12164) 
v_mul_f32 v16, v7, v12 // (op 12164) 
v_mul_f32 v17, v7, v13 // (op 12164) 

v_cvt_f16_f32 v10, v14 // (op 12166) 
v_cvt_f16_f32 v11, v15 // (op 12166) 
v_cvt_f16_f32 v12, v16 // (op 12166) 
v_cvt_f16_f32 v13, v17 // (op 12166) 


v_pack_b32_f16 v4, v10, v11 // (op 12148) 
v_pack_b32_f16 v5, v12, v13 // (op 12148) 

s_lshl_b32 s0, s4, 5 // (op 12148) 
s_add_u32 s0, s0, 224 // (op 12148) 
v_add_u32 v3, v9, s0 // (op 12148) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 12148) Store value



v_accvgpr_read v10, a232 // (op 12170) 
v_accvgpr_read v11, a233 // (op 12170) 
v_accvgpr_read v12, a234 // (op 12170) 
v_accvgpr_read v13, a235 // (op 12170) 
v_mul_f32 v14, v7, v10 // (op 12170) 
v_mul_f32 v15, v7, v11 // (op 12170) 
v_mul_f32 v16, v7, v12 // (op 12170) 
v_mul_f32 v17, v7, v13 // (op 12170) 

v_cvt_f16_f32 v10, v14 // (op 12172) 
v_cvt_f16_f32 v11, v15 // (op 12172) 
v_cvt_f16_f32 v12, v16 // (op 12172) 
v_cvt_f16_f32 v13, v17 // (op 12172) 


v_pack_b32_f16 v4, v10, v11 // (op 12150) 
v_pack_b32_f16 v5, v12, v13 // (op 12150) 

s_lshl_b32 s0, s4, 6 // (op 12150) 
s_add_u32 s0, s0, 224 // (op 12150) 
v_add_u32 v3, v9, s0 // (op 12150) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 12150) Store value



v_accvgpr_read v10, a236 // (op 12176) 
v_accvgpr_read v11, a237 // (op 12176) 
v_accvgpr_read v12, a238 // (op 12176) 
v_accvgpr_read v13, a239 // (op 12176) 
v_mul_f32 v14, v7, v10 // (op 12176) 
v_mul_f32 v15, v7, v11 // (op 12176) 
v_mul_f32 v16, v7, v12 // (op 12176) 
v_mul_f32 v17, v7, v13 // (op 12176) 

v_cvt_f16_f32 v10, v14 // (op 12178) 
v_cvt_f16_f32 v11, v15 // (op 12178) 
v_cvt_f16_f32 v12, v16 // (op 12178) 
v_cvt_f16_f32 v13, v17 // (op 12178) 


v_pack_b32_f16 v4, v10, v11 // (op 12152) 
v_pack_b32_f16 v5, v12, v13 // (op 12152) 

s_lshl_b32 s0, s4, 5 // (op 12152) 
s_mul_i32 s1, 3, s0 // (op 12152) 
s_add_u32 s1, s1, 224 // (op 12152) 

v_add_u32 v3, v9, s1 // (op 12152) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 12152) Store value



v_accvgpr_read v10, a240 // (op 12182) 
v_accvgpr_read v11, a241 // (op 12182) 
v_accvgpr_read v12, a242 // (op 12182) 
v_accvgpr_read v13, a243 // (op 12182) 
v_mul_f32 v14, v7, v10 // (op 12182) 
v_mul_f32 v15, v7, v11 // (op 12182) 
v_mul_f32 v16, v7, v12 // (op 12182) 
v_mul_f32 v17, v7, v13 // (op 12182) 

v_cvt_f16_f32 v10, v14 // (op 12184) 
v_cvt_f16_f32 v11, v15 // (op 12184) 
v_cvt_f16_f32 v12, v16 // (op 12184) 
v_cvt_f16_f32 v13, v17 // (op 12184) 


v_pack_b32_f16 v4, v10, v11 // (op 12154) 
v_pack_b32_f16 v5, v12, v13 // (op 12154) 

s_lshl_b32 s0, s4, 7 // (op 12154) 
s_add_u32 s0, s0, 224 // (op 12154) 
v_add_u32 v3, v9, s0 // (op 12154) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 12154) Store value



v_accvgpr_read v10, a244 // (op 12188) 
v_accvgpr_read v11, a245 // (op 12188) 
v_accvgpr_read v12, a246 // (op 12188) 
v_accvgpr_read v13, a247 // (op 12188) 
v_mul_f32 v14, v7, v10 // (op 12188) 
v_mul_f32 v15, v7, v11 // (op 12188) 
v_mul_f32 v16, v7, v12 // (op 12188) 
v_mul_f32 v17, v7, v13 // (op 12188) 

v_cvt_f16_f32 v10, v14 // (op 12190) 
v_cvt_f16_f32 v11, v15 // (op 12190) 
v_cvt_f16_f32 v12, v16 // (op 12190) 
v_cvt_f16_f32 v13, v17 // (op 12190) 


v_pack_b32_f16 v4, v10, v11 // (op 12156) 
v_pack_b32_f16 v5, v12, v13 // (op 12156) 

s_lshl_b32 s0, s4, 5 // (op 12156) 
s_mul_i32 s1, 5, s0 // (op 12156) 
s_add_u32 s1, s1, 224 // (op 12156) 

v_add_u32 v3, v9, s1 // (op 12156) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 12156) Store value



v_accvgpr_read v10, a248 // (op 12194) 
v_accvgpr_read v11, a249 // (op 12194) 
v_accvgpr_read v12, a250 // (op 12194) 
v_accvgpr_read v13, a251 // (op 12194) 
v_mul_f32 v14, v7, v10 // (op 12194) 
v_mul_f32 v15, v7, v11 // (op 12194) 
v_mul_f32 v16, v7, v12 // (op 12194) 
v_mul_f32 v17, v7, v13 // (op 12194) 

v_cvt_f16_f32 v10, v14 // (op 12196) 
v_cvt_f16_f32 v11, v15 // (op 12196) 
v_cvt_f16_f32 v12, v16 // (op 12196) 
v_cvt_f16_f32 v13, v17 // (op 12196) 


v_pack_b32_f16 v4, v10, v11 // (op 12158) 
v_pack_b32_f16 v5, v12, v13 // (op 12158) 

s_lshl_b32 s0, s4, 5 // (op 12158) 
s_mul_i32 s1, 6, s0 // (op 12158) 
s_add_u32 s1, s1, 224 // (op 12158) 

v_add_u32 v3, v9, s1 // (op 12158) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 12158) Store value



v_accvgpr_read v10, a252 // (op 12200) 
v_accvgpr_read v11, a253 // (op 12200) 
v_accvgpr_read v12, a254 // (op 12200) 
v_accvgpr_read v13, a255 // (op 12200) 
v_mul_f32 v14, v7, v10 // (op 12200) 
v_mul_f32 v15, v7, v11 // (op 12200) 
v_mul_f32 v16, v7, v12 // (op 12200) 
v_mul_f32 v17, v7, v13 // (op 12200) 

v_cvt_f16_f32 v10, v14 // (op 12202) 
v_cvt_f16_f32 v11, v15 // (op 12202) 
v_cvt_f16_f32 v12, v16 // (op 12202) 
v_cvt_f16_f32 v13, v17 // (op 12202) 


v_pack_b32_f16 v4, v10, v11 // (op 12160) 
v_pack_b32_f16 v5, v12, v13 // (op 12160) 

s_lshl_b32 s0, s4, 5 // (op 12160) 
s_mul_i32 s1, 7, s0 // (op 12160) 
s_add_u32 s1, s1, 224 // (op 12160) 

v_add_u32 v3, v9, s1 // (op 12160) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 12160) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalBottom_0_beta_116
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalFalse_0_beta_116:


v_add_u32 v3, 224, v8 // (op 13720) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13720) Load value

v_accvgpr_read v10, a224 // (op 13699) 
v_accvgpr_read v11, a225 // (op 13699) 
v_accvgpr_read v12, a226 // (op 13699) 
v_accvgpr_read v13, a227 // (op 13699) 
v_mul_f32 v14, v7, v10 // (op 13699) 
v_mul_f32 v15, v7, v11 // (op 13699) 
v_mul_f32 v16, v7, v12 // (op 13699) 
v_mul_f32 v17, v7, v13 // (op 13699) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13719) 
v_lshrrev_b32 v11, 16, v4 // (op 13719) 
v_cvt_f32_f16 v11, v11 // (op 13719) 
v_mul_f32 v18, v0, v10 // (op 13719) 
v_mul_f32 v19, v0, v11 // (op 13719) 
v_cvt_f32_f16 v10, v5 // (op 13719) 
v_lshrrev_b32 v11, 16, v5 // (op 13719) 
v_cvt_f32_f16 v11, v11 // (op 13719) 
v_mul_f32 v20, v0, v10 // (op 13719) 
v_mul_f32 v21, v0, v11 // (op 13719) 


v_add_f32 v10, v18, v14 // (op 13700) 
v_add_f32 v11, v19, v15 // (op 13700) 
v_add_f32 v12, v20, v16 // (op 13700) 
v_add_f32 v13, v21, v17 // (op 13700) 


v_cvt_f16_f32 v14, v10 // (op 13701) 
v_cvt_f16_f32 v15, v11 // (op 13701) 
v_cvt_f16_f32 v16, v12 // (op 13701) 
v_cvt_f16_f32 v17, v13 // (op 13701) 


v_pack_b32_f16 v10, v14, v15 // (op 13703) 
v_pack_b32_f16 v11, v16, v17 // (op 13703) 

v_add_u32 v3, 224, v9 // (op 13703) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13703) Store value





s_lshl_b32 s0, s2, 5 // (op 13725) 
s_add_u32 s0, s0, 224 // (op 13725) 
v_add_u32 v3, v8, s0 // (op 13725) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13725) Load value

v_accvgpr_read v10, a228 // (op 13757) 
v_accvgpr_read v11, a229 // (op 13757) 
v_accvgpr_read v12, a230 // (op 13757) 
v_accvgpr_read v13, a231 // (op 13757) 
v_mul_f32 v14, v7, v10 // (op 13757) 
v_mul_f32 v15, v7, v11 // (op 13757) 
v_mul_f32 v16, v7, v12 // (op 13757) 
v_mul_f32 v17, v7, v13 // (op 13757) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13722) 
v_lshrrev_b32 v11, 16, v4 // (op 13722) 
v_cvt_f32_f16 v11, v11 // (op 13722) 
v_mul_f32 v18, v0, v10 // (op 13722) 
v_mul_f32 v19, v0, v11 // (op 13722) 
v_cvt_f32_f16 v10, v5 // (op 13722) 
v_lshrrev_b32 v11, 16, v5 // (op 13722) 
v_cvt_f32_f16 v11, v11 // (op 13722) 
v_mul_f32 v20, v0, v10 // (op 13722) 
v_mul_f32 v21, v0, v11 // (op 13722) 


v_add_f32 v10, v18, v14 // (op 13723) 
v_add_f32 v11, v19, v15 // (op 13723) 
v_add_f32 v12, v20, v16 // (op 13723) 
v_add_f32 v13, v21, v17 // (op 13723) 


v_cvt_f16_f32 v14, v10 // (op 13724) 
v_cvt_f16_f32 v15, v11 // (op 13724) 
v_cvt_f16_f32 v16, v12 // (op 13724) 
v_cvt_f16_f32 v17, v13 // (op 13724) 


v_pack_b32_f16 v10, v14, v15 // (op 13705) 
v_pack_b32_f16 v11, v16, v17 // (op 13705) 

s_lshl_b32 s0, s4, 5 // (op 13705) 
s_add_u32 s0, s0, 224 // (op 13705) 
v_add_u32 v3, v9, s0 // (op 13705) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13705) Store value





s_lshl_b32 s0, s2, 6 // (op 13730) 
s_add_u32 s0, s0, 224 // (op 13730) 
v_add_u32 v3, v8, s0 // (op 13730) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13730) Load value

v_accvgpr_read v10, a232 // (op 13759) 
v_accvgpr_read v11, a233 // (op 13759) 
v_accvgpr_read v12, a234 // (op 13759) 
v_accvgpr_read v13, a235 // (op 13759) 
v_mul_f32 v14, v7, v10 // (op 13759) 
v_mul_f32 v15, v7, v11 // (op 13759) 
v_mul_f32 v16, v7, v12 // (op 13759) 
v_mul_f32 v17, v7, v13 // (op 13759) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13727) 
v_lshrrev_b32 v11, 16, v4 // (op 13727) 
v_cvt_f32_f16 v11, v11 // (op 13727) 
v_mul_f32 v18, v0, v10 // (op 13727) 
v_mul_f32 v19, v0, v11 // (op 13727) 
v_cvt_f32_f16 v10, v5 // (op 13727) 
v_lshrrev_b32 v11, 16, v5 // (op 13727) 
v_cvt_f32_f16 v11, v11 // (op 13727) 
v_mul_f32 v20, v0, v10 // (op 13727) 
v_mul_f32 v21, v0, v11 // (op 13727) 


v_add_f32 v10, v18, v14 // (op 13728) 
v_add_f32 v11, v19, v15 // (op 13728) 
v_add_f32 v12, v20, v16 // (op 13728) 
v_add_f32 v13, v21, v17 // (op 13728) 


v_cvt_f16_f32 v14, v10 // (op 13729) 
v_cvt_f16_f32 v15, v11 // (op 13729) 
v_cvt_f16_f32 v16, v12 // (op 13729) 
v_cvt_f16_f32 v17, v13 // (op 13729) 


v_pack_b32_f16 v10, v14, v15 // (op 13707) 
v_pack_b32_f16 v11, v16, v17 // (op 13707) 

s_lshl_b32 s0, s4, 6 // (op 13707) 
s_add_u32 s0, s0, 224 // (op 13707) 
v_add_u32 v3, v9, s0 // (op 13707) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13707) Store value





s_lshl_b32 s0, s2, 5 // (op 13735) 
s_mul_i32 s1, 3, s0 // (op 13735) 
s_add_u32 s1, s1, 224 // (op 13735) 

v_add_u32 v3, v8, s1 // (op 13735) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13735) Load value

v_accvgpr_read v10, a236 // (op 13761) 
v_accvgpr_read v11, a237 // (op 13761) 
v_accvgpr_read v12, a238 // (op 13761) 
v_accvgpr_read v13, a239 // (op 13761) 
v_mul_f32 v14, v7, v10 // (op 13761) 
v_mul_f32 v15, v7, v11 // (op 13761) 
v_mul_f32 v16, v7, v12 // (op 13761) 
v_mul_f32 v17, v7, v13 // (op 13761) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13732) 
v_lshrrev_b32 v11, 16, v4 // (op 13732) 
v_cvt_f32_f16 v11, v11 // (op 13732) 
v_mul_f32 v18, v0, v10 // (op 13732) 
v_mul_f32 v19, v0, v11 // (op 13732) 
v_cvt_f32_f16 v10, v5 // (op 13732) 
v_lshrrev_b32 v11, 16, v5 // (op 13732) 
v_cvt_f32_f16 v11, v11 // (op 13732) 
v_mul_f32 v20, v0, v10 // (op 13732) 
v_mul_f32 v21, v0, v11 // (op 13732) 


v_add_f32 v10, v18, v14 // (op 13733) 
v_add_f32 v11, v19, v15 // (op 13733) 
v_add_f32 v12, v20, v16 // (op 13733) 
v_add_f32 v13, v21, v17 // (op 13733) 


v_cvt_f16_f32 v14, v10 // (op 13734) 
v_cvt_f16_f32 v15, v11 // (op 13734) 
v_cvt_f16_f32 v16, v12 // (op 13734) 
v_cvt_f16_f32 v17, v13 // (op 13734) 


v_pack_b32_f16 v10, v14, v15 // (op 13709) 
v_pack_b32_f16 v11, v16, v17 // (op 13709) 

s_lshl_b32 s0, s4, 5 // (op 13709) 
s_mul_i32 s1, 3, s0 // (op 13709) 
s_add_u32 s1, s1, 224 // (op 13709) 

v_add_u32 v3, v9, s1 // (op 13709) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13709) Store value





s_lshl_b32 s0, s2, 7 // (op 13740) 
s_add_u32 s0, s0, 224 // (op 13740) 
v_add_u32 v3, v8, s0 // (op 13740) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13740) Load value

v_accvgpr_read v10, a240 // (op 13763) 
v_accvgpr_read v11, a241 // (op 13763) 
v_accvgpr_read v12, a242 // (op 13763) 
v_accvgpr_read v13, a243 // (op 13763) 
v_mul_f32 v14, v7, v10 // (op 13763) 
v_mul_f32 v15, v7, v11 // (op 13763) 
v_mul_f32 v16, v7, v12 // (op 13763) 
v_mul_f32 v17, v7, v13 // (op 13763) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13737) 
v_lshrrev_b32 v11, 16, v4 // (op 13737) 
v_cvt_f32_f16 v11, v11 // (op 13737) 
v_mul_f32 v18, v0, v10 // (op 13737) 
v_mul_f32 v19, v0, v11 // (op 13737) 
v_cvt_f32_f16 v10, v5 // (op 13737) 
v_lshrrev_b32 v11, 16, v5 // (op 13737) 
v_cvt_f32_f16 v11, v11 // (op 13737) 
v_mul_f32 v20, v0, v10 // (op 13737) 
v_mul_f32 v21, v0, v11 // (op 13737) 


v_add_f32 v10, v18, v14 // (op 13738) 
v_add_f32 v11, v19, v15 // (op 13738) 
v_add_f32 v12, v20, v16 // (op 13738) 
v_add_f32 v13, v21, v17 // (op 13738) 


v_cvt_f16_f32 v14, v10 // (op 13739) 
v_cvt_f16_f32 v15, v11 // (op 13739) 
v_cvt_f16_f32 v16, v12 // (op 13739) 
v_cvt_f16_f32 v17, v13 // (op 13739) 


v_pack_b32_f16 v10, v14, v15 // (op 13711) 
v_pack_b32_f16 v11, v16, v17 // (op 13711) 

s_lshl_b32 s0, s4, 7 // (op 13711) 
s_add_u32 s0, s0, 224 // (op 13711) 
v_add_u32 v3, v9, s0 // (op 13711) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13711) Store value





s_lshl_b32 s0, s2, 5 // (op 13745) 
s_mul_i32 s1, 5, s0 // (op 13745) 
s_add_u32 s1, s1, 224 // (op 13745) 

v_add_u32 v3, v8, s1 // (op 13745) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13745) Load value

v_accvgpr_read v10, a244 // (op 13765) 
v_accvgpr_read v11, a245 // (op 13765) 
v_accvgpr_read v12, a246 // (op 13765) 
v_accvgpr_read v13, a247 // (op 13765) 
v_mul_f32 v14, v7, v10 // (op 13765) 
v_mul_f32 v15, v7, v11 // (op 13765) 
v_mul_f32 v16, v7, v12 // (op 13765) 
v_mul_f32 v17, v7, v13 // (op 13765) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13742) 
v_lshrrev_b32 v11, 16, v4 // (op 13742) 
v_cvt_f32_f16 v11, v11 // (op 13742) 
v_mul_f32 v18, v0, v10 // (op 13742) 
v_mul_f32 v19, v0, v11 // (op 13742) 
v_cvt_f32_f16 v10, v5 // (op 13742) 
v_lshrrev_b32 v11, 16, v5 // (op 13742) 
v_cvt_f32_f16 v11, v11 // (op 13742) 
v_mul_f32 v20, v0, v10 // (op 13742) 
v_mul_f32 v21, v0, v11 // (op 13742) 


v_add_f32 v10, v18, v14 // (op 13743) 
v_add_f32 v11, v19, v15 // (op 13743) 
v_add_f32 v12, v20, v16 // (op 13743) 
v_add_f32 v13, v21, v17 // (op 13743) 


v_cvt_f16_f32 v14, v10 // (op 13744) 
v_cvt_f16_f32 v15, v11 // (op 13744) 
v_cvt_f16_f32 v16, v12 // (op 13744) 
v_cvt_f16_f32 v17, v13 // (op 13744) 


v_pack_b32_f16 v10, v14, v15 // (op 13713) 
v_pack_b32_f16 v11, v16, v17 // (op 13713) 

s_lshl_b32 s0, s4, 5 // (op 13713) 
s_mul_i32 s1, 5, s0 // (op 13713) 
s_add_u32 s1, s1, 224 // (op 13713) 

v_add_u32 v3, v9, s1 // (op 13713) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13713) Store value





s_lshl_b32 s0, s2, 5 // (op 13750) 
s_mul_i32 s1, 6, s0 // (op 13750) 
s_add_u32 s1, s1, 224 // (op 13750) 

v_add_u32 v3, v8, s1 // (op 13750) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13750) Load value

v_accvgpr_read v10, a248 // (op 13767) 
v_accvgpr_read v11, a249 // (op 13767) 
v_accvgpr_read v12, a250 // (op 13767) 
v_accvgpr_read v13, a251 // (op 13767) 
v_mul_f32 v14, v7, v10 // (op 13767) 
v_mul_f32 v15, v7, v11 // (op 13767) 
v_mul_f32 v16, v7, v12 // (op 13767) 
v_mul_f32 v17, v7, v13 // (op 13767) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13747) 
v_lshrrev_b32 v11, 16, v4 // (op 13747) 
v_cvt_f32_f16 v11, v11 // (op 13747) 
v_mul_f32 v18, v0, v10 // (op 13747) 
v_mul_f32 v19, v0, v11 // (op 13747) 
v_cvt_f32_f16 v10, v5 // (op 13747) 
v_lshrrev_b32 v11, 16, v5 // (op 13747) 
v_cvt_f32_f16 v11, v11 // (op 13747) 
v_mul_f32 v20, v0, v10 // (op 13747) 
v_mul_f32 v21, v0, v11 // (op 13747) 


v_add_f32 v10, v18, v14 // (op 13748) 
v_add_f32 v11, v19, v15 // (op 13748) 
v_add_f32 v12, v20, v16 // (op 13748) 
v_add_f32 v13, v21, v17 // (op 13748) 


v_cvt_f16_f32 v14, v10 // (op 13749) 
v_cvt_f16_f32 v15, v11 // (op 13749) 
v_cvt_f16_f32 v16, v12 // (op 13749) 
v_cvt_f16_f32 v17, v13 // (op 13749) 


v_pack_b32_f16 v10, v14, v15 // (op 13715) 
v_pack_b32_f16 v11, v16, v17 // (op 13715) 

s_lshl_b32 s0, s4, 5 // (op 13715) 
s_mul_i32 s1, 6, s0 // (op 13715) 
s_add_u32 s1, s1, 224 // (op 13715) 

v_add_u32 v3, v9, s1 // (op 13715) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13715) Store value





s_lshl_b32 s0, s2, 5 // (op 13755) 
s_mul_i32 s1, 7, s0 // (op 13755) 
s_add_u32 s1, s1, 224 // (op 13755) 

v_add_u32 v3, v8, s1 // (op 13755) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13755) Load value

v_accvgpr_read v10, a252 // (op 13769) 
v_accvgpr_read v11, a253 // (op 13769) 
v_accvgpr_read v12, a254 // (op 13769) 
v_accvgpr_read v13, a255 // (op 13769) 
v_mul_f32 v14, v7, v10 // (op 13769) 
v_mul_f32 v15, v7, v11 // (op 13769) 
v_mul_f32 v16, v7, v12 // (op 13769) 
v_mul_f32 v17, v7, v13 // (op 13769) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13752) 
v_lshrrev_b32 v11, 16, v4 // (op 13752) 
v_cvt_f32_f16 v11, v11 // (op 13752) 
v_mul_f32 v18, v0, v10 // (op 13752) 
v_mul_f32 v19, v0, v11 // (op 13752) 
v_cvt_f32_f16 v10, v5 // (op 13752) 
v_lshrrev_b32 v11, 16, v5 // (op 13752) 
v_cvt_f32_f16 v11, v11 // (op 13752) 
v_mul_f32 v20, v0, v10 // (op 13752) 
v_mul_f32 v21, v0, v11 // (op 13752) 


v_add_f32 v10, v18, v14 // (op 13753) 
v_add_f32 v11, v19, v15 // (op 13753) 
v_add_f32 v12, v20, v16 // (op 13753) 
v_add_f32 v13, v21, v17 // (op 13753) 


v_cvt_f16_f32 v14, v10 // (op 13754) 
v_cvt_f16_f32 v15, v11 // (op 13754) 
v_cvt_f16_f32 v16, v12 // (op 13754) 
v_cvt_f16_f32 v17, v13 // (op 13754) 


v_pack_b32_f16 v10, v14, v15 // (op 13717) 
v_pack_b32_f16 v11, v16, v17 // (op 13717) 

s_lshl_b32 s0, s4, 5 // (op 13717) 
s_mul_i32 s1, 7, s0 // (op 13717) 
s_add_u32 s1, s1, 224 // (op 13717) 

v_add_u32 v3, v9, s1 // (op 13717) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13717) Store value





v_add_u32 v3, 160, v8 // (op 13378) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13378) Load value

v_accvgpr_read v10, a160 // (op 13357) 
v_accvgpr_read v11, a161 // (op 13357) 
v_accvgpr_read v12, a162 // (op 13357) 
v_accvgpr_read v13, a163 // (op 13357) 
v_mul_f32 v14, v7, v10 // (op 13357) 
v_mul_f32 v15, v7, v11 // (op 13357) 
v_mul_f32 v16, v7, v12 // (op 13357) 
v_mul_f32 v17, v7, v13 // (op 13357) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13377) 
v_lshrrev_b32 v11, 16, v4 // (op 13377) 
v_cvt_f32_f16 v11, v11 // (op 13377) 
v_mul_f32 v18, v0, v10 // (op 13377) 
v_mul_f32 v19, v0, v11 // (op 13377) 
v_cvt_f32_f16 v10, v5 // (op 13377) 
v_lshrrev_b32 v11, 16, v5 // (op 13377) 
v_cvt_f32_f16 v11, v11 // (op 13377) 
v_mul_f32 v20, v0, v10 // (op 13377) 
v_mul_f32 v21, v0, v11 // (op 13377) 


v_add_f32 v10, v18, v14 // (op 13358) 
v_add_f32 v11, v19, v15 // (op 13358) 
v_add_f32 v12, v20, v16 // (op 13358) 
v_add_f32 v13, v21, v17 // (op 13358) 


v_cvt_f16_f32 v14, v10 // (op 13359) 
v_cvt_f16_f32 v15, v11 // (op 13359) 
v_cvt_f16_f32 v16, v12 // (op 13359) 
v_cvt_f16_f32 v17, v13 // (op 13359) 


v_pack_b32_f16 v10, v14, v15 // (op 13361) 
v_pack_b32_f16 v11, v16, v17 // (op 13361) 

v_add_u32 v3, 160, v9 // (op 13361) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13361) Store value





s_lshl_b32 s0, s2, 5 // (op 13383) 
s_add_u32 s0, s0, 160 // (op 13383) 
v_add_u32 v3, v8, s0 // (op 13383) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13383) Load value

v_accvgpr_read v10, a164 // (op 13415) 
v_accvgpr_read v11, a165 // (op 13415) 
v_accvgpr_read v12, a166 // (op 13415) 
v_accvgpr_read v13, a167 // (op 13415) 
v_mul_f32 v14, v7, v10 // (op 13415) 
v_mul_f32 v15, v7, v11 // (op 13415) 
v_mul_f32 v16, v7, v12 // (op 13415) 
v_mul_f32 v17, v7, v13 // (op 13415) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13380) 
v_lshrrev_b32 v11, 16, v4 // (op 13380) 
v_cvt_f32_f16 v11, v11 // (op 13380) 
v_mul_f32 v18, v0, v10 // (op 13380) 
v_mul_f32 v19, v0, v11 // (op 13380) 
v_cvt_f32_f16 v10, v5 // (op 13380) 
v_lshrrev_b32 v11, 16, v5 // (op 13380) 
v_cvt_f32_f16 v11, v11 // (op 13380) 
v_mul_f32 v20, v0, v10 // (op 13380) 
v_mul_f32 v21, v0, v11 // (op 13380) 


v_add_f32 v10, v18, v14 // (op 13381) 
v_add_f32 v11, v19, v15 // (op 13381) 
v_add_f32 v12, v20, v16 // (op 13381) 
v_add_f32 v13, v21, v17 // (op 13381) 


v_cvt_f16_f32 v14, v10 // (op 13382) 
v_cvt_f16_f32 v15, v11 // (op 13382) 
v_cvt_f16_f32 v16, v12 // (op 13382) 
v_cvt_f16_f32 v17, v13 // (op 13382) 


v_pack_b32_f16 v10, v14, v15 // (op 13363) 
v_pack_b32_f16 v11, v16, v17 // (op 13363) 

s_lshl_b32 s0, s4, 5 // (op 13363) 
s_add_u32 s0, s0, 160 // (op 13363) 
v_add_u32 v3, v9, s0 // (op 13363) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13363) Store value





s_lshl_b32 s0, s2, 6 // (op 13388) 
s_add_u32 s0, s0, 160 // (op 13388) 
v_add_u32 v3, v8, s0 // (op 13388) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13388) Load value

v_accvgpr_read v10, a168 // (op 13417) 
v_accvgpr_read v11, a169 // (op 13417) 
v_accvgpr_read v12, a170 // (op 13417) 
v_accvgpr_read v13, a171 // (op 13417) 
v_mul_f32 v14, v7, v10 // (op 13417) 
v_mul_f32 v15, v7, v11 // (op 13417) 
v_mul_f32 v16, v7, v12 // (op 13417) 
v_mul_f32 v17, v7, v13 // (op 13417) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13385) 
v_lshrrev_b32 v11, 16, v4 // (op 13385) 
v_cvt_f32_f16 v11, v11 // (op 13385) 
v_mul_f32 v18, v0, v10 // (op 13385) 
v_mul_f32 v19, v0, v11 // (op 13385) 
v_cvt_f32_f16 v10, v5 // (op 13385) 
v_lshrrev_b32 v11, 16, v5 // (op 13385) 
v_cvt_f32_f16 v11, v11 // (op 13385) 
v_mul_f32 v20, v0, v10 // (op 13385) 
v_mul_f32 v21, v0, v11 // (op 13385) 


v_add_f32 v10, v18, v14 // (op 13386) 
v_add_f32 v11, v19, v15 // (op 13386) 
v_add_f32 v12, v20, v16 // (op 13386) 
v_add_f32 v13, v21, v17 // (op 13386) 


v_cvt_f16_f32 v14, v10 // (op 13387) 
v_cvt_f16_f32 v15, v11 // (op 13387) 
v_cvt_f16_f32 v16, v12 // (op 13387) 
v_cvt_f16_f32 v17, v13 // (op 13387) 


v_pack_b32_f16 v10, v14, v15 // (op 13365) 
v_pack_b32_f16 v11, v16, v17 // (op 13365) 

s_lshl_b32 s0, s4, 6 // (op 13365) 
s_add_u32 s0, s0, 160 // (op 13365) 
v_add_u32 v3, v9, s0 // (op 13365) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13365) Store value





s_lshl_b32 s0, s2, 5 // (op 13393) 
s_mul_i32 s1, 3, s0 // (op 13393) 
s_add_u32 s1, s1, 160 // (op 13393) 

v_add_u32 v3, v8, s1 // (op 13393) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13393) Load value

v_accvgpr_read v10, a172 // (op 13419) 
v_accvgpr_read v11, a173 // (op 13419) 
v_accvgpr_read v12, a174 // (op 13419) 
v_accvgpr_read v13, a175 // (op 13419) 
v_mul_f32 v14, v7, v10 // (op 13419) 
v_mul_f32 v15, v7, v11 // (op 13419) 
v_mul_f32 v16, v7, v12 // (op 13419) 
v_mul_f32 v17, v7, v13 // (op 13419) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13390) 
v_lshrrev_b32 v11, 16, v4 // (op 13390) 
v_cvt_f32_f16 v11, v11 // (op 13390) 
v_mul_f32 v18, v0, v10 // (op 13390) 
v_mul_f32 v19, v0, v11 // (op 13390) 
v_cvt_f32_f16 v10, v5 // (op 13390) 
v_lshrrev_b32 v11, 16, v5 // (op 13390) 
v_cvt_f32_f16 v11, v11 // (op 13390) 
v_mul_f32 v20, v0, v10 // (op 13390) 
v_mul_f32 v21, v0, v11 // (op 13390) 


v_add_f32 v10, v18, v14 // (op 13391) 
v_add_f32 v11, v19, v15 // (op 13391) 
v_add_f32 v12, v20, v16 // (op 13391) 
v_add_f32 v13, v21, v17 // (op 13391) 


v_cvt_f16_f32 v14, v10 // (op 13392) 
v_cvt_f16_f32 v15, v11 // (op 13392) 
v_cvt_f16_f32 v16, v12 // (op 13392) 
v_cvt_f16_f32 v17, v13 // (op 13392) 


v_pack_b32_f16 v10, v14, v15 // (op 13367) 
v_pack_b32_f16 v11, v16, v17 // (op 13367) 

s_lshl_b32 s0, s4, 5 // (op 13367) 
s_mul_i32 s1, 3, s0 // (op 13367) 
s_add_u32 s1, s1, 160 // (op 13367) 

v_add_u32 v3, v9, s1 // (op 13367) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13367) Store value





s_lshl_b32 s0, s2, 7 // (op 13398) 
s_add_u32 s0, s0, 160 // (op 13398) 
v_add_u32 v3, v8, s0 // (op 13398) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13398) Load value

v_accvgpr_read v10, a176 // (op 13421) 
v_accvgpr_read v11, a177 // (op 13421) 
v_accvgpr_read v12, a178 // (op 13421) 
v_accvgpr_read v13, a179 // (op 13421) 
v_mul_f32 v14, v7, v10 // (op 13421) 
v_mul_f32 v15, v7, v11 // (op 13421) 
v_mul_f32 v16, v7, v12 // (op 13421) 
v_mul_f32 v17, v7, v13 // (op 13421) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13395) 
v_lshrrev_b32 v11, 16, v4 // (op 13395) 
v_cvt_f32_f16 v11, v11 // (op 13395) 
v_mul_f32 v18, v0, v10 // (op 13395) 
v_mul_f32 v19, v0, v11 // (op 13395) 
v_cvt_f32_f16 v10, v5 // (op 13395) 
v_lshrrev_b32 v11, 16, v5 // (op 13395) 
v_cvt_f32_f16 v11, v11 // (op 13395) 
v_mul_f32 v20, v0, v10 // (op 13395) 
v_mul_f32 v21, v0, v11 // (op 13395) 


v_add_f32 v10, v18, v14 // (op 13396) 
v_add_f32 v11, v19, v15 // (op 13396) 
v_add_f32 v12, v20, v16 // (op 13396) 
v_add_f32 v13, v21, v17 // (op 13396) 


v_cvt_f16_f32 v14, v10 // (op 13397) 
v_cvt_f16_f32 v15, v11 // (op 13397) 
v_cvt_f16_f32 v16, v12 // (op 13397) 
v_cvt_f16_f32 v17, v13 // (op 13397) 


v_pack_b32_f16 v10, v14, v15 // (op 13369) 
v_pack_b32_f16 v11, v16, v17 // (op 13369) 

s_lshl_b32 s0, s4, 7 // (op 13369) 
s_add_u32 s0, s0, 160 // (op 13369) 
v_add_u32 v3, v9, s0 // (op 13369) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13369) Store value





s_lshl_b32 s0, s2, 5 // (op 13403) 
s_mul_i32 s1, 5, s0 // (op 13403) 
s_add_u32 s1, s1, 160 // (op 13403) 

v_add_u32 v3, v8, s1 // (op 13403) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13403) Load value

v_accvgpr_read v10, a180 // (op 13423) 
v_accvgpr_read v11, a181 // (op 13423) 
v_accvgpr_read v12, a182 // (op 13423) 
v_accvgpr_read v13, a183 // (op 13423) 
v_mul_f32 v14, v7, v10 // (op 13423) 
v_mul_f32 v15, v7, v11 // (op 13423) 
v_mul_f32 v16, v7, v12 // (op 13423) 
v_mul_f32 v17, v7, v13 // (op 13423) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13400) 
v_lshrrev_b32 v11, 16, v4 // (op 13400) 
v_cvt_f32_f16 v11, v11 // (op 13400) 
v_mul_f32 v18, v0, v10 // (op 13400) 
v_mul_f32 v19, v0, v11 // (op 13400) 
v_cvt_f32_f16 v10, v5 // (op 13400) 
v_lshrrev_b32 v11, 16, v5 // (op 13400) 
v_cvt_f32_f16 v11, v11 // (op 13400) 
v_mul_f32 v20, v0, v10 // (op 13400) 
v_mul_f32 v21, v0, v11 // (op 13400) 


v_add_f32 v10, v18, v14 // (op 13401) 
v_add_f32 v11, v19, v15 // (op 13401) 
v_add_f32 v12, v20, v16 // (op 13401) 
v_add_f32 v13, v21, v17 // (op 13401) 


v_cvt_f16_f32 v14, v10 // (op 13402) 
v_cvt_f16_f32 v15, v11 // (op 13402) 
v_cvt_f16_f32 v16, v12 // (op 13402) 
v_cvt_f16_f32 v17, v13 // (op 13402) 


v_pack_b32_f16 v10, v14, v15 // (op 13371) 
v_pack_b32_f16 v11, v16, v17 // (op 13371) 

s_lshl_b32 s0, s4, 5 // (op 13371) 
s_mul_i32 s1, 5, s0 // (op 13371) 
s_add_u32 s1, s1, 160 // (op 13371) 

v_add_u32 v3, v9, s1 // (op 13371) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13371) Store value





s_lshl_b32 s0, s2, 5 // (op 13408) 
s_mul_i32 s1, 6, s0 // (op 13408) 
s_add_u32 s1, s1, 160 // (op 13408) 

v_add_u32 v3, v8, s1 // (op 13408) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13408) Load value

v_accvgpr_read v10, a184 // (op 13425) 
v_accvgpr_read v11, a185 // (op 13425) 
v_accvgpr_read v12, a186 // (op 13425) 
v_accvgpr_read v13, a187 // (op 13425) 
v_mul_f32 v14, v7, v10 // (op 13425) 
v_mul_f32 v15, v7, v11 // (op 13425) 
v_mul_f32 v16, v7, v12 // (op 13425) 
v_mul_f32 v17, v7, v13 // (op 13425) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13405) 
v_lshrrev_b32 v11, 16, v4 // (op 13405) 
v_cvt_f32_f16 v11, v11 // (op 13405) 
v_mul_f32 v18, v0, v10 // (op 13405) 
v_mul_f32 v19, v0, v11 // (op 13405) 
v_cvt_f32_f16 v10, v5 // (op 13405) 
v_lshrrev_b32 v11, 16, v5 // (op 13405) 
v_cvt_f32_f16 v11, v11 // (op 13405) 
v_mul_f32 v20, v0, v10 // (op 13405) 
v_mul_f32 v21, v0, v11 // (op 13405) 


v_add_f32 v10, v18, v14 // (op 13406) 
v_add_f32 v11, v19, v15 // (op 13406) 
v_add_f32 v12, v20, v16 // (op 13406) 
v_add_f32 v13, v21, v17 // (op 13406) 


v_cvt_f16_f32 v14, v10 // (op 13407) 
v_cvt_f16_f32 v15, v11 // (op 13407) 
v_cvt_f16_f32 v16, v12 // (op 13407) 
v_cvt_f16_f32 v17, v13 // (op 13407) 


v_pack_b32_f16 v10, v14, v15 // (op 13373) 
v_pack_b32_f16 v11, v16, v17 // (op 13373) 

s_lshl_b32 s0, s4, 5 // (op 13373) 
s_mul_i32 s1, 6, s0 // (op 13373) 
s_add_u32 s1, s1, 160 // (op 13373) 

v_add_u32 v3, v9, s1 // (op 13373) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13373) Store value





s_lshl_b32 s0, s2, 5 // (op 13413) 
s_mul_i32 s1, 7, s0 // (op 13413) 
s_add_u32 s1, s1, 160 // (op 13413) 

v_add_u32 v3, v8, s1 // (op 13413) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13413) Load value

v_accvgpr_read v10, a188 // (op 13427) 
v_accvgpr_read v11, a189 // (op 13427) 
v_accvgpr_read v12, a190 // (op 13427) 
v_accvgpr_read v13, a191 // (op 13427) 
v_mul_f32 v14, v7, v10 // (op 13427) 
v_mul_f32 v15, v7, v11 // (op 13427) 
v_mul_f32 v16, v7, v12 // (op 13427) 
v_mul_f32 v17, v7, v13 // (op 13427) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13410) 
v_lshrrev_b32 v11, 16, v4 // (op 13410) 
v_cvt_f32_f16 v11, v11 // (op 13410) 
v_mul_f32 v18, v0, v10 // (op 13410) 
v_mul_f32 v19, v0, v11 // (op 13410) 
v_cvt_f32_f16 v10, v5 // (op 13410) 
v_lshrrev_b32 v11, 16, v5 // (op 13410) 
v_cvt_f32_f16 v11, v11 // (op 13410) 
v_mul_f32 v20, v0, v10 // (op 13410) 
v_mul_f32 v21, v0, v11 // (op 13410) 


v_add_f32 v10, v18, v14 // (op 13411) 
v_add_f32 v11, v19, v15 // (op 13411) 
v_add_f32 v12, v20, v16 // (op 13411) 
v_add_f32 v13, v21, v17 // (op 13411) 


v_cvt_f16_f32 v14, v10 // (op 13412) 
v_cvt_f16_f32 v15, v11 // (op 13412) 
v_cvt_f16_f32 v16, v12 // (op 13412) 
v_cvt_f16_f32 v17, v13 // (op 13412) 


v_pack_b32_f16 v10, v14, v15 // (op 13375) 
v_pack_b32_f16 v11, v16, v17 // (op 13375) 

s_lshl_b32 s0, s4, 5 // (op 13375) 
s_mul_i32 s1, 7, s0 // (op 13375) 
s_add_u32 s1, s1, 160 // (op 13375) 

v_add_u32 v3, v9, s1 // (op 13375) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13375) Store value





v_add_u32 v3, 128, v8 // (op 13207) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13207) Load value

v_accvgpr_read v10, a128 // (op 13186) 
v_accvgpr_read v11, a129 // (op 13186) 
v_accvgpr_read v12, a130 // (op 13186) 
v_accvgpr_read v13, a131 // (op 13186) 
v_mul_f32 v14, v7, v10 // (op 13186) 
v_mul_f32 v15, v7, v11 // (op 13186) 
v_mul_f32 v16, v7, v12 // (op 13186) 
v_mul_f32 v17, v7, v13 // (op 13186) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13206) 
v_lshrrev_b32 v11, 16, v4 // (op 13206) 
v_cvt_f32_f16 v11, v11 // (op 13206) 
v_mul_f32 v18, v0, v10 // (op 13206) 
v_mul_f32 v19, v0, v11 // (op 13206) 
v_cvt_f32_f16 v10, v5 // (op 13206) 
v_lshrrev_b32 v11, 16, v5 // (op 13206) 
v_cvt_f32_f16 v11, v11 // (op 13206) 
v_mul_f32 v20, v0, v10 // (op 13206) 
v_mul_f32 v21, v0, v11 // (op 13206) 


v_add_f32 v10, v18, v14 // (op 13187) 
v_add_f32 v11, v19, v15 // (op 13187) 
v_add_f32 v12, v20, v16 // (op 13187) 
v_add_f32 v13, v21, v17 // (op 13187) 


v_cvt_f16_f32 v14, v10 // (op 13188) 
v_cvt_f16_f32 v15, v11 // (op 13188) 
v_cvt_f16_f32 v16, v12 // (op 13188) 
v_cvt_f16_f32 v17, v13 // (op 13188) 


v_pack_b32_f16 v10, v14, v15 // (op 13190) 
v_pack_b32_f16 v11, v16, v17 // (op 13190) 

v_add_u32 v3, 128, v9 // (op 13190) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13190) Store value





s_lshl_b32 s0, s2, 5 // (op 13212) 
s_add_u32 s0, s0, 128 // (op 13212) 
v_add_u32 v3, v8, s0 // (op 13212) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13212) Load value

v_accvgpr_read v10, a132 // (op 13244) 
v_accvgpr_read v11, a133 // (op 13244) 
v_accvgpr_read v12, a134 // (op 13244) 
v_accvgpr_read v13, a135 // (op 13244) 
v_mul_f32 v14, v7, v10 // (op 13244) 
v_mul_f32 v15, v7, v11 // (op 13244) 
v_mul_f32 v16, v7, v12 // (op 13244) 
v_mul_f32 v17, v7, v13 // (op 13244) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13209) 
v_lshrrev_b32 v11, 16, v4 // (op 13209) 
v_cvt_f32_f16 v11, v11 // (op 13209) 
v_mul_f32 v18, v0, v10 // (op 13209) 
v_mul_f32 v19, v0, v11 // (op 13209) 
v_cvt_f32_f16 v10, v5 // (op 13209) 
v_lshrrev_b32 v11, 16, v5 // (op 13209) 
v_cvt_f32_f16 v11, v11 // (op 13209) 
v_mul_f32 v20, v0, v10 // (op 13209) 
v_mul_f32 v21, v0, v11 // (op 13209) 


v_add_f32 v10, v18, v14 // (op 13210) 
v_add_f32 v11, v19, v15 // (op 13210) 
v_add_f32 v12, v20, v16 // (op 13210) 
v_add_f32 v13, v21, v17 // (op 13210) 


v_cvt_f16_f32 v14, v10 // (op 13211) 
v_cvt_f16_f32 v15, v11 // (op 13211) 
v_cvt_f16_f32 v16, v12 // (op 13211) 
v_cvt_f16_f32 v17, v13 // (op 13211) 


v_pack_b32_f16 v10, v14, v15 // (op 13192) 
v_pack_b32_f16 v11, v16, v17 // (op 13192) 

s_lshl_b32 s0, s4, 5 // (op 13192) 
s_add_u32 s0, s0, 128 // (op 13192) 
v_add_u32 v3, v9, s0 // (op 13192) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13192) Store value





s_lshl_b32 s0, s2, 6 // (op 13217) 
s_add_u32 s0, s0, 128 // (op 13217) 
v_add_u32 v3, v8, s0 // (op 13217) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13217) Load value

v_accvgpr_read v10, a136 // (op 13246) 
v_accvgpr_read v11, a137 // (op 13246) 
v_accvgpr_read v12, a138 // (op 13246) 
v_accvgpr_read v13, a139 // (op 13246) 
v_mul_f32 v14, v7, v10 // (op 13246) 
v_mul_f32 v15, v7, v11 // (op 13246) 
v_mul_f32 v16, v7, v12 // (op 13246) 
v_mul_f32 v17, v7, v13 // (op 13246) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13214) 
v_lshrrev_b32 v11, 16, v4 // (op 13214) 
v_cvt_f32_f16 v11, v11 // (op 13214) 
v_mul_f32 v18, v0, v10 // (op 13214) 
v_mul_f32 v19, v0, v11 // (op 13214) 
v_cvt_f32_f16 v10, v5 // (op 13214) 
v_lshrrev_b32 v11, 16, v5 // (op 13214) 
v_cvt_f32_f16 v11, v11 // (op 13214) 
v_mul_f32 v20, v0, v10 // (op 13214) 
v_mul_f32 v21, v0, v11 // (op 13214) 


v_add_f32 v10, v18, v14 // (op 13215) 
v_add_f32 v11, v19, v15 // (op 13215) 
v_add_f32 v12, v20, v16 // (op 13215) 
v_add_f32 v13, v21, v17 // (op 13215) 


v_cvt_f16_f32 v14, v10 // (op 13216) 
v_cvt_f16_f32 v15, v11 // (op 13216) 
v_cvt_f16_f32 v16, v12 // (op 13216) 
v_cvt_f16_f32 v17, v13 // (op 13216) 


v_pack_b32_f16 v10, v14, v15 // (op 13194) 
v_pack_b32_f16 v11, v16, v17 // (op 13194) 

s_lshl_b32 s0, s4, 6 // (op 13194) 
s_add_u32 s0, s0, 128 // (op 13194) 
v_add_u32 v3, v9, s0 // (op 13194) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13194) Store value





s_lshl_b32 s0, s2, 5 // (op 13222) 
s_mul_i32 s1, 3, s0 // (op 13222) 
s_add_u32 s1, s1, 128 // (op 13222) 

v_add_u32 v3, v8, s1 // (op 13222) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13222) Load value

v_accvgpr_read v10, a140 // (op 13248) 
v_accvgpr_read v11, a141 // (op 13248) 
v_accvgpr_read v12, a142 // (op 13248) 
v_accvgpr_read v13, a143 // (op 13248) 
v_mul_f32 v14, v7, v10 // (op 13248) 
v_mul_f32 v15, v7, v11 // (op 13248) 
v_mul_f32 v16, v7, v12 // (op 13248) 
v_mul_f32 v17, v7, v13 // (op 13248) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13219) 
v_lshrrev_b32 v11, 16, v4 // (op 13219) 
v_cvt_f32_f16 v11, v11 // (op 13219) 
v_mul_f32 v18, v0, v10 // (op 13219) 
v_mul_f32 v19, v0, v11 // (op 13219) 
v_cvt_f32_f16 v10, v5 // (op 13219) 
v_lshrrev_b32 v11, 16, v5 // (op 13219) 
v_cvt_f32_f16 v11, v11 // (op 13219) 
v_mul_f32 v20, v0, v10 // (op 13219) 
v_mul_f32 v21, v0, v11 // (op 13219) 


v_add_f32 v10, v18, v14 // (op 13220) 
v_add_f32 v11, v19, v15 // (op 13220) 
v_add_f32 v12, v20, v16 // (op 13220) 
v_add_f32 v13, v21, v17 // (op 13220) 


v_cvt_f16_f32 v14, v10 // (op 13221) 
v_cvt_f16_f32 v15, v11 // (op 13221) 
v_cvt_f16_f32 v16, v12 // (op 13221) 
v_cvt_f16_f32 v17, v13 // (op 13221) 


v_pack_b32_f16 v10, v14, v15 // (op 13196) 
v_pack_b32_f16 v11, v16, v17 // (op 13196) 

s_lshl_b32 s0, s4, 5 // (op 13196) 
s_mul_i32 s1, 3, s0 // (op 13196) 
s_add_u32 s1, s1, 128 // (op 13196) 

v_add_u32 v3, v9, s1 // (op 13196) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13196) Store value





s_lshl_b32 s0, s2, 7 // (op 13227) 
s_add_u32 s0, s0, 128 // (op 13227) 
v_add_u32 v3, v8, s0 // (op 13227) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13227) Load value

v_accvgpr_read v10, a144 // (op 13250) 
v_accvgpr_read v11, a145 // (op 13250) 
v_accvgpr_read v12, a146 // (op 13250) 
v_accvgpr_read v13, a147 // (op 13250) 
v_mul_f32 v14, v7, v10 // (op 13250) 
v_mul_f32 v15, v7, v11 // (op 13250) 
v_mul_f32 v16, v7, v12 // (op 13250) 
v_mul_f32 v17, v7, v13 // (op 13250) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13224) 
v_lshrrev_b32 v11, 16, v4 // (op 13224) 
v_cvt_f32_f16 v11, v11 // (op 13224) 
v_mul_f32 v18, v0, v10 // (op 13224) 
v_mul_f32 v19, v0, v11 // (op 13224) 
v_cvt_f32_f16 v10, v5 // (op 13224) 
v_lshrrev_b32 v11, 16, v5 // (op 13224) 
v_cvt_f32_f16 v11, v11 // (op 13224) 
v_mul_f32 v20, v0, v10 // (op 13224) 
v_mul_f32 v21, v0, v11 // (op 13224) 


v_add_f32 v10, v18, v14 // (op 13225) 
v_add_f32 v11, v19, v15 // (op 13225) 
v_add_f32 v12, v20, v16 // (op 13225) 
v_add_f32 v13, v21, v17 // (op 13225) 


v_cvt_f16_f32 v14, v10 // (op 13226) 
v_cvt_f16_f32 v15, v11 // (op 13226) 
v_cvt_f16_f32 v16, v12 // (op 13226) 
v_cvt_f16_f32 v17, v13 // (op 13226) 


v_pack_b32_f16 v10, v14, v15 // (op 13198) 
v_pack_b32_f16 v11, v16, v17 // (op 13198) 

s_lshl_b32 s0, s4, 7 // (op 13198) 
s_add_u32 s0, s0, 128 // (op 13198) 
v_add_u32 v3, v9, s0 // (op 13198) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13198) Store value





s_lshl_b32 s0, s2, 5 // (op 13232) 
s_mul_i32 s1, 5, s0 // (op 13232) 
s_add_u32 s1, s1, 128 // (op 13232) 

v_add_u32 v3, v8, s1 // (op 13232) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13232) Load value

v_accvgpr_read v10, a148 // (op 13252) 
v_accvgpr_read v11, a149 // (op 13252) 
v_accvgpr_read v12, a150 // (op 13252) 
v_accvgpr_read v13, a151 // (op 13252) 
v_mul_f32 v14, v7, v10 // (op 13252) 
v_mul_f32 v15, v7, v11 // (op 13252) 
v_mul_f32 v16, v7, v12 // (op 13252) 
v_mul_f32 v17, v7, v13 // (op 13252) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13229) 
v_lshrrev_b32 v11, 16, v4 // (op 13229) 
v_cvt_f32_f16 v11, v11 // (op 13229) 
v_mul_f32 v18, v0, v10 // (op 13229) 
v_mul_f32 v19, v0, v11 // (op 13229) 
v_cvt_f32_f16 v10, v5 // (op 13229) 
v_lshrrev_b32 v11, 16, v5 // (op 13229) 
v_cvt_f32_f16 v11, v11 // (op 13229) 
v_mul_f32 v20, v0, v10 // (op 13229) 
v_mul_f32 v21, v0, v11 // (op 13229) 


v_add_f32 v10, v18, v14 // (op 13230) 
v_add_f32 v11, v19, v15 // (op 13230) 
v_add_f32 v12, v20, v16 // (op 13230) 
v_add_f32 v13, v21, v17 // (op 13230) 


v_cvt_f16_f32 v14, v10 // (op 13231) 
v_cvt_f16_f32 v15, v11 // (op 13231) 
v_cvt_f16_f32 v16, v12 // (op 13231) 
v_cvt_f16_f32 v17, v13 // (op 13231) 


v_pack_b32_f16 v10, v14, v15 // (op 13200) 
v_pack_b32_f16 v11, v16, v17 // (op 13200) 

s_lshl_b32 s0, s4, 5 // (op 13200) 
s_mul_i32 s1, 5, s0 // (op 13200) 
s_add_u32 s1, s1, 128 // (op 13200) 

v_add_u32 v3, v9, s1 // (op 13200) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13200) Store value





s_lshl_b32 s0, s2, 5 // (op 13237) 
s_mul_i32 s1, 6, s0 // (op 13237) 
s_add_u32 s1, s1, 128 // (op 13237) 

v_add_u32 v3, v8, s1 // (op 13237) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13237) Load value

v_accvgpr_read v10, a152 // (op 13254) 
v_accvgpr_read v11, a153 // (op 13254) 
v_accvgpr_read v12, a154 // (op 13254) 
v_accvgpr_read v13, a155 // (op 13254) 
v_mul_f32 v14, v7, v10 // (op 13254) 
v_mul_f32 v15, v7, v11 // (op 13254) 
v_mul_f32 v16, v7, v12 // (op 13254) 
v_mul_f32 v17, v7, v13 // (op 13254) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13234) 
v_lshrrev_b32 v11, 16, v4 // (op 13234) 
v_cvt_f32_f16 v11, v11 // (op 13234) 
v_mul_f32 v18, v0, v10 // (op 13234) 
v_mul_f32 v19, v0, v11 // (op 13234) 
v_cvt_f32_f16 v10, v5 // (op 13234) 
v_lshrrev_b32 v11, 16, v5 // (op 13234) 
v_cvt_f32_f16 v11, v11 // (op 13234) 
v_mul_f32 v20, v0, v10 // (op 13234) 
v_mul_f32 v21, v0, v11 // (op 13234) 


v_add_f32 v10, v18, v14 // (op 13235) 
v_add_f32 v11, v19, v15 // (op 13235) 
v_add_f32 v12, v20, v16 // (op 13235) 
v_add_f32 v13, v21, v17 // (op 13235) 


v_cvt_f16_f32 v14, v10 // (op 13236) 
v_cvt_f16_f32 v15, v11 // (op 13236) 
v_cvt_f16_f32 v16, v12 // (op 13236) 
v_cvt_f16_f32 v17, v13 // (op 13236) 


v_pack_b32_f16 v10, v14, v15 // (op 13202) 
v_pack_b32_f16 v11, v16, v17 // (op 13202) 

s_lshl_b32 s0, s4, 5 // (op 13202) 
s_mul_i32 s1, 6, s0 // (op 13202) 
s_add_u32 s1, s1, 128 // (op 13202) 

v_add_u32 v3, v9, s1 // (op 13202) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13202) Store value





s_lshl_b32 s0, s2, 5 // (op 13242) 
s_mul_i32 s1, 7, s0 // (op 13242) 
s_add_u32 s1, s1, 128 // (op 13242) 

v_add_u32 v3, v8, s1 // (op 13242) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13242) Load value

v_accvgpr_read v10, a156 // (op 13256) 
v_accvgpr_read v11, a157 // (op 13256) 
v_accvgpr_read v12, a158 // (op 13256) 
v_accvgpr_read v13, a159 // (op 13256) 
v_mul_f32 v14, v7, v10 // (op 13256) 
v_mul_f32 v15, v7, v11 // (op 13256) 
v_mul_f32 v16, v7, v12 // (op 13256) 
v_mul_f32 v17, v7, v13 // (op 13256) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13239) 
v_lshrrev_b32 v11, 16, v4 // (op 13239) 
v_cvt_f32_f16 v11, v11 // (op 13239) 
v_mul_f32 v18, v0, v10 // (op 13239) 
v_mul_f32 v19, v0, v11 // (op 13239) 
v_cvt_f32_f16 v10, v5 // (op 13239) 
v_lshrrev_b32 v11, 16, v5 // (op 13239) 
v_cvt_f32_f16 v11, v11 // (op 13239) 
v_mul_f32 v20, v0, v10 // (op 13239) 
v_mul_f32 v21, v0, v11 // (op 13239) 


v_add_f32 v10, v18, v14 // (op 13240) 
v_add_f32 v11, v19, v15 // (op 13240) 
v_add_f32 v12, v20, v16 // (op 13240) 
v_add_f32 v13, v21, v17 // (op 13240) 


v_cvt_f16_f32 v14, v10 // (op 13241) 
v_cvt_f16_f32 v15, v11 // (op 13241) 
v_cvt_f16_f32 v16, v12 // (op 13241) 
v_cvt_f16_f32 v17, v13 // (op 13241) 


v_pack_b32_f16 v10, v14, v15 // (op 13204) 
v_pack_b32_f16 v11, v16, v17 // (op 13204) 

s_lshl_b32 s0, s4, 5 // (op 13204) 
s_mul_i32 s1, 7, s0 // (op 13204) 
s_add_u32 s1, s1, 128 // (op 13204) 

v_add_u32 v3, v9, s1 // (op 13204) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13204) Store value





v_add_u32 v3, 96, v8 // (op 13036) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13036) Load value

v_accvgpr_read v10, a96 // (op 13015) 
v_accvgpr_read v11, a97 // (op 13015) 
v_accvgpr_read v12, a98 // (op 13015) 
v_accvgpr_read v13, a99 // (op 13015) 
v_mul_f32 v14, v7, v10 // (op 13015) 
v_mul_f32 v15, v7, v11 // (op 13015) 
v_mul_f32 v16, v7, v12 // (op 13015) 
v_mul_f32 v17, v7, v13 // (op 13015) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13035) 
v_lshrrev_b32 v11, 16, v4 // (op 13035) 
v_cvt_f32_f16 v11, v11 // (op 13035) 
v_mul_f32 v18, v0, v10 // (op 13035) 
v_mul_f32 v19, v0, v11 // (op 13035) 
v_cvt_f32_f16 v10, v5 // (op 13035) 
v_lshrrev_b32 v11, 16, v5 // (op 13035) 
v_cvt_f32_f16 v11, v11 // (op 13035) 
v_mul_f32 v20, v0, v10 // (op 13035) 
v_mul_f32 v21, v0, v11 // (op 13035) 


v_add_f32 v10, v18, v14 // (op 13016) 
v_add_f32 v11, v19, v15 // (op 13016) 
v_add_f32 v12, v20, v16 // (op 13016) 
v_add_f32 v13, v21, v17 // (op 13016) 


v_cvt_f16_f32 v14, v10 // (op 13017) 
v_cvt_f16_f32 v15, v11 // (op 13017) 
v_cvt_f16_f32 v16, v12 // (op 13017) 
v_cvt_f16_f32 v17, v13 // (op 13017) 


v_pack_b32_f16 v10, v14, v15 // (op 13019) 
v_pack_b32_f16 v11, v16, v17 // (op 13019) 

v_add_u32 v3, 96, v9 // (op 13019) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13019) Store value





s_lshl_b32 s0, s2, 5 // (op 13041) 
s_add_u32 s0, s0, 96 // (op 13041) 
v_add_u32 v3, v8, s0 // (op 13041) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13041) Load value

v_accvgpr_read v10, a100 // (op 13073) 
v_accvgpr_read v11, a101 // (op 13073) 
v_accvgpr_read v12, a102 // (op 13073) 
v_accvgpr_read v13, a103 // (op 13073) 
v_mul_f32 v14, v7, v10 // (op 13073) 
v_mul_f32 v15, v7, v11 // (op 13073) 
v_mul_f32 v16, v7, v12 // (op 13073) 
v_mul_f32 v17, v7, v13 // (op 13073) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13038) 
v_lshrrev_b32 v11, 16, v4 // (op 13038) 
v_cvt_f32_f16 v11, v11 // (op 13038) 
v_mul_f32 v18, v0, v10 // (op 13038) 
v_mul_f32 v19, v0, v11 // (op 13038) 
v_cvt_f32_f16 v10, v5 // (op 13038) 
v_lshrrev_b32 v11, 16, v5 // (op 13038) 
v_cvt_f32_f16 v11, v11 // (op 13038) 
v_mul_f32 v20, v0, v10 // (op 13038) 
v_mul_f32 v21, v0, v11 // (op 13038) 


v_add_f32 v10, v18, v14 // (op 13039) 
v_add_f32 v11, v19, v15 // (op 13039) 
v_add_f32 v12, v20, v16 // (op 13039) 
v_add_f32 v13, v21, v17 // (op 13039) 


v_cvt_f16_f32 v14, v10 // (op 13040) 
v_cvt_f16_f32 v15, v11 // (op 13040) 
v_cvt_f16_f32 v16, v12 // (op 13040) 
v_cvt_f16_f32 v17, v13 // (op 13040) 


v_pack_b32_f16 v10, v14, v15 // (op 13021) 
v_pack_b32_f16 v11, v16, v17 // (op 13021) 

s_lshl_b32 s0, s4, 5 // (op 13021) 
s_add_u32 s0, s0, 96 // (op 13021) 
v_add_u32 v3, v9, s0 // (op 13021) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13021) Store value





s_lshl_b32 s0, s2, 6 // (op 13046) 
s_add_u32 s0, s0, 96 // (op 13046) 
v_add_u32 v3, v8, s0 // (op 13046) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13046) Load value

v_accvgpr_read v10, a104 // (op 13075) 
v_accvgpr_read v11, a105 // (op 13075) 
v_accvgpr_read v12, a106 // (op 13075) 
v_accvgpr_read v13, a107 // (op 13075) 
v_mul_f32 v14, v7, v10 // (op 13075) 
v_mul_f32 v15, v7, v11 // (op 13075) 
v_mul_f32 v16, v7, v12 // (op 13075) 
v_mul_f32 v17, v7, v13 // (op 13075) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13043) 
v_lshrrev_b32 v11, 16, v4 // (op 13043) 
v_cvt_f32_f16 v11, v11 // (op 13043) 
v_mul_f32 v18, v0, v10 // (op 13043) 
v_mul_f32 v19, v0, v11 // (op 13043) 
v_cvt_f32_f16 v10, v5 // (op 13043) 
v_lshrrev_b32 v11, 16, v5 // (op 13043) 
v_cvt_f32_f16 v11, v11 // (op 13043) 
v_mul_f32 v20, v0, v10 // (op 13043) 
v_mul_f32 v21, v0, v11 // (op 13043) 


v_add_f32 v10, v18, v14 // (op 13044) 
v_add_f32 v11, v19, v15 // (op 13044) 
v_add_f32 v12, v20, v16 // (op 13044) 
v_add_f32 v13, v21, v17 // (op 13044) 


v_cvt_f16_f32 v14, v10 // (op 13045) 
v_cvt_f16_f32 v15, v11 // (op 13045) 
v_cvt_f16_f32 v16, v12 // (op 13045) 
v_cvt_f16_f32 v17, v13 // (op 13045) 


v_pack_b32_f16 v10, v14, v15 // (op 13023) 
v_pack_b32_f16 v11, v16, v17 // (op 13023) 

s_lshl_b32 s0, s4, 6 // (op 13023) 
s_add_u32 s0, s0, 96 // (op 13023) 
v_add_u32 v3, v9, s0 // (op 13023) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13023) Store value





s_lshl_b32 s0, s2, 5 // (op 13051) 
s_mul_i32 s1, 3, s0 // (op 13051) 
s_add_u32 s1, s1, 96 // (op 13051) 

v_add_u32 v3, v8, s1 // (op 13051) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13051) Load value

v_accvgpr_read v10, a108 // (op 13077) 
v_accvgpr_read v11, a109 // (op 13077) 
v_accvgpr_read v12, a110 // (op 13077) 
v_accvgpr_read v13, a111 // (op 13077) 
v_mul_f32 v14, v7, v10 // (op 13077) 
v_mul_f32 v15, v7, v11 // (op 13077) 
v_mul_f32 v16, v7, v12 // (op 13077) 
v_mul_f32 v17, v7, v13 // (op 13077) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13048) 
v_lshrrev_b32 v11, 16, v4 // (op 13048) 
v_cvt_f32_f16 v11, v11 // (op 13048) 
v_mul_f32 v18, v0, v10 // (op 13048) 
v_mul_f32 v19, v0, v11 // (op 13048) 
v_cvt_f32_f16 v10, v5 // (op 13048) 
v_lshrrev_b32 v11, 16, v5 // (op 13048) 
v_cvt_f32_f16 v11, v11 // (op 13048) 
v_mul_f32 v20, v0, v10 // (op 13048) 
v_mul_f32 v21, v0, v11 // (op 13048) 


v_add_f32 v10, v18, v14 // (op 13049) 
v_add_f32 v11, v19, v15 // (op 13049) 
v_add_f32 v12, v20, v16 // (op 13049) 
v_add_f32 v13, v21, v17 // (op 13049) 


v_cvt_f16_f32 v14, v10 // (op 13050) 
v_cvt_f16_f32 v15, v11 // (op 13050) 
v_cvt_f16_f32 v16, v12 // (op 13050) 
v_cvt_f16_f32 v17, v13 // (op 13050) 


v_pack_b32_f16 v10, v14, v15 // (op 13025) 
v_pack_b32_f16 v11, v16, v17 // (op 13025) 

s_lshl_b32 s0, s4, 5 // (op 13025) 
s_mul_i32 s1, 3, s0 // (op 13025) 
s_add_u32 s1, s1, 96 // (op 13025) 

v_add_u32 v3, v9, s1 // (op 13025) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13025) Store value





s_lshl_b32 s0, s2, 7 // (op 13056) 
s_add_u32 s0, s0, 96 // (op 13056) 
v_add_u32 v3, v8, s0 // (op 13056) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13056) Load value

v_accvgpr_read v10, a112 // (op 13079) 
v_accvgpr_read v11, a113 // (op 13079) 
v_accvgpr_read v12, a114 // (op 13079) 
v_accvgpr_read v13, a115 // (op 13079) 
v_mul_f32 v14, v7, v10 // (op 13079) 
v_mul_f32 v15, v7, v11 // (op 13079) 
v_mul_f32 v16, v7, v12 // (op 13079) 
v_mul_f32 v17, v7, v13 // (op 13079) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13053) 
v_lshrrev_b32 v11, 16, v4 // (op 13053) 
v_cvt_f32_f16 v11, v11 // (op 13053) 
v_mul_f32 v18, v0, v10 // (op 13053) 
v_mul_f32 v19, v0, v11 // (op 13053) 
v_cvt_f32_f16 v10, v5 // (op 13053) 
v_lshrrev_b32 v11, 16, v5 // (op 13053) 
v_cvt_f32_f16 v11, v11 // (op 13053) 
v_mul_f32 v20, v0, v10 // (op 13053) 
v_mul_f32 v21, v0, v11 // (op 13053) 


v_add_f32 v10, v18, v14 // (op 13054) 
v_add_f32 v11, v19, v15 // (op 13054) 
v_add_f32 v12, v20, v16 // (op 13054) 
v_add_f32 v13, v21, v17 // (op 13054) 


v_cvt_f16_f32 v14, v10 // (op 13055) 
v_cvt_f16_f32 v15, v11 // (op 13055) 
v_cvt_f16_f32 v16, v12 // (op 13055) 
v_cvt_f16_f32 v17, v13 // (op 13055) 


v_pack_b32_f16 v10, v14, v15 // (op 13027) 
v_pack_b32_f16 v11, v16, v17 // (op 13027) 

s_lshl_b32 s0, s4, 7 // (op 13027) 
s_add_u32 s0, s0, 96 // (op 13027) 
v_add_u32 v3, v9, s0 // (op 13027) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13027) Store value





s_lshl_b32 s0, s2, 5 // (op 13061) 
s_mul_i32 s1, 5, s0 // (op 13061) 
s_add_u32 s1, s1, 96 // (op 13061) 

v_add_u32 v3, v8, s1 // (op 13061) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13061) Load value

v_accvgpr_read v10, a116 // (op 13081) 
v_accvgpr_read v11, a117 // (op 13081) 
v_accvgpr_read v12, a118 // (op 13081) 
v_accvgpr_read v13, a119 // (op 13081) 
v_mul_f32 v14, v7, v10 // (op 13081) 
v_mul_f32 v15, v7, v11 // (op 13081) 
v_mul_f32 v16, v7, v12 // (op 13081) 
v_mul_f32 v17, v7, v13 // (op 13081) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13058) 
v_lshrrev_b32 v11, 16, v4 // (op 13058) 
v_cvt_f32_f16 v11, v11 // (op 13058) 
v_mul_f32 v18, v0, v10 // (op 13058) 
v_mul_f32 v19, v0, v11 // (op 13058) 
v_cvt_f32_f16 v10, v5 // (op 13058) 
v_lshrrev_b32 v11, 16, v5 // (op 13058) 
v_cvt_f32_f16 v11, v11 // (op 13058) 
v_mul_f32 v20, v0, v10 // (op 13058) 
v_mul_f32 v21, v0, v11 // (op 13058) 


v_add_f32 v10, v18, v14 // (op 13059) 
v_add_f32 v11, v19, v15 // (op 13059) 
v_add_f32 v12, v20, v16 // (op 13059) 
v_add_f32 v13, v21, v17 // (op 13059) 


v_cvt_f16_f32 v14, v10 // (op 13060) 
v_cvt_f16_f32 v15, v11 // (op 13060) 
v_cvt_f16_f32 v16, v12 // (op 13060) 
v_cvt_f16_f32 v17, v13 // (op 13060) 


v_pack_b32_f16 v10, v14, v15 // (op 13029) 
v_pack_b32_f16 v11, v16, v17 // (op 13029) 

s_lshl_b32 s0, s4, 5 // (op 13029) 
s_mul_i32 s1, 5, s0 // (op 13029) 
s_add_u32 s1, s1, 96 // (op 13029) 

v_add_u32 v3, v9, s1 // (op 13029) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13029) Store value





s_lshl_b32 s0, s2, 5 // (op 13066) 
s_mul_i32 s1, 6, s0 // (op 13066) 
s_add_u32 s1, s1, 96 // (op 13066) 

v_add_u32 v3, v8, s1 // (op 13066) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13066) Load value

v_accvgpr_read v10, a120 // (op 13083) 
v_accvgpr_read v11, a121 // (op 13083) 
v_accvgpr_read v12, a122 // (op 13083) 
v_accvgpr_read v13, a123 // (op 13083) 
v_mul_f32 v14, v7, v10 // (op 13083) 
v_mul_f32 v15, v7, v11 // (op 13083) 
v_mul_f32 v16, v7, v12 // (op 13083) 
v_mul_f32 v17, v7, v13 // (op 13083) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13063) 
v_lshrrev_b32 v11, 16, v4 // (op 13063) 
v_cvt_f32_f16 v11, v11 // (op 13063) 
v_mul_f32 v18, v0, v10 // (op 13063) 
v_mul_f32 v19, v0, v11 // (op 13063) 
v_cvt_f32_f16 v10, v5 // (op 13063) 
v_lshrrev_b32 v11, 16, v5 // (op 13063) 
v_cvt_f32_f16 v11, v11 // (op 13063) 
v_mul_f32 v20, v0, v10 // (op 13063) 
v_mul_f32 v21, v0, v11 // (op 13063) 


v_add_f32 v10, v18, v14 // (op 13064) 
v_add_f32 v11, v19, v15 // (op 13064) 
v_add_f32 v12, v20, v16 // (op 13064) 
v_add_f32 v13, v21, v17 // (op 13064) 


v_cvt_f16_f32 v14, v10 // (op 13065) 
v_cvt_f16_f32 v15, v11 // (op 13065) 
v_cvt_f16_f32 v16, v12 // (op 13065) 
v_cvt_f16_f32 v17, v13 // (op 13065) 


v_pack_b32_f16 v10, v14, v15 // (op 13031) 
v_pack_b32_f16 v11, v16, v17 // (op 13031) 

s_lshl_b32 s0, s4, 5 // (op 13031) 
s_mul_i32 s1, 6, s0 // (op 13031) 
s_add_u32 s1, s1, 96 // (op 13031) 

v_add_u32 v3, v9, s1 // (op 13031) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13031) Store value





s_lshl_b32 s0, s2, 5 // (op 13071) 
s_mul_i32 s1, 7, s0 // (op 13071) 
s_add_u32 s1, s1, 96 // (op 13071) 

v_add_u32 v3, v8, s1 // (op 13071) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13071) Load value

v_accvgpr_read v10, a124 // (op 13085) 
v_accvgpr_read v11, a125 // (op 13085) 
v_accvgpr_read v12, a126 // (op 13085) 
v_accvgpr_read v13, a127 // (op 13085) 
v_mul_f32 v14, v7, v10 // (op 13085) 
v_mul_f32 v15, v7, v11 // (op 13085) 
v_mul_f32 v16, v7, v12 // (op 13085) 
v_mul_f32 v17, v7, v13 // (op 13085) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13068) 
v_lshrrev_b32 v11, 16, v4 // (op 13068) 
v_cvt_f32_f16 v11, v11 // (op 13068) 
v_mul_f32 v18, v0, v10 // (op 13068) 
v_mul_f32 v19, v0, v11 // (op 13068) 
v_cvt_f32_f16 v10, v5 // (op 13068) 
v_lshrrev_b32 v11, 16, v5 // (op 13068) 
v_cvt_f32_f16 v11, v11 // (op 13068) 
v_mul_f32 v20, v0, v10 // (op 13068) 
v_mul_f32 v21, v0, v11 // (op 13068) 


v_add_f32 v10, v18, v14 // (op 13069) 
v_add_f32 v11, v19, v15 // (op 13069) 
v_add_f32 v12, v20, v16 // (op 13069) 
v_add_f32 v13, v21, v17 // (op 13069) 


v_cvt_f16_f32 v14, v10 // (op 13070) 
v_cvt_f16_f32 v15, v11 // (op 13070) 
v_cvt_f16_f32 v16, v12 // (op 13070) 
v_cvt_f16_f32 v17, v13 // (op 13070) 


v_pack_b32_f16 v10, v14, v15 // (op 13033) 
v_pack_b32_f16 v11, v16, v17 // (op 13033) 

s_lshl_b32 s0, s4, 5 // (op 13033) 
s_mul_i32 s1, 7, s0 // (op 13033) 
s_add_u32 s1, s1, 96 // (op 13033) 

v_add_u32 v3, v9, s1 // (op 13033) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13033) Store value





v_add_u32 v3, 64, v8 // (op 12865) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12865) Load value

v_accvgpr_read v10, a64 // (op 12844) 
v_accvgpr_read v11, a65 // (op 12844) 
v_accvgpr_read v12, a66 // (op 12844) 
v_accvgpr_read v13, a67 // (op 12844) 
v_mul_f32 v14, v7, v10 // (op 12844) 
v_mul_f32 v15, v7, v11 // (op 12844) 
v_mul_f32 v16, v7, v12 // (op 12844) 
v_mul_f32 v17, v7, v13 // (op 12844) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12864) 
v_lshrrev_b32 v11, 16, v4 // (op 12864) 
v_cvt_f32_f16 v11, v11 // (op 12864) 
v_mul_f32 v18, v0, v10 // (op 12864) 
v_mul_f32 v19, v0, v11 // (op 12864) 
v_cvt_f32_f16 v10, v5 // (op 12864) 
v_lshrrev_b32 v11, 16, v5 // (op 12864) 
v_cvt_f32_f16 v11, v11 // (op 12864) 
v_mul_f32 v20, v0, v10 // (op 12864) 
v_mul_f32 v21, v0, v11 // (op 12864) 


v_add_f32 v10, v18, v14 // (op 12845) 
v_add_f32 v11, v19, v15 // (op 12845) 
v_add_f32 v12, v20, v16 // (op 12845) 
v_add_f32 v13, v21, v17 // (op 12845) 


v_cvt_f16_f32 v14, v10 // (op 12846) 
v_cvt_f16_f32 v15, v11 // (op 12846) 
v_cvt_f16_f32 v16, v12 // (op 12846) 
v_cvt_f16_f32 v17, v13 // (op 12846) 


v_pack_b32_f16 v10, v14, v15 // (op 12848) 
v_pack_b32_f16 v11, v16, v17 // (op 12848) 

v_add_u32 v3, 64, v9 // (op 12848) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12848) Store value





s_lshl_b32 s0, s2, 5 // (op 12870) 
s_add_u32 s0, s0, 64 // (op 12870) 
v_add_u32 v3, v8, s0 // (op 12870) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12870) Load value

v_accvgpr_read v10, a68 // (op 12902) 
v_accvgpr_read v11, a69 // (op 12902) 
v_accvgpr_read v12, a70 // (op 12902) 
v_accvgpr_read v13, a71 // (op 12902) 
v_mul_f32 v14, v7, v10 // (op 12902) 
v_mul_f32 v15, v7, v11 // (op 12902) 
v_mul_f32 v16, v7, v12 // (op 12902) 
v_mul_f32 v17, v7, v13 // (op 12902) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12867) 
v_lshrrev_b32 v11, 16, v4 // (op 12867) 
v_cvt_f32_f16 v11, v11 // (op 12867) 
v_mul_f32 v18, v0, v10 // (op 12867) 
v_mul_f32 v19, v0, v11 // (op 12867) 
v_cvt_f32_f16 v10, v5 // (op 12867) 
v_lshrrev_b32 v11, 16, v5 // (op 12867) 
v_cvt_f32_f16 v11, v11 // (op 12867) 
v_mul_f32 v20, v0, v10 // (op 12867) 
v_mul_f32 v21, v0, v11 // (op 12867) 


v_add_f32 v10, v18, v14 // (op 12868) 
v_add_f32 v11, v19, v15 // (op 12868) 
v_add_f32 v12, v20, v16 // (op 12868) 
v_add_f32 v13, v21, v17 // (op 12868) 


v_cvt_f16_f32 v14, v10 // (op 12869) 
v_cvt_f16_f32 v15, v11 // (op 12869) 
v_cvt_f16_f32 v16, v12 // (op 12869) 
v_cvt_f16_f32 v17, v13 // (op 12869) 


v_pack_b32_f16 v10, v14, v15 // (op 12850) 
v_pack_b32_f16 v11, v16, v17 // (op 12850) 

s_lshl_b32 s0, s4, 5 // (op 12850) 
s_add_u32 s0, s0, 64 // (op 12850) 
v_add_u32 v3, v9, s0 // (op 12850) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12850) Store value





s_lshl_b32 s0, s2, 6 // (op 12875) 
s_add_u32 s0, s0, 64 // (op 12875) 
v_add_u32 v3, v8, s0 // (op 12875) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12875) Load value

v_accvgpr_read v10, a72 // (op 12904) 
v_accvgpr_read v11, a73 // (op 12904) 
v_accvgpr_read v12, a74 // (op 12904) 
v_accvgpr_read v13, a75 // (op 12904) 
v_mul_f32 v14, v7, v10 // (op 12904) 
v_mul_f32 v15, v7, v11 // (op 12904) 
v_mul_f32 v16, v7, v12 // (op 12904) 
v_mul_f32 v17, v7, v13 // (op 12904) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12872) 
v_lshrrev_b32 v11, 16, v4 // (op 12872) 
v_cvt_f32_f16 v11, v11 // (op 12872) 
v_mul_f32 v18, v0, v10 // (op 12872) 
v_mul_f32 v19, v0, v11 // (op 12872) 
v_cvt_f32_f16 v10, v5 // (op 12872) 
v_lshrrev_b32 v11, 16, v5 // (op 12872) 
v_cvt_f32_f16 v11, v11 // (op 12872) 
v_mul_f32 v20, v0, v10 // (op 12872) 
v_mul_f32 v21, v0, v11 // (op 12872) 


v_add_f32 v10, v18, v14 // (op 12873) 
v_add_f32 v11, v19, v15 // (op 12873) 
v_add_f32 v12, v20, v16 // (op 12873) 
v_add_f32 v13, v21, v17 // (op 12873) 


v_cvt_f16_f32 v14, v10 // (op 12874) 
v_cvt_f16_f32 v15, v11 // (op 12874) 
v_cvt_f16_f32 v16, v12 // (op 12874) 
v_cvt_f16_f32 v17, v13 // (op 12874) 


v_pack_b32_f16 v10, v14, v15 // (op 12852) 
v_pack_b32_f16 v11, v16, v17 // (op 12852) 

s_lshl_b32 s0, s4, 6 // (op 12852) 
s_add_u32 s0, s0, 64 // (op 12852) 
v_add_u32 v3, v9, s0 // (op 12852) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12852) Store value





s_lshl_b32 s0, s2, 5 // (op 12880) 
s_mul_i32 s1, 3, s0 // (op 12880) 
s_add_u32 s1, s1, 64 // (op 12880) 

v_add_u32 v3, v8, s1 // (op 12880) 


v_accvgpr_read v10, a76 // (op 12906) 
v_accvgpr_read v11, a77 // (op 12906) 
v_accvgpr_read v12, a78 // (op 12906) 
v_accvgpr_read v13, a79 // (op 12906) 
v_mul_f32 v14, v7, v10 // (op 12906) 
v_mul_f32 v15, v7, v11 // (op 12906) 
v_mul_f32 v16, v7, v12 // (op 12906) 
v_mul_f32 v17, v7, v13 // (op 12906) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12880) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12877) 
v_lshrrev_b32 v11, 16, v4 // (op 12877) 
v_cvt_f32_f16 v11, v11 // (op 12877) 
v_mul_f32 v18, v0, v10 // (op 12877) 
v_mul_f32 v19, v0, v11 // (op 12877) 
v_cvt_f32_f16 v10, v5 // (op 12877) 
v_lshrrev_b32 v11, 16, v5 // (op 12877) 
v_cvt_f32_f16 v11, v11 // (op 12877) 
v_mul_f32 v20, v0, v10 // (op 12877) 
v_mul_f32 v21, v0, v11 // (op 12877) 


v_add_f32 v10, v18, v14 // (op 12878) 
v_add_f32 v11, v19, v15 // (op 12878) 
v_add_f32 v12, v20, v16 // (op 12878) 
v_add_f32 v13, v21, v17 // (op 12878) 


v_cvt_f16_f32 v14, v10 // (op 12879) 
v_cvt_f16_f32 v15, v11 // (op 12879) 
v_cvt_f16_f32 v16, v12 // (op 12879) 
v_cvt_f16_f32 v17, v13 // (op 12879) 


v_pack_b32_f16 v10, v14, v15 // (op 12854) 
v_pack_b32_f16 v11, v16, v17 // (op 12854) 

s_lshl_b32 s0, s4, 5 // (op 12854) 
s_mul_i32 s1, 3, s0 // (op 12854) 
s_add_u32 s1, s1, 64 // (op 12854) 

v_add_u32 v3, v9, s1 // (op 12854) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12854) Store value





s_lshl_b32 s0, s2, 7 // (op 12885) 
s_add_u32 s0, s0, 64 // (op 12885) 
v_add_u32 v3, v8, s0 // (op 12885) 


v_accvgpr_read v10, a80 // (op 12908) 
v_accvgpr_read v11, a81 // (op 12908) 
v_accvgpr_read v12, a82 // (op 12908) 
v_accvgpr_read v13, a83 // (op 12908) 
v_mul_f32 v14, v7, v10 // (op 12908) 
v_mul_f32 v15, v7, v11 // (op 12908) 
v_mul_f32 v16, v7, v12 // (op 12908) 
v_mul_f32 v17, v7, v13 // (op 12908) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12885) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12882) 
v_lshrrev_b32 v11, 16, v4 // (op 12882) 
v_cvt_f32_f16 v11, v11 // (op 12882) 
v_mul_f32 v18, v0, v10 // (op 12882) 
v_mul_f32 v19, v0, v11 // (op 12882) 
v_cvt_f32_f16 v10, v5 // (op 12882) 
v_lshrrev_b32 v11, 16, v5 // (op 12882) 
v_cvt_f32_f16 v11, v11 // (op 12882) 
v_mul_f32 v20, v0, v10 // (op 12882) 
v_mul_f32 v21, v0, v11 // (op 12882) 


v_add_f32 v10, v18, v14 // (op 12883) 
v_add_f32 v11, v19, v15 // (op 12883) 
v_add_f32 v12, v20, v16 // (op 12883) 
v_add_f32 v13, v21, v17 // (op 12883) 


v_cvt_f16_f32 v14, v10 // (op 12884) 
v_cvt_f16_f32 v15, v11 // (op 12884) 
v_cvt_f16_f32 v16, v12 // (op 12884) 
v_cvt_f16_f32 v17, v13 // (op 12884) 


v_pack_b32_f16 v10, v14, v15 // (op 12856) 
v_pack_b32_f16 v11, v16, v17 // (op 12856) 

s_lshl_b32 s0, s4, 7 // (op 12856) 
s_add_u32 s0, s0, 64 // (op 12856) 
v_add_u32 v3, v9, s0 // (op 12856) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12856) Store value





s_lshl_b32 s0, s2, 5 // (op 12890) 
s_mul_i32 s1, 5, s0 // (op 12890) 
s_add_u32 s1, s1, 64 // (op 12890) 

v_add_u32 v3, v8, s1 // (op 12890) 


v_accvgpr_read v10, a84 // (op 12910) 
v_accvgpr_read v11, a85 // (op 12910) 
v_accvgpr_read v12, a86 // (op 12910) 
v_accvgpr_read v13, a87 // (op 12910) 
v_mul_f32 v14, v7, v10 // (op 12910) 
v_mul_f32 v15, v7, v11 // (op 12910) 
v_mul_f32 v16, v7, v12 // (op 12910) 
v_mul_f32 v17, v7, v13 // (op 12910) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12890) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12887) 
v_lshrrev_b32 v11, 16, v4 // (op 12887) 
v_cvt_f32_f16 v11, v11 // (op 12887) 
v_mul_f32 v18, v0, v10 // (op 12887) 
v_mul_f32 v19, v0, v11 // (op 12887) 
v_cvt_f32_f16 v10, v5 // (op 12887) 
v_lshrrev_b32 v11, 16, v5 // (op 12887) 
v_cvt_f32_f16 v11, v11 // (op 12887) 
v_mul_f32 v20, v0, v10 // (op 12887) 
v_mul_f32 v21, v0, v11 // (op 12887) 


v_add_f32 v10, v18, v14 // (op 12888) 
v_add_f32 v11, v19, v15 // (op 12888) 
v_add_f32 v12, v20, v16 // (op 12888) 
v_add_f32 v13, v21, v17 // (op 12888) 


v_cvt_f16_f32 v14, v10 // (op 12889) 
v_cvt_f16_f32 v15, v11 // (op 12889) 
v_cvt_f16_f32 v16, v12 // (op 12889) 
v_cvt_f16_f32 v17, v13 // (op 12889) 


v_pack_b32_f16 v10, v14, v15 // (op 12858) 
v_pack_b32_f16 v11, v16, v17 // (op 12858) 

s_lshl_b32 s0, s4, 5 // (op 12858) 
s_mul_i32 s1, 5, s0 // (op 12858) 
s_add_u32 s1, s1, 64 // (op 12858) 

v_add_u32 v3, v9, s1 // (op 12858) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12858) Store value





s_lshl_b32 s0, s2, 5 // (op 12895) 
s_mul_i32 s1, 6, s0 // (op 12895) 
s_add_u32 s1, s1, 64 // (op 12895) 

v_add_u32 v3, v8, s1 // (op 12895) 


v_accvgpr_read v10, a88 // (op 12912) 
v_accvgpr_read v11, a89 // (op 12912) 
v_accvgpr_read v12, a90 // (op 12912) 
v_accvgpr_read v13, a91 // (op 12912) 
v_mul_f32 v14, v7, v10 // (op 12912) 
v_mul_f32 v15, v7, v11 // (op 12912) 
v_mul_f32 v16, v7, v12 // (op 12912) 
v_mul_f32 v17, v7, v13 // (op 12912) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12895) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12892) 
v_lshrrev_b32 v11, 16, v4 // (op 12892) 
v_cvt_f32_f16 v11, v11 // (op 12892) 
v_mul_f32 v18, v0, v10 // (op 12892) 
v_mul_f32 v19, v0, v11 // (op 12892) 
v_cvt_f32_f16 v10, v5 // (op 12892) 
v_lshrrev_b32 v11, 16, v5 // (op 12892) 
v_cvt_f32_f16 v11, v11 // (op 12892) 
v_mul_f32 v20, v0, v10 // (op 12892) 
v_mul_f32 v21, v0, v11 // (op 12892) 


v_add_f32 v10, v18, v14 // (op 12893) 
v_add_f32 v11, v19, v15 // (op 12893) 
v_add_f32 v12, v20, v16 // (op 12893) 
v_add_f32 v13, v21, v17 // (op 12893) 


v_cvt_f16_f32 v14, v10 // (op 12894) 
v_cvt_f16_f32 v15, v11 // (op 12894) 
v_cvt_f16_f32 v16, v12 // (op 12894) 
v_cvt_f16_f32 v17, v13 // (op 12894) 


v_pack_b32_f16 v10, v14, v15 // (op 12860) 
v_pack_b32_f16 v11, v16, v17 // (op 12860) 

s_lshl_b32 s0, s4, 5 // (op 12860) 
s_mul_i32 s1, 6, s0 // (op 12860) 
s_add_u32 s1, s1, 64 // (op 12860) 

v_add_u32 v3, v9, s1 // (op 12860) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12860) Store value





s_lshl_b32 s0, s2, 5 // (op 12900) 
s_mul_i32 s1, 7, s0 // (op 12900) 
s_add_u32 s1, s1, 64 // (op 12900) 

v_add_u32 v3, v8, s1 // (op 12900) 


v_accvgpr_read v10, a92 // (op 12914) 
v_accvgpr_read v11, a93 // (op 12914) 
v_accvgpr_read v12, a94 // (op 12914) 
v_accvgpr_read v13, a95 // (op 12914) 
v_mul_f32 v14, v7, v10 // (op 12914) 
v_mul_f32 v15, v7, v11 // (op 12914) 
v_mul_f32 v16, v7, v12 // (op 12914) 
v_mul_f32 v17, v7, v13 // (op 12914) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12900) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12897) 
v_lshrrev_b32 v11, 16, v4 // (op 12897) 
v_cvt_f32_f16 v11, v11 // (op 12897) 
v_mul_f32 v18, v0, v10 // (op 12897) 
v_mul_f32 v19, v0, v11 // (op 12897) 
v_cvt_f32_f16 v10, v5 // (op 12897) 
v_lshrrev_b32 v11, 16, v5 // (op 12897) 
v_cvt_f32_f16 v11, v11 // (op 12897) 
v_mul_f32 v20, v0, v10 // (op 12897) 
v_mul_f32 v21, v0, v11 // (op 12897) 


v_add_f32 v10, v18, v14 // (op 12898) 
v_add_f32 v11, v19, v15 // (op 12898) 
v_add_f32 v12, v20, v16 // (op 12898) 
v_add_f32 v13, v21, v17 // (op 12898) 


v_cvt_f16_f32 v14, v10 // (op 12899) 
v_cvt_f16_f32 v15, v11 // (op 12899) 
v_cvt_f16_f32 v16, v12 // (op 12899) 
v_cvt_f16_f32 v17, v13 // (op 12899) 


v_pack_b32_f16 v10, v14, v15 // (op 12862) 
v_pack_b32_f16 v11, v16, v17 // (op 12862) 

s_lshl_b32 s0, s4, 5 // (op 12862) 
s_mul_i32 s1, 7, s0 // (op 12862) 
s_add_u32 s1, s1, 64 // (op 12862) 

v_add_u32 v3, v9, s1 // (op 12862) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12862) Store value





v_add_u32 v3, 32, v8 // (op 12694) 

v_accvgpr_read v10, a32 // (op 12673) 
v_accvgpr_read v11, a33 // (op 12673) 
v_accvgpr_read v12, a34 // (op 12673) 
v_accvgpr_read v13, a35 // (op 12673) 
v_mul_f32 v14, v7, v10 // (op 12673) 
v_mul_f32 v15, v7, v11 // (op 12673) 
v_mul_f32 v16, v7, v12 // (op 12673) 
v_mul_f32 v17, v7, v13 // (op 12673) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12694) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12693) 
v_lshrrev_b32 v11, 16, v4 // (op 12693) 
v_cvt_f32_f16 v11, v11 // (op 12693) 
v_mul_f32 v18, v0, v10 // (op 12693) 
v_mul_f32 v19, v0, v11 // (op 12693) 
v_cvt_f32_f16 v10, v5 // (op 12693) 
v_lshrrev_b32 v11, 16, v5 // (op 12693) 
v_cvt_f32_f16 v11, v11 // (op 12693) 
v_mul_f32 v20, v0, v10 // (op 12693) 
v_mul_f32 v21, v0, v11 // (op 12693) 


v_add_f32 v10, v18, v14 // (op 12674) 
v_add_f32 v11, v19, v15 // (op 12674) 
v_add_f32 v12, v20, v16 // (op 12674) 
v_add_f32 v13, v21, v17 // (op 12674) 


v_cvt_f16_f32 v14, v10 // (op 12675) 
v_cvt_f16_f32 v15, v11 // (op 12675) 
v_cvt_f16_f32 v16, v12 // (op 12675) 
v_cvt_f16_f32 v17, v13 // (op 12675) 


v_pack_b32_f16 v10, v14, v15 // (op 12677) 
v_pack_b32_f16 v11, v16, v17 // (op 12677) 

v_add_u32 v3, 32, v9 // (op 12677) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12677) Store value





s_lshl_b32 s0, s2, 5 // (op 12699) 
s_add_u32 s0, s0, 32 // (op 12699) 
v_add_u32 v3, v8, s0 // (op 12699) 


v_accvgpr_read v10, a36 // (op 12731) 
v_accvgpr_read v11, a37 // (op 12731) 
v_accvgpr_read v12, a38 // (op 12731) 
v_accvgpr_read v13, a39 // (op 12731) 
v_mul_f32 v14, v7, v10 // (op 12731) 
v_mul_f32 v15, v7, v11 // (op 12731) 
v_mul_f32 v16, v7, v12 // (op 12731) 
v_mul_f32 v17, v7, v13 // (op 12731) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12699) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12696) 
v_lshrrev_b32 v11, 16, v4 // (op 12696) 
v_cvt_f32_f16 v11, v11 // (op 12696) 
v_mul_f32 v18, v0, v10 // (op 12696) 
v_mul_f32 v19, v0, v11 // (op 12696) 
v_cvt_f32_f16 v10, v5 // (op 12696) 
v_lshrrev_b32 v11, 16, v5 // (op 12696) 
v_cvt_f32_f16 v11, v11 // (op 12696) 
v_mul_f32 v20, v0, v10 // (op 12696) 
v_mul_f32 v21, v0, v11 // (op 12696) 


v_add_f32 v10, v18, v14 // (op 12697) 
v_add_f32 v11, v19, v15 // (op 12697) 
v_add_f32 v12, v20, v16 // (op 12697) 
v_add_f32 v13, v21, v17 // (op 12697) 


v_cvt_f16_f32 v14, v10 // (op 12698) 
v_cvt_f16_f32 v15, v11 // (op 12698) 
v_cvt_f16_f32 v16, v12 // (op 12698) 
v_cvt_f16_f32 v17, v13 // (op 12698) 


v_pack_b32_f16 v10, v14, v15 // (op 12679) 
v_pack_b32_f16 v11, v16, v17 // (op 12679) 

s_lshl_b32 s0, s4, 5 // (op 12679) 
s_add_u32 s0, s0, 32 // (op 12679) 
v_add_u32 v3, v9, s0 // (op 12679) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12679) Store value





s_lshl_b32 s0, s2, 6 // (op 12704) 
s_add_u32 s0, s0, 32 // (op 12704) 
v_add_u32 v3, v8, s0 // (op 12704) 


v_accvgpr_read v10, a40 // (op 12733) 
v_accvgpr_read v11, a41 // (op 12733) 
v_accvgpr_read v12, a42 // (op 12733) 
v_accvgpr_read v13, a43 // (op 12733) 
v_mul_f32 v14, v7, v10 // (op 12733) 
v_mul_f32 v15, v7, v11 // (op 12733) 
v_mul_f32 v16, v7, v12 // (op 12733) 
v_mul_f32 v17, v7, v13 // (op 12733) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12704) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12701) 
v_lshrrev_b32 v11, 16, v4 // (op 12701) 
v_cvt_f32_f16 v11, v11 // (op 12701) 
v_mul_f32 v18, v0, v10 // (op 12701) 
v_mul_f32 v19, v0, v11 // (op 12701) 
v_cvt_f32_f16 v10, v5 // (op 12701) 
v_lshrrev_b32 v11, 16, v5 // (op 12701) 
v_cvt_f32_f16 v11, v11 // (op 12701) 
v_mul_f32 v20, v0, v10 // (op 12701) 
v_mul_f32 v21, v0, v11 // (op 12701) 


v_add_f32 v10, v18, v14 // (op 12702) 
v_add_f32 v11, v19, v15 // (op 12702) 
v_add_f32 v12, v20, v16 // (op 12702) 
v_add_f32 v13, v21, v17 // (op 12702) 


v_cvt_f16_f32 v14, v10 // (op 12703) 
v_cvt_f16_f32 v15, v11 // (op 12703) 
v_cvt_f16_f32 v16, v12 // (op 12703) 
v_cvt_f16_f32 v17, v13 // (op 12703) 


v_pack_b32_f16 v10, v14, v15 // (op 12681) 
v_pack_b32_f16 v11, v16, v17 // (op 12681) 

s_lshl_b32 s0, s4, 6 // (op 12681) 
s_add_u32 s0, s0, 32 // (op 12681) 
v_add_u32 v3, v9, s0 // (op 12681) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12681) Store value





s_lshl_b32 s0, s2, 5 // (op 12709) 
s_mul_i32 s1, 3, s0 // (op 12709) 
s_add_u32 s1, s1, 32 // (op 12709) 

v_add_u32 v3, v8, s1 // (op 12709) 


v_accvgpr_read v10, a44 // (op 12735) 
v_accvgpr_read v11, a45 // (op 12735) 
v_accvgpr_read v12, a46 // (op 12735) 
v_accvgpr_read v13, a47 // (op 12735) 
v_mul_f32 v14, v7, v10 // (op 12735) 
v_mul_f32 v15, v7, v11 // (op 12735) 
v_mul_f32 v16, v7, v12 // (op 12735) 
v_mul_f32 v17, v7, v13 // (op 12735) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12709) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12706) 
v_lshrrev_b32 v11, 16, v4 // (op 12706) 
v_cvt_f32_f16 v11, v11 // (op 12706) 
v_mul_f32 v18, v0, v10 // (op 12706) 
v_mul_f32 v19, v0, v11 // (op 12706) 
v_cvt_f32_f16 v10, v5 // (op 12706) 
v_lshrrev_b32 v11, 16, v5 // (op 12706) 
v_cvt_f32_f16 v11, v11 // (op 12706) 
v_mul_f32 v20, v0, v10 // (op 12706) 
v_mul_f32 v21, v0, v11 // (op 12706) 


v_add_f32 v10, v18, v14 // (op 12707) 
v_add_f32 v11, v19, v15 // (op 12707) 
v_add_f32 v12, v20, v16 // (op 12707) 
v_add_f32 v13, v21, v17 // (op 12707) 


v_cvt_f16_f32 v14, v10 // (op 12708) 
v_cvt_f16_f32 v15, v11 // (op 12708) 
v_cvt_f16_f32 v16, v12 // (op 12708) 
v_cvt_f16_f32 v17, v13 // (op 12708) 


v_pack_b32_f16 v10, v14, v15 // (op 12683) 
v_pack_b32_f16 v11, v16, v17 // (op 12683) 

s_lshl_b32 s0, s4, 5 // (op 12683) 
s_mul_i32 s1, 3, s0 // (op 12683) 
s_add_u32 s1, s1, 32 // (op 12683) 

v_add_u32 v3, v9, s1 // (op 12683) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12683) Store value





s_lshl_b32 s0, s2, 7 // (op 12714) 
s_add_u32 s0, s0, 32 // (op 12714) 
v_add_u32 v3, v8, s0 // (op 12714) 


v_accvgpr_read v10, a48 // (op 12737) 
v_accvgpr_read v11, a49 // (op 12737) 
v_accvgpr_read v12, a50 // (op 12737) 
v_accvgpr_read v13, a51 // (op 12737) 
v_mul_f32 v14, v7, v10 // (op 12737) 
v_mul_f32 v15, v7, v11 // (op 12737) 
v_mul_f32 v16, v7, v12 // (op 12737) 
v_mul_f32 v17, v7, v13 // (op 12737) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12714) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12711) 
v_lshrrev_b32 v11, 16, v4 // (op 12711) 
v_cvt_f32_f16 v11, v11 // (op 12711) 
v_mul_f32 v18, v0, v10 // (op 12711) 
v_mul_f32 v19, v0, v11 // (op 12711) 
v_cvt_f32_f16 v10, v5 // (op 12711) 
v_lshrrev_b32 v11, 16, v5 // (op 12711) 
v_cvt_f32_f16 v11, v11 // (op 12711) 
v_mul_f32 v20, v0, v10 // (op 12711) 
v_mul_f32 v21, v0, v11 // (op 12711) 


v_add_f32 v10, v18, v14 // (op 12712) 
v_add_f32 v11, v19, v15 // (op 12712) 
v_add_f32 v12, v20, v16 // (op 12712) 
v_add_f32 v13, v21, v17 // (op 12712) 


v_cvt_f16_f32 v14, v10 // (op 12713) 
v_cvt_f16_f32 v15, v11 // (op 12713) 
v_cvt_f16_f32 v16, v12 // (op 12713) 
v_cvt_f16_f32 v17, v13 // (op 12713) 


v_pack_b32_f16 v10, v14, v15 // (op 12685) 
v_pack_b32_f16 v11, v16, v17 // (op 12685) 

s_lshl_b32 s0, s4, 7 // (op 12685) 
s_add_u32 s0, s0, 32 // (op 12685) 
v_add_u32 v3, v9, s0 // (op 12685) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12685) Store value





s_lshl_b32 s0, s2, 5 // (op 12719) 
s_mul_i32 s1, 5, s0 // (op 12719) 
s_add_u32 s1, s1, 32 // (op 12719) 

v_add_u32 v3, v8, s1 // (op 12719) 


v_accvgpr_read v10, a52 // (op 12739) 
v_accvgpr_read v11, a53 // (op 12739) 
v_accvgpr_read v12, a54 // (op 12739) 
v_accvgpr_read v13, a55 // (op 12739) 
v_mul_f32 v14, v7, v10 // (op 12739) 
v_mul_f32 v15, v7, v11 // (op 12739) 
v_mul_f32 v16, v7, v12 // (op 12739) 
v_mul_f32 v17, v7, v13 // (op 12739) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12719) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12716) 
v_lshrrev_b32 v11, 16, v4 // (op 12716) 
v_cvt_f32_f16 v11, v11 // (op 12716) 
v_mul_f32 v18, v0, v10 // (op 12716) 
v_mul_f32 v19, v0, v11 // (op 12716) 
v_cvt_f32_f16 v10, v5 // (op 12716) 
v_lshrrev_b32 v11, 16, v5 // (op 12716) 
v_cvt_f32_f16 v11, v11 // (op 12716) 
v_mul_f32 v20, v0, v10 // (op 12716) 
v_mul_f32 v21, v0, v11 // (op 12716) 


v_add_f32 v10, v18, v14 // (op 12717) 
v_add_f32 v11, v19, v15 // (op 12717) 
v_add_f32 v12, v20, v16 // (op 12717) 
v_add_f32 v13, v21, v17 // (op 12717) 


v_cvt_f16_f32 v14, v10 // (op 12718) 
v_cvt_f16_f32 v15, v11 // (op 12718) 
v_cvt_f16_f32 v16, v12 // (op 12718) 
v_cvt_f16_f32 v17, v13 // (op 12718) 


v_pack_b32_f16 v10, v14, v15 // (op 12687) 
v_pack_b32_f16 v11, v16, v17 // (op 12687) 

s_lshl_b32 s0, s4, 5 // (op 12687) 
s_mul_i32 s1, 5, s0 // (op 12687) 
s_add_u32 s1, s1, 32 // (op 12687) 

v_add_u32 v3, v9, s1 // (op 12687) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12687) Store value





s_lshl_b32 s0, s2, 5 // (op 12724) 
s_mul_i32 s1, 6, s0 // (op 12724) 
s_add_u32 s1, s1, 32 // (op 12724) 

v_add_u32 v3, v8, s1 // (op 12724) 


v_accvgpr_read v10, a56 // (op 12741) 
v_accvgpr_read v11, a57 // (op 12741) 
v_accvgpr_read v12, a58 // (op 12741) 
v_accvgpr_read v13, a59 // (op 12741) 
v_mul_f32 v14, v7, v10 // (op 12741) 
v_mul_f32 v15, v7, v11 // (op 12741) 
v_mul_f32 v16, v7, v12 // (op 12741) 
v_mul_f32 v17, v7, v13 // (op 12741) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12724) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12721) 
v_lshrrev_b32 v11, 16, v4 // (op 12721) 
v_cvt_f32_f16 v11, v11 // (op 12721) 
v_mul_f32 v18, v0, v10 // (op 12721) 
v_mul_f32 v19, v0, v11 // (op 12721) 
v_cvt_f32_f16 v10, v5 // (op 12721) 
v_lshrrev_b32 v11, 16, v5 // (op 12721) 
v_cvt_f32_f16 v11, v11 // (op 12721) 
v_mul_f32 v20, v0, v10 // (op 12721) 
v_mul_f32 v21, v0, v11 // (op 12721) 


v_add_f32 v10, v18, v14 // (op 12722) 
v_add_f32 v11, v19, v15 // (op 12722) 
v_add_f32 v12, v20, v16 // (op 12722) 
v_add_f32 v13, v21, v17 // (op 12722) 


v_cvt_f16_f32 v14, v10 // (op 12723) 
v_cvt_f16_f32 v15, v11 // (op 12723) 
v_cvt_f16_f32 v16, v12 // (op 12723) 
v_cvt_f16_f32 v17, v13 // (op 12723) 


v_pack_b32_f16 v10, v14, v15 // (op 12689) 
v_pack_b32_f16 v11, v16, v17 // (op 12689) 

s_lshl_b32 s0, s4, 5 // (op 12689) 
s_mul_i32 s1, 6, s0 // (op 12689) 
s_add_u32 s1, s1, 32 // (op 12689) 

v_add_u32 v3, v9, s1 // (op 12689) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12689) Store value





s_lshl_b32 s0, s2, 5 // (op 12729) 
s_mul_i32 s1, 7, s0 // (op 12729) 
s_add_u32 s1, s1, 32 // (op 12729) 

v_add_u32 v3, v8, s1 // (op 12729) 


v_accvgpr_read v10, a60 // (op 12743) 
v_accvgpr_read v11, a61 // (op 12743) 
v_accvgpr_read v12, a62 // (op 12743) 
v_accvgpr_read v13, a63 // (op 12743) 
v_mul_f32 v14, v7, v10 // (op 12743) 
v_mul_f32 v15, v7, v11 // (op 12743) 
v_mul_f32 v16, v7, v12 // (op 12743) 
v_mul_f32 v17, v7, v13 // (op 12743) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12729) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12726) 
v_lshrrev_b32 v11, 16, v4 // (op 12726) 
v_cvt_f32_f16 v11, v11 // (op 12726) 
v_mul_f32 v18, v0, v10 // (op 12726) 
v_mul_f32 v19, v0, v11 // (op 12726) 
v_cvt_f32_f16 v10, v5 // (op 12726) 
v_lshrrev_b32 v11, 16, v5 // (op 12726) 
v_cvt_f32_f16 v11, v11 // (op 12726) 
v_mul_f32 v20, v0, v10 // (op 12726) 
v_mul_f32 v21, v0, v11 // (op 12726) 


v_add_f32 v10, v18, v14 // (op 12727) 
v_add_f32 v11, v19, v15 // (op 12727) 
v_add_f32 v12, v20, v16 // (op 12727) 
v_add_f32 v13, v21, v17 // (op 12727) 


v_cvt_f16_f32 v14, v10 // (op 12728) 
v_cvt_f16_f32 v15, v11 // (op 12728) 
v_cvt_f16_f32 v16, v12 // (op 12728) 
v_cvt_f16_f32 v17, v13 // (op 12728) 


v_pack_b32_f16 v10, v14, v15 // (op 12691) 
v_pack_b32_f16 v11, v16, v17 // (op 12691) 

s_lshl_b32 s0, s4, 5 // (op 12691) 
s_mul_i32 s1, 7, s0 // (op 12691) 
s_add_u32 s1, s1, 32 // (op 12691) 

v_add_u32 v3, v9, s1 // (op 12691) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12691) Store value





v_add_u32 v3, 192, v8 // (op 13549) 

v_accvgpr_read v10, a192 // (op 13528) 
v_accvgpr_read v11, a193 // (op 13528) 
v_accvgpr_read v12, a194 // (op 13528) 
v_accvgpr_read v13, a195 // (op 13528) 
v_mul_f32 v14, v7, v10 // (op 13528) 
v_mul_f32 v15, v7, v11 // (op 13528) 
v_mul_f32 v16, v7, v12 // (op 13528) 
v_mul_f32 v17, v7, v13 // (op 13528) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13549) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13548) 
v_lshrrev_b32 v11, 16, v4 // (op 13548) 
v_cvt_f32_f16 v11, v11 // (op 13548) 
v_mul_f32 v18, v0, v10 // (op 13548) 
v_mul_f32 v19, v0, v11 // (op 13548) 
v_cvt_f32_f16 v10, v5 // (op 13548) 
v_lshrrev_b32 v11, 16, v5 // (op 13548) 
v_cvt_f32_f16 v11, v11 // (op 13548) 
v_mul_f32 v20, v0, v10 // (op 13548) 
v_mul_f32 v21, v0, v11 // (op 13548) 


v_add_f32 v10, v18, v14 // (op 13529) 
v_add_f32 v11, v19, v15 // (op 13529) 
v_add_f32 v12, v20, v16 // (op 13529) 
v_add_f32 v13, v21, v17 // (op 13529) 


v_cvt_f16_f32 v14, v10 // (op 13530) 
v_cvt_f16_f32 v15, v11 // (op 13530) 
v_cvt_f16_f32 v16, v12 // (op 13530) 
v_cvt_f16_f32 v17, v13 // (op 13530) 


v_pack_b32_f16 v10, v14, v15 // (op 13532) 
v_pack_b32_f16 v11, v16, v17 // (op 13532) 

v_add_u32 v3, 192, v9 // (op 13532) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13532) Store value





s_lshl_b32 s0, s2, 5 // (op 13554) 
s_add_u32 s0, s0, 192 // (op 13554) 
v_add_u32 v3, v8, s0 // (op 13554) 


v_accvgpr_read v10, a196 // (op 13586) 
v_accvgpr_read v11, a197 // (op 13586) 
v_accvgpr_read v12, a198 // (op 13586) 
v_accvgpr_read v13, a199 // (op 13586) 
v_mul_f32 v14, v7, v10 // (op 13586) 
v_mul_f32 v15, v7, v11 // (op 13586) 
v_mul_f32 v16, v7, v12 // (op 13586) 
v_mul_f32 v17, v7, v13 // (op 13586) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13554) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13551) 
v_lshrrev_b32 v11, 16, v4 // (op 13551) 
v_cvt_f32_f16 v11, v11 // (op 13551) 
v_mul_f32 v18, v0, v10 // (op 13551) 
v_mul_f32 v19, v0, v11 // (op 13551) 
v_cvt_f32_f16 v10, v5 // (op 13551) 
v_lshrrev_b32 v11, 16, v5 // (op 13551) 
v_cvt_f32_f16 v11, v11 // (op 13551) 
v_mul_f32 v20, v0, v10 // (op 13551) 
v_mul_f32 v21, v0, v11 // (op 13551) 


v_add_f32 v10, v18, v14 // (op 13552) 
v_add_f32 v11, v19, v15 // (op 13552) 
v_add_f32 v12, v20, v16 // (op 13552) 
v_add_f32 v13, v21, v17 // (op 13552) 


v_cvt_f16_f32 v14, v10 // (op 13553) 
v_cvt_f16_f32 v15, v11 // (op 13553) 
v_cvt_f16_f32 v16, v12 // (op 13553) 
v_cvt_f16_f32 v17, v13 // (op 13553) 


v_pack_b32_f16 v10, v14, v15 // (op 13534) 
v_pack_b32_f16 v11, v16, v17 // (op 13534) 

s_lshl_b32 s0, s4, 5 // (op 13534) 
s_add_u32 s0, s0, 192 // (op 13534) 
v_add_u32 v3, v9, s0 // (op 13534) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13534) Store value





s_lshl_b32 s0, s2, 6 // (op 13559) 
s_add_u32 s0, s0, 192 // (op 13559) 
v_add_u32 v3, v8, s0 // (op 13559) 


v_accvgpr_read v10, a200 // (op 13588) 
v_accvgpr_read v11, a201 // (op 13588) 
v_accvgpr_read v12, a202 // (op 13588) 
v_accvgpr_read v13, a203 // (op 13588) 
v_mul_f32 v14, v7, v10 // (op 13588) 
v_mul_f32 v15, v7, v11 // (op 13588) 
v_mul_f32 v16, v7, v12 // (op 13588) 
v_mul_f32 v17, v7, v13 // (op 13588) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13559) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13556) 
v_lshrrev_b32 v11, 16, v4 // (op 13556) 
v_cvt_f32_f16 v11, v11 // (op 13556) 
v_mul_f32 v18, v0, v10 // (op 13556) 
v_mul_f32 v19, v0, v11 // (op 13556) 
v_cvt_f32_f16 v10, v5 // (op 13556) 
v_lshrrev_b32 v11, 16, v5 // (op 13556) 
v_cvt_f32_f16 v11, v11 // (op 13556) 
v_mul_f32 v20, v0, v10 // (op 13556) 
v_mul_f32 v21, v0, v11 // (op 13556) 


v_add_f32 v10, v18, v14 // (op 13557) 
v_add_f32 v11, v19, v15 // (op 13557) 
v_add_f32 v12, v20, v16 // (op 13557) 
v_add_f32 v13, v21, v17 // (op 13557) 


v_cvt_f16_f32 v14, v10 // (op 13558) 
v_cvt_f16_f32 v15, v11 // (op 13558) 
v_cvt_f16_f32 v16, v12 // (op 13558) 
v_cvt_f16_f32 v17, v13 // (op 13558) 


v_pack_b32_f16 v10, v14, v15 // (op 13536) 
v_pack_b32_f16 v11, v16, v17 // (op 13536) 

s_lshl_b32 s0, s4, 6 // (op 13536) 
s_add_u32 s0, s0, 192 // (op 13536) 
v_add_u32 v3, v9, s0 // (op 13536) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13536) Store value





s_lshl_b32 s0, s2, 5 // (op 13564) 
s_mul_i32 s1, 3, s0 // (op 13564) 
s_add_u32 s1, s1, 192 // (op 13564) 

v_add_u32 v3, v8, s1 // (op 13564) 


v_accvgpr_read v10, a204 // (op 13590) 
v_accvgpr_read v11, a205 // (op 13590) 
v_accvgpr_read v12, a206 // (op 13590) 
v_accvgpr_read v13, a207 // (op 13590) 
v_mul_f32 v14, v7, v10 // (op 13590) 
v_mul_f32 v15, v7, v11 // (op 13590) 
v_mul_f32 v16, v7, v12 // (op 13590) 
v_mul_f32 v17, v7, v13 // (op 13590) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13564) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13561) 
v_lshrrev_b32 v11, 16, v4 // (op 13561) 
v_cvt_f32_f16 v11, v11 // (op 13561) 
v_mul_f32 v18, v0, v10 // (op 13561) 
v_mul_f32 v19, v0, v11 // (op 13561) 
v_cvt_f32_f16 v10, v5 // (op 13561) 
v_lshrrev_b32 v11, 16, v5 // (op 13561) 
v_cvt_f32_f16 v11, v11 // (op 13561) 
v_mul_f32 v20, v0, v10 // (op 13561) 
v_mul_f32 v21, v0, v11 // (op 13561) 


v_add_f32 v10, v18, v14 // (op 13562) 
v_add_f32 v11, v19, v15 // (op 13562) 
v_add_f32 v12, v20, v16 // (op 13562) 
v_add_f32 v13, v21, v17 // (op 13562) 


v_cvt_f16_f32 v14, v10 // (op 13563) 
v_cvt_f16_f32 v15, v11 // (op 13563) 
v_cvt_f16_f32 v16, v12 // (op 13563) 
v_cvt_f16_f32 v17, v13 // (op 13563) 


v_pack_b32_f16 v10, v14, v15 // (op 13538) 
v_pack_b32_f16 v11, v16, v17 // (op 13538) 

s_lshl_b32 s0, s4, 5 // (op 13538) 
s_mul_i32 s1, 3, s0 // (op 13538) 
s_add_u32 s1, s1, 192 // (op 13538) 

v_add_u32 v3, v9, s1 // (op 13538) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13538) Store value





s_lshl_b32 s0, s2, 7 // (op 13569) 
s_add_u32 s0, s0, 192 // (op 13569) 
v_add_u32 v3, v8, s0 // (op 13569) 


v_accvgpr_read v10, a208 // (op 13592) 
v_accvgpr_read v11, a209 // (op 13592) 
v_accvgpr_read v12, a210 // (op 13592) 
v_accvgpr_read v13, a211 // (op 13592) 
v_mul_f32 v14, v7, v10 // (op 13592) 
v_mul_f32 v15, v7, v11 // (op 13592) 
v_mul_f32 v16, v7, v12 // (op 13592) 
v_mul_f32 v17, v7, v13 // (op 13592) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13569) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13566) 
v_lshrrev_b32 v11, 16, v4 // (op 13566) 
v_cvt_f32_f16 v11, v11 // (op 13566) 
v_mul_f32 v18, v0, v10 // (op 13566) 
v_mul_f32 v19, v0, v11 // (op 13566) 
v_cvt_f32_f16 v10, v5 // (op 13566) 
v_lshrrev_b32 v11, 16, v5 // (op 13566) 
v_cvt_f32_f16 v11, v11 // (op 13566) 
v_mul_f32 v20, v0, v10 // (op 13566) 
v_mul_f32 v21, v0, v11 // (op 13566) 


v_add_f32 v10, v18, v14 // (op 13567) 
v_add_f32 v11, v19, v15 // (op 13567) 
v_add_f32 v12, v20, v16 // (op 13567) 
v_add_f32 v13, v21, v17 // (op 13567) 


v_cvt_f16_f32 v14, v10 // (op 13568) 
v_cvt_f16_f32 v15, v11 // (op 13568) 
v_cvt_f16_f32 v16, v12 // (op 13568) 
v_cvt_f16_f32 v17, v13 // (op 13568) 


v_pack_b32_f16 v10, v14, v15 // (op 13540) 
v_pack_b32_f16 v11, v16, v17 // (op 13540) 

s_lshl_b32 s0, s4, 7 // (op 13540) 
s_add_u32 s0, s0, 192 // (op 13540) 
v_add_u32 v3, v9, s0 // (op 13540) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13540) Store value





s_lshl_b32 s0, s2, 5 // (op 13574) 
s_mul_i32 s1, 5, s0 // (op 13574) 
s_add_u32 s1, s1, 192 // (op 13574) 

v_add_u32 v3, v8, s1 // (op 13574) 


v_accvgpr_read v10, a212 // (op 13594) 
v_accvgpr_read v11, a213 // (op 13594) 
v_accvgpr_read v12, a214 // (op 13594) 
v_accvgpr_read v13, a215 // (op 13594) 
v_mul_f32 v14, v7, v10 // (op 13594) 
v_mul_f32 v15, v7, v11 // (op 13594) 
v_mul_f32 v16, v7, v12 // (op 13594) 
v_mul_f32 v17, v7, v13 // (op 13594) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13574) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13571) 
v_lshrrev_b32 v11, 16, v4 // (op 13571) 
v_cvt_f32_f16 v11, v11 // (op 13571) 
v_mul_f32 v18, v0, v10 // (op 13571) 
v_mul_f32 v19, v0, v11 // (op 13571) 
v_cvt_f32_f16 v10, v5 // (op 13571) 
v_lshrrev_b32 v11, 16, v5 // (op 13571) 
v_cvt_f32_f16 v11, v11 // (op 13571) 
v_mul_f32 v20, v0, v10 // (op 13571) 
v_mul_f32 v21, v0, v11 // (op 13571) 


v_add_f32 v10, v18, v14 // (op 13572) 
v_add_f32 v11, v19, v15 // (op 13572) 
v_add_f32 v12, v20, v16 // (op 13572) 
v_add_f32 v13, v21, v17 // (op 13572) 


v_cvt_f16_f32 v14, v10 // (op 13573) 
v_cvt_f16_f32 v15, v11 // (op 13573) 
v_cvt_f16_f32 v16, v12 // (op 13573) 
v_cvt_f16_f32 v17, v13 // (op 13573) 


v_pack_b32_f16 v10, v14, v15 // (op 13542) 
v_pack_b32_f16 v11, v16, v17 // (op 13542) 

s_lshl_b32 s0, s4, 5 // (op 13542) 
s_mul_i32 s1, 5, s0 // (op 13542) 
s_add_u32 s1, s1, 192 // (op 13542) 

v_add_u32 v3, v9, s1 // (op 13542) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13542) Store value





s_lshl_b32 s0, s2, 5 // (op 13579) 
s_mul_i32 s1, 6, s0 // (op 13579) 
s_add_u32 s1, s1, 192 // (op 13579) 

v_add_u32 v3, v8, s1 // (op 13579) 


v_accvgpr_read v10, a216 // (op 13596) 
v_accvgpr_read v11, a217 // (op 13596) 
v_accvgpr_read v12, a218 // (op 13596) 
v_accvgpr_read v13, a219 // (op 13596) 
v_mul_f32 v14, v7, v10 // (op 13596) 
v_mul_f32 v15, v7, v11 // (op 13596) 
v_mul_f32 v16, v7, v12 // (op 13596) 
v_mul_f32 v17, v7, v13 // (op 13596) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13579) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13576) 
v_lshrrev_b32 v11, 16, v4 // (op 13576) 
v_cvt_f32_f16 v11, v11 // (op 13576) 
v_mul_f32 v18, v0, v10 // (op 13576) 
v_mul_f32 v19, v0, v11 // (op 13576) 
v_cvt_f32_f16 v10, v5 // (op 13576) 
v_lshrrev_b32 v11, 16, v5 // (op 13576) 
v_cvt_f32_f16 v11, v11 // (op 13576) 
v_mul_f32 v20, v0, v10 // (op 13576) 
v_mul_f32 v21, v0, v11 // (op 13576) 


v_add_f32 v10, v18, v14 // (op 13577) 
v_add_f32 v11, v19, v15 // (op 13577) 
v_add_f32 v12, v20, v16 // (op 13577) 
v_add_f32 v13, v21, v17 // (op 13577) 


v_cvt_f16_f32 v14, v10 // (op 13578) 
v_cvt_f16_f32 v15, v11 // (op 13578) 
v_cvt_f16_f32 v16, v12 // (op 13578) 
v_cvt_f16_f32 v17, v13 // (op 13578) 


v_pack_b32_f16 v10, v14, v15 // (op 13544) 
v_pack_b32_f16 v11, v16, v17 // (op 13544) 

s_lshl_b32 s0, s4, 5 // (op 13544) 
s_mul_i32 s1, 6, s0 // (op 13544) 
s_add_u32 s1, s1, 192 // (op 13544) 

v_add_u32 v3, v9, s1 // (op 13544) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13544) Store value





s_lshl_b32 s0, s2, 5 // (op 13584) 
s_mul_i32 s1, 7, s0 // (op 13584) 
s_add_u32 s1, s1, 192 // (op 13584) 

v_add_u32 v3, v8, s1 // (op 13584) 


v_accvgpr_read v10, a220 // (op 13598) 
v_accvgpr_read v11, a221 // (op 13598) 
v_accvgpr_read v12, a222 // (op 13598) 
v_accvgpr_read v13, a223 // (op 13598) 
v_mul_f32 v14, v7, v10 // (op 13598) 
v_mul_f32 v15, v7, v11 // (op 13598) 
v_mul_f32 v16, v7, v12 // (op 13598) 
v_mul_f32 v17, v7, v13 // (op 13598) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 13584) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 13581) 
v_lshrrev_b32 v11, 16, v4 // (op 13581) 
v_cvt_f32_f16 v11, v11 // (op 13581) 
v_mul_f32 v18, v0, v10 // (op 13581) 
v_mul_f32 v19, v0, v11 // (op 13581) 
v_cvt_f32_f16 v10, v5 // (op 13581) 
v_lshrrev_b32 v11, 16, v5 // (op 13581) 
v_cvt_f32_f16 v11, v11 // (op 13581) 
v_mul_f32 v20, v0, v10 // (op 13581) 
v_mul_f32 v21, v0, v11 // (op 13581) 


v_add_f32 v10, v18, v14 // (op 13582) 
v_add_f32 v11, v19, v15 // (op 13582) 
v_add_f32 v12, v20, v16 // (op 13582) 
v_add_f32 v13, v21, v17 // (op 13582) 


v_cvt_f16_f32 v14, v10 // (op 13583) 
v_cvt_f16_f32 v15, v11 // (op 13583) 
v_cvt_f16_f32 v16, v12 // (op 13583) 
v_cvt_f16_f32 v17, v13 // (op 13583) 


v_pack_b32_f16 v10, v14, v15 // (op 13546) 
v_pack_b32_f16 v11, v16, v17 // (op 13546) 

s_lshl_b32 s0, s4, 5 // (op 13546) 
s_mul_i32 s1, 7, s0 // (op 13546) 
s_add_u32 s1, s1, 192 // (op 13546) 

v_add_u32 v3, v9, s1 // (op 13546) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 13546) Store value





v_mov_b32 v3, v8 // (op 129) reg expression

v_accvgpr_read v10, a0 // (op 125) 
v_accvgpr_read v11, a1 // (op 125) 
v_accvgpr_read v12, a2 // (op 125) 
v_accvgpr_read v13, a3 // (op 125) 
v_mul_f32 v14, v7, v10 // (op 125) 
v_mul_f32 v15, v7, v11 // (op 125) 
v_mul_f32 v16, v7, v12 // (op 125) 
v_mul_f32 v17, v7, v13 // (op 125) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 129) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 130) 
v_lshrrev_b32 v11, 16, v4 // (op 130) 
v_cvt_f32_f16 v11, v11 // (op 130) 
v_mul_f32 v18, v0, v10 // (op 130) 
v_mul_f32 v19, v0, v11 // (op 130) 
v_cvt_f32_f16 v10, v5 // (op 130) 
v_lshrrev_b32 v11, 16, v5 // (op 130) 
v_cvt_f32_f16 v11, v11 // (op 130) 
v_mul_f32 v20, v0, v10 // (op 130) 
v_mul_f32 v21, v0, v11 // (op 130) 


v_add_f32 v10, v18, v14 // (op 126) 
v_add_f32 v11, v19, v15 // (op 126) 
v_add_f32 v12, v20, v16 // (op 126) 
v_add_f32 v13, v21, v17 // (op 126) 


v_cvt_f16_f32 v14, v10 // (op 127) 
v_cvt_f16_f32 v15, v11 // (op 127) 
v_cvt_f16_f32 v16, v12 // (op 127) 
v_cvt_f16_f32 v17, v13 // (op 127) 


v_pack_b32_f16 v10, v14, v15 // (op 128) 
v_pack_b32_f16 v11, v16, v17 // (op 128) 

v_mov_b32 v3, v9 // (op 128) reg expression

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 128) Store value





s_lshl_b32 s0, s2, 5 // (op 12498) 
v_add_u32 v3, v8, s0 // (op 12498) 


v_accvgpr_read v10, a4 // (op 12494) 
v_accvgpr_read v11, a5 // (op 12494) 
v_accvgpr_read v12, a6 // (op 12494) 
v_accvgpr_read v13, a7 // (op 12494) 
v_mul_f32 v14, v7, v10 // (op 12494) 
v_mul_f32 v15, v7, v11 // (op 12494) 
v_mul_f32 v16, v7, v12 // (op 12494) 
v_mul_f32 v17, v7, v13 // (op 12494) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12498) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12499) 
v_lshrrev_b32 v11, 16, v4 // (op 12499) 
v_cvt_f32_f16 v11, v11 // (op 12499) 
v_mul_f32 v18, v0, v10 // (op 12499) 
v_mul_f32 v19, v0, v11 // (op 12499) 
v_cvt_f32_f16 v10, v5 // (op 12499) 
v_lshrrev_b32 v11, 16, v5 // (op 12499) 
v_cvt_f32_f16 v11, v11 // (op 12499) 
v_mul_f32 v20, v0, v10 // (op 12499) 
v_mul_f32 v21, v0, v11 // (op 12499) 


v_add_f32 v10, v18, v14 // (op 12495) 
v_add_f32 v11, v19, v15 // (op 12495) 
v_add_f32 v12, v20, v16 // (op 12495) 
v_add_f32 v13, v21, v17 // (op 12495) 


v_cvt_f16_f32 v14, v10 // (op 12496) 
v_cvt_f16_f32 v15, v11 // (op 12496) 
v_cvt_f16_f32 v16, v12 // (op 12496) 
v_cvt_f16_f32 v17, v13 // (op 12496) 


v_pack_b32_f16 v10, v14, v15 // (op 12497) 
v_pack_b32_f16 v11, v16, v17 // (op 12497) 

s_lshl_b32 s0, s4, 5 // (op 12497) 
v_add_u32 v3, v9, s0 // (op 12497) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12497) Store value





s_lshl_b32 s0, s2, 6 // (op 12512) 
v_add_u32 v3, v8, s0 // (op 12512) 


v_accvgpr_read v10, a8 // (op 12508) 
v_accvgpr_read v11, a9 // (op 12508) 
v_accvgpr_read v12, a10 // (op 12508) 
v_accvgpr_read v13, a11 // (op 12508) 
v_mul_f32 v14, v7, v10 // (op 12508) 
v_mul_f32 v15, v7, v11 // (op 12508) 
v_mul_f32 v16, v7, v12 // (op 12508) 
v_mul_f32 v17, v7, v13 // (op 12508) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12512) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12513) 
v_lshrrev_b32 v11, 16, v4 // (op 12513) 
v_cvt_f32_f16 v11, v11 // (op 12513) 
v_mul_f32 v18, v0, v10 // (op 12513) 
v_mul_f32 v19, v0, v11 // (op 12513) 
v_cvt_f32_f16 v10, v5 // (op 12513) 
v_lshrrev_b32 v11, 16, v5 // (op 12513) 
v_cvt_f32_f16 v11, v11 // (op 12513) 
v_mul_f32 v20, v0, v10 // (op 12513) 
v_mul_f32 v21, v0, v11 // (op 12513) 


v_add_f32 v10, v18, v14 // (op 12509) 
v_add_f32 v11, v19, v15 // (op 12509) 
v_add_f32 v12, v20, v16 // (op 12509) 
v_add_f32 v13, v21, v17 // (op 12509) 


v_cvt_f16_f32 v14, v10 // (op 12510) 
v_cvt_f16_f32 v15, v11 // (op 12510) 
v_cvt_f16_f32 v16, v12 // (op 12510) 
v_cvt_f16_f32 v17, v13 // (op 12510) 


v_pack_b32_f16 v10, v14, v15 // (op 12511) 
v_pack_b32_f16 v11, v16, v17 // (op 12511) 

s_lshl_b32 s0, s4, 6 // (op 12511) 
v_add_u32 v3, v9, s0 // (op 12511) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12511) Store value





s_lshl_b32 s0, s2, 5 // (op 12526) 
s_mul_i32 s1, 3, s0 // (op 12526) 

v_add_u32 v3, v8, s1 // (op 12526) 


v_accvgpr_read v10, a12 // (op 12522) 
v_accvgpr_read v11, a13 // (op 12522) 
v_accvgpr_read v12, a14 // (op 12522) 
v_accvgpr_read v13, a15 // (op 12522) 
v_mul_f32 v14, v7, v10 // (op 12522) 
v_mul_f32 v15, v7, v11 // (op 12522) 
v_mul_f32 v16, v7, v12 // (op 12522) 
v_mul_f32 v17, v7, v13 // (op 12522) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12526) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12527) 
v_lshrrev_b32 v11, 16, v4 // (op 12527) 
v_cvt_f32_f16 v11, v11 // (op 12527) 
v_mul_f32 v18, v0, v10 // (op 12527) 
v_mul_f32 v19, v0, v11 // (op 12527) 
v_cvt_f32_f16 v10, v5 // (op 12527) 
v_lshrrev_b32 v11, 16, v5 // (op 12527) 
v_cvt_f32_f16 v11, v11 // (op 12527) 
v_mul_f32 v20, v0, v10 // (op 12527) 
v_mul_f32 v21, v0, v11 // (op 12527) 


v_add_f32 v10, v18, v14 // (op 12523) 
v_add_f32 v11, v19, v15 // (op 12523) 
v_add_f32 v12, v20, v16 // (op 12523) 
v_add_f32 v13, v21, v17 // (op 12523) 


v_cvt_f16_f32 v14, v10 // (op 12524) 
v_cvt_f16_f32 v15, v11 // (op 12524) 
v_cvt_f16_f32 v16, v12 // (op 12524) 
v_cvt_f16_f32 v17, v13 // (op 12524) 


v_pack_b32_f16 v10, v14, v15 // (op 12525) 
v_pack_b32_f16 v11, v16, v17 // (op 12525) 

s_lshl_b32 s0, s4, 5 // (op 12525) 
s_mul_i32 s1, 3, s0 // (op 12525) 

v_add_u32 v3, v9, s1 // (op 12525) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12525) Store value





s_lshl_b32 s0, s2, 7 // (op 12540) 
v_add_u32 v3, v8, s0 // (op 12540) 


v_accvgpr_read v10, a16 // (op 12536) 
v_accvgpr_read v11, a17 // (op 12536) 
v_accvgpr_read v12, a18 // (op 12536) 
v_accvgpr_read v13, a19 // (op 12536) 
v_mul_f32 v14, v7, v10 // (op 12536) 
v_mul_f32 v15, v7, v11 // (op 12536) 
v_mul_f32 v16, v7, v12 // (op 12536) 
v_mul_f32 v17, v7, v13 // (op 12536) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12540) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12541) 
v_lshrrev_b32 v11, 16, v4 // (op 12541) 
v_cvt_f32_f16 v11, v11 // (op 12541) 
v_mul_f32 v18, v0, v10 // (op 12541) 
v_mul_f32 v19, v0, v11 // (op 12541) 
v_cvt_f32_f16 v10, v5 // (op 12541) 
v_lshrrev_b32 v11, 16, v5 // (op 12541) 
v_cvt_f32_f16 v11, v11 // (op 12541) 
v_mul_f32 v20, v0, v10 // (op 12541) 
v_mul_f32 v21, v0, v11 // (op 12541) 


v_add_f32 v10, v18, v14 // (op 12537) 
v_add_f32 v11, v19, v15 // (op 12537) 
v_add_f32 v12, v20, v16 // (op 12537) 
v_add_f32 v13, v21, v17 // (op 12537) 


v_cvt_f16_f32 v14, v10 // (op 12538) 
v_cvt_f16_f32 v15, v11 // (op 12538) 
v_cvt_f16_f32 v16, v12 // (op 12538) 
v_cvt_f16_f32 v17, v13 // (op 12538) 


v_pack_b32_f16 v10, v14, v15 // (op 12539) 
v_pack_b32_f16 v11, v16, v17 // (op 12539) 

s_lshl_b32 s0, s4, 7 // (op 12539) 
v_add_u32 v3, v9, s0 // (op 12539) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12539) Store value





s_lshl_b32 s0, s2, 5 // (op 12554) 
s_mul_i32 s1, 5, s0 // (op 12554) 

v_add_u32 v3, v8, s1 // (op 12554) 


v_accvgpr_read v10, a20 // (op 12550) 
v_accvgpr_read v11, a21 // (op 12550) 
v_accvgpr_read v12, a22 // (op 12550) 
v_accvgpr_read v13, a23 // (op 12550) 
v_mul_f32 v14, v7, v10 // (op 12550) 
v_mul_f32 v15, v7, v11 // (op 12550) 
v_mul_f32 v16, v7, v12 // (op 12550) 
v_mul_f32 v17, v7, v13 // (op 12550) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12554) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12555) 
v_lshrrev_b32 v11, 16, v4 // (op 12555) 
v_cvt_f32_f16 v11, v11 // (op 12555) 
v_mul_f32 v18, v0, v10 // (op 12555) 
v_mul_f32 v19, v0, v11 // (op 12555) 
v_cvt_f32_f16 v10, v5 // (op 12555) 
v_lshrrev_b32 v11, 16, v5 // (op 12555) 
v_cvt_f32_f16 v11, v11 // (op 12555) 
v_mul_f32 v20, v0, v10 // (op 12555) 
v_mul_f32 v21, v0, v11 // (op 12555) 


v_add_f32 v10, v18, v14 // (op 12551) 
v_add_f32 v11, v19, v15 // (op 12551) 
v_add_f32 v12, v20, v16 // (op 12551) 
v_add_f32 v13, v21, v17 // (op 12551) 


v_cvt_f16_f32 v14, v10 // (op 12552) 
v_cvt_f16_f32 v15, v11 // (op 12552) 
v_cvt_f16_f32 v16, v12 // (op 12552) 
v_cvt_f16_f32 v17, v13 // (op 12552) 


v_pack_b32_f16 v10, v14, v15 // (op 12553) 
v_pack_b32_f16 v11, v16, v17 // (op 12553) 

s_lshl_b32 s0, s4, 5 // (op 12553) 
s_mul_i32 s1, 5, s0 // (op 12553) 

v_add_u32 v3, v9, s1 // (op 12553) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12553) Store value





s_lshl_b32 s0, s2, 5 // (op 12568) 
s_mul_i32 s1, 6, s0 // (op 12568) 

v_add_u32 v3, v8, s1 // (op 12568) 


v_accvgpr_read v10, a24 // (op 12564) 
v_accvgpr_read v11, a25 // (op 12564) 
v_accvgpr_read v12, a26 // (op 12564) 
v_accvgpr_read v13, a27 // (op 12564) 
v_mul_f32 v14, v7, v10 // (op 12564) 
v_mul_f32 v15, v7, v11 // (op 12564) 
v_mul_f32 v16, v7, v12 // (op 12564) 
v_mul_f32 v17, v7, v13 // (op 12564) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12568) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12569) 
v_lshrrev_b32 v11, 16, v4 // (op 12569) 
v_cvt_f32_f16 v11, v11 // (op 12569) 
v_mul_f32 v18, v0, v10 // (op 12569) 
v_mul_f32 v19, v0, v11 // (op 12569) 
v_cvt_f32_f16 v10, v5 // (op 12569) 
v_lshrrev_b32 v11, 16, v5 // (op 12569) 
v_cvt_f32_f16 v11, v11 // (op 12569) 
v_mul_f32 v20, v0, v10 // (op 12569) 
v_mul_f32 v21, v0, v11 // (op 12569) 


v_add_f32 v10, v18, v14 // (op 12565) 
v_add_f32 v11, v19, v15 // (op 12565) 
v_add_f32 v12, v20, v16 // (op 12565) 
v_add_f32 v13, v21, v17 // (op 12565) 


v_cvt_f16_f32 v14, v10 // (op 12566) 
v_cvt_f16_f32 v15, v11 // (op 12566) 
v_cvt_f16_f32 v16, v12 // (op 12566) 
v_cvt_f16_f32 v17, v13 // (op 12566) 


v_pack_b32_f16 v10, v14, v15 // (op 12567) 
v_pack_b32_f16 v11, v16, v17 // (op 12567) 

s_lshl_b32 s0, s4, 5 // (op 12567) 
s_mul_i32 s1, 6, s0 // (op 12567) 

v_add_u32 v3, v9, s1 // (op 12567) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12567) Store value





s_lshl_b32 s0, s2, 5 // (op 12582) 
s_mul_i32 s1, 7, s0 // (op 12582) 

v_add_u32 v3, v8, s1 // (op 12582) 


v_accvgpr_read v10, a28 // (op 12578) 
v_accvgpr_read v11, a29 // (op 12578) 
v_accvgpr_read v12, a30 // (op 12578) 
v_accvgpr_read v13, a31 // (op 12578) 
v_mul_f32 v14, v7, v10 // (op 12578) 
v_mul_f32 v15, v7, v11 // (op 12578) 
v_mul_f32 v16, v7, v12 // (op 12578) 
v_mul_f32 v17, v7, v13 // (op 12578) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 12582) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 12583) 
v_lshrrev_b32 v11, 16, v4 // (op 12583) 
v_cvt_f32_f16 v11, v11 // (op 12583) 
v_mul_f32 v18, v0, v10 // (op 12583) 
v_mul_f32 v19, v0, v11 // (op 12583) 
v_cvt_f32_f16 v10, v5 // (op 12583) 
v_lshrrev_b32 v11, 16, v5 // (op 12583) 
v_cvt_f32_f16 v11, v11 // (op 12583) 
v_mul_f32 v20, v0, v10 // (op 12583) 
v_mul_f32 v21, v0, v11 // (op 12583) 


v_add_f32 v10, v18, v14 // (op 12579) 
v_add_f32 v11, v19, v15 // (op 12579) 
v_add_f32 v12, v20, v16 // (op 12579) 
v_add_f32 v13, v21, v17 // (op 12579) 


v_cvt_f16_f32 v14, v10 // (op 12580) 
v_cvt_f16_f32 v15, v11 // (op 12580) 
v_cvt_f16_f32 v16, v12 // (op 12580) 
v_cvt_f16_f32 v17, v13 // (op 12580) 


v_pack_b32_f16 v10, v14, v15 // (op 12581) 
v_pack_b32_f16 v11, v16, v17 // (op 12581) 

s_lshl_b32 s0, s4, 5 // (op 12581) 
s_mul_i32 s1, 7, s0 // (op 12581) 

v_add_u32 v3, v9, s1 // (op 12581) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 12581) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalBottom_0_beta_116:








































































s_endpgm  // End of RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__end-RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 444
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 98304 // lds bytes
  .amdhsa_accum_offset 188
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
    DataType: F8
    DataTypeA: F8
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
    macrotile: [256, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 98304
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 186
    .agpr_count: 256
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 156
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 160
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
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

