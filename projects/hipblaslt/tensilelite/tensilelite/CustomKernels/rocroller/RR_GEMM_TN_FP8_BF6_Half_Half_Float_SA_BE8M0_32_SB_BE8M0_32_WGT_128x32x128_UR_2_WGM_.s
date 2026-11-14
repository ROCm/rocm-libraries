// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 2047) 
s_mul_hi_i32 s1, s16, s10 // (op 2235) 
v_lshrrev_b32 v0, 6, v1 // (op 2161) 
v_and_b32 v3, 1, v0 // (op 2161) 

v_and_b32 v0, 15, v1 // (op 2161) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 2161) 


v_and_b32 v0, 63, v1 // (op 2161) 
v_lshrrev_b32 v3, 4, v0 // (op 2161) 

v_lshlrev_b32 v0, 5, v3 // (op 2161) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 2161) 


v_mul_lo_u32 v0, v3, 6 // (op 2161) 

v_lshrrev_b32 v3, 3, v0 // (op 2161) 

v_lshrrev_b32 v0, 6, v1 // (op 2305) 
v_and_b32 v4, 1, v0 // (op 2305) 

v_and_b32 v0, 15, v1 // (op 2305) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 2305) 


v_and_b32 v0, 63, v1 // (op 2305) 
v_lshrrev_b32 v4, 4, v0 // (op 2305) 

v_lshlrev_b32 v0, 5, v4 // (op 2305) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 2305) 


v_mul_lo_u32 v0, v4, 6 // (op 2305) 

v_lshrrev_b32 v4, 3, v0 // (op 2305) 

v_lshrrev_b32 v0, 7, v1 // (op 2047) 
s_add_i32 s18, s0, s16 // (op 2047) 

v_and_b32 v5, 63, v1 // (op 2047) 
s_ashr_i32 s0, s18, 31 // (op 2047) 
v_lshrrev_b32 v6, 4, v5 // (op 2047) 
s_cmp_eq_i32 s10, 0 // (op 2047) 
s_mov_b32 s19, scc // (op 2047) 
v_lshlrev_b32 v7, 2, v6 // (op 2047) 

s_cmp_eq_i32 s19, 0 // (op 2047) 
s_cselect_b32 s49, 0, -1 // (op 2047) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 2047) 


s_and_b32 s19, s13, 31 // (op 2047) 
v_lshrrev_b32 v0, 2, v5 // (op 2047) 

v_and_b32 v5, 3, v0 // (op 2047) 

v_lshrrev_b32 v0, 7, v1 // (op 2108) 
v_and_b32 v7, 15, v1 // (op 2108) 
v_lshl_add_u32 v8, v0, 4, v7 // (op 2108) 


v_and_b32 v0, 48, v1 // (op 2108) 
v_lshl_add_u32 v7, v8, 7, v0 // (op 2108) 


v_lshrrev_b32 v0, 7, v1 // (op 2219) 
v_and_b32 v8, 15, v1 // (op 2219) 
v_lshl_add_u32 v9, v0, 4, v8 // (op 2219) 


v_and_b32 v0, 63, v1 // (op 2219) 
v_lshrrev_b32 v8, 4, v0 // (op 2219) 

v_lshl_add_u32 v0, v9, 2, v8 // (op 2219) 


v_lshrrev_b32 v8, 7, v1 // (op 2235) 
v_and_b32 v9, 63, v1 // (op 2235) 
v_lshrrev_b32 v10, 7, v1 // (op 2281) 
v_and_b32 v11, 15, v1 // (op 2281) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 2281) 


v_and_b32 v10, 48, v1 // (op 2281) 
v_lshl_add_u32 v11, v12, 7, v10 // (op 2281) 


v_lshrrev_b32 v10, 7, v1 // (op 2334) 
v_and_b32 v12, 15, v1 // (op 2334) 
v_lshl_add_u32 v13, v10, 4, v12 // (op 2334) 


v_and_b32 v10, 63, v1 // (op 2334) 
v_lshrrev_b32 v12, 4, v10 // (op 2334) 

v_lshl_add_u32 v10, v13, 2, v12 // (op 2334) 


s_lshl_b32 s50, 1, s19 // (op 2047) 
v_and_b32 v12, 3, v1 // (op 2047) 
s_add_i32 s51, s50, s49 // (op 2047) 


s_and_b32 s49, s0, s51 // (op 2047) 


s_add_i32 s0, s1, s16 // (op 2235) 

s_ashr_i32 s1, s0, 31 // (op 2235) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 2047) 


s_add_i32 s50, s18, s49 // (op 2047) 


v_lshrrev_b32 v5, 6, v1 // (op 2047) 
s_ashr_i32 s18, s50, s19 // (op 2047) 


v_and_b32 v12, 1, v5 // (op 2047) 

s_lshl_b32 s19, s13, 24 // (op 2047) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 2047) 


s_ashr_i32 s49, s19, 31 // (op 2047) 

s_xor_b32 s19, s18, s49 // (op 2047) 

s_sub_i32 s18, s19, s49 // (op 2047) 


s_mul_hi_i32 s19, s18, s14 // (op 2047) 
s_add_i32 s49, s19, s18 // (op 2047) 

s_ashr_i32 s19, s49, 31 // (op 2047) 
s_cmp_eq_i32 s14, 0 // (op 2047) 
s_mov_b32 s50, scc // (op 2047) 
s_cmp_eq_i32 s50, 0 // (op 2047) 
s_cselect_b32 s51, 0, -1 // (op 2047) 

s_and_b32 s50, s15, 31 // (op 2047) 
v_lshrrev_b32 v12, 4, v9 // (op 2235) 
s_cmp_eq_i32 s10, 0 // (op 2235) 
s_mov_b32 s52, scc // (op 2235) 
v_lshlrev_b32 v13, 2, v12 // (op 2235) 

s_cmp_eq_i32 s52, 0 // (op 2235) 
s_cselect_b32 s53, 0, -1 // (op 2235) 

s_lshl_b32 s52, 1, s50 // (op 2047) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 2235) 


v_lshrrev_b32 v8, 2, v9 // (op 2235) 

s_add_i32 s54, s52, s51 // (op 2047) 


s_and_b32 s51, s19, s54 // (op 2047) 


s_add_i32 s19, s49, s51 // (op 2047) 


s_ashr_i32 s49, s19, s50 // (op 2047) 


s_lshl_b32 s19, s15, 24 // (op 2047) 
s_ashr_i32 s50, s19, 31 // (op 2047) 

s_xor_b32 s19, s49, s50 // (op 2047) 

s_sub_i32 s49, s19, s50 // (op 2047) 


s_mul_i32 s19, s18, s11 // (op 2047) 
s_sub_u32 s50, s16, s19 // (op 2047) 

s_mul_i32 s19, s49, s11 // (op 2047) 
s_add_u32 s19, s19, s50 // (op 2047) 

s_mul_hi_i32 s50, s16, s9 // (op 2047) 
s_add_i32 s51, s50, s16 // (op 2047) 

s_ashr_i32 s50, s51, 31 // (op 2047) 
s_cmp_eq_i32 s9, 0 // (op 2047) 
s_mov_b32 s52, scc // (op 2047) 
s_cmp_eq_i32 s52, 0 // (op 2047) 
s_cselect_b32 s54, 0, -1 // (op 2047) 

s_and_b32 s52, s13, 31 // (op 2235) 
v_and_b32 v9, 3, v8 // (op 2235) 

s_lshl_b32 s55, 1, s52 // (op 2235) 
v_and_b32 v8, 3, v1 // (op 2235) 
s_add_i32 s56, s55, s53 // (op 2235) 


s_and_b32 s53, s1, s56 // (op 2235) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 2235) 


s_add_i32 s1, s0, s53 // (op 2235) 


v_lshrrev_b32 v8, 6, v1 // (op 2235) 
s_ashr_i32 s0, s1, s52 // (op 2235) 


v_and_b32 v9, 1, v8 // (op 2235) 

s_lshl_b32 s1, s13, 24 // (op 2235) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 2235) 


s_ashr_i32 s52, s1, 31 // (op 2235) 

s_xor_b32 s1, s0, s52 // (op 2235) 

s_sub_i32 s0, s1, s52 // (op 2235) 


s_mul_hi_i32 s1, s0, s14 // (op 2235) 
s_add_i32 s52, s1, s0 // (op 2235) 

s_ashr_i32 s1, s52, 31 // (op 2235) 
s_cmp_eq_i32 s14, 0 // (op 2235) 
s_mov_b32 s53, scc // (op 2235) 
s_cmp_eq_i32 s53, 0 // (op 2235) 
s_cselect_b32 s55, 0, -1 // (op 2235) 

s_and_b32 s53, s15, 31 // (op 2235) 
s_lshl_b32 s56, 1, s53 // (op 2235) 
s_add_i32 s57, s56, s55 // (op 2235) 


s_and_b32 s55, s1, s57 // (op 2235) 


s_add_i32 s1, s52, s55 // (op 2235) 


s_ashr_i32 s52, s1, s53 // (op 2235) 


s_lshl_b32 s1, s15, 24 // (op 2235) 
s_ashr_i32 s53, s1, 31 // (op 2235) 

s_xor_b32 s1, s52, s53 // (op 2235) 

s_sub_i32 s52, s1, s53 // (op 2235) 


s_mul_i32 s1, s0, s11 // (op 2235) 
s_sub_u32 s53, s16, s1 // (op 2235) 

s_mul_i32 s1, s52, s11 // (op 2235) 
s_add_u32 s1, s1, s53 // (op 2235) 

s_mul_hi_i32 s53, s16, s9 // (op 2235) 
s_add_i32 s55, s53, s16 // (op 2235) 

s_ashr_i32 s53, s55, 31 // (op 2235) 
s_cmp_eq_i32 s9, 0 // (op 2235) 
s_mov_b32 s56, scc // (op 2235) 
s_cmp_eq_i32 s56, 0 // (op 2235) 
s_cselect_b32 s57, 0, -1 // (op 2235) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 2047) 
s_lshl_b32 s58, 1, s56 // (op 2047) 
s_add_i32 s59, s58, s54 // (op 2047) 


s_and_b32 s54, s50, s59 // (op 2047) 


s_add_i32 s50, s51, s54 // (op 2047) 


s_ashr_i32 s51, s50, s56 // (op 2047) 


s_lshl_b32 s50, s45, 24 // (op 2047) 
s_ashr_i32 s54, s50, 31 // (op 2047) 

s_xor_b32 s50, s51, s54 // (op 2047) 

s_sub_i32 s51, s50, s54 // (op 2047) 


s_and_b32 s50, s45, 31 // (op 2235) 
s_lshl_b32 s54, 1, s50 // (op 2235) 
s_add_i32 s56, s54, s57 // (op 2235) 


s_and_b32 s54, s53, s56 // (op 2235) 


s_add_i32 s53, s55, s54 // (op 2235) 


s_ashr_i32 s54, s53, s50 // (op 2235) 


s_lshl_b32 s50, s45, 24 // (op 2235) 
s_ashr_i32 s53, s50, 31 // (op 2235) 

s_xor_b32 s50, s54, s53 // (op 2235) 

s_sub_i32 s54, s50, s53 // (op 2235) 


s_add_u32 s56, s4, 32 // (op 2047) Start of Int64 add, locking SCC
s_addc_u32 s57, s5, 0 // (op 2047) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 2047) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 2047) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 2047) 

s_mul_i32 s50, s11, s56 // (op 2047) 
s_mul_i32 s53, s8, s50 // (op 2047) 

s_mul_i32 s50, s51, s53 // (op 2047) 

s_sub_u32 s53, s16, s50 // (op 2047) 

s_mov_b32 s50, s53 // (op 2047) 
s_mul_hi_i32 s55, s50, s12 // (op 2047) 
s_add_i32 s58, s55, s50 // (op 2047) 


s_ashr_i32 s50, s58, 31 // (op 2047) 
s_cmp_eq_i32 s12, 0 // (op 2047) 
s_mov_b32 s55, scc // (op 2047) 
s_cmp_eq_i32 s55, 0 // (op 2047) 
s_cselect_b32 s59, 0, -1 // (op 2047) 

s_and_b32 s55, s44, 31 // (op 2047) 
s_lshl_b32 s60, 1, s55 // (op 2047) 
s_add_i32 s61, s60, s59 // (op 2047) 


s_and_b32 s59, s50, s61 // (op 2047) 


s_add_i32 s50, s58, s59 // (op 2047) 


s_ashr_i32 s58, s50, s55 // (op 2047) 


s_lshl_b32 s50, s44, 24 // (op 2047) 
s_ashr_i32 s55, s50, 31 // (op 2047) 

s_xor_b32 s50, s58, s55 // (op 2047) 

s_sub_i32 s58, s50, s55 // (op 2047) 


s_mul_i32 s50, s46, s11 // (op 2047) 
s_add_u32 s60, s6, 128 // (op 2047) Start of Int64 add, locking SCC
s_addc_u32 s61, s7, 0 // (op 2047) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 2047) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 2047) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 7 // (op 2047) 

s_sub_i32 s55, s60, s50 // (op 2047) 

s_mul_i32 s59, s58, s55 // (op 2047) 

s_sub_u32 s55, s53, s59 // (op 2047) 


s_add_u32 s53, s55, s50 // (op 2047) 


s_cmp_eq_i32 s51, 0 // (op 2047) 
s_mov_b32 s50, scc // (op 2047) 

s_cmp_eq_i32 s50, 0 // (op 2047) 
s_cselect_b32 s51, s53, s19 // (op 2047) 


s_mul_i32 s19, s49, s56 // (op 2047) 


v_lshl_add_u32 v9, s51, 7, v6 // (op 2047) 


s_sub_i32 s49, s18, s19 // (op 2047) 


s_cmp_eq_i32 s50, 0 // (op 2047) 
s_cselect_b32 s51, s58, s49 // (op 2047) 



v_lshl_add_u32 v6, s51, 5, v5 // (op 2047) 


v_mul_lo_u32 v5, v6, s2 // (op 2047) 

v_add_lshl_u32 v6, v9, v5, 1 // (op 2047) 


s_add_u32 s18, s4, 32 // (op 2235) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 2235) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 2235) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 2235) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 2235) 

s_mul_i32 s53, s11, s18 // (op 2235) 
s_mul_i32 s49, s8, s53 // (op 2235) 

s_mul_i32 s53, s54, s49 // (op 2235) 

s_sub_u32 s49, s16, s53 // (op 2235) 

s_mov_b32 s53, s49 // (op 2235) 
s_mul_hi_i32 s50, s53, s12 // (op 2235) 
s_add_i32 s51, s50, s53 // (op 2235) 


s_ashr_i32 s50, s51, 31 // (op 2235) 
s_cmp_eq_i32 s12, 0 // (op 2235) 
s_mov_b32 s53, scc // (op 2235) 
s_cmp_eq_i32 s53, 0 // (op 2235) 
s_cselect_b32 s55, 0, -1 // (op 2235) 

s_and_b32 s53, s44, 31 // (op 2235) 
s_lshl_b32 s56, 1, s53 // (op 2235) 
s_add_i32 s57, s56, s55 // (op 2235) 


s_and_b32 s55, s50, s57 // (op 2235) 


s_add_i32 s50, s51, s55 // (op 2235) 


s_ashr_i32 s51, s50, s53 // (op 2235) 


s_lshl_b32 s50, s44, 24 // (op 2235) 
s_ashr_i32 s53, s50, 31 // (op 2235) 

s_xor_b32 s50, s51, s53 // (op 2235) 

s_sub_i32 s51, s50, s53 // (op 2235) 


s_mul_i32 s50, s46, s11 // (op 2235) 
s_add_u32 s56, s6, 128 // (op 2235) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 2235) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 2235) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 2235) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 2235) 

s_sub_i32 s53, s56, s50 // (op 2235) 

s_mul_i32 s55, s51, s53 // (op 2235) 

s_sub_u32 s53, s49, s55 // (op 2235) 


s_add_u32 s49, s53, s50 // (op 2235) 


s_cmp_eq_i32 s54, 0 // (op 2235) 
s_mov_b32 s50, scc // (op 2235) 

s_cmp_eq_i32 s50, 0 // (op 2235) 
s_cselect_b32 s53, s49, s1 // (op 2235) 


s_mul_i32 s1, s52, s18 // (op 2235) 


v_lshl_add_u32 v5, s53, 7, v12 // (op 2235) 


s_sub_i32 s49, s0, s1 // (op 2235) 


s_cmp_eq_i32 s50, 0 // (op 2235) 
s_cselect_b32 s0, s51, s49 // (op 2235) 



v_lshl_add_u32 v9, s0, 5, v8 // (op 2235) 


v_mul_lo_u32 v8, v9, s20 // (op 2235) 

v_add_lshl_u32 v9, v5, v8, 1 // (op 2235) 


s_mul_i32 s1, s2, s5 // (op 2049) most significant: low of low * high
s_mul_i32 s18, s3, s4 // (op 2049) most significant: low of high * low
s_mul_i32 s0, s2, s4 // (op 2049) least significant
s_mul_hi_u32 s19, s2, s4 // (op 2049) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2049) most significant: sum
s_add_u32 s1, s1, s19 // (op 2049) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 2049) 

s_mul_i32 s1, s20, s5 // (op 2237) most significant: low of low * high
s_mul_i32 s49, s21, s4 // (op 2237) most significant: low of high * low
s_mul_i32 s0, s20, s4 // (op 2237) least significant
s_mul_hi_u32 s50, s20, s4 // (op 2237) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 2237) most significant: sum
s_add_u32 s1, s1, s50 // (op 2237) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 2237) 

s_mov_b32 s52, s22 // (op 2049) reg expression
s_mov_b32 s53, s23 // (op 2049) reg expression
s_mov_b32 s54, s18 // (op 2049) reg expression
s_mov_b32 s55, 131072 // (op 2049) call()

s_mov_b32 s56, s24 // (op 2237) reg expression

s_mov_b32 s57, s25 // (op 2237) reg expression
s_mov_b32 s58, s50 // (op 2237) reg expression
s_mov_b32 s59, 131072 // (op 2237) call()


v_mov_b32 v5, s47 // (op 2) Move value
v_mov_b32 v8, s48 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 313) call()
v_accvgpr_write a5, 0.00000 // (op 313) call()
v_accvgpr_write a6, 0.00000 // (op 313) call()
v_accvgpr_write a7, 0.00000 // (op 313) call()
v_accvgpr_write a8, 0.00000 // (op 475) call()
v_accvgpr_write a9, 0.00000 // (op 475) call()
v_accvgpr_write a10, 0.00000 // (op 475) call()
v_accvgpr_write a11, 0.00000 // (op 475) call()
v_accvgpr_write a12, 0.00000 // (op 637) call()
v_accvgpr_write a13, 0.00000 // (op 637) call()
v_accvgpr_write a14, 0.00000 // (op 637) call()
v_accvgpr_write a15, 0.00000 // (op 637) call()
s_mul_hi_i32 s0, s16, s10 // (op 1814) 
v_lshrrev_b32 v12, 3, v1 // (op 1814) 
s_add_i32 s1, s0, s16 // (op 1814) 

v_and_b32 v13, -8, v1 // (op 1814) 
s_ashr_i32 s0, s1, 31 // (op 1814) 
v_sub_u32 v14, v1, v13 // (op 1814) 

s_cmp_eq_i32 s10, 0 // (op 1814) 
s_mov_b32 s18, scc // (op 1814) 
v_lshlrev_b32 v13, 4, v14 // (op 1814) 

s_cmp_eq_i32 s18, 0 // (op 1814) 
s_cselect_b32 s19, 0, -1 // (op 1814) 

v_mov_b32 v15, 0 // (op 1814) convert
v_mov_b32 v14, v13 // (op 1814) convert

s_and_b32 s18, s13, 31 // (op 1814) 
s_lshl_b32 s22, 1, s18 // (op 1814) 
s_add_i32 s23, s22, s19 // (op 1814) 


s_and_b32 s19, s0, s23 // (op 1814) 


s_add_i32 s0, s1, s19 // (op 1814) 


s_ashr_i32 s1, s0, s18 // (op 1814) 


s_lshl_b32 s0, s13, 24 // (op 1814) 
s_ashr_i32 s18, s0, 31 // (op 1814) 

s_xor_b32 s0, s1, s18 // (op 1814) 

s_sub_i32 s1, s0, s18 // (op 1814) 


s_mul_hi_i32 s0, s1, s14 // (op 1814) 
s_add_i32 s18, s0, s1 // (op 1814) 

s_ashr_i32 s0, s18, 31 // (op 1814) 
s_cmp_eq_i32 s14, 0 // (op 1814) 
s_mov_b32 s19, scc // (op 1814) 
s_cmp_eq_i32 s19, 0 // (op 1814) 
s_cselect_b32 s22, 0, -1 // (op 1814) 

s_and_b32 s19, s15, 31 // (op 1814) 
s_lshl_b32 s23, 1, s19 // (op 1814) 
s_add_i32 s24, s23, s22 // (op 1814) 


s_and_b32 s25, s0, s24 // (op 1814) 


s_add_i32 s0, s18, s25 // (op 1814) 


s_ashr_i32 s18, s0, s19 // (op 1814) 


s_lshl_b32 s0, s15, 24 // (op 1814) 
s_ashr_i32 s19, s0, 31 // (op 1814) 

s_xor_b32 s0, s18, s19 // (op 1814) 

s_sub_i32 s18, s0, s19 // (op 1814) 


s_mul_i32 s0, s1, s11 // (op 1814) 

s_sub_u32 s1, s16, s0 // (op 1814) 

s_mul_i32 s0, s18, s11 // (op 1814) 
s_add_u32 s0, s0, s1 // (op 1814) 


s_mul_hi_i32 s1, s16, s9 // (op 1814) 
s_add_i32 s18, s1, s16 // (op 1814) 

s_ashr_i32 s1, s18, 31 // (op 1814) 
s_cmp_eq_i32 s9, 0 // (op 1814) 
s_mov_b32 s19, scc // (op 1814) 
s_cmp_eq_i32 s19, 0 // (op 1814) 
s_cselect_b32 s22, 0, -1 // (op 1814) 

s_and_b32 s19, s45, 31 // (op 1814) 
s_lshl_b32 s23, 1, s19 // (op 1814) 
s_add_i32 s24, s23, s22 // (op 1814) 


s_and_b32 s25, s1, s24 // (op 1814) 


s_add_i32 s1, s18, s25 // (op 1814) 


s_ashr_i32 s18, s1, s19 // (op 1814) 


s_lshl_b32 s1, s45, 24 // (op 1814) 
s_ashr_i32 s19, s1, 31 // (op 1814) 

s_xor_b32 s1, s18, s19 // (op 1814) 

s_sub_i32 s18, s1, s19 // (op 1814) 


s_add_u32 s22, s4, 32 // (op 1814) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1814) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1814) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1814) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1814) 

s_mul_i32 s1, s11, s22 // (op 1814) 

s_mul_i32 s19, s8, s1 // (op 1814) 

s_mul_i32 s1, s18, s19 // (op 1814) 

s_sub_u32 s19, s16, s1 // (op 1814) 

s_mov_b32 s1, s19 // (op 1814) 
s_mul_hi_i32 s22, s1, s12 // (op 1814) 
s_add_i32 s23, s22, s1 // (op 1814) 


s_ashr_i32 s1, s23, 31 // (op 1814) 
s_cmp_eq_i32 s12, 0 // (op 1814) 
s_mov_b32 s22, scc // (op 1814) 
s_cmp_eq_i32 s22, 0 // (op 1814) 
s_cselect_b32 s24, 0, -1 // (op 1814) 

s_and_b32 s22, s44, 31 // (op 1814) 
s_lshl_b32 s25, 1, s22 // (op 1814) 
s_add_i32 s47, s25, s24 // (op 1814) 


s_and_b32 s24, s1, s47 // (op 1814) 


s_add_i32 s1, s23, s24 // (op 1814) 


s_ashr_i32 s23, s1, s22 // (op 1814) 


s_lshl_b32 s1, s44, 24 // (op 1814) 
s_ashr_i32 s22, s1, 31 // (op 1814) 

s_xor_b32 s1, s23, s22 // (op 1814) 

s_sub_i32 s23, s1, s22 // (op 1814) 


s_mul_i32 s1, s46, s11 // (op 1814) 
s_add_u32 s24, s6, 128 // (op 1814) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1814) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1814) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1814) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 7 // (op 1814) 

s_sub_i32 s22, s24, s1 // (op 1814) 

s_mul_i32 s24, s23, s22 // (op 1814) 


s_sub_u32 s25, s19, s24 // (op 1814) 


s_add_u32 s19, s25, s1 // (op 1814) 


s_cmp_eq_i32 s18, 0 // (op 1814) 
s_mov_b32 s1, scc // (op 1814) 

s_cmp_eq_i32 s1, 0 // (op 1814) 
s_cselect_b32 s18, s19, s0 // (op 1814) 



v_lshl_add_u32 v13, s18, 7, v12 // (op 1814) 


v_mul_lo_u32 v17, v13, s27 // (op 1814) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 1814) most significant: high of low * low
v_mul_lo_u32 v16, v13, s26 // (op 1814) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 1814) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 1814) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 1814) most significant half



s_mul_i32 s1, s26, s7 // (op 1816) most significant: low of low * high
s_mul_i32 s18, s27, s6 // (op 1816) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 1816) least significant
s_mul_hi_u32 s19, s26, s6 // (op 1816) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1816) most significant: sum
s_add_u32 s1, s1, s19 // (op 1816) most significant: sum


s_mov_b32 s48, s28 // (op 1816) reg expression
s_mov_b32 s49, s29 // (op 1816) reg expression
s_mov_b32 s50, s0 // (op 1816) reg expression
s_mov_b32 s51, 131072 // (op 1816) call()

s_mul_hi_i32 s47, s16, s10 // (op 1833) 
v_lshrrev_b32 v12, 3, v1 // (op 1833) 
s_add_i32 s0, s47, s16 // (op 1833) 

v_and_b32 v13, -8, v1 // (op 1833) 
s_ashr_i32 s1, s0, 31 // (op 1833) 
v_sub_u32 v14, v1, v13 // (op 1833) 

s_cmp_eq_i32 s10, 0 // (op 1833) 
s_mov_b32 s47, scc // (op 1833) 
v_lshlrev_b32 v13, 4, v14 // (op 1833) 

s_cmp_eq_i32 s47, 0 // (op 1833) 
s_cselect_b32 s18, 0, -1 // (op 1833) 

v_mov_b32 v15, 0 // (op 1833) convert
v_mov_b32 v14, v13 // (op 1833) convert

s_and_b32 s19, s13, 31 // (op 1833) 
s_lshl_b32 s47, 1, s19 // (op 1833) 
s_add_i32 s22, s47, s18 // (op 1833) 


s_and_b32 s18, s1, s22 // (op 1833) 


s_add_i32 s1, s0, s18 // (op 1833) 


s_ashr_i32 s0, s1, s19 // (op 1833) 


s_lshl_b32 s1, s13, 24 // (op 1833) 
s_ashr_i32 s47, s1, 31 // (op 1833) 

s_xor_b32 s1, s0, s47 // (op 1833) 

s_sub_i32 s0, s1, s47 // (op 1833) 


s_mul_hi_i32 s1, s0, s14 // (op 1833) 
s_add_i32 s47, s1, s0 // (op 1833) 

s_ashr_i32 s1, s47, 31 // (op 1833) 
s_cmp_eq_i32 s14, 0 // (op 1833) 
s_mov_b32 s18, scc // (op 1833) 
s_cmp_eq_i32 s18, 0 // (op 1833) 
s_cselect_b32 s19, 0, -1 // (op 1833) 

s_and_b32 s18, s15, 31 // (op 1833) 
s_lshl_b32 s22, 1, s18 // (op 1833) 
s_add_i32 s23, s22, s19 // (op 1833) 


s_and_b32 s19, s1, s23 // (op 1833) 


s_add_i32 s1, s47, s19 // (op 1833) 


s_ashr_i32 s19, s1, s18 // (op 1833) 


s_lshl_b32 s1, s15, 24 // (op 1833) 
s_ashr_i32 s18, s1, 31 // (op 1833) 

s_xor_b32 s1, s19, s18 // (op 1833) 

s_sub_i32 s19, s1, s18 // (op 1833) 


s_mul_i32 s1, s0, s11 // (op 1833) 

s_sub_u32 s0, s16, s1 // (op 1833) 

s_mul_i32 s1, s19, s11 // (op 1833) 
s_add_u32 s1, s1, s0 // (op 1833) 


s_mul_hi_i32 s0, s16, s9 // (op 1833) 
s_add_i32 s47, s0, s16 // (op 1833) 

s_ashr_i32 s0, s47, 31 // (op 1833) 
s_cmp_eq_i32 s9, 0 // (op 1833) 
s_mov_b32 s18, scc // (op 1833) 
s_cmp_eq_i32 s18, 0 // (op 1833) 
s_cselect_b32 s19, 0, -1 // (op 1833) 

s_and_b32 s18, s45, 31 // (op 1833) 
s_lshl_b32 s22, 1, s18 // (op 1833) 
s_add_i32 s23, s22, s19 // (op 1833) 


s_and_b32 s19, s0, s23 // (op 1833) 


s_add_i32 s0, s47, s19 // (op 1833) 


s_ashr_i32 s19, s0, s18 // (op 1833) 


s_lshl_b32 s0, s45, 24 // (op 1833) 
s_ashr_i32 s18, s0, 31 // (op 1833) 

s_xor_b32 s0, s19, s18 // (op 1833) 

s_sub_i32 s19, s0, s18 // (op 1833) 


s_add_u32 s22, s4, 32 // (op 1833) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1833) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1833) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1833) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1833) 

s_mul_i32 s0, s11, s22 // (op 1833) 

s_mul_i32 s18, s8, s0 // (op 1833) 

s_mul_i32 s0, s19, s18 // (op 1833) 

s_sub_u32 s18, s16, s0 // (op 1833) 

s_mov_b32 s0, s18 // (op 1833) 
s_mul_hi_i32 s47, s0, s12 // (op 1833) 
s_add_i32 s22, s47, s0 // (op 1833) 


s_ashr_i32 s0, s22, 31 // (op 1833) 
s_cmp_eq_i32 s12, 0 // (op 1833) 
s_mov_b32 s47, scc // (op 1833) 
s_cmp_eq_i32 s47, 0 // (op 1833) 
s_cselect_b32 s23, 0, -1 // (op 1833) 

s_and_b32 s47, s44, 31 // (op 1833) 
s_lshl_b32 s24, 1, s47 // (op 1833) 
s_add_i32 s25, s24, s23 // (op 1833) 


s_and_b32 s23, s0, s25 // (op 1833) 


s_add_i32 s0, s22, s23 // (op 1833) 


s_ashr_i32 s22, s0, s47 // (op 1833) 


s_lshl_b32 s0, s44, 24 // (op 1833) 
s_ashr_i32 s47, s0, 31 // (op 1833) 

s_xor_b32 s0, s22, s47 // (op 1833) 

s_sub_i32 s22, s0, s47 // (op 1833) 


s_mul_i32 s0, s46, s11 // (op 1833) 
s_add_u32 s24, s6, 128 // (op 1833) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1833) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 1833) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 1833) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 7 // (op 1833) 

s_sub_i32 s23, s24, s0 // (op 1833) 

s_mul_i32 s47, s22, s23 // (op 1833) 


s_sub_u32 s22, s18, s47 // (op 1833) 


s_add_u32 s18, s22, s0 // (op 1833) 


s_cmp_eq_i32 s19, 0 // (op 1833) 
s_mov_b32 s0, scc // (op 1833) 

s_cmp_eq_i32 s0, 0 // (op 1833) 
s_cselect_b32 s19, s18, s1 // (op 1833) 



v_lshl_add_u32 v13, s19, 7, v12 // (op 1833) 


v_mul_lo_u32 v19, v13, s27 // (op 1833) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 1833) most significant: high of low * low
v_mul_lo_u32 v18, v13, s26 // (op 1833) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 1833) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 1833) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 1833) most significant half



s_mul_i32 s1, s26, s7 // (op 1835) most significant: low of low * high
s_mul_i32 s47, s27, s6 // (op 1835) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 1835) least significant
s_mul_hi_u32 s18, s26, s6 // (op 1835) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1835) most significant: sum
s_add_u32 s1, s1, s18 // (op 1835) most significant: sum


s_mov_b32 s48, s28 // (op 1835) reg expression
s_mov_b32 s49, s29 // (op 1835) reg expression
s_mov_b32 s50, s0 // (op 1835) reg expression
s_mov_b32 s51, 131072 // (op 1835) call()

s_mul_hi_i32 s47, s16, s10 // (op 1878) 
v_lshrrev_b32 v12, 3, v1 // (op 1878) 
s_add_i32 s0, s47, s16 // (op 1878) 

v_and_b32 v13, -8, v1 // (op 1878) 
s_ashr_i32 s1, s0, 31 // (op 1878) 
v_sub_u32 v14, v1, v13 // (op 1878) 

s_cmp_eq_i32 s10, 0 // (op 1878) 
s_mov_b32 s47, scc // (op 1878) 
v_lshlrev_b32 v13, 4, v14 // (op 1878) 

s_cmp_eq_i32 s47, 0 // (op 1878) 
s_cselect_b32 s18, 0, -1 // (op 1878) 

v_mov_b32 v15, 0 // (op 1878) convert
v_mov_b32 v14, v13 // (op 1878) convert

s_and_b32 s19, s13, 31 // (op 1878) 
s_lshl_b32 s47, 1, s19 // (op 1878) 
s_add_i32 s22, s47, s18 // (op 1878) 


s_and_b32 s18, s1, s22 // (op 1878) 


s_add_i32 s1, s0, s18 // (op 1878) 


s_ashr_i32 s0, s1, s19 // (op 1878) 


s_lshl_b32 s1, s13, 24 // (op 1878) 
s_ashr_i32 s47, s1, 31 // (op 1878) 

s_xor_b32 s1, s0, s47 // (op 1878) 

s_sub_i32 s0, s1, s47 // (op 1878) 


s_mul_hi_i32 s1, s0, s14 // (op 1878) 
s_add_i32 s47, s1, s0 // (op 1878) 

s_ashr_i32 s1, s47, 31 // (op 1878) 
s_cmp_eq_i32 s14, 0 // (op 1878) 
s_mov_b32 s18, scc // (op 1878) 
s_cmp_eq_i32 s18, 0 // (op 1878) 
s_cselect_b32 s19, 0, -1 // (op 1878) 

s_and_b32 s18, s15, 31 // (op 1878) 
s_lshl_b32 s22, 1, s18 // (op 1878) 
s_add_i32 s23, s22, s19 // (op 1878) 


s_and_b32 s19, s1, s23 // (op 1878) 


s_add_i32 s1, s47, s19 // (op 1878) 


s_ashr_i32 s19, s1, s18 // (op 1878) 


s_lshl_b32 s1, s15, 24 // (op 1878) 
s_ashr_i32 s18, s1, 31 // (op 1878) 

s_xor_b32 s1, s19, s18 // (op 1878) 

s_sub_i32 s19, s1, s18 // (op 1878) 


s_add_u32 s22, s4, 32 // (op 1878) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1878) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1878) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1878) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1878) 

s_mul_i32 s1, s19, s22 // (op 1878) 

s_sub_i32 s47, s0, s1 // (op 1878) 


s_mul_hi_i32 s0, s16, s9 // (op 1878) 
s_add_i32 s1, s0, s16 // (op 1878) 

s_ashr_i32 s0, s1, 31 // (op 1878) 
s_cmp_eq_i32 s9, 0 // (op 1878) 
s_mov_b32 s18, scc // (op 1878) 
s_cmp_eq_i32 s18, 0 // (op 1878) 
s_cselect_b32 s19, 0, -1 // (op 1878) 

s_and_b32 s18, s45, 31 // (op 1878) 
s_lshl_b32 s24, 1, s18 // (op 1878) 
s_add_i32 s25, s24, s19 // (op 1878) 


s_and_b32 s19, s0, s25 // (op 1878) 


s_add_i32 s0, s1, s19 // (op 1878) 


s_ashr_i32 s1, s0, s18 // (op 1878) 


s_lshl_b32 s0, s45, 24 // (op 1878) 
s_ashr_i32 s18, s0, 31 // (op 1878) 

s_xor_b32 s0, s1, s18 // (op 1878) 

s_sub_i32 s1, s0, s18 // (op 1878) 


s_mul_i32 s0, s11, s22 // (op 1878) 

s_mul_i32 s18, s8, s0 // (op 1878) 

s_mul_i32 s0, s1, s18 // (op 1878) 

s_sub_u32 s18, s16, s0 // (op 1878) 

s_mul_hi_i32 s0, s18, s12 // (op 1878) 
s_add_i32 s19, s0, s18 // (op 1878) 


s_ashr_i32 s0, s19, 31 // (op 1878) 
s_cmp_eq_i32 s12, 0 // (op 1878) 
s_mov_b32 s18, scc // (op 1878) 
s_cmp_eq_i32 s18, 0 // (op 1878) 
s_cselect_b32 s22, 0, -1 // (op 1878) 

s_and_b32 s18, s44, 31 // (op 1878) 
s_lshl_b32 s23, 1, s18 // (op 1878) 
s_add_i32 s24, s23, s22 // (op 1878) 


s_and_b32 s25, s0, s24 // (op 1878) 


s_add_i32 s0, s19, s25 // (op 1878) 


s_ashr_i32 s19, s0, s18 // (op 1878) 


s_lshl_b32 s0, s44, 24 // (op 1878) 
s_ashr_i32 s18, s0, 31 // (op 1878) 

s_xor_b32 s0, s19, s18 // (op 1878) 

s_sub_i32 s19, s0, s18 // (op 1878) 


s_cmp_eq_i32 s1, 0 // (op 1878) 
s_mov_b32 s0, scc // (op 1878) 

s_cmp_eq_i32 s0, 0 // (op 1878) 
s_cselect_b32 s1, s19, s47 // (op 1878) 



v_lshl_add_u32 v13, s1, 5, v12 // (op 1878) 


v_mul_lo_u32 v21, v13, s31 // (op 1878) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 1878) most significant: high of low * low
v_mul_lo_u32 v20, v13, s30 // (op 1878) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 1878) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 1878) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 1878) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 1878) most significant: low of high * low
v_mul_hi_u32 v20, v12, 6 // (op 1878) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 1878) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 1878) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 1878) 

v_mov_b32 v14, v12 // (op 1878) convert

s_mul_i32 s1, s5, 6 // (op 1880) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 1880) least significant
s_mul_hi_u32 s47, s4, 6 // (op 1880) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1880) most significant: sum

s_mul_i32 s19, s30, s1 // (op 1880) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 1880) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 1880) least significant
s_mul_hi_u32 s22, s30, s0 // (op 1880) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 1880) most significant: sum
s_add_u32 s19, s19, s22 // (op 1880) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1880) 

s_mov_b32 s60, s32 // (op 1880) reg expression
s_mov_b32 s61, s33 // (op 1880) reg expression
s_mov_b32 s62, s0 // (op 1880) reg expression
s_mov_b32 s63, 131072 // (op 1880) call()

s_mul_hi_i32 s47, s16, s10 // (op 1897) 
v_lshrrev_b32 v15, 3, v1 // (op 1897) 
s_add_i32 s0, s47, s16 // (op 1897) 

v_and_b32 v12, -8, v1 // (op 1897) 
s_ashr_i32 s1, s0, 31 // (op 1897) 
v_sub_u32 v13, v1, v12 // (op 1897) 

s_cmp_eq_i32 s10, 0 // (op 1897) 
s_mov_b32 s47, scc // (op 1897) 
v_lshlrev_b32 v12, 4, v13 // (op 1897) 

s_cmp_eq_i32 s47, 0 // (op 1897) 
s_cselect_b32 s18, 0, -1 // (op 1897) 

v_mov_b32 v21, 0 // (op 1897) convert
v_mov_b32 v20, v12 // (op 1897) convert

s_and_b32 s19, s13, 31 // (op 1897) 
s_lshl_b32 s47, 1, s19 // (op 1897) 
s_add_i32 s22, s47, s18 // (op 1897) 


s_and_b32 s18, s1, s22 // (op 1897) 


s_add_i32 s1, s0, s18 // (op 1897) 


s_ashr_i32 s0, s1, s19 // (op 1897) 


s_lshl_b32 s1, s13, 24 // (op 1897) 
s_ashr_i32 s47, s1, 31 // (op 1897) 

s_xor_b32 s1, s0, s47 // (op 1897) 

s_sub_i32 s0, s1, s47 // (op 1897) 


s_mul_hi_i32 s1, s0, s14 // (op 1897) 
s_add_i32 s47, s1, s0 // (op 1897) 

s_ashr_i32 s1, s47, 31 // (op 1897) 
s_cmp_eq_i32 s14, 0 // (op 1897) 
s_mov_b32 s18, scc // (op 1897) 
s_cmp_eq_i32 s18, 0 // (op 1897) 
s_cselect_b32 s19, 0, -1 // (op 1897) 

s_and_b32 s18, s15, 31 // (op 1897) 
s_lshl_b32 s22, 1, s18 // (op 1897) 
s_add_i32 s23, s22, s19 // (op 1897) 


s_and_b32 s19, s1, s23 // (op 1897) 


s_add_i32 s1, s47, s19 // (op 1897) 


s_ashr_i32 s19, s1, s18 // (op 1897) 


s_lshl_b32 s1, s15, 24 // (op 1897) 
s_ashr_i32 s18, s1, 31 // (op 1897) 

s_xor_b32 s1, s19, s18 // (op 1897) 

s_sub_i32 s19, s1, s18 // (op 1897) 


s_add_u32 s22, s4, 32 // (op 1897) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1897) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1897) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1897) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1897) 

s_mul_i32 s1, s19, s22 // (op 1897) 

s_sub_i32 s47, s0, s1 // (op 1897) 


s_mul_hi_i32 s0, s16, s9 // (op 1897) 
s_add_i32 s1, s0, s16 // (op 1897) 

s_ashr_i32 s0, s1, 31 // (op 1897) 
s_cmp_eq_i32 s9, 0 // (op 1897) 
s_mov_b32 s18, scc // (op 1897) 
s_cmp_eq_i32 s18, 0 // (op 1897) 
s_cselect_b32 s19, 0, -1 // (op 1897) 

s_and_b32 s18, s45, 31 // (op 1897) 
s_lshl_b32 s24, 1, s18 // (op 1897) 
s_add_i32 s25, s24, s19 // (op 1897) 


s_and_b32 s19, s0, s25 // (op 1897) 


s_add_i32 s0, s1, s19 // (op 1897) 


s_ashr_i32 s1, s0, s18 // (op 1897) 


s_lshl_b32 s0, s45, 24 // (op 1897) 
s_ashr_i32 s18, s0, 31 // (op 1897) 

s_xor_b32 s0, s1, s18 // (op 1897) 

s_sub_i32 s1, s0, s18 // (op 1897) 


s_mul_i32 s0, s11, s22 // (op 1897) 

s_mul_i32 s18, s8, s0 // (op 1897) 

s_mul_i32 s0, s1, s18 // (op 1897) 

s_sub_u32 s18, s16, s0 // (op 1897) 

s_mul_hi_i32 s0, s18, s12 // (op 1897) 
s_add_i32 s19, s0, s18 // (op 1897) 


s_ashr_i32 s0, s19, 31 // (op 1897) 
s_cmp_eq_i32 s12, 0 // (op 1897) 
s_mov_b32 s18, scc // (op 1897) 
s_cmp_eq_i32 s18, 0 // (op 1897) 
s_cselect_b32 s22, 0, -1 // (op 1897) 

s_and_b32 s18, s44, 31 // (op 1897) 
s_lshl_b32 s23, 1, s18 // (op 1897) 
s_add_i32 s24, s23, s22 // (op 1897) 


s_and_b32 s25, s0, s24 // (op 1897) 


s_add_i32 s0, s19, s25 // (op 1897) 


s_ashr_i32 s19, s0, s18 // (op 1897) 


s_lshl_b32 s0, s44, 24 // (op 1897) 
s_ashr_i32 s18, s0, 31 // (op 1897) 

s_xor_b32 s0, s19, s18 // (op 1897) 

s_sub_i32 s19, s0, s18 // (op 1897) 


s_cmp_eq_i32 s1, 0 // (op 1897) 
s_mov_b32 s0, scc // (op 1897) 

s_cmp_eq_i32 s0, 0 // (op 1897) 
s_cselect_b32 s1, s19, s47 // (op 1897) 



v_lshl_add_u32 v12, s1, 5, v15 // (op 1897) 


v_mul_lo_u32 v23, v12, s31 // (op 1897) most significant: low of low * high
v_mul_hi_u32 v13, v12, s30 // (op 1897) most significant: high of low * low
v_mul_lo_u32 v22, v12, s30 // (op 1897) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 1897) most significant: sum


v_add_co_u32 v12, s[0:1], v20, v22 // (op 1897) least significant half
v_addc_co_u32 v13, s[0:1], v21, v23, s[0:1] // (op 1897) most significant half



v_mul_lo_u32 v21, v13, 6 // (op 1897) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 1897) most significant: high of low * low
v_mul_lo_u32 v20, v12, 6 // (op 1897) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 1897) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[20:21] // (op 1897) 

v_mov_b32 v15, v12 // (op 1897) convert

s_mul_i32 s1, s5, 6 // (op 1899) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 1899) least significant
s_mul_hi_u32 s47, s4, 6 // (op 1899) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1899) most significant: sum

s_mul_i32 s19, s30, s1 // (op 1899) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 1899) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 1899) least significant
s_mul_hi_u32 s22, s30, s0 // (op 1899) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 1899) most significant: sum
s_add_u32 s19, s19, s22 // (op 1899) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1899) 

s_mov_b32 s60, s32 // (op 1899) reg expression
s_mov_b32 s61, s33 // (op 1899) reg expression
s_mov_b32 s62, s0 // (op 1899) reg expression
s_mov_b32 s63, 131072 // (op 1899) call()

s_mul_hi_i32 s47, s16, s10 // (op 1942) 
v_lshrrev_b32 v12, 2, v1 // (op 1942) 
s_add_i32 s0, s47, s16 // (op 1942) 

v_lshlrev_b32 v13, 1, v12 // (op 1942) 

s_ashr_i32 s1, s0, 31 // (op 1942) 
v_and_b32 v12, -4, v1 // (op 1942) 
s_cmp_eq_i32 s10, 0 // (op 1942) 
s_mov_b32 s47, scc // (op 1942) 
v_sub_u32 v20, v1, v12 // (op 1942) 

s_cmp_eq_i32 s47, 0 // (op 1942) 
s_cselect_b32 s18, 0, -1 // (op 1942) 

s_and_b32 s19, s13, 31 // (op 1942) 
s_lshl_b32 s47, 1, s19 // (op 1942) 
s_add_i32 s22, s47, s18 // (op 1942) 


s_and_b32 s18, s1, s22 // (op 1942) 


s_add_i32 s1, s0, s18 // (op 1942) 


s_ashr_i32 s0, s1, s19 // (op 1942) 


s_lshl_b32 s1, s13, 24 // (op 1942) 
s_ashr_i32 s47, s1, 31 // (op 1942) 

s_xor_b32 s1, s0, s47 // (op 1942) 

s_sub_i32 s0, s1, s47 // (op 1942) 


s_mul_hi_i32 s1, s0, s14 // (op 1942) 
s_add_i32 s47, s1, s0 // (op 1942) 

s_ashr_i32 s1, s47, 31 // (op 1942) 
s_cmp_eq_i32 s14, 0 // (op 1942) 
s_mov_b32 s18, scc // (op 1942) 
s_cmp_eq_i32 s18, 0 // (op 1942) 
s_cselect_b32 s19, 0, -1 // (op 1942) 

s_and_b32 s18, s15, 31 // (op 1942) 
s_lshl_b32 s22, 1, s18 // (op 1942) 
s_add_i32 s23, s22, s19 // (op 1942) 


s_and_b32 s19, s1, s23 // (op 1942) 


s_add_i32 s1, s47, s19 // (op 1942) 


s_ashr_i32 s19, s1, s18 // (op 1942) 


s_lshl_b32 s1, s15, 24 // (op 1942) 
s_ashr_i32 s18, s1, 31 // (op 1942) 

s_xor_b32 s1, s19, s18 // (op 1942) 

s_sub_i32 s19, s1, s18 // (op 1942) 


s_mul_i32 s1, s0, s11 // (op 1942) 

s_sub_u32 s0, s16, s1 // (op 1942) 

s_mul_i32 s1, s19, s11 // (op 1942) 
s_add_u32 s1, s1, s0 // (op 1942) 


s_mul_hi_i32 s0, s16, s9 // (op 1942) 
s_add_i32 s47, s0, s16 // (op 1942) 

s_ashr_i32 s0, s47, 31 // (op 1942) 
s_cmp_eq_i32 s9, 0 // (op 1942) 
s_mov_b32 s18, scc // (op 1942) 
s_cmp_eq_i32 s18, 0 // (op 1942) 
s_cselect_b32 s19, 0, -1 // (op 1942) 

s_and_b32 s18, s45, 31 // (op 1942) 
s_lshl_b32 s22, 1, s18 // (op 1942) 
s_add_i32 s23, s22, s19 // (op 1942) 


s_and_b32 s19, s0, s23 // (op 1942) 


s_add_i32 s0, s47, s19 // (op 1942) 


s_ashr_i32 s19, s0, s18 // (op 1942) 


s_lshl_b32 s0, s45, 24 // (op 1942) 
s_ashr_i32 s18, s0, 31 // (op 1942) 

s_xor_b32 s0, s19, s18 // (op 1942) 

s_sub_i32 s19, s0, s18 // (op 1942) 


s_add_u32 s22, s4, 32 // (op 1942) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1942) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1942) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1942) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1942) 

s_mul_i32 s0, s11, s22 // (op 1942) 

s_mul_i32 s18, s8, s0 // (op 1942) 

s_mul_i32 s0, s19, s18 // (op 1942) 

s_sub_u32 s18, s16, s0 // (op 1942) 

s_mov_b32 s0, s18 // (op 1942) 
s_mul_hi_i32 s47, s0, s12 // (op 1942) 
s_add_i32 s22, s47, s0 // (op 1942) 


s_ashr_i32 s0, s22, 31 // (op 1942) 
s_cmp_eq_i32 s12, 0 // (op 1942) 
s_mov_b32 s47, scc // (op 1942) 
s_cmp_eq_i32 s47, 0 // (op 1942) 
s_cselect_b32 s23, 0, -1 // (op 1942) 

s_and_b32 s47, s44, 31 // (op 1942) 
s_lshl_b32 s24, 1, s47 // (op 1942) 
s_add_i32 s25, s24, s23 // (op 1942) 


s_and_b32 s23, s0, s25 // (op 1942) 


s_add_i32 s0, s22, s23 // (op 1942) 


s_ashr_i32 s22, s0, s47 // (op 1942) 


s_lshl_b32 s0, s44, 24 // (op 1942) 
s_ashr_i32 s47, s0, 31 // (op 1942) 

s_xor_b32 s0, s22, s47 // (op 1942) 

s_sub_i32 s22, s0, s47 // (op 1942) 


s_mul_i32 s0, s46, s11 // (op 1942) 
s_add_u32 s24, s6, 128 // (op 1942) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1942) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 1942) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 1942) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 1942) 

s_sub_i32 s23, s24, s0 // (op 1942) 

s_mul_i32 s47, s22, s23 // (op 1942) 


s_sub_u32 s22, s18, s47 // (op 1942) 


s_add_u32 s18, s22, s0 // (op 1942) 


s_cmp_eq_i32 s19, 0 // (op 1942) 
s_mov_b32 s0, scc // (op 1942) 

s_cmp_eq_i32 s0, 0 // (op 1942) 
s_cselect_b32 s19, s18, s1 // (op 1942) 



v_lshl_add_u32 v12, s19, 7, v13 // (op 1942) 


v_mul_lo_u32 v13, v12, s34 // (op 1942) 
v_add_u32 v13, v13, v20 // (op 1942) 


s_mul_i32 s1, s34, s7 // (op 1944) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 1944) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 1944) least significant
s_mul_hi_u32 s18, s34, s6 // (op 1944) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1944) most significant: sum
s_add_u32 s1, s1, s18 // (op 1944) most significant: sum


s_mov_b32 s64, s36 // (op 1944) reg expression
s_mov_b32 s65, s37 // (op 1944) reg expression
s_mov_b32 s66, s0 // (op 1944) reg expression
s_mov_b32 s67, 131072 // (op 1944) call()

s_mul_hi_i32 s47, s16, s10 // (op 1961) 
v_lshrrev_b32 v12, 2, v1 // (op 1961) 
s_add_i32 s0, s47, s16 // (op 1961) 

v_lshlrev_b32 v20, 1, v12 // (op 1961) 

s_ashr_i32 s1, s0, 31 // (op 1961) 
v_and_b32 v12, -4, v1 // (op 1961) 
s_cmp_eq_i32 s10, 0 // (op 1961) 
s_mov_b32 s47, scc // (op 1961) 
v_sub_u32 v21, v1, v12 // (op 1961) 

s_cmp_eq_i32 s47, 0 // (op 1961) 
s_cselect_b32 s18, 0, -1 // (op 1961) 

s_and_b32 s19, s13, 31 // (op 1961) 
s_lshl_b32 s47, 1, s19 // (op 1961) 
s_add_i32 s22, s47, s18 // (op 1961) 


s_and_b32 s18, s1, s22 // (op 1961) 


s_add_i32 s1, s0, s18 // (op 1961) 


s_ashr_i32 s0, s1, s19 // (op 1961) 


s_lshl_b32 s1, s13, 24 // (op 1961) 
s_ashr_i32 s47, s1, 31 // (op 1961) 

s_xor_b32 s1, s0, s47 // (op 1961) 

s_sub_i32 s0, s1, s47 // (op 1961) 


s_mul_hi_i32 s1, s0, s14 // (op 1961) 
s_add_i32 s47, s1, s0 // (op 1961) 

s_ashr_i32 s1, s47, 31 // (op 1961) 
s_cmp_eq_i32 s14, 0 // (op 1961) 
s_mov_b32 s18, scc // (op 1961) 
s_cmp_eq_i32 s18, 0 // (op 1961) 
s_cselect_b32 s19, 0, -1 // (op 1961) 

s_and_b32 s18, s15, 31 // (op 1961) 
s_lshl_b32 s22, 1, s18 // (op 1961) 
s_add_i32 s23, s22, s19 // (op 1961) 


s_and_b32 s19, s1, s23 // (op 1961) 


s_add_i32 s1, s47, s19 // (op 1961) 


s_ashr_i32 s19, s1, s18 // (op 1961) 


s_lshl_b32 s1, s15, 24 // (op 1961) 
s_ashr_i32 s18, s1, 31 // (op 1961) 

s_xor_b32 s1, s19, s18 // (op 1961) 

s_sub_i32 s19, s1, s18 // (op 1961) 


s_mul_i32 s1, s0, s11 // (op 1961) 

s_sub_u32 s0, s16, s1 // (op 1961) 

s_mul_i32 s1, s19, s11 // (op 1961) 
s_add_u32 s1, s1, s0 // (op 1961) 


s_mul_hi_i32 s0, s16, s9 // (op 1961) 
s_add_i32 s47, s0, s16 // (op 1961) 

s_ashr_i32 s0, s47, 31 // (op 1961) 
s_cmp_eq_i32 s9, 0 // (op 1961) 
s_mov_b32 s18, scc // (op 1961) 
s_cmp_eq_i32 s18, 0 // (op 1961) 
s_cselect_b32 s19, 0, -1 // (op 1961) 

s_and_b32 s18, s45, 31 // (op 1961) 
s_lshl_b32 s22, 1, s18 // (op 1961) 
s_add_i32 s23, s22, s19 // (op 1961) 


s_and_b32 s19, s0, s23 // (op 1961) 


s_add_i32 s0, s47, s19 // (op 1961) 


s_ashr_i32 s19, s0, s18 // (op 1961) 


s_lshl_b32 s0, s45, 24 // (op 1961) 
s_ashr_i32 s18, s0, 31 // (op 1961) 

s_xor_b32 s0, s19, s18 // (op 1961) 

s_sub_i32 s19, s0, s18 // (op 1961) 


s_add_u32 s22, s4, 32 // (op 1961) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1961) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1961) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1961) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1961) 

s_mul_i32 s0, s11, s22 // (op 1961) 

s_mul_i32 s18, s8, s0 // (op 1961) 

s_mul_i32 s0, s19, s18 // (op 1961) 

s_sub_u32 s18, s16, s0 // (op 1961) 

s_mov_b32 s0, s18 // (op 1961) 
s_mul_hi_i32 s47, s0, s12 // (op 1961) 
s_add_i32 s22, s47, s0 // (op 1961) 


s_ashr_i32 s0, s22, 31 // (op 1961) 
s_cmp_eq_i32 s12, 0 // (op 1961) 
s_mov_b32 s47, scc // (op 1961) 
s_cmp_eq_i32 s47, 0 // (op 1961) 
s_cselect_b32 s23, 0, -1 // (op 1961) 

s_and_b32 s47, s44, 31 // (op 1961) 
s_lshl_b32 s24, 1, s47 // (op 1961) 
s_add_i32 s25, s24, s23 // (op 1961) 


s_and_b32 s23, s0, s25 // (op 1961) 


s_add_i32 s0, s22, s23 // (op 1961) 


s_ashr_i32 s22, s0, s47 // (op 1961) 


s_lshl_b32 s0, s44, 24 // (op 1961) 
s_ashr_i32 s47, s0, 31 // (op 1961) 

s_xor_b32 s0, s22, s47 // (op 1961) 

s_sub_i32 s22, s0, s47 // (op 1961) 


s_mul_i32 s0, s46, s11 // (op 1961) 
s_add_u32 s24, s6, 128 // (op 1961) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1961) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 1961) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 1961) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 7 // (op 1961) 

s_sub_i32 s23, s24, s0 // (op 1961) 

s_mul_i32 s47, s22, s23 // (op 1961) 


s_sub_u32 s22, s18, s47 // (op 1961) 


s_add_u32 s18, s22, s0 // (op 1961) 


s_cmp_eq_i32 s19, 0 // (op 1961) 
s_mov_b32 s0, scc // (op 1961) 

s_cmp_eq_i32 s0, 0 // (op 1961) 
s_cselect_b32 s19, s18, s1 // (op 1961) 



v_lshl_add_u32 v12, s19, 7, v20 // (op 1961) 


v_mul_lo_u32 v20, v12, s34 // (op 1961) 
v_add_u32 v20, v20, v21 // (op 1961) 


s_mul_i32 s1, s34, s7 // (op 1963) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 1963) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 1963) least significant
s_mul_hi_u32 s18, s34, s6 // (op 1963) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1963) most significant: sum
s_add_u32 s1, s1, s18 // (op 1963) most significant: sum


s_mov_b32 s64, s36 // (op 1963) reg expression
s_mov_b32 s65, s37 // (op 1963) reg expression
s_mov_b32 s66, s0 // (op 1963) reg expression
s_mov_b32 s67, 131072 // (op 1963) call()

s_mul_hi_i32 s47, s16, s10 // (op 2006) 
v_lshrrev_b32 v12, 6, v1 // (op 2006) 
s_add_i32 s0, s47, s16 // (op 2006) 

v_and_b32 v21, 1, v12 // (op 2006) 

s_ashr_i32 s1, s0, 31 // (op 2006) 
v_and_b32 v12, 15, v1 // (op 2006) 
s_cmp_eq_i32 s10, 0 // (op 2006) 
s_mov_b32 s47, scc // (op 2006) 
v_lshl_add_u32 v22, v21, 4, v12 // (op 2006) 


s_cmp_eq_i32 s47, 0 // (op 2006) 
s_cselect_b32 s18, 0, -1 // (op 2006) 

v_and_b32 v12, 63, v1 // (op 2006) 
s_and_b32 s19, s13, 31 // (op 2006) 
v_lshrrev_b32 v21, 4, v12 // (op 2006) 

s_lshl_b32 s47, 1, s19 // (op 2006) 
s_add_i32 s22, s47, s18 // (op 2006) 


s_and_b32 s18, s1, s22 // (op 2006) 


s_add_i32 s1, s0, s18 // (op 2006) 


s_ashr_i32 s0, s1, s19 // (op 2006) 


s_lshl_b32 s1, s13, 24 // (op 2006) 
s_ashr_i32 s47, s1, 31 // (op 2006) 

s_xor_b32 s1, s0, s47 // (op 2006) 

s_sub_i32 s0, s1, s47 // (op 2006) 


s_mul_hi_i32 s1, s0, s14 // (op 2006) 
s_add_i32 s47, s1, s0 // (op 2006) 

s_ashr_i32 s1, s47, 31 // (op 2006) 
s_cmp_eq_i32 s14, 0 // (op 2006) 
s_mov_b32 s18, scc // (op 2006) 
s_cmp_eq_i32 s18, 0 // (op 2006) 
s_cselect_b32 s19, 0, -1 // (op 2006) 

s_and_b32 s18, s15, 31 // (op 2006) 
s_lshl_b32 s22, 1, s18 // (op 2006) 
s_add_i32 s23, s22, s19 // (op 2006) 


s_and_b32 s19, s1, s23 // (op 2006) 


s_add_i32 s1, s47, s19 // (op 2006) 


s_ashr_i32 s19, s1, s18 // (op 2006) 


s_lshl_b32 s1, s15, 24 // (op 2006) 
s_ashr_i32 s18, s1, 31 // (op 2006) 

s_xor_b32 s1, s19, s18 // (op 2006) 

s_sub_i32 s19, s1, s18 // (op 2006) 


s_add_u32 s22, s4, 32 // (op 2006) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 2006) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2006) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2006) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 2006) 

s_mul_i32 s1, s19, s22 // (op 2006) 

s_sub_i32 s47, s0, s1 // (op 2006) 


s_mul_hi_i32 s0, s16, s9 // (op 2006) 
s_add_i32 s1, s0, s16 // (op 2006) 

s_ashr_i32 s0, s1, 31 // (op 2006) 
s_cmp_eq_i32 s9, 0 // (op 2006) 
s_mov_b32 s18, scc // (op 2006) 
s_cmp_eq_i32 s18, 0 // (op 2006) 
s_cselect_b32 s19, 0, -1 // (op 2006) 

s_and_b32 s18, s45, 31 // (op 2006) 
s_lshl_b32 s24, 1, s18 // (op 2006) 
s_add_i32 s25, s24, s19 // (op 2006) 


s_and_b32 s19, s0, s25 // (op 2006) 


s_add_i32 s0, s1, s19 // (op 2006) 


s_ashr_i32 s1, s0, s18 // (op 2006) 


s_lshl_b32 s0, s45, 24 // (op 2006) 
s_ashr_i32 s18, s0, 31 // (op 2006) 

s_xor_b32 s0, s1, s18 // (op 2006) 

s_sub_i32 s1, s0, s18 // (op 2006) 


s_mul_i32 s0, s11, s22 // (op 2006) 

s_mul_i32 s18, s8, s0 // (op 2006) 

s_mul_i32 s0, s1, s18 // (op 2006) 

s_sub_u32 s18, s16, s0 // (op 2006) 

s_mul_hi_i32 s0, s18, s12 // (op 2006) 
s_add_i32 s19, s0, s18 // (op 2006) 


s_ashr_i32 s0, s19, 31 // (op 2006) 
s_cmp_eq_i32 s12, 0 // (op 2006) 
s_mov_b32 s18, scc // (op 2006) 
s_cmp_eq_i32 s18, 0 // (op 2006) 
s_cselect_b32 s22, 0, -1 // (op 2006) 

s_and_b32 s18, s44, 31 // (op 2006) 
s_lshl_b32 s23, 1, s18 // (op 2006) 
s_add_i32 s24, s23, s22 // (op 2006) 


s_and_b32 s25, s0, s24 // (op 2006) 


s_add_i32 s0, s19, s25 // (op 2006) 


s_ashr_i32 s19, s0, s18 // (op 2006) 


s_lshl_b32 s0, s44, 24 // (op 2006) 
s_ashr_i32 s18, s0, 31 // (op 2006) 

s_xor_b32 s0, s19, s18 // (op 2006) 

s_sub_i32 s19, s0, s18 // (op 2006) 


s_cmp_eq_i32 s1, 0 // (op 2006) 
s_mov_b32 s0, scc // (op 2006) 

s_cmp_eq_i32 s0, 0 // (op 2006) 
s_cselect_b32 s1, s19, s47 // (op 2006) 



v_lshl_add_u32 v12, s1, 5, v22 // (op 2006) 


v_mul_lo_u32 v22, v12, s38 // (op 2006) 

v_add_u32 v12, v21, v22 // (op 2006) 


s_mul_i32 s1, s38, s5 // (op 2008) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 2008) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 2008) least significant
s_mul_hi_u32 s18, s38, s4 // (op 2008) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2008) most significant: sum
s_add_u32 s1, s1, s18 // (op 2008) most significant: sum


s_mov_b32 s68, s40 // (op 2008) reg expression
s_mov_b32 s69, s41 // (op 2008) reg expression
s_mov_b32 s70, s0 // (op 2008) reg expression
s_mov_b32 s71, 131072 // (op 2008) call()

s_mul_hi_i32 s47, s16, s10 // (op 2025) 
v_lshrrev_b32 v21, 6, v1 // (op 2025) 
s_add_i32 s0, s47, s16 // (op 2025) 

v_and_b32 v22, 1, v21 // (op 2025) 

s_ashr_i32 s1, s0, 31 // (op 2025) 
v_and_b32 v21, 15, v1 // (op 2025) 
s_cmp_eq_i32 s10, 0 // (op 2025) 
s_mov_b32 s47, scc // (op 2025) 
v_lshl_add_u32 v23, v22, 4, v21 // (op 2025) 


s_cmp_eq_i32 s47, 0 // (op 2025) 
s_cselect_b32 s18, 0, -1 // (op 2025) 

v_and_b32 v21, 63, v1 // (op 2025) 
s_and_b32 s19, s13, 31 // (op 2025) 
v_lshrrev_b32 v22, 4, v21 // (op 2025) 

s_lshl_b32 s47, 1, s19 // (op 2025) 
s_add_i32 s22, s47, s18 // (op 2025) 


s_and_b32 s18, s1, s22 // (op 2025) 


s_add_i32 s1, s0, s18 // (op 2025) 


s_ashr_i32 s0, s1, s19 // (op 2025) 


s_lshl_b32 s1, s13, 24 // (op 2025) 
s_ashr_i32 s47, s1, 31 // (op 2025) 

s_xor_b32 s1, s0, s47 // (op 2025) 

s_sub_i32 s0, s1, s47 // (op 2025) 


s_mul_hi_i32 s1, s0, s14 // (op 2025) 
s_add_i32 s47, s1, s0 // (op 2025) 

s_ashr_i32 s1, s47, 31 // (op 2025) 
s_cmp_eq_i32 s14, 0 // (op 2025) 
s_mov_b32 s18, scc // (op 2025) 
s_cmp_eq_i32 s18, 0 // (op 2025) 
s_cselect_b32 s19, 0, -1 // (op 2025) 

s_and_b32 s18, s15, 31 // (op 2025) 
s_lshl_b32 s22, 1, s18 // (op 2025) 
s_add_i32 s23, s22, s19 // (op 2025) 


s_and_b32 s19, s1, s23 // (op 2025) 


s_add_i32 s1, s47, s19 // (op 2025) 


s_ashr_i32 s19, s1, s18 // (op 2025) 


s_lshl_b32 s1, s15, 24 // (op 2025) 
s_ashr_i32 s18, s1, 31 // (op 2025) 

s_xor_b32 s1, s19, s18 // (op 2025) 

s_sub_i32 s19, s1, s18 // (op 2025) 


s_add_u32 s22, s4, 32 // (op 2025) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 2025) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2025) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2025) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 2025) 

s_mul_i32 s1, s19, s22 // (op 2025) 

s_sub_i32 s47, s0, s1 // (op 2025) 


s_mul_hi_i32 s0, s16, s9 // (op 2025) 
s_add_i32 s1, s0, s16 // (op 2025) 

s_ashr_i32 s0, s1, 31 // (op 2025) 
s_cmp_eq_i32 s9, 0 // (op 2025) 
s_mov_b32 s18, scc // (op 2025) 
s_cmp_eq_i32 s18, 0 // (op 2025) 
s_cselect_b32 s19, 0, -1 // (op 2025) 

s_and_b32 s18, s45, 31 // (op 2025) 
s_lshl_b32 s24, 1, s18 // (op 2025) 
s_add_i32 s25, s24, s19 // (op 2025) 


s_and_b32 s19, s0, s25 // (op 2025) 


s_add_i32 s0, s1, s19 // (op 2025) 


s_ashr_i32 s1, s0, s18 // (op 2025) 


s_lshl_b32 s0, s45, 24 // (op 2025) 
s_ashr_i32 s18, s0, 31 // (op 2025) 

s_xor_b32 s0, s1, s18 // (op 2025) 

s_sub_i32 s1, s0, s18 // (op 2025) 


s_mul_i32 s0, s11, s22 // (op 2025) 

s_mul_i32 s18, s8, s0 // (op 2025) 

s_mul_i32 s0, s1, s18 // (op 2025) 

s_sub_u32 s18, s16, s0 // (op 2025) 

s_mul_hi_i32 s0, s18, s12 // (op 2025) 
s_add_i32 s19, s0, s18 // (op 2025) 


s_ashr_i32 s0, s19, 31 // (op 2025) 
s_cmp_eq_i32 s12, 0 // (op 2025) 
s_mov_b32 s18, scc // (op 2025) 
s_cmp_eq_i32 s18, 0 // (op 2025) 
s_cselect_b32 s22, 0, -1 // (op 2025) 

s_and_b32 s18, s44, 31 // (op 2025) 
s_lshl_b32 s23, 1, s18 // (op 2025) 
s_add_i32 s24, s23, s22 // (op 2025) 


s_and_b32 s25, s0, s24 // (op 2025) 


s_add_i32 s0, s19, s25 // (op 2025) 


s_ashr_i32 s19, s0, s18 // (op 2025) 


s_lshl_b32 s0, s44, 24 // (op 2025) 
s_ashr_i32 s18, s0, 31 // (op 2025) 

s_xor_b32 s0, s19, s18 // (op 2025) 

s_sub_i32 s19, s0, s18 // (op 2025) 


s_cmp_eq_i32 s1, 0 // (op 2025) 
s_mov_b32 s0, scc // (op 2025) 

s_cmp_eq_i32 s0, 0 // (op 2025) 
s_cselect_b32 s1, s19, s47 // (op 2025) 



v_lshl_add_u32 v21, s1, 5, v23 // (op 2025) 


v_mul_lo_u32 v23, v21, s38 // (op 2025) 

v_add_u32 v21, v22, v23 // (op 2025) 


s_mul_i32 s1, s38, s5 // (op 2027) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 2027) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 2027) least significant
s_mul_hi_u32 s18, s38, s4 // (op 2027) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2027) most significant: sum
s_add_u32 s1, s1, s18 // (op 2027) most significant: sum


s_mov_b32 s68, s40 // (op 2027) reg expression
s_mov_b32 s69, s41 // (op 2027) reg expression
s_mov_b32 s70, s0 // (op 2027) reg expression
s_mov_b32 s71, 131072 // (op 2027) call()




buffer_load_ubyte v22, v12, s[68:71], 0 offen offset: 0 // (op 1616) Load value



buffer_load_dwordx3 v[24:26], v14, s[60:63], 0 offen // (op 1620) Load value
v_and_b32 v23, -8, v1 // (op 2091) 
v_sub_u32 v27, v1, v23 // (op 2091) 

v_lshlrev_b32 v23, 4, v27 // (op 2091) 

v_lshrrev_b32 v27, 3, v1 // (op 2091) 
v_lshl_add_u32 v28, v27, 7, v23 // (op 2091) 


s_nop 0
v_readfirstlane_b32 s47, v28 // (op 2091) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 1767) 
v_mov_b64 v[28:29], s[0:1] // (op 1767) convert


v_mov_b64 v[30:31], v[16:17] // (op 1767) 
s_mov_b32 m0, s47 // (op 1767) reg expression
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 1767) Load value direct to lds
v_add_co_u32 v30, s[0:1], v30, v28 // (op 1767) least significant half
v_addc_co_u32 v31, s[0:1], 0, v31, s[0:1] // (op 1767) most significant half

s_add_u32 m0, m0, 4096 // (op 1767) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 1767) Load value direct to lds
v_add_co_u32 v30, s[0:1], v30, v28 // (op 1767) least significant half
v_addc_co_u32 v31, s[0:1], 0, v31, s[0:1] // (op 1767) most significant half

s_add_u32 m0, m0, 4096 // (op 1767) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 1767) Load value direct to lds
v_add_co_u32 v30, s[0:1], v30, v28 // (op 1767) least significant half
v_addc_co_u32 v31, s[0:1], 0, v31, s[0:1] // (op 1767) most significant half

s_add_u32 m0, m0, 4096 // (op 1767) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 1767) Load value direct to lds




v_mov_b64 v[28:29], s[34:35] // (op 1626) convert

v_mov_b32 v23, v13 // (op 1626) 
buffer_load_ubyte v30, v23, s[64:67], 0 offen offset: 0 // (op 1626) Load value
v_add_u32 v23, v23, v28 // (op 1626) 
buffer_load_ubyte v31, v23, s[64:67], 0 offen offset: 0 // (op 1626) Load value





buffer_load_ubyte v23, v12, s[68:71], 0 offen offset:4 // (op 1630) Load value




buffer_load_dwordx3 v[32:34], v14, s[60:63], 0 offen offset:96 // (op 1634) Load value

v_and_b32 v12, -8, v1 // (op 2264) 
v_sub_u32 v14, v1, v12 // (op 2264) 

v_lshlrev_b32 v12, 4, v14 // (op 2264) 

v_lshrrev_b32 v14, 3, v1 // (op 2264) 
v_lshl_add_u32 v27, v14, 7, v12 // (op 2264) 


s_nop 0
v_readfirstlane_b32 s47, v27 // (op 2264) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 1772) 
v_mov_b64 v[28:29], s[0:1] // (op 1772) convert


v_mov_b64 v[36:37], v[16:17] // (op 1772) 
s_add_u32 s0, 16384, s47 // (op 1772) 
s_mov_b32 m0, s0 // (op 1772) reg expression
s_mov_b32 s1, 128 // (op 1772) call()
buffer_load_dwordx4 v36, s[48:51], s1 offen offset: 0 lds // (op 1772) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v28 // (op 1772) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1772) most significant half

s_add_u32 m0, m0, 4096 // (op 1772) 
buffer_load_dwordx4 v36, s[48:51], s1 offen offset: 0 lds // (op 1772) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v28 // (op 1772) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1772) most significant half

s_add_u32 m0, m0, 4096 // (op 1772) 
buffer_load_dwordx4 v36, s[48:51], s1 offen offset: 0 lds // (op 1772) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v28 // (op 1772) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 1772) most significant half

s_add_u32 m0, m0, 4096 // (op 1772) 
buffer_load_dwordx4 v36, s[48:51], s1 offen offset: 0 lds // (op 1772) Load value direct to lds







v_mov_b64 v[16:17], s[34:35] // (op 1640) convert

v_mov_b32 v12, v13 // (op 1640) 
buffer_load_ubyte v28, v12, s[64:67], 0 offen offset:4 // (op 1640) Load value
v_add_u32 v12, v12, v16 // (op 1640) 
buffer_load_ubyte v29, v12, s[64:67], 0 offen offset:4 // (op 1640) Load value



v_and_b32 v27, -8, v1 // (op 2149) 
v_sub_u32 v12, v1, v27 // (op 2149) 

v_lshlrev_b32 v27, 4, v12 // (op 2149) 

v_lshrrev_b32 v12, 3, v1 // (op 2149) 
v_lshl_add_u32 v13, v12, 7, v27 // (op 2149) 


v_mul_lo_u32 v12, v13, 6 // (op 2149) 

v_lshrrev_b32 v27, 3, v12 // (op 2149) 




s_waitcnt vmcnt(14)
ds_write_b96 v27, v[24:26] offset:32768 // (op 1642) Store local data 

v_and_b32 v27, -4, v1 // (op 2202) 
v_sub_u32 v12, v1, v27 // (op 2202) 

v_lshrrev_b32 v27, 2, v1 // (op 2202) 
v_lshl_add_u32 v13, v27, 3, v12 // (op 2202) 





s_waitcnt vmcnt(9)
ds_write_b8 v13, v30 offset:35840 // (op 1644) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v13, v31 offset:35844 // (op 1644) Store local data 

v_and_b32 v27, -8, v1 // (op 2063) 
v_sub_u32 v12, v1, v27 // (op 2063) 

v_lshlrev_b32 v27, 4, v12 // (op 2063) 

v_lshrrev_b32 v12, 3, v1 // (op 2063) 
v_lshl_add_u32 v13, v12, 7, v27 // (op 2063) 


s_nop 0
v_readfirstlane_b32 s47, v13 // (op 2063) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -8, v1 // (op 2120) 
v_sub_u32 v12, v1, v27 // (op 2120) 

v_lshlrev_b32 v27, 4, v12 // (op 2120) 

v_lshrrev_b32 v12, 3, v1 // (op 2120) 
v_lshl_add_u32 v13, v12, 7, v27 // (op 2120) 


v_mul_lo_u32 v12, v13, 6 // (op 2120) 

v_lshrrev_b32 v27, 3, v12 // (op 2120) 

v_and_b32 v12, -4, v1 // (op 2173) 
v_sub_u32 v13, v1, v12 // (op 2173) 

v_lshrrev_b32 v12, 2, v1 // (op 2173) 
v_lshl_add_u32 v14, v12, 3, v13 // (op 2173) 


v_and_b32 v12, -8, v1 // (op 2249) 
v_sub_u32 v13, v1, v12 // (op 2249) 

v_lshlrev_b32 v12, 4, v13 // (op 2249) 

v_lshrrev_b32 v13, 3, v1 // (op 2249) 
v_lshl_add_u32 v16, v13, 7, v12 // (op 2249) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 2249) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -8, v1 // (op 2293) 
v_sub_u32 v13, v1, v12 // (op 2293) 

v_lshlrev_b32 v12, 4, v13 // (op 2293) 

v_lshrrev_b32 v13, 3, v1 // (op 2293) 
v_lshl_add_u32 v16, v13, 7, v12 // (op 2293) 


v_mul_lo_u32 v17, v16, 6 // (op 2293) 

v_lshrrev_b32 v16, 3, v17 // (op 2293) 

v_and_b32 v17, -4, v1 // (op 2317) 
v_sub_u32 v12, v1, v17 // (op 2317) 

v_lshrrev_b32 v13, 2, v1 // (op 2317) 
v_lshl_add_u32 v17, v13, 3, v12 // (op 2317) 


s_mov_b32 s1, 0 // (op 639) call()
s_ashr_i64 s[18:19], s[42:43], 7 // (op 638) 
s_mov_b32 s22, s18 // (op 638) convert

s_and_b32 s18, s22, 4294967294 // (op 638) 

s_cmp_lt_u32 s1, s18 // (op 638) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoop_638 // (op 638) Condition: Top (jump to RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoop_638 if false)
RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoop_638:
s_barrier  // (op 2381) 
s_add_u32 s1, s1, 2 // (op 640) 



ds_read_b128 v[36:39], v7 // (op 67) Load local data 
ds_read_b128 v[40:43], v7 offset:64 // (op 67) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2394) 



ds_read_b128 v[44:47], v3 offset:32768 // (op 72) Load local data 
ds_read_b64 v[48:49], v3 offset:32784 // (op 72) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2408) 



ds_read_u8 v35, v0 offset:35840 // (op 77) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:49], a[0:3], v35, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 84) 
ds_read_b128 v[50:53], v7 offset:4096 // (op 344) Load local data 


ds_read_b128 v[54:57], v7 offset:4160 // (op 344) Load local data 



ds_read_u8 v12, v0 offset:35968 // (op 348) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[44:49], a[4:7], v12, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 350) 
ds_read_b128 v[36:39], v7 offset:8192 // (op 506) Load local data 


ds_read_b128 v[40:43], v7 offset:8256 // (op 506) Load local data 



ds_read_u8 v35, v0 offset:36096 // (op 510) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[36:43], v[44:49], a[8:11], v35, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 512) 
ds_read_b128 v[50:53], v7 offset:12288 // (op 668) Load local data 


ds_read_b128 v[54:57], v7 offset:12352 // (op 668) Load local data 



ds_read_u8 v12, v0 offset:36224 // (op 672) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:57], v[44:49], a[12:15], v12, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 674) 





v_mov_b32 v12, v21 // (op 82) 

buffer_load_ubyte v22, v12, s[68:71], 0 offen offset:8 // (op 82) Load value


v_mov_b32 v13, v15 // (op 10) 

buffer_load_dwordx3 v[24:26], v13, s[60:63], 0 offen offset:192 // (op 10) Load value
s_barrier  // (op 2384) 

s_lshl_b64 s[18:19], s[26:27], 5 // (op 1751) 
v_mov_b64 v[36:37], s[18:19] // (op 1751) convert


v_mov_b64 v[38:39], v[18:19] // (op 1751) 
v_mov_b64 v[40:41], v[38:39] // (op 1751) 
s_mov_b32 m0, s47 // (op 1751) reg expression
s_mov_b32 s18, 256 // (op 1751) call()
buffer_load_dwordx4 v40, s[48:51], s18 offen offset: 0 lds // (op 1751) Load value direct to lds
v_add_co_u32 v40, s[22:23], v40, v36 // (op 1751) least significant half
v_addc_co_u32 v41, s[22:23], 0, v41, s[22:23] // (op 1751) most significant half

s_add_u32 m0, m0, 4096 // (op 1751) 
buffer_load_dwordx4 v40, s[48:51], s18 offen offset: 0 lds // (op 1751) Load value direct to lds
v_add_co_u32 v40, s[22:23], v40, v36 // (op 1751) least significant half
v_addc_co_u32 v41, s[22:23], 0, v41, s[22:23] // (op 1751) most significant half

s_add_u32 m0, m0, 4096 // (op 1751) 
buffer_load_dwordx4 v40, s[48:51], s18 offen offset: 0 lds // (op 1751) Load value direct to lds
v_add_co_u32 v40, s[22:23], v40, v36 // (op 1751) least significant half
v_addc_co_u32 v41, s[22:23], 0, v41, s[22:23] // (op 1751) most significant half

s_add_u32 m0, m0, 4096 // (op 1751) 
buffer_load_dwordx4 v40, s[48:51], s18 offen offset: 0 lds // (op 1751) Load value direct to lds




v_mov_b64 v[36:37], s[34:35] // (op 17) convert

v_mov_b32 v35, v20 // (op 17) 
v_mov_b32 v40, v35 // (op 17) 
buffer_load_ubyte v30, v40, s[64:67], 0 offen offset:8 // (op 17) Load value
v_add_u32 v40, v40, v36 // (op 17) 
buffer_load_ubyte v31, v40, s[64:67], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 2431) 



s_waitcnt vmcnt(14)
ds_write_b96 v16, v[32:34] offset:36352 // (op 239) Store local data 
s_barrier  // (op 2437) 



s_waitcnt vmcnt(9)
ds_write_b8 v17, v28 offset:39424 // (op 240) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v17, v29 offset:39428 // (op 240) Store local data 
s_barrier  // (op 2422) 



ds_read_b128 v[40:43], v11 offset:16384 // (op 231) Load local data 
ds_read_b128 v[44:47], v11 offset:16448 // (op 231) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2428) 



ds_read_b128 v[48:51], v4 offset:36352 // (op 233) Load local data 
ds_read_b64 v[52:53], v4 offset:36368 // (op 233) Load local data 
s_barrier  // (op 2470) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 2434) 



ds_read_u8 v36, v10 offset:39424 // (op 235) Load local data 



ds_read_b128 v[54:57], v11 offset:20480 // (op 357) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[48:53], a[0:3], v36, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 238) 
ds_read_b128 v[58:61], v11 offset:20544 // (op 357) Load local data 





ds_read_u8 v36, v10 offset:39552 // (op 361) Load local data 



ds_read_b128 v[40:43], v11 offset:24576 // (op 519) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:61], v[48:53], a[4:7], v36, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 351) 
ds_read_b128 v[44:47], v11 offset:24640 // (op 519) Load local data 





ds_read_u8 v36, v10 offset:39680 // (op 523) Load local data 



ds_read_b128 v[54:57], v11 offset:28672 // (op 681) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[40:47], v[48:53], a[8:11], v36, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 513) 
ds_read_b128 v[58:61], v11 offset:28736 // (op 681) Load local data 





ds_read_u8 v36, v10 offset:39808 // (op 685) Load local data 
s_barrier  // (op 2520) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:61], v[48:53], a[12:15], v36, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 675) 






buffer_load_ubyte v23, v12, s[68:71], 0 offen offset:12 // (op 237) Load value




v_add_u32 v21, 8, v21 // (op 2017) 
buffer_load_dwordx3 v[32:34], v13, s[60:63], 0 offen offset:288 // (op 241) Load value

s_barrier  // (op 2425) 
v_add_u32 v15, 192, v15 // (op 1889) 

s_lshl_b64 s[18:19], s[26:27], 5 // (op 1762) 
v_mov_b64 v[12:13], s[18:19] // (op 1762) convert


v_mov_b64 v[36:37], v[38:39] // (op 1762) 
s_add_u32 s18, 16384, s0 // (op 1762) 
s_mov_b32 m0, s18 // (op 1762) reg expression
s_mov_b32 s19, 384 // (op 1762) call()
buffer_load_dwordx4 v36, s[48:51], s19 offen offset: 0 lds // (op 1762) Load value direct to lds
v_add_co_u32 v36, s[22:23], v36, v12 // (op 1762) least significant half
v_addc_co_u32 v37, s[22:23], 0, v37, s[22:23] // (op 1762) most significant half

s_add_u32 m0, m0, 4096 // (op 1762) 
buffer_load_dwordx4 v36, s[48:51], s19 offen offset: 0 lds // (op 1762) Load value direct to lds
v_add_co_u32 v36, s[22:23], v36, v12 // (op 1762) least significant half
v_addc_co_u32 v37, s[22:23], 0, v37, s[22:23] // (op 1762) most significant half

s_add_u32 m0, m0, 4096 // (op 1762) 
buffer_load_dwordx4 v36, s[48:51], s19 offen offset: 0 lds // (op 1762) Load value direct to lds
v_add_co_u32 v36, s[22:23], v36, v12 // (op 1762) least significant half
v_addc_co_u32 v37, s[22:23], 0, v37, s[22:23] // (op 1762) most significant half

s_add_u32 m0, m0, 4096 // (op 1762) 
buffer_load_dwordx4 v36, s[48:51], s19 offen offset: 0 lds // (op 1762) Load value direct to lds






v_add_co_u32 v18, vcc, 256, v18 // (op 1825) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 1825) most significant half
v_mov_b64 v[12:13], s[34:35] // (op 242) convert

v_mov_b32 v36, v35 // (op 242) 
buffer_load_ubyte v28, v36, s[64:67], 0 offen offset:12 // (op 242) Load value
v_add_u32 v36, v36, v12 // (op 242) 
buffer_load_ubyte v29, v36, s[64:67], 0 offen offset:12 // (op 242) Load value



s_barrier  // (op 2397) 
v_add_u32 v20, 8, v20 // (op 1953) 



s_waitcnt vmcnt(14)
ds_write_b96 v27, v[24:26] offset:32768 // (op 13) Store local data 
s_barrier  // (op 2411) 



s_waitcnt vmcnt(9)
ds_write_b8 v14, v30 offset:35840 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v14, v31 offset:35844 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[42:43], 7 // (op 638) 
s_mov_b32 s22, s18 // (op 638) convert

s_and_b32 s18, s22, 4294967294 // (op 638) 

s_cmp_lt_u32 s1, s18 // (op 638) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoop_638 // (op 638) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoop_638 if true)
RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoop_638:














s_barrier  // (op 2502) 
s_mul_hi_i32 s47, s16, s10 // (op 1785) 
v_lshrrev_b32 v12, 3, v1 // (op 1785) 
s_add_i32 s0, s47, s16 // (op 1785) 

v_and_b32 v13, -8, v1 // (op 1785) 
s_ashr_i32 s1, s0, 31 // (op 1785) 
v_sub_u32 v14, v1, v13 // (op 1785) 

s_cmp_eq_i32 s10, 0 // (op 1785) 
s_mov_b32 s47, scc // (op 1785) 
v_lshlrev_b32 v13, 4, v14 // (op 1785) 

s_cmp_eq_i32 s47, 0 // (op 1785) 
s_cselect_b32 s18, 0, -1 // (op 1785) 

s_and_b32 s19, s13, 31 // (op 1785) 
s_lshl_b32 s47, 1, s19 // (op 1785) 
s_add_i32 s22, s47, s18 // (op 1785) 


s_and_b32 s18, s1, s22 // (op 1785) 


s_add_i32 s1, s0, s18 // (op 1785) 


s_ashr_i32 s0, s1, s19 // (op 1785) 


s_lshl_b32 s1, s13, 24 // (op 1785) 
s_ashr_i32 s47, s1, 31 // (op 1785) 

s_xor_b32 s1, s0, s47 // (op 1785) 

s_sub_i32 s0, s1, s47 // (op 1785) 


s_mul_hi_i32 s1, s0, s14 // (op 1785) 
s_add_i32 s47, s1, s0 // (op 1785) 

s_ashr_i32 s1, s47, 31 // (op 1785) 
s_cmp_eq_i32 s14, 0 // (op 1785) 
s_mov_b32 s18, scc // (op 1785) 
s_cmp_eq_i32 s18, 0 // (op 1785) 
s_cselect_b32 s19, 0, -1 // (op 1785) 

s_and_b32 s18, s15, 31 // (op 1785) 
s_lshl_b32 s22, 1, s18 // (op 1785) 
s_add_i32 s23, s22, s19 // (op 1785) 


s_and_b32 s19, s1, s23 // (op 1785) 


s_add_i32 s1, s47, s19 // (op 1785) 


s_ashr_i32 s19, s1, s18 // (op 1785) 


s_lshl_b32 s1, s15, 24 // (op 1785) 
s_ashr_i32 s18, s1, 31 // (op 1785) 

s_xor_b32 s1, s19, s18 // (op 1785) 

s_sub_i32 s19, s1, s18 // (op 1785) 


s_mul_i32 s1, s0, s11 // (op 1785) 

s_sub_u32 s0, s16, s1 // (op 1785) 

s_mul_i32 s1, s19, s11 // (op 1785) 
s_add_u32 s1, s1, s0 // (op 1785) 


s_mul_hi_i32 s0, s16, s9 // (op 1785) 
s_add_i32 s47, s0, s16 // (op 1785) 

s_ashr_i32 s0, s47, 31 // (op 1785) 
s_cmp_eq_i32 s9, 0 // (op 1785) 
s_mov_b32 s18, scc // (op 1785) 
s_cmp_eq_i32 s18, 0 // (op 1785) 
s_cselect_b32 s19, 0, -1 // (op 1785) 

s_and_b32 s18, s45, 31 // (op 1785) 
s_lshl_b32 s22, 1, s18 // (op 1785) 
s_add_i32 s23, s22, s19 // (op 1785) 


s_and_b32 s19, s0, s23 // (op 1785) 


s_add_i32 s0, s47, s19 // (op 1785) 


s_ashr_i32 s19, s0, s18 // (op 1785) 


s_lshl_b32 s0, s45, 24 // (op 1785) 
s_ashr_i32 s18, s0, 31 // (op 1785) 

s_xor_b32 s0, s19, s18 // (op 1785) 

s_sub_i32 s19, s0, s18 // (op 1785) 


s_add_u32 s22, s4, 32 // (op 1785) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1785) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1785) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1785) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1785) 

s_mul_i32 s0, s11, s22 // (op 1785) 

s_mul_i32 s18, s8, s0 // (op 1785) 

s_mul_i32 s0, s19, s18 // (op 1785) 

s_sub_u32 s18, s16, s0 // (op 1785) 

s_mov_b32 s0, s18 // (op 1785) 
s_mul_hi_i32 s47, s0, s12 // (op 1785) 
s_add_i32 s22, s47, s0 // (op 1785) 


s_ashr_i32 s0, s22, 31 // (op 1785) 
s_cmp_eq_i32 s12, 0 // (op 1785) 
s_mov_b32 s47, scc // (op 1785) 
s_cmp_eq_i32 s47, 0 // (op 1785) 
s_cselect_b32 s23, 0, -1 // (op 1785) 

s_and_b32 s47, s44, 31 // (op 1785) 
s_lshl_b32 s24, 1, s47 // (op 1785) 
s_add_i32 s25, s24, s23 // (op 1785) 


s_and_b32 s23, s0, s25 // (op 1785) 


s_add_i32 s0, s22, s23 // (op 1785) 


s_ashr_i32 s22, s0, s47 // (op 1785) 


s_lshl_b32 s0, s44, 24 // (op 1785) 
s_ashr_i32 s47, s0, 31 // (op 1785) 

s_xor_b32 s0, s22, s47 // (op 1785) 

s_sub_i32 s22, s0, s47 // (op 1785) 


s_mul_i32 s0, s46, s11 // (op 1785) 
s_add_u32 s24, s6, 128 // (op 1785) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1785) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 1785) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 1785) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 7 // (op 1785) 

s_sub_i32 s23, s24, s0 // (op 1785) 

s_mul_i32 s47, s22, s23 // (op 1785) 


s_sub_u32 s22, s18, s47 // (op 1785) 


s_add_u32 s18, s22, s0 // (op 1785) 


s_cmp_eq_i32 s19, 0 // (op 1785) 
s_mov_b32 s0, scc // (op 1785) 

s_cmp_eq_i32 s0, 0 // (op 1785) 
s_cselect_b32 s19, s18, s1 // (op 1785) 



v_lshl_add_u32 v14, s19, 7, v12 // (op 1785) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 1785) 
s_lshr_b32 s47, s0, 1 // (op 1785) 

v_lshl_add_u32 v12, s47, 8, v13 // (op 1785) 


v_mov_b32 v31, 0 // (op 1785) convert
v_mov_b32 v30, v12 // (op 1785) convert

v_mul_lo_u32 v13, v14, s27 // (op 1785) most significant: low of low * high
v_mul_hi_u32 v15, v14, s26 // (op 1785) most significant: high of low * low
v_mul_lo_u32 v12, v14, s26 // (op 1785) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 1785) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v30 // (op 1785) least significant half
v_addc_co_u32 v13, s[0:1], v13, v31, s[0:1] // (op 1785) most significant half



s_mul_i32 s1, s26, s7 // (op 1787) most significant: low of low * high
s_mul_i32 s47, s27, s6 // (op 1787) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 1787) least significant
s_mul_hi_u32 s18, s26, s6 // (op 1787) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1787) most significant: sum
s_add_u32 s1, s1, s18 // (op 1787) most significant: sum


s_mov_b32 s48, s28 // (op 1787) reg expression
s_mov_b32 s49, s29 // (op 1787) reg expression
s_mov_b32 s50, s0 // (op 1787) reg expression
s_mov_b32 s51, 131072 // (op 1787) call()

s_mul_hi_i32 s47, s16, s10 // (op 1855) 
v_lshrrev_b32 v14, 3, v1 // (op 1855) 
s_add_i32 s0, s47, s16 // (op 1855) 

v_and_b32 v15, -8, v1 // (op 1855) 
s_ashr_i32 s1, s0, 31 // (op 1855) 
v_sub_u32 v16, v1, v15 // (op 1855) 

s_cmp_eq_i32 s10, 0 // (op 1855) 
s_mov_b32 s47, scc // (op 1855) 
v_lshlrev_b32 v15, 4, v16 // (op 1855) 

s_cmp_eq_i32 s47, 0 // (op 1855) 
s_cselect_b32 s18, 0, -1 // (op 1855) 

s_and_b32 s19, s13, 31 // (op 1855) 
s_lshl_b32 s47, 1, s19 // (op 1855) 
s_add_i32 s22, s47, s18 // (op 1855) 


s_and_b32 s18, s1, s22 // (op 1855) 


s_add_i32 s1, s0, s18 // (op 1855) 


s_ashr_i32 s0, s1, s19 // (op 1855) 


s_lshl_b32 s1, s13, 24 // (op 1855) 
s_ashr_i32 s47, s1, 31 // (op 1855) 

s_xor_b32 s1, s0, s47 // (op 1855) 

s_sub_i32 s0, s1, s47 // (op 1855) 


s_mul_hi_i32 s1, s0, s14 // (op 1855) 
s_add_i32 s47, s1, s0 // (op 1855) 

s_ashr_i32 s1, s47, 31 // (op 1855) 
s_cmp_eq_i32 s14, 0 // (op 1855) 
s_mov_b32 s18, scc // (op 1855) 
s_cmp_eq_i32 s18, 0 // (op 1855) 
s_cselect_b32 s19, 0, -1 // (op 1855) 

s_and_b32 s18, s15, 31 // (op 1855) 
s_lshl_b32 s22, 1, s18 // (op 1855) 
s_add_i32 s23, s22, s19 // (op 1855) 


s_and_b32 s19, s1, s23 // (op 1855) 


s_add_i32 s1, s47, s19 // (op 1855) 


s_ashr_i32 s19, s1, s18 // (op 1855) 


s_lshl_b32 s1, s15, 24 // (op 1855) 
s_ashr_i32 s18, s1, 31 // (op 1855) 

s_xor_b32 s1, s19, s18 // (op 1855) 

s_sub_i32 s19, s1, s18 // (op 1855) 


s_add_u32 s22, s4, 32 // (op 1855) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1855) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1855) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1855) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1855) 

s_mul_i32 s1, s19, s22 // (op 1855) 

s_sub_i32 s47, s0, s1 // (op 1855) 


s_mul_hi_i32 s0, s16, s9 // (op 1855) 
s_add_i32 s1, s0, s16 // (op 1855) 

s_ashr_i32 s0, s1, 31 // (op 1855) 
s_cmp_eq_i32 s9, 0 // (op 1855) 
s_mov_b32 s18, scc // (op 1855) 
s_cmp_eq_i32 s18, 0 // (op 1855) 
s_cselect_b32 s19, 0, -1 // (op 1855) 

s_and_b32 s18, s45, 31 // (op 1855) 
s_lshl_b32 s24, 1, s18 // (op 1855) 
s_add_i32 s25, s24, s19 // (op 1855) 


s_and_b32 s19, s0, s25 // (op 1855) 


s_add_i32 s0, s1, s19 // (op 1855) 


s_ashr_i32 s1, s0, s18 // (op 1855) 


s_lshl_b32 s0, s45, 24 // (op 1855) 
s_ashr_i32 s18, s0, 31 // (op 1855) 

s_xor_b32 s0, s1, s18 // (op 1855) 

s_sub_i32 s1, s0, s18 // (op 1855) 


s_mul_i32 s0, s11, s22 // (op 1855) 

s_mul_i32 s18, s8, s0 // (op 1855) 

s_mul_i32 s0, s1, s18 // (op 1855) 

s_sub_u32 s18, s16, s0 // (op 1855) 

s_mul_hi_i32 s0, s18, s12 // (op 1855) 
s_add_i32 s19, s0, s18 // (op 1855) 


s_ashr_i32 s0, s19, 31 // (op 1855) 
s_cmp_eq_i32 s12, 0 // (op 1855) 
s_mov_b32 s18, scc // (op 1855) 
s_cmp_eq_i32 s18, 0 // (op 1855) 
s_cselect_b32 s22, 0, -1 // (op 1855) 

s_and_b32 s18, s44, 31 // (op 1855) 
s_lshl_b32 s23, 1, s18 // (op 1855) 
s_add_i32 s24, s23, s22 // (op 1855) 


s_and_b32 s25, s0, s24 // (op 1855) 


s_add_i32 s0, s19, s25 // (op 1855) 


s_ashr_i32 s19, s0, s18 // (op 1855) 


s_lshl_b32 s0, s44, 24 // (op 1855) 
s_ashr_i32 s18, s0, 31 // (op 1855) 

s_xor_b32 s0, s19, s18 // (op 1855) 

s_sub_i32 s19, s0, s18 // (op 1855) 


s_cmp_eq_i32 s1, 0 // (op 1855) 
s_mov_b32 s0, scc // (op 1855) 

s_cmp_eq_i32 s0, 0 // (op 1855) 
s_cselect_b32 s1, s19, s47 // (op 1855) 



v_lshl_add_u32 v16, s1, 5, v14 // (op 1855) 


v_mul_lo_u32 v31, v16, s31 // (op 1855) most significant: low of low * high
v_mul_hi_u32 v14, v16, s30 // (op 1855) most significant: high of low * low
v_mul_lo_u32 v30, v16, s30 // (op 1855) least significant: low of low * low
v_add_u32 v31, v31, v14 // (op 1855) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 1855) 
s_lshr_b32 s47, s0, 1 // (op 1855) 

v_lshl_add_u32 v14, s47, 8, v15 // (op 1855) 


v_mov_b32 v25, 0 // (op 1855) convert
v_mov_b32 v24, v14 // (op 1855) convert

v_add_co_u32 v26, s[0:1], v24, v30 // (op 1855) least significant half
v_addc_co_u32 v27, s[0:1], v25, v31, s[0:1] // (op 1855) most significant half



v_mul_lo_u32 v25, v27, 6 // (op 1855) most significant: low of high * low
v_mul_hi_u32 v14, v26, 6 // (op 1855) most significant: high of low * low
v_mul_lo_u32 v24, v26, 6 // (op 1855) least significant: low of low * low
v_add_u32 v25, v25, v14 // (op 1855) most significant: sum


v_lshrrev_b64 v[26:27], 3, v[24:25] // (op 1855) 

v_mov_b32 v14, v26 // (op 1855) convert

s_mul_i32 s1, s5, 6 // (op 1857) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 1857) least significant
s_mul_hi_u32 s47, s4, 6 // (op 1857) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1857) most significant: sum

s_mul_i32 s19, s30, s1 // (op 1857) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 1857) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 1857) least significant
s_mul_hi_u32 s22, s30, s0 // (op 1857) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 1857) most significant: sum
s_add_u32 s19, s19, s22 // (op 1857) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1857) 

s_mov_b32 s60, s32 // (op 1857) reg expression
s_mov_b32 s61, s33 // (op 1857) reg expression
s_mov_b32 s62, s0 // (op 1857) reg expression
s_mov_b32 s63, 131072 // (op 1857) call()

s_mul_hi_i32 s47, s16, s10 // (op 1919) 
v_lshrrev_b32 v15, 2, v1 // (op 1919) 
s_add_i32 s0, s47, s16 // (op 1919) 

v_lshlrev_b32 v16, 1, v15 // (op 1919) 

s_ashr_i32 s1, s0, 31 // (op 1919) 
v_and_b32 v15, -4, v1 // (op 1919) 
s_cmp_eq_i32 s10, 0 // (op 1919) 
s_mov_b32 s47, scc // (op 1919) 
v_sub_u32 v17, v1, v15 // (op 1919) 

s_cmp_eq_i32 s47, 0 // (op 1919) 
s_cselect_b32 s18, 0, -1 // (op 1919) 

s_and_b32 s19, s13, 31 // (op 1919) 
s_lshl_b32 s47, 1, s19 // (op 1919) 
s_add_i32 s22, s47, s18 // (op 1919) 


s_and_b32 s18, s1, s22 // (op 1919) 


s_add_i32 s1, s0, s18 // (op 1919) 


s_ashr_i32 s0, s1, s19 // (op 1919) 


s_lshl_b32 s1, s13, 24 // (op 1919) 
s_ashr_i32 s47, s1, 31 // (op 1919) 

s_xor_b32 s1, s0, s47 // (op 1919) 

s_sub_i32 s0, s1, s47 // (op 1919) 


s_mul_hi_i32 s1, s0, s14 // (op 1919) 
s_add_i32 s47, s1, s0 // (op 1919) 

s_ashr_i32 s1, s47, 31 // (op 1919) 
s_cmp_eq_i32 s14, 0 // (op 1919) 
s_mov_b32 s18, scc // (op 1919) 
s_cmp_eq_i32 s18, 0 // (op 1919) 
s_cselect_b32 s19, 0, -1 // (op 1919) 

s_and_b32 s18, s15, 31 // (op 1919) 
s_lshl_b32 s22, 1, s18 // (op 1919) 
s_add_i32 s23, s22, s19 // (op 1919) 


s_and_b32 s19, s1, s23 // (op 1919) 


s_add_i32 s1, s47, s19 // (op 1919) 


s_ashr_i32 s19, s1, s18 // (op 1919) 


s_lshl_b32 s1, s15, 24 // (op 1919) 
s_ashr_i32 s18, s1, 31 // (op 1919) 

s_xor_b32 s1, s19, s18 // (op 1919) 

s_sub_i32 s19, s1, s18 // (op 1919) 


s_mul_i32 s1, s0, s11 // (op 1919) 

s_sub_u32 s0, s16, s1 // (op 1919) 

s_mul_i32 s1, s19, s11 // (op 1919) 
s_add_u32 s1, s1, s0 // (op 1919) 


s_mul_hi_i32 s0, s16, s9 // (op 1919) 
s_add_i32 s47, s0, s16 // (op 1919) 

s_ashr_i32 s0, s47, 31 // (op 1919) 
s_cmp_eq_i32 s9, 0 // (op 1919) 
s_mov_b32 s18, scc // (op 1919) 
s_cmp_eq_i32 s18, 0 // (op 1919) 
s_cselect_b32 s19, 0, -1 // (op 1919) 

s_and_b32 s18, s45, 31 // (op 1919) 
s_lshl_b32 s22, 1, s18 // (op 1919) 
s_add_i32 s23, s22, s19 // (op 1919) 


s_and_b32 s19, s0, s23 // (op 1919) 


s_add_i32 s0, s47, s19 // (op 1919) 


s_ashr_i32 s19, s0, s18 // (op 1919) 


s_lshl_b32 s0, s45, 24 // (op 1919) 
s_ashr_i32 s18, s0, 31 // (op 1919) 

s_xor_b32 s0, s19, s18 // (op 1919) 

s_sub_i32 s19, s0, s18 // (op 1919) 


s_add_u32 s22, s4, 32 // (op 1919) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1919) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1919) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1919) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1919) 

s_mul_i32 s0, s11, s22 // (op 1919) 

s_mul_i32 s18, s8, s0 // (op 1919) 

s_mul_i32 s0, s19, s18 // (op 1919) 

s_sub_u32 s18, s16, s0 // (op 1919) 

s_mov_b32 s0, s18 // (op 1919) 
s_mul_hi_i32 s47, s0, s12 // (op 1919) 
s_add_i32 s22, s47, s0 // (op 1919) 


s_ashr_i32 s0, s22, 31 // (op 1919) 
s_cmp_eq_i32 s12, 0 // (op 1919) 
s_mov_b32 s47, scc // (op 1919) 
s_cmp_eq_i32 s47, 0 // (op 1919) 
s_cselect_b32 s23, 0, -1 // (op 1919) 

s_and_b32 s47, s44, 31 // (op 1919) 
s_lshl_b32 s24, 1, s47 // (op 1919) 
s_add_i32 s25, s24, s23 // (op 1919) 


s_and_b32 s23, s0, s25 // (op 1919) 


s_add_i32 s0, s22, s23 // (op 1919) 


s_ashr_i32 s22, s0, s47 // (op 1919) 


s_lshl_b32 s0, s44, 24 // (op 1919) 
s_ashr_i32 s47, s0, 31 // (op 1919) 

s_xor_b32 s0, s22, s47 // (op 1919) 

s_sub_i32 s22, s0, s47 // (op 1919) 


s_mul_i32 s0, s46, s11 // (op 1919) 
s_add_u32 s24, s6, 128 // (op 1919) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1919) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 1919) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 1919) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 7 // (op 1919) 

s_sub_i32 s23, s24, s0 // (op 1919) 

s_mul_i32 s47, s22, s23 // (op 1919) 


s_sub_u32 s22, s18, s47 // (op 1919) 


s_add_u32 s18, s22, s0 // (op 1919) 


s_cmp_eq_i32 s19, 0 // (op 1919) 
s_mov_b32 s0, scc // (op 1919) 

s_cmp_eq_i32 s0, 0 // (op 1919) 
s_cselect_b32 s19, s18, s1 // (op 1919) 



v_lshl_add_u32 v15, s19, 7, v16 // (op 1919) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 1919) 
s_lshr_b32 s47, s0, 1 // (op 1919) 

v_lshl_add_u32 v16, s47, 3, v17 // (op 1919) 


v_mul_lo_u32 v17, v15, s34 // (op 1919) 
v_add_u32 v17, v17, v16 // (op 1919) 


s_mul_i32 s1, s34, s7 // (op 1921) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 1921) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 1921) least significant
s_mul_hi_u32 s18, s34, s6 // (op 1921) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1921) most significant: sum
s_add_u32 s1, s1, s18 // (op 1921) most significant: sum


s_mov_b32 s64, s36 // (op 1921) reg expression
s_mov_b32 s65, s37 // (op 1921) reg expression
s_mov_b32 s66, s0 // (op 1921) reg expression
s_mov_b32 s67, 131072 // (op 1921) call()

s_mul_hi_i32 s47, s16, s10 // (op 1983) 
v_lshrrev_b32 v15, 6, v1 // (op 1983) 
s_add_i32 s0, s47, s16 // (op 1983) 

v_and_b32 v16, 1, v15 // (op 1983) 

s_ashr_i32 s1, s0, 31 // (op 1983) 
v_and_b32 v15, 15, v1 // (op 1983) 
s_cmp_eq_i32 s10, 0 // (op 1983) 
s_mov_b32 s47, scc // (op 1983) 
v_lshl_add_u32 v18, v16, 4, v15 // (op 1983) 


s_cmp_eq_i32 s47, 0 // (op 1983) 
s_cselect_b32 s18, 0, -1 // (op 1983) 

v_and_b32 v15, 63, v1 // (op 1983) 
s_and_b32 s19, s13, 31 // (op 1983) 
v_lshrrev_b32 v16, 4, v15 // (op 1983) 

s_lshl_b32 s47, 1, s19 // (op 1983) 
s_add_i32 s22, s47, s18 // (op 1983) 


s_and_b32 s18, s1, s22 // (op 1983) 


s_add_i32 s1, s0, s18 // (op 1983) 


s_ashr_i32 s0, s1, s19 // (op 1983) 


s_lshl_b32 s1, s13, 24 // (op 1983) 
s_ashr_i32 s47, s1, 31 // (op 1983) 

s_xor_b32 s1, s0, s47 // (op 1983) 

s_sub_i32 s0, s1, s47 // (op 1983) 


s_mul_hi_i32 s1, s0, s14 // (op 1983) 
s_add_i32 s47, s1, s0 // (op 1983) 

s_ashr_i32 s1, s47, 31 // (op 1983) 
s_cmp_eq_i32 s14, 0 // (op 1983) 
s_mov_b32 s18, scc // (op 1983) 
s_cmp_eq_i32 s18, 0 // (op 1983) 
s_cselect_b32 s19, 0, -1 // (op 1983) 

s_and_b32 s18, s15, 31 // (op 1983) 
s_lshl_b32 s22, 1, s18 // (op 1983) 
s_add_i32 s23, s22, s19 // (op 1983) 


s_and_b32 s19, s1, s23 // (op 1983) 


s_add_i32 s1, s47, s19 // (op 1983) 


s_ashr_i32 s19, s1, s18 // (op 1983) 


s_lshl_b32 s1, s15, 24 // (op 1983) 
s_ashr_i32 s18, s1, 31 // (op 1983) 

s_xor_b32 s1, s19, s18 // (op 1983) 

s_sub_i32 s19, s1, s18 // (op 1983) 


s_add_u32 s22, s4, 32 // (op 1983) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1983) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1983) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1983) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1983) 

s_mul_i32 s1, s19, s22 // (op 1983) 

s_sub_i32 s47, s0, s1 // (op 1983) 


s_mul_hi_i32 s0, s16, s9 // (op 1983) 
s_add_i32 s1, s0, s16 // (op 1983) 

s_ashr_i32 s0, s1, 31 // (op 1983) 
s_cmp_eq_i32 s9, 0 // (op 1983) 
s_mov_b32 s18, scc // (op 1983) 
s_cmp_eq_i32 s18, 0 // (op 1983) 
s_cselect_b32 s19, 0, -1 // (op 1983) 

s_and_b32 s18, s45, 31 // (op 1983) 
s_lshl_b32 s24, 1, s18 // (op 1983) 
s_add_i32 s25, s24, s19 // (op 1983) 


s_and_b32 s19, s0, s25 // (op 1983) 


s_add_i32 s0, s1, s19 // (op 1983) 


s_ashr_i32 s1, s0, s18 // (op 1983) 


s_lshl_b32 s0, s45, 24 // (op 1983) 
s_ashr_i32 s18, s0, 31 // (op 1983) 

s_xor_b32 s0, s1, s18 // (op 1983) 

s_sub_i32 s1, s0, s18 // (op 1983) 


s_mul_i32 s0, s11, s22 // (op 1983) 

s_mul_i32 s18, s8, s0 // (op 1983) 

s_mul_i32 s0, s1, s18 // (op 1983) 

s_sub_u32 s18, s16, s0 // (op 1983) 

s_mul_hi_i32 s0, s18, s12 // (op 1983) 
s_add_i32 s19, s0, s18 // (op 1983) 


s_ashr_i32 s0, s19, 31 // (op 1983) 
s_cmp_eq_i32 s12, 0 // (op 1983) 
s_mov_b32 s18, scc // (op 1983) 
s_cmp_eq_i32 s18, 0 // (op 1983) 
s_cselect_b32 s22, 0, -1 // (op 1983) 

s_and_b32 s18, s44, 31 // (op 1983) 
s_lshl_b32 s23, 1, s18 // (op 1983) 
s_add_i32 s24, s23, s22 // (op 1983) 


s_and_b32 s25, s0, s24 // (op 1983) 


s_add_i32 s0, s19, s25 // (op 1983) 


s_ashr_i32 s19, s0, s18 // (op 1983) 


s_lshl_b32 s0, s44, 24 // (op 1983) 
s_ashr_i32 s18, s0, 31 // (op 1983) 

s_xor_b32 s0, s19, s18 // (op 1983) 

s_sub_i32 s19, s0, s18 // (op 1983) 


s_cmp_eq_i32 s1, 0 // (op 1983) 
s_mov_b32 s0, scc // (op 1983) 

s_cmp_eq_i32 s0, 0 // (op 1983) 
s_cselect_b32 s1, s19, s47 // (op 1983) 



v_lshl_add_u32 v15, s1, 5, v18 // (op 1983) 


v_mul_lo_u32 v18, v15, s38 // (op 1983) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 1983) 
s_lshr_b32 s47, s0, 1 // (op 1983) 

v_lshl_add_u32 v15, s47, 3, v16 // (op 1983) 


v_add_u32 v16, v15, v18 // (op 1983) 


s_mul_i32 s1, s38, s5 // (op 1985) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 1985) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 1985) least significant
s_mul_hi_u32 s18, s38, s4 // (op 1985) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1985) most significant: sum
s_add_u32 s1, s1, s18 // (op 1985) most significant: sum


s_mov_b32 s68, s40 // (op 1985) reg expression
s_mov_b32 s69, s41 // (op 1985) reg expression
s_mov_b32 s70, s0 // (op 1985) reg expression
s_mov_b32 s71, 131072 // (op 1985) call()

v_and_b32 v15, -8, v1 // (op 2076) 
v_sub_u32 v18, v1, v15 // (op 2076) 

v_lshlrev_b32 v15, 4, v18 // (op 2076) 

v_lshrrev_b32 v18, 3, v1 // (op 2076) 
v_lshl_add_u32 v19, v18, 7, v15 // (op 2076) 


s_nop 0
v_readfirstlane_b32 s47, v19 // (op 2076) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -8, v1 // (op 2133) 
v_sub_u32 v18, v1, v15 // (op 2133) 

v_lshlrev_b32 v15, 4, v18 // (op 2133) 

v_lshrrev_b32 v18, 3, v1 // (op 2133) 
v_lshl_add_u32 v19, v18, 7, v15 // (op 2133) 


v_mul_lo_u32 v15, v19, 6 // (op 2133) 

v_lshrrev_b32 v18, 3, v15 // (op 2133) 

v_and_b32 v15, -4, v1 // (op 2186) 
v_sub_u32 v19, v1, v15 // (op 2186) 

v_lshrrev_b32 v15, 2, v1 // (op 2186) 
v_lshl_add_u32 v20, v15, 3, v19 // (op 2186) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 644) 
s_and_b32 s18, s0, 4294967294 // (op 644) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 643) 
s_mov_b32 s19, s0 // (op 643) convert

s_cmp_lt_u32 s18, s19 // (op 643) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_643 // (op 643) Condition: Top (jump to RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_643 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoopTail_643:
s_barrier  // (op 2387) 
s_add_u32 s18, s18, 1 // (op 645) 

s_lshl_b64 s[0:1], s[26:27], 5 // (op 1756) 
v_mov_b64 v[22:23], s[0:1] // (op 1756) convert


v_mov_b64 v[24:25], v[12:13] // (op 1756) 
v_mov_b64 v[26:27], v[24:25] // (op 1756) 
s_mov_b32 m0, s47 // (op 1756) reg expression
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 1756) Load value direct to lds
v_add_co_u32 v26, s[0:1], v26, v22 // (op 1756) least significant half
v_addc_co_u32 v27, s[0:1], 0, v27, s[0:1] // (op 1756) most significant half

s_add_u32 m0, m0, 4096 // (op 1756) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 1756) Load value direct to lds
v_add_co_u32 v26, s[0:1], v26, v22 // (op 1756) least significant half
v_addc_co_u32 v27, s[0:1], 0, v27, s[0:1] // (op 1756) most significant half

s_add_u32 m0, m0, 4096 // (op 1756) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 1756) Load value direct to lds
v_add_co_u32 v26, s[0:1], v26, v22 // (op 1756) least significant half
v_addc_co_u32 v27, s[0:1], 0, v27, s[0:1] // (op 1756) most significant half

s_add_u32 m0, m0, 4096 // (op 1756) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 1756) Load value direct to lds



v_add_co_u32 v12, vcc, 128, v12 // (op 1775) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 1775) most significant half


v_mov_b32 v15, v14 // (op 183) 

buffer_load_dwordx3 v[22:24], v15, s[60:63], 0 offen // (op 183) Load value

v_add_u32 v14, 96, v14 // (op 1847) 

s_barrier  // (op 2400) 
v_mov_b64 v[26:27], s[34:35] // (op 184) convert




v_mov_b32 v15, v17 // (op 184) 
v_mov_b32 v19, v15 // (op 184) 
buffer_load_ubyte v28, v19, s[64:67], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v19, v19, v26 // (op 184) 
s_waitcnt vmcnt(1)
ds_write_b96 v18, v[22:24] offset:32768 // (op 181) Store local data 
buffer_load_ubyte v29, v19, s[64:67], 0 offen offset: 0 // (op 184) Load value




v_add_u32 v17, 4, v17 // (op 1911) 
s_barrier  // (op 2390) 
s_barrier  // (op 2414) 






ds_read_b128 v[30:33], v7 // (op 173) Load local data 
ds_read_b128 v[34:37], v7 offset:64 // (op 173) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v20, v28 offset:35840 // (op 182) Store local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 2404) 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v29 offset:35844 // (op 182) Store local data 




ds_read_b128 v[24:27], v3 offset:32768 // (op 175) Load local data 
ds_read_b64 v[28:29], v3 offset:32784 // (op 175) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 2418) 



ds_read_u8 v15, v0 offset:35840 // (op 177) Load local data 


v_mov_b32 v19, v16 // (op 179) 

buffer_load_ubyte v21, v19, s[68:71], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v16, 4, v16 // (op 1975) 



ds_read_b128 v[38:41], v7 offset:4096 // (op 325) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[24:29], a[0:3], v15, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 180) 
ds_read_b128 v[42:45], v7 offset:4160 // (op 325) Load local data 





ds_read_u8 v15, v0 offset:35968 // (op 329) Load local data 



ds_read_b128 v[30:33], v7 offset:8192 // (op 487) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:45], v[24:29], a[4:7], v15, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 331) 
ds_read_b128 v[34:37], v7 offset:8256 // (op 487) Load local data 





ds_read_u8 v15, v0 offset:36096 // (op 491) Load local data 



ds_read_b128 v[38:41], v7 offset:12288 // (op 649) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[30:37], v[24:29], a[8:11], v15, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 493) 
ds_read_b128 v[42:45], v7 offset:12352 // (op 649) Load local data 





ds_read_u8 v15, v0 offset:36224 // (op 653) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[38:45], v[24:29], a[12:15], v15, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 655) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 643) 
s_mov_b32 s19, s0 // (op 643) convert

s_cmp_lt_u32 s18, s19 // (op 643) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoopTail_643 // (op 643) Condition: Bottom (jump to RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopTop_KLoopTail_643 if true)
RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_643:





















s_barrier  // (op 2506) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalFalse_0_beta_125
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v8, v10 // (op 46) 
v_mul_f32 v15, v8, v11 // (op 46) 
v_mul_f32 v16, v8, v12 // (op 46) 
v_mul_f32 v17, v8, v13 // (op 46) 

v_cvt_f16_f32 v10, v14 // (op 52) 
v_cvt_f16_f32 v11, v15 // (op 52) 
v_cvt_f16_f32 v12, v16 // (op 52) 
v_cvt_f16_f32 v13, v17 // (op 52) 


v_pack_b32_f16 v14, v10, v11 // (op 54) 
v_pack_b32_f16 v15, v12, v13 // (op 54) 


buffer_store_dwordx2 v[14:15], v9, s[56:59], 0 offen // (op 54) Store value


v_accvgpr_read v10, a4 // (op 885) 
v_accvgpr_read v11, a5 // (op 885) 
v_accvgpr_read v12, a6 // (op 885) 
v_accvgpr_read v13, a7 // (op 885) 
v_mul_f32 v14, v8, v10 // (op 885) 
v_mul_f32 v15, v8, v11 // (op 885) 
v_mul_f32 v16, v8, v12 // (op 885) 
v_mul_f32 v17, v8, v13 // (op 885) 

v_cvt_f16_f32 v10, v14 // (op 887) 
v_cvt_f16_f32 v11, v15 // (op 887) 
v_cvt_f16_f32 v12, v16 // (op 887) 
v_cvt_f16_f32 v13, v17 // (op 887) 


v_pack_b32_f16 v14, v10, v11 // (op 888) 
v_pack_b32_f16 v15, v12, v13 // (op 888) 


buffer_store_dwordx2 v[14:15], v9, s[56:59], 0 offen offset:64 // (op 888) Store value


v_accvgpr_read v10, a8 // (op 905) 
v_accvgpr_read v11, a9 // (op 905) 
v_accvgpr_read v12, a10 // (op 905) 
v_accvgpr_read v13, a11 // (op 905) 
v_mul_f32 v14, v8, v10 // (op 905) 
v_mul_f32 v15, v8, v11 // (op 905) 
v_mul_f32 v16, v8, v12 // (op 905) 
v_mul_f32 v17, v8, v13 // (op 905) 

v_cvt_f16_f32 v10, v14 // (op 907) 
v_cvt_f16_f32 v11, v15 // (op 907) 
v_cvt_f16_f32 v12, v16 // (op 907) 
v_cvt_f16_f32 v13, v17 // (op 907) 


v_pack_b32_f16 v14, v10, v11 // (op 908) 
v_pack_b32_f16 v15, v12, v13 // (op 908) 


buffer_store_dwordx2 v[14:15], v9, s[56:59], 0 offen offset:128 // (op 908) Store value


v_accvgpr_read v10, a12 // (op 925) 
v_accvgpr_read v11, a13 // (op 925) 
v_accvgpr_read v12, a14 // (op 925) 
v_accvgpr_read v13, a15 // (op 925) 
v_mul_f32 v14, v8, v10 // (op 925) 
v_mul_f32 v15, v8, v11 // (op 925) 
v_mul_f32 v16, v8, v12 // (op 925) 
v_mul_f32 v17, v8, v13 // (op 925) 

v_cvt_f16_f32 v10, v14 // (op 927) 
v_cvt_f16_f32 v11, v15 // (op 927) 
v_cvt_f16_f32 v12, v16 // (op 927) 
v_cvt_f16_f32 v13, v17 // (op 927) 


v_pack_b32_f16 v14, v10, v11 // (op 928) 
v_pack_b32_f16 v15, v12, v13 // (op 928) 


buffer_store_dwordx2 v[14:15], v9, s[56:59], 0 offen offset:192 // (op 928) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[10:11], v6, s[52:55], 0 offen offset:192 // (op 1008) Load value
v_accvgpr_read v12, a12 // (op 1004) 
v_accvgpr_read v13, a13 // (op 1004) 
v_accvgpr_read v14, a14 // (op 1004) 
v_accvgpr_read v15, a15 // (op 1004) 
v_mul_f32 v16, v8, v12 // (op 1004) 
v_mul_f32 v17, v8, v13 // (op 1004) 
v_mul_f32 v18, v8, v14 // (op 1004) 
v_mul_f32 v19, v8, v15 // (op 1004) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 1009) 
v_lshrrev_b32 v13, 16, v10 // (op 1009) 
v_cvt_f32_f16 v13, v13 // (op 1009) 
v_mul_f32 v20, v5, v12 // (op 1009) 
v_mul_f32 v21, v5, v13 // (op 1009) 
v_cvt_f32_f16 v12, v11 // (op 1009) 
v_lshrrev_b32 v13, 16, v11 // (op 1009) 
v_cvt_f32_f16 v13, v13 // (op 1009) 
v_mul_f32 v22, v5, v12 // (op 1009) 
v_mul_f32 v23, v5, v13 // (op 1009) 


v_add_f32 v10, v20, v16 // (op 1005) 
v_add_f32 v11, v21, v17 // (op 1005) 
v_add_f32 v12, v22, v18 // (op 1005) 
v_add_f32 v13, v23, v19 // (op 1005) 


v_cvt_f16_f32 v14, v10 // (op 1006) 
v_cvt_f16_f32 v15, v11 // (op 1006) 
v_cvt_f16_f32 v16, v12 // (op 1006) 
v_cvt_f16_f32 v17, v13 // (op 1006) 


v_pack_b32_f16 v10, v14, v15 // (op 1007) 
v_pack_b32_f16 v11, v16, v17 // (op 1007) 


buffer_store_dwordx2 v[10:11], v9, s[56:59], 0 offen offset:192 // (op 1007) Store value





buffer_load_dwordx2 v[10:11], v6, s[52:55], 0 offen offset:128 // (op 988) Load value
v_accvgpr_read v12, a8 // (op 984) 
v_accvgpr_read v13, a9 // (op 984) 
v_accvgpr_read v14, a10 // (op 984) 
v_accvgpr_read v15, a11 // (op 984) 
v_mul_f32 v16, v8, v12 // (op 984) 
v_mul_f32 v17, v8, v13 // (op 984) 
v_mul_f32 v18, v8, v14 // (op 984) 
v_mul_f32 v19, v8, v15 // (op 984) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 989) 
v_lshrrev_b32 v13, 16, v10 // (op 989) 
v_cvt_f32_f16 v13, v13 // (op 989) 
v_mul_f32 v20, v5, v12 // (op 989) 
v_mul_f32 v21, v5, v13 // (op 989) 
v_cvt_f32_f16 v12, v11 // (op 989) 
v_lshrrev_b32 v13, 16, v11 // (op 989) 
v_cvt_f32_f16 v13, v13 // (op 989) 
v_mul_f32 v22, v5, v12 // (op 989) 
v_mul_f32 v23, v5, v13 // (op 989) 


v_add_f32 v10, v20, v16 // (op 985) 
v_add_f32 v11, v21, v17 // (op 985) 
v_add_f32 v12, v22, v18 // (op 985) 
v_add_f32 v13, v23, v19 // (op 985) 


v_cvt_f16_f32 v14, v10 // (op 986) 
v_cvt_f16_f32 v15, v11 // (op 986) 
v_cvt_f16_f32 v16, v12 // (op 986) 
v_cvt_f16_f32 v17, v13 // (op 986) 


v_pack_b32_f16 v10, v14, v15 // (op 987) 
v_pack_b32_f16 v11, v16, v17 // (op 987) 


buffer_store_dwordx2 v[10:11], v9, s[56:59], 0 offen offset:128 // (op 987) Store value





buffer_load_dwordx2 v[10:11], v6, s[52:55], 0 offen offset:64 // (op 968) Load value
v_accvgpr_read v12, a4 // (op 964) 
v_accvgpr_read v13, a5 // (op 964) 
v_accvgpr_read v14, a6 // (op 964) 
v_accvgpr_read v15, a7 // (op 964) 
v_mul_f32 v16, v8, v12 // (op 964) 
v_mul_f32 v17, v8, v13 // (op 964) 
v_mul_f32 v18, v8, v14 // (op 964) 
v_mul_f32 v19, v8, v15 // (op 964) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 969) 
v_lshrrev_b32 v13, 16, v10 // (op 969) 
v_cvt_f32_f16 v13, v13 // (op 969) 
v_mul_f32 v20, v5, v12 // (op 969) 
v_mul_f32 v21, v5, v13 // (op 969) 
v_cvt_f32_f16 v12, v11 // (op 969) 
v_lshrrev_b32 v13, 16, v11 // (op 969) 
v_cvt_f32_f16 v13, v13 // (op 969) 
v_mul_f32 v22, v5, v12 // (op 969) 
v_mul_f32 v23, v5, v13 // (op 969) 


v_add_f32 v10, v20, v16 // (op 965) 
v_add_f32 v11, v21, v17 // (op 965) 
v_add_f32 v12, v22, v18 // (op 965) 
v_add_f32 v13, v23, v19 // (op 965) 


v_cvt_f16_f32 v14, v10 // (op 966) 
v_cvt_f16_f32 v15, v11 // (op 966) 
v_cvt_f16_f32 v16, v12 // (op 966) 
v_cvt_f16_f32 v17, v13 // (op 966) 


v_pack_b32_f16 v10, v14, v15 // (op 967) 
v_pack_b32_f16 v11, v16, v17 // (op 967) 


buffer_store_dwordx2 v[10:11], v9, s[56:59], 0 offen offset:64 // (op 967) Store value





buffer_load_dwordx2 v[10:11], v6, s[52:55], 0 offen // (op 138) Load value
v_accvgpr_read v12, a0 // (op 134) 
v_accvgpr_read v13, a1 // (op 134) 
v_accvgpr_read v14, a2 // (op 134) 
v_accvgpr_read v15, a3 // (op 134) 
v_mul_f32 v16, v8, v12 // (op 134) 
v_mul_f32 v17, v8, v13 // (op 134) 
v_mul_f32 v18, v8, v14 // (op 134) 
v_mul_f32 v19, v8, v15 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 139) 
v_lshrrev_b32 v13, 16, v10 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v20, v5, v12 // (op 139) 
v_mul_f32 v21, v5, v13 // (op 139) 
v_cvt_f32_f16 v12, v11 // (op 139) 
v_lshrrev_b32 v13, 16, v11 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v22, v5, v12 // (op 139) 
v_mul_f32 v23, v5, v13 // (op 139) 


v_add_f32 v10, v20, v16 // (op 135) 
v_add_f32 v11, v21, v17 // (op 135) 
v_add_f32 v12, v22, v18 // (op 135) 
v_add_f32 v13, v23, v19 // (op 135) 


v_cvt_f16_f32 v14, v10 // (op 136) 
v_cvt_f16_f32 v15, v11 // (op 136) 
v_cvt_f16_f32 v16, v12 // (op 136) 
v_cvt_f16_f32 v17, v13 // (op 136) 


v_pack_b32_f16 v10, v14, v15 // (op 137) 
v_pack_b32_f16 v11, v16, v17 // (op 137) 


buffer_store_dwordx2 v[10:11], v9, s[56:59], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__ConditionalBottom_0_beta_125:












s_endpgm  // End of RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM__end-RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 80
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 39936 // lds bytes
  .amdhsa_accum_offset 64
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
    DataTypeB: B6
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
            { type: int64, semantic: SizeFree1 },
            { type: int64, semantic: SizeFree0 },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int64, semantic: StrideD0 },
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
    macrotile: [128, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 39936
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 62
    .agpr_count: 16
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 16
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 24
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 56
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 64
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 72
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 80
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 88
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 152
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 156
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 160
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 164
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 168
        .value_kind: by_value
...

.end_amdgpu_metadata

