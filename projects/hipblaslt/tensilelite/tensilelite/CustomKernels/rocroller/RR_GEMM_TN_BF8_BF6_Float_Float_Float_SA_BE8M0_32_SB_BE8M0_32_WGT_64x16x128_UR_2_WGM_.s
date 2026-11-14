// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_,@function
RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 1159) 
s_mul_hi_i32 s1, s16, s10 // (op 1347) 
v_and_b32 v0, 15, v1 // (op 1273) 
v_and_b32 v3, 63, v1 // (op 1273) 
v_lshrrev_b32 v4, 4, v3 // (op 1273) 

v_lshlrev_b32 v3, 5, v4 // (op 1273) 

v_lshl_add_u32 v4, v0, 7, v3 // (op 1273) 


v_mul_lo_u32 v0, v4, 6 // (op 1273) 

v_lshrrev_b32 v3, 3, v0 // (op 1273) 

v_and_b32 v0, 15, v1 // (op 1417) 
v_and_b32 v4, 63, v1 // (op 1417) 
v_lshrrev_b32 v5, 4, v4 // (op 1417) 

v_lshlrev_b32 v4, 5, v5 // (op 1417) 

v_lshl_add_u32 v5, v0, 7, v4 // (op 1417) 


v_mul_lo_u32 v0, v5, 6 // (op 1417) 

v_lshrrev_b32 v4, 3, v0 // (op 1417) 

v_lshrrev_b32 v0, 6, v1 // (op 1159) 
s_add_i32 s18, s0, s16 // (op 1159) 

v_and_b32 v5, 63, v1 // (op 1159) 
s_ashr_i32 s0, s18, 31 // (op 1159) 
v_lshrrev_b32 v6, 4, v5 // (op 1159) 
s_cmp_eq_i32 s10, 0 // (op 1159) 
s_mov_b32 s19, scc // (op 1159) 
v_lshlrev_b32 v7, 2, v6 // (op 1159) 

s_cmp_eq_i32 s19, 0 // (op 1159) 
s_cselect_b32 s49, 0, -1 // (op 1159) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1159) 


s_and_b32 s19, s13, 31 // (op 1159) 
v_lshrrev_b32 v0, 2, v5 // (op 1159) 

v_and_b32 v5, 3, v0 // (op 1159) 

v_lshrrev_b32 v0, 6, v1 // (op 1220) 
v_and_b32 v7, 15, v1 // (op 1220) 
v_lshl_add_u32 v8, v0, 4, v7 // (op 1220) 


v_and_b32 v0, 48, v1 // (op 1220) 
v_lshl_add_u32 v7, v8, 7, v0 // (op 1220) 


v_lshrrev_b32 v0, 6, v1 // (op 1331) 
v_and_b32 v8, 15, v1 // (op 1331) 
v_lshl_add_u32 v9, v0, 4, v8 // (op 1331) 


v_and_b32 v0, 63, v1 // (op 1331) 
v_lshrrev_b32 v8, 4, v0 // (op 1331) 

v_lshl_add_u32 v0, v9, 2, v8 // (op 1331) 


v_lshrrev_b32 v8, 6, v1 // (op 1347) 
v_and_b32 v9, 63, v1 // (op 1347) 
v_lshrrev_b32 v10, 6, v1 // (op 1393) 
v_and_b32 v11, 15, v1 // (op 1393) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 1393) 


v_and_b32 v10, 48, v1 // (op 1393) 
v_lshl_add_u32 v11, v12, 7, v10 // (op 1393) 


v_lshrrev_b32 v10, 6, v1 // (op 1446) 
v_and_b32 v12, 15, v1 // (op 1446) 
v_lshl_add_u32 v13, v10, 4, v12 // (op 1446) 


v_and_b32 v10, 63, v1 // (op 1446) 
v_lshrrev_b32 v12, 4, v10 // (op 1446) 

v_lshl_add_u32 v10, v13, 2, v12 // (op 1446) 


s_lshl_b32 s50, 1, s19 // (op 1159) 
v_and_b32 v12, 3, v1 // (op 1159) 
s_add_i32 s51, s50, s49 // (op 1159) 


s_and_b32 s49, s0, s51 // (op 1159) 


s_add_i32 s0, s1, s16 // (op 1347) 

s_ashr_i32 s1, s0, 31 // (op 1347) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1159) 


s_add_i32 s50, s18, s49 // (op 1159) 


s_ashr_i32 s18, s50, s19 // (op 1159) 


s_lshl_b32 s19, s13, 24 // (op 1159) 
s_ashr_i32 s49, s19, 31 // (op 1159) 

s_xor_b32 s19, s18, s49 // (op 1159) 

s_sub_i32 s18, s19, s49 // (op 1159) 


s_mul_hi_i32 s19, s18, s14 // (op 1159) 
s_add_i32 s49, s19, s18 // (op 1159) 

s_ashr_i32 s19, s49, 31 // (op 1159) 
s_cmp_eq_i32 s14, 0 // (op 1159) 
s_mov_b32 s50, scc // (op 1159) 
s_cmp_eq_i32 s50, 0 // (op 1159) 
s_cselect_b32 s51, 0, -1 // (op 1159) 

s_and_b32 s50, s15, 31 // (op 1159) 
v_lshrrev_b32 v5, 4, v9 // (op 1347) 
s_cmp_eq_i32 s10, 0 // (op 1347) 
s_mov_b32 s52, scc // (op 1347) 
v_lshlrev_b32 v12, 2, v5 // (op 1347) 

s_cmp_eq_i32 s52, 0 // (op 1347) 
s_cselect_b32 s53, 0, -1 // (op 1347) 

s_lshl_b32 s52, 1, s50 // (op 1159) 
v_lshl_add_u32 v5, v8, 4, v12 // (op 1347) 


v_lshrrev_b32 v8, 2, v9 // (op 1347) 

s_add_i32 s54, s52, s51 // (op 1159) 


s_and_b32 s51, s19, s54 // (op 1159) 


s_add_i32 s19, s49, s51 // (op 1159) 


s_ashr_i32 s49, s19, s50 // (op 1159) 


s_lshl_b32 s19, s15, 24 // (op 1159) 
s_ashr_i32 s50, s19, 31 // (op 1159) 

s_xor_b32 s19, s49, s50 // (op 1159) 

s_sub_i32 s49, s19, s50 // (op 1159) 


s_mul_i32 s19, s18, s11 // (op 1159) 
s_sub_u32 s50, s16, s19 // (op 1159) 

s_mul_i32 s19, s49, s11 // (op 1159) 
s_add_u32 s19, s19, s50 // (op 1159) 

s_mul_hi_i32 s50, s16, s9 // (op 1159) 
s_add_i32 s51, s50, s16 // (op 1159) 

s_ashr_i32 s50, s51, 31 // (op 1159) 
s_cmp_eq_i32 s9, 0 // (op 1159) 
s_mov_b32 s52, scc // (op 1159) 
s_cmp_eq_i32 s52, 0 // (op 1159) 
s_cselect_b32 s54, 0, -1 // (op 1159) 

s_and_b32 s52, s13, 31 // (op 1347) 
v_and_b32 v9, 3, v8 // (op 1347) 

s_lshl_b32 s55, 1, s52 // (op 1347) 
v_and_b32 v8, 3, v1 // (op 1347) 
s_add_i32 s56, s55, s53 // (op 1347) 


s_and_b32 s53, s1, s56 // (op 1347) 


v_lshl_add_u32 v12, v9, 2, v8 // (op 1347) 


s_add_i32 s1, s0, s53 // (op 1347) 


s_ashr_i32 s0, s1, s52 // (op 1347) 


s_lshl_b32 s1, s13, 24 // (op 1347) 
s_ashr_i32 s52, s1, 31 // (op 1347) 

s_xor_b32 s1, s0, s52 // (op 1347) 

s_sub_i32 s0, s1, s52 // (op 1347) 


s_mul_hi_i32 s1, s0, s14 // (op 1347) 
s_add_i32 s52, s1, s0 // (op 1347) 

s_ashr_i32 s1, s52, 31 // (op 1347) 
s_cmp_eq_i32 s14, 0 // (op 1347) 
s_mov_b32 s53, scc // (op 1347) 
s_cmp_eq_i32 s53, 0 // (op 1347) 
s_cselect_b32 s55, 0, -1 // (op 1347) 

s_and_b32 s53, s15, 31 // (op 1347) 
s_lshl_b32 s56, 1, s53 // (op 1347) 
s_add_i32 s57, s56, s55 // (op 1347) 


s_and_b32 s55, s1, s57 // (op 1347) 


s_add_i32 s1, s52, s55 // (op 1347) 


s_ashr_i32 s52, s1, s53 // (op 1347) 


s_lshl_b32 s1, s15, 24 // (op 1347) 
s_ashr_i32 s53, s1, 31 // (op 1347) 

s_xor_b32 s1, s52, s53 // (op 1347) 

s_sub_i32 s52, s1, s53 // (op 1347) 


s_mul_i32 s1, s0, s11 // (op 1347) 
s_sub_u32 s53, s16, s1 // (op 1347) 

s_mul_i32 s1, s52, s11 // (op 1347) 
s_add_u32 s1, s1, s53 // (op 1347) 

s_mul_hi_i32 s53, s16, s9 // (op 1347) 
s_add_i32 s55, s53, s16 // (op 1347) 

s_ashr_i32 s53, s55, 31 // (op 1347) 
s_cmp_eq_i32 s9, 0 // (op 1347) 
s_mov_b32 s56, scc // (op 1347) 
s_cmp_eq_i32 s56, 0 // (op 1347) 
s_cselect_b32 s57, 0, -1 // (op 1347) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 1159) 
s_lshl_b32 s58, 1, s56 // (op 1159) 
s_add_i32 s59, s58, s54 // (op 1159) 


s_and_b32 s54, s50, s59 // (op 1159) 


s_add_i32 s50, s51, s54 // (op 1159) 


s_ashr_i32 s51, s50, s56 // (op 1159) 


s_lshl_b32 s50, s45, 24 // (op 1159) 
s_ashr_i32 s54, s50, 31 // (op 1159) 

s_xor_b32 s50, s51, s54 // (op 1159) 

s_sub_i32 s51, s50, s54 // (op 1159) 


s_and_b32 s50, s45, 31 // (op 1347) 
s_lshl_b32 s54, 1, s50 // (op 1347) 
s_add_i32 s56, s54, s57 // (op 1347) 


s_and_b32 s54, s53, s56 // (op 1347) 


s_add_i32 s53, s55, s54 // (op 1347) 


s_ashr_i32 s54, s53, s50 // (op 1347) 


s_lshl_b32 s50, s45, 24 // (op 1347) 
s_ashr_i32 s53, s50, 31 // (op 1347) 

s_xor_b32 s50, s54, s53 // (op 1347) 

s_sub_i32 s54, s50, s53 // (op 1347) 


s_add_u32 s56, s4, 16 // (op 1159) Start of Int64 add, locking SCC
s_addc_u32 s57, s5, 0 // (op 1159) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1159) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1159) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 1159) 

s_mul_i32 s50, s11, s56 // (op 1159) 
s_mul_i32 s53, s8, s50 // (op 1159) 

s_mul_i32 s50, s51, s53 // (op 1159) 

s_sub_u32 s53, s16, s50 // (op 1159) 

s_mov_b32 s50, s53 // (op 1159) 
s_mul_hi_i32 s55, s50, s12 // (op 1159) 
s_add_i32 s58, s55, s50 // (op 1159) 


s_ashr_i32 s50, s58, 31 // (op 1159) 
s_cmp_eq_i32 s12, 0 // (op 1159) 
s_mov_b32 s55, scc // (op 1159) 
s_cmp_eq_i32 s55, 0 // (op 1159) 
s_cselect_b32 s59, 0, -1 // (op 1159) 

s_and_b32 s55, s44, 31 // (op 1159) 
s_lshl_b32 s60, 1, s55 // (op 1159) 
s_add_i32 s61, s60, s59 // (op 1159) 


s_and_b32 s59, s50, s61 // (op 1159) 


s_add_i32 s50, s58, s59 // (op 1159) 


s_ashr_i32 s58, s50, s55 // (op 1159) 


s_lshl_b32 s50, s44, 24 // (op 1159) 
s_ashr_i32 s55, s50, 31 // (op 1159) 

s_xor_b32 s50, s58, s55 // (op 1159) 

s_sub_i32 s58, s50, s55 // (op 1159) 


s_mul_i32 s50, s46, s11 // (op 1159) 
s_add_u32 s60, s6, 64 // (op 1159) Start of Int64 add, locking SCC
s_addc_u32 s61, s7, 0 // (op 1159) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 1159) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 1159) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 6 // (op 1159) 

s_sub_i32 s55, s60, s50 // (op 1159) 

s_mul_i32 s59, s58, s55 // (op 1159) 

s_sub_u32 s55, s53, s59 // (op 1159) 


s_add_u32 s53, s55, s50 // (op 1159) 


s_cmp_eq_i32 s51, 0 // (op 1159) 
s_mov_b32 s50, scc // (op 1159) 

s_cmp_eq_i32 s50, 0 // (op 1159) 
s_cselect_b32 s51, s53, s19 // (op 1159) 


s_mul_i32 s19, s49, s56 // (op 1159) 


v_lshl_add_u32 v8, s51, 6, v6 // (op 1159) 


s_sub_i32 s49, s18, s19 // (op 1159) 


s_cmp_eq_i32 s50, 0 // (op 1159) 
s_cselect_b32 s51, s58, s49 // (op 1159) 



v_lshl_add_u32 v6, s51, 4, v13 // (op 1159) 


v_mul_lo_u32 v9, v6, s2 // (op 1159) 

v_add_lshl_u32 v6, v8, v9, 2 // (op 1159) 


s_add_u32 s18, s4, 16 // (op 1347) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 1347) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 1347) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 1347) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 1347) 

s_mul_i32 s53, s11, s18 // (op 1347) 
s_mul_i32 s49, s8, s53 // (op 1347) 

s_mul_i32 s53, s54, s49 // (op 1347) 

s_sub_u32 s49, s16, s53 // (op 1347) 

s_mov_b32 s53, s49 // (op 1347) 
s_mul_hi_i32 s50, s53, s12 // (op 1347) 
s_add_i32 s51, s50, s53 // (op 1347) 


s_ashr_i32 s50, s51, 31 // (op 1347) 
s_cmp_eq_i32 s12, 0 // (op 1347) 
s_mov_b32 s53, scc // (op 1347) 
s_cmp_eq_i32 s53, 0 // (op 1347) 
s_cselect_b32 s55, 0, -1 // (op 1347) 

s_and_b32 s53, s44, 31 // (op 1347) 
s_lshl_b32 s56, 1, s53 // (op 1347) 
s_add_i32 s57, s56, s55 // (op 1347) 


s_and_b32 s55, s50, s57 // (op 1347) 


s_add_i32 s50, s51, s55 // (op 1347) 


s_ashr_i32 s51, s50, s53 // (op 1347) 


s_lshl_b32 s50, s44, 24 // (op 1347) 
s_ashr_i32 s53, s50, 31 // (op 1347) 

s_xor_b32 s50, s51, s53 // (op 1347) 

s_sub_i32 s51, s50, s53 // (op 1347) 


s_mul_i32 s50, s46, s11 // (op 1347) 
s_add_u32 s56, s6, 64 // (op 1347) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1347) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1347) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1347) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1347) 

s_sub_i32 s53, s56, s50 // (op 1347) 

s_mul_i32 s55, s51, s53 // (op 1347) 

s_sub_u32 s53, s49, s55 // (op 1347) 


s_add_u32 s49, s53, s50 // (op 1347) 


s_cmp_eq_i32 s54, 0 // (op 1347) 
s_mov_b32 s50, scc // (op 1347) 

s_cmp_eq_i32 s50, 0 // (op 1347) 
s_cselect_b32 s53, s49, s1 // (op 1347) 


s_mul_i32 s1, s52, s18 // (op 1347) 


v_lshl_add_u32 v8, s53, 6, v5 // (op 1347) 


s_sub_i32 s49, s0, s1 // (op 1347) 


s_cmp_eq_i32 s50, 0 // (op 1347) 
s_cselect_b32 s0, s51, s49 // (op 1347) 



v_lshl_add_u32 v5, s0, 4, v12 // (op 1347) 


v_mul_lo_u32 v9, v5, s20 // (op 1347) 

v_add_lshl_u32 v5, v8, v9, 2 // (op 1347) 


s_mul_i32 s1, s2, s5 // (op 1161) most significant: low of low * high
s_mul_i32 s18, s3, s4 // (op 1161) most significant: low of high * low
s_mul_i32 s0, s2, s4 // (op 1161) least significant
s_mul_hi_u32 s19, s2, s4 // (op 1161) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1161) most significant: sum
s_add_u32 s1, s1, s19 // (op 1161) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1161) 

s_mul_i32 s1, s20, s5 // (op 1349) most significant: low of low * high
s_mul_i32 s49, s21, s4 // (op 1349) most significant: low of high * low
s_mul_i32 s0, s20, s4 // (op 1349) least significant
s_mul_hi_u32 s50, s20, s4 // (op 1349) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 1349) most significant: sum
s_add_u32 s1, s1, s50 // (op 1349) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 2 // (op 1349) 

s_mov_b32 s52, s22 // (op 1161) reg expression
s_mov_b32 s53, s23 // (op 1161) reg expression
s_mov_b32 s54, s18 // (op 1161) reg expression
s_mov_b32 s55, 131072 // (op 1161) call()

s_mov_b32 s56, s24 // (op 1349) reg expression

s_mov_b32 s57, s25 // (op 1349) reg expression
s_mov_b32 s58, s50 // (op 1349) reg expression
s_mov_b32 s59, 131072 // (op 1349) call()


v_mov_b32 v8, s47 // (op 2) Move value
v_mov_b32 v9, s48 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 309) call()
v_accvgpr_write a5, 0.00000 // (op 309) call()
v_accvgpr_write a6, 0.00000 // (op 309) call()
v_accvgpr_write a7, 0.00000 // (op 309) call()
s_mul_hi_i32 s0, s16, s10 // (op 926) 
v_lshrrev_b32 v12, 3, v1 // (op 926) 
s_add_i32 s1, s0, s16 // (op 926) 

v_and_b32 v13, -8, v1 // (op 926) 
s_ashr_i32 s0, s1, 31 // (op 926) 
v_sub_u32 v14, v1, v13 // (op 926) 

s_cmp_eq_i32 s10, 0 // (op 926) 
s_mov_b32 s18, scc // (op 926) 
v_lshlrev_b32 v13, 4, v14 // (op 926) 

s_cmp_eq_i32 s18, 0 // (op 926) 
s_cselect_b32 s19, 0, -1 // (op 926) 

v_mov_b32 v15, 0 // (op 926) convert
v_mov_b32 v14, v13 // (op 926) convert

s_and_b32 s18, s13, 31 // (op 926) 
s_lshl_b32 s22, 1, s18 // (op 926) 
s_add_i32 s23, s22, s19 // (op 926) 


s_and_b32 s19, s0, s23 // (op 926) 


s_add_i32 s0, s1, s19 // (op 926) 


s_ashr_i32 s1, s0, s18 // (op 926) 


s_lshl_b32 s0, s13, 24 // (op 926) 
s_ashr_i32 s18, s0, 31 // (op 926) 

s_xor_b32 s0, s1, s18 // (op 926) 

s_sub_i32 s1, s0, s18 // (op 926) 


s_mul_hi_i32 s0, s1, s14 // (op 926) 
s_add_i32 s18, s0, s1 // (op 926) 

s_ashr_i32 s0, s18, 31 // (op 926) 
s_cmp_eq_i32 s14, 0 // (op 926) 
s_mov_b32 s19, scc // (op 926) 
s_cmp_eq_i32 s19, 0 // (op 926) 
s_cselect_b32 s22, 0, -1 // (op 926) 

s_and_b32 s19, s15, 31 // (op 926) 
s_lshl_b32 s23, 1, s19 // (op 926) 
s_add_i32 s24, s23, s22 // (op 926) 


s_and_b32 s25, s0, s24 // (op 926) 


s_add_i32 s0, s18, s25 // (op 926) 


s_ashr_i32 s18, s0, s19 // (op 926) 


s_lshl_b32 s0, s15, 24 // (op 926) 
s_ashr_i32 s19, s0, 31 // (op 926) 

s_xor_b32 s0, s18, s19 // (op 926) 

s_sub_i32 s18, s0, s19 // (op 926) 


s_mul_i32 s0, s1, s11 // (op 926) 

s_sub_u32 s1, s16, s0 // (op 926) 

s_mul_i32 s0, s18, s11 // (op 926) 
s_add_u32 s0, s0, s1 // (op 926) 


s_mul_hi_i32 s1, s16, s9 // (op 926) 
s_add_i32 s18, s1, s16 // (op 926) 

s_ashr_i32 s1, s18, 31 // (op 926) 
s_cmp_eq_i32 s9, 0 // (op 926) 
s_mov_b32 s19, scc // (op 926) 
s_cmp_eq_i32 s19, 0 // (op 926) 
s_cselect_b32 s22, 0, -1 // (op 926) 

s_and_b32 s19, s45, 31 // (op 926) 
s_lshl_b32 s23, 1, s19 // (op 926) 
s_add_i32 s24, s23, s22 // (op 926) 


s_and_b32 s25, s1, s24 // (op 926) 


s_add_i32 s1, s18, s25 // (op 926) 


s_ashr_i32 s18, s1, s19 // (op 926) 


s_lshl_b32 s1, s45, 24 // (op 926) 
s_ashr_i32 s19, s1, 31 // (op 926) 

s_xor_b32 s1, s18, s19 // (op 926) 

s_sub_i32 s18, s1, s19 // (op 926) 


s_add_u32 s22, s4, 16 // (op 926) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 926) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 926) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 926) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 926) 

s_mul_i32 s1, s11, s22 // (op 926) 

s_mul_i32 s19, s8, s1 // (op 926) 

s_mul_i32 s1, s18, s19 // (op 926) 

s_sub_u32 s19, s16, s1 // (op 926) 

s_mov_b32 s1, s19 // (op 926) 
s_mul_hi_i32 s22, s1, s12 // (op 926) 
s_add_i32 s23, s22, s1 // (op 926) 


s_ashr_i32 s1, s23, 31 // (op 926) 
s_cmp_eq_i32 s12, 0 // (op 926) 
s_mov_b32 s22, scc // (op 926) 
s_cmp_eq_i32 s22, 0 // (op 926) 
s_cselect_b32 s24, 0, -1 // (op 926) 

s_and_b32 s22, s44, 31 // (op 926) 
s_lshl_b32 s25, 1, s22 // (op 926) 
s_add_i32 s47, s25, s24 // (op 926) 


s_and_b32 s24, s1, s47 // (op 926) 


s_add_i32 s1, s23, s24 // (op 926) 


s_ashr_i32 s23, s1, s22 // (op 926) 


s_lshl_b32 s1, s44, 24 // (op 926) 
s_ashr_i32 s22, s1, 31 // (op 926) 

s_xor_b32 s1, s23, s22 // (op 926) 

s_sub_i32 s23, s1, s22 // (op 926) 


s_mul_i32 s1, s46, s11 // (op 926) 
s_add_u32 s24, s6, 64 // (op 926) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 926) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 926) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 926) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 926) 

s_sub_i32 s22, s24, s1 // (op 926) 

s_mul_i32 s24, s23, s22 // (op 926) 


s_sub_u32 s25, s19, s24 // (op 926) 


s_add_u32 s19, s25, s1 // (op 926) 


s_cmp_eq_i32 s18, 0 // (op 926) 
s_mov_b32 s1, scc // (op 926) 

s_cmp_eq_i32 s1, 0 // (op 926) 
s_cselect_b32 s18, s19, s0 // (op 926) 



v_lshl_add_u32 v13, s18, 6, v12 // (op 926) 


v_mul_lo_u32 v17, v13, s27 // (op 926) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 926) most significant: high of low * low
v_mul_lo_u32 v16, v13, s26 // (op 926) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 926) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 926) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 926) most significant half



s_mul_i32 s1, s26, s7 // (op 928) most significant: low of low * high
s_mul_i32 s18, s27, s6 // (op 928) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 928) least significant
s_mul_hi_u32 s19, s26, s6 // (op 928) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 928) most significant: sum
s_add_u32 s1, s1, s19 // (op 928) most significant: sum


s_mov_b32 s48, s28 // (op 928) reg expression
s_mov_b32 s49, s29 // (op 928) reg expression
s_mov_b32 s50, s0 // (op 928) reg expression
s_mov_b32 s51, 131072 // (op 928) call()

s_mul_hi_i32 s47, s16, s10 // (op 945) 
v_lshrrev_b32 v12, 3, v1 // (op 945) 
s_add_i32 s0, s47, s16 // (op 945) 

v_and_b32 v13, -8, v1 // (op 945) 
s_ashr_i32 s1, s0, 31 // (op 945) 
v_sub_u32 v14, v1, v13 // (op 945) 

s_cmp_eq_i32 s10, 0 // (op 945) 
s_mov_b32 s47, scc // (op 945) 
v_lshlrev_b32 v13, 4, v14 // (op 945) 

s_cmp_eq_i32 s47, 0 // (op 945) 
s_cselect_b32 s18, 0, -1 // (op 945) 

v_mov_b32 v15, 0 // (op 945) convert
v_mov_b32 v14, v13 // (op 945) convert

s_and_b32 s19, s13, 31 // (op 945) 
s_lshl_b32 s47, 1, s19 // (op 945) 
s_add_i32 s22, s47, s18 // (op 945) 


s_and_b32 s18, s1, s22 // (op 945) 


s_add_i32 s1, s0, s18 // (op 945) 


s_ashr_i32 s0, s1, s19 // (op 945) 


s_lshl_b32 s1, s13, 24 // (op 945) 
s_ashr_i32 s47, s1, 31 // (op 945) 

s_xor_b32 s1, s0, s47 // (op 945) 

s_sub_i32 s0, s1, s47 // (op 945) 


s_mul_hi_i32 s1, s0, s14 // (op 945) 
s_add_i32 s47, s1, s0 // (op 945) 

s_ashr_i32 s1, s47, 31 // (op 945) 
s_cmp_eq_i32 s14, 0 // (op 945) 
s_mov_b32 s18, scc // (op 945) 
s_cmp_eq_i32 s18, 0 // (op 945) 
s_cselect_b32 s19, 0, -1 // (op 945) 

s_and_b32 s18, s15, 31 // (op 945) 
s_lshl_b32 s22, 1, s18 // (op 945) 
s_add_i32 s23, s22, s19 // (op 945) 


s_and_b32 s19, s1, s23 // (op 945) 


s_add_i32 s1, s47, s19 // (op 945) 


s_ashr_i32 s19, s1, s18 // (op 945) 


s_lshl_b32 s1, s15, 24 // (op 945) 
s_ashr_i32 s18, s1, 31 // (op 945) 

s_xor_b32 s1, s19, s18 // (op 945) 

s_sub_i32 s19, s1, s18 // (op 945) 


s_mul_i32 s1, s0, s11 // (op 945) 

s_sub_u32 s0, s16, s1 // (op 945) 

s_mul_i32 s1, s19, s11 // (op 945) 
s_add_u32 s1, s1, s0 // (op 945) 


s_mul_hi_i32 s0, s16, s9 // (op 945) 
s_add_i32 s47, s0, s16 // (op 945) 

s_ashr_i32 s0, s47, 31 // (op 945) 
s_cmp_eq_i32 s9, 0 // (op 945) 
s_mov_b32 s18, scc // (op 945) 
s_cmp_eq_i32 s18, 0 // (op 945) 
s_cselect_b32 s19, 0, -1 // (op 945) 

s_and_b32 s18, s45, 31 // (op 945) 
s_lshl_b32 s22, 1, s18 // (op 945) 
s_add_i32 s23, s22, s19 // (op 945) 


s_and_b32 s19, s0, s23 // (op 945) 


s_add_i32 s0, s47, s19 // (op 945) 


s_ashr_i32 s19, s0, s18 // (op 945) 


s_lshl_b32 s0, s45, 24 // (op 945) 
s_ashr_i32 s18, s0, 31 // (op 945) 

s_xor_b32 s0, s19, s18 // (op 945) 

s_sub_i32 s19, s0, s18 // (op 945) 


s_add_u32 s22, s4, 16 // (op 945) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 945) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 945) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 945) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 945) 

s_mul_i32 s0, s11, s22 // (op 945) 

s_mul_i32 s18, s8, s0 // (op 945) 

s_mul_i32 s0, s19, s18 // (op 945) 

s_sub_u32 s18, s16, s0 // (op 945) 

s_mov_b32 s0, s18 // (op 945) 
s_mul_hi_i32 s47, s0, s12 // (op 945) 
s_add_i32 s22, s47, s0 // (op 945) 


s_ashr_i32 s0, s22, 31 // (op 945) 
s_cmp_eq_i32 s12, 0 // (op 945) 
s_mov_b32 s47, scc // (op 945) 
s_cmp_eq_i32 s47, 0 // (op 945) 
s_cselect_b32 s23, 0, -1 // (op 945) 

s_and_b32 s47, s44, 31 // (op 945) 
s_lshl_b32 s24, 1, s47 // (op 945) 
s_add_i32 s25, s24, s23 // (op 945) 


s_and_b32 s23, s0, s25 // (op 945) 


s_add_i32 s0, s22, s23 // (op 945) 


s_ashr_i32 s22, s0, s47 // (op 945) 


s_lshl_b32 s0, s44, 24 // (op 945) 
s_ashr_i32 s47, s0, 31 // (op 945) 

s_xor_b32 s0, s22, s47 // (op 945) 

s_sub_i32 s22, s0, s47 // (op 945) 


s_mul_i32 s0, s46, s11 // (op 945) 
s_add_u32 s24, s6, 64 // (op 945) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 945) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 945) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 945) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 6 // (op 945) 

s_sub_i32 s23, s24, s0 // (op 945) 

s_mul_i32 s47, s22, s23 // (op 945) 


s_sub_u32 s22, s18, s47 // (op 945) 


s_add_u32 s18, s22, s0 // (op 945) 


s_cmp_eq_i32 s19, 0 // (op 945) 
s_mov_b32 s0, scc // (op 945) 

s_cmp_eq_i32 s0, 0 // (op 945) 
s_cselect_b32 s19, s18, s1 // (op 945) 



v_lshl_add_u32 v13, s19, 6, v12 // (op 945) 


v_mul_lo_u32 v19, v13, s27 // (op 945) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 945) most significant: high of low * low
v_mul_lo_u32 v18, v13, s26 // (op 945) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 945) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 945) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 945) most significant half



s_mul_i32 s1, s26, s7 // (op 947) most significant: low of low * high
s_mul_i32 s47, s27, s6 // (op 947) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 947) least significant
s_mul_hi_u32 s18, s26, s6 // (op 947) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 947) most significant: sum
s_add_u32 s1, s1, s18 // (op 947) most significant: sum


s_mov_b32 s48, s28 // (op 947) reg expression
s_mov_b32 s49, s29 // (op 947) reg expression
s_mov_b32 s50, s0 // (op 947) reg expression
s_mov_b32 s51, 131072 // (op 947) call()

s_mul_hi_i32 s47, s16, s10 // (op 990) 
v_lshrrev_b32 v12, 3, v1 // (op 990) 
s_add_i32 s0, s47, s16 // (op 990) 

v_and_b32 v13, -8, v1 // (op 990) 
s_ashr_i32 s1, s0, 31 // (op 990) 
v_sub_u32 v14, v1, v13 // (op 990) 

s_cmp_eq_i32 s10, 0 // (op 990) 
s_mov_b32 s47, scc // (op 990) 
v_lshlrev_b32 v13, 4, v14 // (op 990) 

s_cmp_eq_i32 s47, 0 // (op 990) 
s_cselect_b32 s18, 0, -1 // (op 990) 

v_mov_b32 v15, 0 // (op 990) convert
v_mov_b32 v14, v13 // (op 990) convert

s_and_b32 s19, s13, 31 // (op 990) 
s_lshl_b32 s47, 1, s19 // (op 990) 
s_add_i32 s22, s47, s18 // (op 990) 


s_and_b32 s18, s1, s22 // (op 990) 


s_add_i32 s1, s0, s18 // (op 990) 


s_ashr_i32 s0, s1, s19 // (op 990) 


s_lshl_b32 s1, s13, 24 // (op 990) 
s_ashr_i32 s47, s1, 31 // (op 990) 

s_xor_b32 s1, s0, s47 // (op 990) 

s_sub_i32 s0, s1, s47 // (op 990) 


s_mul_hi_i32 s1, s0, s14 // (op 990) 
s_add_i32 s47, s1, s0 // (op 990) 

s_ashr_i32 s1, s47, 31 // (op 990) 
s_cmp_eq_i32 s14, 0 // (op 990) 
s_mov_b32 s18, scc // (op 990) 
s_cmp_eq_i32 s18, 0 // (op 990) 
s_cselect_b32 s19, 0, -1 // (op 990) 

s_and_b32 s18, s15, 31 // (op 990) 
s_lshl_b32 s22, 1, s18 // (op 990) 
s_add_i32 s23, s22, s19 // (op 990) 


s_and_b32 s19, s1, s23 // (op 990) 


s_add_i32 s1, s47, s19 // (op 990) 


s_ashr_i32 s19, s1, s18 // (op 990) 


s_lshl_b32 s1, s15, 24 // (op 990) 
s_ashr_i32 s18, s1, 31 // (op 990) 

s_xor_b32 s1, s19, s18 // (op 990) 

s_sub_i32 s19, s1, s18 // (op 990) 


s_add_u32 s22, s4, 16 // (op 990) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 990) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 990) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 990) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 990) 

s_mul_i32 s1, s19, s22 // (op 990) 

s_sub_i32 s47, s0, s1 // (op 990) 


s_mul_hi_i32 s0, s16, s9 // (op 990) 
s_add_i32 s1, s0, s16 // (op 990) 

s_ashr_i32 s0, s1, 31 // (op 990) 
s_cmp_eq_i32 s9, 0 // (op 990) 
s_mov_b32 s18, scc // (op 990) 
s_cmp_eq_i32 s18, 0 // (op 990) 
s_cselect_b32 s19, 0, -1 // (op 990) 

s_and_b32 s18, s45, 31 // (op 990) 
s_lshl_b32 s24, 1, s18 // (op 990) 
s_add_i32 s25, s24, s19 // (op 990) 


s_and_b32 s19, s0, s25 // (op 990) 


s_add_i32 s0, s1, s19 // (op 990) 


s_ashr_i32 s1, s0, s18 // (op 990) 


s_lshl_b32 s0, s45, 24 // (op 990) 
s_ashr_i32 s18, s0, 31 // (op 990) 

s_xor_b32 s0, s1, s18 // (op 990) 

s_sub_i32 s1, s0, s18 // (op 990) 


s_mul_i32 s0, s11, s22 // (op 990) 

s_mul_i32 s18, s8, s0 // (op 990) 

s_mul_i32 s0, s1, s18 // (op 990) 

s_sub_u32 s18, s16, s0 // (op 990) 

s_mul_hi_i32 s0, s18, s12 // (op 990) 
s_add_i32 s19, s0, s18 // (op 990) 


s_ashr_i32 s0, s19, 31 // (op 990) 
s_cmp_eq_i32 s12, 0 // (op 990) 
s_mov_b32 s18, scc // (op 990) 
s_cmp_eq_i32 s18, 0 // (op 990) 
s_cselect_b32 s22, 0, -1 // (op 990) 

s_and_b32 s18, s44, 31 // (op 990) 
s_lshl_b32 s23, 1, s18 // (op 990) 
s_add_i32 s24, s23, s22 // (op 990) 


s_and_b32 s25, s0, s24 // (op 990) 


s_add_i32 s0, s19, s25 // (op 990) 


s_ashr_i32 s19, s0, s18 // (op 990) 


s_lshl_b32 s0, s44, 24 // (op 990) 
s_ashr_i32 s18, s0, 31 // (op 990) 

s_xor_b32 s0, s19, s18 // (op 990) 

s_sub_i32 s19, s0, s18 // (op 990) 


s_cmp_eq_i32 s1, 0 // (op 990) 
s_mov_b32 s0, scc // (op 990) 

s_cmp_eq_i32 s0, 0 // (op 990) 
s_cselect_b32 s1, s19, s47 // (op 990) 



v_lshl_add_u32 v13, s1, 4, v12 // (op 990) 


v_mul_lo_u32 v21, v13, s31 // (op 990) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 990) most significant: high of low * low
v_mul_lo_u32 v20, v13, s30 // (op 990) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 990) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 990) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 990) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 990) most significant: low of high * low
v_mul_hi_u32 v20, v12, 6 // (op 990) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 990) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 990) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 990) 

v_mov_b32 v14, v12 // (op 990) convert

s_mul_i32 s1, s5, 6 // (op 992) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 992) least significant
s_mul_hi_u32 s47, s4, 6 // (op 992) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 992) most significant: sum

s_mul_i32 s19, s30, s1 // (op 992) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 992) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 992) least significant
s_mul_hi_u32 s22, s30, s0 // (op 992) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 992) most significant: sum
s_add_u32 s19, s19, s22 // (op 992) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 992) 

s_mov_b32 s60, s32 // (op 992) reg expression
s_mov_b32 s61, s33 // (op 992) reg expression
s_mov_b32 s62, s0 // (op 992) reg expression
s_mov_b32 s63, 131072 // (op 992) call()

s_mul_hi_i32 s47, s16, s10 // (op 1009) 
v_lshrrev_b32 v15, 3, v1 // (op 1009) 
s_add_i32 s0, s47, s16 // (op 1009) 

v_and_b32 v12, -8, v1 // (op 1009) 
s_ashr_i32 s1, s0, 31 // (op 1009) 
v_sub_u32 v13, v1, v12 // (op 1009) 

s_cmp_eq_i32 s10, 0 // (op 1009) 
s_mov_b32 s47, scc // (op 1009) 
v_lshlrev_b32 v12, 4, v13 // (op 1009) 

s_cmp_eq_i32 s47, 0 // (op 1009) 
s_cselect_b32 s18, 0, -1 // (op 1009) 

v_mov_b32 v21, 0 // (op 1009) convert
v_mov_b32 v20, v12 // (op 1009) convert

s_and_b32 s19, s13, 31 // (op 1009) 
s_lshl_b32 s47, 1, s19 // (op 1009) 
s_add_i32 s22, s47, s18 // (op 1009) 


s_and_b32 s18, s1, s22 // (op 1009) 


s_add_i32 s1, s0, s18 // (op 1009) 


s_ashr_i32 s0, s1, s19 // (op 1009) 


s_lshl_b32 s1, s13, 24 // (op 1009) 
s_ashr_i32 s47, s1, 31 // (op 1009) 

s_xor_b32 s1, s0, s47 // (op 1009) 

s_sub_i32 s0, s1, s47 // (op 1009) 


s_mul_hi_i32 s1, s0, s14 // (op 1009) 
s_add_i32 s47, s1, s0 // (op 1009) 

s_ashr_i32 s1, s47, 31 // (op 1009) 
s_cmp_eq_i32 s14, 0 // (op 1009) 
s_mov_b32 s18, scc // (op 1009) 
s_cmp_eq_i32 s18, 0 // (op 1009) 
s_cselect_b32 s19, 0, -1 // (op 1009) 

s_and_b32 s18, s15, 31 // (op 1009) 
s_lshl_b32 s22, 1, s18 // (op 1009) 
s_add_i32 s23, s22, s19 // (op 1009) 


s_and_b32 s19, s1, s23 // (op 1009) 


s_add_i32 s1, s47, s19 // (op 1009) 


s_ashr_i32 s19, s1, s18 // (op 1009) 


s_lshl_b32 s1, s15, 24 // (op 1009) 
s_ashr_i32 s18, s1, 31 // (op 1009) 

s_xor_b32 s1, s19, s18 // (op 1009) 

s_sub_i32 s19, s1, s18 // (op 1009) 


s_add_u32 s22, s4, 16 // (op 1009) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1009) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1009) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1009) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1009) 

s_mul_i32 s1, s19, s22 // (op 1009) 

s_sub_i32 s47, s0, s1 // (op 1009) 


s_mul_hi_i32 s0, s16, s9 // (op 1009) 
s_add_i32 s1, s0, s16 // (op 1009) 

s_ashr_i32 s0, s1, 31 // (op 1009) 
s_cmp_eq_i32 s9, 0 // (op 1009) 
s_mov_b32 s18, scc // (op 1009) 
s_cmp_eq_i32 s18, 0 // (op 1009) 
s_cselect_b32 s19, 0, -1 // (op 1009) 

s_and_b32 s18, s45, 31 // (op 1009) 
s_lshl_b32 s24, 1, s18 // (op 1009) 
s_add_i32 s25, s24, s19 // (op 1009) 


s_and_b32 s19, s0, s25 // (op 1009) 


s_add_i32 s0, s1, s19 // (op 1009) 


s_ashr_i32 s1, s0, s18 // (op 1009) 


s_lshl_b32 s0, s45, 24 // (op 1009) 
s_ashr_i32 s18, s0, 31 // (op 1009) 

s_xor_b32 s0, s1, s18 // (op 1009) 

s_sub_i32 s1, s0, s18 // (op 1009) 


s_mul_i32 s0, s11, s22 // (op 1009) 

s_mul_i32 s18, s8, s0 // (op 1009) 

s_mul_i32 s0, s1, s18 // (op 1009) 

s_sub_u32 s18, s16, s0 // (op 1009) 

s_mul_hi_i32 s0, s18, s12 // (op 1009) 
s_add_i32 s19, s0, s18 // (op 1009) 


s_ashr_i32 s0, s19, 31 // (op 1009) 
s_cmp_eq_i32 s12, 0 // (op 1009) 
s_mov_b32 s18, scc // (op 1009) 
s_cmp_eq_i32 s18, 0 // (op 1009) 
s_cselect_b32 s22, 0, -1 // (op 1009) 

s_and_b32 s18, s44, 31 // (op 1009) 
s_lshl_b32 s23, 1, s18 // (op 1009) 
s_add_i32 s24, s23, s22 // (op 1009) 


s_and_b32 s25, s0, s24 // (op 1009) 


s_add_i32 s0, s19, s25 // (op 1009) 


s_ashr_i32 s19, s0, s18 // (op 1009) 


s_lshl_b32 s0, s44, 24 // (op 1009) 
s_ashr_i32 s18, s0, 31 // (op 1009) 

s_xor_b32 s0, s19, s18 // (op 1009) 

s_sub_i32 s19, s0, s18 // (op 1009) 


s_cmp_eq_i32 s1, 0 // (op 1009) 
s_mov_b32 s0, scc // (op 1009) 

s_cmp_eq_i32 s0, 0 // (op 1009) 
s_cselect_b32 s1, s19, s47 // (op 1009) 



v_lshl_add_u32 v12, s1, 4, v15 // (op 1009) 


v_mul_lo_u32 v23, v12, s31 // (op 1009) most significant: low of low * high
v_mul_hi_u32 v13, v12, s30 // (op 1009) most significant: high of low * low
v_mul_lo_u32 v22, v12, s30 // (op 1009) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 1009) most significant: sum


v_add_co_u32 v12, s[0:1], v20, v22 // (op 1009) least significant half
v_addc_co_u32 v13, s[0:1], v21, v23, s[0:1] // (op 1009) most significant half



v_mul_lo_u32 v21, v13, 6 // (op 1009) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 1009) most significant: high of low * low
v_mul_lo_u32 v20, v12, 6 // (op 1009) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 1009) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[20:21] // (op 1009) 

v_mov_b32 v15, v12 // (op 1009) convert

s_mul_i32 s1, s5, 6 // (op 1011) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 1011) least significant
s_mul_hi_u32 s47, s4, 6 // (op 1011) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1011) most significant: sum

s_mul_i32 s19, s30, s1 // (op 1011) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 1011) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 1011) least significant
s_mul_hi_u32 s22, s30, s0 // (op 1011) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 1011) most significant: sum
s_add_u32 s19, s19, s22 // (op 1011) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1011) 

s_mov_b32 s60, s32 // (op 1011) reg expression
s_mov_b32 s61, s33 // (op 1011) reg expression
s_mov_b32 s62, s0 // (op 1011) reg expression
s_mov_b32 s63, 131072 // (op 1011) call()

s_mul_hi_i32 s47, s16, s10 // (op 1054) 
v_lshrrev_b32 v12, 2, v1 // (op 1054) 
s_add_i32 s0, s47, s16 // (op 1054) 

v_lshlrev_b32 v13, 1, v12 // (op 1054) 

s_ashr_i32 s1, s0, 31 // (op 1054) 
v_and_b32 v12, -4, v1 // (op 1054) 
s_cmp_eq_i32 s10, 0 // (op 1054) 
s_mov_b32 s47, scc // (op 1054) 
v_sub_u32 v20, v1, v12 // (op 1054) 

s_cmp_eq_i32 s47, 0 // (op 1054) 
s_cselect_b32 s18, 0, -1 // (op 1054) 

s_and_b32 s19, s13, 31 // (op 1054) 
s_lshl_b32 s47, 1, s19 // (op 1054) 
s_add_i32 s22, s47, s18 // (op 1054) 


s_and_b32 s18, s1, s22 // (op 1054) 


s_add_i32 s1, s0, s18 // (op 1054) 


s_ashr_i32 s0, s1, s19 // (op 1054) 


s_lshl_b32 s1, s13, 24 // (op 1054) 
s_ashr_i32 s47, s1, 31 // (op 1054) 

s_xor_b32 s1, s0, s47 // (op 1054) 

s_sub_i32 s0, s1, s47 // (op 1054) 


s_mul_hi_i32 s1, s0, s14 // (op 1054) 
s_add_i32 s47, s1, s0 // (op 1054) 

s_ashr_i32 s1, s47, 31 // (op 1054) 
s_cmp_eq_i32 s14, 0 // (op 1054) 
s_mov_b32 s18, scc // (op 1054) 
s_cmp_eq_i32 s18, 0 // (op 1054) 
s_cselect_b32 s19, 0, -1 // (op 1054) 

s_and_b32 s18, s15, 31 // (op 1054) 
s_lshl_b32 s22, 1, s18 // (op 1054) 
s_add_i32 s23, s22, s19 // (op 1054) 


s_and_b32 s19, s1, s23 // (op 1054) 


s_add_i32 s1, s47, s19 // (op 1054) 


s_ashr_i32 s19, s1, s18 // (op 1054) 


s_lshl_b32 s1, s15, 24 // (op 1054) 
s_ashr_i32 s18, s1, 31 // (op 1054) 

s_xor_b32 s1, s19, s18 // (op 1054) 

s_sub_i32 s19, s1, s18 // (op 1054) 


s_mul_i32 s1, s0, s11 // (op 1054) 

s_sub_u32 s0, s16, s1 // (op 1054) 

s_mul_i32 s1, s19, s11 // (op 1054) 
s_add_u32 s1, s1, s0 // (op 1054) 


s_mul_hi_i32 s0, s16, s9 // (op 1054) 
s_add_i32 s47, s0, s16 // (op 1054) 

s_ashr_i32 s0, s47, 31 // (op 1054) 
s_cmp_eq_i32 s9, 0 // (op 1054) 
s_mov_b32 s18, scc // (op 1054) 
s_cmp_eq_i32 s18, 0 // (op 1054) 
s_cselect_b32 s19, 0, -1 // (op 1054) 

s_and_b32 s18, s45, 31 // (op 1054) 
s_lshl_b32 s22, 1, s18 // (op 1054) 
s_add_i32 s23, s22, s19 // (op 1054) 


s_and_b32 s19, s0, s23 // (op 1054) 


s_add_i32 s0, s47, s19 // (op 1054) 


s_ashr_i32 s19, s0, s18 // (op 1054) 


s_lshl_b32 s0, s45, 24 // (op 1054) 
s_ashr_i32 s18, s0, 31 // (op 1054) 

s_xor_b32 s0, s19, s18 // (op 1054) 

s_sub_i32 s19, s0, s18 // (op 1054) 


s_add_u32 s22, s4, 16 // (op 1054) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1054) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1054) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1054) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1054) 

s_mul_i32 s0, s11, s22 // (op 1054) 

s_mul_i32 s18, s8, s0 // (op 1054) 

s_mul_i32 s0, s19, s18 // (op 1054) 

s_sub_u32 s18, s16, s0 // (op 1054) 

s_mov_b32 s0, s18 // (op 1054) 
s_mul_hi_i32 s47, s0, s12 // (op 1054) 
s_add_i32 s22, s47, s0 // (op 1054) 


s_ashr_i32 s0, s22, 31 // (op 1054) 
s_cmp_eq_i32 s12, 0 // (op 1054) 
s_mov_b32 s47, scc // (op 1054) 
s_cmp_eq_i32 s47, 0 // (op 1054) 
s_cselect_b32 s23, 0, -1 // (op 1054) 

s_and_b32 s47, s44, 31 // (op 1054) 
s_lshl_b32 s24, 1, s47 // (op 1054) 
s_add_i32 s25, s24, s23 // (op 1054) 


s_and_b32 s23, s0, s25 // (op 1054) 


s_add_i32 s0, s22, s23 // (op 1054) 


s_ashr_i32 s22, s0, s47 // (op 1054) 


s_lshl_b32 s0, s44, 24 // (op 1054) 
s_ashr_i32 s47, s0, 31 // (op 1054) 

s_xor_b32 s0, s22, s47 // (op 1054) 

s_sub_i32 s22, s0, s47 // (op 1054) 


s_mul_i32 s0, s46, s11 // (op 1054) 
s_add_u32 s24, s6, 64 // (op 1054) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1054) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 1054) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 1054) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 6 // (op 1054) 

s_sub_i32 s23, s24, s0 // (op 1054) 

s_mul_i32 s47, s22, s23 // (op 1054) 


s_sub_u32 s22, s18, s47 // (op 1054) 


s_add_u32 s18, s22, s0 // (op 1054) 


s_cmp_eq_i32 s19, 0 // (op 1054) 
s_mov_b32 s0, scc // (op 1054) 

s_cmp_eq_i32 s0, 0 // (op 1054) 
s_cselect_b32 s19, s18, s1 // (op 1054) 



v_lshl_add_u32 v12, s19, 6, v13 // (op 1054) 


v_mul_lo_u32 v13, v12, s34 // (op 1054) 
v_add_u32 v13, v13, v20 // (op 1054) 


s_mul_i32 s1, s34, s7 // (op 1056) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 1056) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 1056) least significant
s_mul_hi_u32 s18, s34, s6 // (op 1056) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1056) most significant: sum
s_add_u32 s1, s1, s18 // (op 1056) most significant: sum


s_mov_b32 s64, s36 // (op 1056) reg expression
s_mov_b32 s65, s37 // (op 1056) reg expression
s_mov_b32 s66, s0 // (op 1056) reg expression
s_mov_b32 s67, 131072 // (op 1056) call()

s_mul_hi_i32 s47, s16, s10 // (op 1073) 
v_lshrrev_b32 v12, 2, v1 // (op 1073) 
s_add_i32 s0, s47, s16 // (op 1073) 

v_lshlrev_b32 v20, 1, v12 // (op 1073) 

s_ashr_i32 s1, s0, 31 // (op 1073) 
v_and_b32 v12, -4, v1 // (op 1073) 
s_cmp_eq_i32 s10, 0 // (op 1073) 
s_mov_b32 s47, scc // (op 1073) 
v_sub_u32 v21, v1, v12 // (op 1073) 

s_cmp_eq_i32 s47, 0 // (op 1073) 
s_cselect_b32 s18, 0, -1 // (op 1073) 

s_and_b32 s19, s13, 31 // (op 1073) 
s_lshl_b32 s47, 1, s19 // (op 1073) 
s_add_i32 s22, s47, s18 // (op 1073) 


s_and_b32 s18, s1, s22 // (op 1073) 


s_add_i32 s1, s0, s18 // (op 1073) 


s_ashr_i32 s0, s1, s19 // (op 1073) 


s_lshl_b32 s1, s13, 24 // (op 1073) 
s_ashr_i32 s47, s1, 31 // (op 1073) 

s_xor_b32 s1, s0, s47 // (op 1073) 

s_sub_i32 s0, s1, s47 // (op 1073) 


s_mul_hi_i32 s1, s0, s14 // (op 1073) 
s_add_i32 s47, s1, s0 // (op 1073) 

s_ashr_i32 s1, s47, 31 // (op 1073) 
s_cmp_eq_i32 s14, 0 // (op 1073) 
s_mov_b32 s18, scc // (op 1073) 
s_cmp_eq_i32 s18, 0 // (op 1073) 
s_cselect_b32 s19, 0, -1 // (op 1073) 

s_and_b32 s18, s15, 31 // (op 1073) 
s_lshl_b32 s22, 1, s18 // (op 1073) 
s_add_i32 s23, s22, s19 // (op 1073) 


s_and_b32 s19, s1, s23 // (op 1073) 


s_add_i32 s1, s47, s19 // (op 1073) 


s_ashr_i32 s19, s1, s18 // (op 1073) 


s_lshl_b32 s1, s15, 24 // (op 1073) 
s_ashr_i32 s18, s1, 31 // (op 1073) 

s_xor_b32 s1, s19, s18 // (op 1073) 

s_sub_i32 s19, s1, s18 // (op 1073) 


s_mul_i32 s1, s0, s11 // (op 1073) 

s_sub_u32 s0, s16, s1 // (op 1073) 

s_mul_i32 s1, s19, s11 // (op 1073) 
s_add_u32 s1, s1, s0 // (op 1073) 


s_mul_hi_i32 s0, s16, s9 // (op 1073) 
s_add_i32 s47, s0, s16 // (op 1073) 

s_ashr_i32 s0, s47, 31 // (op 1073) 
s_cmp_eq_i32 s9, 0 // (op 1073) 
s_mov_b32 s18, scc // (op 1073) 
s_cmp_eq_i32 s18, 0 // (op 1073) 
s_cselect_b32 s19, 0, -1 // (op 1073) 

s_and_b32 s18, s45, 31 // (op 1073) 
s_lshl_b32 s22, 1, s18 // (op 1073) 
s_add_i32 s23, s22, s19 // (op 1073) 


s_and_b32 s19, s0, s23 // (op 1073) 


s_add_i32 s0, s47, s19 // (op 1073) 


s_ashr_i32 s19, s0, s18 // (op 1073) 


s_lshl_b32 s0, s45, 24 // (op 1073) 
s_ashr_i32 s18, s0, 31 // (op 1073) 

s_xor_b32 s0, s19, s18 // (op 1073) 

s_sub_i32 s19, s0, s18 // (op 1073) 


s_add_u32 s22, s4, 16 // (op 1073) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1073) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1073) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1073) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1073) 

s_mul_i32 s0, s11, s22 // (op 1073) 

s_mul_i32 s18, s8, s0 // (op 1073) 

s_mul_i32 s0, s19, s18 // (op 1073) 

s_sub_u32 s18, s16, s0 // (op 1073) 

s_mov_b32 s0, s18 // (op 1073) 
s_mul_hi_i32 s47, s0, s12 // (op 1073) 
s_add_i32 s22, s47, s0 // (op 1073) 


s_ashr_i32 s0, s22, 31 // (op 1073) 
s_cmp_eq_i32 s12, 0 // (op 1073) 
s_mov_b32 s47, scc // (op 1073) 
s_cmp_eq_i32 s47, 0 // (op 1073) 
s_cselect_b32 s23, 0, -1 // (op 1073) 

s_and_b32 s47, s44, 31 // (op 1073) 
s_lshl_b32 s24, 1, s47 // (op 1073) 
s_add_i32 s25, s24, s23 // (op 1073) 


s_and_b32 s23, s0, s25 // (op 1073) 


s_add_i32 s0, s22, s23 // (op 1073) 


s_ashr_i32 s22, s0, s47 // (op 1073) 


s_lshl_b32 s0, s44, 24 // (op 1073) 
s_ashr_i32 s47, s0, 31 // (op 1073) 

s_xor_b32 s0, s22, s47 // (op 1073) 

s_sub_i32 s22, s0, s47 // (op 1073) 


s_mul_i32 s0, s46, s11 // (op 1073) 
s_add_u32 s24, s6, 64 // (op 1073) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1073) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 1073) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 1073) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 6 // (op 1073) 

s_sub_i32 s23, s24, s0 // (op 1073) 

s_mul_i32 s47, s22, s23 // (op 1073) 


s_sub_u32 s22, s18, s47 // (op 1073) 


s_add_u32 s18, s22, s0 // (op 1073) 


s_cmp_eq_i32 s19, 0 // (op 1073) 
s_mov_b32 s0, scc // (op 1073) 

s_cmp_eq_i32 s0, 0 // (op 1073) 
s_cselect_b32 s19, s18, s1 // (op 1073) 



v_lshl_add_u32 v12, s19, 6, v20 // (op 1073) 


v_mul_lo_u32 v20, v12, s34 // (op 1073) 
v_add_u32 v20, v20, v21 // (op 1073) 


s_mul_i32 s1, s34, s7 // (op 1075) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 1075) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 1075) least significant
s_mul_hi_u32 s18, s34, s6 // (op 1075) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1075) most significant: sum
s_add_u32 s1, s1, s18 // (op 1075) most significant: sum


s_mov_b32 s64, s36 // (op 1075) reg expression
s_mov_b32 s65, s37 // (op 1075) reg expression
s_mov_b32 s66, s0 // (op 1075) reg expression
s_mov_b32 s67, 131072 // (op 1075) call()

s_mul_hi_i32 s47, s16, s10 // (op 1118) 
v_and_b32 v12, 15, v1 // (op 1118) 
s_add_i32 s0, s47, s16 // (op 1118) 

v_and_b32 v21, 63, v1 // (op 1118) 
s_ashr_i32 s1, s0, 31 // (op 1118) 
v_lshrrev_b32 v22, 4, v21 // (op 1118) 

s_cmp_eq_i32 s10, 0 // (op 1118) 
s_mov_b32 s47, scc // (op 1118) 
s_cmp_eq_i32 s47, 0 // (op 1118) 
s_cselect_b32 s18, 0, -1 // (op 1118) 

s_and_b32 s19, s13, 31 // (op 1118) 
s_lshl_b32 s47, 1, s19 // (op 1118) 
s_add_i32 s22, s47, s18 // (op 1118) 


s_and_b32 s18, s1, s22 // (op 1118) 


s_add_i32 s1, s0, s18 // (op 1118) 


s_ashr_i32 s0, s1, s19 // (op 1118) 


s_lshl_b32 s1, s13, 24 // (op 1118) 
s_ashr_i32 s47, s1, 31 // (op 1118) 

s_xor_b32 s1, s0, s47 // (op 1118) 

s_sub_i32 s0, s1, s47 // (op 1118) 


s_mul_hi_i32 s1, s0, s14 // (op 1118) 
s_add_i32 s47, s1, s0 // (op 1118) 

s_ashr_i32 s1, s47, 31 // (op 1118) 
s_cmp_eq_i32 s14, 0 // (op 1118) 
s_mov_b32 s18, scc // (op 1118) 
s_cmp_eq_i32 s18, 0 // (op 1118) 
s_cselect_b32 s19, 0, -1 // (op 1118) 

s_and_b32 s18, s15, 31 // (op 1118) 
s_lshl_b32 s22, 1, s18 // (op 1118) 
s_add_i32 s23, s22, s19 // (op 1118) 


s_and_b32 s19, s1, s23 // (op 1118) 


s_add_i32 s1, s47, s19 // (op 1118) 


s_ashr_i32 s19, s1, s18 // (op 1118) 


s_lshl_b32 s1, s15, 24 // (op 1118) 
s_ashr_i32 s18, s1, 31 // (op 1118) 

s_xor_b32 s1, s19, s18 // (op 1118) 

s_sub_i32 s19, s1, s18 // (op 1118) 


s_add_u32 s22, s4, 16 // (op 1118) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1118) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1118) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1118) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1118) 

s_mul_i32 s1, s19, s22 // (op 1118) 

s_sub_i32 s47, s0, s1 // (op 1118) 


s_mul_hi_i32 s0, s16, s9 // (op 1118) 
s_add_i32 s1, s0, s16 // (op 1118) 

s_ashr_i32 s0, s1, 31 // (op 1118) 
s_cmp_eq_i32 s9, 0 // (op 1118) 
s_mov_b32 s18, scc // (op 1118) 
s_cmp_eq_i32 s18, 0 // (op 1118) 
s_cselect_b32 s19, 0, -1 // (op 1118) 

s_and_b32 s18, s45, 31 // (op 1118) 
s_lshl_b32 s24, 1, s18 // (op 1118) 
s_add_i32 s25, s24, s19 // (op 1118) 


s_and_b32 s19, s0, s25 // (op 1118) 


s_add_i32 s0, s1, s19 // (op 1118) 


s_ashr_i32 s1, s0, s18 // (op 1118) 


s_lshl_b32 s0, s45, 24 // (op 1118) 
s_ashr_i32 s18, s0, 31 // (op 1118) 

s_xor_b32 s0, s1, s18 // (op 1118) 

s_sub_i32 s1, s0, s18 // (op 1118) 


s_mul_i32 s0, s11, s22 // (op 1118) 

s_mul_i32 s18, s8, s0 // (op 1118) 

s_mul_i32 s0, s1, s18 // (op 1118) 

s_sub_u32 s18, s16, s0 // (op 1118) 

s_mul_hi_i32 s0, s18, s12 // (op 1118) 
s_add_i32 s19, s0, s18 // (op 1118) 


s_ashr_i32 s0, s19, 31 // (op 1118) 
s_cmp_eq_i32 s12, 0 // (op 1118) 
s_mov_b32 s18, scc // (op 1118) 
s_cmp_eq_i32 s18, 0 // (op 1118) 
s_cselect_b32 s22, 0, -1 // (op 1118) 

s_and_b32 s18, s44, 31 // (op 1118) 
s_lshl_b32 s23, 1, s18 // (op 1118) 
s_add_i32 s24, s23, s22 // (op 1118) 


s_and_b32 s25, s0, s24 // (op 1118) 


s_add_i32 s0, s19, s25 // (op 1118) 


s_ashr_i32 s19, s0, s18 // (op 1118) 


s_lshl_b32 s0, s44, 24 // (op 1118) 
s_ashr_i32 s18, s0, 31 // (op 1118) 

s_xor_b32 s0, s19, s18 // (op 1118) 

s_sub_i32 s19, s0, s18 // (op 1118) 


s_cmp_eq_i32 s1, 0 // (op 1118) 
s_mov_b32 s0, scc // (op 1118) 

s_cmp_eq_i32 s0, 0 // (op 1118) 
s_cselect_b32 s1, s19, s47 // (op 1118) 



v_lshl_add_u32 v21, s1, 4, v12 // (op 1118) 


v_mul_lo_u32 v12, v21, s38 // (op 1118) 

v_add_u32 v21, v22, v12 // (op 1118) 


s_mul_i32 s1, s38, s5 // (op 1120) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 1120) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 1120) least significant
s_mul_hi_u32 s18, s38, s4 // (op 1120) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1120) most significant: sum
s_add_u32 s1, s1, s18 // (op 1120) most significant: sum


s_mov_b32 s68, s40 // (op 1120) reg expression
s_mov_b32 s69, s41 // (op 1120) reg expression
s_mov_b32 s70, s0 // (op 1120) reg expression
s_mov_b32 s71, 131072 // (op 1120) call()

s_mul_hi_i32 s47, s16, s10 // (op 1137) 
v_and_b32 v12, 15, v1 // (op 1137) 
s_add_i32 s0, s47, s16 // (op 1137) 

v_and_b32 v22, 63, v1 // (op 1137) 
s_ashr_i32 s1, s0, 31 // (op 1137) 
v_lshrrev_b32 v23, 4, v22 // (op 1137) 

s_cmp_eq_i32 s10, 0 // (op 1137) 
s_mov_b32 s47, scc // (op 1137) 
s_cmp_eq_i32 s47, 0 // (op 1137) 
s_cselect_b32 s18, 0, -1 // (op 1137) 

s_and_b32 s19, s13, 31 // (op 1137) 
s_lshl_b32 s47, 1, s19 // (op 1137) 
s_add_i32 s22, s47, s18 // (op 1137) 


s_and_b32 s18, s1, s22 // (op 1137) 


s_add_i32 s1, s0, s18 // (op 1137) 


s_ashr_i32 s0, s1, s19 // (op 1137) 


s_lshl_b32 s1, s13, 24 // (op 1137) 
s_ashr_i32 s47, s1, 31 // (op 1137) 

s_xor_b32 s1, s0, s47 // (op 1137) 

s_sub_i32 s0, s1, s47 // (op 1137) 


s_mul_hi_i32 s1, s0, s14 // (op 1137) 
s_add_i32 s47, s1, s0 // (op 1137) 

s_ashr_i32 s1, s47, 31 // (op 1137) 
s_cmp_eq_i32 s14, 0 // (op 1137) 
s_mov_b32 s18, scc // (op 1137) 
s_cmp_eq_i32 s18, 0 // (op 1137) 
s_cselect_b32 s19, 0, -1 // (op 1137) 

s_and_b32 s18, s15, 31 // (op 1137) 
s_lshl_b32 s22, 1, s18 // (op 1137) 
s_add_i32 s23, s22, s19 // (op 1137) 


s_and_b32 s19, s1, s23 // (op 1137) 


s_add_i32 s1, s47, s19 // (op 1137) 


s_ashr_i32 s19, s1, s18 // (op 1137) 


s_lshl_b32 s1, s15, 24 // (op 1137) 
s_ashr_i32 s18, s1, 31 // (op 1137) 

s_xor_b32 s1, s19, s18 // (op 1137) 

s_sub_i32 s19, s1, s18 // (op 1137) 


s_add_u32 s22, s4, 16 // (op 1137) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1137) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1137) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1137) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1137) 

s_mul_i32 s1, s19, s22 // (op 1137) 

s_sub_i32 s47, s0, s1 // (op 1137) 


s_mul_hi_i32 s0, s16, s9 // (op 1137) 
s_add_i32 s1, s0, s16 // (op 1137) 

s_ashr_i32 s0, s1, 31 // (op 1137) 
s_cmp_eq_i32 s9, 0 // (op 1137) 
s_mov_b32 s18, scc // (op 1137) 
s_cmp_eq_i32 s18, 0 // (op 1137) 
s_cselect_b32 s19, 0, -1 // (op 1137) 

s_and_b32 s18, s45, 31 // (op 1137) 
s_lshl_b32 s24, 1, s18 // (op 1137) 
s_add_i32 s25, s24, s19 // (op 1137) 


s_and_b32 s19, s0, s25 // (op 1137) 


s_add_i32 s0, s1, s19 // (op 1137) 


s_ashr_i32 s1, s0, s18 // (op 1137) 


s_lshl_b32 s0, s45, 24 // (op 1137) 
s_ashr_i32 s18, s0, 31 // (op 1137) 

s_xor_b32 s0, s1, s18 // (op 1137) 

s_sub_i32 s1, s0, s18 // (op 1137) 


s_mul_i32 s0, s11, s22 // (op 1137) 

s_mul_i32 s18, s8, s0 // (op 1137) 

s_mul_i32 s0, s1, s18 // (op 1137) 

s_sub_u32 s18, s16, s0 // (op 1137) 

s_mul_hi_i32 s0, s18, s12 // (op 1137) 
s_add_i32 s19, s0, s18 // (op 1137) 


s_ashr_i32 s0, s19, 31 // (op 1137) 
s_cmp_eq_i32 s12, 0 // (op 1137) 
s_mov_b32 s18, scc // (op 1137) 
s_cmp_eq_i32 s18, 0 // (op 1137) 
s_cselect_b32 s22, 0, -1 // (op 1137) 

s_and_b32 s18, s44, 31 // (op 1137) 
s_lshl_b32 s23, 1, s18 // (op 1137) 
s_add_i32 s24, s23, s22 // (op 1137) 


s_and_b32 s25, s0, s24 // (op 1137) 


s_add_i32 s0, s19, s25 // (op 1137) 


s_ashr_i32 s19, s0, s18 // (op 1137) 


s_lshl_b32 s0, s44, 24 // (op 1137) 
s_ashr_i32 s18, s0, 31 // (op 1137) 

s_xor_b32 s0, s19, s18 // (op 1137) 

s_sub_i32 s19, s0, s18 // (op 1137) 


s_cmp_eq_i32 s1, 0 // (op 1137) 
s_mov_b32 s0, scc // (op 1137) 

s_cmp_eq_i32 s0, 0 // (op 1137) 
s_cselect_b32 s1, s19, s47 // (op 1137) 



v_lshl_add_u32 v22, s1, 4, v12 // (op 1137) 


v_mul_lo_u32 v12, v22, s38 // (op 1137) 

v_add_u32 v22, v23, v12 // (op 1137) 


s_mul_i32 s1, s38, s5 // (op 1139) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 1139) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 1139) least significant
s_mul_hi_u32 s18, s38, s4 // (op 1139) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1139) most significant: sum
s_add_u32 s1, s1, s18 // (op 1139) most significant: sum


s_mov_b32 s68, s40 // (op 1139) reg expression
s_mov_b32 s69, s41 // (op 1139) reg expression
s_mov_b32 s70, s0 // (op 1139) reg expression
s_mov_b32 s71, 131072 // (op 1139) call()




buffer_load_ubyte v12, v21, s[68:71], 0 offen offset: 0 // (op 774) Load value



buffer_load_dwordx3 v[24:26], v14, s[60:63], 0 offen // (op 778) Load value
v_and_b32 v23, -8, v1 // (op 1203) 
v_sub_u32 v27, v1, v23 // (op 1203) 

v_lshlrev_b32 v23, 4, v27 // (op 1203) 

v_lshrrev_b32 v27, 3, v1 // (op 1203) 
v_lshl_add_u32 v28, v27, 7, v23 // (op 1203) 


s_nop 0
v_readfirstlane_b32 s47, v28 // (op 1203) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 4 // (op 881) 
v_mov_b64 v[28:29], s[0:1] // (op 881) convert


v_mov_b64 v[30:31], v[16:17] // (op 881) 
s_mov_b32 m0, s47 // (op 881) reg expression
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 881) Load value direct to lds
v_add_co_u32 v30, s[0:1], v30, v28 // (op 881) least significant half
v_addc_co_u32 v31, s[0:1], 0, v31, s[0:1] // (op 881) most significant half

s_add_u32 m0, m0, 2048 // (op 881) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 881) Load value direct to lds
v_add_co_u32 v30, s[0:1], v30, v28 // (op 881) least significant half
v_addc_co_u32 v31, s[0:1], 0, v31, s[0:1] // (op 881) most significant half

s_add_u32 m0, m0, 2048 // (op 881) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 881) Load value direct to lds
v_add_co_u32 v30, s[0:1], v30, v28 // (op 881) least significant half
v_addc_co_u32 v31, s[0:1], 0, v31, s[0:1] // (op 881) most significant half

s_add_u32 m0, m0, 2048 // (op 881) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 881) Load value direct to lds




v_mov_b64 v[28:29], s[34:35] // (op 784) convert

v_mov_b32 v23, v13 // (op 784) 
buffer_load_ubyte v30, v23, s[64:67], 0 offen offset: 0 // (op 784) Load value
v_add_u32 v23, v23, v28 // (op 784) 
buffer_load_ubyte v31, v23, s[64:67], 0 offen offset: 0 // (op 784) Load value





buffer_load_ubyte v23, v21, s[68:71], 0 offen offset:4 // (op 788) Load value




buffer_load_dwordx3 v[32:34], v14, s[60:63], 0 offen offset:96 // (op 792) Load value

v_and_b32 v14, -8, v1 // (op 1376) 
v_sub_u32 v21, v1, v14 // (op 1376) 

v_lshlrev_b32 v14, 4, v21 // (op 1376) 

v_lshrrev_b32 v21, 3, v1 // (op 1376) 
v_lshl_add_u32 v27, v21, 7, v14 // (op 1376) 


s_nop 0
v_readfirstlane_b32 s47, v27 // (op 1376) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 4 // (op 886) 
v_mov_b64 v[28:29], s[0:1] // (op 886) convert


v_mov_b64 v[36:37], v[16:17] // (op 886) 
s_add_u32 s0, 8192, s47 // (op 886) 
s_mov_b32 m0, s0 // (op 886) reg expression
s_mov_b32 s1, 128 // (op 886) call()
buffer_load_dwordx4 v36, s[48:51], s1 offen offset: 0 lds // (op 886) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v28 // (op 886) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 886) most significant half

s_add_u32 m0, m0, 2048 // (op 886) 
buffer_load_dwordx4 v36, s[48:51], s1 offen offset: 0 lds // (op 886) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v28 // (op 886) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 886) most significant half

s_add_u32 m0, m0, 2048 // (op 886) 
buffer_load_dwordx4 v36, s[48:51], s1 offen offset: 0 lds // (op 886) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v28 // (op 886) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 886) most significant half

s_add_u32 m0, m0, 2048 // (op 886) 
buffer_load_dwordx4 v36, s[48:51], s1 offen offset: 0 lds // (op 886) Load value direct to lds







v_mov_b64 v[16:17], s[34:35] // (op 798) convert

v_mov_b32 v14, v13 // (op 798) 
buffer_load_ubyte v28, v14, s[64:67], 0 offen offset:4 // (op 798) Load value
v_add_u32 v14, v14, v16 // (op 798) 
buffer_load_ubyte v29, v14, s[64:67], 0 offen offset:4 // (op 798) Load value



v_and_b32 v21, -8, v1 // (op 1261) 
v_sub_u32 v27, v1, v21 // (op 1261) 

v_lshlrev_b32 v21, 4, v27 // (op 1261) 

v_lshrrev_b32 v27, 3, v1 // (op 1261) 
v_lshl_add_u32 v13, v27, 7, v21 // (op 1261) 


v_mul_lo_u32 v14, v13, 6 // (op 1261) 

v_lshrrev_b32 v13, 3, v14 // (op 1261) 




s_waitcnt vmcnt(14)
ds_write_b96 v13, v[24:26] offset:16384 // (op 800) Store local data 

v_and_b32 v21, -4, v1 // (op 1314) 
v_sub_u32 v27, v1, v21 // (op 1314) 

v_lshrrev_b32 v21, 2, v1 // (op 1314) 
v_lshl_add_u32 v13, v21, 3, v27 // (op 1314) 





s_waitcnt vmcnt(9)
ds_write_b8 v13, v30 offset:17920 // (op 802) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v13, v31 offset:17924 // (op 802) Store local data 

v_and_b32 v21, -8, v1 // (op 1175) 
v_sub_u32 v27, v1, v21 // (op 1175) 

v_lshlrev_b32 v21, 4, v27 // (op 1175) 

v_lshrrev_b32 v27, 3, v1 // (op 1175) 
v_lshl_add_u32 v13, v27, 7, v21 // (op 1175) 


s_nop 0
v_readfirstlane_b32 s47, v13 // (op 1175) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -8, v1 // (op 1232) 
v_sub_u32 v27, v1, v21 // (op 1232) 

v_lshlrev_b32 v21, 4, v27 // (op 1232) 

v_lshrrev_b32 v27, 3, v1 // (op 1232) 
v_lshl_add_u32 v13, v27, 7, v21 // (op 1232) 


v_mul_lo_u32 v14, v13, 6 // (op 1232) 

v_lshrrev_b32 v13, 3, v14 // (op 1232) 

v_and_b32 v14, -4, v1 // (op 1285) 
v_sub_u32 v21, v1, v14 // (op 1285) 

v_lshrrev_b32 v14, 2, v1 // (op 1285) 
v_lshl_add_u32 v27, v14, 3, v21 // (op 1285) 


v_and_b32 v14, -8, v1 // (op 1361) 
v_sub_u32 v21, v1, v14 // (op 1361) 

v_lshlrev_b32 v14, 4, v21 // (op 1361) 

v_lshrrev_b32 v21, 3, v1 // (op 1361) 
v_lshl_add_u32 v16, v21, 7, v14 // (op 1361) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 1361) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -8, v1 // (op 1405) 
v_sub_u32 v21, v1, v14 // (op 1405) 

v_lshlrev_b32 v14, 4, v21 // (op 1405) 

v_lshrrev_b32 v21, 3, v1 // (op 1405) 
v_lshl_add_u32 v16, v21, 7, v14 // (op 1405) 


v_mul_lo_u32 v14, v16, 6 // (op 1405) 

v_lshrrev_b32 v21, 3, v14 // (op 1405) 

v_and_b32 v14, -4, v1 // (op 1429) 
v_sub_u32 v16, v1, v14 // (op 1429) 

v_lshrrev_b32 v14, 2, v1 // (op 1429) 
v_lshl_add_u32 v17, v14, 3, v16 // (op 1429) 


s_mov_b32 s1, 0 // (op 311) call()
s_ashr_i64 s[18:19], s[42:43], 7 // (op 310) 
s_mov_b32 s22, s18 // (op 310) convert

s_and_b32 s18, s22, 4294967294 // (op 310) 

s_cmp_lt_u32 s1, s18 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoop_310 // (op 310) Condition: Top (jump to RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoop_310 if false)
RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoop_310:
s_barrier  // (op 1487) 
s_add_u32 s1, s1, 2 // (op 312) 



ds_read_b128 v[36:39], v7 // (op 65) Load local data 
ds_read_b128 v[40:43], v7 offset:64 // (op 65) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1500) 



ds_read_b128 v[44:47], v3 offset:16384 // (op 70) Load local data 
ds_read_b64 v[48:49], v3 offset:16400 // (op 70) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1514) 



ds_read_u8 v14, v0 offset:17920 // (op 75) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:49], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 82) 
ds_read_b128 v[50:53], v7 offset:4096 // (op 340) Load local data 


ds_read_b128 v[54:57], v7 offset:4160 // (op 340) Load local data 



ds_read_u8 v14, v0 offset:18048 // (op 344) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[44:49], a[4:7], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 346) 





v_mov_b32 v14, v22 // (op 80) 

buffer_load_ubyte v12, v14, s[68:71], 0 offen offset:8 // (op 80) Load value


v_mov_b32 v16, v15 // (op 10) 

buffer_load_dwordx3 v[24:26], v16, s[60:63], 0 offen offset:192 // (op 10) Load value
s_barrier  // (op 1490) 

s_lshl_b64 s[18:19], s[26:27], 4 // (op 865) 
v_mov_b64 v[36:37], s[18:19] // (op 865) convert


v_mov_b64 v[38:39], v[18:19] // (op 865) 
v_mov_b64 v[40:41], v[38:39] // (op 865) 
s_mov_b32 m0, s47 // (op 865) reg expression
s_mov_b32 s18, 256 // (op 865) call()
buffer_load_dwordx4 v40, s[48:51], s18 offen offset: 0 lds // (op 865) Load value direct to lds
v_add_co_u32 v40, s[22:23], v40, v36 // (op 865) least significant half
v_addc_co_u32 v41, s[22:23], 0, v41, s[22:23] // (op 865) most significant half

s_add_u32 m0, m0, 2048 // (op 865) 
buffer_load_dwordx4 v40, s[48:51], s18 offen offset: 0 lds // (op 865) Load value direct to lds
v_add_co_u32 v40, s[22:23], v40, v36 // (op 865) least significant half
v_addc_co_u32 v41, s[22:23], 0, v41, s[22:23] // (op 865) most significant half

s_add_u32 m0, m0, 2048 // (op 865) 
buffer_load_dwordx4 v40, s[48:51], s18 offen offset: 0 lds // (op 865) Load value direct to lds
v_add_co_u32 v40, s[22:23], v40, v36 // (op 865) least significant half
v_addc_co_u32 v41, s[22:23], 0, v41, s[22:23] // (op 865) most significant half

s_add_u32 m0, m0, 2048 // (op 865) 
buffer_load_dwordx4 v40, s[48:51], s18 offen offset: 0 lds // (op 865) Load value direct to lds




v_mov_b64 v[36:37], s[34:35] // (op 17) convert

v_mov_b32 v35, v20 // (op 17) 
v_mov_b32 v40, v35 // (op 17) 
buffer_load_ubyte v30, v40, s[64:67], 0 offen offset:8 // (op 17) Load value
v_add_u32 v40, v40, v36 // (op 17) 
buffer_load_ubyte v31, v40, s[64:67], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 1537) 



s_waitcnt vmcnt(14)
ds_write_b96 v21, v[32:34] offset:18176 // (op 235) Store local data 
s_barrier  // (op 1543) 



s_waitcnt vmcnt(9)
ds_write_b8 v17, v28 offset:19712 // (op 236) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v17, v29 offset:19716 // (op 236) Store local data 
s_barrier  // (op 1528) 



ds_read_b128 v[40:43], v11 offset:8192 // (op 227) Load local data 
ds_read_b128 v[44:47], v11 offset:8256 // (op 227) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1534) 



ds_read_b128 v[48:51], v4 offset:18176 // (op 229) Load local data 
ds_read_b64 v[52:53], v4 offset:18192 // (op 229) Load local data 
s_barrier  // (op 1572) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1540) 



ds_read_u8 v36, v10 offset:19712 // (op 231) Load local data 



ds_read_b128 v[54:57], v11 offset:12288 // (op 353) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[48:53], a[0:3], v36, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 234) 
ds_read_b128 v[58:61], v11 offset:12352 // (op 353) Load local data 





ds_read_u8 v36, v10 offset:19840 // (op 357) Load local data 
s_barrier  // (op 1610) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:61], v[48:53], a[4:7], v36, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 347) 






buffer_load_ubyte v23, v14, s[68:71], 0 offen offset:12 // (op 233) Load value




v_add_u32 v22, 8, v22 // (op 1129) 
buffer_load_dwordx3 v[32:34], v16, s[60:63], 0 offen offset:288 // (op 237) Load value

s_barrier  // (op 1531) 
v_add_u32 v15, 192, v15 // (op 1001) 

s_lshl_b64 s[18:19], s[26:27], 4 // (op 876) 
v_mov_b64 v[36:37], s[18:19] // (op 876) convert


v_mov_b64 v[40:41], v[38:39] // (op 876) 
s_add_u32 s18, 8192, s0 // (op 876) 
s_mov_b32 m0, s18 // (op 876) reg expression
s_mov_b32 s19, 384 // (op 876) call()
buffer_load_dwordx4 v40, s[48:51], s19 offen offset: 0 lds // (op 876) Load value direct to lds
v_add_co_u32 v40, s[22:23], v40, v36 // (op 876) least significant half
v_addc_co_u32 v41, s[22:23], 0, v41, s[22:23] // (op 876) most significant half

s_add_u32 m0, m0, 2048 // (op 876) 
buffer_load_dwordx4 v40, s[48:51], s19 offen offset: 0 lds // (op 876) Load value direct to lds
v_add_co_u32 v40, s[22:23], v40, v36 // (op 876) least significant half
v_addc_co_u32 v41, s[22:23], 0, v41, s[22:23] // (op 876) most significant half

s_add_u32 m0, m0, 2048 // (op 876) 
buffer_load_dwordx4 v40, s[48:51], s19 offen offset: 0 lds // (op 876) Load value direct to lds
v_add_co_u32 v40, s[22:23], v40, v36 // (op 876) least significant half
v_addc_co_u32 v41, s[22:23], 0, v41, s[22:23] // (op 876) most significant half

s_add_u32 m0, m0, 2048 // (op 876) 
buffer_load_dwordx4 v40, s[48:51], s19 offen offset: 0 lds // (op 876) Load value direct to lds






v_add_co_u32 v18, vcc, 256, v18 // (op 937) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 937) most significant half
v_mov_b64 v[36:37], s[34:35] // (op 238) convert

v_mov_b32 v14, v35 // (op 238) 
buffer_load_ubyte v28, v14, s[64:67], 0 offen offset:12 // (op 238) Load value
v_add_u32 v14, v14, v36 // (op 238) 
buffer_load_ubyte v29, v14, s[64:67], 0 offen offset:12 // (op 238) Load value



s_barrier  // (op 1503) 
v_add_u32 v20, 8, v20 // (op 1065) 



s_waitcnt vmcnt(14)
ds_write_b96 v13, v[24:26] offset:16384 // (op 13) Store local data 
s_barrier  // (op 1517) 



s_waitcnt vmcnt(9)
ds_write_b8 v27, v30 offset:17920 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v27, v31 offset:17924 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[42:43], 7 // (op 310) 
s_mov_b32 s22, s18 // (op 310) convert

s_and_b32 s18, s22, 4294967294 // (op 310) 

s_cmp_lt_u32 s1, s18 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoop_310 // (op 310) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoop_310 if true)
RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoop_310:














s_barrier  // (op 1592) 
s_mul_hi_i32 s47, s16, s10 // (op 899) 
v_lshrrev_b32 v12, 3, v1 // (op 899) 
s_add_i32 s0, s47, s16 // (op 899) 

v_and_b32 v13, -8, v1 // (op 899) 
s_ashr_i32 s1, s0, 31 // (op 899) 
v_sub_u32 v14, v1, v13 // (op 899) 

s_cmp_eq_i32 s10, 0 // (op 899) 
s_mov_b32 s47, scc // (op 899) 
v_lshlrev_b32 v13, 4, v14 // (op 899) 

s_cmp_eq_i32 s47, 0 // (op 899) 
s_cselect_b32 s18, 0, -1 // (op 899) 

s_and_b32 s19, s13, 31 // (op 899) 
s_lshl_b32 s47, 1, s19 // (op 899) 
s_add_i32 s22, s47, s18 // (op 899) 


s_and_b32 s18, s1, s22 // (op 899) 


s_add_i32 s1, s0, s18 // (op 899) 


s_ashr_i32 s0, s1, s19 // (op 899) 


s_lshl_b32 s1, s13, 24 // (op 899) 
s_ashr_i32 s47, s1, 31 // (op 899) 

s_xor_b32 s1, s0, s47 // (op 899) 

s_sub_i32 s0, s1, s47 // (op 899) 


s_mul_hi_i32 s1, s0, s14 // (op 899) 
s_add_i32 s47, s1, s0 // (op 899) 

s_ashr_i32 s1, s47, 31 // (op 899) 
s_cmp_eq_i32 s14, 0 // (op 899) 
s_mov_b32 s18, scc // (op 899) 
s_cmp_eq_i32 s18, 0 // (op 899) 
s_cselect_b32 s19, 0, -1 // (op 899) 

s_and_b32 s18, s15, 31 // (op 899) 
s_lshl_b32 s22, 1, s18 // (op 899) 
s_add_i32 s23, s22, s19 // (op 899) 


s_and_b32 s19, s1, s23 // (op 899) 


s_add_i32 s1, s47, s19 // (op 899) 


s_ashr_i32 s19, s1, s18 // (op 899) 


s_lshl_b32 s1, s15, 24 // (op 899) 
s_ashr_i32 s18, s1, 31 // (op 899) 

s_xor_b32 s1, s19, s18 // (op 899) 

s_sub_i32 s19, s1, s18 // (op 899) 


s_mul_i32 s1, s0, s11 // (op 899) 

s_sub_u32 s0, s16, s1 // (op 899) 

s_mul_i32 s1, s19, s11 // (op 899) 
s_add_u32 s1, s1, s0 // (op 899) 


s_mul_hi_i32 s0, s16, s9 // (op 899) 
s_add_i32 s47, s0, s16 // (op 899) 

s_ashr_i32 s0, s47, 31 // (op 899) 
s_cmp_eq_i32 s9, 0 // (op 899) 
s_mov_b32 s18, scc // (op 899) 
s_cmp_eq_i32 s18, 0 // (op 899) 
s_cselect_b32 s19, 0, -1 // (op 899) 

s_and_b32 s18, s45, 31 // (op 899) 
s_lshl_b32 s22, 1, s18 // (op 899) 
s_add_i32 s23, s22, s19 // (op 899) 


s_and_b32 s19, s0, s23 // (op 899) 


s_add_i32 s0, s47, s19 // (op 899) 


s_ashr_i32 s19, s0, s18 // (op 899) 


s_lshl_b32 s0, s45, 24 // (op 899) 
s_ashr_i32 s18, s0, 31 // (op 899) 

s_xor_b32 s0, s19, s18 // (op 899) 

s_sub_i32 s19, s0, s18 // (op 899) 


s_add_u32 s22, s4, 16 // (op 899) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 899) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 899) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 899) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 899) 

s_mul_i32 s0, s11, s22 // (op 899) 

s_mul_i32 s18, s8, s0 // (op 899) 

s_mul_i32 s0, s19, s18 // (op 899) 

s_sub_u32 s18, s16, s0 // (op 899) 

s_mov_b32 s0, s18 // (op 899) 
s_mul_hi_i32 s47, s0, s12 // (op 899) 
s_add_i32 s22, s47, s0 // (op 899) 


s_ashr_i32 s0, s22, 31 // (op 899) 
s_cmp_eq_i32 s12, 0 // (op 899) 
s_mov_b32 s47, scc // (op 899) 
s_cmp_eq_i32 s47, 0 // (op 899) 
s_cselect_b32 s23, 0, -1 // (op 899) 

s_and_b32 s47, s44, 31 // (op 899) 
s_lshl_b32 s24, 1, s47 // (op 899) 
s_add_i32 s25, s24, s23 // (op 899) 


s_and_b32 s23, s0, s25 // (op 899) 


s_add_i32 s0, s22, s23 // (op 899) 


s_ashr_i32 s22, s0, s47 // (op 899) 


s_lshl_b32 s0, s44, 24 // (op 899) 
s_ashr_i32 s47, s0, 31 // (op 899) 

s_xor_b32 s0, s22, s47 // (op 899) 

s_sub_i32 s22, s0, s47 // (op 899) 


s_mul_i32 s0, s46, s11 // (op 899) 
s_add_u32 s24, s6, 64 // (op 899) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 899) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 899) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 899) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 899) 

s_sub_i32 s23, s24, s0 // (op 899) 

s_mul_i32 s47, s22, s23 // (op 899) 


s_sub_u32 s22, s18, s47 // (op 899) 


s_add_u32 s18, s22, s0 // (op 899) 


s_cmp_eq_i32 s19, 0 // (op 899) 
s_mov_b32 s0, scc // (op 899) 

s_cmp_eq_i32 s0, 0 // (op 899) 
s_cselect_b32 s19, s18, s1 // (op 899) 



v_lshl_add_u32 v14, s19, 6, v12 // (op 899) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 899) 
s_lshr_b32 s47, s0, 1 // (op 899) 

v_lshl_add_u32 v12, s47, 8, v13 // (op 899) 


v_mov_b32 v31, 0 // (op 899) convert
v_mov_b32 v30, v12 // (op 899) convert

v_mul_lo_u32 v13, v14, s27 // (op 899) most significant: low of low * high
v_mul_hi_u32 v15, v14, s26 // (op 899) most significant: high of low * low
v_mul_lo_u32 v12, v14, s26 // (op 899) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 899) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v30 // (op 899) least significant half
v_addc_co_u32 v13, s[0:1], v13, v31, s[0:1] // (op 899) most significant half



s_mul_i32 s1, s26, s7 // (op 901) most significant: low of low * high
s_mul_i32 s47, s27, s6 // (op 901) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 901) least significant
s_mul_hi_u32 s18, s26, s6 // (op 901) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 901) most significant: sum
s_add_u32 s1, s1, s18 // (op 901) most significant: sum


s_mov_b32 s48, s28 // (op 901) reg expression
s_mov_b32 s49, s29 // (op 901) reg expression
s_mov_b32 s50, s0 // (op 901) reg expression
s_mov_b32 s51, 131072 // (op 901) call()

s_mul_hi_i32 s47, s16, s10 // (op 967) 
v_lshrrev_b32 v14, 3, v1 // (op 967) 
s_add_i32 s0, s47, s16 // (op 967) 

v_and_b32 v15, -8, v1 // (op 967) 
s_ashr_i32 s1, s0, 31 // (op 967) 
v_sub_u32 v16, v1, v15 // (op 967) 

s_cmp_eq_i32 s10, 0 // (op 967) 
s_mov_b32 s47, scc // (op 967) 
v_lshlrev_b32 v15, 4, v16 // (op 967) 

s_cmp_eq_i32 s47, 0 // (op 967) 
s_cselect_b32 s18, 0, -1 // (op 967) 

s_and_b32 s19, s13, 31 // (op 967) 
s_lshl_b32 s47, 1, s19 // (op 967) 
s_add_i32 s22, s47, s18 // (op 967) 


s_and_b32 s18, s1, s22 // (op 967) 


s_add_i32 s1, s0, s18 // (op 967) 


s_ashr_i32 s0, s1, s19 // (op 967) 


s_lshl_b32 s1, s13, 24 // (op 967) 
s_ashr_i32 s47, s1, 31 // (op 967) 

s_xor_b32 s1, s0, s47 // (op 967) 

s_sub_i32 s0, s1, s47 // (op 967) 


s_mul_hi_i32 s1, s0, s14 // (op 967) 
s_add_i32 s47, s1, s0 // (op 967) 

s_ashr_i32 s1, s47, 31 // (op 967) 
s_cmp_eq_i32 s14, 0 // (op 967) 
s_mov_b32 s18, scc // (op 967) 
s_cmp_eq_i32 s18, 0 // (op 967) 
s_cselect_b32 s19, 0, -1 // (op 967) 

s_and_b32 s18, s15, 31 // (op 967) 
s_lshl_b32 s22, 1, s18 // (op 967) 
s_add_i32 s23, s22, s19 // (op 967) 


s_and_b32 s19, s1, s23 // (op 967) 


s_add_i32 s1, s47, s19 // (op 967) 


s_ashr_i32 s19, s1, s18 // (op 967) 


s_lshl_b32 s1, s15, 24 // (op 967) 
s_ashr_i32 s18, s1, 31 // (op 967) 

s_xor_b32 s1, s19, s18 // (op 967) 

s_sub_i32 s19, s1, s18 // (op 967) 


s_add_u32 s22, s4, 16 // (op 967) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 967) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 967) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 967) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 967) 

s_mul_i32 s1, s19, s22 // (op 967) 

s_sub_i32 s47, s0, s1 // (op 967) 


s_mul_hi_i32 s0, s16, s9 // (op 967) 
s_add_i32 s1, s0, s16 // (op 967) 

s_ashr_i32 s0, s1, 31 // (op 967) 
s_cmp_eq_i32 s9, 0 // (op 967) 
s_mov_b32 s18, scc // (op 967) 
s_cmp_eq_i32 s18, 0 // (op 967) 
s_cselect_b32 s19, 0, -1 // (op 967) 

s_and_b32 s18, s45, 31 // (op 967) 
s_lshl_b32 s24, 1, s18 // (op 967) 
s_add_i32 s25, s24, s19 // (op 967) 


s_and_b32 s19, s0, s25 // (op 967) 


s_add_i32 s0, s1, s19 // (op 967) 


s_ashr_i32 s1, s0, s18 // (op 967) 


s_lshl_b32 s0, s45, 24 // (op 967) 
s_ashr_i32 s18, s0, 31 // (op 967) 

s_xor_b32 s0, s1, s18 // (op 967) 

s_sub_i32 s1, s0, s18 // (op 967) 


s_mul_i32 s0, s11, s22 // (op 967) 

s_mul_i32 s18, s8, s0 // (op 967) 

s_mul_i32 s0, s1, s18 // (op 967) 

s_sub_u32 s18, s16, s0 // (op 967) 

s_mul_hi_i32 s0, s18, s12 // (op 967) 
s_add_i32 s19, s0, s18 // (op 967) 


s_ashr_i32 s0, s19, 31 // (op 967) 
s_cmp_eq_i32 s12, 0 // (op 967) 
s_mov_b32 s18, scc // (op 967) 
s_cmp_eq_i32 s18, 0 // (op 967) 
s_cselect_b32 s22, 0, -1 // (op 967) 

s_and_b32 s18, s44, 31 // (op 967) 
s_lshl_b32 s23, 1, s18 // (op 967) 
s_add_i32 s24, s23, s22 // (op 967) 


s_and_b32 s25, s0, s24 // (op 967) 


s_add_i32 s0, s19, s25 // (op 967) 


s_ashr_i32 s19, s0, s18 // (op 967) 


s_lshl_b32 s0, s44, 24 // (op 967) 
s_ashr_i32 s18, s0, 31 // (op 967) 

s_xor_b32 s0, s19, s18 // (op 967) 

s_sub_i32 s19, s0, s18 // (op 967) 


s_cmp_eq_i32 s1, 0 // (op 967) 
s_mov_b32 s0, scc // (op 967) 

s_cmp_eq_i32 s0, 0 // (op 967) 
s_cselect_b32 s1, s19, s47 // (op 967) 



v_lshl_add_u32 v16, s1, 4, v14 // (op 967) 


v_mul_lo_u32 v31, v16, s31 // (op 967) most significant: low of low * high
v_mul_hi_u32 v14, v16, s30 // (op 967) most significant: high of low * low
v_mul_lo_u32 v30, v16, s30 // (op 967) least significant: low of low * low
v_add_u32 v31, v31, v14 // (op 967) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 967) 
s_lshr_b32 s47, s0, 1 // (op 967) 

v_lshl_add_u32 v14, s47, 8, v15 // (op 967) 


v_mov_b32 v25, 0 // (op 967) convert
v_mov_b32 v24, v14 // (op 967) convert

v_add_co_u32 v26, s[0:1], v24, v30 // (op 967) least significant half
v_addc_co_u32 v27, s[0:1], v25, v31, s[0:1] // (op 967) most significant half



v_mul_lo_u32 v25, v27, 6 // (op 967) most significant: low of high * low
v_mul_hi_u32 v14, v26, 6 // (op 967) most significant: high of low * low
v_mul_lo_u32 v24, v26, 6 // (op 967) least significant: low of low * low
v_add_u32 v25, v25, v14 // (op 967) most significant: sum


v_lshrrev_b64 v[26:27], 3, v[24:25] // (op 967) 

v_mov_b32 v14, v26 // (op 967) convert

s_mul_i32 s1, s5, 6 // (op 969) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 969) least significant
s_mul_hi_u32 s47, s4, 6 // (op 969) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 969) most significant: sum

s_mul_i32 s19, s30, s1 // (op 969) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 969) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 969) least significant
s_mul_hi_u32 s22, s30, s0 // (op 969) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 969) most significant: sum
s_add_u32 s19, s19, s22 // (op 969) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 969) 

s_mov_b32 s60, s32 // (op 969) reg expression
s_mov_b32 s61, s33 // (op 969) reg expression
s_mov_b32 s62, s0 // (op 969) reg expression
s_mov_b32 s63, 131072 // (op 969) call()

s_mul_hi_i32 s47, s16, s10 // (op 1031) 
v_lshrrev_b32 v15, 2, v1 // (op 1031) 
s_add_i32 s0, s47, s16 // (op 1031) 

v_lshlrev_b32 v16, 1, v15 // (op 1031) 

s_ashr_i32 s1, s0, 31 // (op 1031) 
v_and_b32 v15, -4, v1 // (op 1031) 
s_cmp_eq_i32 s10, 0 // (op 1031) 
s_mov_b32 s47, scc // (op 1031) 
v_sub_u32 v17, v1, v15 // (op 1031) 

s_cmp_eq_i32 s47, 0 // (op 1031) 
s_cselect_b32 s18, 0, -1 // (op 1031) 

s_and_b32 s19, s13, 31 // (op 1031) 
s_lshl_b32 s47, 1, s19 // (op 1031) 
s_add_i32 s22, s47, s18 // (op 1031) 


s_and_b32 s18, s1, s22 // (op 1031) 


s_add_i32 s1, s0, s18 // (op 1031) 


s_ashr_i32 s0, s1, s19 // (op 1031) 


s_lshl_b32 s1, s13, 24 // (op 1031) 
s_ashr_i32 s47, s1, 31 // (op 1031) 

s_xor_b32 s1, s0, s47 // (op 1031) 

s_sub_i32 s0, s1, s47 // (op 1031) 


s_mul_hi_i32 s1, s0, s14 // (op 1031) 
s_add_i32 s47, s1, s0 // (op 1031) 

s_ashr_i32 s1, s47, 31 // (op 1031) 
s_cmp_eq_i32 s14, 0 // (op 1031) 
s_mov_b32 s18, scc // (op 1031) 
s_cmp_eq_i32 s18, 0 // (op 1031) 
s_cselect_b32 s19, 0, -1 // (op 1031) 

s_and_b32 s18, s15, 31 // (op 1031) 
s_lshl_b32 s22, 1, s18 // (op 1031) 
s_add_i32 s23, s22, s19 // (op 1031) 


s_and_b32 s19, s1, s23 // (op 1031) 


s_add_i32 s1, s47, s19 // (op 1031) 


s_ashr_i32 s19, s1, s18 // (op 1031) 


s_lshl_b32 s1, s15, 24 // (op 1031) 
s_ashr_i32 s18, s1, 31 // (op 1031) 

s_xor_b32 s1, s19, s18 // (op 1031) 

s_sub_i32 s19, s1, s18 // (op 1031) 


s_mul_i32 s1, s0, s11 // (op 1031) 

s_sub_u32 s0, s16, s1 // (op 1031) 

s_mul_i32 s1, s19, s11 // (op 1031) 
s_add_u32 s1, s1, s0 // (op 1031) 


s_mul_hi_i32 s0, s16, s9 // (op 1031) 
s_add_i32 s47, s0, s16 // (op 1031) 

s_ashr_i32 s0, s47, 31 // (op 1031) 
s_cmp_eq_i32 s9, 0 // (op 1031) 
s_mov_b32 s18, scc // (op 1031) 
s_cmp_eq_i32 s18, 0 // (op 1031) 
s_cselect_b32 s19, 0, -1 // (op 1031) 

s_and_b32 s18, s45, 31 // (op 1031) 
s_lshl_b32 s22, 1, s18 // (op 1031) 
s_add_i32 s23, s22, s19 // (op 1031) 


s_and_b32 s19, s0, s23 // (op 1031) 


s_add_i32 s0, s47, s19 // (op 1031) 


s_ashr_i32 s19, s0, s18 // (op 1031) 


s_lshl_b32 s0, s45, 24 // (op 1031) 
s_ashr_i32 s18, s0, 31 // (op 1031) 

s_xor_b32 s0, s19, s18 // (op 1031) 

s_sub_i32 s19, s0, s18 // (op 1031) 


s_add_u32 s22, s4, 16 // (op 1031) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1031) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1031) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1031) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1031) 

s_mul_i32 s0, s11, s22 // (op 1031) 

s_mul_i32 s18, s8, s0 // (op 1031) 

s_mul_i32 s0, s19, s18 // (op 1031) 

s_sub_u32 s18, s16, s0 // (op 1031) 

s_mov_b32 s0, s18 // (op 1031) 
s_mul_hi_i32 s47, s0, s12 // (op 1031) 
s_add_i32 s22, s47, s0 // (op 1031) 


s_ashr_i32 s0, s22, 31 // (op 1031) 
s_cmp_eq_i32 s12, 0 // (op 1031) 
s_mov_b32 s47, scc // (op 1031) 
s_cmp_eq_i32 s47, 0 // (op 1031) 
s_cselect_b32 s23, 0, -1 // (op 1031) 

s_and_b32 s47, s44, 31 // (op 1031) 
s_lshl_b32 s24, 1, s47 // (op 1031) 
s_add_i32 s25, s24, s23 // (op 1031) 


s_and_b32 s23, s0, s25 // (op 1031) 


s_add_i32 s0, s22, s23 // (op 1031) 


s_ashr_i32 s22, s0, s47 // (op 1031) 


s_lshl_b32 s0, s44, 24 // (op 1031) 
s_ashr_i32 s47, s0, 31 // (op 1031) 

s_xor_b32 s0, s22, s47 // (op 1031) 

s_sub_i32 s22, s0, s47 // (op 1031) 


s_mul_i32 s0, s46, s11 // (op 1031) 
s_add_u32 s24, s6, 64 // (op 1031) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1031) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 1031) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 1031) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 1031) 

s_sub_i32 s23, s24, s0 // (op 1031) 

s_mul_i32 s47, s22, s23 // (op 1031) 


s_sub_u32 s22, s18, s47 // (op 1031) 


s_add_u32 s18, s22, s0 // (op 1031) 


s_cmp_eq_i32 s19, 0 // (op 1031) 
s_mov_b32 s0, scc // (op 1031) 

s_cmp_eq_i32 s0, 0 // (op 1031) 
s_cselect_b32 s19, s18, s1 // (op 1031) 



v_lshl_add_u32 v15, s19, 6, v16 // (op 1031) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 1031) 
s_lshr_b32 s47, s0, 1 // (op 1031) 

v_lshl_add_u32 v16, s47, 3, v17 // (op 1031) 


v_mul_lo_u32 v17, v15, s34 // (op 1031) 
v_add_u32 v17, v17, v16 // (op 1031) 


s_mul_i32 s1, s34, s7 // (op 1033) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 1033) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 1033) least significant
s_mul_hi_u32 s18, s34, s6 // (op 1033) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1033) most significant: sum
s_add_u32 s1, s1, s18 // (op 1033) most significant: sum


s_mov_b32 s64, s36 // (op 1033) reg expression
s_mov_b32 s65, s37 // (op 1033) reg expression
s_mov_b32 s66, s0 // (op 1033) reg expression
s_mov_b32 s67, 131072 // (op 1033) call()

s_mul_hi_i32 s47, s16, s10 // (op 1095) 
v_and_b32 v15, 15, v1 // (op 1095) 
s_add_i32 s0, s47, s16 // (op 1095) 

v_and_b32 v16, 63, v1 // (op 1095) 
s_ashr_i32 s1, s0, 31 // (op 1095) 
v_lshrrev_b32 v18, 4, v16 // (op 1095) 

s_cmp_eq_i32 s10, 0 // (op 1095) 
s_mov_b32 s47, scc // (op 1095) 
s_cmp_eq_i32 s47, 0 // (op 1095) 
s_cselect_b32 s18, 0, -1 // (op 1095) 

s_and_b32 s19, s13, 31 // (op 1095) 
s_lshl_b32 s47, 1, s19 // (op 1095) 
s_add_i32 s22, s47, s18 // (op 1095) 


s_and_b32 s18, s1, s22 // (op 1095) 


s_add_i32 s1, s0, s18 // (op 1095) 


s_ashr_i32 s0, s1, s19 // (op 1095) 


s_lshl_b32 s1, s13, 24 // (op 1095) 
s_ashr_i32 s47, s1, 31 // (op 1095) 

s_xor_b32 s1, s0, s47 // (op 1095) 

s_sub_i32 s0, s1, s47 // (op 1095) 


s_mul_hi_i32 s1, s0, s14 // (op 1095) 
s_add_i32 s47, s1, s0 // (op 1095) 

s_ashr_i32 s1, s47, 31 // (op 1095) 
s_cmp_eq_i32 s14, 0 // (op 1095) 
s_mov_b32 s18, scc // (op 1095) 
s_cmp_eq_i32 s18, 0 // (op 1095) 
s_cselect_b32 s19, 0, -1 // (op 1095) 

s_and_b32 s18, s15, 31 // (op 1095) 
s_lshl_b32 s22, 1, s18 // (op 1095) 
s_add_i32 s23, s22, s19 // (op 1095) 


s_and_b32 s19, s1, s23 // (op 1095) 


s_add_i32 s1, s47, s19 // (op 1095) 


s_ashr_i32 s19, s1, s18 // (op 1095) 


s_lshl_b32 s1, s15, 24 // (op 1095) 
s_ashr_i32 s18, s1, 31 // (op 1095) 

s_xor_b32 s1, s19, s18 // (op 1095) 

s_sub_i32 s19, s1, s18 // (op 1095) 


s_add_u32 s22, s4, 16 // (op 1095) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1095) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1095) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1095) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1095) 

s_mul_i32 s1, s19, s22 // (op 1095) 

s_sub_i32 s47, s0, s1 // (op 1095) 


s_mul_hi_i32 s0, s16, s9 // (op 1095) 
s_add_i32 s1, s0, s16 // (op 1095) 

s_ashr_i32 s0, s1, 31 // (op 1095) 
s_cmp_eq_i32 s9, 0 // (op 1095) 
s_mov_b32 s18, scc // (op 1095) 
s_cmp_eq_i32 s18, 0 // (op 1095) 
s_cselect_b32 s19, 0, -1 // (op 1095) 

s_and_b32 s18, s45, 31 // (op 1095) 
s_lshl_b32 s24, 1, s18 // (op 1095) 
s_add_i32 s25, s24, s19 // (op 1095) 


s_and_b32 s19, s0, s25 // (op 1095) 


s_add_i32 s0, s1, s19 // (op 1095) 


s_ashr_i32 s1, s0, s18 // (op 1095) 


s_lshl_b32 s0, s45, 24 // (op 1095) 
s_ashr_i32 s18, s0, 31 // (op 1095) 

s_xor_b32 s0, s1, s18 // (op 1095) 

s_sub_i32 s1, s0, s18 // (op 1095) 


s_mul_i32 s0, s11, s22 // (op 1095) 

s_mul_i32 s18, s8, s0 // (op 1095) 

s_mul_i32 s0, s1, s18 // (op 1095) 

s_sub_u32 s18, s16, s0 // (op 1095) 

s_mul_hi_i32 s0, s18, s12 // (op 1095) 
s_add_i32 s19, s0, s18 // (op 1095) 


s_ashr_i32 s0, s19, 31 // (op 1095) 
s_cmp_eq_i32 s12, 0 // (op 1095) 
s_mov_b32 s18, scc // (op 1095) 
s_cmp_eq_i32 s18, 0 // (op 1095) 
s_cselect_b32 s22, 0, -1 // (op 1095) 

s_and_b32 s18, s44, 31 // (op 1095) 
s_lshl_b32 s23, 1, s18 // (op 1095) 
s_add_i32 s24, s23, s22 // (op 1095) 


s_and_b32 s25, s0, s24 // (op 1095) 


s_add_i32 s0, s19, s25 // (op 1095) 


s_ashr_i32 s19, s0, s18 // (op 1095) 


s_lshl_b32 s0, s44, 24 // (op 1095) 
s_ashr_i32 s18, s0, 31 // (op 1095) 

s_xor_b32 s0, s19, s18 // (op 1095) 

s_sub_i32 s19, s0, s18 // (op 1095) 


s_cmp_eq_i32 s1, 0 // (op 1095) 
s_mov_b32 s0, scc // (op 1095) 

s_cmp_eq_i32 s0, 0 // (op 1095) 
s_cselect_b32 s1, s19, s47 // (op 1095) 



v_lshl_add_u32 v16, s1, 4, v15 // (op 1095) 


v_mul_lo_u32 v15, v16, s38 // (op 1095) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 1095) 
s_lshr_b32 s47, s0, 1 // (op 1095) 

v_lshl_add_u32 v16, s47, 3, v18 // (op 1095) 


v_add_u32 v18, v16, v15 // (op 1095) 


s_mul_i32 s1, s38, s5 // (op 1097) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 1097) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 1097) least significant
s_mul_hi_u32 s18, s38, s4 // (op 1097) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1097) most significant: sum
s_add_u32 s1, s1, s18 // (op 1097) most significant: sum


s_mov_b32 s68, s40 // (op 1097) reg expression
s_mov_b32 s69, s41 // (op 1097) reg expression
s_mov_b32 s70, s0 // (op 1097) reg expression
s_mov_b32 s71, 131072 // (op 1097) call()

v_and_b32 v15, -8, v1 // (op 1188) 
v_sub_u32 v16, v1, v15 // (op 1188) 

v_lshlrev_b32 v15, 4, v16 // (op 1188) 

v_lshrrev_b32 v16, 3, v1 // (op 1188) 
v_lshl_add_u32 v19, v16, 7, v15 // (op 1188) 


s_nop 0
v_readfirstlane_b32 s47, v19 // (op 1188) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -8, v1 // (op 1245) 
v_sub_u32 v16, v1, v15 // (op 1245) 

v_lshlrev_b32 v15, 4, v16 // (op 1245) 

v_lshrrev_b32 v16, 3, v1 // (op 1245) 
v_lshl_add_u32 v19, v16, 7, v15 // (op 1245) 


v_mul_lo_u32 v15, v19, 6 // (op 1245) 

v_lshrrev_b32 v16, 3, v15 // (op 1245) 

v_and_b32 v15, -4, v1 // (op 1298) 
v_sub_u32 v19, v1, v15 // (op 1298) 

v_lshrrev_b32 v15, 2, v1 // (op 1298) 
v_lshl_add_u32 v20, v15, 3, v19 // (op 1298) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 316) 
s_and_b32 s18, s0, 4294967294 // (op 316) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 315) 
s_mov_b32 s19, s0 // (op 315) convert

s_cmp_lt_u32 s18, s19 // (op 315) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_315 // (op 315) Condition: Top (jump to RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_315 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoopTail_315:
s_barrier  // (op 1493) 
s_add_u32 s18, s18, 1 // (op 317) 

s_lshl_b64 s[0:1], s[26:27], 4 // (op 870) 
v_mov_b64 v[22:23], s[0:1] // (op 870) convert


v_mov_b64 v[24:25], v[12:13] // (op 870) 
v_mov_b64 v[26:27], v[24:25] // (op 870) 
s_mov_b32 m0, s47 // (op 870) reg expression
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 870) Load value direct to lds
v_add_co_u32 v26, s[0:1], v26, v22 // (op 870) least significant half
v_addc_co_u32 v27, s[0:1], 0, v27, s[0:1] // (op 870) most significant half

s_add_u32 m0, m0, 2048 // (op 870) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 870) Load value direct to lds
v_add_co_u32 v26, s[0:1], v26, v22 // (op 870) least significant half
v_addc_co_u32 v27, s[0:1], 0, v27, s[0:1] // (op 870) most significant half

s_add_u32 m0, m0, 2048 // (op 870) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 870) Load value direct to lds
v_add_co_u32 v26, s[0:1], v26, v22 // (op 870) least significant half
v_addc_co_u32 v27, s[0:1], 0, v27, s[0:1] // (op 870) most significant half

s_add_u32 m0, m0, 2048 // (op 870) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 870) Load value direct to lds



v_add_co_u32 v12, vcc, 128, v12 // (op 889) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 889) most significant half


v_mov_b32 v15, v14 // (op 179) 

buffer_load_dwordx3 v[22:24], v15, s[60:63], 0 offen // (op 179) Load value

v_add_u32 v14, 96, v14 // (op 959) 

s_barrier  // (op 1506) 
v_mov_b64 v[26:27], s[34:35] // (op 180) convert




v_mov_b32 v15, v17 // (op 180) 
v_mov_b32 v19, v15 // (op 180) 
buffer_load_ubyte v28, v19, s[64:67], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v19, v19, v26 // (op 180) 
s_waitcnt vmcnt(1)
ds_write_b96 v16, v[22:24] offset:16384 // (op 177) Store local data 
buffer_load_ubyte v29, v19, s[64:67], 0 offen offset: 0 // (op 180) Load value




v_add_u32 v17, 4, v17 // (op 1023) 
s_barrier  // (op 1496) 
s_barrier  // (op 1520) 






ds_read_b128 v[30:33], v7 // (op 169) Load local data 
ds_read_b128 v[34:37], v7 offset:64 // (op 169) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v20, v28 offset:17920 // (op 178) Store local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1510) 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v29 offset:17924 // (op 178) Store local data 




ds_read_b128 v[24:27], v3 offset:16384 // (op 171) Load local data 
ds_read_b64 v[28:29], v3 offset:16400 // (op 171) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1524) 



ds_read_u8 v15, v0 offset:17920 // (op 173) Load local data 


v_mov_b32 v19, v18 // (op 175) 

buffer_load_ubyte v21, v19, s[68:71], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v18, 4, v18 // (op 1087) 



ds_read_b128 v[38:41], v7 offset:4096 // (op 321) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[24:29], a[0:3], v15, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 176) 
ds_read_b128 v[42:45], v7 offset:4160 // (op 321) Load local data 





ds_read_u8 v15, v0 offset:18048 // (op 325) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:45], v[24:29], a[4:7], v15, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 327) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 315) 
s_mov_b32 s19, s0 // (op 315) convert

s_cmp_lt_u32 s18, s19 // (op 315) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoopTail_315 // (op 315) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoopTail_315 if true)
RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_315:





















s_barrier  // (op 1596) 



















v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalFalse_0_beta_123
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v9, v10 // (op 46) 
v_mul_f32 v15, v9, v11 // (op 46) 
v_mul_f32 v16, v9, v12 // (op 46) 
v_mul_f32 v17, v9, v13 // (op 46) 




buffer_store_dwordx4 v[14:17], v5, s[56:59], 0 offen // (op 52) Store value

v_accvgpr_read v10, a4 // (op 513) 
v_accvgpr_read v11, a5 // (op 513) 
v_accvgpr_read v12, a6 // (op 513) 
v_accvgpr_read v13, a7 // (op 513) 
v_mul_f32 v14, v9, v10 // (op 513) 
v_mul_f32 v15, v9, v11 // (op 513) 
v_mul_f32 v16, v9, v12 // (op 513) 
v_mul_f32 v17, v9, v13 // (op 513) 




buffer_store_dwordx4 v[14:17], v5, s[56:59], 0 offen offset:128 // (op 515) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalFalse_0_beta_123:



buffer_load_dwordx4 v[10:13], v6, s[52:55], 0 offen offset:128 // (op 543) Load value
v_accvgpr_read v14, a4 // (op 540) 
v_accvgpr_read v15, a5 // (op 540) 
v_accvgpr_read v16, a6 // (op 540) 
v_accvgpr_read v17, a7 // (op 540) 
v_mul_f32 v18, v9, v14 // (op 540) 
v_mul_f32 v19, v9, v15 // (op 540) 
v_mul_f32 v20, v9, v16 // (op 540) 
v_mul_f32 v21, v9, v17 // (op 540) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 544) 
v_mul_f32 v15, v8, v11 // (op 544) 
v_mul_f32 v16, v8, v12 // (op 544) 
v_mul_f32 v17, v8, v13 // (op 544) 

v_add_f32 v10, v14, v18 // (op 541) 
v_add_f32 v11, v15, v19 // (op 541) 
v_add_f32 v12, v16, v20 // (op 541) 
v_add_f32 v13, v17, v21 // (op 541) 





buffer_store_dwordx4 v[10:13], v5, s[56:59], 0 offen offset:128 // (op 542) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[52:55], 0 offen // (op 135) Load value
v_accvgpr_read v14, a0 // (op 132) 
v_accvgpr_read v15, a1 // (op 132) 
v_accvgpr_read v16, a2 // (op 132) 
v_accvgpr_read v17, a3 // (op 132) 
v_mul_f32 v18, v9, v14 // (op 132) 
v_mul_f32 v19, v9, v15 // (op 132) 
v_mul_f32 v20, v9, v16 // (op 132) 
v_mul_f32 v21, v9, v17 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 136) 
v_mul_f32 v15, v8, v11 // (op 136) 
v_mul_f32 v16, v8, v12 // (op 136) 
v_mul_f32 v17, v8, v13 // (op 136) 

v_add_f32 v10, v14, v18 // (op 133) 
v_add_f32 v11, v15, v19 // (op 133) 
v_add_f32 v12, v16, v20 // (op 133) 
v_add_f32 v13, v17, v21 // (op 133) 





buffer_store_dwordx4 v[10:13], v5, s[56:59], 0 offen // (op 134) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
.LRR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__end:
.size RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_, .LRR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__end-RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 19968 // lds bytes
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
    DataType: B8
    DataTypeA: B8
    DataTypeB: B6
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
    macrotile: [64, 16, 128]
    threads: [128, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 19968
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 62
    .agpr_count: 8
    .max_flat_workgroup_size: 128
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

