// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT



.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 3832) 
s_mul_hi_i32 s1, s16, s10 // (op 4021) 
v_lshrrev_b32 v0, 6, v1 // (op 3947) 
v_and_b32 v3, 1, v0 // (op 3947) 

v_and_b32 v0, 15, v1 // (op 3947) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 3947) 


v_and_b32 v0, 63, v1 // (op 3947) 
v_lshrrev_b32 v3, 4, v0 // (op 3947) 

v_lshlrev_b32 v0, 5, v3 // (op 3947) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 3947) 


v_lshlrev_b32 v0, 2, v3 // (op 3947) 

v_lshrrev_b32 v3, 3, v0 // (op 3947) 

v_lshrrev_b32 v0, 6, v1 // (op 4076) 
v_and_b32 v4, 1, v0 // (op 4076) 

v_and_b32 v0, 15, v1 // (op 4076) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 4076) 


v_and_b32 v0, 63, v1 // (op 4076) 
v_lshrrev_b32 v4, 4, v0 // (op 4076) 

v_lshlrev_b32 v0, 5, v4 // (op 4076) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 4076) 


v_lshlrev_b32 v0, 2, v4 // (op 4076) 

v_lshrrev_b32 v4, 3, v0 // (op 4076) 

v_lshrrev_b32 v0, 7, v1 // (op 3832) 
s_add_i32 s18, s0, s16 // (op 3832) 

v_and_b32 v5, 63, v1 // (op 3832) 
s_ashr_i32 s0, s18, 31 // (op 3832) 
v_lshrrev_b32 v6, 4, v5 // (op 3832) 
s_cmp_eq_i32 s10, 0 // (op 3832) 
s_mov_b32 s19, scc // (op 3832) 
v_lshlrev_b32 v7, 2, v6 // (op 3832) 

s_cmp_eq_i32 s19, 0 // (op 3832) 
s_cselect_b32 s49, 0, -1 // (op 3832) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 3832) 


s_and_b32 s19, s13, 31 // (op 3832) 
v_lshrrev_b32 v0, 2, v5 // (op 3832) 

v_and_b32 v5, 3, v0 // (op 3832) 

v_lshrrev_b32 v0, 7, v1 // (op 3894) 
v_and_b32 v7, 15, v1 // (op 3894) 
v_lshl_add_u32 v8, v0, 4, v7 // (op 3894) 


v_and_b32 v0, 48, v1 // (op 3894) 
v_lshl_add_u32 v7, v8, 7, v0 // (op 3894) 


v_lshrrev_b32 v0, 7, v1 // (op 4005) 
v_and_b32 v8, 15, v1 // (op 4005) 
v_lshl_add_u32 v9, v0, 4, v8 // (op 4005) 


v_and_b32 v0, 63, v1 // (op 4005) 
v_lshrrev_b32 v8, 4, v0 // (op 4005) 

v_lshl_add_u32 v0, v9, 2, v8 // (op 4005) 


v_lshrrev_b32 v8, 7, v1 // (op 4021) 
v_and_b32 v9, 63, v1 // (op 4021) 
v_lshrrev_b32 v10, 7, v1 // (op 4052) 
v_and_b32 v11, 15, v1 // (op 4052) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 4052) 


v_and_b32 v10, 48, v1 // (op 4052) 
v_lshl_add_u32 v11, v12, 7, v10 // (op 4052) 


v_lshrrev_b32 v10, 7, v1 // (op 4105) 
v_and_b32 v12, 15, v1 // (op 4105) 
v_lshl_add_u32 v13, v10, 4, v12 // (op 4105) 


v_and_b32 v10, 63, v1 // (op 4105) 
v_lshrrev_b32 v12, 4, v10 // (op 4105) 

v_lshl_add_u32 v10, v13, 2, v12 // (op 4105) 


s_lshl_b32 s50, 1, s19 // (op 3832) 
v_and_b32 v12, 3, v1 // (op 3832) 
s_add_i32 s51, s50, s49 // (op 3832) 


s_and_b32 s49, s0, s51 // (op 3832) 


s_add_i32 s0, s1, s16 // (op 4021) 

s_ashr_i32 s1, s0, 31 // (op 4021) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 3832) 


s_add_i32 s50, s18, s49 // (op 3832) 


v_lshrrev_b32 v5, 6, v1 // (op 3832) 
s_ashr_i32 s18, s50, s19 // (op 3832) 


v_and_b32 v12, 1, v5 // (op 3832) 

s_lshl_b32 s19, s13, 24 // (op 3832) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 3832) 


s_ashr_i32 s49, s19, 31 // (op 3832) 

s_xor_b32 s19, s18, s49 // (op 3832) 

s_sub_i32 s18, s19, s49 // (op 3832) 


s_mul_hi_i32 s19, s18, s14 // (op 3832) 
s_add_i32 s49, s19, s18 // (op 3832) 

s_ashr_i32 s19, s49, 31 // (op 3832) 
s_cmp_eq_i32 s14, 0 // (op 3832) 
s_mov_b32 s50, scc // (op 3832) 
s_cmp_eq_i32 s50, 0 // (op 3832) 
s_cselect_b32 s51, 0, -1 // (op 3832) 

s_and_b32 s50, s15, 31 // (op 3832) 
v_lshrrev_b32 v12, 4, v9 // (op 4021) 
s_cmp_eq_i32 s10, 0 // (op 4021) 
s_mov_b32 s52, scc // (op 4021) 
v_lshlrev_b32 v13, 2, v12 // (op 4021) 

s_cmp_eq_i32 s52, 0 // (op 4021) 
s_cselect_b32 s53, 0, -1 // (op 4021) 

s_lshl_b32 s52, 1, s50 // (op 3832) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 4021) 


v_lshrrev_b32 v8, 2, v9 // (op 4021) 

s_add_i32 s54, s52, s51 // (op 3832) 


s_and_b32 s51, s19, s54 // (op 3832) 


s_add_i32 s19, s49, s51 // (op 3832) 


s_ashr_i32 s49, s19, s50 // (op 3832) 


s_lshl_b32 s19, s15, 24 // (op 3832) 
s_ashr_i32 s50, s19, 31 // (op 3832) 

s_xor_b32 s19, s49, s50 // (op 3832) 

s_sub_i32 s49, s19, s50 // (op 3832) 


s_mul_i32 s19, s18, s11 // (op 3832) 
s_sub_u32 s50, s16, s19 // (op 3832) 

s_mul_i32 s19, s49, s11 // (op 3832) 
s_add_u32 s19, s19, s50 // (op 3832) 

s_mul_hi_i32 s50, s16, s9 // (op 3832) 
s_add_i32 s51, s50, s16 // (op 3832) 

s_ashr_i32 s50, s51, 31 // (op 3832) 
s_cmp_eq_i32 s9, 0 // (op 3832) 
s_mov_b32 s52, scc // (op 3832) 
s_cmp_eq_i32 s52, 0 // (op 3832) 
s_cselect_b32 s54, 0, -1 // (op 3832) 

s_and_b32 s52, s13, 31 // (op 4021) 
v_and_b32 v9, 3, v8 // (op 4021) 

s_lshl_b32 s55, 1, s52 // (op 4021) 
v_and_b32 v8, 3, v1 // (op 4021) 
s_add_i32 s56, s55, s53 // (op 4021) 


s_and_b32 s53, s1, s56 // (op 4021) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 4021) 


s_add_i32 s1, s0, s53 // (op 4021) 


v_lshrrev_b32 v8, 6, v1 // (op 4021) 
s_ashr_i32 s0, s1, s52 // (op 4021) 


v_and_b32 v9, 1, v8 // (op 4021) 

s_lshl_b32 s1, s13, 24 // (op 4021) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 4021) 


s_ashr_i32 s52, s1, 31 // (op 4021) 

s_xor_b32 s1, s0, s52 // (op 4021) 

s_sub_i32 s0, s1, s52 // (op 4021) 


s_mul_hi_i32 s1, s0, s14 // (op 4021) 
s_add_i32 s52, s1, s0 // (op 4021) 

s_ashr_i32 s1, s52, 31 // (op 4021) 
s_cmp_eq_i32 s14, 0 // (op 4021) 
s_mov_b32 s53, scc // (op 4021) 
s_cmp_eq_i32 s53, 0 // (op 4021) 
s_cselect_b32 s55, 0, -1 // (op 4021) 

s_and_b32 s53, s15, 31 // (op 4021) 
s_lshl_b32 s56, 1, s53 // (op 4021) 
s_add_i32 s57, s56, s55 // (op 4021) 


s_and_b32 s55, s1, s57 // (op 4021) 


s_add_i32 s1, s52, s55 // (op 4021) 


s_ashr_i32 s52, s1, s53 // (op 4021) 


s_lshl_b32 s1, s15, 24 // (op 4021) 
s_ashr_i32 s53, s1, 31 // (op 4021) 

s_xor_b32 s1, s52, s53 // (op 4021) 

s_sub_i32 s52, s1, s53 // (op 4021) 


s_mul_i32 s1, s0, s11 // (op 4021) 
s_sub_u32 s53, s16, s1 // (op 4021) 

s_mul_i32 s1, s52, s11 // (op 4021) 
s_add_u32 s1, s1, s53 // (op 4021) 

s_mul_hi_i32 s53, s16, s9 // (op 4021) 
s_add_i32 s55, s53, s16 // (op 4021) 

s_ashr_i32 s53, s55, 31 // (op 4021) 
s_cmp_eq_i32 s9, 0 // (op 4021) 
s_mov_b32 s56, scc // (op 4021) 
s_cmp_eq_i32 s56, 0 // (op 4021) 
s_cselect_b32 s57, 0, -1 // (op 4021) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 3832) 
s_lshl_b32 s58, 1, s56 // (op 3832) 
s_add_i32 s59, s58, s54 // (op 3832) 


s_and_b32 s54, s50, s59 // (op 3832) 


s_add_i32 s50, s51, s54 // (op 3832) 


s_ashr_i32 s51, s50, s56 // (op 3832) 


s_lshl_b32 s50, s45, 24 // (op 3832) 
s_ashr_i32 s54, s50, 31 // (op 3832) 

s_xor_b32 s50, s51, s54 // (op 3832) 

s_sub_i32 s51, s50, s54 // (op 3832) 


s_and_b32 s50, s45, 31 // (op 4021) 
s_lshl_b32 s54, 1, s50 // (op 4021) 
s_add_i32 s56, s54, s57 // (op 4021) 


s_and_b32 s54, s53, s56 // (op 4021) 


s_add_i32 s53, s55, s54 // (op 4021) 


s_ashr_i32 s54, s53, s50 // (op 4021) 


s_lshl_b32 s50, s45, 24 // (op 4021) 
s_ashr_i32 s53, s50, 31 // (op 4021) 

s_xor_b32 s50, s54, s53 // (op 4021) 

s_sub_i32 s54, s50, s53 // (op 4021) 


s_add_u32 s56, s4, 32 // (op 3832) Start of Int64 add, locking SCC
s_addc_u32 s57, s5, 0 // (op 3832) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 3832) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 3832) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 3832) 

s_mul_i32 s50, s11, s56 // (op 3832) 
s_mul_i32 s53, s8, s50 // (op 3832) 

s_mul_i32 s50, s51, s53 // (op 3832) 

s_sub_u32 s53, s16, s50 // (op 3832) 

s_mov_b32 s50, s53 // (op 3832) 
s_mul_hi_i32 s55, s50, s12 // (op 3832) 
s_add_i32 s58, s55, s50 // (op 3832) 


s_ashr_i32 s50, s58, 31 // (op 3832) 
s_cmp_eq_i32 s12, 0 // (op 3832) 
s_mov_b32 s55, scc // (op 3832) 
s_cmp_eq_i32 s55, 0 // (op 3832) 
s_cselect_b32 s59, 0, -1 // (op 3832) 

s_and_b32 s55, s44, 31 // (op 3832) 
s_lshl_b32 s60, 1, s55 // (op 3832) 
s_add_i32 s61, s60, s59 // (op 3832) 


s_and_b32 s59, s50, s61 // (op 3832) 


s_add_i32 s50, s58, s59 // (op 3832) 


s_ashr_i32 s58, s50, s55 // (op 3832) 


s_lshl_b32 s50, s44, 24 // (op 3832) 
s_ashr_i32 s55, s50, 31 // (op 3832) 

s_xor_b32 s50, s58, s55 // (op 3832) 

s_sub_i32 s58, s50, s55 // (op 3832) 


s_mul_i32 s50, s46, s11 // (op 3832) 
s_add_u32 s60, s6, 256 // (op 3832) Start of Int64 add, locking SCC
s_addc_u32 s61, s7, 0 // (op 3832) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 3832) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 3832) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 8 // (op 3832) 

s_sub_i32 s55, s60, s50 // (op 3832) 

s_mul_i32 s59, s58, s55 // (op 3832) 

s_sub_u32 s55, s53, s59 // (op 3832) 


s_add_u32 s53, s55, s50 // (op 3832) 


s_cmp_eq_i32 s51, 0 // (op 3832) 
s_mov_b32 s50, scc // (op 3832) 

s_cmp_eq_i32 s50, 0 // (op 3832) 
s_cselect_b32 s51, s53, s19 // (op 3832) 


s_mul_i32 s19, s49, s56 // (op 3832) 


v_lshl_add_u32 v9, s51, 8, v6 // (op 3832) 


s_sub_i32 s49, s18, s19 // (op 3832) 


s_cmp_eq_i32 s50, 0 // (op 3832) 
s_cselect_b32 s51, s58, s49 // (op 3832) 



v_lshl_add_u32 v6, s51, 5, v5 // (op 3832) 


v_mul_lo_u32 v5, v6, s2 // (op 3832) 

v_add_lshl_u32 v6, v9, v5, 1 // (op 3832) 


s_add_u32 s18, s4, 32 // (op 4021) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 4021) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 4021) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 4021) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 4021) 

s_mul_i32 s53, s11, s18 // (op 4021) 
s_mul_i32 s49, s8, s53 // (op 4021) 

s_mul_i32 s53, s54, s49 // (op 4021) 

s_sub_u32 s49, s16, s53 // (op 4021) 

s_mov_b32 s53, s49 // (op 4021) 
s_mul_hi_i32 s50, s53, s12 // (op 4021) 
s_add_i32 s51, s50, s53 // (op 4021) 


s_ashr_i32 s50, s51, 31 // (op 4021) 
s_cmp_eq_i32 s12, 0 // (op 4021) 
s_mov_b32 s53, scc // (op 4021) 
s_cmp_eq_i32 s53, 0 // (op 4021) 
s_cselect_b32 s55, 0, -1 // (op 4021) 

s_and_b32 s53, s44, 31 // (op 4021) 
s_lshl_b32 s56, 1, s53 // (op 4021) 
s_add_i32 s57, s56, s55 // (op 4021) 


s_and_b32 s55, s50, s57 // (op 4021) 


s_add_i32 s50, s51, s55 // (op 4021) 


s_ashr_i32 s51, s50, s53 // (op 4021) 


s_lshl_b32 s50, s44, 24 // (op 4021) 
s_ashr_i32 s53, s50, 31 // (op 4021) 

s_xor_b32 s50, s51, s53 // (op 4021) 

s_sub_i32 s51, s50, s53 // (op 4021) 


s_mul_i32 s50, s46, s11 // (op 4021) 
s_add_u32 s56, s6, 256 // (op 4021) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 4021) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 4021) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 4021) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 4021) 

s_sub_i32 s53, s56, s50 // (op 4021) 

s_mul_i32 s55, s51, s53 // (op 4021) 

s_sub_u32 s53, s49, s55 // (op 4021) 


s_add_u32 s49, s53, s50 // (op 4021) 


s_cmp_eq_i32 s54, 0 // (op 4021) 
s_mov_b32 s50, scc // (op 4021) 

s_cmp_eq_i32 s50, 0 // (op 4021) 
s_cselect_b32 s53, s49, s1 // (op 4021) 


s_mul_i32 s1, s52, s18 // (op 4021) 


v_lshl_add_u32 v5, s53, 8, v12 // (op 4021) 


s_sub_i32 s49, s0, s1 // (op 4021) 


s_cmp_eq_i32 s50, 0 // (op 4021) 
s_cselect_b32 s0, s51, s49 // (op 4021) 



v_lshl_add_u32 v9, s0, 5, v8 // (op 4021) 


v_mul_lo_u32 v8, v9, s20 // (op 4021) 

v_add_lshl_u32 v9, v5, v8, 1 // (op 4021) 


s_mul_i32 s1, s2, s5 // (op 3834) most significant: low of low * high
s_mul_i32 s18, s3, s4 // (op 3834) most significant: low of high * low
s_mul_i32 s0, s2, s4 // (op 3834) least significant
s_mul_hi_u32 s19, s2, s4 // (op 3834) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3834) most significant: sum
s_add_u32 s1, s1, s19 // (op 3834) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 3834) 

s_mul_i32 s1, s20, s5 // (op 4023) most significant: low of low * high
s_mul_i32 s49, s21, s4 // (op 4023) most significant: low of high * low
s_mul_i32 s0, s20, s4 // (op 4023) least significant
s_mul_hi_u32 s50, s20, s4 // (op 4023) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 4023) most significant: sum
s_add_u32 s1, s1, s50 // (op 4023) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 4023) 

s_mov_b32 s52, s22 // (op 3834) reg expression
s_mov_b32 s53, s23 // (op 3834) reg expression
s_mov_b32 s54, s18 // (op 3834) reg expression
s_mov_b32 s55, 131072 // (op 3834) call()

s_mov_b32 s56, s24 // (op 4023) reg expression

s_mov_b32 s57, s25 // (op 4023) reg expression
s_mov_b32 s58, s50 // (op 4023) reg expression
s_mov_b32 s59, 131072 // (op 4023) call()


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
v_accvgpr_write a16, 0.00000 // (op 799) call()
v_accvgpr_write a17, 0.00000 // (op 799) call()
v_accvgpr_write a18, 0.00000 // (op 799) call()
v_accvgpr_write a19, 0.00000 // (op 799) call()
v_accvgpr_write a20, 0.00000 // (op 961) call()
v_accvgpr_write a21, 0.00000 // (op 961) call()
v_accvgpr_write a22, 0.00000 // (op 961) call()
v_accvgpr_write a23, 0.00000 // (op 961) call()
v_accvgpr_write a24, 0.00000 // (op 1123) call()
v_accvgpr_write a25, 0.00000 // (op 1123) call()
v_accvgpr_write a26, 0.00000 // (op 1123) call()
v_accvgpr_write a27, 0.00000 // (op 1123) call()
v_accvgpr_write a28, 0.00000 // (op 1285) call()
v_accvgpr_write a29, 0.00000 // (op 1285) call()
v_accvgpr_write a30, 0.00000 // (op 1285) call()
v_accvgpr_write a31, 0.00000 // (op 1285) call()
s_mul_hi_i32 s0, s16, s10 // (op 3599) 
v_and_b32 v12, 4294967288, v1 // (op 3599) 
s_add_i32 s1, s0, s16 // (op 3599) 

v_and_b32 v13, -8, v1 // (op 3599) 
s_ashr_i32 s0, s1, 31 // (op 3599) 
v_sub_u32 v14, v1, v13 // (op 3599) 

s_cmp_eq_i32 s10, 0 // (op 3599) 
s_mov_b32 s18, scc // (op 3599) 
v_lshlrev_b32 v13, 4, v14 // (op 3599) 

s_cmp_eq_i32 s18, 0 // (op 3599) 
s_cselect_b32 s19, 0, -1 // (op 3599) 

s_and_b32 s18, s13, 31 // (op 3599) 
s_lshl_b32 s22, 1, s18 // (op 3599) 
s_add_i32 s23, s22, s19 // (op 3599) 


s_and_b32 s19, s0, s23 // (op 3599) 


s_add_i32 s0, s1, s19 // (op 3599) 


s_ashr_i32 s1, s0, s18 // (op 3599) 


s_lshl_b32 s0, s13, 24 // (op 3599) 
s_ashr_i32 s18, s0, 31 // (op 3599) 

s_xor_b32 s0, s1, s18 // (op 3599) 

s_sub_i32 s1, s0, s18 // (op 3599) 


s_mul_hi_i32 s0, s1, s14 // (op 3599) 
s_add_i32 s18, s0, s1 // (op 3599) 

s_ashr_i32 s0, s18, 31 // (op 3599) 
s_cmp_eq_i32 s14, 0 // (op 3599) 
s_mov_b32 s19, scc // (op 3599) 
s_cmp_eq_i32 s19, 0 // (op 3599) 
s_cselect_b32 s22, 0, -1 // (op 3599) 

s_and_b32 s19, s15, 31 // (op 3599) 
s_lshl_b32 s23, 1, s19 // (op 3599) 
s_add_i32 s24, s23, s22 // (op 3599) 


s_and_b32 s25, s0, s24 // (op 3599) 


s_add_i32 s0, s18, s25 // (op 3599) 


s_ashr_i32 s18, s0, s19 // (op 3599) 


s_lshl_b32 s0, s15, 24 // (op 3599) 
s_ashr_i32 s19, s0, 31 // (op 3599) 

s_xor_b32 s0, s18, s19 // (op 3599) 

s_sub_i32 s18, s0, s19 // (op 3599) 


s_mul_i32 s0, s1, s11 // (op 3599) 

s_sub_u32 s1, s16, s0 // (op 3599) 

s_mul_i32 s0, s18, s11 // (op 3599) 
s_add_u32 s0, s0, s1 // (op 3599) 


s_mul_hi_i32 s1, s16, s9 // (op 3599) 
s_add_i32 s18, s1, s16 // (op 3599) 

s_ashr_i32 s1, s18, 31 // (op 3599) 
s_cmp_eq_i32 s9, 0 // (op 3599) 
s_mov_b32 s19, scc // (op 3599) 
s_cmp_eq_i32 s19, 0 // (op 3599) 
s_cselect_b32 s22, 0, -1 // (op 3599) 

s_and_b32 s19, s45, 31 // (op 3599) 
s_lshl_b32 s23, 1, s19 // (op 3599) 
s_add_i32 s24, s23, s22 // (op 3599) 


s_and_b32 s25, s1, s24 // (op 3599) 


s_add_i32 s1, s18, s25 // (op 3599) 


s_ashr_i32 s18, s1, s19 // (op 3599) 


s_lshl_b32 s1, s45, 24 // (op 3599) 
s_ashr_i32 s19, s1, 31 // (op 3599) 

s_xor_b32 s1, s18, s19 // (op 3599) 

s_sub_i32 s18, s1, s19 // (op 3599) 


s_add_u32 s22, s4, 32 // (op 3599) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3599) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3599) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3599) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 3599) 

s_mul_i32 s1, s11, s22 // (op 3599) 

s_mul_i32 s19, s8, s1 // (op 3599) 

s_mul_i32 s1, s18, s19 // (op 3599) 

s_sub_u32 s19, s16, s1 // (op 3599) 

s_mov_b32 s1, s19 // (op 3599) 
s_mul_hi_i32 s22, s1, s12 // (op 3599) 
s_add_i32 s23, s22, s1 // (op 3599) 


s_ashr_i32 s1, s23, 31 // (op 3599) 
s_cmp_eq_i32 s12, 0 // (op 3599) 
s_mov_b32 s22, scc // (op 3599) 
s_cmp_eq_i32 s22, 0 // (op 3599) 
s_cselect_b32 s24, 0, -1 // (op 3599) 

s_and_b32 s22, s44, 31 // (op 3599) 
s_lshl_b32 s25, 1, s22 // (op 3599) 
s_add_i32 s47, s25, s24 // (op 3599) 


s_and_b32 s24, s1, s47 // (op 3599) 


s_add_i32 s1, s23, s24 // (op 3599) 


s_ashr_i32 s23, s1, s22 // (op 3599) 


s_lshl_b32 s1, s44, 24 // (op 3599) 
s_ashr_i32 s22, s1, 31 // (op 3599) 

s_xor_b32 s1, s23, s22 // (op 3599) 

s_sub_i32 s23, s1, s22 // (op 3599) 


s_mul_i32 s1, s46, s11 // (op 3599) 
s_add_u32 s24, s6, 256 // (op 3599) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 3599) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 3599) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 3599) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 3599) 

s_sub_i32 s22, s24, s1 // (op 3599) 

s_mul_i32 s24, s23, s22 // (op 3599) 


s_sub_u32 s25, s19, s24 // (op 3599) 


s_add_u32 s19, s25, s1 // (op 3599) 


s_cmp_eq_i32 s18, 0 // (op 3599) 
s_mov_b32 s1, scc // (op 3599) 

s_cmp_eq_i32 s1, 0 // (op 3599) 
s_cselect_b32 s18, s19, s0 // (op 3599) 



v_lshl_add_u32 v14, s18, 8, v12 // (op 3599) 


v_mul_lo_u32 v12, v14, s26 // (op 3599) 
v_add_u32 v12, v12, v13 // (op 3599) 


s_mul_i32 s1, s26, s7 // (op 3601) most significant: low of low * high
s_mul_i32 s18, s27, s6 // (op 3601) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 3601) least significant
s_mul_hi_u32 s19, s26, s6 // (op 3601) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3601) most significant: sum
s_add_u32 s1, s1, s19 // (op 3601) most significant: sum


s_mov_b32 s48, s28 // (op 3601) reg expression
s_mov_b32 s49, s29 // (op 3601) reg expression
s_mov_b32 s50, s0 // (op 3601) reg expression
s_mov_b32 s51, 131072 // (op 3601) call()

s_mul_hi_i32 s47, s16, s10 // (op 3618) 
v_and_b32 v13, 4294967288, v1 // (op 3618) 
s_add_i32 s0, s47, s16 // (op 3618) 

v_and_b32 v14, -8, v1 // (op 3618) 
s_ashr_i32 s1, s0, 31 // (op 3618) 
v_sub_u32 v15, v1, v14 // (op 3618) 

s_cmp_eq_i32 s10, 0 // (op 3618) 
s_mov_b32 s47, scc // (op 3618) 
v_lshlrev_b32 v14, 4, v15 // (op 3618) 

s_cmp_eq_i32 s47, 0 // (op 3618) 
s_cselect_b32 s18, 0, -1 // (op 3618) 

s_and_b32 s19, s13, 31 // (op 3618) 
s_lshl_b32 s47, 1, s19 // (op 3618) 
s_add_i32 s22, s47, s18 // (op 3618) 


s_and_b32 s18, s1, s22 // (op 3618) 


s_add_i32 s1, s0, s18 // (op 3618) 


s_ashr_i32 s0, s1, s19 // (op 3618) 


s_lshl_b32 s1, s13, 24 // (op 3618) 
s_ashr_i32 s47, s1, 31 // (op 3618) 

s_xor_b32 s1, s0, s47 // (op 3618) 

s_sub_i32 s0, s1, s47 // (op 3618) 


s_mul_hi_i32 s1, s0, s14 // (op 3618) 
s_add_i32 s47, s1, s0 // (op 3618) 

s_ashr_i32 s1, s47, 31 // (op 3618) 
s_cmp_eq_i32 s14, 0 // (op 3618) 
s_mov_b32 s18, scc // (op 3618) 
s_cmp_eq_i32 s18, 0 // (op 3618) 
s_cselect_b32 s19, 0, -1 // (op 3618) 

s_and_b32 s18, s15, 31 // (op 3618) 
s_lshl_b32 s22, 1, s18 // (op 3618) 
s_add_i32 s23, s22, s19 // (op 3618) 


s_and_b32 s19, s1, s23 // (op 3618) 


s_add_i32 s1, s47, s19 // (op 3618) 


s_ashr_i32 s19, s1, s18 // (op 3618) 


s_lshl_b32 s1, s15, 24 // (op 3618) 
s_ashr_i32 s18, s1, 31 // (op 3618) 

s_xor_b32 s1, s19, s18 // (op 3618) 

s_sub_i32 s19, s1, s18 // (op 3618) 


s_mul_i32 s1, s0, s11 // (op 3618) 

s_sub_u32 s0, s16, s1 // (op 3618) 

s_mul_i32 s1, s19, s11 // (op 3618) 
s_add_u32 s1, s1, s0 // (op 3618) 


s_mul_hi_i32 s0, s16, s9 // (op 3618) 
s_add_i32 s47, s0, s16 // (op 3618) 

s_ashr_i32 s0, s47, 31 // (op 3618) 
s_cmp_eq_i32 s9, 0 // (op 3618) 
s_mov_b32 s18, scc // (op 3618) 
s_cmp_eq_i32 s18, 0 // (op 3618) 
s_cselect_b32 s19, 0, -1 // (op 3618) 

s_and_b32 s18, s45, 31 // (op 3618) 
s_lshl_b32 s22, 1, s18 // (op 3618) 
s_add_i32 s23, s22, s19 // (op 3618) 


s_and_b32 s19, s0, s23 // (op 3618) 


s_add_i32 s0, s47, s19 // (op 3618) 


s_ashr_i32 s19, s0, s18 // (op 3618) 


s_lshl_b32 s0, s45, 24 // (op 3618) 
s_ashr_i32 s18, s0, 31 // (op 3618) 

s_xor_b32 s0, s19, s18 // (op 3618) 

s_sub_i32 s19, s0, s18 // (op 3618) 


s_add_u32 s22, s4, 32 // (op 3618) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3618) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3618) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3618) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 3618) 

s_mul_i32 s0, s11, s22 // (op 3618) 

s_mul_i32 s18, s8, s0 // (op 3618) 

s_mul_i32 s0, s19, s18 // (op 3618) 

s_sub_u32 s18, s16, s0 // (op 3618) 

s_mov_b32 s0, s18 // (op 3618) 
s_mul_hi_i32 s47, s0, s12 // (op 3618) 
s_add_i32 s22, s47, s0 // (op 3618) 


s_ashr_i32 s0, s22, 31 // (op 3618) 
s_cmp_eq_i32 s12, 0 // (op 3618) 
s_mov_b32 s47, scc // (op 3618) 
s_cmp_eq_i32 s47, 0 // (op 3618) 
s_cselect_b32 s23, 0, -1 // (op 3618) 

s_and_b32 s47, s44, 31 // (op 3618) 
s_lshl_b32 s24, 1, s47 // (op 3618) 
s_add_i32 s25, s24, s23 // (op 3618) 


s_and_b32 s23, s0, s25 // (op 3618) 


s_add_i32 s0, s22, s23 // (op 3618) 


s_ashr_i32 s22, s0, s47 // (op 3618) 


s_lshl_b32 s0, s44, 24 // (op 3618) 
s_ashr_i32 s47, s0, 31 // (op 3618) 

s_xor_b32 s0, s22, s47 // (op 3618) 

s_sub_i32 s22, s0, s47 // (op 3618) 


s_mul_i32 s0, s46, s11 // (op 3618) 
s_add_u32 s24, s6, 256 // (op 3618) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 3618) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 3618) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 3618) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 8 // (op 3618) 

s_sub_i32 s23, s24, s0 // (op 3618) 

s_mul_i32 s47, s22, s23 // (op 3618) 


s_sub_u32 s22, s18, s47 // (op 3618) 


s_add_u32 s18, s22, s0 // (op 3618) 


s_cmp_eq_i32 s19, 0 // (op 3618) 
s_mov_b32 s0, scc // (op 3618) 

s_cmp_eq_i32 s0, 0 // (op 3618) 
s_cselect_b32 s19, s18, s1 // (op 3618) 



v_lshl_add_u32 v15, s19, 8, v13 // (op 3618) 


v_mul_lo_u32 v13, v15, s26 // (op 3618) 
v_add_u32 v13, v13, v14 // (op 3618) 


s_mul_i32 s1, s26, s7 // (op 3620) most significant: low of low * high
s_mul_i32 s47, s27, s6 // (op 3620) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 3620) least significant
s_mul_hi_u32 s18, s26, s6 // (op 3620) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3620) most significant: sum
s_add_u32 s1, s1, s18 // (op 3620) most significant: sum


s_mov_b32 s48, s28 // (op 3620) reg expression
s_mov_b32 s49, s29 // (op 3620) reg expression
s_mov_b32 s50, s0 // (op 3620) reg expression
s_mov_b32 s51, 131072 // (op 3620) call()

s_mul_hi_i32 s47, s16, s10 // (op 3663) 
v_lshrrev_b32 v14, 3, v1 // (op 3663) 
s_add_i32 s0, s47, s16 // (op 3663) 

v_and_b32 v15, -8, v1 // (op 3663) 
s_ashr_i32 s1, s0, 31 // (op 3663) 
v_sub_u32 v16, v1, v15 // (op 3663) 

s_cmp_eq_i32 s10, 0 // (op 3663) 
s_mov_b32 s47, scc // (op 3663) 
v_lshlrev_b32 v15, 4, v16 // (op 3663) 

s_cmp_eq_i32 s47, 0 // (op 3663) 
s_cselect_b32 s18, 0, -1 // (op 3663) 

v_mov_b32 v17, 0 // (op 3663) convert
v_mov_b32 v16, v15 // (op 3663) convert

s_and_b32 s19, s13, 31 // (op 3663) 
s_lshl_b32 s47, 1, s19 // (op 3663) 
s_add_i32 s22, s47, s18 // (op 3663) 


s_and_b32 s18, s1, s22 // (op 3663) 


s_add_i32 s1, s0, s18 // (op 3663) 


s_ashr_i32 s0, s1, s19 // (op 3663) 


s_lshl_b32 s1, s13, 24 // (op 3663) 
s_ashr_i32 s47, s1, 31 // (op 3663) 

s_xor_b32 s1, s0, s47 // (op 3663) 

s_sub_i32 s0, s1, s47 // (op 3663) 


s_mul_hi_i32 s1, s0, s14 // (op 3663) 
s_add_i32 s47, s1, s0 // (op 3663) 

s_ashr_i32 s1, s47, 31 // (op 3663) 
s_cmp_eq_i32 s14, 0 // (op 3663) 
s_mov_b32 s18, scc // (op 3663) 
s_cmp_eq_i32 s18, 0 // (op 3663) 
s_cselect_b32 s19, 0, -1 // (op 3663) 

s_and_b32 s18, s15, 31 // (op 3663) 
s_lshl_b32 s22, 1, s18 // (op 3663) 
s_add_i32 s23, s22, s19 // (op 3663) 


s_and_b32 s19, s1, s23 // (op 3663) 


s_add_i32 s1, s47, s19 // (op 3663) 


s_ashr_i32 s19, s1, s18 // (op 3663) 


s_lshl_b32 s1, s15, 24 // (op 3663) 
s_ashr_i32 s18, s1, 31 // (op 3663) 

s_xor_b32 s1, s19, s18 // (op 3663) 

s_sub_i32 s19, s1, s18 // (op 3663) 


s_add_u32 s22, s4, 32 // (op 3663) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3663) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3663) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3663) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 3663) 

s_mul_i32 s1, s19, s22 // (op 3663) 

s_sub_i32 s47, s0, s1 // (op 3663) 


s_mul_hi_i32 s0, s16, s9 // (op 3663) 
s_add_i32 s1, s0, s16 // (op 3663) 

s_ashr_i32 s0, s1, 31 // (op 3663) 
s_cmp_eq_i32 s9, 0 // (op 3663) 
s_mov_b32 s18, scc // (op 3663) 
s_cmp_eq_i32 s18, 0 // (op 3663) 
s_cselect_b32 s19, 0, -1 // (op 3663) 

s_and_b32 s18, s45, 31 // (op 3663) 
s_lshl_b32 s24, 1, s18 // (op 3663) 
s_add_i32 s25, s24, s19 // (op 3663) 


s_and_b32 s19, s0, s25 // (op 3663) 


s_add_i32 s0, s1, s19 // (op 3663) 


s_ashr_i32 s1, s0, s18 // (op 3663) 


s_lshl_b32 s0, s45, 24 // (op 3663) 
s_ashr_i32 s18, s0, 31 // (op 3663) 

s_xor_b32 s0, s1, s18 // (op 3663) 

s_sub_i32 s1, s0, s18 // (op 3663) 


s_mul_i32 s0, s11, s22 // (op 3663) 

s_mul_i32 s18, s8, s0 // (op 3663) 

s_mul_i32 s0, s1, s18 // (op 3663) 

s_sub_u32 s18, s16, s0 // (op 3663) 

s_mul_hi_i32 s0, s18, s12 // (op 3663) 
s_add_i32 s19, s0, s18 // (op 3663) 


s_ashr_i32 s0, s19, 31 // (op 3663) 
s_cmp_eq_i32 s12, 0 // (op 3663) 
s_mov_b32 s18, scc // (op 3663) 
s_cmp_eq_i32 s18, 0 // (op 3663) 
s_cselect_b32 s22, 0, -1 // (op 3663) 

s_and_b32 s18, s44, 31 // (op 3663) 
s_lshl_b32 s23, 1, s18 // (op 3663) 
s_add_i32 s24, s23, s22 // (op 3663) 


s_and_b32 s25, s0, s24 // (op 3663) 


s_add_i32 s0, s19, s25 // (op 3663) 


s_ashr_i32 s19, s0, s18 // (op 3663) 


s_lshl_b32 s0, s44, 24 // (op 3663) 
s_ashr_i32 s18, s0, 31 // (op 3663) 

s_xor_b32 s0, s19, s18 // (op 3663) 

s_sub_i32 s19, s0, s18 // (op 3663) 


s_cmp_eq_i32 s1, 0 // (op 3663) 
s_mov_b32 s0, scc // (op 3663) 

s_cmp_eq_i32 s0, 0 // (op 3663) 
s_cselect_b32 s1, s19, s47 // (op 3663) 



v_lshl_add_u32 v15, s1, 5, v14 // (op 3663) 


v_mul_lo_u32 v19, v15, s31 // (op 3663) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 3663) most significant: high of low * low
v_mul_lo_u32 v18, v15, s30 // (op 3663) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 3663) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 3663) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 3663) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 3663) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 3663) 

v_mov_b32 v14, v16 // (op 3663) convert

s_mul_i32 s1, s30, s5 // (op 3665) most significant: low of low * high
s_mul_i32 s47, s31, s4 // (op 3665) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 3665) least significant
s_mul_hi_u32 s18, s30, s4 // (op 3665) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3665) most significant: sum
s_add_u32 s1, s1, s18 // (op 3665) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3665) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3665) 

s_mov_b32 s60, s32 // (op 3665) reg expression
s_mov_b32 s61, s33 // (op 3665) reg expression
s_mov_b32 s62, s0 // (op 3665) reg expression
s_mov_b32 s63, 131072 // (op 3665) call()

s_mul_hi_i32 s47, s16, s10 // (op 3682) 
v_lshrrev_b32 v15, 3, v1 // (op 3682) 
s_add_i32 s0, s47, s16 // (op 3682) 

v_and_b32 v16, -8, v1 // (op 3682) 
s_ashr_i32 s1, s0, 31 // (op 3682) 
v_sub_u32 v17, v1, v16 // (op 3682) 

s_cmp_eq_i32 s10, 0 // (op 3682) 
s_mov_b32 s47, scc // (op 3682) 
v_lshlrev_b32 v16, 4, v17 // (op 3682) 

s_cmp_eq_i32 s47, 0 // (op 3682) 
s_cselect_b32 s18, 0, -1 // (op 3682) 

v_mov_b32 v19, 0 // (op 3682) convert
v_mov_b32 v18, v16 // (op 3682) convert

s_and_b32 s19, s13, 31 // (op 3682) 
s_lshl_b32 s47, 1, s19 // (op 3682) 
s_add_i32 s22, s47, s18 // (op 3682) 


s_and_b32 s18, s1, s22 // (op 3682) 


s_add_i32 s1, s0, s18 // (op 3682) 


s_ashr_i32 s0, s1, s19 // (op 3682) 


s_lshl_b32 s1, s13, 24 // (op 3682) 
s_ashr_i32 s47, s1, 31 // (op 3682) 

s_xor_b32 s1, s0, s47 // (op 3682) 

s_sub_i32 s0, s1, s47 // (op 3682) 


s_mul_hi_i32 s1, s0, s14 // (op 3682) 
s_add_i32 s47, s1, s0 // (op 3682) 

s_ashr_i32 s1, s47, 31 // (op 3682) 
s_cmp_eq_i32 s14, 0 // (op 3682) 
s_mov_b32 s18, scc // (op 3682) 
s_cmp_eq_i32 s18, 0 // (op 3682) 
s_cselect_b32 s19, 0, -1 // (op 3682) 

s_and_b32 s18, s15, 31 // (op 3682) 
s_lshl_b32 s22, 1, s18 // (op 3682) 
s_add_i32 s23, s22, s19 // (op 3682) 


s_and_b32 s19, s1, s23 // (op 3682) 


s_add_i32 s1, s47, s19 // (op 3682) 


s_ashr_i32 s19, s1, s18 // (op 3682) 


s_lshl_b32 s1, s15, 24 // (op 3682) 
s_ashr_i32 s18, s1, 31 // (op 3682) 

s_xor_b32 s1, s19, s18 // (op 3682) 

s_sub_i32 s19, s1, s18 // (op 3682) 


s_add_u32 s22, s4, 32 // (op 3682) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3682) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3682) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3682) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 3682) 

s_mul_i32 s1, s19, s22 // (op 3682) 

s_sub_i32 s47, s0, s1 // (op 3682) 


s_mul_hi_i32 s0, s16, s9 // (op 3682) 
s_add_i32 s1, s0, s16 // (op 3682) 

s_ashr_i32 s0, s1, 31 // (op 3682) 
s_cmp_eq_i32 s9, 0 // (op 3682) 
s_mov_b32 s18, scc // (op 3682) 
s_cmp_eq_i32 s18, 0 // (op 3682) 
s_cselect_b32 s19, 0, -1 // (op 3682) 

s_and_b32 s18, s45, 31 // (op 3682) 
s_lshl_b32 s24, 1, s18 // (op 3682) 
s_add_i32 s25, s24, s19 // (op 3682) 


s_and_b32 s19, s0, s25 // (op 3682) 


s_add_i32 s0, s1, s19 // (op 3682) 


s_ashr_i32 s1, s0, s18 // (op 3682) 


s_lshl_b32 s0, s45, 24 // (op 3682) 
s_ashr_i32 s18, s0, 31 // (op 3682) 

s_xor_b32 s0, s1, s18 // (op 3682) 

s_sub_i32 s1, s0, s18 // (op 3682) 


s_mul_i32 s0, s11, s22 // (op 3682) 

s_mul_i32 s18, s8, s0 // (op 3682) 

s_mul_i32 s0, s1, s18 // (op 3682) 

s_sub_u32 s18, s16, s0 // (op 3682) 

s_mul_hi_i32 s0, s18, s12 // (op 3682) 
s_add_i32 s19, s0, s18 // (op 3682) 


s_ashr_i32 s0, s19, 31 // (op 3682) 
s_cmp_eq_i32 s12, 0 // (op 3682) 
s_mov_b32 s18, scc // (op 3682) 
s_cmp_eq_i32 s18, 0 // (op 3682) 
s_cselect_b32 s22, 0, -1 // (op 3682) 

s_and_b32 s18, s44, 31 // (op 3682) 
s_lshl_b32 s23, 1, s18 // (op 3682) 
s_add_i32 s24, s23, s22 // (op 3682) 


s_and_b32 s25, s0, s24 // (op 3682) 


s_add_i32 s0, s19, s25 // (op 3682) 


s_ashr_i32 s19, s0, s18 // (op 3682) 


s_lshl_b32 s0, s44, 24 // (op 3682) 
s_ashr_i32 s18, s0, 31 // (op 3682) 

s_xor_b32 s0, s19, s18 // (op 3682) 

s_sub_i32 s19, s0, s18 // (op 3682) 


s_cmp_eq_i32 s1, 0 // (op 3682) 
s_mov_b32 s0, scc // (op 3682) 

s_cmp_eq_i32 s0, 0 // (op 3682) 
s_cselect_b32 s1, s19, s47 // (op 3682) 



v_lshl_add_u32 v16, s1, 5, v15 // (op 3682) 


v_mul_lo_u32 v21, v16, s31 // (op 3682) most significant: low of low * high
v_mul_hi_u32 v15, v16, s30 // (op 3682) most significant: high of low * low
v_mul_lo_u32 v20, v16, s30 // (op 3682) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 3682) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 3682) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 3682) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 3682) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 3682) 

v_mov_b32 v15, v18 // (op 3682) convert

s_mul_i32 s1, s30, s5 // (op 3684) most significant: low of low * high
s_mul_i32 s47, s31, s4 // (op 3684) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 3684) least significant
s_mul_hi_u32 s18, s30, s4 // (op 3684) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3684) most significant: sum
s_add_u32 s1, s1, s18 // (op 3684) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3684) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3684) 

s_mov_b32 s60, s32 // (op 3684) reg expression
s_mov_b32 s61, s33 // (op 3684) reg expression
s_mov_b32 s62, s0 // (op 3684) reg expression
s_mov_b32 s63, 131072 // (op 3684) call()

s_mul_hi_i32 s47, s16, s10 // (op 3727) 
s_add_i32 s0, s47, s16 // (op 3727) 

s_ashr_i32 s1, s0, 31 // (op 3727) 
s_cmp_eq_i32 s10, 0 // (op 3727) 
s_mov_b32 s47, scc // (op 3727) 
s_cmp_eq_i32 s47, 0 // (op 3727) 
s_cselect_b32 s18, 0, -1 // (op 3727) 

s_and_b32 s19, s13, 31 // (op 3727) 
s_lshl_b32 s47, 1, s19 // (op 3727) 
s_add_i32 s22, s47, s18 // (op 3727) 


s_and_b32 s18, s1, s22 // (op 3727) 


s_add_i32 s1, s0, s18 // (op 3727) 


s_ashr_i32 s0, s1, s19 // (op 3727) 


s_lshl_b32 s1, s13, 24 // (op 3727) 
s_ashr_i32 s47, s1, 31 // (op 3727) 

s_xor_b32 s1, s0, s47 // (op 3727) 

s_sub_i32 s0, s1, s47 // (op 3727) 


s_mul_hi_i32 s1, s0, s14 // (op 3727) 
s_add_i32 s47, s1, s0 // (op 3727) 

s_ashr_i32 s1, s47, 31 // (op 3727) 
s_cmp_eq_i32 s14, 0 // (op 3727) 
s_mov_b32 s18, scc // (op 3727) 
s_cmp_eq_i32 s18, 0 // (op 3727) 
s_cselect_b32 s19, 0, -1 // (op 3727) 

s_and_b32 s18, s15, 31 // (op 3727) 
s_lshl_b32 s22, 1, s18 // (op 3727) 
s_add_i32 s23, s22, s19 // (op 3727) 


s_and_b32 s19, s1, s23 // (op 3727) 


s_add_i32 s1, s47, s19 // (op 3727) 


s_ashr_i32 s19, s1, s18 // (op 3727) 


s_lshl_b32 s1, s15, 24 // (op 3727) 
s_ashr_i32 s18, s1, 31 // (op 3727) 

s_xor_b32 s1, s19, s18 // (op 3727) 

s_sub_i32 s19, s1, s18 // (op 3727) 


s_mul_i32 s1, s0, s11 // (op 3727) 

s_sub_u32 s0, s16, s1 // (op 3727) 

s_mul_i32 s1, s19, s11 // (op 3727) 
s_add_u32 s1, s1, s0 // (op 3727) 


s_mul_hi_i32 s0, s16, s9 // (op 3727) 
s_add_i32 s47, s0, s16 // (op 3727) 

s_ashr_i32 s0, s47, 31 // (op 3727) 
s_cmp_eq_i32 s9, 0 // (op 3727) 
s_mov_b32 s18, scc // (op 3727) 
s_cmp_eq_i32 s18, 0 // (op 3727) 
s_cselect_b32 s19, 0, -1 // (op 3727) 

s_and_b32 s18, s45, 31 // (op 3727) 
s_lshl_b32 s22, 1, s18 // (op 3727) 
s_add_i32 s23, s22, s19 // (op 3727) 


s_and_b32 s19, s0, s23 // (op 3727) 


s_add_i32 s0, s47, s19 // (op 3727) 


s_ashr_i32 s19, s0, s18 // (op 3727) 


s_lshl_b32 s0, s45, 24 // (op 3727) 
s_ashr_i32 s18, s0, 31 // (op 3727) 

s_xor_b32 s0, s19, s18 // (op 3727) 

s_sub_i32 s19, s0, s18 // (op 3727) 


s_add_u32 s22, s4, 32 // (op 3727) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3727) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3727) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3727) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 3727) 

s_mul_i32 s0, s11, s22 // (op 3727) 

s_mul_i32 s18, s8, s0 // (op 3727) 

s_mul_i32 s0, s19, s18 // (op 3727) 

s_sub_u32 s18, s16, s0 // (op 3727) 

s_mov_b32 s0, s18 // (op 3727) 
s_mul_hi_i32 s47, s0, s12 // (op 3727) 
s_add_i32 s22, s47, s0 // (op 3727) 


s_ashr_i32 s0, s22, 31 // (op 3727) 
s_cmp_eq_i32 s12, 0 // (op 3727) 
s_mov_b32 s47, scc // (op 3727) 
s_cmp_eq_i32 s47, 0 // (op 3727) 
s_cselect_b32 s23, 0, -1 // (op 3727) 

s_and_b32 s47, s44, 31 // (op 3727) 
s_lshl_b32 s24, 1, s47 // (op 3727) 
s_add_i32 s25, s24, s23 // (op 3727) 


s_and_b32 s23, s0, s25 // (op 3727) 


s_add_i32 s0, s22, s23 // (op 3727) 


s_ashr_i32 s22, s0, s47 // (op 3727) 


s_lshl_b32 s0, s44, 24 // (op 3727) 
s_ashr_i32 s47, s0, 31 // (op 3727) 

s_xor_b32 s0, s22, s47 // (op 3727) 

s_sub_i32 s22, s0, s47 // (op 3727) 


s_mul_i32 s0, s46, s11 // (op 3727) 
s_add_u32 s24, s6, 256 // (op 3727) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 3727) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 3727) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 3727) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 3727) 

s_sub_i32 s23, s24, s0 // (op 3727) 

s_mul_i32 s47, s22, s23 // (op 3727) 


s_sub_u32 s22, s18, s47 // (op 3727) 


s_add_u32 s18, s22, s0 // (op 3727) 


s_cmp_eq_i32 s19, 0 // (op 3727) 
s_mov_b32 s0, scc // (op 3727) 

s_cmp_eq_i32 s0, 0 // (op 3727) 
s_cselect_b32 s19, s18, s1 // (op 3727) 



v_lshl_add_u32 v16, s19, 8, v1 // (op 3727) 

v_mul_lo_u32 v17, v16, s34 // (op 3727) 

s_mul_i32 s1, s34, s7 // (op 3729) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 3729) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 3729) least significant
s_mul_hi_u32 s18, s34, s6 // (op 3729) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3729) most significant: sum
s_add_u32 s1, s1, s18 // (op 3729) most significant: sum


s_mov_b32 s64, s36 // (op 3729) reg expression
s_mov_b32 s65, s37 // (op 3729) reg expression
s_mov_b32 s66, s0 // (op 3729) reg expression
s_mov_b32 s67, 131072 // (op 3729) call()

s_mul_hi_i32 s47, s16, s10 // (op 3746) 
s_add_i32 s0, s47, s16 // (op 3746) 

s_ashr_i32 s1, s0, 31 // (op 3746) 
s_cmp_eq_i32 s10, 0 // (op 3746) 
s_mov_b32 s47, scc // (op 3746) 
s_cmp_eq_i32 s47, 0 // (op 3746) 
s_cselect_b32 s18, 0, -1 // (op 3746) 

s_and_b32 s19, s13, 31 // (op 3746) 
s_lshl_b32 s47, 1, s19 // (op 3746) 
s_add_i32 s22, s47, s18 // (op 3746) 


s_and_b32 s18, s1, s22 // (op 3746) 


s_add_i32 s1, s0, s18 // (op 3746) 


s_ashr_i32 s0, s1, s19 // (op 3746) 


s_lshl_b32 s1, s13, 24 // (op 3746) 
s_ashr_i32 s47, s1, 31 // (op 3746) 

s_xor_b32 s1, s0, s47 // (op 3746) 

s_sub_i32 s0, s1, s47 // (op 3746) 


s_mul_hi_i32 s1, s0, s14 // (op 3746) 
s_add_i32 s47, s1, s0 // (op 3746) 

s_ashr_i32 s1, s47, 31 // (op 3746) 
s_cmp_eq_i32 s14, 0 // (op 3746) 
s_mov_b32 s18, scc // (op 3746) 
s_cmp_eq_i32 s18, 0 // (op 3746) 
s_cselect_b32 s19, 0, -1 // (op 3746) 

s_and_b32 s18, s15, 31 // (op 3746) 
s_lshl_b32 s22, 1, s18 // (op 3746) 
s_add_i32 s23, s22, s19 // (op 3746) 


s_and_b32 s19, s1, s23 // (op 3746) 


s_add_i32 s1, s47, s19 // (op 3746) 


s_ashr_i32 s19, s1, s18 // (op 3746) 


s_lshl_b32 s1, s15, 24 // (op 3746) 
s_ashr_i32 s18, s1, 31 // (op 3746) 

s_xor_b32 s1, s19, s18 // (op 3746) 

s_sub_i32 s19, s1, s18 // (op 3746) 


s_mul_i32 s1, s0, s11 // (op 3746) 

s_sub_u32 s0, s16, s1 // (op 3746) 

s_mul_i32 s1, s19, s11 // (op 3746) 
s_add_u32 s1, s1, s0 // (op 3746) 


s_mul_hi_i32 s0, s16, s9 // (op 3746) 
s_add_i32 s47, s0, s16 // (op 3746) 

s_ashr_i32 s0, s47, 31 // (op 3746) 
s_cmp_eq_i32 s9, 0 // (op 3746) 
s_mov_b32 s18, scc // (op 3746) 
s_cmp_eq_i32 s18, 0 // (op 3746) 
s_cselect_b32 s19, 0, -1 // (op 3746) 

s_and_b32 s18, s45, 31 // (op 3746) 
s_lshl_b32 s22, 1, s18 // (op 3746) 
s_add_i32 s23, s22, s19 // (op 3746) 


s_and_b32 s19, s0, s23 // (op 3746) 


s_add_i32 s0, s47, s19 // (op 3746) 


s_ashr_i32 s19, s0, s18 // (op 3746) 


s_lshl_b32 s0, s45, 24 // (op 3746) 
s_ashr_i32 s18, s0, 31 // (op 3746) 

s_xor_b32 s0, s19, s18 // (op 3746) 

s_sub_i32 s19, s0, s18 // (op 3746) 


s_add_u32 s22, s4, 32 // (op 3746) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3746) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3746) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3746) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 3746) 

s_mul_i32 s0, s11, s22 // (op 3746) 

s_mul_i32 s18, s8, s0 // (op 3746) 

s_mul_i32 s0, s19, s18 // (op 3746) 

s_sub_u32 s18, s16, s0 // (op 3746) 

s_mov_b32 s0, s18 // (op 3746) 
s_mul_hi_i32 s47, s0, s12 // (op 3746) 
s_add_i32 s22, s47, s0 // (op 3746) 


s_ashr_i32 s0, s22, 31 // (op 3746) 
s_cmp_eq_i32 s12, 0 // (op 3746) 
s_mov_b32 s47, scc // (op 3746) 
s_cmp_eq_i32 s47, 0 // (op 3746) 
s_cselect_b32 s23, 0, -1 // (op 3746) 

s_and_b32 s47, s44, 31 // (op 3746) 
s_lshl_b32 s24, 1, s47 // (op 3746) 
s_add_i32 s25, s24, s23 // (op 3746) 


s_and_b32 s23, s0, s25 // (op 3746) 


s_add_i32 s0, s22, s23 // (op 3746) 


s_ashr_i32 s22, s0, s47 // (op 3746) 


s_lshl_b32 s0, s44, 24 // (op 3746) 
s_ashr_i32 s47, s0, 31 // (op 3746) 

s_xor_b32 s0, s22, s47 // (op 3746) 

s_sub_i32 s22, s0, s47 // (op 3746) 


s_mul_i32 s0, s46, s11 // (op 3746) 
s_add_u32 s24, s6, 256 // (op 3746) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 3746) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 3746) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 3746) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 8 // (op 3746) 

s_sub_i32 s23, s24, s0 // (op 3746) 

s_mul_i32 s47, s22, s23 // (op 3746) 


s_sub_u32 s22, s18, s47 // (op 3746) 


s_add_u32 s18, s22, s0 // (op 3746) 


s_cmp_eq_i32 s19, 0 // (op 3746) 
s_mov_b32 s0, scc // (op 3746) 

s_cmp_eq_i32 s0, 0 // (op 3746) 
s_cselect_b32 s19, s18, s1 // (op 3746) 



v_lshl_add_u32 v16, s19, 8, v1 // (op 3746) 

v_mul_lo_u32 v18, v16, s34 // (op 3746) 

s_mul_i32 s1, s34, s7 // (op 3748) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 3748) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 3748) least significant
s_mul_hi_u32 s18, s34, s6 // (op 3748) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3748) most significant: sum
s_add_u32 s1, s1, s18 // (op 3748) most significant: sum


s_mov_b32 s64, s36 // (op 3748) reg expression
s_mov_b32 s65, s37 // (op 3748) reg expression
s_mov_b32 s66, s0 // (op 3748) reg expression
s_mov_b32 s67, 131072 // (op 3748) call()

s_mul_hi_i32 s47, s16, s10 // (op 3791) 
v_lshrrev_b32 v16, 6, v1 // (op 3791) 
s_add_i32 s0, s47, s16 // (op 3791) 

v_and_b32 v19, 1, v16 // (op 3791) 

s_ashr_i32 s1, s0, 31 // (op 3791) 
v_and_b32 v16, 15, v1 // (op 3791) 
s_cmp_eq_i32 s10, 0 // (op 3791) 
s_mov_b32 s47, scc // (op 3791) 
v_lshl_add_u32 v20, v19, 4, v16 // (op 3791) 


s_cmp_eq_i32 s47, 0 // (op 3791) 
s_cselect_b32 s18, 0, -1 // (op 3791) 

v_and_b32 v16, 63, v1 // (op 3791) 
s_and_b32 s19, s13, 31 // (op 3791) 
v_lshrrev_b32 v19, 4, v16 // (op 3791) 

s_lshl_b32 s47, 1, s19 // (op 3791) 
s_add_i32 s22, s47, s18 // (op 3791) 


s_and_b32 s18, s1, s22 // (op 3791) 


s_add_i32 s1, s0, s18 // (op 3791) 


s_ashr_i32 s0, s1, s19 // (op 3791) 


s_lshl_b32 s1, s13, 24 // (op 3791) 
s_ashr_i32 s47, s1, 31 // (op 3791) 

s_xor_b32 s1, s0, s47 // (op 3791) 

s_sub_i32 s0, s1, s47 // (op 3791) 


s_mul_hi_i32 s1, s0, s14 // (op 3791) 
s_add_i32 s47, s1, s0 // (op 3791) 

s_ashr_i32 s1, s47, 31 // (op 3791) 
s_cmp_eq_i32 s14, 0 // (op 3791) 
s_mov_b32 s18, scc // (op 3791) 
s_cmp_eq_i32 s18, 0 // (op 3791) 
s_cselect_b32 s19, 0, -1 // (op 3791) 

s_and_b32 s18, s15, 31 // (op 3791) 
s_lshl_b32 s22, 1, s18 // (op 3791) 
s_add_i32 s23, s22, s19 // (op 3791) 


s_and_b32 s19, s1, s23 // (op 3791) 


s_add_i32 s1, s47, s19 // (op 3791) 


s_ashr_i32 s19, s1, s18 // (op 3791) 


s_lshl_b32 s1, s15, 24 // (op 3791) 
s_ashr_i32 s18, s1, 31 // (op 3791) 

s_xor_b32 s1, s19, s18 // (op 3791) 

s_sub_i32 s19, s1, s18 // (op 3791) 


s_add_u32 s22, s4, 32 // (op 3791) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3791) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3791) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3791) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 3791) 

s_mul_i32 s1, s19, s22 // (op 3791) 

s_sub_i32 s47, s0, s1 // (op 3791) 


s_mul_hi_i32 s0, s16, s9 // (op 3791) 
s_add_i32 s1, s0, s16 // (op 3791) 

s_ashr_i32 s0, s1, 31 // (op 3791) 
s_cmp_eq_i32 s9, 0 // (op 3791) 
s_mov_b32 s18, scc // (op 3791) 
s_cmp_eq_i32 s18, 0 // (op 3791) 
s_cselect_b32 s19, 0, -1 // (op 3791) 

s_and_b32 s18, s45, 31 // (op 3791) 
s_lshl_b32 s24, 1, s18 // (op 3791) 
s_add_i32 s25, s24, s19 // (op 3791) 


s_and_b32 s19, s0, s25 // (op 3791) 


s_add_i32 s0, s1, s19 // (op 3791) 


s_ashr_i32 s1, s0, s18 // (op 3791) 


s_lshl_b32 s0, s45, 24 // (op 3791) 
s_ashr_i32 s18, s0, 31 // (op 3791) 

s_xor_b32 s0, s1, s18 // (op 3791) 

s_sub_i32 s1, s0, s18 // (op 3791) 


s_mul_i32 s0, s11, s22 // (op 3791) 

s_mul_i32 s18, s8, s0 // (op 3791) 

s_mul_i32 s0, s1, s18 // (op 3791) 

s_sub_u32 s18, s16, s0 // (op 3791) 

s_mul_hi_i32 s0, s18, s12 // (op 3791) 
s_add_i32 s19, s0, s18 // (op 3791) 


s_ashr_i32 s0, s19, 31 // (op 3791) 
s_cmp_eq_i32 s12, 0 // (op 3791) 
s_mov_b32 s18, scc // (op 3791) 
s_cmp_eq_i32 s18, 0 // (op 3791) 
s_cselect_b32 s22, 0, -1 // (op 3791) 

s_and_b32 s18, s44, 31 // (op 3791) 
s_lshl_b32 s23, 1, s18 // (op 3791) 
s_add_i32 s24, s23, s22 // (op 3791) 


s_and_b32 s25, s0, s24 // (op 3791) 


s_add_i32 s0, s19, s25 // (op 3791) 


s_ashr_i32 s19, s0, s18 // (op 3791) 


s_lshl_b32 s0, s44, 24 // (op 3791) 
s_ashr_i32 s18, s0, 31 // (op 3791) 

s_xor_b32 s0, s19, s18 // (op 3791) 

s_sub_i32 s19, s0, s18 // (op 3791) 


s_cmp_eq_i32 s1, 0 // (op 3791) 
s_mov_b32 s0, scc // (op 3791) 

s_cmp_eq_i32 s0, 0 // (op 3791) 
s_cselect_b32 s1, s19, s47 // (op 3791) 



v_lshl_add_u32 v16, s1, 5, v20 // (op 3791) 


v_mul_lo_u32 v20, v16, s38 // (op 3791) 

v_add_u32 v16, v19, v20 // (op 3791) 


s_mul_i32 s1, s38, s5 // (op 3793) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 3793) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 3793) least significant
s_mul_hi_u32 s18, s38, s4 // (op 3793) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3793) most significant: sum
s_add_u32 s1, s1, s18 // (op 3793) most significant: sum


s_mov_b32 s68, s40 // (op 3793) reg expression
s_mov_b32 s69, s41 // (op 3793) reg expression
s_mov_b32 s70, s0 // (op 3793) reg expression
s_mov_b32 s71, 131072 // (op 3793) call()

s_mul_hi_i32 s47, s16, s10 // (op 3810) 
v_lshrrev_b32 v19, 6, v1 // (op 3810) 
s_add_i32 s0, s47, s16 // (op 3810) 

v_and_b32 v20, 1, v19 // (op 3810) 

s_ashr_i32 s1, s0, 31 // (op 3810) 
v_and_b32 v19, 15, v1 // (op 3810) 
s_cmp_eq_i32 s10, 0 // (op 3810) 
s_mov_b32 s47, scc // (op 3810) 
v_lshl_add_u32 v21, v20, 4, v19 // (op 3810) 


s_cmp_eq_i32 s47, 0 // (op 3810) 
s_cselect_b32 s18, 0, -1 // (op 3810) 

v_and_b32 v19, 63, v1 // (op 3810) 
s_and_b32 s19, s13, 31 // (op 3810) 
v_lshrrev_b32 v20, 4, v19 // (op 3810) 

s_lshl_b32 s47, 1, s19 // (op 3810) 
s_add_i32 s22, s47, s18 // (op 3810) 


s_and_b32 s18, s1, s22 // (op 3810) 


s_add_i32 s1, s0, s18 // (op 3810) 


s_ashr_i32 s0, s1, s19 // (op 3810) 


s_lshl_b32 s1, s13, 24 // (op 3810) 
s_ashr_i32 s47, s1, 31 // (op 3810) 

s_xor_b32 s1, s0, s47 // (op 3810) 

s_sub_i32 s0, s1, s47 // (op 3810) 


s_mul_hi_i32 s1, s0, s14 // (op 3810) 
s_add_i32 s47, s1, s0 // (op 3810) 

s_ashr_i32 s1, s47, 31 // (op 3810) 
s_cmp_eq_i32 s14, 0 // (op 3810) 
s_mov_b32 s18, scc // (op 3810) 
s_cmp_eq_i32 s18, 0 // (op 3810) 
s_cselect_b32 s19, 0, -1 // (op 3810) 

s_and_b32 s18, s15, 31 // (op 3810) 
s_lshl_b32 s22, 1, s18 // (op 3810) 
s_add_i32 s23, s22, s19 // (op 3810) 


s_and_b32 s19, s1, s23 // (op 3810) 


s_add_i32 s1, s47, s19 // (op 3810) 


s_ashr_i32 s19, s1, s18 // (op 3810) 


s_lshl_b32 s1, s15, 24 // (op 3810) 
s_ashr_i32 s18, s1, 31 // (op 3810) 

s_xor_b32 s1, s19, s18 // (op 3810) 

s_sub_i32 s19, s1, s18 // (op 3810) 


s_add_u32 s22, s4, 32 // (op 3810) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3810) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3810) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3810) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 3810) 

s_mul_i32 s1, s19, s22 // (op 3810) 

s_sub_i32 s47, s0, s1 // (op 3810) 


s_mul_hi_i32 s0, s16, s9 // (op 3810) 
s_add_i32 s1, s0, s16 // (op 3810) 

s_ashr_i32 s0, s1, 31 // (op 3810) 
s_cmp_eq_i32 s9, 0 // (op 3810) 
s_mov_b32 s18, scc // (op 3810) 
s_cmp_eq_i32 s18, 0 // (op 3810) 
s_cselect_b32 s19, 0, -1 // (op 3810) 

s_and_b32 s18, s45, 31 // (op 3810) 
s_lshl_b32 s24, 1, s18 // (op 3810) 
s_add_i32 s25, s24, s19 // (op 3810) 


s_and_b32 s19, s0, s25 // (op 3810) 


s_add_i32 s0, s1, s19 // (op 3810) 


s_ashr_i32 s1, s0, s18 // (op 3810) 


s_lshl_b32 s0, s45, 24 // (op 3810) 
s_ashr_i32 s18, s0, 31 // (op 3810) 

s_xor_b32 s0, s1, s18 // (op 3810) 

s_sub_i32 s1, s0, s18 // (op 3810) 


s_mul_i32 s0, s11, s22 // (op 3810) 

s_mul_i32 s18, s8, s0 // (op 3810) 

s_mul_i32 s0, s1, s18 // (op 3810) 

s_sub_u32 s18, s16, s0 // (op 3810) 

s_mul_hi_i32 s0, s18, s12 // (op 3810) 
s_add_i32 s19, s0, s18 // (op 3810) 


s_ashr_i32 s0, s19, 31 // (op 3810) 
s_cmp_eq_i32 s12, 0 // (op 3810) 
s_mov_b32 s18, scc // (op 3810) 
s_cmp_eq_i32 s18, 0 // (op 3810) 
s_cselect_b32 s22, 0, -1 // (op 3810) 

s_and_b32 s18, s44, 31 // (op 3810) 
s_lshl_b32 s23, 1, s18 // (op 3810) 
s_add_i32 s24, s23, s22 // (op 3810) 


s_and_b32 s25, s0, s24 // (op 3810) 


s_add_i32 s0, s19, s25 // (op 3810) 


s_ashr_i32 s19, s0, s18 // (op 3810) 


s_lshl_b32 s0, s44, 24 // (op 3810) 
s_ashr_i32 s18, s0, 31 // (op 3810) 

s_xor_b32 s0, s19, s18 // (op 3810) 

s_sub_i32 s19, s0, s18 // (op 3810) 


s_cmp_eq_i32 s1, 0 // (op 3810) 
s_mov_b32 s0, scc // (op 3810) 

s_cmp_eq_i32 s0, 0 // (op 3810) 
s_cselect_b32 s1, s19, s47 // (op 3810) 



v_lshl_add_u32 v19, s1, 5, v21 // (op 3810) 


v_mul_lo_u32 v21, v19, s38 // (op 3810) 

v_add_u32 v19, v20, v21 // (op 3810) 


s_mul_i32 s1, s38, s5 // (op 3812) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 3812) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 3812) least significant
s_mul_hi_u32 s18, s38, s4 // (op 3812) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3812) most significant: sum
s_add_u32 s1, s1, s18 // (op 3812) most significant: sum


s_mov_b32 s68, s40 // (op 3812) reg expression
s_mov_b32 s69, s41 // (op 3812) reg expression
s_mov_b32 s70, s0 // (op 3812) reg expression
s_mov_b32 s71, 131072 // (op 3812) call()




buffer_load_ubyte v20, v16, s[68:71], 0 offen offset: 0 // (op 3284) Load value



buffer_load_dwordx2 v[22:23], v14, s[60:63], 0 offen // (op 3288) Load value

v_mov_b64 v[24:25], s[26:27] // (op 3291) convert

v_mov_b32 v21, v12 // (op 3291) 
buffer_load_dwordx4 v[26:29], v21, s[48:51], 0 offen // (op 3291) Load value
v_add_u32 v21, v21, v24 // (op 3291) 
buffer_load_dwordx4 v[30:33], v21, s[48:51], 0 offen // (op 3291) Load value
v_add_u32 v21, v21, v24 // (op 3291) 
buffer_load_dwordx4 v[34:37], v21, s[48:51], 0 offen // (op 3291) Load value
v_add_u32 v21, v21, v24 // (op 3291) 
buffer_load_dwordx4 v[38:41], v21, s[48:51], 0 offen // (op 3291) Load value
v_add_u32 v21, v21, v24 // (op 3291) 
buffer_load_dwordx4 v[42:45], v21, s[48:51], 0 offen // (op 3291) Load value
v_add_u32 v21, v21, v24 // (op 3291) 
buffer_load_dwordx4 v[46:49], v21, s[48:51], 0 offen // (op 3291) Load value
v_add_u32 v21, v21, v24 // (op 3291) 
buffer_load_dwordx4 v[50:53], v21, s[48:51], 0 offen // (op 3291) Load value
v_add_u32 v21, v21, v24 // (op 3291) 
buffer_load_dwordx4 v[54:57], v21, s[48:51], 0 offen // (op 3291) Load value





buffer_load_dword v21, v17, s[64:67], 0 offen // (op 3294) Load value



buffer_load_ubyte v24, v16, s[68:71], 0 offen offset:4 // (op 3298) Load value




buffer_load_dwordx2 v[58:59], v14, s[60:63], 0 offen offset:64 // (op 3302) Load value


v_mov_b64 v[60:61], s[26:27] // (op 3305) convert

v_mov_b32 v14, v12 // (op 3305) 
buffer_load_dwordx4 v[62:65], v14, s[48:51], 0 offen offset:128 // (op 3305) Load value
v_add_u32 v14, v14, v60 // (op 3305) 
buffer_load_dwordx4 v[66:69], v14, s[48:51], 0 offen offset:128 // (op 3305) Load value
v_add_u32 v14, v14, v60 // (op 3305) 
buffer_load_dwordx4 v[70:73], v14, s[48:51], 0 offen offset:128 // (op 3305) Load value
v_add_u32 v14, v14, v60 // (op 3305) 
buffer_load_dwordx4 v[74:77], v14, s[48:51], 0 offen offset:128 // (op 3305) Load value
v_add_u32 v14, v14, v60 // (op 3305) 
buffer_load_dwordx4 v[78:81], v14, s[48:51], 0 offen offset:128 // (op 3305) Load value
v_add_u32 v14, v14, v60 // (op 3305) 
buffer_load_dwordx4 v[82:85], v14, s[48:51], 0 offen offset:128 // (op 3305) Load value
v_add_u32 v14, v14, v60 // (op 3305) 
buffer_load_dwordx4 v[86:89], v14, s[48:51], 0 offen offset:128 // (op 3305) Load value
v_add_u32 v14, v14, v60 // (op 3305) 
buffer_load_dwordx4 v[90:93], v14, s[48:51], 0 offen offset:128 // (op 3305) Load value






buffer_load_dword v12, v17, s[64:67], 0 offen offset:4 // (op 3308) Load value

v_and_b32 v14, -8, v1 // (op 3935) 
v_sub_u32 v25, v1, v14 // (op 3935) 

v_lshlrev_b32 v14, 4, v25 // (op 3935) 

v_lshrrev_b32 v25, 3, v1 // (op 3935) 
v_lshl_add_u32 v16, v25, 7, v14 // (op 3935) 


v_lshlrev_b32 v14, 2, v16 // (op 3935) 

v_lshrrev_b32 v25, 3, v14 // (op 3935) 




s_waitcnt vmcnt(20)
ds_write_b64 v25, v[22:23] // (op 3310) Store local data 

v_and_b32 v14, -8, v1 // (op 3877) 
v_sub_u32 v25, v1, v14 // (op 3877) 

v_lshlrev_b32 v14, 4, v25 // (op 3877) 

v_lshrrev_b32 v25, 3, v1 // (op 3877) 
v_lshl_add_u32 v16, v25, 10, v14 // (op 3877) 





s_waitcnt vmcnt(19)
ds_write_b128 v16, v[26:29] offset:2048 // (op 3311) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v16, v[30:33] offset:2176 // (op 3311) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v16, v[34:37] offset:2304 // (op 3311) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v16, v[38:41] offset:2432 // (op 3311) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v16, v[42:45] offset:2560 // (op 3311) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v16, v[46:49] offset:2688 // (op 3311) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v16, v[50:53] offset:2816 // (op 3311) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v16, v[54:57] offset:2944 // (op 3311) Store local data 

v_lshlrev_b32 v14, 2, v1 // (op 3988) 



s_waitcnt vmcnt(11)
ds_write_b32 v14, v21 offset:34816 // (op 3312) Store local data 

v_and_b32 v14, -8, v1 // (op 3848) 
v_sub_u32 v25, v1, v14 // (op 3848) 

v_lshlrev_b32 v14, 4, v25 // (op 3848) 

v_lshrrev_b32 v25, 3, v1 // (op 3848) 
v_lshl_add_u32 v16, v25, 10, v14 // (op 3848) 


v_and_b32 v14, -8, v1 // (op 3906) 
v_sub_u32 v17, v1, v14 // (op 3906) 

v_lshlrev_b32 v14, 4, v17 // (op 3906) 

v_lshrrev_b32 v17, 3, v1 // (op 3906) 
v_lshl_add_u32 v25, v17, 7, v14 // (op 3906) 


v_lshlrev_b32 v14, 2, v25 // (op 3906) 

v_lshrrev_b32 v17, 3, v14 // (op 3906) 

v_lshlrev_b32 v14, 2, v1 // (op 3959) 
v_and_b32 v25, -8, v1 // (op 4035) 
v_sub_u32 v60, v1, v25 // (op 4035) 

v_lshlrev_b32 v25, 4, v60 // (op 4035) 

v_lshrrev_b32 v60, 3, v1 // (op 4035) 
v_lshl_add_u32 v61, v60, 10, v25 // (op 4035) 


v_and_b32 v25, -8, v1 // (op 4064) 
v_sub_u32 v60, v1, v25 // (op 4064) 

v_lshlrev_b32 v25, 4, v60 // (op 4064) 

v_lshrrev_b32 v60, 3, v1 // (op 4064) 
v_lshl_add_u32 v94, v60, 7, v25 // (op 4064) 


v_lshlrev_b32 v25, 2, v94 // (op 4064) 

v_lshrrev_b32 v60, 3, v25 // (op 4064) 

v_lshlrev_b32 v25, 2, v1 // (op 4088) 
s_mov_b32 s47, 0 // (op 1287) call()
s_ashr_i64 s[0:1], s[42:43], 7 // (op 1286) 
s_mov_b32 s18, s0 // (op 1286) convert

s_and_b32 s19, s18, 4294967294 // (op 1286) 

s_cmp_lt_u32 s47, s19 // (op 1286) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopBottom_KLoop_1286 // (op 1286) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopBottom_KLoop_1286 if false)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopTop_KLoop_1286:
s_add_u32 s47, s47, 2 // (op 1288) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 4164) 



ds_read_b128 v[94:97], v7 offset:2048 // (op 67) Load local data 
ds_read_b128 v[98:101], v7 offset:2112 // (op 67) Load local data 
s_barrier  // (op 4177) 



ds_read_b128 v[102:105], v3 // (op 72) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 4191) 



ds_read_u8 v106, v0 offset:34816 // (op 77) Load local data 



ds_read_b128 v[108:111], v7 offset:6144 // (op 344) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[94:101], v[102:105], a[0:3], v106, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 84) 
ds_read_b128 v[112:115], v7 offset:6208 // (op 344) Load local data 





ds_read_u8 v94, v0 offset:34944 // (op 348) Load local data 



ds_read_b128 v[116:119], v7 offset:10240 // (op 506) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[108:115], v[102:105], a[4:7], v94, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 350) 
ds_read_b128 v[120:123], v7 offset:10304 // (op 506) Load local data 





ds_read_u8 v94, v0 offset:35072 // (op 510) Load local data 



ds_read_b128 v[106:109], v7 offset:14336 // (op 668) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[116:123], v[102:105], a[8:11], v94, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 512) 
ds_read_b128 v[110:113], v7 offset:14400 // (op 668) Load local data 





ds_read_u8 v94, v0 offset:35200 // (op 672) Load local data 



ds_read_b128 v[114:117], v7 offset:18432 // (op 830) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[106:113], v[102:105], a[12:15], v94, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 674) 
ds_read_b128 v[118:121], v7 offset:18496 // (op 830) Load local data 





ds_read_u8 v94, v0 offset:35328 // (op 834) Load local data 



ds_read_b128 v[106:109], v7 offset:22528 // (op 992) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[114:121], v[102:105], a[16:19], v94, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 836) 
ds_read_b128 v[110:113], v7 offset:22592 // (op 992) Load local data 





ds_read_u8 v94, v0 offset:35456 // (op 996) Load local data 



ds_read_b128 v[114:117], v7 offset:26624 // (op 1154) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[106:113], v[102:105], a[20:23], v94, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 998) 
ds_read_b128 v[118:121], v7 offset:26688 // (op 1154) Load local data 





ds_read_u8 v94, v0 offset:35584 // (op 1158) Load local data 



ds_read_b128 v[106:109], v7 offset:30720 // (op 1316) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[114:121], v[102:105], a[24:27], v94, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1160) 
ds_read_b128 v[110:113], v7 offset:30784 // (op 1316) Load local data 





ds_read_u8 v94, v0 offset:35712 // (op 1320) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[106:113], v[102:105], a[28:31], v94, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1322) 





v_mov_b32 v94, v19 // (op 82) 

buffer_load_ubyte v20, v94, s[68:71], 0 offen offset:8 // (op 82) Load value


v_mov_b32 v95, v15 // (op 10) 

buffer_load_dwordx2 v[22:23], v95, s[60:63], 0 offen offset:128 // (op 10) Load value

v_mov_b64 v[96:97], s[26:27] // (op 4) convert

v_mov_b32 v98, v13 // (op 4) 
v_mov_b32 v99, v98 // (op 4) 
buffer_load_dwordx4 v[26:29], v99, s[48:51], 0 offen offset:256 // (op 4) Load value
v_add_u32 v99, v99, v96 // (op 4) 
buffer_load_dwordx4 v[30:33], v99, s[48:51], 0 offen offset:256 // (op 4) Load value
v_add_u32 v99, v99, v96 // (op 4) 
buffer_load_dwordx4 v[34:37], v99, s[48:51], 0 offen offset:256 // (op 4) Load value
v_add_u32 v99, v99, v96 // (op 4) 
buffer_load_dwordx4 v[38:41], v99, s[48:51], 0 offen offset:256 // (op 4) Load value
v_add_u32 v99, v99, v96 // (op 4) 
buffer_load_dwordx4 v[42:45], v99, s[48:51], 0 offen offset:256 // (op 4) Load value
v_add_u32 v99, v99, v96 // (op 4) 
buffer_load_dwordx4 v[46:49], v99, s[48:51], 0 offen offset:256 // (op 4) Load value
v_add_u32 v99, v99, v96 // (op 4) 
buffer_load_dwordx4 v[50:53], v99, s[48:51], 0 offen offset:256 // (op 4) Load value
v_add_u32 v99, v99, v96 // (op 4) 
buffer_load_dwordx4 v[54:57], v99, s[48:51], 0 offen offset:256 // (op 4) Load value




v_mov_b32 v96, v18 // (op 17) 

buffer_load_dword v21, v96, s[64:67], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 4214) 



s_waitcnt vmcnt(20)
ds_write_b64 v60, v[58:59] offset:35840 // (op 239) Store local data 
s_barrier  // (op 4208) 



s_waitcnt vmcnt(19)
ds_write_b128 v61, v[62:65] offset:37888 // (op 229) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v61, v[66:69] offset:38016 // (op 229) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v61, v[70:73] offset:38144 // (op 229) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v61, v[74:77] offset:38272 // (op 229) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v61, v[78:81] offset:38400 // (op 229) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v61, v[82:85] offset:38528 // (op 229) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v61, v[86:89] offset:38656 // (op 229) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v61, v[90:93] offset:38784 // (op 229) Store local data 
s_barrier  // (op 4220) 



v_add_u32 v97, 70656, v25 // (op 240) 
s_waitcnt vmcnt(11)
ds_write_b32 v97, v12 // (op 240) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 4205) 



ds_read_b128 v[100:103], v11 offset:37888 // (op 231) Load local data 
ds_read_b128 v[104:107], v11 offset:37952 // (op 231) Load local data 
s_barrier  // (op 4211) 



ds_read_b128 v[108:111], v4 offset:35840 // (op 233) Load local data 
s_barrier  // (op 4261) 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 4217) 



v_add_u32 v97, 70656, v10 // (op 235) 
ds_read_u8 v99, v97 // (op 235) Load local data 




ds_read_b128 v[112:115], v11 offset:41984 // (op 357) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[100:107], v[108:111], a[0:3], v99, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 238) 
ds_read_b128 v[116:119], v11 offset:42048 // (op 357) Load local data 





v_add_u32 v97, 70784, v10 // (op 361) 
ds_read_u8 v99, v97 // (op 361) Load local data 




ds_read_b128 v[100:103], v11 offset:46080 // (op 519) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[112:119], v[108:111], a[4:7], v99, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 351) 
ds_read_b128 v[104:107], v11 offset:46144 // (op 519) Load local data 





v_add_u32 v97, 70912, v10 // (op 523) 
ds_read_u8 v99, v97 // (op 523) Load local data 




ds_read_b128 v[112:115], v11 offset:50176 // (op 681) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[100:107], v[108:111], a[8:11], v99, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 513) 
ds_read_b128 v[116:119], v11 offset:50240 // (op 681) Load local data 





v_add_u32 v97, 71040, v10 // (op 685) 
ds_read_u8 v99, v97 // (op 685) Load local data 




ds_read_b128 v[100:103], v11 offset:54272 // (op 843) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:119], v[108:111], a[12:15], v99, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 675) 
ds_read_b128 v[104:107], v11 offset:54336 // (op 843) Load local data 





v_add_u32 v97, 71168, v10 // (op 847) 
ds_read_u8 v99, v97 // (op 847) Load local data 




ds_read_b128 v[112:115], v11 offset:58368 // (op 1005) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[100:107], v[108:111], a[16:19], v99, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 837) 
ds_read_b128 v[116:119], v11 offset:58432 // (op 1005) Load local data 





v_add_u32 v97, 71296, v10 // (op 1009) 
ds_read_u8 v99, v97 // (op 1009) Load local data 




ds_read_b128 v[100:103], v11 offset:62464 // (op 1167) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[112:119], v[108:111], a[20:23], v99, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 999) 
ds_read_b128 v[104:107], v11 offset:62528 // (op 1167) Load local data 





v_add_u32 v97, 71424, v10 // (op 1171) 
ds_read_u8 v99, v97 // (op 1171) Load local data 




v_add_u32 v97, 66560, v11 // (op 1329) 
ds_read_b128 v[112:115], v97 // (op 1329) Load local data 

v_add_u32 v97, 66624, v11 // (op 1329) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[100:107], v[108:111], a[24:27], v99, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1161) 
ds_read_b128 v[116:119], v97 // (op 1329) Load local data 






s_barrier  // (op 4331) 
v_add_u32 v97, 71552, v10 // (op 1333) 
ds_read_u8 v99, v97 // (op 1333) Load local data 

s_barrier  // (op 4335) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[112:119], v[108:111], a[28:31], v99, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1323) 






buffer_load_ubyte v24, v94, s[68:71], 0 offen offset:12 // (op 237) Load value




v_add_u32 v19, 8, v19 // (op 3802) 
buffer_load_dwordx2 v[58:59], v95, s[60:63], 0 offen offset:192 // (op 241) Load value


v_add_u32 v15, 128, v15 // (op 3674) 
v_mov_b64 v[94:95], s[26:27] // (op 228) convert

v_mov_b32 v97, v98 // (op 228) 
buffer_load_dwordx4 v[62:65], v97, s[48:51], 0 offen offset:384 // (op 228) Load value
v_add_u32 v97, v97, v94 // (op 228) 
buffer_load_dwordx4 v[66:69], v97, s[48:51], 0 offen offset:384 // (op 228) Load value
v_add_u32 v97, v97, v94 // (op 228) 
buffer_load_dwordx4 v[70:73], v97, s[48:51], 0 offen offset:384 // (op 228) Load value
v_add_u32 v97, v97, v94 // (op 228) 
buffer_load_dwordx4 v[74:77], v97, s[48:51], 0 offen offset:384 // (op 228) Load value
v_add_u32 v97, v97, v94 // (op 228) 
buffer_load_dwordx4 v[78:81], v97, s[48:51], 0 offen offset:384 // (op 228) Load value
v_add_u32 v97, v97, v94 // (op 228) 
buffer_load_dwordx4 v[82:85], v97, s[48:51], 0 offen offset:384 // (op 228) Load value
v_add_u32 v97, v97, v94 // (op 228) 
buffer_load_dwordx4 v[86:89], v97, s[48:51], 0 offen offset:384 // (op 228) Load value
v_add_u32 v97, v97, v94 // (op 228) 
buffer_load_dwordx4 v[90:93], v97, s[48:51], 0 offen offset:384 // (op 228) Load value






v_add_u32 v13, 256, v13 // (op 3610) 
buffer_load_dword v12, v96, s[64:67], 0 offen offset:12 // (op 242) Load value

s_barrier  // (op 4180) 
v_add_u32 v18, 8, v18 // (op 3738) 



s_waitcnt vmcnt(20)
ds_write_b64 v17, v[22:23] // (op 13) Store local data 
s_barrier  // (op 4167) 



s_waitcnt vmcnt(19)
ds_write_b128 v16, v[26:29] offset:2048 // (op 6) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v16, v[30:33] offset:2176 // (op 6) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v16, v[34:37] offset:2304 // (op 6) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v16, v[38:41] offset:2432 // (op 6) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v16, v[42:45] offset:2560 // (op 6) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v16, v[46:49] offset:2688 // (op 6) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v16, v[50:53] offset:2816 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v16, v[54:57] offset:2944 // (op 6) Store local data 
s_barrier  // (op 4194) 



s_waitcnt vmcnt(11)
ds_write_b32 v14, v21 offset:34816 // (op 21) Store local data 
s_ashr_i64 s[0:1], s[42:43], 7 // (op 1286) 
s_mov_b32 s18, s0 // (op 1286) convert

s_and_b32 s19, s18, 4294967294 // (op 1286) 

s_cmp_lt_u32 s47, s19 // (op 1286) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopTop_KLoop_1286 // (op 1286) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopTop_KLoop_1286 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopBottom_KLoop_1286:















s_mul_hi_i32 s47, s16, s10 // (op 3566) 
v_and_b32 v13, 4294967288, v1 // (op 3566) 
s_add_i32 s0, s47, s16 // (op 3566) 

v_and_b32 v14, -8, v1 // (op 3566) 
s_ashr_i32 s1, s0, 31 // (op 3566) 
v_sub_u32 v15, v1, v14 // (op 3566) 

s_cmp_eq_i32 s10, 0 // (op 3566) 
s_mov_b32 s47, scc // (op 3566) 
v_lshlrev_b32 v14, 4, v15 // (op 3566) 

s_cmp_eq_i32 s47, 0 // (op 3566) 
s_cselect_b32 s18, 0, -1 // (op 3566) 

s_and_b32 s19, s13, 31 // (op 3566) 
s_lshl_b32 s47, 1, s19 // (op 3566) 
s_add_i32 s22, s47, s18 // (op 3566) 


s_and_b32 s18, s1, s22 // (op 3566) 


s_add_i32 s1, s0, s18 // (op 3566) 


s_ashr_i32 s0, s1, s19 // (op 3566) 


s_lshl_b32 s1, s13, 24 // (op 3566) 
s_ashr_i32 s47, s1, 31 // (op 3566) 

s_xor_b32 s1, s0, s47 // (op 3566) 

s_sub_i32 s0, s1, s47 // (op 3566) 


s_mul_hi_i32 s1, s0, s14 // (op 3566) 
s_add_i32 s47, s1, s0 // (op 3566) 

s_ashr_i32 s1, s47, 31 // (op 3566) 
s_cmp_eq_i32 s14, 0 // (op 3566) 
s_mov_b32 s18, scc // (op 3566) 
s_cmp_eq_i32 s18, 0 // (op 3566) 
s_cselect_b32 s19, 0, -1 // (op 3566) 

s_and_b32 s18, s15, 31 // (op 3566) 
s_lshl_b32 s22, 1, s18 // (op 3566) 
s_add_i32 s23, s22, s19 // (op 3566) 


s_and_b32 s19, s1, s23 // (op 3566) 


s_add_i32 s1, s47, s19 // (op 3566) 


s_ashr_i32 s19, s1, s18 // (op 3566) 


s_lshl_b32 s1, s15, 24 // (op 3566) 
s_ashr_i32 s18, s1, 31 // (op 3566) 

s_xor_b32 s1, s19, s18 // (op 3566) 

s_sub_i32 s19, s1, s18 // (op 3566) 


s_mul_i32 s1, s0, s11 // (op 3566) 

s_sub_u32 s0, s16, s1 // (op 3566) 

s_mul_i32 s1, s19, s11 // (op 3566) 
s_add_u32 s1, s1, s0 // (op 3566) 


s_mul_hi_i32 s0, s16, s9 // (op 3566) 
s_add_i32 s47, s0, s16 // (op 3566) 

s_ashr_i32 s0, s47, 31 // (op 3566) 
s_cmp_eq_i32 s9, 0 // (op 3566) 
s_mov_b32 s18, scc // (op 3566) 
s_cmp_eq_i32 s18, 0 // (op 3566) 
s_cselect_b32 s19, 0, -1 // (op 3566) 

s_and_b32 s18, s45, 31 // (op 3566) 
s_lshl_b32 s22, 1, s18 // (op 3566) 
s_add_i32 s23, s22, s19 // (op 3566) 


s_and_b32 s19, s0, s23 // (op 3566) 


s_add_i32 s0, s47, s19 // (op 3566) 


s_ashr_i32 s19, s0, s18 // (op 3566) 


s_lshl_b32 s0, s45, 24 // (op 3566) 
s_ashr_i32 s18, s0, 31 // (op 3566) 

s_xor_b32 s0, s19, s18 // (op 3566) 

s_sub_i32 s19, s0, s18 // (op 3566) 


s_add_u32 s22, s4, 32 // (op 3566) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3566) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3566) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3566) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 3566) 

s_mul_i32 s0, s11, s22 // (op 3566) 

s_mul_i32 s18, s8, s0 // (op 3566) 

s_mul_i32 s0, s19, s18 // (op 3566) 

s_sub_u32 s18, s16, s0 // (op 3566) 

s_mov_b32 s0, s18 // (op 3566) 
s_mul_hi_i32 s47, s0, s12 // (op 3566) 
s_add_i32 s22, s47, s0 // (op 3566) 


s_ashr_i32 s0, s22, 31 // (op 3566) 
s_cmp_eq_i32 s12, 0 // (op 3566) 
s_mov_b32 s47, scc // (op 3566) 
s_cmp_eq_i32 s47, 0 // (op 3566) 
s_cselect_b32 s23, 0, -1 // (op 3566) 

s_and_b32 s47, s44, 31 // (op 3566) 
s_lshl_b32 s24, 1, s47 // (op 3566) 
s_add_i32 s25, s24, s23 // (op 3566) 


s_and_b32 s23, s0, s25 // (op 3566) 


s_add_i32 s0, s22, s23 // (op 3566) 


s_ashr_i32 s22, s0, s47 // (op 3566) 


s_lshl_b32 s0, s44, 24 // (op 3566) 
s_ashr_i32 s47, s0, 31 // (op 3566) 

s_xor_b32 s0, s22, s47 // (op 3566) 

s_sub_i32 s22, s0, s47 // (op 3566) 


s_mul_i32 s0, s46, s11 // (op 3566) 
s_add_u32 s24, s6, 256 // (op 3566) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 3566) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 3566) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 3566) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 8 // (op 3566) 

s_sub_i32 s23, s24, s0 // (op 3566) 

s_mul_i32 s47, s22, s23 // (op 3566) 


s_sub_u32 s22, s18, s47 // (op 3566) 


s_add_u32 s18, s22, s0 // (op 3566) 


s_cmp_eq_i32 s19, 0 // (op 3566) 
s_mov_b32 s0, scc // (op 3566) 

s_cmp_eq_i32 s0, 0 // (op 3566) 
s_cselect_b32 s19, s18, s1 // (op 3566) 



v_lshl_add_u32 v15, s19, 8, v13 // (op 3566) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 3566) 
s_lshr_b32 s47, s0, 1 // (op 3566) 

v_lshl_add_u32 v13, s47, 8, v14 // (op 3566) 


v_mul_lo_u32 v14, v15, s26 // (op 3566) 
v_add_u32 v14, v14, v13 // (op 3566) 


s_mul_i32 s1, s26, s7 // (op 3568) most significant: low of low * high
s_mul_i32 s47, s27, s6 // (op 3568) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 3568) least significant
s_mul_hi_u32 s18, s26, s6 // (op 3568) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3568) most significant: sum
s_add_u32 s1, s1, s18 // (op 3568) most significant: sum


s_mov_b32 s48, s28 // (op 3568) reg expression
s_mov_b32 s49, s29 // (op 3568) reg expression
s_mov_b32 s50, s0 // (op 3568) reg expression
s_mov_b32 s51, 131072 // (op 3568) call()

s_mul_hi_i32 s47, s16, s10 // (op 3640) 
v_lshrrev_b32 v13, 3, v1 // (op 3640) 
s_add_i32 s0, s47, s16 // (op 3640) 

v_and_b32 v15, -8, v1 // (op 3640) 
s_ashr_i32 s1, s0, 31 // (op 3640) 
v_sub_u32 v16, v1, v15 // (op 3640) 

s_cmp_eq_i32 s10, 0 // (op 3640) 
s_mov_b32 s47, scc // (op 3640) 
v_lshlrev_b32 v15, 4, v16 // (op 3640) 

s_cmp_eq_i32 s47, 0 // (op 3640) 
s_cselect_b32 s18, 0, -1 // (op 3640) 

s_and_b32 s19, s13, 31 // (op 3640) 
s_lshl_b32 s47, 1, s19 // (op 3640) 
s_add_i32 s22, s47, s18 // (op 3640) 


s_and_b32 s18, s1, s22 // (op 3640) 


s_add_i32 s1, s0, s18 // (op 3640) 


s_ashr_i32 s0, s1, s19 // (op 3640) 


s_lshl_b32 s1, s13, 24 // (op 3640) 
s_ashr_i32 s47, s1, 31 // (op 3640) 

s_xor_b32 s1, s0, s47 // (op 3640) 

s_sub_i32 s0, s1, s47 // (op 3640) 


s_mul_hi_i32 s1, s0, s14 // (op 3640) 
s_add_i32 s47, s1, s0 // (op 3640) 

s_ashr_i32 s1, s47, 31 // (op 3640) 
s_cmp_eq_i32 s14, 0 // (op 3640) 
s_mov_b32 s18, scc // (op 3640) 
s_cmp_eq_i32 s18, 0 // (op 3640) 
s_cselect_b32 s19, 0, -1 // (op 3640) 

s_and_b32 s18, s15, 31 // (op 3640) 
s_lshl_b32 s22, 1, s18 // (op 3640) 
s_add_i32 s23, s22, s19 // (op 3640) 


s_and_b32 s19, s1, s23 // (op 3640) 


s_add_i32 s1, s47, s19 // (op 3640) 


s_ashr_i32 s19, s1, s18 // (op 3640) 


s_lshl_b32 s1, s15, 24 // (op 3640) 
s_ashr_i32 s18, s1, 31 // (op 3640) 

s_xor_b32 s1, s19, s18 // (op 3640) 

s_sub_i32 s19, s1, s18 // (op 3640) 


s_add_u32 s22, s4, 32 // (op 3640) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3640) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3640) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3640) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 3640) 

s_mul_i32 s1, s19, s22 // (op 3640) 

s_sub_i32 s47, s0, s1 // (op 3640) 


s_mul_hi_i32 s0, s16, s9 // (op 3640) 
s_add_i32 s1, s0, s16 // (op 3640) 

s_ashr_i32 s0, s1, 31 // (op 3640) 
s_cmp_eq_i32 s9, 0 // (op 3640) 
s_mov_b32 s18, scc // (op 3640) 
s_cmp_eq_i32 s18, 0 // (op 3640) 
s_cselect_b32 s19, 0, -1 // (op 3640) 

s_and_b32 s18, s45, 31 // (op 3640) 
s_lshl_b32 s24, 1, s18 // (op 3640) 
s_add_i32 s25, s24, s19 // (op 3640) 


s_and_b32 s19, s0, s25 // (op 3640) 


s_add_i32 s0, s1, s19 // (op 3640) 


s_ashr_i32 s1, s0, s18 // (op 3640) 


s_lshl_b32 s0, s45, 24 // (op 3640) 
s_ashr_i32 s18, s0, 31 // (op 3640) 

s_xor_b32 s0, s1, s18 // (op 3640) 

s_sub_i32 s1, s0, s18 // (op 3640) 


s_mul_i32 s0, s11, s22 // (op 3640) 

s_mul_i32 s18, s8, s0 // (op 3640) 

s_mul_i32 s0, s1, s18 // (op 3640) 

s_sub_u32 s18, s16, s0 // (op 3640) 

s_mul_hi_i32 s0, s18, s12 // (op 3640) 
s_add_i32 s19, s0, s18 // (op 3640) 


s_ashr_i32 s0, s19, 31 // (op 3640) 
s_cmp_eq_i32 s12, 0 // (op 3640) 
s_mov_b32 s18, scc // (op 3640) 
s_cmp_eq_i32 s18, 0 // (op 3640) 
s_cselect_b32 s22, 0, -1 // (op 3640) 

s_and_b32 s18, s44, 31 // (op 3640) 
s_lshl_b32 s23, 1, s18 // (op 3640) 
s_add_i32 s24, s23, s22 // (op 3640) 


s_and_b32 s25, s0, s24 // (op 3640) 


s_add_i32 s0, s19, s25 // (op 3640) 


s_ashr_i32 s19, s0, s18 // (op 3640) 


s_lshl_b32 s0, s44, 24 // (op 3640) 
s_ashr_i32 s18, s0, 31 // (op 3640) 

s_xor_b32 s0, s19, s18 // (op 3640) 

s_sub_i32 s19, s0, s18 // (op 3640) 


s_cmp_eq_i32 s1, 0 // (op 3640) 
s_mov_b32 s0, scc // (op 3640) 

s_cmp_eq_i32 s0, 0 // (op 3640) 
s_cselect_b32 s1, s19, s47 // (op 3640) 



v_lshl_add_u32 v16, s1, 5, v13 // (op 3640) 


v_mul_lo_u32 v61, v16, s31 // (op 3640) most significant: low of low * high
v_mul_hi_u32 v13, v16, s30 // (op 3640) most significant: high of low * low
v_mul_lo_u32 v60, v16, s30 // (op 3640) least significant: low of low * low
v_add_u32 v61, v61, v13 // (op 3640) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 3640) 
s_lshr_b32 s47, s0, 1 // (op 3640) 

v_lshl_add_u32 v13, s47, 8, v15 // (op 3640) 


v_mov_b32 v23, 0 // (op 3640) convert
v_mov_b32 v22, v13 // (op 3640) convert

v_add_co_u32 v20, s[0:1], v22, v60 // (op 3640) least significant half
v_addc_co_u32 v21, s[0:1], v23, v61, s[0:1] // (op 3640) most significant half

v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 3640) 


v_lshrrev_b64 v[22:23], 3, v[20:21] // (op 3640) 

v_mov_b32 v13, v22 // (op 3640) convert

s_mul_i32 s1, s30, s5 // (op 3642) most significant: low of low * high
s_mul_i32 s47, s31, s4 // (op 3642) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 3642) least significant
s_mul_hi_u32 s18, s30, s4 // (op 3642) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3642) most significant: sum
s_add_u32 s1, s1, s18 // (op 3642) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3642) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3642) 

s_mov_b32 s60, s32 // (op 3642) reg expression
s_mov_b32 s61, s33 // (op 3642) reg expression
s_mov_b32 s62, s0 // (op 3642) reg expression
s_mov_b32 s63, 131072 // (op 3642) call()

s_mul_hi_i32 s47, s16, s10 // (op 3704) 
s_add_i32 s0, s47, s16 // (op 3704) 

s_ashr_i32 s1, s0, 31 // (op 3704) 
s_cmp_eq_i32 s10, 0 // (op 3704) 
s_mov_b32 s47, scc // (op 3704) 
s_cmp_eq_i32 s47, 0 // (op 3704) 
s_cselect_b32 s18, 0, -1 // (op 3704) 

s_and_b32 s19, s13, 31 // (op 3704) 
s_lshl_b32 s47, 1, s19 // (op 3704) 
s_add_i32 s22, s47, s18 // (op 3704) 


s_and_b32 s18, s1, s22 // (op 3704) 


s_add_i32 s1, s0, s18 // (op 3704) 


s_ashr_i32 s0, s1, s19 // (op 3704) 


s_lshl_b32 s1, s13, 24 // (op 3704) 
s_ashr_i32 s47, s1, 31 // (op 3704) 

s_xor_b32 s1, s0, s47 // (op 3704) 

s_sub_i32 s0, s1, s47 // (op 3704) 


s_mul_hi_i32 s1, s0, s14 // (op 3704) 
s_add_i32 s47, s1, s0 // (op 3704) 

s_ashr_i32 s1, s47, 31 // (op 3704) 
s_cmp_eq_i32 s14, 0 // (op 3704) 
s_mov_b32 s18, scc // (op 3704) 
s_cmp_eq_i32 s18, 0 // (op 3704) 
s_cselect_b32 s19, 0, -1 // (op 3704) 

s_and_b32 s18, s15, 31 // (op 3704) 
s_lshl_b32 s22, 1, s18 // (op 3704) 
s_add_i32 s23, s22, s19 // (op 3704) 


s_and_b32 s19, s1, s23 // (op 3704) 


s_add_i32 s1, s47, s19 // (op 3704) 


s_ashr_i32 s19, s1, s18 // (op 3704) 


s_lshl_b32 s1, s15, 24 // (op 3704) 
s_ashr_i32 s18, s1, 31 // (op 3704) 

s_xor_b32 s1, s19, s18 // (op 3704) 

s_sub_i32 s19, s1, s18 // (op 3704) 


s_mul_i32 s1, s0, s11 // (op 3704) 

s_sub_u32 s0, s16, s1 // (op 3704) 

s_mul_i32 s1, s19, s11 // (op 3704) 
s_add_u32 s1, s1, s0 // (op 3704) 


s_mul_hi_i32 s0, s16, s9 // (op 3704) 
s_add_i32 s47, s0, s16 // (op 3704) 

s_ashr_i32 s0, s47, 31 // (op 3704) 
s_cmp_eq_i32 s9, 0 // (op 3704) 
s_mov_b32 s18, scc // (op 3704) 
s_cmp_eq_i32 s18, 0 // (op 3704) 
s_cselect_b32 s19, 0, -1 // (op 3704) 

s_and_b32 s18, s45, 31 // (op 3704) 
s_lshl_b32 s22, 1, s18 // (op 3704) 
s_add_i32 s23, s22, s19 // (op 3704) 


s_and_b32 s19, s0, s23 // (op 3704) 


s_add_i32 s0, s47, s19 // (op 3704) 


s_ashr_i32 s19, s0, s18 // (op 3704) 


s_lshl_b32 s0, s45, 24 // (op 3704) 
s_ashr_i32 s18, s0, 31 // (op 3704) 

s_xor_b32 s0, s19, s18 // (op 3704) 

s_sub_i32 s19, s0, s18 // (op 3704) 


s_add_u32 s22, s4, 32 // (op 3704) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3704) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3704) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3704) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 3704) 

s_mul_i32 s0, s11, s22 // (op 3704) 

s_mul_i32 s18, s8, s0 // (op 3704) 

s_mul_i32 s0, s19, s18 // (op 3704) 

s_sub_u32 s18, s16, s0 // (op 3704) 

s_mov_b32 s0, s18 // (op 3704) 
s_mul_hi_i32 s47, s0, s12 // (op 3704) 
s_add_i32 s22, s47, s0 // (op 3704) 


s_ashr_i32 s0, s22, 31 // (op 3704) 
s_cmp_eq_i32 s12, 0 // (op 3704) 
s_mov_b32 s47, scc // (op 3704) 
s_cmp_eq_i32 s47, 0 // (op 3704) 
s_cselect_b32 s23, 0, -1 // (op 3704) 

s_and_b32 s47, s44, 31 // (op 3704) 
s_lshl_b32 s24, 1, s47 // (op 3704) 
s_add_i32 s25, s24, s23 // (op 3704) 


s_and_b32 s23, s0, s25 // (op 3704) 


s_add_i32 s0, s22, s23 // (op 3704) 


s_ashr_i32 s22, s0, s47 // (op 3704) 


s_lshl_b32 s0, s44, 24 // (op 3704) 
s_ashr_i32 s47, s0, 31 // (op 3704) 

s_xor_b32 s0, s22, s47 // (op 3704) 

s_sub_i32 s22, s0, s47 // (op 3704) 


s_mul_i32 s0, s46, s11 // (op 3704) 
s_add_u32 s24, s6, 256 // (op 3704) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 3704) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 3704) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 3704) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 8 // (op 3704) 

s_sub_i32 s23, s24, s0 // (op 3704) 

s_mul_i32 s47, s22, s23 // (op 3704) 


s_sub_u32 s22, s18, s47 // (op 3704) 


s_add_u32 s18, s22, s0 // (op 3704) 


s_cmp_eq_i32 s19, 0 // (op 3704) 
s_mov_b32 s0, scc // (op 3704) 

s_cmp_eq_i32 s0, 0 // (op 3704) 
s_cselect_b32 s19, s18, s1 // (op 3704) 



v_lshl_add_u32 v15, s19, 8, v1 // (op 3704) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 3704) 
s_lshr_b32 s47, s0, 1 // (op 3704) 

s_lshl_b32 s0, s47, 3 // (op 3704) 

v_mul_lo_u32 v16, v15, s34 // (op 3704) 
v_add_u32 v16, v16, s0 // (op 3704) 


s_mul_i32 s1, s34, s7 // (op 3706) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 3706) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 3706) least significant
s_mul_hi_u32 s18, s34, s6 // (op 3706) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3706) most significant: sum
s_add_u32 s1, s1, s18 // (op 3706) most significant: sum


s_mov_b32 s64, s36 // (op 3706) reg expression
s_mov_b32 s65, s37 // (op 3706) reg expression
s_mov_b32 s66, s0 // (op 3706) reg expression
s_mov_b32 s67, 131072 // (op 3706) call()

s_mul_hi_i32 s47, s16, s10 // (op 3768) 
v_lshrrev_b32 v15, 6, v1 // (op 3768) 
s_add_i32 s0, s47, s16 // (op 3768) 

v_and_b32 v17, 1, v15 // (op 3768) 

s_ashr_i32 s1, s0, 31 // (op 3768) 
v_and_b32 v15, 15, v1 // (op 3768) 
s_cmp_eq_i32 s10, 0 // (op 3768) 
s_mov_b32 s47, scc // (op 3768) 
v_lshl_add_u32 v18, v17, 4, v15 // (op 3768) 


s_cmp_eq_i32 s47, 0 // (op 3768) 
s_cselect_b32 s18, 0, -1 // (op 3768) 

v_and_b32 v15, 63, v1 // (op 3768) 
s_and_b32 s19, s13, 31 // (op 3768) 
v_lshrrev_b32 v17, 4, v15 // (op 3768) 

s_lshl_b32 s47, 1, s19 // (op 3768) 
s_add_i32 s22, s47, s18 // (op 3768) 


s_and_b32 s18, s1, s22 // (op 3768) 


s_add_i32 s1, s0, s18 // (op 3768) 


s_ashr_i32 s0, s1, s19 // (op 3768) 


s_lshl_b32 s1, s13, 24 // (op 3768) 
s_ashr_i32 s47, s1, 31 // (op 3768) 

s_xor_b32 s1, s0, s47 // (op 3768) 

s_sub_i32 s0, s1, s47 // (op 3768) 


s_mul_hi_i32 s1, s0, s14 // (op 3768) 
s_add_i32 s47, s1, s0 // (op 3768) 

s_ashr_i32 s1, s47, 31 // (op 3768) 
s_cmp_eq_i32 s14, 0 // (op 3768) 
s_mov_b32 s18, scc // (op 3768) 
s_cmp_eq_i32 s18, 0 // (op 3768) 
s_cselect_b32 s19, 0, -1 // (op 3768) 

s_and_b32 s18, s15, 31 // (op 3768) 
s_lshl_b32 s22, 1, s18 // (op 3768) 
s_add_i32 s23, s22, s19 // (op 3768) 


s_and_b32 s19, s1, s23 // (op 3768) 


s_add_i32 s1, s47, s19 // (op 3768) 


s_ashr_i32 s19, s1, s18 // (op 3768) 


s_lshl_b32 s1, s15, 24 // (op 3768) 
s_ashr_i32 s18, s1, 31 // (op 3768) 

s_xor_b32 s1, s19, s18 // (op 3768) 

s_sub_i32 s19, s1, s18 // (op 3768) 


s_add_u32 s22, s4, 32 // (op 3768) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3768) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3768) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3768) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 3768) 

s_mul_i32 s1, s19, s22 // (op 3768) 

s_sub_i32 s47, s0, s1 // (op 3768) 


s_mul_hi_i32 s0, s16, s9 // (op 3768) 
s_add_i32 s1, s0, s16 // (op 3768) 

s_ashr_i32 s0, s1, 31 // (op 3768) 
s_cmp_eq_i32 s9, 0 // (op 3768) 
s_mov_b32 s18, scc // (op 3768) 
s_cmp_eq_i32 s18, 0 // (op 3768) 
s_cselect_b32 s19, 0, -1 // (op 3768) 

s_and_b32 s18, s45, 31 // (op 3768) 
s_lshl_b32 s24, 1, s18 // (op 3768) 
s_add_i32 s25, s24, s19 // (op 3768) 


s_and_b32 s19, s0, s25 // (op 3768) 


s_add_i32 s0, s1, s19 // (op 3768) 


s_ashr_i32 s1, s0, s18 // (op 3768) 


s_lshl_b32 s0, s45, 24 // (op 3768) 
s_ashr_i32 s18, s0, 31 // (op 3768) 

s_xor_b32 s0, s1, s18 // (op 3768) 

s_sub_i32 s1, s0, s18 // (op 3768) 


s_mul_i32 s0, s11, s22 // (op 3768) 

s_mul_i32 s18, s8, s0 // (op 3768) 

s_mul_i32 s0, s1, s18 // (op 3768) 

s_sub_u32 s18, s16, s0 // (op 3768) 

s_mul_hi_i32 s0, s18, s12 // (op 3768) 
s_add_i32 s19, s0, s18 // (op 3768) 


s_ashr_i32 s0, s19, 31 // (op 3768) 
s_cmp_eq_i32 s12, 0 // (op 3768) 
s_mov_b32 s18, scc // (op 3768) 
s_cmp_eq_i32 s18, 0 // (op 3768) 
s_cselect_b32 s22, 0, -1 // (op 3768) 

s_and_b32 s18, s44, 31 // (op 3768) 
s_lshl_b32 s23, 1, s18 // (op 3768) 
s_add_i32 s24, s23, s22 // (op 3768) 


s_and_b32 s25, s0, s24 // (op 3768) 


s_add_i32 s0, s19, s25 // (op 3768) 


s_ashr_i32 s19, s0, s18 // (op 3768) 


s_lshl_b32 s0, s44, 24 // (op 3768) 
s_ashr_i32 s18, s0, 31 // (op 3768) 

s_xor_b32 s0, s19, s18 // (op 3768) 

s_sub_i32 s19, s0, s18 // (op 3768) 


s_cmp_eq_i32 s1, 0 // (op 3768) 
s_mov_b32 s0, scc // (op 3768) 

s_cmp_eq_i32 s0, 0 // (op 3768) 
s_cselect_b32 s1, s19, s47 // (op 3768) 



v_lshl_add_u32 v15, s1, 5, v18 // (op 3768) 


v_mul_lo_u32 v18, v15, s38 // (op 3768) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 3768) 
s_lshr_b32 s47, s0, 1 // (op 3768) 

v_lshl_add_u32 v15, s47, 3, v17 // (op 3768) 


v_add_u32 v17, v15, v18 // (op 3768) 


s_mul_i32 s1, s38, s5 // (op 3770) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 3770) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 3770) least significant
s_mul_hi_u32 s18, s38, s4 // (op 3770) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3770) most significant: sum
s_add_u32 s1, s1, s18 // (op 3770) most significant: sum


s_mov_b32 s68, s40 // (op 3770) reg expression
s_mov_b32 s69, s41 // (op 3770) reg expression
s_mov_b32 s70, s0 // (op 3770) reg expression
s_mov_b32 s71, 131072 // (op 3770) call()

v_and_b32 v15, -8, v1 // (op 3861) 
v_sub_u32 v18, v1, v15 // (op 3861) 

v_lshlrev_b32 v15, 4, v18 // (op 3861) 

v_lshrrev_b32 v18, 3, v1 // (op 3861) 
v_lshl_add_u32 v19, v18, 10, v15 // (op 3861) 


v_and_b32 v15, -8, v1 // (op 3919) 
v_sub_u32 v18, v1, v15 // (op 3919) 

v_lshlrev_b32 v15, 4, v18 // (op 3919) 

v_lshrrev_b32 v18, 3, v1 // (op 3919) 
v_lshl_add_u32 v20, v18, 7, v15 // (op 3919) 


v_lshlrev_b32 v15, 2, v20 // (op 3919) 

v_lshrrev_b32 v18, 3, v15 // (op 3919) 

v_lshlrev_b32 v15, 2, v1 // (op 3972) 
s_ashr_i64 s[0:1], s[42:43], 7 // (op 1292) 
s_and_b32 s47, s0, 4294967294 // (op 1292) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 1291) 
s_mov_b32 s18, s0 // (op 1291) convert

s_cmp_lt_u32 s47, s18 // (op 1291) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_1291 // (op 1291) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_1291 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopTop_KLoopTail_1291:

s_add_u32 s47, s47, 1 // (op 1293) 
v_mov_b64 v[20:21], s[26:27] // (op 170) convert

v_mov_b32 v12, v14 // (op 170) 
v_mov_b32 v22, v12 // (op 170) 
buffer_load_dwordx4 v[24:27], v22, s[48:51], 0 offen // (op 170) Load value
v_add_u32 v22, v22, v20 // (op 170) 
buffer_load_dwordx4 v[28:31], v22, s[48:51], 0 offen // (op 170) Load value
v_add_u32 v22, v22, v20 // (op 170) 
buffer_load_dwordx4 v[32:35], v22, s[48:51], 0 offen // (op 170) Load value
v_add_u32 v22, v22, v20 // (op 170) 
buffer_load_dwordx4 v[36:39], v22, s[48:51], 0 offen // (op 170) Load value
v_add_u32 v22, v22, v20 // (op 170) 
buffer_load_dwordx4 v[40:43], v22, s[48:51], 0 offen // (op 170) Load value
v_add_u32 v22, v22, v20 // (op 170) 
buffer_load_dwordx4 v[44:47], v22, s[48:51], 0 offen // (op 170) Load value
v_add_u32 v22, v22, v20 // (op 170) 
buffer_load_dwordx4 v[48:51], v22, s[48:51], 0 offen // (op 170) Load value
v_add_u32 v22, v22, v20 // (op 170) 
buffer_load_dwordx4 v[52:55], v22, s[48:51], 0 offen // (op 170) Load value



v_add_u32 v14, 128, v14 // (op 3556) 


s_barrier  // (op 4170) 
v_mov_b32 v12, v13 // (op 183) 




buffer_load_dwordx2 v[20:21], v12, s[60:63], 0 offen // (op 183) Load value

v_add_u32 v13, 64, v13 // (op 3632) 
s_waitcnt vmcnt(8)
ds_write_b128 v19, v[24:27] offset:2048 // (op 171) Store local data 


v_mov_b32 v12, v16 // (op 184) 

s_waitcnt vmcnt(7)
ds_write_b128 v19, v[28:31] offset:2176 // (op 171) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v19, v[32:35] offset:2304 // (op 171) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v19, v[36:39] offset:2432 // (op 171) Store local data 
s_waitcnt vmcnt(4)
ds_write_b128 v19, v[40:43] offset:2560 // (op 171) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v19, v[44:47] offset:2688 // (op 171) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v19, v[48:51] offset:2816 // (op 171) Store local data 
buffer_load_dword v22, v12, s[64:67], 0 offen // (op 184) Load value

v_add_u32 v16, 4, v16 // (op 3696) 
s_waitcnt vmcnt(2)
ds_write_b128 v19, v[52:55] offset:2944 // (op 171) Store local data 

s_barrier  // (op 4183) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4173) 






ds_read_b128 v[24:27], v7 offset:2048 // (op 173) Load local data 
ds_read_b128 v[28:31], v7 offset:2112 // (op 173) Load local data 
s_waitcnt vmcnt(1)
ds_write_b64 v18, v[20:21] // (op 181) Store local data 

s_barrier  // (op 4197) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4187) 






ds_read_b128 v[32:35], v3 // (op 175) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v15, v22 offset:34816 // (op 182) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 4201) 



ds_read_u8 v12, v0 offset:34816 // (op 177) Load local data 


v_mov_b32 v20, v17 // (op 179) 

buffer_load_ubyte v21, v20, s[68:71], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v17, 4, v17 // (op 3760) 



ds_read_b128 v[36:39], v7 offset:6144 // (op 325) Load local data 
ds_read_b128 v[40:43], v7 offset:6208 // (op 325) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:31], v[32:35], a[0:3], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 180) 





ds_read_u8 v12, v0 offset:34944 // (op 329) Load local data 



ds_read_b128 v[22:25], v7 offset:10240 // (op 487) Load local data 
ds_read_b128 v[26:29], v7 offset:10304 // (op 487) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:43], v[32:35], a[4:7], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 331) 





ds_read_u8 v12, v0 offset:35072 // (op 491) Load local data 



ds_read_b128 v[36:39], v7 offset:14336 // (op 649) Load local data 
ds_read_b128 v[40:43], v7 offset:14400 // (op 649) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:29], v[32:35], a[8:11], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 493) 





ds_read_u8 v12, v0 offset:35200 // (op 653) Load local data 



ds_read_b128 v[22:25], v7 offset:18432 // (op 811) Load local data 
ds_read_b128 v[26:29], v7 offset:18496 // (op 811) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[36:43], v[32:35], a[12:15], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 655) 





ds_read_u8 v12, v0 offset:35328 // (op 815) Load local data 



ds_read_b128 v[36:39], v7 offset:22528 // (op 973) Load local data 
ds_read_b128 v[40:43], v7 offset:22592 // (op 973) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:29], v[32:35], a[16:19], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 817) 





ds_read_u8 v12, v0 offset:35456 // (op 977) Load local data 



ds_read_b128 v[22:25], v7 offset:26624 // (op 1135) Load local data 
ds_read_b128 v[26:29], v7 offset:26688 // (op 1135) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:43], v[32:35], a[20:23], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 979) 





ds_read_u8 v12, v0 offset:35584 // (op 1139) Load local data 



ds_read_b128 v[36:39], v7 offset:30720 // (op 1297) Load local data 
ds_read_b128 v[40:43], v7 offset:30784 // (op 1297) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:29], v[32:35], a[24:27], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1141) 





ds_read_u8 v12, v0 offset:35712 // (op 1301) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[36:43], v[32:35], a[28:31], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1303) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 1291) 
s_mov_b32 s18, s0 // (op 1291) convert

s_cmp_lt_u32 s47, s18 // (op 1291) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopTop_KLoopTail_1291 // (op 1291) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopTop_KLoopTail_1291 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_1291:





















s_barrier  // (op 4319) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ConditionalFalse_0_beta_125
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


v_accvgpr_read v10, a4 // (op 1621) 
v_accvgpr_read v11, a5 // (op 1621) 
v_accvgpr_read v12, a6 // (op 1621) 
v_accvgpr_read v13, a7 // (op 1621) 
v_mul_f32 v14, v8, v10 // (op 1621) 
v_mul_f32 v15, v8, v11 // (op 1621) 
v_mul_f32 v16, v8, v12 // (op 1621) 
v_mul_f32 v17, v8, v13 // (op 1621) 

v_cvt_f16_f32 v10, v14 // (op 1623) 
v_cvt_f16_f32 v11, v15 // (op 1623) 
v_cvt_f16_f32 v12, v16 // (op 1623) 
v_cvt_f16_f32 v13, v17 // (op 1623) 


v_pack_b32_f16 v14, v10, v11 // (op 1624) 
v_pack_b32_f16 v15, v12, v13 // (op 1624) 


buffer_store_dwordx2 v[14:15], v9, s[56:59], 0 offen offset:64 // (op 1624) Store value


v_accvgpr_read v10, a8 // (op 1641) 
v_accvgpr_read v11, a9 // (op 1641) 
v_accvgpr_read v12, a10 // (op 1641) 
v_accvgpr_read v13, a11 // (op 1641) 
v_mul_f32 v14, v8, v10 // (op 1641) 
v_mul_f32 v15, v8, v11 // (op 1641) 
v_mul_f32 v16, v8, v12 // (op 1641) 
v_mul_f32 v17, v8, v13 // (op 1641) 

v_cvt_f16_f32 v10, v14 // (op 1643) 
v_cvt_f16_f32 v11, v15 // (op 1643) 
v_cvt_f16_f32 v12, v16 // (op 1643) 
v_cvt_f16_f32 v13, v17 // (op 1643) 


v_pack_b32_f16 v14, v10, v11 // (op 1644) 
v_pack_b32_f16 v15, v12, v13 // (op 1644) 


buffer_store_dwordx2 v[14:15], v9, s[56:59], 0 offen offset:128 // (op 1644) Store value


v_accvgpr_read v10, a12 // (op 1661) 
v_accvgpr_read v11, a13 // (op 1661) 
v_accvgpr_read v12, a14 // (op 1661) 
v_accvgpr_read v13, a15 // (op 1661) 
v_mul_f32 v14, v8, v10 // (op 1661) 
v_mul_f32 v15, v8, v11 // (op 1661) 
v_mul_f32 v16, v8, v12 // (op 1661) 
v_mul_f32 v17, v8, v13 // (op 1661) 

v_cvt_f16_f32 v10, v14 // (op 1663) 
v_cvt_f16_f32 v11, v15 // (op 1663) 
v_cvt_f16_f32 v12, v16 // (op 1663) 
v_cvt_f16_f32 v13, v17 // (op 1663) 


v_pack_b32_f16 v14, v10, v11 // (op 1664) 
v_pack_b32_f16 v15, v12, v13 // (op 1664) 


buffer_store_dwordx2 v[14:15], v9, s[56:59], 0 offen offset:192 // (op 1664) Store value


v_accvgpr_read v10, a16 // (op 1681) 
v_accvgpr_read v11, a17 // (op 1681) 
v_accvgpr_read v12, a18 // (op 1681) 
v_accvgpr_read v13, a19 // (op 1681) 
v_mul_f32 v14, v8, v10 // (op 1681) 
v_mul_f32 v15, v8, v11 // (op 1681) 
v_mul_f32 v16, v8, v12 // (op 1681) 
v_mul_f32 v17, v8, v13 // (op 1681) 

v_cvt_f16_f32 v10, v14 // (op 1683) 
v_cvt_f16_f32 v11, v15 // (op 1683) 
v_cvt_f16_f32 v12, v16 // (op 1683) 
v_cvt_f16_f32 v13, v17 // (op 1683) 


v_pack_b32_f16 v14, v10, v11 // (op 1684) 
v_pack_b32_f16 v15, v12, v13 // (op 1684) 


buffer_store_dwordx2 v[14:15], v9, s[56:59], 0 offen offset:256 // (op 1684) Store value


v_accvgpr_read v10, a20 // (op 1701) 
v_accvgpr_read v11, a21 // (op 1701) 
v_accvgpr_read v12, a22 // (op 1701) 
v_accvgpr_read v13, a23 // (op 1701) 
v_mul_f32 v14, v8, v10 // (op 1701) 
v_mul_f32 v15, v8, v11 // (op 1701) 
v_mul_f32 v16, v8, v12 // (op 1701) 
v_mul_f32 v17, v8, v13 // (op 1701) 

v_cvt_f16_f32 v10, v14 // (op 1703) 
v_cvt_f16_f32 v11, v15 // (op 1703) 
v_cvt_f16_f32 v12, v16 // (op 1703) 
v_cvt_f16_f32 v13, v17 // (op 1703) 


v_pack_b32_f16 v14, v10, v11 // (op 1704) 
v_pack_b32_f16 v15, v12, v13 // (op 1704) 


buffer_store_dwordx2 v[14:15], v9, s[56:59], 0 offen offset:320 // (op 1704) Store value


v_accvgpr_read v10, a24 // (op 1721) 
v_accvgpr_read v11, a25 // (op 1721) 
v_accvgpr_read v12, a26 // (op 1721) 
v_accvgpr_read v13, a27 // (op 1721) 
v_mul_f32 v14, v8, v10 // (op 1721) 
v_mul_f32 v15, v8, v11 // (op 1721) 
v_mul_f32 v16, v8, v12 // (op 1721) 
v_mul_f32 v17, v8, v13 // (op 1721) 

v_cvt_f16_f32 v10, v14 // (op 1723) 
v_cvt_f16_f32 v11, v15 // (op 1723) 
v_cvt_f16_f32 v12, v16 // (op 1723) 
v_cvt_f16_f32 v13, v17 // (op 1723) 


v_pack_b32_f16 v14, v10, v11 // (op 1724) 
v_pack_b32_f16 v15, v12, v13 // (op 1724) 


buffer_store_dwordx2 v[14:15], v9, s[56:59], 0 offen offset:384 // (op 1724) Store value


v_accvgpr_read v10, a28 // (op 1741) 
v_accvgpr_read v11, a29 // (op 1741) 
v_accvgpr_read v12, a30 // (op 1741) 
v_accvgpr_read v13, a31 // (op 1741) 
v_mul_f32 v14, v8, v10 // (op 1741) 
v_mul_f32 v15, v8, v11 // (op 1741) 
v_mul_f32 v16, v8, v12 // (op 1741) 
v_mul_f32 v17, v8, v13 // (op 1741) 

v_cvt_f16_f32 v10, v14 // (op 1743) 
v_cvt_f16_f32 v11, v15 // (op 1743) 
v_cvt_f16_f32 v12, v16 // (op 1743) 
v_cvt_f16_f32 v13, v17 // (op 1743) 


v_pack_b32_f16 v14, v10, v11 // (op 1744) 
v_pack_b32_f16 v15, v12, v13 // (op 1744) 


buffer_store_dwordx2 v[14:15], v9, s[56:59], 0 offen offset:448 // (op 1744) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[10:11], v6, s[52:55], 0 offen offset:448 // (op 1924) Load value
v_accvgpr_read v12, a28 // (op 1920) 
v_accvgpr_read v13, a29 // (op 1920) 
v_accvgpr_read v14, a30 // (op 1920) 
v_accvgpr_read v15, a31 // (op 1920) 
v_mul_f32 v16, v8, v12 // (op 1920) 
v_mul_f32 v17, v8, v13 // (op 1920) 
v_mul_f32 v18, v8, v14 // (op 1920) 
v_mul_f32 v19, v8, v15 // (op 1920) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 1925) 
v_lshrrev_b32 v13, 16, v10 // (op 1925) 
v_cvt_f32_f16 v13, v13 // (op 1925) 
v_mul_f32 v20, v5, v12 // (op 1925) 
v_mul_f32 v21, v5, v13 // (op 1925) 
v_cvt_f32_f16 v12, v11 // (op 1925) 
v_lshrrev_b32 v13, 16, v11 // (op 1925) 
v_cvt_f32_f16 v13, v13 // (op 1925) 
v_mul_f32 v22, v5, v12 // (op 1925) 
v_mul_f32 v23, v5, v13 // (op 1925) 


v_add_f32 v10, v20, v16 // (op 1921) 
v_add_f32 v11, v21, v17 // (op 1921) 
v_add_f32 v12, v22, v18 // (op 1921) 
v_add_f32 v13, v23, v19 // (op 1921) 


v_cvt_f16_f32 v14, v10 // (op 1922) 
v_cvt_f16_f32 v15, v11 // (op 1922) 
v_cvt_f16_f32 v16, v12 // (op 1922) 
v_cvt_f16_f32 v17, v13 // (op 1922) 


v_pack_b32_f16 v10, v14, v15 // (op 1923) 
v_pack_b32_f16 v11, v16, v17 // (op 1923) 


buffer_store_dwordx2 v[10:11], v9, s[56:59], 0 offen offset:448 // (op 1923) Store value





buffer_load_dwordx2 v[10:11], v6, s[52:55], 0 offen offset:384 // (op 1904) Load value
v_accvgpr_read v12, a24 // (op 1900) 
v_accvgpr_read v13, a25 // (op 1900) 
v_accvgpr_read v14, a26 // (op 1900) 
v_accvgpr_read v15, a27 // (op 1900) 
v_mul_f32 v16, v8, v12 // (op 1900) 
v_mul_f32 v17, v8, v13 // (op 1900) 
v_mul_f32 v18, v8, v14 // (op 1900) 
v_mul_f32 v19, v8, v15 // (op 1900) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 1905) 
v_lshrrev_b32 v13, 16, v10 // (op 1905) 
v_cvt_f32_f16 v13, v13 // (op 1905) 
v_mul_f32 v20, v5, v12 // (op 1905) 
v_mul_f32 v21, v5, v13 // (op 1905) 
v_cvt_f32_f16 v12, v11 // (op 1905) 
v_lshrrev_b32 v13, 16, v11 // (op 1905) 
v_cvt_f32_f16 v13, v13 // (op 1905) 
v_mul_f32 v22, v5, v12 // (op 1905) 
v_mul_f32 v23, v5, v13 // (op 1905) 


v_add_f32 v10, v20, v16 // (op 1901) 
v_add_f32 v11, v21, v17 // (op 1901) 
v_add_f32 v12, v22, v18 // (op 1901) 
v_add_f32 v13, v23, v19 // (op 1901) 


v_cvt_f16_f32 v14, v10 // (op 1902) 
v_cvt_f16_f32 v15, v11 // (op 1902) 
v_cvt_f16_f32 v16, v12 // (op 1902) 
v_cvt_f16_f32 v17, v13 // (op 1902) 


v_pack_b32_f16 v10, v14, v15 // (op 1903) 
v_pack_b32_f16 v11, v16, v17 // (op 1903) 


buffer_store_dwordx2 v[10:11], v9, s[56:59], 0 offen offset:384 // (op 1903) Store value





buffer_load_dwordx2 v[10:11], v6, s[52:55], 0 offen offset:320 // (op 1884) Load value
v_accvgpr_read v12, a20 // (op 1880) 
v_accvgpr_read v13, a21 // (op 1880) 
v_accvgpr_read v14, a22 // (op 1880) 
v_accvgpr_read v15, a23 // (op 1880) 
v_mul_f32 v16, v8, v12 // (op 1880) 
v_mul_f32 v17, v8, v13 // (op 1880) 
v_mul_f32 v18, v8, v14 // (op 1880) 
v_mul_f32 v19, v8, v15 // (op 1880) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 1885) 
v_lshrrev_b32 v13, 16, v10 // (op 1885) 
v_cvt_f32_f16 v13, v13 // (op 1885) 
v_mul_f32 v20, v5, v12 // (op 1885) 
v_mul_f32 v21, v5, v13 // (op 1885) 
v_cvt_f32_f16 v12, v11 // (op 1885) 
v_lshrrev_b32 v13, 16, v11 // (op 1885) 
v_cvt_f32_f16 v13, v13 // (op 1885) 
v_mul_f32 v22, v5, v12 // (op 1885) 
v_mul_f32 v23, v5, v13 // (op 1885) 


v_add_f32 v10, v20, v16 // (op 1881) 
v_add_f32 v11, v21, v17 // (op 1881) 
v_add_f32 v12, v22, v18 // (op 1881) 
v_add_f32 v13, v23, v19 // (op 1881) 


v_cvt_f16_f32 v14, v10 // (op 1882) 
v_cvt_f16_f32 v15, v11 // (op 1882) 
v_cvt_f16_f32 v16, v12 // (op 1882) 
v_cvt_f16_f32 v17, v13 // (op 1882) 


v_pack_b32_f16 v10, v14, v15 // (op 1883) 
v_pack_b32_f16 v11, v16, v17 // (op 1883) 


buffer_store_dwordx2 v[10:11], v9, s[56:59], 0 offen offset:320 // (op 1883) Store value





buffer_load_dwordx2 v[10:11], v6, s[52:55], 0 offen offset:256 // (op 1864) Load value
v_accvgpr_read v12, a16 // (op 1860) 
v_accvgpr_read v13, a17 // (op 1860) 
v_accvgpr_read v14, a18 // (op 1860) 
v_accvgpr_read v15, a19 // (op 1860) 
v_mul_f32 v16, v8, v12 // (op 1860) 
v_mul_f32 v17, v8, v13 // (op 1860) 
v_mul_f32 v18, v8, v14 // (op 1860) 
v_mul_f32 v19, v8, v15 // (op 1860) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 1865) 
v_lshrrev_b32 v13, 16, v10 // (op 1865) 
v_cvt_f32_f16 v13, v13 // (op 1865) 
v_mul_f32 v20, v5, v12 // (op 1865) 
v_mul_f32 v21, v5, v13 // (op 1865) 
v_cvt_f32_f16 v12, v11 // (op 1865) 
v_lshrrev_b32 v13, 16, v11 // (op 1865) 
v_cvt_f32_f16 v13, v13 // (op 1865) 
v_mul_f32 v22, v5, v12 // (op 1865) 
v_mul_f32 v23, v5, v13 // (op 1865) 


v_add_f32 v10, v20, v16 // (op 1861) 
v_add_f32 v11, v21, v17 // (op 1861) 
v_add_f32 v12, v22, v18 // (op 1861) 
v_add_f32 v13, v23, v19 // (op 1861) 


v_cvt_f16_f32 v14, v10 // (op 1862) 
v_cvt_f16_f32 v15, v11 // (op 1862) 
v_cvt_f16_f32 v16, v12 // (op 1862) 
v_cvt_f16_f32 v17, v13 // (op 1862) 


v_pack_b32_f16 v10, v14, v15 // (op 1863) 
v_pack_b32_f16 v11, v16, v17 // (op 1863) 


buffer_store_dwordx2 v[10:11], v9, s[56:59], 0 offen offset:256 // (op 1863) Store value





buffer_load_dwordx2 v[10:11], v6, s[52:55], 0 offen offset:192 // (op 1844) Load value
v_accvgpr_read v12, a12 // (op 1840) 
v_accvgpr_read v13, a13 // (op 1840) 
v_accvgpr_read v14, a14 // (op 1840) 
v_accvgpr_read v15, a15 // (op 1840) 
v_mul_f32 v16, v8, v12 // (op 1840) 
v_mul_f32 v17, v8, v13 // (op 1840) 
v_mul_f32 v18, v8, v14 // (op 1840) 
v_mul_f32 v19, v8, v15 // (op 1840) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 1845) 
v_lshrrev_b32 v13, 16, v10 // (op 1845) 
v_cvt_f32_f16 v13, v13 // (op 1845) 
v_mul_f32 v20, v5, v12 // (op 1845) 
v_mul_f32 v21, v5, v13 // (op 1845) 
v_cvt_f32_f16 v12, v11 // (op 1845) 
v_lshrrev_b32 v13, 16, v11 // (op 1845) 
v_cvt_f32_f16 v13, v13 // (op 1845) 
v_mul_f32 v22, v5, v12 // (op 1845) 
v_mul_f32 v23, v5, v13 // (op 1845) 


v_add_f32 v10, v20, v16 // (op 1841) 
v_add_f32 v11, v21, v17 // (op 1841) 
v_add_f32 v12, v22, v18 // (op 1841) 
v_add_f32 v13, v23, v19 // (op 1841) 


v_cvt_f16_f32 v14, v10 // (op 1842) 
v_cvt_f16_f32 v15, v11 // (op 1842) 
v_cvt_f16_f32 v16, v12 // (op 1842) 
v_cvt_f16_f32 v17, v13 // (op 1842) 


v_pack_b32_f16 v10, v14, v15 // (op 1843) 
v_pack_b32_f16 v11, v16, v17 // (op 1843) 


buffer_store_dwordx2 v[10:11], v9, s[56:59], 0 offen offset:192 // (op 1843) Store value





buffer_load_dwordx2 v[10:11], v6, s[52:55], 0 offen offset:128 // (op 1824) Load value
v_accvgpr_read v12, a8 // (op 1820) 
v_accvgpr_read v13, a9 // (op 1820) 
v_accvgpr_read v14, a10 // (op 1820) 
v_accvgpr_read v15, a11 // (op 1820) 
v_mul_f32 v16, v8, v12 // (op 1820) 
v_mul_f32 v17, v8, v13 // (op 1820) 
v_mul_f32 v18, v8, v14 // (op 1820) 
v_mul_f32 v19, v8, v15 // (op 1820) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 1825) 
v_lshrrev_b32 v13, 16, v10 // (op 1825) 
v_cvt_f32_f16 v13, v13 // (op 1825) 
v_mul_f32 v20, v5, v12 // (op 1825) 
v_mul_f32 v21, v5, v13 // (op 1825) 
v_cvt_f32_f16 v12, v11 // (op 1825) 
v_lshrrev_b32 v13, 16, v11 // (op 1825) 
v_cvt_f32_f16 v13, v13 // (op 1825) 
v_mul_f32 v22, v5, v12 // (op 1825) 
v_mul_f32 v23, v5, v13 // (op 1825) 


v_add_f32 v10, v20, v16 // (op 1821) 
v_add_f32 v11, v21, v17 // (op 1821) 
v_add_f32 v12, v22, v18 // (op 1821) 
v_add_f32 v13, v23, v19 // (op 1821) 


v_cvt_f16_f32 v14, v10 // (op 1822) 
v_cvt_f16_f32 v15, v11 // (op 1822) 
v_cvt_f16_f32 v16, v12 // (op 1822) 
v_cvt_f16_f32 v17, v13 // (op 1822) 


v_pack_b32_f16 v10, v14, v15 // (op 1823) 
v_pack_b32_f16 v11, v16, v17 // (op 1823) 


buffer_store_dwordx2 v[10:11], v9, s[56:59], 0 offen offset:128 // (op 1823) Store value





buffer_load_dwordx2 v[10:11], v6, s[52:55], 0 offen offset:64 // (op 1804) Load value
v_accvgpr_read v12, a4 // (op 1800) 
v_accvgpr_read v13, a5 // (op 1800) 
v_accvgpr_read v14, a6 // (op 1800) 
v_accvgpr_read v15, a7 // (op 1800) 
v_mul_f32 v16, v8, v12 // (op 1800) 
v_mul_f32 v17, v8, v13 // (op 1800) 
v_mul_f32 v18, v8, v14 // (op 1800) 
v_mul_f32 v19, v8, v15 // (op 1800) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 1805) 
v_lshrrev_b32 v13, 16, v10 // (op 1805) 
v_cvt_f32_f16 v13, v13 // (op 1805) 
v_mul_f32 v20, v5, v12 // (op 1805) 
v_mul_f32 v21, v5, v13 // (op 1805) 
v_cvt_f32_f16 v12, v11 // (op 1805) 
v_lshrrev_b32 v13, 16, v11 // (op 1805) 
v_cvt_f32_f16 v13, v13 // (op 1805) 
v_mul_f32 v22, v5, v12 // (op 1805) 
v_mul_f32 v23, v5, v13 // (op 1805) 


v_add_f32 v10, v20, v16 // (op 1801) 
v_add_f32 v11, v21, v17 // (op 1801) 
v_add_f32 v12, v22, v18 // (op 1801) 
v_add_f32 v13, v23, v19 // (op 1801) 


v_cvt_f16_f32 v14, v10 // (op 1802) 
v_cvt_f16_f32 v15, v11 // (op 1802) 
v_cvt_f16_f32 v16, v12 // (op 1802) 
v_cvt_f16_f32 v17, v13 // (op 1802) 


v_pack_b32_f16 v10, v14, v15 // (op 1803) 
v_pack_b32_f16 v11, v16, v17 // (op 1803) 


buffer_store_dwordx2 v[10:11], v9, s[56:59], 0 offen offset:64 // (op 1803) Store value





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
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ConditionalBottom_0_beta_125:
















s_endpgm  // End of RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__end-RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 156
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 71680 // lds bytes
  .amdhsa_accum_offset 124
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
    macrotile: [256, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 71680
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 124
    .agpr_count: 32
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 24
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}}
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

