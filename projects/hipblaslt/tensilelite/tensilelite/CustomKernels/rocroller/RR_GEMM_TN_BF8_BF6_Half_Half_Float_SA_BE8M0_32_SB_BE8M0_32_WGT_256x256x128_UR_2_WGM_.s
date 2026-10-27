// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_,@function
RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 27407) 
s_mul_hi_i32 s1, s16, s10 // (op 27547) 
v_and_b32 v0, 4294967168, v1 // (op 27468) 
v_and_b32 v3, 15, v1 // (op 27468) 
v_add_lshl_u32 v4, v0, v3, 7 // (op 27468) 


v_and_b32 v0, 48, v1 // (op 27468) 
v_add_u32 v3, v4, v0 // (op 27468) 


v_lshrrev_b32 v0, 6, v1 // (op 27526) 
v_and_b32 v4, 1, v0 // (op 27526) 

v_and_b32 v0, 15, v1 // (op 27526) 
v_lshl_add_u32 v5, v4, 7, v0 // (op 27526) 


v_and_b32 v0, 63, v1 // (op 27526) 
v_lshrrev_b32 v4, 4, v0 // (op 27526) 

v_lshlrev_b32 v0, 5, v4 // (op 27526) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 27526) 


v_mul_lo_u32 v0, v4, 6 // (op 27526) 

v_lshrrev_b32 v4, 3, v0 // (op 27526) 

v_and_b32 v0, 4294967168, v1 // (op 27593) 
v_and_b32 v5, 15, v1 // (op 27593) 
v_add_lshl_u32 v6, v0, v5, 7 // (op 27593) 


v_and_b32 v0, 48, v1 // (op 27593) 
v_add_u32 v5, v6, v0 // (op 27593) 


v_lshrrev_b32 v0, 6, v1 // (op 27622) 
v_and_b32 v6, 1, v0 // (op 27622) 

v_and_b32 v0, 15, v1 // (op 27622) 
v_lshl_add_u32 v7, v6, 7, v0 // (op 27622) 


v_and_b32 v0, 63, v1 // (op 27622) 
v_lshrrev_b32 v6, 4, v0 // (op 27622) 

v_lshlrev_b32 v0, 5, v6 // (op 27622) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 27622) 


v_mul_lo_u32 v0, v6, 6 // (op 27622) 

v_lshrrev_b32 v6, 3, v0 // (op 27622) 

v_and_b32 v0, 4294967168, v1 // (op 27407) 
s_add_i32 s18, s0, s16 // (op 27407) 

v_and_b32 v7, 63, v1 // (op 27407) 
s_ashr_i32 s0, s18, 31 // (op 27407) 
v_lshrrev_b32 v8, 4, v7 // (op 27407) 
s_cmp_eq_i32 s10, 0 // (op 27407) 
s_mov_b32 s19, scc // (op 27407) 
v_lshlrev_b32 v9, 2, v8 // (op 27407) 

s_cmp_eq_i32 s19, 0 // (op 27407) 
s_cselect_b32 s49, 0, -1 // (op 27407) 

v_add_u32 v8, v0, v9 // (op 27407) 


s_and_b32 s19, s13, 31 // (op 27407) 
v_lshrrev_b32 v0, 2, v7 // (op 27407) 

v_and_b32 v7, 3, v0 // (op 27407) 

v_and_b32 v0, 4294967168, v1 // (op 27547) 
v_and_b32 v9, 63, v1 // (op 27547) 
s_lshl_b32 s50, 1, s19 // (op 27407) 
v_and_b32 v10, 3, v1 // (op 27407) 
s_add_i32 s51, s50, s49 // (op 27407) 


s_and_b32 s49, s0, s51 // (op 27407) 


s_add_i32 s0, s1, s16 // (op 27547) 

s_ashr_i32 s1, s0, 31 // (op 27547) 
v_lshl_add_u32 v11, v7, 2, v10 // (op 27407) 


s_add_i32 s50, s18, s49 // (op 27407) 


v_lshrrev_b32 v7, 6, v1 // (op 27407) 
s_ashr_i32 s18, s50, s19 // (op 27407) 


v_and_b32 v10, 1, v7 // (op 27407) 

s_lshl_b32 s19, s13, 24 // (op 27407) 
v_lshl_add_u32 v7, v10, 7, v11 // (op 27407) 


s_ashr_i32 s49, s19, 31 // (op 27407) 

s_xor_b32 s19, s18, s49 // (op 27407) 

s_sub_i32 s18, s19, s49 // (op 27407) 


s_mul_hi_i32 s19, s18, s14 // (op 27407) 
s_add_i32 s49, s19, s18 // (op 27407) 

s_ashr_i32 s19, s49, 31 // (op 27407) 
s_cmp_eq_i32 s14, 0 // (op 27407) 
s_mov_b32 s50, scc // (op 27407) 
s_cmp_eq_i32 s50, 0 // (op 27407) 
s_cselect_b32 s51, 0, -1 // (op 27407) 

s_and_b32 s50, s15, 31 // (op 27407) 
v_lshrrev_b32 v10, 4, v9 // (op 27547) 
s_cmp_eq_i32 s10, 0 // (op 27547) 
s_mov_b32 s52, scc // (op 27547) 
v_lshlrev_b32 v11, 2, v10 // (op 27547) 

s_cmp_eq_i32 s52, 0 // (op 27547) 
s_cselect_b32 s53, 0, -1 // (op 27547) 

s_lshl_b32 s52, 1, s50 // (op 27407) 
v_add_u32 v10, v0, v11 // (op 27547) 


v_lshrrev_b32 v0, 2, v9 // (op 27547) 

s_add_i32 s54, s52, s51 // (op 27407) 


s_and_b32 s51, s19, s54 // (op 27407) 


s_add_i32 s19, s49, s51 // (op 27407) 


s_ashr_i32 s49, s19, s50 // (op 27407) 


s_lshl_b32 s19, s15, 24 // (op 27407) 
s_ashr_i32 s50, s19, 31 // (op 27407) 

s_xor_b32 s19, s49, s50 // (op 27407) 

s_sub_i32 s49, s19, s50 // (op 27407) 


s_mul_i32 s19, s18, s11 // (op 27407) 
s_sub_u32 s50, s16, s19 // (op 27407) 

s_mul_i32 s19, s49, s11 // (op 27407) 
s_add_u32 s19, s19, s50 // (op 27407) 

s_mul_hi_i32 s50, s16, s9 // (op 27407) 
s_add_i32 s51, s50, s16 // (op 27407) 

s_ashr_i32 s50, s51, 31 // (op 27407) 
s_cmp_eq_i32 s9, 0 // (op 27407) 
s_mov_b32 s52, scc // (op 27407) 
s_cmp_eq_i32 s52, 0 // (op 27407) 
s_cselect_b32 s54, 0, -1 // (op 27407) 

s_and_b32 s52, s13, 31 // (op 27547) 
v_and_b32 v9, 3, v0 // (op 27547) 

s_lshl_b32 s55, 1, s52 // (op 27547) 
v_and_b32 v0, 3, v1 // (op 27547) 
s_add_i32 s56, s55, s53 // (op 27547) 


s_and_b32 s53, s1, s56 // (op 27547) 


v_lshl_add_u32 v11, v9, 2, v0 // (op 27547) 


s_add_i32 s1, s0, s53 // (op 27547) 


v_lshrrev_b32 v0, 6, v1 // (op 27547) 
s_ashr_i32 s0, s1, s52 // (op 27547) 


v_and_b32 v9, 1, v0 // (op 27547) 

s_lshl_b32 s1, s13, 24 // (op 27547) 
v_lshl_add_u32 v0, v9, 7, v11 // (op 27547) 


s_ashr_i32 s52, s1, 31 // (op 27547) 

s_xor_b32 s1, s0, s52 // (op 27547) 

s_sub_i32 s0, s1, s52 // (op 27547) 


s_mul_hi_i32 s1, s0, s14 // (op 27547) 
s_add_i32 s52, s1, s0 // (op 27547) 

s_ashr_i32 s1, s52, 31 // (op 27547) 
s_cmp_eq_i32 s14, 0 // (op 27547) 
s_mov_b32 s53, scc // (op 27547) 
s_cmp_eq_i32 s53, 0 // (op 27547) 
s_cselect_b32 s55, 0, -1 // (op 27547) 

s_and_b32 s53, s15, 31 // (op 27547) 
s_lshl_b32 s56, 1, s53 // (op 27547) 
s_add_i32 s57, s56, s55 // (op 27547) 


s_and_b32 s55, s1, s57 // (op 27547) 


s_add_i32 s1, s52, s55 // (op 27547) 


s_ashr_i32 s52, s1, s53 // (op 27547) 


s_lshl_b32 s1, s15, 24 // (op 27547) 
s_ashr_i32 s53, s1, 31 // (op 27547) 

s_xor_b32 s1, s52, s53 // (op 27547) 

s_sub_i32 s52, s1, s53 // (op 27547) 


s_mul_i32 s1, s0, s11 // (op 27547) 
s_sub_u32 s53, s16, s1 // (op 27547) 

s_mul_i32 s1, s52, s11 // (op 27547) 
s_add_u32 s1, s1, s53 // (op 27547) 

s_mul_hi_i32 s53, s16, s9 // (op 27547) 
s_add_i32 s55, s53, s16 // (op 27547) 

s_ashr_i32 s53, s55, 31 // (op 27547) 
s_cmp_eq_i32 s9, 0 // (op 27547) 
s_mov_b32 s56, scc // (op 27547) 
s_cmp_eq_i32 s56, 0 // (op 27547) 
s_cselect_b32 s57, 0, -1 // (op 27547) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 27407) 
s_lshl_b32 s58, 1, s56 // (op 27407) 
s_add_i32 s59, s58, s54 // (op 27407) 


s_and_b32 s54, s50, s59 // (op 27407) 


s_add_i32 s50, s51, s54 // (op 27407) 


s_ashr_i32 s51, s50, s56 // (op 27407) 


s_lshl_b32 s50, s45, 24 // (op 27407) 
s_ashr_i32 s54, s50, 31 // (op 27407) 

s_xor_b32 s50, s51, s54 // (op 27407) 

s_sub_i32 s51, s50, s54 // (op 27407) 


s_and_b32 s50, s45, 31 // (op 27547) 
s_lshl_b32 s54, 1, s50 // (op 27547) 
s_add_i32 s56, s54, s57 // (op 27547) 


s_and_b32 s54, s53, s56 // (op 27547) 


s_add_i32 s53, s55, s54 // (op 27547) 


s_ashr_i32 s54, s53, s50 // (op 27547) 


s_lshl_b32 s50, s45, 24 // (op 27547) 
s_ashr_i32 s53, s50, 31 // (op 27547) 

s_xor_b32 s50, s54, s53 // (op 27547) 

s_sub_i32 s54, s50, s53 // (op 27547) 


s_add_u32 s56, s6, 256 // (op 27407) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 27407) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 27407) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 27407) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 27407) 

s_mul_i32 s50, s11, s56 // (op 27407) 
s_mul_i32 s53, s8, s50 // (op 27407) 

s_mul_i32 s50, s51, s53 // (op 27407) 

s_sub_u32 s53, s16, s50 // (op 27407) 

s_mov_b32 s50, s53 // (op 27407) 
s_mul_hi_i32 s55, s50, s12 // (op 27407) 
s_add_i32 s58, s55, s50 // (op 27407) 


s_ashr_i32 s50, s58, 31 // (op 27407) 
s_cmp_eq_i32 s12, 0 // (op 27407) 
s_mov_b32 s55, scc // (op 27407) 
s_cmp_eq_i32 s55, 0 // (op 27407) 
s_cselect_b32 s59, 0, -1 // (op 27407) 

s_and_b32 s55, s44, 31 // (op 27407) 
s_lshl_b32 s60, 1, s55 // (op 27407) 
s_add_i32 s61, s60, s59 // (op 27407) 


s_and_b32 s59, s50, s61 // (op 27407) 


s_add_i32 s50, s58, s59 // (op 27407) 


s_ashr_i32 s58, s50, s55 // (op 27407) 


s_lshl_b32 s50, s44, 24 // (op 27407) 
s_ashr_i32 s55, s50, 31 // (op 27407) 

s_xor_b32 s50, s58, s55 // (op 27407) 

s_sub_i32 s58, s50, s55 // (op 27407) 


s_mul_i32 s50, s46, s11 // (op 27407) 
s_add_u32 s60, s20, 256 // (op 27407) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 27407) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 27407) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 27407) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 8 // (op 27407) 

s_sub_i32 s55, s60, s50 // (op 27407) 

s_mul_i32 s59, s58, s55 // (op 27407) 

s_sub_u32 s55, s53, s59 // (op 27407) 


s_add_u32 s53, s55, s50 // (op 27407) 


s_cmp_eq_i32 s51, 0 // (op 27407) 
s_mov_b32 s50, scc // (op 27407) 

s_cmp_eq_i32 s50, 0 // (op 27407) 
s_cselect_b32 s51, s53, s19 // (op 27407) 


s_mul_i32 s19, s49, s56 // (op 27407) 


v_lshl_add_u32 v9, s51, 8, v8 // (op 27407) 


s_sub_i32 s49, s18, s19 // (op 27407) 


s_cmp_eq_i32 s50, 0 // (op 27407) 
s_cselect_b32 s51, s58, s49 // (op 27407) 



v_lshl_add_u32 v8, s51, 8, v7 // (op 27407) 


v_mul_lo_u32 v7, v8, s2 // (op 27407) 

v_add_lshl_u32 v8, v9, v7, 1 // (op 27407) 


s_add_u32 s18, s6, 256 // (op 27547) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 27547) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 27547) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 27547) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 8 // (op 27547) 

s_mul_i32 s53, s11, s18 // (op 27547) 
s_mul_i32 s49, s8, s53 // (op 27547) 

s_mul_i32 s53, s54, s49 // (op 27547) 

s_sub_u32 s49, s16, s53 // (op 27547) 

s_mov_b32 s53, s49 // (op 27547) 
s_mul_hi_i32 s50, s53, s12 // (op 27547) 
s_add_i32 s51, s50, s53 // (op 27547) 


s_ashr_i32 s50, s51, 31 // (op 27547) 
s_cmp_eq_i32 s12, 0 // (op 27547) 
s_mov_b32 s53, scc // (op 27547) 
s_cmp_eq_i32 s53, 0 // (op 27547) 
s_cselect_b32 s55, 0, -1 // (op 27547) 

s_and_b32 s53, s44, 31 // (op 27547) 
s_lshl_b32 s56, 1, s53 // (op 27547) 
s_add_i32 s57, s56, s55 // (op 27547) 


s_and_b32 s55, s50, s57 // (op 27547) 


s_add_i32 s50, s51, s55 // (op 27547) 


s_ashr_i32 s51, s50, s53 // (op 27547) 


s_lshl_b32 s50, s44, 24 // (op 27547) 
s_ashr_i32 s53, s50, 31 // (op 27547) 

s_xor_b32 s50, s51, s53 // (op 27547) 

s_sub_i32 s51, s50, s53 // (op 27547) 


s_mul_i32 s50, s46, s11 // (op 27547) 
s_add_u32 s56, s20, 256 // (op 27547) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 27547) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 27547) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 27547) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 27547) 

s_sub_i32 s53, s56, s50 // (op 27547) 

s_mul_i32 s55, s51, s53 // (op 27547) 

s_sub_u32 s53, s49, s55 // (op 27547) 


s_add_u32 s49, s53, s50 // (op 27547) 


s_cmp_eq_i32 s54, 0 // (op 27547) 
s_mov_b32 s50, scc // (op 27547) 

s_cmp_eq_i32 s50, 0 // (op 27547) 
s_cselect_b32 s53, s49, s1 // (op 27547) 


s_mul_i32 s1, s52, s18 // (op 27547) 


v_lshl_add_u32 v7, s53, 8, v10 // (op 27547) 


s_sub_i32 s49, s0, s1 // (op 27547) 


s_cmp_eq_i32 s50, 0 // (op 27547) 
s_cselect_b32 s0, s51, s49 // (op 27547) 



v_lshl_add_u32 v9, s0, 8, v0 // (op 27547) 


v_mul_lo_u32 v0, v9, s4 // (op 27547) 

v_add_lshl_u32 v9, v7, v0, 1 // (op 27547) 


s_mul_i32 s1, s2, s7 // (op 27409) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 27409) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 27409) least significant
s_mul_hi_u32 s19, s2, s6 // (op 27409) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 27409) most significant: sum
s_add_u32 s1, s1, s19 // (op 27409) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 27409) 

s_mul_i32 s1, s4, s7 // (op 27549) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 27549) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 27549) least significant
s_mul_hi_u32 s50, s4, s6 // (op 27549) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 27549) most significant: sum
s_add_u32 s1, s1, s50 // (op 27549) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 27549) 

s_mov_b32 s52, s22 // (op 27409) reg expression
s_mov_b32 s53, s23 // (op 27409) reg expression
s_mov_b32 s54, s18 // (op 27409) reg expression
s_mov_b32 s55, 131072 // (op 27409) call()

s_mov_b32 s56, s24 // (op 27549) reg expression

s_mov_b32 s57, s25 // (op 27549) reg expression
s_mov_b32 s58, s50 // (op 27549) reg expression
s_mov_b32 s59, 131072 // (op 27549) call()


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
s_mul_hi_i32 s0, s16, s10 // (op 27297) 
v_lshrrev_b32 v10, 3, v1 // (op 27297) 
s_add_i32 s1, s0, s16 // (op 27297) 

v_and_b32 v11, -8, v1 // (op 27297) 
s_ashr_i32 s0, s1, 31 // (op 27297) 
v_sub_u32 v12, v1, v11 // (op 27297) 

s_cmp_eq_i32 s10, 0 // (op 27297) 
s_mov_b32 s18, scc // (op 27297) 
v_lshlrev_b32 v11, 4, v12 // (op 27297) 

s_cmp_eq_i32 s18, 0 // (op 27297) 
s_cselect_b32 s19, 0, -1 // (op 27297) 

v_mov_b32 v13, 0 // (op 27297) convert
v_mov_b32 v12, v11 // (op 27297) convert

s_and_b32 s18, s13, 31 // (op 27297) 
s_lshl_b32 s22, 1, s18 // (op 27297) 
s_add_i32 s23, s22, s19 // (op 27297) 


s_and_b32 s19, s0, s23 // (op 27297) 


s_add_i32 s0, s1, s19 // (op 27297) 


s_ashr_i32 s1, s0, s18 // (op 27297) 


s_lshl_b32 s0, s13, 24 // (op 27297) 
s_ashr_i32 s18, s0, 31 // (op 27297) 

s_xor_b32 s0, s1, s18 // (op 27297) 

s_sub_i32 s1, s0, s18 // (op 27297) 


s_mul_hi_i32 s0, s1, s14 // (op 27297) 
s_add_i32 s18, s0, s1 // (op 27297) 

s_ashr_i32 s0, s18, 31 // (op 27297) 
s_cmp_eq_i32 s14, 0 // (op 27297) 
s_mov_b32 s19, scc // (op 27297) 
s_cmp_eq_i32 s19, 0 // (op 27297) 
s_cselect_b32 s22, 0, -1 // (op 27297) 

s_and_b32 s19, s15, 31 // (op 27297) 
s_lshl_b32 s23, 1, s19 // (op 27297) 
s_add_i32 s24, s23, s22 // (op 27297) 


s_and_b32 s25, s0, s24 // (op 27297) 


s_add_i32 s0, s18, s25 // (op 27297) 


s_ashr_i32 s18, s0, s19 // (op 27297) 


s_lshl_b32 s0, s15, 24 // (op 27297) 
s_ashr_i32 s19, s0, 31 // (op 27297) 

s_xor_b32 s0, s18, s19 // (op 27297) 

s_sub_i32 s18, s0, s19 // (op 27297) 


s_mul_i32 s0, s1, s11 // (op 27297) 

s_sub_u32 s1, s16, s0 // (op 27297) 

s_mul_i32 s0, s18, s11 // (op 27297) 
s_add_u32 s0, s0, s1 // (op 27297) 


s_mul_hi_i32 s1, s16, s9 // (op 27297) 
s_add_i32 s18, s1, s16 // (op 27297) 

s_ashr_i32 s1, s18, 31 // (op 27297) 
s_cmp_eq_i32 s9, 0 // (op 27297) 
s_mov_b32 s19, scc // (op 27297) 
s_cmp_eq_i32 s19, 0 // (op 27297) 
s_cselect_b32 s22, 0, -1 // (op 27297) 

s_and_b32 s19, s45, 31 // (op 27297) 
s_lshl_b32 s23, 1, s19 // (op 27297) 
s_add_i32 s24, s23, s22 // (op 27297) 


s_and_b32 s25, s1, s24 // (op 27297) 


s_add_i32 s1, s18, s25 // (op 27297) 


s_ashr_i32 s18, s1, s19 // (op 27297) 


s_lshl_b32 s1, s45, 24 // (op 27297) 
s_ashr_i32 s19, s1, 31 // (op 27297) 

s_xor_b32 s1, s18, s19 // (op 27297) 

s_sub_i32 s18, s1, s19 // (op 27297) 


s_add_u32 s22, s6, 256 // (op 27297) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27297) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27297) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27297) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27297) 

s_mul_i32 s1, s11, s22 // (op 27297) 

s_mul_i32 s19, s8, s1 // (op 27297) 

s_mul_i32 s1, s18, s19 // (op 27297) 

s_sub_u32 s19, s16, s1 // (op 27297) 

s_mov_b32 s1, s19 // (op 27297) 
s_mul_hi_i32 s22, s1, s12 // (op 27297) 
s_add_i32 s23, s22, s1 // (op 27297) 


s_ashr_i32 s1, s23, 31 // (op 27297) 
s_cmp_eq_i32 s12, 0 // (op 27297) 
s_mov_b32 s22, scc // (op 27297) 
s_cmp_eq_i32 s22, 0 // (op 27297) 
s_cselect_b32 s24, 0, -1 // (op 27297) 

s_and_b32 s22, s44, 31 // (op 27297) 
s_lshl_b32 s25, 1, s22 // (op 27297) 
s_add_i32 s47, s25, s24 // (op 27297) 


s_and_b32 s24, s1, s47 // (op 27297) 


s_add_i32 s1, s23, s24 // (op 27297) 


s_ashr_i32 s23, s1, s22 // (op 27297) 


s_lshl_b32 s1, s44, 24 // (op 27297) 
s_ashr_i32 s22, s1, 31 // (op 27297) 

s_xor_b32 s1, s23, s22 // (op 27297) 

s_sub_i32 s23, s1, s22 // (op 27297) 


s_mul_i32 s1, s46, s11 // (op 27297) 
s_add_u32 s24, s20, 256 // (op 27297) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 27297) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 27297) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 27297) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 27297) 

s_sub_i32 s22, s24, s1 // (op 27297) 

s_mul_i32 s24, s23, s22 // (op 27297) 


s_sub_u32 s25, s19, s24 // (op 27297) 


s_add_u32 s19, s25, s1 // (op 27297) 


s_cmp_eq_i32 s18, 0 // (op 27297) 
s_mov_b32 s1, scc // (op 27297) 

s_cmp_eq_i32 s1, 0 // (op 27297) 
s_cselect_b32 s18, s19, s0 // (op 27297) 



v_lshl_add_u32 v11, s18, 8, v10 // (op 27297) 


v_mul_lo_u32 v15, v11, s27 // (op 27297) most significant: low of low * high
v_mul_hi_u32 v10, v11, s26 // (op 27297) most significant: high of low * low
v_mul_lo_u32 v14, v11, s26 // (op 27297) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 27297) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 27297) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 27297) most significant half



s_mul_i32 s1, s26, s21 // (op 27299) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 27299) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 27299) least significant
s_mul_hi_u32 s19, s26, s20 // (op 27299) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 27299) most significant: sum
s_add_u32 s1, s1, s19 // (op 27299) most significant: sum


s_mov_b32 s48, s28 // (op 27299) reg expression
s_mov_b32 s49, s29 // (op 27299) reg expression
s_mov_b32 s50, s0 // (op 27299) reg expression
s_mov_b32 s51, 131072 // (op 27299) call()

s_mul_hi_i32 s47, s16, s10 // (op 27316) 
v_lshrrev_b32 v10, 3, v1 // (op 27316) 
s_add_i32 s0, s47, s16 // (op 27316) 

v_and_b32 v11, -8, v1 // (op 27316) 
s_ashr_i32 s1, s0, 31 // (op 27316) 
v_sub_u32 v12, v1, v11 // (op 27316) 

s_cmp_eq_i32 s10, 0 // (op 27316) 
s_mov_b32 s47, scc // (op 27316) 
v_lshlrev_b32 v11, 4, v12 // (op 27316) 

s_cmp_eq_i32 s47, 0 // (op 27316) 
s_cselect_b32 s18, 0, -1 // (op 27316) 

v_mov_b32 v13, 0 // (op 27316) convert
v_mov_b32 v12, v11 // (op 27316) convert

s_and_b32 s19, s13, 31 // (op 27316) 
s_lshl_b32 s47, 1, s19 // (op 27316) 
s_add_i32 s22, s47, s18 // (op 27316) 


s_and_b32 s18, s1, s22 // (op 27316) 


s_add_i32 s1, s0, s18 // (op 27316) 


s_ashr_i32 s0, s1, s19 // (op 27316) 


s_lshl_b32 s1, s13, 24 // (op 27316) 
s_ashr_i32 s47, s1, 31 // (op 27316) 

s_xor_b32 s1, s0, s47 // (op 27316) 

s_sub_i32 s0, s1, s47 // (op 27316) 


s_mul_hi_i32 s1, s0, s14 // (op 27316) 
s_add_i32 s47, s1, s0 // (op 27316) 

s_ashr_i32 s1, s47, 31 // (op 27316) 
s_cmp_eq_i32 s14, 0 // (op 27316) 
s_mov_b32 s18, scc // (op 27316) 
s_cmp_eq_i32 s18, 0 // (op 27316) 
s_cselect_b32 s19, 0, -1 // (op 27316) 

s_and_b32 s18, s15, 31 // (op 27316) 
s_lshl_b32 s22, 1, s18 // (op 27316) 
s_add_i32 s23, s22, s19 // (op 27316) 


s_and_b32 s19, s1, s23 // (op 27316) 


s_add_i32 s1, s47, s19 // (op 27316) 


s_ashr_i32 s19, s1, s18 // (op 27316) 


s_lshl_b32 s1, s15, 24 // (op 27316) 
s_ashr_i32 s18, s1, 31 // (op 27316) 

s_xor_b32 s1, s19, s18 // (op 27316) 

s_sub_i32 s19, s1, s18 // (op 27316) 


s_mul_i32 s1, s0, s11 // (op 27316) 

s_sub_u32 s0, s16, s1 // (op 27316) 

s_mul_i32 s1, s19, s11 // (op 27316) 
s_add_u32 s1, s1, s0 // (op 27316) 


s_mul_hi_i32 s0, s16, s9 // (op 27316) 
s_add_i32 s47, s0, s16 // (op 27316) 

s_ashr_i32 s0, s47, 31 // (op 27316) 
s_cmp_eq_i32 s9, 0 // (op 27316) 
s_mov_b32 s18, scc // (op 27316) 
s_cmp_eq_i32 s18, 0 // (op 27316) 
s_cselect_b32 s19, 0, -1 // (op 27316) 

s_and_b32 s18, s45, 31 // (op 27316) 
s_lshl_b32 s22, 1, s18 // (op 27316) 
s_add_i32 s23, s22, s19 // (op 27316) 


s_and_b32 s19, s0, s23 // (op 27316) 


s_add_i32 s0, s47, s19 // (op 27316) 


s_ashr_i32 s19, s0, s18 // (op 27316) 


s_lshl_b32 s0, s45, 24 // (op 27316) 
s_ashr_i32 s18, s0, 31 // (op 27316) 

s_xor_b32 s0, s19, s18 // (op 27316) 

s_sub_i32 s19, s0, s18 // (op 27316) 


s_add_u32 s22, s6, 256 // (op 27316) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27316) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27316) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27316) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27316) 

s_mul_i32 s0, s11, s22 // (op 27316) 

s_mul_i32 s18, s8, s0 // (op 27316) 

s_mul_i32 s0, s19, s18 // (op 27316) 

s_sub_u32 s18, s16, s0 // (op 27316) 

s_mov_b32 s0, s18 // (op 27316) 
s_mul_hi_i32 s47, s0, s12 // (op 27316) 
s_add_i32 s22, s47, s0 // (op 27316) 


s_ashr_i32 s0, s22, 31 // (op 27316) 
s_cmp_eq_i32 s12, 0 // (op 27316) 
s_mov_b32 s47, scc // (op 27316) 
s_cmp_eq_i32 s47, 0 // (op 27316) 
s_cselect_b32 s23, 0, -1 // (op 27316) 

s_and_b32 s47, s44, 31 // (op 27316) 
s_lshl_b32 s24, 1, s47 // (op 27316) 
s_add_i32 s25, s24, s23 // (op 27316) 


s_and_b32 s23, s0, s25 // (op 27316) 


s_add_i32 s0, s22, s23 // (op 27316) 


s_ashr_i32 s22, s0, s47 // (op 27316) 


s_lshl_b32 s0, s44, 24 // (op 27316) 
s_ashr_i32 s47, s0, 31 // (op 27316) 

s_xor_b32 s0, s22, s47 // (op 27316) 

s_sub_i32 s22, s0, s47 // (op 27316) 


s_mul_i32 s0, s46, s11 // (op 27316) 
s_add_u32 s24, s20, 256 // (op 27316) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 27316) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 27316) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 27316) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 8 // (op 27316) 

s_sub_i32 s23, s24, s0 // (op 27316) 

s_mul_i32 s47, s22, s23 // (op 27316) 


s_sub_u32 s22, s18, s47 // (op 27316) 


s_add_u32 s18, s22, s0 // (op 27316) 


s_cmp_eq_i32 s19, 0 // (op 27316) 
s_mov_b32 s0, scc // (op 27316) 

s_cmp_eq_i32 s0, 0 // (op 27316) 
s_cselect_b32 s19, s18, s1 // (op 27316) 



v_lshl_add_u32 v11, s19, 8, v10 // (op 27316) 


v_mul_lo_u32 v17, v11, s27 // (op 27316) most significant: low of low * high
v_mul_hi_u32 v10, v11, s26 // (op 27316) most significant: high of low * low
v_mul_lo_u32 v16, v11, s26 // (op 27316) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 27316) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v12 // (op 27316) least significant half
v_addc_co_u32 v17, s[0:1], v17, v13, s[0:1] // (op 27316) most significant half



s_mul_i32 s1, s26, s21 // (op 27318) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 27318) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 27318) least significant
s_mul_hi_u32 s18, s26, s20 // (op 27318) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27318) most significant: sum
s_add_u32 s1, s1, s18 // (op 27318) most significant: sum


s_mov_b32 s48, s28 // (op 27318) reg expression
s_mov_b32 s49, s29 // (op 27318) reg expression
s_mov_b32 s50, s0 // (op 27318) reg expression
s_mov_b32 s51, 131072 // (op 27318) call()

s_mul_hi_i32 s47, s16, s10 // (op 27361) 
v_and_b32 v10, 4294967294, v1 // (op 27361) 
s_add_i32 s0, s47, s16 // (op 27361) 

v_and_b32 v11, -2, v1 // (op 27361) 
s_ashr_i32 s1, s0, 31 // (op 27361) 
v_sub_u32 v12, v1, v11 // (op 27361) 

s_cmp_eq_i32 s10, 0 // (op 27361) 
s_mov_b32 s47, scc // (op 27361) 
v_lshlrev_b32 v11, 6, v12 // (op 27361) 

s_cmp_eq_i32 s47, 0 // (op 27361) 
s_cselect_b32 s18, 0, -1 // (op 27361) 

v_mov_b32 v13, 0 // (op 27361) convert
v_mov_b32 v12, v11 // (op 27361) convert

s_and_b32 s19, s13, 31 // (op 27361) 
s_lshl_b32 s47, 1, s19 // (op 27361) 
s_add_i32 s22, s47, s18 // (op 27361) 


s_and_b32 s18, s1, s22 // (op 27361) 


s_add_i32 s1, s0, s18 // (op 27361) 


s_ashr_i32 s0, s1, s19 // (op 27361) 


s_lshl_b32 s1, s13, 24 // (op 27361) 
s_ashr_i32 s47, s1, 31 // (op 27361) 

s_xor_b32 s1, s0, s47 // (op 27361) 

s_sub_i32 s0, s1, s47 // (op 27361) 


s_mul_hi_i32 s1, s0, s14 // (op 27361) 
s_add_i32 s47, s1, s0 // (op 27361) 

s_ashr_i32 s1, s47, 31 // (op 27361) 
s_cmp_eq_i32 s14, 0 // (op 27361) 
s_mov_b32 s18, scc // (op 27361) 
s_cmp_eq_i32 s18, 0 // (op 27361) 
s_cselect_b32 s19, 0, -1 // (op 27361) 

s_and_b32 s18, s15, 31 // (op 27361) 
s_lshl_b32 s22, 1, s18 // (op 27361) 
s_add_i32 s23, s22, s19 // (op 27361) 


s_and_b32 s19, s1, s23 // (op 27361) 


s_add_i32 s1, s47, s19 // (op 27361) 


s_ashr_i32 s19, s1, s18 // (op 27361) 


s_lshl_b32 s1, s15, 24 // (op 27361) 
s_ashr_i32 s18, s1, 31 // (op 27361) 

s_xor_b32 s1, s19, s18 // (op 27361) 

s_sub_i32 s19, s1, s18 // (op 27361) 


s_add_u32 s22, s6, 256 // (op 27361) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27361) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27361) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27361) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27361) 

s_mul_i32 s1, s19, s22 // (op 27361) 

s_sub_i32 s47, s0, s1 // (op 27361) 


s_mul_hi_i32 s0, s16, s9 // (op 27361) 
s_add_i32 s1, s0, s16 // (op 27361) 

s_ashr_i32 s0, s1, 31 // (op 27361) 
s_cmp_eq_i32 s9, 0 // (op 27361) 
s_mov_b32 s18, scc // (op 27361) 
s_cmp_eq_i32 s18, 0 // (op 27361) 
s_cselect_b32 s19, 0, -1 // (op 27361) 

s_and_b32 s18, s45, 31 // (op 27361) 
s_lshl_b32 s24, 1, s18 // (op 27361) 
s_add_i32 s25, s24, s19 // (op 27361) 


s_and_b32 s19, s0, s25 // (op 27361) 


s_add_i32 s0, s1, s19 // (op 27361) 


s_ashr_i32 s1, s0, s18 // (op 27361) 


s_lshl_b32 s0, s45, 24 // (op 27361) 
s_ashr_i32 s18, s0, 31 // (op 27361) 

s_xor_b32 s0, s1, s18 // (op 27361) 

s_sub_i32 s1, s0, s18 // (op 27361) 


s_mul_i32 s0, s11, s22 // (op 27361) 

s_mul_i32 s18, s8, s0 // (op 27361) 

s_mul_i32 s0, s1, s18 // (op 27361) 

s_sub_u32 s18, s16, s0 // (op 27361) 

s_mul_hi_i32 s0, s18, s12 // (op 27361) 
s_add_i32 s19, s0, s18 // (op 27361) 


s_ashr_i32 s0, s19, 31 // (op 27361) 
s_cmp_eq_i32 s12, 0 // (op 27361) 
s_mov_b32 s18, scc // (op 27361) 
s_cmp_eq_i32 s18, 0 // (op 27361) 
s_cselect_b32 s22, 0, -1 // (op 27361) 

s_and_b32 s18, s44, 31 // (op 27361) 
s_lshl_b32 s23, 1, s18 // (op 27361) 
s_add_i32 s24, s23, s22 // (op 27361) 


s_and_b32 s25, s0, s24 // (op 27361) 


s_add_i32 s0, s19, s25 // (op 27361) 


s_ashr_i32 s19, s0, s18 // (op 27361) 


s_lshl_b32 s0, s44, 24 // (op 27361) 
s_ashr_i32 s18, s0, 31 // (op 27361) 

s_xor_b32 s0, s19, s18 // (op 27361) 

s_sub_i32 s19, s0, s18 // (op 27361) 


s_cmp_eq_i32 s1, 0 // (op 27361) 
s_mov_b32 s0, scc // (op 27361) 

s_cmp_eq_i32 s0, 0 // (op 27361) 
s_cselect_b32 s1, s19, s47 // (op 27361) 



v_lshl_add_u32 v11, s1, 8, v10 // (op 27361) 


v_mul_lo_u32 v19, v11, s31 // (op 27361) most significant: low of low * high
v_mul_hi_u32 v10, v11, s30 // (op 27361) most significant: high of low * low
v_mul_lo_u32 v18, v11, s30 // (op 27361) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 27361) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v18 // (op 27361) least significant half
v_addc_co_u32 v11, s[0:1], v13, v19, s[0:1] // (op 27361) most significant half



v_mul_lo_u32 v13, v11, 6 // (op 27361) most significant: low of high * low
v_mul_hi_u32 v18, v10, 6 // (op 27361) most significant: high of low * low
v_mul_lo_u32 v12, v10, 6 // (op 27361) least significant: low of low * low
v_add_u32 v13, v13, v18 // (op 27361) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 27361) 

v_mov_b32 v12, v10 // (op 27361) convert

s_mul_i32 s1, s7, 6 // (op 27363) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 27363) least significant
s_mul_hi_u32 s47, s6, 6 // (op 27363) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27363) most significant: sum

s_mul_i32 s19, s30, s1 // (op 27363) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 27363) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 27363) least significant
s_mul_hi_u32 s22, s30, s0 // (op 27363) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 27363) most significant: sum
s_add_u32 s19, s19, s22 // (op 27363) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 27363) 

s_mov_b32 s60, s32 // (op 27363) reg expression
s_mov_b32 s61, s33 // (op 27363) reg expression
s_mov_b32 s62, s0 // (op 27363) reg expression
s_mov_b32 s63, 131072 // (op 27363) call()

s_mul_hi_i32 s47, s16, s10 // (op 27380) 
v_and_b32 v13, 4294967294, v1 // (op 27380) 
s_add_i32 s0, s47, s16 // (op 27380) 

v_and_b32 v10, -2, v1 // (op 27380) 
s_ashr_i32 s1, s0, 31 // (op 27380) 
v_sub_u32 v11, v1, v10 // (op 27380) 

s_cmp_eq_i32 s10, 0 // (op 27380) 
s_mov_b32 s47, scc // (op 27380) 
v_lshlrev_b32 v10, 6, v11 // (op 27380) 

s_cmp_eq_i32 s47, 0 // (op 27380) 
s_cselect_b32 s18, 0, -1 // (op 27380) 

v_mov_b32 v19, 0 // (op 27380) convert
v_mov_b32 v18, v10 // (op 27380) convert

s_and_b32 s19, s13, 31 // (op 27380) 
s_lshl_b32 s47, 1, s19 // (op 27380) 
s_add_i32 s22, s47, s18 // (op 27380) 


s_and_b32 s18, s1, s22 // (op 27380) 


s_add_i32 s1, s0, s18 // (op 27380) 


s_ashr_i32 s0, s1, s19 // (op 27380) 


s_lshl_b32 s1, s13, 24 // (op 27380) 
s_ashr_i32 s47, s1, 31 // (op 27380) 

s_xor_b32 s1, s0, s47 // (op 27380) 

s_sub_i32 s0, s1, s47 // (op 27380) 


s_mul_hi_i32 s1, s0, s14 // (op 27380) 
s_add_i32 s47, s1, s0 // (op 27380) 

s_ashr_i32 s1, s47, 31 // (op 27380) 
s_cmp_eq_i32 s14, 0 // (op 27380) 
s_mov_b32 s18, scc // (op 27380) 
s_cmp_eq_i32 s18, 0 // (op 27380) 
s_cselect_b32 s19, 0, -1 // (op 27380) 

s_and_b32 s18, s15, 31 // (op 27380) 
s_lshl_b32 s22, 1, s18 // (op 27380) 
s_add_i32 s23, s22, s19 // (op 27380) 


s_and_b32 s19, s1, s23 // (op 27380) 


s_add_i32 s1, s47, s19 // (op 27380) 


s_ashr_i32 s19, s1, s18 // (op 27380) 


s_lshl_b32 s1, s15, 24 // (op 27380) 
s_ashr_i32 s18, s1, 31 // (op 27380) 

s_xor_b32 s1, s19, s18 // (op 27380) 

s_sub_i32 s19, s1, s18 // (op 27380) 


s_add_u32 s22, s6, 256 // (op 27380) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27380) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27380) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27380) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27380) 

s_mul_i32 s1, s19, s22 // (op 27380) 

s_sub_i32 s47, s0, s1 // (op 27380) 


s_mul_hi_i32 s0, s16, s9 // (op 27380) 
s_add_i32 s1, s0, s16 // (op 27380) 

s_ashr_i32 s0, s1, 31 // (op 27380) 
s_cmp_eq_i32 s9, 0 // (op 27380) 
s_mov_b32 s18, scc // (op 27380) 
s_cmp_eq_i32 s18, 0 // (op 27380) 
s_cselect_b32 s19, 0, -1 // (op 27380) 

s_and_b32 s18, s45, 31 // (op 27380) 
s_lshl_b32 s24, 1, s18 // (op 27380) 
s_add_i32 s25, s24, s19 // (op 27380) 


s_and_b32 s19, s0, s25 // (op 27380) 


s_add_i32 s0, s1, s19 // (op 27380) 


s_ashr_i32 s1, s0, s18 // (op 27380) 


s_lshl_b32 s0, s45, 24 // (op 27380) 
s_ashr_i32 s18, s0, 31 // (op 27380) 

s_xor_b32 s0, s1, s18 // (op 27380) 

s_sub_i32 s1, s0, s18 // (op 27380) 


s_mul_i32 s0, s11, s22 // (op 27380) 

s_mul_i32 s18, s8, s0 // (op 27380) 

s_mul_i32 s0, s1, s18 // (op 27380) 

s_sub_u32 s18, s16, s0 // (op 27380) 

s_mul_hi_i32 s0, s18, s12 // (op 27380) 
s_add_i32 s19, s0, s18 // (op 27380) 


s_ashr_i32 s0, s19, 31 // (op 27380) 
s_cmp_eq_i32 s12, 0 // (op 27380) 
s_mov_b32 s18, scc // (op 27380) 
s_cmp_eq_i32 s18, 0 // (op 27380) 
s_cselect_b32 s22, 0, -1 // (op 27380) 

s_and_b32 s18, s44, 31 // (op 27380) 
s_lshl_b32 s23, 1, s18 // (op 27380) 
s_add_i32 s24, s23, s22 // (op 27380) 


s_and_b32 s25, s0, s24 // (op 27380) 


s_add_i32 s0, s19, s25 // (op 27380) 


s_ashr_i32 s19, s0, s18 // (op 27380) 


s_lshl_b32 s0, s44, 24 // (op 27380) 
s_ashr_i32 s18, s0, 31 // (op 27380) 

s_xor_b32 s0, s19, s18 // (op 27380) 

s_sub_i32 s19, s0, s18 // (op 27380) 


s_cmp_eq_i32 s1, 0 // (op 27380) 
s_mov_b32 s0, scc // (op 27380) 

s_cmp_eq_i32 s0, 0 // (op 27380) 
s_cselect_b32 s1, s19, s47 // (op 27380) 



v_lshl_add_u32 v10, s1, 8, v13 // (op 27380) 


v_mul_lo_u32 v21, v10, s31 // (op 27380) most significant: low of low * high
v_mul_hi_u32 v11, v10, s30 // (op 27380) most significant: high of low * low
v_mul_lo_u32 v20, v10, s30 // (op 27380) least significant: low of low * low
v_add_u32 v21, v21, v11 // (op 27380) most significant: sum


v_add_co_u32 v10, s[0:1], v18, v20 // (op 27380) least significant half
v_addc_co_u32 v11, s[0:1], v19, v21, s[0:1] // (op 27380) most significant half



v_mul_lo_u32 v19, v11, 6 // (op 27380) most significant: low of high * low
v_mul_hi_u32 v13, v10, 6 // (op 27380) most significant: high of low * low
v_mul_lo_u32 v18, v10, 6 // (op 27380) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 27380) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[18:19] // (op 27380) 

v_mov_b32 v13, v10 // (op 27380) convert

s_mul_i32 s1, s7, 6 // (op 27382) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 27382) least significant
s_mul_hi_u32 s47, s6, 6 // (op 27382) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27382) most significant: sum

s_mul_i32 s19, s30, s1 // (op 27382) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 27382) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 27382) least significant
s_mul_hi_u32 s22, s30, s0 // (op 27382) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 27382) most significant: sum
s_add_u32 s19, s19, s22 // (op 27382) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 27382) 

s_mov_b32 s60, s32 // (op 27382) reg expression
s_mov_b32 s61, s33 // (op 27382) reg expression
s_mov_b32 s62, s0 // (op 27382) reg expression
s_mov_b32 s63, 131072 // (op 27382) call()

s_mul_hi_i32 s47, s16, s10 // (op 27644) 
v_and_b32 v10, 63, v1 // (op 27644) 
s_add_i32 s0, s47, s16 // (op 27644) 

v_lshrrev_b32 v11, 4, v10 // (op 27644) 
s_ashr_i32 s1, s0, 31 // (op 27644) 
v_and_b32 v18, 3, v11 // (op 27644) 

s_cmp_eq_i32 s10, 0 // (op 27644) 
s_mov_b32 s47, scc // (op 27644) 
v_and_b32 v11, 15, v1 // (op 27644) 
s_cmp_eq_i32 s47, 0 // (op 27644) 
s_cselect_b32 s18, 0, -1 // (op 27644) 

v_lshl_add_u32 v19, v18, 4, v11 // (op 27644) 


s_and_b32 s19, s13, 31 // (op 27644) 
v_and_b32 v11, 4294967168, v1 // (op 27644) 
s_lshl_b32 s47, 1, s19 // (op 27644) 
v_add_u32 v18, v11, v19 // (op 27644) 


s_add_i32 s22, s47, s18 // (op 27644) 


v_lshrrev_b32 v11, 6, v10 // (op 27644) 

s_and_b32 s18, s1, s22 // (op 27644) 


v_lshlrev_b32 v10, 2, v11 // (op 27644) 

s_add_i32 s1, s0, s18 // (op 27644) 


s_ashr_i32 s0, s1, s19 // (op 27644) 


s_lshl_b32 s1, s13, 24 // (op 27644) 
s_ashr_i32 s47, s1, 31 // (op 27644) 

s_xor_b32 s1, s0, s47 // (op 27644) 

s_sub_i32 s0, s1, s47 // (op 27644) 


s_mul_hi_i32 s1, s0, s14 // (op 27644) 
s_add_i32 s47, s1, s0 // (op 27644) 

s_ashr_i32 s1, s47, 31 // (op 27644) 
s_cmp_eq_i32 s14, 0 // (op 27644) 
s_mov_b32 s18, scc // (op 27644) 
s_cmp_eq_i32 s18, 0 // (op 27644) 
s_cselect_b32 s19, 0, -1 // (op 27644) 

s_and_b32 s18, s15, 31 // (op 27644) 
s_lshl_b32 s22, 1, s18 // (op 27644) 
s_add_i32 s23, s22, s19 // (op 27644) 


s_and_b32 s19, s1, s23 // (op 27644) 


s_add_i32 s1, s47, s19 // (op 27644) 


s_ashr_i32 s19, s1, s18 // (op 27644) 


s_lshl_b32 s1, s15, 24 // (op 27644) 
s_ashr_i32 s18, s1, 31 // (op 27644) 

s_xor_b32 s1, s19, s18 // (op 27644) 

s_sub_i32 s19, s1, s18 // (op 27644) 


s_mul_i32 s1, s0, s11 // (op 27644) 

s_sub_u32 s0, s16, s1 // (op 27644) 

s_mul_i32 s1, s19, s11 // (op 27644) 
s_add_u32 s1, s1, s0 // (op 27644) 


s_mul_hi_i32 s0, s16, s9 // (op 27644) 
s_add_i32 s47, s0, s16 // (op 27644) 

s_ashr_i32 s0, s47, 31 // (op 27644) 
s_cmp_eq_i32 s9, 0 // (op 27644) 
s_mov_b32 s18, scc // (op 27644) 
s_cmp_eq_i32 s18, 0 // (op 27644) 
s_cselect_b32 s19, 0, -1 // (op 27644) 

s_and_b32 s18, s45, 31 // (op 27644) 
s_lshl_b32 s22, 1, s18 // (op 27644) 
s_add_i32 s23, s22, s19 // (op 27644) 


s_and_b32 s19, s0, s23 // (op 27644) 


s_add_i32 s0, s47, s19 // (op 27644) 


s_ashr_i32 s19, s0, s18 // (op 27644) 


s_lshl_b32 s0, s45, 24 // (op 27644) 
s_ashr_i32 s18, s0, 31 // (op 27644) 

s_xor_b32 s0, s19, s18 // (op 27644) 

s_sub_i32 s19, s0, s18 // (op 27644) 


s_add_u32 s22, s6, 256 // (op 27644) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27644) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27644) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27644) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27644) 

s_mul_i32 s0, s11, s22 // (op 27644) 

s_mul_i32 s18, s8, s0 // (op 27644) 

s_mul_i32 s0, s19, s18 // (op 27644) 

s_sub_u32 s18, s16, s0 // (op 27644) 

s_mov_b32 s0, s18 // (op 27644) 
s_mul_hi_i32 s47, s0, s12 // (op 27644) 
s_add_i32 s22, s47, s0 // (op 27644) 


s_ashr_i32 s0, s22, 31 // (op 27644) 
s_cmp_eq_i32 s12, 0 // (op 27644) 
s_mov_b32 s47, scc // (op 27644) 
s_cmp_eq_i32 s47, 0 // (op 27644) 
s_cselect_b32 s23, 0, -1 // (op 27644) 

s_and_b32 s47, s44, 31 // (op 27644) 
s_lshl_b32 s24, 1, s47 // (op 27644) 
s_add_i32 s25, s24, s23 // (op 27644) 


s_and_b32 s23, s0, s25 // (op 27644) 


s_add_i32 s0, s22, s23 // (op 27644) 


s_ashr_i32 s22, s0, s47 // (op 27644) 


s_lshl_b32 s0, s44, 24 // (op 27644) 
s_ashr_i32 s47, s0, 31 // (op 27644) 

s_xor_b32 s0, s22, s47 // (op 27644) 

s_sub_i32 s22, s0, s47 // (op 27644) 


s_mul_i32 s0, s46, s11 // (op 27644) 
s_add_u32 s24, s20, 256 // (op 27644) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 27644) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 27644) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 27644) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 27644) 

s_sub_i32 s23, s24, s0 // (op 27644) 

s_mul_i32 s47, s22, s23 // (op 27644) 


s_sub_u32 s22, s18, s47 // (op 27644) 


s_add_u32 s18, s22, s0 // (op 27644) 


s_cmp_eq_i32 s19, 0 // (op 27644) 
s_mov_b32 s0, scc // (op 27644) 

s_cmp_eq_i32 s0, 0 // (op 27644) 
s_cselect_b32 s19, s18, s1 // (op 27644) 



v_lshl_add_u32 v11, s19, 8, v18 // (op 27644) 


v_mul_lo_u32 v18, v11, s34 // (op 27644) 
v_add_u32 v18, v18, v10 // (op 27644) 


s_mul_i32 s1, s34, s21 // (op 27646) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 27646) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 27646) least significant
s_mul_hi_u32 s18, s34, s20 // (op 27646) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27646) most significant: sum
s_add_u32 s1, s1, s18 // (op 27646) most significant: sum


s_mov_b32 s64, s36 // (op 27646) reg expression
s_mov_b32 s65, s37 // (op 27646) reg expression
s_mov_b32 s66, s0 // (op 27646) reg expression
s_mov_b32 s67, 131072 // (op 27646) call()

s_mul_hi_i32 s47, s16, s10 // (op 27665) 
v_and_b32 v10, 63, v1 // (op 27665) 
s_add_i32 s0, s47, s16 // (op 27665) 

v_lshrrev_b32 v11, 4, v10 // (op 27665) 
s_ashr_i32 s1, s0, 31 // (op 27665) 
v_and_b32 v19, 3, v11 // (op 27665) 

s_cmp_eq_i32 s10, 0 // (op 27665) 
s_mov_b32 s47, scc // (op 27665) 
v_and_b32 v11, 15, v1 // (op 27665) 
s_cmp_eq_i32 s47, 0 // (op 27665) 
s_cselect_b32 s18, 0, -1 // (op 27665) 

v_lshl_add_u32 v20, v19, 4, v11 // (op 27665) 


s_and_b32 s19, s13, 31 // (op 27665) 
v_and_b32 v11, 4294967168, v1 // (op 27665) 
s_lshl_b32 s47, 1, s19 // (op 27665) 
v_add_u32 v19, v11, v20 // (op 27665) 


s_add_i32 s22, s47, s18 // (op 27665) 


v_lshrrev_b32 v11, 6, v10 // (op 27665) 

s_and_b32 s18, s1, s22 // (op 27665) 


v_lshlrev_b32 v10, 2, v11 // (op 27665) 

s_add_i32 s1, s0, s18 // (op 27665) 


s_ashr_i32 s0, s1, s19 // (op 27665) 


s_lshl_b32 s1, s13, 24 // (op 27665) 
s_ashr_i32 s47, s1, 31 // (op 27665) 

s_xor_b32 s1, s0, s47 // (op 27665) 

s_sub_i32 s0, s1, s47 // (op 27665) 


s_mul_hi_i32 s1, s0, s14 // (op 27665) 
s_add_i32 s47, s1, s0 // (op 27665) 

s_ashr_i32 s1, s47, 31 // (op 27665) 
s_cmp_eq_i32 s14, 0 // (op 27665) 
s_mov_b32 s18, scc // (op 27665) 
s_cmp_eq_i32 s18, 0 // (op 27665) 
s_cselect_b32 s19, 0, -1 // (op 27665) 

s_and_b32 s18, s15, 31 // (op 27665) 
s_lshl_b32 s22, 1, s18 // (op 27665) 
s_add_i32 s23, s22, s19 // (op 27665) 


s_and_b32 s19, s1, s23 // (op 27665) 


s_add_i32 s1, s47, s19 // (op 27665) 


s_ashr_i32 s19, s1, s18 // (op 27665) 


s_lshl_b32 s1, s15, 24 // (op 27665) 
s_ashr_i32 s18, s1, 31 // (op 27665) 

s_xor_b32 s1, s19, s18 // (op 27665) 

s_sub_i32 s19, s1, s18 // (op 27665) 


s_mul_i32 s1, s0, s11 // (op 27665) 

s_sub_u32 s0, s16, s1 // (op 27665) 

s_mul_i32 s1, s19, s11 // (op 27665) 
s_add_u32 s1, s1, s0 // (op 27665) 


s_mul_hi_i32 s0, s16, s9 // (op 27665) 
s_add_i32 s47, s0, s16 // (op 27665) 

s_ashr_i32 s0, s47, 31 // (op 27665) 
s_cmp_eq_i32 s9, 0 // (op 27665) 
s_mov_b32 s18, scc // (op 27665) 
s_cmp_eq_i32 s18, 0 // (op 27665) 
s_cselect_b32 s19, 0, -1 // (op 27665) 

s_and_b32 s18, s45, 31 // (op 27665) 
s_lshl_b32 s22, 1, s18 // (op 27665) 
s_add_i32 s23, s22, s19 // (op 27665) 


s_and_b32 s19, s0, s23 // (op 27665) 


s_add_i32 s0, s47, s19 // (op 27665) 


s_ashr_i32 s19, s0, s18 // (op 27665) 


s_lshl_b32 s0, s45, 24 // (op 27665) 
s_ashr_i32 s18, s0, 31 // (op 27665) 

s_xor_b32 s0, s19, s18 // (op 27665) 

s_sub_i32 s19, s0, s18 // (op 27665) 


s_add_u32 s22, s6, 256 // (op 27665) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27665) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27665) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27665) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27665) 

s_mul_i32 s0, s11, s22 // (op 27665) 

s_mul_i32 s18, s8, s0 // (op 27665) 

s_mul_i32 s0, s19, s18 // (op 27665) 

s_sub_u32 s18, s16, s0 // (op 27665) 

s_mov_b32 s0, s18 // (op 27665) 
s_mul_hi_i32 s47, s0, s12 // (op 27665) 
s_add_i32 s22, s47, s0 // (op 27665) 


s_ashr_i32 s0, s22, 31 // (op 27665) 
s_cmp_eq_i32 s12, 0 // (op 27665) 
s_mov_b32 s47, scc // (op 27665) 
s_cmp_eq_i32 s47, 0 // (op 27665) 
s_cselect_b32 s23, 0, -1 // (op 27665) 

s_and_b32 s47, s44, 31 // (op 27665) 
s_lshl_b32 s24, 1, s47 // (op 27665) 
s_add_i32 s25, s24, s23 // (op 27665) 


s_and_b32 s23, s0, s25 // (op 27665) 


s_add_i32 s0, s22, s23 // (op 27665) 


s_ashr_i32 s22, s0, s47 // (op 27665) 


s_lshl_b32 s0, s44, 24 // (op 27665) 
s_ashr_i32 s47, s0, 31 // (op 27665) 

s_xor_b32 s0, s22, s47 // (op 27665) 

s_sub_i32 s22, s0, s47 // (op 27665) 


s_mul_i32 s0, s46, s11 // (op 27665) 
s_add_u32 s24, s20, 256 // (op 27665) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 27665) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 27665) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 27665) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 8 // (op 27665) 

s_sub_i32 s23, s24, s0 // (op 27665) 

s_mul_i32 s47, s22, s23 // (op 27665) 


s_sub_u32 s22, s18, s47 // (op 27665) 


s_add_u32 s18, s22, s0 // (op 27665) 


s_cmp_eq_i32 s19, 0 // (op 27665) 
s_mov_b32 s0, scc // (op 27665) 

s_cmp_eq_i32 s0, 0 // (op 27665) 
s_cselect_b32 s19, s18, s1 // (op 27665) 



v_lshl_add_u32 v11, s19, 8, v19 // (op 27665) 


v_mul_lo_u32 v19, v11, s34 // (op 27665) 
v_add_u32 v19, v19, v10 // (op 27665) 


s_mul_i32 s1, s34, s21 // (op 27667) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 27667) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 27667) least significant
s_mul_hi_u32 s18, s34, s20 // (op 27667) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27667) most significant: sum
s_add_u32 s1, s1, s18 // (op 27667) most significant: sum


s_mov_b32 s64, s36 // (op 27667) reg expression
s_mov_b32 s65, s37 // (op 27667) reg expression
s_mov_b32 s66, s0 // (op 27667) reg expression
s_mov_b32 s67, 131072 // (op 27667) call()

s_mul_hi_i32 s47, s16, s10 // (op 27696) 
v_and_b32 v10, 63, v1 // (op 27696) 
s_add_i32 s0, s47, s16 // (op 27696) 

v_lshrrev_b32 v11, 4, v10 // (op 27696) 
s_ashr_i32 s1, s0, 31 // (op 27696) 
v_and_b32 v20, 3, v11 // (op 27696) 

s_cmp_eq_i32 s10, 0 // (op 27696) 
s_mov_b32 s47, scc // (op 27696) 
v_and_b32 v11, 15, v1 // (op 27696) 
s_cmp_eq_i32 s47, 0 // (op 27696) 
s_cselect_b32 s18, 0, -1 // (op 27696) 

v_lshl_add_u32 v21, v20, 4, v11 // (op 27696) 


s_and_b32 s19, s13, 31 // (op 27696) 
v_lshrrev_b32 v11, 6, v1 // (op 27696) 
s_lshl_b32 s47, 1, s19 // (op 27696) 
v_and_b32 v20, 1, v11 // (op 27696) 

s_add_i32 s22, s47, s18 // (op 27696) 


v_lshl_add_u32 v11, v20, 7, v21 // (op 27696) 


s_and_b32 s18, s1, s22 // (op 27696) 


v_lshrrev_b32 v20, 6, v10 // (op 27696) 

s_add_i32 s1, s0, s18 // (op 27696) 


s_ashr_i32 s0, s1, s19 // (op 27696) 


s_lshl_b32 s1, s13, 24 // (op 27696) 
s_ashr_i32 s47, s1, 31 // (op 27696) 

s_xor_b32 s1, s0, s47 // (op 27696) 

s_sub_i32 s0, s1, s47 // (op 27696) 


s_mul_hi_i32 s1, s0, s14 // (op 27696) 
s_add_i32 s47, s1, s0 // (op 27696) 

s_ashr_i32 s1, s47, 31 // (op 27696) 
s_cmp_eq_i32 s14, 0 // (op 27696) 
s_mov_b32 s18, scc // (op 27696) 
s_cmp_eq_i32 s18, 0 // (op 27696) 
s_cselect_b32 s19, 0, -1 // (op 27696) 

s_and_b32 s18, s15, 31 // (op 27696) 
s_lshl_b32 s22, 1, s18 // (op 27696) 
s_add_i32 s23, s22, s19 // (op 27696) 


s_and_b32 s19, s1, s23 // (op 27696) 


s_add_i32 s1, s47, s19 // (op 27696) 


s_ashr_i32 s19, s1, s18 // (op 27696) 


s_lshl_b32 s1, s15, 24 // (op 27696) 
s_ashr_i32 s18, s1, 31 // (op 27696) 

s_xor_b32 s1, s19, s18 // (op 27696) 

s_sub_i32 s19, s1, s18 // (op 27696) 


s_add_u32 s22, s6, 256 // (op 27696) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27696) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27696) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27696) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27696) 

s_mul_i32 s1, s19, s22 // (op 27696) 

s_sub_i32 s47, s0, s1 // (op 27696) 


s_mul_hi_i32 s0, s16, s9 // (op 27696) 
s_add_i32 s1, s0, s16 // (op 27696) 

s_ashr_i32 s0, s1, 31 // (op 27696) 
s_cmp_eq_i32 s9, 0 // (op 27696) 
s_mov_b32 s18, scc // (op 27696) 
s_cmp_eq_i32 s18, 0 // (op 27696) 
s_cselect_b32 s19, 0, -1 // (op 27696) 

s_and_b32 s18, s45, 31 // (op 27696) 
s_lshl_b32 s24, 1, s18 // (op 27696) 
s_add_i32 s25, s24, s19 // (op 27696) 


s_and_b32 s19, s0, s25 // (op 27696) 


s_add_i32 s0, s1, s19 // (op 27696) 


s_ashr_i32 s1, s0, s18 // (op 27696) 


s_lshl_b32 s0, s45, 24 // (op 27696) 
s_ashr_i32 s18, s0, 31 // (op 27696) 

s_xor_b32 s0, s1, s18 // (op 27696) 

s_sub_i32 s1, s0, s18 // (op 27696) 


s_mul_i32 s0, s11, s22 // (op 27696) 

s_mul_i32 s18, s8, s0 // (op 27696) 

s_mul_i32 s0, s1, s18 // (op 27696) 

s_sub_u32 s18, s16, s0 // (op 27696) 

s_mul_hi_i32 s0, s18, s12 // (op 27696) 
s_add_i32 s19, s0, s18 // (op 27696) 


s_ashr_i32 s0, s19, 31 // (op 27696) 
s_cmp_eq_i32 s12, 0 // (op 27696) 
s_mov_b32 s18, scc // (op 27696) 
s_cmp_eq_i32 s18, 0 // (op 27696) 
s_cselect_b32 s22, 0, -1 // (op 27696) 

s_and_b32 s18, s44, 31 // (op 27696) 
s_lshl_b32 s23, 1, s18 // (op 27696) 
s_add_i32 s24, s23, s22 // (op 27696) 


s_and_b32 s25, s0, s24 // (op 27696) 


s_add_i32 s0, s19, s25 // (op 27696) 


s_ashr_i32 s19, s0, s18 // (op 27696) 


s_lshl_b32 s0, s44, 24 // (op 27696) 
s_ashr_i32 s18, s0, 31 // (op 27696) 

s_xor_b32 s0, s19, s18 // (op 27696) 

s_sub_i32 s19, s0, s18 // (op 27696) 


s_cmp_eq_i32 s1, 0 // (op 27696) 
s_mov_b32 s0, scc // (op 27696) 

s_cmp_eq_i32 s0, 0 // (op 27696) 
s_cselect_b32 s1, s19, s47 // (op 27696) 



v_lshl_add_u32 v10, s1, 8, v11 // (op 27696) 


v_mul_lo_u32 v11, v10, s38 // (op 27696) 

v_lshl_add_u32 v10, v20, 2, v11 // (op 27696) 


s_mul_i32 s1, s38, s7 // (op 27698) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 27698) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 27698) least significant
s_mul_hi_u32 s18, s38, s6 // (op 27698) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27698) most significant: sum
s_add_u32 s1, s1, s18 // (op 27698) most significant: sum


s_mov_b32 s68, s40 // (op 27698) reg expression
s_mov_b32 s69, s41 // (op 27698) reg expression
s_mov_b32 s70, s0 // (op 27698) reg expression
s_mov_b32 s71, 131072 // (op 27698) call()

s_mul_hi_i32 s47, s16, s10 // (op 27717) 
v_and_b32 v11, 63, v1 // (op 27717) 
s_add_i32 s0, s47, s16 // (op 27717) 

v_lshrrev_b32 v20, 4, v11 // (op 27717) 
s_ashr_i32 s1, s0, 31 // (op 27717) 
v_and_b32 v21, 3, v20 // (op 27717) 

s_cmp_eq_i32 s10, 0 // (op 27717) 
s_mov_b32 s47, scc // (op 27717) 
v_and_b32 v20, 15, v1 // (op 27717) 
s_cmp_eq_i32 s47, 0 // (op 27717) 
s_cselect_b32 s18, 0, -1 // (op 27717) 

v_lshl_add_u32 v22, v21, 4, v20 // (op 27717) 


s_and_b32 s19, s13, 31 // (op 27717) 
v_lshrrev_b32 v20, 6, v1 // (op 27717) 
s_lshl_b32 s47, 1, s19 // (op 27717) 
v_and_b32 v21, 1, v20 // (op 27717) 

s_add_i32 s22, s47, s18 // (op 27717) 


v_lshl_add_u32 v20, v21, 7, v22 // (op 27717) 


s_and_b32 s18, s1, s22 // (op 27717) 


v_lshrrev_b32 v21, 6, v11 // (op 27717) 

s_add_i32 s1, s0, s18 // (op 27717) 


s_ashr_i32 s0, s1, s19 // (op 27717) 


s_lshl_b32 s1, s13, 24 // (op 27717) 
s_ashr_i32 s47, s1, 31 // (op 27717) 

s_xor_b32 s1, s0, s47 // (op 27717) 

s_sub_i32 s0, s1, s47 // (op 27717) 


s_mul_hi_i32 s1, s0, s14 // (op 27717) 
s_add_i32 s47, s1, s0 // (op 27717) 

s_ashr_i32 s1, s47, 31 // (op 27717) 
s_cmp_eq_i32 s14, 0 // (op 27717) 
s_mov_b32 s18, scc // (op 27717) 
s_cmp_eq_i32 s18, 0 // (op 27717) 
s_cselect_b32 s19, 0, -1 // (op 27717) 

s_and_b32 s18, s15, 31 // (op 27717) 
s_lshl_b32 s22, 1, s18 // (op 27717) 
s_add_i32 s23, s22, s19 // (op 27717) 


s_and_b32 s19, s1, s23 // (op 27717) 


s_add_i32 s1, s47, s19 // (op 27717) 


s_ashr_i32 s19, s1, s18 // (op 27717) 


s_lshl_b32 s1, s15, 24 // (op 27717) 
s_ashr_i32 s18, s1, 31 // (op 27717) 

s_xor_b32 s1, s19, s18 // (op 27717) 

s_sub_i32 s19, s1, s18 // (op 27717) 


s_add_u32 s22, s6, 256 // (op 27717) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27717) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27717) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27717) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27717) 

s_mul_i32 s1, s19, s22 // (op 27717) 

s_sub_i32 s47, s0, s1 // (op 27717) 


s_mul_hi_i32 s0, s16, s9 // (op 27717) 
s_add_i32 s1, s0, s16 // (op 27717) 

s_ashr_i32 s0, s1, 31 // (op 27717) 
s_cmp_eq_i32 s9, 0 // (op 27717) 
s_mov_b32 s18, scc // (op 27717) 
s_cmp_eq_i32 s18, 0 // (op 27717) 
s_cselect_b32 s19, 0, -1 // (op 27717) 

s_and_b32 s18, s45, 31 // (op 27717) 
s_lshl_b32 s24, 1, s18 // (op 27717) 
s_add_i32 s25, s24, s19 // (op 27717) 


s_and_b32 s19, s0, s25 // (op 27717) 


s_add_i32 s0, s1, s19 // (op 27717) 


s_ashr_i32 s1, s0, s18 // (op 27717) 


s_lshl_b32 s0, s45, 24 // (op 27717) 
s_ashr_i32 s18, s0, 31 // (op 27717) 

s_xor_b32 s0, s1, s18 // (op 27717) 

s_sub_i32 s1, s0, s18 // (op 27717) 


s_mul_i32 s0, s11, s22 // (op 27717) 

s_mul_i32 s18, s8, s0 // (op 27717) 

s_mul_i32 s0, s1, s18 // (op 27717) 

s_sub_u32 s18, s16, s0 // (op 27717) 

s_mul_hi_i32 s0, s18, s12 // (op 27717) 
s_add_i32 s19, s0, s18 // (op 27717) 


s_ashr_i32 s0, s19, 31 // (op 27717) 
s_cmp_eq_i32 s12, 0 // (op 27717) 
s_mov_b32 s18, scc // (op 27717) 
s_cmp_eq_i32 s18, 0 // (op 27717) 
s_cselect_b32 s22, 0, -1 // (op 27717) 

s_and_b32 s18, s44, 31 // (op 27717) 
s_lshl_b32 s23, 1, s18 // (op 27717) 
s_add_i32 s24, s23, s22 // (op 27717) 


s_and_b32 s25, s0, s24 // (op 27717) 


s_add_i32 s0, s19, s25 // (op 27717) 


s_ashr_i32 s19, s0, s18 // (op 27717) 


s_lshl_b32 s0, s44, 24 // (op 27717) 
s_ashr_i32 s18, s0, 31 // (op 27717) 

s_xor_b32 s0, s19, s18 // (op 27717) 

s_sub_i32 s19, s0, s18 // (op 27717) 


s_cmp_eq_i32 s1, 0 // (op 27717) 
s_mov_b32 s0, scc // (op 27717) 

s_cmp_eq_i32 s0, 0 // (op 27717) 
s_cselect_b32 s1, s19, s47 // (op 27717) 



v_lshl_add_u32 v11, s1, 8, v20 // (op 27717) 


v_mul_lo_u32 v20, v11, s38 // (op 27717) 

v_lshl_add_u32 v11, v21, 2, v20 // (op 27717) 


s_mul_i32 s1, s38, s7 // (op 27719) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 27719) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 27719) least significant
s_mul_hi_u32 s18, s38, s6 // (op 27719) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27719) most significant: sum
s_add_u32 s1, s1, s18 // (op 27719) most significant: sum


s_mov_b32 s68, s40 // (op 27719) reg expression
s_mov_b32 s69, s41 // (op 27719) reg expression
s_mov_b32 s70, s0 // (op 27719) reg expression
s_mov_b32 s71, 131072 // (op 27719) call()


s_mul_i32 s1, s31, 6 // (op 26133) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 26133) least significant
s_mul_hi_u32 s47, s30, 6 // (op 26133) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 26133) most significant: sum

v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 26133) 


v_mov_b32 v22, v12 // (op 26133) 
buffer_load_dwordx4 v[24:27], v22, s[60:63], 0 offen // (op 26133) Load value
buffer_load_dwordx2 v[28:29], v22, s[60:63], 0 offen offset:16 // (op 26133) Load value
buffer_load_dwordx4 v[30:33], v22, s[60:63], 0 offen offset:24 // (op 26133) Load value
buffer_load_dwordx2 v[34:35], v22, s[60:63], 0 offen offset:40 // (op 26133) Load value
v_add_u32 v22, v22, v20 // (op 26133) 
buffer_load_dwordx4 v[36:39], v22, s[60:63], 0 offen // (op 26133) Load value
buffer_load_dwordx2 v[40:41], v22, s[60:63], 0 offen offset:16 // (op 26133) Load value
buffer_load_dwordx4 v[42:45], v22, s[60:63], 0 offen offset:24 // (op 26133) Load value
buffer_load_dwordx2 v[46:47], v22, s[60:63], 0 offen offset:40 // (op 26133) Load value


v_and_b32 v20, -8, v1 // (op 27451) 
v_sub_u32 v21, v1, v20 // (op 27451) 

v_lshlrev_b32 v20, 4, v21 // (op 27451) 

v_lshrrev_b32 v21, 3, v1 // (op 27451) 
v_lshl_add_u32 v22, v21, 7, v20 // (op 27451) 


s_nop 0
v_readfirstlane_b32 s47, v22 // (op 27451) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 27190) 
v_mov_b64 v[20:21], s[0:1] // (op 27190) convert


v_mov_b64 v[22:23], v[14:15] // (op 27190) 
s_mov_b32 m0, s47 // (op 27190) reg expression
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 27190) Load value direct to lds
v_add_co_u32 v22, s[0:1], v22, v20 // (op 27190) least significant half
v_addc_co_u32 v23, s[0:1], 0, v23, s[0:1] // (op 27190) most significant half

s_add_u32 m0, m0, 4096 // (op 27190) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 27190) Load value direct to lds
v_add_co_u32 v22, s[0:1], v22, v20 // (op 27190) least significant half
v_addc_co_u32 v23, s[0:1], 0, v23, s[0:1] // (op 27190) most significant half

s_add_u32 m0, m0, 4096 // (op 27190) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 27190) Load value direct to lds
v_add_co_u32 v22, s[0:1], v22, v20 // (op 27190) least significant half
v_addc_co_u32 v23, s[0:1], 0, v23, s[0:1] // (op 27190) most significant half

s_add_u32 m0, m0, 4096 // (op 27190) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 27190) Load value direct to lds
v_add_co_u32 v22, s[0:1], v22, v20 // (op 27190) least significant half
v_addc_co_u32 v23, s[0:1], 0, v23, s[0:1] // (op 27190) most significant half

s_add_u32 m0, m0, 4096 // (op 27190) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 27190) Load value direct to lds
v_add_co_u32 v22, s[0:1], v22, v20 // (op 27190) least significant half
v_addc_co_u32 v23, s[0:1], 0, v23, s[0:1] // (op 27190) most significant half

s_add_u32 m0, m0, 4096 // (op 27190) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 27190) Load value direct to lds
v_add_co_u32 v22, s[0:1], v22, v20 // (op 27190) least significant half
v_addc_co_u32 v23, s[0:1], 0, v23, s[0:1] // (op 27190) most significant half

s_add_u32 m0, m0, 4096 // (op 27190) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 27190) Load value direct to lds
v_add_co_u32 v22, s[0:1], v22, v20 // (op 27190) least significant half
v_addc_co_u32 v23, s[0:1], 0, v23, s[0:1] // (op 27190) most significant half

s_add_u32 m0, m0, 4096 // (op 27190) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 27190) Load value direct to lds





v_mov_b32 v20, v10 // (op 76) reg expression

buffer_load_dwordx2 v[22:23], v20, s[68:71], 0 offen // (op 76) Load value



s_lshl_b32 s47, s38, 6 // (op 726) 
v_add_u32 v20, v10, s47 // (op 726) 


buffer_load_dwordx2 v[48:49], v20, s[68:71], 0 offen // (op 726) Load value




s_lshl_b32 s47, s34, 6 // (op 5311) 
v_add_u32 v10, v18, s47 // (op 5311) 


buffer_load_dwordx2 v[20:21], v10, s[64:67], 0 offen // (op 5311) Load value



v_mov_b32 v10, v18 // (op 72) reg expression

buffer_load_dwordx2 v[50:51], v10, s[64:67], 0 offen // (op 72) Load value



s_mul_i32 s1, s31, 6 // (op 26139) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 26139) least significant
s_mul_hi_u32 s47, s30, 6 // (op 26139) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 26139) most significant: sum

v_lshrrev_b64 v[52:53], 3, s[0:1] // (op 26139) 


v_mov_b32 v10, v12 // (op 26139) 
buffer_load_dwordx4 v[54:57], v10, s[60:63], 0 offen offset:96 // (op 26139) Load value
buffer_load_dwordx2 v[58:59], v10, s[60:63], 0 offen offset:112 // (op 26139) Load value
buffer_load_dwordx4 v[60:63], v10, s[60:63], 0 offen offset:120 // (op 26139) Load value
buffer_load_dwordx2 v[64:65], v10, s[60:63], 0 offen offset:136 // (op 26139) Load value
v_add_u32 v10, v10, v52 // (op 26139) 
buffer_load_dwordx4 v[66:69], v10, s[60:63], 0 offen offset:96 // (op 26139) Load value
buffer_load_dwordx2 v[70:71], v10, s[60:63], 0 offen offset:112 // (op 26139) Load value
buffer_load_dwordx4 v[72:75], v10, s[60:63], 0 offen offset:120 // (op 26139) Load value
buffer_load_dwordx2 v[76:77], v10, s[60:63], 0 offen offset:136 // (op 26139) Load value



v_and_b32 v10, -8, v1 // (op 27576) 
v_sub_u32 v12, v1, v10 // (op 27576) 

v_lshlrev_b32 v10, 4, v12 // (op 27576) 

v_lshrrev_b32 v12, 3, v1 // (op 27576) 
v_lshl_add_u32 v18, v12, 7, v10 // (op 27576) 


s_nop 0
v_readfirstlane_b32 s47, v18 // (op 27576) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 27195) 
v_mov_b64 v[52:53], s[0:1] // (op 27195) convert


v_mov_b64 v[78:79], v[14:15] // (op 27195) 
s_add_u32 s0, 32768, s47 // (op 27195) 
s_mov_b32 m0, s0 // (op 27195) reg expression
s_mov_b32 s1, 128 // (op 27195) call()
buffer_load_dwordx4 v78, s[48:51], s1 offen offset: 0 lds // (op 27195) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v52 // (op 27195) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 27195) most significant half

s_add_u32 m0, m0, 4096 // (op 27195) 
buffer_load_dwordx4 v78, s[48:51], s1 offen offset: 0 lds // (op 27195) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v52 // (op 27195) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 27195) most significant half

s_add_u32 m0, m0, 4096 // (op 27195) 
buffer_load_dwordx4 v78, s[48:51], s1 offen offset: 0 lds // (op 27195) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v52 // (op 27195) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 27195) most significant half

s_add_u32 m0, m0, 4096 // (op 27195) 
buffer_load_dwordx4 v78, s[48:51], s1 offen offset: 0 lds // (op 27195) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v52 // (op 27195) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 27195) most significant half

s_add_u32 m0, m0, 4096 // (op 27195) 
buffer_load_dwordx4 v78, s[48:51], s1 offen offset: 0 lds // (op 27195) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v52 // (op 27195) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 27195) most significant half

s_add_u32 m0, m0, 4096 // (op 27195) 
buffer_load_dwordx4 v78, s[48:51], s1 offen offset: 0 lds // (op 27195) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v52 // (op 27195) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 27195) most significant half

s_add_u32 m0, m0, 4096 // (op 27195) 
buffer_load_dwordx4 v78, s[48:51], s1 offen offset: 0 lds // (op 27195) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v52 // (op 27195) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 27195) most significant half

s_add_u32 m0, m0, 4096 // (op 27195) 
buffer_load_dwordx4 v78, s[48:51], s1 offen offset: 0 lds // (op 27195) Load value direct to lds






v_and_b32 v10, -2, v1 // (op 27509) 
v_sub_u32 v12, v1, v10 // (op 27509) 

v_lshlrev_b32 v10, 6, v12 // (op 27509) 

v_lshrrev_b32 v12, 1, v1 // (op 27509) 
v_lshl_add_u32 v18, v12, 8, v10 // (op 27509) 


v_mul_lo_u32 v10, v18, 6 // (op 27509) 

v_lshrrev_b32 v12, 3, v10 // (op 27509) 




v_add_u32 v10, 65536, v12 // (op 26144) 
s_waitcnt vmcnt(35)
ds_write_b128 v10, v[24:27] // (op 26144) Store local data 
s_waitcnt vmcnt(34)
ds_write_b64 v10, v[28:29] offset:16 // (op 26144) Store local data 

v_add_u32 v10, 65560, v12 // (op 26144) 
s_waitcnt vmcnt(33)
ds_write_b128 v10, v[30:33] // (op 26144) Store local data 
s_waitcnt vmcnt(32)
ds_write_b64 v10, v[34:35] offset:16 // (op 26144) Store local data 

v_add_u32 v10, 65632, v12 // (op 26144) 
s_waitcnt vmcnt(31)
ds_write_b128 v10, v[36:39] // (op 26144) Store local data 
s_waitcnt vmcnt(30)
ds_write_b64 v10, v[40:41] offset:16 // (op 26144) Store local data 

v_add_u32 v10, 65656, v12 // (op 26144) 
s_waitcnt vmcnt(29)
ds_write_b128 v10, v[42:45] // (op 26144) Store local data 
s_waitcnt vmcnt(28)
ds_write_b64 v10, v[46:47] offset:16 // (op 26144) Store local data 


v_and_b32 v10, -8, v1 // (op 27436) 
v_sub_u32 v12, v1, v10 // (op 27436) 

v_lshlrev_b32 v10, 4, v12 // (op 27436) 

v_lshrrev_b32 v12, 3, v1 // (op 27436) 
v_lshl_add_u32 v18, v12, 7, v10 // (op 27436) 


s_nop 0
v_readfirstlane_b32 s47, v18 // (op 27436) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -2, v1 // (op 27493) 
v_sub_u32 v12, v1, v10 // (op 27493) 

v_lshlrev_b32 v10, 6, v12 // (op 27493) 

v_lshrrev_b32 v12, 1, v1 // (op 27493) 
v_lshl_add_u32 v18, v12, 8, v10 // (op 27493) 


v_mul_lo_u32 v10, v18, 6 // (op 27493) 

v_lshrrev_b32 v12, 3, v10 // (op 27493) 

v_and_b32 v10, -8, v1 // (op 27561) 
v_sub_u32 v18, v1, v10 // (op 27561) 

v_lshlrev_b32 v10, 4, v18 // (op 27561) 

v_lshrrev_b32 v18, 3, v1 // (op 27561) 
v_lshl_add_u32 v14, v18, 7, v10 // (op 27561) 


s_nop 0
v_readfirstlane_b32 s0, v14 // (op 27561) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -2, v1 // (op 27605) 
v_sub_u32 v18, v1, v10 // (op 27605) 

v_lshlrev_b32 v10, 6, v18 // (op 27605) 

v_lshrrev_b32 v18, 1, v1 // (op 27605) 
v_lshl_add_u32 v14, v18, 8, v10 // (op 27605) 


v_mul_lo_u32 v10, v14, 6 // (op 27605) 

v_lshrrev_b32 v18, 3, v10 // (op 27605) 

s_mov_b32 s1, 0 // (op 8908) call()
s_ashr_i64 s[18:19], s[42:43], 7 // (op 8907) 
s_mov_b32 s22, s18 // (op 8907) convert

s_and_b32 s18, s22, 4294967294 // (op 8907) 

s_cmp_lt_u32 s1, s18 // (op 8907) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoop_8907 // (op 8907) Condition: Top (jump to RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoop_8907 if false)
RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoop_8907:
s_add_u32 s1, s1, 2 // (op 8909) 
s_waitcnt vmcnt(20)
s_barrier  // (op 28021) 



ds_read_b128 v[78:81], v3 // (op 62) Load local data 
ds_read_b128 v[82:85], v3 offset:64 // (op 62) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 28037) 



v_add_u32 v10, 65536, v4 // (op 67) 
ds_read_b128 v[86:89], v10 // (op 67) Load local data 
ds_read_b64 v[90:91], v10 offset:16 // (op 67) Load local data 




v_add_u32 v10, 67072, v4 // (op 312) 
s_waitcnt vmcnt(19)
v_mov_b32 v14, v22 // (op 27090) reg expression
v_mov_b32 v15, v23 // (op 27090) reg expression
s_waitcnt vmcnt(16)
v_mov_b32 v52, v50 // (op 27131) reg expression
v_mov_b32 v53, v51 // (op 27131) reg expression
ds_read_b128 v[92:95], v10 // (op 312) Load local data 
v_bfe_u32 v98, v52, 0, 8 // (op 24807) 
v_bfe_u32 v99, v52, 8, 8 // (op 24807) 
v_bfe_u32 v100, v52, 16, 8 // (op 24807) 
v_bfe_u32 v101, v52, 24, 8 // (op 24807) 
v_permlane16_swap_b32 v98, v99 // (op 24807) 
ds_read_b64 v[96:97], v10 offset:16 // (op 312) Load local data 

v_permlane16_swap_b32 v100, v101 // (op 24807) 



v_add_u32 v10, 68608, v4 // (op 448) 
s_nop 0
v_permlane32_swap_b32 v98, v100 // (op 24807) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v99, v101 // (op 24807) 
ds_read_b128 v[102:105], v10 // (op 448) Load local data 
ds_read_b64 v[106:107], v10 offset:16 // (op 448) Load local data 




v_add_u32 v10, 70144, v4 // (op 584) 
v_bfe_u32 v108, v14, 0, 8 // (op 24998) 
v_bfe_u32 v109, v14, 8, 8 // (op 24998) 
v_bfe_u32 v110, v14, 16, 8 // (op 24998) 
v_bfe_u32 v111, v14, 24, 8 // (op 24998) 
v_permlane16_swap_b32 v108, v109 // (op 24998) 
ds_read_b128 v[112:115], v10 // (op 584) Load local data 
v_permlane16_swap_b32 v110, v111 // (op 24998) 
s_nop 1
v_permlane32_swap_b32 v108, v110 // (op 24998) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v109, v111 // (op 24998) 
ds_read_b64 v[116:117], v10 offset:16 // (op 584) Load local data 




s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:85], v[86:91], a[0:3], v98, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 78) 
v_add_u32 v10, 71680, v4 // (op 720) 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:85], v[92:97], a[4:7], v98, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 315) 
ds_read_b128 v[118:121], v10 // (op 720) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:85], v[102:107], a[8:11], v98, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 451) 
ds_read_b64 v[122:123], v10 offset:16 // (op 720) Load local data 




v_add_u32 v10, 73216, v4 // (op 856) 
v_mov_b32 v124, v48 // (op 27103) reg expression
v_mov_b32 v125, v49 // (op 27103) reg expression
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[78:85], v[112:117], a[12:15], v98, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 587) 
v_bfe_u32 v126, v124, 0, 8 // (op 25030) 
v_bfe_u32 v127, v124, 8, 8 // (op 25030) 
v_bfe_u32 v128, v124, 16, 8 // (op 25030) 
v_bfe_u32 v129, v124, 24, 8 // (op 25030) 
v_permlane16_swap_b32 v126, v127 // (op 25030) 
ds_read_b128 v[130:133], v10 // (op 856) Load local data 
v_permlane16_swap_b32 v128, v129 // (op 25030) 
s_nop 1
v_permlane32_swap_b32 v126, v128 // (op 25030) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v127, v129 // (op 25030) 
ds_read_b64 v[134:135], v10 offset:16 // (op 856) Load local data 




v_add_u32 v10, 74752, v4 // (op 992) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[78:85], v[118:123], a[16:19], v98, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 723) 
ds_read_b128 v[136:139], v10 // (op 992) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:85], v[130:135], a[20:23], v98, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 859) 
ds_read_b64 v[140:141], v10 offset:16 // (op 992) Load local data 




v_add_u32 v10, 76288, v4 // (op 1128) 
ds_read_b128 v[142:145], v10 // (op 1128) Load local data 
ds_read_b64 v[146:147], v10 offset:16 // (op 1128) Load local data 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[78:85], v[136:141], a[24:27], v98, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 995) 



ds_read_b128 v[148:151], v3 offset:2048 // (op 1629) Load local data 
ds_read_b128 v[152:155], v3 offset:2112 // (op 1629) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:85], v[142:147], a[28:31], v98, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1131) 




ds_read_b128 v[78:81], v3 offset:4096 // (op 2847) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[148:155], v[86:91], a[32:35], v99, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1899) 
ds_read_b128 v[82:85], v3 offset:4160 // (op 2847) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[148:155], v[92:97], a[36:39], v99, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1861) 
ds_read_b128 v[156:159], v3 offset:6144 // (op 4065) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:155], v[102:107], a[40:43], v99, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1823) 
ds_read_b128 v[160:163], v3 offset:6208 // (op 4065) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[148:155], v[112:117], a[44:47], v99, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1785) 
ds_read_b128 v[164:167], v3 offset:8192 // (op 5283) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[148:155], v[118:123], a[48:51], v99, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1747) 
ds_read_b128 v[168:171], v3 offset:8256 // (op 5283) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[148:155], v[130:135], a[52:55], v99, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1709) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:155], v[136:141], a[56:59], v99, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1671) 
s_mul_i32 s19, s31, 6 // (op 10) most significant: low of high * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[148:155], v[142:147], a[60:63], v99, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1633) 
s_mul_i32 s18, s30, 6 // (op 10) least significant

s_mul_hi_u32 s22, s30, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 10) most significant: sum

v_lshrrev_b64 v[148:149], 3, s[18:19] // (op 10) 


v_mov_b32 v10, v13 // (op 10) 
v_mov_b32 v150, v10 // (op 10) 
buffer_load_dwordx4 v[24:27], v150, s[60:63], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[28:29], v150, s[60:63], 0 offen offset:208 // (op 10) Load value
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[78:85], v[86:91], a[64:67], v100, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 3117) 
buffer_load_dwordx4 v[30:33], v150, s[60:63], 0 offen offset:216 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[78:85], v[92:97], a[68:71], v100, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 3079) 
v_mov_b32 v154, v20 // (op 27117) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[78:85], v[102:107], a[72:75], v100, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 3041) 
v_mov_b32 v155, v21 // (op 27117) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[78:85], v[112:117], a[76:79], v100, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 3003) 
v_bfe_u32 v172, v154, 0, 8 // (op 24838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[78:85], v[118:123], a[80:83], v100, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 2965) 
v_bfe_u32 v173, v154, 8, 8 // (op 24838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:85], v[130:135], a[84:87], v100, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 2927) 
v_bfe_u32 v174, v154, 16, 8 // (op 24838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[78:85], v[136:141], a[88:91], v100, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 2889) 
v_bfe_u32 v175, v154, 24, 8 // (op 24838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[78:85], v[142:147], a[92:95], v100, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 2851) 
v_permlane16_swap_b32 v172, v173 // (op 24838) 

v_permlane16_swap_b32 v174, v175 // (op 24838) 
ds_read_b128 v[78:81], v3 offset:10240 // (op 6501) Load local data 
ds_read_b128 v[82:85], v3 offset:10304 // (op 6501) Load local data 
v_permlane32_swap_b32 v172, v174 // (op 24838) 



v_permlane32_swap_b32 v173, v175 // (op 24838) 
ds_read_b128 v[176:179], v3 offset:12288 // (op 7719) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[156:163], v[86:91], a[96:99], v101, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 4335) 
ds_read_b128 v[180:183], v3 offset:12352 // (op 7719) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[156:163], v[92:97], a[100:103], v101, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 4297) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 28030) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[156:163], v[102:107], a[104:107], v101, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 4259) 
ds_read_b128 v[184:187], v3 offset:14336 // (op 8937) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[156:163], v[112:117], a[108:111], v101, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 4221) 
ds_read_b128 v[188:191], v3 offset:14400 // (op 8937) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[156:163], v[118:123], a[112:115], v101, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 4183) 
buffer_load_dwordx2 v[34:35], v150, s[60:63], 0 offen offset:232 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[156:163], v[130:135], a[116:119], v101, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 4145) 
v_add_u32 v150, v150, v148 // (op 10) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[156:163], v[136:141], a[120:123], v101, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 4107) 
buffer_load_dwordx4 v[36:39], v150, s[60:63], 0 offen offset:192 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[156:163], v[142:147], a[124:127], v101, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 4069) 
buffer_load_dwordx2 v[40:41], v150, s[60:63], 0 offen offset:208 // (op 10) Load value


buffer_load_dwordx4 v[42:45], v150, s[60:63], 0 offen offset:216 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[164:171], v[92:97], a[132:135], v172, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5515) 
buffer_load_dwordx2 v[46:47], v150, s[60:63], 0 offen offset:232 // (op 10) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[164:171], v[102:107], a[136:139], v172, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5477) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 28024) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[164:171], v[112:117], a[140:143], v172, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5439) 
s_lshl_b64 s[18:19], s[26:27], 5 // (op 27174) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[164:171], v[130:135], a[148:151], v172, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5363) 
v_mov_b64 v[98:99], s[18:19] // (op 27174) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[164:171], v[136:141], a[152:155], v172, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5325) 
v_mov_b64 v[100:101], v[16:17] // (op 27174) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[164:171], v[142:147], a[156:159], v172, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5287) 
v_mov_b64 v[148:149], v[100:101] // (op 27174) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[164:171], v[86:91], a[128:131], v172, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5553) 
s_mov_b32 m0, s47 // (op 27174) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[164:171], v[118:123], a[144:147], v172, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5401) 
s_mov_b32 s18, 256 // (op 27174) call()

buffer_load_dwordx4 v148, s[48:51], s18 offen offset: 0 lds // (op 27174) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[78:85], v[92:97], a[164:167], v173, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6733) 
v_add_co_u32 v148, s[22:23], v148, v98 // (op 27174) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:85], v[102:107], a[168:171], v173, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6695) 
v_addc_co_u32 v149, s[22:23], 0, v149, s[22:23] // (op 27174) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[78:85], v[112:117], a[172:175], v173, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6657) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[78:85], v[130:135], a[180:183], v173, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6581) 
s_add_u32 m0, m0, 4096 // (op 27174) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[78:85], v[136:141], a[184:187], v173, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6543) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[78:85], v[142:147], a[188:191], v173, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6505) 
buffer_load_dwordx4 v148, s[48:51], s18 offen offset: 0 lds // (op 27174) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[78:85], v[86:91], a[160:163], v173, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6771) 
v_add_co_u32 v148, s[22:23], v148, v98 // (op 27174) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[78:85], v[118:123], a[176:179], v173, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6619) 
v_addc_co_u32 v149, s[22:23], 0, v149, s[22:23] // (op 27174) most significant half


v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[176:183], v[92:97], a[196:199], v174, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7951) 
s_add_u32 m0, m0, 4096 // (op 27174) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[176:183], v[102:107], a[200:203], v174, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7913) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[176:183], v[112:117], a[204:207], v174, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7875) 
buffer_load_dwordx4 v148, s[48:51], s18 offen offset: 0 lds // (op 27174) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[176:183], v[130:135], a[212:215], v174, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7799) 
v_add_co_u32 v148, s[22:23], v148, v98 // (op 27174) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[176:183], v[136:141], a[216:219], v174, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7761) 
v_addc_co_u32 v149, s[22:23], 0, v149, s[22:23] // (op 27174) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[176:183], v[142:147], a[220:223], v174, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7723) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[176:183], v[86:91], a[192:195], v174, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7989) 
s_add_u32 m0, m0, 4096 // (op 27174) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[176:183], v[118:123], a[208:211], v174, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7837) 

buffer_load_dwordx4 v148, s[48:51], s18 offen offset: 0 lds // (op 27174) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[184:191], v[92:97], a[228:231], v175, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 9169) 
v_add_co_u32 v148, s[22:23], v148, v98 // (op 27174) least significant half

v_addc_co_u32 v149, s[22:23], 0, v149, s[22:23] // (op 27174) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[184:191], v[102:107], a[232:235], v175, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 9131) 

s_add_u32 m0, m0, 4096 // (op 27174) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[184:191], v[112:117], a[236:239], v175, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 9093) 

buffer_load_dwordx4 v148, s[48:51], s18 offen offset: 0 lds // (op 27174) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[184:191], v[130:135], a[244:247], v175, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 9017) 
v_add_co_u32 v148, s[22:23], v148, v98 // (op 27174) least significant half

v_addc_co_u32 v149, s[22:23], 0, v149, s[22:23] // (op 27174) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[184:191], v[136:141], a[248:251], v175, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 8979) 

s_add_u32 m0, m0, 4096 // (op 27174) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[184:191], v[142:147], a[252:255], v175, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 8941) 

buffer_load_dwordx4 v148, s[48:51], s18 offen offset: 0 lds // (op 27174) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[184:191], v[86:91], a[224:227], v175, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 9207) 
v_add_co_u32 v148, s[22:23], v148, v98 // (op 27174) least significant half


v_addc_co_u32 v149, s[22:23], 0, v149, s[22:23] // (op 27174) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[184:191], v[118:123], a[240:243], v175, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 9055) 




s_add_u32 m0, m0, 4096 // (op 27174) 
buffer_load_dwordx4 v148, s[48:51], s18 offen offset: 0 lds // (op 27174) Load value direct to lds
v_add_co_u32 v148, s[22:23], v148, v98 // (op 27174) least significant half
v_addc_co_u32 v149, s[22:23], 0, v149, s[22:23] // (op 27174) most significant half

s_add_u32 m0, m0, 4096 // (op 27174) 
buffer_load_dwordx4 v148, s[48:51], s18 offen offset: 0 lds // (op 27174) Load value direct to lds





s_barrier  // (op 28063) 
v_mov_b32 v78, v11 // (op 27098) 



v_add_u32 v79, 8, v78 // (op 27098) 

v_add_u32 v80, 90112, v18 // (op 222) 
s_waitcnt vmcnt(31)
ds_write_b128 v80, v[54:57] // (op 222) Store local data 
s_waitcnt vmcnt(30)
ds_write_b64 v80, v[58:59] offset:16 // (op 222) Store local data 

v_add_u32 v80, 90136, v18 // (op 222) 
s_waitcnt vmcnt(29)
ds_write_b128 v80, v[60:63] // (op 222) Store local data 
buffer_load_dwordx2 v[22:23], v79, s[68:71], 0 offen // (op 27098) Load value

s_waitcnt vmcnt(29)
ds_write_b64 v80, v[64:65] offset:16 // (op 222) Store local data 



s_lshl_b32 s18, s38, 6 // (op 27112) 
s_add_u32 s18, s18, 8 // (op 27112) 
v_add_u32 v79, 90208, v18 // (op 222) 
v_add_u32 v80, v78, s18 // (op 27112) 


buffer_load_dwordx2 v[48:49], v80, s[68:71], 0 offen // (op 27112) Load value

s_waitcnt vmcnt(29)
ds_write_b128 v79, v[66:69] // (op 222) Store local data 

s_waitcnt vmcnt(28)
ds_write_b64 v79, v[70:71] offset:16 // (op 222) Store local data 



v_add_u32 v11, 8, v11 // (op 27707) 
v_add_u32 v78, 90232, v18 // (op 222) 
v_mov_b32 v79, v19 // (op 27126) 
s_lshl_b32 s18, s34, 6 // (op 27126) 
s_add_u32 s19, 8, s18 // (op 27126) 

v_add_u32 v80, v79, s19 // (op 27126) 


s_waitcnt vmcnt(27)
ds_write_b128 v78, v[72:75] // (op 222) Store local data 
s_waitcnt vmcnt(26)
ds_write_b64 v78, v[76:77] offset:16 // (op 222) Store local data 

buffer_load_dwordx2 v[20:21], v80, s[64:67], 0 offen // (op 27126) Load value



v_add_u32 v78, 8, v79 // (op 27139) 

buffer_load_dwordx2 v[50:51], v78, s[64:67], 0 offen // (op 27139) Load value


s_waitcnt vmcnt(20)
s_barrier  // (op 28051) 




v_add_u32 v19, 8, v19 // (op 27655) 
s_mul_i32 s19, s31, 6 // (op 223) most significant: low of high * low
s_mul_i32 s18, s30, 6 // (op 223) least significant
s_mul_hi_u32 s22, s30, 6 // (op 223) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 223) most significant: sum

v_lshrrev_b64 v[78:79], 3, s[18:19] // (op 223) 


v_mov_b32 v80, v10 // (op 223) 
ds_read_b128 v[92:95], v5 offset:32768 // (op 214) Load local data 
ds_read_b128 v[96:99], v5 offset:32832 // (op 214) Load local data 
buffer_load_dwordx4 v[54:57], v80, s[60:63], 0 offen offset:288 // (op 223) Load value
s_waitcnt lgkmcnt(2)
s_barrier  // (op 28060) 
buffer_load_dwordx2 v[58:59], v80, s[60:63], 0 offen offset:304 // (op 223) Load value



v_add_u32 v81, 90112, v6 // (op 216) 
ds_read_b128 v[82:85], v81 // (op 216) Load local data 
ds_read_b64 v[86:87], v81 offset:16 // (op 216) Load local data 

buffer_load_dwordx4 v[60:63], v80, s[60:63], 0 offen offset:312 // (op 223) Load value



v_add_u32 v81, 91648, v6 // (op 324) 
v_bfe_u32 v88, v53, 0, 8 // (op 24822) 
v_bfe_u32 v89, v53, 8, 8 // (op 24822) 
v_bfe_u32 v90, v53, 16, 8 // (op 24822) 
v_bfe_u32 v91, v53, 24, 8 // (op 24822) 
v_permlane16_swap_b32 v88, v89 // (op 24822) 
ds_read_b128 v[102:105], v81 // (op 324) Load local data 
v_permlane16_swap_b32 v90, v91 // (op 24822) 
s_nop 1
v_permlane32_swap_b32 v88, v90 // (op 24822) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v89, v91 // (op 24822) 
ds_read_b64 v[106:107], v81 offset:16 // (op 324) Load local data 





v_add_u32 v81, 93184, v6 // (op 460) 
v_bfe_u32 v108, v15, 0, 8 // (op 25013) 
v_bfe_u32 v109, v15, 8, 8 // (op 25013) 
v_bfe_u32 v110, v15, 16, 8 // (op 25013) 
v_bfe_u32 v111, v15, 24, 8 // (op 25013) 
v_permlane16_swap_b32 v108, v109 // (op 25013) 
ds_read_b128 v[112:115], v81 // (op 460) Load local data 
v_permlane16_swap_b32 v110, v111 // (op 25013) 
buffer_load_dwordx2 v[64:65], v80, s[60:63], 0 offen offset:328 // (op 223) Load value
v_add_u32 v80, v80, v78 // (op 223) 
v_permlane32_swap_b32 v108, v110 // (op 25013) 
v_permlane32_swap_b32 v109, v111 // (op 25013) 
ds_read_b64 v[116:117], v81 offset:16 // (op 460) Load local data 





s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[92:99], v[82:87], a[0:3], v88, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 221) 
v_add_u32 v81, 94720, v6 // (op 596) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[92:99], v[102:107], a[4:7], v88, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 316) 
ds_read_b128 v[118:121], v81 // (op 596) Load local data 
ds_read_b64 v[122:123], v81 offset:16 // (op 596) Load local data 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[92:99], v[112:117], a[8:11], v88, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 452) 



v_add_u32 v81, 96256, v6 // (op 732) 
ds_read_b128 v[126:129], v81 // (op 732) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:99], v[118:123], a[12:15], v88, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 588) 
ds_read_b64 v[130:131], v81 offset:16 // (op 732) Load local data 

buffer_load_dwordx4 v[66:69], v80, s[60:63], 0 offen offset:288 // (op 223) Load value



v_add_u32 v81, 97792, v6 // (op 868) 
v_bfe_u32 v132, v125, 0, 8 // (op 25045) 
v_bfe_u32 v133, v125, 8, 8 // (op 25045) 
v_bfe_u32 v134, v125, 16, 8 // (op 25045) 
v_bfe_u32 v135, v125, 24, 8 // (op 25045) 
v_permlane16_swap_b32 v132, v133 // (op 25045) 
ds_read_b128 v[136:139], v81 // (op 868) Load local data 
ds_read_b64 v[140:141], v81 offset:16 // (op 868) Load local data 

v_permlane16_swap_b32 v134, v135 // (op 25045) 



v_add_u32 v81, 99328, v6 // (op 1004) 
ds_read_b128 v[142:145], v81 // (op 1004) Load local data 
v_permlane32_swap_b32 v132, v134 // (op 25045) 
v_permlane32_swap_b32 v133, v135 // (op 25045) 
buffer_load_dwordx2 v[70:71], v80, s[60:63], 0 offen offset:304 // (op 223) Load value

ds_read_b64 v[146:147], v81 offset:16 // (op 1004) Load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[92:99], v[126:131], a[16:19], v88, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 724) 
v_add_u32 v81, 100864, v6 // (op 1140) 
ds_read_b128 v[148:151], v81 // (op 1140) Load local data 
ds_read_b64 v[152:153], v81 offset:16 // (op 1140) Load local data 

buffer_load_dwordx4 v[72:75], v80, s[60:63], 0 offen offset:312 // (op 223) Load value
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[92:99], v[136:141], a[20:23], v88, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 860) 
s_barrier  // (op 28057) 
s_barrier  // (op 28102) 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:99], v[142:147], a[24:27], v88, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 996) 
ds_read_b128 v[156:159], v5 offset:34816 // (op 1641) Load local data 
ds_read_b128 v[160:163], v5 offset:34880 // (op 1641) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:99], v[148:153], a[28:31], v88, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1132) 




ds_read_b128 v[92:95], v5 offset:36864 // (op 2859) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[156:163], v[82:87], a[32:35], v89, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1900) 
ds_read_b128 v[96:99], v5 offset:36928 // (op 2859) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[156:163], v[112:117], a[40:43], v89, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1824) 
ds_read_b128 v[164:167], v5 offset:38912 // (op 4077) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[156:163], v[118:123], a[44:47], v89, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1786) 
ds_read_b128 v[168:171], v5 offset:38976 // (op 4077) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[156:163], v[102:107], a[36:39], v89, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1862) 
ds_read_b128 v[172:175], v5 offset:40960 // (op 5295) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[156:163], v[126:131], a[48:51], v89, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1748) 
ds_read_b128 v[176:179], v5 offset:41024 // (op 5295) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[156:163], v[142:147], a[56:59], v89, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1672) 
v_bfe_u32 v180, v155, 0, 8 // (op 24853) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[156:163], v[148:153], a[60:63], v89, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1634) 
v_bfe_u32 v181, v155, 8, 8 // (op 24853) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:163], v[136:141], a[52:55], v89, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1710) 
v_bfe_u32 v182, v155, 16, 8 // (op 24853) 

v_bfe_u32 v183, v155, 24, 8 // (op 24853) 
v_permlane16_swap_b32 v180, v181 // (op 24853) 
ds_read_b128 v[156:159], v5 offset:43008 // (op 6513) Load local data 
v_permlane16_swap_b32 v182, v183 // (op 24853) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[92:99], v[82:87], a[64:67], v90, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 3118) 
s_nop 0
v_permlane32_swap_b32 v180, v182 // (op 24853) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[92:99], v[112:117], a[72:75], v90, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 3042) 
v_permlane32_swap_b32 v181, v183 // (op 24853) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[92:99], v[118:123], a[76:79], v90, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 3004) 
ds_read_b128 v[160:163], v5 offset:43072 // (op 6513) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[92:99], v[102:107], a[68:71], v90, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 3080) 
ds_read_b128 v[184:187], v5 offset:45056 // (op 7731) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[92:99], v[126:131], a[80:83], v90, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 2966) 
ds_read_b128 v[188:191], v5 offset:45120 // (op 7731) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[92:99], v[142:147], a[88:91], v90, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 2890) 
ds_read_b128 v[192:195], v5 offset:47104 // (op 8949) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[92:99], v[148:153], a[92:95], v90, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 2852) 
ds_read_b128 v[196:199], v5 offset:47168 // (op 8949) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[92:99], v[136:141], a[84:87], v90, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 2928) 
buffer_load_dwordx2 v[76:77], v80, s[60:63], 0 offen offset:328 // (op 223) Load value




s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[164:171], v[82:87], a[96:99], v91, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 4336) 
v_add_u32 v13, 192, v13 // (op 27372) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[164:171], v[112:117], a[104:107], v91, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 4260) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 28054) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[164:171], v[118:123], a[108:111], v91, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 4222) 
s_lshl_b64 s[18:19], s[26:27], 5 // (op 27185) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[164:171], v[102:107], a[100:103], v91, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 4298) 
v_mov_b64 v[14:15], s[18:19] // (op 27185) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[164:171], v[126:131], a[112:115], v91, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 4184) 
v_mov_b64 v[52:53], v[100:101] // (op 27185) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[164:171], v[142:147], a[120:123], v91, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 4108) 
s_add_u32 s18, 32768, s0 // (op 27185) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[164:171], v[148:153], a[124:127], v91, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 4070) 
s_mov_b32 m0, s18 // (op 27185) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[164:171], v[136:141], a[116:119], v91, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 4146) 
s_mov_b32 s19, 384 // (op 27185) call()


buffer_load_dwordx4 v52, s[48:51], s19 offen offset: 0 lds // (op 27185) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[172:179], v[102:107], a[132:135], v180, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5516) 
v_add_co_u32 v52, s[22:23], v52, v14 // (op 27185) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[172:179], v[136:141], a[148:151], v180, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5364) 
v_addc_co_u32 v53, s[22:23], 0, v53, s[22:23] // (op 27185) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[172:179], v[82:87], a[128:131], v180, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5554) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[172:179], v[112:117], a[136:139], v180, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5478) 
s_add_u32 m0, m0, 4096 // (op 27185) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[172:179], v[118:123], a[140:143], v180, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5440) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[172:179], v[126:131], a[144:147], v180, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5402) 
buffer_load_dwordx4 v52, s[48:51], s19 offen offset: 0 lds // (op 27185) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[172:179], v[142:147], a[152:155], v180, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5326) 
v_add_co_u32 v52, s[22:23], v52, v14 // (op 27185) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[172:179], v[148:153], a[156:159], v180, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5288) 
v_addc_co_u32 v53, s[22:23], 0, v53, s[22:23] // (op 27185) most significant half


v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[156:163], v[102:107], a[164:167], v181, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6734) 
s_add_u32 m0, m0, 4096 // (op 27185) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[156:163], v[136:141], a[180:183], v181, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6582) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[156:163], v[82:87], a[160:163], v181, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6772) 
buffer_load_dwordx4 v52, s[48:51], s19 offen offset: 0 lds // (op 27185) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[156:163], v[112:117], a[168:171], v181, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6696) 
v_add_co_u32 v52, s[22:23], v52, v14 // (op 27185) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[156:163], v[118:123], a[172:175], v181, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6658) 
v_addc_co_u32 v53, s[22:23], 0, v53, s[22:23] // (op 27185) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[156:163], v[126:131], a[176:179], v181, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6620) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[156:163], v[142:147], a[184:187], v181, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6544) 
s_add_u32 m0, m0, 4096 // (op 27185) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[156:163], v[148:153], a[188:191], v181, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6506) 

buffer_load_dwordx4 v52, s[48:51], s19 offen offset: 0 lds // (op 27185) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[184:191], v[102:107], a[196:199], v182, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7952) 
v_add_co_u32 v52, s[22:23], v52, v14 // (op 27185) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[184:191], v[136:141], a[212:215], v182, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7800) 
v_addc_co_u32 v53, s[22:23], 0, v53, s[22:23] // (op 27185) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[184:191], v[82:87], a[192:195], v182, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7990) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[184:191], v[112:117], a[200:203], v182, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7914) 
s_add_u32 m0, m0, 4096 // (op 27185) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[184:191], v[118:123], a[204:207], v182, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7876) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[184:191], v[126:131], a[208:211], v182, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7838) 
buffer_load_dwordx4 v52, s[48:51], s19 offen offset: 0 lds // (op 27185) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[184:191], v[142:147], a[216:219], v182, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7762) 
v_add_co_u32 v52, s[22:23], v52, v14 // (op 27185) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[184:191], v[148:153], a[220:223], v182, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7724) 
v_addc_co_u32 v53, s[22:23], 0, v53, s[22:23] // (op 27185) most significant half


v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[192:199], v[102:107], a[228:231], v183, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 9170) 
s_add_u32 m0, m0, 4096 // (op 27185) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[192:199], v[136:141], a[244:247], v183, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 9018) 
buffer_load_dwordx4 v52, s[48:51], s19 offen offset: 0 lds // (op 27185) Load value direct to lds

v_add_co_u32 v52, s[22:23], v52, v14 // (op 27185) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[192:199], v[82:87], a[224:227], v183, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 9208) 
v_addc_co_u32 v53, s[22:23], 0, v53, s[22:23] // (op 27185) most significant half


v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[192:199], v[112:117], a[232:235], v183, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 9132) 
s_add_u32 m0, m0, 4096 // (op 27185) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[192:199], v[118:123], a[236:239], v183, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 9094) 
buffer_load_dwordx4 v52, s[48:51], s19 offen offset: 0 lds // (op 27185) Load value direct to lds


v_add_co_u32 v52, s[22:23], v52, v14 // (op 27185) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[192:199], v[126:131], a[240:243], v183, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 9056) 
v_addc_co_u32 v53, s[22:23], 0, v53, s[22:23] // (op 27185) most significant half


v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[192:199], v[142:147], a[248:251], v183, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 8980) 
s_add_u32 m0, m0, 4096 // (op 27185) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[192:199], v[148:153], a[252:255], v183, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 8942) 
buffer_load_dwordx4 v52, s[48:51], s19 offen offset: 0 lds // (op 27185) Load value direct to lds









s_barrier  // (op 28040) 
v_add_co_u32 v16, vcc, 256, v16 // (op 27308) least significant half



v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 27308) most significant half
v_add_u32 v10, 65536, v12 // (op 13) 
s_waitcnt vmcnt(35)
ds_write_b128 v10, v[24:27] // (op 13) Store local data 
s_waitcnt vmcnt(34)
ds_write_b64 v10, v[28:29] offset:16 // (op 13) Store local data 

v_add_u32 v10, 65560, v12 // (op 13) 
s_waitcnt vmcnt(33)
ds_write_b128 v10, v[30:33] // (op 13) Store local data 
s_waitcnt vmcnt(32)
ds_write_b64 v10, v[34:35] offset:16 // (op 13) Store local data 

v_add_u32 v10, 65632, v12 // (op 13) 
s_waitcnt vmcnt(31)
ds_write_b128 v10, v[36:39] // (op 13) Store local data 
s_waitcnt vmcnt(30)
ds_write_b64 v10, v[40:41] offset:16 // (op 13) Store local data 

v_add_u32 v10, 65656, v12 // (op 13) 
s_waitcnt vmcnt(29)
ds_write_b128 v10, v[42:45] // (op 13) Store local data 
s_waitcnt vmcnt(28)
ds_write_b64 v10, v[46:47] offset:16 // (op 13) Store local data 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 8907) 
s_mov_b32 s22, s18 // (op 8907) convert

s_and_b32 s18, s22, 4294967294 // (op 8907) 

s_cmp_lt_u32 s1, s18 // (op 8907) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoop_8907 // (op 8907) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoop_8907 if true)
RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoop_8907:












s_barrier  // (op 28164) 
s_mul_hi_i32 s47, s16, s10 // (op 27208) 
v_lshrrev_b32 v10, 3, v1 // (op 27208) 
s_add_i32 s0, s47, s16 // (op 27208) 

v_and_b32 v11, -8, v1 // (op 27208) 
s_ashr_i32 s1, s0, 31 // (op 27208) 
v_sub_u32 v12, v1, v11 // (op 27208) 

s_cmp_eq_i32 s10, 0 // (op 27208) 
s_mov_b32 s47, scc // (op 27208) 
v_lshlrev_b32 v11, 4, v12 // (op 27208) 

s_cmp_eq_i32 s47, 0 // (op 27208) 
s_cselect_b32 s18, 0, -1 // (op 27208) 

s_and_b32 s19, s13, 31 // (op 27208) 
s_lshl_b32 s47, 1, s19 // (op 27208) 
s_add_i32 s22, s47, s18 // (op 27208) 


s_and_b32 s18, s1, s22 // (op 27208) 


s_add_i32 s1, s0, s18 // (op 27208) 


s_ashr_i32 s0, s1, s19 // (op 27208) 


s_lshl_b32 s1, s13, 24 // (op 27208) 
s_ashr_i32 s47, s1, 31 // (op 27208) 

s_xor_b32 s1, s0, s47 // (op 27208) 

s_sub_i32 s0, s1, s47 // (op 27208) 


s_mul_hi_i32 s1, s0, s14 // (op 27208) 
s_add_i32 s47, s1, s0 // (op 27208) 

s_ashr_i32 s1, s47, 31 // (op 27208) 
s_cmp_eq_i32 s14, 0 // (op 27208) 
s_mov_b32 s18, scc // (op 27208) 
s_cmp_eq_i32 s18, 0 // (op 27208) 
s_cselect_b32 s19, 0, -1 // (op 27208) 

s_and_b32 s18, s15, 31 // (op 27208) 
s_lshl_b32 s22, 1, s18 // (op 27208) 
s_add_i32 s23, s22, s19 // (op 27208) 


s_and_b32 s19, s1, s23 // (op 27208) 


s_add_i32 s1, s47, s19 // (op 27208) 


s_ashr_i32 s19, s1, s18 // (op 27208) 


s_lshl_b32 s1, s15, 24 // (op 27208) 
s_ashr_i32 s18, s1, 31 // (op 27208) 

s_xor_b32 s1, s19, s18 // (op 27208) 

s_sub_i32 s19, s1, s18 // (op 27208) 


s_mul_i32 s1, s0, s11 // (op 27208) 

s_sub_u32 s0, s16, s1 // (op 27208) 

s_mul_i32 s1, s19, s11 // (op 27208) 
s_add_u32 s1, s1, s0 // (op 27208) 


s_mul_hi_i32 s0, s16, s9 // (op 27208) 
s_add_i32 s47, s0, s16 // (op 27208) 

s_ashr_i32 s0, s47, 31 // (op 27208) 
s_cmp_eq_i32 s9, 0 // (op 27208) 
s_mov_b32 s18, scc // (op 27208) 
s_cmp_eq_i32 s18, 0 // (op 27208) 
s_cselect_b32 s19, 0, -1 // (op 27208) 

s_and_b32 s18, s45, 31 // (op 27208) 
s_lshl_b32 s22, 1, s18 // (op 27208) 
s_add_i32 s23, s22, s19 // (op 27208) 


s_and_b32 s19, s0, s23 // (op 27208) 


s_add_i32 s0, s47, s19 // (op 27208) 


s_ashr_i32 s19, s0, s18 // (op 27208) 


s_lshl_b32 s0, s45, 24 // (op 27208) 
s_ashr_i32 s18, s0, 31 // (op 27208) 

s_xor_b32 s0, s19, s18 // (op 27208) 

s_sub_i32 s19, s0, s18 // (op 27208) 


s_add_u32 s22, s6, 256 // (op 27208) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27208) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27208) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27208) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27208) 

s_mul_i32 s0, s11, s22 // (op 27208) 

s_mul_i32 s18, s8, s0 // (op 27208) 

s_mul_i32 s0, s19, s18 // (op 27208) 

s_sub_u32 s18, s16, s0 // (op 27208) 

s_mov_b32 s0, s18 // (op 27208) 
s_mul_hi_i32 s47, s0, s12 // (op 27208) 
s_add_i32 s22, s47, s0 // (op 27208) 


s_ashr_i32 s0, s22, 31 // (op 27208) 
s_cmp_eq_i32 s12, 0 // (op 27208) 
s_mov_b32 s47, scc // (op 27208) 
s_cmp_eq_i32 s47, 0 // (op 27208) 
s_cselect_b32 s23, 0, -1 // (op 27208) 

s_and_b32 s47, s44, 31 // (op 27208) 
s_lshl_b32 s24, 1, s47 // (op 27208) 
s_add_i32 s25, s24, s23 // (op 27208) 


s_and_b32 s23, s0, s25 // (op 27208) 


s_add_i32 s0, s22, s23 // (op 27208) 


s_ashr_i32 s22, s0, s47 // (op 27208) 


s_lshl_b32 s0, s44, 24 // (op 27208) 
s_ashr_i32 s47, s0, 31 // (op 27208) 

s_xor_b32 s0, s22, s47 // (op 27208) 

s_sub_i32 s22, s0, s47 // (op 27208) 


s_mul_i32 s0, s46, s11 // (op 27208) 
s_add_u32 s24, s20, 256 // (op 27208) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 27208) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 27208) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 27208) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 27208) 

s_sub_i32 s23, s24, s0 // (op 27208) 

s_mul_i32 s47, s22, s23 // (op 27208) 


s_sub_u32 s22, s18, s47 // (op 27208) 


s_add_u32 s18, s22, s0 // (op 27208) 


s_cmp_eq_i32 s19, 0 // (op 27208) 
s_mov_b32 s0, scc // (op 27208) 

s_cmp_eq_i32 s0, 0 // (op 27208) 
s_cselect_b32 s19, s18, s1 // (op 27208) 



v_lshl_add_u32 v12, s19, 8, v10 // (op 27208) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 27208) 
s_lshr_b32 s47, s0, 1 // (op 27208) 

v_lshl_add_u32 v10, s47, 8, v11 // (op 27208) 


v_mov_b32 v53, 0 // (op 27208) convert
v_mov_b32 v52, v10 // (op 27208) convert

v_mul_lo_u32 v11, v12, s27 // (op 27208) most significant: low of low * high
v_mul_hi_u32 v13, v12, s26 // (op 27208) most significant: high of low * low
v_mul_lo_u32 v10, v12, s26 // (op 27208) least significant: low of low * low
v_add_u32 v11, v11, v13 // (op 27208) most significant: sum

v_add_co_u32 v10, s[0:1], v10, v52 // (op 27208) least significant half
v_addc_co_u32 v11, s[0:1], v11, v53, s[0:1] // (op 27208) most significant half



s_mul_i32 s1, s26, s21 // (op 27210) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 27210) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 27210) least significant
s_mul_hi_u32 s18, s26, s20 // (op 27210) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27210) most significant: sum
s_add_u32 s1, s1, s18 // (op 27210) most significant: sum


s_mov_b32 s48, s28 // (op 27210) reg expression
s_mov_b32 s49, s29 // (op 27210) reg expression
s_mov_b32 s50, s0 // (op 27210) reg expression
s_mov_b32 s51, 131072 // (op 27210) call()

s_mul_hi_i32 s47, s16, s10 // (op 27338) 
v_and_b32 v12, 4294967294, v1 // (op 27338) 
s_add_i32 s0, s47, s16 // (op 27338) 

v_and_b32 v13, -2, v1 // (op 27338) 
s_ashr_i32 s1, s0, 31 // (op 27338) 
v_sub_u32 v14, v1, v13 // (op 27338) 

s_cmp_eq_i32 s10, 0 // (op 27338) 
s_mov_b32 s47, scc // (op 27338) 
v_lshlrev_b32 v13, 6, v14 // (op 27338) 

s_cmp_eq_i32 s47, 0 // (op 27338) 
s_cselect_b32 s18, 0, -1 // (op 27338) 

s_and_b32 s19, s13, 31 // (op 27338) 
s_lshl_b32 s47, 1, s19 // (op 27338) 
s_add_i32 s22, s47, s18 // (op 27338) 


s_and_b32 s18, s1, s22 // (op 27338) 


s_add_i32 s1, s0, s18 // (op 27338) 


s_ashr_i32 s0, s1, s19 // (op 27338) 


s_lshl_b32 s1, s13, 24 // (op 27338) 
s_ashr_i32 s47, s1, 31 // (op 27338) 

s_xor_b32 s1, s0, s47 // (op 27338) 

s_sub_i32 s0, s1, s47 // (op 27338) 


s_mul_hi_i32 s1, s0, s14 // (op 27338) 
s_add_i32 s47, s1, s0 // (op 27338) 

s_ashr_i32 s1, s47, 31 // (op 27338) 
s_cmp_eq_i32 s14, 0 // (op 27338) 
s_mov_b32 s18, scc // (op 27338) 
s_cmp_eq_i32 s18, 0 // (op 27338) 
s_cselect_b32 s19, 0, -1 // (op 27338) 

s_and_b32 s18, s15, 31 // (op 27338) 
s_lshl_b32 s22, 1, s18 // (op 27338) 
s_add_i32 s23, s22, s19 // (op 27338) 


s_and_b32 s19, s1, s23 // (op 27338) 


s_add_i32 s1, s47, s19 // (op 27338) 


s_ashr_i32 s19, s1, s18 // (op 27338) 


s_lshl_b32 s1, s15, 24 // (op 27338) 
s_ashr_i32 s18, s1, 31 // (op 27338) 

s_xor_b32 s1, s19, s18 // (op 27338) 

s_sub_i32 s19, s1, s18 // (op 27338) 


s_add_u32 s22, s6, 256 // (op 27338) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27338) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27338) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27338) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27338) 

s_mul_i32 s1, s19, s22 // (op 27338) 

s_sub_i32 s47, s0, s1 // (op 27338) 


s_mul_hi_i32 s0, s16, s9 // (op 27338) 
s_add_i32 s1, s0, s16 // (op 27338) 

s_ashr_i32 s0, s1, 31 // (op 27338) 
s_cmp_eq_i32 s9, 0 // (op 27338) 
s_mov_b32 s18, scc // (op 27338) 
s_cmp_eq_i32 s18, 0 // (op 27338) 
s_cselect_b32 s19, 0, -1 // (op 27338) 

s_and_b32 s18, s45, 31 // (op 27338) 
s_lshl_b32 s24, 1, s18 // (op 27338) 
s_add_i32 s25, s24, s19 // (op 27338) 


s_and_b32 s19, s0, s25 // (op 27338) 


s_add_i32 s0, s1, s19 // (op 27338) 


s_ashr_i32 s1, s0, s18 // (op 27338) 


s_lshl_b32 s0, s45, 24 // (op 27338) 
s_ashr_i32 s18, s0, 31 // (op 27338) 

s_xor_b32 s0, s1, s18 // (op 27338) 

s_sub_i32 s1, s0, s18 // (op 27338) 


s_mul_i32 s0, s11, s22 // (op 27338) 

s_mul_i32 s18, s8, s0 // (op 27338) 

s_mul_i32 s0, s1, s18 // (op 27338) 

s_sub_u32 s18, s16, s0 // (op 27338) 

s_mul_hi_i32 s0, s18, s12 // (op 27338) 
s_add_i32 s19, s0, s18 // (op 27338) 


s_ashr_i32 s0, s19, 31 // (op 27338) 
s_cmp_eq_i32 s12, 0 // (op 27338) 
s_mov_b32 s18, scc // (op 27338) 
s_cmp_eq_i32 s18, 0 // (op 27338) 
s_cselect_b32 s22, 0, -1 // (op 27338) 

s_and_b32 s18, s44, 31 // (op 27338) 
s_lshl_b32 s23, 1, s18 // (op 27338) 
s_add_i32 s24, s23, s22 // (op 27338) 


s_and_b32 s25, s0, s24 // (op 27338) 


s_add_i32 s0, s19, s25 // (op 27338) 


s_ashr_i32 s19, s0, s18 // (op 27338) 


s_lshl_b32 s0, s44, 24 // (op 27338) 
s_ashr_i32 s18, s0, 31 // (op 27338) 

s_xor_b32 s0, s19, s18 // (op 27338) 

s_sub_i32 s19, s0, s18 // (op 27338) 


s_cmp_eq_i32 s1, 0 // (op 27338) 
s_mov_b32 s0, scc // (op 27338) 

s_cmp_eq_i32 s0, 0 // (op 27338) 
s_cselect_b32 s1, s19, s47 // (op 27338) 



v_lshl_add_u32 v14, s1, 8, v12 // (op 27338) 


v_mul_lo_u32 v53, v14, s31 // (op 27338) most significant: low of low * high
v_mul_hi_u32 v12, v14, s30 // (op 27338) most significant: high of low * low
v_mul_lo_u32 v52, v14, s30 // (op 27338) least significant: low of low * low
v_add_u32 v53, v53, v12 // (op 27338) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 27338) 
s_lshr_b32 s47, s0, 1 // (op 27338) 

v_lshl_add_u32 v12, s47, 8, v13 // (op 27338) 


v_mov_b32 v19, 0 // (op 27338) convert
v_mov_b32 v18, v12 // (op 27338) convert

v_add_co_u32 v12, s[0:1], v18, v52 // (op 27338) least significant half
v_addc_co_u32 v13, s[0:1], v19, v53, s[0:1] // (op 27338) most significant half



v_mul_lo_u32 v53, v13, 6 // (op 27338) most significant: low of high * low
v_mul_hi_u32 v14, v12, 6 // (op 27338) most significant: high of low * low
v_mul_lo_u32 v52, v12, 6 // (op 27338) least significant: low of low * low
v_add_u32 v53, v53, v14 // (op 27338) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[52:53] // (op 27338) 

v_mov_b32 v14, v12 // (op 27338) convert

s_mul_i32 s1, s7, 6 // (op 27340) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 27340) least significant
s_mul_hi_u32 s47, s6, 6 // (op 27340) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27340) most significant: sum

s_mul_i32 s19, s30, s1 // (op 27340) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 27340) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 27340) least significant
s_mul_hi_u32 s22, s30, s0 // (op 27340) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 27340) most significant: sum
s_add_u32 s19, s19, s22 // (op 27340) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 27340) 

s_mov_b32 s60, s32 // (op 27340) reg expression
s_mov_b32 s61, s33 // (op 27340) reg expression
s_mov_b32 s62, s0 // (op 27340) reg expression
s_mov_b32 s63, 131072 // (op 27340) call()

v_and_b32 v12, -8, v1 // (op 27421) 
v_sub_u32 v13, v1, v12 // (op 27421) 

v_lshlrev_b32 v12, 4, v13 // (op 27421) 

v_lshrrev_b32 v13, 3, v1 // (op 27421) 
v_lshl_add_u32 v15, v13, 7, v12 // (op 27421) 


s_nop 0
v_readfirstlane_b32 s47, v15 // (op 27421) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -2, v1 // (op 27480) 
v_sub_u32 v13, v1, v12 // (op 27480) 

v_lshlrev_b32 v12, 6, v13 // (op 27480) 

v_lshrrev_b32 v13, 1, v1 // (op 27480) 
v_lshl_add_u32 v15, v13, 8, v12 // (op 27480) 


v_mul_lo_u32 v12, v15, 6 // (op 27480) 

v_lshrrev_b32 v13, 3, v12 // (op 27480) 

s_mul_hi_i32 s0, s16, s10 // (op 27744) 
v_and_b32 v12, 63, v1 // (op 27744) 
s_add_i32 s1, s0, s16 // (op 27744) 

v_lshrrev_b32 v15, 4, v12 // (op 27744) 
s_ashr_i32 s0, s1, 31 // (op 27744) 
v_and_b32 v16, 3, v15 // (op 27744) 

s_cmp_eq_i32 s10, 0 // (op 27744) 
s_mov_b32 s18, scc // (op 27744) 
v_and_b32 v15, 15, v1 // (op 27744) 
s_cmp_eq_i32 s18, 0 // (op 27744) 
s_cselect_b32 s19, 0, -1 // (op 27744) 

v_lshl_add_u32 v17, v16, 4, v15 // (op 27744) 


s_and_b32 s18, s13, 31 // (op 27744) 
v_and_b32 v15, 4294967168, v1 // (op 27744) 
s_lshl_b32 s22, 1, s18 // (op 27744) 
v_add_u32 v16, v15, v17 // (op 27744) 


s_add_i32 s23, s22, s19 // (op 27744) 


v_lshrrev_b32 v15, 6, v12 // (op 27744) 

s_and_b32 s19, s0, s23 // (op 27744) 


v_lshlrev_b32 v12, 2, v15 // (op 27744) 

s_add_i32 s0, s1, s19 // (op 27744) 


s_ashr_i32 s1, s0, s18 // (op 27744) 


s_lshl_b32 s0, s13, 24 // (op 27744) 
s_ashr_i32 s18, s0, 31 // (op 27744) 

s_xor_b32 s0, s1, s18 // (op 27744) 

s_sub_i32 s1, s0, s18 // (op 27744) 


s_mul_hi_i32 s0, s1, s14 // (op 27744) 
s_add_i32 s18, s0, s1 // (op 27744) 

s_ashr_i32 s0, s18, 31 // (op 27744) 
s_cmp_eq_i32 s14, 0 // (op 27744) 
s_mov_b32 s19, scc // (op 27744) 
s_cmp_eq_i32 s19, 0 // (op 27744) 
s_cselect_b32 s22, 0, -1 // (op 27744) 

s_and_b32 s19, s15, 31 // (op 27744) 
s_lshl_b32 s23, 1, s19 // (op 27744) 
s_add_i32 s24, s23, s22 // (op 27744) 


s_and_b32 s25, s0, s24 // (op 27744) 


s_add_i32 s0, s18, s25 // (op 27744) 


s_ashr_i32 s18, s0, s19 // (op 27744) 


s_lshl_b32 s0, s15, 24 // (op 27744) 
s_ashr_i32 s19, s0, 31 // (op 27744) 

s_xor_b32 s0, s18, s19 // (op 27744) 

s_sub_i32 s18, s0, s19 // (op 27744) 


s_mul_i32 s0, s1, s11 // (op 27744) 

s_sub_u32 s1, s16, s0 // (op 27744) 

s_mul_i32 s0, s18, s11 // (op 27744) 
s_add_u32 s0, s0, s1 // (op 27744) 


s_mul_hi_i32 s1, s16, s9 // (op 27744) 
s_add_i32 s18, s1, s16 // (op 27744) 

s_ashr_i32 s1, s18, 31 // (op 27744) 
s_cmp_eq_i32 s9, 0 // (op 27744) 
s_mov_b32 s19, scc // (op 27744) 
s_cmp_eq_i32 s19, 0 // (op 27744) 
s_cselect_b32 s22, 0, -1 // (op 27744) 

s_and_b32 s19, s45, 31 // (op 27744) 
s_lshl_b32 s23, 1, s19 // (op 27744) 
s_add_i32 s24, s23, s22 // (op 27744) 


s_and_b32 s25, s1, s24 // (op 27744) 


s_add_i32 s1, s18, s25 // (op 27744) 


s_ashr_i32 s18, s1, s19 // (op 27744) 


s_lshl_b32 s1, s45, 24 // (op 27744) 
s_ashr_i32 s19, s1, 31 // (op 27744) 

s_xor_b32 s1, s18, s19 // (op 27744) 

s_sub_i32 s18, s1, s19 // (op 27744) 


s_add_u32 s22, s6, 256 // (op 27744) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27744) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27744) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27744) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27744) 

s_mul_i32 s1, s11, s22 // (op 27744) 

s_mul_i32 s19, s8, s1 // (op 27744) 

s_mul_i32 s1, s18, s19 // (op 27744) 

s_sub_u32 s19, s16, s1 // (op 27744) 

s_mov_b32 s1, s19 // (op 27744) 
s_mul_hi_i32 s22, s1, s12 // (op 27744) 
s_add_i32 s23, s22, s1 // (op 27744) 


s_ashr_i32 s1, s23, 31 // (op 27744) 
s_cmp_eq_i32 s12, 0 // (op 27744) 
s_mov_b32 s22, scc // (op 27744) 
s_cmp_eq_i32 s22, 0 // (op 27744) 
s_cselect_b32 s24, 0, -1 // (op 27744) 

s_and_b32 s22, s44, 31 // (op 27744) 
s_lshl_b32 s25, 1, s22 // (op 27744) 
s_add_i32 s64, s25, s24 // (op 27744) 


s_and_b32 s24, s1, s64 // (op 27744) 


s_add_i32 s1, s23, s24 // (op 27744) 


s_ashr_i32 s23, s1, s22 // (op 27744) 


s_lshl_b32 s1, s44, 24 // (op 27744) 
s_ashr_i32 s22, s1, 31 // (op 27744) 

s_xor_b32 s1, s23, s22 // (op 27744) 

s_sub_i32 s23, s1, s22 // (op 27744) 


s_mul_i32 s1, s46, s11 // (op 27744) 
s_add_u32 s24, s20, 256 // (op 27744) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 27744) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 27744) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 27744) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 27744) 

s_sub_i32 s22, s24, s1 // (op 27744) 

s_mul_i32 s24, s23, s22 // (op 27744) 


s_sub_u32 s25, s19, s24 // (op 27744) 


s_add_u32 s19, s25, s1 // (op 27744) 


s_cmp_eq_i32 s18, 0 // (op 27744) 
s_mov_b32 s1, scc // (op 27744) 

s_cmp_eq_i32 s1, 0 // (op 27744) 
s_cselect_b32 s18, s19, s0 // (op 27744) 



v_lshl_add_u32 v15, s18, 8, v16 // (op 27744) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 27744) 
s_lshr_b32 s18, s0, 1 // (op 27744) 

v_lshl_add_u32 v16, s18, 3, v12 // (op 27744) 


v_mul_lo_u32 v12, v15, s34 // (op 27744) 
v_add_u32 v12, v12, v16 // (op 27744) 


s_mul_i32 s1, s34, s21 // (op 27746) most significant: low of low * high
s_mul_i32 s18, s35, s20 // (op 27746) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 27746) least significant
s_mul_hi_u32 s19, s34, s20 // (op 27746) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 27746) most significant: sum
s_add_u32 s1, s1, s19 // (op 27746) most significant: sum


s_mov_b32 s64, s36 // (op 27746) reg expression
s_mov_b32 s65, s37 // (op 27746) reg expression
s_mov_b32 s66, s0 // (op 27746) reg expression
s_mov_b32 s67, 131072 // (op 27746) call()

s_mul_hi_i32 s0, s16, s10 // (op 27771) 
v_and_b32 v15, 63, v1 // (op 27771) 
s_add_i32 s1, s0, s16 // (op 27771) 

v_lshrrev_b32 v16, 4, v15 // (op 27771) 
s_ashr_i32 s0, s1, 31 // (op 27771) 
v_and_b32 v17, 3, v16 // (op 27771) 

s_cmp_eq_i32 s10, 0 // (op 27771) 
s_mov_b32 s18, scc // (op 27771) 
v_and_b32 v16, 15, v1 // (op 27771) 
s_cmp_eq_i32 s18, 0 // (op 27771) 
s_cselect_b32 s19, 0, -1 // (op 27771) 

v_lshl_add_u32 v18, v17, 4, v16 // (op 27771) 


s_and_b32 s18, s13, 31 // (op 27771) 
v_lshrrev_b32 v19, 6, v1 // (op 27771) 
s_lshl_b32 s22, 1, s18 // (op 27771) 
v_and_b32 v16, 1, v19 // (op 27771) 

s_add_i32 s23, s22, s19 // (op 27771) 


v_lshl_add_u32 v17, v16, 7, v18 // (op 27771) 


s_and_b32 s19, s0, s23 // (op 27771) 


v_lshrrev_b32 v16, 6, v15 // (op 27771) 

s_add_i32 s0, s1, s19 // (op 27771) 


v_lshlrev_b32 v15, 2, v16 // (op 27771) 

s_ashr_i32 s1, s0, s18 // (op 27771) 


s_lshl_b32 s0, s13, 24 // (op 27771) 
s_ashr_i32 s18, s0, 31 // (op 27771) 

s_xor_b32 s0, s1, s18 // (op 27771) 

s_sub_i32 s1, s0, s18 // (op 27771) 


s_mul_hi_i32 s0, s1, s14 // (op 27771) 
s_add_i32 s18, s0, s1 // (op 27771) 

s_ashr_i32 s0, s18, 31 // (op 27771) 
s_cmp_eq_i32 s14, 0 // (op 27771) 
s_mov_b32 s19, scc // (op 27771) 
s_cmp_eq_i32 s19, 0 // (op 27771) 
s_cselect_b32 s22, 0, -1 // (op 27771) 

s_and_b32 s19, s15, 31 // (op 27771) 
s_lshl_b32 s23, 1, s19 // (op 27771) 
s_add_i32 s24, s23, s22 // (op 27771) 


s_and_b32 s25, s0, s24 // (op 27771) 


s_add_i32 s0, s18, s25 // (op 27771) 


s_ashr_i32 s18, s0, s19 // (op 27771) 


s_lshl_b32 s0, s15, 24 // (op 27771) 
s_ashr_i32 s19, s0, 31 // (op 27771) 

s_xor_b32 s0, s18, s19 // (op 27771) 

s_sub_i32 s18, s0, s19 // (op 27771) 


s_add_u32 s22, s6, 256 // (op 27771) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27771) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27771) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27771) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27771) 

s_mul_i32 s0, s18, s22 // (op 27771) 

s_sub_i32 s18, s1, s0 // (op 27771) 


s_mul_hi_i32 s19, s16, s9 // (op 27771) 
s_add_i32 s0, s19, s16 // (op 27771) 

s_ashr_i32 s1, s0, 31 // (op 27771) 
s_cmp_eq_i32 s9, 0 // (op 27771) 
s_mov_b32 s19, scc // (op 27771) 
s_cmp_eq_i32 s19, 0 // (op 27771) 
s_cselect_b32 s24, 0, -1 // (op 27771) 

s_and_b32 s19, s45, 31 // (op 27771) 
s_lshl_b32 s25, 1, s19 // (op 27771) 
s_add_i32 s68, s25, s24 // (op 27771) 


s_and_b32 s24, s1, s68 // (op 27771) 


s_add_i32 s1, s0, s24 // (op 27771) 


s_ashr_i32 s0, s1, s19 // (op 27771) 


s_lshl_b32 s1, s45, 24 // (op 27771) 
s_ashr_i32 s19, s1, 31 // (op 27771) 

s_xor_b32 s1, s0, s19 // (op 27771) 

s_sub_i32 s0, s1, s19 // (op 27771) 


s_mul_i32 s1, s11, s22 // (op 27771) 

s_mul_i32 s19, s8, s1 // (op 27771) 

s_mul_i32 s1, s0, s19 // (op 27771) 

s_sub_u32 s19, s16, s1 // (op 27771) 

s_mul_hi_i32 s1, s19, s12 // (op 27771) 
s_add_i32 s22, s1, s19 // (op 27771) 


s_ashr_i32 s1, s22, 31 // (op 27771) 
s_cmp_eq_i32 s12, 0 // (op 27771) 
s_mov_b32 s19, scc // (op 27771) 
s_cmp_eq_i32 s19, 0 // (op 27771) 
s_cselect_b32 s23, 0, -1 // (op 27771) 

s_and_b32 s19, s44, 31 // (op 27771) 
s_lshl_b32 s24, 1, s19 // (op 27771) 
s_add_i32 s25, s24, s23 // (op 27771) 


s_and_b32 s23, s1, s25 // (op 27771) 


s_add_i32 s1, s22, s23 // (op 27771) 


s_ashr_i32 s22, s1, s19 // (op 27771) 


s_lshl_b32 s1, s44, 24 // (op 27771) 
s_ashr_i32 s19, s1, 31 // (op 27771) 

s_xor_b32 s1, s22, s19 // (op 27771) 

s_sub_i32 s22, s1, s19 // (op 27771) 


s_cmp_eq_i32 s0, 0 // (op 27771) 
s_mov_b32 s1, scc // (op 27771) 

s_cmp_eq_i32 s1, 0 // (op 27771) 
s_cselect_b32 s0, s22, s18 // (op 27771) 



v_lshl_add_u32 v16, s0, 8, v17 // (op 27771) 


v_mul_lo_u32 v17, v16, s38 // (op 27771) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 27771) 
s_lshr_b32 s18, s0, 1 // (op 27771) 

v_lshl_add_u32 v16, s18, 3, v15 // (op 27771) 


v_add_u32 v15, v16, v17 // (op 27771) 


s_mul_i32 s1, s38, s7 // (op 27773) most significant: low of low * high
s_mul_i32 s18, s39, s6 // (op 27773) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 27773) least significant
s_mul_hi_u32 s19, s38, s6 // (op 27773) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 27773) most significant: sum
s_add_u32 s1, s1, s19 // (op 27773) most significant: sum


s_mov_b32 s68, s40 // (op 27773) reg expression
s_mov_b32 s69, s41 // (op 27773) reg expression
s_mov_b32 s70, s0 // (op 27773) reg expression
s_mov_b32 s71, 131072 // (op 27773) call()

s_ashr_i64 s[0:1], s[42:43], 7 // (op 2823) 
s_and_b32 s18, s0, 4294967294 // (op 2823) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 2822) 
s_mov_b32 s19, s0 // (op 2822) convert

s_cmp_lt_u32 s18, s19 // (op 2822) 






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_2822 // (op 2822) Condition: Top (jump to RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_2822 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoopTail_2822:
s_barrier  // (op 28027) 
s_add_u32 s18, s18, 1 // (op 2824) 

s_lshl_b64 s[0:1], s[26:27], 5 // (op 27179) 
v_mov_b64 v[16:17], s[0:1] // (op 27179) convert


v_mov_b64 v[18:19], v[10:11] // (op 27179) 
v_mov_b64 v[20:21], v[18:19] // (op 27179) 
s_mov_b32 m0, s47 // (op 27179) reg expression
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 27179) Load value direct to lds
v_add_co_u32 v20, s[0:1], v20, v16 // (op 27179) least significant half
v_addc_co_u32 v21, s[0:1], 0, v21, s[0:1] // (op 27179) most significant half

s_add_u32 m0, m0, 4096 // (op 27179) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 27179) Load value direct to lds
v_add_co_u32 v20, s[0:1], v20, v16 // (op 27179) least significant half
v_addc_co_u32 v21, s[0:1], 0, v21, s[0:1] // (op 27179) most significant half

s_add_u32 m0, m0, 4096 // (op 27179) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 27179) Load value direct to lds
v_add_co_u32 v20, s[0:1], v20, v16 // (op 27179) least significant half
v_addc_co_u32 v21, s[0:1], 0, v21, s[0:1] // (op 27179) most significant half

s_add_u32 m0, m0, 4096 // (op 27179) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 27179) Load value direct to lds
v_add_co_u32 v20, s[0:1], v20, v16 // (op 27179) least significant half
v_addc_co_u32 v21, s[0:1], 0, v21, s[0:1] // (op 27179) most significant half

s_add_u32 m0, m0, 4096 // (op 27179) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 27179) Load value direct to lds
v_add_co_u32 v20, s[0:1], v20, v16 // (op 27179) least significant half
v_addc_co_u32 v21, s[0:1], 0, v21, s[0:1] // (op 27179) most significant half

s_add_u32 m0, m0, 4096 // (op 27179) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 27179) Load value direct to lds
v_add_co_u32 v20, s[0:1], v20, v16 // (op 27179) least significant half
v_addc_co_u32 v21, s[0:1], 0, v21, s[0:1] // (op 27179) most significant half

s_add_u32 m0, m0, 4096 // (op 27179) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 27179) Load value direct to lds
v_add_co_u32 v20, s[0:1], v20, v16 // (op 27179) least significant half
v_addc_co_u32 v21, s[0:1], 0, v21, s[0:1] // (op 27179) most significant half

s_add_u32 m0, m0, 4096 // (op 27179) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 27179) Load value direct to lds



v_add_co_u32 v10, vcc, 128, v10 // (op 27198) least significant half
v_addc_co_u32 v11, vcc, 0, v11, vcc // (op 27198) most significant half

s_mul_i32 s1, s31, 6 // (op 173) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 173) least significant
s_mul_hi_u32 s19, s30, 6 // (op 173) most significant: high of low * low
s_add_u32 s1, s1, s19 // (op 173) most significant: sum

v_lshrrev_b64 v[16:17], 3, s[0:1] // (op 173) 


v_mov_b32 v18, v14 // (op 173) 
v_mov_b32 v19, v18 // (op 173) 
buffer_load_dwordx4 v[20:23], v19, s[60:63], 0 offen // (op 173) Load value
buffer_load_dwordx2 v[24:25], v19, s[60:63], 0 offen offset:16 // (op 173) Load value
buffer_load_dwordx4 v[26:29], v19, s[60:63], 0 offen offset:24 // (op 173) Load value
buffer_load_dwordx2 v[30:31], v19, s[60:63], 0 offen offset:40 // (op 173) Load value
v_add_u32 v19, v19, v16 // (op 173) 
buffer_load_dwordx4 v[32:35], v19, s[60:63], 0 offen // (op 173) Load value
buffer_load_dwordx2 v[36:37], v19, s[60:63], 0 offen offset:16 // (op 173) Load value
buffer_load_dwordx4 v[38:41], v19, s[60:63], 0 offen offset:24 // (op 173) Load value
buffer_load_dwordx2 v[42:43], v19, s[60:63], 0 offen offset:40 // (op 173) Load value



v_add_u32 v14, 96, v14 // (op 27330) 
s_barrier  // (op 28043) 
s_waitcnt vmcnt(8)
s_barrier  // (op 28033) 






v_add_u32 v16, 65536, v13 // (op 172) 
ds_read_b128 v[44:47], v3 // (op 164) Load local data 
ds_read_b128 v[48:51], v3 offset:64 // (op 164) Load local data 
s_waitcnt vmcnt(7)
ds_write_b128 v16, v[20:23] // (op 172) Store local data 
s_waitcnt vmcnt(6)
ds_write_b64 v16, v[24:25] offset:16 // (op 172) Store local data 

v_add_u32 v16, 65560, v13 // (op 172) 
s_waitcnt vmcnt(5)
ds_write_b128 v16, v[26:29] // (op 172) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v16, v[30:31] offset:16 // (op 172) Store local data 

v_add_u32 v16, 65632, v13 // (op 172) 
s_waitcnt vmcnt(3)
ds_write_b128 v16, v[32:35] // (op 172) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v16, v[36:37] offset:16 // (op 172) Store local data 

v_add_u32 v16, 65656, v13 // (op 172) 
s_waitcnt vmcnt(1)
ds_write_b128 v16, v[38:41] // (op 172) Store local data 
s_waitcnt vmcnt(0)
ds_write_b64 v16, v[42:43] offset:16 // (op 172) Store local data 


s_waitcnt lgkmcnt(0)
s_barrier  // (op 28047) 



v_add_u32 v16, 65536, v4 // (op 166) 
ds_read_b128 v[38:41], v16 // (op 166) Load local data 
ds_read_b64 v[42:43], v16 offset:16 // (op 166) Load local data 



v_mov_b32 v16, v12 // (op 168) 
v_mov_b32 v17, v16 // (op 168) reg expression

buffer_load_dwordx2 v[18:19], v17, s[64:67], 0 offen // (op 168) Load value



v_mov_b32 v17, v15 // (op 170) 
v_mov_b32 v20, v17 // (op 170) reg expression

buffer_load_dwordx2 v[36:37], v20, s[68:71], 0 offen // (op 170) Load value

s_waitcnt vmcnt(1)
v_bfe_u32 v20, v18, 0, 8 // (op 25190) 



v_bfe_u32 v21, v18, 8, 8 // (op 25190) 
v_bfe_u32 v22, v18, 16, 8 // (op 25190) 
v_bfe_u32 v23, v18, 24, 8 // (op 25190) 
v_permlane16_swap_b32 v20, v21 // (op 25190) 
v_add_u32 v24, 67072, v4 // (op 295) 
v_permlane16_swap_b32 v22, v23 // (op 25190) 
ds_read_b128 v[30:33], v24 // (op 295) Load local data 
s_nop 0
v_permlane32_swap_b32 v20, v22 // (op 25190) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v21, v23 // (op 25190) 
ds_read_b64 v[34:35], v24 offset:16 // (op 295) Load local data 





v_add_u32 v18, 68608, v4 // (op 431) 
ds_read_b128 v[24:27], v18 // (op 431) Load local data 
ds_read_b64 v[28:29], v18 offset:16 // (op 431) Load local data 

s_waitcnt vmcnt(0)
v_bfe_u32 v52, v36, 0, 8 // (op 25282) 



v_bfe_u32 v53, v36, 8, 8 // (op 25282) 
v_bfe_u32 v54, v36, 16, 8 // (op 25282) 
v_bfe_u32 v55, v36, 24, 8 // (op 25282) 
v_permlane16_swap_b32 v52, v53 // (op 25282) 
v_add_u32 v18, 70144, v4 // (op 567) 
v_permlane16_swap_b32 v54, v55 // (op 25282) 
s_nop 1
v_permlane32_swap_b32 v52, v54 // (op 25282) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v53, v55 // (op 25282) 
ds_read_b128 v[56:59], v18 // (op 567) Load local data 

ds_read_b64 v[60:61], v18 offset:16 // (op 567) Load local data 




s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[38:43], a[0:3], v20, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 171) 
v_add_u32 v18, 71680, v4 // (op 703) 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[30:35], a[4:7], v20, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 298) 
ds_read_b128 v[62:65], v18 // (op 703) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[44:51], v[24:29], a[8:11], v20, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 434) 
ds_read_b64 v[66:67], v18 offset:16 // (op 703) Load local data 



s_lshl_b32 s19, s38, 6 // (op 708) 
v_add_u32 v18, v17, s19 // (op 708) 


buffer_load_dwordx2 v[36:37], v18, s[68:71], 0 offen // (op 708) Load value

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[44:51], v[56:61], a[12:15], v20, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 570) 




v_add_u32 v15, 4, v15 // (op 27761) 
v_add_u32 v17, 73216, v4 // (op 839) 
ds_read_b128 v[68:71], v17 // (op 839) Load local data 
ds_read_b64 v[72:73], v17 offset:16 // (op 839) Load local data 

s_waitcnt vmcnt(0)
v_bfe_u32 v74, v36, 0, 8 // (op 25296) 



v_bfe_u32 v75, v36, 8, 8 // (op 25296) 
v_bfe_u32 v76, v36, 16, 8 // (op 25296) 
v_bfe_u32 v77, v36, 24, 8 // (op 25296) 
v_permlane16_swap_b32 v74, v75 // (op 25296) 
v_add_u32 v17, 74752, v4 // (op 975) 
v_permlane16_swap_b32 v76, v77 // (op 25296) 
s_nop 1
v_permlane32_swap_b32 v74, v76 // (op 25296) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v75, v77 // (op 25296) 
ds_read_b128 v[78:81], v17 // (op 975) Load local data 

ds_read_b64 v[82:83], v17 offset:16 // (op 975) Load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[44:51], v[62:67], a[16:19], v20, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 706) 
v_add_u32 v17, 76288, v4 // (op 1111) 
ds_read_b128 v[84:87], v17 // (op 1111) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[44:51], v[68:73], a[20:23], v20, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 842) 
ds_read_b64 v[88:89], v17 offset:16 // (op 1111) Load local data 




s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[44:51], v[78:83], a[24:27], v20, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 978) 
ds_read_b128 v[90:93], v3 offset:2048 // (op 1421) Load local data 
ds_read_b128 v[94:97], v3 offset:2112 // (op 1421) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[44:51], v[84:89], a[28:31], v20, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1114) 




ds_read_b128 v[44:47], v3 offset:4096 // (op 2639) Load local data 
ds_read_b128 v[48:51], v3 offset:4160 // (op 2639) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:97], v[38:43], a[32:35], v21, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1425) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:97], v[24:29], a[40:43], v21, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1479) 
ds_read_b128 v[98:101], v3 offset:6144 // (op 3857) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[90:97], v[56:61], a[44:47], v21, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1506) 
ds_read_b128 v[102:105], v3 offset:6208 // (op 3857) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[90:97], v[30:35], a[36:39], v21, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1452) 
ds_read_b128 v[106:109], v3 offset:8192 // (op 5075) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:97], v[62:67], a[48:51], v21, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1533) 
ds_read_b128 v[110:113], v3 offset:8256 // (op 5075) Load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:97], v[78:83], a[56:59], v21, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1587) 
s_lshl_b32 s19, s34, 6 // (op 5084) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:97], v[84:89], a[60:63], v21, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1614) 
v_add_u32 v17, v16, s19 // (op 5084) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:97], v[68:73], a[52:55], v21, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1560) 
buffer_load_dwordx2 v[18:19], v17, s[64:67], 0 offen // (op 5084) Load value






v_add_u32 v12, 4, v12 // (op 27734) 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[44:51], v[38:43], a[64:67], v22, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 2643) 
s_waitcnt vmcnt(0)
v_bfe_u32 v90, v18, 0, 8 // (op 25204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[44:51], v[24:29], a[72:75], v22, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 2697) 
v_bfe_u32 v91, v18, 8, 8 // (op 25204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[44:51], v[56:61], a[76:79], v22, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 2724) 
v_bfe_u32 v92, v18, 16, 8 // (op 25204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[44:51], v[30:35], a[68:71], v22, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 2670) 
v_bfe_u32 v93, v18, 24, 8 // (op 25204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[44:51], v[62:67], a[80:83], v22, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 2751) 
v_permlane16_swap_b32 v90, v91 // (op 25204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[44:51], v[78:83], a[88:91], v22, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 2805) 
v_permlane16_swap_b32 v92, v93 // (op 25204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[44:51], v[84:89], a[92:95], v22, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 2832) 
ds_read_b128 v[114:117], v3 offset:10240 // (op 6293) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[44:51], v[68:73], a[84:87], v22, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 2778) 
ds_read_b128 v[118:121], v3 offset:10304 // (op 6293) Load local data 




v_permlane32_swap_b32 v90, v92 // (op 25204) 
v_permlane32_swap_b32 v91, v93 // (op 25204) 
ds_read_b128 v[44:47], v3 offset:12288 // (op 7511) Load local data 

s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[98:105], v[38:43], a[96:99], v23, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 3861) 
ds_read_b128 v[48:51], v3 offset:12352 // (op 7511) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[98:105], v[24:29], a[104:107], v23, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 3915) 
ds_read_b128 v[122:125], v3 offset:14336 // (op 8729) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[98:105], v[56:61], a[108:111], v23, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 3942) 
ds_read_b128 v[126:129], v3 offset:14400 // (op 8729) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[98:105], v[30:35], a[100:103], v23, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 3888) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[98:105], v[62:67], a[112:115], v23, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 3969) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[98:105], v[78:83], a[120:123], v23, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 4023) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[98:105], v[84:89], a[124:127], v23, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 4050) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[98:105], v[68:73], a[116:119], v23, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 3996) 


s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[106:113], v[30:35], a[132:135], v90, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5106) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[106:113], v[68:73], a[148:151], v90, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5214) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[106:113], v[38:43], a[128:131], v90, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5079) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[106:113], v[24:29], a[136:139], v90, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5133) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[106:113], v[56:61], a[140:143], v90, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5160) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[106:113], v[62:67], a[144:147], v90, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5187) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[106:113], v[78:83], a[152:155], v90, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5241) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[106:113], v[84:89], a[156:159], v90, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 5268) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[114:121], v[30:35], a[164:167], v91, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6324) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[114:121], v[68:73], a[180:183], v91, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6432) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[114:121], v[38:43], a[160:163], v91, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6297) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[114:121], v[24:29], a[168:171], v91, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6351) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[114:121], v[56:61], a[172:175], v91, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6378) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[114:121], v[62:67], a[176:179], v91, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6405) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[114:121], v[78:83], a[184:187], v91, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6459) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[114:121], v[84:89], a[188:191], v91, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 6486) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[44:51], v[30:35], a[196:199], v92, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7542) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[44:51], v[68:73], a[212:215], v92, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7650) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[44:51], v[38:43], a[192:195], v92, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7515) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[44:51], v[24:29], a[200:203], v92, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7569) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[44:51], v[56:61], a[204:207], v92, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7596) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[44:51], v[62:67], a[208:211], v92, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7623) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[44:51], v[78:83], a[216:219], v92, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7677) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[44:51], v[84:89], a[220:223], v92, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 7704) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[122:129], v[30:35], a[228:231], v93, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 8760) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[122:129], v[68:73], a[244:247], v93, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 8868) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[122:129], v[38:43], a[224:227], v93, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 8733) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[122:129], v[24:29], a[232:235], v93, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 8787) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[122:129], v[56:61], a[236:239], v93, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 8814) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[122:129], v[62:67], a[240:243], v93, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 8841) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[122:129], v[78:83], a[248:251], v93, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 8895) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[122:129], v[84:89], a[252:255], v93, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 8922) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 2822) 
s_mov_b32 s19, s0 // (op 2822) convert

s_cmp_lt_u32 s18, s19 // (op 2822) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoopTail_2822 // (op 2822) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoopTail_2822 if true)
RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_2822:




















s_barrier  // (op 28114) 

















v_cmp_eq_f32 vcc, 0.00000, v0 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalFalse_0_beta_116
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
s_branch RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalBottom_0_beta_116
RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalFalse_0_beta_116:


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
RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalBottom_0_beta_116:








































































s_endpgm  // End of RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
.LRR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__end:
.size RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_, .LRR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__end-RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 456
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 114688 // lds bytes
  .amdhsa_accum_offset 200
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
    DataType: B8
    DataTypeA: B8
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
  - .name: RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 114688
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 200
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF6, pointerType: PointerGlobal}
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

