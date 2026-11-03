// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT



.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 13705) 
s_mul_hi_i32 s1, s16, s10 // (op 13844) 
v_and_b32 v0, 4294967168, v1 // (op 13766) 
v_and_b32 v3, 15, v1 // (op 13766) 
v_add_lshl_u32 v4, v0, v3, 7 // (op 13766) 


v_and_b32 v0, 48, v1 // (op 13766) 
v_add_u32 v3, v4, v0 // (op 13766) 


v_lshrrev_b32 v0, 6, v1 // (op 13823) 
v_and_b32 v4, 1, v0 // (op 13823) 

v_and_b32 v0, 15, v1 // (op 13823) 
v_lshl_add_u32 v5, v4, 6, v0 // (op 13823) 


v_and_b32 v0, 63, v1 // (op 13823) 
v_lshrrev_b32 v4, 4, v0 // (op 13823) 

v_lshlrev_b32 v0, 5, v4 // (op 13823) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 13823) 


v_lshlrev_b32 v0, 2, v4 // (op 13823) 

v_lshrrev_b32 v4, 3, v0 // (op 13823) 

v_and_b32 v0, 4294967168, v1 // (op 13890) 
v_and_b32 v5, 15, v1 // (op 13890) 
v_add_lshl_u32 v6, v0, v5, 7 // (op 13890) 


v_and_b32 v0, 48, v1 // (op 13890) 
v_add_u32 v5, v6, v0 // (op 13890) 


v_lshrrev_b32 v0, 6, v1 // (op 13934) 
v_and_b32 v6, 1, v0 // (op 13934) 

v_and_b32 v0, 15, v1 // (op 13934) 
v_lshl_add_u32 v7, v6, 6, v0 // (op 13934) 


v_and_b32 v0, 63, v1 // (op 13934) 
v_lshrrev_b32 v6, 4, v0 // (op 13934) 

v_lshlrev_b32 v0, 5, v6 // (op 13934) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 13934) 


v_lshlrev_b32 v0, 2, v6 // (op 13934) 

v_lshrrev_b32 v6, 3, v0 // (op 13934) 

v_and_b32 v0, 4294967168, v1 // (op 13705) 
s_add_i32 s18, s0, s16 // (op 13705) 

v_and_b32 v7, 63, v1 // (op 13705) 
s_ashr_i32 s0, s18, 31 // (op 13705) 
v_lshrrev_b32 v8, 4, v7 // (op 13705) 
s_cmp_eq_i32 s10, 0 // (op 13705) 
s_mov_b32 s19, scc // (op 13705) 
v_lshlrev_b32 v9, 2, v8 // (op 13705) 

s_cmp_eq_i32 s19, 0 // (op 13705) 
s_cselect_b32 s49, 0, -1 // (op 13705) 

v_add_u32 v8, v0, v9 // (op 13705) 


s_and_b32 s19, s13, 31 // (op 13705) 
v_lshrrev_b32 v0, 2, v7 // (op 13705) 

v_and_b32 v7, 3, v0 // (op 13705) 

v_and_b32 v0, 4294967168, v1 // (op 13844) 
v_and_b32 v9, 63, v1 // (op 13844) 
s_lshl_b32 s50, 1, s19 // (op 13705) 
v_and_b32 v10, 3, v1 // (op 13705) 
s_add_i32 s51, s50, s49 // (op 13705) 


s_and_b32 s49, s0, s51 // (op 13705) 


s_add_i32 s0, s1, s16 // (op 13844) 

s_ashr_i32 s1, s0, 31 // (op 13844) 
v_lshl_add_u32 v11, v7, 2, v10 // (op 13705) 


s_add_i32 s50, s18, s49 // (op 13705) 


v_lshrrev_b32 v7, 6, v1 // (op 13705) 
s_ashr_i32 s18, s50, s19 // (op 13705) 


v_and_b32 v10, 1, v7 // (op 13705) 

s_lshl_b32 s19, s13, 24 // (op 13705) 
v_lshl_add_u32 v7, v10, 6, v11 // (op 13705) 


s_ashr_i32 s49, s19, 31 // (op 13705) 

s_xor_b32 s19, s18, s49 // (op 13705) 

s_sub_i32 s18, s19, s49 // (op 13705) 


s_mul_hi_i32 s19, s18, s14 // (op 13705) 
s_add_i32 s49, s19, s18 // (op 13705) 

s_ashr_i32 s19, s49, 31 // (op 13705) 
s_cmp_eq_i32 s14, 0 // (op 13705) 
s_mov_b32 s50, scc // (op 13705) 
s_cmp_eq_i32 s50, 0 // (op 13705) 
s_cselect_b32 s51, 0, -1 // (op 13705) 

s_and_b32 s50, s15, 31 // (op 13705) 
v_lshrrev_b32 v10, 4, v9 // (op 13844) 
s_cmp_eq_i32 s10, 0 // (op 13844) 
s_mov_b32 s52, scc // (op 13844) 
v_lshlrev_b32 v11, 2, v10 // (op 13844) 

s_cmp_eq_i32 s52, 0 // (op 13844) 
s_cselect_b32 s53, 0, -1 // (op 13844) 

s_lshl_b32 s52, 1, s50 // (op 13705) 
v_add_u32 v10, v0, v11 // (op 13844) 


v_lshrrev_b32 v0, 2, v9 // (op 13844) 

s_add_i32 s54, s52, s51 // (op 13705) 


s_and_b32 s51, s19, s54 // (op 13705) 


s_add_i32 s19, s49, s51 // (op 13705) 


s_ashr_i32 s49, s19, s50 // (op 13705) 


s_lshl_b32 s19, s15, 24 // (op 13705) 
s_ashr_i32 s50, s19, 31 // (op 13705) 

s_xor_b32 s19, s49, s50 // (op 13705) 

s_sub_i32 s49, s19, s50 // (op 13705) 


s_mul_i32 s19, s18, s11 // (op 13705) 
s_sub_u32 s50, s16, s19 // (op 13705) 

s_mul_i32 s19, s49, s11 // (op 13705) 
s_add_u32 s19, s19, s50 // (op 13705) 

s_mul_hi_i32 s50, s16, s9 // (op 13705) 
s_add_i32 s51, s50, s16 // (op 13705) 

s_ashr_i32 s50, s51, 31 // (op 13705) 
s_cmp_eq_i32 s9, 0 // (op 13705) 
s_mov_b32 s52, scc // (op 13705) 
s_cmp_eq_i32 s52, 0 // (op 13705) 
s_cselect_b32 s54, 0, -1 // (op 13705) 

s_and_b32 s52, s13, 31 // (op 13844) 
v_and_b32 v9, 3, v0 // (op 13844) 

s_lshl_b32 s55, 1, s52 // (op 13844) 
v_and_b32 v0, 3, v1 // (op 13844) 
s_add_i32 s56, s55, s53 // (op 13844) 


s_and_b32 s53, s1, s56 // (op 13844) 


v_lshl_add_u32 v11, v9, 2, v0 // (op 13844) 


s_add_i32 s1, s0, s53 // (op 13844) 


v_lshrrev_b32 v0, 6, v1 // (op 13844) 
s_ashr_i32 s0, s1, s52 // (op 13844) 


v_and_b32 v9, 1, v0 // (op 13844) 

s_lshl_b32 s1, s13, 24 // (op 13844) 
v_lshl_add_u32 v0, v9, 6, v11 // (op 13844) 


s_ashr_i32 s52, s1, 31 // (op 13844) 

s_xor_b32 s1, s0, s52 // (op 13844) 

s_sub_i32 s0, s1, s52 // (op 13844) 


s_mul_hi_i32 s1, s0, s14 // (op 13844) 
s_add_i32 s52, s1, s0 // (op 13844) 

s_ashr_i32 s1, s52, 31 // (op 13844) 
s_cmp_eq_i32 s14, 0 // (op 13844) 
s_mov_b32 s53, scc // (op 13844) 
s_cmp_eq_i32 s53, 0 // (op 13844) 
s_cselect_b32 s55, 0, -1 // (op 13844) 

s_and_b32 s53, s15, 31 // (op 13844) 
s_lshl_b32 s56, 1, s53 // (op 13844) 
s_add_i32 s57, s56, s55 // (op 13844) 


s_and_b32 s55, s1, s57 // (op 13844) 


s_add_i32 s1, s52, s55 // (op 13844) 


s_ashr_i32 s52, s1, s53 // (op 13844) 


s_lshl_b32 s1, s15, 24 // (op 13844) 
s_ashr_i32 s53, s1, 31 // (op 13844) 

s_xor_b32 s1, s52, s53 // (op 13844) 

s_sub_i32 s52, s1, s53 // (op 13844) 


s_mul_i32 s1, s0, s11 // (op 13844) 
s_sub_u32 s53, s16, s1 // (op 13844) 

s_mul_i32 s1, s52, s11 // (op 13844) 
s_add_u32 s1, s1, s53 // (op 13844) 

s_mul_hi_i32 s53, s16, s9 // (op 13844) 
s_add_i32 s55, s53, s16 // (op 13844) 

s_ashr_i32 s53, s55, 31 // (op 13844) 
s_cmp_eq_i32 s9, 0 // (op 13844) 
s_mov_b32 s56, scc // (op 13844) 
s_cmp_eq_i32 s56, 0 // (op 13844) 
s_cselect_b32 s57, 0, -1 // (op 13844) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 13705) 
s_lshl_b32 s58, 1, s56 // (op 13705) 
s_add_i32 s59, s58, s54 // (op 13705) 


s_and_b32 s54, s50, s59 // (op 13705) 


s_add_i32 s50, s51, s54 // (op 13705) 


s_ashr_i32 s51, s50, s56 // (op 13705) 


s_lshl_b32 s50, s45, 24 // (op 13705) 
s_ashr_i32 s54, s50, 31 // (op 13705) 

s_xor_b32 s50, s51, s54 // (op 13705) 

s_sub_i32 s51, s50, s54 // (op 13705) 


s_and_b32 s50, s45, 31 // (op 13844) 
s_lshl_b32 s54, 1, s50 // (op 13844) 
s_add_i32 s56, s54, s57 // (op 13844) 


s_and_b32 s54, s53, s56 // (op 13844) 


s_add_i32 s53, s55, s54 // (op 13844) 


s_ashr_i32 s54, s53, s50 // (op 13844) 


s_lshl_b32 s50, s45, 24 // (op 13844) 
s_ashr_i32 s53, s50, 31 // (op 13844) 

s_xor_b32 s50, s54, s53 // (op 13844) 

s_sub_i32 s54, s50, s53 // (op 13844) 


s_add_u32 s56, s6, 128 // (op 13705) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 13705) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 13705) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 13705) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 13705) 

s_mul_i32 s50, s11, s56 // (op 13705) 
s_mul_i32 s53, s8, s50 // (op 13705) 

s_mul_i32 s50, s51, s53 // (op 13705) 

s_sub_u32 s53, s16, s50 // (op 13705) 

s_mov_b32 s50, s53 // (op 13705) 
s_mul_hi_i32 s55, s50, s12 // (op 13705) 
s_add_i32 s58, s55, s50 // (op 13705) 


s_ashr_i32 s50, s58, 31 // (op 13705) 
s_cmp_eq_i32 s12, 0 // (op 13705) 
s_mov_b32 s55, scc // (op 13705) 
s_cmp_eq_i32 s55, 0 // (op 13705) 
s_cselect_b32 s59, 0, -1 // (op 13705) 

s_and_b32 s55, s44, 31 // (op 13705) 
s_lshl_b32 s60, 1, s55 // (op 13705) 
s_add_i32 s61, s60, s59 // (op 13705) 


s_and_b32 s59, s50, s61 // (op 13705) 


s_add_i32 s50, s58, s59 // (op 13705) 


s_ashr_i32 s58, s50, s55 // (op 13705) 


s_lshl_b32 s50, s44, 24 // (op 13705) 
s_ashr_i32 s55, s50, 31 // (op 13705) 

s_xor_b32 s50, s58, s55 // (op 13705) 

s_sub_i32 s58, s50, s55 // (op 13705) 


s_mul_i32 s50, s46, s11 // (op 13705) 
s_add_u32 s60, s20, 256 // (op 13705) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 13705) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 13705) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 13705) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 8 // (op 13705) 

s_sub_i32 s55, s60, s50 // (op 13705) 

s_mul_i32 s59, s58, s55 // (op 13705) 

s_sub_u32 s55, s53, s59 // (op 13705) 


s_add_u32 s53, s55, s50 // (op 13705) 


s_cmp_eq_i32 s51, 0 // (op 13705) 
s_mov_b32 s50, scc // (op 13705) 

s_cmp_eq_i32 s50, 0 // (op 13705) 
s_cselect_b32 s51, s53, s19 // (op 13705) 


s_mul_i32 s19, s49, s56 // (op 13705) 


v_lshl_add_u32 v9, s51, 8, v8 // (op 13705) 


s_sub_i32 s49, s18, s19 // (op 13705) 


s_cmp_eq_i32 s50, 0 // (op 13705) 
s_cselect_b32 s51, s58, s49 // (op 13705) 



v_lshl_add_u32 v8, s51, 7, v7 // (op 13705) 


v_mul_lo_u32 v7, v8, s2 // (op 13705) 

v_add_lshl_u32 v8, v9, v7, 1 // (op 13705) 


s_add_u32 s18, s6, 128 // (op 13844) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 13844) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 13844) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 13844) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 7 // (op 13844) 

s_mul_i32 s53, s11, s18 // (op 13844) 
s_mul_i32 s49, s8, s53 // (op 13844) 

s_mul_i32 s53, s54, s49 // (op 13844) 

s_sub_u32 s49, s16, s53 // (op 13844) 

s_mov_b32 s53, s49 // (op 13844) 
s_mul_hi_i32 s50, s53, s12 // (op 13844) 
s_add_i32 s51, s50, s53 // (op 13844) 


s_ashr_i32 s50, s51, 31 // (op 13844) 
s_cmp_eq_i32 s12, 0 // (op 13844) 
s_mov_b32 s53, scc // (op 13844) 
s_cmp_eq_i32 s53, 0 // (op 13844) 
s_cselect_b32 s55, 0, -1 // (op 13844) 

s_and_b32 s53, s44, 31 // (op 13844) 
s_lshl_b32 s56, 1, s53 // (op 13844) 
s_add_i32 s57, s56, s55 // (op 13844) 


s_and_b32 s55, s50, s57 // (op 13844) 


s_add_i32 s50, s51, s55 // (op 13844) 


s_ashr_i32 s51, s50, s53 // (op 13844) 


s_lshl_b32 s50, s44, 24 // (op 13844) 
s_ashr_i32 s53, s50, 31 // (op 13844) 

s_xor_b32 s50, s51, s53 // (op 13844) 

s_sub_i32 s51, s50, s53 // (op 13844) 


s_mul_i32 s50, s46, s11 // (op 13844) 
s_add_u32 s56, s20, 256 // (op 13844) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 13844) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 13844) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 13844) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 13844) 

s_sub_i32 s53, s56, s50 // (op 13844) 

s_mul_i32 s55, s51, s53 // (op 13844) 

s_sub_u32 s53, s49, s55 // (op 13844) 


s_add_u32 s49, s53, s50 // (op 13844) 


s_cmp_eq_i32 s54, 0 // (op 13844) 
s_mov_b32 s50, scc // (op 13844) 

s_cmp_eq_i32 s50, 0 // (op 13844) 
s_cselect_b32 s53, s49, s1 // (op 13844) 


s_mul_i32 s1, s52, s18 // (op 13844) 


v_lshl_add_u32 v7, s53, 8, v10 // (op 13844) 


s_sub_i32 s49, s0, s1 // (op 13844) 


s_cmp_eq_i32 s50, 0 // (op 13844) 
s_cselect_b32 s0, s51, s49 // (op 13844) 



v_lshl_add_u32 v9, s0, 7, v0 // (op 13844) 


v_mul_lo_u32 v0, v9, s4 // (op 13844) 

v_add_lshl_u32 v9, v7, v0, 1 // (op 13844) 


s_mul_i32 s1, s2, s7 // (op 13707) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 13707) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 13707) least significant
s_mul_hi_u32 s19, s2, s6 // (op 13707) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13707) most significant: sum
s_add_u32 s1, s1, s19 // (op 13707) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 13707) 

s_mul_i32 s1, s4, s7 // (op 13846) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 13846) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 13846) least significant
s_mul_hi_u32 s50, s4, s6 // (op 13846) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 13846) most significant: sum
s_add_u32 s1, s1, s50 // (op 13846) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 13846) 

s_mov_b32 s52, s22 // (op 13707) reg expression
s_mov_b32 s53, s23 // (op 13707) reg expression
s_mov_b32 s54, s18 // (op 13707) reg expression
s_mov_b32 s55, 131072 // (op 13707) call()

s_mov_b32 s56, s24 // (op 13846) reg expression

s_mov_b32 s57, s25 // (op 13846) reg expression
s_mov_b32 s58, s50 // (op 13846) reg expression
s_mov_b32 s59, 131072 // (op 13846) call()


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
v_accvgpr_write a16, 0.00000 // (op 777) call()
v_accvgpr_write a17, 0.00000 // (op 777) call()
v_accvgpr_write a18, 0.00000 // (op 777) call()
v_accvgpr_write a19, 0.00000 // (op 777) call()
v_accvgpr_write a20, 0.00000 // (op 1038) call()
v_accvgpr_write a21, 0.00000 // (op 1038) call()
v_accvgpr_write a22, 0.00000 // (op 1038) call()
v_accvgpr_write a23, 0.00000 // (op 1038) call()
v_accvgpr_write a24, 0.00000 // (op 1039) call()
v_accvgpr_write a25, 0.00000 // (op 1039) call()
v_accvgpr_write a26, 0.00000 // (op 1039) call()
v_accvgpr_write a27, 0.00000 // (op 1039) call()
v_accvgpr_write a28, 0.00000 // (op 1040) call()
v_accvgpr_write a29, 0.00000 // (op 1040) call()
v_accvgpr_write a30, 0.00000 // (op 1040) call()
v_accvgpr_write a31, 0.00000 // (op 1040) call()
v_accvgpr_write a32, 0.00000 // (op 1387) call()
v_accvgpr_write a33, 0.00000 // (op 1387) call()
v_accvgpr_write a34, 0.00000 // (op 1387) call()
v_accvgpr_write a35, 0.00000 // (op 1387) call()
v_accvgpr_write a36, 0.00000 // (op 1648) call()
v_accvgpr_write a37, 0.00000 // (op 1648) call()
v_accvgpr_write a38, 0.00000 // (op 1648) call()
v_accvgpr_write a39, 0.00000 // (op 1648) call()
v_accvgpr_write a40, 0.00000 // (op 1649) call()
v_accvgpr_write a41, 0.00000 // (op 1649) call()
v_accvgpr_write a42, 0.00000 // (op 1649) call()
v_accvgpr_write a43, 0.00000 // (op 1649) call()
v_accvgpr_write a44, 0.00000 // (op 1650) call()
v_accvgpr_write a45, 0.00000 // (op 1650) call()
v_accvgpr_write a46, 0.00000 // (op 1650) call()
v_accvgpr_write a47, 0.00000 // (op 1650) call()
v_accvgpr_write a48, 0.00000 // (op 1997) call()
v_accvgpr_write a49, 0.00000 // (op 1997) call()
v_accvgpr_write a50, 0.00000 // (op 1997) call()
v_accvgpr_write a51, 0.00000 // (op 1997) call()
v_accvgpr_write a52, 0.00000 // (op 2258) call()
v_accvgpr_write a53, 0.00000 // (op 2258) call()
v_accvgpr_write a54, 0.00000 // (op 2258) call()
v_accvgpr_write a55, 0.00000 // (op 2258) call()
v_accvgpr_write a56, 0.00000 // (op 2259) call()
v_accvgpr_write a57, 0.00000 // (op 2259) call()
v_accvgpr_write a58, 0.00000 // (op 2259) call()
v_accvgpr_write a59, 0.00000 // (op 2259) call()
v_accvgpr_write a60, 0.00000 // (op 2260) call()
v_accvgpr_write a61, 0.00000 // (op 2260) call()
v_accvgpr_write a62, 0.00000 // (op 2260) call()
v_accvgpr_write a63, 0.00000 // (op 2260) call()
v_accvgpr_write a64, 0.00000 // (op 2607) call()
v_accvgpr_write a65, 0.00000 // (op 2607) call()
v_accvgpr_write a66, 0.00000 // (op 2607) call()
v_accvgpr_write a67, 0.00000 // (op 2607) call()
v_accvgpr_write a68, 0.00000 // (op 2868) call()
v_accvgpr_write a69, 0.00000 // (op 2868) call()
v_accvgpr_write a70, 0.00000 // (op 2868) call()
v_accvgpr_write a71, 0.00000 // (op 2868) call()
v_accvgpr_write a72, 0.00000 // (op 2869) call()
v_accvgpr_write a73, 0.00000 // (op 2869) call()
v_accvgpr_write a74, 0.00000 // (op 2869) call()
v_accvgpr_write a75, 0.00000 // (op 2869) call()
v_accvgpr_write a76, 0.00000 // (op 2870) call()
v_accvgpr_write a77, 0.00000 // (op 2870) call()
v_accvgpr_write a78, 0.00000 // (op 2870) call()
v_accvgpr_write a79, 0.00000 // (op 2870) call()
v_accvgpr_write a80, 0.00000 // (op 3217) call()
v_accvgpr_write a81, 0.00000 // (op 3217) call()
v_accvgpr_write a82, 0.00000 // (op 3217) call()
v_accvgpr_write a83, 0.00000 // (op 3217) call()
v_accvgpr_write a84, 0.00000 // (op 3478) call()
v_accvgpr_write a85, 0.00000 // (op 3478) call()
v_accvgpr_write a86, 0.00000 // (op 3478) call()
v_accvgpr_write a87, 0.00000 // (op 3478) call()
v_accvgpr_write a88, 0.00000 // (op 3479) call()
v_accvgpr_write a89, 0.00000 // (op 3479) call()
v_accvgpr_write a90, 0.00000 // (op 3479) call()
v_accvgpr_write a91, 0.00000 // (op 3479) call()
v_accvgpr_write a92, 0.00000 // (op 3480) call()
v_accvgpr_write a93, 0.00000 // (op 3480) call()
v_accvgpr_write a94, 0.00000 // (op 3480) call()
v_accvgpr_write a95, 0.00000 // (op 3480) call()
v_accvgpr_write a96, 0.00000 // (op 3827) call()
v_accvgpr_write a97, 0.00000 // (op 3827) call()
v_accvgpr_write a98, 0.00000 // (op 3827) call()
v_accvgpr_write a99, 0.00000 // (op 3827) call()
v_accvgpr_write a100, 0.00000 // (op 4088) call()
v_accvgpr_write a101, 0.00000 // (op 4088) call()
v_accvgpr_write a102, 0.00000 // (op 4088) call()
v_accvgpr_write a103, 0.00000 // (op 4088) call()
v_accvgpr_write a104, 0.00000 // (op 4089) call()
v_accvgpr_write a105, 0.00000 // (op 4089) call()
v_accvgpr_write a106, 0.00000 // (op 4089) call()
v_accvgpr_write a107, 0.00000 // (op 4089) call()
v_accvgpr_write a108, 0.00000 // (op 4090) call()
v_accvgpr_write a109, 0.00000 // (op 4090) call()
v_accvgpr_write a110, 0.00000 // (op 4090) call()
v_accvgpr_write a111, 0.00000 // (op 4090) call()
v_accvgpr_write a112, 0.00000 // (op 4437) call()
v_accvgpr_write a113, 0.00000 // (op 4437) call()
v_accvgpr_write a114, 0.00000 // (op 4437) call()
v_accvgpr_write a115, 0.00000 // (op 4437) call()
v_accvgpr_write a116, 0.00000 // (op 4698) call()
v_accvgpr_write a117, 0.00000 // (op 4698) call()
v_accvgpr_write a118, 0.00000 // (op 4698) call()
v_accvgpr_write a119, 0.00000 // (op 4698) call()
v_accvgpr_write a120, 0.00000 // (op 4699) call()
v_accvgpr_write a121, 0.00000 // (op 4699) call()
v_accvgpr_write a122, 0.00000 // (op 4699) call()
v_accvgpr_write a123, 0.00000 // (op 4699) call()
v_accvgpr_write a124, 0.00000 // (op 4700) call()
v_accvgpr_write a125, 0.00000 // (op 4700) call()
v_accvgpr_write a126, 0.00000 // (op 4700) call()
v_accvgpr_write a127, 0.00000 // (op 4700) call()
s_mul_hi_i32 s0, s16, s10 // (op 13595) 
v_lshrrev_b32 v10, 3, v1 // (op 13595) 
s_add_i32 s1, s0, s16 // (op 13595) 

v_and_b32 v11, -8, v1 // (op 13595) 
s_ashr_i32 s0, s1, 31 // (op 13595) 
v_sub_u32 v12, v1, v11 // (op 13595) 

s_cmp_eq_i32 s10, 0 // (op 13595) 
s_mov_b32 s18, scc // (op 13595) 
v_lshlrev_b32 v11, 4, v12 // (op 13595) 

s_cmp_eq_i32 s18, 0 // (op 13595) 
s_cselect_b32 s19, 0, -1 // (op 13595) 

v_mov_b32 v13, 0 // (op 13595) convert
v_mov_b32 v12, v11 // (op 13595) convert

s_and_b32 s18, s13, 31 // (op 13595) 
s_lshl_b32 s22, 1, s18 // (op 13595) 
s_add_i32 s23, s22, s19 // (op 13595) 


s_and_b32 s19, s0, s23 // (op 13595) 


s_add_i32 s0, s1, s19 // (op 13595) 


s_ashr_i32 s1, s0, s18 // (op 13595) 


s_lshl_b32 s0, s13, 24 // (op 13595) 
s_ashr_i32 s18, s0, 31 // (op 13595) 

s_xor_b32 s0, s1, s18 // (op 13595) 

s_sub_i32 s1, s0, s18 // (op 13595) 


s_mul_hi_i32 s0, s1, s14 // (op 13595) 
s_add_i32 s18, s0, s1 // (op 13595) 

s_ashr_i32 s0, s18, 31 // (op 13595) 
s_cmp_eq_i32 s14, 0 // (op 13595) 
s_mov_b32 s19, scc // (op 13595) 
s_cmp_eq_i32 s19, 0 // (op 13595) 
s_cselect_b32 s22, 0, -1 // (op 13595) 

s_and_b32 s19, s15, 31 // (op 13595) 
s_lshl_b32 s23, 1, s19 // (op 13595) 
s_add_i32 s24, s23, s22 // (op 13595) 


s_and_b32 s25, s0, s24 // (op 13595) 


s_add_i32 s0, s18, s25 // (op 13595) 


s_ashr_i32 s18, s0, s19 // (op 13595) 


s_lshl_b32 s0, s15, 24 // (op 13595) 
s_ashr_i32 s19, s0, 31 // (op 13595) 

s_xor_b32 s0, s18, s19 // (op 13595) 

s_sub_i32 s18, s0, s19 // (op 13595) 


s_mul_i32 s0, s1, s11 // (op 13595) 

s_sub_u32 s1, s16, s0 // (op 13595) 

s_mul_i32 s0, s18, s11 // (op 13595) 
s_add_u32 s0, s0, s1 // (op 13595) 


s_mul_hi_i32 s1, s16, s9 // (op 13595) 
s_add_i32 s18, s1, s16 // (op 13595) 

s_ashr_i32 s1, s18, 31 // (op 13595) 
s_cmp_eq_i32 s9, 0 // (op 13595) 
s_mov_b32 s19, scc // (op 13595) 
s_cmp_eq_i32 s19, 0 // (op 13595) 
s_cselect_b32 s22, 0, -1 // (op 13595) 

s_and_b32 s19, s45, 31 // (op 13595) 
s_lshl_b32 s23, 1, s19 // (op 13595) 
s_add_i32 s24, s23, s22 // (op 13595) 


s_and_b32 s25, s1, s24 // (op 13595) 


s_add_i32 s1, s18, s25 // (op 13595) 


s_ashr_i32 s18, s1, s19 // (op 13595) 


s_lshl_b32 s1, s45, 24 // (op 13595) 
s_ashr_i32 s19, s1, 31 // (op 13595) 

s_xor_b32 s1, s18, s19 // (op 13595) 

s_sub_i32 s18, s1, s19 // (op 13595) 


s_add_u32 s22, s6, 128 // (op 13595) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13595) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13595) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13595) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13595) 

s_mul_i32 s1, s11, s22 // (op 13595) 

s_mul_i32 s19, s8, s1 // (op 13595) 

s_mul_i32 s1, s18, s19 // (op 13595) 

s_sub_u32 s19, s16, s1 // (op 13595) 

s_mov_b32 s1, s19 // (op 13595) 
s_mul_hi_i32 s22, s1, s12 // (op 13595) 
s_add_i32 s23, s22, s1 // (op 13595) 


s_ashr_i32 s1, s23, 31 // (op 13595) 
s_cmp_eq_i32 s12, 0 // (op 13595) 
s_mov_b32 s22, scc // (op 13595) 
s_cmp_eq_i32 s22, 0 // (op 13595) 
s_cselect_b32 s24, 0, -1 // (op 13595) 

s_and_b32 s22, s44, 31 // (op 13595) 
s_lshl_b32 s25, 1, s22 // (op 13595) 
s_add_i32 s47, s25, s24 // (op 13595) 


s_and_b32 s24, s1, s47 // (op 13595) 


s_add_i32 s1, s23, s24 // (op 13595) 


s_ashr_i32 s23, s1, s22 // (op 13595) 


s_lshl_b32 s1, s44, 24 // (op 13595) 
s_ashr_i32 s22, s1, 31 // (op 13595) 

s_xor_b32 s1, s23, s22 // (op 13595) 

s_sub_i32 s23, s1, s22 // (op 13595) 


s_mul_i32 s1, s46, s11 // (op 13595) 
s_add_u32 s24, s20, 256 // (op 13595) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13595) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 13595) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 13595) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 13595) 

s_sub_i32 s22, s24, s1 // (op 13595) 

s_mul_i32 s24, s23, s22 // (op 13595) 


s_sub_u32 s25, s19, s24 // (op 13595) 


s_add_u32 s19, s25, s1 // (op 13595) 


s_cmp_eq_i32 s18, 0 // (op 13595) 
s_mov_b32 s1, scc // (op 13595) 

s_cmp_eq_i32 s1, 0 // (op 13595) 
s_cselect_b32 s18, s19, s0 // (op 13595) 



v_lshl_add_u32 v11, s18, 8, v10 // (op 13595) 


v_mul_lo_u32 v15, v11, s27 // (op 13595) most significant: low of low * high
v_mul_hi_u32 v10, v11, s26 // (op 13595) most significant: high of low * low
v_mul_lo_u32 v14, v11, s26 // (op 13595) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 13595) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 13595) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 13595) most significant half



s_mul_i32 s1, s26, s21 // (op 13597) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 13597) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 13597) least significant
s_mul_hi_u32 s19, s26, s20 // (op 13597) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13597) most significant: sum
s_add_u32 s1, s1, s19 // (op 13597) most significant: sum


s_mov_b32 s48, s28 // (op 13597) reg expression
s_mov_b32 s49, s29 // (op 13597) reg expression
s_mov_b32 s50, s0 // (op 13597) reg expression
s_mov_b32 s51, 131072 // (op 13597) call()

s_mul_hi_i32 s47, s16, s10 // (op 13614) 
v_lshrrev_b32 v10, 3, v1 // (op 13614) 
s_add_i32 s0, s47, s16 // (op 13614) 

v_and_b32 v11, -8, v1 // (op 13614) 
s_ashr_i32 s1, s0, 31 // (op 13614) 
v_sub_u32 v12, v1, v11 // (op 13614) 

s_cmp_eq_i32 s10, 0 // (op 13614) 
s_mov_b32 s47, scc // (op 13614) 
v_lshlrev_b32 v11, 4, v12 // (op 13614) 

s_cmp_eq_i32 s47, 0 // (op 13614) 
s_cselect_b32 s18, 0, -1 // (op 13614) 

v_mov_b32 v13, 0 // (op 13614) convert
v_mov_b32 v12, v11 // (op 13614) convert

s_and_b32 s19, s13, 31 // (op 13614) 
s_lshl_b32 s47, 1, s19 // (op 13614) 
s_add_i32 s22, s47, s18 // (op 13614) 


s_and_b32 s18, s1, s22 // (op 13614) 


s_add_i32 s1, s0, s18 // (op 13614) 


s_ashr_i32 s0, s1, s19 // (op 13614) 


s_lshl_b32 s1, s13, 24 // (op 13614) 
s_ashr_i32 s47, s1, 31 // (op 13614) 

s_xor_b32 s1, s0, s47 // (op 13614) 

s_sub_i32 s0, s1, s47 // (op 13614) 


s_mul_hi_i32 s1, s0, s14 // (op 13614) 
s_add_i32 s47, s1, s0 // (op 13614) 

s_ashr_i32 s1, s47, 31 // (op 13614) 
s_cmp_eq_i32 s14, 0 // (op 13614) 
s_mov_b32 s18, scc // (op 13614) 
s_cmp_eq_i32 s18, 0 // (op 13614) 
s_cselect_b32 s19, 0, -1 // (op 13614) 

s_and_b32 s18, s15, 31 // (op 13614) 
s_lshl_b32 s22, 1, s18 // (op 13614) 
s_add_i32 s23, s22, s19 // (op 13614) 


s_and_b32 s19, s1, s23 // (op 13614) 


s_add_i32 s1, s47, s19 // (op 13614) 


s_ashr_i32 s19, s1, s18 // (op 13614) 


s_lshl_b32 s1, s15, 24 // (op 13614) 
s_ashr_i32 s18, s1, 31 // (op 13614) 

s_xor_b32 s1, s19, s18 // (op 13614) 

s_sub_i32 s19, s1, s18 // (op 13614) 


s_mul_i32 s1, s0, s11 // (op 13614) 

s_sub_u32 s0, s16, s1 // (op 13614) 

s_mul_i32 s1, s19, s11 // (op 13614) 
s_add_u32 s1, s1, s0 // (op 13614) 


s_mul_hi_i32 s0, s16, s9 // (op 13614) 
s_add_i32 s47, s0, s16 // (op 13614) 

s_ashr_i32 s0, s47, 31 // (op 13614) 
s_cmp_eq_i32 s9, 0 // (op 13614) 
s_mov_b32 s18, scc // (op 13614) 
s_cmp_eq_i32 s18, 0 // (op 13614) 
s_cselect_b32 s19, 0, -1 // (op 13614) 

s_and_b32 s18, s45, 31 // (op 13614) 
s_lshl_b32 s22, 1, s18 // (op 13614) 
s_add_i32 s23, s22, s19 // (op 13614) 


s_and_b32 s19, s0, s23 // (op 13614) 


s_add_i32 s0, s47, s19 // (op 13614) 


s_ashr_i32 s19, s0, s18 // (op 13614) 


s_lshl_b32 s0, s45, 24 // (op 13614) 
s_ashr_i32 s18, s0, 31 // (op 13614) 

s_xor_b32 s0, s19, s18 // (op 13614) 

s_sub_i32 s19, s0, s18 // (op 13614) 


s_add_u32 s22, s6, 128 // (op 13614) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13614) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13614) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13614) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13614) 

s_mul_i32 s0, s11, s22 // (op 13614) 

s_mul_i32 s18, s8, s0 // (op 13614) 

s_mul_i32 s0, s19, s18 // (op 13614) 

s_sub_u32 s18, s16, s0 // (op 13614) 

s_mov_b32 s0, s18 // (op 13614) 
s_mul_hi_i32 s47, s0, s12 // (op 13614) 
s_add_i32 s22, s47, s0 // (op 13614) 


s_ashr_i32 s0, s22, 31 // (op 13614) 
s_cmp_eq_i32 s12, 0 // (op 13614) 
s_mov_b32 s47, scc // (op 13614) 
s_cmp_eq_i32 s47, 0 // (op 13614) 
s_cselect_b32 s23, 0, -1 // (op 13614) 

s_and_b32 s47, s44, 31 // (op 13614) 
s_lshl_b32 s24, 1, s47 // (op 13614) 
s_add_i32 s25, s24, s23 // (op 13614) 


s_and_b32 s23, s0, s25 // (op 13614) 


s_add_i32 s0, s22, s23 // (op 13614) 


s_ashr_i32 s22, s0, s47 // (op 13614) 


s_lshl_b32 s0, s44, 24 // (op 13614) 
s_ashr_i32 s47, s0, 31 // (op 13614) 

s_xor_b32 s0, s22, s47 // (op 13614) 

s_sub_i32 s22, s0, s47 // (op 13614) 


s_mul_i32 s0, s46, s11 // (op 13614) 
s_add_u32 s24, s20, 256 // (op 13614) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13614) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 13614) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 13614) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 8 // (op 13614) 

s_sub_i32 s23, s24, s0 // (op 13614) 

s_mul_i32 s47, s22, s23 // (op 13614) 


s_sub_u32 s22, s18, s47 // (op 13614) 


s_add_u32 s18, s22, s0 // (op 13614) 


s_cmp_eq_i32 s19, 0 // (op 13614) 
s_mov_b32 s0, scc // (op 13614) 

s_cmp_eq_i32 s0, 0 // (op 13614) 
s_cselect_b32 s19, s18, s1 // (op 13614) 



v_lshl_add_u32 v11, s19, 8, v10 // (op 13614) 


v_mul_lo_u32 v17, v11, s27 // (op 13614) most significant: low of low * high
v_mul_hi_u32 v10, v11, s26 // (op 13614) most significant: high of low * low
v_mul_lo_u32 v16, v11, s26 // (op 13614) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 13614) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v12 // (op 13614) least significant half
v_addc_co_u32 v17, s[0:1], v17, v13, s[0:1] // (op 13614) most significant half



s_mul_i32 s1, s26, s21 // (op 13616) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 13616) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 13616) least significant
s_mul_hi_u32 s18, s26, s20 // (op 13616) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13616) most significant: sum
s_add_u32 s1, s1, s18 // (op 13616) most significant: sum


s_mov_b32 s48, s28 // (op 13616) reg expression
s_mov_b32 s49, s29 // (op 13616) reg expression
s_mov_b32 s50, s0 // (op 13616) reg expression
s_mov_b32 s51, 131072 // (op 13616) call()

s_mul_hi_i32 s47, s16, s10 // (op 13659) 
v_lshrrev_b32 v10, 2, v1 // (op 13659) 
s_add_i32 s0, s47, s16 // (op 13659) 

v_and_b32 v11, -4, v1 // (op 13659) 
s_ashr_i32 s1, s0, 31 // (op 13659) 
v_sub_u32 v12, v1, v11 // (op 13659) 

s_cmp_eq_i32 s10, 0 // (op 13659) 
s_mov_b32 s47, scc // (op 13659) 
v_lshlrev_b32 v11, 5, v12 // (op 13659) 

s_cmp_eq_i32 s47, 0 // (op 13659) 
s_cselect_b32 s18, 0, -1 // (op 13659) 

v_mov_b32 v13, 0 // (op 13659) convert
v_mov_b32 v12, v11 // (op 13659) convert

s_and_b32 s19, s13, 31 // (op 13659) 
s_lshl_b32 s47, 1, s19 // (op 13659) 
s_add_i32 s22, s47, s18 // (op 13659) 


s_and_b32 s18, s1, s22 // (op 13659) 


s_add_i32 s1, s0, s18 // (op 13659) 


s_ashr_i32 s0, s1, s19 // (op 13659) 


s_lshl_b32 s1, s13, 24 // (op 13659) 
s_ashr_i32 s47, s1, 31 // (op 13659) 

s_xor_b32 s1, s0, s47 // (op 13659) 

s_sub_i32 s0, s1, s47 // (op 13659) 


s_mul_hi_i32 s1, s0, s14 // (op 13659) 
s_add_i32 s47, s1, s0 // (op 13659) 

s_ashr_i32 s1, s47, 31 // (op 13659) 
s_cmp_eq_i32 s14, 0 // (op 13659) 
s_mov_b32 s18, scc // (op 13659) 
s_cmp_eq_i32 s18, 0 // (op 13659) 
s_cselect_b32 s19, 0, -1 // (op 13659) 

s_and_b32 s18, s15, 31 // (op 13659) 
s_lshl_b32 s22, 1, s18 // (op 13659) 
s_add_i32 s23, s22, s19 // (op 13659) 


s_and_b32 s19, s1, s23 // (op 13659) 


s_add_i32 s1, s47, s19 // (op 13659) 


s_ashr_i32 s19, s1, s18 // (op 13659) 


s_lshl_b32 s1, s15, 24 // (op 13659) 
s_ashr_i32 s18, s1, 31 // (op 13659) 

s_xor_b32 s1, s19, s18 // (op 13659) 

s_sub_i32 s19, s1, s18 // (op 13659) 


s_add_u32 s22, s6, 128 // (op 13659) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13659) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13659) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13659) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13659) 

s_mul_i32 s1, s19, s22 // (op 13659) 

s_sub_i32 s47, s0, s1 // (op 13659) 


s_mul_hi_i32 s0, s16, s9 // (op 13659) 
s_add_i32 s1, s0, s16 // (op 13659) 

s_ashr_i32 s0, s1, 31 // (op 13659) 
s_cmp_eq_i32 s9, 0 // (op 13659) 
s_mov_b32 s18, scc // (op 13659) 
s_cmp_eq_i32 s18, 0 // (op 13659) 
s_cselect_b32 s19, 0, -1 // (op 13659) 

s_and_b32 s18, s45, 31 // (op 13659) 
s_lshl_b32 s24, 1, s18 // (op 13659) 
s_add_i32 s25, s24, s19 // (op 13659) 


s_and_b32 s19, s0, s25 // (op 13659) 


s_add_i32 s0, s1, s19 // (op 13659) 


s_ashr_i32 s1, s0, s18 // (op 13659) 


s_lshl_b32 s0, s45, 24 // (op 13659) 
s_ashr_i32 s18, s0, 31 // (op 13659) 

s_xor_b32 s0, s1, s18 // (op 13659) 

s_sub_i32 s1, s0, s18 // (op 13659) 


s_mul_i32 s0, s11, s22 // (op 13659) 

s_mul_i32 s18, s8, s0 // (op 13659) 

s_mul_i32 s0, s1, s18 // (op 13659) 

s_sub_u32 s18, s16, s0 // (op 13659) 

s_mul_hi_i32 s0, s18, s12 // (op 13659) 
s_add_i32 s19, s0, s18 // (op 13659) 


s_ashr_i32 s0, s19, 31 // (op 13659) 
s_cmp_eq_i32 s12, 0 // (op 13659) 
s_mov_b32 s18, scc // (op 13659) 
s_cmp_eq_i32 s18, 0 // (op 13659) 
s_cselect_b32 s22, 0, -1 // (op 13659) 

s_and_b32 s18, s44, 31 // (op 13659) 
s_lshl_b32 s23, 1, s18 // (op 13659) 
s_add_i32 s24, s23, s22 // (op 13659) 


s_and_b32 s25, s0, s24 // (op 13659) 


s_add_i32 s0, s19, s25 // (op 13659) 


s_ashr_i32 s19, s0, s18 // (op 13659) 


s_lshl_b32 s0, s44, 24 // (op 13659) 
s_ashr_i32 s18, s0, 31 // (op 13659) 

s_xor_b32 s0, s19, s18 // (op 13659) 

s_sub_i32 s19, s0, s18 // (op 13659) 


s_cmp_eq_i32 s1, 0 // (op 13659) 
s_mov_b32 s0, scc // (op 13659) 

s_cmp_eq_i32 s0, 0 // (op 13659) 
s_cselect_b32 s1, s19, s47 // (op 13659) 



v_lshl_add_u32 v11, s1, 7, v10 // (op 13659) 


v_mul_lo_u32 v19, v11, s31 // (op 13659) most significant: low of low * high
v_mul_hi_u32 v10, v11, s30 // (op 13659) most significant: high of low * low
v_mul_lo_u32 v18, v11, s30 // (op 13659) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 13659) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v18 // (op 13659) least significant half
v_addc_co_u32 v11, s[0:1], v13, v19, s[0:1] // (op 13659) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 13659) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 13659) 

s_mul_i32 s1, s30, s7 // (op 13661) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 13661) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 13661) least significant
s_mul_hi_u32 s18, s30, s6 // (op 13661) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13661) most significant: sum
s_add_u32 s1, s1, s18 // (op 13661) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13661) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 13661) 

s_mov_b32 s60, s32 // (op 13661) reg expression
s_mov_b32 s61, s33 // (op 13661) reg expression
s_mov_b32 s62, s0 // (op 13661) reg expression
s_mov_b32 s63, 131072 // (op 13661) call()

s_mul_hi_i32 s47, s16, s10 // (op 13678) 
v_lshrrev_b32 v10, 2, v1 // (op 13678) 
s_add_i32 s0, s47, s16 // (op 13678) 

v_and_b32 v11, -4, v1 // (op 13678) 
s_ashr_i32 s1, s0, 31 // (op 13678) 
v_sub_u32 v18, v1, v11 // (op 13678) 

s_cmp_eq_i32 s10, 0 // (op 13678) 
s_mov_b32 s47, scc // (op 13678) 
v_lshlrev_b32 v11, 5, v18 // (op 13678) 

s_cmp_eq_i32 s47, 0 // (op 13678) 
s_cselect_b32 s18, 0, -1 // (op 13678) 

v_mov_b32 v19, 0 // (op 13678) convert
v_mov_b32 v18, v11 // (op 13678) convert

s_and_b32 s19, s13, 31 // (op 13678) 
s_lshl_b32 s47, 1, s19 // (op 13678) 
s_add_i32 s22, s47, s18 // (op 13678) 


s_and_b32 s18, s1, s22 // (op 13678) 


s_add_i32 s1, s0, s18 // (op 13678) 


s_ashr_i32 s0, s1, s19 // (op 13678) 


s_lshl_b32 s1, s13, 24 // (op 13678) 
s_ashr_i32 s47, s1, 31 // (op 13678) 

s_xor_b32 s1, s0, s47 // (op 13678) 

s_sub_i32 s0, s1, s47 // (op 13678) 


s_mul_hi_i32 s1, s0, s14 // (op 13678) 
s_add_i32 s47, s1, s0 // (op 13678) 

s_ashr_i32 s1, s47, 31 // (op 13678) 
s_cmp_eq_i32 s14, 0 // (op 13678) 
s_mov_b32 s18, scc // (op 13678) 
s_cmp_eq_i32 s18, 0 // (op 13678) 
s_cselect_b32 s19, 0, -1 // (op 13678) 

s_and_b32 s18, s15, 31 // (op 13678) 
s_lshl_b32 s22, 1, s18 // (op 13678) 
s_add_i32 s23, s22, s19 // (op 13678) 


s_and_b32 s19, s1, s23 // (op 13678) 


s_add_i32 s1, s47, s19 // (op 13678) 


s_ashr_i32 s19, s1, s18 // (op 13678) 


s_lshl_b32 s1, s15, 24 // (op 13678) 
s_ashr_i32 s18, s1, 31 // (op 13678) 

s_xor_b32 s1, s19, s18 // (op 13678) 

s_sub_i32 s19, s1, s18 // (op 13678) 


s_add_u32 s22, s6, 128 // (op 13678) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13678) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13678) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13678) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13678) 

s_mul_i32 s1, s19, s22 // (op 13678) 

s_sub_i32 s47, s0, s1 // (op 13678) 


s_mul_hi_i32 s0, s16, s9 // (op 13678) 
s_add_i32 s1, s0, s16 // (op 13678) 

s_ashr_i32 s0, s1, 31 // (op 13678) 
s_cmp_eq_i32 s9, 0 // (op 13678) 
s_mov_b32 s18, scc // (op 13678) 
s_cmp_eq_i32 s18, 0 // (op 13678) 
s_cselect_b32 s19, 0, -1 // (op 13678) 

s_and_b32 s18, s45, 31 // (op 13678) 
s_lshl_b32 s24, 1, s18 // (op 13678) 
s_add_i32 s25, s24, s19 // (op 13678) 


s_and_b32 s19, s0, s25 // (op 13678) 


s_add_i32 s0, s1, s19 // (op 13678) 


s_ashr_i32 s1, s0, s18 // (op 13678) 


s_lshl_b32 s0, s45, 24 // (op 13678) 
s_ashr_i32 s18, s0, 31 // (op 13678) 

s_xor_b32 s0, s1, s18 // (op 13678) 

s_sub_i32 s1, s0, s18 // (op 13678) 


s_mul_i32 s0, s11, s22 // (op 13678) 

s_mul_i32 s18, s8, s0 // (op 13678) 

s_mul_i32 s0, s1, s18 // (op 13678) 

s_sub_u32 s18, s16, s0 // (op 13678) 

s_mul_hi_i32 s0, s18, s12 // (op 13678) 
s_add_i32 s19, s0, s18 // (op 13678) 


s_ashr_i32 s0, s19, 31 // (op 13678) 
s_cmp_eq_i32 s12, 0 // (op 13678) 
s_mov_b32 s18, scc // (op 13678) 
s_cmp_eq_i32 s18, 0 // (op 13678) 
s_cselect_b32 s22, 0, -1 // (op 13678) 

s_and_b32 s18, s44, 31 // (op 13678) 
s_lshl_b32 s23, 1, s18 // (op 13678) 
s_add_i32 s24, s23, s22 // (op 13678) 


s_and_b32 s25, s0, s24 // (op 13678) 


s_add_i32 s0, s19, s25 // (op 13678) 


s_ashr_i32 s19, s0, s18 // (op 13678) 


s_lshl_b32 s0, s44, 24 // (op 13678) 
s_ashr_i32 s18, s0, 31 // (op 13678) 

s_xor_b32 s0, s19, s18 // (op 13678) 

s_sub_i32 s19, s0, s18 // (op 13678) 


s_cmp_eq_i32 s1, 0 // (op 13678) 
s_mov_b32 s0, scc // (op 13678) 

s_cmp_eq_i32 s0, 0 // (op 13678) 
s_cselect_b32 s1, s19, s47 // (op 13678) 



v_lshl_add_u32 v11, s1, 7, v10 // (op 13678) 


v_mul_lo_u32 v21, v11, s31 // (op 13678) most significant: low of low * high
v_mul_hi_u32 v10, v11, s30 // (op 13678) most significant: high of low * low
v_mul_lo_u32 v20, v11, s30 // (op 13678) least significant: low of low * low
v_add_u32 v21, v21, v10 // (op 13678) most significant: sum


v_add_co_u32 v10, s[0:1], v18, v20 // (op 13678) least significant half
v_addc_co_u32 v11, s[0:1], v19, v21, s[0:1] // (op 13678) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 13678) 


v_lshrrev_b64 v[18:19], 3, v[10:11] // (op 13678) 

s_mul_i32 s1, s30, s7 // (op 13680) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 13680) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 13680) least significant
s_mul_hi_u32 s18, s30, s6 // (op 13680) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13680) most significant: sum
s_add_u32 s1, s1, s18 // (op 13680) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13680) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 13680) 

s_mov_b32 s60, s32 // (op 13680) reg expression
s_mov_b32 s61, s33 // (op 13680) reg expression
s_mov_b32 s62, s0 // (op 13680) reg expression
s_mov_b32 s63, 131072 // (op 13680) call()

s_mul_hi_i32 s47, s16, s10 // (op 13956) 
v_and_b32 v10, 63, v1 // (op 13956) 
s_add_i32 s0, s47, s16 // (op 13956) 

v_lshrrev_b32 v11, 4, v10 // (op 13956) 
s_ashr_i32 s1, s0, 31 // (op 13956) 
v_and_b32 v20, 3, v11 // (op 13956) 

s_cmp_eq_i32 s10, 0 // (op 13956) 
s_mov_b32 s47, scc // (op 13956) 
v_and_b32 v11, 15, v1 // (op 13956) 
s_cmp_eq_i32 s47, 0 // (op 13956) 
s_cselect_b32 s18, 0, -1 // (op 13956) 

v_lshl_add_u32 v21, v20, 4, v11 // (op 13956) 


s_and_b32 s19, s13, 31 // (op 13956) 
v_and_b32 v11, 4294967168, v1 // (op 13956) 
s_lshl_b32 s47, 1, s19 // (op 13956) 
v_add_u32 v20, v11, v21 // (op 13956) 


s_add_i32 s22, s47, s18 // (op 13956) 


v_lshrrev_b32 v11, 6, v10 // (op 13956) 

s_and_b32 s18, s1, s22 // (op 13956) 


v_lshlrev_b32 v10, 2, v11 // (op 13956) 

s_add_i32 s1, s0, s18 // (op 13956) 


s_ashr_i32 s0, s1, s19 // (op 13956) 


s_lshl_b32 s1, s13, 24 // (op 13956) 
s_ashr_i32 s47, s1, 31 // (op 13956) 

s_xor_b32 s1, s0, s47 // (op 13956) 

s_sub_i32 s0, s1, s47 // (op 13956) 


s_mul_hi_i32 s1, s0, s14 // (op 13956) 
s_add_i32 s47, s1, s0 // (op 13956) 

s_ashr_i32 s1, s47, 31 // (op 13956) 
s_cmp_eq_i32 s14, 0 // (op 13956) 
s_mov_b32 s18, scc // (op 13956) 
s_cmp_eq_i32 s18, 0 // (op 13956) 
s_cselect_b32 s19, 0, -1 // (op 13956) 

s_and_b32 s18, s15, 31 // (op 13956) 
s_lshl_b32 s22, 1, s18 // (op 13956) 
s_add_i32 s23, s22, s19 // (op 13956) 


s_and_b32 s19, s1, s23 // (op 13956) 


s_add_i32 s1, s47, s19 // (op 13956) 


s_ashr_i32 s19, s1, s18 // (op 13956) 


s_lshl_b32 s1, s15, 24 // (op 13956) 
s_ashr_i32 s18, s1, 31 // (op 13956) 

s_xor_b32 s1, s19, s18 // (op 13956) 

s_sub_i32 s19, s1, s18 // (op 13956) 


s_mul_i32 s1, s0, s11 // (op 13956) 

s_sub_u32 s0, s16, s1 // (op 13956) 

s_mul_i32 s1, s19, s11 // (op 13956) 
s_add_u32 s1, s1, s0 // (op 13956) 


s_mul_hi_i32 s0, s16, s9 // (op 13956) 
s_add_i32 s47, s0, s16 // (op 13956) 

s_ashr_i32 s0, s47, 31 // (op 13956) 
s_cmp_eq_i32 s9, 0 // (op 13956) 
s_mov_b32 s18, scc // (op 13956) 
s_cmp_eq_i32 s18, 0 // (op 13956) 
s_cselect_b32 s19, 0, -1 // (op 13956) 

s_and_b32 s18, s45, 31 // (op 13956) 
s_lshl_b32 s22, 1, s18 // (op 13956) 
s_add_i32 s23, s22, s19 // (op 13956) 


s_and_b32 s19, s0, s23 // (op 13956) 


s_add_i32 s0, s47, s19 // (op 13956) 


s_ashr_i32 s19, s0, s18 // (op 13956) 


s_lshl_b32 s0, s45, 24 // (op 13956) 
s_ashr_i32 s18, s0, 31 // (op 13956) 

s_xor_b32 s0, s19, s18 // (op 13956) 

s_sub_i32 s19, s0, s18 // (op 13956) 


s_add_u32 s22, s6, 128 // (op 13956) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13956) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13956) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13956) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13956) 

s_mul_i32 s0, s11, s22 // (op 13956) 

s_mul_i32 s18, s8, s0 // (op 13956) 

s_mul_i32 s0, s19, s18 // (op 13956) 

s_sub_u32 s18, s16, s0 // (op 13956) 

s_mov_b32 s0, s18 // (op 13956) 
s_mul_hi_i32 s47, s0, s12 // (op 13956) 
s_add_i32 s22, s47, s0 // (op 13956) 


s_ashr_i32 s0, s22, 31 // (op 13956) 
s_cmp_eq_i32 s12, 0 // (op 13956) 
s_mov_b32 s47, scc // (op 13956) 
s_cmp_eq_i32 s47, 0 // (op 13956) 
s_cselect_b32 s23, 0, -1 // (op 13956) 

s_and_b32 s47, s44, 31 // (op 13956) 
s_lshl_b32 s24, 1, s47 // (op 13956) 
s_add_i32 s25, s24, s23 // (op 13956) 


s_and_b32 s23, s0, s25 // (op 13956) 


s_add_i32 s0, s22, s23 // (op 13956) 


s_ashr_i32 s22, s0, s47 // (op 13956) 


s_lshl_b32 s0, s44, 24 // (op 13956) 
s_ashr_i32 s47, s0, 31 // (op 13956) 

s_xor_b32 s0, s22, s47 // (op 13956) 

s_sub_i32 s22, s0, s47 // (op 13956) 


s_mul_i32 s0, s46, s11 // (op 13956) 
s_add_u32 s24, s20, 256 // (op 13956) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13956) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 13956) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 13956) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 13956) 

s_sub_i32 s23, s24, s0 // (op 13956) 

s_mul_i32 s47, s22, s23 // (op 13956) 


s_sub_u32 s22, s18, s47 // (op 13956) 


s_add_u32 s18, s22, s0 // (op 13956) 


s_cmp_eq_i32 s19, 0 // (op 13956) 
s_mov_b32 s0, scc // (op 13956) 

s_cmp_eq_i32 s0, 0 // (op 13956) 
s_cselect_b32 s19, s18, s1 // (op 13956) 



v_lshl_add_u32 v11, s19, 8, v20 // (op 13956) 


v_mul_lo_u32 v20, v11, s34 // (op 13956) 
v_add_u32 v20, v20, v10 // (op 13956) 


s_mul_i32 s1, s34, s21 // (op 13958) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 13958) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 13958) least significant
s_mul_hi_u32 s18, s34, s20 // (op 13958) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13958) most significant: sum
s_add_u32 s1, s1, s18 // (op 13958) most significant: sum


s_mov_b32 s64, s36 // (op 13958) reg expression
s_mov_b32 s65, s37 // (op 13958) reg expression
s_mov_b32 s66, s0 // (op 13958) reg expression
s_mov_b32 s67, 131072 // (op 13958) call()

s_mul_hi_i32 s47, s16, s10 // (op 13977) 
v_and_b32 v10, 63, v1 // (op 13977) 
s_add_i32 s0, s47, s16 // (op 13977) 

v_lshrrev_b32 v11, 4, v10 // (op 13977) 
s_ashr_i32 s1, s0, 31 // (op 13977) 
v_and_b32 v21, 3, v11 // (op 13977) 

s_cmp_eq_i32 s10, 0 // (op 13977) 
s_mov_b32 s47, scc // (op 13977) 
v_and_b32 v11, 15, v1 // (op 13977) 
s_cmp_eq_i32 s47, 0 // (op 13977) 
s_cselect_b32 s18, 0, -1 // (op 13977) 

v_lshl_add_u32 v22, v21, 4, v11 // (op 13977) 


s_and_b32 s19, s13, 31 // (op 13977) 
v_and_b32 v11, 4294967168, v1 // (op 13977) 
s_lshl_b32 s47, 1, s19 // (op 13977) 
v_add_u32 v21, v11, v22 // (op 13977) 


s_add_i32 s22, s47, s18 // (op 13977) 


v_lshrrev_b32 v11, 6, v10 // (op 13977) 

s_and_b32 s18, s1, s22 // (op 13977) 


v_lshlrev_b32 v10, 2, v11 // (op 13977) 

s_add_i32 s1, s0, s18 // (op 13977) 


s_ashr_i32 s0, s1, s19 // (op 13977) 


s_lshl_b32 s1, s13, 24 // (op 13977) 
s_ashr_i32 s47, s1, 31 // (op 13977) 

s_xor_b32 s1, s0, s47 // (op 13977) 

s_sub_i32 s0, s1, s47 // (op 13977) 


s_mul_hi_i32 s1, s0, s14 // (op 13977) 
s_add_i32 s47, s1, s0 // (op 13977) 

s_ashr_i32 s1, s47, 31 // (op 13977) 
s_cmp_eq_i32 s14, 0 // (op 13977) 
s_mov_b32 s18, scc // (op 13977) 
s_cmp_eq_i32 s18, 0 // (op 13977) 
s_cselect_b32 s19, 0, -1 // (op 13977) 

s_and_b32 s18, s15, 31 // (op 13977) 
s_lshl_b32 s22, 1, s18 // (op 13977) 
s_add_i32 s23, s22, s19 // (op 13977) 


s_and_b32 s19, s1, s23 // (op 13977) 


s_add_i32 s1, s47, s19 // (op 13977) 


s_ashr_i32 s19, s1, s18 // (op 13977) 


s_lshl_b32 s1, s15, 24 // (op 13977) 
s_ashr_i32 s18, s1, 31 // (op 13977) 

s_xor_b32 s1, s19, s18 // (op 13977) 

s_sub_i32 s19, s1, s18 // (op 13977) 


s_mul_i32 s1, s0, s11 // (op 13977) 

s_sub_u32 s0, s16, s1 // (op 13977) 

s_mul_i32 s1, s19, s11 // (op 13977) 
s_add_u32 s1, s1, s0 // (op 13977) 


s_mul_hi_i32 s0, s16, s9 // (op 13977) 
s_add_i32 s47, s0, s16 // (op 13977) 

s_ashr_i32 s0, s47, 31 // (op 13977) 
s_cmp_eq_i32 s9, 0 // (op 13977) 
s_mov_b32 s18, scc // (op 13977) 
s_cmp_eq_i32 s18, 0 // (op 13977) 
s_cselect_b32 s19, 0, -1 // (op 13977) 

s_and_b32 s18, s45, 31 // (op 13977) 
s_lshl_b32 s22, 1, s18 // (op 13977) 
s_add_i32 s23, s22, s19 // (op 13977) 


s_and_b32 s19, s0, s23 // (op 13977) 


s_add_i32 s0, s47, s19 // (op 13977) 


s_ashr_i32 s19, s0, s18 // (op 13977) 


s_lshl_b32 s0, s45, 24 // (op 13977) 
s_ashr_i32 s18, s0, 31 // (op 13977) 

s_xor_b32 s0, s19, s18 // (op 13977) 

s_sub_i32 s19, s0, s18 // (op 13977) 


s_add_u32 s22, s6, 128 // (op 13977) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13977) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13977) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13977) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13977) 

s_mul_i32 s0, s11, s22 // (op 13977) 

s_mul_i32 s18, s8, s0 // (op 13977) 

s_mul_i32 s0, s19, s18 // (op 13977) 

s_sub_u32 s18, s16, s0 // (op 13977) 

s_mov_b32 s0, s18 // (op 13977) 
s_mul_hi_i32 s47, s0, s12 // (op 13977) 
s_add_i32 s22, s47, s0 // (op 13977) 


s_ashr_i32 s0, s22, 31 // (op 13977) 
s_cmp_eq_i32 s12, 0 // (op 13977) 
s_mov_b32 s47, scc // (op 13977) 
s_cmp_eq_i32 s47, 0 // (op 13977) 
s_cselect_b32 s23, 0, -1 // (op 13977) 

s_and_b32 s47, s44, 31 // (op 13977) 
s_lshl_b32 s24, 1, s47 // (op 13977) 
s_add_i32 s25, s24, s23 // (op 13977) 


s_and_b32 s23, s0, s25 // (op 13977) 


s_add_i32 s0, s22, s23 // (op 13977) 


s_ashr_i32 s22, s0, s47 // (op 13977) 


s_lshl_b32 s0, s44, 24 // (op 13977) 
s_ashr_i32 s47, s0, 31 // (op 13977) 

s_xor_b32 s0, s22, s47 // (op 13977) 

s_sub_i32 s22, s0, s47 // (op 13977) 


s_mul_i32 s0, s46, s11 // (op 13977) 
s_add_u32 s24, s20, 256 // (op 13977) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13977) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 13977) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 13977) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 8 // (op 13977) 

s_sub_i32 s23, s24, s0 // (op 13977) 

s_mul_i32 s47, s22, s23 // (op 13977) 


s_sub_u32 s22, s18, s47 // (op 13977) 


s_add_u32 s18, s22, s0 // (op 13977) 


s_cmp_eq_i32 s19, 0 // (op 13977) 
s_mov_b32 s0, scc // (op 13977) 

s_cmp_eq_i32 s0, 0 // (op 13977) 
s_cselect_b32 s19, s18, s1 // (op 13977) 



v_lshl_add_u32 v11, s19, 8, v21 // (op 13977) 


v_mul_lo_u32 v21, v11, s34 // (op 13977) 
v_add_u32 v21, v21, v10 // (op 13977) 


s_mul_i32 s1, s34, s21 // (op 13979) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 13979) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 13979) least significant
s_mul_hi_u32 s18, s34, s20 // (op 13979) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13979) most significant: sum
s_add_u32 s1, s1, s18 // (op 13979) most significant: sum


s_mov_b32 s64, s36 // (op 13979) reg expression
s_mov_b32 s65, s37 // (op 13979) reg expression
s_mov_b32 s66, s0 // (op 13979) reg expression
s_mov_b32 s67, 131072 // (op 13979) call()

s_mul_hi_i32 s47, s16, s10 // (op 14008) 
v_and_b32 v10, 63, v1 // (op 14008) 
s_add_i32 s0, s47, s16 // (op 14008) 

v_lshrrev_b32 v11, 4, v10 // (op 14008) 
s_ashr_i32 s1, s0, 31 // (op 14008) 
v_and_b32 v22, 3, v11 // (op 14008) 

s_cmp_eq_i32 s10, 0 // (op 14008) 
s_mov_b32 s47, scc // (op 14008) 
v_and_b32 v11, 15, v1 // (op 14008) 
s_cmp_eq_i32 s47, 0 // (op 14008) 
s_cselect_b32 s18, 0, -1 // (op 14008) 

v_lshl_add_u32 v23, v22, 4, v11 // (op 14008) 


s_and_b32 s19, s13, 31 // (op 14008) 
v_lshrrev_b32 v11, 6, v1 // (op 14008) 
s_lshl_b32 s47, 1, s19 // (op 14008) 
v_and_b32 v22, 1, v11 // (op 14008) 

s_add_i32 s22, s47, s18 // (op 14008) 


v_lshl_add_u32 v11, v22, 6, v23 // (op 14008) 


s_and_b32 s18, s1, s22 // (op 14008) 


v_lshrrev_b32 v22, 6, v10 // (op 14008) 

s_add_i32 s1, s0, s18 // (op 14008) 


s_ashr_i32 s0, s1, s19 // (op 14008) 


s_lshl_b32 s1, s13, 24 // (op 14008) 
s_ashr_i32 s47, s1, 31 // (op 14008) 

s_xor_b32 s1, s0, s47 // (op 14008) 

s_sub_i32 s0, s1, s47 // (op 14008) 


s_mul_hi_i32 s1, s0, s14 // (op 14008) 
s_add_i32 s47, s1, s0 // (op 14008) 

s_ashr_i32 s1, s47, 31 // (op 14008) 
s_cmp_eq_i32 s14, 0 // (op 14008) 
s_mov_b32 s18, scc // (op 14008) 
s_cmp_eq_i32 s18, 0 // (op 14008) 
s_cselect_b32 s19, 0, -1 // (op 14008) 

s_and_b32 s18, s15, 31 // (op 14008) 
s_lshl_b32 s22, 1, s18 // (op 14008) 
s_add_i32 s23, s22, s19 // (op 14008) 


s_and_b32 s19, s1, s23 // (op 14008) 


s_add_i32 s1, s47, s19 // (op 14008) 


s_ashr_i32 s19, s1, s18 // (op 14008) 


s_lshl_b32 s1, s15, 24 // (op 14008) 
s_ashr_i32 s18, s1, 31 // (op 14008) 

s_xor_b32 s1, s19, s18 // (op 14008) 

s_sub_i32 s19, s1, s18 // (op 14008) 


s_add_u32 s22, s6, 128 // (op 14008) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 14008) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 14008) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 14008) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 14008) 

s_mul_i32 s1, s19, s22 // (op 14008) 

s_sub_i32 s47, s0, s1 // (op 14008) 


s_mul_hi_i32 s0, s16, s9 // (op 14008) 
s_add_i32 s1, s0, s16 // (op 14008) 

s_ashr_i32 s0, s1, 31 // (op 14008) 
s_cmp_eq_i32 s9, 0 // (op 14008) 
s_mov_b32 s18, scc // (op 14008) 
s_cmp_eq_i32 s18, 0 // (op 14008) 
s_cselect_b32 s19, 0, -1 // (op 14008) 

s_and_b32 s18, s45, 31 // (op 14008) 
s_lshl_b32 s24, 1, s18 // (op 14008) 
s_add_i32 s25, s24, s19 // (op 14008) 


s_and_b32 s19, s0, s25 // (op 14008) 


s_add_i32 s0, s1, s19 // (op 14008) 


s_ashr_i32 s1, s0, s18 // (op 14008) 


s_lshl_b32 s0, s45, 24 // (op 14008) 
s_ashr_i32 s18, s0, 31 // (op 14008) 

s_xor_b32 s0, s1, s18 // (op 14008) 

s_sub_i32 s1, s0, s18 // (op 14008) 


s_mul_i32 s0, s11, s22 // (op 14008) 

s_mul_i32 s18, s8, s0 // (op 14008) 

s_mul_i32 s0, s1, s18 // (op 14008) 

s_sub_u32 s18, s16, s0 // (op 14008) 

s_mul_hi_i32 s0, s18, s12 // (op 14008) 
s_add_i32 s19, s0, s18 // (op 14008) 


s_ashr_i32 s0, s19, 31 // (op 14008) 
s_cmp_eq_i32 s12, 0 // (op 14008) 
s_mov_b32 s18, scc // (op 14008) 
s_cmp_eq_i32 s18, 0 // (op 14008) 
s_cselect_b32 s22, 0, -1 // (op 14008) 

s_and_b32 s18, s44, 31 // (op 14008) 
s_lshl_b32 s23, 1, s18 // (op 14008) 
s_add_i32 s24, s23, s22 // (op 14008) 


s_and_b32 s25, s0, s24 // (op 14008) 


s_add_i32 s0, s19, s25 // (op 14008) 


s_ashr_i32 s19, s0, s18 // (op 14008) 


s_lshl_b32 s0, s44, 24 // (op 14008) 
s_ashr_i32 s18, s0, 31 // (op 14008) 

s_xor_b32 s0, s19, s18 // (op 14008) 

s_sub_i32 s19, s0, s18 // (op 14008) 


s_cmp_eq_i32 s1, 0 // (op 14008) 
s_mov_b32 s0, scc // (op 14008) 

s_cmp_eq_i32 s0, 0 // (op 14008) 
s_cselect_b32 s1, s19, s47 // (op 14008) 



v_lshl_add_u32 v10, s1, 7, v11 // (op 14008) 


v_mul_lo_u32 v11, v10, s38 // (op 14008) 

v_lshl_add_u32 v10, v22, 2, v11 // (op 14008) 


s_mul_i32 s1, s38, s7 // (op 14010) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 14010) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 14010) least significant
s_mul_hi_u32 s18, s38, s6 // (op 14010) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 14010) most significant: sum
s_add_u32 s1, s1, s18 // (op 14010) most significant: sum


s_mov_b32 s68, s40 // (op 14010) reg expression
s_mov_b32 s69, s41 // (op 14010) reg expression
s_mov_b32 s70, s0 // (op 14010) reg expression
s_mov_b32 s71, 131072 // (op 14010) call()

s_mul_hi_i32 s47, s16, s10 // (op 14029) 
v_and_b32 v11, 63, v1 // (op 14029) 
s_add_i32 s0, s47, s16 // (op 14029) 

v_lshrrev_b32 v22, 4, v11 // (op 14029) 
s_ashr_i32 s1, s0, 31 // (op 14029) 
v_and_b32 v23, 3, v22 // (op 14029) 

s_cmp_eq_i32 s10, 0 // (op 14029) 
s_mov_b32 s47, scc // (op 14029) 
v_and_b32 v22, 15, v1 // (op 14029) 
s_cmp_eq_i32 s47, 0 // (op 14029) 
s_cselect_b32 s18, 0, -1 // (op 14029) 

v_lshl_add_u32 v24, v23, 4, v22 // (op 14029) 


s_and_b32 s19, s13, 31 // (op 14029) 
v_lshrrev_b32 v22, 6, v1 // (op 14029) 
s_lshl_b32 s47, 1, s19 // (op 14029) 
v_and_b32 v23, 1, v22 // (op 14029) 

s_add_i32 s22, s47, s18 // (op 14029) 


v_lshl_add_u32 v22, v23, 6, v24 // (op 14029) 


s_and_b32 s18, s1, s22 // (op 14029) 


v_lshrrev_b32 v23, 6, v11 // (op 14029) 

s_add_i32 s1, s0, s18 // (op 14029) 


s_ashr_i32 s0, s1, s19 // (op 14029) 


s_lshl_b32 s1, s13, 24 // (op 14029) 
s_ashr_i32 s47, s1, 31 // (op 14029) 

s_xor_b32 s1, s0, s47 // (op 14029) 

s_sub_i32 s0, s1, s47 // (op 14029) 


s_mul_hi_i32 s1, s0, s14 // (op 14029) 
s_add_i32 s47, s1, s0 // (op 14029) 

s_ashr_i32 s1, s47, 31 // (op 14029) 
s_cmp_eq_i32 s14, 0 // (op 14029) 
s_mov_b32 s18, scc // (op 14029) 
s_cmp_eq_i32 s18, 0 // (op 14029) 
s_cselect_b32 s19, 0, -1 // (op 14029) 

s_and_b32 s18, s15, 31 // (op 14029) 
s_lshl_b32 s22, 1, s18 // (op 14029) 
s_add_i32 s23, s22, s19 // (op 14029) 


s_and_b32 s19, s1, s23 // (op 14029) 


s_add_i32 s1, s47, s19 // (op 14029) 


s_ashr_i32 s19, s1, s18 // (op 14029) 


s_lshl_b32 s1, s15, 24 // (op 14029) 
s_ashr_i32 s18, s1, 31 // (op 14029) 

s_xor_b32 s1, s19, s18 // (op 14029) 

s_sub_i32 s19, s1, s18 // (op 14029) 


s_add_u32 s22, s6, 128 // (op 14029) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 14029) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 14029) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 14029) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 14029) 

s_mul_i32 s1, s19, s22 // (op 14029) 

s_sub_i32 s47, s0, s1 // (op 14029) 


s_mul_hi_i32 s0, s16, s9 // (op 14029) 
s_add_i32 s1, s0, s16 // (op 14029) 

s_ashr_i32 s0, s1, 31 // (op 14029) 
s_cmp_eq_i32 s9, 0 // (op 14029) 
s_mov_b32 s18, scc // (op 14029) 
s_cmp_eq_i32 s18, 0 // (op 14029) 
s_cselect_b32 s19, 0, -1 // (op 14029) 

s_and_b32 s18, s45, 31 // (op 14029) 
s_lshl_b32 s24, 1, s18 // (op 14029) 
s_add_i32 s25, s24, s19 // (op 14029) 


s_and_b32 s19, s0, s25 // (op 14029) 


s_add_i32 s0, s1, s19 // (op 14029) 


s_ashr_i32 s1, s0, s18 // (op 14029) 


s_lshl_b32 s0, s45, 24 // (op 14029) 
s_ashr_i32 s18, s0, 31 // (op 14029) 

s_xor_b32 s0, s1, s18 // (op 14029) 

s_sub_i32 s1, s0, s18 // (op 14029) 


s_mul_i32 s0, s11, s22 // (op 14029) 

s_mul_i32 s18, s8, s0 // (op 14029) 

s_mul_i32 s0, s1, s18 // (op 14029) 

s_sub_u32 s18, s16, s0 // (op 14029) 

s_mul_hi_i32 s0, s18, s12 // (op 14029) 
s_add_i32 s19, s0, s18 // (op 14029) 


s_ashr_i32 s0, s19, 31 // (op 14029) 
s_cmp_eq_i32 s12, 0 // (op 14029) 
s_mov_b32 s18, scc // (op 14029) 
s_cmp_eq_i32 s18, 0 // (op 14029) 
s_cselect_b32 s22, 0, -1 // (op 14029) 

s_and_b32 s18, s44, 31 // (op 14029) 
s_lshl_b32 s23, 1, s18 // (op 14029) 
s_add_i32 s24, s23, s22 // (op 14029) 


s_and_b32 s25, s0, s24 // (op 14029) 


s_add_i32 s0, s19, s25 // (op 14029) 


s_ashr_i32 s19, s0, s18 // (op 14029) 


s_lshl_b32 s0, s44, 24 // (op 14029) 
s_ashr_i32 s18, s0, 31 // (op 14029) 

s_xor_b32 s0, s19, s18 // (op 14029) 

s_sub_i32 s19, s0, s18 // (op 14029) 


s_cmp_eq_i32 s1, 0 // (op 14029) 
s_mov_b32 s0, scc // (op 14029) 

s_cmp_eq_i32 s0, 0 // (op 14029) 
s_cselect_b32 s1, s19, s47 // (op 14029) 



v_lshl_add_u32 v11, s1, 7, v22 // (op 14029) 


v_mul_lo_u32 v22, v11, s38 // (op 14029) 

v_lshl_add_u32 v11, v23, 2, v22 // (op 14029) 


s_mul_i32 s1, s38, s7 // (op 14031) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 14031) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 14031) least significant
s_mul_hi_u32 s18, s38, s6 // (op 14031) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 14031) most significant: sum
s_add_u32 s1, s1, s18 // (op 14031) most significant: sum


s_mov_b32 s68, s40 // (op 14031) reg expression
s_mov_b32 s69, s41 // (op 14031) reg expression
s_mov_b32 s70, s0 // (op 14031) reg expression
s_mov_b32 s71, 131072 // (op 14031) call()

v_and_b32 v22, -4, v1 // (op 13806) 
v_sub_u32 v23, v1, v22 // (op 13806) 

v_lshlrev_b32 v22, 5, v23 // (op 13806) 

v_lshrrev_b32 v23, 2, v1 // (op 13806) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 13806) 


v_lshlrev_b32 v22, 2, v24 // (op 13806) 

v_lshrrev_b32 v23, 3, v22 // (op 13806) 

s_nop 0
v_readfirstlane_b32 s47, v23 // (op 13806) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 13513) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 13513) 

v_lshrrev_b64 v[22:23], 3, s[18:19] // (op 13513) 


v_mov_b64 v[24:25], v[12:13] // (op 13513) 
s_mov_b32 m0, s47 // (op 13513) reg expression
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 13513) Load value direct to lds
v_add_co_u32 v24, s[0:1], v24, v22 // (op 13513) least significant half
v_addc_co_u32 v25, s[0:1], 0, v25, s[0:1] // (op 13513) most significant half

s_add_u32 m0, m0, 4096 // (op 13513) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 13513) Load value direct to lds



v_and_b32 v22, -8, v1 // (op 13749) 
v_sub_u32 v23, v1, v22 // (op 13749) 

v_lshlrev_b32 v22, 4, v23 // (op 13749) 

v_lshrrev_b32 v23, 3, v1 // (op 13749) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 13749) 


s_nop 0
v_readfirstlane_b32 s47, v24 // (op 13749) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 13518) 
v_mov_b64 v[22:23], s[0:1] // (op 13518) convert


v_mov_b64 v[24:25], v[14:15] // (op 13518) 
s_add_u32 s0, 8192, s47 // (op 13518) 
s_mov_b32 m0, s0 // (op 13518) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 13518) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 13518) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 13518) most significant half

s_add_u32 m0, m0, 4096 // (op 13518) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 13518) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 13518) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 13518) most significant half

s_add_u32 m0, m0, 4096 // (op 13518) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 13518) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 13518) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 13518) most significant half

s_add_u32 m0, m0, 4096 // (op 13518) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 13518) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 13518) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 13518) most significant half

s_add_u32 m0, m0, 4096 // (op 13518) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 13518) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 13518) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 13518) most significant half

s_add_u32 m0, m0, 4096 // (op 13518) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 13518) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 13518) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 13518) most significant half

s_add_u32 m0, m0, 4096 // (op 13518) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 13518) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 13518) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 13518) most significant half

s_add_u32 m0, m0, 4096 // (op 13518) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 13518) Load value direct to lds






v_mov_b32 v22, v10 // (op 76) reg expression

buffer_load_dwordx2 v[24:25], v22, s[68:71], 0 offen // (op 76) Load value




s_lshl_b32 s47, s34, 6 // (op 2747) 
v_add_u32 v10, v20, s47 // (op 2747) 


buffer_load_dwordx2 v[22:23], v10, s[64:67], 0 offen // (op 2747) Load value



v_mov_b32 v10, v20 // (op 72) reg expression

buffer_load_dwordx2 v[26:27], v10, s[64:67], 0 offen // (op 72) Load value


v_and_b32 v10, -4, v1 // (op 13917) 
v_sub_u32 v20, v1, v10 // (op 13917) 

v_lshlrev_b32 v10, 5, v20 // (op 13917) 

v_lshrrev_b32 v20, 2, v1 // (op 13917) 
v_lshl_add_u32 v28, v20, 7, v10 // (op 13917) 


v_lshlrev_b32 v10, 2, v28 // (op 13917) 

v_lshrrev_b32 v20, 3, v10 // (op 13917) 

s_nop 0
v_readfirstlane_b32 s47, v20 // (op 13917) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 13523) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 13523) 

v_lshrrev_b64 v[28:29], 3, s[18:19] // (op 13523) 


v_mov_b64 v[30:31], v[12:13] // (op 13523) 
s_add_u32 s0, 40960, s47 // (op 13523) 
s_mov_b32 m0, s0 // (op 13523) reg expression
buffer_load_dwordx4 v30, s[60:63], 64 offen offset: 0 lds // (op 13523) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 13523) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 13523) most significant half

s_add_u32 m0, m0, 4096 // (op 13523) 
buffer_load_dwordx4 v30, s[60:63], 64 offen offset: 0 lds // (op 13523) Load value direct to lds





v_and_b32 v10, -8, v1 // (op 13873) 
v_sub_u32 v20, v1, v10 // (op 13873) 

v_lshlrev_b32 v10, 4, v20 // (op 13873) 

v_lshrrev_b32 v20, 3, v1 // (op 13873) 
v_lshl_add_u32 v12, v20, 7, v10 // (op 13873) 


s_nop 0
v_readfirstlane_b32 s47, v12 // (op 13873) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 13525) 
v_mov_b64 v[12:13], s[0:1] // (op 13525) convert


v_mov_b64 v[28:29], v[14:15] // (op 13525) 
s_add_u32 s0, 49152, s47 // (op 13525) 
s_mov_b32 m0, s0 // (op 13525) reg expression
s_mov_b32 s1, 128 // (op 13525) call()
buffer_load_dwordx4 v28, s[48:51], s1 offen offset: 0 lds // (op 13525) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v12 // (op 13525) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 13525) most significant half

s_add_u32 m0, m0, 4096 // (op 13525) 
buffer_load_dwordx4 v28, s[48:51], s1 offen offset: 0 lds // (op 13525) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v12 // (op 13525) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 13525) most significant half

s_add_u32 m0, m0, 4096 // (op 13525) 
buffer_load_dwordx4 v28, s[48:51], s1 offen offset: 0 lds // (op 13525) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v12 // (op 13525) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 13525) most significant half

s_add_u32 m0, m0, 4096 // (op 13525) 
buffer_load_dwordx4 v28, s[48:51], s1 offen offset: 0 lds // (op 13525) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v12 // (op 13525) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 13525) most significant half

s_add_u32 m0, m0, 4096 // (op 13525) 
buffer_load_dwordx4 v28, s[48:51], s1 offen offset: 0 lds // (op 13525) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v12 // (op 13525) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 13525) most significant half

s_add_u32 m0, m0, 4096 // (op 13525) 
buffer_load_dwordx4 v28, s[48:51], s1 offen offset: 0 lds // (op 13525) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v12 // (op 13525) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 13525) most significant half

s_add_u32 m0, m0, 4096 // (op 13525) 
buffer_load_dwordx4 v28, s[48:51], s1 offen offset: 0 lds // (op 13525) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v12 // (op 13525) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 13525) most significant half

s_add_u32 m0, m0, 4096 // (op 13525) 
buffer_load_dwordx4 v28, s[48:51], s1 offen offset: 0 lds // (op 13525) Load value direct to lds






s_waitcnt vmcnt(21)
s_barrier  // (op 14302) 



ds_read_b128 v[12:15], v4 // (op 13121) Load local data 



ds_read_b128 v[28:31], v4 offset:1024 // (op 13126) Load local data 



ds_read_b128 v[32:35], v4 offset:2048 // (op 13131) Load local data 



ds_read_b128 v[36:39], v4 offset:3072 // (op 13136) Load local data 
s_waitcnt vmcnt(13)
s_barrier  // (op 14286) 



ds_read_b128 v[40:43], v3 offset:8192 // (op 13081) Load local data 
ds_read_b128 v[44:47], v3 offset:8256 // (op 13081) Load local data 



ds_read_b128 v[48:51], v3 offset:10240 // (op 13086) Load local data 
ds_read_b128 v[52:55], v3 offset:10304 // (op 13086) Load local data 



ds_read_b128 v[56:59], v3 offset:12288 // (op 13091) Load local data 
ds_read_b128 v[60:63], v3 offset:12352 // (op 13091) Load local data 



ds_read_b128 v[64:67], v3 offset:14336 // (op 13096) Load local data 
ds_read_b128 v[68:71], v3 offset:14400 // (op 13096) Load local data 



ds_read_b128 v[72:75], v3 offset:16384 // (op 13101) Load local data 
ds_read_b128 v[76:79], v3 offset:16448 // (op 13101) Load local data 



ds_read_b128 v[80:83], v3 offset:18432 // (op 13106) Load local data 
ds_read_b128 v[84:87], v3 offset:18496 // (op 13106) Load local data 



ds_read_b128 v[88:91], v3 offset:20480 // (op 13111) Load local data 
ds_read_b128 v[92:95], v3 offset:20544 // (op 13111) Load local data 



ds_read_b128 v[96:99], v3 offset:22528 // (op 13116) Load local data 
ds_read_b128 v[100:103], v3 offset:22592 // (op 13116) Load local data 
v_and_b32 v10, -8, v1 // (op 13721) 
v_sub_u32 v20, v1, v10 // (op 13721) 

v_lshlrev_b32 v10, 4, v20 // (op 13721) 

v_lshrrev_b32 v20, 3, v1 // (op 13721) 
v_lshl_add_u32 v104, v20, 7, v10 // (op 13721) 


s_nop 0
v_readfirstlane_b32 s47, v104 // (op 13721) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -4, v1 // (op 13778) 
v_sub_u32 v20, v1, v10 // (op 13778) 

v_lshlrev_b32 v10, 5, v20 // (op 13778) 

v_lshrrev_b32 v20, 2, v1 // (op 13778) 
v_lshl_add_u32 v104, v20, 7, v10 // (op 13778) 


v_lshlrev_b32 v10, 2, v104 // (op 13778) 

v_lshrrev_b32 v20, 3, v10 // (op 13778) 

s_nop 0
v_readfirstlane_b32 s0, v20 // (op 13778) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -8, v1 // (op 13858) 
v_sub_u32 v20, v1, v10 // (op 13858) 

v_lshlrev_b32 v10, 4, v20 // (op 13858) 

v_lshrrev_b32 v20, 3, v1 // (op 13858) 
v_lshl_add_u32 v104, v20, 7, v10 // (op 13858) 


s_nop 0
v_readfirstlane_b32 s1, v104 // (op 13858) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -4, v1 // (op 13902) 
v_sub_u32 v20, v1, v10 // (op 13902) 

v_lshlrev_b32 v10, 5, v20 // (op 13902) 

v_lshrrev_b32 v20, 2, v1 // (op 13902) 
v_lshl_add_u32 v104, v20, 7, v10 // (op 13902) 


v_lshlrev_b32 v10, 2, v104 // (op 13902) 

v_lshrrev_b32 v20, 3, v10 // (op 13902) 

s_nop 0
v_readfirstlane_b32 s18, v20 // (op 13902) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s19, 0 // (op 3300) call()
s_ashr_i64 s[22:23], s[42:43], 7 // (op 3299) 
s_mov_b32 s24, s22 // (op 3299) convert

s_and_b32 s25, s24, 4294967294 // (op 3299) 

s_cmp_lt_u32 s19, s25 // (op 3299) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopBottom_KLoop_3299 // (op 3299) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopBottom_KLoop_3299 if false)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopTop_KLoop_3299:
s_add_u32 s19, s19, 2 // (op 3301) 
s_waitcnt vmcnt(12)
v_mov_b32 v104, v24 // (op 13418) reg expression
s_barrier  // (op 14305) 
v_mov_b32 v105, v25 // (op 13418) reg expression

s_lshl_b64 s[22:23], s[30:31], 6 // (op 13486) 
s_lshl_b64 s[24:25], s[22:23], 2 // (op 13486) 

v_lshrrev_b64 v[106:107], 3, s[24:25] // (op 13486) 


v_mov_b64 v[108:109], v[18:19] // (op 13486) 
v_mov_b64 v[110:111], v[108:109] // (op 13486) 
s_mov_b32 m0, s0 // (op 13486) reg expression
s_mov_b32 s22, 128 // (op 13486) call()
buffer_load_dwordx4 v110, s[60:63], s22 offen offset: 0 lds // (op 13486) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v106 // (op 13486) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 13486) most significant half

s_add_u32 m0, m0, 4096 // (op 13486) 
buffer_load_dwordx4 v110, s[60:63], s22 offen offset: 0 lds // (op 13486) Load value direct to lds



s_waitcnt vmcnt(13)
v_mov_b32 v106, v22 // (op 13431) reg expression
v_mov_b32 v107, v23 // (op 13431) reg expression
s_waitcnt lgkmcnt(0)
s_barrier  // (op 14289) 
s_waitcnt vmcnt(12)
v_mov_b32 v110, v26 // (op 13444) reg expression

v_mov_b32 v111, v27 // (op 13444) reg expression
s_lshl_b64 s[22:23], s[26:27], 5 // (op 13481) 
v_mov_b64 v[112:113], s[22:23] // (op 13481) convert


v_mov_b64 v[114:115], v[16:17] // (op 13481) 
v_mov_b64 v[116:117], v[114:115] // (op 13481) 
s_add_u32 s22, 8192, s47 // (op 13481) 
s_mov_b32 m0, s22 // (op 13481) reg expression
s_mov_b32 s23, 256 // (op 13481) call()
buffer_load_dwordx4 v116, s[48:51], s23 offen offset: 0 lds // (op 13481) Load value direct to lds
v_add_co_u32 v116, s[24:25], v116, v112 // (op 13481) least significant half
v_addc_co_u32 v117, s[24:25], 0, v117, s[24:25] // (op 13481) most significant half

s_add_u32 m0, m0, 4096 // (op 13481) 
v_bfe_u32 v118, v110, 0, 8 // (op 12384) 
v_bfe_u32 v119, v110, 8, 8 // (op 12384) 
v_bfe_u32 v120, v110, 16, 8 // (op 12384) 
v_bfe_u32 v121, v110, 24, 8 // (op 12384) 
v_permlane16_swap_b32 v118, v119 // (op 12384) 
s_nop 0
v_permlane16_swap_b32 v120, v121 // (op 12384) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v118, v120 // (op 12384) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v119, v121 // (op 12384) 
buffer_load_dwordx4 v116, s[48:51], s23 offen offset: 0 lds // (op 13481) Load value direct to lds
v_add_co_u32 v116, s[24:25], v116, v112 // (op 13481) least significant half
v_addc_co_u32 v117, s[24:25], 0, v117, s[24:25] // (op 13481) most significant half

s_add_u32 m0, m0, 4096 // (op 13481) 
buffer_load_dwordx4 v116, s[48:51], s23 offen offset: 0 lds // (op 13481) Load value direct to lds
v_add_co_u32 v116, s[24:25], v116, v112 // (op 13481) least significant half
v_addc_co_u32 v117, s[24:25], 0, v117, s[24:25] // (op 13481) most significant half

s_add_u32 m0, m0, 4096 // (op 13481) 
v_bfe_u32 v122, v104, 0, 8 // (op 12511) 
v_bfe_u32 v123, v104, 8, 8 // (op 12511) 
v_bfe_u32 v124, v104, 16, 8 // (op 12511) 
v_bfe_u32 v125, v104, 24, 8 // (op 12511) 
v_permlane16_swap_b32 v122, v123 // (op 12511) 
s_nop 0
v_permlane16_swap_b32 v124, v125 // (op 12511) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v122, v124 // (op 12511) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v123, v125 // (op 12511) 
buffer_load_dwordx4 v116, s[48:51], s23 offen offset: 0 lds // (op 13481) Load value direct to lds
v_add_co_u32 v116, s[24:25], v116, v112 // (op 13481) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[12:15], a[0:3], v118, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 78) 
v_addc_co_u32 v117, s[24:25], 0, v117, s[24:25] // (op 13481) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[40:47], v[28:31], a[4:7], v118, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 315) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[40:47], v[32:35], a[8:11], v118, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 451) 
s_add_u32 m0, m0, 4096 // (op 13481) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[40:47], v[36:39], a[12:15], v118, v125 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 587) 
v_bfe_u32 v126, v106, 0, 8 // (op 12415) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[48:55], v[12:15], a[16:19], v119, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1007) 
v_bfe_u32 v127, v106, 8, 8 // (op 12415) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[48:55], v[28:31], a[20:23], v119, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 969) 
v_bfe_u32 v128, v106, 16, 8 // (op 12415) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[48:55], v[32:35], a[24:27], v119, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 931) 
v_bfe_u32 v129, v106, 24, 8 // (op 12415) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:55], v[36:39], a[28:31], v119, v125 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 893) 
v_permlane16_swap_b32 v126, v127 // (op 12415) 
v_permlane16_swap_b32 v128, v129 // (op 12415) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[56:63], v[12:15], a[32:35], v120, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1617) 
s_nop 0
v_permlane32_swap_b32 v126, v128 // (op 12415) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[56:63], v[28:31], a[36:39], v120, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1579) 
v_permlane32_swap_b32 v127, v129 // (op 12415) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[56:63], v[32:35], a[40:43], v120, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1541) 
s_waitcnt vmcnt(14)
s_barrier  // (op 14324) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[56:63], v[36:39], a[44:47], v120, v125 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1503) 
ds_read_b128 v[130:133], v6 offset:40960 // (op 216) Load local data 
buffer_load_dwordx4 v116, s[48:51], s23 offen offset: 0 lds // (op 13481) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[64:71], v[12:15], a[48:51], v121, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2227) 
v_add_co_u32 v116, s[24:25], v116, v112 // (op 13481) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[64:71], v[28:31], a[52:55], v121, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2189) 
v_addc_co_u32 v117, s[24:25], 0, v117, s[24:25] // (op 13481) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[64:71], v[32:35], a[56:59], v121, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2151) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[64:71], v[36:39], a[60:63], v121, v125 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2113) 
s_add_u32 m0, m0, 4096 // (op 13481) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[72:79], v[12:15], a[64:67], v126, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2837) 
buffer_load_dwordx4 v116, s[48:51], s23 offen offset: 0 lds // (op 13481) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[72:79], v[28:31], a[68:71], v126, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2799) 
v_add_co_u32 v116, s[24:25], v116, v112 // (op 13481) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:79], v[32:35], a[72:75], v126, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2761) 
v_addc_co_u32 v117, s[24:25], 0, v117, s[24:25] // (op 13481) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[72:79], v[36:39], a[76:79], v126, v125 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2723) 
s_add_u32 m0, m0, 4096 // (op 13481) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[80:87], v[12:15], a[80:83], v127, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3447) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[80:87], v[28:31], a[84:87], v127, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3409) 
buffer_load_dwordx4 v116, s[48:51], s23 offen offset: 0 lds // (op 13481) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[80:87], v[32:35], a[88:91], v127, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3371) 
v_add_co_u32 v116, s[24:25], v116, v112 // (op 13481) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[80:87], v[36:39], a[92:95], v127, v125 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3333) 
v_addc_co_u32 v117, s[24:25], 0, v117, s[24:25] // (op 13481) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[88:95], v[12:15], a[96:99], v128, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4057) 
s_add_u32 m0, m0, 4096 // (op 13481) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[88:95], v[28:31], a[100:103], v128, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4019) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[88:95], v[32:35], a[104:107], v128, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3981) 
buffer_load_dwordx4 v116, s[48:51], s23 offen offset: 0 lds // (op 13481) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[88:95], v[36:39], a[108:111], v128, v125 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3943) 
v_mov_b32 v10, v11 // (op 13426) 
v_add_u32 v20, 8, v10 // (op 13426) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[96:103], v[12:15], a[112:115], v129, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4667) 
buffer_load_dwordx2 v[24:25], v20, s[68:71], 0 offen // (op 13426) Load value


v_add_u32 v11, 8, v11 // (op 14019) 





v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[96:103], v[28:31], a[116:119], v129, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4629) 
v_mov_b32 v10, v21 // (op 13439) 
s_lshl_b32 s22, s34, 6 // (op 13439) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[96:103], v[32:35], a[120:123], v129, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4591) 
s_add_u32 s23, 8, s22 // (op 13439) 

v_add_u32 v20, v10, s23 // (op 13439) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[96:103], v[36:39], a[124:127], v129, v125 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4553) 
ds_read_b128 v[116:119], v6 offset:41984 // (op 324) Load local data 





ds_read_b128 v[120:123], v6 offset:43008 // (op 460) Load local data 
buffer_load_dwordx2 v[22:23], v20, s[64:67], 0 offen // (op 13439) Load value






v_add_u32 v20, 8, v10 // (op 13452) 

ds_read_b128 v[124:127], v6 offset:44032 // (op 596) Load local data 
buffer_load_dwordx2 v[26:27], v20, s[64:67], 0 offen // (op 13452) Load value


v_add_u32 v21, 8, v21 // (op 13967) 
s_waitcnt vmcnt(13)
s_barrier  // (op 14318) 



ds_read_b128 v[40:43], v5 offset:49152 // (op 214) Load local data 
ds_read_b128 v[44:47], v5 offset:49216 // (op 214) Load local data 



ds_read_b128 v[48:51], v5 offset:51200 // (op 901) Load local data 
ds_read_b128 v[52:55], v5 offset:51264 // (op 901) Load local data 



ds_read_b128 v[56:59], v5 offset:53248 // (op 1511) Load local data 
ds_read_b128 v[60:63], v5 offset:53312 // (op 1511) Load local data 



ds_read_b128 v[64:67], v5 offset:55296 // (op 2121) Load local data 
ds_read_b128 v[68:71], v5 offset:55360 // (op 2121) Load local data 



ds_read_b128 v[72:75], v5 offset:57344 // (op 2731) Load local data 
ds_read_b128 v[76:79], v5 offset:57408 // (op 2731) Load local data 



ds_read_b128 v[80:83], v5 offset:59392 // (op 3341) Load local data 
ds_read_b128 v[84:87], v5 offset:59456 // (op 3341) Load local data 



ds_read_b128 v[88:91], v5 offset:61440 // (op 3951) Load local data 
ds_read_b128 v[92:95], v5 offset:61504 // (op 3951) Load local data 



ds_read_b128 v[134:137], v5 offset:63488 // (op 4561) Load local data 
ds_read_b128 v[138:141], v5 offset:63552 // (op 4561) Load local data 
v_bfe_u32 v142, v111, 0, 8 // (op 12399) 
v_bfe_u32 v143, v111, 8, 8 // (op 12399) 
v_bfe_u32 v144, v111, 16, 8 // (op 12399) 
v_bfe_u32 v145, v111, 24, 8 // (op 12399) 
v_permlane16_swap_b32 v142, v143 // (op 12399) 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 14328) 
v_permlane16_swap_b32 v144, v145 // (op 12399) 

s_lshl_b64 s[22:23], s[30:31], 6 // (op 13508) 
s_lshl_b64 s[24:25], s[22:23], 2 // (op 13508) 

v_permlane32_swap_b32 v142, v144 // (op 12399) 
v_permlane32_swap_b32 v143, v145 // (op 12399) 
v_lshrrev_b64 v[112:113], 3, s[24:25] // (op 13508) 



v_mov_b64 v[110:111], v[108:109] // (op 13508) 
s_add_u32 s22, 40960, s18 // (op 13508) 
s_mov_b32 m0, s22 // (op 13508) reg expression
s_mov_b32 s23, 192 // (op 13508) call()
buffer_load_dwordx4 v110, s[60:63], s23 offen offset: 0 lds // (op 13508) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v112 // (op 13508) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 13508) most significant half

s_add_u32 m0, m0, 4096 // (op 13508) 
buffer_load_dwordx4 v110, s[60:63], s23 offen offset: 0 lds // (op 13508) Load value direct to lds




v_bfe_u32 v110, v105, 0, 8 // (op 12526) 

v_bfe_u32 v111, v105, 8, 8 // (op 12526) 
v_bfe_u32 v112, v105, 16, 8 // (op 12526) 
v_bfe_u32 v113, v105, 24, 8 // (op 12526) 
v_permlane16_swap_b32 v110, v111 // (op 12526) 
v_add_co_u32 v18, vcc, 128, v18 // (op 13670) least significant half
v_permlane16_swap_b32 v112, v113 // (op 12526) 
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 13670) most significant half
s_nop 0
v_permlane32_swap_b32 v110, v112 // (op 12526) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v111, v113 // (op 12526) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 14321) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[130:133], a[0:3], v142, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 221) 
s_lshl_b64 s[22:23], s[26:27], 5 // (op 13503) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[40:47], v[116:119], a[4:7], v142, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 316) 
v_mov_b64 v[104:105], s[22:23] // (op 13503) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[40:47], v[120:123], a[8:11], v142, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 452) 
v_mov_b64 v[108:109], v[114:115] // (op 13503) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[40:47], v[124:127], a[12:15], v142, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 588) 
s_add_u32 s22, 49152, s1 // (op 13503) 
s_mov_b32 m0, s22 // (op 13503) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[48:55], v[130:133], a[16:19], v143, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1008) 
s_mov_b32 s23, 384 // (op 13503) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[48:55], v[116:119], a[20:23], v143, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 970) 
buffer_load_dwordx4 v108, s[48:51], s23 offen offset: 0 lds // (op 13503) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[48:55], v[120:123], a[24:27], v143, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 932) 
v_add_co_u32 v108, s[24:25], v108, v104 // (op 13503) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:55], v[124:127], a[28:31], v143, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 894) 
v_addc_co_u32 v109, s[24:25], 0, v109, s[24:25] // (op 13503) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[56:63], v[130:133], a[32:35], v144, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1618) 
s_add_u32 m0, m0, 4096 // (op 13503) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[56:63], v[116:119], a[36:39], v144, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1580) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[56:63], v[120:123], a[40:43], v144, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1542) 
v_bfe_u32 v146, v107, 0, 8 // (op 12430) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[56:63], v[124:127], a[44:47], v144, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1504) 
v_bfe_u32 v147, v107, 8, 8 // (op 12430) 
v_bfe_u32 v148, v107, 16, 8 // (op 12430) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[64:71], v[130:133], a[48:51], v145, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2228) 
v_bfe_u32 v149, v107, 24, 8 // (op 12430) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[64:71], v[116:119], a[52:55], v145, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2190) 
v_permlane16_swap_b32 v146, v147 // (op 12430) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[64:71], v[120:123], a[56:59], v145, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2152) 
v_permlane16_swap_b32 v148, v149 // (op 12430) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[64:71], v[124:127], a[60:63], v145, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2114) 
s_nop 0
v_permlane32_swap_b32 v146, v148 // (op 12430) Wait state hazard: VALU Write Readlane Hazard

v_permlane32_swap_b32 v147, v149 // (op 12430) 
s_waitcnt vmcnt(14)
s_barrier  // (op 14311) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[72:79], v[130:133], a[64:67], v146, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2838) 
ds_read_b128 v[12:15], v4 // (op 67) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[72:79], v[116:119], a[68:71], v146, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2800) 
buffer_load_dwordx4 v108, s[48:51], s23 offen offset: 0 lds // (op 13503) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:79], v[120:123], a[72:75], v146, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2762) 
v_add_co_u32 v108, s[24:25], v108, v104 // (op 13503) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[72:79], v[124:127], a[76:79], v146, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2724) 
v_addc_co_u32 v109, s[24:25], 0, v109, s[24:25] // (op 13503) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[80:87], v[130:133], a[80:83], v147, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3448) 
s_add_u32 m0, m0, 4096 // (op 13503) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[80:87], v[116:119], a[84:87], v147, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3410) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[80:87], v[120:123], a[88:91], v147, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3372) 
buffer_load_dwordx4 v108, s[48:51], s23 offen offset: 0 lds // (op 13503) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[80:87], v[124:127], a[92:95], v147, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3334) 
v_add_co_u32 v108, s[24:25], v108, v104 // (op 13503) least significant half
v_addc_co_u32 v109, s[24:25], 0, v109, s[24:25] // (op 13503) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[88:95], v[130:133], a[96:99], v148, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4058) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[88:95], v[116:119], a[100:103], v148, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4020) 
s_add_u32 m0, m0, 4096 // (op 13503) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[88:95], v[120:123], a[104:107], v148, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3982) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[88:95], v[124:127], a[108:111], v148, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3944) 
buffer_load_dwordx4 v108, s[48:51], s23 offen offset: 0 lds // (op 13503) Load value direct to lds
v_add_co_u32 v108, s[24:25], v108, v104 // (op 13503) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[134:141], v[130:133], a[112:115], v149, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4668) 
v_addc_co_u32 v109, s[24:25], 0, v109, s[24:25] // (op 13503) most significant half





v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[134:141], v[116:119], a[116:119], v149, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4630) 
s_add_u32 m0, m0, 4096 // (op 13503) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[134:141], v[120:123], a[120:123], v149, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4592) 
ds_read_b128 v[28:31], v4 offset:1024 // (op 312) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[134:141], v[124:127], a[124:127], v149, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4554) 
ds_read_b128 v[32:35], v4 offset:2048 // (op 448) Load local data 







ds_read_b128 v[36:39], v4 offset:3072 // (op 584) Load local data 
buffer_load_dwordx4 v108, s[48:51], s23 offen offset: 0 lds // (op 13503) Load value direct to lds
v_add_co_u32 v108, s[24:25], v108, v104 // (op 13503) least significant half
v_addc_co_u32 v109, s[24:25], 0, v109, s[24:25] // (op 13503) most significant half

s_add_u32 m0, m0, 4096 // (op 13503) 
s_waitcnt vmcnt(10)
s_barrier  // (op 14295) 
buffer_load_dwordx4 v108, s[48:51], s23 offen offset: 0 lds // (op 13503) Load value direct to lds



ds_read_b128 v[40:43], v3 offset:8192 // (op 62) Load local data 
ds_read_b128 v[44:47], v3 offset:8256 // (op 62) Load local data 
v_add_co_u32 v108, s[24:25], v108, v104 // (op 13503) least significant half



ds_read_b128 v[48:51], v3 offset:10240 // (op 889) Load local data 
v_addc_co_u32 v109, s[24:25], 0, v109, s[24:25] // (op 13503) most significant half

s_add_u32 m0, m0, 4096 // (op 13503) 
buffer_load_dwordx4 v108, s[48:51], s23 offen offset: 0 lds // (op 13503) Load value direct to lds
v_add_co_u32 v108, s[24:25], v108, v104 // (op 13503) least significant half
v_addc_co_u32 v109, s[24:25], 0, v109, s[24:25] // (op 13503) most significant half

s_add_u32 m0, m0, 4096 // (op 13503) 
ds_read_b128 v[52:55], v3 offset:10304 // (op 889) Load local data 
buffer_load_dwordx4 v108, s[48:51], s23 offen offset: 0 lds // (op 13503) Load value direct to lds








ds_read_b128 v[56:59], v3 offset:12288 // (op 1499) Load local data 
ds_read_b128 v[60:63], v3 offset:12352 // (op 1499) Load local data 
v_add_co_u32 v16, vcc, 256, v16 // (op 13606) least significant half



ds_read_b128 v[64:67], v3 offset:14336 // (op 2109) Load local data 
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 13606) most significant half
ds_read_b128 v[68:71], v3 offset:14400 // (op 2109) Load local data 



ds_read_b128 v[72:75], v3 offset:16384 // (op 2719) Load local data 
ds_read_b128 v[76:79], v3 offset:16448 // (op 2719) Load local data 



ds_read_b128 v[80:83], v3 offset:18432 // (op 3329) Load local data 
ds_read_b128 v[84:87], v3 offset:18496 // (op 3329) Load local data 



ds_read_b128 v[88:91], v3 offset:20480 // (op 3939) Load local data 
ds_read_b128 v[92:95], v3 offset:20544 // (op 3939) Load local data 



ds_read_b128 v[96:99], v3 offset:22528 // (op 4549) Load local data 
ds_read_b128 v[100:103], v3 offset:22592 // (op 4549) Load local data 
s_ashr_i64 s[22:23], s[42:43], 7 // (op 3299) 
s_mov_b32 s24, s22 // (op 3299) convert

s_and_b32 s25, s24, 4294967294 // (op 3299) 

s_cmp_lt_u32 s19, s25 // (op 3299) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopTop_KLoop_3299 // (op 3299) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopTop_KLoop_3299 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopBottom_KLoop_3299:















s_barrier  // (op 14390) 
s_mul_hi_i32 s47, s16, s10 // (op 13538) 
v_lshrrev_b32 v10, 3, v1 // (op 13538) 
s_add_i32 s0, s47, s16 // (op 13538) 

v_and_b32 v11, -8, v1 // (op 13538) 
s_ashr_i32 s1, s0, 31 // (op 13538) 
v_sub_u32 v12, v1, v11 // (op 13538) 

s_cmp_eq_i32 s10, 0 // (op 13538) 
s_mov_b32 s47, scc // (op 13538) 
v_lshlrev_b32 v11, 4, v12 // (op 13538) 

s_cmp_eq_i32 s47, 0 // (op 13538) 
s_cselect_b32 s18, 0, -1 // (op 13538) 

s_and_b32 s19, s13, 31 // (op 13538) 
s_lshl_b32 s47, 1, s19 // (op 13538) 
s_add_i32 s22, s47, s18 // (op 13538) 


s_and_b32 s18, s1, s22 // (op 13538) 


s_add_i32 s1, s0, s18 // (op 13538) 


s_ashr_i32 s0, s1, s19 // (op 13538) 


s_lshl_b32 s1, s13, 24 // (op 13538) 
s_ashr_i32 s47, s1, 31 // (op 13538) 

s_xor_b32 s1, s0, s47 // (op 13538) 

s_sub_i32 s0, s1, s47 // (op 13538) 


s_mul_hi_i32 s1, s0, s14 // (op 13538) 
s_add_i32 s47, s1, s0 // (op 13538) 

s_ashr_i32 s1, s47, 31 // (op 13538) 
s_cmp_eq_i32 s14, 0 // (op 13538) 
s_mov_b32 s18, scc // (op 13538) 
s_cmp_eq_i32 s18, 0 // (op 13538) 
s_cselect_b32 s19, 0, -1 // (op 13538) 

s_and_b32 s18, s15, 31 // (op 13538) 
s_lshl_b32 s22, 1, s18 // (op 13538) 
s_add_i32 s23, s22, s19 // (op 13538) 


s_and_b32 s19, s1, s23 // (op 13538) 


s_add_i32 s1, s47, s19 // (op 13538) 


s_ashr_i32 s19, s1, s18 // (op 13538) 


s_lshl_b32 s1, s15, 24 // (op 13538) 
s_ashr_i32 s18, s1, 31 // (op 13538) 

s_xor_b32 s1, s19, s18 // (op 13538) 

s_sub_i32 s19, s1, s18 // (op 13538) 


s_mul_i32 s1, s0, s11 // (op 13538) 

s_sub_u32 s0, s16, s1 // (op 13538) 

s_mul_i32 s1, s19, s11 // (op 13538) 
s_add_u32 s1, s1, s0 // (op 13538) 


s_mul_hi_i32 s0, s16, s9 // (op 13538) 
s_add_i32 s47, s0, s16 // (op 13538) 

s_ashr_i32 s0, s47, 31 // (op 13538) 
s_cmp_eq_i32 s9, 0 // (op 13538) 
s_mov_b32 s18, scc // (op 13538) 
s_cmp_eq_i32 s18, 0 // (op 13538) 
s_cselect_b32 s19, 0, -1 // (op 13538) 

s_and_b32 s18, s45, 31 // (op 13538) 
s_lshl_b32 s22, 1, s18 // (op 13538) 
s_add_i32 s23, s22, s19 // (op 13538) 


s_and_b32 s19, s0, s23 // (op 13538) 


s_add_i32 s0, s47, s19 // (op 13538) 


s_ashr_i32 s19, s0, s18 // (op 13538) 


s_lshl_b32 s0, s45, 24 // (op 13538) 
s_ashr_i32 s18, s0, 31 // (op 13538) 

s_xor_b32 s0, s19, s18 // (op 13538) 

s_sub_i32 s19, s0, s18 // (op 13538) 


s_add_u32 s22, s6, 128 // (op 13538) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13538) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13538) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13538) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13538) 

s_mul_i32 s0, s11, s22 // (op 13538) 

s_mul_i32 s18, s8, s0 // (op 13538) 

s_mul_i32 s0, s19, s18 // (op 13538) 

s_sub_u32 s18, s16, s0 // (op 13538) 

s_mov_b32 s0, s18 // (op 13538) 
s_mul_hi_i32 s47, s0, s12 // (op 13538) 
s_add_i32 s22, s47, s0 // (op 13538) 


s_ashr_i32 s0, s22, 31 // (op 13538) 
s_cmp_eq_i32 s12, 0 // (op 13538) 
s_mov_b32 s47, scc // (op 13538) 
s_cmp_eq_i32 s47, 0 // (op 13538) 
s_cselect_b32 s23, 0, -1 // (op 13538) 

s_and_b32 s47, s44, 31 // (op 13538) 
s_lshl_b32 s24, 1, s47 // (op 13538) 
s_add_i32 s25, s24, s23 // (op 13538) 


s_and_b32 s23, s0, s25 // (op 13538) 


s_add_i32 s0, s22, s23 // (op 13538) 


s_ashr_i32 s22, s0, s47 // (op 13538) 


s_lshl_b32 s0, s44, 24 // (op 13538) 
s_ashr_i32 s47, s0, 31 // (op 13538) 

s_xor_b32 s0, s22, s47 // (op 13538) 

s_sub_i32 s22, s0, s47 // (op 13538) 


s_mul_i32 s0, s46, s11 // (op 13538) 
s_add_u32 s24, s20, 256 // (op 13538) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13538) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 13538) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 13538) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 13538) 

s_sub_i32 s23, s24, s0 // (op 13538) 

s_mul_i32 s47, s22, s23 // (op 13538) 


s_sub_u32 s22, s18, s47 // (op 13538) 


s_add_u32 s18, s22, s0 // (op 13538) 


s_cmp_eq_i32 s19, 0 // (op 13538) 
s_mov_b32 s0, scc // (op 13538) 

s_cmp_eq_i32 s0, 0 // (op 13538) 
s_cselect_b32 s19, s18, s1 // (op 13538) 



v_lshl_add_u32 v12, s19, 8, v10 // (op 13538) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 13538) 
s_lshr_b32 s47, s0, 1 // (op 13538) 

v_lshl_add_u32 v10, s47, 8, v11 // (op 13538) 


v_mov_b32 v21, 0 // (op 13538) convert
v_mov_b32 v20, v10 // (op 13538) convert

v_mul_lo_u32 v11, v12, s27 // (op 13538) most significant: low of low * high
v_mul_hi_u32 v13, v12, s26 // (op 13538) most significant: high of low * low
v_mul_lo_u32 v10, v12, s26 // (op 13538) least significant: low of low * low
v_add_u32 v11, v11, v13 // (op 13538) most significant: sum

v_add_co_u32 v10, s[0:1], v10, v20 // (op 13538) least significant half
v_addc_co_u32 v11, s[0:1], v11, v21, s[0:1] // (op 13538) most significant half



s_mul_i32 s1, s26, s21 // (op 13540) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 13540) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 13540) least significant
s_mul_hi_u32 s18, s26, s20 // (op 13540) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13540) most significant: sum
s_add_u32 s1, s1, s18 // (op 13540) most significant: sum


s_mov_b32 s48, s28 // (op 13540) reg expression
s_mov_b32 s49, s29 // (op 13540) reg expression
s_mov_b32 s50, s0 // (op 13540) reg expression
s_mov_b32 s51, 131072 // (op 13540) call()

s_mul_hi_i32 s47, s16, s10 // (op 13636) 
v_lshrrev_b32 v12, 2, v1 // (op 13636) 
s_add_i32 s0, s47, s16 // (op 13636) 

v_and_b32 v13, -4, v1 // (op 13636) 
s_ashr_i32 s1, s0, 31 // (op 13636) 
v_sub_u32 v14, v1, v13 // (op 13636) 

s_cmp_eq_i32 s10, 0 // (op 13636) 
s_mov_b32 s47, scc // (op 13636) 
v_lshlrev_b32 v13, 5, v14 // (op 13636) 

s_cmp_eq_i32 s47, 0 // (op 13636) 
s_cselect_b32 s18, 0, -1 // (op 13636) 

s_and_b32 s19, s13, 31 // (op 13636) 
s_lshl_b32 s47, 1, s19 // (op 13636) 
s_add_i32 s22, s47, s18 // (op 13636) 


s_and_b32 s18, s1, s22 // (op 13636) 


s_add_i32 s1, s0, s18 // (op 13636) 


s_ashr_i32 s0, s1, s19 // (op 13636) 


s_lshl_b32 s1, s13, 24 // (op 13636) 
s_ashr_i32 s47, s1, 31 // (op 13636) 

s_xor_b32 s1, s0, s47 // (op 13636) 

s_sub_i32 s0, s1, s47 // (op 13636) 


s_mul_hi_i32 s1, s0, s14 // (op 13636) 
s_add_i32 s47, s1, s0 // (op 13636) 

s_ashr_i32 s1, s47, 31 // (op 13636) 
s_cmp_eq_i32 s14, 0 // (op 13636) 
s_mov_b32 s18, scc // (op 13636) 
s_cmp_eq_i32 s18, 0 // (op 13636) 
s_cselect_b32 s19, 0, -1 // (op 13636) 

s_and_b32 s18, s15, 31 // (op 13636) 
s_lshl_b32 s22, 1, s18 // (op 13636) 
s_add_i32 s23, s22, s19 // (op 13636) 


s_and_b32 s19, s1, s23 // (op 13636) 


s_add_i32 s1, s47, s19 // (op 13636) 


s_ashr_i32 s19, s1, s18 // (op 13636) 


s_lshl_b32 s1, s15, 24 // (op 13636) 
s_ashr_i32 s18, s1, 31 // (op 13636) 

s_xor_b32 s1, s19, s18 // (op 13636) 

s_sub_i32 s19, s1, s18 // (op 13636) 


s_add_u32 s22, s6, 128 // (op 13636) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13636) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13636) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13636) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13636) 

s_mul_i32 s1, s19, s22 // (op 13636) 

s_sub_i32 s47, s0, s1 // (op 13636) 


s_mul_hi_i32 s0, s16, s9 // (op 13636) 
s_add_i32 s1, s0, s16 // (op 13636) 

s_ashr_i32 s0, s1, 31 // (op 13636) 
s_cmp_eq_i32 s9, 0 // (op 13636) 
s_mov_b32 s18, scc // (op 13636) 
s_cmp_eq_i32 s18, 0 // (op 13636) 
s_cselect_b32 s19, 0, -1 // (op 13636) 

s_and_b32 s18, s45, 31 // (op 13636) 
s_lshl_b32 s24, 1, s18 // (op 13636) 
s_add_i32 s25, s24, s19 // (op 13636) 


s_and_b32 s19, s0, s25 // (op 13636) 


s_add_i32 s0, s1, s19 // (op 13636) 


s_ashr_i32 s1, s0, s18 // (op 13636) 


s_lshl_b32 s0, s45, 24 // (op 13636) 
s_ashr_i32 s18, s0, 31 // (op 13636) 

s_xor_b32 s0, s1, s18 // (op 13636) 

s_sub_i32 s1, s0, s18 // (op 13636) 


s_mul_i32 s0, s11, s22 // (op 13636) 

s_mul_i32 s18, s8, s0 // (op 13636) 

s_mul_i32 s0, s1, s18 // (op 13636) 

s_sub_u32 s18, s16, s0 // (op 13636) 

s_mul_hi_i32 s0, s18, s12 // (op 13636) 
s_add_i32 s19, s0, s18 // (op 13636) 


s_ashr_i32 s0, s19, 31 // (op 13636) 
s_cmp_eq_i32 s12, 0 // (op 13636) 
s_mov_b32 s18, scc // (op 13636) 
s_cmp_eq_i32 s18, 0 // (op 13636) 
s_cselect_b32 s22, 0, -1 // (op 13636) 

s_and_b32 s18, s44, 31 // (op 13636) 
s_lshl_b32 s23, 1, s18 // (op 13636) 
s_add_i32 s24, s23, s22 // (op 13636) 


s_and_b32 s25, s0, s24 // (op 13636) 


s_add_i32 s0, s19, s25 // (op 13636) 


s_ashr_i32 s19, s0, s18 // (op 13636) 


s_lshl_b32 s0, s44, 24 // (op 13636) 
s_ashr_i32 s18, s0, 31 // (op 13636) 

s_xor_b32 s0, s19, s18 // (op 13636) 

s_sub_i32 s19, s0, s18 // (op 13636) 


s_cmp_eq_i32 s1, 0 // (op 13636) 
s_mov_b32 s0, scc // (op 13636) 

s_cmp_eq_i32 s0, 0 // (op 13636) 
s_cselect_b32 s1, s19, s47 // (op 13636) 



v_lshl_add_u32 v14, s1, 7, v12 // (op 13636) 


v_mul_lo_u32 v21, v14, s31 // (op 13636) most significant: low of low * high
v_mul_hi_u32 v12, v14, s30 // (op 13636) most significant: high of low * low
v_mul_lo_u32 v20, v14, s30 // (op 13636) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 13636) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 13636) 
s_lshr_b32 s47, s0, 1 // (op 13636) 

v_lshl_add_u32 v12, s47, 8, v13 // (op 13636) 


v_mov_b32 v19, 0 // (op 13636) convert
v_mov_b32 v18, v12 // (op 13636) convert

v_add_co_u32 v12, s[0:1], v18, v20 // (op 13636) least significant half
v_addc_co_u32 v13, s[0:1], v19, v21, s[0:1] // (op 13636) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 13636) 


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 13636) 

s_mul_i32 s1, s30, s7 // (op 13638) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 13638) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 13638) least significant
s_mul_hi_u32 s18, s30, s6 // (op 13638) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13638) most significant: sum
s_add_u32 s1, s1, s18 // (op 13638) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13638) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 13638) 

s_mov_b32 s60, s32 // (op 13638) reg expression
s_mov_b32 s61, s33 // (op 13638) reg expression
s_mov_b32 s62, s0 // (op 13638) reg expression
s_mov_b32 s63, 131072 // (op 13638) call()

v_and_b32 v12, -8, v1 // (op 13734) 
v_sub_u32 v13, v1, v12 // (op 13734) 

v_lshlrev_b32 v12, 4, v13 // (op 13734) 

v_lshrrev_b32 v13, 3, v1 // (op 13734) 
v_lshl_add_u32 v16, v13, 7, v12 // (op 13734) 


s_nop 0
v_readfirstlane_b32 s47, v16 // (op 13734) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -4, v1 // (op 13791) 
v_sub_u32 v13, v1, v12 // (op 13791) 

v_lshlrev_b32 v12, 5, v13 // (op 13791) 

v_lshrrev_b32 v13, 2, v1 // (op 13791) 
v_lshl_add_u32 v16, v13, 7, v12 // (op 13791) 


v_lshlrev_b32 v12, 2, v16 // (op 13791) 

v_lshrrev_b32 v13, 3, v12 // (op 13791) 

s_nop 0
v_readfirstlane_b32 s0, v13 // (op 13791) Wait state hazard: VALU Write Readlane Hazard

s_mul_hi_i32 s1, s16, s10 // (op 14056) 
v_and_b32 v12, 63, v1 // (op 14056) 
s_add_i32 s18, s1, s16 // (op 14056) 

v_lshrrev_b32 v13, 4, v12 // (op 14056) 
s_ashr_i32 s1, s18, 31 // (op 14056) 
v_and_b32 v16, 3, v13 // (op 14056) 

s_cmp_eq_i32 s10, 0 // (op 14056) 
s_mov_b32 s19, scc // (op 14056) 
v_and_b32 v13, 15, v1 // (op 14056) 
s_cmp_eq_i32 s19, 0 // (op 14056) 
s_cselect_b32 s22, 0, -1 // (op 14056) 

v_lshl_add_u32 v17, v16, 4, v13 // (op 14056) 


s_and_b32 s19, s13, 31 // (op 14056) 
v_and_b32 v13, 4294967168, v1 // (op 14056) 
s_lshl_b32 s23, 1, s19 // (op 14056) 
v_add_u32 v16, v13, v17 // (op 14056) 


s_add_i32 s24, s23, s22 // (op 14056) 


v_lshrrev_b32 v13, 6, v12 // (op 14056) 

s_and_b32 s25, s1, s24 // (op 14056) 


v_lshlrev_b32 v12, 2, v13 // (op 14056) 

s_add_i32 s1, s18, s25 // (op 14056) 


s_ashr_i32 s18, s1, s19 // (op 14056) 


s_lshl_b32 s1, s13, 24 // (op 14056) 
s_ashr_i32 s19, s1, 31 // (op 14056) 

s_xor_b32 s1, s18, s19 // (op 14056) 

s_sub_i32 s18, s1, s19 // (op 14056) 


s_mul_hi_i32 s1, s18, s14 // (op 14056) 
s_add_i32 s19, s1, s18 // (op 14056) 

s_ashr_i32 s1, s19, 31 // (op 14056) 
s_cmp_eq_i32 s14, 0 // (op 14056) 
s_mov_b32 s22, scc // (op 14056) 
s_cmp_eq_i32 s22, 0 // (op 14056) 
s_cselect_b32 s23, 0, -1 // (op 14056) 

s_and_b32 s22, s15, 31 // (op 14056) 
s_lshl_b32 s24, 1, s22 // (op 14056) 
s_add_i32 s25, s24, s23 // (op 14056) 


s_and_b32 s23, s1, s25 // (op 14056) 


s_add_i32 s1, s19, s23 // (op 14056) 


s_ashr_i32 s19, s1, s22 // (op 14056) 


s_lshl_b32 s1, s15, 24 // (op 14056) 
s_ashr_i32 s22, s1, 31 // (op 14056) 

s_xor_b32 s1, s19, s22 // (op 14056) 

s_sub_i32 s19, s1, s22 // (op 14056) 


s_mul_i32 s1, s18, s11 // (op 14056) 

s_sub_u32 s18, s16, s1 // (op 14056) 

s_mul_i32 s1, s19, s11 // (op 14056) 
s_add_u32 s1, s1, s18 // (op 14056) 


s_mul_hi_i32 s18, s16, s9 // (op 14056) 
s_add_i32 s19, s18, s16 // (op 14056) 

s_ashr_i32 s18, s19, 31 // (op 14056) 
s_cmp_eq_i32 s9, 0 // (op 14056) 
s_mov_b32 s22, scc // (op 14056) 
s_cmp_eq_i32 s22, 0 // (op 14056) 
s_cselect_b32 s23, 0, -1 // (op 14056) 

s_and_b32 s22, s45, 31 // (op 14056) 
s_lshl_b32 s24, 1, s22 // (op 14056) 
s_add_i32 s25, s24, s23 // (op 14056) 


s_and_b32 s23, s18, s25 // (op 14056) 


s_add_i32 s18, s19, s23 // (op 14056) 


s_ashr_i32 s19, s18, s22 // (op 14056) 


s_lshl_b32 s18, s45, 24 // (op 14056) 
s_ashr_i32 s22, s18, 31 // (op 14056) 

s_xor_b32 s18, s19, s22 // (op 14056) 

s_sub_i32 s19, s18, s22 // (op 14056) 


s_add_u32 s22, s6, 128 // (op 14056) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 14056) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 14056) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 14056) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 14056) 

s_mul_i32 s18, s11, s22 // (op 14056) 

s_mul_i32 s22, s8, s18 // (op 14056) 

s_mul_i32 s18, s19, s22 // (op 14056) 

s_sub_u32 s22, s16, s18 // (op 14056) 

s_mov_b32 s18, s22 // (op 14056) 
s_mul_hi_i32 s23, s18, s12 // (op 14056) 
s_add_i32 s24, s23, s18 // (op 14056) 


s_ashr_i32 s18, s24, 31 // (op 14056) 
s_cmp_eq_i32 s12, 0 // (op 14056) 
s_mov_b32 s23, scc // (op 14056) 
s_cmp_eq_i32 s23, 0 // (op 14056) 
s_cselect_b32 s25, 0, -1 // (op 14056) 

s_and_b32 s23, s44, 31 // (op 14056) 
s_lshl_b32 s64, 1, s23 // (op 14056) 
s_add_i32 s65, s64, s25 // (op 14056) 


s_and_b32 s25, s18, s65 // (op 14056) 


s_add_i32 s18, s24, s25 // (op 14056) 


s_ashr_i32 s24, s18, s23 // (op 14056) 


s_lshl_b32 s18, s44, 24 // (op 14056) 
s_ashr_i32 s23, s18, 31 // (op 14056) 

s_xor_b32 s18, s24, s23 // (op 14056) 

s_sub_i32 s24, s18, s23 // (op 14056) 


s_mul_i32 s18, s46, s11 // (op 14056) 
s_add_u32 s64, s20, 256 // (op 14056) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 14056) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 14056) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 14056) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 8 // (op 14056) 

s_sub_i32 s23, s64, s18 // (op 14056) 

s_mul_i32 s25, s24, s23 // (op 14056) 


s_sub_u32 s23, s22, s25 // (op 14056) 


s_add_u32 s22, s23, s18 // (op 14056) 


s_cmp_eq_i32 s19, 0 // (op 14056) 
s_mov_b32 s18, scc // (op 14056) 

s_cmp_eq_i32 s18, 0 // (op 14056) 
s_cselect_b32 s19, s22, s1 // (op 14056) 



v_lshl_add_u32 v13, s19, 8, v16 // (op 14056) 


s_ashr_i64 s[18:19], s[42:43], 7 // (op 14056) 
s_lshr_b32 s1, s18, 1 // (op 14056) 

v_lshl_add_u32 v16, s1, 3, v12 // (op 14056) 


v_mul_lo_u32 v12, v13, s34 // (op 14056) 
v_add_u32 v12, v12, v16 // (op 14056) 


s_mul_i32 s19, s34, s21 // (op 14058) most significant: low of low * high
s_mul_i32 s1, s35, s20 // (op 14058) most significant: low of high * low
s_mul_i32 s18, s34, s20 // (op 14058) least significant
s_mul_hi_u32 s22, s34, s20 // (op 14058) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 14058) most significant: sum
s_add_u32 s19, s19, s22 // (op 14058) most significant: sum


s_mov_b32 s64, s36 // (op 14058) reg expression
s_mov_b32 s65, s37 // (op 14058) reg expression
s_mov_b32 s66, s18 // (op 14058) reg expression
s_mov_b32 s67, 131072 // (op 14058) call()

s_mul_hi_i32 s1, s16, s10 // (op 14083) 
v_and_b32 v13, 63, v1 // (op 14083) 
s_add_i32 s18, s1, s16 // (op 14083) 

v_lshrrev_b32 v16, 4, v13 // (op 14083) 
s_ashr_i32 s1, s18, 31 // (op 14083) 
v_and_b32 v17, 3, v16 // (op 14083) 

s_cmp_eq_i32 s10, 0 // (op 14083) 
s_mov_b32 s19, scc // (op 14083) 
v_and_b32 v16, 15, v1 // (op 14083) 
s_cmp_eq_i32 s19, 0 // (op 14083) 
s_cselect_b32 s22, 0, -1 // (op 14083) 

v_lshl_add_u32 v18, v17, 4, v16 // (op 14083) 


s_and_b32 s19, s13, 31 // (op 14083) 
v_lshrrev_b32 v16, 6, v1 // (op 14083) 
s_lshl_b32 s23, 1, s19 // (op 14083) 
v_and_b32 v17, 1, v16 // (op 14083) 

s_add_i32 s24, s23, s22 // (op 14083) 


v_lshl_add_u32 v16, v17, 6, v18 // (op 14083) 


s_and_b32 s25, s1, s24 // (op 14083) 


v_lshrrev_b32 v17, 6, v13 // (op 14083) 

s_add_i32 s1, s18, s25 // (op 14083) 


v_lshlrev_b32 v13, 2, v17 // (op 14083) 

s_ashr_i32 s18, s1, s19 // (op 14083) 


s_lshl_b32 s1, s13, 24 // (op 14083) 
s_ashr_i32 s19, s1, 31 // (op 14083) 

s_xor_b32 s1, s18, s19 // (op 14083) 

s_sub_i32 s18, s1, s19 // (op 14083) 


s_mul_hi_i32 s1, s18, s14 // (op 14083) 
s_add_i32 s19, s1, s18 // (op 14083) 

s_ashr_i32 s1, s19, 31 // (op 14083) 
s_cmp_eq_i32 s14, 0 // (op 14083) 
s_mov_b32 s22, scc // (op 14083) 
s_cmp_eq_i32 s22, 0 // (op 14083) 
s_cselect_b32 s23, 0, -1 // (op 14083) 

s_and_b32 s22, s15, 31 // (op 14083) 
s_lshl_b32 s24, 1, s22 // (op 14083) 
s_add_i32 s25, s24, s23 // (op 14083) 


s_and_b32 s23, s1, s25 // (op 14083) 


s_add_i32 s1, s19, s23 // (op 14083) 


s_ashr_i32 s19, s1, s22 // (op 14083) 


s_lshl_b32 s1, s15, 24 // (op 14083) 
s_ashr_i32 s22, s1, 31 // (op 14083) 

s_xor_b32 s1, s19, s22 // (op 14083) 

s_sub_i32 s19, s1, s22 // (op 14083) 


s_add_u32 s22, s6, 128 // (op 14083) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 14083) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 14083) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 14083) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 14083) 

s_mul_i32 s1, s19, s22 // (op 14083) 

s_sub_i32 s19, s18, s1 // (op 14083) 


s_mul_hi_i32 s1, s16, s9 // (op 14083) 
s_add_i32 s18, s1, s16 // (op 14083) 

s_ashr_i32 s1, s18, 31 // (op 14083) 
s_cmp_eq_i32 s9, 0 // (op 14083) 
s_mov_b32 s24, scc // (op 14083) 
s_cmp_eq_i32 s24, 0 // (op 14083) 
s_cselect_b32 s25, 0, -1 // (op 14083) 

s_and_b32 s24, s45, 31 // (op 14083) 
s_lshl_b32 s68, 1, s24 // (op 14083) 
s_add_i32 s69, s68, s25 // (op 14083) 


s_and_b32 s25, s1, s69 // (op 14083) 


s_add_i32 s1, s18, s25 // (op 14083) 


s_ashr_i32 s18, s1, s24 // (op 14083) 


s_lshl_b32 s1, s45, 24 // (op 14083) 
s_ashr_i32 s24, s1, 31 // (op 14083) 

s_xor_b32 s1, s18, s24 // (op 14083) 

s_sub_i32 s18, s1, s24 // (op 14083) 


s_mul_i32 s1, s11, s22 // (op 14083) 

s_mul_i32 s22, s8, s1 // (op 14083) 

s_mul_i32 s1, s18, s22 // (op 14083) 

s_sub_u32 s22, s16, s1 // (op 14083) 

s_mul_hi_i32 s1, s22, s12 // (op 14083) 
s_add_i32 s23, s1, s22 // (op 14083) 


s_ashr_i32 s1, s23, 31 // (op 14083) 
s_cmp_eq_i32 s12, 0 // (op 14083) 
s_mov_b32 s22, scc // (op 14083) 
s_cmp_eq_i32 s22, 0 // (op 14083) 
s_cselect_b32 s24, 0, -1 // (op 14083) 

s_and_b32 s22, s44, 31 // (op 14083) 
s_lshl_b32 s25, 1, s22 // (op 14083) 
s_add_i32 s68, s25, s24 // (op 14083) 


s_and_b32 s24, s1, s68 // (op 14083) 


s_add_i32 s1, s23, s24 // (op 14083) 


s_ashr_i32 s23, s1, s22 // (op 14083) 


s_lshl_b32 s1, s44, 24 // (op 14083) 
s_ashr_i32 s22, s1, 31 // (op 14083) 

s_xor_b32 s1, s23, s22 // (op 14083) 

s_sub_i32 s23, s1, s22 // (op 14083) 


s_cmp_eq_i32 s18, 0 // (op 14083) 
s_mov_b32 s1, scc // (op 14083) 

s_cmp_eq_i32 s1, 0 // (op 14083) 
s_cselect_b32 s18, s23, s19 // (op 14083) 



v_lshl_add_u32 v17, s18, 7, v16 // (op 14083) 


v_mul_lo_u32 v16, v17, s38 // (op 14083) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 14083) 
s_lshr_b32 s1, s18, 1 // (op 14083) 

v_lshl_add_u32 v17, s1, 3, v13 // (op 14083) 


v_add_u32 v13, v17, v16 // (op 14083) 


s_mul_i32 s19, s38, s7 // (op 14085) most significant: low of low * high
s_mul_i32 s1, s39, s6 // (op 14085) most significant: low of high * low
s_mul_i32 s18, s38, s6 // (op 14085) least significant
s_mul_hi_u32 s22, s38, s6 // (op 14085) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 14085) most significant: sum
s_add_u32 s19, s19, s22 // (op 14085) most significant: sum


s_mov_b32 s68, s40 // (op 14085) reg expression
s_mov_b32 s69, s41 // (op 14085) reg expression
s_mov_b32 s70, s18 // (op 14085) reg expression
s_mov_b32 s71, 131072 // (op 14085) call()

s_ashr_i64 s[18:19], s[42:43], 7 // (op 4525) 
s_and_b32 s1, s18, 4294967294 // (op 4525) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 4524) 
s_mov_b32 s22, s18 // (op 4524) convert

s_cmp_lt_u32 s1, s22 // (op 4524) 












s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_4524 // (op 4524) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_4524 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopTop_KLoopTail_4524:
s_barrier  // (op 14292) 
s_add_u32 s1, s1, 1 // (op 4526) 

s_lshl_b64 s[18:19], s[26:27], 5 // (op 13491) 
v_mov_b64 v[16:17], s[18:19] // (op 13491) convert


v_mov_b64 v[18:19], v[10:11] // (op 13491) 
v_mov_b64 v[20:21], v[18:19] // (op 13491) 
s_add_u32 s18, 8192, s47 // (op 13491) 
s_mov_b32 m0, s18 // (op 13491) reg expression
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 13491) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 13491) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 13491) most significant half

s_add_u32 m0, m0, 4096 // (op 13491) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 13491) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 13491) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 13491) most significant half

s_add_u32 m0, m0, 4096 // (op 13491) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 13491) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 13491) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 13491) most significant half

s_add_u32 m0, m0, 4096 // (op 13491) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 13491) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 13491) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 13491) most significant half

s_add_u32 m0, m0, 4096 // (op 13491) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 13491) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 13491) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 13491) most significant half

s_add_u32 m0, m0, 4096 // (op 13491) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 13491) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 13491) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 13491) most significant half

s_add_u32 m0, m0, 4096 // (op 13491) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 13491) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 13491) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 13491) most significant half

s_add_u32 m0, m0, 4096 // (op 13491) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 13491) Load value direct to lds




v_add_co_u32 v10, vcc, 128, v10 // (op 13528) least significant half
v_addc_co_u32 v11, vcc, 0, v11, vcc // (op 13528) most significant half
s_barrier  // (op 14308) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 13497) 
s_lshl_b64 s[22:23], s[18:19], 2 // (op 13497) 

v_lshrrev_b64 v[16:17], 3, s[22:23] // (op 13497) 


v_mov_b64 v[18:19], v[14:15] // (op 13497) 
v_mov_b64 v[20:21], v[18:19] // (op 13497) 
s_mov_b32 m0, s0 // (op 13497) reg expression
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 13497) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 13497) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 13497) most significant half

s_add_u32 m0, m0, 4096 // (op 13497) 
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 13497) Load value direct to lds



v_add_co_u32 v14, s[18:19], 64, v14 // (op 13628) least significant half
v_addc_co_u32 v15, s[18:19], 0, v15, s[18:19] // (op 13628) most significant half

s_waitcnt vmcnt(2)
s_barrier  // (op 14298) 



ds_read_b128 v[16:19], v3 offset:8192 // (op 164) Load local data 
ds_read_b128 v[20:23], v3 offset:8256 // (op 164) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 14314) 



ds_read_b128 v[24:27], v4 // (op 166) Load local data 


v_mov_b32 v28, v12 // (op 168) 
v_mov_b32 v29, v28 // (op 168) reg expression

buffer_load_dwordx2 v[30:31], v29, s[64:67], 0 offen // (op 168) Load value



v_mov_b32 v29, v13 // (op 170) 
v_mov_b32 v32, v29 // (op 170) reg expression

buffer_load_dwordx2 v[34:35], v32, s[68:71], 0 offen // (op 170) Load value


v_add_u32 v13, 4, v13 // (op 14073) 
s_waitcnt vmcnt(1)
v_bfe_u32 v36, v30, 0, 8 // (op 12607) 



v_bfe_u32 v37, v30, 8, 8 // (op 12607) 
v_bfe_u32 v38, v30, 16, 8 // (op 12607) 
v_bfe_u32 v39, v30, 24, 8 // (op 12607) 
v_permlane16_swap_b32 v36, v37 // (op 12607) 
s_nop 0
v_permlane16_swap_b32 v38, v39 // (op 12607) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v36, v38 // (op 12607) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v37, v39 // (op 12607) 
ds_read_b128 v[40:43], v4 offset:1024 // (op 295) Load local data 




ds_read_b128 v[30:33], v4 offset:2048 // (op 431) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v44, v34, 0, 8 // (op 12667) 



v_bfe_u32 v45, v34, 8, 8 // (op 12667) 
v_bfe_u32 v46, v34, 16, 8 // (op 12667) 
v_bfe_u32 v47, v34, 24, 8 // (op 12667) 
v_permlane16_swap_b32 v44, v45 // (op 12667) 
ds_read_b128 v[48:51], v4 offset:3072 // (op 567) Load local data 
v_permlane16_swap_b32 v46, v47 // (op 12667) 



s_nop 1
v_permlane32_swap_b32 v44, v46 // (op 12667) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v45, v47 // (op 12667) 
ds_read_b128 v[52:55], v3 offset:10240 // (op 789) Load local data 

ds_read_b128 v[56:59], v3 offset:10304 // (op 789) Load local data 



s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:23], v[24:27], a[0:3], v36, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 171) 
ds_read_b128 v[60:63], v3 offset:12288 // (op 1399) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[16:23], v[40:43], a[4:7], v36, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 298) 
ds_read_b128 v[64:67], v3 offset:12352 // (op 1399) Load local data 



s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[16:23], v[30:33], a[8:11], v36, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 434) 
ds_read_b128 v[68:71], v3 offset:14336 // (op 2009) Load local data 
ds_read_b128 v[72:75], v3 offset:14400 // (op 2009) Load local data 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[16:23], v[48:51], a[12:15], v36, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 570) 




ds_read_b128 v[16:19], v3 offset:16384 // (op 2619) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[52:59], v[24:27], a[16:19], v37, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 793) 
ds_read_b128 v[20:23], v3 offset:16448 // (op 2619) Load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[52:59], v[30:33], a[24:27], v37, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 847) 
s_lshl_b32 s18, s34, 6 // (op 2628) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[52:59], v[48:51], a[28:31], v37, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 874) 
v_add_u32 v29, v28, s18 // (op 2628) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[52:59], v[40:43], a[20:23], v37, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 820) 
buffer_load_dwordx2 v[34:35], v29, s[64:67], 0 offen // (op 2628) Load value






v_add_u32 v12, 4, v12 // (op 14046) 
ds_read_b128 v[52:55], v3 offset:18432 // (op 3229) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[60:67], v[24:27], a[32:35], v38, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1403) 
s_waitcnt vmcnt(0)
v_bfe_u32 v76, v34, 0, 8 // (op 12621) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[60:67], v[30:33], a[40:43], v38, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1457) 
v_bfe_u32 v77, v34, 8, 8 // (op 12621) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[60:67], v[48:51], a[44:47], v38, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1484) 
v_bfe_u32 v78, v34, 16, 8 // (op 12621) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[60:67], v[40:43], a[36:39], v38, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1430) 
v_bfe_u32 v79, v34, 24, 8 // (op 12621) 

v_permlane16_swap_b32 v76, v77 // (op 12621) 
s_nop 0
v_permlane16_swap_b32 v78, v79 // (op 12621) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v76, v78 // (op 12621) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v77, v79 // (op 12621) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:75], v[24:27], a[48:51], v39, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2013) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:75], v[30:33], a[56:59], v39, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2067) 
ds_read_b128 v[56:59], v3 offset:18496 // (op 3229) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:75], v[48:51], a[60:63], v39, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2094) 
ds_read_b128 v[60:63], v3 offset:20480 // (op 3839) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:75], v[40:43], a[52:55], v39, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2040) 
ds_read_b128 v[64:67], v3 offset:20544 // (op 3839) Load local data 





s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[16:23], v[40:43], a[68:71], v76, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2650) 
ds_read_b128 v[68:71], v3 offset:22528 // (op 4449) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[16:23], v[24:27], a[64:67], v76, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2623) 
ds_read_b128 v[72:75], v3 offset:22592 // (op 4449) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[16:23], v[30:33], a[72:75], v76, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2677) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[16:23], v[48:51], a[76:79], v76, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2704) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[52:59], v[40:43], a[84:87], v77, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3260) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[52:59], v[24:27], a[80:83], v77, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3233) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[52:59], v[30:33], a[88:91], v77, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3287) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[52:59], v[48:51], a[92:95], v77, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3314) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[60:67], v[40:43], a[100:103], v78, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3870) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[60:67], v[24:27], a[96:99], v78, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3843) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[60:67], v[30:33], a[104:107], v78, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3897) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[60:67], v[48:51], a[108:111], v78, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3924) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[68:75], v[40:43], a[116:119], v79, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4480) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[68:75], v[24:27], a[112:115], v79, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4453) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[68:75], v[30:33], a[120:123], v79, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4507) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[68:75], v[48:51], a[124:127], v79, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4534) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 4524) 
s_mov_b32 s22, s18 // (op 4524) convert

s_cmp_lt_u32 s1, s22 // (op 4524) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopTop_KLoopTail_4524 // (op 4524) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopTop_KLoopTail_4524 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_4524:




















s_barrier  // (op 14412) 

















v_cmp_eq_f32 vcc, 0.00000, v0 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ConditionalFalse_0_beta_116
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



v_accvgpr_read v10, a4 // (op 5650) 
v_accvgpr_read v11, a5 // (op 5650) 
v_accvgpr_read v12, a6 // (op 5650) 
v_accvgpr_read v13, a7 // (op 5650) 
v_mul_f32 v14, v7, v10 // (op 5650) 
v_mul_f32 v15, v7, v11 // (op 5650) 
v_mul_f32 v16, v7, v12 // (op 5650) 
v_mul_f32 v17, v7, v13 // (op 5650) 

v_cvt_f16_f32 v10, v14 // (op 5652) 
v_cvt_f16_f32 v11, v15 // (op 5652) 
v_cvt_f16_f32 v12, v16 // (op 5652) 
v_cvt_f16_f32 v13, v17 // (op 5652) 


v_pack_b32_f16 v4, v10, v11 // (op 5653) 
v_pack_b32_f16 v5, v12, v13 // (op 5653) 

s_lshl_b32 s0, s4, 5 // (op 5653) 
v_add_u32 v3, v9, s0 // (op 5653) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5653) Store value



v_accvgpr_read v10, a8 // (op 5664) 
v_accvgpr_read v11, a9 // (op 5664) 
v_accvgpr_read v12, a10 // (op 5664) 
v_accvgpr_read v13, a11 // (op 5664) 
v_mul_f32 v14, v7, v10 // (op 5664) 
v_mul_f32 v15, v7, v11 // (op 5664) 
v_mul_f32 v16, v7, v12 // (op 5664) 
v_mul_f32 v17, v7, v13 // (op 5664) 

v_cvt_f16_f32 v10, v14 // (op 5666) 
v_cvt_f16_f32 v11, v15 // (op 5666) 
v_cvt_f16_f32 v12, v16 // (op 5666) 
v_cvt_f16_f32 v13, v17 // (op 5666) 


v_pack_b32_f16 v4, v10, v11 // (op 5667) 
v_pack_b32_f16 v5, v12, v13 // (op 5667) 

s_lshl_b32 s0, s4, 6 // (op 5667) 
v_add_u32 v3, v9, s0 // (op 5667) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5667) Store value



v_accvgpr_read v10, a12 // (op 5678) 
v_accvgpr_read v11, a13 // (op 5678) 
v_accvgpr_read v12, a14 // (op 5678) 
v_accvgpr_read v13, a15 // (op 5678) 
v_mul_f32 v14, v7, v10 // (op 5678) 
v_mul_f32 v15, v7, v11 // (op 5678) 
v_mul_f32 v16, v7, v12 // (op 5678) 
v_mul_f32 v17, v7, v13 // (op 5678) 

v_cvt_f16_f32 v10, v14 // (op 5680) 
v_cvt_f16_f32 v11, v15 // (op 5680) 
v_cvt_f16_f32 v12, v16 // (op 5680) 
v_cvt_f16_f32 v13, v17 // (op 5680) 


v_pack_b32_f16 v4, v10, v11 // (op 5681) 
v_pack_b32_f16 v5, v12, v13 // (op 5681) 

s_lshl_b32 s0, s4, 5 // (op 5681) 
s_mul_i32 s1, 3, s0 // (op 5681) 

v_add_u32 v3, v9, s1 // (op 5681) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5681) Store value



v_accvgpr_read v10, a16 // (op 5714) 
v_accvgpr_read v11, a17 // (op 5714) 
v_accvgpr_read v12, a18 // (op 5714) 
v_accvgpr_read v13, a19 // (op 5714) 
v_mul_f32 v14, v7, v10 // (op 5714) 
v_mul_f32 v15, v7, v11 // (op 5714) 
v_mul_f32 v16, v7, v12 // (op 5714) 
v_mul_f32 v17, v7, v13 // (op 5714) 

v_cvt_f16_f32 v10, v14 // (op 5716) 
v_cvt_f16_f32 v11, v15 // (op 5716) 
v_cvt_f16_f32 v12, v16 // (op 5716) 
v_cvt_f16_f32 v13, v17 // (op 5716) 


v_pack_b32_f16 v4, v10, v11 // (op 5718) 
v_pack_b32_f16 v5, v12, v13 // (op 5718) 

v_add_u32 v3, 32, v9 // (op 5718) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5718) Store value



v_accvgpr_read v10, a20 // (op 5728) 
v_accvgpr_read v11, a21 // (op 5728) 
v_accvgpr_read v12, a22 // (op 5728) 
v_accvgpr_read v13, a23 // (op 5728) 
v_mul_f32 v14, v7, v10 // (op 5728) 
v_mul_f32 v15, v7, v11 // (op 5728) 
v_mul_f32 v16, v7, v12 // (op 5728) 
v_mul_f32 v17, v7, v13 // (op 5728) 

v_cvt_f16_f32 v10, v14 // (op 5730) 
v_cvt_f16_f32 v11, v15 // (op 5730) 
v_cvt_f16_f32 v12, v16 // (op 5730) 
v_cvt_f16_f32 v13, v17 // (op 5730) 


v_pack_b32_f16 v4, v10, v11 // (op 5720) 
v_pack_b32_f16 v5, v12, v13 // (op 5720) 

s_lshl_b32 s0, s4, 5 // (op 5720) 
s_add_u32 s0, s0, 32 // (op 5720) 
v_add_u32 v3, v9, s0 // (op 5720) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5720) Store value



v_accvgpr_read v10, a24 // (op 5734) 
v_accvgpr_read v11, a25 // (op 5734) 
v_accvgpr_read v12, a26 // (op 5734) 
v_accvgpr_read v13, a27 // (op 5734) 
v_mul_f32 v14, v7, v10 // (op 5734) 
v_mul_f32 v15, v7, v11 // (op 5734) 
v_mul_f32 v16, v7, v12 // (op 5734) 
v_mul_f32 v17, v7, v13 // (op 5734) 

v_cvt_f16_f32 v10, v14 // (op 5736) 
v_cvt_f16_f32 v11, v15 // (op 5736) 
v_cvt_f16_f32 v12, v16 // (op 5736) 
v_cvt_f16_f32 v13, v17 // (op 5736) 


v_pack_b32_f16 v4, v10, v11 // (op 5722) 
v_pack_b32_f16 v5, v12, v13 // (op 5722) 

s_lshl_b32 s0, s4, 6 // (op 5722) 
s_add_u32 s0, s0, 32 // (op 5722) 
v_add_u32 v3, v9, s0 // (op 5722) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5722) Store value



v_accvgpr_read v10, a28 // (op 5740) 
v_accvgpr_read v11, a29 // (op 5740) 
v_accvgpr_read v12, a30 // (op 5740) 
v_accvgpr_read v13, a31 // (op 5740) 
v_mul_f32 v14, v7, v10 // (op 5740) 
v_mul_f32 v15, v7, v11 // (op 5740) 
v_mul_f32 v16, v7, v12 // (op 5740) 
v_mul_f32 v17, v7, v13 // (op 5740) 

v_cvt_f16_f32 v10, v14 // (op 5742) 
v_cvt_f16_f32 v11, v15 // (op 5742) 
v_cvt_f16_f32 v12, v16 // (op 5742) 
v_cvt_f16_f32 v13, v17 // (op 5742) 


v_pack_b32_f16 v4, v10, v11 // (op 5724) 
v_pack_b32_f16 v5, v12, v13 // (op 5724) 

s_lshl_b32 s0, s4, 5 // (op 5724) 
s_mul_i32 s1, 3, s0 // (op 5724) 
s_add_u32 s1, s1, 32 // (op 5724) 

v_add_u32 v3, v9, s1 // (op 5724) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5724) Store value



v_accvgpr_read v10, a32 // (op 5790) 
v_accvgpr_read v11, a33 // (op 5790) 
v_accvgpr_read v12, a34 // (op 5790) 
v_accvgpr_read v13, a35 // (op 5790) 
v_mul_f32 v14, v7, v10 // (op 5790) 
v_mul_f32 v15, v7, v11 // (op 5790) 
v_mul_f32 v16, v7, v12 // (op 5790) 
v_mul_f32 v17, v7, v13 // (op 5790) 

v_cvt_f16_f32 v10, v14 // (op 5792) 
v_cvt_f16_f32 v11, v15 // (op 5792) 
v_cvt_f16_f32 v12, v16 // (op 5792) 
v_cvt_f16_f32 v13, v17 // (op 5792) 


v_pack_b32_f16 v4, v10, v11 // (op 5794) 
v_pack_b32_f16 v5, v12, v13 // (op 5794) 

v_add_u32 v3, 64, v9 // (op 5794) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5794) Store value



v_accvgpr_read v10, a36 // (op 5804) 
v_accvgpr_read v11, a37 // (op 5804) 
v_accvgpr_read v12, a38 // (op 5804) 
v_accvgpr_read v13, a39 // (op 5804) 
v_mul_f32 v14, v7, v10 // (op 5804) 
v_mul_f32 v15, v7, v11 // (op 5804) 
v_mul_f32 v16, v7, v12 // (op 5804) 
v_mul_f32 v17, v7, v13 // (op 5804) 

v_cvt_f16_f32 v10, v14 // (op 5806) 
v_cvt_f16_f32 v11, v15 // (op 5806) 
v_cvt_f16_f32 v12, v16 // (op 5806) 
v_cvt_f16_f32 v13, v17 // (op 5806) 


v_pack_b32_f16 v4, v10, v11 // (op 5796) 
v_pack_b32_f16 v5, v12, v13 // (op 5796) 

s_lshl_b32 s0, s4, 5 // (op 5796) 
s_add_u32 s0, s0, 64 // (op 5796) 
v_add_u32 v3, v9, s0 // (op 5796) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5796) Store value



v_accvgpr_read v10, a40 // (op 5810) 
v_accvgpr_read v11, a41 // (op 5810) 
v_accvgpr_read v12, a42 // (op 5810) 
v_accvgpr_read v13, a43 // (op 5810) 
v_mul_f32 v14, v7, v10 // (op 5810) 
v_mul_f32 v15, v7, v11 // (op 5810) 
v_mul_f32 v16, v7, v12 // (op 5810) 
v_mul_f32 v17, v7, v13 // (op 5810) 

v_cvt_f16_f32 v10, v14 // (op 5812) 
v_cvt_f16_f32 v11, v15 // (op 5812) 
v_cvt_f16_f32 v12, v16 // (op 5812) 
v_cvt_f16_f32 v13, v17 // (op 5812) 


v_pack_b32_f16 v4, v10, v11 // (op 5798) 
v_pack_b32_f16 v5, v12, v13 // (op 5798) 

s_lshl_b32 s0, s4, 6 // (op 5798) 
s_add_u32 s0, s0, 64 // (op 5798) 
v_add_u32 v3, v9, s0 // (op 5798) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5798) Store value



v_accvgpr_read v10, a44 // (op 5816) 
v_accvgpr_read v11, a45 // (op 5816) 
v_accvgpr_read v12, a46 // (op 5816) 
v_accvgpr_read v13, a47 // (op 5816) 
v_mul_f32 v14, v7, v10 // (op 5816) 
v_mul_f32 v15, v7, v11 // (op 5816) 
v_mul_f32 v16, v7, v12 // (op 5816) 
v_mul_f32 v17, v7, v13 // (op 5816) 

v_cvt_f16_f32 v10, v14 // (op 5818) 
v_cvt_f16_f32 v11, v15 // (op 5818) 
v_cvt_f16_f32 v12, v16 // (op 5818) 
v_cvt_f16_f32 v13, v17 // (op 5818) 


v_pack_b32_f16 v4, v10, v11 // (op 5800) 
v_pack_b32_f16 v5, v12, v13 // (op 5800) 

s_lshl_b32 s0, s4, 5 // (op 5800) 
s_mul_i32 s1, 3, s0 // (op 5800) 
s_add_u32 s1, s1, 64 // (op 5800) 

v_add_u32 v3, v9, s1 // (op 5800) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5800) Store value



v_accvgpr_read v10, a48 // (op 5866) 
v_accvgpr_read v11, a49 // (op 5866) 
v_accvgpr_read v12, a50 // (op 5866) 
v_accvgpr_read v13, a51 // (op 5866) 
v_mul_f32 v14, v7, v10 // (op 5866) 
v_mul_f32 v15, v7, v11 // (op 5866) 
v_mul_f32 v16, v7, v12 // (op 5866) 
v_mul_f32 v17, v7, v13 // (op 5866) 

v_cvt_f16_f32 v10, v14 // (op 5868) 
v_cvt_f16_f32 v11, v15 // (op 5868) 
v_cvt_f16_f32 v12, v16 // (op 5868) 
v_cvt_f16_f32 v13, v17 // (op 5868) 


v_pack_b32_f16 v4, v10, v11 // (op 5870) 
v_pack_b32_f16 v5, v12, v13 // (op 5870) 

v_add_u32 v3, 96, v9 // (op 5870) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5870) Store value



v_accvgpr_read v10, a52 // (op 5880) 
v_accvgpr_read v11, a53 // (op 5880) 
v_accvgpr_read v12, a54 // (op 5880) 
v_accvgpr_read v13, a55 // (op 5880) 
v_mul_f32 v14, v7, v10 // (op 5880) 
v_mul_f32 v15, v7, v11 // (op 5880) 
v_mul_f32 v16, v7, v12 // (op 5880) 
v_mul_f32 v17, v7, v13 // (op 5880) 

v_cvt_f16_f32 v10, v14 // (op 5882) 
v_cvt_f16_f32 v11, v15 // (op 5882) 
v_cvt_f16_f32 v12, v16 // (op 5882) 
v_cvt_f16_f32 v13, v17 // (op 5882) 


v_pack_b32_f16 v4, v10, v11 // (op 5872) 
v_pack_b32_f16 v5, v12, v13 // (op 5872) 

s_lshl_b32 s0, s4, 5 // (op 5872) 
s_add_u32 s0, s0, 96 // (op 5872) 
v_add_u32 v3, v9, s0 // (op 5872) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5872) Store value



v_accvgpr_read v10, a56 // (op 5886) 
v_accvgpr_read v11, a57 // (op 5886) 
v_accvgpr_read v12, a58 // (op 5886) 
v_accvgpr_read v13, a59 // (op 5886) 
v_mul_f32 v14, v7, v10 // (op 5886) 
v_mul_f32 v15, v7, v11 // (op 5886) 
v_mul_f32 v16, v7, v12 // (op 5886) 
v_mul_f32 v17, v7, v13 // (op 5886) 

v_cvt_f16_f32 v10, v14 // (op 5888) 
v_cvt_f16_f32 v11, v15 // (op 5888) 
v_cvt_f16_f32 v12, v16 // (op 5888) 
v_cvt_f16_f32 v13, v17 // (op 5888) 


v_pack_b32_f16 v4, v10, v11 // (op 5874) 
v_pack_b32_f16 v5, v12, v13 // (op 5874) 

s_lshl_b32 s0, s4, 6 // (op 5874) 
s_add_u32 s0, s0, 96 // (op 5874) 
v_add_u32 v3, v9, s0 // (op 5874) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5874) Store value



v_accvgpr_read v10, a60 // (op 5892) 
v_accvgpr_read v11, a61 // (op 5892) 
v_accvgpr_read v12, a62 // (op 5892) 
v_accvgpr_read v13, a63 // (op 5892) 
v_mul_f32 v14, v7, v10 // (op 5892) 
v_mul_f32 v15, v7, v11 // (op 5892) 
v_mul_f32 v16, v7, v12 // (op 5892) 
v_mul_f32 v17, v7, v13 // (op 5892) 

v_cvt_f16_f32 v10, v14 // (op 5894) 
v_cvt_f16_f32 v11, v15 // (op 5894) 
v_cvt_f16_f32 v12, v16 // (op 5894) 
v_cvt_f16_f32 v13, v17 // (op 5894) 


v_pack_b32_f16 v4, v10, v11 // (op 5876) 
v_pack_b32_f16 v5, v12, v13 // (op 5876) 

s_lshl_b32 s0, s4, 5 // (op 5876) 
s_mul_i32 s1, 3, s0 // (op 5876) 
s_add_u32 s1, s1, 96 // (op 5876) 

v_add_u32 v3, v9, s1 // (op 5876) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5876) Store value



v_accvgpr_read v10, a64 // (op 5942) 
v_accvgpr_read v11, a65 // (op 5942) 
v_accvgpr_read v12, a66 // (op 5942) 
v_accvgpr_read v13, a67 // (op 5942) 
v_mul_f32 v14, v7, v10 // (op 5942) 
v_mul_f32 v15, v7, v11 // (op 5942) 
v_mul_f32 v16, v7, v12 // (op 5942) 
v_mul_f32 v17, v7, v13 // (op 5942) 

v_cvt_f16_f32 v10, v14 // (op 5944) 
v_cvt_f16_f32 v11, v15 // (op 5944) 
v_cvt_f16_f32 v12, v16 // (op 5944) 
v_cvt_f16_f32 v13, v17 // (op 5944) 


v_pack_b32_f16 v4, v10, v11 // (op 5946) 
v_pack_b32_f16 v5, v12, v13 // (op 5946) 

v_add_u32 v3, 128, v9 // (op 5946) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5946) Store value



v_accvgpr_read v10, a68 // (op 5956) 
v_accvgpr_read v11, a69 // (op 5956) 
v_accvgpr_read v12, a70 // (op 5956) 
v_accvgpr_read v13, a71 // (op 5956) 
v_mul_f32 v14, v7, v10 // (op 5956) 
v_mul_f32 v15, v7, v11 // (op 5956) 
v_mul_f32 v16, v7, v12 // (op 5956) 
v_mul_f32 v17, v7, v13 // (op 5956) 

v_cvt_f16_f32 v10, v14 // (op 5958) 
v_cvt_f16_f32 v11, v15 // (op 5958) 
v_cvt_f16_f32 v12, v16 // (op 5958) 
v_cvt_f16_f32 v13, v17 // (op 5958) 


v_pack_b32_f16 v4, v10, v11 // (op 5948) 
v_pack_b32_f16 v5, v12, v13 // (op 5948) 

s_lshl_b32 s0, s4, 5 // (op 5948) 
s_add_u32 s0, s0, 128 // (op 5948) 
v_add_u32 v3, v9, s0 // (op 5948) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5948) Store value



v_accvgpr_read v10, a72 // (op 5962) 
v_accvgpr_read v11, a73 // (op 5962) 
v_accvgpr_read v12, a74 // (op 5962) 
v_accvgpr_read v13, a75 // (op 5962) 
v_mul_f32 v14, v7, v10 // (op 5962) 
v_mul_f32 v15, v7, v11 // (op 5962) 
v_mul_f32 v16, v7, v12 // (op 5962) 
v_mul_f32 v17, v7, v13 // (op 5962) 

v_cvt_f16_f32 v10, v14 // (op 5964) 
v_cvt_f16_f32 v11, v15 // (op 5964) 
v_cvt_f16_f32 v12, v16 // (op 5964) 
v_cvt_f16_f32 v13, v17 // (op 5964) 


v_pack_b32_f16 v4, v10, v11 // (op 5950) 
v_pack_b32_f16 v5, v12, v13 // (op 5950) 

s_lshl_b32 s0, s4, 6 // (op 5950) 
s_add_u32 s0, s0, 128 // (op 5950) 
v_add_u32 v3, v9, s0 // (op 5950) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5950) Store value



v_accvgpr_read v10, a76 // (op 5968) 
v_accvgpr_read v11, a77 // (op 5968) 
v_accvgpr_read v12, a78 // (op 5968) 
v_accvgpr_read v13, a79 // (op 5968) 
v_mul_f32 v14, v7, v10 // (op 5968) 
v_mul_f32 v15, v7, v11 // (op 5968) 
v_mul_f32 v16, v7, v12 // (op 5968) 
v_mul_f32 v17, v7, v13 // (op 5968) 

v_cvt_f16_f32 v10, v14 // (op 5970) 
v_cvt_f16_f32 v11, v15 // (op 5970) 
v_cvt_f16_f32 v12, v16 // (op 5970) 
v_cvt_f16_f32 v13, v17 // (op 5970) 


v_pack_b32_f16 v4, v10, v11 // (op 5952) 
v_pack_b32_f16 v5, v12, v13 // (op 5952) 

s_lshl_b32 s0, s4, 5 // (op 5952) 
s_mul_i32 s1, 3, s0 // (op 5952) 
s_add_u32 s1, s1, 128 // (op 5952) 

v_add_u32 v3, v9, s1 // (op 5952) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5952) Store value



v_accvgpr_read v10, a80 // (op 6018) 
v_accvgpr_read v11, a81 // (op 6018) 
v_accvgpr_read v12, a82 // (op 6018) 
v_accvgpr_read v13, a83 // (op 6018) 
v_mul_f32 v14, v7, v10 // (op 6018) 
v_mul_f32 v15, v7, v11 // (op 6018) 
v_mul_f32 v16, v7, v12 // (op 6018) 
v_mul_f32 v17, v7, v13 // (op 6018) 

v_cvt_f16_f32 v10, v14 // (op 6020) 
v_cvt_f16_f32 v11, v15 // (op 6020) 
v_cvt_f16_f32 v12, v16 // (op 6020) 
v_cvt_f16_f32 v13, v17 // (op 6020) 


v_pack_b32_f16 v4, v10, v11 // (op 6022) 
v_pack_b32_f16 v5, v12, v13 // (op 6022) 

v_add_u32 v3, 160, v9 // (op 6022) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6022) Store value



v_accvgpr_read v10, a84 // (op 6032) 
v_accvgpr_read v11, a85 // (op 6032) 
v_accvgpr_read v12, a86 // (op 6032) 
v_accvgpr_read v13, a87 // (op 6032) 
v_mul_f32 v14, v7, v10 // (op 6032) 
v_mul_f32 v15, v7, v11 // (op 6032) 
v_mul_f32 v16, v7, v12 // (op 6032) 
v_mul_f32 v17, v7, v13 // (op 6032) 

v_cvt_f16_f32 v10, v14 // (op 6034) 
v_cvt_f16_f32 v11, v15 // (op 6034) 
v_cvt_f16_f32 v12, v16 // (op 6034) 
v_cvt_f16_f32 v13, v17 // (op 6034) 


v_pack_b32_f16 v4, v10, v11 // (op 6024) 
v_pack_b32_f16 v5, v12, v13 // (op 6024) 

s_lshl_b32 s0, s4, 5 // (op 6024) 
s_add_u32 s0, s0, 160 // (op 6024) 
v_add_u32 v3, v9, s0 // (op 6024) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6024) Store value



v_accvgpr_read v10, a88 // (op 6038) 
v_accvgpr_read v11, a89 // (op 6038) 
v_accvgpr_read v12, a90 // (op 6038) 
v_accvgpr_read v13, a91 // (op 6038) 
v_mul_f32 v14, v7, v10 // (op 6038) 
v_mul_f32 v15, v7, v11 // (op 6038) 
v_mul_f32 v16, v7, v12 // (op 6038) 
v_mul_f32 v17, v7, v13 // (op 6038) 

v_cvt_f16_f32 v10, v14 // (op 6040) 
v_cvt_f16_f32 v11, v15 // (op 6040) 
v_cvt_f16_f32 v12, v16 // (op 6040) 
v_cvt_f16_f32 v13, v17 // (op 6040) 


v_pack_b32_f16 v4, v10, v11 // (op 6026) 
v_pack_b32_f16 v5, v12, v13 // (op 6026) 

s_lshl_b32 s0, s4, 6 // (op 6026) 
s_add_u32 s0, s0, 160 // (op 6026) 
v_add_u32 v3, v9, s0 // (op 6026) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6026) Store value



v_accvgpr_read v10, a92 // (op 6044) 
v_accvgpr_read v11, a93 // (op 6044) 
v_accvgpr_read v12, a94 // (op 6044) 
v_accvgpr_read v13, a95 // (op 6044) 
v_mul_f32 v14, v7, v10 // (op 6044) 
v_mul_f32 v15, v7, v11 // (op 6044) 
v_mul_f32 v16, v7, v12 // (op 6044) 
v_mul_f32 v17, v7, v13 // (op 6044) 

v_cvt_f16_f32 v10, v14 // (op 6046) 
v_cvt_f16_f32 v11, v15 // (op 6046) 
v_cvt_f16_f32 v12, v16 // (op 6046) 
v_cvt_f16_f32 v13, v17 // (op 6046) 


v_pack_b32_f16 v4, v10, v11 // (op 6028) 
v_pack_b32_f16 v5, v12, v13 // (op 6028) 

s_lshl_b32 s0, s4, 5 // (op 6028) 
s_mul_i32 s1, 3, s0 // (op 6028) 
s_add_u32 s1, s1, 160 // (op 6028) 

v_add_u32 v3, v9, s1 // (op 6028) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6028) Store value



v_accvgpr_read v10, a96 // (op 6094) 
v_accvgpr_read v11, a97 // (op 6094) 
v_accvgpr_read v12, a98 // (op 6094) 
v_accvgpr_read v13, a99 // (op 6094) 
v_mul_f32 v14, v7, v10 // (op 6094) 
v_mul_f32 v15, v7, v11 // (op 6094) 
v_mul_f32 v16, v7, v12 // (op 6094) 
v_mul_f32 v17, v7, v13 // (op 6094) 

v_cvt_f16_f32 v10, v14 // (op 6096) 
v_cvt_f16_f32 v11, v15 // (op 6096) 
v_cvt_f16_f32 v12, v16 // (op 6096) 
v_cvt_f16_f32 v13, v17 // (op 6096) 


v_pack_b32_f16 v4, v10, v11 // (op 6098) 
v_pack_b32_f16 v5, v12, v13 // (op 6098) 

v_add_u32 v3, 192, v9 // (op 6098) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6098) Store value



v_accvgpr_read v10, a100 // (op 6108) 
v_accvgpr_read v11, a101 // (op 6108) 
v_accvgpr_read v12, a102 // (op 6108) 
v_accvgpr_read v13, a103 // (op 6108) 
v_mul_f32 v14, v7, v10 // (op 6108) 
v_mul_f32 v15, v7, v11 // (op 6108) 
v_mul_f32 v16, v7, v12 // (op 6108) 
v_mul_f32 v17, v7, v13 // (op 6108) 

v_cvt_f16_f32 v10, v14 // (op 6110) 
v_cvt_f16_f32 v11, v15 // (op 6110) 
v_cvt_f16_f32 v12, v16 // (op 6110) 
v_cvt_f16_f32 v13, v17 // (op 6110) 


v_pack_b32_f16 v4, v10, v11 // (op 6100) 
v_pack_b32_f16 v5, v12, v13 // (op 6100) 

s_lshl_b32 s0, s4, 5 // (op 6100) 
s_add_u32 s0, s0, 192 // (op 6100) 
v_add_u32 v3, v9, s0 // (op 6100) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6100) Store value



v_accvgpr_read v10, a104 // (op 6114) 
v_accvgpr_read v11, a105 // (op 6114) 
v_accvgpr_read v12, a106 // (op 6114) 
v_accvgpr_read v13, a107 // (op 6114) 
v_mul_f32 v14, v7, v10 // (op 6114) 
v_mul_f32 v15, v7, v11 // (op 6114) 
v_mul_f32 v16, v7, v12 // (op 6114) 
v_mul_f32 v17, v7, v13 // (op 6114) 

v_cvt_f16_f32 v10, v14 // (op 6116) 
v_cvt_f16_f32 v11, v15 // (op 6116) 
v_cvt_f16_f32 v12, v16 // (op 6116) 
v_cvt_f16_f32 v13, v17 // (op 6116) 


v_pack_b32_f16 v4, v10, v11 // (op 6102) 
v_pack_b32_f16 v5, v12, v13 // (op 6102) 

s_lshl_b32 s0, s4, 6 // (op 6102) 
s_add_u32 s0, s0, 192 // (op 6102) 
v_add_u32 v3, v9, s0 // (op 6102) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6102) Store value



v_accvgpr_read v10, a108 // (op 6120) 
v_accvgpr_read v11, a109 // (op 6120) 
v_accvgpr_read v12, a110 // (op 6120) 
v_accvgpr_read v13, a111 // (op 6120) 
v_mul_f32 v14, v7, v10 // (op 6120) 
v_mul_f32 v15, v7, v11 // (op 6120) 
v_mul_f32 v16, v7, v12 // (op 6120) 
v_mul_f32 v17, v7, v13 // (op 6120) 

v_cvt_f16_f32 v10, v14 // (op 6122) 
v_cvt_f16_f32 v11, v15 // (op 6122) 
v_cvt_f16_f32 v12, v16 // (op 6122) 
v_cvt_f16_f32 v13, v17 // (op 6122) 


v_pack_b32_f16 v4, v10, v11 // (op 6104) 
v_pack_b32_f16 v5, v12, v13 // (op 6104) 

s_lshl_b32 s0, s4, 5 // (op 6104) 
s_mul_i32 s1, 3, s0 // (op 6104) 
s_add_u32 s1, s1, 192 // (op 6104) 

v_add_u32 v3, v9, s1 // (op 6104) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6104) Store value



v_accvgpr_read v10, a112 // (op 6170) 
v_accvgpr_read v11, a113 // (op 6170) 
v_accvgpr_read v12, a114 // (op 6170) 
v_accvgpr_read v13, a115 // (op 6170) 
v_mul_f32 v14, v7, v10 // (op 6170) 
v_mul_f32 v15, v7, v11 // (op 6170) 
v_mul_f32 v16, v7, v12 // (op 6170) 
v_mul_f32 v17, v7, v13 // (op 6170) 

v_cvt_f16_f32 v10, v14 // (op 6172) 
v_cvt_f16_f32 v11, v15 // (op 6172) 
v_cvt_f16_f32 v12, v16 // (op 6172) 
v_cvt_f16_f32 v13, v17 // (op 6172) 


v_pack_b32_f16 v4, v10, v11 // (op 6174) 
v_pack_b32_f16 v5, v12, v13 // (op 6174) 

v_add_u32 v3, 224, v9 // (op 6174) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6174) Store value



v_accvgpr_read v10, a116 // (op 6184) 
v_accvgpr_read v11, a117 // (op 6184) 
v_accvgpr_read v12, a118 // (op 6184) 
v_accvgpr_read v13, a119 // (op 6184) 
v_mul_f32 v14, v7, v10 // (op 6184) 
v_mul_f32 v15, v7, v11 // (op 6184) 
v_mul_f32 v16, v7, v12 // (op 6184) 
v_mul_f32 v17, v7, v13 // (op 6184) 

v_cvt_f16_f32 v10, v14 // (op 6186) 
v_cvt_f16_f32 v11, v15 // (op 6186) 
v_cvt_f16_f32 v12, v16 // (op 6186) 
v_cvt_f16_f32 v13, v17 // (op 6186) 


v_pack_b32_f16 v4, v10, v11 // (op 6176) 
v_pack_b32_f16 v5, v12, v13 // (op 6176) 

s_lshl_b32 s0, s4, 5 // (op 6176) 
s_add_u32 s0, s0, 224 // (op 6176) 
v_add_u32 v3, v9, s0 // (op 6176) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6176) Store value



v_accvgpr_read v10, a120 // (op 6190) 
v_accvgpr_read v11, a121 // (op 6190) 
v_accvgpr_read v12, a122 // (op 6190) 
v_accvgpr_read v13, a123 // (op 6190) 
v_mul_f32 v14, v7, v10 // (op 6190) 
v_mul_f32 v15, v7, v11 // (op 6190) 
v_mul_f32 v16, v7, v12 // (op 6190) 
v_mul_f32 v17, v7, v13 // (op 6190) 

v_cvt_f16_f32 v10, v14 // (op 6192) 
v_cvt_f16_f32 v11, v15 // (op 6192) 
v_cvt_f16_f32 v12, v16 // (op 6192) 
v_cvt_f16_f32 v13, v17 // (op 6192) 


v_pack_b32_f16 v4, v10, v11 // (op 6178) 
v_pack_b32_f16 v5, v12, v13 // (op 6178) 

s_lshl_b32 s0, s4, 6 // (op 6178) 
s_add_u32 s0, s0, 224 // (op 6178) 
v_add_u32 v3, v9, s0 // (op 6178) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6178) Store value



v_accvgpr_read v10, a124 // (op 6196) 
v_accvgpr_read v11, a125 // (op 6196) 
v_accvgpr_read v12, a126 // (op 6196) 
v_accvgpr_read v13, a127 // (op 6196) 
v_mul_f32 v14, v7, v10 // (op 6196) 
v_mul_f32 v15, v7, v11 // (op 6196) 
v_mul_f32 v16, v7, v12 // (op 6196) 
v_mul_f32 v17, v7, v13 // (op 6196) 

v_cvt_f16_f32 v10, v14 // (op 6198) 
v_cvt_f16_f32 v11, v15 // (op 6198) 
v_cvt_f16_f32 v12, v16 // (op 6198) 
v_cvt_f16_f32 v13, v17 // (op 6198) 


v_pack_b32_f16 v4, v10, v11 // (op 6180) 
v_pack_b32_f16 v5, v12, v13 // (op 6180) 

s_lshl_b32 s0, s4, 5 // (op 6180) 
s_mul_i32 s1, 3, s0 // (op 6180) 
s_add_u32 s1, s1, 224 // (op 6180) 

v_add_u32 v3, v9, s1 // (op 6180) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6180) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ConditionalBottom_0_beta_116
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ConditionalFalse_0_beta_116:


v_add_u32 v3, 224, v8 // (op 6972) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6972) Load value

v_accvgpr_read v10, a112 // (op 6959) 
v_accvgpr_read v11, a113 // (op 6959) 
v_accvgpr_read v12, a114 // (op 6959) 
v_accvgpr_read v13, a115 // (op 6959) 
v_mul_f32 v14, v7, v10 // (op 6959) 
v_mul_f32 v15, v7, v11 // (op 6959) 
v_mul_f32 v16, v7, v12 // (op 6959) 
v_mul_f32 v17, v7, v13 // (op 6959) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6971) 
v_lshrrev_b32 v11, 16, v4 // (op 6971) 
v_cvt_f32_f16 v11, v11 // (op 6971) 
v_mul_f32 v18, v0, v10 // (op 6971) 
v_mul_f32 v19, v0, v11 // (op 6971) 
v_cvt_f32_f16 v10, v5 // (op 6971) 
v_lshrrev_b32 v11, 16, v5 // (op 6971) 
v_cvt_f32_f16 v11, v11 // (op 6971) 
v_mul_f32 v20, v0, v10 // (op 6971) 
v_mul_f32 v21, v0, v11 // (op 6971) 


v_add_f32 v10, v18, v14 // (op 6960) 
v_add_f32 v11, v19, v15 // (op 6960) 
v_add_f32 v12, v20, v16 // (op 6960) 
v_add_f32 v13, v21, v17 // (op 6960) 


v_cvt_f16_f32 v14, v10 // (op 6961) 
v_cvt_f16_f32 v15, v11 // (op 6961) 
v_cvt_f16_f32 v16, v12 // (op 6961) 
v_cvt_f16_f32 v17, v13 // (op 6961) 


v_pack_b32_f16 v10, v14, v15 // (op 6963) 
v_pack_b32_f16 v11, v16, v17 // (op 6963) 

v_add_u32 v3, 224, v9 // (op 6963) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6963) Store value





s_lshl_b32 s0, s2, 5 // (op 6977) 
s_add_u32 s0, s0, 224 // (op 6977) 
v_add_u32 v3, v8, s0 // (op 6977) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6977) Load value

v_accvgpr_read v10, a116 // (op 6989) 
v_accvgpr_read v11, a117 // (op 6989) 
v_accvgpr_read v12, a118 // (op 6989) 
v_accvgpr_read v13, a119 // (op 6989) 
v_mul_f32 v14, v7, v10 // (op 6989) 
v_mul_f32 v15, v7, v11 // (op 6989) 
v_mul_f32 v16, v7, v12 // (op 6989) 
v_mul_f32 v17, v7, v13 // (op 6989) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6974) 
v_lshrrev_b32 v11, 16, v4 // (op 6974) 
v_cvt_f32_f16 v11, v11 // (op 6974) 
v_mul_f32 v18, v0, v10 // (op 6974) 
v_mul_f32 v19, v0, v11 // (op 6974) 
v_cvt_f32_f16 v10, v5 // (op 6974) 
v_lshrrev_b32 v11, 16, v5 // (op 6974) 
v_cvt_f32_f16 v11, v11 // (op 6974) 
v_mul_f32 v20, v0, v10 // (op 6974) 
v_mul_f32 v21, v0, v11 // (op 6974) 


v_add_f32 v10, v18, v14 // (op 6975) 
v_add_f32 v11, v19, v15 // (op 6975) 
v_add_f32 v12, v20, v16 // (op 6975) 
v_add_f32 v13, v21, v17 // (op 6975) 


v_cvt_f16_f32 v14, v10 // (op 6976) 
v_cvt_f16_f32 v15, v11 // (op 6976) 
v_cvt_f16_f32 v16, v12 // (op 6976) 
v_cvt_f16_f32 v17, v13 // (op 6976) 


v_pack_b32_f16 v10, v14, v15 // (op 6965) 
v_pack_b32_f16 v11, v16, v17 // (op 6965) 

s_lshl_b32 s0, s4, 5 // (op 6965) 
s_add_u32 s0, s0, 224 // (op 6965) 
v_add_u32 v3, v9, s0 // (op 6965) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6965) Store value





s_lshl_b32 s0, s2, 6 // (op 6982) 
s_add_u32 s0, s0, 224 // (op 6982) 
v_add_u32 v3, v8, s0 // (op 6982) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6982) Load value

v_accvgpr_read v10, a120 // (op 6991) 
v_accvgpr_read v11, a121 // (op 6991) 
v_accvgpr_read v12, a122 // (op 6991) 
v_accvgpr_read v13, a123 // (op 6991) 
v_mul_f32 v14, v7, v10 // (op 6991) 
v_mul_f32 v15, v7, v11 // (op 6991) 
v_mul_f32 v16, v7, v12 // (op 6991) 
v_mul_f32 v17, v7, v13 // (op 6991) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6979) 
v_lshrrev_b32 v11, 16, v4 // (op 6979) 
v_cvt_f32_f16 v11, v11 // (op 6979) 
v_mul_f32 v18, v0, v10 // (op 6979) 
v_mul_f32 v19, v0, v11 // (op 6979) 
v_cvt_f32_f16 v10, v5 // (op 6979) 
v_lshrrev_b32 v11, 16, v5 // (op 6979) 
v_cvt_f32_f16 v11, v11 // (op 6979) 
v_mul_f32 v20, v0, v10 // (op 6979) 
v_mul_f32 v21, v0, v11 // (op 6979) 


v_add_f32 v10, v18, v14 // (op 6980) 
v_add_f32 v11, v19, v15 // (op 6980) 
v_add_f32 v12, v20, v16 // (op 6980) 
v_add_f32 v13, v21, v17 // (op 6980) 


v_cvt_f16_f32 v14, v10 // (op 6981) 
v_cvt_f16_f32 v15, v11 // (op 6981) 
v_cvt_f16_f32 v16, v12 // (op 6981) 
v_cvt_f16_f32 v17, v13 // (op 6981) 


v_pack_b32_f16 v10, v14, v15 // (op 6967) 
v_pack_b32_f16 v11, v16, v17 // (op 6967) 

s_lshl_b32 s0, s4, 6 // (op 6967) 
s_add_u32 s0, s0, 224 // (op 6967) 
v_add_u32 v3, v9, s0 // (op 6967) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6967) Store value





s_lshl_b32 s0, s2, 5 // (op 6987) 
s_mul_i32 s1, 3, s0 // (op 6987) 
s_add_u32 s1, s1, 224 // (op 6987) 

v_add_u32 v3, v8, s1 // (op 6987) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6987) Load value

v_accvgpr_read v10, a124 // (op 6993) 
v_accvgpr_read v11, a125 // (op 6993) 
v_accvgpr_read v12, a126 // (op 6993) 
v_accvgpr_read v13, a127 // (op 6993) 
v_mul_f32 v14, v7, v10 // (op 6993) 
v_mul_f32 v15, v7, v11 // (op 6993) 
v_mul_f32 v16, v7, v12 // (op 6993) 
v_mul_f32 v17, v7, v13 // (op 6993) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6984) 
v_lshrrev_b32 v11, 16, v4 // (op 6984) 
v_cvt_f32_f16 v11, v11 // (op 6984) 
v_mul_f32 v18, v0, v10 // (op 6984) 
v_mul_f32 v19, v0, v11 // (op 6984) 
v_cvt_f32_f16 v10, v5 // (op 6984) 
v_lshrrev_b32 v11, 16, v5 // (op 6984) 
v_cvt_f32_f16 v11, v11 // (op 6984) 
v_mul_f32 v20, v0, v10 // (op 6984) 
v_mul_f32 v21, v0, v11 // (op 6984) 


v_add_f32 v10, v18, v14 // (op 6985) 
v_add_f32 v11, v19, v15 // (op 6985) 
v_add_f32 v12, v20, v16 // (op 6985) 
v_add_f32 v13, v21, v17 // (op 6985) 


v_cvt_f16_f32 v14, v10 // (op 6986) 
v_cvt_f16_f32 v15, v11 // (op 6986) 
v_cvt_f16_f32 v16, v12 // (op 6986) 
v_cvt_f16_f32 v17, v13 // (op 6986) 


v_pack_b32_f16 v10, v14, v15 // (op 6969) 
v_pack_b32_f16 v11, v16, v17 // (op 6969) 

s_lshl_b32 s0, s4, 5 // (op 6969) 
s_mul_i32 s1, 3, s0 // (op 6969) 
s_add_u32 s1, s1, 224 // (op 6969) 

v_add_u32 v3, v9, s1 // (op 6969) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6969) Store value





v_add_u32 v3, 160, v8 // (op 6798) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6798) Load value

v_accvgpr_read v10, a80 // (op 6785) 
v_accvgpr_read v11, a81 // (op 6785) 
v_accvgpr_read v12, a82 // (op 6785) 
v_accvgpr_read v13, a83 // (op 6785) 
v_mul_f32 v14, v7, v10 // (op 6785) 
v_mul_f32 v15, v7, v11 // (op 6785) 
v_mul_f32 v16, v7, v12 // (op 6785) 
v_mul_f32 v17, v7, v13 // (op 6785) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6797) 
v_lshrrev_b32 v11, 16, v4 // (op 6797) 
v_cvt_f32_f16 v11, v11 // (op 6797) 
v_mul_f32 v18, v0, v10 // (op 6797) 
v_mul_f32 v19, v0, v11 // (op 6797) 
v_cvt_f32_f16 v10, v5 // (op 6797) 
v_lshrrev_b32 v11, 16, v5 // (op 6797) 
v_cvt_f32_f16 v11, v11 // (op 6797) 
v_mul_f32 v20, v0, v10 // (op 6797) 
v_mul_f32 v21, v0, v11 // (op 6797) 


v_add_f32 v10, v18, v14 // (op 6786) 
v_add_f32 v11, v19, v15 // (op 6786) 
v_add_f32 v12, v20, v16 // (op 6786) 
v_add_f32 v13, v21, v17 // (op 6786) 


v_cvt_f16_f32 v14, v10 // (op 6787) 
v_cvt_f16_f32 v15, v11 // (op 6787) 
v_cvt_f16_f32 v16, v12 // (op 6787) 
v_cvt_f16_f32 v17, v13 // (op 6787) 


v_pack_b32_f16 v10, v14, v15 // (op 6789) 
v_pack_b32_f16 v11, v16, v17 // (op 6789) 

v_add_u32 v3, 160, v9 // (op 6789) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6789) Store value





s_lshl_b32 s0, s2, 5 // (op 6803) 
s_add_u32 s0, s0, 160 // (op 6803) 
v_add_u32 v3, v8, s0 // (op 6803) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6803) Load value

v_accvgpr_read v10, a84 // (op 6815) 
v_accvgpr_read v11, a85 // (op 6815) 
v_accvgpr_read v12, a86 // (op 6815) 
v_accvgpr_read v13, a87 // (op 6815) 
v_mul_f32 v14, v7, v10 // (op 6815) 
v_mul_f32 v15, v7, v11 // (op 6815) 
v_mul_f32 v16, v7, v12 // (op 6815) 
v_mul_f32 v17, v7, v13 // (op 6815) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6800) 
v_lshrrev_b32 v11, 16, v4 // (op 6800) 
v_cvt_f32_f16 v11, v11 // (op 6800) 
v_mul_f32 v18, v0, v10 // (op 6800) 
v_mul_f32 v19, v0, v11 // (op 6800) 
v_cvt_f32_f16 v10, v5 // (op 6800) 
v_lshrrev_b32 v11, 16, v5 // (op 6800) 
v_cvt_f32_f16 v11, v11 // (op 6800) 
v_mul_f32 v20, v0, v10 // (op 6800) 
v_mul_f32 v21, v0, v11 // (op 6800) 


v_add_f32 v10, v18, v14 // (op 6801) 
v_add_f32 v11, v19, v15 // (op 6801) 
v_add_f32 v12, v20, v16 // (op 6801) 
v_add_f32 v13, v21, v17 // (op 6801) 


v_cvt_f16_f32 v14, v10 // (op 6802) 
v_cvt_f16_f32 v15, v11 // (op 6802) 
v_cvt_f16_f32 v16, v12 // (op 6802) 
v_cvt_f16_f32 v17, v13 // (op 6802) 


v_pack_b32_f16 v10, v14, v15 // (op 6791) 
v_pack_b32_f16 v11, v16, v17 // (op 6791) 

s_lshl_b32 s0, s4, 5 // (op 6791) 
s_add_u32 s0, s0, 160 // (op 6791) 
v_add_u32 v3, v9, s0 // (op 6791) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6791) Store value





s_lshl_b32 s0, s2, 6 // (op 6808) 
s_add_u32 s0, s0, 160 // (op 6808) 
v_add_u32 v3, v8, s0 // (op 6808) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6808) Load value

v_accvgpr_read v10, a88 // (op 6817) 
v_accvgpr_read v11, a89 // (op 6817) 
v_accvgpr_read v12, a90 // (op 6817) 
v_accvgpr_read v13, a91 // (op 6817) 
v_mul_f32 v14, v7, v10 // (op 6817) 
v_mul_f32 v15, v7, v11 // (op 6817) 
v_mul_f32 v16, v7, v12 // (op 6817) 
v_mul_f32 v17, v7, v13 // (op 6817) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6805) 
v_lshrrev_b32 v11, 16, v4 // (op 6805) 
v_cvt_f32_f16 v11, v11 // (op 6805) 
v_mul_f32 v18, v0, v10 // (op 6805) 
v_mul_f32 v19, v0, v11 // (op 6805) 
v_cvt_f32_f16 v10, v5 // (op 6805) 
v_lshrrev_b32 v11, 16, v5 // (op 6805) 
v_cvt_f32_f16 v11, v11 // (op 6805) 
v_mul_f32 v20, v0, v10 // (op 6805) 
v_mul_f32 v21, v0, v11 // (op 6805) 


v_add_f32 v10, v18, v14 // (op 6806) 
v_add_f32 v11, v19, v15 // (op 6806) 
v_add_f32 v12, v20, v16 // (op 6806) 
v_add_f32 v13, v21, v17 // (op 6806) 


v_cvt_f16_f32 v14, v10 // (op 6807) 
v_cvt_f16_f32 v15, v11 // (op 6807) 
v_cvt_f16_f32 v16, v12 // (op 6807) 
v_cvt_f16_f32 v17, v13 // (op 6807) 


v_pack_b32_f16 v10, v14, v15 // (op 6793) 
v_pack_b32_f16 v11, v16, v17 // (op 6793) 

s_lshl_b32 s0, s4, 6 // (op 6793) 
s_add_u32 s0, s0, 160 // (op 6793) 
v_add_u32 v3, v9, s0 // (op 6793) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6793) Store value





s_lshl_b32 s0, s2, 5 // (op 6813) 
s_mul_i32 s1, 3, s0 // (op 6813) 
s_add_u32 s1, s1, 160 // (op 6813) 

v_add_u32 v3, v8, s1 // (op 6813) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6813) Load value

v_accvgpr_read v10, a92 // (op 6819) 
v_accvgpr_read v11, a93 // (op 6819) 
v_accvgpr_read v12, a94 // (op 6819) 
v_accvgpr_read v13, a95 // (op 6819) 
v_mul_f32 v14, v7, v10 // (op 6819) 
v_mul_f32 v15, v7, v11 // (op 6819) 
v_mul_f32 v16, v7, v12 // (op 6819) 
v_mul_f32 v17, v7, v13 // (op 6819) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6810) 
v_lshrrev_b32 v11, 16, v4 // (op 6810) 
v_cvt_f32_f16 v11, v11 // (op 6810) 
v_mul_f32 v18, v0, v10 // (op 6810) 
v_mul_f32 v19, v0, v11 // (op 6810) 
v_cvt_f32_f16 v10, v5 // (op 6810) 
v_lshrrev_b32 v11, 16, v5 // (op 6810) 
v_cvt_f32_f16 v11, v11 // (op 6810) 
v_mul_f32 v20, v0, v10 // (op 6810) 
v_mul_f32 v21, v0, v11 // (op 6810) 


v_add_f32 v10, v18, v14 // (op 6811) 
v_add_f32 v11, v19, v15 // (op 6811) 
v_add_f32 v12, v20, v16 // (op 6811) 
v_add_f32 v13, v21, v17 // (op 6811) 


v_cvt_f16_f32 v14, v10 // (op 6812) 
v_cvt_f16_f32 v15, v11 // (op 6812) 
v_cvt_f16_f32 v16, v12 // (op 6812) 
v_cvt_f16_f32 v17, v13 // (op 6812) 


v_pack_b32_f16 v10, v14, v15 // (op 6795) 
v_pack_b32_f16 v11, v16, v17 // (op 6795) 

s_lshl_b32 s0, s4, 5 // (op 6795) 
s_mul_i32 s1, 3, s0 // (op 6795) 
s_add_u32 s1, s1, 160 // (op 6795) 

v_add_u32 v3, v9, s1 // (op 6795) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6795) Store value





v_add_u32 v3, 128, v8 // (op 6711) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6711) Load value

v_accvgpr_read v10, a64 // (op 6698) 
v_accvgpr_read v11, a65 // (op 6698) 
v_accvgpr_read v12, a66 // (op 6698) 
v_accvgpr_read v13, a67 // (op 6698) 
v_mul_f32 v14, v7, v10 // (op 6698) 
v_mul_f32 v15, v7, v11 // (op 6698) 
v_mul_f32 v16, v7, v12 // (op 6698) 
v_mul_f32 v17, v7, v13 // (op 6698) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6710) 
v_lshrrev_b32 v11, 16, v4 // (op 6710) 
v_cvt_f32_f16 v11, v11 // (op 6710) 
v_mul_f32 v18, v0, v10 // (op 6710) 
v_mul_f32 v19, v0, v11 // (op 6710) 
v_cvt_f32_f16 v10, v5 // (op 6710) 
v_lshrrev_b32 v11, 16, v5 // (op 6710) 
v_cvt_f32_f16 v11, v11 // (op 6710) 
v_mul_f32 v20, v0, v10 // (op 6710) 
v_mul_f32 v21, v0, v11 // (op 6710) 


v_add_f32 v10, v18, v14 // (op 6699) 
v_add_f32 v11, v19, v15 // (op 6699) 
v_add_f32 v12, v20, v16 // (op 6699) 
v_add_f32 v13, v21, v17 // (op 6699) 


v_cvt_f16_f32 v14, v10 // (op 6700) 
v_cvt_f16_f32 v15, v11 // (op 6700) 
v_cvt_f16_f32 v16, v12 // (op 6700) 
v_cvt_f16_f32 v17, v13 // (op 6700) 


v_pack_b32_f16 v10, v14, v15 // (op 6702) 
v_pack_b32_f16 v11, v16, v17 // (op 6702) 

v_add_u32 v3, 128, v9 // (op 6702) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6702) Store value





s_lshl_b32 s0, s2, 5 // (op 6716) 
s_add_u32 s0, s0, 128 // (op 6716) 
v_add_u32 v3, v8, s0 // (op 6716) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6716) Load value

v_accvgpr_read v10, a68 // (op 6728) 
v_accvgpr_read v11, a69 // (op 6728) 
v_accvgpr_read v12, a70 // (op 6728) 
v_accvgpr_read v13, a71 // (op 6728) 
v_mul_f32 v14, v7, v10 // (op 6728) 
v_mul_f32 v15, v7, v11 // (op 6728) 
v_mul_f32 v16, v7, v12 // (op 6728) 
v_mul_f32 v17, v7, v13 // (op 6728) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6713) 
v_lshrrev_b32 v11, 16, v4 // (op 6713) 
v_cvt_f32_f16 v11, v11 // (op 6713) 
v_mul_f32 v18, v0, v10 // (op 6713) 
v_mul_f32 v19, v0, v11 // (op 6713) 
v_cvt_f32_f16 v10, v5 // (op 6713) 
v_lshrrev_b32 v11, 16, v5 // (op 6713) 
v_cvt_f32_f16 v11, v11 // (op 6713) 
v_mul_f32 v20, v0, v10 // (op 6713) 
v_mul_f32 v21, v0, v11 // (op 6713) 


v_add_f32 v10, v18, v14 // (op 6714) 
v_add_f32 v11, v19, v15 // (op 6714) 
v_add_f32 v12, v20, v16 // (op 6714) 
v_add_f32 v13, v21, v17 // (op 6714) 


v_cvt_f16_f32 v14, v10 // (op 6715) 
v_cvt_f16_f32 v15, v11 // (op 6715) 
v_cvt_f16_f32 v16, v12 // (op 6715) 
v_cvt_f16_f32 v17, v13 // (op 6715) 


v_pack_b32_f16 v10, v14, v15 // (op 6704) 
v_pack_b32_f16 v11, v16, v17 // (op 6704) 

s_lshl_b32 s0, s4, 5 // (op 6704) 
s_add_u32 s0, s0, 128 // (op 6704) 
v_add_u32 v3, v9, s0 // (op 6704) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6704) Store value





s_lshl_b32 s0, s2, 6 // (op 6721) 
s_add_u32 s0, s0, 128 // (op 6721) 
v_add_u32 v3, v8, s0 // (op 6721) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6721) Load value

v_accvgpr_read v10, a72 // (op 6730) 
v_accvgpr_read v11, a73 // (op 6730) 
v_accvgpr_read v12, a74 // (op 6730) 
v_accvgpr_read v13, a75 // (op 6730) 
v_mul_f32 v14, v7, v10 // (op 6730) 
v_mul_f32 v15, v7, v11 // (op 6730) 
v_mul_f32 v16, v7, v12 // (op 6730) 
v_mul_f32 v17, v7, v13 // (op 6730) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6718) 
v_lshrrev_b32 v11, 16, v4 // (op 6718) 
v_cvt_f32_f16 v11, v11 // (op 6718) 
v_mul_f32 v18, v0, v10 // (op 6718) 
v_mul_f32 v19, v0, v11 // (op 6718) 
v_cvt_f32_f16 v10, v5 // (op 6718) 
v_lshrrev_b32 v11, 16, v5 // (op 6718) 
v_cvt_f32_f16 v11, v11 // (op 6718) 
v_mul_f32 v20, v0, v10 // (op 6718) 
v_mul_f32 v21, v0, v11 // (op 6718) 


v_add_f32 v10, v18, v14 // (op 6719) 
v_add_f32 v11, v19, v15 // (op 6719) 
v_add_f32 v12, v20, v16 // (op 6719) 
v_add_f32 v13, v21, v17 // (op 6719) 


v_cvt_f16_f32 v14, v10 // (op 6720) 
v_cvt_f16_f32 v15, v11 // (op 6720) 
v_cvt_f16_f32 v16, v12 // (op 6720) 
v_cvt_f16_f32 v17, v13 // (op 6720) 


v_pack_b32_f16 v10, v14, v15 // (op 6706) 
v_pack_b32_f16 v11, v16, v17 // (op 6706) 

s_lshl_b32 s0, s4, 6 // (op 6706) 
s_add_u32 s0, s0, 128 // (op 6706) 
v_add_u32 v3, v9, s0 // (op 6706) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6706) Store value





s_lshl_b32 s0, s2, 5 // (op 6726) 
s_mul_i32 s1, 3, s0 // (op 6726) 
s_add_u32 s1, s1, 128 // (op 6726) 

v_add_u32 v3, v8, s1 // (op 6726) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6726) Load value

v_accvgpr_read v10, a76 // (op 6732) 
v_accvgpr_read v11, a77 // (op 6732) 
v_accvgpr_read v12, a78 // (op 6732) 
v_accvgpr_read v13, a79 // (op 6732) 
v_mul_f32 v14, v7, v10 // (op 6732) 
v_mul_f32 v15, v7, v11 // (op 6732) 
v_mul_f32 v16, v7, v12 // (op 6732) 
v_mul_f32 v17, v7, v13 // (op 6732) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6723) 
v_lshrrev_b32 v11, 16, v4 // (op 6723) 
v_cvt_f32_f16 v11, v11 // (op 6723) 
v_mul_f32 v18, v0, v10 // (op 6723) 
v_mul_f32 v19, v0, v11 // (op 6723) 
v_cvt_f32_f16 v10, v5 // (op 6723) 
v_lshrrev_b32 v11, 16, v5 // (op 6723) 
v_cvt_f32_f16 v11, v11 // (op 6723) 
v_mul_f32 v20, v0, v10 // (op 6723) 
v_mul_f32 v21, v0, v11 // (op 6723) 


v_add_f32 v10, v18, v14 // (op 6724) 
v_add_f32 v11, v19, v15 // (op 6724) 
v_add_f32 v12, v20, v16 // (op 6724) 
v_add_f32 v13, v21, v17 // (op 6724) 


v_cvt_f16_f32 v14, v10 // (op 6725) 
v_cvt_f16_f32 v15, v11 // (op 6725) 
v_cvt_f16_f32 v16, v12 // (op 6725) 
v_cvt_f16_f32 v17, v13 // (op 6725) 


v_pack_b32_f16 v10, v14, v15 // (op 6708) 
v_pack_b32_f16 v11, v16, v17 // (op 6708) 

s_lshl_b32 s0, s4, 5 // (op 6708) 
s_mul_i32 s1, 3, s0 // (op 6708) 
s_add_u32 s1, s1, 128 // (op 6708) 

v_add_u32 v3, v9, s1 // (op 6708) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6708) Store value





v_add_u32 v3, 96, v8 // (op 6624) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6624) Load value

v_accvgpr_read v10, a48 // (op 6611) 
v_accvgpr_read v11, a49 // (op 6611) 
v_accvgpr_read v12, a50 // (op 6611) 
v_accvgpr_read v13, a51 // (op 6611) 
v_mul_f32 v14, v7, v10 // (op 6611) 
v_mul_f32 v15, v7, v11 // (op 6611) 
v_mul_f32 v16, v7, v12 // (op 6611) 
v_mul_f32 v17, v7, v13 // (op 6611) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6623) 
v_lshrrev_b32 v11, 16, v4 // (op 6623) 
v_cvt_f32_f16 v11, v11 // (op 6623) 
v_mul_f32 v18, v0, v10 // (op 6623) 
v_mul_f32 v19, v0, v11 // (op 6623) 
v_cvt_f32_f16 v10, v5 // (op 6623) 
v_lshrrev_b32 v11, 16, v5 // (op 6623) 
v_cvt_f32_f16 v11, v11 // (op 6623) 
v_mul_f32 v20, v0, v10 // (op 6623) 
v_mul_f32 v21, v0, v11 // (op 6623) 


v_add_f32 v10, v18, v14 // (op 6612) 
v_add_f32 v11, v19, v15 // (op 6612) 
v_add_f32 v12, v20, v16 // (op 6612) 
v_add_f32 v13, v21, v17 // (op 6612) 


v_cvt_f16_f32 v14, v10 // (op 6613) 
v_cvt_f16_f32 v15, v11 // (op 6613) 
v_cvt_f16_f32 v16, v12 // (op 6613) 
v_cvt_f16_f32 v17, v13 // (op 6613) 


v_pack_b32_f16 v10, v14, v15 // (op 6615) 
v_pack_b32_f16 v11, v16, v17 // (op 6615) 

v_add_u32 v3, 96, v9 // (op 6615) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6615) Store value





s_lshl_b32 s0, s2, 5 // (op 6629) 
s_add_u32 s0, s0, 96 // (op 6629) 
v_add_u32 v3, v8, s0 // (op 6629) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6629) Load value

v_accvgpr_read v10, a52 // (op 6641) 
v_accvgpr_read v11, a53 // (op 6641) 
v_accvgpr_read v12, a54 // (op 6641) 
v_accvgpr_read v13, a55 // (op 6641) 
v_mul_f32 v14, v7, v10 // (op 6641) 
v_mul_f32 v15, v7, v11 // (op 6641) 
v_mul_f32 v16, v7, v12 // (op 6641) 
v_mul_f32 v17, v7, v13 // (op 6641) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6626) 
v_lshrrev_b32 v11, 16, v4 // (op 6626) 
v_cvt_f32_f16 v11, v11 // (op 6626) 
v_mul_f32 v18, v0, v10 // (op 6626) 
v_mul_f32 v19, v0, v11 // (op 6626) 
v_cvt_f32_f16 v10, v5 // (op 6626) 
v_lshrrev_b32 v11, 16, v5 // (op 6626) 
v_cvt_f32_f16 v11, v11 // (op 6626) 
v_mul_f32 v20, v0, v10 // (op 6626) 
v_mul_f32 v21, v0, v11 // (op 6626) 


v_add_f32 v10, v18, v14 // (op 6627) 
v_add_f32 v11, v19, v15 // (op 6627) 
v_add_f32 v12, v20, v16 // (op 6627) 
v_add_f32 v13, v21, v17 // (op 6627) 


v_cvt_f16_f32 v14, v10 // (op 6628) 
v_cvt_f16_f32 v15, v11 // (op 6628) 
v_cvt_f16_f32 v16, v12 // (op 6628) 
v_cvt_f16_f32 v17, v13 // (op 6628) 


v_pack_b32_f16 v10, v14, v15 // (op 6617) 
v_pack_b32_f16 v11, v16, v17 // (op 6617) 

s_lshl_b32 s0, s4, 5 // (op 6617) 
s_add_u32 s0, s0, 96 // (op 6617) 
v_add_u32 v3, v9, s0 // (op 6617) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6617) Store value





s_lshl_b32 s0, s2, 6 // (op 6634) 
s_add_u32 s0, s0, 96 // (op 6634) 
v_add_u32 v3, v8, s0 // (op 6634) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6634) Load value

v_accvgpr_read v10, a56 // (op 6643) 
v_accvgpr_read v11, a57 // (op 6643) 
v_accvgpr_read v12, a58 // (op 6643) 
v_accvgpr_read v13, a59 // (op 6643) 
v_mul_f32 v14, v7, v10 // (op 6643) 
v_mul_f32 v15, v7, v11 // (op 6643) 
v_mul_f32 v16, v7, v12 // (op 6643) 
v_mul_f32 v17, v7, v13 // (op 6643) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6631) 
v_lshrrev_b32 v11, 16, v4 // (op 6631) 
v_cvt_f32_f16 v11, v11 // (op 6631) 
v_mul_f32 v18, v0, v10 // (op 6631) 
v_mul_f32 v19, v0, v11 // (op 6631) 
v_cvt_f32_f16 v10, v5 // (op 6631) 
v_lshrrev_b32 v11, 16, v5 // (op 6631) 
v_cvt_f32_f16 v11, v11 // (op 6631) 
v_mul_f32 v20, v0, v10 // (op 6631) 
v_mul_f32 v21, v0, v11 // (op 6631) 


v_add_f32 v10, v18, v14 // (op 6632) 
v_add_f32 v11, v19, v15 // (op 6632) 
v_add_f32 v12, v20, v16 // (op 6632) 
v_add_f32 v13, v21, v17 // (op 6632) 


v_cvt_f16_f32 v14, v10 // (op 6633) 
v_cvt_f16_f32 v15, v11 // (op 6633) 
v_cvt_f16_f32 v16, v12 // (op 6633) 
v_cvt_f16_f32 v17, v13 // (op 6633) 


v_pack_b32_f16 v10, v14, v15 // (op 6619) 
v_pack_b32_f16 v11, v16, v17 // (op 6619) 

s_lshl_b32 s0, s4, 6 // (op 6619) 
s_add_u32 s0, s0, 96 // (op 6619) 
v_add_u32 v3, v9, s0 // (op 6619) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6619) Store value





s_lshl_b32 s0, s2, 5 // (op 6639) 
s_mul_i32 s1, 3, s0 // (op 6639) 
s_add_u32 s1, s1, 96 // (op 6639) 

v_add_u32 v3, v8, s1 // (op 6639) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6639) Load value

v_accvgpr_read v10, a60 // (op 6645) 
v_accvgpr_read v11, a61 // (op 6645) 
v_accvgpr_read v12, a62 // (op 6645) 
v_accvgpr_read v13, a63 // (op 6645) 
v_mul_f32 v14, v7, v10 // (op 6645) 
v_mul_f32 v15, v7, v11 // (op 6645) 
v_mul_f32 v16, v7, v12 // (op 6645) 
v_mul_f32 v17, v7, v13 // (op 6645) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6636) 
v_lshrrev_b32 v11, 16, v4 // (op 6636) 
v_cvt_f32_f16 v11, v11 // (op 6636) 
v_mul_f32 v18, v0, v10 // (op 6636) 
v_mul_f32 v19, v0, v11 // (op 6636) 
v_cvt_f32_f16 v10, v5 // (op 6636) 
v_lshrrev_b32 v11, 16, v5 // (op 6636) 
v_cvt_f32_f16 v11, v11 // (op 6636) 
v_mul_f32 v20, v0, v10 // (op 6636) 
v_mul_f32 v21, v0, v11 // (op 6636) 


v_add_f32 v10, v18, v14 // (op 6637) 
v_add_f32 v11, v19, v15 // (op 6637) 
v_add_f32 v12, v20, v16 // (op 6637) 
v_add_f32 v13, v21, v17 // (op 6637) 


v_cvt_f16_f32 v14, v10 // (op 6638) 
v_cvt_f16_f32 v15, v11 // (op 6638) 
v_cvt_f16_f32 v16, v12 // (op 6638) 
v_cvt_f16_f32 v17, v13 // (op 6638) 


v_pack_b32_f16 v10, v14, v15 // (op 6621) 
v_pack_b32_f16 v11, v16, v17 // (op 6621) 

s_lshl_b32 s0, s4, 5 // (op 6621) 
s_mul_i32 s1, 3, s0 // (op 6621) 
s_add_u32 s1, s1, 96 // (op 6621) 

v_add_u32 v3, v9, s1 // (op 6621) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6621) Store value





v_add_u32 v3, 64, v8 // (op 6537) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6537) Load value

v_accvgpr_read v10, a32 // (op 6524) 
v_accvgpr_read v11, a33 // (op 6524) 
v_accvgpr_read v12, a34 // (op 6524) 
v_accvgpr_read v13, a35 // (op 6524) 
v_mul_f32 v14, v7, v10 // (op 6524) 
v_mul_f32 v15, v7, v11 // (op 6524) 
v_mul_f32 v16, v7, v12 // (op 6524) 
v_mul_f32 v17, v7, v13 // (op 6524) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6536) 
v_lshrrev_b32 v11, 16, v4 // (op 6536) 
v_cvt_f32_f16 v11, v11 // (op 6536) 
v_mul_f32 v18, v0, v10 // (op 6536) 
v_mul_f32 v19, v0, v11 // (op 6536) 
v_cvt_f32_f16 v10, v5 // (op 6536) 
v_lshrrev_b32 v11, 16, v5 // (op 6536) 
v_cvt_f32_f16 v11, v11 // (op 6536) 
v_mul_f32 v20, v0, v10 // (op 6536) 
v_mul_f32 v21, v0, v11 // (op 6536) 


v_add_f32 v10, v18, v14 // (op 6525) 
v_add_f32 v11, v19, v15 // (op 6525) 
v_add_f32 v12, v20, v16 // (op 6525) 
v_add_f32 v13, v21, v17 // (op 6525) 


v_cvt_f16_f32 v14, v10 // (op 6526) 
v_cvt_f16_f32 v15, v11 // (op 6526) 
v_cvt_f16_f32 v16, v12 // (op 6526) 
v_cvt_f16_f32 v17, v13 // (op 6526) 


v_pack_b32_f16 v10, v14, v15 // (op 6528) 
v_pack_b32_f16 v11, v16, v17 // (op 6528) 

v_add_u32 v3, 64, v9 // (op 6528) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6528) Store value





s_lshl_b32 s0, s2, 5 // (op 6542) 
s_add_u32 s0, s0, 64 // (op 6542) 
v_add_u32 v3, v8, s0 // (op 6542) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6542) Load value

v_accvgpr_read v10, a36 // (op 6554) 
v_accvgpr_read v11, a37 // (op 6554) 
v_accvgpr_read v12, a38 // (op 6554) 
v_accvgpr_read v13, a39 // (op 6554) 
v_mul_f32 v14, v7, v10 // (op 6554) 
v_mul_f32 v15, v7, v11 // (op 6554) 
v_mul_f32 v16, v7, v12 // (op 6554) 
v_mul_f32 v17, v7, v13 // (op 6554) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6539) 
v_lshrrev_b32 v11, 16, v4 // (op 6539) 
v_cvt_f32_f16 v11, v11 // (op 6539) 
v_mul_f32 v18, v0, v10 // (op 6539) 
v_mul_f32 v19, v0, v11 // (op 6539) 
v_cvt_f32_f16 v10, v5 // (op 6539) 
v_lshrrev_b32 v11, 16, v5 // (op 6539) 
v_cvt_f32_f16 v11, v11 // (op 6539) 
v_mul_f32 v20, v0, v10 // (op 6539) 
v_mul_f32 v21, v0, v11 // (op 6539) 


v_add_f32 v10, v18, v14 // (op 6540) 
v_add_f32 v11, v19, v15 // (op 6540) 
v_add_f32 v12, v20, v16 // (op 6540) 
v_add_f32 v13, v21, v17 // (op 6540) 


v_cvt_f16_f32 v14, v10 // (op 6541) 
v_cvt_f16_f32 v15, v11 // (op 6541) 
v_cvt_f16_f32 v16, v12 // (op 6541) 
v_cvt_f16_f32 v17, v13 // (op 6541) 


v_pack_b32_f16 v10, v14, v15 // (op 6530) 
v_pack_b32_f16 v11, v16, v17 // (op 6530) 

s_lshl_b32 s0, s4, 5 // (op 6530) 
s_add_u32 s0, s0, 64 // (op 6530) 
v_add_u32 v3, v9, s0 // (op 6530) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6530) Store value





s_lshl_b32 s0, s2, 6 // (op 6547) 
s_add_u32 s0, s0, 64 // (op 6547) 
v_add_u32 v3, v8, s0 // (op 6547) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6547) Load value

v_accvgpr_read v10, a40 // (op 6556) 
v_accvgpr_read v11, a41 // (op 6556) 
v_accvgpr_read v12, a42 // (op 6556) 
v_accvgpr_read v13, a43 // (op 6556) 
v_mul_f32 v14, v7, v10 // (op 6556) 
v_mul_f32 v15, v7, v11 // (op 6556) 
v_mul_f32 v16, v7, v12 // (op 6556) 
v_mul_f32 v17, v7, v13 // (op 6556) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6544) 
v_lshrrev_b32 v11, 16, v4 // (op 6544) 
v_cvt_f32_f16 v11, v11 // (op 6544) 
v_mul_f32 v18, v0, v10 // (op 6544) 
v_mul_f32 v19, v0, v11 // (op 6544) 
v_cvt_f32_f16 v10, v5 // (op 6544) 
v_lshrrev_b32 v11, 16, v5 // (op 6544) 
v_cvt_f32_f16 v11, v11 // (op 6544) 
v_mul_f32 v20, v0, v10 // (op 6544) 
v_mul_f32 v21, v0, v11 // (op 6544) 


v_add_f32 v10, v18, v14 // (op 6545) 
v_add_f32 v11, v19, v15 // (op 6545) 
v_add_f32 v12, v20, v16 // (op 6545) 
v_add_f32 v13, v21, v17 // (op 6545) 


v_cvt_f16_f32 v14, v10 // (op 6546) 
v_cvt_f16_f32 v15, v11 // (op 6546) 
v_cvt_f16_f32 v16, v12 // (op 6546) 
v_cvt_f16_f32 v17, v13 // (op 6546) 


v_pack_b32_f16 v10, v14, v15 // (op 6532) 
v_pack_b32_f16 v11, v16, v17 // (op 6532) 

s_lshl_b32 s0, s4, 6 // (op 6532) 
s_add_u32 s0, s0, 64 // (op 6532) 
v_add_u32 v3, v9, s0 // (op 6532) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6532) Store value





s_lshl_b32 s0, s2, 5 // (op 6552) 
s_mul_i32 s1, 3, s0 // (op 6552) 
s_add_u32 s1, s1, 64 // (op 6552) 

v_add_u32 v3, v8, s1 // (op 6552) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6552) Load value

v_accvgpr_read v10, a44 // (op 6558) 
v_accvgpr_read v11, a45 // (op 6558) 
v_accvgpr_read v12, a46 // (op 6558) 
v_accvgpr_read v13, a47 // (op 6558) 
v_mul_f32 v14, v7, v10 // (op 6558) 
v_mul_f32 v15, v7, v11 // (op 6558) 
v_mul_f32 v16, v7, v12 // (op 6558) 
v_mul_f32 v17, v7, v13 // (op 6558) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6549) 
v_lshrrev_b32 v11, 16, v4 // (op 6549) 
v_cvt_f32_f16 v11, v11 // (op 6549) 
v_mul_f32 v18, v0, v10 // (op 6549) 
v_mul_f32 v19, v0, v11 // (op 6549) 
v_cvt_f32_f16 v10, v5 // (op 6549) 
v_lshrrev_b32 v11, 16, v5 // (op 6549) 
v_cvt_f32_f16 v11, v11 // (op 6549) 
v_mul_f32 v20, v0, v10 // (op 6549) 
v_mul_f32 v21, v0, v11 // (op 6549) 


v_add_f32 v10, v18, v14 // (op 6550) 
v_add_f32 v11, v19, v15 // (op 6550) 
v_add_f32 v12, v20, v16 // (op 6550) 
v_add_f32 v13, v21, v17 // (op 6550) 


v_cvt_f16_f32 v14, v10 // (op 6551) 
v_cvt_f16_f32 v15, v11 // (op 6551) 
v_cvt_f16_f32 v16, v12 // (op 6551) 
v_cvt_f16_f32 v17, v13 // (op 6551) 


v_pack_b32_f16 v10, v14, v15 // (op 6534) 
v_pack_b32_f16 v11, v16, v17 // (op 6534) 

s_lshl_b32 s0, s4, 5 // (op 6534) 
s_mul_i32 s1, 3, s0 // (op 6534) 
s_add_u32 s1, s1, 64 // (op 6534) 

v_add_u32 v3, v9, s1 // (op 6534) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6534) Store value





v_add_u32 v3, 32, v8 // (op 6450) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6450) Load value

v_accvgpr_read v10, a16 // (op 6437) 
v_accvgpr_read v11, a17 // (op 6437) 
v_accvgpr_read v12, a18 // (op 6437) 
v_accvgpr_read v13, a19 // (op 6437) 
v_mul_f32 v14, v7, v10 // (op 6437) 
v_mul_f32 v15, v7, v11 // (op 6437) 
v_mul_f32 v16, v7, v12 // (op 6437) 
v_mul_f32 v17, v7, v13 // (op 6437) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6449) 
v_lshrrev_b32 v11, 16, v4 // (op 6449) 
v_cvt_f32_f16 v11, v11 // (op 6449) 
v_mul_f32 v18, v0, v10 // (op 6449) 
v_mul_f32 v19, v0, v11 // (op 6449) 
v_cvt_f32_f16 v10, v5 // (op 6449) 
v_lshrrev_b32 v11, 16, v5 // (op 6449) 
v_cvt_f32_f16 v11, v11 // (op 6449) 
v_mul_f32 v20, v0, v10 // (op 6449) 
v_mul_f32 v21, v0, v11 // (op 6449) 


v_add_f32 v10, v18, v14 // (op 6438) 
v_add_f32 v11, v19, v15 // (op 6438) 
v_add_f32 v12, v20, v16 // (op 6438) 
v_add_f32 v13, v21, v17 // (op 6438) 


v_cvt_f16_f32 v14, v10 // (op 6439) 
v_cvt_f16_f32 v15, v11 // (op 6439) 
v_cvt_f16_f32 v16, v12 // (op 6439) 
v_cvt_f16_f32 v17, v13 // (op 6439) 


v_pack_b32_f16 v10, v14, v15 // (op 6441) 
v_pack_b32_f16 v11, v16, v17 // (op 6441) 

v_add_u32 v3, 32, v9 // (op 6441) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6441) Store value





s_lshl_b32 s0, s2, 5 // (op 6455) 
s_add_u32 s0, s0, 32 // (op 6455) 
v_add_u32 v3, v8, s0 // (op 6455) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6455) Load value

v_accvgpr_read v10, a20 // (op 6467) 
v_accvgpr_read v11, a21 // (op 6467) 
v_accvgpr_read v12, a22 // (op 6467) 
v_accvgpr_read v13, a23 // (op 6467) 
v_mul_f32 v14, v7, v10 // (op 6467) 
v_mul_f32 v15, v7, v11 // (op 6467) 
v_mul_f32 v16, v7, v12 // (op 6467) 
v_mul_f32 v17, v7, v13 // (op 6467) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6452) 
v_lshrrev_b32 v11, 16, v4 // (op 6452) 
v_cvt_f32_f16 v11, v11 // (op 6452) 
v_mul_f32 v18, v0, v10 // (op 6452) 
v_mul_f32 v19, v0, v11 // (op 6452) 
v_cvt_f32_f16 v10, v5 // (op 6452) 
v_lshrrev_b32 v11, 16, v5 // (op 6452) 
v_cvt_f32_f16 v11, v11 // (op 6452) 
v_mul_f32 v20, v0, v10 // (op 6452) 
v_mul_f32 v21, v0, v11 // (op 6452) 


v_add_f32 v10, v18, v14 // (op 6453) 
v_add_f32 v11, v19, v15 // (op 6453) 
v_add_f32 v12, v20, v16 // (op 6453) 
v_add_f32 v13, v21, v17 // (op 6453) 


v_cvt_f16_f32 v14, v10 // (op 6454) 
v_cvt_f16_f32 v15, v11 // (op 6454) 
v_cvt_f16_f32 v16, v12 // (op 6454) 
v_cvt_f16_f32 v17, v13 // (op 6454) 


v_pack_b32_f16 v10, v14, v15 // (op 6443) 
v_pack_b32_f16 v11, v16, v17 // (op 6443) 

s_lshl_b32 s0, s4, 5 // (op 6443) 
s_add_u32 s0, s0, 32 // (op 6443) 
v_add_u32 v3, v9, s0 // (op 6443) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6443) Store value





s_lshl_b32 s0, s2, 6 // (op 6460) 
s_add_u32 s0, s0, 32 // (op 6460) 
v_add_u32 v3, v8, s0 // (op 6460) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6460) Load value

v_accvgpr_read v10, a24 // (op 6469) 
v_accvgpr_read v11, a25 // (op 6469) 
v_accvgpr_read v12, a26 // (op 6469) 
v_accvgpr_read v13, a27 // (op 6469) 
v_mul_f32 v14, v7, v10 // (op 6469) 
v_mul_f32 v15, v7, v11 // (op 6469) 
v_mul_f32 v16, v7, v12 // (op 6469) 
v_mul_f32 v17, v7, v13 // (op 6469) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6457) 
v_lshrrev_b32 v11, 16, v4 // (op 6457) 
v_cvt_f32_f16 v11, v11 // (op 6457) 
v_mul_f32 v18, v0, v10 // (op 6457) 
v_mul_f32 v19, v0, v11 // (op 6457) 
v_cvt_f32_f16 v10, v5 // (op 6457) 
v_lshrrev_b32 v11, 16, v5 // (op 6457) 
v_cvt_f32_f16 v11, v11 // (op 6457) 
v_mul_f32 v20, v0, v10 // (op 6457) 
v_mul_f32 v21, v0, v11 // (op 6457) 


v_add_f32 v10, v18, v14 // (op 6458) 
v_add_f32 v11, v19, v15 // (op 6458) 
v_add_f32 v12, v20, v16 // (op 6458) 
v_add_f32 v13, v21, v17 // (op 6458) 


v_cvt_f16_f32 v14, v10 // (op 6459) 
v_cvt_f16_f32 v15, v11 // (op 6459) 
v_cvt_f16_f32 v16, v12 // (op 6459) 
v_cvt_f16_f32 v17, v13 // (op 6459) 


v_pack_b32_f16 v10, v14, v15 // (op 6445) 
v_pack_b32_f16 v11, v16, v17 // (op 6445) 

s_lshl_b32 s0, s4, 6 // (op 6445) 
s_add_u32 s0, s0, 32 // (op 6445) 
v_add_u32 v3, v9, s0 // (op 6445) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6445) Store value





s_lshl_b32 s0, s2, 5 // (op 6465) 
s_mul_i32 s1, 3, s0 // (op 6465) 
s_add_u32 s1, s1, 32 // (op 6465) 

v_add_u32 v3, v8, s1 // (op 6465) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6465) Load value

v_accvgpr_read v10, a28 // (op 6471) 
v_accvgpr_read v11, a29 // (op 6471) 
v_accvgpr_read v12, a30 // (op 6471) 
v_accvgpr_read v13, a31 // (op 6471) 
v_mul_f32 v14, v7, v10 // (op 6471) 
v_mul_f32 v15, v7, v11 // (op 6471) 
v_mul_f32 v16, v7, v12 // (op 6471) 
v_mul_f32 v17, v7, v13 // (op 6471) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6462) 
v_lshrrev_b32 v11, 16, v4 // (op 6462) 
v_cvt_f32_f16 v11, v11 // (op 6462) 
v_mul_f32 v18, v0, v10 // (op 6462) 
v_mul_f32 v19, v0, v11 // (op 6462) 
v_cvt_f32_f16 v10, v5 // (op 6462) 
v_lshrrev_b32 v11, 16, v5 // (op 6462) 
v_cvt_f32_f16 v11, v11 // (op 6462) 
v_mul_f32 v20, v0, v10 // (op 6462) 
v_mul_f32 v21, v0, v11 // (op 6462) 


v_add_f32 v10, v18, v14 // (op 6463) 
v_add_f32 v11, v19, v15 // (op 6463) 
v_add_f32 v12, v20, v16 // (op 6463) 
v_add_f32 v13, v21, v17 // (op 6463) 


v_cvt_f16_f32 v14, v10 // (op 6464) 
v_cvt_f16_f32 v15, v11 // (op 6464) 
v_cvt_f16_f32 v16, v12 // (op 6464) 
v_cvt_f16_f32 v17, v13 // (op 6464) 


v_pack_b32_f16 v10, v14, v15 // (op 6447) 
v_pack_b32_f16 v11, v16, v17 // (op 6447) 

s_lshl_b32 s0, s4, 5 // (op 6447) 
s_mul_i32 s1, 3, s0 // (op 6447) 
s_add_u32 s1, s1, 32 // (op 6447) 

v_add_u32 v3, v9, s1 // (op 6447) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6447) Store value





v_add_u32 v3, 192, v8 // (op 6885) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6885) Load value

v_accvgpr_read v10, a96 // (op 6872) 
v_accvgpr_read v11, a97 // (op 6872) 
v_accvgpr_read v12, a98 // (op 6872) 
v_accvgpr_read v13, a99 // (op 6872) 
v_mul_f32 v14, v7, v10 // (op 6872) 
v_mul_f32 v15, v7, v11 // (op 6872) 
v_mul_f32 v16, v7, v12 // (op 6872) 
v_mul_f32 v17, v7, v13 // (op 6872) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6884) 
v_lshrrev_b32 v11, 16, v4 // (op 6884) 
v_cvt_f32_f16 v11, v11 // (op 6884) 
v_mul_f32 v18, v0, v10 // (op 6884) 
v_mul_f32 v19, v0, v11 // (op 6884) 
v_cvt_f32_f16 v10, v5 // (op 6884) 
v_lshrrev_b32 v11, 16, v5 // (op 6884) 
v_cvt_f32_f16 v11, v11 // (op 6884) 
v_mul_f32 v20, v0, v10 // (op 6884) 
v_mul_f32 v21, v0, v11 // (op 6884) 


v_add_f32 v10, v18, v14 // (op 6873) 
v_add_f32 v11, v19, v15 // (op 6873) 
v_add_f32 v12, v20, v16 // (op 6873) 
v_add_f32 v13, v21, v17 // (op 6873) 


v_cvt_f16_f32 v14, v10 // (op 6874) 
v_cvt_f16_f32 v15, v11 // (op 6874) 
v_cvt_f16_f32 v16, v12 // (op 6874) 
v_cvt_f16_f32 v17, v13 // (op 6874) 


v_pack_b32_f16 v10, v14, v15 // (op 6876) 
v_pack_b32_f16 v11, v16, v17 // (op 6876) 

v_add_u32 v3, 192, v9 // (op 6876) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6876) Store value





s_lshl_b32 s0, s2, 5 // (op 6890) 
s_add_u32 s0, s0, 192 // (op 6890) 
v_add_u32 v3, v8, s0 // (op 6890) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6890) Load value

v_accvgpr_read v10, a100 // (op 6902) 
v_accvgpr_read v11, a101 // (op 6902) 
v_accvgpr_read v12, a102 // (op 6902) 
v_accvgpr_read v13, a103 // (op 6902) 
v_mul_f32 v14, v7, v10 // (op 6902) 
v_mul_f32 v15, v7, v11 // (op 6902) 
v_mul_f32 v16, v7, v12 // (op 6902) 
v_mul_f32 v17, v7, v13 // (op 6902) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6887) 
v_lshrrev_b32 v11, 16, v4 // (op 6887) 
v_cvt_f32_f16 v11, v11 // (op 6887) 
v_mul_f32 v18, v0, v10 // (op 6887) 
v_mul_f32 v19, v0, v11 // (op 6887) 
v_cvt_f32_f16 v10, v5 // (op 6887) 
v_lshrrev_b32 v11, 16, v5 // (op 6887) 
v_cvt_f32_f16 v11, v11 // (op 6887) 
v_mul_f32 v20, v0, v10 // (op 6887) 
v_mul_f32 v21, v0, v11 // (op 6887) 


v_add_f32 v10, v18, v14 // (op 6888) 
v_add_f32 v11, v19, v15 // (op 6888) 
v_add_f32 v12, v20, v16 // (op 6888) 
v_add_f32 v13, v21, v17 // (op 6888) 


v_cvt_f16_f32 v14, v10 // (op 6889) 
v_cvt_f16_f32 v15, v11 // (op 6889) 
v_cvt_f16_f32 v16, v12 // (op 6889) 
v_cvt_f16_f32 v17, v13 // (op 6889) 


v_pack_b32_f16 v10, v14, v15 // (op 6878) 
v_pack_b32_f16 v11, v16, v17 // (op 6878) 

s_lshl_b32 s0, s4, 5 // (op 6878) 
s_add_u32 s0, s0, 192 // (op 6878) 
v_add_u32 v3, v9, s0 // (op 6878) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6878) Store value





s_lshl_b32 s0, s2, 6 // (op 6895) 
s_add_u32 s0, s0, 192 // (op 6895) 
v_add_u32 v3, v8, s0 // (op 6895) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6895) Load value

v_accvgpr_read v10, a104 // (op 6904) 
v_accvgpr_read v11, a105 // (op 6904) 
v_accvgpr_read v12, a106 // (op 6904) 
v_accvgpr_read v13, a107 // (op 6904) 
v_mul_f32 v14, v7, v10 // (op 6904) 
v_mul_f32 v15, v7, v11 // (op 6904) 
v_mul_f32 v16, v7, v12 // (op 6904) 
v_mul_f32 v17, v7, v13 // (op 6904) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6892) 
v_lshrrev_b32 v11, 16, v4 // (op 6892) 
v_cvt_f32_f16 v11, v11 // (op 6892) 
v_mul_f32 v18, v0, v10 // (op 6892) 
v_mul_f32 v19, v0, v11 // (op 6892) 
v_cvt_f32_f16 v10, v5 // (op 6892) 
v_lshrrev_b32 v11, 16, v5 // (op 6892) 
v_cvt_f32_f16 v11, v11 // (op 6892) 
v_mul_f32 v20, v0, v10 // (op 6892) 
v_mul_f32 v21, v0, v11 // (op 6892) 


v_add_f32 v10, v18, v14 // (op 6893) 
v_add_f32 v11, v19, v15 // (op 6893) 
v_add_f32 v12, v20, v16 // (op 6893) 
v_add_f32 v13, v21, v17 // (op 6893) 


v_cvt_f16_f32 v14, v10 // (op 6894) 
v_cvt_f16_f32 v15, v11 // (op 6894) 
v_cvt_f16_f32 v16, v12 // (op 6894) 
v_cvt_f16_f32 v17, v13 // (op 6894) 


v_pack_b32_f16 v10, v14, v15 // (op 6880) 
v_pack_b32_f16 v11, v16, v17 // (op 6880) 

s_lshl_b32 s0, s4, 6 // (op 6880) 
s_add_u32 s0, s0, 192 // (op 6880) 
v_add_u32 v3, v9, s0 // (op 6880) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6880) Store value





s_lshl_b32 s0, s2, 5 // (op 6900) 
s_mul_i32 s1, 3, s0 // (op 6900) 
s_add_u32 s1, s1, 192 // (op 6900) 

v_add_u32 v3, v8, s1 // (op 6900) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6900) Load value

v_accvgpr_read v10, a108 // (op 6906) 
v_accvgpr_read v11, a109 // (op 6906) 
v_accvgpr_read v12, a110 // (op 6906) 
v_accvgpr_read v13, a111 // (op 6906) 
v_mul_f32 v14, v7, v10 // (op 6906) 
v_mul_f32 v15, v7, v11 // (op 6906) 
v_mul_f32 v16, v7, v12 // (op 6906) 
v_mul_f32 v17, v7, v13 // (op 6906) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6897) 
v_lshrrev_b32 v11, 16, v4 // (op 6897) 
v_cvt_f32_f16 v11, v11 // (op 6897) 
v_mul_f32 v18, v0, v10 // (op 6897) 
v_mul_f32 v19, v0, v11 // (op 6897) 
v_cvt_f32_f16 v10, v5 // (op 6897) 
v_lshrrev_b32 v11, 16, v5 // (op 6897) 
v_cvt_f32_f16 v11, v11 // (op 6897) 
v_mul_f32 v20, v0, v10 // (op 6897) 
v_mul_f32 v21, v0, v11 // (op 6897) 


v_add_f32 v10, v18, v14 // (op 6898) 
v_add_f32 v11, v19, v15 // (op 6898) 
v_add_f32 v12, v20, v16 // (op 6898) 
v_add_f32 v13, v21, v17 // (op 6898) 


v_cvt_f16_f32 v14, v10 // (op 6899) 
v_cvt_f16_f32 v15, v11 // (op 6899) 
v_cvt_f16_f32 v16, v12 // (op 6899) 
v_cvt_f16_f32 v17, v13 // (op 6899) 


v_pack_b32_f16 v10, v14, v15 // (op 6882) 
v_pack_b32_f16 v11, v16, v17 // (op 6882) 

s_lshl_b32 s0, s4, 5 // (op 6882) 
s_mul_i32 s1, 3, s0 // (op 6882) 
s_add_u32 s1, s1, 192 // (op 6882) 

v_add_u32 v3, v9, s1 // (op 6882) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6882) Store value





v_mov_b32 v3, v8 // (op 129) reg expression

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 129) Load value

v_accvgpr_read v10, a0 // (op 125) 
v_accvgpr_read v11, a1 // (op 125) 
v_accvgpr_read v12, a2 // (op 125) 
v_accvgpr_read v13, a3 // (op 125) 
v_mul_f32 v14, v7, v10 // (op 125) 
v_mul_f32 v15, v7, v11 // (op 125) 
v_mul_f32 v16, v7, v12 // (op 125) 
v_mul_f32 v17, v7, v13 // (op 125) 

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





s_lshl_b32 s0, s2, 5 // (op 6358) 
v_add_u32 v3, v8, s0 // (op 6358) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6358) Load value

v_accvgpr_read v10, a4 // (op 6354) 
v_accvgpr_read v11, a5 // (op 6354) 
v_accvgpr_read v12, a6 // (op 6354) 
v_accvgpr_read v13, a7 // (op 6354) 
v_mul_f32 v14, v7, v10 // (op 6354) 
v_mul_f32 v15, v7, v11 // (op 6354) 
v_mul_f32 v16, v7, v12 // (op 6354) 
v_mul_f32 v17, v7, v13 // (op 6354) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6359) 
v_lshrrev_b32 v11, 16, v4 // (op 6359) 
v_cvt_f32_f16 v11, v11 // (op 6359) 
v_mul_f32 v18, v0, v10 // (op 6359) 
v_mul_f32 v19, v0, v11 // (op 6359) 
v_cvt_f32_f16 v10, v5 // (op 6359) 
v_lshrrev_b32 v11, 16, v5 // (op 6359) 
v_cvt_f32_f16 v11, v11 // (op 6359) 
v_mul_f32 v20, v0, v10 // (op 6359) 
v_mul_f32 v21, v0, v11 // (op 6359) 


v_add_f32 v10, v18, v14 // (op 6355) 
v_add_f32 v11, v19, v15 // (op 6355) 
v_add_f32 v12, v20, v16 // (op 6355) 
v_add_f32 v13, v21, v17 // (op 6355) 


v_cvt_f16_f32 v14, v10 // (op 6356) 
v_cvt_f16_f32 v15, v11 // (op 6356) 
v_cvt_f16_f32 v16, v12 // (op 6356) 
v_cvt_f16_f32 v17, v13 // (op 6356) 


v_pack_b32_f16 v10, v14, v15 // (op 6357) 
v_pack_b32_f16 v11, v16, v17 // (op 6357) 

s_lshl_b32 s0, s4, 5 // (op 6357) 
v_add_u32 v3, v9, s0 // (op 6357) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6357) Store value





s_lshl_b32 s0, s2, 6 // (op 6372) 
v_add_u32 v3, v8, s0 // (op 6372) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6372) Load value

v_accvgpr_read v10, a8 // (op 6368) 
v_accvgpr_read v11, a9 // (op 6368) 
v_accvgpr_read v12, a10 // (op 6368) 
v_accvgpr_read v13, a11 // (op 6368) 
v_mul_f32 v14, v7, v10 // (op 6368) 
v_mul_f32 v15, v7, v11 // (op 6368) 
v_mul_f32 v16, v7, v12 // (op 6368) 
v_mul_f32 v17, v7, v13 // (op 6368) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6373) 
v_lshrrev_b32 v11, 16, v4 // (op 6373) 
v_cvt_f32_f16 v11, v11 // (op 6373) 
v_mul_f32 v18, v0, v10 // (op 6373) 
v_mul_f32 v19, v0, v11 // (op 6373) 
v_cvt_f32_f16 v10, v5 // (op 6373) 
v_lshrrev_b32 v11, 16, v5 // (op 6373) 
v_cvt_f32_f16 v11, v11 // (op 6373) 
v_mul_f32 v20, v0, v10 // (op 6373) 
v_mul_f32 v21, v0, v11 // (op 6373) 


v_add_f32 v10, v18, v14 // (op 6369) 
v_add_f32 v11, v19, v15 // (op 6369) 
v_add_f32 v12, v20, v16 // (op 6369) 
v_add_f32 v13, v21, v17 // (op 6369) 


v_cvt_f16_f32 v14, v10 // (op 6370) 
v_cvt_f16_f32 v15, v11 // (op 6370) 
v_cvt_f16_f32 v16, v12 // (op 6370) 
v_cvt_f16_f32 v17, v13 // (op 6370) 


v_pack_b32_f16 v10, v14, v15 // (op 6371) 
v_pack_b32_f16 v11, v16, v17 // (op 6371) 

s_lshl_b32 s0, s4, 6 // (op 6371) 
v_add_u32 v3, v9, s0 // (op 6371) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6371) Store value





s_lshl_b32 s0, s2, 5 // (op 6386) 
s_mul_i32 s1, 3, s0 // (op 6386) 

v_add_u32 v3, v8, s1 // (op 6386) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6386) Load value

v_accvgpr_read v10, a12 // (op 6382) 
v_accvgpr_read v11, a13 // (op 6382) 
v_accvgpr_read v12, a14 // (op 6382) 
v_accvgpr_read v13, a15 // (op 6382) 
v_mul_f32 v14, v7, v10 // (op 6382) 
v_mul_f32 v15, v7, v11 // (op 6382) 
v_mul_f32 v16, v7, v12 // (op 6382) 
v_mul_f32 v17, v7, v13 // (op 6382) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6387) 
v_lshrrev_b32 v11, 16, v4 // (op 6387) 
v_cvt_f32_f16 v11, v11 // (op 6387) 
v_mul_f32 v18, v0, v10 // (op 6387) 
v_mul_f32 v19, v0, v11 // (op 6387) 
v_cvt_f32_f16 v10, v5 // (op 6387) 
v_lshrrev_b32 v11, 16, v5 // (op 6387) 
v_cvt_f32_f16 v11, v11 // (op 6387) 
v_mul_f32 v20, v0, v10 // (op 6387) 
v_mul_f32 v21, v0, v11 // (op 6387) 


v_add_f32 v10, v18, v14 // (op 6383) 
v_add_f32 v11, v19, v15 // (op 6383) 
v_add_f32 v12, v20, v16 // (op 6383) 
v_add_f32 v13, v21, v17 // (op 6383) 


v_cvt_f16_f32 v14, v10 // (op 6384) 
v_cvt_f16_f32 v15, v11 // (op 6384) 
v_cvt_f16_f32 v16, v12 // (op 6384) 
v_cvt_f16_f32 v17, v13 // (op 6384) 


v_pack_b32_f16 v10, v14, v15 // (op 6385) 
v_pack_b32_f16 v11, v16, v17 // (op 6385) 

s_lshl_b32 s0, s4, 5 // (op 6385) 
s_mul_i32 s1, 3, s0 // (op 6385) 

v_add_u32 v3, v9, s1 // (op 6385) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6385) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ConditionalBottom_0_beta_116:








































s_endpgm  // End of RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__end-RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 280
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 81920 // lds bytes
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
    macrotile: [256, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 81920
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 150
    .agpr_count: 128
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}}
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

