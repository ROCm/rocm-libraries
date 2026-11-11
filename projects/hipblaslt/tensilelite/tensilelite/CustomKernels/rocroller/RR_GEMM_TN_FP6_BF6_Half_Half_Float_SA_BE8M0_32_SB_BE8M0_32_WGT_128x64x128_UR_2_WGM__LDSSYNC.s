// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC
.p2align 8
.type RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC,@function
RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 4550) 
s_mul_hi_i32 s1, s16, s10 // (op 4807) 
v_lshrrev_b32 v0, 7, v1 // (op 4612) 
v_and_b32 v3, 15, v1 // (op 4612) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 4612) 


v_and_b32 v0, 63, v1 // (op 4612) 
v_lshrrev_b32 v3, 4, v0 // (op 4612) 

v_lshlrev_b32 v0, 5, v3 // (op 4612) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 4612) 


v_mul_lo_u32 v0, v3, 6 // (op 4612) 

v_lshrrev_b32 v3, 3, v0 // (op 4612) 

v_lshrrev_b32 v0, 6, v1 // (op 4670) 
v_and_b32 v4, 1, v0 // (op 4670) 

v_and_b32 v0, 15, v1 // (op 4670) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 4670) 


v_and_b32 v0, 63, v1 // (op 4670) 
v_lshrrev_b32 v4, 4, v0 // (op 4670) 

v_lshlrev_b32 v0, 5, v4 // (op 4670) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 4670) 


v_mul_lo_u32 v0, v4, 6 // (op 4670) 

v_lshrrev_b32 v4, 3, v0 // (op 4670) 

v_lshrrev_b32 v0, 7, v1 // (op 4728) 
v_and_b32 v5, 15, v1 // (op 4728) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 4728) 


v_and_b32 v0, 63, v1 // (op 4728) 
v_lshrrev_b32 v5, 4, v0 // (op 4728) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 4728) 


v_lshrrev_b32 v5, 6, v1 // (op 4786) 
v_and_b32 v6, 1, v5 // (op 4786) 

v_and_b32 v5, 15, v1 // (op 4786) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 4786) 


v_and_b32 v5, 63, v1 // (op 4786) 
v_lshrrev_b32 v6, 4, v5 // (op 4786) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 4786) 


v_lshrrev_b32 v6, 7, v1 // (op 4838) 
v_and_b32 v7, 15, v1 // (op 4838) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 4838) 


v_and_b32 v6, 63, v1 // (op 4838) 
v_lshrrev_b32 v7, 4, v6 // (op 4838) 

v_lshlrev_b32 v6, 5, v7 // (op 4838) 

v_lshl_add_u32 v7, v8, 7, v6 // (op 4838) 


v_mul_lo_u32 v6, v7, 6 // (op 4838) 

v_lshrrev_b32 v7, 3, v6 // (op 4838) 

v_lshrrev_b32 v6, 6, v1 // (op 4867) 
v_and_b32 v8, 1, v6 // (op 4867) 

v_and_b32 v6, 15, v1 // (op 4867) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 4867) 


v_and_b32 v6, 63, v1 // (op 4867) 
v_lshrrev_b32 v8, 4, v6 // (op 4867) 

v_lshlrev_b32 v6, 5, v8 // (op 4867) 

v_lshl_add_u32 v8, v9, 7, v6 // (op 4867) 


v_mul_lo_u32 v6, v8, 6 // (op 4867) 

v_lshrrev_b32 v8, 3, v6 // (op 4867) 

v_lshrrev_b32 v6, 7, v1 // (op 4896) 
v_and_b32 v9, 15, v1 // (op 4896) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 4896) 


v_and_b32 v6, 63, v1 // (op 4896) 
v_lshrrev_b32 v9, 4, v6 // (op 4896) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 4896) 


v_lshrrev_b32 v9, 6, v1 // (op 4925) 
v_and_b32 v10, 1, v9 // (op 4925) 

v_and_b32 v9, 15, v1 // (op 4925) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 4925) 


v_and_b32 v9, 63, v1 // (op 4925) 
v_lshrrev_b32 v10, 4, v9 // (op 4925) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 4925) 


v_lshrrev_b32 v10, 7, v1 // (op 4550) 
s_add_i32 s18, s0, s16 // (op 4550) 

v_and_b32 v11, 63, v1 // (op 4550) 
s_ashr_i32 s0, s18, 31 // (op 4550) 
v_lshrrev_b32 v12, 4, v11 // (op 4550) 
s_cmp_eq_i32 s10, 0 // (op 4550) 
s_mov_b32 s19, scc // (op 4550) 
v_lshlrev_b32 v13, 2, v12 // (op 4550) 

s_cmp_eq_i32 s19, 0 // (op 4550) 
s_cselect_b32 s49, 0, -1 // (op 4550) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 4550) 


s_and_b32 s19, s13, 31 // (op 4550) 
v_lshrrev_b32 v10, 2, v11 // (op 4550) 

v_and_b32 v11, 3, v10 // (op 4550) 

v_lshrrev_b32 v10, 7, v1 // (op 4807) 
v_and_b32 v13, 63, v1 // (op 4807) 
s_lshl_b32 s50, 1, s19 // (op 4550) 
v_and_b32 v14, 3, v1 // (op 4550) 
s_add_i32 s51, s50, s49 // (op 4550) 


s_and_b32 s49, s0, s51 // (op 4550) 


s_add_i32 s0, s1, s16 // (op 4807) 

s_ashr_i32 s1, s0, 31 // (op 4807) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 4550) 


s_add_i32 s50, s18, s49 // (op 4550) 


v_lshrrev_b32 v11, 6, v1 // (op 4550) 
s_ashr_i32 s18, s50, s19 // (op 4550) 


v_and_b32 v14, 1, v11 // (op 4550) 

s_lshl_b32 s19, s13, 24 // (op 4550) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 4550) 


s_ashr_i32 s49, s19, 31 // (op 4550) 

s_xor_b32 s19, s18, s49 // (op 4550) 

s_sub_i32 s18, s19, s49 // (op 4550) 


s_mul_hi_i32 s19, s18, s14 // (op 4550) 
s_add_i32 s49, s19, s18 // (op 4550) 

s_ashr_i32 s19, s49, 31 // (op 4550) 
s_cmp_eq_i32 s14, 0 // (op 4550) 
s_mov_b32 s50, scc // (op 4550) 
s_cmp_eq_i32 s50, 0 // (op 4550) 
s_cselect_b32 s51, 0, -1 // (op 4550) 

s_and_b32 s50, s15, 31 // (op 4550) 
v_lshrrev_b32 v14, 4, v13 // (op 4807) 
s_cmp_eq_i32 s10, 0 // (op 4807) 
s_mov_b32 s52, scc // (op 4807) 
v_lshlrev_b32 v15, 2, v14 // (op 4807) 

s_cmp_eq_i32 s52, 0 // (op 4807) 
s_cselect_b32 s53, 0, -1 // (op 4807) 

s_lshl_b32 s52, 1, s50 // (op 4550) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 4807) 


v_lshrrev_b32 v10, 2, v13 // (op 4807) 

s_add_i32 s54, s52, s51 // (op 4550) 


s_and_b32 s51, s19, s54 // (op 4550) 


s_add_i32 s19, s49, s51 // (op 4550) 


s_ashr_i32 s49, s19, s50 // (op 4550) 


s_lshl_b32 s19, s15, 24 // (op 4550) 
s_ashr_i32 s50, s19, 31 // (op 4550) 

s_xor_b32 s19, s49, s50 // (op 4550) 

s_sub_i32 s49, s19, s50 // (op 4550) 


s_mul_i32 s19, s18, s11 // (op 4550) 
s_sub_u32 s50, s16, s19 // (op 4550) 

s_mul_i32 s19, s49, s11 // (op 4550) 
s_add_u32 s19, s19, s50 // (op 4550) 

s_mul_hi_i32 s50, s16, s9 // (op 4550) 
s_add_i32 s51, s50, s16 // (op 4550) 

s_ashr_i32 s50, s51, 31 // (op 4550) 
s_cmp_eq_i32 s9, 0 // (op 4550) 
s_mov_b32 s52, scc // (op 4550) 
s_cmp_eq_i32 s52, 0 // (op 4550) 
s_cselect_b32 s54, 0, -1 // (op 4550) 

s_and_b32 s52, s13, 31 // (op 4807) 
v_and_b32 v13, 3, v10 // (op 4807) 

s_lshl_b32 s55, 1, s52 // (op 4807) 
v_and_b32 v10, 3, v1 // (op 4807) 
s_add_i32 s56, s55, s53 // (op 4807) 


s_and_b32 s53, s1, s56 // (op 4807) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 4807) 


s_add_i32 s1, s0, s53 // (op 4807) 


v_lshrrev_b32 v10, 6, v1 // (op 4807) 
s_ashr_i32 s0, s1, s52 // (op 4807) 


v_and_b32 v13, 1, v10 // (op 4807) 

s_lshl_b32 s1, s13, 24 // (op 4807) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 4807) 


s_ashr_i32 s52, s1, 31 // (op 4807) 

s_xor_b32 s1, s0, s52 // (op 4807) 

s_sub_i32 s0, s1, s52 // (op 4807) 


s_mul_hi_i32 s1, s0, s14 // (op 4807) 
s_add_i32 s52, s1, s0 // (op 4807) 

s_ashr_i32 s1, s52, 31 // (op 4807) 
s_cmp_eq_i32 s14, 0 // (op 4807) 
s_mov_b32 s53, scc // (op 4807) 
s_cmp_eq_i32 s53, 0 // (op 4807) 
s_cselect_b32 s55, 0, -1 // (op 4807) 

s_and_b32 s53, s15, 31 // (op 4807) 
s_lshl_b32 s56, 1, s53 // (op 4807) 
s_add_i32 s57, s56, s55 // (op 4807) 


s_and_b32 s55, s1, s57 // (op 4807) 


s_add_i32 s1, s52, s55 // (op 4807) 


s_ashr_i32 s52, s1, s53 // (op 4807) 


s_lshl_b32 s1, s15, 24 // (op 4807) 
s_ashr_i32 s53, s1, 31 // (op 4807) 

s_xor_b32 s1, s52, s53 // (op 4807) 

s_sub_i32 s52, s1, s53 // (op 4807) 


s_mul_i32 s1, s0, s11 // (op 4807) 
s_sub_u32 s53, s16, s1 // (op 4807) 

s_mul_i32 s1, s52, s11 // (op 4807) 
s_add_u32 s1, s1, s53 // (op 4807) 

s_mul_hi_i32 s53, s16, s9 // (op 4807) 
s_add_i32 s55, s53, s16 // (op 4807) 

s_ashr_i32 s53, s55, 31 // (op 4807) 
s_cmp_eq_i32 s9, 0 // (op 4807) 
s_mov_b32 s56, scc // (op 4807) 
s_cmp_eq_i32 s56, 0 // (op 4807) 
s_cselect_b32 s57, 0, -1 // (op 4807) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 4550) 
s_lshl_b32 s58, 1, s56 // (op 4550) 
s_add_i32 s59, s58, s54 // (op 4550) 


s_and_b32 s54, s50, s59 // (op 4550) 


s_add_i32 s50, s51, s54 // (op 4550) 


s_ashr_i32 s51, s50, s56 // (op 4550) 


s_lshl_b32 s50, s45, 24 // (op 4550) 
s_ashr_i32 s54, s50, 31 // (op 4550) 

s_xor_b32 s50, s51, s54 // (op 4550) 

s_sub_i32 s51, s50, s54 // (op 4550) 


s_and_b32 s50, s45, 31 // (op 4807) 
s_lshl_b32 s54, 1, s50 // (op 4807) 
s_add_i32 s56, s54, s57 // (op 4807) 


s_and_b32 s54, s53, s56 // (op 4807) 


s_add_i32 s53, s55, s54 // (op 4807) 


s_ashr_i32 s54, s53, s50 // (op 4807) 


s_lshl_b32 s50, s45, 24 // (op 4807) 
s_ashr_i32 s53, s50, 31 // (op 4807) 

s_xor_b32 s50, s54, s53 // (op 4807) 

s_sub_i32 s54, s50, s53 // (op 4807) 


s_add_u32 s56, s6, 64 // (op 4550) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 4550) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 4550) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 4550) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 4550) 

s_mul_i32 s50, s11, s56 // (op 4550) 
s_mul_i32 s53, s8, s50 // (op 4550) 

s_mul_i32 s50, s51, s53 // (op 4550) 

s_sub_u32 s53, s16, s50 // (op 4550) 

s_mov_b32 s50, s53 // (op 4550) 
s_mul_hi_i32 s55, s50, s12 // (op 4550) 
s_add_i32 s58, s55, s50 // (op 4550) 


s_ashr_i32 s50, s58, 31 // (op 4550) 
s_cmp_eq_i32 s12, 0 // (op 4550) 
s_mov_b32 s55, scc // (op 4550) 
s_cmp_eq_i32 s55, 0 // (op 4550) 
s_cselect_b32 s59, 0, -1 // (op 4550) 

s_and_b32 s55, s44, 31 // (op 4550) 
s_lshl_b32 s60, 1, s55 // (op 4550) 
s_add_i32 s61, s60, s59 // (op 4550) 


s_and_b32 s59, s50, s61 // (op 4550) 


s_add_i32 s50, s58, s59 // (op 4550) 


s_ashr_i32 s58, s50, s55 // (op 4550) 


s_lshl_b32 s50, s44, 24 // (op 4550) 
s_ashr_i32 s55, s50, 31 // (op 4550) 

s_xor_b32 s50, s58, s55 // (op 4550) 

s_sub_i32 s58, s50, s55 // (op 4550) 


s_mul_i32 s50, s46, s11 // (op 4550) 
s_add_u32 s60, s20, 128 // (op 4550) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 4550) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 4550) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 4550) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 7 // (op 4550) 

s_sub_i32 s55, s60, s50 // (op 4550) 

s_mul_i32 s59, s58, s55 // (op 4550) 

s_sub_u32 s55, s53, s59 // (op 4550) 


s_add_u32 s53, s55, s50 // (op 4550) 


s_cmp_eq_i32 s51, 0 // (op 4550) 
s_mov_b32 s50, scc // (op 4550) 

s_cmp_eq_i32 s50, 0 // (op 4550) 
s_cselect_b32 s51, s53, s19 // (op 4550) 


s_mul_i32 s19, s49, s56 // (op 4550) 


v_lshl_add_u32 v13, s51, 7, v12 // (op 4550) 


s_sub_i32 s49, s18, s19 // (op 4550) 


s_cmp_eq_i32 s50, 0 // (op 4550) 
s_cselect_b32 s51, s58, s49 // (op 4550) 



v_lshl_add_u32 v12, s51, 6, v11 // (op 4550) 


v_mul_lo_u32 v11, v12, s2 // (op 4550) 

v_add_lshl_u32 v12, v13, v11, 1 // (op 4550) 


s_add_u32 s18, s6, 64 // (op 4807) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 4807) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 4807) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 4807) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 6 // (op 4807) 

s_mul_i32 s53, s11, s18 // (op 4807) 
s_mul_i32 s49, s8, s53 // (op 4807) 

s_mul_i32 s53, s54, s49 // (op 4807) 

s_sub_u32 s49, s16, s53 // (op 4807) 

s_mov_b32 s53, s49 // (op 4807) 
s_mul_hi_i32 s50, s53, s12 // (op 4807) 
s_add_i32 s51, s50, s53 // (op 4807) 


s_ashr_i32 s50, s51, 31 // (op 4807) 
s_cmp_eq_i32 s12, 0 // (op 4807) 
s_mov_b32 s53, scc // (op 4807) 
s_cmp_eq_i32 s53, 0 // (op 4807) 
s_cselect_b32 s55, 0, -1 // (op 4807) 

s_and_b32 s53, s44, 31 // (op 4807) 
s_lshl_b32 s56, 1, s53 // (op 4807) 
s_add_i32 s57, s56, s55 // (op 4807) 


s_and_b32 s55, s50, s57 // (op 4807) 


s_add_i32 s50, s51, s55 // (op 4807) 


s_ashr_i32 s51, s50, s53 // (op 4807) 


s_lshl_b32 s50, s44, 24 // (op 4807) 
s_ashr_i32 s53, s50, 31 // (op 4807) 

s_xor_b32 s50, s51, s53 // (op 4807) 

s_sub_i32 s51, s50, s53 // (op 4807) 


s_mul_i32 s50, s46, s11 // (op 4807) 
s_add_u32 s56, s20, 128 // (op 4807) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 4807) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 4807) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 4807) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 4807) 

s_sub_i32 s53, s56, s50 // (op 4807) 

s_mul_i32 s55, s51, s53 // (op 4807) 

s_sub_u32 s53, s49, s55 // (op 4807) 


s_add_u32 s49, s53, s50 // (op 4807) 


s_cmp_eq_i32 s54, 0 // (op 4807) 
s_mov_b32 s50, scc // (op 4807) 

s_cmp_eq_i32 s50, 0 // (op 4807) 
s_cselect_b32 s53, s49, s1 // (op 4807) 


s_mul_i32 s1, s52, s18 // (op 4807) 


v_lshl_add_u32 v11, s53, 7, v14 // (op 4807) 


s_sub_i32 s49, s0, s1 // (op 4807) 


s_cmp_eq_i32 s50, 0 // (op 4807) 
s_cselect_b32 s0, s51, s49 // (op 4807) 



v_lshl_add_u32 v13, s0, 6, v10 // (op 4807) 


v_mul_lo_u32 v10, v13, s4 // (op 4807) 

v_add_lshl_u32 v13, v11, v10, 1 // (op 4807) 


s_mul_i32 s1, s2, s7 // (op 4552) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 4552) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 4552) least significant
s_mul_hi_u32 s19, s2, s6 // (op 4552) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4552) most significant: sum
s_add_u32 s1, s1, s19 // (op 4552) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 4552) 

s_mul_i32 s1, s4, s7 // (op 4809) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 4809) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 4809) least significant
s_mul_hi_u32 s50, s4, s6 // (op 4809) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 4809) most significant: sum
s_add_u32 s1, s1, s50 // (op 4809) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 4809) 

s_mov_b32 s52, s22 // (op 4552) reg expression
s_mov_b32 s53, s23 // (op 4552) reg expression
s_mov_b32 s54, s18 // (op 4552) reg expression
s_mov_b32 s55, 131072 // (op 4552) call()

s_mov_b32 s56, s24 // (op 4809) reg expression

s_mov_b32 s57, s25 // (op 4809) reg expression
s_mov_b32 s58, s50 // (op 4809) reg expression
s_mov_b32 s59, 131072 // (op 4809) call()


v_mov_b32 v10, s47 // (op 2) Move value
v_mov_b32 v11, s48 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 340) call()
v_accvgpr_write a5, 0.00000 // (op 340) call()
v_accvgpr_write a6, 0.00000 // (op 340) call()
v_accvgpr_write a7, 0.00000 // (op 340) call()
v_accvgpr_write a8, 0.00000 // (op 560) call()
v_accvgpr_write a9, 0.00000 // (op 560) call()
v_accvgpr_write a10, 0.00000 // (op 560) call()
v_accvgpr_write a11, 0.00000 // (op 560) call()
v_accvgpr_write a12, 0.00000 // (op 715) call()
v_accvgpr_write a13, 0.00000 // (op 715) call()
v_accvgpr_write a14, 0.00000 // (op 715) call()
v_accvgpr_write a15, 0.00000 // (op 715) call()
v_accvgpr_write a16, 0.00000 // (op 946) call()
v_accvgpr_write a17, 0.00000 // (op 946) call()
v_accvgpr_write a18, 0.00000 // (op 946) call()
v_accvgpr_write a19, 0.00000 // (op 946) call()
v_accvgpr_write a20, 0.00000 // (op 1101) call()
v_accvgpr_write a21, 0.00000 // (op 1101) call()
v_accvgpr_write a22, 0.00000 // (op 1101) call()
v_accvgpr_write a23, 0.00000 // (op 1101) call()
v_accvgpr_write a24, 0.00000 // (op 1332) call()
v_accvgpr_write a25, 0.00000 // (op 1332) call()
v_accvgpr_write a26, 0.00000 // (op 1332) call()
v_accvgpr_write a27, 0.00000 // (op 1332) call()
v_accvgpr_write a28, 0.00000 // (op 1487) call()
v_accvgpr_write a29, 0.00000 // (op 1487) call()
v_accvgpr_write a30, 0.00000 // (op 1487) call()
v_accvgpr_write a31, 0.00000 // (op 1487) call()
s_mul_hi_i32 s0, s16, s10 // (op 4312) 
v_lshrrev_b32 v14, 1, v1 // (op 4312) 
s_add_i32 s1, s0, s16 // (op 4312) 

v_and_b32 v15, -2, v1 // (op 4312) 
s_ashr_i32 s0, s1, 31 // (op 4312) 
v_sub_u32 v16, v1, v15 // (op 4312) 

s_cmp_eq_i32 s10, 0 // (op 4312) 
s_mov_b32 s18, scc // (op 4312) 
v_lshlrev_b32 v15, 6, v16 // (op 4312) 

s_cmp_eq_i32 s18, 0 // (op 4312) 
s_cselect_b32 s19, 0, -1 // (op 4312) 

v_mov_b32 v17, 0 // (op 4312) convert
v_mov_b32 v16, v15 // (op 4312) convert

s_and_b32 s18, s13, 31 // (op 4312) 
s_lshl_b32 s22, 1, s18 // (op 4312) 
s_add_i32 s23, s22, s19 // (op 4312) 


s_and_b32 s19, s0, s23 // (op 4312) 


s_add_i32 s0, s1, s19 // (op 4312) 


s_ashr_i32 s1, s0, s18 // (op 4312) 


s_lshl_b32 s0, s13, 24 // (op 4312) 
s_ashr_i32 s18, s0, 31 // (op 4312) 

s_xor_b32 s0, s1, s18 // (op 4312) 

s_sub_i32 s1, s0, s18 // (op 4312) 


s_mul_hi_i32 s0, s1, s14 // (op 4312) 
s_add_i32 s18, s0, s1 // (op 4312) 

s_ashr_i32 s0, s18, 31 // (op 4312) 
s_cmp_eq_i32 s14, 0 // (op 4312) 
s_mov_b32 s19, scc // (op 4312) 
s_cmp_eq_i32 s19, 0 // (op 4312) 
s_cselect_b32 s22, 0, -1 // (op 4312) 

s_and_b32 s19, s15, 31 // (op 4312) 
s_lshl_b32 s23, 1, s19 // (op 4312) 
s_add_i32 s24, s23, s22 // (op 4312) 


s_and_b32 s25, s0, s24 // (op 4312) 


s_add_i32 s0, s18, s25 // (op 4312) 


s_ashr_i32 s18, s0, s19 // (op 4312) 


s_lshl_b32 s0, s15, 24 // (op 4312) 
s_ashr_i32 s19, s0, 31 // (op 4312) 

s_xor_b32 s0, s18, s19 // (op 4312) 

s_sub_i32 s18, s0, s19 // (op 4312) 


s_mul_i32 s0, s1, s11 // (op 4312) 

s_sub_u32 s1, s16, s0 // (op 4312) 

s_mul_i32 s0, s18, s11 // (op 4312) 
s_add_u32 s0, s0, s1 // (op 4312) 


s_mul_hi_i32 s1, s16, s9 // (op 4312) 
s_add_i32 s18, s1, s16 // (op 4312) 

s_ashr_i32 s1, s18, 31 // (op 4312) 
s_cmp_eq_i32 s9, 0 // (op 4312) 
s_mov_b32 s19, scc // (op 4312) 
s_cmp_eq_i32 s19, 0 // (op 4312) 
s_cselect_b32 s22, 0, -1 // (op 4312) 

s_and_b32 s19, s45, 31 // (op 4312) 
s_lshl_b32 s23, 1, s19 // (op 4312) 
s_add_i32 s24, s23, s22 // (op 4312) 


s_and_b32 s25, s1, s24 // (op 4312) 


s_add_i32 s1, s18, s25 // (op 4312) 


s_ashr_i32 s18, s1, s19 // (op 4312) 


s_lshl_b32 s1, s45, 24 // (op 4312) 
s_ashr_i32 s19, s1, 31 // (op 4312) 

s_xor_b32 s1, s18, s19 // (op 4312) 

s_sub_i32 s18, s1, s19 // (op 4312) 


s_add_u32 s22, s6, 64 // (op 4312) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4312) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4312) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4312) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4312) 

s_mul_i32 s1, s11, s22 // (op 4312) 

s_mul_i32 s19, s8, s1 // (op 4312) 

s_mul_i32 s1, s18, s19 // (op 4312) 

s_sub_u32 s19, s16, s1 // (op 4312) 

s_mov_b32 s1, s19 // (op 4312) 
s_mul_hi_i32 s22, s1, s12 // (op 4312) 
s_add_i32 s23, s22, s1 // (op 4312) 


s_ashr_i32 s1, s23, 31 // (op 4312) 
s_cmp_eq_i32 s12, 0 // (op 4312) 
s_mov_b32 s22, scc // (op 4312) 
s_cmp_eq_i32 s22, 0 // (op 4312) 
s_cselect_b32 s24, 0, -1 // (op 4312) 

s_and_b32 s22, s44, 31 // (op 4312) 
s_lshl_b32 s25, 1, s22 // (op 4312) 
s_add_i32 s47, s25, s24 // (op 4312) 


s_and_b32 s24, s1, s47 // (op 4312) 


s_add_i32 s1, s23, s24 // (op 4312) 


s_ashr_i32 s23, s1, s22 // (op 4312) 


s_lshl_b32 s1, s44, 24 // (op 4312) 
s_ashr_i32 s22, s1, 31 // (op 4312) 

s_xor_b32 s1, s23, s22 // (op 4312) 

s_sub_i32 s23, s1, s22 // (op 4312) 


s_mul_i32 s1, s46, s11 // (op 4312) 
s_add_u32 s24, s20, 128 // (op 4312) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4312) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 4312) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 4312) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 7 // (op 4312) 

s_sub_i32 s22, s24, s1 // (op 4312) 

s_mul_i32 s24, s23, s22 // (op 4312) 


s_sub_u32 s25, s19, s24 // (op 4312) 


s_add_u32 s19, s25, s1 // (op 4312) 


s_cmp_eq_i32 s18, 0 // (op 4312) 
s_mov_b32 s1, scc // (op 4312) 

s_cmp_eq_i32 s1, 0 // (op 4312) 
s_cselect_b32 s18, s19, s0 // (op 4312) 



v_lshl_add_u32 v15, s18, 7, v14 // (op 4312) 


v_mul_lo_u32 v19, v15, s27 // (op 4312) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 4312) most significant: high of low * low
v_mul_lo_u32 v18, v15, s26 // (op 4312) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 4312) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 4312) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 4312) most significant half



v_mul_lo_u32 v15, v19, 6 // (op 4312) most significant: low of high * low
v_mul_hi_u32 v16, v18, 6 // (op 4312) most significant: high of low * low
v_mul_lo_u32 v14, v18, 6 // (op 4312) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 4312) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 4312) 

v_mov_b32 v14, v16 // (op 4312) convert

s_mul_i32 s1, s21, 6 // (op 4314) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 4314) least significant
s_mul_hi_u32 s18, s20, 6 // (op 4314) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4314) most significant: sum

s_mul_i32 s19, s26, s1 // (op 4314) most significant: low of low * high
s_mul_i32 s22, s27, s0 // (op 4314) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 4314) least significant
s_mul_hi_u32 s23, s26, s0 // (op 4314) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 4314) most significant: sum
s_add_u32 s19, s19, s23 // (op 4314) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4314) 

s_mov_b32 s48, s28 // (op 4314) reg expression
s_mov_b32 s49, s29 // (op 4314) reg expression
s_mov_b32 s50, s0 // (op 4314) reg expression
s_mov_b32 s51, 131072 // (op 4314) call()

s_mul_hi_i32 s47, s16, s10 // (op 4331) 
v_lshrrev_b32 v15, 1, v1 // (op 4331) 
s_add_i32 s0, s47, s16 // (op 4331) 

v_and_b32 v16, -2, v1 // (op 4331) 
s_ashr_i32 s1, s0, 31 // (op 4331) 
v_sub_u32 v17, v1, v16 // (op 4331) 

s_cmp_eq_i32 s10, 0 // (op 4331) 
s_mov_b32 s47, scc // (op 4331) 
v_lshlrev_b32 v16, 6, v17 // (op 4331) 

s_cmp_eq_i32 s47, 0 // (op 4331) 
s_cselect_b32 s18, 0, -1 // (op 4331) 

v_mov_b32 v19, 0 // (op 4331) convert
v_mov_b32 v18, v16 // (op 4331) convert

s_and_b32 s19, s13, 31 // (op 4331) 
s_lshl_b32 s47, 1, s19 // (op 4331) 
s_add_i32 s22, s47, s18 // (op 4331) 


s_and_b32 s18, s1, s22 // (op 4331) 


s_add_i32 s1, s0, s18 // (op 4331) 


s_ashr_i32 s0, s1, s19 // (op 4331) 


s_lshl_b32 s1, s13, 24 // (op 4331) 
s_ashr_i32 s47, s1, 31 // (op 4331) 

s_xor_b32 s1, s0, s47 // (op 4331) 

s_sub_i32 s0, s1, s47 // (op 4331) 


s_mul_hi_i32 s1, s0, s14 // (op 4331) 
s_add_i32 s47, s1, s0 // (op 4331) 

s_ashr_i32 s1, s47, 31 // (op 4331) 
s_cmp_eq_i32 s14, 0 // (op 4331) 
s_mov_b32 s18, scc // (op 4331) 
s_cmp_eq_i32 s18, 0 // (op 4331) 
s_cselect_b32 s19, 0, -1 // (op 4331) 

s_and_b32 s18, s15, 31 // (op 4331) 
s_lshl_b32 s22, 1, s18 // (op 4331) 
s_add_i32 s23, s22, s19 // (op 4331) 


s_and_b32 s19, s1, s23 // (op 4331) 


s_add_i32 s1, s47, s19 // (op 4331) 


s_ashr_i32 s19, s1, s18 // (op 4331) 


s_lshl_b32 s1, s15, 24 // (op 4331) 
s_ashr_i32 s18, s1, 31 // (op 4331) 

s_xor_b32 s1, s19, s18 // (op 4331) 

s_sub_i32 s19, s1, s18 // (op 4331) 


s_mul_i32 s1, s0, s11 // (op 4331) 

s_sub_u32 s0, s16, s1 // (op 4331) 

s_mul_i32 s1, s19, s11 // (op 4331) 
s_add_u32 s1, s1, s0 // (op 4331) 


s_mul_hi_i32 s0, s16, s9 // (op 4331) 
s_add_i32 s47, s0, s16 // (op 4331) 

s_ashr_i32 s0, s47, 31 // (op 4331) 
s_cmp_eq_i32 s9, 0 // (op 4331) 
s_mov_b32 s18, scc // (op 4331) 
s_cmp_eq_i32 s18, 0 // (op 4331) 
s_cselect_b32 s19, 0, -1 // (op 4331) 

s_and_b32 s18, s45, 31 // (op 4331) 
s_lshl_b32 s22, 1, s18 // (op 4331) 
s_add_i32 s23, s22, s19 // (op 4331) 


s_and_b32 s19, s0, s23 // (op 4331) 


s_add_i32 s0, s47, s19 // (op 4331) 


s_ashr_i32 s19, s0, s18 // (op 4331) 


s_lshl_b32 s0, s45, 24 // (op 4331) 
s_ashr_i32 s18, s0, 31 // (op 4331) 

s_xor_b32 s0, s19, s18 // (op 4331) 

s_sub_i32 s19, s0, s18 // (op 4331) 


s_add_u32 s22, s6, 64 // (op 4331) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4331) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4331) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4331) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4331) 

s_mul_i32 s0, s11, s22 // (op 4331) 

s_mul_i32 s18, s8, s0 // (op 4331) 

s_mul_i32 s0, s19, s18 // (op 4331) 

s_sub_u32 s18, s16, s0 // (op 4331) 

s_mov_b32 s0, s18 // (op 4331) 
s_mul_hi_i32 s47, s0, s12 // (op 4331) 
s_add_i32 s22, s47, s0 // (op 4331) 


s_ashr_i32 s0, s22, 31 // (op 4331) 
s_cmp_eq_i32 s12, 0 // (op 4331) 
s_mov_b32 s47, scc // (op 4331) 
s_cmp_eq_i32 s47, 0 // (op 4331) 
s_cselect_b32 s23, 0, -1 // (op 4331) 

s_and_b32 s47, s44, 31 // (op 4331) 
s_lshl_b32 s24, 1, s47 // (op 4331) 
s_add_i32 s25, s24, s23 // (op 4331) 


s_and_b32 s23, s0, s25 // (op 4331) 


s_add_i32 s0, s22, s23 // (op 4331) 


s_ashr_i32 s22, s0, s47 // (op 4331) 


s_lshl_b32 s0, s44, 24 // (op 4331) 
s_ashr_i32 s47, s0, 31 // (op 4331) 

s_xor_b32 s0, s22, s47 // (op 4331) 

s_sub_i32 s22, s0, s47 // (op 4331) 


s_mul_i32 s0, s46, s11 // (op 4331) 
s_add_u32 s24, s20, 128 // (op 4331) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4331) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 4331) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 4331) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 7 // (op 4331) 

s_sub_i32 s23, s24, s0 // (op 4331) 

s_mul_i32 s47, s22, s23 // (op 4331) 


s_sub_u32 s22, s18, s47 // (op 4331) 


s_add_u32 s18, s22, s0 // (op 4331) 


s_cmp_eq_i32 s19, 0 // (op 4331) 
s_mov_b32 s0, scc // (op 4331) 

s_cmp_eq_i32 s0, 0 // (op 4331) 
s_cselect_b32 s19, s18, s1 // (op 4331) 



v_lshl_add_u32 v16, s19, 7, v15 // (op 4331) 


v_mul_lo_u32 v21, v16, s27 // (op 4331) most significant: low of low * high
v_mul_hi_u32 v15, v16, s26 // (op 4331) most significant: high of low * low
v_mul_lo_u32 v20, v16, s26 // (op 4331) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 4331) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 4331) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 4331) most significant half



v_mul_lo_u32 v19, v21, 6 // (op 4331) most significant: low of high * low
v_mul_hi_u32 v15, v20, 6 // (op 4331) most significant: high of low * low
v_mul_lo_u32 v18, v20, 6 // (op 4331) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 4331) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 4331) 

v_mov_b32 v15, v16 // (op 4331) convert

s_mul_i32 s1, s21, 6 // (op 4333) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 4333) least significant
s_mul_hi_u32 s47, s20, 6 // (op 4333) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4333) most significant: sum

s_mul_i32 s19, s26, s1 // (op 4333) most significant: low of low * high
s_mul_i32 s47, s27, s0 // (op 4333) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 4333) least significant
s_mul_hi_u32 s22, s26, s0 // (op 4333) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 4333) most significant: sum
s_add_u32 s19, s19, s22 // (op 4333) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4333) 

s_mov_b32 s48, s28 // (op 4333) reg expression
s_mov_b32 s49, s29 // (op 4333) reg expression
s_mov_b32 s50, s0 // (op 4333) reg expression
s_mov_b32 s51, 131072 // (op 4333) call()

s_mul_hi_i32 s47, s16, s10 // (op 4376) 
v_lshrrev_b32 v16, 2, v1 // (op 4376) 
s_add_i32 s0, s47, s16 // (op 4376) 

v_and_b32 v17, -4, v1 // (op 4376) 
s_ashr_i32 s1, s0, 31 // (op 4376) 
v_sub_u32 v18, v1, v17 // (op 4376) 

s_cmp_eq_i32 s10, 0 // (op 4376) 
s_mov_b32 s47, scc // (op 4376) 
v_lshlrev_b32 v17, 5, v18 // (op 4376) 

s_cmp_eq_i32 s47, 0 // (op 4376) 
s_cselect_b32 s18, 0, -1 // (op 4376) 

v_mov_b32 v19, 0 // (op 4376) convert
v_mov_b32 v18, v17 // (op 4376) convert

s_and_b32 s19, s13, 31 // (op 4376) 
s_lshl_b32 s47, 1, s19 // (op 4376) 
s_add_i32 s22, s47, s18 // (op 4376) 


s_and_b32 s18, s1, s22 // (op 4376) 


s_add_i32 s1, s0, s18 // (op 4376) 


s_ashr_i32 s0, s1, s19 // (op 4376) 


s_lshl_b32 s1, s13, 24 // (op 4376) 
s_ashr_i32 s47, s1, 31 // (op 4376) 

s_xor_b32 s1, s0, s47 // (op 4376) 

s_sub_i32 s0, s1, s47 // (op 4376) 


s_mul_hi_i32 s1, s0, s14 // (op 4376) 
s_add_i32 s47, s1, s0 // (op 4376) 

s_ashr_i32 s1, s47, 31 // (op 4376) 
s_cmp_eq_i32 s14, 0 // (op 4376) 
s_mov_b32 s18, scc // (op 4376) 
s_cmp_eq_i32 s18, 0 // (op 4376) 
s_cselect_b32 s19, 0, -1 // (op 4376) 

s_and_b32 s18, s15, 31 // (op 4376) 
s_lshl_b32 s22, 1, s18 // (op 4376) 
s_add_i32 s23, s22, s19 // (op 4376) 


s_and_b32 s19, s1, s23 // (op 4376) 


s_add_i32 s1, s47, s19 // (op 4376) 


s_ashr_i32 s19, s1, s18 // (op 4376) 


s_lshl_b32 s1, s15, 24 // (op 4376) 
s_ashr_i32 s18, s1, 31 // (op 4376) 

s_xor_b32 s1, s19, s18 // (op 4376) 

s_sub_i32 s19, s1, s18 // (op 4376) 


s_add_u32 s22, s6, 64 // (op 4376) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4376) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4376) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4376) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4376) 

s_mul_i32 s1, s19, s22 // (op 4376) 

s_sub_i32 s47, s0, s1 // (op 4376) 


s_mul_hi_i32 s0, s16, s9 // (op 4376) 
s_add_i32 s1, s0, s16 // (op 4376) 

s_ashr_i32 s0, s1, 31 // (op 4376) 
s_cmp_eq_i32 s9, 0 // (op 4376) 
s_mov_b32 s18, scc // (op 4376) 
s_cmp_eq_i32 s18, 0 // (op 4376) 
s_cselect_b32 s19, 0, -1 // (op 4376) 

s_and_b32 s18, s45, 31 // (op 4376) 
s_lshl_b32 s24, 1, s18 // (op 4376) 
s_add_i32 s25, s24, s19 // (op 4376) 


s_and_b32 s19, s0, s25 // (op 4376) 


s_add_i32 s0, s1, s19 // (op 4376) 


s_ashr_i32 s1, s0, s18 // (op 4376) 


s_lshl_b32 s0, s45, 24 // (op 4376) 
s_ashr_i32 s18, s0, 31 // (op 4376) 

s_xor_b32 s0, s1, s18 // (op 4376) 

s_sub_i32 s1, s0, s18 // (op 4376) 


s_mul_i32 s0, s11, s22 // (op 4376) 

s_mul_i32 s18, s8, s0 // (op 4376) 

s_mul_i32 s0, s1, s18 // (op 4376) 

s_sub_u32 s18, s16, s0 // (op 4376) 

s_mul_hi_i32 s0, s18, s12 // (op 4376) 
s_add_i32 s19, s0, s18 // (op 4376) 


s_ashr_i32 s0, s19, 31 // (op 4376) 
s_cmp_eq_i32 s12, 0 // (op 4376) 
s_mov_b32 s18, scc // (op 4376) 
s_cmp_eq_i32 s18, 0 // (op 4376) 
s_cselect_b32 s22, 0, -1 // (op 4376) 

s_and_b32 s18, s44, 31 // (op 4376) 
s_lshl_b32 s23, 1, s18 // (op 4376) 
s_add_i32 s24, s23, s22 // (op 4376) 


s_and_b32 s25, s0, s24 // (op 4376) 


s_add_i32 s0, s19, s25 // (op 4376) 


s_ashr_i32 s19, s0, s18 // (op 4376) 


s_lshl_b32 s0, s44, 24 // (op 4376) 
s_ashr_i32 s18, s0, 31 // (op 4376) 

s_xor_b32 s0, s19, s18 // (op 4376) 

s_sub_i32 s19, s0, s18 // (op 4376) 


s_cmp_eq_i32 s1, 0 // (op 4376) 
s_mov_b32 s0, scc // (op 4376) 

s_cmp_eq_i32 s0, 0 // (op 4376) 
s_cselect_b32 s1, s19, s47 // (op 4376) 



v_lshl_add_u32 v17, s1, 6, v16 // (op 4376) 


v_mul_lo_u32 v21, v17, s31 // (op 4376) most significant: low of low * high
v_mul_hi_u32 v16, v17, s30 // (op 4376) most significant: high of low * low
v_mul_lo_u32 v20, v17, s30 // (op 4376) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 4376) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 4376) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 4376) most significant half



v_mul_lo_u32 v19, v17, 6 // (op 4376) most significant: low of high * low
v_mul_hi_u32 v20, v16, 6 // (op 4376) most significant: high of low * low
v_mul_lo_u32 v18, v16, 6 // (op 4376) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 4376) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 4376) 

v_mov_b32 v18, v16 // (op 4376) convert

s_mul_i32 s1, s7, 6 // (op 4378) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 4378) least significant
s_mul_hi_u32 s47, s6, 6 // (op 4378) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4378) most significant: sum

s_mul_i32 s19, s30, s1 // (op 4378) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 4378) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 4378) least significant
s_mul_hi_u32 s22, s30, s0 // (op 4378) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 4378) most significant: sum
s_add_u32 s19, s19, s22 // (op 4378) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4378) 

s_mov_b32 s60, s32 // (op 4378) reg expression
s_mov_b32 s61, s33 // (op 4378) reg expression
s_mov_b32 s62, s0 // (op 4378) reg expression
s_mov_b32 s63, 131072 // (op 4378) call()

s_mul_hi_i32 s47, s16, s10 // (op 4395) 
v_lshrrev_b32 v16, 2, v1 // (op 4395) 
s_add_i32 s0, s47, s16 // (op 4395) 

v_and_b32 v17, -4, v1 // (op 4395) 
s_ashr_i32 s1, s0, 31 // (op 4395) 
v_sub_u32 v19, v1, v17 // (op 4395) 

s_cmp_eq_i32 s10, 0 // (op 4395) 
s_mov_b32 s47, scc // (op 4395) 
v_lshlrev_b32 v17, 5, v19 // (op 4395) 

s_cmp_eq_i32 s47, 0 // (op 4395) 
s_cselect_b32 s18, 0, -1 // (op 4395) 

v_mov_b32 v21, 0 // (op 4395) convert
v_mov_b32 v20, v17 // (op 4395) convert

s_and_b32 s19, s13, 31 // (op 4395) 
s_lshl_b32 s47, 1, s19 // (op 4395) 
s_add_i32 s22, s47, s18 // (op 4395) 


s_and_b32 s18, s1, s22 // (op 4395) 


s_add_i32 s1, s0, s18 // (op 4395) 


s_ashr_i32 s0, s1, s19 // (op 4395) 


s_lshl_b32 s1, s13, 24 // (op 4395) 
s_ashr_i32 s47, s1, 31 // (op 4395) 

s_xor_b32 s1, s0, s47 // (op 4395) 

s_sub_i32 s0, s1, s47 // (op 4395) 


s_mul_hi_i32 s1, s0, s14 // (op 4395) 
s_add_i32 s47, s1, s0 // (op 4395) 

s_ashr_i32 s1, s47, 31 // (op 4395) 
s_cmp_eq_i32 s14, 0 // (op 4395) 
s_mov_b32 s18, scc // (op 4395) 
s_cmp_eq_i32 s18, 0 // (op 4395) 
s_cselect_b32 s19, 0, -1 // (op 4395) 

s_and_b32 s18, s15, 31 // (op 4395) 
s_lshl_b32 s22, 1, s18 // (op 4395) 
s_add_i32 s23, s22, s19 // (op 4395) 


s_and_b32 s19, s1, s23 // (op 4395) 


s_add_i32 s1, s47, s19 // (op 4395) 


s_ashr_i32 s19, s1, s18 // (op 4395) 


s_lshl_b32 s1, s15, 24 // (op 4395) 
s_ashr_i32 s18, s1, 31 // (op 4395) 

s_xor_b32 s1, s19, s18 // (op 4395) 

s_sub_i32 s19, s1, s18 // (op 4395) 


s_add_u32 s22, s6, 64 // (op 4395) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4395) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4395) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4395) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4395) 

s_mul_i32 s1, s19, s22 // (op 4395) 

s_sub_i32 s47, s0, s1 // (op 4395) 


s_mul_hi_i32 s0, s16, s9 // (op 4395) 
s_add_i32 s1, s0, s16 // (op 4395) 

s_ashr_i32 s0, s1, 31 // (op 4395) 
s_cmp_eq_i32 s9, 0 // (op 4395) 
s_mov_b32 s18, scc // (op 4395) 
s_cmp_eq_i32 s18, 0 // (op 4395) 
s_cselect_b32 s19, 0, -1 // (op 4395) 

s_and_b32 s18, s45, 31 // (op 4395) 
s_lshl_b32 s24, 1, s18 // (op 4395) 
s_add_i32 s25, s24, s19 // (op 4395) 


s_and_b32 s19, s0, s25 // (op 4395) 


s_add_i32 s0, s1, s19 // (op 4395) 


s_ashr_i32 s1, s0, s18 // (op 4395) 


s_lshl_b32 s0, s45, 24 // (op 4395) 
s_ashr_i32 s18, s0, 31 // (op 4395) 

s_xor_b32 s0, s1, s18 // (op 4395) 

s_sub_i32 s1, s0, s18 // (op 4395) 


s_mul_i32 s0, s11, s22 // (op 4395) 

s_mul_i32 s18, s8, s0 // (op 4395) 

s_mul_i32 s0, s1, s18 // (op 4395) 

s_sub_u32 s18, s16, s0 // (op 4395) 

s_mul_hi_i32 s0, s18, s12 // (op 4395) 
s_add_i32 s19, s0, s18 // (op 4395) 


s_ashr_i32 s0, s19, 31 // (op 4395) 
s_cmp_eq_i32 s12, 0 // (op 4395) 
s_mov_b32 s18, scc // (op 4395) 
s_cmp_eq_i32 s18, 0 // (op 4395) 
s_cselect_b32 s22, 0, -1 // (op 4395) 

s_and_b32 s18, s44, 31 // (op 4395) 
s_lshl_b32 s23, 1, s18 // (op 4395) 
s_add_i32 s24, s23, s22 // (op 4395) 


s_and_b32 s25, s0, s24 // (op 4395) 


s_add_i32 s0, s19, s25 // (op 4395) 


s_ashr_i32 s19, s0, s18 // (op 4395) 


s_lshl_b32 s0, s44, 24 // (op 4395) 
s_ashr_i32 s18, s0, 31 // (op 4395) 

s_xor_b32 s0, s19, s18 // (op 4395) 

s_sub_i32 s19, s0, s18 // (op 4395) 


s_cmp_eq_i32 s1, 0 // (op 4395) 
s_mov_b32 s0, scc // (op 4395) 

s_cmp_eq_i32 s0, 0 // (op 4395) 
s_cselect_b32 s1, s19, s47 // (op 4395) 



v_lshl_add_u32 v17, s1, 6, v16 // (op 4395) 


v_mul_lo_u32 v23, v17, s31 // (op 4395) most significant: low of low * high
v_mul_hi_u32 v16, v17, s30 // (op 4395) most significant: high of low * low
v_mul_lo_u32 v22, v17, s30 // (op 4395) least significant: low of low * low
v_add_u32 v23, v23, v16 // (op 4395) most significant: sum


v_add_co_u32 v16, s[0:1], v20, v22 // (op 4395) least significant half
v_addc_co_u32 v17, s[0:1], v21, v23, s[0:1] // (op 4395) most significant half



v_mul_lo_u32 v21, v17, 6 // (op 4395) most significant: low of high * low
v_mul_hi_u32 v19, v16, 6 // (op 4395) most significant: high of low * low
v_mul_lo_u32 v20, v16, 6 // (op 4395) least significant: low of low * low
v_add_u32 v21, v21, v19 // (op 4395) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[20:21] // (op 4395) 

v_mov_b32 v19, v16 // (op 4395) convert

s_mul_i32 s1, s7, 6 // (op 4397) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 4397) least significant
s_mul_hi_u32 s47, s6, 6 // (op 4397) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4397) most significant: sum

s_mul_i32 s19, s30, s1 // (op 4397) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 4397) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 4397) least significant
s_mul_hi_u32 s22, s30, s0 // (op 4397) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 4397) most significant: sum
s_add_u32 s19, s19, s22 // (op 4397) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4397) 

s_mov_b32 s60, s32 // (op 4397) reg expression
s_mov_b32 s61, s33 // (op 4397) reg expression
s_mov_b32 s62, s0 // (op 4397) reg expression
s_mov_b32 s63, 131072 // (op 4397) call()

s_mul_hi_i32 s47, s16, s10 // (op 4440) 
v_lshrrev_b32 v16, 2, v1 // (op 4440) 
s_add_i32 s0, s47, s16 // (op 4440) 

v_lshlrev_b32 v17, 1, v16 // (op 4440) 

s_ashr_i32 s1, s0, 31 // (op 4440) 
v_and_b32 v16, -4, v1 // (op 4440) 
s_cmp_eq_i32 s10, 0 // (op 4440) 
s_mov_b32 s47, scc // (op 4440) 
v_sub_u32 v20, v1, v16 // (op 4440) 

s_cmp_eq_i32 s47, 0 // (op 4440) 
s_cselect_b32 s18, 0, -1 // (op 4440) 

s_and_b32 s19, s13, 31 // (op 4440) 
s_lshl_b32 s47, 1, s19 // (op 4440) 
s_add_i32 s22, s47, s18 // (op 4440) 


s_and_b32 s18, s1, s22 // (op 4440) 


s_add_i32 s1, s0, s18 // (op 4440) 


s_ashr_i32 s0, s1, s19 // (op 4440) 


s_lshl_b32 s1, s13, 24 // (op 4440) 
s_ashr_i32 s47, s1, 31 // (op 4440) 

s_xor_b32 s1, s0, s47 // (op 4440) 

s_sub_i32 s0, s1, s47 // (op 4440) 


s_mul_hi_i32 s1, s0, s14 // (op 4440) 
s_add_i32 s47, s1, s0 // (op 4440) 

s_ashr_i32 s1, s47, 31 // (op 4440) 
s_cmp_eq_i32 s14, 0 // (op 4440) 
s_mov_b32 s18, scc // (op 4440) 
s_cmp_eq_i32 s18, 0 // (op 4440) 
s_cselect_b32 s19, 0, -1 // (op 4440) 

s_and_b32 s18, s15, 31 // (op 4440) 
s_lshl_b32 s22, 1, s18 // (op 4440) 
s_add_i32 s23, s22, s19 // (op 4440) 


s_and_b32 s19, s1, s23 // (op 4440) 


s_add_i32 s1, s47, s19 // (op 4440) 


s_ashr_i32 s19, s1, s18 // (op 4440) 


s_lshl_b32 s1, s15, 24 // (op 4440) 
s_ashr_i32 s18, s1, 31 // (op 4440) 

s_xor_b32 s1, s19, s18 // (op 4440) 

s_sub_i32 s19, s1, s18 // (op 4440) 


s_mul_i32 s1, s0, s11 // (op 4440) 

s_sub_u32 s0, s16, s1 // (op 4440) 

s_mul_i32 s1, s19, s11 // (op 4440) 
s_add_u32 s1, s1, s0 // (op 4440) 


s_mul_hi_i32 s0, s16, s9 // (op 4440) 
s_add_i32 s47, s0, s16 // (op 4440) 

s_ashr_i32 s0, s47, 31 // (op 4440) 
s_cmp_eq_i32 s9, 0 // (op 4440) 
s_mov_b32 s18, scc // (op 4440) 
s_cmp_eq_i32 s18, 0 // (op 4440) 
s_cselect_b32 s19, 0, -1 // (op 4440) 

s_and_b32 s18, s45, 31 // (op 4440) 
s_lshl_b32 s22, 1, s18 // (op 4440) 
s_add_i32 s23, s22, s19 // (op 4440) 


s_and_b32 s19, s0, s23 // (op 4440) 


s_add_i32 s0, s47, s19 // (op 4440) 


s_ashr_i32 s19, s0, s18 // (op 4440) 


s_lshl_b32 s0, s45, 24 // (op 4440) 
s_ashr_i32 s18, s0, 31 // (op 4440) 

s_xor_b32 s0, s19, s18 // (op 4440) 

s_sub_i32 s19, s0, s18 // (op 4440) 


s_add_u32 s22, s6, 64 // (op 4440) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4440) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4440) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4440) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4440) 

s_mul_i32 s0, s11, s22 // (op 4440) 

s_mul_i32 s18, s8, s0 // (op 4440) 

s_mul_i32 s0, s19, s18 // (op 4440) 

s_sub_u32 s18, s16, s0 // (op 4440) 

s_mov_b32 s0, s18 // (op 4440) 
s_mul_hi_i32 s47, s0, s12 // (op 4440) 
s_add_i32 s22, s47, s0 // (op 4440) 


s_ashr_i32 s0, s22, 31 // (op 4440) 
s_cmp_eq_i32 s12, 0 // (op 4440) 
s_mov_b32 s47, scc // (op 4440) 
s_cmp_eq_i32 s47, 0 // (op 4440) 
s_cselect_b32 s23, 0, -1 // (op 4440) 

s_and_b32 s47, s44, 31 // (op 4440) 
s_lshl_b32 s24, 1, s47 // (op 4440) 
s_add_i32 s25, s24, s23 // (op 4440) 


s_and_b32 s23, s0, s25 // (op 4440) 


s_add_i32 s0, s22, s23 // (op 4440) 


s_ashr_i32 s22, s0, s47 // (op 4440) 


s_lshl_b32 s0, s44, 24 // (op 4440) 
s_ashr_i32 s47, s0, 31 // (op 4440) 

s_xor_b32 s0, s22, s47 // (op 4440) 

s_sub_i32 s22, s0, s47 // (op 4440) 


s_mul_i32 s0, s46, s11 // (op 4440) 
s_add_u32 s24, s20, 128 // (op 4440) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4440) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 4440) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 4440) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 4440) 

s_sub_i32 s23, s24, s0 // (op 4440) 

s_mul_i32 s47, s22, s23 // (op 4440) 


s_sub_u32 s22, s18, s47 // (op 4440) 


s_add_u32 s18, s22, s0 // (op 4440) 


s_cmp_eq_i32 s19, 0 // (op 4440) 
s_mov_b32 s0, scc // (op 4440) 

s_cmp_eq_i32 s0, 0 // (op 4440) 
s_cselect_b32 s19, s18, s1 // (op 4440) 



v_lshl_add_u32 v16, s19, 7, v17 // (op 4440) 


v_mul_lo_u32 v17, v16, s34 // (op 4440) 
v_add_u32 v17, v17, v20 // (op 4440) 


s_mul_i32 s1, s34, s21 // (op 4442) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 4442) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 4442) least significant
s_mul_hi_u32 s18, s34, s20 // (op 4442) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4442) most significant: sum
s_add_u32 s1, s1, s18 // (op 4442) most significant: sum


s_mov_b32 s64, s36 // (op 4442) reg expression
s_mov_b32 s65, s37 // (op 4442) reg expression
s_mov_b32 s66, s0 // (op 4442) reg expression
s_mov_b32 s67, 131072 // (op 4442) call()

s_mul_hi_i32 s47, s16, s10 // (op 4459) 
v_lshrrev_b32 v16, 2, v1 // (op 4459) 
s_add_i32 s0, s47, s16 // (op 4459) 

v_lshlrev_b32 v20, 1, v16 // (op 4459) 

s_ashr_i32 s1, s0, 31 // (op 4459) 
v_and_b32 v16, -4, v1 // (op 4459) 
s_cmp_eq_i32 s10, 0 // (op 4459) 
s_mov_b32 s47, scc // (op 4459) 
v_sub_u32 v21, v1, v16 // (op 4459) 

s_cmp_eq_i32 s47, 0 // (op 4459) 
s_cselect_b32 s18, 0, -1 // (op 4459) 

s_and_b32 s19, s13, 31 // (op 4459) 
s_lshl_b32 s47, 1, s19 // (op 4459) 
s_add_i32 s22, s47, s18 // (op 4459) 


s_and_b32 s18, s1, s22 // (op 4459) 


s_add_i32 s1, s0, s18 // (op 4459) 


s_ashr_i32 s0, s1, s19 // (op 4459) 


s_lshl_b32 s1, s13, 24 // (op 4459) 
s_ashr_i32 s47, s1, 31 // (op 4459) 

s_xor_b32 s1, s0, s47 // (op 4459) 

s_sub_i32 s0, s1, s47 // (op 4459) 


s_mul_hi_i32 s1, s0, s14 // (op 4459) 
s_add_i32 s47, s1, s0 // (op 4459) 

s_ashr_i32 s1, s47, 31 // (op 4459) 
s_cmp_eq_i32 s14, 0 // (op 4459) 
s_mov_b32 s18, scc // (op 4459) 
s_cmp_eq_i32 s18, 0 // (op 4459) 
s_cselect_b32 s19, 0, -1 // (op 4459) 

s_and_b32 s18, s15, 31 // (op 4459) 
s_lshl_b32 s22, 1, s18 // (op 4459) 
s_add_i32 s23, s22, s19 // (op 4459) 


s_and_b32 s19, s1, s23 // (op 4459) 


s_add_i32 s1, s47, s19 // (op 4459) 


s_ashr_i32 s19, s1, s18 // (op 4459) 


s_lshl_b32 s1, s15, 24 // (op 4459) 
s_ashr_i32 s18, s1, 31 // (op 4459) 

s_xor_b32 s1, s19, s18 // (op 4459) 

s_sub_i32 s19, s1, s18 // (op 4459) 


s_mul_i32 s1, s0, s11 // (op 4459) 

s_sub_u32 s0, s16, s1 // (op 4459) 

s_mul_i32 s1, s19, s11 // (op 4459) 
s_add_u32 s1, s1, s0 // (op 4459) 


s_mul_hi_i32 s0, s16, s9 // (op 4459) 
s_add_i32 s47, s0, s16 // (op 4459) 

s_ashr_i32 s0, s47, 31 // (op 4459) 
s_cmp_eq_i32 s9, 0 // (op 4459) 
s_mov_b32 s18, scc // (op 4459) 
s_cmp_eq_i32 s18, 0 // (op 4459) 
s_cselect_b32 s19, 0, -1 // (op 4459) 

s_and_b32 s18, s45, 31 // (op 4459) 
s_lshl_b32 s22, 1, s18 // (op 4459) 
s_add_i32 s23, s22, s19 // (op 4459) 


s_and_b32 s19, s0, s23 // (op 4459) 


s_add_i32 s0, s47, s19 // (op 4459) 


s_ashr_i32 s19, s0, s18 // (op 4459) 


s_lshl_b32 s0, s45, 24 // (op 4459) 
s_ashr_i32 s18, s0, 31 // (op 4459) 

s_xor_b32 s0, s19, s18 // (op 4459) 

s_sub_i32 s19, s0, s18 // (op 4459) 


s_add_u32 s22, s6, 64 // (op 4459) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4459) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4459) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4459) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4459) 

s_mul_i32 s0, s11, s22 // (op 4459) 

s_mul_i32 s18, s8, s0 // (op 4459) 

s_mul_i32 s0, s19, s18 // (op 4459) 

s_sub_u32 s18, s16, s0 // (op 4459) 

s_mov_b32 s0, s18 // (op 4459) 
s_mul_hi_i32 s47, s0, s12 // (op 4459) 
s_add_i32 s22, s47, s0 // (op 4459) 


s_ashr_i32 s0, s22, 31 // (op 4459) 
s_cmp_eq_i32 s12, 0 // (op 4459) 
s_mov_b32 s47, scc // (op 4459) 
s_cmp_eq_i32 s47, 0 // (op 4459) 
s_cselect_b32 s23, 0, -1 // (op 4459) 

s_and_b32 s47, s44, 31 // (op 4459) 
s_lshl_b32 s24, 1, s47 // (op 4459) 
s_add_i32 s25, s24, s23 // (op 4459) 


s_and_b32 s23, s0, s25 // (op 4459) 


s_add_i32 s0, s22, s23 // (op 4459) 


s_ashr_i32 s22, s0, s47 // (op 4459) 


s_lshl_b32 s0, s44, 24 // (op 4459) 
s_ashr_i32 s47, s0, 31 // (op 4459) 

s_xor_b32 s0, s22, s47 // (op 4459) 

s_sub_i32 s22, s0, s47 // (op 4459) 


s_mul_i32 s0, s46, s11 // (op 4459) 
s_add_u32 s24, s20, 128 // (op 4459) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4459) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 4459) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 4459) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 7 // (op 4459) 

s_sub_i32 s23, s24, s0 // (op 4459) 

s_mul_i32 s47, s22, s23 // (op 4459) 


s_sub_u32 s22, s18, s47 // (op 4459) 


s_add_u32 s18, s22, s0 // (op 4459) 


s_cmp_eq_i32 s19, 0 // (op 4459) 
s_mov_b32 s0, scc // (op 4459) 

s_cmp_eq_i32 s0, 0 // (op 4459) 
s_cselect_b32 s19, s18, s1 // (op 4459) 



v_lshl_add_u32 v16, s19, 7, v20 // (op 4459) 


v_mul_lo_u32 v20, v16, s34 // (op 4459) 
v_add_u32 v20, v20, v21 // (op 4459) 


s_mul_i32 s1, s34, s21 // (op 4461) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 4461) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 4461) least significant
s_mul_hi_u32 s18, s34, s20 // (op 4461) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4461) most significant: sum
s_add_u32 s1, s1, s18 // (op 4461) most significant: sum


s_mov_b32 s64, s36 // (op 4461) reg expression
s_mov_b32 s65, s37 // (op 4461) reg expression
s_mov_b32 s66, s0 // (op 4461) reg expression
s_mov_b32 s67, 131072 // (op 4461) call()

s_mul_hi_i32 s47, s16, s10 // (op 4504) 
v_lshrrev_b32 v16, 2, v1 // (op 4504) 
s_add_i32 s0, s47, s16 // (op 4504) 

v_and_b32 v21, -4, v1 // (op 4504) 
s_ashr_i32 s1, s0, 31 // (op 4504) 
v_sub_u32 v22, v1, v21 // (op 4504) 

s_cmp_eq_i32 s10, 0 // (op 4504) 
s_mov_b32 s47, scc // (op 4504) 
s_cmp_eq_i32 s47, 0 // (op 4504) 
s_cselect_b32 s18, 0, -1 // (op 4504) 

s_and_b32 s19, s13, 31 // (op 4504) 
s_lshl_b32 s47, 1, s19 // (op 4504) 
s_add_i32 s22, s47, s18 // (op 4504) 


s_and_b32 s18, s1, s22 // (op 4504) 


s_add_i32 s1, s0, s18 // (op 4504) 


s_ashr_i32 s0, s1, s19 // (op 4504) 


s_lshl_b32 s1, s13, 24 // (op 4504) 
s_ashr_i32 s47, s1, 31 // (op 4504) 

s_xor_b32 s1, s0, s47 // (op 4504) 

s_sub_i32 s0, s1, s47 // (op 4504) 


s_mul_hi_i32 s1, s0, s14 // (op 4504) 
s_add_i32 s47, s1, s0 // (op 4504) 

s_ashr_i32 s1, s47, 31 // (op 4504) 
s_cmp_eq_i32 s14, 0 // (op 4504) 
s_mov_b32 s18, scc // (op 4504) 
s_cmp_eq_i32 s18, 0 // (op 4504) 
s_cselect_b32 s19, 0, -1 // (op 4504) 

s_and_b32 s18, s15, 31 // (op 4504) 
s_lshl_b32 s22, 1, s18 // (op 4504) 
s_add_i32 s23, s22, s19 // (op 4504) 


s_and_b32 s19, s1, s23 // (op 4504) 


s_add_i32 s1, s47, s19 // (op 4504) 


s_ashr_i32 s19, s1, s18 // (op 4504) 


s_lshl_b32 s1, s15, 24 // (op 4504) 
s_ashr_i32 s18, s1, 31 // (op 4504) 

s_xor_b32 s1, s19, s18 // (op 4504) 

s_sub_i32 s19, s1, s18 // (op 4504) 


s_add_u32 s22, s6, 64 // (op 4504) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4504) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4504) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4504) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4504) 

s_mul_i32 s1, s19, s22 // (op 4504) 

s_sub_i32 s47, s0, s1 // (op 4504) 


s_mul_hi_i32 s0, s16, s9 // (op 4504) 
s_add_i32 s1, s0, s16 // (op 4504) 

s_ashr_i32 s0, s1, 31 // (op 4504) 
s_cmp_eq_i32 s9, 0 // (op 4504) 
s_mov_b32 s18, scc // (op 4504) 
s_cmp_eq_i32 s18, 0 // (op 4504) 
s_cselect_b32 s19, 0, -1 // (op 4504) 

s_and_b32 s18, s45, 31 // (op 4504) 
s_lshl_b32 s24, 1, s18 // (op 4504) 
s_add_i32 s25, s24, s19 // (op 4504) 


s_and_b32 s19, s0, s25 // (op 4504) 


s_add_i32 s0, s1, s19 // (op 4504) 


s_ashr_i32 s1, s0, s18 // (op 4504) 


s_lshl_b32 s0, s45, 24 // (op 4504) 
s_ashr_i32 s18, s0, 31 // (op 4504) 

s_xor_b32 s0, s1, s18 // (op 4504) 

s_sub_i32 s1, s0, s18 // (op 4504) 


s_mul_i32 s0, s11, s22 // (op 4504) 

s_mul_i32 s18, s8, s0 // (op 4504) 

s_mul_i32 s0, s1, s18 // (op 4504) 

s_sub_u32 s18, s16, s0 // (op 4504) 

s_mul_hi_i32 s0, s18, s12 // (op 4504) 
s_add_i32 s19, s0, s18 // (op 4504) 


s_ashr_i32 s0, s19, 31 // (op 4504) 
s_cmp_eq_i32 s12, 0 // (op 4504) 
s_mov_b32 s18, scc // (op 4504) 
s_cmp_eq_i32 s18, 0 // (op 4504) 
s_cselect_b32 s22, 0, -1 // (op 4504) 

s_and_b32 s18, s44, 31 // (op 4504) 
s_lshl_b32 s23, 1, s18 // (op 4504) 
s_add_i32 s24, s23, s22 // (op 4504) 


s_and_b32 s25, s0, s24 // (op 4504) 


s_add_i32 s0, s19, s25 // (op 4504) 


s_ashr_i32 s19, s0, s18 // (op 4504) 


s_lshl_b32 s0, s44, 24 // (op 4504) 
s_ashr_i32 s18, s0, 31 // (op 4504) 

s_xor_b32 s0, s19, s18 // (op 4504) 

s_sub_i32 s19, s0, s18 // (op 4504) 


s_cmp_eq_i32 s1, 0 // (op 4504) 
s_mov_b32 s0, scc // (op 4504) 

s_cmp_eq_i32 s0, 0 // (op 4504) 
s_cselect_b32 s1, s19, s47 // (op 4504) 



v_lshl_add_u32 v21, s1, 6, v16 // (op 4504) 


v_mul_lo_u32 v16, v21, s38 // (op 4504) 

v_add_u32 v21, v22, v16 // (op 4504) 


s_mul_i32 s1, s38, s7 // (op 4506) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 4506) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 4506) least significant
s_mul_hi_u32 s18, s38, s6 // (op 4506) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4506) most significant: sum
s_add_u32 s1, s1, s18 // (op 4506) most significant: sum


s_mov_b32 s68, s40 // (op 4506) reg expression
s_mov_b32 s69, s41 // (op 4506) reg expression
s_mov_b32 s70, s0 // (op 4506) reg expression
s_mov_b32 s71, 131072 // (op 4506) call()

s_mul_hi_i32 s47, s16, s10 // (op 4523) 
v_lshrrev_b32 v16, 2, v1 // (op 4523) 
s_add_i32 s0, s47, s16 // (op 4523) 

v_and_b32 v22, -4, v1 // (op 4523) 
s_ashr_i32 s1, s0, 31 // (op 4523) 
v_sub_u32 v23, v1, v22 // (op 4523) 

s_cmp_eq_i32 s10, 0 // (op 4523) 
s_mov_b32 s47, scc // (op 4523) 
s_cmp_eq_i32 s47, 0 // (op 4523) 
s_cselect_b32 s18, 0, -1 // (op 4523) 

s_and_b32 s19, s13, 31 // (op 4523) 
s_lshl_b32 s47, 1, s19 // (op 4523) 
s_add_i32 s22, s47, s18 // (op 4523) 


s_and_b32 s18, s1, s22 // (op 4523) 


s_add_i32 s1, s0, s18 // (op 4523) 


s_ashr_i32 s0, s1, s19 // (op 4523) 


s_lshl_b32 s1, s13, 24 // (op 4523) 
s_ashr_i32 s47, s1, 31 // (op 4523) 

s_xor_b32 s1, s0, s47 // (op 4523) 

s_sub_i32 s0, s1, s47 // (op 4523) 


s_mul_hi_i32 s1, s0, s14 // (op 4523) 
s_add_i32 s47, s1, s0 // (op 4523) 

s_ashr_i32 s1, s47, 31 // (op 4523) 
s_cmp_eq_i32 s14, 0 // (op 4523) 
s_mov_b32 s18, scc // (op 4523) 
s_cmp_eq_i32 s18, 0 // (op 4523) 
s_cselect_b32 s19, 0, -1 // (op 4523) 

s_and_b32 s18, s15, 31 // (op 4523) 
s_lshl_b32 s22, 1, s18 // (op 4523) 
s_add_i32 s23, s22, s19 // (op 4523) 


s_and_b32 s19, s1, s23 // (op 4523) 


s_add_i32 s1, s47, s19 // (op 4523) 


s_ashr_i32 s19, s1, s18 // (op 4523) 


s_lshl_b32 s1, s15, 24 // (op 4523) 
s_ashr_i32 s18, s1, 31 // (op 4523) 

s_xor_b32 s1, s19, s18 // (op 4523) 

s_sub_i32 s19, s1, s18 // (op 4523) 


s_add_u32 s22, s6, 64 // (op 4523) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4523) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4523) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4523) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4523) 

s_mul_i32 s1, s19, s22 // (op 4523) 

s_sub_i32 s47, s0, s1 // (op 4523) 


s_mul_hi_i32 s0, s16, s9 // (op 4523) 
s_add_i32 s1, s0, s16 // (op 4523) 

s_ashr_i32 s0, s1, 31 // (op 4523) 
s_cmp_eq_i32 s9, 0 // (op 4523) 
s_mov_b32 s18, scc // (op 4523) 
s_cmp_eq_i32 s18, 0 // (op 4523) 
s_cselect_b32 s19, 0, -1 // (op 4523) 

s_and_b32 s18, s45, 31 // (op 4523) 
s_lshl_b32 s24, 1, s18 // (op 4523) 
s_add_i32 s25, s24, s19 // (op 4523) 


s_and_b32 s19, s0, s25 // (op 4523) 


s_add_i32 s0, s1, s19 // (op 4523) 


s_ashr_i32 s1, s0, s18 // (op 4523) 


s_lshl_b32 s0, s45, 24 // (op 4523) 
s_ashr_i32 s18, s0, 31 // (op 4523) 

s_xor_b32 s0, s1, s18 // (op 4523) 

s_sub_i32 s1, s0, s18 // (op 4523) 


s_mul_i32 s0, s11, s22 // (op 4523) 

s_mul_i32 s18, s8, s0 // (op 4523) 

s_mul_i32 s0, s1, s18 // (op 4523) 

s_sub_u32 s18, s16, s0 // (op 4523) 

s_mul_hi_i32 s0, s18, s12 // (op 4523) 
s_add_i32 s19, s0, s18 // (op 4523) 


s_ashr_i32 s0, s19, 31 // (op 4523) 
s_cmp_eq_i32 s12, 0 // (op 4523) 
s_mov_b32 s18, scc // (op 4523) 
s_cmp_eq_i32 s18, 0 // (op 4523) 
s_cselect_b32 s22, 0, -1 // (op 4523) 

s_and_b32 s18, s44, 31 // (op 4523) 
s_lshl_b32 s23, 1, s18 // (op 4523) 
s_add_i32 s24, s23, s22 // (op 4523) 


s_and_b32 s25, s0, s24 // (op 4523) 


s_add_i32 s0, s19, s25 // (op 4523) 


s_ashr_i32 s19, s0, s18 // (op 4523) 


s_lshl_b32 s0, s44, 24 // (op 4523) 
s_ashr_i32 s18, s0, 31 // (op 4523) 

s_xor_b32 s0, s19, s18 // (op 4523) 

s_sub_i32 s19, s0, s18 // (op 4523) 


s_cmp_eq_i32 s1, 0 // (op 4523) 
s_mov_b32 s0, scc // (op 4523) 

s_cmp_eq_i32 s0, 0 // (op 4523) 
s_cselect_b32 s1, s19, s47 // (op 4523) 



v_lshl_add_u32 v22, s1, 6, v16 // (op 4523) 


v_mul_lo_u32 v16, v22, s38 // (op 4523) 

v_add_u32 v22, v23, v16 // (op 4523) 


s_mul_i32 s1, s38, s7 // (op 4525) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 4525) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 4525) least significant
s_mul_hi_u32 s18, s38, s6 // (op 4525) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4525) most significant: sum
s_add_u32 s1, s1, s18 // (op 4525) most significant: sum


s_mov_b32 s68, s40 // (op 4525) reg expression
s_mov_b32 s69, s41 // (op 4525) reg expression
s_mov_b32 s70, s0 // (op 4525) reg expression
s_mov_b32 s71, 131072 // (op 4525) call()




buffer_load_dwordx4 v[24:27], v18, s[60:63], 0 offen // (op 4056) Load value
buffer_load_dwordx2 v[28:29], v18, s[60:63], 0 offen offset:16 // (op 4056) Load value



buffer_load_ubyte v16, v21, s[68:71], 0 offen offset: 0 // (op 4059) Load value



buffer_load_dwordx4 v[30:33], v14, s[48:51], 0 offen // (op 4062) Load value
buffer_load_dwordx2 v[34:35], v14, s[48:51], 0 offen offset:16 // (op 4062) Load value
buffer_load_dwordx4 v[36:39], v14, s[48:51], 0 offen offset:24 // (op 4062) Load value
buffer_load_dwordx2 v[40:41], v14, s[48:51], 0 offen offset:40 // (op 4062) Load value

v_mov_b64 v[42:43], s[34:35] // (op 4065) convert

v_mov_b32 v23, v17 // (op 4065) 
buffer_load_ubyte v44, v23, s[64:67], 0 offen offset: 0 // (op 4065) Load value
v_add_u32 v23, v23, v42 // (op 4065) 
buffer_load_ubyte v45, v23, s[64:67], 0 offen offset: 0 // (op 4065) Load value





buffer_load_dwordx4 v[46:49], v18, s[60:63], 0 offen offset:96 // (op 4068) Load value
buffer_load_dwordx2 v[50:51], v18, s[60:63], 0 offen offset:112 // (op 4068) Load value




buffer_load_ubyte v18, v21, s[68:71], 0 offen offset:4 // (op 4071) Load value




buffer_load_dwordx4 v[52:55], v14, s[48:51], 0 offen offset:96 // (op 4074) Load value
buffer_load_dwordx2 v[56:57], v14, s[48:51], 0 offen offset:112 // (op 4074) Load value
buffer_load_dwordx4 v[58:61], v14, s[48:51], 0 offen offset:120 // (op 4074) Load value
buffer_load_dwordx2 v[62:63], v14, s[48:51], 0 offen offset:136 // (op 4074) Load value


v_mov_b64 v[42:43], s[34:35] // (op 4077) convert

v_mov_b32 v14, v17 // (op 4077) 
buffer_load_ubyte v64, v14, s[64:67], 0 offen offset:4 // (op 4077) Load value
v_add_u32 v14, v14, v42 // (op 4077) 
buffer_load_ubyte v65, v14, s[64:67], 0 offen offset:4 // (op 4077) Load value



v_and_b32 v14, -4, v1 // (op 4653) 
v_sub_u32 v17, v1, v14 // (op 4653) 

v_lshlrev_b32 v14, 5, v17 // (op 4653) 

v_lshrrev_b32 v17, 2, v1 // (op 4653) 
v_lshl_add_u32 v21, v17, 7, v14 // (op 4653) 


v_mul_lo_u32 v14, v21, 6 // (op 4653) 

v_lshrrev_b32 v17, 3, v14 // (op 4653) 




s_waitcnt vmcnt(17)
ds_write_b128 v17, v[24:27] // (op 4079) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v17, v[28:29] offset:16 // (op 4079) Store local data 

v_and_b32 v14, -4, v1 // (op 4769) 
v_sub_u32 v17, v1, v14 // (op 4769) 

v_and_b32 v14, 4294967292, v1 // (op 4769) 
v_add_u32 v21, v14, v17 // (op 4769) 





s_waitcnt vmcnt(15)
ds_write_b8 v21, v16 offset:6144 // (op 4080) Store local data 

v_and_b32 v14, -2, v1 // (op 4595) 
v_sub_u32 v17, v1, v14 // (op 4595) 

v_lshlrev_b32 v14, 6, v17 // (op 4595) 

v_lshrrev_b32 v17, 1, v1 // (op 4595) 
v_lshl_add_u32 v21, v17, 7, v14 // (op 4595) 


v_mul_lo_u32 v14, v21, 6 // (op 4595) 

v_lshrrev_b32 v17, 3, v14 // (op 4595) 




s_waitcnt vmcnt(14)
ds_write_b128 v17, v[30:33] offset:6400 // (op 4081) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v17, v[34:35] offset:6416 // (op 4081) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v17, v[36:39] offset:6424 // (op 4081) Store local data 
s_waitcnt vmcnt(11)
ds_write_b64 v17, v[40:41] offset:6440 // (op 4081) Store local data 

v_and_b32 v14, -4, v1 // (op 4711) 
v_sub_u32 v17, v1, v14 // (op 4711) 

v_lshrrev_b32 v14, 2, v1 // (op 4711) 
v_lshl_add_u32 v21, v14, 3, v17 // (op 4711) 





s_waitcnt vmcnt(10)
ds_write_b8 v21, v44 offset:18688 // (op 4082) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v21, v45 offset:18692 // (op 4082) Store local data 

v_and_b32 v14, -2, v1 // (op 4566) 
v_sub_u32 v17, v1, v14 // (op 4566) 

v_lshlrev_b32 v14, 6, v17 // (op 4566) 

v_lshrrev_b32 v17, 1, v1 // (op 4566) 
v_lshl_add_u32 v21, v17, 7, v14 // (op 4566) 


v_mul_lo_u32 v14, v21, 6 // (op 4566) 

v_lshrrev_b32 v17, 3, v14 // (op 4566) 

v_and_b32 v14, -4, v1 // (op 4624) 
v_sub_u32 v21, v1, v14 // (op 4624) 

v_lshlrev_b32 v14, 5, v21 // (op 4624) 

v_lshrrev_b32 v21, 2, v1 // (op 4624) 
v_lshl_add_u32 v23, v21, 7, v14 // (op 4624) 


v_mul_lo_u32 v14, v23, 6 // (op 4624) 

v_lshrrev_b32 v21, 3, v14 // (op 4624) 

v_and_b32 v14, -4, v1 // (op 4682) 
v_sub_u32 v23, v1, v14 // (op 4682) 

v_lshrrev_b32 v14, 2, v1 // (op 4682) 
v_lshl_add_u32 v42, v14, 3, v23 // (op 4682) 


v_and_b32 v14, -4, v1 // (op 4740) 
v_sub_u32 v23, v1, v14 // (op 4740) 

v_and_b32 v14, 4294967292, v1 // (op 4740) 
v_add_u32 v43, v14, v23 // (op 4740) 


v_and_b32 v14, -2, v1 // (op 4821) 
v_sub_u32 v23, v1, v14 // (op 4821) 

v_lshlrev_b32 v14, 6, v23 // (op 4821) 

v_lshrrev_b32 v23, 1, v1 // (op 4821) 
v_lshl_add_u32 v66, v23, 7, v14 // (op 4821) 


v_mul_lo_u32 v14, v66, 6 // (op 4821) 

v_lshrrev_b32 v23, 3, v14 // (op 4821) 

v_and_b32 v14, -4, v1 // (op 4850) 
v_sub_u32 v66, v1, v14 // (op 4850) 

v_lshlrev_b32 v14, 5, v66 // (op 4850) 

v_lshrrev_b32 v66, 2, v1 // (op 4850) 
v_lshl_add_u32 v67, v66, 7, v14 // (op 4850) 


v_mul_lo_u32 v14, v67, 6 // (op 4850) 

v_lshrrev_b32 v66, 3, v14 // (op 4850) 

v_and_b32 v14, -4, v1 // (op 4879) 
v_sub_u32 v67, v1, v14 // (op 4879) 

v_lshrrev_b32 v14, 2, v1 // (op 4879) 
v_lshl_add_u32 v68, v14, 3, v67 // (op 4879) 


v_and_b32 v14, -4, v1 // (op 4908) 
v_sub_u32 v67, v1, v14 // (op 4908) 

v_and_b32 v14, 4294967292, v1 // (op 4908) 
v_add_u32 v69, v14, v67 // (op 4908) 


s_mov_b32 s47, 0 // (op 1365) call()
s_ashr_i64 s[0:1], s[42:43], 7 // (op 1364) 
s_mov_b32 s18, s0 // (op 1364) convert

s_and_b32 s19, s18, 4294967294 // (op 1364) 

s_cmp_lt_u32 s47, s19 // (op 1364) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_1364 // (op 1364) Condition: Top (jump to RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_1364 if false)
RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_1364:


s_barrier  // (op 4988) 
s_add_u32 s47, s47, 2 // (op 1366) 



v_mov_b32 v14, v19 // (op 10) 

ds_read_b128 v[70:73], v3 offset:6400 // (op 72) Load local data 
ds_read_b64 v[74:75], v3 offset:6416 // (op 72) Load local data 
buffer_load_dwordx4 v[24:27], v14, s[60:63], 0 offen offset:192 // (op 10) Load value
s_barrier  // (op 5001) 
buffer_load_dwordx2 v[28:29], v14, s[60:63], 0 offen offset:208 // (op 10) Load value





v_mov_b32 v67, v22 // (op 25) 

buffer_load_ubyte v16, v67, s[68:71], 0 offen offset:8 // (op 25) Load value
ds_read_b128 v[76:79], v4 // (op 77) Load local data 


ds_read_b64 v[80:81], v4 offset:16 // (op 77) Load local data 
v_mov_b32 v82, v15 // (op 4) 

s_barrier  // (op 5015) 
buffer_load_dwordx4 v[30:33], v82, s[48:51], 0 offen offset:192 // (op 4) Load value



ds_read_u8 v83, v0 offset:18688 // (op 82) Load local data 
buffer_load_dwordx2 v[34:35], v82, s[48:51], 0 offen offset:208 // (op 4) Load value
s_barrier  // (op 5029) 
buffer_load_dwordx4 v[36:39], v82, s[48:51], 0 offen offset:216 // (op 4) Load value



ds_read_u8 v84, v5 offset:6144 // (op 87) Load local data 
buffer_load_dwordx2 v[40:41], v82, s[48:51], 0 offen offset:232 // (op 4) Load value




v_mov_b64 v[86:87], s[34:35] // (op 17) convert

v_mov_b32 v85, v20 // (op 17) 
v_mov_b32 v88, v85 // (op 17) 
buffer_load_ubyte v44, v88, s[64:67], 0 offen offset:8 // (op 17) Load value
v_add_u32 v88, v88, v86 // (op 17) 
buffer_load_ubyte v45, v88, s[64:67], 0 offen offset:8 // (op 17) Load value


ds_read_b128 v[86:89], v4 offset:3072 // (op 375) Load local data 
s_barrier  // (op 5052) 
ds_read_b64 v[90:91], v4 offset:3088 // (op 375) Load local data 






ds_read_u8 v92, v5 offset:6272 // (op 379) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:75], v[76:81], a[0:3], v83, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 90) 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:75], v[86:91], a[4:7], v83, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 380) 
ds_read_b128 v[94:97], v3 offset:9472 // (op 626) Load local data 


ds_read_b64 v[98:99], v3 offset:9488 // (op 626) Load local data 
s_waitcnt vmcnt(17)
ds_write_b128 v66, v[46:49] offset:19200 // (op 257) Store local data 



ds_read_u8 v83, v0 offset:18816 // (op 629) Load local data 
s_waitcnt vmcnt(16)
ds_write_b64 v66, v[50:51] offset:19216 // (op 257) Store local data 



ds_read_b128 v[70:73], v3 offset:12544 // (op 1012) Load local data 
s_barrier  // (op 5064) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[94:99], v[76:81], a[8:11], v83, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 677) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[94:99], v[86:91], a[12:15], v83, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 631) 
ds_read_b64 v[74:75], v3 offset:12560 // (op 1012) Load local data 





ds_read_u8 v83, v0 offset:18944 // (op 1015) Load local data 
s_waitcnt vmcnt(15)
ds_write_b8 v69, v18 offset:25344 // (op 259) Store local data 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:75], v[76:81], a[16:19], v83, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1063) 
s_barrier  // (op 5046) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:75], v[86:91], a[20:23], v83, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1017) 





ds_read_b128 v[70:73], v3 offset:15616 // (op 1398) Load local data 
ds_read_b64 v[74:75], v3 offset:15632 // (op 1398) Load local data 
s_waitcnt vmcnt(14)
ds_write_b128 v23, v[52:55] offset:25600 // (op 247) Store local data 



ds_read_u8 v83, v0 offset:19072 // (op 1401) Load local data 
s_waitcnt vmcnt(13)
ds_write_b64 v23, v[56:57] offset:25616 // (op 247) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:75], v[76:81], a[24:27], v83, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1449) 
s_waitcnt vmcnt(12)
ds_write_b128 v23, v[58:61] offset:25624 // (op 247) Store local data 


s_waitcnt vmcnt(11)
ds_write_b64 v23, v[62:63] offset:25640 // (op 247) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:75], v[86:91], a[28:31], v83, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1403) 




s_barrier  // (op 5058) 



s_waitcnt vmcnt(10)
ds_write_b8 v68, v64 offset:37888 // (op 258) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v68, v65 offset:37892 // (op 258) Store local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 5043) 
buffer_load_dwordx4 v[46:49], v14, s[60:63], 0 offen offset:288 // (op 260) Load value



ds_read_b128 v[70:73], v7 offset:25600 // (op 249) Load local data 
ds_read_b64 v[74:75], v7 offset:25616 // (op 249) Load local data 
buffer_load_dwordx2 v[50:51], v14, s[60:63], 0 offen offset:304 // (op 260) Load value

s_barrier  // (op 5049) 






buffer_load_ubyte v18, v67, s[68:71], 0 offen offset:12 // (op 262) Load value
v_add_u32 v19, 192, v19 // (op 4387) 

ds_read_b128 v[76:79], v8 offset:19200 // (op 251) Load local data 



v_add_u32 v22, 8, v22 // (op 4515) 
ds_read_b64 v[80:81], v8 offset:19216 // (op 251) Load local data 
buffer_load_dwordx4 v[52:55], v82, s[48:51], 0 offen offset:288 // (op 246) Load value
buffer_load_dwordx2 v[56:57], v82, s[48:51], 0 offen offset:304 // (op 246) Load value
buffer_load_dwordx4 v[58:61], v82, s[48:51], 0 offen offset:312 // (op 246) Load value
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5055) 
buffer_load_dwordx2 v[62:63], v82, s[48:51], 0 offen offset:328 // (op 246) Load value




ds_read_u8 v14, v6 offset:37888 // (op 253) Load local data 

s_barrier  // (op 5061) 
v_add_u32 v15, 192, v15 // (op 4323) 



ds_read_u8 v67, v9 offset:25344 // (op 255) Load local data 
v_mov_b64 v[82:83], s[34:35] // (op 261) convert




v_mov_b32 v84, v85 // (op 261) 
buffer_load_ubyte v64, v84, s[64:67], 0 offen offset:12 // (op 261) Load value
v_add_u32 v84, v84, v82 // (op 261) 
buffer_load_ubyte v65, v84, s[64:67], 0 offen offset:12 // (op 261) Load value


ds_read_b128 v[86:89], v8 offset:22272 // (op 385) Load local data 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:75], v[76:81], a[0:3], v14, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 256) 
s_barrier  // (op 5004) 
ds_read_b64 v[90:91], v8 offset:22288 // (op 385) Load local data 



v_add_u32 v20, 8, v20 // (op 4451) 



s_barrier  // (op 5125) 
ds_read_u8 v82, v9 offset:25472 // (op 389) Load local data 
s_waitcnt vmcnt(17)
ds_write_b128 v21, v[24:27] // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v21, v[28:29] offset:16 // (op 13) Store local data 



s_barrier  // (op 5129) 
s_barrier  // (op 5032) 
ds_read_b128 v[92:95], v7 offset:28672 // (op 634) Load local data 



ds_read_b64 v[96:97], v7 offset:28688 // (op 634) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:75], v[86:91], a[4:7], v14, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 381) 





ds_read_u8 v14, v6 offset:38016 // (op 637) Load local data 
s_waitcnt vmcnt(15)
ds_write_b8 v43, v16 offset:6144 // (op 30) Store local data 



ds_read_b128 v[70:73], v7 offset:31744 // (op 1020) Load local data 
s_barrier  // (op 4991) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[92:97], v[76:81], a[8:11], v14, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 678) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:97], v[86:91], a[12:15], v14, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 632) 
ds_read_b64 v[74:75], v7 offset:31760 // (op 1020) Load local data 





ds_read_u8 v14, v6 offset:38144 // (op 1023) Load local data 
s_waitcnt vmcnt(14)
ds_write_b128 v17, v[30:33] offset:6400 // (op 6) Store local data 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:75], v[76:81], a[16:19], v14, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1064) 
ds_read_b128 v[92:95], v7 offset:34816 // (op 1406) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:75], v[86:91], a[20:23], v14, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1018) 
ds_read_b64 v[96:97], v7 offset:34832 // (op 1406) Load local data 


s_waitcnt vmcnt(13)
ds_write_b64 v17, v[34:35] offset:6416 // (op 6) Store local data 



s_barrier  // (op 5161) 
ds_read_u8 v14, v6 offset:38272 // (op 1409) Load local data 
s_waitcnt vmcnt(12)
ds_write_b128 v17, v[36:39] offset:6424 // (op 6) Store local data 
s_waitcnt vmcnt(11)
ds_write_b64 v17, v[40:41] offset:6440 // (op 6) Store local data 
s_barrier  // (op 5165) 
s_barrier  // (op 5018) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:97], v[76:81], a[24:27], v14, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1450) 





s_waitcnt vmcnt(10)
ds_write_b8 v42, v44 offset:18688 // (op 21) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:97], v[86:91], a[28:31], v14, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1404) 
s_waitcnt vmcnt(9)
ds_write_b8 v42, v45 offset:18692 // (op 21) Store local data 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 1364) 
s_mov_b32 s18, s0 // (op 1364) convert

s_and_b32 s19, s18, 4294967294 // (op 1364) 

s_cmp_lt_u32 s47, s19 // (op 1364) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_1364 // (op 1364) Condition: Bottom (jump to RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_1364 if true)
RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_1364:

















s_mul_hi_i32 s47, s16, s10 // (op 4279) 
v_lshrrev_b32 v14, 1, v1 // (op 4279) 
s_add_i32 s0, s47, s16 // (op 4279) 

v_and_b32 v15, -2, v1 // (op 4279) 
s_ashr_i32 s1, s0, 31 // (op 4279) 
v_sub_u32 v16, v1, v15 // (op 4279) 

s_cmp_eq_i32 s10, 0 // (op 4279) 
s_mov_b32 s47, scc // (op 4279) 
v_lshlrev_b32 v15, 6, v16 // (op 4279) 

s_cmp_eq_i32 s47, 0 // (op 4279) 
s_cselect_b32 s18, 0, -1 // (op 4279) 

s_and_b32 s19, s13, 31 // (op 4279) 
s_lshl_b32 s47, 1, s19 // (op 4279) 
s_add_i32 s22, s47, s18 // (op 4279) 


s_and_b32 s18, s1, s22 // (op 4279) 


s_add_i32 s1, s0, s18 // (op 4279) 


s_ashr_i32 s0, s1, s19 // (op 4279) 


s_lshl_b32 s1, s13, 24 // (op 4279) 
s_ashr_i32 s47, s1, 31 // (op 4279) 

s_xor_b32 s1, s0, s47 // (op 4279) 

s_sub_i32 s0, s1, s47 // (op 4279) 


s_mul_hi_i32 s1, s0, s14 // (op 4279) 
s_add_i32 s47, s1, s0 // (op 4279) 

s_ashr_i32 s1, s47, 31 // (op 4279) 
s_cmp_eq_i32 s14, 0 // (op 4279) 
s_mov_b32 s18, scc // (op 4279) 
s_cmp_eq_i32 s18, 0 // (op 4279) 
s_cselect_b32 s19, 0, -1 // (op 4279) 

s_and_b32 s18, s15, 31 // (op 4279) 
s_lshl_b32 s22, 1, s18 // (op 4279) 
s_add_i32 s23, s22, s19 // (op 4279) 


s_and_b32 s19, s1, s23 // (op 4279) 


s_add_i32 s1, s47, s19 // (op 4279) 


s_ashr_i32 s19, s1, s18 // (op 4279) 


s_lshl_b32 s1, s15, 24 // (op 4279) 
s_ashr_i32 s18, s1, 31 // (op 4279) 

s_xor_b32 s1, s19, s18 // (op 4279) 

s_sub_i32 s19, s1, s18 // (op 4279) 


s_mul_i32 s1, s0, s11 // (op 4279) 

s_sub_u32 s0, s16, s1 // (op 4279) 

s_mul_i32 s1, s19, s11 // (op 4279) 
s_add_u32 s1, s1, s0 // (op 4279) 


s_mul_hi_i32 s0, s16, s9 // (op 4279) 
s_add_i32 s47, s0, s16 // (op 4279) 

s_ashr_i32 s0, s47, 31 // (op 4279) 
s_cmp_eq_i32 s9, 0 // (op 4279) 
s_mov_b32 s18, scc // (op 4279) 
s_cmp_eq_i32 s18, 0 // (op 4279) 
s_cselect_b32 s19, 0, -1 // (op 4279) 

s_and_b32 s18, s45, 31 // (op 4279) 
s_lshl_b32 s22, 1, s18 // (op 4279) 
s_add_i32 s23, s22, s19 // (op 4279) 


s_and_b32 s19, s0, s23 // (op 4279) 


s_add_i32 s0, s47, s19 // (op 4279) 


s_ashr_i32 s19, s0, s18 // (op 4279) 


s_lshl_b32 s0, s45, 24 // (op 4279) 
s_ashr_i32 s18, s0, 31 // (op 4279) 

s_xor_b32 s0, s19, s18 // (op 4279) 

s_sub_i32 s19, s0, s18 // (op 4279) 


s_add_u32 s22, s6, 64 // (op 4279) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4279) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4279) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4279) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4279) 

s_mul_i32 s0, s11, s22 // (op 4279) 

s_mul_i32 s18, s8, s0 // (op 4279) 

s_mul_i32 s0, s19, s18 // (op 4279) 

s_sub_u32 s18, s16, s0 // (op 4279) 

s_mov_b32 s0, s18 // (op 4279) 
s_mul_hi_i32 s47, s0, s12 // (op 4279) 
s_add_i32 s22, s47, s0 // (op 4279) 


s_ashr_i32 s0, s22, 31 // (op 4279) 
s_cmp_eq_i32 s12, 0 // (op 4279) 
s_mov_b32 s47, scc // (op 4279) 
s_cmp_eq_i32 s47, 0 // (op 4279) 
s_cselect_b32 s23, 0, -1 // (op 4279) 

s_and_b32 s47, s44, 31 // (op 4279) 
s_lshl_b32 s24, 1, s47 // (op 4279) 
s_add_i32 s25, s24, s23 // (op 4279) 


s_and_b32 s23, s0, s25 // (op 4279) 


s_add_i32 s0, s22, s23 // (op 4279) 


s_ashr_i32 s22, s0, s47 // (op 4279) 


s_lshl_b32 s0, s44, 24 // (op 4279) 
s_ashr_i32 s47, s0, 31 // (op 4279) 

s_xor_b32 s0, s22, s47 // (op 4279) 

s_sub_i32 s22, s0, s47 // (op 4279) 


s_mul_i32 s0, s46, s11 // (op 4279) 
s_add_u32 s24, s20, 128 // (op 4279) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4279) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 4279) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 4279) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 7 // (op 4279) 

s_sub_i32 s23, s24, s0 // (op 4279) 

s_mul_i32 s47, s22, s23 // (op 4279) 


s_sub_u32 s22, s18, s47 // (op 4279) 


s_add_u32 s18, s22, s0 // (op 4279) 


s_cmp_eq_i32 s19, 0 // (op 4279) 
s_mov_b32 s0, scc // (op 4279) 

s_cmp_eq_i32 s0, 0 // (op 4279) 
s_cselect_b32 s19, s18, s1 // (op 4279) 



v_lshl_add_u32 v16, s19, 7, v14 // (op 4279) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 4279) 
s_lshr_b32 s47, s0, 1 // (op 4279) 

v_lshl_add_u32 v14, s47, 8, v15 // (op 4279) 


v_mov_b32 v45, 0 // (op 4279) convert
v_mov_b32 v44, v14 // (op 4279) convert

v_mul_lo_u32 v15, v16, s27 // (op 4279) most significant: low of low * high
v_mul_hi_u32 v17, v16, s26 // (op 4279) most significant: high of low * low
v_mul_lo_u32 v14, v16, s26 // (op 4279) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 4279) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v44 // (op 4279) least significant half
v_addc_co_u32 v15, s[0:1], v15, v45, s[0:1] // (op 4279) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 4279) most significant: low of high * low
v_mul_hi_u32 v19, v14, 6 // (op 4279) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 4279) least significant: low of low * low
v_add_u32 v17, v17, v19 // (op 4279) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 4279) 

v_mov_b32 v16, v14 // (op 4279) convert

s_mul_i32 s1, s21, 6 // (op 4281) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 4281) least significant
s_mul_hi_u32 s47, s20, 6 // (op 4281) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4281) most significant: sum

s_mul_i32 s19, s26, s1 // (op 4281) most significant: low of low * high
s_mul_i32 s47, s27, s0 // (op 4281) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 4281) least significant
s_mul_hi_u32 s22, s26, s0 // (op 4281) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 4281) most significant: sum
s_add_u32 s19, s19, s22 // (op 4281) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4281) 

s_mov_b32 s48, s28 // (op 4281) reg expression
s_mov_b32 s49, s29 // (op 4281) reg expression
s_mov_b32 s50, s0 // (op 4281) reg expression
s_mov_b32 s51, 131072 // (op 4281) call()

s_mul_hi_i32 s47, s16, s10 // (op 4353) 
v_lshrrev_b32 v17, 2, v1 // (op 4353) 
s_add_i32 s0, s47, s16 // (op 4353) 

v_and_b32 v14, -4, v1 // (op 4353) 
s_ashr_i32 s1, s0, 31 // (op 4353) 
v_sub_u32 v15, v1, v14 // (op 4353) 

s_cmp_eq_i32 s10, 0 // (op 4353) 
s_mov_b32 s47, scc // (op 4353) 
v_lshlrev_b32 v14, 5, v15 // (op 4353) 

s_cmp_eq_i32 s47, 0 // (op 4353) 
s_cselect_b32 s18, 0, -1 // (op 4353) 

s_and_b32 s19, s13, 31 // (op 4353) 
s_lshl_b32 s47, 1, s19 // (op 4353) 
s_add_i32 s22, s47, s18 // (op 4353) 


s_and_b32 s18, s1, s22 // (op 4353) 


s_add_i32 s1, s0, s18 // (op 4353) 


s_ashr_i32 s0, s1, s19 // (op 4353) 


s_lshl_b32 s1, s13, 24 // (op 4353) 
s_ashr_i32 s47, s1, 31 // (op 4353) 

s_xor_b32 s1, s0, s47 // (op 4353) 

s_sub_i32 s0, s1, s47 // (op 4353) 


s_mul_hi_i32 s1, s0, s14 // (op 4353) 
s_add_i32 s47, s1, s0 // (op 4353) 

s_ashr_i32 s1, s47, 31 // (op 4353) 
s_cmp_eq_i32 s14, 0 // (op 4353) 
s_mov_b32 s18, scc // (op 4353) 
s_cmp_eq_i32 s18, 0 // (op 4353) 
s_cselect_b32 s19, 0, -1 // (op 4353) 

s_and_b32 s18, s15, 31 // (op 4353) 
s_lshl_b32 s22, 1, s18 // (op 4353) 
s_add_i32 s23, s22, s19 // (op 4353) 


s_and_b32 s19, s1, s23 // (op 4353) 


s_add_i32 s1, s47, s19 // (op 4353) 


s_ashr_i32 s19, s1, s18 // (op 4353) 


s_lshl_b32 s1, s15, 24 // (op 4353) 
s_ashr_i32 s18, s1, 31 // (op 4353) 

s_xor_b32 s1, s19, s18 // (op 4353) 

s_sub_i32 s19, s1, s18 // (op 4353) 


s_add_u32 s22, s6, 64 // (op 4353) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4353) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4353) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4353) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4353) 

s_mul_i32 s1, s19, s22 // (op 4353) 

s_sub_i32 s47, s0, s1 // (op 4353) 


s_mul_hi_i32 s0, s16, s9 // (op 4353) 
s_add_i32 s1, s0, s16 // (op 4353) 

s_ashr_i32 s0, s1, 31 // (op 4353) 
s_cmp_eq_i32 s9, 0 // (op 4353) 
s_mov_b32 s18, scc // (op 4353) 
s_cmp_eq_i32 s18, 0 // (op 4353) 
s_cselect_b32 s19, 0, -1 // (op 4353) 

s_and_b32 s18, s45, 31 // (op 4353) 
s_lshl_b32 s24, 1, s18 // (op 4353) 
s_add_i32 s25, s24, s19 // (op 4353) 


s_and_b32 s19, s0, s25 // (op 4353) 


s_add_i32 s0, s1, s19 // (op 4353) 


s_ashr_i32 s1, s0, s18 // (op 4353) 


s_lshl_b32 s0, s45, 24 // (op 4353) 
s_ashr_i32 s18, s0, 31 // (op 4353) 

s_xor_b32 s0, s1, s18 // (op 4353) 

s_sub_i32 s1, s0, s18 // (op 4353) 


s_mul_i32 s0, s11, s22 // (op 4353) 

s_mul_i32 s18, s8, s0 // (op 4353) 

s_mul_i32 s0, s1, s18 // (op 4353) 

s_sub_u32 s18, s16, s0 // (op 4353) 

s_mul_hi_i32 s0, s18, s12 // (op 4353) 
s_add_i32 s19, s0, s18 // (op 4353) 


s_ashr_i32 s0, s19, 31 // (op 4353) 
s_cmp_eq_i32 s12, 0 // (op 4353) 
s_mov_b32 s18, scc // (op 4353) 
s_cmp_eq_i32 s18, 0 // (op 4353) 
s_cselect_b32 s22, 0, -1 // (op 4353) 

s_and_b32 s18, s44, 31 // (op 4353) 
s_lshl_b32 s23, 1, s18 // (op 4353) 
s_add_i32 s24, s23, s22 // (op 4353) 


s_and_b32 s25, s0, s24 // (op 4353) 


s_add_i32 s0, s19, s25 // (op 4353) 


s_ashr_i32 s19, s0, s18 // (op 4353) 


s_lshl_b32 s0, s44, 24 // (op 4353) 
s_ashr_i32 s18, s0, 31 // (op 4353) 

s_xor_b32 s0, s19, s18 // (op 4353) 

s_sub_i32 s19, s0, s18 // (op 4353) 


s_cmp_eq_i32 s1, 0 // (op 4353) 
s_mov_b32 s0, scc // (op 4353) 

s_cmp_eq_i32 s0, 0 // (op 4353) 
s_cselect_b32 s1, s19, s47 // (op 4353) 



v_lshl_add_u32 v15, s1, 6, v17 // (op 4353) 


v_mul_lo_u32 v45, v15, s31 // (op 4353) most significant: low of low * high
v_mul_hi_u32 v17, v15, s30 // (op 4353) most significant: high of low * low
v_mul_lo_u32 v44, v15, s30 // (op 4353) least significant: low of low * low
v_add_u32 v45, v45, v17 // (op 4353) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 4353) 
s_lshr_b32 s47, s0, 1 // (op 4353) 

v_lshl_add_u32 v15, s47, 8, v14 // (op 4353) 


v_mov_b32 v43, 0 // (op 4353) convert
v_mov_b32 v42, v15 // (op 4353) convert

v_add_co_u32 v14, s[0:1], v42, v44 // (op 4353) least significant half
v_addc_co_u32 v15, s[0:1], v43, v45, s[0:1] // (op 4353) most significant half



v_mul_lo_u32 v45, v15, 6 // (op 4353) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 4353) most significant: high of low * low
v_mul_lo_u32 v44, v14, 6 // (op 4353) least significant: low of low * low
v_add_u32 v45, v45, v17 // (op 4353) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[44:45] // (op 4353) 

v_mov_b32 v17, v14 // (op 4353) convert

s_mul_i32 s1, s7, 6 // (op 4355) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 4355) least significant
s_mul_hi_u32 s47, s6, 6 // (op 4355) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4355) most significant: sum

s_mul_i32 s19, s30, s1 // (op 4355) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 4355) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 4355) least significant
s_mul_hi_u32 s22, s30, s0 // (op 4355) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 4355) most significant: sum
s_add_u32 s19, s19, s22 // (op 4355) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4355) 

s_mov_b32 s60, s32 // (op 4355) reg expression
s_mov_b32 s61, s33 // (op 4355) reg expression
s_mov_b32 s62, s0 // (op 4355) reg expression
s_mov_b32 s63, 131072 // (op 4355) call()

s_mul_hi_i32 s47, s16, s10 // (op 4417) 
v_lshrrev_b32 v14, 2, v1 // (op 4417) 
s_add_i32 s0, s47, s16 // (op 4417) 

v_lshlrev_b32 v15, 1, v14 // (op 4417) 

s_ashr_i32 s1, s0, 31 // (op 4417) 
v_and_b32 v14, -4, v1 // (op 4417) 
s_cmp_eq_i32 s10, 0 // (op 4417) 
s_mov_b32 s47, scc // (op 4417) 
v_sub_u32 v19, v1, v14 // (op 4417) 

s_cmp_eq_i32 s47, 0 // (op 4417) 
s_cselect_b32 s18, 0, -1 // (op 4417) 

s_and_b32 s19, s13, 31 // (op 4417) 
s_lshl_b32 s47, 1, s19 // (op 4417) 
s_add_i32 s22, s47, s18 // (op 4417) 


s_and_b32 s18, s1, s22 // (op 4417) 


s_add_i32 s1, s0, s18 // (op 4417) 


s_ashr_i32 s0, s1, s19 // (op 4417) 


s_lshl_b32 s1, s13, 24 // (op 4417) 
s_ashr_i32 s47, s1, 31 // (op 4417) 

s_xor_b32 s1, s0, s47 // (op 4417) 

s_sub_i32 s0, s1, s47 // (op 4417) 


s_mul_hi_i32 s1, s0, s14 // (op 4417) 
s_add_i32 s47, s1, s0 // (op 4417) 

s_ashr_i32 s1, s47, 31 // (op 4417) 
s_cmp_eq_i32 s14, 0 // (op 4417) 
s_mov_b32 s18, scc // (op 4417) 
s_cmp_eq_i32 s18, 0 // (op 4417) 
s_cselect_b32 s19, 0, -1 // (op 4417) 

s_and_b32 s18, s15, 31 // (op 4417) 
s_lshl_b32 s22, 1, s18 // (op 4417) 
s_add_i32 s23, s22, s19 // (op 4417) 


s_and_b32 s19, s1, s23 // (op 4417) 


s_add_i32 s1, s47, s19 // (op 4417) 


s_ashr_i32 s19, s1, s18 // (op 4417) 


s_lshl_b32 s1, s15, 24 // (op 4417) 
s_ashr_i32 s18, s1, 31 // (op 4417) 

s_xor_b32 s1, s19, s18 // (op 4417) 

s_sub_i32 s19, s1, s18 // (op 4417) 


s_mul_i32 s1, s0, s11 // (op 4417) 

s_sub_u32 s0, s16, s1 // (op 4417) 

s_mul_i32 s1, s19, s11 // (op 4417) 
s_add_u32 s1, s1, s0 // (op 4417) 


s_mul_hi_i32 s0, s16, s9 // (op 4417) 
s_add_i32 s47, s0, s16 // (op 4417) 

s_ashr_i32 s0, s47, 31 // (op 4417) 
s_cmp_eq_i32 s9, 0 // (op 4417) 
s_mov_b32 s18, scc // (op 4417) 
s_cmp_eq_i32 s18, 0 // (op 4417) 
s_cselect_b32 s19, 0, -1 // (op 4417) 

s_and_b32 s18, s45, 31 // (op 4417) 
s_lshl_b32 s22, 1, s18 // (op 4417) 
s_add_i32 s23, s22, s19 // (op 4417) 


s_and_b32 s19, s0, s23 // (op 4417) 


s_add_i32 s0, s47, s19 // (op 4417) 


s_ashr_i32 s19, s0, s18 // (op 4417) 


s_lshl_b32 s0, s45, 24 // (op 4417) 
s_ashr_i32 s18, s0, 31 // (op 4417) 

s_xor_b32 s0, s19, s18 // (op 4417) 

s_sub_i32 s19, s0, s18 // (op 4417) 


s_add_u32 s22, s6, 64 // (op 4417) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4417) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4417) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4417) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4417) 

s_mul_i32 s0, s11, s22 // (op 4417) 

s_mul_i32 s18, s8, s0 // (op 4417) 

s_mul_i32 s0, s19, s18 // (op 4417) 

s_sub_u32 s18, s16, s0 // (op 4417) 

s_mov_b32 s0, s18 // (op 4417) 
s_mul_hi_i32 s47, s0, s12 // (op 4417) 
s_add_i32 s22, s47, s0 // (op 4417) 


s_ashr_i32 s0, s22, 31 // (op 4417) 
s_cmp_eq_i32 s12, 0 // (op 4417) 
s_mov_b32 s47, scc // (op 4417) 
s_cmp_eq_i32 s47, 0 // (op 4417) 
s_cselect_b32 s23, 0, -1 // (op 4417) 

s_and_b32 s47, s44, 31 // (op 4417) 
s_lshl_b32 s24, 1, s47 // (op 4417) 
s_add_i32 s25, s24, s23 // (op 4417) 


s_and_b32 s23, s0, s25 // (op 4417) 


s_add_i32 s0, s22, s23 // (op 4417) 


s_ashr_i32 s22, s0, s47 // (op 4417) 


s_lshl_b32 s0, s44, 24 // (op 4417) 
s_ashr_i32 s47, s0, 31 // (op 4417) 

s_xor_b32 s0, s22, s47 // (op 4417) 

s_sub_i32 s22, s0, s47 // (op 4417) 


s_mul_i32 s0, s46, s11 // (op 4417) 
s_add_u32 s24, s20, 128 // (op 4417) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4417) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 4417) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 4417) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 7 // (op 4417) 

s_sub_i32 s23, s24, s0 // (op 4417) 

s_mul_i32 s47, s22, s23 // (op 4417) 


s_sub_u32 s22, s18, s47 // (op 4417) 


s_add_u32 s18, s22, s0 // (op 4417) 


s_cmp_eq_i32 s19, 0 // (op 4417) 
s_mov_b32 s0, scc // (op 4417) 

s_cmp_eq_i32 s0, 0 // (op 4417) 
s_cselect_b32 s19, s18, s1 // (op 4417) 



v_lshl_add_u32 v14, s19, 7, v15 // (op 4417) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 4417) 
s_lshr_b32 s47, s0, 1 // (op 4417) 

v_lshl_add_u32 v15, s47, 3, v19 // (op 4417) 


v_mul_lo_u32 v19, v14, s34 // (op 4417) 
v_add_u32 v19, v19, v15 // (op 4417) 


s_mul_i32 s1, s34, s21 // (op 4419) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 4419) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 4419) least significant
s_mul_hi_u32 s18, s34, s20 // (op 4419) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4419) most significant: sum
s_add_u32 s1, s1, s18 // (op 4419) most significant: sum


s_mov_b32 s64, s36 // (op 4419) reg expression
s_mov_b32 s65, s37 // (op 4419) reg expression
s_mov_b32 s66, s0 // (op 4419) reg expression
s_mov_b32 s67, 131072 // (op 4419) call()

s_mul_hi_i32 s47, s16, s10 // (op 4481) 
v_lshrrev_b32 v14, 2, v1 // (op 4481) 
s_add_i32 s0, s47, s16 // (op 4481) 

v_and_b32 v15, -4, v1 // (op 4481) 
s_ashr_i32 s1, s0, 31 // (op 4481) 
v_sub_u32 v20, v1, v15 // (op 4481) 

s_cmp_eq_i32 s10, 0 // (op 4481) 
s_mov_b32 s47, scc // (op 4481) 
s_cmp_eq_i32 s47, 0 // (op 4481) 
s_cselect_b32 s18, 0, -1 // (op 4481) 

s_and_b32 s19, s13, 31 // (op 4481) 
s_lshl_b32 s47, 1, s19 // (op 4481) 
s_add_i32 s22, s47, s18 // (op 4481) 


s_and_b32 s18, s1, s22 // (op 4481) 


s_add_i32 s1, s0, s18 // (op 4481) 


s_ashr_i32 s0, s1, s19 // (op 4481) 


s_lshl_b32 s1, s13, 24 // (op 4481) 
s_ashr_i32 s47, s1, 31 // (op 4481) 

s_xor_b32 s1, s0, s47 // (op 4481) 

s_sub_i32 s0, s1, s47 // (op 4481) 


s_mul_hi_i32 s1, s0, s14 // (op 4481) 
s_add_i32 s47, s1, s0 // (op 4481) 

s_ashr_i32 s1, s47, 31 // (op 4481) 
s_cmp_eq_i32 s14, 0 // (op 4481) 
s_mov_b32 s18, scc // (op 4481) 
s_cmp_eq_i32 s18, 0 // (op 4481) 
s_cselect_b32 s19, 0, -1 // (op 4481) 

s_and_b32 s18, s15, 31 // (op 4481) 
s_lshl_b32 s22, 1, s18 // (op 4481) 
s_add_i32 s23, s22, s19 // (op 4481) 


s_and_b32 s19, s1, s23 // (op 4481) 


s_add_i32 s1, s47, s19 // (op 4481) 


s_ashr_i32 s19, s1, s18 // (op 4481) 


s_lshl_b32 s1, s15, 24 // (op 4481) 
s_ashr_i32 s18, s1, 31 // (op 4481) 

s_xor_b32 s1, s19, s18 // (op 4481) 

s_sub_i32 s19, s1, s18 // (op 4481) 


s_add_u32 s22, s6, 64 // (op 4481) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4481) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4481) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4481) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4481) 

s_mul_i32 s1, s19, s22 // (op 4481) 

s_sub_i32 s47, s0, s1 // (op 4481) 


s_mul_hi_i32 s0, s16, s9 // (op 4481) 
s_add_i32 s1, s0, s16 // (op 4481) 

s_ashr_i32 s0, s1, 31 // (op 4481) 
s_cmp_eq_i32 s9, 0 // (op 4481) 
s_mov_b32 s18, scc // (op 4481) 
s_cmp_eq_i32 s18, 0 // (op 4481) 
s_cselect_b32 s19, 0, -1 // (op 4481) 

s_and_b32 s18, s45, 31 // (op 4481) 
s_lshl_b32 s24, 1, s18 // (op 4481) 
s_add_i32 s25, s24, s19 // (op 4481) 


s_and_b32 s19, s0, s25 // (op 4481) 


s_add_i32 s0, s1, s19 // (op 4481) 


s_ashr_i32 s1, s0, s18 // (op 4481) 


s_lshl_b32 s0, s45, 24 // (op 4481) 
s_ashr_i32 s18, s0, 31 // (op 4481) 

s_xor_b32 s0, s1, s18 // (op 4481) 

s_sub_i32 s1, s0, s18 // (op 4481) 


s_mul_i32 s0, s11, s22 // (op 4481) 

s_mul_i32 s18, s8, s0 // (op 4481) 

s_mul_i32 s0, s1, s18 // (op 4481) 

s_sub_u32 s18, s16, s0 // (op 4481) 

s_mul_hi_i32 s0, s18, s12 // (op 4481) 
s_add_i32 s19, s0, s18 // (op 4481) 


s_ashr_i32 s0, s19, 31 // (op 4481) 
s_cmp_eq_i32 s12, 0 // (op 4481) 
s_mov_b32 s18, scc // (op 4481) 
s_cmp_eq_i32 s18, 0 // (op 4481) 
s_cselect_b32 s22, 0, -1 // (op 4481) 

s_and_b32 s18, s44, 31 // (op 4481) 
s_lshl_b32 s23, 1, s18 // (op 4481) 
s_add_i32 s24, s23, s22 // (op 4481) 


s_and_b32 s25, s0, s24 // (op 4481) 


s_add_i32 s0, s19, s25 // (op 4481) 


s_ashr_i32 s19, s0, s18 // (op 4481) 


s_lshl_b32 s0, s44, 24 // (op 4481) 
s_ashr_i32 s18, s0, 31 // (op 4481) 

s_xor_b32 s0, s19, s18 // (op 4481) 

s_sub_i32 s19, s0, s18 // (op 4481) 


s_cmp_eq_i32 s1, 0 // (op 4481) 
s_mov_b32 s0, scc // (op 4481) 

s_cmp_eq_i32 s0, 0 // (op 4481) 
s_cselect_b32 s1, s19, s47 // (op 4481) 



v_lshl_add_u32 v15, s1, 6, v14 // (op 4481) 


v_mul_lo_u32 v14, v15, s38 // (op 4481) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 4481) 
s_lshr_b32 s47, s0, 1 // (op 4481) 

v_lshl_add_u32 v15, s47, 3, v20 // (op 4481) 


v_add_u32 v20, v15, v14 // (op 4481) 


s_mul_i32 s1, s38, s7 // (op 4483) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 4483) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 4483) least significant
s_mul_hi_u32 s18, s38, s6 // (op 4483) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4483) most significant: sum
s_add_u32 s1, s1, s18 // (op 4483) most significant: sum


s_mov_b32 s68, s40 // (op 4483) reg expression
s_mov_b32 s69, s41 // (op 4483) reg expression
s_mov_b32 s70, s0 // (op 4483) reg expression
s_mov_b32 s71, 131072 // (op 4483) call()

v_and_b32 v14, -2, v1 // (op 4579) 
v_sub_u32 v15, v1, v14 // (op 4579) 

v_lshlrev_b32 v14, 6, v15 // (op 4579) 

v_lshrrev_b32 v15, 1, v1 // (op 4579) 
v_lshl_add_u32 v21, v15, 7, v14 // (op 4579) 


v_mul_lo_u32 v14, v21, 6 // (op 4579) 

v_lshrrev_b32 v15, 3, v14 // (op 4579) 

v_and_b32 v14, -4, v1 // (op 4637) 
v_sub_u32 v21, v1, v14 // (op 4637) 

v_lshlrev_b32 v14, 5, v21 // (op 4637) 

v_lshrrev_b32 v21, 2, v1 // (op 4637) 
v_lshl_add_u32 v22, v21, 7, v14 // (op 4637) 


v_mul_lo_u32 v14, v22, 6 // (op 4637) 

v_lshrrev_b32 v21, 3, v14 // (op 4637) 

v_and_b32 v14, -4, v1 // (op 4695) 
v_sub_u32 v22, v1, v14 // (op 4695) 

v_lshrrev_b32 v14, 2, v1 // (op 4695) 
v_lshl_add_u32 v23, v14, 3, v22 // (op 4695) 


v_and_b32 v14, -4, v1 // (op 4753) 
v_sub_u32 v22, v1, v14 // (op 4753) 

v_and_b32 v14, 4294967292, v1 // (op 4753) 
v_add_u32 v24, v14, v22 // (op 4753) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 1370) 
s_and_b32 s47, s0, 4294967294 // (op 1370) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 1369) 
s_mov_b32 s18, s0 // (op 1369) convert

s_cmp_lt_u32 s47, s18 // (op 1369) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_1369 // (op 1369) Condition: Top (jump to RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_1369 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_1369:


s_add_u32 s47, s47, 1 // (op 1371) 
v_mov_b32 v14, v16 // (op 180) 

buffer_load_dwordx4 v[26:29], v14, s[48:51], 0 offen // (op 180) Load value
buffer_load_dwordx2 v[30:31], v14, s[48:51], 0 offen offset:16 // (op 180) Load value
buffer_load_dwordx4 v[32:35], v14, s[48:51], 0 offen offset:24 // (op 180) Load value
buffer_load_dwordx2 v[36:37], v14, s[48:51], 0 offen offset:40 // (op 180) Load value

v_add_u32 v16, 96, v16 // (op 4269) 


s_barrier  // (op 4994) 
v_mov_b32 v14, v17 // (op 194) 




buffer_load_dwordx4 v[38:41], v14, s[60:63], 0 offen // (op 194) Load value
buffer_load_dwordx2 v[42:43], v14, s[60:63], 0 offen offset:16 // (op 194) Load value

v_add_u32 v17, 96, v17 // (op 4345) 
s_waitcnt vmcnt(5)
ds_write_b128 v15, v[26:29] offset:6400 // (op 181) Store local data 

v_mov_b64 v[44:45], s[34:35] // (op 195) convert

v_mov_b32 v14, v19 // (op 195) 
v_mov_b32 v18, v14 // (op 195) 
s_waitcnt vmcnt(4)
ds_write_b64 v15, v[30:31] offset:6416 // (op 181) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v15, v[32:35] offset:6424 // (op 181) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v15, v[36:37] offset:6440 // (op 181) Store local data 
buffer_load_ubyte v46, v18, s[64:67], 0 offen offset: 0 // (op 195) Load value

v_add_u32 v18, v18, v44 // (op 195) 
s_barrier  // (op 5007) 
buffer_load_ubyte v47, v18, s[64:67], 0 offen offset: 0 // (op 195) Load value






v_add_u32 v19, 4, v19 // (op 4409) 
s_waitcnt vmcnt(3)
ds_write_b128 v21, v[38:41] // (op 191) Store local data 


v_mov_b32 v14, v20 // (op 196) 

buffer_load_ubyte v18, v14, s[68:71], 0 offen offset: 0 // (op 196) Load value

v_add_u32 v20, 4, v20 // (op 4473) 
s_waitcnt vmcnt(3)
ds_write_b64 v21, v[42:43] offset:16 // (op 191) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 4997) 



s_barrier  // (op 5021) 
ds_read_b128 v[40:43], v3 offset:6400 // (op 183) Load local data 



ds_read_b64 v[44:45], v3 offset:6416 // (op 183) Load local data 
s_waitcnt vmcnt(2)
ds_write_b8 v23, v46 offset:18688 // (op 192) Store local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 5011) 
s_waitcnt vmcnt(1)
ds_write_b8 v23, v47 offset:18692 // (op 192) Store local data 




ds_read_b128 v[34:37], v4 // (op 185) Load local data 
s_barrier  // (op 5035) 
ds_read_b64 v[38:39], v4 offset:16 // (op 185) Load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 5025) 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v18 offset:6144 // (op 193) Store local data 




ds_read_u8 v14, v0 offset:18688 // (op 187) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 5039) 



ds_read_u8 v18, v5 offset:6144 // (op 189) Load local data 



ds_read_b128 v[28:31], v4 offset:3072 // (op 354) Load local data 
ds_read_b64 v[32:33], v4 offset:3088 // (op 354) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:45], v[34:39], a[0:3], v14, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 190) 



ds_read_u8 v22, v5 offset:6272 // (op 358) Load local data 



ds_read_b128 v[46:49], v3 offset:9472 // (op 572) Load local data 
ds_read_b64 v[50:51], v3 offset:9488 // (op 572) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[40:45], v[28:33], a[4:7], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 359) 





ds_read_u8 v14, v0 offset:18816 // (op 575) Load local data 



ds_read_b128 v[40:43], v3 offset:12544 // (op 958) Load local data 
ds_read_b64 v[44:45], v3 offset:12560 // (op 958) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[46:51], v[34:39], a[8:11], v14, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 577) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[46:51], v[28:33], a[12:15], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 608) 
ds_read_u8 v25, v0 offset:18944 // (op 961) Load local data 





ds_read_b128 v[46:49], v3 offset:15616 // (op 1344) Load local data 
ds_read_b64 v[50:51], v3 offset:15632 // (op 1344) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[40:45], v[34:39], a[16:19], v25, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 963) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[40:45], v[28:33], a[20:23], v25, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 994) 
ds_read_u8 v14, v0 offset:19072 // (op 1347) Load local data 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[46:51], v[34:39], a[24:27], v14, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1349) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[46:51], v[28:33], a[28:31], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1380) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 1369) 
s_mov_b32 s18, s0 // (op 1369) convert

s_cmp_lt_u32 s47, s18 // (op 1369) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_1369 // (op 1369) Condition: Bottom (jump to RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_1369 if true)
RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_1369:






















s_barrier  // (op 5149) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_135
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 

v_cvt_f16_f32 v6, v14 // (op 56) 
v_cvt_f16_f32 v7, v15 // (op 56) 
v_cvt_f16_f32 v8, v16 // (op 56) 
v_cvt_f16_f32 v9, v17 // (op 56) 


v_pack_b32_f16 v4, v6, v7 // (op 58) 
v_pack_b32_f16 v5, v8, v9 // (op 58) 

v_mov_b32 v0, v13 // (op 58) reg expression

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 58) Store value



v_accvgpr_read v6, a4 // (op 1873) 
v_accvgpr_read v7, a5 // (op 1873) 
v_accvgpr_read v8, a6 // (op 1873) 
v_accvgpr_read v9, a7 // (op 1873) 
v_mul_f32 v14, v11, v6 // (op 1873) 
v_mul_f32 v15, v11, v7 // (op 1873) 
v_mul_f32 v16, v11, v8 // (op 1873) 
v_mul_f32 v17, v11, v9 // (op 1873) 

v_cvt_f16_f32 v6, v14 // (op 1875) 
v_cvt_f16_f32 v7, v15 // (op 1875) 
v_cvt_f16_f32 v8, v16 // (op 1875) 
v_cvt_f16_f32 v9, v17 // (op 1875) 


v_pack_b32_f16 v4, v6, v7 // (op 1876) 
v_pack_b32_f16 v5, v8, v9 // (op 1876) 

s_lshl_b32 s0, s4, 6 // (op 1876) 
v_add_u32 v0, v13, s0 // (op 1876) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 1876) Store value



v_accvgpr_read v6, a8 // (op 1899) 
v_accvgpr_read v7, a9 // (op 1899) 
v_accvgpr_read v8, a10 // (op 1899) 
v_accvgpr_read v9, a11 // (op 1899) 
v_mul_f32 v14, v11, v6 // (op 1899) 
v_mul_f32 v15, v11, v7 // (op 1899) 
v_mul_f32 v16, v11, v8 // (op 1899) 
v_mul_f32 v17, v11, v9 // (op 1899) 

v_cvt_f16_f32 v6, v14 // (op 1901) 
v_cvt_f16_f32 v7, v15 // (op 1901) 
v_cvt_f16_f32 v8, v16 // (op 1901) 
v_cvt_f16_f32 v9, v17 // (op 1901) 


v_pack_b32_f16 v4, v6, v7 // (op 1903) 
v_pack_b32_f16 v5, v8, v9 // (op 1903) 

v_add_u32 v0, 64, v13 // (op 1903) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 1903) Store value



v_accvgpr_read v6, a12 // (op 1909) 
v_accvgpr_read v7, a13 // (op 1909) 
v_accvgpr_read v8, a14 // (op 1909) 
v_accvgpr_read v9, a15 // (op 1909) 
v_mul_f32 v14, v11, v6 // (op 1909) 
v_mul_f32 v15, v11, v7 // (op 1909) 
v_mul_f32 v16, v11, v8 // (op 1909) 
v_mul_f32 v17, v11, v9 // (op 1909) 

v_cvt_f16_f32 v6, v14 // (op 1911) 
v_cvt_f16_f32 v7, v15 // (op 1911) 
v_cvt_f16_f32 v8, v16 // (op 1911) 
v_cvt_f16_f32 v9, v17 // (op 1911) 


v_pack_b32_f16 v4, v6, v7 // (op 1905) 
v_pack_b32_f16 v5, v8, v9 // (op 1905) 

s_lshl_b32 s0, s4, 6 // (op 1905) 
s_add_u32 s0, s0, 64 // (op 1905) 
v_add_u32 v0, v13, s0 // (op 1905) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 1905) Store value



v_accvgpr_read v6, a16 // (op 1939) 
v_accvgpr_read v7, a17 // (op 1939) 
v_accvgpr_read v8, a18 // (op 1939) 
v_accvgpr_read v9, a19 // (op 1939) 
v_mul_f32 v14, v11, v6 // (op 1939) 
v_mul_f32 v15, v11, v7 // (op 1939) 
v_mul_f32 v16, v11, v8 // (op 1939) 
v_mul_f32 v17, v11, v9 // (op 1939) 

v_cvt_f16_f32 v6, v14 // (op 1941) 
v_cvt_f16_f32 v7, v15 // (op 1941) 
v_cvt_f16_f32 v8, v16 // (op 1941) 
v_cvt_f16_f32 v9, v17 // (op 1941) 


v_pack_b32_f16 v4, v6, v7 // (op 1943) 
v_pack_b32_f16 v5, v8, v9 // (op 1943) 

v_add_u32 v0, 128, v13 // (op 1943) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 1943) Store value



v_accvgpr_read v6, a20 // (op 1949) 
v_accvgpr_read v7, a21 // (op 1949) 
v_accvgpr_read v8, a22 // (op 1949) 
v_accvgpr_read v9, a23 // (op 1949) 
v_mul_f32 v14, v11, v6 // (op 1949) 
v_mul_f32 v15, v11, v7 // (op 1949) 
v_mul_f32 v16, v11, v8 // (op 1949) 
v_mul_f32 v17, v11, v9 // (op 1949) 

v_cvt_f16_f32 v6, v14 // (op 1951) 
v_cvt_f16_f32 v7, v15 // (op 1951) 
v_cvt_f16_f32 v8, v16 // (op 1951) 
v_cvt_f16_f32 v9, v17 // (op 1951) 


v_pack_b32_f16 v4, v6, v7 // (op 1945) 
v_pack_b32_f16 v5, v8, v9 // (op 1945) 

s_lshl_b32 s0, s4, 6 // (op 1945) 
s_add_u32 s0, s0, 128 // (op 1945) 
v_add_u32 v0, v13, s0 // (op 1945) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 1945) Store value



v_accvgpr_read v6, a24 // (op 1979) 
v_accvgpr_read v7, a25 // (op 1979) 
v_accvgpr_read v8, a26 // (op 1979) 
v_accvgpr_read v9, a27 // (op 1979) 
v_mul_f32 v14, v11, v6 // (op 1979) 
v_mul_f32 v15, v11, v7 // (op 1979) 
v_mul_f32 v16, v11, v8 // (op 1979) 
v_mul_f32 v17, v11, v9 // (op 1979) 

v_cvt_f16_f32 v6, v14 // (op 1981) 
v_cvt_f16_f32 v7, v15 // (op 1981) 
v_cvt_f16_f32 v8, v16 // (op 1981) 
v_cvt_f16_f32 v9, v17 // (op 1981) 


v_pack_b32_f16 v4, v6, v7 // (op 1983) 
v_pack_b32_f16 v5, v8, v9 // (op 1983) 

v_add_u32 v0, 192, v13 // (op 1983) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 1983) Store value



v_accvgpr_read v6, a28 // (op 1989) 
v_accvgpr_read v7, a29 // (op 1989) 
v_accvgpr_read v8, a30 // (op 1989) 
v_accvgpr_read v9, a31 // (op 1989) 
v_mul_f32 v14, v11, v6 // (op 1989) 
v_mul_f32 v15, v11, v7 // (op 1989) 
v_mul_f32 v16, v11, v8 // (op 1989) 
v_mul_f32 v17, v11, v9 // (op 1989) 

v_cvt_f16_f32 v6, v14 // (op 1991) 
v_cvt_f16_f32 v7, v15 // (op 1991) 
v_cvt_f16_f32 v8, v16 // (op 1991) 
v_cvt_f16_f32 v9, v17 // (op 1991) 


v_pack_b32_f16 v4, v6, v7 // (op 1985) 
v_pack_b32_f16 v5, v8, v9 // (op 1985) 

s_lshl_b32 s0, s4, 6 // (op 1985) 
s_add_u32 s0, s0, 192 // (op 1985) 
v_add_u32 v0, v13, s0 // (op 1985) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 1985) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_135
RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_135:


v_add_u32 v0, 128, v12 // (op 2136) 

buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 2136) Load value

v_accvgpr_read v4, a16 // (op 2127) 
v_accvgpr_read v5, a17 // (op 2127) 
v_accvgpr_read v6, a18 // (op 2127) 
v_accvgpr_read v7, a19 // (op 2127) 
v_mul_f32 v14, v11, v4 // (op 2127) 
v_mul_f32 v15, v11, v5 // (op 2127) 
v_mul_f32 v16, v11, v6 // (op 2127) 
v_mul_f32 v17, v11, v7 // (op 2127) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2135) 
v_lshrrev_b32 v7, 16, v8 // (op 2135) 
v_cvt_f32_f16 v7, v7 // (op 2135) 
v_mul_f32 v18, v10, v6 // (op 2135) 
v_mul_f32 v19, v10, v7 // (op 2135) 
v_cvt_f32_f16 v6, v9 // (op 2135) 
v_lshrrev_b32 v7, 16, v9 // (op 2135) 
v_cvt_f32_f16 v7, v7 // (op 2135) 
v_mul_f32 v20, v10, v6 // (op 2135) 
v_mul_f32 v21, v10, v7 // (op 2135) 


v_add_f32 v6, v18, v14 // (op 2128) 
v_add_f32 v7, v19, v15 // (op 2128) 
v_add_f32 v8, v20, v16 // (op 2128) 
v_add_f32 v9, v21, v17 // (op 2128) 


v_cvt_f16_f32 v14, v6 // (op 2129) 
v_cvt_f16_f32 v15, v7 // (op 2129) 
v_cvt_f16_f32 v16, v8 // (op 2129) 
v_cvt_f16_f32 v17, v9 // (op 2129) 


v_pack_b32_f16 v8, v14, v15 // (op 2131) 
v_pack_b32_f16 v9, v16, v17 // (op 2131) 

v_add_u32 v0, 128, v13 // (op 2131) 

buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2131) Store value





s_lshl_b32 s0, s2, 6 // (op 2141) 
s_add_u32 s0, s0, 128 // (op 2141) 
v_add_u32 v0, v12, s0 // (op 2141) 


buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 2141) Load value

v_accvgpr_read v4, a20 // (op 2143) 
v_accvgpr_read v5, a21 // (op 2143) 
v_accvgpr_read v6, a22 // (op 2143) 
v_accvgpr_read v7, a23 // (op 2143) 
v_mul_f32 v14, v11, v4 // (op 2143) 
v_mul_f32 v15, v11, v5 // (op 2143) 
v_mul_f32 v16, v11, v6 // (op 2143) 
v_mul_f32 v17, v11, v7 // (op 2143) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2138) 
v_lshrrev_b32 v7, 16, v8 // (op 2138) 
v_cvt_f32_f16 v7, v7 // (op 2138) 
v_mul_f32 v18, v10, v6 // (op 2138) 
v_mul_f32 v19, v10, v7 // (op 2138) 
v_cvt_f32_f16 v6, v9 // (op 2138) 
v_lshrrev_b32 v7, 16, v9 // (op 2138) 
v_cvt_f32_f16 v7, v7 // (op 2138) 
v_mul_f32 v20, v10, v6 // (op 2138) 
v_mul_f32 v21, v10, v7 // (op 2138) 


v_add_f32 v6, v18, v14 // (op 2139) 
v_add_f32 v7, v19, v15 // (op 2139) 
v_add_f32 v8, v20, v16 // (op 2139) 
v_add_f32 v9, v21, v17 // (op 2139) 


v_cvt_f16_f32 v14, v6 // (op 2140) 
v_cvt_f16_f32 v15, v7 // (op 2140) 
v_cvt_f16_f32 v16, v8 // (op 2140) 
v_cvt_f16_f32 v17, v9 // (op 2140) 


v_pack_b32_f16 v8, v14, v15 // (op 2133) 
v_pack_b32_f16 v9, v16, v17 // (op 2133) 

s_lshl_b32 s0, s4, 6 // (op 2133) 
s_add_u32 s0, s0, 128 // (op 2133) 
v_add_u32 v0, v13, s0 // (op 2133) 


buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2133) Store value





v_add_u32 v0, 64, v12 // (op 2091) 

buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 2091) Load value

v_accvgpr_read v4, a8 // (op 2082) 
v_accvgpr_read v5, a9 // (op 2082) 
v_accvgpr_read v6, a10 // (op 2082) 
v_accvgpr_read v7, a11 // (op 2082) 
v_mul_f32 v14, v11, v4 // (op 2082) 
v_mul_f32 v15, v11, v5 // (op 2082) 
v_mul_f32 v16, v11, v6 // (op 2082) 
v_mul_f32 v17, v11, v7 // (op 2082) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2090) 
v_lshrrev_b32 v7, 16, v8 // (op 2090) 
v_cvt_f32_f16 v7, v7 // (op 2090) 
v_mul_f32 v18, v10, v6 // (op 2090) 
v_mul_f32 v19, v10, v7 // (op 2090) 
v_cvt_f32_f16 v6, v9 // (op 2090) 
v_lshrrev_b32 v7, 16, v9 // (op 2090) 
v_cvt_f32_f16 v7, v7 // (op 2090) 
v_mul_f32 v20, v10, v6 // (op 2090) 
v_mul_f32 v21, v10, v7 // (op 2090) 


v_add_f32 v6, v18, v14 // (op 2083) 
v_add_f32 v7, v19, v15 // (op 2083) 
v_add_f32 v8, v20, v16 // (op 2083) 
v_add_f32 v9, v21, v17 // (op 2083) 


v_cvt_f16_f32 v14, v6 // (op 2084) 
v_cvt_f16_f32 v15, v7 // (op 2084) 
v_cvt_f16_f32 v16, v8 // (op 2084) 
v_cvt_f16_f32 v17, v9 // (op 2084) 


v_pack_b32_f16 v8, v14, v15 // (op 2086) 
v_pack_b32_f16 v9, v16, v17 // (op 2086) 

v_add_u32 v0, 64, v13 // (op 2086) 

buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2086) Store value





s_lshl_b32 s0, s2, 6 // (op 2096) 
s_add_u32 s0, s0, 64 // (op 2096) 
v_add_u32 v0, v12, s0 // (op 2096) 


buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 2096) Load value

v_accvgpr_read v4, a12 // (op 2098) 
v_accvgpr_read v5, a13 // (op 2098) 
v_accvgpr_read v6, a14 // (op 2098) 
v_accvgpr_read v7, a15 // (op 2098) 
v_mul_f32 v14, v11, v4 // (op 2098) 
v_mul_f32 v15, v11, v5 // (op 2098) 
v_mul_f32 v16, v11, v6 // (op 2098) 
v_mul_f32 v17, v11, v7 // (op 2098) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2093) 
v_lshrrev_b32 v7, 16, v8 // (op 2093) 
v_cvt_f32_f16 v7, v7 // (op 2093) 
v_mul_f32 v18, v10, v6 // (op 2093) 
v_mul_f32 v19, v10, v7 // (op 2093) 
v_cvt_f32_f16 v6, v9 // (op 2093) 
v_lshrrev_b32 v7, 16, v9 // (op 2093) 
v_cvt_f32_f16 v7, v7 // (op 2093) 
v_mul_f32 v20, v10, v6 // (op 2093) 
v_mul_f32 v21, v10, v7 // (op 2093) 


v_add_f32 v6, v18, v14 // (op 2094) 
v_add_f32 v7, v19, v15 // (op 2094) 
v_add_f32 v8, v20, v16 // (op 2094) 
v_add_f32 v9, v21, v17 // (op 2094) 


v_cvt_f16_f32 v14, v6 // (op 2095) 
v_cvt_f16_f32 v15, v7 // (op 2095) 
v_cvt_f16_f32 v16, v8 // (op 2095) 
v_cvt_f16_f32 v17, v9 // (op 2095) 


v_pack_b32_f16 v8, v14, v15 // (op 2088) 
v_pack_b32_f16 v9, v16, v17 // (op 2088) 

s_lshl_b32 s0, s4, 6 // (op 2088) 
s_add_u32 s0, s0, 64 // (op 2088) 
v_add_u32 v0, v13, s0 // (op 2088) 


buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2088) Store value





v_add_u32 v0, 192, v12 // (op 2181) 

buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 2181) Load value

v_accvgpr_read v4, a24 // (op 2172) 
v_accvgpr_read v5, a25 // (op 2172) 
v_accvgpr_read v6, a26 // (op 2172) 
v_accvgpr_read v7, a27 // (op 2172) 
v_mul_f32 v14, v11, v4 // (op 2172) 
v_mul_f32 v15, v11, v5 // (op 2172) 
v_mul_f32 v16, v11, v6 // (op 2172) 
v_mul_f32 v17, v11, v7 // (op 2172) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2180) 
v_lshrrev_b32 v7, 16, v8 // (op 2180) 
v_cvt_f32_f16 v7, v7 // (op 2180) 
v_mul_f32 v18, v10, v6 // (op 2180) 
v_mul_f32 v19, v10, v7 // (op 2180) 
v_cvt_f32_f16 v6, v9 // (op 2180) 
v_lshrrev_b32 v7, 16, v9 // (op 2180) 
v_cvt_f32_f16 v7, v7 // (op 2180) 
v_mul_f32 v20, v10, v6 // (op 2180) 
v_mul_f32 v21, v10, v7 // (op 2180) 


v_add_f32 v6, v18, v14 // (op 2173) 
v_add_f32 v7, v19, v15 // (op 2173) 
v_add_f32 v8, v20, v16 // (op 2173) 
v_add_f32 v9, v21, v17 // (op 2173) 


v_cvt_f16_f32 v14, v6 // (op 2174) 
v_cvt_f16_f32 v15, v7 // (op 2174) 
v_cvt_f16_f32 v16, v8 // (op 2174) 
v_cvt_f16_f32 v17, v9 // (op 2174) 


v_pack_b32_f16 v8, v14, v15 // (op 2176) 
v_pack_b32_f16 v9, v16, v17 // (op 2176) 

v_add_u32 v0, 192, v13 // (op 2176) 

buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2176) Store value





s_lshl_b32 s0, s2, 6 // (op 2186) 
s_add_u32 s0, s0, 192 // (op 2186) 
v_add_u32 v0, v12, s0 // (op 2186) 


buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 2186) Load value

v_accvgpr_read v4, a28 // (op 2188) 
v_accvgpr_read v5, a29 // (op 2188) 
v_accvgpr_read v6, a30 // (op 2188) 
v_accvgpr_read v7, a31 // (op 2188) 
v_mul_f32 v14, v11, v4 // (op 2188) 
v_mul_f32 v15, v11, v5 // (op 2188) 
v_mul_f32 v16, v11, v6 // (op 2188) 
v_mul_f32 v17, v11, v7 // (op 2188) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2183) 
v_lshrrev_b32 v7, 16, v8 // (op 2183) 
v_cvt_f32_f16 v7, v7 // (op 2183) 
v_mul_f32 v18, v10, v6 // (op 2183) 
v_mul_f32 v19, v10, v7 // (op 2183) 
v_cvt_f32_f16 v6, v9 // (op 2183) 
v_lshrrev_b32 v7, 16, v9 // (op 2183) 
v_cvt_f32_f16 v7, v7 // (op 2183) 
v_mul_f32 v20, v10, v6 // (op 2183) 
v_mul_f32 v21, v10, v7 // (op 2183) 


v_add_f32 v6, v18, v14 // (op 2184) 
v_add_f32 v7, v19, v15 // (op 2184) 
v_add_f32 v8, v20, v16 // (op 2184) 
v_add_f32 v9, v21, v17 // (op 2184) 


v_cvt_f16_f32 v14, v6 // (op 2185) 
v_cvt_f16_f32 v15, v7 // (op 2185) 
v_cvt_f16_f32 v16, v8 // (op 2185) 
v_cvt_f16_f32 v17, v9 // (op 2185) 


v_pack_b32_f16 v8, v14, v15 // (op 2178) 
v_pack_b32_f16 v9, v16, v17 // (op 2178) 

s_lshl_b32 s0, s4, 6 // (op 2178) 
s_add_u32 s0, s0, 192 // (op 2178) 
v_add_u32 v0, v13, s0 // (op 2178) 


buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2178) Store value





v_mov_b32 v0, v12 // (op 148) reg expression

buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 148) Load value

v_accvgpr_read v4, a0 // (op 144) 
v_accvgpr_read v5, a1 // (op 144) 
v_accvgpr_read v6, a2 // (op 144) 
v_accvgpr_read v7, a3 // (op 144) 
v_mul_f32 v14, v11, v4 // (op 144) 
v_mul_f32 v15, v11, v5 // (op 144) 
v_mul_f32 v16, v11, v6 // (op 144) 
v_mul_f32 v17, v11, v7 // (op 144) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 149) 
v_lshrrev_b32 v7, 16, v8 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v18, v10, v6 // (op 149) 
v_mul_f32 v19, v10, v7 // (op 149) 
v_cvt_f32_f16 v6, v9 // (op 149) 
v_lshrrev_b32 v7, 16, v9 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v20, v10, v6 // (op 149) 
v_mul_f32 v21, v10, v7 // (op 149) 


v_add_f32 v6, v18, v14 // (op 145) 
v_add_f32 v7, v19, v15 // (op 145) 
v_add_f32 v8, v20, v16 // (op 145) 
v_add_f32 v9, v21, v17 // (op 145) 


v_cvt_f16_f32 v14, v6 // (op 146) 
v_cvt_f16_f32 v15, v7 // (op 146) 
v_cvt_f16_f32 v16, v8 // (op 146) 
v_cvt_f16_f32 v17, v9 // (op 146) 


v_pack_b32_f16 v8, v14, v15 // (op 147) 
v_pack_b32_f16 v9, v16, v17 // (op 147) 

v_mov_b32 v0, v13 // (op 147) reg expression

buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 147) Store value





s_lshl_b32 s0, s2, 6 // (op 2051) 
v_add_u32 v0, v12, s0 // (op 2051) 


buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 2051) Load value

v_accvgpr_read v4, a4 // (op 2047) 
v_accvgpr_read v5, a5 // (op 2047) 
v_accvgpr_read v6, a6 // (op 2047) 
v_accvgpr_read v7, a7 // (op 2047) 
v_mul_f32 v14, v11, v4 // (op 2047) 
v_mul_f32 v15, v11, v5 // (op 2047) 
v_mul_f32 v16, v11, v6 // (op 2047) 
v_mul_f32 v17, v11, v7 // (op 2047) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2052) 
v_lshrrev_b32 v7, 16, v8 // (op 2052) 
v_cvt_f32_f16 v7, v7 // (op 2052) 
v_mul_f32 v18, v10, v6 // (op 2052) 
v_mul_f32 v19, v10, v7 // (op 2052) 
v_cvt_f32_f16 v6, v9 // (op 2052) 
v_lshrrev_b32 v7, 16, v9 // (op 2052) 
v_cvt_f32_f16 v7, v7 // (op 2052) 
v_mul_f32 v20, v10, v6 // (op 2052) 
v_mul_f32 v21, v10, v7 // (op 2052) 


v_add_f32 v6, v18, v14 // (op 2048) 
v_add_f32 v7, v19, v15 // (op 2048) 
v_add_f32 v8, v20, v16 // (op 2048) 
v_add_f32 v9, v21, v17 // (op 2048) 


v_cvt_f16_f32 v14, v6 // (op 2049) 
v_cvt_f16_f32 v15, v7 // (op 2049) 
v_cvt_f16_f32 v16, v8 // (op 2049) 
v_cvt_f16_f32 v17, v9 // (op 2049) 


v_pack_b32_f16 v8, v14, v15 // (op 2050) 
v_pack_b32_f16 v9, v16, v17 // (op 2050) 

s_lshl_b32 s0, s4, 6 // (op 2050) 
v_add_u32 v0, v13, s0 // (op 2050) 


buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2050) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_135:
















s_endpgm  // End of RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC
.LRR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_end:
.size RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC, .LRR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_end-RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC
  .amdhsa_next_free_vgpr 132
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 38400 // lds bytes
  .amdhsa_accum_offset 100
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
    DataType: F6
    DataTypeA: F6
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
    macrotile: [128, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC
    .symbol: RR_GEMM_TN_FP6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 38400
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 100
    .agpr_count: 32
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Tensor_2_size_1_4
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
      - .name: Tensor_0_size_0_1
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

