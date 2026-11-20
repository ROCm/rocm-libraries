// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC
.p2align 8
.type RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC,@function
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 8854) 
s_mul_hi_i32 s1, s16, s10 // (op 9111) 
v_lshrrev_b32 v0, 7, v1 // (op 8916) 
v_and_b32 v3, 15, v1 // (op 8916) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 8916) 


v_and_b32 v0, 48, v1 // (op 8916) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 8916) 


v_lshrrev_b32 v0, 6, v1 // (op 8974) 
v_and_b32 v4, 1, v0 // (op 8974) 

v_and_b32 v0, 15, v1 // (op 8974) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 8974) 


v_and_b32 v0, 63, v1 // (op 8974) 
v_lshrrev_b32 v4, 4, v0 // (op 8974) 

v_lshlrev_b32 v0, 5, v4 // (op 8974) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 8974) 


v_lshlrev_b32 v0, 2, v4 // (op 8974) 

v_lshrrev_b32 v4, 3, v0 // (op 8974) 

v_lshrrev_b32 v0, 7, v1 // (op 9032) 
v_and_b32 v5, 15, v1 // (op 9032) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 9032) 


v_and_b32 v0, 63, v1 // (op 9032) 
v_lshrrev_b32 v5, 4, v0 // (op 9032) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 9032) 


v_lshrrev_b32 v5, 6, v1 // (op 9090) 
v_and_b32 v6, 1, v5 // (op 9090) 

v_and_b32 v5, 15, v1 // (op 9090) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 9090) 


v_and_b32 v5, 63, v1 // (op 9090) 
v_lshrrev_b32 v6, 4, v5 // (op 9090) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 9090) 


v_lshrrev_b32 v6, 7, v1 // (op 9142) 
v_and_b32 v7, 15, v1 // (op 9142) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 9142) 


v_and_b32 v6, 48, v1 // (op 9142) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 9142) 


v_lshrrev_b32 v6, 6, v1 // (op 9171) 
v_and_b32 v8, 1, v6 // (op 9171) 

v_and_b32 v6, 15, v1 // (op 9171) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 9171) 


v_and_b32 v6, 63, v1 // (op 9171) 
v_lshrrev_b32 v8, 4, v6 // (op 9171) 

v_lshlrev_b32 v6, 5, v8 // (op 9171) 

v_lshl_add_u32 v8, v9, 7, v6 // (op 9171) 


v_lshlrev_b32 v6, 2, v8 // (op 9171) 

v_lshrrev_b32 v8, 3, v6 // (op 9171) 

v_lshrrev_b32 v6, 7, v1 // (op 9200) 
v_and_b32 v9, 15, v1 // (op 9200) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 9200) 


v_and_b32 v6, 63, v1 // (op 9200) 
v_lshrrev_b32 v9, 4, v6 // (op 9200) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 9200) 


v_lshrrev_b32 v9, 6, v1 // (op 9229) 
v_and_b32 v10, 1, v9 // (op 9229) 

v_and_b32 v9, 15, v1 // (op 9229) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 9229) 


v_and_b32 v9, 63, v1 // (op 9229) 
v_lshrrev_b32 v10, 4, v9 // (op 9229) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 9229) 


v_lshrrev_b32 v10, 7, v1 // (op 8854) 
s_add_i32 s18, s0, s16 // (op 8854) 

v_and_b32 v11, 63, v1 // (op 8854) 
s_ashr_i32 s0, s18, 31 // (op 8854) 
v_lshrrev_b32 v12, 4, v11 // (op 8854) 
s_cmp_eq_i32 s10, 0 // (op 8854) 
s_mov_b32 s19, scc // (op 8854) 
v_lshlrev_b32 v13, 2, v12 // (op 8854) 

s_cmp_eq_i32 s19, 0 // (op 8854) 
s_cselect_b32 s49, 0, -1 // (op 8854) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 8854) 


s_and_b32 s19, s13, 31 // (op 8854) 
v_lshrrev_b32 v10, 2, v11 // (op 8854) 

v_and_b32 v11, 3, v10 // (op 8854) 

v_lshrrev_b32 v10, 7, v1 // (op 9111) 
v_and_b32 v13, 63, v1 // (op 9111) 
s_lshl_b32 s50, 1, s19 // (op 8854) 
v_and_b32 v14, 3, v1 // (op 8854) 
s_add_i32 s51, s50, s49 // (op 8854) 


s_and_b32 s49, s0, s51 // (op 8854) 


s_add_i32 s0, s1, s16 // (op 9111) 

s_ashr_i32 s1, s0, 31 // (op 9111) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 8854) 


s_add_i32 s50, s18, s49 // (op 8854) 


v_lshrrev_b32 v11, 6, v1 // (op 8854) 
s_ashr_i32 s18, s50, s19 // (op 8854) 


v_and_b32 v14, 1, v11 // (op 8854) 

s_lshl_b32 s19, s13, 24 // (op 8854) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 8854) 


s_ashr_i32 s49, s19, 31 // (op 8854) 

s_xor_b32 s19, s18, s49 // (op 8854) 

s_sub_i32 s18, s19, s49 // (op 8854) 


s_mul_hi_i32 s19, s18, s14 // (op 8854) 
s_add_i32 s49, s19, s18 // (op 8854) 

s_ashr_i32 s19, s49, 31 // (op 8854) 
s_cmp_eq_i32 s14, 0 // (op 8854) 
s_mov_b32 s50, scc // (op 8854) 
s_cmp_eq_i32 s50, 0 // (op 8854) 
s_cselect_b32 s51, 0, -1 // (op 8854) 

s_and_b32 s50, s15, 31 // (op 8854) 
v_lshrrev_b32 v14, 4, v13 // (op 9111) 
s_cmp_eq_i32 s10, 0 // (op 9111) 
s_mov_b32 s52, scc // (op 9111) 
v_lshlrev_b32 v15, 2, v14 // (op 9111) 

s_cmp_eq_i32 s52, 0 // (op 9111) 
s_cselect_b32 s53, 0, -1 // (op 9111) 

s_lshl_b32 s52, 1, s50 // (op 8854) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 9111) 


v_lshrrev_b32 v10, 2, v13 // (op 9111) 

s_add_i32 s54, s52, s51 // (op 8854) 


s_and_b32 s51, s19, s54 // (op 8854) 


s_add_i32 s19, s49, s51 // (op 8854) 


s_ashr_i32 s49, s19, s50 // (op 8854) 


s_lshl_b32 s19, s15, 24 // (op 8854) 
s_ashr_i32 s50, s19, 31 // (op 8854) 

s_xor_b32 s19, s49, s50 // (op 8854) 

s_sub_i32 s49, s19, s50 // (op 8854) 


s_mul_i32 s19, s18, s11 // (op 8854) 
s_sub_u32 s50, s16, s19 // (op 8854) 

s_mul_i32 s19, s49, s11 // (op 8854) 
s_add_u32 s19, s19, s50 // (op 8854) 

s_mul_hi_i32 s50, s16, s9 // (op 8854) 
s_add_i32 s51, s50, s16 // (op 8854) 

s_ashr_i32 s50, s51, 31 // (op 8854) 
s_cmp_eq_i32 s9, 0 // (op 8854) 
s_mov_b32 s52, scc // (op 8854) 
s_cmp_eq_i32 s52, 0 // (op 8854) 
s_cselect_b32 s54, 0, -1 // (op 8854) 

s_and_b32 s52, s13, 31 // (op 9111) 
v_and_b32 v13, 3, v10 // (op 9111) 

s_lshl_b32 s55, 1, s52 // (op 9111) 
v_and_b32 v10, 3, v1 // (op 9111) 
s_add_i32 s56, s55, s53 // (op 9111) 


s_and_b32 s53, s1, s56 // (op 9111) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 9111) 


s_add_i32 s1, s0, s53 // (op 9111) 


v_lshrrev_b32 v10, 6, v1 // (op 9111) 
s_ashr_i32 s0, s1, s52 // (op 9111) 


v_and_b32 v13, 1, v10 // (op 9111) 

s_lshl_b32 s1, s13, 24 // (op 9111) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 9111) 


s_ashr_i32 s52, s1, 31 // (op 9111) 

s_xor_b32 s1, s0, s52 // (op 9111) 

s_sub_i32 s0, s1, s52 // (op 9111) 


s_mul_hi_i32 s1, s0, s14 // (op 9111) 
s_add_i32 s52, s1, s0 // (op 9111) 

s_ashr_i32 s1, s52, 31 // (op 9111) 
s_cmp_eq_i32 s14, 0 // (op 9111) 
s_mov_b32 s53, scc // (op 9111) 
s_cmp_eq_i32 s53, 0 // (op 9111) 
s_cselect_b32 s55, 0, -1 // (op 9111) 

s_and_b32 s53, s15, 31 // (op 9111) 
s_lshl_b32 s56, 1, s53 // (op 9111) 
s_add_i32 s57, s56, s55 // (op 9111) 


s_and_b32 s55, s1, s57 // (op 9111) 


s_add_i32 s1, s52, s55 // (op 9111) 


s_ashr_i32 s52, s1, s53 // (op 9111) 


s_lshl_b32 s1, s15, 24 // (op 9111) 
s_ashr_i32 s53, s1, 31 // (op 9111) 

s_xor_b32 s1, s52, s53 // (op 9111) 

s_sub_i32 s52, s1, s53 // (op 9111) 


s_mul_i32 s1, s0, s11 // (op 9111) 
s_sub_u32 s53, s16, s1 // (op 9111) 

s_mul_i32 s1, s52, s11 // (op 9111) 
s_add_u32 s1, s1, s53 // (op 9111) 

s_mul_hi_i32 s53, s16, s9 // (op 9111) 
s_add_i32 s55, s53, s16 // (op 9111) 

s_ashr_i32 s53, s55, 31 // (op 9111) 
s_cmp_eq_i32 s9, 0 // (op 9111) 
s_mov_b32 s56, scc // (op 9111) 
s_cmp_eq_i32 s56, 0 // (op 9111) 
s_cselect_b32 s57, 0, -1 // (op 9111) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 8854) 
s_lshl_b32 s58, 1, s56 // (op 8854) 
s_add_i32 s59, s58, s54 // (op 8854) 


s_and_b32 s54, s50, s59 // (op 8854) 


s_add_i32 s50, s51, s54 // (op 8854) 


s_ashr_i32 s51, s50, s56 // (op 8854) 


s_lshl_b32 s50, s45, 24 // (op 8854) 
s_ashr_i32 s54, s50, 31 // (op 8854) 

s_xor_b32 s50, s51, s54 // (op 8854) 

s_sub_i32 s51, s50, s54 // (op 8854) 


s_and_b32 s50, s45, 31 // (op 9111) 
s_lshl_b32 s54, 1, s50 // (op 9111) 
s_add_i32 s56, s54, s57 // (op 9111) 


s_and_b32 s54, s53, s56 // (op 9111) 


s_add_i32 s53, s55, s54 // (op 9111) 


s_ashr_i32 s54, s53, s50 // (op 9111) 


s_lshl_b32 s50, s45, 24 // (op 9111) 
s_ashr_i32 s53, s50, 31 // (op 9111) 

s_xor_b32 s50, s54, s53 // (op 9111) 

s_sub_i32 s54, s50, s53 // (op 9111) 


s_add_u32 s56, s6, 256 // (op 8854) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 8854) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 8854) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 8854) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 8854) 

s_mul_i32 s50, s11, s56 // (op 8854) 
s_mul_i32 s53, s8, s50 // (op 8854) 

s_mul_i32 s50, s51, s53 // (op 8854) 

s_sub_u32 s53, s16, s50 // (op 8854) 

s_mov_b32 s50, s53 // (op 8854) 
s_mul_hi_i32 s55, s50, s12 // (op 8854) 
s_add_i32 s58, s55, s50 // (op 8854) 


s_ashr_i32 s50, s58, 31 // (op 8854) 
s_cmp_eq_i32 s12, 0 // (op 8854) 
s_mov_b32 s55, scc // (op 8854) 
s_cmp_eq_i32 s55, 0 // (op 8854) 
s_cselect_b32 s59, 0, -1 // (op 8854) 

s_and_b32 s55, s44, 31 // (op 8854) 
s_lshl_b32 s60, 1, s55 // (op 8854) 
s_add_i32 s61, s60, s59 // (op 8854) 


s_and_b32 s59, s50, s61 // (op 8854) 


s_add_i32 s50, s58, s59 // (op 8854) 


s_ashr_i32 s58, s50, s55 // (op 8854) 


s_lshl_b32 s50, s44, 24 // (op 8854) 
s_ashr_i32 s55, s50, 31 // (op 8854) 

s_xor_b32 s50, s58, s55 // (op 8854) 

s_sub_i32 s58, s50, s55 // (op 8854) 


s_mul_i32 s50, s46, s11 // (op 8854) 
s_add_u32 s60, s20, 64 // (op 8854) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 8854) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 8854) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 8854) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 6 // (op 8854) 

s_sub_i32 s55, s60, s50 // (op 8854) 

s_mul_i32 s59, s58, s55 // (op 8854) 

s_sub_u32 s55, s53, s59 // (op 8854) 


s_add_u32 s53, s55, s50 // (op 8854) 


s_cmp_eq_i32 s51, 0 // (op 8854) 
s_mov_b32 s50, scc // (op 8854) 

s_cmp_eq_i32 s50, 0 // (op 8854) 
s_cselect_b32 s51, s53, s19 // (op 8854) 


s_mul_i32 s19, s49, s56 // (op 8854) 


v_lshl_add_u32 v13, s51, 6, v12 // (op 8854) 


s_sub_i32 s49, s18, s19 // (op 8854) 


s_cmp_eq_i32 s50, 0 // (op 8854) 
s_cselect_b32 s51, s58, s49 // (op 8854) 



v_lshl_add_u32 v12, s51, 8, v11 // (op 8854) 


v_mul_lo_u32 v11, v12, s2 // (op 8854) 

v_add_lshl_u32 v12, v13, v11, 2 // (op 8854) 


s_add_u32 s18, s6, 256 // (op 9111) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 9111) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 9111) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 9111) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 8 // (op 9111) 

s_mul_i32 s53, s11, s18 // (op 9111) 
s_mul_i32 s49, s8, s53 // (op 9111) 

s_mul_i32 s53, s54, s49 // (op 9111) 

s_sub_u32 s49, s16, s53 // (op 9111) 

s_mov_b32 s53, s49 // (op 9111) 
s_mul_hi_i32 s50, s53, s12 // (op 9111) 
s_add_i32 s51, s50, s53 // (op 9111) 


s_ashr_i32 s50, s51, 31 // (op 9111) 
s_cmp_eq_i32 s12, 0 // (op 9111) 
s_mov_b32 s53, scc // (op 9111) 
s_cmp_eq_i32 s53, 0 // (op 9111) 
s_cselect_b32 s55, 0, -1 // (op 9111) 

s_and_b32 s53, s44, 31 // (op 9111) 
s_lshl_b32 s56, 1, s53 // (op 9111) 
s_add_i32 s57, s56, s55 // (op 9111) 


s_and_b32 s55, s50, s57 // (op 9111) 


s_add_i32 s50, s51, s55 // (op 9111) 


s_ashr_i32 s51, s50, s53 // (op 9111) 


s_lshl_b32 s50, s44, 24 // (op 9111) 
s_ashr_i32 s53, s50, 31 // (op 9111) 

s_xor_b32 s50, s51, s53 // (op 9111) 

s_sub_i32 s51, s50, s53 // (op 9111) 


s_mul_i32 s50, s46, s11 // (op 9111) 
s_add_u32 s56, s20, 64 // (op 9111) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 9111) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 9111) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 9111) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 9111) 

s_sub_i32 s53, s56, s50 // (op 9111) 

s_mul_i32 s55, s51, s53 // (op 9111) 

s_sub_u32 s53, s49, s55 // (op 9111) 


s_add_u32 s49, s53, s50 // (op 9111) 


s_cmp_eq_i32 s54, 0 // (op 9111) 
s_mov_b32 s50, scc // (op 9111) 

s_cmp_eq_i32 s50, 0 // (op 9111) 
s_cselect_b32 s53, s49, s1 // (op 9111) 


s_mul_i32 s1, s52, s18 // (op 9111) 


v_lshl_add_u32 v11, s53, 6, v14 // (op 9111) 


s_sub_i32 s49, s0, s1 // (op 9111) 


s_cmp_eq_i32 s50, 0 // (op 9111) 
s_cselect_b32 s0, s51, s49 // (op 9111) 



v_lshl_add_u32 v13, s0, 8, v10 // (op 9111) 


v_mul_lo_u32 v10, v13, s4 // (op 9111) 

v_add_lshl_u32 v13, v11, v10, 2 // (op 9111) 


s_mul_i32 s1, s2, s7 // (op 8856) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 8856) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 8856) least significant
s_mul_hi_u32 s19, s2, s6 // (op 8856) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8856) most significant: sum
s_add_u32 s1, s1, s19 // (op 8856) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 8856) 

s_mul_i32 s1, s4, s7 // (op 9113) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 9113) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 9113) least significant
s_mul_hi_u32 s50, s4, s6 // (op 9113) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 9113) most significant: sum
s_add_u32 s1, s1, s50 // (op 9113) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 2 // (op 9113) 

s_mov_b32 s52, s22 // (op 8856) reg expression
s_mov_b32 s53, s23 // (op 8856) reg expression
s_mov_b32 s54, s18 // (op 8856) reg expression
s_mov_b32 s55, 131072 // (op 8856) call()

s_mov_b32 s56, s24 // (op 9113) reg expression

s_mov_b32 s57, s25 // (op 9113) reg expression
s_mov_b32 s58, s50 // (op 9113) reg expression
s_mov_b32 s59, 131072 // (op 9113) call()


v_mov_b32 v10, s47 // (op 2) Move value
v_mov_b32 v11, s48 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 336) call()
v_accvgpr_write a5, 0.00000 // (op 336) call()
v_accvgpr_write a6, 0.00000 // (op 336) call()
v_accvgpr_write a7, 0.00000 // (op 336) call()
v_accvgpr_write a8, 0.00000 // (op 512) call()
v_accvgpr_write a9, 0.00000 // (op 512) call()
v_accvgpr_write a10, 0.00000 // (op 512) call()
v_accvgpr_write a11, 0.00000 // (op 512) call()
v_accvgpr_write a12, 0.00000 // (op 688) call()
v_accvgpr_write a13, 0.00000 // (op 688) call()
v_accvgpr_write a14, 0.00000 // (op 688) call()
v_accvgpr_write a15, 0.00000 // (op 688) call()
v_accvgpr_write a16, 0.00000 // (op 864) call()
v_accvgpr_write a17, 0.00000 // (op 864) call()
v_accvgpr_write a18, 0.00000 // (op 864) call()
v_accvgpr_write a19, 0.00000 // (op 864) call()
v_accvgpr_write a20, 0.00000 // (op 1040) call()
v_accvgpr_write a21, 0.00000 // (op 1040) call()
v_accvgpr_write a22, 0.00000 // (op 1040) call()
v_accvgpr_write a23, 0.00000 // (op 1040) call()
v_accvgpr_write a24, 0.00000 // (op 1216) call()
v_accvgpr_write a25, 0.00000 // (op 1216) call()
v_accvgpr_write a26, 0.00000 // (op 1216) call()
v_accvgpr_write a27, 0.00000 // (op 1216) call()
v_accvgpr_write a28, 0.00000 // (op 1392) call()
v_accvgpr_write a29, 0.00000 // (op 1392) call()
v_accvgpr_write a30, 0.00000 // (op 1392) call()
v_accvgpr_write a31, 0.00000 // (op 1392) call()
v_accvgpr_write a32, 0.00000 // (op 1744) call()
v_accvgpr_write a33, 0.00000 // (op 1744) call()
v_accvgpr_write a34, 0.00000 // (op 1744) call()
v_accvgpr_write a35, 0.00000 // (op 1744) call()
v_accvgpr_write a36, 0.00000 // (op 2361) call()
v_accvgpr_write a37, 0.00000 // (op 2361) call()
v_accvgpr_write a38, 0.00000 // (op 2361) call()
v_accvgpr_write a39, 0.00000 // (op 2361) call()
v_accvgpr_write a40, 0.00000 // (op 2362) call()
v_accvgpr_write a41, 0.00000 // (op 2362) call()
v_accvgpr_write a42, 0.00000 // (op 2362) call()
v_accvgpr_write a43, 0.00000 // (op 2362) call()
v_accvgpr_write a44, 0.00000 // (op 2363) call()
v_accvgpr_write a45, 0.00000 // (op 2363) call()
v_accvgpr_write a46, 0.00000 // (op 2363) call()
v_accvgpr_write a47, 0.00000 // (op 2363) call()
v_accvgpr_write a48, 0.00000 // (op 2364) call()
v_accvgpr_write a49, 0.00000 // (op 2364) call()
v_accvgpr_write a50, 0.00000 // (op 2364) call()
v_accvgpr_write a51, 0.00000 // (op 2364) call()
v_accvgpr_write a52, 0.00000 // (op 2365) call()
v_accvgpr_write a53, 0.00000 // (op 2365) call()
v_accvgpr_write a54, 0.00000 // (op 2365) call()
v_accvgpr_write a55, 0.00000 // (op 2365) call()
v_accvgpr_write a56, 0.00000 // (op 2366) call()
v_accvgpr_write a57, 0.00000 // (op 2366) call()
v_accvgpr_write a58, 0.00000 // (op 2366) call()
v_accvgpr_write a59, 0.00000 // (op 2366) call()
v_accvgpr_write a60, 0.00000 // (op 2367) call()
v_accvgpr_write a61, 0.00000 // (op 2367) call()
v_accvgpr_write a62, 0.00000 // (op 2367) call()
v_accvgpr_write a63, 0.00000 // (op 2367) call()
s_mul_hi_i32 s0, s16, s10 // (op 8616) 
v_lshrrev_b32 v14, 3, v1 // (op 8616) 
s_add_i32 s1, s0, s16 // (op 8616) 

v_lshlrev_b32 v15, 1, v14 // (op 8616) 

s_ashr_i32 s0, s1, 31 // (op 8616) 
v_and_b32 v14, -8, v1 // (op 8616) 
s_cmp_eq_i32 s10, 0 // (op 8616) 
s_mov_b32 s18, scc // (op 8616) 
v_sub_u32 v16, v1, v14 // (op 8616) 

s_cmp_eq_i32 s18, 0 // (op 8616) 
s_cselect_b32 s19, 0, -1 // (op 8616) 

v_lshlrev_b32 v14, 4, v16 // (op 8616) 

s_and_b32 s18, s13, 31 // (op 8616) 
s_lshl_b32 s22, 1, s18 // (op 8616) 
s_add_i32 s23, s22, s19 // (op 8616) 


s_and_b32 s19, s0, s23 // (op 8616) 


s_add_i32 s0, s1, s19 // (op 8616) 


s_ashr_i32 s1, s0, s18 // (op 8616) 


s_lshl_b32 s0, s13, 24 // (op 8616) 
s_ashr_i32 s18, s0, 31 // (op 8616) 

s_xor_b32 s0, s1, s18 // (op 8616) 

s_sub_i32 s1, s0, s18 // (op 8616) 


s_mul_hi_i32 s0, s1, s14 // (op 8616) 
s_add_i32 s18, s0, s1 // (op 8616) 

s_ashr_i32 s0, s18, 31 // (op 8616) 
s_cmp_eq_i32 s14, 0 // (op 8616) 
s_mov_b32 s19, scc // (op 8616) 
s_cmp_eq_i32 s19, 0 // (op 8616) 
s_cselect_b32 s22, 0, -1 // (op 8616) 

s_and_b32 s19, s15, 31 // (op 8616) 
s_lshl_b32 s23, 1, s19 // (op 8616) 
s_add_i32 s24, s23, s22 // (op 8616) 


s_and_b32 s25, s0, s24 // (op 8616) 


s_add_i32 s0, s18, s25 // (op 8616) 


s_ashr_i32 s18, s0, s19 // (op 8616) 


s_lshl_b32 s0, s15, 24 // (op 8616) 
s_ashr_i32 s19, s0, 31 // (op 8616) 

s_xor_b32 s0, s18, s19 // (op 8616) 

s_sub_i32 s18, s0, s19 // (op 8616) 


s_mul_i32 s0, s1, s11 // (op 8616) 

s_sub_u32 s1, s16, s0 // (op 8616) 

s_mul_i32 s0, s18, s11 // (op 8616) 
s_add_u32 s0, s0, s1 // (op 8616) 


s_mul_hi_i32 s1, s16, s9 // (op 8616) 
s_add_i32 s18, s1, s16 // (op 8616) 

s_ashr_i32 s1, s18, 31 // (op 8616) 
s_cmp_eq_i32 s9, 0 // (op 8616) 
s_mov_b32 s19, scc // (op 8616) 
s_cmp_eq_i32 s19, 0 // (op 8616) 
s_cselect_b32 s22, 0, -1 // (op 8616) 

s_and_b32 s19, s45, 31 // (op 8616) 
s_lshl_b32 s23, 1, s19 // (op 8616) 
s_add_i32 s24, s23, s22 // (op 8616) 


s_and_b32 s25, s1, s24 // (op 8616) 


s_add_i32 s1, s18, s25 // (op 8616) 


s_ashr_i32 s18, s1, s19 // (op 8616) 


s_lshl_b32 s1, s45, 24 // (op 8616) 
s_ashr_i32 s19, s1, 31 // (op 8616) 

s_xor_b32 s1, s18, s19 // (op 8616) 

s_sub_i32 s18, s1, s19 // (op 8616) 


s_add_u32 s22, s6, 256 // (op 8616) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8616) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8616) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8616) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8616) 

s_mul_i32 s1, s11, s22 // (op 8616) 

s_mul_i32 s19, s8, s1 // (op 8616) 

s_mul_i32 s1, s18, s19 // (op 8616) 

s_sub_u32 s19, s16, s1 // (op 8616) 

s_mov_b32 s1, s19 // (op 8616) 
s_mul_hi_i32 s22, s1, s12 // (op 8616) 
s_add_i32 s23, s22, s1 // (op 8616) 


s_ashr_i32 s1, s23, 31 // (op 8616) 
s_cmp_eq_i32 s12, 0 // (op 8616) 
s_mov_b32 s22, scc // (op 8616) 
s_cmp_eq_i32 s22, 0 // (op 8616) 
s_cselect_b32 s24, 0, -1 // (op 8616) 

s_and_b32 s22, s44, 31 // (op 8616) 
s_lshl_b32 s25, 1, s22 // (op 8616) 
s_add_i32 s47, s25, s24 // (op 8616) 


s_and_b32 s24, s1, s47 // (op 8616) 


s_add_i32 s1, s23, s24 // (op 8616) 


s_ashr_i32 s23, s1, s22 // (op 8616) 


s_lshl_b32 s1, s44, 24 // (op 8616) 
s_ashr_i32 s22, s1, 31 // (op 8616) 

s_xor_b32 s1, s23, s22 // (op 8616) 

s_sub_i32 s23, s1, s22 // (op 8616) 


s_mul_i32 s1, s46, s11 // (op 8616) 
s_add_u32 s24, s20, 64 // (op 8616) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8616) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 8616) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 8616) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 8616) 

s_sub_i32 s22, s24, s1 // (op 8616) 

s_mul_i32 s24, s23, s22 // (op 8616) 


s_sub_u32 s25, s19, s24 // (op 8616) 


s_add_u32 s19, s25, s1 // (op 8616) 


s_cmp_eq_i32 s18, 0 // (op 8616) 
s_mov_b32 s1, scc // (op 8616) 

s_cmp_eq_i32 s1, 0 // (op 8616) 
s_cselect_b32 s18, s19, s0 // (op 8616) 



v_lshl_add_u32 v16, s18, 6, v15 // (op 8616) 


v_mul_lo_u32 v15, v16, s26 // (op 8616) 
v_add_u32 v15, v15, v14 // (op 8616) 


s_mul_i32 s1, s26, s21 // (op 8618) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 8618) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 8618) least significant
s_mul_hi_u32 s19, s26, s20 // (op 8618) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8618) most significant: sum
s_add_u32 s1, s1, s19 // (op 8618) most significant: sum


s_mov_b32 s48, s28 // (op 8618) reg expression
s_mov_b32 s49, s29 // (op 8618) reg expression
s_mov_b32 s50, s0 // (op 8618) reg expression
s_mov_b32 s51, 131072 // (op 8618) call()

s_mul_hi_i32 s47, s16, s10 // (op 8635) 
v_lshrrev_b32 v14, 3, v1 // (op 8635) 
s_add_i32 s0, s47, s16 // (op 8635) 

v_lshlrev_b32 v16, 1, v14 // (op 8635) 

s_ashr_i32 s1, s0, 31 // (op 8635) 
v_and_b32 v14, -8, v1 // (op 8635) 
s_cmp_eq_i32 s10, 0 // (op 8635) 
s_mov_b32 s47, scc // (op 8635) 
v_sub_u32 v17, v1, v14 // (op 8635) 

s_cmp_eq_i32 s47, 0 // (op 8635) 
s_cselect_b32 s18, 0, -1 // (op 8635) 

v_lshlrev_b32 v14, 4, v17 // (op 8635) 

s_and_b32 s19, s13, 31 // (op 8635) 
s_lshl_b32 s47, 1, s19 // (op 8635) 
s_add_i32 s22, s47, s18 // (op 8635) 


s_and_b32 s18, s1, s22 // (op 8635) 


s_add_i32 s1, s0, s18 // (op 8635) 


s_ashr_i32 s0, s1, s19 // (op 8635) 


s_lshl_b32 s1, s13, 24 // (op 8635) 
s_ashr_i32 s47, s1, 31 // (op 8635) 

s_xor_b32 s1, s0, s47 // (op 8635) 

s_sub_i32 s0, s1, s47 // (op 8635) 


s_mul_hi_i32 s1, s0, s14 // (op 8635) 
s_add_i32 s47, s1, s0 // (op 8635) 

s_ashr_i32 s1, s47, 31 // (op 8635) 
s_cmp_eq_i32 s14, 0 // (op 8635) 
s_mov_b32 s18, scc // (op 8635) 
s_cmp_eq_i32 s18, 0 // (op 8635) 
s_cselect_b32 s19, 0, -1 // (op 8635) 

s_and_b32 s18, s15, 31 // (op 8635) 
s_lshl_b32 s22, 1, s18 // (op 8635) 
s_add_i32 s23, s22, s19 // (op 8635) 


s_and_b32 s19, s1, s23 // (op 8635) 


s_add_i32 s1, s47, s19 // (op 8635) 


s_ashr_i32 s19, s1, s18 // (op 8635) 


s_lshl_b32 s1, s15, 24 // (op 8635) 
s_ashr_i32 s18, s1, 31 // (op 8635) 

s_xor_b32 s1, s19, s18 // (op 8635) 

s_sub_i32 s19, s1, s18 // (op 8635) 


s_mul_i32 s1, s0, s11 // (op 8635) 

s_sub_u32 s0, s16, s1 // (op 8635) 

s_mul_i32 s1, s19, s11 // (op 8635) 
s_add_u32 s1, s1, s0 // (op 8635) 


s_mul_hi_i32 s0, s16, s9 // (op 8635) 
s_add_i32 s47, s0, s16 // (op 8635) 

s_ashr_i32 s0, s47, 31 // (op 8635) 
s_cmp_eq_i32 s9, 0 // (op 8635) 
s_mov_b32 s18, scc // (op 8635) 
s_cmp_eq_i32 s18, 0 // (op 8635) 
s_cselect_b32 s19, 0, -1 // (op 8635) 

s_and_b32 s18, s45, 31 // (op 8635) 
s_lshl_b32 s22, 1, s18 // (op 8635) 
s_add_i32 s23, s22, s19 // (op 8635) 


s_and_b32 s19, s0, s23 // (op 8635) 


s_add_i32 s0, s47, s19 // (op 8635) 


s_ashr_i32 s19, s0, s18 // (op 8635) 


s_lshl_b32 s0, s45, 24 // (op 8635) 
s_ashr_i32 s18, s0, 31 // (op 8635) 

s_xor_b32 s0, s19, s18 // (op 8635) 

s_sub_i32 s19, s0, s18 // (op 8635) 


s_add_u32 s22, s6, 256 // (op 8635) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8635) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8635) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8635) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8635) 

s_mul_i32 s0, s11, s22 // (op 8635) 

s_mul_i32 s18, s8, s0 // (op 8635) 

s_mul_i32 s0, s19, s18 // (op 8635) 

s_sub_u32 s18, s16, s0 // (op 8635) 

s_mov_b32 s0, s18 // (op 8635) 
s_mul_hi_i32 s47, s0, s12 // (op 8635) 
s_add_i32 s22, s47, s0 // (op 8635) 


s_ashr_i32 s0, s22, 31 // (op 8635) 
s_cmp_eq_i32 s12, 0 // (op 8635) 
s_mov_b32 s47, scc // (op 8635) 
s_cmp_eq_i32 s47, 0 // (op 8635) 
s_cselect_b32 s23, 0, -1 // (op 8635) 

s_and_b32 s47, s44, 31 // (op 8635) 
s_lshl_b32 s24, 1, s47 // (op 8635) 
s_add_i32 s25, s24, s23 // (op 8635) 


s_and_b32 s23, s0, s25 // (op 8635) 


s_add_i32 s0, s22, s23 // (op 8635) 


s_ashr_i32 s22, s0, s47 // (op 8635) 


s_lshl_b32 s0, s44, 24 // (op 8635) 
s_ashr_i32 s47, s0, 31 // (op 8635) 

s_xor_b32 s0, s22, s47 // (op 8635) 

s_sub_i32 s22, s0, s47 // (op 8635) 


s_mul_i32 s0, s46, s11 // (op 8635) 
s_add_u32 s24, s20, 64 // (op 8635) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8635) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 8635) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 8635) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 6 // (op 8635) 

s_sub_i32 s23, s24, s0 // (op 8635) 

s_mul_i32 s47, s22, s23 // (op 8635) 


s_sub_u32 s22, s18, s47 // (op 8635) 


s_add_u32 s18, s22, s0 // (op 8635) 


s_cmp_eq_i32 s19, 0 // (op 8635) 
s_mov_b32 s0, scc // (op 8635) 

s_cmp_eq_i32 s0, 0 // (op 8635) 
s_cselect_b32 s19, s18, s1 // (op 8635) 



v_lshl_add_u32 v17, s19, 6, v16 // (op 8635) 


v_mul_lo_u32 v16, v17, s26 // (op 8635) 
v_add_u32 v16, v16, v14 // (op 8635) 


s_mul_i32 s1, s26, s21 // (op 8637) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 8637) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 8637) least significant
s_mul_hi_u32 s18, s26, s20 // (op 8637) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8637) most significant: sum
s_add_u32 s1, s1, s18 // (op 8637) most significant: sum


s_mov_b32 s48, s28 // (op 8637) reg expression
s_mov_b32 s49, s29 // (op 8637) reg expression
s_mov_b32 s50, s0 // (op 8637) reg expression
s_mov_b32 s51, 131072 // (op 8637) call()

s_mul_hi_i32 s47, s16, s10 // (op 8680) 
v_and_b32 v14, 4294967292, v1 // (op 8680) 
s_add_i32 s0, s47, s16 // (op 8680) 

v_and_b32 v17, -4, v1 // (op 8680) 
s_ashr_i32 s1, s0, 31 // (op 8680) 
v_sub_u32 v18, v1, v17 // (op 8680) 

s_cmp_eq_i32 s10, 0 // (op 8680) 
s_mov_b32 s47, scc // (op 8680) 
v_lshlrev_b32 v17, 5, v18 // (op 8680) 

s_cmp_eq_i32 s47, 0 // (op 8680) 
s_cselect_b32 s18, 0, -1 // (op 8680) 

v_mov_b32 v19, 0 // (op 8680) convert
v_mov_b32 v18, v17 // (op 8680) convert

s_and_b32 s19, s13, 31 // (op 8680) 
s_lshl_b32 s47, 1, s19 // (op 8680) 
s_add_i32 s22, s47, s18 // (op 8680) 


s_and_b32 s18, s1, s22 // (op 8680) 


s_add_i32 s1, s0, s18 // (op 8680) 


s_ashr_i32 s0, s1, s19 // (op 8680) 


s_lshl_b32 s1, s13, 24 // (op 8680) 
s_ashr_i32 s47, s1, 31 // (op 8680) 

s_xor_b32 s1, s0, s47 // (op 8680) 

s_sub_i32 s0, s1, s47 // (op 8680) 


s_mul_hi_i32 s1, s0, s14 // (op 8680) 
s_add_i32 s47, s1, s0 // (op 8680) 

s_ashr_i32 s1, s47, 31 // (op 8680) 
s_cmp_eq_i32 s14, 0 // (op 8680) 
s_mov_b32 s18, scc // (op 8680) 
s_cmp_eq_i32 s18, 0 // (op 8680) 
s_cselect_b32 s19, 0, -1 // (op 8680) 

s_and_b32 s18, s15, 31 // (op 8680) 
s_lshl_b32 s22, 1, s18 // (op 8680) 
s_add_i32 s23, s22, s19 // (op 8680) 


s_and_b32 s19, s1, s23 // (op 8680) 


s_add_i32 s1, s47, s19 // (op 8680) 


s_ashr_i32 s19, s1, s18 // (op 8680) 


s_lshl_b32 s1, s15, 24 // (op 8680) 
s_ashr_i32 s18, s1, 31 // (op 8680) 

s_xor_b32 s1, s19, s18 // (op 8680) 

s_sub_i32 s19, s1, s18 // (op 8680) 


s_add_u32 s22, s6, 256 // (op 8680) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8680) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8680) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8680) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8680) 

s_mul_i32 s1, s19, s22 // (op 8680) 

s_sub_i32 s47, s0, s1 // (op 8680) 


s_mul_hi_i32 s0, s16, s9 // (op 8680) 
s_add_i32 s1, s0, s16 // (op 8680) 

s_ashr_i32 s0, s1, 31 // (op 8680) 
s_cmp_eq_i32 s9, 0 // (op 8680) 
s_mov_b32 s18, scc // (op 8680) 
s_cmp_eq_i32 s18, 0 // (op 8680) 
s_cselect_b32 s19, 0, -1 // (op 8680) 

s_and_b32 s18, s45, 31 // (op 8680) 
s_lshl_b32 s24, 1, s18 // (op 8680) 
s_add_i32 s25, s24, s19 // (op 8680) 


s_and_b32 s19, s0, s25 // (op 8680) 


s_add_i32 s0, s1, s19 // (op 8680) 


s_ashr_i32 s1, s0, s18 // (op 8680) 


s_lshl_b32 s0, s45, 24 // (op 8680) 
s_ashr_i32 s18, s0, 31 // (op 8680) 

s_xor_b32 s0, s1, s18 // (op 8680) 

s_sub_i32 s1, s0, s18 // (op 8680) 


s_mul_i32 s0, s11, s22 // (op 8680) 

s_mul_i32 s18, s8, s0 // (op 8680) 

s_mul_i32 s0, s1, s18 // (op 8680) 

s_sub_u32 s18, s16, s0 // (op 8680) 

s_mul_hi_i32 s0, s18, s12 // (op 8680) 
s_add_i32 s19, s0, s18 // (op 8680) 


s_ashr_i32 s0, s19, 31 // (op 8680) 
s_cmp_eq_i32 s12, 0 // (op 8680) 
s_mov_b32 s18, scc // (op 8680) 
s_cmp_eq_i32 s18, 0 // (op 8680) 
s_cselect_b32 s22, 0, -1 // (op 8680) 

s_and_b32 s18, s44, 31 // (op 8680) 
s_lshl_b32 s23, 1, s18 // (op 8680) 
s_add_i32 s24, s23, s22 // (op 8680) 


s_and_b32 s25, s0, s24 // (op 8680) 


s_add_i32 s0, s19, s25 // (op 8680) 


s_ashr_i32 s19, s0, s18 // (op 8680) 


s_lshl_b32 s0, s44, 24 // (op 8680) 
s_ashr_i32 s18, s0, 31 // (op 8680) 

s_xor_b32 s0, s19, s18 // (op 8680) 

s_sub_i32 s19, s0, s18 // (op 8680) 


s_cmp_eq_i32 s1, 0 // (op 8680) 
s_mov_b32 s0, scc // (op 8680) 

s_cmp_eq_i32 s0, 0 // (op 8680) 
s_cselect_b32 s1, s19, s47 // (op 8680) 



v_lshl_add_u32 v17, s1, 8, v14 // (op 8680) 


v_mul_lo_u32 v21, v17, s31 // (op 8680) most significant: low of low * high
v_mul_hi_u32 v14, v17, s30 // (op 8680) most significant: high of low * low
v_mul_lo_u32 v20, v17, s30 // (op 8680) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 8680) most significant: sum


v_add_co_u32 v22, s[0:1], v18, v20 // (op 8680) least significant half
v_addc_co_u32 v23, s[0:1], v19, v21, s[0:1] // (op 8680) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 8680) 


v_lshrrev_b64 v[20:21], 3, v[22:23] // (op 8680) 

v_mov_b32 v14, v20 // (op 8680) convert

s_mul_i32 s1, s30, s7 // (op 8682) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 8682) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 8682) least significant
s_mul_hi_u32 s18, s30, s6 // (op 8682) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8682) most significant: sum
s_add_u32 s1, s1, s18 // (op 8682) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 8682) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 8682) 

s_mov_b32 s60, s32 // (op 8682) reg expression
s_mov_b32 s61, s33 // (op 8682) reg expression
s_mov_b32 s62, s0 // (op 8682) reg expression
s_mov_b32 s63, 131072 // (op 8682) call()

s_mul_hi_i32 s47, s16, s10 // (op 8699) 
v_and_b32 v17, 4294967292, v1 // (op 8699) 
s_add_i32 s0, s47, s16 // (op 8699) 

v_and_b32 v18, -4, v1 // (op 8699) 
s_ashr_i32 s1, s0, 31 // (op 8699) 
v_sub_u32 v19, v1, v18 // (op 8699) 

s_cmp_eq_i32 s10, 0 // (op 8699) 
s_mov_b32 s47, scc // (op 8699) 
v_lshlrev_b32 v18, 5, v19 // (op 8699) 

s_cmp_eq_i32 s47, 0 // (op 8699) 
s_cselect_b32 s18, 0, -1 // (op 8699) 

v_mov_b32 v21, 0 // (op 8699) convert
v_mov_b32 v20, v18 // (op 8699) convert

s_and_b32 s19, s13, 31 // (op 8699) 
s_lshl_b32 s47, 1, s19 // (op 8699) 
s_add_i32 s22, s47, s18 // (op 8699) 


s_and_b32 s18, s1, s22 // (op 8699) 


s_add_i32 s1, s0, s18 // (op 8699) 


s_ashr_i32 s0, s1, s19 // (op 8699) 


s_lshl_b32 s1, s13, 24 // (op 8699) 
s_ashr_i32 s47, s1, 31 // (op 8699) 

s_xor_b32 s1, s0, s47 // (op 8699) 

s_sub_i32 s0, s1, s47 // (op 8699) 


s_mul_hi_i32 s1, s0, s14 // (op 8699) 
s_add_i32 s47, s1, s0 // (op 8699) 

s_ashr_i32 s1, s47, 31 // (op 8699) 
s_cmp_eq_i32 s14, 0 // (op 8699) 
s_mov_b32 s18, scc // (op 8699) 
s_cmp_eq_i32 s18, 0 // (op 8699) 
s_cselect_b32 s19, 0, -1 // (op 8699) 

s_and_b32 s18, s15, 31 // (op 8699) 
s_lshl_b32 s22, 1, s18 // (op 8699) 
s_add_i32 s23, s22, s19 // (op 8699) 


s_and_b32 s19, s1, s23 // (op 8699) 


s_add_i32 s1, s47, s19 // (op 8699) 


s_ashr_i32 s19, s1, s18 // (op 8699) 


s_lshl_b32 s1, s15, 24 // (op 8699) 
s_ashr_i32 s18, s1, 31 // (op 8699) 

s_xor_b32 s1, s19, s18 // (op 8699) 

s_sub_i32 s19, s1, s18 // (op 8699) 


s_add_u32 s22, s6, 256 // (op 8699) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8699) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8699) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8699) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8699) 

s_mul_i32 s1, s19, s22 // (op 8699) 

s_sub_i32 s47, s0, s1 // (op 8699) 


s_mul_hi_i32 s0, s16, s9 // (op 8699) 
s_add_i32 s1, s0, s16 // (op 8699) 

s_ashr_i32 s0, s1, 31 // (op 8699) 
s_cmp_eq_i32 s9, 0 // (op 8699) 
s_mov_b32 s18, scc // (op 8699) 
s_cmp_eq_i32 s18, 0 // (op 8699) 
s_cselect_b32 s19, 0, -1 // (op 8699) 

s_and_b32 s18, s45, 31 // (op 8699) 
s_lshl_b32 s24, 1, s18 // (op 8699) 
s_add_i32 s25, s24, s19 // (op 8699) 


s_and_b32 s19, s0, s25 // (op 8699) 


s_add_i32 s0, s1, s19 // (op 8699) 


s_ashr_i32 s1, s0, s18 // (op 8699) 


s_lshl_b32 s0, s45, 24 // (op 8699) 
s_ashr_i32 s18, s0, 31 // (op 8699) 

s_xor_b32 s0, s1, s18 // (op 8699) 

s_sub_i32 s1, s0, s18 // (op 8699) 


s_mul_i32 s0, s11, s22 // (op 8699) 

s_mul_i32 s18, s8, s0 // (op 8699) 

s_mul_i32 s0, s1, s18 // (op 8699) 

s_sub_u32 s18, s16, s0 // (op 8699) 

s_mul_hi_i32 s0, s18, s12 // (op 8699) 
s_add_i32 s19, s0, s18 // (op 8699) 


s_ashr_i32 s0, s19, 31 // (op 8699) 
s_cmp_eq_i32 s12, 0 // (op 8699) 
s_mov_b32 s18, scc // (op 8699) 
s_cmp_eq_i32 s18, 0 // (op 8699) 
s_cselect_b32 s22, 0, -1 // (op 8699) 

s_and_b32 s18, s44, 31 // (op 8699) 
s_lshl_b32 s23, 1, s18 // (op 8699) 
s_add_i32 s24, s23, s22 // (op 8699) 


s_and_b32 s25, s0, s24 // (op 8699) 


s_add_i32 s0, s19, s25 // (op 8699) 


s_ashr_i32 s19, s0, s18 // (op 8699) 


s_lshl_b32 s0, s44, 24 // (op 8699) 
s_ashr_i32 s18, s0, 31 // (op 8699) 

s_xor_b32 s0, s19, s18 // (op 8699) 

s_sub_i32 s19, s0, s18 // (op 8699) 


s_cmp_eq_i32 s1, 0 // (op 8699) 
s_mov_b32 s0, scc // (op 8699) 

s_cmp_eq_i32 s0, 0 // (op 8699) 
s_cselect_b32 s1, s19, s47 // (op 8699) 



v_lshl_add_u32 v18, s1, 8, v17 // (op 8699) 


v_mul_lo_u32 v23, v18, s31 // (op 8699) most significant: low of low * high
v_mul_hi_u32 v17, v18, s30 // (op 8699) most significant: high of low * low
v_mul_lo_u32 v22, v18, s30 // (op 8699) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 8699) most significant: sum


v_add_co_u32 v18, s[0:1], v20, v22 // (op 8699) least significant half
v_addc_co_u32 v19, s[0:1], v21, v23, s[0:1] // (op 8699) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 8699) 


v_lshrrev_b64 v[20:21], 3, v[18:19] // (op 8699) 

v_mov_b32 v17, v20 // (op 8699) convert

s_mul_i32 s1, s30, s7 // (op 8701) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 8701) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 8701) least significant
s_mul_hi_u32 s18, s30, s6 // (op 8701) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8701) most significant: sum
s_add_u32 s1, s1, s18 // (op 8701) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 8701) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 8701) 

s_mov_b32 s60, s32 // (op 8701) reg expression
s_mov_b32 s61, s33 // (op 8701) reg expression
s_mov_b32 s62, s0 // (op 8701) reg expression
s_mov_b32 s63, 131072 // (op 8701) call()

s_mul_hi_i32 s47, s16, s10 // (op 8744) 
v_lshrrev_b32 v18, 2, v1 // (op 8744) 
s_add_i32 s0, s47, s16 // (op 8744) 

v_and_b32 v19, -4, v1 // (op 8744) 
s_ashr_i32 s1, s0, 31 // (op 8744) 
v_sub_u32 v20, v1, v19 // (op 8744) 

s_cmp_eq_i32 s10, 0 // (op 8744) 
s_mov_b32 s47, scc // (op 8744) 
s_cmp_eq_i32 s47, 0 // (op 8744) 
s_cselect_b32 s18, 0, -1 // (op 8744) 

s_and_b32 s19, s13, 31 // (op 8744) 
s_lshl_b32 s47, 1, s19 // (op 8744) 
s_add_i32 s22, s47, s18 // (op 8744) 


s_and_b32 s18, s1, s22 // (op 8744) 


s_add_i32 s1, s0, s18 // (op 8744) 


s_ashr_i32 s0, s1, s19 // (op 8744) 


s_lshl_b32 s1, s13, 24 // (op 8744) 
s_ashr_i32 s47, s1, 31 // (op 8744) 

s_xor_b32 s1, s0, s47 // (op 8744) 

s_sub_i32 s0, s1, s47 // (op 8744) 


s_mul_hi_i32 s1, s0, s14 // (op 8744) 
s_add_i32 s47, s1, s0 // (op 8744) 

s_ashr_i32 s1, s47, 31 // (op 8744) 
s_cmp_eq_i32 s14, 0 // (op 8744) 
s_mov_b32 s18, scc // (op 8744) 
s_cmp_eq_i32 s18, 0 // (op 8744) 
s_cselect_b32 s19, 0, -1 // (op 8744) 

s_and_b32 s18, s15, 31 // (op 8744) 
s_lshl_b32 s22, 1, s18 // (op 8744) 
s_add_i32 s23, s22, s19 // (op 8744) 


s_and_b32 s19, s1, s23 // (op 8744) 


s_add_i32 s1, s47, s19 // (op 8744) 


s_ashr_i32 s19, s1, s18 // (op 8744) 


s_lshl_b32 s1, s15, 24 // (op 8744) 
s_ashr_i32 s18, s1, 31 // (op 8744) 

s_xor_b32 s1, s19, s18 // (op 8744) 

s_sub_i32 s19, s1, s18 // (op 8744) 


s_mul_i32 s1, s0, s11 // (op 8744) 

s_sub_u32 s0, s16, s1 // (op 8744) 

s_mul_i32 s1, s19, s11 // (op 8744) 
s_add_u32 s1, s1, s0 // (op 8744) 


s_mul_hi_i32 s0, s16, s9 // (op 8744) 
s_add_i32 s47, s0, s16 // (op 8744) 

s_ashr_i32 s0, s47, 31 // (op 8744) 
s_cmp_eq_i32 s9, 0 // (op 8744) 
s_mov_b32 s18, scc // (op 8744) 
s_cmp_eq_i32 s18, 0 // (op 8744) 
s_cselect_b32 s19, 0, -1 // (op 8744) 

s_and_b32 s18, s45, 31 // (op 8744) 
s_lshl_b32 s22, 1, s18 // (op 8744) 
s_add_i32 s23, s22, s19 // (op 8744) 


s_and_b32 s19, s0, s23 // (op 8744) 


s_add_i32 s0, s47, s19 // (op 8744) 


s_ashr_i32 s19, s0, s18 // (op 8744) 


s_lshl_b32 s0, s45, 24 // (op 8744) 
s_ashr_i32 s18, s0, 31 // (op 8744) 

s_xor_b32 s0, s19, s18 // (op 8744) 

s_sub_i32 s19, s0, s18 // (op 8744) 


s_add_u32 s22, s6, 256 // (op 8744) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8744) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8744) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8744) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8744) 

s_mul_i32 s0, s11, s22 // (op 8744) 

s_mul_i32 s18, s8, s0 // (op 8744) 

s_mul_i32 s0, s19, s18 // (op 8744) 

s_sub_u32 s18, s16, s0 // (op 8744) 

s_mov_b32 s0, s18 // (op 8744) 
s_mul_hi_i32 s47, s0, s12 // (op 8744) 
s_add_i32 s22, s47, s0 // (op 8744) 


s_ashr_i32 s0, s22, 31 // (op 8744) 
s_cmp_eq_i32 s12, 0 // (op 8744) 
s_mov_b32 s47, scc // (op 8744) 
s_cmp_eq_i32 s47, 0 // (op 8744) 
s_cselect_b32 s23, 0, -1 // (op 8744) 

s_and_b32 s47, s44, 31 // (op 8744) 
s_lshl_b32 s24, 1, s47 // (op 8744) 
s_add_i32 s25, s24, s23 // (op 8744) 


s_and_b32 s23, s0, s25 // (op 8744) 


s_add_i32 s0, s22, s23 // (op 8744) 


s_ashr_i32 s22, s0, s47 // (op 8744) 


s_lshl_b32 s0, s44, 24 // (op 8744) 
s_ashr_i32 s47, s0, 31 // (op 8744) 

s_xor_b32 s0, s22, s47 // (op 8744) 

s_sub_i32 s22, s0, s47 // (op 8744) 


s_mul_i32 s0, s46, s11 // (op 8744) 
s_add_u32 s24, s20, 64 // (op 8744) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8744) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 8744) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 8744) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 6 // (op 8744) 

s_sub_i32 s23, s24, s0 // (op 8744) 

s_mul_i32 s47, s22, s23 // (op 8744) 


s_sub_u32 s22, s18, s47 // (op 8744) 


s_add_u32 s18, s22, s0 // (op 8744) 


s_cmp_eq_i32 s19, 0 // (op 8744) 
s_mov_b32 s0, scc // (op 8744) 

s_cmp_eq_i32 s0, 0 // (op 8744) 
s_cselect_b32 s19, s18, s1 // (op 8744) 



v_lshl_add_u32 v19, s19, 6, v18 // (op 8744) 


v_mul_lo_u32 v18, v19, s34 // (op 8744) 
v_add_u32 v18, v18, v20 // (op 8744) 


s_mul_i32 s1, s34, s21 // (op 8746) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 8746) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 8746) least significant
s_mul_hi_u32 s18, s34, s20 // (op 8746) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8746) most significant: sum
s_add_u32 s1, s1, s18 // (op 8746) most significant: sum


s_mov_b32 s64, s36 // (op 8746) reg expression
s_mov_b32 s65, s37 // (op 8746) reg expression
s_mov_b32 s66, s0 // (op 8746) reg expression
s_mov_b32 s67, 131072 // (op 8746) call()

s_mul_hi_i32 s47, s16, s10 // (op 8763) 
v_lshrrev_b32 v19, 2, v1 // (op 8763) 
s_add_i32 s0, s47, s16 // (op 8763) 

v_and_b32 v20, -4, v1 // (op 8763) 
s_ashr_i32 s1, s0, 31 // (op 8763) 
v_sub_u32 v21, v1, v20 // (op 8763) 

s_cmp_eq_i32 s10, 0 // (op 8763) 
s_mov_b32 s47, scc // (op 8763) 
s_cmp_eq_i32 s47, 0 // (op 8763) 
s_cselect_b32 s18, 0, -1 // (op 8763) 

s_and_b32 s19, s13, 31 // (op 8763) 
s_lshl_b32 s47, 1, s19 // (op 8763) 
s_add_i32 s22, s47, s18 // (op 8763) 


s_and_b32 s18, s1, s22 // (op 8763) 


s_add_i32 s1, s0, s18 // (op 8763) 


s_ashr_i32 s0, s1, s19 // (op 8763) 


s_lshl_b32 s1, s13, 24 // (op 8763) 
s_ashr_i32 s47, s1, 31 // (op 8763) 

s_xor_b32 s1, s0, s47 // (op 8763) 

s_sub_i32 s0, s1, s47 // (op 8763) 


s_mul_hi_i32 s1, s0, s14 // (op 8763) 
s_add_i32 s47, s1, s0 // (op 8763) 

s_ashr_i32 s1, s47, 31 // (op 8763) 
s_cmp_eq_i32 s14, 0 // (op 8763) 
s_mov_b32 s18, scc // (op 8763) 
s_cmp_eq_i32 s18, 0 // (op 8763) 
s_cselect_b32 s19, 0, -1 // (op 8763) 

s_and_b32 s18, s15, 31 // (op 8763) 
s_lshl_b32 s22, 1, s18 // (op 8763) 
s_add_i32 s23, s22, s19 // (op 8763) 


s_and_b32 s19, s1, s23 // (op 8763) 


s_add_i32 s1, s47, s19 // (op 8763) 


s_ashr_i32 s19, s1, s18 // (op 8763) 


s_lshl_b32 s1, s15, 24 // (op 8763) 
s_ashr_i32 s18, s1, 31 // (op 8763) 

s_xor_b32 s1, s19, s18 // (op 8763) 

s_sub_i32 s19, s1, s18 // (op 8763) 


s_mul_i32 s1, s0, s11 // (op 8763) 

s_sub_u32 s0, s16, s1 // (op 8763) 

s_mul_i32 s1, s19, s11 // (op 8763) 
s_add_u32 s1, s1, s0 // (op 8763) 


s_mul_hi_i32 s0, s16, s9 // (op 8763) 
s_add_i32 s47, s0, s16 // (op 8763) 

s_ashr_i32 s0, s47, 31 // (op 8763) 
s_cmp_eq_i32 s9, 0 // (op 8763) 
s_mov_b32 s18, scc // (op 8763) 
s_cmp_eq_i32 s18, 0 // (op 8763) 
s_cselect_b32 s19, 0, -1 // (op 8763) 

s_and_b32 s18, s45, 31 // (op 8763) 
s_lshl_b32 s22, 1, s18 // (op 8763) 
s_add_i32 s23, s22, s19 // (op 8763) 


s_and_b32 s19, s0, s23 // (op 8763) 


s_add_i32 s0, s47, s19 // (op 8763) 


s_ashr_i32 s19, s0, s18 // (op 8763) 


s_lshl_b32 s0, s45, 24 // (op 8763) 
s_ashr_i32 s18, s0, 31 // (op 8763) 

s_xor_b32 s0, s19, s18 // (op 8763) 

s_sub_i32 s19, s0, s18 // (op 8763) 


s_add_u32 s22, s6, 256 // (op 8763) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8763) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8763) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8763) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8763) 

s_mul_i32 s0, s11, s22 // (op 8763) 

s_mul_i32 s18, s8, s0 // (op 8763) 

s_mul_i32 s0, s19, s18 // (op 8763) 

s_sub_u32 s18, s16, s0 // (op 8763) 

s_mov_b32 s0, s18 // (op 8763) 
s_mul_hi_i32 s47, s0, s12 // (op 8763) 
s_add_i32 s22, s47, s0 // (op 8763) 


s_ashr_i32 s0, s22, 31 // (op 8763) 
s_cmp_eq_i32 s12, 0 // (op 8763) 
s_mov_b32 s47, scc // (op 8763) 
s_cmp_eq_i32 s47, 0 // (op 8763) 
s_cselect_b32 s23, 0, -1 // (op 8763) 

s_and_b32 s47, s44, 31 // (op 8763) 
s_lshl_b32 s24, 1, s47 // (op 8763) 
s_add_i32 s25, s24, s23 // (op 8763) 


s_and_b32 s23, s0, s25 // (op 8763) 


s_add_i32 s0, s22, s23 // (op 8763) 


s_ashr_i32 s22, s0, s47 // (op 8763) 


s_lshl_b32 s0, s44, 24 // (op 8763) 
s_ashr_i32 s47, s0, 31 // (op 8763) 

s_xor_b32 s0, s22, s47 // (op 8763) 

s_sub_i32 s22, s0, s47 // (op 8763) 


s_mul_i32 s0, s46, s11 // (op 8763) 
s_add_u32 s24, s20, 64 // (op 8763) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8763) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 8763) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 8763) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 6 // (op 8763) 

s_sub_i32 s23, s24, s0 // (op 8763) 

s_mul_i32 s47, s22, s23 // (op 8763) 


s_sub_u32 s22, s18, s47 // (op 8763) 


s_add_u32 s18, s22, s0 // (op 8763) 


s_cmp_eq_i32 s19, 0 // (op 8763) 
s_mov_b32 s0, scc // (op 8763) 

s_cmp_eq_i32 s0, 0 // (op 8763) 
s_cselect_b32 s19, s18, s1 // (op 8763) 



v_lshl_add_u32 v20, s19, 6, v19 // (op 8763) 


v_mul_lo_u32 v19, v20, s34 // (op 8763) 
v_add_u32 v19, v19, v21 // (op 8763) 


s_mul_i32 s1, s34, s21 // (op 8765) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 8765) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 8765) least significant
s_mul_hi_u32 s18, s34, s20 // (op 8765) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8765) most significant: sum
s_add_u32 s1, s1, s18 // (op 8765) most significant: sum


s_mov_b32 s64, s36 // (op 8765) reg expression
s_mov_b32 s65, s37 // (op 8765) reg expression
s_mov_b32 s66, s0 // (op 8765) reg expression
s_mov_b32 s67, 131072 // (op 8765) call()

s_mul_hi_i32 s47, s16, s10 // (op 8808) 
s_add_i32 s0, s47, s16 // (op 8808) 

s_ashr_i32 s1, s0, 31 // (op 8808) 
s_cmp_eq_i32 s10, 0 // (op 8808) 
s_mov_b32 s47, scc // (op 8808) 
s_cmp_eq_i32 s47, 0 // (op 8808) 
s_cselect_b32 s18, 0, -1 // (op 8808) 

s_and_b32 s19, s13, 31 // (op 8808) 
s_lshl_b32 s47, 1, s19 // (op 8808) 
s_add_i32 s22, s47, s18 // (op 8808) 


s_and_b32 s18, s1, s22 // (op 8808) 


s_add_i32 s1, s0, s18 // (op 8808) 


s_ashr_i32 s0, s1, s19 // (op 8808) 


s_lshl_b32 s1, s13, 24 // (op 8808) 
s_ashr_i32 s47, s1, 31 // (op 8808) 

s_xor_b32 s1, s0, s47 // (op 8808) 

s_sub_i32 s0, s1, s47 // (op 8808) 


s_mul_hi_i32 s1, s0, s14 // (op 8808) 
s_add_i32 s47, s1, s0 // (op 8808) 

s_ashr_i32 s1, s47, 31 // (op 8808) 
s_cmp_eq_i32 s14, 0 // (op 8808) 
s_mov_b32 s18, scc // (op 8808) 
s_cmp_eq_i32 s18, 0 // (op 8808) 
s_cselect_b32 s19, 0, -1 // (op 8808) 

s_and_b32 s18, s15, 31 // (op 8808) 
s_lshl_b32 s22, 1, s18 // (op 8808) 
s_add_i32 s23, s22, s19 // (op 8808) 


s_and_b32 s19, s1, s23 // (op 8808) 


s_add_i32 s1, s47, s19 // (op 8808) 


s_ashr_i32 s19, s1, s18 // (op 8808) 


s_lshl_b32 s1, s15, 24 // (op 8808) 
s_ashr_i32 s18, s1, 31 // (op 8808) 

s_xor_b32 s1, s19, s18 // (op 8808) 

s_sub_i32 s19, s1, s18 // (op 8808) 


s_add_u32 s22, s6, 256 // (op 8808) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8808) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8808) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8808) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8808) 

s_mul_i32 s1, s19, s22 // (op 8808) 

s_sub_i32 s47, s0, s1 // (op 8808) 


s_mul_hi_i32 s0, s16, s9 // (op 8808) 
s_add_i32 s1, s0, s16 // (op 8808) 

s_ashr_i32 s0, s1, 31 // (op 8808) 
s_cmp_eq_i32 s9, 0 // (op 8808) 
s_mov_b32 s18, scc // (op 8808) 
s_cmp_eq_i32 s18, 0 // (op 8808) 
s_cselect_b32 s19, 0, -1 // (op 8808) 

s_and_b32 s18, s45, 31 // (op 8808) 
s_lshl_b32 s24, 1, s18 // (op 8808) 
s_add_i32 s25, s24, s19 // (op 8808) 


s_and_b32 s19, s0, s25 // (op 8808) 


s_add_i32 s0, s1, s19 // (op 8808) 


s_ashr_i32 s1, s0, s18 // (op 8808) 


s_lshl_b32 s0, s45, 24 // (op 8808) 
s_ashr_i32 s18, s0, 31 // (op 8808) 

s_xor_b32 s0, s1, s18 // (op 8808) 

s_sub_i32 s1, s0, s18 // (op 8808) 


s_mul_i32 s0, s11, s22 // (op 8808) 

s_mul_i32 s18, s8, s0 // (op 8808) 

s_mul_i32 s0, s1, s18 // (op 8808) 

s_sub_u32 s18, s16, s0 // (op 8808) 

s_mul_hi_i32 s0, s18, s12 // (op 8808) 
s_add_i32 s19, s0, s18 // (op 8808) 


s_ashr_i32 s0, s19, 31 // (op 8808) 
s_cmp_eq_i32 s12, 0 // (op 8808) 
s_mov_b32 s18, scc // (op 8808) 
s_cmp_eq_i32 s18, 0 // (op 8808) 
s_cselect_b32 s22, 0, -1 // (op 8808) 

s_and_b32 s18, s44, 31 // (op 8808) 
s_lshl_b32 s23, 1, s18 // (op 8808) 
s_add_i32 s24, s23, s22 // (op 8808) 


s_and_b32 s25, s0, s24 // (op 8808) 


s_add_i32 s0, s19, s25 // (op 8808) 


s_ashr_i32 s19, s0, s18 // (op 8808) 


s_lshl_b32 s0, s44, 24 // (op 8808) 
s_ashr_i32 s18, s0, 31 // (op 8808) 

s_xor_b32 s0, s19, s18 // (op 8808) 

s_sub_i32 s19, s0, s18 // (op 8808) 


s_cmp_eq_i32 s1, 0 // (op 8808) 
s_mov_b32 s0, scc // (op 8808) 

s_cmp_eq_i32 s0, 0 // (op 8808) 
s_cselect_b32 s1, s19, s47 // (op 8808) 



v_lshl_add_u32 v20, s1, 8, v1 // (op 8808) 

v_mul_lo_u32 v21, v20, s38 // (op 8808) 

s_mul_i32 s1, s38, s7 // (op 8810) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 8810) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 8810) least significant
s_mul_hi_u32 s18, s38, s6 // (op 8810) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8810) most significant: sum
s_add_u32 s1, s1, s18 // (op 8810) most significant: sum


s_mov_b32 s68, s40 // (op 8810) reg expression
s_mov_b32 s69, s41 // (op 8810) reg expression
s_mov_b32 s70, s0 // (op 8810) reg expression
s_mov_b32 s71, 131072 // (op 8810) call()

s_mul_hi_i32 s47, s16, s10 // (op 8827) 
s_add_i32 s0, s47, s16 // (op 8827) 

s_ashr_i32 s1, s0, 31 // (op 8827) 
s_cmp_eq_i32 s10, 0 // (op 8827) 
s_mov_b32 s47, scc // (op 8827) 
s_cmp_eq_i32 s47, 0 // (op 8827) 
s_cselect_b32 s18, 0, -1 // (op 8827) 

s_and_b32 s19, s13, 31 // (op 8827) 
s_lshl_b32 s47, 1, s19 // (op 8827) 
s_add_i32 s22, s47, s18 // (op 8827) 


s_and_b32 s18, s1, s22 // (op 8827) 


s_add_i32 s1, s0, s18 // (op 8827) 


s_ashr_i32 s0, s1, s19 // (op 8827) 


s_lshl_b32 s1, s13, 24 // (op 8827) 
s_ashr_i32 s47, s1, 31 // (op 8827) 

s_xor_b32 s1, s0, s47 // (op 8827) 

s_sub_i32 s0, s1, s47 // (op 8827) 


s_mul_hi_i32 s1, s0, s14 // (op 8827) 
s_add_i32 s47, s1, s0 // (op 8827) 

s_ashr_i32 s1, s47, 31 // (op 8827) 
s_cmp_eq_i32 s14, 0 // (op 8827) 
s_mov_b32 s18, scc // (op 8827) 
s_cmp_eq_i32 s18, 0 // (op 8827) 
s_cselect_b32 s19, 0, -1 // (op 8827) 

s_and_b32 s18, s15, 31 // (op 8827) 
s_lshl_b32 s22, 1, s18 // (op 8827) 
s_add_i32 s23, s22, s19 // (op 8827) 


s_and_b32 s19, s1, s23 // (op 8827) 


s_add_i32 s1, s47, s19 // (op 8827) 


s_ashr_i32 s19, s1, s18 // (op 8827) 


s_lshl_b32 s1, s15, 24 // (op 8827) 
s_ashr_i32 s18, s1, 31 // (op 8827) 

s_xor_b32 s1, s19, s18 // (op 8827) 

s_sub_i32 s19, s1, s18 // (op 8827) 


s_add_u32 s22, s6, 256 // (op 8827) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8827) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8827) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8827) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8827) 

s_mul_i32 s1, s19, s22 // (op 8827) 

s_sub_i32 s47, s0, s1 // (op 8827) 


s_mul_hi_i32 s0, s16, s9 // (op 8827) 
s_add_i32 s1, s0, s16 // (op 8827) 

s_ashr_i32 s0, s1, 31 // (op 8827) 
s_cmp_eq_i32 s9, 0 // (op 8827) 
s_mov_b32 s18, scc // (op 8827) 
s_cmp_eq_i32 s18, 0 // (op 8827) 
s_cselect_b32 s19, 0, -1 // (op 8827) 

s_and_b32 s18, s45, 31 // (op 8827) 
s_lshl_b32 s24, 1, s18 // (op 8827) 
s_add_i32 s25, s24, s19 // (op 8827) 


s_and_b32 s19, s0, s25 // (op 8827) 


s_add_i32 s0, s1, s19 // (op 8827) 


s_ashr_i32 s1, s0, s18 // (op 8827) 


s_lshl_b32 s0, s45, 24 // (op 8827) 
s_ashr_i32 s18, s0, 31 // (op 8827) 

s_xor_b32 s0, s1, s18 // (op 8827) 

s_sub_i32 s1, s0, s18 // (op 8827) 


s_mul_i32 s0, s11, s22 // (op 8827) 

s_mul_i32 s18, s8, s0 // (op 8827) 

s_mul_i32 s0, s1, s18 // (op 8827) 

s_sub_u32 s18, s16, s0 // (op 8827) 

s_mul_hi_i32 s0, s18, s12 // (op 8827) 
s_add_i32 s19, s0, s18 // (op 8827) 


s_ashr_i32 s0, s19, 31 // (op 8827) 
s_cmp_eq_i32 s12, 0 // (op 8827) 
s_mov_b32 s18, scc // (op 8827) 
s_cmp_eq_i32 s18, 0 // (op 8827) 
s_cselect_b32 s22, 0, -1 // (op 8827) 

s_and_b32 s18, s44, 31 // (op 8827) 
s_lshl_b32 s23, 1, s18 // (op 8827) 
s_add_i32 s24, s23, s22 // (op 8827) 


s_and_b32 s25, s0, s24 // (op 8827) 


s_add_i32 s0, s19, s25 // (op 8827) 


s_ashr_i32 s19, s0, s18 // (op 8827) 


s_lshl_b32 s0, s44, 24 // (op 8827) 
s_ashr_i32 s18, s0, 31 // (op 8827) 

s_xor_b32 s0, s19, s18 // (op 8827) 

s_sub_i32 s19, s0, s18 // (op 8827) 


s_cmp_eq_i32 s1, 0 // (op 8827) 
s_mov_b32 s0, scc // (op 8827) 

s_cmp_eq_i32 s0, 0 // (op 8827) 
s_cselect_b32 s1, s19, s47 // (op 8827) 



v_lshl_add_u32 v20, s1, 8, v1 // (op 8827) 

v_mul_lo_u32 v22, v20, s38 // (op 8827) 

s_mul_i32 s1, s38, s7 // (op 8829) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 8829) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 8829) least significant
s_mul_hi_u32 s18, s38, s6 // (op 8829) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8829) most significant: sum
s_add_u32 s1, s1, s18 // (op 8829) most significant: sum


s_mov_b32 s68, s40 // (op 8829) reg expression
s_mov_b32 s69, s41 // (op 8829) reg expression
s_mov_b32 s70, s0 // (op 8829) reg expression
s_mov_b32 s71, 131072 // (op 8829) call()


s_lshl_b64 s[0:1], s[30:31], 2 // (op 8136) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 8136) 


v_mov_b32 v20, v14 // (op 8136) 
buffer_load_dwordx4 v[26:29], v20, s[60:63], 0 offen // (op 8136) Load value
v_add_u32 v20, v20, v24 // (op 8136) 
buffer_load_dwordx4 v[30:33], v20, s[60:63], 0 offen // (op 8136) Load value
v_add_u32 v20, v20, v24 // (op 8136) 
buffer_load_dwordx4 v[34:37], v20, s[60:63], 0 offen // (op 8136) Load value
v_add_u32 v20, v20, v24 // (op 8136) 
buffer_load_dwordx4 v[38:41], v20, s[60:63], 0 offen // (op 8136) Load value





buffer_load_dword v20, v21, s[68:71], 0 offen // (op 8139) Load value

v_mov_b64 v[24:25], s[26:27] // (op 8142) convert

v_mov_b32 v23, v15 // (op 8142) 
buffer_load_dwordx4 v[42:45], v23, s[48:51], 0 offen // (op 8142) Load value
v_add_u32 v23, v23, v24 // (op 8142) 
buffer_load_dwordx4 v[46:49], v23, s[48:51], 0 offen // (op 8142) Load value





buffer_load_ubyte v23, v18, s[64:67], 0 offen offset: 0 // (op 8145) Load value

s_lshl_b64 s[0:1], s[30:31], 2 // (op 8148) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 8148) 


v_mov_b32 v50, v14 // (op 8148) 
buffer_load_dwordx4 v[52:55], v50, s[60:63], 0 offen offset:64 // (op 8148) Load value
v_add_u32 v50, v50, v24 // (op 8148) 
buffer_load_dwordx4 v[56:59], v50, s[60:63], 0 offen offset:64 // (op 8148) Load value
v_add_u32 v50, v50, v24 // (op 8148) 
buffer_load_dwordx4 v[60:63], v50, s[60:63], 0 offen offset:64 // (op 8148) Load value
v_add_u32 v50, v50, v24 // (op 8148) 
buffer_load_dwordx4 v[64:67], v50, s[60:63], 0 offen offset:64 // (op 8148) Load value






buffer_load_dword v14, v21, s[68:71], 0 offen offset:4 // (op 8151) Load value


v_mov_b64 v[24:25], s[26:27] // (op 8154) convert

v_mov_b32 v21, v15 // (op 8154) 
buffer_load_dwordx4 v[68:71], v21, s[48:51], 0 offen offset:128 // (op 8154) Load value
v_add_u32 v21, v21, v24 // (op 8154) 
buffer_load_dwordx4 v[72:75], v21, s[48:51], 0 offen offset:128 // (op 8154) Load value






buffer_load_ubyte v15, v18, s[64:67], 0 offen offset:4 // (op 8157) Load value

v_and_b32 v18, -4, v1 // (op 8957) 
v_sub_u32 v21, v1, v18 // (op 8957) 

v_lshlrev_b32 v18, 5, v21 // (op 8957) 

v_lshrrev_b32 v21, 2, v1 // (op 8957) 
v_lshl_add_u32 v24, v21, 9, v18 // (op 8957) 


v_lshlrev_b32 v18, 2, v24 // (op 8957) 

v_lshrrev_b32 v21, 3, v18 // (op 8957) 




s_waitcnt vmcnt(15)
ds_write_b128 v21, v[26:29] // (op 8159) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v21, v[30:33] offset:64 // (op 8159) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v21, v[34:37] offset:128 // (op 8159) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v21, v[38:41] offset:192 // (op 8159) Store local data 

v_lshlrev_b32 v18, 2, v1 // (op 9073) 



s_waitcnt vmcnt(11)
ds_write_b32 v18, v20 offset:16384 // (op 8160) Store local data 

v_and_b32 v18, -8, v1 // (op 8899) 
v_sub_u32 v21, v1, v18 // (op 8899) 

v_lshlrev_b32 v18, 4, v21 // (op 8899) 

v_lshrrev_b32 v21, 3, v1 // (op 8899) 
v_lshl_add_u32 v24, v21, 8, v18 // (op 8899) 





s_waitcnt vmcnt(10)
ds_write_b128 v24, v[42:45] offset:17408 // (op 8161) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v24, v[46:49] offset:17536 // (op 8161) Store local data 

v_and_b32 v18, -4, v1 // (op 9015) 
v_sub_u32 v21, v1, v18 // (op 9015) 

v_and_b32 v18, 4294967292, v1 // (op 9015) 
v_add_u32 v24, v18, v21 // (op 9015) 





s_waitcnt vmcnt(8)
ds_write_b8 v24, v23 offset:25600 // (op 8162) Store local data 

v_and_b32 v18, -8, v1 // (op 8870) 
v_sub_u32 v21, v1, v18 // (op 8870) 

v_lshlrev_b32 v18, 4, v21 // (op 8870) 

v_lshrrev_b32 v21, 3, v1 // (op 8870) 
v_lshl_add_u32 v24, v21, 8, v18 // (op 8870) 


v_and_b32 v18, -4, v1 // (op 8928) 
v_sub_u32 v21, v1, v18 // (op 8928) 

v_lshlrev_b32 v18, 5, v21 // (op 8928) 

v_lshrrev_b32 v21, 2, v1 // (op 8928) 
v_lshl_add_u32 v25, v21, 9, v18 // (op 8928) 


v_lshlrev_b32 v18, 2, v25 // (op 8928) 

v_lshrrev_b32 v21, 3, v18 // (op 8928) 

v_and_b32 v18, -4, v1 // (op 8986) 
v_sub_u32 v25, v1, v18 // (op 8986) 

v_and_b32 v18, 4294967292, v1 // (op 8986) 
v_add_u32 v50, v18, v25 // (op 8986) 


v_lshlrev_b32 v18, 2, v1 // (op 9044) 
v_and_b32 v25, -8, v1 // (op 9125) 
v_sub_u32 v51, v1, v25 // (op 9125) 

v_lshlrev_b32 v25, 4, v51 // (op 9125) 

v_lshrrev_b32 v51, 3, v1 // (op 9125) 
v_lshl_add_u32 v76, v51, 8, v25 // (op 9125) 


v_and_b32 v25, -4, v1 // (op 9154) 
v_sub_u32 v51, v1, v25 // (op 9154) 

v_lshlrev_b32 v25, 5, v51 // (op 9154) 

v_lshrrev_b32 v51, 2, v1 // (op 9154) 
v_lshl_add_u32 v77, v51, 9, v25 // (op 9154) 


v_lshlrev_b32 v25, 2, v77 // (op 9154) 

v_lshrrev_b32 v51, 3, v25 // (op 9154) 

v_and_b32 v25, -4, v1 // (op 9183) 
v_sub_u32 v77, v1, v25 // (op 9183) 

v_and_b32 v25, 4294967292, v1 // (op 9183) 
v_add_u32 v78, v25, v77 // (op 9183) 


v_lshlrev_b32 v25, 2, v1 // (op 9212) 
s_mov_b32 s47, 0 // (op 1394) call()
s_ashr_i64 s[0:1], s[42:43], 7 // (op 1393) 
s_mov_b32 s18, s0 // (op 1393) convert

s_and_b32 s19, s18, 4294967294 // (op 1393) 

s_cmp_lt_u32 s47, s19 // (op 1393) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_1393 // (op 1393) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_1393 if false)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_1393:

s_barrier  // (op 9316) 
s_add_u32 s47, s47, 2 // (op 1395) 



s_lshl_b64 s[0:1], s[30:31], 2 // (op 10) 
v_lshrrev_b64 v[80:81], 3, s[0:1] // (op 10) 


v_mov_b32 v77, v17 // (op 10) 
v_mov_b32 v79, v77 // (op 10) 
buffer_load_dwordx4 v[26:29], v79, s[60:63], 0 offen offset:128 // (op 10) Load value
v_add_u32 v79, v79, v80 // (op 10) 
buffer_load_dwordx4 v[30:33], v79, s[60:63], 0 offen offset:128 // (op 10) Load value
v_add_u32 v79, v79, v80 // (op 10) 
ds_read_b128 v[82:85], v3 offset:17408 // (op 70) Load local data 
ds_read_b128 v[86:89], v3 offset:17472 // (op 70) Load local data 
buffer_load_dwordx4 v[34:37], v79, s[60:63], 0 offen offset:128 // (op 10) Load value
s_barrier  // (op 9329) 
v_add_u32 v79, v79, v80 // (op 10) 



ds_read_b128 v[90:93], v4 // (op 75) Load local data 
buffer_load_dwordx4 v[38:41], v79, s[60:63], 0 offen offset:128 // (op 10) Load value




s_barrier  // (op 9343) 
v_mov_b32 v79, v22 // (op 25) 




buffer_load_dword v20, v79, s[68:71], 0 offen offset:8 // (op 25) Load value
ds_read_u8 v80, v0 offset:25600 // (op 80) Load local data 

s_barrier  // (op 9357) 
v_mov_b64 v[94:95], s[26:27] // (op 4) convert




ds_read_u8 v81, v5 offset:16384 // (op 85) Load local data 
v_mov_b32 v96, v16 // (op 4) 



v_mov_b32 v97, v96 // (op 4) 
ds_read_b128 v[98:101], v4 offset:2048 // (op 371) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:89], v[90:93], a[0:3], v80, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 88) 



ds_read_u8 v102, v5 offset:16512 // (op 375) Load local data 
buffer_load_dwordx4 v[42:45], v97, s[48:51], 0 offen offset:256 // (op 4) Load value



v_add_u32 v97, v97, v94 // (op 4) 
ds_read_b128 v[104:107], v4 offset:4096 // (op 547) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:89], v[98:101], a[4:7], v80, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 376) 



ds_read_u8 v103, v5 offset:16640 // (op 551) Load local data 
buffer_load_dwordx4 v[46:49], v97, s[48:51], 0 offen offset:256 // (op 4) Load value







v_mov_b32 v97, v19 // (op 17) 

ds_read_b128 v[108:111], v4 offset:6144 // (op 723) Load local data 
buffer_load_ubyte v23, v97, s[64:67], 0 offen offset:8 // (op 17) Load value



s_barrier  // (op 9380) 
ds_read_u8 v94, v5 offset:16768 // (op 727) Load local data 






ds_read_b128 v[112:115], v4 offset:8192 // (op 899) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[82:89], v[104:107], a[8:11], v80, v103 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 552) 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[82:89], v[108:111], a[12:15], v80, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 728) 
ds_read_u8 v95, v5 offset:16896 // (op 903) Load local data 
s_waitcnt vmcnt(15)
ds_write_b128 v51, v[52:55] offset:25856 // (op 253) Store local data 



ds_read_b128 v[116:119], v4 offset:10240 // (op 1075) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[82:89], v[112:115], a[16:19], v80, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 904) 



ds_read_u8 v120, v5 offset:17024 // (op 1079) Load local data 
s_waitcnt vmcnt(14)
ds_write_b128 v51, v[56:59] offset:25920 // (op 253) Store local data 



ds_read_b128 v[122:125], v4 offset:12288 // (op 1251) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[82:89], v[116:119], a[20:23], v80, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1080) 



ds_read_u8 v121, v5 offset:17152 // (op 1255) Load local data 
s_waitcnt vmcnt(13)
ds_write_b128 v51, v[60:63] offset:25984 // (op 253) Store local data 



ds_read_b128 v[126:129], v4 offset:14336 // (op 1427) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[82:89], v[122:125], a[24:27], v80, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1256) 



ds_read_u8 v130, v5 offset:17280 // (op 1431) Load local data 
s_waitcnt vmcnt(12)
ds_write_b128 v51, v[64:67] offset:26048 // (op 253) Store local data 



ds_read_b128 v[132:135], v3 offset:21504 // (op 1996) Load local data 
s_barrier  // (op 9392) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[82:89], v[126:129], a[28:31], v80, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1432) 





ds_read_b128 v[136:139], v3 offset:21568 // (op 1996) Load local data 
s_waitcnt vmcnt(11)
ds_write_b32 v25, v14 offset:42240 // (op 255) Store local data 



ds_read_u8 v80, v0 offset:25728 // (op 1999) Load local data 
s_barrier  // (op 9374) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[132:139], v[90:93], a[32:35], v80, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2323) 





s_waitcnt vmcnt(10)
ds_write_b128 v76, v[68:71] offset:43264 // (op 243) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[132:139], v[98:101], a[36:39], v80, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2277) 
s_waitcnt vmcnt(9)
ds_write_b128 v76, v[72:75] offset:43392 // (op 243) Store local data 


s_barrier  // (op 9386) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[132:139], v[104:107], a[40:43], v80, v103 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2231) 





s_waitcnt vmcnt(8)
ds_write_b8 v78, v15 offset:51456 // (op 254) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[132:139], v[108:111], a[44:47], v80, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2185) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[132:139], v[112:115], a[48:51], v80, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2139) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[132:139], v[116:119], a[52:55], v80, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2093) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:139], v[122:125], a[56:59], v80, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2047) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:139], v[126:129], a[60:63], v80, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2001) 





s_lshl_b64 s[0:1], s[30:31], 2 // (op 256) 
v_lshrrev_b64 v[80:81], 3, s[0:1] // (op 256) 


v_mov_b32 v82, v77 // (op 256) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 9371) 
buffer_load_dwordx4 v[52:55], v82, s[60:63], 0 offen offset:192 // (op 256) Load value



v_add_u32 v82, v82, v80 // (op 256) 
ds_read_b128 v[88:91], v7 offset:43264 // (op 245) Load local data 
ds_read_b128 v[92:95], v7 offset:43328 // (op 245) Load local data 
buffer_load_dwordx4 v[56:59], v82, s[60:63], 0 offen offset:192 // (op 256) Load value
s_barrier  // (op 9377) 
v_add_u32 v82, v82, v80 // (op 256) 



buffer_load_dwordx4 v[60:63], v82, s[60:63], 0 offen offset:192 // (op 256) Load value
v_add_u32 v82, v82, v80 // (op 256) 
ds_read_b128 v[84:87], v8 offset:25856 // (op 247) Load local data 
buffer_load_dwordx4 v[64:67], v82, s[60:63], 0 offen offset:192 // (op 256) Load value



s_waitcnt lgkmcnt(3)
s_barrier  // (op 9383) 






v_add_u32 v17, 128, v17 // (op 8691) 
ds_read_u8 v77, v6 offset:51456 // (op 249) Load local data 
buffer_load_dword v14, v79, s[68:71], 0 offen offset:12 // (op 258) Load value

s_barrier  // (op 9389) 




v_add_u32 v22, 8, v22 // (op 8819) 
ds_read_u8 v79, v9 offset:42240 // (op 251) Load local data 
v_mov_b64 v[80:81], s[26:27] // (op 242) convert




v_mov_b32 v82, v96 // (op 242) 
buffer_load_dwordx4 v[68:71], v82, s[48:51], 0 offen offset:384 // (op 242) Load value
v_add_u32 v82, v82, v80 // (op 242) 
ds_read_b128 v[98:101], v8 offset:27904 // (op 381) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:95], v[84:87], a[0:3], v77, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 252) 



ds_read_u8 v83, v9 offset:42368 // (op 385) Load local data 
buffer_load_dwordx4 v[72:75], v82, s[48:51], 0 offen offset:384 // (op 242) Load value






ds_read_b128 v[102:105], v8 offset:29952 // (op 557) Load local data 






v_add_u32 v16, 256, v16 // (op 8627) 
ds_read_u8 v96, v9 offset:42496 // (op 561) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:95], v[98:101], a[4:7], v77, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 377) 



ds_read_b128 v[106:109], v8 offset:32000 // (op 733) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:95], v[102:105], a[8:11], v77, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 553) 



ds_read_u8 v80, v9 offset:42624 // (op 737) Load local data 
buffer_load_ubyte v15, v97, s[64:67], 0 offen offset:12 // (op 257) Load value




ds_read_b128 v[110:113], v8 offset:34048 // (op 909) Load local data 



s_barrier  // (op 9332) 
v_add_u32 v19, 8, v19 // (op 8755) 



ds_read_u8 v97, v9 offset:42752 // (op 913) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:95], v[106:109], a[12:15], v77, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 729) 



ds_read_b128 v[114:117], v8 offset:36096 // (op 1085) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:95], v[110:113], a[16:19], v77, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 905) 



ds_read_u8 v81, v9 offset:42880 // (op 1089) Load local data 
s_waitcnt vmcnt(15)
ds_write_b128 v21, v[26:29] // (op 13) Store local data 



ds_read_b128 v[118:121], v8 offset:38144 // (op 1261) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:95], v[114:117], a[20:23], v77, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1081) 



ds_read_u8 v82, v9 offset:43008 // (op 1265) Load local data 
s_waitcnt vmcnt(14)
ds_write_b128 v21, v[30:33] offset:64 // (op 13) Store local data 



ds_read_b128 v[122:125], v8 offset:40192 // (op 1437) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[88:95], v[118:121], a[24:27], v77, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1257) 
s_waitcnt vmcnt(13)
ds_write_b128 v21, v[34:37] offset:128 // (op 13) Store local data 



s_barrier  // (op 9451) 
ds_read_u8 v126, v9 offset:43136 // (op 1441) Load local data 
s_waitcnt vmcnt(12)
ds_write_b128 v21, v[38:41] offset:192 // (op 13) Store local data 



s_barrier  // (op 9455) 
s_barrier  // (op 9360) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[88:95], v[122:125], a[28:31], v77, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1433) 





ds_read_b128 v[88:91], v7 offset:47360 // (op 2004) Load local data 
ds_read_b128 v[92:95], v7 offset:47424 // (op 2004) Load local data 
s_waitcnt vmcnt(11)
ds_write_b32 v18, v20 offset:16384 // (op 30) Store local data 



s_barrier  // (op 9485) 
s_barrier  // (op 9319) 
ds_read_u8 v77, v6 offset:51584 // (op 2007) Load local data 



s_waitcnt vmcnt(10)
ds_write_b128 v24, v[42:45] offset:17408 // (op 6) Store local data 
s_barrier  // (op 9489) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:95], v[84:87], a[32:35], v77, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2324) 
s_waitcnt vmcnt(9)
ds_write_b128 v24, v[46:49] offset:17536 // (op 6) Store local data 


s_barrier  // (op 9346) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:95], v[98:101], a[36:39], v77, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2278) 





s_waitcnt vmcnt(8)
ds_write_b8 v50, v23 offset:25600 // (op 21) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:95], v[102:105], a[40:43], v77, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2232) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:95], v[106:109], a[44:47], v77, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2186) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[88:95], v[110:113], a[48:51], v77, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2140) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[88:95], v[114:117], a[52:55], v77, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2094) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[88:95], v[118:121], a[56:59], v77, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2048) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[88:95], v[122:125], a[60:63], v77, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2002) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 1393) 
s_mov_b32 s18, s0 // (op 1393) convert

s_and_b32 s19, s18, 4294967294 // (op 1393) 

s_cmp_lt_u32 s47, s19 // (op 1393) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_1393 // (op 1393) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_1393 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_1393:

















s_mul_hi_i32 s47, s16, s10 // (op 8575) 
v_lshrrev_b32 v16, 3, v1 // (op 8575) 
s_add_i32 s0, s47, s16 // (op 8575) 

v_lshlrev_b32 v17, 1, v16 // (op 8575) 

s_ashr_i32 s1, s0, 31 // (op 8575) 
v_and_b32 v16, -8, v1 // (op 8575) 
s_cmp_eq_i32 s10, 0 // (op 8575) 
s_mov_b32 s47, scc // (op 8575) 
v_sub_u32 v18, v1, v16 // (op 8575) 

s_cmp_eq_i32 s47, 0 // (op 8575) 
s_cselect_b32 s18, 0, -1 // (op 8575) 

v_lshlrev_b32 v16, 4, v18 // (op 8575) 

s_and_b32 s19, s13, 31 // (op 8575) 
s_lshl_b32 s47, 1, s19 // (op 8575) 
s_add_i32 s22, s47, s18 // (op 8575) 


s_and_b32 s18, s1, s22 // (op 8575) 


s_add_i32 s1, s0, s18 // (op 8575) 


s_ashr_i32 s0, s1, s19 // (op 8575) 


s_lshl_b32 s1, s13, 24 // (op 8575) 
s_ashr_i32 s47, s1, 31 // (op 8575) 

s_xor_b32 s1, s0, s47 // (op 8575) 

s_sub_i32 s0, s1, s47 // (op 8575) 


s_mul_hi_i32 s1, s0, s14 // (op 8575) 
s_add_i32 s47, s1, s0 // (op 8575) 

s_ashr_i32 s1, s47, 31 // (op 8575) 
s_cmp_eq_i32 s14, 0 // (op 8575) 
s_mov_b32 s18, scc // (op 8575) 
s_cmp_eq_i32 s18, 0 // (op 8575) 
s_cselect_b32 s19, 0, -1 // (op 8575) 

s_and_b32 s18, s15, 31 // (op 8575) 
s_lshl_b32 s22, 1, s18 // (op 8575) 
s_add_i32 s23, s22, s19 // (op 8575) 


s_and_b32 s19, s1, s23 // (op 8575) 


s_add_i32 s1, s47, s19 // (op 8575) 


s_ashr_i32 s19, s1, s18 // (op 8575) 


s_lshl_b32 s1, s15, 24 // (op 8575) 
s_ashr_i32 s18, s1, 31 // (op 8575) 

s_xor_b32 s1, s19, s18 // (op 8575) 

s_sub_i32 s19, s1, s18 // (op 8575) 


s_mul_i32 s1, s0, s11 // (op 8575) 

s_sub_u32 s0, s16, s1 // (op 8575) 

s_mul_i32 s1, s19, s11 // (op 8575) 
s_add_u32 s1, s1, s0 // (op 8575) 


s_mul_hi_i32 s0, s16, s9 // (op 8575) 
s_add_i32 s47, s0, s16 // (op 8575) 

s_ashr_i32 s0, s47, 31 // (op 8575) 
s_cmp_eq_i32 s9, 0 // (op 8575) 
s_mov_b32 s18, scc // (op 8575) 
s_cmp_eq_i32 s18, 0 // (op 8575) 
s_cselect_b32 s19, 0, -1 // (op 8575) 

s_and_b32 s18, s45, 31 // (op 8575) 
s_lshl_b32 s22, 1, s18 // (op 8575) 
s_add_i32 s23, s22, s19 // (op 8575) 


s_and_b32 s19, s0, s23 // (op 8575) 


s_add_i32 s0, s47, s19 // (op 8575) 


s_ashr_i32 s19, s0, s18 // (op 8575) 


s_lshl_b32 s0, s45, 24 // (op 8575) 
s_ashr_i32 s18, s0, 31 // (op 8575) 

s_xor_b32 s0, s19, s18 // (op 8575) 

s_sub_i32 s19, s0, s18 // (op 8575) 


s_add_u32 s22, s6, 256 // (op 8575) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8575) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8575) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8575) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8575) 

s_mul_i32 s0, s11, s22 // (op 8575) 

s_mul_i32 s18, s8, s0 // (op 8575) 

s_mul_i32 s0, s19, s18 // (op 8575) 

s_sub_u32 s18, s16, s0 // (op 8575) 

s_mov_b32 s0, s18 // (op 8575) 
s_mul_hi_i32 s47, s0, s12 // (op 8575) 
s_add_i32 s22, s47, s0 // (op 8575) 


s_ashr_i32 s0, s22, 31 // (op 8575) 
s_cmp_eq_i32 s12, 0 // (op 8575) 
s_mov_b32 s47, scc // (op 8575) 
s_cmp_eq_i32 s47, 0 // (op 8575) 
s_cselect_b32 s23, 0, -1 // (op 8575) 

s_and_b32 s47, s44, 31 // (op 8575) 
s_lshl_b32 s24, 1, s47 // (op 8575) 
s_add_i32 s25, s24, s23 // (op 8575) 


s_and_b32 s23, s0, s25 // (op 8575) 


s_add_i32 s0, s22, s23 // (op 8575) 


s_ashr_i32 s22, s0, s47 // (op 8575) 


s_lshl_b32 s0, s44, 24 // (op 8575) 
s_ashr_i32 s47, s0, 31 // (op 8575) 

s_xor_b32 s0, s22, s47 // (op 8575) 

s_sub_i32 s22, s0, s47 // (op 8575) 


s_mul_i32 s0, s46, s11 // (op 8575) 
s_add_u32 s24, s20, 64 // (op 8575) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8575) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 8575) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 8575) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 8575) 

s_sub_i32 s23, s24, s0 // (op 8575) 

s_mul_i32 s47, s22, s23 // (op 8575) 


s_sub_u32 s22, s18, s47 // (op 8575) 


s_add_u32 s18, s22, s0 // (op 8575) 


s_cmp_eq_i32 s19, 0 // (op 8575) 
s_mov_b32 s0, scc // (op 8575) 

s_cmp_eq_i32 s0, 0 // (op 8575) 
s_cselect_b32 s19, s18, s1 // (op 8575) 



v_lshl_add_u32 v18, s19, 6, v17 // (op 8575) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 8575) 
s_lshr_b32 s47, s0, 1 // (op 8575) 

v_lshl_add_u32 v17, s47, 8, v16 // (op 8575) 


v_mul_lo_u32 v16, v18, s26 // (op 8575) 
v_add_u32 v16, v16, v17 // (op 8575) 


s_mul_i32 s1, s26, s21 // (op 8577) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 8577) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 8577) least significant
s_mul_hi_u32 s18, s26, s20 // (op 8577) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8577) most significant: sum
s_add_u32 s1, s1, s18 // (op 8577) most significant: sum


s_mov_b32 s48, s28 // (op 8577) reg expression
s_mov_b32 s49, s29 // (op 8577) reg expression
s_mov_b32 s50, s0 // (op 8577) reg expression
s_mov_b32 s51, 131072 // (op 8577) call()

s_mul_hi_i32 s47, s16, s10 // (op 8657) 
v_and_b32 v17, 4294967292, v1 // (op 8657) 
s_add_i32 s0, s47, s16 // (op 8657) 

v_and_b32 v18, -4, v1 // (op 8657) 
s_ashr_i32 s1, s0, 31 // (op 8657) 
v_sub_u32 v19, v1, v18 // (op 8657) 

s_cmp_eq_i32 s10, 0 // (op 8657) 
s_mov_b32 s47, scc // (op 8657) 
v_lshlrev_b32 v18, 5, v19 // (op 8657) 

s_cmp_eq_i32 s47, 0 // (op 8657) 
s_cselect_b32 s18, 0, -1 // (op 8657) 

s_and_b32 s19, s13, 31 // (op 8657) 
s_lshl_b32 s47, 1, s19 // (op 8657) 
s_add_i32 s22, s47, s18 // (op 8657) 


s_and_b32 s18, s1, s22 // (op 8657) 


s_add_i32 s1, s0, s18 // (op 8657) 


s_ashr_i32 s0, s1, s19 // (op 8657) 


s_lshl_b32 s1, s13, 24 // (op 8657) 
s_ashr_i32 s47, s1, 31 // (op 8657) 

s_xor_b32 s1, s0, s47 // (op 8657) 

s_sub_i32 s0, s1, s47 // (op 8657) 


s_mul_hi_i32 s1, s0, s14 // (op 8657) 
s_add_i32 s47, s1, s0 // (op 8657) 

s_ashr_i32 s1, s47, 31 // (op 8657) 
s_cmp_eq_i32 s14, 0 // (op 8657) 
s_mov_b32 s18, scc // (op 8657) 
s_cmp_eq_i32 s18, 0 // (op 8657) 
s_cselect_b32 s19, 0, -1 // (op 8657) 

s_and_b32 s18, s15, 31 // (op 8657) 
s_lshl_b32 s22, 1, s18 // (op 8657) 
s_add_i32 s23, s22, s19 // (op 8657) 


s_and_b32 s19, s1, s23 // (op 8657) 


s_add_i32 s1, s47, s19 // (op 8657) 


s_ashr_i32 s19, s1, s18 // (op 8657) 


s_lshl_b32 s1, s15, 24 // (op 8657) 
s_ashr_i32 s18, s1, 31 // (op 8657) 

s_xor_b32 s1, s19, s18 // (op 8657) 

s_sub_i32 s19, s1, s18 // (op 8657) 


s_add_u32 s22, s6, 256 // (op 8657) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8657) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8657) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8657) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8657) 

s_mul_i32 s1, s19, s22 // (op 8657) 

s_sub_i32 s47, s0, s1 // (op 8657) 


s_mul_hi_i32 s0, s16, s9 // (op 8657) 
s_add_i32 s1, s0, s16 // (op 8657) 

s_ashr_i32 s0, s1, 31 // (op 8657) 
s_cmp_eq_i32 s9, 0 // (op 8657) 
s_mov_b32 s18, scc // (op 8657) 
s_cmp_eq_i32 s18, 0 // (op 8657) 
s_cselect_b32 s19, 0, -1 // (op 8657) 

s_and_b32 s18, s45, 31 // (op 8657) 
s_lshl_b32 s24, 1, s18 // (op 8657) 
s_add_i32 s25, s24, s19 // (op 8657) 


s_and_b32 s19, s0, s25 // (op 8657) 


s_add_i32 s0, s1, s19 // (op 8657) 


s_ashr_i32 s1, s0, s18 // (op 8657) 


s_lshl_b32 s0, s45, 24 // (op 8657) 
s_ashr_i32 s18, s0, 31 // (op 8657) 

s_xor_b32 s0, s1, s18 // (op 8657) 

s_sub_i32 s1, s0, s18 // (op 8657) 


s_mul_i32 s0, s11, s22 // (op 8657) 

s_mul_i32 s18, s8, s0 // (op 8657) 

s_mul_i32 s0, s1, s18 // (op 8657) 

s_sub_u32 s18, s16, s0 // (op 8657) 

s_mul_hi_i32 s0, s18, s12 // (op 8657) 
s_add_i32 s19, s0, s18 // (op 8657) 


s_ashr_i32 s0, s19, 31 // (op 8657) 
s_cmp_eq_i32 s12, 0 // (op 8657) 
s_mov_b32 s18, scc // (op 8657) 
s_cmp_eq_i32 s18, 0 // (op 8657) 
s_cselect_b32 s22, 0, -1 // (op 8657) 

s_and_b32 s18, s44, 31 // (op 8657) 
s_lshl_b32 s23, 1, s18 // (op 8657) 
s_add_i32 s24, s23, s22 // (op 8657) 


s_and_b32 s25, s0, s24 // (op 8657) 


s_add_i32 s0, s19, s25 // (op 8657) 


s_ashr_i32 s19, s0, s18 // (op 8657) 


s_lshl_b32 s0, s44, 24 // (op 8657) 
s_ashr_i32 s18, s0, 31 // (op 8657) 

s_xor_b32 s0, s19, s18 // (op 8657) 

s_sub_i32 s19, s0, s18 // (op 8657) 


s_cmp_eq_i32 s1, 0 // (op 8657) 
s_mov_b32 s0, scc // (op 8657) 

s_cmp_eq_i32 s0, 0 // (op 8657) 
s_cselect_b32 s1, s19, s47 // (op 8657) 



v_lshl_add_u32 v19, s1, 8, v17 // (op 8657) 


v_mul_lo_u32 v21, v19, s31 // (op 8657) most significant: low of low * high
v_mul_hi_u32 v17, v19, s30 // (op 8657) most significant: high of low * low
v_mul_lo_u32 v20, v19, s30 // (op 8657) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 8657) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 8657) 
s_lshr_b32 s47, s0, 1 // (op 8657) 

v_lshl_add_u32 v17, s47, 8, v18 // (op 8657) 


v_mov_b32 v19, 0 // (op 8657) convert
v_mov_b32 v18, v17 // (op 8657) convert

v_add_co_u32 v22, s[0:1], v18, v20 // (op 8657) least significant half
v_addc_co_u32 v23, s[0:1], v19, v21, s[0:1] // (op 8657) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 8657) 


v_lshrrev_b64 v[20:21], 3, v[22:23] // (op 8657) 

v_mov_b32 v17, v20 // (op 8657) convert

s_mul_i32 s1, s30, s7 // (op 8659) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 8659) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 8659) least significant
s_mul_hi_u32 s18, s30, s6 // (op 8659) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8659) most significant: sum
s_add_u32 s1, s1, s18 // (op 8659) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 8659) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 8659) 

s_mov_b32 s60, s32 // (op 8659) reg expression
s_mov_b32 s61, s33 // (op 8659) reg expression
s_mov_b32 s62, s0 // (op 8659) reg expression
s_mov_b32 s63, 131072 // (op 8659) call()

s_mul_hi_i32 s47, s16, s10 // (op 8721) 
v_lshrrev_b32 v18, 2, v1 // (op 8721) 
s_add_i32 s0, s47, s16 // (op 8721) 

v_and_b32 v19, -4, v1 // (op 8721) 
s_ashr_i32 s1, s0, 31 // (op 8721) 
v_sub_u32 v20, v1, v19 // (op 8721) 

s_cmp_eq_i32 s10, 0 // (op 8721) 
s_mov_b32 s47, scc // (op 8721) 
s_cmp_eq_i32 s47, 0 // (op 8721) 
s_cselect_b32 s18, 0, -1 // (op 8721) 

s_and_b32 s19, s13, 31 // (op 8721) 
s_lshl_b32 s47, 1, s19 // (op 8721) 
s_add_i32 s22, s47, s18 // (op 8721) 


s_and_b32 s18, s1, s22 // (op 8721) 


s_add_i32 s1, s0, s18 // (op 8721) 


s_ashr_i32 s0, s1, s19 // (op 8721) 


s_lshl_b32 s1, s13, 24 // (op 8721) 
s_ashr_i32 s47, s1, 31 // (op 8721) 

s_xor_b32 s1, s0, s47 // (op 8721) 

s_sub_i32 s0, s1, s47 // (op 8721) 


s_mul_hi_i32 s1, s0, s14 // (op 8721) 
s_add_i32 s47, s1, s0 // (op 8721) 

s_ashr_i32 s1, s47, 31 // (op 8721) 
s_cmp_eq_i32 s14, 0 // (op 8721) 
s_mov_b32 s18, scc // (op 8721) 
s_cmp_eq_i32 s18, 0 // (op 8721) 
s_cselect_b32 s19, 0, -1 // (op 8721) 

s_and_b32 s18, s15, 31 // (op 8721) 
s_lshl_b32 s22, 1, s18 // (op 8721) 
s_add_i32 s23, s22, s19 // (op 8721) 


s_and_b32 s19, s1, s23 // (op 8721) 


s_add_i32 s1, s47, s19 // (op 8721) 


s_ashr_i32 s19, s1, s18 // (op 8721) 


s_lshl_b32 s1, s15, 24 // (op 8721) 
s_ashr_i32 s18, s1, 31 // (op 8721) 

s_xor_b32 s1, s19, s18 // (op 8721) 

s_sub_i32 s19, s1, s18 // (op 8721) 


s_mul_i32 s1, s0, s11 // (op 8721) 

s_sub_u32 s0, s16, s1 // (op 8721) 

s_mul_i32 s1, s19, s11 // (op 8721) 
s_add_u32 s1, s1, s0 // (op 8721) 


s_mul_hi_i32 s0, s16, s9 // (op 8721) 
s_add_i32 s47, s0, s16 // (op 8721) 

s_ashr_i32 s0, s47, 31 // (op 8721) 
s_cmp_eq_i32 s9, 0 // (op 8721) 
s_mov_b32 s18, scc // (op 8721) 
s_cmp_eq_i32 s18, 0 // (op 8721) 
s_cselect_b32 s19, 0, -1 // (op 8721) 

s_and_b32 s18, s45, 31 // (op 8721) 
s_lshl_b32 s22, 1, s18 // (op 8721) 
s_add_i32 s23, s22, s19 // (op 8721) 


s_and_b32 s19, s0, s23 // (op 8721) 


s_add_i32 s0, s47, s19 // (op 8721) 


s_ashr_i32 s19, s0, s18 // (op 8721) 


s_lshl_b32 s0, s45, 24 // (op 8721) 
s_ashr_i32 s18, s0, 31 // (op 8721) 

s_xor_b32 s0, s19, s18 // (op 8721) 

s_sub_i32 s19, s0, s18 // (op 8721) 


s_add_u32 s22, s6, 256 // (op 8721) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8721) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8721) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8721) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8721) 

s_mul_i32 s0, s11, s22 // (op 8721) 

s_mul_i32 s18, s8, s0 // (op 8721) 

s_mul_i32 s0, s19, s18 // (op 8721) 

s_sub_u32 s18, s16, s0 // (op 8721) 

s_mov_b32 s0, s18 // (op 8721) 
s_mul_hi_i32 s47, s0, s12 // (op 8721) 
s_add_i32 s22, s47, s0 // (op 8721) 


s_ashr_i32 s0, s22, 31 // (op 8721) 
s_cmp_eq_i32 s12, 0 // (op 8721) 
s_mov_b32 s47, scc // (op 8721) 
s_cmp_eq_i32 s47, 0 // (op 8721) 
s_cselect_b32 s23, 0, -1 // (op 8721) 

s_and_b32 s47, s44, 31 // (op 8721) 
s_lshl_b32 s24, 1, s47 // (op 8721) 
s_add_i32 s25, s24, s23 // (op 8721) 


s_and_b32 s23, s0, s25 // (op 8721) 


s_add_i32 s0, s22, s23 // (op 8721) 


s_ashr_i32 s22, s0, s47 // (op 8721) 


s_lshl_b32 s0, s44, 24 // (op 8721) 
s_ashr_i32 s47, s0, 31 // (op 8721) 

s_xor_b32 s0, s22, s47 // (op 8721) 

s_sub_i32 s22, s0, s47 // (op 8721) 


s_mul_i32 s0, s46, s11 // (op 8721) 
s_add_u32 s24, s20, 64 // (op 8721) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8721) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 8721) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 8721) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 8721) 

s_sub_i32 s23, s24, s0 // (op 8721) 

s_mul_i32 s47, s22, s23 // (op 8721) 


s_sub_u32 s22, s18, s47 // (op 8721) 


s_add_u32 s18, s22, s0 // (op 8721) 


s_cmp_eq_i32 s19, 0 // (op 8721) 
s_mov_b32 s0, scc // (op 8721) 

s_cmp_eq_i32 s0, 0 // (op 8721) 
s_cselect_b32 s19, s18, s1 // (op 8721) 



v_lshl_add_u32 v19, s19, 6, v18 // (op 8721) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 8721) 
s_lshr_b32 s47, s0, 1 // (op 8721) 

v_lshl_add_u32 v18, s47, 3, v20 // (op 8721) 


v_mul_lo_u32 v20, v19, s34 // (op 8721) 
v_add_u32 v20, v20, v18 // (op 8721) 


s_mul_i32 s1, s34, s21 // (op 8723) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 8723) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 8723) least significant
s_mul_hi_u32 s18, s34, s20 // (op 8723) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8723) most significant: sum
s_add_u32 s1, s1, s18 // (op 8723) most significant: sum


s_mov_b32 s64, s36 // (op 8723) reg expression
s_mov_b32 s65, s37 // (op 8723) reg expression
s_mov_b32 s66, s0 // (op 8723) reg expression
s_mov_b32 s67, 131072 // (op 8723) call()

s_mul_hi_i32 s47, s16, s10 // (op 8785) 
s_add_i32 s0, s47, s16 // (op 8785) 

s_ashr_i32 s1, s0, 31 // (op 8785) 
s_cmp_eq_i32 s10, 0 // (op 8785) 
s_mov_b32 s47, scc // (op 8785) 
s_cmp_eq_i32 s47, 0 // (op 8785) 
s_cselect_b32 s18, 0, -1 // (op 8785) 

s_and_b32 s19, s13, 31 // (op 8785) 
s_lshl_b32 s47, 1, s19 // (op 8785) 
s_add_i32 s22, s47, s18 // (op 8785) 


s_and_b32 s18, s1, s22 // (op 8785) 


s_add_i32 s1, s0, s18 // (op 8785) 


s_ashr_i32 s0, s1, s19 // (op 8785) 


s_lshl_b32 s1, s13, 24 // (op 8785) 
s_ashr_i32 s47, s1, 31 // (op 8785) 

s_xor_b32 s1, s0, s47 // (op 8785) 

s_sub_i32 s0, s1, s47 // (op 8785) 


s_mul_hi_i32 s1, s0, s14 // (op 8785) 
s_add_i32 s47, s1, s0 // (op 8785) 

s_ashr_i32 s1, s47, 31 // (op 8785) 
s_cmp_eq_i32 s14, 0 // (op 8785) 
s_mov_b32 s18, scc // (op 8785) 
s_cmp_eq_i32 s18, 0 // (op 8785) 
s_cselect_b32 s19, 0, -1 // (op 8785) 

s_and_b32 s18, s15, 31 // (op 8785) 
s_lshl_b32 s22, 1, s18 // (op 8785) 
s_add_i32 s23, s22, s19 // (op 8785) 


s_and_b32 s19, s1, s23 // (op 8785) 


s_add_i32 s1, s47, s19 // (op 8785) 


s_ashr_i32 s19, s1, s18 // (op 8785) 


s_lshl_b32 s1, s15, 24 // (op 8785) 
s_ashr_i32 s18, s1, 31 // (op 8785) 

s_xor_b32 s1, s19, s18 // (op 8785) 

s_sub_i32 s19, s1, s18 // (op 8785) 


s_add_u32 s22, s6, 256 // (op 8785) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8785) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8785) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8785) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8785) 

s_mul_i32 s1, s19, s22 // (op 8785) 

s_sub_i32 s47, s0, s1 // (op 8785) 


s_mul_hi_i32 s0, s16, s9 // (op 8785) 
s_add_i32 s1, s0, s16 // (op 8785) 

s_ashr_i32 s0, s1, 31 // (op 8785) 
s_cmp_eq_i32 s9, 0 // (op 8785) 
s_mov_b32 s18, scc // (op 8785) 
s_cmp_eq_i32 s18, 0 // (op 8785) 
s_cselect_b32 s19, 0, -1 // (op 8785) 

s_and_b32 s18, s45, 31 // (op 8785) 
s_lshl_b32 s24, 1, s18 // (op 8785) 
s_add_i32 s25, s24, s19 // (op 8785) 


s_and_b32 s19, s0, s25 // (op 8785) 


s_add_i32 s0, s1, s19 // (op 8785) 


s_ashr_i32 s1, s0, s18 // (op 8785) 


s_lshl_b32 s0, s45, 24 // (op 8785) 
s_ashr_i32 s18, s0, 31 // (op 8785) 

s_xor_b32 s0, s1, s18 // (op 8785) 

s_sub_i32 s1, s0, s18 // (op 8785) 


s_mul_i32 s0, s11, s22 // (op 8785) 

s_mul_i32 s18, s8, s0 // (op 8785) 

s_mul_i32 s0, s1, s18 // (op 8785) 

s_sub_u32 s18, s16, s0 // (op 8785) 

s_mul_hi_i32 s0, s18, s12 // (op 8785) 
s_add_i32 s19, s0, s18 // (op 8785) 


s_ashr_i32 s0, s19, 31 // (op 8785) 
s_cmp_eq_i32 s12, 0 // (op 8785) 
s_mov_b32 s18, scc // (op 8785) 
s_cmp_eq_i32 s18, 0 // (op 8785) 
s_cselect_b32 s22, 0, -1 // (op 8785) 

s_and_b32 s18, s44, 31 // (op 8785) 
s_lshl_b32 s23, 1, s18 // (op 8785) 
s_add_i32 s24, s23, s22 // (op 8785) 


s_and_b32 s25, s0, s24 // (op 8785) 


s_add_i32 s0, s19, s25 // (op 8785) 


s_ashr_i32 s19, s0, s18 // (op 8785) 


s_lshl_b32 s0, s44, 24 // (op 8785) 
s_ashr_i32 s18, s0, 31 // (op 8785) 

s_xor_b32 s0, s19, s18 // (op 8785) 

s_sub_i32 s19, s0, s18 // (op 8785) 


s_cmp_eq_i32 s1, 0 // (op 8785) 
s_mov_b32 s0, scc // (op 8785) 

s_cmp_eq_i32 s0, 0 // (op 8785) 
s_cselect_b32 s1, s19, s47 // (op 8785) 



v_lshl_add_u32 v18, s1, 8, v1 // (op 8785) 

v_mul_lo_u32 v19, v18, s38 // (op 8785) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 8785) 
s_lshr_b32 s47, s0, 1 // (op 8785) 

v_lshl_add_u32 v18, s47, 3, v19 // (op 8785) 


s_mul_i32 s1, s38, s7 // (op 8787) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 8787) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 8787) least significant
s_mul_hi_u32 s18, s38, s6 // (op 8787) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8787) most significant: sum
s_add_u32 s1, s1, s18 // (op 8787) most significant: sum


s_mov_b32 s68, s40 // (op 8787) reg expression
s_mov_b32 s69, s41 // (op 8787) reg expression
s_mov_b32 s70, s0 // (op 8787) reg expression
s_mov_b32 s71, 131072 // (op 8787) call()

v_and_b32 v19, -8, v1 // (op 8883) 
v_sub_u32 v21, v1, v19 // (op 8883) 

v_lshlrev_b32 v19, 4, v21 // (op 8883) 

v_lshrrev_b32 v21, 3, v1 // (op 8883) 
v_lshl_add_u32 v22, v21, 8, v19 // (op 8883) 


v_and_b32 v19, -4, v1 // (op 8941) 
v_sub_u32 v21, v1, v19 // (op 8941) 

v_lshlrev_b32 v19, 5, v21 // (op 8941) 

v_lshrrev_b32 v21, 2, v1 // (op 8941) 
v_lshl_add_u32 v23, v21, 9, v19 // (op 8941) 


v_lshlrev_b32 v19, 2, v23 // (op 8941) 

v_lshrrev_b32 v21, 3, v19 // (op 8941) 

v_and_b32 v19, -4, v1 // (op 8999) 
v_sub_u32 v23, v1, v19 // (op 8999) 

v_and_b32 v19, 4294967292, v1 // (op 8999) 
v_add_u32 v24, v19, v23 // (op 8999) 


v_lshlrev_b32 v19, 2, v1 // (op 9057) 
s_ashr_i64 s[0:1], s[42:43], 7 // (op 1968) 
s_and_b32 s47, s0, 4294967294 // (op 1968) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 1967) 
s_mov_b32 s18, s0 // (op 1967) convert

s_cmp_lt_u32 s47, s18 // (op 1967) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_1967 // (op 1967) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_1967 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_1967:

s_add_u32 s47, s47, 1 // (op 1969) 
v_mov_b64 v[14:15], s[26:27] // (op 176) convert

v_mov_b32 v23, v16 // (op 176) 
v_mov_b32 v25, v23 // (op 176) 
buffer_load_dwordx4 v[26:29], v25, s[48:51], 0 offen // (op 176) Load value
v_add_u32 v25, v25, v14 // (op 176) 
buffer_load_dwordx4 v[30:33], v25, s[48:51], 0 offen // (op 176) Load value



v_add_u32 v16, 128, v16 // (op 8565) 

s_barrier  // (op 9322) 
s_lshl_b64 s[0:1], s[30:31], 2 // (op 190) 



v_lshrrev_b64 v[14:15], 3, s[0:1] // (op 190) 


v_mov_b32 v23, v17 // (op 190) 
v_mov_b32 v25, v23 // (op 190) 
buffer_load_dwordx4 v[34:37], v25, s[60:63], 0 offen // (op 190) Load value
v_add_u32 v25, v25, v14 // (op 190) 
s_waitcnt vmcnt(2)
ds_write_b128 v22, v[26:29] offset:17408 // (op 177) Store local data 
buffer_load_dwordx4 v[38:41], v25, s[60:63], 0 offen // (op 190) Load value
v_add_u32 v25, v25, v14 // (op 190) 
buffer_load_dwordx4 v[42:45], v25, s[60:63], 0 offen // (op 190) Load value
v_add_u32 v25, v25, v14 // (op 190) 
buffer_load_dwordx4 v[46:49], v25, s[60:63], 0 offen // (op 190) Load value



v_add_u32 v17, 64, v17 // (op 8649) 
s_waitcnt vmcnt(4)
ds_write_b128 v22, v[30:33] offset:17536 // (op 177) Store local data 



v_mov_b32 v23, v20 // (op 191) 

s_barrier  // (op 9335) 
buffer_load_ubyte v14, v23, s[64:67], 0 offen offset: 0 // (op 191) Load value




v_add_u32 v20, 4, v20 // (op 8713) 
s_waitcnt vmcnt(4)
ds_write_b128 v21, v[34:37] // (op 187) Store local data 


v_mov_b32 v15, v18 // (op 192) 

buffer_load_dword v23, v15, s[68:71], 0 offen // (op 192) Load value

v_add_u32 v18, 4, v18 // (op 8777) 
s_waitcnt vmcnt(4)
ds_write_b128 v21, v[38:41] offset:64 // (op 187) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 9325) 
s_waitcnt vmcnt(3)
ds_write_b128 v21, v[42:45] offset:128 // (op 187) Store local data 



ds_read_b128 v[26:29], v3 offset:17408 // (op 179) Load local data 
ds_read_b128 v[30:33], v3 offset:17472 // (op 179) Load local data 
s_waitcnt vmcnt(2)
ds_write_b128 v21, v[46:49] offset:192 // (op 187) Store local data 

s_barrier  // (op 9349) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 9339) 






ds_read_b128 v[34:37], v4 // (op 181) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v14 offset:25600 // (op 188) Store local data 

s_barrier  // (op 9363) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 9353) 






ds_read_u8 v25, v0 offset:25600 // (op 183) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v19, v23 offset:16384 // (op 189) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 9367) 



ds_read_u8 v23, v5 offset:16384 // (op 185) Load local data 



ds_read_b128 v[38:41], v4 offset:2048 // (op 350) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:37], a[0:3], v25, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 186) 



ds_read_u8 v14, v5 offset:16512 // (op 354) Load local data 



ds_read_b128 v[42:45], v4 offset:4096 // (op 526) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:33], v[38:41], a[4:7], v25, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 355) 



ds_read_u8 v15, v5 offset:16640 // (op 530) Load local data 



ds_read_b128 v[46:49], v4 offset:6144 // (op 702) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:33], v[42:45], a[8:11], v25, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 531) 



ds_read_u8 v50, v5 offset:16768 // (op 706) Load local data 



ds_read_b128 v[52:55], v4 offset:8192 // (op 878) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:33], v[46:49], a[12:15], v25, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 707) 



ds_read_u8 v51, v5 offset:16896 // (op 882) Load local data 



ds_read_b128 v[56:59], v4 offset:10240 // (op 1054) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:33], v[52:55], a[16:19], v25, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 883) 



ds_read_u8 v60, v5 offset:17024 // (op 1058) Load local data 



ds_read_b128 v[62:65], v4 offset:12288 // (op 1230) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:33], v[56:59], a[20:23], v25, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1059) 



ds_read_u8 v61, v5 offset:17152 // (op 1234) Load local data 



ds_read_b128 v[66:69], v4 offset:14336 // (op 1406) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:33], v[62:65], a[24:27], v25, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1235) 



ds_read_u8 v70, v5 offset:17280 // (op 1410) Load local data 



ds_read_b128 v[72:75], v3 offset:21504 // (op 1756) Load local data 
ds_read_b128 v[76:79], v3 offset:21568 // (op 1756) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[26:33], v[66:69], a[28:31], v25, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1411) 





ds_read_u8 v71, v0 offset:25728 // (op 1759) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[72:79], v[34:37], a[32:35], v71, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1761) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[72:79], v[38:41], a[36:39], v71, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1792) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[72:79], v[42:45], a[40:43], v71, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1823) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:79], v[46:49], a[44:47], v71, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1854) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[72:79], v[52:55], a[48:51], v71, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1885) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[72:79], v[56:59], a[52:55], v71, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1916) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[72:79], v[62:65], a[56:59], v71, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1947) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[72:79], v[66:69], a[60:63], v71, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1978) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 1967) 
s_mov_b32 s18, s0 // (op 1967) convert

s_cmp_lt_u32 s47, s18 // (op 1967) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_1967 // (op 1967) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_1967 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_1967:






















s_barrier  // (op 9473) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_133
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 



v_mov_b32 v0, v13 // (op 56) reg expression

buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 56) Store value


v_accvgpr_read v6, a4 // (op 3573) 
v_accvgpr_read v7, a5 // (op 3573) 
v_accvgpr_read v8, a6 // (op 3573) 
v_accvgpr_read v9, a7 // (op 3573) 
v_mul_f32 v14, v11, v6 // (op 3573) 
v_mul_f32 v15, v11, v7 // (op 3573) 
v_mul_f32 v16, v11, v8 // (op 3573) 
v_mul_f32 v17, v11, v9 // (op 3573) 



s_lshl_b32 s0, s4, 7 // (op 3575) 
v_add_u32 v0, v13, s0 // (op 3575) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3575) Store value


v_accvgpr_read v6, a8 // (op 3585) 
v_accvgpr_read v7, a9 // (op 3585) 
v_accvgpr_read v8, a10 // (op 3585) 
v_accvgpr_read v9, a11 // (op 3585) 
v_mul_f32 v14, v11, v6 // (op 3585) 
v_mul_f32 v15, v11, v7 // (op 3585) 
v_mul_f32 v16, v11, v8 // (op 3585) 
v_mul_f32 v17, v11, v9 // (op 3585) 



s_lshl_b32 s0, s4, 8 // (op 3587) 
v_add_u32 v0, v13, s0 // (op 3587) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3587) Store value


v_accvgpr_read v6, a12 // (op 3597) 
v_accvgpr_read v7, a13 // (op 3597) 
v_accvgpr_read v8, a14 // (op 3597) 
v_accvgpr_read v9, a15 // (op 3597) 
v_mul_f32 v14, v11, v6 // (op 3597) 
v_mul_f32 v15, v11, v7 // (op 3597) 
v_mul_f32 v16, v11, v8 // (op 3597) 
v_mul_f32 v17, v11, v9 // (op 3597) 



s_lshl_b32 s0, s4, 7 // (op 3599) 
s_mul_i32 s1, 3, s0 // (op 3599) 

v_add_u32 v0, v13, s1 // (op 3599) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3599) Store value


v_accvgpr_read v6, a16 // (op 3609) 
v_accvgpr_read v7, a17 // (op 3609) 
v_accvgpr_read v8, a18 // (op 3609) 
v_accvgpr_read v9, a19 // (op 3609) 
v_mul_f32 v14, v11, v6 // (op 3609) 
v_mul_f32 v15, v11, v7 // (op 3609) 
v_mul_f32 v16, v11, v8 // (op 3609) 
v_mul_f32 v17, v11, v9 // (op 3609) 



s_lshl_b32 s0, s4, 9 // (op 3611) 
v_add_u32 v0, v13, s0 // (op 3611) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3611) Store value


v_accvgpr_read v6, a20 // (op 3621) 
v_accvgpr_read v7, a21 // (op 3621) 
v_accvgpr_read v8, a22 // (op 3621) 
v_accvgpr_read v9, a23 // (op 3621) 
v_mul_f32 v14, v11, v6 // (op 3621) 
v_mul_f32 v15, v11, v7 // (op 3621) 
v_mul_f32 v16, v11, v8 // (op 3621) 
v_mul_f32 v17, v11, v9 // (op 3621) 



s_lshl_b32 s0, s4, 7 // (op 3623) 
s_mul_i32 s1, 5, s0 // (op 3623) 

v_add_u32 v0, v13, s1 // (op 3623) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3623) Store value


v_accvgpr_read v6, a24 // (op 3633) 
v_accvgpr_read v7, a25 // (op 3633) 
v_accvgpr_read v8, a26 // (op 3633) 
v_accvgpr_read v9, a27 // (op 3633) 
v_mul_f32 v14, v11, v6 // (op 3633) 
v_mul_f32 v15, v11, v7 // (op 3633) 
v_mul_f32 v16, v11, v8 // (op 3633) 
v_mul_f32 v17, v11, v9 // (op 3633) 



s_lshl_b32 s0, s4, 7 // (op 3635) 
s_mul_i32 s1, 6, s0 // (op 3635) 

v_add_u32 v0, v13, s1 // (op 3635) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3635) Store value


v_accvgpr_read v6, a28 // (op 3645) 
v_accvgpr_read v7, a29 // (op 3645) 
v_accvgpr_read v8, a30 // (op 3645) 
v_accvgpr_read v9, a31 // (op 3645) 
v_mul_f32 v14, v11, v6 // (op 3645) 
v_mul_f32 v15, v11, v7 // (op 3645) 
v_mul_f32 v16, v11, v8 // (op 3645) 
v_mul_f32 v17, v11, v9 // (op 3645) 



s_lshl_b32 s0, s4, 7 // (op 3647) 
s_mul_i32 s1, 7, s0 // (op 3647) 

v_add_u32 v0, v13, s1 // (op 3647) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3647) Store value


v_accvgpr_read v6, a32 // (op 3699) 
v_accvgpr_read v7, a33 // (op 3699) 
v_accvgpr_read v8, a34 // (op 3699) 
v_accvgpr_read v9, a35 // (op 3699) 
v_mul_f32 v14, v11, v6 // (op 3699) 
v_mul_f32 v15, v11, v7 // (op 3699) 
v_mul_f32 v16, v11, v8 // (op 3699) 
v_mul_f32 v17, v11, v9 // (op 3699) 



v_add_u32 v0, 128, v13 // (op 3702) 

buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3702) Store value


v_accvgpr_read v6, a36 // (op 3720) 
v_accvgpr_read v7, a37 // (op 3720) 
v_accvgpr_read v8, a38 // (op 3720) 
v_accvgpr_read v9, a39 // (op 3720) 
v_mul_f32 v14, v11, v6 // (op 3720) 
v_mul_f32 v15, v11, v7 // (op 3720) 
v_mul_f32 v16, v11, v8 // (op 3720) 
v_mul_f32 v17, v11, v9 // (op 3720) 



s_lshl_b32 s0, s4, 7 // (op 3704) 
s_add_u32 s0, s0, 128 // (op 3704) 
v_add_u32 v0, v13, s0 // (op 3704) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3704) Store value


v_accvgpr_read v6, a40 // (op 3725) 
v_accvgpr_read v7, a41 // (op 3725) 
v_accvgpr_read v8, a42 // (op 3725) 
v_accvgpr_read v9, a43 // (op 3725) 
v_mul_f32 v14, v11, v6 // (op 3725) 
v_mul_f32 v15, v11, v7 // (op 3725) 
v_mul_f32 v16, v11, v8 // (op 3725) 
v_mul_f32 v17, v11, v9 // (op 3725) 



s_lshl_b32 s0, s4, 8 // (op 3706) 
s_add_u32 s0, s0, 128 // (op 3706) 
v_add_u32 v0, v13, s0 // (op 3706) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3706) Store value


v_accvgpr_read v6, a44 // (op 3730) 
v_accvgpr_read v7, a45 // (op 3730) 
v_accvgpr_read v8, a46 // (op 3730) 
v_accvgpr_read v9, a47 // (op 3730) 
v_mul_f32 v14, v11, v6 // (op 3730) 
v_mul_f32 v15, v11, v7 // (op 3730) 
v_mul_f32 v16, v11, v8 // (op 3730) 
v_mul_f32 v17, v11, v9 // (op 3730) 



s_lshl_b32 s0, s4, 7 // (op 3708) 
s_mul_i32 s1, 3, s0 // (op 3708) 
s_add_u32 s1, s1, 128 // (op 3708) 

v_add_u32 v0, v13, s1 // (op 3708) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3708) Store value


v_accvgpr_read v6, a48 // (op 3735) 
v_accvgpr_read v7, a49 // (op 3735) 
v_accvgpr_read v8, a50 // (op 3735) 
v_accvgpr_read v9, a51 // (op 3735) 
v_mul_f32 v14, v11, v6 // (op 3735) 
v_mul_f32 v15, v11, v7 // (op 3735) 
v_mul_f32 v16, v11, v8 // (op 3735) 
v_mul_f32 v17, v11, v9 // (op 3735) 



s_lshl_b32 s0, s4, 9 // (op 3710) 
s_add_u32 s0, s0, 128 // (op 3710) 
v_add_u32 v0, v13, s0 // (op 3710) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3710) Store value


v_accvgpr_read v6, a52 // (op 3740) 
v_accvgpr_read v7, a53 // (op 3740) 
v_accvgpr_read v8, a54 // (op 3740) 
v_accvgpr_read v9, a55 // (op 3740) 
v_mul_f32 v14, v11, v6 // (op 3740) 
v_mul_f32 v15, v11, v7 // (op 3740) 
v_mul_f32 v16, v11, v8 // (op 3740) 
v_mul_f32 v17, v11, v9 // (op 3740) 



s_lshl_b32 s0, s4, 7 // (op 3712) 
s_mul_i32 s1, 5, s0 // (op 3712) 
s_add_u32 s1, s1, 128 // (op 3712) 

v_add_u32 v0, v13, s1 // (op 3712) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3712) Store value


v_accvgpr_read v6, a56 // (op 3745) 
v_accvgpr_read v7, a57 // (op 3745) 
v_accvgpr_read v8, a58 // (op 3745) 
v_accvgpr_read v9, a59 // (op 3745) 
v_mul_f32 v14, v11, v6 // (op 3745) 
v_mul_f32 v15, v11, v7 // (op 3745) 
v_mul_f32 v16, v11, v8 // (op 3745) 
v_mul_f32 v17, v11, v9 // (op 3745) 



s_lshl_b32 s0, s4, 7 // (op 3714) 
s_mul_i32 s1, 6, s0 // (op 3714) 
s_add_u32 s1, s1, 128 // (op 3714) 

v_add_u32 v0, v13, s1 // (op 3714) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3714) Store value


v_accvgpr_read v6, a60 // (op 3750) 
v_accvgpr_read v7, a61 // (op 3750) 
v_accvgpr_read v8, a62 // (op 3750) 
v_accvgpr_read v9, a63 // (op 3750) 
v_mul_f32 v14, v11, v6 // (op 3750) 
v_mul_f32 v15, v11, v7 // (op 3750) 
v_mul_f32 v16, v11, v8 // (op 3750) 
v_mul_f32 v17, v11, v9 // (op 3750) 



s_lshl_b32 s0, s4, 7 // (op 3716) 
s_mul_i32 s1, 7, s0 // (op 3716) 
s_add_u32 s1, s1, 128 // (op 3716) 

v_add_u32 v0, v13, s1 // (op 3716) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3716) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_133
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_133:


v_add_u32 v0, 128, v12 // (op 4064) 

buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 4064) Load value

v_accvgpr_read v14, a32 // (op 4044) 
v_accvgpr_read v15, a33 // (op 4044) 
v_accvgpr_read v16, a34 // (op 4044) 
v_accvgpr_read v17, a35 // (op 4044) 
v_mul_f32 v18, v11, v14 // (op 4044) 
v_mul_f32 v19, v11, v15 // (op 4044) 
v_mul_f32 v20, v11, v16 // (op 4044) 
v_mul_f32 v21, v11, v17 // (op 4044) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4063) 
v_mul_f32 v15, v10, v7 // (op 4063) 
v_mul_f32 v16, v10, v8 // (op 4063) 
v_mul_f32 v17, v10, v9 // (op 4063) 

v_add_f32 v6, v14, v18 // (op 4045) 
v_add_f32 v7, v15, v19 // (op 4045) 
v_add_f32 v8, v16, v20 // (op 4045) 
v_add_f32 v9, v17, v21 // (op 4045) 




v_add_u32 v0, 128, v13 // (op 4047) 

buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4047) Store value




s_lshl_b32 s0, s2, 7 // (op 4068) 
s_add_u32 s0, s0, 128 // (op 4068) 
v_add_u32 v0, v12, s0 // (op 4068) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 4068) Load value

v_accvgpr_read v14, a36 // (op 4094) 
v_accvgpr_read v15, a37 // (op 4094) 
v_accvgpr_read v16, a38 // (op 4094) 
v_accvgpr_read v17, a39 // (op 4094) 
v_mul_f32 v18, v11, v14 // (op 4094) 
v_mul_f32 v19, v11, v15 // (op 4094) 
v_mul_f32 v20, v11, v16 // (op 4094) 
v_mul_f32 v21, v11, v17 // (op 4094) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4066) 
v_mul_f32 v15, v10, v7 // (op 4066) 
v_mul_f32 v16, v10, v8 // (op 4066) 
v_mul_f32 v17, v10, v9 // (op 4066) 

v_add_f32 v6, v14, v18 // (op 4067) 
v_add_f32 v7, v15, v19 // (op 4067) 
v_add_f32 v8, v16, v20 // (op 4067) 
v_add_f32 v9, v17, v21 // (op 4067) 




s_lshl_b32 s0, s4, 7 // (op 4049) 
s_add_u32 s0, s0, 128 // (op 4049) 
v_add_u32 v0, v13, s0 // (op 4049) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4049) Store value




s_lshl_b32 s0, s2, 8 // (op 4072) 
s_add_u32 s0, s0, 128 // (op 4072) 
v_add_u32 v0, v12, s0 // (op 4072) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 4072) Load value

v_accvgpr_read v14, a40 // (op 4096) 
v_accvgpr_read v15, a41 // (op 4096) 
v_accvgpr_read v16, a42 // (op 4096) 
v_accvgpr_read v17, a43 // (op 4096) 
v_mul_f32 v18, v11, v14 // (op 4096) 
v_mul_f32 v19, v11, v15 // (op 4096) 
v_mul_f32 v20, v11, v16 // (op 4096) 
v_mul_f32 v21, v11, v17 // (op 4096) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4070) 
v_mul_f32 v15, v10, v7 // (op 4070) 
v_mul_f32 v16, v10, v8 // (op 4070) 
v_mul_f32 v17, v10, v9 // (op 4070) 

v_add_f32 v6, v14, v18 // (op 4071) 
v_add_f32 v7, v15, v19 // (op 4071) 
v_add_f32 v8, v16, v20 // (op 4071) 
v_add_f32 v9, v17, v21 // (op 4071) 




s_lshl_b32 s0, s4, 8 // (op 4051) 
s_add_u32 s0, s0, 128 // (op 4051) 
v_add_u32 v0, v13, s0 // (op 4051) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4051) Store value




s_lshl_b32 s0, s2, 7 // (op 4076) 
s_mul_i32 s1, 3, s0 // (op 4076) 
s_add_u32 s1, s1, 128 // (op 4076) 

v_add_u32 v0, v12, s1 // (op 4076) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 4076) Load value

v_accvgpr_read v14, a44 // (op 4098) 
v_accvgpr_read v15, a45 // (op 4098) 
v_accvgpr_read v16, a46 // (op 4098) 
v_accvgpr_read v17, a47 // (op 4098) 
v_mul_f32 v18, v11, v14 // (op 4098) 
v_mul_f32 v19, v11, v15 // (op 4098) 
v_mul_f32 v20, v11, v16 // (op 4098) 
v_mul_f32 v21, v11, v17 // (op 4098) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4074) 
v_mul_f32 v15, v10, v7 // (op 4074) 
v_mul_f32 v16, v10, v8 // (op 4074) 
v_mul_f32 v17, v10, v9 // (op 4074) 

v_add_f32 v6, v14, v18 // (op 4075) 
v_add_f32 v7, v15, v19 // (op 4075) 
v_add_f32 v8, v16, v20 // (op 4075) 
v_add_f32 v9, v17, v21 // (op 4075) 




s_lshl_b32 s0, s4, 7 // (op 4053) 
s_mul_i32 s1, 3, s0 // (op 4053) 
s_add_u32 s1, s1, 128 // (op 4053) 

v_add_u32 v0, v13, s1 // (op 4053) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4053) Store value




s_lshl_b32 s0, s2, 9 // (op 4080) 
s_add_u32 s0, s0, 128 // (op 4080) 
v_add_u32 v0, v12, s0 // (op 4080) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 4080) Load value

v_accvgpr_read v14, a48 // (op 4100) 
v_accvgpr_read v15, a49 // (op 4100) 
v_accvgpr_read v16, a50 // (op 4100) 
v_accvgpr_read v17, a51 // (op 4100) 
v_mul_f32 v18, v11, v14 // (op 4100) 
v_mul_f32 v19, v11, v15 // (op 4100) 
v_mul_f32 v20, v11, v16 // (op 4100) 
v_mul_f32 v21, v11, v17 // (op 4100) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4078) 
v_mul_f32 v15, v10, v7 // (op 4078) 
v_mul_f32 v16, v10, v8 // (op 4078) 
v_mul_f32 v17, v10, v9 // (op 4078) 

v_add_f32 v6, v14, v18 // (op 4079) 
v_add_f32 v7, v15, v19 // (op 4079) 
v_add_f32 v8, v16, v20 // (op 4079) 
v_add_f32 v9, v17, v21 // (op 4079) 




s_lshl_b32 s0, s4, 9 // (op 4055) 
s_add_u32 s0, s0, 128 // (op 4055) 
v_add_u32 v0, v13, s0 // (op 4055) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4055) Store value




s_lshl_b32 s0, s2, 7 // (op 4084) 
s_mul_i32 s1, 5, s0 // (op 4084) 
s_add_u32 s1, s1, 128 // (op 4084) 

v_add_u32 v0, v12, s1 // (op 4084) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 4084) Load value

v_accvgpr_read v14, a52 // (op 4102) 
v_accvgpr_read v15, a53 // (op 4102) 
v_accvgpr_read v16, a54 // (op 4102) 
v_accvgpr_read v17, a55 // (op 4102) 
v_mul_f32 v18, v11, v14 // (op 4102) 
v_mul_f32 v19, v11, v15 // (op 4102) 
v_mul_f32 v20, v11, v16 // (op 4102) 
v_mul_f32 v21, v11, v17 // (op 4102) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4082) 
v_mul_f32 v15, v10, v7 // (op 4082) 
v_mul_f32 v16, v10, v8 // (op 4082) 
v_mul_f32 v17, v10, v9 // (op 4082) 

v_add_f32 v6, v14, v18 // (op 4083) 
v_add_f32 v7, v15, v19 // (op 4083) 
v_add_f32 v8, v16, v20 // (op 4083) 
v_add_f32 v9, v17, v21 // (op 4083) 




s_lshl_b32 s0, s4, 7 // (op 4057) 
s_mul_i32 s1, 5, s0 // (op 4057) 
s_add_u32 s1, s1, 128 // (op 4057) 

v_add_u32 v0, v13, s1 // (op 4057) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4057) Store value




s_lshl_b32 s0, s2, 7 // (op 4088) 
s_mul_i32 s1, 6, s0 // (op 4088) 
s_add_u32 s1, s1, 128 // (op 4088) 

v_add_u32 v0, v12, s1 // (op 4088) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 4088) Load value

v_accvgpr_read v14, a56 // (op 4104) 
v_accvgpr_read v15, a57 // (op 4104) 
v_accvgpr_read v16, a58 // (op 4104) 
v_accvgpr_read v17, a59 // (op 4104) 
v_mul_f32 v18, v11, v14 // (op 4104) 
v_mul_f32 v19, v11, v15 // (op 4104) 
v_mul_f32 v20, v11, v16 // (op 4104) 
v_mul_f32 v21, v11, v17 // (op 4104) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4086) 
v_mul_f32 v15, v10, v7 // (op 4086) 
v_mul_f32 v16, v10, v8 // (op 4086) 
v_mul_f32 v17, v10, v9 // (op 4086) 

v_add_f32 v6, v14, v18 // (op 4087) 
v_add_f32 v7, v15, v19 // (op 4087) 
v_add_f32 v8, v16, v20 // (op 4087) 
v_add_f32 v9, v17, v21 // (op 4087) 




s_lshl_b32 s0, s4, 7 // (op 4059) 
s_mul_i32 s1, 6, s0 // (op 4059) 
s_add_u32 s1, s1, 128 // (op 4059) 

v_add_u32 v0, v13, s1 // (op 4059) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4059) Store value




s_lshl_b32 s0, s2, 7 // (op 4092) 
s_mul_i32 s1, 7, s0 // (op 4092) 
s_add_u32 s1, s1, 128 // (op 4092) 

v_add_u32 v0, v12, s1 // (op 4092) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 4092) Load value

v_accvgpr_read v14, a60 // (op 4106) 
v_accvgpr_read v15, a61 // (op 4106) 
v_accvgpr_read v16, a62 // (op 4106) 
v_accvgpr_read v17, a63 // (op 4106) 
v_mul_f32 v18, v11, v14 // (op 4106) 
v_mul_f32 v19, v11, v15 // (op 4106) 
v_mul_f32 v20, v11, v16 // (op 4106) 
v_mul_f32 v21, v11, v17 // (op 4106) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4090) 
v_mul_f32 v15, v10, v7 // (op 4090) 
v_mul_f32 v16, v10, v8 // (op 4090) 
v_mul_f32 v17, v10, v9 // (op 4090) 

v_add_f32 v6, v14, v18 // (op 4091) 
v_add_f32 v7, v15, v19 // (op 4091) 
v_add_f32 v8, v16, v20 // (op 4091) 
v_add_f32 v9, v17, v21 // (op 4091) 




s_lshl_b32 s0, s4, 7 // (op 4061) 
s_mul_i32 s1, 7, s0 // (op 4061) 
s_add_u32 s1, s1, 128 // (op 4061) 

v_add_u32 v0, v13, s1 // (op 4061) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4061) Store value




v_mov_b32 v0, v12 // (op 145) reg expression

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 145) Load value

v_accvgpr_read v14, a0 // (op 142) 
v_accvgpr_read v15, a1 // (op 142) 
v_accvgpr_read v16, a2 // (op 142) 
v_accvgpr_read v17, a3 // (op 142) 
v_mul_f32 v18, v11, v14 // (op 142) 
v_mul_f32 v19, v11, v15 // (op 142) 
v_mul_f32 v20, v11, v16 // (op 142) 
v_mul_f32 v21, v11, v17 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 146) 
v_mul_f32 v15, v10, v7 // (op 146) 
v_mul_f32 v16, v10, v8 // (op 146) 
v_mul_f32 v17, v10, v9 // (op 146) 

v_add_f32 v6, v14, v18 // (op 143) 
v_add_f32 v7, v15, v19 // (op 143) 
v_add_f32 v8, v16, v20 // (op 143) 
v_add_f32 v9, v17, v21 // (op 143) 




v_mov_b32 v0, v13 // (op 144) reg expression

buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 144) Store value




s_lshl_b32 s0, s2, 7 // (op 3882) 
v_add_u32 v0, v12, s0 // (op 3882) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 3882) Load value

v_accvgpr_read v14, a4 // (op 3879) 
v_accvgpr_read v15, a5 // (op 3879) 
v_accvgpr_read v16, a6 // (op 3879) 
v_accvgpr_read v17, a7 // (op 3879) 
v_mul_f32 v18, v11, v14 // (op 3879) 
v_mul_f32 v19, v11, v15 // (op 3879) 
v_mul_f32 v20, v11, v16 // (op 3879) 
v_mul_f32 v21, v11, v17 // (op 3879) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3883) 
v_mul_f32 v15, v10, v7 // (op 3883) 
v_mul_f32 v16, v10, v8 // (op 3883) 
v_mul_f32 v17, v10, v9 // (op 3883) 

v_add_f32 v6, v14, v18 // (op 3880) 
v_add_f32 v7, v15, v19 // (op 3880) 
v_add_f32 v8, v16, v20 // (op 3880) 
v_add_f32 v9, v17, v21 // (op 3880) 




s_lshl_b32 s0, s4, 7 // (op 3881) 
v_add_u32 v0, v13, s0 // (op 3881) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3881) Store value




s_lshl_b32 s0, s2, 8 // (op 3894) 
v_add_u32 v0, v12, s0 // (op 3894) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 3894) Load value

v_accvgpr_read v14, a8 // (op 3891) 
v_accvgpr_read v15, a9 // (op 3891) 
v_accvgpr_read v16, a10 // (op 3891) 
v_accvgpr_read v17, a11 // (op 3891) 
v_mul_f32 v18, v11, v14 // (op 3891) 
v_mul_f32 v19, v11, v15 // (op 3891) 
v_mul_f32 v20, v11, v16 // (op 3891) 
v_mul_f32 v21, v11, v17 // (op 3891) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3895) 
v_mul_f32 v15, v10, v7 // (op 3895) 
v_mul_f32 v16, v10, v8 // (op 3895) 
v_mul_f32 v17, v10, v9 // (op 3895) 

v_add_f32 v6, v14, v18 // (op 3892) 
v_add_f32 v7, v15, v19 // (op 3892) 
v_add_f32 v8, v16, v20 // (op 3892) 
v_add_f32 v9, v17, v21 // (op 3892) 




s_lshl_b32 s0, s4, 8 // (op 3893) 
v_add_u32 v0, v13, s0 // (op 3893) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3893) Store value




s_lshl_b32 s0, s2, 7 // (op 3906) 
s_mul_i32 s1, 3, s0 // (op 3906) 

v_add_u32 v0, v12, s1 // (op 3906) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 3906) Load value

v_accvgpr_read v14, a12 // (op 3903) 
v_accvgpr_read v15, a13 // (op 3903) 
v_accvgpr_read v16, a14 // (op 3903) 
v_accvgpr_read v17, a15 // (op 3903) 
v_mul_f32 v18, v11, v14 // (op 3903) 
v_mul_f32 v19, v11, v15 // (op 3903) 
v_mul_f32 v20, v11, v16 // (op 3903) 
v_mul_f32 v21, v11, v17 // (op 3903) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3907) 
v_mul_f32 v15, v10, v7 // (op 3907) 
v_mul_f32 v16, v10, v8 // (op 3907) 
v_mul_f32 v17, v10, v9 // (op 3907) 

v_add_f32 v6, v14, v18 // (op 3904) 
v_add_f32 v7, v15, v19 // (op 3904) 
v_add_f32 v8, v16, v20 // (op 3904) 
v_add_f32 v9, v17, v21 // (op 3904) 




s_lshl_b32 s0, s4, 7 // (op 3905) 
s_mul_i32 s1, 3, s0 // (op 3905) 

v_add_u32 v0, v13, s1 // (op 3905) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3905) Store value




s_lshl_b32 s0, s2, 9 // (op 3918) 
v_add_u32 v0, v12, s0 // (op 3918) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 3918) Load value

v_accvgpr_read v14, a16 // (op 3915) 
v_accvgpr_read v15, a17 // (op 3915) 
v_accvgpr_read v16, a18 // (op 3915) 
v_accvgpr_read v17, a19 // (op 3915) 
v_mul_f32 v18, v11, v14 // (op 3915) 
v_mul_f32 v19, v11, v15 // (op 3915) 
v_mul_f32 v20, v11, v16 // (op 3915) 
v_mul_f32 v21, v11, v17 // (op 3915) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3919) 
v_mul_f32 v15, v10, v7 // (op 3919) 
v_mul_f32 v16, v10, v8 // (op 3919) 
v_mul_f32 v17, v10, v9 // (op 3919) 

v_add_f32 v6, v14, v18 // (op 3916) 
v_add_f32 v7, v15, v19 // (op 3916) 
v_add_f32 v8, v16, v20 // (op 3916) 
v_add_f32 v9, v17, v21 // (op 3916) 




s_lshl_b32 s0, s4, 9 // (op 3917) 
v_add_u32 v0, v13, s0 // (op 3917) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3917) Store value




s_lshl_b32 s0, s2, 7 // (op 3930) 
s_mul_i32 s1, 5, s0 // (op 3930) 

v_add_u32 v0, v12, s1 // (op 3930) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 3930) Load value

v_accvgpr_read v14, a20 // (op 3927) 
v_accvgpr_read v15, a21 // (op 3927) 
v_accvgpr_read v16, a22 // (op 3927) 
v_accvgpr_read v17, a23 // (op 3927) 
v_mul_f32 v18, v11, v14 // (op 3927) 
v_mul_f32 v19, v11, v15 // (op 3927) 
v_mul_f32 v20, v11, v16 // (op 3927) 
v_mul_f32 v21, v11, v17 // (op 3927) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3931) 
v_mul_f32 v15, v10, v7 // (op 3931) 
v_mul_f32 v16, v10, v8 // (op 3931) 
v_mul_f32 v17, v10, v9 // (op 3931) 

v_add_f32 v6, v14, v18 // (op 3928) 
v_add_f32 v7, v15, v19 // (op 3928) 
v_add_f32 v8, v16, v20 // (op 3928) 
v_add_f32 v9, v17, v21 // (op 3928) 




s_lshl_b32 s0, s4, 7 // (op 3929) 
s_mul_i32 s1, 5, s0 // (op 3929) 

v_add_u32 v0, v13, s1 // (op 3929) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3929) Store value




s_lshl_b32 s0, s2, 7 // (op 3942) 
s_mul_i32 s1, 6, s0 // (op 3942) 

v_add_u32 v0, v12, s1 // (op 3942) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 3942) Load value

v_accvgpr_read v14, a24 // (op 3939) 
v_accvgpr_read v15, a25 // (op 3939) 
v_accvgpr_read v16, a26 // (op 3939) 
v_accvgpr_read v17, a27 // (op 3939) 
v_mul_f32 v18, v11, v14 // (op 3939) 
v_mul_f32 v19, v11, v15 // (op 3939) 
v_mul_f32 v20, v11, v16 // (op 3939) 
v_mul_f32 v21, v11, v17 // (op 3939) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3943) 
v_mul_f32 v15, v10, v7 // (op 3943) 
v_mul_f32 v16, v10, v8 // (op 3943) 
v_mul_f32 v17, v10, v9 // (op 3943) 

v_add_f32 v6, v14, v18 // (op 3940) 
v_add_f32 v7, v15, v19 // (op 3940) 
v_add_f32 v8, v16, v20 // (op 3940) 
v_add_f32 v9, v17, v21 // (op 3940) 




s_lshl_b32 s0, s4, 7 // (op 3941) 
s_mul_i32 s1, 6, s0 // (op 3941) 

v_add_u32 v0, v13, s1 // (op 3941) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3941) Store value




s_lshl_b32 s0, s2, 7 // (op 3954) 
s_mul_i32 s1, 7, s0 // (op 3954) 

v_add_u32 v0, v12, s1 // (op 3954) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 3954) Load value

v_accvgpr_read v14, a28 // (op 3951) 
v_accvgpr_read v15, a29 // (op 3951) 
v_accvgpr_read v16, a30 // (op 3951) 
v_accvgpr_read v17, a31 // (op 3951) 
v_mul_f32 v18, v11, v14 // (op 3951) 
v_mul_f32 v19, v11, v15 // (op 3951) 
v_mul_f32 v20, v11, v16 // (op 3951) 
v_mul_f32 v21, v11, v17 // (op 3951) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3955) 
v_mul_f32 v15, v10, v7 // (op 3955) 
v_mul_f32 v16, v10, v8 // (op 3955) 
v_mul_f32 v17, v10, v9 // (op 3955) 

v_add_f32 v6, v14, v18 // (op 3952) 
v_add_f32 v7, v15, v19 // (op 3952) 
v_add_f32 v8, v16, v20 // (op 3952) 
v_add_f32 v9, v17, v21 // (op 3952) 




s_lshl_b32 s0, s4, 7 // (op 3953) 
s_mul_i32 s1, 7, s0 // (op 3953) 

v_add_u32 v0, v13, s1 // (op 3953) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3953) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_133:
























s_endpgm  // End of RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC
.LRR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_end:
.size RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC, .LRR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC_end-RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC
  .amdhsa_next_free_vgpr 204
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 51712 // lds bytes
  .amdhsa_accum_offset 140
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
    macrotile: [64, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC
    .symbol: RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__LDSSYNC.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 51712
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 140
    .agpr_count: 64
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

