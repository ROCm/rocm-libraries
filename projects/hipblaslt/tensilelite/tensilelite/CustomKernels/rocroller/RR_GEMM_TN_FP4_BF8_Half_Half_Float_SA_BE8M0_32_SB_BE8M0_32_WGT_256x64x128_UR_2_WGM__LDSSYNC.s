// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC
.p2align 8
.type RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC,@function
RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 9007) 
s_mul_hi_i32 s1, s16, s10 // (op 9264) 
v_lshrrev_b32 v0, 7, v1 // (op 9069) 
v_and_b32 v3, 15, v1 // (op 9069) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 9069) 


v_and_b32 v0, 63, v1 // (op 9069) 
v_lshrrev_b32 v3, 4, v0 // (op 9069) 

v_lshlrev_b32 v0, 5, v3 // (op 9069) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 9069) 


v_lshlrev_b32 v0, 2, v3 // (op 9069) 

v_lshrrev_b32 v3, 3, v0 // (op 9069) 

v_lshrrev_b32 v0, 6, v1 // (op 9127) 
v_and_b32 v4, 1, v0 // (op 9127) 

v_and_b32 v0, 15, v1 // (op 9127) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 9127) 


v_and_b32 v0, 48, v1 // (op 9127) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 9127) 


v_lshrrev_b32 v0, 7, v1 // (op 9185) 
v_and_b32 v5, 15, v1 // (op 9185) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 9185) 


v_and_b32 v0, 63, v1 // (op 9185) 
v_lshrrev_b32 v5, 4, v0 // (op 9185) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 9185) 


v_lshrrev_b32 v5, 6, v1 // (op 9243) 
v_and_b32 v6, 1, v5 // (op 9243) 

v_and_b32 v5, 15, v1 // (op 9243) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 9243) 


v_and_b32 v5, 63, v1 // (op 9243) 
v_lshrrev_b32 v6, 4, v5 // (op 9243) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 9243) 


v_lshrrev_b32 v6, 7, v1 // (op 9295) 
v_and_b32 v7, 15, v1 // (op 9295) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 9295) 


v_and_b32 v6, 63, v1 // (op 9295) 
v_lshrrev_b32 v7, 4, v6 // (op 9295) 

v_lshlrev_b32 v6, 5, v7 // (op 9295) 

v_lshl_add_u32 v7, v8, 7, v6 // (op 9295) 


v_lshlrev_b32 v6, 2, v7 // (op 9295) 

v_lshrrev_b32 v7, 3, v6 // (op 9295) 

v_lshrrev_b32 v6, 6, v1 // (op 9324) 
v_and_b32 v8, 1, v6 // (op 9324) 

v_and_b32 v6, 15, v1 // (op 9324) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 9324) 


v_and_b32 v6, 48, v1 // (op 9324) 
v_lshl_add_u32 v8, v9, 7, v6 // (op 9324) 


v_lshrrev_b32 v6, 7, v1 // (op 9353) 
v_and_b32 v9, 15, v1 // (op 9353) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 9353) 


v_and_b32 v6, 63, v1 // (op 9353) 
v_lshrrev_b32 v9, 4, v6 // (op 9353) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 9353) 


v_lshrrev_b32 v9, 6, v1 // (op 9382) 
v_and_b32 v10, 1, v9 // (op 9382) 

v_and_b32 v9, 15, v1 // (op 9382) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 9382) 


v_and_b32 v9, 63, v1 // (op 9382) 
v_lshrrev_b32 v10, 4, v9 // (op 9382) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 9382) 


v_lshrrev_b32 v10, 7, v1 // (op 9007) 
s_add_i32 s18, s0, s16 // (op 9007) 

v_and_b32 v11, 63, v1 // (op 9007) 
s_ashr_i32 s0, s18, 31 // (op 9007) 
v_lshrrev_b32 v12, 4, v11 // (op 9007) 
s_cmp_eq_i32 s10, 0 // (op 9007) 
s_mov_b32 s19, scc // (op 9007) 
v_lshlrev_b32 v13, 2, v12 // (op 9007) 

s_cmp_eq_i32 s19, 0 // (op 9007) 
s_cselect_b32 s49, 0, -1 // (op 9007) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 9007) 


s_and_b32 s19, s13, 31 // (op 9007) 
v_lshrrev_b32 v10, 2, v11 // (op 9007) 

v_and_b32 v11, 3, v10 // (op 9007) 

v_lshrrev_b32 v10, 7, v1 // (op 9264) 
v_and_b32 v13, 63, v1 // (op 9264) 
s_lshl_b32 s50, 1, s19 // (op 9007) 
v_and_b32 v14, 3, v1 // (op 9007) 
s_add_i32 s51, s50, s49 // (op 9007) 


s_and_b32 s49, s0, s51 // (op 9007) 


s_add_i32 s0, s1, s16 // (op 9264) 

s_ashr_i32 s1, s0, 31 // (op 9264) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 9007) 


s_add_i32 s50, s18, s49 // (op 9007) 


v_lshrrev_b32 v11, 6, v1 // (op 9007) 
s_ashr_i32 s18, s50, s19 // (op 9007) 


v_and_b32 v14, 1, v11 // (op 9007) 

s_lshl_b32 s19, s13, 24 // (op 9007) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 9007) 


s_ashr_i32 s49, s19, 31 // (op 9007) 

s_xor_b32 s19, s18, s49 // (op 9007) 

s_sub_i32 s18, s19, s49 // (op 9007) 


s_mul_hi_i32 s19, s18, s14 // (op 9007) 
s_add_i32 s49, s19, s18 // (op 9007) 

s_ashr_i32 s19, s49, 31 // (op 9007) 
s_cmp_eq_i32 s14, 0 // (op 9007) 
s_mov_b32 s50, scc // (op 9007) 
s_cmp_eq_i32 s50, 0 // (op 9007) 
s_cselect_b32 s51, 0, -1 // (op 9007) 

s_and_b32 s50, s15, 31 // (op 9007) 
v_lshrrev_b32 v14, 4, v13 // (op 9264) 
s_cmp_eq_i32 s10, 0 // (op 9264) 
s_mov_b32 s52, scc // (op 9264) 
v_lshlrev_b32 v15, 2, v14 // (op 9264) 

s_cmp_eq_i32 s52, 0 // (op 9264) 
s_cselect_b32 s53, 0, -1 // (op 9264) 

s_lshl_b32 s52, 1, s50 // (op 9007) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 9264) 


v_lshrrev_b32 v10, 2, v13 // (op 9264) 

s_add_i32 s54, s52, s51 // (op 9007) 


s_and_b32 s51, s19, s54 // (op 9007) 


s_add_i32 s19, s49, s51 // (op 9007) 


s_ashr_i32 s49, s19, s50 // (op 9007) 


s_lshl_b32 s19, s15, 24 // (op 9007) 
s_ashr_i32 s50, s19, 31 // (op 9007) 

s_xor_b32 s19, s49, s50 // (op 9007) 

s_sub_i32 s49, s19, s50 // (op 9007) 


s_mul_i32 s19, s18, s11 // (op 9007) 
s_sub_u32 s50, s16, s19 // (op 9007) 

s_mul_i32 s19, s49, s11 // (op 9007) 
s_add_u32 s19, s19, s50 // (op 9007) 

s_mul_hi_i32 s50, s16, s9 // (op 9007) 
s_add_i32 s51, s50, s16 // (op 9007) 

s_ashr_i32 s50, s51, 31 // (op 9007) 
s_cmp_eq_i32 s9, 0 // (op 9007) 
s_mov_b32 s52, scc // (op 9007) 
s_cmp_eq_i32 s52, 0 // (op 9007) 
s_cselect_b32 s54, 0, -1 // (op 9007) 

s_and_b32 s52, s13, 31 // (op 9264) 
v_and_b32 v13, 3, v10 // (op 9264) 

s_lshl_b32 s55, 1, s52 // (op 9264) 
v_and_b32 v10, 3, v1 // (op 9264) 
s_add_i32 s56, s55, s53 // (op 9264) 


s_and_b32 s53, s1, s56 // (op 9264) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 9264) 


s_add_i32 s1, s0, s53 // (op 9264) 


v_lshrrev_b32 v10, 6, v1 // (op 9264) 
s_ashr_i32 s0, s1, s52 // (op 9264) 


v_and_b32 v13, 1, v10 // (op 9264) 

s_lshl_b32 s1, s13, 24 // (op 9264) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 9264) 


s_ashr_i32 s52, s1, 31 // (op 9264) 

s_xor_b32 s1, s0, s52 // (op 9264) 

s_sub_i32 s0, s1, s52 // (op 9264) 


s_mul_hi_i32 s1, s0, s14 // (op 9264) 
s_add_i32 s52, s1, s0 // (op 9264) 

s_ashr_i32 s1, s52, 31 // (op 9264) 
s_cmp_eq_i32 s14, 0 // (op 9264) 
s_mov_b32 s53, scc // (op 9264) 
s_cmp_eq_i32 s53, 0 // (op 9264) 
s_cselect_b32 s55, 0, -1 // (op 9264) 

s_and_b32 s53, s15, 31 // (op 9264) 
s_lshl_b32 s56, 1, s53 // (op 9264) 
s_add_i32 s57, s56, s55 // (op 9264) 


s_and_b32 s55, s1, s57 // (op 9264) 


s_add_i32 s1, s52, s55 // (op 9264) 


s_ashr_i32 s52, s1, s53 // (op 9264) 


s_lshl_b32 s1, s15, 24 // (op 9264) 
s_ashr_i32 s53, s1, 31 // (op 9264) 

s_xor_b32 s1, s52, s53 // (op 9264) 

s_sub_i32 s52, s1, s53 // (op 9264) 


s_mul_i32 s1, s0, s11 // (op 9264) 
s_sub_u32 s53, s16, s1 // (op 9264) 

s_mul_i32 s1, s52, s11 // (op 9264) 
s_add_u32 s1, s1, s53 // (op 9264) 

s_mul_hi_i32 s53, s16, s9 // (op 9264) 
s_add_i32 s55, s53, s16 // (op 9264) 

s_ashr_i32 s53, s55, 31 // (op 9264) 
s_cmp_eq_i32 s9, 0 // (op 9264) 
s_mov_b32 s56, scc // (op 9264) 
s_cmp_eq_i32 s56, 0 // (op 9264) 
s_cselect_b32 s57, 0, -1 // (op 9264) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 9007) 
s_lshl_b32 s58, 1, s56 // (op 9007) 
s_add_i32 s59, s58, s54 // (op 9007) 


s_and_b32 s54, s50, s59 // (op 9007) 


s_add_i32 s50, s51, s54 // (op 9007) 


s_ashr_i32 s51, s50, s56 // (op 9007) 


s_lshl_b32 s50, s45, 24 // (op 9007) 
s_ashr_i32 s54, s50, 31 // (op 9007) 

s_xor_b32 s50, s51, s54 // (op 9007) 

s_sub_i32 s51, s50, s54 // (op 9007) 


s_and_b32 s50, s45, 31 // (op 9264) 
s_lshl_b32 s54, 1, s50 // (op 9264) 
s_add_i32 s56, s54, s57 // (op 9264) 


s_and_b32 s54, s53, s56 // (op 9264) 


s_add_i32 s53, s55, s54 // (op 9264) 


s_ashr_i32 s54, s53, s50 // (op 9264) 


s_lshl_b32 s50, s45, 24 // (op 9264) 
s_ashr_i32 s53, s50, 31 // (op 9264) 

s_xor_b32 s50, s54, s53 // (op 9264) 

s_sub_i32 s54, s50, s53 // (op 9264) 


s_add_u32 s56, s6, 64 // (op 9007) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 9007) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 9007) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 9007) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 9007) 

s_mul_i32 s50, s11, s56 // (op 9007) 
s_mul_i32 s53, s8, s50 // (op 9007) 

s_mul_i32 s50, s51, s53 // (op 9007) 

s_sub_u32 s53, s16, s50 // (op 9007) 

s_mov_b32 s50, s53 // (op 9007) 
s_mul_hi_i32 s55, s50, s12 // (op 9007) 
s_add_i32 s58, s55, s50 // (op 9007) 


s_ashr_i32 s50, s58, 31 // (op 9007) 
s_cmp_eq_i32 s12, 0 // (op 9007) 
s_mov_b32 s55, scc // (op 9007) 
s_cmp_eq_i32 s55, 0 // (op 9007) 
s_cselect_b32 s59, 0, -1 // (op 9007) 

s_and_b32 s55, s44, 31 // (op 9007) 
s_lshl_b32 s60, 1, s55 // (op 9007) 
s_add_i32 s61, s60, s59 // (op 9007) 


s_and_b32 s59, s50, s61 // (op 9007) 


s_add_i32 s50, s58, s59 // (op 9007) 


s_ashr_i32 s58, s50, s55 // (op 9007) 


s_lshl_b32 s50, s44, 24 // (op 9007) 
s_ashr_i32 s55, s50, 31 // (op 9007) 

s_xor_b32 s50, s58, s55 // (op 9007) 

s_sub_i32 s58, s50, s55 // (op 9007) 


s_mul_i32 s50, s46, s11 // (op 9007) 
s_add_u32 s60, s20, 256 // (op 9007) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 9007) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 9007) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 9007) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 8 // (op 9007) 

s_sub_i32 s55, s60, s50 // (op 9007) 

s_mul_i32 s59, s58, s55 // (op 9007) 

s_sub_u32 s55, s53, s59 // (op 9007) 


s_add_u32 s53, s55, s50 // (op 9007) 


s_cmp_eq_i32 s51, 0 // (op 9007) 
s_mov_b32 s50, scc // (op 9007) 

s_cmp_eq_i32 s50, 0 // (op 9007) 
s_cselect_b32 s51, s53, s19 // (op 9007) 


s_mul_i32 s19, s49, s56 // (op 9007) 


v_lshl_add_u32 v13, s51, 8, v12 // (op 9007) 


s_sub_i32 s49, s18, s19 // (op 9007) 


s_cmp_eq_i32 s50, 0 // (op 9007) 
s_cselect_b32 s51, s58, s49 // (op 9007) 



v_lshl_add_u32 v12, s51, 6, v11 // (op 9007) 


v_mul_lo_u32 v11, v12, s2 // (op 9007) 

v_add_lshl_u32 v12, v13, v11, 1 // (op 9007) 


s_add_u32 s18, s6, 64 // (op 9264) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 9264) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 9264) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 9264) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 6 // (op 9264) 

s_mul_i32 s53, s11, s18 // (op 9264) 
s_mul_i32 s49, s8, s53 // (op 9264) 

s_mul_i32 s53, s54, s49 // (op 9264) 

s_sub_u32 s49, s16, s53 // (op 9264) 

s_mov_b32 s53, s49 // (op 9264) 
s_mul_hi_i32 s50, s53, s12 // (op 9264) 
s_add_i32 s51, s50, s53 // (op 9264) 


s_ashr_i32 s50, s51, 31 // (op 9264) 
s_cmp_eq_i32 s12, 0 // (op 9264) 
s_mov_b32 s53, scc // (op 9264) 
s_cmp_eq_i32 s53, 0 // (op 9264) 
s_cselect_b32 s55, 0, -1 // (op 9264) 

s_and_b32 s53, s44, 31 // (op 9264) 
s_lshl_b32 s56, 1, s53 // (op 9264) 
s_add_i32 s57, s56, s55 // (op 9264) 


s_and_b32 s55, s50, s57 // (op 9264) 


s_add_i32 s50, s51, s55 // (op 9264) 


s_ashr_i32 s51, s50, s53 // (op 9264) 


s_lshl_b32 s50, s44, 24 // (op 9264) 
s_ashr_i32 s53, s50, 31 // (op 9264) 

s_xor_b32 s50, s51, s53 // (op 9264) 

s_sub_i32 s51, s50, s53 // (op 9264) 


s_mul_i32 s50, s46, s11 // (op 9264) 
s_add_u32 s56, s20, 256 // (op 9264) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 9264) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 9264) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 9264) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 9264) 

s_sub_i32 s53, s56, s50 // (op 9264) 

s_mul_i32 s55, s51, s53 // (op 9264) 

s_sub_u32 s53, s49, s55 // (op 9264) 


s_add_u32 s49, s53, s50 // (op 9264) 


s_cmp_eq_i32 s54, 0 // (op 9264) 
s_mov_b32 s50, scc // (op 9264) 

s_cmp_eq_i32 s50, 0 // (op 9264) 
s_cselect_b32 s53, s49, s1 // (op 9264) 


s_mul_i32 s1, s52, s18 // (op 9264) 


v_lshl_add_u32 v11, s53, 8, v14 // (op 9264) 


s_sub_i32 s49, s0, s1 // (op 9264) 


s_cmp_eq_i32 s50, 0 // (op 9264) 
s_cselect_b32 s0, s51, s49 // (op 9264) 



v_lshl_add_u32 v13, s0, 6, v10 // (op 9264) 


v_mul_lo_u32 v10, v13, s4 // (op 9264) 

v_add_lshl_u32 v13, v11, v10, 1 // (op 9264) 


s_mul_i32 s1, s2, s7 // (op 9009) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 9009) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 9009) least significant
s_mul_hi_u32 s19, s2, s6 // (op 9009) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 9009) most significant: sum
s_add_u32 s1, s1, s19 // (op 9009) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 9009) 

s_mul_i32 s1, s4, s7 // (op 9266) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 9266) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 9266) least significant
s_mul_hi_u32 s50, s4, s6 // (op 9266) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 9266) most significant: sum
s_add_u32 s1, s1, s50 // (op 9266) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 9266) 

s_mov_b32 s52, s22 // (op 9009) reg expression
s_mov_b32 s53, s23 // (op 9009) reg expression
s_mov_b32 s54, s18 // (op 9009) reg expression
s_mov_b32 s55, 131072 // (op 9009) call()

s_mov_b32 s56, s24 // (op 9266) reg expression

s_mov_b32 s57, s25 // (op 9266) reg expression
s_mov_b32 s58, s50 // (op 9266) reg expression
s_mov_b32 s59, 131072 // (op 9266) call()


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
v_accvgpr_write a32, 0.00000 // (op 1718) call()
v_accvgpr_write a33, 0.00000 // (op 1718) call()
v_accvgpr_write a34, 0.00000 // (op 1718) call()
v_accvgpr_write a35, 0.00000 // (op 1718) call()
v_accvgpr_write a36, 0.00000 // (op 1873) call()
v_accvgpr_write a37, 0.00000 // (op 1873) call()
v_accvgpr_write a38, 0.00000 // (op 1873) call()
v_accvgpr_write a39, 0.00000 // (op 1873) call()
v_accvgpr_write a40, 0.00000 // (op 2104) call()
v_accvgpr_write a41, 0.00000 // (op 2104) call()
v_accvgpr_write a42, 0.00000 // (op 2104) call()
v_accvgpr_write a43, 0.00000 // (op 2104) call()
v_accvgpr_write a44, 0.00000 // (op 2259) call()
v_accvgpr_write a45, 0.00000 // (op 2259) call()
v_accvgpr_write a46, 0.00000 // (op 2259) call()
v_accvgpr_write a47, 0.00000 // (op 2259) call()
v_accvgpr_write a48, 0.00000 // (op 2490) call()
v_accvgpr_write a49, 0.00000 // (op 2490) call()
v_accvgpr_write a50, 0.00000 // (op 2490) call()
v_accvgpr_write a51, 0.00000 // (op 2490) call()
v_accvgpr_write a52, 0.00000 // (op 2645) call()
v_accvgpr_write a53, 0.00000 // (op 2645) call()
v_accvgpr_write a54, 0.00000 // (op 2645) call()
v_accvgpr_write a55, 0.00000 // (op 2645) call()
v_accvgpr_write a56, 0.00000 // (op 2876) call()
v_accvgpr_write a57, 0.00000 // (op 2876) call()
v_accvgpr_write a58, 0.00000 // (op 2876) call()
v_accvgpr_write a59, 0.00000 // (op 2876) call()
v_accvgpr_write a60, 0.00000 // (op 3031) call()
v_accvgpr_write a61, 0.00000 // (op 3031) call()
v_accvgpr_write a62, 0.00000 // (op 3031) call()
v_accvgpr_write a63, 0.00000 // (op 3031) call()
s_mul_hi_i32 s0, s16, s10 // (op 8769) 
v_and_b32 v14, 4294967292, v1 // (op 8769) 
s_add_i32 s1, s0, s16 // (op 8769) 

v_and_b32 v15, -4, v1 // (op 8769) 
s_ashr_i32 s0, s1, 31 // (op 8769) 
v_sub_u32 v16, v1, v15 // (op 8769) 

s_cmp_eq_i32 s10, 0 // (op 8769) 
s_mov_b32 s18, scc // (op 8769) 
v_lshlrev_b32 v15, 5, v16 // (op 8769) 

s_cmp_eq_i32 s18, 0 // (op 8769) 
s_cselect_b32 s19, 0, -1 // (op 8769) 

v_mov_b32 v17, 0 // (op 8769) convert
v_mov_b32 v16, v15 // (op 8769) convert

s_and_b32 s18, s13, 31 // (op 8769) 
s_lshl_b32 s22, 1, s18 // (op 8769) 
s_add_i32 s23, s22, s19 // (op 8769) 


s_and_b32 s19, s0, s23 // (op 8769) 


s_add_i32 s0, s1, s19 // (op 8769) 


s_ashr_i32 s1, s0, s18 // (op 8769) 


s_lshl_b32 s0, s13, 24 // (op 8769) 
s_ashr_i32 s18, s0, 31 // (op 8769) 

s_xor_b32 s0, s1, s18 // (op 8769) 

s_sub_i32 s1, s0, s18 // (op 8769) 


s_mul_hi_i32 s0, s1, s14 // (op 8769) 
s_add_i32 s18, s0, s1 // (op 8769) 

s_ashr_i32 s0, s18, 31 // (op 8769) 
s_cmp_eq_i32 s14, 0 // (op 8769) 
s_mov_b32 s19, scc // (op 8769) 
s_cmp_eq_i32 s19, 0 // (op 8769) 
s_cselect_b32 s22, 0, -1 // (op 8769) 

s_and_b32 s19, s15, 31 // (op 8769) 
s_lshl_b32 s23, 1, s19 // (op 8769) 
s_add_i32 s24, s23, s22 // (op 8769) 


s_and_b32 s25, s0, s24 // (op 8769) 


s_add_i32 s0, s18, s25 // (op 8769) 


s_ashr_i32 s18, s0, s19 // (op 8769) 


s_lshl_b32 s0, s15, 24 // (op 8769) 
s_ashr_i32 s19, s0, 31 // (op 8769) 

s_xor_b32 s0, s18, s19 // (op 8769) 

s_sub_i32 s18, s0, s19 // (op 8769) 


s_mul_i32 s0, s1, s11 // (op 8769) 

s_sub_u32 s1, s16, s0 // (op 8769) 

s_mul_i32 s0, s18, s11 // (op 8769) 
s_add_u32 s0, s0, s1 // (op 8769) 


s_mul_hi_i32 s1, s16, s9 // (op 8769) 
s_add_i32 s18, s1, s16 // (op 8769) 

s_ashr_i32 s1, s18, 31 // (op 8769) 
s_cmp_eq_i32 s9, 0 // (op 8769) 
s_mov_b32 s19, scc // (op 8769) 
s_cmp_eq_i32 s19, 0 // (op 8769) 
s_cselect_b32 s22, 0, -1 // (op 8769) 

s_and_b32 s19, s45, 31 // (op 8769) 
s_lshl_b32 s23, 1, s19 // (op 8769) 
s_add_i32 s24, s23, s22 // (op 8769) 


s_and_b32 s25, s1, s24 // (op 8769) 


s_add_i32 s1, s18, s25 // (op 8769) 


s_ashr_i32 s18, s1, s19 // (op 8769) 


s_lshl_b32 s1, s45, 24 // (op 8769) 
s_ashr_i32 s19, s1, 31 // (op 8769) 

s_xor_b32 s1, s18, s19 // (op 8769) 

s_sub_i32 s18, s1, s19 // (op 8769) 


s_add_u32 s22, s6, 64 // (op 8769) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8769) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8769) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8769) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 8769) 

s_mul_i32 s1, s11, s22 // (op 8769) 

s_mul_i32 s19, s8, s1 // (op 8769) 

s_mul_i32 s1, s18, s19 // (op 8769) 

s_sub_u32 s19, s16, s1 // (op 8769) 

s_mov_b32 s1, s19 // (op 8769) 
s_mul_hi_i32 s22, s1, s12 // (op 8769) 
s_add_i32 s23, s22, s1 // (op 8769) 


s_ashr_i32 s1, s23, 31 // (op 8769) 
s_cmp_eq_i32 s12, 0 // (op 8769) 
s_mov_b32 s22, scc // (op 8769) 
s_cmp_eq_i32 s22, 0 // (op 8769) 
s_cselect_b32 s24, 0, -1 // (op 8769) 

s_and_b32 s22, s44, 31 // (op 8769) 
s_lshl_b32 s25, 1, s22 // (op 8769) 
s_add_i32 s47, s25, s24 // (op 8769) 


s_and_b32 s24, s1, s47 // (op 8769) 


s_add_i32 s1, s23, s24 // (op 8769) 


s_ashr_i32 s23, s1, s22 // (op 8769) 


s_lshl_b32 s1, s44, 24 // (op 8769) 
s_ashr_i32 s22, s1, 31 // (op 8769) 

s_xor_b32 s1, s23, s22 // (op 8769) 

s_sub_i32 s23, s1, s22 // (op 8769) 


s_mul_i32 s1, s46, s11 // (op 8769) 
s_add_u32 s24, s20, 256 // (op 8769) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8769) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 8769) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 8769) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 8769) 

s_sub_i32 s22, s24, s1 // (op 8769) 

s_mul_i32 s24, s23, s22 // (op 8769) 


s_sub_u32 s25, s19, s24 // (op 8769) 


s_add_u32 s19, s25, s1 // (op 8769) 


s_cmp_eq_i32 s18, 0 // (op 8769) 
s_mov_b32 s1, scc // (op 8769) 

s_cmp_eq_i32 s1, 0 // (op 8769) 
s_cselect_b32 s18, s19, s0 // (op 8769) 



v_lshl_add_u32 v15, s18, 8, v14 // (op 8769) 


v_mul_lo_u32 v19, v15, s27 // (op 8769) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 8769) most significant: high of low * low
v_mul_lo_u32 v18, v15, s26 // (op 8769) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 8769) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 8769) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 8769) most significant half



v_lshlrev_b64 v[14:15], 2, v[18:19] // (op 8769) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 8769) 

v_mov_b32 v14, v16 // (op 8769) convert

s_mul_i32 s1, s26, s21 // (op 8771) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 8771) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 8771) least significant
s_mul_hi_u32 s19, s26, s20 // (op 8771) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8771) most significant: sum
s_add_u32 s1, s1, s19 // (op 8771) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 8771) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 8771) 

s_mov_b32 s48, s28 // (op 8771) reg expression
s_mov_b32 s49, s29 // (op 8771) reg expression
s_mov_b32 s50, s0 // (op 8771) reg expression
s_mov_b32 s51, 131072 // (op 8771) call()

s_mul_hi_i32 s47, s16, s10 // (op 8788) 
v_and_b32 v15, 4294967292, v1 // (op 8788) 
s_add_i32 s0, s47, s16 // (op 8788) 

v_and_b32 v16, -4, v1 // (op 8788) 
s_ashr_i32 s1, s0, 31 // (op 8788) 
v_sub_u32 v17, v1, v16 // (op 8788) 

s_cmp_eq_i32 s10, 0 // (op 8788) 
s_mov_b32 s47, scc // (op 8788) 
v_lshlrev_b32 v16, 5, v17 // (op 8788) 

s_cmp_eq_i32 s47, 0 // (op 8788) 
s_cselect_b32 s18, 0, -1 // (op 8788) 

v_mov_b32 v19, 0 // (op 8788) convert
v_mov_b32 v18, v16 // (op 8788) convert

s_and_b32 s19, s13, 31 // (op 8788) 
s_lshl_b32 s47, 1, s19 // (op 8788) 
s_add_i32 s22, s47, s18 // (op 8788) 


s_and_b32 s18, s1, s22 // (op 8788) 


s_add_i32 s1, s0, s18 // (op 8788) 


s_ashr_i32 s0, s1, s19 // (op 8788) 


s_lshl_b32 s1, s13, 24 // (op 8788) 
s_ashr_i32 s47, s1, 31 // (op 8788) 

s_xor_b32 s1, s0, s47 // (op 8788) 

s_sub_i32 s0, s1, s47 // (op 8788) 


s_mul_hi_i32 s1, s0, s14 // (op 8788) 
s_add_i32 s47, s1, s0 // (op 8788) 

s_ashr_i32 s1, s47, 31 // (op 8788) 
s_cmp_eq_i32 s14, 0 // (op 8788) 
s_mov_b32 s18, scc // (op 8788) 
s_cmp_eq_i32 s18, 0 // (op 8788) 
s_cselect_b32 s19, 0, -1 // (op 8788) 

s_and_b32 s18, s15, 31 // (op 8788) 
s_lshl_b32 s22, 1, s18 // (op 8788) 
s_add_i32 s23, s22, s19 // (op 8788) 


s_and_b32 s19, s1, s23 // (op 8788) 


s_add_i32 s1, s47, s19 // (op 8788) 


s_ashr_i32 s19, s1, s18 // (op 8788) 


s_lshl_b32 s1, s15, 24 // (op 8788) 
s_ashr_i32 s18, s1, 31 // (op 8788) 

s_xor_b32 s1, s19, s18 // (op 8788) 

s_sub_i32 s19, s1, s18 // (op 8788) 


s_mul_i32 s1, s0, s11 // (op 8788) 

s_sub_u32 s0, s16, s1 // (op 8788) 

s_mul_i32 s1, s19, s11 // (op 8788) 
s_add_u32 s1, s1, s0 // (op 8788) 


s_mul_hi_i32 s0, s16, s9 // (op 8788) 
s_add_i32 s47, s0, s16 // (op 8788) 

s_ashr_i32 s0, s47, 31 // (op 8788) 
s_cmp_eq_i32 s9, 0 // (op 8788) 
s_mov_b32 s18, scc // (op 8788) 
s_cmp_eq_i32 s18, 0 // (op 8788) 
s_cselect_b32 s19, 0, -1 // (op 8788) 

s_and_b32 s18, s45, 31 // (op 8788) 
s_lshl_b32 s22, 1, s18 // (op 8788) 
s_add_i32 s23, s22, s19 // (op 8788) 


s_and_b32 s19, s0, s23 // (op 8788) 


s_add_i32 s0, s47, s19 // (op 8788) 


s_ashr_i32 s19, s0, s18 // (op 8788) 


s_lshl_b32 s0, s45, 24 // (op 8788) 
s_ashr_i32 s18, s0, 31 // (op 8788) 

s_xor_b32 s0, s19, s18 // (op 8788) 

s_sub_i32 s19, s0, s18 // (op 8788) 


s_add_u32 s22, s6, 64 // (op 8788) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8788) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8788) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8788) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 8788) 

s_mul_i32 s0, s11, s22 // (op 8788) 

s_mul_i32 s18, s8, s0 // (op 8788) 

s_mul_i32 s0, s19, s18 // (op 8788) 

s_sub_u32 s18, s16, s0 // (op 8788) 

s_mov_b32 s0, s18 // (op 8788) 
s_mul_hi_i32 s47, s0, s12 // (op 8788) 
s_add_i32 s22, s47, s0 // (op 8788) 


s_ashr_i32 s0, s22, 31 // (op 8788) 
s_cmp_eq_i32 s12, 0 // (op 8788) 
s_mov_b32 s47, scc // (op 8788) 
s_cmp_eq_i32 s47, 0 // (op 8788) 
s_cselect_b32 s23, 0, -1 // (op 8788) 

s_and_b32 s47, s44, 31 // (op 8788) 
s_lshl_b32 s24, 1, s47 // (op 8788) 
s_add_i32 s25, s24, s23 // (op 8788) 


s_and_b32 s23, s0, s25 // (op 8788) 


s_add_i32 s0, s22, s23 // (op 8788) 


s_ashr_i32 s22, s0, s47 // (op 8788) 


s_lshl_b32 s0, s44, 24 // (op 8788) 
s_ashr_i32 s47, s0, 31 // (op 8788) 

s_xor_b32 s0, s22, s47 // (op 8788) 

s_sub_i32 s22, s0, s47 // (op 8788) 


s_mul_i32 s0, s46, s11 // (op 8788) 
s_add_u32 s24, s20, 256 // (op 8788) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8788) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 8788) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 8788) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 8 // (op 8788) 

s_sub_i32 s23, s24, s0 // (op 8788) 

s_mul_i32 s47, s22, s23 // (op 8788) 


s_sub_u32 s22, s18, s47 // (op 8788) 


s_add_u32 s18, s22, s0 // (op 8788) 


s_cmp_eq_i32 s19, 0 // (op 8788) 
s_mov_b32 s0, scc // (op 8788) 

s_cmp_eq_i32 s0, 0 // (op 8788) 
s_cselect_b32 s19, s18, s1 // (op 8788) 



v_lshl_add_u32 v16, s19, 8, v15 // (op 8788) 


v_mul_lo_u32 v21, v16, s27 // (op 8788) most significant: low of low * high
v_mul_hi_u32 v15, v16, s26 // (op 8788) most significant: high of low * low
v_mul_lo_u32 v20, v16, s26 // (op 8788) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 8788) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 8788) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 8788) most significant half



v_lshlrev_b64 v[18:19], 2, v[20:21] // (op 8788) 

v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 8788) 

v_mov_b32 v15, v16 // (op 8788) convert

s_mul_i32 s1, s26, s21 // (op 8790) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 8790) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 8790) least significant
s_mul_hi_u32 s18, s26, s20 // (op 8790) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8790) most significant: sum
s_add_u32 s1, s1, s18 // (op 8790) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 8790) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 8790) 

s_mov_b32 s48, s28 // (op 8790) reg expression
s_mov_b32 s49, s29 // (op 8790) reg expression
s_mov_b32 s50, s0 // (op 8790) reg expression
s_mov_b32 s51, 131072 // (op 8790) call()

s_mul_hi_i32 s47, s16, s10 // (op 8833) 
v_lshrrev_b32 v16, 3, v1 // (op 8833) 
s_add_i32 s0, s47, s16 // (op 8833) 

v_lshlrev_b32 v17, 1, v16 // (op 8833) 

s_ashr_i32 s1, s0, 31 // (op 8833) 
v_and_b32 v16, -8, v1 // (op 8833) 
s_cmp_eq_i32 s10, 0 // (op 8833) 
s_mov_b32 s47, scc // (op 8833) 
v_sub_u32 v18, v1, v16 // (op 8833) 

s_cmp_eq_i32 s47, 0 // (op 8833) 
s_cselect_b32 s18, 0, -1 // (op 8833) 

s_and_b32 s19, s13, 31 // (op 8833) 
s_lshl_b32 s47, 1, s19 // (op 8833) 
s_add_i32 s22, s47, s18 // (op 8833) 


s_and_b32 s18, s1, s22 // (op 8833) 


s_add_i32 s1, s0, s18 // (op 8833) 


s_ashr_i32 s0, s1, s19 // (op 8833) 


s_lshl_b32 s1, s13, 24 // (op 8833) 
s_ashr_i32 s47, s1, 31 // (op 8833) 

s_xor_b32 s1, s0, s47 // (op 8833) 

s_sub_i32 s0, s1, s47 // (op 8833) 


s_mul_hi_i32 s1, s0, s14 // (op 8833) 
s_add_i32 s47, s1, s0 // (op 8833) 

s_ashr_i32 s1, s47, 31 // (op 8833) 
s_cmp_eq_i32 s14, 0 // (op 8833) 
s_mov_b32 s18, scc // (op 8833) 
s_cmp_eq_i32 s18, 0 // (op 8833) 
s_cselect_b32 s19, 0, -1 // (op 8833) 

s_and_b32 s18, s15, 31 // (op 8833) 
s_lshl_b32 s22, 1, s18 // (op 8833) 
s_add_i32 s23, s22, s19 // (op 8833) 


s_and_b32 s19, s1, s23 // (op 8833) 


s_add_i32 s1, s47, s19 // (op 8833) 


s_ashr_i32 s19, s1, s18 // (op 8833) 


s_lshl_b32 s1, s15, 24 // (op 8833) 
s_ashr_i32 s18, s1, 31 // (op 8833) 

s_xor_b32 s1, s19, s18 // (op 8833) 

s_sub_i32 s19, s1, s18 // (op 8833) 


s_add_u32 s22, s6, 64 // (op 8833) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8833) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8833) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8833) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 8833) 

s_mul_i32 s1, s19, s22 // (op 8833) 

s_sub_i32 s47, s0, s1 // (op 8833) 


s_mul_hi_i32 s0, s16, s9 // (op 8833) 
s_add_i32 s1, s0, s16 // (op 8833) 

s_ashr_i32 s0, s1, 31 // (op 8833) 
s_cmp_eq_i32 s9, 0 // (op 8833) 
s_mov_b32 s18, scc // (op 8833) 
s_cmp_eq_i32 s18, 0 // (op 8833) 
s_cselect_b32 s19, 0, -1 // (op 8833) 

s_and_b32 s18, s45, 31 // (op 8833) 
s_lshl_b32 s24, 1, s18 // (op 8833) 
s_add_i32 s25, s24, s19 // (op 8833) 


s_and_b32 s19, s0, s25 // (op 8833) 


s_add_i32 s0, s1, s19 // (op 8833) 


s_ashr_i32 s1, s0, s18 // (op 8833) 


s_lshl_b32 s0, s45, 24 // (op 8833) 
s_ashr_i32 s18, s0, 31 // (op 8833) 

s_xor_b32 s0, s1, s18 // (op 8833) 

s_sub_i32 s1, s0, s18 // (op 8833) 


s_mul_i32 s0, s11, s22 // (op 8833) 

s_mul_i32 s18, s8, s0 // (op 8833) 

s_mul_i32 s0, s1, s18 // (op 8833) 

s_sub_u32 s18, s16, s0 // (op 8833) 

s_mul_hi_i32 s0, s18, s12 // (op 8833) 
s_add_i32 s19, s0, s18 // (op 8833) 


s_ashr_i32 s0, s19, 31 // (op 8833) 
s_cmp_eq_i32 s12, 0 // (op 8833) 
s_mov_b32 s18, scc // (op 8833) 
s_cmp_eq_i32 s18, 0 // (op 8833) 
s_cselect_b32 s22, 0, -1 // (op 8833) 

s_and_b32 s18, s44, 31 // (op 8833) 
s_lshl_b32 s23, 1, s18 // (op 8833) 
s_add_i32 s24, s23, s22 // (op 8833) 


s_and_b32 s25, s0, s24 // (op 8833) 


s_add_i32 s0, s19, s25 // (op 8833) 


s_ashr_i32 s19, s0, s18 // (op 8833) 


s_lshl_b32 s0, s44, 24 // (op 8833) 
s_ashr_i32 s18, s0, 31 // (op 8833) 

s_xor_b32 s0, s19, s18 // (op 8833) 

s_sub_i32 s19, s0, s18 // (op 8833) 


s_cmp_eq_i32 s1, 0 // (op 8833) 
s_mov_b32 s0, scc // (op 8833) 

s_cmp_eq_i32 s0, 0 // (op 8833) 
s_cselect_b32 s1, s19, s47 // (op 8833) 



v_lshl_add_u32 v16, s1, 6, v17 // (op 8833) 


v_mul_lo_u32 v17, v16, s30 // (op 8833) 

v_lshl_add_u32 v16, v18, 4, v17 // (op 8833) 


s_mul_i32 s1, s30, s7 // (op 8835) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 8835) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 8835) least significant
s_mul_hi_u32 s18, s30, s6 // (op 8835) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8835) most significant: sum
s_add_u32 s1, s1, s18 // (op 8835) most significant: sum


s_mov_b32 s60, s32 // (op 8835) reg expression
s_mov_b32 s61, s33 // (op 8835) reg expression
s_mov_b32 s62, s0 // (op 8835) reg expression
s_mov_b32 s63, 131072 // (op 8835) call()

s_mul_hi_i32 s47, s16, s10 // (op 8852) 
v_lshrrev_b32 v17, 3, v1 // (op 8852) 
s_add_i32 s0, s47, s16 // (op 8852) 

v_lshlrev_b32 v18, 1, v17 // (op 8852) 

s_ashr_i32 s1, s0, 31 // (op 8852) 
v_and_b32 v17, -8, v1 // (op 8852) 
s_cmp_eq_i32 s10, 0 // (op 8852) 
s_mov_b32 s47, scc // (op 8852) 
v_sub_u32 v19, v1, v17 // (op 8852) 

s_cmp_eq_i32 s47, 0 // (op 8852) 
s_cselect_b32 s18, 0, -1 // (op 8852) 

s_and_b32 s19, s13, 31 // (op 8852) 
s_lshl_b32 s47, 1, s19 // (op 8852) 
s_add_i32 s22, s47, s18 // (op 8852) 


s_and_b32 s18, s1, s22 // (op 8852) 


s_add_i32 s1, s0, s18 // (op 8852) 


s_ashr_i32 s0, s1, s19 // (op 8852) 


s_lshl_b32 s1, s13, 24 // (op 8852) 
s_ashr_i32 s47, s1, 31 // (op 8852) 

s_xor_b32 s1, s0, s47 // (op 8852) 

s_sub_i32 s0, s1, s47 // (op 8852) 


s_mul_hi_i32 s1, s0, s14 // (op 8852) 
s_add_i32 s47, s1, s0 // (op 8852) 

s_ashr_i32 s1, s47, 31 // (op 8852) 
s_cmp_eq_i32 s14, 0 // (op 8852) 
s_mov_b32 s18, scc // (op 8852) 
s_cmp_eq_i32 s18, 0 // (op 8852) 
s_cselect_b32 s19, 0, -1 // (op 8852) 

s_and_b32 s18, s15, 31 // (op 8852) 
s_lshl_b32 s22, 1, s18 // (op 8852) 
s_add_i32 s23, s22, s19 // (op 8852) 


s_and_b32 s19, s1, s23 // (op 8852) 


s_add_i32 s1, s47, s19 // (op 8852) 


s_ashr_i32 s19, s1, s18 // (op 8852) 


s_lshl_b32 s1, s15, 24 // (op 8852) 
s_ashr_i32 s18, s1, 31 // (op 8852) 

s_xor_b32 s1, s19, s18 // (op 8852) 

s_sub_i32 s19, s1, s18 // (op 8852) 


s_add_u32 s22, s6, 64 // (op 8852) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8852) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8852) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8852) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 8852) 

s_mul_i32 s1, s19, s22 // (op 8852) 

s_sub_i32 s47, s0, s1 // (op 8852) 


s_mul_hi_i32 s0, s16, s9 // (op 8852) 
s_add_i32 s1, s0, s16 // (op 8852) 

s_ashr_i32 s0, s1, 31 // (op 8852) 
s_cmp_eq_i32 s9, 0 // (op 8852) 
s_mov_b32 s18, scc // (op 8852) 
s_cmp_eq_i32 s18, 0 // (op 8852) 
s_cselect_b32 s19, 0, -1 // (op 8852) 

s_and_b32 s18, s45, 31 // (op 8852) 
s_lshl_b32 s24, 1, s18 // (op 8852) 
s_add_i32 s25, s24, s19 // (op 8852) 


s_and_b32 s19, s0, s25 // (op 8852) 


s_add_i32 s0, s1, s19 // (op 8852) 


s_ashr_i32 s1, s0, s18 // (op 8852) 


s_lshl_b32 s0, s45, 24 // (op 8852) 
s_ashr_i32 s18, s0, 31 // (op 8852) 

s_xor_b32 s0, s1, s18 // (op 8852) 

s_sub_i32 s1, s0, s18 // (op 8852) 


s_mul_i32 s0, s11, s22 // (op 8852) 

s_mul_i32 s18, s8, s0 // (op 8852) 

s_mul_i32 s0, s1, s18 // (op 8852) 

s_sub_u32 s18, s16, s0 // (op 8852) 

s_mul_hi_i32 s0, s18, s12 // (op 8852) 
s_add_i32 s19, s0, s18 // (op 8852) 


s_ashr_i32 s0, s19, 31 // (op 8852) 
s_cmp_eq_i32 s12, 0 // (op 8852) 
s_mov_b32 s18, scc // (op 8852) 
s_cmp_eq_i32 s18, 0 // (op 8852) 
s_cselect_b32 s22, 0, -1 // (op 8852) 

s_and_b32 s18, s44, 31 // (op 8852) 
s_lshl_b32 s23, 1, s18 // (op 8852) 
s_add_i32 s24, s23, s22 // (op 8852) 


s_and_b32 s25, s0, s24 // (op 8852) 


s_add_i32 s0, s19, s25 // (op 8852) 


s_ashr_i32 s19, s0, s18 // (op 8852) 


s_lshl_b32 s0, s44, 24 // (op 8852) 
s_ashr_i32 s18, s0, 31 // (op 8852) 

s_xor_b32 s0, s19, s18 // (op 8852) 

s_sub_i32 s19, s0, s18 // (op 8852) 


s_cmp_eq_i32 s1, 0 // (op 8852) 
s_mov_b32 s0, scc // (op 8852) 

s_cmp_eq_i32 s0, 0 // (op 8852) 
s_cselect_b32 s1, s19, s47 // (op 8852) 



v_lshl_add_u32 v17, s1, 6, v18 // (op 8852) 


v_mul_lo_u32 v18, v17, s30 // (op 8852) 

v_lshl_add_u32 v17, v19, 4, v18 // (op 8852) 


s_mul_i32 s1, s30, s7 // (op 8854) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 8854) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 8854) least significant
s_mul_hi_u32 s18, s30, s6 // (op 8854) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8854) most significant: sum
s_add_u32 s1, s1, s18 // (op 8854) most significant: sum


s_mov_b32 s60, s32 // (op 8854) reg expression
s_mov_b32 s61, s33 // (op 8854) reg expression
s_mov_b32 s62, s0 // (op 8854) reg expression
s_mov_b32 s63, 131072 // (op 8854) call()

s_mul_hi_i32 s47, s16, s10 // (op 8897) 
s_add_i32 s0, s47, s16 // (op 8897) 

s_ashr_i32 s1, s0, 31 // (op 8897) 
s_cmp_eq_i32 s10, 0 // (op 8897) 
s_mov_b32 s47, scc // (op 8897) 
s_cmp_eq_i32 s47, 0 // (op 8897) 
s_cselect_b32 s18, 0, -1 // (op 8897) 

s_and_b32 s19, s13, 31 // (op 8897) 
s_lshl_b32 s47, 1, s19 // (op 8897) 
s_add_i32 s22, s47, s18 // (op 8897) 


s_and_b32 s18, s1, s22 // (op 8897) 


s_add_i32 s1, s0, s18 // (op 8897) 


s_ashr_i32 s0, s1, s19 // (op 8897) 


s_lshl_b32 s1, s13, 24 // (op 8897) 
s_ashr_i32 s47, s1, 31 // (op 8897) 

s_xor_b32 s1, s0, s47 // (op 8897) 

s_sub_i32 s0, s1, s47 // (op 8897) 


s_mul_hi_i32 s1, s0, s14 // (op 8897) 
s_add_i32 s47, s1, s0 // (op 8897) 

s_ashr_i32 s1, s47, 31 // (op 8897) 
s_cmp_eq_i32 s14, 0 // (op 8897) 
s_mov_b32 s18, scc // (op 8897) 
s_cmp_eq_i32 s18, 0 // (op 8897) 
s_cselect_b32 s19, 0, -1 // (op 8897) 

s_and_b32 s18, s15, 31 // (op 8897) 
s_lshl_b32 s22, 1, s18 // (op 8897) 
s_add_i32 s23, s22, s19 // (op 8897) 


s_and_b32 s19, s1, s23 // (op 8897) 


s_add_i32 s1, s47, s19 // (op 8897) 


s_ashr_i32 s19, s1, s18 // (op 8897) 


s_lshl_b32 s1, s15, 24 // (op 8897) 
s_ashr_i32 s18, s1, 31 // (op 8897) 

s_xor_b32 s1, s19, s18 // (op 8897) 

s_sub_i32 s19, s1, s18 // (op 8897) 


s_mul_i32 s1, s0, s11 // (op 8897) 

s_sub_u32 s0, s16, s1 // (op 8897) 

s_mul_i32 s1, s19, s11 // (op 8897) 
s_add_u32 s1, s1, s0 // (op 8897) 


s_mul_hi_i32 s0, s16, s9 // (op 8897) 
s_add_i32 s47, s0, s16 // (op 8897) 

s_ashr_i32 s0, s47, 31 // (op 8897) 
s_cmp_eq_i32 s9, 0 // (op 8897) 
s_mov_b32 s18, scc // (op 8897) 
s_cmp_eq_i32 s18, 0 // (op 8897) 
s_cselect_b32 s19, 0, -1 // (op 8897) 

s_and_b32 s18, s45, 31 // (op 8897) 
s_lshl_b32 s22, 1, s18 // (op 8897) 
s_add_i32 s23, s22, s19 // (op 8897) 


s_and_b32 s19, s0, s23 // (op 8897) 


s_add_i32 s0, s47, s19 // (op 8897) 


s_ashr_i32 s19, s0, s18 // (op 8897) 


s_lshl_b32 s0, s45, 24 // (op 8897) 
s_ashr_i32 s18, s0, 31 // (op 8897) 

s_xor_b32 s0, s19, s18 // (op 8897) 

s_sub_i32 s19, s0, s18 // (op 8897) 


s_add_u32 s22, s6, 64 // (op 8897) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8897) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8897) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8897) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 8897) 

s_mul_i32 s0, s11, s22 // (op 8897) 

s_mul_i32 s18, s8, s0 // (op 8897) 

s_mul_i32 s0, s19, s18 // (op 8897) 

s_sub_u32 s18, s16, s0 // (op 8897) 

s_mov_b32 s0, s18 // (op 8897) 
s_mul_hi_i32 s47, s0, s12 // (op 8897) 
s_add_i32 s22, s47, s0 // (op 8897) 


s_ashr_i32 s0, s22, 31 // (op 8897) 
s_cmp_eq_i32 s12, 0 // (op 8897) 
s_mov_b32 s47, scc // (op 8897) 
s_cmp_eq_i32 s47, 0 // (op 8897) 
s_cselect_b32 s23, 0, -1 // (op 8897) 

s_and_b32 s47, s44, 31 // (op 8897) 
s_lshl_b32 s24, 1, s47 // (op 8897) 
s_add_i32 s25, s24, s23 // (op 8897) 


s_and_b32 s23, s0, s25 // (op 8897) 


s_add_i32 s0, s22, s23 // (op 8897) 


s_ashr_i32 s22, s0, s47 // (op 8897) 


s_lshl_b32 s0, s44, 24 // (op 8897) 
s_ashr_i32 s47, s0, 31 // (op 8897) 

s_xor_b32 s0, s22, s47 // (op 8897) 

s_sub_i32 s22, s0, s47 // (op 8897) 


s_mul_i32 s0, s46, s11 // (op 8897) 
s_add_u32 s24, s20, 256 // (op 8897) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8897) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 8897) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 8897) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 8897) 

s_sub_i32 s23, s24, s0 // (op 8897) 

s_mul_i32 s47, s22, s23 // (op 8897) 


s_sub_u32 s22, s18, s47 // (op 8897) 


s_add_u32 s18, s22, s0 // (op 8897) 


s_cmp_eq_i32 s19, 0 // (op 8897) 
s_mov_b32 s0, scc // (op 8897) 

s_cmp_eq_i32 s0, 0 // (op 8897) 
s_cselect_b32 s19, s18, s1 // (op 8897) 



v_lshl_add_u32 v18, s19, 8, v1 // (op 8897) 

v_mul_lo_u32 v19, v18, s34 // (op 8897) 

s_mul_i32 s1, s34, s21 // (op 8899) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 8899) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 8899) least significant
s_mul_hi_u32 s18, s34, s20 // (op 8899) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8899) most significant: sum
s_add_u32 s1, s1, s18 // (op 8899) most significant: sum


s_mov_b32 s64, s36 // (op 8899) reg expression
s_mov_b32 s65, s37 // (op 8899) reg expression
s_mov_b32 s66, s0 // (op 8899) reg expression
s_mov_b32 s67, 131072 // (op 8899) call()

s_mul_hi_i32 s47, s16, s10 // (op 8916) 
s_add_i32 s0, s47, s16 // (op 8916) 

s_ashr_i32 s1, s0, 31 // (op 8916) 
s_cmp_eq_i32 s10, 0 // (op 8916) 
s_mov_b32 s47, scc // (op 8916) 
s_cmp_eq_i32 s47, 0 // (op 8916) 
s_cselect_b32 s18, 0, -1 // (op 8916) 

s_and_b32 s19, s13, 31 // (op 8916) 
s_lshl_b32 s47, 1, s19 // (op 8916) 
s_add_i32 s22, s47, s18 // (op 8916) 


s_and_b32 s18, s1, s22 // (op 8916) 


s_add_i32 s1, s0, s18 // (op 8916) 


s_ashr_i32 s0, s1, s19 // (op 8916) 


s_lshl_b32 s1, s13, 24 // (op 8916) 
s_ashr_i32 s47, s1, 31 // (op 8916) 

s_xor_b32 s1, s0, s47 // (op 8916) 

s_sub_i32 s0, s1, s47 // (op 8916) 


s_mul_hi_i32 s1, s0, s14 // (op 8916) 
s_add_i32 s47, s1, s0 // (op 8916) 

s_ashr_i32 s1, s47, 31 // (op 8916) 
s_cmp_eq_i32 s14, 0 // (op 8916) 
s_mov_b32 s18, scc // (op 8916) 
s_cmp_eq_i32 s18, 0 // (op 8916) 
s_cselect_b32 s19, 0, -1 // (op 8916) 

s_and_b32 s18, s15, 31 // (op 8916) 
s_lshl_b32 s22, 1, s18 // (op 8916) 
s_add_i32 s23, s22, s19 // (op 8916) 


s_and_b32 s19, s1, s23 // (op 8916) 


s_add_i32 s1, s47, s19 // (op 8916) 


s_ashr_i32 s19, s1, s18 // (op 8916) 


s_lshl_b32 s1, s15, 24 // (op 8916) 
s_ashr_i32 s18, s1, 31 // (op 8916) 

s_xor_b32 s1, s19, s18 // (op 8916) 

s_sub_i32 s19, s1, s18 // (op 8916) 


s_mul_i32 s1, s0, s11 // (op 8916) 

s_sub_u32 s0, s16, s1 // (op 8916) 

s_mul_i32 s1, s19, s11 // (op 8916) 
s_add_u32 s1, s1, s0 // (op 8916) 


s_mul_hi_i32 s0, s16, s9 // (op 8916) 
s_add_i32 s47, s0, s16 // (op 8916) 

s_ashr_i32 s0, s47, 31 // (op 8916) 
s_cmp_eq_i32 s9, 0 // (op 8916) 
s_mov_b32 s18, scc // (op 8916) 
s_cmp_eq_i32 s18, 0 // (op 8916) 
s_cselect_b32 s19, 0, -1 // (op 8916) 

s_and_b32 s18, s45, 31 // (op 8916) 
s_lshl_b32 s22, 1, s18 // (op 8916) 
s_add_i32 s23, s22, s19 // (op 8916) 


s_and_b32 s19, s0, s23 // (op 8916) 


s_add_i32 s0, s47, s19 // (op 8916) 


s_ashr_i32 s19, s0, s18 // (op 8916) 


s_lshl_b32 s0, s45, 24 // (op 8916) 
s_ashr_i32 s18, s0, 31 // (op 8916) 

s_xor_b32 s0, s19, s18 // (op 8916) 

s_sub_i32 s19, s0, s18 // (op 8916) 


s_add_u32 s22, s6, 64 // (op 8916) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8916) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8916) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8916) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 8916) 

s_mul_i32 s0, s11, s22 // (op 8916) 

s_mul_i32 s18, s8, s0 // (op 8916) 

s_mul_i32 s0, s19, s18 // (op 8916) 

s_sub_u32 s18, s16, s0 // (op 8916) 

s_mov_b32 s0, s18 // (op 8916) 
s_mul_hi_i32 s47, s0, s12 // (op 8916) 
s_add_i32 s22, s47, s0 // (op 8916) 


s_ashr_i32 s0, s22, 31 // (op 8916) 
s_cmp_eq_i32 s12, 0 // (op 8916) 
s_mov_b32 s47, scc // (op 8916) 
s_cmp_eq_i32 s47, 0 // (op 8916) 
s_cselect_b32 s23, 0, -1 // (op 8916) 

s_and_b32 s47, s44, 31 // (op 8916) 
s_lshl_b32 s24, 1, s47 // (op 8916) 
s_add_i32 s25, s24, s23 // (op 8916) 


s_and_b32 s23, s0, s25 // (op 8916) 


s_add_i32 s0, s22, s23 // (op 8916) 


s_ashr_i32 s22, s0, s47 // (op 8916) 


s_lshl_b32 s0, s44, 24 // (op 8916) 
s_ashr_i32 s47, s0, 31 // (op 8916) 

s_xor_b32 s0, s22, s47 // (op 8916) 

s_sub_i32 s22, s0, s47 // (op 8916) 


s_mul_i32 s0, s46, s11 // (op 8916) 
s_add_u32 s24, s20, 256 // (op 8916) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8916) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 8916) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 8916) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 8 // (op 8916) 

s_sub_i32 s23, s24, s0 // (op 8916) 

s_mul_i32 s47, s22, s23 // (op 8916) 


s_sub_u32 s22, s18, s47 // (op 8916) 


s_add_u32 s18, s22, s0 // (op 8916) 


s_cmp_eq_i32 s19, 0 // (op 8916) 
s_mov_b32 s0, scc // (op 8916) 

s_cmp_eq_i32 s0, 0 // (op 8916) 
s_cselect_b32 s19, s18, s1 // (op 8916) 



v_lshl_add_u32 v18, s19, 8, v1 // (op 8916) 

v_mul_lo_u32 v20, v18, s34 // (op 8916) 

s_mul_i32 s1, s34, s21 // (op 8918) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 8918) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 8918) least significant
s_mul_hi_u32 s18, s34, s20 // (op 8918) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8918) most significant: sum
s_add_u32 s1, s1, s18 // (op 8918) most significant: sum


s_mov_b32 s64, s36 // (op 8918) reg expression
s_mov_b32 s65, s37 // (op 8918) reg expression
s_mov_b32 s66, s0 // (op 8918) reg expression
s_mov_b32 s67, 131072 // (op 8918) call()

s_mul_hi_i32 s47, s16, s10 // (op 8961) 
v_lshrrev_b32 v18, 2, v1 // (op 8961) 
s_add_i32 s0, s47, s16 // (op 8961) 

v_and_b32 v21, -4, v1 // (op 8961) 
s_ashr_i32 s1, s0, 31 // (op 8961) 
v_sub_u32 v22, v1, v21 // (op 8961) 

s_cmp_eq_i32 s10, 0 // (op 8961) 
s_mov_b32 s47, scc // (op 8961) 
s_cmp_eq_i32 s47, 0 // (op 8961) 
s_cselect_b32 s18, 0, -1 // (op 8961) 

s_and_b32 s19, s13, 31 // (op 8961) 
s_lshl_b32 s47, 1, s19 // (op 8961) 
s_add_i32 s22, s47, s18 // (op 8961) 


s_and_b32 s18, s1, s22 // (op 8961) 


s_add_i32 s1, s0, s18 // (op 8961) 


s_ashr_i32 s0, s1, s19 // (op 8961) 


s_lshl_b32 s1, s13, 24 // (op 8961) 
s_ashr_i32 s47, s1, 31 // (op 8961) 

s_xor_b32 s1, s0, s47 // (op 8961) 

s_sub_i32 s0, s1, s47 // (op 8961) 


s_mul_hi_i32 s1, s0, s14 // (op 8961) 
s_add_i32 s47, s1, s0 // (op 8961) 

s_ashr_i32 s1, s47, 31 // (op 8961) 
s_cmp_eq_i32 s14, 0 // (op 8961) 
s_mov_b32 s18, scc // (op 8961) 
s_cmp_eq_i32 s18, 0 // (op 8961) 
s_cselect_b32 s19, 0, -1 // (op 8961) 

s_and_b32 s18, s15, 31 // (op 8961) 
s_lshl_b32 s22, 1, s18 // (op 8961) 
s_add_i32 s23, s22, s19 // (op 8961) 


s_and_b32 s19, s1, s23 // (op 8961) 


s_add_i32 s1, s47, s19 // (op 8961) 


s_ashr_i32 s19, s1, s18 // (op 8961) 


s_lshl_b32 s1, s15, 24 // (op 8961) 
s_ashr_i32 s18, s1, 31 // (op 8961) 

s_xor_b32 s1, s19, s18 // (op 8961) 

s_sub_i32 s19, s1, s18 // (op 8961) 


s_add_u32 s22, s6, 64 // (op 8961) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8961) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8961) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8961) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 8961) 

s_mul_i32 s1, s19, s22 // (op 8961) 

s_sub_i32 s47, s0, s1 // (op 8961) 


s_mul_hi_i32 s0, s16, s9 // (op 8961) 
s_add_i32 s1, s0, s16 // (op 8961) 

s_ashr_i32 s0, s1, 31 // (op 8961) 
s_cmp_eq_i32 s9, 0 // (op 8961) 
s_mov_b32 s18, scc // (op 8961) 
s_cmp_eq_i32 s18, 0 // (op 8961) 
s_cselect_b32 s19, 0, -1 // (op 8961) 

s_and_b32 s18, s45, 31 // (op 8961) 
s_lshl_b32 s24, 1, s18 // (op 8961) 
s_add_i32 s25, s24, s19 // (op 8961) 


s_and_b32 s19, s0, s25 // (op 8961) 


s_add_i32 s0, s1, s19 // (op 8961) 


s_ashr_i32 s1, s0, s18 // (op 8961) 


s_lshl_b32 s0, s45, 24 // (op 8961) 
s_ashr_i32 s18, s0, 31 // (op 8961) 

s_xor_b32 s0, s1, s18 // (op 8961) 

s_sub_i32 s1, s0, s18 // (op 8961) 


s_mul_i32 s0, s11, s22 // (op 8961) 

s_mul_i32 s18, s8, s0 // (op 8961) 

s_mul_i32 s0, s1, s18 // (op 8961) 

s_sub_u32 s18, s16, s0 // (op 8961) 

s_mul_hi_i32 s0, s18, s12 // (op 8961) 
s_add_i32 s19, s0, s18 // (op 8961) 


s_ashr_i32 s0, s19, 31 // (op 8961) 
s_cmp_eq_i32 s12, 0 // (op 8961) 
s_mov_b32 s18, scc // (op 8961) 
s_cmp_eq_i32 s18, 0 // (op 8961) 
s_cselect_b32 s22, 0, -1 // (op 8961) 

s_and_b32 s18, s44, 31 // (op 8961) 
s_lshl_b32 s23, 1, s18 // (op 8961) 
s_add_i32 s24, s23, s22 // (op 8961) 


s_and_b32 s25, s0, s24 // (op 8961) 


s_add_i32 s0, s19, s25 // (op 8961) 


s_ashr_i32 s19, s0, s18 // (op 8961) 


s_lshl_b32 s0, s44, 24 // (op 8961) 
s_ashr_i32 s18, s0, 31 // (op 8961) 

s_xor_b32 s0, s19, s18 // (op 8961) 

s_sub_i32 s19, s0, s18 // (op 8961) 


s_cmp_eq_i32 s1, 0 // (op 8961) 
s_mov_b32 s0, scc // (op 8961) 

s_cmp_eq_i32 s0, 0 // (op 8961) 
s_cselect_b32 s1, s19, s47 // (op 8961) 



v_lshl_add_u32 v21, s1, 6, v18 // (op 8961) 


v_mul_lo_u32 v18, v21, s38 // (op 8961) 

v_add_u32 v21, v22, v18 // (op 8961) 


s_mul_i32 s1, s38, s7 // (op 8963) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 8963) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 8963) least significant
s_mul_hi_u32 s18, s38, s6 // (op 8963) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8963) most significant: sum
s_add_u32 s1, s1, s18 // (op 8963) most significant: sum


s_mov_b32 s68, s40 // (op 8963) reg expression
s_mov_b32 s69, s41 // (op 8963) reg expression
s_mov_b32 s70, s0 // (op 8963) reg expression
s_mov_b32 s71, 131072 // (op 8963) call()

s_mul_hi_i32 s47, s16, s10 // (op 8980) 
v_lshrrev_b32 v18, 2, v1 // (op 8980) 
s_add_i32 s0, s47, s16 // (op 8980) 

v_and_b32 v22, -4, v1 // (op 8980) 
s_ashr_i32 s1, s0, 31 // (op 8980) 
v_sub_u32 v23, v1, v22 // (op 8980) 

s_cmp_eq_i32 s10, 0 // (op 8980) 
s_mov_b32 s47, scc // (op 8980) 
s_cmp_eq_i32 s47, 0 // (op 8980) 
s_cselect_b32 s18, 0, -1 // (op 8980) 

s_and_b32 s19, s13, 31 // (op 8980) 
s_lshl_b32 s47, 1, s19 // (op 8980) 
s_add_i32 s22, s47, s18 // (op 8980) 


s_and_b32 s18, s1, s22 // (op 8980) 


s_add_i32 s1, s0, s18 // (op 8980) 


s_ashr_i32 s0, s1, s19 // (op 8980) 


s_lshl_b32 s1, s13, 24 // (op 8980) 
s_ashr_i32 s47, s1, 31 // (op 8980) 

s_xor_b32 s1, s0, s47 // (op 8980) 

s_sub_i32 s0, s1, s47 // (op 8980) 


s_mul_hi_i32 s1, s0, s14 // (op 8980) 
s_add_i32 s47, s1, s0 // (op 8980) 

s_ashr_i32 s1, s47, 31 // (op 8980) 
s_cmp_eq_i32 s14, 0 // (op 8980) 
s_mov_b32 s18, scc // (op 8980) 
s_cmp_eq_i32 s18, 0 // (op 8980) 
s_cselect_b32 s19, 0, -1 // (op 8980) 

s_and_b32 s18, s15, 31 // (op 8980) 
s_lshl_b32 s22, 1, s18 // (op 8980) 
s_add_i32 s23, s22, s19 // (op 8980) 


s_and_b32 s19, s1, s23 // (op 8980) 


s_add_i32 s1, s47, s19 // (op 8980) 


s_ashr_i32 s19, s1, s18 // (op 8980) 


s_lshl_b32 s1, s15, 24 // (op 8980) 
s_ashr_i32 s18, s1, 31 // (op 8980) 

s_xor_b32 s1, s19, s18 // (op 8980) 

s_sub_i32 s19, s1, s18 // (op 8980) 


s_add_u32 s22, s6, 64 // (op 8980) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8980) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8980) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8980) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 8980) 

s_mul_i32 s1, s19, s22 // (op 8980) 

s_sub_i32 s47, s0, s1 // (op 8980) 


s_mul_hi_i32 s0, s16, s9 // (op 8980) 
s_add_i32 s1, s0, s16 // (op 8980) 

s_ashr_i32 s0, s1, 31 // (op 8980) 
s_cmp_eq_i32 s9, 0 // (op 8980) 
s_mov_b32 s18, scc // (op 8980) 
s_cmp_eq_i32 s18, 0 // (op 8980) 
s_cselect_b32 s19, 0, -1 // (op 8980) 

s_and_b32 s18, s45, 31 // (op 8980) 
s_lshl_b32 s24, 1, s18 // (op 8980) 
s_add_i32 s25, s24, s19 // (op 8980) 


s_and_b32 s19, s0, s25 // (op 8980) 


s_add_i32 s0, s1, s19 // (op 8980) 


s_ashr_i32 s1, s0, s18 // (op 8980) 


s_lshl_b32 s0, s45, 24 // (op 8980) 
s_ashr_i32 s18, s0, 31 // (op 8980) 

s_xor_b32 s0, s1, s18 // (op 8980) 

s_sub_i32 s1, s0, s18 // (op 8980) 


s_mul_i32 s0, s11, s22 // (op 8980) 

s_mul_i32 s18, s8, s0 // (op 8980) 

s_mul_i32 s0, s1, s18 // (op 8980) 

s_sub_u32 s18, s16, s0 // (op 8980) 

s_mul_hi_i32 s0, s18, s12 // (op 8980) 
s_add_i32 s19, s0, s18 // (op 8980) 


s_ashr_i32 s0, s19, 31 // (op 8980) 
s_cmp_eq_i32 s12, 0 // (op 8980) 
s_mov_b32 s18, scc // (op 8980) 
s_cmp_eq_i32 s18, 0 // (op 8980) 
s_cselect_b32 s22, 0, -1 // (op 8980) 

s_and_b32 s18, s44, 31 // (op 8980) 
s_lshl_b32 s23, 1, s18 // (op 8980) 
s_add_i32 s24, s23, s22 // (op 8980) 


s_and_b32 s25, s0, s24 // (op 8980) 


s_add_i32 s0, s19, s25 // (op 8980) 


s_ashr_i32 s19, s0, s18 // (op 8980) 


s_lshl_b32 s0, s44, 24 // (op 8980) 
s_ashr_i32 s18, s0, 31 // (op 8980) 

s_xor_b32 s0, s19, s18 // (op 8980) 

s_sub_i32 s19, s0, s18 // (op 8980) 


s_cmp_eq_i32 s1, 0 // (op 8980) 
s_mov_b32 s0, scc // (op 8980) 

s_cmp_eq_i32 s0, 0 // (op 8980) 
s_cselect_b32 s1, s19, s47 // (op 8980) 



v_lshl_add_u32 v22, s1, 6, v18 // (op 8980) 


v_mul_lo_u32 v18, v22, s38 // (op 8980) 

v_add_u32 v22, v23, v18 // (op 8980) 


s_mul_i32 s1, s38, s7 // (op 8982) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 8982) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 8982) least significant
s_mul_hi_u32 s18, s38, s6 // (op 8982) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8982) most significant: sum
s_add_u32 s1, s1, s18 // (op 8982) most significant: sum


s_mov_b32 s68, s40 // (op 8982) reg expression
s_mov_b32 s69, s41 // (op 8982) reg expression
s_mov_b32 s70, s0 // (op 8982) reg expression
s_mov_b32 s71, 131072 // (op 8982) call()


v_mov_b64 v[24:25], s[30:31] // (op 8289) convert

v_mov_b32 v18, v16 // (op 8289) 
buffer_load_dwordx4 v[26:29], v18, s[60:63], 0 offen // (op 8289) Load value
v_add_u32 v18, v18, v24 // (op 8289) 
buffer_load_dwordx4 v[30:33], v18, s[60:63], 0 offen // (op 8289) Load value





buffer_load_ubyte v18, v21, s[68:71], 0 offen offset: 0 // (op 8292) Load value

s_lshl_b64 s[0:1], s[26:27], 2 // (op 8295) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 8295) 


v_mov_b32 v23, v14 // (op 8295) 
buffer_load_dwordx4 v[34:37], v23, s[48:51], 0 offen // (op 8295) Load value
v_add_u32 v23, v23, v24 // (op 8295) 
buffer_load_dwordx4 v[38:41], v23, s[48:51], 0 offen // (op 8295) Load value
v_add_u32 v23, v23, v24 // (op 8295) 
buffer_load_dwordx4 v[42:45], v23, s[48:51], 0 offen // (op 8295) Load value
v_add_u32 v23, v23, v24 // (op 8295) 
buffer_load_dwordx4 v[46:49], v23, s[48:51], 0 offen // (op 8295) Load value





buffer_load_dword v23, v19, s[64:67], 0 offen // (op 8298) Load value

v_mov_b64 v[24:25], s[30:31] // (op 8301) convert

v_mov_b32 v50, v16 // (op 8301) 
buffer_load_dwordx4 v[52:55], v50, s[60:63], 0 offen offset:128 // (op 8301) Load value
v_add_u32 v50, v50, v24 // (op 8301) 
buffer_load_dwordx4 v[56:59], v50, s[60:63], 0 offen offset:128 // (op 8301) Load value






buffer_load_ubyte v16, v21, s[68:71], 0 offen offset:4 // (op 8304) Load value


s_lshl_b64 s[0:1], s[26:27], 2 // (op 8307) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 8307) 


v_mov_b32 v21, v14 // (op 8307) 
buffer_load_dwordx4 v[60:63], v21, s[48:51], 0 offen offset:64 // (op 8307) Load value
v_add_u32 v21, v21, v24 // (op 8307) 
buffer_load_dwordx4 v[64:67], v21, s[48:51], 0 offen offset:64 // (op 8307) Load value
v_add_u32 v21, v21, v24 // (op 8307) 
buffer_load_dwordx4 v[68:71], v21, s[48:51], 0 offen offset:64 // (op 8307) Load value
v_add_u32 v21, v21, v24 // (op 8307) 
buffer_load_dwordx4 v[72:75], v21, s[48:51], 0 offen offset:64 // (op 8307) Load value






buffer_load_dword v14, v19, s[64:67], 0 offen offset:4 // (op 8310) Load value

v_and_b32 v19, -8, v1 // (op 9110) 
v_sub_u32 v21, v1, v19 // (op 9110) 

v_lshlrev_b32 v19, 4, v21 // (op 9110) 

v_lshrrev_b32 v21, 3, v1 // (op 9110) 
v_lshl_add_u32 v24, v21, 8, v19 // (op 9110) 





s_waitcnt vmcnt(15)
ds_write_b128 v24, v[26:29] // (op 8312) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v24, v[30:33] offset:128 // (op 8312) Store local data 

v_and_b32 v19, -4, v1 // (op 9226) 
v_sub_u32 v21, v1, v19 // (op 9226) 

v_and_b32 v19, 4294967292, v1 // (op 9226) 
v_add_u32 v24, v19, v21 // (op 9226) 





s_waitcnt vmcnt(13)
ds_write_b8 v24, v18 offset:8192 // (op 8313) Store local data 

v_and_b32 v19, -4, v1 // (op 9052) 
v_sub_u32 v21, v1, v19 // (op 9052) 

v_lshlrev_b32 v19, 5, v21 // (op 9052) 

v_lshrrev_b32 v21, 2, v1 // (op 9052) 
v_lshl_add_u32 v24, v21, 9, v19 // (op 9052) 


v_lshlrev_b32 v19, 2, v24 // (op 9052) 

v_lshrrev_b32 v21, 3, v19 // (op 9052) 




s_waitcnt vmcnt(12)
ds_write_b128 v21, v[34:37] offset:8448 // (op 8314) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v21, v[38:41] offset:8512 // (op 8314) Store local data 
s_waitcnt vmcnt(10)
ds_write_b128 v21, v[42:45] offset:8576 // (op 8314) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v21, v[46:49] offset:8640 // (op 8314) Store local data 

v_lshlrev_b32 v19, 2, v1 // (op 9168) 



s_waitcnt vmcnt(8)
ds_write_b32 v19, v23 offset:24832 // (op 8315) Store local data 

v_and_b32 v19, -4, v1 // (op 9036) 
v_sub_u32 v21, v1, v19 // (op 9036) 

v_lshlrev_b32 v19, 5, v21 // (op 9036) 

v_lshrrev_b32 v21, 2, v1 // (op 9036) 
v_lshl_add_u32 v24, v21, 9, v19 // (op 9036) 


v_lshlrev_b32 v19, 2, v24 // (op 9036) 

v_lshrrev_b32 v21, 3, v19 // (op 9036) 

v_and_b32 v19, -8, v1 // (op 9094) 
v_sub_u32 v24, v1, v19 // (op 9094) 

v_lshlrev_b32 v19, 4, v24 // (op 9094) 

v_lshrrev_b32 v24, 3, v1 // (op 9094) 
v_lshl_add_u32 v25, v24, 8, v19 // (op 9094) 


v_lshlrev_b32 v19, 2, v1 // (op 9152) 
v_and_b32 v24, -4, v1 // (op 9210) 
v_sub_u32 v50, v1, v24 // (op 9210) 

v_and_b32 v24, 4294967292, v1 // (op 9210) 
v_add_u32 v51, v24, v50 // (op 9210) 


v_and_b32 v24, -4, v1 // (op 9278) 
v_sub_u32 v50, v1, v24 // (op 9278) 

v_lshlrev_b32 v24, 5, v50 // (op 9278) 

v_lshrrev_b32 v50, 2, v1 // (op 9278) 
v_lshl_add_u32 v76, v50, 9, v24 // (op 9278) 


v_lshlrev_b32 v24, 2, v76 // (op 9278) 

v_lshrrev_b32 v50, 3, v24 // (op 9278) 

v_and_b32 v24, -8, v1 // (op 9307) 
v_sub_u32 v76, v1, v24 // (op 9307) 

v_lshlrev_b32 v24, 4, v76 // (op 9307) 

v_lshrrev_b32 v76, 3, v1 // (op 9307) 
v_lshl_add_u32 v77, v76, 8, v24 // (op 9307) 


v_lshlrev_b32 v24, 2, v1 // (op 9336) 
v_and_b32 v76, -4, v1 // (op 9365) 
v_sub_u32 v78, v1, v76 // (op 9365) 

v_and_b32 v76, 4294967292, v1 // (op 9365) 
v_add_u32 v79, v76, v78 // (op 9365) 


s_mov_b32 s47, 0 // (op 2909) call()
s_ashr_i64 s[0:1], s[42:43], 7 // (op 2908) 
s_mov_b32 s18, s0 // (op 2908) convert

s_and_b32 s19, s18, 4294967294 // (op 2908) 

s_cmp_lt_u32 s47, s19 // (op 2908) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_2908 // (op 2908) Condition: Top (jump to RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_2908 if false)
RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_2908:

s_barrier  // (op 9469) 
s_add_u32 s47, s47, 2 // (op 2910) 



v_mov_b64 v[80:81], s[30:31] // (op 10) convert

v_mov_b32 v76, v17 // (op 10) 
v_mov_b32 v78, v76 // (op 10) 
buffer_load_dwordx4 v[26:29], v78, s[60:63], 0 offen offset:256 // (op 10) Load value
v_add_u32 v78, v78, v80 // (op 10) 
buffer_load_dwordx4 v[30:33], v78, s[60:63], 0 offen offset:256 // (op 10) Load value


ds_read_b128 v[80:83], v3 offset:8448 // (op 72) Load local data 


s_barrier  // (op 9482) 
v_mov_b32 v78, v22 // (op 25) 




ds_read_b128 v[84:87], v4 // (op 77) Load local data 
ds_read_b128 v[88:91], v4 offset:64 // (op 77) Load local data 
buffer_load_ubyte v18, v78, s[68:71], 0 offen offset:8 // (op 25) Load value

s_barrier  // (op 9496) 
s_lshl_b64 s[0:1], s[26:27], 2 // (op 4) 



ds_read_u8 v92, v0 offset:24832 // (op 82) Load local data 
v_lshrrev_b64 v[94:95], 3, s[0:1] // (op 4) 


s_barrier  // (op 9510) 
v_mov_b32 v93, v15 // (op 4) 



ds_read_u8 v96, v5 offset:8192 // (op 87) Load local data 
v_mov_b32 v97, v93 // (op 4) 



ds_read_b128 v[98:101], v4 offset:4096 // (op 375) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:83], v[84:91], a[0:3], v92, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 90) 
ds_read_b128 v[102:105], v4 offset:4160 // (op 375) Load local data 
buffer_load_dwordx4 v[34:37], v97, s[48:51], 0 offen offset:128 // (op 4) Load value



v_add_u32 v97, v97, v94 // (op 4) 
buffer_load_dwordx4 v[38:41], v97, s[48:51], 0 offen offset:128 // (op 4) Load value
v_add_u32 v97, v97, v94 // (op 4) 
ds_read_u8 v106, v5 offset:8320 // (op 379) Load local data 
buffer_load_dwordx4 v[42:45], v97, s[48:51], 0 offen offset:128 // (op 4) Load value



v_add_u32 v97, v97, v94 // (op 4) 
ds_read_b128 v[108:111], v3 offset:10496 // (op 626) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[80:83], v[98:105], a[4:7], v92, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 380) 





ds_read_u8 v92, v0 offset:24960 // (op 629) Load local data 
buffer_load_dwordx4 v[46:49], v97, s[48:51], 0 offen offset:128 // (op 4) Load value







v_mov_b32 v97, v20 // (op 17) 

buffer_load_dword v23, v97, s[64:67], 0 offen offset:8 // (op 17) Load value
ds_read_b128 v[80:83], v3 offset:12544 // (op 1012) Load local data 



s_barrier  // (op 9533) 
ds_read_u8 v107, v0 offset:25088 // (op 1015) Load local data 






ds_read_b128 v[112:115], v3 offset:14592 // (op 1398) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[84:91], a[8:11], v92, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 677) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[98:105], a[12:15], v92, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 631) 
ds_read_u8 v94, v0 offset:25216 // (op 1401) Load local data 





ds_read_b128 v[108:111], v3 offset:16640 // (op 1784) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[80:83], v[84:91], a[16:19], v107, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1063) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[80:83], v[98:105], a[20:23], v107, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1017) 
ds_read_u8 v92, v0 offset:25344 // (op 1787) Load local data 





s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[112:115], v[84:91], a[24:27], v94, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1449) 
ds_read_b128 v[80:83], v3 offset:18688 // (op 2170) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[112:115], v[98:105], a[28:31], v94, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1403) 
ds_read_u8 v95, v0 offset:25472 // (op 2173) Load local data 





ds_read_b128 v[112:115], v3 offset:20736 // (op 2556) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[108:111], v[84:91], a[32:35], v92, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1835) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[108:111], v[98:105], a[36:39], v92, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1789) 
ds_read_u8 v94, v0 offset:25600 // (op 2559) Load local data 





s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[80:83], v[84:91], a[40:43], v95, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2221) 
ds_read_b128 v[108:111], v3 offset:22784 // (op 2942) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[80:83], v[98:105], a[44:47], v95, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2175) 
ds_read_u8 v92, v0 offset:25728 // (op 2945) Load local data 


s_waitcnt vmcnt(15)
ds_write_b128 v77, v[52:55] offset:25856 // (op 257) Store local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[112:115], v[84:91], a[48:51], v94, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2607) 
s_waitcnt vmcnt(14)
ds_write_b128 v77, v[56:59] offset:25984 // (op 257) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[98:105], a[52:55], v94, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2561) 


s_barrier  // (op 9545) 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[108:111], v[84:91], a[56:59], v92, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2993) 
s_waitcnt vmcnt(13)
ds_write_b8 v79, v16 offset:34048 // (op 259) Store local data 


s_barrier  // (op 9527) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[108:111], v[98:105], a[60:63], v92, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2947) 







s_waitcnt vmcnt(12)
ds_write_b128 v50, v[60:63] offset:34304 // (op 247) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v50, v[64:67] offset:34368 // (op 247) Store local data 
s_waitcnt vmcnt(10)
ds_write_b128 v50, v[68:71] offset:34432 // (op 247) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v50, v[72:75] offset:34496 // (op 247) Store local data 
s_barrier  // (op 9539) 



s_waitcnt vmcnt(8)
ds_write_b32 v24, v14 offset:50688 // (op 258) Store local data 

v_mov_b64 v[80:81], s[30:31] // (op 260) convert

v_mov_b32 v82, v76 // (op 260) 
buffer_load_dwordx4 v[52:55], v82, s[60:63], 0 offen offset:384 // (op 260) Load value
v_add_u32 v82, v82, v80 // (op 260) 
buffer_load_dwordx4 v[56:59], v82, s[60:63], 0 offen offset:384 // (op 260) Load value


s_waitcnt lgkmcnt(1)
s_barrier  // (op 9524) 







buffer_load_ubyte v16, v78, s[68:71], 0 offen offset:12 // (op 262) Load value
v_add_u32 v17, 256, v17 // (op 8844) 

ds_read_b128 v[80:83], v7 offset:34304 // (op 249) Load local data 

s_barrier  // (op 9530) 
v_add_u32 v22, 8, v22 // (op 8972) 



s_lshl_b64 s[0:1], s[26:27], 2 // (op 246) 
v_lshrrev_b64 v[84:85], 3, s[0:1] // (op 246) 


v_mov_b32 v76, v93 // (op 246) 
ds_read_b128 v[98:101], v8 offset:25856 // (op 251) Load local data 
ds_read_b128 v[102:105], v8 offset:25920 // (op 251) Load local data 
buffer_load_dwordx4 v[60:63], v76, s[48:51], 0 offen offset:192 // (op 246) Load value
v_add_u32 v76, v76, v84 // (op 246) 
buffer_load_dwordx4 v[64:67], v76, s[48:51], 0 offen offset:192 // (op 246) Load value
v_add_u32 v76, v76, v84 // (op 246) 
buffer_load_dwordx4 v[68:71], v76, s[48:51], 0 offen offset:192 // (op 246) Load value
v_add_u32 v76, v76, v84 // (op 246) 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 9536) 
buffer_load_dwordx4 v[72:75], v76, s[48:51], 0 offen offset:192 // (op 246) Load value






ds_read_u8 v76, v6 offset:50688 // (op 253) Load local data 



s_barrier  // (op 9542) 
buffer_load_dword v14, v97, s[64:67], 0 offen offset:12 // (op 261) Load value




v_add_u32 v15, 128, v15 // (op 8780) 
s_barrier  // (op 9485) 
v_add_u32 v20, 8, v20 // (op 8908) 



ds_read_u8 v78, v9 offset:34048 // (op 255) Load local data 
s_waitcnt vmcnt(15)
ds_write_b128 v25, v[26:29] // (op 13) Store local data 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:83], v[98:105], a[0:3], v76, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 256) 
ds_read_b128 v[84:87], v8 offset:29952 // (op 385) Load local data 
ds_read_b128 v[88:91], v8 offset:30016 // (op 385) Load local data 
s_waitcnt vmcnt(14)
ds_write_b128 v25, v[30:33] offset:128 // (op 13) Store local data 



s_barrier  // (op 9606) 
s_barrier  // (op 9513) 
ds_read_u8 v92, v9 offset:34176 // (op 389) Load local data 



s_waitcnt vmcnt(13)
ds_write_b8 v51, v18 offset:8192 // (op 30) Store local data 



s_barrier  // (op 9610) 
s_barrier  // (op 9472) 
ds_read_b128 v[94:97], v7 offset:36352 // (op 634) Load local data 






s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[80:83], v[84:91], a[4:7], v76, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 381) 
ds_read_u8 v93, v6 offset:50816 // (op 637) Load local data 





ds_read_b128 v[80:83], v7 offset:38400 // (op 1020) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[94:97], v[98:105], a[8:11], v93, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 678) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[94:97], v[84:91], a[12:15], v93, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 632) 
ds_read_u8 v76, v6 offset:50944 // (op 1023) Load local data 





ds_read_b128 v[94:97], v7 offset:40448 // (op 1406) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[80:83], v[98:105], a[16:19], v76, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1064) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[80:83], v[84:91], a[20:23], v76, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1018) 
ds_read_u8 v93, v6 offset:51072 // (op 1409) Load local data 





ds_read_b128 v[80:83], v7 offset:42496 // (op 1792) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[94:97], v[98:105], a[24:27], v93, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1450) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[94:97], v[84:91], a[28:31], v93, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1404) 
ds_read_u8 v76, v6 offset:51200 // (op 1795) Load local data 





ds_read_b128 v[94:97], v7 offset:44544 // (op 2178) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:83], v[98:105], a[32:35], v76, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1836) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:83], v[84:91], a[36:39], v76, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1790) 
ds_read_u8 v93, v6 offset:51328 // (op 2181) Load local data 





ds_read_b128 v[80:83], v7 offset:46592 // (op 2564) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[94:97], v[98:105], a[40:43], v93, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2222) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[94:97], v[84:91], a[44:47], v93, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2176) 
ds_read_u8 v76, v6 offset:51456 // (op 2567) Load local data 





ds_read_b128 v[94:97], v7 offset:48640 // (op 2950) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[98:105], a[48:51], v76, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2608) 



s_barrier  // (op 9666) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[84:91], a[52:55], v76, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2562) 
ds_read_u8 v93, v6 offset:51584 // (op 2953) Load local data 


s_waitcnt vmcnt(12)
ds_write_b128 v21, v[34:37] offset:8448 // (op 6) Store local data 
s_barrier  // (op 9670) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[94:97], v[98:105], a[56:59], v93, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2994) 
s_waitcnt vmcnt(11)
ds_write_b128 v21, v[38:41] offset:8512 // (op 6) Store local data 


s_waitcnt vmcnt(10)
ds_write_b128 v21, v[42:45] offset:8576 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[94:97], v[84:91], a[60:63], v93, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2948) 
s_waitcnt vmcnt(9)
ds_write_b128 v21, v[46:49] offset:8640 // (op 6) Store local data 




s_barrier  // (op 9499) 



s_waitcnt vmcnt(8)
ds_write_b32 v19, v23 offset:24832 // (op 21) Store local data 
s_ashr_i64 s[0:1], s[42:43], 7 // (op 2908) 
s_mov_b32 s18, s0 // (op 2908) convert

s_and_b32 s19, s18, 4294967294 // (op 2908) 

s_cmp_lt_u32 s47, s19 // (op 2908) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_2908 // (op 2908) Condition: Bottom (jump to RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_2908 if true)
RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_2908:

















s_mul_hi_i32 s47, s16, s10 // (op 8728) 
v_and_b32 v15, 4294967292, v1 // (op 8728) 
s_add_i32 s0, s47, s16 // (op 8728) 

v_and_b32 v17, -4, v1 // (op 8728) 
s_ashr_i32 s1, s0, 31 // (op 8728) 
v_sub_u32 v18, v1, v17 // (op 8728) 

s_cmp_eq_i32 s10, 0 // (op 8728) 
s_mov_b32 s47, scc // (op 8728) 
v_lshlrev_b32 v17, 5, v18 // (op 8728) 

s_cmp_eq_i32 s47, 0 // (op 8728) 
s_cselect_b32 s18, 0, -1 // (op 8728) 

s_and_b32 s19, s13, 31 // (op 8728) 
s_lshl_b32 s47, 1, s19 // (op 8728) 
s_add_i32 s22, s47, s18 // (op 8728) 


s_and_b32 s18, s1, s22 // (op 8728) 


s_add_i32 s1, s0, s18 // (op 8728) 


s_ashr_i32 s0, s1, s19 // (op 8728) 


s_lshl_b32 s1, s13, 24 // (op 8728) 
s_ashr_i32 s47, s1, 31 // (op 8728) 

s_xor_b32 s1, s0, s47 // (op 8728) 

s_sub_i32 s0, s1, s47 // (op 8728) 


s_mul_hi_i32 s1, s0, s14 // (op 8728) 
s_add_i32 s47, s1, s0 // (op 8728) 

s_ashr_i32 s1, s47, 31 // (op 8728) 
s_cmp_eq_i32 s14, 0 // (op 8728) 
s_mov_b32 s18, scc // (op 8728) 
s_cmp_eq_i32 s18, 0 // (op 8728) 
s_cselect_b32 s19, 0, -1 // (op 8728) 

s_and_b32 s18, s15, 31 // (op 8728) 
s_lshl_b32 s22, 1, s18 // (op 8728) 
s_add_i32 s23, s22, s19 // (op 8728) 


s_and_b32 s19, s1, s23 // (op 8728) 


s_add_i32 s1, s47, s19 // (op 8728) 


s_ashr_i32 s19, s1, s18 // (op 8728) 


s_lshl_b32 s1, s15, 24 // (op 8728) 
s_ashr_i32 s18, s1, 31 // (op 8728) 

s_xor_b32 s1, s19, s18 // (op 8728) 

s_sub_i32 s19, s1, s18 // (op 8728) 


s_mul_i32 s1, s0, s11 // (op 8728) 

s_sub_u32 s0, s16, s1 // (op 8728) 

s_mul_i32 s1, s19, s11 // (op 8728) 
s_add_u32 s1, s1, s0 // (op 8728) 


s_mul_hi_i32 s0, s16, s9 // (op 8728) 
s_add_i32 s47, s0, s16 // (op 8728) 

s_ashr_i32 s0, s47, 31 // (op 8728) 
s_cmp_eq_i32 s9, 0 // (op 8728) 
s_mov_b32 s18, scc // (op 8728) 
s_cmp_eq_i32 s18, 0 // (op 8728) 
s_cselect_b32 s19, 0, -1 // (op 8728) 

s_and_b32 s18, s45, 31 // (op 8728) 
s_lshl_b32 s22, 1, s18 // (op 8728) 
s_add_i32 s23, s22, s19 // (op 8728) 


s_and_b32 s19, s0, s23 // (op 8728) 


s_add_i32 s0, s47, s19 // (op 8728) 


s_ashr_i32 s19, s0, s18 // (op 8728) 


s_lshl_b32 s0, s45, 24 // (op 8728) 
s_ashr_i32 s18, s0, 31 // (op 8728) 

s_xor_b32 s0, s19, s18 // (op 8728) 

s_sub_i32 s19, s0, s18 // (op 8728) 


s_add_u32 s22, s6, 64 // (op 8728) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8728) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8728) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8728) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 8728) 

s_mul_i32 s0, s11, s22 // (op 8728) 

s_mul_i32 s18, s8, s0 // (op 8728) 

s_mul_i32 s0, s19, s18 // (op 8728) 

s_sub_u32 s18, s16, s0 // (op 8728) 

s_mov_b32 s0, s18 // (op 8728) 
s_mul_hi_i32 s47, s0, s12 // (op 8728) 
s_add_i32 s22, s47, s0 // (op 8728) 


s_ashr_i32 s0, s22, 31 // (op 8728) 
s_cmp_eq_i32 s12, 0 // (op 8728) 
s_mov_b32 s47, scc // (op 8728) 
s_cmp_eq_i32 s47, 0 // (op 8728) 
s_cselect_b32 s23, 0, -1 // (op 8728) 

s_and_b32 s47, s44, 31 // (op 8728) 
s_lshl_b32 s24, 1, s47 // (op 8728) 
s_add_i32 s25, s24, s23 // (op 8728) 


s_and_b32 s23, s0, s25 // (op 8728) 


s_add_i32 s0, s22, s23 // (op 8728) 


s_ashr_i32 s22, s0, s47 // (op 8728) 


s_lshl_b32 s0, s44, 24 // (op 8728) 
s_ashr_i32 s47, s0, 31 // (op 8728) 

s_xor_b32 s0, s22, s47 // (op 8728) 

s_sub_i32 s22, s0, s47 // (op 8728) 


s_mul_i32 s0, s46, s11 // (op 8728) 
s_add_u32 s24, s20, 256 // (op 8728) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8728) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 8728) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 8728) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 8 // (op 8728) 

s_sub_i32 s23, s24, s0 // (op 8728) 

s_mul_i32 s47, s22, s23 // (op 8728) 


s_sub_u32 s22, s18, s47 // (op 8728) 


s_add_u32 s18, s22, s0 // (op 8728) 


s_cmp_eq_i32 s19, 0 // (op 8728) 
s_mov_b32 s0, scc // (op 8728) 

s_cmp_eq_i32 s0, 0 // (op 8728) 
s_cselect_b32 s19, s18, s1 // (op 8728) 



v_lshl_add_u32 v18, s19, 8, v15 // (op 8728) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 8728) 
s_lshr_b32 s47, s0, 1 // (op 8728) 

v_lshl_add_u32 v15, s47, 8, v17 // (op 8728) 


v_mov_b32 v51, 0 // (op 8728) convert
v_mov_b32 v50, v15 // (op 8728) convert

v_mul_lo_u32 v49, v18, s27 // (op 8728) most significant: low of low * high
v_mul_hi_u32 v15, v18, s26 // (op 8728) most significant: high of low * low
v_mul_lo_u32 v48, v18, s26 // (op 8728) least significant: low of low * low
v_add_u32 v49, v49, v15 // (op 8728) most significant: sum

v_add_co_u32 v48, s[0:1], v48, v50 // (op 8728) least significant half
v_addc_co_u32 v49, s[0:1], v49, v51, s[0:1] // (op 8728) most significant half



v_lshlrev_b64 v[50:51], 2, v[48:49] // (op 8728) 

v_lshrrev_b64 v[48:49], 3, v[50:51] // (op 8728) 

v_mov_b32 v15, v48 // (op 8728) convert

s_mul_i32 s1, s26, s21 // (op 8730) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 8730) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 8730) least significant
s_mul_hi_u32 s18, s26, s20 // (op 8730) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8730) most significant: sum
s_add_u32 s1, s1, s18 // (op 8730) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 8730) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 8730) 

s_mov_b32 s48, s28 // (op 8730) reg expression
s_mov_b32 s49, s29 // (op 8730) reg expression
s_mov_b32 s50, s0 // (op 8730) reg expression
s_mov_b32 s51, 131072 // (op 8730) call()

s_mul_hi_i32 s47, s16, s10 // (op 8810) 
v_lshrrev_b32 v17, 3, v1 // (op 8810) 
s_add_i32 s0, s47, s16 // (op 8810) 

v_lshlrev_b32 v18, 1, v17 // (op 8810) 

s_ashr_i32 s1, s0, 31 // (op 8810) 
v_and_b32 v17, -8, v1 // (op 8810) 
s_cmp_eq_i32 s10, 0 // (op 8810) 
s_mov_b32 s47, scc // (op 8810) 
v_sub_u32 v19, v1, v17 // (op 8810) 

s_cmp_eq_i32 s47, 0 // (op 8810) 
s_cselect_b32 s18, 0, -1 // (op 8810) 

v_lshlrev_b32 v17, 4, v19 // (op 8810) 

s_and_b32 s19, s13, 31 // (op 8810) 
s_lshl_b32 s47, 1, s19 // (op 8810) 
s_add_i32 s22, s47, s18 // (op 8810) 


s_and_b32 s18, s1, s22 // (op 8810) 


s_add_i32 s1, s0, s18 // (op 8810) 


s_ashr_i32 s0, s1, s19 // (op 8810) 


s_lshl_b32 s1, s13, 24 // (op 8810) 
s_ashr_i32 s47, s1, 31 // (op 8810) 

s_xor_b32 s1, s0, s47 // (op 8810) 

s_sub_i32 s0, s1, s47 // (op 8810) 


s_mul_hi_i32 s1, s0, s14 // (op 8810) 
s_add_i32 s47, s1, s0 // (op 8810) 

s_ashr_i32 s1, s47, 31 // (op 8810) 
s_cmp_eq_i32 s14, 0 // (op 8810) 
s_mov_b32 s18, scc // (op 8810) 
s_cmp_eq_i32 s18, 0 // (op 8810) 
s_cselect_b32 s19, 0, -1 // (op 8810) 

s_and_b32 s18, s15, 31 // (op 8810) 
s_lshl_b32 s22, 1, s18 // (op 8810) 
s_add_i32 s23, s22, s19 // (op 8810) 


s_and_b32 s19, s1, s23 // (op 8810) 


s_add_i32 s1, s47, s19 // (op 8810) 


s_ashr_i32 s19, s1, s18 // (op 8810) 


s_lshl_b32 s1, s15, 24 // (op 8810) 
s_ashr_i32 s18, s1, 31 // (op 8810) 

s_xor_b32 s1, s19, s18 // (op 8810) 

s_sub_i32 s19, s1, s18 // (op 8810) 


s_add_u32 s22, s6, 64 // (op 8810) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8810) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8810) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8810) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 8810) 

s_mul_i32 s1, s19, s22 // (op 8810) 

s_sub_i32 s47, s0, s1 // (op 8810) 


s_mul_hi_i32 s0, s16, s9 // (op 8810) 
s_add_i32 s1, s0, s16 // (op 8810) 

s_ashr_i32 s0, s1, 31 // (op 8810) 
s_cmp_eq_i32 s9, 0 // (op 8810) 
s_mov_b32 s18, scc // (op 8810) 
s_cmp_eq_i32 s18, 0 // (op 8810) 
s_cselect_b32 s19, 0, -1 // (op 8810) 

s_and_b32 s18, s45, 31 // (op 8810) 
s_lshl_b32 s24, 1, s18 // (op 8810) 
s_add_i32 s25, s24, s19 // (op 8810) 


s_and_b32 s19, s0, s25 // (op 8810) 


s_add_i32 s0, s1, s19 // (op 8810) 


s_ashr_i32 s1, s0, s18 // (op 8810) 


s_lshl_b32 s0, s45, 24 // (op 8810) 
s_ashr_i32 s18, s0, 31 // (op 8810) 

s_xor_b32 s0, s1, s18 // (op 8810) 

s_sub_i32 s1, s0, s18 // (op 8810) 


s_mul_i32 s0, s11, s22 // (op 8810) 

s_mul_i32 s18, s8, s0 // (op 8810) 

s_mul_i32 s0, s1, s18 // (op 8810) 

s_sub_u32 s18, s16, s0 // (op 8810) 

s_mul_hi_i32 s0, s18, s12 // (op 8810) 
s_add_i32 s19, s0, s18 // (op 8810) 


s_ashr_i32 s0, s19, 31 // (op 8810) 
s_cmp_eq_i32 s12, 0 // (op 8810) 
s_mov_b32 s18, scc // (op 8810) 
s_cmp_eq_i32 s18, 0 // (op 8810) 
s_cselect_b32 s22, 0, -1 // (op 8810) 

s_and_b32 s18, s44, 31 // (op 8810) 
s_lshl_b32 s23, 1, s18 // (op 8810) 
s_add_i32 s24, s23, s22 // (op 8810) 


s_and_b32 s25, s0, s24 // (op 8810) 


s_add_i32 s0, s19, s25 // (op 8810) 


s_ashr_i32 s19, s0, s18 // (op 8810) 


s_lshl_b32 s0, s44, 24 // (op 8810) 
s_ashr_i32 s18, s0, 31 // (op 8810) 

s_xor_b32 s0, s19, s18 // (op 8810) 

s_sub_i32 s19, s0, s18 // (op 8810) 


s_cmp_eq_i32 s1, 0 // (op 8810) 
s_mov_b32 s0, scc // (op 8810) 

s_cmp_eq_i32 s0, 0 // (op 8810) 
s_cselect_b32 s1, s19, s47 // (op 8810) 



v_lshl_add_u32 v19, s1, 6, v18 // (op 8810) 


v_mul_lo_u32 v18, v19, s30 // (op 8810) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 8810) 
s_lshr_b32 s47, s0, 1 // (op 8810) 

v_lshl_add_u32 v19, s47, 8, v17 // (op 8810) 


v_add_u32 v17, v19, v18 // (op 8810) 


s_mul_i32 s1, s30, s7 // (op 8812) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 8812) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 8812) least significant
s_mul_hi_u32 s18, s30, s6 // (op 8812) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8812) most significant: sum
s_add_u32 s1, s1, s18 // (op 8812) most significant: sum


s_mov_b32 s60, s32 // (op 8812) reg expression
s_mov_b32 s61, s33 // (op 8812) reg expression
s_mov_b32 s62, s0 // (op 8812) reg expression
s_mov_b32 s63, 131072 // (op 8812) call()

s_mul_hi_i32 s47, s16, s10 // (op 8874) 
s_add_i32 s0, s47, s16 // (op 8874) 

s_ashr_i32 s1, s0, 31 // (op 8874) 
s_cmp_eq_i32 s10, 0 // (op 8874) 
s_mov_b32 s47, scc // (op 8874) 
s_cmp_eq_i32 s47, 0 // (op 8874) 
s_cselect_b32 s18, 0, -1 // (op 8874) 

s_and_b32 s19, s13, 31 // (op 8874) 
s_lshl_b32 s47, 1, s19 // (op 8874) 
s_add_i32 s22, s47, s18 // (op 8874) 


s_and_b32 s18, s1, s22 // (op 8874) 


s_add_i32 s1, s0, s18 // (op 8874) 


s_ashr_i32 s0, s1, s19 // (op 8874) 


s_lshl_b32 s1, s13, 24 // (op 8874) 
s_ashr_i32 s47, s1, 31 // (op 8874) 

s_xor_b32 s1, s0, s47 // (op 8874) 

s_sub_i32 s0, s1, s47 // (op 8874) 


s_mul_hi_i32 s1, s0, s14 // (op 8874) 
s_add_i32 s47, s1, s0 // (op 8874) 

s_ashr_i32 s1, s47, 31 // (op 8874) 
s_cmp_eq_i32 s14, 0 // (op 8874) 
s_mov_b32 s18, scc // (op 8874) 
s_cmp_eq_i32 s18, 0 // (op 8874) 
s_cselect_b32 s19, 0, -1 // (op 8874) 

s_and_b32 s18, s15, 31 // (op 8874) 
s_lshl_b32 s22, 1, s18 // (op 8874) 
s_add_i32 s23, s22, s19 // (op 8874) 


s_and_b32 s19, s1, s23 // (op 8874) 


s_add_i32 s1, s47, s19 // (op 8874) 


s_ashr_i32 s19, s1, s18 // (op 8874) 


s_lshl_b32 s1, s15, 24 // (op 8874) 
s_ashr_i32 s18, s1, 31 // (op 8874) 

s_xor_b32 s1, s19, s18 // (op 8874) 

s_sub_i32 s19, s1, s18 // (op 8874) 


s_mul_i32 s1, s0, s11 // (op 8874) 

s_sub_u32 s0, s16, s1 // (op 8874) 

s_mul_i32 s1, s19, s11 // (op 8874) 
s_add_u32 s1, s1, s0 // (op 8874) 


s_mul_hi_i32 s0, s16, s9 // (op 8874) 
s_add_i32 s47, s0, s16 // (op 8874) 

s_ashr_i32 s0, s47, 31 // (op 8874) 
s_cmp_eq_i32 s9, 0 // (op 8874) 
s_mov_b32 s18, scc // (op 8874) 
s_cmp_eq_i32 s18, 0 // (op 8874) 
s_cselect_b32 s19, 0, -1 // (op 8874) 

s_and_b32 s18, s45, 31 // (op 8874) 
s_lshl_b32 s22, 1, s18 // (op 8874) 
s_add_i32 s23, s22, s19 // (op 8874) 


s_and_b32 s19, s0, s23 // (op 8874) 


s_add_i32 s0, s47, s19 // (op 8874) 


s_ashr_i32 s19, s0, s18 // (op 8874) 


s_lshl_b32 s0, s45, 24 // (op 8874) 
s_ashr_i32 s18, s0, 31 // (op 8874) 

s_xor_b32 s0, s19, s18 // (op 8874) 

s_sub_i32 s19, s0, s18 // (op 8874) 


s_add_u32 s22, s6, 64 // (op 8874) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8874) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8874) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8874) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 8874) 

s_mul_i32 s0, s11, s22 // (op 8874) 

s_mul_i32 s18, s8, s0 // (op 8874) 

s_mul_i32 s0, s19, s18 // (op 8874) 

s_sub_u32 s18, s16, s0 // (op 8874) 

s_mov_b32 s0, s18 // (op 8874) 
s_mul_hi_i32 s47, s0, s12 // (op 8874) 
s_add_i32 s22, s47, s0 // (op 8874) 


s_ashr_i32 s0, s22, 31 // (op 8874) 
s_cmp_eq_i32 s12, 0 // (op 8874) 
s_mov_b32 s47, scc // (op 8874) 
s_cmp_eq_i32 s47, 0 // (op 8874) 
s_cselect_b32 s23, 0, -1 // (op 8874) 

s_and_b32 s47, s44, 31 // (op 8874) 
s_lshl_b32 s24, 1, s47 // (op 8874) 
s_add_i32 s25, s24, s23 // (op 8874) 


s_and_b32 s23, s0, s25 // (op 8874) 


s_add_i32 s0, s22, s23 // (op 8874) 


s_ashr_i32 s22, s0, s47 // (op 8874) 


s_lshl_b32 s0, s44, 24 // (op 8874) 
s_ashr_i32 s47, s0, 31 // (op 8874) 

s_xor_b32 s0, s22, s47 // (op 8874) 

s_sub_i32 s22, s0, s47 // (op 8874) 


s_mul_i32 s0, s46, s11 // (op 8874) 
s_add_u32 s24, s20, 256 // (op 8874) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8874) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 8874) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 8874) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 8 // (op 8874) 

s_sub_i32 s23, s24, s0 // (op 8874) 

s_mul_i32 s47, s22, s23 // (op 8874) 


s_sub_u32 s22, s18, s47 // (op 8874) 


s_add_u32 s18, s22, s0 // (op 8874) 


s_cmp_eq_i32 s19, 0 // (op 8874) 
s_mov_b32 s0, scc // (op 8874) 

s_cmp_eq_i32 s0, 0 // (op 8874) 
s_cselect_b32 s19, s18, s1 // (op 8874) 



v_lshl_add_u32 v18, s19, 8, v1 // (op 8874) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 8874) 
s_lshr_b32 s47, s0, 1 // (op 8874) 

s_lshl_b32 s0, s47, 3 // (op 8874) 

v_mul_lo_u32 v19, v18, s34 // (op 8874) 
v_add_u32 v19, v19, s0 // (op 8874) 


s_mul_i32 s1, s34, s21 // (op 8876) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 8876) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 8876) least significant
s_mul_hi_u32 s18, s34, s20 // (op 8876) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8876) most significant: sum
s_add_u32 s1, s1, s18 // (op 8876) most significant: sum


s_mov_b32 s64, s36 // (op 8876) reg expression
s_mov_b32 s65, s37 // (op 8876) reg expression
s_mov_b32 s66, s0 // (op 8876) reg expression
s_mov_b32 s67, 131072 // (op 8876) call()

s_mul_hi_i32 s47, s16, s10 // (op 8938) 
v_lshrrev_b32 v18, 2, v1 // (op 8938) 
s_add_i32 s0, s47, s16 // (op 8938) 

v_and_b32 v20, -4, v1 // (op 8938) 
s_ashr_i32 s1, s0, 31 // (op 8938) 
v_sub_u32 v21, v1, v20 // (op 8938) 

s_cmp_eq_i32 s10, 0 // (op 8938) 
s_mov_b32 s47, scc // (op 8938) 
s_cmp_eq_i32 s47, 0 // (op 8938) 
s_cselect_b32 s18, 0, -1 // (op 8938) 

s_and_b32 s19, s13, 31 // (op 8938) 
s_lshl_b32 s47, 1, s19 // (op 8938) 
s_add_i32 s22, s47, s18 // (op 8938) 


s_and_b32 s18, s1, s22 // (op 8938) 


s_add_i32 s1, s0, s18 // (op 8938) 


s_ashr_i32 s0, s1, s19 // (op 8938) 


s_lshl_b32 s1, s13, 24 // (op 8938) 
s_ashr_i32 s47, s1, 31 // (op 8938) 

s_xor_b32 s1, s0, s47 // (op 8938) 

s_sub_i32 s0, s1, s47 // (op 8938) 


s_mul_hi_i32 s1, s0, s14 // (op 8938) 
s_add_i32 s47, s1, s0 // (op 8938) 

s_ashr_i32 s1, s47, 31 // (op 8938) 
s_cmp_eq_i32 s14, 0 // (op 8938) 
s_mov_b32 s18, scc // (op 8938) 
s_cmp_eq_i32 s18, 0 // (op 8938) 
s_cselect_b32 s19, 0, -1 // (op 8938) 

s_and_b32 s18, s15, 31 // (op 8938) 
s_lshl_b32 s22, 1, s18 // (op 8938) 
s_add_i32 s23, s22, s19 // (op 8938) 


s_and_b32 s19, s1, s23 // (op 8938) 


s_add_i32 s1, s47, s19 // (op 8938) 


s_ashr_i32 s19, s1, s18 // (op 8938) 


s_lshl_b32 s1, s15, 24 // (op 8938) 
s_ashr_i32 s18, s1, 31 // (op 8938) 

s_xor_b32 s1, s19, s18 // (op 8938) 

s_sub_i32 s19, s1, s18 // (op 8938) 


s_add_u32 s22, s6, 64 // (op 8938) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8938) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8938) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8938) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 8938) 

s_mul_i32 s1, s19, s22 // (op 8938) 

s_sub_i32 s47, s0, s1 // (op 8938) 


s_mul_hi_i32 s0, s16, s9 // (op 8938) 
s_add_i32 s1, s0, s16 // (op 8938) 

s_ashr_i32 s0, s1, 31 // (op 8938) 
s_cmp_eq_i32 s9, 0 // (op 8938) 
s_mov_b32 s18, scc // (op 8938) 
s_cmp_eq_i32 s18, 0 // (op 8938) 
s_cselect_b32 s19, 0, -1 // (op 8938) 

s_and_b32 s18, s45, 31 // (op 8938) 
s_lshl_b32 s24, 1, s18 // (op 8938) 
s_add_i32 s25, s24, s19 // (op 8938) 


s_and_b32 s19, s0, s25 // (op 8938) 


s_add_i32 s0, s1, s19 // (op 8938) 


s_ashr_i32 s1, s0, s18 // (op 8938) 


s_lshl_b32 s0, s45, 24 // (op 8938) 
s_ashr_i32 s18, s0, 31 // (op 8938) 

s_xor_b32 s0, s1, s18 // (op 8938) 

s_sub_i32 s1, s0, s18 // (op 8938) 


s_mul_i32 s0, s11, s22 // (op 8938) 

s_mul_i32 s18, s8, s0 // (op 8938) 

s_mul_i32 s0, s1, s18 // (op 8938) 

s_sub_u32 s18, s16, s0 // (op 8938) 

s_mul_hi_i32 s0, s18, s12 // (op 8938) 
s_add_i32 s19, s0, s18 // (op 8938) 


s_ashr_i32 s0, s19, 31 // (op 8938) 
s_cmp_eq_i32 s12, 0 // (op 8938) 
s_mov_b32 s18, scc // (op 8938) 
s_cmp_eq_i32 s18, 0 // (op 8938) 
s_cselect_b32 s22, 0, -1 // (op 8938) 

s_and_b32 s18, s44, 31 // (op 8938) 
s_lshl_b32 s23, 1, s18 // (op 8938) 
s_add_i32 s24, s23, s22 // (op 8938) 


s_and_b32 s25, s0, s24 // (op 8938) 


s_add_i32 s0, s19, s25 // (op 8938) 


s_ashr_i32 s19, s0, s18 // (op 8938) 


s_lshl_b32 s0, s44, 24 // (op 8938) 
s_ashr_i32 s18, s0, 31 // (op 8938) 

s_xor_b32 s0, s19, s18 // (op 8938) 

s_sub_i32 s19, s0, s18 // (op 8938) 


s_cmp_eq_i32 s1, 0 // (op 8938) 
s_mov_b32 s0, scc // (op 8938) 

s_cmp_eq_i32 s0, 0 // (op 8938) 
s_cselect_b32 s1, s19, s47 // (op 8938) 



v_lshl_add_u32 v20, s1, 6, v18 // (op 8938) 


v_mul_lo_u32 v18, v20, s38 // (op 8938) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 8938) 
s_lshr_b32 s47, s0, 1 // (op 8938) 

v_lshl_add_u32 v20, s47, 3, v21 // (op 8938) 


v_add_u32 v21, v20, v18 // (op 8938) 


s_mul_i32 s1, s38, s7 // (op 8940) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 8940) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 8940) least significant
s_mul_hi_u32 s18, s38, s6 // (op 8940) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8940) most significant: sum
s_add_u32 s1, s1, s18 // (op 8940) most significant: sum


s_mov_b32 s68, s40 // (op 8940) reg expression
s_mov_b32 s69, s41 // (op 8940) reg expression
s_mov_b32 s70, s0 // (op 8940) reg expression
s_mov_b32 s71, 131072 // (op 8940) call()

v_and_b32 v18, -4, v1 // (op 9021) 
v_sub_u32 v20, v1, v18 // (op 9021) 

v_lshlrev_b32 v18, 5, v20 // (op 9021) 

v_lshrrev_b32 v20, 2, v1 // (op 9021) 
v_lshl_add_u32 v22, v20, 9, v18 // (op 9021) 


v_lshlrev_b32 v18, 2, v22 // (op 9021) 

v_lshrrev_b32 v20, 3, v18 // (op 9021) 

v_and_b32 v18, -8, v1 // (op 9081) 
v_sub_u32 v22, v1, v18 // (op 9081) 

v_lshlrev_b32 v18, 4, v22 // (op 9081) 

v_lshrrev_b32 v22, 3, v1 // (op 9081) 
v_lshl_add_u32 v23, v22, 8, v18 // (op 9081) 


v_lshlrev_b32 v18, 2, v1 // (op 9139) 
v_and_b32 v22, -4, v1 // (op 9197) 
v_sub_u32 v24, v1, v22 // (op 9197) 

v_and_b32 v22, 4294967292, v1 // (op 9197) 
v_add_u32 v25, v22, v24 // (op 9197) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 2883) 
s_and_b32 s47, s0, 4294967294 // (op 2883) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 2882) 
s_mov_b32 s18, s0 // (op 2882) convert

s_cmp_lt_u32 s47, s18 // (op 2882) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_2882 // (op 2882) Condition: Top (jump to RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_2882 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_2882:

s_add_u32 s47, s47, 1 // (op 2884) 
s_lshl_b64 s[0:1], s[26:27], 2 // (op 180) 
v_lshrrev_b64 v[26:27], 3, s[0:1] // (op 180) 


v_mov_b32 v14, v15 // (op 180) 
v_mov_b32 v16, v14 // (op 180) 
buffer_load_dwordx4 v[28:31], v16, s[48:51], 0 offen // (op 180) Load value
v_add_u32 v16, v16, v26 // (op 180) 
buffer_load_dwordx4 v[32:35], v16, s[48:51], 0 offen // (op 180) Load value
v_add_u32 v16, v16, v26 // (op 180) 
buffer_load_dwordx4 v[36:39], v16, s[48:51], 0 offen // (op 180) Load value
v_add_u32 v16, v16, v26 // (op 180) 
buffer_load_dwordx4 v[40:43], v16, s[48:51], 0 offen // (op 180) Load value



v_add_u32 v15, 64, v15 // (op 8718) 

s_barrier  // (op 9475) 
v_mov_b64 v[26:27], s[30:31] // (op 194) convert




v_mov_b32 v14, v17 // (op 194) 
v_mov_b32 v16, v14 // (op 194) 
buffer_load_dwordx4 v[44:47], v16, s[60:63], 0 offen // (op 194) Load value
v_add_u32 v16, v16, v26 // (op 194) 
buffer_load_dwordx4 v[48:51], v16, s[60:63], 0 offen // (op 194) Load value



v_add_u32 v17, 128, v17 // (op 8802) 
s_waitcnt vmcnt(5)
ds_write_b128 v20, v[28:31] offset:8448 // (op 181) Store local data 


v_mov_b32 v14, v19 // (op 195) 

s_waitcnt vmcnt(4)
ds_write_b128 v20, v[32:35] offset:8512 // (op 181) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v20, v[36:39] offset:8576 // (op 181) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[40:43] offset:8640 // (op 181) Store local data 
buffer_load_dword v16, v14, s[64:67], 0 offen // (op 195) Load value


v_add_u32 v19, 4, v19 // (op 8866) 


s_barrier  // (op 9488) 
v_mov_b32 v14, v21 // (op 196) 




buffer_load_ubyte v22, v14, s[68:71], 0 offen offset: 0 // (op 196) Load value

v_add_u32 v21, 4, v21 // (op 8930) 
s_waitcnt vmcnt(3)
ds_write_b128 v23, v[44:47] // (op 191) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 9478) 
s_waitcnt vmcnt(2)
ds_write_b128 v23, v[48:51] offset:128 // (op 191) Store local data 




ds_read_b128 v[26:29], v3 offset:8448 // (op 183) Load local data 
s_barrier  // (op 9502) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 9492) 






ds_read_b128 v[30:33], v4 // (op 185) Load local data 
ds_read_b128 v[34:37], v4 offset:64 // (op 185) Load local data 
s_waitcnt vmcnt(1)
ds_write_b32 v18, v16 offset:24832 // (op 192) Store local data 

s_barrier  // (op 9516) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 9506) 






ds_read_u8 v14, v0 offset:24832 // (op 187) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v25, v22 offset:8192 // (op 193) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 9520) 



ds_read_u8 v16, v5 offset:8192 // (op 189) Load local data 



ds_read_b128 v[38:41], v4 offset:4096 // (op 354) Load local data 
ds_read_b128 v[42:45], v4 offset:4160 // (op 354) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[30:37], a[0:3], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 190) 



ds_read_u8 v22, v5 offset:8320 // (op 358) Load local data 



ds_read_b128 v[46:49], v3 offset:10496 // (op 572) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:29], v[38:45], a[4:7], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 359) 





ds_read_u8 v14, v0 offset:24960 // (op 575) Load local data 



ds_read_b128 v[26:29], v3 offset:12544 // (op 958) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[46:49], v[30:37], a[8:11], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 577) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[46:49], v[38:45], a[12:15], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 608) 
ds_read_u8 v24, v0 offset:25088 // (op 961) Load local data 





ds_read_b128 v[46:49], v3 offset:14592 // (op 1344) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:29], v[30:37], a[16:19], v24, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 963) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:29], v[38:45], a[20:23], v24, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 994) 
ds_read_u8 v14, v0 offset:25216 // (op 1347) Load local data 





ds_read_b128 v[26:29], v3 offset:16640 // (op 1730) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[46:49], v[30:37], a[24:27], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1349) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[46:49], v[38:45], a[28:31], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1380) 
ds_read_u8 v24, v0 offset:25344 // (op 1733) Load local data 





ds_read_b128 v[46:49], v3 offset:18688 // (op 2116) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[26:29], v[30:37], a[32:35], v24, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1735) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[26:29], v[38:45], a[36:39], v24, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 1766) 
ds_read_u8 v14, v0 offset:25472 // (op 2119) Load local data 





ds_read_b128 v[26:29], v3 offset:20736 // (op 2502) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[46:49], v[30:37], a[40:43], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2121) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[46:49], v[38:45], a[44:47], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2152) 
ds_read_u8 v24, v0 offset:25600 // (op 2505) Load local data 





ds_read_b128 v[46:49], v3 offset:22784 // (op 2888) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[26:29], v[30:37], a[48:51], v24, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2507) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[26:29], v[38:45], a[52:55], v24, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2538) 
ds_read_u8 v14, v0 offset:25728 // (op 2891) Load local data 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[46:49], v[30:37], a[56:59], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2893) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[46:49], v[38:45], a[60:63], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 2924) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 2882) 
s_mov_b32 s18, s0 // (op 2882) convert

s_cmp_lt_u32 s47, s18 // (op 2882) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_2882 // (op 2882) Condition: Bottom (jump to RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_2882 if true)
RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_2882:






















s_barrier  // (op 9650) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_135
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



v_accvgpr_read v6, a4 // (op 3577) 
v_accvgpr_read v7, a5 // (op 3577) 
v_accvgpr_read v8, a6 // (op 3577) 
v_accvgpr_read v9, a7 // (op 3577) 
v_mul_f32 v14, v11, v6 // (op 3577) 
v_mul_f32 v15, v11, v7 // (op 3577) 
v_mul_f32 v16, v11, v8 // (op 3577) 
v_mul_f32 v17, v11, v9 // (op 3577) 

v_cvt_f16_f32 v6, v14 // (op 3579) 
v_cvt_f16_f32 v7, v15 // (op 3579) 
v_cvt_f16_f32 v8, v16 // (op 3579) 
v_cvt_f16_f32 v9, v17 // (op 3579) 


v_pack_b32_f16 v4, v6, v7 // (op 3580) 
v_pack_b32_f16 v5, v8, v9 // (op 3580) 

s_lshl_b32 s0, s4, 6 // (op 3580) 
v_add_u32 v0, v13, s0 // (op 3580) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 3580) Store value



v_accvgpr_read v6, a8 // (op 3603) 
v_accvgpr_read v7, a9 // (op 3603) 
v_accvgpr_read v8, a10 // (op 3603) 
v_accvgpr_read v9, a11 // (op 3603) 
v_mul_f32 v14, v11, v6 // (op 3603) 
v_mul_f32 v15, v11, v7 // (op 3603) 
v_mul_f32 v16, v11, v8 // (op 3603) 
v_mul_f32 v17, v11, v9 // (op 3603) 

v_cvt_f16_f32 v6, v14 // (op 3605) 
v_cvt_f16_f32 v7, v15 // (op 3605) 
v_cvt_f16_f32 v8, v16 // (op 3605) 
v_cvt_f16_f32 v9, v17 // (op 3605) 


v_pack_b32_f16 v4, v6, v7 // (op 3607) 
v_pack_b32_f16 v5, v8, v9 // (op 3607) 

v_add_u32 v0, 64, v13 // (op 3607) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 3607) Store value



v_accvgpr_read v6, a12 // (op 3613) 
v_accvgpr_read v7, a13 // (op 3613) 
v_accvgpr_read v8, a14 // (op 3613) 
v_accvgpr_read v9, a15 // (op 3613) 
v_mul_f32 v14, v11, v6 // (op 3613) 
v_mul_f32 v15, v11, v7 // (op 3613) 
v_mul_f32 v16, v11, v8 // (op 3613) 
v_mul_f32 v17, v11, v9 // (op 3613) 

v_cvt_f16_f32 v6, v14 // (op 3615) 
v_cvt_f16_f32 v7, v15 // (op 3615) 
v_cvt_f16_f32 v8, v16 // (op 3615) 
v_cvt_f16_f32 v9, v17 // (op 3615) 


v_pack_b32_f16 v4, v6, v7 // (op 3609) 
v_pack_b32_f16 v5, v8, v9 // (op 3609) 

s_lshl_b32 s0, s4, 6 // (op 3609) 
s_add_u32 s0, s0, 64 // (op 3609) 
v_add_u32 v0, v13, s0 // (op 3609) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 3609) Store value



v_accvgpr_read v6, a16 // (op 3643) 
v_accvgpr_read v7, a17 // (op 3643) 
v_accvgpr_read v8, a18 // (op 3643) 
v_accvgpr_read v9, a19 // (op 3643) 
v_mul_f32 v14, v11, v6 // (op 3643) 
v_mul_f32 v15, v11, v7 // (op 3643) 
v_mul_f32 v16, v11, v8 // (op 3643) 
v_mul_f32 v17, v11, v9 // (op 3643) 

v_cvt_f16_f32 v6, v14 // (op 3645) 
v_cvt_f16_f32 v7, v15 // (op 3645) 
v_cvt_f16_f32 v8, v16 // (op 3645) 
v_cvt_f16_f32 v9, v17 // (op 3645) 


v_pack_b32_f16 v4, v6, v7 // (op 3647) 
v_pack_b32_f16 v5, v8, v9 // (op 3647) 

v_add_u32 v0, 128, v13 // (op 3647) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 3647) Store value



v_accvgpr_read v6, a20 // (op 3653) 
v_accvgpr_read v7, a21 // (op 3653) 
v_accvgpr_read v8, a22 // (op 3653) 
v_accvgpr_read v9, a23 // (op 3653) 
v_mul_f32 v14, v11, v6 // (op 3653) 
v_mul_f32 v15, v11, v7 // (op 3653) 
v_mul_f32 v16, v11, v8 // (op 3653) 
v_mul_f32 v17, v11, v9 // (op 3653) 

v_cvt_f16_f32 v6, v14 // (op 3655) 
v_cvt_f16_f32 v7, v15 // (op 3655) 
v_cvt_f16_f32 v8, v16 // (op 3655) 
v_cvt_f16_f32 v9, v17 // (op 3655) 


v_pack_b32_f16 v4, v6, v7 // (op 3649) 
v_pack_b32_f16 v5, v8, v9 // (op 3649) 

s_lshl_b32 s0, s4, 6 // (op 3649) 
s_add_u32 s0, s0, 128 // (op 3649) 
v_add_u32 v0, v13, s0 // (op 3649) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 3649) Store value



v_accvgpr_read v6, a24 // (op 3683) 
v_accvgpr_read v7, a25 // (op 3683) 
v_accvgpr_read v8, a26 // (op 3683) 
v_accvgpr_read v9, a27 // (op 3683) 
v_mul_f32 v14, v11, v6 // (op 3683) 
v_mul_f32 v15, v11, v7 // (op 3683) 
v_mul_f32 v16, v11, v8 // (op 3683) 
v_mul_f32 v17, v11, v9 // (op 3683) 

v_cvt_f16_f32 v6, v14 // (op 3685) 
v_cvt_f16_f32 v7, v15 // (op 3685) 
v_cvt_f16_f32 v8, v16 // (op 3685) 
v_cvt_f16_f32 v9, v17 // (op 3685) 


v_pack_b32_f16 v4, v6, v7 // (op 3687) 
v_pack_b32_f16 v5, v8, v9 // (op 3687) 

v_add_u32 v0, 192, v13 // (op 3687) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 3687) Store value



v_accvgpr_read v6, a28 // (op 3693) 
v_accvgpr_read v7, a29 // (op 3693) 
v_accvgpr_read v8, a30 // (op 3693) 
v_accvgpr_read v9, a31 // (op 3693) 
v_mul_f32 v14, v11, v6 // (op 3693) 
v_mul_f32 v15, v11, v7 // (op 3693) 
v_mul_f32 v16, v11, v8 // (op 3693) 
v_mul_f32 v17, v11, v9 // (op 3693) 

v_cvt_f16_f32 v6, v14 // (op 3695) 
v_cvt_f16_f32 v7, v15 // (op 3695) 
v_cvt_f16_f32 v8, v16 // (op 3695) 
v_cvt_f16_f32 v9, v17 // (op 3695) 


v_pack_b32_f16 v4, v6, v7 // (op 3689) 
v_pack_b32_f16 v5, v8, v9 // (op 3689) 

s_lshl_b32 s0, s4, 6 // (op 3689) 
s_add_u32 s0, s0, 192 // (op 3689) 
v_add_u32 v0, v13, s0 // (op 3689) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 3689) Store value



v_accvgpr_read v6, a32 // (op 3723) 
v_accvgpr_read v7, a33 // (op 3723) 
v_accvgpr_read v8, a34 // (op 3723) 
v_accvgpr_read v9, a35 // (op 3723) 
v_mul_f32 v14, v11, v6 // (op 3723) 
v_mul_f32 v15, v11, v7 // (op 3723) 
v_mul_f32 v16, v11, v8 // (op 3723) 
v_mul_f32 v17, v11, v9 // (op 3723) 

v_cvt_f16_f32 v6, v14 // (op 3725) 
v_cvt_f16_f32 v7, v15 // (op 3725) 
v_cvt_f16_f32 v8, v16 // (op 3725) 
v_cvt_f16_f32 v9, v17 // (op 3725) 


v_pack_b32_f16 v4, v6, v7 // (op 3727) 
v_pack_b32_f16 v5, v8, v9 // (op 3727) 

v_add_u32 v0, 256, v13 // (op 3727) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 3727) Store value



v_accvgpr_read v6, a36 // (op 3733) 
v_accvgpr_read v7, a37 // (op 3733) 
v_accvgpr_read v8, a38 // (op 3733) 
v_accvgpr_read v9, a39 // (op 3733) 
v_mul_f32 v14, v11, v6 // (op 3733) 
v_mul_f32 v15, v11, v7 // (op 3733) 
v_mul_f32 v16, v11, v8 // (op 3733) 
v_mul_f32 v17, v11, v9 // (op 3733) 

v_cvt_f16_f32 v6, v14 // (op 3735) 
v_cvt_f16_f32 v7, v15 // (op 3735) 
v_cvt_f16_f32 v8, v16 // (op 3735) 
v_cvt_f16_f32 v9, v17 // (op 3735) 


v_pack_b32_f16 v4, v6, v7 // (op 3729) 
v_pack_b32_f16 v5, v8, v9 // (op 3729) 

s_lshl_b32 s0, s4, 6 // (op 3729) 
s_add_u32 s0, s0, 256 // (op 3729) 
v_add_u32 v0, v13, s0 // (op 3729) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 3729) Store value



v_accvgpr_read v6, a40 // (op 3763) 
v_accvgpr_read v7, a41 // (op 3763) 
v_accvgpr_read v8, a42 // (op 3763) 
v_accvgpr_read v9, a43 // (op 3763) 
v_mul_f32 v14, v11, v6 // (op 3763) 
v_mul_f32 v15, v11, v7 // (op 3763) 
v_mul_f32 v16, v11, v8 // (op 3763) 
v_mul_f32 v17, v11, v9 // (op 3763) 

v_cvt_f16_f32 v6, v14 // (op 3765) 
v_cvt_f16_f32 v7, v15 // (op 3765) 
v_cvt_f16_f32 v8, v16 // (op 3765) 
v_cvt_f16_f32 v9, v17 // (op 3765) 


v_pack_b32_f16 v4, v6, v7 // (op 3767) 
v_pack_b32_f16 v5, v8, v9 // (op 3767) 

v_add_u32 v0, 320, v13 // (op 3767) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 3767) Store value



v_accvgpr_read v6, a44 // (op 3773) 
v_accvgpr_read v7, a45 // (op 3773) 
v_accvgpr_read v8, a46 // (op 3773) 
v_accvgpr_read v9, a47 // (op 3773) 
v_mul_f32 v14, v11, v6 // (op 3773) 
v_mul_f32 v15, v11, v7 // (op 3773) 
v_mul_f32 v16, v11, v8 // (op 3773) 
v_mul_f32 v17, v11, v9 // (op 3773) 

v_cvt_f16_f32 v6, v14 // (op 3775) 
v_cvt_f16_f32 v7, v15 // (op 3775) 
v_cvt_f16_f32 v8, v16 // (op 3775) 
v_cvt_f16_f32 v9, v17 // (op 3775) 


v_pack_b32_f16 v4, v6, v7 // (op 3769) 
v_pack_b32_f16 v5, v8, v9 // (op 3769) 

s_lshl_b32 s0, s4, 6 // (op 3769) 
s_add_u32 s0, s0, 320 // (op 3769) 
v_add_u32 v0, v13, s0 // (op 3769) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 3769) Store value



v_accvgpr_read v6, a48 // (op 3803) 
v_accvgpr_read v7, a49 // (op 3803) 
v_accvgpr_read v8, a50 // (op 3803) 
v_accvgpr_read v9, a51 // (op 3803) 
v_mul_f32 v14, v11, v6 // (op 3803) 
v_mul_f32 v15, v11, v7 // (op 3803) 
v_mul_f32 v16, v11, v8 // (op 3803) 
v_mul_f32 v17, v11, v9 // (op 3803) 

v_cvt_f16_f32 v6, v14 // (op 3805) 
v_cvt_f16_f32 v7, v15 // (op 3805) 
v_cvt_f16_f32 v8, v16 // (op 3805) 
v_cvt_f16_f32 v9, v17 // (op 3805) 


v_pack_b32_f16 v4, v6, v7 // (op 3807) 
v_pack_b32_f16 v5, v8, v9 // (op 3807) 

v_add_u32 v0, 384, v13 // (op 3807) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 3807) Store value



v_accvgpr_read v6, a52 // (op 3813) 
v_accvgpr_read v7, a53 // (op 3813) 
v_accvgpr_read v8, a54 // (op 3813) 
v_accvgpr_read v9, a55 // (op 3813) 
v_mul_f32 v14, v11, v6 // (op 3813) 
v_mul_f32 v15, v11, v7 // (op 3813) 
v_mul_f32 v16, v11, v8 // (op 3813) 
v_mul_f32 v17, v11, v9 // (op 3813) 

v_cvt_f16_f32 v6, v14 // (op 3815) 
v_cvt_f16_f32 v7, v15 // (op 3815) 
v_cvt_f16_f32 v8, v16 // (op 3815) 
v_cvt_f16_f32 v9, v17 // (op 3815) 


v_pack_b32_f16 v4, v6, v7 // (op 3809) 
v_pack_b32_f16 v5, v8, v9 // (op 3809) 

s_lshl_b32 s0, s4, 6 // (op 3809) 
s_add_u32 s0, s0, 384 // (op 3809) 
v_add_u32 v0, v13, s0 // (op 3809) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 3809) Store value



v_accvgpr_read v6, a56 // (op 3843) 
v_accvgpr_read v7, a57 // (op 3843) 
v_accvgpr_read v8, a58 // (op 3843) 
v_accvgpr_read v9, a59 // (op 3843) 
v_mul_f32 v14, v11, v6 // (op 3843) 
v_mul_f32 v15, v11, v7 // (op 3843) 
v_mul_f32 v16, v11, v8 // (op 3843) 
v_mul_f32 v17, v11, v9 // (op 3843) 

v_cvt_f16_f32 v6, v14 // (op 3845) 
v_cvt_f16_f32 v7, v15 // (op 3845) 
v_cvt_f16_f32 v8, v16 // (op 3845) 
v_cvt_f16_f32 v9, v17 // (op 3845) 


v_pack_b32_f16 v4, v6, v7 // (op 3847) 
v_pack_b32_f16 v5, v8, v9 // (op 3847) 

v_add_u32 v0, 448, v13 // (op 3847) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 3847) Store value



v_accvgpr_read v6, a60 // (op 3853) 
v_accvgpr_read v7, a61 // (op 3853) 
v_accvgpr_read v8, a62 // (op 3853) 
v_accvgpr_read v9, a63 // (op 3853) 
v_mul_f32 v14, v11, v6 // (op 3853) 
v_mul_f32 v15, v11, v7 // (op 3853) 
v_mul_f32 v16, v11, v8 // (op 3853) 
v_mul_f32 v17, v11, v9 // (op 3853) 

v_cvt_f16_f32 v6, v14 // (op 3855) 
v_cvt_f16_f32 v7, v15 // (op 3855) 
v_cvt_f16_f32 v8, v16 // (op 3855) 
v_cvt_f16_f32 v9, v17 // (op 3855) 


v_pack_b32_f16 v4, v6, v7 // (op 3849) 
v_pack_b32_f16 v5, v8, v9 // (op 3849) 

s_lshl_b32 s0, s4, 6 // (op 3849) 
s_add_u32 s0, s0, 448 // (op 3849) 
v_add_u32 v0, v13, s0 // (op 3849) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 3849) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_135
RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_135:


v_add_u32 v0, 448, v12 // (op 4257) 

buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 4257) Load value

v_accvgpr_read v4, a56 // (op 4248) 
v_accvgpr_read v5, a57 // (op 4248) 
v_accvgpr_read v6, a58 // (op 4248) 
v_accvgpr_read v7, a59 // (op 4248) 
v_mul_f32 v14, v11, v4 // (op 4248) 
v_mul_f32 v15, v11, v5 // (op 4248) 
v_mul_f32 v16, v11, v6 // (op 4248) 
v_mul_f32 v17, v11, v7 // (op 4248) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4256) 
v_lshrrev_b32 v7, 16, v8 // (op 4256) 
v_cvt_f32_f16 v7, v7 // (op 4256) 
v_mul_f32 v18, v10, v6 // (op 4256) 
v_mul_f32 v19, v10, v7 // (op 4256) 
v_cvt_f32_f16 v6, v9 // (op 4256) 
v_lshrrev_b32 v7, 16, v9 // (op 4256) 
v_cvt_f32_f16 v7, v7 // (op 4256) 
v_mul_f32 v20, v10, v6 // (op 4256) 
v_mul_f32 v21, v10, v7 // (op 4256) 


v_add_f32 v6, v18, v14 // (op 4249) 
v_add_f32 v7, v19, v15 // (op 4249) 
v_add_f32 v8, v20, v16 // (op 4249) 
v_add_f32 v9, v21, v17 // (op 4249) 


v_cvt_f16_f32 v14, v6 // (op 4250) 
v_cvt_f16_f32 v15, v7 // (op 4250) 
v_cvt_f16_f32 v16, v8 // (op 4250) 
v_cvt_f16_f32 v17, v9 // (op 4250) 


v_pack_b32_f16 v8, v14, v15 // (op 4252) 
v_pack_b32_f16 v9, v16, v17 // (op 4252) 

v_add_u32 v0, 448, v13 // (op 4252) 

buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4252) Store value





s_lshl_b32 s0, s2, 6 // (op 4262) 
s_add_u32 s0, s0, 448 // (op 4262) 
v_add_u32 v0, v12, s0 // (op 4262) 


buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 4262) Load value

v_accvgpr_read v4, a60 // (op 4264) 
v_accvgpr_read v5, a61 // (op 4264) 
v_accvgpr_read v6, a62 // (op 4264) 
v_accvgpr_read v7, a63 // (op 4264) 
v_mul_f32 v14, v11, v4 // (op 4264) 
v_mul_f32 v15, v11, v5 // (op 4264) 
v_mul_f32 v16, v11, v6 // (op 4264) 
v_mul_f32 v17, v11, v7 // (op 4264) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4259) 
v_lshrrev_b32 v7, 16, v8 // (op 4259) 
v_cvt_f32_f16 v7, v7 // (op 4259) 
v_mul_f32 v18, v10, v6 // (op 4259) 
v_mul_f32 v19, v10, v7 // (op 4259) 
v_cvt_f32_f16 v6, v9 // (op 4259) 
v_lshrrev_b32 v7, 16, v9 // (op 4259) 
v_cvt_f32_f16 v7, v7 // (op 4259) 
v_mul_f32 v20, v10, v6 // (op 4259) 
v_mul_f32 v21, v10, v7 // (op 4259) 


v_add_f32 v6, v18, v14 // (op 4260) 
v_add_f32 v7, v19, v15 // (op 4260) 
v_add_f32 v8, v20, v16 // (op 4260) 
v_add_f32 v9, v21, v17 // (op 4260) 


v_cvt_f16_f32 v14, v6 // (op 4261) 
v_cvt_f16_f32 v15, v7 // (op 4261) 
v_cvt_f16_f32 v16, v8 // (op 4261) 
v_cvt_f16_f32 v17, v9 // (op 4261) 


v_pack_b32_f16 v8, v14, v15 // (op 4254) 
v_pack_b32_f16 v9, v16, v17 // (op 4254) 

s_lshl_b32 s0, s4, 6 // (op 4254) 
s_add_u32 s0, s0, 448 // (op 4254) 
v_add_u32 v0, v13, s0 // (op 4254) 


buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4254) Store value





v_add_u32 v0, 384, v12 // (op 4212) 

buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 4212) Load value

v_accvgpr_read v4, a48 // (op 4203) 
v_accvgpr_read v5, a49 // (op 4203) 
v_accvgpr_read v6, a50 // (op 4203) 
v_accvgpr_read v7, a51 // (op 4203) 
v_mul_f32 v14, v11, v4 // (op 4203) 
v_mul_f32 v15, v11, v5 // (op 4203) 
v_mul_f32 v16, v11, v6 // (op 4203) 
v_mul_f32 v17, v11, v7 // (op 4203) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4211) 
v_lshrrev_b32 v7, 16, v8 // (op 4211) 
v_cvt_f32_f16 v7, v7 // (op 4211) 
v_mul_f32 v18, v10, v6 // (op 4211) 
v_mul_f32 v19, v10, v7 // (op 4211) 
v_cvt_f32_f16 v6, v9 // (op 4211) 
v_lshrrev_b32 v7, 16, v9 // (op 4211) 
v_cvt_f32_f16 v7, v7 // (op 4211) 
v_mul_f32 v20, v10, v6 // (op 4211) 
v_mul_f32 v21, v10, v7 // (op 4211) 


v_add_f32 v6, v18, v14 // (op 4204) 
v_add_f32 v7, v19, v15 // (op 4204) 
v_add_f32 v8, v20, v16 // (op 4204) 
v_add_f32 v9, v21, v17 // (op 4204) 


v_cvt_f16_f32 v14, v6 // (op 4205) 
v_cvt_f16_f32 v15, v7 // (op 4205) 
v_cvt_f16_f32 v16, v8 // (op 4205) 
v_cvt_f16_f32 v17, v9 // (op 4205) 


v_pack_b32_f16 v8, v14, v15 // (op 4207) 
v_pack_b32_f16 v9, v16, v17 // (op 4207) 

v_add_u32 v0, 384, v13 // (op 4207) 

buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4207) Store value





s_lshl_b32 s0, s2, 6 // (op 4217) 
s_add_u32 s0, s0, 384 // (op 4217) 
v_add_u32 v0, v12, s0 // (op 4217) 


buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 4217) Load value

v_accvgpr_read v4, a52 // (op 4219) 
v_accvgpr_read v5, a53 // (op 4219) 
v_accvgpr_read v6, a54 // (op 4219) 
v_accvgpr_read v7, a55 // (op 4219) 
v_mul_f32 v14, v11, v4 // (op 4219) 
v_mul_f32 v15, v11, v5 // (op 4219) 
v_mul_f32 v16, v11, v6 // (op 4219) 
v_mul_f32 v17, v11, v7 // (op 4219) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4214) 
v_lshrrev_b32 v7, 16, v8 // (op 4214) 
v_cvt_f32_f16 v7, v7 // (op 4214) 
v_mul_f32 v18, v10, v6 // (op 4214) 
v_mul_f32 v19, v10, v7 // (op 4214) 
v_cvt_f32_f16 v6, v9 // (op 4214) 
v_lshrrev_b32 v7, 16, v9 // (op 4214) 
v_cvt_f32_f16 v7, v7 // (op 4214) 
v_mul_f32 v20, v10, v6 // (op 4214) 
v_mul_f32 v21, v10, v7 // (op 4214) 


v_add_f32 v6, v18, v14 // (op 4215) 
v_add_f32 v7, v19, v15 // (op 4215) 
v_add_f32 v8, v20, v16 // (op 4215) 
v_add_f32 v9, v21, v17 // (op 4215) 


v_cvt_f16_f32 v14, v6 // (op 4216) 
v_cvt_f16_f32 v15, v7 // (op 4216) 
v_cvt_f16_f32 v16, v8 // (op 4216) 
v_cvt_f16_f32 v17, v9 // (op 4216) 


v_pack_b32_f16 v8, v14, v15 // (op 4209) 
v_pack_b32_f16 v9, v16, v17 // (op 4209) 

s_lshl_b32 s0, s4, 6 // (op 4209) 
s_add_u32 s0, s0, 384 // (op 4209) 
v_add_u32 v0, v13, s0 // (op 4209) 


buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4209) Store value





v_add_u32 v0, 320, v12 // (op 4167) 

buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 4167) Load value

v_accvgpr_read v4, a40 // (op 4158) 
v_accvgpr_read v5, a41 // (op 4158) 
v_accvgpr_read v6, a42 // (op 4158) 
v_accvgpr_read v7, a43 // (op 4158) 
v_mul_f32 v14, v11, v4 // (op 4158) 
v_mul_f32 v15, v11, v5 // (op 4158) 
v_mul_f32 v16, v11, v6 // (op 4158) 
v_mul_f32 v17, v11, v7 // (op 4158) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4166) 
v_lshrrev_b32 v7, 16, v8 // (op 4166) 
v_cvt_f32_f16 v7, v7 // (op 4166) 
v_mul_f32 v18, v10, v6 // (op 4166) 
v_mul_f32 v19, v10, v7 // (op 4166) 
v_cvt_f32_f16 v6, v9 // (op 4166) 
v_lshrrev_b32 v7, 16, v9 // (op 4166) 
v_cvt_f32_f16 v7, v7 // (op 4166) 
v_mul_f32 v20, v10, v6 // (op 4166) 
v_mul_f32 v21, v10, v7 // (op 4166) 


v_add_f32 v6, v18, v14 // (op 4159) 
v_add_f32 v7, v19, v15 // (op 4159) 
v_add_f32 v8, v20, v16 // (op 4159) 
v_add_f32 v9, v21, v17 // (op 4159) 


v_cvt_f16_f32 v14, v6 // (op 4160) 
v_cvt_f16_f32 v15, v7 // (op 4160) 
v_cvt_f16_f32 v16, v8 // (op 4160) 
v_cvt_f16_f32 v17, v9 // (op 4160) 


v_pack_b32_f16 v8, v14, v15 // (op 4162) 
v_pack_b32_f16 v9, v16, v17 // (op 4162) 

v_add_u32 v0, 320, v13 // (op 4162) 

buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4162) Store value





s_lshl_b32 s0, s2, 6 // (op 4172) 
s_add_u32 s0, s0, 320 // (op 4172) 
v_add_u32 v0, v12, s0 // (op 4172) 


buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 4172) Load value

v_accvgpr_read v4, a44 // (op 4174) 
v_accvgpr_read v5, a45 // (op 4174) 
v_accvgpr_read v6, a46 // (op 4174) 
v_accvgpr_read v7, a47 // (op 4174) 
v_mul_f32 v14, v11, v4 // (op 4174) 
v_mul_f32 v15, v11, v5 // (op 4174) 
v_mul_f32 v16, v11, v6 // (op 4174) 
v_mul_f32 v17, v11, v7 // (op 4174) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4169) 
v_lshrrev_b32 v7, 16, v8 // (op 4169) 
v_cvt_f32_f16 v7, v7 // (op 4169) 
v_mul_f32 v18, v10, v6 // (op 4169) 
v_mul_f32 v19, v10, v7 // (op 4169) 
v_cvt_f32_f16 v6, v9 // (op 4169) 
v_lshrrev_b32 v7, 16, v9 // (op 4169) 
v_cvt_f32_f16 v7, v7 // (op 4169) 
v_mul_f32 v20, v10, v6 // (op 4169) 
v_mul_f32 v21, v10, v7 // (op 4169) 


v_add_f32 v6, v18, v14 // (op 4170) 
v_add_f32 v7, v19, v15 // (op 4170) 
v_add_f32 v8, v20, v16 // (op 4170) 
v_add_f32 v9, v21, v17 // (op 4170) 


v_cvt_f16_f32 v14, v6 // (op 4171) 
v_cvt_f16_f32 v15, v7 // (op 4171) 
v_cvt_f16_f32 v16, v8 // (op 4171) 
v_cvt_f16_f32 v17, v9 // (op 4171) 


v_pack_b32_f16 v8, v14, v15 // (op 4164) 
v_pack_b32_f16 v9, v16, v17 // (op 4164) 

s_lshl_b32 s0, s4, 6 // (op 4164) 
s_add_u32 s0, s0, 320 // (op 4164) 
v_add_u32 v0, v13, s0 // (op 4164) 


buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4164) Store value





v_add_u32 v0, 256, v12 // (op 4122) 

buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 4122) Load value

v_accvgpr_read v4, a32 // (op 4113) 
v_accvgpr_read v5, a33 // (op 4113) 
v_accvgpr_read v6, a34 // (op 4113) 
v_accvgpr_read v7, a35 // (op 4113) 
v_mul_f32 v14, v11, v4 // (op 4113) 
v_mul_f32 v15, v11, v5 // (op 4113) 
v_mul_f32 v16, v11, v6 // (op 4113) 
v_mul_f32 v17, v11, v7 // (op 4113) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4121) 
v_lshrrev_b32 v7, 16, v8 // (op 4121) 
v_cvt_f32_f16 v7, v7 // (op 4121) 
v_mul_f32 v18, v10, v6 // (op 4121) 
v_mul_f32 v19, v10, v7 // (op 4121) 
v_cvt_f32_f16 v6, v9 // (op 4121) 
v_lshrrev_b32 v7, 16, v9 // (op 4121) 
v_cvt_f32_f16 v7, v7 // (op 4121) 
v_mul_f32 v20, v10, v6 // (op 4121) 
v_mul_f32 v21, v10, v7 // (op 4121) 


v_add_f32 v6, v18, v14 // (op 4114) 
v_add_f32 v7, v19, v15 // (op 4114) 
v_add_f32 v8, v20, v16 // (op 4114) 
v_add_f32 v9, v21, v17 // (op 4114) 


v_cvt_f16_f32 v14, v6 // (op 4115) 
v_cvt_f16_f32 v15, v7 // (op 4115) 
v_cvt_f16_f32 v16, v8 // (op 4115) 
v_cvt_f16_f32 v17, v9 // (op 4115) 


v_pack_b32_f16 v8, v14, v15 // (op 4117) 
v_pack_b32_f16 v9, v16, v17 // (op 4117) 

v_add_u32 v0, 256, v13 // (op 4117) 

buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4117) Store value





s_lshl_b32 s0, s2, 6 // (op 4127) 
s_add_u32 s0, s0, 256 // (op 4127) 
v_add_u32 v0, v12, s0 // (op 4127) 


buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 4127) Load value

v_accvgpr_read v4, a36 // (op 4129) 
v_accvgpr_read v5, a37 // (op 4129) 
v_accvgpr_read v6, a38 // (op 4129) 
v_accvgpr_read v7, a39 // (op 4129) 
v_mul_f32 v14, v11, v4 // (op 4129) 
v_mul_f32 v15, v11, v5 // (op 4129) 
v_mul_f32 v16, v11, v6 // (op 4129) 
v_mul_f32 v17, v11, v7 // (op 4129) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4124) 
v_lshrrev_b32 v7, 16, v8 // (op 4124) 
v_cvt_f32_f16 v7, v7 // (op 4124) 
v_mul_f32 v18, v10, v6 // (op 4124) 
v_mul_f32 v19, v10, v7 // (op 4124) 
v_cvt_f32_f16 v6, v9 // (op 4124) 
v_lshrrev_b32 v7, 16, v9 // (op 4124) 
v_cvt_f32_f16 v7, v7 // (op 4124) 
v_mul_f32 v20, v10, v6 // (op 4124) 
v_mul_f32 v21, v10, v7 // (op 4124) 


v_add_f32 v6, v18, v14 // (op 4125) 
v_add_f32 v7, v19, v15 // (op 4125) 
v_add_f32 v8, v20, v16 // (op 4125) 
v_add_f32 v9, v21, v17 // (op 4125) 


v_cvt_f16_f32 v14, v6 // (op 4126) 
v_cvt_f16_f32 v15, v7 // (op 4126) 
v_cvt_f16_f32 v16, v8 // (op 4126) 
v_cvt_f16_f32 v17, v9 // (op 4126) 


v_pack_b32_f16 v8, v14, v15 // (op 4119) 
v_pack_b32_f16 v9, v16, v17 // (op 4119) 

s_lshl_b32 s0, s4, 6 // (op 4119) 
s_add_u32 s0, s0, 256 // (op 4119) 
v_add_u32 v0, v13, s0 // (op 4119) 


buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4119) Store value





v_add_u32 v0, 192, v12 // (op 4077) 

buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 4077) Load value

v_accvgpr_read v4, a24 // (op 4068) 
v_accvgpr_read v5, a25 // (op 4068) 
v_accvgpr_read v6, a26 // (op 4068) 
v_accvgpr_read v7, a27 // (op 4068) 
v_mul_f32 v14, v11, v4 // (op 4068) 
v_mul_f32 v15, v11, v5 // (op 4068) 
v_mul_f32 v16, v11, v6 // (op 4068) 
v_mul_f32 v17, v11, v7 // (op 4068) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4076) 
v_lshrrev_b32 v7, 16, v8 // (op 4076) 
v_cvt_f32_f16 v7, v7 // (op 4076) 
v_mul_f32 v18, v10, v6 // (op 4076) 
v_mul_f32 v19, v10, v7 // (op 4076) 
v_cvt_f32_f16 v6, v9 // (op 4076) 
v_lshrrev_b32 v7, 16, v9 // (op 4076) 
v_cvt_f32_f16 v7, v7 // (op 4076) 
v_mul_f32 v20, v10, v6 // (op 4076) 
v_mul_f32 v21, v10, v7 // (op 4076) 


v_add_f32 v6, v18, v14 // (op 4069) 
v_add_f32 v7, v19, v15 // (op 4069) 
v_add_f32 v8, v20, v16 // (op 4069) 
v_add_f32 v9, v21, v17 // (op 4069) 


v_cvt_f16_f32 v14, v6 // (op 4070) 
v_cvt_f16_f32 v15, v7 // (op 4070) 
v_cvt_f16_f32 v16, v8 // (op 4070) 
v_cvt_f16_f32 v17, v9 // (op 4070) 


v_pack_b32_f16 v8, v14, v15 // (op 4072) 
v_pack_b32_f16 v9, v16, v17 // (op 4072) 

v_add_u32 v0, 192, v13 // (op 4072) 

buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4072) Store value





s_lshl_b32 s0, s2, 6 // (op 4082) 
s_add_u32 s0, s0, 192 // (op 4082) 
v_add_u32 v0, v12, s0 // (op 4082) 


buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 4082) Load value

v_accvgpr_read v4, a28 // (op 4084) 
v_accvgpr_read v5, a29 // (op 4084) 
v_accvgpr_read v6, a30 // (op 4084) 
v_accvgpr_read v7, a31 // (op 4084) 
v_mul_f32 v14, v11, v4 // (op 4084) 
v_mul_f32 v15, v11, v5 // (op 4084) 
v_mul_f32 v16, v11, v6 // (op 4084) 
v_mul_f32 v17, v11, v7 // (op 4084) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4079) 
v_lshrrev_b32 v7, 16, v8 // (op 4079) 
v_cvt_f32_f16 v7, v7 // (op 4079) 
v_mul_f32 v18, v10, v6 // (op 4079) 
v_mul_f32 v19, v10, v7 // (op 4079) 
v_cvt_f32_f16 v6, v9 // (op 4079) 
v_lshrrev_b32 v7, 16, v9 // (op 4079) 
v_cvt_f32_f16 v7, v7 // (op 4079) 
v_mul_f32 v20, v10, v6 // (op 4079) 
v_mul_f32 v21, v10, v7 // (op 4079) 


v_add_f32 v6, v18, v14 // (op 4080) 
v_add_f32 v7, v19, v15 // (op 4080) 
v_add_f32 v8, v20, v16 // (op 4080) 
v_add_f32 v9, v21, v17 // (op 4080) 


v_cvt_f16_f32 v14, v6 // (op 4081) 
v_cvt_f16_f32 v15, v7 // (op 4081) 
v_cvt_f16_f32 v16, v8 // (op 4081) 
v_cvt_f16_f32 v17, v9 // (op 4081) 


v_pack_b32_f16 v8, v14, v15 // (op 4074) 
v_pack_b32_f16 v9, v16, v17 // (op 4074) 

s_lshl_b32 s0, s4, 6 // (op 4074) 
s_add_u32 s0, s0, 192 // (op 4074) 
v_add_u32 v0, v13, s0 // (op 4074) 


buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4074) Store value





v_add_u32 v0, 128, v12 // (op 4032) 

buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 4032) Load value

v_accvgpr_read v4, a16 // (op 4023) 
v_accvgpr_read v5, a17 // (op 4023) 
v_accvgpr_read v6, a18 // (op 4023) 
v_accvgpr_read v7, a19 // (op 4023) 
v_mul_f32 v14, v11, v4 // (op 4023) 
v_mul_f32 v15, v11, v5 // (op 4023) 
v_mul_f32 v16, v11, v6 // (op 4023) 
v_mul_f32 v17, v11, v7 // (op 4023) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4031) 
v_lshrrev_b32 v7, 16, v8 // (op 4031) 
v_cvt_f32_f16 v7, v7 // (op 4031) 
v_mul_f32 v18, v10, v6 // (op 4031) 
v_mul_f32 v19, v10, v7 // (op 4031) 
v_cvt_f32_f16 v6, v9 // (op 4031) 
v_lshrrev_b32 v7, 16, v9 // (op 4031) 
v_cvt_f32_f16 v7, v7 // (op 4031) 
v_mul_f32 v20, v10, v6 // (op 4031) 
v_mul_f32 v21, v10, v7 // (op 4031) 


v_add_f32 v6, v18, v14 // (op 4024) 
v_add_f32 v7, v19, v15 // (op 4024) 
v_add_f32 v8, v20, v16 // (op 4024) 
v_add_f32 v9, v21, v17 // (op 4024) 


v_cvt_f16_f32 v14, v6 // (op 4025) 
v_cvt_f16_f32 v15, v7 // (op 4025) 
v_cvt_f16_f32 v16, v8 // (op 4025) 
v_cvt_f16_f32 v17, v9 // (op 4025) 


v_pack_b32_f16 v8, v14, v15 // (op 4027) 
v_pack_b32_f16 v9, v16, v17 // (op 4027) 

v_add_u32 v0, 128, v13 // (op 4027) 

buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4027) Store value





s_lshl_b32 s0, s2, 6 // (op 4037) 
s_add_u32 s0, s0, 128 // (op 4037) 
v_add_u32 v0, v12, s0 // (op 4037) 


buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 4037) Load value

v_accvgpr_read v4, a20 // (op 4039) 
v_accvgpr_read v5, a21 // (op 4039) 
v_accvgpr_read v6, a22 // (op 4039) 
v_accvgpr_read v7, a23 // (op 4039) 
v_mul_f32 v14, v11, v4 // (op 4039) 
v_mul_f32 v15, v11, v5 // (op 4039) 
v_mul_f32 v16, v11, v6 // (op 4039) 
v_mul_f32 v17, v11, v7 // (op 4039) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 4034) 
v_lshrrev_b32 v7, 16, v8 // (op 4034) 
v_cvt_f32_f16 v7, v7 // (op 4034) 
v_mul_f32 v18, v10, v6 // (op 4034) 
v_mul_f32 v19, v10, v7 // (op 4034) 
v_cvt_f32_f16 v6, v9 // (op 4034) 
v_lshrrev_b32 v7, 16, v9 // (op 4034) 
v_cvt_f32_f16 v7, v7 // (op 4034) 
v_mul_f32 v20, v10, v6 // (op 4034) 
v_mul_f32 v21, v10, v7 // (op 4034) 


v_add_f32 v6, v18, v14 // (op 4035) 
v_add_f32 v7, v19, v15 // (op 4035) 
v_add_f32 v8, v20, v16 // (op 4035) 
v_add_f32 v9, v21, v17 // (op 4035) 


v_cvt_f16_f32 v14, v6 // (op 4036) 
v_cvt_f16_f32 v15, v7 // (op 4036) 
v_cvt_f16_f32 v16, v8 // (op 4036) 
v_cvt_f16_f32 v17, v9 // (op 4036) 


v_pack_b32_f16 v8, v14, v15 // (op 4029) 
v_pack_b32_f16 v9, v16, v17 // (op 4029) 

s_lshl_b32 s0, s4, 6 // (op 4029) 
s_add_u32 s0, s0, 128 // (op 4029) 
v_add_u32 v0, v13, s0 // (op 4029) 


buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 4029) Store value





v_add_u32 v0, 64, v12 // (op 3987) 

buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 3987) Load value

v_accvgpr_read v4, a8 // (op 3978) 
v_accvgpr_read v5, a9 // (op 3978) 
v_accvgpr_read v6, a10 // (op 3978) 
v_accvgpr_read v7, a11 // (op 3978) 
v_mul_f32 v14, v11, v4 // (op 3978) 
v_mul_f32 v15, v11, v5 // (op 3978) 
v_mul_f32 v16, v11, v6 // (op 3978) 
v_mul_f32 v17, v11, v7 // (op 3978) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 3986) 
v_lshrrev_b32 v7, 16, v8 // (op 3986) 
v_cvt_f32_f16 v7, v7 // (op 3986) 
v_mul_f32 v18, v10, v6 // (op 3986) 
v_mul_f32 v19, v10, v7 // (op 3986) 
v_cvt_f32_f16 v6, v9 // (op 3986) 
v_lshrrev_b32 v7, 16, v9 // (op 3986) 
v_cvt_f32_f16 v7, v7 // (op 3986) 
v_mul_f32 v20, v10, v6 // (op 3986) 
v_mul_f32 v21, v10, v7 // (op 3986) 


v_add_f32 v6, v18, v14 // (op 3979) 
v_add_f32 v7, v19, v15 // (op 3979) 
v_add_f32 v8, v20, v16 // (op 3979) 
v_add_f32 v9, v21, v17 // (op 3979) 


v_cvt_f16_f32 v14, v6 // (op 3980) 
v_cvt_f16_f32 v15, v7 // (op 3980) 
v_cvt_f16_f32 v16, v8 // (op 3980) 
v_cvt_f16_f32 v17, v9 // (op 3980) 


v_pack_b32_f16 v8, v14, v15 // (op 3982) 
v_pack_b32_f16 v9, v16, v17 // (op 3982) 

v_add_u32 v0, 64, v13 // (op 3982) 

buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 3982) Store value





s_lshl_b32 s0, s2, 6 // (op 3992) 
s_add_u32 s0, s0, 64 // (op 3992) 
v_add_u32 v0, v12, s0 // (op 3992) 


buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 3992) Load value

v_accvgpr_read v4, a12 // (op 3994) 
v_accvgpr_read v5, a13 // (op 3994) 
v_accvgpr_read v6, a14 // (op 3994) 
v_accvgpr_read v7, a15 // (op 3994) 
v_mul_f32 v14, v11, v4 // (op 3994) 
v_mul_f32 v15, v11, v5 // (op 3994) 
v_mul_f32 v16, v11, v6 // (op 3994) 
v_mul_f32 v17, v11, v7 // (op 3994) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 3989) 
v_lshrrev_b32 v7, 16, v8 // (op 3989) 
v_cvt_f32_f16 v7, v7 // (op 3989) 
v_mul_f32 v18, v10, v6 // (op 3989) 
v_mul_f32 v19, v10, v7 // (op 3989) 
v_cvt_f32_f16 v6, v9 // (op 3989) 
v_lshrrev_b32 v7, 16, v9 // (op 3989) 
v_cvt_f32_f16 v7, v7 // (op 3989) 
v_mul_f32 v20, v10, v6 // (op 3989) 
v_mul_f32 v21, v10, v7 // (op 3989) 


v_add_f32 v6, v18, v14 // (op 3990) 
v_add_f32 v7, v19, v15 // (op 3990) 
v_add_f32 v8, v20, v16 // (op 3990) 
v_add_f32 v9, v21, v17 // (op 3990) 


v_cvt_f16_f32 v14, v6 // (op 3991) 
v_cvt_f16_f32 v15, v7 // (op 3991) 
v_cvt_f16_f32 v16, v8 // (op 3991) 
v_cvt_f16_f32 v17, v9 // (op 3991) 


v_pack_b32_f16 v8, v14, v15 // (op 3984) 
v_pack_b32_f16 v9, v16, v17 // (op 3984) 

s_lshl_b32 s0, s4, 6 // (op 3984) 
s_add_u32 s0, s0, 64 // (op 3984) 
v_add_u32 v0, v13, s0 // (op 3984) 


buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 3984) Store value





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





s_lshl_b32 s0, s2, 6 // (op 3947) 
v_add_u32 v0, v12, s0 // (op 3947) 


buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 3947) Load value

v_accvgpr_read v4, a4 // (op 3943) 
v_accvgpr_read v5, a5 // (op 3943) 
v_accvgpr_read v6, a6 // (op 3943) 
v_accvgpr_read v7, a7 // (op 3943) 
v_mul_f32 v14, v11, v4 // (op 3943) 
v_mul_f32 v15, v11, v5 // (op 3943) 
v_mul_f32 v16, v11, v6 // (op 3943) 
v_mul_f32 v17, v11, v7 // (op 3943) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 3948) 
v_lshrrev_b32 v7, 16, v8 // (op 3948) 
v_cvt_f32_f16 v7, v7 // (op 3948) 
v_mul_f32 v18, v10, v6 // (op 3948) 
v_mul_f32 v19, v10, v7 // (op 3948) 
v_cvt_f32_f16 v6, v9 // (op 3948) 
v_lshrrev_b32 v7, 16, v9 // (op 3948) 
v_cvt_f32_f16 v7, v7 // (op 3948) 
v_mul_f32 v20, v10, v6 // (op 3948) 
v_mul_f32 v21, v10, v7 // (op 3948) 


v_add_f32 v6, v18, v14 // (op 3944) 
v_add_f32 v7, v19, v15 // (op 3944) 
v_add_f32 v8, v20, v16 // (op 3944) 
v_add_f32 v9, v21, v17 // (op 3944) 


v_cvt_f16_f32 v14, v6 // (op 3945) 
v_cvt_f16_f32 v15, v7 // (op 3945) 
v_cvt_f16_f32 v16, v8 // (op 3945) 
v_cvt_f16_f32 v17, v9 // (op 3945) 


v_pack_b32_f16 v8, v14, v15 // (op 3946) 
v_pack_b32_f16 v9, v16, v17 // (op 3946) 

s_lshl_b32 s0, s4, 6 // (op 3946) 
v_add_u32 v0, v13, s0 // (op 3946) 


buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 3946) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_135:
























s_endpgm  // End of RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC
.LRR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_end:
.size RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC, .LRR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC_end-RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC
  .amdhsa_next_free_vgpr 180
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 51712 // lds bytes
  .amdhsa_accum_offset 116
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
    DataType: F4
    DataTypeA: F4
    DataTypeB: B8
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
    macrotile: [256, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC
    .symbol: RR_GEMM_TN_FP4_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_WGM__LDSSYNC.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 51712
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 116
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

