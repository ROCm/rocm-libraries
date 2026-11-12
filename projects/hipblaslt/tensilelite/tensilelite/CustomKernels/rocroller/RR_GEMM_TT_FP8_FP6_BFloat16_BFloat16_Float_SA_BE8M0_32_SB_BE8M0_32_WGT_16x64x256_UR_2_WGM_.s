// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_,@function
RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__exec_begin:


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
s_load_dwordx2 s[48:49], s[0:1], 168 // Load scalar value
s_load_dword s50, s[0:1], 176 // Load scalar value

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


s_mul_hi_i32 s0, s16, s11 // (op 1469) 
s_mul_hi_i32 s1, s16, s11 // (op 1710) 
v_lshrrev_b32 v0, 7, v1 // (op 1525) 
v_and_b32 v3, 15, v1 // (op 1525) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1525) 


v_and_b32 v0, 48, v1 // (op 1525) 
v_lshl_add_u32 v3, v4, 8, v0 // (op 1525) 


v_lshrrev_b32 v0, 7, v1 // (op 1636) 
v_and_b32 v4, 15, v1 // (op 1636) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 1636) 


v_and_b32 v0, 63, v1 // (op 1636) 
v_lshrrev_b32 v4, 4, v0 // (op 1636) 

v_lshl_add_u32 v0, v5, 3, v4 // (op 1636) 


v_lshrrev_b32 v4, 7, v1 // (op 1751) 
v_and_b32 v5, 15, v1 // (op 1751) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 1751) 


v_and_b32 v4, 48, v1 // (op 1751) 
v_lshl_add_u32 v5, v6, 8, v4 // (op 1751) 


v_lshrrev_b32 v4, 7, v1 // (op 1804) 
v_and_b32 v6, 15, v1 // (op 1804) 
v_lshl_add_u32 v7, v4, 4, v6 // (op 1804) 


v_and_b32 v4, 63, v1 // (op 1804) 
v_lshrrev_b32 v6, 4, v4 // (op 1804) 

v_lshl_add_u32 v4, v7, 3, v6 // (op 1804) 


v_lshrrev_b32 v6, 7, v1 // (op 1469) 
s_add_i32 s18, s0, s16 // (op 1469) 

v_and_b32 v7, 63, v1 // (op 1469) 
s_ashr_i32 s0, s18, 31 // (op 1469) 
v_lshrrev_b32 v8, 4, v7 // (op 1469) 
s_cmp_eq_i32 s11, 0 // (op 1469) 
s_mov_b32 s19, scc // (op 1469) 
v_lshlrev_b32 v9, 2, v8 // (op 1469) 

s_cmp_eq_i32 s19, 0 // (op 1469) 
s_cselect_b32 s51, 0, -1 // (op 1469) 

v_lshl_add_u32 v8, v6, 4, v9 // (op 1469) 


s_and_b32 s19, s12, 31 // (op 1469) 
v_lshrrev_b32 v6, 2, v7 // (op 1469) 

v_and_b32 v7, 3, v6 // (op 1469) 

v_and_b32 v6, 63, v1 // (op 1583) 
v_and_b32 v9, 48, v1 // (op 1583) 
v_sub_u32 v10, v6, v9 // (op 1583) 

v_lshrrev_b32 v9, 4, v6 // (op 1583) 

v_lshl_add_u32 v6, v9, 5, v10 // (op 1583) 


v_lshrrev_b32 v9, 6, v1 // (op 1583) 
v_and_b32 v10, 1, v9 // (op 1583) 

v_lshlrev_b32 v9, 4, v10 // (op 1583) 

v_lshl_add_u32 v10, v6, 6, v9 // (op 1583) 


v_mul_lo_u32 v6, v10, 6 // (op 1583) 
v_lshrrev_b32 v9, 3, v6 // (op 1583) 

v_lshrrev_b32 v6, 4, v10 // (op 1583) 

v_lshlrev_b32 v10, 2, v6 // (op 1583) 

v_add_u32 v6, v9, v10 // (op 1583) 


v_lshrrev_b32 v9, 6, v1 // (op 1694) 
v_and_b32 v10, 1, v9 // (op 1694) 

v_and_b32 v9, 15, v1 // (op 1694) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 1694) 


v_and_b32 v9, 63, v1 // (op 1694) 
v_lshrrev_b32 v10, 4, v9 // (op 1694) 

v_lshl_add_u32 v9, v10, 6, v11 // (op 1694) 


v_lshrrev_b32 v10, 7, v1 // (op 1710) 
v_and_b32 v11, 63, v1 // (op 1710) 
v_and_b32 v12, 63, v1 // (op 1780) 
v_and_b32 v13, 48, v1 // (op 1780) 
v_sub_u32 v14, v12, v13 // (op 1780) 

v_lshrrev_b32 v13, 4, v12 // (op 1780) 

v_lshl_add_u32 v12, v13, 5, v14 // (op 1780) 


v_lshrrev_b32 v13, 6, v1 // (op 1780) 
v_and_b32 v14, 1, v13 // (op 1780) 

v_lshlrev_b32 v13, 4, v14 // (op 1780) 

v_lshl_add_u32 v14, v12, 6, v13 // (op 1780) 


v_mul_lo_u32 v12, v14, 6 // (op 1780) 
v_lshrrev_b32 v13, 3, v12 // (op 1780) 

v_lshrrev_b32 v12, 4, v14 // (op 1780) 

v_lshlrev_b32 v14, 2, v12 // (op 1780) 

v_add_u32 v12, v13, v14 // (op 1780) 


v_lshrrev_b32 v13, 6, v1 // (op 1833) 
v_and_b32 v14, 1, v13 // (op 1833) 

v_and_b32 v13, 15, v1 // (op 1833) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 1833) 


v_and_b32 v13, 63, v1 // (op 1833) 
v_lshrrev_b32 v14, 4, v13 // (op 1833) 

v_lshl_add_u32 v13, v14, 6, v15 // (op 1833) 


s_lshl_b32 s52, 1, s19 // (op 1469) 
v_and_b32 v14, 3, v1 // (op 1469) 
s_add_i32 s53, s52, s51 // (op 1469) 


s_and_b32 s51, s0, s53 // (op 1469) 


s_add_i32 s0, s1, s16 // (op 1710) 

s_ashr_i32 s1, s0, 31 // (op 1710) 
v_lshl_add_u32 v15, v7, 2, v14 // (op 1469) 


s_add_i32 s52, s18, s51 // (op 1469) 


v_lshrrev_b32 v7, 6, v1 // (op 1469) 
s_ashr_i32 s18, s52, s19 // (op 1469) 


v_and_b32 v14, 1, v7 // (op 1469) 

s_lshl_b32 s19, s12, 24 // (op 1469) 
v_lshl_add_u32 v7, v14, 4, v15 // (op 1469) 


s_ashr_i32 s51, s19, 31 // (op 1469) 

s_xor_b32 s19, s18, s51 // (op 1469) 

s_sub_i32 s18, s19, s51 // (op 1469) 


s_mul_hi_i32 s19, s18, s13 // (op 1469) 
s_add_i32 s51, s19, s18 // (op 1469) 

s_ashr_i32 s19, s51, 31 // (op 1469) 
s_cmp_eq_i32 s13, 0 // (op 1469) 
s_mov_b32 s52, scc // (op 1469) 
s_cmp_eq_i32 s52, 0 // (op 1469) 
s_cselect_b32 s53, 0, -1 // (op 1469) 

s_and_b32 s52, s14, 31 // (op 1469) 
v_lshrrev_b32 v14, 4, v11 // (op 1710) 
s_cmp_eq_i32 s11, 0 // (op 1710) 
s_mov_b32 s54, scc // (op 1710) 
v_lshlrev_b32 v15, 2, v14 // (op 1710) 

s_cmp_eq_i32 s54, 0 // (op 1710) 
s_cselect_b32 s55, 0, -1 // (op 1710) 

s_lshl_b32 s54, 1, s52 // (op 1469) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 1710) 


v_lshrrev_b32 v10, 2, v11 // (op 1710) 

s_add_i32 s56, s54, s53 // (op 1469) 


s_and_b32 s53, s19, s56 // (op 1469) 


s_add_i32 s19, s51, s53 // (op 1469) 


s_ashr_i32 s51, s19, s52 // (op 1469) 


s_lshl_b32 s19, s14, 24 // (op 1469) 
s_ashr_i32 s52, s19, 31 // (op 1469) 

s_xor_b32 s19, s51, s52 // (op 1469) 

s_sub_i32 s51, s19, s52 // (op 1469) 


s_and_b32 s19, s12, 31 // (op 1710) 
s_lshl_b32 s52, 1, s19 // (op 1710) 
v_and_b32 v11, 3, v10 // (op 1710) 

s_add_i32 s53, s52, s55 // (op 1710) 


v_and_b32 v10, 3, v1 // (op 1710) 
s_and_b32 s52, s1, s53 // (op 1710) 


v_lshl_add_u32 v15, v11, 2, v10 // (op 1710) 


s_add_i32 s1, s0, s52 // (op 1710) 


v_lshrrev_b32 v10, 6, v1 // (op 1710) 
s_ashr_i32 s0, s1, s19 // (op 1710) 


v_and_b32 v11, 1, v10 // (op 1710) 

s_lshl_b32 s1, s12, 24 // (op 1710) 
v_lshl_add_u32 v10, v11, 4, v15 // (op 1710) 


s_ashr_i32 s19, s1, 31 // (op 1710) 

s_xor_b32 s1, s0, s19 // (op 1710) 

s_sub_i32 s0, s1, s19 // (op 1710) 


s_mul_hi_i32 s1, s0, s13 // (op 1710) 
s_add_i32 s19, s1, s0 // (op 1710) 

s_ashr_i32 s1, s19, 31 // (op 1710) 
s_cmp_eq_i32 s13, 0 // (op 1710) 
s_mov_b32 s52, scc // (op 1710) 
s_cmp_eq_i32 s52, 0 // (op 1710) 
s_cselect_b32 s53, 0, -1 // (op 1710) 

s_and_b32 s52, s14, 31 // (op 1710) 
s_lshl_b32 s54, 1, s52 // (op 1710) 
s_add_i32 s55, s54, s53 // (op 1710) 


s_and_b32 s53, s1, s55 // (op 1710) 


s_add_i32 s1, s19, s53 // (op 1710) 


s_ashr_i32 s19, s1, s52 // (op 1710) 


s_lshl_b32 s1, s14, 24 // (op 1710) 
s_ashr_i32 s52, s1, 31 // (op 1710) 

s_xor_b32 s1, s19, s52 // (op 1710) 

s_sub_i32 s19, s1, s52 // (op 1710) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s18, s44 // (op 1469) 
s_sub_u32 s52, s16, s1 // (op 1469) 

s_mul_i32 s1, s51, s44 // (op 1469) 
s_add_u32 s1, s1, s52 // (op 1469) 

s_mul_hi_i32 s52, s16, s10 // (op 1469) 
s_add_i32 s53, s52, s16 // (op 1469) 

s_ashr_i32 s52, s53, 31 // (op 1469) 
s_cmp_eq_i32 s10, 0 // (op 1469) 
s_mov_b32 s54, scc // (op 1469) 
s_cmp_eq_i32 s54, 0 // (op 1469) 
s_cselect_b32 s55, 0, -1 // (op 1469) 

s_and_b32 s54, s47, 31 // (op 1469) 
s_lshl_b32 s56, 1, s54 // (op 1469) 
s_add_i32 s57, s56, s55 // (op 1469) 


s_and_b32 s55, s52, s57 // (op 1469) 


s_add_i32 s52, s53, s55 // (op 1469) 


s_ashr_i32 s53, s52, s54 // (op 1469) 


s_lshl_b32 s52, s47, 24 // (op 1469) 
s_ashr_i32 s54, s52, 31 // (op 1469) 

s_xor_b32 s52, s53, s54 // (op 1469) 

s_sub_i32 s53, s52, s54 // (op 1469) 


s_mul_i32 s52, s0, s44 // (op 1710) 
s_sub_u32 s54, s16, s52 // (op 1710) 

s_mul_i32 s52, s19, s44 // (op 1710) 
s_add_u32 s52, s52, s54 // (op 1710) 

s_mul_hi_i32 s54, s16, s10 // (op 1710) 
s_add_i32 s55, s54, s16 // (op 1710) 

s_ashr_i32 s54, s55, 31 // (op 1710) 
s_cmp_eq_i32 s10, 0 // (op 1710) 
s_mov_b32 s56, scc // (op 1710) 
s_cmp_eq_i32 s56, 0 // (op 1710) 
s_cselect_b32 s57, 0, -1 // (op 1710) 

s_and_b32 s56, s47, 31 // (op 1710) 
s_lshl_b32 s58, 1, s56 // (op 1710) 
s_add_i32 s59, s58, s57 // (op 1710) 


s_and_b32 s57, s54, s59 // (op 1710) 


s_add_i32 s54, s55, s57 // (op 1710) 


s_ashr_i32 s55, s54, s56 // (op 1710) 


s_lshl_b32 s54, s47, 24 // (op 1710) 
s_ashr_i32 s56, s54, 31 // (op 1710) 

s_xor_b32 s54, s55, s56 // (op 1710) 

s_sub_i32 s55, s54, s56 // (op 1710) 


s_add_u32 s56, s6, 64 // (op 1469) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1469) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1469) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1469) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1469) 

s_mul_i32 s54, s44, s56 // (op 1469) 
s_mul_i32 s58, s48, s54 // (op 1469) 

s_mul_i32 s54, s53, s58 // (op 1469) 

s_sub_u32 s58, s16, s54 // (op 1469) 

s_mov_b32 s54, s58 // (op 1469) 
s_mul_hi_i32 s59, s54, s15 // (op 1469) 
s_add_i32 s60, s59, s54 // (op 1469) 


s_ashr_i32 s54, s60, 31 // (op 1469) 
s_cmp_eq_i32 s15, 0 // (op 1469) 
s_mov_b32 s59, scc // (op 1469) 
s_cmp_eq_i32 s59, 0 // (op 1469) 
s_cselect_b32 s61, 0, -1 // (op 1469) 

s_and_b32 s59, s46, 31 // (op 1469) 
s_lshl_b32 s62, 1, s59 // (op 1469) 
s_add_i32 s63, s62, s61 // (op 1469) 


s_and_b32 s61, s54, s63 // (op 1469) 


s_add_i32 s54, s60, s61 // (op 1469) 


s_ashr_i32 s60, s54, s59 // (op 1469) 


s_lshl_b32 s54, s46, 24 // (op 1469) 
s_ashr_i32 s59, s54, 31 // (op 1469) 

s_xor_b32 s54, s60, s59 // (op 1469) 

s_sub_i32 s60, s54, s59 // (op 1469) 


s_mul_i32 s54, s45, s44 // (op 1469) 
s_add_u32 s62, s8, 16 // (op 1469) Start of Int64 add, locking SCC
s_addc_u32 s63, s9, 0 // (op 1469) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1469) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1469) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 4 // (op 1469) 

s_sub_i32 s59, s62, s54 // (op 1469) 

s_mul_i32 s61, s60, s59 // (op 1469) 

s_sub_u32 s59, s58, s61 // (op 1469) 


s_add_u32 s58, s59, s54 // (op 1469) 


s_cmp_eq_i32 s53, 0 // (op 1469) 
s_mov_b32 s54, scc // (op 1469) 

s_cmp_eq_i32 s54, 0 // (op 1469) 
s_cselect_b32 s53, s58, s1 // (op 1469) 


s_mul_i32 s1, s51, s56 // (op 1469) 


v_lshl_add_u32 v11, s53, 4, v8 // (op 1469) 


s_sub_i32 s51, s18, s1 // (op 1469) 


s_cmp_eq_i32 s54, 0 // (op 1469) 
s_cselect_b32 s1, s60, s51 // (op 1469) 



v_lshl_add_u32 v8, s1, 6, v7 // (op 1469) 


v_mul_lo_u32 v7, v8, s2 // (op 1469) 

v_add_lshl_u32 v8, v11, v7, 1 // (op 1469) 


s_add_u32 s56, s6, 64 // (op 1710) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1710) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1710) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1710) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1710) 

s_mul_i32 s1, s44, s56 // (op 1710) 
s_mul_i32 s18, s48, s1 // (op 1710) 

s_mul_i32 s1, s55, s18 // (op 1710) 

s_sub_u32 s18, s16, s1 // (op 1710) 

s_mov_b32 s1, s18 // (op 1710) 
s_mul_hi_i32 s51, s1, s15 // (op 1710) 
s_add_i32 s53, s51, s1 // (op 1710) 


s_ashr_i32 s1, s53, 31 // (op 1710) 
s_cmp_eq_i32 s15, 0 // (op 1710) 
s_mov_b32 s51, scc // (op 1710) 
s_cmp_eq_i32 s51, 0 // (op 1710) 
s_cselect_b32 s54, 0, -1 // (op 1710) 

s_and_b32 s51, s46, 31 // (op 1710) 
s_lshl_b32 s58, 1, s51 // (op 1710) 
s_add_i32 s59, s58, s54 // (op 1710) 


s_and_b32 s54, s1, s59 // (op 1710) 


s_add_i32 s1, s53, s54 // (op 1710) 


s_ashr_i32 s53, s1, s51 // (op 1710) 


s_lshl_b32 s1, s46, 24 // (op 1710) 
s_ashr_i32 s51, s1, 31 // (op 1710) 

s_xor_b32 s1, s53, s51 // (op 1710) 

s_sub_i32 s53, s1, s51 // (op 1710) 


s_mul_i32 s1, s45, s44 // (op 1710) 
s_add_u32 s58, s8, 16 // (op 1710) Start of Int64 add, locking SCC
s_addc_u32 s59, s9, 0 // (op 1710) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1710) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1710) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 4 // (op 1710) 

s_sub_i32 s51, s58, s1 // (op 1710) 

s_mul_i32 s54, s53, s51 // (op 1710) 

s_sub_u32 s51, s18, s54 // (op 1710) 


s_add_u32 s18, s51, s1 // (op 1710) 


s_cmp_eq_i32 s55, 0 // (op 1710) 
s_mov_b32 s1, scc // (op 1710) 

s_cmp_eq_i32 s1, 0 // (op 1710) 
s_cselect_b32 s51, s18, s52 // (op 1710) 


s_mul_i32 s18, s19, s56 // (op 1710) 


v_lshl_add_u32 v7, s51, 4, v14 // (op 1710) 


s_sub_i32 s19, s0, s18 // (op 1710) 


s_cmp_eq_i32 s1, 0 // (op 1710) 
s_cselect_b32 s0, s53, s19 // (op 1710) 



v_lshl_add_u32 v11, s0, 6, v10 // (op 1710) 


v_mul_lo_u32 v10, v11, s4 // (op 1710) 

v_add_lshl_u32 v11, v7, v10, 1 // (op 1710) 


s_mul_i32 s1, s2, s7 // (op 1471) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 1471) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 1471) least significant
s_mul_hi_u32 s19, s2, s6 // (op 1471) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1471) most significant: sum
s_add_u32 s1, s1, s19 // (op 1471) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1471) 

s_mul_i32 s1, s4, s7 // (op 1712) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 1712) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 1712) least significant
s_mul_hi_u32 s52, s4, s6 // (op 1712) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1712) most significant: sum
s_add_u32 s1, s1, s52 // (op 1712) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 1712) 

s_mov_b32 s56, s20 // (op 1471) reg expression
s_mov_b32 s57, s21 // (op 1471) reg expression
s_mov_b32 s58, s18 // (op 1471) reg expression
s_mov_b32 s59, 131072 // (op 1471) call()

s_mov_b32 s60, s22 // (op 1712) reg expression

s_mov_b32 s61, s23 // (op 1712) reg expression
s_mov_b32 s62, s52 // (op 1712) reg expression
s_mov_b32 s63, 131072 // (op 1712) call()


v_mov_b32 v7, s49 // (op 2) Move value
v_mov_b32 v10, s50 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 402) call()
v_accvgpr_write a5, 0.00000 // (op 402) call()
v_accvgpr_write a6, 0.00000 // (op 402) call()
v_accvgpr_write a7, 0.00000 // (op 402) call()
s_mul_hi_i32 s0, s16, s11 // (op 1236) 
v_lshrrev_b32 v14, 4, v1 // (op 1236) 
s_add_i32 s1, s0, s16 // (op 1236) 

v_and_b32 v15, -16, v1 // (op 1236) 
s_ashr_i32 s0, s1, 31 // (op 1236) 
v_sub_u32 v16, v1, v15 // (op 1236) 

s_cmp_eq_i32 s11, 0 // (op 1236) 
s_mov_b32 s18, scc // (op 1236) 
v_lshlrev_b32 v15, 4, v16 // (op 1236) 

s_cmp_eq_i32 s18, 0 // (op 1236) 
s_cselect_b32 s19, 0, -1 // (op 1236) 

v_mov_b32 v17, 0 // (op 1236) convert
v_mov_b32 v16, v15 // (op 1236) convert

s_and_b32 s18, s12, 31 // (op 1236) 
s_lshl_b32 s20, 1, s18 // (op 1236) 
s_add_i32 s21, s20, s19 // (op 1236) 


s_and_b32 s19, s0, s21 // (op 1236) 


s_add_i32 s0, s1, s19 // (op 1236) 


s_ashr_i32 s1, s0, s18 // (op 1236) 


s_lshl_b32 s0, s12, 24 // (op 1236) 
s_ashr_i32 s18, s0, 31 // (op 1236) 

s_xor_b32 s0, s1, s18 // (op 1236) 

s_sub_i32 s1, s0, s18 // (op 1236) 


s_mul_hi_i32 s0, s1, s13 // (op 1236) 
s_add_i32 s18, s0, s1 // (op 1236) 

s_ashr_i32 s0, s18, 31 // (op 1236) 
s_cmp_eq_i32 s13, 0 // (op 1236) 
s_mov_b32 s19, scc // (op 1236) 
s_cmp_eq_i32 s19, 0 // (op 1236) 
s_cselect_b32 s20, 0, -1 // (op 1236) 

s_and_b32 s19, s14, 31 // (op 1236) 
s_lshl_b32 s21, 1, s19 // (op 1236) 
s_add_i32 s22, s21, s20 // (op 1236) 


s_and_b32 s23, s0, s22 // (op 1236) 


s_add_i32 s0, s18, s23 // (op 1236) 


s_ashr_i32 s18, s0, s19 // (op 1236) 


s_lshl_b32 s0, s14, 24 // (op 1236) 
s_ashr_i32 s19, s0, 31 // (op 1236) 

s_xor_b32 s0, s18, s19 // (op 1236) 

s_sub_i32 s18, s0, s19 // (op 1236) 


s_mul_i32 s0, s1, s44 // (op 1236) 

s_sub_u32 s1, s16, s0 // (op 1236) 

s_mul_i32 s0, s18, s44 // (op 1236) 
s_add_u32 s0, s0, s1 // (op 1236) 


s_mul_hi_i32 s1, s16, s10 // (op 1236) 
s_add_i32 s18, s1, s16 // (op 1236) 

s_ashr_i32 s1, s18, 31 // (op 1236) 
s_cmp_eq_i32 s10, 0 // (op 1236) 
s_mov_b32 s19, scc // (op 1236) 
s_cmp_eq_i32 s19, 0 // (op 1236) 
s_cselect_b32 s20, 0, -1 // (op 1236) 

s_and_b32 s19, s47, 31 // (op 1236) 
s_lshl_b32 s21, 1, s19 // (op 1236) 
s_add_i32 s22, s21, s20 // (op 1236) 


s_and_b32 s23, s1, s22 // (op 1236) 


s_add_i32 s1, s18, s23 // (op 1236) 


s_ashr_i32 s18, s1, s19 // (op 1236) 


s_lshl_b32 s1, s47, 24 // (op 1236) 
s_ashr_i32 s19, s1, 31 // (op 1236) 

s_xor_b32 s1, s18, s19 // (op 1236) 

s_sub_i32 s18, s1, s19 // (op 1236) 


s_add_u32 s22, s6, 64 // (op 1236) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1236) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 1236) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 1236) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 6 // (op 1236) 

s_mul_i32 s1, s44, s22 // (op 1236) 

s_mul_i32 s19, s48, s1 // (op 1236) 

s_mul_i32 s1, s18, s19 // (op 1236) 

s_sub_u32 s19, s16, s1 // (op 1236) 

s_mov_b32 s1, s19 // (op 1236) 
s_mul_hi_i32 s20, s1, s15 // (op 1236) 
s_add_i32 s21, s20, s1 // (op 1236) 


s_ashr_i32 s1, s21, 31 // (op 1236) 
s_cmp_eq_i32 s15, 0 // (op 1236) 
s_mov_b32 s20, scc // (op 1236) 
s_cmp_eq_i32 s20, 0 // (op 1236) 
s_cselect_b32 s22, 0, -1 // (op 1236) 

s_and_b32 s20, s46, 31 // (op 1236) 
s_lshl_b32 s23, 1, s20 // (op 1236) 
s_add_i32 s49, s23, s22 // (op 1236) 


s_and_b32 s22, s1, s49 // (op 1236) 


s_add_i32 s1, s21, s22 // (op 1236) 


s_ashr_i32 s21, s1, s20 // (op 1236) 


s_lshl_b32 s1, s46, 24 // (op 1236) 
s_ashr_i32 s20, s1, 31 // (op 1236) 

s_xor_b32 s1, s21, s20 // (op 1236) 

s_sub_i32 s21, s1, s20 // (op 1236) 


s_mul_i32 s1, s45, s44 // (op 1236) 
s_add_u32 s22, s8, 16 // (op 1236) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 1236) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 1236) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 1236) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 4 // (op 1236) 

s_sub_i32 s20, s22, s1 // (op 1236) 

s_mul_i32 s22, s21, s20 // (op 1236) 


s_sub_u32 s23, s19, s22 // (op 1236) 


s_add_u32 s19, s23, s1 // (op 1236) 


s_cmp_eq_i32 s18, 0 // (op 1236) 
s_mov_b32 s1, scc // (op 1236) 

s_cmp_eq_i32 s1, 0 // (op 1236) 
s_cselect_b32 s18, s19, s0 // (op 1236) 



v_lshl_add_u32 v15, s18, 4, v14 // (op 1236) 


v_mul_lo_u32 v19, v15, s25 // (op 1236) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1236) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 1236) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 1236) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 1236) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 1236) most significant half



s_mul_i32 s1, s24, s9 // (op 1238) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 1238) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1238) least significant
s_mul_hi_u32 s19, s24, s8 // (op 1238) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1238) most significant: sum
s_add_u32 s1, s1, s19 // (op 1238) most significant: sum


s_mov_b32 s20, s26 // (op 1238) reg expression
s_mov_b32 s21, s27 // (op 1238) reg expression
s_mov_b32 s22, s0 // (op 1238) reg expression
s_mov_b32 s23, 131072 // (op 1238) call()

s_mul_hi_i32 s0, s16, s11 // (op 1255) 
v_lshrrev_b32 v14, 4, v1 // (op 1255) 
s_add_i32 s1, s0, s16 // (op 1255) 

v_and_b32 v15, -16, v1 // (op 1255) 
s_ashr_i32 s0, s1, 31 // (op 1255) 
v_sub_u32 v16, v1, v15 // (op 1255) 

s_cmp_eq_i32 s11, 0 // (op 1255) 
s_mov_b32 s18, scc // (op 1255) 
v_lshlrev_b32 v15, 4, v16 // (op 1255) 

s_cmp_eq_i32 s18, 0 // (op 1255) 
s_cselect_b32 s19, 0, -1 // (op 1255) 

v_mov_b32 v17, 0 // (op 1255) convert
v_mov_b32 v16, v15 // (op 1255) convert

s_and_b32 s18, s12, 31 // (op 1255) 
s_lshl_b32 s49, 1, s18 // (op 1255) 
s_add_i32 s50, s49, s19 // (op 1255) 


s_and_b32 s19, s0, s50 // (op 1255) 


s_add_i32 s0, s1, s19 // (op 1255) 


s_ashr_i32 s1, s0, s18 // (op 1255) 


s_lshl_b32 s0, s12, 24 // (op 1255) 
s_ashr_i32 s18, s0, 31 // (op 1255) 

s_xor_b32 s0, s1, s18 // (op 1255) 

s_sub_i32 s1, s0, s18 // (op 1255) 


s_mul_hi_i32 s0, s1, s13 // (op 1255) 
s_add_i32 s18, s0, s1 // (op 1255) 

s_ashr_i32 s0, s18, 31 // (op 1255) 
s_cmp_eq_i32 s13, 0 // (op 1255) 
s_mov_b32 s19, scc // (op 1255) 
s_cmp_eq_i32 s19, 0 // (op 1255) 
s_cselect_b32 s49, 0, -1 // (op 1255) 

s_and_b32 s19, s14, 31 // (op 1255) 
s_lshl_b32 s50, 1, s19 // (op 1255) 
s_add_i32 s51, s50, s49 // (op 1255) 


s_and_b32 s49, s0, s51 // (op 1255) 


s_add_i32 s0, s18, s49 // (op 1255) 


s_ashr_i32 s18, s0, s19 // (op 1255) 


s_lshl_b32 s0, s14, 24 // (op 1255) 
s_ashr_i32 s19, s0, 31 // (op 1255) 

s_xor_b32 s0, s18, s19 // (op 1255) 

s_sub_i32 s18, s0, s19 // (op 1255) 


s_mul_i32 s0, s1, s44 // (op 1255) 

s_sub_u32 s1, s16, s0 // (op 1255) 

s_mul_i32 s0, s18, s44 // (op 1255) 
s_add_u32 s0, s0, s1 // (op 1255) 


s_mul_hi_i32 s1, s16, s10 // (op 1255) 
s_add_i32 s18, s1, s16 // (op 1255) 

s_ashr_i32 s1, s18, 31 // (op 1255) 
s_cmp_eq_i32 s10, 0 // (op 1255) 
s_mov_b32 s19, scc // (op 1255) 
s_cmp_eq_i32 s19, 0 // (op 1255) 
s_cselect_b32 s49, 0, -1 // (op 1255) 

s_and_b32 s19, s47, 31 // (op 1255) 
s_lshl_b32 s50, 1, s19 // (op 1255) 
s_add_i32 s51, s50, s49 // (op 1255) 


s_and_b32 s49, s1, s51 // (op 1255) 


s_add_i32 s1, s18, s49 // (op 1255) 


s_ashr_i32 s18, s1, s19 // (op 1255) 


s_lshl_b32 s1, s47, 24 // (op 1255) 
s_ashr_i32 s19, s1, 31 // (op 1255) 

s_xor_b32 s1, s18, s19 // (op 1255) 

s_sub_i32 s18, s1, s19 // (op 1255) 


s_add_u32 s54, s6, 64 // (op 1255) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 1255) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1255) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1255) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 1255) 

s_mul_i32 s1, s44, s50 // (op 1255) 

s_mul_i32 s19, s48, s1 // (op 1255) 

s_mul_i32 s1, s18, s19 // (op 1255) 

s_sub_u32 s19, s16, s1 // (op 1255) 

s_mov_b32 s1, s19 // (op 1255) 
s_mul_hi_i32 s49, s1, s15 // (op 1255) 
s_add_i32 s50, s49, s1 // (op 1255) 


s_ashr_i32 s1, s50, 31 // (op 1255) 
s_cmp_eq_i32 s15, 0 // (op 1255) 
s_mov_b32 s49, scc // (op 1255) 
s_cmp_eq_i32 s49, 0 // (op 1255) 
s_cselect_b32 s51, 0, -1 // (op 1255) 

s_and_b32 s49, s46, 31 // (op 1255) 
s_lshl_b32 s52, 1, s49 // (op 1255) 
s_add_i32 s53, s52, s51 // (op 1255) 


s_and_b32 s51, s1, s53 // (op 1255) 


s_add_i32 s1, s50, s51 // (op 1255) 


s_ashr_i32 s50, s1, s49 // (op 1255) 


s_lshl_b32 s1, s46, 24 // (op 1255) 
s_ashr_i32 s49, s1, 31 // (op 1255) 

s_xor_b32 s1, s50, s49 // (op 1255) 

s_sub_i32 s50, s1, s49 // (op 1255) 


s_mul_i32 s1, s45, s44 // (op 1255) 
s_add_u32 s54, s8, 16 // (op 1255) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 1255) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1255) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1255) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 4 // (op 1255) 

s_sub_i32 s49, s54, s1 // (op 1255) 

s_mul_i32 s51, s50, s49 // (op 1255) 


s_sub_u32 s49, s19, s51 // (op 1255) 


s_add_u32 s19, s49, s1 // (op 1255) 


s_cmp_eq_i32 s18, 0 // (op 1255) 
s_mov_b32 s1, scc // (op 1255) 

s_cmp_eq_i32 s1, 0 // (op 1255) 
s_cselect_b32 s18, s19, s0 // (op 1255) 



v_lshl_add_u32 v15, s18, 4, v14 // (op 1255) 


v_mul_lo_u32 v21, v15, s25 // (op 1255) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1255) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 1255) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 1255) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 1255) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 1255) most significant half



s_mul_i32 s1, s24, s9 // (op 1257) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 1257) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1257) least significant
s_mul_hi_u32 s19, s24, s8 // (op 1257) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1257) most significant: sum
s_add_u32 s1, s1, s19 // (op 1257) most significant: sum


s_mov_b32 s20, s26 // (op 1257) reg expression
s_mov_b32 s21, s27 // (op 1257) reg expression
s_mov_b32 s22, s0 // (op 1257) reg expression
s_mov_b32 s23, 131072 // (op 1257) call()

s_mul_hi_i32 s0, s16, s11 // (op 1300) 
v_lshlrev_b32 v14, 1, v1 // (op 1300) 
s_add_i32 s1, s0, s16 // (op 1300) 

s_ashr_i32 s0, s1, 31 // (op 1300) 
s_cmp_eq_i32 s11, 0 // (op 1300) 
s_mov_b32 s18, scc // (op 1300) 
s_cmp_eq_i32 s18, 0 // (op 1300) 
s_cselect_b32 s19, 0, -1 // (op 1300) 

s_and_b32 s18, s12, 31 // (op 1300) 
s_lshl_b32 s49, 1, s18 // (op 1300) 
s_add_i32 s50, s49, s19 // (op 1300) 


s_and_b32 s19, s0, s50 // (op 1300) 


s_add_i32 s0, s1, s19 // (op 1300) 


s_ashr_i32 s1, s0, s18 // (op 1300) 


s_lshl_b32 s0, s12, 24 // (op 1300) 
s_ashr_i32 s18, s0, 31 // (op 1300) 

s_xor_b32 s0, s1, s18 // (op 1300) 

s_sub_i32 s1, s0, s18 // (op 1300) 


s_mul_hi_i32 s0, s1, s13 // (op 1300) 
s_add_i32 s18, s0, s1 // (op 1300) 

s_ashr_i32 s0, s18, 31 // (op 1300) 
s_cmp_eq_i32 s13, 0 // (op 1300) 
s_mov_b32 s19, scc // (op 1300) 
s_cmp_eq_i32 s19, 0 // (op 1300) 
s_cselect_b32 s49, 0, -1 // (op 1300) 

s_and_b32 s19, s14, 31 // (op 1300) 
s_lshl_b32 s50, 1, s19 // (op 1300) 
s_add_i32 s51, s50, s49 // (op 1300) 


s_and_b32 s49, s0, s51 // (op 1300) 


s_add_i32 s0, s18, s49 // (op 1300) 


s_ashr_i32 s18, s0, s19 // (op 1300) 


s_lshl_b32 s0, s14, 24 // (op 1300) 
s_ashr_i32 s19, s0, 31 // (op 1300) 

s_xor_b32 s0, s18, s19 // (op 1300) 

s_sub_i32 s18, s0, s19 // (op 1300) 


s_add_u32 s54, s6, 64 // (op 1300) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 1300) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1300) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1300) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 1300) 

s_mul_i32 s0, s18, s50 // (op 1300) 

s_sub_i32 s49, s1, s0 // (op 1300) 


s_mul_hi_i32 s0, s16, s10 // (op 1300) 
s_add_i32 s1, s0, s16 // (op 1300) 

s_ashr_i32 s0, s1, 31 // (op 1300) 
s_cmp_eq_i32 s10, 0 // (op 1300) 
s_mov_b32 s18, scc // (op 1300) 
s_cmp_eq_i32 s18, 0 // (op 1300) 
s_cselect_b32 s19, 0, -1 // (op 1300) 

s_and_b32 s18, s47, 31 // (op 1300) 
s_lshl_b32 s52, 1, s18 // (op 1300) 
s_add_i32 s53, s52, s19 // (op 1300) 


s_and_b32 s19, s0, s53 // (op 1300) 


s_add_i32 s0, s1, s19 // (op 1300) 


s_ashr_i32 s1, s0, s18 // (op 1300) 


s_lshl_b32 s0, s47, 24 // (op 1300) 
s_ashr_i32 s18, s0, 31 // (op 1300) 

s_xor_b32 s0, s1, s18 // (op 1300) 

s_sub_i32 s1, s0, s18 // (op 1300) 


s_mul_i32 s0, s44, s50 // (op 1300) 

s_mul_i32 s18, s48, s0 // (op 1300) 

s_mul_i32 s0, s1, s18 // (op 1300) 

s_sub_u32 s18, s16, s0 // (op 1300) 

s_mul_hi_i32 s0, s18, s15 // (op 1300) 
s_add_i32 s19, s0, s18 // (op 1300) 


s_ashr_i32 s0, s19, 31 // (op 1300) 
s_cmp_eq_i32 s15, 0 // (op 1300) 
s_mov_b32 s18, scc // (op 1300) 
s_cmp_eq_i32 s18, 0 // (op 1300) 
s_cselect_b32 s50, 0, -1 // (op 1300) 

s_and_b32 s18, s46, 31 // (op 1300) 
s_lshl_b32 s51, 1, s18 // (op 1300) 
s_add_i32 s52, s51, s50 // (op 1300) 


s_and_b32 s50, s0, s52 // (op 1300) 


s_add_i32 s0, s19, s50 // (op 1300) 


s_ashr_i32 s19, s0, s18 // (op 1300) 


s_lshl_b32 s0, s46, 24 // (op 1300) 
s_ashr_i32 s18, s0, 31 // (op 1300) 

s_xor_b32 s0, s19, s18 // (op 1300) 

s_sub_i32 s19, s0, s18 // (op 1300) 


s_cmp_eq_i32 s1, 0 // (op 1300) 
s_mov_b32 s0, scc // (op 1300) 

s_cmp_eq_i32 s0, 0 // (op 1300) 
s_cselect_b32 s1, s19, s49 // (op 1300) 



s_lshl_b32 s0, s1, 6 // (op 1300) 

s_mov_b32 s19, 0 // (op 1300) convert
s_mov_b32 s18, s0 // (op 1300) convert

v_mul_lo_u32 v17, v14, s29 // (op 1300) most significant: low of low * high
v_mul_hi_u32 v15, v14, s28 // (op 1300) most significant: high of low * low
v_mul_lo_u32 v16, v14, s28 // (op 1300) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 1300) most significant: sum

v_mov_b32 v15, s18 // (op 1300) 
v_mov_b32 v22, s19 // (op 1300) 
v_add_co_u32 v16, s[0:1], v16, v15 // (op 1300) least significant half
v_addc_co_u32 v17, s[0:1], v17, v22, s[0:1] // (op 1300) most significant half





v_mul_lo_u32 v15, v17, 6 // (op 1300) most significant: low of high * low
v_mul_hi_u32 v22, v16, 6 // (op 1300) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 1300) least significant: low of low * low
v_add_u32 v15, v15, v22 // (op 1300) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 1300) 

v_mov_b32 v14, v16 // (op 1300) convert

s_mul_i32 s1, s33, 6 // (op 1302) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 1302) least significant
s_mul_hi_u32 s18, s32, 6 // (op 1302) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1302) most significant: sum

s_mul_i32 s19, s28, s1 // (op 1302) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 1302) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 1302) least significant
s_mul_hi_u32 s50, s28, s0 // (op 1302) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1302) most significant: sum
s_add_u32 s19, s19, s50 // (op 1302) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1302) 

s_mov_b32 s52, s30 // (op 1302) reg expression
s_mov_b32 s53, s31 // (op 1302) reg expression
s_mov_b32 s54, s0 // (op 1302) reg expression
s_mov_b32 s55, 131072 // (op 1302) call()

s_mul_hi_i32 s0, s16, s11 // (op 1319) 
v_lshlrev_b32 v15, 1, v1 // (op 1319) 
s_add_i32 s1, s0, s16 // (op 1319) 

s_ashr_i32 s0, s1, 31 // (op 1319) 
s_cmp_eq_i32 s11, 0 // (op 1319) 
s_mov_b32 s18, scc // (op 1319) 
s_cmp_eq_i32 s18, 0 // (op 1319) 
s_cselect_b32 s19, 0, -1 // (op 1319) 

s_and_b32 s18, s12, 31 // (op 1319) 
s_lshl_b32 s49, 1, s18 // (op 1319) 
s_add_i32 s50, s49, s19 // (op 1319) 


s_and_b32 s19, s0, s50 // (op 1319) 


s_add_i32 s0, s1, s19 // (op 1319) 


s_ashr_i32 s1, s0, s18 // (op 1319) 


s_lshl_b32 s0, s12, 24 // (op 1319) 
s_ashr_i32 s18, s0, 31 // (op 1319) 

s_xor_b32 s0, s1, s18 // (op 1319) 

s_sub_i32 s1, s0, s18 // (op 1319) 


s_mul_hi_i32 s0, s1, s13 // (op 1319) 
s_add_i32 s18, s0, s1 // (op 1319) 

s_ashr_i32 s0, s18, 31 // (op 1319) 
s_cmp_eq_i32 s13, 0 // (op 1319) 
s_mov_b32 s19, scc // (op 1319) 
s_cmp_eq_i32 s19, 0 // (op 1319) 
s_cselect_b32 s49, 0, -1 // (op 1319) 

s_and_b32 s19, s14, 31 // (op 1319) 
s_lshl_b32 s50, 1, s19 // (op 1319) 
s_add_i32 s51, s50, s49 // (op 1319) 


s_and_b32 s49, s0, s51 // (op 1319) 


s_add_i32 s0, s18, s49 // (op 1319) 


s_ashr_i32 s18, s0, s19 // (op 1319) 


s_lshl_b32 s0, s14, 24 // (op 1319) 
s_ashr_i32 s19, s0, 31 // (op 1319) 

s_xor_b32 s0, s18, s19 // (op 1319) 

s_sub_i32 s18, s0, s19 // (op 1319) 


s_add_u32 s50, s6, 64 // (op 1319) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1319) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1319) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1319) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1319) 

s_mul_i32 s0, s18, s50 // (op 1319) 

s_sub_i32 s49, s1, s0 // (op 1319) 


s_mul_hi_i32 s0, s16, s10 // (op 1319) 
s_add_i32 s1, s0, s16 // (op 1319) 

s_ashr_i32 s0, s1, 31 // (op 1319) 
s_cmp_eq_i32 s10, 0 // (op 1319) 
s_mov_b32 s18, scc // (op 1319) 
s_cmp_eq_i32 s18, 0 // (op 1319) 
s_cselect_b32 s19, 0, -1 // (op 1319) 

s_and_b32 s18, s47, 31 // (op 1319) 
s_lshl_b32 s64, 1, s18 // (op 1319) 
s_add_i32 s65, s64, s19 // (op 1319) 


s_and_b32 s19, s0, s65 // (op 1319) 


s_add_i32 s0, s1, s19 // (op 1319) 


s_ashr_i32 s1, s0, s18 // (op 1319) 


s_lshl_b32 s0, s47, 24 // (op 1319) 
s_ashr_i32 s18, s0, 31 // (op 1319) 

s_xor_b32 s0, s1, s18 // (op 1319) 

s_sub_i32 s1, s0, s18 // (op 1319) 


s_mul_i32 s0, s44, s50 // (op 1319) 

s_mul_i32 s18, s48, s0 // (op 1319) 

s_mul_i32 s0, s1, s18 // (op 1319) 

s_sub_u32 s18, s16, s0 // (op 1319) 

s_mul_hi_i32 s0, s18, s15 // (op 1319) 
s_add_i32 s19, s0, s18 // (op 1319) 


s_ashr_i32 s0, s19, 31 // (op 1319) 
s_cmp_eq_i32 s15, 0 // (op 1319) 
s_mov_b32 s18, scc // (op 1319) 
s_cmp_eq_i32 s18, 0 // (op 1319) 
s_cselect_b32 s50, 0, -1 // (op 1319) 

s_and_b32 s18, s46, 31 // (op 1319) 
s_lshl_b32 s51, 1, s18 // (op 1319) 
s_add_i32 s64, s51, s50 // (op 1319) 


s_and_b32 s50, s0, s64 // (op 1319) 


s_add_i32 s0, s19, s50 // (op 1319) 


s_ashr_i32 s19, s0, s18 // (op 1319) 


s_lshl_b32 s0, s46, 24 // (op 1319) 
s_ashr_i32 s18, s0, 31 // (op 1319) 

s_xor_b32 s0, s19, s18 // (op 1319) 

s_sub_i32 s19, s0, s18 // (op 1319) 


s_cmp_eq_i32 s1, 0 // (op 1319) 
s_mov_b32 s0, scc // (op 1319) 

s_cmp_eq_i32 s0, 0 // (op 1319) 
s_cselect_b32 s1, s19, s49 // (op 1319) 



s_lshl_b32 s0, s1, 6 // (op 1319) 

s_mov_b32 s19, 0 // (op 1319) convert
s_mov_b32 s18, s0 // (op 1319) convert

v_mul_lo_u32 v17, v15, s29 // (op 1319) most significant: low of low * high
v_mul_hi_u32 v22, v15, s28 // (op 1319) most significant: high of low * low
v_mul_lo_u32 v16, v15, s28 // (op 1319) least significant: low of low * low
v_add_u32 v17, v17, v22 // (op 1319) most significant: sum

v_mov_b32 v22, s18 // (op 1319) 
v_mov_b32 v23, s19 // (op 1319) 
v_add_co_u32 v16, s[0:1], v16, v22 // (op 1319) least significant half
v_addc_co_u32 v17, s[0:1], v17, v23, s[0:1] // (op 1319) most significant half





v_mul_lo_u32 v23, v17, 6 // (op 1319) most significant: low of high * low
v_mul_hi_u32 v15, v16, 6 // (op 1319) most significant: high of low * low
v_mul_lo_u32 v22, v16, 6 // (op 1319) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 1319) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[22:23] // (op 1319) 

v_mov_b32 v15, v16 // (op 1319) convert

s_mul_i32 s1, s33, 6 // (op 1321) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 1321) least significant
s_mul_hi_u32 s18, s32, 6 // (op 1321) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1321) most significant: sum

s_mul_i32 s19, s28, s1 // (op 1321) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 1321) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 1321) least significant
s_mul_hi_u32 s50, s28, s0 // (op 1321) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1321) most significant: sum
s_add_u32 s19, s19, s50 // (op 1321) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1321) 

s_mov_b32 s52, s30 // (op 1321) reg expression
s_mov_b32 s53, s31 // (op 1321) reg expression
s_mov_b32 s54, s0 // (op 1321) reg expression
s_mov_b32 s55, 131072 // (op 1321) call()

s_mul_hi_i32 s0, s16, s11 // (op 1364) 
v_lshrrev_b32 v16, 3, v1 // (op 1364) 
s_add_i32 s1, s0, s16 // (op 1364) 

v_and_b32 v17, -8, v1 // (op 1364) 
s_ashr_i32 s0, s1, 31 // (op 1364) 
v_sub_u32 v22, v1, v17 // (op 1364) 

s_cmp_eq_i32 s11, 0 // (op 1364) 
s_mov_b32 s18, scc // (op 1364) 
s_cmp_eq_i32 s18, 0 // (op 1364) 
s_cselect_b32 s19, 0, -1 // (op 1364) 

s_and_b32 s18, s12, 31 // (op 1364) 
s_lshl_b32 s49, 1, s18 // (op 1364) 
s_add_i32 s50, s49, s19 // (op 1364) 


s_and_b32 s19, s0, s50 // (op 1364) 


s_add_i32 s0, s1, s19 // (op 1364) 


s_ashr_i32 s1, s0, s18 // (op 1364) 


s_lshl_b32 s0, s12, 24 // (op 1364) 
s_ashr_i32 s18, s0, 31 // (op 1364) 

s_xor_b32 s0, s1, s18 // (op 1364) 

s_sub_i32 s1, s0, s18 // (op 1364) 


s_mul_hi_i32 s0, s1, s13 // (op 1364) 
s_add_i32 s18, s0, s1 // (op 1364) 

s_ashr_i32 s0, s18, 31 // (op 1364) 
s_cmp_eq_i32 s13, 0 // (op 1364) 
s_mov_b32 s19, scc // (op 1364) 
s_cmp_eq_i32 s19, 0 // (op 1364) 
s_cselect_b32 s49, 0, -1 // (op 1364) 

s_and_b32 s19, s14, 31 // (op 1364) 
s_lshl_b32 s50, 1, s19 // (op 1364) 
s_add_i32 s51, s50, s49 // (op 1364) 


s_and_b32 s49, s0, s51 // (op 1364) 


s_add_i32 s0, s18, s49 // (op 1364) 


s_ashr_i32 s18, s0, s19 // (op 1364) 


s_lshl_b32 s0, s14, 24 // (op 1364) 
s_ashr_i32 s19, s0, 31 // (op 1364) 

s_xor_b32 s0, s18, s19 // (op 1364) 

s_sub_i32 s18, s0, s19 // (op 1364) 


s_mul_i32 s0, s1, s44 // (op 1364) 

s_sub_u32 s1, s16, s0 // (op 1364) 

s_mul_i32 s0, s18, s44 // (op 1364) 
s_add_u32 s0, s0, s1 // (op 1364) 


s_mul_hi_i32 s1, s16, s10 // (op 1364) 
s_add_i32 s18, s1, s16 // (op 1364) 

s_ashr_i32 s1, s18, 31 // (op 1364) 
s_cmp_eq_i32 s10, 0 // (op 1364) 
s_mov_b32 s19, scc // (op 1364) 
s_cmp_eq_i32 s19, 0 // (op 1364) 
s_cselect_b32 s49, 0, -1 // (op 1364) 

s_and_b32 s19, s47, 31 // (op 1364) 
s_lshl_b32 s50, 1, s19 // (op 1364) 
s_add_i32 s51, s50, s49 // (op 1364) 


s_and_b32 s49, s1, s51 // (op 1364) 


s_add_i32 s1, s18, s49 // (op 1364) 


s_ashr_i32 s18, s1, s19 // (op 1364) 


s_lshl_b32 s1, s47, 24 // (op 1364) 
s_ashr_i32 s19, s1, 31 // (op 1364) 

s_xor_b32 s1, s18, s19 // (op 1364) 

s_sub_i32 s18, s1, s19 // (op 1364) 


s_add_u32 s50, s6, 64 // (op 1364) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1364) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1364) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1364) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1364) 

s_mul_i32 s1, s44, s50 // (op 1364) 

s_mul_i32 s19, s48, s1 // (op 1364) 

s_mul_i32 s1, s18, s19 // (op 1364) 

s_sub_u32 s19, s16, s1 // (op 1364) 

s_mov_b32 s1, s19 // (op 1364) 
s_mul_hi_i32 s49, s1, s15 // (op 1364) 
s_add_i32 s50, s49, s1 // (op 1364) 


s_ashr_i32 s1, s50, 31 // (op 1364) 
s_cmp_eq_i32 s15, 0 // (op 1364) 
s_mov_b32 s49, scc // (op 1364) 
s_cmp_eq_i32 s49, 0 // (op 1364) 
s_cselect_b32 s51, 0, -1 // (op 1364) 

s_and_b32 s49, s46, 31 // (op 1364) 
s_lshl_b32 s64, 1, s49 // (op 1364) 
s_add_i32 s65, s64, s51 // (op 1364) 


s_and_b32 s51, s1, s65 // (op 1364) 


s_add_i32 s1, s50, s51 // (op 1364) 


s_ashr_i32 s50, s1, s49 // (op 1364) 


s_lshl_b32 s1, s46, 24 // (op 1364) 
s_ashr_i32 s49, s1, 31 // (op 1364) 

s_xor_b32 s1, s50, s49 // (op 1364) 

s_sub_i32 s50, s1, s49 // (op 1364) 


s_mul_i32 s1, s45, s44 // (op 1364) 
s_add_u32 s64, s8, 16 // (op 1364) Start of Int64 add, locking SCC
s_addc_u32 s65, s9, 0 // (op 1364) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 1364) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 1364) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 1364) 

s_sub_i32 s49, s64, s1 // (op 1364) 

s_mul_i32 s51, s50, s49 // (op 1364) 


s_sub_u32 s49, s19, s51 // (op 1364) 


s_add_u32 s19, s49, s1 // (op 1364) 


s_cmp_eq_i32 s18, 0 // (op 1364) 
s_mov_b32 s1, scc // (op 1364) 

s_cmp_eq_i32 s1, 0 // (op 1364) 
s_cselect_b32 s18, s19, s0 // (op 1364) 



v_lshl_add_u32 v17, s18, 4, v16 // (op 1364) 


v_mul_lo_u32 v16, v17, s34 // (op 1364) 
v_add_u32 v16, v16, v22 // (op 1364) 


s_mul_i32 s1, s34, s9 // (op 1366) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1366) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1366) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1366) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1366) most significant: sum
s_add_u32 s1, s1, s19 // (op 1366) most significant: sum


s_mov_b32 s64, s36 // (op 1366) reg expression
s_mov_b32 s65, s37 // (op 1366) reg expression
s_mov_b32 s66, s0 // (op 1366) reg expression
s_mov_b32 s67, 131072 // (op 1366) call()

s_mul_hi_i32 s0, s16, s11 // (op 1383) 
v_lshrrev_b32 v17, 3, v1 // (op 1383) 
s_add_i32 s1, s0, s16 // (op 1383) 

v_and_b32 v22, -8, v1 // (op 1383) 
s_ashr_i32 s0, s1, 31 // (op 1383) 
v_sub_u32 v23, v1, v22 // (op 1383) 

s_cmp_eq_i32 s11, 0 // (op 1383) 
s_mov_b32 s18, scc // (op 1383) 
s_cmp_eq_i32 s18, 0 // (op 1383) 
s_cselect_b32 s19, 0, -1 // (op 1383) 

s_and_b32 s18, s12, 31 // (op 1383) 
s_lshl_b32 s49, 1, s18 // (op 1383) 
s_add_i32 s50, s49, s19 // (op 1383) 


s_and_b32 s19, s0, s50 // (op 1383) 


s_add_i32 s0, s1, s19 // (op 1383) 


s_ashr_i32 s1, s0, s18 // (op 1383) 


s_lshl_b32 s0, s12, 24 // (op 1383) 
s_ashr_i32 s18, s0, 31 // (op 1383) 

s_xor_b32 s0, s1, s18 // (op 1383) 

s_sub_i32 s1, s0, s18 // (op 1383) 


s_mul_hi_i32 s0, s1, s13 // (op 1383) 
s_add_i32 s18, s0, s1 // (op 1383) 

s_ashr_i32 s0, s18, 31 // (op 1383) 
s_cmp_eq_i32 s13, 0 // (op 1383) 
s_mov_b32 s19, scc // (op 1383) 
s_cmp_eq_i32 s19, 0 // (op 1383) 
s_cselect_b32 s49, 0, -1 // (op 1383) 

s_and_b32 s19, s14, 31 // (op 1383) 
s_lshl_b32 s50, 1, s19 // (op 1383) 
s_add_i32 s51, s50, s49 // (op 1383) 


s_and_b32 s49, s0, s51 // (op 1383) 


s_add_i32 s0, s18, s49 // (op 1383) 


s_ashr_i32 s18, s0, s19 // (op 1383) 


s_lshl_b32 s0, s14, 24 // (op 1383) 
s_ashr_i32 s19, s0, 31 // (op 1383) 

s_xor_b32 s0, s18, s19 // (op 1383) 

s_sub_i32 s18, s0, s19 // (op 1383) 


s_mul_i32 s0, s1, s44 // (op 1383) 

s_sub_u32 s1, s16, s0 // (op 1383) 

s_mul_i32 s0, s18, s44 // (op 1383) 
s_add_u32 s0, s0, s1 // (op 1383) 


s_mul_hi_i32 s1, s16, s10 // (op 1383) 
s_add_i32 s18, s1, s16 // (op 1383) 

s_ashr_i32 s1, s18, 31 // (op 1383) 
s_cmp_eq_i32 s10, 0 // (op 1383) 
s_mov_b32 s19, scc // (op 1383) 
s_cmp_eq_i32 s19, 0 // (op 1383) 
s_cselect_b32 s49, 0, -1 // (op 1383) 

s_and_b32 s19, s47, 31 // (op 1383) 
s_lshl_b32 s50, 1, s19 // (op 1383) 
s_add_i32 s51, s50, s49 // (op 1383) 


s_and_b32 s49, s1, s51 // (op 1383) 


s_add_i32 s1, s18, s49 // (op 1383) 


s_ashr_i32 s18, s1, s19 // (op 1383) 


s_lshl_b32 s1, s47, 24 // (op 1383) 
s_ashr_i32 s19, s1, 31 // (op 1383) 

s_xor_b32 s1, s18, s19 // (op 1383) 

s_sub_i32 s18, s1, s19 // (op 1383) 


s_add_u32 s50, s6, 64 // (op 1383) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1383) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1383) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1383) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1383) 

s_mul_i32 s1, s44, s50 // (op 1383) 

s_mul_i32 s19, s48, s1 // (op 1383) 

s_mul_i32 s1, s18, s19 // (op 1383) 

s_sub_u32 s19, s16, s1 // (op 1383) 

s_mov_b32 s1, s19 // (op 1383) 
s_mul_hi_i32 s49, s1, s15 // (op 1383) 
s_add_i32 s50, s49, s1 // (op 1383) 


s_ashr_i32 s1, s50, 31 // (op 1383) 
s_cmp_eq_i32 s15, 0 // (op 1383) 
s_mov_b32 s49, scc // (op 1383) 
s_cmp_eq_i32 s49, 0 // (op 1383) 
s_cselect_b32 s51, 0, -1 // (op 1383) 

s_and_b32 s49, s46, 31 // (op 1383) 
s_lshl_b32 s68, 1, s49 // (op 1383) 
s_add_i32 s69, s68, s51 // (op 1383) 


s_and_b32 s51, s1, s69 // (op 1383) 


s_add_i32 s1, s50, s51 // (op 1383) 


s_ashr_i32 s50, s1, s49 // (op 1383) 


s_lshl_b32 s1, s46, 24 // (op 1383) 
s_ashr_i32 s49, s1, 31 // (op 1383) 

s_xor_b32 s1, s50, s49 // (op 1383) 

s_sub_i32 s50, s1, s49 // (op 1383) 


s_mul_i32 s1, s45, s44 // (op 1383) 
s_add_u32 s68, s8, 16 // (op 1383) Start of Int64 add, locking SCC
s_addc_u32 s69, s9, 0 // (op 1383) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 1383) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 1383) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 1383) 

s_sub_i32 s49, s68, s1 // (op 1383) 

s_mul_i32 s51, s50, s49 // (op 1383) 


s_sub_u32 s49, s19, s51 // (op 1383) 


s_add_u32 s19, s49, s1 // (op 1383) 


s_cmp_eq_i32 s18, 0 // (op 1383) 
s_mov_b32 s1, scc // (op 1383) 

s_cmp_eq_i32 s1, 0 // (op 1383) 
s_cselect_b32 s18, s19, s0 // (op 1383) 



v_lshl_add_u32 v22, s18, 4, v17 // (op 1383) 


v_mul_lo_u32 v17, v22, s34 // (op 1383) 
v_add_u32 v17, v17, v23 // (op 1383) 


s_mul_i32 s1, s34, s9 // (op 1385) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1385) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1385) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1385) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1385) most significant: sum
s_add_u32 s1, s1, s19 // (op 1385) most significant: sum


s_mov_b32 s64, s36 // (op 1385) reg expression
s_mov_b32 s65, s37 // (op 1385) reg expression
s_mov_b32 s66, s0 // (op 1385) reg expression
s_mov_b32 s67, 131072 // (op 1385) call()

s_mul_hi_i32 s0, s16, s11 // (op 1428) 
v_and_b32 v22, -16, v1 // (op 1428) 
s_add_i32 s1, s0, s16 // (op 1428) 

v_sub_u32 v23, v1, v22 // (op 1428) 

s_ashr_i32 s0, s1, 31 // (op 1428) 
v_lshlrev_b32 v22, 2, v23 // (op 1428) 

s_cmp_eq_i32 s11, 0 // (op 1428) 
s_mov_b32 s18, scc // (op 1428) 
v_lshrrev_b32 v23, 4, v1 // (op 1428) 
s_cmp_eq_i32 s18, 0 // (op 1428) 
s_cselect_b32 s19, 0, -1 // (op 1428) 

s_and_b32 s18, s12, 31 // (op 1428) 
s_lshl_b32 s49, 1, s18 // (op 1428) 
s_add_i32 s50, s49, s19 // (op 1428) 


s_and_b32 s19, s0, s50 // (op 1428) 


s_add_i32 s0, s1, s19 // (op 1428) 


s_ashr_i32 s1, s0, s18 // (op 1428) 


s_lshl_b32 s0, s12, 24 // (op 1428) 
s_ashr_i32 s18, s0, 31 // (op 1428) 

s_xor_b32 s0, s1, s18 // (op 1428) 

s_sub_i32 s1, s0, s18 // (op 1428) 


s_mul_hi_i32 s0, s1, s13 // (op 1428) 
s_add_i32 s18, s0, s1 // (op 1428) 

s_ashr_i32 s0, s18, 31 // (op 1428) 
s_cmp_eq_i32 s13, 0 // (op 1428) 
s_mov_b32 s19, scc // (op 1428) 
s_cmp_eq_i32 s19, 0 // (op 1428) 
s_cselect_b32 s49, 0, -1 // (op 1428) 

s_and_b32 s19, s14, 31 // (op 1428) 
s_lshl_b32 s50, 1, s19 // (op 1428) 
s_add_i32 s51, s50, s49 // (op 1428) 


s_and_b32 s49, s0, s51 // (op 1428) 


s_add_i32 s0, s18, s49 // (op 1428) 


s_ashr_i32 s18, s0, s19 // (op 1428) 


s_lshl_b32 s0, s14, 24 // (op 1428) 
s_ashr_i32 s19, s0, 31 // (op 1428) 

s_xor_b32 s0, s18, s19 // (op 1428) 

s_sub_i32 s18, s0, s19 // (op 1428) 


s_add_u32 s50, s6, 64 // (op 1428) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1428) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1428) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1428) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1428) 

s_mul_i32 s0, s18, s50 // (op 1428) 

s_sub_i32 s49, s1, s0 // (op 1428) 


s_mul_hi_i32 s0, s16, s10 // (op 1428) 
s_add_i32 s1, s0, s16 // (op 1428) 

s_ashr_i32 s0, s1, 31 // (op 1428) 
s_cmp_eq_i32 s10, 0 // (op 1428) 
s_mov_b32 s18, scc // (op 1428) 
s_cmp_eq_i32 s18, 0 // (op 1428) 
s_cselect_b32 s19, 0, -1 // (op 1428) 

s_and_b32 s18, s47, 31 // (op 1428) 
s_lshl_b32 s68, 1, s18 // (op 1428) 
s_add_i32 s69, s68, s19 // (op 1428) 


s_and_b32 s19, s0, s69 // (op 1428) 


s_add_i32 s0, s1, s19 // (op 1428) 


s_ashr_i32 s1, s0, s18 // (op 1428) 


s_lshl_b32 s0, s47, 24 // (op 1428) 
s_ashr_i32 s18, s0, 31 // (op 1428) 

s_xor_b32 s0, s1, s18 // (op 1428) 

s_sub_i32 s1, s0, s18 // (op 1428) 


s_mul_i32 s0, s44, s50 // (op 1428) 

s_mul_i32 s18, s48, s0 // (op 1428) 

s_mul_i32 s0, s1, s18 // (op 1428) 

s_sub_u32 s18, s16, s0 // (op 1428) 

s_mul_hi_i32 s0, s18, s15 // (op 1428) 
s_add_i32 s19, s0, s18 // (op 1428) 


s_ashr_i32 s0, s19, 31 // (op 1428) 
s_cmp_eq_i32 s15, 0 // (op 1428) 
s_mov_b32 s18, scc // (op 1428) 
s_cmp_eq_i32 s18, 0 // (op 1428) 
s_cselect_b32 s50, 0, -1 // (op 1428) 

s_and_b32 s18, s46, 31 // (op 1428) 
s_lshl_b32 s51, 1, s18 // (op 1428) 
s_add_i32 s68, s51, s50 // (op 1428) 


s_and_b32 s50, s0, s68 // (op 1428) 


s_add_i32 s0, s19, s50 // (op 1428) 


s_ashr_i32 s19, s0, s18 // (op 1428) 


s_lshl_b32 s0, s46, 24 // (op 1428) 
s_ashr_i32 s18, s0, 31 // (op 1428) 

s_xor_b32 s0, s19, s18 // (op 1428) 

s_sub_i32 s19, s0, s18 // (op 1428) 


s_cmp_eq_i32 s1, 0 // (op 1428) 
s_mov_b32 s0, scc // (op 1428) 

s_cmp_eq_i32 s0, 0 // (op 1428) 
s_cselect_b32 s1, s19, s49 // (op 1428) 



v_lshl_add_u32 v24, s1, 6, v22 // (op 1428) 


v_mul_lo_u32 v22, v23, s38 // (op 1428) 
v_add_u32 v22, v22, v24 // (op 1428) 


s_mul_i32 s1, s38, s43 // (op 1430) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1430) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1430) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1430) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1430) most significant: sum
s_add_u32 s1, s1, s19 // (op 1430) most significant: sum


s_mov_b32 s68, s40 // (op 1430) reg expression
s_mov_b32 s69, s41 // (op 1430) reg expression
s_mov_b32 s70, s0 // (op 1430) reg expression
s_mov_b32 s71, 131072 // (op 1430) call()

s_mul_hi_i32 s0, s16, s11 // (op 1447) 
v_and_b32 v23, -16, v1 // (op 1447) 
s_add_i32 s1, s0, s16 // (op 1447) 

v_sub_u32 v24, v1, v23 // (op 1447) 

s_ashr_i32 s0, s1, 31 // (op 1447) 
v_lshlrev_b32 v23, 2, v24 // (op 1447) 

s_cmp_eq_i32 s11, 0 // (op 1447) 
s_mov_b32 s18, scc // (op 1447) 
v_lshrrev_b32 v24, 4, v1 // (op 1447) 
s_cmp_eq_i32 s18, 0 // (op 1447) 
s_cselect_b32 s19, 0, -1 // (op 1447) 

s_and_b32 s18, s12, 31 // (op 1447) 
s_lshl_b32 s49, 1, s18 // (op 1447) 
s_add_i32 s50, s49, s19 // (op 1447) 


s_and_b32 s19, s0, s50 // (op 1447) 


s_add_i32 s0, s1, s19 // (op 1447) 


s_ashr_i32 s1, s0, s18 // (op 1447) 


s_lshl_b32 s0, s12, 24 // (op 1447) 
s_ashr_i32 s18, s0, 31 // (op 1447) 

s_xor_b32 s0, s1, s18 // (op 1447) 

s_sub_i32 s1, s0, s18 // (op 1447) 


s_mul_hi_i32 s0, s1, s13 // (op 1447) 
s_add_i32 s18, s0, s1 // (op 1447) 

s_ashr_i32 s0, s18, 31 // (op 1447) 
s_cmp_eq_i32 s13, 0 // (op 1447) 
s_mov_b32 s19, scc // (op 1447) 
s_cmp_eq_i32 s19, 0 // (op 1447) 
s_cselect_b32 s49, 0, -1 // (op 1447) 

s_and_b32 s19, s14, 31 // (op 1447) 
s_lshl_b32 s50, 1, s19 // (op 1447) 
s_add_i32 s51, s50, s49 // (op 1447) 


s_and_b32 s49, s0, s51 // (op 1447) 


s_add_i32 s0, s18, s49 // (op 1447) 


s_ashr_i32 s18, s0, s19 // (op 1447) 


s_lshl_b32 s0, s14, 24 // (op 1447) 
s_ashr_i32 s19, s0, 31 // (op 1447) 

s_xor_b32 s0, s18, s19 // (op 1447) 

s_sub_i32 s18, s0, s19 // (op 1447) 


s_add_u32 s50, s6, 64 // (op 1447) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1447) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1447) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1447) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1447) 

s_mul_i32 s0, s18, s50 // (op 1447) 

s_sub_i32 s49, s1, s0 // (op 1447) 


s_mul_hi_i32 s0, s16, s10 // (op 1447) 
s_add_i32 s1, s0, s16 // (op 1447) 

s_ashr_i32 s0, s1, 31 // (op 1447) 
s_cmp_eq_i32 s10, 0 // (op 1447) 
s_mov_b32 s18, scc // (op 1447) 
s_cmp_eq_i32 s18, 0 // (op 1447) 
s_cselect_b32 s19, 0, -1 // (op 1447) 

s_and_b32 s18, s47, 31 // (op 1447) 
s_lshl_b32 s72, 1, s18 // (op 1447) 
s_add_i32 s73, s72, s19 // (op 1447) 


s_and_b32 s19, s0, s73 // (op 1447) 


s_add_i32 s0, s1, s19 // (op 1447) 


s_ashr_i32 s1, s0, s18 // (op 1447) 


s_lshl_b32 s0, s47, 24 // (op 1447) 
s_ashr_i32 s18, s0, 31 // (op 1447) 

s_xor_b32 s0, s1, s18 // (op 1447) 

s_sub_i32 s1, s0, s18 // (op 1447) 


s_mul_i32 s0, s44, s50 // (op 1447) 

s_mul_i32 s18, s48, s0 // (op 1447) 

s_mul_i32 s0, s1, s18 // (op 1447) 

s_sub_u32 s18, s16, s0 // (op 1447) 

s_mul_hi_i32 s0, s18, s15 // (op 1447) 
s_add_i32 s19, s0, s18 // (op 1447) 


s_ashr_i32 s0, s19, 31 // (op 1447) 
s_cmp_eq_i32 s15, 0 // (op 1447) 
s_mov_b32 s18, scc // (op 1447) 
s_cmp_eq_i32 s18, 0 // (op 1447) 
s_cselect_b32 s50, 0, -1 // (op 1447) 

s_and_b32 s18, s46, 31 // (op 1447) 
s_lshl_b32 s51, 1, s18 // (op 1447) 
s_add_i32 s72, s51, s50 // (op 1447) 


s_and_b32 s50, s0, s72 // (op 1447) 


s_add_i32 s0, s19, s50 // (op 1447) 


s_ashr_i32 s19, s0, s18 // (op 1447) 


s_lshl_b32 s0, s46, 24 // (op 1447) 
s_ashr_i32 s18, s0, 31 // (op 1447) 

s_xor_b32 s0, s19, s18 // (op 1447) 

s_sub_i32 s19, s0, s18 // (op 1447) 


s_cmp_eq_i32 s1, 0 // (op 1447) 
s_mov_b32 s0, scc // (op 1447) 

s_cmp_eq_i32 s0, 0 // (op 1447) 
s_cselect_b32 s1, s19, s49 // (op 1447) 



v_lshl_add_u32 v25, s1, 6, v23 // (op 1447) 


v_mul_lo_u32 v23, v24, s38 // (op 1447) 
v_add_u32 v23, v23, v25 // (op 1447) 


s_mul_i32 s1, s38, s43 // (op 1449) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1449) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1449) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1449) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1449) most significant: sum
s_add_u32 s1, s1, s19 // (op 1449) most significant: sum


s_mov_b32 s68, s40 // (op 1449) reg expression
s_mov_b32 s69, s41 // (op 1449) reg expression
s_mov_b32 s70, s0 // (op 1449) reg expression
s_mov_b32 s71, 131072 // (op 1449) call()


s_mul_i32 s1, s29, 6 // (op 1037) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 1037) least significant
s_mul_hi_u32 s18, s28, 6 // (op 1037) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1037) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 1037) 


v_mov_b32 v26, v14 // (op 1037) reg expression
buffer_load_dwordx3 v[28:30], v26, s[52:55], 0 offen // (op 1037) Load value
buffer_load_dwordx3 v[32:34], v26, s[52:55], 0 offen offset:12 // (op 1037) Load value
buffer_load_dwordx3 v[36:38], v26, s[52:55], 0 offen offset:24 // (op 1037) Load value
buffer_load_dwordx3 v[40:42], v26, s[52:55], 0 offen offset:36 // (op 1037) Load value
v_add_u32 v26, v26, v24 // (op 1037) 
buffer_load_dwordx3 v[44:46], v26, s[52:55], 0 offen // (op 1037) Load value
buffer_load_dwordx3 v[48:50], v26, s[52:55], 0 offen offset:12 // (op 1037) Load value
buffer_load_dwordx3 v[52:54], v26, s[52:55], 0 offen offset:24 // (op 1037) Load value
buffer_load_dwordx3 v[56:58], v26, s[52:55], 0 offen offset:36 // (op 1037) Load value




v_mov_b32 v31, v22 // (op 1040) reg expression

buffer_load_dword v35, v31, s[68:71], 0 offen // (op 1040) Load value

v_and_b32 v31, -16, v1 // (op 1513) 
v_sub_u32 v39, v1, v31 // (op 1513) 

v_lshlrev_b32 v31, 4, v39 // (op 1513) 

v_lshrrev_b32 v39, 4, v1 // (op 1513) 
v_lshl_add_u32 v43, v39, 8, v31 // (op 1513) 


s_nop 0
v_readfirstlane_b32 s0, v43 // (op 1513) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 3 // (op 1191) 
v_mov_b64 v[24:25], s[18:19] // (op 1191) convert


v_mov_b64 v[26:27], v[18:19] // (op 1191) 
s_mov_b32 m0, s0 // (op 1191) reg expression
buffer_load_dwordx4 v26, s[20:23], 0 offen offset: 0 lds // (op 1191) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 1191) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 1191) most significant half

s_add_u32 m0, m0, 2048 // (op 1191) 
buffer_load_dwordx4 v26, s[20:23], 0 offen offset: 0 lds // (op 1191) Load value direct to lds






buffer_load_ubyte v31, v16, s[64:67], 0 offen offset: 0 // (op 1046) Load value

s_mul_i32 s1, s29, 6 // (op 1049) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 1049) least significant
s_mul_hi_u32 s18, s28, 6 // (op 1049) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1049) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 1049) 


s_lshl_b64 s[0:1], s[28:29], 8 // (op 1049) 
s_mul_i32 s19, s1, 6 // (op 1049) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 1049) least significant
s_mul_hi_u32 s49, s0, 6 // (op 1049) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1049) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 1049) 

v_add_u32 v39, v14, s0 // (op 1049) 

buffer_load_dwordx3 v[60:62], v39, s[52:55], 0 offen // (op 1049) Load value
buffer_load_dwordx3 v[64:66], v39, s[52:55], 0 offen offset:12 // (op 1049) Load value
buffer_load_dwordx3 v[68:70], v39, s[52:55], 0 offen offset:24 // (op 1049) Load value
buffer_load_dwordx3 v[72:74], v39, s[52:55], 0 offen offset:36 // (op 1049) Load value
v_add_u32 v39, v39, v24 // (op 1049) 
buffer_load_dwordx3 v[76:78], v39, s[52:55], 0 offen // (op 1049) Load value
buffer_load_dwordx3 v[80:82], v39, s[52:55], 0 offen offset:12 // (op 1049) Load value
buffer_load_dwordx3 v[84:86], v39, s[52:55], 0 offen offset:24 // (op 1049) Load value
buffer_load_dwordx3 v[88:90], v39, s[52:55], 0 offen offset:36 // (op 1049) Load value





s_lshl_b32 s0, s38, 3 // (op 1052) 
v_add_u32 v14, v22, s0 // (op 1052) 


buffer_load_dword v39, v14, s[68:71], 0 offen // (op 1052) Load value


v_and_b32 v14, -16, v1 // (op 1739) 
v_sub_u32 v22, v1, v14 // (op 1739) 

v_lshlrev_b32 v14, 4, v22 // (op 1739) 

v_lshrrev_b32 v22, 4, v1 // (op 1739) 
v_lshl_add_u32 v43, v22, 8, v14 // (op 1739) 


s_nop 0
v_readfirstlane_b32 s0, v43 // (op 1739) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 3 // (op 1196) 
v_mov_b64 v[24:25], s[18:19] // (op 1196) convert


v_mov_b64 v[26:27], v[18:19] // (op 1196) 
s_add_u32 s1, 4096, s0 // (op 1196) 
s_mov_b32 m0, s1 // (op 1196) reg expression
s_mov_b32 s18, 256 // (op 1196) call()
buffer_load_dwordx4 v26, s[20:23], s18 offen offset: 0 lds // (op 1196) Load value direct to lds
v_add_co_u32 v26, s[50:51], v26, v24 // (op 1196) least significant half
v_addc_co_u32 v27, s[50:51], 0, v27, s[50:51] // (op 1196) most significant half

s_add_u32 m0, m0, 2048 // (op 1196) 
buffer_load_dwordx4 v26, s[20:23], s18 offen offset: 0 lds // (op 1196) Load value direct to lds









buffer_load_ubyte v14, v16, s[64:67], 0 offen offset:8 // (op 1058) Load value

v_lshlrev_b32 v16, 7, v1 // (op 1566) 
v_mul_lo_u32 v22, v16, 6 // (op 1566) 
v_lshrrev_b32 v43, 3, v22 // (op 1566) 

v_lshrrev_b32 v22, 4, v16 // (op 1566) 

v_lshlrev_b32 v16, 2, v22 // (op 1566) 

v_add_u32 v22, v43, v16 // (op 1566) 





s_waitcnt vmcnt(23)
ds_write_b96 v22, v[28:30] offset:8192 // (op 1060) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v22, v[32:34] offset:8208 // (op 1060) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v22, v[36:38] offset:8224 // (op 1060) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v22, v[40:42] offset:8240 // (op 1060) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v22, v[44:46] offset:8256 // (op 1060) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v22, v[48:50] offset:8272 // (op 1060) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v22, v[52:54] offset:8288 // (op 1060) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v22, v[56:58] offset:8304 // (op 1060) Store local data 

v_and_b32 v16, -16, v1 // (op 1677) 
v_sub_u32 v22, v1, v16 // (op 1677) 

v_lshlrev_b32 v16, 2, v22 // (op 1677) 

v_lshrrev_b32 v22, 4, v1 // (op 1677) 
v_lshl_add_u32 v43, v22, 6, v16 // (op 1677) 





s_waitcnt vmcnt(15)
ds_write_b32 v43, v35 offset:24576 // (op 1061) Store local data 

v_and_b32 v16, -8, v1 // (op 1624) 
v_sub_u32 v22, v1, v16 // (op 1624) 

v_and_b32 v16, 4294967288, v1 // (op 1624) 
v_add_u32 v43, v16, v22 // (op 1624) 





s_waitcnt vmcnt(12)
ds_write_b8 v43, v31 offset:25088 // (op 1063) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1903) 



ds_read_b96_tr_b6 v[92:94], v6 offset:8192 // (op 1081) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v6 offset:9216 // (op 1081) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1944) 


ds_read_u8 v16, v9 offset:24576 // (op 1094) Load local data 



ds_read_b96_tr_b6 v[98:100], v6 offset:8224 // (op 1086) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v6 offset:9248 // (op 1086) Transpose load local data 


ds_read_u8 v22, v9 offset:24608 // (op 1099) Load local data 
s_barrier  // (op 1884) 



ds_read_b128 v[104:107], v3 // (op 1077) Load local data 
ds_read_b128 v[108:111], v3 offset:64 // (op 1077) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1924) 



ds_read_u8 v43, v0 offset:25088 // (op 1090) Load local data 
v_and_b32 v47, -16, v1 // (op 1485) 
v_sub_u32 v51, v1, v47 // (op 1485) 

v_lshlrev_b32 v47, 4, v51 // (op 1485) 

v_lshrrev_b32 v51, 4, v1 // (op 1485) 
v_lshl_add_u32 v55, v51, 8, v47 // (op 1485) 


s_nop 0
v_readfirstlane_b32 s0, v55 // (op 1485) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v47, 7, v1 // (op 1537) 
v_mul_lo_u32 v51, v47, 6 // (op 1537) 
v_lshrrev_b32 v55, 3, v51 // (op 1537) 

v_lshrrev_b32 v51, 4, v47 // (op 1537) 

v_lshlrev_b32 v47, 2, v51 // (op 1537) 

v_add_u32 v51, v55, v47 // (op 1537) 


v_and_b32 v47, -8, v1 // (op 1595) 
v_sub_u32 v55, v1, v47 // (op 1595) 

v_and_b32 v47, 4294967288, v1 // (op 1595) 
v_add_u32 v59, v47, v55 // (op 1595) 


v_and_b32 v47, -16, v1 // (op 1648) 
v_sub_u32 v55, v1, v47 // (op 1648) 

v_lshlrev_b32 v47, 2, v55 // (op 1648) 

v_lshrrev_b32 v55, 4, v1 // (op 1648) 
v_lshl_add_u32 v63, v55, 6, v47 // (op 1648) 


v_and_b32 v47, -16, v1 // (op 1724) 
v_sub_u32 v55, v1, v47 // (op 1724) 

v_lshlrev_b32 v47, 4, v55 // (op 1724) 

v_lshrrev_b32 v55, 4, v1 // (op 1724) 
v_lshl_add_u32 v67, v55, 8, v47 // (op 1724) 


s_nop 0
v_readfirstlane_b32 s1, v67 // (op 1724) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v47, 7, v1 // (op 1763) 
v_mul_lo_u32 v55, v47, 6 // (op 1763) 
v_lshrrev_b32 v67, 3, v55 // (op 1763) 

v_lshrrev_b32 v55, 4, v47 // (op 1763) 

v_lshlrev_b32 v47, 2, v55 // (op 1763) 

v_add_u32 v55, v67, v47 // (op 1763) 


v_and_b32 v47, -8, v1 // (op 1792) 
v_sub_u32 v67, v1, v47 // (op 1792) 

v_and_b32 v47, 4294967288, v1 // (op 1792) 
v_add_u32 v71, v47, v67 // (op 1792) 


v_and_b32 v47, -16, v1 // (op 1816) 
v_sub_u32 v67, v1, v47 // (op 1816) 

v_lshlrev_b32 v47, 2, v67 // (op 1816) 

v_lshrrev_b32 v67, 4, v1 // (op 1816) 
v_lshl_add_u32 v75, v67, 6, v47 // (op 1816) 


s_mov_b32 s18, 0 // (op 404) call()
s_ashr_i64 s[50:51], s[32:33], 8 // (op 403) 
s_mov_b32 s19, s50 // (op 403) convert

s_and_b32 s49, s19, 4294967294 // (op 403) 

s_cmp_lt_u32 s18, s49 // (op 403) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_403 // (op 403) Condition: Top (jump to RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_403 if false)
RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_403:

s_add_u32 s18, s18, 2 // (op 405) 
s_mul_i32 s51, s29, 6 // (op 10) most significant: low of high * low
s_mul_i32 s50, s28, 6 // (op 10) least significant
s_mul_hi_u32 s19, s28, 6 // (op 10) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 10) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[50:51] // (op 10) 


v_mov_b32 v47, v15 // (op 10) 
s_lshl_b64 s[50:51], s[28:29], 8 // (op 10) 
s_mul_i32 s73, s51, 6 // (op 10) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 10) least significant
s_mul_hi_u32 s19, s50, 6 // (op 10) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 10) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 10) 

s_lshl_b32 s19, s50, 1 // (op 10) 

v_add_u32 v67, v47, s19 // (op 10) 

buffer_load_dwordx3 v[28:30], v67, s[52:55], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[32:34], v67, s[52:55], 0 offen offset:12 // (op 10) Load value
buffer_load_dwordx3 v[36:38], v67, s[52:55], 0 offen offset:24 // (op 10) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:111], v[92:97], a[0:3], v43, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 90) 
buffer_load_dwordx3 v[40:42], v67, s[52:55], 0 offen offset:36 // (op 10) Load value
s_barrier  // (op 1893) 
v_add_u32 v67, v67, v18 // (op 10) 



buffer_load_dwordx3 v[44:46], v67, s[52:55], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[48:50], v67, s[52:55], 0 offen offset:12 // (op 10) Load value
ds_read_b128 v[112:115], v3 offset:128 // (op 93) Load local data 
ds_read_b128 v[116:119], v3 offset:192 // (op 93) Load local data 
buffer_load_dwordx3 v[52:54], v67, s[52:55], 0 offen offset:24 // (op 10) Load value
s_barrier  // (op 1913) 
buffer_load_dwordx3 v[56:58], v67, s[52:55], 0 offen offset:36 // (op 10) Load value







ds_read_b96_tr_b6 v[92:94], v6 offset:16384 // (op 97) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v6 offset:17408 // (op 97) Transpose load local data 
v_mov_b32 v67, v23 // (op 25) 
s_barrier  // (op 1934) 
s_lshl_b32 s19, s38, 4 // (op 25) 



ds_read_u8 v79, v0 offset:25092 // (op 101) Load local data 
v_add_u32 v83, v67, s19 // (op 25) 


s_barrier  // (op 1954) 
buffer_load_dword v35, v83, s[68:71], 0 offen // (op 25) Load value



s_waitcnt lgkmcnt(3)
s_barrier  // (op 1887) 
ds_read_u8 v16, v9 offset:24832 // (op 105) Load local data 

s_lshl_b64 s[50:51], s[24:25], 3 // (op 1175) 
v_mov_b64 v[18:19], s[50:51] // (op 1175) convert


v_mov_b64 v[24:25], v[20:21] // (op 1175) 
v_mov_b64 v[26:27], v[24:25] // (op 1175) 
s_mov_b32 m0, s0 // (op 1175) reg expression
s_mov_b32 s19, 512 // (op 1175) call()
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:119], v[92:97], a[0:3], v79, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 107) 
buffer_load_dwordx4 v26, s[20:23], s19 offen offset: 0 lds // (op 1175) Load value direct to lds
v_add_co_u32 v26, s[50:51], v26, v18 // (op 1175) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:111], v[98:103], a[4:7], v43, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 451) 
v_addc_co_u32 v27, s[50:51], 0, v27, s[50:51] // (op 1175) most significant half




s_add_u32 m0, m0, 2048 // (op 1175) 
ds_read_b96_tr_b6 v[98:100], v6 offset:16416 // (op 455) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v6 offset:17440 // (op 455) Transpose load local data 
buffer_load_dwordx4 v26, s[20:23], s19 offen offset: 0 lds // (op 1175) Load value direct to lds







ds_read_u8 v22, v9 offset:24864 // (op 459) Load local data 
v_mov_b32 v83, v17 // (op 17) 

buffer_load_ubyte v31, v83, s[64:67], 0 offen offset:16 // (op 17) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[112:119], v[98:103], a[4:7], v79, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 460) 


s_barrier  // (op 1974) 



s_waitcnt vmcnt(23)
ds_write_b96 v55, v[60:62] offset:25216 // (op 306) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v55, v[64:66] offset:25232 // (op 306) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v55, v[68:70] offset:25248 // (op 306) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v55, v[72:74] offset:25264 // (op 306) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v55, v[76:78] offset:25280 // (op 306) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v55, v[80:82] offset:25296 // (op 306) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v55, v[84:86] offset:25312 // (op 306) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v55, v[88:90] offset:25328 // (op 306) Store local data 
s_barrier  // (op 1986) 



s_waitcnt vmcnt(15)
ds_write_b32 v75, v39 offset:41600 // (op 308) Store local data 
s_barrier  // (op 1980) 



s_waitcnt vmcnt(12)
ds_write_b8 v71, v14 offset:42112 // (op 307) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1970) 



ds_read_b96_tr_b6 v[92:94], v12 offset:25216 // (op 291) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v12 offset:26240 // (op 291) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1983) 


ds_read_u8 v16, v13 offset:41600 // (op 295) Load local data 



ds_read_b96_tr_b6 v[112:114], v12 offset:25248 // (op 474) Transpose load local data 
ds_read_b96_tr_b6 v[115:117], v12 offset:26272 // (op 474) Transpose load local data 


ds_read_u8 v79, v13 offset:41632 // (op 478) Load local data 
s_barrier  // (op 1964) 



ds_read_b128 v[104:107], v5 offset:4096 // (op 289) Load local data 
ds_read_b128 v[108:111], v5 offset:4160 // (op 289) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1977) 



ds_read_u8 v43, v4 offset:42112 // (op 293) Load local data 

s_mul_i32 s51, s29, 6 // (op 309) most significant: low of high * low
s_mul_i32 s50, s28, 6 // (op 309) least significant
s_mul_hi_u32 s19, s28, 6 // (op 309) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 309) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[50:51] // (op 309) 


s_lshl_b64 s[50:51], s[28:29], 8 // (op 309) 
s_mul_i32 s73, s51, 6 // (op 309) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 309) least significant
s_mul_hi_u32 s19, s50, 6 // (op 309) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 309) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 309) 

s_mul_i32 s19, 3, s50 // (op 309) 

v_add_u32 v87, v47, s19 // (op 309) 

buffer_load_dwordx3 v[60:62], v87, s[52:55], 0 offen // (op 309) Load value
buffer_load_dwordx3 v[64:66], v87, s[52:55], 0 offen offset:12 // (op 309) Load value
buffer_load_dwordx3 v[68:70], v87, s[52:55], 0 offen offset:24 // (op 309) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:111], v[92:97], a[0:3], v43, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 296) 
buffer_load_dwordx3 v[72:74], v87, s[52:55], 0 offen offset:36 // (op 309) Load value
v_add_u32 v87, v87, v18 // (op 309) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:111], v[112:117], a[4:7], v43, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 461) 
buffer_load_dwordx3 v[76:78], v87, s[52:55], 0 offen // (op 309) Load value


buffer_load_dwordx3 v[80:82], v87, s[52:55], 0 offen offset:12 // (op 309) Load value
ds_read_b128 v[112:115], v5 offset:4224 // (op 298) Load local data 
ds_read_b128 v[116:119], v5 offset:4288 // (op 298) Load local data 
buffer_load_dwordx3 v[84:86], v87, s[52:55], 0 offen offset:24 // (op 309) Load value
buffer_load_dwordx3 v[88:90], v87, s[52:55], 0 offen offset:36 // (op 309) Load value






ds_read_b96_tr_b6 v[120:122], v12 offset:33408 // (op 300) Transpose load local data 


ds_read_b96_tr_b6 v[123:125], v12 offset:34432 // (op 300) Transpose load local data 
s_lshl_b32 s19, s38, 3 // (op 311) 



ds_read_u8 v47, v4 offset:42116 // (op 302) Load local data 
s_mul_i32 s49, 3, s19 // (op 311) 

v_add_u32 v79, v67, s49 // (op 311) 




s_barrier  // (op 2029) 
buffer_load_dword v39, v79, s[68:71], 0 offen // (op 311) Load value

s_lshl_b64 s[50:51], s[28:29], 8 // (op 1311) 

s_mul_i32 s73, s51, 6 // (op 1311) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 1311) least significant
s_mul_hi_u32 s19, s50, 6 // (op 1311) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 1311) most significant: sum


s_lshl_b32 s19, s38, 4 // (op 1439) 
v_add_u32 v23, v23, s19 // (op 1439) 

s_lshr_b64 s[50:51], s[72:73], 3 // (op 1311) 

s_lshl_b32 s19, s50, 1 // (op 1311) 

v_add_u32 v15, v15, s19 // (op 1311) 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1967) 
ds_read_u8 v67, v13 offset:41856 // (op 304) Load local data 




s_lshl_b64 s[50:51], s[24:25], 3 // (op 1186) 
v_mov_b64 v[18:19], s[50:51] // (op 1186) convert


v_mov_b64 v[26:27], v[24:25] // (op 1186) 
s_add_u32 s19, 4096, s1 // (op 1186) 
s_mov_b32 m0, s19 // (op 1186) reg expression
s_mov_b32 s49, 768 // (op 1186) call()
ds_read_b96_tr_b6 v[126:128], v12 offset:33440 // (op 465) Transpose load local data 
ds_read_b96_tr_b6 v[129:131], v12 offset:34464 // (op 465) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:119], v[120:125], a[0:3], v47, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 305) 




s_barrier  // (op 2067) 
ds_read_u8 v67, v13 offset:41888 // (op 469) Load local data 
buffer_load_dwordx4 v26, s[20:23], s49 offen offset: 0 lds // (op 1186) Load value direct to lds
v_add_co_u32 v26, s[50:51], v26, v18 // (op 1186) least significant half
s_barrier  // (op 2071) 
v_addc_co_u32 v27, s[50:51], 0, v27, s[50:51] // (op 1186) most significant half

s_add_u32 m0, m0, 2048 // (op 1186) 
buffer_load_dwordx4 v26, s[20:23], s49 offen offset: 0 lds // (op 1186) Load value direct to lds




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[112:119], v[126:131], a[4:7], v47, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 470) 








v_add_co_u32 v20, vcc, 512, v20 // (op 1247) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 1247) most significant half
buffer_load_ubyte v14, v83, s[64:67], 0 offen offset:24 // (op 310) Load value

s_barrier  // (op 1906) 
v_add_u32 v17, 16, v17 // (op 1375) 



s_waitcnt vmcnt(23)
ds_write_b96 v51, v[28:30] offset:8192 // (op 13) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v51, v[32:34] offset:8208 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v51, v[36:38] offset:8224 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v51, v[40:42] offset:8240 // (op 13) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v51, v[44:46] offset:8256 // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v51, v[48:50] offset:8272 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v51, v[52:54] offset:8288 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v51, v[56:58] offset:8304 // (op 13) Store local data 
s_barrier  // (op 1947) 



s_waitcnt vmcnt(15)
ds_write_b32 v63, v35 offset:24576 // (op 30) Store local data 
s_barrier  // (op 1927) 



s_waitcnt vmcnt(12)
ds_write_b8 v59, v31 offset:25088 // (op 21) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1916) 



ds_read_b96_tr_b6 v[92:94], v6 offset:8192 // (op 77) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v6 offset:9216 // (op 77) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1957) 


ds_read_u8 v16, v9 offset:24576 // (op 87) Load local data 



ds_read_b96_tr_b6 v[98:100], v6 offset:8224 // (op 446) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v6 offset:9248 // (op 446) Transpose load local data 


ds_read_u8 v22, v9 offset:24608 // (op 450) Load local data 
s_barrier  // (op 1896) 



ds_read_b128 v[104:107], v3 // (op 72) Load local data 
ds_read_b128 v[108:111], v3 offset:64 // (op 72) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1937) 



ds_read_u8 v43, v0 offset:25088 // (op 82) Load local data 
s_ashr_i64 s[50:51], s[32:33], 8 // (op 403) 
s_mov_b32 s19, s50 // (op 403) convert

s_and_b32 s49, s19, 4294967294 // (op 403) 

s_cmp_lt_u32 s18, s49 // (op 403) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_403 // (op 403) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_403 if true)
RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_403:
















s_barrier  // (op 2049) 
s_mul_hi_i32 s0, s16, s11 // (op 1209) 
v_lshrrev_b32 v15, 4, v1 // (op 1209) 
s_add_i32 s1, s0, s16 // (op 1209) 

v_and_b32 v63, -16, v1 // (op 1209) 
s_ashr_i32 s0, s1, 31 // (op 1209) 
v_sub_u32 v67, v1, v63 // (op 1209) 

s_cmp_eq_i32 s11, 0 // (op 1209) 
s_mov_b32 s18, scc // (op 1209) 
v_lshlrev_b32 v63, 4, v67 // (op 1209) 

s_cmp_eq_i32 s18, 0 // (op 1209) 
s_cselect_b32 s19, 0, -1 // (op 1209) 

s_and_b32 s18, s12, 31 // (op 1209) 
s_lshl_b32 s49, 1, s18 // (op 1209) 
s_add_i32 s50, s49, s19 // (op 1209) 


s_and_b32 s19, s0, s50 // (op 1209) 


s_add_i32 s0, s1, s19 // (op 1209) 


s_ashr_i32 s1, s0, s18 // (op 1209) 


s_lshl_b32 s0, s12, 24 // (op 1209) 
s_ashr_i32 s18, s0, 31 // (op 1209) 

s_xor_b32 s0, s1, s18 // (op 1209) 

s_sub_i32 s1, s0, s18 // (op 1209) 


s_mul_hi_i32 s0, s1, s13 // (op 1209) 
s_add_i32 s18, s0, s1 // (op 1209) 

s_ashr_i32 s0, s18, 31 // (op 1209) 
s_cmp_eq_i32 s13, 0 // (op 1209) 
s_mov_b32 s19, scc // (op 1209) 
s_cmp_eq_i32 s19, 0 // (op 1209) 
s_cselect_b32 s49, 0, -1 // (op 1209) 

s_and_b32 s19, s14, 31 // (op 1209) 
s_lshl_b32 s50, 1, s19 // (op 1209) 
s_add_i32 s51, s50, s49 // (op 1209) 


s_and_b32 s49, s0, s51 // (op 1209) 


s_add_i32 s0, s18, s49 // (op 1209) 


s_ashr_i32 s18, s0, s19 // (op 1209) 


s_lshl_b32 s0, s14, 24 // (op 1209) 
s_ashr_i32 s19, s0, 31 // (op 1209) 

s_xor_b32 s0, s18, s19 // (op 1209) 

s_sub_i32 s18, s0, s19 // (op 1209) 


s_mul_i32 s0, s1, s44 // (op 1209) 

s_sub_u32 s1, s16, s0 // (op 1209) 

s_mul_i32 s0, s18, s44 // (op 1209) 
s_add_u32 s0, s0, s1 // (op 1209) 


s_mul_hi_i32 s1, s16, s10 // (op 1209) 
s_add_i32 s18, s1, s16 // (op 1209) 

s_ashr_i32 s1, s18, 31 // (op 1209) 
s_cmp_eq_i32 s10, 0 // (op 1209) 
s_mov_b32 s19, scc // (op 1209) 
s_cmp_eq_i32 s19, 0 // (op 1209) 
s_cselect_b32 s49, 0, -1 // (op 1209) 

s_and_b32 s19, s47, 31 // (op 1209) 
s_lshl_b32 s50, 1, s19 // (op 1209) 
s_add_i32 s51, s50, s49 // (op 1209) 


s_and_b32 s49, s1, s51 // (op 1209) 


s_add_i32 s1, s18, s49 // (op 1209) 


s_ashr_i32 s18, s1, s19 // (op 1209) 


s_lshl_b32 s1, s47, 24 // (op 1209) 
s_ashr_i32 s19, s1, 31 // (op 1209) 

s_xor_b32 s1, s18, s19 // (op 1209) 

s_sub_i32 s18, s1, s19 // (op 1209) 


s_add_u32 s50, s6, 64 // (op 1209) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1209) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1209) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1209) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1209) 

s_mul_i32 s1, s44, s50 // (op 1209) 

s_mul_i32 s19, s48, s1 // (op 1209) 

s_mul_i32 s1, s18, s19 // (op 1209) 

s_sub_u32 s19, s16, s1 // (op 1209) 

s_mov_b32 s1, s19 // (op 1209) 
s_mul_hi_i32 s49, s1, s15 // (op 1209) 
s_add_i32 s50, s49, s1 // (op 1209) 


s_ashr_i32 s1, s50, 31 // (op 1209) 
s_cmp_eq_i32 s15, 0 // (op 1209) 
s_mov_b32 s49, scc // (op 1209) 
s_cmp_eq_i32 s49, 0 // (op 1209) 
s_cselect_b32 s51, 0, -1 // (op 1209) 

s_and_b32 s49, s46, 31 // (op 1209) 
s_lshl_b32 s72, 1, s49 // (op 1209) 
s_add_i32 s73, s72, s51 // (op 1209) 


s_and_b32 s51, s1, s73 // (op 1209) 


s_add_i32 s1, s50, s51 // (op 1209) 


s_ashr_i32 s50, s1, s49 // (op 1209) 


s_lshl_b32 s1, s46, 24 // (op 1209) 
s_ashr_i32 s49, s1, 31 // (op 1209) 

s_xor_b32 s1, s50, s49 // (op 1209) 

s_sub_i32 s50, s1, s49 // (op 1209) 


s_mul_i32 s1, s45, s44 // (op 1209) 
s_add_u32 s72, s8, 16 // (op 1209) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 1209) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1209) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1209) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 1209) 

s_sub_i32 s49, s72, s1 // (op 1209) 

s_mul_i32 s51, s50, s49 // (op 1209) 


s_sub_u32 s49, s19, s51 // (op 1209) 


s_add_u32 s19, s49, s1 // (op 1209) 


s_cmp_eq_i32 s18, 0 // (op 1209) 
s_mov_b32 s1, scc // (op 1209) 

s_cmp_eq_i32 s1, 0 // (op 1209) 
s_cselect_b32 s18, s19, s0 // (op 1209) 



v_lshl_add_u32 v67, s18, 4, v15 // (op 1209) 


s_ashr_i64 s[0:1], s[32:33], 8 // (op 1209) 
s_lshr_b32 s18, s0, 1 // (op 1209) 

v_lshl_add_u32 v15, s18, 9, v63 // (op 1209) 


v_mov_b32 v21, 0 // (op 1209) convert
v_mov_b32 v20, v15 // (op 1209) convert

v_mul_lo_u32 v19, v67, s25 // (op 1209) most significant: low of low * high
v_mul_hi_u32 v15, v67, s24 // (op 1209) most significant: high of low * low
v_mul_lo_u32 v18, v67, s24 // (op 1209) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 1209) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v20 // (op 1209) least significant half
v_addc_co_u32 v19, s[0:1], v19, v21, s[0:1] // (op 1209) most significant half



s_mul_i32 s1, s24, s9 // (op 1211) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 1211) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1211) least significant
s_mul_hi_u32 s19, s24, s8 // (op 1211) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1211) most significant: sum
s_add_u32 s1, s1, s19 // (op 1211) most significant: sum


s_mov_b32 s20, s26 // (op 1211) reg expression
s_mov_b32 s21, s27 // (op 1211) reg expression
s_mov_b32 s22, s0 // (op 1211) reg expression
s_mov_b32 s23, 131072 // (op 1211) call()

s_mul_hi_i32 s0, s16, s11 // (op 1277) 
v_lshlrev_b32 v15, 1, v1 // (op 1277) 
s_add_i32 s1, s0, s16 // (op 1277) 

s_ashr_i32 s0, s1, 31 // (op 1277) 
s_cmp_eq_i32 s11, 0 // (op 1277) 
s_mov_b32 s18, scc // (op 1277) 
s_cmp_eq_i32 s18, 0 // (op 1277) 
s_cselect_b32 s19, 0, -1 // (op 1277) 

s_and_b32 s18, s12, 31 // (op 1277) 
s_lshl_b32 s49, 1, s18 // (op 1277) 
s_add_i32 s50, s49, s19 // (op 1277) 


s_and_b32 s19, s0, s50 // (op 1277) 


s_add_i32 s0, s1, s19 // (op 1277) 


s_ashr_i32 s1, s0, s18 // (op 1277) 


s_lshl_b32 s0, s12, 24 // (op 1277) 
s_ashr_i32 s18, s0, 31 // (op 1277) 

s_xor_b32 s0, s1, s18 // (op 1277) 

s_sub_i32 s1, s0, s18 // (op 1277) 


s_mul_hi_i32 s0, s1, s13 // (op 1277) 
s_add_i32 s18, s0, s1 // (op 1277) 

s_ashr_i32 s0, s18, 31 // (op 1277) 
s_cmp_eq_i32 s13, 0 // (op 1277) 
s_mov_b32 s19, scc // (op 1277) 
s_cmp_eq_i32 s19, 0 // (op 1277) 
s_cselect_b32 s49, 0, -1 // (op 1277) 

s_and_b32 s19, s14, 31 // (op 1277) 
s_lshl_b32 s50, 1, s19 // (op 1277) 
s_add_i32 s51, s50, s49 // (op 1277) 


s_and_b32 s49, s0, s51 // (op 1277) 


s_add_i32 s0, s18, s49 // (op 1277) 


s_ashr_i32 s18, s0, s19 // (op 1277) 


s_lshl_b32 s0, s14, 24 // (op 1277) 
s_ashr_i32 s19, s0, 31 // (op 1277) 

s_xor_b32 s0, s18, s19 // (op 1277) 

s_sub_i32 s18, s0, s19 // (op 1277) 


s_add_u32 s50, s6, 64 // (op 1277) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1277) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1277) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1277) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1277) 

s_mul_i32 s0, s18, s50 // (op 1277) 

s_sub_i32 s49, s1, s0 // (op 1277) 


s_mul_hi_i32 s0, s16, s10 // (op 1277) 
s_add_i32 s1, s0, s16 // (op 1277) 

s_ashr_i32 s0, s1, 31 // (op 1277) 
s_cmp_eq_i32 s10, 0 // (op 1277) 
s_mov_b32 s18, scc // (op 1277) 
s_cmp_eq_i32 s18, 0 // (op 1277) 
s_cselect_b32 s19, 0, -1 // (op 1277) 

s_and_b32 s18, s47, 31 // (op 1277) 
s_lshl_b32 s72, 1, s18 // (op 1277) 
s_add_i32 s73, s72, s19 // (op 1277) 


s_and_b32 s19, s0, s73 // (op 1277) 


s_add_i32 s0, s1, s19 // (op 1277) 


s_ashr_i32 s1, s0, s18 // (op 1277) 


s_lshl_b32 s0, s47, 24 // (op 1277) 
s_ashr_i32 s18, s0, 31 // (op 1277) 

s_xor_b32 s0, s1, s18 // (op 1277) 

s_sub_i32 s1, s0, s18 // (op 1277) 


s_mul_i32 s0, s44, s50 // (op 1277) 

s_mul_i32 s18, s48, s0 // (op 1277) 

s_mul_i32 s0, s1, s18 // (op 1277) 

s_sub_u32 s18, s16, s0 // (op 1277) 

s_mul_hi_i32 s0, s18, s15 // (op 1277) 
s_add_i32 s19, s0, s18 // (op 1277) 


s_ashr_i32 s0, s19, 31 // (op 1277) 
s_cmp_eq_i32 s15, 0 // (op 1277) 
s_mov_b32 s18, scc // (op 1277) 
s_cmp_eq_i32 s18, 0 // (op 1277) 
s_cselect_b32 s50, 0, -1 // (op 1277) 

s_and_b32 s18, s46, 31 // (op 1277) 
s_lshl_b32 s51, 1, s18 // (op 1277) 
s_add_i32 s72, s51, s50 // (op 1277) 


s_and_b32 s50, s0, s72 // (op 1277) 


s_add_i32 s0, s19, s50 // (op 1277) 


s_ashr_i32 s19, s0, s18 // (op 1277) 


s_lshl_b32 s0, s46, 24 // (op 1277) 
s_ashr_i32 s18, s0, 31 // (op 1277) 

s_xor_b32 s0, s19, s18 // (op 1277) 

s_sub_i32 s19, s0, s18 // (op 1277) 


s_cmp_eq_i32 s1, 0 // (op 1277) 
s_mov_b32 s0, scc // (op 1277) 

s_cmp_eq_i32 s0, 0 // (op 1277) 
s_cselect_b32 s1, s19, s49 // (op 1277) 



s_lshl_b32 s0, s1, 6 // (op 1277) 

s_mov_b32 s19, 0 // (op 1277) convert
s_mov_b32 s18, s0 // (op 1277) convert

s_ashr_i64 s[0:1], s[32:33], 8 // (op 1277) 
s_lshr_b32 s49, s0, 1 // (op 1277) 

v_lshl_add_u32 v17, s49, 9, v15 // (op 1277) 


v_mul_lo_u32 v21, v17, s29 // (op 1277) most significant: low of low * high
v_mul_hi_u32 v15, v17, s28 // (op 1277) most significant: high of low * low
v_mul_lo_u32 v20, v17, s28 // (op 1277) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 1277) most significant: sum

v_mov_b32 v15, s18 // (op 1277) 
v_mov_b32 v63, s19 // (op 1277) 
v_add_co_u32 v20, s[0:1], v20, v15 // (op 1277) least significant half
v_addc_co_u32 v21, s[0:1], v21, v63, s[0:1] // (op 1277) most significant half





v_mul_lo_u32 v41, v21, 6 // (op 1277) most significant: low of high * low
v_mul_hi_u32 v15, v20, 6 // (op 1277) most significant: high of low * low
v_mul_lo_u32 v40, v20, 6 // (op 1277) least significant: low of low * low
v_add_u32 v41, v41, v15 // (op 1277) most significant: sum


v_lshrrev_b64 v[20:21], 3, v[40:41] // (op 1277) 

v_mov_b32 v15, v20 // (op 1277) convert

s_mul_i32 s1, s33, 6 // (op 1279) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 1279) least significant
s_mul_hi_u32 s18, s32, 6 // (op 1279) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1279) most significant: sum

s_mul_i32 s19, s28, s1 // (op 1279) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 1279) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 1279) least significant
s_mul_hi_u32 s50, s28, s0 // (op 1279) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1279) most significant: sum
s_add_u32 s19, s19, s50 // (op 1279) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1279) 

s_mov_b32 s52, s30 // (op 1279) reg expression
s_mov_b32 s53, s31 // (op 1279) reg expression
s_mov_b32 s54, s0 // (op 1279) reg expression
s_mov_b32 s55, 131072 // (op 1279) call()

s_mul_hi_i32 s0, s16, s11 // (op 1341) 
v_lshrrev_b32 v17, 3, v1 // (op 1341) 
s_add_i32 s1, s0, s16 // (op 1341) 

v_and_b32 v63, -8, v1 // (op 1341) 
s_ashr_i32 s0, s1, 31 // (op 1341) 
v_sub_u32 v67, v1, v63 // (op 1341) 

s_cmp_eq_i32 s11, 0 // (op 1341) 
s_mov_b32 s18, scc // (op 1341) 
s_cmp_eq_i32 s18, 0 // (op 1341) 
s_cselect_b32 s19, 0, -1 // (op 1341) 

s_and_b32 s18, s12, 31 // (op 1341) 
s_lshl_b32 s49, 1, s18 // (op 1341) 
s_add_i32 s50, s49, s19 // (op 1341) 


s_and_b32 s19, s0, s50 // (op 1341) 


s_add_i32 s0, s1, s19 // (op 1341) 


s_ashr_i32 s1, s0, s18 // (op 1341) 


s_lshl_b32 s0, s12, 24 // (op 1341) 
s_ashr_i32 s18, s0, 31 // (op 1341) 

s_xor_b32 s0, s1, s18 // (op 1341) 

s_sub_i32 s1, s0, s18 // (op 1341) 


s_mul_hi_i32 s0, s1, s13 // (op 1341) 
s_add_i32 s18, s0, s1 // (op 1341) 

s_ashr_i32 s0, s18, 31 // (op 1341) 
s_cmp_eq_i32 s13, 0 // (op 1341) 
s_mov_b32 s19, scc // (op 1341) 
s_cmp_eq_i32 s19, 0 // (op 1341) 
s_cselect_b32 s49, 0, -1 // (op 1341) 

s_and_b32 s19, s14, 31 // (op 1341) 
s_lshl_b32 s50, 1, s19 // (op 1341) 
s_add_i32 s51, s50, s49 // (op 1341) 


s_and_b32 s49, s0, s51 // (op 1341) 


s_add_i32 s0, s18, s49 // (op 1341) 


s_ashr_i32 s18, s0, s19 // (op 1341) 


s_lshl_b32 s0, s14, 24 // (op 1341) 
s_ashr_i32 s19, s0, 31 // (op 1341) 

s_xor_b32 s0, s18, s19 // (op 1341) 

s_sub_i32 s18, s0, s19 // (op 1341) 


s_mul_i32 s0, s1, s44 // (op 1341) 

s_sub_u32 s1, s16, s0 // (op 1341) 

s_mul_i32 s0, s18, s44 // (op 1341) 
s_add_u32 s0, s0, s1 // (op 1341) 


s_mul_hi_i32 s1, s16, s10 // (op 1341) 
s_add_i32 s18, s1, s16 // (op 1341) 

s_ashr_i32 s1, s18, 31 // (op 1341) 
s_cmp_eq_i32 s10, 0 // (op 1341) 
s_mov_b32 s19, scc // (op 1341) 
s_cmp_eq_i32 s19, 0 // (op 1341) 
s_cselect_b32 s49, 0, -1 // (op 1341) 

s_and_b32 s19, s47, 31 // (op 1341) 
s_lshl_b32 s50, 1, s19 // (op 1341) 
s_add_i32 s51, s50, s49 // (op 1341) 


s_and_b32 s49, s1, s51 // (op 1341) 


s_add_i32 s1, s18, s49 // (op 1341) 


s_ashr_i32 s18, s1, s19 // (op 1341) 


s_lshl_b32 s1, s47, 24 // (op 1341) 
s_ashr_i32 s19, s1, 31 // (op 1341) 

s_xor_b32 s1, s18, s19 // (op 1341) 

s_sub_i32 s18, s1, s19 // (op 1341) 


s_add_u32 s50, s6, 64 // (op 1341) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1341) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1341) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1341) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1341) 

s_mul_i32 s1, s44, s50 // (op 1341) 

s_mul_i32 s19, s48, s1 // (op 1341) 

s_mul_i32 s1, s18, s19 // (op 1341) 

s_sub_u32 s19, s16, s1 // (op 1341) 

s_mov_b32 s1, s19 // (op 1341) 
s_mul_hi_i32 s49, s1, s15 // (op 1341) 
s_add_i32 s50, s49, s1 // (op 1341) 


s_ashr_i32 s1, s50, 31 // (op 1341) 
s_cmp_eq_i32 s15, 0 // (op 1341) 
s_mov_b32 s49, scc // (op 1341) 
s_cmp_eq_i32 s49, 0 // (op 1341) 
s_cselect_b32 s51, 0, -1 // (op 1341) 

s_and_b32 s49, s46, 31 // (op 1341) 
s_lshl_b32 s72, 1, s49 // (op 1341) 
s_add_i32 s73, s72, s51 // (op 1341) 


s_and_b32 s51, s1, s73 // (op 1341) 


s_add_i32 s1, s50, s51 // (op 1341) 


s_ashr_i32 s50, s1, s49 // (op 1341) 


s_lshl_b32 s1, s46, 24 // (op 1341) 
s_ashr_i32 s49, s1, 31 // (op 1341) 

s_xor_b32 s1, s50, s49 // (op 1341) 

s_sub_i32 s50, s1, s49 // (op 1341) 


s_mul_i32 s1, s45, s44 // (op 1341) 
s_add_u32 s72, s8, 16 // (op 1341) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 1341) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1341) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1341) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 1341) 

s_sub_i32 s49, s72, s1 // (op 1341) 

s_mul_i32 s51, s50, s49 // (op 1341) 


s_sub_u32 s49, s19, s51 // (op 1341) 


s_add_u32 s19, s49, s1 // (op 1341) 


s_cmp_eq_i32 s18, 0 // (op 1341) 
s_mov_b32 s1, scc // (op 1341) 

s_cmp_eq_i32 s1, 0 // (op 1341) 
s_cselect_b32 s18, s19, s0 // (op 1341) 



v_lshl_add_u32 v63, s18, 4, v17 // (op 1341) 


s_ashr_i64 s[0:1], s[32:33], 8 // (op 1341) 
s_lshr_b32 s18, s0, 1 // (op 1341) 

v_lshl_add_u32 v17, s18, 4, v67 // (op 1341) 


v_mul_lo_u32 v67, v63, s34 // (op 1341) 
v_add_u32 v67, v67, v17 // (op 1341) 


s_mul_i32 s1, s34, s9 // (op 1343) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1343) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1343) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1343) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1343) most significant: sum
s_add_u32 s1, s1, s19 // (op 1343) most significant: sum


s_mov_b32 s64, s36 // (op 1343) reg expression
s_mov_b32 s65, s37 // (op 1343) reg expression
s_mov_b32 s66, s0 // (op 1343) reg expression
s_mov_b32 s67, 131072 // (op 1343) call()

s_mul_hi_i32 s0, s16, s11 // (op 1405) 
v_and_b32 v17, -16, v1 // (op 1405) 
s_add_i32 s1, s0, s16 // (op 1405) 

v_sub_u32 v63, v1, v17 // (op 1405) 

s_ashr_i32 s0, s1, 31 // (op 1405) 
v_lshlrev_b32 v17, 2, v63 // (op 1405) 

s_cmp_eq_i32 s11, 0 // (op 1405) 
s_mov_b32 s18, scc // (op 1405) 
v_lshrrev_b32 v63, 4, v1 // (op 1405) 
s_cmp_eq_i32 s18, 0 // (op 1405) 
s_cselect_b32 s19, 0, -1 // (op 1405) 

s_and_b32 s18, s12, 31 // (op 1405) 
s_lshl_b32 s49, 1, s18 // (op 1405) 
s_add_i32 s50, s49, s19 // (op 1405) 


s_and_b32 s19, s0, s50 // (op 1405) 


s_add_i32 s0, s1, s19 // (op 1405) 


s_ashr_i32 s1, s0, s18 // (op 1405) 


s_lshl_b32 s0, s12, 24 // (op 1405) 
s_ashr_i32 s18, s0, 31 // (op 1405) 

s_xor_b32 s0, s1, s18 // (op 1405) 

s_sub_i32 s1, s0, s18 // (op 1405) 


s_mul_hi_i32 s0, s1, s13 // (op 1405) 
s_add_i32 s18, s0, s1 // (op 1405) 

s_ashr_i32 s0, s18, 31 // (op 1405) 
s_cmp_eq_i32 s13, 0 // (op 1405) 
s_mov_b32 s19, scc // (op 1405) 
s_cmp_eq_i32 s19, 0 // (op 1405) 
s_cselect_b32 s49, 0, -1 // (op 1405) 

s_and_b32 s19, s14, 31 // (op 1405) 
s_lshl_b32 s50, 1, s19 // (op 1405) 
s_add_i32 s51, s50, s49 // (op 1405) 


s_and_b32 s49, s0, s51 // (op 1405) 


s_add_i32 s0, s18, s49 // (op 1405) 


s_ashr_i32 s18, s0, s19 // (op 1405) 


s_lshl_b32 s0, s14, 24 // (op 1405) 
s_ashr_i32 s19, s0, 31 // (op 1405) 

s_xor_b32 s0, s18, s19 // (op 1405) 

s_sub_i32 s18, s0, s19 // (op 1405) 


s_add_u32 s50, s6, 64 // (op 1405) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1405) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1405) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1405) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1405) 

s_mul_i32 s0, s18, s50 // (op 1405) 

s_sub_i32 s49, s1, s0 // (op 1405) 


s_mul_hi_i32 s0, s16, s10 // (op 1405) 
s_add_i32 s1, s0, s16 // (op 1405) 

s_ashr_i32 s0, s1, 31 // (op 1405) 
s_cmp_eq_i32 s10, 0 // (op 1405) 
s_mov_b32 s18, scc // (op 1405) 
s_cmp_eq_i32 s18, 0 // (op 1405) 
s_cselect_b32 s19, 0, -1 // (op 1405) 

s_and_b32 s18, s47, 31 // (op 1405) 
s_lshl_b32 s72, 1, s18 // (op 1405) 
s_add_i32 s73, s72, s19 // (op 1405) 


s_and_b32 s19, s0, s73 // (op 1405) 


s_add_i32 s0, s1, s19 // (op 1405) 


s_ashr_i32 s1, s0, s18 // (op 1405) 


s_lshl_b32 s0, s47, 24 // (op 1405) 
s_ashr_i32 s18, s0, 31 // (op 1405) 

s_xor_b32 s0, s1, s18 // (op 1405) 

s_sub_i32 s1, s0, s18 // (op 1405) 


s_mul_i32 s0, s44, s50 // (op 1405) 

s_mul_i32 s18, s48, s0 // (op 1405) 

s_mul_i32 s0, s1, s18 // (op 1405) 

s_sub_u32 s18, s16, s0 // (op 1405) 

s_mul_hi_i32 s0, s18, s15 // (op 1405) 
s_add_i32 s19, s0, s18 // (op 1405) 


s_ashr_i32 s0, s19, 31 // (op 1405) 
s_cmp_eq_i32 s15, 0 // (op 1405) 
s_mov_b32 s18, scc // (op 1405) 
s_cmp_eq_i32 s18, 0 // (op 1405) 
s_cselect_b32 s50, 0, -1 // (op 1405) 

s_and_b32 s18, s46, 31 // (op 1405) 
s_lshl_b32 s51, 1, s18 // (op 1405) 
s_add_i32 s72, s51, s50 // (op 1405) 


s_and_b32 s50, s0, s72 // (op 1405) 


s_add_i32 s0, s19, s50 // (op 1405) 


s_ashr_i32 s19, s0, s18 // (op 1405) 


s_lshl_b32 s0, s46, 24 // (op 1405) 
s_ashr_i32 s18, s0, 31 // (op 1405) 

s_xor_b32 s0, s19, s18 // (op 1405) 

s_sub_i32 s19, s0, s18 // (op 1405) 


s_cmp_eq_i32 s1, 0 // (op 1405) 
s_mov_b32 s0, scc // (op 1405) 

s_cmp_eq_i32 s0, 0 // (op 1405) 
s_cselect_b32 s1, s19, s49 // (op 1405) 



v_lshl_add_u32 v71, s1, 6, v17 // (op 1405) 


s_ashr_i64 s[0:1], s[32:33], 8 // (op 1405) 
s_lshr_b32 s18, s0, 1 // (op 1405) 

v_lshl_add_u32 v17, s18, 4, v63 // (op 1405) 


v_mul_lo_u32 v63, v17, s38 // (op 1405) 
v_add_u32 v63, v63, v71 // (op 1405) 


s_mul_i32 s1, s38, s43 // (op 1407) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1407) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1407) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1407) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1407) most significant: sum
s_add_u32 s1, s1, s19 // (op 1407) most significant: sum


s_mov_b32 s68, s40 // (op 1407) reg expression
s_mov_b32 s69, s41 // (op 1407) reg expression
s_mov_b32 s70, s0 // (op 1407) reg expression
s_mov_b32 s71, 131072 // (op 1407) call()

v_and_b32 v17, -16, v1 // (op 1498) 
v_sub_u32 v71, v1, v17 // (op 1498) 

v_lshlrev_b32 v17, 4, v71 // (op 1498) 

v_lshrrev_b32 v71, 4, v1 // (op 1498) 
v_lshl_add_u32 v75, v71, 8, v17 // (op 1498) 


s_nop 0
v_readfirstlane_b32 s0, v75 // (op 1498) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v17, 7, v1 // (op 1550) 
v_mul_lo_u32 v71, v17, 6 // (op 1550) 
v_lshrrev_b32 v75, 3, v71 // (op 1550) 

v_lshrrev_b32 v71, 4, v17 // (op 1550) 

v_lshlrev_b32 v17, 2, v71 // (op 1550) 

v_add_u32 v71, v75, v17 // (op 1550) 


v_and_b32 v17, -8, v1 // (op 1608) 
v_sub_u32 v75, v1, v17 // (op 1608) 

v_and_b32 v17, 4294967288, v1 // (op 1608) 
v_add_u32 v79, v17, v75 // (op 1608) 


v_and_b32 v17, -16, v1 // (op 1661) 
v_sub_u32 v75, v1, v17 // (op 1661) 

v_lshlrev_b32 v17, 2, v75 // (op 1661) 

v_lshrrev_b32 v75, 4, v1 // (op 1661) 
v_lshl_add_u32 v83, v75, 6, v17 // (op 1661) 


s_ashr_i64 s[18:19], s[32:33], 8 // (op 409) 
s_and_b32 s1, s18, 4294967294 // (op 409) 

s_ashr_i64 s[18:19], s[32:33], 8 // (op 408) 
s_mov_b32 s49, s18 // (op 408) convert

s_cmp_lt_u32 s1, s49 // (op 408) 










s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_408 // (op 408) Condition: Top (jump to RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_408 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_408:
s_barrier  // (op 1890) 
s_add_u32 s1, s1, 1 // (op 410) 

s_lshl_b64 s[18:19], s[24:25], 3 // (op 1180) 
v_mov_b64 v[16:17], s[18:19] // (op 1180) convert


v_mov_b64 v[20:21], v[18:19] // (op 1180) 
v_mov_b64 v[22:23], v[20:21] // (op 1180) 
s_mov_b32 m0, s0 // (op 1180) reg expression
buffer_load_dwordx4 v22, s[20:23], 0 offen offset: 0 lds // (op 1180) Load value direct to lds
v_add_co_u32 v22, s[18:19], v22, v16 // (op 1180) least significant half
v_addc_co_u32 v23, s[18:19], 0, v23, s[18:19] // (op 1180) most significant half

s_add_u32 m0, m0, 2048 // (op 1180) 
buffer_load_dwordx4 v22, s[20:23], 0 offen offset: 0 lds // (op 1180) Load value direct to lds



v_add_co_u32 v18, vcc, 256, v18 // (op 1199) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 1199) most significant half

s_mul_i32 s19, s29, 6 // (op 225) most significant: low of high * low
s_mul_i32 s18, s28, 6 // (op 225) least significant
s_mul_hi_u32 s49, s28, 6 // (op 225) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 225) most significant: sum

v_lshrrev_b64 v[16:17], 3, s[18:19] // (op 225) 


v_mov_b32 v14, v15 // (op 225) 
v_mov_b32 v20, v14 // (op 225) reg expression
buffer_load_dwordx3 v[22:24], v20, s[52:55], 0 offen // (op 225) Load value
buffer_load_dwordx3 v[26:28], v20, s[52:55], 0 offen offset:12 // (op 225) Load value
buffer_load_dwordx3 v[30:32], v20, s[52:55], 0 offen offset:24 // (op 225) Load value
buffer_load_dwordx3 v[34:36], v20, s[52:55], 0 offen offset:36 // (op 225) Load value
v_add_u32 v20, v20, v16 // (op 225) 
buffer_load_dwordx3 v[38:40], v20, s[52:55], 0 offen // (op 225) Load value
buffer_load_dwordx3 v[42:44], v20, s[52:55], 0 offen offset:12 // (op 225) Load value
buffer_load_dwordx3 v[46:48], v20, s[52:55], 0 offen offset:24 // (op 225) Load value
buffer_load_dwordx3 v[50:52], v20, s[52:55], 0 offen offset:36 // (op 225) Load value



s_lshl_b64 s[18:19], s[28:29], 8 // (op 1269) 
s_mul_i32 s51, s19, 6 // (op 1269) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 1269) least significant
s_mul_hi_u32 s49, s18, 6 // (op 1269) most significant: high of low * low
s_add_u32 s51, s51, s49 // (op 1269) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 1269) 

v_add_u32 v15, v15, s18 // (op 1269) 



s_barrier  // (op 1909) 
v_mov_b32 v14, v67 // (op 226) 




buffer_load_ubyte v25, v14, s[64:67], 0 offen offset: 0 // (op 226) Load value

v_add_u32 v67, 8, v67 // (op 1333) 
s_waitcnt vmcnt(8)
ds_write_b96 v71, v[22:24] offset:8192 // (op 222) Store local data 


v_mov_b32 v14, v63 // (op 227) 
v_mov_b32 v29, v14 // (op 227) reg expression

buffer_load_dword v33, v29, s[68:71], 0 offen // (op 227) Load value


s_lshl_b32 s18, s38, 3 // (op 1397) 
v_add_u32 v63, v63, s18 // (op 1397) 

s_waitcnt vmcnt(8)
ds_write_b96 v71, v[26:28] offset:8208 // (op 222) Store local data 
s_barrier  // (op 1899) 
s_waitcnt vmcnt(7)
ds_write_b96 v71, v[30:32] offset:8224 // (op 222) Store local data 



ds_read_b128 v[54:57], v3 // (op 205) Load local data 
ds_read_b128 v[58:61], v3 offset:64 // (op 205) Load local data 
s_waitcnt vmcnt(6)
ds_write_b96 v71, v[34:36] offset:8240 // (op 222) Store local data 
s_waitcnt vmcnt(5)
ds_write_b96 v71, v[38:40] offset:8256 // (op 222) Store local data 
s_waitcnt vmcnt(4)
ds_write_b96 v71, v[42:44] offset:8272 // (op 222) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v71, v[46:48] offset:8288 // (op 222) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v71, v[50:52] offset:8304 // (op 222) Store local data 

s_barrier  // (op 1930) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1920) 






ds_read_b96_tr_b6 v[26:28], v6 offset:8192 // (op 207) Transpose load local data 
ds_read_b96_tr_b6 v[29:31], v6 offset:9216 // (op 207) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v79, v25 offset:25088 // (op 223) Store local data 

s_barrier  // (op 1950) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1940) 






ds_read_u8 v14, v0 offset:25088 // (op 209) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v83, v33 offset:24576 // (op 224) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1960) 


ds_read_u8 v62, v9 offset:24576 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:61], v[26:31], a[0:3], v14, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 212) 





ds_read_b128 v[20:23], v3 offset:128 // (op 214) Load local data 
ds_read_b128 v[24:27], v3 offset:192 // (op 214) Load local data 



ds_read_b96_tr_b6 v[28:30], v6 offset:16384 // (op 216) Transpose load local data 
ds_read_b96_tr_b6 v[31:33], v6 offset:17408 // (op 216) Transpose load local data 



ds_read_u8 v62, v0 offset:25092 // (op 218) Load local data 


ds_read_u8 v16, v9 offset:24832 // (op 220) Load local data 



ds_read_b96_tr_b6 v[34:36], v6 offset:8224 // (op 416) Transpose load local data 
ds_read_b96_tr_b6 v[37:39], v6 offset:9248 // (op 416) Transpose load local data 


ds_read_u8 v17, v9 offset:24608 // (op 420) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:61], v[34:39], a[4:7], v14, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 421) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:27], v[28:33], a[0:3], v62, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 221) 
ds_read_b96_tr_b6 v[34:36], v6 offset:16416 // (op 425) Transpose load local data 


ds_read_b96_tr_b6 v[37:39], v6 offset:17440 // (op 425) Transpose load local data 


ds_read_u8 v14, v9 offset:24864 // (op 429) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:27], v[34:39], a[4:7], v62, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 430) 




s_ashr_i64 s[18:19], s[32:33], 8 // (op 408) 
s_mov_b32 s49, s18 // (op 408) convert

s_cmp_lt_u32 s1, s49 // (op 408) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_408 // (op 408) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_408 if true)
RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_408:























s_barrier  // (op 2053) 





















v_cmp_eq_f32 vcc, 0.00000, v7 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_157
v_accvgpr_read v12, a0 // (op 50) 
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_mul_f32 v16, v10, v12 // (op 50) 
v_mul_f32 v17, v10, v13 // (op 50) 
v_mul_f32 v18, v10, v14 // (op 50) 
v_mul_f32 v19, v10, v15 // (op 50) 

v_lshrrev_b32 v12, 16, v16 // (op 56) 
v_lshrrev_b32 v13, 16, v17 // (op 56) 
v_lshrrev_b32 v14, 16, v18 // (op 56) 
v_lshrrev_b32 v15, 16, v19 // (op 56) 


v_pack_b32_f16 v4, v12, v13 // (op 58) 
v_pack_b32_f16 v5, v14, v15 // (op 58) 

v_mov_b32 v0, v11 // (op 58) reg expression

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 58) Store value



v_accvgpr_read v12, a4 // (op 714) 
v_accvgpr_read v13, a5 // (op 714) 
v_accvgpr_read v14, a6 // (op 714) 
v_accvgpr_read v15, a7 // (op 714) 
v_mul_f32 v16, v10, v12 // (op 714) 
v_mul_f32 v17, v10, v13 // (op 714) 
v_mul_f32 v18, v10, v14 // (op 714) 
v_mul_f32 v19, v10, v15 // (op 714) 

v_lshrrev_b32 v12, 16, v16 // (op 716) 
v_lshrrev_b32 v13, 16, v17 // (op 716) 
v_lshrrev_b32 v14, 16, v18 // (op 716) 
v_lshrrev_b32 v15, 16, v19 // (op 716) 


v_pack_b32_f16 v4, v12, v13 // (op 717) 
v_pack_b32_f16 v5, v14, v15 // (op 717) 

s_lshl_b32 s0, s4, 6 // (op 717) 
v_add_u32 v0, v11, s0 // (op 717) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 717) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_157
RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_157:


v_mov_b32 v0, v8 // (op 170) reg expression

buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 170) Load value

v_accvgpr_read v12, a0 // (op 166) 
v_accvgpr_read v13, a1 // (op 166) 
v_accvgpr_read v14, a2 // (op 166) 
v_accvgpr_read v15, a3 // (op 166) 
v_mul_f32 v16, v10, v12 // (op 166) 
v_mul_f32 v17, v10, v13 // (op 166) 
v_mul_f32 v18, v10, v14 // (op 166) 
v_mul_f32 v19, v10, v15 // (op 166) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 171) 
v_lshrrev_b32 v13, 16, v4 // (op 171) 
v_lshlrev_b32 v12, 16, v12 // (op 171) 
v_lshlrev_b32 v13, 16, v13 // (op 171) 
v_mul_f32 v20, v7, v12 // (op 171) 
v_mul_f32 v21, v7, v13 // (op 171) 
v_and_b32 v12, 65535, v5 // (op 171) 
v_lshrrev_b32 v13, 16, v5 // (op 171) 
v_lshlrev_b32 v12, 16, v12 // (op 171) 
v_lshlrev_b32 v13, 16, v13 // (op 171) 
v_mul_f32 v22, v7, v12 // (op 171) 
v_mul_f32 v23, v7, v13 // (op 171) 


v_add_f32 v12, v20, v16 // (op 167) 
v_add_f32 v13, v21, v17 // (op 167) 
v_add_f32 v14, v22, v18 // (op 167) 
v_add_f32 v15, v23, v19 // (op 167) 


v_lshrrev_b32 v16, 16, v12 // (op 168) 
v_lshrrev_b32 v17, 16, v13 // (op 168) 
v_lshrrev_b32 v18, 16, v14 // (op 168) 
v_lshrrev_b32 v19, 16, v15 // (op 168) 


v_pack_b32_f16 v12, v16, v17 // (op 169) 
v_pack_b32_f16 v13, v18, v19 // (op 169) 

v_mov_b32 v0, v11 // (op 169) reg expression

buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 169) Store value





s_lshl_b32 s0, s2, 6 // (op 741) 
v_add_u32 v0, v8, s0 // (op 741) 


buffer_load_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 741) Load value

v_accvgpr_read v12, a4 // (op 737) 
v_accvgpr_read v13, a5 // (op 737) 
v_accvgpr_read v14, a6 // (op 737) 
v_accvgpr_read v15, a7 // (op 737) 
v_mul_f32 v16, v10, v12 // (op 737) 
v_mul_f32 v17, v10, v13 // (op 737) 
v_mul_f32 v18, v10, v14 // (op 737) 
v_mul_f32 v19, v10, v15 // (op 737) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 742) 
v_lshrrev_b32 v13, 16, v4 // (op 742) 
v_lshlrev_b32 v12, 16, v12 // (op 742) 
v_lshlrev_b32 v13, 16, v13 // (op 742) 
v_mul_f32 v20, v7, v12 // (op 742) 
v_mul_f32 v21, v7, v13 // (op 742) 
v_and_b32 v12, 65535, v5 // (op 742) 
v_lshrrev_b32 v13, 16, v5 // (op 742) 
v_lshlrev_b32 v12, 16, v12 // (op 742) 
v_lshlrev_b32 v13, 16, v13 // (op 742) 
v_mul_f32 v22, v7, v12 // (op 742) 
v_mul_f32 v23, v7, v13 // (op 742) 


v_add_f32 v12, v20, v16 // (op 738) 
v_add_f32 v13, v21, v17 // (op 738) 
v_add_f32 v14, v22, v18 // (op 738) 
v_add_f32 v15, v23, v19 // (op 738) 


v_lshrrev_b32 v16, 16, v12 // (op 739) 
v_lshrrev_b32 v17, 16, v13 // (op 739) 
v_lshrrev_b32 v18, 16, v14 // (op 739) 
v_lshrrev_b32 v19, 16, v15 // (op 739) 


v_pack_b32_f16 v12, v16, v17 // (op 740) 
v_pack_b32_f16 v13, v18, v19 // (op 740) 

s_lshl_b32 s0, s4, 6 // (op 740) 
v_add_u32 v0, v11, s0 // (op 740) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 740) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_157:










s_endpgm  // End of RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.LRR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__end:
.size RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_, .LRR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__end-RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
  .amdhsa_next_free_vgpr 140
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 42240 // lds bytes
  .amdhsa_accum_offset 132
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
    DataTypeB: F6
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: True
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
            { type: int64, semantic: SizeFree0 },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: address, semantic: AddressC },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSumScaleA },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha } ]
    macrotile: [16, 64, 256]
    threads: [128, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
    .symbol: RR_GEMM_TT_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 42240
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 132
    .agpr_count: 8
    .max_flat_workgroup_size: 128
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_11
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Tensor_2_size_1_12
        .size: 8
        .offset: 16
        .value_kind: by_value
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 56
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 64
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_4
        .size: 8
        .offset: 104
        .value_kind: by_value
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
      - .name: Tensor_7_stride_0_8
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_9
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 152
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 156
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 160
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 164
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 168
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 172
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 176
        .value_kind: by_value
...

.end_amdgpu_metadata

