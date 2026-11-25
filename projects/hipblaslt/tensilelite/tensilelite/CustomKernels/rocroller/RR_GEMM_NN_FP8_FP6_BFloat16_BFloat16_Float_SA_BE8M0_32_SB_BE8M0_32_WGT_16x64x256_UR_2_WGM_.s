// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_,@function
RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 1469) 
s_mul_hi_i32 s1, s16, s10 // (op 1710) 
v_and_b32 v0, 48, v1 // (op 1525) 
v_and_b32 v3, 63, v1 // (op 1525) 
v_sub_u32 v4, v3, v0 // (op 1525) 

v_lshrrev_b32 v3, 1, v4 // (op 1525) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 1525) 


v_and_b32 v0, 1, v4 // (op 1525) 

v_lshlrev_b32 v3, 3, v0 // (op 1525) 

v_lshrrev_b32 v0, 7, v1 // (op 1525) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1525) 


v_add_u32 v0, v5, v4 // (op 1525) 


v_lshrrev_b32 v3, 7, v1 // (op 1636) 
v_and_b32 v4, 15, v1 // (op 1636) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 1636) 


v_and_b32 v3, 48, v1 // (op 1636) 
v_add_u32 v4, v3, v5 // (op 1636) 


v_and_b32 v3, 48, v1 // (op 1751) 
v_and_b32 v5, 63, v1 // (op 1751) 
v_sub_u32 v6, v5, v3 // (op 1751) 

v_lshrrev_b32 v5, 1, v6 // (op 1751) 
v_add_lshl_u32 v7, v3, v5, 4 // (op 1751) 


v_and_b32 v3, 1, v6 // (op 1751) 

v_lshlrev_b32 v5, 3, v3 // (op 1751) 

v_lshrrev_b32 v3, 7, v1 // (op 1751) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 1751) 


v_add_u32 v3, v7, v6 // (op 1751) 


v_lshrrev_b32 v5, 7, v1 // (op 1804) 
v_and_b32 v6, 15, v1 // (op 1804) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 1804) 


v_and_b32 v5, 48, v1 // (op 1804) 
v_add_u32 v6, v5, v7 // (op 1804) 


v_lshrrev_b32 v5, 7, v1 // (op 1469) 
s_add_i32 s18, s0, s16 // (op 1469) 

v_and_b32 v7, 63, v1 // (op 1469) 
s_ashr_i32 s0, s18, 31 // (op 1469) 
v_lshrrev_b32 v8, 4, v7 // (op 1469) 
s_cmp_eq_i32 s10, 0 // (op 1469) 
s_mov_b32 s19, scc // (op 1469) 
v_lshlrev_b32 v9, 2, v8 // (op 1469) 

s_cmp_eq_i32 s19, 0 // (op 1469) 
s_cselect_b32 s51, 0, -1 // (op 1469) 

v_lshl_add_u32 v8, v5, 4, v9 // (op 1469) 


s_and_b32 s19, s11, 31 // (op 1469) 
v_lshrrev_b32 v5, 2, v7 // (op 1469) 

v_and_b32 v7, 3, v5 // (op 1469) 

v_lshrrev_b32 v5, 6, v1 // (op 1583) 
v_and_b32 v9, 1, v5 // (op 1583) 

v_and_b32 v5, 15, v1 // (op 1583) 
v_lshl_add_u32 v10, v9, 4, v5 // (op 1583) 


v_and_b32 v5, 63, v1 // (op 1583) 
v_lshrrev_b32 v9, 4, v5 // (op 1583) 

v_lshlrev_b32 v5, 5, v9 // (op 1583) 

v_lshl_add_u32 v9, v10, 8, v5 // (op 1583) 


v_mul_lo_u32 v5, v9, 6 // (op 1583) 

v_lshrrev_b32 v9, 3, v5 // (op 1583) 

v_lshrrev_b32 v5, 6, v1 // (op 1694) 
v_and_b32 v10, 1, v5 // (op 1694) 

v_and_b32 v5, 15, v1 // (op 1694) 
v_lshl_add_u32 v11, v10, 4, v5 // (op 1694) 


v_and_b32 v5, 63, v1 // (op 1694) 
v_lshrrev_b32 v10, 4, v5 // (op 1694) 

v_lshl_add_u32 v5, v11, 3, v10 // (op 1694) 


v_lshrrev_b32 v10, 7, v1 // (op 1710) 
v_and_b32 v11, 63, v1 // (op 1710) 
v_lshrrev_b32 v12, 6, v1 // (op 1780) 
v_and_b32 v13, 1, v12 // (op 1780) 

v_and_b32 v12, 15, v1 // (op 1780) 
v_lshl_add_u32 v14, v13, 4, v12 // (op 1780) 


v_and_b32 v12, 63, v1 // (op 1780) 
v_lshrrev_b32 v13, 4, v12 // (op 1780) 

v_lshlrev_b32 v12, 5, v13 // (op 1780) 

v_lshl_add_u32 v13, v14, 8, v12 // (op 1780) 


v_mul_lo_u32 v12, v13, 6 // (op 1780) 

v_lshrrev_b32 v13, 3, v12 // (op 1780) 

v_lshrrev_b32 v12, 6, v1 // (op 1833) 
v_and_b32 v14, 1, v12 // (op 1833) 

v_and_b32 v12, 15, v1 // (op 1833) 
v_lshl_add_u32 v15, v14, 4, v12 // (op 1833) 


v_and_b32 v12, 63, v1 // (op 1833) 
v_lshrrev_b32 v14, 4, v12 // (op 1833) 

v_lshl_add_u32 v12, v15, 3, v14 // (op 1833) 


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

s_lshl_b32 s19, s11, 24 // (op 1469) 
v_lshl_add_u32 v7, v14, 4, v15 // (op 1469) 


s_ashr_i32 s51, s19, 31 // (op 1469) 

s_xor_b32 s19, s18, s51 // (op 1469) 

s_sub_i32 s18, s19, s51 // (op 1469) 


s_mul_hi_i32 s19, s18, s12 // (op 1469) 
s_add_i32 s51, s19, s18 // (op 1469) 

s_ashr_i32 s19, s51, 31 // (op 1469) 
s_cmp_eq_i32 s12, 0 // (op 1469) 
s_mov_b32 s52, scc // (op 1469) 
s_cmp_eq_i32 s52, 0 // (op 1469) 
s_cselect_b32 s53, 0, -1 // (op 1469) 

s_and_b32 s52, s13, 31 // (op 1469) 
v_lshrrev_b32 v14, 4, v11 // (op 1710) 
s_cmp_eq_i32 s10, 0 // (op 1710) 
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


s_lshl_b32 s19, s13, 24 // (op 1469) 
s_ashr_i32 s52, s19, 31 // (op 1469) 

s_xor_b32 s19, s51, s52 // (op 1469) 

s_sub_i32 s51, s19, s52 // (op 1469) 


s_mul_i32 s19, s18, s15 // (op 1469) 
s_sub_u32 s52, s16, s19 // (op 1469) 

s_mul_i32 s19, s51, s15 // (op 1469) 
s_add_u32 s19, s19, s52 // (op 1469) 

s_mul_hi_i32 s52, s16, s9 // (op 1469) 
s_add_i32 s53, s52, s16 // (op 1469) 

s_ashr_i32 s52, s53, 31 // (op 1469) 
s_cmp_eq_i32 s9, 0 // (op 1469) 
s_mov_b32 s54, scc // (op 1469) 
s_cmp_eq_i32 s54, 0 // (op 1469) 
s_cselect_b32 s56, 0, -1 // (op 1469) 

s_and_b32 s54, s11, 31 // (op 1710) 
v_and_b32 v11, 3, v10 // (op 1710) 

s_lshl_b32 s57, 1, s54 // (op 1710) 
v_and_b32 v10, 3, v1 // (op 1710) 
s_add_i32 s58, s57, s55 // (op 1710) 


s_and_b32 s55, s1, s58 // (op 1710) 


v_lshl_add_u32 v15, v11, 2, v10 // (op 1710) 


s_add_i32 s1, s0, s55 // (op 1710) 


v_lshrrev_b32 v10, 6, v1 // (op 1710) 
s_ashr_i32 s0, s1, s54 // (op 1710) 


v_and_b32 v11, 1, v10 // (op 1710) 

s_lshl_b32 s1, s11, 24 // (op 1710) 
v_lshl_add_u32 v10, v11, 4, v15 // (op 1710) 


s_ashr_i32 s54, s1, 31 // (op 1710) 

s_xor_b32 s1, s0, s54 // (op 1710) 

s_sub_i32 s0, s1, s54 // (op 1710) 


s_mul_hi_i32 s1, s0, s12 // (op 1710) 
s_add_i32 s54, s1, s0 // (op 1710) 

s_ashr_i32 s1, s54, 31 // (op 1710) 
s_cmp_eq_i32 s12, 0 // (op 1710) 
s_mov_b32 s55, scc // (op 1710) 
s_cmp_eq_i32 s55, 0 // (op 1710) 
s_cselect_b32 s57, 0, -1 // (op 1710) 

s_and_b32 s55, s13, 31 // (op 1710) 
s_lshl_b32 s58, 1, s55 // (op 1710) 
s_add_i32 s59, s58, s57 // (op 1710) 


s_and_b32 s57, s1, s59 // (op 1710) 


s_add_i32 s1, s54, s57 // (op 1710) 


s_ashr_i32 s54, s1, s55 // (op 1710) 


s_lshl_b32 s1, s13, 24 // (op 1710) 
s_ashr_i32 s55, s1, 31 // (op 1710) 

s_xor_b32 s1, s54, s55 // (op 1710) 

s_sub_i32 s54, s1, s55 // (op 1710) 


s_mul_i32 s1, s0, s15 // (op 1710) 
s_sub_u32 s55, s16, s1 // (op 1710) 

s_mul_i32 s1, s54, s15 // (op 1710) 
s_add_u32 s1, s1, s55 // (op 1710) 

s_mul_hi_i32 s55, s16, s9 // (op 1710) 
s_add_i32 s57, s55, s16 // (op 1710) 

s_ashr_i32 s55, s57, 31 // (op 1710) 
s_cmp_eq_i32 s9, 0 // (op 1710) 
s_mov_b32 s58, scc // (op 1710) 
s_cmp_eq_i32 s58, 0 // (op 1710) 
s_cselect_b32 s59, 0, -1 // (op 1710) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s46, 31 // (op 1469) 
s_lshl_b32 s60, 1, s58 // (op 1469) 
s_add_i32 s61, s60, s56 // (op 1469) 


s_and_b32 s56, s52, s61 // (op 1469) 


s_add_i32 s52, s53, s56 // (op 1469) 


s_ashr_i32 s53, s52, s58 // (op 1469) 


s_lshl_b32 s52, s46, 24 // (op 1469) 
s_ashr_i32 s56, s52, 31 // (op 1469) 

s_xor_b32 s52, s53, s56 // (op 1469) 

s_sub_i32 s53, s52, s56 // (op 1469) 


s_and_b32 s52, s46, 31 // (op 1710) 
s_lshl_b32 s56, 1, s52 // (op 1710) 
s_add_i32 s58, s56, s59 // (op 1710) 


s_and_b32 s56, s55, s58 // (op 1710) 


s_add_i32 s55, s57, s56 // (op 1710) 


s_ashr_i32 s56, s55, s52 // (op 1710) 


s_lshl_b32 s52, s46, 24 // (op 1710) 
s_ashr_i32 s55, s52, 31 // (op 1710) 

s_xor_b32 s52, s56, s55 // (op 1710) 

s_sub_i32 s56, s52, s55 // (op 1710) 


s_add_u32 s58, s6, 64 // (op 1469) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 1469) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1469) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1469) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 1469) 

s_mul_i32 s52, s15, s58 // (op 1469) 
s_mul_i32 s55, s8, s52 // (op 1469) 

s_mul_i32 s52, s53, s55 // (op 1469) 

s_sub_u32 s55, s16, s52 // (op 1469) 

s_mov_b32 s52, s55 // (op 1469) 
s_mul_hi_i32 s57, s52, s14 // (op 1469) 
s_add_i32 s60, s57, s52 // (op 1469) 


s_ashr_i32 s52, s60, 31 // (op 1469) 
s_cmp_eq_i32 s14, 0 // (op 1469) 
s_mov_b32 s57, scc // (op 1469) 
s_cmp_eq_i32 s57, 0 // (op 1469) 
s_cselect_b32 s61, 0, -1 // (op 1469) 

s_and_b32 s57, s48, 31 // (op 1469) 
s_lshl_b32 s62, 1, s57 // (op 1469) 
s_add_i32 s63, s62, s61 // (op 1469) 


s_and_b32 s61, s52, s63 // (op 1469) 


s_add_i32 s52, s60, s61 // (op 1469) 


s_ashr_i32 s60, s52, s57 // (op 1469) 


s_lshl_b32 s52, s48, 24 // (op 1469) 
s_ashr_i32 s57, s52, 31 // (op 1469) 

s_xor_b32 s52, s60, s57 // (op 1469) 

s_sub_i32 s60, s52, s57 // (op 1469) 


s_mul_i32 s52, s47, s15 // (op 1469) 
s_add_u32 s62, s20, 16 // (op 1469) Start of Int64 add, locking SCC
s_addc_u32 s63, s21, 0 // (op 1469) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1469) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1469) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 4 // (op 1469) 

s_sub_i32 s57, s62, s52 // (op 1469) 

s_mul_i32 s61, s60, s57 // (op 1469) 

s_sub_u32 s57, s55, s61 // (op 1469) 


s_add_u32 s55, s57, s52 // (op 1469) 


s_cmp_eq_i32 s53, 0 // (op 1469) 
s_mov_b32 s52, scc // (op 1469) 

s_cmp_eq_i32 s52, 0 // (op 1469) 
s_cselect_b32 s53, s55, s19 // (op 1469) 


s_mul_i32 s19, s51, s58 // (op 1469) 


v_lshl_add_u32 v11, s53, 4, v8 // (op 1469) 


s_sub_i32 s51, s18, s19 // (op 1469) 


s_cmp_eq_i32 s52, 0 // (op 1469) 
s_cselect_b32 s53, s60, s51 // (op 1469) 



v_lshl_add_u32 v8, s53, 6, v7 // (op 1469) 


v_mul_lo_u32 v7, v8, s2 // (op 1469) 

v_add_lshl_u32 v8, v11, v7, 1 // (op 1469) 


s_add_u32 s18, s6, 64 // (op 1710) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 1710) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 1710) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 1710) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 6 // (op 1710) 

s_mul_i32 s55, s15, s18 // (op 1710) 
s_mul_i32 s51, s8, s55 // (op 1710) 

s_mul_i32 s55, s56, s51 // (op 1710) 

s_sub_u32 s51, s16, s55 // (op 1710) 

s_mov_b32 s55, s51 // (op 1710) 
s_mul_hi_i32 s52, s55, s14 // (op 1710) 
s_add_i32 s53, s52, s55 // (op 1710) 


s_ashr_i32 s52, s53, 31 // (op 1710) 
s_cmp_eq_i32 s14, 0 // (op 1710) 
s_mov_b32 s55, scc // (op 1710) 
s_cmp_eq_i32 s55, 0 // (op 1710) 
s_cselect_b32 s57, 0, -1 // (op 1710) 

s_and_b32 s55, s48, 31 // (op 1710) 
s_lshl_b32 s58, 1, s55 // (op 1710) 
s_add_i32 s59, s58, s57 // (op 1710) 


s_and_b32 s57, s52, s59 // (op 1710) 


s_add_i32 s52, s53, s57 // (op 1710) 


s_ashr_i32 s53, s52, s55 // (op 1710) 


s_lshl_b32 s52, s48, 24 // (op 1710) 
s_ashr_i32 s55, s52, 31 // (op 1710) 

s_xor_b32 s52, s53, s55 // (op 1710) 

s_sub_i32 s53, s52, s55 // (op 1710) 


s_mul_i32 s52, s47, s15 // (op 1710) 
s_add_u32 s58, s20, 16 // (op 1710) Start of Int64 add, locking SCC
s_addc_u32 s59, s21, 0 // (op 1710) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1710) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1710) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 4 // (op 1710) 

s_sub_i32 s55, s58, s52 // (op 1710) 

s_mul_i32 s57, s53, s55 // (op 1710) 

s_sub_u32 s55, s51, s57 // (op 1710) 


s_add_u32 s51, s55, s52 // (op 1710) 


s_cmp_eq_i32 s56, 0 // (op 1710) 
s_mov_b32 s52, scc // (op 1710) 

s_cmp_eq_i32 s52, 0 // (op 1710) 
s_cselect_b32 s55, s51, s1 // (op 1710) 


s_mul_i32 s1, s54, s18 // (op 1710) 


v_lshl_add_u32 v7, s55, 4, v14 // (op 1710) 


s_sub_i32 s51, s0, s1 // (op 1710) 


s_cmp_eq_i32 s52, 0 // (op 1710) 
s_cselect_b32 s0, s53, s51 // (op 1710) 



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

s_mov_b32 s56, s22 // (op 1471) reg expression
s_mov_b32 s57, s23 // (op 1471) reg expression
s_mov_b32 s58, s18 // (op 1471) reg expression
s_mov_b32 s59, 131072 // (op 1471) call()

s_mov_b32 s60, s24 // (op 1712) reg expression

s_mov_b32 s61, s25 // (op 1712) reg expression
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
s_mul_hi_i32 s0, s16, s10 // (op 1236) 
s_add_i32 s1, s0, s16 // (op 1236) 

s_ashr_i32 s0, s1, 31 // (op 1236) 
s_cmp_eq_i32 s10, 0 // (op 1236) 
s_mov_b32 s18, scc // (op 1236) 
s_cmp_eq_i32 s18, 0 // (op 1236) 
s_cselect_b32 s19, 0, -1 // (op 1236) 

s_and_b32 s18, s11, 31 // (op 1236) 
s_lshl_b32 s22, 1, s18 // (op 1236) 
s_add_i32 s23, s22, s19 // (op 1236) 


s_and_b32 s19, s0, s23 // (op 1236) 


s_add_i32 s0, s1, s19 // (op 1236) 


s_ashr_i32 s1, s0, s18 // (op 1236) 


s_lshl_b32 s0, s11, 24 // (op 1236) 
s_ashr_i32 s18, s0, 31 // (op 1236) 

s_xor_b32 s0, s1, s18 // (op 1236) 

s_sub_i32 s1, s0, s18 // (op 1236) 


s_mul_hi_i32 s0, s1, s12 // (op 1236) 
s_add_i32 s18, s0, s1 // (op 1236) 

s_ashr_i32 s0, s18, 31 // (op 1236) 
s_cmp_eq_i32 s12, 0 // (op 1236) 
s_mov_b32 s19, scc // (op 1236) 
s_cmp_eq_i32 s19, 0 // (op 1236) 
s_cselect_b32 s22, 0, -1 // (op 1236) 

s_and_b32 s19, s13, 31 // (op 1236) 
s_lshl_b32 s23, 1, s19 // (op 1236) 
s_add_i32 s24, s23, s22 // (op 1236) 


s_and_b32 s25, s0, s24 // (op 1236) 


s_add_i32 s0, s18, s25 // (op 1236) 


s_ashr_i32 s18, s0, s19 // (op 1236) 


s_lshl_b32 s0, s13, 24 // (op 1236) 
s_ashr_i32 s19, s0, 31 // (op 1236) 

s_xor_b32 s0, s18, s19 // (op 1236) 

s_sub_i32 s18, s0, s19 // (op 1236) 


s_mul_i32 s0, s1, s15 // (op 1236) 

s_sub_u32 s1, s16, s0 // (op 1236) 

s_mul_i32 s0, s18, s15 // (op 1236) 
s_add_u32 s0, s0, s1 // (op 1236) 


s_mul_hi_i32 s1, s16, s9 // (op 1236) 
s_add_i32 s18, s1, s16 // (op 1236) 

s_ashr_i32 s1, s18, 31 // (op 1236) 
s_cmp_eq_i32 s9, 0 // (op 1236) 
s_mov_b32 s19, scc // (op 1236) 
s_cmp_eq_i32 s19, 0 // (op 1236) 
s_cselect_b32 s22, 0, -1 // (op 1236) 

s_and_b32 s19, s46, 31 // (op 1236) 
s_lshl_b32 s23, 1, s19 // (op 1236) 
s_add_i32 s24, s23, s22 // (op 1236) 


s_and_b32 s25, s1, s24 // (op 1236) 


s_add_i32 s1, s18, s25 // (op 1236) 


s_ashr_i32 s18, s1, s19 // (op 1236) 


s_lshl_b32 s1, s46, 24 // (op 1236) 
s_ashr_i32 s19, s1, 31 // (op 1236) 

s_xor_b32 s1, s18, s19 // (op 1236) 

s_sub_i32 s18, s1, s19 // (op 1236) 


s_add_u32 s22, s6, 64 // (op 1236) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1236) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1236) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1236) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1236) 

s_mul_i32 s1, s15, s22 // (op 1236) 

s_mul_i32 s19, s8, s1 // (op 1236) 

s_mul_i32 s1, s18, s19 // (op 1236) 

s_sub_u32 s19, s16, s1 // (op 1236) 

s_mov_b32 s1, s19 // (op 1236) 
s_mul_hi_i32 s22, s1, s14 // (op 1236) 
s_add_i32 s23, s22, s1 // (op 1236) 


s_ashr_i32 s1, s23, 31 // (op 1236) 
s_cmp_eq_i32 s14, 0 // (op 1236) 
s_mov_b32 s22, scc // (op 1236) 
s_cmp_eq_i32 s22, 0 // (op 1236) 
s_cselect_b32 s24, 0, -1 // (op 1236) 

s_and_b32 s22, s48, 31 // (op 1236) 
s_lshl_b32 s25, 1, s22 // (op 1236) 
s_add_i32 s49, s25, s24 // (op 1236) 


s_and_b32 s24, s1, s49 // (op 1236) 


s_add_i32 s1, s23, s24 // (op 1236) 


s_ashr_i32 s23, s1, s22 // (op 1236) 


s_lshl_b32 s1, s48, 24 // (op 1236) 
s_ashr_i32 s22, s1, 31 // (op 1236) 

s_xor_b32 s1, s23, s22 // (op 1236) 

s_sub_i32 s23, s1, s22 // (op 1236) 


s_mul_i32 s1, s47, s15 // (op 1236) 
s_add_u32 s24, s20, 16 // (op 1236) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1236) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 1236) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 1236) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 4 // (op 1236) 

s_sub_i32 s22, s24, s1 // (op 1236) 

s_mul_i32 s24, s23, s22 // (op 1236) 


s_sub_u32 s25, s19, s24 // (op 1236) 


s_add_u32 s19, s25, s1 // (op 1236) 


s_cmp_eq_i32 s18, 0 // (op 1236) 
s_mov_b32 s1, scc // (op 1236) 

s_cmp_eq_i32 s1, 0 // (op 1236) 
s_cselect_b32 s18, s19, s0 // (op 1236) 



s_lshl_b32 s19, s18, 4 // (op 1236) 

s_mov_b32 s1, 0 // (op 1236) convert
s_mov_b32 s0, s19 // (op 1236) convert

v_mul_lo_u32 v15, v1, s27 // (op 1236) most significant: low of low * high
v_mul_hi_u32 v16, v1, s26 // (op 1236) most significant: high of low * low
v_mul_lo_u32 v14, v1, s26 // (op 1236) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 1236) most significant: sum

v_mov_b32 v16, s0 // (op 1236) 
v_mov_b32 v17, s1 // (op 1236) 
v_add_co_u32 v18, s[18:19], v16, v14 // (op 1236) least significant half
v_addc_co_u32 v19, s[18:19], v17, v15, s[18:19] // (op 1236) most significant half





s_mul_i32 s1, s26, s31 // (op 1238) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 1238) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 1238) least significant
s_mul_hi_u32 s19, s26, s30 // (op 1238) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1238) most significant: sum
s_add_u32 s1, s1, s19 // (op 1238) most significant: sum


s_mov_b32 s52, s28 // (op 1238) reg expression
s_mov_b32 s53, s29 // (op 1238) reg expression
s_mov_b32 s54, s0 // (op 1238) reg expression
s_mov_b32 s55, 131072 // (op 1238) call()

s_mul_hi_i32 s0, s16, s10 // (op 1255) 
s_add_i32 s1, s0, s16 // (op 1255) 

s_ashr_i32 s0, s1, 31 // (op 1255) 
s_cmp_eq_i32 s10, 0 // (op 1255) 
s_mov_b32 s18, scc // (op 1255) 
s_cmp_eq_i32 s18, 0 // (op 1255) 
s_cselect_b32 s19, 0, -1 // (op 1255) 

s_and_b32 s18, s11, 31 // (op 1255) 
s_lshl_b32 s22, 1, s18 // (op 1255) 
s_add_i32 s23, s22, s19 // (op 1255) 


s_and_b32 s19, s0, s23 // (op 1255) 


s_add_i32 s0, s1, s19 // (op 1255) 


s_ashr_i32 s1, s0, s18 // (op 1255) 


s_lshl_b32 s0, s11, 24 // (op 1255) 
s_ashr_i32 s18, s0, 31 // (op 1255) 

s_xor_b32 s0, s1, s18 // (op 1255) 

s_sub_i32 s1, s0, s18 // (op 1255) 


s_mul_hi_i32 s0, s1, s12 // (op 1255) 
s_add_i32 s18, s0, s1 // (op 1255) 

s_ashr_i32 s0, s18, 31 // (op 1255) 
s_cmp_eq_i32 s12, 0 // (op 1255) 
s_mov_b32 s19, scc // (op 1255) 
s_cmp_eq_i32 s19, 0 // (op 1255) 
s_cselect_b32 s22, 0, -1 // (op 1255) 

s_and_b32 s19, s13, 31 // (op 1255) 
s_lshl_b32 s23, 1, s19 // (op 1255) 
s_add_i32 s24, s23, s22 // (op 1255) 


s_and_b32 s25, s0, s24 // (op 1255) 


s_add_i32 s0, s18, s25 // (op 1255) 


s_ashr_i32 s18, s0, s19 // (op 1255) 


s_lshl_b32 s0, s13, 24 // (op 1255) 
s_ashr_i32 s19, s0, 31 // (op 1255) 

s_xor_b32 s0, s18, s19 // (op 1255) 

s_sub_i32 s18, s0, s19 // (op 1255) 


s_mul_i32 s0, s1, s15 // (op 1255) 

s_sub_u32 s1, s16, s0 // (op 1255) 

s_mul_i32 s0, s18, s15 // (op 1255) 
s_add_u32 s0, s0, s1 // (op 1255) 


s_mul_hi_i32 s1, s16, s9 // (op 1255) 
s_add_i32 s18, s1, s16 // (op 1255) 

s_ashr_i32 s1, s18, 31 // (op 1255) 
s_cmp_eq_i32 s9, 0 // (op 1255) 
s_mov_b32 s19, scc // (op 1255) 
s_cmp_eq_i32 s19, 0 // (op 1255) 
s_cselect_b32 s22, 0, -1 // (op 1255) 

s_and_b32 s19, s46, 31 // (op 1255) 
s_lshl_b32 s23, 1, s19 // (op 1255) 
s_add_i32 s24, s23, s22 // (op 1255) 


s_and_b32 s25, s1, s24 // (op 1255) 


s_add_i32 s1, s18, s25 // (op 1255) 


s_ashr_i32 s18, s1, s19 // (op 1255) 


s_lshl_b32 s1, s46, 24 // (op 1255) 
s_ashr_i32 s19, s1, 31 // (op 1255) 

s_xor_b32 s1, s18, s19 // (op 1255) 

s_sub_i32 s18, s1, s19 // (op 1255) 


s_add_u32 s50, s6, 64 // (op 1255) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1255) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1255) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1255) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1255) 

s_mul_i32 s1, s15, s24 // (op 1255) 

s_mul_i32 s19, s8, s1 // (op 1255) 

s_mul_i32 s1, s18, s19 // (op 1255) 

s_sub_u32 s19, s16, s1 // (op 1255) 

s_mov_b32 s1, s19 // (op 1255) 
s_mul_hi_i32 s22, s1, s14 // (op 1255) 
s_add_i32 s23, s22, s1 // (op 1255) 


s_ashr_i32 s1, s23, 31 // (op 1255) 
s_cmp_eq_i32 s14, 0 // (op 1255) 
s_mov_b32 s22, scc // (op 1255) 
s_cmp_eq_i32 s22, 0 // (op 1255) 
s_cselect_b32 s24, 0, -1 // (op 1255) 

s_and_b32 s22, s48, 31 // (op 1255) 
s_lshl_b32 s25, 1, s22 // (op 1255) 
s_add_i32 s49, s25, s24 // (op 1255) 


s_and_b32 s24, s1, s49 // (op 1255) 


s_add_i32 s1, s23, s24 // (op 1255) 


s_ashr_i32 s23, s1, s22 // (op 1255) 


s_lshl_b32 s1, s48, 24 // (op 1255) 
s_ashr_i32 s22, s1, 31 // (op 1255) 

s_xor_b32 s1, s23, s22 // (op 1255) 

s_sub_i32 s23, s1, s22 // (op 1255) 


s_mul_i32 s1, s47, s15 // (op 1255) 
s_add_u32 s24, s20, 16 // (op 1255) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1255) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1255) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1255) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1255) 

s_sub_i32 s22, s24, s1 // (op 1255) 

s_mul_i32 s24, s23, s22 // (op 1255) 


s_sub_u32 s25, s19, s24 // (op 1255) 


s_add_u32 s19, s25, s1 // (op 1255) 


s_cmp_eq_i32 s18, 0 // (op 1255) 
s_mov_b32 s1, scc // (op 1255) 

s_cmp_eq_i32 s1, 0 // (op 1255) 
s_cselect_b32 s18, s19, s0 // (op 1255) 



s_lshl_b32 s19, s18, 4 // (op 1255) 

s_mov_b32 s1, 0 // (op 1255) convert
s_mov_b32 s0, s19 // (op 1255) convert

v_mul_lo_u32 v15, v1, s27 // (op 1255) most significant: low of low * high
v_mul_hi_u32 v16, v1, s26 // (op 1255) most significant: high of low * low
v_mul_lo_u32 v14, v1, s26 // (op 1255) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 1255) most significant: sum

v_mov_b32 v16, s0 // (op 1255) 
v_mov_b32 v17, s1 // (op 1255) 
v_add_co_u32 v20, s[18:19], v16, v14 // (op 1255) least significant half
v_addc_co_u32 v21, s[18:19], v17, v15, s[18:19] // (op 1255) most significant half





s_mul_i32 s1, s26, s31 // (op 1257) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 1257) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 1257) least significant
s_mul_hi_u32 s19, s26, s30 // (op 1257) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1257) most significant: sum
s_add_u32 s1, s1, s19 // (op 1257) most significant: sum


s_mov_b32 s52, s28 // (op 1257) reg expression
s_mov_b32 s53, s29 // (op 1257) reg expression
s_mov_b32 s54, s0 // (op 1257) reg expression
s_mov_b32 s55, 131072 // (op 1257) call()

s_mul_hi_i32 s0, s16, s10 // (op 1300) 
v_lshrrev_b32 v14, 2, v1 // (op 1300) 
s_add_i32 s1, s0, s16 // (op 1300) 

v_lshlrev_b32 v15, 1, v14 // (op 1300) 

s_ashr_i32 s0, s1, 31 // (op 1300) 
v_and_b32 v14, -4, v1 // (op 1300) 
s_cmp_eq_i32 s10, 0 // (op 1300) 
s_mov_b32 s18, scc // (op 1300) 
v_sub_u32 v16, v1, v14 // (op 1300) 

s_cmp_eq_i32 s18, 0 // (op 1300) 
s_cselect_b32 s19, 0, -1 // (op 1300) 

v_lshlrev_b32 v14, 6, v16 // (op 1300) 

v_mov_b32 v17, 0 // (op 1300) convert
v_mov_b32 v16, v14 // (op 1300) convert

s_and_b32 s18, s11, 31 // (op 1300) 
s_lshl_b32 s22, 1, s18 // (op 1300) 
s_add_i32 s23, s22, s19 // (op 1300) 


s_and_b32 s19, s0, s23 // (op 1300) 


s_add_i32 s0, s1, s19 // (op 1300) 


s_ashr_i32 s1, s0, s18 // (op 1300) 


s_lshl_b32 s0, s11, 24 // (op 1300) 
s_ashr_i32 s18, s0, 31 // (op 1300) 

s_xor_b32 s0, s1, s18 // (op 1300) 

s_sub_i32 s1, s0, s18 // (op 1300) 


s_mul_hi_i32 s0, s1, s12 // (op 1300) 
s_add_i32 s18, s0, s1 // (op 1300) 

s_ashr_i32 s0, s18, 31 // (op 1300) 
s_cmp_eq_i32 s12, 0 // (op 1300) 
s_mov_b32 s19, scc // (op 1300) 
s_cmp_eq_i32 s19, 0 // (op 1300) 
s_cselect_b32 s22, 0, -1 // (op 1300) 

s_and_b32 s19, s13, 31 // (op 1300) 
s_lshl_b32 s23, 1, s19 // (op 1300) 
s_add_i32 s24, s23, s22 // (op 1300) 


s_and_b32 s25, s0, s24 // (op 1300) 


s_add_i32 s0, s18, s25 // (op 1300) 


s_ashr_i32 s18, s0, s19 // (op 1300) 


s_lshl_b32 s0, s13, 24 // (op 1300) 
s_ashr_i32 s19, s0, 31 // (op 1300) 

s_xor_b32 s0, s18, s19 // (op 1300) 

s_sub_i32 s18, s0, s19 // (op 1300) 


s_add_u32 s50, s6, 64 // (op 1300) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1300) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1300) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1300) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1300) 

s_mul_i32 s0, s18, s24 // (op 1300) 

s_sub_i32 s18, s1, s0 // (op 1300) 


s_mul_hi_i32 s19, s16, s9 // (op 1300) 
s_add_i32 s0, s19, s16 // (op 1300) 

s_ashr_i32 s1, s0, 31 // (op 1300) 
s_cmp_eq_i32 s9, 0 // (op 1300) 
s_mov_b32 s19, scc // (op 1300) 
s_cmp_eq_i32 s19, 0 // (op 1300) 
s_cselect_b32 s22, 0, -1 // (op 1300) 

s_and_b32 s19, s46, 31 // (op 1300) 
s_lshl_b32 s23, 1, s19 // (op 1300) 
s_add_i32 s49, s23, s22 // (op 1300) 


s_and_b32 s22, s1, s49 // (op 1300) 


s_add_i32 s1, s0, s22 // (op 1300) 


s_ashr_i32 s0, s1, s19 // (op 1300) 


s_lshl_b32 s1, s46, 24 // (op 1300) 
s_ashr_i32 s19, s1, 31 // (op 1300) 

s_xor_b32 s1, s0, s19 // (op 1300) 

s_sub_i32 s0, s1, s19 // (op 1300) 


s_mul_i32 s1, s15, s24 // (op 1300) 

s_mul_i32 s19, s8, s1 // (op 1300) 

s_mul_i32 s1, s0, s19 // (op 1300) 

s_sub_u32 s19, s16, s1 // (op 1300) 

s_mul_hi_i32 s1, s19, s14 // (op 1300) 
s_add_i32 s22, s1, s19 // (op 1300) 


s_ashr_i32 s1, s22, 31 // (op 1300) 
s_cmp_eq_i32 s14, 0 // (op 1300) 
s_mov_b32 s19, scc // (op 1300) 
s_cmp_eq_i32 s19, 0 // (op 1300) 
s_cselect_b32 s23, 0, -1 // (op 1300) 

s_and_b32 s19, s48, 31 // (op 1300) 
s_lshl_b32 s24, 1, s19 // (op 1300) 
s_add_i32 s25, s24, s23 // (op 1300) 


s_and_b32 s23, s1, s25 // (op 1300) 


s_add_i32 s1, s22, s23 // (op 1300) 


s_ashr_i32 s22, s1, s19 // (op 1300) 


s_lshl_b32 s1, s48, 24 // (op 1300) 
s_ashr_i32 s19, s1, 31 // (op 1300) 

s_xor_b32 s1, s22, s19 // (op 1300) 

s_sub_i32 s22, s1, s19 // (op 1300) 


s_cmp_eq_i32 s0, 0 // (op 1300) 
s_mov_b32 s1, scc // (op 1300) 

s_cmp_eq_i32 s1, 0 // (op 1300) 
s_cselect_b32 s0, s22, s18 // (op 1300) 



v_lshl_add_u32 v14, s0, 6, v15 // (op 1300) 


v_mul_lo_u32 v23, v14, s33 // (op 1300) most significant: low of low * high
v_mul_hi_u32 v15, v14, s32 // (op 1300) most significant: high of low * low
v_mul_lo_u32 v22, v14, s32 // (op 1300) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 1300) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v22 // (op 1300) least significant half
v_addc_co_u32 v15, s[0:1], v17, v23, s[0:1] // (op 1300) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 1300) most significant: low of high * low
v_mul_hi_u32 v22, v14, 6 // (op 1300) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 1300) least significant: low of low * low
v_add_u32 v17, v17, v22 // (op 1300) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 1300) 

v_mov_b32 v16, v14 // (op 1300) convert

s_mul_i32 s1, s7, 6 // (op 1302) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 1302) least significant
s_mul_hi_u32 s18, s6, 6 // (op 1302) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1302) most significant: sum

s_mul_i32 s19, s32, s1 // (op 1302) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 1302) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 1302) least significant
s_mul_hi_u32 s23, s32, s0 // (op 1302) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 1302) most significant: sum
s_add_u32 s19, s19, s23 // (op 1302) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1302) 

s_mov_b32 s64, s34 // (op 1302) reg expression
s_mov_b32 s65, s35 // (op 1302) reg expression
s_mov_b32 s66, s0 // (op 1302) reg expression
s_mov_b32 s67, 131072 // (op 1302) call()

s_mul_hi_i32 s0, s16, s10 // (op 1319) 
v_lshrrev_b32 v17, 2, v1 // (op 1319) 
s_add_i32 s1, s0, s16 // (op 1319) 

v_lshlrev_b32 v14, 1, v17 // (op 1319) 

s_ashr_i32 s0, s1, 31 // (op 1319) 
v_and_b32 v15, -4, v1 // (op 1319) 
s_cmp_eq_i32 s10, 0 // (op 1319) 
s_mov_b32 s18, scc // (op 1319) 
v_sub_u32 v17, v1, v15 // (op 1319) 

s_cmp_eq_i32 s18, 0 // (op 1319) 
s_cselect_b32 s19, 0, -1 // (op 1319) 

v_lshlrev_b32 v15, 6, v17 // (op 1319) 

v_mov_b32 v23, 0 // (op 1319) convert
v_mov_b32 v22, v15 // (op 1319) convert

s_and_b32 s18, s11, 31 // (op 1319) 
s_lshl_b32 s22, 1, s18 // (op 1319) 
s_add_i32 s23, s22, s19 // (op 1319) 


s_and_b32 s19, s0, s23 // (op 1319) 


s_add_i32 s0, s1, s19 // (op 1319) 


s_ashr_i32 s1, s0, s18 // (op 1319) 


s_lshl_b32 s0, s11, 24 // (op 1319) 
s_ashr_i32 s18, s0, 31 // (op 1319) 

s_xor_b32 s0, s1, s18 // (op 1319) 

s_sub_i32 s1, s0, s18 // (op 1319) 


s_mul_hi_i32 s0, s1, s12 // (op 1319) 
s_add_i32 s18, s0, s1 // (op 1319) 

s_ashr_i32 s0, s18, 31 // (op 1319) 
s_cmp_eq_i32 s12, 0 // (op 1319) 
s_mov_b32 s19, scc // (op 1319) 
s_cmp_eq_i32 s19, 0 // (op 1319) 
s_cselect_b32 s22, 0, -1 // (op 1319) 

s_and_b32 s19, s13, 31 // (op 1319) 
s_lshl_b32 s23, 1, s19 // (op 1319) 
s_add_i32 s24, s23, s22 // (op 1319) 


s_and_b32 s25, s0, s24 // (op 1319) 


s_add_i32 s0, s18, s25 // (op 1319) 


s_ashr_i32 s18, s0, s19 // (op 1319) 


s_lshl_b32 s0, s13, 24 // (op 1319) 
s_ashr_i32 s19, s0, 31 // (op 1319) 

s_xor_b32 s0, s18, s19 // (op 1319) 

s_sub_i32 s18, s0, s19 // (op 1319) 


s_add_u32 s50, s6, 64 // (op 1319) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1319) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1319) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1319) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1319) 

s_mul_i32 s0, s18, s24 // (op 1319) 

s_sub_i32 s18, s1, s0 // (op 1319) 


s_mul_hi_i32 s19, s16, s9 // (op 1319) 
s_add_i32 s0, s19, s16 // (op 1319) 

s_ashr_i32 s1, s0, 31 // (op 1319) 
s_cmp_eq_i32 s9, 0 // (op 1319) 
s_mov_b32 s19, scc // (op 1319) 
s_cmp_eq_i32 s19, 0 // (op 1319) 
s_cselect_b32 s22, 0, -1 // (op 1319) 

s_and_b32 s19, s46, 31 // (op 1319) 
s_lshl_b32 s23, 1, s19 // (op 1319) 
s_add_i32 s49, s23, s22 // (op 1319) 


s_and_b32 s22, s1, s49 // (op 1319) 


s_add_i32 s1, s0, s22 // (op 1319) 


s_ashr_i32 s0, s1, s19 // (op 1319) 


s_lshl_b32 s1, s46, 24 // (op 1319) 
s_ashr_i32 s19, s1, 31 // (op 1319) 

s_xor_b32 s1, s0, s19 // (op 1319) 

s_sub_i32 s0, s1, s19 // (op 1319) 


s_mul_i32 s1, s15, s24 // (op 1319) 

s_mul_i32 s19, s8, s1 // (op 1319) 

s_mul_i32 s1, s0, s19 // (op 1319) 

s_sub_u32 s19, s16, s1 // (op 1319) 

s_mul_hi_i32 s1, s19, s14 // (op 1319) 
s_add_i32 s22, s1, s19 // (op 1319) 


s_ashr_i32 s1, s22, 31 // (op 1319) 
s_cmp_eq_i32 s14, 0 // (op 1319) 
s_mov_b32 s19, scc // (op 1319) 
s_cmp_eq_i32 s19, 0 // (op 1319) 
s_cselect_b32 s23, 0, -1 // (op 1319) 

s_and_b32 s19, s48, 31 // (op 1319) 
s_lshl_b32 s24, 1, s19 // (op 1319) 
s_add_i32 s25, s24, s23 // (op 1319) 


s_and_b32 s23, s1, s25 // (op 1319) 


s_add_i32 s1, s22, s23 // (op 1319) 


s_ashr_i32 s22, s1, s19 // (op 1319) 


s_lshl_b32 s1, s48, 24 // (op 1319) 
s_ashr_i32 s19, s1, 31 // (op 1319) 

s_xor_b32 s1, s22, s19 // (op 1319) 

s_sub_i32 s22, s1, s19 // (op 1319) 


s_cmp_eq_i32 s0, 0 // (op 1319) 
s_mov_b32 s1, scc // (op 1319) 

s_cmp_eq_i32 s1, 0 // (op 1319) 
s_cselect_b32 s0, s22, s18 // (op 1319) 



v_lshl_add_u32 v15, s0, 6, v14 // (op 1319) 


v_mul_lo_u32 v25, v15, s33 // (op 1319) most significant: low of low * high
v_mul_hi_u32 v14, v15, s32 // (op 1319) most significant: high of low * low
v_mul_lo_u32 v24, v15, s32 // (op 1319) least significant: low of low * low
v_add_u32 v25, v25, v14 // (op 1319) most significant: sum


v_add_co_u32 v14, s[0:1], v22, v24 // (op 1319) least significant half
v_addc_co_u32 v15, s[0:1], v23, v25, s[0:1] // (op 1319) most significant half



v_mul_lo_u32 v23, v15, 6 // (op 1319) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 1319) most significant: high of low * low
v_mul_lo_u32 v22, v14, 6 // (op 1319) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 1319) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[22:23] // (op 1319) 

v_mov_b32 v17, v14 // (op 1319) convert

s_mul_i32 s1, s7, 6 // (op 1321) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 1321) least significant
s_mul_hi_u32 s18, s6, 6 // (op 1321) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1321) most significant: sum

s_mul_i32 s19, s32, s1 // (op 1321) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 1321) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 1321) least significant
s_mul_hi_u32 s23, s32, s0 // (op 1321) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 1321) most significant: sum
s_add_u32 s19, s19, s23 // (op 1321) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1321) 

s_mov_b32 s64, s34 // (op 1321) reg expression
s_mov_b32 s65, s35 // (op 1321) reg expression
s_mov_b32 s66, s0 // (op 1321) reg expression
s_mov_b32 s67, 131072 // (op 1321) call()

s_mul_hi_i32 s0, s16, s10 // (op 1364) 
v_and_b32 v14, -16, v1 // (op 1364) 
s_add_i32 s1, s0, s16 // (op 1364) 

v_sub_u32 v15, v1, v14 // (op 1364) 

s_ashr_i32 s0, s1, 31 // (op 1364) 
v_lshrrev_b32 v14, 4, v1 // (op 1364) 
s_cmp_eq_i32 s10, 0 // (op 1364) 
s_mov_b32 s18, scc // (op 1364) 
s_cmp_eq_i32 s18, 0 // (op 1364) 
s_cselect_b32 s19, 0, -1 // (op 1364) 

s_and_b32 s18, s11, 31 // (op 1364) 
s_lshl_b32 s22, 1, s18 // (op 1364) 
s_add_i32 s23, s22, s19 // (op 1364) 


s_and_b32 s19, s0, s23 // (op 1364) 


s_add_i32 s0, s1, s19 // (op 1364) 


s_ashr_i32 s1, s0, s18 // (op 1364) 


s_lshl_b32 s0, s11, 24 // (op 1364) 
s_ashr_i32 s18, s0, 31 // (op 1364) 

s_xor_b32 s0, s1, s18 // (op 1364) 

s_sub_i32 s1, s0, s18 // (op 1364) 


s_mul_hi_i32 s0, s1, s12 // (op 1364) 
s_add_i32 s18, s0, s1 // (op 1364) 

s_ashr_i32 s0, s18, 31 // (op 1364) 
s_cmp_eq_i32 s12, 0 // (op 1364) 
s_mov_b32 s19, scc // (op 1364) 
s_cmp_eq_i32 s19, 0 // (op 1364) 
s_cselect_b32 s22, 0, -1 // (op 1364) 

s_and_b32 s19, s13, 31 // (op 1364) 
s_lshl_b32 s23, 1, s19 // (op 1364) 
s_add_i32 s24, s23, s22 // (op 1364) 


s_and_b32 s25, s0, s24 // (op 1364) 


s_add_i32 s0, s18, s25 // (op 1364) 


s_ashr_i32 s18, s0, s19 // (op 1364) 


s_lshl_b32 s0, s13, 24 // (op 1364) 
s_ashr_i32 s19, s0, 31 // (op 1364) 

s_xor_b32 s0, s18, s19 // (op 1364) 

s_sub_i32 s18, s0, s19 // (op 1364) 


s_mul_i32 s0, s1, s15 // (op 1364) 

s_sub_u32 s1, s16, s0 // (op 1364) 

s_mul_i32 s0, s18, s15 // (op 1364) 
s_add_u32 s0, s0, s1 // (op 1364) 


s_mul_hi_i32 s1, s16, s9 // (op 1364) 
s_add_i32 s18, s1, s16 // (op 1364) 

s_ashr_i32 s1, s18, 31 // (op 1364) 
s_cmp_eq_i32 s9, 0 // (op 1364) 
s_mov_b32 s19, scc // (op 1364) 
s_cmp_eq_i32 s19, 0 // (op 1364) 
s_cselect_b32 s22, 0, -1 // (op 1364) 

s_and_b32 s19, s46, 31 // (op 1364) 
s_lshl_b32 s23, 1, s19 // (op 1364) 
s_add_i32 s24, s23, s22 // (op 1364) 


s_and_b32 s25, s1, s24 // (op 1364) 


s_add_i32 s1, s18, s25 // (op 1364) 


s_ashr_i32 s18, s1, s19 // (op 1364) 


s_lshl_b32 s1, s46, 24 // (op 1364) 
s_ashr_i32 s19, s1, 31 // (op 1364) 

s_xor_b32 s1, s18, s19 // (op 1364) 

s_sub_i32 s18, s1, s19 // (op 1364) 


s_add_u32 s50, s6, 64 // (op 1364) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1364) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1364) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1364) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1364) 

s_mul_i32 s1, s15, s24 // (op 1364) 

s_mul_i32 s19, s8, s1 // (op 1364) 

s_mul_i32 s1, s18, s19 // (op 1364) 

s_sub_u32 s19, s16, s1 // (op 1364) 

s_mov_b32 s1, s19 // (op 1364) 
s_mul_hi_i32 s22, s1, s14 // (op 1364) 
s_add_i32 s23, s22, s1 // (op 1364) 


s_ashr_i32 s1, s23, 31 // (op 1364) 
s_cmp_eq_i32 s14, 0 // (op 1364) 
s_mov_b32 s22, scc // (op 1364) 
s_cmp_eq_i32 s22, 0 // (op 1364) 
s_cselect_b32 s24, 0, -1 // (op 1364) 

s_and_b32 s22, s48, 31 // (op 1364) 
s_lshl_b32 s25, 1, s22 // (op 1364) 
s_add_i32 s49, s25, s24 // (op 1364) 


s_and_b32 s24, s1, s49 // (op 1364) 


s_add_i32 s1, s23, s24 // (op 1364) 


s_ashr_i32 s23, s1, s22 // (op 1364) 


s_lshl_b32 s1, s48, 24 // (op 1364) 
s_ashr_i32 s22, s1, 31 // (op 1364) 

s_xor_b32 s1, s23, s22 // (op 1364) 

s_sub_i32 s23, s1, s22 // (op 1364) 


s_mul_i32 s1, s47, s15 // (op 1364) 
s_add_u32 s24, s20, 16 // (op 1364) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1364) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1364) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1364) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1364) 

s_sub_i32 s22, s24, s1 // (op 1364) 

s_mul_i32 s24, s23, s22 // (op 1364) 


s_sub_u32 s25, s19, s24 // (op 1364) 


s_add_u32 s19, s25, s1 // (op 1364) 


s_cmp_eq_i32 s18, 0 // (op 1364) 
s_mov_b32 s1, scc // (op 1364) 

s_cmp_eq_i32 s1, 0 // (op 1364) 
s_cselect_b32 s18, s19, s0 // (op 1364) 



v_lshl_add_u32 v22, s18, 4, v15 // (op 1364) 


v_mul_lo_u32 v15, v14, s36 // (op 1364) 

v_add_u32 v14, v22, v15 // (op 1364) 


s_mul_i32 s1, s36, s41 // (op 1366) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1366) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1366) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1366) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1366) most significant: sum
s_add_u32 s1, s1, s19 // (op 1366) most significant: sum


s_mov_b32 s68, s38 // (op 1366) reg expression
s_mov_b32 s69, s39 // (op 1366) reg expression
s_mov_b32 s70, s0 // (op 1366) reg expression
s_mov_b32 s71, 131072 // (op 1366) call()

s_mul_hi_i32 s0, s16, s10 // (op 1383) 
v_and_b32 v15, -16, v1 // (op 1383) 
s_add_i32 s1, s0, s16 // (op 1383) 

v_sub_u32 v22, v1, v15 // (op 1383) 

s_ashr_i32 s0, s1, 31 // (op 1383) 
v_lshrrev_b32 v15, 4, v1 // (op 1383) 
s_cmp_eq_i32 s10, 0 // (op 1383) 
s_mov_b32 s18, scc // (op 1383) 
s_cmp_eq_i32 s18, 0 // (op 1383) 
s_cselect_b32 s19, 0, -1 // (op 1383) 

s_and_b32 s18, s11, 31 // (op 1383) 
s_lshl_b32 s22, 1, s18 // (op 1383) 
s_add_i32 s23, s22, s19 // (op 1383) 


s_and_b32 s19, s0, s23 // (op 1383) 


s_add_i32 s0, s1, s19 // (op 1383) 


s_ashr_i32 s1, s0, s18 // (op 1383) 


s_lshl_b32 s0, s11, 24 // (op 1383) 
s_ashr_i32 s18, s0, 31 // (op 1383) 

s_xor_b32 s0, s1, s18 // (op 1383) 

s_sub_i32 s1, s0, s18 // (op 1383) 


s_mul_hi_i32 s0, s1, s12 // (op 1383) 
s_add_i32 s18, s0, s1 // (op 1383) 

s_ashr_i32 s0, s18, 31 // (op 1383) 
s_cmp_eq_i32 s12, 0 // (op 1383) 
s_mov_b32 s19, scc // (op 1383) 
s_cmp_eq_i32 s19, 0 // (op 1383) 
s_cselect_b32 s22, 0, -1 // (op 1383) 

s_and_b32 s19, s13, 31 // (op 1383) 
s_lshl_b32 s23, 1, s19 // (op 1383) 
s_add_i32 s24, s23, s22 // (op 1383) 


s_and_b32 s25, s0, s24 // (op 1383) 


s_add_i32 s0, s18, s25 // (op 1383) 


s_ashr_i32 s18, s0, s19 // (op 1383) 


s_lshl_b32 s0, s13, 24 // (op 1383) 
s_ashr_i32 s19, s0, 31 // (op 1383) 

s_xor_b32 s0, s18, s19 // (op 1383) 

s_sub_i32 s18, s0, s19 // (op 1383) 


s_mul_i32 s0, s1, s15 // (op 1383) 

s_sub_u32 s1, s16, s0 // (op 1383) 

s_mul_i32 s0, s18, s15 // (op 1383) 
s_add_u32 s0, s0, s1 // (op 1383) 


s_mul_hi_i32 s1, s16, s9 // (op 1383) 
s_add_i32 s18, s1, s16 // (op 1383) 

s_ashr_i32 s1, s18, 31 // (op 1383) 
s_cmp_eq_i32 s9, 0 // (op 1383) 
s_mov_b32 s19, scc // (op 1383) 
s_cmp_eq_i32 s19, 0 // (op 1383) 
s_cselect_b32 s22, 0, -1 // (op 1383) 

s_and_b32 s19, s46, 31 // (op 1383) 
s_lshl_b32 s23, 1, s19 // (op 1383) 
s_add_i32 s24, s23, s22 // (op 1383) 


s_and_b32 s25, s1, s24 // (op 1383) 


s_add_i32 s1, s18, s25 // (op 1383) 


s_ashr_i32 s18, s1, s19 // (op 1383) 


s_lshl_b32 s1, s46, 24 // (op 1383) 
s_ashr_i32 s19, s1, 31 // (op 1383) 

s_xor_b32 s1, s18, s19 // (op 1383) 

s_sub_i32 s18, s1, s19 // (op 1383) 


s_add_u32 s50, s6, 64 // (op 1383) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1383) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1383) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1383) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1383) 

s_mul_i32 s1, s15, s24 // (op 1383) 

s_mul_i32 s19, s8, s1 // (op 1383) 

s_mul_i32 s1, s18, s19 // (op 1383) 

s_sub_u32 s19, s16, s1 // (op 1383) 

s_mov_b32 s1, s19 // (op 1383) 
s_mul_hi_i32 s22, s1, s14 // (op 1383) 
s_add_i32 s23, s22, s1 // (op 1383) 


s_ashr_i32 s1, s23, 31 // (op 1383) 
s_cmp_eq_i32 s14, 0 // (op 1383) 
s_mov_b32 s22, scc // (op 1383) 
s_cmp_eq_i32 s22, 0 // (op 1383) 
s_cselect_b32 s24, 0, -1 // (op 1383) 

s_and_b32 s22, s48, 31 // (op 1383) 
s_lshl_b32 s25, 1, s22 // (op 1383) 
s_add_i32 s49, s25, s24 // (op 1383) 


s_and_b32 s24, s1, s49 // (op 1383) 


s_add_i32 s1, s23, s24 // (op 1383) 


s_ashr_i32 s23, s1, s22 // (op 1383) 


s_lshl_b32 s1, s48, 24 // (op 1383) 
s_ashr_i32 s22, s1, 31 // (op 1383) 

s_xor_b32 s1, s23, s22 // (op 1383) 

s_sub_i32 s23, s1, s22 // (op 1383) 


s_mul_i32 s1, s47, s15 // (op 1383) 
s_add_u32 s24, s20, 16 // (op 1383) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1383) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1383) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1383) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1383) 

s_sub_i32 s22, s24, s1 // (op 1383) 

s_mul_i32 s24, s23, s22 // (op 1383) 


s_sub_u32 s25, s19, s24 // (op 1383) 


s_add_u32 s19, s25, s1 // (op 1383) 


s_cmp_eq_i32 s18, 0 // (op 1383) 
s_mov_b32 s1, scc // (op 1383) 

s_cmp_eq_i32 s1, 0 // (op 1383) 
s_cselect_b32 s18, s19, s0 // (op 1383) 



v_lshl_add_u32 v23, s18, 4, v22 // (op 1383) 


v_mul_lo_u32 v22, v15, s36 // (op 1383) 

v_add_u32 v15, v23, v22 // (op 1383) 


s_mul_i32 s1, s36, s41 // (op 1385) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1385) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1385) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1385) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1385) most significant: sum
s_add_u32 s1, s1, s19 // (op 1385) most significant: sum


s_mov_b32 s68, s38 // (op 1385) reg expression
s_mov_b32 s69, s39 // (op 1385) reg expression
s_mov_b32 s70, s0 // (op 1385) reg expression
s_mov_b32 s71, 131072 // (op 1385) call()

s_mul_hi_i32 s0, s16, s10 // (op 1428) 
v_lshrrev_b32 v22, 1, v1 // (op 1428) 
s_add_i32 s1, s0, s16 // (op 1428) 

v_and_b32 v23, -2, v1 // (op 1428) 
s_ashr_i32 s0, s1, 31 // (op 1428) 
v_sub_u32 v24, v1, v23 // (op 1428) 

s_cmp_eq_i32 s10, 0 // (op 1428) 
s_mov_b32 s18, scc // (op 1428) 
s_cmp_eq_i32 s18, 0 // (op 1428) 
s_cselect_b32 s19, 0, -1 // (op 1428) 

s_and_b32 s18, s11, 31 // (op 1428) 
s_lshl_b32 s22, 1, s18 // (op 1428) 
s_add_i32 s23, s22, s19 // (op 1428) 


s_and_b32 s19, s0, s23 // (op 1428) 


s_add_i32 s0, s1, s19 // (op 1428) 


s_ashr_i32 s1, s0, s18 // (op 1428) 


s_lshl_b32 s0, s11, 24 // (op 1428) 
s_ashr_i32 s18, s0, 31 // (op 1428) 

s_xor_b32 s0, s1, s18 // (op 1428) 

s_sub_i32 s1, s0, s18 // (op 1428) 


s_mul_hi_i32 s0, s1, s12 // (op 1428) 
s_add_i32 s18, s0, s1 // (op 1428) 

s_ashr_i32 s0, s18, 31 // (op 1428) 
s_cmp_eq_i32 s12, 0 // (op 1428) 
s_mov_b32 s19, scc // (op 1428) 
s_cmp_eq_i32 s19, 0 // (op 1428) 
s_cselect_b32 s22, 0, -1 // (op 1428) 

s_and_b32 s19, s13, 31 // (op 1428) 
s_lshl_b32 s23, 1, s19 // (op 1428) 
s_add_i32 s24, s23, s22 // (op 1428) 


s_and_b32 s25, s0, s24 // (op 1428) 


s_add_i32 s0, s18, s25 // (op 1428) 


s_ashr_i32 s18, s0, s19 // (op 1428) 


s_lshl_b32 s0, s13, 24 // (op 1428) 
s_ashr_i32 s19, s0, 31 // (op 1428) 

s_xor_b32 s0, s18, s19 // (op 1428) 

s_sub_i32 s18, s0, s19 // (op 1428) 


s_add_u32 s50, s6, 64 // (op 1428) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1428) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1428) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1428) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1428) 

s_mul_i32 s0, s18, s24 // (op 1428) 

s_sub_i32 s18, s1, s0 // (op 1428) 


s_mul_hi_i32 s19, s16, s9 // (op 1428) 
s_add_i32 s0, s19, s16 // (op 1428) 

s_ashr_i32 s1, s0, 31 // (op 1428) 
s_cmp_eq_i32 s9, 0 // (op 1428) 
s_mov_b32 s19, scc // (op 1428) 
s_cmp_eq_i32 s19, 0 // (op 1428) 
s_cselect_b32 s22, 0, -1 // (op 1428) 

s_and_b32 s19, s46, 31 // (op 1428) 
s_lshl_b32 s23, 1, s19 // (op 1428) 
s_add_i32 s49, s23, s22 // (op 1428) 


s_and_b32 s22, s1, s49 // (op 1428) 


s_add_i32 s1, s0, s22 // (op 1428) 


s_ashr_i32 s0, s1, s19 // (op 1428) 


s_lshl_b32 s1, s46, 24 // (op 1428) 
s_ashr_i32 s19, s1, 31 // (op 1428) 

s_xor_b32 s1, s0, s19 // (op 1428) 

s_sub_i32 s0, s1, s19 // (op 1428) 


s_mul_i32 s1, s15, s24 // (op 1428) 

s_mul_i32 s19, s8, s1 // (op 1428) 

s_mul_i32 s1, s0, s19 // (op 1428) 

s_sub_u32 s19, s16, s1 // (op 1428) 

s_mul_hi_i32 s1, s19, s14 // (op 1428) 
s_add_i32 s22, s1, s19 // (op 1428) 


s_ashr_i32 s1, s22, 31 // (op 1428) 
s_cmp_eq_i32 s14, 0 // (op 1428) 
s_mov_b32 s19, scc // (op 1428) 
s_cmp_eq_i32 s19, 0 // (op 1428) 
s_cselect_b32 s23, 0, -1 // (op 1428) 

s_and_b32 s19, s48, 31 // (op 1428) 
s_lshl_b32 s24, 1, s19 // (op 1428) 
s_add_i32 s25, s24, s23 // (op 1428) 


s_and_b32 s23, s1, s25 // (op 1428) 


s_add_i32 s1, s22, s23 // (op 1428) 


s_ashr_i32 s22, s1, s19 // (op 1428) 


s_lshl_b32 s1, s48, 24 // (op 1428) 
s_ashr_i32 s19, s1, 31 // (op 1428) 

s_xor_b32 s1, s22, s19 // (op 1428) 

s_sub_i32 s22, s1, s19 // (op 1428) 


s_cmp_eq_i32 s0, 0 // (op 1428) 
s_mov_b32 s1, scc // (op 1428) 

s_cmp_eq_i32 s1, 0 // (op 1428) 
s_cselect_b32 s0, s22, s18 // (op 1428) 



v_lshl_add_u32 v23, s0, 6, v22 // (op 1428) 


v_mul_lo_u32 v22, v23, s42 // (op 1428) 

v_lshl_add_u32 v23, v24, 2, v22 // (op 1428) 


s_mul_i32 s1, s42, s7 // (op 1430) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 1430) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 1430) least significant
s_mul_hi_u32 s19, s42, s6 // (op 1430) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1430) most significant: sum
s_add_u32 s1, s1, s19 // (op 1430) most significant: sum


s_mov_b32 s72, s44 // (op 1430) reg expression
s_mov_b32 s73, s45 // (op 1430) reg expression
s_mov_b32 s74, s0 // (op 1430) reg expression
s_mov_b32 s75, 131072 // (op 1430) call()

s_mul_hi_i32 s0, s16, s10 // (op 1447) 
v_lshrrev_b32 v22, 1, v1 // (op 1447) 
s_add_i32 s1, s0, s16 // (op 1447) 

v_and_b32 v24, -2, v1 // (op 1447) 
s_ashr_i32 s0, s1, 31 // (op 1447) 
v_sub_u32 v25, v1, v24 // (op 1447) 

s_cmp_eq_i32 s10, 0 // (op 1447) 
s_mov_b32 s18, scc // (op 1447) 
s_cmp_eq_i32 s18, 0 // (op 1447) 
s_cselect_b32 s19, 0, -1 // (op 1447) 

s_and_b32 s18, s11, 31 // (op 1447) 
s_lshl_b32 s22, 1, s18 // (op 1447) 
s_add_i32 s23, s22, s19 // (op 1447) 


s_and_b32 s19, s0, s23 // (op 1447) 


s_add_i32 s0, s1, s19 // (op 1447) 


s_ashr_i32 s1, s0, s18 // (op 1447) 


s_lshl_b32 s0, s11, 24 // (op 1447) 
s_ashr_i32 s18, s0, 31 // (op 1447) 

s_xor_b32 s0, s1, s18 // (op 1447) 

s_sub_i32 s1, s0, s18 // (op 1447) 


s_mul_hi_i32 s0, s1, s12 // (op 1447) 
s_add_i32 s18, s0, s1 // (op 1447) 

s_ashr_i32 s0, s18, 31 // (op 1447) 
s_cmp_eq_i32 s12, 0 // (op 1447) 
s_mov_b32 s19, scc // (op 1447) 
s_cmp_eq_i32 s19, 0 // (op 1447) 
s_cselect_b32 s22, 0, -1 // (op 1447) 

s_and_b32 s19, s13, 31 // (op 1447) 
s_lshl_b32 s23, 1, s19 // (op 1447) 
s_add_i32 s24, s23, s22 // (op 1447) 


s_and_b32 s25, s0, s24 // (op 1447) 


s_add_i32 s0, s18, s25 // (op 1447) 


s_ashr_i32 s18, s0, s19 // (op 1447) 


s_lshl_b32 s0, s13, 24 // (op 1447) 
s_ashr_i32 s19, s0, 31 // (op 1447) 

s_xor_b32 s0, s18, s19 // (op 1447) 

s_sub_i32 s18, s0, s19 // (op 1447) 


s_add_u32 s50, s6, 64 // (op 1447) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1447) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1447) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1447) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1447) 

s_mul_i32 s0, s18, s24 // (op 1447) 

s_sub_i32 s18, s1, s0 // (op 1447) 


s_mul_hi_i32 s19, s16, s9 // (op 1447) 
s_add_i32 s0, s19, s16 // (op 1447) 

s_ashr_i32 s1, s0, 31 // (op 1447) 
s_cmp_eq_i32 s9, 0 // (op 1447) 
s_mov_b32 s19, scc // (op 1447) 
s_cmp_eq_i32 s19, 0 // (op 1447) 
s_cselect_b32 s22, 0, -1 // (op 1447) 

s_and_b32 s19, s46, 31 // (op 1447) 
s_lshl_b32 s23, 1, s19 // (op 1447) 
s_add_i32 s49, s23, s22 // (op 1447) 


s_and_b32 s22, s1, s49 // (op 1447) 


s_add_i32 s1, s0, s22 // (op 1447) 


s_ashr_i32 s0, s1, s19 // (op 1447) 


s_lshl_b32 s1, s46, 24 // (op 1447) 
s_ashr_i32 s19, s1, 31 // (op 1447) 

s_xor_b32 s1, s0, s19 // (op 1447) 

s_sub_i32 s0, s1, s19 // (op 1447) 


s_mul_i32 s1, s15, s24 // (op 1447) 

s_mul_i32 s19, s8, s1 // (op 1447) 

s_mul_i32 s1, s0, s19 // (op 1447) 

s_sub_u32 s19, s16, s1 // (op 1447) 

s_mul_hi_i32 s1, s19, s14 // (op 1447) 
s_add_i32 s22, s1, s19 // (op 1447) 


s_ashr_i32 s1, s22, 31 // (op 1447) 
s_cmp_eq_i32 s14, 0 // (op 1447) 
s_mov_b32 s19, scc // (op 1447) 
s_cmp_eq_i32 s19, 0 // (op 1447) 
s_cselect_b32 s23, 0, -1 // (op 1447) 

s_and_b32 s19, s48, 31 // (op 1447) 
s_lshl_b32 s24, 1, s19 // (op 1447) 
s_add_i32 s25, s24, s23 // (op 1447) 


s_and_b32 s23, s1, s25 // (op 1447) 


s_add_i32 s1, s22, s23 // (op 1447) 


s_ashr_i32 s22, s1, s19 // (op 1447) 


s_lshl_b32 s1, s48, 24 // (op 1447) 
s_ashr_i32 s19, s1, 31 // (op 1447) 

s_xor_b32 s1, s22, s19 // (op 1447) 

s_sub_i32 s22, s1, s19 // (op 1447) 


s_cmp_eq_i32 s0, 0 // (op 1447) 
s_mov_b32 s1, scc // (op 1447) 

s_cmp_eq_i32 s1, 0 // (op 1447) 
s_cselect_b32 s0, s22, s18 // (op 1447) 



v_lshl_add_u32 v24, s0, 6, v22 // (op 1447) 


v_mul_lo_u32 v22, v24, s42 // (op 1447) 

v_lshl_add_u32 v24, v25, 2, v22 // (op 1447) 


s_mul_i32 s1, s42, s7 // (op 1449) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 1449) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 1449) least significant
s_mul_hi_u32 s19, s42, s6 // (op 1449) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1449) most significant: sum
s_add_u32 s1, s1, s19 // (op 1449) most significant: sum


s_mov_b32 s72, s44 // (op 1449) reg expression
s_mov_b32 s73, s45 // (op 1449) reg expression
s_mov_b32 s74, s0 // (op 1449) reg expression
s_mov_b32 s75, 131072 // (op 1449) call()


s_mul_i32 s1, s33, 6 // (op 1037) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 1037) least significant
s_mul_hi_u32 s18, s32, 6 // (op 1037) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1037) most significant: sum

v_lshrrev_b64 v[26:27], 3, s[0:1] // (op 1037) 


v_mov_b32 v22, v16 // (op 1037) 
buffer_load_dwordx4 v[28:31], v22, s[64:67], 0 offen // (op 1037) Load value
buffer_load_dwordx2 v[32:33], v22, s[64:67], 0 offen offset:16 // (op 1037) Load value
buffer_load_dwordx4 v[34:37], v22, s[64:67], 0 offen offset:24 // (op 1037) Load value
buffer_load_dwordx2 v[38:39], v22, s[64:67], 0 offen offset:40 // (op 1037) Load value
v_add_u32 v22, v22, v26 // (op 1037) 
buffer_load_dwordx4 v[40:43], v22, s[64:67], 0 offen // (op 1037) Load value
buffer_load_dwordx2 v[44:45], v22, s[64:67], 0 offen offset:16 // (op 1037) Load value
buffer_load_dwordx4 v[46:49], v22, s[64:67], 0 offen offset:24 // (op 1037) Load value
buffer_load_dwordx2 v[50:51], v22, s[64:67], 0 offen offset:40 // (op 1037) Load value





buffer_load_dword v22, v23, s[72:75], 0 offen // (op 1040) Load value
v_lshlrev_b32 v25, 4, v1 // (op 1513) 
s_nop 0
v_readfirstlane_b32 s0, v25 // (op 1513) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[26:27], 7 // (op 1191) 
v_mov_b64 v[26:27], s[18:19] // (op 1191) convert


v_mov_b64 v[52:53], v[18:19] // (op 1191) reg expression
s_mov_b32 m0, s0 // (op 1191) reg expression
buffer_load_dwordx4 v52, s[52:55], 0 offen offset: 0 lds // (op 1191) Load value direct to lds
v_add_co_u32 v52, s[18:19], v52, v26 // (op 1191) least significant half
v_addc_co_u32 v53, s[18:19], 0, v53, s[18:19] // (op 1191) most significant half

s_add_u32 m0, m0, 2048 // (op 1191) 
buffer_load_dwordx4 v52, s[52:55], 0 offen offset: 0 lds // (op 1191) Load value direct to lds





v_mov_b32 v25, v14 // (op 1046) reg expression

buffer_load_ubyte v26, v25, s[68:71], 0 offen offset: 0 // (op 1046) Load value


s_mul_i32 s1, s33, 6 // (op 1049) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 1049) least significant
s_mul_hi_u32 s18, s32, 6 // (op 1049) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1049) most significant: sum

v_lshrrev_b64 v[52:53], 3, s[0:1] // (op 1049) 


v_mov_b32 v25, v16 // (op 1049) 
buffer_load_dwordx4 v[54:57], v25, s[64:67], 0 offen offset:192 // (op 1049) Load value
buffer_load_dwordx2 v[58:59], v25, s[64:67], 0 offen offset:208 // (op 1049) Load value
buffer_load_dwordx4 v[60:63], v25, s[64:67], 0 offen offset:216 // (op 1049) Load value
buffer_load_dwordx2 v[64:65], v25, s[64:67], 0 offen offset:232 // (op 1049) Load value
v_add_u32 v25, v25, v52 // (op 1049) 
buffer_load_dwordx4 v[66:69], v25, s[64:67], 0 offen offset:192 // (op 1049) Load value
buffer_load_dwordx2 v[70:71], v25, s[64:67], 0 offen offset:208 // (op 1049) Load value
buffer_load_dwordx4 v[72:75], v25, s[64:67], 0 offen offset:216 // (op 1049) Load value
buffer_load_dwordx2 v[76:77], v25, s[64:67], 0 offen offset:232 // (op 1049) Load value






buffer_load_dword v16, v23, s[72:75], 0 offen offset:8 // (op 1052) Load value

v_lshlrev_b32 v23, 4, v1 // (op 1739) 
s_nop 0
v_readfirstlane_b32 s0, v23 // (op 1739) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[26:27], 7 // (op 1196) 
v_mov_b64 v[52:53], s[18:19] // (op 1196) convert


s_lshl_b64 s[18:19], s[26:27], 8 // (op 1196) 
v_mov_b32 v23, s18 // (op 1196) 
v_mov_b32 v25, s19 // (op 1196) 
v_add_co_u32 v78, s[50:51], v18, v23 // (op 1196) least significant half
v_addc_co_u32 v79, s[50:51], v19, v25, s[50:51] // (op 1196) most significant half




s_add_u32 s1, 4096, s0 // (op 1196) 
s_mov_b32 m0, s1 // (op 1196) reg expression
buffer_load_dwordx4 v78, s[52:55], 0 offen offset: 0 lds // (op 1196) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v52 // (op 1196) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 1196) most significant half

s_add_u32 m0, m0, 2048 // (op 1196) 
buffer_load_dwordx4 v78, s[52:55], 0 offen offset: 0 lds // (op 1196) Load value direct to lds







s_lshl_b32 s0, s36, 3 // (op 1058) 
v_add_u32 v23, v14, s0 // (op 1058) 


buffer_load_ubyte v25, v23, s[68:71], 0 offen offset: 0 // (op 1058) Load value


v_and_b32 v14, -4, v1 // (op 1566) 
v_sub_u32 v23, v1, v14 // (op 1566) 

v_lshlrev_b32 v14, 6, v23 // (op 1566) 

v_lshrrev_b32 v23, 2, v1 // (op 1566) 
v_lshl_add_u32 v27, v23, 9, v14 // (op 1566) 


v_mul_lo_u32 v14, v27, 6 // (op 1566) 

v_lshrrev_b32 v23, 3, v14 // (op 1566) 




s_waitcnt vmcnt(23)
ds_write_b128 v23, v[28:31] offset:8192 // (op 1060) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v23, v[32:33] offset:8208 // (op 1060) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v23, v[34:37] offset:8216 // (op 1060) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v23, v[38:39] offset:8232 // (op 1060) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v23, v[40:43] offset:8384 // (op 1060) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v23, v[44:45] offset:8400 // (op 1060) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v23, v[46:49] offset:8408 // (op 1060) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v23, v[50:51] offset:8424 // (op 1060) Store local data 

v_and_b32 v14, -2, v1 // (op 1677) 
v_sub_u32 v23, v1, v14 // (op 1677) 

v_lshlrev_b32 v14, 2, v23 // (op 1677) 

v_lshrrev_b32 v23, 1, v1 // (op 1677) 
v_lshl_add_u32 v27, v23, 3, v14 // (op 1677) 





s_waitcnt vmcnt(15)
ds_write_b32 v27, v22 offset:20480 // (op 1061) Store local data 

v_and_b32 v14, -16, v1 // (op 1624) 
v_sub_u32 v23, v1, v14 // (op 1624) 

v_and_b32 v14, 4294967280, v1 // (op 1624) 
v_add_u32 v27, v14, v23 // (op 1624) 





s_waitcnt vmcnt(12)
ds_write_b8 v27, v26 offset:20992 // (op 1063) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1903) 



ds_read_b128 v[78:81], v9 offset:8192 // (op 1081) Load local data 
ds_read_b64 v[82:83], v9 offset:8208 // (op 1081) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1944) 



ds_read_u8 v14, v5 offset:20480 // (op 1094) Load local data 



ds_read_b128 v[84:87], v9 offset:14336 // (op 1086) Load local data 
ds_read_b64 v[88:89], v9 offset:14352 // (op 1086) Load local data 



ds_read_u8 v23, v5 offset:20736 // (op 1099) Load local data 
s_barrier  // (op 1884) 



ds_read_b64_tr_b8 v[90:91], v0 // (op 1077) Transpose load local data 
ds_read_b64_tr_b8 v[92:93], v0 offset:128 // (op 1077) Transpose load local data 
ds_read_b64_tr_b8 v[94:95], v0 offset:1024 // (op 1077) Transpose load local data 
ds_read_b64_tr_b8 v[96:97], v0 offset:1152 // (op 1077) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1924) 


ds_read_u8 v27, v4 offset:20992 // (op 1090) Load local data 
v_lshlrev_b32 v18, 4, v1 // (op 1485) 
s_nop 0
v_readfirstlane_b32 s0, v18 // (op 1485) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -4, v1 // (op 1537) 
v_sub_u32 v19, v1, v18 // (op 1537) 

v_lshlrev_b32 v18, 6, v19 // (op 1537) 

v_lshrrev_b32 v19, 2, v1 // (op 1537) 
v_lshl_add_u32 v52, v19, 9, v18 // (op 1537) 


v_mul_lo_u32 v53, v52, 6 // (op 1537) 

v_lshrrev_b32 v52, 3, v53 // (op 1537) 

v_and_b32 v53, -16, v1 // (op 1595) 
v_sub_u32 v18, v1, v53 // (op 1595) 

v_and_b32 v19, 4294967280, v1 // (op 1595) 
v_add_u32 v53, v19, v18 // (op 1595) 


v_and_b32 v18, -2, v1 // (op 1648) 
v_sub_u32 v19, v1, v18 // (op 1648) 

v_lshlrev_b32 v18, 2, v19 // (op 1648) 

v_lshrrev_b32 v19, 1, v1 // (op 1648) 
v_lshl_add_u32 v98, v19, 3, v18 // (op 1648) 


v_lshlrev_b32 v18, 4, v1 // (op 1724) 
s_nop 0
v_readfirstlane_b32 s1, v18 // (op 1724) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -4, v1 // (op 1763) 
v_sub_u32 v19, v1, v18 // (op 1763) 

v_lshlrev_b32 v18, 6, v19 // (op 1763) 

v_lshrrev_b32 v19, 2, v1 // (op 1763) 
v_lshl_add_u32 v99, v19, 9, v18 // (op 1763) 


v_mul_lo_u32 v18, v99, 6 // (op 1763) 

v_lshrrev_b32 v19, 3, v18 // (op 1763) 

v_and_b32 v18, -16, v1 // (op 1792) 
v_sub_u32 v99, v1, v18 // (op 1792) 

v_and_b32 v18, 4294967280, v1 // (op 1792) 
v_add_u32 v100, v18, v99 // (op 1792) 


v_and_b32 v18, -2, v1 // (op 1816) 
v_sub_u32 v99, v1, v18 // (op 1816) 

v_lshlrev_b32 v18, 2, v99 // (op 1816) 

v_lshrrev_b32 v99, 1, v1 // (op 1816) 
v_lshl_add_u32 v101, v99, 3, v18 // (op 1816) 


s_mov_b32 s18, 0 // (op 404) call()
s_ashr_i64 s[50:51], s[30:31], 8 // (op 403) 
s_mov_b32 s19, s50 // (op 403) convert

s_and_b32 s22, s19, 4294967294 // (op 403) 

s_cmp_lt_u32 s18, s22 // (op 403) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_403 // (op 403) Condition: Top (jump to RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_403 if false)
RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_403:

s_add_u32 s18, s18, 2 // (op 405) 
s_mul_i32 s51, s33, 6 // (op 10) most significant: low of high * low
s_mul_i32 s50, s32, 6 // (op 10) least significant
s_mul_hi_u32 s19, s32, 6 // (op 10) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 10) most significant: sum

v_lshrrev_b64 v[102:103], 3, s[50:51] // (op 10) 


v_mov_b32 v18, v17 // (op 10) 
v_mov_b32 v99, v18 // (op 10) 
buffer_load_dwordx4 v[28:31], v99, s[64:67], 0 offen offset:384 // (op 10) Load value
buffer_load_dwordx2 v[32:33], v99, s[64:67], 0 offen offset:400 // (op 10) Load value
buffer_load_dwordx4 v[34:37], v99, s[64:67], 0 offen offset:408 // (op 10) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:97], v[78:83], a[0:3], v27, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 90) 
buffer_load_dwordx2 v[38:39], v99, s[64:67], 0 offen offset:424 // (op 10) Load value
s_barrier  // (op 1893) 
v_add_u32 v99, v99, v102 // (op 10) 



buffer_load_dwordx4 v[40:43], v99, s[64:67], 0 offen offset:384 // (op 10) Load value
buffer_load_dwordx2 v[44:45], v99, s[64:67], 0 offen offset:400 // (op 10) Load value
ds_read_b64_tr_b8 v[104:105], v0 offset:2048 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[106:107], v0 offset:2176 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[108:109], v0 offset:3072 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[110:111], v0 offset:3200 // (op 93) Transpose load local data 
buffer_load_dwordx4 v[46:49], v99, s[64:67], 0 offen offset:408 // (op 10) Load value
s_barrier  // (op 1913) 
buffer_load_dwordx2 v[50:51], v99, s[64:67], 0 offen offset:424 // (op 10) Load value







ds_read_b128 v[78:81], v9 offset:8288 // (op 97) Load local data 
ds_read_b64 v[82:83], v9 offset:8304 // (op 97) Load local data 
v_mov_b32 v99, v24 // (op 25) 

s_barrier  // (op 1934) 
buffer_load_dword v22, v99, s[72:75], 0 offen offset:16 // (op 25) Load value


ds_read_u8 v102, v4 offset:21056 // (op 101) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1887) 

s_barrier  // (op 1954) 
s_lshl_b64 s[50:51], s[26:27], 7 // (op 1175) 



v_mov_b64 v[112:113], s[50:51] // (op 1175) convert


v_mov_b64 v[114:115], v[20:21] // (op 1175) 
s_lshl_b64 s[50:51], s[26:27], 8 // (op 1175) 
s_lshl_b64 s[22:23], s[50:51], 1 // (op 1175) 

v_mov_b32 v103, s22 // (op 1175) 
v_mov_b32 v116, s23 // (op 1175) 
v_add_co_u32 v118, s[24:25], v114, v103 // (op 1175) least significant half
v_addc_co_u32 v119, s[24:25], v115, v116, s[24:25] // (op 1175) most significant half




s_mov_b32 m0, s0 // (op 1175) reg expression
ds_read_u8 v14, v5 offset:20484 // (op 105) Load local data 
buffer_load_dwordx4 v118, s[52:55], 0 offen offset: 0 lds // (op 1175) Load value direct to lds
v_add_co_u32 v118, s[50:51], v118, v112 // (op 1175) least significant half
v_addc_co_u32 v119, s[50:51], 0, v119, s[50:51] // (op 1175) most significant half

s_add_u32 m0, m0, 2048 // (op 1175) 
buffer_load_dwordx4 v118, s[52:55], 0 offen offset: 0 lds // (op 1175) Load value direct to lds


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:111], v[78:83], a[0:3], v102, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 107) 


v_mov_b32 v103, v15 // (op 17) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:97], v[84:89], a[4:7], v27, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 451) 
s_lshl_b32 s19, s36, 4 // (op 17) 



ds_read_b128 v[84:87], v9 offset:14432 // (op 455) Load local data 
ds_read_b64 v[88:89], v9 offset:14448 // (op 455) Load local data 
v_add_u32 v112, v103, s19 // (op 17) 





ds_read_u8 v23, v5 offset:20740 // (op 459) Load local data 
buffer_load_ubyte v26, v112, s[68:71], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1974) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:111], v[84:89], a[4:7], v102, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 460) 





s_waitcnt vmcnt(23)
ds_write_b128 v19, v[54:57] offset:21120 // (op 306) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v19, v[58:59] offset:21136 // (op 306) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v19, v[60:63] offset:21144 // (op 306) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v19, v[64:65] offset:21160 // (op 306) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v19, v[66:69] offset:21312 // (op 306) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v19, v[70:71] offset:21328 // (op 306) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v19, v[72:75] offset:21336 // (op 306) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v19, v[76:77] offset:21352 // (op 306) Store local data 
s_barrier  // (op 1986) 



s_waitcnt vmcnt(15)
ds_write_b32 v101, v16 offset:33408 // (op 308) Store local data 
s_barrier  // (op 1980) 



s_waitcnt vmcnt(12)
ds_write_b8 v100, v25 offset:33920 // (op 307) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1970) 



ds_read_b128 v[78:81], v13 offset:21120 // (op 291) Load local data 
ds_read_b64 v[82:83], v13 offset:21136 // (op 291) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1983) 



ds_read_u8 v14, v12 offset:33408 // (op 295) Load local data 



ds_read_b128 v[104:107], v13 offset:27264 // (op 474) Load local data 
ds_read_b64 v[108:109], v13 offset:27280 // (op 474) Load local data 



ds_read_u8 v102, v12 offset:33664 // (op 478) Load local data 
s_barrier  // (op 1964) 



ds_read_b64_tr_b8 v[90:91], v3 offset:4096 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[92:93], v3 offset:4224 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[94:95], v3 offset:5120 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[96:97], v3 offset:5248 // (op 289) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1977) 


ds_read_u8 v27, v6 offset:33920 // (op 293) Load local data 

s_mul_i32 s51, s33, 6 // (op 309) most significant: low of high * low
s_mul_i32 s50, s32, 6 // (op 309) least significant
s_mul_hi_u32 s19, s32, 6 // (op 309) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 309) most significant: sum

v_lshrrev_b64 v[110:111], 3, s[50:51] // (op 309) 


v_mov_b32 v112, v18 // (op 309) 
buffer_load_dwordx4 v[54:57], v112, s[64:67], 0 offen offset:576 // (op 309) Load value
buffer_load_dwordx2 v[58:59], v112, s[64:67], 0 offen offset:592 // (op 309) Load value
buffer_load_dwordx4 v[60:63], v112, s[64:67], 0 offen offset:600 // (op 309) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:97], v[78:83], a[0:3], v27, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 296) 
buffer_load_dwordx2 v[64:65], v112, s[64:67], 0 offen offset:616 // (op 309) Load value
v_add_u32 v112, v112, v110 // (op 309) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:97], v[104:109], a[4:7], v27, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 461) 
buffer_load_dwordx4 v[66:69], v112, s[64:67], 0 offen offset:576 // (op 309) Load value


buffer_load_dwordx2 v[70:71], v112, s[64:67], 0 offen offset:592 // (op 309) Load value
ds_read_b64_tr_b8 v[116:117], v3 offset:6144 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[118:119], v3 offset:6272 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[120:121], v3 offset:7168 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[122:123], v3 offset:7296 // (op 298) Transpose load local data 
buffer_load_dwordx4 v[72:75], v112, s[64:67], 0 offen offset:600 // (op 309) Load value
buffer_load_dwordx2 v[76:77], v112, s[64:67], 0 offen offset:616 // (op 309) Load value






ds_read_b128 v[104:107], v13 offset:21216 // (op 300) Load local data 



ds_read_b64 v[108:109], v13 offset:21232 // (op 300) Load local data 
buffer_load_dword v16, v99, s[72:75], 0 offen offset:24 // (op 311) Load value



v_add_u32 v17, 384, v17 // (op 1311) 
v_add_u32 v24, 16, v24 // (op 1439) 
ds_read_u8 v18, v6 offset:33984 // (op 302) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1967) 




s_barrier  // (op 2029) 
s_lshl_b64 s[50:51], s[26:27], 7 // (op 1186) 
v_mov_b64 v[110:111], s[50:51] // (op 1186) convert


s_lshl_b64 s[50:51], s[26:27], 8 // (op 1186) 
s_mul_i32 s23, 3, s51 // (op 1186) most significant: low of low * high
s_mul_i32 s22, 3, s50 // (op 1186) least significant
s_mul_hi_u32 s19, 3, s50 // (op 1186) most significant: high of low * low
s_add_u32 s23, s23, s19 // (op 1186) most significant: sum


v_mov_b32 v99, s22 // (op 1186) 
v_mov_b32 v102, s23 // (op 1186) 
v_add_co_u32 v112, s[24:25], v114, v99 // (op 1186) least significant half
v_addc_co_u32 v113, s[24:25], v115, v102, s[24:25] // (op 1186) most significant half




s_add_u32 s19, 4096, s1 // (op 1186) 
s_mov_b32 m0, s19 // (op 1186) reg expression
ds_read_u8 v99, v12 offset:33412 // (op 304) Load local data 
buffer_load_dwordx4 v112, s[52:55], 0 offen offset: 0 lds // (op 1186) Load value direct to lds



v_add_co_u32 v112, s[50:51], v112, v110 // (op 1186) least significant half
v_addc_co_u32 v113, s[50:51], 0, v113, s[50:51] // (op 1186) most significant half

s_add_u32 m0, m0, 2048 // (op 1186) 
buffer_load_dwordx4 v112, s[52:55], 0 offen offset: 0 lds // (op 1186) Load value direct to lds



ds_read_b128 v[124:127], v13 offset:27360 // (op 465) Load local data 

ds_read_b64 v[128:129], v13 offset:27376 // (op 465) Load local data 


s_lshl_b32 s19, s36, 3 // (op 310) 



s_barrier  // (op 2067) 
ds_read_u8 v102, v12 offset:33668 // (op 469) Load local data 
s_mul_i32 s22, 3, s19 // (op 310) 

v_add_u32 v110, v103, s22 // (op 310) 


s_barrier  // (op 2071) 
buffer_load_ubyte v25, v110, s[68:71], 0 offen offset: 0 // (op 310) Load value

s_lshl_b64 s[50:51], s[26:27], 8 // (op 1247) 

s_lshl_b64 s[22:23], s[50:51], 1 // (op 1247) 

s_barrier  // (op 1906) 
s_lshl_b32 s19, s36, 4 // (op 1375) 



v_add_u32 v15, v15, s19 // (op 1375) 

v_mov_b32 v103, s22 // (op 1247) 
v_mov_b32 v110, s23 // (op 1247) 
v_add_co_u32 v20, s[24:25], v20, v103 // (op 1247) least significant half
v_addc_co_u32 v21, s[24:25], v21, v110, s[24:25] // (op 1247) most significant half




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[116:123], v[104:109], a[0:3], v18, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 305) 
s_waitcnt vmcnt(23)
ds_write_b128 v52, v[28:31] offset:8192 // (op 13) Store local data 


s_waitcnt vmcnt(22)
ds_write_b64 v52, v[32:33] offset:8208 // (op 13) Store local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[116:123], v[124:129], a[4:7], v18, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 470) 
s_waitcnt vmcnt(21)
ds_write_b128 v52, v[34:37] offset:8216 // (op 13) Store local data 




s_waitcnt vmcnt(20)
ds_write_b64 v52, v[38:39] offset:8232 // (op 13) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v52, v[40:43] offset:8384 // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v52, v[44:45] offset:8400 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v52, v[46:49] offset:8408 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v52, v[50:51] offset:8424 // (op 13) Store local data 
s_barrier  // (op 1947) 



s_waitcnt vmcnt(15)
ds_write_b32 v98, v22 offset:20480 // (op 30) Store local data 
s_barrier  // (op 1927) 



s_waitcnt vmcnt(12)
ds_write_b8 v53, v26 offset:20992 // (op 21) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1916) 



ds_read_b128 v[78:81], v9 offset:8192 // (op 77) Load local data 
ds_read_b64 v[82:83], v9 offset:8208 // (op 77) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1957) 



ds_read_u8 v14, v5 offset:20480 // (op 87) Load local data 



ds_read_b128 v[84:87], v9 offset:14336 // (op 446) Load local data 
ds_read_b64 v[88:89], v9 offset:14352 // (op 446) Load local data 



ds_read_u8 v23, v5 offset:20736 // (op 450) Load local data 
s_barrier  // (op 1896) 



ds_read_b64_tr_b8 v[90:91], v0 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[92:93], v0 offset:128 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[94:95], v0 offset:1024 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[96:97], v0 offset:1152 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1937) 


ds_read_u8 v27, v4 offset:20992 // (op 82) Load local data 
s_ashr_i64 s[50:51], s[30:31], 8 // (op 403) 
s_mov_b32 s19, s50 // (op 403) convert

s_and_b32 s22, s19, 4294967294 // (op 403) 

s_cmp_lt_u32 s18, s22 // (op 403) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_403 // (op 403) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_403 if true)
RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_403:
















s_barrier  // (op 2049) 
s_mul_hi_i32 s0, s16, s10 // (op 1209) 
s_add_i32 s1, s0, s16 // (op 1209) 

s_ashr_i32 s0, s1, 31 // (op 1209) 
s_cmp_eq_i32 s10, 0 // (op 1209) 
s_mov_b32 s18, scc // (op 1209) 
s_cmp_eq_i32 s18, 0 // (op 1209) 
s_cselect_b32 s19, 0, -1 // (op 1209) 

s_and_b32 s18, s11, 31 // (op 1209) 
s_lshl_b32 s22, 1, s18 // (op 1209) 
s_add_i32 s23, s22, s19 // (op 1209) 


s_and_b32 s19, s0, s23 // (op 1209) 


s_add_i32 s0, s1, s19 // (op 1209) 


s_ashr_i32 s1, s0, s18 // (op 1209) 


s_lshl_b32 s0, s11, 24 // (op 1209) 
s_ashr_i32 s18, s0, 31 // (op 1209) 

s_xor_b32 s0, s1, s18 // (op 1209) 

s_sub_i32 s1, s0, s18 // (op 1209) 


s_mul_hi_i32 s0, s1, s12 // (op 1209) 
s_add_i32 s18, s0, s1 // (op 1209) 

s_ashr_i32 s0, s18, 31 // (op 1209) 
s_cmp_eq_i32 s12, 0 // (op 1209) 
s_mov_b32 s19, scc // (op 1209) 
s_cmp_eq_i32 s19, 0 // (op 1209) 
s_cselect_b32 s22, 0, -1 // (op 1209) 

s_and_b32 s19, s13, 31 // (op 1209) 
s_lshl_b32 s23, 1, s19 // (op 1209) 
s_add_i32 s24, s23, s22 // (op 1209) 


s_and_b32 s25, s0, s24 // (op 1209) 


s_add_i32 s0, s18, s25 // (op 1209) 


s_ashr_i32 s18, s0, s19 // (op 1209) 


s_lshl_b32 s0, s13, 24 // (op 1209) 
s_ashr_i32 s19, s0, 31 // (op 1209) 

s_xor_b32 s0, s18, s19 // (op 1209) 

s_sub_i32 s18, s0, s19 // (op 1209) 


s_mul_i32 s0, s1, s15 // (op 1209) 

s_sub_u32 s1, s16, s0 // (op 1209) 

s_mul_i32 s0, s18, s15 // (op 1209) 
s_add_u32 s0, s0, s1 // (op 1209) 


s_mul_hi_i32 s1, s16, s9 // (op 1209) 
s_add_i32 s18, s1, s16 // (op 1209) 

s_ashr_i32 s1, s18, 31 // (op 1209) 
s_cmp_eq_i32 s9, 0 // (op 1209) 
s_mov_b32 s19, scc // (op 1209) 
s_cmp_eq_i32 s19, 0 // (op 1209) 
s_cselect_b32 s22, 0, -1 // (op 1209) 

s_and_b32 s19, s46, 31 // (op 1209) 
s_lshl_b32 s23, 1, s19 // (op 1209) 
s_add_i32 s24, s23, s22 // (op 1209) 


s_and_b32 s25, s1, s24 // (op 1209) 


s_add_i32 s1, s18, s25 // (op 1209) 


s_ashr_i32 s18, s1, s19 // (op 1209) 


s_lshl_b32 s1, s46, 24 // (op 1209) 
s_ashr_i32 s19, s1, 31 // (op 1209) 

s_xor_b32 s1, s18, s19 // (op 1209) 

s_sub_i32 s18, s1, s19 // (op 1209) 


s_add_u32 s50, s6, 64 // (op 1209) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1209) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1209) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1209) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1209) 

s_mul_i32 s1, s15, s24 // (op 1209) 

s_mul_i32 s19, s8, s1 // (op 1209) 

s_mul_i32 s1, s18, s19 // (op 1209) 

s_sub_u32 s19, s16, s1 // (op 1209) 

s_mov_b32 s1, s19 // (op 1209) 
s_mul_hi_i32 s22, s1, s14 // (op 1209) 
s_add_i32 s23, s22, s1 // (op 1209) 


s_ashr_i32 s1, s23, 31 // (op 1209) 
s_cmp_eq_i32 s14, 0 // (op 1209) 
s_mov_b32 s22, scc // (op 1209) 
s_cmp_eq_i32 s22, 0 // (op 1209) 
s_cselect_b32 s24, 0, -1 // (op 1209) 

s_and_b32 s22, s48, 31 // (op 1209) 
s_lshl_b32 s25, 1, s22 // (op 1209) 
s_add_i32 s49, s25, s24 // (op 1209) 


s_and_b32 s24, s1, s49 // (op 1209) 


s_add_i32 s1, s23, s24 // (op 1209) 


s_ashr_i32 s23, s1, s22 // (op 1209) 


s_lshl_b32 s1, s48, 24 // (op 1209) 
s_ashr_i32 s22, s1, 31 // (op 1209) 

s_xor_b32 s1, s23, s22 // (op 1209) 

s_sub_i32 s23, s1, s22 // (op 1209) 


s_mul_i32 s1, s47, s15 // (op 1209) 
s_add_u32 s24, s20, 16 // (op 1209) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1209) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1209) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1209) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1209) 

s_sub_i32 s22, s24, s1 // (op 1209) 

s_mul_i32 s24, s23, s22 // (op 1209) 


s_sub_u32 s25, s19, s24 // (op 1209) 


s_add_u32 s19, s25, s1 // (op 1209) 


s_cmp_eq_i32 s18, 0 // (op 1209) 
s_mov_b32 s1, scc // (op 1209) 

s_cmp_eq_i32 s1, 0 // (op 1209) 
s_cselect_b32 s18, s19, s0 // (op 1209) 



s_lshl_b32 s19, s18, 4 // (op 1209) 

s_mov_b32 s1, 0 // (op 1209) convert
s_mov_b32 s0, s19 // (op 1209) convert

s_ashr_i64 s[18:19], s[30:31], 8 // (op 1209) 
s_lshr_b32 s22, s18, 1 // (op 1209) 

v_lshl_add_u32 v15, s22, 9, v1 // (op 1209) 

v_mul_lo_u32 v29, v15, s27 // (op 1209) most significant: low of low * high
v_mul_hi_u32 v24, v15, s26 // (op 1209) most significant: high of low * low
v_mul_lo_u32 v28, v15, s26 // (op 1209) least significant: low of low * low
v_add_u32 v29, v29, v24 // (op 1209) most significant: sum


v_mov_b32 v15, s0 // (op 1209) 
v_mov_b32 v24, s1 // (op 1209) 
v_add_co_u32 v30, s[18:19], v15, v28 // (op 1209) least significant half
v_addc_co_u32 v31, s[18:19], v24, v29, s[18:19] // (op 1209) most significant half





s_mul_i32 s1, s26, s31 // (op 1211) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 1211) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 1211) least significant
s_mul_hi_u32 s19, s26, s30 // (op 1211) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1211) most significant: sum
s_add_u32 s1, s1, s19 // (op 1211) most significant: sum


s_mov_b32 s52, s28 // (op 1211) reg expression
s_mov_b32 s53, s29 // (op 1211) reg expression
s_mov_b32 s54, s0 // (op 1211) reg expression
s_mov_b32 s55, 131072 // (op 1211) call()

s_mul_hi_i32 s0, s16, s10 // (op 1277) 
v_lshrrev_b32 v15, 2, v1 // (op 1277) 
s_add_i32 s1, s0, s16 // (op 1277) 

v_lshlrev_b32 v24, 1, v15 // (op 1277) 

s_ashr_i32 s0, s1, 31 // (op 1277) 
v_and_b32 v15, -4, v1 // (op 1277) 
s_cmp_eq_i32 s10, 0 // (op 1277) 
s_mov_b32 s18, scc // (op 1277) 
v_sub_u32 v26, v1, v15 // (op 1277) 

s_cmp_eq_i32 s18, 0 // (op 1277) 
s_cselect_b32 s19, 0, -1 // (op 1277) 

v_lshlrev_b32 v15, 6, v26 // (op 1277) 

s_and_b32 s18, s11, 31 // (op 1277) 
s_lshl_b32 s22, 1, s18 // (op 1277) 
s_add_i32 s23, s22, s19 // (op 1277) 


s_and_b32 s19, s0, s23 // (op 1277) 


s_add_i32 s0, s1, s19 // (op 1277) 


s_ashr_i32 s1, s0, s18 // (op 1277) 


s_lshl_b32 s0, s11, 24 // (op 1277) 
s_ashr_i32 s18, s0, 31 // (op 1277) 

s_xor_b32 s0, s1, s18 // (op 1277) 

s_sub_i32 s1, s0, s18 // (op 1277) 


s_mul_hi_i32 s0, s1, s12 // (op 1277) 
s_add_i32 s18, s0, s1 // (op 1277) 

s_ashr_i32 s0, s18, 31 // (op 1277) 
s_cmp_eq_i32 s12, 0 // (op 1277) 
s_mov_b32 s19, scc // (op 1277) 
s_cmp_eq_i32 s19, 0 // (op 1277) 
s_cselect_b32 s22, 0, -1 // (op 1277) 

s_and_b32 s19, s13, 31 // (op 1277) 
s_lshl_b32 s23, 1, s19 // (op 1277) 
s_add_i32 s24, s23, s22 // (op 1277) 


s_and_b32 s25, s0, s24 // (op 1277) 


s_add_i32 s0, s18, s25 // (op 1277) 


s_ashr_i32 s18, s0, s19 // (op 1277) 


s_lshl_b32 s0, s13, 24 // (op 1277) 
s_ashr_i32 s19, s0, 31 // (op 1277) 

s_xor_b32 s0, s18, s19 // (op 1277) 

s_sub_i32 s18, s0, s19 // (op 1277) 


s_add_u32 s50, s6, 64 // (op 1277) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1277) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1277) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1277) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1277) 

s_mul_i32 s0, s18, s24 // (op 1277) 

s_sub_i32 s18, s1, s0 // (op 1277) 


s_mul_hi_i32 s19, s16, s9 // (op 1277) 
s_add_i32 s0, s19, s16 // (op 1277) 

s_ashr_i32 s1, s0, 31 // (op 1277) 
s_cmp_eq_i32 s9, 0 // (op 1277) 
s_mov_b32 s19, scc // (op 1277) 
s_cmp_eq_i32 s19, 0 // (op 1277) 
s_cselect_b32 s22, 0, -1 // (op 1277) 

s_and_b32 s19, s46, 31 // (op 1277) 
s_lshl_b32 s23, 1, s19 // (op 1277) 
s_add_i32 s49, s23, s22 // (op 1277) 


s_and_b32 s22, s1, s49 // (op 1277) 


s_add_i32 s1, s0, s22 // (op 1277) 


s_ashr_i32 s0, s1, s19 // (op 1277) 


s_lshl_b32 s1, s46, 24 // (op 1277) 
s_ashr_i32 s19, s1, 31 // (op 1277) 

s_xor_b32 s1, s0, s19 // (op 1277) 

s_sub_i32 s0, s1, s19 // (op 1277) 


s_mul_i32 s1, s15, s24 // (op 1277) 

s_mul_i32 s19, s8, s1 // (op 1277) 

s_mul_i32 s1, s0, s19 // (op 1277) 

s_sub_u32 s19, s16, s1 // (op 1277) 

s_mul_hi_i32 s1, s19, s14 // (op 1277) 
s_add_i32 s22, s1, s19 // (op 1277) 


s_ashr_i32 s1, s22, 31 // (op 1277) 
s_cmp_eq_i32 s14, 0 // (op 1277) 
s_mov_b32 s19, scc // (op 1277) 
s_cmp_eq_i32 s19, 0 // (op 1277) 
s_cselect_b32 s23, 0, -1 // (op 1277) 

s_and_b32 s19, s48, 31 // (op 1277) 
s_lshl_b32 s24, 1, s19 // (op 1277) 
s_add_i32 s25, s24, s23 // (op 1277) 


s_and_b32 s23, s1, s25 // (op 1277) 


s_add_i32 s1, s22, s23 // (op 1277) 


s_ashr_i32 s22, s1, s19 // (op 1277) 


s_lshl_b32 s1, s48, 24 // (op 1277) 
s_ashr_i32 s19, s1, 31 // (op 1277) 

s_xor_b32 s1, s22, s19 // (op 1277) 

s_sub_i32 s22, s1, s19 // (op 1277) 


s_cmp_eq_i32 s0, 0 // (op 1277) 
s_mov_b32 s1, scc // (op 1277) 

s_cmp_eq_i32 s1, 0 // (op 1277) 
s_cselect_b32 s0, s22, s18 // (op 1277) 



v_lshl_add_u32 v26, s0, 6, v24 // (op 1277) 


v_mul_lo_u32 v29, v26, s33 // (op 1277) most significant: low of low * high
v_mul_hi_u32 v24, v26, s32 // (op 1277) most significant: high of low * low
v_mul_lo_u32 v28, v26, s32 // (op 1277) least significant: low of low * low
v_add_u32 v29, v29, v24 // (op 1277) most significant: sum


s_ashr_i64 s[0:1], s[30:31], 8 // (op 1277) 
s_lshr_b32 s18, s0, 1 // (op 1277) 

v_lshl_add_u32 v24, s18, 9, v15 // (op 1277) 


v_mov_b32 v33, 0 // (op 1277) convert
v_mov_b32 v32, v24 // (op 1277) convert

v_add_co_u32 v34, s[0:1], v32, v28 // (op 1277) least significant half
v_addc_co_u32 v35, s[0:1], v33, v29, s[0:1] // (op 1277) most significant half



v_mul_lo_u32 v29, v35, 6 // (op 1277) most significant: low of high * low
v_mul_hi_u32 v15, v34, 6 // (op 1277) most significant: high of low * low
v_mul_lo_u32 v28, v34, 6 // (op 1277) least significant: low of low * low
v_add_u32 v29, v29, v15 // (op 1277) most significant: sum


v_lshrrev_b64 v[32:33], 3, v[28:29] // (op 1277) 

v_mov_b32 v15, v32 // (op 1277) convert

s_mul_i32 s1, s7, 6 // (op 1279) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 1279) least significant
s_mul_hi_u32 s18, s6, 6 // (op 1279) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1279) most significant: sum

s_mul_i32 s19, s32, s1 // (op 1279) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 1279) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 1279) least significant
s_mul_hi_u32 s23, s32, s0 // (op 1279) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 1279) most significant: sum
s_add_u32 s19, s19, s23 // (op 1279) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1279) 

s_mov_b32 s64, s34 // (op 1279) reg expression
s_mov_b32 s65, s35 // (op 1279) reg expression
s_mov_b32 s66, s0 // (op 1279) reg expression
s_mov_b32 s67, 131072 // (op 1279) call()

s_mul_hi_i32 s0, s16, s10 // (op 1341) 
v_and_b32 v24, -16, v1 // (op 1341) 
s_add_i32 s1, s0, s16 // (op 1341) 

v_sub_u32 v26, v1, v24 // (op 1341) 

s_ashr_i32 s0, s1, 31 // (op 1341) 
v_lshrrev_b32 v24, 4, v1 // (op 1341) 
s_cmp_eq_i32 s10, 0 // (op 1341) 
s_mov_b32 s18, scc // (op 1341) 
s_cmp_eq_i32 s18, 0 // (op 1341) 
s_cselect_b32 s19, 0, -1 // (op 1341) 

s_and_b32 s18, s11, 31 // (op 1341) 
s_lshl_b32 s22, 1, s18 // (op 1341) 
s_add_i32 s23, s22, s19 // (op 1341) 


s_and_b32 s19, s0, s23 // (op 1341) 


s_add_i32 s0, s1, s19 // (op 1341) 


s_ashr_i32 s1, s0, s18 // (op 1341) 


s_lshl_b32 s0, s11, 24 // (op 1341) 
s_ashr_i32 s18, s0, 31 // (op 1341) 

s_xor_b32 s0, s1, s18 // (op 1341) 

s_sub_i32 s1, s0, s18 // (op 1341) 


s_mul_hi_i32 s0, s1, s12 // (op 1341) 
s_add_i32 s18, s0, s1 // (op 1341) 

s_ashr_i32 s0, s18, 31 // (op 1341) 
s_cmp_eq_i32 s12, 0 // (op 1341) 
s_mov_b32 s19, scc // (op 1341) 
s_cmp_eq_i32 s19, 0 // (op 1341) 
s_cselect_b32 s22, 0, -1 // (op 1341) 

s_and_b32 s19, s13, 31 // (op 1341) 
s_lshl_b32 s23, 1, s19 // (op 1341) 
s_add_i32 s24, s23, s22 // (op 1341) 


s_and_b32 s25, s0, s24 // (op 1341) 


s_add_i32 s0, s18, s25 // (op 1341) 


s_ashr_i32 s18, s0, s19 // (op 1341) 


s_lshl_b32 s0, s13, 24 // (op 1341) 
s_ashr_i32 s19, s0, 31 // (op 1341) 

s_xor_b32 s0, s18, s19 // (op 1341) 

s_sub_i32 s18, s0, s19 // (op 1341) 


s_mul_i32 s0, s1, s15 // (op 1341) 

s_sub_u32 s1, s16, s0 // (op 1341) 

s_mul_i32 s0, s18, s15 // (op 1341) 
s_add_u32 s0, s0, s1 // (op 1341) 


s_mul_hi_i32 s1, s16, s9 // (op 1341) 
s_add_i32 s18, s1, s16 // (op 1341) 

s_ashr_i32 s1, s18, 31 // (op 1341) 
s_cmp_eq_i32 s9, 0 // (op 1341) 
s_mov_b32 s19, scc // (op 1341) 
s_cmp_eq_i32 s19, 0 // (op 1341) 
s_cselect_b32 s22, 0, -1 // (op 1341) 

s_and_b32 s19, s46, 31 // (op 1341) 
s_lshl_b32 s23, 1, s19 // (op 1341) 
s_add_i32 s24, s23, s22 // (op 1341) 


s_and_b32 s25, s1, s24 // (op 1341) 


s_add_i32 s1, s18, s25 // (op 1341) 


s_ashr_i32 s18, s1, s19 // (op 1341) 


s_lshl_b32 s1, s46, 24 // (op 1341) 
s_ashr_i32 s19, s1, 31 // (op 1341) 

s_xor_b32 s1, s18, s19 // (op 1341) 

s_sub_i32 s18, s1, s19 // (op 1341) 


s_add_u32 s50, s6, 64 // (op 1341) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1341) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1341) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1341) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1341) 

s_mul_i32 s1, s15, s24 // (op 1341) 

s_mul_i32 s19, s8, s1 // (op 1341) 

s_mul_i32 s1, s18, s19 // (op 1341) 

s_sub_u32 s19, s16, s1 // (op 1341) 

s_mov_b32 s1, s19 // (op 1341) 
s_mul_hi_i32 s22, s1, s14 // (op 1341) 
s_add_i32 s23, s22, s1 // (op 1341) 


s_ashr_i32 s1, s23, 31 // (op 1341) 
s_cmp_eq_i32 s14, 0 // (op 1341) 
s_mov_b32 s22, scc // (op 1341) 
s_cmp_eq_i32 s22, 0 // (op 1341) 
s_cselect_b32 s24, 0, -1 // (op 1341) 

s_and_b32 s22, s48, 31 // (op 1341) 
s_lshl_b32 s25, 1, s22 // (op 1341) 
s_add_i32 s49, s25, s24 // (op 1341) 


s_and_b32 s24, s1, s49 // (op 1341) 


s_add_i32 s1, s23, s24 // (op 1341) 


s_ashr_i32 s23, s1, s22 // (op 1341) 


s_lshl_b32 s1, s48, 24 // (op 1341) 
s_ashr_i32 s22, s1, 31 // (op 1341) 

s_xor_b32 s1, s23, s22 // (op 1341) 

s_sub_i32 s23, s1, s22 // (op 1341) 


s_mul_i32 s1, s47, s15 // (op 1341) 
s_add_u32 s24, s20, 16 // (op 1341) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1341) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1341) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1341) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1341) 

s_sub_i32 s22, s24, s1 // (op 1341) 

s_mul_i32 s24, s23, s22 // (op 1341) 


s_sub_u32 s25, s19, s24 // (op 1341) 


s_add_u32 s19, s25, s1 // (op 1341) 


s_cmp_eq_i32 s18, 0 // (op 1341) 
s_mov_b32 s1, scc // (op 1341) 

s_cmp_eq_i32 s1, 0 // (op 1341) 
s_cselect_b32 s18, s19, s0 // (op 1341) 



v_lshl_add_u32 v17, s18, 4, v26 // (op 1341) 


s_ashr_i64 s[0:1], s[30:31], 8 // (op 1341) 
s_lshr_b32 s18, s0, 1 // (op 1341) 

v_lshl_add_u32 v26, s18, 4, v24 // (op 1341) 


v_mul_lo_u32 v24, v26, s36 // (op 1341) 

v_add_u32 v26, v17, v24 // (op 1341) 


s_mul_i32 s1, s36, s41 // (op 1343) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1343) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1343) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1343) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1343) most significant: sum
s_add_u32 s1, s1, s19 // (op 1343) most significant: sum


s_mov_b32 s68, s38 // (op 1343) reg expression
s_mov_b32 s69, s39 // (op 1343) reg expression
s_mov_b32 s70, s0 // (op 1343) reg expression
s_mov_b32 s71, 131072 // (op 1343) call()

s_mul_hi_i32 s0, s16, s10 // (op 1405) 
v_lshrrev_b32 v24, 1, v1 // (op 1405) 
s_add_i32 s1, s0, s16 // (op 1405) 

v_and_b32 v17, -2, v1 // (op 1405) 
s_ashr_i32 s0, s1, 31 // (op 1405) 
v_sub_u32 v18, v1, v17 // (op 1405) 

s_cmp_eq_i32 s10, 0 // (op 1405) 
s_mov_b32 s18, scc // (op 1405) 
v_lshlrev_b32 v17, 2, v18 // (op 1405) 

s_cmp_eq_i32 s18, 0 // (op 1405) 
s_cselect_b32 s19, 0, -1 // (op 1405) 

s_and_b32 s18, s11, 31 // (op 1405) 
s_lshl_b32 s22, 1, s18 // (op 1405) 
s_add_i32 s23, s22, s19 // (op 1405) 


s_and_b32 s19, s0, s23 // (op 1405) 


s_add_i32 s0, s1, s19 // (op 1405) 


s_ashr_i32 s1, s0, s18 // (op 1405) 


s_lshl_b32 s0, s11, 24 // (op 1405) 
s_ashr_i32 s18, s0, 31 // (op 1405) 

s_xor_b32 s0, s1, s18 // (op 1405) 

s_sub_i32 s1, s0, s18 // (op 1405) 


s_mul_hi_i32 s0, s1, s12 // (op 1405) 
s_add_i32 s18, s0, s1 // (op 1405) 

s_ashr_i32 s0, s18, 31 // (op 1405) 
s_cmp_eq_i32 s12, 0 // (op 1405) 
s_mov_b32 s19, scc // (op 1405) 
s_cmp_eq_i32 s19, 0 // (op 1405) 
s_cselect_b32 s22, 0, -1 // (op 1405) 

s_and_b32 s19, s13, 31 // (op 1405) 
s_lshl_b32 s23, 1, s19 // (op 1405) 
s_add_i32 s24, s23, s22 // (op 1405) 


s_and_b32 s25, s0, s24 // (op 1405) 


s_add_i32 s0, s18, s25 // (op 1405) 


s_ashr_i32 s18, s0, s19 // (op 1405) 


s_lshl_b32 s0, s13, 24 // (op 1405) 
s_ashr_i32 s19, s0, 31 // (op 1405) 

s_xor_b32 s0, s18, s19 // (op 1405) 

s_sub_i32 s18, s0, s19 // (op 1405) 


s_add_u32 s50, s6, 64 // (op 1405) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1405) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1405) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1405) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1405) 

s_mul_i32 s0, s18, s24 // (op 1405) 

s_sub_i32 s18, s1, s0 // (op 1405) 


s_mul_hi_i32 s19, s16, s9 // (op 1405) 
s_add_i32 s0, s19, s16 // (op 1405) 

s_ashr_i32 s1, s0, 31 // (op 1405) 
s_cmp_eq_i32 s9, 0 // (op 1405) 
s_mov_b32 s19, scc // (op 1405) 
s_cmp_eq_i32 s19, 0 // (op 1405) 
s_cselect_b32 s22, 0, -1 // (op 1405) 

s_and_b32 s19, s46, 31 // (op 1405) 
s_lshl_b32 s23, 1, s19 // (op 1405) 
s_add_i32 s49, s23, s22 // (op 1405) 


s_and_b32 s22, s1, s49 // (op 1405) 


s_add_i32 s1, s0, s22 // (op 1405) 


s_ashr_i32 s0, s1, s19 // (op 1405) 


s_lshl_b32 s1, s46, 24 // (op 1405) 
s_ashr_i32 s19, s1, 31 // (op 1405) 

s_xor_b32 s1, s0, s19 // (op 1405) 

s_sub_i32 s0, s1, s19 // (op 1405) 


s_mul_i32 s1, s15, s24 // (op 1405) 

s_mul_i32 s19, s8, s1 // (op 1405) 

s_mul_i32 s1, s0, s19 // (op 1405) 

s_sub_u32 s19, s16, s1 // (op 1405) 

s_mul_hi_i32 s1, s19, s14 // (op 1405) 
s_add_i32 s22, s1, s19 // (op 1405) 


s_ashr_i32 s1, s22, 31 // (op 1405) 
s_cmp_eq_i32 s14, 0 // (op 1405) 
s_mov_b32 s19, scc // (op 1405) 
s_cmp_eq_i32 s19, 0 // (op 1405) 
s_cselect_b32 s23, 0, -1 // (op 1405) 

s_and_b32 s19, s48, 31 // (op 1405) 
s_lshl_b32 s24, 1, s19 // (op 1405) 
s_add_i32 s25, s24, s23 // (op 1405) 


s_and_b32 s23, s1, s25 // (op 1405) 


s_add_i32 s1, s22, s23 // (op 1405) 


s_ashr_i32 s22, s1, s19 // (op 1405) 


s_lshl_b32 s1, s48, 24 // (op 1405) 
s_ashr_i32 s19, s1, 31 // (op 1405) 

s_xor_b32 s1, s22, s19 // (op 1405) 

s_sub_i32 s22, s1, s19 // (op 1405) 


s_cmp_eq_i32 s0, 0 // (op 1405) 
s_mov_b32 s1, scc // (op 1405) 

s_cmp_eq_i32 s1, 0 // (op 1405) 
s_cselect_b32 s0, s22, s18 // (op 1405) 



v_lshl_add_u32 v18, s0, 6, v24 // (op 1405) 


v_mul_lo_u32 v24, v18, s42 // (op 1405) 

s_ashr_i64 s[0:1], s[30:31], 8 // (op 1405) 
s_lshr_b32 s18, s0, 1 // (op 1405) 

v_lshl_add_u32 v18, s18, 4, v17 // (op 1405) 


v_add_u32 v17, v18, v24 // (op 1405) 


s_mul_i32 s1, s42, s7 // (op 1407) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 1407) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 1407) least significant
s_mul_hi_u32 s19, s42, s6 // (op 1407) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1407) most significant: sum
s_add_u32 s1, s1, s19 // (op 1407) most significant: sum


s_mov_b32 s72, s44 // (op 1407) reg expression
s_mov_b32 s73, s45 // (op 1407) reg expression
s_mov_b32 s74, s0 // (op 1407) reg expression
s_mov_b32 s75, 131072 // (op 1407) call()

v_lshlrev_b32 v24, 4, v1 // (op 1498) 
s_nop 0
v_readfirstlane_b32 s0, v24 // (op 1498) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -4, v1 // (op 1550) 
v_sub_u32 v18, v1, v24 // (op 1550) 

v_lshlrev_b32 v24, 6, v18 // (op 1550) 

v_lshrrev_b32 v18, 2, v1 // (op 1550) 
v_lshl_add_u32 v19, v18, 9, v24 // (op 1550) 


v_mul_lo_u32 v18, v19, 6 // (op 1550) 

v_lshrrev_b32 v24, 3, v18 // (op 1550) 

v_and_b32 v18, -16, v1 // (op 1608) 
v_sub_u32 v19, v1, v18 // (op 1608) 

v_and_b32 v18, 4294967280, v1 // (op 1608) 
v_add_u32 v20, v18, v19 // (op 1608) 


v_and_b32 v18, -2, v1 // (op 1661) 
v_sub_u32 v19, v1, v18 // (op 1661) 

v_lshlrev_b32 v18, 2, v19 // (op 1661) 

v_lshrrev_b32 v19, 1, v1 // (op 1661) 
v_lshl_add_u32 v21, v19, 3, v18 // (op 1661) 


s_ashr_i64 s[18:19], s[30:31], 8 // (op 409) 
s_and_b32 s1, s18, 4294967294 // (op 409) 

s_ashr_i64 s[18:19], s[30:31], 8 // (op 408) 
s_mov_b32 s22, s18 // (op 408) convert

s_cmp_lt_u32 s1, s22 // (op 408) 










s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_408 // (op 408) Condition: Top (jump to RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_408 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_408:
s_barrier  // (op 1890) 
s_add_u32 s1, s1, 1 // (op 410) 

s_lshl_b64 s[18:19], s[26:27], 7 // (op 1180) 
v_mov_b64 v[18:19], s[18:19] // (op 1180) convert


v_mov_b64 v[22:23], v[30:31] // (op 1180) 
v_mov_b64 v[28:29], v[22:23] // (op 1180) reg expression
s_mov_b32 m0, s0 // (op 1180) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1180) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v18 // (op 1180) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 1180) most significant half

s_add_u32 m0, m0, 2048 // (op 1180) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1180) Load value direct to lds



s_lshl_b64 s[18:19], s[26:27], 8 // (op 1199) 
v_mov_b32 v14, s18 // (op 1199) 
v_mov_b32 v16, s19 // (op 1199) 
v_add_co_u32 v30, s[50:51], v30, v14 // (op 1199) least significant half
v_addc_co_u32 v31, s[50:51], v31, v16, s[50:51] // (op 1199) most significant half





s_mul_i32 s19, s33, 6 // (op 225) most significant: low of high * low
s_mul_i32 s18, s32, 6 // (op 225) least significant
s_mul_hi_u32 s22, s32, 6 // (op 225) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 225) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[18:19] // (op 225) 


v_mov_b32 v14, v15 // (op 225) 
v_mov_b32 v16, v14 // (op 225) 
buffer_load_dwordx4 v[32:35], v16, s[64:67], 0 offen // (op 225) Load value
buffer_load_dwordx2 v[36:37], v16, s[64:67], 0 offen offset:16 // (op 225) Load value
buffer_load_dwordx4 v[38:41], v16, s[64:67], 0 offen offset:24 // (op 225) Load value
buffer_load_dwordx2 v[42:43], v16, s[64:67], 0 offen offset:40 // (op 225) Load value
v_add_u32 v16, v16, v18 // (op 225) 
buffer_load_dwordx4 v[44:47], v16, s[64:67], 0 offen // (op 225) Load value
buffer_load_dwordx2 v[48:49], v16, s[64:67], 0 offen offset:16 // (op 225) Load value
buffer_load_dwordx4 v[50:53], v16, s[64:67], 0 offen offset:24 // (op 225) Load value
buffer_load_dwordx2 v[54:55], v16, s[64:67], 0 offen offset:40 // (op 225) Load value



v_add_u32 v15, 192, v15 // (op 1269) 


s_barrier  // (op 1909) 
v_mov_b32 v14, v26 // (op 226) 



v_mov_b32 v16, v14 // (op 226) reg expression

buffer_load_ubyte v25, v16, s[68:71], 0 offen offset: 0 // (op 226) Load value


s_lshl_b32 s18, s36, 3 // (op 1333) 
v_add_u32 v26, v26, s18 // (op 1333) 

s_waitcnt vmcnt(8)
ds_write_b128 v24, v[32:35] offset:8192 // (op 222) Store local data 


v_mov_b32 v14, v17 // (op 227) 

buffer_load_dword v16, v14, s[72:75], 0 offen // (op 227) Load value

v_add_u32 v17, 8, v17 // (op 1397) 
s_waitcnt vmcnt(8)
ds_write_b64 v24, v[36:37] offset:8208 // (op 222) Store local data 
s_barrier  // (op 1899) 
s_waitcnt vmcnt(7)
ds_write_b128 v24, v[38:41] offset:8216 // (op 222) Store local data 



ds_read_b64_tr_b8 v[56:57], v0 // (op 205) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:128 // (op 205) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:1024 // (op 205) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:1152 // (op 205) Transpose load local data 
s_waitcnt vmcnt(6)
ds_write_b64 v24, v[42:43] offset:8232 // (op 222) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v24, v[44:47] offset:8384 // (op 222) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v24, v[48:49] offset:8400 // (op 222) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v24, v[50:53] offset:8408 // (op 222) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v24, v[54:55] offset:8424 // (op 222) Store local data 

s_barrier  // (op 1930) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1920) 






ds_read_b128 v[32:35], v9 offset:8192 // (op 207) Load local data 
ds_read_b64 v[36:37], v9 offset:8208 // (op 207) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v20, v25 offset:20992 // (op 223) Store local data 

s_barrier  // (op 1950) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1940) 





ds_read_u8 v14, v4 offset:20992 // (op 209) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v21, v16 offset:20480 // (op 224) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1960) 



ds_read_u8 v16, v5 offset:20480 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[32:37], a[0:3], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 212) 





ds_read_b64_tr_b8 v[32:33], v0 offset:2048 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:2176 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:3072 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:3200 // (op 214) Transpose load local data 



ds_read_b128 v[40:43], v9 offset:8288 // (op 216) Load local data 
ds_read_b64 v[44:45], v9 offset:8304 // (op 216) Load local data 


ds_read_u8 v16, v4 offset:21056 // (op 218) Load local data 



ds_read_u8 v25, v5 offset:20484 // (op 220) Load local data 



ds_read_b128 v[46:49], v9 offset:14336 // (op 416) Load local data 
ds_read_b64 v[50:51], v9 offset:14352 // (op 416) Load local data 



ds_read_u8 v18, v5 offset:20736 // (op 420) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:63], v[46:51], a[4:7], v14, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 421) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[40:45], a[0:3], v16, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 221) 
ds_read_b128 v[46:49], v9 offset:14432 // (op 425) Load local data 


ds_read_b64 v[50:51], v9 offset:14448 // (op 425) Load local data 



ds_read_u8 v14, v5 offset:20740 // (op 429) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:39], v[46:51], a[4:7], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 430) 




s_ashr_i64 s[18:19], s[30:31], 8 // (op 408) 
s_mov_b32 s22, s18 // (op 408) convert

s_cmp_lt_u32 s1, s22 // (op 408) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_408 // (op 408) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_408 if true)
RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_408:























s_barrier  // (op 2053) 





















v_cmp_eq_f32 vcc, 0.00000, v7 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_157
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
s_branch RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_157
RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_157:


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
RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_157:










s_endpgm  // End of RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.LRR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__end:
.size RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_, .LRR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__end-RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
  .amdhsa_next_free_vgpr 140
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 34048 // lds bytes
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
    TransposeA: False
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
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int64, semantic: SizeFree0 },
            { type: address, semantic: AddressC },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
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
  - .name: RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
    .symbol: RR_GEMM_NN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 34048
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 130
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
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_0_size_0_15
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
      - .name: Tensor_0_stride_1_0
        .size: 8
        .offset: 80
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 88
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_1
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 112
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_6
        .size: 8
        .offset: 120
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 128
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_7
        .size: 8
        .offset: 136
        .value_kind: by_value
      - .name: Tensor_7_stride_1_9
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 152
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 160
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 164
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
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

