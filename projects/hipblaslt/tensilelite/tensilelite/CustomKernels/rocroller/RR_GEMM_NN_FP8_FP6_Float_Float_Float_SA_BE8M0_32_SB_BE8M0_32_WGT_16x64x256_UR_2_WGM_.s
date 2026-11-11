// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_,@function
RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 1461) 
s_mul_hi_i32 s1, s16, s10 // (op 1702) 
v_and_b32 v0, 48, v1 // (op 1517) 
v_and_b32 v3, 63, v1 // (op 1517) 
v_sub_u32 v4, v3, v0 // (op 1517) 

v_lshrrev_b32 v3, 1, v4 // (op 1517) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 1517) 


v_and_b32 v0, 1, v4 // (op 1517) 

v_lshlrev_b32 v3, 3, v0 // (op 1517) 

v_lshrrev_b32 v0, 7, v1 // (op 1517) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1517) 


v_add_u32 v0, v5, v4 // (op 1517) 


v_lshrrev_b32 v3, 7, v1 // (op 1628) 
v_and_b32 v4, 15, v1 // (op 1628) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 1628) 


v_and_b32 v3, 48, v1 // (op 1628) 
v_add_u32 v4, v3, v5 // (op 1628) 


v_and_b32 v3, 48, v1 // (op 1743) 
v_and_b32 v5, 63, v1 // (op 1743) 
v_sub_u32 v6, v5, v3 // (op 1743) 

v_lshrrev_b32 v5, 1, v6 // (op 1743) 
v_add_lshl_u32 v7, v3, v5, 4 // (op 1743) 


v_and_b32 v3, 1, v6 // (op 1743) 

v_lshlrev_b32 v5, 3, v3 // (op 1743) 

v_lshrrev_b32 v3, 7, v1 // (op 1743) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 1743) 


v_add_u32 v3, v7, v6 // (op 1743) 


v_lshrrev_b32 v5, 7, v1 // (op 1796) 
v_and_b32 v6, 15, v1 // (op 1796) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 1796) 


v_and_b32 v5, 48, v1 // (op 1796) 
v_add_u32 v6, v5, v7 // (op 1796) 


v_lshrrev_b32 v5, 7, v1 // (op 1461) 
s_add_i32 s18, s0, s16 // (op 1461) 

v_and_b32 v7, 63, v1 // (op 1461) 
s_ashr_i32 s0, s18, 31 // (op 1461) 
v_lshrrev_b32 v8, 4, v7 // (op 1461) 
s_cmp_eq_i32 s10, 0 // (op 1461) 
s_mov_b32 s19, scc // (op 1461) 
v_lshlrev_b32 v9, 2, v8 // (op 1461) 

s_cmp_eq_i32 s19, 0 // (op 1461) 
s_cselect_b32 s51, 0, -1 // (op 1461) 

v_lshl_add_u32 v8, v5, 4, v9 // (op 1461) 


s_and_b32 s19, s11, 31 // (op 1461) 
v_lshrrev_b32 v5, 2, v7 // (op 1461) 

v_and_b32 v7, 3, v5 // (op 1461) 

v_lshrrev_b32 v5, 6, v1 // (op 1575) 
v_and_b32 v9, 1, v5 // (op 1575) 

v_and_b32 v5, 15, v1 // (op 1575) 
v_lshl_add_u32 v10, v9, 4, v5 // (op 1575) 


v_and_b32 v5, 63, v1 // (op 1575) 
v_lshrrev_b32 v9, 4, v5 // (op 1575) 

v_lshlrev_b32 v5, 5, v9 // (op 1575) 

v_lshl_add_u32 v9, v10, 8, v5 // (op 1575) 


v_mul_lo_u32 v5, v9, 6 // (op 1575) 

v_lshrrev_b32 v9, 3, v5 // (op 1575) 

v_lshrrev_b32 v5, 6, v1 // (op 1686) 
v_and_b32 v10, 1, v5 // (op 1686) 

v_and_b32 v5, 15, v1 // (op 1686) 
v_lshl_add_u32 v11, v10, 4, v5 // (op 1686) 


v_and_b32 v5, 63, v1 // (op 1686) 
v_lshrrev_b32 v10, 4, v5 // (op 1686) 

v_lshl_add_u32 v5, v11, 3, v10 // (op 1686) 


v_lshrrev_b32 v10, 7, v1 // (op 1702) 
v_and_b32 v11, 63, v1 // (op 1702) 
v_lshrrev_b32 v12, 6, v1 // (op 1772) 
v_and_b32 v13, 1, v12 // (op 1772) 

v_and_b32 v12, 15, v1 // (op 1772) 
v_lshl_add_u32 v14, v13, 4, v12 // (op 1772) 


v_and_b32 v12, 63, v1 // (op 1772) 
v_lshrrev_b32 v13, 4, v12 // (op 1772) 

v_lshlrev_b32 v12, 5, v13 // (op 1772) 

v_lshl_add_u32 v13, v14, 8, v12 // (op 1772) 


v_mul_lo_u32 v12, v13, 6 // (op 1772) 

v_lshrrev_b32 v13, 3, v12 // (op 1772) 

v_lshrrev_b32 v12, 6, v1 // (op 1825) 
v_and_b32 v14, 1, v12 // (op 1825) 

v_and_b32 v12, 15, v1 // (op 1825) 
v_lshl_add_u32 v15, v14, 4, v12 // (op 1825) 


v_and_b32 v12, 63, v1 // (op 1825) 
v_lshrrev_b32 v14, 4, v12 // (op 1825) 

v_lshl_add_u32 v12, v15, 3, v14 // (op 1825) 


s_lshl_b32 s52, 1, s19 // (op 1461) 
v_and_b32 v14, 3, v1 // (op 1461) 
s_add_i32 s53, s52, s51 // (op 1461) 


s_and_b32 s51, s0, s53 // (op 1461) 


s_add_i32 s0, s1, s16 // (op 1702) 

s_ashr_i32 s1, s0, 31 // (op 1702) 
v_lshl_add_u32 v15, v7, 2, v14 // (op 1461) 


s_add_i32 s52, s18, s51 // (op 1461) 


v_lshrrev_b32 v7, 6, v1 // (op 1461) 
s_ashr_i32 s18, s52, s19 // (op 1461) 


v_and_b32 v14, 1, v7 // (op 1461) 

s_lshl_b32 s19, s11, 24 // (op 1461) 
v_lshl_add_u32 v7, v14, 4, v15 // (op 1461) 


s_ashr_i32 s51, s19, 31 // (op 1461) 

s_xor_b32 s19, s18, s51 // (op 1461) 

s_sub_i32 s18, s19, s51 // (op 1461) 


s_mul_hi_i32 s19, s18, s12 // (op 1461) 
s_add_i32 s51, s19, s18 // (op 1461) 

s_ashr_i32 s19, s51, 31 // (op 1461) 
s_cmp_eq_i32 s12, 0 // (op 1461) 
s_mov_b32 s52, scc // (op 1461) 
s_cmp_eq_i32 s52, 0 // (op 1461) 
s_cselect_b32 s53, 0, -1 // (op 1461) 

s_and_b32 s52, s13, 31 // (op 1461) 
v_lshrrev_b32 v14, 4, v11 // (op 1702) 
s_cmp_eq_i32 s10, 0 // (op 1702) 
s_mov_b32 s54, scc // (op 1702) 
v_lshlrev_b32 v15, 2, v14 // (op 1702) 

s_cmp_eq_i32 s54, 0 // (op 1702) 
s_cselect_b32 s55, 0, -1 // (op 1702) 

s_lshl_b32 s54, 1, s52 // (op 1461) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 1702) 


v_lshrrev_b32 v10, 2, v11 // (op 1702) 

s_add_i32 s56, s54, s53 // (op 1461) 


s_and_b32 s53, s19, s56 // (op 1461) 


s_add_i32 s19, s51, s53 // (op 1461) 


s_ashr_i32 s51, s19, s52 // (op 1461) 


s_lshl_b32 s19, s13, 24 // (op 1461) 
s_ashr_i32 s52, s19, 31 // (op 1461) 

s_xor_b32 s19, s51, s52 // (op 1461) 

s_sub_i32 s51, s19, s52 // (op 1461) 


s_mul_i32 s19, s18, s15 // (op 1461) 
s_sub_u32 s52, s16, s19 // (op 1461) 

s_mul_i32 s19, s51, s15 // (op 1461) 
s_add_u32 s19, s19, s52 // (op 1461) 

s_mul_hi_i32 s52, s16, s9 // (op 1461) 
s_add_i32 s53, s52, s16 // (op 1461) 

s_ashr_i32 s52, s53, 31 // (op 1461) 
s_cmp_eq_i32 s9, 0 // (op 1461) 
s_mov_b32 s54, scc // (op 1461) 
s_cmp_eq_i32 s54, 0 // (op 1461) 
s_cselect_b32 s56, 0, -1 // (op 1461) 

s_and_b32 s54, s11, 31 // (op 1702) 
v_and_b32 v11, 3, v10 // (op 1702) 

s_lshl_b32 s57, 1, s54 // (op 1702) 
v_and_b32 v10, 3, v1 // (op 1702) 
s_add_i32 s58, s57, s55 // (op 1702) 


s_and_b32 s55, s1, s58 // (op 1702) 


v_lshl_add_u32 v15, v11, 2, v10 // (op 1702) 


s_add_i32 s1, s0, s55 // (op 1702) 


v_lshrrev_b32 v10, 6, v1 // (op 1702) 
s_ashr_i32 s0, s1, s54 // (op 1702) 


v_and_b32 v11, 1, v10 // (op 1702) 

s_lshl_b32 s1, s11, 24 // (op 1702) 
v_lshl_add_u32 v10, v11, 4, v15 // (op 1702) 


s_ashr_i32 s54, s1, 31 // (op 1702) 

s_xor_b32 s1, s0, s54 // (op 1702) 

s_sub_i32 s0, s1, s54 // (op 1702) 


s_mul_hi_i32 s1, s0, s12 // (op 1702) 
s_add_i32 s54, s1, s0 // (op 1702) 

s_ashr_i32 s1, s54, 31 // (op 1702) 
s_cmp_eq_i32 s12, 0 // (op 1702) 
s_mov_b32 s55, scc // (op 1702) 
s_cmp_eq_i32 s55, 0 // (op 1702) 
s_cselect_b32 s57, 0, -1 // (op 1702) 

s_and_b32 s55, s13, 31 // (op 1702) 
s_lshl_b32 s58, 1, s55 // (op 1702) 
s_add_i32 s59, s58, s57 // (op 1702) 


s_and_b32 s57, s1, s59 // (op 1702) 


s_add_i32 s1, s54, s57 // (op 1702) 


s_ashr_i32 s54, s1, s55 // (op 1702) 


s_lshl_b32 s1, s13, 24 // (op 1702) 
s_ashr_i32 s55, s1, 31 // (op 1702) 

s_xor_b32 s1, s54, s55 // (op 1702) 

s_sub_i32 s54, s1, s55 // (op 1702) 


s_mul_i32 s1, s0, s15 // (op 1702) 
s_sub_u32 s55, s16, s1 // (op 1702) 

s_mul_i32 s1, s54, s15 // (op 1702) 
s_add_u32 s1, s1, s55 // (op 1702) 

s_mul_hi_i32 s55, s16, s9 // (op 1702) 
s_add_i32 s57, s55, s16 // (op 1702) 

s_ashr_i32 s55, s57, 31 // (op 1702) 
s_cmp_eq_i32 s9, 0 // (op 1702) 
s_mov_b32 s58, scc // (op 1702) 
s_cmp_eq_i32 s58, 0 // (op 1702) 
s_cselect_b32 s59, 0, -1 // (op 1702) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s46, 31 // (op 1461) 
s_lshl_b32 s60, 1, s58 // (op 1461) 
s_add_i32 s61, s60, s56 // (op 1461) 


s_and_b32 s56, s52, s61 // (op 1461) 


s_add_i32 s52, s53, s56 // (op 1461) 


s_ashr_i32 s53, s52, s58 // (op 1461) 


s_lshl_b32 s52, s46, 24 // (op 1461) 
s_ashr_i32 s56, s52, 31 // (op 1461) 

s_xor_b32 s52, s53, s56 // (op 1461) 

s_sub_i32 s53, s52, s56 // (op 1461) 


s_and_b32 s52, s46, 31 // (op 1702) 
s_lshl_b32 s56, 1, s52 // (op 1702) 
s_add_i32 s58, s56, s59 // (op 1702) 


s_and_b32 s56, s55, s58 // (op 1702) 


s_add_i32 s55, s57, s56 // (op 1702) 


s_ashr_i32 s56, s55, s52 // (op 1702) 


s_lshl_b32 s52, s46, 24 // (op 1702) 
s_ashr_i32 s55, s52, 31 // (op 1702) 

s_xor_b32 s52, s56, s55 // (op 1702) 

s_sub_i32 s56, s52, s55 // (op 1702) 


s_add_u32 s58, s6, 64 // (op 1461) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 1461) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1461) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1461) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 1461) 

s_mul_i32 s52, s15, s58 // (op 1461) 
s_mul_i32 s55, s8, s52 // (op 1461) 

s_mul_i32 s52, s53, s55 // (op 1461) 

s_sub_u32 s55, s16, s52 // (op 1461) 

s_mov_b32 s52, s55 // (op 1461) 
s_mul_hi_i32 s57, s52, s14 // (op 1461) 
s_add_i32 s60, s57, s52 // (op 1461) 


s_ashr_i32 s52, s60, 31 // (op 1461) 
s_cmp_eq_i32 s14, 0 // (op 1461) 
s_mov_b32 s57, scc // (op 1461) 
s_cmp_eq_i32 s57, 0 // (op 1461) 
s_cselect_b32 s61, 0, -1 // (op 1461) 

s_and_b32 s57, s48, 31 // (op 1461) 
s_lshl_b32 s62, 1, s57 // (op 1461) 
s_add_i32 s63, s62, s61 // (op 1461) 


s_and_b32 s61, s52, s63 // (op 1461) 


s_add_i32 s52, s60, s61 // (op 1461) 


s_ashr_i32 s60, s52, s57 // (op 1461) 


s_lshl_b32 s52, s48, 24 // (op 1461) 
s_ashr_i32 s57, s52, 31 // (op 1461) 

s_xor_b32 s52, s60, s57 // (op 1461) 

s_sub_i32 s60, s52, s57 // (op 1461) 


s_mul_i32 s52, s47, s15 // (op 1461) 
s_add_u32 s62, s20, 16 // (op 1461) Start of Int64 add, locking SCC
s_addc_u32 s63, s21, 0 // (op 1461) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1461) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1461) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 4 // (op 1461) 

s_sub_i32 s57, s62, s52 // (op 1461) 

s_mul_i32 s61, s60, s57 // (op 1461) 

s_sub_u32 s57, s55, s61 // (op 1461) 


s_add_u32 s55, s57, s52 // (op 1461) 


s_cmp_eq_i32 s53, 0 // (op 1461) 
s_mov_b32 s52, scc // (op 1461) 

s_cmp_eq_i32 s52, 0 // (op 1461) 
s_cselect_b32 s53, s55, s19 // (op 1461) 


s_mul_i32 s19, s51, s58 // (op 1461) 


v_lshl_add_u32 v11, s53, 4, v8 // (op 1461) 


s_sub_i32 s51, s18, s19 // (op 1461) 


s_cmp_eq_i32 s52, 0 // (op 1461) 
s_cselect_b32 s53, s60, s51 // (op 1461) 



v_lshl_add_u32 v8, s53, 6, v7 // (op 1461) 


v_mul_lo_u32 v7, v8, s2 // (op 1461) 

v_add_lshl_u32 v8, v11, v7, 2 // (op 1461) 


s_add_u32 s18, s6, 64 // (op 1702) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 1702) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 1702) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 1702) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 6 // (op 1702) 

s_mul_i32 s55, s15, s18 // (op 1702) 
s_mul_i32 s51, s8, s55 // (op 1702) 

s_mul_i32 s55, s56, s51 // (op 1702) 

s_sub_u32 s51, s16, s55 // (op 1702) 

s_mov_b32 s55, s51 // (op 1702) 
s_mul_hi_i32 s52, s55, s14 // (op 1702) 
s_add_i32 s53, s52, s55 // (op 1702) 


s_ashr_i32 s52, s53, 31 // (op 1702) 
s_cmp_eq_i32 s14, 0 // (op 1702) 
s_mov_b32 s55, scc // (op 1702) 
s_cmp_eq_i32 s55, 0 // (op 1702) 
s_cselect_b32 s57, 0, -1 // (op 1702) 

s_and_b32 s55, s48, 31 // (op 1702) 
s_lshl_b32 s58, 1, s55 // (op 1702) 
s_add_i32 s59, s58, s57 // (op 1702) 


s_and_b32 s57, s52, s59 // (op 1702) 


s_add_i32 s52, s53, s57 // (op 1702) 


s_ashr_i32 s53, s52, s55 // (op 1702) 


s_lshl_b32 s52, s48, 24 // (op 1702) 
s_ashr_i32 s55, s52, 31 // (op 1702) 

s_xor_b32 s52, s53, s55 // (op 1702) 

s_sub_i32 s53, s52, s55 // (op 1702) 


s_mul_i32 s52, s47, s15 // (op 1702) 
s_add_u32 s58, s20, 16 // (op 1702) Start of Int64 add, locking SCC
s_addc_u32 s59, s21, 0 // (op 1702) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1702) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1702) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 4 // (op 1702) 

s_sub_i32 s55, s58, s52 // (op 1702) 

s_mul_i32 s57, s53, s55 // (op 1702) 

s_sub_u32 s55, s51, s57 // (op 1702) 


s_add_u32 s51, s55, s52 // (op 1702) 


s_cmp_eq_i32 s56, 0 // (op 1702) 
s_mov_b32 s52, scc // (op 1702) 

s_cmp_eq_i32 s52, 0 // (op 1702) 
s_cselect_b32 s55, s51, s1 // (op 1702) 


s_mul_i32 s1, s54, s18 // (op 1702) 


v_lshl_add_u32 v7, s55, 4, v14 // (op 1702) 


s_sub_i32 s51, s0, s1 // (op 1702) 


s_cmp_eq_i32 s52, 0 // (op 1702) 
s_cselect_b32 s0, s53, s51 // (op 1702) 



v_lshl_add_u32 v11, s0, 6, v10 // (op 1702) 


v_mul_lo_u32 v10, v11, s4 // (op 1702) 

v_add_lshl_u32 v11, v7, v10, 2 // (op 1702) 


s_mul_i32 s1, s2, s7 // (op 1463) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 1463) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 1463) least significant
s_mul_hi_u32 s19, s2, s6 // (op 1463) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1463) most significant: sum
s_add_u32 s1, s1, s19 // (op 1463) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1463) 

s_mul_i32 s1, s4, s7 // (op 1704) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 1704) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 1704) least significant
s_mul_hi_u32 s52, s4, s6 // (op 1704) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1704) most significant: sum
s_add_u32 s1, s1, s52 // (op 1704) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 1704) 

s_mov_b32 s56, s22 // (op 1463) reg expression
s_mov_b32 s57, s23 // (op 1463) reg expression
s_mov_b32 s58, s18 // (op 1463) reg expression
s_mov_b32 s59, 131072 // (op 1463) call()

s_mov_b32 s60, s24 // (op 1704) reg expression

s_mov_b32 s61, s25 // (op 1704) reg expression
s_mov_b32 s62, s52 // (op 1704) reg expression
s_mov_b32 s63, 131072 // (op 1704) call()


v_mov_b32 v7, s49 // (op 2) Move value
v_mov_b32 v10, s50 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 398) call()
v_accvgpr_write a5, 0.00000 // (op 398) call()
v_accvgpr_write a6, 0.00000 // (op 398) call()
v_accvgpr_write a7, 0.00000 // (op 398) call()
s_mul_hi_i32 s0, s16, s10 // (op 1228) 
s_add_i32 s1, s0, s16 // (op 1228) 

s_ashr_i32 s0, s1, 31 // (op 1228) 
s_cmp_eq_i32 s10, 0 // (op 1228) 
s_mov_b32 s18, scc // (op 1228) 
s_cmp_eq_i32 s18, 0 // (op 1228) 
s_cselect_b32 s19, 0, -1 // (op 1228) 

s_and_b32 s18, s11, 31 // (op 1228) 
s_lshl_b32 s22, 1, s18 // (op 1228) 
s_add_i32 s23, s22, s19 // (op 1228) 


s_and_b32 s19, s0, s23 // (op 1228) 


s_add_i32 s0, s1, s19 // (op 1228) 


s_ashr_i32 s1, s0, s18 // (op 1228) 


s_lshl_b32 s0, s11, 24 // (op 1228) 
s_ashr_i32 s18, s0, 31 // (op 1228) 

s_xor_b32 s0, s1, s18 // (op 1228) 

s_sub_i32 s1, s0, s18 // (op 1228) 


s_mul_hi_i32 s0, s1, s12 // (op 1228) 
s_add_i32 s18, s0, s1 // (op 1228) 

s_ashr_i32 s0, s18, 31 // (op 1228) 
s_cmp_eq_i32 s12, 0 // (op 1228) 
s_mov_b32 s19, scc // (op 1228) 
s_cmp_eq_i32 s19, 0 // (op 1228) 
s_cselect_b32 s22, 0, -1 // (op 1228) 

s_and_b32 s19, s13, 31 // (op 1228) 
s_lshl_b32 s23, 1, s19 // (op 1228) 
s_add_i32 s24, s23, s22 // (op 1228) 


s_and_b32 s25, s0, s24 // (op 1228) 


s_add_i32 s0, s18, s25 // (op 1228) 


s_ashr_i32 s18, s0, s19 // (op 1228) 


s_lshl_b32 s0, s13, 24 // (op 1228) 
s_ashr_i32 s19, s0, 31 // (op 1228) 

s_xor_b32 s0, s18, s19 // (op 1228) 

s_sub_i32 s18, s0, s19 // (op 1228) 


s_mul_i32 s0, s1, s15 // (op 1228) 

s_sub_u32 s1, s16, s0 // (op 1228) 

s_mul_i32 s0, s18, s15 // (op 1228) 
s_add_u32 s0, s0, s1 // (op 1228) 


s_mul_hi_i32 s1, s16, s9 // (op 1228) 
s_add_i32 s18, s1, s16 // (op 1228) 

s_ashr_i32 s1, s18, 31 // (op 1228) 
s_cmp_eq_i32 s9, 0 // (op 1228) 
s_mov_b32 s19, scc // (op 1228) 
s_cmp_eq_i32 s19, 0 // (op 1228) 
s_cselect_b32 s22, 0, -1 // (op 1228) 

s_and_b32 s19, s46, 31 // (op 1228) 
s_lshl_b32 s23, 1, s19 // (op 1228) 
s_add_i32 s24, s23, s22 // (op 1228) 


s_and_b32 s25, s1, s24 // (op 1228) 


s_add_i32 s1, s18, s25 // (op 1228) 


s_ashr_i32 s18, s1, s19 // (op 1228) 


s_lshl_b32 s1, s46, 24 // (op 1228) 
s_ashr_i32 s19, s1, 31 // (op 1228) 

s_xor_b32 s1, s18, s19 // (op 1228) 

s_sub_i32 s18, s1, s19 // (op 1228) 


s_add_u32 s22, s6, 64 // (op 1228) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1228) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1228) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1228) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1228) 

s_mul_i32 s1, s15, s22 // (op 1228) 

s_mul_i32 s19, s8, s1 // (op 1228) 

s_mul_i32 s1, s18, s19 // (op 1228) 

s_sub_u32 s19, s16, s1 // (op 1228) 

s_mov_b32 s1, s19 // (op 1228) 
s_mul_hi_i32 s22, s1, s14 // (op 1228) 
s_add_i32 s23, s22, s1 // (op 1228) 


s_ashr_i32 s1, s23, 31 // (op 1228) 
s_cmp_eq_i32 s14, 0 // (op 1228) 
s_mov_b32 s22, scc // (op 1228) 
s_cmp_eq_i32 s22, 0 // (op 1228) 
s_cselect_b32 s24, 0, -1 // (op 1228) 

s_and_b32 s22, s48, 31 // (op 1228) 
s_lshl_b32 s25, 1, s22 // (op 1228) 
s_add_i32 s49, s25, s24 // (op 1228) 


s_and_b32 s24, s1, s49 // (op 1228) 


s_add_i32 s1, s23, s24 // (op 1228) 


s_ashr_i32 s23, s1, s22 // (op 1228) 


s_lshl_b32 s1, s48, 24 // (op 1228) 
s_ashr_i32 s22, s1, 31 // (op 1228) 

s_xor_b32 s1, s23, s22 // (op 1228) 

s_sub_i32 s23, s1, s22 // (op 1228) 


s_mul_i32 s1, s47, s15 // (op 1228) 
s_add_u32 s24, s20, 16 // (op 1228) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1228) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 1228) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 1228) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 4 // (op 1228) 

s_sub_i32 s22, s24, s1 // (op 1228) 

s_mul_i32 s24, s23, s22 // (op 1228) 


s_sub_u32 s25, s19, s24 // (op 1228) 


s_add_u32 s19, s25, s1 // (op 1228) 


s_cmp_eq_i32 s18, 0 // (op 1228) 
s_mov_b32 s1, scc // (op 1228) 

s_cmp_eq_i32 s1, 0 // (op 1228) 
s_cselect_b32 s18, s19, s0 // (op 1228) 



s_lshl_b32 s19, s18, 4 // (op 1228) 

s_mov_b32 s1, 0 // (op 1228) convert
s_mov_b32 s0, s19 // (op 1228) convert

v_mul_lo_u32 v15, v1, s27 // (op 1228) most significant: low of low * high
v_mul_hi_u32 v16, v1, s26 // (op 1228) most significant: high of low * low
v_mul_lo_u32 v14, v1, s26 // (op 1228) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 1228) most significant: sum

v_mov_b32 v16, s0 // (op 1228) 
v_mov_b32 v17, s1 // (op 1228) 
v_add_co_u32 v18, s[18:19], v16, v14 // (op 1228) least significant half
v_addc_co_u32 v19, s[18:19], v17, v15, s[18:19] // (op 1228) most significant half





s_mul_i32 s1, s26, s31 // (op 1230) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 1230) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 1230) least significant
s_mul_hi_u32 s19, s26, s30 // (op 1230) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1230) most significant: sum
s_add_u32 s1, s1, s19 // (op 1230) most significant: sum


s_mov_b32 s52, s28 // (op 1230) reg expression
s_mov_b32 s53, s29 // (op 1230) reg expression
s_mov_b32 s54, s0 // (op 1230) reg expression
s_mov_b32 s55, 131072 // (op 1230) call()

s_mul_hi_i32 s0, s16, s10 // (op 1247) 
s_add_i32 s1, s0, s16 // (op 1247) 

s_ashr_i32 s0, s1, 31 // (op 1247) 
s_cmp_eq_i32 s10, 0 // (op 1247) 
s_mov_b32 s18, scc // (op 1247) 
s_cmp_eq_i32 s18, 0 // (op 1247) 
s_cselect_b32 s19, 0, -1 // (op 1247) 

s_and_b32 s18, s11, 31 // (op 1247) 
s_lshl_b32 s22, 1, s18 // (op 1247) 
s_add_i32 s23, s22, s19 // (op 1247) 


s_and_b32 s19, s0, s23 // (op 1247) 


s_add_i32 s0, s1, s19 // (op 1247) 


s_ashr_i32 s1, s0, s18 // (op 1247) 


s_lshl_b32 s0, s11, 24 // (op 1247) 
s_ashr_i32 s18, s0, 31 // (op 1247) 

s_xor_b32 s0, s1, s18 // (op 1247) 

s_sub_i32 s1, s0, s18 // (op 1247) 


s_mul_hi_i32 s0, s1, s12 // (op 1247) 
s_add_i32 s18, s0, s1 // (op 1247) 

s_ashr_i32 s0, s18, 31 // (op 1247) 
s_cmp_eq_i32 s12, 0 // (op 1247) 
s_mov_b32 s19, scc // (op 1247) 
s_cmp_eq_i32 s19, 0 // (op 1247) 
s_cselect_b32 s22, 0, -1 // (op 1247) 

s_and_b32 s19, s13, 31 // (op 1247) 
s_lshl_b32 s23, 1, s19 // (op 1247) 
s_add_i32 s24, s23, s22 // (op 1247) 


s_and_b32 s25, s0, s24 // (op 1247) 


s_add_i32 s0, s18, s25 // (op 1247) 


s_ashr_i32 s18, s0, s19 // (op 1247) 


s_lshl_b32 s0, s13, 24 // (op 1247) 
s_ashr_i32 s19, s0, 31 // (op 1247) 

s_xor_b32 s0, s18, s19 // (op 1247) 

s_sub_i32 s18, s0, s19 // (op 1247) 


s_mul_i32 s0, s1, s15 // (op 1247) 

s_sub_u32 s1, s16, s0 // (op 1247) 

s_mul_i32 s0, s18, s15 // (op 1247) 
s_add_u32 s0, s0, s1 // (op 1247) 


s_mul_hi_i32 s1, s16, s9 // (op 1247) 
s_add_i32 s18, s1, s16 // (op 1247) 

s_ashr_i32 s1, s18, 31 // (op 1247) 
s_cmp_eq_i32 s9, 0 // (op 1247) 
s_mov_b32 s19, scc // (op 1247) 
s_cmp_eq_i32 s19, 0 // (op 1247) 
s_cselect_b32 s22, 0, -1 // (op 1247) 

s_and_b32 s19, s46, 31 // (op 1247) 
s_lshl_b32 s23, 1, s19 // (op 1247) 
s_add_i32 s24, s23, s22 // (op 1247) 


s_and_b32 s25, s1, s24 // (op 1247) 


s_add_i32 s1, s18, s25 // (op 1247) 


s_ashr_i32 s18, s1, s19 // (op 1247) 


s_lshl_b32 s1, s46, 24 // (op 1247) 
s_ashr_i32 s19, s1, 31 // (op 1247) 

s_xor_b32 s1, s18, s19 // (op 1247) 

s_sub_i32 s18, s1, s19 // (op 1247) 


s_add_u32 s50, s6, 64 // (op 1247) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1247) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1247) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1247) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1247) 

s_mul_i32 s1, s15, s24 // (op 1247) 

s_mul_i32 s19, s8, s1 // (op 1247) 

s_mul_i32 s1, s18, s19 // (op 1247) 

s_sub_u32 s19, s16, s1 // (op 1247) 

s_mov_b32 s1, s19 // (op 1247) 
s_mul_hi_i32 s22, s1, s14 // (op 1247) 
s_add_i32 s23, s22, s1 // (op 1247) 


s_ashr_i32 s1, s23, 31 // (op 1247) 
s_cmp_eq_i32 s14, 0 // (op 1247) 
s_mov_b32 s22, scc // (op 1247) 
s_cmp_eq_i32 s22, 0 // (op 1247) 
s_cselect_b32 s24, 0, -1 // (op 1247) 

s_and_b32 s22, s48, 31 // (op 1247) 
s_lshl_b32 s25, 1, s22 // (op 1247) 
s_add_i32 s49, s25, s24 // (op 1247) 


s_and_b32 s24, s1, s49 // (op 1247) 


s_add_i32 s1, s23, s24 // (op 1247) 


s_ashr_i32 s23, s1, s22 // (op 1247) 


s_lshl_b32 s1, s48, 24 // (op 1247) 
s_ashr_i32 s22, s1, 31 // (op 1247) 

s_xor_b32 s1, s23, s22 // (op 1247) 

s_sub_i32 s23, s1, s22 // (op 1247) 


s_mul_i32 s1, s47, s15 // (op 1247) 
s_add_u32 s24, s20, 16 // (op 1247) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1247) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1247) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1247) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1247) 

s_sub_i32 s22, s24, s1 // (op 1247) 

s_mul_i32 s24, s23, s22 // (op 1247) 


s_sub_u32 s25, s19, s24 // (op 1247) 


s_add_u32 s19, s25, s1 // (op 1247) 


s_cmp_eq_i32 s18, 0 // (op 1247) 
s_mov_b32 s1, scc // (op 1247) 

s_cmp_eq_i32 s1, 0 // (op 1247) 
s_cselect_b32 s18, s19, s0 // (op 1247) 



s_lshl_b32 s19, s18, 4 // (op 1247) 

s_mov_b32 s1, 0 // (op 1247) convert
s_mov_b32 s0, s19 // (op 1247) convert

v_mul_lo_u32 v15, v1, s27 // (op 1247) most significant: low of low * high
v_mul_hi_u32 v16, v1, s26 // (op 1247) most significant: high of low * low
v_mul_lo_u32 v14, v1, s26 // (op 1247) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 1247) most significant: sum

v_mov_b32 v16, s0 // (op 1247) 
v_mov_b32 v17, s1 // (op 1247) 
v_add_co_u32 v20, s[18:19], v16, v14 // (op 1247) least significant half
v_addc_co_u32 v21, s[18:19], v17, v15, s[18:19] // (op 1247) most significant half





s_mul_i32 s1, s26, s31 // (op 1249) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 1249) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 1249) least significant
s_mul_hi_u32 s19, s26, s30 // (op 1249) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1249) most significant: sum
s_add_u32 s1, s1, s19 // (op 1249) most significant: sum


s_mov_b32 s52, s28 // (op 1249) reg expression
s_mov_b32 s53, s29 // (op 1249) reg expression
s_mov_b32 s54, s0 // (op 1249) reg expression
s_mov_b32 s55, 131072 // (op 1249) call()

s_mul_hi_i32 s0, s16, s10 // (op 1292) 
v_lshrrev_b32 v14, 2, v1 // (op 1292) 
s_add_i32 s1, s0, s16 // (op 1292) 

v_lshlrev_b32 v15, 1, v14 // (op 1292) 

s_ashr_i32 s0, s1, 31 // (op 1292) 
v_and_b32 v14, -4, v1 // (op 1292) 
s_cmp_eq_i32 s10, 0 // (op 1292) 
s_mov_b32 s18, scc // (op 1292) 
v_sub_u32 v16, v1, v14 // (op 1292) 

s_cmp_eq_i32 s18, 0 // (op 1292) 
s_cselect_b32 s19, 0, -1 // (op 1292) 

v_lshlrev_b32 v14, 6, v16 // (op 1292) 

v_mov_b32 v17, 0 // (op 1292) convert
v_mov_b32 v16, v14 // (op 1292) convert

s_and_b32 s18, s11, 31 // (op 1292) 
s_lshl_b32 s22, 1, s18 // (op 1292) 
s_add_i32 s23, s22, s19 // (op 1292) 


s_and_b32 s19, s0, s23 // (op 1292) 


s_add_i32 s0, s1, s19 // (op 1292) 


s_ashr_i32 s1, s0, s18 // (op 1292) 


s_lshl_b32 s0, s11, 24 // (op 1292) 
s_ashr_i32 s18, s0, 31 // (op 1292) 

s_xor_b32 s0, s1, s18 // (op 1292) 

s_sub_i32 s1, s0, s18 // (op 1292) 


s_mul_hi_i32 s0, s1, s12 // (op 1292) 
s_add_i32 s18, s0, s1 // (op 1292) 

s_ashr_i32 s0, s18, 31 // (op 1292) 
s_cmp_eq_i32 s12, 0 // (op 1292) 
s_mov_b32 s19, scc // (op 1292) 
s_cmp_eq_i32 s19, 0 // (op 1292) 
s_cselect_b32 s22, 0, -1 // (op 1292) 

s_and_b32 s19, s13, 31 // (op 1292) 
s_lshl_b32 s23, 1, s19 // (op 1292) 
s_add_i32 s24, s23, s22 // (op 1292) 


s_and_b32 s25, s0, s24 // (op 1292) 


s_add_i32 s0, s18, s25 // (op 1292) 


s_ashr_i32 s18, s0, s19 // (op 1292) 


s_lshl_b32 s0, s13, 24 // (op 1292) 
s_ashr_i32 s19, s0, 31 // (op 1292) 

s_xor_b32 s0, s18, s19 // (op 1292) 

s_sub_i32 s18, s0, s19 // (op 1292) 


s_add_u32 s50, s6, 64 // (op 1292) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1292) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1292) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1292) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1292) 

s_mul_i32 s0, s18, s24 // (op 1292) 

s_sub_i32 s18, s1, s0 // (op 1292) 


s_mul_hi_i32 s19, s16, s9 // (op 1292) 
s_add_i32 s0, s19, s16 // (op 1292) 

s_ashr_i32 s1, s0, 31 // (op 1292) 
s_cmp_eq_i32 s9, 0 // (op 1292) 
s_mov_b32 s19, scc // (op 1292) 
s_cmp_eq_i32 s19, 0 // (op 1292) 
s_cselect_b32 s22, 0, -1 // (op 1292) 

s_and_b32 s19, s46, 31 // (op 1292) 
s_lshl_b32 s23, 1, s19 // (op 1292) 
s_add_i32 s49, s23, s22 // (op 1292) 


s_and_b32 s22, s1, s49 // (op 1292) 


s_add_i32 s1, s0, s22 // (op 1292) 


s_ashr_i32 s0, s1, s19 // (op 1292) 


s_lshl_b32 s1, s46, 24 // (op 1292) 
s_ashr_i32 s19, s1, 31 // (op 1292) 

s_xor_b32 s1, s0, s19 // (op 1292) 

s_sub_i32 s0, s1, s19 // (op 1292) 


s_mul_i32 s1, s15, s24 // (op 1292) 

s_mul_i32 s19, s8, s1 // (op 1292) 

s_mul_i32 s1, s0, s19 // (op 1292) 

s_sub_u32 s19, s16, s1 // (op 1292) 

s_mul_hi_i32 s1, s19, s14 // (op 1292) 
s_add_i32 s22, s1, s19 // (op 1292) 


s_ashr_i32 s1, s22, 31 // (op 1292) 
s_cmp_eq_i32 s14, 0 // (op 1292) 
s_mov_b32 s19, scc // (op 1292) 
s_cmp_eq_i32 s19, 0 // (op 1292) 
s_cselect_b32 s23, 0, -1 // (op 1292) 

s_and_b32 s19, s48, 31 // (op 1292) 
s_lshl_b32 s24, 1, s19 // (op 1292) 
s_add_i32 s25, s24, s23 // (op 1292) 


s_and_b32 s23, s1, s25 // (op 1292) 


s_add_i32 s1, s22, s23 // (op 1292) 


s_ashr_i32 s22, s1, s19 // (op 1292) 


s_lshl_b32 s1, s48, 24 // (op 1292) 
s_ashr_i32 s19, s1, 31 // (op 1292) 

s_xor_b32 s1, s22, s19 // (op 1292) 

s_sub_i32 s22, s1, s19 // (op 1292) 


s_cmp_eq_i32 s0, 0 // (op 1292) 
s_mov_b32 s1, scc // (op 1292) 

s_cmp_eq_i32 s1, 0 // (op 1292) 
s_cselect_b32 s0, s22, s18 // (op 1292) 



v_lshl_add_u32 v14, s0, 6, v15 // (op 1292) 


v_mul_lo_u32 v23, v14, s33 // (op 1292) most significant: low of low * high
v_mul_hi_u32 v15, v14, s32 // (op 1292) most significant: high of low * low
v_mul_lo_u32 v22, v14, s32 // (op 1292) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 1292) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v22 // (op 1292) least significant half
v_addc_co_u32 v15, s[0:1], v17, v23, s[0:1] // (op 1292) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 1292) most significant: low of high * low
v_mul_hi_u32 v22, v14, 6 // (op 1292) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 1292) least significant: low of low * low
v_add_u32 v17, v17, v22 // (op 1292) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 1292) 

v_mov_b32 v16, v14 // (op 1292) convert

s_mul_i32 s1, s7, 6 // (op 1294) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 1294) least significant
s_mul_hi_u32 s18, s6, 6 // (op 1294) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1294) most significant: sum

s_mul_i32 s19, s32, s1 // (op 1294) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 1294) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 1294) least significant
s_mul_hi_u32 s23, s32, s0 // (op 1294) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 1294) most significant: sum
s_add_u32 s19, s19, s23 // (op 1294) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1294) 

s_mov_b32 s64, s34 // (op 1294) reg expression
s_mov_b32 s65, s35 // (op 1294) reg expression
s_mov_b32 s66, s0 // (op 1294) reg expression
s_mov_b32 s67, 131072 // (op 1294) call()

s_mul_hi_i32 s0, s16, s10 // (op 1311) 
v_lshrrev_b32 v17, 2, v1 // (op 1311) 
s_add_i32 s1, s0, s16 // (op 1311) 

v_lshlrev_b32 v14, 1, v17 // (op 1311) 

s_ashr_i32 s0, s1, 31 // (op 1311) 
v_and_b32 v15, -4, v1 // (op 1311) 
s_cmp_eq_i32 s10, 0 // (op 1311) 
s_mov_b32 s18, scc // (op 1311) 
v_sub_u32 v17, v1, v15 // (op 1311) 

s_cmp_eq_i32 s18, 0 // (op 1311) 
s_cselect_b32 s19, 0, -1 // (op 1311) 

v_lshlrev_b32 v15, 6, v17 // (op 1311) 

v_mov_b32 v23, 0 // (op 1311) convert
v_mov_b32 v22, v15 // (op 1311) convert

s_and_b32 s18, s11, 31 // (op 1311) 
s_lshl_b32 s22, 1, s18 // (op 1311) 
s_add_i32 s23, s22, s19 // (op 1311) 


s_and_b32 s19, s0, s23 // (op 1311) 


s_add_i32 s0, s1, s19 // (op 1311) 


s_ashr_i32 s1, s0, s18 // (op 1311) 


s_lshl_b32 s0, s11, 24 // (op 1311) 
s_ashr_i32 s18, s0, 31 // (op 1311) 

s_xor_b32 s0, s1, s18 // (op 1311) 

s_sub_i32 s1, s0, s18 // (op 1311) 


s_mul_hi_i32 s0, s1, s12 // (op 1311) 
s_add_i32 s18, s0, s1 // (op 1311) 

s_ashr_i32 s0, s18, 31 // (op 1311) 
s_cmp_eq_i32 s12, 0 // (op 1311) 
s_mov_b32 s19, scc // (op 1311) 
s_cmp_eq_i32 s19, 0 // (op 1311) 
s_cselect_b32 s22, 0, -1 // (op 1311) 

s_and_b32 s19, s13, 31 // (op 1311) 
s_lshl_b32 s23, 1, s19 // (op 1311) 
s_add_i32 s24, s23, s22 // (op 1311) 


s_and_b32 s25, s0, s24 // (op 1311) 


s_add_i32 s0, s18, s25 // (op 1311) 


s_ashr_i32 s18, s0, s19 // (op 1311) 


s_lshl_b32 s0, s13, 24 // (op 1311) 
s_ashr_i32 s19, s0, 31 // (op 1311) 

s_xor_b32 s0, s18, s19 // (op 1311) 

s_sub_i32 s18, s0, s19 // (op 1311) 


s_add_u32 s50, s6, 64 // (op 1311) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1311) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1311) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1311) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1311) 

s_mul_i32 s0, s18, s24 // (op 1311) 

s_sub_i32 s18, s1, s0 // (op 1311) 


s_mul_hi_i32 s19, s16, s9 // (op 1311) 
s_add_i32 s0, s19, s16 // (op 1311) 

s_ashr_i32 s1, s0, 31 // (op 1311) 
s_cmp_eq_i32 s9, 0 // (op 1311) 
s_mov_b32 s19, scc // (op 1311) 
s_cmp_eq_i32 s19, 0 // (op 1311) 
s_cselect_b32 s22, 0, -1 // (op 1311) 

s_and_b32 s19, s46, 31 // (op 1311) 
s_lshl_b32 s23, 1, s19 // (op 1311) 
s_add_i32 s49, s23, s22 // (op 1311) 


s_and_b32 s22, s1, s49 // (op 1311) 


s_add_i32 s1, s0, s22 // (op 1311) 


s_ashr_i32 s0, s1, s19 // (op 1311) 


s_lshl_b32 s1, s46, 24 // (op 1311) 
s_ashr_i32 s19, s1, 31 // (op 1311) 

s_xor_b32 s1, s0, s19 // (op 1311) 

s_sub_i32 s0, s1, s19 // (op 1311) 


s_mul_i32 s1, s15, s24 // (op 1311) 

s_mul_i32 s19, s8, s1 // (op 1311) 

s_mul_i32 s1, s0, s19 // (op 1311) 

s_sub_u32 s19, s16, s1 // (op 1311) 

s_mul_hi_i32 s1, s19, s14 // (op 1311) 
s_add_i32 s22, s1, s19 // (op 1311) 


s_ashr_i32 s1, s22, 31 // (op 1311) 
s_cmp_eq_i32 s14, 0 // (op 1311) 
s_mov_b32 s19, scc // (op 1311) 
s_cmp_eq_i32 s19, 0 // (op 1311) 
s_cselect_b32 s23, 0, -1 // (op 1311) 

s_and_b32 s19, s48, 31 // (op 1311) 
s_lshl_b32 s24, 1, s19 // (op 1311) 
s_add_i32 s25, s24, s23 // (op 1311) 


s_and_b32 s23, s1, s25 // (op 1311) 


s_add_i32 s1, s22, s23 // (op 1311) 


s_ashr_i32 s22, s1, s19 // (op 1311) 


s_lshl_b32 s1, s48, 24 // (op 1311) 
s_ashr_i32 s19, s1, 31 // (op 1311) 

s_xor_b32 s1, s22, s19 // (op 1311) 

s_sub_i32 s22, s1, s19 // (op 1311) 


s_cmp_eq_i32 s0, 0 // (op 1311) 
s_mov_b32 s1, scc // (op 1311) 

s_cmp_eq_i32 s1, 0 // (op 1311) 
s_cselect_b32 s0, s22, s18 // (op 1311) 



v_lshl_add_u32 v15, s0, 6, v14 // (op 1311) 


v_mul_lo_u32 v25, v15, s33 // (op 1311) most significant: low of low * high
v_mul_hi_u32 v14, v15, s32 // (op 1311) most significant: high of low * low
v_mul_lo_u32 v24, v15, s32 // (op 1311) least significant: low of low * low
v_add_u32 v25, v25, v14 // (op 1311) most significant: sum


v_add_co_u32 v14, s[0:1], v22, v24 // (op 1311) least significant half
v_addc_co_u32 v15, s[0:1], v23, v25, s[0:1] // (op 1311) most significant half



v_mul_lo_u32 v23, v15, 6 // (op 1311) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 1311) most significant: high of low * low
v_mul_lo_u32 v22, v14, 6 // (op 1311) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 1311) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[22:23] // (op 1311) 

v_mov_b32 v17, v14 // (op 1311) convert

s_mul_i32 s1, s7, 6 // (op 1313) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 1313) least significant
s_mul_hi_u32 s18, s6, 6 // (op 1313) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1313) most significant: sum

s_mul_i32 s19, s32, s1 // (op 1313) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 1313) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 1313) least significant
s_mul_hi_u32 s23, s32, s0 // (op 1313) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 1313) most significant: sum
s_add_u32 s19, s19, s23 // (op 1313) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1313) 

s_mov_b32 s64, s34 // (op 1313) reg expression
s_mov_b32 s65, s35 // (op 1313) reg expression
s_mov_b32 s66, s0 // (op 1313) reg expression
s_mov_b32 s67, 131072 // (op 1313) call()

s_mul_hi_i32 s0, s16, s10 // (op 1356) 
v_and_b32 v14, -16, v1 // (op 1356) 
s_add_i32 s1, s0, s16 // (op 1356) 

v_sub_u32 v15, v1, v14 // (op 1356) 

s_ashr_i32 s0, s1, 31 // (op 1356) 
v_lshrrev_b32 v14, 4, v1 // (op 1356) 
s_cmp_eq_i32 s10, 0 // (op 1356) 
s_mov_b32 s18, scc // (op 1356) 
s_cmp_eq_i32 s18, 0 // (op 1356) 
s_cselect_b32 s19, 0, -1 // (op 1356) 

s_and_b32 s18, s11, 31 // (op 1356) 
s_lshl_b32 s22, 1, s18 // (op 1356) 
s_add_i32 s23, s22, s19 // (op 1356) 


s_and_b32 s19, s0, s23 // (op 1356) 


s_add_i32 s0, s1, s19 // (op 1356) 


s_ashr_i32 s1, s0, s18 // (op 1356) 


s_lshl_b32 s0, s11, 24 // (op 1356) 
s_ashr_i32 s18, s0, 31 // (op 1356) 

s_xor_b32 s0, s1, s18 // (op 1356) 

s_sub_i32 s1, s0, s18 // (op 1356) 


s_mul_hi_i32 s0, s1, s12 // (op 1356) 
s_add_i32 s18, s0, s1 // (op 1356) 

s_ashr_i32 s0, s18, 31 // (op 1356) 
s_cmp_eq_i32 s12, 0 // (op 1356) 
s_mov_b32 s19, scc // (op 1356) 
s_cmp_eq_i32 s19, 0 // (op 1356) 
s_cselect_b32 s22, 0, -1 // (op 1356) 

s_and_b32 s19, s13, 31 // (op 1356) 
s_lshl_b32 s23, 1, s19 // (op 1356) 
s_add_i32 s24, s23, s22 // (op 1356) 


s_and_b32 s25, s0, s24 // (op 1356) 


s_add_i32 s0, s18, s25 // (op 1356) 


s_ashr_i32 s18, s0, s19 // (op 1356) 


s_lshl_b32 s0, s13, 24 // (op 1356) 
s_ashr_i32 s19, s0, 31 // (op 1356) 

s_xor_b32 s0, s18, s19 // (op 1356) 

s_sub_i32 s18, s0, s19 // (op 1356) 


s_mul_i32 s0, s1, s15 // (op 1356) 

s_sub_u32 s1, s16, s0 // (op 1356) 

s_mul_i32 s0, s18, s15 // (op 1356) 
s_add_u32 s0, s0, s1 // (op 1356) 


s_mul_hi_i32 s1, s16, s9 // (op 1356) 
s_add_i32 s18, s1, s16 // (op 1356) 

s_ashr_i32 s1, s18, 31 // (op 1356) 
s_cmp_eq_i32 s9, 0 // (op 1356) 
s_mov_b32 s19, scc // (op 1356) 
s_cmp_eq_i32 s19, 0 // (op 1356) 
s_cselect_b32 s22, 0, -1 // (op 1356) 

s_and_b32 s19, s46, 31 // (op 1356) 
s_lshl_b32 s23, 1, s19 // (op 1356) 
s_add_i32 s24, s23, s22 // (op 1356) 


s_and_b32 s25, s1, s24 // (op 1356) 


s_add_i32 s1, s18, s25 // (op 1356) 


s_ashr_i32 s18, s1, s19 // (op 1356) 


s_lshl_b32 s1, s46, 24 // (op 1356) 
s_ashr_i32 s19, s1, 31 // (op 1356) 

s_xor_b32 s1, s18, s19 // (op 1356) 

s_sub_i32 s18, s1, s19 // (op 1356) 


s_add_u32 s50, s6, 64 // (op 1356) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1356) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1356) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1356) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1356) 

s_mul_i32 s1, s15, s24 // (op 1356) 

s_mul_i32 s19, s8, s1 // (op 1356) 

s_mul_i32 s1, s18, s19 // (op 1356) 

s_sub_u32 s19, s16, s1 // (op 1356) 

s_mov_b32 s1, s19 // (op 1356) 
s_mul_hi_i32 s22, s1, s14 // (op 1356) 
s_add_i32 s23, s22, s1 // (op 1356) 


s_ashr_i32 s1, s23, 31 // (op 1356) 
s_cmp_eq_i32 s14, 0 // (op 1356) 
s_mov_b32 s22, scc // (op 1356) 
s_cmp_eq_i32 s22, 0 // (op 1356) 
s_cselect_b32 s24, 0, -1 // (op 1356) 

s_and_b32 s22, s48, 31 // (op 1356) 
s_lshl_b32 s25, 1, s22 // (op 1356) 
s_add_i32 s49, s25, s24 // (op 1356) 


s_and_b32 s24, s1, s49 // (op 1356) 


s_add_i32 s1, s23, s24 // (op 1356) 


s_ashr_i32 s23, s1, s22 // (op 1356) 


s_lshl_b32 s1, s48, 24 // (op 1356) 
s_ashr_i32 s22, s1, 31 // (op 1356) 

s_xor_b32 s1, s23, s22 // (op 1356) 

s_sub_i32 s23, s1, s22 // (op 1356) 


s_mul_i32 s1, s47, s15 // (op 1356) 
s_add_u32 s24, s20, 16 // (op 1356) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1356) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1356) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1356) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1356) 

s_sub_i32 s22, s24, s1 // (op 1356) 

s_mul_i32 s24, s23, s22 // (op 1356) 


s_sub_u32 s25, s19, s24 // (op 1356) 


s_add_u32 s19, s25, s1 // (op 1356) 


s_cmp_eq_i32 s18, 0 // (op 1356) 
s_mov_b32 s1, scc // (op 1356) 

s_cmp_eq_i32 s1, 0 // (op 1356) 
s_cselect_b32 s18, s19, s0 // (op 1356) 



v_lshl_add_u32 v22, s18, 4, v15 // (op 1356) 


v_mul_lo_u32 v15, v14, s36 // (op 1356) 

v_add_u32 v14, v22, v15 // (op 1356) 


s_mul_i32 s1, s36, s41 // (op 1358) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1358) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1358) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1358) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1358) most significant: sum
s_add_u32 s1, s1, s19 // (op 1358) most significant: sum


s_mov_b32 s68, s38 // (op 1358) reg expression
s_mov_b32 s69, s39 // (op 1358) reg expression
s_mov_b32 s70, s0 // (op 1358) reg expression
s_mov_b32 s71, 131072 // (op 1358) call()

s_mul_hi_i32 s0, s16, s10 // (op 1375) 
v_and_b32 v15, -16, v1 // (op 1375) 
s_add_i32 s1, s0, s16 // (op 1375) 

v_sub_u32 v22, v1, v15 // (op 1375) 

s_ashr_i32 s0, s1, 31 // (op 1375) 
v_lshrrev_b32 v15, 4, v1 // (op 1375) 
s_cmp_eq_i32 s10, 0 // (op 1375) 
s_mov_b32 s18, scc // (op 1375) 
s_cmp_eq_i32 s18, 0 // (op 1375) 
s_cselect_b32 s19, 0, -1 // (op 1375) 

s_and_b32 s18, s11, 31 // (op 1375) 
s_lshl_b32 s22, 1, s18 // (op 1375) 
s_add_i32 s23, s22, s19 // (op 1375) 


s_and_b32 s19, s0, s23 // (op 1375) 


s_add_i32 s0, s1, s19 // (op 1375) 


s_ashr_i32 s1, s0, s18 // (op 1375) 


s_lshl_b32 s0, s11, 24 // (op 1375) 
s_ashr_i32 s18, s0, 31 // (op 1375) 

s_xor_b32 s0, s1, s18 // (op 1375) 

s_sub_i32 s1, s0, s18 // (op 1375) 


s_mul_hi_i32 s0, s1, s12 // (op 1375) 
s_add_i32 s18, s0, s1 // (op 1375) 

s_ashr_i32 s0, s18, 31 // (op 1375) 
s_cmp_eq_i32 s12, 0 // (op 1375) 
s_mov_b32 s19, scc // (op 1375) 
s_cmp_eq_i32 s19, 0 // (op 1375) 
s_cselect_b32 s22, 0, -1 // (op 1375) 

s_and_b32 s19, s13, 31 // (op 1375) 
s_lshl_b32 s23, 1, s19 // (op 1375) 
s_add_i32 s24, s23, s22 // (op 1375) 


s_and_b32 s25, s0, s24 // (op 1375) 


s_add_i32 s0, s18, s25 // (op 1375) 


s_ashr_i32 s18, s0, s19 // (op 1375) 


s_lshl_b32 s0, s13, 24 // (op 1375) 
s_ashr_i32 s19, s0, 31 // (op 1375) 

s_xor_b32 s0, s18, s19 // (op 1375) 

s_sub_i32 s18, s0, s19 // (op 1375) 


s_mul_i32 s0, s1, s15 // (op 1375) 

s_sub_u32 s1, s16, s0 // (op 1375) 

s_mul_i32 s0, s18, s15 // (op 1375) 
s_add_u32 s0, s0, s1 // (op 1375) 


s_mul_hi_i32 s1, s16, s9 // (op 1375) 
s_add_i32 s18, s1, s16 // (op 1375) 

s_ashr_i32 s1, s18, 31 // (op 1375) 
s_cmp_eq_i32 s9, 0 // (op 1375) 
s_mov_b32 s19, scc // (op 1375) 
s_cmp_eq_i32 s19, 0 // (op 1375) 
s_cselect_b32 s22, 0, -1 // (op 1375) 

s_and_b32 s19, s46, 31 // (op 1375) 
s_lshl_b32 s23, 1, s19 // (op 1375) 
s_add_i32 s24, s23, s22 // (op 1375) 


s_and_b32 s25, s1, s24 // (op 1375) 


s_add_i32 s1, s18, s25 // (op 1375) 


s_ashr_i32 s18, s1, s19 // (op 1375) 


s_lshl_b32 s1, s46, 24 // (op 1375) 
s_ashr_i32 s19, s1, 31 // (op 1375) 

s_xor_b32 s1, s18, s19 // (op 1375) 

s_sub_i32 s18, s1, s19 // (op 1375) 


s_add_u32 s50, s6, 64 // (op 1375) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1375) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1375) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1375) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1375) 

s_mul_i32 s1, s15, s24 // (op 1375) 

s_mul_i32 s19, s8, s1 // (op 1375) 

s_mul_i32 s1, s18, s19 // (op 1375) 

s_sub_u32 s19, s16, s1 // (op 1375) 

s_mov_b32 s1, s19 // (op 1375) 
s_mul_hi_i32 s22, s1, s14 // (op 1375) 
s_add_i32 s23, s22, s1 // (op 1375) 


s_ashr_i32 s1, s23, 31 // (op 1375) 
s_cmp_eq_i32 s14, 0 // (op 1375) 
s_mov_b32 s22, scc // (op 1375) 
s_cmp_eq_i32 s22, 0 // (op 1375) 
s_cselect_b32 s24, 0, -1 // (op 1375) 

s_and_b32 s22, s48, 31 // (op 1375) 
s_lshl_b32 s25, 1, s22 // (op 1375) 
s_add_i32 s49, s25, s24 // (op 1375) 


s_and_b32 s24, s1, s49 // (op 1375) 


s_add_i32 s1, s23, s24 // (op 1375) 


s_ashr_i32 s23, s1, s22 // (op 1375) 


s_lshl_b32 s1, s48, 24 // (op 1375) 
s_ashr_i32 s22, s1, 31 // (op 1375) 

s_xor_b32 s1, s23, s22 // (op 1375) 

s_sub_i32 s23, s1, s22 // (op 1375) 


s_mul_i32 s1, s47, s15 // (op 1375) 
s_add_u32 s24, s20, 16 // (op 1375) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1375) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1375) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1375) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1375) 

s_sub_i32 s22, s24, s1 // (op 1375) 

s_mul_i32 s24, s23, s22 // (op 1375) 


s_sub_u32 s25, s19, s24 // (op 1375) 


s_add_u32 s19, s25, s1 // (op 1375) 


s_cmp_eq_i32 s18, 0 // (op 1375) 
s_mov_b32 s1, scc // (op 1375) 

s_cmp_eq_i32 s1, 0 // (op 1375) 
s_cselect_b32 s18, s19, s0 // (op 1375) 



v_lshl_add_u32 v23, s18, 4, v22 // (op 1375) 


v_mul_lo_u32 v22, v15, s36 // (op 1375) 

v_add_u32 v15, v23, v22 // (op 1375) 


s_mul_i32 s1, s36, s41 // (op 1377) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1377) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1377) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1377) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1377) most significant: sum
s_add_u32 s1, s1, s19 // (op 1377) most significant: sum


s_mov_b32 s68, s38 // (op 1377) reg expression
s_mov_b32 s69, s39 // (op 1377) reg expression
s_mov_b32 s70, s0 // (op 1377) reg expression
s_mov_b32 s71, 131072 // (op 1377) call()

s_mul_hi_i32 s0, s16, s10 // (op 1420) 
v_lshrrev_b32 v22, 1, v1 // (op 1420) 
s_add_i32 s1, s0, s16 // (op 1420) 

v_and_b32 v23, -2, v1 // (op 1420) 
s_ashr_i32 s0, s1, 31 // (op 1420) 
v_sub_u32 v24, v1, v23 // (op 1420) 

s_cmp_eq_i32 s10, 0 // (op 1420) 
s_mov_b32 s18, scc // (op 1420) 
s_cmp_eq_i32 s18, 0 // (op 1420) 
s_cselect_b32 s19, 0, -1 // (op 1420) 

s_and_b32 s18, s11, 31 // (op 1420) 
s_lshl_b32 s22, 1, s18 // (op 1420) 
s_add_i32 s23, s22, s19 // (op 1420) 


s_and_b32 s19, s0, s23 // (op 1420) 


s_add_i32 s0, s1, s19 // (op 1420) 


s_ashr_i32 s1, s0, s18 // (op 1420) 


s_lshl_b32 s0, s11, 24 // (op 1420) 
s_ashr_i32 s18, s0, 31 // (op 1420) 

s_xor_b32 s0, s1, s18 // (op 1420) 

s_sub_i32 s1, s0, s18 // (op 1420) 


s_mul_hi_i32 s0, s1, s12 // (op 1420) 
s_add_i32 s18, s0, s1 // (op 1420) 

s_ashr_i32 s0, s18, 31 // (op 1420) 
s_cmp_eq_i32 s12, 0 // (op 1420) 
s_mov_b32 s19, scc // (op 1420) 
s_cmp_eq_i32 s19, 0 // (op 1420) 
s_cselect_b32 s22, 0, -1 // (op 1420) 

s_and_b32 s19, s13, 31 // (op 1420) 
s_lshl_b32 s23, 1, s19 // (op 1420) 
s_add_i32 s24, s23, s22 // (op 1420) 


s_and_b32 s25, s0, s24 // (op 1420) 


s_add_i32 s0, s18, s25 // (op 1420) 


s_ashr_i32 s18, s0, s19 // (op 1420) 


s_lshl_b32 s0, s13, 24 // (op 1420) 
s_ashr_i32 s19, s0, 31 // (op 1420) 

s_xor_b32 s0, s18, s19 // (op 1420) 

s_sub_i32 s18, s0, s19 // (op 1420) 


s_add_u32 s50, s6, 64 // (op 1420) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1420) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1420) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1420) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1420) 

s_mul_i32 s0, s18, s24 // (op 1420) 

s_sub_i32 s18, s1, s0 // (op 1420) 


s_mul_hi_i32 s19, s16, s9 // (op 1420) 
s_add_i32 s0, s19, s16 // (op 1420) 

s_ashr_i32 s1, s0, 31 // (op 1420) 
s_cmp_eq_i32 s9, 0 // (op 1420) 
s_mov_b32 s19, scc // (op 1420) 
s_cmp_eq_i32 s19, 0 // (op 1420) 
s_cselect_b32 s22, 0, -1 // (op 1420) 

s_and_b32 s19, s46, 31 // (op 1420) 
s_lshl_b32 s23, 1, s19 // (op 1420) 
s_add_i32 s49, s23, s22 // (op 1420) 


s_and_b32 s22, s1, s49 // (op 1420) 


s_add_i32 s1, s0, s22 // (op 1420) 


s_ashr_i32 s0, s1, s19 // (op 1420) 


s_lshl_b32 s1, s46, 24 // (op 1420) 
s_ashr_i32 s19, s1, 31 // (op 1420) 

s_xor_b32 s1, s0, s19 // (op 1420) 

s_sub_i32 s0, s1, s19 // (op 1420) 


s_mul_i32 s1, s15, s24 // (op 1420) 

s_mul_i32 s19, s8, s1 // (op 1420) 

s_mul_i32 s1, s0, s19 // (op 1420) 

s_sub_u32 s19, s16, s1 // (op 1420) 

s_mul_hi_i32 s1, s19, s14 // (op 1420) 
s_add_i32 s22, s1, s19 // (op 1420) 


s_ashr_i32 s1, s22, 31 // (op 1420) 
s_cmp_eq_i32 s14, 0 // (op 1420) 
s_mov_b32 s19, scc // (op 1420) 
s_cmp_eq_i32 s19, 0 // (op 1420) 
s_cselect_b32 s23, 0, -1 // (op 1420) 

s_and_b32 s19, s48, 31 // (op 1420) 
s_lshl_b32 s24, 1, s19 // (op 1420) 
s_add_i32 s25, s24, s23 // (op 1420) 


s_and_b32 s23, s1, s25 // (op 1420) 


s_add_i32 s1, s22, s23 // (op 1420) 


s_ashr_i32 s22, s1, s19 // (op 1420) 


s_lshl_b32 s1, s48, 24 // (op 1420) 
s_ashr_i32 s19, s1, 31 // (op 1420) 

s_xor_b32 s1, s22, s19 // (op 1420) 

s_sub_i32 s22, s1, s19 // (op 1420) 


s_cmp_eq_i32 s0, 0 // (op 1420) 
s_mov_b32 s1, scc // (op 1420) 

s_cmp_eq_i32 s1, 0 // (op 1420) 
s_cselect_b32 s0, s22, s18 // (op 1420) 



v_lshl_add_u32 v23, s0, 6, v22 // (op 1420) 


v_mul_lo_u32 v22, v23, s42 // (op 1420) 

v_lshl_add_u32 v23, v24, 2, v22 // (op 1420) 


s_mul_i32 s1, s42, s7 // (op 1422) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 1422) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 1422) least significant
s_mul_hi_u32 s19, s42, s6 // (op 1422) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1422) most significant: sum
s_add_u32 s1, s1, s19 // (op 1422) most significant: sum


s_mov_b32 s72, s44 // (op 1422) reg expression
s_mov_b32 s73, s45 // (op 1422) reg expression
s_mov_b32 s74, s0 // (op 1422) reg expression
s_mov_b32 s75, 131072 // (op 1422) call()

s_mul_hi_i32 s0, s16, s10 // (op 1439) 
v_lshrrev_b32 v22, 1, v1 // (op 1439) 
s_add_i32 s1, s0, s16 // (op 1439) 

v_and_b32 v24, -2, v1 // (op 1439) 
s_ashr_i32 s0, s1, 31 // (op 1439) 
v_sub_u32 v25, v1, v24 // (op 1439) 

s_cmp_eq_i32 s10, 0 // (op 1439) 
s_mov_b32 s18, scc // (op 1439) 
s_cmp_eq_i32 s18, 0 // (op 1439) 
s_cselect_b32 s19, 0, -1 // (op 1439) 

s_and_b32 s18, s11, 31 // (op 1439) 
s_lshl_b32 s22, 1, s18 // (op 1439) 
s_add_i32 s23, s22, s19 // (op 1439) 


s_and_b32 s19, s0, s23 // (op 1439) 


s_add_i32 s0, s1, s19 // (op 1439) 


s_ashr_i32 s1, s0, s18 // (op 1439) 


s_lshl_b32 s0, s11, 24 // (op 1439) 
s_ashr_i32 s18, s0, 31 // (op 1439) 

s_xor_b32 s0, s1, s18 // (op 1439) 

s_sub_i32 s1, s0, s18 // (op 1439) 


s_mul_hi_i32 s0, s1, s12 // (op 1439) 
s_add_i32 s18, s0, s1 // (op 1439) 

s_ashr_i32 s0, s18, 31 // (op 1439) 
s_cmp_eq_i32 s12, 0 // (op 1439) 
s_mov_b32 s19, scc // (op 1439) 
s_cmp_eq_i32 s19, 0 // (op 1439) 
s_cselect_b32 s22, 0, -1 // (op 1439) 

s_and_b32 s19, s13, 31 // (op 1439) 
s_lshl_b32 s23, 1, s19 // (op 1439) 
s_add_i32 s24, s23, s22 // (op 1439) 


s_and_b32 s25, s0, s24 // (op 1439) 


s_add_i32 s0, s18, s25 // (op 1439) 


s_ashr_i32 s18, s0, s19 // (op 1439) 


s_lshl_b32 s0, s13, 24 // (op 1439) 
s_ashr_i32 s19, s0, 31 // (op 1439) 

s_xor_b32 s0, s18, s19 // (op 1439) 

s_sub_i32 s18, s0, s19 // (op 1439) 


s_add_u32 s50, s6, 64 // (op 1439) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1439) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1439) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1439) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1439) 

s_mul_i32 s0, s18, s24 // (op 1439) 

s_sub_i32 s18, s1, s0 // (op 1439) 


s_mul_hi_i32 s19, s16, s9 // (op 1439) 
s_add_i32 s0, s19, s16 // (op 1439) 

s_ashr_i32 s1, s0, 31 // (op 1439) 
s_cmp_eq_i32 s9, 0 // (op 1439) 
s_mov_b32 s19, scc // (op 1439) 
s_cmp_eq_i32 s19, 0 // (op 1439) 
s_cselect_b32 s22, 0, -1 // (op 1439) 

s_and_b32 s19, s46, 31 // (op 1439) 
s_lshl_b32 s23, 1, s19 // (op 1439) 
s_add_i32 s49, s23, s22 // (op 1439) 


s_and_b32 s22, s1, s49 // (op 1439) 


s_add_i32 s1, s0, s22 // (op 1439) 


s_ashr_i32 s0, s1, s19 // (op 1439) 


s_lshl_b32 s1, s46, 24 // (op 1439) 
s_ashr_i32 s19, s1, 31 // (op 1439) 

s_xor_b32 s1, s0, s19 // (op 1439) 

s_sub_i32 s0, s1, s19 // (op 1439) 


s_mul_i32 s1, s15, s24 // (op 1439) 

s_mul_i32 s19, s8, s1 // (op 1439) 

s_mul_i32 s1, s0, s19 // (op 1439) 

s_sub_u32 s19, s16, s1 // (op 1439) 

s_mul_hi_i32 s1, s19, s14 // (op 1439) 
s_add_i32 s22, s1, s19 // (op 1439) 


s_ashr_i32 s1, s22, 31 // (op 1439) 
s_cmp_eq_i32 s14, 0 // (op 1439) 
s_mov_b32 s19, scc // (op 1439) 
s_cmp_eq_i32 s19, 0 // (op 1439) 
s_cselect_b32 s23, 0, -1 // (op 1439) 

s_and_b32 s19, s48, 31 // (op 1439) 
s_lshl_b32 s24, 1, s19 // (op 1439) 
s_add_i32 s25, s24, s23 // (op 1439) 


s_and_b32 s23, s1, s25 // (op 1439) 


s_add_i32 s1, s22, s23 // (op 1439) 


s_ashr_i32 s22, s1, s19 // (op 1439) 


s_lshl_b32 s1, s48, 24 // (op 1439) 
s_ashr_i32 s19, s1, 31 // (op 1439) 

s_xor_b32 s1, s22, s19 // (op 1439) 

s_sub_i32 s22, s1, s19 // (op 1439) 


s_cmp_eq_i32 s0, 0 // (op 1439) 
s_mov_b32 s1, scc // (op 1439) 

s_cmp_eq_i32 s1, 0 // (op 1439) 
s_cselect_b32 s0, s22, s18 // (op 1439) 



v_lshl_add_u32 v24, s0, 6, v22 // (op 1439) 


v_mul_lo_u32 v22, v24, s42 // (op 1439) 

v_lshl_add_u32 v24, v25, 2, v22 // (op 1439) 


s_mul_i32 s1, s42, s7 // (op 1441) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 1441) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 1441) least significant
s_mul_hi_u32 s19, s42, s6 // (op 1441) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1441) most significant: sum
s_add_u32 s1, s1, s19 // (op 1441) most significant: sum


s_mov_b32 s72, s44 // (op 1441) reg expression
s_mov_b32 s73, s45 // (op 1441) reg expression
s_mov_b32 s74, s0 // (op 1441) reg expression
s_mov_b32 s75, 131072 // (op 1441) call()


s_mul_i32 s1, s33, 6 // (op 1029) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 1029) least significant
s_mul_hi_u32 s18, s32, 6 // (op 1029) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1029) most significant: sum

v_lshrrev_b64 v[26:27], 3, s[0:1] // (op 1029) 


v_mov_b32 v22, v16 // (op 1029) 
buffer_load_dwordx4 v[28:31], v22, s[64:67], 0 offen // (op 1029) Load value
buffer_load_dwordx2 v[32:33], v22, s[64:67], 0 offen offset:16 // (op 1029) Load value
buffer_load_dwordx4 v[34:37], v22, s[64:67], 0 offen offset:24 // (op 1029) Load value
buffer_load_dwordx2 v[38:39], v22, s[64:67], 0 offen offset:40 // (op 1029) Load value
v_add_u32 v22, v22, v26 // (op 1029) 
buffer_load_dwordx4 v[40:43], v22, s[64:67], 0 offen // (op 1029) Load value
buffer_load_dwordx2 v[44:45], v22, s[64:67], 0 offen offset:16 // (op 1029) Load value
buffer_load_dwordx4 v[46:49], v22, s[64:67], 0 offen offset:24 // (op 1029) Load value
buffer_load_dwordx2 v[50:51], v22, s[64:67], 0 offen offset:40 // (op 1029) Load value





buffer_load_dword v22, v23, s[72:75], 0 offen // (op 1032) Load value
v_lshlrev_b32 v25, 4, v1 // (op 1505) 
s_nop 0
v_readfirstlane_b32 s0, v25 // (op 1505) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[26:27], 7 // (op 1183) 
v_mov_b64 v[26:27], s[18:19] // (op 1183) convert


v_mov_b64 v[52:53], v[18:19] // (op 1183) reg expression
s_mov_b32 m0, s0 // (op 1183) reg expression
buffer_load_dwordx4 v52, s[52:55], 0 offen offset: 0 lds // (op 1183) Load value direct to lds
v_add_co_u32 v52, s[18:19], v52, v26 // (op 1183) least significant half
v_addc_co_u32 v53, s[18:19], 0, v53, s[18:19] // (op 1183) most significant half

s_add_u32 m0, m0, 2048 // (op 1183) 
buffer_load_dwordx4 v52, s[52:55], 0 offen offset: 0 lds // (op 1183) Load value direct to lds





v_mov_b32 v25, v14 // (op 1038) reg expression

buffer_load_ubyte v26, v25, s[68:71], 0 offen offset: 0 // (op 1038) Load value


s_mul_i32 s1, s33, 6 // (op 1041) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 1041) least significant
s_mul_hi_u32 s18, s32, 6 // (op 1041) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1041) most significant: sum

v_lshrrev_b64 v[52:53], 3, s[0:1] // (op 1041) 


v_mov_b32 v25, v16 // (op 1041) 
buffer_load_dwordx4 v[54:57], v25, s[64:67], 0 offen offset:192 // (op 1041) Load value
buffer_load_dwordx2 v[58:59], v25, s[64:67], 0 offen offset:208 // (op 1041) Load value
buffer_load_dwordx4 v[60:63], v25, s[64:67], 0 offen offset:216 // (op 1041) Load value
buffer_load_dwordx2 v[64:65], v25, s[64:67], 0 offen offset:232 // (op 1041) Load value
v_add_u32 v25, v25, v52 // (op 1041) 
buffer_load_dwordx4 v[66:69], v25, s[64:67], 0 offen offset:192 // (op 1041) Load value
buffer_load_dwordx2 v[70:71], v25, s[64:67], 0 offen offset:208 // (op 1041) Load value
buffer_load_dwordx4 v[72:75], v25, s[64:67], 0 offen offset:216 // (op 1041) Load value
buffer_load_dwordx2 v[76:77], v25, s[64:67], 0 offen offset:232 // (op 1041) Load value






buffer_load_dword v16, v23, s[72:75], 0 offen offset:8 // (op 1044) Load value

v_lshlrev_b32 v23, 4, v1 // (op 1731) 
s_nop 0
v_readfirstlane_b32 s0, v23 // (op 1731) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[26:27], 7 // (op 1188) 
v_mov_b64 v[52:53], s[18:19] // (op 1188) convert


s_lshl_b64 s[18:19], s[26:27], 8 // (op 1188) 
v_mov_b32 v23, s18 // (op 1188) 
v_mov_b32 v25, s19 // (op 1188) 
v_add_co_u32 v78, s[50:51], v18, v23 // (op 1188) least significant half
v_addc_co_u32 v79, s[50:51], v19, v25, s[50:51] // (op 1188) most significant half




s_add_u32 s1, 4096, s0 // (op 1188) 
s_mov_b32 m0, s1 // (op 1188) reg expression
buffer_load_dwordx4 v78, s[52:55], 0 offen offset: 0 lds // (op 1188) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v52 // (op 1188) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 1188) most significant half

s_add_u32 m0, m0, 2048 // (op 1188) 
buffer_load_dwordx4 v78, s[52:55], 0 offen offset: 0 lds // (op 1188) Load value direct to lds







s_lshl_b32 s0, s36, 3 // (op 1050) 
v_add_u32 v23, v14, s0 // (op 1050) 


buffer_load_ubyte v25, v23, s[68:71], 0 offen offset: 0 // (op 1050) Load value


v_and_b32 v14, -4, v1 // (op 1558) 
v_sub_u32 v23, v1, v14 // (op 1558) 

v_lshlrev_b32 v14, 6, v23 // (op 1558) 

v_lshrrev_b32 v23, 2, v1 // (op 1558) 
v_lshl_add_u32 v27, v23, 9, v14 // (op 1558) 


v_mul_lo_u32 v14, v27, 6 // (op 1558) 

v_lshrrev_b32 v23, 3, v14 // (op 1558) 




s_waitcnt vmcnt(23)
ds_write_b128 v23, v[28:31] offset:8192 // (op 1052) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v23, v[32:33] offset:8208 // (op 1052) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v23, v[34:37] offset:8216 // (op 1052) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v23, v[38:39] offset:8232 // (op 1052) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v23, v[40:43] offset:8384 // (op 1052) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v23, v[44:45] offset:8400 // (op 1052) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v23, v[46:49] offset:8408 // (op 1052) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v23, v[50:51] offset:8424 // (op 1052) Store local data 

v_and_b32 v14, -2, v1 // (op 1669) 
v_sub_u32 v23, v1, v14 // (op 1669) 

v_lshlrev_b32 v14, 2, v23 // (op 1669) 

v_lshrrev_b32 v23, 1, v1 // (op 1669) 
v_lshl_add_u32 v27, v23, 3, v14 // (op 1669) 





s_waitcnt vmcnt(15)
ds_write_b32 v27, v22 offset:20480 // (op 1053) Store local data 

v_and_b32 v14, -16, v1 // (op 1616) 
v_sub_u32 v23, v1, v14 // (op 1616) 

v_and_b32 v14, 4294967280, v1 // (op 1616) 
v_add_u32 v27, v14, v23 // (op 1616) 





s_waitcnt vmcnt(12)
ds_write_b8 v27, v26 offset:20992 // (op 1055) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1895) 



ds_read_b128 v[78:81], v9 offset:8192 // (op 1073) Load local data 
ds_read_b64 v[82:83], v9 offset:8208 // (op 1073) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1936) 



ds_read_u8 v14, v5 offset:20480 // (op 1086) Load local data 



ds_read_b128 v[84:87], v9 offset:14336 // (op 1078) Load local data 
ds_read_b64 v[88:89], v9 offset:14352 // (op 1078) Load local data 



ds_read_u8 v23, v5 offset:20736 // (op 1091) Load local data 
s_barrier  // (op 1876) 



ds_read_b64_tr_b8 v[90:91], v0 // (op 1069) Transpose load local data 
ds_read_b64_tr_b8 v[92:93], v0 offset:128 // (op 1069) Transpose load local data 
ds_read_b64_tr_b8 v[94:95], v0 offset:1024 // (op 1069) Transpose load local data 
ds_read_b64_tr_b8 v[96:97], v0 offset:1152 // (op 1069) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1916) 


ds_read_u8 v27, v4 offset:20992 // (op 1082) Load local data 
v_lshlrev_b32 v18, 4, v1 // (op 1477) 
s_nop 0
v_readfirstlane_b32 s0, v18 // (op 1477) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -4, v1 // (op 1529) 
v_sub_u32 v19, v1, v18 // (op 1529) 

v_lshlrev_b32 v18, 6, v19 // (op 1529) 

v_lshrrev_b32 v19, 2, v1 // (op 1529) 
v_lshl_add_u32 v52, v19, 9, v18 // (op 1529) 


v_mul_lo_u32 v53, v52, 6 // (op 1529) 

v_lshrrev_b32 v52, 3, v53 // (op 1529) 

v_and_b32 v53, -16, v1 // (op 1587) 
v_sub_u32 v18, v1, v53 // (op 1587) 

v_and_b32 v19, 4294967280, v1 // (op 1587) 
v_add_u32 v53, v19, v18 // (op 1587) 


v_and_b32 v18, -2, v1 // (op 1640) 
v_sub_u32 v19, v1, v18 // (op 1640) 

v_lshlrev_b32 v18, 2, v19 // (op 1640) 

v_lshrrev_b32 v19, 1, v1 // (op 1640) 
v_lshl_add_u32 v98, v19, 3, v18 // (op 1640) 


v_lshlrev_b32 v18, 4, v1 // (op 1716) 
s_nop 0
v_readfirstlane_b32 s1, v18 // (op 1716) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -4, v1 // (op 1755) 
v_sub_u32 v19, v1, v18 // (op 1755) 

v_lshlrev_b32 v18, 6, v19 // (op 1755) 

v_lshrrev_b32 v19, 2, v1 // (op 1755) 
v_lshl_add_u32 v99, v19, 9, v18 // (op 1755) 


v_mul_lo_u32 v18, v99, 6 // (op 1755) 

v_lshrrev_b32 v19, 3, v18 // (op 1755) 

v_and_b32 v18, -16, v1 // (op 1784) 
v_sub_u32 v99, v1, v18 // (op 1784) 

v_and_b32 v18, 4294967280, v1 // (op 1784) 
v_add_u32 v100, v18, v99 // (op 1784) 


v_and_b32 v18, -2, v1 // (op 1808) 
v_sub_u32 v99, v1, v18 // (op 1808) 

v_lshlrev_b32 v18, 2, v99 // (op 1808) 

v_lshrrev_b32 v99, 1, v1 // (op 1808) 
v_lshl_add_u32 v101, v99, 3, v18 // (op 1808) 


s_mov_b32 s18, 0 // (op 400) call()
s_ashr_i64 s[50:51], s[30:31], 8 // (op 399) 
s_mov_b32 s19, s50 // (op 399) convert

s_and_b32 s22, s19, 4294967294 // (op 399) 

s_cmp_lt_u32 s18, s22 // (op 399) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_399 // (op 399) Condition: Top (jump to RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_399 if false)
RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_399:

s_add_u32 s18, s18, 2 // (op 401) 
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
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:97], v[78:83], a[0:3], v27, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 88) 
buffer_load_dwordx2 v[38:39], v99, s[64:67], 0 offen offset:424 // (op 10) Load value
s_barrier  // (op 1885) 
v_add_u32 v99, v99, v102 // (op 10) 



buffer_load_dwordx4 v[40:43], v99, s[64:67], 0 offen offset:384 // (op 10) Load value
buffer_load_dwordx2 v[44:45], v99, s[64:67], 0 offen offset:400 // (op 10) Load value
ds_read_b64_tr_b8 v[104:105], v0 offset:2048 // (op 91) Transpose load local data 
ds_read_b64_tr_b8 v[106:107], v0 offset:2176 // (op 91) Transpose load local data 
ds_read_b64_tr_b8 v[108:109], v0 offset:3072 // (op 91) Transpose load local data 
ds_read_b64_tr_b8 v[110:111], v0 offset:3200 // (op 91) Transpose load local data 
buffer_load_dwordx4 v[46:49], v99, s[64:67], 0 offen offset:408 // (op 10) Load value
s_barrier  // (op 1905) 
buffer_load_dwordx2 v[50:51], v99, s[64:67], 0 offen offset:424 // (op 10) Load value







ds_read_b128 v[78:81], v9 offset:8288 // (op 95) Load local data 
ds_read_b64 v[82:83], v9 offset:8304 // (op 95) Load local data 
v_mov_b32 v99, v24 // (op 25) 

s_barrier  // (op 1926) 
buffer_load_dword v22, v99, s[72:75], 0 offen offset:16 // (op 25) Load value


ds_read_u8 v102, v4 offset:21056 // (op 99) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1879) 

s_barrier  // (op 1946) 
s_lshl_b64 s[50:51], s[26:27], 7 // (op 1167) 



v_mov_b64 v[112:113], s[50:51] // (op 1167) convert


v_mov_b64 v[114:115], v[20:21] // (op 1167) 
s_lshl_b64 s[50:51], s[26:27], 8 // (op 1167) 
s_lshl_b64 s[22:23], s[50:51], 1 // (op 1167) 

v_mov_b32 v103, s22 // (op 1167) 
v_mov_b32 v116, s23 // (op 1167) 
v_add_co_u32 v118, s[24:25], v114, v103 // (op 1167) least significant half
v_addc_co_u32 v119, s[24:25], v115, v116, s[24:25] // (op 1167) most significant half




s_mov_b32 m0, s0 // (op 1167) reg expression
ds_read_u8 v14, v5 offset:20484 // (op 103) Load local data 
buffer_load_dwordx4 v118, s[52:55], 0 offen offset: 0 lds // (op 1167) Load value direct to lds
v_add_co_u32 v118, s[50:51], v118, v112 // (op 1167) least significant half
v_addc_co_u32 v119, s[50:51], 0, v119, s[50:51] // (op 1167) most significant half

s_add_u32 m0, m0, 2048 // (op 1167) 
buffer_load_dwordx4 v118, s[52:55], 0 offen offset: 0 lds // (op 1167) Load value direct to lds


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:111], v[78:83], a[0:3], v102, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 105) 


v_mov_b32 v103, v15 // (op 17) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:97], v[84:89], a[4:7], v27, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 447) 
s_lshl_b32 s19, s36, 4 // (op 17) 



ds_read_b128 v[84:87], v9 offset:14432 // (op 451) Load local data 
ds_read_b64 v[88:89], v9 offset:14448 // (op 451) Load local data 
v_add_u32 v112, v103, s19 // (op 17) 





ds_read_u8 v23, v5 offset:20740 // (op 455) Load local data 
buffer_load_ubyte v26, v112, s[68:71], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1966) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:111], v[84:89], a[4:7], v102, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 456) 





s_waitcnt vmcnt(23)
ds_write_b128 v19, v[54:57] offset:21120 // (op 302) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v19, v[58:59] offset:21136 // (op 302) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v19, v[60:63] offset:21144 // (op 302) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v19, v[64:65] offset:21160 // (op 302) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v19, v[66:69] offset:21312 // (op 302) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v19, v[70:71] offset:21328 // (op 302) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v19, v[72:75] offset:21336 // (op 302) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v19, v[76:77] offset:21352 // (op 302) Store local data 
s_barrier  // (op 1978) 



s_waitcnt vmcnt(15)
ds_write_b32 v101, v16 offset:33408 // (op 304) Store local data 
s_barrier  // (op 1972) 



s_waitcnt vmcnt(12)
ds_write_b8 v100, v25 offset:33920 // (op 303) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1962) 



ds_read_b128 v[78:81], v13 offset:21120 // (op 287) Load local data 
ds_read_b64 v[82:83], v13 offset:21136 // (op 287) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1975) 



ds_read_u8 v14, v12 offset:33408 // (op 291) Load local data 



ds_read_b128 v[104:107], v13 offset:27264 // (op 470) Load local data 
ds_read_b64 v[108:109], v13 offset:27280 // (op 470) Load local data 



ds_read_u8 v102, v12 offset:33664 // (op 474) Load local data 
s_barrier  // (op 1956) 



ds_read_b64_tr_b8 v[90:91], v3 offset:4096 // (op 285) Transpose load local data 
ds_read_b64_tr_b8 v[92:93], v3 offset:4224 // (op 285) Transpose load local data 
ds_read_b64_tr_b8 v[94:95], v3 offset:5120 // (op 285) Transpose load local data 
ds_read_b64_tr_b8 v[96:97], v3 offset:5248 // (op 285) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1969) 


ds_read_u8 v27, v6 offset:33920 // (op 289) Load local data 

s_mul_i32 s51, s33, 6 // (op 305) most significant: low of high * low
s_mul_i32 s50, s32, 6 // (op 305) least significant
s_mul_hi_u32 s19, s32, 6 // (op 305) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 305) most significant: sum

v_lshrrev_b64 v[110:111], 3, s[50:51] // (op 305) 


v_mov_b32 v112, v18 // (op 305) 
buffer_load_dwordx4 v[54:57], v112, s[64:67], 0 offen offset:576 // (op 305) Load value
buffer_load_dwordx2 v[58:59], v112, s[64:67], 0 offen offset:592 // (op 305) Load value
buffer_load_dwordx4 v[60:63], v112, s[64:67], 0 offen offset:600 // (op 305) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:97], v[78:83], a[0:3], v27, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 292) 
buffer_load_dwordx2 v[64:65], v112, s[64:67], 0 offen offset:616 // (op 305) Load value
v_add_u32 v112, v112, v110 // (op 305) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:97], v[104:109], a[4:7], v27, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 457) 
buffer_load_dwordx4 v[66:69], v112, s[64:67], 0 offen offset:576 // (op 305) Load value


buffer_load_dwordx2 v[70:71], v112, s[64:67], 0 offen offset:592 // (op 305) Load value
ds_read_b64_tr_b8 v[116:117], v3 offset:6144 // (op 294) Transpose load local data 
ds_read_b64_tr_b8 v[118:119], v3 offset:6272 // (op 294) Transpose load local data 
ds_read_b64_tr_b8 v[120:121], v3 offset:7168 // (op 294) Transpose load local data 
ds_read_b64_tr_b8 v[122:123], v3 offset:7296 // (op 294) Transpose load local data 
buffer_load_dwordx4 v[72:75], v112, s[64:67], 0 offen offset:600 // (op 305) Load value
buffer_load_dwordx2 v[76:77], v112, s[64:67], 0 offen offset:616 // (op 305) Load value






ds_read_b128 v[104:107], v13 offset:21216 // (op 296) Load local data 



ds_read_b64 v[108:109], v13 offset:21232 // (op 296) Load local data 
buffer_load_dword v16, v99, s[72:75], 0 offen offset:24 // (op 307) Load value



v_add_u32 v17, 384, v17 // (op 1303) 
v_add_u32 v24, 16, v24 // (op 1431) 
ds_read_u8 v18, v6 offset:33984 // (op 298) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1959) 




s_barrier  // (op 2017) 
s_lshl_b64 s[50:51], s[26:27], 7 // (op 1178) 
v_mov_b64 v[110:111], s[50:51] // (op 1178) convert


s_lshl_b64 s[50:51], s[26:27], 8 // (op 1178) 
s_mul_i32 s23, 3, s51 // (op 1178) most significant: low of low * high
s_mul_i32 s22, 3, s50 // (op 1178) least significant
s_mul_hi_u32 s19, 3, s50 // (op 1178) most significant: high of low * low
s_add_u32 s23, s23, s19 // (op 1178) most significant: sum


v_mov_b32 v99, s22 // (op 1178) 
v_mov_b32 v102, s23 // (op 1178) 
v_add_co_u32 v112, s[24:25], v114, v99 // (op 1178) least significant half
v_addc_co_u32 v113, s[24:25], v115, v102, s[24:25] // (op 1178) most significant half




s_add_u32 s19, 4096, s1 // (op 1178) 
s_mov_b32 m0, s19 // (op 1178) reg expression
ds_read_u8 v99, v12 offset:33412 // (op 300) Load local data 
buffer_load_dwordx4 v112, s[52:55], 0 offen offset: 0 lds // (op 1178) Load value direct to lds



v_add_co_u32 v112, s[50:51], v112, v110 // (op 1178) least significant half
v_addc_co_u32 v113, s[50:51], 0, v113, s[50:51] // (op 1178) most significant half

s_add_u32 m0, m0, 2048 // (op 1178) 
buffer_load_dwordx4 v112, s[52:55], 0 offen offset: 0 lds // (op 1178) Load value direct to lds



ds_read_b128 v[124:127], v13 offset:27360 // (op 461) Load local data 

ds_read_b64 v[128:129], v13 offset:27376 // (op 461) Load local data 


s_lshl_b32 s19, s36, 3 // (op 306) 



s_barrier  // (op 2055) 
ds_read_u8 v102, v12 offset:33668 // (op 465) Load local data 
s_mul_i32 s22, 3, s19 // (op 306) 

v_add_u32 v110, v103, s22 // (op 306) 


s_barrier  // (op 2059) 
buffer_load_ubyte v25, v110, s[68:71], 0 offen offset: 0 // (op 306) Load value

s_lshl_b64 s[50:51], s[26:27], 8 // (op 1239) 

s_lshl_b64 s[22:23], s[50:51], 1 // (op 1239) 

s_barrier  // (op 1898) 
s_lshl_b32 s19, s36, 4 // (op 1367) 



v_add_u32 v15, v15, s19 // (op 1367) 

v_mov_b32 v103, s22 // (op 1239) 
v_mov_b32 v110, s23 // (op 1239) 
v_add_co_u32 v20, s[24:25], v20, v103 // (op 1239) least significant half
v_addc_co_u32 v21, s[24:25], v21, v110, s[24:25] // (op 1239) most significant half




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[116:123], v[104:109], a[0:3], v18, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 301) 
s_waitcnt vmcnt(23)
ds_write_b128 v52, v[28:31] offset:8192 // (op 13) Store local data 


s_waitcnt vmcnt(22)
ds_write_b64 v52, v[32:33] offset:8208 // (op 13) Store local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[116:123], v[124:129], a[4:7], v18, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 466) 
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
s_barrier  // (op 1939) 



s_waitcnt vmcnt(15)
ds_write_b32 v98, v22 offset:20480 // (op 30) Store local data 
s_barrier  // (op 1919) 



s_waitcnt vmcnt(12)
ds_write_b8 v53, v26 offset:20992 // (op 21) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1908) 



ds_read_b128 v[78:81], v9 offset:8192 // (op 75) Load local data 
ds_read_b64 v[82:83], v9 offset:8208 // (op 75) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1949) 



ds_read_u8 v14, v5 offset:20480 // (op 85) Load local data 



ds_read_b128 v[84:87], v9 offset:14336 // (op 442) Load local data 
ds_read_b64 v[88:89], v9 offset:14352 // (op 442) Load local data 



ds_read_u8 v23, v5 offset:20736 // (op 446) Load local data 
s_barrier  // (op 1888) 



ds_read_b64_tr_b8 v[90:91], v0 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[92:93], v0 offset:128 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[94:95], v0 offset:1024 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[96:97], v0 offset:1152 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1929) 


ds_read_u8 v27, v4 offset:20992 // (op 80) Load local data 
s_ashr_i64 s[50:51], s[30:31], 8 // (op 399) 
s_mov_b32 s19, s50 // (op 399) convert

s_and_b32 s22, s19, 4294967294 // (op 399) 

s_cmp_lt_u32 s18, s22 // (op 399) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_399 // (op 399) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_399 if true)
RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_399:
















s_barrier  // (op 2037) 
s_mul_hi_i32 s0, s16, s10 // (op 1201) 
s_add_i32 s1, s0, s16 // (op 1201) 

s_ashr_i32 s0, s1, 31 // (op 1201) 
s_cmp_eq_i32 s10, 0 // (op 1201) 
s_mov_b32 s18, scc // (op 1201) 
s_cmp_eq_i32 s18, 0 // (op 1201) 
s_cselect_b32 s19, 0, -1 // (op 1201) 

s_and_b32 s18, s11, 31 // (op 1201) 
s_lshl_b32 s22, 1, s18 // (op 1201) 
s_add_i32 s23, s22, s19 // (op 1201) 


s_and_b32 s19, s0, s23 // (op 1201) 


s_add_i32 s0, s1, s19 // (op 1201) 


s_ashr_i32 s1, s0, s18 // (op 1201) 


s_lshl_b32 s0, s11, 24 // (op 1201) 
s_ashr_i32 s18, s0, 31 // (op 1201) 

s_xor_b32 s0, s1, s18 // (op 1201) 

s_sub_i32 s1, s0, s18 // (op 1201) 


s_mul_hi_i32 s0, s1, s12 // (op 1201) 
s_add_i32 s18, s0, s1 // (op 1201) 

s_ashr_i32 s0, s18, 31 // (op 1201) 
s_cmp_eq_i32 s12, 0 // (op 1201) 
s_mov_b32 s19, scc // (op 1201) 
s_cmp_eq_i32 s19, 0 // (op 1201) 
s_cselect_b32 s22, 0, -1 // (op 1201) 

s_and_b32 s19, s13, 31 // (op 1201) 
s_lshl_b32 s23, 1, s19 // (op 1201) 
s_add_i32 s24, s23, s22 // (op 1201) 


s_and_b32 s25, s0, s24 // (op 1201) 


s_add_i32 s0, s18, s25 // (op 1201) 


s_ashr_i32 s18, s0, s19 // (op 1201) 


s_lshl_b32 s0, s13, 24 // (op 1201) 
s_ashr_i32 s19, s0, 31 // (op 1201) 

s_xor_b32 s0, s18, s19 // (op 1201) 

s_sub_i32 s18, s0, s19 // (op 1201) 


s_mul_i32 s0, s1, s15 // (op 1201) 

s_sub_u32 s1, s16, s0 // (op 1201) 

s_mul_i32 s0, s18, s15 // (op 1201) 
s_add_u32 s0, s0, s1 // (op 1201) 


s_mul_hi_i32 s1, s16, s9 // (op 1201) 
s_add_i32 s18, s1, s16 // (op 1201) 

s_ashr_i32 s1, s18, 31 // (op 1201) 
s_cmp_eq_i32 s9, 0 // (op 1201) 
s_mov_b32 s19, scc // (op 1201) 
s_cmp_eq_i32 s19, 0 // (op 1201) 
s_cselect_b32 s22, 0, -1 // (op 1201) 

s_and_b32 s19, s46, 31 // (op 1201) 
s_lshl_b32 s23, 1, s19 // (op 1201) 
s_add_i32 s24, s23, s22 // (op 1201) 


s_and_b32 s25, s1, s24 // (op 1201) 


s_add_i32 s1, s18, s25 // (op 1201) 


s_ashr_i32 s18, s1, s19 // (op 1201) 


s_lshl_b32 s1, s46, 24 // (op 1201) 
s_ashr_i32 s19, s1, 31 // (op 1201) 

s_xor_b32 s1, s18, s19 // (op 1201) 

s_sub_i32 s18, s1, s19 // (op 1201) 


s_add_u32 s50, s6, 64 // (op 1201) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1201) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1201) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1201) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1201) 

s_mul_i32 s1, s15, s24 // (op 1201) 

s_mul_i32 s19, s8, s1 // (op 1201) 

s_mul_i32 s1, s18, s19 // (op 1201) 

s_sub_u32 s19, s16, s1 // (op 1201) 

s_mov_b32 s1, s19 // (op 1201) 
s_mul_hi_i32 s22, s1, s14 // (op 1201) 
s_add_i32 s23, s22, s1 // (op 1201) 


s_ashr_i32 s1, s23, 31 // (op 1201) 
s_cmp_eq_i32 s14, 0 // (op 1201) 
s_mov_b32 s22, scc // (op 1201) 
s_cmp_eq_i32 s22, 0 // (op 1201) 
s_cselect_b32 s24, 0, -1 // (op 1201) 

s_and_b32 s22, s48, 31 // (op 1201) 
s_lshl_b32 s25, 1, s22 // (op 1201) 
s_add_i32 s49, s25, s24 // (op 1201) 


s_and_b32 s24, s1, s49 // (op 1201) 


s_add_i32 s1, s23, s24 // (op 1201) 


s_ashr_i32 s23, s1, s22 // (op 1201) 


s_lshl_b32 s1, s48, 24 // (op 1201) 
s_ashr_i32 s22, s1, 31 // (op 1201) 

s_xor_b32 s1, s23, s22 // (op 1201) 

s_sub_i32 s23, s1, s22 // (op 1201) 


s_mul_i32 s1, s47, s15 // (op 1201) 
s_add_u32 s24, s20, 16 // (op 1201) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1201) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1201) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1201) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1201) 

s_sub_i32 s22, s24, s1 // (op 1201) 

s_mul_i32 s24, s23, s22 // (op 1201) 


s_sub_u32 s25, s19, s24 // (op 1201) 


s_add_u32 s19, s25, s1 // (op 1201) 


s_cmp_eq_i32 s18, 0 // (op 1201) 
s_mov_b32 s1, scc // (op 1201) 

s_cmp_eq_i32 s1, 0 // (op 1201) 
s_cselect_b32 s18, s19, s0 // (op 1201) 



s_lshl_b32 s19, s18, 4 // (op 1201) 

s_mov_b32 s1, 0 // (op 1201) convert
s_mov_b32 s0, s19 // (op 1201) convert

s_ashr_i64 s[18:19], s[30:31], 8 // (op 1201) 
s_lshr_b32 s22, s18, 1 // (op 1201) 

v_lshl_add_u32 v15, s22, 9, v1 // (op 1201) 

v_mul_lo_u32 v29, v15, s27 // (op 1201) most significant: low of low * high
v_mul_hi_u32 v24, v15, s26 // (op 1201) most significant: high of low * low
v_mul_lo_u32 v28, v15, s26 // (op 1201) least significant: low of low * low
v_add_u32 v29, v29, v24 // (op 1201) most significant: sum


v_mov_b32 v15, s0 // (op 1201) 
v_mov_b32 v24, s1 // (op 1201) 
v_add_co_u32 v30, s[18:19], v15, v28 // (op 1201) least significant half
v_addc_co_u32 v31, s[18:19], v24, v29, s[18:19] // (op 1201) most significant half





s_mul_i32 s1, s26, s31 // (op 1203) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 1203) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 1203) least significant
s_mul_hi_u32 s19, s26, s30 // (op 1203) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1203) most significant: sum
s_add_u32 s1, s1, s19 // (op 1203) most significant: sum


s_mov_b32 s52, s28 // (op 1203) reg expression
s_mov_b32 s53, s29 // (op 1203) reg expression
s_mov_b32 s54, s0 // (op 1203) reg expression
s_mov_b32 s55, 131072 // (op 1203) call()

s_mul_hi_i32 s0, s16, s10 // (op 1269) 
v_lshrrev_b32 v15, 2, v1 // (op 1269) 
s_add_i32 s1, s0, s16 // (op 1269) 

v_lshlrev_b32 v24, 1, v15 // (op 1269) 

s_ashr_i32 s0, s1, 31 // (op 1269) 
v_and_b32 v15, -4, v1 // (op 1269) 
s_cmp_eq_i32 s10, 0 // (op 1269) 
s_mov_b32 s18, scc // (op 1269) 
v_sub_u32 v26, v1, v15 // (op 1269) 

s_cmp_eq_i32 s18, 0 // (op 1269) 
s_cselect_b32 s19, 0, -1 // (op 1269) 

v_lshlrev_b32 v15, 6, v26 // (op 1269) 

s_and_b32 s18, s11, 31 // (op 1269) 
s_lshl_b32 s22, 1, s18 // (op 1269) 
s_add_i32 s23, s22, s19 // (op 1269) 


s_and_b32 s19, s0, s23 // (op 1269) 


s_add_i32 s0, s1, s19 // (op 1269) 


s_ashr_i32 s1, s0, s18 // (op 1269) 


s_lshl_b32 s0, s11, 24 // (op 1269) 
s_ashr_i32 s18, s0, 31 // (op 1269) 

s_xor_b32 s0, s1, s18 // (op 1269) 

s_sub_i32 s1, s0, s18 // (op 1269) 


s_mul_hi_i32 s0, s1, s12 // (op 1269) 
s_add_i32 s18, s0, s1 // (op 1269) 

s_ashr_i32 s0, s18, 31 // (op 1269) 
s_cmp_eq_i32 s12, 0 // (op 1269) 
s_mov_b32 s19, scc // (op 1269) 
s_cmp_eq_i32 s19, 0 // (op 1269) 
s_cselect_b32 s22, 0, -1 // (op 1269) 

s_and_b32 s19, s13, 31 // (op 1269) 
s_lshl_b32 s23, 1, s19 // (op 1269) 
s_add_i32 s24, s23, s22 // (op 1269) 


s_and_b32 s25, s0, s24 // (op 1269) 


s_add_i32 s0, s18, s25 // (op 1269) 


s_ashr_i32 s18, s0, s19 // (op 1269) 


s_lshl_b32 s0, s13, 24 // (op 1269) 
s_ashr_i32 s19, s0, 31 // (op 1269) 

s_xor_b32 s0, s18, s19 // (op 1269) 

s_sub_i32 s18, s0, s19 // (op 1269) 


s_add_u32 s50, s6, 64 // (op 1269) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1269) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1269) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1269) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1269) 

s_mul_i32 s0, s18, s24 // (op 1269) 

s_sub_i32 s18, s1, s0 // (op 1269) 


s_mul_hi_i32 s19, s16, s9 // (op 1269) 
s_add_i32 s0, s19, s16 // (op 1269) 

s_ashr_i32 s1, s0, 31 // (op 1269) 
s_cmp_eq_i32 s9, 0 // (op 1269) 
s_mov_b32 s19, scc // (op 1269) 
s_cmp_eq_i32 s19, 0 // (op 1269) 
s_cselect_b32 s22, 0, -1 // (op 1269) 

s_and_b32 s19, s46, 31 // (op 1269) 
s_lshl_b32 s23, 1, s19 // (op 1269) 
s_add_i32 s49, s23, s22 // (op 1269) 


s_and_b32 s22, s1, s49 // (op 1269) 


s_add_i32 s1, s0, s22 // (op 1269) 


s_ashr_i32 s0, s1, s19 // (op 1269) 


s_lshl_b32 s1, s46, 24 // (op 1269) 
s_ashr_i32 s19, s1, 31 // (op 1269) 

s_xor_b32 s1, s0, s19 // (op 1269) 

s_sub_i32 s0, s1, s19 // (op 1269) 


s_mul_i32 s1, s15, s24 // (op 1269) 

s_mul_i32 s19, s8, s1 // (op 1269) 

s_mul_i32 s1, s0, s19 // (op 1269) 

s_sub_u32 s19, s16, s1 // (op 1269) 

s_mul_hi_i32 s1, s19, s14 // (op 1269) 
s_add_i32 s22, s1, s19 // (op 1269) 


s_ashr_i32 s1, s22, 31 // (op 1269) 
s_cmp_eq_i32 s14, 0 // (op 1269) 
s_mov_b32 s19, scc // (op 1269) 
s_cmp_eq_i32 s19, 0 // (op 1269) 
s_cselect_b32 s23, 0, -1 // (op 1269) 

s_and_b32 s19, s48, 31 // (op 1269) 
s_lshl_b32 s24, 1, s19 // (op 1269) 
s_add_i32 s25, s24, s23 // (op 1269) 


s_and_b32 s23, s1, s25 // (op 1269) 


s_add_i32 s1, s22, s23 // (op 1269) 


s_ashr_i32 s22, s1, s19 // (op 1269) 


s_lshl_b32 s1, s48, 24 // (op 1269) 
s_ashr_i32 s19, s1, 31 // (op 1269) 

s_xor_b32 s1, s22, s19 // (op 1269) 

s_sub_i32 s22, s1, s19 // (op 1269) 


s_cmp_eq_i32 s0, 0 // (op 1269) 
s_mov_b32 s1, scc // (op 1269) 

s_cmp_eq_i32 s1, 0 // (op 1269) 
s_cselect_b32 s0, s22, s18 // (op 1269) 



v_lshl_add_u32 v26, s0, 6, v24 // (op 1269) 


v_mul_lo_u32 v29, v26, s33 // (op 1269) most significant: low of low * high
v_mul_hi_u32 v24, v26, s32 // (op 1269) most significant: high of low * low
v_mul_lo_u32 v28, v26, s32 // (op 1269) least significant: low of low * low
v_add_u32 v29, v29, v24 // (op 1269) most significant: sum


s_ashr_i64 s[0:1], s[30:31], 8 // (op 1269) 
s_lshr_b32 s18, s0, 1 // (op 1269) 

v_lshl_add_u32 v24, s18, 9, v15 // (op 1269) 


v_mov_b32 v33, 0 // (op 1269) convert
v_mov_b32 v32, v24 // (op 1269) convert

v_add_co_u32 v34, s[0:1], v32, v28 // (op 1269) least significant half
v_addc_co_u32 v35, s[0:1], v33, v29, s[0:1] // (op 1269) most significant half



v_mul_lo_u32 v29, v35, 6 // (op 1269) most significant: low of high * low
v_mul_hi_u32 v15, v34, 6 // (op 1269) most significant: high of low * low
v_mul_lo_u32 v28, v34, 6 // (op 1269) least significant: low of low * low
v_add_u32 v29, v29, v15 // (op 1269) most significant: sum


v_lshrrev_b64 v[32:33], 3, v[28:29] // (op 1269) 

v_mov_b32 v15, v32 // (op 1269) convert

s_mul_i32 s1, s7, 6 // (op 1271) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 1271) least significant
s_mul_hi_u32 s18, s6, 6 // (op 1271) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1271) most significant: sum

s_mul_i32 s19, s32, s1 // (op 1271) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 1271) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 1271) least significant
s_mul_hi_u32 s23, s32, s0 // (op 1271) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 1271) most significant: sum
s_add_u32 s19, s19, s23 // (op 1271) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1271) 

s_mov_b32 s64, s34 // (op 1271) reg expression
s_mov_b32 s65, s35 // (op 1271) reg expression
s_mov_b32 s66, s0 // (op 1271) reg expression
s_mov_b32 s67, 131072 // (op 1271) call()

s_mul_hi_i32 s0, s16, s10 // (op 1333) 
v_and_b32 v24, -16, v1 // (op 1333) 
s_add_i32 s1, s0, s16 // (op 1333) 

v_sub_u32 v26, v1, v24 // (op 1333) 

s_ashr_i32 s0, s1, 31 // (op 1333) 
v_lshrrev_b32 v24, 4, v1 // (op 1333) 
s_cmp_eq_i32 s10, 0 // (op 1333) 
s_mov_b32 s18, scc // (op 1333) 
s_cmp_eq_i32 s18, 0 // (op 1333) 
s_cselect_b32 s19, 0, -1 // (op 1333) 

s_and_b32 s18, s11, 31 // (op 1333) 
s_lshl_b32 s22, 1, s18 // (op 1333) 
s_add_i32 s23, s22, s19 // (op 1333) 


s_and_b32 s19, s0, s23 // (op 1333) 


s_add_i32 s0, s1, s19 // (op 1333) 


s_ashr_i32 s1, s0, s18 // (op 1333) 


s_lshl_b32 s0, s11, 24 // (op 1333) 
s_ashr_i32 s18, s0, 31 // (op 1333) 

s_xor_b32 s0, s1, s18 // (op 1333) 

s_sub_i32 s1, s0, s18 // (op 1333) 


s_mul_hi_i32 s0, s1, s12 // (op 1333) 
s_add_i32 s18, s0, s1 // (op 1333) 

s_ashr_i32 s0, s18, 31 // (op 1333) 
s_cmp_eq_i32 s12, 0 // (op 1333) 
s_mov_b32 s19, scc // (op 1333) 
s_cmp_eq_i32 s19, 0 // (op 1333) 
s_cselect_b32 s22, 0, -1 // (op 1333) 

s_and_b32 s19, s13, 31 // (op 1333) 
s_lshl_b32 s23, 1, s19 // (op 1333) 
s_add_i32 s24, s23, s22 // (op 1333) 


s_and_b32 s25, s0, s24 // (op 1333) 


s_add_i32 s0, s18, s25 // (op 1333) 


s_ashr_i32 s18, s0, s19 // (op 1333) 


s_lshl_b32 s0, s13, 24 // (op 1333) 
s_ashr_i32 s19, s0, 31 // (op 1333) 

s_xor_b32 s0, s18, s19 // (op 1333) 

s_sub_i32 s18, s0, s19 // (op 1333) 


s_mul_i32 s0, s1, s15 // (op 1333) 

s_sub_u32 s1, s16, s0 // (op 1333) 

s_mul_i32 s0, s18, s15 // (op 1333) 
s_add_u32 s0, s0, s1 // (op 1333) 


s_mul_hi_i32 s1, s16, s9 // (op 1333) 
s_add_i32 s18, s1, s16 // (op 1333) 

s_ashr_i32 s1, s18, 31 // (op 1333) 
s_cmp_eq_i32 s9, 0 // (op 1333) 
s_mov_b32 s19, scc // (op 1333) 
s_cmp_eq_i32 s19, 0 // (op 1333) 
s_cselect_b32 s22, 0, -1 // (op 1333) 

s_and_b32 s19, s46, 31 // (op 1333) 
s_lshl_b32 s23, 1, s19 // (op 1333) 
s_add_i32 s24, s23, s22 // (op 1333) 


s_and_b32 s25, s1, s24 // (op 1333) 


s_add_i32 s1, s18, s25 // (op 1333) 


s_ashr_i32 s18, s1, s19 // (op 1333) 


s_lshl_b32 s1, s46, 24 // (op 1333) 
s_ashr_i32 s19, s1, 31 // (op 1333) 

s_xor_b32 s1, s18, s19 // (op 1333) 

s_sub_i32 s18, s1, s19 // (op 1333) 


s_add_u32 s50, s6, 64 // (op 1333) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1333) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1333) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1333) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1333) 

s_mul_i32 s1, s15, s24 // (op 1333) 

s_mul_i32 s19, s8, s1 // (op 1333) 

s_mul_i32 s1, s18, s19 // (op 1333) 

s_sub_u32 s19, s16, s1 // (op 1333) 

s_mov_b32 s1, s19 // (op 1333) 
s_mul_hi_i32 s22, s1, s14 // (op 1333) 
s_add_i32 s23, s22, s1 // (op 1333) 


s_ashr_i32 s1, s23, 31 // (op 1333) 
s_cmp_eq_i32 s14, 0 // (op 1333) 
s_mov_b32 s22, scc // (op 1333) 
s_cmp_eq_i32 s22, 0 // (op 1333) 
s_cselect_b32 s24, 0, -1 // (op 1333) 

s_and_b32 s22, s48, 31 // (op 1333) 
s_lshl_b32 s25, 1, s22 // (op 1333) 
s_add_i32 s49, s25, s24 // (op 1333) 


s_and_b32 s24, s1, s49 // (op 1333) 


s_add_i32 s1, s23, s24 // (op 1333) 


s_ashr_i32 s23, s1, s22 // (op 1333) 


s_lshl_b32 s1, s48, 24 // (op 1333) 
s_ashr_i32 s22, s1, 31 // (op 1333) 

s_xor_b32 s1, s23, s22 // (op 1333) 

s_sub_i32 s23, s1, s22 // (op 1333) 


s_mul_i32 s1, s47, s15 // (op 1333) 
s_add_u32 s24, s20, 16 // (op 1333) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1333) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1333) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1333) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 4 // (op 1333) 

s_sub_i32 s22, s24, s1 // (op 1333) 

s_mul_i32 s24, s23, s22 // (op 1333) 


s_sub_u32 s25, s19, s24 // (op 1333) 


s_add_u32 s19, s25, s1 // (op 1333) 


s_cmp_eq_i32 s18, 0 // (op 1333) 
s_mov_b32 s1, scc // (op 1333) 

s_cmp_eq_i32 s1, 0 // (op 1333) 
s_cselect_b32 s18, s19, s0 // (op 1333) 



v_lshl_add_u32 v17, s18, 4, v26 // (op 1333) 


s_ashr_i64 s[0:1], s[30:31], 8 // (op 1333) 
s_lshr_b32 s18, s0, 1 // (op 1333) 

v_lshl_add_u32 v26, s18, 4, v24 // (op 1333) 


v_mul_lo_u32 v24, v26, s36 // (op 1333) 

v_add_u32 v26, v17, v24 // (op 1333) 


s_mul_i32 s1, s36, s41 // (op 1335) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1335) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1335) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1335) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1335) most significant: sum
s_add_u32 s1, s1, s19 // (op 1335) most significant: sum


s_mov_b32 s68, s38 // (op 1335) reg expression
s_mov_b32 s69, s39 // (op 1335) reg expression
s_mov_b32 s70, s0 // (op 1335) reg expression
s_mov_b32 s71, 131072 // (op 1335) call()

s_mul_hi_i32 s0, s16, s10 // (op 1397) 
v_lshrrev_b32 v24, 1, v1 // (op 1397) 
s_add_i32 s1, s0, s16 // (op 1397) 

v_and_b32 v17, -2, v1 // (op 1397) 
s_ashr_i32 s0, s1, 31 // (op 1397) 
v_sub_u32 v18, v1, v17 // (op 1397) 

s_cmp_eq_i32 s10, 0 // (op 1397) 
s_mov_b32 s18, scc // (op 1397) 
v_lshlrev_b32 v17, 2, v18 // (op 1397) 

s_cmp_eq_i32 s18, 0 // (op 1397) 
s_cselect_b32 s19, 0, -1 // (op 1397) 

s_and_b32 s18, s11, 31 // (op 1397) 
s_lshl_b32 s22, 1, s18 // (op 1397) 
s_add_i32 s23, s22, s19 // (op 1397) 


s_and_b32 s19, s0, s23 // (op 1397) 


s_add_i32 s0, s1, s19 // (op 1397) 


s_ashr_i32 s1, s0, s18 // (op 1397) 


s_lshl_b32 s0, s11, 24 // (op 1397) 
s_ashr_i32 s18, s0, 31 // (op 1397) 

s_xor_b32 s0, s1, s18 // (op 1397) 

s_sub_i32 s1, s0, s18 // (op 1397) 


s_mul_hi_i32 s0, s1, s12 // (op 1397) 
s_add_i32 s18, s0, s1 // (op 1397) 

s_ashr_i32 s0, s18, 31 // (op 1397) 
s_cmp_eq_i32 s12, 0 // (op 1397) 
s_mov_b32 s19, scc // (op 1397) 
s_cmp_eq_i32 s19, 0 // (op 1397) 
s_cselect_b32 s22, 0, -1 // (op 1397) 

s_and_b32 s19, s13, 31 // (op 1397) 
s_lshl_b32 s23, 1, s19 // (op 1397) 
s_add_i32 s24, s23, s22 // (op 1397) 


s_and_b32 s25, s0, s24 // (op 1397) 


s_add_i32 s0, s18, s25 // (op 1397) 


s_ashr_i32 s18, s0, s19 // (op 1397) 


s_lshl_b32 s0, s13, 24 // (op 1397) 
s_ashr_i32 s19, s0, 31 // (op 1397) 

s_xor_b32 s0, s18, s19 // (op 1397) 

s_sub_i32 s18, s0, s19 // (op 1397) 


s_add_u32 s50, s6, 64 // (op 1397) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1397) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1397) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1397) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 1397) 

s_mul_i32 s0, s18, s24 // (op 1397) 

s_sub_i32 s18, s1, s0 // (op 1397) 


s_mul_hi_i32 s19, s16, s9 // (op 1397) 
s_add_i32 s0, s19, s16 // (op 1397) 

s_ashr_i32 s1, s0, 31 // (op 1397) 
s_cmp_eq_i32 s9, 0 // (op 1397) 
s_mov_b32 s19, scc // (op 1397) 
s_cmp_eq_i32 s19, 0 // (op 1397) 
s_cselect_b32 s22, 0, -1 // (op 1397) 

s_and_b32 s19, s46, 31 // (op 1397) 
s_lshl_b32 s23, 1, s19 // (op 1397) 
s_add_i32 s49, s23, s22 // (op 1397) 


s_and_b32 s22, s1, s49 // (op 1397) 


s_add_i32 s1, s0, s22 // (op 1397) 


s_ashr_i32 s0, s1, s19 // (op 1397) 


s_lshl_b32 s1, s46, 24 // (op 1397) 
s_ashr_i32 s19, s1, 31 // (op 1397) 

s_xor_b32 s1, s0, s19 // (op 1397) 

s_sub_i32 s0, s1, s19 // (op 1397) 


s_mul_i32 s1, s15, s24 // (op 1397) 

s_mul_i32 s19, s8, s1 // (op 1397) 

s_mul_i32 s1, s0, s19 // (op 1397) 

s_sub_u32 s19, s16, s1 // (op 1397) 

s_mul_hi_i32 s1, s19, s14 // (op 1397) 
s_add_i32 s22, s1, s19 // (op 1397) 


s_ashr_i32 s1, s22, 31 // (op 1397) 
s_cmp_eq_i32 s14, 0 // (op 1397) 
s_mov_b32 s19, scc // (op 1397) 
s_cmp_eq_i32 s19, 0 // (op 1397) 
s_cselect_b32 s23, 0, -1 // (op 1397) 

s_and_b32 s19, s48, 31 // (op 1397) 
s_lshl_b32 s24, 1, s19 // (op 1397) 
s_add_i32 s25, s24, s23 // (op 1397) 


s_and_b32 s23, s1, s25 // (op 1397) 


s_add_i32 s1, s22, s23 // (op 1397) 


s_ashr_i32 s22, s1, s19 // (op 1397) 


s_lshl_b32 s1, s48, 24 // (op 1397) 
s_ashr_i32 s19, s1, 31 // (op 1397) 

s_xor_b32 s1, s22, s19 // (op 1397) 

s_sub_i32 s22, s1, s19 // (op 1397) 


s_cmp_eq_i32 s0, 0 // (op 1397) 
s_mov_b32 s1, scc // (op 1397) 

s_cmp_eq_i32 s1, 0 // (op 1397) 
s_cselect_b32 s0, s22, s18 // (op 1397) 



v_lshl_add_u32 v18, s0, 6, v24 // (op 1397) 


v_mul_lo_u32 v24, v18, s42 // (op 1397) 

s_ashr_i64 s[0:1], s[30:31], 8 // (op 1397) 
s_lshr_b32 s18, s0, 1 // (op 1397) 

v_lshl_add_u32 v18, s18, 4, v17 // (op 1397) 


v_add_u32 v17, v18, v24 // (op 1397) 


s_mul_i32 s1, s42, s7 // (op 1399) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 1399) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 1399) least significant
s_mul_hi_u32 s19, s42, s6 // (op 1399) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1399) most significant: sum
s_add_u32 s1, s1, s19 // (op 1399) most significant: sum


s_mov_b32 s72, s44 // (op 1399) reg expression
s_mov_b32 s73, s45 // (op 1399) reg expression
s_mov_b32 s74, s0 // (op 1399) reg expression
s_mov_b32 s75, 131072 // (op 1399) call()

v_lshlrev_b32 v24, 4, v1 // (op 1490) 
s_nop 0
v_readfirstlane_b32 s0, v24 // (op 1490) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -4, v1 // (op 1542) 
v_sub_u32 v18, v1, v24 // (op 1542) 

v_lshlrev_b32 v24, 6, v18 // (op 1542) 

v_lshrrev_b32 v18, 2, v1 // (op 1542) 
v_lshl_add_u32 v19, v18, 9, v24 // (op 1542) 


v_mul_lo_u32 v18, v19, 6 // (op 1542) 

v_lshrrev_b32 v24, 3, v18 // (op 1542) 

v_and_b32 v18, -16, v1 // (op 1600) 
v_sub_u32 v19, v1, v18 // (op 1600) 

v_and_b32 v18, 4294967280, v1 // (op 1600) 
v_add_u32 v20, v18, v19 // (op 1600) 


v_and_b32 v18, -2, v1 // (op 1653) 
v_sub_u32 v19, v1, v18 // (op 1653) 

v_lshlrev_b32 v18, 2, v19 // (op 1653) 

v_lshrrev_b32 v19, 1, v1 // (op 1653) 
v_lshl_add_u32 v21, v19, 3, v18 // (op 1653) 


s_ashr_i64 s[18:19], s[30:31], 8 // (op 405) 
s_and_b32 s1, s18, 4294967294 // (op 405) 

s_ashr_i64 s[18:19], s[30:31], 8 // (op 404) 
s_mov_b32 s22, s18 // (op 404) convert

s_cmp_lt_u32 s1, s22 // (op 404) 










s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_404 // (op 404) Condition: Top (jump to RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_404 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_404:
s_barrier  // (op 1882) 
s_add_u32 s1, s1, 1 // (op 406) 

s_lshl_b64 s[18:19], s[26:27], 7 // (op 1172) 
v_mov_b64 v[18:19], s[18:19] // (op 1172) convert


v_mov_b64 v[22:23], v[30:31] // (op 1172) 
v_mov_b64 v[28:29], v[22:23] // (op 1172) reg expression
s_mov_b32 m0, s0 // (op 1172) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1172) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v18 // (op 1172) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 1172) most significant half

s_add_u32 m0, m0, 2048 // (op 1172) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 1172) Load value direct to lds



s_lshl_b64 s[18:19], s[26:27], 8 // (op 1191) 
v_mov_b32 v14, s18 // (op 1191) 
v_mov_b32 v16, s19 // (op 1191) 
v_add_co_u32 v30, s[50:51], v30, v14 // (op 1191) least significant half
v_addc_co_u32 v31, s[50:51], v31, v16, s[50:51] // (op 1191) most significant half





s_mul_i32 s19, s33, 6 // (op 221) most significant: low of high * low
s_mul_i32 s18, s32, 6 // (op 221) least significant
s_mul_hi_u32 s22, s32, 6 // (op 221) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 221) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[18:19] // (op 221) 


v_mov_b32 v14, v15 // (op 221) 
v_mov_b32 v16, v14 // (op 221) 
buffer_load_dwordx4 v[32:35], v16, s[64:67], 0 offen // (op 221) Load value
buffer_load_dwordx2 v[36:37], v16, s[64:67], 0 offen offset:16 // (op 221) Load value
buffer_load_dwordx4 v[38:41], v16, s[64:67], 0 offen offset:24 // (op 221) Load value
buffer_load_dwordx2 v[42:43], v16, s[64:67], 0 offen offset:40 // (op 221) Load value
v_add_u32 v16, v16, v18 // (op 221) 
buffer_load_dwordx4 v[44:47], v16, s[64:67], 0 offen // (op 221) Load value
buffer_load_dwordx2 v[48:49], v16, s[64:67], 0 offen offset:16 // (op 221) Load value
buffer_load_dwordx4 v[50:53], v16, s[64:67], 0 offen offset:24 // (op 221) Load value
buffer_load_dwordx2 v[54:55], v16, s[64:67], 0 offen offset:40 // (op 221) Load value



v_add_u32 v15, 192, v15 // (op 1261) 


s_barrier  // (op 1901) 
v_mov_b32 v14, v26 // (op 222) 



v_mov_b32 v16, v14 // (op 222) reg expression

buffer_load_ubyte v25, v16, s[68:71], 0 offen offset: 0 // (op 222) Load value


s_lshl_b32 s18, s36, 3 // (op 1325) 
v_add_u32 v26, v26, s18 // (op 1325) 

s_waitcnt vmcnt(8)
ds_write_b128 v24, v[32:35] offset:8192 // (op 218) Store local data 


v_mov_b32 v14, v17 // (op 223) 

buffer_load_dword v16, v14, s[72:75], 0 offen // (op 223) Load value

v_add_u32 v17, 8, v17 // (op 1389) 
s_waitcnt vmcnt(8)
ds_write_b64 v24, v[36:37] offset:8208 // (op 218) Store local data 
s_barrier  // (op 1891) 
s_waitcnt vmcnt(7)
ds_write_b128 v24, v[38:41] offset:8216 // (op 218) Store local data 



ds_read_b64_tr_b8 v[56:57], v0 // (op 201) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:128 // (op 201) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:1024 // (op 201) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:1152 // (op 201) Transpose load local data 
s_waitcnt vmcnt(6)
ds_write_b64 v24, v[42:43] offset:8232 // (op 218) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v24, v[44:47] offset:8384 // (op 218) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v24, v[48:49] offset:8400 // (op 218) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v24, v[50:53] offset:8408 // (op 218) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v24, v[54:55] offset:8424 // (op 218) Store local data 

s_barrier  // (op 1922) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1912) 






ds_read_b128 v[32:35], v9 offset:8192 // (op 203) Load local data 
ds_read_b64 v[36:37], v9 offset:8208 // (op 203) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v20, v25 offset:20992 // (op 219) Store local data 

s_barrier  // (op 1942) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1932) 





ds_read_u8 v14, v4 offset:20992 // (op 205) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v21, v16 offset:20480 // (op 220) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1952) 



ds_read_u8 v16, v5 offset:20480 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[32:37], a[0:3], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 208) 





ds_read_b64_tr_b8 v[32:33], v0 offset:2048 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:2176 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:3072 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:3200 // (op 210) Transpose load local data 



ds_read_b128 v[40:43], v9 offset:8288 // (op 212) Load local data 
ds_read_b64 v[44:45], v9 offset:8304 // (op 212) Load local data 


ds_read_u8 v16, v4 offset:21056 // (op 214) Load local data 



ds_read_u8 v25, v5 offset:20484 // (op 216) Load local data 



ds_read_b128 v[46:49], v9 offset:14336 // (op 412) Load local data 
ds_read_b64 v[50:51], v9 offset:14352 // (op 412) Load local data 



ds_read_u8 v18, v5 offset:20736 // (op 416) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:63], v[46:51], a[4:7], v14, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 417) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[40:45], a[0:3], v16, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 217) 
ds_read_b128 v[46:49], v9 offset:14432 // (op 421) Load local data 


ds_read_b64 v[50:51], v9 offset:14448 // (op 421) Load local data 



ds_read_u8 v14, v5 offset:20740 // (op 425) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:39], v[46:51], a[4:7], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 426) 




s_ashr_i64 s[18:19], s[30:31], 8 // (op 404) 
s_mov_b32 s22, s18 // (op 404) convert

s_cmp_lt_u32 s1, s22 // (op 404) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_404 // (op 404) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_404 if true)
RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_404:























s_barrier  // (op 2041) 





















v_cmp_eq_f32 vcc, 0.00000, v7 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_155
v_accvgpr_read v12, a0 // (op 50) 
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_mul_f32 v16, v10, v12 // (op 50) 
v_mul_f32 v17, v10, v13 // (op 50) 
v_mul_f32 v18, v10, v14 // (op 50) 
v_mul_f32 v19, v10, v15 // (op 50) 



v_mov_b32 v0, v11 // (op 56) reg expression

buffer_store_dwordx4 v[16:19], v0, s[60:63], 0 offen // (op 56) Store value


v_accvgpr_read v12, a4 // (op 710) 
v_accvgpr_read v13, a5 // (op 710) 
v_accvgpr_read v14, a6 // (op 710) 
v_accvgpr_read v15, a7 // (op 710) 
v_mul_f32 v16, v10, v12 // (op 710) 
v_mul_f32 v17, v10, v13 // (op 710) 
v_mul_f32 v18, v10, v14 // (op 710) 
v_mul_f32 v19, v10, v15 // (op 710) 



s_lshl_b32 s0, s4, 7 // (op 712) 
v_add_u32 v0, v11, s0 // (op 712) 


buffer_store_dwordx4 v[16:19], v0, s[60:63], 0 offen // (op 712) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_155
RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_155:


v_mov_b32 v0, v8 // (op 167) reg expression

buffer_load_dwordx4 v[12:15], v0, s[56:59], 0 offen // (op 167) Load value

v_accvgpr_read v16, a0 // (op 164) 
v_accvgpr_read v17, a1 // (op 164) 
v_accvgpr_read v18, a2 // (op 164) 
v_accvgpr_read v19, a3 // (op 164) 
v_mul_f32 v20, v10, v16 // (op 164) 
v_mul_f32 v21, v10, v17 // (op 164) 
v_mul_f32 v22, v10, v18 // (op 164) 
v_mul_f32 v23, v10, v19 // (op 164) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v7, v12 // (op 168) 
v_mul_f32 v17, v7, v13 // (op 168) 
v_mul_f32 v18, v7, v14 // (op 168) 
v_mul_f32 v19, v7, v15 // (op 168) 

v_add_f32 v12, v16, v20 // (op 165) 
v_add_f32 v13, v17, v21 // (op 165) 
v_add_f32 v14, v18, v22 // (op 165) 
v_add_f32 v15, v19, v23 // (op 165) 




v_mov_b32 v0, v11 // (op 166) reg expression

buffer_store_dwordx4 v[12:15], v0, s[60:63], 0 offen // (op 166) Store value




s_lshl_b32 s0, s2, 7 // (op 734) 
v_add_u32 v0, v8, s0 // (op 734) 


buffer_load_dwordx4 v[12:15], v0, s[56:59], 0 offen // (op 734) Load value

v_accvgpr_read v16, a4 // (op 731) 
v_accvgpr_read v17, a5 // (op 731) 
v_accvgpr_read v18, a6 // (op 731) 
v_accvgpr_read v19, a7 // (op 731) 
v_mul_f32 v20, v10, v16 // (op 731) 
v_mul_f32 v21, v10, v17 // (op 731) 
v_mul_f32 v22, v10, v18 // (op 731) 
v_mul_f32 v23, v10, v19 // (op 731) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v7, v12 // (op 735) 
v_mul_f32 v17, v7, v13 // (op 735) 
v_mul_f32 v18, v7, v14 // (op 735) 
v_mul_f32 v19, v7, v15 // (op 735) 

v_add_f32 v12, v16, v20 // (op 732) 
v_add_f32 v13, v17, v21 // (op 732) 
v_add_f32 v14, v18, v22 // (op 732) 
v_add_f32 v15, v19, v23 // (op 732) 




s_lshl_b32 s0, s4, 7 // (op 733) 
v_add_u32 v0, v11, s0 // (op 733) 


buffer_store_dwordx4 v[12:15], v0, s[60:63], 0 offen // (op 733) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_155:










s_endpgm  // End of RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.LRR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__end:
.size RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_, .LRR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__end-RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
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
    DestDataType: s
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
  - .name: RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
    .symbol: RR_GEMM_NN_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_.kd
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

